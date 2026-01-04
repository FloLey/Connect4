from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, update
from typing import List
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timedelta, timezone

from backend.app.core.database import get_db, get_session_maker
from backend.app.models.game_model import Game
from backend.app.schemas.game_schema import GameCreate, GameResponse
from backend.app.api.websocket_manager import manager
from backend.app.api.stats import router as stats_router
from backend.app.api.admin import router as admin_router
from backend.app.api.tournament import router as tournament_router
from backend.app.core.model_registry import registry
from backend.app.core.events import game_events
from backend.app.engine.elo import update_elo_on_complete
from backend.app.models.enums import GameStatus, PlayerType
from backend.app.services.game_runner import game_runner
from backend.app.services.game_service import GameState, game_service
from backend.app.services.tournament_service import tournament_service

# --- LIFESPAN MANAGER (Auto-Resume on Startup) ---
async def tournament_watcher(env: str):
    """Background task to feed the tournament for specific environment"""
    print(f"👀 Starting Watcher for [{env}]")
    SessionLocal = get_session_maker(env)
    from backend.app.services.tournament_bus import tournament_bus
    
    while True:
        try:
            # Wait for a signal OR a 30s heartbeat safety fallback
            signal_received = await tournament_bus.wait_for_signal(timeout=30.0)
            
            async with SessionLocal() as db:
                # Pass 'env' so service knows which runner to trigger
                await tournament_service.tick(db, env=env)
        except Exception as e:
            print(f"Error in {env} watcher: {e}")

async def resume_games(env: str):
    """Resume interrupted AI vs AI games for a specific environment"""
    SessionLocal = get_session_maker(env)
    async with SessionLocal() as db:
        query = select(Game).where(
            Game.status == GameStatus.IN_PROGRESS,
            Game.player_1_type != PlayerType.HUMAN,
            Game.player_2_type != PlayerType.HUMAN
        )
        result = await db.execute(query)
        games = result.scalars().all()
        
        for game in games:
            print(f"▶️ Resuming Game {game.id} in [{env}]")
            await game_runner.start_game_if_ai_vs_ai(game.id, env)

async def run_cleanup_periodically():
    """Periodic background task to clean up abandoned games"""
    while True:
        try: # <--- Wrap logic in a try block INSIDE the while loop
            await asyncio.sleep(900)  # Run every 15 minutes
            for env in ["prod", "test"]:
                SessionLocal = get_session_maker(env)
                async with SessionLocal() as db:
                    # Conservative cleanup rules:
                    # 1. NEVER clean tournament games (tournament_id IS NOT NULL) - let tournament system handle them
                    # 2. Only clean regular games (tournament_id IS NULL) after 6 hours
                    # 3. Never clean PAUSED games (they're intentionally waiting)
                    
                    cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
                    
                    # Only clean old regular games (non-tournament)
                    result = await db.execute(
                        update(Game).where(
                            Game.status == GameStatus.IN_PROGRESS,
                            Game.tournament_id.is_(None),  # Only non-tournament games
                            Game.created_at < cutoff
                        ).values(status="ABANDONED")
                    )
                    cleaned_count = result.rowcount
                    
                    await db.commit()
                    
                    if cleaned_count > 0:
                        print(f"🧹 Cleanup: Marked {cleaned_count} regular (non-tournament) games as ABANDONED")
                    
        except Exception as e:
            print(f"Janitor Task Error: {e}")
            await asyncio.sleep(60) # Wait a minute before retrying

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wire the event listener
    game_events.subscribe_complete(update_elo_on_complete)
    
    # Start BOTH environments
    for env in ["prod", "test"]:
        # 1. Resume interrupted games
        await resume_games(env)
        # 2. Start specific watcher
        asyncio.create_task(tournament_watcher(env))
    
    # Start periodic cleanup task
    asyncio.create_task(run_cleanup_periodically())
    
    yield
    # Shutdown logic (optional)
# -------------------------------------------------

app = FastAPI(title="Connect Four LLM Arena", lifespan=lifespan)

# --- ADD CORS MIDDLEWARE HERE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], # Allow Vite (5173) and React default (3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------

# Register routers
app.include_router(stats_router, prefix="/stats", tags=["Stats"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(tournament_router, prefix="/tournament", tags=["Tournament"])

# --- NEW ENDPOINT ---
@app.get("/models")
async def get_available_models():
    """Returns the list of supported LLMs and their display labels."""
    # Transform dict to list of objects for easier frontend consumption
    return [
        {"id": key, "provider": val.provider, "label": val.label}
        for key, val in registry.list_all().items()
    ]
# --------------------

@app.post("/games", response_model=GameResponse)
async def create_game(game_data: GameCreate, request: Request, db: AsyncSession = Depends(get_db)):
    import uuid
    
    # Generate session tokens for human players
    player_1_token = str(uuid.uuid4()) if game_data.player_1 == PlayerType.HUMAN else None
    player_2_token = str(uuid.uuid4()) if game_data.player_2 == PlayerType.HUMAN else None
    
    new_game = Game(
        player_1_type=game_data.player_1,
        player_2_type=game_data.player_2,
        player_1_token=player_1_token,
        player_2_token=player_2_token,
        history=[]
    )
    db.add(new_game)
    await db.commit()
    await db.refresh(new_game)
    
    # --- TRIGGER BACKGROUND RUNNER ---
    # If it's AI vs AI, start the background loop immediately.
    # The user doesn't even need to open the websocket.
    env = request.headers.get("x-db-env", "prod")
    if env not in ["prod", "test"]:
        env = "prod"
    await game_runner.start_game_if_ai_vs_ai(new_game.id, env)
    # ---------------------------------
    
    return new_game

# --- FIX: Move this endpoint UP, before /games/{game_id} ---
@app.get("/games/history", response_model=List[GameResponse])
async def get_game_history(
    skip: int = 0, 
    limit: int = 50, 
    db: AsyncSession = Depends(get_db)
):
    """
    Fetch completed games (COMPLETED or DRAW) for the history tab.
    Sorted by newest first.
    """
    query = select(Game).where(
        Game.status.in_([GameStatus.COMPLETED, GameStatus.DRAW])
    ).order_by(Game.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    games = result.scalars().all()
    return games
# -----------------------------------------------------------

@app.get("/games/pending-human", response_model=List[int])
async def get_pending_human_games(db: AsyncSession = Depends(get_db)):
    """
    Returns a list of Game IDs where it is currently a human's turn.
    Used for the 'Next Actionable Game' navigation feature.
    """
    # Fetch all IN_PROGRESS games involving at least one human
    query = select(Game).where(
        Game.status == GameStatus.IN_PROGRESS,
        or_(Game.player_1_type == PlayerType.HUMAN, Game.player_2_type == PlayerType.HUMAN)
    ).order_by(Game.id)
    
    result = await db.execute(query)
    games = result.scalars().all()
    
    actionable_ids = []
    
    for game in games:
        # --- FIX: Handle None history safely ---
        history = game.history if game.history is not None else []
        move_count = len(history)
        
        is_p1_turn = (move_count % 2) == 0
        
        is_p1_human = game.player_1_type == PlayerType.HUMAN
        is_p2_human = game.player_2_type == PlayerType.HUMAN
        
        # Check if it's a human's turn
        if (is_p1_turn and is_p1_human) or (not is_p1_turn and is_p2_human):
            actionable_ids.append(game.id)
            
    return actionable_ids

@app.get("/games/{game_id}", response_model=GameResponse)
async def get_game(game_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Game).where(Game.id == game_id))
    game = result.scalar_one_or_none()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game

@app.get("/games", response_model=List[GameResponse])
async def list_games(status: str = GameStatus.IN_PROGRESS, limit: int = 20, db: AsyncSession = Depends(get_db)):
    """List games by status. Useful for finding active AI vs AI games to spectate."""
    query = select(Game).where(Game.status == status).order_by(Game.created_at.desc()).limit(limit)
    result = await db.execute(query)
    games = result.scalars().all()
    return games

@app.post("/games/{game_id}/recover")
async def recover_stuck_game(game_id: int, db: AsyncSession = Depends(get_db)):
    """Manually trigger recovery for a stuck Human vs AI game"""
    try:
        game_db, engine = await game_service.get_game_state(db, game_id)
        current_state = GameState(game_db, engine)
        
        # Only recover Human vs AI games that are in progress
        is_human_vs_ai = (
            (game_db.player_1_type == PlayerType.HUMAN and game_db.player_2_type != PlayerType.HUMAN) or
            (game_db.player_1_type != PlayerType.HUMAN and game_db.player_2_type == PlayerType.HUMAN)
        )
        
        if game_db.status == GameStatus.IN_PROGRESS and is_human_vs_ai and not current_state.winner and not current_state.is_draw:
            # Determine if it's AI's turn
            current_ai_model = (current_state.player_1_type if current_state.current_turn == 1 
                              else current_state.player_2_type)
            
            if current_ai_model != PlayerType.HUMAN:
                # Trigger AI move
                new_state = await game_service.step_ai_turn(db, game_id)
                if new_state:
                    await manager.broadcast(game_id, manager._build_state_message(new_state))
                    return {"message": "Game recovered successfully", "ai_moved": True}
                else:
                    return {"message": "Failed to execute AI move", "ai_moved": False}
            else:
                return {"message": "It's human's turn, no recovery needed", "ai_moved": False}
        else:
            return {"message": "Game doesn't need recovery", "ai_moved": False}
            
    except ValueError:
        raise HTTPException(status_code=404, detail="Game not found")

@app.websocket("/games/{game_id}/ws")
async def game_websocket(websocket: WebSocket, game_id: int):
    await manager.handle_game_session(websocket, game_id)