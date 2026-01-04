from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, or_
from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict
from datetime import datetime

from backend.app.core.database import get_db
from backend.app.models.elo_model import EloRating, EloHistory
from backend.app.models.game_model import Game
from backend.app.models.enums import GameStatus
from backend.app.core.model_registry import registry
from backend.app.engine.game import ConnectFour

router = APIRouter()

# --- Schemas ---
class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    rating: float
    matches_played: int
    wins: int
    losses: int
    draws: int
    
    # --- Expanded Stats ---
    mean_time_per_move: float
    avg_moves_per_game: float
    mean_tokens_out_per_move: float
    total_tokens_out: int
    
    # --- Cost Stats ---
    avg_cost_per_move: float
    avg_cost_per_game: float
    total_cost: float

class HistoryPoint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    rating: float
    timestamp: datetime

class LiveGameSummary(BaseModel):
    id: int
    player_1: str
    player_2: str
    move_count: int
    created_at: datetime
    board: List[List[int]] # Added board state (6 rows x 7 cols)

# --- NEW: Matrix Schemas ---
class MatrixCell(BaseModel):
    wins: int
    losses: int
    draws: int
    total: int
    win_rate: float # 0.0 to 100.0

class MatrixResponse(BaseModel):
    models: List[str] # Ordered list of model names (rows/cols)
    grid: Dict[str, Dict[str, MatrixCell]] # grid[row_model][col_model] -> Stats

# --- Endpoints ---

@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(db: AsyncSession = Depends(get_db)):
    """Returns models sorted by ELO rating with detailed stats."""
    result = await db.execute(select(EloRating).order_by(desc(EloRating.rating)))
    ratings = result.scalars().all()
    
    output = []
    for r in ratings:
        # 1. Base Stats
        total_moves = r.total_moves or 0
        matches = r.matches_played or 0
        
        # 2. Averages
        mean_time = (r.total_duration_seconds / total_moves) if total_moves > 0 else 0.0
        mean_tokens_out = (r.total_output_tokens / total_moves) if total_moves > 0 else 0.0
        avg_moves_game = (total_moves / matches) if matches > 0 else 0.0
        
        # 3. Cost Calculations (USD)
        # Pricing is per 1 Million tokens
        config = registry.get(r.model_name)
        pricing = config.pricing if config else {"input": 0, "output": 0}
        
        cost_input = (r.total_input_tokens or 0) / 1_000_000 * pricing.get("input", 0)
        cost_output = (r.total_output_tokens or 0) / 1_000_000 * pricing.get("output", 0)
        total_cost_usd = cost_input + cost_output
        
        avg_cost_game = (total_cost_usd / matches) if matches > 0 else 0.0
        avg_cost_move = (total_cost_usd / total_moves) if total_moves > 0 else 0.0
        
        output.append(LeaderboardEntry(
            model_name=r.model_name,
            rating=r.rating,
            matches_played=matches,
            wins=r.wins,
            losses=r.losses,
            draws=r.draws,
            
            # Time & Volume
            mean_time_per_move=round(mean_time, 2),
            avg_moves_per_game=round(avg_moves_game, 1),
            mean_tokens_out_per_move=round(mean_tokens_out, 1),
            total_tokens_out=r.total_output_tokens or 0,
            
            # Economics
            avg_cost_per_move=round(avg_cost_move, 5),
            avg_cost_per_game=round(avg_cost_game, 4),
            total_cost=round(total_cost_usd, 4)
        ))
        
    return output

@router.get("/history", response_model=List[HistoryPoint])
async def get_rating_history(model: str = None, db: AsyncSession = Depends(get_db)):
    """Returns time-series data. Optional filter by specific model."""
    query = select(EloHistory).order_by(EloHistory.timestamp)
    if model:
        query = query.where(EloHistory.model_name == model)
    
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/history-plot")
async def get_history_plot_data(db: AsyncSession = Depends(get_db)):
    """
    Returns data formatted for Recharts where X-axis = Personal Match Count.
    Example: 
      [
        { "match_number": 0, "gpt-4": 1200, "claude": 1200 },
        { "match_number": 1, "gpt-4": 1210 },  <-- gpt-4's 1st game
        { "match_number": 1, "claude": 1190 }  <-- claude's 1st game (happened much later in real time)
      ]
    """
    # 1. Fetch ALL history ordered by timestamp (ASC)
    result = await db.execute(select(EloHistory).order_by(EloHistory.timestamp.asc()))
    records = result.scalars().all()

    # 2. Identify all models present in history
    all_models = set(r.model_name for r in records)

    # 3. Initialize separate counters for each model
    # This tracks "How many games has THIS model played so far?"
    model_counters = {m: 0 for m in all_models}

    # 4. Initialize Data Structure
    # data_map key = The personal match number (0, 1, 2...)
    data_map = {}

    # Initialize Baseline (Game 0)
    data_map[0] = { "match_number": 0 }
    for m in all_models:
        data_map[0][m] = 1200  # Everyone starts at 1200

    # 5. Aggregate Data
    for record in records:
        model = record.model_name
        
        # Increment ONLY this model's counter
        model_counters[model] += 1
        current_match_num = model_counters[model]

        # Ensure the row exists in our result set
        if current_match_num not in data_map:
            data_map[current_match_num] = { "match_number": current_match_num }
        
        # Add the rating
        data_map[current_match_num][model] = round(record.rating)

    # 6. Sort by match number (0, 1, 2, 3...)
    plot_data = sorted(data_map.values(), key=lambda x: x["match_number"])
    
    # Return ALL data (removed the [-500:] slice)
    return plot_data

@router.get("/active-games", response_model=List[LiveGameSummary])
async def get_active_games(db: AsyncSession = Depends(get_db)):
    """Returns games currently IN_PROGRESS with reconstructed board state."""
    query = select(Game).where(Game.status == GameStatus.IN_PROGRESS).order_by(desc(Game.created_at)).limit(20)
    result = await db.execute(query)
    games = result.scalars().all()
    
    summaries = []
    for g in games:
        # Use ConnectFour engine to reconstruct board state
        if g.history:
            game_engine = ConnectFour.from_history(g.history)
            board = game_engine.board
        else:
            board = [[0 for _ in range(7)] for _ in range(6)]

        summaries.append(LiveGameSummary(
            id=g.id,
            player_1=g.player_1_type,
            player_2=g.player_2_type,
            move_count=len(g.history) if g.history else 0,
            created_at=g.created_at,
            board=board
        ))
    
    return summaries

@router.get("/matrix", response_model=MatrixResponse)
async def get_win_rate_matrix(db: AsyncSession = Depends(get_db)):
    """
    Calculates the N x N win rate matrix for all models.
    Aggregates data server-side to prevent frontend heaviness.
    """
    # 1. Get all models sorted by Rating (High to Low)
    # We want the matrix rows/cols to be ordered by skill
    models_result = await db.execute(select(EloRating.model_name).order_by(desc(EloRating.rating)))
    models = models_result.scalars().all()
    
    # 2. Get all completed games
    games_result = await db.execute(
        select(Game.player_1_type, Game.player_2_type, Game.winner)
        .where(Game.status.in_([GameStatus.COMPLETED, GameStatus.DRAW]))
    )
    games = games_result.all()
    
    # 3. Initialize Data Structure
    # stats[model_A][model_B] = {wins, losses, draws}
    stats = {m: {opp: {"w": 0, "l": 0, "d": 0} for opp in models} for m in models}
    
    # 4. Aggregate
    for p1, p2, winner in games:
        # Skip if models were deleted or unknown
        if p1 not in stats or p2 not in stats:
            continue
            
        if winner == 1:
            stats[p1][p2]["w"] += 1
            stats[p2][p1]["l"] += 1
        elif winner == 2:
            stats[p1][p2]["l"] += 1
            stats[p2][p1]["w"] += 1
        else: # Draw
            stats[p1][p2]["d"] += 1
            stats[p2][p1]["d"] += 1
            
    # 5. Format Response
    grid_output = {}
    
    for model_row in models:
        row_data = {}
        for model_col in models:
            if model_row == model_col:
                # Identity cell (vs self) - usually null or specific marker
                row_data[model_col] = MatrixCell(wins=0, losses=0, draws=0, total=0, win_rate=0.0)
                continue
                
            s = stats[model_row][model_col]
            total = s["w"] + s["l"] + s["d"]
            
            # Formula: (Wins + 0.5 * Draws) / Total
            if total > 0:
                score = s["w"] + (0.5 * s["d"])
                win_rate = (score / total) * 100.0
            else:
                win_rate = 0.0
                
            row_data[model_col] = MatrixCell(
                wins=s["w"],
                losses=s["l"],
                draws=s["d"],
                total=total,
                win_rate=round(win_rate, 1)
            )
        grid_output[model_row] = row_data
        
    return MatrixResponse(models=models, grid=grid_output)