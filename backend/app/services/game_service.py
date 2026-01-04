"""
Game Service - Centralized Game Logic

This service is the single source of truth for all game state modifications.
It handles:
- Game creation
- Move processing (human and AI)
- Database updates 
- ELO calculations
- Game completion logic

Used by both WebSocket manager (live games) and tournament scripts (background games).
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.attributes import flag_modified

from backend.app.models.game_model import Game
from backend.app.models.enums import GameStatus, PlayerType
from backend.app.engine.game import ConnectFour
from backend.app.engine.ai import ConnectFourAI
from backend.app.core.events import game_events
from backend.app.core.model_registry import registry
from backend.app.services.tournament_bus import tournament_bus


class GameState:
    """Represents the current state of a game for API responses"""
    def __init__(self, game_db: Game, engine: ConnectFour):
        self.game_id = game_db.id
        self.board = engine.board
        self.current_turn = engine.current_turn
        self.winner = engine.winner
        self.status = game_db.status
        self.is_draw = engine.is_draw()
        self.last_move = game_db.history[-1] if game_db.history else None
        self.player_1_type = game_db.player_1_type
        self.player_2_type = game_db.player_2_type


class GameService:
    """Centralized service for all game operations"""
    
    async def create_game(self, db: AsyncSession, player_1_type: str, player_2_type: str) -> Game:
        """Create a new game in the database"""
        new_game = Game(
            player_1_type=player_1_type,
            player_2_type=player_2_type,
            history=[]
        )
        db.add(new_game)
        await db.commit()
        await db.refresh(new_game)
        return new_game
    
    async def get_game_state(self, db: AsyncSession, game_id: int) -> Tuple[Game, ConnectFour]:
        """Load game from DB and reconstruct engine state (READ ONLY)"""
        result = await db.execute(select(Game).where(Game.id == game_id))
        game_db = result.scalar_one_or_none()
        
        if not game_db:
            raise ValueError(f"Game {game_id} not found")
        
        # Reconstruct game engine from history
        engine = ConnectFour()
        for move in game_db.history:
            engine.drop_piece(move['column'])
            
        return game_db, engine
    
    async def _get_game_for_update(self, db: AsyncSession, game_id: int) -> Tuple[Game, ConnectFour]:
        """
        Load game from DB with row locking (FOR UPDATE).
        This prevents other transactions from modifying the game while we process a move.
        """
        result = await db.execute(select(Game).where(Game.id == game_id).with_for_update())
        game_db = result.scalar_one_or_none()
        
        if not game_db:
            raise ValueError(f"Game {game_id} not found")
            
        engine = ConnectFour()
        for move in game_db.history:
            engine.drop_piece(move['column'])
            
        return game_db, engine

    async def process_human_move(self, db: AsyncSession, game_id: int, column: int, provided_token: str = None) -> GameState:
        """Process a human player's move with locking and token validation"""
        # Start timer for human (optional, but keeps data consistent)
        start_time = time.time()
        
        # Acquire Lock immediately
        game_db, engine = await self._get_game_for_update(db, game_id)
        
        if game_db.status != GameStatus.IN_PROGRESS:
            return GameState(game_db, engine)
        
        # Token validation logic moved here (inside transaction)
        move_count = len(game_db.history)
        is_p1_turn = (move_count % 2 == 0)
        
        required_token = game_db.player_1_token if is_p1_turn else game_db.player_2_token
        if required_token and required_token != provided_token:
            raise ValueError("Unauthorized: Invalid player token")

        # Validate move
        if not engine.drop_piece(column):
            raise ValueError(f"Invalid move: column {column}")
        
        end_time = time.time()
        duration = round(end_time - start_time, 3)
        
        # Create move record
        move_record = {
            "player": engine.current_turn - 1 if engine.current_turn == 2 else 1,  # Previous player
            "column": column,
            "reasoning": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration": duration,
            "is_fallback": False,
            "cost_usd": 0.0  # Human moves have no cost
        }
        
        # Save move
        await self._save_move_to_db(db, game_db, move_record, engine)
        
        return GameState(game_db, engine)
    
    async def step_ai_turn(self, db: AsyncSession, game_id: int) -> Optional[GameState]:
        """Execute one AI turn and return new game state"""
        
        # 1. READ (No Lock) - Get snapshot for LLM to think
        try:
            game_db_snapshot, engine_snapshot = await self.get_game_state(db, game_id)
        except ValueError:
            return None

        # --- CRITICAL FIX START: Capture exact move count ---
        # We handle the case where history might be None
        history_snapshot = game_db_snapshot.history or []
        snapshot_move_count = len(history_snapshot)
        # --- CRITICAL FIX END ---

        # Check if game is already finished
        if engine_snapshot.winner or engine_snapshot.is_draw() or game_db_snapshot.status != GameStatus.IN_PROGRESS:
            return GameState(game_db_snapshot, engine_snapshot)
        
        # Determine which AI should play
        current_player = engine_snapshot.current_turn
        ai_model = game_db_snapshot.player_1_type if current_player == 1 else game_db_snapshot.player_2_type
        
        # Skip if current player is human
        if ai_model == PlayerType.HUMAN:
            return GameState(game_db_snapshot, engine_snapshot)
        
        # --- TIMER START ---
        start_time = time.time()
        
        # 2. THINK (Slow Operation - No DB Lock held)
        ai_agent = ConnectFourAI(player_id=current_player, model_name=ai_model)
        
        try:
            result = await ai_agent.get_move_async(engine_snapshot)
            decision = result["decision"]
            usage = result["usage"]
            
            # --- TIMER END ---
            end_time = time.time()
            duration = round(end_time - start_time, 3)
            
            # 3. WRITE (Acquire Lock) - Re-validate and Save
            # This ensures that if the state changed while AI was thinking, we handle it correctly
            game_db, engine = await self._get_game_for_update(db, game_id)
            
            # --- CRITICAL FIX START: Compare Move Counts ---
            current_history = game_db.history or []
            current_move_count = len(current_history)

            # If the number of moves in DB is different from when we started thinking,
            # the state has drifted (someone else played). ABORT.
            if current_move_count != snapshot_move_count:
                print(f"⚠️ Stale AI Move: Game advanced from {snapshot_move_count} to {current_move_count} while thinking. Discarding.")
                return GameState(game_db, engine)
            # --- CRITICAL FIX END ---

            # Re-verification inside lock (Status check)
            if game_db.status != GameStatus.IN_PROGRESS:
                print(f"Game {game_id} finished while AI was thinking.")
                return GameState(game_db, engine)

            # Validate and execute move on the LOCKED engine
            if not engine.drop_piece(decision.column):
                print(f"AI generated invalid move for current board state: {decision.column}")
                # Optional: Retry logic could go here. For now, we abort the turn.
                return GameState(game_db, engine)
            
            # Create move record with immutable cost tracking
            config = registry.get(ai_model)
            pricing = config.pricing if config else {"input": 0, "output": 0}
            
            move_cost = (
                (usage.get("input_tokens", 0) / 1_000_000 * pricing.get("input", 0)) +
                (usage.get("output_tokens", 0) / 1_000_000 * pricing.get("output", 0))
            )
            
            move_record = {
                "player": current_player,
                "column": decision.column,
                "reasoning": decision.reasoning,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "duration": duration,
                "is_fallback": getattr(decision, 'is_fallback', False),
                "cost_usd": move_cost  # Permanent record of what this move cost at that time
            }
            
            # Save move
            await self._save_move_to_db(db, game_db, move_record, engine)
            
            return GameState(game_db, engine)
            
        except Exception as e:
            err_msg = str(e).lower()
            
            # Check for Rate Limit signatures
            is_rate_limit = any(x in err_msg for x in ["429", "rate_limit", "rate limit", "throttled", "quota exceeded", "too many requests"])
            
            if is_rate_limit:
                print(f"⏳ Rate limit detected for {ai_model}. Snoozing game {game_id} for 10 minutes.")
                
                # 1. Acquire Lock for update
                game_db, engine = await self._get_game_for_update(db, game_id)
                
                # 2. Set Cooldown: 10 Minutes
                game_db.status = GameStatus.PAUSED
                game_db.retry_after = datetime.now(timezone.utc) + timedelta(minutes=10)
                
                await db.commit()
                print(f"⏳ Game {game_id} rate limited. Snoozing until {game_db.retry_after}")
                return None  # This stops the GameRunner loop for this game
            
            print(f"AI turn failed for {ai_model}: {e}")
            # Don't raise here, just return current state so game doesn't crash
            return GameState(game_db_snapshot, engine_snapshot)
    
    
    def _calculate_game_stats(self, game_db: Game) -> Dict[str, Any]:
        """Calculate final stats for a completed game using stored cost_usd values"""
        if not game_db.history:
            return {}
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_duration = 0.0
        total_cost_usd = 0.0
        
        for i, move in enumerate(game_db.history):
            input_tokens = move.get('input_tokens', 0)
            output_tokens = move.get('output_tokens', 0)
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_duration += move.get('duration', 0.0)
            total_cost_usd += move.get('cost_usd', 0.0)  # Use stored cost from move time
        
        total_tokens = total_input_tokens + total_output_tokens
        
        return {
            "total_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_duration": round(total_duration, 3),
            "total_cost_usd": round(total_cost_usd, 6)  # Store with high precision
        }
    
    async def _save_move_to_db(self, db: AsyncSession, game_db: Game, move_record: Dict[str, Any], engine: ConnectFour):
        """Save move to database with transaction safety"""
        try:
            # Update history
            new_history = list(game_db.history)
            new_history.append(move_record)
            game_db.history = new_history
            flag_modified(game_db, "history")  # Ensure SQLAlchemy detects JSONB changes
            
            # Update game status if finished
            if engine.winner:
                game_db.winner = engine.winner
                game_db.status = GameStatus.COMPLETED
            elif engine.is_draw():
                game_db.status = GameStatus.DRAW
            
            # Handle ELO updates for all completed games BEFORE committing
            if game_db.status in [GameStatus.COMPLETED, GameStatus.DRAW]:
                winner_id = engine.winner if engine.winner else 0  # 0 for draw
                
                # Calculate and store game stats
                game_db.stats = self._calculate_game_stats(game_db)
                
                # REPLACED DIRECT CALL WITH EVENT NOTIFICATION
                await game_events.notify_complete(db, game_db, winner_id)
            
            # Single atomic commit for both game result and ELO updates
            await db.commit()
            await db.refresh(game_db)
            
            # Trigger tournament bus to check for new games immediately
            tournament_bus.trigger()
            
        except Exception as e:
            print(f"DB Save Error: {e}")
            await db.rollback()
            raise ValueError(f"Failed to save move: {e}")


# Singleton instance
game_service = GameService()