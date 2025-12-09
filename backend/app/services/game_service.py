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
from typing import Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.attributes import flag_modified

from backend.app.models.game_model import Game
from backend.app.engine.game import ConnectFour
from backend.app.engine.ai import ConnectFourAI
from backend.app.engine.elo import update_elo


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

    async def process_human_move(self, db: AsyncSession, game_id: int, column: int) -> GameState:
        """Process a human player's move with locking"""
        # Start timer for human (optional, but keeps data consistent)
        start_time = time.time()
        
        # Acquire Lock immediately
        game_db, engine = await self._get_game_for_update(db, game_id)
        
        if game_db.status != "IN_PROGRESS":
            return GameState(game_db, engine)

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
            "duration": duration  # <--- Added Duration
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

        # Check if game is already finished
        if engine_snapshot.winner or engine_snapshot.is_draw() or game_db_snapshot.status != "IN_PROGRESS":
            return GameState(game_db_snapshot, engine_snapshot)
        
        # Determine which AI should play
        current_player = engine_snapshot.current_turn
        ai_model = game_db_snapshot.player_1_type if current_player == 1 else game_db_snapshot.player_2_type
        
        # Skip if current player is human
        if ai_model == "human":
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
            
            # Re-verification inside lock
            if game_db.status != "IN_PROGRESS":
                print(f"Game {game_id} finished while AI was thinking.")
                return GameState(game_db, engine)
                
            if engine.current_turn != current_player:
                print(f"Turn changed while AI was thinking (Race condition prevented).")
                return GameState(game_db, engine)

            # Validate and execute move on the LOCKED engine
            if not engine.drop_piece(decision.column):
                print(f"AI generated invalid move for current board state: {decision.column}")
                # Optional: Retry logic could go here. For now, we abort the turn.
                return GameState(game_db, engine)
            
            # Create move record
            move_record = {
                "player": current_player,
                "column": decision.column,
                "reasoning": decision.reasoning,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "duration": duration  # <--- Added Duration
            }
            
            # Save move
            await self._save_move_to_db(db, game_db, move_record, engine)
            
            return GameState(game_db, engine)
            
        except Exception as e:
            print(f"AI turn failed for {ai_model}: {e}")
            # Don't raise here, just return current state so game doesn't crash
            return GameState(game_db_snapshot, engine_snapshot)
    
    
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
                game_db.status = "COMPLETED"
            elif engine.is_draw():
                game_db.status = "DRAW"
            
            # Handle ELO updates for all completed games BEFORE committing
            if game_db.status in ["COMPLETED", "DRAW"]:
                
                winner_id = engine.winner if engine.winner else 0  # 0 for draw
                
                # The update_elo function now handles idempotency internally
                await update_elo(
                    db,
                    game_db.player_1_type,
                    game_db.player_2_type,
                    winner_id,
                    game_db.id
                )
            
            # Single atomic commit for both game result and ELO updates
            await db.commit()
            await db.refresh(game_db)
            
        except Exception as e:
            print(f"DB Save Error: {e}")
            await db.rollback()
            raise ValueError(f"Failed to save move: {e}")


# Singleton instance
game_service = GameService()