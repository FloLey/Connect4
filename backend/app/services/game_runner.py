"""
Game Runner - Background Worker System

This service runs AI vs AI games in the background, independent of WebSocket connections.
It automatically resumes interrupted games on server startup and ensures games complete
even if users close their browsers.
"""

import asyncio
from sqlalchemy.future import select
from backend.app.core.config import settings
from backend.app.core.database import get_session_maker
from backend.app.core.logging import get_logger
from backend.app.models.game_model import Game
from backend.app.models.enums import GameStatus, PlayerType
from backend.app.services.game_service import game_service, GameState

logger = get_logger(__name__)


class GameRunner:
    def __init__(self):
        self.running_tasks = {}  # key format: "{env}_{game_id}" -> asyncio.Task

    def _make_key(self, env, game_id):
        return f"{env}_{game_id}"

    async def start_game_if_ai_vs_ai(self, game_id: int, env: str = "prod"):
        """Checks if a game is AI vs AI and starts the background loop if so."""
        key = self._make_key(env, game_id)
        if key in self.running_tasks:
            return  # Already running
        
        SessionLocal = get_session_maker(env)
        async with SessionLocal() as db:
            game, _ = await game_service.get_game_state(db, game_id)
            
            # Condition: Both players are AI
            is_ai_vs_ai = (game.player_1_type != PlayerType.HUMAN and game.player_2_type != PlayerType.HUMAN)
            
            if is_ai_vs_ai and game.status == GameStatus.IN_PROGRESS:
                logger.info("background_runner_start", game_id=game_id, env=env)
                self.running_tasks[key] = asyncio.create_task(self._game_loop(game_id, env))

    async def _game_loop(self, game_id: int, env: str):
        """The main loop that plays the game until completion."""
        # Late import to avoid circular imports
        from backend.app.api.websocket_manager import manager
        
        key = self._make_key(env, game_id)
        try:
            while True:
                # 1. Brief pause to simulate thinking/pacing
                await asyncio.sleep(settings.game_runner_pacing_seconds)

                SessionLocal = get_session_maker(env)
                async with SessionLocal() as db:
                    # 2. Execute Step
                    # Note: step_ai_turn handles locking and validation internally
                    new_state = await game_service.step_ai_turn(db, game_id)
                    
                    if not new_state:
                        # Game likely finished or error
                        break

                    # 3. Broadcast to anyone watching
                    await manager.broadcast(game_id, {
                        "type": "UPDATE",
                        "board": new_state.board,
                        "currentTurn": new_state.current_turn,
                        "winner": new_state.winner,
                        "status": new_state.status,
                        "lastMove": new_state.last_move
                    })

                    # 4. Check Exit Conditions
                    if new_state.winner or new_state.is_draw or new_state.status != GameStatus.IN_PROGRESS:
                        logger.info("background_game_finished", game_id=game_id)
                        break

        except asyncio.CancelledError:
            # Game was paused/cancelled - don't mark as failed, just exit
            logger.info("background_game_cancelled", game_id=game_id)
            raise  # Re-raise to ensure proper cleanup
        except Exception as e:
            logger.error("background_runner_error", game_id=game_id, error=str(e))
        finally:
            if key in self.running_tasks:
                del self.running_tasks[key]

    def is_game_running(self, game_id: int, env: str = "prod") -> bool:
        """Check if a game is currently being processed in background."""
        key = self._make_key(env, game_id)
        return key in self.running_tasks

    async def stop_game(self, game_id: int, env: str = "prod"):
        """Stop background processing for a specific game."""
        key = self._make_key(env, game_id)
        if key in self.running_tasks:
            self.running_tasks[key].cancel()
            del self.running_tasks[key]
            logger.info("background_runner_stopped", game_id=game_id, env=env)


# Singleton
game_runner = GameRunner()