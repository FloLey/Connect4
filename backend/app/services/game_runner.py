"""
Game Runner - Background Worker System

This service runs AI vs AI games in the background, independent of WebSocket connections.
It automatically resumes interrupted games on server startup and ensures games complete
even if users close their browsers.
"""

import asyncio
from sqlalchemy.future import select
from backend.app.core.database import AsyncSessionLocal
from backend.app.models.game_model import Game
from backend.app.services.game_service import game_service, GameState


class GameRunner:
    def __init__(self):
        self.running_tasks = {}  # game_id -> asyncio.Task

    async def start_game_if_ai_vs_ai(self, game_id: int):
        """Checks if a game is AI vs AI and starts the background loop if so."""
        async with AsyncSessionLocal() as db:
            game, _ = await game_service.get_game_state(db, game_id)
            
            # Condition: Both players are AI
            is_ai_vs_ai = (game.player_1_type != "human" and game.player_2_type != "human")
            
            if is_ai_vs_ai and game.status == "IN_PROGRESS":
                if game_id not in self.running_tasks:
                    print(f"🚀 Starting Background Runner for Game {game_id}")
                    self.running_tasks[game_id] = asyncio.create_task(self._game_loop(game_id))

    async def _game_loop(self, game_id: int):
        """The main loop that plays the game until completion."""
        # Late import to avoid circular imports
        from backend.app.api.websocket_manager import manager
        
        try:
            while True:
                # 1. Brief pause to simulate thinking/pacing
                await asyncio.sleep(1.5)

                async with AsyncSessionLocal() as db:
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
                    if new_state.winner or new_state.is_draw or new_state.status != "IN_PROGRESS":
                        print(f"🏁 Game {game_id} Finished in background.")
                        break
                        
        except Exception as e:
            print(f"❌ Background Runner Error (Game {game_id}): {e}")
        finally:
            if game_id in self.running_tasks:
                del self.running_tasks[game_id]

    def is_game_running(self, game_id: int) -> bool:
        """Check if a game is currently being processed in background."""
        return game_id in self.running_tasks

    async def stop_game(self, game_id: int):
        """Stop background processing for a specific game."""
        if game_id in self.running_tasks:
            self.running_tasks[game_id].cancel()
            del self.running_tasks[game_id]
            print(f"🛑 Stopped Background Runner for Game {game_id}")


# Singleton
game_runner = GameRunner()