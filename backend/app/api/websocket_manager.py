"""
Hybrid WebSocket Manager - Background Game Runner Architecture

This manager handles WebSocket connections with hybrid logic:
- Accepts human moves and forwards them to game service
- Broadcasts game state updates to connected clients
- Triggers AI moves for Human vs AI games
- Does NOT drive AI vs AI games (Background Game Runner handles that)
"""

import json
import asyncio
from typing import Dict, List, Set
from fastapi import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.game_service import game_service, GameState
from backend.app.core.database import get_session_maker
from backend.app.models.enums import GameStatus, PlayerType


class ConnectionManager:
    def __init__(self):
        # Maps game_id -> List of connected WebSockets
        self.active_connections: Dict[int, List[WebSocket]] = {}
        
        # Track games where AI is currently thinking (for Human vs AI games only)
        self.processing_game_ids: Set[int] = set()
        

    async def connect(self, websocket: WebSocket, game_id: int):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: int):
        if game_id in self.active_connections:
            if websocket in self.active_connections[game_id]:
                self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
                # Cleanup AI processing lock if exists
                if game_id in self.processing_game_ids:
                    self.processing_game_ids.remove(game_id)

    async def broadcast(self, game_id: int, message: dict):
        """Broadcast message to all connected clients for this game"""
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id][:]:
                try:
                    await connection.send_json(message)
                except Exception:
                    # Remove dead connections
                    pass

    async def handle_game_session(self, websocket: WebSocket, game_id: int):
        """Handle WebSocket connection for a game session - PASSIVE MODE"""
        # 1. Read Query Params
        env = websocket.query_params.get("env", "prod")
        player_token = websocket.query_params.get("token", None)
        
        await self.connect(websocket, game_id)
        SessionLocal = get_session_maker(env)

        try:
            # Send initial game state
            async with SessionLocal() as db:
                try:
                    game_db, engine = await game_service.get_game_state(db, game_id)
                    current_state = GameState(game_db, engine)
                    await websocket.send_json(self._build_state_message(current_state))
                    
                    # RECOVERY: Check if AI should play after reconnection (for stuck games)
                    if game_db.status == GameStatus.IN_PROGRESS and not current_state.winner and not current_state.is_draw:
                        await self._check_and_trigger_ai_for_human_vs_ai(game_id, current_state, env)
                        
                except ValueError as e:
                    await websocket.close(code=4004)  # Game not found
                    return

            # Listen for human moves only
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                
                if payload.get("action") == "MOVE":
                    await self._handle_human_move(game_id, payload["column"], player_token, env)

        except Exception as e:
            # Silently handle disconnects
            pass
        finally:
            self.disconnect(websocket, game_id)

    async def _handle_human_move(self, game_id: int, column: int, player_token: str = None, env: str = "prod"):
        """Process a human move and trigger AI response if needed (Human vs AI games)"""
        SessionLocal = get_session_maker(env)
        async with SessionLocal() as db:
            try:
                # Let the service handle validation inside the lock
                state = await game_service.process_human_move(db, game_id, column, player_token)
                
                # Broadcast the updated state
                await self.broadcast(game_id, self._build_state_message(state))
                
                # Check if we should trigger AI for Human vs AI games
                # (Background Game Runner handles AI vs AI games)
                if not state.winner and not state.is_draw:
                    await self._check_and_trigger_ai_for_human_vs_ai(game_id, state, env)
                    
            except ValueError as e:
                # Invalid move - could send error message to client
                print(f"Invalid move: {e}")
            except Exception as e:
                print(f"Error processing human move: {e}")

    async def _check_and_trigger_ai_for_human_vs_ai(self, game_id: int, current_state: GameState, env: str = "prod"):
        """Check if AI should play next and trigger it (ONLY for Human vs AI games)"""
        # Determine if current turn is AI
        current_ai_model = (current_state.player_1_type if current_state.current_turn == 1 
                          else current_state.player_2_type)
        
        # Only trigger if:
        # 1. Current turn is AI (not human)
        # 2. This is NOT an AI vs AI game (Background Game Runner handles those)
        is_human_vs_ai = (
            (current_state.player_1_type == PlayerType.HUMAN and current_state.player_2_type != PlayerType.HUMAN) or
            (current_state.player_1_type != PlayerType.HUMAN and current_state.player_2_type == PlayerType.HUMAN)
        )
        
        if current_ai_model != PlayerType.HUMAN and is_human_vs_ai and not current_state.winner and not current_state.is_draw:
            # Trigger AI turn for Human vs AI games only
            asyncio.create_task(self._execute_ai_turn_for_human_vs_ai(game_id, env))

    async def _execute_ai_turn_for_human_vs_ai(self, game_id: int, env: str = "prod"):
        """Execute AI turn for Human vs AI games with proper locking"""
        # 1. LOCK CHECK (prevent double AI turns)
        if game_id in self.processing_game_ids:
            return  # Already thinking! Stop.
            
        self.processing_game_ids.add(game_id)
        
        try:
            await self.broadcast(game_id, {"type": "THINKING_START"})
            
            # 2. PERFORM AI LOGIC
            SessionLocal = get_session_maker(env)
            async with SessionLocal() as db:
                new_state = await game_service.step_ai_turn(db, game_id)
                
                if new_state:
                    await self.broadcast(game_id, {"type": "THINKING_END"})
                    
                    # 3. BROADCAST UPDATE
                    await self.broadcast(game_id, self._build_state_message(new_state))
                        
        except Exception as e:
            print(f"AI turn error (Human vs AI): {e}")
            await self.broadcast(game_id, {"type": "THINKING_END"})
        finally:
            # 4. RELEASE LOCK
            if game_id in self.processing_game_ids:
                self.processing_game_ids.remove(game_id)

    def _build_state_message(self, state: GameState) -> dict:
        """Build WebSocket message from GameState"""
        return {
            "type": "UPDATE",
            "board": state.board,
            "currentTurn": state.current_turn,
            "winner": state.winner,
            "status": state.status,
            "lastMove": state.last_move
        }


# Singleton instance
manager = ConnectionManager()