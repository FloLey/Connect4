"""Tournament-level WebSocket: pushes GAME_STARTED / GAME_COMPLETED events.

Per-game UPDATEs still go through ``websocket_manager.manager``. This channel
is a separate, lightweight broadcast for the *list* of running games — so the
frontend's LiveGamesGrid can add/remove cards sub-second instead of waiting
for the 10 s tournament-config poll.
"""

from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, WebSocket

from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class TournamentConnectionManager:
    """Per-tournament fan-out for control events (no game state)."""

    def __init__(self) -> None:
        self._connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, ws: WebSocket, tournament_id: int) -> None:
        await ws.accept()
        self._connections.setdefault(tournament_id, []).append(ws)

    def disconnect(self, ws: WebSocket, tournament_id: int) -> None:
        conns = self._connections.get(tournament_id)
        if not conns:
            return
        if ws in conns:
            conns.remove(ws)
        if not conns:
            del self._connections[tournament_id]

    async def broadcast(self, tournament_id: int, message: dict) -> None:
        for ws in list(self._connections.get(tournament_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                # Drop dead connections silently.
                pass


tournament_ws_manager = TournamentConnectionManager()


router = APIRouter()


@router.websocket("/tournament/{tournament_id}/ws")
async def tournament_ws(websocket: WebSocket, tournament_id: int) -> None:
    """Subscribe to control-plane events for one tournament.

    Message shapes pushed to clients:
      {"type": "GAME_STARTED",   "game": {...LiveGameSummary fields...}}
      {"type": "GAME_COMPLETED", "game_id": int, "winner": int|null}
    """
    await tournament_ws_manager.connect(websocket, tournament_id)
    try:
        while True:
            # We don't expect inbound messages but keep the receive loop alive
            # so close events from the client are surfaced.
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        tournament_ws_manager.disconnect(websocket, tournament_id)
