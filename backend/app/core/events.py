from typing import Callable, List, Any
import asyncio

from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class GameEvents:
    def __init__(self):
        self._on_complete_listeners: List[Callable] = []

    def subscribe_complete(self, callback: Callable):
        self._on_complete_listeners.append(callback)

    async def notify_complete(self, db, game, winner_id):
        # Fire and forget or await depending on strictness
        for listener in self._on_complete_listeners:
            try:
                await listener(db, game, winner_id)
            except Exception as e:
                logger.error("event_listener_error", error=str(e))


game_events = GameEvents()