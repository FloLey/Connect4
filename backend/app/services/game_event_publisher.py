"""Thin wrapper around the existing event bus + tournament signal.

Centralizes the two callsites in ``GameService`` and gives future events
(stats refresh, audit logs, etc.) a clear home. Uses the existing
``game_events`` and ``tournament_bus`` singletons — no new infra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.app.core.events import game_events
from backend.app.services.tournament_bus import tournament_bus

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.app.models.game_model import Game


class GameEventPublisher:
    """Fan-out for game-completion events plus tournament-tick signal."""

    async def publish_complete(self, db: "AsyncSession", game: "Game", winner_id: int) -> None:
        """Notify subscribers (ELO, stats aggregator, …) that a game finished."""
        await game_events.notify_complete(db, game, winner_id)

    def signal_tournament_tick(self) -> None:
        """Wake the tournament watcher so it can schedule the next slot."""
        tournament_bus.trigger()


# Singleton — no per-game state lives on this object.
game_event_publisher = GameEventPublisher()
