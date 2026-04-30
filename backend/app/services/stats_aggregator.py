"""Materialized-stats refresh listener.

Subscribed to ``game_events.notify_complete`` alongside the ELO updater so
that completed games immediately propagate into ``leaderboard_snapshots`` and
``matrix_cells``. Reads from the ELO tables (which the ELO listener writes
*first*) and from ``registry`` for pricing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import select

from backend.app.core.logging import get_logger
from backend.app.core.model_registry import registry
from backend.app.models.elo_model import EloRating
from backend.app.models.enums import GameStatus
from backend.app.models.game_model import Game
from backend.app.models.stats import LeaderboardSnapshot, MatrixCell

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


class StatsAggregator:
    """Refresh helpers for the materialized stats tables.

    Stateless — both methods take an ``AsyncSession`` and operate inside the
    caller's transaction.
    """

    async def refresh_for_game(
        self, db: "AsyncSession", game: Game, winner_id: int
    ) -> None:
        """Re-derive snapshots for the two players of a just-completed game.

        Call ordering matters: this listener runs *after* the ELO updater so
        that ``EloRating`` rows reflect the new totals.
        """
        p1 = game.player_1_type
        p2 = game.player_2_type
        if not p1 or not p2:
            return  # Should never happen; defensive.

        await self._upsert_leaderboard_for(db, p1)
        if p1 != p2:
            await self._upsert_leaderboard_for(db, p2)

        await self._upsert_matrix_cell(db, p1, p2, winner_id)
        if p1 != p2:
            await self._upsert_matrix_cell(db, p2, p1, _flip(winner_id))

    async def rebuild_all(self, db: "AsyncSession") -> dict:
        """Full rebuild from EloRating + Game tables.

        Used once after the alembic migration adds the new tables, and exposed
        as ``POST /admin/rebuild-stats`` for manual recovery. Returns a small
        summary dict.
        """
        # Wipe both tables, then walk all completed/draw games + ELO records.
        await db.execute(MatrixCell.__table__.delete())
        await db.execute(LeaderboardSnapshot.__table__.delete())

        # 1. Leaderboard snapshots from EloRating rows.
        ratings = (await db.execute(select(EloRating))).scalars().all()
        for rating in ratings:
            db.add(self._build_snapshot(rating))

        # 2. Matrix cells from completed games.
        games = (
            await db.execute(
                select(Game).where(
                    Game.status.in_([GameStatus.COMPLETED, GameStatus.DRAW])
                )
            )
        ).scalars().all()

        cell_state: dict[tuple[str, str], dict[str, int]] = {}
        for game in games:
            p1 = game.player_1_type
            p2 = game.player_2_type
            winner = game.winner or 0
            if not p1 or not p2:
                continue
            self._tally_cell(cell_state, p1, p2, winner)
            if p1 != p2:
                self._tally_cell(cell_state, p2, p1, _flip(winner))

        for (a, b), agg in cell_state.items():
            db.add(
                MatrixCell(
                    player_a=a,
                    player_b=b,
                    wins_a=agg["wins_a"],
                    wins_b=agg["wins_b"],
                    draws=agg["draws"],
                    total=agg["total"],
                )
            )

        await db.commit()
        logger.info(
            "stats_rebuilt",
            leaderboard_rows=len(ratings),
            matrix_rows=len(cell_state),
        )
        return {"leaderboard_rows": len(ratings), "matrix_rows": len(cell_state)}

    # -- helpers -----------------------------------------------------------

    async def _upsert_leaderboard_for(self, db: "AsyncSession", model_name: str) -> None:
        rating = (
            await db.execute(
                select(EloRating).where(EloRating.model_name == model_name)
            )
        ).scalar_one_or_none()
        if rating is None:
            # No ELO yet — nothing to snapshot.
            return

        snap = (
            await db.execute(
                select(LeaderboardSnapshot).where(
                    LeaderboardSnapshot.model_name == model_name
                )
            )
        ).scalar_one_or_none()

        new_snap = self._build_snapshot(rating)
        if snap is None:
            db.add(new_snap)
        else:
            for col in (
                "rating",
                "matches_played",
                "wins",
                "losses",
                "draws",
                "total_input_tokens",
                "total_output_tokens",
                "total_duration_seconds",
                "total_moves",
                "cost_input_total",
                "cost_output_total",
            ):
                setattr(snap, col, getattr(new_snap, col))

    async def _upsert_matrix_cell(
        self, db: "AsyncSession", a: str, b: str, winner_id: int
    ) -> None:
        cell = (
            await db.execute(
                select(MatrixCell).where(
                    MatrixCell.player_a == a, MatrixCell.player_b == b
                )
            )
        ).scalar_one_or_none()
        if cell is None:
            cell = MatrixCell(player_a=a, player_b=b)
            db.add(cell)
            cell.wins_a = 0
            cell.wins_b = 0
            cell.draws = 0
            cell.total = 0

        if winner_id == 1:
            cell.wins_a = (cell.wins_a or 0) + 1
        elif winner_id == 2:
            cell.wins_b = (cell.wins_b or 0) + 1
        else:
            cell.draws = (cell.draws or 0) + 1
        cell.total = (cell.total or 0) + 1

    @staticmethod
    def _build_snapshot(rating: EloRating) -> LeaderboardSnapshot:
        config = registry.get(rating.model_name)
        pricing = config.pricing if config else {"input": 0.0, "output": 0.0}
        cost_input = (rating.total_input_tokens or 0) / 1_000_000 * pricing.get("input", 0.0)
        cost_output = (rating.total_output_tokens or 0) / 1_000_000 * pricing.get("output", 0.0)
        return LeaderboardSnapshot(
            model_name=rating.model_name,
            rating=rating.rating or 0.0,
            matches_played=rating.matches_played or 0,
            wins=rating.wins or 0,
            losses=rating.losses or 0,
            draws=rating.draws or 0,
            total_input_tokens=rating.total_input_tokens or 0,
            total_output_tokens=rating.total_output_tokens or 0,
            total_duration_seconds=rating.total_duration_seconds or 0.0,
            total_moves=rating.total_moves or 0,
            cost_input_total=cost_input,
            cost_output_total=cost_output,
        )

    @staticmethod
    def _tally_cell(state, a, b, winner_id):
        agg = state.setdefault(
            (a, b), {"wins_a": 0, "wins_b": 0, "draws": 0, "total": 0}
        )
        if winner_id == 1:
            agg["wins_a"] += 1
        elif winner_id == 2:
            agg["wins_b"] += 1
        else:
            agg["draws"] += 1
        agg["total"] += 1


def _flip(winner_id: int) -> int:
    """When recording the (b, a) row, flip 1↔2 so player_a/player_b semantics stay consistent."""
    if winner_id == 1:
        return 2
    if winner_id == 2:
        return 1
    return 0


stats_aggregator = StatsAggregator()


async def refresh_stats_on_complete(db, game, winner_id):
    """Listener entry point for ``game_events.subscribe_complete``."""
    await stats_aggregator.refresh_for_game(db, game, winner_id)
