"""Tests for the materialized-stats aggregator (Tier 2.4)."""

import pytest
from sqlalchemy.future import select

from backend.app.models.elo_model import EloRating
from backend.app.models.enums import GameStatus
from backend.app.models.game_model import Game
from backend.app.models.stats import LeaderboardSnapshot, MatrixCell
from backend.app.services.stats_aggregator import stats_aggregator


def _seed_rating(db, *, name, rating=1200.0, **kwargs):
    row = EloRating(model_name=name, rating=rating, **kwargs)
    db.add(row)
    return row


class TestRefreshForGame:
    async def test_creates_snapshots_for_both_players(self, test_db):
        _seed_rating(
            test_db,
            name="alpha",
            rating=1224.0,
            matches_played=2,
            wins=1,
            losses=1,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_moves=20,
        )
        _seed_rating(
            test_db,
            name="beta",
            rating=1176.0,
            matches_played=2,
            wins=1,
            losses=1,
        )
        game = Game(
            player_1_type="alpha",
            player_2_type="beta",
            status=GameStatus.COMPLETED,
            winner=1,
            history=[],
        )
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await stats_aggregator.refresh_for_game(test_db, game, winner_id=1)
        await test_db.commit()

        snaps = (
            await test_db.execute(
                select(LeaderboardSnapshot).order_by(LeaderboardSnapshot.model_name)
            )
        ).scalars().all()
        names = [s.model_name for s in snaps]
        assert names == ["alpha", "beta"]
        alpha = next(s for s in snaps if s.model_name == "alpha")
        assert alpha.rating == 1224.0
        assert alpha.wins == 1
        assert alpha.matches_played == 2

    async def test_writes_matrix_cells_for_both_directions(self, test_db):
        _seed_rating(test_db, name="x")
        _seed_rating(test_db, name="y")
        game = Game(
            player_1_type="x",
            player_2_type="y",
            status=GameStatus.COMPLETED,
            winner=2,
            history=[],
        )
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await stats_aggregator.refresh_for_game(test_db, game, winner_id=2)
        await test_db.commit()

        cells = (await test_db.execute(select(MatrixCell))).scalars().all()
        assert len(cells) == 2  # (x,y) + (y,x)
        xy = next(c for c in cells if c.player_a == "x" and c.player_b == "y")
        yx = next(c for c in cells if c.player_a == "y" and c.player_b == "x")
        assert xy.wins_b == 1  # row x → opponent y won the game
        assert yx.wins_a == 1  # row y → row was the winner

    async def test_idempotent_for_leaderboard(self, test_db):
        """Snapshots are derived from EloRating totals, not deltas — calling
        refresh twice should leave the snapshot unchanged."""
        _seed_rating(test_db, name="solo", rating=1234.5, matches_played=3, wins=2, losses=1)
        game = Game(
            player_1_type="solo",
            player_2_type="other",
            status=GameStatus.COMPLETED,
            winner=1,
            history=[],
        )
        _seed_rating(test_db, name="other")
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await stats_aggregator.refresh_for_game(test_db, game, winner_id=1)
        await test_db.commit()

        first = (
            await test_db.execute(
                select(LeaderboardSnapshot).where(LeaderboardSnapshot.model_name == "solo")
            )
        ).scalar_one()
        first_rating = first.rating
        first_matches = first.matches_played

        await stats_aggregator.refresh_for_game(test_db, game, winner_id=1)
        await test_db.commit()
        await test_db.refresh(first)
        assert first.rating == first_rating
        assert first.matches_played == first_matches

    async def test_draw_increments_draw_counters(self, test_db):
        _seed_rating(test_db, name="a")
        _seed_rating(test_db, name="b")
        game = Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.DRAW,
            winner=None,
            history=[],
        )
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await stats_aggregator.refresh_for_game(test_db, game, winner_id=0)
        await test_db.commit()

        cells = (await test_db.execute(select(MatrixCell))).scalars().all()
        for c in cells:
            assert c.draws == 1
            assert c.total == 1


class TestRebuildAll:
    async def test_rebuilds_from_elo_and_games(self, test_db):
        _seed_rating(test_db, name="a", rating=1300.0, matches_played=1, wins=1)
        _seed_rating(test_db, name="b", rating=1100.0, matches_played=1, losses=1)
        test_db.add(
            Game(
                player_1_type="a",
                player_2_type="b",
                status=GameStatus.COMPLETED,
                winner=1,
                history=[],
            )
        )
        await test_db.commit()

        summary = await stats_aggregator.rebuild_all(test_db)

        assert summary["leaderboard_rows"] == 2
        assert summary["matrix_rows"] == 2  # both directions

        snaps = (await test_db.execute(select(LeaderboardSnapshot))).scalars().all()
        assert {s.model_name for s in snaps} == {"a", "b"}

    async def test_rebuild_wipes_stale_rows(self, test_db):
        # Pre-existing stale snapshot for a model that is no longer in EloRating.
        test_db.add(LeaderboardSnapshot(model_name="ghost", rating=999.0))
        await test_db.commit()

        await stats_aggregator.rebuild_all(test_db)

        names = [
            r[0]
            for r in (
                await test_db.execute(select(LeaderboardSnapshot.model_name))
            ).all()
        ]
        assert "ghost" not in names
