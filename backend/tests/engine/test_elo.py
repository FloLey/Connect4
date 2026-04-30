"""Tests for ELO calculation and idempotency.

Uses the live test DB via the `test_db` fixture so the EloHistory idempotency
check (a SELECT against `elo_history`) actually runs.
"""

import math

import pytest
from sqlalchemy.future import select

from backend.app.core.config import settings
from backend.app.engine.elo import (
    calculate_expected_score,
    get_or_create_rating,
    update_elo,
)
from backend.app.models.elo_model import EloHistory, EloRating
from backend.app.models.enums import GameStatus
from backend.app.models.game_model import Game


class TestExpectedScore:
    def test_equal_ratings_yield_half(self):
        assert calculate_expected_score(1200, 1200) == 0.5

    def test_higher_rating_dominates(self):
        assert calculate_expected_score(1400, 1200) > 0.5
        assert calculate_expected_score(1200, 1400) < 0.5

    def test_400_point_gap_is_about_91_percent(self):
        # Standard ELO property: 400 points → ~10:1 expected odds.
        score = calculate_expected_score(1600, 1200)
        assert math.isclose(score, 1 / (1 + 10 ** (-1)), rel_tol=1e-9)


def _make_completed_game(p1: str, p2: str, winner: int | None) -> Game:
    return Game(
        player_1_type=p1,
        player_2_type=p2,
        winner=winner,
        status=GameStatus.COMPLETED if winner else GameStatus.DRAW,
        history=[
            {"player": 1, "column": 0, "input_tokens": 10, "output_tokens": 5, "duration": 1.0},
            {"player": 2, "column": 1, "input_tokens": 12, "output_tokens": 6, "duration": 1.5},
        ],
    )


class TestUpdateElo:
    async def test_get_or_create_rating_initializes_at_base(self, test_db):
        rating = await get_or_create_rating(test_db, "model-x")
        await test_db.commit()

        assert rating.rating == settings.elo_base_rating

    async def test_winner_gains_loser_loses(self, test_db):
        game = _make_completed_game("alpha", "beta", winner=1)
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await update_elo(test_db, "alpha", "beta", winner_id=1, game_id=game.id)
        await test_db.commit()

        alpha = (await test_db.execute(select(EloRating).where(EloRating.model_name == "alpha"))).scalar_one()
        beta = (await test_db.execute(select(EloRating).where(EloRating.model_name == "beta"))).scalar_one()

        # Both started at 1200; equal expected score 0.5.
        # Winner: 1200 + 32*(1.0 - 0.5) = 1216
        assert math.isclose(alpha.rating, 1200 + settings.elo_k_factor * 0.5)
        assert math.isclose(beta.rating, 1200 - settings.elo_k_factor * 0.5)
        assert alpha.wins == 1
        assert beta.losses == 1

    async def test_draw_pulls_ratings_toward_each_other(self, test_db):
        game = _make_completed_game("strong", "weak", winner=None)
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        # Seed ratings
        strong = await get_or_create_rating(test_db, "strong")
        weak = await get_or_create_rating(test_db, "weak")
        strong.rating = 1400
        weak.rating = 1200
        await test_db.commit()

        await update_elo(test_db, "strong", "weak", winner_id=0, game_id=game.id)
        await test_db.commit()

        await test_db.refresh(strong)
        await test_db.refresh(weak)

        assert strong.rating < 1400  # Higher-rated player loses points on draw
        assert weak.rating > 1200    # Lower-rated player gains points on draw
        assert strong.draws == 1
        assert weak.draws == 1

    async def test_idempotency_via_match_id(self, test_db):
        game = _make_completed_game("a", "b", winner=2)
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await update_elo(test_db, "a", "b", winner_id=2, game_id=game.id)
        await test_db.commit()

        a_first = (await test_db.execute(select(EloRating).where(EloRating.model_name == "a"))).scalar_one()
        rating_after_first = a_first.rating
        wins_after_first = a_first.wins or 0

        # Second call with same game_id should be a no-op.
        await update_elo(test_db, "a", "b", winner_id=2, game_id=game.id)
        await test_db.commit()
        await test_db.refresh(a_first)

        assert a_first.rating == rating_after_first
        assert (a_first.wins or 0) == wins_after_first

        history = (await test_db.execute(select(EloHistory).where(EloHistory.match_id == game.id))).scalars().all()
        assert len(history) == 2  # Only one pair from the first call

    async def test_aggregates_token_and_duration_stats(self, test_db):
        game = _make_completed_game("foo", "bar", winner=1)
        test_db.add(game)
        await test_db.commit()
        await test_db.refresh(game)

        await update_elo(test_db, "foo", "bar", winner_id=1, game_id=game.id)
        await test_db.commit()

        foo = (await test_db.execute(select(EloRating).where(EloRating.model_name == "foo"))).scalar_one()
        bar = (await test_db.execute(select(EloRating).where(EloRating.model_name == "bar"))).scalar_one()

        # Player-1 moves: input=10, output=5, duration=1.0, count=1
        assert foo.total_input_tokens == 10
        assert foo.total_output_tokens == 5
        assert foo.total_moves == 1
        assert math.isclose(foo.total_duration_seconds, 1.0)

        # Player-2 moves: input=12, output=6, duration=1.5, count=1
        assert bar.total_input_tokens == 12
        assert bar.total_output_tokens == 6
        assert bar.total_moves == 1
        assert math.isclose(bar.total_duration_seconds, 1.5)
