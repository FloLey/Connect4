"""HTTP tests for the /stats endpoints."""

import pytest

from backend.app.models.elo_model import EloHistory, EloRating
from backend.app.models.enums import GameStatus
from backend.app.models.game_model import Game
from backend.app.models.stats import LeaderboardSnapshot, MatrixCell


class TestLeaderboard:
    async def test_empty_leaderboard(self, client):
        response = await client.get("/stats/leaderboard")
        assert response.status_code == 200
        assert response.json() == []

    async def test_leaderboard_orders_by_rating_desc(self, client, test_db):
        # Tier 2.4: leaderboard endpoint reads from materialized snapshots, so
        # we seed those directly. The aggregator service is exercised separately
        # in tests/services/test_stats_aggregator.py.
        test_db.add_all(
            [
                LeaderboardSnapshot(model_name="bottom", rating=1100.0, matches_played=1),
                LeaderboardSnapshot(model_name="top", rating=1500.0, matches_played=1),
                LeaderboardSnapshot(model_name="mid", rating=1300.0, matches_played=1),
            ]
        )
        await test_db.commit()

        response = await client.get("/stats/leaderboard")
        assert response.status_code == 200
        names = [entry["model_name"] for entry in response.json()]
        assert names == ["top", "mid", "bottom"]


class TestActiveGames:
    async def test_returns_in_progress_with_reconstructed_board(self, client, test_db):
        game = Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.IN_PROGRESS,
            history=[{"player": 1, "column": 0}],
        )
        test_db.add(game)
        await test_db.commit()

        response = await client.get("/stats/active-games")
        body = response.json()
        assert response.status_code == 200
        assert any(g["id"] == game.id for g in body)
        target = next(g for g in body if g["id"] == game.id)
        # Reconstructed: bottom-left cell = player 1 (= 1).
        assert target["board"][-1][0] == 1
        assert target["move_count"] == 1


class TestMatrix:
    async def test_grid_includes_every_pair_combination(self, client, test_db):
        # Snapshots provide the model order; matrix cells provide the win counts.
        test_db.add_all(
            [
                LeaderboardSnapshot(model_name="alpha", rating=1300.0),
                LeaderboardSnapshot(model_name="beta", rating=1200.0),
                MatrixCell(player_a="alpha", player_b="beta", wins_a=2, wins_b=1, draws=0, total=3),
                MatrixCell(player_a="beta", player_b="alpha", wins_a=1, wins_b=2, draws=0, total=3),
            ]
        )
        await test_db.commit()

        response = await client.get("/stats/matrix")
        assert response.status_code == 200
        body = response.json()
        assert body["models"] == ["alpha", "beta"]
        assert "alpha" in body["grid"]
        assert "beta" in body["grid"]["alpha"]
        # Self-vs-self cell exists with zeroed stats.
        assert body["grid"]["alpha"]["alpha"]["total"] == 0
        # Alpha vs beta: alpha wins 2, beta wins 1, total 3.
        ab = body["grid"]["alpha"]["beta"]
        assert ab["wins"] == 2
        assert ab["losses"] == 1
        assert ab["total"] == 3


class TestHistory:
    async def test_history_filtered_by_model(self, client, test_db):
        test_db.add_all(
            [
                EloHistory(model_name="x", rating=1210.0, match_id=1),
                EloHistory(model_name="y", rating=1190.0, match_id=1),
            ]
        )
        await test_db.commit()

        response = await client.get("/stats/history?model=x")
        assert response.status_code == 200
        names = {row["model_name"] for row in response.json()}
        assert names == {"x"}

    async def test_history_unfiltered_returns_all(self, client, test_db):
        test_db.add_all(
            [
                EloHistory(model_name="x", rating=1210.0, match_id=1),
                EloHistory(model_name="y", rating=1190.0, match_id=1),
            ]
        )
        await test_db.commit()

        response = await client.get("/stats/history")
        names = {row["model_name"] for row in response.json()}
        assert names == {"x", "y"}


class TestHistoryPlot:
    async def test_baseline_includes_every_model_at_match_zero(self, client, test_db):
        test_db.add_all(
            [
                EloHistory(model_name="alpha", rating=1210.0, match_id=1),
                EloHistory(model_name="beta", rating=1190.0, match_id=1),
            ]
        )
        await test_db.commit()

        response = await client.get("/stats/history-plot")
        assert response.status_code == 200
        body = response.json()
        # First entry is the baseline at match_number=0 with both models at 1200.
        assert body[0]["match_number"] == 0
        assert body[0]["alpha"] == 1200
        assert body[0]["beta"] == 1200
