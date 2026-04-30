"""HTTP tests for the /admin endpoints."""

import pytest

from backend.app.core.config import settings
from backend.app.models.elo_model import EloRating
from backend.app.models.enums import GameStatus
from backend.app.models.game_model import Game


class TestAdminStatus:
    async def test_returns_zero_counts_on_empty_db(self, client):
        response = await client.get("/admin/status")
        assert response.status_code == 200
        body = response.json()
        assert body == {"games": 0, "elo_ratings": 0, "elo_history": 0}

    async def test_returns_actual_counts(self, client, test_db):
        test_db.add_all(
            [
                Game(player_1_type="a", player_2_type="b", status=GameStatus.IN_PROGRESS, history=[]),
                EloRating(model_name="a", rating=1200.0),
                EloRating(model_name="b", rating=1200.0),
            ]
        )
        await test_db.commit()

        response = await client.get("/admin/status")
        body = response.json()
        assert body["games"] == 1
        assert body["elo_ratings"] == 2


class TestAdminReset:
    async def test_reset_requires_correct_confirmation(self, client):
        response = await client.delete("/admin/reset?confirmation=wrong")
        assert response.status_code == 400

    async def test_reset_truncates_tables(self, client, test_db):
        test_db.add_all(
            [
                Game(player_1_type="a", player_2_type="b", status=GameStatus.COMPLETED, history=[]),
                EloRating(model_name="a", rating=1300.0),
            ]
        )
        await test_db.commit()

        response = await client.delete(
            "/admin/reset?confirmation=I-UNDERSTAND-THIS-DELETES-EVERYTHING"
        )
        assert response.status_code == 200

        status = (await client.get("/admin/status")).json()
        assert status["games"] == 0
        assert status["elo_ratings"] == 0


class TestAdminAuth:
    async def test_no_token_setting_means_no_auth_required(self, client):
        # settings.admin_token is None by default → routes are open.
        assert settings.admin_token is None
        response = await client.get("/admin/status")
        assert response.status_code == 200

    async def test_wrong_token_when_configured_returns_401(self, client, monkeypatch):
        monkeypatch.setattr(settings, "admin_token", "secret-123")

        response = await client.get("/admin/status")  # client has no admin token
        assert response.status_code == 401

    async def test_correct_token_when_configured_passes(self, client, monkeypatch):
        monkeypatch.setattr(settings, "admin_token", "secret-123")

        response = await client.get(
            "/admin/status", headers={"X-Admin-Token": "secret-123"}
        )
        assert response.status_code == 200


class TestRebuildStats:
    async def test_rebuild_returns_summary(self, client, test_db):
        test_db.add(EloRating(model_name="solo", rating=1234.0, matches_played=1, wins=1))
        test_db.add(
            Game(
                player_1_type="solo",
                player_2_type="other",
                status=GameStatus.COMPLETED,
                winner=1,
                history=[],
            )
        )
        test_db.add(EloRating(model_name="other", rating=1166.0, matches_played=1, losses=1))
        await test_db.commit()

        response = await client.post("/admin/rebuild-stats")
        assert response.status_code == 200
        body = response.json()
        assert body["leaderboard_rows"] == 2
        assert body["matrix_rows"] == 2
        assert "Stats rebuilt" in body["message"]
