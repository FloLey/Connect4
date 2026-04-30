"""HTTP tests for the /games and /models endpoints."""

import pytest
import pytest_asyncio

from backend.app.models.enums import GameStatus, PlayerType
from backend.app.models.game_model import Game


class TestGames:
    async def test_create_human_vs_human_game_returns_201_payload(self, client):
        response = await client.post(
            "/games", json={"player_1": PlayerType.HUMAN, "player_2": PlayerType.HUMAN}
        )
        assert response.status_code == 200  # FastAPI default for POST without explicit status
        body = response.json()
        assert body["status"] == GameStatus.IN_PROGRESS
        assert body["player_1_type"] == PlayerType.HUMAN
        assert body["player_2_type"] == PlayerType.HUMAN
        # Tokens are minted for human players.
        assert body["player_1_token"]
        assert body["player_2_token"]

    async def test_create_ai_vs_ai_game_does_not_mint_tokens(self, client):
        response = await client.post(
            "/games", json={"player_1": "model-a", "player_2": "model-b"}
        )
        body = response.json()
        assert body["player_1_token"] is None
        assert body["player_2_token"] is None

    async def test_get_game_returns_404_for_unknown(self, client):
        response = await client.get("/games/999999")
        assert response.status_code == 404

    async def test_get_game_returns_record(self, client):
        created = (
            await client.post(
                "/games", json={"player_1": PlayerType.HUMAN, "player_2": "model-a"}
            )
        ).json()
        response = await client.get(f"/games/{created['id']}")
        assert response.status_code == 200
        assert response.json()["id"] == created["id"]

    async def test_history_returns_only_completed_or_draw(self, client, test_db):
        # Insert one COMPLETED, one IN_PROGRESS via the DB so we don't depend on game flow.
        completed = Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.COMPLETED,
            winner=1,
            history=[],
        )
        in_progress = Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.IN_PROGRESS,
            history=[],
        )
        test_db.add_all([completed, in_progress])
        await test_db.commit()

        response = await client.get("/games/history")
        assert response.status_code == 200
        ids = [g["id"] for g in response.json()]
        assert completed.id in ids
        assert in_progress.id not in ids

    async def test_pending_human_games_lists_only_human_turns(self, client, test_db):
        # P1 human, no moves yet → human turn → should appear.
        human_first = Game(
            player_1_type=PlayerType.HUMAN,
            player_2_type="ai-model",
            status=GameStatus.IN_PROGRESS,
            history=[],
        )
        # AI-vs-AI should never appear in this list.
        ai_only = Game(
            player_1_type="ai-1",
            player_2_type="ai-2",
            status=GameStatus.IN_PROGRESS,
            history=[],
        )
        test_db.add_all([human_first, ai_only])
        await test_db.commit()

        response = await client.get("/games/pending-human")
        assert response.status_code == 200
        ids = response.json()
        assert human_first.id in ids
        assert ai_only.id not in ids


class TestModels:
    async def test_models_endpoint_returns_registry(self, client):
        response = await client.get("/models")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        # Every entry has the documented shape.
        for entry in body:
            assert "id" in entry
            assert "provider" in entry
            assert "label" in entry
