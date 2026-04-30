"""GameService tests covering process_human_move and step_ai_turn.

`ConnectFourAI` is patched (via the ai_turn_executor module) so we don't make
real LLM calls.
"""

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.future import select

from backend.app.engine.ai import MoveDecision
from backend.app.models.enums import GameStatus, PlayerType
from backend.app.models.game_model import Game
from backend.app.services import ai_turn_executor as ai_turn_executor_module
from backend.app.services.game_service import game_service


class FakeAI:
    """Stand-in for ConnectFourAI with scriptable behavior."""

    def __init__(self, player_id, model_name=None):
        self.player_id = player_id
        self.model_name = model_name

    async def get_move_async(self, engine):
        valid = engine.get_valid_moves()
        # Default: first valid column.
        return {
            "decision": MoveDecision(reasoning="fake", column=valid[0], is_fallback=False),
            "usage": {"input_tokens": 7, "output_tokens": 11},
        }


class RateLimitedAI:
    def __init__(self, player_id, model_name=None):
        self.player_id = player_id
        self.model_name = model_name

    async def get_move_async(self, engine):
        raise RuntimeError("HTTP 429 rate_limit exceeded")


@pytest_asyncio.fixture
def patch_ai(monkeypatch):
    # Replace the ConnectFourAI class that AITurnExecutor instantiates.
    monkeypatch.setattr(
        ai_turn_executor_module.ai_turn_executor, "_ai_factory", FakeAI
    )
    return FakeAI


@pytest_asyncio.fixture
def patch_rate_limited_ai(monkeypatch):
    monkeypatch.setattr(
        ai_turn_executor_module.ai_turn_executor, "_ai_factory", RateLimitedAI
    )
    return RateLimitedAI


class TestCreateGame:
    async def test_creates_record_with_pending_history(self, test_db):
        game = await game_service.create_game(test_db, "model-a", "model-b")
        assert game.id is not None
        assert game.player_1_type == "model-a"
        assert game.history == []


class TestProcessHumanMove:
    async def test_drops_piece_and_records_move(self, test_db):
        game = await game_service.create_game(test_db, PlayerType.HUMAN, PlayerType.HUMAN)
        # Set token so validation accepts an empty provided_token=None path.
        await test_db.commit()

        state = await game_service.process_human_move(
            test_db, game.id, column=3, provided_token=None
        )

        assert state.last_move["column"] == 3
        # Engine current_turn flips to player 2.
        assert state.current_turn == 2

    async def test_invalid_column_raises(self, test_db):
        game = await game_service.create_game(test_db, PlayerType.HUMAN, PlayerType.HUMAN)
        await test_db.commit()

        with pytest.raises(ValueError, match="Invalid move"):
            await game_service.process_human_move(
                test_db, game.id, column=99, provided_token=None
            )

    async def test_wrong_token_raises(self, test_db):
        game = await game_service.create_game(test_db, PlayerType.HUMAN, "ai-model")
        # Assign a token to player 1 directly.
        game.player_1_token = "expected-token"
        await test_db.commit()

        with pytest.raises(ValueError, match="Unauthorized"):
            await game_service.process_human_move(
                test_db, game.id, column=0, provided_token="wrong-token"
            )

    async def test_returns_state_when_game_not_in_progress(self, test_db):
        game = await game_service.create_game(test_db, PlayerType.HUMAN, PlayerType.HUMAN)
        game.status = GameStatus.COMPLETED
        await test_db.commit()

        state = await game_service.process_human_move(
            test_db, game.id, column=0, provided_token=None
        )
        assert state.status == GameStatus.COMPLETED


class TestStepAiTurn:
    async def test_happy_path_records_ai_move(self, test_db, patch_ai):
        game = await game_service.create_game(test_db, "model-a", "model-b")

        state = await game_service.step_ai_turn(test_db, game.id)

        assert state is not None
        assert state.last_move["column"] == 0  # FakeAI plays first valid column
        assert state.last_move["reasoning"] == "fake"
        assert state.last_move["input_tokens"] == 7
        assert state.last_move["output_tokens"] == 11
        assert state.current_turn == 2

    async def test_returns_none_on_rate_limit_and_snoozes_game(
        self, test_db, patch_rate_limited_ai
    ):
        game = await game_service.create_game(test_db, "model-a", "model-b")

        result = await game_service.step_ai_turn(test_db, game.id)

        assert result is None
        await test_db.refresh(game)
        assert game.status == GameStatus.PAUSED
        assert game.retry_after is not None
        assert game.retry_after > datetime.now(timezone.utc)

    async def test_skips_when_current_player_is_human(self, test_db, patch_ai):
        game = await game_service.create_game(test_db, PlayerType.HUMAN, "ai-model")

        state = await game_service.step_ai_turn(test_db, game.id)

        # No move should have been recorded.
        await test_db.refresh(game)
        assert game.history == []
        assert state.current_turn == 1

    async def test_returns_none_for_missing_game(self, test_db, patch_ai):
        result = await game_service.step_ai_turn(test_db, game_id=9999)
        assert result is None
