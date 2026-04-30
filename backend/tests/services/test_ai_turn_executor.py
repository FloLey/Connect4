"""Tests for AITurnExecutor (no DB, fakes the LLM)."""

import pytest

from backend.app.engine.ai import MoveDecision
from backend.app.engine.game import ConnectFour
from backend.app.engine.rate_limit import RateLimitedError
from backend.app.services.ai_turn_executor import AITurnExecutor


class _ScriptedAI:
    """Configurable fake — pluggable into AITurnExecutor via the factory arg."""

    def __init__(self, player_id, model_name=None, behavior=None):
        self.player_id = player_id
        self.model_name = model_name
        self._behavior = behavior or (lambda eng: {
            "decision": MoveDecision(reasoning="ok", column=eng.get_valid_moves()[0]),
            "usage": {"input_tokens": 5, "output_tokens": 10},
        })

    async def get_move_async(self, engine):
        result = self._behavior(engine)
        if isinstance(result, BaseException):
            raise result
        return result


def _factory(behavior=None):
    """Builds a 1-arg-compatible factory matching ConnectFourAI's signature."""
    def _build(player_id, model_name=None):
        return _ScriptedAI(player_id, model_name, behavior=behavior)
    return _build


class TestRun:
    async def test_happy_path_returns_move_record_with_cost(self):
        executor = AITurnExecutor(ai_factory=_factory())
        engine = ConnectFour()

        result = await executor.run(engine, player_id=1, model_name="gpt-4o")

        assert result is not None
        assert result.decision.column in engine.get_valid_moves()
        rec = result.move_record
        assert rec["player"] == 1
        assert rec["input_tokens"] == 5
        assert rec["output_tokens"] == 10
        assert rec["reasoning"] == "ok"
        assert rec["is_fallback"] is False
        # Cost should be tokens × pricing / 1M; gpt-4o pricing comes from registry.
        assert isinstance(rec["cost_usd"], (int, float))

    async def test_re_raises_rate_limited_error(self):
        executor = AITurnExecutor(
            ai_factory=_factory(behavior=lambda _: RateLimitedError(30)),
        )
        engine = ConnectFour()

        with pytest.raises(RateLimitedError) as exc_info:
            await executor.run(engine, player_id=1, model_name="gpt-4o")
        assert exc_info.value.snooze_seconds == 30

    async def test_substring_429_in_runtime_error_is_classified_as_rate_limit(self):
        executor = AITurnExecutor(
            ai_factory=_factory(behavior=lambda _: RuntimeError("HTTP 429 throttled")),
        )
        engine = ConnectFour()

        with pytest.raises(RateLimitedError):
            await executor.run(engine, player_id=1, model_name="gpt-4o")

    async def test_other_exceptions_propagate_unwrapped(self):
        executor = AITurnExecutor(
            ai_factory=_factory(behavior=lambda _: ValueError("malformed json")),
        )
        engine = ConnectFour()

        with pytest.raises(ValueError, match="malformed"):
            await executor.run(engine, player_id=1, model_name="gpt-4o")

    async def test_unknown_model_uses_zero_pricing(self):
        executor = AITurnExecutor(ai_factory=_factory())
        engine = ConnectFour()

        result = await executor.run(engine, player_id=2, model_name="unregistered-model")
        assert result.move_record["cost_usd"] == 0.0
