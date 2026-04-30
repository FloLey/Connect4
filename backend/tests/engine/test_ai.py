"""Tests for ConnectFourAI's get_move_async paths.

The LLM call is faked end-to-end: ``ai.get_llm`` is monkeypatched so __init__
doesn't try to construct a real provider, and the resulting ``chain.ainvoke``
is replaced per-test with a callable that returns / raises whatever shape the
test needs.
"""

from types import SimpleNamespace

import pytest
from langchain_core.runnables import Runnable

from backend.app.engine import ai as ai_module
from backend.app.engine.ai import ConnectFourAI, MoveDecision
from backend.app.engine.game import ConnectFour
from backend.app.engine.rate_limit import RateLimitedError


class _FakeStructuredLLM(Runnable):
    """Stand-in for ``llm.with_structured_output(...)``.

    ``ConnectFourAI.__init__`` does ``prompt | self.structured_llm``; LCEL
    type-checks that, so we must look like a real Runnable. Tests overwrite
    ``ai.chain`` right after construction, so ``invoke`` never actually fires.
    """

    def invoke(self, _input, _config=None, **_kwargs):  # pragma: no cover
        return None


class _FakeLLM:
    def with_structured_output(self, *_args, **_kwargs):
        return _FakeStructuredLLM()


@pytest.fixture
def stub_get_llm(monkeypatch):
    """Replace ``ai.get_llm`` so __init__ doesn't hit real providers."""
    monkeypatch.setattr(ai_module, "get_llm", lambda *a, **kw: _FakeLLM())


def _make_ai(stub_get_llm, chain_ainvoke):
    """Construct a ConnectFourAI and inject a custom ``chain.ainvoke``.

    ``chain_ainvoke`` is an ``async def`` that takes the prompt-input dict and
    returns / raises whatever the test needs.
    """
    ai = ConnectFourAI(player_id=1, model_name="gpt-4o")

    class _Chain:
        async def ainvoke(self, inputs):
            return await chain_ainvoke(inputs)

    ai.chain = _Chain()
    return ai


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_returns_decision_and_usage(stub_get_llm):
    async def chain_call(_inputs):
        return {
            "parsed": MoveDecision(reasoning="ok", column=3),
            "raw": SimpleNamespace(
                usage_metadata={"input_tokens": 12, "output_tokens": 7}
            ),
        }

    ai = _make_ai(stub_get_llm, chain_call)
    result = await ai.get_move_async(ConnectFour())

    assert result["decision"].column == 3
    assert result["decision"].reasoning == "ok"
    assert result["decision"].is_fallback is False
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 7}


async def test_missing_usage_metadata_defaults_to_zeros(stub_get_llm):
    async def chain_call(_inputs):
        return {
            "parsed": MoveDecision(reasoning="ok", column=2),
            "raw": SimpleNamespace(),  # no usage_metadata attr
        }

    ai = _make_ai(stub_get_llm, chain_call)
    result = await ai.get_move_async(ConnectFour())
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


# ---------------------------------------------------------------------------
# Invalid column → corrected to a random valid one
# ---------------------------------------------------------------------------


async def test_invalid_column_is_corrected_to_a_valid_one(stub_get_llm):
    async def chain_call(_inputs):
        return {
            "parsed": MoveDecision(reasoning="bad pick", column=42),
            "raw": SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1}),
        }

    ai = _make_ai(stub_get_llm, chain_call)
    engine = ConnectFour()
    valid = engine.get_valid_moves()

    result = await ai.get_move_async(engine)
    decision = result["decision"]

    assert decision.column in valid
    assert "corrected" in decision.reasoning


# ---------------------------------------------------------------------------
# Decision missing fields / empty → fallback random move with is_fallback=True
# ---------------------------------------------------------------------------


async def test_empty_decision_falls_back_to_random_with_is_fallback_true(stub_get_llm):
    async def chain_call(_inputs):
        return {"parsed": None, "raw": SimpleNamespace()}

    ai = _make_ai(stub_get_llm, chain_call)
    engine = ConnectFour()
    valid = engine.get_valid_moves()

    result = await ai.get_move_async(engine)
    assert result["decision"].is_fallback is True
    assert result["decision"].column in valid
    assert "SYSTEM ERROR" in result["decision"].reasoning


async def test_non_int_column_falls_back(stub_get_llm):
    async def chain_call(_inputs):
        return {
            "parsed": MoveDecision.model_construct(reasoning="x", column="three"),
            "raw": SimpleNamespace(),
        }

    ai = _make_ai(stub_get_llm, chain_call)
    result = await ai.get_move_async(ConnectFour())
    assert result["decision"].is_fallback is True


async def test_decision_missing_attributes_falls_back(stub_get_llm):
    async def chain_call(_inputs):
        # Object without `column` / `reasoning` attributes — simulates a model
        # returning the wrong shape entirely.
        return {"parsed": SimpleNamespace(), "raw": SimpleNamespace()}

    ai = _make_ai(stub_get_llm, chain_call)
    result = await ai.get_move_async(ConnectFour())
    assert result["decision"].is_fallback is True


async def test_chain_raises_arbitrary_error_falls_back(stub_get_llm):
    async def chain_call(_inputs):
        raise ValueError("malformed JSON")

    ai = _make_ai(stub_get_llm, chain_call)
    result = await ai.get_move_async(ConnectFour())
    assert result["decision"].is_fallback is True
    assert "malformed" in result["decision"].reasoning


# ---------------------------------------------------------------------------
# Rate limit → typed RateLimitedError, NOT a fallback
# ---------------------------------------------------------------------------


async def test_substring_429_re_raises_as_rate_limited(stub_get_llm):
    async def chain_call(_inputs):
        raise RuntimeError("HTTP 429 throttled")

    ai = _make_ai(stub_get_llm, chain_call)
    with pytest.raises(RateLimitedError):
        await ai.get_move_async(ConnectFour())


async def test_typed_rate_limit_re_raises_unwrapped(stub_get_llm):
    typed = RateLimitedError(snooze_seconds=42)

    async def chain_call(_inputs):
        raise typed

    ai = _make_ai(stub_get_llm, chain_call)
    with pytest.raises(RateLimitedError) as exc_info:
        await ai.get_move_async(ConnectFour())
    # Detector wraps it as a fresh RateLimitedError with the configured snooze.
    assert exc_info.value.snooze_seconds  # always positive


# ---------------------------------------------------------------------------
# __init__ failure → fallback model
# ---------------------------------------------------------------------------


def test_init_failure_falls_back_to_settings_fallback_model(monkeypatch):
    calls = []

    def get_llm_with_first_failure(model_name, *args, **kwargs):
        calls.append(model_name)
        if model_name == "broken-model":
            raise RuntimeError("provider unreachable")
        return _FakeLLM()

    monkeypatch.setattr(ai_module, "get_llm", get_llm_with_first_failure)

    ai = ConnectFourAI(player_id=2, model_name="broken-model")
    # First call uses the requested model, second uses the configured fallback.
    assert calls == ["broken-model", ai_module.settings.fallback_model]
    assert ai.chain is not None
