"""Stateless orchestrator for a single AI turn.

Given a board snapshot + the model name to play, this calls into the LLM,
parses the decision, computes the cost, and returns a ``move_record`` ready
to be persisted by the caller.

No DB writes happen here. Rate-limit errors surface as :class:`RateLimitedError`
so the caller can decide how to react (the service layer snoozes the game).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from backend.app.core.logging import get_logger
from backend.app.core.model_registry import registry
from backend.app.engine.ai import ConnectFourAI
from backend.app.engine.rate_limit import RateLimitedError, rate_limit_detector

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from backend.app.engine.game import ConnectFour

logger = get_logger(__name__)


@dataclass(frozen=True)
class AITurnResult:
    """What the executor returns when an AI turn produces a move."""

    move_record: Dict[str, Any]
    decision: Any  # MoveDecision from app.engine.ai
    duration_seconds: float


class AITurnExecutor:
    """Runs one AI turn from a board snapshot. Stateless, no DB I/O.

    The class form makes future extension (e.g. retry policy) clean while
    keeping the call site readable.
    """

    def __init__(self, ai_factory=ConnectFourAI):
        # Injectable for tests; defaults to the real ConnectFourAI.
        self._ai_factory = ai_factory

    async def run(
        self,
        engine_snapshot: "ConnectFour",
        player_id: int,
        model_name: str,
    ) -> Optional[AITurnResult]:
        """Execute one AI turn against ``engine_snapshot``.

        Returns ``None`` if the produced column is illegal on the *snapshot*
        (a soft failure — caller should also re-check under lock). Raises
        :class:`RateLimitedError` if the LLM call hits a provider rate limit.
        Other exceptions propagate.
        """
        start_time = time.time()
        ai = self._ai_factory(player_id=player_id, model_name=model_name)

        try:
            result = await ai.get_move_async(engine_snapshot)
        except RateLimitedError:
            # Already a typed rate-limit; let it bubble.
            raise
        except Exception as e:
            # ai.get_move_async re-raises typed RateLimitedError directly. Anything
            # else gets one more chance via the substring fallback.
            rl = rate_limit_detector.detect(e)
            if rl is not None:
                raise rl from e
            raise

        decision = result["decision"]
        usage = result["usage"] or {"input_tokens": 0, "output_tokens": 0}
        duration = round(time.time() - start_time, 3)

        move_record = self._build_move_record(
            decision=decision,
            usage=usage,
            duration=duration,
            player_id=player_id,
            model_name=model_name,
        )
        return AITurnResult(move_record=move_record, decision=decision, duration_seconds=duration)

    @staticmethod
    def _build_move_record(
        *,
        decision: Any,
        usage: Dict[str, int],
        duration: float,
        player_id: int,
        model_name: str,
    ) -> Dict[str, Any]:
        config = registry.get(model_name)
        pricing = config.pricing if config else {"input": 0, "output": 0}

        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cost_usd = (input_tokens / 1_000_000) * pricing.get("input", 0) + (
            output_tokens / 1_000_000
        ) * pricing.get("output", 0)

        return {
            "player": player_id,
            "column": decision.column,
            "reasoning": getattr(decision, "reasoning", None),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration": duration,
            "is_fallback": getattr(decision, "is_fallback", False),
            "cost_usd": cost_usd,
        }


# Singleton — the executor itself is stateless.
ai_turn_executor = AITurnExecutor()
