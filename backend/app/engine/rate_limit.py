"""Provider-aware rate-limit detection.

Replaces the substring-only check that used to live duplicated in
`app/engine/ai.py` and `app/services/game_service.py`.

Detection order (first match wins):
1. ``openai.RateLimitError``
2. ``anthropic.RateLimitError``
3. ``google.api_core.exceptions.ResourceExhausted``
4. ``mistralai.exceptions.MistralAPIException`` with HTTP 429 (best-effort)
5. Substring fallback against ``settings.rate_limit_markers``

Each provider SDK is imported lazily — missing optional providers are skipped
without breaking detection for the remaining providers or the substring path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.core.config import settings


class RateLimitedError(Exception):
    """Raised when a downstream LLM call hit a rate limit.

    Carries the snooze window in seconds — chosen by the detector based on
    settings, not the provider's Retry-After header (we don't trust those to
    be present or accurate across providers).
    """

    def __init__(self, snooze_seconds: int, original: BaseException | None = None):
        super().__init__(f"Rate limit hit; snoozing {snooze_seconds}s")
        self.snooze_seconds = snooze_seconds
        self.original = original


@dataclass(frozen=True)
class _ProviderCheck:
    name: str
    classes: tuple[type, ...]


def _resolve_provider_classes() -> list[_ProviderCheck]:
    """Lazily resolve each provider's rate-limit exception class.

    Imports happen here (not at module top) so that:
    - missing optional SDKs don't break detection for the remaining providers
    - the test suite doesn't require every provider to be installed
    """
    checks: list[_ProviderCheck] = []

    try:
        from openai import RateLimitError as OpenAIRateLimit  # type: ignore
        checks.append(_ProviderCheck("openai", (OpenAIRateLimit,)))
    except Exception:
        pass

    try:
        from anthropic import RateLimitError as AnthropicRateLimit  # type: ignore
        checks.append(_ProviderCheck("anthropic", (AnthropicRateLimit,)))
    except Exception:
        pass

    try:
        from google.api_core.exceptions import ResourceExhausted  # type: ignore
        checks.append(_ProviderCheck("google", (ResourceExhausted,)))
    except Exception:
        pass

    try:
        # mistralai's exception layout has shifted across releases. Try the
        # common locations and union whatever is found.
        mistral_classes: list[type] = []
        try:
            from mistralai.exceptions import MistralAPIException  # type: ignore
            mistral_classes.append(MistralAPIException)
        except Exception:
            pass
        try:
            from mistralai.models import SDKError  # type: ignore
            mistral_classes.append(SDKError)
        except Exception:
            pass
        if mistral_classes:
            checks.append(_ProviderCheck("mistral", tuple(mistral_classes)))
    except Exception:
        pass

    return checks


def _has_429_signal(exc: BaseException) -> bool:
    """Mistral and other generic HTTP exceptions don't always subclass a
    rate-limit-specific class, so we sniff a status_code/http_status attribute
    in addition to the type check."""
    for attr in ("status_code", "http_status", "status"):
        value = getattr(exc, attr, None)
        if value == 429 or value == "429":
            return True
    return False


class RateLimitDetector:
    def __init__(self, provider_checks: Optional[list[_ProviderCheck]] = None):
        # Resolved once on construction; cheap, but avoid re-importing per call.
        self._provider_checks = (
            provider_checks if provider_checks is not None else _resolve_provider_classes()
        )

    def detect(self, exc: BaseException) -> Optional[RateLimitedError]:
        """Return a ``RateLimitedError`` to raise if ``exc`` looks like a rate
        limit, else ``None``. Never raises itself."""
        # 1–4: typed provider exceptions.
        for check in self._provider_checks:
            if isinstance(exc, check.classes):
                # For Mistral and other generic SDKError-style classes, require
                # an HTTP-429 signal so we don't trip on unrelated SDK errors.
                if check.name == "mistral" and not _has_429_signal(exc):
                    continue
                return RateLimitedError(
                    snooze_seconds=settings.rate_limit_snooze_seconds, original=exc
                )

        # 5: substring fallback.
        err_msg = str(exc).lower()
        if any(marker in err_msg for marker in settings.rate_limit_markers):
            return RateLimitedError(
                snooze_seconds=settings.rate_limit_snooze_seconds, original=exc
            )

        return None


# Singleton — provider class resolution is cached on construction.
rate_limit_detector = RateLimitDetector()
