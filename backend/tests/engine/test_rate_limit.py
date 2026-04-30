"""Tests for `RateLimitDetector` (provider-typed + substring fallback)."""

import pytest

from backend.app.core.config import settings
from backend.app.engine.rate_limit import (
    RateLimitDetector,
    RateLimitedError,
    rate_limit_detector,
)


# -- Substring fallback ------------------------------------------------------


@pytest.mark.parametrize(
    "msg, expect_detected",
    [
        ("HTTP 429 Too Many Requests", True),
        ("rate_limit exceeded", True),
        ("Rate limit: 10 requests per minute", True),
        ("rate limit exceeded", True),
        ("quota exceeded for model", True),
        ("throttled by API", True),
        ("too many requests", True),
        ("Connection error", False),
        ("Invalid JSON response", False),
        ("Timeout", False),
        ("", False),
    ],
)
def test_substring_fallback_detection(msg, expect_detected):
    result = rate_limit_detector.detect(Exception(msg))
    if expect_detected:
        assert isinstance(result, RateLimitedError)
        assert result.snooze_seconds == settings.rate_limit_snooze_seconds
    else:
        assert result is None


def test_substring_fallback_uses_settings_markers():
    expected = {"429", "rate_limit", "rate limit", "throttled", "quota exceeded", "too many requests"}
    assert expected.issubset(set(settings.rate_limit_markers))


# -- Typed provider exceptions ----------------------------------------------


class _OpenAIRateLimitFake(Exception):
    """Same shape as openai.RateLimitError for the detector to match by class."""


class _AnthropicRateLimitFake(Exception):
    pass


class _GoogleResourceExhaustedFake(Exception):
    pass


class _MistralSDKErrorFake(Exception):
    """Mistral-style generic SDK error; only counts as a rate limit when it
    carries a 429 status."""

    def __init__(self, message="boom", status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _detector_with(name, classes):
    from backend.app.engine.rate_limit import _ProviderCheck

    return RateLimitDetector(provider_checks=[_ProviderCheck(name=name, classes=classes)])


def test_detects_openai_style_exception_by_type():
    det = _detector_with("openai", (_OpenAIRateLimitFake,))
    result = det.detect(_OpenAIRateLimitFake("anything"))
    assert isinstance(result, RateLimitedError)
    assert result.snooze_seconds == settings.rate_limit_snooze_seconds


def test_detects_anthropic_style_exception_by_type():
    det = _detector_with("anthropic", (_AnthropicRateLimitFake,))
    assert isinstance(det.detect(_AnthropicRateLimitFake("nope")), RateLimitedError)


def test_detects_google_resource_exhausted_by_type():
    det = _detector_with("google", (_GoogleResourceExhaustedFake,))
    assert isinstance(det.detect(_GoogleResourceExhaustedFake("quota")), RateLimitedError)


def test_mistral_style_requires_429_status():
    det = _detector_with("mistral", (_MistralSDKErrorFake,))

    # Without a 429 marker, falls through; substring doesn't trigger on "boom".
    assert det.detect(_MistralSDKErrorFake("boom", status_code=500)) is None

    # With status_code=429, recognized as a rate limit.
    rl = det.detect(_MistralSDKErrorFake("limit hit", status_code=429))
    assert isinstance(rl, RateLimitedError)


def test_typed_check_takes_precedence_over_substring():
    """An exception that is both a typed RateLimitError and has unrelated
    text should still detect — exercises the early return."""
    det = _detector_with("openai", (_OpenAIRateLimitFake,))
    result = det.detect(_OpenAIRateLimitFake("totally unrelated text"))
    assert isinstance(result, RateLimitedError)


def test_unknown_exception_returns_none():
    assert rate_limit_detector.detect(ValueError("unrelated")) is None
