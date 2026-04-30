"""Application settings facade.

There are two layers of values:

1. **Defaults**: ``_BaseSettings`` is a normal ``pydantic-settings`` model
   loaded from env vars / .env at process start. It defines every field's
   type and default.
2. **Runtime overrides**: a small in-memory dict mutated by
   ``services/runtime_settings.py`` (which itself persists to the
   ``app_settings`` table). When a key is in the override dict, reads see
   the override; otherwise they fall through to the pydantic instance.

External code keeps importing ``settings`` and reading attributes the same
way — e.g. ``settings.elo_k_factor``. The facade hides the lookup. Only the
``runtime_settings`` service should call ``_set_override`` / ``_clear_override``.
"""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class _BaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONNECT4_",
        env_file=".env",
        extra="ignore",
    )

    # ELO
    elo_k_factor: int = 32
    elo_base_rating: float = 1200.0

    # AI
    default_temperature: float = 0.2
    fallback_model: str = "gpt-4o"
    rate_limit_markers: list[str] = [
        "429",
        "rate_limit",
        "rate limit",
        "throttled",
        "quota exceeded",
        "too many requests",
    ]
    rate_limit_snooze_seconds: int = 600

    # Background loops
    cleanup_interval_seconds: int = 900
    cleanup_retry_seconds: int = 60
    cleanup_abandon_age_hours: int = 6
    tournament_watcher_heartbeat_seconds: float = 30.0
    game_runner_pacing_seconds: float = 1.5

    # Database pool
    db_pool_size: int = 50
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # Stats
    stats_active_games_limit: int = 20

    # Admin auth — when unset, /admin/* routes are unauthenticated (dev-friendly).
    # Set CONNECT4_ADMIN_TOKEN in any non-trivial deployment.
    admin_token: str | None = None


# Loaded once at module import. Treat as immutable from outside.
_base = _BaseSettings()

# Runtime overrides are written via services.runtime_settings; never mutate
# this directly from app code.
_overrides: dict[str, Any] = {}


class _LiveSettings:
    """Read-only facade combining ``_overrides`` and the pydantic defaults."""

    # Tunables exposed to the runtime override path. Anything not in this set
    # is read straight from the pydantic instance — keeps DB pool / loop
    # constants out of the user-editable surface.
    EDITABLE: frozenset[str] = frozenset({
        "fallback_model",
        "default_temperature",
        "elo_k_factor",
        "rate_limit_snooze_seconds",
        "game_runner_pacing_seconds",
    })

    def __getattr__(self, name: str) -> Any:
        if name in _overrides:
            return _overrides[name]
        return getattr(_base, name)


settings = _LiveSettings()


def _set_override(key: str, value: Any) -> None:
    """Used by runtime_settings.set_tunable; do not call from app code."""
    _overrides[key] = value


def _clear_override(key: str) -> None:
    _overrides.pop(key, None)


def _all_overrides() -> dict[str, Any]:
    return dict(_overrides)
