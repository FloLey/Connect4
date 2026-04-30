"""Runtime-editable settings (Tier 5 — Settings page).

Holds API keys + tunable overrides in memory and persists changes to the
single-row ``app_settings`` table. The pydantic-settings instance in
``app/core/config.py`` provides defaults; this layer overrides them.

Read paths:
- ``get_api_key("openai")`` — returns the override if set, else ``OPENAI_API_KEY``
  from env, else ``None``.
- The ``settings`` facade in ``app.core.config`` checks ``_overrides`` first
  for tunable fields. Calling ``set_tunable`` writes to that dict.

Write paths funnel through ``set_api_key`` / ``set_tunable`` / ``clear_*``;
all write through to the DB so the next process picks them up.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from backend.app.core.config import (
    _LiveSettings,
    _all_overrides,
    _base,
    _clear_override,
    _set_override,
)
from backend.app.core.logging import get_logger
from backend.app.models.app_settings import AppSettings

logger = get_logger(__name__)


# Provider -> env var fallback. Anything not in here is unknown to the
# settings page.
PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# Tunables exposed to the UI. Single source of truth — used by both the
# /settings endpoint (for form generation) and validation.
EDITABLE_TUNABLES = tuple(sorted(_LiveSettings.EDITABLE))


def _mask_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


class RuntimeSettings:
    """In-memory cache of API keys + tunable overrides.

    Loaded from the ``app_settings`` row at startup; mutated via the API.
    """

    def __init__(self) -> None:
        self._api_keys: dict[str, str] = {}

    # -- API keys -----------------------------------------------------------

    def get_api_key(self, provider: str) -> Optional[str]:
        """Override > env > None."""
        if provider in self._api_keys:
            return self._api_keys[provider]
        env_var = PROVIDER_ENV_VAR.get(provider)
        if env_var:
            return os.getenv(env_var)
        return None

    async def set_api_key(self, db: AsyncSession, provider: str, value: str) -> None:
        if provider not in PROVIDER_ENV_VAR:
            raise ValueError(f"Unknown provider: {provider}")
        self._api_keys[provider] = value
        await self._persist(db)

    async def clear_api_key(self, db: AsyncSession, provider: str) -> None:
        self._api_keys.pop(provider, None)
        await self._persist(db)

    # -- Tunables -----------------------------------------------------------

    async def set_tunable(self, db: AsyncSession, key: str, value: Any) -> None:
        if key not in _LiveSettings.EDITABLE:
            raise ValueError(f"Tunable not editable: {key}")
        # Coerce to the same type as the pydantic default to catch bad input
        # early (e.g. a string posted into an int field).
        default = getattr(_base, key)
        if default is not None and not isinstance(value, type(default)):
            try:
                value = type(default)(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Cannot coerce {value!r} to {type(default).__name__}"
                ) from e
        _set_override(key, value)
        await self._persist(db)

    async def clear_tunable(self, db: AsyncSession, key: str) -> None:
        _clear_override(key)
        await self._persist(db)

    # -- Read-only views ----------------------------------------------------

    def dump_safe(self) -> dict:
        """Shape returned by GET /settings. API keys are masked."""
        api_key_view = {}
        for provider, env_var in PROVIDER_ENV_VAR.items():
            override = self._api_keys.get(provider)
            env_value = os.getenv(env_var)
            effective = override or env_value
            api_key_view[provider] = {
                "set": effective is not None,
                "source": "override" if override else ("env" if env_value else None),
                "preview": _mask_key(effective),
            }

        tunables_view = {}
        overrides = _all_overrides()
        for key in EDITABLE_TUNABLES:
            tunables_view[key] = {
                "value": overrides.get(key, getattr(_base, key)),
                "default": getattr(_base, key),
                "overridden": key in overrides,
            }

        return {"api_keys": api_key_view, "tunables": tunables_view}

    # -- Persistence --------------------------------------------------------

    async def _persist(self, db: AsyncSession) -> None:
        row = (
            await db.execute(select(AppSettings).where(AppSettings.id == 1))
        ).scalar_one_or_none()
        if row is None:
            row = AppSettings(id=1, api_keys={}, tunables={})
            db.add(row)
        row.api_keys = dict(self._api_keys)
        row.tunables = _all_overrides()
        flag_modified(row, "api_keys")
        flag_modified(row, "tunables")
        await db.commit()
        logger.info(
            "runtime_settings_persisted",
            api_key_providers=list(self._api_keys.keys()),
            tunables=list(_all_overrides().keys()),
        )

    async def load_from_db(self, db: AsyncSession) -> None:
        row = (
            await db.execute(select(AppSettings).where(AppSettings.id == 1))
        ).scalar_one_or_none()
        if row is None:
            self._api_keys = {}
            return

        self._api_keys = dict(row.api_keys or {})
        # Replay tunable overrides into the config facade.
        for key, value in (row.tunables or {}).items():
            if key in _LiveSettings.EDITABLE:
                _set_override(key, value)
        logger.info(
            "runtime_settings_loaded",
            api_key_providers=list(self._api_keys.keys()),
            tunables=list((row.tunables or {}).keys()),
        )


runtime_settings = RuntimeSettings()
