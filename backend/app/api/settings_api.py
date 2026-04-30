"""HTTP surface for runtime-editable settings.

GET  /settings                       — current state (API keys masked)
PATCH /settings                      — bulk partial update
DELETE /settings/api-keys/{provider} — clear one API key (revert to env)
DELETE /settings/tunables/{key}      — clear one tunable (revert to default)

All routes are gated by ``require_admin``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.admin import require_admin
from backend.app.core.database import get_db
from backend.app.services.runtime_settings import (
    EDITABLE_TUNABLES,
    PROVIDER_ENV_VAR,
    runtime_settings,
)


router = APIRouter(dependencies=[Depends(require_admin)])


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_keys: dict[str, str] = Field(default_factory=dict)
    tunables: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def get_settings():
    """Current runtime settings. API key values are masked to last-4 preview."""
    return {
        **runtime_settings.dump_safe(),
        "providers": list(PROVIDER_ENV_VAR.keys()),
        "editable_tunables": list(EDITABLE_TUNABLES),
    }


@router.patch("")
async def patch_settings(
    payload: SettingsPatch,
    db: AsyncSession = Depends(get_db),
):
    """Apply a partial update of API keys and/or tunables.

    Empty-string values are treated as 'clear this entry' so the same form
    submission can both set and unset things.
    """
    for provider, value in payload.api_keys.items():
        if provider not in PROVIDER_ENV_VAR:
            raise HTTPException(400, f"Unknown provider: {provider}")
        if value == "":
            await runtime_settings.clear_api_key(db, provider)
        else:
            await runtime_settings.set_api_key(db, provider, value)

    for key, value in payload.tunables.items():
        if key not in EDITABLE_TUNABLES:
            raise HTTPException(400, f"Tunable not editable: {key}")
        try:
            await runtime_settings.set_tunable(db, key, value)
        except ValueError as e:
            raise HTTPException(400, str(e))

    return runtime_settings.dump_safe()


@router.delete("/api-keys/{provider}")
async def clear_api_key(provider: str, db: AsyncSession = Depends(get_db)):
    if provider not in PROVIDER_ENV_VAR:
        raise HTTPException(400, f"Unknown provider: {provider}")
    await runtime_settings.clear_api_key(db, provider)
    return runtime_settings.dump_safe()


@router.delete("/tunables/{key}")
async def clear_tunable(key: str, db: AsyncSession = Depends(get_db)):
    if key not in EDITABLE_TUNABLES:
        raise HTTPException(400, f"Tunable not editable: {key}")
    await runtime_settings.clear_tunable(db, key)
    return runtime_settings.dump_safe()
