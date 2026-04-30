"""Tests for the runtime_settings service.

Each test starts from a clean state via `fresh_runtime` so test order
doesn't leak overrides into siblings.
"""

import pytest
from sqlalchemy.future import select

from backend.app.core.config import _all_overrides, _clear_override, _base, settings
from backend.app.models.app_settings import AppSettings
from backend.app.services.runtime_settings import (
    EDITABLE_TUNABLES,
    PROVIDER_ENV_VAR,
    RuntimeSettings,
    runtime_settings,
)


@pytest.fixture(autouse=True)
def _reset_overrides():
    # Clear all in-memory state before AND after each test.
    runtime_settings._api_keys.clear()
    for key in list(_all_overrides().keys()):
        _clear_override(key)
    yield
    runtime_settings._api_keys.clear()
    for key in list(_all_overrides().keys()):
        _clear_override(key)


# -- API keys ----------------------------------------------------------------


def test_get_api_key_returns_none_when_no_override_or_env(monkeypatch):
    for env_var in PROVIDER_ENV_VAR.values():
        monkeypatch.delenv(env_var, raising=False)
    assert runtime_settings.get_api_key("openai") is None


def test_get_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert runtime_settings.get_api_key("openai") == "env-key"


async def test_set_api_key_overrides_env_and_persists(test_db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    await runtime_settings.set_api_key(test_db, "openai", "live-key")

    assert runtime_settings.get_api_key("openai") == "live-key"
    row = (await test_db.execute(select(AppSettings).where(AppSettings.id == 1))).scalar_one()
    assert row.api_keys == {"openai": "live-key"}


async def test_clear_api_key_falls_back_to_env(test_db, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    await runtime_settings.set_api_key(test_db, "openai", "live-key")
    assert runtime_settings.get_api_key("openai") == "live-key"

    await runtime_settings.clear_api_key(test_db, "openai")
    assert runtime_settings.get_api_key("openai") == "env-key"


async def test_set_api_key_unknown_provider_raises(test_db):
    with pytest.raises(ValueError, match="Unknown provider"):
        await runtime_settings.set_api_key(test_db, "wat", "x")


# -- Tunables ----------------------------------------------------------------


async def test_set_tunable_overrides_settings_facade(test_db):
    original = settings.elo_k_factor
    new_value = original + 4

    await runtime_settings.set_tunable(test_db, "elo_k_factor", new_value)
    assert settings.elo_k_factor == new_value


async def test_set_tunable_coerces_str_to_int(test_db):
    await runtime_settings.set_tunable(test_db, "elo_k_factor", "24")
    assert settings.elo_k_factor == 24
    assert isinstance(settings.elo_k_factor, int)


async def test_set_tunable_uneditable_raises(test_db):
    with pytest.raises(ValueError, match="not editable"):
        await runtime_settings.set_tunable(test_db, "db_pool_size", 99)


async def test_set_tunable_bad_value_raises(test_db):
    with pytest.raises(ValueError, match="Cannot coerce"):
        await runtime_settings.set_tunable(test_db, "elo_k_factor", "not-an-int")


async def test_clear_tunable_reverts_to_default(test_db):
    default = _base.elo_k_factor
    await runtime_settings.set_tunable(test_db, "elo_k_factor", default + 100)
    assert settings.elo_k_factor != default

    await runtime_settings.clear_tunable(test_db, "elo_k_factor")
    assert settings.elo_k_factor == default


# -- dump_safe ---------------------------------------------------------------


def test_dump_safe_masks_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-long-secret-XYZ9")
    for env_var in PROVIDER_ENV_VAR.values():
        if env_var != "OPENAI_API_KEY":
            monkeypatch.delenv(env_var, raising=False)

    dump = runtime_settings.dump_safe()
    assert dump["api_keys"]["openai"]["preview"] == "****XYZ9"
    assert dump["api_keys"]["openai"]["set"] is True
    assert dump["api_keys"]["openai"]["source"] == "env"
    assert dump["api_keys"]["anthropic"]["set"] is False
    assert dump["api_keys"]["anthropic"]["preview"] is None


def test_dump_safe_short_keys_collapse_to_stars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ab")
    dump = runtime_settings.dump_safe()
    assert dump["api_keys"]["openai"]["preview"] == "****"


def test_dump_safe_marks_overridden_tunables():
    # Without override.
    dump = runtime_settings.dump_safe()
    assert dump["tunables"]["elo_k_factor"]["overridden"] is False

    # Direct manipulation of the override dict (mimic set_tunable's effect
    # without touching the DB — simpler than awaiting in a sync test).
    from backend.app.core.config import _set_override
    _set_override("elo_k_factor", 99)
    try:
        dump = runtime_settings.dump_safe()
        assert dump["tunables"]["elo_k_factor"]["overridden"] is True
        assert dump["tunables"]["elo_k_factor"]["value"] == 99
    finally:
        from backend.app.core.config import _clear_override
        _clear_override("elo_k_factor")


# -- load_from_db -----------------------------------------------------------


async def test_load_from_db_restores_keys_and_tunables(test_db):
    test_db.add(
        AppSettings(
            id=1,
            api_keys={"openai": "from-db", "google": "google-key"},
            tunables={"elo_k_factor": 11, "default_temperature": 0.7},
        )
    )
    await test_db.commit()

    fresh = RuntimeSettings()
    await fresh.load_from_db(test_db)

    assert fresh.get_api_key("openai") == "from-db"
    assert fresh.get_api_key("google") == "google-key"
    assert settings.elo_k_factor == 11
    assert settings.default_temperature == 0.7


async def test_load_from_db_skips_uneditable_tunables(test_db):
    test_db.add(
        AppSettings(
            id=1,
            api_keys={},
            tunables={"db_pool_size": 1, "elo_k_factor": 5},
        )
    )
    await test_db.commit()

    fresh = RuntimeSettings()
    await fresh.load_from_db(test_db)

    # db_pool_size shouldn't be applied (not in EDITABLE).
    assert settings.db_pool_size == _base.db_pool_size
    # elo_k_factor IS editable.
    assert settings.elo_k_factor == 5


def test_editable_tunables_match_facade():
    """The exposed list and the facade allowlist must agree, otherwise the
    UI offers fields it can't actually mutate."""
    from backend.app.core.config import _LiveSettings

    assert set(EDITABLE_TUNABLES) == _LiveSettings.EDITABLE
