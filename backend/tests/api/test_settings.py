"""HTTP tests for /settings."""

import pytest
from sqlalchemy.future import select

from backend.app.core.config import _all_overrides, _clear_override, settings
from backend.app.models.app_settings import AppSettings
from backend.app.services.runtime_settings import runtime_settings


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    runtime_settings._api_keys.clear()
    for key in list(_all_overrides().keys()):
        _clear_override(key)
    yield
    runtime_settings._api_keys.clear()
    for key in list(_all_overrides().keys()):
        _clear_override(key)


class TestGet:
    async def test_get_returns_providers_and_editable_tunables(self, client):
        response = await client.get("/settings")
        assert response.status_code == 200
        body = response.json()
        # Five providers; six tunables.
        assert sorted(body["providers"]) == [
            "anthropic", "deepseek", "google", "mistral", "openai",
        ]
        assert "elo_k_factor" in body["editable_tunables"]
        assert all(
            k in body["api_keys"]
            for k in ("openai", "anthropic", "google", "deepseek", "mistral")
        )

    async def test_keys_are_masked_in_response(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-tail-1234")
        response = await client.get("/settings")
        body = response.json()
        assert body["api_keys"]["openai"]["set"] is True
        assert body["api_keys"]["openai"]["preview"] == "****1234"
        # Full secret never crosses the wire.
        assert "sk-secret-tail-1234" not in response.text


class TestPatch:
    async def test_patch_sets_api_key_and_tunable(self, client, test_db):
        response = await client.patch(
            "/settings",
            json={
                "api_keys": {"openai": "live-key"},
                "tunables": {"elo_k_factor": 7},
            },
        )
        assert response.status_code == 200
        # Persisted.
        row = (await test_db.execute(select(AppSettings).where(AppSettings.id == 1))).scalar_one()
        assert row.api_keys == {"openai": "live-key"}
        assert row.tunables == {"elo_k_factor": 7}
        assert settings.elo_k_factor == 7

    async def test_patch_with_empty_string_clears_api_key(self, client, test_db):
        await client.patch("/settings", json={"api_keys": {"openai": "live"}})
        assert runtime_settings.get_api_key("openai") == "live"

        await client.patch("/settings", json={"api_keys": {"openai": ""}})
        assert "openai" not in runtime_settings._api_keys

    async def test_patch_unknown_provider_returns_400(self, client):
        response = await client.patch(
            "/settings", json={"api_keys": {"banana": "x"}}
        )
        assert response.status_code == 400

    async def test_patch_uneditable_tunable_returns_400(self, client):
        response = await client.patch(
            "/settings", json={"tunables": {"db_pool_size": 1}}
        )
        assert response.status_code == 400

    async def test_patch_extra_top_level_fields_rejected(self, client):
        response = await client.patch(
            "/settings", json={"api_keys": {}, "tunables": {}, "extra": True}
        )
        assert response.status_code == 422


class TestDelete:
    async def test_delete_api_key(self, client):
        await client.patch("/settings", json={"api_keys": {"openai": "x"}})
        response = await client.delete("/settings/api-keys/openai")
        assert response.status_code == 200
        assert "openai" not in runtime_settings._api_keys

    async def test_delete_unknown_provider_returns_400(self, client):
        response = await client.delete("/settings/api-keys/wat")
        assert response.status_code == 400

    async def test_delete_tunable(self, client):
        await client.patch("/settings", json={"tunables": {"elo_k_factor": 99}})
        assert settings.elo_k_factor == 99

        response = await client.delete("/settings/tunables/elo_k_factor")
        assert response.status_code == 200
        # Reverts to default.
        from backend.app.core.config import _base
        assert settings.elo_k_factor == _base.elo_k_factor


class TestAdminGate:
    async def test_wrong_token_returns_401(self, client, monkeypatch):
        monkeypatch.setattr(settings, "admin_token", "secret")
        response = await client.get("/settings")
        assert response.status_code == 401

    async def test_correct_token_passes(self, client, monkeypatch):
        monkeypatch.setattr(settings, "admin_token", "secret")
        response = await client.get(
            "/settings", headers={"X-Admin-Token": "secret"}
        )
        assert response.status_code == 200
