"""HTTP tests for the /tournament endpoints.

The GameRunner is replaced with a no-op fake so creating/starting a
tournament doesn't spawn real background tasks.
"""

import pytest
import pytest_asyncio

from backend.app.models.enums import TournamentStatus
from backend.app.services import tournament_service as ts_module


class FakeRunner:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.running = set()

    async def start_game_if_ai_vs_ai(self, game_id, env="prod"):
        self.started.append((game_id, env))
        self.running.add((game_id, env))

    def is_game_running(self, game_id, env="prod"):
        return (game_id, env) in self.running

    async def stop_game(self, game_id, env="prod"):
        self.stopped.append((game_id, env))
        self.running.discard((game_id, env))


@pytest_asyncio.fixture(autouse=True)
def patch_runner(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(ts_module, "game_runner", runner)
    return runner


class TestCreateTournament:
    async def test_create_round_robin_returns_total_matches(self, client):
        response = await client.post(
            "/tournament/create",
            json={"models": ["a", "b", "c"], "rounds": 2, "concurrency": 2},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total_matches"] == 3 * 2 * 2  # n*(n-1)*rounds
        assert body["status"] == TournamentStatus.SETUP

    async def test_create_evaluation_returns_total_matches(self, client):
        response = await client.post(
            "/tournament/create-evaluation",
            json={
                "target_model": "challenger",
                "benchmark_models": ["b1", "b2"],
                "rounds": 2,
                "concurrency": 1,
            },
        )
        body = response.json()
        # 2 benchmarks × 2 orderings × 2 rounds = 8
        assert body["total_matches"] == 8


class TestStartStopPauseResume:
    async def test_start_unknown_returns_404(self, client):
        response = await client.post("/tournament/9999/start")
        assert response.status_code == 404

    async def test_start_then_pause_then_resume(self, client):
        created = (
            await client.post(
                "/tournament/create",
                json={"models": ["a", "b"], "rounds": 1, "concurrency": 1},
            )
        ).json()
        tid = created["id"]

        assert (await client.post(f"/tournament/{tid}/start")).status_code == 200
        current = (await client.get("/tournament/current")).json()
        assert current["id"] == tid

        assert (await client.post(f"/tournament/{tid}/pause")).status_code == 200
        assert (await client.post(f"/tournament/{tid}/resume")).status_code == 200

    async def test_resume_rejects_non_paused(self, client):
        created = (
            await client.post(
                "/tournament/create",
                json={"models": ["a", "b"], "rounds": 1, "concurrency": 1},
            )
        ).json()

        # Setup → cannot resume directly.
        response = await client.post(f"/tournament/{created['id']}/resume")
        assert response.status_code == 400


class TestUpdateConfig:
    async def test_update_concurrency_persists(self, client):
        created = (
            await client.post(
                "/tournament/create",
                json={"models": ["a", "b"], "rounds": 1, "concurrency": 2},
            )
        ).json()

        response = await client.patch(
            f"/tournament/{created['id']}/config",
            json={"concurrency": 7},
        )
        assert response.status_code == 200

        current = (await client.get("/tournament/current")).json()
        assert current["config"]["concurrency"] == 7

    async def test_update_unknown_returns_404(self, client):
        response = await client.patch("/tournament/9999/config", json={"concurrency": 1})
        assert response.status_code == 404


class TestStop:
    async def test_stop_marks_tournament_stopped(self, client):
        created = (
            await client.post(
                "/tournament/create",
                json={"models": ["a", "b"], "rounds": 1, "concurrency": 1},
            )
        ).json()
        await client.post(f"/tournament/{created['id']}/start")
        response = await client.post(f"/tournament/{created['id']}/stop")
        assert response.status_code == 200

        current = (await client.get("/tournament/current")).json()
        assert current["status"] == TournamentStatus.STOPPED


class TestCurrent:
    async def test_returns_null_when_no_tournament(self, client):
        response = await client.get("/tournament/current")
        assert response.status_code == 200
        assert response.json() is None
