"""Tournament scheduling tests.

Exercises round-robin and evaluation generation, lifecycle transitions, and
the tick-driven concurrency / orphan-recovery / cooldown logic. The
GameRunner is patched out so no real background tasks are spawned.
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func
from sqlalchemy.future import select

from backend.app.models.enums import GameStatus, TournamentStatus
from backend.app.models.game_model import Game
from backend.app.models.tournament_model import Tournament
from backend.app.services import tournament_service as ts_module
from backend.app.services.tournament_service import tournament_service


class FakeRunner:
    """In-memory replacement for game_runner used across these tests."""

    def __init__(self):
        self.started: list[tuple[int, str]] = []
        self.stopped: list[tuple[int, str]] = []
        self.running: set[tuple[int, str]] = set()

    async def start_game_if_ai_vs_ai(self, game_id, env="prod"):
        self.started.append((game_id, env))
        self.running.add((game_id, env))

    def is_game_running(self, game_id, env="prod"):
        return (game_id, env) in self.running

    async def stop_game(self, game_id, env="prod"):
        self.stopped.append((game_id, env))
        self.running.discard((game_id, env))


@pytest_asyncio.fixture
def fake_runner(monkeypatch):
    runner = FakeRunner()
    monkeypatch.setattr(ts_module, "game_runner", runner)
    return runner


class TestCreateTournament:
    async def test_round_robin_generates_n_times_n_minus_one_games_per_round(
        self, test_db
    ):
        models = ["a", "b", "c"]
        rounds = 2

        tournament = await tournament_service.create_tournament(
            test_db, models=models, rounds=rounds, concurrency=2
        )

        # n*(n-1) ordered pairs per round = 6 per round.
        expected = len(models) * (len(models) - 1) * rounds
        assert tournament.total_matches == expected

        count = await test_db.execute(
            select(func.count(Game.id)).where(Game.tournament_id == tournament.id)
        )
        assert count.scalar() == expected

        # Every game should be PENDING and have a round_number in [1, rounds].
        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id)
            )
        ).scalars().all()
        assert all(g.status == GameStatus.PENDING for g in games)
        assert all(1 <= g.round_number <= rounds for g in games)

    async def test_evaluation_generates_two_games_per_benchmark_per_round(
        self, test_db
    ):
        target = "challenger"
        benchmarks = ["b1", "b2"]
        rounds = 3

        tournament = await tournament_service.create_evaluation_tournament(
            test_db, target=target, benchmarks=benchmarks, rounds=rounds, concurrency=2
        )

        expected = len(benchmarks) * 2 * rounds
        assert tournament.total_matches == expected

        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id)
            )
        ).scalars().all()

        # Each (target, benchmark) pair appears exactly `rounds` times in each ordering.
        for benchmark in benchmarks:
            target_first = sum(
                1 for g in games if g.player_1_type == target and g.player_2_type == benchmark
            )
            opp_first = sum(
                1 for g in games if g.player_1_type == benchmark and g.player_2_type == target
            )
            assert target_first == rounds
            assert opp_first == rounds


class TestLifecycle:
    async def test_start_transitions_to_in_progress(self, test_db):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=1
        )

        ok = await tournament_service.start_tournament(test_db, tournament.id)

        await test_db.refresh(tournament)
        assert ok is True
        assert tournament.status == TournamentStatus.IN_PROGRESS

    async def test_stop_returns_false_for_missing_tournament(self, test_db):
        ok = await tournament_service.stop_tournament(test_db, tournament_id=9999)
        assert ok is False

    async def test_pause_stops_in_progress_games(self, test_db, fake_runner):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=2
        )
        # Mark two games IN_PROGRESS to simulate active state.
        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id).limit(2)
            )
        ).scalars().all()
        for g in games:
            g.status = GameStatus.IN_PROGRESS
        await test_db.commit()

        ok = await tournament_service.pause_tournament(test_db, tournament.id, env="test")

        await test_db.refresh(tournament)
        assert ok is True
        assert tournament.status == TournamentStatus.PAUSED
        assert sorted(fake_runner.stopped) == sorted([(g.id, "test") for g in games])

    async def test_update_concurrency_persists_to_jsonb(self, test_db):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=2
        )

        ok = await tournament_service.update_concurrency(test_db, tournament.id, 5)

        await test_db.refresh(tournament)
        assert ok is True
        assert tournament.config["concurrency"] == 5


class TestTick:
    async def test_completes_tournament_when_no_unfinished_games(
        self, test_db, fake_runner
    ):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=1
        )
        tournament.status = TournamentStatus.IN_PROGRESS
        # Mark every game COMPLETED so unfinished == 0.
        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id)
            )
        ).scalars().all()
        for g in games:
            g.status = GameStatus.COMPLETED
        await test_db.commit()

        await tournament_service.tick(test_db, env="test")

        await test_db.refresh(tournament)
        assert tournament.status == TournamentStatus.COMPLETED

    async def test_launches_pending_games_up_to_concurrency_limit(
        self, test_db, fake_runner
    ):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b", "c"], rounds=1, concurrency=2
        )
        tournament.status = TournamentStatus.IN_PROGRESS
        await test_db.commit()

        await tournament_service.tick(test_db, env="test")

        # All started games are tracked by fake_runner.
        assert len(fake_runner.started) == 2

        # Those games should now be IN_PROGRESS in the DB.
        in_progress = (
            await test_db.execute(
                select(func.count(Game.id)).where(
                    Game.tournament_id == tournament.id,
                    Game.status == GameStatus.IN_PROGRESS,
                )
            )
        ).scalar()
        assert in_progress == 2

    async def test_resumes_orphaned_in_progress_games_first(
        self, test_db, fake_runner
    ):
        """A game already IN_PROGRESS in DB but not in the runner is an orphan; it
        should consume a slot before any new PENDING game starts."""
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=1
        )
        tournament.status = TournamentStatus.IN_PROGRESS
        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id)
            )
        ).scalars().all()
        orphan = games[0]
        orphan.status = GameStatus.IN_PROGRESS
        await test_db.commit()

        await tournament_service.tick(test_db, env="test")

        # Orphan revived first.
        assert fake_runner.started[0][0] == orphan.id

    async def test_paused_game_with_expired_retry_after_is_resumed(
        self, test_db, fake_runner
    ):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=1
        )
        tournament.status = TournamentStatus.IN_PROGRESS
        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id)
            )
        ).scalars().all()

        # First game PAUSED with expired retry_after; second still PENDING.
        paused = games[0]
        paused.status = GameStatus.PAUSED
        paused.retry_after = datetime.now(timezone.utc) - timedelta(minutes=1)
        await test_db.commit()

        await tournament_service.tick(test_db, env="test")

        await test_db.refresh(paused)
        assert paused.status == GameStatus.IN_PROGRESS
        assert paused.retry_after is None
        assert (paused.id, "test") in fake_runner.running

    async def test_paused_game_with_future_retry_after_is_skipped(
        self, test_db, fake_runner
    ):
        tournament = await tournament_service.create_tournament(
            test_db, models=["a", "b"], rounds=1, concurrency=1
        )
        tournament.status = TournamentStatus.IN_PROGRESS
        games = (
            await test_db.execute(
                select(Game).where(Game.tournament_id == tournament.id)
            )
        ).scalars().all()
        for g in games:
            g.status = GameStatus.PAUSED
            g.retry_after = datetime.now(timezone.utc) + timedelta(minutes=5)
        await test_db.commit()

        await tournament_service.tick(test_db, env="test")

        # Nothing eligible to start.
        assert fake_runner.started == []
