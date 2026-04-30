"""Tests for the GameRunner background loop.

The DB session is faked to a no-op context manager and ``asyncio.sleep`` is
short-circuited so the loop runs at full speed. We drive ``step_ai_turn`` with
a scripted sequence of ``GameState`` snapshots / sentinels and assert that the
loop exits on each terminal condition.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from backend.app.models.enums import GameStatus
from backend.app.services import game_runner as runner_module
from backend.app.services.game_runner import GameRunner


@pytest.fixture(autouse=True)
def fast_loop(monkeypatch):
    """Make asyncio.sleep instant so the runner loop doesn't pause."""
    async def _instant(_seconds):
        return None

    monkeypatch.setattr(runner_module.asyncio, "sleep", _instant)


@pytest.fixture
def fake_session(monkeypatch):
    """Replace get_session_maker with one that yields a no-op session."""

    class _NoOpSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def _maker(_env):
        def factory():
            return _NoOpSession()

        return factory

    monkeypatch.setattr(runner_module, "get_session_maker", _maker)


@pytest.fixture
def fake_broadcast(monkeypatch):
    """Replace the WebSocket manager's broadcast with a recording fake.

    Imported lazily inside _game_loop so we patch via sys.modules.
    """
    sent = []

    class _Manager:
        async def broadcast(self, game_id, message):
            sent.append((game_id, message))

    # The runner does `from backend.app.api.websocket_manager import manager`
    # *inside* _game_loop. Patch the module attribute so the late import sees us.
    import backend.app.api.websocket_manager as ws_module
    monkeypatch.setattr(ws_module, "manager", _Manager())
    return sent


def _state(*, winner=None, is_draw=False, status=GameStatus.IN_PROGRESS):
    return SimpleNamespace(
        board=[[0] * 7 for _ in range(6)],
        current_turn=1,
        winner=winner,
        is_draw=is_draw,
        status=status,
        last_move=None,
    )


def _scripted_step_ai_turn(monkeypatch, sequence):
    """Patch game_service.step_ai_turn to yield a fixed sequence of return values.

    Each call pops one value off; if it's an Exception it's raised, otherwise
    returned.
    """
    iterator = iter(sequence)

    async def _step(_db, _game_id):
        value = next(iterator)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(runner_module.game_service, "step_ai_turn", _step)


# ---------------------------------------------------------------------------


async def test_loop_terminates_on_winner(monkeypatch, fake_session, fake_broadcast):
    _scripted_step_ai_turn(monkeypatch, [_state(winner=1)])

    runner = GameRunner()
    await runner._game_loop(game_id=1, env="test")

    assert fake_broadcast == [
        (1, _broadcast_payload(_state(winner=1))),
    ]


async def test_loop_terminates_on_draw(monkeypatch, fake_session, fake_broadcast):
    _scripted_step_ai_turn(monkeypatch, [_state(is_draw=True)])

    runner = GameRunner()
    await runner._game_loop(game_id=2, env="test")

    assert len(fake_broadcast) == 1


async def test_loop_terminates_on_none_returned(monkeypatch, fake_session, fake_broadcast):
    """step_ai_turn returns None when the game has been snoozed for rate-limit."""
    _scripted_step_ai_turn(monkeypatch, [None])

    runner = GameRunner()
    await runner._game_loop(game_id=3, env="test")

    # No broadcast at all — runner exits before hitting the broadcast call.
    assert fake_broadcast == []


async def test_loop_terminates_when_status_changes(
    monkeypatch, fake_session, fake_broadcast
):
    """A PAUSED state from outside (e.g. tournament pause) should end the loop."""
    _scripted_step_ai_turn(monkeypatch, [_state(status=GameStatus.PAUSED)])

    runner = GameRunner()
    await runner._game_loop(game_id=4, env="test")

    assert len(fake_broadcast) == 1


async def test_loop_handles_multiple_steps_then_winner(
    monkeypatch, fake_session, fake_broadcast
):
    _scripted_step_ai_turn(
        monkeypatch,
        [_state(), _state(), _state(winner=2)],
    )

    runner = GameRunner()
    await runner._game_loop(game_id=5, env="test")

    # Three broadcasts — one per step.
    assert len(fake_broadcast) == 3


async def test_loop_swallows_exceptions(monkeypatch, fake_session, fake_broadcast):
    """An unexpected exception logs and exits the loop without re-raising."""
    _scripted_step_ai_turn(monkeypatch, [RuntimeError("boom")])

    runner = GameRunner()
    # No raise — would crash the test if it did.
    await runner._game_loop(game_id=6, env="test")
    assert fake_broadcast == []


async def test_loop_propagates_cancellation(
    monkeypatch, fake_session, fake_broadcast
):
    """asyncio.CancelledError must propagate so the surrounding task cleans up."""
    import asyncio as _asyncio

    _scripted_step_ai_turn(monkeypatch, [_asyncio.CancelledError()])

    runner = GameRunner()
    with pytest.raises(_asyncio.CancelledError):
        await runner._game_loop(game_id=7, env="test")


# ---------------------------------------------------------------------------
# Bookkeeping helpers
# ---------------------------------------------------------------------------


def test_make_key_format():
    runner = GameRunner()
    assert runner._make_key("prod", 12) == "prod_12"


def test_is_game_running_reflects_running_tasks():
    runner = GameRunner()
    assert runner.is_game_running(99, "prod") is False
    runner.running_tasks["prod_99"] = object()
    assert runner.is_game_running(99, "prod") is True


async def test_stop_game_cancels_and_removes(monkeypatch):
    runner = GameRunner()

    cancelled = {"called": False}

    class _Task:
        def cancel(self):
            cancelled["called"] = True

    runner.running_tasks["test_50"] = _Task()
    await runner.stop_game(50, "test")

    assert cancelled["called"] is True
    assert "test_50" not in runner.running_tasks


# ---------------------------------------------------------------------------
# Helpers used by assertions above
# ---------------------------------------------------------------------------


def _broadcast_payload(state):
    return {
        "type": "UPDATE",
        "board": state.board,
        "currentTurn": state.current_turn,
        "winner": state.winner,
        "status": state.status,
        "lastMove": state.last_move,
    }
