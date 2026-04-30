"""WebSocket integration tests via FastAPI TestClient.

The tricky bit: TestClient's WebSocket support runs the app inside an internal
event loop *and* triggers the app's lifespan when the ``with`` block enters.
Our real lifespan kicks off prod/test DB watcher coroutines that interfere
with the test loop, so we replace it with a no-op for these tests via a fixture.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.app.core.database import Base, engines, session_makers
from backend.app.main import app
from backend.app.models.enums import PlayerType
from backend.app.models.game_model import Game


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


@pytest.fixture
def ws_client(monkeypatch):
    """TestClient with the real lifespan replaced by a no-op for the
    duration of the test, so spawning DB watchers doesn't block the loop.

    Disposes the SQLAlchemy connection pool on teardown so the next async
    test's loop gets fresh asyncpg connections.
    """
    monkeypatch.setattr(app.router, "lifespan_context", _noop_lifespan)
    _truncate_sync()

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    # Dispose engines first — TestClient created connections in its own loop;
    # _truncate_sync uses asyncio.run() with a fresh loop and would otherwise
    # try to reuse those defunct connections.
    async def _dispose():
        await engines["test"].dispose()
        await engines["prod"].dispose()
    asyncio.run(_dispose())
    _truncate_sync()


def _truncate_sync():
    async def _run():
        async with engines["test"].begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                await conn.execute(
                    text(f'TRUNCATE TABLE "{table.name}" RESTART IDENTITY CASCADE')
                )
        await engines["test"].dispose()
        await engines["prod"].dispose()

    asyncio.run(_run())


def _create_game(client, p1, p2):
    response = client.post(
        "/games", json={"player_1": p1, "player_2": p2}, headers={"x-db-env": "test"}
    )
    assert response.status_code == 200
    return response.json()


def test_websocket_emits_initial_update_for_empty_game(ws_client):
    game = _create_game(ws_client, PlayerType.HUMAN, PlayerType.HUMAN)
    token = game["player_1_token"]

    with ws_client.websocket_connect(
        f"/games/{game['id']}/ws?env=test&token={token}"
    ) as ws:
        first = ws.receive_json()
        assert first["type"] == "UPDATE"
        assert len(first["board"]) == 6
        assert all(len(row) == 7 for row in first["board"])
        assert all(cell == 0 for row in first["board"] for cell in row)
        assert first["currentTurn"] == 1


def test_websocket_human_move_round_trip(ws_client):
    game = _create_game(ws_client, PlayerType.HUMAN, PlayerType.HUMAN)

    with ws_client.websocket_connect(
        f"/games/{game['id']}/ws?env=test&token={game['player_1_token']}"
    ) as ws:
        ws.receive_json()  # discard initial UPDATE
        ws.send_json({"action": "MOVE", "column": 3})

        update = ws.receive_json()
        assert update["type"] == "UPDATE"
        assert update["lastMove"]["column"] == 3
        assert update["board"][-1][3] == 1
        assert update["currentTurn"] == 2


def test_websocket_rejects_invalid_token_and_drops_move(ws_client):
    """Wrong token → ValueError on the server side; move never lands. The
    connection stays open. We verify by reading the DB after disconnect."""
    game = _create_game(ws_client, PlayerType.HUMAN, PlayerType.HUMAN)

    with ws_client.websocket_connect(
        f"/games/{game['id']}/ws?env=test&token=wrong-token"
    ) as ws:
        ws.receive_json()  # initial UPDATE
        ws.send_json({"action": "MOVE", "column": 0})
        # Server logs the rejection silently — no UPDATE is pushed.

    # Drain the pool — TestClient created connections in its own event loop;
    # _check() below runs in a fresh asyncio.run() loop and would otherwise
    # try to reuse those defunct connections.
    async def _drain():
        await engines["test"].dispose()
        await engines["prod"].dispose()

    asyncio.run(_drain())

    async def _check():
        async with session_makers["test"]() as session:
            row = await session.get(Game, game["id"])
            assert row.history == []
        await engines["test"].dispose()
        await engines["prod"].dispose()

    asyncio.run(_check())
