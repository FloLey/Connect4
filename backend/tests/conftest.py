"""Shared pytest fixtures.

Defaults DATABASE_URL / TEST_DATABASE_URL to the docker-compose local mapping
(localhost:5433) so tests work without docker exec. Set those env vars to
override (e.g. when running inside the backend container).
"""

import asyncio
import os
from pathlib import Path

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5433/connect4_arena",
)
os.environ.setdefault(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost:5433/connect4_test",
)

import pytest
import pytest_asyncio
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.app.core.database import Base, engines, session_makers
from backend.app.main import app


_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config_for_test_db() -> AlembicConfig:
    """Build an Alembic config that points at the test DB.

    Alembic's env.py reads ``DATABASE_URL`` for its target. We swap that env
    var to ``TEST_DATABASE_URL`` for the duration of the migration.
    """
    cfg = AlembicConfig(str(_BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_ROOT / "alembic"))
    return cfg


def _ensure_schema_sync():
    """Bring the test DB schema up to date via Alembic, then dispose the pool.

    Disposing matters: SQLAlchemy's async engine caches connections per pool,
    and asyncpg connections are bound to the event loop that created them.
    Without dispose, every per-test loop would inherit dead connections from
    the throwaway loop used here.

    Implementation note: alembic's runner is sync and reads DATABASE_URL. We
    swap DATABASE_URL → TEST_DATABASE_URL just for the migration, then put
    it back so the rest of the test suite still talks to engines["test"]
    via the regular SQLAlchemy URLs.
    """
    saved_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
    try:
        cfg = _alembic_config_for_test_db()
        alembic_command.upgrade(cfg, "head")
    finally:
        if saved_db_url is not None:
            os.environ["DATABASE_URL"] = saved_db_url

    async def _dispose():
        await engines["test"].dispose()
        await engines["prod"].dispose()

    asyncio.run(_dispose())


# Create the schema exactly once per pytest invocation, before any test or
# fixture runs. Using a fixture for this collides with pytest-asyncio 1.x's
# strict event-loop scoping rules.
_ensure_schema_sync()


async def _truncate_all():
    async with engines["test"].begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(
                text(f'TRUNCATE TABLE "{table.name}" RESTART IDENTITY CASCADE')
            )


async def _dispose_engines():
    """Drain pool entries between tests so per-test event loops don't inherit
    asyncpg connections bound to a defunct loop."""
    await engines["test"].dispose()
    await engines["prod"].dispose()


@pytest_asyncio.fixture
async def test_db():
    """Yield a session bound to the test DB. Truncates and disposes the pool
    between tests so connections don't leak across event loops."""
    SessionLocal = session_makers["test"]
    async with SessionLocal() as session:
        yield session

    await _truncate_all()
    await _dispose_engines()


@pytest_asyncio.fixture
async def client():
    """ASGI httpx client that injects x-db-env=test on every request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"x-db-env": "test"},
    ) as c:
        yield c

    await _truncate_all()
    await _dispose_engines()
