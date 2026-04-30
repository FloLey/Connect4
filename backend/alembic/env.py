"""Alembic environment for Connect4.

Reads ``DATABASE_URL`` from the environment (set via docker-compose / .env) and
swaps the asyncpg driver for psycopg2 so Alembic's sync runner can connect.
``Base.metadata`` is populated by importing every model module — autogenerate
needs every table model to be imported before introspection.
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `backend.app.*` importable. alembic runs from backend/, so we add the
# project root (one level up) to sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from backend.app.core.database import Base  # noqa: E402

# Importing the package re-exports every model class and registers it on
# Base.metadata so autogenerate sees the full schema.
from backend.app import models  # noqa: F401, E402


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_database_url() -> str:
    """Return a sync (psycopg2) URL derived from the configured DATABASE_URL.

    Alembic's runner is sync, so we strip the ``+asyncpg`` driver suffix.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL env var is required for alembic")
    return url.replace("+asyncpg", "+psycopg2")


def run_migrations_offline() -> None:
    """Generate SQL without connecting to a database (for review / CI dry runs)."""
    context.configure(
        url=_sync_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _sync_database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
