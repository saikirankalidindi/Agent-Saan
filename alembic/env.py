"""Alembic environment configuration.

Reads DATABASE_URL from the application settings (or the DATABASE_URL environment
variable directly) and configures both online (async) and offline migration modes.
"""

import asyncio
import os
import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Alembic Config object — gives access to values in alembic.ini
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import the SQLAlchemy metadata from our ORM models so that autogenerate
# can detect schema changes.
# ---------------------------------------------------------------------------
# Add the src/ directory to sys.path so the package is importable.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_saan.db.base import Base  # noqa: E402  (import after sys.path update)
from agent_saan.db import models  # noqa: F401, E402  (import all ORM models to register them)

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Resolve the database URL
# ---------------------------------------------------------------------------
# Priority: DATABASE_URL env var → alembic.ini sqlalchemy.url
# Alembic (and psycopg2) need a *sync* URL for offline mode.
# For online (async) mode we use asyncpg via async_engine_from_config.
_raw_url: str = os.environ.get("DATABASE_URL", "") or config.get_main_option(
    "sqlalchemy.url", ""
)

# Convert asyncpg URL to sync psycopg2 URL for offline mode
_sync_url = re.sub(r"^postgresql\+asyncpg", "postgresql", _raw_url)

# Set the sync URL on the config so offline mode works
config.set_main_option("sqlalchemy.url", _sync_url)


# ---------------------------------------------------------------------------
# Offline migration (generates SQL without a live DB connection)
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an
    Engine is acceptable here as well.  By skipping the Engine creation we
    don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration (runs against a live DB using asyncpg)
# ---------------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations."""
    # Build the async URL (replace psycopg2 scheme back to asyncpg)
    async_url = re.sub(r"^postgresql(?!\+)", "postgresql+asyncpg", _sync_url)

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = async_url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
