"""Async SQLAlchemy engine and session factory.

Usage
-----
    from agent_saan.db.session import get_session

    async with get_session() as session:
        result = await session.execute(select(SessionORM))
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent_saan.config import get_settings

_settings = get_settings()

# Create the async engine once at module import time.
engine = create_async_engine(
    _settings.database_url,
    echo=_settings.log_level == "DEBUG",
    pool_pre_ping=True,
)

# Session factory — use as an async context manager.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
