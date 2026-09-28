"""Async engine and session factory.

Kept separate from ``app/dependencies.py`` (which builds its own engine at import
time) so tests can point at SQLite without importing the JWT/Redis machinery.
Pool sizing is skipped for SQLite, which rejects those arguments.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings


def make_engine(url: str | None = None, **kwargs):
    """Build an async engine for ``url`` (defaults to settings.DATABASE_URL)."""
    url = url or settings.DATABASE_URL
    options: dict = {"echo": False, "future": True}

    if url.startswith("sqlite"):
        # aiosqlite takes neither pool_size nor max_overflow.
        options["connect_args"] = {"check_same_thread": False}
    else:
        options["pool_size"] = settings.WEB_CONCURRENCY * 2
        options["max_overflow"] = settings.WEB_CONCURRENCY * 3
        options["pool_pre_ping"] = True

    options.update(kwargs)
    return create_async_engine(url, **options)


engine = make_engine()
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a session that rolls back on error."""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
