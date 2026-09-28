"""Service test fixtures: SQLite + fakeredis, no containers (spec section 11).

The app is built against an in-memory SQLite database and a fake Redis so the
suite runs anywhere. CI additionally runs the same tests against a PostgreSQL
service container, which is what catches dialect-specific mistakes — the reason
``app/db/types.GUID`` exists.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# Settings are read at import time, so the environment must be complete before
# anything under app.* is imported.
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:5173")
os.environ.setdefault("SEED_ON_STARTUP", "false")

pytest_plugins = ("anyio",)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

artifacts_required = pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACT_DIR, "xgb_model.json")),
    reason="artifacts not built; run `make train`",
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def engine():
    """A fresh in-memory database per test, with the full schema created."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    from app.db.models import Base
    import app.db.models  # noqa: F401  (registers the demo tables)

    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        # One shared connection, or ":memory:" gives each checkout its own
        # empty database and nothing persists between statements.
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        yield s


@pytest.fixture
async def fake_redis():
    """A fresh fake Redis bound to the test's own event loop.

    This has to be an async fixture. Built synchronously, the client binds to
    whichever loop happens to be current, and later cache calls fail against a
    closed loop. Scoring swallows cache errors by design, so the symptom is not
    an exception but a cache that silently never hits.
    """
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
async def client(engine, fake_redis, monkeypatch):
    """HTTP client against the real app, wired to SQLite and fakeredis."""
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    import app.db.session as session_module

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", maker)

    from app.main import create_app

    application = create_app()

    async def override_session():
        async with maker() as s:
            yield s

    application.dependency_overrides[session_module.get_session] = override_session

    import redis.asyncio as redis_async

    monkeypatch.setattr(redis_async, "from_url", lambda *a, **k: fake_redis)

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        async with application.router.lifespan_context(application):
            yield c


# --- data builders ---------------------------------------------------------

BASE_TIME = datetime(2026, 9, 1, tzinfo=timezone.utc)


def make_deal(deal_id: str = "LIVE-0001", **overrides):
    from app.db.models import Deal

    fields = {
        "id": deal_id,
        "company": "Example Holdings",
        "industry": "Technology",
        "region": "EMEA",
        "deal_size": 120000,
        "stage": "Proposal",
        "source": "Inbound",
        "owner_rep": "Dana Reed",
        "created_at": BASE_TIME - timedelta(days=40),
        "expected_close": BASE_TIME + timedelta(days=20),
        "closed_at": None,
        "won": None,
        "ingested_at": BASE_TIME,
    }
    fields.update(overrides)
    return Deal(**fields)


def make_activity(deal_id: str, kind: str, day: int, activity_id: str | None = None):
    from app.db.models import Activity

    return Activity(
        id=activity_id or f"A-{deal_id}-{kind}-{day}",
        deal_id=deal_id,
        type=kind,
        occurred_at=BASE_TIME - timedelta(days=40) + timedelta(days=day),
        origin="seed",
    )


def make_score(deal_id: str, win_prob: float = 0.61, minutes_ago: int = 0, **overrides):
    from app.db.models import Score

    fields = {
        "deal_id": deal_id,
        "win_prob": win_prob,
        "days_to_close": 34.0,
        "model_version": "demo-test",
        "shap_top": [{"feature": "has_champion", "shap": 0.4}],
        "feature_hash": "abc123",
        "latency_ms": {"total": 12.0},
        "cache_hit": False,
        "scored_at": BASE_TIME - timedelta(minutes=minutes_ago),
    }
    fields.update(overrides)
    return Score(**fields)


def make_action(deal_id: str, priority: str = "HIGH", **overrides):
    from app.db.models import Action, ActionStatus

    fields = {
        "deal_id": deal_id,
        "rule_id": "R-TEST",
        "action_type": "schedule_call",
        "reason": "Test reason citing a driver.",
        "priority": priority,
        "urgency_score": 1.0,
        "status": ActionStatus.suggested,
        "created_at": BASE_TIME,
    }
    fields.update(overrides)
    return Action(**fields)
