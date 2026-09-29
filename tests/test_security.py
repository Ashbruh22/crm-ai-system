"""Phase 8 acceptance: 401, 429 and CORS behave (spec sections 10, 14)."""

from __future__ import annotations

import pytest

from tests.conftest import artifacts_required, make_activity, make_deal

pytestmark = [pytest.mark.anyio, artifacts_required]

GOOD_TOKEN = "a-sufficiently-long-admin-token"


@pytest.fixture(autouse=True)
def reset_limiter():
    """Rate limits are process-global; leaking counts between tests makes
    unrelated assertions fail in whatever order pytest happens to run."""
    from app.security.ratelimit import limiter

    limiter.reset()
    yield
    limiter.reset()


async def seed(session, deal_id="LIVE-0001"):
    session.add(make_deal(deal_id))
    session.add(make_activity(deal_id, "email_replied", 2))
    await session.commit()


# --- 401 -------------------------------------------------------------------


async def test_admin_reset_without_a_token_is_401(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_TOKEN", GOOD_TOKEN)

    response = await client.post("/api/admin/reset")
    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


async def test_admin_reset_with_a_wrong_token_is_401(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_TOKEN", GOOD_TOKEN)

    response = await client.post(
        "/api/admin/reset", headers={"Authorization": "Bearer not-the-token"}
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


async def test_admin_is_disabled_rather_than_open_when_unconfigured(
    client, monkeypatch
):
    """An unset token must not mean "no check"."""
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_TOKEN", None)

    response = await client.post(
        "/api/admin/reset", headers={"Authorization": "Bearer anything"}
    )
    assert response.status_code == 503
    assert "ADMIN_TOKEN" in response.json()["detail"]


async def test_a_short_token_is_refused_as_no_protection(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_TOKEN", "short")

    response = await client.post(
        "/api/admin/reset", headers={"Authorization": "Bearer short"}
    )
    assert response.status_code == 503


async def test_admin_reset_with_the_right_token_works(client, session, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ADMIN_TOKEN", GOOD_TOKEN)
    await seed(session)
    await client.post("/api/deals/LIVE-0001/score")

    response = await client.post(
        "/api/admin/reset", headers={"Authorization": f"Bearer {GOOD_TOKEN}"}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["reset"] is True
    # The hand-made deal is gone and the synthetic live set is back.
    assert body["before"]["scores"] >= 1
    assert body["after"]["scores"] == 0
    assert body["after"]["deals"] == 60


async def test_webhook_without_a_signature_is_401(client):
    response = await client.post("/webhooks/crm", content=b"{}")
    assert response.status_code == 401


# --- 429 -------------------------------------------------------------------


async def test_event_ingestion_is_rate_limited(client, session):
    """Ingestion creates work for the consumer, so it has the tightest cap."""
    from app.security.ratelimit import EVENT_LIMIT

    await seed(session)
    cap = int(EVENT_LIMIT.split("/")[0])

    statuses = []
    for _ in range(cap + 5):
        r = await client.post(
            "/api/events", json={"deal_id": "LIVE-0001", "type": "email_sent"}
        )
        statuses.append(r.status_code)

    assert 429 in statuses, f"never rate limited: {sorted(set(statuses))}"
    assert statuses[0] == 202, "the first call should be allowed"
    # Once limited, it stays limited for the window.
    assert statuses[-1] == 429


async def test_a_429_explains_itself_and_says_when_to_retry(client, session):
    from app.security.ratelimit import EVENT_LIMIT

    await seed(session)
    cap = int(EVENT_LIMIT.split("/")[0])

    last = None
    for _ in range(cap + 3):
        last = await client.post(
            "/api/events", json={"deal_id": "LIVE-0001", "type": "email_sent"}
        )

    assert last is not None and last.status_code == 429
    assert "Retry-After" in last.headers
    assert int(last.headers["Retry-After"]) > 0

    detail = last.json()["detail"]
    assert "Too many requests" in detail
    # It should say why, not just refuse.
    assert "free-tier" in detail


async def test_what_if_has_its_own_looser_cap(client, session):
    """Sliders fire often, so what-if must not share ingestion's tight limit."""
    from app.security.ratelimit import EVENT_LIMIT, WHATIF_LIMIT

    assert int(WHATIF_LIMIT.split("/")[0]) > int(EVENT_LIMIT.split("/")[0])

    await seed(session)
    await client.post("/api/deals/LIVE-0001/score")

    # Comfortably past the ingestion cap, still fine for what-if.
    for _ in range(int(EVENT_LIMIT.split("/")[0]) + 2):
        r = await client.post(
            "/api/deals/LIVE-0001/what-if",
            json={"overrides": {"n_stakeholders": 3}},
        )
        assert r.status_code == 200, r.text


async def test_reads_are_not_limited_at_demo_browsing_rates(client, session):
    await seed(session)
    for _ in range(30):
        assert (await client.get("/api/deals")).status_code == 200


async def test_the_proxy_header_identifies_the_caller(client, session):
    """Behind Render every request shares one source IP; without honouring
    X-Forwarded-For the whole world would be rate-limited as one visitor."""
    from starlette.datastructures import Headers

    from app.security.ratelimit import client_key

    class FakeRequest:
        def __init__(self, headers):
            self.headers = Headers(headers)
            self.client = type("C", (), {"host": "10.0.0.1"})()

    assert (
        client_key(FakeRequest({"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}))
        == "203.0.113.7"
    )
    assert client_key(FakeRequest({"X-Real-IP": "203.0.113.9"})) == "203.0.113.9"
    assert client_key(FakeRequest({})) == "10.0.0.1"


# --- CORS ------------------------------------------------------------------


async def test_allowed_origin_gets_the_cors_header(client):
    response = await client.get(
        "/healthz", headers={"Origin": "http://localhost:5173"}
    )
    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:5173"
    )


async def test_disallowed_origin_gets_no_cors_header(client):
    response = await client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert response.headers.get("access-control-allow-origin") is None


async def test_preflight_from_an_allowed_origin_is_accepted(client):
    response = await client.options(
        "/api/events",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:5173"
    )


async def test_preflight_from_a_disallowed_origin_is_refused(client):
    response = await client.options(
        "/api/events",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") is None


def test_wildcard_origin_is_rejected_at_configuration_time():
    """The old code shipped allow_origins=["*"] with allow_credentials=True."""
    from app.config import Settings

    settings = Settings(
        SECRET_KEY="x" * 16,
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        REDIS_URL="redis://localhost:6379/0",
        ALLOWED_ORIGINS="*",
    )
    with pytest.raises(ValueError, match="explicitly"):
        _ = settings.allowed_origins
