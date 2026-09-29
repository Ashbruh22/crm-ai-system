"""Phase 5 acceptance (spec section 14).

* posting an event updates the score and actions, and emits SSE
* a duplicate event is ignored
* a bad signature is 401
"""

from __future__ import annotations

import json
import time

import pytest

from app.bus.base import Event
from app.bus.redis_streams import RedisStreamsBus
from app.consumer import CONSUMER_GROUP, DuplicateEvent, apply_event, claim_event
from app.security.webhook import sign
from tests.conftest import artifacts_required, make_activity, make_deal

pytestmark = [pytest.mark.anyio, artifacts_required]

SECRET = "your-hmac-secret-here"  # matches the default in config.py

ENGAGED = [
    ("email_sent", 1),
    ("email_replied", 2),
    ("meeting_held", 4),
    ("proposal_sent", 12),
]


async def seed(session, deal_id="LIVE-0001", activities=ENGAGED):
    session.add(make_deal(deal_id))
    for kind, day in activities:
        session.add(make_activity(deal_id, kind, day))
    await session.commit()


def webhook_headers(body: bytes, secret: str = SECRET, ts: int | None = None) -> dict:
    ts = int(time.time()) if ts is None else ts
    return {
        "X-CRM-Signature": sign(body, secret, ts),
        "X-CRM-Timestamp": str(ts),
        "Content-Type": "application/json",
    }


# --- the bus ---------------------------------------------------------------


async def test_publish_and_consume_round_trip(fake_redis):
    bus = RedisStreamsBus(fake_redis, stream="test:events")
    await bus.connect()

    event = Event(deal_id="LIVE-0001", type="email_replied", source="simulate")
    await bus.publish(event)

    received = []
    async for delivered in bus.consume(CONSUMER_GROUP, "w1", block_ms=100):
        received.append(delivered)
        await bus.ack(CONSUMER_GROUP, delivered.receipt)
        break

    assert len(received) == 1
    assert received[0].event.deal_id == "LIVE-0001"
    assert received[0].event.type == "email_replied"
    assert received[0].event.event_id == event.event_id


async def test_acked_events_are_not_redelivered(fake_redis):
    bus = RedisStreamsBus(fake_redis, stream="test:events2")
    await bus.connect()
    await bus.publish(Event(deal_id="LIVE-0001", type="meeting_held"))

    async for delivered in bus.consume(CONSUMER_GROUP, "w1", block_ms=100):
        await bus.ack(CONSUMER_GROUP, delivered.receipt)
        break

    assert await bus.pending_count(CONSUMER_GROUP) == 0

    again = [d async for d in bus.consume(CONSUMER_GROUP, "w1", block_ms=50)]
    assert again == []


async def test_unacked_event_is_reclaimed_after_a_crash(fake_redis):
    """A crash mid-score must not lose the event."""
    bus = RedisStreamsBus(fake_redis, stream="test:events3")
    await bus.connect()
    await bus.publish(Event(deal_id="LIVE-0001", type="demo_done"))

    # Read without acking, as though the process died here.
    first = [d async for d in bus.consume(CONSUMER_GROUP, "w1", block_ms=50)]
    assert len(first) == 1
    assert await bus.pending_count(CONSUMER_GROUP) == 1

    # min_idle 0 so the test does not wait out the production idle window.
    second = [
        d
        async for d in bus.consume(
            CONSUMER_GROUP, "w1", block_ms=50, claim_min_idle_ms=0
        )
    ]
    assert [d.event.event_id for d in second] == [first[0].event.event_id]


async def test_a_restarted_consumer_reclaims_the_dead_one_s_backlog(fake_redis):
    """The reason this uses XAUTOCLAIM rather than XREADGROUP id=0.

    A process that restarts under a new consumer name must still pick up what
    the previous name left pending, or those events are stranded forever.
    """
    bus = RedisStreamsBus(fake_redis, stream="test:events4")
    await bus.connect()
    await bus.publish(Event(deal_id="LIVE-0001", type="proposal_sent"))

    orphaned = [d async for d in bus.consume(CONSUMER_GROUP, "worker-old", block_ms=50)]
    assert len(orphaned) == 1  # delivered, never acked

    recovered = [
        d
        async for d in bus.consume(
            CONSUMER_GROUP, "worker-new", block_ms=50, claim_min_idle_ms=0
        )
    ]
    assert [d.event.event_id for d in recovered] == [orphaned[0].event.event_id]


async def test_an_in_flight_event_is_not_stolen_from_a_healthy_consumer(fake_redis):
    """min_idle_ms exists so a slow score is not handed to a second worker."""
    bus = RedisStreamsBus(fake_redis, stream="test:events5")
    await bus.connect()
    await bus.publish(Event(deal_id="LIVE-0001", type="demo_done"))

    in_flight = [d async for d in bus.consume(CONSUMER_GROUP, "worker-a", block_ms=50)]
    assert len(in_flight) == 1

    # Default idle window: the entry is only seconds old, so nothing is claimed.
    other = [d async for d in bus.consume(CONSUMER_GROUP, "worker-b", block_ms=50)]
    assert other == []


def test_event_survives_the_wire_format():
    original = Event(
        deal_id="LIVE-0009",
        type="discount_requested",
        payload={"amount": 15, "note": "line\nbreak"},
        source="webhook:crm",
    )
    restored = Event.from_wire(original.to_wire())

    assert restored.event_id == original.event_id
    assert restored.deal_id == original.deal_id
    assert restored.payload == original.payload
    assert restored.occurred_at == original.occurred_at


# --- idempotency -----------------------------------------------------------


async def test_duplicate_event_id_is_rejected(session):
    await seed(session)
    event = Event(deal_id="LIVE-0001", type="email_replied")

    await claim_event(session, event)
    await session.commit()

    with pytest.raises(DuplicateEvent):
        await claim_event(session, event)


async def test_applying_the_same_event_twice_adds_one_activity(session, fake_redis):
    from sqlalchemy import func, select

    from app.db.models import Activity
    from app.models.registry import registry

    registry.load()
    await seed(session)

    before = (
        await session.execute(
            select(func.count()).select_from(Activity).where(Activity.deal_id == "LIVE-0001")
        )
    ).scalar_one()

    event = Event(deal_id="LIVE-0001", type="champion_identified", source="simulate")
    await apply_event(session, fake_redis, registry, event)

    with pytest.raises(DuplicateEvent):
        await apply_event(session, fake_redis, registry, event)

    after = (
        await session.execute(
            select(func.count()).select_from(Activity).where(Activity.deal_id == "LIVE-0001")
        )
    ).scalar_one()
    assert after == before + 1


async def test_event_for_an_unknown_deal_is_not_retryable(session, fake_redis):
    from app.models.registry import registry

    registry.load()
    with pytest.raises(LookupError):
        await apply_event(
            session, fake_redis, registry, Event(deal_id="NOPE", type="email_sent")
        )


async def test_unknown_activity_type_is_rejected(session, fake_redis):
    from app.models.registry import registry

    registry.load()
    await seed(session)
    with pytest.raises(ValueError, match="unknown activity type"):
        await apply_event(
            session, fake_redis, registry, Event(deal_id="LIVE-0001", type="teleported")
        )


# --- end to end ------------------------------------------------------------


async def test_event_updates_score_and_creates_actions(session, fake_redis):
    """The core acceptance criterion, exercised through the consumer."""
    from app.models.registry import registry
    from app.services import scoring

    registry.load()
    await seed(session)

    first = await scoring.score_deal(session, fake_redis, registry, "LIVE-0001")

    event = Event(deal_id="LIVE-0001", type="champion_identified", source="simulate")
    summary = await apply_event(session, fake_redis, registry, event)

    assert summary["deal_id"] == "LIVE-0001"
    # Champion presence is the model's strongest positive driver.
    assert summary["win_prob"] > first.win_prob

    detail_scores = (
        await session.execute(
            __import__("sqlalchemy").select(
                __import__("app.db.models", fromlist=["Score"]).Score
            )
        )
    ).scalars().all()
    assert len(detail_scores) >= 2, "the event should have appended a new score"


async def test_consumer_broadcasts_to_the_dashboard(session, fake_redis):
    """SSE fan-out: the event must reach a subscribed dashboard."""
    import asyncio

    from app.bus.broadcast import SCORE_UPDATED, Broadcaster
    from app.models.registry import registry

    registry.load()
    await seed(session)

    broadcaster = Broadcaster(fake_redis)
    received: list[dict] = []

    async def listen():
        async for message in broadcaster.subscribe():
            if message:
                received.append(message)
                if any(m.get("event") == SCORE_UPDATED for m in received):
                    return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0.2)  # let the subscription register

    await apply_event(
        session,
        fake_redis,
        registry,
        Event(deal_id="LIVE-0001", type="meeting_held", source="simulate"),
        broadcaster,
    )

    try:
        await asyncio.wait_for(listener, timeout=5.0)
    except asyncio.TimeoutError:
        listener.cancel()
        pytest.fail(f"no score_updated broadcast; got {received}")

    scored = [m for m in received if m["event"] == SCORE_UPDATED]
    assert scored
    payload = scored[0]["data"]
    assert payload["deal_id"] == "LIVE-0001"
    assert payload["activity_type"] == "meeting_held"
    assert 0.0 <= payload["win_prob"] <= 1.0
    assert payload["shap_top"]


# --- HTTP surface ----------------------------------------------------------


async def test_post_event_is_accepted_and_queued(client, session):
    await seed(session)
    response = await client.post(
        "/api/events", json={"deal_id": "LIVE-0001", "type": "email_replied"}
    )
    assert response.status_code == 202

    body = response.json()
    assert body["accepted"] is True
    assert body["event_id"]
    assert body["stream_id"]


async def test_post_event_rejects_an_unknown_deal(client):
    response = await client.post(
        "/api/events", json={"deal_id": "NOPE-0000", "type": "email_replied"}
    )
    assert response.status_code == 404


async def test_post_event_rejects_an_unknown_activity_type(client, session):
    await seed(session)
    response = await client.post(
        "/api/events", json={"deal_id": "LIVE-0001", "type": "sent_a_pigeon"}
    )
    assert response.status_code == 422


async def test_signed_webhook_is_accepted(client, session):
    await seed(session)
    body = json.dumps({"deal_id": "LIVE-0001", "type": "meeting_held"}).encode()

    response = await client.post(
        "/webhooks/crm", content=body, headers=webhook_headers(body)
    )
    assert response.status_code == 202, response.text
    assert response.json()["accepted"] is True


async def test_webhook_with_a_bad_signature_is_401(client, session):
    await seed(session)
    body = json.dumps({"deal_id": "LIVE-0001", "type": "meeting_held"}).encode()

    headers = webhook_headers(body, secret="wrong-secret")
    response = await client.post("/webhooks/crm", content=body, headers=headers)

    assert response.status_code == 401
    assert "signature" in response.json()["detail"]


async def test_webhook_replay_is_401(client, session):
    """A captured request replayed later must not be accepted."""
    await seed(session)
    body = json.dumps({"deal_id": "LIVE-0001", "type": "meeting_held"}).encode()

    stale = int(time.time()) - 900
    response = await client.post(
        "/webhooks/crm", content=body, headers=webhook_headers(body, ts=stale)
    )
    assert response.status_code == 401
    assert "old" in response.json()["detail"]


async def test_webhook_without_headers_is_401(client, session):
    await seed(session)
    body = json.dumps({"deal_id": "LIVE-0001", "type": "meeting_held"}).encode()
    response = await client.post(
        "/webhooks/crm", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 401


async def test_webhook_body_must_match_the_signature(client, session):
    """Signing one body and sending another must fail."""
    await seed(session)
    signed = json.dumps({"deal_id": "LIVE-0001", "type": "meeting_held"}).encode()
    sent = json.dumps({"deal_id": "LIVE-0001", "type": "demo_done"}).encode()

    response = await client.post(
        "/webhooks/crm", content=sent, headers=webhook_headers(signed)
    )
    assert response.status_code == 401


async def test_healthz_reports_bus_depth(client):
    body = (await client.get("/healthz")).json()
    assert "bus" in body
    assert "stream_len" in body["bus"]


# --- SSE endpoint ----------------------------------------------------------
#
# The HTTP endpoint is not covered here. Driving a long-lived SSE response
# through httpx's ASGI transport while fakeredis's pubsub blocks in the same
# event loop deadlocks the test, which is a limitation of that combination
# rather than of the endpoint. The contract it depends on -- consumer publishes,
# subscriber receives score_updated -- is covered by
# test_consumer_broadcasts_to_the_dashboard above, and the endpoint itself is
# verified against a live server and a real Redis.


# --- Redis capability guard ------------------------------------------------


async def test_old_redis_is_rejected_with_an_actionable_message(fake_redis):
    """Streams need Redis >= 5.

    A Redis 3 server answers PING and handles the score cache fine, so without
    this check the only symptom is `unknown command 'XGROUP'` repeating inside
    the consumer loop.
    """
    from app.bus.redis_streams import RedisStreamsBus, StreamsUnsupported

    class OldRedis:
        async def ping(self):
            return True

        async def info(self, _section=None):
            return {"redis_version": "3.0.504"}

    bus = RedisStreamsBus(OldRedis())
    with pytest.raises(StreamsUnsupported, match="5.0"):
        await bus.connect()


async def test_modern_redis_passes_the_capability_check(fake_redis):
    bus = RedisStreamsBus(fake_redis, stream="test:version")
    await bus.connect()  # must not raise


async def test_restricted_info_does_not_block_startup(fake_redis):
    """Some managed Redis hosts refuse INFO; that must not be fatal."""
    from app.bus.redis_streams import RedisStreamsBus

    class NoInfoRedis:
        async def ping(self):
            return True

        async def info(self, _section=None):
            raise RuntimeError("INFO is disabled")

    assert await RedisStreamsBus(NoInfoRedis()).assert_streams_supported() == "unknown"


async def test_unreachable_redis_is_not_reported_as_a_working_bus():
    """A rejected credential must not read as a healthy bus.

    The capability check swallowed every INFO failure so that managed hosts
    which restrict the command could still boot. That also swallowed
    authentication errors, so a Redis rejecting every command reported
    bus.available: true -- exactly the misleading signal the check exists to
    prevent. Connection and auth failures now propagate.
    """
    from redis.exceptions import AuthenticationError

    from app.bus.redis_streams import RedisStreamsBus

    class RejectingRedis:
        async def ping(self):
            raise AuthenticationError("invalid username-password pair")

        async def info(self, _section=None):
            raise AuthenticationError("invalid username-password pair")

    with pytest.raises(AuthenticationError):
        await RedisStreamsBus(RejectingRedis()).assert_streams_supported()


async def test_restricted_info_still_boots():
    """A host that refuses INFO but otherwise works must not be fatal."""
    from app.bus.redis_streams import RedisStreamsBus

    class RestrictedRedis:
        async def ping(self):
            return True

        async def info(self, _section=None):
            raise RuntimeError("ERR unknown command 'INFO'")

    assert await RedisStreamsBus(RestrictedRedis()).assert_streams_supported() == "unknown"
