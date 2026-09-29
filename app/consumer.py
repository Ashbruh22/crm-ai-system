"""Stream consumer: event -> activity -> score -> actions -> SSE (spec section 7).

Runs as a background task inside the same process as the API, so the hosted demo
needs one Render service rather than two. The consumer group means a restart
resumes where it stopped instead of replaying or skipping.

Ordering of the handler is deliberate:

1. Claim the event id in ``processed_events``. **First**, before any work — a
   redelivery must be rejected even if the previous attempt crashed halfway.
2. Insert the activity.
3. Invalidate the deal's cached score, so the re-score reads fresh features.
4. Re-score, which also runs the NBA rules and writes the action ledger.
5. Broadcast to connected dashboards.
6. Acknowledge the stream entry.

The ack comes last on purpose: if the process dies at step 4, the event is still
pending and gets redelivered. The idempotency claim in step 1 then makes that
redelivery a no-op rather than a duplicate activity.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.bus.base import Event
from app.bus.broadcast import ACTION_CREATED, SCORE_UPDATED, Broadcaster
from app.db.models import Activity, Deal, ProcessedEvent
from app.features.build import ACTIVITY_TYPES
from app.services import scoring

log = logging.getLogger("crm_ai.consumer")

CONSUMER_GROUP = "crm-ai-scorer"


class DuplicateEvent(Exception):
    """This event id has already been processed."""


async def claim_event(
    session: AsyncSession, event: Event, consumer: str = CONSUMER_GROUP
) -> None:
    """Record the event id, or raise DuplicateEvent if it is already there.

    Relies on the primary key rather than a SELECT-then-INSERT: two workers
    handling the same redelivery concurrently would both pass a existence check,
    but only one can win the insert.
    """
    session.add(
        ProcessedEvent(
            event_id=event.event_id,
            deal_id=event.deal_id,
            consumer=consumer,
            processed_at=datetime.now(timezone.utc),
        )
    )
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise DuplicateEvent(event.event_id)


async def apply_event(
    session: AsyncSession,
    redis,
    registry,
    event: Event,
    broadcaster: Broadcaster | None = None,
) -> dict:
    """Handle one event end to end. Returns a summary of what happened."""
    if event.type not in ACTIVITY_TYPES:
        raise ValueError(
            f"unknown activity type {event.type!r}; expected one of {list(ACTIVITY_TYPES)}"
        )

    deal = await session.get(Deal, event.deal_id)
    if deal is None:
        raise LookupError(f"deal {event.deal_id!r} not found")

    await claim_event(session, event)

    session.add(
        Activity(
            id=f"EV-{event.event_id[:34]}",
            deal_id=event.deal_id,
            type=event.type,
            occurred_at=event.occurred_at,
            payload=event.payload or None,
            origin=event.source.split(":")[0][:16] or "event",
        )
    )
    await session.commit()

    # The deal's features have changed, so any cached score is stale.
    await scoring.invalidate_deal(redis, event.deal_id)

    result = await scoring.score_deal(
        session, redis, registry, event.deal_id, bypass_cache=True, persist=True
    )

    if broadcaster is not None:
        await broadcaster.publish(
            SCORE_UPDATED,
            {
                "deal_id": event.deal_id,
                "event_id": event.event_id,
                "activity_type": event.type,
                "win_prob": result.win_prob,
                "days_to_close": result.days_to_close,
                "model_version": result.model_version,
                "shap_top": result.shap_top[:5],
                "latency_ms": result.latency_ms,
                "scored_at": result.scored_at.isoformat(),
            },
        )
        for rec in result.recommendations:
            await broadcaster.publish(
                ACTION_CREATED,
                {
                    "deal_id": event.deal_id,
                    "rule_id": rec.rule_id,
                    "action_type": rec.action_type,
                    "reason": rec.reason,
                    "priority": rec.priority,
                },
            )

    return {
        "deal_id": event.deal_id,
        "event_id": event.event_id,
        "win_prob": result.win_prob,
        "days_to_close": result.days_to_close,
        "actions": len(result.recommendations),
    }


async def run_consumer(
    bus,
    session_factory,
    redis,
    registry,
    broadcaster: Broadcaster | None = None,
    consumer_name: str = "worker-1",
    stop_event: asyncio.Event | None = None,
) -> None:
    """Consume forever. Cancelled by the app's lifespan on shutdown."""
    from app.bus.redis_streams import StreamsUnsupported

    try:
        await bus.connect()
    except StreamsUnsupported as exc:
        # Retrying cannot fix the server's version. Log once and stop, rather
        # than emitting the same error every two seconds forever.
        log.error("event consumer disabled: %s", exc)
        return

    log.info("consumer %s started on group %s", consumer_name, CONSUMER_GROUP)

    while stop_event is None or not stop_event.is_set():
        try:
            async for delivered in bus.consume(
                CONSUMER_GROUP, consumer_name, block_ms=2000, count=10
            ):
                await _handle_one(
                    bus, session_factory, redis, registry, broadcaster, delivered
                )
        except asyncio.CancelledError:
            log.info("consumer %s stopping", consumer_name)
            raise
        except Exception as exc:  # noqa: BLE001
            # A failure in the read loop must not kill the consumer; back off
            # briefly rather than spinning on a persistent error.
            log.exception("consumer loop error: %s", exc)
            await asyncio.sleep(2.0)


async def _handle_one(bus, session_factory, redis, registry, broadcaster, delivered):
    event = delivered.event
    async with session_factory() as session:
        try:
            summary = await apply_event(session, redis, registry, event, broadcaster)
            log.info("scored %s -> %.4f", event.deal_id, summary["win_prob"])
        except DuplicateEvent:
            log.info("duplicate event %s ignored", event.event_id)
        except LookupError as exc:
            # An event for a deal we do not have is not retryable.
            log.warning("dropping event %s: %s", event.event_id, exc)
        except ValueError as exc:
            log.warning("dropping malformed event %s: %s", event.event_id, exc)
        except Exception as exc:  # noqa: BLE001
            # Leave it unacked so it is redelivered and can be retried.
            log.exception("failed to handle %s: %s", event.event_id, exc)
            return

    await bus.ack(CONSUMER_GROUP, delivered.receipt)
