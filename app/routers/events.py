"""Event ingestion (spec section 7).

``POST /api/events``    the demo's simulate button
``POST /webhooks/crm``  signed CRM ingestion

Both paths converge on the same bus publish, which is the point: the demo button
is not a special case that bypasses the pipeline, it is a CRM event with a
different source label.

Ingestion returns 202. The score is computed by the consumer and arrives at the
dashboard over SSE, so a slow model never blocks the webhook sender.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.bus.base import Event
from app.config import settings
from app.db.models import Deal
from app.db.session import get_session
from app.features.build import ACTIVITY_TYPES
from app.security.webhook import SignatureError, verify

log = logging.getLogger("crm_ai.events")

router = APIRouter()
webhook_router = APIRouter()


class EventIn(BaseModel):
    """A simulated CRM activity."""

    deal_id: str = Field(max_length=32)
    type: str
    occurred_at: datetime | None = None
    payload: dict = Field(default_factory=dict)
    #: Supply a stable id to make retries idempotent; generated otherwise.
    event_id: str | None = Field(default=None, max_length=128)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in ACTIVITY_TYPES:
            raise ValueError(
                f"unknown activity type {v!r}. Allowed: {list(ACTIVITY_TYPES)}"
            )
        return v


def get_bus(request: Request):
    bus = getattr(request.app.state, "bus", None)
    if bus is None:
        raise HTTPException(status_code=503, detail="event bus unavailable")
    return bus


async def _publish(bus, event: Event) -> dict:
    entry_id = await bus.publish(event)
    return {
        "accepted": True,
        "event_id": event.event_id,
        "deal_id": event.deal_id,
        "type": event.type,
        "stream_id": entry_id,
        "note": "queued for scoring; watch GET /api/stream for the result",
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def simulate_event(
    body: EventIn,
    response: Response,
    bus=Depends(get_bus),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Publish a simulated CRM event (the dashboard's demo buttons)."""
    if not settings.CRM_SIMULATE_ENABLED:
        raise HTTPException(status_code=403, detail="event simulation is disabled")

    # Fail fast on an unknown deal rather than letting the consumer drop it,
    # so the dashboard button reports the problem immediately.
    if await session.get(Deal, body.deal_id) is None:
        raise HTTPException(status_code=404, detail=f"deal {body.deal_id!r} not found")

    event = Event(
        deal_id=body.deal_id,
        type=body.type,
        occurred_at=body.occurred_at or datetime.now(timezone.utc),
        payload=body.payload,
        source="simulate",
        **({"event_id": body.event_id} if body.event_id else {}),
    )
    result = await _publish(bus, event)
    response.status_code = status.HTTP_202_ACCEPTED
    return result


@webhook_router.post("/crm", status_code=status.HTTP_202_ACCEPTED)
async def crm_webhook(
    request: Request,
    response: Response,
    bus=Depends(get_bus),
) -> dict:
    """Signed CRM webhook.

    Requires ``X-CRM-Signature`` (HMAC-SHA256 of ``{timestamp}.{body}``) and
    ``X-CRM-Timestamp``. A bad or stale signature is 401.
    """
    # The raw body must be read before parsing: signatures cover the exact
    # bytes sent, and re-serialising parsed JSON changes them.
    raw = await request.body()

    try:
        verify(
            raw,
            request.headers.get("X-CRM-Signature"),
            request.headers.get("X-CRM-Timestamp"),
            settings.CRM_WEBHOOK_SECRET,
            tolerance_seconds=settings.WEBHOOK_TIMESTAMP_TOLERANCE,
        )
    except SignatureError as exc:
        log.warning("rejected webhook: %s", exc)
        raise HTTPException(status_code=401, detail=str(exc))

    import json

    try:
        body = json.loads(raw or b"{}")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="body is not valid JSON")

    try:
        parsed = EventIn.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc))

    event = Event(
        deal_id=parsed.deal_id,
        type=parsed.type,
        occurred_at=parsed.occurred_at or datetime.now(timezone.utc),
        payload=parsed.payload,
        source="webhook:crm",
        # Delivery id if the sender provides one, so redelivery is idempotent.
        **(
            {"event_id": parsed.event_id}
            if parsed.event_id
            else (
                {"event_id": request.headers.get("X-CRM-Delivery-Id")}
                if request.headers.get("X-CRM-Delivery-Id")
                else {}
            )
        ),
    )
    result = await _publish(bus, event)
    response.status_code = status.HTTP_202_ACCEPTED
    return result
