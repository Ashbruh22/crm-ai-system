"""Fan-out to connected dashboards (spec section 7, ``GET /api/stream``).

Uses Redis pub/sub rather than an in-process list of queues. On one Render
service either would work, but pub/sub means a second worker — or the consumer
running in a separate process — still reaches every open dashboard. It also
costs nothing extra: the Redis connection is already there for the cache.

Delivery is best-effort by design. A dashboard that is not connected when a
score lands simply does not see that notification; it fetches current state when
it next loads. Nothing here is a system of record — the database is.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

log = logging.getLogger("crm_ai.broadcast")

CHANNEL = "crm:stream"

#: Event names the dashboard listens for.
SCORE_UPDATED = "score_updated"
ACTION_CREATED = "action_created"
EVENT_RECEIVED = "event_received"


class Broadcaster:
    """Publishes UI notifications and yields them back to SSE connections."""

    def __init__(self, redis, channel: str = CHANNEL):
        self.redis = redis
        self.channel = channel

    async def publish(self, event_type: str, data: dict) -> None:
        if self.redis is None:
            return
        message = {
            "event": event_type,
            "data": data,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await self.redis.publish(self.channel, json.dumps(message, default=str))
        except Exception as exc:  # noqa: BLE001 - never fail scoring on fan-out
            log.warning("broadcast failed: %s", exc)

    async def subscribe(self) -> AsyncIterator[dict]:
        """Yield messages until the caller stops iterating."""
        if self.redis is None:
            return

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.channel)
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is None:
                    # Nothing to send; let the caller emit a keep-alive.
                    yield {}
                    continue

                raw: Any = message.get("data")
                if isinstance(raw, bytes):
                    raw = raw.decode()
                try:
                    yield json.loads(raw)
                except (TypeError, ValueError):
                    continue
        finally:
            try:
                await pubsub.unsubscribe(self.channel)
                await pubsub.aclose()
            except Exception:  # noqa: BLE001
                pass


def sse_format(event_type: str, data: dict) -> str:
    """Render one Server-Sent Event frame.

    Newlines inside the payload would terminate the frame early, so the JSON is
    serialised without them.
    """
    body = json.dumps(data, default=str).replace("\n", " ")
    return f"event: {event_type}\ndata: {body}\n\n"


def sse_comment(text: str = "keep-alive") -> str:
    """A comment frame. Ignored by EventSource but keeps proxies from timing out."""
    return f": {text}\n\n"
