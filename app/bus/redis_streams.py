"""Redis Streams adapter — the hosted demo's ingestion path (spec section 1).

Chosen over Kafka because Upstash offers a usable free tier and Kafka does not.
Streams give the two properties that actually matter here: a consumer group that
survives a restart, and explicit acknowledgement so an event is not lost if the
process dies mid-score.

Works against Upstash and a local ``redis:7-alpine`` alike.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.bus.base import DeliveredEvent, Event, MessageBus

log = logging.getLogger("crm_ai.bus")

DEFAULT_STREAM = "crm:events"

#: Cap the stream so a long-running demo cannot grow without bound. Approximate
#: trimming (~) lets Redis trim on whole nodes, which is much cheaper.
MAX_STREAM_LEN = 10_000

#: How long an entry must sit unacknowledged before another consumer may claim
#: it. Long enough that a healthy consumer mid-score is never robbed of its
#: work; short enough that a crashed one's backlog is picked up promptly.
CLAIM_MIN_IDLE_MS = 60_000


class StreamsUnsupported(RuntimeError):
    """The connected Redis is too old to support streams."""


class RedisStreamsBus(MessageBus):
    def __init__(
        self,
        redis,
        stream: str = DEFAULT_STREAM,
        maxlen: int = MAX_STREAM_LEN,
        claim_min_idle_ms: int = CLAIM_MIN_IDLE_MS,
    ):
        self.redis = redis
        self.stream = stream
        self.maxlen = maxlen
        self.claim_min_idle_ms = claim_min_idle_ms
        self._groups: set[str] = set()

    async def connect(self) -> None:
        """Ping, then confirm the server actually speaks streams.

        Streams arrived in Redis 5.0. Older servers answer PING and handle
        GET/SETEX/PUBLISH perfectly well, so the cache and the SSE fan-out look
        healthy while every XADD fails with a bare "unknown command 'XGROUP'"
        from inside the consumer loop. Checking once at startup turns that into
        one actionable message instead of an error every two seconds.
        """
        await self.redis.ping()
        await self.assert_streams_supported()

    async def assert_streams_supported(self) -> str:
        """Raise StreamsUnsupported unless the server is Redis >= 5.

        A server that refuses INFO but otherwise works is fine — some managed
        hosts restrict it — so that case returns "unknown" and lets startup
        continue. A server we cannot reach or authenticate against is not fine,
        and those errors propagate: swallowing them reported a healthy bus over
        a Redis that was rejecting every command, which is precisely the
        misleading signal this check exists to prevent.
        """
        from redis.exceptions import AuthenticationError, ConnectionError, TimeoutError

        try:
            info = await self.redis.info("server")
        except (AuthenticationError, ConnectionError, TimeoutError):
            raise
        except Exception:  # noqa: BLE001 - INFO restricted, but the server works
            return "unknown"

        version = str(info.get("redis_version", "")) or "unknown"
        try:
            major = int(version.split(".")[0])
        except (ValueError, IndexError):
            return version

        if major < 5:
            raise StreamsUnsupported(
                f"Redis {version} does not support streams (added in 5.0). "
                "The event bus needs XADD/XGROUP/XREADGROUP. Use redis:7-alpine "
                "(`docker compose up redis`) or an Upstash instance. The score "
                "cache works on older servers, which is why PING passes."
            )
        return version

    async def close(self) -> None:
        # The Redis client is owned by the app lifespan, not by the bus.
        return None

    async def ensure_group(self, group: str) -> None:
        """Create the consumer group, tolerating races and restarts.

        ``mkstream=True`` creates the stream if the first consumer starts before
        the first publisher. ``id="0"`` starts at the beginning so events
        published before the group existed are still delivered.
        """
        if group in self._groups:
            return
        try:
            await self.redis.xgroup_create(
                name=self.stream, groupname=group, id="0", mkstream=True
            )
        except Exception as exc:  # noqa: BLE001
            # BUSYGROUP means another worker created it first, which is fine.
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups.add(group)

    async def publish(self, event: Event) -> str:
        entry_id = await self.redis.xadd(
            self.stream, event.to_wire(), maxlen=self.maxlen, approximate=True
        )
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    @staticmethod
    def _decode(fields: dict) -> dict:
        return {
            (k.decode() if isinstance(k, bytes) else k): v for k, v in fields.items()
        }

    async def _reclaim(
        self, group: str, consumer: str, min_idle_ms: int, count: int
    ) -> AsyncIterator[DeliveredEvent]:
        """Take over entries a dead consumer never acknowledged.

        ``XAUTOCLAIM`` rather than ``XREADGROUP`` with id ``0``: the latter only
        returns *this* consumer's pending list, so a process that restarts under
        a different consumer name would leave the previous name's backlog
        stranded forever. XAUTOCLAIM reclaims across consumers, which is what
        makes "a restart resumes where it left off" actually true.

        ``min_idle_ms`` stops a healthy consumer mid-score from being robbed.
        """
        # Exactly one batch per call, deliberately not a loop.
        #
        # XAUTOCLAIM returns a cursor to continue from, but the value is not
        # uniformly "the next id to scan" across implementations — paging on it
        # can hand back the same entry and spin forever. One bounded batch
        # terminates unconditionally, and the consumer's next iteration picks up
        # whatever is left, so a large backlog still drains.
        try:
            result = await self.redis.xautoclaim(
                name=self.stream,
                groupname=group,
                consumername=consumer,
                min_idle_time=min_idle_ms,
                start_id="0-0",
                count=count,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("xautoclaim unavailable: %s", exc)
            return

        if not result or len(result) < 2:
            return

        for entry_id, fields in result[1] or []:
            if not fields:
                await self.ack(group, entry_id)
                continue
            yield DeliveredEvent(Event.from_wire(self._decode(fields)), receipt=entry_id)

    async def consume(
        self,
        group: str,
        consumer: str,
        block_ms: int = 5000,
        count: int = 10,
        claim_min_idle_ms: int | None = None,
    ) -> AsyncIterator[DeliveredEvent]:
        """Yield events for this group: stranded entries first, then new ones."""
        await self.ensure_group(group)

        idle = self.claim_min_idle_ms if claim_min_idle_ms is None else claim_min_idle_ms
        async for delivered in self._reclaim(group, consumer, idle, count):
            yield delivered

        # Drain what is already queued without blocking, and only then wait.
        #
        # Two reasons, not one. A non-blocking read returns immediately when
        # there is a backlog, so a burst is not paced at one batch per block
        # interval. And fakeredis — which the test suite runs against — returns
        # nothing at all from a *blocking* xreadgroup even when entries are
        # waiting, so a block-only implementation would look correct in tests
        # for the wrong reason.
        try:
            response = await self.redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={self.stream: ">"},
                count=count,
            )
            if not response or not response[0][1]:
                if block_ms > 0:
                    response = await self.redis.xreadgroup(
                        groupname=group,
                        consumername=consumer,
                        streams={self.stream: ">"},
                        count=count,
                        block=block_ms,
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("xreadgroup failed: %s", exc)
            return

        for _stream, entries in response or []:
            for entry_id, fields in entries:
                if not fields:
                    # A trimmed entry leaves a tombstone; ack and move on.
                    await self.ack(group, entry_id)
                    continue
                yield DeliveredEvent(
                    Event.from_wire(self._decode(fields)), receipt=entry_id
                )

    async def ack(self, group: str, receipt: Any) -> None:
        await self.redis.xack(self.stream, group, receipt)

    async def pending_count(self, group: str) -> int:
        try:
            info = await self.redis.xpending(self.stream, group)
        except Exception:  # noqa: BLE001
            return 0
        if isinstance(info, dict):
            return int(info.get("pending", 0))
        return int(info[0]) if info else 0

    async def length(self) -> int:
        try:
            return int(await self.redis.xlen(self.stream))
        except Exception:  # noqa: BLE001
            return 0
