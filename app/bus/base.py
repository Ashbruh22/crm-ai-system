"""MessageBus interface and event envelope (spec section 1).

The interface exists so that "Kafka or Redis Streams" is a deployment decision,
not an architectural one. Both adapters implement the same three operations, and
``consumer.py`` never learns which one it is talking to.

Consumer-group semantics are part of the contract, not an implementation detail:
both backends must let a restarted process resume from where it stopped rather
than replaying the stream or skipping what it missed.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Event:
    """One CRM activity, on its way to being scored.

    ``event_id`` is the idempotency key. It is supplied by the sender when the
    sender has a stable id (a CRM webhook delivery id, say) and generated here
    otherwise. The consumer records it in ``processed_events`` and skips repeats,
    which is what makes redelivery safe.
    """

    deal_id: str
    type: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=_now)
    payload: dict[str, Any] = field(default_factory=dict)
    #: Where it came from: "simulate", "webhook:salesforce", "webhook:hubspot".
    source: str = "unknown"

    # --- wire format ------------------------------------------------------

    def to_wire(self) -> dict[str, str]:
        """Flatten to the string-only mapping stream backends accept."""
        return {
            "event_id": self.event_id,
            "deal_id": self.deal_id,
            "type": self.type,
            "occurred_at": self.occurred_at.isoformat(),
            "source": self.source,
            "payload": json.dumps(self.payload or {}),
        }

    @classmethod
    def from_wire(cls, raw: dict) -> "Event":
        def _get(key: str, default: str = "") -> str:
            value = raw.get(key, default)
            # redis-py returns bytes unless decode_responses is set.
            return value.decode() if isinstance(value, bytes) else str(value)

        occurred = _get("occurred_at")
        try:
            occurred_at = datetime.fromisoformat(occurred) if occurred else _now()
        except ValueError:
            occurred_at = _now()
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)

        try:
            payload = json.loads(_get("payload", "{}") or "{}")
        except (TypeError, ValueError):
            payload = {}

        return cls(
            event_id=_get("event_id") or str(uuid.uuid4()),
            deal_id=_get("deal_id"),
            type=_get("type"),
            occurred_at=occurred_at,
            payload=payload,
            source=_get("source", "unknown"),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["occurred_at"] = self.occurred_at.isoformat()
        return data


@dataclass
class DeliveredEvent:
    """An event plus whatever the backend needs to acknowledge it."""

    event: Event
    #: Opaque backend handle (Redis stream id, Kafka offset).
    receipt: Any


class MessageBus(ABC):
    """Publish/consume with at-least-once delivery and consumer groups."""

    @abstractmethod
    async def connect(self) -> None:
        """Open connections and create the consumer group if absent."""

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def publish(self, event: Event) -> str:
        """Append an event. Returns the backend's id for it."""

    @abstractmethod
    async def consume(
        self, group: str, consumer: str, block_ms: int = 5000, count: int = 10
    ) -> AsyncIterator[DeliveredEvent]:
        """Yield undelivered events for this consumer group."""

    @abstractmethod
    async def ack(self, group: str, receipt: Any) -> None:
        """Mark an event handled so it is not redelivered."""

    @abstractmethod
    async def pending_count(self, group: str) -> int:
        """Events delivered to the group but not yet acknowledged."""
