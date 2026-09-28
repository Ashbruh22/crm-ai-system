"""Kafka adapter — the paper's ingestion path, kept runnable locally.

    docker compose --profile kafka up

The peer-reviewed system ingests through Apache Kafka. This adapter keeps that
path alive behind the same ``MessageBus`` interface the hosted demo uses, so the
architecture in the paper and the architecture on Render differ by one
constructor call.

**Honesty note.** The original repository never contained Kafka code — "kafka"
appeared only in a docstring describing future work. This adapter was written
new for the demo, and it is exercised only against a local broker via the compose
profile; the automated test suite covers ``RedisStreamsBus``, which is what the
hosted demo actually runs. Treat it as a working reference implementation, not
as battle-tested production code.

``aiokafka`` is an optional dependency. Importing this module without it raises a
clear error rather than failing somewhere deeper.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.bus.base import DeliveredEvent, Event, MessageBus

log = logging.getLogger("crm_ai.bus.kafka")

DEFAULT_TOPIC = "crm.events"


class KafkaBus(MessageBus):
    """Kafka-backed bus. Requires ``pip install aiokafka``."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = DEFAULT_TOPIC,
        group_id: str = "crm-ai-scorer",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._producer = None
        self._consumer = None

    @staticmethod
    def _require_aiokafka():
        try:
            import aiokafka  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "KafkaBus needs aiokafka: pip install aiokafka. The hosted demo "
                "uses RedisStreamsBus instead; run the Kafka path with "
                "`docker compose --profile kafka up`."
            ) from exc
        return aiokafka

    async def connect(self) -> None:  # pragma: no cover - needs a broker
        aiokafka = self._require_aiokafka()

        self._producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers
        )
        await self._producer.start()

        self._consumer = aiokafka.AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            # Manual commit is the Kafka equivalent of xack: the offset moves
            # only once the event has actually been scored and persisted.
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        await self._consumer.start()

    async def close(self) -> None:  # pragma: no cover - needs a broker
        if self._producer is not None:
            await self._producer.stop()
        if self._consumer is not None:
            await self._consumer.stop()

    async def publish(self, event: Event) -> str:  # pragma: no cover
        if self._producer is None:
            raise RuntimeError("KafkaBus.connect() must be awaited first")
        import json

        metadata = await self._producer.send_and_wait(
            self.topic,
            json.dumps(event.to_wire()).encode(),
            # Partitioning by deal keeps one deal's events in order, which
            # matters because activities build a sequence.
            key=event.deal_id.encode(),
        )
        return f"{metadata.partition}:{metadata.offset}"

    async def consume(  # pragma: no cover - needs a broker
        self, group: str, consumer: str, block_ms: int = 5000, count: int = 10
    ) -> AsyncIterator[DeliveredEvent]:
        if self._consumer is None:
            raise RuntimeError("KafkaBus.connect() must be awaited first")
        import json

        batch = await self._consumer.getmany(timeout_ms=block_ms, max_records=count)
        for _partition, messages in batch.items():
            for message in messages:
                try:
                    raw = json.loads(message.value.decode())
                except (TypeError, ValueError):
                    log.warning("undecodable kafka message at %s", message.offset)
                    continue
                yield DeliveredEvent(Event.from_wire(raw), receipt=message)

    async def ack(self, group: str, receipt: Any) -> None:  # pragma: no cover
        if self._consumer is None:
            return
        from aiokafka import TopicPartition

        partition = TopicPartition(receipt.topic, receipt.partition)
        await self._consumer.commit({partition: receipt.offset + 1})

    async def pending_count(self, group: str) -> int:  # pragma: no cover
        # Kafka exposes consumer lag rather than a pending list; the demo's
        # health endpoint does not depend on this number.
        return 0
