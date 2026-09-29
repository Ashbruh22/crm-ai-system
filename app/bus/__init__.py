"""Ingestion bus (spec section 1).

The original design ingests through Apache Kafka. Kafka has no reasonable free
hosting, so the hosted demo runs on Redis Streams instead — behind a
``MessageBus`` interface, so the choice is a deployment detail rather than a
rewrite.

* ``base.py``          the interface and the event envelope
* ``redis_streams.py`` the hosted demo path (Upstash-compatible)
* ``kafka.py``         the Kafka adapter, local only, `--profile kafka`
* ``broadcast.py``     fan-out to connected dashboards over Redis pub/sub
"""

from app.bus.base import Event, MessageBus

__all__ = ["Event", "MessageBus"]
