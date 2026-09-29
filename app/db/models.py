"""Demo schema (spec section 6).

Five tables:

* ``deals``            the pipeline, seeded from the synthetic live set
* ``activities``       append-only event log, indexed ``(deal_id, occurred_at)``
* ``scores``           append-only score history, so score-over-time can be charted
* ``actions``          the next-best-action ledger with an audit trail
* ``processed_events`` consumer idempotency

Two deliberate departures from the legacy models in ``app/models/db.py``:

1. **Real ``DateTime`` columns, not ``String``.** The legacy tables store every
   timestamp as an ISO string, which cannot be range-queried or ordered
   correctly in SQL and makes the score-history chart unreliable.
2. **Portable ID types** via ``app.db.types.GUID`` instead of the PostgreSQL
   ``UUID`` dialect type.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Shared with the legacy models so there is one metadata and one migration
# history for the whole service.
from app.models.db import Base
from app.db.types import GUID, new_uuid


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use naive datetimes in the schema."""
    return datetime.now(timezone.utc)


class ActionStatus(str, enum.Enum):
    """Lifecycle of a recommendation (spec section 6)."""

    suggested = "suggested"
    accepted = "accepted"
    dismissed = "dismissed"


#: ``native_enum=False`` renders VARCHAR + CHECK rather than a PostgreSQL ENUM
#: type, which keeps SQLite test runs working and avoids an ALTER TYPE migration
#: every time a status is added.
_action_status = SAEnum(
    ActionStatus,
    name="action_status",
    native_enum=False,
    values_callable=lambda e: [m.value for m in e],
    length=16,
)


class Deal(Base):
    """An opportunity in the pipeline.

    The primary key is the CRM's own deal id (e.g. ``LIVE-0007``) rather than a
    surrogate UUID: it is what webhooks arrive with and what the dashboard URLs
    show, so a join-free lookup is the common case.
    """

    __tablename__ = "deals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    company: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str] = mapped_column(String(64), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    deal_size: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_rep: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expected_close: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    #: NULL while the deal is open. Set only on historical/closed deals.
    won: Mapped[bool | None] = mapped_column(Boolean)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    activities: Mapped[list["Activity"]] = relationship(
        back_populates="deal", cascade="all, delete-orphan", passive_deletes=True
    )
    scores: Mapped[list["Score"]] = relationship(
        back_populates="deal", cascade="all, delete-orphan", passive_deletes=True
    )
    actions: Mapped[list["Action"]] = relationship(
        back_populates="deal", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<Deal {self.id} {self.company!r} {self.stage}>"


class Activity(Base):
    """One CRM event. Append-only; never updated in place."""

    __tablename__ = "activities"
    __table_args__ = (
        # The feature builder always reads one deal's events in time order.
        Index("ix_activities_deal_occurred", "deal_id", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    deal_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("deals.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    payload: Mapped[dict | None] = mapped_column(JSON)
    #: Where the event came from: seed, webhook, or the demo simulate button.
    origin: Mapped[str] = mapped_column(String(16), default="seed", nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    deal: Mapped[Deal] = relationship(back_populates="activities")

    def __repr__(self) -> str:
        return f"<Activity {self.type} {self.deal_id} @{self.occurred_at:%Y-%m-%d}>"


class Score(Base):
    """One scoring result. Append-only, so score history can be charted."""

    __tablename__ = "scores"
    __table_args__ = (Index("ix_scores_deal_scored", "deal_id", "scored_at"),)

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    deal_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("deals.id", ondelete="CASCADE"), nullable=False
    )
    win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    days_to_close: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    #: Top-k SHAP contributions: [{feature, label, value, shap}, ...]
    shap_top: Mapped[list | None] = mapped_column(JSON)
    #: Stable hash of the feature vector; the Redis cache key is derived from it.
    feature_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    #: Per-stage timings: {features, xgb, lstm, shap, total}
    latency_ms: Mapped[dict | None] = mapped_column(JSON)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    deal: Mapped[Deal] = relationship(back_populates="scores")
    actions: Mapped[list["Action"]] = relationship(back_populates="score")

    def __repr__(self) -> str:
        return f"<Score {self.deal_id} p={self.win_prob:.3f}>"


class Action(Base):
    """A recommended next-best action, and what the user did with it.

    ``rule_id`` records which rule in ``app/agent/nba.py`` fired, which is what
    makes this table an audit trail rather than a list of suggestions.
    """

    __tablename__ = "actions"
    __table_args__ = (Index("ix_actions_deal_status", "deal_id", "status"),)

    id: Mapped[str] = mapped_column(GUID, primary_key=True, default=new_uuid)
    deal_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("deals.id", ondelete="CASCADE"), nullable=False
    )
    score_id: Mapped[str | None] = mapped_column(
        GUID, ForeignKey("scores.id", ondelete="SET NULL")
    )

    rule_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Human-readable and citing the drivers, per spec section 8.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False)
    urgency_score: Mapped[float | None] = mapped_column(Float)

    status: Mapped[ActionStatus] = mapped_column(
        _action_status, default=ActionStatus.suggested, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    #: When the user accepted or dismissed it. NULL while still suggested.
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    deal: Mapped[Deal] = relationship(back_populates="actions")
    score: Mapped[Score | None] = relationship(back_populates="actions")

    def __repr__(self) -> str:
        return f"<Action {self.rule_id} {self.deal_id} {self.status.value}>"


class ProcessedEvent(Base):
    """Idempotency ledger: the consumer skips any event id already here.

    Without this, a redelivered webhook or a stream message re-read after a
    restart would append a duplicate activity and re-score the deal.
    """

    __tablename__ = "processed_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    deal_id: Mapped[str | None] = mapped_column(String(32), index=True)
    #: Which bus/consumer handled it, for debugging redelivery.
    consumer: Mapped[str | None] = mapped_column(String(64))
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    def __repr__(self) -> str:
        return f"<ProcessedEvent {self.event_id}>"
