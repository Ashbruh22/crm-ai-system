"""Demo schema: deals, activities, scores, actions, processed_events

Adds the spec section 6 tables alongside the legacy ones. Nothing existing is
dropped here — the legacy opportunities/predictions/recommendations tables are
removed once their routers are ported (phases 3-5).

Design notes:
* Timestamps are real ``DateTime(timezone=True)``, not the ISO strings the
  legacy tables use, so score history can be ordered and range-queried in SQL.
* ``action_status`` is a VARCHAR + CHECK rather than a native PostgreSQL ENUM,
  so the same migration runs on SQLite in tests and adding a status later needs
  no ALTER TYPE.
* Primary keys use CHAR(36)/UUID via app.db.types.GUID, not the PostgreSQL
  dialect UUID the legacy models import.

Revision ID: 005_demo_schema
Revises: 004_drift
"""

from alembic import op
import sqlalchemy as sa

from app.db.types import GUID

revision = "005_demo_schema"
down_revision = "004_drift"
branch_labels = None
depends_on = None

ACTION_STATUSES = ("suggested", "accepted", "dismissed")


def upgrade() -> None:
    op.create_table(
        "deals",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("company", sa.String(200), nullable=False),
        sa.Column("industry", sa.String(64), nullable=False),
        sa.Column("region", sa.String(64), nullable=False),
        sa.Column("deal_size", sa.Numeric(14, 2), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("owner_rep", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expected_close", sa.DateTime(timezone=True)),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("won", sa.Boolean()),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_deals_stage", "deals", ["stage"])
    op.create_index("ix_deals_owner_rep", "deals", ["owner_rep"])

    op.create_table(
        "activities",
        sa.Column("id", sa.String(40), primary_key=True),
        sa.Column(
            "deal_id",
            sa.String(32),
            sa.ForeignKey("deals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON()),
        sa.Column("origin", sa.String(16), nullable=False, server_default="seed"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_activities_deal_occurred", "activities", ["deal_id", "occurred_at"]
    )

    op.create_table(
        "scores",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "deal_id",
            sa.String(32),
            sa.ForeignKey("deals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("win_prob", sa.Float(), nullable=False),
        sa.Column("days_to_close", sa.Float()),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("shap_top", sa.JSON()),
        sa.Column("feature_hash", sa.String(64)),
        sa.Column("latency_ms", sa.JSON()),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_scores_deal_scored", "scores", ["deal_id", "scored_at"])
    op.create_index("ix_scores_feature_hash", "scores", ["feature_hash"])

    op.create_table(
        "actions",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "deal_id",
            sa.String(32),
            sa.ForeignKey("deals.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("score_id", GUID(), sa.ForeignKey("scores.id", ondelete="SET NULL")),
        sa.Column("rule_id", sa.String(64), nullable=False),
        sa.Column("action_type", sa.String(64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("priority", sa.String(16), nullable=False),
        sa.Column("urgency_score", sa.Float()),
        sa.Column(
            "status",
            sa.Enum(
                *ACTION_STATUSES,
                name="action_status",
                native_enum=False,
                length=16,
            ),
            nullable=False,
            server_default="suggested",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_actions_deal_status", "actions", ["deal_id", "status"])

    op.create_table(
        "processed_events",
        sa.Column("event_id", sa.String(128), primary_key=True),
        sa.Column("deal_id", sa.String(32)),
        sa.Column("consumer", sa.String(64)),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_index("ix_processed_events_deal_id", "processed_events", ["deal_id"])


def downgrade() -> None:
    op.drop_table("processed_events")
    op.drop_index("ix_actions_deal_status", table_name="actions")
    op.drop_table("actions")
    op.drop_index("ix_scores_feature_hash", table_name="scores")
    op.drop_index("ix_scores_deal_scored", table_name="scores")
    op.drop_table("scores")
    op.drop_index("ix_activities_deal_occurred", table_name="activities")
    op.drop_table("activities")
    op.drop_index("ix_deals_owner_rep", table_name="deals")
    op.drop_index("ix_deals_stage", table_name="deals")
    op.drop_table("deals")
