"""Seeding the live demo set (spec section 6)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from app.seed import SYNTHETIC_DIR, count_deals, reset, seed, seed_if_empty

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not os.path.exists(os.path.join(SYNTHETIC_DIR, "live_deals.csv")),
        reason="synthetic live set not generated; run `make data`",
    ),
]


async def test_seed_loads_the_live_set(session):
    result = await seed(session)

    assert result["deals"] == 60
    assert result["activities"] > 0
    assert result["skipped_orphan_activities"] == 0
    assert await count_deals(session) == 60


async def test_seeded_deals_are_open_and_unlabelled(session):
    from sqlalchemy import select

    from app.db.models import Deal

    await seed(session)
    deals = list((await session.execute(select(Deal))).scalars())

    assert all(d.closed_at is None for d in deals)
    assert all(d.won is None for d in deals)
    assert all(d.stage not in ("Closed Won", "Closed Lost") for d in deals)


async def test_timeline_is_anchored_to_now_by_default(session):
    """Otherwise the demo ages: every deal looks abandoned a month later.

    The generator pins a fixed WORLD_NOW so its CSVs stay reproducible. Loading
    those dates verbatim means the whole pipeline drifts into "gone quiet" as
    real time passes, which would make every deal CRITICAL on a live deploy.
    """
    from sqlalchemy import func, select

    from app.db.models import Activity

    result = await seed(session)
    assert result["anchored_to_now"] is True

    newest = (
        await session.execute(select(func.max(Activity.occurred_at)))
    ).scalar_one()
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)

    age_days = (datetime.now(timezone.utc) - newest).total_seconds() / 86400.0
    assert age_days < 14, f"freshest activity is {age_days:.0f} days old"


async def test_anchoring_preserves_relative_history(session):
    """Shifting must move the whole timeline, not distort it."""
    from sqlalchemy import select

    from app.db.models import Activity, Deal

    await seed(session)

    deal = (await session.execute(select(Deal).limit(1))).scalars().first()
    activities = list(
        (
            await session.execute(
                select(Activity)
                .where(Activity.deal_id == deal.id)
                .order_by(Activity.occurred_at)
            )
        ).scalars()
    )
    if len(activities) < 2:
        pytest.skip("need a deal with at least two activities")

    created = deal.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    first = activities[0].occurred_at
    if first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)

    # Activities still follow the deal's creation, and remain ordered.
    assert first >= created
    stamps = [a.occurred_at for a in activities]
    assert stamps == sorted(stamps)


async def test_raw_dates_can_be_loaded_unshifted(session):
    result = await seed(session, anchor_to_now=False)
    assert result["anchored_to_now"] is False
    assert result["timeline_shift_days"] == 0


async def test_seed_if_empty_is_idempotent(session):
    first = await seed_if_empty(session)
    assert first["seeded"] is True

    second = await seed_if_empty(session)
    assert second["seeded"] is False
    assert second["existing_deals"] == 60
    assert await count_deals(session) == 60


async def test_reset_wipes_scores_and_actions_then_reloads(session):
    from sqlalchemy import func, select

    from app.db.models import Action, Score
    from tests.conftest import make_action, make_score

    await seed(session)
    session.add(make_score("LIVE-0001"))
    session.add(make_action("LIVE-0001"))
    await session.commit()

    result = await reset(session)

    assert result["reset"] is True
    assert result["deals"] == 60
    assert (await session.execute(select(func.count()).select_from(Score))).scalar_one() == 0
    assert (await session.execute(select(func.count()).select_from(Action))).scalar_one() == 0
    assert await count_deals(session) == 60
