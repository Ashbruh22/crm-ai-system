"""Load the synthetic live set into the database (spec section 6).

Run standalone:

    python -m app.seed            # no-op if deals already exist
    python -m app.seed --force    # wipe demo tables and reload

The service also calls :func:`seed_if_empty` at startup so a fresh deploy has a
pipeline to show without a manual step.

Only ``data/synthetic/live_deals.csv`` and ``live_activities.csv`` are read —
the 60 open deals. The 2,000 closed training deals stay out of the demo
database; they are training input, not pipeline.

Timeline anchoring
------------------
The generator pins its world to a fixed ``WORLD_NOW`` so the CSVs are
reproducible from a seed. Loaded verbatim, though, every deal ages in real time:
a month after generation the whole pipeline looks abandoned and every deal draws
a "gone quiet" recommendation. Seeding therefore shifts every timestamp by
``now - world_now``, preserving the relative shape of each deal's history while
keeping the demo current. Set ``anchor_to_now=False`` to load the raw dates.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Action, Activity, Deal, ProcessedEvent, Score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(REPO_ROOT, "data", "synthetic")


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO timestamp from the generator into aware UTC."""
    if value in (None, "", "None"):
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _world_now(data_dir: str) -> datetime | None:
    """The generator's fixed 'now', from the manifest it writes alongside."""
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh).get("world_now")
    return _parse_dt(raw) if raw else None


def _offset(data_dir: str, anchor_to_now: bool) -> timedelta:
    if not anchor_to_now:
        return timedelta(0)
    world_now = _world_now(data_dir)
    if world_now is None:
        return timedelta(0)
    return datetime.now(timezone.utc) - world_now


def _shift(value: datetime | None, delta: timedelta) -> datetime | None:
    return None if value is None else value + delta


def load_live_rows(data_dir: str = SYNTHETIC_DIR) -> tuple[list[dict], list[dict]]:
    deals_path = os.path.join(data_dir, "live_deals.csv")
    acts_path = os.path.join(data_dir, "live_activities.csv")
    for p in (deals_path, acts_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run `make data` (or .\\train.ps1) to generate it."
            )
    return _read_csv(deals_path), _read_csv(acts_path)


async def count_deals(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count()).select_from(Deal))).scalar_one())


async def clear_demo_tables(session: AsyncSession) -> None:
    """Drop demo rows in FK-safe order. Leaves the legacy tables alone."""
    for model in (Action, Score, Activity, ProcessedEvent, Deal):
        await session.execute(delete(model))
    await session.commit()


async def seed(
    session: AsyncSession,
    data_dir: str = SYNTHETIC_DIR,
    anchor_to_now: bool = True,
) -> dict:
    """Insert the live set. Assumes the demo tables are empty."""
    deal_rows, activity_rows = load_live_rows(data_dir)
    delta = _offset(data_dir, anchor_to_now)

    for row in deal_rows:
        session.add(
            Deal(
                id=row["id"],
                company=row["company"],
                industry=row["industry"],
                region=row["region"],
                deal_size=Decimal(row["deal_size"]),
                stage=row["stage"],
                source=row["source"],
                owner_rep=row["owner_rep"],
                created_at=_shift(_parse_dt(row["created_at"]), delta),
                expected_close=_shift(_parse_dt(row.get("expected_close")), delta),
                closed_at=_shift(_parse_dt(row.get("closed_at")), delta),
                # Live deals are open and therefore unlabelled.
                won=None,
            )
        )

    known = {r["id"] for r in deal_rows}
    skipped = 0
    for row in activity_rows:
        if row["deal_id"] not in known:
            skipped += 1
            continue
        session.add(
            Activity(
                id=row["id"],
                deal_id=row["deal_id"],
                type=row["type"],
                occurred_at=_shift(_parse_dt(row["occurred_at"]), delta),
                origin="seed",
            )
        )

    await session.commit()
    return {
        "deals": len(deal_rows),
        "activities": len(activity_rows) - skipped,
        "skipped_orphan_activities": skipped,
        "anchored_to_now": anchor_to_now,
        "timeline_shift_days": round(delta.total_seconds() / 86400.0, 2),
    }


async def seed_if_empty(
    session: AsyncSession,
    data_dir: str = SYNTHETIC_DIR,
    anchor_to_now: bool = True,
) -> dict:
    """Seed only when there is nothing there. Safe to call on every boot."""
    existing = await count_deals(session)
    if existing:
        return {"seeded": False, "existing_deals": existing}
    result = await seed(session, data_dir, anchor_to_now)
    return {"seeded": True, **result}


async def reset(
    session: AsyncSession,
    data_dir: str = SYNTHETIC_DIR,
    anchor_to_now: bool = True,
) -> dict:
    """Wipe and reload — what POST /api/admin/reset and the nightly job use.

    Re-anchoring on every reset is why the nightly job keeps the demo looking
    live rather than progressively abandoned.
    """
    await clear_demo_tables(session)
    result = await seed(session, data_dir, anchor_to_now)
    return {"reset": True, **result}


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="wipe demo tables and reload"
    )
    parser.add_argument("--data-dir", default=SYNTHETIC_DIR)
    args = parser.parse_args()

    from app.db.session import SessionLocal, engine

    async with SessionLocal() as session:
        if args.force:
            result = await reset(session, args.data_dir)
        else:
            result = await seed_if_empty(session, args.data_dir)
    await engine.dispose()

    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(_main())
