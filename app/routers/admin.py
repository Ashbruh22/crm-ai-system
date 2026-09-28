"""Admin endpoints (spec section 7).

``POST /api/admin/reset``  wipe the demo tables and reload the live set
``GET  /api/admin/status`` what the reset would affect, without doing it

Reset exists because a public demo accumulates whatever visitors do to it:
scores, dismissed recommendations, simulated events. A nightly run puts the
pipeline back to a clean sixty deals, and re-anchors their timeline to now so
the board does not drift into looking abandoned.

Destructive, so it is token-gated *and* rate-limited, and it reports what it
removed rather than answering with a bare 200.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Action, Activity, Deal, ProcessedEvent, Score
from app.db.session import get_session
from app.security.admin import require_admin
from app.security.ratelimit import ADMIN_LIMIT, limiter
from app.services.scoring import CACHE_PREFIX

log = logging.getLogger("crm_ai.admin")

router = APIRouter()


async def _counts(session: AsyncSession) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, model in (
        ("deals", Deal),
        ("activities", Activity),
        ("scores", Score),
        ("actions", Action),
        ("processed_events", ProcessedEvent),
    ):
        out[name] = int(
            (await session.execute(select(func.count()).select_from(model))).scalar_one()
        )
    return out


@router.get("/status")
async def admin_status(
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict:
    """Row counts, so an operator can see what a reset would clear."""
    return {"counts": await _counts(session), "synthetic_data": True}


@router.post("/reset")
@limiter.limit(ADMIN_LIMIT)
async def admin_reset(
    request: Request,
    session: AsyncSession = Depends(get_session),
    _: str = Depends(require_admin),
) -> dict:
    """Wipe the demo tables and reload the synthetic live set."""
    before = await _counts(session)

    from app.seed import reset

    result = await reset(session, settings.SYNTHETIC_DATA_DIR)

    # Cached scores refer to rows that no longer exist.
    cleared = 0
    redis = getattr(request.app.state, "redis", None)
    if redis is not None:
        try:
            async for key in redis.scan_iter(match=f"{CACHE_PREFIX}:*"):
                await redis.delete(key)
                cleared += 1
        except Exception as exc:  # noqa: BLE001 - a stale cache is not fatal
            log.warning("cache clear during reset failed: %s", exc)

    after = await _counts(session)
    log.info("demo reset: %s -> %s", before, after)

    return {
        "reset": True,
        "before": before,
        "after": after,
        "cache_keys_cleared": cleared,
        "timeline_shift_days": result.get("timeline_shift_days"),
        "synthetic_data": True,
    }
