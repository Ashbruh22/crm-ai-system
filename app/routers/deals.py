"""Pipeline endpoints (spec section 7).

``GET /api/deals``       paginated pipeline with each deal's latest score
``GET /api/deals/{id}``  one deal with activity timeline, score history, actions

Both are read-only and unauthenticated: the hosted demo is public, the data is
synthetic, and asking a reviewer to log in before they can see anything would
cost more than it protects. Write paths (scoring, events, action updates) keep
their auth and rate limits.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Action, ActionStatus, Activity, Deal, Score
from app.db.session import get_session

router = APIRouter()

SortKey = Literal["win_prob", "deal_size", "days_to_close", "created_at", "expected_close"]


def _deal_summary(deal: Deal, score: Score | None, top_action: Action | None) -> dict:
    return {
        "id": deal.id,
        "company": deal.company,
        "industry": deal.industry,
        "region": deal.region,
        "deal_size": float(deal.deal_size),
        "stage": deal.stage,
        "source": deal.source,
        "owner_rep": deal.owner_rep,
        "created_at": deal.created_at,
        "expected_close": deal.expected_close,
        "score": (
            {
                "win_prob": score.win_prob,
                "days_to_close": score.days_to_close,
                "model_version": score.model_version,
                "scored_at": score.scored_at,
            }
            if score
            else None
        ),
        "top_action": (
            {
                "id": str(top_action.id),
                "action_type": top_action.action_type,
                "reason": top_action.reason,
                "priority": top_action.priority,
                "status": top_action.status.value,
            }
            if top_action
            else None
        ),
    }


async def _latest_scores(session: AsyncSession, deal_ids: list[str]) -> dict[str, Score]:
    """Most recent score per deal, in one round trip.

    A correlated subquery per row would be N+1; this takes the max scored_at per
    deal and joins back to it.
    """
    if not deal_ids:
        return {}

    newest = (
        select(Score.deal_id, func.max(Score.scored_at).label("scored_at"))
        .where(Score.deal_id.in_(deal_ids))
        .group_by(Score.deal_id)
        .subquery()
    )
    rows = await session.execute(
        select(Score).join(
            newest,
            (Score.deal_id == newest.c.deal_id)
            & (Score.scored_at == newest.c.scored_at),
        )
    )
    return {s.deal_id: s for s in rows.scalars()}


async def _top_actions(session: AsyncSession, deal_ids: list[str]) -> dict[str, Action]:
    """Highest-priority still-suggested action per deal."""
    if not deal_ids:
        return {}

    priority_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    rows = await session.execute(
        select(Action)
        .where(
            Action.deal_id.in_(deal_ids),
            Action.status == ActionStatus.suggested,
        )
        .order_by(Action.deal_id, desc(Action.created_at))
    )

    best: dict[str, Action] = {}
    for action in rows.scalars():
        current = best.get(action.deal_id)
        if current is None or priority_rank.get(
            action.priority, 9
        ) < priority_rank.get(current.priority, 9):
            best[action.deal_id] = action
    return best


@router.get("")
async def list_deals(
    session: AsyncSession = Depends(get_session),
    stage: str | None = Query(None, description="filter to one pipeline stage"),
    owner_rep: str | None = Query(None),
    sort: SortKey = Query("win_prob"),
    order: Literal["asc", "desc"] = Query("desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """Open pipeline with each deal's latest score, paginated."""
    filters = [Deal.closed_at.is_(None)]
    if stage:
        filters.append(Deal.stage == stage)
    if owner_rep:
        filters.append(Deal.owner_rep == owner_rep)

    total = int(
        (
            await session.execute(select(func.count()).select_from(Deal).where(*filters))
        ).scalar_one()
    )

    # Sorting by score means ordering on another table, so those two keys are
    # applied in Python after the latest scores are fetched. Deal-column sorts
    # stay in SQL where they can use an index.
    score_sort = sort in ("win_prob", "days_to_close")
    query = select(Deal).where(*filters)
    if not score_sort:
        column = getattr(Deal, sort)
        query = query.order_by(desc(column) if order == "desc" else column)
        query = query.limit(limit).offset(offset)

    deals = list((await session.execute(query)).scalars())
    scores = await _latest_scores(session, [d.id for d in deals])
    actions = await _top_actions(session, [d.id for d in deals])

    if score_sort:
        def key(deal: Deal):
            score = scores.get(deal.id)
            if score is None:
                return float("-inf")
            return getattr(score, sort) or 0.0

        deals.sort(key=key, reverse=(order == "desc"))
        deals = deals[offset : offset + limit]

    return {
        "items": [_deal_summary(d, scores.get(d.id), actions.get(d.id)) for d in deals],
        "total": total,
        "limit": limit,
        "offset": offset,
        "sort": sort,
        "order": order,
        "synthetic_data": True,
    }


@router.get("/{deal_id}")
async def get_deal(
    deal_id: str,
    session: AsyncSession = Depends(get_session),
    history_limit: int = Query(50, ge=1, le=500),
    activity_limit: int = Query(200, ge=1, le=1000),
) -> dict:
    """One deal: timeline, latest score, score history, and actions."""
    deal = (
        await session.execute(select(Deal).where(Deal.id == deal_id))
    ).scalar_one_or_none()
    if deal is None:
        raise HTTPException(status_code=404, detail=f"deal {deal_id!r} not found")

    activities = list(
        (
            await session.execute(
                select(Activity)
                .where(Activity.deal_id == deal_id)
                .order_by(desc(Activity.occurred_at))
                .limit(activity_limit)
            )
        ).scalars()
    )

    history = list(
        (
            await session.execute(
                select(Score)
                .where(Score.deal_id == deal_id)
                .order_by(desc(Score.scored_at))
                .limit(history_limit)
            )
        ).scalars()
    )

    actions = list(
        (
            await session.execute(
                select(Action)
                .where(Action.deal_id == deal_id)
                .order_by(desc(Action.created_at))
            )
        ).scalars()
    )

    latest = history[0] if history else None

    return {
        "deal": {
            "id": deal.id,
            "company": deal.company,
            "industry": deal.industry,
            "region": deal.region,
            "deal_size": float(deal.deal_size),
            "stage": deal.stage,
            "source": deal.source,
            "owner_rep": deal.owner_rep,
            "created_at": deal.created_at,
            "expected_close": deal.expected_close,
            "closed_at": deal.closed_at,
        },
        "latest_score": (
            {
                "win_prob": latest.win_prob,
                "days_to_close": latest.days_to_close,
                "model_version": latest.model_version,
                "shap_top": latest.shap_top,
                "latency_ms": latest.latency_ms,
                "cache_hit": latest.cache_hit,
                "scored_at": latest.scored_at,
            }
            if latest
            else None
        ),
        # Oldest first so the dashboard can plot it without reversing.
        "score_history": [
            {
                "win_prob": s.win_prob,
                "days_to_close": s.days_to_close,
                "scored_at": s.scored_at,
            }
            for s in reversed(history)
        ],
        "timeline": [
            {
                "id": a.id,
                "type": a.type,
                "occurred_at": a.occurred_at,
                "origin": a.origin,
                "payload": a.payload,
            }
            for a in activities
        ],
        "actions": [
            {
                "id": str(a.id),
                "rule_id": a.rule_id,
                "action_type": a.action_type,
                "reason": a.reason,
                "priority": a.priority,
                "urgency_score": a.urgency_score,
                "status": a.status.value,
                "created_at": a.created_at,
                "resolved_at": a.resolved_at,
            }
            for a in actions
        ],
        "synthetic_data": True,
    }
