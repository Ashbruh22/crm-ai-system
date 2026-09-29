"""Action ledger endpoints (spec section 7).

``PATCH /api/actions/{id}``  accept or dismiss a recommendation
``GET   /api/actions``       the open queue across the pipeline
``GET   /api/actions/rules`` the rule catalogue, for the architecture page
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.nba import PRIORITY_RANK, RULES
from app.db.models import Action, ActionStatus, Deal
from app.db.session import get_session

router = APIRouter()


class ActionUpdate(BaseModel):
    """Only the two terminal states are settable.

    ``suggested`` is not accepted: an action cannot be un-resolved through the
    API, because the ledger is an audit trail of what the user actually did.
    """

    status: Literal["accepted", "dismissed"]


def _serialise(action: Action, deal: Deal | None = None) -> dict:
    payload = {
        "id": str(action.id),
        "deal_id": action.deal_id,
        "rule_id": action.rule_id,
        "action_type": action.action_type,
        "reason": action.reason,
        "priority": action.priority,
        "urgency_score": action.urgency_score,
        "status": action.status.value,
        "created_at": action.created_at,
        "resolved_at": action.resolved_at,
    }
    if deal is not None:
        payload["company"] = deal.company
        payload["deal_size"] = float(deal.deal_size)
        payload["stage"] = deal.stage
        payload["owner_rep"] = deal.owner_rep
    return payload


@router.get("")
async def list_actions(
    session: AsyncSession = Depends(get_session),
    status: Literal["suggested", "accepted", "dismissed"] | None = Query(None),
    deal_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    """The action queue, most urgent first."""
    filters = []
    if status:
        filters.append(Action.status == ActionStatus(status))
    if deal_id:
        filters.append(Action.deal_id == deal_id)

    rows = list(
        (
            await session.execute(
                select(Action, Deal)
                .join(Deal, Deal.id == Action.deal_id)
                .where(*filters)
                .order_by(desc(Action.created_at))
            )
        ).all()
    )

    # Priority is a label, not a sortable column; rank it in Python.
    rows.sort(key=lambda r: (PRIORITY_RANK.get(r[0].priority, 9), -(r[0].urgency_score or 0)))

    return {
        "items": [_serialise(action, deal) for action, deal in rows[:limit]],
        "total": len(rows),
        "synthetic_data": True,
    }


@router.get("/rules")
async def list_rules() -> dict:
    """The rule catalogue behind every recommendation.

    Exposed so the dashboard can show *which* rules exist, not just which fired
    — the point of section 8 being an auditable decision tree rather than a
    black box.
    """
    return {
        "rules": [
            {
                "id": rule.id,
                "action_type": rule.action_type,
                "priority": rule.priority,
                "description": rule.description,
            }
            for rule in RULES
        ],
        "count": len(RULES),
    }


@router.patch("/{action_id}")
async def update_action(
    action_id: str,
    body: ActionUpdate,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Accept or dismiss a recommendation."""
    action = (
        await session.execute(select(Action).where(Action.id == action_id))
    ).scalar_one_or_none()
    if action is None:
        raise HTTPException(status_code=404, detail=f"action {action_id!r} not found")

    if action.status != ActionStatus.suggested:
        raise HTTPException(
            status_code=409,
            detail=(
                f"action is already {action.status.value}; "
                "resolved actions are immutable"
            ),
        )

    action.status = ActionStatus(body.status)
    action.resolved_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(action)

    return _serialise(action)
