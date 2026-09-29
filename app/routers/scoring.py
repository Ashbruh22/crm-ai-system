"""Scoring, explanation and what-if endpoints (spec section 7).

Mounted under ``/api/deals`` alongside ``deals.py``:

``POST /api/deals/{id}/score``     force a re-score, with latency breakdown
``GET  /api/deals/{id}/explain``   full SHAP values for the latest score
``POST /api/deals/{id}/what-if``   score a hypothetical change, never saved
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.features.build import build_feature_vector
from app.models.registry import ModelRegistry
from app.schemas.scoring import (
    ExplainResponse,
    ScoreRequest,
    ScoreResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from app.security.ratelimit import SCORE_LIMIT, WHATIF_LIMIT, limiter
from app.services import scoring

router = APIRouter()


def get_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None or registry.booster is None:
        raise HTTPException(
            status_code=503,
            detail="models not loaded; run `make train` and restart",
        )
    return registry


def get_redis(request: Request):
    return getattr(request.app.state, "redis", None)


@router.post("/{deal_id}/score", response_model=ScoreResponse)
@limiter.limit(SCORE_LIMIT)
async def score_deal(
    request: Request,
    deal_id: str,
    body: ScoreRequest | None = None,
    session: AsyncSession = Depends(get_session),
    registry: ModelRegistry = Depends(get_registry),
    redis=Depends(get_redis),
) -> ScoreResponse:
    """Score a deal now and append the result to its history."""
    body = body or ScoreRequest()
    try:
        result = await scoring.score_deal(
            session,
            redis,
            registry,
            deal_id,
            bypass_cache=body.bypass_cache,
            persist=True,
        )
    except scoring.DealNotFound:
        raise HTTPException(status_code=404, detail=f"deal {deal_id!r} not found")

    return ScoreResponse(**result.to_dict())


@router.get("/{deal_id}/explain", response_model=ExplainResponse)
async def explain_deal(
    deal_id: str,
    session: AsyncSession = Depends(get_session),
    registry: ModelRegistry = Depends(get_registry),
) -> ExplainResponse:
    """Full SHAP attribution for the deal's current features.

    Returns every feature's contribution, not the stored top-k, so the dashboard
    waterfall can show the long tail. Values are log-odds and additive:
    ``base_value + sum(shap)`` reconstructs the model's margin.
    """
    try:
        deal, activities = await scoring.load_deal(session, deal_id)
    except scoring.DealNotFound:
        raise HTTPException(status_code=404, detail=f"deal {deal_id!r} not found")

    vector = build_feature_vector(
        scoring.deal_mapping(deal),
        scoring.activity_mappings(activities),
        rep_stats=registry.rep_stats,
    )
    drivers, base = scoring.shap_drivers(registry, vector, top_k=None)
    win_prob = float(registry.predict_win_prob(vector)[0])

    stored = await scoring.latest_score(session, deal_id)
    margin = base + float(np.sum([d["shap"] for d in drivers]))

    return ExplainResponse(
        deal_id=deal_id,
        win_prob=win_prob,
        model_version=registry.model_version,
        shap_values=drivers,
        shap_base_value=base,
        margin=round(margin, 5),
        scored_at=(
            stored.scored_at.isoformat()
            if stored
            else datetime.now(timezone.utc).isoformat()
        ),
    )


@router.post("/{deal_id}/what-if", response_model=WhatIfResponse)
@limiter.limit(WHATIF_LIMIT)
async def what_if(
    request: Request,
    deal_id: str,
    body: WhatIfRequest,
    session: AsyncSession = Depends(get_session),
    registry: ModelRegistry = Depends(get_registry),
) -> WhatIfResponse:
    """Score a hypothetical feature change without saving it."""
    try:
        result = await scoring.what_if(session, registry, deal_id, body.overrides)
    except scoring.DealNotFound:
        raise HTTPException(status_code=404, detail=f"deal {deal_id!r} not found")
    except scoring.WhatIfError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return WhatIfResponse(**result)
