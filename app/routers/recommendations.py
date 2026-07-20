from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.schemas.api import RecommendationsResponse, RecommendationItem, FeedbackRequest
from app.dependencies import get_db, get_current_user
from app.models.db import Recommendation, Prediction, Opportunity

router = APIRouter()

@router.get("/{opp_id}", response_model=RecommendationsResponse)
async def get_recommendations(
    opp_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    result = await db.execute(
        select(Recommendation)
        .filter(Recommendation.opportunity_id == opp_id)
        .order_by(Recommendation.urgency_score.desc())
    )
    recs = result.scalars().all()
    
    return RecommendationsResponse(recommendations=[
        RecommendationItem(
            action=r.action,
            rationale=r.rationale,
            expected_impact=r.expected_impact,
            priority=r.priority,
            urgency_score=r.urgency_score
        ) for r in recs
    ])

@router.get("/team/{agent}", response_model=RecommendationsResponse)
async def get_team_recommendations(
    agent: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    # Join with Opportunity to filter by agent
    result = await db.execute(
        select(Recommendation)
        .join(Prediction, Recommendation.prediction_id == Prediction.id)
        .join(Opportunity, Prediction.opportunity_id == Opportunity.opportunity_id)
        .filter(Opportunity.sales_agent == agent)
        .order_by(Recommendation.urgency_score.desc())
    )
    recs = result.scalars().all()
    
    return RecommendationsResponse(recommendations=[
        RecommendationItem(
            action=r.action,
            rationale=r.rationale,
            expected_impact=r.expected_impact,
            priority=r.priority,
            urgency_score=r.urgency_score
        ) for r in recs
    ])

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    # Update recommendation with feedback
    # (Implementation simplified for brevity)
    return {"status": "success"}
