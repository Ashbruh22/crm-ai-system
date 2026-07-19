import asyncio
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from app.schemas.api import (
    OutcomePredictRequest, OutcomePredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    CyclePredictRequest, CyclePredictResponse
)
from app.dependencies import get_db, get_redis, get_current_user
from app.services.prediction_service import predict_outcome
from app.services.lstm_service import predict_cycle
from app.config import settings

router = APIRouter()

@router.post("/outcome", response_model=OutcomePredictResponse)
async def predict_outcome_endpoint(
    request: OutcomePredictRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
    user: dict = Depends(get_current_user)
):
    return await predict_outcome(request, db, redis_client, req.app.state)

@router.post("/outcome/batch", response_model=BatchPredictResponse)
async def batch_predict_endpoint(
    request: BatchPredictRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
    user: dict = Depends(get_current_user)
):
    # Use semaphore to limit concurrency
    sem = asyncio.Semaphore(settings.BATCH_CONCURRENCY)
    
    async def predict_one_with_sem(deal):
        async with sem:
            return await predict_outcome(deal, db, redis_client, req.app.state)
            
    tasks = [predict_one_with_sem(deal) for deal in request.deals]
    results = await asyncio.gather(*tasks)
    return BatchPredictResponse(results=results)

@router.post("/cycle", response_model=CyclePredictResponse)
async def predict_cycle_endpoint(
    request: CyclePredictRequest,
    req: Request,
    user: dict = Depends(get_current_user)
):
    return await predict_cycle(request, req.app.state)

@router.get("/history/{opp_id}")
async def get_prediction_history(
    opp_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    from sqlalchemy.future import select
    from app.models.db import Prediction
    
    result = await db.execute(select(Prediction).filter(Prediction.opportunity_id == opp_id).order_by(Prediction.created_at.desc()))
    predictions = result.scalars().all()
    
    # Return raw dicts for simplicity in this router
    return [{"id": str(p.id), "win_probability": p.win_probability, "outcome_prediction": p.outcome_prediction, "created_at": p.created_at} for p in predictions]
