from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.schemas.api import GlobalImportanceResponse, LocalExplainResponse
from app.dependencies import get_db, get_current_user
from app.services.shap_service import get_global_importance, get_local_explanation
from app.models.db import Prediction

router = APIRouter()

@router.get("/global", response_model=GlobalImportanceResponse)
async def explain_global(user: dict = Depends(get_current_user)):
    rankings = get_global_importance()
    return GlobalImportanceResponse(rankings=rankings)

@router.get("/local/{opp_id}", response_model=LocalExplainResponse)
async def explain_local(
    opp_id: str,
    req: Request,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    # Fetch latest prediction for this opportunity
    result = await db.execute(
        select(Prediction)
        .filter(Prediction.opportunity_id == opp_id)
        .order_by(Prediction.created_at.desc())
    )
    prediction = result.scalars().first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found for this opportunity")
        
    explanation = get_local_explanation(prediction, req.app.state)
    return LocalExplainResponse(**explanation)
