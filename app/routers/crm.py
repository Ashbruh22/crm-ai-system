from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from typing import Optional

from app.schemas.crm import SalesforceWebhookPayload, HubSpotWebhookPayload, CRMSimulatePayload
from app.schemas.api import OutcomePredictRequest, CyclePredictRequest
from app.dependencies import get_db, get_redis
from app.services.prediction_service import predict_outcome
from app.services.lstm_service import predict_cycle
from app.services.crm_mapper import map_salesforce_to_internal, map_hubspot_to_internal
from app.utils.hmac_validator import validate_salesforce_signature, validate_hubspot_signature
from app.config import settings

router = APIRouter()

async def _process_crm_deal(deal_data: dict, db: AsyncSession, redis_client: redis.Redis, app_state) -> tuple[dict, int]:
    # 1. Outcome prediction (XGBoost + SHAP + Recommendations + DB save)
    req_outcome = OutcomePredictRequest(
        opportunity_id=deal_data["opportunity_id"],
        sales_agent=deal_data["sales_agent"],
        product=deal_data["product"],
        engage_date=deal_data["engage_date"]
    )
    outcome_res = await predict_outcome(req_outcome, db, redis_client, app_state)
    
    # 2. Cycle forecast (LSTM)
    cycle_days = 14 # default fallback
    try:
        req_cycle = CyclePredictRequest(
            opportunity_id=deal_data["opportunity_id"],
            sales_agent=deal_data["sales_agent"],
            product=deal_data["product"],
            engage_date=deal_data["engage_date"],
            current_stage=deal_data.get("deal_stage", "proposal")
        )
        cycle_res = await predict_cycle(req_cycle, app_state)
        cycle_days = int(cycle_res.predicted_days_remaining)
    except Exception:
        pass

    # Extract Risk Factors & Recommendations
    top_factors = outcome_res.shap_explanation.top_factors
    risk_factors = [f.feature for f in top_factors if f.direction == "negative"]
    risk_str = ", ".join(risk_factors[:3]) if risk_factors else "None"
    
    rec_actions = [r.action for r in outcome_res.recommendations]
    rec_str = "; ".join(rec_actions[:2]) if rec_actions else "None"
    
    return {
        "outcome": outcome_res,
        "cycle_days": cycle_days,
        "risk_str": risk_str,
        "rec_str": rec_str
    }, cycle_days

@router.post("/webhook/salesforce")
async def salesforce_webhook(
    request: Request,
    payload: SalesforceWebhookPayload,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
    x_salesforce_signature: Optional[str] = Header(None, alias="X-Salesforce-Signature")
):
    body_bytes = await request.body()
    if not validate_salesforce_signature(body_bytes, x_salesforce_signature or "", settings.CRM_WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid Salesforce HMAC signature")
        
    sobject_dict = payload.sobject.model_dump()
    deal_data = map_salesforce_to_internal(sobject_dict)
    
    res, cycle_days = await _process_crm_deal(deal_data, db, redis_client, request.app.state)
    outcome = res["outcome"]
    
    write_back = {
        "AI_Win_Probability__c": round(outcome.win_probability, 3),
        "AI_Sales_Cycle_Forecast__c": cycle_days,
        "AI_Risk_Factors__c": res["risk_str"],
        "AI_Recommended_Actions__c": res["rec_str"]
    }
    return write_back

@router.post("/webhook/hubspot")
async def hubspot_webhook(
    request: Request,
    payload: HubSpotWebhookPayload,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
    x_hubspot_signature: Optional[str] = Header(None, alias="X-HubSpot-Signature-v3"),
    x_hubspot_request_timestamp: Optional[str] = Header(None, alias="X-HubSpot-Request-Timestamp")
):
    body_bytes = await request.body()
    if not validate_hubspot_signature(body_bytes, x_hubspot_request_timestamp or "", x_hubspot_signature or "", settings.CRM_WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid HubSpot HMAC signature")
        
    deal_props = payload.deal_properties or {}
    deal_data = map_hubspot_to_internal(deal_props)
    
    res, cycle_days = await _process_crm_deal(deal_data, db, redis_client, request.app.state)
    outcome = res["outcome"]
    
    write_back = {
        "ai_win_probability": str(round(outcome.win_probability, 3)),
        "ai_sales_cycle_forecast": str(cycle_days),
        "ai_risk_factors": res["risk_str"],
        "ai_recommended_actions": res["rec_str"]
    }
    return write_back

@router.post("/webhook/simulate")
async def simulate_webhook(
    request: Request,
    payload: CRMSimulatePayload,
    db: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    if not settings.CRM_SIMULATE_ENABLED:
        raise HTTPException(status_code=404, detail="Simulation endpoint is disabled in production")
        
    deal_data = {
        "opportunity_id": payload.opportunity_id,
        "sales_agent": payload.sales_agent,
        "product": payload.product,
        "engage_date": payload.engage_date,
        "deal_stage": payload.deal_stage
    }
    
    res, cycle_days = await _process_crm_deal(deal_data, db, redis_client, request.app.state)
    outcome = res["outcome"]
    
    if payload.platform.lower() == "salesforce":
        write_back = {
            "AI_Win_Probability__c": round(outcome.win_probability, 3),
            "AI_Sales_Cycle_Forecast__c": cycle_days,
            "AI_Risk_Factors__c": res["risk_str"],
            "AI_Recommended_Actions__c": res["rec_str"]
        }
    else:
        write_back = {
            "ai_win_probability": str(round(outcome.win_probability, 3)),
            "ai_sales_cycle_forecast": str(cycle_days),
            "ai_risk_factors": res["risk_str"],
            "ai_recommended_actions": res["rec_str"]
        }
        
    return {
        "platform": payload.platform,
        "opportunity_id": payload.opportunity_id,
        "outcome": outcome,
        "cycle_forecast_days": cycle_days,
        "write_back": write_back
    }
