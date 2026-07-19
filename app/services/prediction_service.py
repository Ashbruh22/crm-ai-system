import time
import json
import random
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete as sa_delete
import joblib
import redis.asyncio as redis

from app.schemas.api import OutcomePredictRequest, OutcomePredictResponse
from app.models.db import Opportunity, Prediction, Recommendation, ModelVersion
from app.services.recommendation_service import generate_recommendations
from app.services.shap_service import get_shap_explainer

XGBOOST_MODEL_VERSION = "xgboost-v1.0"
CACHE_TTL = 4 * 60 * 60  # 4 hours

async def predict_outcome(
    request: OutcomePredictRequest, 
    db: AsyncSession, 
    redis_client: redis.Redis,
    state
) -> OutcomePredictResponse:
    start_time = time.perf_counter()
    
    # Generate feature hash for cache key
    # Simple hash based on the input values
    feature_string = f"{request.opportunity_id}_{request.sales_agent}_{request.product}_{request.engage_date}"
    feature_hash = hash(feature_string)
    cache_key = f"pred:{request.opportunity_id}:{XGBOOST_MODEL_VERSION}:{feature_hash}"
    
    # 1. Check Redis Cache
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        data = json.loads(cached_data)
        data['cached'] = True
        return OutcomePredictResponse(**data)
    
    # 2. Prepare Data
    df = pd.DataFrame([{
        'opportunity_id': request.opportunity_id,
        'sales_agent': request.sales_agent,
        'product': request.product,
        'engage_date': request.engage_date,
        # Dummy values for pipeline
        'close_value': 0.0,
        'deal_stage': 'pipeline',
        'close_date': None,
        'account': 'Unknown'
    }])
    
    # 3. Transform via Pipeline
    pipeline = state.pipeline
    transformed_features = pipeline.transform(df)
    
    # 4. Check for A/B candidate model
    ab_res = await db.execute(select(ModelVersion).filter(ModelVersion.is_ab_candidate == True))
    ab_candidate = ab_res.scalars().first()

    is_ab_request = False
    model_version = XGBOOST_MODEL_VERSION
    xgb_model = state.xgboost_model

    if ab_candidate and random.random() < 0.20:
        try:
            candidate_model = joblib.load(ab_candidate.artifact_path)
            xgb_model = candidate_model
            model_version = ab_candidate.version_tag
            is_ab_request = True
        except Exception:
            pass

    win_probability = float(xgb_model.predict_proba(transformed_features)[0, 1])
    outcome_prediction = "Won" if win_probability >= 0.5 else "Lost"
    
    # Confidence Interval - Since we can't easily get per-tree variance from predict_proba without margin,
    # and the user requested we omit it if we don't do it properly, we'll set it to None.
    confidence_interval = None
    
    # 5. SHAP Explanation
    shap_explainer = get_shap_explainer(state)
    # The pipeline transforms to a numpy array, we need to pass a single row to explain_single
    base_value, shap_array = shap_explainer.explain_single(transformed_features[0])
    
    # Get waterfall data for NL explanation
    waterfall_data = shap_explainer.local_waterfall_data(shap_array, base_value, transformed_features[0])
    nl_explanation = shap_explainer.nl_explanation(waterfall_data)
    
    # Extract top factors from waterfall data
    top_factors = []
    for f in waterfall_data['features']:
        top_factors.append({
            'feature': f['name'],
            'shap': f['shap'],
            'direction': 'positive' if f['shap'] > 0 else 'negative'
        })
        
    shap_explanation = {
        'nl_explanation': nl_explanation,
        'top_factors': top_factors
    }
    
    # 6. Recommendations
    # Construct a dummy DB prediction object for the recommender
    # We pass win_prob and nl_explanation
    class DummyPrediction:
        pass
    dummy_pred = DummyPrediction()
    dummy_pred.win_probability = win_probability
    dummy_pred.nl_explanation = nl_explanation
    
    # Also need pipeline stats for urgency score
    # We can fetch global/agent stats from the AgentPerformanceAggregator
    agent_agg = pipeline.named_steps['agent_features']
    agent_avg_value = agent_agg.agent_stats_['avg_deal_value'].get(
        request.sales_agent, 
        agent_agg.agent_stats_['global_avg_deal_value']
    )
    
    class DummyOpportunity:
        pass
    dummy_opp = DummyOpportunity()
    dummy_opp.engage_date = request.engage_date
    
    recommendations = generate_recommendations(dummy_pred, dummy_opp, agent_avg_value)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # 7. Construct Response
    response_data = {
        "opportunity_id": request.opportunity_id,
        "win_probability": win_probability,
        "outcome_prediction": outcome_prediction,
        "confidence_interval": confidence_interval,
        "shap_explanation": shap_explanation,
        "recommendations": recommendations,
        "model_version": XGBOOST_MODEL_VERSION,
        "inference_latency_ms": latency_ms,
        "cached": False
    }
    
    # 8. Cache & DB Save
    await redis_client.setex(cache_key, CACHE_TTL, json.dumps(response_data))
    
    # Upsert Opportunity
    # For a real system we'd use a postgres specific upsert or fetch first
    # This is a simplified fetch or create
    # In production, use stmt = insert(Opportunity).values(...).on_conflict_do_update(...)
    # Here we are relying on standard SQLAlchemy for brevity, but note this could race.
    result = await db.execute(select(Opportunity).filter(Opportunity.opportunity_id == request.opportunity_id))
    opp = result.scalars().first()
    if not opp:
        opp = Opportunity(
            opportunity_id=request.opportunity_id,
            sales_agent=request.sales_agent,
            product=request.product,
            engage_date=request.engage_date
        )
        db.add(opp)
        await db.flush() # flush to get id if needed
        
    db_pred = Prediction(
        opportunity_id=request.opportunity_id,
        model_version=model_version,
        win_probability=win_probability,
        outcome_prediction=outcome_prediction,
        confidence_lower=None,
        confidence_upper=None,
        shap_values=shap_array, # Store raw SHAP array as JSON
        nl_explanation=nl_explanation,
        inference_latency_ms=latency_ms,
        cached=False,
        is_ab_request=is_ab_request
    )
    db.add(db_pred)

    # Upsert recommendations: clear stale rows for this opportunity, then
    # insert fresh ones tied to the new prediction so GET /recommendations
    # always reflects the latest win_probability — fixing the CRITICAL-at-70% bug.
    await db.execute(
        sa_delete(Recommendation).where(Recommendation.opportunity_id == request.opportunity_id)
    )
    for rec_dict in recommendations:
        rec = Recommendation(
            prediction_id=db_pred.id,
            opportunity_id=request.opportunity_id,
            action=rec_dict["action"],
            rationale=rec_dict["rationale"],
            expected_impact=rec_dict["expected_impact"],
            priority=rec_dict["priority"],
            urgency_score=rec_dict["urgency_score"],
        )
        db.add(rec)

    await db.commit()

    return OutcomePredictResponse(**response_data)
