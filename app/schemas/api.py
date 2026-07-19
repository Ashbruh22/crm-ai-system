from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Tuple

class OutcomePredictRequest(BaseModel):
    opportunity_id: str
    sales_agent: str
    product: str
    engage_date: str

class TopFactor(BaseModel):
    feature: str
    shap: float
    direction: str

class ShapExplanation(BaseModel):
    nl_explanation: str
    top_factors: List[TopFactor]

class RecommendationItem(BaseModel):
    action: str
    rationale: str
    expected_impact: Optional[str] = None
    priority: str
    urgency_score: float

class OutcomePredictResponse(BaseModel):
    opportunity_id: str
    win_probability: float
    outcome_prediction: str
    confidence_interval: Optional[Tuple[float, float]] = None
    shap_explanation: ShapExplanation
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    model_version: str
    inference_latency_ms: float
    cached: bool

class BatchPredictRequest(BaseModel):
    deals: List[OutcomePredictRequest] = Field(max_length=100)

class BatchPredictResponse(BaseModel):
    results: List[OutcomePredictResponse]

class CyclePredictRequest(BaseModel):
    opportunity_id: str
    sales_agent: Optional[str] = None
    product: Optional[str] = None
    engage_date: Optional[str] = None
    days_elapsed: Optional[float] = 0.0
    current_stage: Optional[str] = "prospecting"
    num_activities: Optional[float] = 0.0
    days_since_last_activity: Optional[float] = 0.0

class CyclePredictResponse(BaseModel):
    opportunity_id: str
    predicted_days_remaining: int
    model_version: str
    cached: bool

class GlobalImportanceResponse(BaseModel):
    rankings: List[dict]

class LocalExplainResponse(BaseModel):
    waterfall_data: dict
    force_plot_data: dict

class RecommendationsResponse(BaseModel):
    recommendations: List[RecommendationItem]

class FeedbackRequest(BaseModel):
    prediction_id: str
    adopted_at: str
    outcome_delta: float

class DriftResponse(BaseModel):
    ks_stat: float
    p_value: float
    drift_detected: bool

class PerformanceResponse(BaseModel):
    accuracy: float
    auc_roc: float

class HealthResponse(BaseModel):
    status: str

class ReadyResponse(BaseModel):
    status: str
    db_status: str
    redis_status: str
