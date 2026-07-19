import numpy as np
import pandas as pd
from app.schemas.api import CyclePredictRequest, CyclePredictResponse

LSTM_MODEL_VERSION = "lstm-v1.0"

async def predict_cycle(request: CyclePredictRequest, state) -> CyclePredictResponse:
    # 1. Map current stage
    STAGE_TO_INT = {"prospecting": 0, "qualification": 1, "proposal": 2, "negotiation": 3}
    stage_int = STAGE_TO_INT.get(request.current_stage, 0)
    
    # 2. Build the 7 features
    max_day = 90.0
    max_activity = 30.0
    max_gap = 30.0
    max_deal_value = 300_000.0
    
    # Mocking static features as requested or to defaults
    agent_wr = 0.5
    product_wr = 0.5
    agent_adv = 50000.0

    day_offset = float(request.days_elapsed)
    acts = float(request.num_activities)
    gap = float(request.days_since_last_activity)

    feature_vec = np.array([
        min(day_offset, max_day) / max_day,
        stage_int / max(1, len(STAGE_TO_INT) - 1),
        min(acts, max_activity) / max_activity,
        min(gap, max_gap) / max_gap,
        agent_wr,
        product_wr,
        min(agent_adv, max_deal_value) / max_deal_value,
    ], dtype=np.float32)
    
    # 3. Create (1, 60, 7) sequence array with right-padding
    X_seq = np.zeros((1, 60, 7), dtype=np.float32)
    
    cutoff = min(60, int(day_offset) + 1)
    for t in range(cutoff):
        X_seq[0, t, :] = feature_vec
        
    # 4. Predict
    lstm_model = state.lstm_model
    y_pred_log = lstm_model.predict(X_seq, verbose=0).flatten()[0]
    predicted_days_remaining = int(np.expm1(y_pred_log))
    
    return CyclePredictResponse(
        opportunity_id=request.opportunity_id,
        predicted_days_remaining=predicted_days_remaining,
        model_version=LSTM_MODEL_VERSION,
        cached=False
    )
