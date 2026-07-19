import os
import joblib
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import redis.asyncio as redis

from app.config import settings
from app.middleware import RequestIDMiddleware
from app.routers import predict, explain, recommendations, admin, auth, crm

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # XGBoost
    xgb_path = os.path.join(base_dir, 'ml', 'xgboost_model.joblib')
    app.state.xgboost_model = joblib.load(xgb_path)
    
    # Pipeline
    pipeline_path = os.path.join(base_dir, 'ml', 'pipeline.joblib')
    app.state.pipeline = joblib.load(pipeline_path)
    
    # LSTM Problem 2 Model
    lstm_path = os.path.join(base_dir, 'ml', 'lstm_problem2_live_reforecast.keras')
    app.state.lstm_model = tf.keras.models.load_model(lstm_path)
    
    # Assertion as requested: fail loudly if input shape is wrong
    expected_shape = (None, 60, 7)
    actual_shape = app.state.lstm_model.input_shape
    assert actual_shape == expected_shape, f"CRITICAL: LSTM model shape mismatch. Expected {expected_shape}, got {actual_shape}."
    
    # Initialize Redis connection
    app.state.redis = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    
    yield
    
    # Cleanup
    await app.state.redis.aclose()

app = FastAPI(
    title="CRM AI System",
    description="Phase 4 Production API Stack",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DOCS_ENABLED else None,
    redoc_url=None
)

# Middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Include Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["Explainability"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(crm.router, prefix="/api/v1/crm", tags=["CRM Integration"])

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe. Returns 200 OK immediately without checking dependencies."""
    return {"status": "ok"}

@app.get("/ready", tags=["System"])
async def readiness_check():
    """Readiness probe. Checks DB and Redis connectivity."""
    db_status = "ok"
    redis_status = "ok"
    
    # Check Redis
    try:
        await app.state.redis.ping()
    except Exception:
        redis_status = "error"
        
    # Check DB
    try:
        from app.dependencies import engine
        async with engine.connect() as conn:
            pass # Just connecting is enough for a basic check
    except Exception:
        db_status = "error"
        
    status = "ok" if db_status == "ok" and redis_status == "ok" else "error"
    return {"status": status, "db_status": db_status, "redis_status": redis_status}
