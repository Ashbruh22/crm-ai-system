"""FastAPI app factory.

Phase 2 changes from the original startup path:

* **No TensorFlow.** The old module scope did ``import tensorflow as tf`` and
  loaded a ``.keras`` file plus ``ml/xgboost_model.joblib`` — neither of which
  was committed, so a fresh clone could not boot. Models now come from
  ``artifacts/`` via ``app.models.registry``: XGBoost from native JSON, the LSTM
  from ONNX Runtime.
* **CORS is restricted** to ``ALLOWED_ORIGINS``. It was ``allow_origins=["*"]``
  with ``allow_credentials=True``, which browsers reject and the spec forbids.
* **The feature schema is validated at startup** and the app refuses to boot on
  a mismatch.
* ``/healthz`` reports model, DB, and Redis status (``/health`` and ``/ready``
  are kept as aliases so existing probes keep working).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.middleware import RequestIDMiddleware
from app.models.registry import ArtifactError, ModelRegistry, registry
from app.routers import actions, deals, meta, scoring

log = logging.getLogger("crm_ai")

# Routers still on the pre-phase-1 artifacts (ml/pipeline.joblib, the Keras
# model) and the legacy opportunities/predictions tables. They cannot load
# their models any more, so mounting them would break startup. Each is restored
# against the new registry in the phase noted:
#   crm.py                  -> phase 5 (MessageBus + signed webhook)
#   admin.py, auth.py       -> phase 8 (security pass)
# Superseded: predict.py + explain.py by routers/scoring.py (phase 3),
# recommendations.py by agent/nba.py + routers/actions.py (phase 4).
LEGACY_ROUTERS_DISABLED = ("crm", "admin", "auth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- models ---------------------------------------------------------
    app.state.registry = registry
    registry.artifact_dir = settings.ARTIFACT_DIR
    try:
        registry.load()
        if settings.MODEL_VERSION:
            registry.model_version = settings.MODEL_VERSION
        log.info("artifacts loaded: %s", registry.health())
    except ArtifactError:
        if settings.REQUIRE_ARTIFACTS:
            # Deliberate: serving a model whose features have drifted produces
            # confidently wrong explanations, which is worse than not starting.
            raise
        log.warning("artifacts unavailable; continuing without models")
        app.state.registry = ModelRegistry(artifact_dir=settings.ARTIFACT_DIR)

    # --- redis ----------------------------------------------------------
    app.state.redis = redis.from_url(
        settings.REDIS_URL, encoding="utf-8", decode_responses=True
    )

    # --- database -------------------------------------------------------
    if settings.SEED_ON_STARTUP:
        try:
            from app.db.session import SessionLocal
            from app.seed import seed_if_empty

            async with SessionLocal() as session:
                result = await seed_if_empty(session, settings.SYNTHETIC_DATA_DIR)
            log.info("seed: %s", result)
        except Exception as exc:  # noqa: BLE001 - never block boot on seeding
            log.warning("seeding skipped: %s", exc)

    yield

    await app.state.redis.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="CRM AI Core",
        description=(
            "Real-time deal scoring, SHAP explanations, and next-best-action "
            "recommendations. Runs entirely on synthetic data."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.DOCS_ENABLED else None,
        redoc_url=None,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    app.include_router(deals.router, prefix="/api/deals", tags=["Deals"])
    # Same prefix as deals.py; the paths do not collide ({id} vs {id}/score).
    app.include_router(scoring.router, prefix="/api/deals", tags=["Scoring"])
    app.include_router(actions.router, prefix="/api/actions", tags=["Actions"])
    app.include_router(meta.router, prefix="/api/meta", tags=["Meta"])

    _register_health(app)
    return app


def _register_health(app: FastAPI) -> None:
    async def healthz() -> dict:
        """Status of models, database, and Redis."""
        reg: ModelRegistry = getattr(app.state, "registry", registry)

        redis_status = "ok"
        try:
            await app.state.redis.ping()
        except Exception as exc:  # noqa: BLE001
            redis_status = f"error: {type(exc).__name__}"

        db_status = "ok"
        try:
            from sqlalchemy import text

            from app.db.session import engine

            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            db_status = f"error: {type(exc).__name__}"

        models = reg.health()
        healthy = (
            redis_status == "ok" and db_status == "ok" and models["xgb_loaded"]
        )
        return {
            "status": "ok" if healthy else "degraded",
            "version": app.version,
            "models": models,
            "db": db_status,
            "redis": redis_status,
            "synthetic_data": True,
        }

    app.add_api_route("/healthz", healthz, methods=["GET"], tags=["System"])
    # Kept so existing Docker/Render probes do not break.
    app.add_api_route("/health", healthz, methods=["GET"], tags=["System"])
    app.add_api_route("/ready", healthz, methods=["GET"], tags=["System"])


app = create_app()
