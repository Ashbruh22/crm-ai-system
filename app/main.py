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

import asyncio
import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.middleware import RequestIDMiddleware
from app.models.registry import ArtifactError, ModelRegistry, registry
from app.routers import actions, deals, events, meta, scoring, stream

log = logging.getLogger("crm_ai")

# Routers still on the pre-phase-1 artifacts (ml/pipeline.joblib, the Keras
# model) and the legacy opportunities/predictions tables. They cannot load
# their models any more, so mounting them would break startup. Each is restored
# against the new registry in the phase noted:
#   admin.py, auth.py       -> phase 8 (security pass)
# Superseded: predict.py + explain.py by routers/scoring.py (phase 3),
# recommendations.py by agent/nba.py + routers/actions.py (phase 4),
# crm.py by bus/ + routers/events.py (phase 5).
LEGACY_ROUTERS_DISABLED = ("admin", "auth")


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

    # --- bus + broadcaster ----------------------------------------------
    from app.bus.broadcast import Broadcaster
    from app.bus.redis_streams import RedisStreamsBus

    app.state.bus = RedisStreamsBus(app.state.redis)
    app.state.broadcaster = Broadcaster(app.state.redis)
    app.state.bus_error = None

    from app.bus.redis_streams import StreamsUnsupported

    try:
        await app.state.bus.assert_streams_supported()
    except StreamsUnsupported as exc:
        # Start anyway: everything except event ingestion still works, and a
        # dead /healthz is a worse way to learn about this than a clear flag.
        app.state.bus_error = str(exc)
        log.error("ingestion disabled: %s", exc)

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

    # --- stream consumer -------------------------------------------------
    # Runs in-process so the hosted demo is one Render service, not two.
    consumer_task: asyncio.Task | None = None
    if settings.RUN_CONSUMER and not app.state.bus_error:
        from app.consumer import run_consumer
        from app.db.session import SessionLocal

        consumer_task = asyncio.create_task(
            run_consumer(
                app.state.bus,
                SessionLocal,
                app.state.redis,
                app.state.registry,
                app.state.broadcaster,
                consumer_name=settings.CONSUMER_NAME,
            ),
            name="crm-ai-consumer",
        )

    yield

    if consumer_task is not None:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

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
    app.include_router(events.router, prefix="/api/events", tags=["Events"])
    app.include_router(events.webhook_router, prefix="/webhooks", tags=["Events"])
    app.include_router(stream.router, prefix="/api/stream", tags=["Stream"])
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

        bus_status: dict = {}
        bus = getattr(app.state, "bus", None)
        bus_error = getattr(app.state, "bus_error", None)
        if bus_error:
            bus_status = {"available": False, "error": bus_error}
        elif bus is not None:
            try:
                bus_status = {
                    "available": True,
                    "stream_len": await bus.length(),
                    "pending": await bus.pending_count("crm-ai-scorer"),
                    "consumer_running": settings.RUN_CONSUMER,
                }
            except Exception as exc:  # noqa: BLE001
                bus_status = {"available": False, "error": type(exc).__name__}

        models = reg.health()
        # Ingestion being down is a real degradation: the demo can still score
        # on request, but the "fire an event and watch it update" path is dead.
        healthy = (
            redis_status == "ok"
            and db_status == "ok"
            and models["xgb_loaded"]
            and bus_status.get("available", False)
        )
        return {
            "status": "ok" if healthy else "degraded",
            "version": app.version,
            "models": models,
            "db": db_status,
            "redis": redis_status,
            "bus": bus_status,
            "synthetic_data": True,
        }

    app.add_api_route("/healthz", healthz, methods=["GET"], tags=["System"])
    # Kept so existing Docker/Render probes do not break.
    app.add_api_route("/health", healthz, methods=["GET"], tags=["System"])
    app.add_api_route("/ready", healthz, methods=["GET"], tags=["System"])


app = create_app()
