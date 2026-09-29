"""Metrics and model-card metadata (spec section 7).

``GET /api/meta/metrics``     demo model metrics
``GET /api/meta/model-card``  the rendered model card
``GET /api/meta/features``    the frozen feature schema, for the SHAP chart labels
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from app.features.build import FEATURE_LABELS
from app.models.registry import registry

router = APIRouter()


@router.get("/metrics")
async def get_metrics() -> dict:
    """Demo model metrics, read from artifacts/metrics.json.

    Everything here was measured on synthetic data and is labelled as such, so
    the dashboard cannot present it as anything else.
    """
    metrics = registry.metrics
    if not metrics:
        raise HTTPException(
            status_code=503,
            detail="metrics.json not loaded; run `make train`",
        )

    demo = metrics.get("demo_models", {})
    return {
        "model_version": metrics.get("model_version"),
        "generated_on": metrics.get("generated_on"),
        "demo": {
            "label": "Demo model (synthetic data)",
            "synthetic_data": True,
            "win_probability": demo.get("win_probability", {}),
            "days_to_close": demo.get("days_to_close", {}),
            "data": metrics.get("data", {}),
        },
        "global_drivers": metrics.get("global_drivers", []),
        "latency_ms": metrics.get("latency_ms", {}),
        "note": (
            "Measured on generated data and reproducible from the repository "
            "by rerunning `make train` from the seed."
        ),
    }


@router.get("/model-card")
async def get_model_card() -> dict:
    path = registry.path("model_card.md")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="model_card.md not built")
    with open(path, encoding="utf-8") as fh:
        return {"format": "markdown", "content": fh.read()}


@router.get("/features")
async def get_feature_schema() -> dict:
    """The feature contract, with plain-English labels for the dashboard."""
    if not registry.schema:
        raise HTTPException(status_code=503, detail="feature schema not loaded")
    return {**registry.schema, "labels": dict(FEATURE_LABELS)}
