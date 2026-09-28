"""Metrics and model-card metadata (spec section 7).

``GET /api/meta/metrics``     demo metrics, with the paper's kept separate
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

    The response keeps ``demo`` and ``paper`` in separate objects and labels both,
    so the dashboard cannot accidentally present a pilot number as a demo one.
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
        "paper": {
            "label": "Paper (real pilot data)",
            "reproducible_here": False,
            **metrics.get("paper_reference", {}),
        },
        "global_drivers": metrics.get("global_drivers", []),
        "latency_ms": metrics.get("latency_ms", {}),
        "comparison_note": (
            "Demo models are retrained on generated data and score lower than "
            "the paper's pilot models. The two sets are not comparable and are "
            "never averaged."
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
