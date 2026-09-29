"""Artifact contract tests: the files the service boots from (spec section 11).

These guard the three things that would break a deploy silently:

* the frozen feature schema drifting from the code,
* the ONNX graph carrying weights in a sidecar file or refusing batches, and
* metrics losing the label that says they are synthetic.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from app.features.build import FEATURE_NAMES, SEQ_FEATURE_NAMES, SEQ_LEN, feature_schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACT_DIR, "lstm.onnx")),
    reason="artifacts not built; run `make train`",
)


def _load(name: str) -> dict:
    with open(os.path.join(ARTIFACT_DIR, name)) as fh:
        return json.load(fh)


def test_expected_artifacts_exist():
    for name in (
        "xgb_model.json",
        "lstm.onnx",
        "feature_schema.json",
        "rep_stats.json",
        "metrics.json",
        "model_card.md",
    ):
        path = os.path.join(ARTIFACT_DIR, name)
        assert os.path.exists(path), name
        assert os.path.getsize(path) > 0, name


def test_saved_schema_matches_code():
    """The check the service performs at startup, run in CI."""
    saved = _load("feature_schema.json")
    assert saved == feature_schema(), (
        "artifacts/feature_schema.json is stale; retrain so the service can boot"
    )
    assert saved["n_features"] == len(FEATURE_NAMES)


def test_onnx_is_self_contained_with_dynamic_batch():
    import onnx

    path = os.path.join(ARTIFACT_DIR, "lstm.onnx")
    model = onnx.load(path)

    # A sidecar .data file breaks any deploy that copies only the .onnx.
    assert not os.path.exists(path + ".data")
    external = [t.name for t in model.graph.initializer if t.data_location != 0]
    assert external == [], f"weights stored externally: {external}"

    (inp,) = model.graph.input
    (out,) = model.graph.output
    in_dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]

    assert in_dims == ["batch", SEQ_LEN, len(SEQ_FEATURE_NAMES)]
    assert out_dims[0] == "batch", "batch axis must stay dynamic for batch scoring"


def test_onnx_runs_on_a_batch_and_predicts_positive_days():
    import onnxruntime as ort

    session = ort.InferenceSession(
        os.path.join(ARTIFACT_DIR, "lstm.onnx"), providers=["CPUExecutionProvider"]
    )
    batch = np.zeros((8, SEQ_LEN, len(SEQ_FEATURE_NAMES)), dtype=np.float32)
    batch[:, -1, :] = 0.5
    out = session.run(None, {"activity_sequence": batch})[0]

    assert out.shape == (8, 1)
    # softplus head: a deal can never be predicted to close in negative days.
    assert np.all(out > 0), out


def test_recorded_onnx_parity_passed():
    parity = _load("lstm_metrics.json")["onnx_parity"]
    assert parity["passed"], parity
    assert parity["max_abs_diff_days"] < parity["tolerance"]
    assert parity["n_compared"] > 1, "parity must be checked on a real batch"


def test_xgb_model_loads_natively_and_explains_additively():
    """The runtime needs no pickled sklearn pipeline and no shap package.

    Explanations come from XGBoost's own TreeSHAP. The additivity assertion is
    the point: the shap package's TreeExplainer, against xgboost 3.2, returned
    attributions that did NOT sum to the prediction (errors up to 0.38 in
    log-odds), which would have made every dashboard explanation wrong.
    """
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(os.path.join(ARTIFACT_DIR, "xgb_model.json"))

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (8, len(FEATURE_NAMES))).astype(np.float32)
    dmatrix = xgb.DMatrix(X, feature_names=list(FEATURE_NAMES))

    contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
    assert contribs.shape == (8, len(FEATURE_NAMES) + 1), "last column is the bias"

    margin = booster.predict(dmatrix, output_margin=True)
    assert np.max(np.abs(contribs.sum(axis=1) - margin)) < 1e-5

    # And the margin reconstructs the probability.
    prob = booster.predict(dmatrix)
    assert np.allclose(1.0 / (1.0 + np.exp(-margin)), prob, atol=1e-6)


def test_runtime_requirements_exclude_the_shap_package():
    """shap is a training-only dependency; it pulls numba + llvmlite (~70 MB)."""
    root = os.path.dirname(ARTIFACT_DIR)
    with open(os.path.join(root, "requirements.txt"), encoding="utf-8") as fh:
        runtime = [
            line.split("#")[0].strip()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        ]
    assert not any(r.lower().startswith("shap") for r in runtime), runtime


def test_metrics_are_labelled_synthetic_and_plausible():
    metrics = _load("metrics.json")

    assert metrics["synthetic_data"] is True
    demo = metrics["demo_models"]

    # No metric block may claim a provenance the repo cannot reproduce.
    assert "paper_reference" not in metrics
    assert 0.0 < demo["win_probability"]["auc_roc"] < 1.0
    assert demo["days_to_close"]["mae_days"] > 0

    # A latency figure may appear only once it has actually been observed, and
    # must say where. Null is fine; a number with no provenance is not.
    latency = metrics["latency_ms"]
    if latency.get("measured") is not None:
        assert latency["measured"] > 0
        assert latency.get("host"), "a measured latency must name the host"
        assert latency.get("measured_at"), "a measured latency must be dated"


def test_days_to_close_model_beats_the_mean_baseline():
    cyc = _load("metrics.json")["demo_models"]["days_to_close"]
    assert cyc["mae_days"] < cyc["baseline_mae_days"], (
        "LSTM does not beat predicting the mean; not worth serving"
    )


def test_rep_stats_are_probabilities():
    stats = _load("rep_stats.json")
    assert 0.0 < stats["global_win_rate"] < 1.0
    assert stats["by_rep"], "no per-rep rates fitted"
    assert all(0.0 <= v <= 1.0 for v in stats["by_rep"].values())


def test_model_card_carries_the_synthetic_disclaimer():
    with open(os.path.join(ARTIFACT_DIR, "model_card.md"), encoding="utf-8") as fh:
        card = fh.read().lower()
    assert "synthetic data only" in card
    assert "no real crm records" in card
    # Every figure on the card must be reproducible from this repository.
    assert "reproducible" in card
    assert "paper" not in card
