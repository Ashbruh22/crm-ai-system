"""Consolidate metrics and write the model card (spec section 5).

Writes to ``artifacts/``:

* ``metrics.json``   what ``GET /api/meta/metrics`` serves
* ``model_card.md``  data, features, metrics, limitations, disclaimer

The paper's pilot metrics and this demo's synthetic metrics are kept in two
separate blocks and never averaged, blended, or presented as one number. The
demo models are retrained on generated data and score materially lower; saying
so plainly is the point.

Global driver importance uses XGBoost's own TreeSHAP (``pred_contribs=True``)
rather than the ``shap`` package, matching what the service serves. The shap
package's TreeExplainer re-parses the booster and, against xgboost 3.2, produced
attributions that did not sum to the prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.features.build import FEATURE_LABELS, FEATURE_NAMES  # noqa: E402
from train_xgb import build_training_matrix, load_raw  # noqa: E402

SEED = 42
DATA_DIR = os.path.join(REPO_ROOT, "data", "synthetic")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

#: Metrics reported in the peer-reviewed paper, measured on the real pilot CRM
#: data. Quoted here for comparison only — NOT reproducible from this repo,
#: which contains no pilot data.
PAPER_METRICS = {
    "source": "ICIDS 2026 paper, real pilot CRM data (not in this repo)",
    "win_model_accuracy": 0.873,
    "win_model_auc_roc": 0.92,
    "reproducible_here": False,
    "note": (
        "Measured on the partner organisation's real CRM data under the "
        "original pilot. The data cannot be published, so these numbers "
        "cannot be regenerated from this repository."
    ),
}

SHAP_SAMPLE = 400
TOP_K = 12


def global_shap_importance(booster: xgb.Booster, X) -> list[dict]:
    """Mean |SHAP| per feature, from XGBoost's native TreeSHAP."""
    sample = X.iloc[:SHAP_SAMPLE]
    dmatrix = xgb.DMatrix(sample, feature_names=list(FEATURE_NAMES))
    # Last column is the bias term, not a feature.
    values = np.asarray(booster.predict(dmatrix, pred_contribs=True))[:, :-1]
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {
            "feature": FEATURE_NAMES[i],
            "label": FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
            "mean_abs_shap": round(float(mean_abs[i]), 5),
        }
        for i in order[:TOP_K]
    ]


def render_model_card(metrics: dict) -> str:
    demo = metrics["demo_models"]
    win = demo["win_probability"]
    cyc = demo["days_to_close"]
    data = metrics["data"]
    paper = metrics["paper_reference"]
    drivers = metrics["global_drivers"]

    driver_rows = "\n".join(
        f"| {i + 1} | {d['label']} | {d['mean_abs_shap']:.4f} |"
        for i, d in enumerate(drivers)
    )

    return f"""# Model card — CRM AI Core (hosted demo)

**Generated:** {metrics['generated_on']} · **Model version:** `{metrics['model_version']}`

> ## Synthetic data only
> Both models below are trained on **generated data** from
> `training/generate_synthetic.py`. No real CRM records, customer names, or
> pilot data are used anywhere in this repository.
>
> The metrics in the peer-reviewed paper were measured on the partner
> organisation's real pilot data and are **higher**. The two sets are listed
> separately and must not be compared as like for like or averaged together.

## Intended use

A public, clickable demonstration of the architecture: real-time deal scoring,
SHAP explanations, and rule-based next-best-action recommendations. It is a
portfolio and review artifact, **not** a system for making real sales decisions.

## Out-of-scope use

Do not use these artifacts to forecast real revenue, rank real opportunities, or
make decisions about real customers or employees. The models have learned the
generator's structure, not any real market's.

## Training data

| Property | Value |
|---|---|
| Source | `training/generate_synthetic.py`, seed `{data['seed']}` |
| Historical deals | {data['n_historical_deals']:,} (closed, labelled) |
| Activity events | {data['n_activities']:,} |
| Live demo deals | {data['n_live_deals']} (open, unlabelled) |
| Base win rate | {data['historical_win_rate']:.1%} |
| Mean cycle | {data['mean_cycle_days']:.0f} days |
| Activities per deal | {data['activities_per_deal']:.1f} |
| Real data used | **None** |

Deals are featurised at a **random mid-flight cut** (35–100% of deal life), not
at close, because the service scores open deals with partial activity logs.
Training on complete histories would put every served vector out of
distribution.

## Features

{metrics['n_features']} features, frozen in `artifacts/feature_schema.json` and
validated at service startup — the service refuses to boot on a mismatch.
Per-rep win rates are fitted on the **training split only** so a test-set
outcome never leaks into a training feature.

## Demo metrics (synthetic data)

### Win probability — XGBoost classifier

| Metric | Value |
|---|---|
| Accuracy | {win['accuracy']:.3f} |
| AUC-ROC | {win['auc_roc']:.3f} |
| Precision | {win['precision']:.3f} |
| Recall | {win['recall']:.3f} |
| F1 | {win['f1']:.3f} |
| Brier score | {win['brier']:.3f} |
| Test rows | {win['n_test']:,} |

### Days to close — LSTM regressor (served as ONNX)

| Metric | Value |
|---|---|
| MAE | {cyc['mae_days']:.1f} days |
| RMSE | {cyc['rmse_days']:.1f} days |
| Mean-prediction baseline MAE | {cyc['baseline_mae_days']:.1f} days |
| Improvement vs baseline | {cyc['improvement_vs_baseline']:+.1%} |
| Input shape | {cyc['input_shape'][0]} timesteps x {cyc['input_shape'][1]} features |
| ONNX parity (max abs diff) | {cyc['onnx_parity']['max_abs_diff_days']:.2e} days |

The LSTM is trained in PyTorch and served through ONNX Runtime. Neither PyTorch
nor TensorFlow is installed in the runtime image; the parity check above
confirms the exported graph reproduces the framework output.

## Paper metrics (real pilot data — for reference only)

| Metric | Value |
|---|---|
| Accuracy | {paper['win_model_accuracy']:.1%} |
| AUC-ROC | {paper['win_model_auc_roc']:.2f} |
| Reproducible from this repo | **No** |

{paper['note']}

**Why the demo scores lower:** the generator's signal is deliberately noisy
(`NOISE_SD`), it has fewer features than the pilot pipeline, and it contains
none of the firmographic and historical-relationship context that the real CRM
carried. A synthetic model matching 0.92 AUC would mean the generator had
leaked the label, not that the model was good.

## Top global drivers (mean |SHAP|)

| # | Driver | Mean abs SHAP |
|---|---|---|
{driver_rows}

## Limitations

- Trained entirely on generated data; no external validity.
- The generator's win function is a logistic model over activity counts, so the
  classifier can in principle recover it almost exactly. Residual noise is what
  keeps measured AUC near {win['auc_roc']:.2f} rather than 1.0.
- Rep win rate is a fitted statistic; a rep unseen at training time falls back
  to the global base rate.
- Calibration is reported (Brier {win['brier']:.3f}) but the model is not
  explicitly calibrated; treat probabilities as ordinal.
- The days-to-close model sees only activity cadence and deal size — not
  seasonality, quota pressure, or procurement cycles.

## Credits

Peer-reviewed research (ICIDS 2026) with co-author and faculty mentor credited
in the repository README, which also links the paper.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    def _read(name: str, where: str) -> dict:
        with open(os.path.join(where, name)) as fh:
            return json.load(fh)

    xgb_metrics = _read("xgb_metrics.json", args.artifact_dir)
    lstm_metrics = _read("lstm_metrics.json", args.artifact_dir)
    manifest = _read("manifest.json", args.data_dir)
    schema = _read("feature_schema.json", args.artifact_dir)

    booster = xgb.Booster()
    booster.load_model(os.path.join(args.artifact_dir, "xgb_model.json"))

    deals, activities = load_raw(args.data_dir)
    _, test_deals = train_test_split(
        deals, test_size=0.2, random_state=args.seed, stratify=deals["won"]
    )
    rep_stats = _read("rep_stats.json", args.artifact_dir)
    X_test, _, _ = build_training_matrix(
        test_deals, activities, rep_stats["by_rep"], seed=args.seed + 1
    )
    drivers = global_shap_importance(booster, X_test)

    model_version = args.model_version or f"demo-{date.today().isoformat()}"
    metrics = {
        "generated_on": date.today().isoformat(),
        "model_version": model_version,
        "synthetic_data": True,
        "n_features": schema["n_features"],
        "data": manifest,
        "demo_models": {
            "win_probability": xgb_metrics,
            "days_to_close": lstm_metrics,
        },
        "paper_reference": PAPER_METRICS,
        "global_drivers": drivers,
        "latency_ms": {
            "measured": None,
            "note": (
                "Measured on the hosted deployment in phase 9 and reported in "
                "the README. Not yet measured; the paper's 180 ms figure is "
                "from different hardware and is not reused here."
            ),
        },
    }

    metrics_path = os.path.join(args.artifact_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    card_path = os.path.join(args.artifact_dir, "model_card.md")
    with open(card_path, "w", encoding="utf-8") as fh:
        fh.write(render_model_card(metrics))

    print(f"  wrote {metrics_path}")
    print(f"  wrote {card_path}")
    print(f"  native TreeSHAP: {len(drivers)} drivers ranked")
    print(f"  top driver: {drivers[0]['label']} ({drivers[0]['mean_abs_shap']:.4f})")


if __name__ == "__main__":
    main()
