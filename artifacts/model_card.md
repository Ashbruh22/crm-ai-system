# Model card — CRM AI Core (hosted demo)

**Generated:** 2026-09-29 · **Model version:** `demo-2026-09-29`

> ## Synthetic data only
> Both models below are trained on **generated data** from
> `training/generate_synthetic.py`. No real CRM records, customer names, or
> production data are used anywhere in this repository.
>
> Every figure on this card was measured on synthetic data and is reproducible
> by rerunning `make train` from the seed.

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
| Source | `training/generate_synthetic.py`, seed `42` |
| Historical deals | 2,000 (closed, labelled) |
| Activity events | 46,135 |
| Live demo deals | 60 (open, unlabelled) |
| Base win rate | 38.5% |
| Mean cycle | 90 days |
| Activities per deal | 23.1 |
| Real data used | **None** |

Deals are featurised at a **random mid-flight cut** (35–100% of deal life), not
at close, because the service scores open deals with partial activity logs.
Training on complete histories would put every served vector out of
distribution.

## Features

32 features, frozen in `artifacts/feature_schema.json` and
validated at service startup — the service refuses to boot on a mismatch.
Per-rep win rates are fitted on the **training split only** so a test-set
outcome never leaks into a training feature.

## Demo metrics (synthetic data)

### Win probability — XGBoost classifier

| Metric | Value |
|---|---|
| Accuracy | 0.738 |
| AUC-ROC | 0.788 |
| Precision | 0.709 |
| Recall | 0.539 |
| F1 | 0.613 |
| Brier score | 0.180 |
| Test rows | 400 |

### Days to close — LSTM regressor (served as ONNX)

| Metric | Value |
|---|---|
| MAE | 11.1 days |
| RMSE | 14.3 days |
| Mean-prediction baseline MAE | 15.2 days |
| Improvement vs baseline | +27.1% |
| Input shape | 60 timesteps x 7 features |
| ONNX parity (max abs diff) | 1.20e-07 days |

The LSTM is trained in PyTorch and served through ONNX Runtime. Neither PyTorch
nor TensorFlow is installed in the runtime image; the parity check above
confirms the exported graph reproduces the framework output.

## Top global drivers (mean |SHAP|)

| # | Driver | Mean abs SHAP |
|---|---|---|
| 1 | Champion identified | 0.5291 |
| 2 | 7-day silence gaps | 0.4300 |
| 3 | Email reply rate | 0.2582 |
| 4 | Days since last touch | 0.2341 |
| 5 | Meetings held | 0.1370 |
| 6 | Emails replied to | 0.1304 |
| 7 | Deal size ($k) | 0.1251 |
| 8 | Demos delivered | 0.1132 |
| 9 | Activities per week | 0.1092 |
| 10 | Total activities | 0.0747 |
| 11 | Rep historical win rate | 0.0693 |
| 12 | Days open | 0.0552 |

## Limitations

- Trained entirely on generated data; no external validity.
- The generator's win function is a logistic model over activity counts, so the
  classifier can in principle recover it almost exactly. Residual noise is what
  keeps measured AUC near 0.79 rather than 1.0.
- Rep win rate is a fitted statistic; a rep unseen at training time falls back
  to the global base rate.
- Calibration is reported (Brier 0.180) but the model is not
  explicitly calibrated; treat probabilities as ordinal.
- The days-to-close model sees only activity cadence and deal size — not
  seasonality, quota pressure, or procurement cycles.

