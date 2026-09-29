"""Train the XGBoost win-probability classifier (spec section 5).

Writes to ``artifacts/``:

* ``xgb_model.json``      native XGBoost format, loadable by ``TreeExplainer``
                          directly and free of any pickle/scikit-learn version pin
* ``feature_schema.json`` the frozen feature contract
* ``rep_stats.json``      per-rep historical win rate, fitted on the TRAIN split only

Mid-flight truncation
---------------------
Historical deals are not featurised at the moment they closed. Each one is cut
at a random point between 35% and 100% of its life and featurised there, with
only the activities that had happened by then. The label stays the final
outcome.

This matters: the service scores *open* deals, where ``days_open`` is small and
the activity log is partial. Training on complete histories would make every
served feature vector out-of-distribution, and the demo's scores meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from app.features.build import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_vector,
    feature_schema,
)

SEED = 42
DATA_DIR = os.path.join(REPO_ROOT, "data", "synthetic")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

#: Fraction-of-life window at which a historical deal is featurised.
TRUNCATE_MIN, TRUNCATE_MAX = 0.35, 1.0


def load_raw(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    deals = pd.read_csv(os.path.join(data_dir, "deals.csv"))
    activities = pd.read_csv(os.path.join(data_dir, "activities.csv"))
    for col in ("created_at", "closed_at"):
        deals[col] = pd.to_datetime(deals[col])
    activities["occurred_at"] = pd.to_datetime(activities["occurred_at"])
    return deals, activities


def build_training_matrix(
    deals: pd.DataFrame,
    activities: pd.DataFrame,
    rep_stats: dict[str, float],
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Featurise each deal at a random mid-flight cut. Returns (X, y, as_of)."""
    rng = np.random.default_rng(seed)
    by_deal = {
        deal_id: group[["type", "occurred_at"]].to_dict("records")
        for deal_id, group in activities.groupby("deal_id", sort=False)
    }

    rows, labels, as_ofs = [], [], []
    for deal in deals.to_dict("records"):
        life = (deal["closed_at"] - deal["created_at"]).total_seconds()
        frac = rng.uniform(TRUNCATE_MIN, TRUNCATE_MAX)
        as_of = deal["created_at"] + pd.Timedelta(seconds=life * frac)

        seen = [a for a in by_deal.get(deal["id"], []) if a["occurred_at"] <= as_of]
        rows.append(build_feature_vector(deal, seen, rep_stats=rep_stats, as_of=as_of))
        labels.append(int(deal["won"]))
        as_ofs.append(as_of)

    X = pd.DataFrame(np.vstack(rows), columns=list(FEATURE_NAMES))
    return X, np.asarray(labels), pd.Series(as_ofs, name="as_of")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    deals, activities = load_raw(args.data_dir)

    # Split on deals FIRST so rep win rates can be fitted without leaking the
    # test set's outcomes into a training feature.
    train_deals, test_deals = train_test_split(
        deals,
        test_size=0.2,
        random_state=args.seed,
        stratify=deals["won"],
    )

    global_rate = float(train_deals["won"].mean())
    rep_stats = (
        train_deals.groupby("owner_rep")["won"].mean().round(4).to_dict()
    )
    rep_stats = {str(k): float(v) for k, v in rep_stats.items()}

    X_train, y_train, _ = build_training_matrix(
        train_deals, activities, rep_stats, seed=args.seed
    )
    X_test, y_test, _ = build_training_matrix(
        test_deals, activities, rep_stats, seed=args.seed + 1
    )

    # Carve a validation slice out of train for early stopping.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=args.seed, stratify=y_train
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=40,
        random_state=args.seed,
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "best_iteration": int(model.best_iteration),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }

    model_path = os.path.join(args.artifact_dir, "xgb_model.json")
    model.get_booster().save_model(model_path)

    with open(os.path.join(args.artifact_dir, "feature_schema.json"), "w") as fh:
        json.dump(feature_schema(), fh, indent=2)

    with open(os.path.join(args.artifact_dir, "rep_stats.json"), "w") as fh:
        json.dump(
            {"global_win_rate": round(global_rate, 4), "by_rep": rep_stats},
            fh,
            indent=2,
        )

    with open(os.path.join(args.artifact_dir, "xgb_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"  saved {model_path}")
    print(
        "  demo model (synthetic data): "
        f"acc {metrics['accuracy']:.3f} | AUC {metrics['auc_roc']:.3f} | "
        f"P {metrics['precision']:.3f} | R {metrics['recall']:.3f} | "
        f"Brier {metrics['brier']:.3f} | best_iter {metrics['best_iteration']}"
    )


if __name__ == "__main__":
    main()
