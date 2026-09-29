"""Train the days-to-close LSTM and export it to ONNX (spec section 5).

Writes to ``artifacts/``:

* ``lstm.onnx``          the served model, opset 17
* ``lstm_metrics.json``  MAE / RMSE in days, plus the ONNX parity result

Why PyTorch rather than Keras
-----------------------------
The original research model was Keras. It is rebuilt here in PyTorch for one
reason: ``torch.onnx.export`` handles LSTMs reliably, whereas ``tf2onnx`` has
poor Keras 3 support. The runtime never sees either framework — it loads
``lstm.onnx`` through onnxruntime (~15 MB against TensorFlow's ~600 MB), which
is what brings the service inside the 512 MB budget.

Input is the fixed ``(60, 7)`` pre-padded activity sequence from
``app.features.build.build_sequence``. The batch axis is dynamic; the 60
timesteps are not.

Target is days from the truncation point to the close date, matching how the
service uses it: how much longer will this open deal take?
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from app.features.build import SEQ_FEATURE_NAMES, SEQ_LEN, build_sequence  # noqa: E402

SEED = 42
DATA_DIR = os.path.join(REPO_ROOT, "data", "synthetic")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

HIDDEN = 48
EPOCHS = 60
BATCH = 64
LR = 3e-3
#: Targets are divided by this before training and multiplied back after, which
#: keeps the loss in a sane range without a separate scaler artifact.
DAY_SCALE = 100.0
TRUNCATE_MIN, TRUNCATE_MAX = 0.30, 0.95
#: Max absolute difference in days allowed between torch and onnxruntime.
PARITY_TOL = 1e-3


class DaysToCloseLSTM(nn.Module):
    """Single-layer LSTM over the activity sequence, read out at the last step.

    Pre-padding puts the most recent event at the final timestep, so the last
    hidden state always sees it regardless of how many events the deal has.
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Softplus keeps the prediction positive — a deal cannot close in
        # negative days, and an unbounded head sometimes says it will.
        return nn.functional.softplus(self.head(out[:, -1, :]))


def build_sequences(
    deals: pd.DataFrame,
    activities: pd.DataFrame,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_deal = {
        deal_id: group[["type", "occurred_at"]].to_dict("records")
        for deal_id, group in activities.groupby("deal_id", sort=False)
    }

    seqs, targets = [], []
    for deal in deals.to_dict("records"):
        life = (deal["closed_at"] - deal["created_at"]).total_seconds()
        frac = rng.uniform(TRUNCATE_MIN, TRUNCATE_MAX)
        as_of = deal["created_at"] + pd.Timedelta(seconds=life * frac)

        seen = [a for a in by_deal.get(deal["id"], []) if a["occurred_at"] <= as_of]
        if not seen:
            continue

        days_remaining = (deal["closed_at"] - as_of).total_seconds() / 86400.0
        if days_remaining <= 0:
            continue

        seqs.append(build_sequence(deal, seen))
        targets.append(days_remaining)

    return (
        np.stack(seqs).astype(np.float32),
        np.asarray(targets, dtype=np.float32).reshape(-1, 1),
    )


def export_onnx(model: nn.Module, path: str) -> None:
    """Export a single self-contained .onnx with a dynamic batch axis.

    ``dynamo=False`` selects the legacy TorchScript exporter deliberately. The
    dynamo exporter spills weights into a sidecar ``lstm.onnx.data`` file and
    hard-codes the output shape to ``[1, 1]``; the first breaks any deploy that
    copies only the ``.onnx``, the second makes batch inference warn on every
    call. The legacy path honours ``dynamic_axes`` and writes one file.
    """
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, len(SEQ_FEATURE_NAMES), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["activity_sequence"],
        output_names=["days_to_close"],
        dynamic_axes={
            "activity_sequence": {0: "batch"},
            "days_to_close": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # A stray sidecar from an earlier dynamo export would shadow the real
    # weights on the next load, so clear it out.
    sidecar = path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)


def check_parity(model: nn.Module, onnx_path: str, X: np.ndarray) -> dict:
    """Confirm onnxruntime reproduces the torch output within tolerance."""
    import onnxruntime as ort

    model.eval()
    sample = X[:256]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"activity_sequence": sample})[0]

    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    return {
        "n_compared": int(len(sample)),
        "max_abs_diff_days": round(max_abs, 8),
        "tolerance": PARITY_TOL,
        "passed": bool(max_abs < PARITY_TOL),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    deals = pd.read_csv(os.path.join(args.data_dir, "deals.csv"))
    activities = pd.read_csv(os.path.join(args.data_dir, "activities.csv"))
    for col in ("created_at", "closed_at"):
        deals[col] = pd.to_datetime(deals[col])
    activities["occurred_at"] = pd.to_datetime(activities["occurred_at"])

    X, y = build_sequences(deals, activities, seed=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=args.seed
    )

    model = DaysToCloseLSTM(len(SEQ_FEATURE_NAMES))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    fit_x = torch.from_numpy(X_fit)
    fit_y = torch.from_numpy(y_fit / DAY_SCALE)
    val_x = torch.from_numpy(X_val)
    val_y = torch.from_numpy(y_val / DAY_SCALE)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(fit_x))
        for i in range(0, len(perm), BATCH):
            idx = perm[i : i + BATCH]
            optimiser.zero_grad()
            loss = loss_fn(model(fit_x[idx]), fit_y[idx])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_mae = float(
                torch.mean(torch.abs(model(val_x) - val_y)).item() * DAY_SCALE
            )

        if val_mae < best_val - 1e-4:
            best_val, patience = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_test)).numpy() * DAY_SCALE

    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    # Baseline: always predict the training mean. A model that cannot beat this
    # is not worth serving, so the number goes in the model card.
    baseline_mae = float(np.mean(np.abs(y_test - y_train.mean())))

    onnx_path = os.path.join(args.artifact_dir, "lstm.onnx")
    export_onnx(model, onnx_path)
    parity = check_parity(model, onnx_path, X_test)

    metrics = {
        "mae_days": round(mae, 3),
        "rmse_days": round(rmse, 3),
        "baseline_mae_days": round(baseline_mae, 3),
        "improvement_vs_baseline": round(1.0 - mae / baseline_mae, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "input_shape": [SEQ_LEN, len(SEQ_FEATURE_NAMES)],
        "day_scale": DAY_SCALE,
        "onnx_opset": 17,
        "onnx_parity": parity,
    }
    with open(os.path.join(args.artifact_dir, "lstm_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  saved {onnx_path} ({size_mb:.2f} MB)")
    print(
        f"  demo model (synthetic data): MAE {mae:.1f}d | RMSE {rmse:.1f}d | "
        f"baseline MAE {baseline_mae:.1f}d "
        f"({metrics['improvement_vs_baseline']:+.1%} vs baseline)"
    )
    print(
        f"  ONNX parity: max abs diff {parity['max_abs_diff_days']:.2e} days "
        f"over {parity['n_compared']} rows -> "
        f"{'PASS' if parity['passed'] else 'FAIL'}"
    )

    if not parity["passed"]:
        raise SystemExit("ONNX parity check failed; refusing to ship this artifact.")


if __name__ == "__main__":
    main()
