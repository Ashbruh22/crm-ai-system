"""Artifact loading with a fail-fast schema check (spec section 5).

Replaces the old startup path in ``app/main.py``, which imported TensorFlow at
module scope and loaded ``ml/xgboost_model.joblib`` plus a ``.keras`` file —
neither of which was in git, so a fresh clone could not boot.

Now: XGBoost from its native JSON, the LSTM from ONNX Runtime, and no deep
learning framework in the image at all.

Explanations use **XGBoost's own TreeSHAP** (``pred_contribs=True``) rather than
the ``shap`` package. Same algorithm, but computed by the library that built the
trees, so ``bias + sum(contribs)`` reproduces the model margin to ~3e-07. The
``shap`` package's ``TreeExplainer`` re-parses the booster itself and, against
xgboost 3.2, returned attributions that did not sum to the prediction (errors up
to 0.38 in log-odds). An explanation that disagrees with the score it explains is
worse than no explanation. Dropping the dependency also removes numba and
llvmlite from the runtime image.

The service **refuses to start** if ``artifacts/feature_schema.json`` disagrees
with ``app.features.build``. A silent mismatch is worse than downtime: the model
would score a vector whose columns mean something other than what it learned,
and every explanation on the dashboard would be confidently wrong.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.features.build import FEATURE_NAMES, SEQ_FEATURE_NAMES, SEQ_LEN, feature_schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")


class ArtifactError(RuntimeError):
    """Raised when artifacts are missing or disagree with the code."""


class FeatureSchemaMismatch(ArtifactError):
    """The frozen schema does not match app.features.build."""


@dataclass
class ModelRegistry:
    """Everything the scoring path needs, loaded once at startup."""

    artifact_dir: str = DEFAULT_ARTIFACT_DIR
    booster: Any = None
    lstm_session: Any = None
    rep_stats: dict[str, float] = field(default_factory=dict)
    global_win_rate: float = 0.4
    model_version: str = "unknown"
    metrics: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)

    # --- paths ---------------------------------------------------------------

    def path(self, name: str) -> str:
        return os.path.join(self.artifact_dir, name)

    def _require(self, name: str) -> str:
        p = self.path(name)
        if not os.path.exists(p):
            raise ArtifactError(
                f"missing artifact {name!r} in {self.artifact_dir}. "
                "Run `make train` (or .\\train.ps1) to build it."
            )
        return p

    # --- loading -------------------------------------------------------------

    def validate_schema(self) -> dict:
        """Compare the frozen schema against the live code. Raises on drift."""
        with open(self._require("feature_schema.json")) as fh:
            saved = json.load(fh)

        expected = feature_schema()
        if saved == expected:
            return saved

        saved_names = [f["name"] for f in saved.get("features", [])]
        expected_names = list(FEATURE_NAMES)

        details: list[str] = []
        if saved_names != expected_names:
            missing = [n for n in expected_names if n not in saved_names]
            extra = [n for n in saved_names if n not in expected_names]
            if missing:
                details.append(f"code has features the artifact lacks: {missing}")
            if extra:
                details.append(f"artifact has features the code lacks: {extra}")
            if not missing and not extra:
                details.append("same features in a different order")
        if saved.get("sequence", {}).get("len") != SEQ_LEN:
            details.append(
                f"sequence length {saved.get('sequence', {}).get('len')} != {SEQ_LEN}"
            )

        raise FeatureSchemaMismatch(
            "artifacts/feature_schema.json does not match app.features.build "
            "(" + "; ".join(details or ["schema metadata differs"]) + "). "
            "Retrain with `make train` so the model and the feature code agree."
        )

    def load(self) -> "ModelRegistry":
        """Load every artifact. Call once, at startup."""
        # Imported here rather than at module scope so that importing the
        # registry (for tests, or for `--help`) does not pull in xgboost.
        import onnxruntime as ort
        import xgboost as xgb

        self.schema = self.validate_schema()

        self.booster = xgb.Booster()
        self.booster.load_model(self._require("xgb_model.json"))

        self.lstm_session = ort.InferenceSession(
            self._require("lstm.onnx"), providers=["CPUExecutionProvider"]
        )
        self._check_lstm_signature()

        with open(self._require("rep_stats.json")) as fh:
            stats = json.load(fh)
        self.rep_stats = stats.get("by_rep", {})
        self.global_win_rate = stats.get("global_win_rate", 0.4)

        metrics_path = self.path("metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as fh:
                self.metrics = json.load(fh)
            self.model_version = self.metrics.get("model_version", "unknown")

        return self

    def _check_lstm_signature(self) -> None:
        """Fail fast if the ONNX graph is not the shape the features produce."""
        inputs = self.lstm_session.get_inputs()
        if len(inputs) != 1:
            raise ArtifactError(f"lstm.onnx expects {len(inputs)} inputs, want 1")

        shape = inputs[0].shape  # e.g. ['batch', 60, 7]
        if len(shape) != 3:
            raise ArtifactError(f"lstm.onnx input rank {len(shape)}, want 3")

        _, timesteps, n_feat = shape
        if timesteps != SEQ_LEN or n_feat != len(SEQ_FEATURE_NAMES):
            raise ArtifactError(
                f"lstm.onnx input is (batch, {timesteps}, {n_feat}) but the "
                f"feature code produces (batch, {SEQ_LEN}, "
                f"{len(SEQ_FEATURE_NAMES)}). Retrain and re-export."
            )

    # --- inference helpers ---------------------------------------------------

    def predict_win_prob(self, X: np.ndarray) -> np.ndarray:
        """Win probability for a (n, n_features) matrix."""
        import xgboost as xgb

        if X.ndim == 1:
            X = X.reshape(1, -1)
        dmatrix = xgb.DMatrix(X, feature_names=list(FEATURE_NAMES))
        return np.asarray(self.booster.predict(dmatrix), dtype=np.float64)

    def shap_contribs(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Exact TreeSHAP attributions in log-odds, plus the bias term.

        Returns ``(contribs, bias)`` where ``contribs`` is ``(n, n_features)``
        and ``bias + contribs[i].sum()`` equals the model's margin for row i.
        """
        import xgboost as xgb

        if X.ndim == 1:
            X = X.reshape(1, -1)
        dmatrix = xgb.DMatrix(X, feature_names=list(FEATURE_NAMES))
        raw = np.asarray(self.booster.predict(dmatrix, pred_contribs=True))
        # Last column is the bias, identical for every row.
        return raw[:, :-1], float(raw[0, -1])

    def predict_days_to_close(self, sequences: np.ndarray) -> np.ndarray:
        """Days to close for a (n, SEQ_LEN, n_seq_features) batch."""
        if sequences.ndim == 2:
            sequences = sequences[None, ...]
        out = self.lstm_session.run(
            None, {"activity_sequence": sequences.astype(np.float32)}
        )[0]
        # The training script scales targets by DAY_SCALE before fitting.
        scale = (
            self.metrics.get("demo_models", {})
            .get("days_to_close", {})
            .get("day_scale", 100.0)
        )
        return np.asarray(out, dtype=np.float64).reshape(-1) * scale

    def health(self) -> dict:
        return {
            "model_version": self.model_version,
            "xgb_loaded": self.booster is not None,
            "lstm_loaded": self.lstm_session is not None,
            "explainer": "xgboost-treeshap",
            "n_features": len(FEATURE_NAMES),
            "sequence_len": SEQ_LEN,
            "reps_known": len(self.rep_stats),
        }


#: Process-wide registry. Populated by the app's lifespan handler.
registry = ModelRegistry()
