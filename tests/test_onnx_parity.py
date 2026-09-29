"""Training-side ONNX parity (spec section 11).

``test_artifacts.py`` asserts the parity *recorded* when the shipped artifact was
built. This file tests the export *code path* itself against a freshly
initialised model, so a regression in ``export_onnx`` is caught even when nobody
has retrained.

PyTorch is a training-only dependency, so these tests skip when it is absent —
which is the expected state inside the runtime image.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is a training-only dependency")
ort = pytest.importorskip("onnxruntime")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

from train_lstm import PARITY_TOL, DaysToCloseLSTM, export_onnx  # noqa: E402

from app.features.build import SEQ_FEATURE_NAMES, SEQ_LEN  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(7)
    model = DaysToCloseLSTM(len(SEQ_FEATURE_NAMES))
    model.eval()
    path = str(tmp_path_factory.mktemp("onnx") / "lstm.onnx")
    export_onnx(model, path)
    return model, path


def _random_batch(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, SEQ_LEN, len(SEQ_FEATURE_NAMES))).astype(np.float32)


@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_onnx_matches_torch_at_several_batch_sizes(exported, batch_size):
    model, path = exported
    x = _random_batch(batch_size, seed=batch_size)

    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"activity_sequence": x})[0]

    assert actual.shape == expected.shape == (batch_size, 1)
    assert np.max(np.abs(expected - actual)) < PARITY_TOL


def test_export_writes_one_file_without_a_sidecar(exported):
    _, path = exported
    assert os.path.exists(path)
    assert not os.path.exists(path + ".data")


def test_padded_sequence_is_handled(exported):
    """A deal with two events is mostly zero padding; both engines must agree."""
    model, path = exported
    x = np.zeros((1, SEQ_LEN, len(SEQ_FEATURE_NAMES)), dtype=np.float32)
    x[0, -2:, :] = 0.4

    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"activity_sequence": x})[0]

    assert np.max(np.abs(expected - actual)) < PARITY_TOL
