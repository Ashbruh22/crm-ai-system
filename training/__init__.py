"""Offline training pipeline: synthetic data, model training, ONNX export.

Nothing in this package is imported by the runtime service. Its dependencies
(PyTorch, Optuna) live in requirements-train.txt and are deliberately absent
from the runtime image.
"""
