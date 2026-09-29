# CRM AI Core — hosted demo build targets.
#
# `make train` is the spec's phase 1 acceptance criterion: it regenerates the
# synthetic data and both model artifacts from a fixed seed, so a reviewer can
# reproduce everything in artifacts/ from a clean checkout.

SEED ?= 42
PY ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))

# Torch's ONNX exporter prints emoji; a cp1252 Windows console raises
# UnicodeEncodeError on them.
export PYTHONIOENCODING = utf-8

.PHONY: help venv install install-train data train train-xgb train-lstm evaluate test test-fast lint clean-artifacts

help:
	@echo "make venv          create .venv"
	@echo "make install       runtime deps only (what the image gets)"
	@echo "make install-train runtime + training deps (torch, faker, optuna)"
	@echo "make data          regenerate synthetic CRM data from SEED"
	@echo "make train         data + xgb + lstm(+onnx) + metrics + model card"
	@echo "make test          full test suite"
	@echo "make test-fast     skip the full-scale regeneration test"

venv:
	python -m venv .venv
	@echo "created .venv — now run: make install-train"

install:
	$(PY) -m pip install -r requirements.txt

install-train:
	$(PY) -m pip install -r requirements.txt -r requirements-train.txt

data:
	@echo "==> generating synthetic CRM data (seed $(SEED))"
	$(PY) training/generate_synthetic.py --seed $(SEED)

train-xgb:
	@echo "==> training XGBoost win-probability classifier"
	$(PY) training/train_xgb.py --seed $(SEED)

train-lstm:
	@echo "==> training days-to-close LSTM and exporting to ONNX"
	$(PY) training/train_lstm.py --seed $(SEED)

evaluate:
	@echo "==> writing metrics.json and model_card.md"
	$(PY) training/evaluate.py --seed $(SEED)

# Order matters: evaluate reads both models' metrics files.
train: data train-xgb train-lstm evaluate
	@echo ""
	@echo "==> artifacts/ ready:"
	@ls -1 artifacts/

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

lint:
	$(PY) -m pytest --collect-only -q > /dev/null && echo "collection ok"

clean-artifacts:
	rm -f artifacts/*.json artifacts/*.onnx artifacts/*.onnx.data artifacts/model_card.md
	rm -rf data/synthetic
