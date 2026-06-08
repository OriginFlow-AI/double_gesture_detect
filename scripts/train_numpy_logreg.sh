#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src "$PYTHON_BIN" -m double_ok_gesture.train \
  --input data/processed/hagrid_ok_features.csv \
  --output models/ok_hand_numpy_logreg.pkl \
  --model numpy_logreg
