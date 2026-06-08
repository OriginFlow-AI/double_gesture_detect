#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src "$PYTHON_BIN" -m double_ok_gesture.demo \
  --camera "${1:-0}" \
  --config configs/default.json \
  --model models/ok_hand_numpy_logreg.pkl \
  --capture-gate
