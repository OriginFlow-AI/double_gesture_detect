#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/prepare_hagrid.sh /path/to/hagrid_annotations"
  exit 2
fi

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTHONPATH=src "$PYTHON_BIN" -m double_ok_gesture.prepare_hagrid \
  --annotations-dir "$1" \
  --output data/processed/hagrid_ok_features.csv
