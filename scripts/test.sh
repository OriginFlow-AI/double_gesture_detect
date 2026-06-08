#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src "$PYTHON_BIN" -m pytest -q
