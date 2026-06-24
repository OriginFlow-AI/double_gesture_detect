#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CAMERA_SOURCE="${1:-/dev/video0}"
if [[ $# -gt 0 ]]; then
  shift
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/lib"

exec bin/double-ok-headless \
  --camera "$CAMERA_SOURCE" \
  --config configs/rv1126_uvc.json \
  --capture-gate \
  --headless \
  --landmark-backend rknn \
  "$@"
