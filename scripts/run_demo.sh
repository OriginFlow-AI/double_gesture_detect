#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"

CAMERA_SOURCE="${1:-/dev/video0}"
if [[ $# -gt 0 ]]; then
  shift
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-demo -j "$JOBS" >/dev/null

"$BUILD_DIR/double-ok-demo" \
  --camera "$CAMERA_SOURCE" \
  --config configs/default.json \
  --model models/ok_hand_numpy_logreg.txt \
  --capture-gate \
  "$@"
