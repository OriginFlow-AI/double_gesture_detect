#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-gui -j "$JOBS" >/dev/null

"$BUILD_DIR/double-ok-gui" \
  --config configs/default.json \
  --csv data/processed/hagrid_ok_features.csv \
  --model models/ok_hand_numpy_logreg.pkl \
  --output reports/gui/index.html
