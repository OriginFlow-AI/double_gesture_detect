#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/prepare_hagrid.sh /path/to/hagrid_annotations"
  exit 2
fi

cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" >/dev/null
cmake --build "$BUILD_DIR" --target double-ok-prepare-hagrid -j "$JOBS" >/dev/null

"$BUILD_DIR/double-ok-prepare-hagrid" \
  --annotations-dir "$1" \
  --output data/processed/hagrid_ok_features.csv
