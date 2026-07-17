#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep test-only CMake options from replacing desktop targets in build/.
BUILD_DIR="${BUILD_DIR:-build-test}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"
BUILD_QT_DEMO="${DOUBLE_OK_BUILD_QT_DEMO:-OFF}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DBUILD_TESTING=ON \
  -DDOUBLE_OK_BUILD_QT_DEMO="$BUILD_QT_DEMO"
cmake --build "$BUILD_DIR" -j "$JOBS"
ctest --test-dir "$BUILD_DIR" --output-on-failure
