#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${RV1126_TOOLCHAIN_PREFIX:?Set RV1126_TOOLCHAIN_PREFIX, for example arm-linux-gnueabihf-}"
: "${RV1126_SYSROOT:?Set RV1126_SYSROOT to the RV1126 target sysroot containing OpenCV and runtime libraries}"

BUILD_DIR="${BUILD_DIR:-build-rv1126}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-2}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
  -DDOUBLE_OK_BUILD_QT_DEMO=OFF \
  -DDOUBLE_OK_BUILD_CAPTURE_TOOL=OFF \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/rv1126-linux-gnueabihf.cmake \
  -DCMAKE_PREFIX_PATH="$RV1126_SYSROOT/usr;$RV1126_SYSROOT/usr/local"

cmake --build "$BUILD_DIR" --target double-ok-headless double-ok-camera-check -j "$JOBS"
