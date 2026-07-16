#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build-rk3588}"
JOBS="${JOBS:-2}"
RKNN_ROOT="${RKNN_ROOT:-$PWD/third_party/rknn_runtime}"
TOOLCHAIN_FILE="$PWD/cmake/toolchains/rk3588-linux-aarch64.cmake"

if [[ ! -f "$RKNN_ROOT/include/rknn_api.h" || ! -f "$RKNN_ROOT/aarch64/librknnrt.so" ]]; then
  echo "RKNN_ROOT must contain include/rknn_api.h and aarch64/librknnrt.so: $RKNN_ROOT" >&2
  exit 2
fi

compiler_prefix="${RK3588_TOOLCHAIN_PREFIX:-aarch64-linux-gnu-}"
if ! command -v "${compiler_prefix}g++" >/dev/null 2>&1; then
  echo "RK3588 AArch64 compiler not found: ${compiler_prefix}g++" >&2
  echo "Set RK3588_TOOLCHAIN_PREFIX and RK3588_SYSROOT to the BSP toolchain/sysroot." >&2
  exit 2
fi
if [[ -z "${RK3588_SYSROOT:-}" || ! -d "${RK3588_SYSROOT}" ]]; then
  echo "RK3588_SYSROOT must point to an RK3588 target sysroot containing OpenCV." >&2
  exit 2
fi

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DRKNN_ROOT="$RKNN_ROOT" \
  -DDOUBLE_OK_ENABLE_RKNN=ON \
  -DDOUBLE_OK_BUILD_QT_DEMO=OFF \
  -DDOUBLE_OK_BUILD_CAPTURE_TOOL=OFF
cmake --build "$BUILD_DIR" --target double-ok-headless double-ok-camera-check -j "$JOBS"
