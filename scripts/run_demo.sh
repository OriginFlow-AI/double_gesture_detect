#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -gt 1 ]]; then
  echo "用法: scripts/run_demo.sh [/dev/videoN]" >&2
  exit 2
fi

CAMERA_SOURCE="${1:-/dev/video6}"
JOBS="${JOBS:-4}"
HOST_ARCH="$(uname -m)"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DDOUBLE_OK_BUILD_QT_DEMO=ON
)

case "$HOST_ARCH" in
  aarch64|arm64)
    BUILD_DIR="${BUILD_DIR:-build-rk3588}"
    LANDMARK_BACKEND="rknn"
    CMAKE_ARGS+=(
      -DDOUBLE_OK_ENABLE_RKNN=ON
      -DDOUBLE_OK_REQUIRE_RKNN=ON
    )
    if [[ -n "${RKNN_SDK_ROOT:-}" ]]; then
      CMAKE_ARGS+=(-DRKNN_SDK_ROOT="$RKNN_SDK_ROOT")
    fi
    echo "[double-ok] 检测到 $HOST_ARCH，使用 RK3588 NPU。" >&2
    ;;
  *)
    BUILD_DIR="${BUILD_DIR:-build-pc}"
    LANDMARK_BACKEND="onnx"
    CMAKE_ARGS+=(
      -DDOUBLE_OK_ENABLE_RKNN=OFF
      -DDOUBLE_OK_REQUIRE_RKNN=OFF
    )
    echo "[double-ok] 检测到 $HOST_ARCH，当前电脑没有 RK3588 NPU，使用 ONNX CPU 验证。" >&2
    ;;
esac

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  CACHED_ARCH="$(sed -n 's/^CMAKE_SYSTEM_PROCESSOR:UNINITIALIZED=//p; s/^CMAKE_SYSTEM_PROCESSOR:STRING=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)"
  if [[ -n "$CACHED_ARCH" && "$CACHED_ARCH" != "$HOST_ARCH" ]]; then
    echo "错误：$BUILD_DIR 是为 $CACHED_ARCH 生成的，请改用新的 BUILD_DIR。" >&2
    exit 2
  fi
fi

echo "[double-ok] 配置 $LANDMARK_BACKEND 版本..." >&2
cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"

echo "[double-ok] 编译实时界面..." >&2
cmake --build "$BUILD_DIR" --target double-ok-demo -j "$JOBS"

without_sogou_qt() {
  local value="${1:-}"
  local item=""
  local cleaned=""
  local entries=()
  IFS=':' read -r -a entries <<< "$value"
  for item in "${entries[@]}"; do
    [[ -z "$item" || "$item" == /opt/sogoupinyin/* ]] && continue
    cleaned="${cleaned:+$cleaned:}$item"
  done
  printf '%s' "$cleaned"
}

CLEAN_LD_LIBRARY_PATH="$(without_sogou_qt "${LD_LIBRARY_PATH:-}")"
CLEAN_QT_PLUGIN_PATH="$(without_sogou_qt "${QT_PLUGIN_PATH:-}")"
CLEAN_QT_PLATFORM_PLUGIN_PATH="$(without_sogou_qt "${QT_QPA_PLATFORM_PLUGIN_PATH:-}")"

echo "[double-ok] 启动相机 $CAMERA_SOURCE..." >&2
exec env \
  LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" \
  QT_PLUGIN_PATH="$CLEAN_QT_PLUGIN_PATH" \
  QT_QPA_PLATFORM_PLUGIN_PATH="$CLEAN_QT_PLATFORM_PLUGIN_PATH" \
  "$BUILD_DIR/double-ok-demo" \
  --camera "$CAMERA_SOURCE" \
  --config configs/default.json \
  --landmark-backend "$LANDMARK_BACKEND" \
  --capture-gate \
  --disable-auto-capture \
  --target-fps 15 \
  --log-level INFO
