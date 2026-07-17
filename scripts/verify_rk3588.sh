#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -gt 1 ]]; then
  echo "用法: scripts/verify_rk3588.sh [/dev/videoN]" >&2
  exit 2
fi

CAMERA_SOURCE="${1:-/dev/video6}"
BUILD_DIR="${BUILD_DIR:-build-rk3588}"
JOBS="${JOBS:-4}"
HOST_ARCH="$(uname -m)"

fail() {
  echo "[RK3588 预检] 失败：$*" >&2
  exit 1
}

warn() {
  echo "[RK3588 预检] 警告：$*" >&2
}

case "$HOST_ARCH" in
  aarch64|arm64) ;;
  *) fail "当前架构为 $HOST_ARCH，必须在 RK3588 的 aarch64/arm64 系统运行。" ;;
esac

for required_command in cmake c++ sha256sum ldd; do
  command -v "$required_command" >/dev/null 2>&1 || \
    fail "缺少命令 $required_command，请先安装板端构建依赖。"
done

[[ -c "$CAMERA_SOURCE" ]] || fail "相机节点不存在或不是字符设备：$CAMERA_SOURCE"
[[ -r "$CAMERA_SOURCE" && -w "$CAMERA_SOURCE" ]] || \
  fail "当前用户没有相机读写权限：$CAMERA_SOURCE"

if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl -d "$CAMERA_SOURCE" --all >/dev/null 2>&1 || \
    warn "v4l2-ctl 无法读取 $CAMERA_SOURCE，请确认它是彩色相机节点。"
else
  warn "未安装 v4l2-ctl，跳过相机格式检查。"
fi

echo "[RK3588 预检] 校验 FP16 RKNN 模型..."
sha256sum -c models/rk3588/SHA256SUMS

echo "[RK3588 预检] 校验 RKNN Runtime..."
(cd third_party/rknn && sha256sum -c SHA256SUMS)

RKNN_SDK_ROOT="${RKNN_SDK_ROOT:-$PWD/third_party/rknn}"
RKNN_HEADER=""
for header_candidate in \
  "$RKNN_SDK_ROOT/include/rknn_api.h" \
  "$RKNN_SDK_ROOT/runtime/Linux/librknn_api/include/rknn_api.h" \
  "$RKNN_SDK_ROOT/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h"; do
  if [[ -f "$header_candidate" ]]; then
    RKNN_HEADER="$header_candidate"
    break
  fi
done
[[ -n "$RKNN_HEADER" ]] || fail "RKNN SDK 缺少 rknn_api.h：$RKNN_SDK_ROOT"

RKNN_RUNTIME=""
for runtime_candidate in \
  "$RKNN_SDK_ROOT/aarch64/librknnrt.so" \
  "$RKNN_SDK_ROOT/lib/librknnrt.so" \
  "$RKNN_SDK_ROOT/lib64/librknnrt.so" \
  "$RKNN_SDK_ROOT/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
  "$RKNN_SDK_ROOT/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"; do
  if [[ -f "$runtime_candidate" ]]; then
    RKNN_RUNTIME="$runtime_candidate"
    break
  fi
done
[[ -n "$RKNN_RUNTIME" ]] || fail "RKNN SDK 缺少 librknnrt.so：$RKNN_SDK_ROOT"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  warn "当前会话没有 DISPLAY/WAYLAND_DISPLAY；可完成编译，但 Qt 界面可能无法启动。"
fi

echo "[RK3588 预检] 强制配置并编译 RKNN NPU 版本..."
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DDOUBLE_OK_BUILD_QT_DEMO=ON \
  -DDOUBLE_OK_BUILD_CAMERA_CHECK=OFF \
  -DDOUBLE_OK_ENABLE_RKNN=ON \
  -DDOUBLE_OK_REQUIRE_RKNN=ON \
  -DRKNN_SDK_ROOT="$RKNN_SDK_ROOT"
cmake --build "$BUILD_DIR" --target double-ok-demo -j "$JOBS"

RUNTIME_DIR="$(dirname "$RKNN_RUNTIME")"
LDD_OUTPUT="$(LD_LIBRARY_PATH="$RUNTIME_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$BUILD_DIR/double-ok-demo")"
if grep -q 'not found' <<< "$LDD_OUTPUT"; then
  printf '%s\n' "$LDD_OUTPUT" >&2
  fail "运行时动态库不完整。"
fi

echo "[RK3588 预检] 通过。下一步执行：scripts/run_demo.sh $CAMERA_SOURCE"
