#!/usr/bin/env bash
# 编译 RK3576 板端 demo 与单元测试
# 用法：
#   ./scripts/build_rk3576.sh                # 默认 Release + 板端路径
#   ./scripts/build_rk3576.sh Debug          # Debug 构建
#   ./scripts/build_rk3576.sh clean          # 清理 build 目录
#
# 前置条件：
#   - apt: qtbase5-dev libopencv-dev libgstreamer1.0-dev librga-dev
#   - RKNN SDK: third_party/rknn/include + aarch64/librknnrt.so
#   - HEVC 摄像头设备 /dev/video0 (可选，用于测试)

set -euo pipefail

# 项目根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 构建配置
BUILD_TYPE="${1:-Release}"
BUILD_DIR="$ROOT_DIR/build_rk3576"
RKNN_SDK="${RKNN_SDK_ROOT:-$ROOT_DIR/third_party/rknn}"

echo "[build] BUILD_TYPE=$BUILD_TYPE"
echo "[build] BUILD_DIR=$BUILD_DIR"
echo "[build] RKNN_SDK=$RKNN_SDK"

# 检查 RKNN SDK
if [[ ! -f "$RKNN_SDK/include/rknn_api.h" ]]; then
    echo "[error] RKNN SDK 头文件未找到: $RKNN_SDK/include/rknn_api.h" >&2
    echo "        请安装 RKNN Toolkit2 SDK 或设置 RKNN_SDK_ROOT 环境变量" >&2
    exit 1
fi

# 清理模式
if [[ "$BUILD_TYPE" == "clean" ]]; then
    echo "[build] 清理 build 目录..."
    rm -rf "$BUILD_DIR"
    echo "[build] 完成"
    exit 0
fi

# 准备 build 目录
mkdir -p "$BUILD_DIR"

# CMake 配置
echo "[build] CMake 配置..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTING=ON \
    -DDOUBLE_OK_BUILD_QT_DEMO=ON \
    -DDOUBLE_OK_BUILD_CAMERA_CHECK=ON \
    -DRKNN_SDK_ROOT="$RKNN_SDK" \
    2>&1 | tail -10

# 编译 demo 主目标
echo "[build] 编译 double-ok-demo-hevc..."
cmake --build "$BUILD_DIR" --target double-ok-demo-hevc -- -j"$(nproc)" 2>&1 | tail -10

# 编译单元测试（如启用）
echo "[build] 编译单元测试..."
cmake --build "$BUILD_DIR" -- -j"$(nproc)" 2>&1 | tail -5 || true

# 检查结果
if [[ -x "$BUILD_DIR/double-ok-demo-hevc" ]]; then
    echo "[build] 成功: $BUILD_DIR/double-ok-demo-hevc"
    file "$BUILD_DIR/double-ok-demo-hevc"
else
    echo "[error] 编译失败" >&2
    exit 1
fi

# 列出生成的可执行文件
echo "[build] 生成产物:"
ls -la "$BUILD_DIR"/double-ok-* 2>/dev/null || true
ls -la "$BUILD_DIR"/*tests* 2>/dev/null || true
