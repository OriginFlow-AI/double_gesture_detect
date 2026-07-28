#!/usr/bin/env bash
# 跑实时摄像头测试 (用 v4l2src 直接拉摄像头流)
# 用法：
#   ./scripts/run_camera_demo.sh                # 默认 /dev/video0, 2560x1024
#   ./scripts/run_camera_demo.sh /dev/video2 3840 1080
#
# 参数：
#   $1: v4l2 设备路径（默认 /dev/video0）
#   $2: width（默认 2560）
#   $3: height（默认 1024）
#
# 环境变量：
#   VNC_PORT: VNC 端口（默认 5900）
#   DETECTION_INTERVAL: 跳帧检测间隔（默认 5）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${1:-/dev/video0}"
WIDTH="${2:-2560}"
HEIGHT="${3:-1024}"
VNC_PORT="${VNC_PORT:-5900}"
DETECTION_INTERVAL="${DETECTION_INTERVAL:-5}"

# 检查设备
if [[ ! -c "$DEVICE" ]]; then
    echo "[error] 设备不存在: $DEVICE" >&2
    echo "        可用设备:" >&2
    ls -la /dev/video* 2>/dev/null | head -5 >&2
    exit 1
fi

# 检查占用
if lsof "$DEVICE" 2>/dev/null | grep -q double-ok; then
    echo "[warn] $DEVICE 被占用，先清理..."
    pkill -9 double-ok-demo 2>/dev/null || true
    sleep 2
fi

# 清理 dmesg
if command -v sudo >/dev/null 2>&1; then
    sudo dmesg -c >/dev/null 2>&1 || true
fi

# 检查可执行文件
if [[ ! -x "$ROOT_DIR/build_rk3576/double-ok-demo-hevc" ]]; then
    echo "[error] demo 未编译" >&2
    echo "        请先运行: ./scripts/build_rk3576.sh" >&2
    exit 1
fi

# 设置模型路径
PALM_MODEL="$ROOT_DIR/models/rk3576/palm_detection_mediapipe_2023feb_fp16.rknn"
HAND_MODEL="$ROOT_DIR/models/rk3576/handpose_estimation_mediapipe_2023feb_fp16.rknn"
CONFIG="$ROOT_DIR/configs/default.json"

# 启动 demo
echo "[demo] 启动实时摄像头: $DEVICE (${WIDTH}x${HEIGHT}) -> VNC :$VNC_PORT"
echo "[demo] 按 Ctrl+C 停止"

QT_QPA_PLATFORM="vnc:size=1280x720,port=$VNC_PORT" \
"$ROOT_DIR/build_rk3576/double-ok-demo-hevc" \
    --camera "$DEVICE" \
    --width "$WIDTH" --height "$HEIGHT" --camera-fps 30 --fourcc HEVC \
    --config "$CONFIG" \
    --landmark-backend rknn \
    --palm-model "$PALM_MODEL" \
    --hand-model "$HAND_MODEL" \
    --screenshot-dir "$ROOT_DIR/recordings/" \
    --dashboard-width 1280 --dashboard-height 720 \
    --detection-interval "$DETECTION_INTERVAL" \
    --log-level INFO
