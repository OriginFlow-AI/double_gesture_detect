#!/usr/bin/env bash
# 跑 h265 文件循环播放（自动重启）
# 用法：
#   ./scripts/run_loop_demo.sh recordings/video0_480.h265 2560 1024
#   ./scripts/run_loop_demo.sh                      # 默认 video0
#
# 行为：每次跑完（EOS 或 max-frames）后 sleep 2s 自动重启
# 适用：长时间 VNC 演示
#
# 停止：按 Ctrl+C，会清理所有 demo 进程

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

H265_FILE="${1:-$ROOT_DIR/recordings/video0_480.h265}"
WIDTH="${2:-2560}"
HEIGHT="${3:-1024}"
VNC_PORT="${VNC_PORT:-5900}"
DETECTION_INTERVAL="${DETECTION_INTERVAL:-5}"
MAX_FRAMES="${MAX_FRAMES:-480}"
PAUSE_BETWEEN="${PAUSE_BETWEEN:-2}"

# 优雅退出处理
cleanup() {
    echo "[loop] 收到退出信号，清理 demo 进程..."
    pkill -9 double-ok-demo 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

if [[ ! -f "$H265_FILE" ]]; then
    echo "[error] 文件不存在: $H265_FILE" >&2
    exit 1
fi

if [[ ! -x "$ROOT_DIR/build_rk3576/double-ok-demo-hevc" ]]; then
    echo "[error] demo 未编译，请先: ./scripts/build_rk3576.sh" >&2
    exit 1
fi

# 清理 dmesg
if command -v sudo >/dev/null 2>&1; then
    sudo dmesg -c >/dev/null 2>&1 || true
fi

PALM_MODEL="$ROOT_DIR/models/rk3576/palm_detection_mediapipe_2023feb_fp16.rknn"
HAND_MODEL="$ROOT_DIR/models/rk3576/handpose_estimation_mediapipe_2023feb_fp16.rknn"
CONFIG="$ROOT_DIR/configs/default.json"

echo "[loop] 循环播放: $H265_FILE (${WIDTH}x${HEIGHT})"
echo "[loop] max-frames=$MAX_FRAMES, VNC :$VNC_PORT, 间隔 ${PAUSE_BETWEEN}s"
echo "[loop] 按 Ctrl+C 停止"

while true; do
    echo "[loop] [$(date +%H:%M:%S)] 启动 demo"
    QT_QPA_PLATFORM="vnc:size=1280x720,port=$VNC_PORT" \
    "$ROOT_DIR/build_rk3576/double-ok-demo-hevc" \
        --camera "$H265_FILE" \
        --width "$WIDTH" --height "$HEIGHT" --camera-fps 30 --fourcc HEVC \
        --config "$CONFIG" \
        --landmark-backend rknn \
        --palm-model "$PALM_MODEL" \
        --hand-model "$HAND_MODEL" \
        --screenshot-dir "$ROOT_DIR/recordings/" \
        --dashboard-width 1280 --dashboard-height 720 \
        --detection-interval "$DETECTION_INTERVAL" \
        --max-frames "$MAX_FRAMES" \
        --log-level INFO || true
    echo "[loop] [$(date +%H:%M:%S)] demo 退出, ${PAUSE_BETWEEN}s 后重启"
    sleep "$PAUSE_BETWEEN"
done
