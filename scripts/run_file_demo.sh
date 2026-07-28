#!/usr/bin/env bash
# 跑 h265 文件输入测试 (用录制的文件作为输入)
# 用法：
#   ./scripts/run_file_demo.sh recordings/video0_480.h265 2560 1024
#   ./scripts/run_file_demo.sh recordings/video2_480.h265 3840 1080
#   ./scripts/run_file_demo.sh                      # 默认 video0
#
# 参数：
#   $1: h265 文件路径（默认 recordings/video0_480.h265）
#   $2: width（默认 2560）
#   $3: height（默认 1024）
#
# 环境变量：
#   VNC_PORT: VNC 端口（默认 5900）
#   DETECTION_INTERVAL: 跳帧检测间隔（默认 5）
#   MAX_FRAMES: 单次运行最大帧数（默认 480，约 16s@30fps）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

H265_FILE="${1:-$ROOT_DIR/recordings/video0_480.h265}"
WIDTH="${2:-2560}"
HEIGHT="${3:-1024}"
VNC_PORT="${VNC_PORT:-5900}"
DETECTION_INTERVAL="${DETECTION_INTERVAL:-5}"
MAX_FRAMES="${MAX_FRAMES:-480}"

# 检查文件
if [[ ! -f "$H265_FILE" ]]; then
    echo "[error] 文件不存在: $H265_FILE" >&2
    echo "        可用录制示例:" >&2
    ls -la "$ROOT_DIR/recordings/"*.h265 2>/dev/null | head -5 >&2
    exit 1
fi

# 检查可执行文件
if [[ ! -x "$ROOT_DIR/build_rk3576/double-ok-demo-hevc" ]]; then
    echo "[error] demo 未编译: $ROOT_DIR/build_rk3576/double-ok-demo-hevc" >&2
    echo "        请先运行: ./scripts/build_rk3576.sh" >&2
    exit 1
fi

# 检查摄像头/视频设备空闲
if lsof /dev/video0 2>/dev/null | grep -q double-ok; then
    echo "[warn] /dev/video0 被占用，先清理..."
    pkill -9 double-ok-demo 2>/dev/null || true
    sleep 2
fi

# 清理 dmesg 错误（避免历史 rk_vcodec 错误影响 mppvideodec）
if command -v sudo >/dev/null 2>&1; then
    sudo dmesg -c >/dev/null 2>&1 || true
fi

# 设置模型路径
PALM_MODEL="$ROOT_DIR/models/rk3576/palm_detection_mediapipe_2023feb_fp16.rknn"
HAND_MODEL="$ROOT_DIR/models/rk3576/handpose_estimation_mediapipe_2023feb_fp16.rknn"
CONFIG="$ROOT_DIR/configs/default.json"

# 启动 demo
echo "[demo] 启动: $H265_FILE (${WIDTH}x${HEIGHT}) -> VNC :$VNC_PORT"
echo "[demo] 按 Ctrl+C 停止"

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
    --log-level INFO
