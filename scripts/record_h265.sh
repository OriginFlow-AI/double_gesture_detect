#!/usr/bin/env bash
# 录制 HEVC 视频文件
# 用法：
#   ./scripts/record_h265.sh                    # 默认 /dev/video0, 16s, 2560x1024
#   ./scripts/record_h265.sh /dev/video2 3840 1080
#   ./scripts/record_h265.sh /dev/video0 2560 1024 30 my_video.h265
#
# 参数：
#   $1: v4l2 设备路径
#   $2: width
#   $3: height
#   $4: 录制时长（秒，默认 16）
#   $5: 输出文件名（默认 recordings/<device>_N.h265）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${1:-/dev/video0}"
WIDTH="${2:-2560}"
HEIGHT="${3:-1024}"
DURATION="${4:-16}"

# 输出文件名
if [[ -n "${5:-}" ]]; then
    OUT_FILE="$5"
    if [[ "$OUT_FILE" != /* ]]; then
        OUT_FILE="$ROOT_DIR/recordings/$OUT_FILE"
    fi
else
    # 自动命名：recordings/videoN_<timestamp>.h265
    mkdir -p "$ROOT_DIR/recordings"
    OUT_FILE="$ROOT_DIR/recordings/$(basename "$DEVICE")_$(date +%H%M%S).h265"
fi

# 检查设备
if [[ ! -c "$DEVICE" ]]; then
    echo "[error] 设备不存在: $DEVICE" >&2
    exit 1
fi

# 检查占用
if lsof "$DEVICE" 2>/dev/null | grep -q .; then
    echo "[warn] $DEVICE 被占用，先清理..."
    pkill -9 double-ok-demo 2>/dev/null || true
    pkill -9 gst-launch 2>/dev/null || true
    sleep 2
fi

# 检查 gst-launch
if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
    echo "[error] gst-launch-1.0 未安装" >&2
    exit 1
fi

echo "[record] 录制 $DURATION 秒: $DEVICE (${WIDTH}x${HEIGHT})"
echo "[record] 输出: $OUT_FILE"

# 录制（用 h265parse config-interval=-1 强制每个 IDR 注入 VPS/SPS/PPS）
timeout "$DURATION" gst-launch-1.0 -e \
    v4l2src device="$DEVICE" \
    ! "video/x-h265,stream-format=byte-stream,width=$WIDTH,height=$HEIGHT,framerate=30/1" \
    ! h265parse config-interval=-1 \
    ! filesink location="$OUT_FILE" \
    2>&1 | tail -3

# 验证
if [[ -f "$OUT_FILE" ]]; then
    SIZE=$(stat -c %s "$OUT_FILE")
    echo "[record] 完成: $OUT_FILE ($SIZE bytes, $(awk "BEGIN{printf \"%.1f\", $SIZE/1024/1024}") MB)"
else
    echo "[error] 录制失败" >&2
    exit 1
fi
