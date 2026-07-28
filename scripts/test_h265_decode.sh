#!/usr/bin/env bash
# 测试 h265 文件能否被 mppvideodec 解码
# 用法：
#   ./scripts/test_h265_decode.sh recordings/video0_480.h265
#
# 参数：
#   $1: h265 文件路径
#
# 行为：
#   1. 用 fakesink 完整 pipeline 测解码
#   2. 用 fpsdisplaysink 看实际解码 fps
#   3. 报告文件健康度

set -euo pipefail

H265_FILE="${1:?Usage: $0 <h265_file>}"

if [[ ! -f "$H265_FILE" ]]; then
    echo "[error] 文件不存在: $H265_FILE" >&2
    exit 1
fi

# 1. NAL 结构分析
echo "=========================================="
echo "文件 NAL 结构分析"
echo "=========================================="
python3 <<EOF
import os
with open("$H265_FILE", 'rb') as f:
    data = f.read(min(500000, os.path.getsize("$H265_FILE")))
nals = []
i = 0
while i < len(data) - 4:
    if data[i:i+4] == b'\x00\x00\x00\x01':
        j = i + 4
        while j < len(data) - 4:
            if data[j:j+4] == b'\x00\x00\x00\x01' or data[j:j+3] == b'\x00\x00\x01':
                break
            j += 1
        else:
            j = len(data)
        nt = (data[i+4] >> 1) & 0x3F
        nals.append((i, nt, j-i))
    i += 1
type_names = {32:'VPS', 33:'SPS', 34:'PPS', 39:'SEI', 19:'IDR_W_RADL', 20:'IDR_N_LP', 0:'TRAIL_N', 1:'TRAIL_R'}
type_count = {}
for _, t, _ in nals:
    type_count[t] = type_count.get(t, 0) + 1
print(f"  总 NAL 数: {len(nals)}")
for t, c in sorted(type_count.items()):
    tname = type_names.get(t, f'type{t}')
    print(f"  {tname}({t}): {c} 个")
for pos, t, sz in nals:
    if t in {19, 20}:
        print(f"  第一个 IDR at offset {pos}, length {sz} bytes")
        break
else:
    print("  ⚠️  无 IDR 帧")
EOF

# 2. gst-launch 测解码
echo ""
echo "=========================================="
echo "gst-launch 解码测试 (fakesink)"
echo "=========================================="
echo "[test] v4l2src 模拟 pipeline..."
START=$(date +%s.%N)
timeout 8 gst-launch-1.0 -e filesrc location="$H265_FILE" \
    ! h265parse ! mppvideodec ! videoconvert ! fakesink 2>&1 | tail -3
END=$(date +%s.%N)
ELAPSED=$(awk -v s="$START" -v e="$END" 'BEGIN{print e-s}')
echo "[test] elapsed=${ELAPSED}s"

# 3. dmesg 错误
echo ""
echo "=========================================="
echo "dmesg 错误统计"
echo "=========================================="
echo "rk_vcodec 错误: $(dmesg 2>/dev/null | grep -c rk_vcodec || echo 0)"
echo "H265D_PARSER 错误: $(dmesg 2>/dev/null | grep -c H265D_PARSER || echo 0)"
echo ""
echo "建议："
echo "  - 如果 0 帧输出 + 错误多：清理 dmesg 重试 (./scripts/cleanup_dmesg.sh)"
echo "  - 如果 IDR 帧长度 > 100KB：可能是 GOP 损坏，需要重新录制"
echo "  - 实时摄像头比文件更稳定：./scripts/run_camera_demo.sh"
