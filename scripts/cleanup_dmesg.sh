#!/usr/bin/env bash
# 清理内核日志 (解决 mppvideodec 因 dmesg 错误累积导致的卡死)
# 用法：
#   ./scripts/cleanup_dmesg.sh         # 清理 dmesg
#   ./scripts/cleanup_dmesg.sh status  # 查看当前错误数
#
# 背景：
#   RK3576 mppvideodec 在处理 H265 流时可能产生大量 "rk_vcodec" 错误
#   当 dmesg 错误累积到几千条时，mpp 内核模块进入异常状态，decoder 0 帧
#   解决方案：定期清理 dmesg 缓冲区

set -euo pipefail

if [[ "${1:-}" == "status" ]]; then
    COUNT=$(dmesg 2>/dev/null | grep -c "rk_vcodec\|H265D_PARSER" || true)
    COUNT=${COUNT:-0}
    echo "[status] rk_vcodec / H265D_PARSER 错误数: $COUNT"
    if [[ "$COUNT" -gt 1000 ]]; then
        echo "[status] ⚠️  错误过多，建议运行: ./scripts/cleanup_dmesg.sh"
    else
        echo "[status] ✅ 错误数正常"
    fi
    exit 0
fi

# 检查 sudo
if ! command -v sudo >/dev/null 2>&1; then
    echo "[error] sudo 未安装" >&2
    exit 1
fi

# 清理 dmesg
echo "[cleanup] 清理 dmesg 内核日志..."
sudo dmesg -c >/dev/null 2>&1

# 验证
sleep 1
COUNT=$(dmesg 2>/dev/null | grep -c "rk_vcodec\|H265D_PARSER" || echo 0)
echo "[cleanup] 清理后错误数: $COUNT"

if [[ "$COUNT" -gt 100 ]]; then
    echo "[warn] 错误数仍较多，可能需要重启板子"
    echo "       sudo reboot"
fi
