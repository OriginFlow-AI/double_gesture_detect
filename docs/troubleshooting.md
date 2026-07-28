# 故障排除 (RK3576)

> **目标平台**：RK3576 板端
> **最后更新**：2026-07-28
> **相关脚本**：`scripts/cleanup_dmesg.sh`、`scripts/test_h265_decode.sh`

## 快速诊断

```bash
# 1. 检查 demo 进程
ps -ef | grep double-ok

# 2. 检查 VNC 端口
ss -tln | grep 5900

# 3. 检查 dmesg 错误
./scripts/cleanup_dmesg.sh status

# 4. 检查 demo 日志最新输出
tail -30 /tmp/demo_*.log

# 5. 测试 h265 文件
./scripts/test_h265_decode.sh your_file.h265
```

## 常见问题

### Q1: 启动后 VNC 看到空白画面

**症状**：
- VNC 5900 端口监听
- 截屏 48KB（基本纯黑）
- demo 日志只有"appsink initialized"无 Decode

**诊断**：
```bash
grep "Decode:" /tmp/demo_*.log | tail -3
dmesg 2>/dev/null | grep -c rk_vcodec
```

**可能原因**：

#### 1.1 dmesg 错误累积
**症状**：`dmesg | grep -c rk_vcodec > 1000`

**修复**：
```bash
./scripts/cleanup_dmesg.sh
./scripts/run_camera_demo.sh /dev/video0 2560 1024
```

#### 1.2 h265 文件 GOP 损坏
**症状**：dmesg 0 错误但仍 0 帧

**诊断**：
```bash
./scripts/test_h265_decode.sh recordings/video0_480.h265
# 看 IDR 帧长度是否 > 100KB
```

**修复**：
- 改用实时摄像头：`./scripts/run_camera_demo.sh /dev/video0`
- 或重新录制：`./scripts/record_h265.sh /dev/video0 2560 1024`

#### 1.3 RKNN 模型加载失败
**症状**：日志 `rknn_init failed` 或 `rknn_load_model failed`

**修复**：
```bash
# 确认模型是 RK3576 平台
cat models/rk3576/manifest.json | grep target_platform
# 应该是 "rk3576"

# 确认 librknnrt.so 在 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/third_party/rknn/aarch64:$LD_LIBRARY_PATH
```

### Q2: 帧率过低 (< 5 fps)

**诊断**：
```bash
grep "Qt:" /tmp/demo_*.log | tail -3
# 输出类似：infer=85ms draw=290ms
```

**优化**：
```bash
# 1. 增大跳帧间隔（减少推理频率）
./scripts/run_camera_demo.sh /dev/video0 2560 1024
# 脚本默认 detection-interval=5

# 2. 缩小 dashboard 尺寸
# 修改脚本中的 --dashboard-width 1280 --dashboard-height 720 为 640x360

# 3. 关闭 capture gate（减少分支）
# 编辑 configs/default.json: data_capture.enabled = false
```

### Q3: 双 demo 同时跑卡死

**症状**：
- 两个 demo 都在跑
- 都卡在 0 帧
- dmesg 错误暴增

**修复**：
```bash
# 只跑一个 demo
pkill -9 double-ok-demo
./scripts/cleanup_dmesg.sh
./scripts/run_camera_demo.sh /dev/video0 2560 1024
```

**根本原因**：RK3576 NPU 资源有限，并发 RKNN 推理 + mppvideodec 解码会争抢。

### Q4: 摄像头在 VNC 看到但 dashboard 显示"等待双手"

**原因**：摄像头角度问题，手在画面外。

**修复**：
- 让用户在画面下半部分做 OK 手势
- 调整摄像头物理位置（如果是生产部署）

### Q5: 编译错误

#### 5.1 找不到 Qt5
```bash
sudo apt install qtbase5-dev
```

#### 5.2 找不到 RKNN SDK
```bash
# 设置环境变量
export RKNN_SDK_ROOT=/path/to/rknn_sdk
./scripts/build_rk3576.sh
```

详细见 [build.md](build.md) 故障排除。

### Q6: 视频文件 NAL 结构异常

**症状**：
- 录制的 h265 文件 mppvideodec 解不出
- 截屏 48KB 空白

**诊断**：
```bash
./scripts/test_h265_decode.sh recordings/your.h265
```

**看 IDR 帧长度**：
- 正常：30-50 KB
- 异常：> 100 KB（说明 GOP 损坏）

**修复**：
```bash
# 1. 重新录制
./scripts/record_h265.sh /dev/video0 2560 1024 16 fresh.h265

# 2. 用实时摄像头代替
./scripts/run_camera_demo.sh /dev/video0 2560 1024

# 3. 拼接法修复 (高级)：
#    从 video2_480.h265 提取 VPS+SPS+PPS
#    拼接到 video0 IDR 之后
```

## RK3576 已知限制

### mppvideodec 文件兼容性问题

**症状**：v4l2src 录的 h265 文件 mppvideodec file 模式 0 帧

**根因**：
- HEVC 流 GOP 起始的 IDR 帧不完整
- mppvideodec 找不到参考帧

**影响范围**：所有用 v4l2src + filesink 录的 .h265 文件

**临时解决**：
- 实时摄像头（v4l2src 直连）能正常工作（24-36 fps）
- 录制文件后**用 mppvideodec 实时回放**（即不要先录再播）

### dmesg 错误累积

**症状**：dmesg 错误 > 1000 条时 mppvideodec 异常

**监控脚本**：
```bash
# 启动 demo 前清理
./scripts/cleanup_dmesg.sh

# 后台定期清理
nohup bash -c "while true; do sudo dmesg -c >/dev/null 2>&1; sleep 60; done" &
```

### 并发资源限制

**症状**：两个 demo 进程同时跑导致系统卡死

**限制**：
- RK3576 NPU 一次只能跑一个 RKNN 模型推理进程
- mppvideodec 一次只能跑一个解码 pipeline（实际）

**解决**：
- 单 demo 运行
- 用交替脚本切换

## 性能调优

### 帧率优化清单

| 操作 | 效果 | 风险 |
|------|------|------|
| `--detection-interval 10` | 显示 fps 提升 | 检测延迟 333ms |
| `--dashboard-width 640 --height 360` | 绘制快 4x | 画面模糊 |
| 关闭 capture gate (`data_capture.enabled=false`) | 减 5-10ms | 不存数据 |
| 用实时摄像头 (代替文件) | 帧率 24-36 → 7-30 | 需要摄像头 |

### 解码优化清单

| 操作 | 效果 | 风险 |
|------|------|------|
| 清理 dmesg (启动前) | mppvideodec 不卡 | 需 sudo |
| 用实时摄像头 | 稳定 7+ fps | 需要摄像头 |
| 单 demo 运行 | 系统不卡 | 只能看一个画面 |

## 监控指标

### 关键指标

| 指标 | 正常值 | 异常值 |
|------|--------|--------|
| dmesg rk_vcodec 错误 | < 100 | > 1000 |
| demo 启动 5s 内 Decode 日志 | ≥ 1 | 0 |
| demo 30s 累计 Decode 帧 | > 100 | < 10 |
| Qt 推理时间 | < 80ms | > 200ms |
| Qt 绘制时间 | < 150ms | > 500ms |
| VNC 截屏大小 | > 500KB | < 100KB |

### 监控脚本

```bash
# 综合健康度检查
./scripts/cleanup_dmesg.sh status && \
ps -p $(cat /tmp/demo_pid | cut -d= -f2) -o etime= 2>/dev/null && \
echo "Decode: $(grep -c 'Decode:' /tmp/demo_*.log | tail -1)" && \
echo "FPS: $(grep 'Qt:' /tmp/demo_*.log | tail -1 | grep -oE 'infer=[0-9.]+' || echo 'N/A')"
```

## 紧急恢复

如果系统完全卡死：

```bash
# 1. 强杀所有 demo
pkill -9 double-ok-demo

# 2. 清理 dmesg（需 sudo）
sudo dmesg -c

# 3. 重启 VNC（如需要）
vncserver :0

# 4. 重新启动
./scripts/cleanup_dmesg.sh
./scripts/run_camera_demo.sh /dev/video0 2560 1024

# 5. 如果还是卡，重启板子
sudo reboot
```

## 联系支持

如本手册未涵盖的问题：

1. 查看 `docs/rk3576_deployment.md` 部署说明
2. 查看 `docs/architecture.md` 架构说明
3. 查看 `docs/runtime_configuration.md` 配置说明
4. 检查 GitHub Issues

## 相关文档

- [testing.md](testing.md) - 测试脚本使用
- [build.md](build.md) - 编译详细说明
- [rk3576_deployment.md](rk3576_deployment.md) - 部署说明
