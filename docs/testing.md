# 测试文档 (RK3576)

> **目标平台**：RK3576 板端
> **最后更新**：2026-07-28
> **相关脚本**：`scripts/` 目录下

## 快速开始

```bash
# 1. 编译（一次性）
./scripts/build_rk3576.sh

# 2. 测试实时摄像头
./scripts/run_camera_demo.sh /dev/video2 3840 1080

# 3. 测试 h265 文件
./scripts/run_file_demo.sh recordings/video2_480.h265 3840 1080

# 4. VNC 连接
vncviewer 127.0.0.1:5900
```

## 测试矩阵

| 模式 | 脚本 | 输入 | 用途 |
|------|------|------|------|
| 实时摄像头 | `run_camera_demo.sh` | `/dev/video*` v4l2 | 现场演示、生产部署 |
| 文件回放 | `run_file_demo.sh` | `*.h265` | 离线分析、视频回放 |
| 循环播放 | `run_loop_demo.sh` | `*.h265` | 长时间 VNC 演示 |
| 文件录制 | `record_h265.sh` | `/dev/video*` → `*.h265` | 采集测试数据 |
| 单元测试 | `test_h265_decode.sh` | `*.h265` | 文件健康度检查 |
| 资源清理 | `cleanup_dmesg.sh` | - | 解决 mppvideodec 卡死 |

## 输入模式说明

demo 支持两种输入模式，**根据 `--camera` 参数自动选择**：

| 参数形式 | 模式 | Pipeline |
|---------|------|----------|
| `/dev/video0` | 实时 (v4l2src) | v4l2src → h265parse → mppvideodec → ... |
| `recordings/*.h265` | 文件 (filesrc) | filesrc → h265parse → mppvideodec → ... |

### 模式选择建议

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 生产部署 | 实时 (v4l2src) | mppvideodec 实时模式稳定 |
| 视频回放分析 | 文件 (filesrc) | 可重复回放同一画面 |
| 长时间 VNC 演示 | 实时 + 循环脚本 | 不卡死 |
| 离线截图 | 文件 + `--max-frames` | 精确控制 |

## 性能指标

### 实时摄像头模式（推荐）

| 摄像头 | 分辨率 | 帧率 | 推理 | 说明 |
|--------|--------|------|------|------|
| `/dev/video0` (Originflow) | 2560x1024 → 1280x1024 | 7.3 fps | 20ms | 拼接摄像头，视野含办公桌+用户 |
| `/dev/video2` (YCTC) | 3840x1080 → 1920x1080 | 7.3 fps | 30ms | 横构图，适合 16:9 训练集 |

### 文件回放模式

| 文件 | 状态 | 帧率 |
|------|------|------|
| 干净文件 (VPS+SPS+PPS+IDR 完整) | ✅ 能解 | 2-3 fps |
| GOP 损坏 (IDR 170KB+ 异常大) | ❌ 0 帧 | 0 |
| 实时 v4l2 流直接录的 | ❌ 0 帧 | 0 |

## 测试场景

### 场景 1：板端实时摄像头测试

```bash
# 1. 编译
./scripts/build_rk3576.sh

# 2. 检查摄像头
ls -la /dev/video*
v4l2-ctl -d /dev/video0 --get-fmt-video

# 3. 清理内核日志
./scripts/cleanup_dmesg.sh

# 4. 跑实时摄像头
./scripts/run_camera_demo.sh /dev/video2 3840 1080

# 5. VNC 连接看画面
vncviewer 127.0.0.1:5900

# 6. 做 OK 手势测试
#    - 双手放在画面下半部分
#    - 看到 dashboard 显示 "双手 2/2 OK" + 21 关键点骨架

# 7. 停止
Ctrl+C
```

### 场景 2：录制 + 文件回放对比

```bash
# 1. 录制两个摄像头 16 秒视频
./scripts/record_h265.sh /dev/video0 2560 1024 16 video0_test.h265
./scripts/record_h265.sh /dev/video2 3840 1080 16 video2_test.h265

# 2. 验证文件健康度
./scripts/test_h265_decode.sh video0_test.h265
./scripts/test_h265_decode.sh video2_test.h265

# 3. 跑回放（实时摄像头更好，但可作为离线备份）
./scripts/run_file_demo.sh video2_test.h265 3840 1080

# 4. VNC 对比
vncviewer 127.0.0.1:5900
```

### 场景 3：长时间 VNC 演示（自动重启）

```bash
# 自动循环播放（每次跑完 480 帧 ≈ 16s 后 sleep 2s 重启）
./scripts/run_loop_demo.sh recordings/video0_480.h265 2560 1024

# VNC viewer 保持连接即可看到持续画面
# 停止：Ctrl+C
```

### 场景 4：单元测试

```bash
# 编译单元测试
cd build_rk3576
make double_ok_gesture_tests double_ok_config_io_tests \
     double_ok_model_replacement_tests double_ok_hevc_reader_tests -j4

# 跑所有测试
ctest --output-on-failure

# 跑单个测试
ctest -R hevc_reader --output-on-failure
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera` | (必填) | 输入源（/dev/videoX 或 .h265 路径） |
| `--width --height` | (必填) | 输入分辨率 |
| `--camera-fps` | 30 | 输入帧率 |
| `--fourcc` | HEVC | 编码格式 (HEVC/MJPG/H264) |
| `--landmark-backend` | rknn | 推理后端 (rknn/onnx) |
| `--palm-model --hand-model` | (自动) | RKNN 模型路径 |
| `--detection-interval` | 5 | 每 N 帧做一次推理（1=每帧） |
| `--max-frames` | 0 (无限) | 限制单次运行最大帧数 |
| `--loop-file` | false | 文件 EOS 时自动 seek 重置循环（**不推荐**，mppvideodec 不支持） |
| `--dashboard-width --height` | 1280x720 | VNC 画布尺寸 |
| `--log-level` | INFO | DEBUG/INFO/WARN/ERROR |

## 已知问题与缓解

### 问题 1：dmesg 错误累积导致 mppvideodec 卡死

**症状**：
- 启动 demo 后 Decode 日志为 0
- dmesg 持续输出 `rk_vcodec: mpp_task_dump_hw_reg`
- 截屏显示空白 VNC

**根因**：
- mppvideodec 处理 H265 流时若产生异常（NAL 错误、参考帧丢失）
- 持续刷错误日志污染 dmesg ring buffer
- 累积到几千条后 mppvideodec 内核模块进入异常状态

**缓解**：
```bash
# 启动 demo 前清理
./scripts/cleanup_dmesg.sh

# 或定期清理（dmesg > 1000 时）
while true; do
    ./scripts/cleanup_dmesg.sh
    sleep 60
done
```

### 问题 2：v4l2src 录的 h265 文件不能用 mppvideodec 解码

**症状**：
- 录制文件后用 demo 跑不出帧（Decode 日志为 0）
- gst-launch 1.5s 完成但实际 0 帧

**根因**：
- v4l2src + h265parse config-interval=-1 + filesink 录的 HEVC 流
- IDR 帧长度异常大（170KB+）
- mppvideodec 找不到参考帧

**缓解**：
- **不要依赖文件回放**，用实时摄像头
- 或用 v4l2-ctl --stream-mmap 直接抓帧（不经过 gst-launch）

### 问题 3：两个 demo 同时跑导致资源争抢

**症状**：
- 两个 demo 都卡死
- dmesg 错误暴增
- 系统响应变慢

**根因**：
- RK3576 NPU 资源有限
- 两个 RKNN 推理进程 + 两个 mppvideodec 解码争抢

**缓解**：
- **只跑一个 demo 进程**
- 用交替脚本切换（每次只跑一个）

## 相关文档

- [build.md](build.md) - 编译详细说明
- [troubleshooting.md](troubleshooting.md) - 故障排除
- [rk3576_deployment.md](rk3576_deployment.md) - 部署说明
- [runtime_configuration.md](runtime_configuration.md) - 配置说明
