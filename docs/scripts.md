# 脚本说明 (RK3576)

> **最后更新**：2026-07-28
> **位置**：`scripts/`

## 脚本总览

| 脚本 | 功能 | 用法 |
|------|------|------|
| [build_rk3576.sh](#build_rk3576sh) | 编译 demo + 单元测试 | `./scripts/build_rk3576.sh [Debug\|clean]` |
| [run_camera_demo.sh](#run_camera_demosh) | 跑实时摄像头 demo | `./scripts/run_camera_demo.sh [device] [w] [h]` |
| [run_file_demo.sh](#run_file_demosh) | 跑 h265 文件 demo | `./scripts/run_file_demo.sh [file] [w] [h]` |
| [run_loop_demo.sh](#run_loop_demosh) | 循环播放 h265 文件 | `./scripts/run_loop_demo.sh [file] [w] [h]` |
| [record_h265.sh](#record_h265sh) | 录制 HEVC 视频 | `./scripts/record_h265.sh [device] [w] [h] [duration] [out]` |
| [test_h265_decode.sh](#test_h265_decodesh) | 测试 h265 文件健康度 | `./scripts/test_h265_decode.sh <file>` |
| [cleanup_dmesg.sh](#cleanup_dmesgsh) | 清理 dmesg 错误 | `./scripts/cleanup_dmesg.sh [status]` |

## 详细说明

### build_rk3576.sh

编译 RK3576 板端 demo。

**用法**：
```bash
./scripts/build_rk3576.sh            # 默认 Release
./scripts/build_rk3576.sh Debug      # Debug 模式
./scripts/build_rk3576.sh clean      # 清理 build_rk3576/
```

**环境变量**：
- `RKNN_SDK_ROOT`：RKNN SDK 路径（默认 `third_party/rknn`）

**输出**：
- `build_rk3576/double-ok-demo-hevc`：主 demo 可执行
- `build_rk3576/double-ok-camera-check`：摄像头诊断工具
- `build_rk3576/double_ok_*_tests`：单元测试

### run_camera_demo.sh

跑实时摄像头 demo（v4l2 模式，最稳定）。

**用法**：
```bash
./scripts/run_camera_demo.sh                          # 默认 /dev/video0, 2560x1024
./scripts/run_camera_demo.sh /dev/video2 3840 1080  # 自定义设备 + 分辨率
```

**环境变量**：
- `VNC_PORT`：VNC 端口（默认 5900）
- `DETECTION_INTERVAL`：跳帧检测间隔（默认 5）

**前置条件**：
- 设备存在（如 `/dev/video0`）
- 已编译

### run_file_demo.sh

跑 h265 文件 demo（filesrc 模式）。

**用法**：
```bash
./scripts/run_file_demo.sh recordings/video0_480.h265 2560 1024
./scripts/run_file_demo.sh recordings/video2_480.h265 3840 1080
```

**环境变量**：
- `VNC_PORT`：VNC 端口（默认 5900）
- `DETECTION_INTERVAL`：跳帧检测间隔（默认 5）
- `MAX_FRAMES`：单次运行最大帧数（默认 480 = 16s@30fps）

**注意**：
- 录制的 h265 文件 mppvideodec file 模式可能 0 帧
- 优先用实时摄像头 (`run_camera_demo.sh`)

### run_loop_demo.sh

循环播放 h265 文件（自动重启）。

**用法**：
```bash
./scripts/run_loop_demo.sh recordings/video0_480.h265 2560 1024
```

**行为**：
- 每次跑完（EOS 或 max-frames）后 sleep 后自动重启
- 适用：长时间 VNC 演示

**环境变量**：
- `VNC_PORT`：VNC 端口（默认 5900）
- `DETECTION_INTERVAL`：跳帧检测间隔（默认 5）
- `MAX_FRAMES`：单次运行最大帧数（默认 480）
- `PAUSE_BETWEEN`：两次启动间隔秒数（默认 2）

**停止**：Ctrl+C，自动清理 demo 进程

### record_h265.sh

用 gst-launch 录制 HEVC 视频。

**用法**：
```bash
./scripts/record_h265.sh                                      # 默认 /dev/video0, 16s
./scripts/record_h265.sh /dev/video2 3840 1080                # 自定义设备
./scripts/record_h265.sh /dev/video0 2560 1024 30 my.h265     # 自定义时长 + 文件名
```

**参数**：
- `$1`：v4l2 设备路径（默认 `/dev/video0`）
- `$2`：width
- `$3`：height
- `$4`：录制时长秒（默认 16）
- `$5`：输出文件名（默认 `recordings/<device>_<timestamp>.h265`）

**注意**：
- 录制文件 mppvideodec file 模式可能 0 帧
- 用 `test_h265_decode.sh` 验证文件健康度

### test_h265_decode.sh

测试 h265 文件能否被 mppvideodec 解码。

**用法**：
```bash
./scripts/test_h265_decode.sh recordings/video0_480.h265
```

**输出**：
1. 文件 NAL 结构分析（VPS/SPS/PPS/IDR/P帧 数量）
2. IDR 帧位置和长度（> 100KB 异常）
3. gst-launch 解码测试（elapsed 时间）
4. dmesg 错误统计
5. 诊断建议

**关键判断**：
- IDR 长度 > 100KB → GOP 损坏
- 0 帧输出 + 多错误 → dmesg 累积（清理）

### cleanup_dmesg.sh

清理 dmesg 内核日志。

**用法**：
```bash
./scripts/cleanup_dmesg.sh         # 清理 dmesg
./scripts/cleanup_dmesg.sh status  # 查看错误数
```

**背景**：
- mppvideodec 处理异常 HEVC 流时刷错误日志
- 累积到几千条时 mppvideodec 内核模块异常
- 启动 demo 前清理可避免此问题

**输出**：
- 清理前/后错误数
- 错误过多时建议重启板子

## 完整工作流

### 1. 一次性环境准备

```bash
# 1.1 编译
./scripts/build_rk3576.sh

# 1.2 验证编译结果
./build_rk3576/double-ok-demo-hevc --help 2>&1 | head -5

# 1.3 清理 dmesg
./scripts/cleanup_dmesg.sh
```

### 2. 快速测试（实时摄像头）

```bash
./scripts/run_camera_demo.sh /dev/video0 2560 1024
# VNC: vncviewer 127.0.0.1:5900
```

### 3. 离线分析（h265 文件）

```bash
# 3.1 录制
./scripts/record_h265.sh /dev/video0 2560 1024 16 test.h265

# 3.2 验证文件
./scripts/test_h265_decode.sh test.h265

# 3.3 跑回放（如文件健康）
./scripts/run_file_demo.sh test.h265 2560 1024
```

### 4. 长时间 VNC 演示

```bash
./scripts/run_loop_demo.sh recordings/video2_480.h265 3840 1080
# VNC viewer 保持连接即可看到持续画面
# Ctrl+C 停止
```

### 5. 单元测试

```bash
cd build_rk3576
ctest --output-on-failure
```

## 环境变量总结

| 变量 | 用途 | 默认 |
|------|------|------|
| `RKNN_SDK_ROOT` | RKNN SDK 路径 | `third_party/rknn` |
| `VNC_PORT` | demo VNC 端口 | 5900 |
| `DETECTION_INTERVAL` | 跳帧检测间隔 | 5 |
| `MAX_FRAMES` | 单次运行最大帧 | 480 |
| `PAUSE_BETWEEN` | loop 重启间隔 | 2 |

## 相关文档

- [testing.md](testing.md) - 测试场景说明
- [build.md](build.md) - 编译详细
- [troubleshooting.md](troubleshooting.md) - 故障排除
