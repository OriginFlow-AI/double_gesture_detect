# RK3588 NPU 双手 OK 手势识别部署版

版本：`0.3.0-rk3588-npu`

## Git 交付记录

- 仓库：`git@github.com:wangjh2024/00_double_gesture_detect.git`
- HTTPS：`https://github.com/wangjh2024/00_double_gesture_detect.git`
- 交付分支：`model-replacement-0612`
- 已发布部署提交：`ea6e499e8088c42ae64a0942642352f80018acec`
- 提交名称：`版本名称：RK3588 NPU 双手 OK 手势识别部署版`

同事拉取命令：

```bash
git clone -b model-replacement-0612 --single-branch \
  git@github.com:wangjh2024/00_double_gesture_detect.git
cd 00_double_gesture_detect
git log -1 --oneline
cat VERSION
```

没有配置 GitHub SSH 时，将地址替换为上面的 HTTPS 地址。

## 目标环境

- RK3588 / RK3588S，AArch64 Linux
- 与 RKNN Runtime 2.3.2 兼容的 RKNPU2 内核驱动
- CMake 3.20+、C++20 编译器、OpenCV 4、Qt5 Widgets
- V4L2 彩色相机，默认设备 `/dev/video6`

Ubuntu/Debian 系统可安装构建依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev qtbase5-dev
```

## 部署运行

```bash
tar -xzf double-ok-rk3588-0.3.0-rk3588-npu.tar.gz
cd double-ok-rk3588-0.3.0-rk3588-npu
sha256sum -c models/rk3588/SHA256SUMS
(cd third_party/rknn && sha256sum -c SHA256SUMS)
scripts/verify_rk3588.sh /dev/video6
scripts/run_demo.sh
```

相机不是 `/dev/video6` 时，只传入设备节点：

```bash
scripts/run_demo.sh /dev/video0
```

首次上板应先运行 `scripts/verify_rk3588.sh /dev/videoN`。预检会确认系统架构、相机
权限、模型和 RKNN Runtime 校验和，并以 `DOUBLE_OK_REQUIRE_RKNN=ON` 强制编译 NPU
版本；任何 CPU 回退或动态库缺失都会直接失败。

脚本自动配置、编译并启动 Qt 界面。日志必须显示：

```text
检测到 aarch64，使用 RK3588 NPU。
runtime initialized: backend=rknn
```

按 `Q` 或 `Esc` 退出，按 `S` 保存界面截图。

## RKNN Runtime

部署包自带 Rockchip RKNN Runtime 2.3.2 的最小 AArch64 SDK，包含头文件、动态库和
Apache-2.0 许可证。板端 BSP 的 NPU 驱动仍须兼容该版本。如果设备厂商要求使用其
BSP 配套 Runtime，请设置后再启动：

```bash
export RKNN_SDK_ROOT=/path/to/vendor/rknpu2
scripts/run_demo.sh
```

模型是面向 RK3588 转换的非量化 FP16 双模型，不需要在板端安装 Python、ONNX
Runtime 或 RKNN Toolkit2。
