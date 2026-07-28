# 编译文档 (RK3576)

> **目标平台**：RK3576 板端
> **最后更新**：2026-07-28
> **编译脚本**：`scripts/build_rk3576.sh`

## 前置条件

### 1. 系统依赖 (apt)

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake pkg-config \
    qtbase5-dev libqt5gui5 libqt5widgets5 \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libgl1-mesa-dev libegl1-mesa-dev \
    librga-dev  # Rockchip RGA 加速
```

### 2. RKNN SDK

RKNN Toolkit2 NPU 运行时（必须 aarch64 Linux 版本）：

```bash
# 推荐：vendored SDK（已包含在项目中）
ls third_party/rknn/
# ├── aarch64/librknnrt.so
# ├── include/rknn_api.h
# └── ...

# 备选：官方 SDK
# 从 https://github.com/airockchip/rknn-toolkit2/releases 下载
# 解压到 /opt/rknn_sdk/
# export RKNN_SDK_ROOT=/opt/rknn_sdk

# 验证
ls -la third_party/rknn/include/rknn_api.h
ls -la third_party/rknn/aarch64/librknnrt.so
```

### 3. 摄像头设备

```bash
# 检查 USB 摄像头
ls -la /dev/video*
lsusb

# 验证 HEVC 摄像头能力
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## 编译步骤

### 快速编译

```bash
# 默认 Release 编译
./scripts/build_rk3576.sh

# Debug 编译（带调试符号）
./scripts/build_rk3576.sh Debug

# 清理构建
./scripts/build_rk3576.sh clean
```

### 手动编译（详细控制）

```bash
# 1. 创建 build 目录
mkdir -p build_rk3576
cd build_rk3576

# 2. CMake 配置
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DDOUBLE_OK_BUILD_QT_DEMO=ON \
    -DDOUBLE_OK_BUILD_CAMERA_CHECK=ON \
    -DRKNN_SDK_ROOT=$(pwd)/../third_party/rknn

# 3. 编译 demo 主目标
make double-ok-demo-hevc -j$(nproc)

# 4. 编译单元测试
make double_ok_gesture_tests \
     double_ok_config_io_tests \
     double_ok_model_replacement_tests \
     double_ok_hevc_reader_tests -j$(nproc)

# 5. 跑测试
ctest --output-on-failure
```

## 编译选项 (CMake)

| 选项 | 默认 | 说明 |
|------|------|------|
| `CMAKE_BUILD_TYPE` | (空) | Release/Debug/RelWithDebInfo |
| `BUILD_TESTING` | OFF | 启用 CTest 单元测试 |
| `DOUBLE_OK_BUILD_QT_DEMO` | ON | 构建 Qt GUI demo (double-ok-demo-hevc) |
| `DOUBLE_OK_BUILD_CAMERA_CHECK` | ON | 构建摄像头诊断工具 (double-ok-camera-check) |
| `DOUBLE_OK_ENABLE_RKNN` | ON | 启用 RKNN NPU 后端 |
| `DOUBLE_OK_REQUIRE_RKNN` | OFF | 找不到 RKNN SDK 时报错退出 |
| `RKNN_SDK_ROOT` | - | RKNN SDK 路径 |
| `DOUBLE_OK_ENABLE_MPP` | ON | 启用 MPP 硬件解码 (需要 RK3576 板端) |
| `MPP_SYSROOT` | - | MPP 交叉编译 sysroot 路径 |

## 编译产物

编译成功后 `build_rk3576/` 目录下：

```
build_rk3576/
├── double-ok-demo-hevc              # 主 demo 可执行文件
├── double-ok-camera-check           # 摄像头诊断工具
├── libdouble_ok_gesture_core.a      # 核心库
├── libdouble_ok_gesture_demo_app.a  # demo 应用库
├── double_ok_gesture_tests          # 单元测试
├── double_ok_config_io_tests        # 配置 IO 测试
├── double_ok_model_replacement_tests # 模型替换测试
└── double_ok_hevc_reader_tests      # HEVC 读取器测试
```

## 验证编译

```bash
# 1. 检查可执行文件
file build_rk3576/double-ok-demo-hevc
# 输出: ELF 64-bit LSB pie executable, ARM aarch64, ...

# 2. 验证 RKNN 库链接
ldd build_rk3576/double-ok-demo-hevc | grep -E "rknn|opencv|Qt5"
# 应包含: librknnrt.so, libopencv_*, libQt5*

# 3. 跑单元测试
cd build_rk3576
ctest --output-on-failure
```

## 交叉编译（如在 x86 主机上构建）

```bash
# 安装交叉编译工具链
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 配置交叉编译
mkdir -p build_aarch64
cd build_aarch64

cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_FIND_ROOT_PATH=/opt/rk3576-sysroot \
    -DRKNN_SDK_ROOT=/opt/rk3576-sysroot/usr \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

## 故障排除

### Q: cmake 找不到 Qt5

```
By not providing "FindQt5.cmake" in CMAKE_MODULE_PATH...
```

**解决**：
```bash
sudo apt install qtbase5-dev
```

### Q: 找不到 RKNN SDK

```
Could not find RKNN runtime library
```

**解决**：
```bash
# 验证 SDK 存在
ls third_party/rknn/include/rknn_api.h
ls third_party/rknn/aarch64/librknnrt.so

# 或指定 SDK 路径
export RKNN_SDK_ROOT=/path/to/rknn_sdk
./scripts/build_rk3576.sh
```

### Q: 链接错误 undefined reference to cv::*

**解决**：
```bash
sudo apt install libopencv-dev
# 或指定 OpenCV 路径
cmake .. -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4
```

### Q: MPP 头文件未找到

```
fatal error: rockchip/rk_mpi.h: No such file or directory
```

**解决**：
```bash
# 安装 librga + librockchip-mpp-dev
sudo apt install librga-dev librockchip-mpp-dev
```

## 编译后部署

```bash
# 打包编译产物
tar czf double-ok-rk3576.tar.gz \
    build_rk3576/double-ok-demo-hevc \
    build_rk3576/double-ok-camera-check \
    build_rk3576/*.a \
    models/rk3576/*.rknn \
    configs/default.json \
    scripts/

# 传输到板端
scp double-ok-rk3576.tar.gz rpdzkj@10.42.0.132:/data1/project/double_gesture_detect/

# 在板端解压并运行
ssh rpdzkj@10.42.0.132
cd /data1/project/double_gesture_detect
tar xzf double-ok-rk3576.tar.gz
./scripts/run_camera_demo.sh /dev/video0 2560 1024
```
