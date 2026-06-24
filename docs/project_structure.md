# Project Structure

当前工程是 C++/CMake 项目。

目标分层和演进原则见 [architecture.md](architecture.md)。本页描述当前落地文件结构。

```text
.
├── CMakeLists.txt
├── include/double_ok_gesture/   # 公共 C++ 头文件
├── src/                         # 核心库实现
├── apps/                        # 命令行入口
├── tests/cpp/                   # C++ 单元测试
├── configs/                     # 运行配置与标定配置
├── data/                        # 原始与处理后数据
├── models/                      # 本地模型产物
├── reports/                     # 报告和截图
├── scripts/                     # 构建与运行脚本
└── docs/                        # 工程说明
```

核心模块：

```text
features.hpp/cpp       21 点手部特征、几何 OK 分数
recognizer.hpp/cpp     单手 OK、双手 OK、稳定窗口
capture_gate.hpp/cpp   姿态、入框、居中、间距、手势门控
config.hpp/cpp         当前 JSON 配置的 C++ 读取与校验
camera.hpp/cpp         OpenCV 摄像头打开、重试、设备枚举
landmark_provider.hpp/cpp hand landmark provider 抽象与当前调试 provider
runtime.hpp/cpp        FPS、处理延迟
runtime_pipeline.hpp/cpp 单帧运行流水线、后端选择、运行时装配
capture_writer.hpp/cpp ready 后保存原始帧和 metadata
demo_app.hpp/cpp      double-ok-demo 的 CLI、headless smoke 和运行选项映射
qt_dashboard.hpp/cpp Qt 实时 dashboard 控件、状态刷新、截图和事件日志
report.hpp/cpp        静态 GUI 报告数据扫描与 HTML 渲染
training.hpp/cpp       特征 CSV、分层切分、逻辑回归、指标
model_io.hpp/cpp       C++ 文本模型保存和加载
json.hpp/cpp           小型 JSON 解析器，用于 HaGRID 转换
```

`src/demo_app.cpp` 编成非 Qt 的 `double_ok_gesture_demo_app` 小库，供 demo 和测试复用。`src/qt_dashboard.cpp` 只编进 `double-ok-demo`，不进入 `double_ok_gesture_core`，保证核心库继续不依赖 Qt。
`double-ok-headless` 是无 Qt 运行入口，供板端和最小运行环境使用；桌面 `double-ok-demo` 默认保留 Qt dashboard。

RV1126 相关：

```text
cmake/toolchains/rv1126-linux-gnueabihf.cmake
configs/rv1126_uvc.json
scripts/build_rv1126.sh
scripts/convert_hand_landmark_to_rknn.sh
scripts/package_rv1126.sh
scripts/run_rv1126.sh
docs/rv1126_deployment.md
docs/current_main_contract.md
```

命令入口：

```text
double-ok-demo
double-ok-headless
double-ok-camera-check
double-ok-capture
double-ok-prepare-hagrid
double-ok-train
double-ok-eval
double-ok-gui
```

验证：

```bash
scripts/test.sh
```
