# Project Structure

当前工程是 C++/CMake 项目。

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
runtime.hpp/cpp        FPS、处理延迟
training.hpp/cpp       特征 CSV、分层切分、逻辑回归、指标
model_io.hpp/cpp       C++ 文本模型保存和加载
json.hpp/cpp           小型 JSON 解析器，用于 HaGRID 转换
```

命令入口：

```text
double-ok-demo
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
