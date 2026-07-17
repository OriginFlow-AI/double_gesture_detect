# 模型

默认运行时使用 `opencv_zoo/` 中的两个 FP32 ONNX 文件组成一条流水线：
先检测掌心，再在旋转后的手部区域估计 21 个关键点。

模型来源、许可、兼容处理和 SHA-256 见 `opencv_zoo/README.md`。
