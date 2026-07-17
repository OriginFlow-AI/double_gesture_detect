# 模型

RK3588 NPU 使用 `rk3588/` 中的两个 FP16 RKNN 文件；桌面对照路径使用
`opencv_zoo/` 中的 FP32 ONNX 文件。两条路径共享同一套掌心解码、旋转 ROI、
21 点坐标还原和手势判断。

模型来源、许可、转换合同和 SHA-256 见两个子目录的 README 与 manifest。
