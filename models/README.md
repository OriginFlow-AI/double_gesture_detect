# 模型资产状态

## x86_64 桌面端：YOLOv8-Pose ONNX

- 本机文件：`hand_pose.onnx`（由 `.gitignore` 排除，不随源码提交）
- 文件大小：`16,610,827` 字节
- SHA-256：`5dadc44325569fb7c1901b9fd8eaee2275d1636e51f75ce4f22f1e2ada7122fc`
- 输出：单个 FP32 `[1,68,8400]` 张量，即 4 个 box 值、1 个 hand 置信度和
  21 组 `(x,y,visibility)`
- 上游仓库：https://github.com/NeuralVulture/yolo-hand-pose
- 固定提交：`c04c88847b0272d77cfa88fbbc14898f40bed1aa`
- 原始权重：`model/best.pt`，SHA-256
  `9a98cb80bcb56a565132308433015c346f1e676a4bed856180856f541a9ce090`

该 ONNX 由 Ultralytics 8.2.15 以 640×640、batch 1、opset 12、静态尺寸、无
end-to-end NMS 导出。为兼容本机 OpenCV 4.6 DNN，广播常量已显式展开；ONNX
Runtime 对比原始简化图的输出完全一致，OpenCV 4.6 与 ONNX Runtime 的最大绝对
误差为 `2.136e-4`。当前只作为研究和非商业 PoC 使用；闭源或商业使用前请审查
checkpoint 内嵌的 Ultralytics AGPL-3.0 元数据以及训练数据的 CC BY-NC-SA 4.0
许可。

## 模型一：YOLOv8-Pose 手部 21 点

- RKNN：`rk3588/hand_pose_640_fp.rknn`
- manifest：`rk3588/hand_pose_640_fp.rknn.manifest.json`
- SHA-256：`835be29087828e7992b7ae8c5b3fc9410b4c5f11afd31d9ff062aabf8f8ff25f`
- 目标：RK3588，FP，640×640，单类 `hand`
- 输出：三层 65 通道检测特征图和一个 `[1,21,3,8400]` 关键点张量

关键点第三维是 `visibility/confidence`，不是 z 深度。真实 RKNN tensor
的 type/layout/量化参数仍须在 RK3588 上由 Runtime query 核验；manifest 不能替代板端查询。

来源、固定提交和许可风险见 `rk3588/SOURCE_AND_LICENSE.md`。

## 模型二：手部属性双输出分类器

当前仓库没有已训练的模型二，也没有足够的奥比 Gemini 335 目标域标注数据。
生产 `yolov8-rknn` 后端因此会明确初始化失败，不会回退到几何规则、MediaPipe、
OpenCV heuristic 或 JSON fixture。

运行时已固定 `double_ok_hand_attribute_v1` 合同：输入 21 组归一化
`x,y,visibility`，共 63 个值；两个线性输出头分别给出 Left/Right 概率和
OK/Not-OK 概率。低于 handedness 阈值时返回 `Unknown`。该 JSON 合同只是模型
序列化格式，不代表仓库内已有可用权重。

历史 `ok_hand_numpy_logreg.txt/.pkl` 使用的特征合同与 YOLOv8 二维关键点不一致，
不得作为生产模型二使用。其训练、报告入口仅为 0612 历史兼容工具。
