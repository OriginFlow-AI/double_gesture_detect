# YOLOv8n 手部 21 点候选权重（研究用途）

## 固定来源

- 上游仓库：https://github.com/NeuralVulture/yolo-hand-pose
- 固定提交：`c04c88847b0272d77cfa88fbbc14898f40bed1aa`
- 权重路径：`model/best.pt`
- 文件大小：`7,003,330` 字节
- SHA-256：`9a98cb80bcb56a565132308433015c346f1e676a4bed856180856f541a9ce090`
- 模型：YOLOv8n-Pose，单类 `hand`，`kpt_shape=[21,3]`

上游 `results.csv` 最后一轮（epoch 200）自报 Pose 指标为：precision
0.90371、recall 0.87923、mAP50 0.90043、mAP50-95 0.75514。这不是本工程的
独立复测结果；上游 README 同时声明权重只训练到其预期的 50%，因此只把它当作
RK3588 转换和精度 PoC 的起点。

## 安全检查

`.pt` 是 PyTorch ZIP/Pickle 容器，不应在主开发环境直接执行。当前只做了 ZIP
结构和 Pickle 指令静态反汇编；全局对象只见 Ultralytics、PyTorch、collections
等模型常用类型，未见 `os`、`subprocess`、`eval`、`exec` 等明显异常引用。
静态检查不能证明文件绝对安全，导出 ONNX 时仍应使用隔离容器或一次性虚拟机。

## 输出契约差异

第三个关键点分量是可见性/置信度，不是 `z`；模型也不输出 Left/Right
handedness、presence 或 world landmarks。当前 YOLO Pose provider 将 z 固定为 0、
handedness 固定为 `Unknown`。依赖 z 和左右手归一化的旧分类器不满足输入合同，
不能用于该模型；仍需采集目标域数据重新训练二维分类器。

## 许可边界

上游仓库外层标注 Apache-2.0，但 checkpoint 内嵌 Ultralytics AGPL-3.0 许可元数据；
训练数据另为 CC BY-NC-SA 4.0。仓库许可证不能覆盖或消除模型框架和数据集的独立
许可义务。该副本当前仅用于技术研究和非商业 PoC；闭源商用前必须完成许可审查，
并优先改用自采或明确授权的数据重新训练。
