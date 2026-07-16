# Current Main Contract

当前改造分支以提交 `261902a4`（0612）为 GUI、CLI 和门控行为基准。详细双模型合同、
运行命令与验收缺口见 [model_replacement_0612.md](model_replacement_0612.md)。

## 固定用户效果

- Qt 深色 dashboard 的布局、颜色、中文文字、画面区、五项门控和系统区保持 0612；
- Q/Esc 退出、S 截图、全屏、窗口尺寸和相机参数含义不变；
- 仍使用奥比 Gemini 335 的 V4L2 BGR 帧；
- 原有稳定窗口、capture gate、capture writer 和阻断原因不改；
- 顶部延续 0612 chip 样式，并按新模型合同区分 inference 与 total；
- JSON 功能测试明确显示“测试后端 / 非 YOLOv8 推理”。

## 生产模型合同

```text
BGR frame
-> YOLOv8-Pose RKNN: hand box + 21 x/y/visibility, max 2
-> 63-value x/y/visibility features
-> dual-output Left/Right + OK classifier
-> exactly 2 hands, trusted Left + Right, both OK
-> 0612 temporal stability and capture gate
```

关键点第三维不是 z。YOLO 不提供 handedness；禁止按画面 x 位置伪造左右手。
第二模型置信度不足返回 Unknown。

## 后端规则

- `yolov8-rknn`：唯一生产路径；`rknn` 只是同路径兼容别名；
- `landmarks-json`、`mediapipe`、`opencv-heuristic`、`none`：必须显式选择的测试或
  调试路径；
- 生产模型、manifest、属性模型、AArch64 或 RKNN Runtime 任一不满足即初始化失败；
- 禁止静默回退。

## 当前阻塞

仓库存在模型一，但缺已训练模型二、目标域评估集和 RK3588 真板环境。此状态只完成
可由当前资产确定的接入合同和测试，未完成精度、性能、长稳和生产部署验收。
