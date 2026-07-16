# Recognition Accuracy and Diagnostics

## 当前桌面链路

```text
Gemini 335 BGR
-> YOLOv8-Pose ONNX（框、21 点、visibility）
-> 可选属性模型二；缺失时使用实验几何 scorer
-> 单手 OK 阈值
-> 5 帧窗口内至少 3 帧双手 OK
-> 位置/完整性/间距/姿态采集门控
```

Pose 检测分、Left/Right 置信度和 OK 分是三种不同量。无属性模型时手别保持
`Unknown`，界面显示的 OK 值只是几何匹配分，不是经过目标域校准的概率。

## 已修复的误差来源

- 1280×720 的 x/y 曾分别除以宽高后直接计算距离与角度，导致非等比失真。现在 UI
  继续使用 `[0,1]` 点，分类独立使用原图像素等距点。
- 旧公式把条件相加，拇食指一旦重合，即使中/无名/小指全部弯曲也至少得到 0.70，
  高于默认阈值 0.68。现在 pinch、食指弯曲与其余三指展开是联合条件；张掌即使
  因关键点抖动出现指尖接近，也会被食指伸直证据否决。
- 指尖遮挡会使 Pose 输出出现合理抖动。新 scorer 对这一误差连续容忍，但三指未
  展开时仍被压到阈值以下。
- 退化的 21 个重合点不再被当作高分 OK。
- 低 visibility 手势点会保留检测框、但安全判为非 OK；低质量候选也会在 NMS/
  `max_hands` 前移除。

## 现场运行与诊断

当前 ONNX 权重在现场会把开放掌与真 OK 输出成近似关键点，因此只建议用于 Pose
链路诊断。桌面交互优先显式选择本机已安装的 MediaPipe Full：

```bash
scripts/check_camera.sh /dev/video6
DOUBLE_OK_MEDIAPIPE_PYTHON=/path/to/mediapipe/venv/bin/python \
scripts/run_demo.sh /dev/video6 \
  --landmark-backend mediapipe \
  --disable-auto-capture \
  --log-level INFO
```

本机现场已验证 MediaPipe 约 29–30 FPS，开放掌连续帧保持非 OK；它仍是显式桌面
调试后端，不替代 RK3588 的生产双模型合同。

ONNX Headless 可逐帧查看 Pose 分、OK 分和拇/食指尖 visibility：

```bash
build/double-ok-headless \
  --camera /dev/video6 \
  --config configs/default.json \
  --capture-gate \
  --landmark-backend yolov8-onnx \
  --pose-model models/hand_pose.onnx \
  --disable-auto-capture \
  --status-interval 0 \
  --max-frames 100 \
  --log-level INFO
```

如有模型二，在以上命令增加：

```bash
--model /path/to/trained_hand_attribute_v1.json
```

## 数据与验收边界

自动采集的 `double_ok=true` 是运行时预测，不是人工标签。新 metadata 会写入
`schema=double_ok_capture_v2`、`label_source=runtime_prediction`、逐手 21 点、
visibility、框和分数，以便之后人工复核和离线回放；不得直接把自预测标签作为训练
真值。

生产验收至少要按人员和采集会话隔离数据，覆盖标准 OK、张掌、拳头、V、点赞、
遮挡、裁边和快速运动，分别统计 Pose 检出率、单手 OK precision/recall、稳定双手
事件召回和持续负样本误触发。阈值只在验证集选择，测试集不得再次调参。

当前仓库仍缺 Gemini 335 人工标注目标域集和已训练属性模型二。因此本轮修复能解决
明确的坐标、候选筛选、规则逻辑和展示缺陷，但当前 `hand_pose.onnx` 本身不能在
关键点空间可靠分离真 OK/张掌。生产精度必须换或微调 Pose，或训练模型二，不能再
通过调宽 pinch 阈值宣称解决。
