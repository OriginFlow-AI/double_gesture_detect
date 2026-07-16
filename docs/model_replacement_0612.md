# 0612 GUI 保持与双模型替换说明

## 1. 范围和当前结论

用户可见基准是提交 `261902a4d6e76360e69a6f1851b4afe77ea18dda`。
本次只替换关键点 provider、手属性分类器及其必要的数据合同；Qt 窗口、深色布局、
门控顺序、采集流程、Q/Esc/S、全屏和尺寸参数继续沿用 0612。

当前仓库具备模型一及 YOLOv8 后处理代码，但不具备已训练的模型二，也没有可用于
报告准确率、召回率、误触发率和稳定性的 Gemini 335 目标域正负样本集。因此当前
状态是“接口和静态合同已接入、等待模型二和 RK3588 真板验收”，不能称为生产可用
或精度达标。

## 2. 模型资产

### 2.1 YOLOv8-Pose

| 项目 | 当前值 |
|---|---|
| 类型 | YOLOv8n-Pose，单类 `hand`，21 点 |
| RKNN 路径 | `models/rk3588/hand_pose_640_fp.rknn` |
| manifest | `models/rk3588/hand_pose_640_fp.rknn.manifest.json` |
| 文件大小 | 8,710,115 bytes |
| SHA-256 | `835be29087828e7992b7ae8c5b3fc9410b4c5f11afd31d9ff062aabf8f8ff25f` |
| 目标 | RK3588 / FP / RKNN Toolkit2 2.3.2 |
| 主机输入 | RGB uint8 NHWC，640×640，letterbox=114 |
| 归一化 | 模型转换合同 mean=`[0,0,0]`，std=`[255,255,255]` |
| 静态输出 | `[1,65,80,80]`、`[1,65,40,40]`、`[1,65,20,20]`、`[1,21,3,8400]` |

65 个检测通道由 64 个 DFL 框回归通道和 1 个 hand 类别分数组成。21 点顺序为
`wrist`，随后依次是 thumb、index、middle、ring、pinky 各自的四个关节。
第三维严格定义为 `visibility/confidence`，运行时把兼容结构中的 z 固定为 0。
YOLOv8-Pose 本身不输出可信的 Left/Right。

manifest 是静态转换记录。provider 初始化时还会调用
`RKNN_QUERY_IN_OUT_NUM`、`RKNN_QUERY_INPUT_ATTR`、`RKNN_QUERY_OUTPUT_ATTR`，
要求 1 输入/4 输出，核对 shape/layout/type，并记录每个 tensor 的
`qnt_type/zp/scale`。本机没有 RK3588，故这些真实板端值目前没有验收结果，manifest
中的 `board_validation.completed` 保持 `false`。

### 2.2 手部属性分类模型

当前缺失实际权重文件。已实现的目标 artifact schema 为
`double_ok_hand_attribute_v1`：

```json
{
  "schema": "double_ok_hand_attribute_v1",
  "feature_count": 63,
  "mean": ["63 finite values"],
  "scale": ["63 finite non-zero values"],
  "handedness_coef": ["63 finite values"],
  "handedness_intercept": 0.0,
  "ok_coef": ["63 finite values"],
  "ok_intercept": 0.0
}
```

输入是以 wrist 为原点、以最远关键点距离为尺度的 21 组
`normalized_x, normalized_y, visibility`，共 63 个 float 特征，不读取 z。
`input_mirrored=true` 时在分类前反转归一化 x；离线镜像增强必须同时交换 Left/Right
标签。handedness 头输出 `P(Right)`，置信度不足时为 `Unknown`；OK 头输出
`P(OK)`。默认阈值分别为 0.65 和 0.68。

## 3. 真实运行流水线

```text
Orbbec BGR frame
  -> validate frame
  -> letterbox + BGR to RGB
  -> YOLOv8-Pose RKNN
  -> DFL decode + keypoint decode + NMS (max 2)
  -> inverse letterbox to original normalized x/y + visibility
  -> 63-value single-hand feature normalization
  -> Left/Right + OK dual-output classifier
  -> exactly two hands, trusted Left + Right, both OK
  -> existing 5-frame / 3-positive temporal window
  -> existing glasses/visible/center/spacing/gesture capture gate
  -> existing GUI, capture writer and status output
```

| 步骤 | 实现 | 输入 | 输出 | 失败方式 |
|---|---|---|---|---|
| 1 输入校验 | `YoloV8RknnHandLandmarkProvider::detect` | 非空 `CV_8UC3` BGR 原图 | 合法帧尺寸 | 空帧或格式错误抛异常 |
| 2 预处理 | `prepare_letterboxed_rgb` | 原图像素坐标 | 640×640 RGB uint8、letterbox 变换 | 无效缩放/填充抛异常 |
| 3 RKNN 推理 | `RknnYoloV8PoseModel::infer` | 1×640×640×3 NHWC | 4 个转为 float 的输出 buffer | 模型、Runtime、tensor 或运行错误直接失败 |
| 4 Pose 解码 | `decode_three_scale_pose` | 3 个检测特征图 + 21×3×8400 | hand 候选、score、21 点 visibility | shape/数值不匹配抛异常 |
| 5 NMS/坐标恢复 | `nms`、`map_box_to_source`、`map_keypoint_to_source` | 模型输入像素坐标 | 最多 2 手，原图像素坐标 | 低阈值/低可靠点候选被丢弃 |
| 6 属性特征 | `hand_attribute_features` | 原图归一化 x/y + 21 visibility | 63 个无单位特征 | 缺 visibility、退化手型或非有限值失败 |
| 7 双输出分类 | `HandAttributeClassifier::predict` | 63 特征 | handedness/confidence、OK score/is_ok | 缺模型、schema/长度错误直接失败 |
| 8 即时双 OK | `DoubleOKRecognizer::process_hands` | 最多 2 个单手结果 | `double_ok` | 非恰好两手、Unknown、同侧或任一非 OK 均 false |
| 9 时序稳定 | `DoubleOKRecognizer` history | 每帧 `double_ok` | `stable_double_ok` | reset 清空；默认 5 帧至少 3 帧阳性 |
| 10 门控与输出 | `evaluate_capture_gate` / dashboard | 稳定结果、框/关键点、可选眼镜姿态 | ready、阻断原因、GUI、截图/采集 | 任一门控失败时不采集 |

推理耗时只覆盖 landmark provider 的 `detect()`；总耗时从单帧 pipeline 开始到
识别和 gate 完成后由 metrics 记录。相机等待和 Qt 绘制不计入该 total。

## 4. 严格后端规则

`yolov8-rknn` 是生产路径，旧参数值 `rknn` 仅作为同一后端别名。它要求：

1. CMake 使用 `DOUBLE_OK_ENABLE_RKNN=ON`；
2. 进程为 AArch64；
3. YOLO 模型与 manifest 的 size/SHA/合同匹配；
4. `--model` 或 `attribute_model_path` 指向真实模型二。

任何一项不满足都会初始化失败，不会静默切换。`landmarks-json`、`mediapipe`、
`opencv-heuristic` 仅能显式选择；GUI 对 JSON 标注“测试后端 / 非 YOLOv8 推理”。

## 5. 构建和运行

桌面构建及测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

无 Qt/headless：

```bash
cmake -S . -B build-headless -DCMAKE_BUILD_TYPE=Release \
  -DDOUBLE_OK_BUILD_QT_DEMO=OFF \
  -DDOUBLE_OK_BUILD_CAPTURE_TOOL=OFF
cmake --build build-headless -j 2
ctest --test-dir build-headless --output-on-failure
```

RK3588 交叉编译需要 BSP 工具链和包含 OpenCV 的 sysroot：

```bash
export RK3588_TOOLCHAIN_PREFIX=/opt/rk3588-toolchain/bin/aarch64-linux-gnu-
export RK3588_SYSROOT=/opt/rk3588-sysroot
scripts/build_rk3588.sh
```

板端 GUI（`--model` 必须替换为真实模型二）：

```bash
scripts/run_demo.sh /dev/video6 \
  --width 640 --height 480 --camera-fps 30 --fourcc MJPG \
  --landmark-backend yolov8-rknn \
  --pose-model models/rk3588/hand_pose_640_fp.rknn \
  --pose-manifest models/rk3588/hand_pose_640_fp.rknn.manifest.json \
  --model /path/to/trained_hand_attribute_v1.json
```

只验证 0612 GUI 显示链，不代表 YOLOv8 推理：

```bash
scripts/run_demo.sh /dev/video6 \
  --landmark-backend landmarks-json \
  --landmarks-json configs/debug_landmarks.json
```

## 6. 验收缺口和许可风险

- 缺模型二权重、模型 SHA 和独立 tensor/artifact 评审；
- 缺 Gemini 335 左/右 × OK/非 OK 及困难负样本标注集；
- 尚无目标域 handedness/OK precision、recall、F1、误触发率；
- 尚无 RK3588 Runtime query 记录、真实 FPS、推理耗时、总耗时和长稳结果；
- 当前测试主机不能代替 RK3588/RKNPU 验收；
- YOLO 候选权重包含 Ultralytics AGPL 元数据，训练数据为 CC BY-NC-SA，闭源商用前
  必须完成许可审查并优先用自采或明确授权数据重新训练；
- RKNN SDK/Runtime 需要与目标 BSP 驱动版本匹配，其许可见
  `third_party/rknn_runtime/LICENSE.rknn_model_zoo`。
