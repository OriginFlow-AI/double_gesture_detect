# Runtime Configuration

运行入口默认读取 `configs/default.json`。配置文件必须是合法 JSON 对象；字段按
所属对象读取，不会在不同嵌套层之间串值，未知字段会被拒绝以避免拼写错误静默生效。
数字必须为有限值，整数字段不能写成小数。

## 顶层识别与模型字段

| 字段 | 类型/约束 | 作用 |
| --- | --- | --- |
| `max_num_hands` | 整数，1 或 2 | 单帧最多保留的手数 |
| `ok_threshold` | `[0,1]` | 单手 OK 判定阈值 |
| `stable_window` | 正整数 | 时序窗口帧数 |
| `stable_min_positive` | `1..stable_window` | 窗口内最少阳性帧数 |
| `min_detection_confidence` | `[0,1]` | 历史调试后端检测阈值 |
| `min_tracking_confidence` | `[0,1]` | 历史调试后端跟踪阈值 |
| `handedness_confidence_threshold` | `[0.5,1]` | Left/Right 最低可信度 |
| `input_mirrored` | 布尔 | 属性特征是否需要取消镜像 |
| `pose_model_path` | 字符串路径 | 默认 RKNN Pose 模型 |
| `pose_manifest_path` | 字符串路径 | RKNN 模型合同 manifest |
| `pose_input_size` | `>=32` 且为 32 的倍数 | Pose 网络输入边长 |
| `pose_min_detection_confidence` | `[0,1]` | Pose 候选阈值 |
| `pose_min_keypoint_visibility` | `[0,1]` | 关键点可靠阈值 |
| `pose_nms_iou_threshold` | `[0,1]` | NMS IoU 阈值 |
| `pose_min_reliable_keypoints` | `1..21` | 单手最少可靠关键点数 |
| `attribute_model_path` | 字符串路径 | Left/Right + OK 属性模型；ONNX 可选、RKNN 必需 |

`pose_min_reliable_keypoints` 在 NMS/最多两手截断前生效，避免高检测分但关键点不可用
的候选挤掉有效手。实验几何 scorer 还会检查手势所需远端点是否达到
`pose_min_keypoint_visibility`；不可靠时保留检测框但输出非 OK。

## `capture_gate`

`pitch/roll/yaw_min/max` 必须有限且最小值不大于最大值；`center_x/y_min/max`
必须满足 `0 <= min < max <= 1`；`frame_margin` 范围为 `[0,0.5)`；
`min_hand_separation` 必须非负。布尔开关保持现有门控语义：姿态、完整入框、居中、
双手间距和稳定 Double-OK 都在同一核心函数内判断，UI 不会重算规则。

## `data_capture`

| 字段 | 类型/约束 | 作用 |
| --- | --- | --- |
| `enabled` | 布尔 | 是否允许 gate ready 后自动保存 |
| `output_dir` | 非空字符串（启用时） | 原始帧和 metadata 输出目录 |
| `cooldown_sec` | 有限非负数 | 两次自动保存的最短间隔 |

命令行覆盖在应用后会再次执行同一套校验，因此 `--threshold`、
`--capture-cooldown` 等参数不能绕过配置合同。CLI 数值要求整个 token 都能解析，
例如 `--width 640px` 会直接报错。

## 日志与失败策略

`--log-level` 支持 `DEBUG`、`INFO`、`WARNING`、`ERROR`。日志写到标准错误，状态与
评估结果继续写到标准输出，便于脚本分流。模型、manifest、配置或生产后端缺失时
启动失败；调试后端不会作为生产后端的自动 fallback。
