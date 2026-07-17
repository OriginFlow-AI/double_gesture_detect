# Runtime Configuration

默认读取 `configs/default.json`。未知字段、错误类型、非有限数和越界值都会在启动时
报错。

## 顶层字段

| 字段 | 约束 | 作用 |
| --- | --- | --- |
| `max_num_hands` | 1 或 2 | 最多保留的手数 |
| `ok_threshold` | `[0,1]` | 单手 OK 阈值 |
| `stable_window` | 正整数 | 时序窗口长度 |
| `stable_min_positive` | `1..stable_window` | 窗口内最少阳性帧数 |
| `input_mirrored` | 布尔 | 输入是否为镜像画面，用于解释手别 |
| `palm_model_path` | 非空 ONNX 路径 | 掌心检测模型 |
| `hand_model_path` | 非空 ONNX 路径 | 21 点手部模型 |
| `rknn_palm_model_path` | 非空 RKNN 路径 | NPU 掌心检测模型 |
| `rknn_hand_model_path` | 非空 RKNN 路径 | NPU 21 点手部模型 |
| `palm_detection_threshold` | `[0,1]` | 掌心候选阈值 |
| `hand_presence_threshold` | `[0,1]` | 21 点结果存在阈值 |
| `palm_nms_threshold` | `[0,1]` | 掌心 NMS IoU 阈值 |

`capture_gate` 控制姿态、完整入框、居中、双手间距和稳定 Double-OK 条件。
`data_capture` 的 `enabled`、`output_dir` 和 `cooldown_sec` 控制自动保存。

CLI 可用 `--palm-model`、`--hand-model`、`--threshold`、
`--capture-output-dir` 和 `--capture-cooldown` 覆盖对应配置。数值必须完整可解析，
例如 `--width 640px` 会报错。

`--landmark-backend` 支持 `rknn`、`onnx`、`landmarks-json`、`none`。启用 RKNN
的构建默认使用 `rknn`，其他构建默认使用 `onnx`。显式请求不可用的 RKNN 会报错，
不会自动切换 CPU。
`--log-level` 支持 `DEBUG`、`INFO`、`WARNING`、`ERROR`。
