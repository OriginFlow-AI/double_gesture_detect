# Double OK Gesture Capture Gate

在 GLASSES 端开始采集前，确认以下条件同时满足：

1. 眼镜姿态在允许范围内。
2. 两只手完整进入相机 FOV 的中心区域。
3. 两只手保持足够距离。
4. 两只手稳定做出 OK 手势。

全部满足时门控返回 `ready=true`，采集流程才保存画面或触发下一步动作。

## 环境安装

项目要求 Python 3.10 或更高版本。推荐使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

项目统一使用 `opencv-contrib-python`。不要同时安装 `opencv-python`，否则两个发行包会共同提供 `cv2`，可能产生二进制冲突。旧环境建议删除后重建。

## 核心流程

```text
摄像头 BGR 帧
-> MediaPipe 检测两只手
-> 每只手提取 21 点归一化特征
-> 模型或几何规则判断单手 OK
-> 时间窗口判断稳定双手 OK
-> 姿态、完整入框、中心位置、双手距离门控
-> ready / 阻断原因
```

主要模块：

```text
config.py          JSON 配置加载与公共配置转换
features.py        landmark 校验、归一化与特征提取
recognizer.py      单手分类、双手组合与时间稳定判断
capture_gate.py    姿态、FOV、中心区域和距离门控
prepare_hagrid.py  HaGRID 标注转换为特征 CSV
train.py           NumPy / scikit-learn 模型训练
evaluate.py        独立 split 模型评估
demo.py            单目或双目实时演示
capture_samples.py 本地样本采集
gui.py             静态 HTML 状态报告
```

## 实时运行

先确认 Orbbec 设备可以打开：

```bash
scripts/check_camera.sh /dev/video0
```

使用 Orbbec Gemini 335 单目彩色流：

```bash
scripts/run_demo.sh /dev/video0
```

实时窗口显示手部框、21 点、左右手 OK 分数、门控检查项、FPS、处理延迟和实际分辨率。按
`Q` 或 `Esc` 退出。设备刚接入尚未就绪时会自动重试；连续读帧失败会明确报错。

直接运行 Python 入口：

```bash
python -m double_ok_gesture.demo \
  --camera /dev/video0 \
  --model models/ok_hand_numpy_logreg.pkl \
  --capture-gate
```

双目设备，以左眼作为门控基准：

```bash
python -m double_ok_gesture.demo \
  --left-camera /dev/video-left \
  --right-camera /dev/video-right \
  --model models/ok_hand_numpy_logreg.pkl \
  --capture-gate \
  --stereo-gate left
```

要求左右眼同时满足条件：

```bash
python -m double_ok_gesture.demo \
  --left-camera /dev/video-left \
  --right-camera /dev/video-right \
  --model models/ok_hand_numpy_logreg.pkl \
  --capture-gate \
  --stereo-gate both
```

不要把 V4L2 metadata 节点当作图像流；可先用 `--list-cameras` 或
`scripts/check_camera.sh` 检查。生产部署建议通过 udev 为左右相机建立稳定设备别名。

显式传入的模型文件不存在或特征结构不兼容时，程序会立即报错，不会静默切换分类策略。不传 `--model` 时才使用几何规则。

## 姿态输入

GLASSES 端可持续写入包含完整角度的 JSON 文件：

```json
{"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
```

启用姿态门控：

```bash
python -m double_ok_gesture.demo \
  --camera /dev/video0 \
  --capture-gate \
  --require-glasses-pose \
  --glasses-pose /path/to/glasses_pose.json
```

当文件正在被替换、JSON 暂时不完整或缺少任一角度时，门控返回“等待眼镜姿态数据”，不会误判为姿态合格。

## 训练与评估

```bash
scripts/prepare_hagrid.sh data/raw/hagrid/annotations
scripts/train_numpy_logreg.sh
scripts/evaluate_numpy_logreg.sh
```

负样本上限按 `split + gesture_label` 分别计算，避免某个 split 抢占全部负样本。评估默认选择 `test`，没有 `test` 时选择 `val`，不会默认在训练全集上报告指标：

```bash
python -m double_ok_gesture.evaluate --split auto
```

训练阶段仅在 `val` 同时包含正负类时使用它；否则从 `train` 内部分层留出验证集。独立 `test`
始终只由评估命令使用。

模型使用 joblib/pickle 格式，只能加载本项目生成或其他可信来源的文件；这类格式在反序列化时可以执行代码。

## 本地采集

手动采集，门控未通过时空格键不会保存：

```bash
python -m double_ok_gesture.capture_samples \
  --label double_ok \
  --gate \
  --camera /dev/video0
```

满足条件后自动采集：

```bash
python -m double_ok_gesture.capture_samples \
  --label double_ok \
  --gate \
  --auto-capture \
  --camera /dev/video0
```

采集负样本时，门控要求双手完整、居中并保持间距，同时明确阻止双手 OK，避免标签污染：

```bash
python -m double_ok_gesture.capture_samples \
  --label not_double_ok \
  --gate \
  --camera /dev/video0
```

## 测试与报告

```bash
scripts/test.sh
scripts/gui_report.sh
```

`scripts/test.sh` 会隔离机器上与本项目无关的 pytest 插件。静态数据与门控模拟报告输出到
`reports/gui/index.html`；实时测试界面由 `scripts/run_demo.sh` 启动。

所有识别与门控阈值集中在 `configs/default.json`。生产审计、验收与遗留风险见
`docs/production_readiness.md`。
