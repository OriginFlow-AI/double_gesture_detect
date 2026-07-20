# RK3576 部署指南

## 硬件配置

- 设备: RK3576 开发板
- 摄像头: YCTC 3840x1080 MJPG

## 模型转换

在 x86 主机上转换 RK3576 可用的 RKNN 模型：

```bash
# 安装依赖
pip3 install -r scripts/requirements-rknn.txt --break-system-packages

# 转换模型 (target_platform=rk3576)
python3 scripts/convert_rknn_models.py --output-dir models/rk3576

# 传输到设备
scp models/rk3576/*.rknn rpdzkj@10.42.0.132:/data1/project/double_gesture_detect/models/rk3576/
```

## 编译

```bash
ssh rpdzkj@10.42.0.132
cd /data1/project/double_gesture_detect
mkdir -p build && cd build
cmake .. -DDOUBLE_OK_REQUIRE_RKNN=OFF -DRKNN_SDK_ROOT=/data1/project/models_rknn/rknn_sdk
make -j4
```

## 运行

```bash
export DISPLAY=:0
export LD_LIBRARY_PATH=/data1/project/models_rknn/rknn_sdk/lib:$LD_LIBRARY_PATH

./build/double-ok-demo \
  --camera /dev/video0 --width 3840 --height 1080 --camera-fps 30 --fourcc MJPG \
  --config configs/default.json \
  --landmark-backend rknn \
  --palm-model models/rk3576/palm_detection_mediapipe_2023feb_fp16.rknn \
  --hand-model models/rk3576/handpose_estimation_mediapipe_2023feb_fp16.rknn \
  --right-half
```

## 性能数据

### 单帧处理时间 (RKNN NPU 加速)

#### 单手检测 (palms=1, hands=1)

| 阶段 | 平均耗时 | 说明 |
|------|----------|------|
| Palm NPU | ~8.5ms | Palm detection 模型推理 |
| Hand prepare | ~5-10ms | 裁剪 + 旋转校正 |
| Hand NPU | ~6-8ms | Hand landmark 模型推理 |
| Hand post | ~0.01ms | 坐标逆变换 |
| **单手总计** | **~20-27ms** | |

#### 双手检测 (palms=2, hands=2)

| 阶段 | 平均耗时 | 说明 |
|------|----------|------|
| Palm NPU | ~8.5ms | 不变 |
| Hand prepare | ~17-21ms | 2x 单手 |
| Hand NPU | ~12-15ms | 2x 单手 |
| Hand post | ~0.02ms | 2x 单手 |
| **双手总计** | **~38-45ms** | |

### 资源配置

| 配置 | 分辨率 | 后端 | CPU占用 | 内存 |
|------|--------|------|---------|------|
| RK3576 | 3840x1080 右半裁剪 1280x720 | RKNN FP16 NPU | ~109% | ~197MB |

### 处理流程图

```
输入 1280x720
    │
    ▼
┌─────────────────┐
│  Palm NPU        │ ~8.5ms
│  (192x192)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hand Prepare     │ ~5-10ms
│ 裁剪+旋转校正    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hand NPU        │ ~6-8ms
│  (224x224)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hand Post        │ ~0.01ms
│ 坐标逆变换      │
└────────┬────────┘
         │
         ▼
    21个关键点
         │
         ▼
┌─────────────────┐
│ OK手势分类       │ <1ms (C++ 规则计算)
│ (C++ rule-based)│
└─────────────────┘
```

## HEVC 摄像头性能测试

### 摄像头
- 分辨率: 2560x1024 HEVC 输出
- 处理时 crop 到 1280x1024

### GStreamer Pipeline
```
v4l2src device=/dev/video0
  ! video/x-h265,stream-format=byte-stream,width=2560,height=1024,framerate=30/1
  ! h265parse
  ! queue leaky=2 max-size-buffers=1
  ! mppvideodec
  ! videoconvert
  ! video/x-raw,format=RGB
  ! appsink name=sink emit-signals=false sync=false
```

### 各环节耗时（稳定后）

| 环节 | 耗时 | 说明 |
|------|------|------|
| Decode pull | 60-200ms | 初始缓冲 ~800ms，稳定后 60-200ms |
| Convert NV12→RGB | 1-4ms | videoconvert 完成 |
| Queue 丢帧 | 1-2ms | leaky=2 控制缓冲 |
| Crop 1280x1024 | 1-2ms | Qt OpenCV 完成 |
| Inference | 18-47ms | palm+landmark+分类 |
| **Draw Qt** | **71-95ms** | **主要瓶颈** |

### 已知问题

1. H265_PARSER_REF 警告: 参考帧丢失导致短暂花屏
2. stride=2752 vs width=2560: 解码后数据有 stride padding

## 颜色空间优化分析

### 当前路径 (3 次 cvtColor)

```
GStreamer RGB888
  ↓ cvtColor RGB2BGR  [hevc_async_reader.cpp:78]
BGR888 (OpenCV 内部)
  ↓ cvtColor BGR2RGB  [onnx:536, palm prepare]
RGB (Palm NPU)
  ↓ cvtColor BGR2RGB  [onnx:667, hand prepare]
RGB (Hand NPU)
```

### 优化路径 (0 次 cvtColor)

```
GStreamer RGB888
  ↓ (直接传递)
RGB (Palm NPU)
  ↓ (直接传递)
RGB (Hand NPU)
```

### 验证

单元测试 `tests/cpp/test_color_space.cpp` 验证：
- GStreamer RGB 输出正确 ✓
- RGB → BGR 转换正确 ✓
- BGR → RGB for NPU 正确 ✓
- 直接 RGB to NPU（无转换）正确 ✓

### 改动范围

- `src/hevc_async_reader.cpp`: 移除 RGB→BGR cvtColor
- `src/onnx_mediapipe_provider.cpp`: 移除 palm/hand prepare 中的 BGR2RGB
- `src/qt_dashboard.cpp`: 调整显示路径

### 风险

- `draw_hand_tracking` 和 `render_dashboard` 假设输入是 BGR
- 需要全面回归测试
- `live_ui.cpp` 中所有 cv::Scalar 颜色都是 BGR 顺序
- 改动会破坏所有路径（包括非 HEVC）

### 结论

**未实施改动**。颜色空间优化需要全面重构（包括 draw 函数和 Qt 显示路径），风险大于收益。当前的 3 次 cvtColor 耗时 < 1ms，相比 Qt 渲染 75ms 不是瓶颈。

## 完整数据流程 (1280x1024 BGR → OK + 骨骼点)

### 输入

- 摄像头输出: 2560x1024 HEVC
- GStreamer 解码后: 2560x1024 RGB888
- hevc_async_reader 转换: RGB → BGR
- Qt crop 到 1280x1024 BGR
- 进入 landmark_provider 的帧: 1280x1024 BGR888

### 步骤 1: Palm Detection 输入准备

```
1280x1024 BGR
   ↓
resize (INTER_LINEAR, 等比例)
   scale = min(192/1280, 192/1024) = 0.15
   ↓
192x154 BGR
   ↓
copyMakeBorder (BORDER_CONSTANT, Scalar(0,0,0))
   pad: top=19, bottom=19, left=0, right=0
   ↓
192x192 BGR
   ↓
cvtColor COLOR_BGR2RGB
   ↓
192x192 RGB  ← Palm NPU 输入
```

### 步骤 2: Palm NPU 推理

```
输入:  192x192x3 RGB (FP16)
输出:
  - regression: 2944 个 anchor 的 18 维坐标 (keypoints + box)
  - scores: 2944 个 anchor 的置信度
   ↓
筛选 score > palm_detection_threshold 的 palm
   ↓
NMS (IoU 阈值 = palm_nms_threshold)
   ↓
PalmDetection { box(x,y,w,h), 7 个关键点, score }
```

### 步骤 3: Hand 输入准备 (每个 palm)

```
原图 1280x1024 BGR (frame_bgr)
   ↓
crop_and_pad (按 palm.box, for_rotation=true)
   - crop palm 矩形区域
   - side = max(crop.rows, crop.cols, hypot)
   - 方形 pad (黑色)
   ↓
方形 BGR (~300x300, 带 7 个 landmarks 的 bias)
   ↓
cvtColor COLOR_BGR2RGB
   ↓
方形 RGB
   ↓
按 wrist→middle_finger 计算旋转角 (radians)
   ↓
getRotationMatrix2D + warpAffine
   ↓
旋转后方形 RGB
   ↓
按 7 个 landmark 计算新 box
   ↓
crop_and_pad (第二次, 取 landmark 包围盒)
   ↓
resize 到 224x224
   ↓
224x224 RGB  ← Hand NPU 输入
```

### 步骤 4: Hand NPU 推理

```
输入:  224x224x3 RGB (FP16)
输出 (4 个张量):
  - landmarks: 63 维 float32 (21 个点 × xyz)
  - presence:  1 维 float32 (手存在置信度)
  - handedness: 1 维 float32 (左右手)
  - flag:      63 维 float32
   ↓
hand_presence_threshold 过滤
   ↓
21 个 landmarks (在 224x224 输入坐标系)
```

### 步骤 5: Hand Post (坐标逆变换)

```
224x224 坐标系 landmarks
   ↓
应用逆变换 (rot^(-1) - bias)
   ↓
恢复到原图 1280x1024 BGR 坐标系的 21 个 landmarks
   ↓
可选: hand_landmark_full 输出 → 直接使用绝对坐标
```

### 步骤 6: OK 手势分类 (C++ 规则)

```
输入: 21 个 landmarks (像素坐标)
   ↓
规则计算 (OKHandClassifier):
  - 计算各指尖到掌心距离
  - 计算各指弯曲度
  - 比对阈值
   ↓
bool is_ok (是 OK 手势)
```

### 最终输出

```cpp
struct DetectedHand {
    cv::Rect2d box;          // palm 检测框
    Landmarks landmarks;     // 21 个 3D 关键点
    double score;            // palm 置信度
    std::optional<bool> handedness;  // 左右手
};

struct DoubleOKResult {
    bool is_double_ok;       // 是否为双手 OK
    double positive_ratio;   // 稳定窗口内 OK 比例
    int positive_frames;
    int stable_window;
};
```

### 数据格式总结

| 阶段 | Shape | 类型 | 关键参数 |
|------|-------|------|---------|
| 输入帧 | 1280x1024x3 | uint8 BGR | - |
| Palm NPU 输入 | 192x192x3 | FP16 RGB | letterbox (黑色 0) |
| Palm NPU 输出 | 2944×18 / 2944 | FP32 | anchor + 7 关键点 |
| Hand NPU 输入 | 224x224x3 | FP16 RGB | 旋转校正 + 裁剪 |
| Hand NPU 输出 | 63, 1, 1, 63 | FP32 | 21 关键点 xyz + presence |
| 最终 landmarks | 21×3 | double | 原图坐标系 |
| OK 输出 | bool + 比例 | - | 规则计算 |
