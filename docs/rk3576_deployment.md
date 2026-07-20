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
