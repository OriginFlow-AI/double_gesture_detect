# Gemini 335 Kalibr 推荐配置

配置目录：

```text
configs/kalibr/gemini335/
├── aprilgrid_6x6_a3.yaml
├── imu.yaml
└── recommended_values.yaml
```

## 1. 最终推荐组合

优先使用左 IR 相机，而不是 RGB：

```text
相机：left_ir，全局快门
图像：1280 x 800 @ 30 Hz
模型：pinhole-radtan
图像 topic：/camera/left_ir/image_raw
IMU topic：/camera/gyro_accel/sample
IMU：200 Hz、±4g、±1000 dps
```

Gemini 335 的 RGB 是滚动快门。标准相机-IMU Kalibr 对快速运动下的滚动快门更敏感；左 IR 是全局快门，更适合先得到稳定外参和时延。如果业务必须使用 RGB，应单独进行滚动快门内参标定，或者使用 SDK 提供的 left-IR 到 RGB 出厂外参完成坐标链转换。

## 2. 厂家值、计算值与选择

| 项目 | 厂家值 | 计算或建议值 | 最终选择 |
|---|---:|---:|---:|
| IMU 频率 | 50/100/200/500/1000 Hz | 静态筛选上限和数据量折中 | 200 Hz |
| 加速度范围 | ±4g | `4 × 9.80665 = 39.2266 m/s²` | ±4g |
| 陀螺仪范围 | ±1000 dps | `1000 × π/180 = 17.453293 rad/s` | ±1000 dps |
| 左 IR FOV | 约 91° × 65° | `fx≈628.93, fy≈627.87` | 仅作结果核对 |
| 1280×800 典型深度内参 | `fx=620, fy=620, cx=640, cy=400` | 与 FOV 反算差约 1.44%/1.27% | 两组都只作核对 |
| RGB FOV | 约 86° × 55° | `fx≈686.32, fy≈691.55` @ 1280×720 | 仅作结果核对 |
| 相机频率 | 最高 30 Hz | 与生产模式一致 | 30 Hz |
| IMU 噪声密度 | 厂家未公开 | 由首轮静态筛选标准差换算 | 使用 `imu.yaml` 起始值 |
| IMU random walk | 厂家未公开 | 低成本 MEMS 保守起始值 | 使用 `imu.yaml`，随后用 Allan 替换 |

FOV 反算内参：

```text
fx = width  / (2 × tan(horizontal_FOV / 2))
fy = height / (2 × tan(vertical_FOV / 2))
```

这些内参只是数量级核对，不能写成最终 `camchain.yaml`。厂家表格给的是典型深度内参，FOV 反算又忽略了畸变和 FOV 公差；畸变、主点和真实焦距必须由 Kalibr 实测。

厂家给出的坐标方向是 X 向右、Y 向下、Z 向前；左 IR 光心也是深度坐标原点。IMU 相对左 IR 光心的厂家参考位置为：

```text
[-0.246, -0.065, -16.948] mm
中心距离约 16.950 mm
```

这个位置只能在统一 Kalibr 坐标系和变换方向后用于结果核对，不能直接抄成 `T_cam_imu`。

## 3. IMU YAML 全部数值

```yaml
imu0:
  model: calibrated
  rostopic: /camera/gyro_accel/sample
  update_rate: 200.0
  gyroscope_noise_density: 7.0711e-4
  gyroscope_random_walk: 4.0e-5
  accelerometer_noise_density: 5.6569e-3
  accelerometer_random_walk: 4.0e-3
```

前两个 noise density 来自首轮静止筛选标准差上限：

```text
gyro:  0.01 / sqrt(200) = 7.0711e-4 rad/s/sqrt(Hz)
accel: 0.08 / sqrt(200) = 5.6569e-3 m/s²/sqrt(Hz)
```

两个 random walk 没有厂家数据支撑，是首次优化用的保守起始值。正式结果应录制恒温静止 IMU 数据并用 Allan deviation 替换；Kalibr 官方建议约 15–24 小时。

## 4. 标定板全部数值

```yaml
target_type: aprilgrid
tagCols: 6
tagRows: 6
tagSize: 0.036
tagSpacing: 0.30
```

推荐 A3、100% 比例打印。理论网格宽度：

```text
6 × 36 mm + 5 × (36 mm × 0.30) = 270 mm
```

打印后用卡尺测量标签黑色外边缘。实测标签边长与 `36.00 mm` 相差超过 `0.20 mm` 时，必须把 YAML 中 `tagSize` 改成实测均值。

## 5. Orbbec ROS2 参数

使用官方 `v2-main` ROS2 wrapper 和 `gemini_330_series.launch.py`。关键值：

```text
enable_left_ir=true
left_ir=1280x800@30
enable_accel=true
enable_gyro=true
enable_sync_output_accel_gyro=true
accel_rate=200hz
gyro_rate=200hz
accel_range=4g
gyro_range=1000dps
enable_accel_data_correction=true
enable_gyro_data_correction=true
mirror=false
flip=false
rotation=0
```

不同 wrapper 版本的 IR 参数名可能是 `ir_*` 或 `left_ir_*`。启动后以实际 topic 和 profile 列表为准：

```bash
ros2 topic list
ros2 topic hz /camera/left_ir/image_raw
ros2 topic hz /camera/gyro_accel/sample
```

必须确认图像约 30 Hz、IMU 约 200 Hz，且 IMU 消息同时含线加速度与角速度。

## 6. 相机内参数据集

```text
录制：90 秒
原始图像：30 Hz
Kalibr --bag-freq：4 Hz
距离：0.35–1.20 m
标定板覆盖画面：15%–80%
有效检测帧：至少 80，推荐 150+
```

运动应覆盖画面中心、四角、各边缘，并包含明显的倾斜和距离变化。不要只让标定板平行于成像面。

运行：

```bash
kalibr_calibrate_cameras \
  --bag gemini335_camera.bag \
  --topics /camera/left_ir/image_raw \
  --models pinhole-radtan \
  --target configs/kalibr/gemini335/aprilgrid_6x6_a3.yaml \
  --bag-freq 4.0
```

输出的 `camchain-*.yaml` 才是下一阶段输入。

## 7. 相机-IMU 动态数据集

```text
预热：60 秒
总时长：150 秒，允许 120–180 秒
开始静止：5 秒
结束静止：5 秒
图像：30 Hz
IMU：200 Hz
目标可见帧比例：至少 70%
各轴旋转：至少 3 轮
各轴平移：至少 3 轮
主要角速度：30–90 dps
硬上限：150 dps
初始时延猜测：0 秒，由 Kalibr 估计
```

光照充足时关闭自动曝光，曝光从 `3000 µs` 开始，尽量不超过 `5000 µs`，把标签边缘运动模糊控制在约 1.5 像素以内。

运行：

```bash
kalibr_calibrate_imu_camera \
  --bag gemini335_camera_imu.bag \
  --cam camchain-gemini335_camera.yaml \
  --imu configs/kalibr/gemini335/imu.yaml \
  --target configs/kalibr/gemini335/aprilgrid_6x6_a3.yaml
```

## 8. 验收标准

至少独立录制并运行三次：

```text
相机内参平均重投影误差：推荐 ≤0.20 px，最大 ≤0.30 px
相机-IMU平均重投影误差：≤0.50 px
三次旋转外参差异：≤0.50°
三次平移外参差异：≤5 mm
三次时延差异：≤2 ms
```

同时检查报告中预测角速度、加速度与 IMU 测量是否贴合。单次优化成功但三次结果不一致，不能作为最终外参。

## 9. 当前环境缺口

本机当前只有 ROS2 Jazzy，尚未安装：

```text
OrbbecSDK ROS2 wrapper
pyorbbecsdk
Kalibr
```

传统 Kalibr 常使用 ROS1 `.bag`；如果安装的是 ROS2 兼容版本，可直接使用 rosbag2，否则需要把采集结果转换为 ROS1 bag。
