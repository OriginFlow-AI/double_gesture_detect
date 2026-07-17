# Recognition Accuracy and Diagnostics

当前默认模型是 OpenCV Zoo MediaPipe FP32 双模型。掌心网络先给出旋转方向，手部网络
再输出 21 点、手存在置信度和手别；FP32 版本用于效果优先，未采用会明显损失精度的
量化版本。

OK 分数仍是 21 点几何匹配分，不是经过目标场景校准的概率。规则同时检查拇食指
接近、食指弯曲和其余三指展开，并使用原图像素等距坐标，避免宽高分别归一化带来的
形变。

逐帧诊断：

```bash
build/double-ok-headless \
  --camera /dev/video6 \
  --disable-auto-capture \
  --status-interval 0 \
  --max-frames 100 \
  --log-level INFO
```

自动采集的 `double_ok=true` 是运行时预测，不是人工真值。正式验收应按人员和会话
隔离数据，覆盖标准 OK、张掌、拳头、V、点赞、遮挡、裁边、旋转和快速运动，分别
统计手检出率、单手 OK precision/recall、稳定双手事件召回和持续负样本误触发。
阈值只在验证集选择，测试集不得再次调参。

当前自动测试验证模型合同、OpenCV 4.6 加载、黑帧负样本和核心规则；真实摄像头已
验证掌心阶段约 7–10 ms/帧（当前 x86 主机）。具体 RK3588 FPS、长稳和目标域精度
仍须在目标板与目标人群上实测。
