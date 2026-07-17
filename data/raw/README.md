# Raw Data

本目录保存实时门控产生的原始图片和配套 metadata，不提交大文件。

默认输出目录：

```text
captures/
```

每张图片对应一个 `.jpg.json` 文件，使用 `double_ok_capture_v2` schema。自动采集标签
来自运行时预测，后续训练前仍需人工复核。
