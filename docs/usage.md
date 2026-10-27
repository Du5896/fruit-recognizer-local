# 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision PyQt6 opencv-python Pillow
```

### 2. 下载模型权重

```bash
python train/train_model.py
```

这会自动下载 MobileNetV2 预训练权重到 `model/` 目录。

### 3. 运行应用

```bash
python app/fruit_local.py
```

## 打包为 EXE

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name "水果识别" \
  --add-data "model;model" \
  app/fruit_local.py
```

生成的 EXE 在 `dist/` 目录。

## 常见问题

**Q: 摄像头打不开？**
A: 检查是否被其他应用占用，或在系统设置中允许摄像头权限。

**Q: 识别不准？**
A: 确保光线充足，水果对准摄像头中央，背景简洁。

**Q: 支持哪些水果？**
A: 当前支持 9 种：草莓、橙子、柠檬、无花果、菠萝、香蕉、青苹果、番荔枝、石榴。
