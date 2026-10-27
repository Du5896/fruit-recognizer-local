# 🍎 水果识别本地版 (Fruit Recognizer Local)

> 基于 MobileNetV2 的本地离线水果识别应用，完全不依赖云端 API，摄像头实时识别，毛玻璃 UI。

署名：**深深**

## ✨ 特性

- 🚀 **完全离线**：本地 MobileNetV2 模型，不需要网络，不需要 API Key
- 📷 **摄像头实时识别**：打开摄像头对准水果即可识别
- 🎨 **毛玻璃 UI**：深色/浅色主题切换，圆角磨边，流畅动画
- 🪶 **轻量**：模型仅 14MB，启动即用
- 🐍 **纯 Python**：PyQt6 + PyTorch，跨平台

## 📦 支持识别的水果

基于 ImageNet 预训练 MobileNetV2（1000 类），包含以下常见水果：

| 水果 | 英文 |
|------|------|
| 🍓 草莓 | strawberry |
| 🍊 橙子 | orange |
| 🍋 柠檬 | lemon |
| 🥝 无花果 | fig |
| 🍍 菠萝 | pineapple |
| 🍌 香蕉 | banana |
| 🍎 青苹果 | Granny Smith |
| 🍈 石榴 | pomegranate |
| 🍐 番荔枝 | custard apple |

> 更多水果类别可通过下载 Fruits-360 数据集微调扩展（131 类）。

## 🛠 技术栈

- **模型**: MobileNetV2 (PyTorch, ImageNet 预训练)
- **UI**: PyQt6
- **图像处理**: OpenCV + Pillow
- **离线推理**: 本地 .pt 模型文件，无需 ONNX/TensorFlow

## 📁 项目结构

```
fruit-recognizer-local/
├── README.md                 # 本文件
├── LICENSE                   # MIT 开源协议
├── requirements.txt          # Python 依赖
├── .gitignore
├── app/
│   └── fruit_local.py        # 主应用（PyQt6 单文件）
├── model/
│   ├── fruit_mobilenetv2.pt  # MobileNetV2 预训练权重 (14MB)
│   └── imagenet_labels.json   # ImageNet 1000 类标签
├── train/
│   ├── train_model.py        # 模型下载与导出脚本
│   └── test_model.py          # 模型推理测试
└── docs/
    └── screenshots/           # 应用截图
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 摄像头（可选，用于实时识别）

### 安装依赖

```bash
pip install torch torchvision PyQt6 opencv-python Pillow
```

### 运行

```bash
cd app
python fruit_local.py
```

### 打包为 EXE（Windows）

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name "水果识别" ^
  --add-data "model;model" ^
  app/fruit_local.py
```

## 🎯 使用方法

1. 启动应用
2. 授权摄像头访问
3. 把水果对准摄像头
4. 界面显示识别结果和置信度
5. 点击右上角按钮切换深色/浅色主题

## 📝 关于

- 作者：深深
- 模型：MobileNetV2 (ImageNet)
- 许可证：MIT

## ⚠️ 注意

- 识别准确率取决于光照和拍摄角度，建议在明亮光线下对准水果
- 模型基于 ImageNet 预训练，对常见水果识别效果较好
- 非水果物体置信度会较低，界面会提示"未识别到水果"
