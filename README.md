# 🍎 水果识别本地版 (Fruit Recognizer Local)

> 基于 MobileNetV2 的本地离线水果识别应用，完全不依赖云端 API。

署名：**深深**

## 📦 两部分结构

本项目分为两部分：

### 第一部分：模型源文件（纯模型，不含 UI）

```
core/
└── fruit_model.py    # FruitRecognizer 类，纯模型推理
```

- 只依赖 PyTorch + OpenCV + Pillow
- 可独立使用，不依赖 PyQt
- 支持三种输入：PIL 图像、OpenCV BGR 帧、图片文件
- 适合嵌入到自己的项目中

```python
from core.fruit_model import FruitRecognizer
rec = FruitRecognizer("model/")
results = rec.predict_file("apple.jpg")  # [("Granny Smith", 0.95), ...]
```

### 第二部分：带 UI 的完整应用（一键识别）

```
app/
└── fruit_app.py      # PyQt6 毛玻璃 UI 应用
```

- 自带模型，一键运行
- 摄像头实时识别
- 毛玻璃 UI，深浅色主题切换
- 无边框窗口，可拖拽

```bash
python app/fruit_app.py
```

## 🛠 技术栈

| 组件 | 选择 |
|------|------|
| 模型 | MobileNetV2 (PyTorch, ImageNet 预训练) |
| UI | PyQt6 |
| 图像处理 | OpenCV + Pillow |
| 推理 | 本地 .pt 模型，完全离线 |

## 📁 完整项目结构

```
fruit-recognizer-local/
├── core/                        # 第一部分：模型源文件
│   └── fruit_model.py
├── app/                         # 第二部分：带 UI 的应用
│   └── fruit_app.py
├── model/                       # 模型权重
│   └── imagenet_labels.json
├── train/                       # 训练脚本
│   ├── train_model.py
│   └── test_model.py
├── docs/                        # 文档
│   ├── architecture.md
│   └── usage.md
├── screenshots/                 # 截图
├── .github/                     # GitHub 模板
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## 🚀 快速开始

```bash
# 安装依赖
pip install torch torchvision PyQt6 opencv-python Pillow

# 下载模型权重
python train/train_model.py

# 运行带 UI 的应用
python app/fruit_app.py
```

## 📝 关于

- 作者：深深
- 模型：MobileNetV2 (ImageNet)
- 许可证：MIT
