# -*- coding: utf-8 -*-
"""
水果识别核心模型（纯模型，不含 UI）
这是第一部分：模型源文件
只依赖 PyTorch + OpenCV + Pillow，可独立使用
署名：深深
"""
import os, json
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2

IMG_SIZE = 224

class FruitRecognizer:
    """纯模型识别引擎，不含任何 UI 依赖
    
    用法：
        rec = FruitRecognizer()
        results = rec.predict_frame(frame_bgr)  # OpenCV BGR 帧
        results = rec.predict_pil(pil_image)     # PIL 图像
        results = rec.predict_file("apple.jpg")  # 图片文件
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent
        self.model_dir = Path(model_dir)
        self.model = None
        self.class_names = None
        self.mode = "未加载"
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._load()

    def _load(self):
        pt_path = self.model_dir / "fruit_mobilenetv2.pt"
        labels_path = self.model_dir / "imagenet_labels.json"
        if pt_path.exists():
            self.model = models.mobilenet_v2(weights=None)
            self.model.load_state_dict(
                torch.load(pt_path, map_location="cpu", weights_only=True))
            self.model.eval()
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)
            else:
                self.class_names = [f"class_{i}" for i in range(1000)]
            self.mode = f"MobileNetV2 (离线, {len(self.class_names)}类)"

    def _predict_tensor(self, tensor):
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top5 = torch.topk(probs, 5)
        results = []
        for idx, prob in zip(top5.indices, top5.values):
            name = self.class_names[idx] if self.class_names else f"class_{idx}"
            results.append((name, float(prob)))
        return results

    def predict_pil(self, pil_img):
        """识别 PIL 图像"""
        if self.model is None:
            return [("模型未加载", 0.0)]
        tensor = self.transform(pil_img.convert("RGB")).unsqueeze(0)
        return self._predict_tensor(tensor)

    def predict_frame(self, frame_bgr):
        """识别 OpenCV BGR 帧"""
        if self.model is None:
            return [("模型未加载", 0.0)]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.predict_pil(pil)

    def predict_file(self, image_path):
        """识别图片文件"""
        pil = Image.open(image_path)
        return self.predict_pil(pil)

    def is_fruit(self, results, threshold=0.15):
        """判断是否识别到水果"""
        if not results:
            return False
        return results[0][1] >= threshold

# 快速测试
if __name__ == "__main__":
    rec = FruitRecognizer()
    print(f"模型: {rec.mode}")
    # 测试随机图
    import numpy as np
    fake = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    r = rec.predict_frame(fake)
    print("随机图 Top3:")
    for name, conf in r[:3]:
        print(f"  {name} ({conf*100:.1f}%)")
