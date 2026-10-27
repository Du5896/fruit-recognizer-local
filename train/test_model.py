# -*- coding: utf-8 -*-
"""模型推理测试"""
import os, json, torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

model = models.mobilenet_v2(weights=None)
model.load_state_dict(torch.load(os.path.join(model_dir, "fruit_mobilenetv2.pt"), map_location="cpu", weights_only=True))
model.eval()

with open(os.path.join(model_dir, "imagenet_labels.json"), "r", encoding="utf-8") as f:
    labels = json.load(f)

img = Image.new("RGB", (224, 224), (245, 245, 245))
draw = ImageDraw.Draw(img)
draw.ellipse([50, 40, 174, 180], fill=(200, 30, 30))

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    output = model(tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5 = torch.topk(probs, 5)

print("测试结果:")
for idx, prob in zip(top5.indices, top5.values):
    print(f"  {labels[idx]} ({prob*100:.1f}%)")
