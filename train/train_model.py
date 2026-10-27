# -*- coding: utf-8 -*-
"""下载 MobileNetV2 预训练模型并保存"""
import os, json, torch
import torchvision.models as models

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
os.makedirs(model_dir, exist_ok=True)

print("加载预训练 MobileNetV2...")
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

pt_path = os.path.join(model_dir, "fruit_mobilenetv2.pt")
torch.save(model.state_dict(), pt_path)
print(f"模型保存到 {pt_path} ({os.path.getsize(pt_path)/1024/1024:.1f}MB)")

import urllib.request
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
with urllib.request.urlopen(url, timeout=30) as r:
    labels = [l.decode().strip() for l in r.readlines()]

labels_path = os.path.join(model_dir, "imagenet_labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)
print(f"标签保存到 {labels_path}")
