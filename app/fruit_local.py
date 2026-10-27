# -*- coding: utf-8 -*-
"""
水果识别本地版 - PyQt6 单文件应用
使用 MobileNetV2 预训练模型，完全离线运行
毛玻璃 + 动画 + 深浅色主题
署名：深深
"""
import sys, os, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
IMG_SIZE = 224

class FruitModel:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.transform = T.Compose([
            T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.load()

    def load(self):
        pt_path = MODEL_DIR / "fruit_mobilenetv2.pt"
        labels_path = MODEL_DIR / "imagenet_labels.json"
        if pt_path.exists():
            self.model = models.mobilenet_v2(weights=None)
            self.model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
            self.model.eval()
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)
            else:
                self.class_names = [f"class_{i}" for i in range(1000)]
            self.mode = f"MobileNetV2 (本地离线, {len(self.class_names)}类)"
        else:
            self.mode = "未找到模型文件"

    def predict(self, frame_bgr):
        if self.model is None:
            return [("模型未加载", 0.0)]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top5 = torch.topk(probs, 5)
        results = []
        for idx, prob in zip(top5.indices, top5.values):
            name = self.class_names[idx] if self.class_names else f"class_{idx}"
            results.append((name, float(prob)))
        return results

class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.running = False
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            time.sleep(0.033)
        cap.release()

class GlassFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glass")

class FruitApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水果识别 - 本地版")
        self.setFixedSize(780, 620)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.model = FruitModel()
        self.camera_thread = None
        self.is_dark = True
        self.drag_pos = None
        self.setup_ui()
        self.apply_theme()
        self.start_camera()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)
        glass = GlassFrame()
        glass_layout = QVBoxLayout(glass)
        glass_layout.setContentsMargins(20, 16, 20, 16)
        glass_layout.setSpacing(10)
        root.addWidget(glass)
        title_bar = QHBoxLayout()
        title = QLabel("🍎 水果识别")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        title_bar.addWidget(title)
        title_bar.addStretch()
        self.mode_label = QLabel(self.model.mode)
        self.mode_label.setStyleSheet("color: gray; font-size: 11px;")
        title_bar.addWidget(self.mode_label)
        title_bar.addSpacing(12)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("QPushButton { border: none; border-radius: 14px; background: rgba(255,80,80,0.3); color: white; font-size: 14px; } QPushButton:hover { background: rgba(255,80,80,0.6); }")
        title_bar.addWidget(close_btn)
        glass_layout.addLayout(title_bar)
        self.cam_label = QLabel()
        self.cam_label.setFixedSize(680, 400)
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setStyleSheet("background: rgba(0,0,0,0.15); border-radius: 16px;")
        glass_layout.addWidget(self.cam_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.result_label = QLabel("📷 对准水果开始识别...")
        self.result_label.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFixedHeight(50)
        glass_layout.addWidget(self.result_label)
        btn_bar = QHBoxLayout()
        self.theme_btn = QPushButton("🌙 深色模式")
        self.theme_btn.setFixedHeight(36)
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self.toggle_theme)
        btn_bar.addWidget(self.theme_btn)
        btn_bar.addStretch()
        info_btn = QPushButton("ℹ️ 关于")
        info_btn.setFixedHeight(36)
        info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        info_btn.clicked.connect(self.show_about)
        btn_bar.addWidget(info_btn)
        glass_layout.addLayout(btn_bar)

    def apply_theme(self):
        if self.is_dark:
            self.setStyleSheet("""
                QMainWindow { background: transparent; }
                QFrame#glass { background: rgba(30, 30, 45, 220); border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
                QLabel { color: white; }
                QPushButton { background: rgba(255,255,255,0.12); color: white; border: none; border-radius: 10px; padding: 6px 16px; font-size: 13px; }
                QPushButton:hover { background: rgba(255,255,255,0.22); }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background: transparent; }
                QFrame#glass { background: rgba(245, 245, 250, 230); border-radius: 20px; border: 1px solid rgba(0,0,0,0.08); }
                QLabel { color: #333; }
                QPushButton { background: rgba(0,0,0,0.08); color: #333; border: none; border-radius: 10px; padding: 6px 16px; font-size: 13px; }
                QPushButton:hover { background: rgba(0,0,0,0.15); }
            """)

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.theme_btn.setText("🌙 深色模式" if self.is_dark else "☀️ 浅色模式")
        self.apply_theme()

    def show_about(self):
        fruits = ["草莓", "橙子", "柠檬", "无花果", "菠萝", "香蕉", "番荔枝", "石榴", "青苹果"]
        self.result_label.setText("🍎 支持识别: " + "、".join(fruits))

    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.start()

    def on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.cam_label.setPixmap(pix)
        results = self.model.predict(frame)
        top_name, top_conf = results[0]
        if top_conf > 0.15:
            self.result_label.setText(f"✅ {top_name}  ({top_conf*100:.1f}%)")
        else:
            self.result_label.setText("🔍 未识别到水果，请对准水果")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_pos)

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.running = False
            self.camera_thread.wait(2000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FruitApp()
    window.show()
    sys.exit(app.exec())
