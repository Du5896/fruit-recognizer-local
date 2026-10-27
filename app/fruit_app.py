# -*- coding: utf-8 -*-
"""
水果识别应用 - 第二部分：带 UI 的完整版本
自带模型，一键运行
PyQt6 毛玻璃 UI + 摄像头实时识别
署名：深深
"""
import sys, os, time
from pathlib import Path
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

# 导入核心模型模块
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from fruit_model import FruitRecognizer

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR.parent / "model"

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
        self.setWindowTitle("🍎 水果识别")
        self.setFixedSize(780, 620)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.rec = FruitRecognizer(MODEL_DIR)
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
        gl = QVBoxLayout(glass)
        gl.setContentsMargins(20, 16, 20, 16)
        gl.setSpacing(10)
        root.addWidget(glass)

        bar = QHBoxLayout()
        title = QLabel("🍎 水果识别")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Weight.Bold))
        bar.addWidget(title)
        bar.addStretch()
        self.mode_label = QLabel(self.rec.mode)
        self.mode_label.setStyleSheet("color: gray; font-size: 11px;")
        bar.addWidget(self.mode_label)
        bar.addSpacing(12)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("QPushButton{border:none;border-radius:14px;background:rgba(255,80,80,0.3);color:white;font-size:14px;}QPushButton:hover{background:rgba(255,80,80,0.6);}")
        bar.addWidget(close_btn)
        gl.addLayout(bar)

        self.cam_label = QLabel()
        self.cam_label.setFixedSize(680, 400)
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setStyleSheet("background: rgba(0,0,0,0.15); border-radius: 16px;")
        gl.addWidget(self.cam_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.result_label = QLabel("📷 对准水果开始识别...")
        self.result_label.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFixedHeight(50)
        gl.addWidget(self.result_label)

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
        gl.addLayout(btn_bar)

    def apply_theme(self):
        if self.is_dark:
            self.setStyleSheet("""
                QMainWindow{background:transparent;}
                QFrame#glass{background:rgba(30,30,45,220);border-radius:20px;border:1px solid rgba(255,255,255,0.1);}
                QLabel{color:white;}
                QPushButton{background:rgba(255,255,255,0.12);color:white;border:none;border-radius:10px;padding:6px 16px;font-size:13px;}
                QPushButton:hover{background:rgba(255,255,255,0.22);}
            """)
        else:
            self.setStyleSheet("""
                QMainWindow{background:transparent;}
                QFrame#glass{background:rgba(245,245,250,230);border-radius:20px;border:1px solid rgba(0,0,0,0.08);}
                QLabel{color:#333;}
                QPushButton{background:rgba(0,0,0,0.08);color:#333;border:none;border-radius:10px;padding:6px 16px;font-size:13px;}
                QPushButton:hover{background:rgba(0,0,0,0.15);}
            """)

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.theme_btn.setText("🌙 深色模式" if self.is_dark else "☀️ 浅色模式")
        self.apply_theme()

    def show_about(self):
        fruits = ["草莓","橙子","柠檬","无花果","菠萝","香蕉","番荔枝","石榴","青苹果"]
        self.result_label.setText("🍎 支持: " + "、".join(fruits))

    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.start()

    def on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        results = self.rec.predict_frame(frame)
        if results and results[0][1] >= 0.15:
            self.result_label.setText(f"✅ {results[0][0]} ({results[0][1]*100:.1f}%)")
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
    w = FruitApp()
    w.show()
    sys.exit(app.exec())
