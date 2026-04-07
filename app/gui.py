from __future__ import annotations

from datetime import datetime
import argparse
import os
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QPoint, QProcess, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import AppPaths, DEFAULT_DECISIONS, sanitize_session_name
from .config import DEFAULT_CLASSES
from .dataset import (
    list_capture_meta,
    list_sessions,
    load_capture_meta,
    save_capture,
    save_capture_meta,
)
from .inference import Detection, Detector, draw_detections

TABLE_HEADERS = ["label", "confidence", "x1", "y1", "x2", "y2"]


def _list_model_files(paths: AppPaths) -> list[str]:
    exts = {".pt", ".onnx"}
    found: list[Path] = []

    for base in [paths.models_dir, paths.runs_dir]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                found.append(path)
    for path in paths.root.iterdir():
        if path.is_file() and path.suffix.lower() in exts:
            found.append(path)

    result: list[str] = []
    for path in sorted(set(found)):
        try:
            result.append(str(path.relative_to(paths.root)))
        except Exception:
            result.append(str(path))
    return result


def _scan_cameras(max_index: int = 6) -> list[int]:
    available: list[int] = []
    consecutive_failures = 0
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        cap.release()
        if ok:
            available.append(idx)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if idx >= 1 and consecutive_failures >= 2:
                break
    return available


def _detection_from_dict(item: dict) -> Detection:
    label = str(item.get("label", "obstacle"))
    confidence = float(item.get("confidence", 1.0))
    bbox = item.get("bbox", [0, 0, 1, 1])
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = [0, 0, 1, 1]
    decision_hint = "stop_wait" if label == "person" else "bypass_candidate"
    return Detection(
        label=label,
        confidence=confidence,
        bbox=tuple(int(v) for v in bbox),
        decision_hint=decision_hint,
    )


class ReviewImageLabel(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.source_frame = None
        self.detections: list[Detection] = []
        self.draft_bbox: tuple[int, int, int, int] | None = None
        self.drawing_start: tuple[int, int] | None = None
        self.on_box_created = None
        self.setMouseTracking(True)

    def set_box_created_callback(self, callback) -> None:
        self.on_box_created = callback

    def set_review_content(self, frame, detections: list[Detection], draft_bbox: tuple[int, int, int, int] | None = None) -> None:
        self.source_frame = None if frame is None else frame.copy()
        self.detections = detections
        self.draft_bbox = draft_bbox
        self._redraw()

    def _redraw(self) -> None:
        if self.source_frame is None:
            self.setText("Review preview")
            self.setPixmap(QPixmap())
            return

        display = draw_detections(self.source_frame, self.detections)
        if self.draft_bbox is not None:
            x1, y1, x2, y2 = self.draft_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (59, 130, 246), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def _widget_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        if self.source_frame is None:
            return None

        img_h, img_w = self.source_frame.shape[:2]
        label_w = max(1, self.width())
        label_h = max(1, self.height())
        scale = min(label_w / img_w, label_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        offset_x = (label_w - draw_w) / 2.0
        offset_y = (label_h - draw_h) / 2.0

        px = pos.x()
        py = pos.y()
        if px < offset_x or py < offset_y or px > offset_x + draw_w or py > offset_y + draw_h:
            return None

        img_x = int((px - offset_x) / scale)
        img_y = int((py - offset_y) / scale)
        img_x = max(0, min(img_w - 1, img_x))
        img_y = max(0, min(img_h - 1, img_y))
        return (img_x, img_y)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._redraw()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            point = self._widget_to_image(event.position().toPoint())
            if point is not None:
                self.drawing_start = point
                self.draft_bbox = None
                self._redraw()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.drawing_start is not None:
            point = self._widget_to_image(event.position().toPoint())
            if point is not None:
                x1, y1 = self.drawing_start
                x2, y2 = point
                self.draft_bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self._redraw()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self.drawing_start is not None:
            point = self._widget_to_image(event.position().toPoint())
            if point is not None:
                x1, y1 = self.drawing_start
                x2, y2 = point
                bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                if (bbox[2] - bbox[0]) >= 4 and (bbox[3] - bbox[1]) >= 4 and self.on_box_created is not None:
                    self.on_box_created(bbox)
            self.drawing_start = None
            self.draft_bbox = None
            self._redraw()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = AppPaths()
        self.paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.detector = Detector()
        self.capture = None
        self.current_frame = None
        self.current_detections: list[Detection] = []
        self.current_display_frame = None
        self.review_frame = None
        self.review_meta_path: Path | None = None
        self.active_process: QProcess | None = None
        self.active_process_name = ""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.setWindowTitle("Robot Vision Data Studio")
        self.resize(1360, 860)
        self.setMinimumSize(1024, 720)
        self._apply_styles()
        self._build_ui()
        self._create_menu()
        self._update_decision_banner()
        self.refresh_review_sessions()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f7fb;
                color: #162033;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #d7e0ea;
                border-radius: 14px;
                background: #f8fbff;
            }
            QTabBar::tab {
                background: #dfe8f3;
                color: #425069;
                padding: 10px 16px;
                margin-right: 6px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QTabBar::tab:selected {
                background: #f8fbff;
                color: #122033;
                font-weight: 600;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7e0ea;
                border-radius: 14px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #1f2f46;
            }
            QLineEdit, QComboBox, QTextEdit, QTableWidget {
                background: #fbfdff;
                border: 1px solid #ccd7e3;
                border-radius: 10px;
                padding: 7px 9px;
                selection-background-color: #2f6fed;
            }
            QComboBox::drop-down {
                border: 0;
                width: 24px;
            }
            QPushButton {
                background: #1d4ed8;
                color: white;
                border: 0;
                border-radius: 10px;
                padding: 9px 14px;
                min-height: 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1e40af;
            }
            QPushButton:pressed {
                background: #1a388f;
            }
            QHeaderView::section {
                background: #eaf0f6;
                color: #334155;
                border: 0;
                border-bottom: 1px solid #d7e0ea;
                padding: 8px;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #e2e8f0;
            }
            QSlider::groove:horizontal {
                background: #dbe4f0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1d4ed8;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            """
        )

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_live_tab(), "Live Capture")
        tabs.addTab(self._build_review_tab(), "Review / Label")
        tabs.addTab(self._build_train_tab(), "Train / Evaluate")
        layout.addWidget(tabs)

    def _build_live_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(380)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(self._build_source_box())
        left.addStretch(1)

        self.video_label = QLabel("Camera preview")
        self.video_label.setMinimumSize(420, 260)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #0f172a; color: #e2e8f0; border-radius: 18px; padding: 12px;"
        )
        self.decision_label = QLabel("Decision: clear")
        self.decision_label.setAlignment(Qt.AlignCenter)
        self.decision_label.setMinimumHeight(56)

        center_panel = QWidget()
        center = QVBoxLayout(center_panel)
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(12)
        center.addWidget(self.decision_label)
        center.addWidget(self.video_label, 1)

        right_panel = QWidget()
        right_panel.setMinimumWidth(320)
        right_panel.setMaximumWidth(420)
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        right.addWidget(self._build_capture_box())
        right.addWidget(self._build_notes_box(), 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setOpaqueResize(True)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([320, 820, 340])

        layout.addWidget(splitter, 1)
        return tab

    def _build_review_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        left.addWidget(self._build_review_browser_box())
        left.addWidget(self._build_review_meta_box())
        left.addStretch(1)

        self.review_image_label = ReviewImageLabel()
        self.review_image_label.setMinimumSize(420, 260)
        self.review_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.review_image_label.setAlignment(Qt.AlignCenter)
        self.review_image_label.setStyleSheet(
            "background-color: #111827; color: #e5e7eb; border-radius: 18px; padding: 12px;"
        )
        self.review_image_label.set_box_created_callback(self._handle_review_box_created)

        center_panel = QWidget()
        center = QVBoxLayout(center_panel)
        center.setContentsMargins(0, 0, 0, 0)
        center.addWidget(self.review_image_label, 1)

        right_panel = QWidget()
        right_panel.setMinimumWidth(360)
        right_panel.setMaximumWidth(480)
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self._build_review_table_box(), 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setOpaqueResize(True)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 760, 390])

        layout.addWidget(splitter, 1)
        return tab

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Live Test")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.addItems(["0", "1", "2", "3"])
        refresh_cameras_button = QPushButton("Refresh")
        refresh_cameras_button.clicked.connect(self.refresh_cameras)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText("Select a model from models/ or paste a path (pt/onnx)")
        refresh_models_button = QPushButton("Refresh List")
        refresh_models_button.clicked.connect(self.refresh_models)
        browse_model_button = QPushButton("Choose File")
        browse_model_button.clicked.connect(self.choose_model_file)
        model_button = QPushButton("Load Model")
        model_button.clicked.connect(self.load_model)

        self.loaded_model_label = QLabel("Loaded: none")
        self.live_status_label = QLabel("Camera stopped")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(25)
        self.confidence_value = QLabel("0.25")
        self.confidence_slider.valueChanged.connect(self._update_conf_label)

        start_button = QPushButton("Start Camera")
        start_button.clicked.connect(self.start_camera)
        stop_button = QPushButton("Stop Camera")
        stop_button.clicked.connect(self.stop_camera)

        camera_row = QHBoxLayout()
        camera_row.addWidget(self.camera_combo)
        camera_row.addWidget(refresh_cameras_button)

        model_row = QGridLayout()
        model_row.setHorizontalSpacing(8)
        model_row.setVerticalSpacing(8)
        model_row.addWidget(self.model_combo, 0, 0, 1, 3)
        model_row.addWidget(refresh_models_button, 1, 0)
        model_row.addWidget(browse_model_button, 1, 1)
        model_row.addWidget(model_button, 1, 2)

        live_row = QHBoxLayout()
        live_row.addWidget(start_button)
        live_row.addWidget(stop_button)

        conf_row = QHBoxLayout()
        conf_row.addWidget(self.confidence_slider)
        conf_row.addWidget(self.confidence_value)

        form.addRow("Camera", camera_row)
        form.addRow("Model", model_row)
        form.addRow("", self.loaded_model_label)
        form.addRow("Status", self.live_status_label)
        form.addRow("Confidence", conf_row)
        form.addRow("", live_row)

        self.refresh_models()
        return box

    def _build_train_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setOpaqueResize(True)

        controls_panel = QWidget()
        controls_panel.setMinimumWidth(360)
        controls_panel.setMaximumWidth(460)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)
        controls_layout.addWidget(self._build_export_box())
        controls_layout.addWidget(self._build_training_box())
        controls_layout.addWidget(self._build_run_box())
        controls_layout.addStretch(1)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(12)
        log_layout.addWidget(self._build_train_log_box(), 1)

        splitter.addWidget(controls_panel)
        splitter.addWidget(log_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 920])

        layout.addWidget(splitter, 1)
        return tab

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("Dataset Export")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.export_raw_input = QLineEdit("data/raw")
        self.export_out_input = QLineEdit("datasets/robot_obstacle")
        self.export_train_split = QDoubleSpinBox()
        self.export_train_split.setRange(0.0, 1.0)
        self.export_train_split.setSingleStep(0.05)
        self.export_train_split.setValue(0.8)
        self.export_val_split = QDoubleSpinBox()
        self.export_val_split.setRange(0.0, 1.0)
        self.export_val_split.setSingleStep(0.05)
        self.export_val_split.setValue(0.1)
        self.export_test_split = QDoubleSpinBox()
        self.export_test_split.setRange(0.0, 1.0)
        self.export_test_split.setSingleStep(0.05)
        self.export_test_split.setValue(0.1)
        self.export_min_conf = QDoubleSpinBox()
        self.export_min_conf.setRange(0.0, 1.0)
        self.export_min_conf.setSingleStep(0.05)
        self.export_min_conf.setValue(0.0)
        self.export_clear_out = QCheckBox("Replace previous export before writing")
        self.export_status_label = QLabel("Ready to export dataset")
        self.export_status_label.setWordWrap(True)
        export_button = QPushButton("Export Dataset")
        export_button.clicked.connect(self.export_dataset)
        self.export_button = export_button

        split_row = QGridLayout()
        split_row.setHorizontalSpacing(8)
        split_row.addWidget(QLabel("Train"), 0, 0)
        split_row.addWidget(self.export_train_split, 0, 1)
        split_row.addWidget(QLabel("Val"), 0, 2)
        split_row.addWidget(self.export_val_split, 0, 3)
        split_row.addWidget(QLabel("Test"), 0, 4)
        split_row.addWidget(self.export_test_split, 0, 5)

        form.addRow("Raw Data", self.export_raw_input)
        form.addRow("Output Dir", self.export_out_input)
        form.addRow("Split", split_row)
        form.addRow("Min Confidence", self.export_min_conf)
        form.addRow("", self.export_clear_out)
        form.addRow("Status", self.export_status_label)
        form.addRow("", export_button)
        return box

    def _build_training_box(self) -> QGroupBox:
        box = QGroupBox("Training")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.train_data_input = QLineEdit("configs/dataset.yaml")
        self.train_model_input = QLineEdit("models/yolo11n.pt")
        self.train_epochs_input = QSpinBox()
        self.train_epochs_input.setRange(1, 1000)
        self.train_epochs_input.setValue(50)
        self.train_imgsz_input = QSpinBox()
        self.train_imgsz_input.setRange(64, 4096)
        self.train_imgsz_input.setSingleStep(32)
        self.train_imgsz_input.setValue(640)
        self.train_batch_input = QSpinBox()
        self.train_batch_input.setRange(1, 512)
        self.train_batch_input.setValue(16)
        self.train_project_input = QLineEdit("runs/train")
        self.train_name_input = QLineEdit("robot_obstacle")
        self.train_status_label = QLabel("Ready to train")
        self.train_status_label.setWordWrap(True)
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)

        form.addRow("Dataset YAML", self.train_data_input)
        form.addRow("Base Model", self.train_model_input)
        form.addRow("Epochs", self.train_epochs_input)
        form.addRow("Image Size", self.train_imgsz_input)
        form.addRow("Batch", self.train_batch_input)
        form.addRow("Project Dir", self.train_project_input)
        form.addRow("Run Name", self.train_name_input)
        form.addRow("Status", self.train_status_label)
        form.addRow("", self.train_button)
        return box

    def _build_run_box(self) -> QGroupBox:
        box = QGroupBox("Run Control")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self.process_status_label = QLabel("No active task")
        self.process_status_label.setWordWrap(True)
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self._clear_train_log)
        self.stop_process_button = QPushButton("Stop Task")
        self.stop_process_button.clicked.connect(self.stop_active_process)
        self.stop_process_button.setEnabled(False)

        row = QGridLayout()
        row.setHorizontalSpacing(8)
        row.setVerticalSpacing(8)
        row.addWidget(clear_button, 0, 0)
        row.addWidget(self.stop_process_button, 0, 1)

        layout.addWidget(self.process_status_label)
        layout.addLayout(row)
        return box

    def _build_train_log_box(self) -> QGroupBox:
        box = QGroupBox("Run Log")
        layout = QVBoxLayout(box)
        self.train_log_output = QTextEdit()
        self.train_log_output.setReadOnly(True)
        self.train_log_output.setPlaceholderText("Export and training logs will appear here.")
        layout.addWidget(self.train_log_output, 1)
        return box

    def _build_capture_box(self) -> QGroupBox:
        box = QGroupBox("Collection")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.session_input = QLineEdit("field_session_001")
        self.source_name_input = QLineEdit("robot_webcam_front")
        self.scene_tag_input = QComboBox()
        self.scene_tag_input.addItems(DEFAULT_DECISIONS)

        capture_button = QPushButton("Capture Frame")
        capture_button.clicked.connect(self.capture_frame)

        form.addRow("Session", self.session_input)
        form.addRow("Camera Name", self.source_name_input)
        form.addRow("Scene Tag", self.scene_tag_input)
        form.addRow("", capture_button)
        return box

    def _build_notes_box(self) -> QGroupBox:
        box = QGroupBox("Operator Notes")
        layout = QVBoxLayout(box)
        self.notes_input = QTextEdit()
        self.notes_input.setMinimumHeight(220)
        self.notes_input.setPlaceholderText("Example: person crossing from left to right, hallway narrow, low light")
        layout.addWidget(self.notes_input)
        return box

    def _build_review_browser_box(self) -> QGroupBox:
        box = QGroupBox("Review Browser")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.review_session_combo = QComboBox()
        self.review_session_combo.currentTextChanged.connect(self.refresh_review_captures)
        self.review_capture_combo = QComboBox()

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_review_sessions)
        load_button = QPushButton("Load Review")
        load_button.clicked.connect(self.load_review_capture)
        auto_button = QPushButton("Auto Label")
        auto_button.clicked.connect(self.auto_label_review_capture)

        row = QGridLayout()
        row.setHorizontalSpacing(8)
        row.setVerticalSpacing(8)
        row.addWidget(refresh_button, 0, 0)
        row.addWidget(load_button, 0, 1)
        row.addWidget(auto_button, 1, 0, 1, 2)

        form.addRow("Session", self.review_session_combo)
        form.addRow("Capture", self.review_capture_combo)
        form.addRow("", row)
        return box

    def _build_review_meta_box(self) -> QGroupBox:
        box = QGroupBox("Review Metadata")
        form = QFormLayout(box)
        form.setSpacing(10)

        self.review_scene_tag_combo = QComboBox()
        self.review_scene_tag_combo.addItems(DEFAULT_DECISIONS)
        self.review_new_label_combo = QComboBox()
        self.review_new_label_combo.addItems(DEFAULT_CLASSES)
        self.review_note_input = QTextEdit()
        self.review_note_input.setMinimumHeight(180)
        self.review_note_input.setPlaceholderText("Notes for corrected sample")
        self.review_status_label = QLabel("No capture loaded")
        self.review_status_label.setWordWrap(True)

        form.addRow("Scene Tag", self.review_scene_tag_combo)
        form.addRow("Draw Label", self.review_new_label_combo)
        form.addRow("Notes", self.review_note_input)
        form.addRow("Status", self.review_status_label)
        return box

    def _build_review_table_box(self) -> QGroupBox:
        box = QGroupBox("Label Editor")
        layout = QVBoxLayout(box)

        self.review_table = QTableWidget(0, len(TABLE_HEADERS))
        self.review_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.review_table.setAlternatingRowColors(True)
        header = self.review_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for col in range(2, len(TABLE_HEADERS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.review_table.itemChanged.connect(self._review_table_changed)

        button_row = QGridLayout()
        button_row.setHorizontalSpacing(8)
        button_row.setVerticalSpacing(8)
        add_button = QPushButton("Add Box")
        add_button.clicked.connect(self.add_review_row)
        remove_button = QPushButton("Remove Row")
        remove_button.clicked.connect(self.remove_review_row)
        save_button = QPushButton("Save Labels")
        save_button.clicked.connect(self.save_review_labels)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.load_review_capture)
        button_row.addWidget(add_button, 0, 0)
        button_row.addWidget(remove_button, 0, 1)
        button_row.addWidget(reload_button, 1, 0)
        button_row.addWidget(save_button, 1, 1)

        layout.addWidget(self.review_table, 1)
        layout.addLayout(button_row)
        return box

    def _create_menu(self) -> None:
        menu = self.menuBar().addMenu("File")

        choose_model = QAction("Choose Model File", self)
        choose_model.triggered.connect(self.choose_model_file)
        menu.addAction(choose_model)

        refresh = QAction("Refresh Cameras/Models", self)
        refresh.triggered.connect(self._refresh_all)
        menu.addAction(refresh)

        open_data = QAction("Open Raw Data Folder", self)
        open_data.triggered.connect(self.show_data_folder)
        menu.addAction(open_data)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def _update_conf_label(self) -> None:
        self.confidence_value.setText(f"{self.confidence_slider.value() / 100:.2f}")

    def _set_status(self, text: str) -> None:
        self.live_status_label.setText(text)
        self.statusBar().showMessage(text, 4000)

    def _set_review_status(self, text: str) -> None:
        self.review_status_label.setText(text)
        self.statusBar().showMessage(text, 4000)

    def _set_process_status(self, text: str) -> None:
        self.process_status_label.setText(text)
        self.statusBar().showMessage(text, 4000)

    def _append_train_log(self, text: str) -> None:
        cleaned = text.rstrip()
        if not cleaned:
            return
        self.train_log_output.append(cleaned)

    def _clear_train_log(self) -> None:
        self.train_log_output.clear()

    def _set_task_buttons_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled)
        self.train_button.setEnabled(enabled)
        self.stop_process_button.setEnabled(not enabled)

    def _resolve_existing_path(self, raw: str, label: str) -> Path | None:
        path = self.paths.resolve_in_project(raw.strip())
        if path.exists():
            return path
        QMessageBox.warning(self, "Path Not Found", f"{label} not found:\n{path}")
        return None

    def _start_process(self, title: str, script_rel: str, args: list[str]) -> bool:
        if self.active_process is not None:
            QMessageBox.information(self, "Task Running", "Please wait for the current task to finish or stop it first.")
            return False

        process = QProcess(self)
        process.setProgram(sys.executable or "python3")
        process.setArguments([str(self.paths.root / script_rel), *args])
        process.setWorkingDirectory(str(self.paths.root))
        process.setProcessChannelMode(QProcess.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_process_stdout)
        process.readyReadStandardError.connect(self._read_process_stderr)
        process.finished.connect(self._process_finished)

        self.active_process = process
        self.active_process_name = title
        self._set_task_buttons_enabled(False)
        self._set_process_status(f"{title} is running")
        self._append_train_log(f"$ {process.program()} {' '.join(process.arguments())}")
        process.start()
        if not process.waitForStarted(3000):
            self._append_train_log(f"{title} failed to start.")
            self._set_process_status(f"{title} failed to start")
            self.active_process = None
            self.active_process_name = ""
            self._set_task_buttons_enabled(True)
            return False
        return True

    def _read_process_stdout(self) -> None:
        if self.active_process is None:
            return
        data = bytes(self.active_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_train_log(data)

    def _read_process_stderr(self) -> None:
        if self.active_process is None:
            return
        data = bytes(self.active_process.readAllStandardError()).decode("utf-8", errors="replace")
        self._append_train_log(data)

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        task_name = self.active_process_name or "Task"
        ok = exit_code == 0
        message = f"{task_name} completed successfully." if ok else f"{task_name} failed with exit code {exit_code}."
        self._append_train_log(message)
        self._set_process_status(message)
        if "Export" in task_name:
            self.export_status_label.setText(message)
        elif "Training" in task_name:
            self.train_status_label.setText(message)
        self.active_process = None
        self.active_process_name = ""
        self._set_task_buttons_enabled(True)

    def export_dataset(self) -> None:
        raw_dir = self._resolve_existing_path(self.export_raw_input.text(), "Raw data directory")
        if raw_dir is None:
            return
        out_dir = self.paths.resolve_in_project(self.export_out_input.text().strip())
        args = [
            "--raw",
            str(raw_dir),
            "--out",
            str(out_dir),
            "--train",
            str(self.export_train_split.value()),
            "--val",
            str(self.export_val_split.value()),
            "--test",
            str(self.export_test_split.value()),
            "--min-conf",
            str(self.export_min_conf.value()),
        ]
        if self.export_clear_out.isChecked():
            args.append("--clear-out")
        self.export_status_label.setText("Export started...")
        self._start_process("Dataset Export", "scripts/export_yolo_dataset.py", args)

    def start_training(self) -> None:
        data_path = self._resolve_existing_path(self.train_data_input.text(), "Dataset YAML")
        if data_path is None:
            return
        model_path = self._resolve_existing_path(self.train_model_input.text(), "Base model")
        if model_path is None:
            return
        project_dir = self.paths.resolve_in_project(self.train_project_input.text().strip())
        args = [
            "--data",
            str(data_path),
            "--model",
            str(model_path),
            "--epochs",
            str(self.train_epochs_input.value()),
            "--imgsz",
            str(self.train_imgsz_input.value()),
            "--project",
            str(project_dir),
            "--name",
            self.train_name_input.text().strip() or "robot_obstacle",
            "--batch",
            str(self.train_batch_input.value()),
        ]
        self.train_status_label.setText("Training started...")
        self._start_process("Training", "scripts/train.py", args)

    def stop_active_process(self) -> None:
        if self.active_process is None:
            return
        task_name = self.active_process_name or "Task"
        self._append_train_log(f"Stopping {task_name}...")
        self.active_process.kill()
        self._set_process_status(f"Stopping {task_name}...")

    def show_about_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("About Robot Vision Data Studio")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        title = QLabel("Robot Vision Data Studio")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #122033;")
        version = QLabel("Version 1.0")
        version.setStyleSheet("color: #475569; font-size: 14px;")
        description = QLabel(
            "Desktop toolkit for collecting robot-vision data, reviewing labels, and training YOLO models in one place."
        )
        description.setWordWrap(True)
        developer = QLabel(
            "Developer: Nakarin Sripanya\n"
            "Electrical Engineering Program\n"
            "Faculty of Industry and Technology\n"
            "Rajamangala University of Technology Isan"
        )
        developer.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)

        layout.addWidget(title)
        layout.addWidget(version)
        layout.addWidget(description)
        layout.addWidget(developer)
        layout.addWidget(buttons)
        dialog.exec()

    def _resolve_model_path(self, raw: str) -> Path:
        return self.paths.resolve_in_project(raw)

    def _current_decision(self) -> str:
        if any(det.label == "person" for det in self.current_detections):
            return "stop_wait"
        if self.current_detections:
            return "bypass_candidate"
        return "clear"

    def _update_decision_banner(self) -> None:
        decision = self._current_decision()
        labels = {
            "stop_wait": "Stop / Wait",
            "bypass_candidate": "Bypass Candidate",
            "clear": "Path Clear",
        }
        colors = {
            "stop_wait": ("#fee2e2", "#991b1b"),
            "bypass_candidate": ("#ffedd5", "#9a3412"),
            "clear": ("#d1fae5", "#065f46"),
        }
        bg, fg = colors.get(decision, ("#e2e8f0", "#0f172a"))
        self.decision_label.setText(f"Decision: {labels.get(decision, decision)}")
        self.decision_label.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {fg}; background: {bg}; "
            "padding: 12px 16px; border-radius: 14px; border: 1px solid rgba(15, 23, 42, 0.08);"
        )

    def _refresh_all(self) -> None:
        self.refresh_cameras()
        self.refresh_models()
        self.refresh_review_sessions()

    def refresh_models(self) -> None:
        current = self.model_combo.currentText().strip()
        self.model_combo.clear()
        for model_path in _list_model_files(self.paths):
            self.model_combo.addItem(model_path)
        if current:
            self.model_combo.setEditText(current)

    def refresh_cameras(self) -> None:
        current = self.camera_combo.currentText().strip()
        self.camera_combo.clear()
        cameras = _scan_cameras(max_index=8)
        if not cameras:
            self.camera_combo.addItem("0")
        else:
            for idx in cameras:
                self.camera_combo.addItem(str(idx))
        if current:
            self.camera_combo.setEditText(current)

    def choose_model_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose model", str(self.paths.root), "Model Files (*.pt *.onnx)")
        if file_path:
            try:
                rel = str(Path(file_path).resolve().relative_to(self.paths.root))
                self.model_combo.setEditText(rel)
            except Exception:
                self.model_combo.setEditText(file_path)

    def load_model(self) -> None:
        raw = self.model_combo.currentText().strip()
        if not raw:
            QMessageBox.warning(self, "Model", "Select a model file first.")
            return
        model_path = self._resolve_model_path(raw)
        if not model_path.exists():
            QMessageBox.warning(self, "Model", f"Model file not found:\n{model_path}")
            return
        try:
            self.detector.load(str(model_path))
        except Exception as exc:
            QMessageBox.critical(self, "Model Error", str(exc))
            return
        self.loaded_model_label.setText(f"Loaded: {self.detector.model_name}")
        self._set_status("Model loaded")

    def start_camera(self) -> None:
        self.stop_camera()
        camera_id_text = self.camera_combo.currentText().strip()
        camera_id = int(camera_id_text) if camera_id_text.isdigit() else camera_id_text
        self.capture = cv2.VideoCapture(camera_id)
        if not self.capture.isOpened():
            QMessageBox.critical(self, "Camera Error", f"Could not open camera source: {camera_id_text}")
            self.capture = None
            self._set_status("Camera open failed")
            return
        self.timer.start(30)
        self._set_status(f"Camera running: {camera_id_text}")

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._set_status("Camera stopped")

    def update_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self._set_status("Camera read failed")
            return
        self.current_frame = frame
        conf = self.confidence_slider.value() / 100.0
        self.current_detections = self.detector.predict(frame, conf=conf)
        self._update_decision_banner()
        self.current_display_frame = draw_detections(frame, self.current_detections)
        self.video_label.setPixmap(self._to_pixmap(self.current_display_frame, self.video_label))
        self.video_label.setText("")

    def _to_pixmap(self, frame, target: QLabel) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image).scaled(
            target.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def capture_frame(self) -> None:
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "Start the camera first.")
            return

        session_name = sanitize_session_name(self.session_input.text().strip() or "session")
        self.session_input.setText(session_name)
        source_name = self.source_name_input.text().strip() or "webcam"
        scene_tag = self.scene_tag_input.currentText()
        note = self.notes_input.toPlainText().strip()
        conf = self.confidence_slider.value() / 100.0
        detections = [
            {
                "label": det.label,
                "confidence": det.confidence,
                "bbox": list(det.bbox),
                "decision_hint": det.decision_hint,
            }
            for det in self.current_detections
        ]

        try:
            record = save_capture(
                base_dir=self.paths.raw_data_dir,
                session_name=session_name,
                frame=self.current_frame,
                source_name=source_name,
                scene_tag=scene_tag,
                operator_note=note,
                model_name=self.detector.model_name,
                confidence_threshold=conf,
                detections=detections,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._set_status(f"Saved: {record.image_path}")
        self.refresh_review_sessions()
        self.review_session_combo.setCurrentText(record.session_name)
        self.refresh_review_captures()
        if self.review_capture_combo.count():
            self.review_capture_combo.setCurrentIndex(self.review_capture_combo.count() - 1)
        QMessageBox.information(self, "Saved", f"Saved {record.image_path}")

    def refresh_review_sessions(self) -> None:
        current = self.review_session_combo.currentText().strip()
        self.review_session_combo.blockSignals(True)
        self.review_session_combo.clear()
        for session_name in list_sessions(self.paths.raw_data_dir):
            self.review_session_combo.addItem(session_name)
        self.review_session_combo.blockSignals(False)
        if current:
            self.review_session_combo.setCurrentText(current)
        if self.review_session_combo.count() and not self.review_capture_combo.count():
            self.refresh_review_captures()

    def refresh_review_captures(self) -> None:
        session_name = self.review_session_combo.currentText().strip()
        current = self.review_capture_combo.currentText().strip()
        self.review_capture_combo.clear()
        if not session_name:
            return
        for meta_path in list_capture_meta(self.paths.raw_data_dir, session_name):
            self.review_capture_combo.addItem(meta_path.stem, str(meta_path))
        if current:
            self.review_capture_combo.setCurrentText(current)
        elif self.review_capture_combo.count():
            self.review_capture_combo.setCurrentIndex(self.review_capture_combo.count() - 1)

    def _review_meta(self) -> dict | None:
        if self.review_meta_path is None:
            return None
        return load_capture_meta(self.review_meta_path)

    def _preferred_review_detections(self, meta: dict) -> list[dict]:
        corrected = meta.get("corrected_detections")
        if isinstance(corrected, list) and (meta.get("corrected_at") or corrected):
            return corrected
        detections = meta.get("detections")
        if isinstance(detections, list):
            return detections
        return []

    def load_review_capture(self) -> None:
        data = self.review_capture_combo.currentData()
        if not data:
            QMessageBox.warning(self, "Review", "Select a capture first.")
            return
        self.review_meta_path = Path(str(data))
        meta = load_capture_meta(self.review_meta_path)
        image_path = self.paths.raw_data_dir / meta["image_path"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            QMessageBox.critical(self, "Review", f"Image not found or unreadable:\n{image_path}")
            return
        self.review_frame = frame
        self.review_scene_tag_combo.setCurrentText(str(meta.get("scene_tag", "clear")))
        self.review_note_input.setPlainText(str(meta.get("operator_note", "")))
        self._set_review_table(self._preferred_review_detections(meta))
        self._refresh_review_preview()
        self._set_review_status(f"Loaded: {self.review_meta_path.name}")

    def _set_review_table(self, detections: list[dict]) -> None:
        self.review_table.blockSignals(True)
        self.review_table.setRowCount(0)
        for det in detections:
            self._append_review_row(det)
        self.review_table.blockSignals(False)

    def _append_review_row(self, det: dict) -> None:
        row = self.review_table.rowCount()
        self.review_table.insertRow(row)
        bbox = det.get("bbox", [0, 0, 1, 1])
        values = [
            str(det.get("label", "obstacle")),
            f"{float(det.get('confidence', 1.0)):.2f}",
            str(int(bbox[0])),
            str(int(bbox[1])),
            str(int(bbox[2])),
            str(int(bbox[3])),
        ]
        for col, value in enumerate(values):
            self.review_table.setItem(row, col, QTableWidgetItem(value))

    def _review_table_detections(self) -> list[dict]:
        detections: list[dict] = []
        for row in range(self.review_table.rowCount()):
            try:
                label = self.review_table.item(row, 0).text().strip() or "obstacle"
                confidence = float(self.review_table.item(row, 1).text().strip())
                x1 = int(float(self.review_table.item(row, 2).text().strip()))
                y1 = int(float(self.review_table.item(row, 3).text().strip()))
                x2 = int(float(self.review_table.item(row, 4).text().strip()))
                y2 = int(float(self.review_table.item(row, 5).text().strip()))
            except Exception:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "decision_hint": "stop_wait" if label == "person" else "bypass_candidate",
                }
            )
        return detections

    def _refresh_review_preview(self) -> None:
        detections = [_detection_from_dict(item) for item in self._review_table_detections()]
        self.review_image_label.set_review_content(self.review_frame, detections)

    def _review_table_changed(self, _item=None) -> None:
        self._refresh_review_preview()

    def _handle_review_box_created(self, bbox: tuple[int, int, int, int]) -> None:
        label = self.review_new_label_combo.currentText().strip() or "obstacle"
        self._append_review_row(
            {
                "label": label,
                "confidence": 1.0,
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "decision_hint": "stop_wait" if label == "person" else "bypass_candidate",
            }
        )
        self._refresh_review_preview()
        self._set_review_status("Added box from mouse drag")

    def add_review_row(self) -> None:
        self._append_review_row({"label": "obstacle", "confidence": 1.0, "bbox": [10, 10, 100, 100]})
        self._refresh_review_preview()

    def remove_review_row(self) -> None:
        selected_rows = sorted({index.row() for index in self.review_table.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return
        for row in selected_rows:
            self.review_table.removeRow(row)
        self._refresh_review_preview()

    def auto_label_review_capture(self) -> None:
        if self.review_frame is None:
            QMessageBox.warning(self, "Review", "Load a capture first.")
            return
        if self.detector.model is None:
            QMessageBox.warning(self, "Review", "Load a model first.")
            return
        conf = self.confidence_slider.value() / 100.0
        detections = self.detector.predict(self.review_frame, conf=conf)
        self._set_review_table(
            [
                {
                    "label": det.label,
                    "confidence": det.confidence,
                    "bbox": list(det.bbox),
                    "decision_hint": det.decision_hint,
                }
                for det in detections
            ]
        )
        self._refresh_review_preview()
        self._set_review_status("Auto labels applied from model")

    def save_review_labels(self) -> None:
        if self.review_meta_path is None:
            QMessageBox.warning(self, "Review", "Load a capture first.")
            return
        meta = load_capture_meta(self.review_meta_path)
        meta["scene_tag"] = self.review_scene_tag_combo.currentText()
        meta["operator_note"] = self.review_note_input.toPlainText().strip()
        meta["corrected_detections"] = self._review_table_detections()
        meta["corrected_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        save_capture_meta(self.review_meta_path, meta)
        self._set_review_status(f"Saved corrections: {self.review_meta_path.name}")
        QMessageBox.information(self, "Review", "Saved corrected labels.")

    def show_data_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.raw_data_dir)))

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self.current_display_frame is not None and self.video_label.width() > 0 and self.video_label.height() > 0:
            self.video_label.setPixmap(self._to_pixmap(self.current_display_frame, self.video_label))
            self.video_label.setText("")
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.active_process is not None:
            self.active_process.kill()
            self.active_process.waitForFinished(2000)
        self.stop_camera()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True, description="Robot Vision Data Studio GUI")
    parser.add_argument(
        "--platform",
        choices=["xcb", "wayland", "offscreen", "minimal"],
        default=None,
        help="Override Qt platform plugin (sets QT_QPA_PLATFORM).",
    )
    args, qt_args = parser.parse_known_args(sys.argv[1:])

    if args.platform and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = args.platform

    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        platform = os.environ.get("QT_QPA_PLATFORM")
        if not has_display and platform not in {"offscreen", "minimal"}:
            sys.stderr.write(
                "No graphical session detected (DISPLAY/WAYLAND_DISPLAY is empty).\n"
                "If you are on a headless Ubuntu server/SSH session, the GUI cannot open.\n"
                "Try one of:\n"
                "- run on a desktop session\n"
                "- use X11 forwarding (ssh -X) and ensure DISPLAY is set\n"
                "- run diagnostics: python scripts/diagnose_gui.py\n"
                "- for CI/headless only: QT_QPA_PLATFORM=offscreen python -m app.gui\n"
            )
            return 2

    app = QApplication([sys.argv[0], *qt_args])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
