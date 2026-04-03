from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import AppPaths, DEFAULT_DECISIONS, sanitize_session_name
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
        self.review_frame = None
        self.review_meta_path: Path | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.setWindowTitle("Robot Vision Data Studio")
        self.resize(1440, 900)
        self._build_ui()
        self._create_menu()
        self._update_decision_banner()
        self.refresh_review_sessions()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)

        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "Live Capture")
        tabs.addTab(self._build_review_tab(), "Review / Label")
        layout.addWidget(tabs)

    def _build_live_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        left = QVBoxLayout()
        left.addWidget(self._build_source_box())
        left.addStretch(1)

        self.video_label = QLabel("Camera preview")
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #101820; color: #f1f5f9;")
        self.decision_label = QLabel("Decision: clear")
        self.decision_label.setAlignment(Qt.AlignCenter)

        center = QVBoxLayout()
        center.addWidget(self.decision_label)
        center.addWidget(self.video_label, 1)

        right = QVBoxLayout()
        right.addWidget(self._build_capture_box())
        right.addWidget(self._build_notes_box(), 1)

        layout.addLayout(left, 0)
        layout.addLayout(center, 1)
        layout.addLayout(right, 0)
        return tab

    def _build_review_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        left = QVBoxLayout()
        left.addWidget(self._build_review_browser_box())
        left.addWidget(self._build_review_meta_box())
        left.addStretch(1)

        self.review_image_label = QLabel("Review preview")
        self.review_image_label.setMinimumSize(960, 540)
        self.review_image_label.setAlignment(Qt.AlignCenter)
        self.review_image_label.setStyleSheet("background-color: #111827; color: #e5e7eb;")

        center = QVBoxLayout()
        center.addWidget(self.review_image_label, 1)

        right = QVBoxLayout()
        right.addWidget(self._build_review_table_box(), 1)

        layout.addLayout(left, 0)
        layout.addLayout(center, 1)
        layout.addLayout(right, 0)
        return tab

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Live Test")
        form = QFormLayout(box)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.camera_combo.addItems(["0", "1", "2", "3"])
        refresh_cameras_button = QPushButton("Refresh")
        refresh_cameras_button.clicked.connect(self.refresh_cameras)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setPlaceholderText("Select a model from models/ or paste a path (pt/onnx)")
        refresh_models_button = QPushButton("Refresh")
        refresh_models_button.clicked.connect(self.refresh_models)
        browse_model_button = QPushButton("Browse")
        browse_model_button.clicked.connect(self.choose_model_file)
        model_button = QPushButton("Load")
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

        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo)
        model_row.addWidget(refresh_models_button)
        model_row.addWidget(browse_model_button)
        model_row.addWidget(model_button)

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

    def _build_capture_box(self) -> QGroupBox:
        box = QGroupBox("Collection")
        form = QFormLayout(box)

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
        self.notes_input.setPlaceholderText("Example: person crossing from left to right, hallway narrow, low light")
        layout.addWidget(self.notes_input)
        return box

    def _build_review_browser_box(self) -> QGroupBox:
        box = QGroupBox("Review Browser")
        form = QFormLayout(box)

        self.review_session_combo = QComboBox()
        self.review_session_combo.currentTextChanged.connect(self.refresh_review_captures)
        self.review_capture_combo = QComboBox()

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_review_sessions)
        load_button = QPushButton("Load Capture")
        load_button.clicked.connect(self.load_review_capture)
        auto_button = QPushButton("Auto From Model")
        auto_button.clicked.connect(self.auto_label_review_capture)

        row = QHBoxLayout()
        row.addWidget(refresh_button)
        row.addWidget(load_button)
        row.addWidget(auto_button)

        form.addRow("Session", self.review_session_combo)
        form.addRow("Capture", self.review_capture_combo)
        form.addRow("", row)
        return box

    def _build_review_meta_box(self) -> QGroupBox:
        box = QGroupBox("Review Metadata")
        form = QFormLayout(box)

        self.review_scene_tag_combo = QComboBox()
        self.review_scene_tag_combo.addItems(DEFAULT_DECISIONS)
        self.review_note_input = QTextEdit()
        self.review_note_input.setPlaceholderText("Notes for corrected sample")
        self.review_status_label = QLabel("No capture loaded")

        form.addRow("Scene Tag", self.review_scene_tag_combo)
        form.addRow("Notes", self.review_note_input)
        form.addRow("Status", self.review_status_label)
        return box

    def _build_review_table_box(self) -> QGroupBox:
        box = QGroupBox("Label Editor")
        layout = QVBoxLayout(box)

        self.review_table = QTableWidget(0, len(TABLE_HEADERS))
        self.review_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.review_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.review_table.itemChanged.connect(self._review_table_changed)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add Box")
        add_button.clicked.connect(self.add_review_row)
        remove_button = QPushButton("Remove Row")
        remove_button.clicked.connect(self.remove_review_row)
        save_button = QPushButton("Save Labels")
        save_button.clicked.connect(self.save_review_labels)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.load_review_capture)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(reload_button)
        button_row.addWidget(save_button)

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

    def _update_conf_label(self) -> None:
        self.confidence_value.setText(f"{self.confidence_slider.value() / 100:.2f}")

    def _set_status(self, text: str) -> None:
        self.live_status_label.setText(text)

    def _set_review_status(self, text: str) -> None:
        self.review_status_label.setText(text)

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
        colors = {
            "stop_wait": ("#fee2e2", "#991b1b"),
            "bypass_candidate": ("#ffedd5", "#9a3412"),
            "clear": ("#d1fae5", "#065f46"),
        }
        bg, fg = colors.get(decision, ("#e2e8f0", "#0f172a"))
        self.decision_label.setText(f"Decision: {decision}")
        self.decision_label.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {fg}; background: {bg}; padding: 10px; border-radius: 8px;"
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
        display = draw_detections(frame, self.current_detections)
        self.video_label.setPixmap(self._to_pixmap(display, self.video_label))

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
        if "corrected_detections" in meta and isinstance(meta.get("corrected_detections"), list):
            detections = meta.get("corrected_detections") or []
        else:
            detections = meta.get("detections") or []
        self._set_review_table(detections)
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
        if self.review_frame is None:
            return
        detections = [_detection_from_dict(item) for item in self._review_table_detections()]
        display = draw_detections(self.review_frame, detections)
        self.review_image_label.setPixmap(self._to_pixmap(display, self.review_image_label))

    def _review_table_changed(self, _item=None) -> None:
        self._refresh_review_preview()

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

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_camera()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
