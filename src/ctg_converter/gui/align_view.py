"""Visual Tool 1: CTG Enhancement & 90-Degree Metric Grid Alignment Inspector."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ctg_converter.dataset.manifest import DatasetManifest, DatasetPaths
from ctg_converter.pipelines.enhance import EnhanceConfig, run_enhance_pipeline
from ctg_converter.pipelines.rectify import RectifyConfig, run_rectify_pipeline

from .tiled_canvas import TiledImageCanvas


class AlignEnhanceTab(QWidget):
    """Interactive visual tool for inspecting & tuning frame enhancement and 90-degree alignment."""

    status_message = Signal(str)
    frames_updated = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._paths: DatasetPaths | None = None
        self._manifest: DatasetManifest | None = None
        self._current_index: int = 1

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter)

        # Left panel: Frame list + Pipeline parameters + Action buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)

        grp_frames = QGroupBox("Einzelbilder im Datensatz")
        vf = QVBoxLayout(grp_frames)
        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self._on_frame_selected)
        vf.addWidget(self.frame_list)
        left_layout.addWidget(grp_frames, stretch=3)

        grp_params = QGroupBox("Vorverarbeitung & 90°-Gitter-Parameter")
        form = QFormLayout(grp_params)

        self.cmb_rotation = QComboBox()
        self.cmb_rotation.addItem("Automatisch (90°-Erkennung)", None)
        self.cmb_rotation.addItem("0° (Original)", 0)
        self.cmb_rotation.addItem("90° im Uhrzeigersinn", 90)
        self.cmb_rotation.addItem("180° gedreht", 180)
        self.cmb_rotation.addItem("270° im Uhrzeigersinn", 270)
        form.addRow("Grob-Orientierung:", self.cmb_rotation)

        self.spn_grid_gain = QDoubleSpinBox()
        self.spn_grid_gain.setRange(0.3, 3.0)
        self.spn_grid_gain.setSingleStep(0.1)
        self.spn_grid_gain.setValue(1.45)
        form.addRow("Grün-Gitter-Kontrast:", self.spn_grid_gain)

        self.spn_trace_gain = QDoubleSpinBox()
        self.spn_trace_gain.setRange(0.5, 3.0)
        self.spn_trace_gain.setSingleStep(0.1)
        self.spn_trace_gain.setValue(1.35)
        form.addRow("Schwarz-Kurven-Kontrast:", self.spn_trace_gain)

        self.spn_sharpen = QDoubleSpinBox()
        self.spn_sharpen.setRange(0.0, 1.5)
        self.spn_sharpen.setSingleStep(0.05)
        self.spn_sharpen.setValue(0.35)
        form.addRow("Detail-Schärfung:", self.spn_sharpen)

        self.spn_px_cm = QSpinBox()
        self.spn_px_cm.setRange(40, 200)
        self.spn_px_cm.setSingleStep(10)
        self.spn_px_cm.setValue(100)
        form.addRow("Zielmaßstab (px/cm):", self.spn_px_cm)

        left_layout.addWidget(grp_params)

        self.lbl_diag = QLabel("Keine Bilddiagnose geladen.")
        self.lbl_diag.setWordWrap(True)
        self.lbl_diag.setStyleSheet(
            "background: #23272e; color: #dcdcdc; padding: 8px; border-radius: 4px; font-family: monospace;"
        )
        left_layout.addWidget(self.lbl_diag)

        self.btn_run_single = QPushButton("Aktuelles Bild verbessern & 90°-ausrichten")
        self.btn_run_single.clicked.connect(self._run_current_frame)
        left_layout.addWidget(self.btn_run_single)

        self.btn_run_all = QPushButton("Alle Bilder im Datensatz ausrichten")
        self.btn_run_all.clicked.connect(self._run_all_frames)
        left_layout.addWidget(self.btn_run_all)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        splitter.addWidget(left_panel)

        # Right panel: View mode bar + Tiled canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Ansichts-Stufe:"))
        self.cmb_layer = QComboBox()
        self.cmb_layer.addItems(
            [
                "5. 90°-Entzerrt & Metrisch normiert (Ergebnis)",
                "2. Verbessert (Flachfeld & Kontrast)",
                "3. Isoliertes Grün-Gitter (Gitter-Maske)",
                "4. Isolierte Schwarz-Kurve (Spur-Maske)",
                "1. Rohbild (Unverändert)",
            ]
        )
        self.cmb_layer.currentIndexChanged.connect(lambda _: self._refresh_canvas(preserve_view=False))
        toolbar.addWidget(self.cmb_layer, stretch=2)

        self.chk_grid_overlay = QCheckBox("Ideales 90°-Kontrollgitter einblenden")
        self.chk_grid_overlay.setChecked(True)
        self.chk_grid_overlay.toggled.connect(lambda _: self._update_overlays())
        toolbar.addWidget(self.chk_grid_overlay)

        self.canvas = TiledImageCanvas()
        self.canvas.cursor_metric_changed.connect(self.status_message.emit)

        btn_zin = QPushButton("Zoom +")
        btn_zin.clicked.connect(lambda: self.canvas.zoom_by(1.25))
        btn_zout = QPushButton("Zoom -")
        btn_zout.clicked.connect(lambda: self.canvas.zoom_by(0.8))
        btn_fit = QPushButton("Einpassen")
        btn_fit.clicked.connect(self.canvas.fit_all)
        toolbar.addWidget(btn_zin)
        toolbar.addWidget(btn_zout)
        toolbar.addWidget(btn_fit)

        right_layout.addLayout(toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([360, 980])

    def load_dataset(self, paths: DatasetPaths) -> None:
        self._paths = paths
        self._manifest = DatasetManifest.load(paths.manifest_path)
        self.spn_px_cm.setValue(self._manifest.target_px_per_cm)

        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for entry in self._manifest.frames:
            stem = Path(entry.filename).stem
            rect_meta_p = paths.rectified_dir / f"{stem}_rect_meta.json"
            status_tag = "ausgerichtet" if rect_meta_p.exists() else "roh"
            item = QListWidgetItem(
                f"[{entry.index:02d}] {entry.filename}  ({entry.width}x{entry.height}) — {status_tag}"
            )
            self.frame_list.addItem(item)
        self.frame_list.blockSignals(False)

        if self.frame_list.count() > 0:
            self.frame_list.setCurrentRow(0)
            self._on_frame_selected(0)

    def _on_frame_selected(self, row: int) -> None:
        if self._manifest is None or row < 0 or row >= len(self._manifest.frames):
            return
        entry = self._manifest.frames[row]
        self._current_index = entry.index

        # Sync rotation override combo
        idx_rot = self.cmb_rotation.findData(entry.rotation_override_deg)
        self.cmb_rotation.setCurrentIndex(max(0, idx_rot))

        self._refresh_diagnostics(entry.filename)
        self._refresh_canvas(preserve_view=False)

    def _refresh_diagnostics(self, filename: str) -> None:
        if self._paths is None:
            return
        stem = Path(filename).stem
        enh_meta_p = self._paths.enhanced_dir / f"{stem}_meta.json"
        rect_meta_p = self._paths.rectified_dir / f"{stem}_rect_meta.json"

        lines = [f"Datei: {filename}"]
        if enh_meta_p.exists():
            em = json.loads(enh_meta_p.read_text(encoding="utf-8"))
            lines.append(f"90°-Drehung:   {em['applied_rotation_cw_deg']}° CW")
            lines.append(f"Gültiges Papier: {em['valid_paper_fraction']*100:.1f}%")
        if rect_meta_p.exists():
            rm = json.loads(rect_meta_p.read_text(encoding="utf-8"))
            lines.append(f"Entzerrte Größe: {rm['rectified_width']} x {rm['rectified_height']} px")
            lines.append(f"Schiefwinkel vorher: {rm['initial_skew_deg']:+.2f}°")
            lines.append(f"Gitterwinkel nachher: {rm['orthogonal_angle_deg']:.2f}° (Ziel 90.00°)")
            lines.append(
                f"Kästchen-Raster (2.5mm): X={rm['measured_grid_period_x_px']:.2f}px, Y={rm['measured_grid_period_y_px']:.2f}px"
            )
        else:
            lines.append("Status: Noch nicht 90°-entzerrt.")
        self.lbl_diag.setText("\n".join(lines))

    def _refresh_canvas(self, preserve_view: bool = False) -> None:
        if self._paths is None or self._manifest is None:
            return
        entry = self._manifest.frames[self._current_index - 1]
        stem = Path(entry.filename).stem
        mode_txt = self.cmb_layer.currentText()

        img = None
        is_rectified_view = False

        if mode_txt.startswith("5."):
            p = self._paths.rectified_dir / f"{stem}_rectified.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                is_rectified_view = True
        elif mode_txt.startswith("2."):
            p = self._paths.enhanced_dir / f"{stem}_enhanced.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        elif mode_txt.startswith("3."):
            p = self._paths.enhanced_dir / f"{stem}_grid.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        elif mode_txt.startswith("4."):
            p = self._paths.enhanced_dir / f"{stem}_trace.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raw_p = self._paths.raw_dir / entry.filename
            img = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)

        if img is None:
            return

        px_cm = self.spn_px_cm.value() if is_rectified_view else None
        self.canvas.set_numpy_image(
            img,
            px_per_cm=px_cm,
            paper_speed_cm_per_min=self._manifest.paper_speed_cm_per_min,
            preserve_view=preserve_view,
        )
        self._update_overlays()

    def _update_overlays(self) -> None:
        if self.cmb_layer.currentText().startswith("5.") and self.chk_grid_overlay.isChecked():
            self.canvas.draw_canonical_grid_overlay(self.spn_px_cm.value())
        else:
            self.canvas.clear_overlays()

    def _build_configs(self) -> tuple[EnhanceConfig, RectifyConfig]:
        ecfg = EnhanceConfig(
            grid_gain=float(self.spn_grid_gain.value()),
            trace_gain=float(self.spn_trace_gain.value()),
            sharpen_amount=float(self.spn_sharpen.value()),
        )
        rcfg = RectifyConfig(target_px_per_cm=int(self.spn_px_cm.value()))
        return ecfg, rcfg

    def _run_current_frame(self) -> None:
        if self._paths is None or self._manifest is None:
            return
        entry = self._manifest.frames[self._current_index - 1]
        entry.rotation_override_deg = self.cmb_rotation.currentData()
        self._manifest.target_px_per_cm = int(self.spn_px_cm.value())
        self._manifest.save(self._paths.manifest_path)

        ecfg, rcfg = self._build_configs()
        self.status_message.emit(f"Verarbeite Bild {entry.filename} ...")
        QApplication.processEvents()

        run_enhance_pipeline(self._paths, cfg=ecfg, frame_indices=[entry.index])
        run_rectify_pipeline(self._paths, rect_cfg=rcfg, enh_cfg=ecfg, frame_indices=[entry.index])

        row = self._current_index - 1
        self.frame_list.item(row).setText(
            f"[{entry.index:02d}] {entry.filename}  ({entry.width}x{entry.height}) — ausgerichtet"
        )
        self._refresh_diagnostics(entry.filename)
        self._refresh_canvas(preserve_view=True)
        self.frames_updated.emit()
        self.status_message.emit(f"Bild {entry.filename} erfolgreich verbessert & auf 90° ausgerichtet.")

    def _run_all_frames(self) -> None:
        if self._paths is None or self._manifest is None:
            return
        self._manifest.target_px_per_cm = int(self.spn_px_cm.value())
        self._manifest.save(self._paths.manifest_path)
        ecfg, rcfg = self._build_configs()

        total = len(self._manifest.frames)
        self.progress.setVisible(True)
        self.progress.setRange(0, total)
        for i, entry in enumerate(self._manifest.frames, start=1):
            self.status_message.emit(f"Richte Bild {i}/{total} ({entry.filename}) aus ...")
            self.progress.setValue(i - 1)
            QApplication.processEvents()
            run_enhance_pipeline(self._paths, cfg=ecfg, frame_indices=[entry.index])
            run_rectify_pipeline(self._paths, rect_cfg=rcfg, enh_cfg=ecfg, frame_indices=[entry.index])
            self.frame_list.item(i - 1).setText(
                f"[{entry.index:02d}] {entry.filename}  ({entry.width}x{entry.height}) — ausgerichtet"
            )

        self.progress.setValue(total)
        self.progress.setVisible(False)
        self._refresh_diagnostics(self._manifest.frames[self._current_index - 1].filename)
        self._refresh_canvas(preserve_view=False)
        self.frames_updated.emit()
        self.status_message.emit(f"Alle {total} Bilder erfolgreich auf 90°-Gitter ausgerichtet.")
