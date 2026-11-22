"""Visual Tool 2: Interactive CTG Panorama Stitcher, Overlap Inspector & PNG/WebP Exporter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ctg_converter.dataset.manifest import DatasetManifest, DatasetPaths
from ctg_converter.pipelines.extract import (
    ExtractConfig,
    run_extract_pipeline,
)
from ctg_converter.pipelines.web_export import default_web_dir, export_to_web_app
from ctg_converter.pipelines.stitch import (
    WEBP_MAX_DIMENSION_PX,
    StitchConfig,
    run_stitch_pipeline,
)

from .tiled_canvas import TiledImageCanvas


def _median_present(values: list) -> float:
    """Median of the numeric samples, skipping gaps the extractor left as null."""
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return float(np.median(nums)) if nums else 0.0


class StitchPanoramaTab(QWidget):
    """Visual tool for inspecting pairwise overlaps, nudging seams, and exporting giant PNG/WebP panoramas."""

    status_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._paths: DatasetPaths | None = None
        self._manifest: DatasetManifest | None = None
        self._stitch_meta: dict | None = None
        self._selected_pair_idx: int = 0

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter)

        # Left side: Pairwise transitions table + Manual overlap nudging + Export buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)

        grp_table = QGroupBox("Bild-Übergänge & Überlappungs-Qualität")
        vt = QVBoxLayout(grp_table)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Paar", "dx (px)", "dy", "Überlapp", "ONNX", "Konfidenz"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.currentCellChanged.connect(self._on_pair_selected)
        vt.addWidget(self.table)
        left_layout.addWidget(grp_table, stretch=3)

        grp_nudge = QGroupBox("Manuelle Übergangs-Feinjustierung (Optional)")
        vn = QVBoxLayout(grp_nudge)
        row_dx = QHBoxLayout()
        row_dx.addWidget(QLabel("dx (px):"))
        self.spn_dx = QSpinBox()
        self.spn_dx.setRange(-2500, 3500)
        self.spn_dx.valueChanged.connect(self._on_nudge_preview)
        row_dx.addWidget(self.spn_dx)

        btn_box_m = QPushButton("-1 Kästchen (-25px)")
        btn_box_m.clicked.connect(lambda: self.spn_dx.setValue(self.spn_dx.value() - 25))
        btn_box_p = QPushButton("+1 Kästchen (+25px)")
        btn_box_p.clicked.connect(lambda: self.spn_dx.setValue(self.spn_dx.value() + 25))
        row_dx.addWidget(btn_box_m)
        row_dx.addWidget(btn_box_p)
        vn.addLayout(row_dx)

        row_dy = QHBoxLayout()
        row_dy.addWidget(QLabel("dy (px):"))
        self.spn_dy = QSpinBox()
        self.spn_dy.setRange(-50, 50)
        self.spn_dy.valueChanged.connect(self._on_nudge_preview)
        row_dy.addWidget(self.spn_dy)

        btn_save_override = QPushButton("Versatz übernehmen & neu zusammenfügen")
        btn_save_override.clicked.connect(self._apply_manual_override)
        btn_reset_override = QPushButton("Auto-Match")
        btn_reset_override.clicked.connect(self._reset_manual_override)
        row_dy.addWidget(btn_save_override)
        row_dy.addWidget(btn_reset_override)
        vn.addLayout(row_dy)
        left_layout.addWidget(grp_nudge)

        self.lbl_summary = QLabel("Kein Panorama geladen.")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet(
            "background: #23272e; color: #dcdcdc; padding: 8px; border-radius: 4px; font-family: monospace;"
        )
        left_layout.addWidget(self.lbl_summary)

        self.btn_stitch = QPushButton("Enden zusammenfügen (Panorama-Pipeline starten)")
        self.btn_stitch.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_stitch.clicked.connect(self.run_stitch)
        left_layout.addWidget(self.btn_stitch)

        export_row = QHBoxLayout()
        self.btn_exp_png = QPushButton("Als PNG speichern...")
        self.btn_exp_png.clicked.connect(lambda: self._export_custom("png"))
        self.btn_exp_webp = QPushButton("Als WebP speichern...")
        self.btn_exp_webp.clicked.connect(lambda: self._export_custom("webp"))
        export_row.addWidget(self.btn_exp_png)
        export_row.addWidget(self.btn_exp_webp)
        left_layout.addLayout(export_row)

        bridge_row = QHBoxLayout()
        self.btn_extract = QPushButton("Werte extrahieren (CSV + JSON)")
        self.btn_extract.clicked.connect(self.run_extract)
        self.btn_export_web = QPushButton("An CTG-Blatt Web-App senden...")
        self.btn_export_web.clicked.connect(self._export_web_repo)
        bridge_row.addWidget(self.btn_extract)
        bridge_row.addWidget(self.btn_export_web)
        left_layout.addLayout(bridge_row)

        splitter.addWidget(left_panel)

        # Right side: View selector (Full Panorama vs Pair Overlap Inspector) + Tiled Canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Ansicht:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(
            [
                "Gesamt-Panorama (Riesiges CTG-Streifenbild)",
                "Paar-Deckungskontrolle (Rot/Cyan-Überlappung des gewählten Übergangs)",
                "Paar-Überblendung (50/50 Alpha-Vorschau des gewählten Übergangs)",
            ]
        )
        self.cmb_mode.currentIndexChanged.connect(lambda _: self._refresh_view())
        bar.addWidget(self.cmb_mode, stretch=2)

        self.chk_seams = QCheckBox("Nähte & Bildgrenzen anzeigen")
        self.chk_seams.setChecked(True)
        self.chk_seams.toggled.connect(lambda _: self._update_pano_overlays())
        bar.addWidget(self.chk_seams)

        self.chk_grid = QCheckBox("90°-Metrikgitter")
        self.chk_grid.setChecked(False)
        self.chk_grid.toggled.connect(lambda _: self._update_pano_overlays())
        bar.addWidget(self.chk_grid)

        self.chk_curves = QCheckBox("Extrahierte Kurven (FHF/Puls/Toco)")
        self.chk_curves.setChecked(True)
        self.chk_curves.toggled.connect(lambda _: self._update_pano_overlays())
        bar.addWidget(self.chk_curves)

        self.canvas = TiledImageCanvas()
        self.canvas.cursor_metric_changed.connect(self.status_message.emit)

        btn_zin = QPushButton("Zoom +")
        btn_zin.clicked.connect(lambda: self.canvas.zoom_by(1.25))
        btn_zout = QPushButton("Zoom -")
        btn_zout.clicked.connect(lambda: self.canvas.zoom_by(0.8))
        btn_fit_h = QPushButton("Höhe einpassen")
        btn_fit_h.clicked.connect(self.canvas.fit_height_or_all)
        btn_fit_all = QPushButton("Gesamtbreite")
        btn_fit_all.clicked.connect(self.canvas.fit_all)
        bar.addWidget(btn_zin)
        bar.addWidget(btn_zout)
        bar.addWidget(btn_fit_h)
        bar.addWidget(btn_fit_all)

        right_layout.addLayout(bar)
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([430, 910])

    def load_dataset(self, paths: DatasetPaths) -> None:
        self._paths = paths
        self._manifest = DatasetManifest.load(paths.manifest_path)
        meta_path = paths.output_dir / "stitch_metadata.json"
        if meta_path.exists():
            self._stitch_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._populate_table()
            self._refresh_summary()
            self._refresh_view()
        else:
            self._stitch_meta = None
            self.table.setRowCount(0)
            self.canvas.clear_canvas()
            self.lbl_summary.setText(
                "Noch kein Panorama erstellt.\nKlicke auf 'Enden zusammenfügen', um das Gesamt-PNG und WebP zu berechnen."
            )

    def _populate_table(self) -> None:
        if not self._stitch_meta:
            return
        pms = self._stitch_meta.get("pair_matches", [])
        self.table.blockSignals(True)
        self.table.setRowCount(len(pms))
        for r, pm in enumerate(pms):
            conf = float(pm["confidence"])
            pair_txt = f"{pm['left_index']:02d} -> {pm['right_index']:02d}"
            vals = [
                pair_txt,
                f"{pm['dx']:+.0f}",
                f"{pm['dy']:+.0f}",
                f"{pm['overlap_width_px']}",
                f"{pm['onnx_inliers']}",
                f"{conf:.2f}" + (" [M]" if pm.get("manual_override") else ""),
            ]
            bg = (
                QColor(35, 85, 50)
                if conf >= 0.60
                else (QColor(95, 80, 30) if conf >= 0.42 else QColor(105, 40, 40))
            )
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setBackground(bg)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        if pms:
            row = min(self._selected_pair_idx, len(pms) - 1)
            self.table.selectRow(row)
            self._sync_spinboxes_from_pair(row)

    def _refresh_summary(self) -> None:
        if not self._stitch_meta or not self._paths:
            return
        m = self._stitch_meta
        png_p = self._paths.output_dir / "panorama.png"
        webp_p = self._paths.output_dir / "panorama.webp"
        png_mb = png_p.stat().st_size / (1024 * 1024) if png_p.exists() else 0.0
        webp_mb = webp_p.stat().st_size / (1024 * 1024) if webp_p.exists() else 0.0
        extra_line = ""
        sess_p = self._paths.output_dir / "ctg_session.json"
        if sess_p.exists():
            sess = json.loads(sess_p.read_text(encoding="utf-8"))
            f_med = _median_present(sess.get("fhr") or [])
            m_med = _median_present(sess.get("mhr") or [])
            extra_line = f"\nWerte (1 Hz): {len(sess.get('fhr', []))}s | FHF ~{f_med:.0f} bpm | Puls ~{m_med:.0f} bpm"
        self.lbl_summary.setText(
            f"Panorama:    {m['panorama_width_px']} x {m['panorama_height_px']} px\n"
            f"CTG-Länge:   {m['estimated_strip_length_cm']:.1f} cm ({m['estimated_duration_minutes']:.1f} Min @ 1 cm/min)\n"
            f"Auflösung:   {m['target_px_per_cm']} px/cm (1 Kästchen = {m['target_px_per_cm']*0.25:.0f} px)\n"
            f"Dateigrößen: PNG = {png_mb:.2f} MB  |  WebP = {webp_mb:.2f} MB{extra_line}"
        )

    def _sync_spinboxes_from_pair(self, row: int) -> None:
        if not self._stitch_meta:
            return
        pms = self._stitch_meta.get("pair_matches", [])
        if 0 <= row < len(pms):
            self._selected_pair_idx = row
            self.spn_dx.blockSignals(True)
            self.spn_dy.blockSignals(True)
            self.spn_dx.setValue(int(round(pms[row]["dx"])))
            self.spn_dy.setValue(int(round(pms[row]["dy"])))
            self.spn_dx.blockSignals(False)
            self.spn_dy.blockSignals(False)

    def _on_pair_selected(self, current_row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if current_row < 0 or not self._stitch_meta:
            return
        self._sync_spinboxes_from_pair(current_row)
        if self.cmb_mode.currentIndex() == 0:
            # Scroll panorama view smoothly to the selected seam!
            seams = self._stitch_meta.get("seam_x_positions_px", [])
            if current_row < len(seams) and seams[current_row] is not None:
                sx = float(seams[current_row])
                self.canvas.centerOn(sx, float(self._stitch_meta["panorama_height_px"]) * 0.5)
        else:
            self._refresh_view(preserve_view=False)

    def _on_nudge_preview(self) -> None:
        if self.cmb_mode.currentIndex() != 0:
            self._refresh_view(preserve_view=True)

    def _render_pair_comparison(self, pair_idx: int, dx: int, dy: int, red_cyan: bool) -> np.ndarray | None:
        if self._paths is None or self._manifest is None:
            return None
        if pair_idx < 0 or pair_idx + 1 >= len(self._manifest.frames):
            return None

        f1 = self._manifest.frames[pair_idx]
        f2 = self._manifest.frames[pair_idx + 1]
        s1 = Path(f1.filename).stem
        s2 = Path(f2.filename).stem

        if red_cyan:
            t1 = cv2.imread(str(self._paths.rectified_dir / f"{s1}_rect_trace.png"), cv2.IMREAD_GRAYSCALE)
            t2 = cv2.imread(str(self._paths.rectified_dir / f"{s2}_rect_trace.png"), cv2.IMREAD_GRAYSCALE)
            if t1 is None or t2 is None:
                return None
            h1, w1 = t1.shape
            h2, w2 = t2.shape
            x_min = min(0, dx)
            x_max = max(w1, dx + w2)
            cw = x_max - x_min
            ch = max(h1, h2)
            c1 = np.zeros((ch, cw), dtype=np.uint8)
            c2 = np.zeros((ch, cw), dtype=np.uint8)
            c1[:h1, -x_min : -x_min + w1] = t1

            y2_0 = max(0, dy)
            y2_1 = min(ch, dy + h2)
            sy0 = y2_0 - dy
            sy1 = y2_1 - dy
            c2[y2_0:y2_1, dx - x_min : dx - x_min + w2] = t2[sy0:sy1, :]

            # White background: left trace absorbs G&B (appears Red), right trace absorbs R (appears Cyan);
            # exact alignment absorbs R, G & B -> crisp pure BLACK!
            out_r = np.clip(252 - c2.astype(np.int32), 0, 255).astype(np.uint8)
            out_g = np.clip(252 - c1.astype(np.int32), 0, 255).astype(np.uint8)
            out_b = np.clip(252 - c1.astype(np.int32), 0, 255).astype(np.uint8)
            return np.stack([out_b, out_g, out_r], axis=-1)

        b1 = cv2.imread(str(self._paths.rectified_dir / f"{s1}_rectified.png"), cv2.IMREAD_COLOR)
        b2 = cv2.imread(str(self._paths.rectified_dir / f"{s2}_rectified.png"), cv2.IMREAD_COLOR)
        m1 = cv2.imread(str(self._paths.rectified_dir / f"{s1}_rect_mask.png"), cv2.IMREAD_GRAYSCALE)
        m2 = cv2.imread(str(self._paths.rectified_dir / f"{s2}_rect_mask.png"), cv2.IMREAD_GRAYSCALE)
        if b1 is None or b2 is None or m1 is None or m2 is None:
            return None

        h1, w1 = b1.shape[:2]
        h2, w2 = b2.shape[:2]
        x_min = min(0, dx)
        x_max = max(w1, dx + w2)
        cw = x_max - x_min
        ch = max(h1, h2)
        acc = np.zeros((ch, cw, 3), dtype=np.float32)
        wgt = np.zeros((ch, cw, 1), dtype=np.float32)

        v1 = (m1 > 0).astype(np.float32)[:, :, None]
        acc[:h1, -x_min : -x_min + w1] += b1.astype(np.float32) * v1
        wgt[:h1, -x_min : -x_min + w1] += v1

        y2_0 = max(0, dy)
        y2_1 = min(ch, dy + h2)
        sy0, sy1 = y2_0 - dy, y2_1 - dy
        v2 = (m2[sy0:sy1, :] > 0).astype(np.float32)[:, :, None]
        acc[y2_0:y2_1, dx - x_min : dx - x_min + w2] += b2[sy0:sy1, :].astype(np.float32) * v2
        wgt[y2_0:y2_1, dx - x_min : dx - x_min + w2] += v2

        out = np.where(wgt > 0, acc / np.maximum(wgt, 1.0), 245.0).astype(np.uint8)
        return out

    def _refresh_view(self, preserve_view: bool = False) -> None:
        if self._paths is None or self._manifest is None:
            return
        mode = self.cmb_mode.currentIndex()
        if mode == 0:
            pano_p = self._paths.output_dir / "panorama.png"
            if not pano_p.exists():
                return
            pano = cv2.imread(str(pano_p), cv2.IMREAD_COLOR)
            if pano is None:
                return
            self.canvas.set_numpy_image(
                pano,
                px_per_cm=self._manifest.target_px_per_cm,
                paper_speed_cm_per_min=self._manifest.paper_speed_cm_per_min,
                preserve_view=preserve_view,
            )
            self._update_pano_overlays()
        else:
            comp = self._render_pair_comparison(
                self._selected_pair_idx,
                self.spn_dx.value(),
                self.spn_dy.value(),
                red_cyan=(mode == 1),
            )
            if comp is not None:
                self.canvas.set_numpy_image(
                    comp,
                    px_per_cm=self._manifest.target_px_per_cm,
                    paper_speed_cm_per_min=self._manifest.paper_speed_cm_per_min,
                    preserve_view=preserve_view,
                )

    def _update_pano_overlays(self) -> None:
        if self.cmb_mode.currentIndex() != 0 or not self._stitch_meta or not self._manifest:
            self.canvas.clear_overlays()
            return
        self.canvas.clear_overlays()
        if self.chk_grid.isChecked():
            self.canvas.draw_canonical_grid_overlay(self._manifest.target_px_per_cm, clear_first=False)
        if self.chk_seams.isChecked():
            self.canvas.draw_seam_overlays(
                self._stitch_meta.get("seam_x_positions_px", []),
                self._stitch_meta.get("frame_placements", []),
                clear_first=False,
            )
        if self.chk_curves.isChecked() and self._paths is not None:
            cp = self._paths.output_dir / "extracted_curves_px.json"
            if cp.exists():
                curves_px = json.loads(cp.read_text(encoding="utf-8"))
                self.canvas.draw_extracted_curves_overlay(curves_px, clear_first=False)

    def run_extract(self) -> None:
        if self._paths is None:
            return
        self.status_message.emit("Extrahiere kalibrierte 1-Hz FHF-, Puls- und Toco-Zeitreihen via Viterbi-Tracking ...")
        QApplication.processEvents()
        res = run_extract_pipeline(self._paths, ext_cfg=ExtractConfig())
        self.chk_curves.setChecked(True)
        self._update_pano_overlays()
        self._refresh_summary()
        self.status_message.emit(
            f"Werte extrahiert ({res['samples_1hz']} Sek): FHF Median {res['fhr_median_bpm']} bpm | "
            f"Puls {res['mhr_median_bpm']} bpm | Toco {res['toco_median_pct']}% -> trace.csv & ctg_session.json"
        )

    def _export_web_repo(self) -> None:
        if self._paths is None:
            return
        default_dir = default_web_dir()
        folder = QFileDialog.getExistingDirectory(
            self,
            "CTG-Blatt Web-App-Verzeichnis",
            str(default_dir if default_dir.is_dir() else Path.home()),
        )
        if not folder:
            return
        out = export_to_web_app(self._paths, Path(folder))
        self.status_message.emit(
            f"CTG-Blatt Web-Export abgeschlossen: {out['session_json']} ({out['scans']} Bilder)"
        )

    def run_stitch(self) -> None:
        if self._paths is None:
            return
        self.status_message.emit("Berechne Überlappungen, minimale Fehler-Nähte und exportiere PNG & WebP ...")
        QApplication.processEvents()
        report = run_stitch_pipeline(self._paths, stitch_cfg=StitchConfig())
        self._stitch_meta = report
        self._populate_table()
        self._refresh_summary()
        self._refresh_view(preserve_view=False)
        self.status_message.emit(
            f"Panorama fertiggestellt: {report['panorama_width_px']}x{report['panorama_height_px']} px (PNG & WebP gespeichert)."
        )

    def _apply_manual_override(self) -> None:
        if self._paths is None or self._manifest is None:
            return
        right_frame_idx = self._selected_pair_idx + 1
        if 0 < right_frame_idx < len(self._manifest.frames):
            entry = self._manifest.frames[right_frame_idx]
            entry.manual_dx_override = float(self.spn_dx.value())
            entry.manual_dy_override = float(self.spn_dy.value())
            self._manifest.save(self._paths.manifest_path)
            self.run_stitch()

    def _reset_manual_override(self) -> None:
        if self._paths is None or self._manifest is None:
            return
        right_frame_idx = self._selected_pair_idx + 1
        if 0 < right_frame_idx < len(self._manifest.frames):
            entry = self._manifest.frames[right_frame_idx]
            entry.manual_dx_override = None
            entry.manual_dy_override = None
            self._manifest.save(self._paths.manifest_path)
            self.run_stitch()

    def _export_custom(self, fmt: str) -> None:
        if self._paths is None:
            return
        src_png = self._paths.output_dir / "panorama.png"
        if not src_png.exists():
            self.run_stitch()
        if not src_png.exists():
            return

        default_name = f"{self._paths.dataset_id}_panorama.{fmt}"
        filter_str = "PNG Image (*.png)" if fmt == "png" else "WebP Image (*.webp)"
        dst_str, _ = QFileDialog.getSaveFileName(
            self, f"Panorama als {fmt.upper()} exportieren", str(Path.home() / default_name), filter_str
        )
        if not dst_str:
            return
        dst = Path(dst_str)
        if fmt == "png":
            shutil.copy2(src_png, dst)
        else:
            src_webp = self._paths.output_dir / "panorama.webp"
            if src_webp.exists():
                shutil.copy2(src_webp, dst)
            else:
                bgr = cv2.imread(str(src_png), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                if w > WEBP_MAX_DIMENSION_PX:
                    sc = (WEBP_MAX_DIMENSION_PX - 3) / float(w)
                    rgb = cv2.resize(rgb, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                Image.fromarray(rgb).save(str(dst), format="WEBP", quality=92)
        self.status_message.emit(f"Exportiert nach: {dst}")
