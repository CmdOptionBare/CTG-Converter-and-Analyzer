"""Main PySide6 Studio window hosting the Dataset Importer, Alignment Tool, and Panorama Stitcher."""

from __future__ import annotations

import resource
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ctg_converter.dataset import (
    DatasetPaths,
    import_folder_as_dataset,
    list_datasets,
)

from .align_view import AlignEnhanceTab
from .stitch_view import StitchPanoramaTab


def _apply_dark_fusion_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(43, 47, 54))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(232, 236, 242))
    pal.setColor(QPalette.ColorRole.Base, QColor(31, 34, 40))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 47, 54))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(240, 240, 240))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
    pal.setColor(QPalette.ColorRole.Text, QColor(232, 236, 242))
    pal.setColor(QPalette.ColorRole.Button, QColor(56, 62, 72))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(235, 238, 244))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(46, 134, 193))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)


class CTGStudioWindow(QMainWindow):
    """Unified PySide6 application for Step 1: CTG Alignment, Enhancement & Panorama Stitching."""

    def __init__(
        self,
        data_root: Path,
        initial_dataset: DatasetPaths | None = None,
        initial_tab: str = "align",
    ) -> None:
        super().__init__()
        self.data_root = data_root.resolve()
        self._datasets: list[DatasetPaths] = []

        self.setWindowTitle("CTG Converter & Analyzer — Schritt 1: 90°-Ausrichtung & Panorama-Stitching")
        self.resize(1440, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 4)

        # Top Dataset Bar
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Aktiver Datensatz:"))
        self.cmb_dataset = QComboBox()
        self.cmb_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        top_bar.addWidget(self.cmb_dataset, stretch=2)

        self.btn_import = QPushButton("Neuen Bildordner als Datensatz importieren...")
        self.btn_import.clicked.connect(self._import_new_folder)
        top_bar.addWidget(self.btn_import)

        self.lbl_ram = QLabel("RAM: -- MB")
        self.lbl_ram.setStyleSheet("color: #8ab4f8; padding-left: 12px;")
        top_bar.addWidget(self.lbl_ram)
        vbox.addLayout(top_bar)

        # Main Two-Tool Tabs
        self.tabs = QTabWidget()
        self.align_tab = AlignEnhanceTab(self)
        self.stitch_tab = StitchPanoramaTab(self)

        self.tabs.addTab(
            self.align_tab,
            "Werkzeug 1: Bildverbesserung & 90°-Kästchen-Ausrichtung",
        )
        self.tabs.addTab(
            self.stitch_tab,
            "Werkzeug 2: Enden zusammenfügen & Panorama-Export (PNG / WebP)",
        )
        vbox.addWidget(self.tabs, stretch=1)

        self.align_tab.status_message.connect(self.statusBar().showMessage)
        self.stitch_tab.status_message.connect(self.statusBar().showMessage)

        self._ram_timer = QTimer(self)
        self._ram_timer.timeout.connect(self._update_ram_badge)
        self._ram_timer.start(1500)
        self._update_ram_badge()

        self._reload_dataset_list(select=initial_dataset)
        if initial_tab == "stitch":
            self.tabs.setCurrentIndex(1)

    def _update_ram_badge(self) -> None:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        self.lbl_ram.setText(f"Peak RAM: {rss_mb:.0f} MB")

    def _reload_dataset_list(self, select: DatasetPaths | None = None) -> None:
        self._datasets = list_datasets(self.data_root)
        self.cmb_dataset.blockSignals(True)
        self.cmb_dataset.clear()
        sel_idx = 0
        for i, ds in enumerate(self._datasets):
            self.cmb_dataset.addItem(f"{ds.split} / {ds.dataset_id}  ({ds.input_dir})", ds)
            if select and ds.dataset_id == select.dataset_id and ds.split == select.split:
                sel_idx = i
        self.cmb_dataset.blockSignals(False)
        if self._datasets:
            self.cmb_dataset.setCurrentIndex(sel_idx)
            self._on_dataset_changed(sel_idx)

    def _on_dataset_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._datasets):
            return
        ds = self._datasets[idx]
        self.align_tab.load_dataset(ds)
        self.stitch_tab.load_dataset(ds)
        self.statusBar().showMessage(f"Datensatz geladen: {ds.split}/{ds.dataset_id}")

    def _import_new_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Ordner mit CTG-Einzelbildern auswählen", str(Path.home())
        )
        if not folder:
            return
        src_path = Path(folder)
        default_id = src_path.name.lower().replace(" ", "_")
        ds_id, ok = QInputDialog.getText(
            self, "Datensatz-Name", "Eindeutige Datensatz-ID:", text=default_id
        )
        if not ok or not ds_id.strip():
            return
        try:
            new_ds = import_folder_as_dataset(
                source_dir=src_path,
                data_root=self.data_root,
                dataset_id=ds_id.strip(),
                split="imports",
            )
            self._reload_dataset_list(select=new_ds)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import fehlgeschlagen", str(exc))


def launch_gui(
    initial_dataset: DatasetPaths | None = None,
    initial_tab: str = "align",
    data_root: Path | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    _apply_dark_fusion_theme(app)

    root = data_root or Path("data").resolve()
    win = CTGStudioWindow(
        data_root=root,
        initial_dataset=initial_dataset,
        initial_tab=initial_tab,
    )
    win.show()
    return app.exec()
