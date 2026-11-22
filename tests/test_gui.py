"""Offscreen PySide6 GUI tests for TiledImageCanvas, AlignEnhanceTab, StitchPanoramaTab, and CTGStudioWindow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ctg_converter.dataset import DatasetPaths
from ctg_converter.gui.app import CTGStudioWindow, _apply_dark_fusion_theme


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    _apply_dark_fusion_theme(app)
    return app


def test_studio_window_loads_real_dataset_and_switches_views(qapp: QApplication) -> None:
    ds = DatasetPaths(REPO_ROOT / "data", "test", "2026-11-22_philips_ctg_001")
    win = CTGStudioWindow(
        data_root=REPO_ROOT / "data",
        initial_dataset=ds,
        initial_tab="align",
    )
    win.show()
    qapp.processEvents()

    assert win.align_tab.frame_list.count() == 21
    # Cycle through all 5 layer modes in AlignEnhanceTab
    for mode_idx in range(win.align_tab.cmb_layer.count()):
        win.align_tab.cmb_layer.setCurrentIndex(mode_idx)
        qapp.processEvents()
        assert win.align_tab.canvas.scene().items()

    # Switch to Frame 15 (90-degree rotated camera frame)
    win.align_tab.frame_list.setCurrentRow(14)
    qapp.processEvents()
    assert "frame_0015.png" in win.align_tab.lbl_diag.text()

    # Switch to StitchPanoramaTab and inspect panorama + Red/Cyan overlap + 50/50 blend
    win.tabs.setCurrentIndex(1)
    qapp.processEvents()
    assert win.stitch_tab.table.rowCount() == 20

    for view_idx in range(win.stitch_tab.cmb_mode.count()):
        win.stitch_tab.cmb_mode.setCurrentIndex(view_idx)
        qapp.processEvents()
        assert win.stitch_tab.canvas.scene().items()

    win.close()
