"""Memory-efficient tiled QGraphicsView image viewer for very wide CTG panoramas and grid overlays."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from ctg_converter.pipelines.rectify import (
    FHR_S_BOT_CM,
    FHR_S_TOP_CM,
    TOCO_S_BOT_CM,
    TOCO_S_TOP_CM,
)


class TiledImageCanvas(QGraphicsView):
    """Hardware/tile-accelerated viewer supporting arbitrarily wide images without X11 pixmap limits."""

    cursor_metric_changed = Signal(str)

    def __init__(self, tile_size: int = 1024, parent=None) -> None:
        super().__init__(parent)
        self.tile_size = tile_size
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(34, 37, 42))
        self.setMouseTracking(True)

        self._img_w = 0
        self._img_h = 0
        self._px_per_cm: int | None = None
        self._paper_speed: float = 1.0
        self._overlay_items: list = []

    def clear_canvas(self) -> None:
        self._scene.clear()
        self._overlay_items.clear()
        self._img_w = 0
        self._img_h = 0

    def set_numpy_image(
        self,
        image: np.ndarray,
        px_per_cm: int | None = None,
        paper_speed_cm_per_min: float = 1.0,
        preserve_view: bool = False,
    ) -> None:
        """Display a BGR (H, W, 3) or grayscale (H, W) numpy array sliced into tiles."""
        old_transform = self.transform() if preserve_view else None
        hbar_val = self.horizontalScrollBar().value() if preserve_view else 0
        vbar_val = self.verticalScrollBar().value() if preserve_view else 0

        self.clear_canvas()
        self._px_per_cm = px_per_cm
        self._paper_speed = paper_speed_cm_per_min

        if image.ndim == 2:
            h, w = image.shape
            is_color = False
        else:
            h, w = image.shape[:2]
            is_color = True

        self._img_w = w
        self._img_h = h
        ts = self.tile_size

        for y0 in range(0, h, ts):
            y1 = min(h, y0 + ts)
            for x0 in range(0, w, ts):
                x1 = min(w, x0 + ts)
                if is_color:
                    # Convert BGR slice to contiguous RGB for QImage
                    tile_rgb = np.ascontiguousarray(image[y0:y1, x0:x1, ::-1])
                    th, tw, _ = tile_rgb.shape
                    qimg = QImage(
                        tile_rgb.data,
                        tw,
                        th,
                        3 * tw,
                        QImage.Format.Format_RGB888,
                    ).copy()
                else:
                    tile_g = np.ascontiguousarray(image[y0:y1, x0:x1])
                    th, tw = tile_g.shape
                    qimg = QImage(
                        tile_g.data,
                        tw,
                        th,
                        tw,
                        QImage.Format.Format_Grayscale8,
                    ).copy()

                item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
                item.setOffset(float(x0), float(y0))
                item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                self._scene.addItem(item)

        self._scene.setSceneRect(0.0, 0.0, float(w), float(h))
        if preserve_view and old_transform is not None:
            self.setTransform(old_transform)
            self.horizontalScrollBar().setValue(hbar_val)
            self.verticalScrollBar().setValue(vbar_val)
        else:
            self.fit_height_or_all()

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def draw_canonical_grid_overlay(self, px_per_cm: int) -> None:
        """Overlay exact 90-degree metric reference lines (1 cm major + FHR/Toco borders)."""
        self.clear_overlays()
        if self._img_w <= 0 or self._img_h <= 0:
            return

        pen_major = QPen(QColor(0, 150, 255, 130), 1.2, Qt.PenStyle.DashLine)
        pen_major.setCosmetic(True)
        pen_border = QPen(QColor(255, 70, 70, 190), 2.0, Qt.PenStyle.SolidLine)
        pen_border.setCosmetic(True)

        # Horizontal canonical borders (FHR 210, FHR 50, Toco 100, Toco 0)
        for s_cm in (FHR_S_TOP_CM, FHR_S_BOT_CM, TOCO_S_TOP_CM, TOCO_S_BOT_CM):
            y = s_cm * px_per_cm
            line = QGraphicsLineItem(0.0, y, float(self._img_w), y)
            line.setPen(pen_border)
            line.setZValue(10.0)
            self._scene.addItem(line)
            self._overlay_items.append(line)

        # Every 1.0 cm (20 bpm) in FHR and 0.80 cm (20 units) in Toco
        for k in range(0, 9):
            y = (FHR_S_TOP_CM + float(k)) * px_per_cm
            line = QGraphicsLineItem(0.0, y, float(self._img_w), y)
            line.setPen(pen_major)
            line.setZValue(9.0)
            self._scene.addItem(line)
            self._overlay_items.append(line)

        # Vertical 1.0 cm (1 minute) lines across width
        step_x = float(px_per_cm)
        x = 0.0
        while x <= self._img_w:
            vline = QGraphicsLineItem(x, 0.0, x, float(self._img_h))
            vline.setPen(pen_major)
            vline.setZValue(9.0)
            self._scene.addItem(vline)
            self._overlay_items.append(vline)
            x += step_x

    def draw_seam_overlays(self, seam_xs: list[int | None], placements: list[dict]) -> None:
        """Draw vertical seam lines and frame index badges over the stitched panorama."""
        self.clear_overlays()
        if self._img_w <= 0 or self._img_h <= 0:
            return

        pen_seam = QPen(QColor(255, 140, 0, 210), 2.0, Qt.PenStyle.DashLine)
        pen_seam.setCosmetic(True)
        for idx, sx in enumerate(seam_xs, start=1):
            if sx is None or sx < 0 or sx >= self._img_w:
                continue
            line = QGraphicsLineItem(float(sx), 0.0, float(sx), float(self._img_h))
            line.setPen(pen_seam)
            line.setZValue(12.0)
            self._scene.addItem(line)
            self._overlay_items.append(line)

            lbl = QGraphicsSimpleTextItem(f"Naht {idx:02d}->{idx+1:02d} (x={sx}px)")
            lbl.setBrush(QColor(255, 190, 60))
            lbl.setPos(float(sx + 6), 12.0)
            lbl.setZValue(13.0)
            self._scene.addItem(lbl)
            self._overlay_items.append(lbl)

        for pl in placements:
            fx = float(pl.get("x_px", 0))
            fidx = pl.get("frame_index", 0)
            tag = QGraphicsSimpleTextItem(f"[F{fidx:02d}]")
            tag.setBrush(QColor(90, 210, 255))
            tag.setPos(max(4.0, fx + 6.0), float(self._img_h - 36))
            tag.setZValue(13.0)
            self._scene.addItem(tag)
            self._overlay_items.append(tag)

    def fit_all(self) -> None:
        if self._img_w > 0 and self._img_h > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def fit_height_or_all(self) -> None:
        if self._img_w <= 0 or self._img_h <= 0:
            return
        # For ultra-wide panoramas (aspect > 2.5), fit height and scroll to left start so curves are readable!
        if self._img_w > 2.5 * self._img_h:
            self.fitInView(
                0.0, 0.0, float(self._img_h * 1.6), float(self._img_h),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            self.horizontalScrollBar().setValue(0)
        else:
            self.fit_all()

    def zoom_by(self, factor: float) -> None:
        self.scale(factor, factor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else (1.0 / 1.18)
        self.scale(factor, factor)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        if self._img_w <= 0 or self._img_h <= 0:
            return
        pt = self.mapToScene(event.position().toPoint())
        x, y = pt.x(), pt.y()
        if not (0 <= x <= self._img_w and 0 <= y <= self._img_h):
            return

        if self._px_per_cm:
            cm_x = x / self._px_per_cm
            minutes = cm_x / max(self._paper_speed, 1e-3)
            s_y = y / self._px_per_cm
            if FHR_S_TOP_CM <= s_y <= FHR_S_BOT_CM:
                bpm = 210.0 - (s_y - FHR_S_TOP_CM) * 20.0
                val_txt = f"FHF: {bpm:5.1f} bpm"
            elif TOCO_S_TOP_CM <= s_y <= TOCO_S_BOT_CM:
                toco = 100.0 - (s_y - TOCO_S_TOP_CM) * 25.0
                val_txt = f"Toco: {toco:4.1f} %"
            else:
                val_txt = "Rand / Trennstreifen"
            self.cursor_metric_changed.emit(
                f"Pixel ({int(x)}, {int(y)})  |  Streifen: {cm_x:5.2f} cm  |  Zeit: +{minutes:5.2f} min  |  {val_txt}"
            )
        else:
            self.cursor_metric_changed.emit(f"Pixel ({int(x)}, {int(y)})  |  Bildgröße: {self._img_w} x {self._img_h} px")
