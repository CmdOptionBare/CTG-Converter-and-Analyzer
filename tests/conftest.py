"""Shared pytest fixtures including synthetic CTG strip generator with perspective & shading."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ctg_converter.pipelines.rectify import (
    CANONICAL_TOTAL_HEIGHT_CM,
    FHR_LINES_CM,
    FHR_S_BOT_CM,
    FHR_S_TOP_CM,
    TOCO_LINES_CM,
    TOCO_S_BOT_CM,
    TOCO_S_TOP_CM,
)


def make_synthetic_canonical_ctg(width_cm: float = 24.0, px_per_cm: int = 120) -> np.ndarray:
    """Create a realistic flat BGR CTG strip with green 2.5mm/4.0mm grid, white separator, and black curves."""
    h = int(round(CANONICAL_TOTAL_HEIGHT_CM * px_per_cm))
    w = int(round(width_cm * px_per_cm))
    bgr = np.full((h, w, 3), (246, 248, 246), dtype=np.uint8)

    fhr_y0 = int(round(FHR_S_TOP_CM * px_per_cm))
    fhr_y1 = int(round(FHR_S_BOT_CM * px_per_cm))
    toco_y0 = int(round(TOCO_S_TOP_CM * px_per_cm))
    toco_y1 = int(round(TOCO_S_BOT_CM * px_per_cm))

    # Draw vertical green grid lines every 0.25 cm inside FHR and Toco (skipping middle separator!)
    n_vx = int(width_cm / 0.25) + 1
    for i in range(n_vx):
        x = int(round(i * 0.25 * px_per_cm))
        if 0 <= x < w:
            thick = 2 if (i % 4 == 0) else 1
            col = (165, 226, 158) if (i % 4 == 0) else (188, 234, 182)
            cv2.line(bgr, (x, fhr_y0), (x, fhr_y1), col, thick)
            cv2.line(bgr, (x, toco_y0), (x, toco_y1), col, thick)

    # Draw 33 horizontal FHR green lines and 11 horizontal Toco green lines
    for i, s_cm in enumerate(FHR_LINES_CM):
        y = int(round(float(s_cm) * px_per_cm))
        thick = 2 if (i % 4 == 0) else 1
        col = (165, 226, 158) if (i % 4 == 0) else (188, 234, 182)
        cv2.line(bgr, (0, y), (w - 1, y), col, thick)

    for i, s_cm in enumerate(TOCO_LINES_CM):
        y = int(round(float(s_cm) * px_per_cm))
        thick = 2 if (i % 2 == 0) else 1
        col = (155, 220, 148) if (i % 2 == 0) else (178, 230, 172)
        cv2.line(bgr, (0, y), (w - 1, y), col, thick)

    # Draw rich non-periodic FHR & Toco black curves + beat-to-beat CTG variability + Kineto blocks
    rng = np.random.default_rng(42)
    xs = np.arange(w, dtype=np.float64)
    t = xs / float(px_per_cm)
    micro_var = np.interp(
        xs,
        np.arange(0, w + 8, 5, dtype=np.float64),
        rng.normal(0.0, 0.022 * (fhr_y1 - fhr_y0), size=(w // 5) + 2),
    )
    fhr_y = (
        0.5 * (fhr_y0 + fhr_y1)
        + 0.18 * (fhr_y1 - fhr_y0) * np.sin(1.7 * t)
        + 0.11 * (fhr_y1 - fhr_y0) * np.cos(4.3 * t + 0.4 * t * t * 0.05)
        + 0.06 * (fhr_y1 - fhr_y0) * np.sin(11.1 * t)
        + micro_var
    )
    pts_fhr = np.column_stack([xs, fhr_y]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(bgr, [pts_fhr], isClosed=False, color=(42, 42, 42), thickness=2, lineType=cv2.LINE_AA)

    toco_y = (
        toco_y1
        - 0.15 * (toco_y1 - toco_y0)
        - 0.35 * (toco_y1 - toco_y0) * np.exp(-((t - width_cm * 0.35) ** 2) / 1.8)
        - 0.45 * (toco_y1 - toco_y0) * np.exp(-((t - width_cm * 0.72) ** 2) / 2.2)
    )
    pts_toco = np.column_stack([xs, toco_y]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(bgr, [pts_toco], isClosed=False, color=(35, 35, 35), thickness=2, lineType=cv2.LINE_AA)

    # Timestamp & Kineto markers
    for mark_cm in np.arange(2.5, width_cm - 1.5, 4.7):
        mx = int(round(mark_cm * px_per_cm))
        cv2.rectangle(bgr, (mx, toco_y0 + 6), (mx + 9, toco_y0 + 26), (30, 30, 30), -1)
        cv2.putText(
            bgr,
            f"T+{mark_cm:.1f}",
            (mx - 10, fhr_y1 + int(0.45 * (toco_y0 - fhr_y1))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    return bgr


@pytest.fixture(scope="session")
def synthetic_strip_bgr() -> np.ndarray:
    return make_synthetic_canonical_ctg(width_cm=26.0, px_per_cm=140)
