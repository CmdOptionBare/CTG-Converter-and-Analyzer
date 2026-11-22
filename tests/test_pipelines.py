"""Tests for Pipeline 1 (Enhance), Pipeline 2 (90-degree Rectify), ONNX Matcher, and Pipeline 3 (Stitch)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ctg_converter.dataset import import_folder_as_dataset
from ctg_converter.models import OnnxPatchMatcher
from ctg_converter.pipelines.enhance import (
    detect_coarse_orientation_cw,
    enhance_ctg_frame,
    rotate_cw,
)
from ctg_converter.pipelines.rectify import RectifyConfig, rectify_ctg_frame
from ctg_converter.pipelines.stitch import (
    WEBP_MAX_DIMENSION_PX,
    StitchConfig,
    run_stitch_pipeline,
)


def _distort_synthetic_frame(clean_crop: np.ndarray, skew_deg: float = -5.5) -> np.ndarray:
    """Apply uneven warm lighting, skew rotation, keystone trapezoid, and a skin-colored finger intrusion."""
    h, w = clean_crop.shape[:2]
    # Add uneven shadow gradient (top-left 0.65x darker & warm yellowish)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    illum = 0.68 + 0.30 * (xx / w) + 0.12 * (yy / h)
    shaded = np.empty_like(clean_crop, dtype=np.float32)
    shaded[:, :, 0] = clean_crop[:, :, 0] * illum * 0.90  # B
    shaded[:, :, 1] = clean_crop[:, :, 1] * illum * 0.97  # G
    shaded[:, :, 2] = clean_crop[:, :, 2] * illum * 1.03  # R
    shaded_u8 = np.clip(shaded, 0, 255).astype(np.uint8)

    # Apply perspective warp (rotation + keystone)
    pad = 120
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 55, dtype=np.uint8)
    canvas[pad : pad + h, pad : pad + w] = shaded_u8
    ch, cw = canvas.shape[:2]

    src_pts = np.array(
        [[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]], dtype=np.float32
    )
    # Tilt + slight trapezoid
    rad = np.deg2rad(skew_deg)
    cx, cy = cw / 2.0, ch / 2.0
    dst_pts = []
    for i, (px, py) in enumerate(src_pts):
        dx, dy = px - cx, py - cy
        rx = cx + dx * np.cos(rad) - dy * np.sin(rad)
        ry = cy + dx * np.sin(rad) + dy * np.cos(rad)
        if i in (0, 1):
            rx = cx + (rx - cx) * 0.95  # Top slightly narrower (further from lens)
        dst_pts.append([rx, ry])

    H = cv2.getPerspectiveTransform(src_pts, np.array(dst_pts, dtype=np.float32))
    warped = cv2.warpPerspective(canvas, H, (cw, ch), borderValue=(45, 45, 45))

    # Draw skin-toned thumb intrusion on left border
    cv2.circle(warped, (25, ch // 2), 95, (125, 148, 208), -1)
    return warped


def test_coarse_90_orientation_all_four_rotations(synthetic_strip_bgr: np.ndarray) -> None:
    crop = synthetic_strip_bgr[:, :1600]
    distorted = _distort_synthetic_frame(crop, skew_deg=-4.0)

    # Rotate upright distorted frame by 0, 90, 180, 270 CCW -> detector must recover CW inverse!
    for put_ccw, expected_cw in [(0, 0), (90, 90), (180, 180), (270, 270)]:
        rotated_input = rotate_cw(distorted, (360 - expected_cw) % 360)
        detected = detect_coarse_orientation_cw(rotated_input)
        assert detected == expected_cw, f"Expected {expected_cw} CW for input rotated {360-expected_cw}, got {detected}"


def test_enhance_and_rectify_recover_90deg_grid(synthetic_strip_bgr: np.ndarray) -> None:
    crop = synthetic_strip_bgr[:, 100:1800]
    distorted = rotate_cw(_distort_synthetic_frame(crop, skew_deg=-6.0), 270)  # Needs 90 CW!

    enh_bgr, grid_u8, trace_u8, valid_mask, rot_deg = enhance_ctg_frame(distorted)
    assert rot_deg == 90
    assert 0.55 < float((valid_mask > 0).mean()) < 0.95

    rect_bgr, rect_grid, rect_trace, rect_mask, quality = rectify_ctg_frame(
        enh_bgr, grid_u8, trace_u8, valid_mask, cfg=RectifyConfig(target_px_per_cm=100)
    )
    assert rect_bgr.shape[0] == 1420
    assert abs(quality["orthogonal_angle_deg"] - 90.0) <= 0.6
    assert abs(quality["measured_grid_period_y_px"] - 25.0) <= 1.2
    assert abs(quality["measured_grid_period_x_px"] - 25.0) <= 1.5
    assert rect_trace.max() > 120


def test_end_to_end_synthetic_3frame_stitch_and_onnx(
    tmp_path: Path, synthetic_strip_bgr: np.ndarray
) -> None:
    # Slice 3 overlapping windows from the 26cm strip (140 px/cm -> 3640 px wide)
    src_dir = tmp_path / "cam"
    src_dir.mkdir()
    slices = [
        (0, 1750, 0),
        (700, 2550, 270),   # Rotated 90 deg CCW -> tests mixed orientation stitching!
        (1550, 3500, 0),
    ]
    for i, (x0, x1, rot_in) in enumerate(slices, start=1):
        dist = _distort_synthetic_frame(synthetic_strip_bgr[:, x0:x1], skew_deg=(-3.5 + i * 2.0))
        dist = rotate_cw(dist, rot_in)
        cv2.imwrite(str(src_dir / f"shot_{i:02d}.png"), dist)

    paths = import_folder_as_dataset(
        source_dir=src_dir,
        data_root=tmp_path / "data",
        dataset_id="synth_ctg",
        split="test",
        target_px_per_cm=100,
    )

    report = run_stitch_pipeline(
        paths,
        stitch_cfg=StitchConfig(use_onnx_matcher=True),
        models_dir=tmp_path / "models",
    )

    assert Path(report["exports"]["png"]).exists()
    assert Path(report["exports"]["webp"]).exists()
    assert len(report["pair_matches"]) == 2
    for pm in report["pair_matches"]:
        assert pm["dx"] > 250
        assert abs(pm["dy"]) <= 6
        assert pm["confidence"] > 0.55

    # Direct ONNX matcher verification
    matcher = OnnxPatchMatcher(tmp_path / "models")
    t1 = cv2.imread(str(paths.rectified_dir / "frame_0001_rect_trace.png"), cv2.IMREAD_GRAYSCALE)
    m1 = cv2.imread(str(paths.rectified_dir / "frame_0001_rect_mask.png"), cv2.IMREAD_GRAYSCALE)
    t2 = cv2.imread(str(paths.rectified_dir / "frame_0002_rect_trace.png"), cv2.IMREAD_GRAYSCALE)
    m2 = cv2.imread(str(paths.rectified_dir / "frame_0002_rect_mask.png"), cv2.IMREAD_GRAYSCALE)
    dx, dy, conf, inl = matcher.match_translation(t1, m1, t2, m2)
    assert inl >= 5
    assert abs(dx - report["pair_matches"][0]["dx"]) <= 12.0
    assert WEBP_MAX_DIMENSION_PX == 16383
