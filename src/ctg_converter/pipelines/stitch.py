"""Pipeline 3: Ambiguity-free Curve + ONNX Overlap Matching, Minimum-Error Seam Cutting,
and Low-RAM Panorama Export to PNG and WebP.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ctg_converter.dataset.manifest import DatasetManifest, DatasetPaths
from ctg_converter.models.onnx_matcher import OnnxPatchMatcher
from ctg_converter.pipelines.enhance import EnhanceConfig
from ctg_converter.pipelines.rectify import (
    CANONICAL_TOTAL_HEIGHT_CM,
    FHR_S_BOT_CM,
    FHR_S_TOP_CM,
    TOCO_S_BOT_CM,
    TOCO_S_TOP_CM,
    RectifyConfig,
    run_rectify_pipeline,
)

WEBP_MAX_DIMENSION_PX = 16383


@dataclass
class StitchConfig:
    """Configuration for CTG strip registration, seam cutting, and panorama export."""

    use_onnx_matcher: bool = True
    min_overlap_px: int = 120
    max_abs_dy_px: int = 10
    feather_width_px: int = 25
    export_png: bool = True
    export_webp: bool = True
    webp_quality: int = 92
    webp_lossless: bool = False


@dataclass
class PairMatchMetadata:
    """Pairwise registration result between frame i and frame i+1.

    Convention: Frame ``right_index`` is placed at ``(X_left + dx, Y_left + dy)`` in panorama space.
    So a positive ``dx`` means the camera moved rightward along the CTG strip by ``dx`` pixels.
    """

    left_index: int
    right_index: int
    left_filename: str
    right_filename: str
    dx: float
    dy: float
    overlap_width_px: int
    trace_ncc: float
    grid_ncc: float
    onnx_dx: float | None
    onnx_dy: float | None
    onnx_inliers: int
    confidence: float
    method_used: str
    manual_override: bool = False


def _masked_fft_ncc_2d(
    img1: np.ndarray,
    mask1: np.ndarray,
    img2: np.ndarray,
    mask2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Padfield's exact Masked Normalized Cross-Correlation for all 2D shifts via FFT.

    Shift convention: ``(sy, sx)`` corresponds to placing ``img2`` at offset ``(sy, sx)`` relative
    to ``img1``'s origin ``(0, 0)``. Output arrays have shape ``(H1 + H2 - 1, W1 + W2 - 1)`` where
    index ``(sy + H2 - 1, sx + W2 - 1)`` stores the correlation at ``(sy, sx)``.
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    fh = int(cv2.getOptimalDFTSize(h1 + h2 - 1))
    fw = int(cv2.getOptimalDFTSize(w1 + w2 - 1))

    m1 = (mask1 > 0).astype(np.float32)
    m2 = (mask2 > 0).astype(np.float32)
    f1 = img1.astype(np.float32) * m1
    f2 = img2.astype(np.float32) * m2

    def rfft2_pad(arr: np.ndarray) -> np.ndarray:
        buf = np.zeros((fh, fw), dtype=np.float32)
        buf[: arr.shape[0], : arr.shape[1]] = arr
        return np.fft.rfft2(buf)

    # Cross-correlation(A, B)[sy, sx] = sum_{y,x} A[y, x] * B[y - sy, x - sx]
    # Using reversed B_rev[u, v] = B[H2 - 1 - u, W2 - 1 - v], convolution A * B_rev gives exact index (sy + H2 - 1, sx + W2 - 1)
    f2_rev = f2[::-1, ::-1]
    m2_rev = m2[::-1, ::-1]
    f2_sq_rev = (f2 * f2)[::-1, ::-1]

    F1 = rfft2_pad(f1)
    F1_sq = rfft2_pad(f1 * f1)
    M1 = rfft2_pad(m1)

    F2r = rfft2_pad(f2_rev)
    F2r_sq = rfft2_pad(f2_sq_rev)
    M2r = rfft2_pad(m2_rev)

    out_h, out_w = h1 + h2 - 1, w1 + w2 - 1

    def irfft_crop(prod: np.ndarray) -> np.ndarray:
        return np.fft.irfft2(prod, s=(fh, fw))[:out_h, :out_w]

    overlap_n = np.maximum(irfft_crop(M1 * M2r), 0.0)
    sum1 = irfft_crop(F1 * M2r)
    sum2 = irfft_crop(M1 * F2r)
    sum1_sq = irfft_crop(F1_sq * M2r)
    sum2_sq = irfft_crop(M1 * F2r_sq)
    cross = irfft_crop(F1 * F2r)

    safe_n = np.maximum(overlap_n, 1.0)
    num = cross - (sum1 * sum2) / safe_n
    var1 = np.maximum(0.0, sum1_sq - (sum1 * sum1) / safe_n)
    var2 = np.maximum(0.0, sum2_sq - (sum2 * sum2) / safe_n)
    denom = np.sqrt(var1 * var2)

    ncc = np.where((overlap_n >= 100.0) & (denom > 1e-3), num / np.maximum(denom, 1e-3), -1.0)
    return np.clip(ncc, -1.0, 1.0).astype(np.float32), overlap_n.astype(np.float32)


def _eval_fullres_overlap(
    img1: np.ndarray,
    mask1: np.ndarray,
    img2: np.ndarray,
    mask2: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[float, int]:
    """Compute exact zero-mean normalized cross-correlation at integer shift (dx, dy)."""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    ox0 = max(0, dx)
    ox1 = min(w1, dx + w2)
    oy0 = max(0, dy)
    oy1 = min(h1, dy + h2)
    if ox1 - ox0 < 40 or oy1 - oy0 < 100:
        return -1.0, 0

    m_both = (mask1[oy0:oy1, ox0:ox1] > 0) & (mask2[oy0 - dy : oy1 - dy, ox0 - dx : ox1 - dx] > 0)
    count = int(np.sum(m_both))
    if count < 8000:
        return -1.0, 0

    v1 = img1[oy0:oy1, ox0:ox1][m_both].astype(np.float32)
    v2 = img2[oy0 - dy : oy1 - dy, ox0 - dx : ox1 - dx][m_both].astype(np.float32)
    v1 -= v1.mean()
    v2 -= v2.mean()
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-4:
        return 0.0, count
    ncc = float(np.dot(v1, v2) / denom)
    # Effective horizontal overlap width
    eff_w = int(np.sum(m_both.any(axis=0)))
    return ncc, eff_w


def match_rectified_pair(
    trace1: np.ndarray,
    grid1: np.ndarray,
    mask1: np.ndarray,
    trace2: np.ndarray,
    grid2: np.ndarray,
    mask2: np.ndarray,
    cfg: StitchConfig,
    onnx_matcher: OnnxPatchMatcher | None = None,
) -> dict[str, Any]:
    """Estimate exact translation (dx, dy) placing frame 2 relative to frame 1."""
    # Downsample 2x for fast global Masked FFT NCC across all valid shifts
    ds = 2
    s_t1 = cv2.resize(trace1, (trace1.shape[1] // ds, trace1.shape[0] // ds), interpolation=cv2.INTER_AREA)
    s_m1 = cv2.resize(mask1, (mask1.shape[1] // ds, mask1.shape[0] // ds), interpolation=cv2.INTER_NEAREST)
    s_t2 = cv2.resize(trace2, (trace2.shape[1] // ds, trace2.shape[0] // ds), interpolation=cv2.INTER_AREA)
    s_m2 = cv2.resize(mask2, (mask2.shape[1] // ds, mask2.shape[0] // ds), interpolation=cv2.INTER_NEAREST)

    ncc_map, ov_map = _masked_fft_ncc_2d(s_t1, s_m1, s_t2, s_m2)
    sh1, sw1 = s_t1.shape
    sh2, sw2 = s_t2.shape

    max_dy_ds = max(3, cfg.max_abs_dy_px // ds + 1)
    min_ov_area_ds = float(max(6000, (cfg.min_overlap_px // ds) * (sh1 * 0.42)))

    # Restrict sy to [-max_dy_ds, +max_dy_ds] and sx to [-0.20*sw2 .. sw1 - min_overlap_ds]
    sy_min = (sh2 - 1) - max_dy_ds
    sy_max = (sh2 - 1) + max_dy_ds + 1
    sx_min = max(0, (sw2 - 1) - int(0.20 * sw2))
    sx_max = min(ncc_map.shape[1], (sw2 - 1) + sw1 - max(25, cfg.min_overlap_px // ds))

    sub_ncc = ncc_map[sy_min:sy_max, sx_min:sx_max].copy()
    sub_ov = ov_map[sy_min:sy_max, sx_min:sx_max]
    # Mildly discount tiny edge-only overlaps so accidental 1-spike matches at the far edge never win
    area_weight = np.minimum(1.0, sub_ov / (min_ov_area_ds * 2.2))
    weighted_ncc = np.where(sub_ov >= min_ov_area_ds, sub_ncc * (0.75 + 0.25 * area_weight), -1.0)

    # Collect top 4 distinct dx candidates from Masked FFT NCC
    cand_seeds: list[tuple[int, int, str]] = []
    work_map = weighted_ncc.copy()
    for _ in range(4):
        flat_idx = int(np.argmax(work_map))
        wy, wx = divmod(flat_idx, work_map.shape[1])
        if float(work_map[wy, wx]) <= 0.05:
            break
        dx_cand = ((sx_min + wx) - (sw2 - 1)) * ds
        dy_cand = ((sy_min + wy) - (sh2 - 1)) * ds
        cand_seeds.append((int(dx_cand), int(dy_cand), "masked_fft_ncc"))
        # Suppress neighborhood +/- 25 px (at 2x = 12 px)
        work_map[:, max(0, wx - 15) : min(work_map.shape[1], wx + 16)] = -1.0

    onnx_dx: float | None = None
    onnx_dy: float | None = None
    onnx_inl = 0
    onnx_conf = 0.0
    if cfg.use_onnx_matcher and onnx_matcher is not None:
        odx, ody, oconf, oinl = onnx_matcher.match_translation(
            trace1, mask1, trace2, mask2, max_abs_dy=float(cfg.max_abs_dy_px + 2)
        )
        if oinl >= 3:
            onnx_dx = round(odx, 2)
            onnx_dy = round(ody, 2)
            onnx_inl = oinl
            onnx_conf = oconf
            cand_seeds.append((int(round(odx)), int(round(ody)), "onnx_patch_ransac"))

    if not cand_seeds:
        cand_seeds.append((int(0.45 * trace1.shape[1]), 0, "fallback_default"))

    # Full-resolution joint refinement (trace NCC + green-grid phase NCC) around each candidate seed!
    best_joint = -2.0
    best_dx = float(cand_seeds[0][0])
    best_dy = float(cand_seeds[0][1])
    best_trace_ncc = 0.0
    best_grid_ncc = 0.0
    best_ov_w = 0
    best_method = cand_seeds[0][2]

    evaluated: set[tuple[int, int]] = set()
    for s_dx, s_dy, m_name in cand_seeds:
        # Coarse 2px step in +/- 6px, then 1px step in +/- 2px around local optimum
        local_best = (-2.0, s_dx, s_dy, 0.0, 0.0, 0)
        for ddy in range(-4, 5, 2):
            cy = int(np.clip(s_dy + ddy, -cfg.max_abs_dy_px, cfg.max_abs_dy_px))
            for ddx in range(-6, 7, 2):
                cx = s_dx + ddx
                if (cx, cy) in evaluated:
                    continue
                evaluated.add((cx, cy))
                tncc, ov_w = _eval_fullres_overlap(trace1, mask1, trace2, mask2, cx, cy)
                if ov_w < cfg.min_overlap_px or tncc < -0.5:
                    continue
                gncc, _ = _eval_fullres_overlap(grid1, mask1, grid2, mask2, cx, cy)
                # Bonus if ONNX RANSAC agrees within 8 px!
                onnx_bonus = 0.0
                if onnx_dx is not None and abs(cx - onnx_dx) <= 8.0 and abs(cy - (onnx_dy or 0.0)) <= 6.0:
                    onnx_bonus = 0.25 * onnx_conf
                ov_factor = min(1.0, ov_w / 550.0) ** 0.35
                joint = (0.78 * tncc + 0.22 * max(0.0, gncc)) * ov_factor + onnx_bonus
                if joint > local_best[0]:
                    local_best = (joint, cx, cy, tncc, gncc, ov_w)

        _, bx, by, _, _, _ = local_best
        for cy in range(max(-cfg.max_abs_dy_px, by - 2), min(cfg.max_abs_dy_px, by + 2) + 1):
            for cx in range(bx - 2, bx + 3):
                tncc, ov_w = _eval_fullres_overlap(trace1, mask1, trace2, mask2, cx, cy)
                if ov_w < cfg.min_overlap_px:
                    continue
                gncc, _ = _eval_fullres_overlap(grid1, mask1, grid2, mask2, cx, cy)
                onnx_bonus = 0.0
                used = m_name
                if onnx_dx is not None and abs(cx - onnx_dx) <= 8.0 and abs(cy - (onnx_dy or 0.0)) <= 6.0:
                    onnx_bonus = 0.25 * onnx_conf
                    used = "hybrid_ncc+onnx"
                ov_factor = min(1.0, ov_w / 550.0) ** 0.35
                joint = (0.78 * tncc + 0.22 * max(0.0, gncc)) * ov_factor + onnx_bonus
                if joint > best_joint:
                    best_joint = joint
                    best_dx = float(cx)
                    best_dy = float(cy)
                    best_trace_ncc = tncc
                    best_grid_ncc = gncc
                    best_ov_w = ov_w
                    best_method = used

    conf = float(np.clip(0.80 * best_trace_ncc + 0.20 * max(0.0, best_grid_ncc), 0.0, 1.0))
    if onnx_dx is not None and abs(best_dx - onnx_dx) <= 8.0:
        conf = float(min(1.0, conf + 0.12 * onnx_conf))

    return {
        "dx": best_dx,
        "dy": best_dy,
        "overlap_width_px": best_ov_w,
        "trace_ncc": round(best_trace_ncc, 4),
        "grid_ncc": round(best_grid_ncc, 4),
        "onnx_dx": onnx_dx,
        "onnx_dy": onnx_dy,
        "onnx_inliers": onnx_inl,
        "confidence": round(conf, 4),
        "method_used": best_method,
    }


def _find_minimum_error_vertical_seam(cost: np.ndarray) -> np.ndarray:
    """Find optimal 8-connected top-to-bottom seam x(y) minimizing cumulative cost[y, x]."""
    h, w = cost.shape
    if w <= 1:
        return np.zeros(h, dtype=np.int32)

    dp = cost[0].astype(np.float64).copy()
    backtrack = np.zeros((h, w), dtype=np.int16)

    for y in range(1, h):
        # Vectorized min over (x-1, x, x+1)
        left = np.empty_like(dp)
        left[0] = 1e18
        left[1:] = dp[:-1]

        right = np.empty_like(dp)
        right[-1] = 1e18
        right[:-1] = dp[1:]

        stacked = np.vstack([left, dp, right])  # (3, w) -> offset -1, 0, +1
        choice = np.argmin(stacked, axis=0)
        backtrack[y] = choice.astype(np.int16) - 1
        dp = cost[y].astype(np.float64) + stacked[choice, np.arange(w)]

    seam_x = np.zeros(h, dtype=np.int32)
    seam_x[h - 1] = int(np.argmin(dp))
    for y in range(h - 1, 0, -1):
        seam_x[y - 1] = int(np.clip(seam_x[y] + int(backtrack[y, seam_x[y]]), 0, w - 1))
    return seam_x


def _blend_frame_onto_canvas(
    canvas_bgr: np.ndarray,
    canvas_mask: np.ndarray,
    frame_bgr: np.ndarray,
    frame_trace: np.ndarray,
    frame_mask: np.ndarray,
    x_off: int,
    y_off: int,
    feather_px: int = 9,
) -> int | None:
    """Composite ``frame_bgr`` onto ``canvas_bgr`` at ``(x_off, y_off)`` using DP minimum-error seam."""
    ch, cw = canvas_bgr.shape[:2]
    fh, fw = frame_bgr.shape[:2]

    cx0 = max(0, x_off)
    cx1 = min(cw, x_off + fw)
    cy0 = max(0, y_off)
    cy1 = min(ch, y_off + fh)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    fx0, fx1 = cx0 - x_off, cx1 - x_off
    fy0, fy1 = cy0 - y_off, cy1 - y_off

    c_roi = canvas_bgr[cy0:cy1, cx0:cx1]
    cm_roi = canvas_mask[cy0:cy1, cx0:cx1] > 0
    f_roi = frame_bgr[fy0:fy1, fx0:fx1]
    ft_roi = frame_trace[fy0:fy1, fx0:fx1]
    fm_roi = frame_mask[fy0:fy1, fx0:fx1] > 0

    only_f = fm_roi & (~cm_roi)
    both = fm_roi & cm_roi

    if not np.any(both):
        c_roi[only_f] = f_roi[only_f]
        canvas_mask[cy0:cy1, cx0:cx1][only_f] = 255
        return None

    # Identify horizontal span where both frames have valid pixels
    both_cols = np.where(both.any(axis=0))[0]
    bx0, bx1 = int(both_cols[0]), int(both_cols[-1]) + 1

    if bx1 - bx0 < 8:
        c_roi[only_f] = f_roi[only_f]
        canvas_mask[cy0:cy1, cx0:cx1][only_f] = 255
        return cx0 + bx0

    c_ov = c_roi[:, bx0:bx1].astype(np.float32)
    f_ov = f_roi[:, bx0:bx1].astype(np.float32)
    both_ov = both[:, bx0:bx1]
    cm_ov = cm_roi[:, bx0:bx1]
    fm_ov = fm_roi[:, bx0:bx1]

    # Color + trace difference cost; strongly penalize cutting outside `both_ov` or right on edge
    rgb_diff = np.mean(np.abs(c_ov - f_ov), axis=2)
    trace_pen = 1.5 * ft_roi[:, bx0:bx1].astype(np.float32)
    ov_w = bx1 - bx0
    x_norm = np.linspace(-1.0, 1.0, ov_w, dtype=np.float32).reshape(1, -1)
    edge_bias = 18.0 * (x_norm ** 4)  # Prefer seam near interior of overlap away from cut boundaries

    cost = rgb_diff + 0.35 * trace_pen + edge_bias
    cost = np.where(both_ov, cost, 5000.0)
    seam_rel = _find_minimum_error_vertical_seam(cost)

    # Build smooth alpha weight for frame_bgr (0 left of seam, 1 right of seam, cosine ramp of width feather_px)
    half_f = max(1.0, 0.5 * feather_px)
    xx = np.arange(ov_w, dtype=np.float32).reshape(1, -1)
    dist = xx - seam_rel.reshape(-1, 1).astype(np.float32)
    alpha_f = 0.5 * (1.0 + np.sin(np.clip(dist / half_f, -1.0, 1.0) * (0.5 * np.pi)))

    # Enforce strict validity where only one image is valid!
    alpha_f = np.where(cm_ov & (~fm_ov), 0.0, alpha_f)
    alpha_f = np.where(fm_ov & (~cm_ov), 1.0, alpha_f)
    alpha_f = np.where((~cm_ov) & (~fm_ov), 0.0, alpha_f)

    blended_ov = np.clip(
        c_ov * (1.0 - alpha_f[:, :, None]) + f_ov * alpha_f[:, :, None], 0.0, 255.0
    ).astype(np.uint8)

    # First copy pure frame-only columns outside [bx0:bx1], then write blended overlap
    c_roi[only_f] = f_roi[only_f]
    c_roi[:, bx0:bx1] = np.where(
        (cm_ov | fm_ov)[:, :, None], blended_ov, c_roi[:, bx0:bx1]
    )
    canvas_mask[cy0:cy1, cx0:cx1] = np.where(
        cm_roi | fm_roi, 255, canvas_mask[cy0:cy1, cx0:cx1]
    )

    return int(cx0 + bx0 + int(np.median(seam_rel)))


def run_stitch_pipeline(
    paths: DatasetPaths,
    stitch_cfg: StitchConfig | None = None,
    rect_cfg: RectifyConfig | None = None,
    enh_cfg: EnhanceConfig | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Pipeline 3: match all rectified frames, blend seams, and write panorama PNG & WebP."""
    paths.ensure_work_dirs()
    manifest = DatasetManifest.load(paths.manifest_path)
    stitch_cfg = stitch_cfg or StitchConfig()
    rect_cfg = rect_cfg or RectifyConfig(target_px_per_cm=manifest.target_px_per_cm)
    enh_cfg = enh_cfg or EnhanceConfig()

    if models_dir is None:
        models_dir = paths.data_root.parent / "models"

    # Ensure all frames have rectified outputs on disk (streaming one at a time)
    missing_indices: list[int] = []
    for entry in manifest.frames:
        stem = Path(entry.filename).stem
        if not (paths.rectified_dir / f"{stem}_rectified.png").exists():
            missing_indices.append(entry.index)
    if missing_indices:
        run_rectify_pipeline(
            paths, rect_cfg=rect_cfg, enh_cfg=enh_cfg, frame_indices=missing_indices
        )

    onnx_matcher = (
        OnnxPatchMatcher(models_dir) if stitch_cfg.use_onnx_matcher else None
    )

    pair_matches: list[PairMatchMetadata] = []
    frame_widths: list[int] = []
    frame_heights: list[int] = []

    prev_trace: np.ndarray | None = None
    prev_grid: np.ndarray | None = None
    prev_mask: np.ndarray | None = None

    # Pass 1: Pairwise registration (holds only 2 grayscale frames in RAM at once!)
    for idx, entry in enumerate(manifest.frames):
        stem = Path(entry.filename).stem
        cur_trace = cv2.imread(
            str(paths.rectified_dir / f"{stem}_rect_trace.png"), cv2.IMREAD_GRAYSCALE
        )
        cur_grid = cv2.imread(
            str(paths.rectified_dir / f"{stem}_rect_grid.png"), cv2.IMREAD_GRAYSCALE
        )
        cur_mask = cv2.imread(
            str(paths.rectified_dir / f"{stem}_rect_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        if cur_trace is None or cur_grid is None or cur_mask is None:
            raise FileNotFoundError(f"Missing rectified layers for {stem}")

        fh, fw = cur_trace.shape
        frame_widths.append(fw)
        frame_heights.append(fh)

        if idx > 0:
            prev_entry = manifest.frames[idx - 1]
            assert prev_trace is not None and prev_grid is not None and prev_mask is not None

            if entry.manual_dx_override is not None:
                dx = float(entry.manual_dx_override)
                dy = float(entry.manual_dy_override or 0.0)
                tncc, ov_w = _eval_fullres_overlap(
                    prev_trace, prev_mask, cur_trace, cur_mask, int(round(dx)), int(round(dy))
                )
                gncc, _ = _eval_fullres_overlap(
                    prev_grid, prev_mask, cur_grid, cur_mask, int(round(dx)), int(round(dy))
                )
                pm = PairMatchMetadata(
                    left_index=prev_entry.index,
                    right_index=entry.index,
                    left_filename=prev_entry.filename,
                    right_filename=entry.filename,
                    dx=dx,
                    dy=dy,
                    overlap_width_px=ov_w,
                    trace_ncc=round(tncc, 4),
                    grid_ncc=round(gncc, 4),
                    onnx_dx=None,
                    onnx_dy=None,
                    onnx_inliers=0,
                    confidence=1.0,
                    method_used="manual_override",
                    manual_override=True,
                )
            else:
                m_res = match_rectified_pair(
                    prev_trace,
                    prev_grid,
                    prev_mask,
                    cur_trace,
                    cur_grid,
                    cur_mask,
                    cfg=stitch_cfg,
                    onnx_matcher=onnx_matcher,
                )
                pm = PairMatchMetadata(
                    left_index=prev_entry.index,
                    right_index=entry.index,
                    left_filename=prev_entry.filename,
                    right_filename=entry.filename,
                    dx=m_res["dx"],
                    dy=m_res["dy"],
                    overlap_width_px=m_res["overlap_width_px"],
                    trace_ncc=m_res["trace_ncc"],
                    grid_ncc=m_res["grid_ncc"],
                    onnx_dx=m_res["onnx_dx"],
                    onnx_dy=m_res["onnx_dy"],
                    onnx_inliers=m_res["onnx_inliers"],
                    confidence=m_res["confidence"],
                    method_used=m_res["method_used"],
                    manual_override=False,
                )
            pair_matches.append(pm)

        prev_trace, prev_grid, prev_mask = cur_trace, cur_grid, cur_mask

    del prev_trace, prev_grid, prev_mask

    (paths.matches_dir / "pair_matches.json").write_text(
        json.dumps([asdict(p) for p in pair_matches], indent=2) + "\n", encoding="utf-8"
    )

    # Compute global integer positions; damp cumulative Y drift by centering Y offsets around 0
    # since every frame is ALREADY canonically anchored to the exact separator & 44 horizontal lines!
    n_frames = len(manifest.frames)
    global_x = np.zeros(n_frames, dtype=np.float64)
    raw_cum_y = np.zeros(n_frames, dtype=np.float64)
    for i, pm in enumerate(pair_matches):
        global_x[i + 1] = global_x[i] + pm.dx
        # High-pass damp cumulative dy so small +/-1px steps never accumulate into vertical drift!
        raw_cum_y[i + 1] = 0.5 * raw_cum_y[i] + pm.dy

    global_x -= np.min(global_x)
    gx_int = np.round(global_x).astype(np.int32)
    gy_int = np.round(raw_cum_y - np.median(raw_cum_y)).astype(np.int32)

    px_per_cm = rect_cfg.target_px_per_cm
    canon_h = int(round(CANONICAL_TOTAL_HEIGHT_CM * px_per_cm))
    pano_w = int(np.max(gx_int + np.array(frame_widths, dtype=np.int32)))

    # Allocate single panorama canvas (e.g. ~8500 x 1420 x 3 = ~36 MB RAM!)
    canvas_bgr = np.full((canon_h, pano_w, 3), 252, dtype=np.uint8)
    canvas_mask = np.zeros((canon_h, pano_w), dtype=np.uint8)
    seam_x_positions: list[int | None] = []

    # Pass 2: Stream one rectified color frame at a time onto panorama canvas
    for idx, entry in enumerate(manifest.frames):
        stem = Path(entry.filename).stem
        f_bgr = cv2.imread(
            str(paths.rectified_dir / f"{stem}_rectified.png"), cv2.IMREAD_COLOR
        )
        f_trace = cv2.imread(
            str(paths.rectified_dir / f"{stem}_rect_trace.png"), cv2.IMREAD_GRAYSCALE
        )
        f_mask = cv2.imread(
            str(paths.rectified_dir / f"{stem}_rect_mask.png"), cv2.IMREAD_GRAYSCALE
        )
        assert f_bgr is not None and f_trace is not None and f_mask is not None

        if idx == 0:
            x0 = int(gx_int[0])
            y0 = int(gy_int[0])
            _blend_frame_onto_canvas(
                canvas_bgr, canvas_mask, f_bgr, f_trace, f_mask, x0, y0, feather_px=stitch_cfg.feather_width_px
            )
        else:
            seam_x = _blend_frame_onto_canvas(
                canvas_bgr,
                canvas_mask,
                f_bgr,
                f_trace,
                f_mask,
                int(gx_int[idx]),
                int(gy_int[idx]),
                feather_px=stitch_cfg.feather_width_px,
            )
            seam_x_positions.append(seam_x)

        del f_bgr, f_trace, f_mask

    # Trim any completely empty columns on far left/right
    valid_pano_cols = np.where(canvas_mask.any(axis=0))[0]
    trim_x0 = int(valid_pano_cols[0]) if len(valid_pano_cols) else 0
    trim_x1 = int(valid_pano_cols[-1]) + 1 if len(valid_pano_cols) else pano_w
    canvas_bgr = canvas_bgr[:, trim_x0:trim_x1]
    canvas_mask = canvas_mask[:, trim_x0:trim_x1]
    final_h, final_w = canvas_bgr.shape[:2]

    png_path: Path | None = None
    webp_path: Path | None = None
    webp_scale = 1.0

    if stitch_cfg.export_png:
        png_path = paths.output_dir / "panorama.png"
        cv2.imwrite(
            str(png_path), canvas_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6]
        )

    if stitch_cfg.export_webp:
        webp_path = paths.output_dir / "panorama.webp"
        rgb_for_webp = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        if final_w > WEBP_MAX_DIMENSION_PX or final_h > WEBP_MAX_DIMENSION_PX:
            webp_scale = min(
                (WEBP_MAX_DIMENSION_PX - 3) / float(final_w),
                (WEBP_MAX_DIMENSION_PX - 3) / float(final_h),
            )
            new_w = max(1, int(round(final_w * webp_scale)))
            new_h = max(1, int(round(final_h * webp_scale)))
            rgb_for_webp = cv2.resize(rgb_for_webp, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(rgb_for_webp)
        pil_img.save(
            str(webp_path),
            format="WEBP",
            quality=stitch_cfg.webp_quality,
            lossless=stitch_cfg.webp_lossless,
            method=4,
        )
        del pil_img, rgb_for_webp

    duration_minutes = (final_w / float(px_per_cm)) / manifest.paper_speed_cm_per_min
    report = {
        "dataset_id": manifest.dataset_id,
        "split": manifest.split,
        "num_frames": n_frames,
        "panorama_width_px": final_w,
        "panorama_height_px": final_h,
        "target_px_per_cm": px_per_cm,
        "paper_speed_cm_per_min": manifest.paper_speed_cm_per_min,
        "px_per_minute": round(px_per_cm * manifest.paper_speed_cm_per_min, 3),
        "estimated_strip_length_cm": round(final_w / float(px_per_cm), 2),
        "estimated_duration_minutes": round(duration_minutes, 2),
        "canonical_calibration": {
            "fhr_top_210bpm_y_px": round(FHR_S_TOP_CM * px_per_cm, 2),
            "fhr_bottom_50bpm_y_px": round(FHR_S_BOT_CM * px_per_cm, 2),
            "fhr_px_per_bpm": round((FHR_S_BOT_CM - FHR_S_TOP_CM) * px_per_cm / 160.0, 4),
            "toco_top_100_y_px": round(TOCO_S_TOP_CM * px_per_cm, 2),
            "toco_bottom_0_y_px": round(TOCO_S_BOT_CM * px_per_cm, 2),
            "toco_px_per_unit": round((TOCO_S_BOT_CM - TOCO_S_TOP_CM) * px_per_cm / 100.0, 4),
        },
        "exports": {
            "png": str(png_path) if png_path else None,
            "webp": str(webp_path) if webp_path else None,
            "webp_scale_factor": round(webp_scale, 5),
        },
        "frame_placements": [
            {
                "frame_index": manifest.frames[i].index,
                "filename": manifest.frames[i].filename,
                "x_px": int(gx_int[i] - trim_x0),
                "y_px": int(gy_int[i]),
                "width_px": frame_widths[i],
                "height_px": frame_heights[i],
            }
            for i in range(n_frames)
        ],
        "seam_x_positions_px": [
            (sx - trim_x0) if sx is not None else None for sx in seam_x_positions
        ],
        "pair_matches": [asdict(pm) for pm in pair_matches],
    }

    meta_path = paths.output_dir / "stitch_metadata.json"
    meta_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
