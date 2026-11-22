"""Pipeline 1: Coarse orientation normalization, finger/border masking, flat-field illumination
correction, and green-grid vs. black-trace channel separation.

Designed for low RAM and fast CPU execution by estimating illumination and masks on downsampled
proxies and streaming one frame at a time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ctg_converter.dataset.manifest import DatasetManifest, DatasetPaths


@dataclass
class EnhanceConfig:
    """Tunable parameters for CTG image enhancement and layer separation."""

    white_target: float = 250.0
    illumination_downsample: int = 8
    illumination_close_px: int = 21
    illumination_sigma_px: float = 18.0
    grid_gain: float = 1.45
    trace_gain: float = 1.35
    green_absorption_in_g: float = 0.28
    sharpen_amount: float = 0.35
    auto_orient_90: bool = True


@dataclass
class EnhanceResultMetadata:
    """Per-frame diagnostics saved alongside enhanced outputs."""

    frame_index: int
    filename: str
    applied_rotation_cw_deg: int
    oriented_width: int
    oriented_height: int
    valid_paper_fraction: float
    mean_grid_strength: float
    mean_trace_strength: float


def rotate_cw(image: np.ndarray, angle_cw_deg: int) -> np.ndarray:
    """Rotate an image clockwise by 0, 90, 180, or 270 degrees."""
    angle = angle_cw_deg % 360
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported coarse rotation angle: {angle_cw_deg}")


def detect_coarse_orientation_cw(bgr: np.ndarray) -> int:
    """Determine the clockwise rotation (0, 90, 180, 270) that puts the CTG upright.

    First removes any skew angle modulo 90 degrees so the narrow white separator strip
    between FHR (16 squares = 8.0 cm) and Toco (10 squares = 5.0 cm) is horizontally sharp,
    then scores all four 90-degree orientations using separator dip position, ribbon border
    continuity, and Toco-vs-FHR green density.
    """
    h, w = bgr.shape[:2]
    sc = 640.0 / max(h, w)
    sm = cv2.resize(
        bgr,
        (max(32, int(round(w * sc))), max(32, int(round(h * sc)))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    bg = cv2.GaussianBlur(sm, (0, 0), 15)
    norm = sm / np.maximum(bg, 25.0)
    b, g, r = norm[:, :, 0], norm[:, :, 1], norm[:, :, 2]

    gray = cv2.cvtColor(sm.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    local_bright = (
        cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 90
    )
    green = np.clip(g - 0.5 * (r + b) - 0.010, 0.0, 0.35)
    green_hp = np.maximum(0.0, green - cv2.GaussianBlur(green, (0, 0), 3.5))
    green_hp[~local_bright] = 0.0

    act = cv2.GaussianBlur(green_hp, (0, 0), 14.0)
    ref = (
        float(np.percentile(act[local_bright], 85))
        if np.any(local_bright)
        else float(np.max(act))
    )
    if ref <= 1e-6:
        return 0

    raw_gm = cv2.morphologyEx(
        (act > ref * 0.22).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    nl, lbl, st, _ = cv2.connectedComponentsWithStats(raw_gm)
    gm = (
        (lbl == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))))
        if nl > 1
        else (raw_gm > 0)
    )

    sh, sw = green_hp.shape
    ys, xs = np.where(gm)
    yc = int(np.median(ys)) if len(ys) else sh // 2
    xc = int(np.median(xs)) if len(xs) else sw // 2
    rad = min(130, sh // 2 - 4, sw // 2 - 4)
    yc = int(np.clip(yc, rad, sh - rad))
    xc = int(np.clip(xc, rad, sw - rad))
    patch = green_hp[yc - rad : yc + rad, xc - rad : xc + rad]

    best_a, best_e = 0.0, -1.0
    for a in np.linspace(-25.0, 25.0, 51):
        mat = cv2.getRotationMatrix2D((rad, rad), float(a), 1.0)
        wp = cv2.warpAffine(patch, mat, (2 * rad, 2 * rad), flags=cv2.INTER_LINEAR)
        inner = wp[rad // 3 : 2 * rad - rad // 3, rad // 3 : 2 * rad - rad // 3]
        energy = float(np.var(inner.mean(axis=1)) + np.var(inner.mean(axis=0)))
        if energy > best_e:
            best_e, best_a = energy, float(a)

    mat_full = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), best_a, 1.0)
    dg = cv2.warpAffine(
        np.where(gm, green_hp, 0.0), mat_full, (sw, sh), flags=cv2.INTER_LINEAR
    )
    dm = (
        cv2.warpAffine(
            gm.astype(np.uint8), mat_full, (sw, sh), flags=cv2.INTER_NEAREST
        )
        > 0
    )

    best_ang = 0
    best_score = -1e9
    for ang in (0, 90, 180, 270):
        rm_raw = rotate_cw(gm.astype(np.uint8), ang) > 0
        lr_touch = 0.5 * (
            float(rm_raw[:, :8].any(axis=1).mean())
            + float(rm_raw[:, -8:].any(axis=1).mean())
        )
        tb_touch = 0.5 * (
            float(rm_raw[:8, :].any(axis=0).mean())
            + float(rm_raw[-8:, :].any(axis=0).mean())
        )
        ribbon_score = lr_touch - tb_touch

        rg = rotate_cw(dg, ang)
        rm = rotate_cw(dm.astype(np.uint8), ang) > 0
        rys, rxs = np.where(rm)
        if len(rys) < 200:
            continue
        y0, y1 = int(np.percentile(rys, 2)), int(np.percentile(rys, 98))
        x0, x1 = int(np.percentile(rxs, 15)), int(np.percentile(rxs, 85))
        if y1 - y0 < 60 or x1 - x0 < 60:
            continue

        crop_g = rg[y0:y1, x0:x1]
        crop_m = rm[y0:y1, x0:x1]
        row_prof = crop_g.sum(axis=1) / np.maximum(crop_m.sum(axis=1), 10)

        n = len(row_prof)
        env = cv2.GaussianBlur(
            row_prof.reshape(-1, 1), (1, max(3, (n // 55) | 1)), 0
        ).ravel()
        broad = cv2.GaussianBlur(
            row_prof.reshape(-1, 1), (1, max(15, (n // 5) | 1)), 0
        ).ravel()
        detrended = env / np.maximum(broad, np.percentile(broad, 25) + 1e-6)

        v_lo, v_hi = int(0.54 * n), int(0.70 * n)
        m_lo, m_hi = int(0.30 * n), int(0.46 * n)
        val_min = float(np.min(detrended[v_lo:v_hi]))
        mir_min = float(np.min(detrended[m_lo:m_hi]))
        dip = 1.0 - val_min
        asym = mir_min - val_min

        upper_raw = float(np.median(env[int(0.12 * n) : int(0.48 * n)]))
        lower_raw = float(np.median(env[int(0.70 * n) : int(0.92 * n)]))
        toco_gt_fhr = (lower_raw - upper_raw) / (lower_raw + upper_raw + 1e-6)

        score = 2.5 * dip + 2.5 * asym + 1.5 * ribbon_score + 1.2 * toco_gt_fhr
        if score > best_score:
            best_score = score
            best_ang = ang

    return best_ang


def compute_valid_paper_mask(bgr: np.ndarray) -> np.ndarray:
    """Segment the valid CTG paper region, excluding fingers, dark table edges, and background."""
    h, w = bgr.shape[:2]
    ds = 4
    sh, sw = max(16, h // ds), max(16, w // ds)
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    # Smooth local maximum brightness so thin dark traces inside paper stay bright
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    local_paper_luma = cv2.dilate(gray, dilate_k)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
    l_chan, a_chan, b_chan = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Local illumination normalisation for robust skin / colored-desk rejection
    bg = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 31)
    norm = np.clip(small.astype(np.float32) / np.maximum(bg, 25.0) * 240.0, 0, 255).astype(
        np.uint8
    )
    lab_norm = cv2.cvtColor(norm, cv2.COLOR_BGR2Lab)
    a_norm = lab_norm[:, :, 1]

    # CTG paper is bright (white or green grid, a_norm <= 134);
    # fingers/skin have noticeably positive redness (a > 136..140) and warm b;
    # green/wooden/dark desk outside paper either has low local_paper_luma or extreme saturation.
    adaptive_luma_thresh = max(75.0, float(np.percentile(local_paper_luma, 15)) * 0.62)
    not_dark_bg = local_paper_luma >= adaptive_luma_thresh
    not_skin = ~((a_chan > 138) & (b_chan > 134)) & (a_norm < 137)
    # Also reject heavily saturated green/blue/colored desk backgrounds outside paper
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    not_vivid_bg = hsv[:, :, 1] < 135

    raw_mask = (not_dark_bg & not_skin & not_vivid_bg).astype(np.uint8) * 255

    # Close interior ink/text holes first, then open to detach border intrusions
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # Keep the largest connected component (the CTG paper sheet)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Fill any remaining interior holes inside the paper (e.g. thick black registration marks)
    inv = cv2.bitwise_not(mask)
    h_m, w_m = inv.shape
    flood = inv.copy()
    ff_mask = np.zeros((h_m + 2, w_m + 2), dtype=np.uint8)
    for bx in range(w_m):
        if flood[0, bx] == 255:
            cv2.floodFill(flood, ff_mask, (bx, 0), 128)
        if flood[h_m - 1, bx] == 255:
            cv2.floodFill(flood, ff_mask, (bx, h_m - 1), 128)
    for by in range(h_m):
        if flood[by, 0] == 255:
            cv2.floodFill(flood, ff_mask, (0, by), 128)
        if flood[by, w_m - 1] == 255:
            cv2.floodFill(flood, ff_mask, (w_m - 1, by), 128)
    interior_holes = flood == 255
    mask[interior_holes] = 255

    # Re-apply skin exclusion near borders so a finger overlapping the edge stays masked out
    border_band = np.zeros_like(mask, dtype=bool)
    margin_y, margin_x = max(8, sh // 7), max(8, sw // 7)
    border_band[:margin_y, :] = True
    border_band[-margin_y:, :] = True
    border_band[:, :margin_x] = True
    border_band[:, -margin_x:] = True
    finger_intrusion = border_band & ((a_chan > 136) & (b_chan > 133))
    if np.any(finger_intrusion):
        dil_finger = cv2.dilate(
            finger_intrusion.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        mask[dil_finger > 0] = 0

    # Slight erosion to avoid dark edge shadow fringes
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    full_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return full_mask


def _estimate_flat_field_white(
    bgr: np.ndarray,
    valid_mask: np.ndarray,
    cfg: EnhanceConfig,
) -> np.ndarray:
    """Estimate smooth per-channel paper white illumination field at full resolution."""
    h, w = bgr.shape[:2]
    ds = max(2, cfg.illumination_downsample)
    sh, sw = max(16, h // ds), max(16, w // ds)

    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    sm_mask = cv2.resize(valid_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    # Morphological dilation/closing on downsampled image bridges across thin black traces & grid lines
    k_size = max(5, cfg.illumination_close_px | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # Replace invalid border pixels with median valid paper color before smoothing so dark borders don't bleed in
    if np.any(sm_mask):
        med_bgr = np.median(small[sm_mask], axis=0)
    else:
        med_bgr = np.array([200.0, 200.0, 200.0], dtype=np.float32)

    filled = small.copy()
    filled[~sm_mask] = med_bgr

    white_small = np.empty_like(filled)
    for c in range(3):
        closed = cv2.morphologyEx(filled[:, :, c], cv2.MORPH_CLOSE, kernel)
        white_small[:, :, c] = cv2.GaussianBlur(
            closed, (0, 0), cfg.illumination_sigma_px
        )

    white_full = cv2.resize(white_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(white_full, 25.0)


def enhance_ctg_frame(
    bgr: np.ndarray,
    cfg: EnhanceConfig | None = None,
    rotation_override_deg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run Pipeline 1 on a single BGR frame.

    Returns
    -------
    enhanced_bgr : uint8 (H, W, 3)
        Illumination-normalized, white-balanced, crisp RGB/BGR CTG image.
    grid_layer : uint8 (H, W)
        Isolated green grid intensity (0 = paper/black trace, 255 = strong green grid line).
    trace_layer : uint8 (H, W)
        Isolated black/gray ink trace & text intensity (0 = paper/green grid, 255 = deep black ink).
    valid_mask : uint8 (H, W)
        255 inside clean CTG paper, 0 on fingers / desk / outside paper.
    applied_rotation_cw_deg : int
        Coarse clockwise rotation applied (0, 90, 180, or 270).
    """
    cfg = cfg or EnhanceConfig()

    if rotation_override_deg is not None:
        rot_deg = rotation_override_deg % 360
    elif cfg.auto_orient_90:
        rot_deg = detect_coarse_orientation_cw(bgr)
    else:
        rot_deg = 0

    oriented = rotate_cw(bgr, rot_deg)
    valid_mask = compute_valid_paper_mask(oriented)
    white_field = _estimate_flat_field_white(oriented, valid_mask, cfg)

    # Flat-field illumination normalization: paper white maps uniformly to cfg.white_target
    norm_bgr = oriented.astype(np.float32) / white_field * cfg.white_target
    norm_bgr = np.clip(norm_bgr, 0.0, 255.0)

    b = norm_bgr[:, :, 0]
    g = norm_bgr[:, :, 1]
    r = norm_bgr[:, :, 2]

    # Green grid ink absorbs R and B strongly while keeping G relatively high
    green_chroma = np.maximum(0.0, g - 0.5 * (r + b))
    # Suppress tiny sensor noise on plain white paper via soft deadzone
    green_clean = np.maximum(0.0, green_chroma - 3.0)

    # Black ink absorbs ALL channels including G; subtract minor G absorption of green ink
    black_density = np.maximum(
        0.0, (cfg.white_target - 2.0 - g) - cfg.green_absorption_in_g * green_clean
    )
    black_clean = np.maximum(0.0, black_density - 4.0)

    # Gentle unsharp mask on black trace/text to crisp up thin FHR spikes and timestamps
    if cfg.sharpen_amount > 0:
        blur_black = cv2.GaussianBlur(black_clean, (0, 0), 1.2)
        black_clean = np.clip(
            black_clean + cfg.sharpen_amount * (black_clean - blur_black), 0.0, 255.0
        )

    # Build isolated 8-bit layers (masked to valid paper)
    valid_bool = valid_mask > 0
    grid_u8 = np.clip(green_clean * 3.8, 0.0, 255.0).astype(np.uint8)
    trace_u8 = np.clip(black_clean * 1.45, 0.0, 255.0).astype(np.uint8)
    grid_u8[~valid_bool] = 0
    trace_u8[~valid_bool] = 0

    # Reconstruct crisp visual BGR image on pure neutral paper background
    paper_base = 252.0
    g_boost = green_clean * cfg.grid_gain
    k_boost = black_clean * cfg.trace_gain

    out_r = np.clip(paper_base - 1.15 * g_boost - k_boost, 0.0, 255.0)
    out_g = np.clip(paper_base - 0.18 * g_boost - k_boost, 0.0, 255.0)
    out_b = np.clip(paper_base - 0.95 * g_boost - k_boost, 0.0, 255.0)

    enhanced_bgr = np.stack([out_b, out_g, out_r], axis=-1).astype(np.uint8)
    # Keep dimmed context outside valid_mask so human inspectors in GUI can see masked borders
    if not np.all(valid_bool):
        dimmed = (oriented.astype(np.float32) * 0.45).astype(np.uint8)
        enhanced_bgr = np.where(valid_bool[:, :, None], enhanced_bgr, dimmed)

    return enhanced_bgr, grid_u8, trace_u8, valid_mask, rot_deg


def run_enhance_pipeline(
    paths: DatasetPaths,
    cfg: EnhanceConfig | None = None,
    frame_indices: list[int] | None = None,
) -> list[EnhanceResultMetadata]:
    """Stream all (or selected) frames of a dataset through Pipeline 1 to disk."""
    cfg = cfg or EnhanceConfig()
    paths.ensure_work_dirs()
    manifest = DatasetManifest.load(paths.manifest_path)

    results: list[EnhanceResultMetadata] = []
    selected = set(frame_indices) if frame_indices is not None else None

    for entry in manifest.frames:
        if selected is not None and entry.index not in selected:
            continue

        src_path = paths.raw_dir / entry.filename
        bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {src_path}")

        enhanced_bgr, grid_u8, trace_u8, valid_mask, rot_deg = enhance_ctg_frame(
            bgr,
            cfg=cfg,
            rotation_override_deg=entry.rotation_override_deg,
        )

        stem = Path(entry.filename).stem
        cv2.imwrite(str(paths.enhanced_dir / f"{stem}_enhanced.png"), enhanced_bgr)
        cv2.imwrite(str(paths.enhanced_dir / f"{stem}_grid.png"), grid_u8)
        cv2.imwrite(str(paths.enhanced_dir / f"{stem}_trace.png"), trace_u8)
        cv2.imwrite(str(paths.enhanced_dir / f"{stem}_mask.png"), valid_mask)

        oh, ow = enhanced_bgr.shape[:2]
        meta = EnhanceResultMetadata(
            frame_index=entry.index,
            filename=entry.filename,
            applied_rotation_cw_deg=rot_deg,
            oriented_width=ow,
            oriented_height=oh,
            valid_paper_fraction=round(float((valid_mask > 0).mean()), 4),
            mean_grid_strength=round(float(grid_u8[valid_mask > 0].mean()) if np.any(valid_mask > 0) else 0.0, 3),
            mean_trace_strength=round(float(trace_u8[valid_mask > 0].mean()) if np.any(valid_mask > 0) else 0.0, 3),
        )
        (paths.enhanced_dir / f"{stem}_meta.json").write_text(
            json.dumps(asdict(meta), indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(meta)

        # Free per-frame arrays immediately for low peak RAM
        del bgr, enhanced_bgr, grid_u8, trace_u8, valid_mask

    (paths.enhanced_dir / "summary.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    return results
