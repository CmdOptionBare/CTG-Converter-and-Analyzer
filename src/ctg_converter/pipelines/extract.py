"""Pipeline 4: Calibrated Image-to-Value Curve Extraction (FHR, MHR, Toco, Kineto)."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ctg_converter.dataset.manifest import DatasetManifest, DatasetPaths
from ctg_converter.pipelines.rectify import (
    FHR_S_BOT_CM,
    FHR_S_TOP_CM,
    TOCO_S_BOT_CM,
    TOCO_S_TOP_CM,
)
from ctg_converter.pipelines.stitch import StitchConfig, run_stitch_pipeline
from ctg_converter.pipelines.validate import find_time_stamps


@dataclass
class ExtractConfig:
    """Configuration for calibrated curve extraction."""

    dt_sec: float = 1.0
    started_at_iso: str = "2026-11-22T11:10:12"
    session_id: str | None = None
    session_title: str | None = None
    device_name: str = "Philips M2702A"
    paper_label: str = "M1911A · 1 cm/min"
    write_overlay_png: bool = True


def _extract_black_ink_from_bgr(pano_bgr: np.ndarray) -> np.ndarray:
    """Fallback black-ink extraction from color panorama if ``panorama_trace.png`` is absent."""
    b, g, r = cv2.split(pano_bgr)
    gf = g.astype(np.float32)
    rb = 0.5 * (r.astype(np.float32) + b.astype(np.float32))
    green_excess = np.maximum(0.0, gf - np.maximum(r, b).astype(np.float32))
    neutral_lum = np.maximum(gf, rb + 0.65 * green_excess)
    return np.clip(244.0 - neutral_lum, 0.0, 255.0).astype(np.uint8)


# 100 mmHg = 13.33 kPa. On this strip the green "12" kPa label sits on the
# 90 line of the 0-100 grid and "0" sits on the 0 line, so 100 units = 13.33 kPa.
TOCO_KPA_PER_PCT = 0.1333

# A column whose tracked y has no ink this dark, for longer than _INK_BRIDGE_PX,
# is a real gap (paper edge, pen up, blank leader) and is exported as null.
_INK_MIN = 35
_INK_TOL_PX = 3
_INK_BRIDGE_PX = 8


def _prepare_curve_reward(band_u8: np.ndarray, px_per_cm: int) -> tuple[np.ndarray, np.ndarray]:
    """Suppress printed text while preserving continuous pen strokes.

    A plain connected-component test on the whole band fails: the grid joins the
    pen, the stamps and the scale digits into one component. Word-shaped blocks
    that are still separate (a label, a short stamp stroke) are down-weighted.
    The repeated header line and the vertical time stamp are handled by
    ``_downweight_header_text`` and ``_downweight_left_stamp``.

    Returns ``(reward_float32, long_ribbon_mask_bool)``.
    """
    bw = (band_u8 >= 26).astype(np.uint8)
    # Bridge dotted pen gaps. Do not open horizontally: on the spiky parts of this
    # strip a width-3 opening deletes the fetal stroke and leaves the grid line.
    bridge_w = max(3, int(round(0.05 * px_per_cm)))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((1, bridge_w), dtype=np.uint8))

    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_curve_w = max(28, int(round(0.38 * px_per_cm)))
    max_char_h = max(24, int(round(0.34 * px_per_cm)))

    cc_weights = np.ones(num_cc, dtype=np.float32)
    cc_weights[0] = 0.0
    is_long = np.zeros(num_cc, dtype=bool)
    for cid in range(1, num_cc):
        w_cc = int(stats[cid, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        fill = area / max(1, w_cc * h_cc)
        aspect = w_cc / max(1, h_cc)
        # A printed word ("FHR1", "110-200", "Puls", "200") is a compact block:
        # letter height, high fill, not a long ribbon. A pen stroke over the same
        # width is either much flatter or a thin line through a tall box.
        is_text_block = w_cc >= 18 and 12 <= h_cc <= 48 and fill >= 0.32 and aspect <= 6.0
        if is_text_block:
            cc_weights[cid] = 0.03
        elif w_cc >= min_curve_w and h_cc > 2:
            cc_weights[cid] = 1.0
            is_long[cid] = True
        elif h_cc <= max_char_h:
            cc_weights[cid] = 0.04
        else:
            cc_weights[cid] = 0.35

    weighted = band_u8.astype(np.float32) * cc_weights[labels]
    reward = cv2.GaussianBlur(weighted, (0, 0), sigmaX=1.3, sigmaY=1.6)
    return reward, is_long[labels]


def _attenuate_grid_rows(reward: np.ndarray, period_px: float, phase_px: float, factor: float = 0.15) -> np.ndarray:
    """Dim the pre-printed grid rows so a flat line cannot outscore the pen.

    The black-ink layer still contains the green grid. A Viterbi path will ride
    one of those rows when the pen reward next to it is patchy. The pen is
    several pixels thick, so dimming the grid row itself leaves the stroke.
    """
    out = reward.copy()
    h = out.shape[0]
    y = float(phase_px)
    # One row only. The pen is thicker than the grid line, and the maternal
    # pulse often lies on the 80 bpm line, so a wider notch would erase it.
    while y < h + period_px:
        yi = int(round(y))
        if 0 <= yi < h:
            out[yi] *= factor
        y += period_px
    return out


def _downweight_header_text(band_u8: np.ndarray, reward: np.ndarray, header_rows: int) -> np.ndarray:
    """Down-weight printed header text in the top of the FHR band.

    The repeated ``FHR1 110-200`` / ``Puls`` blocks sit above about 180 bpm and,
    on the full band, a single pen spike joins them to the trace so they inherit
    the trace's weight. Classifying that strip on its own keeps a real high
    fetal rate (a long thin stroke) and drops the word-shaped blocks.
    """
    header_rows = int(max(0, min(header_rows, band_u8.shape[0] // 2)))
    if header_rows < 20:
        return reward
    sub = band_u8[:header_rows]
    bw = (sub >= 26).astype(np.uint8)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 3), dtype=np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((1, 5), dtype=np.uint8))
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    scale = np.ones(num_cc, dtype=np.float32)
    for cid in range(1, num_cc):
        w_cc = int(stats[cid, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        fill = area / max(1, w_cc * h_cc)
        aspect = w_cc / max(1, h_cc)
        is_text_block = w_cc >= 18 and 12 <= h_cc <= 48 and fill >= 0.32 and aspect <= 6.0
        is_small = w_cc < 28 and h_cc <= 32
        if is_text_block or is_small:
            scale[cid] = 0.04
    factor = scale[labels]
    out = reward.copy()
    out[:header_rows] *= factor
    return out


def _downweight_left_stamp(band_u8: np.ndarray, reward: np.ndarray, px_per_cm: int) -> np.ndarray:
    """Drop the vertical time stamp printed at the left edge of the recording.

    Segmented on its own, that stamp is a set of short strokes that never reach
    the rest of the trace. A horizontal opening removes the 1 px grid lines that
    would otherwise join those strokes to the page. The pen, later on, is not in
    this crop.
    """
    crop = int(round(4.0 * px_per_cm))
    crop = max(40, min(crop, band_u8.shape[1] // 3))
    if band_u8.shape[1] < crop + 10:
        return reward
    bw = (band_u8[:, :crop] >= 26).astype(np.uint8)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 3), dtype=np.uint8))
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    kill = np.zeros(num_cc, dtype=bool)
    for cid in range(1, num_cc):
        right = int(stats[cid, cv2.CC_STAT_LEFT] + stats[cid, cv2.CC_STAT_WIDTH])
        if right < crop - 12:
            kill[cid] = True
    if not kill.any():
        return reward
    mask = cv2.dilate(kill[labels].astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    out = reward.copy()
    out[:, :crop] = np.where(mask, out[:, :crop] * 0.03, out[:, :crop])
    return out


def _leading_gridlock_end(
    y_px: np.ndarray,
    origin: float,
    step: float,
    tol: float = 4.0,
    free_need: int = 150,
    lock_need: int = 40,
) -> int:
    """Return the x where a leading grid-line ride gives way to the pen.

    Before the pens are down the tracker sits on a grid line (here the 110 bpm
    line). A scale digit can tug it off that line for a few dozen columns and
    then it locks back on, so a long lock resets the count of off-line columns.
    The walk stops once the path has stayed off the grid, which is the pen.
    """
    rel = np.mod(y_px.astype(np.float64) - origin, step)
    dist = np.minimum(rel, step - rel)
    locked = dist < tol
    cut = 0
    free = 0
    x = 0
    n = len(y_px)
    while x < n and free < free_need:
        if not locked[x]:
            free += 1
            x += 1
            continue
        x1 = x
        while x1 < n and locked[x1]:
            x1 += 1
        if (x1 - x) >= lock_need and free < free_need:
            cut = x1
            free = 0
        x = x1
    # Include the short jump off the line that follows the lock.
    # The path then steps off the line onto the pen in about 15 columns.
    # Those columns are the jump, not a sample of the trace.
    if cut > 0:
        cut = min(n, cut + 18)
    return int(cut)


def _columns_on_ink(trace_u8: np.ndarray, y_px: np.ndarray, tol: int = _INK_TOL_PX, ink_min: int = _INK_MIN) -> np.ndarray:
    """True where ``trace_u8`` has ink within ``tol`` px of the tracked row."""
    h, w = trace_u8.shape
    yi = np.clip(np.round(y_px).astype(np.int32), 0, h - 1)
    acc = np.zeros(w, dtype=np.uint8)
    cols = np.arange(w)
    for delta in range(-tol, tol + 1):
        yy = np.clip(yi + delta, 0, h - 1)
        acc = np.maximum(acc, trace_u8[yy, cols])
    return acc >= ink_min


def _bridge_internal_gaps(valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill only short holes that have ink on both sides. Leading and trailing blanks stay blank."""
    out = valid.astype(bool).copy()
    n = len(out)
    x = 0
    while x < n:
        if out[x]:
            x += 1
            continue
        x1 = x
        while x1 < n and not out[x1]:
            x1 += 1
        if x > 0 and x1 < n and (x1 - x) <= max_gap:
            out[x:x1] = True
        x = x1
    return out


def _resample_with_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    x_query: np.ndarray,
    xs_full: np.ndarray,
    ndigits: int,
) -> list[float | None]:
    """Sample ``values`` at ``x_query``. Columns that are not ``valid`` become ``None``."""
    interp = np.interp(x_query, xs_full, values.astype(np.float64))
    out: list[float | None] = []
    last = len(valid) - 1
    for i, xq in enumerate(x_query):
        xi = int(np.clip(int(round(float(xq))), 0, last))
        if not valid[xi]:
            out.append(None)
        else:
            out.append(round(float(interp[i]), ndigits))
    return out


def _viterbi_ridge_1d(
    reward: np.ndarray,
    max_step: int,
    l1_pen: float,
    l2_pen: float,
    prior_bonus: np.ndarray | None = None,
) -> np.ndarray:
    """Global forward-backward Viterbi dynamic programming ridge tracker ``y(x)`` across ``(H, W)``."""
    h, w = reward.shape
    score = reward.astype(np.float32)
    if prior_bonus is not None:
        score = score + prior_bonus.reshape(h, 1).astype(np.float32)

    dp = -score[:, 0].astype(np.float32).copy()
    back = np.zeros((h, w), dtype=np.int8)

    offsets = list(range(-max_step, max_step + 1))
    step_costs = [float(l1_pen * abs(dy) + l2_pen * (dy * dy)) for dy in offsets]

    best_prev = np.empty(h, dtype=np.float32)
    best_dy = np.empty(h, dtype=np.int8)

    for x in range(1, w):
        best_prev.fill(1e30)
        best_dy.fill(0)
        for dy, sc in zip(offsets, step_costs):
            if dy == 0:
                cand = dp + sc
                better = cand < best_prev
                best_prev = np.where(better, cand, best_prev)
                best_dy = np.where(better, 0, best_dy)
            elif dy > 0:
                # Transition from y_prev = y - dy -> valid for y >= dy
                cand_slice = dp[:-dy] + sc
                better = cand_slice < best_prev[dy:]
                best_prev[dy:] = np.where(better, cand_slice, best_prev[dy:])
                best_dy[dy:] = np.where(better, dy, best_dy[dy:])
            else:
                # dy < 0: transition from y_prev = y - dy = y + |dy| -> valid for y < h - |dy|
                ady = -dy
                cand_slice = dp[ady:] + sc
                better = cand_slice < best_prev[:-ady]
                best_prev[:-ady] = np.where(better, cand_slice, best_prev[:-ady])
                best_dy[:-ady] = np.where(better, dy, best_dy[:-ady])

        dp = best_prev - score[:, x]
        back[:, x] = best_dy

    path = np.zeros(w, dtype=np.float32)
    yi = int(np.argmin(dp))
    path[w - 1] = float(yi)
    for x in range(w - 1, 0, -1):
        dy = int(back[yi, x])
        yi = int(np.clip(yi - dy, 0, h - 1))
        path[x - 1] = float(yi)

    # Parabolic sub-pixel refinement on reward[:, x]
    cols = np.arange(w)
    yi_int = path.astype(np.int32)
    valid_parab = (yi_int >= 1) & (yi_int < h - 1)
    y_c = np.clip(yi_int, 1, h - 2)
    r_m = reward[y_c - 1, cols]
    r_0 = reward[y_c, cols]
    r_p = reward[y_c + 1, cols]
    denom = 2.0 * (2.0 * r_0 - r_m - r_p)
    delta = np.where(
        valid_parab & (denom > 1e-3) & (r_0 >= r_m) & (r_0 >= r_p),
        np.clip((r_p - r_m) / np.maximum(denom, 1e-3), -0.5, 0.5),
        0.0,
    )
    return path + delta.astype(np.float32)


def _detect_kineto_events(
    trace_u8: np.ndarray,
    toco_y0: int,
    px_per_sec: float,
    px_per_cm: int,
) -> list[dict[str, Any]]:
    """Detect printed fetal movement (Kineto) dark marker blocks along the top border of the Toco grid."""
    y_a = max(0, toco_y0 + int(round(0.03 * px_per_cm)))
    y_b = min(trace_u8.shape[0], toco_y0 + int(round(0.32 * px_per_cm)))
    if y_b <= y_a + 4:
        return []

    strip = trace_u8[y_a:y_b, :]
    # Kineto blocks are dense dark vertical/rectangular bars occupying >= 55% of the top Toco margin height
    col_fill = (strip >= 55).mean(axis=0)
    active = col_fill >= 0.58

    events: list[dict[str, Any]] = []
    min_w = max(4, int(round(0.05 * px_per_cm)))
    max_w = int(round(0.55 * px_per_cm))

    x = 0
    w = len(active)
    last_t = -999
    while x < w:
        if not active[x]:
            x += 1
            continue
        x1 = x
        while x1 < w and active[x1]:
            x1 += 1
        span = x1 - x
        if min_w <= span <= max_w:
            t_sec = int(round(0.5 * (x + x1) / px_per_sec))
            if t_sec - last_t >= 25:
                events.append(
                    {
                        "t": t_sec,
                        "kind": "kineto",
                        "text": "Kineto",
                    }
                )
                last_t = t_sec
        x = x1
    return events


def _clock_at_left_edge(header_iso: str, first_stamp_x_px: float | None, px_per_sec: float) -> datetime:
    """Clock time at panorama x = 0.

    The recorder prints its start time to the second in the vertical header and
    then stamps the separator strip at the next full minute (header 11:10:12,
    first stamp 11:11 on the reference strip). The stamp is the better anchor:
    the panorama starts with blank leader, so the header time is not at x = 0.
    Without a stamp the header time is assumed to sit at the left edge.
    """
    header = datetime.fromisoformat(header_iso)
    if first_stamp_x_px is None:
        return header
    stamp_clock = header.replace(second=0, microsecond=0)
    if stamp_clock < header:
        stamp_clock += timedelta(minutes=1)
    return stamp_clock - timedelta(seconds=round(first_stamp_x_px / px_per_sec))


def _clock_str(clock_at_zero: datetime, t_sec: float) -> str:
    return (clock_at_zero + timedelta(seconds=max(0, int(round(t_sec))))).strftime("%H:%M:%S")


def run_extract_pipeline(
    paths: DatasetPaths,
    ext_cfg: ExtractConfig | None = None,
    stitch_cfg: StitchConfig | None = None,
) -> dict[str, Any]:
    """Run Pipeline 4: extract calibrated 1-Hz FHR, MHR & Toco series from the stitched CTG panorama."""
    paths.ensure_work_dirs()
    manifest = DatasetManifest.load(paths.manifest_path)
    ext_cfg = ext_cfg or ExtractConfig()

    meta_path = paths.output_dir / "stitch_metadata.json"
    pano_path = paths.output_dir / "panorama.png"
    if not meta_path.exists() or not pano_path.exists():
        run_stitch_pipeline(paths, stitch_cfg=stitch_cfg or StitchConfig())

    stitch_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pano_bgr = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    if pano_bgr is None:
        raise FileNotFoundError(f"Missing stitched panorama at {pano_path}")

    trace_path = paths.output_dir / "panorama_trace.png"
    trace_u8 = cv2.imread(str(trace_path), cv2.IMREAD_GRAYSCALE) if trace_path.exists() else None
    if trace_u8 is None or trace_u8.shape != pano_bgr.shape[:2]:
        trace_u8 = _extract_black_ink_from_bgr(pano_bgr)

    h, w = trace_u8.shape
    px_per_cm = int(stitch_meta["target_px_per_cm"])
    speed_cm_min = float(stitch_meta["paper_speed_cm_per_min"])
    px_per_sec = (px_per_cm * speed_cm_min) / 60.0

    fhr_y0 = int(round(FHR_S_TOP_CM * px_per_cm))
    fhr_y1 = int(round(FHR_S_BOT_CM * px_per_cm))
    fhr_px_per_bpm = (fhr_y1 - fhr_y0) / 160.0  # 210..50 bpm = 160 bpm

    toco_y0 = int(round(TOCO_S_TOP_CM * px_per_cm))
    toco_y1 = int(round(TOCO_S_BOT_CM * px_per_cm))
    toco_px_per_pct = (toco_y1 - toco_y0) / 100.0

    # --- 1. FHR + Maternal Pulse (MHR) dual-curve Viterbi tracking ---
    fhr_band = trace_u8[fhr_y0:fhr_y1, :]
    fhr_reward, fhr_long_mask = _prepare_curve_reward(fhr_band, px_per_cm)
    # Printed "FHR1 110-200" / "Puls" headers live above ~178 bpm. Real baseline
    # on this recording sits near 140-170, so this does not blank the trace.
    header_rows = int(round((210.0 - 178.0) * fhr_px_per_bpm))
    fhr_reward = _downweight_header_text(fhr_band, fhr_reward, header_rows)
    fhr_reward = _downweight_left_stamp(fhr_band, fhr_reward, px_per_cm)
    # FHR grid is every 0.25 cm. Band row 0 is the 210 bpm line.
    fhr_reward = _attenuate_grid_rows(fhr_reward, period_px=0.25 * px_per_cm, phase_px=0.0)
    fh_h = fhr_band.shape[0]
    ys_rel = np.arange(fh_h, dtype=np.float32)
    bpm_grid = 210.0 - ys_rel / fhr_px_per_bpm

    # Check if a continuous maternal pulse curve exists in [54..106 bpm]
    mhr_zone_mask = (bpm_grid >= 54.0) & (bpm_grid <= 106.0)
    fhr_zone_mask = (bpm_grid >= 112.0) & (bpm_grid <= 198.0)
    mhr_col_cov = float(fhr_long_mask[mhr_zone_mask, :].any(axis=0).mean())
    fhr_col_cov = float(fhr_long_mask[fhr_zone_mask, :].any(axis=0).mean())
    has_mhr_trace = (mhr_col_cov >= 0.18) and (fhr_col_cov >= 0.18)

    fhr_reward_for_fetus = fhr_reward.copy()
    if has_mhr_trace:
        mhr_y_min = int(np.clip(round((210.0 - 114.0) * fhr_px_per_bpm), 0, fh_h - 30))
        mhr_y_max = int(np.clip(round((210.0 - 58.0) * fhr_px_per_bpm), mhr_y_min + 20, fh_h))
        mhr_sub = fhr_reward[mhr_y_min:mhr_y_max, :]
        mhr_bpm_sub = bpm_grid[mhr_y_min:mhr_y_max]
        mhr_prior = np.where((mhr_bpm_sub >= 66.0) & (mhr_bpm_sub <= 96.0), 5.0, -8.0).astype(np.float32)
        y_mhr_sub = _viterbi_ridge_1d(
            mhr_sub,
            max_step=max(2, int(round(0.035 * px_per_cm))),
            l1_pen=2.2,
            l2_pen=0.85,
            prior_bonus=mhr_prior,
        )
        y_mhr_rel = y_mhr_sub + float(mhr_y_min)

        # Attenuate a narrow corridor around MHR so FHR tracker follows the fetal trace above/across it
        corridor = max(6, int(round(2.2 * fhr_px_per_bpm)))
        yy = np.arange(fh_h, dtype=np.float32).reshape(-1, 1)
        near_mhr = np.abs(yy - y_mhr_rel.reshape(1, -1)) <= float(corridor)
        fhr_reward_for_fetus = np.where(near_mhr, fhr_reward * 0.04, fhr_reward)
    else:
        y_mhr_rel = np.full(w, (210.0 - 80.0) * fhr_px_per_bpm, dtype=np.float32)

    fhr_prior = np.where((bpm_grid >= 105.0) & (bpm_grid <= 195.0), 3.5, 0.0).astype(np.float32)
    y_fhr_rel = _viterbi_ridge_1d(
        fhr_reward_for_fetus,
        max_step=max(5, int(round(0.11 * px_per_cm))),
        l1_pen=0.65,
        l2_pen=0.14,
        prior_bonus=fhr_prior,
    )

    # --- 2. Toco uterine contraction Viterbi tracking + Kineto detection ---
    toco_band = trace_u8[toco_y0:toco_y1, :].copy()
    # Suppress Kineto bar strip along very top [0 .. 0.36 cm] of Toco panel
    kineto_h = min(toco_band.shape[0] // 3, int(round(0.36 * px_per_cm)))
    toco_band[:kineto_h, :] = (toco_band[:kineto_h, :].astype(np.float32) * 0.05).astype(np.uint8)
    toco_reward, _ = _prepare_curve_reward(toco_band, px_per_cm)
    # Toco grid is every 0.40 cm. The 100-line is at the top edge of the band.
    toco_phase = (TOCO_S_TOP_CM * px_per_cm) - toco_y0
    toco_reward = _attenuate_grid_rows(toco_reward, period_px=0.40 * px_per_cm, phase_px=toco_phase)
    toco_reward = _downweight_left_stamp(toco_band, toco_reward, px_per_cm)

    th_h = toco_band.shape[0]
    toco_pct_grid = 100.0 - np.arange(th_h, dtype=np.float32) / toco_px_per_pct
    toco_prior = np.where((toco_pct_grid >= 2.0) & (toco_pct_grid <= 75.0), 2.5, -2.0).astype(np.float32)
    y_toco_rel = _viterbi_ridge_1d(
        toco_reward,
        max_step=max(3, int(round(0.05 * px_per_cm))),
        l1_pen=1.1,
        l2_pen=0.40,
        prior_bonus=toco_prior,
    )

    # Convert per-pixel curves to physical units + light 3-px median de-glitch
    def med3(arr: np.ndarray) -> np.ndarray:
        if len(arr) < 3:
            return arr
        p = np.pad(arr, (1, 1), mode="edge")
        return np.median(np.vstack([p[:-2], p[1:-1], p[2:]]), axis=0).astype(np.float32)

    y_fhr_px = float(fhr_y0) + med3(y_fhr_rel)
    y_mhr_px = float(fhr_y0) + med3(y_mhr_rel)
    y_toco_px = float(toco_y0) + med3(y_toco_rel)

    fhr_bpm_px = np.clip(210.0 - (y_fhr_px - fhr_y0) / fhr_px_per_bpm, 50.0, 210.0)
    mhr_bpm_px = (
        np.clip(210.0 - (y_mhr_px - fhr_y0) / fhr_px_per_bpm, 45.0, 140.0)
        if has_mhr_trace
        else np.zeros_like(fhr_bpm_px)
    )
    toco_pct_px = np.clip(100.0 - (y_toco_px - toco_y0) / toco_px_per_pct, 0.0, 100.0)

    # --- 3. Resample onto exact dt_sec (default 1.0 s) time grid ---
    duration_sec = int(np.floor((w - 1) / px_per_sec))
    dt = float(ext_cfg.dt_sec)
    t_grid = np.arange(0.0, duration_sec + 0.5 * dt, dt, dtype=np.float64)
    x_query = np.clip(t_grid * px_per_sec, 0.0, float(w - 1))
    xs_full = np.arange(w, dtype=np.float64)

    # Null where the track is not on ink. Short internal dropouts (dotted pen,
    # a few px of blur) stay filled; paper edges and real gaps do not.
    fhr_valid = _columns_on_ink(trace_u8, y_fhr_px)
    if has_mhr_trace:
        mhr_valid = _columns_on_ink(trace_u8, y_mhr_px)
    else:
        mhr_valid = np.zeros(w, dtype=bool)
    toco_valid = _columns_on_ink(trace_u8, y_toco_px)

    # The fetal path rides the 110 bpm grid line until the pen is down. The
    # maternal path is on the same time stamp until that same column; afterwards
    # it lies on its own pen (often on the 70 or 80 line, so it cannot be told
    # from the grid by y alone).
    fhr_grid_step = 0.25 * px_per_cm
    pen_start = _leading_gridlock_end(y_fhr_px, origin=float(fhr_y0), step=fhr_grid_step)
    if pen_start > 0:
        fhr_valid[:pen_start] = False
        mhr_valid[:pen_start] = False

    # A long dwell on the toco zero line is the tracker falling through the
    # printed "PHILIPS" word in the gap before the pen, not a contraction of 0.
    zero_y = float(toco_y0) + 100.0 * toco_px_per_pct
    on_zero = np.abs(y_toco_px - zero_y) <= 6.0
    zx = 0
    while zx < w:
        if not on_zero[zx]:
            zx += 1
            continue
        zx1 = zx
        while zx1 < w and on_zero[zx1]:
            zx1 += 1
        if zx1 - zx > 20:
            toco_valid[zx:zx1] = False
        zx = zx1

    fhr_valid = _bridge_internal_gaps(fhr_valid, _INK_BRIDGE_PX)
    mhr_valid = _bridge_internal_gaps(mhr_valid, _INK_BRIDGE_PX)
    toco_valid = _bridge_internal_gaps(toco_valid, _INK_BRIDGE_PX)

    fhr_series = _resample_with_gaps(fhr_bpm_px, fhr_valid, x_query, xs_full, 1)
    mhr_series = _resample_with_gaps(mhr_bpm_px, mhr_valid, x_query, xs_full, 1)
    toco_pct_series = _resample_with_gaps(toco_pct_px, toco_valid, x_query, xs_full, 1)
    toco_kpa_series = [
        None if v is None else round(float(v) * TOCO_KPA_PER_PCT, 2) for v in toco_pct_series
    ]

    events = _detect_kineto_events(trace_u8, toco_y0, px_per_sec=px_per_sec, px_per_cm=px_per_cm)

    stamps = find_time_stamps(pano_bgr)
    first_stamp_x = float(stamps[0]["x_px"]) if stamps else None
    clock_at_zero = _clock_at_left_edge(ext_cfg.started_at_iso, first_stamp_x, px_per_sec)

    # Build exact per-scan time intervals [t0, t1] directly from Panorama Stitch placements!
    pms = stitch_meta.get("pair_matches", [])
    placements = stitch_meta.get("frame_placements", [])
    scans_meta: list[dict[str, Any]] = []
    for i, pl in enumerate(placements):
        t0_s = round(max(0.0, float(pl["x_px"]) / px_per_sec), 1)
        t1_s = round(min(float(duration_sec), float(pl["x_px"] + pl["width_px"]) / px_per_sec), 1)
        conf_neighbors: list[float] = []
        if i > 0 and i - 1 < len(pms):
            conf_neighbors.append(float(pms[i - 1]["confidence"]))
        if i < len(pms):
            conf_neighbors.append(float(pms[i]["confidence"]))
        mean_c = float(np.mean(conf_neighbors)) if conf_neighbors else 0.8
        trust = "hoch" if mean_c >= 0.62 else ("mittel" if mean_c >= 0.42 else "niedrig")
        stem = Path(pl["filename"]).stem
        scans_meta.append(
            {
                "file": f"{stem}.jpg",
                "source_frame": pl["filename"],
                "t0": int(round(t0_s)),
                "t1": int(round(t1_s)),
                "trust": trust,
            }
        )

    # Infer session_id (e.g. "2026-11-22" from "2026-11-22_philips_ctg_001")
    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})", manifest.dataset_id)
    session_id = ext_cfg.session_id or (date_match.group(1) if date_match else manifest.dataset_id)
    session_title = ext_cfg.session_title or session_id

    ctg_session = {
        "id": session_id,
        "title": session_title,
        # Clock at sample 0 (the left edge of the panorama), not the header time.
        "startedAt": clock_at_zero.isoformat(timespec="seconds"),
        "recordingStartedAt": ext_cfg.started_at_iso,
        "durationSec": duration_sec,
        "dt": int(dt) if dt.is_integer() else dt,
        "device": ext_cfg.device_name,
        "paper": ext_cfg.paper_label,
        "speedCmMin": speed_cm_min,
        "fhrScale": {"min": 50, "max": 210, "major": 20, "label": "bpm"},
        "tocoScale": {"min": 0, "max": 12, "major": 2, "label": "kPa"},
        "provenance": {
            "trace": f"ctg-converter-viterbi-{px_per_cm}px-cm",
            "scans": "90deg-projective-comb-rectified+hybrid-stitched",
            "note": (
                f"Automatisch extrahiert aus dem 90°-entzerrten und nahtlos gestitchten "
                f"Gesamt-Panorama ({w}x{h} px @ {px_per_cm} px/cm) mittels globalem Viterbi-Kurven-Tracking."
            ),
        },
        "fhr": fhr_series,
        "mhr": mhr_series,
        "toco": toco_kpa_series,
        "tocoPct": toco_pct_series,
        "tocoKpaPerPct": TOCO_KPA_PER_PCT,
        "events": events,
        "scans": [{k: v for k, v in s.items() if k != "source_frame"} for s in scans_meta],
    }

    session_json_path = paths.output_dir / "ctg_session.json"
    session_json_path.write_text(json.dumps(ctg_session, indent=2) + "\n", encoding="utf-8")

    trace_csv_path = paths.output_dir / "trace.csv"
    with trace_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_sec", "clock", "x_cm", "fhr_bpm", "mhr_bpm", "toco_kpa", "toco_pct"])
        def _cell(value: float | None) -> float | str:
            return "" if value is None else value

        for idx_t, t_val in enumerate(t_grid):
            writer.writerow(
                [
                    int(round(t_val)) if dt.is_integer() else round(float(t_val), 2),
                    _clock_str(clock_at_zero, float(t_val)),
                    round(float(t_val) * speed_cm_min / 60.0, 3),
                    _cell(fhr_series[idx_t]),
                    _cell(mhr_series[idx_t]),
                    _cell(toco_kpa_series[idx_t]),
                    _cell(toco_pct_series[idx_t]),
                ]
            )

    events_csv_path = paths.output_dir / "events.csv"
    with events_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_sec", "kind", "text", "value", "extra"])
        for ev in events:
            writer.writerow([ev["t"], ev["kind"], ev["text"], ev.get("value", ""), ev.get("extra", "")])

    # Save pixel-space curve coordinates so PySide6 Studio can toggle live vector overlays!
    curves_px_path = paths.output_dir / "extracted_curves_px.json"
    step_px = max(1, int(round(0.04 * px_per_cm)))  # every 4 px = 0.4 mm for lightweight GUI polyline
    xs_sub = list(range(0, w, step_px))
    if xs_sub[-1] != w - 1:
        xs_sub.append(w - 1)
    curves_px_data = {
        "has_mhr_trace": has_mhr_trace,
        "x_px": xs_sub,
        "fhr_y_px": [round(float(y_fhr_px[x]), 2) for x in xs_sub],
        "mhr_y_px": [round(float(y_mhr_px[x]), 2) for x in xs_sub] if has_mhr_trace else [],
        "toco_y_px": [round(float(y_toco_px[x]), 2) for x in xs_sub],
    }
    curves_px_path.write_text(json.dumps(curves_px_data) + "\n", encoding="utf-8")

    overlay_png_path: Path | None = None
    if ext_cfg.write_overlay_png:
        overlay_png_path = paths.output_dir / "panorama_extracted_overlay.png"
        ov = pano_bgr.copy()

        def _draw_valid(y_px: np.ndarray, valid_px: np.ndarray, color: tuple[int, int, int]) -> None:
            # Break the polyline at gaps so a missing trace is not drawn across blank paper.
            run_x: list[int] = []
            run_y: list[int] = []

            def flush() -> None:
                if len(run_x) >= 2:
                    pts = np.column_stack([run_x, run_y]).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(ov, [pts], False, color, 2, cv2.LINE_AA)

            for x_i in range(w):
                if valid_px[x_i]:
                    run_x.append(x_i)
                    run_y.append(int(round(float(y_px[x_i]))))
                elif run_x:
                    flush()
                    run_x, run_y = [], []
            flush()

        _draw_valid(y_fhr_px, fhr_valid, (30, 45, 235))  # Vivid Red = FHR
        if has_mhr_trace:
            _draw_valid(y_mhr_px, mhr_valid, (0, 140, 255))  # Orange = MHR
        _draw_valid(y_toco_px, toco_valid, (220, 90, 20))  # Blue = Toco
        cv2.imwrite(str(overlay_png_path), ov, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        del ov

    def _finite(series: list[float | None]) -> np.ndarray:
        return np.array([v for v in series if v is not None], dtype=np.float64)

    fhr_arr = _finite(fhr_series)
    mhr_arr = _finite(mhr_series)
    toco_pct_arr = _finite(toco_pct_series)

    summary = {
        "dataset_id": manifest.dataset_id,
        "session_id": session_id,
        "samples_1hz": len(fhr_series),
        "duration_sec": duration_sec,
        "has_mhr_trace": has_mhr_trace,
        "fhr_valid_fraction": round(float(np.mean([v is not None for v in fhr_series])), 4),
        "mhr_valid_fraction": round(float(np.mean([v is not None for v in mhr_series])), 4),
        "toco_valid_fraction": round(float(np.mean([v is not None for v in toco_pct_series])), 4),
        "fhr_median_bpm": round(float(np.median(fhr_arr)), 1) if len(fhr_arr) else None,
        "fhr_p10_p90_bpm": (
            [round(float(np.percentile(fhr_arr, 10)), 1), round(float(np.percentile(fhr_arr, 90)), 1)]
            if len(fhr_arr)
            else None
        ),
        "mhr_median_bpm": round(float(np.median(mhr_arr)), 1) if len(mhr_arr) else None,
        "toco_median_pct": round(float(np.median(toco_pct_arr)), 1) if len(toco_pct_arr) else None,
        "toco_kpa_per_pct": TOCO_KPA_PER_PCT,
        "kineto_events_detected": len(events),
        "time_stamps_found": len(stamps),
        "clock_at_left_edge": clock_at_zero.isoformat(timespec="seconds"),
        "outputs": {
            "ctg_session_json": str(session_json_path),
            "trace_csv": str(trace_csv_path),
            "events_csv": str(events_csv_path),
            "overlay_png": str(overlay_png_path) if overlay_png_path else None,
        },
    }
    return summary

