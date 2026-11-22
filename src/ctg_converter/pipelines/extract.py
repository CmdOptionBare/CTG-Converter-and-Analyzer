"""Pipeline 4: Calibrated Image-to-Value Curve Extraction (FHR, MHR, Toco, Kineto)
and Direct Session Export Bridge for the ``CmdOptionBare/CTG-Project1`` Web Viewer.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
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


@dataclass
class ExtractConfig:
    """Configuration for calibrated curve extraction and CTG-Project1 export."""

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


def _prepare_curve_reward(band_u8: np.ndarray, px_per_cm: int) -> tuple[np.ndarray, np.ndarray]:
    """Suppress isolated printed digits/text while preserving continuous curve strokes.

    Returns ``(reward_float32, long_ribbon_mask_bool)``.
    """
    bw = (band_u8 >= 26).astype(np.uint8)
    # Bridge small 4-5 px horizontal gaps in dotted/fast pen strokes without merging adjacent text lines
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
        if w_cc >= min_curve_w:
            cc_weights[cid] = 1.0
            is_long[cid] = True
        elif w_cc < min_curve_w and h_cc <= max_char_h:
            # Compact isolated character or speck (e.g. printed '200', '180', '80' scale labels)
            cc_weights[cid] = 0.08
        else:
            # Steep vertical spike segment
            cc_weights[cid] = 0.65

    weighted = band_u8.astype(np.float32) * cc_weights[labels]
    reward = cv2.GaussianBlur(weighted, (0, 0), sigmaX=1.3, sigmaY=1.6)
    return reward, is_long[labels]


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
                        "text": "Kindsbewegung (Kineto-Markierung)",
                    }
                )
                last_t = t_sec
        x = x1
    return events


def _clock_str(started_at_iso: str, t_sec: float) -> str:
    m = re.search(r"(\d{2}):(\d{2}):(\d{2})", started_at_iso)
    if m:
        h0, m0, s0 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        h0, m0, s0 = 11, 10, 12
    total = h0 * 3600 + m0 * 60 + s0 + max(0, int(round(t_sec)))
    hh = (total // 3600) % 24
    mm = (total // 60) % 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


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

    fhr_series = [round(float(v), 1) for v in np.interp(x_query, xs_full, fhr_bpm_px)]
    mhr_series = [round(float(v), 1) for v in np.interp(x_query, xs_full, mhr_bpm_px)]
    toco_pct_series = [round(float(v), 1) for v in np.interp(x_query, xs_full, toco_pct_px)]
    # CTG-Project1's paper.tsx maps 0..12 kPa linearly across the full Toco panel height!
    toco_kpa_series = [round(float(v) * 0.12, 2) for v in toco_pct_series]

    events = _detect_kineto_events(trace_u8, toco_y0, px_per_sec=px_per_sec, px_per_cm=px_per_cm)

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
        "startedAt": ext_cfg.started_at_iso,
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
        "events": events,
        "scans": [{k: v for k, v in s.items() if k != "source_frame"} for s in scans_meta],
    }

    session_json_path = paths.output_dir / "ctg_session.json"
    session_json_path.write_text(json.dumps(ctg_session, indent=2) + "\n", encoding="utf-8")

    trace_csv_path = paths.output_dir / "trace.csv"
    with trace_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_sec", "clock", "x_cm", "fhr_bpm", "mhr_bpm", "toco_kpa", "toco_pct"])
        for idx_t, t_val in enumerate(t_grid):
            writer.writerow(
                [
                    int(round(t_val)) if dt.is_integer() else round(float(t_val), 2),
                    _clock_str(ext_cfg.started_at_iso, float(t_val)),
                    round(float(t_val) * speed_cm_min / 60.0, 3),
                    fhr_series[idx_t],
                    mhr_series[idx_t],
                    toco_kpa_series[idx_t],
                    toco_pct_series[idx_t],
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
        pts_f = np.column_stack([np.arange(w), np.round(y_fhr_px)]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(ov, [pts_f], False, (30, 45, 235), 2, cv2.LINE_AA)  # Vivid Red = FHR
        if has_mhr_trace:
            pts_m = np.column_stack([np.arange(w), np.round(y_mhr_px)]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(ov, [pts_m], False, (0, 140, 255), 2, cv2.LINE_AA)  # Orange/Copper = MHR
        pts_t = np.column_stack([np.arange(w), np.round(y_toco_px)]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(ov, [pts_t], False, (220, 90, 20), 2, cv2.LINE_AA)  # Blue = Toco
        cv2.imwrite(str(overlay_png_path), ov, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        del ov

    fhr_arr = np.array(fhr_series, dtype=np.float64)
    mhr_arr = np.array(mhr_series, dtype=np.float64)
    toco_pct_arr = np.array(toco_pct_series, dtype=np.float64)

    summary = {
        "dataset_id": manifest.dataset_id,
        "session_id": session_id,
        "samples_1hz": len(fhr_series),
        "duration_sec": duration_sec,
        "has_mhr_trace": has_mhr_trace,
        "fhr_median_bpm": round(float(np.median(fhr_arr)), 1),
        "fhr_p10_p90_bpm": [round(float(np.percentile(fhr_arr, 10)), 1), round(float(np.percentile(fhr_arr, 90)), 1)],
        "mhr_median_bpm": round(float(np.median(mhr_arr)), 1) if has_mhr_trace else None,
        "toco_median_pct": round(float(np.median(toco_pct_arr)), 1),
        "kineto_events_detected": len(events),
        "outputs": {
            "ctg_session_json": str(session_json_path),
            "trace_csv": str(trace_csv_path),
            "events_csv": str(events_csv_path),
            "overlay_png": str(overlay_png_path) if overlay_png_path else None,
        },
    }
    return summary


def export_to_ctg_project1(
    paths: DatasetPaths,
    target_repo_dir: Path,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Export extracted ``CtgSession`` JSON, CSVs, and ``paper.tsx``-calibrated rectified scans
    directly into a local checkout of ``CmdOptionBare/CTG-Project1``.
    """
    session_src = paths.output_dir / "ctg_session.json"
    if not session_src.exists():
        run_extract_pipeline(paths, ext_cfg=ExtractConfig(session_id=session_id))

    session_data = json.loads(session_src.read_text(encoding="utf-8"))
    if session_id:
        session_data["id"] = session_id
    sid = session_data["id"]

    target_repo_dir = target_repo_dir.resolve()
    out_json = target_repo_dir / "src" / "data" / "ctg" / f"{sid}.json"
    pub_dir = target_repo_dir / "public" / "ctg" / sid
    scans_dir = pub_dir / "scans"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    scans_dir.mkdir(parents=True, exist_ok=True)

    # Remove extra `tocoPct` key only if strict consumer desired? TypeScript structural typing allows extra JSON keys,
    # and keeping `CtgSession` exact keys guarantees 100% clean import in Vite/TS!
    strict_keys = [
        "id",
        "title",
        "startedAt",
        "durationSec",
        "dt",
        "device",
        "paper",
        "speedCmMin",
        "fhrScale",
        "tocoScale",
        "provenance",
        "fhr",
        "mhr",
        "toco",
        "events",
        "scans",
    ]
    clean_session = {k: session_data[k] for k in strict_keys if k in session_data}
    out_json.write_text(json.dumps(clean_session, indent=2) + "\n", encoding="utf-8")

    shutil.copy2(paths.output_dir / "trace.csv", pub_dir / "trace.csv")
    shutil.copy2(paths.output_dir / "events.csv", pub_dir / "events.csv")
    if (paths.output_dir / "panorama.webp").exists():
        shutil.copy2(paths.output_dir / "panorama.webp", pub_dir / "panorama.webp")

    # Format each rectified frame into CTG-Project1 `paper.tsx` vertical layout:
    #   img.height * [0.105 .. 0.505] -> FHR [200 .. 60 bpm]
    #   img.height * [0.560 .. 0.860] -> Toco [12 .. 0 kPa] (= [100 .. 0 %])
    manifest = DatasetManifest.load(paths.manifest_path)
    px_per_cm = manifest.target_px_per_cm
    y_200 = int(round((FHR_S_TOP_CM + (210.0 - 200.0) / 20.0) * px_per_cm))
    y_60 = int(round((FHR_S_TOP_CM + (210.0 - 60.0) / 20.0) * px_per_cm))
    y_t100 = int(round(TOCO_S_TOP_CM * px_per_cm))
    y_t0 = int(round(TOCO_S_BOT_CM * px_per_cm))

    exported_scans = 0
    out_h = 1000
    for entry in manifest.frames:
        stem = Path(entry.filename).stem
        rect_p = paths.rectified_dir / f"{stem}_rectified.png"
        if not rect_p.exists():
            continue
        rbgr = cv2.imread(str(rect_p), cv2.IMREAD_COLOR)
        if rbgr is None:
            continue
        fw = rbgr.shape[1]
        out_w = max(64, int(round(fw * (400.0 / max(1, y_60 - y_200)))))
        formatted = np.full((out_h, out_w, 3), (231, 241, 243), dtype=np.uint8)
        fhr_crop = cv2.resize(rbgr[y_200:y_60, :], (out_w, 400), interpolation=cv2.INTER_AREA)
        toco_crop = cv2.resize(rbgr[y_t100:y_t0, :], (out_w, 300), interpolation=cv2.INTER_AREA)
        formatted[105:505, :] = fhr_crop
        formatted[560:860, :] = toco_crop
        cv2.imwrite(str(scans_dir / f"{stem}.jpg"), formatted, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        exported_scans += 1

    return {
        "target_repo": str(target_repo_dir),
        "session_json": str(out_json),
        "trace_csv": str(pub_dir / "trace.csv"),
        "events_csv": str(pub_dir / "events.csv"),
        "scans_exported": exported_scans,
    }
