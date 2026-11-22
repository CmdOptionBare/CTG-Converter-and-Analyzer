"""Independent checks on a stitched CTG panorama and the extracted series.

The checks do not use the matcher's confidence. They use the printed time
stamps, the printed grid, the black-ink layer, and the printed kPa scale.
Run with ``python -m ctg_converter.pipelines.validate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ctg_converter.dataset.manifest import DatasetPaths

# Ten minutes at 1 cm/min and 100 px/cm.
_TEN_MIN_PX = 1000.0
_SEPARATOR_Y0 = 886
_SEPARATOR_Y1 = 922


def _darkness(bgr: np.ndarray) -> np.ndarray:
    return (255.0 - bgr.min(axis=2)).astype(np.float32)


def find_time_stamps(pano_bgr: np.ndarray) -> list[dict[str, float]]:
    """Left edge of each printed time-stamp block in the separator strip.

    A stamp is a run of dark ink a few hundred pixels wide. Its left edge is
    the arrow in front of the clock time. Ghosted stamps (a seam through the
    text) are wider, so the width limit is loose. The edge is what the spacing
    is measured from; the clock text itself is not read here.
    """
    band = _darkness(pano_bgr)[_SEPARATOR_Y0:_SEPARATOR_Y1]
    energy = np.convolve(band.sum(axis=0), np.ones(11) / 11.0, mode="same")
    # Blank separator sets the median. The stamp ink is several times that.
    level = float(np.median(energy)) * 3.0
    hot = energy > level
    width = int(band.shape[1])
    runs: list[tuple[int, int]] = []
    x = 0
    while x < width:
        if not hot[x]:
            x += 1
            continue
        x1 = x
        while x1 < width and hot[x1]:
            x1 += 1
        runs.append((x, x1))
        x = max(x1, x + 1)
    regions: list[tuple[int, int]] = []
    for left, right in runs:
        if regions and left - regions[-1][1] < 80:
            regions[-1] = (regions[-1][0], right)
        else:
            regions.append((left, right))
    stamps: list[dict[str, float]] = []
    for left, right in regions:
        block_w = right - left
        if 200 <= block_w <= 560 and 40 < left < width - 40:
            stamps.append({"x_px": float(left), "width_px": float(block_w)})
    return stamps


def _grid_offsets(pano_bgr: np.ndarray) -> dict[str, Any]:
    """How far each printed grid line sits from the calibrated row."""
    b, g, r = cv2.split(pano_bgr)
    green = (g.astype(np.int16) > r.astype(np.int16) + 12) & (g.astype(np.int16) > b.astype(np.int16) + 8)
    strength = green[:, 800:7000].sum(axis=1)

    def nearest(expect: float) -> float:
        y = int(round(expect))
        sl = strength[y - 4 : y + 5]
        return float((y - 4 + int(np.argmax(sl))) - expect)

    fhr = [nearest(80.0 + 25.0 * i) for i in range(33)]
    toco = [nearest(947.5 + 40.0 * i) for i in range(11)]
    return {
        "fhr_dy_px": [round(v, 1) for v in fhr],
        "fhr_abs_median_px": float(np.median(np.abs(fhr))),
        "fhr_abs_max_px": float(np.max(np.abs(fhr))),
        "toco_dy_px": [round(v, 1) for v in toco],
        "toco_abs_median_px": float(np.median(np.abs(toco))),
        "toco_abs_max_px": float(np.max(np.abs(toco))),
    }


def _kpa_factor_from_paper(pano_bgr: np.ndarray) -> dict[str, Any]:
    """kPa per percent-unit, from where the green kPa digits sit on the 0-100 grid.

    The digits 12, 10, 8, 6, 4, 2, 0 are in a column just right of the scale.
    Consecutive labels are 2 kPa apart. The 0-100 grid is 4 px per unit.
    """
    b, g, r = cv2.split(pano_bgr)
    diff = g.astype(np.int16) - r.astype(np.int16)
    mask = ((diff > 40) & (g < 210)).astype(np.uint8)
    sub = mask[940:1400, 190:235]
    num, _, stats, cent = cv2.connectedComponentsWithStats(sub, connectivity=8)
    pieces: list[float] = []
    for i in range(1, num):
        _x, _y, w, h, area = (int(stats[i, k]) for k in range(5))
        if area < 30 or h < 10:
            continue
        pieces.append(float(cent[i][1]) + 940.0)
    pieces.sort()
    clusters: list[list[float]] = []
    for y in pieces:
        if not clusters or y - clusters[-1][-1] > 18.0:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    centers = [float(np.mean(c)) for c in clusters if 960.0 <= float(np.mean(c)) <= 1360.0]
    raw_spacings = [b - a for a, b in zip(centers, centers[1:])]
    # A broken glyph can insert a gap of ~30 px. The real labels are 2 kPa
    # apart, which is about 60 px on this grid. Keep those.
    spacings = [s for s in raw_spacings if 45.0 <= s <= 80.0]
    # 2 kPa per gap, 4 px per percent-unit.
    factor = None
    if len(spacings) >= 4:
        factor = round(2.0 / (float(np.median(spacings)) / 4.0), 4)
    return {
        "label_y_px": [round(y, 1) for y in centers],
        "spacing_px": [round(s, 1) for s in spacings],
        "kpa_per_pct": factor,
        "note": (
            "Each gap is 2 kPa. The percent grid is 4 px per unit, so "
            "kPa per unit is 2 / (spacing_px / 4)."
        ),
    }


def _on_ink(trace_u8: np.ndarray, y_px: np.ndarray, rad: int = 3, thr: int = 35) -> np.ndarray:
    h, w = trace_u8.shape
    yi = np.clip(np.round(y_px).astype(np.int32), 0, h - 1)
    cols = np.arange(w)
    ok = np.zeros(w, dtype=bool)
    for d in range(-rad, rad + 1):
        yy = np.clip(yi + d, 0, h - 1)
        ok |= trace_u8[yy, cols] >= thr
    return ok


def _series_stats(values: list[Any]) -> dict[str, Any]:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    first = next((i for i, v in enumerate(values) if v is not None), None)
    return {
        "count": int(len(values)),
        "null_fraction": round(float(np.mean(~np.isfinite(arr))), 4),
        "min": None if finite.size == 0 else round(float(finite.min()), 2),
        "median": None if finite.size == 0 else round(float(np.median(finite)), 2),
        "max": None if finite.size == 0 else round(float(finite.max()), 2),
        "first_index": first,
        "first_value": None if first is None else values[first],
    }


def _low_confidence_pairs(stitch_meta: dict[str, Any], max_confidence: float = 0.45) -> list[dict[str, Any]]:
    out = []
    for pair in stitch_meta.get("pair_matches", []):
        if float(pair.get("confidence", 1.0)) >= max_confidence:
            continue
        out.append(
            {
                "left_index": pair.get("left_index"),
                "right_index": pair.get("right_index"),
                "dx": pair.get("dx"),
                "onnx_dx": pair.get("onnx_dx"),
                "confidence": pair.get("confidence"),
                "trace_ncc": pair.get("trace_ncc"),
            }
        )
    return out


def run_validation(paths: DatasetPaths) -> dict[str, Any]:
    """Measure the panorama and the extracted series. Write ``validation_report.json``."""
    out_dir = paths.output_dir
    pano = cv2.imread(str(out_dir / "panorama.png"), cv2.IMREAD_COLOR)
    trace = cv2.imread(str(out_dir / "panorama_trace.png"), cv2.IMREAD_GRAYSCALE)
    if pano is None or trace is None:
        raise FileNotFoundError(f"panorama or trace missing in {out_dir}")

    stamps = find_time_stamps(pano)
    xs = [s["x_px"] for s in stamps]
    spacings = [round(b - a, 1) for a, b in zip(xs, xs[1:])]
    # A gap near 900 px is the 9-minute mark at the start of this recording,
    # not a broken 10-minute seam. The rest are the 10-minute ruler.
    ten_min = [s for s in spacings if 930.0 <= s <= 1070.0]
    deviations = [round(s - _TEN_MIN_PX, 1) for s in ten_min]

    curves_path = out_dir / "extracted_curves_px.json"
    curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.exists() else {}
    width = trace.shape[1]
    ink: dict[str, Any] = {}
    full_x = np.arange(width)
    for key, name in (("fhr_y_px", "fhr"), ("mhr_y_px", "mhr"), ("toco_y_px", "toco")):
        if key not in curves or not curves[key]:
            continue
        y = np.interp(full_x, np.array(curves["x_px"], dtype=np.float64), np.array(curves[key], dtype=np.float64))
        ok = _on_ink(trace, y)
        interior = ok[400:7600] if width > 7600 else ok
        ink[name] = {
            "on_ink_fraction": round(float(ok.mean()), 3),
            "on_ink_fraction_interior": round(float(interior.mean()), 3),
            "tolerance_px": 3,
            "ink_min": 35,
        }

    session_path = out_dir / "ctg_session.json"
    session = json.loads(session_path.read_text(encoding="utf-8")) if session_path.exists() else {}
    series = {name: _series_stats(session[name]) for name in ("fhr", "mhr", "toco", "tocoPct") if name in session}

    stitch_path = out_dir / "stitch_metadata.json"
    stitch = json.loads(stitch_path.read_text(encoding="utf-8")) if stitch_path.exists() else {}

    report: dict[str, Any] = {
        "dataset_id": paths.dataset_id,
        "time_axis": {
            "expected_spacing_px": _TEN_MIN_PX,
            "stamps": stamps,
            "spacings_px": spacings,
            "ten_minute_spacings_px": ten_min,
            "deviation_from_1000_px": deviations,
            "max_abs_deviation_px": None if not deviations else float(np.max(np.abs(deviations))),
            "method": (
                "Left edge of each dark text block in the separator strip (y 886-922). "
                "Gaps of 930-1070 px are the 10-minute ruler (1000 px at 1 cm/min). "
                "A shorter leading gap is reported in spacings_px and is not scored against 1000."
            ),
        },
        "vertical_grid": _grid_offsets(pano),
        "kpa_scale": _kpa_factor_from_paper(pano),
        "curves_on_ink": ink,
        "series": series,
        "toco_kpa_per_pct_in_session": session.get("tocoKpaPerPct"),
        "low_confidence_pairs": _low_confidence_pairs(stitch),
        "seam_x_px": stitch.get("seam_x_positions_px"),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a stitched CTG panorama against the paper.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-id", default="2026-11-22_philips_ctg_001")
    args = parser.parse_args(argv)
    paths = DatasetPaths(Path(args.data_root), args.split, args.dataset_id)
    report = run_validation(paths)
    axis = report["time_axis"]
    grid = report["vertical_grid"]
    print(
        f"stamps {len(axis['stamps'])} spacings {axis['spacings_px']} "
        f"max |dev| {axis['max_abs_deviation_px']} px"
    )
    print(
        f"grid FHR median {grid['fhr_abs_median_px']} max {grid['fhr_abs_max_px']}; "
        f"toco median {grid['toco_abs_median_px']} max {grid['toco_abs_max_px']}"
    )
    print(f"kPa per unit from paper {report['kpa_scale']['kpa_per_pct']}")
    for name, stats in report["series"].items():
        print(f"{name} median {stats['median']} null {stats['null_fraction']} first {stats['first_index']}")


if __name__ == "__main__":
    main()
