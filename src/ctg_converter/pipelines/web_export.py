"""Export an extracted CTG session into the in-repo CTG-Blatt web app.

The viewer draws the photographed panorama under the extracted curves, so the
session JSON carries the stitch calibration (pixel rows and px/cm) and the
paper geometry from ``rectify.py``. Nothing in the TypeScript app should
restate those numbers.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from ctg_converter.dataset.manifest import DatasetPaths
from ctg_converter.pipelines.rectify import (
    CANONICAL_FHR_HEIGHT_CM,
    CANONICAL_SEP_HEIGHT_CM,
    CANONICAL_TOCO_HEIGHT_CM,
    FHR_LINES_CM,
    TOCO_LINES_CM,
)

# Fine vertical lines use the same 0.25 cm module as the FHR comb
# (2.5 mm = 15 s at 1 cm/min). rectify.py builds that comb as FHR_LINES_CM.
_FHR_MINOR_CM = float(FHR_LINES_CM[1] - FHR_LINES_CM[0])
_TOCO_MINOR_CM = float(TOCO_LINES_CM[1] - TOCO_LINES_CM[0])

_PROVENANCE_NOTE = (
    "Die Werte kommen aus der deterministischen Bildpipeline: das Papier wird entzerrt, "
    "zu einem Panorama gefügt und FHF, Puls und Toco werden auf diesem Bild verfolgt, "
    "eine Zahl pro Sekunde. Es liest keine KI Stützstellen ab. Die gezeichnete Linie ist "
    "diese Serie; Original legt das Panorama darunter, damit sie auf der Druckfarbe geprüft werden kann."
)


def default_web_dir() -> Path:
    """Repository ``web/`` directory next to ``src/``."""
    return Path(__file__).resolve().parents[3] / "web"


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _panorama_block(stitch: dict[str, Any], image_size: tuple[int, int], session: dict[str, Any]) -> dict[str, Any]:
    """Image placement plus paper geometry, scaled into the file the viewer loads."""
    cal = stitch["canonical_calibration"]
    meta_w = float(stitch["panorama_width_px"])
    meta_h = float(stitch["panorama_height_px"])
    width_px, height_px = image_size
    scale_x = (width_px / meta_w) if meta_w else 1.0
    scale_y = (height_px / meta_h) if meta_h else 1.0
    px_per_cm = float(stitch["target_px_per_cm"]) * scale_x

    fhr_scale = session.get("fhrScale") or {}
    toco_px_per_unit = float(cal["toco_px_per_unit"])
    toco_top = (float(cal["toco_bottom_0_y_px"]) - float(cal["toco_top_100_y_px"])) / toco_px_per_unit

    return {
        "file": "panorama.webp",
        "widthPx": int(width_px),
        "heightPx": int(height_px),
        "pxPerCm": _round(px_per_cm, 4),
        "fhr210YPx": _round(float(cal["fhr_top_210bpm_y_px"]) * scale_y, 4),
        "fhr50YPx": _round(float(cal["fhr_bottom_50bpm_y_px"]) * scale_y, 4),
        "toco100YPx": _round(float(cal["toco_top_100_y_px"]) * scale_y, 4),
        "toco0YPx": _round(float(cal["toco_bottom_0_y_px"]) * scale_y, 4),
        "fhrHeightCm": _round(CANONICAL_FHR_HEIGHT_CM, 4),
        "separatorHeightCm": _round(CANONICAL_SEP_HEIGHT_CM, 4),
        "tocoHeightCm": _round(CANONICAL_TOCO_HEIGHT_CM, 4),
        "fhrMinorCm": _round(_FHR_MINOR_CM, 4),
        "tocoMinorCm": _round(_TOCO_MINOR_CM, 4),
        "verticalMinorCm": _round(_FHR_MINOR_CM, 4),
        "fhrTopBpm": float(fhr_scale.get("max", 210)),
        "fhrBotBpm": float(fhr_scale.get("min", 50)),
        "tocoTop": _round(toco_top, 4),
        "tocoBot": 0.0,
    }


def _scans_from_placements(session: dict[str, Any], stitch: dict[str, Any]) -> list[dict[str, Any]]:
    """Time window of each frame from its panorama placement, in seconds."""
    speed = float(stitch.get("paper_speed_cm_per_min", session.get("speedCmMin", 1.0)))
    px_per_sec = float(stitch["target_px_per_cm"]) * speed / 60.0
    duration = float(session["durationSec"])
    existing: dict[str, dict[str, Any]] = {}
    for scan in session.get("scans") or []:
        existing[Path(str(scan.get("file", ""))).stem] = dict(scan)

    scans: list[dict[str, Any]] = []
    for placement in stitch.get("frame_placements") or []:
        stem = Path(str(placement["filename"])).stem
        scan = existing.get(stem, {})
        t0 = max(0.0, float(placement["x_px"]) / px_per_sec)
        t1 = min(duration, float(placement["x_px"] + placement["width_px"]) / px_per_sec)
        scan["file"] = scan.get("file") or f"{stem}.jpg"
        scan["t0"] = int(round(t0))
        scan["t1"] = int(round(t1))
        scan.setdefault("trust", "hoch")
        scan.pop("source_frame", None)
        scans.append(scan)
    return scans


_SERIES_KEYS = ("fhr", "mhr", "toco", "tocoPct")


def validate_session(session: dict[str, Any]) -> None:
    """Raise ``ValueError`` naming the first thing the viewer could not draw."""
    duration = session.get("durationSec")
    dt = session.get("dt")
    for name, value in (("durationSec", duration), ("dt", dt)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if dt <= 0 or duration < 0:
        raise ValueError(f"dt must be positive and durationSec non-negative, got dt={dt}, durationSec={duration}")

    expected = int(round(duration / dt)) + 1
    for key in _SERIES_KEYS:
        series = session.get(key)
        if not isinstance(series, list):
            raise ValueError(f"{key} must be a list")
        if len(series) != expected:
            raise ValueError(f"{key} has {len(series)} samples, expected {expected} for durationSec={duration}, dt={dt}")
        for index, value in enumerate(series):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key}[{index}] must be a finite number or null, got {value!r}")

    for index, event in enumerate(session.get("events") or []):
        t = event.get("t") if isinstance(event, dict) else None
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= duration:
            raise ValueError(f"events[{index}].t must lie within 0..{duration}, got {t!r}")


def regenerate_sessions_ts(web_dir: Path) -> Path:
    """Rewrite ``src/data/sessions.ts`` so every JSON in ``src/data/ctg/`` is listed."""
    data_dir = web_dir / "src" / "data" / "ctg"
    data_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in data_dir.glob("*.json") if p.is_file())
    lines = [
        "// Generated by ctg-tool export-web. Lists every session JSON in this folder.",
        'import type { CtgSession } from "@/lib/ctg/types";',
        "",
    ]
    names: list[str] = []
    for index, path in enumerate(files):
        name = f"session{index}"
        names.append(name)
        lines.append(f'import {name} from "@/data/ctg/{path.name}";')
    lines.append("")
    lines.append("export const sessions: CtgSession[] = [")
    for name in names:
        lines.append(f"  {name} as CtgSession,")
    lines.append("];")
    lines.append("")
    out = web_dir / "src" / "data" / "sessions.ts"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def export_to_web_app(
    paths: DatasetPaths,
    web_dir: Path,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Write the session JSON, panorama, and CSVs into a CTG-Blatt web app tree.

    ``web_dir`` is the web app root (the repository's ``web/`` by default).
    Null samples in ``fhr``, ``mhr``, ``toco`` and ``tocoPct`` are preserved.
    ``tocoKpaPerPct`` is copied when the extractor provided it.
    """
    output_dir = paths.output_dir
    session_src = output_dir / "ctg_session.json"
    stitch_src = output_dir / "stitch_metadata.json"
    trace_src = output_dir / "trace.csv"
    events_src = output_dir / "events.csv"
    webp_src = output_dir / "panorama.webp"
    for required in (session_src, stitch_src, trace_src, events_src, webp_src):
        if not required.is_file():
            raise FileNotFoundError(f"Missing {required}. Run the stitch and extract pipelines first.")

    session = json.loads(session_src.read_text(encoding="utf-8"))
    stitch = json.loads(stitch_src.read_text(encoding="utf-8"))
    validate_session(session)
    if session_id:
        session["id"] = session_id
    sid = str(session["id"])

    with Image.open(webp_src) as image:
        image_size = (int(image.size[0]), int(image.size[1]))

    session["panorama"] = _panorama_block(stitch, image_size, session)
    session["scans"] = _scans_from_placements(session, stitch)
    provenance = dict(session.get("provenance") or {})
    provenance["note"] = _PROVENANCE_NOTE
    session["provenance"] = provenance

    web_dir = Path(web_dir)
    out_json = web_dir / "src" / "data" / "ctg" / f"{sid}.json"
    pub_dir = web_dir / "public" / "ctg" / sid
    out_json.parent.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(session, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(trace_src, pub_dir / "trace.csv")
    shutil.copy2(events_src, pub_dir / "events.csv")
    shutil.copy2(webp_src, pub_dir / "panorama.webp")

    # Per-frame scans and the old orientation file are not used by the panorama viewer.
    scans_dir = pub_dir / "scans"
    if scans_dir.exists():
        shutil.rmtree(scans_dir)
    orient = pub_dir / "orient.json"
    if orient.exists():
        orient.unlink()

    sessions_ts = regenerate_sessions_ts(web_dir)
    return {
        "web_dir": str(web_dir.resolve()),
        "session_id": sid,
        "session_json": str(out_json),
        "trace_csv": str(pub_dir / "trace.csv"),
        "events_csv": str(pub_dir / "events.csv"),
        "panorama_webp": str(pub_dir / "panorama.webp"),
        "sessions_ts": str(sessions_ts),
        "scans": len(session["scans"]),
    }
