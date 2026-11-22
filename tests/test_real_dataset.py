"""Integration tests validating results on the real 21-frame Philips CTG dataset."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ctg_converter.dataset import DatasetManifest, DatasetPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PATHS = DatasetPaths(REPO_ROOT / "data", "test", "2026-11-22_philips_ctg_001")


def test_real_dataset_manifest_and_raw_frames() -> None:
    assert REAL_PATHS.manifest_path.exists()
    manifest = DatasetManifest.load(REAL_PATHS.manifest_path)
    assert len(manifest.frames) == 21
    for entry in manifest.frames:
        assert (REAL_PATHS.raw_dir / entry.filename).exists()


def test_real_dataset_rectification_and_panorama_outputs() -> None:
    rect_summary_path = REAL_PATHS.rectified_dir / "summary.json"
    stitch_meta_path = REAL_PATHS.output_dir / "stitch_metadata.json"
    assert rect_summary_path.exists(), "Run `ctg-tool stitch` before integration assertions"
    assert stitch_meta_path.exists()

    rect_items = json.loads(rect_summary_path.read_text(encoding="utf-8"))
    assert len(rect_items) == 21
    for item in rect_items:
        assert item["rectified_height"] == 1420
        assert abs(item["orthogonal_angle_deg"] - 90.0) <= 0.65
        assert 23.0 <= item["measured_grid_period_x_px"] <= 26.8
        assert 23.5 <= item["measured_grid_period_y_px"] <= 26.5

    stitch_meta = json.loads(stitch_meta_path.read_text(encoding="utf-8"))
    assert stitch_meta["num_frames"] == 21
    assert 7200 <= stitch_meta["panorama_width_px"] <= 8400
    assert stitch_meta["panorama_height_px"] == 1420
    assert len(stitch_meta["pair_matches"]) == 20

    for pm in stitch_meta["pair_matches"]:
        assert pm["dx"] > 100
        assert abs(pm["dy"]) <= 3
        assert pm["onnx_inliers"] >= 10
        assert pm["method_used"] == "hybrid_ncc+onnx"

    png_path = Path(stitch_meta["exports"]["png"])
    webp_path = Path(stitch_meta["exports"]["webp"])
    assert png_path.exists() and png_path.stat().st_size > 1_000_000
    assert webp_path.exists() and webp_path.stat().st_size > 200_000

    with Image.open(png_path) as im_png:
        assert im_png.size == (stitch_meta["panorama_width_px"], 1420)
    with Image.open(webp_path) as im_webp:
        assert im_webp.size == (stitch_meta["panorama_width_px"], 1420)


def test_real_dataset_extraction() -> None:
    session_path = REAL_PATHS.output_dir / "ctg_session.json"
    csv_path = REAL_PATHS.output_dir / "trace.csv"
    assert session_path.exists()
    assert csv_path.exists()

    sess = json.loads(session_path.read_text(encoding="utf-8"))
    assert sess["id"] == "2026-11-22"
    assert sess["dt"] == 1
    assert 4300 <= sess["durationSec"] <= 5100
    for key in ("fhr", "mhr", "toco", "tocoPct"):
        assert len(sess[key]) == sess["durationSec"] + 1
    assert len(sess["scans"]) == 21

    # The panorama starts with blank leader and the printed header, so the
    # heart-rate channels must be empty there instead of inventing values.
    assert all(v is None for v in sess["fhr"][:200])
    assert all(v is None for v in sess["mhr"][:200])

    fhr = sorted(v for v in sess["fhr"] if v is not None)
    mhr = sorted(v for v in sess["mhr"] if v is not None)
    assert len(fhr) >= 0.85 * len(sess["fhr"])
    assert len(mhr) >= 0.85 * len(sess["mhr"])
    assert 135.0 <= fhr[len(fhr) // 2] <= 172.0
    assert 72.0 <= mhr[len(mhr) // 2] <= 90.0

    # Header 11:10:12, first separator stamp 11:11 at about x = 490 px:
    # the left edge of the panorama is then close to 11:06:06.
    assert sess["recordingStartedAt"] == "2026-11-22T11:10:12"
    assert "2026-11-22T11:05:50" <= sess["startedAt"] <= "2026-11-22T11:06:20"
