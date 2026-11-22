"""Export of an extracted CTG session into the CTG-Blatt web app."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ctg_converter.cli import main
from ctg_converter.dataset import DatasetPaths
from ctg_converter.pipelines.rectify import (
    CANONICAL_FHR_HEIGHT_CM,
    CANONICAL_SEP_HEIGHT_CM,
    CANONICAL_TOCO_HEIGHT_CM,
)
from ctg_converter.pipelines.web_export import default_web_dir, export_to_web_app, validate_session


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PATHS = DatasetPaths(REPO_ROOT / "data", "test", "2026-11-22_philips_ctg_001")


def test_default_web_dir_is_repo_web() -> None:
    assert default_web_dir() == REPO_ROOT / "web"


def test_export_real_dataset_writes_calibration(tmp_path: Path) -> None:
    web = tmp_path / "web"
    stale = web / "public" / "ctg" / "2026-11-22" / "scans"
    stale.mkdir(parents=True)
    (stale / "old.jpg").write_bytes(b"not-a-scan")
    (web / "public" / "ctg" / "2026-11-22" / "orient.json").write_text("{}\n", encoding="utf-8")

    out = export_to_web_app(REAL_PATHS, web)
    sid = "2026-11-22"
    session_path = Path(out["session_json"])
    pub = web / "public" / "ctg" / sid

    assert out["scans"] == 21
    assert session_path == web / "src" / "data" / "ctg" / f"{sid}.json"
    assert (pub / "trace.csv").read_bytes() == (REAL_PATHS.output_dir / "trace.csv").read_bytes()
    assert (pub / "events.csv").read_bytes() == (REAL_PATHS.output_dir / "events.csv").read_bytes()
    assert (pub / "panorama.webp").read_bytes() == (REAL_PATHS.output_dir / "panorama.webp").read_bytes()
    assert not (pub / "scans").exists()
    assert not (pub / "orient.json").exists()

    source = json.loads((REAL_PATHS.output_dir / "ctg_session.json").read_text(encoding="utf-8"))
    stitch = json.loads((REAL_PATHS.output_dir / "stitch_metadata.json").read_text(encoding="utf-8"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    pano = session["panorama"]

    with Image.open(pub / "panorama.webp") as image:
        assert (pano["widthPx"], pano["heightPx"]) == image.size
    cal = stitch["canonical_calibration"]
    assert pano["file"] == "panorama.webp"
    assert pano["pxPerCm"] == stitch["target_px_per_cm"]
    assert pano["fhr210YPx"] == cal["fhr_top_210bpm_y_px"]
    assert pano["fhr50YPx"] == cal["fhr_bottom_50bpm_y_px"]
    assert pano["toco100YPx"] == cal["toco_top_100_y_px"]
    assert pano["toco0YPx"] == cal["toco_bottom_0_y_px"]
    assert pano["fhrHeightCm"] == CANONICAL_FHR_HEIGHT_CM
    assert pano["separatorHeightCm"] == CANONICAL_SEP_HEIGHT_CM
    assert pano["tocoHeightCm"] == CANONICAL_TOCO_HEIGHT_CM
    assert pano["fhrMinorCm"] == 0.25
    assert pano["tocoMinorCm"] == 0.4
    assert pano["verticalMinorCm"] == 0.25
    assert pano["fhrTopBpm"] == source["fhrScale"]["max"]
    assert pano["fhrBotBpm"] == source["fhrScale"]["min"]
    assert pano["tocoTop"] == 100
    assert pano["tocoBot"] == 0

    assert session["fhr"] == source["fhr"]
    assert session["mhr"] == source["mhr"]
    assert session["tocoPct"] == source["tocoPct"]
    assert session["toco"] == source["toco"]
    assert session.get("tocoKpaPerPct") == source.get("tocoKpaPerPct")
    assert "keine KI" in session["provenance"]["note"]
    assert "Panorama" in session["provenance"]["note"]
    assert session["provenance"]["note"] != source["provenance"]["note"]

    px_per_sec = stitch["target_px_per_cm"] * stitch["paper_speed_cm_per_min"] / 60.0
    assert len(session["scans"]) == len(stitch["frame_placements"])
    for scan, placement in zip(session["scans"], stitch["frame_placements"], strict=True):
        t0 = int(round(max(0.0, placement["x_px"] / px_per_sec)))
        t1 = int(
            round(
                min(
                    float(source["durationSec"]),
                    (placement["x_px"] + placement["width_px"]) / px_per_sec,
                )
            )
        )
        assert scan["t0"] == t0
        assert scan["t1"] == t1
        assert Path(scan["file"]).stem == Path(placement["filename"]).stem
    assert session["scans"][0]["t0"] == 0
    assert session["scans"][0]["t1"] == 665

    sessions_ts = (web / "src" / "data" / "sessions.ts").read_text(encoding="utf-8")
    assert 'import session0 from "@/data/ctg/2026-11-22.json";' in sessions_ts
    assert "session0 as CtgSession" in sessions_ts


def test_export_preserves_nulls_scales_image_and_lists_every_session(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = data / "outputs" / "test" / "demo"
    output.mkdir(parents=True)
    session = {
        "id": "null-demo",
        "title": "Null",
        "startedAt": "2026-11-22T11:10:12",
        "durationSec": 2,
        "dt": 1,
        "device": "Philips M2702A",
        "paper": "M1911A",
        "speedCmMin": 1.0,
        "fhrScale": {"min": 50, "max": 210, "major": 20, "label": "bpm"},
        "tocoScale": {"min": 0, "max": 12, "major": 2, "label": "kPa"},
        "provenance": {"trace": "t", "scans": "s", "note": "alte Notiz"},
        "fhr": [140.0, None, 142.0],
        "mhr": [80.0, None, None],
        "toco": [1.2, None, 1.4],
        "tocoPct": [10.0, None, 12.0],
        "tocoKpaPerPct": 0.12,
        "events": [],
        "scans": [{"file": "frame_0001.jpg", "t0": 9, "t1": 9, "trust": "hoch"}],
    }
    stitch = {
        "panorama_width_px": 20,
        "panorama_height_px": 10,
        "target_px_per_cm": 100,
        "paper_speed_cm_per_min": 1.0,
        "canonical_calibration": {
            "fhr_top_210bpm_y_px": 2.0,
            "fhr_bottom_50bpm_y_px": 6.0,
            "fhr_px_per_bpm": 0.025,
            "toco_top_100_y_px": 7.0,
            "toco_bottom_0_y_px": 9.0,
            "toco_px_per_unit": 0.02,
        },
        "frame_placements": [
            {"filename": "frame_0001.png", "x_px": 0, "width_px": 20, "y_px": 0, "height_px": 10}
        ],
    }
    (output / "ctg_session.json").write_text(json.dumps(session), encoding="utf-8")
    (output / "stitch_metadata.json").write_text(json.dumps(stitch), encoding="utf-8")
    (output / "trace.csv").write_text("t_sec\n", encoding="utf-8")
    (output / "events.csv").write_text("t_sec\n", encoding="utf-8")
    Image.new("RGB", (10, 5), (240, 240, 230)).save(output / "panorama.webp", format="WEBP")

    web = tmp_path / "web"
    (web / "src" / "data" / "ctg").mkdir(parents=True)
    (web / "src" / "data" / "ctg" / "aaa.json").write_text("{}\n", encoding="utf-8")

    out = export_to_web_app(DatasetPaths(data, "test", "demo"), web, session_id="custom")
    written = json.loads(Path(out["session_json"]).read_text(encoding="utf-8"))
    assert written["id"] == "custom"
    assert written["fhr"] == [140.0, None, 142.0]
    assert written["mhr"] == [80.0, None, None]
    assert written["toco"] == [1.2, None, 1.4]
    assert written["tocoPct"] == [10.0, None, 12.0]
    assert written["tocoKpaPerPct"] == 0.12
    assert written["scans"][0]["t0"] == 0
    assert written["scans"][0]["t1"] == 2
    assert written["scans"][0]["trust"] == "hoch"
    pano = written["panorama"]
    assert (pano["widthPx"], pano["heightPx"]) == (10, 5)
    assert pano["pxPerCm"] == 50
    assert pano["fhr210YPx"] == 1
    assert pano["fhr50YPx"] == 3
    assert pano["toco100YPx"] == 3.5
    assert pano["toco0YPx"] == 4.5
    assert "alte Notiz" not in written["provenance"]["note"]

    text = (web / "src" / "data" / "sessions.ts").read_text(encoding="utf-8")
    assert text.index("aaa.json") < text.index("custom.json")
    assert "session0 as CtgSession" in text
    assert "session1 as CtgSession" in text


def test_cli_export_web_defaults_are_optional(tmp_path: Path) -> None:
    rc = main(
        [
            "--data-root",
            str(REPO_ROOT / "data"),
            "export-web",
            "--web-dir",
            str(tmp_path / "web"),
            "--split",
            "test",
            "--dataset-id",
            "2026-11-22_philips_ctg_001",
        ]
    )
    assert rc == 0
    assert (tmp_path / "web" / "public" / "ctg" / "2026-11-22" / "panorama.webp").is_file()
    assert (tmp_path / "web" / "src" / "data" / "sessions.ts").is_file()


def _minimal_session() -> dict:
    return {
        "durationSec": 2,
        "dt": 1,
        "fhr": [140.0, None, 142.0],
        "mhr": [80.0, 81.0, None],
        "toco": [1.2, 1.3, 1.4],
        "tocoPct": [10.0, 11.0, 12.0],
        "events": [{"t": 1, "kind": "kineto", "text": "x"}],
    }


def test_validate_session_accepts_nulls() -> None:
    validate_session(_minimal_session())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["mhr"].pop(), "mhr has 2 samples, expected 3"),
        (lambda s: s["fhr"].__setitem__(0, float("nan")), r"fhr\[0\] must be a finite number or null"),
        (lambda s: s["toco"].__setitem__(2, "1.4"), r"toco\[2\] must be a finite number or null"),
        (lambda s: s["events"][0].__setitem__("t", 3), r"events\[0\]\.t must lie within 0\.\.2"),
        (lambda s: s.__setitem__("dt", 0), "dt must be positive"),
    ],
)
def test_validate_session_names_the_first_problem(mutate, message) -> None:
    session = _minimal_session()
    mutate(session)
    with pytest.raises(ValueError, match=message):
        validate_session(session)
