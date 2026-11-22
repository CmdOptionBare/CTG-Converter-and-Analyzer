"""Checks that do not trust the pipeline's own confidence numbers.

Thresholds are the measured properties of data/outputs/test/2026-11-22_philips_ctg_001,
with room for a 1 px grid wobble and a few percent of ink. They are not fitted
to hide a failure: a seam error of one minute (100 px) or a tracker that leaves
the pen fails these.
"""

from __future__ import annotations

from pathlib import Path

from ctg_converter.dataset.manifest import DatasetPaths
from ctg_converter.pipelines.validate import run_validation

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PATHS = DatasetPaths(REPO_ROOT / "data", "test", "2026-11-22_philips_ctg_001")


def test_real_panorama_against_the_printed_paper() -> None:
    report = run_validation(REAL_PATHS)
    assert (REAL_PATHS.output_dir / "validation_report.json").exists()

    # Eight stamp blocks: one at about 1 minute, then one every 10 minutes.
    # The first gap is that 9-minute step (measured 879 px, against 900).
    # The other six are the 10-minute ruler (measured 969 to 1002 px).
    axis = report["time_axis"]
    assert len(axis["stamps"]) == 8
    assert len(axis["ten_minute_spacings_px"]) == 6
    for spacing in axis["ten_minute_spacings_px"]:
        assert 960.0 <= spacing <= 1040.0
    assert axis["max_abs_deviation_px"] <= 40.0
    short = [s for s in axis["spacings_px"] if s < 930.0]
    assert len(short) == 1
    assert 850.0 <= short[0] <= 930.0

    # Printed grid vs the calibrated rows. FHR measured max |dy| = 1 px,
    # toco max |dy| = 1.5 px.
    grid = report["vertical_grid"]
    assert grid["fhr_abs_median_px"] <= 1.0
    assert grid["fhr_abs_max_px"] <= 2.0
    assert grid["toco_abs_median_px"] <= 1.5
    assert grid["toco_abs_max_px"] <= 2.0

    # Green kPa digits, re-measured: 0.1327 kPa per percent-unit.
    # 100 mmHg = 13.3 kPa, so 0.133, not the old 0.12 assumption.
    factor = report["kpa_scale"]["kpa_per_pct"]
    assert factor is not None
    assert 0.125 <= factor <= 0.142
    assert 0.130 <= report["toco_kpa_per_pct_in_session"] <= 0.140

    # The extracted line sits on ink. Interior skips the paper edge.
    # Measured interior fractions: FHR 0.93, MHR 0.97, toco 0.999.
    ink = report["curves_on_ink"]
    assert ink["fhr"]["on_ink_fraction_interior"] >= 0.88
    assert ink["mhr"]["on_ink_fraction_interior"] >= 0.90
    assert ink["toco"]["on_ink_fraction_interior"] >= 0.97

    fhr = report["series"]["fhr"]
    mhr = report["series"]["mhr"]
    pct = report["series"]["tocoPct"]
    toco = report["series"]["toco"]
    # The first ~4.5 min is the time stamp and the 110 bpm grid line, now null.
    # Measured first fetal sample at index 268, value 139 bpm.
    assert 200 <= fhr["first_index"] <= 340
    assert 125.0 <= fhr["first_value"] <= 155.0
    assert 0.04 <= fhr["null_fraction"] <= 0.12
    assert 145.0 <= fhr["median"] <= 170.0
    assert fhr["max"] <= 190.0
    assert fhr["min"] >= 100.0
    # Maternal rate stays in the 70-90 band. A swap onto the fetal trace
    # would push the max well above 110.
    assert 74.0 <= mhr["median"] <= 88.0
    assert mhr["max"] <= 110.0
    assert mhr["min"] >= 60.0
    # Toco baseline is a low line. The fall onto the zero line in the
    # PHILIPS gap is null, so the minimum stays above a few percent.
    assert 16.0 <= pct["median"] <= 26.0
    assert pct["min"] >= 5.0
    assert pct["max"] <= 50.0
    assert 2.0 <= toco["median"] <= 3.6
