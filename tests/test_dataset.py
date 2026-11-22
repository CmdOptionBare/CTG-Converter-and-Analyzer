"""Unit tests for dataset manifest schema, folder importer, and dataset discovery."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ctg_converter.dataset import (
    DatasetManifest,
    DatasetPaths,
    import_folder_as_dataset,
    list_datasets,
)


def test_import_folder_and_manifest_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "raw_photos"
    src.mkdir()
    for i, ts in enumerate(["2026-11-22-12-22-30-821", "2026-11-22-12-22-34-492"], start=1):
        img = np.full((120, 90, 3), 240, dtype=np.uint8)
        cv2.imwrite(str(src / f"{ts}.png"), img)

    data_root = tmp_path / "data"
    paths = import_folder_as_dataset(
        source_dir=src,
        data_root=data_root,
        dataset_id="unit_ctg_01",
        split="imports",
        paper_speed_cm_per_min=1.0,
        target_px_per_cm=100,
    )

    assert paths.manifest_path.exists()
    manifest = DatasetManifest.load(paths.manifest_path)
    assert manifest.dataset_id == "unit_ctg_01"
    assert len(manifest.frames) == 2
    assert manifest.frames[0].filename == "frame_0001.png"
    assert manifest.frames[0].capture_timestamp_hint == "2026-11-22T12:22:30.821"
    assert (paths.raw_dir / "frame_0001.png").exists()
    assert (paths.raw_dir / "frame_0002.png").exists()

    discovered = list_datasets(data_root)
    assert len(discovered) == 1
    assert discovered[0].dataset_id == "unit_ctg_01"
    resolved = DatasetPaths.from_input_dir(paths.input_dir, data_root=data_root)
    assert resolved.output_dir == data_root / "outputs" / "imports" / "unit_ctg_01"
