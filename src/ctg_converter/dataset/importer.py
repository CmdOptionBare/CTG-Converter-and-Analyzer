"""Dataset importers for folder uploads, GUI imports, and legacy /images migration."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from PIL import Image

from .manifest import (
    SUPPORTED_IMAGE_EXTENSIONS,
    DatasetManifest,
    DatasetPaths,
    FrameEntry,
)

TIMESTAMP_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})"
)


def _extract_timestamp_hint(filename: str) -> str | None:
    match = TIMESTAMP_PATTERN.search(filename)
    if not match:
        return None
    yyyy, mm, dd, hh, mi, ss, ms = match.groups()
    return f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}.{ms}"


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def import_folder_as_dataset(
    source_dir: Path,
    data_root: Path,
    dataset_id: str,
    split: str = "imports",
    source_type: str = "folder_import",
    paper_speed_cm_per_min: float = 1.0,
    target_px_per_cm: int = 48,
    copy_files: bool = True,
    use_git_mv: bool = False,
    notes: str = "",
) -> DatasetPaths:
    """Import an ordered directory of CTG photos/scans into a standardized dataset folder."""
    source_dir = source_dir.resolve()
    data_root = data_root.resolve()
    image_files = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No supported image files found in {source_dir}")

    paths = DatasetPaths(data_root=data_root, split=split, dataset_id=dataset_id)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)

    frames: list[FrameEntry] = []
    for idx, src_path in enumerate(image_files, start=1):
        ext = src_path.suffix.lower()
        target_name = f"frame_{idx:04d}{ext}"
        dst_path = paths.raw_dir / target_name
        width, height = _read_image_size(src_path)

        if use_git_mv:
            subprocess.run(
                ["git", "mv", str(src_path), str(dst_path)],
                check=True,
                cwd=data_root.parent,
            )
        elif copy_files:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.move(str(src_path), str(dst_path))

        frames.append(
            FrameEntry(
                index=idx,
                filename=target_name,
                original_filename=src_path.name,
                width=width,
                height=height,
                capture_timestamp_hint=_extract_timestamp_hint(src_path.name),
            )
        )

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        split=split,
        source_type=source_type,
        paper_speed_cm_per_min=paper_speed_cm_per_min,
        target_px_per_cm=target_px_per_cm,
        notes=notes,
        frames=frames,
    )
    manifest.save(paths.manifest_path)
    return paths


def migrate_legacy_images_folder(
    repo_root: Path,
    dataset_id: str = "2026-11-22_philips_ctg_001",
    split: str = "test",
) -> DatasetPaths:
    """Move legacy ``images/*.png`` into ``data/inputs/<split>/<dataset_id>/raw/`` via ``git mv``."""
    legacy_dir = repo_root.resolve() / "images"
    data_root = repo_root.resolve() / "data"
    paths = import_folder_as_dataset(
        source_dir=legacy_dir,
        data_root=data_root,
        dataset_id=dataset_id,
        split=split,
        source_type="camera_sequence",
        paper_speed_cm_per_min=1.0,
        target_px_per_cm=48,
        copy_files=False,
        use_git_mv=True,
        notes="Initial 21-frame handheld camera test dataset of a Philips CTG strip (1 cm/min).",
    )
    if legacy_dir.exists() and not any(legacy_dir.iterdir()):
        legacy_dir.rmdir()
    return paths
