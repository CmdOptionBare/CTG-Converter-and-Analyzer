"""Dataset manifest schema and standardized folder layout resolver."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class FrameEntry:
    """Metadata for a single ordered input frame in a CTG dataset."""

    index: int
    filename: str
    original_filename: str
    width: int
    height: int
    capture_timestamp_hint: str | None = None
    rotation_override_deg: int | None = None
    manual_dx_override: float | None = None
    manual_dy_override: float | None = None


@dataclass
class DatasetManifest:
    """Self-contained description of one CTG strip recording dataset."""

    dataset_id: str
    split: str = "test"
    source_type: str = "camera_sequence"
    device_vendor: str = "Philips"
    device_model: str = "M2702A / M1911A"
    paper_speed_cm_per_min: float = 1.0
    fhr_range_bpm: tuple[int, int] = (60, 200)
    toco_range_units: tuple[int, int] = (0, 100)
    target_px_per_cm: int = 100
    notes: str = ""
    frames: list[FrameEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fhr_range_bpm"] = list(self.fhr_range_bpm)
        data["toco_range_units"] = list(self.toco_range_units)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        frames = [FrameEntry(**f) for f in data.get("frames", [])]
        return cls(
            dataset_id=data["dataset_id"],
            split=data.get("split", "test"),
            source_type=data.get("source_type", "camera_sequence"),
            device_vendor=data.get("device_vendor", "Philips"),
            device_model=data.get("device_model", ""),
            paper_speed_cm_per_min=float(data.get("paper_speed_cm_per_min", 1.0)),
            fhr_range_bpm=tuple(data.get("fhr_range_bpm", [60, 200])),  # type: ignore[arg-type]
            toco_range_units=tuple(data.get("toco_range_units", [0, 100])),  # type: ignore[arg-type]
            target_px_per_cm=int(data.get("target_px_per_cm", 100)),
            notes=data.get("notes", ""),
            frames=frames,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved input, interim, and output directories for a dataset."""

    data_root: Path
    split: str
    dataset_id: str

    @property
    def input_dir(self) -> Path:
        return self.data_root / "inputs" / self.split / self.dataset_id

    @property
    def manifest_path(self) -> Path:
        return self.input_dir / "manifest.json"

    @property
    def raw_dir(self) -> Path:
        return self.input_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim" / self.split / self.dataset_id

    @property
    def enhanced_dir(self) -> Path:
        return self.interim_dir / "01_enhanced"

    @property
    def rectified_dir(self) -> Path:
        return self.interim_dir / "02_rectified"

    @property
    def matches_dir(self) -> Path:
        return self.interim_dir / "03_matches"

    @property
    def output_dir(self) -> Path:
        return self.data_root / "outputs" / self.split / self.dataset_id

    def ensure_work_dirs(self) -> None:
        for directory in (
            self.enhanced_dir,
            self.rectified_dir,
            self.matches_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_input_dir(cls, input_dir: Path, data_root: Path | None = None) -> DatasetPaths:
        input_dir = input_dir.resolve()
        manifest_file = input_dir / "manifest.json"
        if manifest_file.exists():
            manifest = DatasetManifest.load(manifest_file)
            inferred_root = input_dir.parents[2] if data_root is None else data_root.resolve()
            return cls(
                data_root=inferred_root,
                split=manifest.split,
                dataset_id=manifest.dataset_id,
            )
        split = input_dir.parent.name
        dataset_id = input_dir.name
        inferred_root = input_dir.parents[2] if data_root is None else data_root.resolve()
        return cls(data_root=inferred_root, split=split, dataset_id=dataset_id)


def list_datasets(data_root: Path) -> list[DatasetPaths]:
    """Discover all datasets under ``data_root/inputs/<split>/<dataset_id>``."""
    inputs_root = data_root.resolve() / "inputs"
    if not inputs_root.exists():
        return []
    found: list[DatasetPaths] = []
    for manifest_file in sorted(inputs_root.glob("*/*/manifest.json")):
        dataset_dir = manifest_file.parent
        found.append(DatasetPaths.from_input_dir(dataset_dir, data_root=data_root.resolve()))
    return found
