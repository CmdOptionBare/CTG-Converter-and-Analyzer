"""Dataset schema, manifest handling, and multi-source importers."""

from .importer import import_folder_as_dataset, migrate_legacy_images_folder
from .manifest import DatasetManifest, DatasetPaths, FrameEntry, list_datasets

__all__ = [
    "DatasetManifest",
    "DatasetPaths",
    "FrameEntry",
    "import_folder_as_dataset",
    "list_datasets",
    "migrate_legacy_images_folder",
]
