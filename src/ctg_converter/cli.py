"""Command-line interface for CTG enhancement, 90-degree rectification, panorama stitching, and GUI."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from ctg_converter.dataset import (
    DatasetPaths,
    import_folder_as_dataset,
    list_datasets,
)
from ctg_converter.pipelines.enhance import EnhanceConfig, run_enhance_pipeline
from ctg_converter.pipelines.extract import (
    ExtractConfig,
    export_to_ctg_project1,
    run_extract_pipeline,
)
from ctg_converter.pipelines.rectify import RectifyConfig, run_rectify_pipeline
from ctg_converter.pipelines.stitch import StitchConfig, run_stitch_pipeline


def _resolve_dataset_paths(args: argparse.Namespace) -> DatasetPaths:
    data_root = Path(args.data_root).resolve()
    if getattr(args, "dataset_dir", None):
        return DatasetPaths.from_input_dir(Path(args.dataset_dir), data_root=data_root)
    return DatasetPaths(
        data_root=data_root,
        split=args.split,
        dataset_id=args.dataset_id,
    )


def _peak_rss_mb() -> float:
    # On Linux ru_maxrss is in KiB
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctg-tool",
        description="CTG Strip Enhancer, 90-Degree Grid Rectifier, and Panorama Stitcher",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root data directory containing inputs/, interim/, outputs/ (default: data)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List available datasets under data/inputs/")

    # import
    p_imp = sub.add_parser("import", help="Import a folder of images as a new dataset")
    p_imp.add_argument("source_dir", help="Folder containing ordered CTG images")
    p_imp.add_argument("--dataset-id", required=True, help="Dataset identifier")
    p_imp.add_argument("--split", default="imports", help="Split folder (default: imports)")
    p_imp.add_argument("--paper-speed", type=float, default=1.0, help="Paper speed in cm/min")
    p_imp.add_argument("--px-per-cm", type=int, default=100, help="Target rectified px/cm")

    # Common dataset selector args helper
    def add_ds_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset-dir", help="Direct path to dataset input directory")
        p.add_argument("--split", default="test", help="Dataset split (default: test)")
        p.add_argument(
            "--dataset-id",
            default="2026-11-22_philips_ctg_001",
            help="Dataset ID (default: 2026-11-22_philips_ctg_001)",
        )

    # enhance
    p_enh = sub.add_parser("enhance", help="Run Pipeline 1 (image enhancement & layer separation)")
    add_ds_args(p_enh)

    # rectify
    p_rec = sub.add_parser("rectify", help="Run Pipeline 2 (90-degree metric grid rectification)")
    add_ds_args(p_rec)
    p_rec.add_argument("--px-per-cm", type=int, default=None, help="Override target px/cm")

    # stitch / run-all
    p_st = sub.add_parser("stitch", help="Run Pipeline 3 (overlap matching + PNG/WebP panorama export)")
    add_ds_args(p_st)
    p_st.add_argument("--px-per-cm", type=int, default=None, help="Override target px/cm")
    p_st.add_argument("--no-onnx", action="store_true", help="Disable ONNX patch matcher")

    p_all = sub.add_parser("run-all", help="Execute Enhance -> Rectify -> Stitch -> Extract end-to-end")
    add_ds_args(p_all)
    p_all.add_argument("--px-per-cm", type=int, default=None, help="Override target px/cm")
    p_all.add_argument("--no-onnx", action="store_true", help="Disable ONNX patch matcher")

    # extract (Pipeline 4: Image-to-Value)
    p_ext = sub.add_parser(
        "extract",
        help="Run Pipeline 4 (Viterbi FHR/MHR/Toco curve extraction -> trace.csv & ctg_session.json)",
    )
    add_ds_args(p_ext)
    p_ext.add_argument("--started-at", default="2026-11-22T11:10:12", help="ISO start timestamp on paper")
    p_ext.add_argument("--session-id", default=None, help="Session ID for CTG-Project1 JSON")

    # export-web (Bridge to CmdOptionBare/CTG-Project1)
    p_web = sub.add_parser(
        "export-web",
        help="Export calibrated CtgSession JSON, trace.csv, and scans into CmdOptionBare/CTG-Project1",
    )
    add_ds_args(p_web)
    p_web.add_argument("--target-repo", required=True, help="Path to local CTG-Project1 repository")
    p_web.add_argument("--session-id", default=None, help="Override session ID (e.g. 2026-11-22)")

    # gui
    p_gui = sub.add_parser("gui", help="Launch visual PySide6 Alignment & Stitching Studio")
    add_ds_args(p_gui)
    p_gui.add_argument(
        "--tab",
        choices=["align", "stitch"],
        default="align",
        help="Initial visual tool tab to focus (align or stitch)",
    )

    args = parser.parse_args(argv)
    t0 = time.perf_counter()

    if args.command == "list":
        datasets = list_datasets(Path(args.data_root))
        for ds in datasets:
            print(f"{ds.split}/{ds.dataset_id} -> {ds.input_dir}")
        return 0

    if args.command == "import":
        paths = import_folder_as_dataset(
            source_dir=Path(args.source_dir),
            data_root=Path(args.data_root),
            dataset_id=args.dataset_id,
            split=args.split,
            paper_speed_cm_per_min=args.paper_speed,
            target_px_per_cm=args.px_per_cm,
        )
        print(f"Imported dataset into {paths.input_dir}")
        return 0

    if args.command == "gui":
        from ctg_converter.gui.app import launch_gui

        paths = _resolve_dataset_paths(args)
        return launch_gui(
            initial_dataset=paths if paths.manifest_path.exists() else None,
            initial_tab=args.tab,
            data_root=Path(args.data_root).resolve(),
        )

    paths = _resolve_dataset_paths(args)

    if args.command == "enhance":
        res = run_enhance_pipeline(paths, cfg=EnhanceConfig())
        print(
            f"Enhanced {len(res)} frames in {time.perf_counter() - t0:.2f}s "
            f"(peak RSS {_peak_rss_mb()} MB) -> {paths.enhanced_dir}"
        )
        return 0

    if args.command == "rectify":
        rcfg = RectifyConfig(target_px_per_cm=args.px_per_cm) if args.px_per_cm else None
        res = run_rectify_pipeline(paths, rect_cfg=rcfg)
        print(
            f"Rectified {len(res)} frames in {time.perf_counter() - t0:.2f}s "
            f"(peak RSS {_peak_rss_mb()} MB) -> {paths.rectified_dir}"
        )
        return 0

    if args.command == "extract":
        ecfg = ExtractConfig(started_at_iso=args.started_at, session_id=args.session_id)
        res = run_extract_pipeline(paths, ext_cfg=ecfg)
        res["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        res["peak_rss_mb"] = _peak_rss_mb()
        print(json.dumps(res, indent=2))
        return 0

    if args.command == "export-web":
        out = export_to_ctg_project1(
            paths,
            target_repo_dir=Path(args.target_repo),
            session_id=args.session_id,
        )
        out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(out, indent=2))
        return 0

    if args.command in ("stitch", "run-all"):
        if args.command == "run-all":
            run_enhance_pipeline(paths, cfg=EnhanceConfig())
            rcfg = RectifyConfig(target_px_per_cm=args.px_per_cm) if args.px_per_cm else None
            run_rectify_pipeline(paths, rect_cfg=rcfg)

        rcfg = RectifyConfig(target_px_per_cm=args.px_per_cm) if args.px_per_cm else None
        scfg = StitchConfig(use_onnx_matcher=not args.no_onnx)
        report = run_stitch_pipeline(paths, stitch_cfg=scfg, rect_cfg=rcfg)
        ext_summary = run_extract_pipeline(paths, ext_cfg=ExtractConfig())
        summary = {
            "dataset_id": report["dataset_id"],
            "panorama_size": f"{report['panorama_width_px']}x{report['panorama_height_px']}",
            "strip_length_cm": report["estimated_strip_length_cm"],
            "duration_minutes": report["estimated_duration_minutes"],
            "png": report["exports"]["png"],
            "webp": report["exports"]["webp"],
            "ctg_session_json": ext_summary["outputs"]["ctg_session_json"],
            "trace_csv": ext_summary["outputs"]["trace_csv"],
            "fhr_median_bpm": ext_summary["fhr_median_bpm"],
            "mhr_median_bpm": ext_summary["mhr_median_bpm"],
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "peak_rss_mb": _peak_rss_mb(),
        }
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
