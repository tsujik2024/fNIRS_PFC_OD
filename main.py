"""fNIRS PFC pipeline CLI.

    python main.py data/ results/
    python main.py data/ results/ --metrics sqi sci psp --sqi-threshold 2.5 --task-filter DT ST
    python main.py data/ results/ --post-walking-trim 0 --skip-zscore
"""
import argparse
import logging
import os
import sys

import matplotlib

from fnirs_PFC_2025.processing.pipeline_manager import PipelineManager
from fnirs_PFC_2025.processing.quality_control import (
    DEFAULT_PSP_THRESHOLD, DEFAULT_SCI_THRESHOLD, DEFAULT_SQI_THRESHOLD,
)


def build_parser():
    p = argparse.ArgumentParser(description="fNIRS PFC pipeline: PHOEBE (SCI/PSP) and optional SQI channel QC.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input_dir", help="Folder tree with the recordings.")
    p.add_argument("output_dir")
    p.add_argument("--fs", type=float, default=50.0, help="Sampling rate (Hz).")
    p.add_argument("--metrics", nargs="+", choices=("sqi", "sci", "psp"), default=["sci", "psp"],
                   help="Metrics that gate channel exclusion; a channel failing any is dropped. Others aren't computed.")
    p.add_argument("--sci-threshold", type=float, default=DEFAULT_SCI_THRESHOLD)
    p.add_argument("--psp-threshold", type=float, default=DEFAULT_PSP_THRESHOLD)
    p.add_argument("--sqi-threshold", type=float, default=DEFAULT_SQI_THRESHOLD)
    p.add_argument("--no-quality-filtering", action="store_true",
                   help="Score channels and report, but don't drop any.")
    p.add_argument("--exclude-failing-short-channels", action="store_true",
                   help="Also drop short channels that fail (kept by default, they're only SCR regressors).")
    p.add_argument("--post-walking-trim", type=float, default=3.0,
                   help="Seconds to skip after the walking-start marker. 0 starts exactly at the marker; "
                        "baseline and end rest are cut either way.")
    p.add_argument("--initial-crop", type=float, default=1.0, help="Seconds dropped from the start of every recording.")
    p.add_argument("--skip-diagnostic-plots", action="store_true", help="Skip the per-stage and summary plots.")
    p.add_argument("--skip-zscore", action="store_true", help="RAW output only.")
    p.add_argument("--task-filter", nargs="+", metavar="TASK", help="e.g. DT ST fTurn")
    p.add_argument("--list-tasks", action="store_true", help="List discovered task types and exit.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--quiet", "-q", action="store_true", help="No console output.")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    matplotlib.use("Agg")

    if not os.path.isdir(args.input_dir):
        parser.error(f"input directory not found: {args.input_dir}")
    if args.post_walking_trim < 0 or args.initial_crop < 0:
        parser.error("--post-walking-trim and --initial-crop can't be negative")
    os.makedirs(args.output_dir, exist_ok=True)

    handlers = [logging.FileHandler(os.path.join(args.output_dir, "fnirs_processing.log"), encoding="utf-8")]
    if not args.quiet:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=args.log_level, handlers=handlers,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    manager = PipelineManager(
        fs=args.fs,
        sqi_threshold=args.sqi_threshold,
        sci_threshold=args.sci_threshold,
        psp_threshold=args.psp_threshold,
        enabled_metrics=tuple(args.metrics),
        enable_quality_filtering=not args.no_quality_filtering,
        exclude_failing_short_channels=args.exclude_failing_short_channels,
        post_walking_trim_seconds=args.post_walking_trim,
        initial_crop_seconds=args.initial_crop,
        skip_diagnostic_plots=args.skip_diagnostic_plots,
        compute_zscore=not args.skip_zscore,
    )

    if args.list_tasks:
        grouped = manager.discover_recordings(args.input_dir, args.task_filter)
        if not grouped:
            print("No matching files found.")
            return 2
        for task, files in sorted(grouped.items()):
            print(f"{task}: {len(files)} file(s)")
        return 0

    try:
        study = manager.run(args.input_dir, args.output_dir, task_filter=args.task_filter,
                            show_progress=not args.quiet)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"\nProcessed {study.n_processed}/{study.total_files} recordings -> {args.output_dir}")
        print(f"{len(study.summary_paths)} per-task summary sheet(s) written")
    return 0 if study.n_processed else 2


if __name__ == "__main__":
    sys.exit(main())
