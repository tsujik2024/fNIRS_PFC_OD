"""Command-line entry point for the fNIRS prefrontal-cortex pipeline.

Runs a whole study folder: discovers recordings, and processes each through
``FileProcessor``, which now performs channel-quality scoring and filtering
internally (any combination of SQI/SCI/PSP - there is no separate prefilter
stage anymore), then writes per-task statistics, a QC roll-up, and a
processing report.

Examples
--------
Process a study tree with the default quality metrics (SCI < 0.75 or
PSP < 0.10 rejected; SQI not computed)::

    python main.py data/ results/

Override the QC thresholds and restrict to two task types::

    python main.py data/ results/ --sci-threshold 0.80 --psp-threshold 0.10 \
        --task-filter DT ST

Also compute and filter on SQI, alongside SCI/PSP::

    python main.py data/ results/ --metrics sqi sci psp --sqi-threshold 2.5

Compute and report quality without dropping any channels::

    python main.py data/ results/ --no-quality-filtering
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

from fnirs_PFC_2025.processing import PipelineManager
from fnirs_PFC_2025.processing.quality_control import (
    DEFAULT_PSP_THRESHOLD,
    DEFAULT_SCI_THRESHOLD,
    DEFAULT_SQI_THRESHOLD,
)

_METRIC_CHOICES = ("sqi", "sci", "psp")
_DEFAULT_METRICS = ("sci", "psp")  # SQI is opt-in


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="fNIRS PFC processing pipeline with SQI/SCI/PSP channel quality control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory tree containing recordings.")
    parser.add_argument("output_dir", help="Directory for processed outputs.")

    parser.add_argument("--fs", type=float, default=50.0, help="Sampling rate (Hz).")

    parser.add_argument("--metrics", nargs="+", metavar="METRIC",
                        choices=_METRIC_CHOICES, default=list(_DEFAULT_METRICS),
                        help=f"Which quality metrics gate channel exclusion "
                             f"(choices: {', '.join(_METRIC_CHOICES)}). A channel is "
                             f"excluded if it fails ANY metric listed here. Metrics not "
                             f"listed are not computed at all.")
    parser.add_argument("--sci-threshold", type=float, default=DEFAULT_SCI_THRESHOLD,
                        help="Reject channels with SCI below this value (only used if "
                             "'sci' is in --metrics).")
    parser.add_argument("--psp-threshold", type=float, default=DEFAULT_PSP_THRESHOLD,
                        help="Reject channels with PSP below this value (only used if "
                             "'psp' is in --metrics).")
    parser.add_argument("--sqi-threshold", type=float, default=DEFAULT_SQI_THRESHOLD,
                        help="Reject channels with SQI below this value (only used if "
                             "'sqi' is in --metrics).")
    parser.add_argument("--no-quality-filtering", action="store_true",
                        help="Compute and report quality metrics but don't drop any "
                             "channels (default: failing channels are dropped).")
    parser.add_argument("--exclude-failing-short-channels", action="store_true",
                        help="Also drop short channels that fail QC (kept by default, "
                             "since they're used as SCR regressors rather than signal).")
    parser.add_argument("--post-walking-trim", type=float, default=3.0,
                        help="Seconds to trim after the walking-start event.")
    parser.add_argument("--initial-crop", type=float, default=1.0,
                        help="Seconds to drop from the start of every recording "
                             "(device/initialization artifacts).")

    parser.add_argument("--task-filter", nargs="+", metavar="TASK",
                        help="Restrict to specific task types (e.g. DT ST fTurn).")
    parser.add_argument("--no-consistent-ylimits", action="store_true",
                        help="Skip the second pass that re-plots with shared y-limits.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List discovered task types and exit.")

    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress console output.")
    return parser


def configure_logging(level: str, log_file: str, quiet: bool) -> None:
    """Set up logging to a file and (unless quiet) the console."""
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    if not quiet:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code (0 ok, 1 error, 2 nothing done)."""
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.input_dir):
        print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if args.post_walking_trim < 0:
        print("Error: --post-walking-trim cannot be negative.", file=sys.stderr)
        return 1
    if args.initial_crop < 0:
        print("Error: --initial-crop cannot be negative.", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    configure_logging(args.log_level, os.path.join(args.output_dir, "fnirs_processing.log"), args.quiet)
    logger = logging.getLogger("fnirs_PFC_2025.main")

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
    )

    if args.list_tasks:
        return _list_tasks(manager, args)

    logger.info("Starting study run: input=%s output=%s", args.input_dir, args.output_dir)
    try:
        study = manager.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            task_filter=args.task_filter,
            consistent_ylimits=not args.no_consistent_ylimits,
            show_progress=not args.quiet,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.critical("Pipeline failed: %s", exc, exc_info=True)
        print(f"Error: pipeline failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(study, args)
    return 0 if study.n_processed else 2


# ----- helpers ------------------------------------------------------------ #
def _list_tasks(manager: PipelineManager, args: argparse.Namespace) -> int:
    """Print discovered task types and exit."""
    grouped = manager._batch.find_input_files(args.input_dir, args.task_filter)
    if not grouped:
        print("No matching files found.")
        return 2
    print("Discovered task types:")
    for task, files in sorted(grouped.items()):
        print(f"  {task}: {len(files)} file(s)")
    print(f"Total: {sum(len(f) for f in grouped.values())} file(s)")
    return 0


def _print_summary(study, args: argparse.Namespace) -> None:
    """Print a concise end-of-run summary to stdout."""
    if args.quiet:
        return
    metrics = "+".join(m.upper() for m in args.metrics) or "none"
    thresholds = []
    if "sqi" in args.metrics:
        thresholds.append(f"SQI >= {args.sqi_threshold}")
    if "sci" in args.metrics:
        thresholds.append(f"SCI >= {args.sci_threshold}")
    if "psp" in args.metrics:
        thresholds.append(f"PSP >= {args.psp_threshold}")
    filtering = "off (report only)" if args.no_quality_filtering else "on"

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Input            : {args.input_dir}")
    print(f"Output           : {args.output_dir}")
    print(f"Quality metrics  : {metrics}")
    print(f"QC thresholds    : {', '.join(thresholds) if thresholds else 'n/a'}")
    print(f"Quality filtering: {filtering}")
    print(f"Recordings       : {study.n_processed}/{study.total_files} processed")
    if study.stats_raw is not None:
        print("RAW statistics   : all_subjects_statistics_RAW.csv")
    if study.stats_zscore is not None:
        print("ZSCORE statistics: all_subjects_statistics_ZSCORE.csv")
    print(f"Per-task sheets  : {len(study.summary_paths)} written")
    if study.qc_summary_path is not None:
        print("QC roll-up       : qc_summary_all_recordings.csv")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
