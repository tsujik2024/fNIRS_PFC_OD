"""Batch execution: discover recordings, run the pipeline.

:class:`BatchProcessor` is the execution engine of the orchestration layer. For
each recording it hands the file straight to
:class:`~fnirs_PFC_2025.processing.file_processor.FileProcessor`, which now
performs channel-quality scoring (any combination of SQI/SCI/PSP) and
filtering internally - there is no separate prefilter here anymore.

It groups recordings by task type, tracks per-task outcomes, and collects the
per-recording quality reports (surfaced by FileProcessor's ``process_file``
return value). It does not compute statistics or plots - that is
:class:`PipelineManager`'s job.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.quality_control import (
    DEFAULT_PSP_THRESHOLD,
    DEFAULT_SCI_THRESHOLD,
    DEFAULT_SQI_THRESHOLD,
    QualityReport,
)
from fnirs_PFC_2025.read.loaders import read_txt_file

logger = logging.getLogger(__name__)

# Outcome categories tracked per task type.
_OUTCOMES = ("processed", "validation_failed", "failed")

# Substrings that mark a failure as a task-validation problem rather than a crash.
_VALIDATION_MARKERS = ("requires at least", "task requirements", "validation")


@dataclass
class BatchResult:
    """Aggregate outcome of a batch run."""

    processed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    total_files: int = 0
    # task_type -> outcome -> list of input file paths
    by_task: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # input file path -> QualityReport for that recording (from FileProcessor)
    qc_reports: Dict[str, QualityReport] = field(default_factory=dict)

    @property
    def n_processed(self) -> int:
        return len(self.processed_files)


class BatchProcessor:
    """Discover recordings under a directory tree and process each one."""

    def __init__(
        self,
        fs: float = 50.0,
        sqi_threshold: float = DEFAULT_SQI_THRESHOLD,
        sci_threshold: float = DEFAULT_SCI_THRESHOLD,
        psp_threshold: float = DEFAULT_PSP_THRESHOLD,
        enabled_metrics: Tuple[str, ...] = ("sci", "psp"),
        enable_quality_filtering: bool = True,
        exclude_failing_short_channels: bool = False,
        post_walking_trim_seconds: float = 3.0,
        initial_crop_seconds: float = 1.0,
        file_extension: str = ".txt",
        read_file_func: Callable = read_txt_file,
    ) -> None:
        """
        Args:
            fs: Sampling frequency in Hz
            sqi_threshold, sci_threshold, psp_threshold: per-metric thresholds,
                passed straight through to FileProcessor.
            enabled_metrics: which of "sqi", "sci", "psp" gate channel exclusion.
                Default ("sci", "psp") - SQI is opt-in. Passed straight through
                to FileProcessor, which is now the single place quality control
                happens for this pipeline.
            enable_quality_filtering: if True (default), channels failing an
                enabled metric are dropped; if False, quality is still computed
                and reported but nothing is removed.
            exclude_failing_short_channels: if True, short channels (CH3/CH5)
                that fail are also dropped, instead of being spared by default.
            post_walking_trim_seconds: seconds to trim after walking start event.
            initial_crop_seconds: seconds to drop from the start of every
                recording (device/initialization artifacts). Default 1.0s,
                matching FullCapProcessor's initial crop.
            file_extension: extension to search for when discovering recordings.
            read_file_func: loader callable, ``read_txt_file``-compatible.
        """
        self.fs = fs
        self.sqi_threshold = sqi_threshold
        self.sci_threshold = sci_threshold
        self.psp_threshold = psp_threshold
        self.enabled_metrics = tuple(m.lower() for m in enabled_metrics)
        self.enable_quality_filtering = enable_quality_filtering
        self.exclude_failing_short_channels = exclude_failing_short_channels
        self.post_walking_trim_seconds = post_walking_trim_seconds
        self.initial_crop_seconds = initial_crop_seconds
        self._extension = file_extension.lower()
        self._read_file_func = read_file_func
        self._processor = FileProcessor(
            fs=fs,
            sqi_threshold=sqi_threshold,
            sci_threshold=sci_threshold,
            psp_threshold=psp_threshold,
            enabled_metrics=self.enabled_metrics,
            enable_quality_filtering=enable_quality_filtering,
            exclude_failing_short_channels=exclude_failing_short_channels,
            post_walking_trim_seconds=post_walking_trim_seconds,
            initial_crop_seconds=initial_crop_seconds,
        )
        logger.info(
            "BatchProcessor ready (fs=%.1f Hz, metrics=%s, thresholds: SQI>=%.2f "
            "SCI>=%.2f PSP>=%.2f, quality filtering=%s, initial crop=%.1fs, "
            "post-walk trim=%.1fs).",
            fs, self.enabled_metrics, sqi_threshold, sci_threshold, psp_threshold,
            enable_quality_filtering, initial_crop_seconds, post_walking_trim_seconds,
        )

    # ----- discovery ------------------------------------------------------ #
    def find_input_files(
        self, input_dir: str, task_filter: Optional[Sequence[str]] = None
    ) -> Dict[str, List[str]]:
        """Return input files under ``input_dir`` grouped by task type."""
        wanted = set(task_filter) if task_filter else None
        grouped: Dict[str, List[str]] = {}
        for root, _dirs, files in os.walk(input_dir):
            for name in files:
                if not name.lower().endswith(self._extension):
                    continue
                task = self._determine_task_type(name)
                if wanted is not None and task not in wanted:
                    continue
                grouped.setdefault(task, []).append(os.path.join(root, name))

        for task in grouped:
            grouped[task].sort()
        logger.info("Found %d file(s) across %d task type(s): %s",
                    sum(len(v) for v in grouped.values()), len(grouped),
                    ", ".join(f"{t}:{len(f)}" for t, f in sorted(grouped.items())))
        return grouped

    # ----- processing ----------------------------------------------------- #
    def process(
        self,
        input_dir: str,
        output_dir: str,
        task_filter: Optional[Sequence[str]] = None,
        subject_y_limits: Optional[Dict[str, Dict[str, float]]] = None,
        show_progress: bool = True,
    ) -> BatchResult:
        """Process every discovered recording, continuing past failures."""
        task_files = self.find_input_files(input_dir, task_filter)
        if not task_files:
            raise FileNotFoundError(f"No {self._extension} files found under {input_dir!r}.")

        os.makedirs(output_dir, exist_ok=True)
        batch = BatchResult(
            total_files=sum(len(f) for f in task_files.values()),
            by_task={t: {o: [] for o in _OUTCOMES} for t in task_files},
        )

        for task, files in task_files.items():
            logger.info("Processing task %s (%d file(s)).", task, len(files))
            for path in self._with_progress(files, task, show_progress):
                outcome = self._process_one(path, input_dir, output_dir, subject_y_limits, batch)
                batch.by_task[task][outcome].append(path)
                (batch.processed_files if outcome == "processed" else batch.skipped_files).append(path)

        logger.info("Batch complete: %d/%d recordings processed.",
                    batch.n_processed, batch.total_files)
        self._write_processing_report(batch, output_dir)
        return batch

    def _process_one(
        self,
        file_path: str,
        input_dir: str,
        output_dir: str,
        subject_y_limits: Optional[Dict[str, Dict[str, float]]],
        batch: BatchResult,
    ) -> str:
        """Process a single recording (FileProcessor does its own quality control
        internally); return its outcome and record the quality report."""
        try:
            result = self._processor.process_file(
                file_path=file_path,
                output_base_dir=output_dir,
                input_base_dir=input_dir,
                subject_y_limits=subject_y_limits,
                read_file_func=self._read_file_func,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch alive on any one file
            outcome = self._classify_error(str(exc))
            logger.error("%s on %s: %s", outcome, os.path.basename(file_path), exc)
            return outcome

        if result is None:
            logger.warning("No result for %s.", os.path.basename(file_path))
            return "failed"

        quality = result.get("quality")
        if quality is not None:
            batch.qc_reports[file_path] = quality

        if result.get("success"):
            return "processed"
        if result.get("validation_failed"):
            logger.warning("Validation failed: %s (%s)",
                           os.path.basename(file_path), result.get("error", "unknown"))
            return "validation_failed"
        logger.warning("Processing failed: %s (%s)",
                       os.path.basename(file_path), result.get("error", "unknown"))
        return "failed"

    # ----- reporting ------------------------------------------------------ #
    def _write_processing_report(self, batch: BatchResult, output_dir: str) -> Path:
        """Write a plain-text per-task processing report."""
        path = Path(output_dir) / "processing_report.txt"
        lines = [
            "fNIRS processing report",
            "=" * 40,
            f"Total recordings : {batch.total_files}",
            f"Processed        : {batch.n_processed}",
            f"Sampling rate    : {self.fs} Hz",
            f"Quality metrics  : {'+'.join(self.enabled_metrics) or 'none'}",
            f"Thresholds       : SQI>={self.sqi_threshold}, SCI>={self.sci_threshold}, PSP>={self.psp_threshold}",
            f"Quality filtering: {'on' if self.enable_quality_filtering else 'off'}",
            f"Post-walk trim   : {self.post_walking_trim_seconds} s",
            "",
        ]
        for task in sorted(batch.by_task):
            counts = {o: len(batch.by_task[task][o]) for o in _OUTCOMES}
            total = sum(counts.values())
            if total == 0:
                continue
            rate = counts["processed"] / total * 100
            lines.append(f"{task}: {counts['processed']}/{total} processed "
                         f"({rate:.0f}%) | validation-failed {counts['validation_failed']} "
                         f"| failed {counts['failed']}")
            for failed_path in batch.by_task[task]["validation_failed"]:
                lines.append(f"    validation: {os.path.basename(failed_path)}")
            for failed_path in batch.by_task[task]["failed"]:
                lines.append(f"    failed    : {os.path.basename(failed_path)}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote processing report: %s", path.name)
        return path

    # ----- internals ------------------------------------------------------ #
    @staticmethod
    def _classify_error(message: str) -> str:
        """Map an exception message to an outcome category."""
        lowered = message.lower()
        if any(marker in lowered for marker in _VALIDATION_MARKERS):
            return "validation_failed"
        return "failed"

    @staticmethod
    def _with_progress(files: Sequence[str], task: str, show_progress: bool):
        """Wrap an iterable in a tqdm bar when available and requested."""
        if not show_progress:
            return files
        try:
            from tqdm import tqdm  # lazy: optional dependency
        except ImportError:
            return files
        return tqdm(files, desc=task, unit="file")

    @staticmethod
    def _determine_task_type(filename: str) -> str:
        """Classify a recording by filename using boundary-aware keyword matching."""
        basename = os.path.basename(filename).upper()
        if "FTURN" in basename or "F_TURN" in basename:
            return "fTurn"
        if "LSHAPE" in basename or "L_SHAPE" in basename:
            return "LShape"
        if "OBSTACLE" in basename:
            return "Obstacle"
        if "NAVIGATION" in basename or re.search(r"\bNAV\b", basename):
            return "Navigation"
        if re.search(r"(^|[^A-Z])DT([^A-Z]|$)", basename):
            return "DT"
        if re.search(r"(^|[^A-Z])ST([^A-Z]|$)", basename):
            return "ST"
        if "WALK" in basename:
            return "LongWalk"
        logger.warning("Could not determine task type from filename: %s", basename)
        return "Unknown"
