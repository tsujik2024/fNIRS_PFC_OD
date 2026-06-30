"""End-to-end study orchestration.

:class:`PipelineManager` is the top-level workflow. It delegates execution to
:class:`BatchProcessor` (which runs SCI/PSP channel quality control before the
pipeline) and aggregation to :class:`StatsCollector`. Because
:class:`FileProcessor` renders its plots while processing, consistent
per-subject y-limits are achieved with two passes: an initial pass to derive the
limits from the processed data, then a second pass that re-renders with those
limits. Pass 2 is skipped when ``consistent_ylimits`` is ``False``.

It also writes a study-wide QC roll-up (``qc_summary_all_recordings.csv``) so the
SCI/PSP rejection outcome for every recording is captured in one place.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from fnirs_PFC_2025.processing.batch_processor import BatchProcessor, BatchResult
from fnirs_PFC_2025.processing.quality_control import (
    DEFAULT_PSP_THRESHOLD,
    DEFAULT_SCI_THRESHOLD,
)
from fnirs_PFC_2025.processing.stats_collector import StatsCollector

logger = logging.getLogger(__name__)


@dataclass
class StudyResult:
    """Everything a study run produces, for programmatic use after the fact."""

    batch: BatchResult
    stats_raw: Optional[pd.DataFrame] = None
    stats_zscore: Optional[pd.DataFrame] = None
    y_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary_paths: List[Path] = field(default_factory=list)
    qc_summary_path: Optional[Path] = None

    @property
    def total_files(self) -> int:
        return self.batch.total_files

    @property
    def n_processed(self) -> int:
        return self.batch.n_processed


class PipelineManager:
    """Run a whole study: QC + process -> y-limits -> statistics -> reports."""

    def __init__(
        self,
        fs: float = 50.0,
        sci_threshold: float = DEFAULT_SCI_THRESHOLD,
        psp_threshold: float = DEFAULT_PSP_THRESHOLD,
        post_walking_trim_seconds: float = 3.0,
        sqi_threshold: float = 2.0,
        enable_sqi_filtering: bool = False,
        short_channels: Sequence[int] = (3, 5),  # CH3, CH5 (0-based) = short channels 4 & 6
        exclude_failing_short_channels: bool = False,
    ) -> None:
        self.fs = fs
        self._batch = BatchProcessor(
            fs=fs,
            sci_threshold=sci_threshold,
            psp_threshold=psp_threshold,
            post_walking_trim_seconds=post_walking_trim_seconds,
            sqi_threshold=sqi_threshold,
            enable_sqi_filtering=enable_sqi_filtering,
            short_channels=short_channels,
            exclude_failing_short_channels=exclude_failing_short_channels,
        )
        self._stats = StatsCollector(fs=fs)

    def run(
        self,
        input_dir: str,
        output_dir: str,
        task_filter: Optional[Sequence[str]] = None,
        consistent_ylimits: bool = True,
        show_progress: bool = True,
    ) -> StudyResult:
        """Execute the full study workflow and return a :class:`StudyResult`."""
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Pass 1: channel QC + initial processing.")
        batch = self._batch.process(
            input_dir, output_dir, task_filter=task_filter, show_progress=show_progress
        )
        if not batch.processed_files:
            logger.warning("No recordings processed successfully; skipping aggregation.")
            study = StudyResult(batch=batch)
            study.qc_summary_path = self._write_qc_summary(batch, output_dir)
            return study

        y_limits = self._stats.calculate_subject_y_limits(
            batch.processed_files, output_dir, input_dir
        )

        if consistent_ylimits and y_limits:
            logger.info("Pass 2: re-processing with consistent per-subject y-limits.")
            batch = self._batch.process(
                input_dir, output_dir, task_filter=task_filter,
                subject_y_limits=y_limits, show_progress=show_progress,
            )

        study = StudyResult(batch=batch, y_limits=y_limits)
        study.stats_raw = self._aggregate(batch.processed_files, input_dir, output_dir, "RAW")
        study.stats_zscore = self._aggregate(batch.processed_files, input_dir, output_dir, "ZSCORE")
        study.summary_paths = self._write_summaries(study, output_dir)
        study.qc_summary_path = self._write_qc_summary(batch, output_dir)
        return study

    # ----- statistics ----------------------------------------------------- #
    def _aggregate(
        self,
        processed_files: Sequence[str],
        input_dir: str,
        output_dir: str,
        file_type: str,
    ) -> Optional[pd.DataFrame]:
        """Run the stats collector for one file type and save the combined CSV."""
        stats = self._stats.run_statistics(processed_files, input_dir, output_dir, file_type)
        if stats is None or stats.empty:
            return stats
        path = Path(output_dir) / f"all_subjects_statistics_{file_type}.csv"
        stats.to_csv(path, index=False)
        logger.info("Wrote combined %s statistics: %s", file_type, path.name)
        return stats

    def _write_summaries(self, study: StudyResult, output_dir: str) -> List[Path]:
        """Write per-task summary sheets for both RAW and ZSCORE statistics."""
        written: List[Path] = []
        for stats, suffix in ((study.stats_raw, "_RAW"), (study.stats_zscore, "_ZSCORE")):
            written.extend(self._stats.create_summary_sheets(stats, output_dir, suffix=suffix))
        return written

    # ----- QC roll-up ----------------------------------------------------- #
    def _write_qc_summary(self, batch: BatchResult, output_dir: str) -> Optional[Path]:
        """Write one row per recording summarising SCI/PSP channel-quality outcomes."""
        rows = []
        for file_path, report in batch.qc_reports.items():
            rows.append({
                "Recording": os.path.splitext(os.path.basename(file_path))[0],
                "Channels retained": len(report.retained),
                "Channels total": len(report.channels),
                "Long retained": report.n_long_retained,
                "Long total": report.n_long_total,
                "Rejected": ";".join(f"CH{c.channel}" for c in report.rejected) or "-",
            })
        if not rows:
            return None
        path = Path(output_dir) / "qc_summary_all_recordings.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Wrote study QC roll-up: %s", path.name)
        return path
