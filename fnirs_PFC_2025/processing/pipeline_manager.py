from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from fnirs_PFC_2025.processing.batch_processor import BatchProcessor, BatchResult
from fnirs_PFC_2025.processing.quality_control import (
    DEFAULT_PSP_THRESHOLD,
    DEFAULT_SCI_THRESHOLD,
    DEFAULT_SQI_THRESHOLD,
)
from fnirs_PFC_2025.processing.stats_collector import StatsCollector

logger = logging.getLogger(__name__)


@dataclass
class StudyResult:
    """Everything a run produces btw"""

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
    """Run a whole study: process (with quality control) -> y-limits -> statistics -> reports."""

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
    ) -> None:
        """
        Args:
            fs: Sampling frequency in Hz
            sqi_threshold, sci_threshold, psp_threshold: per-metric thresholds,
                passed straight through to BatchProcessor -> FileProcessor.
            enabled_metrics: which of "sqi", "sci", "psp" gate channel exclusion.
                Default ("sci", "psp") - SQI is opt-in.
            enable_quality_filtering: if True (default), channels failing an
                enabled metric are dropped; if False, quality is still computed
                and reported but nothing is removed.
            exclude_failing_short_channels: if True, short channels (CH3/CH5)
                that fail are also dropped, instead of being spared by default.
            post_walking_trim_seconds: seconds to trim after walking start event.
            initial_crop_seconds: seconds to drop from the start of every
                recording (device/initialization artifacts). Default 1.0s,
                matching FullCapProcessor's initial crop.
        """
        self.fs = fs
        self._batch = BatchProcessor(
            fs=fs,
            sqi_threshold=sqi_threshold,
            sci_threshold=sci_threshold,
            psp_threshold=psp_threshold,
            enabled_metrics=enabled_metrics,
            enable_quality_filtering=enable_quality_filtering,
            exclude_failing_short_channels=exclude_failing_short_channels,
            post_walking_trim_seconds=post_walking_trim_seconds,
            initial_crop_seconds=initial_crop_seconds,
        )
        self._stats = StatsCollector(fs=fs, enable_quality_filtering=enable_quality_filtering)

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

        logger.info("Pass 1: processing (quality control runs inside FileProcessor).")
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
        """Write one row per recording summarising quality-filtering outcomes.

        Unchanged from before the quality-control consolidation: QualityReport
        still exposes .channels / .retained / .rejected / .n_long_retained /
        .n_long_total, now populated by FileProcessor instead of the old
        external ChannelQualityControl prefilter.
        """
        rows = []
        for file_path, report in batch.qc_reports.items():
            rows.append({
                "Recording": os.path.splitext(os.path.basename(file_path))[0],
                "Metrics used": "+".join(report.metrics_used) or "none",
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
