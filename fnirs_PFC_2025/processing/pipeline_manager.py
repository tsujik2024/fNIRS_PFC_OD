from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
    """Everything a single PipelineManager.run() call produces."""

    batch: BatchResult
    stats_raw: Optional[pd.DataFrame] = None
    stats_zscore: Optional[pd.DataFrame] = None
    summary_paths: List[Path] = field(default_factory=list)
    qc_summary_path: Optional[Path] = None

    @property
    def total_files(self) -> int:
        return self.batch.total_files

    @property
    def n_processed(self) -> int:
        return self.batch.n_processed


class PipelineManager:
    """Runs an entire study end to end: process (with quality control), then statistics, then reports."""

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
        skip_diagnostic_plots: bool = False,
        compute_zscore: bool = True,
    ) -> None:
        """
        fs is the sampling rate in Hz. sqi_threshold, sci_threshold and
        psp_threshold are per-metric thresholds that flow straight through
        to BatchProcessor and then FileProcessor. enabled_metrics picks
        which of "sqi", "sci", "psp" gate channel exclusion - default is
        ("sci", "psp"), with SQI opt-in.

        enable_quality_filtering (default True) drops channels that fail
        an enabled metric; set it False and quality still gets computed and
        reported, just nothing gets removed. exclude_failing_short_channels,
        if True, drops CH3/CH5 too instead of sparing them.

        post_walking_trim_seconds is the trim after the walking-start event.
        initial_crop_seconds trims the start of every recording for device
        warm-up artifacts (1.0s default, matching FullCapProcessor).
        skip_diagnostic_plots turns off the per-stage plots and summary
        panel across the whole study. compute_zscore, if False, skips
        Z-transformation for every recording - only RAW comes out, and the
        ZSCORE statistics pass gets skipped too since there'd be nothing
        to aggregate.
        """
        self.fs = fs
        self._compute_zscore = compute_zscore
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
            skip_diagnostic_plots=skip_diagnostic_plots,
            compute_zscore=compute_zscore,
        )
        self._stats = StatsCollector(fs=fs, enable_quality_filtering=enable_quality_filtering)

    def run(
        self,
        input_dir: str,
        output_dir: str,
        task_filter: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> StudyResult:
        """Run the whole study - processing, then stats, then reports - and hand back a StudyResult."""
        os.makedirs(output_dir, exist_ok=True)

        batch = self._batch.process(
            input_dir, output_dir, task_filter=task_filter, show_progress=show_progress
        )
        if not batch.processed_files:
            logger.warning("No recordings processed successfully; skipping aggregation.")
            study = StudyResult(batch=batch)
            study.qc_summary_path = self._write_qc_summary(batch, output_dir)
            return study

        study = StudyResult(batch=batch)
        study.stats_raw = self._aggregate(batch.processed_files, input_dir, output_dir, "RAW")
        if self._compute_zscore:
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
        """Run the stats collector on one file type (RAW or ZSCORE) and save the combined CSV."""
        stats = self._stats.run_statistics(processed_files, input_dir, output_dir, file_type)
        if stats is None or stats.empty:
            return stats
        path = Path(output_dir) / f"all_subjects_statistics_{file_type}.csv"
        stats.to_csv(path, index=False)
        logger.info("Wrote combined %s statistics: %s", file_type, path.name)
        return stats

    def _write_summaries(self, study: StudyResult, output_dir: str) -> List[Path]:
        """Write the per-task summary sheets - RAW always, ZSCORE too if it got computed."""
        written: List[Path] = []
        pairs = [(study.stats_raw, "_RAW")]
        if self._compute_zscore:
            pairs.append((study.stats_zscore, "_ZSCORE"))
        for stats, suffix in pairs:
            written.extend(self._stats.create_summary_sheets(stats, output_dir, suffix=suffix))
        return written

    # ----- QC roll-up ----------------------------------------------------- #
    def _write_qc_summary(self, batch: BatchResult, output_dir: str) -> Optional[Path]:
        """One row per recording, summarizing how quality filtering went for it.

        Includes the SCR note (blank if SCR ran normally; otherwise which
        fallback kicked in, or that SCR got skipped entirely), so a
        hemisphere borrowing the other side's short channel - or losing SCR
        altogether - shows up at the study level instead of only sitting in
        one recording's logs.
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
                "SCR Note": report.scr_note or "-",
            })
        if not rows:
            return None
        path = Path(output_dir) / "qc_summary_all_recordings.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Wrote study QC roll-up: %s", path.name)
        return path
