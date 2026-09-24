import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from fnirs_PFC_2025.processing.batch_processor import BatchProcessor, BatchResult
from fnirs_PFC_2025.processing.stats_collector import StatsCollector

logger = logging.getLogger(__name__)


@dataclass
class StudyResult:
    batch: BatchResult
    stats_raw: Optional[pd.DataFrame] = None
    stats_zscore: Optional[pd.DataFrame] = None
    summary_paths: List[Path] = field(default_factory=list)
    qc_summary_path: Optional[Path] = None

    @property
    def total_files(self):
        return self.batch.total_files

    @property
    def n_processed(self):
        return self.batch.n_processed


class PipelineManager:
    """Process a study folder, then aggregate stats and write the QC roll-up.

    Accepts the same keyword args as FileProcessor.
    """

    def __init__(self, **processor_kwargs):
        self._batch = BatchProcessor(**processor_kwargs)
        self._stats = StatsCollector()

    def discover_recordings(self, input_dir, task_filter=None):
        return self._batch.find_input_files(input_dir, task_filter)

    def run(self, input_dir, output_dir, task_filter=None, show_progress=True):
        os.makedirs(output_dir, exist_ok=True)
        batch = self._batch.process(input_dir, output_dir, task_filter=task_filter,
                                    show_progress=show_progress)
        study = StudyResult(batch=batch)
        if batch.processed_files:
            study.stats_raw = self._aggregate(study, input_dir, output_dir, "RAW")
            if self._batch.processor.compute_zscore:
                study.stats_zscore = self._aggregate(study, input_dir, output_dir, "ZSCORE")
        else:
            logger.warning("no recordings processed, skipping stats")
        study.qc_summary_path = self._write_qc_summary(batch, output_dir)
        return study

    def _aggregate(self, study, input_dir, output_dir, kind):
        stats = self._stats.run_statistics(study.batch.processed_files, input_dir, output_dir, kind)
        if stats is not None:
            stats.to_csv(Path(output_dir) / f"all_subjects_statistics_{kind}.csv", index=False)
        study.summary_paths += self._stats.create_summary_sheets(stats, output_dir, suffix=f"_{kind}")
        return stats

    @staticmethod
    def _write_qc_summary(batch, output_dir):
        """One row per recording; includes the SCR note so short-channel fallbacks show up study-wide."""
        rows = [{
            "Recording": os.path.splitext(os.path.basename(path))[0],
            "Metrics used": "+".join(rep.metrics_used) or "none",
            "Channels retained": len(rep.retained),
            "Channels total": len(rep.channels),
            "Long retained": rep.n_long_retained,
            "Long total": rep.n_long_total,
            "Rejected": ";".join(f"CH{c.channel}" for c in rep.rejected) or "-",
            "SCR Note": rep.scr_note or "-",
        } for path, rep in batch.qc_reports.items()]
        if not rows:
            return None
        out = Path(output_dir) / "qc_summary_all_recordings.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        return out
