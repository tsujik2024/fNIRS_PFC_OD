import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from fnirs_PFC_2025.processing.file_processor import FileProcessor
from fnirs_PFC_2025.processing.quality_control import QualityReport
from fnirs_PFC_2025.read.loaders import read_txt_file

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

OUTCOMES = ("processed", "validation_failed", "failed")


@dataclass
class BatchResult:
    processed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    total_files: int = 0
    by_task: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # task -> outcome -> paths
    qc_reports: Dict[str, QualityReport] = field(default_factory=dict)  # input path -> report

    @property
    def n_processed(self):
        return len(self.processed_files)


class BatchProcessor:
    """Finds recordings under a folder and runs each through FileProcessor.

    Keyword args other than file_extension/read_file_func go straight to FileProcessor.
    """

    def __init__(self, *, file_extension=".txt", read_file_func=read_txt_file, **processor_kwargs):
        self._ext = file_extension.lower()
        self._read = read_file_func
        self.processor = FileProcessor(**processor_kwargs)

    def find_input_files(self, input_dir, task_filter=None):
        grouped = {}
        for root, _, names in os.walk(input_dir):
            for name in names:
                if not name.lower().endswith(self._ext):
                    continue
                task = FileProcessor.determine_task_type(name)
                if task_filter and task not in task_filter:
                    continue
                grouped.setdefault(task, []).append(os.path.join(root, name))
        return {task: sorted(paths) for task, paths in grouped.items()}

    def process(self, input_dir, output_dir, task_filter=None, show_progress=True):
        task_files = self.find_input_files(input_dir, task_filter)
        if not task_files:
            raise FileNotFoundError(f"No {self._ext} files found under {input_dir!r}")

        os.makedirs(output_dir, exist_ok=True)
        batch = BatchResult(total_files=sum(len(f) for f in task_files.values()),
                            by_task={t: {o: [] for o in OUTCOMES} for t in task_files})

        for task, files in task_files.items():
            it = tqdm(files, desc=task, unit="file") if (show_progress and tqdm) else files
            for path in it:
                outcome = self._process_one(path, input_dir, output_dir, batch)
                batch.by_task[task][outcome].append(path)
                (batch.processed_files if outcome == "processed" else batch.skipped_files).append(path)

        self._write_report(batch, output_dir)
        return batch

    def _process_one(self, path, input_dir, output_dir, batch):
        # one bad recording shouldn't sink the whole batch
        try:
            res = self.processor.process_file(path, output_dir, input_dir, self._read)
        except Exception:
            logger.exception("failed: %s", os.path.basename(path))
            return "failed"

        if res.get('quality') is not None:
            batch.qc_reports[path] = res['quality']
        if res['success']:
            return "processed"
        logger.warning("%s: %s", os.path.basename(path), res.get('error', 'unknown error'))
        return "validation_failed" if res.get('validation_failed') else "failed"

    def _write_report(self, batch, output_dir):
        p = self.processor
        lines = [
            "fNIRS processing report",
            "=" * 40,
            f"Total recordings : {batch.total_files}",
            f"Processed        : {batch.n_processed}",
            f"Sampling rate    : {p.fs} Hz",
            f"Quality metrics  : {'+'.join(p.enabled_metrics) or 'none'}",
            f"Thresholds       : SQI>={p.sqi_threshold}, SCI>={p.sci_threshold}, PSP>={p.psp_threshold}",
            f"Quality filtering: {'on' if p.enable_quality_filtering else 'off'}",
            f"Short-ch. exclude: {'on' if p.exclude_failing_short_channels else 'off'}",
            f"Initial crop     : {p.initial_crop_seconds} s",
            f"Post-walk trim   : {p.post_walking_trim_seconds} s",
            f"Z-score output   : {'on' if p.compute_zscore else 'off'}",
            f"Diagnostic plots : {'off' if p.skip_diagnostic_plots else 'on'}",
            "",
        ]
        for task in sorted(batch.by_task):
            res = batch.by_task[task]
            total = sum(len(v) for v in res.values())
            lines.append(f"{task}: {len(res['processed'])}/{total} processed "
                         f"({len(res['processed']) / total * 100:.0f}%) | "
                         f"validation-failed {len(res['validation_failed'])} | failed {len(res['failed'])}")
            lines += [f"    validation: {os.path.basename(f)}" for f in res['validation_failed']]
            lines += [f"    failed    : {os.path.basename(f)}" for f in res['failed']]
        Path(output_dir, "processing_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
