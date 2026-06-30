"""Cross-subject statistics aggregation for processed recordings.

:class:`StatsCollector` reads the grand-average CSVs written by
:class:`~fnirs_PFC_2025.processing.file_processor.FileProcessor`
(``*_FULLY_PROCESSED_RAW.csv`` / ``*_FULLY_PROCESSED_ZSCORE.csv``), computes
per-recording summary statistics, and pools them into per-task tables that span
all subjects and timepoints.

Subject and timepoint are inferred from the *input path*, not from the file,
because the same task filename recurs across subjects and sessions. Three folder
conventions are supported (see :meth:`StatsCollector.extract_metadata`):

1. timepoint as its own subfolder    - ``.../OHSU_Turn_001/Pre/file.txt``
2. timepoint suffix on a folder name  - ``.../Long_058_V1/file.txt``
3. fallback: deepest folder as subject, timepoint unknown.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns every processed grand-average CSV is expected to contain.
_REQUIRED_COLUMNS = {"grand oxy", "grand deoxy", "Time (s)"}

# Summary column order shared by the per-recording rows and the per-task sheets.
_SUMMARY_COLUMNS = [
    "Subject", "Timepoint", "Condition", "TaskType",
    "Overall grand oxy Mean", "First Half grand oxy Mean", "Second Half grand oxy Mean",
    "Overall grand deoxy Mean", "First Half grand deoxy Mean", "Second Half grand deoxy Mean",
]


class StatsCollector:
    """Aggregate per-recording statistics into cross-subject, per-task tables."""

    def __init__(self, fs: float = 50.0) -> None:
        self.fs = fs

    # ----- public API ----------------------------------------------------- #
    def run_statistics(
        self,
        processed_files: Sequence[str],
        input_base_dir: str,
        output_base_dir: str,
        file_type: str = "RAW",
    ) -> Optional[pd.DataFrame]:
        """Compute one summary row per recording and return them as a DataFrame.

        Parameters
        ----------
        processed_files
            Input file paths that were processed successfully.
        input_base_dir, output_base_dir
            Roots used to locate each recording's processed CSV (the output tree
            mirrors the input tree).
        file_type
            ``"RAW"`` or ``"ZSCORE"`` - selects which grand-average CSV to read.

        Returns
        -------
        pandas.DataFrame or None
            One row per recording, or ``None`` if nothing could be summarised.
        """
        if file_type not in ("RAW", "ZSCORE"):
            raise ValueError(f"file_type must be 'RAW' or 'ZSCORE', got {file_type!r}")

        rows: List[Dict[str, object]] = []
        seen: set = set()
        found = missing = 0

        for file_path in sorted(set(processed_files)):
            csv_path = self._processed_csv_path(file_path, input_base_dir, output_base_dir, file_type)
            if csv_path is None:
                missing += 1
                logger.warning("No %s CSV for %s", file_type, os.path.basename(file_path))
                continue

            frame = self._read_processed_csv(csv_path)
            if frame is None:
                missing += 1
                continue

            subject, timepoint = self.extract_metadata(file_path)
            row = self._summarise(frame, subject, timepoint)
            if row is None:
                continue

            signature = (
                row["Subject"], row["Timepoint"], row["Condition"], row["TaskType"],
                round(float(row["Overall grand oxy Mean"]), 10),
                round(float(row["Overall grand deoxy Mean"]), 10),
            )
            if signature in seen:
                logger.debug("Skipping duplicate summary: %s", signature)
                continue
            seen.add(signature)
            rows.append(row)
            found += 1

        logger.info("%s statistics: summarised %d recording(s), %d missing.",
                    file_type, found, missing)
        if not rows:
            logger.warning("No %s statistics were produced.", file_type)
            return None
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def create_summary_sheets(
        self,
        stats: Optional[pd.DataFrame],
        output_dir: str,
        suffix: str = "",
    ) -> List[Path]:
        """Write one cross-subject CSV per task, returning the written paths.

        The task name is derived from each row's ``Condition`` by stripping the
        per-subject prefix (``Turn_100_Walking_DT`` -> ``Walking_DT``), so rows
        for the same task across subjects land in one file.
        """
        if stats is None or stats.empty:
            logger.warning("No statistics to summarise (suffix=%r).", suffix)
            return []

        df = stats.copy()
        df["_task"] = [
            self._task_from_condition(str(c), str(s))
            for c, s in zip(df["Condition"], df.get("Subject", "Unknown"))
        ]

        written: List[Path] = []
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        for task in sorted(df["_task"].unique()):
            task_df = self._order_for_output(df[df["_task"] == task].drop(columns="_task"))
            if task_df.empty:
                continue
            safe = task.replace("_OD", "").replace(" ", "-")
            path = out_root / f"summary_{safe}{suffix}.csv"
            task_df.to_csv(path, index=False)
            written.append(path)
            logger.info("Wrote task summary (%d rows, %d subjects): %s",
                        len(task_df), task_df["Subject"].nunique(), path.name)
        return written

    def calculate_subject_y_limits(
        self,
        processed_files: Sequence[str],
        output_base_dir: str,
        input_base_dir: str,
    ) -> Dict[str, Dict[str, float]]:
        """Pool each subject's RAW grand-average values to derive shared y-limits.

        Uses robust 1st/99th percentiles with 10% padding so a single subject's
        plots share a consistent y-axis across tasks.
        """
        pooled: Dict[str, List[float]] = {}
        for file_path in processed_files:
            csv_path = self._processed_csv_path(file_path, input_base_dir, output_base_dir, "RAW")
            if csv_path is None:
                continue
            frame = self._read_processed_csv(csv_path)
            if frame is None:
                continue
            subject, _ = self.extract_metadata(file_path)
            values = pd.concat([frame["grand oxy"], frame["grand deoxy"]]).dropna()
            pooled.setdefault(subject, []).extend(values.tolist())

        limits: Dict[str, Dict[str, float]] = {}
        for subject, values in pooled.items():
            if not values:
                continue
            low, high = np.percentile(values, 1), np.percentile(values, 99)
            pad = (high - low) * 0.1 or 0.1
            limits[subject] = {"raw_min": float(low - pad), "raw_max": float(high + pad)}
        logger.info("Derived y-limits for %d subject(s).", len(limits))
        return limits

    # ----- path / metadata helpers --------------------------------------- #
    def extract_metadata(self, file_path: str) -> Tuple[str, str]:
        """Infer ``(subject, timepoint)`` from a recording's path."""
        parts = file_path.split(os.sep)

        for i, part in enumerate(parts):
            normalised = self._normalise_timepoint(part)
            if normalised is not None:
                subject = parts[i - 1] if i > 0 else "Unknown"
                return subject, normalised

        for part in reversed(os.path.dirname(file_path).split(os.sep)):
            match = self._match_embedded_timepoint(part)
            if match is not None:
                return match

        parent = os.path.basename(os.path.dirname(file_path))
        if parent:
            logger.debug("No timepoint detected; using folder as subject: %s", parent)
            return parent, "Unknown"
        return "Unknown", "Unknown"

    @staticmethod
    def _normalise_timepoint(token: str) -> Optional[str]:
        """Map a standalone folder token to ``'Pre'``/``'Post'`` or ``None``."""
        key = token.strip().lower()
        if key in ("pre", "baseline"):
            return "Pre"
        if key in ("post", "post-intervention"):
            return "Post"
        return None

    @staticmethod
    def _match_embedded_timepoint(folder: str) -> Optional[Tuple[str, str]]:
        """Split ``Subject_<timepoint>`` folder names (``_V1``, ``_Pre``, ``_T2``)."""
        patterns = (
            (r"^(.+?)_(V\d+)$", None),
            (r"^(.+?)_(T\d+)$", None),
            (r"^(.+?)_(Pre|Post|Baseline|Post-Intervention)$",
             {"pre": "Pre", "baseline": "Pre", "post": "Post", "post-intervention": "Post"}),
        )
        for pattern, mapping in patterns:
            match = re.match(pattern, folder, re.IGNORECASE)
            if not match:
                continue
            subject, raw = match.group(1), match.group(2)
            timepoint = mapping.get(raw.lower(), raw) if mapping else raw
            return subject, timepoint
        return None

    @staticmethod
    def _task_from_condition(condition: str, subject: str) -> str:
        """Strip a subject prefix from a Condition to get the shared task name."""
        if not condition or condition == "Unknown":
            return condition
        if subject and subject != "Unknown":
            prefix = subject.rstrip("_") + "_"
            if condition.lower().startswith(prefix.lower()):
                return condition[len(prefix):] or condition
        tokens = condition.split("_")
        for i, token in enumerate(tokens):
            if re.fullmatch(r"\d+", token):  # numeric token ends the subject id
                tail = "_".join(tokens[i + 1:])
                if tail:
                    return tail
                break
        return condition

    # ----- internals ------------------------------------------------------ #
    def _processed_csv_path(
        self,
        file_path: str,
        input_base_dir: str,
        output_base_dir: str,
        file_type: str,
    ) -> Optional[Path]:
        """Locate a recording's processed CSV by exact stem in the mirrored tree."""
        stem = os.path.splitext(os.path.basename(file_path))[0]
        filename = f"{stem}_FULLY_PROCESSED_{file_type}.csv"

        relative = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)
        candidate = Path(output_base_dir) / relative / filename
        if candidate.exists():
            return candidate

        matches = list(Path(output_base_dir).rglob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning("Ambiguous matches for %s; skipping.", filename)
        return None

    def _read_processed_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """Read and validate a processed CSV, or return ``None`` if unusable."""
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.error("Could not read %s: %s", path.name, exc)
            return None
        missing = _REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            logger.warning("%s missing columns %s; skipping.", path.name, sorted(missing))
            return None
        if frame.empty:
            logger.warning("%s is empty; skipping.", path.name)
            return None
        return frame

    def _summarise(
        self, frame: pd.DataFrame, subject: str, timepoint: str
    ) -> Optional[Dict[str, object]]:
        """Compute overall and first/second-half means for one recording."""
        n = len(frame)
        half = n // 2
        condition = frame["Condition"].iloc[0] if "Condition" in frame.columns else "Unknown"
        task_type = frame["TaskType"].iloc[0] if "TaskType" in frame.columns else "Unknown"

        def means(column: str) -> Tuple[float, float, float]:
            series = frame[column]
            return (
                float(series.mean(skipna=True)),
                float(series.iloc[:half].mean(skipna=True)),
                float(series.iloc[half:].mean(skipna=True)),
            )

        oxy_all, oxy_first, oxy_second = means("grand oxy")
        deoxy_all, deoxy_first, deoxy_second = means("grand deoxy")
        return {
            "Subject": subject,
            "Timepoint": timepoint,
            "Condition": condition,
            "TaskType": task_type,
            "Overall grand oxy Mean": oxy_all,
            "First Half grand oxy Mean": oxy_first,
            "Second Half grand oxy Mean": oxy_second,
            "Overall grand deoxy Mean": deoxy_all,
            "First Half grand deoxy Mean": deoxy_first,
            "Second Half grand deoxy Mean": deoxy_second,
        }

    @staticmethod
    def _order_for_output(task_df: pd.DataFrame) -> pd.DataFrame:
        """Select known columns in a fixed order and sort by subject/timepoint."""
        columns = [c for c in _SUMMARY_COLUMNS if c in task_df.columns]
        ordered = task_df[columns].copy()
        sort_cols = [c for c in ("Subject", "Timepoint") if c in ordered.columns]
        if sort_cols:
            ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
        return ordered
