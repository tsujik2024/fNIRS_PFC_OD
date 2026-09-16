from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"grand oxy", "grand deoxy", "Time (s)"}

_SUMMARY_COLUMNS = [
    "Subject", "Timepoint", "Condition", "TaskType",
    "Overall grand oxy Mean", "First Half grand oxy Mean", "Second Half grand oxy Mean",
    "Overall grand deoxy Mean", "First Half grand deoxy Mean", "Second Half grand deoxy Mean",
]


class StatsCollector:

    def __init__(self, fs: float = 50.0, enable_quality_filtering: bool = True) -> None:

        self.fs = fs
        self.enable_quality_filtering = enable_quality_filtering

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

    def _processed_csv_path(
        self,
        file_path: str,
        input_base_dir: str,
        output_base_dir: str,
        file_type: str,
    ) -> Optional[Path]:

        basename = os.path.basename(file_path)  
        pattern = f"{basename}_FULLY_PROCESSED_{file_type}*.csv"

        relative = os.path.relpath(os.path.dirname(file_path), start=input_base_dir)
        search_root = Path(output_base_dir) / relative
        if search_root.exists():
            matches = sorted(search_root.rglob(pattern))
            if matches:
                if len(matches) > 1:
                    logger.warning("Multiple %s CSVs matched for %s; using %s",
                                   file_type, basename, matches[0].name)
                return matches[0]

        matches = sorted(Path(output_base_dir).rglob(pattern))
        if matches:
            if len(matches) > 1:
                logger.warning("Multiple %s CSVs matched for %s; using %s",
                               file_type, basename, matches[0].name)
            return matches[0]
        return None

    def _read_processed_csv(self, path: Path) -> Optional[pd.DataFrame]:
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
        columns = [c for c in _SUMMARY_COLUMNS if c in task_df.columns]
        ordered = task_df[columns].copy()
        sort_cols = [c for c in ("Subject", "Timepoint") if c in ordered.columns]
        if sort_cols:
            ordered = ordered.sort_values(sort_cols).reset_index(drop=True)
        return ordered
