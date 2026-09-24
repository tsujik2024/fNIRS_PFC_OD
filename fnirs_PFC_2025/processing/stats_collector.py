import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED = {"grand oxy", "grand deoxy", "Time (s)"}
_COLUMNS = [
    "Subject", "Timepoint", "Condition", "TaskType",
    "Overall grand oxy Mean", "First Half grand oxy Mean", "Second Half grand oxy Mean",
    "Overall grand deoxy Mean", "First Half grand deoxy Mean", "Second Half grand deoxy Mean",
]


class StatsCollector:
    def run_statistics(self, processed_files, input_dir, output_dir, file_type="RAW"):
        """One summary row per processed recording, or None if there's nothing to summarise."""
        if file_type not in ("RAW", "ZSCORE"):
            raise ValueError(f"file_type must be 'RAW' or 'ZSCORE', got {file_type!r}")

        rows, seen = [], set()
        for path in sorted(set(processed_files)):
            csv = self._find_csv(path, input_dir, output_dir, file_type)
            frame = self._read(csv) if csv else None
            if frame is None:
                logger.warning("no usable %s CSV for %s", file_type, os.path.basename(path))
                continue

            row = self._summarise(frame, *self.extract_metadata(path))
            key = (row["Subject"], row["Timepoint"], row["Condition"], row["TaskType"],
                   round(row["Overall grand oxy Mean"], 10), round(row["Overall grand deoxy Mean"], 10))
            if key not in seen:
                seen.add(key)
                rows.append(row)

        return pd.DataFrame(rows, columns=_COLUMNS) if rows else None

    def create_summary_sheets(self, stats, output_dir, suffix=""):
        """One CSV per task, task name taken from Condition minus the subject prefix."""
        if stats is None or stats.empty:
            return []
        df = stats.copy()
        df["_task"] = [self._task_from_condition(str(c), str(s))
                       for c, s in zip(df["Condition"], df["Subject"])]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        written = []
        for task in sorted(df["_task"].unique()):
            sub = df[df["_task"] == task].drop(columns="_task")[_COLUMNS]
            sub = sub.sort_values(["Subject", "Timepoint"]).reset_index(drop=True)
            path = Path(output_dir) / f"summary_{task.replace('_OD', '').replace(' ', '-')}{suffix}.csv"
            sub.to_csv(path, index=False)
            written.append(path)
        return written

    def extract_metadata(self, file_path):
        """(subject, timepoint) from the folder layout: .../<subject>/<Pre|Post>/file, or <subject>_<T1|V2|Pre|Post>."""
        parts = file_path.split(os.sep)
        for i, part in enumerate(parts):
            tp = {"pre": "Pre", "baseline": "Pre", "post": "Post", "post-intervention": "Post"}.get(part.strip().lower())
            if tp:
                return (parts[i - 1] if i > 0 else "Unknown"), tp

        for part in reversed(os.path.dirname(file_path).split(os.sep)):
            m = re.match(r"^(.+?)_(V\d+|T\d+)$", part, re.IGNORECASE)
            if m:
                return m.group(1), m.group(2)
            m = re.match(r"^(.+?)_(Pre|Post|Baseline|Post-Intervention)$", part, re.IGNORECASE)
            if m:
                tp = "Pre" if m.group(2).lower() in ("pre", "baseline") else "Post"
                return m.group(1), tp

        parent = os.path.basename(os.path.dirname(file_path))
        return (parent, "Unknown") if parent else ("Unknown", "Unknown")

    @staticmethod
    def _task_from_condition(condition, subject):
        if not condition or condition == "Unknown":
            return condition
        prefix = subject.rstrip("_") + "_"
        if subject != "Unknown" and condition.lower().startswith(prefix.lower()):
            return condition[len(prefix):] or condition
        tokens = condition.split("_")
        for i, tok in enumerate(tokens):
            if tok.isdigit():  # the numeric token ends the subject id
                return "_".join(tokens[i + 1:]) or condition
        return condition

    @staticmethod
    def _find_csv(file_path, input_dir, output_dir, file_type):
        # only look in the mirrored folder; a global search can pick up another subject's file
        rel = os.path.relpath(os.path.dirname(file_path), start=input_dir)
        root = Path(output_dir) / rel
        if not root.exists():
            return None
        found = sorted(root.rglob(f"{os.path.basename(file_path)}_FULLY_PROCESSED_{file_type}*.csv"))
        if len(found) > 1:
            logger.warning("several %s CSVs for %s, using %s", file_type, file_path, found[0].name)
        return found[0] if found else None

    @staticmethod
    def _read(path):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None
        return frame if not frame.empty and _REQUIRED <= set(frame.columns) else None

    @staticmethod
    def _summarise(frame, subject, timepoint):
        half = len(frame) // 2
        row = {"Subject": subject, "Timepoint": timepoint,
               "Condition": frame["Condition"].iloc[0] if "Condition" in frame else "Unknown",
               "TaskType": frame["TaskType"].iloc[0] if "TaskType" in frame else "Unknown"}
        for col in ("grand oxy", "grand deoxy"):
            s = frame[col]
            row[f"Overall {col} Mean"] = float(s.mean())
            row[f"First Half {col} Mean"] = float(s.iloc[:half].mean())
            row[f"Second Half {col} Mean"] = float(s.iloc[half:].mean())
        return row
