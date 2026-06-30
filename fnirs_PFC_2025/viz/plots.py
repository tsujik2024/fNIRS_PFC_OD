"""Figure builders for fNIRS hemoglobin signals.

Each function builds and returns a Matplotlib :class:`~matplotlib.figure.Figure`
without showing or saving it, keeping plotting free of I/O and global state.
Hemoglobin species are matched by column-name keywords so the same helpers work
for raw concentration frames and grand-average frames:

* oxy   : ``HbO``, ``O2Hb``, ``grand oxy``
* deoxy : ``HbR``, ``HHb``, ``grand deoxy``

Configure a headless backend (``matplotlib.use("Agg")``) before importing pyplot
when running without a display.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

YLim = Tuple[float, float]

_OXY_KEYS = ("HbO", "O2Hb", "grand oxy")
_DEOXY_KEYS = ("HHb", "HbR", "grand deoxy")
_SIGNAL_KEYS = _OXY_KEYS + _DEOXY_KEYS
_PAD_FRACTION = 0.05


def plot_overall_signals(
    frame: pd.DataFrame,
    fs: float,
    title: str = "Grand-average signals",
    subject: str = "",
    condition: str = "",
    y_lim: Optional[YLim] = None,
    events: Optional[pd.DataFrame] = None,
) -> Tuple[Figure, YLim]:
    """Plot mean HbO and HbR over time, optionally annotating event markers.

    Parameters
    ----------
    frame
        Wide frame containing oxy/deoxy columns (raw or grand-averaged).
    fs
        Sampling frequency in Hz, used to build the time axis.
    y_lim
        Explicit y-limits; if ``None`` they are derived from the data.
    events
        Optional table with ``Sample number`` (and optional ``Event``) columns.

    Returns
    -------
    (Figure, y_lim)
        The figure and the y-limits actually applied.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(_compose_title(title, subject, condition))

    time = np.arange(len(frame), dtype="float64") / fs
    hbo = _mean_signal(frame, _OXY_KEYS)
    hbr = _mean_signal(frame, _DEOXY_KEYS)

    pooled: List[float] = []
    for signal, colour, label in ((hbo, "red", "HbO"), (hbr, "blue", "HbR")):
        if signal is None:
            continue
        mask = ~np.isnan(signal)
        ax.plot(time[mask], signal[mask], color=colour, label=label, linewidth=1.5)
        pooled.extend(signal[mask].tolist())

    y_lim = y_lim or _padded_limits(pooled)
    ax.set_ylim(y_lim)
    if events is not None and not events.empty:
        _annotate_events(ax, events, fs, time)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δ[Hb] (µM)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, y_lim


def plot_channels_separately(
    frame: pd.DataFrame,
    fs: float,
    title: str = "Channel signals",
    subject: str = "",
    condition: str = "",
    y_lim: Optional[YLim] = None,
) -> Tuple[Figure, List[Axes], YLim]:
    """Plot each channel's HbO/HbR pair in its own stacked subplot.

    Columns are expected to be named ``"<channel> <species>"`` (for example
    ``"CH1 HbO"``); columns that do not split into at least two tokens are
    ignored.
    """
    channels = _group_columns_by_channel(frame)
    channel_ids = sorted(channels)
    if not channel_ids:
        raise ValueError("No recognisable '<channel> <species>' columns to plot.")

    time = np.arange(len(frame)) / fs
    fig, axes = plt.subplots(
        nrows=len(channel_ids), ncols=1,
        figsize=(10, 3 * len(channel_ids)), sharex=True, squeeze=False,
    )
    axis_list = [row[0] for row in axes]
    fig.suptitle(_compose_title(title, subject, condition))

    if y_lim is None:
        pooled = [v for ch in channel_ids for key in channels[ch]
                  if any(k in key for k in _SIGNAL_KEYS)
                  for v in channels[ch][key].dropna().tolist()]
        y_lim = _padded_limits(pooled)

    for ax, channel_id in zip(axis_list, channel_ids):
        _plot_single_channel(ax, channels[channel_id], channel_id, time)
        ax.set_ylim(y_lim)
        ax.set_ylabel("Δ[Hb] (µM)")
        ax.legend(loc="upper right")
    axis_list[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axis_list, y_lim


def calculate_global_ylim(
    frames: Sequence[pd.DataFrame],
    include_keys: Optional[Sequence[str]] = None,
) -> Optional[YLim]:
    """Pool signal columns across frames to derive shared y-limits, or ``None``."""
    keys = tuple(include_keys) if include_keys else _SIGNAL_KEYS
    pooled: List[float] = []
    for frame in frames:
        if not hasattr(frame, "columns"):
            continue
        for column in frame.columns:
            if any(k in column for k in keys):
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
                pooled.extend(values[np.isfinite(values)].tolist())
    if not pooled:
        return None
    return _padded_limits(pooled)


# ----- internals ---------------------------------------------------------- #
def _mean_signal(frame: pd.DataFrame, keywords: Sequence[str]) -> Optional[np.ndarray]:
    """Average all columns whose name contains any keyword, coercing to numeric."""
    cols = [c for c in frame.columns if any(k in c for k in keywords)]
    if not cols:
        return None
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1).to_numpy()


def _padded_limits(values: Sequence[float], fallback: YLim = (-1.0, 1.0)) -> YLim:
    """Return ``(min, max)`` padded by a small fraction of the range."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return fallback
    low, high = min(finite), max(finite)
    pad = _PAD_FRACTION * (high - low) if high > low else 0.1
    return low - pad, high + pad


def _compose_title(title: str, subject: str, condition: str) -> str:
    """Build a multi-line title from the provided fragments."""
    parts = [title]
    if subject:
        parts.append(f"Subject: {subject}")
    if condition:
        parts.append(f"({condition})")
    return "\n".join(parts)


def _group_columns_by_channel(frame: pd.DataFrame) -> dict:
    """Map ``channel -> {species: Series}`` from ``"<channel> <species>"`` columns."""
    channels: dict = {}
    for column in frame.columns:
        tokens = column.split()
        if len(tokens) < 2:
            continue
        channel_id, species = tokens[0], tokens[1]
        channels.setdefault(channel_id, {})[species] = frame[column]
    return channels


def _plot_single_channel(ax: Axes, series_by_species: dict, channel_id: str,
                         time: np.ndarray) -> None:
    """Plot the first available oxy and deoxy trace for one channel."""
    for keys, colour in ((_OXY_KEYS, "r-"), (_DEOXY_KEYS, "b-")):
        for key in keys:
            if key not in series_by_species:
                continue
            values = series_by_species[key].to_numpy()
            if len(values) == len(time):
                ax.plot(time, values, colour, label=f"{channel_id} {key}")
            break


def _annotate_events(ax: Axes, events: pd.DataFrame, fs: float, time: np.ndarray) -> None:
    """Draw vertical lines (and labels) for events that fall within the trace."""
    if "Sample number" not in events.columns or len(time) == 0:
        return
    text_y = ax.get_ylim()[1] * 0.95
    for _, row in events.iterrows():
        sample = row.get("Sample number")
        if pd.isna(sample):
            continue
        event_time = float(sample) / fs
        if not 0 <= event_time <= time[-1]:
            continue
        ax.axvline(x=event_time, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        label = row.get("Event")
        if pd.notna(label):
            ax.text(event_time, text_y, str(label), rotation=90, va="top", ha="right",
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
