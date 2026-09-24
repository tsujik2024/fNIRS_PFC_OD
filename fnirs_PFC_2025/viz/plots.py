import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OXY = ("HbO", "O2Hb", "grand oxy")
DEOXY = ("HHb", "HbR", "grand deoxy")


def plot_overall_signals(frame, fs, title="Grand-average signals", subject="", condition="", events=None):
    """Mean HbO/HbR over time, with event markers if `events` has 'Sample number'/'Event'."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(_title(title, subject, condition))

    time = np.arange(len(frame)) / fs
    pooled = []
    for keys, color, label in ((OXY, "red", "HbO"), (DEOXY, "blue", "HbR")):
        sig = _mean(frame, keys)
        if sig is None:
            continue
        ok = ~np.isnan(sig)
        ax.plot(time[ok], sig[ok], color=color, label=label, linewidth=1.5)
        pooled.append(sig[ok])
    ax.set_ylim(_limits(np.concatenate(pooled) if pooled else np.array([])))

    if events is not None and len(time):
        for _, row in events.iterrows():
            t = row["Sample number"] / fs
            if 0 <= t <= time[-1]:
                ax.axvline(t, color="gray", linestyle="--", linewidth=1, alpha=0.7)
                lo, hi = ax.get_ylim()
                ax.text(t, hi - 0.05 * (hi - lo), str(row["Event"]), rotation=90, va="top", ha="right",
                        fontsize=8, alpha=0.8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δ[Hb] (µM)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_channels_separately(frame, fs, title="Channel signals", subject="", condition=""):
    """One subplot per channel; columns must look like 'CH1 HbO'."""
    channels = {}
    for col in frame.columns:
        tokens = col.split()
        if len(tokens) >= 2:
            channels.setdefault(tokens[0], {})[tokens[1]] = frame[col]
    if not channels:
        raise ValueError("no '<channel> <species>' columns to plot")

    ids = sorted(channels)
    time = np.arange(len(frame)) / fs
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3 * len(ids)), sharex=True, squeeze=False)
    fig.suptitle(_title(title, subject, condition))

    pooled = np.concatenate([s.dropna().to_numpy() for ch in channels.values()
                             for species, s in ch.items() if species in OXY + DEOXY])
    ylim = _limits(pooled)
    for (ax,), ch in zip(axes, ids):
        for keys, style in ((OXY, "r-"), (DEOXY, "b-")):
            species = next((k for k in keys if k in channels[ch]), None)
            if species:
                ax.plot(time, channels[ch][species].to_numpy(), style, label=f"{ch} {species}")
        ax.set_ylim(ylim)
        ax.set_ylabel("Δ[Hb] (µM)")
        ax.legend(loc="upper right")
    axes[-1][0].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _title(title, subject, condition):
    parts = [title]
    if subject:
        parts.append(f"Subject: {subject}")
    if condition:
        parts.append(f"({condition})")
    return "\n".join(parts)


def _mean(frame, keys):
    cols = [c for c in frame.columns if any(k in c for k in keys)]
    if not cols:
        return None
    return frame[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()


def _limits(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return (-1.0, 1.0)
    lo, hi = values.min(), values.max()
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad
