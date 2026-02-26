import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_channels_separately(data, fs, title="Channel Signals",
                             subject=None, condition=None, y_lim=None):
    if hasattr(data, "columns"):
        channels = {}
        for col in data.columns:
            parts = col.split()
            if len(parts) < 2:
                continue
            ch_id = parts[0]
            signal_type = parts[1]
            if ch_id not in channels:
                channels[ch_id] = {}
            channels[ch_id][signal_type] = data[col]

        channel_ids = sorted(channels.keys())
        time = np.arange(len(data)) / fs

        fig, axes = plt.subplots(nrows=len(channel_ids), ncols=1,
                                 figsize=(10, 3 * len(channel_ids)), sharex=True)
        if len(channel_ids) == 1:
            axes = [axes]

        title_parts = [title]
        if subject: title_parts.append(f"Subject: {subject}")
        if condition: title_parts.append(f"({condition})")
        fig.suptitle("\n".join(title_parts))

        if y_lim is None:
            all_values = []
            for ch in channel_ids:
                for key in channels[ch]:
                    if key in ["O2Hb", "HbO", "HHb", "HbR"]:
                        valid_values = channels[ch][key].dropna().values
                        all_values.extend(valid_values)
            if all_values:
                min_val, max_val = min(all_values), max(all_values)
                range_val = max_val - min_val
                buffer = 0.05 * range_val if range_val > 0 else 0.1
                y_lim = (min_val - buffer, max_val + buffer)

        for i, ch in enumerate(channel_ids):
            ax = axes[i]
            for o2_key in ["O2Hb", "HbO"]:
                if o2_key in channels[ch]:
                    data_vals = channels[ch][o2_key].values
                    if len(time) == len(data_vals):
                        ax.plot(time, data_vals, 'r-', label=f'{ch} {o2_key}')
                    break
            for hb_key in ["HHb", "HbR"]:
                if hb_key in channels[ch]:
                    data_vals = channels[ch][hb_key].values
                    if len(time) == len(data_vals):
                        ax.plot(time, data_vals, 'b-', label=f'{ch} {hb_key}')
                    break
            if y_lim is not None:
                ax.set_ylim(y_lim)
            ax.set_ylabel("Δ[Hb] (µM)")
            ax.legend(loc='upper right')

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig, axes, y_lim


def plot_overall_signals(df: pd.DataFrame,
                         fs: float,
                         title: str,
                         subject: str,
                         condition: str,
                         y_lim: Optional[Tuple[float, float]] = None,
                         events: Optional[pd.DataFrame] = None):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Title formatting
    title_parts = [title]
    if subject: title_parts.append(f"Subject: {subject}")
    if condition: title_parts.append(f"({condition})")
    ax.set_title("\n".join(title_parts))

    # Full resolution time vector
    time = np.arange(len(df), dtype='float64') / fs

    # Extract HbO and HHb signals
    def extract_mean_signal(df_slice, keywords):
        cols = [col for col in df_slice.columns if any(k in col for k in keywords)]
        if cols:
            return df_slice[cols].apply(lambda x: pd.to_numeric(x, errors='coerce')).mean(axis=1).values
        return None

    hbo = extract_mean_signal(df, ['HbO', 'O2Hb', 'Mean HbO', 'grand oxy'])
    hhb = extract_mean_signal(df, ['HHb', 'HbR', 'Mean HHb', 'grand deoxy'])

    # Plot HbO
    if hbo is not None:
        valid_mask = ~np.isnan(hbo)
        ax.plot(time[valid_mask], hbo[valid_mask], color='red', label='HbO', linewidth=1.5)

    # Plot HHb
    if hhb is not None:
        valid_mask = ~np.isnan(hhb)
        ax.plot(time[valid_mask], hhb[valid_mask], color='blue', label='HbR', linewidth=1.5)

    # Determine Y limits
    if y_lim is None:
        y_data = []
        if hbo is not None:
            y_data.extend(hbo[~np.isnan(hbo)])
        if hhb is not None:
            y_data.extend(hhb[~np.isnan(hhb)])
        if y_data:
            min_y, max_y = min(y_data), max(y_data)
            buffer = 0.05 * (max_y - min_y) if max_y > min_y else 0.1
            y_lim = (min_y - buffer, max_y + buffer)
        else:
            y_lim = (-1, 1)
    ax.set_ylim(y_lim)

    # Plot events if provided
    if events is not None and not events.empty:
        current_ylim = ax.get_ylim()
        text_y = current_ylim[1] * 0.95
        for idx, row in events.iterrows():
            try:
                if 'Sample number' in row and pd.notna(row['Sample number']):
                    event_time = float(row['Sample number']) / fs
                    if 0 <= event_time <= time[-1]:
                        ax.axvline(x=event_time, color='gray', linestyle='--',
                                   linewidth=1, alpha=0.7)
                        if 'Event' in row and pd.notna(row['Event']):
                            ax.text(event_time, text_y, str(row['Event']),
                                    rotation=90, va='top', ha='right',
                                    fontsize=8, alpha=0.8,
                                    bbox=dict(boxstyle='round,pad=0.2',
                                              facecolor='white', alpha=0.7))
            except Exception as e:
                print(f"Warning: Skipping event at index {idx} due to error: {e}")

    # Final formatting
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δ[Hb] (µM)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig, y_lim


def calculate_global_ylim(data_list, include_cols=None):
    if include_cols is None:
        include_cols = ["O2Hb", "HbO", "HHb", "HbR", "Mean HbO", "Mean HHb", "grand oxy", "grand deoxy"]

    all_values = []

    for data in data_list:
        if not hasattr(data, "columns"):
            continue
        for col in data.columns:
            if any(pattern in col for pattern in include_cols):
                valid_values = data[col].dropna().values
                finite_values = valid_values[np.isfinite(valid_values)]
                all_values.extend(finite_values)

    if not all_values:
        return None

    min_val = min(all_values)
    max_val = max(all_values)
    range_val = max_val - min_val
    buffer = 0.05 * range_val if range_val > 0 else 0.1
    return (min_val - buffer, max_val + buffer)
