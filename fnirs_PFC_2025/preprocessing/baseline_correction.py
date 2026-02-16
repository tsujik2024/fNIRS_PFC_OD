import re
import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def baseline_subtraction(
        df: pd.DataFrame,
        events_df: pd.DataFrame,
        baseline_type: str = "long_walk"  # "long_walk", "event_based" (aka S1→S2), or "lshape_task"
) -> pd.DataFrame:
    """
    Baseline-subtract fNIRS signals using trimmed mean (10% each tail).
    Supports:
      - "long_walk":      S1 → W1   (fallback: S1 → S2 if W1 missing)
      - "event_based":    S1 → S2
      - "lshape_task":    use explicit 'BaselineStart'/'BaselineEnd' provided by caller
    If 'BaselineStart'/'BaselineEnd' are present in events_df, they take precedence.

    Notes:
      - Only HbO/O2Hb and HbR/HHb channel columns are baseline-subtracted.
      - Sample numbers are treated as **indices** (0-based), end is **exclusive**.
    """
    if events_df is None or events_df.empty:
        raise ValueError("No events dataframe provided for baseline subtraction.")

    # Work on a copy; normalize event labels
    ev = events_df.copy()
    if 'Event' not in ev.columns or 'Sample number' not in ev.columns:
        raise ValueError("events_df must contain 'Event' and 'Sample number' columns.")

    # Clean event labels and sample numbers
    ev['Event'] = ev['Event'].astype(str).str.strip().str.upper()
    ev['Sample number'] = pd.to_numeric(ev['Sample number'], errors='coerce')
    ev = ev.dropna(subset=['Sample number'])
    ev['Sample number'] = ev['Sample number'].astype(int)

    # Keep only events that fall on or within the data range
    n = len(df)
    ev = ev[(ev['Sample number'] >= 0) & (ev['Sample number'] <= n)]
    if ev.empty:
        raise ValueError("All event markers are out of bounds for the data length.")

    # Prefer explicit BaselineStart/BaselineEnd if present (e.g., L-Shape flow)
    if set(['BASELINESTART', 'BASELINEEND']).issubset(set(ev['Event'])):
        start = int(ev.loc[ev['Event'] == 'BASELINESTART', 'Sample number'].iloc[0])
        end   = int(ev.loc[ev['Event'] == 'BASELINEEND',   'Sample number'].iloc[0])
        label = "BaselineStart→BaselineEnd"
    else:
        # Map baseline mode to required end marker
        mode = baseline_type.lower().strip()
        if mode not in ("long_walk", "event_based", "stop_signal", "lshape_task"):
            logger.warning(f"Unknown baseline_type '{baseline_type}', defaulting to 'event_based' (S1→S2).")
            mode = "event_based"

        # Normalize aliases (if present)
        alias_map = {"BASELINESTART": "S1"}
        if mode == "long_walk":
            alias_map["BASELINEEND"] = "W1"
        else:
            alias_map["BASELINEEND"] = "S2"
        ev['Event'] = ev['Event'].replace(alias_map)

        # Must have S1
        if 'S1' not in set(ev['Event']):
            raise ValueError("Missing required 'S1' event for baseline estimation.")

        s1_idx = int(ev.loc[ev['Event'] == 'S1', 'Sample number'].iloc[0])

        if mode == "long_walk":
            # Prefer W1 *after* S1; if none, allow S2 after S1
            candidates = ev[(ev['Sample number'] > s1_idx) & (ev['Event'] == 'W1')]
            if candidates.empty:
                logger.warning("W1 not found after S1; falling back to S2 after S1 for long_walk baseline.")
                candidates = ev[(ev['Sample number'] > s1_idx) & (ev['Event'] == 'S2')]
            if candidates.empty:
                raise ValueError("Missing end marker (W1/S2) after S1 for long_walk baseline.")
            end_idx = int(candidates.sort_values('Sample number').iloc[0]['Sample number'])
            start, end, label = s1_idx, end_idx, "S1→W1"
        else:
            # event_based / stop_signal: S1 → S2 (S2 must be after S1)
            candidates = ev[(ev['Sample number'] > s1_idx) & (ev['Event'] == 'S2')]
            if candidates.empty:
                raise ValueError("Missing 'S2' after 'S1' for event_based baseline.")
            end_idx = int(candidates.sort_values('Sample number').iloc[0]['Sample number'])
            start, end, label = s1_idx, end_idx, "S1→S2"

    # Validate window (end is exclusive)
    if not (0 <= start < n):
        raise ValueError(f"S1/BaselineStart sample {start} out of bounds (0..{n-1}).")
    if not (0 < end <= n):
        raise ValueError(f"End sample {end} out of bounds (1..{n}).")
    if start >= end:
        raise ValueError(f"Invalid baseline window: start ({start}) ≥ end ({end}).")

    # Identify only signal columns to correct (HbO/O2Hb & HbR/HHb)
    ch_pat = re.compile(r'^CH\d+\s+(HbO|O2Hb|HHb|HbR)$')
    signal_cols = [c for c in df.columns if ch_pat.match(str(c))]
    if not signal_cols:
        logger.warning("No HbO/HbR channel columns found; returning input unchanged.")
        return df.copy()

    # Compute trimmed-mean baseline per channel (drop NaNs)
    corrected = df.copy()
    baseline_slice = slice(start, end)  # end is exclusive
    win_len = end - start

    logger.info(
        f"Applying {label} baseline correction with trimmed mean "
        f"(samples {start}–{end-1}, n={win_len}, trim 10% tails)."
    )

    for ch in signal_cols:
        base_vals = corrected.iloc[baseline_slice][ch].astype(float).dropna()
        if base_vals.empty:
            # If baseline window is entirely NaN for this channel, skip correction
            logger.warning(f"Baseline window is empty/NaN for {ch}; skipping baseline subtraction for this channel.")
            continue

        # Guard for tiny windows: trim_mean handles small n (trimming may be 0)
        trimmed_mean = stats.trim_mean(base_vals, proportiontocut=0.10)
        if not np.isfinite(trimmed_mean):
            # Fallback to simple mean if trimming produced non-finite value
            trimmed_mean = float(base_vals.mean())

        corrected[ch] = corrected[ch].astype(float) - trimmed_mean

        # Optional debug
        # regular_mean = float(base_vals.mean())
        # logger.debug(f"{ch}: mean={regular_mean:.6f}, trimmed={trimmed_mean:.6f}")

    return corrected
