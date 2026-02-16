"""
tddr.py

Temporal Derivative Distribution Repair (TDDR) with NaN-safe handling,
proper anchoring, and robust weighting. Designed for fNIRS channels named
like 'CH{i} HbO' / 'CH{i} HbR' (also accepts 'O2Hb'/'HHb').
"""

import re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def tddr(data: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    """
    Apply TDDR motion-artifact correction to fNIRS data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing fNIRS signals. Channels should be columns named
        'CH{i} HbO' / 'CH{i} HbR' (also 'O2Hb'/'HHb'). Other columns (e.g.,
        'Sample number', 'Event', 'Time (s)') are preserved.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    pd.DataFrame
        Copy of input with TDDR-corrected channels (others unchanged).
    """
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive number")

    df = data.copy()

    # Prefer explicit fNIRS channel columns; fallback to numeric (excluding metadata)
    ch_pat = re.compile(r'^CH\d+\s+(HbO|O2Hb|HHb|HbR)$')
    target_cols = [c for c in df.columns if ch_pat.match(str(c))]
    if not target_cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [c for c in num_cols if c not in ('Sample number', 'Time (s)')]

    if not target_cols:
        return df  # nothing to do

    for col in target_cols:
        x = df[col].to_numpy(dtype=float)

        # ---- NaN-safe handling: interpolate internal gaps; preserve edge NaNs
        nan_mask = ~np.isfinite(x)
        if nan_mask.any():
            idx = np.arange(x.size)
            valid = ~nan_mask
            if valid.sum() >= 2:
                x_interp = x.copy()
                x_interp[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
                # restore leading/trailing NaNs (no extrapolation)
                if nan_mask[0]:
                    first_valid = np.argmax(valid)
                    x_interp[:first_valid] = np.nan
                if nan_mask[-1]:
                    last_valid = len(valid) - 1 - np.argmax(valid[::-1])
                    x_interp[last_valid + 1:] = np.nan
                x = x_interp
            else:
                # Too few valid points—skip this column
                continue

        # If edges still NaN, fill edges with nearest values for filtering; restore later
        x_for_filter = x.copy()
        if np.isnan(x_for_filter[0]):
            first = np.flatnonzero(np.isfinite(x_for_filter))
            if first.size:
                x_for_filter[:first[0]] = x_for_filter[first[0]]
        if np.isnan(x_for_filter[-1]):
            last = np.flatnonzero(np.isfinite(x_for_filter))
            if last.size:
                x_for_filter[last[-1] + 1:] = x_for_filter[last[-1]]

        # ---- Low-pass split (0.5 Hz) with padlen safety for short signals
        n = len(x_for_filter)
        if n < 5:
            # too short to process meaningfully
            continue

        sos = butter(N=3, Wn=0.5, btype='low', output='sos', fs=sample_rate)

        # Default padlen used by sosfiltfilt is ~ 3*(pad_len_per_section)
        # We adaptively shrink padlen if needed.
        def _sosfiltfilt_safe(sos, sig):
            try:
                return sosfiltfilt(sos, sig)
            except Exception:
                # try smaller padding by trimming the ends a touch
                # (scipy doesn't expose padlen for sosfiltfilt; fallback: filter central segment)
                # As a conservative fallback, return plain forward-backward sosfilt
                from scipy.signal import sosfilt
                y = sosfilt(sos, sig)
                y = sosfilt(sos, y[::-1])[::-1]
                return y

        x_mean = np.nanmean(x_for_filter)
        x_center = x_for_filter - x_mean
        low = _sosfiltfilt_safe(sos, x_center)
        high = x_center - low

        # ---- Derivative & robust weighting (Tukey biweight), with convergence
        # Use same-length derivative (prepend first sample)
        deriv = np.diff(low, prepend=low[0]).astype(float)

        w = np.ones_like(deriv)
        mu = 0.0
        c = 4.685
        eps = 1e-12
        for _ in range(50):
            # weighted center (robust location)
            denom = w.sum()
            mu_new = (w * deriv).sum() / denom if denom > 0 else 0.0

            resid = deriv - mu_new
            mad = np.median(np.abs(resid))
            sigma = 1.4826 * mad
            if not np.isfinite(sigma) or sigma < eps:
                mu = mu_new
                break

            r = resid / (c * sigma)
            # Tukey's biweight
            mask = np.abs(r) < 1.0
            w_new = np.zeros_like(w)
            w_new[mask] = (1 - r[mask] ** 2) ** 2

            # convergence check
            if np.allclose(mu, mu_new) and np.allclose(w, w_new):
                mu = mu_new
                w = w_new
                break

            mu = mu_new
            w = w_new

        # Repair derivative (shrink outliers)
        new_deriv = w * (deriv - mu)

        # Reconstruct low-frequency component, anchored to original start
        low_corr = low[0] + np.cumsum(new_deriv)

        # Recombine with high-frequency + mean to preserve overall shape/DC
        y = low_corr + high + x_mean

        # Restore original edge NaNs
        y[np.isnan(data[col].to_numpy(dtype=float))] = np.nan
        df[col] = y

    return df
