import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import logging

logger = logging.getLogger(__name__)

_METADATA_COLS = ("Sample number", "Event", "Time (s)", "ADC")


def butterworth_bandpass(df: pd.DataFrame, order: int, Wn: list, fs: int) -> pd.DataFrame:

    filtered_df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data_columns = [c for c in numeric_cols if c not in _METADATA_COLS]

    if not data_columns:
        logger.warning("Butterworth bandpass: no numeric signal columns to filter.")
        return filtered_df

    n_samples = len(df)

    nyquist = fs / 2
    if any(w <= 0 or w >= nyquist for w in Wn):
        logger.error(f"Butterworth bandpass: Wn={Wn} must be strictly between 0 and "
                     f"Nyquist ({nyquist} Hz); returning data unfiltered.")
        return filtered_df

    try:
        sos = butter(order, Wn, btype="bandpass", fs=fs, output="sos")
    except Exception as e:
        logger.error(f"Butterworth bandpass: filter design failed (order={order}, Wn={Wn}, "
                     f"fs={fs}): {e}; returning data unfiltered.")
        return filtered_df

    min_len = 3 * (2 * len(sos) + 1)
    if n_samples <= min_len:
        logger.warning(
            f"Butterworth bandpass: {n_samples} samples is too short for order={order} "
            f"(needs > {min_len}); skipping (returning unfiltered) to avoid corrupting the signal."
        )
        return filtered_df

    logger.debug(f"Butterworth bandpass: {Wn} Hz, order={order}, fs={fs}, columns={data_columns}")

    for ch in data_columns:
        x = np.asarray(df[ch], dtype="float64")
        if not np.all(np.isfinite(x)):
            logger.warning(f"Butterworth bandpass: column {ch} contains non-finite values; leaving unfiltered.")
            continue
        try:
            filtered_df[ch] = sosfiltfilt(sos, x)
        except Exception as e:
            logger.error(f"Butterworth bandpass: failed on column {ch}: {e}; leaving that column unfiltered.")

    return filtered_df
