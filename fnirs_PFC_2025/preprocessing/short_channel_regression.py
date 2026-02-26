"""
Implements short channel correction (short channel regression) to remove
superficial components from long-channel fNIRS signals using channel-specific pairing.

References:
    - Scholkmann et al., 2014 (Physiological Measurement)
    - Gagnon et al., 2014
    - Brigadoi et al., 2014
"""

import pandas as pd
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)


def scr_regression(long_data: pd.DataFrame, short_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply short channel correction to remove the superficial (skin blood flow) component
    from long-channel fNIRS measurements using channel-specific pairing.

    For each long channel, finds the nearest short channel and performs individual
    linear regression to remove the superficial component.

    Parameters
    ----------
    long_data : pd.DataFrame
        DataFrame containing fNIRS data (columns) for the long channels.
        Each column is typically something like "CH1 HbO", "CH1 HbR", etc.
        Rows represent timepoints/samples.
    short_data : pd.DataFrame
        DataFrame containing fNIRS data (columns) for the short reference channels,
        measured at superficial depths. Each column is typically "CHx HbO"/"CHx HbR"
        for short-separation channels (CH7 and CH8 in prefrontal cap).

    Returns
    -------
    long_data_corrected : pd.DataFrame
        DataFrame with the same shape and columns as `long_data`,
        but after subtracting the short-channel component.
    """
    # Copy to avoid mutating the original data
    long_data_corrected = long_data.copy()
    short_data_copy = short_data.copy()

    long_chs = list(long_data.columns)

    for long_ch in long_chs:
        try:
            # Find the appropriate short channel for this specific long channel
            short_ch = _find_matching_short(long_ch, short_data_copy)

            # Extract the signal arrays
            long_array = np.array(long_data[long_ch], dtype='float64')
            short_array = np.array(short_data[short_ch], dtype='float64')

            # Compute regression coefficient: beta = (X^T Y) / (X^T X)
            denom = np.dot(short_array, short_array)
            if denom == 0:
                # If short channel is all zeros, skip correction for this channel
                beta = 0.0
                logger.warning(f"Short channel {short_ch} has zero variance, skipping correction for {long_ch}")
            else:
                beta = np.dot(short_array, long_array) / denom

            # Subtract the superficial component
            corrected = long_array - (beta * short_array)
            long_data_corrected[long_ch] = corrected

            logger.debug(f"SCR: {long_ch} corrected using {short_ch} (beta={beta:.4f})")

        except KeyError as e:
            logger.warning(f"Could not find matching short channel for {long_ch}: {e}")
            # Keep the original signal if no matching short channel found
            continue
        except Exception as e:
            logger.warning(f"Error processing {long_ch}: {e}")
            # Keep the original signal if any other error occurs
            continue

    return long_data_corrected


def _find_matching_short(long_ch: str, short_data: pd.DataFrame) -> str:
    """
    Map a long channel to its paired short channel for SCR.

    Assumes 0-based channel naming: CH0..CH7
    True short channels: CH2 and CH6

    IMPORTANT:
    - If the incoming channel is itself a short channel (CH2/CH6), we refuse to correct it.
    """
    short_chs = list(short_data.columns)

    m = re.match(r'CH(\d+)\s+(HbO|HbR|O2Hb|HHb)$', str(long_ch))
    if not m:
        raise KeyError(f"Invalid channel format: {long_ch}")

    ch_num = int(m.group(1))
    chromo = m.group(2)

    SHORT_IDS = {2, 6}
    if ch_num in SHORT_IDS:
        raise KeyError(f"{long_ch} is a short channel (CH2/CH6) and should not be SCR-corrected.")

    # Pick your mapping. You must choose which long channels pair with which short.
    # If you don't know yet, this is a reasonable starting point:
    channel_mapping = {
        0: 6, 1: 6, 3: 6,   # example grouping to CH6
        4: 2, 5: 2, 7: 2,   # example grouping to CH2
        # NOTE: you must decide where CH7 belongs if CH7 is a long channel in your setup
    }

    target_short_num = channel_mapping.get(ch_num)
    if target_short_num is None:
        raise KeyError(f"No short channel mapping defined for {long_ch} (CH{ch_num}).")

    short_ch_name = f"CH{target_short_num} {chromo}"

    if short_ch_name not in short_chs:
        raise KeyError(f"Mapped short channel {short_ch_name} not found. Available: {short_chs}")

    return short_ch_name
