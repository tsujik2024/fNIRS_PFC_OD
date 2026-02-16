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
    Find and return the short channel for a given long channel using CH{i} naming.

    For prefrontal cap configuration with 0-based indexing (CH0-CH7):
    - Short channels: CH6 and CH7 (both HbO and HbR)
    - Mapping:
        CH0, CH1, CH2 → CH6 (left side)
        CH3, CH4, CH5 → CH7 (right side)

    Parameters
    ----------
    long_ch : str
        Long channel name in format "CH{i} HbO" or "CH{i} HbR"
    short_data : pd.DataFrame
        DataFrame containing short channel data

    Returns
    -------
    str
        Matching short channel name
    """
    short_chs = list(short_data.columns)

    # Extract channel number and chromophore from "CH1 HbO" format
    ch_match = re.match(r'CH(\d+) (HbO|HbR|O2Hb|HHb)', long_ch)
    if not ch_match:
        raise KeyError(f"Invalid long channel format: {long_ch}. Expected 'CH{{i}} HbO' or 'CH{{i}} HbR'")

    ch_num = int(ch_match.group(1))
    oxygenation = ch_match.group(2)

    # For prefrontal cap with 0-based indexing: CH6 and CH7 are short channels
    channel_mapping = {
        # Left hemisphere channels → CH6 (short)
        0: 6, 1: 6, 2: 6,
        # Right hemisphere channels → CH7 (short)
        3: 7, 4: 7, 5: 7,
    }

    target_short_num = channel_mapping.get(ch_num)
    if not target_short_num:
        raise KeyError(f"No short channel mapping defined for {long_ch}")

    short_ch_name = f"CH{target_short_num} {oxygenation}"

    if short_ch_name not in short_chs:
        raise KeyError(f"Mapped short channel {short_ch_name} not found in available short channels: {short_chs}")

    return short_ch_name