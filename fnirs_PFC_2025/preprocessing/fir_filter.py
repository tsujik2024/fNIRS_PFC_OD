import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt

import pandas as pd
import numpy as np
from scipy.signal import firwin, filtfilt
import logging

logger = logging.getLogger(__name__)


def fir_filter(df: pd.DataFrame, order: int, Wn: list, fs: int) -> pd.DataFrame:
    """
    Apply FIR filter to numeric fNIRS data columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with fNIRS channels and metadata
    order : int
        Filter order
    Wn : list
        Cutoff frequencies [low, high]
    fs : int
        Sampling frequency

    Returns:
    --------
    pd.DataFrame
        Filtered data with same structure as input
    """
    filtered_df = df.copy()

    # Select only numeric columns (excluding known metadata columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    data_columns = [col for col in numeric_cols
                    if col not in ['Sample number', 'Event', 'Time (s)']]

    logger.debug(f"Applying FIR filter to columns: {data_columns}")

    for ch in data_columns:
        try:
            ch_asarray = np.array(df[ch], dtype='float64')
            b = firwin(order + 1, Wn, pass_zero=False, fs=fs)
            ch_filtered = filtfilt(b, [1.0], ch_asarray)
            filtered_df[ch] = ch_filtered
        except Exception as e:
            logger.error(f"Failed to filter channel {ch}: {str(e)}")
            raise ValueError(f"FIR filtering failed for column {ch}") from e

    return filtered_df