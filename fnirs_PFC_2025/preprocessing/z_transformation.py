import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


def z_transformation(df: pd.DataFrame, signal_cols: List[str] = None) -> pd.DataFrame:
    """
    Apply Z-transformation to fNIRS signals as described in the paper.

    Paper: 'Finally, we z-transformed each fNIRS channel to be able to average
    over multiple channels belonging to the same cortical region, obtaining
    normally distributed data. We calculated z-scores by subtracting the mean
    HbO or HbR values of each channel and each run, and by dividing those by
    their standard deviation.'

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with fNIRS channels
    signal_cols : List[str], optional
        Specific columns to transform. If None, auto-detect fNIRS columns.

    Returns:
    --------
    pd.DataFrame
        Data with Z-transformed fNIRS channels
    """
    try:
        z_data = df.copy()

        # Auto-detect fNIRS columns if not specified
        if signal_cols is None:
            signal_cols = [col for col in df.columns
                           if any(kw in col for kw in ['HbO', 'HbR', 'O2Hb', 'HHb', 'oxy', 'deoxy'])
                           and pd.api.types.is_numeric_dtype(df[col])]

        if not signal_cols:
            logger.warning("No fNIRS signal columns found for Z-transformation")
            return z_data

        logger.info(f"Applying Z-transformation to {len(signal_cols)} channels")

        transformed_count = 0
        for col in signal_cols:
            if col in z_data.columns:
                signal = z_data[col].values.copy()

                # Handle missing values
                valid_mask = np.isfinite(signal)
                valid_count = np.sum(valid_mask)

                if valid_count < 10:  # Need minimum data points
                    logger.warning(f"Not enough valid data for Z-transformation in {col} ({valid_count} valid points)")
                    continue

                # Calculate statistics on valid data only
                valid_signal = signal[valid_mask]
                mean_val = np.mean(valid_signal)
                std_val = np.std(valid_signal)

                if std_val > 1e-10:  # Avoid division by very small numbers
                    # Apply Z-transformation: (x - μ) / σ
                    signal[valid_mask] = (valid_signal - mean_val) / std_val
                    # Keep original NaN/inf values as they were
                    z_data[col] = signal
                    transformed_count += 1

                    logger.debug(f"Z-transformed {col}: μ={mean_val:.3f}, σ={std_val:.3f}")
                else:
                    logger.warning(f"Near-zero standard deviation in {col}, skipping Z-transformation")
                    # Option: set to zero or keep original? Paper doesn't specify edge cases
                    # Let's keep original to be safe

        logger.info(f"Successfully Z-transformed {transformed_count}/{len(signal_cols)} channels")
        return z_data

    except Exception as e:
        logger.error(f"Z-transformation failed: {str(e)}")
        return df


def z_transformation_by_chromophore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Z-transformation separately for HbO and HbR channels.

    Useful when you want to preserve relative scaling between chromophores.
    """
    try:
        z_data = df.copy()

        # Separate by chromophore
        hbo_cols = [col for col in df.columns if any(kw in col for kw in ['HbO', 'O2Hb', 'oxy'])]
        hbr_cols = [col for col in df.columns if any(kw in col for kw in ['HbR', 'HHb', 'deoxy'])]

        logger.info(f"Applying Z-transformation by chromophore: {len(hbo_cols)} HbO, {len(hbr_cols)} HbR channels")

        # Transform HbO channels
        if hbo_cols:
            hbo_data = z_transformation(df[hbo_cols])
            for col in hbo_cols:
                z_data[col] = hbo_data[col]

        # Transform HbR channels
        if hbr_cols:
            hbr_data = z_transformation(df[hbr_cols])
            for col in hbr_cols:
                z_data[col] = hbr_data[col]

        return z_data

    except Exception as e:
        logger.error(f"Chromophore-wise Z-transformation failed: {str(e)}")
        return df
