import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
import os
logger = logging.getLogger(__name__)


def calc_cv(signal: np.ndarray) -> float:
    """
    Calculate coefficient of variation for an fNIRS signal.

    CV = (standard deviation / |mean|) * 100

    Parameters
    ----------
    signal : np.ndarray
        fNIRS concentration signal (HbO or HbR)

    Returns
    -------
    float
        Coefficient of variation as percentage
        Lower values indicate better signal quality
    """
    # Remove NaN and infinite values
    valid_signal = signal[np.isfinite(signal)]

    if len(valid_signal) < 10:  # Need minimum data points
        return 999.0  # Very high CV indicates poor quality

    mean_val = np.mean(valid_signal)
    std_val = np.std(valid_signal)

    if abs(mean_val) < 1e-10:  # Avoid division by very small numbers
        return 999.0

    cv = (std_val / abs(mean_val)) * 100
    return cv


def assess_cv_quality(cv_value: float) -> str:
    """
    Classify signal quality based on CV value.

    Based on typical fNIRS literature thresholds:
    - CV < 15%: Excellent
    - CV 15-25%: Good
    - CV 25-50%: Fair
    - CV > 50%: Poor

    Parameters
    ----------
    cv_value : float
        Coefficient of variation percentage

    Returns
    -------
    str
        Quality classification
    """
    if cv_value < 15:
        return "EXCELLENT"
    elif cv_value < 25:
        return "GOOD"
    elif cv_value < 50:
        return "FAIR"
    else:
        return "POOR"


def calculate_channel_cv(filtered_data: pd.DataFrame,
                         o2hb_cols: List[str], hhb_cols: List[str],
                         cv_threshold: float = 50.0) -> Tuple[List[dict], List[Tuple]]:
    """
    Calculate CV for all fNIRS channels and identify poor quality channels.

    Parameters
    ----------
    filtered_data : pd.DataFrame
        Filtered fNIRS data with HbO/HbR columns
    o2hb_cols : list
        List of HbO column names
    hhb_cols : list
        List of HbR column names
    cv_threshold : float
        CV threshold above which channels are flagged as poor quality

    Returns
    -------
    tuple
        (all_results, flagged_channels)
        all_results: List of dicts with CV results for each channel
        flagged_channels: List of (channel_id, hbo_cv, hbr_cv, status) tuples
    """
    all_results = []
    flagged = []

    for o2hb_col in o2hb_cols:
        ch_id = o2hb_col.split()[0]
        hhb_col = next((c for c in hhb_cols if c.startswith(ch_id)), None)

        if hhb_col:
            # Calculate CV for both HbO and HbR
            hbo_signal = filtered_data[o2hb_col].to_numpy(dtype=np.float64)
            hbr_signal = filtered_data[hhb_col].to_numpy(dtype=np.float64)

            hbo_cv = calc_cv(hbo_signal)
            hbr_cv = calc_cv(hbr_signal)

            # Store results
            result = {
                'channel_id': ch_id,
                'hbo_cv': hbo_cv,
                'hbr_cv': hbr_cv,
                'hbo_quality': assess_cv_quality(hbo_cv),
                'hbr_quality': assess_cv_quality(hbr_cv),
                'avg_cv': (hbo_cv + hbr_cv) / 2
            }

            # Overall channel quality (worst of the two)
            if hbo_cv > cv_threshold or hbr_cv > cv_threshold:
                result['overall_quality'] = 'POOR'
                flagged.append((ch_id, hbo_cv, hbr_cv, 'POOR'))
            elif max(hbo_cv, hbr_cv) > 25:
                result['overall_quality'] = 'FAIR'
            elif max(hbo_cv, hbr_cv) > 15:
                result['overall_quality'] = 'GOOD'
            else:
                result['overall_quality'] = 'EXCELLENT'

            all_results.append(result)

    return all_results, flagged


# Updated method for your FileProcessor class
def _calculate_cv_quality(self, filtered_data: pd.DataFrame,
                          o2hb_cols: List[str], hhb_cols: List[str],
                          output_dir: str, file_basename: str) -> None:
    """
    Calculate CV quality metrics and log results.
    Replacement for the SCI calculation method.
    """
    all_results, flagged = calculate_channel_cv(
        filtered_data, o2hb_cols, hhb_cols, cv_threshold=self.cv_threshold
    )

    # Save all CV values to comprehensive log
    if all_results:
        all_cv_log = os.path.join(output_dir, f"{os.path.splitext(file_basename)[0]}_all_CV_channels.txt")
        with open(all_cv_log, 'w') as f:
            f.write("Channel\tHbO_CV\tHbR_CV\tAvg_CV\tHbO_Quality\tHbR_Quality\tOverall_Quality\n")
            for result in all_results:
                f.write(f"{result['channel_id']}\t{result['hbo_cv']:.2f}\t{result['hbr_cv']:.2f}\t"
                        f"{result['avg_cv']:.2f}\t{result['hbo_quality']}\t{result['hbr_quality']}\t"
                        f"{result['overall_quality']}\n")
        logger.info(f"Saved all CV values to: {all_cv_log}")

    # Save poor quality channels to separate file
    if flagged:
        poor_cv_log = os.path.join(output_dir, f"{os.path.splitext(file_basename)[0]}_poor_CV_channels.txt")
        with open(poor_cv_log, 'w') as f:
            f.write("Channel\tHbO_CV\tHbR_CV\tStatus\n")
            for ch_id, hbo_cv, hbr_cv, status in flagged:
                f.write(f"{ch_id}\t{hbo_cv:.2f}\t{hbr_cv:.2f}\t{status}\n")
        logger.warning(f"Found {len(flagged)} poor quality channels (CV > {self.cv_threshold}%)")
    else:
        logger.info(f"All {len(all_results)} channels passed CV quality threshold (< {self.cv_threshold}%)")


