import re
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Dict, List

def average_channels(
    df: pd.DataFrame,
    channels_to_exclude: Optional[Iterable[int]] = None,
    left_ids: Optional[Iterable[int]] = None,
    right_ids: Optional[Iterable[int]] = None,
    short_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Average multiple fNIRS channels into hemisphere-level and grand-mean signals.

    Assumes columns are named like 'CH{i} HbO' / 'CH{i} HbR' plus optional
    'Sample number' and 'Event'. Works with zero-based (CH0..) or one-based (CH1..) schemes.

    Parameters
    ----------
    channels_to_exclude : iterable of ints, optional
        Additional channel indices to exclude from all averages.
    left_ids, right_ids : iterable of ints, optional
        Explicit hemisphere maps. If omitted, inferred based on numbering scheme.
    short_ids : iterable of ints, optional
        Channels to exclude by default as short channels. If omitted, inferred.

    Returns
    -------
    ret_df : pd.DataFrame
        Columns include:
          - 'Sample number' (if present)
          - 'Event' (if present)
          - 'left oxy', 'left deoxy'
          - 'right oxy', 'right deoxy'
          - 'grand oxy', 'grand deoxy'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Must provide a DataFrame, not {type(df)}.")

    df_copy = df.copy()
    ch_pat = re.compile(r'^CH(\d+)\s+(HbO|O2Hb|HHb|HbR)$')

    # Discover available channel indices
    present_indices: set[int] = set()
    for c in df_copy.columns:
        m = ch_pat.match(str(c))
        if m:
            present_indices.add(int(m.group(1)))

    if not present_indices:
        # No fNIRS channels found; build a minimal passthrough
        cols = {}
        if 'Sample number' in df_copy: cols['Sample number'] = df_copy['Sample number']
        if 'Event' in df_copy: cols['Event'] = df_copy['Event']
        # Fill averages with NaN
        nan_series = pd.Series(np.nan, index=df_copy.index)
        cols.update({
            'left oxy': nan_series, 'left deoxy': nan_series,
            'right oxy': nan_series, 'right deoxy': nan_series,
            'grand oxy': nan_series, 'grand deoxy': nan_series,
        })
        return pd.DataFrame(cols, index=df_copy.index)

    # Detect numbering scheme
    zero_based = (0 in present_indices)  # True if CH0 exists
    # Default hemisphere & short-channel maps (override if provided)
    if left_ids is None or right_ids is None:
        if zero_based:
            # long: 0..5 ; short: 6..7
            default_left  = [3,4,5]
            default_right = [0,1,2]
        else:
            # long: 1..6 ; short: 7..8
            default_left  = [4,5,6]
            default_right = [1,2,3]
        left_ids  = default_left if left_ids is None else list(left_ids)
        right_ids = default_right if right_ids is None else list(right_ids)

    if short_ids is None:
        short_ids = [6,7] if zero_based else [7,8]

    # Apply excludes
    excludes = set(int(x) for x in (channels_to_exclude or [])) | set(int(x) for x in short_ids)
    left_ids_eff  = [i for i in left_ids  if i in present_indices and i not in excludes]
    right_ids_eff = [i for i in right_ids if i in present_indices and i not in excludes]

    # Helper to collect existing column names for a set of channel ids
    def cols_for(ids: Iterable[int], chromo: str) -> List[str]:
        # Accept either 'HbO' or 'O2Hb' as oxy, and 'HbR' or 'HHb' as deoxy
        names = []
        targets = ('HbO','O2Hb') if chromo == 'oxy' else ('HbR','HHb')
        for i in ids:
            for t in targets:
                col = f'CH{i} {t}'
                if col in df_copy.columns:
                    names.append(col)
        return names

    # Build column groups
    left_hbo_cols  = cols_for(left_ids_eff,  'oxy')
    left_hbr_cols  = cols_for(left_ids_eff,  'deoxy')
    right_hbo_cols = cols_for(right_ids_eff, 'oxy')
    right_hbr_cols = cols_for(right_ids_eff, 'deoxy')

    # Compute means (broadcast NaN if empty)
    def mean_or_nan(cols: List[str]) -> pd.Series:
        return df_copy[cols].mean(axis=1) if cols else pd.Series(np.nan, index=df_copy.index)

    left_oxy    = mean_or_nan(left_hbo_cols)
    left_deoxy  = mean_or_nan(left_hbr_cols)
    right_oxy   = mean_or_nan(right_hbo_cols)
    right_deoxy = mean_or_nan(right_hbr_cols)

    grand_oxy_cols   = left_hbo_cols  + right_hbo_cols
    grand_deoxy_cols = left_hbr_cols  + right_hbr_cols
    grand_oxy   = mean_or_nan(grand_oxy_cols)
    grand_deoxy = mean_or_nan(grand_deoxy_cols)

    # Build return frame
    ret_cols: Dict[str, pd.Series] = {}
    if 'Sample number' in df_copy.columns:
        ret_cols['Sample number'] = df_copy['Sample number']
    if 'Event' in df_copy.columns:
        ret_cols['Event'] = df_copy['Event']

    ret_cols.update({
        'left oxy': left_oxy,   'left deoxy': left_deoxy,
        'right oxy': right_oxy, 'right deoxy': right_deoxy,
        'grand oxy': grand_oxy, 'grand deoxy': grand_deoxy,
    })

    return pd.DataFrame(ret_cols, index=df_copy.index)
