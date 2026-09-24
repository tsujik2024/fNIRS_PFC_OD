import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CH_RE = re.compile(r'^CH(\d+)\s+(HbO|O2Hb|HHb|HbR)$')
_OUT_COLS = ('left oxy', 'left deoxy', 'right oxy', 'right deoxy', 'grand oxy', 'grand deoxy')


def average_channels(df, channels_to_exclude=None, left_ids=None, right_ids=None, short_ids=None):
    """Average CH{n} HbO/HbR into left/right/grand hemisphere means.

    Short channels are always excluded -- they're SCR regressors, not signal.
    If left_ids/right_ids/short_ids aren't given, the montage is guessed from
    which channel numbers are present (see _infer_montage): pass them
    explicitly whenever you know the montage, since the guess is genuinely
    ambiguous in one case (see below).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df)}")

    present = {int(m.group(1)) for c in df.columns if (m := _CH_RE.match(str(c)))}
    if not present:
        logger.warning("no CH{n} HbO/HHb columns found, returning an all-NaN frame")
        return _passthrough(df)

    if left_ids is None or right_ids is None or short_ids is None:
        zero_based, why = _infer_montage(present)
        logger.warning("montage not fully specified, inferred zero_based=%s (%s), present=%s",
                       zero_based, why, sorted(present))
        def_right, def_left, def_short = ([0, 1, 2], [4, 6, 7], [3, 5]) if zero_based \
            else ([1, 2, 3], [5, 7, 8], [4, 6])
        left_ids = def_left if left_ids is None else left_ids
        right_ids = def_right if right_ids is None else right_ids
        short_ids = def_short if short_ids is None else short_ids

    exclude = set(channels_to_exclude or ()) | set(short_ids)
    left = [i for i in left_ids if i in present and i not in exclude]
    right = [i for i in right_ids if i in present and i not in exclude]

    def cols(ids, chromo):
        keys = ('HbO', 'O2Hb') if chromo == 'oxy' else ('HbR', 'HHb')
        return [f'CH{i} {k}' for i in ids for k in keys if f'CH{i} {k}' in df.columns]

    def mean(cols):
        return df[cols].mean(axis=1) if cols else pd.Series(np.nan, index=df.index)

    l_oxy, l_deoxy = cols(left, 'oxy'), cols(left, 'deoxy')
    r_oxy, r_deoxy = cols(right, 'oxy'), cols(right, 'deoxy')

    out = _meta_cols(df)
    out['left oxy'], out['left deoxy'] = mean(l_oxy), mean(l_deoxy)
    out['right oxy'], out['right deoxy'] = mean(r_oxy), mean(r_deoxy)
    out['grand oxy'] = mean(l_oxy + r_oxy)
    out['grand deoxy'] = mean(l_deoxy + r_deoxy)
    return pd.DataFrame(out, index=df.index)


def _meta_cols(df):
    out = {}
    if 'Sample number' in df.columns:
        out['Sample number'] = df['Sample number']
    if 'Event' in df.columns:
        out['Event'] = df['Event']
    return out


def _passthrough(df):
    out = _meta_cols(df)
    nan = pd.Series(np.nan, index=df.index)
    out.update({col: nan for col in _OUT_COLS})
    return pd.DataFrame(out, index=df.index)


def _infer_montage(present):
    """CH0 only exists 0-based, CH8 only exists 1-based (8-channel device).

    If neither shows up -- ex QC dropped CH0 from an otherwise 0-based
    recording, leaving CH1-CH7, which looks exactly like a 1-based montage --
    default to 0-based rather than guess wrong
    """
    if 0 in present:
        return True, "CH0 present"
    if 8 in present:
        return False, "CH8 present"
    return True, "no CH0/CH8 present, defaulting to zero-based"
