import logging

import numpy as np

logger = logging.getLogger(__name__)

_OXY_KEYS = ("HbO", "O2Hb", "_oxy")
_DEOXY_KEYS = ("HbR", "HHb", "_deoxy")


def scr_regression(long_data, short_data):
    """Regress each long channel against the mean of the short channels.

    The regressor is centered before fitting, so a DC offset in the short
    channel can't get scaled up into the corrected long channel -- only the
    correlated varyied part gets subtracted.
    """
    corrected = long_data.copy()

    for keys in (_OXY_KEYS, _DEOXY_KEYS):
        long_cols = [c for c in long_data.columns if any(k in str(c) for k in keys)]
        short_cols = [c for c in short_data.columns if any(k in str(c) for k in keys)]
        if not long_cols:
            continue
        if not short_cols:
            logger.warning("no short-channel reference for %s, leaving uncorrected", long_cols)
            continue

        X = short_data[short_cols].mean(axis=1).to_numpy(dtype=float)
        if not np.isfinite(X).all():
            logger.warning("short-channel reference has non-finite values, skipping %s", long_cols)
            continue

        Xc = X - X.mean()
        denom = np.dot(Xc, Xc)
        if denom == 0:
            logger.warning("short-channel reference has zero variance, skipping %s", long_cols)
            continue

        for col in long_cols:
            Y = long_data[col].to_numpy(dtype=float)
            if not np.isfinite(Y).all():
                logger.warning("%s has non-finite values, left uncorrected", col)
                continue
            beta = np.dot(Xc, Y - Y.mean()) / denom
            corrected[col] = Y - beta * Xc

    return corrected
