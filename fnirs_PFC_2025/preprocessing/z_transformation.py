import logging

import numpy as np

logger = logging.getLogger(__name__)


def z_transformation(df, cols):
    """Z-score each column over the whole recording (population std, finite samples only)."""
    out = df.copy()
    for col in cols:
        x = out[col].to_numpy(dtype=float, copy=True)
        ok = np.isfinite(x)
        if ok.sum() < 10 or x[ok].std() <= 1e-10:
            logger.warning("z-score skipped for %s (too few samples or flat)", col)
            continue
        x[ok] = (x[ok] - x[ok].mean()) / x[ok].std()
        out[col] = x
    return out
