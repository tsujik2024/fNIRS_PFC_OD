import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Keyword groups used to detect a column's chromophore type, independent of
# which pipeline's naming convention produced it. A column is oxygenated if
# ANY of these substrings appear in its name, deoxygenated if any of the
# other group's substrings appear.
_OXY_KEYS = ("HbO", "O2Hb", "_oxy")
_DEOXY_KEYS = ("HbR", "HHb", "_deoxy")


def scr_regression(long_data: pd.DataFrame, short_data: pd.DataFrame) -> pd.DataFrame:
    corrected = long_data.copy()

    for type_name, keys in (("oxygenated", _OXY_KEYS), ("deoxygenated", _DEOXY_KEYS)):
        long_cols = [c for c in long_data.columns if any(k in str(c) for k in keys)]
        short_cols = [c for c in short_data.columns if any(k in str(c) for k in keys)]

        if not long_cols:
            continue  # nothing of this chromophore type in long_data

        if not short_cols:
            logger.warning(
                f"SCR: no {type_name} short-channel reference found in short_data "
                f"(columns: {list(short_data.columns)}); leaving {long_cols} uncorrected."
            )
            continue

        # Reference regressor: mean across all matching short columns. If the
        # caller has already pre-averaged (short_data has exactly one column
        # of this type), this is a no-op mean of a single column.
        X = short_data[short_cols].mean(axis=1).to_numpy(dtype="float64")

        if not np.all(np.isfinite(X)):
            logger.warning(
                f"SCR: {type_name} short-channel reference {short_cols} has non-finite "
                f"values; skipping correction for {long_cols}."
            )
            continue

        # --- OLS WITH INTERCEPT: centre the regressor before estimating beta ---
        # A DC offset in X now contributes nothing to the slope, so an
        # absolute-concentration pedestal in the short channel can no longer be
        # scaled up into the corrected long-channel signal.
        X_mean = np.mean(X)
        Xc = X - X_mean
        denom = np.dot(Xc, Xc)  # == variance(X) * N

        if not np.isfinite(denom) or denom == 0.0:
            logger.warning(
                f"SCR: {type_name} short-channel reference {short_cols} has zero "
                f"variance; skipping correction for {long_cols}."
            )
            continue

        for long_col in long_cols:
            Y = long_data[long_col].to_numpy(dtype="float64")
            if not np.all(np.isfinite(Y)):
                logger.warning(f"SCR: {long_col} has non-finite values; leaving uncorrected.")
                continue

            Yc = Y - np.mean(Y)
            beta = np.dot(Xc, Yc) / denom
            # Subtract only the correlated *varying* superficial component.
            # Using the centred regressor (Xc) preserves the long channel's own
            # mean; the short channel's DC is not reintroduced or removed here.
            corrected[long_col] = Y - beta * Xc
            logger.debug(f"SCR: {long_col} corrected using mean({short_cols}) (beta={beta:.4f})")

    return corrected
