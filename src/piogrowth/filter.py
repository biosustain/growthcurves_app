import numpy as np
import pandas as pd


def out_of_iqr(s: pd.Series, factor: float = 1.5) -> pd.Series:
    """Return a boolean Series indicating whether each value is an outlier based
    on the IQR method."""
    center = s.iloc[len(s) // 2]
    if np.isnan(center):
        return False
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    # center point out of IQR?

    return (center < lower_bound) | (center > upper_bound)
