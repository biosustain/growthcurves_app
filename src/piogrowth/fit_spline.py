import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from scipy.interpolate import make_splrep, splev

SmoothingRange = namedtuple("SmoothingRange", ["s_min", "s", "s_max"])


def get_smoothing_range(m: int):
    """
    Compute the smoothing range for B-spline fitting in scipy interpolate functionality.
    """
    s_min, s, s_max = int(m - np.sqrt(2 * m)), m, int(m + np.sqrt(2 * m))
    s = SmoothingRange(s_min, s, s_max)
    return s


def fit_spline_and_derivatives(
    s: pd.Series,
    smoothing_factor: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit B-splines to each column in the DataFrame and compute specified derivatives.
    Values cannot be missing as NaNs, i.e. on rolling median of data.

    Parameters
    ----------
    s: pd.Series
        Input Series with time series data
    smoothing_factor: float
        Smoothing factor for the spline fitting.
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the fitted spline
                                 and its derivatives.
    """
    # drop NaN values
    s = s.dropna()

    if len(s) < 4:
        raise ValueError(
            "Not enough data points to fit a spline. Need at least 4 non-NaN values."
        )
    if not is_datetime64_any_dtype(s.index.dtype):
        raise TypeError("Index of the input Series must be datetime type.")
    x = (s.index - s.index[0]).total_seconds().to_numpy() / 3_600  # convert to hours

    bspl = make_splrep(
        x,
        s,
        s=smoothing_factor,
        k=3,
    )
    s_fitted = pd.Series(
        splev(x, bspl),
        index=s.index,
    )

    # for order in derivative_ord_ers:
    der = bspl.derivative(nu=1)
    s_first_derivative = pd.Series(
        der(x),
        index=s.index,
    )

    return s_fitted, s_first_derivative


def fit_spline_and_derivatives_one_batch(
    df: pd.DataFrame,
    smoothing_factor: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit B-splines to each column in the DataFrame and compute specified derivatives.
    Values cannot be missing as NaNs, i.e. on rolling median of data.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame with time series data.
    smoothing_factor: float
        Smoothing factor for the spline fitting.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the fitted spline
                                           and its first derivative.
    """
    assert df.isna().sum().sum() == 0, "Input DataFrame contains NaN values"
    df_fitted = pd.DataFrame(index=df.index)
    df_first_derivative = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col]
        s_fitted, s_first_derivative = fit_spline_and_derivatives(s, smoothing_factor)
        df_fitted[f"{col}"] = s_fitted
        df_first_derivative[f"{col}"] = s_first_derivative

    return df_fitted, df_first_derivative


def fit_splines_to_segments(
    s: pd.Series, peaks: pd.Series, smoothing_factor: float = 100.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit splines to segments of the time series data between detected peaks.

    Parameters
    ----------
    s : pd.Series
        _description_
    peaks : pd.Series
        _description_
    smoothing_factor : float, optional
        _description_, by default 100.0

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        _description_
    """
    peak_timepoints = [s.index.min(), *peaks.dropna().index, s.index.max()]
    res_fitted, res_derivative, res_max, res_idx_max = [], [], [], []
    for start, end in itertools.pairwise(peak_timepoints):
        s_segment = s[start:end]
        if len(s_segment) < 4:
            continue
        s_segment_fitted, s_segment_derivative = fit_spline_and_derivatives(
            s_segment, smoothing_factor=smoothing_factor
        )
        res_fitted.append(s_segment_fitted)
        res_derivative.append(s_segment_derivative)
        idx_max = s_segment_derivative.idxmax()
        res_max.append(s_segment.loc[idx_max])
        res_idx_max.append(idx_max)

    res_fitted = pd.concat(res_fitted).sort_index()
    res_fitted = res_fitted.loc[~res_fitted.index.duplicated(keep="first")]
    res_derivative = pd.concat(res_derivative).sort_index()
    res_derivative = res_derivative.loc[~res_derivative.index.duplicated(keep="first")]
    res_max = pd.Series(res_max, index=res_idx_max).sort_index()
    return res_fitted, res_derivative, res_max


def fit_growth_data_w_peaks(
    df_wide: pd.DataFrame,
    peaks: pd.DataFrame,
    smoothing_factor: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit growth data with splines between detected peaks."""
    df_fitted = pd.DataFrame(index=df_wide.index)
    df_first_derivative = pd.DataFrame(index=df_wide.index)
    df_max = {}

    for col in df_wide.columns:
        s = df_wide[col].dropna()
        s_peaks = peaks[col].dropna()
        s_fitted, s_derivative, s_max = fit_splines_to_segments(
            s, s_peaks, smoothing_factor=smoothing_factor
        )
        df_fitted[col] = s_fitted
        df_first_derivative[col] = s_derivative
        df_max[col] = s_max

    return df_fitted, df_first_derivative, df_max
