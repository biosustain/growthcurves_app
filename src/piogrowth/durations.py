"""Operate on boolean series with a timestamp index."""

import pandas as pd


def get_first_idxmax(s: pd.Series):
    """Get idx max in Boolean Series"""
    return s.idxmax() if s.any() else pd.NaT


def get_last_true_index(s: pd.Series):
    """Get the last index where the Series is True.

    Parameters
    ----------
    s : pd.Series
        A boolean Series.

    Returns
    -------
    pd.Timestamp
        The last index where the Series is True, or pd.NaT if none exists.
    """
    return s[s].index[-1] if s.any() else pd.NaT


def find_max_range(s: pd.Series):
    """Find the maximum range of consecutive True values in a boolean Series.

    Parameters
    ----------
    s : pd.Series
        A boolean Series.

    Returns
    -------
    tuple
        A tuple containing the start and end indices of the maximum range of consecutive
        True values. If no True values are found, returns (pd.NaT, pd.NaT).
    """
    s_min = get_first_idxmax(s)
    s_max = get_last_true_index(s)
    if s_min is not pd.NaT and s_max is not pd.NaT:
        duration = s_max - s_min
        continues = s[s_min:s_max].all()
    else:
        duration = pd.NA
        continues = pd.NA
    return pd.Series(
        [s_min, s_max, duration, continues],
        index=["start", "end", "duration", "is_continues"],
    )
