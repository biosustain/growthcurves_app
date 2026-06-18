import pandas as pd
from scipy.signal import find_peaks


def detect_peaks(
    series: pd.Series,
    distance: int = 300,
    prominence: float = None,
) -> pd.Series:
    """Detect peaks in a pandas Series using scipy's find_peaks function.

    Args:
        series (pd.Series): The input time series data.
        distance (int): Minimum horizontal distance (in number of samples)
                        between neighboring peaks.
        prominence (float): Required prominence of peaks. If None, defaults to
                            one-fifth of the maximum value in the series.

    Returns:
        pd.Series: Detected peaks in the series.
    """
    s = series.dropna()
    if prominence is None:
        prominence = s.max() / 5
    peaks, _ = find_peaks(s, distance=distance, prominence=prominence)
    return s.iloc[peaks]
