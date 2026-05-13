# The __init__.py file is loaded when the package is loaded.
# It is used to indicate that the directory in which it resides is a Python package
import logging
from importlib import metadata

import pandas as pd

logger = logging.getLogger(__name__)

__version__ = metadata.version("growthcurve-app")

# The __all__ variable is a list of variables which are imported
# when a user does "from example import *"
__all__ = [
    "reindex_w_relative_time",
    "convert_to_elapsed_hours",
]


def convert_seconds_to_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an index of seconds to hours."""
    df.index = df.index / 3600.0
    df.index.name = "Elapsed time (hours)"
    return df


def reindex_w_relative_time(
    df: pd.DataFrame,
    start_time: pd.Timestamp = None,
    new_index_name: str = "Elapsed time (hours)",
) -> pd.DataFrame:
    """Reindex the DataFrame to use relative time as the index.
    The index of the DataFrame is expected to be timestamps."""
    df = df.copy()  # needed as reindex as DataFrame is a view
    if start_time is None:
        logger.debug("Start time is None, using minimum timestamp.")
        start_time = df.index.min()
    df.index = convert_to_elapsed_hours(df.index, start_time=start_time)
    df.index.name = new_index_name
    return df


def convert_to_elapsed_hours(
    timestamp: pd.Timestamp, start_time: pd.Timestamp
) -> float:
    """Convert a timestamp to elapsed hours since start_time."""
    ret = (timestamp - start_time).total_seconds() / 3_600
    return ret
