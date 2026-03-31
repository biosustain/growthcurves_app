"""Load PioGrowth data from CSV files."""

import pandas as pd

# specify datecolumns for now
COLUMN_TYPES: dict = {
    # need to be callable
    "timestamp_localtime": pd.Timestamp,
    #  'experiment': str,
    #  'pioreactor_unit': str,
    "timestamp": pd.Timestamp,
    # "od_reading": float,
    # "angle": float,
    # "channel": float,
}


def read_csv(file: str) -> pd.DataFrame:
    """Read a CSV file processed with PioGrowth reactor software."""
    return pd.read_csv(file, converters=COLUMN_TYPES).convert_dtypes()
