import pandas as pd

import growthcurve_app


def build_three_row_header(columns):
    """Take sample names from columns, and add replicate and concentration
    rows with default values of 1 and 0, respectively."""

    header = pd.MultiIndex.from_arrays(
        [list(columns), [1] * len(columns), [0] * len(columns)],
        names=["sample", "replicate", "concentration"],
    )
    return header


def to_qurve_format(df, start_time=None):
    df = df.copy()
    df.columns = build_three_row_header(df.columns)
    df = growthcurve_app.reindex_w_relative_time(df, start_time=start_time)
    df.index.name = ""
    df.columns.names = ["Time (h)", "", ""]

    return df
