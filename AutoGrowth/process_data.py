from pathlib import Path

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = {
    "PioReactor": ("timestamp_localtime", "pioreactor_unit", "od_reading"),
    "Chi.Bio": ("exp_time", "od_measured"),
}

PIO_PIVOT_COLUMNS = ("timestamp_rounded", "pioreactor_unit", "od_reading")

REQUIRED_COLUMNS_NAME_MAP = {
    "PioReactor": {
        # "timestamp_localtime": "timestamp",
        "pioreactor_unit": "reactor",
        "od_reading": "od_reading",
    },
    "Chi.Bio": {
        "exp_time": "elapsed_time",
        "reactor": "reactor",  # is added by processing function
        "od_measured": "od_reading",
    },
}


# specify datecolumns for now
COLUMN_TYPES_PIO: dict = {
    # need to be callable
    "timestamp_localtime": pd.Timestamp,
    #  'experiment': str,
    #  'pioreactor_unit': str,
    "timestamp": pd.Timestamp,
    # "od_reading": float,
    # "angle": float,
    # "channel": float,
}


def read_pioreactor_csv(file: str, round_time: int = 60):
    """Read raw OD data from a PioReactor export CSV file and round timestamps."""
    df_raw_od_data = pd.read_csv(file, converters=COLUMN_TYPES_PIO).convert_dtypes()

    # ! add check that required columns are in data and have correct dtypes (pandera)
    msg = (
        f"- Loaded {df_raw_od_data.shape[0]:,d} rows "
        f"and {df_raw_od_data.shape[1]:,d} columns.\n"
    )
    # round timestamp data
    df_raw_od_data.insert(
        0,
        "timestamp_rounded",
        df_raw_od_data["timestamp_localtime"].dt.round(
            f"{round_time}s",
        ),
    )
    return df_raw_od_data, msg


def read_chibio_csv(files: list[Path], round_time: int = 60) -> pd.DataFrame:
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["reactor"] = file.name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # elapsed time in seconds is rounded
    df["elapsed_time_in_seconds"] = (
        df["exp_time"] / round_time
    ).dropna().round().astype(int) * round_time
    msg = f"- Loaded {df.shape[0]:,d} rows " f"and {df.shape[1]:,d} columns.\n"
    return df, msg


def drop_na_pioreactor_raw_od_data(
    df_raw_od_data, subset=("timestamp_rounded", "pioreactor_unit", "od_reading")
):
    """Drop rows with NA values in core columns and return the number of dropped
    rows.
    """
    N_before = df_raw_od_data.shape[0]
    df_raw_od_data = df_raw_od_data.dropna(subset=subset)
    N_after = df_raw_od_data.shape[0]
    N_dropped = N_before - N_after
    return df_raw_od_data, N_dropped


def process_chibio_data(
    files: list[Path],
    round_time: int = 60,
    keep_core_data: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df, msg = read_chibio_csv(files, round_time)
    df, N_dropped = drop_na_pioreactor_raw_od_data(
        df, subset=REQUIRED_COLUMNS["Chi.Bio"]
    )
    msg += f"- Dropped {N_dropped:,d} rows with NA values.\n"
    if keep_core_data:
        # core data and added reactor column only
        df = df[
            [
                "exp_time",
                "elapsed_time_in_seconds",
                "reactor",
                "od_measured",
            ]
        ]
    df_wide = df.pivot(
        index="elapsed_time_in_seconds", columns="reactor", values="od_measured"
    )
    return df, df_wide, msg


def process_od_pioreactor(
    file: str,
    round_time: int = 60,
    keep_core_data: bool = True,
    aggregate_duplicated_rounded_timepoint: bool = True,
    aggregate_duplicated_rounded_timepoint_method: str = "mean",
):
    """Process raw OD data from a PioReactor export CSV file and return both the
    raw and wide formats of the data, along with a summary message and a boolean
    indicating whether this is the first time processing (i.e. whether to trigger a
    re-run of the app to update with the new data). The raw data is processed by
    rounding timestamps, adding elapsed time in seconds, and optionally keeping only
    core data columns. The wide data is processed by pivoting the raw data to have
    timestamp_rounded as the index and pioreactor_unit as columns, with od_reading as
    values. If rounding produces duplicated timepoints in reactors, the duplicates
    are optionally aggregated using the specified method.

    Parameters
    ----------
    file : str
        PioReactor export CSV file containing raw OD data.
    round_time : int, optional
        Time in seconds to round timestamps, by default 60
    keep_core_data : bool, optional
        Whether to keep only core data columns, by default True
    aggregate_duplicated_rounded_timepoint : bool, optional
        Whether to aggregate duplicated rounded timepoints, by default True
    aggregate_duplicated_rounded_timepoint_method : str, optional
        Method to use for aggregating duplicated rounded timepoints, by default "mean".
        Options are what pandas groupby.agg accepts, e.g. "mean", "median",
        "min", "max", etc.

    Returns
    -------
    pd.DataFrame, pd.DataFrame, str
        The processed raw OD data, the processed wide OD data, a summary message of
        the processing steps.
    """
    df_raw_od_data, msg = read_pioreactor_csv(file, round_time)
    # use starttime to compute elapsed time
    start_time = df_raw_od_data["timestamp_rounded"].min()
    st.session_state["start_time"] = start_time
    df_raw_od_data["elapsed_time_in_seconds"] = (
        df_raw_od_data["timestamp_rounded"] - start_time
    ).dt.total_seconds()
    msg += f"- Added elapsed time in seconds since start ({start_time}).\n"
    st.session_state["round_time"] = round_time
    # only keep core data?
    if keep_core_data:
        try:
            df_raw_od_data = df_raw_od_data[
                [
                    "timestamp_rounded",
                    "timestamp_localtime",
                    "elapsed_time_in_seconds",
                    "pioreactor_unit",
                    "od_reading",
                ]
            ]
            msg += "- Kept only core data columns.\n"
        except KeyError:
            st.error(
                "Could not keep only core data columns. "
                "Please check that the uploaded file contains "
                "the required columns: "
                "timestamp_localtime, pioreactor_unit, od_reading."
            )
            st.stop()
    st.session_state["df_raw_od_data"] = df_raw_od_data
    # re-run now with data set

    msg += f"- Wide OD data with rounded timestamps to {round_time} seconds.\n"
    # wide data of raw data
    # - can be used in plot for visualization,
    # - and in curve fitting (where gaps would be interpolated)
    df_raw_od_data, N_dropped = drop_na_pioreactor_raw_od_data(df_raw_od_data)
    if N_dropped > 0:
        msg += (
            f"- Dropped {N_dropped:,d} rows with missing values in core columns "
            "(timestamp_rounded, pioreactor_unit, od_reading).\n"
        )
    try:
        df_wide_raw_od_data = df_raw_od_data.pivot(
            index="timestamp_rounded",
            columns="pioreactor_unit",
            values="od_reading",
        )
    except ValueError as e:
        st.error(
            "Rounding produced duplicated timepoints in reactors; "
            f"consider decreasing the rounding time below {round_time} seconds."
        )
        if not aggregate_duplicated_rounded_timepoint:
            # Clear potentially stale wide/derived data before stopping to avoid
            # inconsistencies with the current df_raw_od_data.
            st.session_state["df_wide_raw_od_data"] = None
            st.session_state["df_wide_raw_od_data_filtered"] = None
            st.info(
                "Consider aggregating duplicated timepoints if you do not "
                "want to decrease the rounding time."
            )
            with st.expander("Show error details"):
                st.write(e)
                st.write(df_raw_od_data)
            st.stop()
        st.warning(
            "Aggregating duplicated timepoint using "
            f"the {aggregate_duplicated_rounded_timepoint_method}."
        )

        df_wide_raw_od_data = (
            df_raw_od_data.groupby(
                ["timestamp_rounded", "pioreactor_unit"], sort=False
            )["od_reading"]
            .agg(aggregate_duplicated_rounded_timepoint_method)
            .reset_index()
        )
        df_wide_raw_od_data = df_wide_raw_od_data.pivot(
            index="timestamp_rounded",
            columns="pioreactor_unit",
            values="od_reading",
        )
    df_raw_od_data = df_raw_od_data.rename(
        columns=REQUIRED_COLUMNS_NAME_MAP["PioReactor"]
    )
    return (df_raw_od_data, df_wide_raw_od_data, msg)
