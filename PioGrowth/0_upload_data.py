import io
import pickle

import growthcurves as gc
import pandas as pd
import streamlit as st
from ui_components import page_header_with_help

import piogrowth

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")
df_wide_raw_od_data_filtered = st.session_state.get("df_wide_raw_od_data_filtered")
min_periods = st.session_state.get("min_periods", 5)
# use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
# initialize
st.session_state.setdefault("USE_ELAPSED_TIME_FOR_PLOTS", True)

UPLOAD_HELP = """
This page loads and preprocesses a single PioReactor OD dataset.

Use this order:
1. Upload the OD data file
2. (Optional) Upload calibration/turbidostat metadata files
3. Configure and apply preprocessing options
4. Review tables and plots on the Data Dashboard page
5. Use the Downloads page for exports
"""

page_header_with_help("Upload Data", UPLOAD_HELP)


def callback_clear_raw_data():
    st.session_state["df_raw_od_data"] = None
    st.session_state["df_wide_raw_od_data"] = None
    st.session_state["df_wide_raw_od_data_filtered"] = None
    st.session_state["masked"] = None
    st.session_state["upload_processing_summary_msg"] = None
    # reset time windows axis and data
    if "min_date" in st.session_state:
        del st.session_state["min_date"]
    if "max_date" in st.session_state:
        del st.session_state["max_date"]


def restore_session_state_from_zip(zip_bytes: bytes) -> list[str]:
    """Restore session state from a session state ZIP. Returns a list of warning strings."""
    import zipfile

    warnings = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        if "session_state.pkl" not in zf.namelist():
            warnings.append("session_state.pkl not found in ZIP — nothing restored.")
            return warnings
        with zf.open("session_state.pkl") as f:
            state_dict = pickle.loads(f.read())  # noqa: S301
    for key, val in state_dict.items():
        st.session_state[key] = val
    return warnings


def apply_linear_adjustments(
    df_rolling: pd.DataFrame, adjustment_table: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    required_columns = {"reactor", "od"}
    missing_columns = required_columns - set(adjustment_table.columns)
    if missing_columns:
        return df_rolling, [
            "Adjustment table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        ]

    warnings = []
    adjusted = df_rolling.copy()
    table = adjustment_table.loc[:, ["reactor", "od"]].dropna(subset=["reactor"])

    for reactor, group in table.groupby("reactor", sort=False):
        if reactor not in adjusted.columns:
            warnings.append(f"Reactor '{reactor}' not found in df_rolling columns.")
            continue
        od_values = group["od"].dropna().tolist()
        if len(od_values) < 2:
            warnings.append(
                f"Reactor '{reactor}' has fewer than two OD values in adjustment table."
            )
            continue

        target_start = od_values[0]
        target_end = od_values[-1]
        series = adjusted[reactor].dropna()
        if series.empty:
            warnings.append(f"Reactor '{reactor}' has no data in df_rolling.")
            continue

        original_start = series.iloc[0]
        original_end = series.iloc[-1]
        if original_end == original_start:
            warnings.append(
                f"Reactor '{reactor}' has identical start and end values in df_rolling."
            )
            continue

        slope = (target_end - target_start) / (original_end - original_start)
        intercept = target_start - slope * original_start
        adjusted[reactor] = adjusted[reactor] * slope + intercept

    return adjusted, warnings


########################################################################################
# Session State Restore
with st.container(border=True):
    st.header("Restore Previous Session (Optional)")
    st.caption(
        "Upload a session state ZIP downloaded from the **Downloads** page "
        "to restore the app exactly as it was — including all data, "
        "preprocessing settings, and analysis results."
    )
    session_zip_upload = st.file_uploader(
        "Upload session state ZIP",
        type=["zip"],
        key="session_state_zip_upload",
    )
    if session_zip_upload is not None:
        restore_btn = st.button(
            "Restore session from ZIP",
            key="restore_session_state_btn",
            type="primary",
        )
        if restore_btn:
            with st.spinner("Restoring session state...", show_time=True):
                restore_warnings = restore_session_state_from_zip(
                    session_zip_upload.getvalue()
                )
            for w in restore_warnings:
                st.warning(w)
            st.success("Session state restored successfully.")
            st.rerun()

########################################################################################
# Step 1: Upload File with OD/bioscatter data
with st.container(border=True):
    # header and example data file with requirements in popover
    header_col, req_col = st.columns([4, 1], vertical_alignment="center")
    with header_col:
        st.header("Step 1. Upload PioReactor OD Data")
    with req_col:
        # Help message
        with st.popover("Requirements", width="stretch"):
            st.markdown("**Expected structure:**")
            st.markdown("- CSV/TXT file readable by `pandas.read_csv`")
            st.markdown(
                "- Required columns: `timestamp_localtime`, `pioreactor_unit`, `od_reading`"
            )
            st.markdown("- One row per measurement")
            st.markdown("\n > Export from PioReactor WebApp or CLI.")
            st.divider()
            st.markdown("**Example file:**")
            example_data = pd.read_csv(
                "data/batch_example/example_batch_data_od_readings.csv"
            )
            st.dataframe(example_data.head(10), hide_index=True, width="stretch")
            st.download_button(
                label="Download example CSV for App testing",
                data=example_data.to_csv(index=False),
                file_name="example_batch_data_od_readings.csv",
                key="download_example_csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
    # File Uploading of main data file
    st.markdown("**Main OD Data**")
    _file_name = st.session_state.get("file_od_upload_name")
    if _file_name is not None:
        st.info(f"File previously uploaded: {_file_name}")
    file = st.file_uploader(
        "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
        type=["csv", "txt"],
        on_change=callback_clear_raw_data,
    )
    if file is not None:
        # st.session_state["file_od_upload_bytes"] = file.getvalue()
        st.session_state["file_od_upload_name"] = file.name
    main_options_cols = st.columns([3, 2], gap="medium")
    with main_options_cols[0]:
        keep_core_data = st.checkbox(
            "Keep only core data columns (timestamp, pioreactor_unit, od_reading)?",
            value=True,
            help="If checked, only the essential columns are kept from the uploaded file.",
        )
    with main_options_cols[1]:
        custom_id = st.text_input(
            "Custom ID for data",
            max_chars=30,
            value=custom_id,
        )

    if file is None:
        if df_raw_od_data is None:
            st.warning("No data uploaded.")
            st.info("Upload a comma-separated (`.csv`) file to get started.")

# Step 2: Optional metadata uploads
with st.container(border=True):
    header_col, req_col = st.columns([4, 1], vertical_alignment="center")
    header_col.header("Step 2. Optional metadata uploads")

    # show both optional uploads
    optional_upload_cols = st.columns([2, 3], gap="small")
    with optional_upload_cols[0]:
        st.markdown("**OD Calibration Table**")
        # help message
        with st.popover("See an Example", width="stretch"):
            st.markdown("**OD Calibration Table**")
            st.markdown(
                "- CSV file with columns `reactor` and `od`.\n"
                "- Used to adjust OD readings by reactor based on calibration data."
            )
            st.divider()
            st.markdown("**Example:**")
            fname = "data/batch_example/example_batch_data_od_readings_calibration.csv"
            example_data = pd.read_csv(fname)
            st.dataframe(example_data, hide_index=True, width="stretch")
            st.download_button(
                label="Download example calibration CSV.",
                data=example_data.to_csv(index=False),
                file_name="example_batch_data_od_readings_calibration.csv",
                key="download_example_csv_calibration",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        # file uploading
        _file_name = st.session_state.get("od_adjustment_upload_name")
        if _file_name is not None:
            st.info(f"File previously uploaded: {_file_name}")
        od_adjustment_upload = st.file_uploader(
            "OD adjustment table",
            type=["csv", "txt"],
            key="upload_page_od_adjustment_table",
        )
    with optional_upload_cols[1]:
        st.markdown("**Turbidostat Metadata**")
        # help message
        with st.popover("See an Example", width="stretch"):
            st.markdown("**Turbidostat Metadata**")
            st.markdown("""
                If provided, peaks are not autodetected.
                
                - CSV file with columns `timestamp_localtime`, `pioreactor_unit`,
                `event_name` and `message` and `data`.

                - Used to parse `DilutionEvents` for turbidostat analysis 
                  based on event descriptions in the metadata. 
                  If not provided, peaks will be autodetected based on OD data.
                """)

            st.markdown("\n > Export from PioReactor WebApp or CLI.")
            st.divider()
            st.markdown("**Example:**")
            fname = (
                "data/turbidostat_example/example_2-Pio_Experiment_dilution_events.csv"
            )
            example_data = pd.read_csv(fname)
            st.dataframe(example_data, hide_index=True, width="stretch")
            st.download_button(
                label="Download example dilution events CSV.",
                data=example_data.to_csv(index=False),
                file_name="example_2-Pio_Experiment_dilution_events.csv",
                key="download_example_csv_dilution",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        _file_name = st.session_state.get("turbidostat_meta_upload_name")
        if _file_name is not None:
            st.info(f"File previously uploaded: {_file_name}")
        turbidostat_meta_upload = st.file_uploader(
            "Dilution metadata (for Turbidostat page)",
            type=["csv"],
            key="upload_page_turbidostat_meta",
        )
        st.session_state.setdefault("turbidostat_timestamp_col", "timestamp_localtime")
        st.session_state.setdefault("turbidostat_reactor_col", "pioreactor_unit")
        st.session_state.setdefault("turbidostat_message_col", "message")
        timestamp_options = ["timestamp", "timestamp_localtime"]
        if st.session_state.get("turbidostat_timestamp_col") not in timestamp_options:
            st.session_state["turbidostat_timestamp_col"] = "timestamp_localtime"
        turbi_cols = st.columns(3, gap="small")
        with turbi_cols[0]:
            st.selectbox(
                "Select timestamp column",
                options=timestamp_options,
                key="turbidostat_timestamp_col",
            )
        with turbi_cols[1]:
            st.text_input(
                "Select column with reactor information",
                key="turbidostat_reactor_col",
            )
        with turbi_cols[2]:
            st.text_input(
                "Select column with event description",
                key="turbidostat_message_col",
            )
    # Save bytes and file name to session state for later processing
    if od_adjustment_upload is not None:
        st.session_state["od_adjustment_upload_bytes"] = od_adjustment_upload.getvalue()
        st.session_state["od_adjustment_upload_name"] = od_adjustment_upload.name
    if turbidostat_meta_upload is not None:
        st.session_state["turbidostat_meta_upload_bytes"] = (
            turbidostat_meta_upload.getvalue()
        )
        st.session_state["turbidostat_meta_upload_name"] = turbidostat_meta_upload.name

# Step 3: Configure preprocessing options
### Form ##############################################################################
with st.container(border=True):
    st.header("Step 3. Configure Processing Options")
    st.warning(
        'Options are only saved if you press "Apply options to uploaded data" button at the end of this section.'
    )

    with st.form("Upload_data_form", clear_on_submit=False):
        st.write("#### Data filtering options:")
        if st.session_state.get("df_raw_od_data") is None:
            available_reactors = []
            reactors_selected = st.multiselect(
                "Select reactors to include in analysis",
                options=available_reactors,
                default=available_reactors,
                help="Upload OD data to populate available reactors.",
            )
        else:
            available_reactors = sorted(
                df_raw_od_data["pioreactor_unit"].dropna().astype(str).unique().tolist()
            )
            reactors_selected = st.multiselect(
                "Select reactors to include in analysis",
                options=available_reactors,
                default=available_reactors,
                help=(
                    "All reactors are selected by default. Remove any reactors you do  "
                    "not want analyzed."
                ),
            )
        filter_columns = st.columns(2)
        with filter_columns[0]:
            negative_options = [
                "Set negative OD readings to missing (NaN)",
                "Impute negative values by moving average",
            ]
            default_negative = st.session_state.get(
                "negative_handling", negative_options[1]
            )
            try:
                default_negative_index = negative_options.index(default_negative)
            except ValueError:
                default_negative_index = 1
            negative_handling = st.radio(
                "How should negative OD readings be handled?",
                options=negative_options,
                index=default_negative_index,
                help=(
                    "Negative values distort curve fitting. Choose whether to convert "
                    "them to missing values or impute them."
                ),
            )
            remove_negative = True
            if negative_handling == "Impute negative values by moving average":
                remove_negative = False
            fill_na = st.checkbox(
                "Impute missing bioscatter readings using forward and backward filling",
                help=(
                    "If checked, missing values will be "
                    "imputed using forward fill and backward fill. This is recommended "
                    "if you expect only a few missing or negative values that are "
                    "likely due to measurement errors.  Note that this will include "
                    "negative zeros which were previously removed using the above "
                    "option."
                ),
                value=st.session_state.get("fill_na", False),
            )
            # ! move to after smoothing is applied?
            remove_downward_trending = st.checkbox(
                label="Remove downward trending data points (negative OD changes) "
                " globally after smoothing the data.",
                value=st.session_state.get("remove_downward_trending", False),
                help=(
                    "This can be used to remove data points that are smaller than a "
                    "previous one. Downward trends will be removed, but the upward "
                    "trend will be kept from a local minimum."
                ),
            )
            remove_max = st.checkbox(
                "Remove maximum OD readings by quantile",
                value=st.session_state.get("remove_max", False),
            )
            _outlier_options = ["None", "IQR", "ECOD"]
            _outlier_default = st.session_state.get("outlier_method", "None")
            if _outlier_default not in _outlier_options:
                _outlier_default = "None"
            outlier_method = st.selectbox(
                "Outlier detection method",
                options=_outlier_options,
                index=_outlier_options.index(_outlier_default),
                help=(
                    "- `None`: no outlier removal.\n"
                    "- `IQR`: remove outliers using Inter-Quartile Range in a rolling "
                    "window of timepoints.\n"
                    "- `ECOD`: remove outliers using the ECOD algorithm "
                    "(Empirical Cumulative distribution-based Outlier Detection)."
                ),
            )
        with filter_columns[1]:
            quantile_max = st.slider(
                "Max quantile for maximum removal",
                0.9,
                1.0,
                st.session_state.get("quantile_max", 0.99),
                step=0.01,
            )
            iqr_range_value = st.slider(
                "IQR factor for outlier removal",
                1.0,
                3.0,
                st.session_state.get("iqr_range_value", 1.5),
                step=0.1,
                help="Used when outlier method is IQR. Multiplier of the IQR.",
            )
            rolling_window = st.slider(
                "Rolling window (of timepoints) for IQR outlier removal",
                11,
                61,
                st.session_state.get("rolling_window", 21),
                step=2,
                help="Used when outlier method is IQR.",
            )
            ecod_factor = st.slider(
                "ECOD factor for outlier removal",
                0.5,
                8.0,
                st.session_state.get("ecod_factor", 4.0),
                step=0.1,
                help="Used when outlier method is ECOD. Anomaly detection sensitivity.",
            )

        st.divider()
        st.write("""
            #### Time and aggregation options:\n

            - round timepoints and handle duplicates due to rounding

            Time window selection has moved to the **Data Dashboard** page.
            """)
        rounding_columns = st.columns(
            [4, 2, 2], gap="large", vertical_alignment="bottom"
        )
        with rounding_columns[0]:
            round_time = st.slider(
                "Round time to nearest second (defining timesteps). "
                "Used to align timeseries "
                "with slight time offsets.",
                1,
                300,
                st.session_state.get("round_time", 5),
                step=1,
                help=(
                    "Rounding helps pivot the data to wide format from the "
                    "long format. If you have multiple measurements for the same "
                    "reactor within the rounded time window, they will be "
                    "aggregated by median or mean (see option to the right) "
                    "if aggregation is enabled."
                ),
            )
        with rounding_columns[1]:
            _value = st.session_state.get(
                "aggregate_duplicated_rounded_timepoint", True
            )
            aggregate_duplicated_rounded_timepoint = st.checkbox(
                "Aggregate data by rounded timepoints?",
                value=_value,
                disabled=False,
                help=(
                    "If checked, multiple measurements for the same reactor within "
                    "the rounded time window will be aggregated by median or mean. "
                    "If not checked, the app will attempt to pivot the data to "
                    "wide format without aggregation, which will lead to errors if "
                    "there are duplicated timepoints after rounding."
                ),
            )
        with rounding_columns[2]:
            _value = st.session_state.get(
                "aggregate_duplicated_rounded_timepoint_method", "median"
            )

            aggregate_duplicated_rounded_timepoint_method = st.radio(
                "If aggregating, which method to use for replicates in same timepoint?",
                options=["median", "mean"],
                index=0 if _value == "median" else 1,
                help=(
                    "If there are multiple measurements for the same reactor within "
                    "the rounded time window, this option determines how they will be "
                    "aggregated when pivoting to wide format. Median is more robust to "
                    "outliers and therefore the default."
                ),
            )
        st.divider()
        button_pressed = st.form_submit_button(
            "Apply options to uploaded data", type="primary", width="stretch"
        )

### save form state
# remember form values for next time page is opened
st.session_state["keep_core_data"] = keep_core_data
st.session_state["custom_id"] = custom_id
st.session_state["reactors_selected"] = reactors_selected
st.session_state["remove_negative"] = remove_negative
st.session_state["negative_handling"] = negative_handling
st.session_state["fill_na"] = fill_na
st.session_state["remove_downward_trending"] = remove_downward_trending
st.session_state["remove_max"] = remove_max
st.session_state["outlier_method"] = outlier_method
st.session_state["quantile_max"] = quantile_max
st.session_state["iqr_range_value"] = iqr_range_value
st.session_state["rolling_window"] = rolling_window
st.session_state["ecod_factor"] = ecod_factor
st.session_state["round_time"] = round_time
st.session_state["aggregate_duplicated_rounded_timepoint"] = (
    aggregate_duplicated_rounded_timepoint
)
st.session_state["aggregate_duplicated_rounded_timepoint_method"] = (
    aggregate_duplicated_rounded_timepoint_method
)

########################################################################################
# Process data

extra_warn = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id

if button_pressed and file is None and df_raw_od_data is None:
    extra_warn.warning("No data uploaded.")
    st.stop()

msg = ""

# File Uploaded ########################################################################
# this runs wheather the button is pressed or not, but only if a file is uploaded
if file is not None:
    df_raw_od_data = piogrowth.load.read_csv(file)

    # ! add check that required columns are in data and have correct dtypes (pandera)
    msg = (
        f"- Loaded {df_raw_od_data.shape[0]:,d} rows "
        f"and {df_raw_od_data.shape[1]:,d} columns.\n"
    )
    # round timestamp data
    # ! 'timestamp_localtime' must be in data (note down requirement)
    df_raw_od_data.insert(
        0,
        "timestamp_rounded",
        df_raw_od_data["timestamp_localtime"].dt.round(
            f"{round_time}s",
        ),
    )
    # use starttime to compute elapsed time
    start_time = df_raw_od_data["timestamp_rounded"].min()
    st.session_state["start_time"] = start_time
    df_raw_od_data["elapsed_time_in_seconds"] = (
        df_raw_od_data["timestamp_rounded"] - start_time
    ).dt.total_seconds()
    msg += f"- Added elapsed time in seconds since start ({start_time}).\n"
    st.session_state["round_time"] = round_time
    rerun = st.session_state.get("df_raw_od_data") is None
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
                "the required columns: timestamp_localtime, pioreactor_unit, od_reading."
            )
            st.stop()
    st.session_state["df_raw_od_data"] = df_raw_od_data
    # re-run now with data set

    msg += f"- Wide OD data with rounded timestamps to {round_time} seconds.\n"
    # wide data of raw data
    # - can be used in plot for visualization,
    # - and in curve fitting (where gaps would be interpolated)
    N_before = df_raw_od_data.shape[0]
    df_raw_od_data = df_raw_od_data.dropna(
        subset=["timestamp_rounded", "pioreactor_unit", "od_reading"]
    )
    N_after = df_raw_od_data.shape[0]
    N_dropped = N_before - N_after
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
    st.session_state["df_wide_raw_od_data"] = df_wide_raw_od_data
    st.session_state["upload_processing_summary_msg"] = msg
    if rerun:
        # ? replace with callback function that creates the input form?
        st.rerun()

### Apply option from form #############################################################
if button_pressed:
    # Keep only reactors selected for analysis
    if not reactors_selected:
        st.warning("No reactors selected. Select at least one reactor to continue.")
        st.stop()
    st.write(f"Reactors included in analysis: {reactors_selected}")
    df_raw_od_data = df_raw_od_data.loc[
        df_raw_od_data["pioreactor_unit"].astype(str).isin(reactors_selected)
    ]

    # initalize masked here
    masked = pd.DataFrame(
        False,
        index=df_wide_raw_od_data.index,
        columns=df_wide_raw_od_data.columns,
    )
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.copy()

    #### Apply Data Filtering options ##################################################
    # Handle negative values
    n_negative = (df_wide_raw_od_data_filtered < 0).sum().sum()
    if n_negative > 0:
        st.warning(f"Found {n_negative:,d} negative OD readings.")
        msg += f"- Found {n_negative:,d} negative OD readings.\n"
    if remove_negative:
        mask_negative = df_wide_raw_od_data_filtered < 0
        msg += (
            f"- Setting {mask_negative.sum().sum():,d} negative OD readings to NaN.\n"
        )
        msg += f"   - in detail: {mask_negative.sum().to_dict()}\n"
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(mask_negative)
        masked = masked | mask_negative
    else:
        mask_negative = df_wide_raw_od_data_filtered < 0
        window = 31
        # Replace negatives with NaN,
        # then compute centered rolling mean over non-missing values
        temp = df_wide_raw_od_data_filtered.mask(mask_negative)
        rolling_mean = temp.rolling(window=window, min_periods=1, center=True).mean()
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
            mask_negative, rolling_mean
        )
        n_imputed = mask_negative.sum().sum()
        msg += (
            f"- Imputed {n_imputed:,d} negative OD readings using"
            f" centered rolling mean (window={window}).\n"
        )
        msg += f"   - in detail: {mask_negative.sum().to_dict()}\n"
        masked = masked | mask_negative
        del temp, rolling_mean, mask_negative
    if fill_na:
        mask_na = df_wide_raw_od_data_filtered.isna()
        msg += f"- Filling {mask_na.sum().sum():,d} missing OD readings.\n"
        msg += f"   - in detail: {mask_na.sum().to_dict()}\n"
        # ! should I visualize the values differently?
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.fillna(
            method="ffill"
        ).fillna(method="bfill")

    # remove quantiles
    if remove_max:
        mask_extreme_values = (
            df_wide_raw_od_data_filtered
            > df_wide_raw_od_data_filtered.quantile(quantile_max)
        )
        msg += (
            f"- Number of extreme values detected: {mask_extreme_values.sum().sum()}\n"
        )
        msg += f"   - in detail: {mask_extreme_values.sum().to_dict()}\n"
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
            mask_extreme_values
        )
        masked = masked | mask_extreme_values

    # outlier detection using IQR on rolling window: sets for center value of window a
    # true or false (this would be arguing maybe for long data format)
    # can be used in plot for visualization
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    print(f"Applying outlier method: {outlier_method}")
    if outlier_method in ("IQR", "ECOD"):
        kwargs = (
            {"method": "iqr", "factor": iqr_range_value, "window_size": rolling_window}
            if outlier_method == "IQR"
            else {"method": "ecod", "factor": ecod_factor}
        )
        # ! not robust to missing values yet.
        if (
            df_wide_raw_od_data_filtered.isna().sum().sum() > 0
            and outlier_method == "ECOD"
        ):
            st.error(
                "Found missing values in the data. ECOD outlier detection does not work"
                " with missing values. Consider setting forward and backward filling "
                " before applying ECOD outlier detection."
            )
            st.stop()
        with st.spinner(f"Applying {outlier_method} outlier removal..."):
            mask_outliers = df_wide_raw_od_data_filtered.apply(
                gc.preprocessing.detect_outliers, raw=False, **kwargs
            ).astype(bool)
            n_out = mask_outliers.sum().sum()
            msg += f"- Number of outliers detected ({outlier_method}): {n_out}\n"
            msg += f"   - in detail: {mask_outliers.sum().to_dict()}\n"
            masked = masked | mask_outliers
            df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
                mask_outliers
            )

    masked = masked.convert_dtypes()

    st.session_state["df_wide_raw_od_data_filtered"] = df_wide_raw_od_data_filtered
    st.session_state["masked"] = masked

    df_rolling = (
        df_wide_raw_od_data_filtered.rolling(
            rolling_window,
            min_periods=min_periods,
            center=True,
        )
        .median()
        .sort_index()
    )

    if remove_downward_trending:
        # Remove downward trending data globally on averaged data
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
            df_wide_raw_od_data_filtered.diff().le(0)
        )
        msg += (
            "- Downward trending data points (negative OD changes) were "
            "removed globally."
        )

    # ? Should it not be possible to be run twice in a single session?
    if od_adjustment_upload is not None:
        if st.session_state.get("is_df_rolling_adjusted"):
            st.warning(
                "OD adjustments have already been applied. "
                "Re-applying will overwrite previous adjustments."
            )
        df_adjustments = pd.read_csv(od_adjustment_upload).convert_dtypes()
        df_rolling, adjustment_warnings = apply_linear_adjustments(
            df_rolling, df_adjustments
        )
        st.session_state["is_df_rolling_adjusted"] = True
        st.session_state["df_rolling"] = df_rolling
        st.session_state["df_od_adjustment"] = df_adjustments
        msg += "- Applied OD adjustments based on uploaded adjustment table.\n"
        if adjustment_warnings:
            msg += "- OD adjustments applied with the following warnings:\n"
        for warning in adjustment_warnings:
            st.warning(warning)
            msg += f"    - {warning}\n"

    #### switch wide data to time eplased in hours #####################################
    df_rolling = piogrowth.reindex_w_relative_time(
        df=df_rolling,
        start_time=st.session_state["start_time"],
    )
    st.session_state["df_rolling"] = df_rolling

    st.session_state["rolling_window"] = int(rolling_window)

    df_time_map = (
        df_raw_od_data[["timestamp_rounded", "elapsed_time_in_seconds"]]
        .drop_duplicates()
        .set_index("timestamp_rounded")
    )
    df_time_map["elapsed_time_in_hours"] = (
        df_time_map["elapsed_time_in_seconds"] / 3600.0
    )
    st.session_state["df_time_map"] = df_time_map
    st.session_state["upload_processing_summary_msg"] = msg
    st.write("### Data processing summary:")
    st.write(msg)


# Debug option to inspect session state variables related to data upload and processing
if st.session_state.get("debug_mode", False):
    with st.expander("Developer inspect (session state)", expanded=False):
        st.write("Session state variables related to data upload and processing:")
        st.write(
            {
                "custom_id": st.session_state.get("custom_id"),
                "keep_core_data": st.session_state.get("keep_core_data"),
                "reactors_selected": st.session_state.get("reactors_selected"),
                "remove_negative": st.session_state.get("remove_negative"),
                "fill_na": st.session_state.get("fill_na"),
                "remove_downward_trending": st.session_state.get(
                    "remove_downward_trending"
                ),
                "remove_max": st.session_state.get("remove_max"),
                "filter_by_iqr_range": st.session_state.get("filter_by_iqr_range"),
                "quantile_max": st.session_state.get("quantile_max"),
                "iqr_range_value": st.session_state.get("iqr_range_value"),
                "rolling_window": st.session_state.get("rolling_window"),
                "round_time": st.session_state.get("round_time"),
                "time_ranges": st.session_state.get("time_ranges"),
                "update_zero_timepoint": st.session_state.get("update_zero_timepoint"),
                "start_time (session.state)": st.session_state.get("start_time"),
            }
        )
        if st.session_state.get("df_raw_od_data") is not None:
            st.write(
                "Raw OD data:",
                st.session_state["df_raw_od_data"],
            )
        if st.session_state.get("df_wide_raw_od_data") is not None:
            st.write(
                "Wide raw OD data:",
                st.session_state["df_wide_raw_od_data"],
            )
