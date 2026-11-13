import numbers

import growthcurves as gc
import pandas as pd
import streamlit as st
from plots import plot_reads
from process_data import (
    REQUIRED_COLUMNS,
    REQUIRED_COLUMNS_NAME_MAP,
    process_chibio_data,
    process_od_pioreactor,
    read_od_adjustment_table,
)
from reactor_selector import reactor_selector
from ui_components import page_header_with_help

import growthcurve_app
from growthcurve_app.session_state import render_restore_session_state_ui

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")

UPLOAD_HELP = """
This page loads, processes and reviews a single OD dataset, step by step:
1. Upload the OD data file and, optionally, calibration/turbidostat metadata files
2. Configure the processing options and the time window, and apply them
3. Review the uploaded and processed reads of each reactor used by the Growth Analysis page

The tables of each step can be downloaded on the Downloads page.
"""

page_header_with_help("Upload Data", UPLOAD_HELP)


TIME_WINDOW_KEYS = (
    "dashboard_min_t",
    "dashboard_max_t",
    "dashboard_time_ranges",
    "window_reactor",
)


def callback_clear_raw_data():
    st.session_state["df_raw_od_data"] = None
    st.session_state["df_wide_raw_od_data"] = None
    st.session_state["df_wide_raw_od_data_filtered"] = None
    st.session_state["masked"] = None
    # processed data of the previous upload
    st.session_state["df_processed"] = None
    st.session_state["df_rolling"] = None
    st.session_state["upload_processing_summary_msg"] = None
    # reset the time windows and reactor selection of the previous data
    for key in list(st.session_state):
        if key in TIME_WINDOW_KEYS or str(key).startswith(
            ("time_window_", "reactor_selector::")
        ):
            del st.session_state[key]


def elapsed_hours(index: pd.Index) -> pd.Index:
    """Index of the wide data (timestamps or seconds) as hours since the start."""
    if st.session_state["reactor_type"] == "PioReactor":
        index = growthcurve_app.convert_to_elapsed_hours(
            index, start_time=st.session_state["start_time"]
        )
    else:
        index = index / 3600.0
    return pd.Index(index, name="Elapsed time (hours)")


def stored_window(stored, lo: float, hi: float) -> tuple[float, float]:
    """A stored (start, end) window if it lies within [lo, hi], else (lo, hi)."""
    if (
        isinstance(stored, (tuple, list))
        and len(stored) == 2
        and all(isinstance(v, numbers.Real) for v in stored)
        and lo <= stored[0] <= stored[1] <= hi
    ):
        return float(stored[0]), float(stored[1])
    return lo, hi


def window_slider(label: str, key: str, stored, lo: float, hi: float, **kwargs):
    """Range slider over [lo, hi] with its value in session state under key; set to
    the stored window (or the full range) when its value does not fit [lo, hi]."""
    st.session_state[key] = stored_window(st.session_state.get(key, stored), lo, hi)
    return st.slider(label, lo, hi, key=key, **kwargs)


def option_key(name: str, value) -> str:
    """Widget key of an option saved under name, set to value when missing (e.g.
    after visiting another page). Keyed widgets keep their identity between reruns."""
    key = f"opt_{name}"
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def toggle_option(option: str, name: str, value: bool, **kwargs) -> bool:
    """A single button, on when selected, filling its column: the button version of
    a checkbox for the option saved under name."""
    key = option_key(f"{name}_pill", option if value else None)
    return (
        st.segmented_control(
            option,
            [option],
            key=key,
            label_visibility="collapsed",
            width="stretch",
            **kwargs,
        )
        == option
    )


def option_input(label: str, name: str, default, show: bool, **kwargs):
    """Number input for a processing option, shown only when the option is used;
    otherwise the saved value."""
    value = st.session_state.get(name, default)
    if not show:
        return value
    return st.number_input(label, key=option_key(name, value), **kwargs)


def move_reactor(step: int, reactors: list):
    ss = st.session_state
    i = reactors.index(ss["review_reactor"])
    ss["review_reactor"] = reactors[(i + step) % len(reactors)]


def reactor_reads(df: pd.DataFrame, reactor: str, windows) -> pd.Series:
    """Column of the wide data against hours, in the global window and the reactor's
    own window."""
    (lo, hi), ranges = windows
    s = df[reactor].set_axis(elapsed_hours(df.index)).loc[lo:hi]
    if reactor in ranges:
        s = s.loc[ranges[reactor][0] : ranges[reactor][1]]
    return s


def apply_time_windows(
    df: pd.DataFrame, global_window: tuple[float, float], ranges: dict
) -> pd.DataFrame:
    """Reads of df in the global window and, per reactor, in its own window.
    Reads outside are set to NaN; df itself is left unchanged."""
    df = df.loc[global_window[0] : global_window[1]].copy()
    t = df.index
    for reactor, (lo, hi) in ranges.items():
        if reactor in df.columns:
            df[reactor] = df[reactor].where((t >= lo) & (t <= hi))
    return df


def apply_linear_adjustments(
    df_rolling: pd.DataFrame, adjustment_table: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    required_columns = {"reactor", "od"}
    missing_columns = required_columns - set(adjustment_table.columns)
    if missing_columns:
        raise KeyError(
            "Adjustment table is missing columns: "
            f"{', '.join(sorted(missing_columns))}.",
        )

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


# region: UI components for upload page
########################################################################################
# Session State Restore
render_restore_session_state_ui()

########################################################################################
# Step 1: Upload the OD data file and the optional metadata files
with st.container(border=True, key="card_upload_1"):
    st.header("Step 1. Upload Data")
    reactor_type = st.session_state.get("reactor_type")
    reactor_type_options = list(REQUIRED_COLUMNS_NAME_MAP.keys())
    reactor_type = st.segmented_control(
        "Choose a supported reactor type",
        options=reactor_type_options,
        key=option_key(
            "reactor_type",
            (
                reactor_type
                if reactor_type in reactor_type_options
                else reactor_type_options[0]
            ),
        ),
        required=True,
        width="stretch",
    )
    st.session_state["reactor_type"] = reactor_type
    od_col, calibration_col, turbidostat_col = st.columns(3, gap="medium")
    with od_col:
        st.markdown("**OD Data**")
        with st.popover("Requirements", width="stretch"):
            st.markdown("**Expected structure:**")
            st.markdown("- CSV/TXT file readable by `pandas.read_csv`")
            st.markdown(
                "- Required columns in combined file for PioReactor: "
                f"{', '.join(f'`{col}`' for col in REQUIRED_COLUMNS['PioReactor'])}.\n"
                "  - Required columns in each file per reactor Chi.Bio: "
                f"{', '.join(f'`{col}`' for col in REQUIRED_COLUMNS['Chi.Bio'])}"
                " (reactor name will be the file name).\n"
            )
            st.markdown("- One row per measurement")
            st.markdown("\n > Export from PioReactor WebApp or CLI.")
            st.divider()
            st.markdown("**Example file for PioReactor:**")
            example_data = pd.read_csv(
                "AutoGrowth/data/batch_example/example_batch_data_od_readings.csv",
                usecols=["timestamp_localtime", "pioreactor_unit", "od_reading"],
            )
            st.dataframe(example_data.head(10), hide_index=True, width="stretch")
            st.download_button(
                label="Download example CSV for App testing",
                data=example_data.to_csv(
                    index=False,
                ),
                file_name="example_batch_data_od_readings.csv",
                key="download_example_csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        if reactor_type == "Chi.Bio":
            file = st.file_uploader(
                "Upload one or more CSV files with Chi.Bio OD data. They will be combined "
                "for analysis.",
                type=["csv", "txt"],
                on_change=callback_clear_raw_data,
                accept_multiple_files=True,
            )
        elif reactor_type == "PioReactor":
            file = st.file_uploader(
                "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
                type=["csv", "txt"],
                on_change=callback_clear_raw_data,
            )
        main_options_cols = st.columns([3, 2], gap="medium")
        with main_options_cols[0]:
            keep_core_data = st.checkbox(
                "Keep only core data columns?",
                value=True,
                help="If checked, only the essential columns are kept from the uploaded "
                "file(s).",
            )
        with main_options_cols[1]:
            custom_id = st.text_input(
                "Custom ID for data",
                max_chars=30,
                key=option_key("custom_id", custom_id),
            )
    with calibration_col:
        st.markdown("**OD Calibration Table** (optional)")
        # help message
        with st.popover("See an Example", width="stretch"):
            st.markdown("**OD Calibration Table**")
            st.markdown(
                "- CSV/TXT (`,` or `;`) or Excel file with columns `reactor` and `od`.\n"
                "- Used to adjust OD readings by reactor based on calibration data."
            )
            st.divider()
            st.markdown("**Example:**")
            fname = (
                "AutoGrowth/data/"
                "batch_example/example_batch_data_od_readings_calibration.csv"
            )
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
        od_adjustment_upload = st.file_uploader(
            "OD adjustment table",
            type=["csv", "txt", "xlsx", "xls"],
            key="upload_page_od_adjustment_table",
        )
    with turbidostat_col:
        st.markdown("**Turbidostat Metadata** (optional, PioReactor only)")
        # help message
        with st.popover("See an Example", width="stretch"):
            st.markdown("**Turbidostat Metadata**")
            st.markdown("""
                If provided, peaks are not autodetected. Only available for
                PioReactor datasets.

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
                "AutoGrowth/data/"
                "turbidostat_example/example_2-Pio_Experiment_dilution_events.csv"
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
        turbidostat_meta_upload = st.file_uploader(
            "Dilution metadata (for Turbidostat page)",
            type=["csv"],
            key="upload_page_turbidostat_meta",
            disabled=reactor_type != "PioReactor",
        )
        st.session_state.setdefault("turbidostat_timestamp_col", "timestamp_localtime")
        st.session_state.setdefault("turbidostat_reactor_col", "pioreactor_unit")
        st.session_state.setdefault("turbidostat_message_col", "message")
        timestamp_options = ["timestamp", "timestamp_localtime"]
        if st.session_state.get("turbidostat_timestamp_col") not in timestamp_options:
            st.session_state["turbidostat_timestamp_col"] = "timestamp_localtime"
        ts_col, reactor_col, event_col = st.columns(3, gap="small")
        ts_col.selectbox(
            "Timestamp",
            options=timestamp_options,
            key="turbidostat_timestamp_col",
            help="Column with the timestamps.",
        )
        reactor_col.text_input(
            "Reactor",
            key="turbidostat_reactor_col",
            help="Column with the reactor names.",
        )
        event_col.text_input(
            "Event",
            key="turbidostat_message_col",
            help="Column with the event descriptions.",
        )
    # Save bytes and file name to session state for later processing
    if turbidostat_meta_upload is not None:
        # file is only processed on the Growth Analysis page
        st.session_state["turbidostat_meta_upload_bytes"] = (
            turbidostat_meta_upload.getvalue()
        )
        st.session_state["turbidostat_meta_upload_name"] = turbidostat_meta_upload.name

    if file is None:
        if df_raw_od_data is None:
            st.warning("No data uploaded.")
            st.toast("Upload a comma-separated (`.csv`) file to get started.")

# Step 2: Configure the processing options and the time window, and apply them
windows = None
with st.container(border=True, key="card_upload_2"):
    st.header("Step 2. Configure Processing Options")

    st.write("#### Data filtering options:")
    options_col, reactors_col = st.columns([1.0, 1.35], gap="large")
    with options_col:
        negative_options = [
            "Set negative OD readings to missing (NaN)",
            "Impute negative values by moving average",
        ]
        default_negative = st.session_state.get(
            "negative_handling", negative_options[0]
        )
        if default_negative not in negative_options:
            default_negative = negative_options[1]
        negative_handling = st.pills(
            "How should negative OD readings be handled?",
            options=negative_options,
            key=option_key("negative_handling", default_negative),
            required=True,
            width="stretch",
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
            key=option_key("fill_na", st.session_state.get("fill_na", False)),
        )
        remove_max = st.checkbox(
            "Remove maximum OD readings by quantile",
            key=option_key("remove_max", st.session_state.get("remove_max", False)),
        )
        quantile_max = option_input(
            "Max quantile for maximum removal",
            "quantile_max",
            0.99,
            remove_max,
            min_value=0.9,
            max_value=1.0,
            step=0.01,
        )
        _outlier_options = ["None", "IQR", "ECOD"]
        _outlier_default = st.session_state.get("outlier_method", "None")
        if _outlier_default not in _outlier_options:
            _outlier_default = "None"
        method_col, param1_col, param2_col = st.columns(3, vertical_alignment="bottom")
        outlier_method = method_col.selectbox(
            "Outlier detection method",
            options=_outlier_options,
            key=option_key("outlier_method", _outlier_default),
            help=(
                "- `None`: no outlier removal.\n"
                "- `IQR`: remove outliers using Inter-Quartile Range in a rolling "
                "window of timepoints.\n"
                "- `ECOD`: remove outliers using the ECOD algorithm "
                "(Empirical Cumulative distribution-based Outlier Detection).\n"
                "> If there are missing values and ECOD is selected, you need to "
                "activate imputation so IQR outlier removal and imputation can "
                "be run before running ECOD. That IQR pass uses the IQR "
                "parameters last set with the IQR method."
            ),
        )
        uses_iqr = outlier_method == "IQR"
        with param1_col:
            iqr_range_value = option_input(
                "IQR factor for outlier removal",
                "iqr_range_value",
                1.5,
                uses_iqr,
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                help=(
                    "Multiplier of the IQR. With ECOD, used for a first IQR pass when "
                    "there are missing values."
                ),
            )
        with param2_col:
            rolling_window_iqr = option_input(
                "Rolling window (timepoints) for IQR outlier removal",
                "rolling_window_iqr",
                21,
                uses_iqr,
                min_value=11,
                max_value=141,
                step=2,
                help="With ECOD, used for a first IQR pass when there are missing values.",
            )
        with param1_col:
            ecod_factor = option_input(
                "ECOD factor for outlier removal",
                "ecod_factor",
                4.0,
                outlier_method == "ECOD",
                min_value=0.5,
                max_value=12.0,
                step=0.1,
                help="Lower values are more sensitive; higher values are less sensitive.",
            )
    with reactors_col:
        st.markdown(
            "**Reactors to include in analysis**",
            help=(
                "Green: included. Red: excluded. Click or drag across reactors to "
                "exclude them, or to include them again when starting on an excluded "
                "reactor."
            ),
        )
        if df_wide_raw_od_data is None:
            reactors_selected = []
            st.caption("Upload OD data to show the reactors.")
        else:
            reactors_selected = reactor_selector(df_wide_raw_od_data.columns.to_list())

    st.divider()
    st.write("#### Time and aggregation options:")
    # fixed columns, so the others stay put when the aggregation method is hidden
    rounding_columns = st.columns(4, gap="medium", vertical_alignment="bottom")
    with rounding_columns[0]:
        # Older session snapshots allowed rounding intervals below 5 seconds.
        st.session_state["round_time"] = max(5, st.session_state.get("round_time", 5))
        round_time = st.number_input(
            "Round time (s)",
            min_value=5,
            max_value=300,
            key=option_key("round_time", st.session_state["round_time"]),
            step=5,
            help=(
                "Round time to the nearest multiple of this many seconds (defining "
                "the timesteps), to align time series with slight time offsets. "
                "Rounding helps pivot the data to wide format from the "
                "long format. If you have multiple measurements for the same "
                "reactor within the rounded time window, they will be "
                "aggregated by median or mean "
                "if aggregation is enabled."
            ),
        )
    with rounding_columns[1]:
        aggregate_duplicated_rounded_timepoint = toggle_option(
            "Aggregate data by rounded timepoints",
            "aggregate_duplicated_rounded_timepoint",
            st.session_state.get("aggregate_duplicated_rounded_timepoint", True),
            help=(
                "If selected, multiple measurements for the same reactor within "
                "the rounded time window will be aggregated by median or mean. "
                "If not selected, the app will attempt to pivot the data to "
                "wide format without aggregation, which will lead to errors if "
                "there are duplicated timepoints after rounding."
            ),
        )
    _value = st.session_state.get(
        "aggregate_duplicated_rounded_timepoint_method", "median"
    )
    aggregate_duplicated_rounded_timepoint_method = _value
    if aggregate_duplicated_rounded_timepoint:
        with rounding_columns[2]:
            aggregate_duplicated_rounded_timepoint_method = st.segmented_control(
                "Aggregation method",
                options=["median", "mean"],
                key=option_key(
                    "aggregate_duplicated_rounded_timepoint_method",
                    "median" if _value == "median" else "mean",
                ),
                required=True,
                width="stretch",
                help=(
                    "How multiple measurements for the same reactor within the "
                    "rounded time window are aggregated when pivoting to wide "
                    "format. Median is more robust to outliers and therefore the "
                    "default."
                ),
            )
    with rounding_columns[-1]:
        aggregate_high_frequency_raw_data = toggle_option(
            "Aggregate reads faster than 15 s (PioReactor)",
            "aggregate_high_frequency_raw_data",
            st.session_state.get("aggregate_high_frequency_raw_data", False),
            disabled=reactor_type != "PioReactor",
            help=(
                "If selected, PioReactor raw OD data sampled faster than every 15 "
                "seconds is aggregated to 15-second time bins before processing."
            ),
        )
    st.divider()
    st.write("#### Time window:")
    if df_wide_raw_od_data is not None:
        ss = st.session_state
        hours = elapsed_hours(df_wide_raw_od_data.index)
        t0, t1 = float(hours.min()), float(hours.max())
        reactor_col, plot_col = st.columns([1.0, 1.35], gap="large")
        with reactor_col:
            min_t, max_t = window_slider(
                "Global time window (hours)",
                "time_window_global",
                (ss.get("dashboard_min_t"), ss.get("dashboard_max_t")),
                t0,
                t1,
                help=(
                    "Applies to all reactors. Only reads inside the time windows are "
                    "analysed. Reads outside are kept and are analysed again when a "
                    "window is widened."
                ),
            )
        ss["dashboard_min_t"], ss["dashboard_max_t"] = min_t, max_t
        reactor = reactor_col.selectbox(
            "Reactor to clip",
            list(df_wide_raw_od_data.columns),
            key="window_reactor",
            help="Set a time window for this reactor only.",
        )
        reads = df_wide_raw_od_data[reactor].set_axis(hours)
        observed = reads.loc[min_t:max_t].dropna().index
        ranges = ss.setdefault("dashboard_time_ranges", {})
        r_min, r_max = min_t, max_t
        if len(observed) > 1 and observed.min() < observed.max():
            lo, hi = float(observed.min()), float(observed.max())
            with reactor_col:
                r_min, r_max = window_slider(
                    f"Time window for {reactor} (hours)",
                    f"time_window_{reactor}",
                    ranges.get(reactor),
                    lo,
                    hi,
                    help=(
                        "Bounded by the global time window. Reads outside are left "
                        "out of the analysis of this reactor."
                    ),
                )
            # store only narrowed windows, so a widened global window applies
            if (r_min, r_max) == (lo, hi):
                ranges.pop(reactor, None)
            else:
                ranges[reactor] = (r_min, r_max)
        lo_t, hi_t = max(min_t, r_min), min(max_t, r_max)
        plot_col.plotly_chart(
            plot_reads(
                reads,
                pd.Series((hours < lo_t) | (hours > hi_t), index=hours),
                title=f"{reactor} (red: outside the time windows)",
            ),
            width="stretch",
        )
        windows = ((min_t, max_t), dict(ranges))
    button_pressed = st.button(
        "Apply options to uploaded data", type="primary", width="stretch"
    )
# endregion

### save form state
# remember form values for next time page is opened
st.session_state["keep_core_data"] = keep_core_data
st.session_state["custom_id"] = custom_id
st.session_state["remove_negative"] = remove_negative
st.session_state["negative_handling"] = negative_handling
st.session_state["fill_na"] = fill_na
st.session_state["remove_max"] = remove_max
st.session_state["outlier_method"] = outlier_method
st.session_state["quantile_max"] = quantile_max
st.session_state["iqr_range_value"] = iqr_range_value
st.session_state["rolling_window_iqr"] = rolling_window_iqr
st.session_state["ecod_factor"] = ecod_factor
st.session_state["round_time"] = round_time
st.session_state["aggregate_duplicated_rounded_timepoint"] = (
    aggregate_duplicated_rounded_timepoint
)
st.session_state["aggregate_duplicated_rounded_timepoint_method"] = (
    aggregate_duplicated_rounded_timepoint_method
)
st.session_state["aggregate_high_frequency_raw_data"] = (
    aggregate_high_frequency_raw_data
)


# region: Process files
########################################################################################
# Process data

extra_warn = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id

if button_pressed and file is None and df_raw_od_data is None:
    extra_warn.warning("No data uploaded.")
    st.stop()

msg = st.session_state.get("upload_processing_summary_msg", "")


# File Uploaded ########################################################################
# runs for a new upload or when the options are applied, not on every rerun
new_upload = st.session_state.get("df_raw_od_data") is None
if file and (new_upload or button_pressed):
    # Chi.Bio: One or more files are processed
    # PioReactor: one file is processed
    if reactor_type == "Chi.Bio":
        missing_files = []
        for uploaded_file in file:
            try:
                columns = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
                uploaded_file.seek(0)
            except (OSError, pd.errors.ParserError, ValueError):
                continue
            missing = [
                column
                for column in REQUIRED_COLUMNS[reactor_type]
                if column not in columns
            ]
            if missing:
                missing_files.append((uploaded_file.name, missing, columns))

        if missing_files:
            _, _, columns = missing_files[0]
            other_type = "PioReactor"
            other_required = REQUIRED_COLUMNS[other_type]
            wrong_type_hint = (
                (
                    f" The files look like **{other_type}** input — "
                    "did you select the wrong reactor type?"
                )
                if not any(column not in columns for column in other_required)
                else ""
            )
            details = ", ".join(
                f"`{name}` is missing {', '.join(f'`{col}`' for col in missing)}"
                for name, missing, _ in missing_files
            )
            st.error(
                "One or more uploaded files are missing required columns for "
                f"**{reactor_type}**. "
                f"{details}." + wrong_type_hint
            )
            st.stop()

        # msg is overwritten here (intended)
        df_raw_od_data, df_wide_raw_od_data, msg = process_chibio_data(
            files=file,
            round_time=round_time,
            keep_core_data=keep_core_data,
        )
    elif reactor_type == "PioReactor":
        try:
            columns = pd.read_csv(file, nrows=0).columns.tolist()
            file.seek(0)
        except (OSError, pd.errors.ParserError, ValueError):
            columns = []

        missing = [
            column for column in REQUIRED_COLUMNS[reactor_type] if column not in columns
        ]
        if missing:
            other_type = "Chi.Bio"
            other_required = REQUIRED_COLUMNS[other_type]
            wrong_type_hint = (
                (
                    f" The file looks like **{other_type}** input — "
                    "did you select the wrong reactor type?"
                )
                if columns
                and not any(column not in columns for column in other_required)
                else ""
            )
            st.error(
                "The uploaded file is missing required columns for "
                f"**{reactor_type}**: "
                f"{', '.join((f'`{col}`' for col in missing))}." + wrong_type_hint
            )
            st.stop()

        # msg is overwritten here (intended)
        df_raw_od_data, df_wide_raw_od_data, msg = process_od_pioreactor(
            file=file,
            round_time=round_time,
            keep_core_data=keep_core_data,
            aggregate_duplicated_rounded_timepoint=aggregate_duplicated_rounded_timepoint,
            aggregate_duplicated_rounded_timepoint_method=aggregate_duplicated_rounded_timepoint_method,
            aggregate_high_frequency_raw_data=aggregate_high_frequency_raw_data,
        )

    st.session_state["df_raw_od_data"] = df_raw_od_data
    st.session_state["df_wide_raw_od_data"] = df_wide_raw_od_data
    st.session_state["upload_processing_summary_msg"] = msg
    if new_upload:
        # rerun so the form lists the uploaded reactors
        st.rerun()

### Apply option from form #############################################################
if button_pressed:
    # Keep only reactors selected for analysis
    if not reactors_selected:
        st.warning("No reactors selected. Select at least one reactor to continue.")
        st.stop()
    st.session_state["reactors_selected"] = reactors_selected
    msg += "reactors included in analysis: " + ", ".join(reactors_selected) + "\n"
    # the uploaded data keeps all reactors, so excluded ones can be included again
    df_wide_raw_od_data = df_wide_raw_od_data[reactors_selected]

    # initalize masked here
    masked = pd.DataFrame(
        False,
        index=df_wide_raw_od_data.index,
        columns=df_wide_raw_od_data.columns,
    )
    # df_wide_raw_od_data_filtered will now be used
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.copy()

    msg += "Applied data filtering options:\n"

    #### Apply Data Filtering options ##################################################
    # all to df_wide_raw_od_data_filtered

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

    if outlier_method == "ECOD":
        mask_na = df_wide_raw_od_data_filtered.isna()
        if not fill_na and mask_na.sum().sum() > 0:
            st.error(
                "Found missing values in the data. ECOD outlier detection does not work"
                " with missing values. Consider setting forward and backward filling "
                " before applying ECOD outlier detection."
            )
            st.stop()

    # outlier detection using IQR on rolling window: sets for center value of window a
    # true or false (this would be arguing maybe for long data format)
    # can be used in plot for visualization
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    if outlier_method in ("IQR", "ECOD"):
        kwargs_iqr = {
            "method": "iqr",
            "factor": iqr_range_value,
            "window_size": rolling_window_iqr,
        }
        kwargs = (
            kwargs_iqr
            if outlier_method == "IQR"
            else {"method": "ecod", "factor": ecod_factor}
        )
        _has_missing = df_wide_raw_od_data_filtered.isna().sum().sum() > 0
        if _has_missing and outlier_method == "ECOD":
            st.toast(
                "Missing values found: removing IQR outliers and filling gaps "
                "before ECOD."
            )
        with st.spinner(f"Applying {outlier_method} outlier removal..."):
            if outlier_method == "ECOD" and _has_missing:
                mask_outliers = df_wide_raw_od_data_filtered.apply(
                    gc.preprocessing.detect_outliers,
                    raw=False,
                    **kwargs_iqr,
                ).astype(bool)
                masked = masked | mask_outliers
                n_out = mask_outliers.sum().sum()
                msg += f"- Number of outliers detected (IQR): {n_out}\n"
                msg += f"   - in detail: {mask_outliers.sum().to_dict()}\n"
                df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
                    mask_outliers
                )
                df_wide_raw_od_data_filtered = (
                    df_wide_raw_od_data_filtered.ffill().bfill()
                )
            mask_outliers = df_wide_raw_od_data_filtered.apply(
                gc.preprocessing.detect_outliers,
                raw=False,
                **kwargs,
            ).astype(bool)
            n_out = mask_outliers.sum().sum()
            msg += f"- Number of outliers detected ({outlier_method}): {n_out}\n"
            msg += f"   - in detail: {mask_outliers.sum().to_dict()}\n"
            masked = masked | mask_outliers
            df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
                mask_outliers
            )

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

    masked = masked.fillna(False).astype(bool)
    st.session_state["masked"] = masked
    st.session_state["df_wide_raw_od_data_filtered"] = df_wide_raw_od_data_filtered

    # Fill missing values for the processed-data view when requested.
    if fill_na:
        mask_na = df_wide_raw_od_data_filtered.isna()
        msg += f"- Filling {mask_na.sum().sum():,d} missing OD readings.\n"
        msg += f"   - in detail: {mask_na.sum().to_dict()}\n"
        # ! should I visualize the values differently?
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.ffill().bfill()

    df_rolling = df_wide_raw_od_data_filtered.sort_index()

    # ? Should it not be possible to be run twice in a single session?
    if od_adjustment_upload is not None:
        if st.session_state.get("is_df_rolling_adjusted"):
            st.warning(
                "OD adjustments have already been applied. "
                "Re-applying will overwrite previous adjustments."
            )
        df_adjustments = read_od_adjustment_table(od_adjustment_upload)
        try:
            df_rolling, adjustment_warnings = apply_linear_adjustments(
                df_rolling, df_adjustments
            )
        except KeyError as e:
            error_text = str(e.args[0]) if e.args else str(e)
            st.error(
                f"Check that the required header and columns are present. {error_text}"
            )
            st.stop()
        st.session_state["is_df_rolling_adjusted"] = True
        st.session_state["df_rolling"] = df_rolling
        st.session_state["df_od_adjustment"] = df_adjustments
        msg += "- Applied OD adjustments based on uploaded adjustment table.\n"
        if adjustment_warnings:
            msg += "- OD adjustments applied with the following warnings:\n"
        for warning in adjustment_warnings:
            st.warning(warning)
            msg += f"    - {warning}\n"

    #### switch wide data to time elapsed in hours ####################################
    df_rolling.index = elapsed_hours(df_rolling.index)

    # Untrimmed processed data; df_rolling (used downstream) is this data in the
    # time windows of Step 2
    st.session_state["df_processed"] = df_rolling
    st.session_state["df_rolling"] = df_rolling
    st.session_state.pop("applied_time_windows", None)

    st.session_state["rolling_window_iqr"] = int(rolling_window_iqr)

    st.session_state["upload_processing_summary_msg"] = msg
# endregion

# region: Review
########################################################################################
ss = st.session_state
df_wide = ss.get("df_wide_raw_od_data")
# sessions saved before df_processed existed: their df_rolling is the base
df_processed = ss.setdefault("df_processed", ss.get("df_rolling"))
if windows is not None and df_processed is not None:
    if ss.get("applied_time_windows") != windows:
        ss["df_rolling"] = apply_time_windows(df_processed, *windows)
        ss["applied_time_windows"] = windows
    reactors = list(df_wide.columns)
    if ss.get("review_reactor") not in reactors:
        ss["review_reactor"] = reactors[0]
    with st.container(border=True, key="card_upload_3"):
        st.header(
            "Step 3. Review Data",
            help=(
                "Reads of each reactor in the time windows. The processed reads are "
                "the data used by the Growth Analysis page: filtered, optionally "
                "imputed and adjusted, and trimmed to the time windows."
            ),
        )
        prev_col, reactor_col, next_col = st.columns(
            [2, 4, 2], vertical_alignment="bottom"
        )
        prev_col.button(
            "",
            icon=":material/arrow_back:",
            width="stretch",
            on_click=move_reactor,
            args=(-1, reactors),
            key="review_prev",
            type="primary",
        )
        reactor = reactor_col.selectbox("Reactor", reactors, key="review_reactor")
        next_col.button(
            "",
            icon=":material/arrow_forward:",
            width="stretch",
            on_click=move_reactor,
            args=(1, reactors),
            key="review_next",
            type="primary",
        )
        processing_summary = ss.get("upload_processing_summary_msg")
        if processing_summary:
            st.subheader("Processing summary of OD readings")
            st.markdown(processing_summary)
        masked = ss.get("masked")
        removed = (
            reactor_reads(masked, reactor, windows)
            if masked is not None and reactor in masked.columns
            else None
        )
        st.plotly_chart(
            plot_reads(
                reactor_reads(df_wide, reactor, windows),
                removed,
                title=(
                    f"Reads – {reactor} (red: removed by the filtering of Step 2)"
                    if reactor in reactors_selected
                    else f"Reads – {reactor} (excluded from analysis)"
                ),
            ),
            width="stretch",
        )
        df_rolling = ss.get("df_rolling")
        if reactor in df_rolling.columns:
            st.plotly_chart(
                plot_reads(df_rolling[reactor], title=f"Processed reads – {reactor}"),
                width="stretch",
            )
# endregion

# region: Debugging and inspection
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
                "remove_max": st.session_state.get("remove_max"),
                "quantile_max": st.session_state.get("quantile_max"),
                "iqr_range_value": st.session_state.get("iqr_range_value"),
                "rolling_window_iqr": st.session_state.get("rolling_window_iqr"),
                "round_time": st.session_state.get("round_time"),
                "time_ranges": st.session_state.get("dashboard_time_ranges"),
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
        if st.session_state.get("df_wide_raw_od_data_filtered") is not None:
            st.write(
                "Wide raw OD data after filtering:",
                st.session_state["df_wide_raw_od_data_filtered"],
            )
        if st.session_state.get("df_rolling") is not None:
            st.write(
                "Processed OD data:",
                st.session_state["df_rolling"],
            )
        if st.session_state.get("df_od_adjustment") is not None:
            st.write(
                "OD adjustment table:",
                st.session_state["df_od_adjustment"],
            )
# endregion
