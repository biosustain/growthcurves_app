import growthcurves as gc
import pandas as pd
import streamlit as st
from process_data import (
    REQUIRED_COLUMNS,
    REQUIRED_COLUMNS_NAME_MAP,
    process_chibio_data,
    process_od_pioreactor,
)
from ui_components import page_header_with_help

import growthcurve_app
from growthcurve_app.session_state import render_restore_session_state_ui

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")
df_wide_raw_od_data_filtered = st.session_state.get("df_wide_raw_od_data_filtered")
min_periods = st.session_state.get("min_periods", 5)
# use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
# initialize
st.session_state.setdefault("USE_ELAPSED_TIME_FOR_PLOTS", True)

UPLOAD_HELP = """
This page loads and preprocesses a single an OD dataset.

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
# Step 1: Upload File with OD/bioscatter data
with st.container(border=True):
    # header and example data file with requirements in popover
    header_col, req_col = st.columns([4, 1], vertical_alignment="center")
    with header_col:
        st.header("Step 1. Upload OD Data")
    with req_col:
        # Help message
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
    # File Uploading of main data file
    st.markdown("**Main OD Data**")
    _file_name = st.session_state.get("file_od_upload_name")
    reactor_type = st.session_state.get("reactor_type")
    reactor_type_options = list(REQUIRED_COLUMNS_NAME_MAP.keys())
    reactor_type = st.radio(
        label="Choose an supported reactor type",
        options=reactor_type_options,
        index=(
            reactor_type_options.index(reactor_type)
            if reactor_type in reactor_type_options
            else 0
        ),
    )
    st.session_state["reactor_type"] = reactor_type
    if _file_name is not None:
        st.info(f"File previously uploaded: {_file_name}")
    if reactor_type == "Chi.Bio":
        file = st.file_uploader(
            "Upload one or more CSV files with Chi.Bio OD data. They will be combined "
            "for analysis.",
            type=["csv", "txt"],
            on_change=callback_clear_raw_data,
            accept_multiple_files=True,
        )
        if file:
            st.session_state["file_od_upload_name"] = ", ".join(f.name for f in file)
    elif reactor_type == "PioReactor":
        file = st.file_uploader(
            "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
            type=["csv", "txt"],
            on_change=callback_clear_raw_data,
            accept_multiple_files=True if reactor_type == "Chi.Bio" else False,
        )
        if file is not None:
            # st.session_state["file_od_upload_bytes"] = file.getvalue()
            st.session_state["file_od_upload_name"] = file.name
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
        st.markdown("**Turbidostat Metadata** (for PioReactor datasets only)")
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
        _file_name = st.session_state.get("turbidostat_meta_upload_name")
        if _file_name is not None:
            st.info(f"File previously uploaded: {_file_name}")
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
        # (st.session_state["od_adjustment_upload_bytes"]
        #  = od_adjustment_upload.getvalue())
        st.session_state["od_adjustment_upload_name"] = od_adjustment_upload.name
    if turbidostat_meta_upload is not None:
        # file is only processed in 2_turbidostat.py
        st.session_state["turbidostat_meta_upload_bytes"] = (
            turbidostat_meta_upload.getvalue()
        )
        st.session_state["turbidostat_meta_upload_name"] = turbidostat_meta_upload.name

# Step 3: Configure preprocessing options
### Form ##############################################################################
with st.container(border=True):
    st.header("Step 3. Configure Processing Options")
    st.warning(
        'Options are only saved if you press "Apply options to uploaded data" button '
        "at the end of this section."
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
                df_raw_od_data["reactor"].dropna().astype(str).unique().tolist()
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
                "negative_handling", negative_options[0]
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
                    "(Empirical Cumulative distribution-based Outlier Detection).\n"
                    "> If there are missing values and ECOD is selected, you need to "
                    "activate imputation so IQR outlier removal and imputation can "
                    "be run before running ECOD."
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
# endregion

### save form state
# remember form values for next time page is opened
st.session_state["keep_core_data"] = keep_core_data
st.session_state["custom_id"] = custom_id
st.session_state["reactors_selected"] = reactors_selected
st.session_state["remove_negative"] = remove_negative
st.session_state["negative_handling"] = negative_handling
st.session_state["fill_na"] = fill_na
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
# this runs wheather the button is pressed or not, but only if a file is uploaded
if file:
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
                f" The files look like **{other_type}** input — did you select the wrong reactor type?"
                if not any(column not in columns for column in other_required)
                else ""
            )
            details = ", ".join(
                f"`{name}` is missing {', '.join(f'`{col}`' for col in missing)}"
                for name, missing, _ in missing_files
            )
            st.error(
                f"One or more uploaded files are missing required columns for **{reactor_type}**. "
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
                f" The file looks like **{other_type}** input — did you select the wrong reactor type?"
                if columns
                and not any(column not in columns for column in other_required)
                else ""
            )
            st.error(
                f"The uploaded file is missing required columns for **{reactor_type}**: "
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
        )

    rerun = st.session_state.get("df_raw_od_data") is None
    st.session_state["df_raw_od_data"] = df_raw_od_data
    st.session_state["df_wide_raw_od_data"] = df_wide_raw_od_data
    st.session_state["upload_processing_summary_msg"] = msg  # ? is it needed
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
        df_raw_od_data["reactor"].astype(str).isin(reactors_selected)
    ]

    # initalize masked here
    masked = pd.DataFrame(
        False,
        index=df_wide_raw_od_data.index,
        columns=df_wide_raw_od_data.columns,
    )
    # df_wide_raw_od_data_filtered will now be used
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.copy()

    msg = "Applied data filtering options:\n"

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
    print(f"Applying outlier method: {outlier_method}")
    if outlier_method in ("IQR", "ECOD"):
        kwargs_iqr = {
            "method": "iqr",
            "factor": iqr_range_value,
            "window_size": rolling_window,
        }
        kwargs = (
            kwargs_iqr
            if outlier_method == "IQR"
            else {"method": "ecod", "factor": ecod_factor}
        )
        _has_missing = df_wide_raw_od_data_filtered.isna().sum().sum() > 0
        if _has_missing and outlier_method == "ECOD":
            st.warning(
                "Found missing values in the data. ECOD outlier detection does not work"
                " with missing values. I will remove using IQR some outliers and then "
                " forward and backward fill values "
                " before applying ECOD outlier detection."
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

    # now only apply NA filling for rolling median data?
    if fill_na:
        mask_na = df_wide_raw_od_data_filtered.isna()
        msg += f"- Filling {mask_na.sum().sum():,d} missing OD readings.\n"
        msg += f"   - in detail: {mask_na.sum().to_dict()}\n"
        # ! should I visualize the values differently?
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.ffill().bfill()

    df_rolling = (
        df_wide_raw_od_data_filtered.rolling(
            rolling_window,
            min_periods=min_periods,
            center=True,
        )
        .median()
        .sort_index()
    )

    # ? Should it not be possible to be run twice in a single session?
    if od_adjustment_upload is not None:
        if st.session_state.get("is_df_rolling_adjusted"):
            st.warning(
                "OD adjustments have already been applied. "
                "Re-applying will overwrite previous adjustments."
            )
        df_adjustments = pd.read_csv(od_adjustment_upload).convert_dtypes()
        try:
            df_rolling, adjustment_warnings = apply_linear_adjustments(
                df_rolling, df_adjustments
            )
        except KeyError as e:
            error_text = str(e.args[0]) if e.args else str(e)
            st.error(
                "Check that the required header and columns are present. "
                f"{error_text}"
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

    #### switch wide data to time eplased in hours #####################################
    if reactor_type == "PioReactor":
        df_rolling = growthcurve_app.reindex_w_relative_time(
            df=df_rolling,
            start_time=st.session_state["start_time"],
        )
    elif reactor_type == "Chi.Bio":

        df_rolling = growthcurve_app.convert_seconds_to_hours(df_rolling)
    else:
        # should not happen
        st.error(f"Unknown reactor type: {reactor_type}")
        st.stop()

    st.session_state["df_rolling"] = df_rolling

    st.session_state["rolling_window"] = int(rolling_window)

    st.session_state["upload_processing_summary_msg"] = msg
    st.write("### Data processing summary:")
    st.write(msg)

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
        if st.session_state.get("df_wide_raw_od_data_filtered") is not None:
            st.write(
                "Wide raw OD data after filtering:",
                st.session_state["df_wide_raw_od_data_filtered"],
            )
        if st.session_state.get("df_rolling") is not None:
            st.write(
                "Rolling OD data:",
                st.session_state["df_rolling"],
            )
        if st.session_state.get("df_od_adjustment") is not None:
            st.write(
                "OD adjustment table:",
                st.session_state["df_od_adjustment"],
            )
# endregion
