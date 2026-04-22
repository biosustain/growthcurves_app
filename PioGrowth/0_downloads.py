import io
import json
import pickle
from io import BytesIO

import streamlit as st
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth.convert_qurve

DOWNLOADS_HELP = """
Download processed exports.

Currently available:
- QurvE-format Excel export generated from filtered OD data
- Session state ZIP for restoring the full app state later
"""


_SESSION_STATE_SCALAR_KEYS = [
    "custom_id",
    "keep_core_data",
    "reactors_selected",
    "remove_negative",
    "negative_handling",
    "fill_na",
    "remove_downward_trending",
    "remove_max",
    "outlier_method",
    "quantile_max",
    "iqr_range_value",
    "rolling_window",
    "ecod_factor",
    "round_time",
    "aggregate_duplicated_rounded_timepoint",
    "aggregate_duplicated_rounded_timepoint_method",
    "USE_ELAPSED_TIME_FOR_PLOTS",
    "is_df_rolling_adjusted",
    "turbidostat_timestamp_col",
    "turbidostat_reactor_col",
    "turbidostat_message_col",
    "upload_processing_summary_msg",
    "file_od_upload_name",
    "od_adjustment_upload_name",
    "turbidostat_meta_upload_name",
]

_SESSION_STATE_DATAFRAME_KEYS = [
    "df_raw_od_data",
    "df_wide_raw_od_data",
    "df_wide_raw_od_data_filtered",
    "df_rolling",
    "df_time_map",
    "masked",
    "df_od_adjustment",
]

_SESSION_STATE_FILE_BYTES_KEYS = [
    "od_adjustment_upload_bytes",
    "turbidostat_meta_upload_bytes",
]


def build_session_state_zip() -> bytes:
    """Serialize the current session state to a ZIP file and return as bytes."""
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Scalar metadata
        metadata = {}
        for key in _SESSION_STATE_SCALAR_KEYS:
            val = st.session_state.get(key)
            if val is not None:
                metadata[key] = val
        start_time = st.session_state.get("start_time")
        if start_time is not None:
            metadata["start_time"] = str(start_time)
        zf.writestr("metadata.json", json.dumps(metadata))

        # DataFrames serialized with pickle
        for key in _SESSION_STATE_DATAFRAME_KEYS:
            df = st.session_state.get(key)
            if df is not None:
                zf.writestr(f"dataframes/{key}.pkl", pickle.dumps(df))

        # Raw file bytes
        for key in _SESSION_STATE_FILE_BYTES_KEYS:
            data = st.session_state.get(key)
            if data is not None:
                zf.writestr(f"files/{key}.bin", data)

    buf.seek(0)
    return buf.getvalue()

page_header_with_help("Downloads", DOWNLOADS_HELP)

custom_id = st.session_state.get("custom_id", "pioreactor_experiment")
df_wide_raw_od_data_filtered = st.session_state.get("df_wide_raw_od_data_filtered")
start_time = st.session_state.get("start_time")
no_data_uploaded = st.session_state.get("df_rolling") is None

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

with st.container(border=True):
    st.header("Export QurvE Format (Optional)")
    st.caption("Create and download an Excel file in QurvE-compatible format.")
    convert = st.button("Store QurvE format data", key="create_qurve_format")

if convert:
    if df_wide_raw_od_data_filtered is not None and start_time is not None:
        with st.spinner("Converting to QurvE format...", show_time=True):
            qurve_data = piogrowth.convert_qurve.to_qurve_format(
                df_wide_raw_od_data_filtered,
                start_time=start_time,
            )
            buffer = BytesIO()
            qurve_data.to_excel(buffer, index=True)
            buffer.seek(0)
        st.session_state["df_qurve_format"] = buffer
    else:
        st.warning("No filtered data available to convert to QurvE format.")

if st.session_state.get("df_qurve_format") is not None:
    st.download_button(
        label="Download QurvE format data",
        data=st.session_state["df_qurve_format"],
        file_name=f"{custom_id}_qurve_format.xlsx",
        mime="mime/xlsx",
        key="download_qurve_format",
    )

with st.container(border=True):
    st.header("Export Session State")
    st.caption(
        "Download a ZIP snapshot of the full app state. "
        "Re-upload it on the **Upload Data** page to restore the session exactly as it was."
    )
    prepare = st.button("Prepare session state download", key="prepare_session_state_zip")

if prepare:
    with st.spinner("Building session state ZIP...", show_time=True):
        st.session_state["session_state_zip"] = build_session_state_zip()

if st.session_state.get("session_state_zip") is not None:
    st.download_button(
        label="Download session state ZIP",
        data=st.session_state["session_state_zip"],
        file_name=f"{custom_id}_session_state.zip",
        mime="application/zip",
        key="download_session_state_zip",
    )
