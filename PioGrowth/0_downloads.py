from io import BytesIO

import streamlit as st
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth.convert_qurve
from piogrowth.session_state import (
    render_export_session_state_ui,
    ui_key_inspector,
    ui_overview_table,
)

MAX_LIST_REPR = 5  # max items shown inline for lists
DOWNLOADS_HELP = """
Download processed exports.

Currently available:
- QurvE-format Excel export generated from filtered OD data
- Session state ZIP for restoring the full app state later
"""

# Keys that should never be included in the snapshot
SNAPSHOT_EXCLUDE_KEYS = frozenset(
    {
        "df_qurve_format",  # regenerable BytesIO output
        "session_state_zip",  # the snapshot itself
        "session_state_zip_upload",  # file-uploader widget on upload page
        "upload_page_od_adjustment_table",  # file-uploader widget state
        "upload_page_turbidostat_meta",  # file-uploader widget state
    }
)

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

render_export_session_state_ui(
    custom_id=custom_id, exclude_keys=SNAPSHOT_EXCLUDE_KEYS
)

if st.session_state.get("debug_mode", False):
    """Debug session state page for development purposes."""
    st.subheader("Debug: Session State Contents")

    # ── Overview table ────────────────────────────────────────────────────────────────
    ui_overview_table(max_list_repr=MAX_LIST_REPR)

    # ── Per-key inspector ─────────────────────────────────────────────────────────────
    ui_key_inspector(max_list_repr=MAX_LIST_REPR)
