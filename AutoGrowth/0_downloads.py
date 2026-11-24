from functools import partial
from io import BytesIO

import streamlit as st
from buttons import convert_data, create_download_button
from ui_components import page_header_with_help, show_warning_to_upload_data

import growthcurve_app.convert_qurve
from growthcurve_app.session_state import (
    render_export_session_state_ui,
    ui_key_inspector,
    ui_overview_table,
)

MAX_LIST_REPR = 5  # max items shown inline for lists
DOWNLOADS_HELP = """
Download processed exports.

Currently available:
- CSV tables of each data processing step on the Upload Data page
- QurvE-format Excel export generated from filtered OD data
- Session state ZIP for restoring the full app state later
"""

# (label, session key, file name) of the data tables of the Upload Data page
DATA_TABLES = [
    ("raw data  \n(long format)", "df_raw_od_data", "data_long_rounded_timestamps.csv"),
    (
        "raw data  \n(wide format)",
        "df_wide_raw_od_data",
        "data_wide_rounded_timestamps.csv",
    ),
    (
        "filtered data",
        "df_wide_raw_od_data_filtered",
        "filtered_data_wide_rounded_timestamps.csv",
    ),
    (
        "processed data",
        "df_rolling",
        "processed_wide_data_with_rounded_timestamps.csv",
    ),
]

# Keys that should never be included in the snapshot
SNAPSHOT_EXCLUDE_KEYS = frozenset(
    {
        "df_qurve_format",  # regenerable BytesIO output
        "session_state_zip",  # the snapshot itself
        "session_state_zip_upload",  # file-uploader widget on upload page
        "upload_page_od_adjustment_table",  # file-uploader widget state
        "upload_page_turbidostat_meta",  # file-uploader widget state
        "turbidostat_meta_upload_bytes",  # file content, ignore if not processed.
        "prepare_session_state_zip",  # do not save state of button pressed
        "create_qurve_format",  # do not save state of button pressed
        "review_prev",  # buttons cannot be restored
        "review_next",
        "reactor_selector::plate_painter",  # component state cannot be restored
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

with st.container(border=True, key="card_downloads_tables"):
    st.header("Data Tables")
    st.caption("CSV files of the data processing steps on the Upload Data page.")
    tables = [
        (label, st.session_state[key], name)
        for label, key, name in DATA_TABLES
        if st.session_state.get(key) is not None
    ]
    for col, (label, df, name) in zip(st.columns(len(tables)), tables, strict=True):
        with col:
            create_download_button(
                label=f"Download {label}",
                data=partial(convert_data, df),  # CSV written only on click
                file_name=name,
                disabled=False,
                mime="text/csv",
            )

with st.container(border=True, key="card_downloads_qurve"):
    st.header("Export QurvE Format (Optional)")
    st.caption("Create and download an Excel file in QurvE-compatible format.")
    convert = st.button("Store QurvE format data", key="create_qurve_format")

if convert:
    if df_wide_raw_od_data_filtered is not None and start_time is not None:
        with st.spinner("Converting to QurvE format...", show_time=True):
            qurve_data = growthcurve_app.convert_qurve.to_qurve_format(
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

render_export_session_state_ui(custom_id=custom_id, exclude_keys=SNAPSHOT_EXCLUDE_KEYS)

if st.session_state.get("debug_mode", False):
    # Debug session state page for development purposes.
    st.subheader("Debug: Session State Contents")
    excluded_keys = [k for k in SNAPSHOT_EXCLUDE_KEYS if k in st.session_state]
    st.write(f"Excluded keys from snapshot (not shown): {excluded_keys}")
    # ── Overview table ────────────────────────────────────────────────────────────────
    ui_overview_table(max_list_repr=MAX_LIST_REPR, exclude_keys=excluded_keys)

    # ── Per-key inspector ─────────────────────────────────────────────────────────────
    ui_key_inspector(max_list_repr=MAX_LIST_REPR, exclude_keys=excluded_keys)
