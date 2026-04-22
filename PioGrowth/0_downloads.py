import io
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

# Keys that should never be included in the snapshot.
# Streamlit internal keys (start with "_" or "FormSubmitter:") are excluded
# automatically; these are additional transient or widget-bound keys.
_SNAPSHOT_EXCLUDE_KEYS = frozenset({
    "df_qurve_format",           # regenerable BytesIO output
    "session_state_zip",         # the snapshot itself
    "session_state_zip_upload",  # file-uploader widget on upload page
    "upload_page_od_adjustment_table",  # file-uploader widget state
    "upload_page_turbidostat_meta",     # file-uploader widget state
})

_SNAPSHOT_EXCLUDE_PREFIXES = ("_", "FormSubmitter:")


def build_session_state_zip() -> bytes:
    """Pickle all serializable session state into a ZIP and return as bytes."""
    import zipfile

    state_to_save = {}
    for key, val in st.session_state.items():
        if key in _SNAPSHOT_EXCLUDE_KEYS:
            continue
        if any(key.startswith(p) for p in _SNAPSHOT_EXCLUDE_PREFIXES):
            continue
        if isinstance(val, io.RawIOBase | io.BufferedIOBase):
            val.seek(0)
            val = io.BytesIO(val.read())
        try:
            pickle.dumps(val)
            state_to_save[key] = val
        except Exception:
            continue

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session_state.pkl", pickle.dumps(state_to_save))
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
