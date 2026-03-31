from io import BytesIO

import streamlit as st
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth.convert_qurve

DOWNLOADS_HELP = """
Download processed exports.

Currently available:
- QurvE-format Excel export generated from filtered OD data
"""

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
