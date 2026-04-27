from pathlib import Path

import streamlit as st
from ui_components import render_markdown

import piogrowth
from piogrowth.styling import green_gradient, green_navbar, red_buttons

logo_path = Path(__file__).with_name("logo.svg")
logo_source = (
    str(logo_path)
    if logo_path.exists()
    else "https://raw.githubusercontent.com/sambra95/TheGrowthAnalysisApp/main/logo.svg"
)

# General configurations
st.set_page_config(
    page_title="PioGrowth",
    layout="wide",
    page_icon=logo_source,
    initial_sidebar_state="expanded",
)

st.logo(logo_source, link="https://github.com/biosustain/PioGrowth")

# Initialize constants
DEFAULT_CUSTOM_ID = "pioreactor_experiment"
st.session_state.setdefault("custom_id", DEFAULT_CUSTOM_ID)
st.session_state.setdefault("df_raw_od_data", None)

st.session_state["DEFAULT_XLABEL_TPS"] = "Timepoints (rounded)"
st.session_state["DEFAULT_XLABEL_REL"] = "Elapsed time (hours)"


# function creating the about page from a markdown file
def render_about():
    render_markdown("PioGrowth/markdowns/about.md")


# Navigation
raw_data = st.Page("0_upload_data.py", title="Upload Data")
data_dashboard = st.Page("0_data_dashboard.py", title="Data Dashboard")
select_data = st.Page("0_select_data.py", title="Select / Exclude Data")
batch_analysis = st.Page("1_batch_analysis.py", title="Batch Growth Analysis")
turbistat_modus = st.Page("2_turbiostat.py", title="Turbidostat Growth Analysis")
comparative_plots = st.Page("3_comparative_plots.py", title="Comparative Plots")
downloads_page = st.Page("0_downloads.py", title="Downloads")
about_page = st.Page(render_about, title="About")

# Sidebar
with st.sidebar:
    st.info("To reset the app, reload the page.")
    with st.container(border=True):
        st.markdown("#### Workflow")
        st.markdown(
            "1. Upload and preprocess data\n"
            "2. Review data dashboard\n"
            "3. Run batch or turbidostat analysis\n"
            "4. Compare metrics across groups\n"
            "5. Export downloads"
        )
    st.caption(f"PioGrowth v{piogrowth.__version__}")

    debug_mode = st.checkbox(
        "Debug Mode",
        value=st.session_state.get("debug_mode", False),
        help=(
            "Enable debug mode for more verbose logging and"
            " additional information in the app."
        ),
    )
    st.session_state["debug_mode"] = debug_mode


# build multi-page app
pg = st.navigation(
    [
        raw_data,
        data_dashboard,
        select_data,
        batch_analysis,
        turbistat_modus,
        comparative_plots,
        downloads_page,
        about_page,
    ],
    position="top",
)

red_buttons()
green_gradient()
green_navbar()

pg.run()
