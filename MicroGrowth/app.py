from pathlib import Path

import streamlit as st
from src.styling import green_gradient, green_navbar, red_buttons

# set the default plotly chart config to export as svg
if not getattr(st.plotly_chart, "_svg_default", False):
    _orig_plotly_chart = st.plotly_chart

    def _plotly_chart(figure_or_data, *args, config=None, **kwargs):
        config = {"toImageButtonOptions": {"format": "svg"}, **(config or {})}
        return _orig_plotly_chart(figure_or_data, *args, config=config, **kwargs)

    _plotly_chart._svg_default = True
    st.plotly_chart = _plotly_chart

APP_VERSION = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()

try:
    from growthcurve_app import __version__ as PACKAGE_VERSION
except Exception:
    PACKAGE_VERSION = "not-installed"

st.set_page_config(
    page_title="MicroGrowth",
    layout="wide",
    page_icon="MicroGrowth/logo.svg",
    initial_sidebar_state="collapsed",
)

with st.sidebar:
    st.caption(f"MicroGrowth app v{APP_VERSION}")
    st.caption(f"growthcurve_app package v{PACKAGE_VERSION}")

# Display logo (will stay visible even when sidebar is collapsed)
st.logo(
    "MicroGrowth/logo.svg",
    link="https://github.com/biosustain/growthcurves_app/tree/main/MicroGrowth",
)


nav = st.navigation(
    [
        st.Page("src/pages/upload_and_analyse.py", title="Upload & Analyse"),
        st.Page("src/pages/plate_overviews.py", title="Plate Overviews"),
        st.Page("src/pages/check_growth_fits.py", title="Check Growth Fits"),
        st.Page("src/pages/create_visualizations.py", title="Create Visualizations"),
        st.Page("src/pages/download_analyzed_data.py", title="Download Analyzed Data"),
    ],
    position="top",
)

red_buttons()
green_gradient()
green_navbar()

nav.run()
