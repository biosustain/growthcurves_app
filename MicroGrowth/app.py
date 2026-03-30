import streamlit as st

from src.styling import green_gradient, green_navbar, red_buttons

st.set_page_config(
    page_title="MicroGrowth", layout="wide", page_icon="MicroGrowth/logo.svg"
)

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
