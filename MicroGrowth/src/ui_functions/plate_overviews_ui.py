"""UI helpers for the Plate Overviews page."""

import streamlit as st
from src.functions.plotting_functions import (
    plot_baseline_by_group,
    plot_replicates_by_sample,
    plot_window_plate,
)


@st.fragment
def ui_replicates(plates: dict):
    """Render a grid of replicate plots by sample."""
    st.subheader("Sample Replicates Across All Plates")
    st.caption("View replicates grouped by sample. Hover over points for details.")
    st.plotly_chart(plot_replicates_by_sample(plates), width="stretch")


@st.cache_data(show_spinner=False)
def _cached_plate_fits_plot(plate_data: dict):
    """Cache the expensive plate fits overview plot generation."""
    return plot_window_plate(plate_data)


@st.fragment
def ui_window_fits_plate_overview(plates: dict):
    """Render baseline and plate-window fits for a selected plate."""
    plate_id = st.selectbox("Plate", sorted(plates), key="winfit_plate_overview")
    st.subheader("Plate Blanks")
    st.caption(
        "Group-specific blank baselines. Lines show per-group means; points show each blank well."
    )
    plate = plates[plate_id]
    blank_group_map = (plate.get("params") or {}).get("blank_group_assignments", {})

    st.plotly_chart(
        plot_baseline_by_group(
            plate["baseline"],
            blank_group_map=blank_group_map,
        )
    )

    st.subheader("Plate Fits Overview")
    st.caption(
        "Model fits for every well. Outline colour shows fit quality: "
        "green passes, red fails, grey has no fit."
    )
    with st.spinner("Creating Plate Fits Overview..."):
        fig = _cached_plate_fits_plot(plate)
        # Hover off (see hovermode); modebar kept for the download button.
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": False,
                "modeBarButtonsToRemove": [
                    "zoom2d",
                    "pan2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                ],
                # Explicit size, else the export clips the axis labels.
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"plate_fits_{plate_id}",
                    "width": 1800,
                    "height": 1000,
                    "scale": 2,
                },
            },
        )


def render_plate_overviews_page():
    """Render the full Plate Overviews page UI."""
