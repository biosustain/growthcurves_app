"""UI helpers for the Plate Overviews page."""

import streamlit as st

from src.functions.plotting_functions import (
    plot_baseline_by_group,
    plot_replicates_by_sample,
    plot_rmse_heatmap,
    plot_window_plate,
)


@st.fragment
def ui_replicates(plates: dict):
    """Render a grid of replicate plots by sample."""
    st.subheader("Sample Replicates Across All Plates")
    st.caption("View replicates grouped by sample. Hover over points for details.")
    st.plotly_chart(plot_replicates_by_sample(plates), width="stretch")


@st.cache_data(show_spinner=False)
def _cached_plate_fits_plot(
    plate_data: dict,
    sharey: bool = True,
    log_scale: bool = False,
    show_fitted_curve: bool = True,
    show_phase_boundaries: bool = True,
    show_crosshairs: bool = True,
    show_od_max_line: bool = True,
    show_n0_line: bool = True,
    show_tangent: bool = True,
):
    """Cache the expensive plate fits overview plot generation."""
    return plot_window_plate(
        plate_data,
        sharey=sharey,
        log_scale=log_scale,
        show_fitted_curve=show_fitted_curve,
        show_phase_boundaries=show_phase_boundaries,
        show_crosshairs=show_crosshairs,
        show_od_max_line=show_od_max_line,
        show_n0_line=show_n0_line,
        show_tangent=show_tangent,
    )


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

    # Header with Options popover
    header_col, options_col = st.columns([1, 4], vertical_alignment="center")
    with header_col:
        st.subheader("Plate Fits Overview")
    with options_col:
        with st.popover("Display Options"):
            col_a, col_b = st.columns(2)
            with col_a:
                sharey = st.checkbox("Share Y-axis", value=True, key="sharey_toggle")
                log_scale = st.checkbox(
                    "Log Y-axis", value=False, key="log_scale_toggle"
                )
                show_fitted_curve = st.checkbox(
                    "Fitted model curve", value=True, key="po_fitted_curve"
                )
                show_phase_boundaries = st.checkbox(
                    "Phase boundaries", value=True, key="po_phase_boundaries"
                )
            with col_b:
                show_crosshairs = st.checkbox(
                    "Crosshairs", value=True, key="po_crosshairs"
                )
                show_od_max_line = st.checkbox(
                    "Max OD line", value=True, key="po_od_max_line"
                )
                show_n0_line = st.checkbox(
                    "Baseline OD line", value=True, key="po_n0_line"
                )
                show_tangent = st.checkbox("Tangent line", value=True, key="po_tangent")
            st.image("info_plots/annotations.png", width="stretch")

    st.caption(
        "Growth curve model fits for all wells in the selected plate. "
        "Hover over points for details."
    )
    with st.spinner("Creating Plate Fits Overview..."):
        fig = _cached_plate_fits_plot(
            plates[plate_id],
            sharey=sharey,
            log_scale=log_scale,
            show_fitted_curve=show_fitted_curve,
            show_phase_boundaries=show_phase_boundaries,
            show_crosshairs=show_crosshairs,
            show_od_max_line=show_od_max_line,
            show_n0_line=show_n0_line,
            show_tangent=show_tangent,
        )
        st.plotly_chart(fig, width="stretch")

    # Show RMSE heatmap if growth stats are available
    growth_stats = plate.get("growth_stats", {})
    if growth_stats:
        st.subheader("Model Fit Quality (RMSE)")
        st.plotly_chart(
            plot_rmse_heatmap(plate),
            width="stretch",
        )


def render_plate_overviews_page():
    """Render the full Plate Overviews page UI."""
