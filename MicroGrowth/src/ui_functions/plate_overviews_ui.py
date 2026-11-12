"""UI helpers for the Plate Overviews page."""

import streamlit as st
from src.functions.plotting_functions import (
    highlight_wells,
    plot_baseline_by_group,
    plot_replicates_by_sample,
    plot_window_plate,
    unshare_axes,
    worst_fit_wells,
)

# Hover only: no toolbar, no zoom
HOVER_ONLY = {"displayModeBar": False, "scrollZoom": False}


# Cached figure builders: reruns (e.g. moving a slider) reuse the figures.
# st.cache_data returns a copy, so callers may modify the figure.
@st.cache_data(show_spinner=False)
def _replicates_plot(plates: dict):
    return plot_replicates_by_sample(plates)


@st.cache_data(show_spinner=False)
def _blanks_plot(baseline, blank_group_map: dict):
    return plot_baseline_by_group(baseline, blank_group_map=blank_group_map)


@st.cache_data(show_spinner=False)
def _plate_fits_plot(plate: dict):
    return plot_window_plate(plate)


def _section_header(title: str, key: str) -> bool:
    """Section heading with a Share axes toggle in the top right; returns it."""
    title_col, toggle_col = st.columns([4, 1], vertical_alignment="center")
    title_col.header(title)
    with toggle_col.container(horizontal_alignment="right"):
        return st.toggle(
            "Share axes",
            value=True,
            key=key,
            help="On: every panel uses the same time and OD scale, so curves can be "
            "compared directly. Off: each panel is scaled to its own data.",
        )


def _scaled(fig, shared: bool):
    """The figure as is if axes are shared, else with per-panel axes."""
    return fig if shared else unshare_axes(fig)


@st.fragment
def ui_replicates(plates: dict):
    """Render a grid of replicate plots by sample."""
    with st.container(border=True, key="card_plate_overviews_replicates"):
        shared = _section_header(
            "Sample Replicates Across All Plates", "share_axes_replicates"
        )
        st.caption(
            "Replicate curves grouped by sample, one colour per replicate. "
            "Hover over points to see the plate and well."
        )
        st.plotly_chart(
            _scaled(_replicates_plot(plates), shared),
            width="stretch",
            config=HOVER_ONLY,
        )


@st.fragment
def ui_window_fits_plate_overview(plates: dict):
    """One section for the selected plate: its blanks and its fits overview."""
    with st.container(border=True, key="card_plate_overviews_plate"):
        shared = _section_header("Plate Blanks and Fits", "share_axes_plate")
        st.subheader("Plate")
        with st.container(key="prominent_select_plate", width=420):
            plate_id = st.selectbox(
                "Plate",
                sorted(plates),
                key="winfit_plate_overview",
                label_visibility="collapsed",
            )
        plate = plates[plate_id]

        st.subheader("Blanks")
        st.caption(
            "One box per blank group: points are the blank wells, the line is the "
            "group mean used for baseline correction. Hover over points to see the "
            "well."
        )
        params = plate.get("params") or {}
        blanks = _blanks_plot(
            plate["baseline"], params.get("blank_group_assignments") or {}
        )
        st.plotly_chart(
            _scaled(blanks, shared),
            width="stretch",
            config=HOVER_ONLY,
        )

        ui_plate_fits(plate, plate_id, shared)


@st.fragment
def ui_plate_fits(plate: dict, plate_id: str, shared: bool):
    """Plate fits overview; its own fragment, so the slider reruns only this."""
    st.subheader("Fits")
    st.caption(
        "Model fits for every well. Darker points were used in the fit; the green "
        "band is the exponential phase. Shaded wells have the poorest fits "
        "(highest RMSE relative to their OD increase)."
    )
    pct = st.slider(
        "Highlight poorest fits (%)",
        min_value=0,
        max_value=100,
        value=0,
        format="%d%%",
        key=f"winfit_pct_highlight_{plate_id}",
        help=(
            "Percentage of fitted wells to shade, ranked by fit RMSE divided by the "
            "well's OD increase. Wells without a fit or without growth are not ranked."
        ),
    )
    with st.spinner("Creating Plate Fits Overview..."):
        fig = _scaled(_plate_fits_plot(plate), shared)
        fig = highlight_wells(fig, plate, worst_fit_wells(plate, pct))
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True})
