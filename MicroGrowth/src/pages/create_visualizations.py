"""Create interactive visualizations of growth data."""

import streamlit as st
from src.functions.common import require_plates
from src.functions.plotting_functions import (
    plot_mean_growth,
    plot_replicates_scatter,
    plot_single_growth_stat,
    style_info_axes,
)
from src.functions.visualization_functions import (
    _build_growth_curves_long_df,
    _build_growth_stats_long_df,
    _max_time_hours,
)
from src.ui_functions.create_visualizations_ui import (
    ui_growth_curves_controls_container,
    ui_growth_selection_container,
    ui_growth_stats_controls_container,
)
from src.ui_functions.ui_components import page_header_with_help

page_header_with_help(
    "Create Visualizations",
    """
**Workflow Overview - Create Visualizations**

Use this page to **generate and explore plots** of your growth analysis results. You can create bar/violin plots of growth parameters and overlay or compare growth curves across samples.

**Step 1 - Select samples**
Use the sample selection panel at the top to choose which plates and samples to include in your plots. You can filter by plate, sample name, or condition.

**Step 2 - Configure growth statistics plots (left column)**
- Choose the **X axis** variable to group samples by strain, condition, or sample name
- Choose the **Legend** variable to colour-code data points by a second grouping (e.g. condition)
- Reorder groups using the order controls to customise the plot layout
- Click **Apply** to generate bar/violin plots for all growth parameters (μ_max, doubling time, lag time, yield, etc.)

**Step 3 - Configure growth curve plots (right column)**
- Set the **time range** to focus on a specific window of the experiment
- Reorder samples in the legend using the order controls
- Enable **Mean curves** to plot the mean ± spread across replicates for each sample
- Enable **Replicate scatter** to overlay individual replicate curves

**Downloading plots**
Each plot can be downloaded individually using the camera icon in the top right corner of the plot. To bulk-export all plots, use the **Download Analyzed Data** page.
""",
)

plates = require_plates()


selection = ui_growth_selection_container(plates)
sel_ids = selection["sel_ids"]
sel_opt = selection["sel_opt"]
sel_sample_names = selection["sel_sample_names"]
has_split = selection["has_split"]

max_t = _max_time_hours(plates)

col1, col2 = st.columns([1, 1])
with col1:
    stats = ui_growth_stats_controls_container(has_split, sel_opt)
with col2:
    curves = ui_growth_curves_controls_container(max_t, sel_sample_names)

# Set by the Generate buttons inside the Step 2 containers
action = st.session_state.pop("viz_action", None)
apply_stats = action == "stats"
apply_mean = action == "mean"
apply_reps = action == "reps"

x_col = stats["x_col"]
legend_col = stats["legend_col"]
x_ordered = stats["x_ordered"]
legend_ordered = stats["legend_ordered"]

curves_t0 = curves["curves_t0"]
curves_t1 = curves["curves_t1"]
curves_ordered = curves["curves_ordered"]

# -----------------------------
# Plots
# -----------------------------
if apply_stats:
    long_df, _ = _build_growth_stats_long_df(plates, sel_ids)
    if long_df.empty:
        st.info(
            "No growth statistics found for the selected samples. "
            "Run analysis first or select samples with analyzed wells."
        )
    else:
        # Display each metric as a separate downloadable plot.
        metrics = [
            "mu_max",
            "intrinsic_growth_rate",
            "doubling_time",
            "max_od",
            "exp_phase_start",
            "exp_phase_end",
            "time_at_umax",
            "od_at_umax",
        ]

        for metric in metrics:
            metric_df = long_df[long_df["metric"] == metric].copy()
            if not metric_df.empty:
                fig = plot_single_growth_stat(
                    metric_df,
                    x_col=x_col,
                    legend_col=legend_col,
                    x_order=x_ordered,
                    legend_order=legend_ordered,
                )
                with st.container(border=True, key=f"card_viz_stat_{metric}"):
                    st.plotly_chart(style_info_axes(fig), width="stretch")

# Build the shared curves DF only if needed
if apply_mean or apply_reps:
    curves_df = _build_growth_curves_long_df(
        plates, sel_sample_names, fitted=curves["use_fitted"]
    )

if apply_mean:
    with st.container(border=True, key="card_viz_mean"):
        st.plotly_chart(
            style_info_axes(
                plot_mean_growth(
                    curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
                )
            ),
            width="stretch",
        )

if apply_reps:
    if curves_df.empty:
        st.info("No replicate data found.")
    else:
        with st.container(border=True, key="card_viz_reps"):
            st.plotly_chart(
                style_info_axes(
                    plot_replicates_scatter(
                        curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
                    )
                ),
                width="stretch",
            )
