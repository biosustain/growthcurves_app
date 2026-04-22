"""Comparative plots page for PioGrowth.

Allows users to assign groups to reactors and generates interactive bar/box/violin
plots comparing growth metrics across groups.

Works with data from either:
- Batch growth analysis (``batch_analysis_summary_df``, simple reactor index)
- Turbidostat analysis (``batch_analysis_summary_df``, MultiIndex reactor × segment)
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from ui_components import page_header_with_help, show_warning_to_upload_data

COMPARATIVE_HELP = """
**Comparative Plots — Workflow**

Use this page to compare growth metrics across user-defined groups of reactors.

**Step 1 — Ensure analysis has been run**
Run either *Batch Growth Analysis* or *Turbidostat Growth Analysis* first so that
summary statistics are available.

**Step 2 — Assign groups**
Edit the table below to assign a group label to each reactor (or reactor × window
pair for turbidostat data). Groups can be any text or number, e.g. `Control`,
`Treatment A`, `1`, `2`.

**Step 3 — Select metrics and plot type**
Choose which growth metrics to visualise and the preferred chart type (bar, box or
violin). Bars show the mean ± 95 % CI; box/violin charts show the full distribution.

**Step 4 — Generate plots**
Click *Generate comparative plots* to render one interactive Plotly chart per
selected metric. Each chart can be downloaded via the camera icon in the top-right
corner.
"""

# ── Metric definitions ──────────────────────────────────────────────────────
METRIC_LABELS: dict[str, str] = {
    "mu_max": "Max Growth Rate µ_max (1/h)",
    "intrinsic_growth_rate": "Intrinsic Growth Rate (1/h)",
    "doubling_time": "Doubling Time (h)",
    "max_od": "Max OD",
    "exp_phase_start": "Lag Phase End / Exp Phase Start (h)",
    "exp_phase_end": "Exp Phase End (h)",
    "time_at_umax": "Time at µ_max (h)",
    "od_at_umax": "OD at µ_max",
    "model_rmse": "Model RMSE",
}

PLOT_TYPES = ["bar", "box", "violin", "strip"]

# ── Page header ─────────────────────────────────────────────────────────────
page_header_with_help("Comparative Plots", COMPARATIVE_HELP)

# ── Session state guard ──────────────────────────────────────────────────────
stats_df_raw: pd.DataFrame | None = st.session_state.get("batch_analysis_summary_df")

if stats_df_raw is None:
    st.info(
        "No analysis results found. "
        "Run **Batch Growth Analysis** or **Turbidostat Growth Analysis** first."
    )
    show_warning_to_upload_data()
    st.stop()

# ── Detect analysis type and build a flat working copy ───────────────────────
is_turbidostat = isinstance(stats_df_raw.index, pd.MultiIndex)

if is_turbidostat:
    # Reset MultiIndex (reactor, segment) into columns so we can work flat.
    stats_df = stats_df_raw.reset_index()
    # reactor column is the first level name ("reactor")
    reactor_col = stats_df_raw.index.names[0]  # "reactor"
    segment_col = stats_df_raw.index.names[1]  # "segment"
    row_label_col = "reactor_segment"
    stats_df[row_label_col] = (
        stats_df[reactor_col].astype(str) + " | " + stats_df[segment_col].astype(str)
    )
    # Group assignment key for unique reactors
    unique_reactors = stats_df[reactor_col].unique().tolist()
    mode_label = "turbidostat"
else:
    stats_df = stats_df_raw.copy()
    stats_df.index.name = stats_df.index.name or "reactor"
    reactor_col = stats_df.index.name
    stats_df = stats_df.reset_index()
    row_label_col = reactor_col
    unique_reactors = stats_df[reactor_col].unique().tolist()
    mode_label = "batch"

st.caption(f"Analysis mode detected: **{mode_label}**  |  {len(unique_reactors)} reactor(s) found.")

# ── Step 1: Assign groups ────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Step 1. Assign Groups to Reactors")
    st.markdown(
        "Edit the **Group** column to label each reactor. "
        "Reactors sharing the same label will be compared together in the plots."
    )

    # Build a default group assignment table (one row per unique reactor)
    group_state_key = "comparative_group_assignments"
    if group_state_key not in st.session_state:
        default_groups = pd.DataFrame(
            {
                "Reactor": unique_reactors,
                "Group": [""] * len(unique_reactors),
            }
        )
        st.session_state[group_state_key] = default_groups

    # Refresh if the set of reactors has changed (e.g. new analysis run)
    stored: pd.DataFrame = st.session_state[group_state_key]
    if set(stored["Reactor"].tolist()) != set(unique_reactors):
        st.session_state[group_state_key] = pd.DataFrame(
            {
                "Reactor": unique_reactors,
                "Group": [""] * len(unique_reactors),
            }
        )

    edited_groups = st.data_editor(
        st.session_state[group_state_key],
        key="comparative_group_editor",
        width="content",
        num_rows="fixed",
        column_config={
            "Reactor": st.column_config.TextColumn("Reactor", disabled=True),
            "Group": st.column_config.TextColumn(
                "Group",
                help="Enter a group label (e.g. 'Control', 'Treatment A', '1', '2'). "
                "Reactors with the same label will be plotted together.",
                width="medium",
            ),
        },
    )
    st.session_state[group_state_key] = edited_groups

    # Download the group table
    st.download_button(
        label="Download group assignments as CSV",
        data=edited_groups.to_csv(index=False).encode("utf-8"),
        file_name="reactor_group_assignments.csv",
        mime="text/csv",
        icon=":material/download:",
    )

# ── Step 2: Select metrics and plot type ─────────────────────────────────────
with st.container(border=True):
    st.subheader("Step 2. Configure Plots")

    available_metrics = [m for m in METRIC_LABELS if m in stats_df.columns]
    if not available_metrics:
        st.warning("No recognised growth metrics found in the analysis results.")
        st.stop()

    col_left, col_right = st.columns(2)
    with col_left:
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            options=available_metrics,
            default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics,
            format_func=lambda m: METRIC_LABELS.get(m, m),
        )
    with col_right:
        plot_type = st.selectbox("Plot type", PLOT_TYPES, index=0)

    generate = st.button(
        "Generate comparative plots",
        type="primary",
        width="stretch",
    )

# ── Step 3: Render plots ─────────────────────────────────────────────────────
if not generate:
    st.stop()

if not selected_metrics:
    st.warning("Please select at least one metric to plot.")
    st.stop()

st.session_state[group_state_key] = edited_groups

# Merge group assignments into the flat stats_df
group_map: dict[str, str] = dict(
    zip(edited_groups["Reactor"].tolist(), edited_groups["Group"].tolist())
)
stats_df["Group"] = stats_df[reactor_col].map(group_map).fillna("")

# Warn if any reactors have no group assigned
ungrouped = stats_df.loc[stats_df["Group"] == "", reactor_col].unique().tolist()
if ungrouped:
    st.warning(
        f"{len(ungrouped)} reactor(s) have no group assigned and will be labelled "
        f"'(ungrouped)': {', '.join(str(r) for r in ungrouped)}"
    )
    stats_df["Group"] = stats_df["Group"].replace("", "(ungrouped)")

with st.container(border=True):
    st.subheader("Step 3. Comparative Plots")
    if is_turbidostat:
        st.caption(
            "Turbidostat mode: each point represents one growth window "
            "(reactor × dilution segment). Reactors are grouped as assigned above."
        )
    for metric in selected_metrics:
        metric_label = METRIC_LABELS.get(metric, metric)

        # Drop rows where the metric is NaN (no growth / bad fit)
        plot_df = stats_df[[reactor_col, "Group", metric]].dropna(subset=[metric]).copy()
        try:
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        except Exception:
            pass
        plot_df = plot_df.dropna(subset=[metric])

        if plot_df.empty:
            st.info(f"No valid data for **{metric_label}** — skipping.")
            continue

        hover_data = {reactor_col: True, "Group": True, metric: True}

        if plot_type == "bar":
            # Bar chart: mean ± 95 % CI with individual data points overlaid
            agg = (
                plot_df.groupby("Group")[metric]
                .agg(mean="mean", sem="sem", count="count")
                .reset_index()
            )
            agg["ci95"] = agg["sem"] * 1.96
            fig = px.bar(
                agg,
                x="Group",
                y="mean",
                color="Group",
                error_y="ci95",
                labels={"Group": "Group", "mean": metric_label},
                title=metric_label,
            )
            # Overlay individual data points as a strip
            strip_fig = px.strip(
                plot_df,
                x="Group",
                y=metric,
                color="Group",
                hover_data=hover_data,
            )
            for trace in strip_fig.data:
                trace.showlegend = False
                fig.add_trace(trace)

        elif plot_type == "box":
            fig = px.box(
                plot_df,
                x="Group",
                y=metric,
                color="Group",
                points="all",
                hover_data=hover_data,
                labels={"Group": "Group", metric: metric_label},
                title=metric_label,
            )

        elif plot_type == "violin":
            fig = px.violin(
                plot_df,
                x="Group",
                y=metric,
                color="Group",
                box=True,
                points="all",
                hover_data=hover_data,
                labels={"Group": "Group", metric: metric_label},
                title=metric_label,
            )

        else:  # strip
            fig = px.strip(
                plot_df,
                x="Group",
                y=metric,
                color="Group",
                hover_data=hover_data,
                labels={"Group": "Group", metric: metric_label},
                title=metric_label,
            )

        fig.update_layout(
            showlegend=(plot_type == "bar"),
            xaxis_title="Group",
            yaxis_title=metric_label,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, width="stretch")
