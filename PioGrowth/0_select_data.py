"""Interactive data exclusion page for df_rolling.

Allows per-column lasso/box selection of points to set to NaN in df_rolling.
Selected points are excluded and the modified df_rolling is stored back in session state.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth.analyze

SELECT_DATA_HELP = """
Interactively exclude data points from the rolling-median table:

1. Select a column (reactor/sample) from the dropdown.
2. Lasso or box-select unwanted points directly on the plot.
3. Click **Exclude selected points** to set them to NaN in `df_rolling`.
4. Use **Reset column** to restore the original values for the active column.
5. Use **Reset all columns** to undo all exclusions.

Changes are applied immediately and carry through to downstream analysis pages.
"""

page_header_with_help("Select / Exclude Data", SELECT_DATA_HELP)

# ── Session state guards ────────────────────────────────────────────────────
df_rolling: pd.DataFrame = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")
use_elapsed_time = bool(st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", True))

if df_rolling is None:
    show_warning_to_upload_data()
    st.stop()

# backup so individual columns can be restored
if "select_data_original_df_rolling" not in st.session_state:
    # ! key needs to be invalidated if df_rolling is overwritten on upload_data page
    st.session_state["select_data_original_df_rolling"] = df_rolling.copy()

original_df: pd.DataFrame = st.session_state["select_data_original_df_rolling"]

# ── Column selector ─────────────────────────────────────────────────────────
columns = df_rolling.columns.tolist()
selected_col = st.selectbox("Select column (reactor / sample)", columns)

# ── Callbacks ───────────────────────────────────────────────────────────────
CHART_KEY = f"select_data_lasso_{selected_col}"


def _get_selected_times() -> np.ndarray:
    return piogrowth.analyze.get_selected_times_from_event(
        st.session_state.get(CHART_KEY)
    )


def _on_exclude():
    xs = _get_selected_times()
    if xs.size == 0:
        st.toast("No points selected – draw a lasso/box selection first.", icon="⚠️")
        return

    df: pd.DataFrame = st.session_state["df_rolling"]
    valid_xs = df.index.intersection(xs)
    if len(valid_xs) == 0:
        st.toast("No matching data points found.", icon="⚠️")
        return

    df.loc[valid_xs, selected_col] = np.nan
    st.session_state["df_rolling"] = df
    st.toast(f"Set {len(valid_xs)} point(s) to NaN in '{selected_col}'.", icon="✅")


def _on_reset_col():
    df: pd.DataFrame = st.session_state["df_rolling"]
    df[selected_col] = original_df[selected_col]
    st.session_state["df_rolling"] = df
    st.toast(f"Restored original values for '{selected_col}'.", icon="🔄")


def _on_reset_all():
    st.session_state["df_rolling"] = original_df.copy()
    st.toast("Restored all columns to original values.", icon="🔄")


# ── Action buttons ──────────────────────────────────────────────────────────
btn_cols = st.columns([2, 2, 2, 4])
with btn_cols[0]:
    st.button("Exclude selected points", on_click=_on_exclude, type="primary")
with btn_cols[1]:
    st.button(f"Reset column '{selected_col}'", on_click=_on_reset_col)
with btn_cols[2]:
    st.button("Reset all columns", on_click=_on_reset_all)

# ── Re-read df_rolling after potential mutation ─────────────────────────────
df_rolling = st.session_state["df_rolling"]

# ── Build plot data ──────────────────────────────────────────────────────────
series: pd.Series = df_rolling[selected_col]
original_series: pd.Series = original_df[selected_col]

t_all = series.index.to_numpy(dtype=float)

# x-axis: elapsed time label or raw index
if use_elapsed_time and start_time is not None:
    x_label = "Elapsed time (h)"
    x_values = t_all
else:
    x_label = "Time index"
    x_values = t_all

# Split into current (non-NaN) and excluded (NaN introduced vs original)
is_excluded = original_series.notna() & series.isna()
is_current = series.notna()

fig = go.Figure()

# Excluded points (shown in red at original y value)
if is_excluded.any():
    fig.add_trace(
        go.Scatter(
            x=x_values[is_excluded.values],
            y=original_series.values[is_excluded.values],
            mode="markers",
            marker=dict(color="red", size=6, symbol="x"),
            name="Excluded (NaN)",
        )
    )

# Active points
fig.add_trace(
    go.Scatter(
        x=x_values[is_current.values],
        y=series.values[is_current.values],
        mode="markers",
        marker=dict(color="steelblue", size=5),
        name="Active data",
    )
)

fig.update_layout(
    dragmode="lasso",
    uirevision=f"select_data_{selected_col}",
    xaxis=dict(title=x_label, showgrid=False),
    yaxis=dict(title="Rolling median OD", showgrid=False),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20),
    height=500,
    title=f"Column: {selected_col}",
)

st.plotly_chart(
    fig,
    key=CHART_KEY,
    selection_mode=["lasso", "box"],
    on_select="rerun",
    width="stretch",
)

# ── Info about current exclusions ───────────────────────────────────────────
n_excluded_col = int(is_excluded.sum())
n_total_col = int(original_series.notna().sum())
st.caption(
    f"**{selected_col}**: {n_excluded_col} / {n_total_col} non-NaN points excluded."
)

# ── Overview table ──────────────────────────────────────────────────────────
with st.expander("Exclusion summary (all columns)"):
    summary_rows = []
    for col in df_rolling.columns:
        orig = original_df[col]
        curr = df_rolling[col]
        excluded_count = int((orig.notna() & curr.isna()).sum())
        total_count = int(orig.notna().sum())
        summary_rows.append(
            {"Column": col, "Excluded": excluded_count, "Total": total_count}
        )
    st.dataframe(pd.DataFrame(summary_rows).set_index("Column"), width="stretch")

# ── Preview of modified df_rolling ──────────────────────────────────────────
with st.expander("Preview modified df_rolling"):
    st.dataframe(df_rolling, width="stretch")
