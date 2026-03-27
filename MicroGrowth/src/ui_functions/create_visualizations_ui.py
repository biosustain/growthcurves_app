"""UI helpers for the Create Visualizations page."""

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_sortables import sort_items

from src.functions.visualization_functions import _unique_preserve_order
from src.styling import data_grid_style


def ui_growth_selection_container(plates: dict) -> dict:
    """Render the sample selection container and return selection context."""
    # Build options once per rerun.
    rows = []
    for pid, p in plates.items():
        by_name = {}
        for well, nm in (p.get("name") or {}).items():
            nm = (nm or "").strip()
            if not nm or nm in {"False"} or nm.upper().startswith("BLANK"):
                continue
            by_name.setdefault(nm, []).append(well)

        for nm, wells in by_name.items():
            rows.append((f"{pid}||{nm}", pid, nm, ", ".join(sorted(wells))))

    opt = (
        pd.DataFrame(rows, columns=["_id", "Plate", "Sample Name", "Wells"])
        .drop_duplicates("_id")
        .sort_values(["Plate", "Sample Name"], kind="stable")
        .reset_index(drop=True)
    )

    has_split = opt["Sample Name"].astype(str).str.contains("_", regex=False).any()
    if has_split and not opt.empty:
        sc = opt["Sample Name"].astype(str).str.split("_", n=1, expand=True)
        opt["Strain"] = sc[0]
        opt["Condition"] = sc[1].fillna("")

    ids = opt["_id"].tolist()

    # -----------------------------
    # Selection state
    # -----------------------------
    sel_key = "growth_combined_sel"
    sel = st.session_state.setdefault(sel_key, {})
    st.session_state[sel_key] = {sid: bool(sel.get(sid, False)) for sid in ids}
    sel = st.session_state[sel_key]
    grid_ver_key = "sample_selection_grid_ver"
    st.session_state.setdefault(grid_ver_key, 0)

    def _selected_ids():
        return [sid for sid in ids if sel.get(sid, False)]

    def _selected_opt_rows(sel_ids: list[str]) -> pd.DataFrame:
        if not sel_ids:
            return opt.iloc[0:0].copy()
        return opt[opt["_id"].isin(sel_ids)].copy()

    # -----------------------------
    # UI: Step 1 selection (outside form so grid changes rerun)
    # -----------------------------
    with st.container(border=True):
        st.header("Step 1. Select Samples for Visualization")

        # Apply styling for data grid (moved to styling.py)
        data_grid_style()

        # Prepare dataframe for display with selection column
        if has_split:
            display_cols = ["Plate", "Sample Name", "Strain", "Condition", "Wells"]
        else:
            display_cols = ["Plate", "Sample Name", "Wells"]

        display_df = opt[display_cols + ["_id"]].copy()

        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_selection(
            "multiple",
            use_checkbox=True,
            rowMultiSelectWithClick=True,
        )
        gb.configure_column("_id", hide=True)
        gb.configure_columns(display_cols, editable=False)
        if display_cols:
            gb.configure_column(
                display_cols[0],
                headerCheckboxSelection=True,
                checkboxSelection=True,
            )
        grid_options = gb.build()
        pre_selected_rows = [idx for idx, sid in enumerate(ids) if sel.get(sid, False)]
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            pre_selected_rows=pre_selected_rows,
            fit_columns_on_grid_load=True,
            height=400,
            width="100%",
            key=f"sample_selection_grid_{st.session_state[grid_ver_key]}",
        )
        selected_rows = grid_response.get("selected_rows")
        if selected_rows is None:
            selected_ids = set()
        elif isinstance(selected_rows, pd.DataFrame):
            selected_ids = set(
                selected_rows.get("_id", pd.Series([], dtype=str)).tolist()
            )
        elif isinstance(selected_rows, list):
            if selected_rows and isinstance(selected_rows[0], dict):
                selected_ids = {
                    row.get("_id") for row in selected_rows if row.get("_id")
                }
            else:
                selected_ids = {row for row in selected_rows if isinstance(row, str)}
        else:
            selected_ids = set()
        for sid in ids:
            sel[sid] = sid in selected_ids

        sel_ids = _selected_ids()
        sel_opt = _selected_opt_rows(sel_ids)
        sel_sample_names = (
            _unique_preserve_order(sel_opt["Sample Name"].astype(str).tolist())
            if not sel_opt.empty
            else []
        )

    return {
        "opt": opt,
        "has_split": has_split,
        "ids": ids,
        "sel_ids": sel_ids,
        "sel_opt": sel_opt,
        "sel_sample_names": sel_sample_names,
    }


@st.fragment
def ui_growth_stats_controls_container(has_split: bool, sel_opt: pd.DataFrame) -> dict:
    """Render growth stats controls and return form selections."""
    # -----------------------------
    # Order state (stats x-axis + legend)
    # -----------------------------
    x_order_key = "growth_stats_x_order"
    x_order_sig_key = "growth_stats_x_order_sig"
    x_order_ver_key = "growth_stats_x_order_ver"

    leg_order_key = "growth_stats_legend_order"
    leg_order_sig_key = "growth_stats_legend_order_sig"
    leg_order_ver_key = "growth_stats_legend_order_ver"

    st.session_state.setdefault(x_order_key, [])
    st.session_state.setdefault(x_order_ver_key, 0)
    st.session_state.setdefault(leg_order_key, [])
    st.session_state.setdefault(leg_order_ver_key, 0)

    with st.container(border=True):
        st.header("Step 2. option a) Plot Growth Statistics")

        x_choices = ["Sample Name"]
        group_choices = ["None"]
        if has_split:
            x_choices += ["Strain", "Condition"]
            group_choices += ["Strain", "Condition"]

        cA, cB = st.columns([1, 1])
        x_col = cA.selectbox(
            "X-axis column",
            options=x_choices,
            index=0,
            key="growth_stats_x_col",
        )
        legend_group = cB.selectbox(
            "Legend grouping",
            options=group_choices,
            index=0,
            key="growth_stats_legend_group",
        )
        legend_col = None if legend_group == "None" else legend_group

        x_vals = (
            _unique_preserve_order(sel_opt[x_col].astype(str).tolist())
            if (not sel_opt.empty and x_col in sel_opt.columns)
            else []
        )
        legend_vals = (
            _unique_preserve_order(sel_opt[legend_col].astype(str).tolist())
            if (legend_col and not sel_opt.empty and legend_col in sel_opt.columns)
            else []
        )

        # drag ordering: x-axis
        cur_x_order = [v for v in st.session_state[x_order_key] if v in x_vals]
        for v in x_vals:
            if v not in cur_x_order:
                cur_x_order.append(v)
        st.session_state[x_order_key] = cur_x_order

        x_sig = (x_col, tuple(x_vals))
        if st.session_state.get(x_order_sig_key) != x_sig:
            st.session_state[x_order_sig_key] = x_sig
            st.session_state[x_order_ver_key] += 1

        if x_vals:
            st.markdown("**Drag to set x-axis order:**")
            st.session_state[x_order_key] = sort_items(
                st.session_state[x_order_key],
                key=f"growth_stats_x_sortable_{st.session_state[x_order_ver_key]}",
            )

        # drag ordering: legend
        if legend_col:
            cur_leg_order = [
                v for v in st.session_state[leg_order_key] if v in legend_vals
            ]
            for v in legend_vals:
                if v not in cur_leg_order:
                    cur_leg_order.append(v)
            st.session_state[leg_order_key] = cur_leg_order

            leg_sig = (legend_col, tuple(legend_vals))
            if st.session_state.get(leg_order_sig_key) != leg_sig:
                st.session_state[leg_order_sig_key] = leg_sig
                st.session_state[leg_order_ver_key] += 1

            if legend_vals:
                st.markdown("**Drag to set legend order:**")
                st.session_state[leg_order_key] = sort_items(
                    st.session_state[leg_order_key],
                    key=f"growth_stats_leg_sortable_{st.session_state[leg_order_ver_key]}",
                )

    x_ordered = [v for v in st.session_state[x_order_key] if v in x_vals]
    legend_ordered = (
        [v for v in st.session_state[leg_order_key] if v in legend_vals]
        if legend_col
        else []
    )

    return {
        "x_col": x_col,
        "legend_col": legend_col,
        "x_ordered": x_ordered,
        "legend_ordered": legend_ordered,
    }


@st.fragment
def ui_growth_curves_controls_container(
    max_t: float, sel_sample_names: list[str]
) -> dict:
    """Render growth curves controls and return form selections."""
    # -----------------------------
    # Order state (curves sample order: mean+reps)
    # -----------------------------
    curves_order_key = "growth_curves_sample_order"
    curves_order_sig_key = "growth_curves_sample_order_sig"
    curves_order_ver_key = "growth_curves_sample_order_ver"
    st.session_state.setdefault(curves_order_key, [])
    st.session_state.setdefault(curves_order_ver_key, 0)

    with st.container(border=True):
        st.header("Step 2. option b) Plot Growth Curves")

        curves_t0, curves_t1 = st.slider(
            "Mean/replicates plot time window (hours)",
            0.0,
            max_t,
            (0.0, min(72.0, max_t)),
            step=0.5,
            key="growth_curves_time_window",
        )

        # drag ordering: sample names (specific to mean+reps)
        cur_curves_order = [
            v for v in st.session_state[curves_order_key] if v in sel_sample_names
        ]
        for v in sel_sample_names:
            if v not in cur_curves_order:
                cur_curves_order.append(v)
        st.session_state[curves_order_key] = cur_curves_order

        curves_sig = tuple(sel_sample_names)
        if st.session_state.get(curves_order_sig_key) != curves_sig:
            st.session_state[curves_order_sig_key] = curves_sig
            st.session_state[curves_order_ver_key] += 1

        if sel_sample_names:
            st.markdown("**Drag to set Sample Name order (mean/replicates):**")
            st.session_state[curves_order_key] = sort_items(
                st.session_state[curves_order_key],
                key=f"growth_curves_sortable_{st.session_state[curves_order_ver_key]}",
            )

    curves_ordered = [
        v for v in st.session_state[curves_order_key] if v in sel_sample_names
    ]

    return {
        "curves_t0": curves_t0,
        "curves_t1": curves_t1,
        "curves_ordered": curves_ordered,
    }
