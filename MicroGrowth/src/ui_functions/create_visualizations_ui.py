"""UI helpers for the Create Visualizations page."""

import pandas as pd
import streamlit as st
from src.functions.visualization_functions import _unique_preserve_order
from src.styling import data_grid_style
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_sortables import sort_items


def _persistent_selectbox(container, label, options, store_key):
    """Selectbox whose selection survives page navigation via a plain state key."""
    wkey = f"{store_key}_widget"
    stored = st.session_state.get(store_key)
    st.session_state.setdefault(wkey, stored if stored in options else options[0])
    value = container.selectbox(label, options, key=wkey)
    st.session_state[store_key] = value
    return value


def _persistent_sortable(order_key, label, values, sig_extra=None):
    """Drag-sortable list whose order survives navigation; empty values keep the order."""
    sig_key = f"{order_key}_sig"
    ver_key = f"{order_key}_ver"
    st.session_state.setdefault(order_key, [])
    st.session_state.setdefault(ver_key, 0)
    if values:
        order = [v for v in st.session_state[order_key] if v in values]
        order += [v for v in values if v not in order]
        st.session_state[order_key] = order

        sig = (sig_extra, tuple(values))
        if st.session_state.get(sig_key) != sig:
            st.session_state[sig_key] = sig
            st.session_state[ver_key] += 1

        st.markdown(f"**{label}**")
        st.session_state[order_key] = sort_items(
            st.session_state[order_key],
            key=f"{order_key}_sortable_{st.session_state[ver_key]}",
        )
    return [v for v in st.session_state[order_key] if v in values]


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
        grid_response = AgGrid(
            display_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            height=400,
            width="100%",
            key="sample_selection_grid",
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

        # The grid remounts unchecked on page navigation, so only overwrite the
        # saved selection when the user explicitly confirms it with this button.
        if st.button("Save selection"):
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
    with st.container(border=True):
        st.header("Step 2. option a) Plot Growth Statistics")

        x_choices = ["Sample Name"]
        group_choices = ["None"]
        if has_split:
            x_choices += ["Strain", "Condition"]
            group_choices += ["Strain", "Condition"]

        cA, cB = st.columns([1, 1])
        x_col = _persistent_selectbox(
            cA, "X-axis column", x_choices, "growth_stats_x_col"
        )
        legend_group = _persistent_selectbox(
            cB, "Legend grouping", group_choices, "growth_stats_legend_group"
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

        x_ordered = _persistent_sortable(
            "growth_stats_x_order", "Drag to set x-axis order:", x_vals, x_col
        )
        legend_ordered = _persistent_sortable(
            "growth_stats_legend_order",
            "Drag to set legend order:",
            legend_vals,
            legend_col,
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
    with st.container(border=True):
        st.header("Step 2. option b) Plot Growth Curves")

        # Persist across navigation via a plain key; re-seed widget when absent.
        tw_key = "growth_curves_time_window"
        tw_wkey = f"{tw_key}_widget"
        lo0, hi0 = st.session_state.get(tw_key, (0.0, min(72.0, max_t)))
        lo0 = min(max(float(lo0), 0.0), max_t)
        hi0 = min(max(float(hi0), lo0), max_t)
        st.session_state.setdefault(tw_wkey, (lo0, hi0))
        curves_t0, curves_t1 = st.slider(
            "Mean/replicates plot time window (hours)",
            0.0,
            max_t,
            step=0.5,
            key=tw_wkey,
        )
        st.session_state[tw_key] = (curves_t0, curves_t1)

        curves_ordered = _persistent_sortable(
            "growth_curves_sample_order",
            "Drag to set Sample Name order (mean/replicates):",
            sel_sample_names,
        )

    return {
        "curves_t0": curves_t0,
        "curves_t1": curves_t1,
        "curves_ordered": curves_ordered,
    }
