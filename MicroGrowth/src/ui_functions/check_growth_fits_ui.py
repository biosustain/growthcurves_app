"""UI helpers for the Check Growth Fits page."""

import numpy as np
import pandas as pd
import streamlit as st
from growthcurves.inference import bad_fit_stats, is_no_growth
from src.functions.check_growth_fits import (
    _analyse_series_with_plate_params,
    _collect_lasso_series,
    _cycle,
    _delete_well_from_plate,
    _sg_params_for_plate,
    analyse_well,
    reanalysis_params,
    update_growth_stats_from_lasso,
)
from src.functions.common import toast_once
from src.functions.constants import plate_format
from src.functions.data_processing import mark_changed
from src.functions.plotting_functions import plot_derivative_metric, plot_well_curve


def _format_growth_stats_table(gs: dict) -> pd.DataFrame:
    """Format growth stats into a displayable table."""
    gs = gs or {}

    # Define metrics to display with nice labels and formatting
    metrics = [
        ("fit_method", "Fit Method", lambda x: str(x) if x else "--"),
        ("model_rmse", "RMSE", lambda x: f"{float(x):.5f}" if pd.notna(x) else "--"),
        ("max_od", "Maximum OD", lambda x: f"{float(x):.4f}" if pd.notna(x) else "--"),
        (
            "mu_max",
            "Maximum Growth Rate (1/h)",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "intrinsic_growth_rate",
            "Intrinsic Growth Rate (1/h)",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "time_at_umax",
            "Time at Max Growth (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "od_at_umax",
            "OD at Max Growth",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "exp_phase_start",
            "Lag Phase End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "exp_phase_end",
            "Exponential Phase End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
    ]

    rows = []
    if is_no_growth(gs):
        reason = gs.get("no_growth_reason", "--")
        rows.append({"Metric": "No growth reason", "Value": reason})

    for key, label, formatter in metrics:
        value = gs.get(key)
        try:
            formatted_value = formatter(value) if value is not None else "--"
        except (ValueError, TypeError):
            formatted_value = "--"
        rows.append({"Metric": label, "Value": formatted_value})

    return pd.DataFrame(rows)


def _format_analysis_params_table(
    gs: dict, plate_params: dict, n_total: int | None = None
) -> pd.DataFrame:
    """Format the analysis parameters actually used into a displayable table.

    Falls back to plate_params when no custom analysis has been run, so the table
    always reflects the actual values used.
    """
    rows = []
    # Use stored custom params if present, otherwise fall back to plate defaults
    analysis_params = gs.get("_analysis_params") or {}
    used_fit_times = gs.get("_used_fit_times")
    growth_method = plate_params["growth_method"]

    n_selected = len(used_fit_times) if used_fit_times else n_total
    total_str = str(n_total) if n_total is not None else "?"
    selected_str = str(n_selected) if n_selected is not None else "?"
    rows.append(
        {"Parameter": "Data subset (points)", "Value": f"{selected_str}/{total_str}"}
    )

    common_params = [
        ("min_od_increase", "Min OD increase", lambda x: f"{float(x):.4f}"),
        ("min_growth_rate", "Min growth rate (1/h)", lambda x: f"{float(x):.5f}"),
        ("min_signal_to_noise", "Min signal-to-noise", lambda x: f"{float(x):.2f}"),
        ("min_data_points", "Min data points", lambda x: str(int(x))),
    ]
    method_params = []
    if growth_method == "Sliding Window":
        method_params = [
            ("window_points", "Minimum window size (points)", lambda x: str(int(x)))
        ]
    elif growth_method == "Spline":
        method_params = [("smooth", "Spline mode", lambda v: str(v).capitalize())]

    for param_name, plabel, formatter in common_params + method_params:
        if param_name == "smooth" and growth_method == "Spline":
            value = analysis_params.get(param_name)
            if value is None:
                value = gs.get("smooth")
            if value is None:
                value = plate_params.get(param_name)
        else:
            value = (
                analysis_params.get(param_name)
                if analysis_params.get(param_name) is not None
                else plate_params.get(param_name)
            )
        if value is not None:
            try:
                rows.append({"Parameter": plabel, "Value": formatter(value)})
            except (ValueError, TypeError):
                pass

    # Analysis window and phase boundary method come from the fit result stored in gs
    fit_metrics = [
        ("t_window_start", "Analysis window start (h)", lambda x: f"{float(x):.2f}"),
        ("t_window_end", "Analysis window end (h)", lambda x: f"{float(x):.2f}"),
        ("phase_boundary_method", "Phase boundary method", lambda x: str(x)),
    ]
    for key, plabel, formatter in fit_metrics:
        value = gs.get(key)
        if value is not None:
            try:
                rows.append({"Parameter": plabel, "Value": formatter(value)})
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(rows)


def _phase_controls(plate: dict, well: str, *, key: str):
    """Render phase/OD sliders and actions; writes into growth_stats."""

    processed = (plate.get("processed_data") or {}).get(well)
    if processed is None or processed.empty:
        toast_once(f"No data for {well}")
        return np.nan, np.nan, True

    t = processed["Time"]
    t_min, t_max = float(t.min()), float(t.max())
    step = float(max((t_max - t_min) / 200.0, 0.01))

    growth_stats = (plate.get("growth_stats") or {}).setdefault(well, {})
    plate_params = plate["params"]
    growth_method = plate_params["growth_method"]

    ss_key = f"phase__{key}"
    maxod_key = f"maxod__{key}"
    lasso_time_key = f"lasso_time__{key}"

    # Session-state keys for re-analysis parameter overrides (scoped to this well)
    _rp_min_od_key = f"rp_min_od__{key}"
    _rp_min_gr_key = f"rp_min_gr__{key}"
    _rp_min_snr_key = f"rp_min_snr__{key}"
    _rp_min_dp_key = f"rp_min_dp__{key}"
    _rp_window_key = f"rp_window__{key}"
    _rp_smooth_key = f"rp_smooth__{key}"
    _rp_spline_s_key = f"rp_spline_s__{key}"

    def _sync_widgets_from_growth_stats():
        """Sync widget state from the current growth_stats dict."""
        lag0 = growth_stats.get("exp_phase_start")
        exp0 = growth_stats.get("exp_phase_end")
        lag0 = float(lag0) if pd.notna(lag0) else t_min
        exp0 = float(exp0) if pd.notna(exp0) else t_min
        lag0 = max(t_min, min(t_max, lag0))
        exp0 = max(t_min, min(t_max, exp0))
        st.session_state[ss_key] = (lag0, exp0)

        # Clamp max_od to the actual max in the processed data to avoid slider errors
        actual_max_od = float(max(processed["baseline_corrected"]))
        stored_max_od = float(growth_stats.get("max_od", 0.0))
        # Handle edge case where all OD values are <= 0
        if actual_max_od > 0:
            st.session_state[maxod_key] = min(stored_max_od, actual_max_od)
        else:
            st.session_state[maxod_key] = 0.0

        # Track the last lasso update time we've synced
        st.session_state[lasso_time_key] = growth_stats.get("_lasso_update_time")

    def _init_reanalyse_params():
        """Reset re-analysis param overrides to the original plate values."""
        st.session_state[_rp_min_od_key] = float(plate_params["min_od_increase"])
        st.session_state[_rp_min_gr_key] = float(plate_params["min_growth_rate"])
        st.session_state[_rp_min_snr_key] = float(plate_params["min_signal_to_noise"])
        st.session_state[_rp_min_dp_key] = int(plate_params["min_data_points"])
        st.session_state[_rp_window_key] = int(plate_params["window_points"])
        st.session_state[_rp_smooth_key] = "auto"
        st.session_state[_rp_spline_s_key] = None

    # Initialise re-analysis params the first time this well is shown
    if _rp_min_od_key not in st.session_state:
        _init_reanalyse_params()

    # Sync widgets if they don't exist OR if growth_stats was updated by lasso selection
    current_lasso_time = growth_stats.get("_lasso_update_time")
    last_synced_time = st.session_state.get(lasso_time_key)

    if ss_key not in st.session_state or current_lasso_time != last_synced_time:
        _sync_widgets_from_growth_stats()

    st.write("")  # just adds some space

    c1, c2 = st.columns(2)
    with c1:
        lag_end, exp_end = st.slider(
            "Set phase boundaries (hours)",
            t_min,
            t_max,
            step=step,
            key=ss_key,
        )

    with c2:
        # Handle edge case where all OD values are 0 or negative
        actual_max_od = max(processed["baseline_corrected"])
        if actual_max_od <= 0:
            toast_once(f"{well}: all OD values are ≤ 0 - no growth detected")
            max_od = 0.0
        else:
            max_od = st.slider(
                "Set maximum OD",
                0.0,
                actual_max_od,
                step=actual_max_od / 120,
                key=maxod_key,
            )

    # Persist boundaries unless we're deleting
    edited = {
        "exp_phase_start": float(lag_end),
        "exp_phase_end": float(exp_end),
        "max_od": float(max_od),
    }
    if any(growth_stats.get(k) != v for k, v in edited.items()):
        growth_stats.update(edited)
        mark_changed(plate)

    c1, c2, c3 = st.columns(3)

    def _on_no_growth():
        """Mark the well as no-growth and reset widgets."""
        new_stats = bad_fit_stats()
        new_stats["no_growth_reason"] = "manually assigned"
        growth_stats.clear()
        growth_stats.update(new_stats)
        mark_changed(plate)
        _sync_widgets_from_growth_stats()

    def _on_reanalyse():
        """Re-run analysis using the current popover parameters.

        If lasso points are selected, re-analyses only that subset.
        Otherwise re-analyses all data for the well.
        """
        effective_p, analysis_params = reanalysis_params(plate, key)

        used_fit_times = growth_stats.get("_used_fit_times")
        if used_fit_times:
            # Re-analyse only the lasso-selected subset with the custom params
            processed = plate.get("processed_data", {}).get(well)
            refit_t, refit_y = _collect_lasso_series(
                processed, np.array(used_fit_times)
            )
            if refit_t.size >= 2:
                fit, fit_result = _analyse_series_with_plate_params(
                    refit_t, refit_y, effective_p
                )
                growth_stats.clear()
                growth_stats.update(fit)
                growth_stats["_used_fit_times"] = (
                    used_fit_times  # preserve lasso selection
                )
                if fit_result is not None:
                    plate.setdefault("fit_parameters", {})[well] = fit_result
                else:
                    plate.get("fit_parameters", {}).pop(well, None)
            else:
                used_fit_times = None  # fall through to full analysis

        if not used_fit_times:
            new_stats = analyse_well(plate, well, params_override=effective_p)
            growth_stats.clear()
            growth_stats.update(new_stats)

        # Record the params actually used for display in the stats table
        growth_stats["_analysis_params"] = analysis_params
        mark_changed(plate)
        _sync_widgets_from_growth_stats()

    def _on_defaults():
        """Rerun analysis with plate default settings on all data, clearing any lasso selection."""
        _init_reanalyse_params()
        new_stats = analyse_well(plate, well, params_override=None)
        growth_stats.clear()
        growth_stats.update(new_stats)
        mark_changed(plate)
        _sync_widgets_from_growth_stats()

    def _on_delete():
        """Remove the well from the plate and clear widget state."""
        _delete_well_from_plate(plate, well)
        for k in (
            ss_key,
            maxod_key,
            lasso_time_key,
            _rp_min_od_key,
            _rp_min_gr_key,
            _rp_min_snr_key,
            _rp_min_dp_key,
            _rp_window_key,
            _rp_smooth_key,
            _rp_spline_s_key,
        ):
            st.session_state.pop(k, None)

        # Update params to include this well in remove_wells list
        remove_wells = list(plate["params"]["remove_wells"] or [])
        if well not in remove_wells:
            plate["params"]["remove_wells"] = remove_wells + [well]

    with c1:
        st.button(
            "No Growth",
            width="stretch",
            type="primary",
            key=f"nogrowth__{key}",
            on_click=_on_no_growth,
        )

    with c2:
        with st.popover("Re-analyse", width="stretch"):
            st.markdown("**No-growth thresholds**")
            st.number_input(
                "Min OD increase",
                min_value=0.0,
                step=0.01,
                format="%.3f",
                key=_rp_min_od_key,
                help="Minimum total OD increase required to classify a well as growing.",
            )
            st.number_input(
                "Min growth rate (1/h)",
                min_value=0.0,
                step=0.0001,
                format="%.4f",
                key=_rp_min_gr_key,
                help="Minimum maximum specific growth rate required to classify a well as growing.",
            )
            st.number_input(
                "Min signal-to-noise",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                key=_rp_min_snr_key,
                help="Minimum signal-to-noise ratio required to classify a well as growing.",
            )
            st.number_input(
                "Min data points",
                min_value=1,
                step=1,
                key=_rp_min_dp_key,
                help="Minimum number of valid data points required for analysis.",
            )
            if growth_method == "Sliding Window":
                st.number_input(
                    "Minimum window size (points)",
                    min_value=3,
                    step=1,
                    key=_rp_window_key,
                    help="Minimum number of data points per sliding window; the window length adapts to the noise in the data.",
                )
            elif growth_method == "Spline":
                st.radio(
                    "Spline fitting mode",
                    options=["auto", "manual"],
                    key=_rp_smooth_key,
                    horizontal=True,
                    format_func=lambda v: v.capitalize(),
                    help=(
                        "Auto chooses the smoothing by REML. "
                        "Manual lets you set the smoothing factor (λ) directly."
                    ),
                )
                if st.session_state.get(_rp_smooth_key) == "manual":
                    st.number_input(
                        "Smoothing factor (λ)",
                        min_value=0.0,
                        value=st.session_state.get(_rp_spline_s_key),
                        step=0.01,
                        format="%.4f",
                        key=_rp_spline_s_key,
                        help=(
                            "Spline smoothing factor λ, relative to the data. "
                            "Larger values produce a smoother (less wiggly) spline. "
                            "0 uses Auto."
                        ),
                    )
            btn_col, restore_col = st.columns(2)
            with btn_col:
                st.button(
                    "Re-analyse",
                    type="primary",
                    width="stretch",
                    key=f"reanalyse__{key}",
                    on_click=_on_reanalyse,
                )
            with restore_col:
                st.button(
                    "Defaults",
                    width="stretch",
                    type="primary",
                    key=f"restore_defaults__{key}",
                    on_click=_on_defaults,
                )

    with c3:
        st.button(
            "Exclude from analysis",
            width="stretch",
            type="tertiary",
            key=f"deletewell__{key}",
            on_click=_on_delete,
        )

    st.write("")  # just adds some space

    return float(lag_end), float(exp_end), False


@st.fragment
def ui_window_fits_well_editor(plates: dict):
    """Render the well editor UI for interactive window fit adjustments."""
    plate_ids = sorted(plates)

    # Initialize plate selection if not set
    if "winfit_plate" not in st.session_state:
        st.session_state["winfit_plate"] = plate_ids[0]

    # Get wells with data from the current plate
    current_plate_id = st.session_state.get("winfit_plate", plate_ids[0])
    current_plate = plates.get(current_plate_id, {})
    processed_data = current_plate.get("processed_data") or {}

    # Get available wells and sort them in plate order (A1, A2, ...)
    all_standard_wells = list(plate_format(current_plate).wells)
    wells = [w for w in all_standard_wells if w in processed_data]

    # If no wells with data, fall back to standard ordering
    if not wells:
        wells = all_standard_wells

    # Ensure the selected well exists in the current plate's wells
    if (
        "winfit_well" not in st.session_state
        or st.session_state["winfit_well"] not in wells
    ):
        st.session_state["winfit_well"] = wells[0]

    def _move_well(step: int):
        """Move the active well forward/backward."""
        st.session_state["winfit_well"] = _cycle(
            wells, st.session_state.get("winfit_well", wells[0]), step
        )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True, key="card_check_growth_fits_1"):
            plate_col, popover_col, toggle_col1 = st.columns(
                [2, 0.9, 0.9], vertical_alignment="bottom", gap="small"
            )
            with plate_col:
                plate_id = st.selectbox("Plate", plate_ids, key="winfit_plate")
            with popover_col:
                with st.popover("Annotations", width="stretch"):
                    show_phase_boundaries = st.toggle(
                        "Phase boundaries",
                        value=st.session_state.get(
                            "show_phase_boundaries_toggle", True
                        ),
                        key="show_phase_boundaries_toggle",
                    )
                    show_umax_point = st.toggle(
                        "Max growth rate point",
                        value=st.session_state.get("show_umax_point_toggle", True),
                        key="show_umax_point_toggle",
                    )
                    show_max_od = st.toggle(
                        "Max OD",
                        value=st.session_state.get("show_max_od_toggle", True),
                        key="show_max_od_toggle",
                    )
                    show_baseline_od = st.toggle(
                        "Baseline OD",
                        value=st.session_state.get("show_baseline_od_toggle", True),
                        key="show_baseline_od_toggle",
                    )
                    show_tangent = st.toggle(
                        "Tangent line at max growth",
                        value=st.session_state.get("show_tangent_toggle", False),
                        key="show_tangent_toggle",
                    )
                    show_fitted_model = st.toggle(
                        "Fitted model curve",
                        value=st.session_state.get("show_fitted_model_toggle", True),
                        key="show_fitted_model_toggle",
                    )
            with toggle_col1:
                log_scale = st.toggle(
                    "Log scale",
                    value=st.session_state.get("log_scale_toggle", False),
                    key="log_scale_toggle",
                )

            prev, mid, next_ = st.columns([2, 4, 2], vertical_alignment="bottom")
            with prev:
                st.button(
                    "",
                    width="stretch",
                    on_click=_move_well,
                    args=(-1,),
                    key="well_prev",
                    shortcut="Left",
                    type="primary",
                )
            with mid:
                well = st.selectbox(
                    "Well",
                    wells,
                    key="winfit_well",
                    index=wells.index(st.session_state["winfit_well"]),
                )
            with next_:
                st.button(
                    "",
                    width="stretch",
                    on_click=_move_well,
                    args=(+1,),
                    key="well_next",
                    shortcut="Right",
                    type="primary",
                )

    with col2:
        with st.container(border=True, key="card_check_growth_fits_2"):
            plate = plates[plate_id]
            key = f"{plate_id}_{well}"

            lag_end, exp_end, no_growth = _phase_controls(plate, well, key=key)
    if no_growth:
        return

    sg_w, sg_p = _sg_params_for_plate(plates, plate_id)
    gs = (plate.get("growth_stats") or {}).get(well) or {}

    # Display growth status indicator and stats table
    status_col, expander_col = st.columns([2, 5])

    with status_col:
        # Visual indicator for growth detection
        if is_no_growth(gs):
            reason = gs.get("no_growth_reason", "No growth detected")
            st.container(border=True).error(f"**No Growth:** {reason}")
        else:
            st.container(border=True).success("**Growth Detected**")

    with expander_col:
        stats_exp_col, params_exp_col = st.columns(2)

        table_key_base = (
            f"{plate_id}_{well}_"
            f"{gs.get('mu_max', 0)}_"
            f"{gs.get('max_od', 0)}_"
            f"{gs.get('exp_phase_start', 0)}_"
            f"{gs.get('exp_phase_end', 0)}_"
            f"{gs.get('model_rmse', 0)}_"
            f"{gs.get('_lasso_update_time', '')}_"
            f"{id(gs.get('_analysis_params'))}"
        )

        with stats_exp_col:
            with st.popover(f"Growth Statistics - {well}", width="stretch"):
                stats_df = _format_growth_stats_table(gs)
                st.dataframe(
                    stats_df,
                    width="stretch",
                    hide_index=True,
                    key=f"stats_{table_key_base}",
                )

        with params_exp_col:
            with st.popover(f"Analysis Parameters - {well}", width="stretch"):
                well_data = (plate.get("processed_data") or {}).get(well)
                n_total = (
                    len(well_data)
                    if well_data is not None and not well_data.empty
                    else None
                )
                params_df = _format_analysis_params_table(
                    gs, plate["params"], n_total=n_total
                )
                st.dataframe(
                    params_df,
                    width="stretch",
                    hide_index=True,
                    key=f"params_{table_key_base}",
                )

        st.caption(
            "💡 **Tip:** Click and drag on the growth curve plot below to select a subset of data points. "
            "The analysis will be automatically rerun using only the selected points to recalculate growth parameters."
        )

    st.divider()

    chart_key = f"lasso_fit_{plate_id}_{well}"

    fig_main = plot_well_curve(
        plate,
        well,
        log_scale=log_scale,
        phase_boundaries=show_phase_boundaries,
        umax_point=show_umax_point,
        max_od=show_max_od,
        baseline_od=show_baseline_od,
        tangent=show_tangent,
        fitted_model=show_fitted_model,
    ).update_layout(uirevision="keep", dragmode="lasso")

    # Grey cards with transparent plot backgrounds, as on the Upload & Analyse page
    with st.container(border=True, key="card_check_growth_fits_3"):
        st.plotly_chart(
            fig_main,
            key=chart_key,
            selection_mode="lasso",
            on_select=lambda: update_growth_stats_from_lasso(
                plates, plate_id, well, chart_key
            ),
            width="stretch",
        )

    # Show derivative plots
    fig_dndt = plot_derivative_metric(
        plate, well, metric="dndt", sg_window=sg_w, sg_poly=sg_p, gs=gs
    )
    fig_mu = plot_derivative_metric(
        plate, well, metric="mu", sg_window=sg_w, sg_poly=sg_p, gs=gs
    )
    for i, fig in enumerate((fig_dndt, fig_mu), start=4):
        with st.container(border=True, key=f"card_check_growth_fits_{i}"):
            st.plotly_chart(fig, width="stretch")
