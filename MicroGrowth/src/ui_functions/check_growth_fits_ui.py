"""UI helpers for the Check Growth Fits page."""

import growthcurves.plot as gc_plot
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from growthcurves.inference import bad_fit_stats, is_no_growth

from src.functions.check_growth_fits import (
    _add_lasso_selected_points,
    _analyse_series_with_plate_params,
    _collect_lasso_series,
    _cycle,
    _delete_well_from_plate,
    _sg_params_for_plate,
    analyse_well,
    update_growth_stats_from_lasso,
)
from src.functions.plotting_functions import _finite_sorted_xy, plot_derivative_metric


def _format_growth_stats_table(gs: dict) -> pd.DataFrame:
    """Format growth stats into a displayable table."""
    gs = gs or {}

    # Define metrics to display with nice labels and formatting
    metrics = [
        ("fit_method", "Fit Method", lambda x: str(x) if x else "sliding_window"),
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
        # Backward compatibility for previously exported/serialized stats.
        if key == "mu_max" and value is None:
            value = gs.get("specific_growth_rate")
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
    growth_method = plate_params.get("growth_method", "Spline")

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
            ("window_points", "Window size (points)", lambda x: str(int(x)))
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
            if value is None:
                value = plate_params.get("spline_s")
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
        st.info(f"No data for {well}")
        return np.nan, np.nan, True

    t = processed["Time"]
    t_min, t_max = float(t.min()), float(t.max())
    step = float(max((t_max - t_min) / 200.0, 0.01))

    growth_stats = (plate.get("growth_stats") or {}).setdefault(well, {})
    plate_params = (plate or {}).get("params") or {}
    growth_method = plate_params.get("growth_method", "Spline")

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
        st.session_state[_rp_min_od_key] = float(
            plate_params.get("min_od_increase", 0.05)
        )
        st.session_state[_rp_min_gr_key] = float(
            plate_params.get("min_growth_rate", 0.001)
        )
        st.session_state[_rp_min_snr_key] = float(
            plate_params.get("min_signal_to_noise", 1.0)
        )
        st.session_state[_rp_min_dp_key] = int(plate_params.get("min_data_points", 5))
        st.session_state[_rp_window_key] = int(plate_params.get("window_points", 15))
        smooth_default = (
            str(plate_params.get("smooth", plate_params.get("spline_s", "fast")))
            .strip()
            .lower()
        )
        st.session_state[_rp_smooth_key] = (
            "slow"
            if smooth_default == "auto"
            else smooth_default if smooth_default in {"fast", "slow"} else "fast"
        )
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
            st.warning("All OD values are ≤ 0 - no growth detected")
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
    growth_stats["exp_phase_start"] = float(lag_end)
    growth_stats["exp_phase_end"] = float(exp_end)
    growth_stats["max_od"] = float(max_od)

    c1, c2, c3 = st.columns(3)

    def _on_no_growth():
        """Mark the well as no-growth and reset widgets."""
        new_stats = bad_fit_stats()
        new_stats["no_growth_reason"] = "manually assigned"
        growth_stats.clear()
        growth_stats.update(new_stats)
        _sync_widgets_from_growth_stats()

    def _on_reanalyse():
        """Re-run analysis using the current popover parameters.

        If lasso points are selected, re-analyses only that subset.
        Otherwise re-analyses all data for the well.
        """
        params_override = {
            "min_od_increase": float(
                st.session_state.get(
                    _rp_min_od_key, plate_params.get("min_od_increase", 0.05)
                )
            ),
            "min_growth_rate": float(
                st.session_state.get(
                    _rp_min_gr_key, plate_params.get("min_growth_rate", 0.001)
                )
            ),
            "min_signal_to_noise": float(
                st.session_state.get(
                    _rp_min_snr_key, plate_params.get("min_signal_to_noise", 1.0)
                )
            ),
            "min_data_points": int(
                st.session_state.get(
                    _rp_min_dp_key, plate_params.get("min_data_points", 5)
                )
            ),
        }
        if growth_method == "Sliding Window":
            params_override["window_points"] = int(
                st.session_state.get(
                    _rp_window_key, plate_params.get("window_points", 15)
                )
            )
        elif growth_method == "Spline":
            smooth_mode = st.session_state.get(_rp_smooth_key, "fast")
            if smooth_mode == "manual":
                params_override["smooth"] = float(
                    st.session_state.get(_rp_spline_s_key) or 0.0
                )
            else:
                params_override["smooth"] = smooth_mode

        effective_p = dict(plate_params)
        effective_p.update(params_override)

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
            new_stats = analyse_well(plate, well, params_override=params_override)
            growth_stats.clear()
            growth_stats.update(new_stats)

        # Record the params actually used for display in the stats table
        analysis_params = {
            "min_od_increase": effective_p.get("min_od_increase"),
            "min_growth_rate": effective_p.get("min_growth_rate"),
            "min_signal_to_noise": effective_p.get("min_signal_to_noise"),
            "min_data_points": effective_p.get("min_data_points"),
        }
        if growth_method == "Sliding Window":
            analysis_params["window_points"] = effective_p.get("window_points")
        elif growth_method == "Spline":
            _sv = (
                str(
                    growth_stats.get(
                        "smooth",
                        effective_p.get("smooth", effective_p.get("spline_s", "fast")),
                    )
                )
                .strip()
                .lower()
            )
            analysis_params["smooth"] = (
                "slow" if _sv == "auto" else _sv if _sv in {"fast", "slow"} else "fast"
            )
        growth_stats["_analysis_params"] = analysis_params

        _sync_widgets_from_growth_stats()

    def _on_defaults():
        """Rerun analysis with plate default settings on all data, clearing any lasso selection."""
        _init_reanalyse_params()
        new_stats = analyse_well(plate, well, params_override=None)
        growth_stats.clear()
        growth_stats.update(new_stats)
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
        params = plate.setdefault("params", {})
        remove_wells = params.get("remove_wells", False)
        if remove_wells is False or not remove_wells:
            params["remove_wells"] = [well]
        elif well not in remove_wells:
            params["remove_wells"] = list(remove_wells) + [well]

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
                    "Window size (points)",
                    min_value=3,
                    step=1,
                    key=_rp_window_key,
                    help="Number of data points in each sliding window used to estimate growth rate.",
                )
            elif growth_method == "Spline":
                st.radio(
                    "Spline fitting mode",
                    options=["fast", "slow", "manual"],
                    key=_rp_smooth_key,
                    horizontal=True,
                    format_func=lambda v: v.capitalize(),
                    help=(
                        "Fast uses auto-default smoothing with OD weights. "
                        "Slow uses weighted GCV smoothing and is typically slower. "
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
                            "Spline smoothing factor λ. "
                            "Larger values produce a smoother (less wiggly) spline."
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

    # Get available wells and sort them in A1-H12 order
    all_standard_wells = [f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)]
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
    st.session_state["winfit_well"]

    def _move_well(step: int):
        """Move the active well forward/backward."""
        st.session_state["winfit_well"] = _cycle(
            wells, st.session_state.get("winfit_well", wells[0]), step
        )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
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
        with st.container(border=True):
            plate = plates[plate_id]
            key = f"{plate_id}_{well}"

            lag_end, exp_end, no_growth = _phase_controls(plate, well, key=key)
    if no_growth:
        return

    sg_w, sg_p, _ = _sg_params_for_plate(plates, plate_id)
    processed = plate.get("processed_data") or {}
    growth_stats = plate.get("growth_stats") or {}
    fit_parameters = plate.get("fit_parameters") or {}
    gs = growth_stats.get(well) or {}

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
            f"{gs.get('mu_max', gs.get('specific_growth_rate', 0))}_"
            f"{gs.get('max_od', 0)}_"
            f"{gs.get('exp_phase_start', 0)}_"
            f"{gs.get('exp_phase_end', 0)}_"
            f"{gs.get('model_rmse', 0)}_"
            f"{gs.get('_lasso_update_time', '')}_"
            f"{id(gs.get('_analysis_params'))}"
        )

        with stats_exp_col:
            with st.popover(f"Growth Statistics — {well}", width="stretch"):
                stats_df = _format_growth_stats_table(gs)
                st.dataframe(
                    stats_df,
                    width="stretch",
                    hide_index=True,
                    key=f"stats_{table_key_base}",
                )

        with params_exp_col:
            with st.popover(f"Analysis Parameters — {well}", width="stretch"):
                well_data = (plate.get("processed_data") or {}).get(well)
                n_total = (
                    len(well_data)
                    if well_data is not None and not well_data.empty
                    else None
                )
                params_df = _format_analysis_params_table(
                    gs, plate.get("params") or {}, n_total=n_total
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

    # Get the processed data for this well
    d = processed.get(well)
    if d is not None and not d.empty:
        # Get time and OD data
        t_raw, y_raw = _finite_sorted_xy(
            d["Time"].to_numpy(), d["baseline_corrected"].to_numpy()
        )

        if t_raw.size > 0:
            # Use hours throughout (no display conversion)
            t_display = t_raw

            # Determine scale
            scale = "log" if log_scale else "linear"

            # Create base plot using growthcurves - this matches the notebook pattern
            fig_main = gc_plot.create_base_plot(t_display, y_raw, scale=scale)

            # Highlight lasso-selected points (default: all points)
            selected_times = gs.get("_used_fit_times")
            if not selected_times:
                selected_times = t_raw.tolist()
            fig_main = _add_lasso_selected_points(
                fig_main,
                t_raw,
                y_raw,
                selected_times,
                scale=scale,
            )

            # Annotate plot with growth stats if available
            if not is_no_growth(gs) and gs:
                # Get fit result from session state
                fit_result = fit_parameters.get(well)

                # Pass the stored growth stats and fit result directly
                # No need to reconstruct - use the original values from the fit
                fig_main = gc_plot.annotate_plot(
                    fig_main,
                    fit_result=fit_result,
                    stats=gs,
                    show_fitted_curve=show_fitted_model,
                    show_phase_boundaries=show_phase_boundaries,
                    show_crosshairs=show_umax_point,
                    show_od_max_line=show_max_od,
                    show_n0_line=show_baseline_od,
                    show_umax_marker=show_umax_point,
                    show_tangent=show_tangent,
                    scale=scale,
                )

            # Update axis labels
            time_label = "Time (hours)"
            y_label = "ln(OD600)" if log_scale else "OD600 (baseline-corrected)"
            # Set x-axis range to exactly match data range (removes gap at y-axis)
            fig_main.update_xaxes(
                title=time_label,
                showgrid=False,
                type="linear",
                range=[float(t_display.min()), float(t_display.max())],
            )
            fig_main.update_yaxes(title=y_label, showgrid=False)

            # Apply layout for lasso selection functionality
            fig_main.update_layout(
                uirevision="keep",
                dragmode="lasso",
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=20),
                height=600,
            )
        else:
            fig_main = go.Figure()
    else:
        fig_main = go.Figure()

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
    st.plotly_chart(fig_dndt, width="stretch")
    st.plotly_chart(fig_mu, width="stretch")
