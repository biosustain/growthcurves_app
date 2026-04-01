"""Batch analysis page for interactive growth-model fitting.

Session state entries used in this module
-----------------------------------------

Many state entries are prefixed with "batch_" to indicate they are used in this batch
analysis page and to avoid naming conflicts with other pages. Some entries are dynamic
keys that are reactor-specific (reactor parameter): `batch_rp_{param}__{reactor}`.

Global/shared input data:
- USE_ELAPSED_TIME_FOR_PLOTS: Global plotting mode flag.
- df_time_map: Optional time mapping table from uploaded data.
- df_rolling: Rolling-median OD dataframe used as analysis input.
- start_time: Experiment start timestamp used in plot labels/download text.
- DEFAULT_XLABEL_TPS: Default x-axis label for timepoint mode.
- DEFAULT_XLABEL_REL: Default x-axis label for elapsed-time mode.

Batch analysis outputs/cache:
- batch_analysis_summary_df: Per-reactor summary statistics dataframe.
- batch_analysis_options: Effective analysis options from Step 1.
- batch_analysis_fit_cache: Dict[reactor -> fit result] for fast redraw/reuse.
- batch_selected_fit_times: Dict[reactor -> selected time list] for subset refits.
- batch_analysis_used_params: Dict[reactor -> analysis parameter overrides].
  - used for modification for re-analysis to store the effective parameters to use
    (manually modified)
ToDo: Check if all of these are used and needed

UI selection/toggle state:
- batch_selected_reactor: Active reactor/sample shown in Step 2.
- batch_show_phase_boundaries: Toggle for phase-boundary annotations.
- batch_show_umax_point: Toggle for max growth point/marker annotations.
- batch_show_max_od: Toggle for max OD annotation.
- batch_show_baseline_od: Toggle for baseline OD annotation.
- batch_show_tangent: Toggle for tangent-at-u_max annotation.
- batch_show_fitted_model: Toggle for fitted model curve.
- batch_log_scale: Toggle for linear vs log plotting.

Per-reactor dynamic keys (created from selected reactor id):
- batch_phase__{reactor}: Tuple[lag_end, exp_end] slider state.
- batch_maxod__{reactor}: Max OD slider state.
- batch_rp_min_od__{reactor}: Re-analysis threshold (min OD increase).
- batch_rp_min_gr__{reactor}: Re-analysis threshold (min growth rate).
- batch_rp_min_snr__{reactor}: Re-analysis threshold (min signal-to-noise).
- batch_rp_min_dp__{reactor}: Re-analysis threshold (min data points).
- batch_rp_window__{reactor}: Re-analysis window size (sliding window method).
- batch_rp_smooth__{reactor}: Re-analysis spline smooth mode (spline method).
- batch_lasso_fit_{reactor}: Plotly lasso selection event payload key.
"""

import inspect
import logging

import growthcurves as gc
import growthcurves.plot as gc_plot
import numpy as np
import pandas as pd
import streamlit as st
from buttons import create_download_button
from growthcurves_options import (
    render_parameter_calculation_table_upload_style,
    render_upload_style_analysis_options,
)

# from names import summary_mapping
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth.analyze
from piogrowth.fit_spline import get_smoothing_range

logger = logging.getLogger(__name__)
if st.session_state.get("debug_mode", False):
    logger.setLevel(logging.DEBUG)

st.session_state.setdefault("batch_analysis_options", {})


########################################################################################
### Callbacks for buttons and interactions #############################################
def _on_no_growth(
    selected_reactor: str,
    fit_cache: dict,
    selected_fit_times_map: dict,
    used_params_map: dict,
    t_all: np.ndarray,
    stats_df: pd.DataFrame,
    batch_options: dict,
):
    new_stats = gc.inference.bad_fit_stats()
    new_stats["no_growth_reason"] = "manually assigned"
    new_stats["elapsed_time"] = np.nan
    new_stats["model_name"] = batch_options["selected_model"]
    piogrowth.analyze.update_reactor_stats(stats_df, selected_reactor, new_stats)
    # ? what is fit_cache?
    fit_cache.pop(selected_reactor, None)
    selected_fit_times_map[selected_reactor] = t_all.tolist()
    # ? is this needed?
    used_params_map[selected_reactor] = piogrowth.analyze.default_analysis_params(
        batch_options
    )
    st.session_state["batch_analysis_summary_df"] = stats_df
    st.session_state["batch_analysis_fit_cache"] = fit_cache
    st.session_state["batch_selected_fit_times"] = selected_fit_times_map
    st.session_state["batch_analysis_used_params"] = used_params_map


# ? why is this needed?
def _build_effective_options_from_widgets(
    batch_options,
    rp_min_od_key,
    rp_min_gr_key,
    rp_min_snr_key,
    rp_min_dp_key,
    rp_window_key,
    rp_smooth_key,
) -> tuple[dict, dict]:
    options_refit = dict(batch_options)
    options_refit["min_od_increase"] = float(st.session_state[rp_min_od_key])
    options_refit["min_growth_rate"] = float(st.session_state[rp_min_gr_key])
    options_refit["min_signal_to_noise"] = float(st.session_state[rp_min_snr_key])
    options_refit["min_data_points"] = int(st.session_state[rp_min_dp_key])

    method = piogrowth.analyze.growth_method_from_model(batch_options["selected_model"])
    # Only update relevant for method
    if method == "Sliding Window":
        options_refit["window_points"] = int(st.session_state[rp_window_key])
    elif method == "Spline":
        options_refit["smooth_mode"] = piogrowth.analyze.normalize_smooth(
            st.session_state[rp_smooth_key]
        )
    return options_refit, None


## ! _on_defaults and _on_reanalyse have a lot of duplicated code. refactor
def _on_defaults(
    t_all: np.ndarray,
    y_all: np.ndarray,
    batch_options: dict,
    selected_reactor: str,
    stats_df: pd.DataFrame,
    fit_cache: dict,
    selected_fit_times_map: dict,
    used_params_map: dict,
    rp_min_od_key: str,
    rp_min_gr_key: str,
    rp_min_snr_key: str,
    rp_min_dp_key: str,
    rp_window_key: str,
    rp_smooth_key: str,
):
    # reset reactor parameters (rp) to defaults from batch_options
    st.session_state[rp_min_od_key] = float(batch_options.get("min_od_increase", 0.05))
    st.session_state[rp_min_gr_key] = float(batch_options.get("min_growth_rate", 0.01))
    st.session_state[rp_min_snr_key] = float(
        batch_options.get("min_signal_to_noise", 1.0)
    )
    st.session_state[rp_min_dp_key] = int(batch_options.get("min_data_points", 50))
    st.session_state[rp_window_key] = int(batch_options.get("window_points", 150))
    st.session_state[rp_smooth_key] = piogrowth.analyze.normalize_smooth(
        batch_options.get("smooth_mode", "fast")
    )
    _fit(
        t=t_all,
        y=y_all,
        analysis_params=batch_options,
        selected_reactor=selected_reactor,
        fit_cache=fit_cache,
        stats_df=stats_df,
        selected_fit_times_map=selected_fit_times_map,
        used_params_map=used_params_map,
        exp_phase_start=None,
        exp_phase_end=None,
        max_od=None,
    )


def _on_reanalyse(
    t_all,
    y_all,
    batch_options,
    selected_reactor,
    stats_df,
    fit_cache,
    selected_fit_times_map,
    used_params_map,
    series,
    lag_end,
    exp_end,
    max_od,
    rp_min_od_key,
    rp_min_gr_key,
    rp_min_snr_key,
    rp_min_dp_key,
    rp_window_key,
    rp_smooth_key,
):
    # update batch_options with used_params_map
    _batch_options = dict(batch_options)
    # Will be updated anyways with all the keys
    # _batch_options.update(used_params_map.get(selected_reactor, {}))
    options_refit, _ = _build_effective_options_from_widgets(
        _batch_options,
        rp_min_od_key,
        rp_min_gr_key,
        rp_min_snr_key,
        rp_min_dp_key,
        rp_window_key,
        rp_smooth_key,
    )
    used_times = selected_fit_times_map.get(selected_reactor)
    if used_times:
        t_refit, y_refit = piogrowth.analyze.collect_selected_series(
            series, np.asarray(used_times, dtype=float)
        )
        if t_refit.size < 2:
            t_refit, y_refit = t_all, y_all
            used_times = t_all.tolist()
    else:
        t_refit, y_refit = t_all, y_all
        used_times = t_all.tolist()
    # ToDo: lag_end and exp_end are not updated in table after manuel adjustment
    _fit(
        t=t_refit,
        y=y_refit,
        analysis_params=options_refit,
        selected_reactor=selected_reactor,
        fit_cache=fit_cache,
        stats_df=stats_df,
        selected_fit_times_map=selected_fit_times_map,
        used_params_map=used_params_map,
        exp_phase_start=lag_end,
        exp_phase_end=exp_end,
        max_od=max_od,
    )


def _on_lasso_select(
    chart_key: str,
    series,
    selected_reactor,
    fit_cache,
    stats_df,
    selected_fit_times_map,
    used_params_map,
    batch_options,
    rp_min_od_key,
    rp_min_gr_key,
    rp_min_snr_key,
    rp_min_dp_key,
    rp_window_key,
    rp_smooth_key,
):
    xs = piogrowth.analyze.get_selected_times_from_event(
        st.session_state.get(chart_key)
    )
    if xs.size < 2:
        return
    refit_t, refit_y = piogrowth.analyze.collect_selected_series(series, xs)
    if refit_t.size < 2:
        return
    options_refit, _ = _build_effective_options_from_widgets(
        batch_options=batch_options,
        rp_min_od_key=rp_min_od_key,
        rp_min_gr_key=rp_min_gr_key,
        rp_min_snr_key=rp_min_snr_key,
        rp_min_dp_key=rp_min_dp_key,
        rp_window_key=rp_window_key,
        rp_smooth_key=rp_smooth_key,
    )
    _fit(
        t=refit_t,
        y=refit_y,
        analysis_params=options_refit,
        selected_reactor=selected_reactor,
        fit_cache=fit_cache,
        stats_df=stats_df,
        selected_fit_times_map=selected_fit_times_map,
        used_params_map=used_params_map,
    )


def _fit(
    t,
    y,
    analysis_params,
    selected_reactor,
    fit_cache,
    stats_df,
    selected_fit_times_map,
    used_params_map,
    exp_phase_start=None,
    exp_phase_end=None,
    max_od=None,
):
    """Fits a single time series based on analysis_params, checks for now
    growth.

    Has several update sub-steps which could be factored out.
    """
    # print("analysis params in _fit")
    # print(analysis_params)
    fit_, stats_ = piogrowth.analyze.fit_single_series(t, y, analysis_params)
    #  allows manual overwriting of these three parameters based on _reanalyse panel
    if exp_phase_start is not None:
        stats_["exp_phase_start"] = float(exp_phase_start)
    if exp_phase_end is not None:
        stats_["exp_phase_end"] = float(exp_phase_end)
    if max_od is not None:
        stats_["max_od"] = float(max_od)
    fit_cache[selected_reactor] = fit_
    res_no_growth = gc.inference.detect_no_growth(
        t=t,
        N=y,
        growth_stats=stats_,
        min_data_points=analysis_params["min_data_points"],
        min_signal_to_noise=analysis_params["min_signal_to_noise"],
        min_od_increase=analysis_params["min_od_increase"],
        min_growth_rate=analysis_params["min_growth_rate"],
    )
    if res_no_growth["is_no_growth"]:
        logger.debug(res_no_growth)
        stats_ = gc.inference.bad_fit_stats()
        stats_["no_growth_reason"] = res_no_growth.get("reason", "No growth detected")
        stats_["elapsed_time"] = np.nan
        stats_["model_name"] = analysis_params["selected_model"]
    fit_cache[selected_reactor] = fit_
    piogrowth.analyze.update_reactor_stats(stats_df, selected_reactor, stats_)
    selected_fit_times_map[selected_reactor] = t.tolist()
    used_params_map[selected_reactor] = analysis_params
    st.session_state["batch_analysis_summary_df"] = stats_df
    st.session_state["batch_analysis_fit_cache"] = fit_cache
    st.session_state["batch_selected_fit_times"] = selected_fit_times_map
    st.session_state["batch_analysis_used_params"] = used_params_map


# other callbacks
def _cycle(items, current, step):
    """Cycle forward/backward through a list."""
    if not items:
        return current
    try:
        idx = items.index(current)
    except ValueError:
        idx = 0
    return items[(idx + step) % len(items)]


def _move_reactor(step: int, reactors):
    st.session_state["batch_selected_reactor"] = _cycle(
        reactors,
        st.session_state.get("batch_selected_reactor", reactors[0]),
        step,
    )


def _on_exclude(
    selected_reactor: str,
    fit_cache: dict,
    selected_fit_times_map: dict,
    used_params_map: dict,
    stats_df: pd.DataFrame,
    df_rolling: pd.DataFrame,
):
    fit_cache.pop(selected_reactor, None)
    selected_fit_times_map.pop(selected_reactor, None)
    used_params_map.pop(selected_reactor, None)
    if selected_reactor in stats_df.index:
        stats_df.drop(index=selected_reactor, inplace=True)
    remaining = [c for c in df_rolling.columns if c in stats_df.index]
    if remaining:
        st.session_state["batch_selected_reactor"] = remaining[0]
    st.session_state["batch_analysis_summary_df"] = stats_df
    st.session_state["batch_analysis_fit_cache"] = fit_cache
    st.session_state["batch_selected_fit_times"] = selected_fit_times_map
    st.session_state["batch_analysis_used_params"] = used_params_map
    st.rerun()


########################################################################################
# state

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_time_map = st.session_state.get("df_time_map")
no_data_uploaded = st.session_state.get("df_rolling") is None
df_rolling = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")

DEFAULT_XLABEL_TPS = st.session_state.get("DEFAULT_XLABEL_TPS", "Timepoints (rounded)")
DEFAULT_XLABEL_REL = st.session_state.get("DEFAULT_XLABEL_REL", "Elapsed time (hours)")
NON_PARAMETRIC_FIT_PARAMS = set(
    inspect.signature(gc.non_parametric.fit_non_parametric).parameters
)
########################################################################################
# UI

BATCH_HELP = """
Run growth-model analysis on the uploaded rolling-median OD data.

Workflow:
1. Configure analysis options and run analysis
2. Review linear/log plots and optionally lasso-select points to re-fit
"""


BATCH_HELP = f"{BATCH_HELP}\n\n---\n\n{piogrowth.analyze.load_method_notes_markdown()}"


page_header_with_help("Analyze growth experiment in batch mode", BATCH_HELP)

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

smoothing_range = get_smoothing_range(len(df_rolling))

### Form ###############################################################################
# default batch analysis options (globally set)
with st.container(border=True):
    st.header("Step 1. Configure Analysis Options")
    analysis_options = render_upload_style_analysis_options(
        s_min=smoothing_range.s_min,
        s_max=smoothing_range.s_max,
        min_window_points=15,
        max_window_points=500,
        default_window_points=st.session_state["batch_analysis_options"].get(
            "window_points", 150
        ),
        window_step_size=5,
        min_data_points_default=st.session_state["batch_analysis_options"].get(
            "min_data_points", 50
        ),
        min_signal_to_noise_default=st.session_state["batch_analysis_options"].get(
            "min_signal_to_noise", 1.0
        ),
        min_od_increase_default=st.session_state["batch_analysis_options"].get(
            "min_od_increase", 0.5
        ),
        min_growth_rate_default=st.session_state["batch_analysis_options"].get(
            "min_growth_rate", 0.01
        ),
    )
    render_parameter_calculation_table_upload_style(analysis_options)
    run_analysis = st.button("Run Analysis", type="primary", width="stretch")
    # remember of any changes to options
    st.session_state["batch_analysis_options"] = analysis_options

### Analyse data after form submission    ##############################################
if run_analysis and not no_data_uploaded:
    stats_df_new, fit_cache = piogrowth.analyze.run_model_fitting_on_df_compat(
        df_rolling,
        model_name=analysis_options["selected_model"],
        n_fits=analysis_options["n_fits"],
        spline_s=analysis_options["spline_smoothing_value"],
        smooth_mode=analysis_options.get("smooth_mode", "fast"),
        window_points=analysis_options["window_points"],
        phase_boundary_method=analysis_options["phase_boundary_method"],
        lag_cutoff=analysis_options["lag_cutoff"],
        exp_cutoff=analysis_options["exp_cutoff"],
        min_data_points=analysis_options["min_data_points"],
        min_signal_to_noise=analysis_options["min_signal_to_noise"],
        min_od_increase=analysis_options["min_od_increase"],
        min_growth_rate=analysis_options["min_growth_rate"],
    )

    st.session_state["batch_analysis_summary_df"] = stats_df_new
    st.session_state["batch_analysis_options"] = analysis_options
    st.session_state["batch_analysis_fit_cache"] = fit_cache
    # ?timepoints used for fitting: Is it needed?
    # ! only set if not already set.
    if "batch_selected_fit_times" not in st.session_state:
        st.session_state["batch_selected_fit_times"] = {}
    if "batch_analysis_used_params" not in st.session_state:
        st.session_state["batch_analysis_used_params"] = {}
    # st.session_state.pop("batch_selection_status", None)


stats_df = st.session_state.get("batch_analysis_summary_df")
batch_options = st.session_state.get("batch_analysis_options")


########################################################################################
### Display results and interactive plot ###############################################
if stats_df is None or batch_options is None:
    st.stop()

# Step 2: Review results, interactively select points to re-fit, and update stats_df
#         and fit_cache accordingly
with st.container(border=True):
    st.header("Step 2. Review Results")

    reactors = [col for col in df_rolling.columns if col in stats_df.index]
    if not reactors:
        st.warning("No reactors available to display.")
        st.stop()

    if st.session_state.get("batch_selected_reactor") not in reactors:
        st.session_state["batch_selected_reactor"] = reactors[0]

    selected_fit_times_map = st.session_state.setdefault("batch_selected_fit_times", {})
    fit_cache = st.session_state.setdefault("batch_analysis_fit_cache", {})
    used_params_map = st.session_state.setdefault("batch_analysis_used_params", {})

    # First two boxes in step 2
    # ToDo: Set borders here to remove one level on indentation
    control_col, phase_col = st.columns(2, gap="large")

    ####################################################################################
    # Select the reactor, which type of plot (linear or log) and requested annotations
    with control_col:
        with st.container(border=True):
            reactor_col, popover_col, toggle_col = st.columns(
                [2, 0.9, 0.9], vertical_alignment="bottom", gap="small"
            )
            with reactor_col:
                sample_label = st.session_state.get(
                    "batch_selected_reactor", reactors[0]
                )
                st.caption(f"Sample: {sample_label}")
            with popover_col:
                with st.popover("Annotations", width="stretch"):
                    show_phase_boundaries = st.toggle(
                        "Phase boundaries",
                        value=st.session_state.get("batch_show_phase_boundaries", True),
                        key="batch_show_phase_boundaries",
                    )
                    show_umax_point = st.toggle(
                        "Max growth rate point",
                        value=st.session_state.get("batch_show_umax_point", True),
                        key="batch_show_umax_point",
                    )
                    show_max_od = st.toggle(
                        "Max OD",
                        value=st.session_state.get("batch_show_max_od", True),
                        key="batch_show_max_od",
                    )
                    show_baseline_od = st.toggle(
                        "Baseline OD",
                        value=st.session_state.get("batch_show_baseline_od", True),
                        key="batch_show_baseline_od",
                    )
                    show_tangent = st.toggle(
                        "Tangent line at max growth",
                        value=st.session_state.get("batch_show_tangent", False),
                        key="batch_show_tangent",
                    )
                    show_fitted_model = st.toggle(
                        "Fitted model curve",
                        value=st.session_state.get("batch_show_fitted_model", True),
                        key="batch_show_fitted_model",
                    )
            with toggle_col:
                log_scale = st.toggle(
                    "Log scale",
                    value=st.session_state.get("batch_log_scale", False),
                    key="batch_log_scale",
                )

            prev_col, sel_col, next_col = st.columns(
                [2, 4, 2], vertical_alignment="bottom"
            )
            with prev_col:
                st.button(
                    "",
                    width="stretch",
                    on_click=_move_reactor,
                    args=(-1, reactors),
                    key="batch_reactor_prev",
                    shortcut="Left",
                    type="primary",
                )
            with sel_col:
                selected_reactor = st.selectbox(
                    "Reactor",
                    reactors,
                    key="batch_selected_reactor",
                    index=reactors.index(st.session_state["batch_selected_reactor"]),
                )
            with next_col:
                st.button(
                    "",
                    width="stretch",
                    on_click=_move_reactor,
                    args=(+1, reactors),
                    key="batch_reactor_next",
                    shortcut="Right",
                    type="primary",
                )

    s = df_rolling[selected_reactor].dropna()
    if s.empty:
        st.warning(f"No valid data points for {selected_reactor}.")
        st.stop()
    t_all = s.index.to_numpy(dtype=float)
    y_all = s.to_numpy(dtype=float)
    actual_max_od = float(np.nanmax(y_all)) if y_all.size else 0.0

    phase_key = f"batch_phase__{selected_reactor}"
    maxod_key = f"batch_maxod__{selected_reactor}"
    rp_min_od_key = f"batch_rp_min_od__{selected_reactor}"
    rp_min_gr_key = f"batch_rp_min_gr__{selected_reactor}"
    rp_min_snr_key = f"batch_rp_min_snr__{selected_reactor}"
    rp_min_dp_key = f"batch_rp_min_dp__{selected_reactor}"
    rp_window_key = f"batch_rp_window__{selected_reactor}"
    rp_smooth_key = f"batch_rp_smooth__{selected_reactor}"

    if phase_key not in st.session_state:
        exp_phase_start = piogrowth.analyze.get_reactor_stat(
            stats_df, selected_reactor, "exp_phase_start"
        )
        exp_phase_end = piogrowth.analyze.get_reactor_stat(
            stats_df, selected_reactor, "exp_phase_end"
        )
        lag0 = (
            float(exp_phase_start) if pd.notna(exp_phase_start) else float(t_all.min())
        )
        exp0 = float(exp_phase_end) if pd.notna(exp_phase_end) else float(t_all.max())
        st.session_state[phase_key] = (lag0, exp0)
    if maxod_key not in st.session_state:
        max_od_stat = piogrowth.analyze.get_reactor_stat(
            stats_df, selected_reactor, "max_od"
        )
        default_max_od = float(max_od_stat) if pd.notna(max_od_stat) else actual_max_od
        st.session_state[maxod_key] = (
            min(default_max_od, actual_max_od) if actual_max_od > 0 else 0.0
        )

    if rp_min_od_key not in st.session_state:
        st.session_state[rp_min_od_key] = float(
            batch_options.get("min_od_increase", 0.05)
        )
    if rp_min_gr_key not in st.session_state:
        st.session_state[rp_min_gr_key] = float(
            batch_options.get("min_growth_rate", 0.001)
        )
    if rp_min_snr_key not in st.session_state:
        st.session_state[rp_min_snr_key] = float(
            batch_options.get("min_signal_to_noise", 1.0)
        )
    if rp_min_dp_key not in st.session_state:
        st.session_state[rp_min_dp_key] = int(batch_options.get("min_data_points", 5))
    if rp_window_key not in st.session_state:
        st.session_state[rp_window_key] = int(batch_options.get("window_points", 10))
    if rp_smooth_key not in st.session_state:
        st.session_state[rp_smooth_key] = piogrowth.analyze.normalize_smooth(
            batch_options.get("smooth_mode", "fast")
        )

    # Right top-box in panel for step 2
    with phase_col:
        with st.container(border=True):
            t_min, t_max = float(t_all.min()), float(t_all.max())
            step = float(max((t_max - t_min) / 200.0, 0.01))
            slider_col1, slider_col2 = st.columns(2)
            with slider_col1:
                lag_end, exp_end = st.slider(
                    "Set phase boundaries (hours)",
                    t_min,
                    t_max,
                    step=step,
                    key=phase_key,
                )
            with slider_col2:
                if actual_max_od <= 0:
                    st.warning("All OD values are ≤ 0 - no growth detected")
                    max_od = 0.0
                else:
                    max_od = st.slider(
                        "Set maximum OD",
                        0.0,
                        actual_max_od,
                        step=float(max(actual_max_od / 120, 1e-6)),
                        key=maxod_key,
                    )
            # ! _on_no_growth does not work if this is actived
            # stats_df.loc[selected_reactor, "exp_phase_start"] = float(lag_end)
            # stats_df.loc[selected_reactor, "exp_phase_end"] = float(exp_end)
            # stats_df.loc[selected_reactor, "max_od"] = float(max_od)

            action_col1, action_col2, action_col3 = st.columns(3)
            # Assign no-growth manually
            with action_col1:
                st.button(
                    "No Growth",
                    width="stretch",
                    type="primary",
                    key=f"batch_nogrowth__{selected_reactor}",
                    on_click=_on_no_growth,
                    args=(
                        selected_reactor,
                        fit_cache,
                        selected_fit_times_map,
                        used_params_map,
                        t_all,
                        stats_df,
                        batch_options,
                    ),
                )
            # Reanalysis Actions (parameter setting and trigger)
            with action_col2:
                with st.popover("Re-analyse", width="stretch"):
                    st.markdown("**No-growth thresholds**")
                    st.number_input(
                        "Min OD increase",
                        min_value=0.0,
                        step=0.01,
                        format="%.3f",
                        key=rp_min_od_key,
                    )
                    st.number_input(
                        "Min growth rate (1/h)",
                        min_value=0.0,
                        step=0.0001,
                        format="%.4f",
                        key=rp_min_gr_key,
                    )
                    st.number_input(
                        "Min signal-to-noise",
                        min_value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=rp_min_snr_key,
                    )
                    st.number_input(
                        "Min data points",
                        min_value=1,
                        step=1,
                        key=rp_min_dp_key,
                    )
                    method = piogrowth.analyze.growth_method_from_model(
                        batch_options["selected_model"]
                    )
                    if method == "Sliding Window":
                        st.number_input(
                            "Window size (points)",
                            min_value=3,
                            step=1,
                            key=rp_window_key,
                        )
                    elif method == "Spline":
                        st.radio(
                            "Spline fitting mode",
                            options=["fast", "slow"],
                            key=rp_smooth_key,
                            horizontal=True,
                            format_func=lambda v: v.capitalize(),
                        )
                    btn_col, defaults_col = st.columns(2)
                    with btn_col:
                        st.button(
                            "Re-analyse",
                            type="primary",
                            width="stretch",
                            key=f"batch_reanalyse__{selected_reactor}",
                            on_click=_on_reanalyse,
                            # ! maybe change to kwargs parameter
                            args=(
                                t_all,
                                y_all,
                                batch_options,
                                selected_reactor,
                                stats_df,
                                fit_cache,
                                selected_fit_times_map,
                                used_params_map,
                                s,
                                lag_end,
                                exp_end,
                                max_od,
                                rp_min_od_key,
                                rp_min_gr_key,
                                rp_min_snr_key,
                                rp_min_dp_key,
                                rp_window_key,
                                rp_smooth_key,
                            ),
                        )
                    with defaults_col:
                        st.button(
                            "Defaults",
                            width="stretch",
                            type="primary",
                            key=f"batch_restore_defaults__{selected_reactor}",
                            on_click=_on_defaults,
                            # ! maybe change to kwargs parameter
                            args=(
                                t_all,
                                y_all,
                                batch_options,
                                selected_reactor,
                                stats_df,
                                fit_cache,
                                selected_fit_times_map,
                                used_params_map,
                                # no s, lag_end, exp_end, max_od needed for defaults,
                                # as they are not changed here
                                # ? should be there maybe for consistency?
                                rp_min_od_key,
                                rp_min_gr_key,
                                rp_min_snr_key,
                                rp_min_dp_key,
                                rp_window_key,
                                rp_smooth_key,
                            ),
                        )
            # Exclude timeseries from analysis
            with action_col3:
                st.button(
                    "Exclude from analysis",
                    width="stretch",
                    type="tertiary",
                    key=f"batch_exclude__{selected_reactor}",
                    on_click=_on_exclude,
                    args=(
                        selected_reactor,
                        fit_cache,
                        selected_fit_times_map,
                        used_params_map,
                        stats_df,
                        df_rolling,
                    ),
                )

    selected_fit_times = selected_fit_times_map.get(selected_reactor)
    if not selected_fit_times:
        selected_fit_times = t_all.tolist()

    reactor_fit = fit_cache.get(selected_reactor)
    current_stats = piogrowth.analyze.get_reactor_stats_dict(stats_df, selected_reactor)
    if (
        reactor_fit is None
        and len(t_all) >= 2
        and not piogrowth.analyze.is_bad_fit(current_stats)
    ):
        fit_result, stats_new = piogrowth.analyze.fit_single_series(
            t_all, y_all, batch_options
        )
        fit_cache[selected_reactor] = fit_result
        piogrowth.analyze.update_reactor_stats(stats_df, selected_reactor, stats_new)
        st.session_state["batch_analysis_summary_df"] = stats_df
        st.session_state["batch_analysis_fit_cache"] = fit_cache
        reactor_fit = fit_result

    stats = piogrowth.analyze.get_reactor_stats_dict(stats_df, selected_reactor)

    ####################################################################################
    # Row with growth information and instructions for interaction with the plot
    status_col, expander_col = st.columns([2, 5])
    with status_col:
        # ! to update to use more advanced function.
        if piogrowth.analyze.is_bad_fit(stats):
            reason = stats.get("no_growth_reason", "No growth detected")
            st.container(border=True).error(f"**No Growth:** {reason}")
        else:
            st.container(border=True).success("**Growth Detected**")
    with expander_col:
        stats_exp_col, params_exp_col = st.columns(2)
        table_key_base = (
            f"{selected_reactor}_"
            f"{stats.get('mu_max', stats.get('specific_growth_rate', 0))}_"
            f"{stats.get('max_od', 0)}_"
            f"{stats.get('exp_phase_start', 0)}_"
            f"{stats.get('exp_phase_end', 0)}_"
            f"{stats.get('model_rmse', 0)}_"
            f"{id(used_params_map.get(selected_reactor))}"
        )
        with stats_exp_col:
            with st.popover(f"Growth Statistics — {selected_reactor}", width="stretch"):
                stats_table = piogrowth.analyze.format_growth_stats_table(stats)
                st.dataframe(
                    stats_table,
                    width="stretch",
                    hide_index=True,
                    key=f"batch_stats_{table_key_base}",
                )
        with params_exp_col:
            with st.popover(
                f"Analysis Parameters — {selected_reactor}", width="stretch"
            ):
                params_table = piogrowth.analyze.format_analysis_params_table(
                    stats,
                    batch_options,
                    used_params_map.get(selected_reactor, {}),
                    n_total=len(s),
                    n_selected=len(selected_fit_times),
                )
                st.dataframe(
                    params_table,
                    width="stretch",
                    hide_index=True,
                    key=f"batch_params_{table_key_base}",
                )
        st.caption(
            "💡 **Tip:** Click and drag on the growth curve plot below to select a "
            "subset of data points. The analysis will be automatically rerun using "
            "only the selected points to recalculate growth parameters."
        )

    ####################################################################################
    # Show timeseries plot for selected time series in panel
    st.divider()

    scale = "log" if log_scale else "linear"
    fig = gc_plot.create_base_plot(
        t_all,
        y_all,
        scale=scale,
        xlabel=DEFAULT_XLABEL_REL + f" since start at {start_time}",
        marker_opacity=0.3,
    )
    fig = piogrowth.analyze.overlay_selected_points(
        fig,
        t_all,
        y_all,
        selected_fit_times,
        scale=scale,
    )
    fig = gc_plot.annotate_plot(
        fig,
        fit_result=reactor_fit,
        stats=stats,
        show_fitted_curve=show_fitted_model,
        show_phase_boundaries=show_phase_boundaries,
        show_crosshairs=show_umax_point,
        show_od_max_line=show_max_od,
        show_n0_line=show_baseline_od,
        show_umax_marker=show_umax_point,
        show_tangent=show_tangent,
        scale=scale,
    )
    y_label = "ln(OD600)" if log_scale else "OD600 (baseline-corrected)"
    fig.update_xaxes(
        title="Time (hours)",
        showgrid=False,
        type="linear",
        range=[float(t_all.min()), float(t_all.max())],
    )
    fig.update_yaxes(title=y_label, showgrid=False)
    fig.update_layout(
        uirevision="batch_lasso_keep",
        dragmode="lasso",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        height=600,
    )
    chart_key = f"batch_lasso_fit_{selected_reactor}"

    st.plotly_chart(
        fig,
        key=chart_key,
        selection_mode="lasso",
        on_select=lambda: _on_lasso_select(
            chart_key,
            s,
            selected_reactor,
            fit_cache,
            stats_df,
            selected_fit_times_map,
            used_params_map,
            batch_options,
            rp_min_od_key,
            rp_min_gr_key,
            rp_min_snr_key,
            rp_min_dp_key,
            rp_window_key,
            rp_smooth_key,
        ),
        width="stretch",
    )

# Step 3: Overview of results and option to download summary statistics
with st.container(border=True):

    st.header("Step 3. Overview and Download Results")
    st.write(
        f"The start time was {start_time}. "
        " Timepoints are relative to this start time."
    )
    st.dataframe(stats_df, width="stretch")
    st.write("")

    used_params_map = st.session_state.get("batch_analysis_used_params", {})
    selected_fit_times_map = st.session_state.get("batch_selected_fit_times", {})
    params_table = piogrowth.analyze.build_analysis_params_per_sample_table(
        stats_df=stats_df,
        df_rolling=df_rolling,
        batch_options=batch_options,
        used_params_map=used_params_map,
        selected_fit_times_map=selected_fit_times_map,
    )

    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        create_download_button(
            label="Download rolling median data",
            data=df_rolling.to_csv(index=True).encode("utf-8"),
            file_name="batch_analysis_rolling_median_data.csv",
            disabled=False,
            mime="text/csv",
        )
    with dl_col2:
        create_download_button(
            label="Download summary statistics",
            data=stats_df.to_csv(index=True).encode("utf-8"),
            file_name="batch_analysis_summary_stats.csv",
            disabled=False,
            mime="text/csv",
        )
    with dl_col3:
        create_download_button(
            label="Download analysis parameters",
            data=params_table.to_csv(index=False).encode("utf-8"),
            file_name="batch_analysis_parameters_by_sample.csv",
            disabled=False,
            mime="text/csv",
        )

# Debug option to inspect session state variables related to batch analysis
if st.session_state.get("debug_mode", False):
    with st.expander("Developer inspect (session state)", expanded=False):
        st.write("Used parameters map for re-analysis selection:")
        st.write(st.session_state.get("batch_analysis_used_params", {}))
        st.write("which updates general options from form for a given reactor:")
        st.write(batch_options)
        st.write("Fit cache (contains fitted curve data for each reactor)")
        st.write(st.session_state.get("batch_analysis_fit_cache", {}))
        # ! is this the best format to store the lasso selected points?
        st.write("Selected fit times map for lasso point selection:")
        st.write(st.session_state.get("batch_selected_fit_times", {}))
