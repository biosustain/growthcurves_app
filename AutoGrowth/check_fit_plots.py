"""Check Growth Fits plots, copied unchanged from the MicroGrowth app.

Function bodies are verbatim copies of
MicroGrowth/src/functions/plotting_functions.py (style_like_info_plots,
style_info_axes, _finite_sorted_xy, _add_lasso_selected_points, plot_well_curve,
plot_derivative_metric) and MicroGrowth/src/functions/check_growth_fits.py
(_match_selected_times). Only the imports differ. Keep them in sync.

The functions take a MicroGrowth-style ``plate`` dict; see ``as_plate``.
"""

import growthcurves.plot as gc_plot
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from growthcurves.inference import (
    compute_first_derivative,
    compute_instantaneous_mu,
    compute_sliding_window_growth_rate,
    is_no_growth,
    smooth,
)
from growthcurves.models import (
    MODEL_REGISTRY,
    evaluate_parametric_model,
    log_to_linear,
    spline_from_params,
)

# Savitzky-Golay defaults of the MicroGrowth app (src/functions/constants.py)
DEFAULT_PARAMS = {
    "sg_window": 15,
    "sg_poly": 2,
}


def as_plate(
    well: str,
    t,
    y,
    *,
    stats: dict | None,
    fit_result: dict | None,
    window_points: int,
    used_fit_times=None,
) -> dict:
    """Wrap one AutoGrowth series in the plate dict the MicroGrowth plots expect."""
    gs = dict(stats or {})
    if used_fit_times is not None:
        gs["_used_fit_times"] = list(used_fit_times)
    return {
        "processed_data": {
            well: pd.DataFrame(
                {
                    "Time": np.asarray(t, dtype=float),
                    "baseline_corrected": np.asarray(y, dtype=float),
                }
            )
        },
        "growth_stats": {well: gs},
        "fit_parameters": {well: fit_result} if fit_result is not None else {},
        "params": {"window_points": int(window_points)},
    }


def _match_selected_times(
    all_t: np.ndarray, selected_times: np.ndarray, *, time_tolerance: float = 0.01
) -> np.ndarray:
    """Return indices in all_t that match selected_times within tolerance.

    The matching tolerance is capped at half the smallest gap between distinct
    timepoints, so a selected time can never fall within tolerance of two
    adjacent timepoints at once (each selection maps to exactly one index).
    """
    if all_t.size == 0 or selected_times.size == 0:
        return np.array([], dtype=int)

    finite_t = all_t[np.isfinite(all_t)]
    # np.unique already returns sorted unique values.
    diffs = np.diff(np.unique(finite_t))
    positive_diffs = diffs[diffs > 0]
    min_spacing = (
        float(np.min(positive_diffs)) if positive_diffs.size else time_tolerance
    )
    effective_tolerance = min(time_tolerance, min_spacing / 2)

    refit_indices = []
    seen = set()
    for sel_t in selected_times:
        distances = np.abs(all_t - sel_t)
        if not np.any(np.isfinite(distances)):
            continue
        idx = int(np.nanargmin(distances))
        if distances[idx] <= effective_tolerance and idx not in seen:
            refit_indices.append(idx)
            seen.add(idx)

    return np.asarray(refit_indices, dtype=int)


# --- info-plot style ----------------------------------------------------------
# Palette of the explanatory figures (info_plots/make_info_plots.py).
INFO_INK = "#262730"
INFO_INK_2 = "#5F6368"
INFO_AXIS = "#9AA0A6"
INFO_REF = "#6B7280"
INFO_CURVE = "#2B7BB9"
INFO_ACCENT = "#E4572E"
INFO_PHASE = "#66BB6A"
INFO_POINTS = "#A7ADB4"
INFO_FADED = "#D9DCE1"

# Trace name -> (colour, line width or marker size)
_INFO_TRACES = {
    "Fitted curve": (INFO_CURVE, 3),
    "umax tangent": (INFO_ACCENT, 2),
    "Raw": (INFO_FADED, 3),
    "Smoothed": (INFO_INK_2, 3),
    "Fitted": (INFO_CURVE, 3),
}


def style_like_info_plots(fig: go.Figure, fit_window=None) -> go.Figure:
    """Restyle a single-well plot to match the explanatory figures on the Upload
    page: their palette for the traces, plus style_info_axes.

    As in the sliding-window figure, points used in the fit (lasso-selected and
    inside fit_window, a (t_min, t_max) pair) are dark grey; all others are light grey.
    """
    for tr in fig.data:
        if tr.name == "Data":  # every point; shows grey where not used
            tr.marker.update(color=INFO_POINTS, size=6, opacity=1)
        elif tr.name == "Selected":
            t0, t1 = (
                np.nan if v is None else float(v) for v in fit_window or (None, None)
            )
            x = np.asarray(tr.x, dtype=float)
            used = (x >= t0) & (x <= t1)
            tr.update(x=x[used], y=np.asarray(tr.y, dtype=float)[used])
            tr.marker.update(color=INFO_INK_2, size=7, line=dict(width=0))
        elif tr.name in _INFO_TRACES:
            color, width = _INFO_TRACES[tr.name]
            if tr.mode == "markers":
                tr.marker.color = color
            else:
                tr.line.color = color
                tr.line.width = width
        elif tr.mode == "markers":  # μmax marker
            tr.marker.update(color=INFO_ACCENT, line=dict(color="white", width=2))
    for s in fig.layout.shapes:  # below the traces, so the fit line stays on top
        if s.type == "rect":  # exponential phase
            s.update(fillcolor=INFO_PHASE, opacity=0.13, layer="below")
        else:  # reference lines
            s.update(line_color=INFO_REF, line_dash="dash", opacity=1, layer="below")
    # Draw order: grey points, used points, other lines, fit line, μmax marker
    rank = {"Data": 0, "Selected": 1, "Fitted curve": 3, "Fitted": 3}
    fig.data = sorted(
        fig.data, key=lambda tr: rank.get(tr.name, 2 if tr.mode == "lines" else 4)
    )
    return style_info_axes(fig)


def style_info_axes(fig: go.Figure) -> go.Figure:
    """Transparent background and grey axis lines, as in the explanatory figures.

    Leaves trace colours alone, so per-sample colours are kept.
    """
    axis = dict(
        showline=True,
        linecolor=INFO_AXIS,
        linewidth=1.5,
        showgrid=False,
        zeroline=False,
        ticks="",
        tickfont=dict(color=INFO_INK_2),
        title_font=dict(size=16, color=INFO_INK),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_layout(
        font=dict(family="Arial, sans-serif", color=INFO_INK_2),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(color=INFO_INK))
    return fig


# --- helpers ------------------------------------------------------------------
def _finite_sorted_xy(time_s, y_s):
    """Return finite x/y arrays sorted by x."""
    t = np.asarray(time_s, float)
    y = np.asarray(y_s, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size:
        o = np.argsort(t)
        t, y = t[o], y[o]
    return t, y


def _add_lasso_selected_points(
    fig: go.Figure,
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    selected_times: list[float] | None,
    *,
    scale: str,
) -> go.Figure:
    """Overlay the points used in the fit (lasso-selected) on an existing figure."""
    if not selected_times:
        return fig

    sel = np.asarray(selected_times, dtype=float)
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return fig

    refit_indices = _match_selected_times(t_arr, sel)
    if refit_indices.size == 0:
        return fig

    t_sel = t_arr[refit_indices]
    y_sel = y_arr[refit_indices]
    valid = np.isfinite(y_sel) & ((y_sel > 0) if scale == "log" else True)
    t_sel, y_sel = t_sel[valid], y_sel[valid]
    if t_sel.size == 0:
        return fig

    fig.add_trace(
        go.Scatter(
            x=t_sel,
            y=np.log(y_sel) if scale == "log" else y_sel,
            mode="markers",
            name="Selected",
            marker=dict(size=8, color="#ef5350", symbol="circle"),
            showlegend=False,
        )
    )
    return fig


# --- single well -----------------------------------------------------------
def plot_well_curve(
    plate: dict,
    well: str,
    *,
    log_scale: bool = False,
    phase_boundaries: bool = True,
    umax_point: bool = True,
    max_od: bool = True,
    baseline_od: bool = True,
    tangent: bool = False,
    fitted_model: bool = True,
) -> go.Figure:
    """Annotated growth curve of one well, as on Check Growth Fits and in exports.

    Points used in the fit (lasso-selected, default all) are drawn darker.
    Returns an empty figure if the well has no data.
    """
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()
    t, y = _finite_sorted_xy(d["Time"], d["baseline_corrected"])
    if t.size == 0:
        return go.Figure()

    gs = (plate.get("growth_stats") or {}).get(well) or {}
    scale = "log" if log_scale else "linear"
    fig = gc_plot.create_base_plot(t, y, scale=scale)
    fig = _add_lasso_selected_points(
        fig, t, y, gs.get("_used_fit_times") or t.tolist(), scale=scale
    )
    if gs and not is_no_growth(gs):
        fig = gc_plot.annotate_plot(
            fig,
            fit_result=(plate.get("fit_parameters") or {}).get(well),
            stats=gs,
            show_fitted_curve=fitted_model,
            show_phase_boundaries=phase_boundaries,
            show_crosshairs=umax_point,
            show_od_max_line=max_od,
            show_n0_line=baseline_od,
            show_umax_marker=umax_point,
            show_tangent=tangent,
            scale=scale,
        )

    # x-axis range matches the data exactly (no gap at the y-axis)
    fig.update_xaxes(
        title="Time (hours)",
        showgrid=False,
        type="linear",
        range=[float(t.min()), float(t.max())],
    )
    fig.update_yaxes(
        title="ln(OD600)" if log_scale else "OD600 (baseline-corrected)",
        showgrid=False,
    )
    fig.update_layout(
        showlegend=False,
        title=f"Fitted growth curve – {well}",
        margin=dict(l=20, r=20, t=60, b=20),
        height=600,
    )
    return style_like_info_plots(
        fig, fit_window=(gs.get("fit_t_min"), gs.get("fit_t_max"))
    )


def plot_derivative_metric(
    plate: dict,
    well: str,
    metric: str,
    sg_window=DEFAULT_PARAMS["sg_window"],
    sg_poly=DEFAULT_PARAMS["sg_poly"],
    gs: dict | None = None,
):
    """Plot either dN/dt or μ (specific growth rate) for a well.

    This function generates three traces:
    1. Raw data metric (light grey)
    2. Smoothed data metric (main trace)
    3. Model fit metric (solid blue line)

    Args:
        plate: Plate dictionary containing processed_data and fit_parameters
        well: Well identifier
        metric: Either "dndt" for dN/dt or "mu" for μ
        sg_window: Savitzky-Golay window size
        sg_poly: Savitzky-Golay polynomial order
        gs: Growth statistics dictionary (optional). If provided with _used_fit_times,
            only the lasso-selected data points will be used for calculation.
    """
    # Validate metric
    if metric not in ["dndt", "mu"]:
        raise ValueError(f"metric must be 'dndt' or 'mu', got '{metric}'")

    # Get processed data
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t_raw = d["Time"].to_numpy(float)
    y_raw = d["baseline_corrected"].to_numpy(float)

    # Full time range for the x-axis, before any filtering
    x_range = [float(t_raw.min()), float(t_raw.max())]

    # Filter to lasso-selected points if available
    gs = gs or {}
    used_times = gs.get("_used_fit_times")
    if used_times is not None and len(used_times) > 0:
        diffs = np.abs(t_raw[:, None] - np.asarray(used_times, dtype=float))
        used_mask = (diffs < 0.01).any(axis=1)
        t_raw = t_raw[used_mask]
        y_raw = y_raw[used_mask]

    if len(t_raw) < 3:
        return go.Figure()

    # Step 1: Calculate metric on raw data
    if metric == "dndt":
        t_metric_raw, metric_raw = compute_first_derivative(t_raw, y_raw)
        metric_label = "dN/dt"
        y_axis_title = "dN/dt"
        plot_title = f"dN/dt – {well}"
    else:  # mu
        t_metric_raw, metric_raw = compute_instantaneous_mu(t_raw, y_raw)
        metric_label = "μ"
        y_axis_title = "μ (h⁻¹)"
        plot_title = f"Specific growth rate – {well}"

    # Step 2: Smooth the data
    y_smooth = smooth(y_raw, sg_window, sg_poly)

    # Step 3: Calculate metric on smoothed data
    if metric == "dndt":
        t_metric_smooth, metric_smooth = compute_first_derivative(t_raw, y_smooth)
    else:  # mu
        t_metric_smooth, metric_smooth = compute_instantaneous_mu(t_raw, y_smooth)

    # Create figure
    fig = go.Figure()

    # Plot raw metric (light grey)
    fig.add_trace(
        go.Scatter(
            x=t_metric_raw,
            y=metric_raw,
            mode="lines",
            line=dict(width=5, color="lightgrey"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} hours<br>{metric_label} (raw)=%{{y:.4f}}<extra></extra>",
            showlegend=False,
            name="Raw",
        )
    )

    # Plot smoothed metric (lighter red)
    fig.add_trace(
        go.Scatter(
            x=t_metric_smooth,
            y=metric_smooth,
            mode="lines",
            line=dict(width=5, color="#FF6692"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} hours<br>{metric_label} (smoothed)=%{{y:.4f}}<extra></extra>",
            showlegend=False,
            name="Smoothed",
        )
    )

    # Step 4 & 5: Generate model metric and plot
    fit_parameters = (plate.get("fit_parameters") or {}).get(well)
    if fit_parameters is not None:
        model_type = fit_parameters.get("model_type", "")
        params = fit_parameters.get("params", {})
        metric_model = None
        t_model = None

        # Get the fitted data range
        fit_t_min = params.get("fit_t_min")
        fit_t_max = params.get("fit_t_max")

        # Filter to fitted range if available
        if fit_t_min is not None and fit_t_max is not None:
            fit_mask = (t_raw >= fit_t_min) & (t_raw <= fit_t_max)
            t_model = t_raw[fit_mask]
            y_model_raw = y_raw[fit_mask]
            y_model_smooth = y_smooth[fit_mask]
        else:
            # Use full range if fit bounds not available
            t_model = t_raw
            y_model_raw = y_raw
            y_model_smooth = y_smooth

        if len(t_model) < 2:
            # Not enough points in fitted range
            t_model = None
        else:
            if model_type == "sliding_window":
                # For sliding window, calculate from raw data (as growthcurves does)
                # The window used in the analysis (the fit result does not store it)
                window_points = int(
                    gs.get("window_points", plate["params"]["window_points"])
                )
                if metric == "dndt":
                    # For dN/dt, we need to smooth first then compute derivative
                    _, metric_model = compute_first_derivative(t_model, y_model_smooth)
                else:  # mu
                    # For μ, use sliding window on raw data
                    _, metric_model = compute_sliding_window_growth_rate(
                        t_model, y_model_raw, window_points=window_points
                    )

            elif model_type in (
                MODEL_REGISTRY["mechanistic"] + MODEL_REGISTRY["phenomenological"]
            ):
                # For parametric models, compute metric from the model
                y_model = evaluate_parametric_model(t_model, model_type, params)
                # Phenomenological models return ln(N/N0), so convert to OD
                if model_type in MODEL_REGISTRY["phenomenological"]:
                    y_model = log_to_linear(y_model, params["N0"])

                # Compute metric from model
                if metric == "dndt":
                    _, metric_model = compute_first_derivative(t_model, y_model)
                else:  # mu
                    _, metric_model = compute_instantaneous_mu(t_model, y_model)

            elif model_type == "spline":
                # For spline model, reconstruct the spline and evaluate
                try:
                    spline = spline_from_params(params)

                    if metric == "dndt":
                        # Spline is fitted to log(y), so exp(spline(t)) gives y
                        y_log_model = spline(t_model)
                        y_model = np.exp(y_log_model)
                        _, metric_model = compute_first_derivative(t_model, y_model)
                    else:  # mu
                        # μ = d(ln(y))/dt, which is the derivative of the spline
                        metric_model = spline.derivative()(t_model)
                except Exception:
                    # If spline reconstruction fails, skip model trace
                    pass

        # Plot model metric if available
        if (
            metric_model is not None
            and t_model is not None
            and np.isfinite(metric_model).any()
        ):
            fig.add_trace(
                go.Scatter(
                    x=t_model,
                    y=metric_model,
                    mode="lines",
                    line=dict(width=5, color="#8dcde0"),
                    hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} hours<br>{metric_label} (fitted)=%{{y:.4f}}<extra></extra>",
                    showlegend=False,
                    name="Fitted",
                )
            )

    # Add phase boundary annotations
    stats_converted = None
    if gs and not is_no_growth(gs):
        exp_start = gs.get("exp_phase_start")
        exp_end = gs.get("exp_phase_end")
        if exp_start is not None and exp_end is not None:
            stats_converted = {
                "exp_phase_start": float(exp_start),
                "exp_phase_end": float(exp_end),
            }

    if stats_converted is not None:
        fig = gc_plot.annotate_plot(
            fig=fig,
            fit_result=None,
            stats=stats_converted,
            show_fitted_curve=False,
            show_phase_boundaries=True,
            show_crosshairs=False,
            show_od_max_line=False,
            show_n0_line=False,
            show_umax_marker=False,
            show_tangent=False,
            scale="linear",
            row=None,
            col=None,
        )

    # Update layout
    fig.update_layout(
        title=plot_title,
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, title="Time (hours)", range=x_range)
    # y range from the smoothed curve, so raw-derivative spikes don't flatten it
    finite = np.asarray(metric_smooth, dtype=float)
    finite = finite[np.isfinite(finite)]
    y_range = None
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.05 * (hi - lo) or 0.05 * abs(hi) or 0.05
        y_range = [lo - pad, hi + pad]
    fig.update_yaxes(showgrid=False, title=y_axis_title, range=y_range)
    return style_like_info_plots(fig)
