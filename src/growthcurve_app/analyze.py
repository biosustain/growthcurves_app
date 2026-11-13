"""Fit options, selection and table helpers used by the AutoGrowth pages."""

import growthcurves as gc
import numpy as np
import pandas as pd

__all__ = [
    "build_fit_kwargs",
    "get_selected_times_from_event",
    "match_selected_times",
    "is_bad_fit",
    "normalize_smooth",
    "smooth_to_number",
    "format_smooth",
    "growth_method_from_model",
    "default_analysis_params",
    "format_growth_stats_table",
    "format_analysis_params_table",
]


def build_fit_kwargs(
    model_name: str,
    window_points: int,
    smooth_mode: str | float,
    min_fold: float = 1.5,
) -> dict:
    """Map UI options to growthcurves fit kwargs.

    - sliding_window: ``window_points`` is the minimum number of readings per window
      (the window length adapts per curve).
    - spline: ``smooth`` is ``"auto"`` (REML) or a fixed positive value.
    - all models: ``min_fold`` is the minimum fold rise of the smoothed OD for growth.
    """
    fit_kwargs = {"min_fold": float(min_fold)}
    if model_name == "sliding_window":
        fit_kwargs["window_points"] = int(window_points)
    elif model_name == "spline":
        fit_kwargs["smooth"] = normalize_smooth(smooth_mode)
    return fit_kwargs


def get_selected_times_from_event(event) -> np.ndarray:
    """Extract selected x-values from Streamlit Plotly selection event."""
    if event is None:
        return np.array([], dtype=float)

    selection = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not selection:
        return np.array([], dtype=float)

    points = (
        selection.get("points")
        if isinstance(selection, dict)
        else getattr(selection, "points", None)
    )
    if not points:
        return np.array([], dtype=float)

    x_values = []
    for point in points:
        try:
            x_values.append(float(point["x"]))
        except (TypeError, ValueError, KeyError):
            continue
    return np.asarray(x_values, dtype=float)


def match_selected_times(
    all_t: np.ndarray,
    selected_times: np.ndarray,
    *,
    time_tolerance: float = 0.01,
) -> np.ndarray:
    """Sorted unique indices of all_t within tolerance of selected_times. The
    tolerance is capped below half the smallest sampling interval, so each selected
    time matches at most one point."""
    if all_t.size == 0 or selected_times.size == 0:
        return np.array([], dtype=int)

    order = np.argsort(all_t)
    t_sorted = all_t[order]
    steps = np.diff(t_sorted)
    steps = steps[steps > 0]
    tol = min(time_tolerance, steps.min() / 2) if steps.size else time_tolerance

    pos = np.searchsorted(t_sorted, selected_times - tol).clip(max=t_sorted.size - 1)
    hit = np.abs(t_sorted[pos] - selected_times) < tol
    return np.unique(order[pos[hit]])


def is_bad_fit(gs: dict) -> bool:
    """Return True when stats indicate no growth or failed fit."""
    return gc.inference.is_no_growth(gs or {})


def normalize_smooth(value) -> str | float:
    """Normalize spline smoothing to ``"auto"`` or a positive float.

    ``growthcurves.non_parametric.fit_spline`` accepts ``smooth="auto"`` (chosen by
    REML) or a fixed positive number. ``None``, ``0``, non-numeric values and the
    legacy ``"fast"``/``"slow"`` modes map to ``"auto"``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "auto"
    return number if np.isfinite(number) and number > 0 else "auto"


def smooth_to_number(value) -> float:
    """Spline smoothing as a number for UI widgets: 0.0 means automatic (REML)."""
    smooth = normalize_smooth(value)
    return 0.0 if smooth == "auto" else float(smooth)


def format_smooth(value) -> str:
    """Format spline smoothing for display."""
    smooth = normalize_smooth(value)
    return "Auto (REML)" if smooth == "auto" else f"{smooth:g}"


def growth_method_from_model(model_name: str) -> str:
    """Map selected model name to growth method label."""
    if model_name == "sliding_window":
        return "Sliding Window"
    if model_name == "spline":
        return "Spline"
    return "Model Fitting"


def default_analysis_params(batch_options: dict) -> dict:
    """Build default analysis parameter dict for readout table."""
    params = {
        "selected_model": batch_options.get("selected_model"),
        "min_fold": batch_options.get("min_fold"),
        "min_od_increase": batch_options.get("min_od_increase"),
        "min_growth_rate": batch_options.get("min_growth_rate"),
        "min_signal_to_noise": batch_options.get("min_signal_to_noise"),
        "min_data_points": batch_options.get("min_data_points"),
    }
    method = growth_method_from_model(batch_options.get("selected_model", ""))
    if method == "Sliding Window":
        params["window_points"] = batch_options.get("window_points")
    elif method == "Spline":
        params["smooth"] = normalize_smooth(batch_options.get("smooth_mode", "auto"))
    return params


def format_growth_stats_table(gs: dict) -> pd.DataFrame:
    """Format growth stats into a displayable table."""
    gs = gs or {}
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
    if is_bad_fit(gs):
        reason = gs.get("no_growth_reason", "--")
        rows.append({"Metric": "No growth reason", "Value": reason})

    for key, label, formatter in metrics:
        value = gs.get(key)
        if key == "mu_max" and value is None:
            value = gs.get("specific_growth_rate")
        try:
            formatted_value = formatter(value) if value is not None else "--"
        except (ValueError, TypeError):
            formatted_value = "--"
        rows.append({"Metric": label, "Value": formatted_value})

    return pd.DataFrame(rows)


def format_analysis_params_table(
    gs: dict,
    batch_options: dict,
    analysis_params: dict,
    n_total: int | None = None,
    n_selected: int | None = None,
) -> pd.DataFrame:
    """Format analysis parameters into a displayable table."""
    rows = []
    total_str = str(n_total) if n_total is not None else "?"
    selected_str = str(n_selected) if n_selected is not None else total_str
    rows.append(
        {"Parameter": "Data subset (points)", "Value": f"{selected_str}/{total_str}"}
    )

    growth_method = growth_method_from_model(batch_options.get("selected_model", ""))
    common_params = [
        ("min_fold", "Min fold change", lambda x: f"{float(x):.2f}"),
        ("min_od_increase", "Min OD increase", lambda x: f"{float(x):.4f}"),
        ("min_growth_rate", "Min growth rate (1/h)", lambda x: f"{float(x):.5f}"),
        ("min_signal_to_noise", "Min signal-to-noise", lambda x: f"{float(x):.2f}"),
        ("min_data_points", "Min data points", lambda x: str(int(x))),
    ]
    method_params = []
    if growth_method == "Sliding Window":
        method_params = [
            ("window_points", "Min. window size (points)", lambda x: str(int(x)))
        ]
    elif growth_method == "Spline":
        method_params = [("smooth", "Spline smoothing", format_smooth)]

    defaults = default_analysis_params(batch_options)
    for param_name, plabel, formatter in common_params + method_params:
        value = analysis_params.get(param_name)
        if value is None:
            value = defaults.get(param_name)
        if value is not None:
            try:
                rows.append({"Parameter": plabel, "Value": formatter(value)})
            except (ValueError, TypeError):
                pass

    fit_metrics = [
        (
            "fit_t_min",
            "Analysis window start (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "fit_t_max",
            "Analysis window end (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
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
