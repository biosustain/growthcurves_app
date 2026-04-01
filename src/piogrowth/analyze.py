"""Functions used in 0_upload_data.py and 2_turbiostat.py"""

import inspect
import logging
import time

import growthcurves as gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

NON_PARAMETRIC_FIT_PARAMS = set(
    inspect.signature(gc.non_parametric.fit_non_parametric).parameters
)

__all__ = [
    "NON_PARAMETRIC_FIT_PARAMS",
    "get_timestamps_from_elapsed_hours",
    "load_method_notes_markdown",
    "run_model_fitting_on_df_compat",
    "build_fit_kwargs",
    "fit_single_series",
    "get_selected_times_from_event",
    "match_selected_times",
    "collect_selected_series",
    "overlay_selected_points",
    "is_bad_fit",
    "get_reactor_stat",
    "get_reactor_stats_dict",
    "normalize_smooth",
    "format_smooth",
    "growth_method_from_model",
    "default_analysis_params",
    "format_growth_stats_table",
    "format_analysis_params_table",
    "update_reactor_stats",
    "build_analysis_params_per_sample_table",
]


def get_timestamps_from_elapsed_hours(
    elapsed_hours, start_time, elapsed_time_unit="h", round_to="s"
):
    return start_time + pd.to_timedelta(elapsed_hours, unit=elapsed_time_unit).dt.round(
        round_to
    )


def load_method_notes_markdown() -> str:
    """Load method notes markdown shown in the Help popover."""
    try:
        with open("app/markdowns/curve_fitting.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "_Method notes file not found._"


def run_model_fitting_on_df_compat(
    df: pd.DataFrame,
    model_name: str,
    n_fits: int,
    spline_s: int,
    smooth_mode: str,
    window_points: int,
    phase_boundary_method: str,
    lag_cutoff: float,
    exp_cutoff: float,
    min_data_points: int,
    min_signal_to_noise: float,
    min_od_increase: float,
    min_growth_rate: float,
) -> tuple[pd.DataFrame, dict]:
    """Run fitting across reactors using gc.fit_model in a version-compatible way."""
    stats_df = {}
    fit_cache = {}

    for col in df.columns:
        s = df[col].dropna()
        t_start = time.time()
        fit_kwargs = build_fit_kwargs(
            model_name=model_name,
            n_fits=n_fits,
            window_points=window_points,
            spline_s=spline_s,
            smooth_mode=smooth_mode,
        )
        _t = s.index.to_numpy()
        _n = s.to_numpy()

        fit_result, stats = gc.fit_model(
            t=_t,
            N=_n,
            model_name=model_name,
            lag_threshold=lag_cutoff,
            exp_threshold=exp_cutoff,
            phase_boundary_method=phase_boundary_method,
            **fit_kwargs,
        )
        res_no_growth = gc.inference.detect_no_growth(
            t=_t,
            N=_n,
            growth_stats=stats,
            min_data_points=min_data_points,
            min_signal_to_noise=min_signal_to_noise,
            min_od_increase=min_od_increase,
            min_growth_rate=min_growth_rate,
        )
        if res_no_growth["is_no_growth"]:
            # overwrite stats with reason
            logger.debug(res_no_growth)
            stats = gc.inference.bad_fit_stats()
            stats["no_growth_reason"] = res_no_growth.get(
                "reason", "No growth detected"
            )
        fit_cache[col] = fit_result
        stats["elapsed_time"] = time.time() - t_start
        stats["model_name"] = model_name
        stats_df[col] = stats

    return pd.DataFrame(stats_df).T, fit_cache


def build_fit_kwargs(
    model_name: str,
    n_fits: int,
    window_points: int,
    spline_s: int,
    smooth_mode: str,
) -> dict:
    """Map UI options to growthcurves fit kwargs across API versions."""
    fit_kwargs = {}
    if model_name == "sliding_window":
        fit_kwargs["n_fits"] = n_fits
        fit_kwargs["window_points"] = window_points
    elif model_name == "spline":
        fit_kwargs["window_points"] = window_points
        if "smooth" in NON_PARAMETRIC_FIT_PARAMS:
            fit_kwargs["smooth"] = smooth_mode
        if "spline_s" in NON_PARAMETRIC_FIT_PARAMS:
            fit_kwargs["spline_s"] = spline_s
    return fit_kwargs


def fit_single_series(
    t_values: np.ndarray,
    n_values: np.ndarray,
    batch_options: dict,
) -> tuple[dict | None, dict]:
    """Fit one reactor series using current batch options."""
    t_arr = np.asarray(t_values, dtype=float)
    n_arr = np.asarray(n_values, dtype=float)
    fit_kwargs = build_fit_kwargs(
        model_name=batch_options["selected_model"],
        n_fits=batch_options["n_fits"],
        window_points=batch_options["window_points"],
        spline_s=batch_options["spline_smoothing_value"],
        smooth_mode=batch_options.get("smooth_mode", "fast"),
    )

    t_start = time.time()
    fit_result, stats = gc.fit_model(
        t=t_arr,
        N=n_arr,
        model_name=batch_options["selected_model"],
        lag_threshold=batch_options["lag_cutoff"],
        exp_threshold=batch_options["exp_cutoff"],
        phase_boundary_method=batch_options["phase_boundary_method"],
        **fit_kwargs,
    )
    stats["elapsed_time"] = time.time() - t_start
    stats["model_name"] = batch_options["selected_model"]
    return fit_result, stats


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
    """Return indices in all_t matching selected_times within tolerance."""
    if all_t.size == 0 or selected_times.size == 0:
        return np.array([], dtype=int)

    matched = []
    seen = set()
    for sel_t in selected_times:
        hits = np.where(np.abs(all_t - sel_t) < time_tolerance)[0]
        if len(hits) == 0:
            continue
        idx = int(hits[0])
        if idx not in seen:
            matched.append(idx)
            seen.add(idx)
    return np.asarray(matched, dtype=int)


def collect_selected_series(
    series: pd.Series,
    selected_times: np.ndarray,
    *,
    time_tolerance: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect selected (t, y) points from a reactor series."""
    s = series.dropna()
    if s.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_all = s.index.to_numpy(dtype=float)
    y_all = s.to_numpy(dtype=float)
    idx = match_selected_times(t_all, selected_times, time_tolerance=time_tolerance)
    if idx.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_refit = t_all[idx]
    y_refit = y_all[idx]
    order = np.argsort(t_refit)
    return t_refit[order], y_refit[order]


def overlay_selected_points(
    fig: go.Figure,
    t_values: np.ndarray,
    y_values: np.ndarray,
    selected_times: list[float] | None,
    *,
    scale: str,
    time_tolerance: float = 0.01,
) -> go.Figure:
    """Overlay included (red) and excluded (gray) points on the growth plot."""
    t_arr = np.asarray(t_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(y_arr) & (y_arr > 0)
    t_arr = t_arr[valid]
    y_arr = y_arr[valid]
    if t_arr.size == 0:
        return fig

    y_plot = np.log(y_arr) if scale == "log" else y_arr
    included = np.ones_like(t_arr, dtype=bool)
    if selected_times:
        sel = np.asarray(selected_times, dtype=float)
        sel = sel[np.isfinite(sel)]
        if sel.size > 0:
            included = np.zeros_like(t_arr, dtype=bool)
            idx = match_selected_times(t_arr, sel, time_tolerance=time_tolerance)
            included[idx] = True

    # Remove default mono-color points before custom overlays.
    fig.data = tuple(
        trace for trace in fig.data if getattr(trace, "name", None) != "Data"
    )

    excluded = ~included
    if excluded.any():
        fig.add_trace(
            go.Scatter(
                x=t_arr[excluded],
                y=y_plot[excluded],
                mode="markers",
                marker=dict(size=7, color="gray", opacity=0.55),
                hovertemplate="Time=%{x:.2f}<br>OD=%{y:.4f}<extra></extra>",
                showlegend=False,
                name="Excluded",
            )
        )

    if included.any():
        fig.add_trace(
            go.Scatter(
                x=t_arr[included],
                y=y_plot[included],
                mode="markers",
                marker=dict(size=8, color="#ef5350", opacity=0.95),
                hovertemplate="Time=%{x:.2f}<br>OD=%{y:.4f}<extra></extra>",
                showlegend=False,
                name="Included",
            )
        )

    return fig


def is_bad_fit(gs: dict) -> bool:
    """Return True when stats indicate no growth or failed fit."""
    return gc.inference.is_no_growth(gs or {})


def get_reactor_stat(stats_df: pd.DataFrame, reactor: str, key: str):
    """Get a scalar stat value even if reactor labels are duplicated."""
    if key not in stats_df.columns or reactor not in stats_df.index:
        return np.nan
    value = stats_df.loc[reactor, key]
    if isinstance(value, pd.Series):
        value = value.dropna()
        return value.iloc[0] if not value.empty else np.nan
    return value


def get_reactor_stats_dict(stats_df: pd.DataFrame, reactor: str) -> dict:
    """Get one reactor stats row as a plain dictionary."""
    if reactor not in stats_df.index:
        return {}
    row = stats_df.loc[reactor]
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return {}
        row = row.iloc[0]
    return row.to_dict()


def normalize_smooth(value) -> str:
    """Normalize spline mode to fast/slow with legacy compatibility."""
    mode = str(value).strip().lower() if value is not None else ""
    if mode == "auto":
        return "slow"
    if mode in {"fast", "slow"}:
        return mode
    return "fast"


def format_smooth(value) -> str:
    """Format spline mode for display."""
    return normalize_smooth(value).capitalize()


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
        "min_od_increase": batch_options.get("min_od_increase"),
        "min_growth_rate": batch_options.get("min_growth_rate"),
        "min_signal_to_noise": batch_options.get("min_signal_to_noise"),
        "min_data_points": batch_options.get("min_data_points"),
    }
    method = growth_method_from_model(batch_options.get("selected_model", ""))
    if method == "Sliding Window":
        params["window_points"] = batch_options.get("window_points")
    elif method == "Spline":
        params["smooth"] = normalize_smooth(batch_options.get("smooth_mode", "fast"))
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
        method_params = [("smooth", "Spline mode", format_smooth)]

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


def update_reactor_stats(
    stats_df: pd.DataFrame,
    reactor: str,
    stats: dict,
):
    """Write a reactor stats dict into the summary DataFrame."""
    for k, v in stats.items():
        stats_df.loc[reactor, k] = v


def build_analysis_params_per_sample_table(
    stats_df: pd.DataFrame,
    df_rolling: pd.DataFrame,
    batch_options: dict,
    used_params_map: dict | None,
    selected_fit_times_map: dict | None,
) -> pd.DataFrame:
    """Build a per-sample table of analysis parameters used."""
    used_params_map = used_params_map or {}
    selected_fit_times_map = selected_fit_times_map or {}

    method = growth_method_from_model(batch_options.get("selected_model", ""))
    base_params = default_analysis_params(batch_options)
    rows = []

    for sample in stats_df.index:
        sample_params = dict(base_params)
        sample_params.update(used_params_map.get(sample, {}))

        total_points = (
            int(df_rolling[sample].dropna().shape[0]) if sample in df_rolling else 0
        )
        selected_times = selected_fit_times_map.get(sample)
        selected_points = len(selected_times) if selected_times else total_points

        row = {
            "sample": sample,
            "model": batch_options.get("selected_model"),
            "growth_method": method,
            "phase_boundary_method": batch_options.get("phase_boundary_method"),
            "min_od_increase": sample_params.get("min_od_increase"),
            "min_growth_rate": sample_params.get("min_growth_rate"),
            "min_signal_to_noise": sample_params.get("min_signal_to_noise"),
            "min_data_points": sample_params.get("min_data_points"),
            "window_points": sample_params.get("window_points"),
            "smooth_mode": sample_params.get("smooth"),
            "selected_points": selected_points,
            "total_points": total_points,
        }

        if sample in stats_df.index:
            sample_stats = get_reactor_stats_dict(stats_df, sample)
            row["fit_t_min"] = sample_stats.get("fit_t_min")
            row["fit_t_max"] = sample_stats.get("fit_t_max")
        rows.append(row)

    return pd.DataFrame(rows)
