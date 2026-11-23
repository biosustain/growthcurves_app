"""Interactive well-by-well growth fit inspection helpers."""

import time

import numpy as np
import pandas as pd
import streamlit as st
from growthcurves.inference import bad_fit_stats
from src.functions.data_processing import mark_changed
from src.functions.fitting_pipeline import fit_growth_series

# ---------------- Selection + stats helpers ----------------


def _cycle(items: list[str], current: str, step: int) -> str:
    """Return the next/previous item from a list, with wraparound."""
    if not items:
        return current
    try:
        i = items.index(current)
    except ValueError:
        i = 0
    return items[(i + step) % len(items)]


def _delete_well_from_plate(plate: dict, well: str) -> None:
    """Remove a well from all per-well containers on a plate (in-place)."""
    for k in ("name", "processed_data", "growth_stats", "fit_parameters"):
        plate.get(k, {}).pop(well, None)
    mark_changed(plate)


def _get_selected_points(event) -> tuple[np.ndarray, np.ndarray]:
    """Extract selected x/y arrays from a Plotly selection event."""
    if event is None:
        return np.array([]), np.array([])

    sel = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not sel:
        return np.array([]), np.array([])

    points = (
        sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    )
    if not points:
        return np.array([]), np.array([])

    xs = np.asarray([float(p["x"]) for p in points])
    ys = np.asarray([float(p["y"]) for p in points])
    return xs, ys


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


def _collect_lasso_series(
    processed: pd.DataFrame, selected_times: np.ndarray, *, time_tolerance: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) arrays for lasso-selected points from processed data."""
    if processed is None or processed.empty:
        return np.array([]), np.array([])

    all_t = processed["Time"].to_numpy(float)
    all_y = processed["baseline_corrected"].to_numpy(float)

    refit_indices = _match_selected_times(
        all_t, selected_times, time_tolerance=time_tolerance
    )
    if refit_indices.size < 2:
        return np.array([]), np.array([])

    refit_t = all_t[refit_indices]
    refit_y = all_y[refit_indices]

    sort_idx = np.argsort(refit_t)
    return refit_t[sort_idx], refit_y[sort_idx]


def _analyse_series_with_plate_params(
    t_arr: np.ndarray, y_arr: np.ndarray, params: dict
) -> tuple[dict, dict | None]:
    """Run the same analysis pipeline as initial plate analysis."""
    if t_arr.size < 2 or y_arr.size < 2:
        return bad_fit_stats(), None

    try:
        # Use unified fitting pipeline
        return fit_growth_series(t_arr, y_arr, params)
    except Exception:
        return bad_fit_stats(), None


def reanalysis_params(plate: dict, key: str) -> tuple[dict, dict]:
    """Plate params with the well's Re-analyse popover values, and the record of
    the values used (shown in the Analysis Parameters table).

    key is the well's widget key, f"{plate_id}_{well}"; the popover values are
    initialised from the plate params when the well is first shown.
    """
    ss = st.session_state
    params = dict(plate["params"])
    params.update(
        min_od_increase=float(ss[f"rp_min_od__{key}"]),
        min_growth_rate=float(ss[f"rp_min_gr__{key}"]),
        min_signal_to_noise=float(ss[f"rp_min_snr__{key}"]),
        min_data_points=int(ss[f"rp_min_dp__{key}"]),
    )
    used = {
        k: params[k]
        for k in (
            "min_od_increase",
            "min_growth_rate",
            "min_signal_to_noise",
            "min_data_points",
        )
    }
    if params["growth_method"] == "Sliding Window":
        params["window_points"] = used["window_points"] = int(ss[f"rp_window__{key}"])
    elif params["growth_method"] == "Spline":
        manual = ss[f"rp_smooth__{key}"] == "manual"
        lam = float(ss.get(f"rp_spline_s__{key}") or 0.0) if manual else 0.0
        params["smooth"] = used["smooth"] = lam if lam > 0 else "auto"
    return params, used


def update_growth_stats_from_lasso(
    plates: dict, pid: str, well: str, chart_key: str
) -> None:
    """Update growth stats based on lasso-selected points.

    Uses any custom re-analysis parameters already set in the popover for this well.
    """
    xs, _ = _get_selected_points(st.session_state.get(chart_key))
    if xs.size < 2:
        return

    plate = plates.get(pid, {})
    gs = plates.setdefault(pid, {}).setdefault("growth_stats", {}).setdefault(well, {})
    fit_parameters = plates.setdefault(pid, {}).setdefault("fit_parameters", {})

    sort_idx = np.argsort(xs)
    selected_times = xs[sort_idx]

    processed = plate.get("processed_data", {}).get(well)
    if processed is None or processed.empty:
        return

    refit_t, refit_y = _collect_lasso_series(processed, selected_times)
    if refit_t.size < 2:
        return

    # Plate params with any custom values set in the Re-analyse popover
    params, analysis_params = reanalysis_params(plate, f"{pid}_{well}")
    fit, fit_result = _analyse_series_with_plate_params(refit_t, refit_y, params)

    # Update gs in-place so all references stay valid
    gs.clear()
    gs.update(fit)
    gs["_used_fit_times"] = refit_t.tolist()
    gs["_analysis_params"] = analysis_params
    gs["_lasso_update_time"] = time.time()

    if fit_result is not None:
        fit_parameters[well] = fit_result
    else:
        fit_parameters.pop(well, None)
    mark_changed(plate)


# ---------------- Data helpers ----------------
def _sg_params_for_plate(plates: dict, plate_id: str) -> tuple[int, int]:
    """Return the Savitzky-Golay window and polynomial order for a plate."""
    params = plates[plate_id]["params"]
    return int(params["sg_window"]), int(params["sg_poly"])


def analyse_well(plate: dict, well: str, params_override: dict | None = None) -> dict:
    """Recompute growth statistics for a single well using existing processed data."""
    p = dict(plate["params"])
    if params_override:
        p.update(params_override)
    processed_data = (plate or {}).get("processed_data") or {}
    fit_parameters = (plate or {}).setdefault("fit_parameters", {})

    well = str(well).upper()

    # Get the already-processed data for this well
    processed = processed_data.get(well)
    if processed is None or processed.empty:
        fit_parameters.pop(well, None)
        return bad_fit_stats()

    try:
        t_arr = processed["Time"].to_numpy(float)
        y_arr = processed["baseline_corrected"].to_numpy(float)
        fit, fit_result = _analyse_series_with_plate_params(t_arr, y_arr, p)
        if fit_result is not None:
            fit_parameters[well] = fit_result
        else:
            fit_parameters.pop(well, None)

    except Exception:
        fit = bad_fit_stats()
        fit_parameters.pop(well, None)

    return fit
