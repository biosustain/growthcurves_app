"""Unified growth curve fitting pipeline.

Extracted from duplicated logic in:
- data_processing.py:analyse_plate (lines 408-497)
- check_growth_fits.py:_analyse_series_with_plate_params (lines 195-265)
"""

from growthcurves.inference import bad_fit_stats, detect_no_growth, extract_stats
from growthcurves.non_parametric import fit_non_parametric
from growthcurves.parametric import fit_parametric


def fit_growth_series(t_arr, y_arr, params: dict) -> tuple[dict, dict | None]:
    """
    Unified growth fitting pipeline (replaces duplicated code).

    This is the single source of truth for growth curve fitting,
    replacing duplicated logic in data_processing.py and check_growth_fits.py.

    Args:
        t_arr: Time array
        y_arr: OD array (baseline-corrected)
        params: Dict with growth_method, model_type, window_points, etc.

    Returns:
        (fit_stats_dict, fit_result_dict | None)
    """
    growth_method = params["growth_method"]
    # Phase boundaries always use the tangent at μmax (models with a fitted lag
    # time use it as the exponential phase start)
    phase_boundary_method = "tangent"

    fit_result = None

    # Method selection (parametric, spline, or sliding window)
    if growth_method == "Model Fitting":
        # Use parametric model fitting
        fit_result = fit_parametric(t_arr, y_arr, method=params["model_type"])
        if fit_result is not None:
            fit = extract_stats(
                fit_result,
                t_arr,
                y_arr,
                phase_boundary_method=phase_boundary_method,
            )
            # Store model parameters
            for param_name, param_val in fit_result["params"].items():
                fit[f"fit_param_{param_name}"] = float(param_val)
        else:
            fit = bad_fit_stats()

    elif growth_method == "Spline":
        # Use non-parametric spline method
        smooth = params["smooth"]
        # growthcurves only accepts "auto" (REML) or a positive number, so anything
        # else (e.g. legacy "fast"/"slow", or a manual value of 0) maps to "auto"
        try:
            smooth_arg = float(smooth) if float(smooth) > 0 else "auto"
        except (TypeError, ValueError):
            smooth_arg = "auto"
        fit_result = fit_non_parametric(
            t_arr,
            y_arr,
            method="spline",
            smooth=smooth_arg,
        )
        if fit_result is not None:
            fit = extract_stats(
                fit_result,
                t_arr,
                y_arr,
                phase_boundary_method=phase_boundary_method,
            )
            fit["smooth"] = smooth
        else:
            fit = bad_fit_stats()

    else:  # Sliding Window (default)
        # Use non-parametric sliding window method
        fit_result = fit_non_parametric(
            t_arr,
            y_arr,
            method="sliding_window",
            window_points=int(params["window_points"]),
        )
        if fit_result is not None:
            fit = extract_stats(
                fit_result,
                t_arr,
                y_arr,
                phase_boundary_method=phase_boundary_method,
            )
            fit["window_points"] = int(params["window_points"])
        else:
            fit = bad_fit_stats()

    # No-growth detection
    no_growth_result = detect_no_growth(
        t_arr,
        y_arr,
        growth_stats=fit,
        min_data_points=int(params["min_data_points"]),
        min_signal_to_noise=float(params["min_signal_to_noise"]),
        min_od_increase=float(params["min_od_increase"]),
        min_growth_rate=float(params["min_growth_rate"]),
    )

    if no_growth_result["is_no_growth"]:
        fit = bad_fit_stats()
        fit["no_growth_reason"] = no_growth_result["reason"]
        fit_result = None
    else:
        fit["phase_boundary_method"] = phase_boundary_method

    return fit, fit_result
