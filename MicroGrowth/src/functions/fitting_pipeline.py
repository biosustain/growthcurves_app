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
    growth_method = params.get("growth_method", "Spline")
    lag_frac = float(params.get("lag_cutoff", 0.15))
    exp_frac = float(params.get("exp_cutoff", 0.15))
    phase_boundary_method = str(params.get("phase_boundary_method", "tangent")).lower()

    fit_result = None

    # Method selection (parametric, spline, or sliding window)
    if growth_method == "Model Fitting":
        # Use parametric model fitting
        model_type = params.get("model_type", "mech_logistic")
        fit_result = fit_parametric(t_arr, y_arr, method=model_type)
        if fit_result is not None:
            fit = extract_stats(
                fit_result,
                t_arr,
                y_arr,
                lag_frac=lag_frac,
                exp_frac=exp_frac,
                phase_boundary_method=phase_boundary_method,
            )
            # Store model parameters
            for param_name, param_val in fit_result["params"].items():
                fit[f"fit_param_{param_name}"] = float(param_val)
        else:
            fit = bad_fit_stats()

    elif growth_method == "Spline":
        # Use non-parametric spline method
        smooth = params.get("smooth", "fast")
        fit_result = fit_non_parametric(
            t_arr,
            y_arr,
            method="spline",
            smooth=smooth,
            exp_start=lag_frac,
            exp_end=exp_frac,
            sg_window=int(params.get("sg_window", 11)),
            sg_poly=int(params.get("sg_poly", 1)),
            use_weights=False,
        )
        if fit_result is not None:
            fit = extract_stats(
                fit_result,
                t_arr,
                y_arr,
                lag_frac=lag_frac,
                exp_frac=exp_frac,
                phase_boundary_method=phase_boundary_method,
            )
            actual_params = fit_result.get("params") or {}
            fit["spline_s"] = actual_params.get("spline_s")
            fit["smooth"] = actual_params.get("smooth", smooth)
        else:
            fit = bad_fit_stats()

    else:  # Sliding Window (default)
        # Use non-parametric sliding window method
        fit_result = fit_non_parametric(
            t_arr,
            y_arr,
            method="sliding_window",
            window_points=int(params.get("window_points", 7)),
        )
        if fit_result is not None:
            fit = extract_stats(
                fit_result,
                t_arr,
                y_arr,
                lag_frac=lag_frac,
                exp_frac=exp_frac,
                phase_boundary_method=phase_boundary_method,
            )
            fit["window_points"] = int(params.get("window_points", 7))
        else:
            fit = bad_fit_stats()

    # No-growth detection
    no_growth_result = detect_no_growth(
        t_arr,
        y_arr,
        growth_stats=fit,
        min_data_points=int(params.get("min_data_points", 5)),
        min_signal_to_noise=float(params.get("min_signal_to_noise", 1.0)),
        min_od_increase=float(params.get("min_od_increase", 0.05)),
        min_growth_rate=float(params.get("min_growth_rate", 0.001)),
    )

    if no_growth_result["is_no_growth"]:
        fit = bad_fit_stats()
        fit["no_growth_reason"] = no_growth_result["reason"]
        fit_result = None
    else:
        fit["phase_boundary_method"] = phase_boundary_method

    return fit, fit_result
