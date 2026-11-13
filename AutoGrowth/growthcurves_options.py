"""Curve fitting parameter selection, matching the MicroGrowth upload page.

Layout, options, equations and explanatory figures follow
``MicroGrowth/src/ui_functions/uploads_ui.py`` (Step 3) and
``MicroGrowth/src/ui_functions/ui_components.py``. AutoGrowth additionally exposes
the minimum fold change used inside the fit (needed for turbidostat intervals).
"""

from pathlib import Path

import streamlit as st
from growthcurves.models import MODEL_REGISTRY

from growthcurve_app.styling import growth_param_table_style

INFO_PLOTS_DIR = Path(__file__).resolve().parent.parent / "info_plots"

# Phase boundaries always use the tangent to the growth curve at μmax, as in
# MicroGrowth. The cutoffs are only used by the threshold method and are kept in
# the returned options so downstream code and saved sessions keep working.
PHASE_BOUNDARY_METHOD = "tangent"
UNUSED_CUTOFF = 0.5


def _get_model_display_name(model_code: str) -> str:
    """Convert growthcurves model code to a user-friendly label."""
    display_names = {
        "mech_logistic": "Logistic",
        "mech_gompertz": "Gompertz",
        "mech_richards": "Richards",
        "mech_baranyi": "Baranyi-Roberts",
        "phenom_logistic": "Logistic",
        "phenom_gompertz": "Gompertz",
        "phenom_gompertz_modified": "Modified Gompertz",
        "phenom_richards": "Richards",
        "sliding_window": "Sliding Window",
        "spline": "Spline",
    }
    return display_names.get(model_code, model_code)


def _ui_model_selection_upload_style():
    """Render model family and growth method selectors."""
    st.caption("Select the model family and growth descriptor method:")
    family_col, method_col, param_col = st.columns([0.8, 0.8, 1.4])

    with family_col:
        model_family = st.selectbox(
            "Model family",
            options=[
                "Mechanistic parametric",
                "Phenomenological parametric",
                "Non-parametric",
            ],
            index=2,
            help=(
                "Mechanistic models use ODE-based biological principles. "
                "Phenomenological models describe growth patterns empirically. "
                "Non-parametric methods are data-driven without a fixed curve shape."
            ),
            key="batch_model_family",
        )

    if model_family == "Mechanistic parametric":
        model_family_internal = "mechanistic"
    elif model_family == "Phenomenological parametric":
        model_family_internal = "phenomenological"
    else:
        model_family_internal = "non_parametric"

    method_options = []
    if model_family == "Mechanistic parametric":
        for model_code in MODEL_REGISTRY["mechanistic"]:
            method_options.append(
                (_get_model_display_name(model_code), model_code, "Model Fitting")
            )
    elif model_family == "Phenomenological parametric":
        for model_code in MODEL_REGISTRY["phenomenological"]:
            method_options.append(
                (_get_model_display_name(model_code), model_code, "Model Fitting")
            )
    else:
        for model_code in MODEL_REGISTRY["non_parametric"]:
            growth_method = (
                "Sliding Window" if model_code == "sliding_window" else "Spline"
            )
            method_options.append(
                (_get_model_display_name(model_code), model_code, growth_method)
            )

    with method_col:
        method_labels = [m[0] for m in method_options]
        default_method_idx = (
            method_labels.index("Spline")
            if model_family == "Non-parametric" and "Spline" in method_labels
            else 0
        )
        selected_method_label = st.selectbox(
            "Growth descriptor method",
            options=method_labels,
            index=default_method_idx,
            help=(
                "Choose between non-parametric (data-driven) or "
                "parametric (model-based) approaches."
            ),
            key="batch_growth_method",
        )

    growth_method = None
    model_type = None
    for label, code, method in method_options:
        if label == selected_method_label:
            growth_method = method
            if method == "Model Fitting":
                model_type = code
            break

    return model_family_internal, growth_method, model_type, param_col


def _ui_method_params_upload_style(
    growth_method: str,
    param_col,
    min_window_points=3,
    max_window_points=200,
    default_window_points=5,
    window_step_size=1,
    default_smooth=0.0,
):
    """Render method-specific controls (window size or spline mode).

    Returns ``(window_points, smooth)`` where ``smooth`` is ``"auto"`` or a positive
    float, as accepted by ``growthcurves.non_parametric.fit_spline``.
    """
    window_points = default_window_points
    smooth = "auto"
    try:
        manual_smooth = float(default_smooth)
    except (TypeError, ValueError):
        manual_smooth = 0.0

    with param_col:
        if growth_method == "Sliding Window":
            window_points = st.number_input(
                "Minimum window size (points)",
                min_window_points,
                max_window_points,
                default_window_points,
                window_step_size,
                help=(
                    "Minimum number of consecutive data points per sliding window; "
                    "the window length adapts to the noise in the data"
                ),
                key="batch_window_points",
            )

        if growth_method == "Spline":
            mode_col, smooth_col = st.columns(2)
            mode = mode_col.radio(
                "Spline fitting mode",
                options=["auto", "manual"],
                index=1 if manual_smooth > 0 else 0,
                horizontal=True,
                format_func=lambda v: v.capitalize(),
                help=(
                    "Auto chooses the smoothing by REML. "
                    "Manual lets you set the smoothing factor (λ) directly."
                ),
                key="batch_spline_mode",
            )
            if mode == "manual":
                smooth_value = smooth_col.number_input(
                    "Smoothing factor (λ)",
                    min_value=0.0,
                    value=manual_smooth if manual_smooth > 0 else 1.0,
                    step=0.01,
                    format="%.4f",
                    help=(
                        "Spline smoothing factor λ, relative to the data. "
                        "Larger values produce a smoother (less wiggly) spline. "
                        "0 uses Auto."
                    ),
                    key="batch_spline_smooth",
                )
                smooth = float(smooth_value) if smooth_value > 0 else "auto"

    return int(window_points), smooth


def _ui_qc_filters_upload_style(
    min_data_points_default=5,
    min_signal_to_noise_default=1.0,
    min_od_increase_default=0.05,
    min_growth_rate_default=0.001,
    min_fold_default=1.5,
):
    """Render quality control filter inputs."""
    st.caption("Curves failing these criteria will be marked as no growth")
    col1, col2, col3, col4 = st.columns(4)

    min_data_points = col1.number_input(
        "Minimum data points",
        1,
        100,
        min_data_points_default,
        1,
        help="Minimum number of valid data points required for growth analysis",
        key="batch_min_data_points",
    )
    min_signal_to_noise = col2.number_input(
        "Minimum signal:noise",
        0.1,
        100.0,
        min_signal_to_noise_default,
        0.1,
        help="Minimum ratio of maximum to minimum OD signal (filters out flat curves)",
        key="batch_min_signal_to_noise",
    )
    min_od_increase = col3.number_input(
        "Minimum OD increase",
        0.0,
        None,
        min_od_increase_default,
        0.001,
        format="%.3f",
        help="Minimum absolute increase in OD from baseline to be considered growth",
        key="batch_min_od_increase",
    )
    min_growth_rate = col4.number_input(
        "Minimum growth rate",
        0.0,
        None,
        min_growth_rate_default,
        0.0001,
        format="%.4f",
        help=(
            "Minimum specific growth rate to be considered growth (wells with lower "
            "rates are marked as no growth)"
        ),
        key="batch_min_growth_rate",
    )

    # AutoGrowth only: fold-rise check applied inside the fit
    fold_col, _, _, _ = st.columns(4, vertical_alignment="bottom")
    min_fold = fold_col.number_input(
        "Minimum fold change",
        1.0,
        None,
        float(min_fold_default),
        0.1,
        format="%.2f",
        help=(
            "Minimum fold rise of the smoothed OD, from its lowest value to its peak, "
            "required to fit a curve. Applied inside the fit, before the "
            "minimum OD increase check."
        ),
        key="batch_min_fold",
    )

    return (
        int(min_data_points),
        float(min_signal_to_noise),
        float(min_od_increase),
        float(min_growth_rate),
        float(min_fold),
    )


def _info_plot_path(filename: str) -> str:
    """Build local file path for info-plot assets."""
    return str(INFO_PLOTS_DIR / filename)


def _equation(expr: str):
    """Render a method equation slightly larger than Streamlit's default."""
    st.latex(r"\large " + expr)


def _render_method_visualization_upload_style(
    growth_method: str, model_type: str | None = None
) -> str | None:
    """Render the description and equation of a growth method.

    Mechanistic model equations and parameter names follow growthcurves.models.

    Returns:
        Path to the explanatory figure, or None
    """
    figure_note = (
        "The figure shows how μ_max, the exponential phase (start and end) and max OD "
        "are calculated with this method."
    )
    if growth_method == "Sliding Window":
        st.markdown("**Sliding Window Method** (Currently Selected)")
        _equation(r"\ln(N(t)) = N_0 + b\,t")
        st.caption(
            "Local linear regression in moving windows. Calculates growth rate from "
            "nearby data points without assuming global curve shape. " + figure_note
        )
        return _info_plot_path("sliding_window.png")

    if growth_method == "Spline":
        st.markdown("**Spline Method** (Currently Selected)")
        _equation(r"\ln(N(t)) = \mathrm{spline}(t)")
        st.caption(
            "Fitted smoothed curve without underlying shape assumptions. Flexible "
            "non-parametric approach. " + figure_note
        )
        return _info_plot_path("spline.png")

    if growth_method != "Model Fitting" or not model_type:
        return None

    mechanistic = {
        "mech_logistic": (
            "Logistic",
            r"N(t) = \frac{K}{1 + \left(\frac{K}{N_0} - 1\right)e^{-\mu t}}",
            "Classic S-shaped curve with symmetric inflection point.",
            "",
        ),
        "mech_gompertz": (
            "Gompertz",
            r"N(t) = K \exp\!\left(\ln\!\left(\frac{N_0}{K}\right)e^{-\mu t}\right)",
            "Gompertz asymmetric S-curve.",
            "",
        ),
        "mech_richards": (
            "Richards",
            r"N(t) = K\left(1 + \left(\left(\frac{K}{N_0}\right)^{\beta} - 1\right)"
            r"e^{-\beta\mu t}\right)^{-1/\beta}",
            "Generalized logistic with shape parameter β. Most flexible - use when "
            "other models don't fit well.",
            ", β = shape parameter",
        ),
        "mech_baranyi": (
            "Baranyi-Roberts",
            # Package closed form; the adjusted time simplified exactly:
            # ln(1 + q0 e^{μt}) - ln(1 + q0) with q0 = 1/(e^{h0} - 1)
            # = ln(e^{μt} + e^{h0} - 1) - h0
            r"N(t) = \frac{K}{1 + \left(\frac{K}{N_0} - 1\right)e^{-\mu a(t)}}, \quad "
            r"\mu a(t) = \ln\!\left(e^{\mu t} + e^{h_0} - 1\right) - h_0",
            "Mechanistic model accounting for cell adaptation during lag phase.",
            ", h₀ = dimensionless lag parameter",
        ),
    }
    if model_type in mechanistic:
        name, solution, description, extra = mechanistic[model_type]
        st.markdown(f"**{name}** (Currently Selected)")
        _equation(solution)
        st.caption(
            f"{description} Parameters: μ = intrinsic growth rate (h⁻¹), "
            f"K = carrying capacity (max OD), N₀ = initial population at t = 0{extra}. "
            + figure_note
        )
        return _info_plot_path(f"{model_type}.png")

    # Parameters shared by all phenomenological models (growthcurves.models)
    phenom_params = (
        "Parameters: N₀ = initial OD (fitted), A = maximum increase in ln(N) "
        "(amplitude), μ_max = maximum specific growth rate (h⁻¹), λ = lag time (h)"
    )
    if model_type == "phenom_logistic":
        st.markdown("**Logistic** (Currently Selected)")
        _equation(
            r"\ln N(t) = \ln N_0 + \frac{A}{1+\exp\!\left(\frac{4\mu_{\max}}{A}"
            r"(\lambda-t)+2\right)}"
        )
        st.caption(
            "Classic S-shaped curve with symmetric inflection point. "
            + phenom_params
            + ". "
            + figure_note
        )
    elif model_type == "phenom_gompertz_modified":
        st.markdown("**Modified Gompertz** (Currently Selected)")
        _equation(
            r"\ln N(t) = \ln N_0 + A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\,e\,"
            r"(\lambda-t)}{A}+1\right)\right]+A\exp\!\left(\alpha(t-t_{\mathrm{shift}})"
            r"\right)"
        )
        st.caption(
            "Modified Gompertz asymmetric S-curve for diphasic growth. "
            + phenom_params
            + ", α = rate of the second growth phase (h⁻¹), "
            "t_shift = time shift of the second growth phase (h). " + figure_note
        )
    elif model_type == "phenom_gompertz":
        st.markdown("**Gompertz** (Currently Selected)")
        _equation(
            r"\ln N(t) = \ln N_0 + A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\,e\,"
            r"(\lambda-t)}{A}+1\right)\right]"
        )
        st.caption("Gompertz asymmetric S-curve. " + phenom_params + ". " + figure_note)
    elif model_type == "phenom_richards":
        st.markdown("**Richards** (Currently Selected)")
        _equation(
            r"\ln N(t) = \ln N_0 + A\left(1+\nu\exp\!\left(1+\nu+\frac{\mu_{\max}"
            r"(1+\nu)^{1+1/\nu}(\lambda-t)}{A}\right)\right)^{-1/\nu}"
        )
        st.caption(
            "Generalized logistic with shape parameter nu. Most flexible - use when "
            "other models don't fit well. "
            + phenom_params
            + ", nu = shape parameter. "
            + figure_note
        )

    return _info_plot_path(f"{model_type}.png")


def render_parameter_calculation_table_upload_style(options: dict):
    """Render a table summarizing how growth parameters are calculated."""
    growth_method = options.get("growth_method")
    model_type = options.get("model_type")
    model_family = options.get("model_family")

    if growth_method == "Model Fitting":
        mu_max_calc = "μ<sub>max</sub>"
        model_rmse_calc = "RMSE over entire curve"
        max_od_calc = "Maximum OD from fitted model"
    else:
        max_od_calc = "Maximum raw OD"
        if growth_method == "Sliding Window":
            mu_max_calc = "b"
            model_rmse_calc = "RMSE (OD) of the μ<sub>max</sub> window line"
        else:
            mu_max_calc = "Max spline derivative"
            model_rmse_calc = "RMSE (OD) over the spline fit range"

    if growth_method == "Model Fitting" and model_family == "mechanistic":
        intrinsic_calc = "Fitted intrinsic μ"
    else:
        intrinsic_calc = "N.a."

    # Phase boundaries always use the tangent to the growth curve at μmax
    boundary_calc = "Tangent to the growth curve at μ<sub>max</sub> meets baseline OD"
    exp_phase_end_calc = "Tangent to the growth curve at μ<sub>max</sub> meets OD(max)"

    if growth_method == "Model Fitting" and model_type in {
        "mech_baranyi",
        "phenom_logistic",
        "phenom_gompertz",
        "phenom_gompertz_modified",
        "phenom_richards",
    }:
        lag_time_calc = "λ"
    else:
        lag_time_calc = boundary_calc

    st.caption("Your selected settings will calculate growth parameters as follows:")
    growth_param_table_style()
    st.markdown(
        f"""
<div class="growth-param-table">

| OD(max) | μ<sub>max</sub> | Intrinsic Growth Rate | Doubling Time | Lag Time | μ<sub>max</sub> Time | μ<sub>max</sub> OD | Exponential End Time | RMSE |
|---|---|---|---|---|---|---|---|---|
| {max_od_calc} | {mu_max_calc} | {intrinsic_calc} | ln(2) / μ<sub>max</sub> | {lag_time_calc} | Time at μ<sub>max</sub> | OD at μ<sub>max</sub> | {exp_phase_end_calc} | {model_rmse_calc} |

</div>
""",
        unsafe_allow_html=True,
    )


def render_upload_style_analysis_options(
    min_window_points=3,
    max_window_points=200,
    default_window_points=5,
    window_step_size=1,
    min_data_points_default=5,
    min_signal_to_noise_default=1.0,
    min_od_increase_default=0.05,
    min_growth_rate_default=0.001,
    min_fold_default=1.5,
    default_smooth=0.0,
):
    """Render curve fitting options laid out as on the MicroGrowth upload page."""
    # Two columns: model options | how the selected method calculates parameters
    model_col, help_col = st.columns(2, gap="large")

    with model_col:
        model_family, growth_method, model_type, param_col = (
            _ui_model_selection_upload_style()
        )
        window_points, smooth_mode = _ui_method_params_upload_style(
            growth_method,
            param_col,
            min_window_points,
            max_window_points,
            default_window_points,
            window_step_size,
            default_smooth,
        )

        st.write("")
        st.write("")

        (
            min_data_points,
            min_signal_to_noise,
            min_od_increase,
            min_growth_rate,
            min_fold,
        ) = _ui_qc_filters_upload_style(
            min_data_points_default=min_data_points_default,
            min_signal_to_noise_default=min_signal_to_noise_default,
            min_od_increase_default=min_od_increase_default,
            min_growth_rate_default=min_growth_rate_default,
            min_fold_default=min_fold_default,
        )

        # Equation and description of the selected method
        st.write("")
        method_image = _render_method_visualization_upload_style(
            growth_method, model_type
        )

    with help_col:
        if method_image is not None:
            st.image(method_image, width="stretch")

    if growth_method == "Model Fitting":
        selected_model = model_type
    elif growth_method == "Sliding Window":
        selected_model = "sliding_window"
    else:
        selected_model = "spline"

    return {
        "selected_model": selected_model,
        "window_points": window_points,
        "phase_boundary_method": PHASE_BOUNDARY_METHOD,
        "lag_cutoff": UNUSED_CUTOFF,
        "exp_cutoff": UNUSED_CUTOFF,
        "min_data_points": min_data_points,
        "min_signal_to_noise": min_signal_to_noise,
        "min_od_increase": min_od_increase,
        "min_growth_rate": min_growth_rate,
        "min_fold": min_fold,
        "growth_method": growth_method,
        "model_family": model_family,
        "model_type": model_type,
        "smooth_mode": smooth_mode,
    }
