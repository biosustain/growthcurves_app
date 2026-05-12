from pathlib import Path

import growthcurves as gc
import streamlit as st
from growthcurves.models import MODEL_REGISTRY

INFO_PLOTS_DIR = Path(__file__).resolve().parent.parent / "info_plots"


def render_options_for_growthcurve_fitting(s_min=3, s_max=1000, s_default=1000):
    st.write("### Model selection")
    selected_model = st.selectbox(
        "See the differences between the options in the"
        " [growthcurves package](https://growthcurves.readthedocs.io)",
        gc.get_all_models(),
        index=7,
    )
    st.session_state["selected_model"] = selected_model
    st.write("#### Spline fitting options:")
    spline_smoothing_value = st.slider(
        "Smoothing of the spline fitted to OD values (zero means no smoothing). "
        "Range suggested using scipy, see "
        "[docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html)",
        1,
        s_max,
        s_default,
        step=1,
    )
    st.write("#### Sliding window options:")
    fits_sliding_window = st.slider(
        "Number of fits used for sliding window to calculate derivatives",
        5,
        200,
        50,
        step=5,
    )
    window_size = st.slider(
        "Window size for sliding window method (in hours)",
        3,
        1000,
        300,
        step=3,
    )
    # ! Add tangent and threshold method options here
    tangent_cols = st.columns(2)
    phase_boundary_method = tangent_cols[0].radio(
        "Select method for exponential phase detection (default recommended):",
        ["default", "tangent", "threshold"],
        help=("""
            Default picks for parametric models the threshold method and
            for phenomenological models, the sliding window and spline method the
            tangent method.

            In short:

            - "threshold": Threshold-based method using fractions of μ_max
            - "tangent": Tangent line method at point of maximum growth rate
            """),
        index=0,
    )
    exp_frac = tangent_cols[1].slider(
        "Define percentage of µmax considered as high for threshold method",
        0,
        100,
        90,
        step=1,
    )
    return (
        selected_model,
        spline_smoothing_value,
        fits_sliding_window,
        window_size,
        phase_boundary_method,
        exp_frac,
    )


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
    family_col, method_col, param_col = st.columns(3)

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
            method_labels.index("Sliding Window")
            if model_family == "Non-parametric" and "Sliding Window" in method_labels
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
    s_min: int,
    s_max: int,
    min_window_points=5,
    max_window_points=200,
    default_window_points=10,
    window_step_size=1,
):
    """Render method-specific controls and convert to spline_s/window_points values."""
    with param_col:
        if growth_method == "Sliding Window":
            window_points = st.number_input(
                "Window size (points)",
                min_window_points,
                max_window_points,
                default_window_points,
                window_step_size,
                help=(
                    "Number of consecutive data points used for sliding window "
                    "linear fit to determine maximum growth rate."
                ),
                key="batch_window_points",
            )
            smooth_mode = "fast"
        elif growth_method == "Spline":
            window_points = default_window_points
            smooth_mode = st.radio(
                "Spline fitting mode",
                options=["fast", "slow"],
                index=0,
                horizontal=True,
                format_func=lambda v: v.capitalize(),
                help=(
                    "Fast uses stronger default smoothing. "
                    "Slow uses lighter smoothing."
                ),
                key="batch_spline_mode",
            )
        else:
            window_points = default_window_points
            smooth_mode = "fast"

    if growth_method == "Spline":
        spline_s = int(s_max if smooth_mode == "fast" else max(s_min, 1))
    else:
        spline_s = int(s_max)

    return int(window_points), smooth_mode, spline_s


# ! to update to save last set state
def _ui_qc_filters_upload_style(
    min_data_points_default=5,
    min_signal_to_noise_default=1.0,
    min_od_increase_default=0.05,
    min_growth_rate_default=0.001,
):
    """Render quality control filter inputs."""
    st.caption("Wells failing these criteria will be marked as no growth")
    col1, col2, col3, col4 = st.columns(4)

    min_data_points = col1.number_input(
        "Minimum data points",
        1,
        100,
        min_data_points_default,
        1,
        help="Minimum number of valid data points required for growth analysis.",
        key="batch_min_data_points",
    )
    min_signal_to_noise = col2.number_input(
        "Minimum signal:noise",
        0.1,
        100.0,
        min_signal_to_noise_default,
        0.1,
        help="Minimum ratio of maximum to minimum OD signal.",
        key="batch_min_signal_to_noise",
    )
    min_od_increase = col3.number_input(
        "Minimum OD increase",
        0.0,
        None,
        min_od_increase_default,
        0.001,
        format="%.3f",
        help="Minimum absolute increase in OD to be considered growth.",
        key="batch_min_od_increase",
    )
    min_growth_rate = col4.number_input(
        "Minimum growth rate",
        0.0,
        None,
        min_growth_rate_default,
        0.0001,
        format="%.4f",
        help="Minimum specific growth rate to be considered growth.",
        key="batch_min_growth_rate",
    )

    return (
        int(min_data_points),
        float(min_signal_to_noise),
        float(min_od_increase),
        float(min_growth_rate),
    )


def _ui_phase_boundaries_upload_style():
    """Render phase boundary controls."""
    st.caption(
        "Phase boundaries define when the lag phase ends and when the exponential phase ends."
    )
    phase_boundary_method = st.selectbox(
        "Phase boundary calculation",
        options=["threshold", "tangent"],
        index=1,
        format_func=lambda v: v.capitalize(),
        help=(
            "Threshold uses fractions of μ_max; tangent uses the tangent at μ_max "
            "to estimate exponential phase bounds."
        ),
        key="batch_phase_boundary_method",
    )
    lag_cutoff = st.number_input(
        "Lag phase cutoff",
        0.01,
        0.5,
        0.5,
        0.01,
        format="%.2f",
        disabled=phase_boundary_method == "tangent",
        help="Fraction of maximum growth rate used to define lag phase end.",
        key="batch_lag_cutoff",
    )
    exp_cutoff = st.number_input(
        "Exponential phase cutoff",
        0.01,
        0.5,
        0.5,
        0.01,
        format="%.2f",
        disabled=phase_boundary_method == "tangent",
        help="Fraction of maximum growth rate used to define exponential phase end.",
        key="batch_exp_cutoff",
    )

    return phase_boundary_method, float(lag_cutoff), float(exp_cutoff)


def _info_plot_url(filename: str) -> str:
    """Build local file path for info-plot assets."""
    return str(INFO_PLOTS_DIR / filename)


def _render_method_visualization_upload_style(
    growth_method: str, model_type: str | None = None
) -> str | None:
    """Render method visualization text and return image URL."""
    if growth_method == "Sliding Window":
        st.markdown("**Sliding Window Method** (Currently Selected)")
        st.latex(r"\ln(N(t)) = N_0 + b\,t")
        st.caption(
            "Local linear regression in moving windows. Calculates growth rate from nearby data points without assuming global curve shape."
        )
        return _info_plot_url("sliding_window.png")

    if growth_method == "Spline":
        st.markdown("**Spline Method** (Currently Selected)")
        st.latex(r"\ln(N(t)) = \mathrm{spline}(t)")
        st.caption(
            "Fitted smoothed curve without underlying shape assumptions. Flexible non-parametric approach."
        )
        return _info_plot_url("spline.png")

    if growth_method == "Model Fitting" and model_type:
        if "logistic" in str(model_type):
            st.markdown("**Logistic** (Currently Selected)")
            if str(model_type).startswith("mech_"):
                st.latex(r"\frac{dN}{dt} = \mu\left(1-\frac{N}{K}\right)N")
            else:
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right) = \frac{A}{1+\exp\!\left(\frac{4\mu_{\max}(\lambda-t)}{A}+2\right)}"
                )
            st.caption(
                "Classic S-shaped curve with symmetric inflection point. Most commonly used for microbial growth."
            )
        elif "gompertz" in str(model_type):
            model_name = (
                "Modified Gompertz" if "modified" in str(model_type) else "Gompertz"
            )
            st.markdown(f"**{model_name}** (Currently Selected)")
            if str(model_type).startswith("mech_"):
                st.latex(r"\frac{dN}{dt} = \mu\log\!\left(\frac{K}{N}\right)N")
            elif "modified" in str(model_type):
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right)=A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\exp(1)(\lambda-t)}{A}+1\right)\right]+A\exp\!\left(\alpha(t-t_{\mathrm{shift}})\right)"
                )
            else:
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right)=A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\exp(1)(\lambda-t)}{A}+1\right)\right]"
                )
            st.caption(
                "Modified Gompertz with baseline offset y₀ and amplitude A = K − y₀. Asymmetric S-curve; often fits bacterial growth better than logistic."
            )
        elif "richards" in str(model_type):
            st.markdown("**Richards** (Currently Selected)")
            if str(model_type).startswith("mech_"):
                st.latex(
                    r"\frac{dN}{dt}=\mu\left(1-\left(\frac{N}{K}\right)^{\beta}\right)N"
                )
            else:
                st.latex(
                    r"\ln\!\left(\frac{N(t)}{N_0}\right)=A\left(1+\nu\exp\!\left(1+\nu+\frac{\mu_{\max}(1+\nu)^{1/\nu}(\lambda-t)}{A}\right)\right)^{-1/\nu}"
                )
            st.caption(
                "Generalized logistic with shape parameter ν. Most flexible - use when other models don't fit well."
            )
        elif "baranyi" in str(model_type):
            st.markdown("**Baranyi-Roberts** (Currently Selected)")
            st.latex(
                r"\frac{dN}{dt}=\mu\frac{\exp(\mu t)}{\exp(\lambda)-1+\exp(\mu t)}\left(1-\frac{N}{K}\right)N"
            )
            st.caption(
                "Baranyi-Roberts model with physiological lag parameter λ. Mechanistic model accounting for cell adaptation during lag phase."
            )

        return _info_plot_url(f"{model_type}.png")

    return None


def _render_phase_boundary_visualization_upload_style(
    phase_boundary_method: str,
) -> str | None:
    """Render phase-boundary text and return image URL."""
    if phase_boundary_method == "threshold":
        st.markdown("**Threshold Method** (Currently Selected)")
        st.latex(r"\text{Lag end: } \mu(t) > f_{\text{lag}} \cdot \mu_{\max}")
        st.caption(
            "Uses threshold fractions of μ_max to identify phase transitions. Adjustable sensitivity via cutoff parameters."
        )
        return _info_plot_url("threshold_demo.png")

    st.markdown("**Tangent Method** (Currently Selected)")
    st.latex(r"\text{Tangent at } \mu_{\max} \text{ intersects baseline and plateau}")
    st.caption(
        "Geometric definition based on tangent line at maximum growth rate. No arbitrary thresholds required."
    )
    return _info_plot_url("tangent_demo.png")


def render_parameter_calculation_table_upload_style(options: dict):
    """Render a markdown table summarizing how growth parameters are calculated."""
    growth_method = options.get("growth_method")
    model_type = options.get("model_type")
    model_family = options.get("model_family")
    phase_boundary_method = options.get("phase_boundary_method")
    lag_cutoff = float(options.get("lag_cutoff", 0.5))
    exp_cutoff = float(options.get("exp_cutoff", 0.5))
    window_points = int(options.get("window_points", 10))

    if growth_method == "Model Fitting":
        mu_max_calc = "μ<sub>max</sub>"
        model_rmse_calc = "RMSE over entire curve"
        max_od_calc = "Maximum OD from fitted model"
    else:
        max_od_calc = "Maximum raw OD"
        if growth_method == "Sliding Window":
            mu_max_calc = "b"
            model_rmse_calc = f"RMSE over {window_points} point sliding-window"
        else:
            mu_max_calc = "Max spline derivative"
            model_rmse_calc = "RMSE over spline fit window (log phase)"

    if growth_method == "Model Fitting" and model_family == "mechanistic":
        intrinsic_calc = "Fitted intrinsic μ"
    else:
        intrinsic_calc = "N.a."

    if phase_boundary_method == "threshold":
        boundary_calc = f"Time at instantaneous μ > {lag_cutoff:.0%} μ<sub>max</sub>"
        exp_phase_end_calc = (
            f"Time at instantaneous μ < {exp_cutoff:.0%} μ<sub>max</sub>"
        )
    else:
        boundary_calc = "μ<sub>max</sub> tangent intersect with OD baseline"
        exp_phase_end_calc = "μ<sub>max</sub> tangent intersect with OD(max)"

    if growth_method == "Model Fitting" and model_type in {
        "phenom_logistic",
        "phenom_gompertz",
        "phenom_gompertz_modified",
        "phenom_richards",
    }:
        lag_time_calc = "λ"
    else:
        lag_time_calc = boundary_calc

    st.caption("Your selected settings will calculate growth parameters as follows:")
    st.markdown(
        f"""
<div class="param-calc-table">
    <table>
        <thead>
            <tr>
                <th>OD(max)</th>
                <th>μ<sub>max</sub></th>
                <th>Intrinsic Growth Rate</th>
                <th>Doubling Time</th>
                <th>Lag Time</th>
                <th>μ<sub>max</sub> Time</th>
                <th>μ<sub>max</sub> OD</th>
                <th>Exponential End Time</th>
                <th>RMSE</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>{max_od_calc}</td>
                <td>{mu_max_calc}</td>
                <td>{intrinsic_calc}</td>
                <td>ln(2) / μ<sub>max</sub></td>
                <td>{lag_time_calc}</td>
                <td>Time at μ<sub>max</sub></td>
                <td>OD at μ<sub>max</sub></td>
                <td>{exp_phase_end_calc}</td>
                <td>{model_rmse_calc}</td>
            </tr>
        </tbody>
    </table>
</div>
<style>
div.param-calc-table table {{
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 1rem;
}}
div.param-calc-table th,
div.param-calc-table td {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    padding: 0.6rem 0.55rem;
    text-align: center;
    vertical-align: middle;
    line-height: 1.35;
}}
div.param-calc-table th {{
    font-size: 1.02rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )
    st.markdown("""
    For more information see the
    [growthcurves documentation](https://growthcurves.readthedocs.io/),
    especially the tutorial on fitting curves with growthcurves
    [here](https://growthcurves.readthedocs.io/en/latest/tutorial/analysis.html)
    """)


def render_upload_style_analysis_options(
    s_min=3,
    s_max=1000,
    min_window_points=5,
    max_window_points=200,
    default_window_points=10,
    window_step_size=1,
    min_data_points_default=5,
    min_signal_to_noise_default=1.0,
    min_od_increase_default=0.05,
    min_growth_rate_default=0.001,
):
    """Render analysis options aligned with TheGrowthAnalysisApp upload page."""
    model_col, boundary_col = st.columns(2, gap="large")

    with model_col:
        model_family, growth_method, model_type, param_col = (
            _ui_model_selection_upload_style()
        )
        window_points, smooth_mode, spline_smoothing_value = (
            _ui_method_params_upload_style(
                growth_method,
                param_col,
                s_min,
                s_max,
                min_window_points,
                max_window_points,
                default_window_points,
                window_step_size,
            )
        )
        st.write("")
        st.write("")
        (
            min_data_points,
            min_signal_to_noise,
            min_od_increase,
            min_growth_rate,
        ) = _ui_qc_filters_upload_style(
            min_data_points_default=min_data_points_default,
            min_signal_to_noise_default=min_signal_to_noise_default,
            min_od_increase_default=min_od_increase_default,
            min_growth_rate_default=min_growth_rate_default,
        )

    with boundary_col:
        phase_boundary_method, lag_cutoff, exp_cutoff = (
            _ui_phase_boundaries_upload_style()
        )

    st.write("")
    help_model_col, help_boundary_col = st.columns(2, gap="large")
    with help_model_col:
        method_image = _render_method_visualization_upload_style(
            growth_method, model_type
        )
    with help_boundary_col:
        boundary_image = _render_phase_boundary_visualization_upload_style(
            phase_boundary_method
        )

    graph_col_model, graph_col_boundary = st.columns(2, gap="large")
    with graph_col_model:
        if method_image is not None:
            st.image(method_image, width="stretch")
    with graph_col_boundary:
        if boundary_image is not None:
            st.image(boundary_image, width="stretch")

    if growth_method == "Model Fitting":
        selected_model = model_type
    elif growth_method == "Sliding Window":
        selected_model = "sliding_window"
    else:
        selected_model = "spline"

    return {
        "selected_model": selected_model,
        "spline_smoothing_value": spline_smoothing_value,
        "n_fits": 50,
        "window_points": window_points,
        "phase_boundary_method": phase_boundary_method,
        "lag_cutoff": lag_cutoff,
        "exp_cutoff": exp_cutoff,
        "min_data_points": min_data_points,
        "min_signal_to_noise": min_signal_to_noise,
        "min_od_increase": min_od_increase,
        "min_growth_rate": min_growth_rate,
        "growth_method": growth_method,
        "model_family": model_family,
        "model_type": model_type,
        "smooth_mode": smooth_mode,
    }
