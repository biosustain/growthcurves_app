"""Reusable UI components for Streamlit pages."""

import streamlit as st


def page_header_with_help(title: str, help_text: str):
    """Render standard page header with help popover."""
    title_col, popover_col = st.columns([9, 2])
    with title_col:
        st.title(title)
    with popover_col:
        st.write("")
        with st.popover("Help", width="stretch"):
            st.markdown(help_text)


def _equation(expr: str):
    """Render a method equation slightly larger than Streamlit's default."""
    st.latex(r"\large " + expr)


def ui_method_visualization(growth_method: str, model_type: str = None):
    """
    Render the description, equations and explanatory figure for a growth method.

    Mechanistic model equations and parameter names follow growthcurves.models.

    Args:
        growth_method: One of "Sliding Window", "Spline", "Model Fitting"
        model_type: Model type for parametric methods (e.g., "mech_logistic")

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
            "Local linear regression in moving windows. Calculates growth rate from nearby data points without assuming global curve shape. "
            + figure_note
        )
        return "MicroGrowth/info_plots/sliding_window.png"

    if growth_method == "Spline":
        st.markdown("**Spline Method** (Currently Selected)")
        _equation(r"\ln(N(t)) = \mathrm{spline}(t)")
        st.caption(
            "Fitted smoothed curve without underlying shape assumptions. Flexible non-parametric approach. "
            + figure_note
        )
        return "MicroGrowth/info_plots/spline.png"

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
            "Generalized logistic with shape parameter β. Most flexible - use when other models don't fit well.",
            ", β = shape parameter",
        ),
        "mech_baranyi": (
            "Baranyi-Roberts",
            # Package closed form; the adjusted time simplified exactly:
            # ln(1 + q0 e^{μt}) - ln(1 + q0) with q0 = 1/(e^{h0} - 1) = ln(e^{μt} + e^{h0} - 1) - h0
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
        return f"MicroGrowth/info_plots/{model_type}.png"

    # Parameters shared by all phenomenological models (growthcurves.models)
    phenom_params = (
        "Parameters: N₀ = initial OD (fitted), A = maximum increase in ln(N) (amplitude), "
        "μ_max = maximum specific growth rate (h⁻¹), λ = lag time (h)"
    )
    if model_type == "phenom_logistic":
        st.markdown("**Logistic** (Currently Selected)")
        _equation(
            r"\ln N(t) = \ln N_0 + \frac{A}{1+\exp\!\left(\frac{4\mu_{\max}}{A}(\lambda-t)+2\right)}"
        )
        st.caption(
            "Classic S-shaped curve with symmetric inflection point. "
            + phenom_params
            + ". "
            + figure_note
        )
    elif model_type in ("phenom_gompertz", "phenom_gompertz_modified"):
        if model_type == "phenom_gompertz_modified":
            st.markdown("**Modified Gompertz** (Currently Selected)")
            _equation(
                r"\ln N(t) = \ln N_0 + A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\,e\,(\lambda-t)}{A}+1\right)\right]+A\exp\!\left(\alpha(t-t_{\mathrm{shift}})\right)"
            )
            st.caption(
                "Modified Gompertz asymmetric S-curve for diphasic growth. "
                + phenom_params
                + ", α = rate of the second growth phase (h⁻¹), "
                "t_shift = time shift of the second growth phase (h). " + figure_note
            )
        else:
            st.markdown("**Gompertz** (Currently Selected)")
            _equation(
                r"\ln N(t) = \ln N_0 + A\exp\!\left[-\exp\!\left(\frac{\mu_{\max}\,e\,(\lambda-t)}{A}+1\right)\right]"
            )
            st.caption(
                "Gompertz asymmetric S-curve. " + phenom_params + ". " + figure_note
            )
    elif model_type == "phenom_richards":
        st.markdown("**Richards** (Currently Selected)")
        _equation(
            r"\ln N(t) = \ln N_0 + A\left(1+\nu\exp\!\left(1+\nu+\frac{\mu_{\max}(1+\nu)^{1+1/\nu}(\lambda-t)}{A}\right)\right)^{-1/\nu}"
        )
        st.caption(
            "Generalized logistic with shape parameter ν. Most flexible - use when other models don't fit well. "
            + phenom_params
            + ", ν = shape parameter. "
            + figure_note
        )

    return f"MicroGrowth/info_plots/{model_type}.png"
