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


def ui_method_visualization(growth_method: str, model_type: str = None):
    """
    Render visualization for selected growth method with description and equation.

    Args:
        growth_method: One of "Sliding Window", "Spline", "Model Fitting"
        model_type: Model type for parametric methods (e.g., "mech_logistic")

    Returns:
        Plotly figure object or None
    """
    if growth_method == "Sliding Window":
        st.markdown("**Sliding Window Method** (Currently Selected)")
        st.latex(r"\ln(N(t)) = N_0 + b\,t")
        st.caption(
            "Local linear regression in moving windows. Calculates growth rate from nearby data points without assuming global curve shape."
        )
        return "MicroGrowth/info_plots/sliding_window.png"

    elif growth_method == "Spline":
        st.markdown("**Spline Method** (Currently Selected)")
        st.latex(r"\ln(N(t)) = \mathrm{spline}(t)")
        st.caption(
            "Fitted smoothed curve without underlying shape assumptions. Flexible non-parametric approach."
        )
        return "MicroGrowth/info_plots/spline.png"

    elif growth_method == "Model Fitting" and model_type:
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

        return f"MicroGrowth/info_plots/{model_type}.png"

    return None


def ui_phase_boundary_visualization(phase_boundary_method: str):
    """
    Render phase boundary method visualization with description and equation.

    Args:
        phase_boundary_method: One of "threshold" or "tangent"

    Returns:
        Path to image file
    """
    if phase_boundary_method == "threshold":
        st.markdown("**Threshold Method** (Currently Selected)")
        st.latex(r"\text{Lag end: } \mu(t) > f_{\text{lag}} \cdot \mu_{\max}")
        st.caption(
            "Uses threshold fractions of μ_max to identify phase transitions. Adjustable sensitivity via cutoff parameters."
        )
        return "MicroGrowth/info_plots/threshold_demo.png"
    else:  # tangent
        st.markdown("**Tangent Method** (Currently Selected)")
        st.latex(
            r"\text{Tangent at } \mu_{\max} \text{ intersects baseline and plateau}"
        )
        st.caption(
            "Geometric definition based on tangent line at maximum growth rate. No arbitrary thresholds required."
        )
        return "MicroGrowth/info_plots/tangent_demo.png"
