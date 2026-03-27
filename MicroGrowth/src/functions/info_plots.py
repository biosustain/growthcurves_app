"""Generate and save informational plots for popovers.

This module generates static PNG images for informational plots used in
popovers throughout the application. Run this script directly to regenerate
all info plots.
"""

from pathlib import Path

import growthcurves.plot as gc_plot
import numpy as np
from growthcurves.inference import extract_stats
from growthcurves.parametric import fit_parametric

MODEL_DISPLAY_NAMES = {
    "mech_logistic": "Logistic (mechanistic)",
    "mech_gompertz": "Gompertz (mechanistic)",
    "mech_richards": "Richards (mechanistic)",
    "mech_baranyi": "Baranyi-Roberts (mechanistic)",
    "phenom_logistic": "Logistic (phenomenological)",
    "phenom_gompertz": "Gompertz (phenomenological)",
    "phenom_gompertz_modified": "Modified Gompertz (phenomenological)",
    "phenom_richards": "Richards (phenomenological)",
}


def create_model_curve_plot(model_name: str, svg_save_path: Path):
    """Create and save a minimal SVG of the fitted curve for one growth model."""
    t = np.linspace(0, 20, 100)
    K = 1.0
    r = 0.5
    N0 = 0.05
    y = K / (1 + ((K - N0) / N0) * np.exp(-r * t))

    np.random.seed(42)
    y_noisy = y + np.random.normal(0, 0.02, len(y))

    fit_result = fit_parametric(t, y_noisy, method=model_name)
    if fit_result is None:
        print(f"  Warning: fitting failed for {model_name}, skipping.")
        return

    growth_stats = extract_stats(fit_result, t, y_noisy)

    fig = gc_plot.create_base_plot(t, y_noisy, scale="linear")
    fig = gc_plot.annotate_plot(
        fig,
        fit_result=fit_result,
        stats=growth_stats,
        show_fitted_curve=True,
        show_phase_boundaries=False,
        show_crosshairs=False,
        show_od_max_line=False,
        show_n0_line=False,
        show_umax_marker=False,
        show_tangent=False,
        scale="linear",
    )

    # Keep only the fitted curve trace (remove raw data point markers)
    fig.data = tuple(
        trace for trace in fig.data if getattr(trace, "mode", "") != "markers"
    )

    # Set curve colour
    for trace in fig.data:
        if hasattr(trace, "line"):
            trace.line.color = "#a6dcea"

    fig.update_layout(
        title=None,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    fig.write_image(svg_save_path, format="svg", width=1977, height=989)


def create_annotation_demo_plot_png(save_path: Path, svg_save_path: Path | None = None):
    """Create and save a demo plot showing what each annotation represents."""
    # Create synthetic growth curve data
    t = np.linspace(0, 20, 100)
    # Logistic growth curve parameters
    K = 1.0  # carrying capacity
    r = 0.5  # growth rate
    N0 = 0.05  # initial population
    y = K / (1 + ((K - N0) / N0) * np.exp(-r * t))

    # Add some noise
    np.random.seed(42)
    y_noisy = y + np.random.normal(0, 0.02, len(y))

    # Fit a Richards model to the noisy data using growthcurves
    fit_result = fit_parametric(t, y_noisy, method="mech_richards")

    # Extract growth stats from the fit object
    growth_stats = extract_stats(fit_result, t, y_noisy)

    # Get annotation positions from growth_stats
    exp_start = growth_stats.get("exp_phase_start")
    exp_end = growth_stats.get("exp_phase_end")
    time_umax = growth_stats.get("time_at_umax")
    od_umax = growth_stats.get("od_at_umax")
    od_max = growth_stats.get("max_od")

    # Create base plot using growthcurves
    fig = gc_plot.create_base_plot(t, y_noisy, scale="linear")

    # Annotate plot with all features
    fig = gc_plot.annotate_plot(
        fig,
        fit_result=fit_result,
        stats=growth_stats,
        show_fitted_curve=True,
        show_phase_boundaries=True,
        show_crosshairs=True,
        show_od_max_line=True,
        show_n0_line=True,
        show_umax_marker=True,
        show_tangent=True,
        scale="linear",
    )

    # Add text annotations/labels to highlight each feature
    # Build annotations dynamically based on what's available
    # Position labels to avoid overlaps
    annotations = []

    # 1. Phase boundaries - label the shaded green region
    if exp_start is not None and exp_end is not None:
        annotations.append(
            {
                "x": (exp_start + exp_end) / 2,
                "y": od_max * 0.95 if od_max else 0.95,
                "text": "Phase boundaries",
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "black",
                "borderwidth": 1,
                "font": dict(size=11),
            }
        )

    # 2. Max growth rate point - label the green dot with arrow
    if time_umax is not None and od_umax is not None:
        annotations.append(
            {
                "x": time_umax,
                "y": od_umax,
                "text": "Max growth<br>rate point",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1,
                "arrowwidth": 2,
                "arrowcolor": "#66BB6A",
                "ax": -60,
                "ay": -50,
                "axref": "pixel",
                "ayref": "pixel",
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "black",
                "borderwidth": 1,
                "font": dict(size=11),
            }
        )

    # 3. Max OD - label the horizontal line at top right
    if od_max is not None:
        annotations.append(
            {
                "x": 19,
                "y": od_max,
                "text": "Max OD",
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "bottom",
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "black",
                "borderwidth": 1,
                "font": dict(size=11),
            }
        )

    # 4. Baseline OD (N0) - label the horizontal line at bottom left
    n0 = growth_stats.get("N0")
    if n0 is not None:
        annotations.append(
            {
                "x": 1,
                "y": n0,
                "text": "Baseline OD",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "black",
                "borderwidth": 1,
                "font": dict(size=11),
            }
        )

    # 5. Tangent line - label the green dashed line
    if time_umax is not None and od_umax is not None:
        annotations.append(
            {
                "x": time_umax + 2,
                "y": od_umax * 1.4,
                "text": "Tangent line",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1,
                "arrowwidth": 2,
                "arrowcolor": "green",
                "ax": 40,
                "ay": 20,
                "axref": "pixel",
                "ayref": "pixel",
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "black",
                "borderwidth": 1,
                "font": dict(size=11),
            }
        )

    # 6. Fitted model curve - label the blue curve
    annotations.append(
        {
            "x": 11,
            "y": 0.78,
            "text": "Fitted model<br>curve",
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1,
            "arrowwidth": 2,
            "arrowcolor": "blue",
            "ax": 0,
            "ay": -40,
            "axref": "pixel",
            "ayref": "pixel",
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": dict(size=11),
        }
    )

    # Add all text annotations
    for annot in annotations:
        fig.add_annotation(annot)

    # Update layout
    fig.update_layout(
        title="Annotation Guide",
        xaxis_title="Time (hours)",
        yaxis_title="OD600 (baseline-corrected)",
        height=400,
        width=800,
        showlegend=False,
    )

    # Save as PNG
    fig.write_image(save_path, format="png", width=800, height=400, scale=2)

    # Save as SVG if path provided
    if svg_save_path is not None:
        fig.write_image(svg_save_path, format="svg", width=800, height=400)


def save_all_info_plots():
    """Generate and save all informational plots as PNG files and SVG curves."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    info_plots_dir = project_root / "info_plots"
    curves_dir = info_plots_dir / "curves"

    # Create directories if they don't exist
    info_plots_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)

    # Generate and save annotation demo plot
    print("Generating annotation demo plot...")
    annotation_path = info_plots_dir / "annotation_demo.png"
    annotation_svg_path = curves_dir / "annotation_demo.svg"
    create_annotation_demo_plot_png(annotation_path, svg_save_path=annotation_svg_path)
    print(f"Saved annotation demo plot to {annotation_path}")
    print(f"Saved annotation demo SVG to {annotation_svg_path}")

    # Generate and save one SVG curve per model
    print("\nGenerating model curve SVGs...")
    for model_name in MODEL_DISPLAY_NAMES:
        print(f"  {model_name}...")
        svg_path = curves_dir / f"{model_name}.svg"
        create_model_curve_plot(model_name, svg_path)
        print(f"    Saved SVG: {svg_path}")

    print("\nAll info plots generated successfully!")


if __name__ == "__main__":
    save_all_info_plots()
