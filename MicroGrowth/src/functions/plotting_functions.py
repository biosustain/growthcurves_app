"""Plotting utilities for growth curves, stats, and window fits."""

import re

import growthcurves.plot as gc_plot
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from growthcurves.inference import (
    compute_first_derivative,
    compute_instantaneous_mu,
    compute_sliding_window_growth_rate,
    is_no_growth,
    smooth,
)
from growthcurves.models import (
    MODEL_REGISTRY,
    mech_baranyi_model,
    mech_gompertz_model,
    mech_logistic_model,
    mech_richards_model,
    phenom_gompertz_model,
    phenom_gompertz_modified_model,
    phenom_logistic_model,
    phenom_richards_model,
    spline_from_params,
)
from plotly.subplots import make_subplots
from src.functions.common import _iter_wells
from src.functions.constants import ALL_WELLS


# --- fit-quality outlines -----------------------------------------------------
# Log-space fits report model_rmse as a relative error already; parametric fits
# report it in OD units, so those are divided by the growth amplitude first.
RMSE_THRESHOLD = 0.05
LOG_SPACE_FITS = ("spline", "sliding_window")

FIT_COLOR_PASS = "rgb(76, 175, 80)"
FIT_COLOR_FAIL = "rgb(211, 47, 47)"
FIT_COLOR_NONE = "rgb(204, 204, 204)"
SUBPLOT_BOX_COLOR = "rgb(204, 204, 204)"


def fit_quality_color(gs: dict | None) -> str:
    """Green if the well's fit meets RMSE_THRESHOLD, red if not, grey if unfitted."""
    if not gs or is_no_growth(gs):
        return FIT_COLOR_NONE

    rmse = gs.get("model_rmse")
    if rmse is None or not np.isfinite(rmse):
        return FIT_COLOR_NONE
    rmse = float(rmse)

    if not any(k in (gs.get("fit_method") or "") for k in LOG_SPACE_FITS):
        max_od, n0 = gs.get("max_od"), gs.get("N0")
        if max_od is None or n0 is None:
            return FIT_COLOR_NONE
        if not (np.isfinite(max_od) and np.isfinite(n0)):
            return FIT_COLOR_NONE
        amplitude = float(max_od) - float(n0)
        if amplitude <= 0:
            return FIT_COLOR_NONE
        rmse /= amplitude

    return FIT_COLOR_PASS if rmse <= RMSE_THRESHOLD else FIT_COLOR_FAIL


def _axis_suffix(i: int) -> str:
    return "" if i == 1 else str(i)


def _retarget(ref: str, suffix: str) -> str:
    """Point an "x"/"y domain" style ref at subplot `suffix`."""
    if not suffix or not ref or ref[0] not in "xy":
        return ref
    return f"{ref[0]}{suffix}{ref[1:]}"


def _base_figure(t, y):
    """Raw-data trace only, for annotate_plot to fill in.

    Mirrors gc_plot.create_base_plot without the layout it sets, which is
    discarded here and costs more than the rest of the plot combined.
    """
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    return go.Figure(
        data=[
            go.Scatter(
                x=t[mask],
                y=y[mask],
                mode="markers",
                name="Data",
                marker=dict(size=5, opacity=0.3, color="gray"),
                showlegend=False,
            )
        ]
    )


def _subplot_bounds(fig, i: int):
    """Paper-coordinate (x0, x1, y0, y1) of subplot i."""
    suffix = _axis_suffix(i)
    x0, x1 = fig.layout[f"xaxis{suffix}"].domain
    y0, y1 = fig.layout[f"yaxis{suffix}"].domain
    return x0, x1, y0, y1


def _subplot_box(fig, i: int, color: str, width: int = 2) -> dict:
    """Rectangle outlining subplot i.

    Paper coordinates: a subplot with no traces drops domain-referenced shapes.
    """
    x0, x1, y0, y1 = _subplot_bounds(fig, i)
    return dict(
        type="rect",
        xref="paper",
        yref="paper",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line=dict(color=color, width=width),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )


def _well_box(fig, i: int, well: str, color: str, muted: bool = False):
    """Return the (shape, annotation) dicts outlining and labelling subplot i."""
    x0, x1, y0, y1 = _subplot_bounds(fig, i)
    box = _subplot_box(fig, i, color)
    label = dict(
        text=well,
        xref="paper",
        yref="paper",
        x=x0 + 0.05 * (x1 - x0),
        y=y1 - 0.05 * (y1 - y0),
        showarrow=False,
        font=dict(size=9, color="lightgray" if muted else "black"),
        xanchor="left",
        yanchor="top",
    )
    return box, label


def convert_hours_to_unit(hours: float | np.ndarray, time_unit: str = "hours"):
    """Convert time from hours to the specified display unit.

    Args:
        hours: Time value(s) in hours
        time_unit: Target unit ("seconds", "minutes", or "hours")

    Returns:
        Time value(s) in the target unit
    """
    if time_unit == "seconds":
        return hours * 3600.0
    elif time_unit == "minutes":
        return hours * 60.0
    else:  # hours
        return hours


# --- helpers ------------------------------------------------------------------
def _finite_sorted_xy(time_s, y_s):
    """Return finite x/y arrays sorted by x."""
    t = np.asarray(time_s, float)
    y = np.asarray(y_s, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size:
        o = np.argsort(t)
        t, y = t[o], y[o]
    return t, y


# --- blanks ----------------------------------------------------------------
def plot_baseline(baseline, name_by_well: dict | None = None, time_unit: str = "hours"):
    """Plot blank wells and mean baseline over time.

    Args:
        baseline: DataFrame with Time index and wells as columns (plus 'Mean' column)
        name_by_well: Optional dict mapping well IDs to sample names for color coding
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    fig = go.Figure()

    # Build color map based on sample names if available
    name_by_well = name_by_well or {}
    palette = px.colors.qualitative.Plotly

    # Get unique sample names from the wells in baseline (excluding 'Mean')
    well_cols = [c for c in baseline.columns if c != "Mean"]
    sample_names = list(dict.fromkeys([name_by_well.get(w, w) for w in well_cols]))
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(sample_names)}

    # Convert time index to display unit
    time_display = convert_hours_to_unit(baseline.index.to_numpy(), time_unit)

    for col in baseline.columns:
        sample_name = name_by_well.get(col, col) if col != "Mean" else "Mean"
        color = color_map.get(sample_name, "black")

        fig.add_scatter(
            x=time_display,
            y=baseline[col],
            mode="markers" if col != "Mean" else "lines+markers",
            name=sample_name if col != "Mean" else "Mean",
            marker=dict(color=color) if col != "Mean" else None,
            line=dict(color=color) if col == "Mean" else None,
        )
    fig.update_xaxes(title=f"Time ({time_unit})")
    fig.update_yaxes(showgrid=False)

    return fig


def _group_sort_key(name: str) -> tuple[int, str]:
    """Sort group labels like Group 1, Group 2 numerically."""
    match = re.search(r"(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return 10**9, str(name)


def plot_baseline_by_group(
    baseline: pd.DataFrame,
    *,
    blank_group_map: dict[str, str] | None = None,
    time_unit: str = "hours",
):
    """Plot grouped blank baselines: per-group mean lines + per-well scatter.

    Args:
        baseline: DataFrame indexed by Time. Contains blank well columns and optional
            per-group mean columns named "{Group Name} Mean".
        blank_group_map: Optional mapping of blank well ID -> group name.
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    fig = go.Figure()
    fig.update_xaxes(title=f"Time ({time_unit})")
    fig.update_yaxes(showgrid=False)

    if baseline is None or baseline.empty:
        return fig

    blank_group_map = {
        str(well).strip().upper(): str(group).strip()
        for well, group in (blank_group_map or {}).items()
        if str(well).strip() and str(group).strip()
    }

    group_mean_cols = [
        c for c in baseline.columns if str(c).endswith(" Mean") and str(c) != "Mean"
    ]
    mean_col_to_group = {c: str(c)[: -len(" Mean")].strip() for c in group_mean_cols}
    group_names = sorted(set(mean_col_to_group.values()), key=_group_sort_key)

    excluded_cols = {"Mean"} | set(group_mean_cols)
    well_cols = [c for c in baseline.columns if c not in excluded_cols]

    # Fallback for legacy analyses with no stored per-group means.
    if not group_names:
        inferred_groups = sorted(
            {blank_group_map.get(str(w).upper(), "Group 1") for w in well_cols},
            key=_group_sort_key,
        )
        if inferred_groups:
            group_names = inferred_groups
            for group in inferred_groups:
                group_wells = [
                    w
                    for w in well_cols
                    if blank_group_map.get(str(w).upper(), "Group 1") == group
                ]
                if group_wells:
                    mean_col_to_group[group] = group

    palette = px.colors.qualitative.Plotly
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
    time_display = convert_hours_to_unit(baseline.index.to_numpy(), time_unit)

    for group in group_names:
        mean_col = None
        for col, col_group in mean_col_to_group.items():
            if col_group == group and col in baseline.columns:
                mean_col = col
                break

        if mean_col is not None:
            group_mean = baseline[mean_col]
        else:
            group_wells = [
                w
                for w in well_cols
                if blank_group_map.get(str(w).upper(), "Group 1") == group
            ]
            if not group_wells:
                continue
            group_mean = baseline[group_wells].mean(axis=1)

        color = group_colors[group]
        fig.add_scatter(
            x=time_display,
            y=group_mean,
            mode="lines",
            name=f"{group} Mean",
            legendgroup=group,
            line=dict(color=color, width=3),
        )

        group_wells = [
            w
            for w in well_cols
            if blank_group_map.get(str(w).upper(), "Group 1") == group
        ]
        for well in group_wells:
            fig.add_scatter(
                x=time_display,
                y=baseline[well],
                mode="markers",
                name=well,
                legendgroup=group,
                marker=dict(color=color, size=7),
                showlegend=False,
            )

    return fig


# --- replicates ----------------------------------------------------------------
def _order_and_colors(d: pd.DataFrame, sample_order: list[str] | None):
    """Return ordered sample names and a deterministic color map."""
    ordered = list(dict.fromkeys([n for n in (sample_order or []) if n]))
    seen = list(pd.unique(d["Sample Name"]))
    names = ordered + [n for n in seen if n not in ordered]
    palette = px.colors.qualitative.Plotly
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    return names, color_map


def plot_replicates_scatter(
    curves_df: pd.DataFrame,
    sample_order: list[str] | None = None,
    t_start=0.0,
    t_end=72.0,
    time_unit: str = "hours",
):
    """Scatter plot of replicate curves for selected samples and time window.

    Args:
        curves_df: DataFrame with Time, baseline_corrected, Sample Name columns
        sample_order: Optional list of sample names for ordering
        t_start: Start time for filtering (in hours)
        t_end: End time for filtering (in hours)
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    time_label = f"Time ({time_unit})"
    fig = go.Figure()
    fig.update_layout(
        xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)", height=600
    )
    if curves_df is None or curves_df.empty:
        return fig

    d = curves_df[(curves_df["Time"] >= t_start) & (curves_df["Time"] <= t_end)].copy()
    # Convert time to display unit
    d["Time_display"] = convert_hours_to_unit(d["Time"].to_numpy(), time_unit)
    names, color_map = _order_and_colors(d, sample_order)

    return px.scatter(
        d,
        x="Time_display",
        y="baseline_corrected",
        color="Sample Name",
        hover_data=["plate", "well", "key"],
        category_orders={"Sample Name": names},
        color_discrete_map=color_map,
    ).update_layout(
        height=600, xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)"
    )


def plot_mean_growth(
    curves_df: pd.DataFrame,
    sample_order: list[str] | None,
    t_start=0.0,
    t_end=72.0,
    time_unit: str = "hours",
):
    """Plot mean curve with +/-1 SD shading for each sample.

    Args:
        curves_df: DataFrame with Time, baseline_corrected, Sample Name columns
        sample_order: Optional list of sample names for ordering
        t_start: Start time for filtering (in hours)
        t_end: End time for filtering (in hours)
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    time_label = f"Time ({time_unit})"
    fig = go.Figure()
    fig.update_layout(
        xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)", height=600
    )
    if curves_df is None or curves_df.empty:
        return fig

    d = curves_df[(curves_df["Time"] >= t_start) & (curves_df["Time"] <= t_end)].copy()
    names, color_map = _order_and_colors(d, sample_order)

    agg = (
        d.groupby(["Sample Name", "Time"], as_index=False)["baseline_corrected"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
    )
    agg["upper"] = agg["mean"] + agg["sd"]
    agg["lower"] = agg["mean"] - agg["sd"]
    # Convert time to display unit
    agg["Time_display"] = convert_hours_to_unit(agg["Time"].to_numpy(), time_unit)

    for nm in names:
        sub = agg[agg["Sample Name"] == nm].sort_values("Time")
        if sub.empty:
            continue
        c = color_map[nm]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["Time_display"], sub["Time_display"][::-1]]),
                y=pd.concat([sub["upper"], sub["lower"][::-1]]),
                fill="toself",
                fillcolor=c,
                line=dict(width=0),
                opacity=0.2,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["Time_display"],
                y=sub["mean"],
                mode="lines",
                name=nm,
                line=dict(color=c),
                text=[nm] * len(sub),
                hovertemplate=f"Sample=%{{text}}<br>Time=%{{x:.2f}} {time_unit}<br>Mean=%{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(legend_traceorder="normal")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_replicates_by_sample(plates: dict, time_unit: str = "hours"):
    """Create a grid of replicate scatter plots grouped by sample.

    Args:
        plates: Dictionary of plate data
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    items = [(pid, well, nm, d) for pid, _, well, nm, d, _ in _iter_wells(plates)]
    names = sorted(
        {
            nm
            for nm in ({(nm or "").strip() for *_, nm, __ in items} - {"", "False"})
            if not nm.upper().startswith("BLANK")
        }
    )

    # Return empty figure if no valid samples found
    if not names:
        fig = go.Figure()
        fig.add_annotation(
            text="No analyzed data available. Please analyze your data first.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    cols = max(1, int(np.sqrt(max(1, len(names)))))
    rows = (len(names) + cols - 1) // cols
    pos = {n: divmod(i, cols) for i, n in enumerate(names)}

    time_label = f"Time ({time_unit})"
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=names,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
        x_title=time_label,
        y_title="OD600 (baseline-corrected)",
    )

    pal = px.colors.qualitative.Plotly
    keys = sorted({f"{pid}_{well}" for pid, well, *_ in items})
    cmap = {k: pal[i % len(pal)] for i, k in enumerate(keys)}

    # Batched: add_trace re-validates the whole figure on every call.
    traces = []
    tmins, tmaxs, ymins, ymaxs = [], [], [], []
    for pid, well, nm, d in items:
        nm = (nm or "").strip()
        if nm not in pos or d is None or d.empty:
            continue
        r, c = pos[nm]
        key = f"{pid}_{well}"

        # Convert time to display unit
        time_display = convert_hours_to_unit(d["Time"].to_numpy(), time_unit)

        suffix = _axis_suffix(r * cols + c + 1)
        traces.append(
            dict(
                type="scatter",
                x=time_display,
                y=d["baseline_corrected"].to_numpy(),
                mode="markers",
                marker=dict(size=3, color=cmap[key]),
                hovertemplate=f"Sample: {nm}<br>Well: {well}<br>Time: %{{x:.2f}} {time_unit}<br>OD: %{{y:.4f}}<extra></extra><br>Plate: {pid}",
                showlegend=False,
                xaxis=f"x{suffix}",
                yaxis=f"y{suffix}",
            )
        )
        tmins.append(float(time_display.min()))
        tmaxs.append(float(time_display.max()))
        ymins.append(float(d["baseline_corrected"].min()))
        ymaxs.append(float(d["baseline_corrected"].max()))

    if traces:
        fig.add_traces(traces)
    fig.update_layout(
        height=1050,
        shapes=[
            _subplot_box(fig, i, SUBPLOT_BOX_COLOR, width=1)
            for i in range(1, len(names) + 1)
        ],
    )
    if tmins:
        fig.update_xaxes(showgrid=False, range=[min(tmins), max(tmaxs)])
    else:
        fig.update_xaxes(showgrid=False)
    if ymins:
        fig.update_yaxes(showgrid=False, range=[min(ymins), max(ymaxs)])
    else:
        fig.update_yaxes(showgrid=False)
    return fig


# --- growth stats ----------------------------------------------------------------
# Mapping of metric names to their units for y-axis labels.
METRIC_UNITS = {
    "mu_max": "h⁻¹",
    "specific_growth_rate": "h⁻¹",
    "intrinsic_growth_rate": "h⁻¹",
    "doubling_time": "hours",
    "max_od": "OD600",
    "exp_phase_start": "hours",
    "exp_phase_end": "hours",
    "time_at_umax": "hours",
    "od_at_umax": "OD600",
}

# Mapping of metric names to display titles
METRIC_TITLES = {
    "mu_max": "Maximum specific growth rate",
    "specific_growth_rate": "Maximum specific growth rate",
    "intrinsic_growth_rate": "Intrinsic growth rate",
    "doubling_time": "Doubling time",
    "max_od": "Maximum OD",
    "exp_phase_start": "Lag phase end",
    "exp_phase_end": "Exponential phase end",
    "time_at_umax": "Time at max growth rate",
    "od_at_umax": "OD at max growth rate",
}

# Mapping of metric names to y-axis labels (using Greek letters where appropriate)
METRIC_Y_LABELS = {
    "mu_max": "μ",
    "specific_growth_rate": "μ",
    "intrinsic_growth_rate": "μᵢ",
    "doubling_time": "tᵈ",
}


def plot_single_growth_stat(
    long_df: pd.DataFrame,
    *,
    x_col: str = "Sample Name",
    legend_col: str | None = None,
    x_order: list[str] | None = None,
    legend_order: list[str] | None = None,
):
    """Plot a single growth metric across samples with optional strain/condition splits."""

    if long_df is None or long_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Growth statistics", height=400)
        return fig

    df = long_df.copy()

    # Get the metric name from the data
    metric = (
        df["metric"].iloc[0] if "metric" in df.columns and not df.empty else "Metric"
    )

    # Get display title (defaults to metric name if not specified)
    display_title = METRIC_TITLES.get(metric, metric)

    # Get the y-axis label (use Greek letter if available, otherwise metric name)
    y_label = METRIC_Y_LABELS.get(metric, metric)
    unit = METRIC_UNITS.get(metric, "")
    y_axis_label = f"{y_label} ({unit})" if unit else y_label

    # ---- optionally derive Strain/Condition from sample_name (split on FIRST underscore) ----
    s = df["sample_name"].astype(str)
    has_split = s.str.contains("_", regex=False).any()
    if has_split:
        sc = s.str.split("_", n=1, expand=True)
        df["Strain"] = sc[0]
        df["Condition"] = sc[1].fillna("")

    # ---- map UI names to actual df columns ----
    col_map = {
        "Sample Name": "sample_name",
        "Strain": "Strain",
        "Condition": "Condition",
    }

    if x_col not in col_map:
        x_col = "Sample Name"
    x_df_col = col_map[x_col]

    legend_df_col = None
    if legend_col is not None:
        if legend_col not in col_map:
            legend_col = None
        else:
            legend_df_col = col_map[legend_col]
            # If user chose Strain/Condition but we didn't create them, drop back safely
            if legend_df_col not in df.columns:
                legend_col = None
                legend_df_col = None

    # If x axis requested Strain/Condition but absent, fall back to Sample Name
    if x_df_col not in df.columns:
        x_col = "Sample Name"
        x_df_col = "sample_name"

    # ---- x order categorical for stable ordering ----
    if not x_order:
        x_order = list(pd.unique(df[x_df_col].astype(str)))

    df["_x"] = (
        df[x_df_col]
        .astype(str)
        .astype(pd.CategoricalDtype(categories=list(x_order), ordered=True))
    )

    # ---- legend groups + stable color mapping ----
    if legend_col is None:
        groups: list[str] = []
        color_map: dict[str, str] = {}
    else:
        assert legend_df_col is not None
        if not legend_order:
            legend_order = list(pd.unique(df[legend_df_col].astype(str)))

        present = set(df[legend_df_col].astype(str))

        # enforce requested legend order
        groups = [str(g) for g in legend_order if str(g) in present]

        # include any stragglers at end
        for g in pd.unique(df[legend_df_col].astype(str)):
            gs = str(g)
            if gs not in groups:
                groups.append(gs)

        base_colors = px.colors.qualitative.Plotly
        color_map = {g: base_colors[i % len(base_colors)] for i, g in enumerate(groups)}

    # ---- aggregate ----
    group_cols = ["_x"]
    if legend_col is not None:
        group_cols.insert(0, legend_df_col)  # type: ignore[arg-type]

    agg = (
        df.groupby(group_cols, as_index=False)["value"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
        .sort_values(group_cols, kind="stable")
    )

    # Create a single plot (not subplots)
    fig = go.Figure()

    if legend_col is None:
        # Single series (no legend grouping)
        fig.add_trace(
            go.Bar(
                x=agg["_x"],
                y=agg["mean"],
                error_y=dict(type="data", array=agg["sd"], visible=True),
                name="",
                showlegend=False,
                hovertemplate=f"{x_col}=%{{x}}<br>Mean=%{{y:.4f}}<extra></extra>",
                marker=dict(line=dict(color="black", width=1.5)),
            )
        )

        fig.add_trace(
            go.Box(
                x=df["_x"],
                y=df["value"].to_numpy(float),
                name="",
                showlegend=False,
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                fillcolor="rgba(0,0,0,0)",
                line=dict(width=0),
                marker=dict(size=6, opacity=0.8),
                text=(df["plate"].astype(str) + " " + df["well"].astype(str)).tolist(),
                hovertemplate="Well=%{text}<br>Value=%{y:.4f}<extra></extra>",
            )
        )
    else:
        # One legend entry per group, in user-defined order, with stable colors
        assert legend_df_col is not None

        for g in groups:
            a_g = agg[agg[legend_df_col].astype(str) == g]
            p_g = df[df[legend_df_col].astype(str) == g]

            fig.add_trace(
                go.Bar(
                    x=a_g["_x"],
                    y=a_g["mean"],
                    error_y=dict(type="data", array=a_g["sd"], visible=True),
                    name=g,
                    legendgroup=g,
                    offsetgroup=g,
                    showlegend=True,
                    hovertemplate=(
                        f"{legend_col}=%{{fullData.name}}<br>"
                        f"{x_col}=%{{x}}<br>"
                        "Mean=%{y:.4f}<extra></extra>"
                    ),
                    marker=dict(
                        color=color_map[g],
                        line=dict(color="black", width=1.5),
                    ),
                )
            )

            fig.add_trace(
                go.Box(
                    x=p_g["_x"],
                    y=p_g["value"].to_numpy(float),
                    name=g,
                    legendgroup=g,
                    offsetgroup=g,
                    showlegend=False,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(width=0),
                    marker=dict(
                        color="black",
                        size=6,
                        opacity=0.8,
                    ),
                    text=(
                        p_g["plate"].astype(str) + " " + p_g["well"].astype(str)
                    ).tolist(),
                    hovertemplate="Well=%{text}<br>Value=%{y:.4f}<extra></extra>",
                )
            )

    fig.update_xaxes(
        showgrid=False,
        type="category",
        categoryorder="array",
        categoryarray=list(x_order),
        title_text=x_col,
    )
    fig.update_yaxes(showgrid=False, range=[0, None], title_text=y_axis_label)

    fig.update_layout(
        title=display_title,
        height=500,
        margin=dict(t=60, b=60),
        barmode="group",
        boxmode="group",
        legend_title_text=(legend_col if legend_col else ""),
        showlegend=bool(legend_col),
    )
    return fig


# --- window fits ----------------------------------------------------------------
def plot_window_plate(plate: dict, time_unit: str = "hours"):
    """Plot a full 96-well plate overview with window-fit overlays.

    Args:
        plate: Plate dictionary containing processed_data, growth_stats, and fit_parameters
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    proc = plate.get("processed_data") or {}
    gs_all = plate.get("growth_stats") or {}
    fit_params = plate.get("fit_parameters") or {}

    fig = make_subplots(
        rows=8,
        cols=12,
        horizontal_spacing=0.008,
        vertical_spacing=0.025,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Batched: adding one at a time re-validates the whole figure each call,
    # which is quadratic over 96 subplots.
    traces, shapes, labels = [], [], []

    def _finish():
        if traces:
            fig.add_traces(traces)
        fig.update_layout(
            shapes=shapes,
            annotations=labels,
            height=900,
            showlegend=False,
            hovermode=False,
        )
        return fig

    def _outline_empty_grid():
        for idx, w in enumerate(ALL_WELLS, 1):
            box, label = _well_box(fig, idx, w, FIT_COLOR_NONE, muted=True)
            shapes.append(box)
            labels.append(label)

    # Check if there's any data
    if not proc:
        _outline_empty_grid()
        fig.update_layout(margin=dict(l=60, r=25, t=60, b=55))
        return _finish()

    # Calculate global ranges
    ts, ys = [], []
    for d in proc.values():
        if d is None or d.empty:
            continue
        ts.append(d["Time"])
        ys.append(d["baseline_corrected"])

    if not ts:
        _outline_empty_grid()
        fig.update_layout(margin=dict(l=60, r=25, t=60, b=55))
        return _finish()

    x_min, x_max = float(min(t.min() for t in ts)), float(max(t.max() for t in ts))
    x_min_display = convert_hours_to_unit(x_min, time_unit)
    x_max_display = convert_hours_to_unit(x_max, time_unit)
    y_min, y_max = float(min(y.min() for y in ys)), float(max(y.max() for y in ys))
    xr, yr = x_max_display - x_min_display, y_max - y_min
    x_range = [x_min_display - 0.02 * xr, x_max_display + 0.02 * xr]
    y_range = [y_min - 0.05 * yr, y_max + 0.05 * yr]

    for i, well in enumerate(ALL_WELLS, 1):
        d = proc.get(well)
        gs = gs_all.get(well) or {}
        fit_result = fit_params.get(well)

        # Box first so empty wells are outlined too.
        empty = d is None or d.empty
        box, label = _well_box(fig, i, well, fit_quality_color(gs), muted=empty)
        shapes.append(box)
        labels.append(label)

        if empty:
            continue

        # Get time and data arrays
        t, y = _finite_sorted_xy(
            d["Time"].to_numpy(),
            d["baseline_corrected"].to_numpy(),
        )

        if t.size == 0:
            continue

        # Convert time to display unit
        t_display = convert_hours_to_unit(t, time_unit)

        # Annotate on a small per-well figure, then retarget onto this subplot;
        # annotate_plot mutates whatever figure it is given.
        temp_fig = _base_figure(t_display, y)

        # Prepare annotation parameters from growth stats
        stats_converted = None
        fitted_model = None

        if not is_no_growth(gs):
            # Create a copy of growth stats with time values converted to display unit
            stats_converted = {}

            # Convert time-based stats
            for key in ["exp_phase_start", "exp_phase_end", "time_at_umax", "lag_time"]:
                if key in gs and gs[key] is not None and np.isfinite(gs[key]):
                    stats_converted[key] = convert_hours_to_unit(
                        float(gs[key]), time_unit
                    )

            # Copy OD-based stats without conversion
            for key in ["od_at_umax", "max_od", "N0", "mu_max"]:
                if key in gs and gs[key] is not None:
                    stats_converted[key] = gs[key]

            # Use the stored fit result directly if available
            if fit_result is not None:
                # Convert time values in fit_result params to display unit
                fitted_model = fit_result.copy()
                if "params" in fitted_model:
                    params_copy = fitted_model["params"].copy()
                    # Convert fit_t_min and fit_t_max to display unit
                    if "fit_t_min" in params_copy:
                        params_copy["fit_t_min"] = convert_hours_to_unit(
                            float(params_copy["fit_t_min"]), time_unit
                        )
                    if "fit_t_max" in params_copy:
                        params_copy["fit_t_max"] = convert_hours_to_unit(
                            float(params_copy["fit_t_max"]), time_unit
                        )
                    fitted_model["params"] = params_copy

        temp_fig = gc_plot.annotate_plot(
            fig=temp_fig,
            fit_result=fitted_model,
            stats=stats_converted,
            show_fitted_curve=True,
            show_phase_boundaries=True,
            show_crosshairs=False,
            show_od_max_line=False,
            show_n0_line=False,
            show_umax_marker=False,
            show_tangent=False,
            scale="linear",
            fitted_curve_width=3,
        )

        suffix = _axis_suffix(i)
        for trace in temp_fig.data:
            d_trace = trace.to_plotly_json()
            d_trace["showlegend"] = False
            d_trace["xaxis"] = f"x{suffix}"
            d_trace["yaxis"] = f"y{suffix}"
            traces.append(d_trace)
        for shape in temp_fig.layout.shapes:
            d_shape = shape.to_plotly_json()
            d_shape["xref"] = _retarget(d_shape.get("xref", "x"), suffix)
            d_shape["yref"] = _retarget(d_shape.get("yref", "y"), suffix)
            shapes.append(d_shape)

    fig.update_layout(margin=dict(l=60, r=25, t=20, b=55))
    fig.update_xaxes(showgrid=False, range=x_range, matches="x")
    fig.update_yaxes(showgrid=False, range=y_range, matches="y")

    return _finish()


# --- derivative models ---------------------------------------------------------
def d1_model(t, A, r, t0):
    """Idealized first-derivative model for growth curves."""
    u = np.exp(-r * (t - t0))
    return A * (u / (1 + u) ** 2)


def d2_model(t, A, r, t0):
    """Idealized second-derivative model for growth curves."""
    u = np.exp(-r * (t - t0))
    return A * r * (u * (u - 1) / (1 + u) ** 3)


def plot_derivative_metric(
    plate: dict,
    well: str,
    metric: str,
    sg_window=11,
    sg_poly=2,
    time_unit: str = "hours",
    gs: dict | None = None,
):
    """Plot either dN/dt or μ (specific growth rate) for a well.

    This function generates three traces:
    1. Raw data metric (light grey)
    2. Smoothed data metric (main trace)
    3. Model fit metric (solid blue line)

    Args:
        plate: Plate dictionary containing processed_data and fit_parameters
        well: Well identifier
        metric: Either "dndt" for dN/dt or "mu" for μ
        sg_window: Savitzky-Golay window size
        sg_poly: Savitzky-Golay polynomial order
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
        gs: Growth statistics dictionary (optional). If provided with _used_fit_times,
            only the lasso-selected data points will be used for calculation.
    """
    # Validate metric
    if metric not in ["dndt", "mu"]:
        raise ValueError(f"metric must be 'dndt' or 'mu', got '{metric}'")

    # Get processed data
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t_full = d["Time"].to_numpy(float)
    y_full = d["baseline_corrected"].to_numpy(float)

    # Store full time range for x-axis before any filtering
    t_full_display = convert_hours_to_unit(t_full, time_unit)
    x_range = [float(t_full_display.min()), float(t_full_display.max())]

    t_raw = t_full.copy()
    y_raw = y_full.copy()

    # Filter to lasso-selected points if available
    gs = gs or {}
    used_times = gs.get("_used_fit_times")
    if used_times is not None and len(used_times) > 0:
        used_times_arr = np.asarray(used_times)
        time_tolerance = 0.01
        used_mask = np.zeros(len(t_raw), dtype=bool)
        for ut in used_times_arr:
            used_mask |= np.abs(t_raw - ut) < time_tolerance
        t_raw = t_raw[used_mask]
        y_raw = y_raw[used_mask]

    if len(t_raw) < 3:
        return go.Figure()

    # Step 1: Calculate metric on raw data
    if metric == "dndt":
        t_metric_raw, metric_raw = compute_first_derivative(t_raw, y_raw)
        metric_label = "dN/dt"
        y_axis_title = "dN/dt"
        plot_title = f"dN/dt – {well}"
    else:  # mu
        t_metric_raw, metric_raw = compute_instantaneous_mu(t_raw, y_raw)
        metric_label = "μ"
        y_axis_title = "μ (h⁻¹)"
        plot_title = f"Specific growth rate – {well}"

    # Step 2: Smooth the data
    y_smooth = smooth(y_raw, sg_window, sg_poly)

    # Step 3: Calculate metric on smoothed data
    if metric == "dndt":
        t_metric_smooth, metric_smooth = compute_first_derivative(t_raw, y_smooth)
    else:  # mu
        t_metric_smooth, metric_smooth = compute_instantaneous_mu(t_raw, y_smooth)

    # Convert time to display unit
    t_display = convert_hours_to_unit(t_metric_smooth, time_unit)

    # Create figure
    fig = go.Figure()

    # Plot raw metric (light grey)
    t_raw_display = convert_hours_to_unit(t_metric_raw, time_unit)
    fig.add_trace(
        go.Scatter(
            x=t_raw_display,
            y=metric_raw,
            mode="lines",
            line=dict(width=5, color="lightgrey"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>{metric_label} (raw)=%{{y:.4f}}<extra></extra>",
            showlegend=False,
            name="Raw",
        )
    )

    # Plot smoothed metric (lighter red)
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=metric_smooth,
            mode="lines",
            line=dict(width=5, color="#FF6692"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>{metric_label} (smoothed)=%{{y:.4f}}<extra></extra>",
            showlegend=False,
            name="Smoothed",
        )
    )

    # Step 4 & 5: Generate model metric and plot
    fit_parameters = (plate.get("fit_parameters") or {}).get(well)
    if fit_parameters is not None:
        model_type = fit_parameters.get("model_type", "")
        params = fit_parameters.get("params", {})
        metric_model = None
        t_model = None

        # Get the fitted data range
        fit_t_min = params.get("fit_t_min")
        fit_t_max = params.get("fit_t_max")

        # Filter to fitted range if available
        if fit_t_min is not None and fit_t_max is not None:
            fit_mask = (t_raw >= fit_t_min) & (t_raw <= fit_t_max)
            t_model = t_raw[fit_mask]
            y_model_raw = y_raw[fit_mask]
            y_model_smooth = y_smooth[fit_mask]
        else:
            # Use full range if fit bounds not available
            t_model = t_raw
            y_model_raw = y_raw
            y_model_smooth = y_smooth

        if len(t_model) < 2:
            # Not enough points in fitted range
            t_model = None
        else:
            if model_type == "sliding_window":
                # For sliding window, calculate from raw data (as growthcurves does)
                window_points = params.get("window_points", 15)
                if metric == "dndt":
                    # For dN/dt, we need to smooth first then compute derivative
                    _, metric_model = compute_first_derivative(t_model, y_model_smooth)
                else:  # mu
                    # For μ, use sliding window on raw data
                    _, metric_model = compute_sliding_window_growth_rate(
                        t_model, y_model_raw, window_points=window_points
                    )

            elif model_type in (
                MODEL_REGISTRY["mechanistic"] + MODEL_REGISTRY["phenomenological"]
            ):
                # For parametric models, compute metric from the model
                model_func = {
                    "mech_logistic": mech_logistic_model,
                    "mech_gompertz": mech_gompertz_model,
                    "mech_richards": mech_richards_model,
                    "mech_baranyi": mech_baranyi_model,
                    "phenom_logistic": phenom_logistic_model,
                    "phenom_gompertz": phenom_gompertz_model,
                    "phenom_gompertz_modified": phenom_gompertz_modified_model,
                    "phenom_richards": phenom_richards_model,
                }.get(model_type)

                if model_func is not None:
                    # Handle parameter name mismatches and filter metadata
                    model_params = params.copy()
                    if "mu_max_param" in model_params:
                        model_params["mu_max"] = model_params.pop("mu_max_param")
                    model_params.pop("fit_t_min", None)
                    model_params.pop("fit_t_max", None)

                    # Evaluate the model on fitted range
                    y_model = model_func(t_model, **model_params)

                    # Compute metric from model
                    if metric == "dndt":
                        _, metric_model = compute_first_derivative(t_model, y_model)
                    else:  # mu
                        _, metric_model = compute_instantaneous_mu(t_model, y_model)

            elif model_type == "spline":
                # For spline model, reconstruct the spline and evaluate
                try:
                    spline = spline_from_params(params)

                    if metric == "dndt":
                        # Spline is fitted to log(y), so exp(spline(t)) gives y
                        y_log_model = spline(t_model)
                        y_model = np.exp(y_log_model)
                        _, metric_model = compute_first_derivative(t_model, y_model)
                    else:  # mu
                        # μ = d(ln(y))/dt, which is the derivative of the spline
                        metric_model = spline.derivative()(t_model)
                except Exception:
                    # If spline reconstruction fails, skip model trace
                    pass

        # Plot model metric if available
        if (
            metric_model is not None
            and t_model is not None
            and np.isfinite(metric_model).any()
        ):
            t_model_display = convert_hours_to_unit(t_model, time_unit)
            fig.add_trace(
                go.Scatter(
                    x=t_model_display,
                    y=metric_model,
                    mode="lines",
                    line=dict(width=5, color="#8dcde0"),
                    hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>{metric_label} (fitted)=%{{y:.4f}}<extra></extra>",
                    showlegend=False,
                    name="Fitted",
                )
            )

    # Add phase boundary annotations
    stats_converted = None
    if gs and not is_no_growth(gs):
        exp_start = gs.get("exp_phase_start")
        exp_end = gs.get("exp_phase_end")
        if exp_start is not None and exp_end is not None:
            stats_converted = {
                "exp_phase_start": convert_hours_to_unit(float(exp_start), time_unit),
                "exp_phase_end": convert_hours_to_unit(float(exp_end), time_unit),
            }

    if stats_converted is not None:
        fig = gc_plot.annotate_plot(
            fig=fig,
            fit_result=None,
            stats=stats_converted,
            show_fitted_curve=False,
            show_phase_boundaries=True,
            show_crosshairs=False,
            show_od_max_line=False,
            show_n0_line=False,
            show_umax_marker=False,
            show_tangent=False,
            scale="linear",
            row=None,
            col=None,
        )

    # Update layout
    fig.update_layout(
        title=plot_title,
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, title=f"Time ({time_unit})", range=x_range)
    fig.update_yaxes(showgrid=False, title=y_axis_title)
    return fig
