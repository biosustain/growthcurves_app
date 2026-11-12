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
    evaluate_parametric_model,
    log_to_linear,
    spline_from_params,
)
from plotly.subplots import make_subplots
from src.functions.common import _iter_wells
from src.functions.constants import DEFAULT_PARAMS, PlateFormat, plate_format

# --- fit-quality highlights ---------------------------------------------------
# Wells are ranked by RMSE relative to their OD increase (model_rmse is in OD
# units); the worst x % get a tinted background on the plate overview.
HIGHLIGHT_FILL = "rgba(239, 83, 80, 0.15)"  # app red (#ef5350), faint

# Panel figures (plate overview, replicates, blanks)
PLATE_ROW_PX = 112.5  # 900 px for a 96-well plate
PLATE_MARGIN = dict(l=30, r=5, t=30, b=5)  # room for the row/column headers
PANEL_MARGIN = dict(l=5, r=5, t=5, b=5)
BLANK_COLS, BLANK_ROW_PX = 4, 260
# Axis padding, as fractions of the data span: (x each side, y below, y above).
# Keeps points inside the rounded cells; panels get headroom for their titles.
PANEL_PAD = (0.04, 0.1, 0.25)
PLATE_PAD = (0.06, 0.1, 0.1)
# App theme (.streamlit/config.toml)
APP_HOVERLABEL = dict(
    bgcolor="white", bordercolor="#66BB6A", font=dict(size=12, color="#262730")
)


def relative_rmse(gs: dict | None) -> float | None:
    """Fit RMSE / OD increase, or None for unfitted or no-growth wells."""
    if not gs or is_no_growth(gs):
        return None
    vals = [gs.get(k) for k in ("model_rmse", "max_od", "N0")]
    if any(v is None or not np.isfinite(v) for v in vals):
        return None
    rmse, max_od, n0 = (float(v) for v in vals)
    amplitude = max_od - n0
    if amplitude <= 0:
        return None
    return rmse / amplitude


def worst_fit_wells(plate: dict, pct: float) -> list[str]:
    """The pct % of ranked wells with the highest relative RMSE, worst first."""
    scores = {
        w: r
        for w, gs in (plate.get("growth_stats") or {}).items()
        if (r := relative_rmse(gs)) is not None
    }
    n = round(pct / 100 * len(scores))
    return sorted(scores, key=scores.get, reverse=True)[:n]


def highlight_wells(fig: go.Figure, plate: dict, wells: list[str]) -> go.Figure:
    """Fill the given wells on a plot_window_plate figure.

    The well cells are the figure's first shapes, in plate order (_plate_frame).
    """
    index = {w: i for i, w in enumerate(plate_format(plate).wells)}
    shapes = list(fig.layout.shapes)
    for w in wells:
        if w in index:
            shapes[index[w]].fillcolor = HIGHLIGHT_FILL
    fig.layout.shapes = shapes
    return fig


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


def _rounded_cell(fig, i: int, height: float, radius: float = 6) -> dict:
    """Rounded white cell behind subplot i, as in the upload plate map.

    radius is in px, for a figure of the given plot height and ~1400 px wide.
    """
    x0, x1, y0, y1 = _subplot_bounds(fig, i)
    rx, ry = radius / 1400, radius / height
    return dict(
        type="path",
        path=(
            f"M{x0 + rx},{y0} L{x1 - rx},{y0} Q{x1},{y0} {x1},{y0 + ry} "
            f"L{x1},{y1 - ry} Q{x1},{y1} {x1 - rx},{y1} L{x0 + rx},{y1} "
            f"Q{x0},{y1} {x0},{y1 - ry} L{x0},{y0 + ry} Q{x0},{y0} {x0 + rx},{y0} Z"
        ),
        xref="paper",
        yref="paper",
        line=dict(color="rgba(0, 0, 0, 0.18)", width=1),
        fillcolor="white",
        layer="below",
    )


def _plate_frame(fig, fmt: PlateFormat):
    """Rounded well cells and row/column headers, styled like the upload plate map."""
    height = PLATE_ROW_PX * len(fmt.rows) - PLATE_MARGIN["t"] - PLATE_MARGIN["b"]
    cells = [_rounded_cell(fig, i, height) for i in range(1, len(fmt.wells) + 1)]
    headers = []
    header = dict(
        xref="paper", yref="paper", showarrow=False, font=dict(size=13, color=INFO_INK)
    )
    n_cols = len(fmt.cols)
    for c, col in enumerate(fmt.cols, 1):  # numbers above the top row
        x0, x1, _, y1 = _subplot_bounds(fig, c)
        top = dict(x=(x0 + x1) / 2, y=y1, yanchor="bottom", yshift=4)
        headers.append(header | top | dict(text=f"<b>{col}</b>"))
    for r, row in enumerate(fmt.rows):  # letters left of the first column
        x0, _, y0, y1 = _subplot_bounds(fig, r * n_cols + 1)
        left = dict(x=x0, y=(y0 + y1) / 2, xanchor="right", xshift=-6)
        headers.append(header | left | dict(text=f"<b>{row}</b>"))
    return cells, headers


# --- info-plot style ----------------------------------------------------------
# Palette of the explanatory figures (info_plots/make_info_plots.py).
INFO_INK = "#262730"
INFO_INK_2 = "#5F6368"
INFO_AXIS = "#9AA0A6"
INFO_REF = "#6B7280"
INFO_CURVE = "#2B7BB9"
INFO_ACCENT = "#E4572E"
INFO_PHASE = "#66BB6A"
INFO_POINTS = "#A7ADB4"
INFO_FADED = "#D9DCE1"

# Sample colours, led by a soft version of the app's green; fixed order,
# validated for colour-blind separation on white and the grey cards. Some sit
# just under 3:1 contrast, so multi-sample plots rely on their legend.
APP_PALETTE = [
    "#66A86B",  # green (app primary family)
    "#5B8FC7",  # blue
    "#E07B5F",  # coral
    "#8C7BD1",  # lavender
    "#22A6A8",  # teal
    "#D19A3C",  # amber
    "#D57A9C",  # rose
]

# Trace name -> (colour, line width or marker size)
_INFO_TRACES = {
    "Fitted curve": (INFO_CURVE, 3),
    "umax tangent": (INFO_ACCENT, 2),
    "Raw": (INFO_FADED, 3),
    "Smoothed": (INFO_INK_2, 3),
    "Fitted": (INFO_CURVE, 3),
}


def style_like_info_plots(fig: go.Figure, fit_window=None) -> go.Figure:
    """Restyle a single-well plot to match the explanatory figures on the Upload
    page: their palette for the traces, plus style_info_axes.

    As in the sliding-window figure, points used in the fit (lasso-selected and
    inside fit_window, a (t_min, t_max) pair) are dark grey; all others are light grey.
    """
    for tr in fig.data:
        if tr.name == "Data":  # every point; shows grey where not used
            tr.marker.update(color=INFO_POINTS, size=6, opacity=1)
        elif tr.name == "Selected":
            t0, t1 = (
                np.nan if v is None else float(v) for v in fit_window or (None, None)
            )
            x = np.asarray(tr.x, dtype=float)
            used = (x >= t0) & (x <= t1)
            tr.update(x=x[used], y=np.asarray(tr.y, dtype=float)[used])
            tr.marker.update(color=INFO_INK_2, size=7, line=dict(width=0))
        elif tr.name in _INFO_TRACES:
            color, width = _INFO_TRACES[tr.name]
            if tr.mode == "markers":
                tr.marker.color = color
            else:
                tr.line.color = color
                tr.line.width = width
        elif tr.mode == "markers":  # μmax marker
            tr.marker.update(color=INFO_ACCENT, line=dict(color="white", width=2))
    for s in fig.layout.shapes:  # below the traces, so the fit line stays on top
        if s.type == "rect":  # exponential phase
            s.update(fillcolor=INFO_PHASE, opacity=0.13, layer="below")
        else:  # reference lines
            s.update(line_color=INFO_REF, line_dash="dash", opacity=1, layer="below")
    # Draw order: grey points, used points, other lines, fit line, μmax marker
    rank = {"Data": 0, "Selected": 1, "Fitted curve": 3, "Fitted": 3}
    fig.data = sorted(
        fig.data, key=lambda tr: rank.get(tr.name, 2 if tr.mode == "lines" else 4)
    )
    return style_info_axes(fig)


def style_info_axes(fig: go.Figure) -> go.Figure:
    """Transparent background and grey axis lines, as in the explanatory figures.

    Leaves trace colours alone, so per-sample colours are kept.
    """
    axis = dict(
        showline=True,
        linecolor=INFO_AXIS,
        linewidth=1.5,
        showgrid=False,
        zeroline=False,
        ticks="",
        tickfont=dict(color=INFO_INK_2),
        title_font=dict(size=16, color=INFO_INK),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_layout(
        font=dict(family="Arial, sans-serif", color=INFO_INK_2),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(color=INFO_INK))
    return fig


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
def plot_baseline(baseline, name_by_well: dict | None = None):
    """Plot blank wells and mean baseline over time.

    Args:
        baseline: DataFrame with Time index and wells as columns (plus 'Mean' column)
        name_by_well: Optional dict mapping well IDs to sample names for color coding
    """
    fig = go.Figure()

    # Build color map based on sample names if available
    name_by_well = name_by_well or {}
    palette = px.colors.qualitative.Plotly

    # Get unique sample names from the wells in baseline (excluding 'Mean')
    well_cols = [c for c in baseline.columns if c != "Mean"]
    sample_names = list(dict.fromkeys([name_by_well.get(w, w) for w in well_cols]))
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(sample_names)}

    time_display = baseline.index.to_numpy()

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
    fig.update_xaxes(title="Time (hours)")
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
):
    """One cell per blank group: each blank well's points plus the group mean.

    Args:
        baseline: DataFrame indexed by Time. Contains blank well columns and optional
            per-group mean columns named "{Group Name} Mean".
        blank_group_map: Optional mapping of blank well ID -> group name.
    """
    if baseline is None or baseline.empty:
        return go.Figure()

    group_of = {
        str(w).strip().upper(): str(g).strip()
        for w, g in (blank_group_map or {}).items()
        if str(w).strip() and str(g).strip()
    }
    mean_cols = {
        str(c)[: -len(" Mean")].strip(): c
        for c in baseline.columns
        if str(c).endswith(" Mean")
    }
    wells = [c for c in baseline.columns if c != "Mean" and c not in mean_cols.values()]
    by_group: dict[str, list] = {}
    for w in wells:
        by_group.setdefault(group_of.get(str(w).upper(), "Group 1"), []).append(w)
    groups = sorted(set(by_group) | set(mean_cols), key=_group_sort_key)

    # At most BLANK_COLS per row; extra groups wrap onto new rows
    cols = min(len(groups), BLANK_COLS)
    rows = -(-len(groups) // cols)
    height = BLANK_ROW_PX * rows
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=10 / height,  # 10 px, whatever the number of rows
    )
    t = baseline.index.to_numpy()
    traces, titles, lims = [], [], {}
    for i, group in enumerate(groups, 1):
        suffix = _axis_suffix(i)
        axes = dict(xaxis=f"x{suffix}", yaxis=f"y{suffix}")
        members = by_group.get(group, [])
        mean = (
            baseline[mean_cols[group]]
            if group in mean_cols
            else baseline[members].mean(axis=1)
        )
        for k, w in enumerate(members):
            traces.append(
                dict(
                    type="scatter",
                    x=t,
                    y=baseline[w].to_numpy(),
                    mode="markers",
                    marker=dict(size=4, color=APP_PALETTE[k % len(APP_PALETTE)]),
                    hovertemplate=(
                        f"<b>{group}</b><br>Well: {w}<br>Time: %{{x:.2f}} "
                        "hours<br>OD: %{y:.4f}<extra></extra>"
                    ),
                    **axes,
                )
            )
        traces.append(
            dict(
                type="scatter",
                x=t,
                y=mean.to_numpy(),
                mode="lines",
                line=dict(color=INFO_INK_2, width=2),
                hovertemplate=(
                    f"<b>{group} mean</b><br>Time: %{{x:.2f}} hours"
                    f"<br>OD: %{{y:.4f}}<extra></extra>"
                ),
                **axes,
            )
        )
        titles.append(f"<b>{group}</b>")
        for y in [baseline[w] for w in members] + [mean]:
            _extend(lims, suffix, t, y)
    fig.add_traces(traces)
    return _style_panels(fig, titles, height, lims)


def _padded(lo: float, hi: float, below: float, above: float) -> list[float]:
    """[lo, hi] widened by fractions of its span (1 if the span is zero)."""
    span = (hi - lo) or 1.0
    return [lo - below * span, hi + above * span]


def _extend(lims: dict, suffix: str, x, y) -> None:
    """Grow panel `suffix`'s data limits [x0, x1, y0, y1] to cover x, y."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size and y.size:
        x0, x1, y0, y1 = lims.get(suffix, (np.inf, -np.inf, np.inf, -np.inf))
        lims[suffix] = [
            min(x0, float(x.min())),
            max(x1, float(x.max())),
            min(y0, float(y.min())),
            max(y1, float(y.max())),
        ]


def _set_ranges(fig: go.Figure, lims: dict, pad) -> None:
    """Shared, padded ranges over all panels; lims and pad kept in layout.meta
    for unshare_axes."""
    v = np.array(list(lims.values()) or [[0, 1, 0, 1]])
    x0, x1, y0, y1 = v[:, 0].min(), v[:, 1].max(), v[:, 2].min(), v[:, 3].max()
    px_, below, above = pad
    fig.update_layout(meta=dict(pad=list(pad), lims=lims))
    fig.update_xaxes(fixedrange=True, range=_padded(x0, x1, px_, px_))
    fig.update_yaxes(fixedrange=True, range=_padded(y0, y1, below, above))


def unshare_axes(fig: go.Figure) -> go.Figure:
    """Give each panel its own padded ranges, from the limits _set_ranges kept."""
    meta = fig.layout.meta or {}
    px_, below, above = meta.get("pad", PANEL_PAD)
    # Figures without stored limits (e.g. cached before they were added) autoscale
    auto = "lims" not in meta
    fig.update_xaxes(matches=None, autorange=auto or None)
    fig.update_yaxes(matches=None, autorange=auto or None)
    layout = {}
    for suffix, (x0, x1, y0, y1) in (meta.get("lims") or {}).items():
        layout[f"xaxis{suffix}"] = dict(range=_padded(x0, x1, px_, px_))
        layout[f"yaxis{suffix}"] = dict(range=_padded(y0, y1, below, above))
    fig.update_layout(layout)
    return fig


def _style_panels(fig, titles: list[str], height: int, lims: dict) -> go.Figure:
    """Rounded cells titled top left, no axes, hover only (no zoom).

    lims holds each panel's data limits (see _extend), for the axis ranges.
    """
    plot_height = height - PANEL_MARGIN["t"] - PANEL_MARGIN["b"]
    fig.update_layout(
        height=height,
        margin=PANEL_MARGIN,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        dragmode=False,
        showlegend=False,
        hoverlabel=APP_HOVERLABEL,
        shapes=[_rounded_cell(fig, i, plot_height) for i in range(1, len(titles) + 1)],
        annotations=[
            dict(
                text=t,
                xref="paper",
                yref="paper",
                x=_subplot_bounds(fig, i)[0],
                y=_subplot_bounds(fig, i)[3],
                xanchor="left",
                yanchor="top",
                xshift=8,
                yshift=-6,
                showarrow=False,
                font=dict(size=13, color=INFO_INK),
            )
            for i, t in enumerate(titles, 1)
        ],
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    _set_ranges(fig, lims, PANEL_PAD)
    return fig


# --- replicates ----------------------------------------------------------------
def _order_and_colors(d: pd.DataFrame, sample_order: list[str] | None):
    """Return ordered sample names and a deterministic color map."""
    ordered = list(dict.fromkeys([n for n in (sample_order or []) if n]))
    seen = list(pd.unique(d["Sample Name"]))
    names = ordered + [n for n in seen if n not in ordered]
    palette = APP_PALETTE
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    return names, color_map


def plot_replicates_scatter(
    curves_df: pd.DataFrame,
    sample_order: list[str] | None = None,
    t_start=0.0,
    t_end=72.0,
):
    """Scatter plot of replicate curves for selected samples and time window.

    Args:
        curves_df: DataFrame with Time, baseline_corrected, Sample Name columns
        sample_order: Optional list of sample names for ordering
        t_start: Start time for filtering (in hours)
        t_end: End time for filtering (in hours)
    """
    time_label = "Time (hours)"
    fig = go.Figure()
    fig.update_layout(
        xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)", height=600
    )
    if curves_df is None or curves_df.empty:
        return fig

    d = curves_df[(curves_df["Time"] >= t_start) & (curves_df["Time"] <= t_end)]
    names, color_map = _order_and_colors(d, sample_order)

    return px.scatter(
        d,
        x="Time",
        y="baseline_corrected",
        color="Sample Name",
        hover_data=["plate", "well"],
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
):
    """Plot mean curve with +/-1 SD shading for each sample.

    Args:
        curves_df: DataFrame with Time, baseline_corrected, Sample Name columns
        sample_order: Optional list of sample names for ordering
        t_start: Start time for filtering (in hours)
        t_end: End time for filtering (in hours)
    """
    time_label = "Time (hours)"
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

    for nm in names:
        sub = agg[agg["Sample Name"] == nm].sort_values("Time")
        if sub.empty:
            continue
        c = color_map[nm]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["Time"], sub["Time"][::-1]]),
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
                x=sub["Time"],
                y=sub["mean"],
                mode="lines",
                name=nm,
                line=dict(color=c),
                text=[nm] * len(sub),
                hovertemplate="Sample=%{text}<br>Time=%{x:.2f} hours<br>Mean=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(legend_traceorder="normal")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_replicates_by_sample(plates: dict):
    """Create a grid of replicate scatter plots grouped by sample."""
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

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.02,
    )

    # Replicates coloured in turn within each sample
    n_seen = dict.fromkeys(names, 0)

    # Batched: add_trace re-validates the whole figure on every call.
    traces = []
    lims = {}
    for pid, well, nm, d in items:
        nm = (nm or "").strip()
        if nm not in pos or d is None or d.empty:
            continue
        r, c = pos[nm]
        color = APP_PALETTE[n_seen[nm] % len(APP_PALETTE)]
        n_seen[nm] += 1

        time_display = d["Time"].to_numpy()

        suffix = _axis_suffix(r * cols + c + 1)
        traces.append(
            dict(
                type="scatter",
                x=time_display,
                y=d["baseline_corrected"].to_numpy(),
                mode="markers",
                marker=dict(size=3, color=color),
                hovertemplate=(
                    f"<b>{nm}</b><br>Plate: {pid}<br>Well: {well}<br>"
                    "Time: %{x:.2f} hours<br>OD: %{y:.4f}<extra></extra>"
                ),
                showlegend=False,
                xaxis=f"x{suffix}",
                yaxis=f"y{suffix}",
            )
        )
        _extend(lims, suffix, time_display, d["baseline_corrected"])

    if traces:
        fig.add_traces(traces)
    titles = [f"<b>{n}</b>" for n in names]
    return _style_panels(fig, titles, 1050, lims)


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

        base_colors = APP_PALETTE
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
                error_y=dict(
                    type="data", array=agg["sd"], visible=True, color=INFO_INK
                ),
                name="",
                showlegend=False,
                hovertemplate=f"{x_col}=%{{x}}<br>Mean=%{{y:.4f}}<extra></extra>",
                marker=dict(color=APP_PALETTE[0], line=dict(color=INFO_INK, width=1)),
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
                marker=dict(color=INFO_INK, size=6, opacity=0.6),
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
                    error_y=dict(
                        type="data", array=a_g["sd"], visible=True, color=INFO_INK
                    ),
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
                        line=dict(color=INFO_INK, width=1),
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
                        color=INFO_INK,
                        size=6,
                        opacity=0.6,
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
    fig.update_yaxes(showgrid=False, rangemode="nonnegative", title_text=y_axis_label)

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
def _assemble_grid(fig, traces, shapes, labels, fmt: PlateFormat):
    """Apply the batched traces, shapes and labels to the plate figure."""
    if traces:
        fig.add_traces(traces)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        shapes=shapes,
        annotations=labels,
        height=round(PLATE_ROW_PX * len(fmt.rows)),
        margin=PLATE_MARGIN,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode=False,
    )
    return fig


def plot_window_plate(plate: dict):
    """Plot a full plate overview (96 or 384 wells) with window-fit overlays.

    Args:
        plate: Plate dictionary containing processed_data, growth_stats, and fit_parameters
    """
    proc = plate.get("processed_data") or {}
    gs_all = plate.get("growth_stats") or {}
    fit_params = plate.get("fit_parameters") or {}

    fmt = plate_format(plate)
    # Spacing is a fraction of the figure, so scale it to keep gaps constant
    fig = make_subplots(
        rows=len(fmt.rows),
        cols=len(fmt.cols),
        horizontal_spacing=0.04 / len(fmt.cols),
        vertical_spacing=0.06 / len(fmt.rows),
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # Batched: adding one at a time re-validates the whole figure each call,
    # which is quadratic in the number of subplots.
    # Well cells first: highlight_wells relies on their position.
    traces = []
    shapes, labels = _plate_frame(fig, fmt)

    lims = {}  # per-well data limits, for the axis ranges

    for i, well in enumerate(fmt.wells, 1):
        d = proc.get(well)
        gs = gs_all.get(well) or {}
        fit_result = fit_params.get(well)

        if d is None or d.empty:
            continue

        # Get time and data arrays
        t, y = _finite_sorted_xy(
            d["Time"].to_numpy(),
            d["baseline_corrected"].to_numpy(),
        )

        if t.size == 0:
            continue

        _extend(lims, _axis_suffix(i), t, y)

        # Annotate on a small per-well figure, then retarget onto this subplot;
        # annotate_plot mutates whatever figure it is given.
        temp_fig = _base_figure(t, y)

        # Annotation inputs from growth stats (none for no-growth wells)
        stats = None
        fit_window = None
        if is_no_growth(gs):
            fit_result = None
        else:
            stats = {
                k: gs[k]
                for k in (
                    "exp_phase_start",
                    "exp_phase_end",
                    "time_at_umax",
                    "lag_time",
                )
                if gs.get(k) is not None and np.isfinite(gs[k])
            } | {
                k: gs[k]
                for k in ("od_at_umax", "max_od", "N0", "mu_max")
                if gs.get(k) is not None
            }

            # Points used in the fit, drawn darker as on Check Growth Fits
            t0, t1 = gs.get("fit_t_min"), gs.get("fit_t_max")
            if t0 is not None and t1 is not None:
                fit_window = (float(t0), float(t1))
                temp_fig.add_trace(
                    go.Scatter(x=t, y=y, mode="markers", name="Selected")
                )

        temp_fig = gc_plot.annotate_plot(
            fig=temp_fig,
            fit_result=fit_result,
            stats=stats,
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
        # Check Growth Fits palette, scaled down for the small panels
        style_like_info_plots(temp_fig, fit_window=fit_window)
        for trace in temp_fig.data:
            if trace.name == "Data":
                trace.marker.size = 3
            elif trace.name == "Selected":
                trace.marker.size = 3.5
            elif trace.mode == "lines":
                trace.line.width = 2

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

    _set_ranges(fig, lims, PLATE_PAD)

    return _assemble_grid(fig, traces, shapes, labels, fmt)


def plot_derivative_metric(
    plate: dict,
    well: str,
    metric: str,
    sg_window=DEFAULT_PARAMS["sg_window"],
    sg_poly=DEFAULT_PARAMS["sg_poly"],
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

    t_raw = d["Time"].to_numpy(float)
    y_raw = d["baseline_corrected"].to_numpy(float)

    # Full time range for the x-axis, before any filtering
    x_range = [float(t_raw.min()), float(t_raw.max())]

    # Filter to lasso-selected points if available
    gs = gs or {}
    used_times = gs.get("_used_fit_times")
    if used_times is not None and len(used_times) > 0:
        diffs = np.abs(t_raw[:, None] - np.asarray(used_times, dtype=float))
        used_mask = (diffs < 0.01).any(axis=1)
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

    # Create figure
    fig = go.Figure()

    # Plot raw metric (light grey)
    fig.add_trace(
        go.Scatter(
            x=t_metric_raw,
            y=metric_raw,
            mode="lines",
            line=dict(width=5, color="lightgrey"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} hours<br>{metric_label} (raw)=%{{y:.4f}}<extra></extra>",
            showlegend=False,
            name="Raw",
        )
    )

    # Plot smoothed metric (lighter red)
    fig.add_trace(
        go.Scatter(
            x=t_metric_smooth,
            y=metric_smooth,
            mode="lines",
            line=dict(width=5, color="#FF6692"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} hours<br>{metric_label} (smoothed)=%{{y:.4f}}<extra></extra>",
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
                # The window used in the analysis (the fit result does not store it)
                window_points = int(
                    gs.get("window_points", plate["params"]["window_points"])
                )
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
                y_model = evaluate_parametric_model(t_model, model_type, params)
                # Phenomenological models return ln(N/N0), so convert to OD
                if model_type in MODEL_REGISTRY["phenomenological"]:
                    y_model = log_to_linear(y_model, params["N0"])

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
            fig.add_trace(
                go.Scatter(
                    x=t_model,
                    y=metric_model,
                    mode="lines",
                    line=dict(width=5, color="#8dcde0"),
                    hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} hours<br>{metric_label} (fitted)=%{{y:.4f}}<extra></extra>",
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
                "exp_phase_start": float(exp_start),
                "exp_phase_end": float(exp_end),
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
    fig.update_xaxes(showgrid=False, title="Time (hours)", range=x_range)
    # y range from the smoothed curve, so raw-derivative spikes don't flatten it
    finite = np.asarray(metric_smooth, dtype=float)
    finite = finite[np.isfinite(finite)]
    y_range = None
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.05 * (hi - lo) or 0.05 * abs(hi) or 0.05
        y_range = [lo - pad, hi + pad]
    fig.update_yaxes(showgrid=False, title=y_axis_title, range=y_range)
    return fig
