import io
from typing import Optional

import growthcurves.plot as gc_plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from check_fit_plots import (
    INFO_ACCENT,
    INFO_CURVE,
    INFO_INK_2,
    INFO_POINTS,
    style_info_axes,
)
from growthcurve_app.analyze import match_selected_times
from growthcurve_app.pipeline import curve_series, segments
from plotly.subplots import make_subplots
from matplotlib.dates import DateFormatter


def create_figure_bytes_to_download(fig: plt.Figure, fmt: str = "pdf") -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt)
    return buf


def plot_growth_data_w_mask(
    df_wide: pd.DataFrame,
    df_mask: pd.DataFrame,
    sharey: bool = False,
    sharex: bool = True,
    is_data_index: bool = True,
    ticks_x_axis_interval: Optional[int] = None,
    ticks_y_axis_nbins: int = 5,
    ticks_x_axis_nbins: Optional[int] = 25,
) -> plt.Figure:
    """Plot optical density (OD) growth data."""
    # ?check that index is datetime and columns are numeric?

    units = df_wide.shape[1]
    fig, axes = plt.subplots(
        units, 1, figsize=(10, 2 * units), sharey=sharey, sharex=sharex, squeeze=False
    )
    axes = axes.flatten()
    df_columns = df_wide.columns
    index_name = df_wide.index.name
    df_wide = df_wide.loc[df_mask.index].reset_index()
    df_mask = df_mask.reset_index()
    # grid container (reactive to UI changes)
    for col, ax in zip(df_columns, axes, strict=True):
        mask = df_mask[col]
        # plot kept values in blue
        df_wide.loc[~mask].plot.scatter(
            x=index_name,
            y=col,
            rot=45,
            c="blue",
            ax=ax,
            alpha=0.1,
            s=1,
            title=f"Reactor: {col}",  # Customize legend text here
        )
        # Plot removed values in red
        df_wide.loc[mask].plot.scatter(
            x=index_name,
            y=col,
            rot=45,
            c="red",
            ax=ax,
            alpha=1.0,
            s=2,
            title=f"Reactor: {col}",  # Customize legend text here
        )
        if not is_data_index:
            if ticks_x_axis_interval:
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(ticks_x_axis_interval)
                )  # tick every x units
            else:
                ax.xaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=ticks_x_axis_nbins)
                )  # tick every x units
        ax.yaxis.set_major_locator(
            ticker.MaxNLocator(nbins=ticks_y_axis_nbins)
        )  # automatic y-axis ticks: apply number

    fig.tight_layout()

    if is_data_index:
        date_form = DateFormatter("%Y-%m-%d %H:%M")
        for ax in axes:
            ax.xaxis.set_major_formatter(date_form)

    return fig


REMOVED_POINTS = "#F2A7A7"  # mild red: reads left out of every fit
UMAX_MARKER = dict(size=10, color=INFO_ACCENT, line=dict(color="white", width=2))


def _umax_points(curves, key: str) -> np.ndarray:
    """(time_at_umax, stats[key]) of each curve where both are finite, as (n, 2)."""
    pts = [(c["stats"].get("time_at_umax"), c["stats"].get(key)) for c in curves]
    return np.array(
        [p for p in pts if None not in p and np.isfinite(p).all()], dtype=float
    ).reshape(-1, 2)


def plot_dilution_overview(
    s: pd.Series,
    curves: list[tuple[dict, dict | None]],
    selected: dict,
    drops=(),
) -> go.Figure:
    """Whole run of one reactor: the curves between dilutions in alternating
    shading (numbered as in the window controls), the reads left out of every fit
    (mild red), the fit and μmax of each curve and the selected curve (outlined).
    Underneath, on the same time axis, the instantaneous specific growth rate of
    each fitted model.

    ``curves`` holds (curve, fit result) pairs; fits are drawn with
    ``growthcurves.plot.prepare_fitted_curve`` and their μ is the "Fitted" trace of
    ``growthcurves.plot.plot_derivative_metric``.
    """
    band = dict(type="rect", xref="x", yref="paper", y0=0, y1=1, layer="below")
    curves, fits = zip(*sorted(curves, key=lambda cf: cf[0]["start"]), strict=True)
    shapes = [
        band | dict(x0=c["start"], x1=c["end"], fillcolor="#E5E7EB", line_width=0)
        for c in curves[::2]
    ]
    shapes.append(
        band
        | dict(
            x0=selected["start"],
            x1=selected["end"],
            line=dict(color=INFO_INK_2, width=2),
            layer="above",
        )
    )
    umax, mu_max = _umax_points(curves, "od_at_umax"), _umax_points(curves, "mu_max")

    # All fits, and the model μ of each, as one line each separated by gaps. μ only
    # over the reads the model was fitted to: the fit's time range (as drawn by
    # prepare_fitted_curve) and, after a lasso selection, the selected reads
    fit_x, fit_y, mu_x, mu_y = [], [], [], []
    for c, fit in zip(curves, fits, strict=True):
        xy = gc_plot.prepare_fitted_curve(fit)
        if xy is None:
            continue
        fit_x += [*xy[0], None]
        fit_y += [*xy[1], None]
        cs = curve_series(s, c["start"], c["end"])
        cs = cs[(cs.index >= xy[0][0]) & (cs.index <= xy[0][-1])]
        if c["used_times"]:
            cs = cs.iloc[
                match_selected_times(
                    cs.index.to_numpy(float), np.asarray(c["used_times"], dtype=float)
                )
            ]
        if len(cs) < 3:
            continue
        mu_fig = gc_plot.plot_derivative_metric(
            cs.index.to_numpy(float), cs.to_numpy(float), metric="mu", fit_result=fit
        )
        for tr in mu_fig.data:
            if tr.name == "Fitted":
                mu_x += [*tr.x, None]
                mu_y += [*tr.y, None]

    # Reads in no window (inside a drop, or trimmed at the edges of the run) are left
    # out of every fit: drawn in mild red
    windows = segments(s.index.to_numpy(float), drops) if len(s) > 1 else []
    in_window = np.zeros(len(s), bool)
    for a, b in windows:
        in_window |= s.index.isin(curve_series(s, a, b).index)
    kept, removed = s[in_window], s[~in_window]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4]
    )
    fig.add_traces(
        [
            go.Scattergl(
                x=kept.index,
                y=kept.to_numpy(),
                mode="markers",
                marker=dict(size=4, color=INFO_POINTS),
                hovertemplate="Time=%{x:.2f} h<br>OD=%{y:.4f}<extra></extra>",
            ),
            go.Scattergl(
                x=removed.index,
                y=removed.to_numpy(),
                mode="markers",
                marker=dict(size=4, color=REMOVED_POINTS),
                hovertemplate=(
                    "Time=%{x:.2f} h<br>OD=%{y:.4f}<br>left out<extra></extra>"
                ),
            ),
            go.Scattergl(
                x=fit_x,
                y=fit_y,
                mode="lines",
                line=dict(color=INFO_CURVE, width=3),
                hoverinfo="skip",
            ),
            # WebGL like the data, so the markers are drawn on top of the points
            go.Scattergl(
                x=umax[:, 0],
                y=umax[:, 1],
                mode="markers",
                marker=UMAX_MARKER,
                hovertemplate="μmax at %{x:.2f} h<br>OD=%{y:.4f}<extra></extra>",
            ),
        ],
        rows=1,
        cols=1,
    )
    fig.add_traces(
        [
            go.Scattergl(
                x=mu_x,
                y=mu_y,
                mode="lines",
                line=dict(color=INFO_CURVE, width=3),
                hovertemplate="Time=%{x:.2f} h<br>μ=%{y:.4f} h⁻¹<extra></extra>",
            ),
            go.Scattergl(
                x=mu_max[:, 0],
                y=mu_max[:, 1],
                mode="markers",
                marker=UMAX_MARKER,
                hovertemplate="μmax=%{y:.4f} h⁻¹ at %{x:.2f} h<extra></extra>",
            ),
        ],
        rows=2,
        cols=1,
    )
    # Window numbers at the top centre of each window (excluded windows counted too,
    # as in the window controls)
    annotations = [
        dict(
            x=(a + b) / 2,
            y=1,
            xref="x",
            yref="paper",
            yanchor="top",
            text=f"<b>{i}</b>",
            showarrow=False,
            font=dict(size=14, color=INFO_INK_2),
        )
        for i, (a, b) in enumerate(windows, start=1)
    ]
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        height=480,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=50),
        title=f"Whole run – {selected['reactor']}",
    )
    # x-axis range matches the data exactly (no gap at the edges)
    fig.update_xaxes(range=[float(s.index.min()), float(s.index.max())])
    fig.update_xaxes(title="Time (hours)", title_standoff=10, row=2, col=1)
    # Zoom and pan along time only
    fig.update_yaxes(title_standoff=10, fixedrange=True)
    fig.update_yaxes(title="OD600", row=1, col=1)
    fig.update_yaxes(title="μ (h⁻¹)", row=2, col=1)
    return style_info_axes(fig)
