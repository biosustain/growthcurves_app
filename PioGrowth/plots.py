import io
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter

st.cache_data()


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
    for col, ax in zip(df_columns, axes):
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


def plot_growth_data_w_peaks(
    df_wide: pd.DataFrame,
    peaks: pd.DataFrame,
    is_data_index: bool = True,
) -> plt.Figure:
    """Plot optical density (OD) growth data."""
    # ?check that index is datetime and columns are numeric?

    units = df_wide.shape[1]
    fig, axes = plt.subplots(
        units, 1, figsize=(10, 2 * units), sharex=True, squeeze=False
    )
    axes = axes.flatten()
    df_columns = df_wide.columns
    index_name = df_wide.index.name
    # grid container (reactive to UI changes)
    df_wide = df_wide.reset_index()
    for i, (col, ax) in enumerate(zip(df_columns, axes)):
        # plot kept values in blue
        df_wide.plot.scatter(
            x=index_name,
            y=col,
            rot=45,
            c=f"C{i}",
            ax=ax,
            alpha=0.1,
            s=1,
            title=f"Reactor: {col}",  # Customize legend text here
        )
        # Plot removed values in red
        peak_times = peaks[col].dropna().index
        for timepoint in peak_times:
            ax.axvline(x=timepoint, color="grey", alpha=0.3, linestyle="--")
    ax = axes[-1]
    if is_data_index:
        date_form = DateFormatter("%Y-%m-%d %H:%M")
        _ = ax.xaxis.set_major_formatter(date_form)
    fig = ax.get_figure()
    fig.tight_layout()

    return fig, axes


def plot_growth_data(
    splines,
    titles=None,
    ylabel="OD readings",
    xlabel="timepoints (rounded)",
):
    rows = (splines.shape[-1] + 1) // 2
    axes = splines.plot.line(
        style=".",
        ms=2,
        subplots=True,
        layout=(-1, 2),
        sharex=True,
        sharey=False,
        title=titles,
        ylabel=ylabel,
        xlabel=xlabel,
        legend=False,
        figsize=(10, rows * 2),
    )
    _axes = axes.flatten()
    fig = _axes[-1].get_figure()
    fig.tight_layout()
    return fig, axes


def plot_derivatives(
    derivatives: pd.DataFrame, titles=None, xlabel: str = "timepoints (rounded)"
) -> plt.Figure:
    rows = (derivatives.shape[-1] + 1) // 2
    axes = derivatives.plot.line(
        style=".",
        ms=2,
        subplots=True,
        layout=(-1, 2),
        title=titles,
        ylabel="1st derivative",
        xlabel=xlabel,
        legend=False,
        sharex=True,
        sharey=False,
        figsize=(10, rows * 2),
    )
    _axes = axes.flatten()
    decimal_place = int(np.log10(derivatives.max().min()) - 1)
    if decimal_place < 1:
        decimal_place = f"%.{np.abs(decimal_place)}f"
    else:
        decimal_place = "%.1f"
    for ax in _axes:
        _ = ax.yaxis.set_major_formatter(FormatStrFormatter(decimal_place))
    ax = _axes[-1]
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, axes
