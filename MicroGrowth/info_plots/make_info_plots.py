"""Generate the explanatory figures shown in the MicroGrowth app.

Run from the repository root (needs matplotlib and growthcurves):

    python MicroGrowth/info_plots/make_info_plots.py

Every PNG is written next to this script (or to $INFO_PLOTS_OUT), overwriting
the previous version.
Curves are computed with growthcurves itself (model functions, sliding window and
spline fits), so the figures match what the app calculates.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from growthcurves import models  # noqa: E402
from growthcurves.non_parametric import fit_sliding_window, fit_spline  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(os.environ.get("INFO_PLOTS_OUT", Path(__file__).resolve().parent))

# --- style --------------------------------------------------------------------
INK = "#262730"  # app text colour
INK_2 = "#5F6368"  # secondary text
AXIS = "#9AA0A6"
REF = "#6B7280"  # reference lines and leaders
GRID = "#D9DCE1"
CURVE = "#2B7BB9"  # categorical slot 1 (validated with slot 2)
ACCENT = "#E4572E"  # categorical slot 2: the feature being explained
PHASE = "#66BB6A"  # app primary colour: exponential phase wash
POINTS = "#A7ADB4"
HEADER = "#EAF4EA"  # table header tint of the app primary colour

DPI = 300
SIZE = (10, 5.5)
FS_AXIS = 17
FS_LABEL = 13.5

plt.rcParams.update(
    {
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.linewidth": 1.4,
        "savefig.facecolor": "white",
    }
)

MU = r"$\mu_{\mathrm{max}}$"


def schematic_axes(xlabel="Time", ylabel=r"$\ln(N)$", size=SIZE):
    """Axes with only left/bottom spines, arrowheads and no ticks."""
    fig, ax = plt.subplots(figsize=size)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(1, 0, ">", color=AXIS, ms=7, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^", color=AXIS, ms=7, transform=ax.transAxes, clip_on=False)
    ax.set_xlabel(xlabel, fontsize=FS_AXIS, color=INK, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS, color=INK, labelpad=12)
    return fig, ax


def label(ax, text, xy, xytext, rad=0.2, ha="center", va="center", arrow=True):
    """Text in a quiet rounded box with an optional curved leader to xy."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=FS_LABEL,
        color=INK,
        ha=ha,
        va=va,
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.4,rounding_size=0.5", fc="white", ec=GRID, lw=1
        ),
        arrowprops=(
            dict(
                arrowstyle="-|>,head_length=0.5,head_width=0.25",
                color=REF,
                lw=1.3,
                shrinkA=2,
                shrinkB=5,
                connectionstyle=f"arc3,rad={rad}",
            )
            if arrow
            else None
        ),
    )


def hline(ax, y, x0=None, x1=None):
    x0 = ax.get_xlim()[0] if x0 is None else x0
    x1 = ax.get_xlim()[1] if x1 is None else x1
    ax.plot([x0, x1], [y, y], ls=(0, (5, 4)), color=REF, lw=1.4, zorder=2)


def dot(ax, x, y, color=ACCENT, size=11, z=6):
    ax.plot(x, y, "o", ms=size, color=color, mec="white", mew=2, zorder=z)


def phase_wash(ax, t0, t1):
    ax.axvspan(t0, t1, color=PHASE, alpha=0.13, lw=0, zorder=0)


def save(fig, name, transparent=False):
    """Save as PNG; transparent=True lets the app's container background show through."""
    fig.savefig(
        OUT / f"{name}.png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.2,
        transparent=transparent,
    )
    plt.close(fig)
    print("wrote", name)


# --- shared synthetic growth curve ---------------------------------------------
T = np.linspace(0, 24, 400)
N0, A_LN, MU_MAX, LAG = 0.02, 4.0, 0.6, 4.0
LN_CURVE = np.log(N0) + models.phenom_gompertz_model_ln(T, A_LN, MU_MAX, LAG)


def noisy_data(n=49, sd=0.035, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, n)
    ln_n = np.log(N0) + models.phenom_gompertz_model_ln(t, A_LN, MU_MAX, LAG)
    return t, np.exp(ln_n) * np.exp(rng.normal(0, sd, n))


def max_slope(t, y):
    """Index of the steepest point of y(t)."""
    return int(np.argmax(np.gradient(y, t)))


# =============================================================================
# One figure per growth method: how μmax, the exponential phase and max OD are
# calculated (as in growthcurves.inference). Drawn in ln(N), where the tangent at
# μmax is a straight line. Phase boundaries always use the tangent method.
# =============================================================================
START_TANGENT = f"Start = tangent to the growth\ncurve at {MU} meets baseline OD"
END_TANGENT = f"End = tangent to the growth\ncurve at {MU} meets max OD"


def method_figure(
    name,
    *,
    tm,
    ym,
    slope,
    base,
    top,
    mu_text,
    start_text,
    end_text,
    max_text,
    curve=None,
    data=None,
    fitted_lag=None,
    extra=None,
    t_max=24,
    pos=None,
):
    """Curve/data, μmax point and tangent, baseline and max OD, exponential phase."""
    t_end = tm + (top - ym) / slope
    t_start = fitted_lag if fitted_lag is not None else tm + (base - ym) / slope
    fig, ax = schematic_axes()
    span = top - base
    ax.set_xlim(-0.5, t_max + 1)
    ax.set_ylim(base - 0.12 * span, top + 0.16 * span)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()

    def frac(fx, fy):
        return x_lo + fx * (x_hi - x_lo), y_lo + fy * (y_hi - y_lo)

    phase_wash(ax, t_start, t_end)
    hline(ax, top)
    hline(ax, base)
    if extra:
        extra(ax)
    if data is not None:
        ax.plot(*data, "o", ms=5.5, color=POINTS, zorder=3)
    if curve is not None:
        ax.plot(*curve, color=CURVE, lw=3, zorder=4)
    # Tangent at μmax; with a fitted lag only the part setting the end is drawn
    tt = np.linspace(tm if fitted_lag is not None else t_start, t_end, 10)
    ax.plot(tt, ym + slope * (tt - tm), color=ACCENT, lw=2.5, zorder=5)
    if fitted_lag is not None:
        ax.plot(
            [t_start, t_start], [y_lo, top + 0.04 * span], color=INK_2, lw=2, zorder=3
        )
    if not np.isclose(t_start, tm):  # no lag: start and μmax are the same point
        dot(ax, t_start, base, color=INK_2, size=9)
    dot(ax, tm, ym)
    dot(ax, t_end, top, color=INK_2, size=9)
    x_txt, ha = 0.5 * (t_start + t_end), "center"
    if (
        x_txt < frac(0.12, 0)[0]
    ):  # narrow phase near the axis: start the text at the phase
        x_txt, ha = t_start + 0.15, "left"
    ax.text(
        x_txt,
        top + 0.07 * span,
        "Exponential phase",
        ha=ha,
        fontsize=FS_LABEL,
        color=INK,
    )

    pos = {
        "max": (0.87, 0.72),
        "end": (0.62, 0.56),
        "mu": (0.7, 0.3),
        "start": (0.14, 0.5),
        **(pos or {}),
    }
    label(
        ax,
        max_text,
        xy=(frac(0.87, 0)[0], top),
        xytext=frac(*pos["max"]),
        rad=0 if pos["max"][0] == 0.87 else 0.2,
    )
    label(ax, end_text, xy=(t_end, top), xytext=frac(*pos["end"]), rad=0.25)
    if mu_text:
        label(ax, mu_text, xy=(tm, ym), xytext=frac(*pos["mu"]), rad=-0.25)
    label(ax, start_text, xy=(t_start, base), xytext=frac(*pos["start"]), rad=0.2)
    # Shown inside the tinted Step 3 container, so no white background
    save(fig, name, transparent=True)


def sliding_window():
    t, n = noisy_data()
    p = fit_sliding_window(t, n, window_points=5)["params"]
    tm = p["time_at_umax"]

    def window(ax):
        w = (t >= p["fit_t_min"]) & (t <= p["fit_t_max"])
        ax.plot(t[w], np.log(n[w]), "o", ms=7, color=INK, mec="white", mew=1, zorder=6)

    method_figure(
        "sliding_window",
        tm=tm,
        ym=p["slope"] * tm + p["intercept"],
        slope=p["slope"],
        base=np.log(n).min(),
        top=np.log(n).max(),
        data=(t, np.log(n)),
        extra=window,
        mu_text=f"{MU} = slope of the best\nsliding window (dark points)",
        start_text=START_TANGENT,
        end_text=END_TANGENT,
        max_text="Max OD =\nmax of data",
        pos={"start": (0.47, 0.14)},
    )


def spline():
    t, n = noisy_data()
    p = fit_spline(t, n)["params"]
    s = models.spline_from_params(p)
    ts = np.linspace(p["fit_t_min"], p["fit_t_max"], 400)
    tm = p["time_at_umax"]
    method_figure(
        "spline",
        tm=tm,
        ym=float(s(tm)),
        slope=p["mu_max"],
        base=np.log(n).min(),
        top=np.log(n).max(),
        data=(t, np.log(n)),
        curve=(ts, s(ts)),
        mu_text=f"{MU} = steepest slope\nof the spline",
        start_text=START_TANGENT,
        end_text=END_TANGENT,
        max_text="Max OD =\nmax of data",
        pos={"start": (0.47, 0.14)},
    )


def _model_figure(
    name,
    t,
    y,
    *,
    top,
    mu_text,
    start_text,
    end_text,
    max_text,
    fitted_lag=None,
    extra=None,
    pos=None,
):
    """Parametric model: μmax at the steepest point of ln N, baseline = min of the fit."""
    i = max_slope(t, y)
    method_figure(
        name,
        tm=t[i],
        ym=y[i],
        slope=np.gradient(y, t)[i],
        base=y.min(),
        top=top,
        curve=(t, y),
        fitted_lag=fitted_lag,
        extra=extra,
        t_max=t[-1],
        pos=pos,
        mu_text=mu_text,
        start_text=start_text,
        end_text=end_text,
        max_text=max_text,
    )


def mech_models():
    t = np.linspace(0, 24, 500)
    K, n0, mu = 1.0, 0.02, 0.55
    common = dict(
        top=np.log(K),
        end_text=f"End = tangent to the growth\ncurve at {MU} meets K",
        max_text="Max OD =\nK (fitted)",
    )
    for name, y, extra in (
        ("mech_logistic", models.mech_logistic_model(t, mu, K, n0), None),
        ("mech_gompertz", models.mech_gompertz_model(t, 0.3, K, n0), None),
        ("mech_richards", models.mech_richards_model(t, mu, K, n0, 1.0), None),
    ):
        # No lag phase: ln N is steepest at the first time point
        _model_figure(
            name,
            t,
            np.log(y),
            mu_text=None,
            extra=extra,
            pos={"start": (0.6, 0.25), **({"end": (0.62, 0.68)} if extra else {})},
            start_text=f"{MU} and start at t = 0: steepest slope\n"
            "of the fitted curve (this model has no lag phase)",
            **common,
        )
    h0, mu_b = 4.0, 0.8
    _model_figure(
        "mech_baranyi",
        t,
        np.log(models.mech_baranyi_model(t, mu_b, K, n0, h0)),
        fitted_lag=h0 / mu_b,
        start_text="Start = fitted lag\n" + r"$h_0/\mu$",
        mu_text=f"{MU} = steepest slope of the\nfitted curve (from μ, K, "
        + r"$N_0$, $h_0$)",
        **common,
    )


def phenom_models():
    t = np.linspace(0, 24, 500)
    A, mu, lam, n0 = 4.0, 0.7, 4.0, 0.02
    common = dict(
        fitted_lag=lam,
        mu_text=f"{MU} = fitted parameter",
        start_text="Start = fitted\nlag " + r"$\lambda$",
        end_text=END_TANGENT,
        max_text="Max OD = max\nof fitted curve",
    )
    for name, y, extra in (
        ("phenom_logistic", models.phenom_logistic_model_ln(t, A, mu, lam), None),
        ("phenom_gompertz", models.phenom_gompertz_model_ln(t, A, mu, lam), None),
        ("phenom_richards", models.phenom_richards_model_ln(t, A, mu, lam, 2.0), None),
    ):
        ln_n = np.log(n0) + y
        _model_figure(
            name,
            t,
            ln_n,
            top=ln_n.max(),
            extra=extra,
            pos={"end": (0.62, 0.68)} if extra else None,
            **common,
        )

    # Biphasic: the second phase stays slower than μmax so μmax is the first phase
    tm = np.linspace(0, 30, 600)
    ln_n = np.log(n0) + models.phenom_gompertz_modified_model_ln(
        tm, A, mu, lam, 0.12, 30.0
    )

    def second_phase(ax):
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        i = np.searchsorted(tm, 26.0)
        label(
            ax,
            "second growth phase\n" + r"(rate $\alpha$, onset $t_{\mathrm{shift}}$)",
            xy=(tm[i], ln_n[i]),
            xytext=(x_lo + 0.86 * (x_hi - x_lo), y_lo + 0.42 * (y_hi - y_lo)),
            rad=0.25,
        )

    _model_figure(
        "phenom_gompertz_modified",
        tm,
        ln_n,
        top=ln_n.max(),
        extra=second_phase,
        pos={"max": (0.72, 0.95), "end": (0.61, 0.74)},
        **common,
    )


def annotations():
    t, n = noisy_data(sd=0.03)
    y = np.exp(LN_CURVE)
    ln_slope = np.gradient(LN_CURVE, T)
    i = int(np.argmax(ln_slope))
    mu_max, tm, nm = ln_slope[i], T[i], y[i]
    base, top = y.min(), y.max()
    t0 = tm + np.log(base / nm) / mu_max
    t1 = tm + np.log(top / nm) / mu_max
    fig, ax = schematic_axes(ylabel="N")
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-0.06 * top, 1.18 * top)
    phase_wash(ax, t0, t1)
    hline(ax, top)
    hline(ax, base)
    ax.plot(t, n, "o", ms=5.5, color=POINTS, zorder=3)
    tt = np.linspace(t0, t1, 200)
    ax.plot(
        tt,
        nm * np.exp(mu_max * (tt - tm)),
        color=ACCENT,
        lw=2.2,
        ls=(0, (5, 3)),
        zorder=5,
    )
    ax.plot(T, y, color=CURVE, lw=3, zorder=4)
    dot(ax, tm, nm)
    label(ax, "Max OD", xy=(21, top), xytext=(21, 0.82 * top), rad=0)
    label(ax, "Baseline OD", xy=(21, base), xytext=(21, 0.2 * top), rad=0)
    label(ax, "Max growth\nrate point", xy=(tm, nm), xytext=(2.6, 0.5 * top), rad=0.25)
    label(
        ax,
        "Tangent line\nat max growth",
        xy=(t1 - 0.25, 0.87 * top),
        xytext=(2.9, 0.86 * top),
        rad=-0.2,
    )
    label(
        ax,
        "Fitted model curve",
        xy=(14, float(np.interp(14, T, y))),
        xytext=(17.6, 0.55 * top),
        rad=0.25,
    )
    ax.text(
        0.5 * (t0 + t1),
        1.07 * top,
        "Phase boundaries",
        ha="center",
        fontsize=FS_LABEL,
        color=INK,
    )
    for tb in (t0, t1):
        ax.plot([tb, tb], [0, 1.03 * top], color=PHASE, lw=1.6, zorder=1)
    save(fig, "annotations")


# =============================================================================
# Upload / naming figures (tables and plate)
# =============================================================================
def draw_table(
    ax, x, y, headers, rows, col_w=1.0, row_h=0.42, more_rows=True, more_cols=False
):
    """Draw a spreadsheet-style table with its top-left corner at (x, y)."""
    n_cols = len(headers)
    for r, row in enumerate([headers, *rows]):
        for c, val in enumerate(row):
            is_head = r == 0 or c == 0
            ax.add_patch(
                Rectangle(
                    (x + c * col_w, y - (r + 1) * row_h),
                    col_w,
                    row_h,
                    fc=HEADER if is_head else "white",
                    ec=GRID,
                    lw=1,
                )
            )
            ax.text(
                x + (c + 0.5) * col_w,
                y - (r + 0.5) * row_h,
                str(val),
                ha="center",
                va="center",
                fontsize=12.5,
                color=INK,
                weight="bold" if is_head else "normal",
            )
    bottom = y - (len(rows) + 1) * row_h
    if more_rows:
        for c in range(n_cols):
            ax.text(
                x + (c + 0.5) * col_w,
                bottom - 0.2,
                "⋮",
                ha="center",
                va="center",
                fontsize=14,
                color=INK_2,
                fontfamily="DejaVu Sans",
            )
    if more_cols:
        for r in range(len(rows) + 1):
            ax.text(
                x + n_cols * col_w + 0.22,
                y - (r + 0.5) * row_h,
                "…",
                ha="center",
                va="center",
                fontsize=14,
                color=INK_2,
            )


def _two_option_figure(name, left, right, size=(12, 4.0)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.55, 4.6)
    ax.axis("off")
    # Centre each option (titles and table) in its half; the divider is at x = 6.05
    for (title, subtitle, table_kw), (lo, hi) in (
        (left, (0.0, 6.05)),
        (right, (6.05, 12.0)),
    ):
        width = len(table_kw["headers"]) * table_kw.get("col_w", 1.0)
        cx = (lo + hi) / 2
        # Centre the table including its "…" continuation column, if any
        x0 = cx - (width + (0.44 if table_kw.get("more_cols") else 0.0)) / 2
        ax.text(
            cx,
            4.45,
            title,
            ha="center",
            va="top",
            fontsize=16,
            weight="bold",
            color=INK,
        )
        ax.text(
            cx,
            3.98,
            subtitle,
            ha="center",
            va="top",
            fontsize=13,
            color=INK_2,
            linespacing=1.35,
        )
        draw_table(ax, x0, 3.05, **table_kw)
    ax.plot([6.05, 6.05], [0.65, 4.4], color=GRID, lw=1)
    # Invisible frame over the full width so the tight crop keeps both halves equal
    ax.add_patch(Rectangle((0, 0.55), 12, 4.05, fill=False, ec="none"))
    save(fig, name)


def data_upload():
    vals = [
        ["0", "0.0496", "0.0500", "0.0484"],
        ["12", "0.0495", "0.0504", "0.0482"],
        ["24", "0.0492", "0.0502", "0.0480"],
        ["36", "0.0491", "0.0501", "0.0480"],
    ]
    _two_option_figure(
        "data_upload",
        (
            "Option 1",
            "Sample names as column headers",
            dict(
                headers=["Time", "Sample 1", "Strain1_LB", "1"],
                rows=vals,
                col_w=1.3,
                more_cols=True,
            ),
        ),
        (
            "Option 2",
            "Well positions as column headers,\nplus a separate name mapping file",
            dict(
                headers=["Time", "A1", "A2", "A3"], rows=vals, col_w=1.3, more_cols=True
            ),
        ),
    )


def naming_upload():
    plate = [
        ["A", "Sample1", "Strain1_LB", "1"],
        ["B", "Sample1", "Strain1_LB", "1"],
        ["C", "Sample2", "Strain2_LB", "2"],
        ["D", "Sample2", "Strain2_LB", "2"],
    ]
    long = [["A1", "Sample1"], ["A2", "Strain1_LB"], ["A3", "1"], ["B1", "Sample1"]]
    _two_option_figure(
        "naming_upload",
        (
            "Option 1",
            "96-well plate map containing\nyour sample names",
            dict(
                headers=["rows", "1", "2", "3"], rows=plate, col_w=1.3, more_cols=True
            ),
        ),
        (
            "Option 2",
            "Long-format table with one\nwell and sample name per row",
            dict(headers=["Well", "Sample"], rows=long, col_w=1.6),
        ),
    )


def naming_convention():
    fig = plt.figure(figsize=(10, 6.0))
    # A sample name, as typed in the plate map, split at its first underscore
    top = fig.add_axes([0, 0.6, 1, 0.4])
    top.set_xlim(0, 10)
    top.set_ylim(0, 1)
    top.axis("off")
    name, sep = "Strain1_Condition1", "Strain1_Condition1".index("_")
    y_txt = 0.56
    txt = top.text(
        5.4,
        y_txt,
        name,
        ha="center",
        va="center",
        fontsize=26,
        color=INK,
        fontfamily="Menlo",
        zorder=3,
    )
    fig.canvas.draw()
    inv = top.transData.inverted()
    x0, x1 = (inv.transform(p)[0] for p in txt.get_window_extent().get_points())
    char = (x1 - x0) / len(name)  # monospace: every character has the same width
    split = x0 + sep * char

    # Spreadsheet-style cell holding the name, with a field label to its left
    pad, cell_lo, cell_hi = 0.3, y_txt - 0.14, y_txt + 0.14
    top.add_patch(
        Rectangle(
            (x0 - pad, cell_lo),
            x1 - x0 + 2 * pad,
            cell_hi - cell_lo,
            fc="white",
            ec=GRID,
            lw=1.2,
        )
    )
    top.text(
        x0 - pad - 0.2,
        y_txt,
        "Sample name",
        ha="right",
        va="center",
        fontsize=FS_LABEL,
        weight="bold",
        color=INK,
    )
    top.add_patch(
        Rectangle(
            (split, cell_lo + 0.03),
            char,
            cell_hi - cell_lo - 0.06,
            fc=ACCENT,
            alpha=0.18,
            lw=0,
            zorder=2,
        )
    )
    label(
        top,
        "split at the first underscore",
        xy=(split + char / 2, cell_hi - 0.01),
        xytext=(split + char / 2 + 2.4, 0.93),
        rad=0.25,
    )

    # Brackets under each part, labelled with what it becomes in plots
    yb = cell_lo - 0.06
    for a, b, color, text in (
        (x0, split, CURVE, "strain\n→ bar colour (legend)"),
        (split + char, x1, INK_2, "condition\n→ x-axis group"),
    ):
        a, b = a + 0.04, b - 0.04
        top.plot(
            [a, a, b, b],
            [yb + 0.04, yb, yb, yb + 0.04],
            color=color,
            lw=2,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        top.plot(
            [(a + b) / 2] * 2,
            [yb, yb - 0.05],
            color=color,
            lw=2,
            solid_capstyle="round",
        )
        top.text(
            (a + b) / 2,
            yb - 0.08,
            text,
            ha="center",
            va="top",
            fontsize=FS_LABEL,
            color=INK,
            linespacing=1.3,
        )

    ax = fig.add_axes([0.12, 0.08, 0.78, 0.47])
    conds = ["Condition1", "Condition2", "Condition3"]
    # Three replicates per sample; bars are their mean, error bars their SD
    reps1 = np.array([[0.47, 0.55, 0.62], [0.99, 0.91, 0.84], [0.72, 0.80, 0.86]])
    reps2 = np.array([[1.02, 0.94, 0.88], [0.53, 0.62, 0.70], [0.53, 0.44, 0.37]])
    x = np.arange(3)
    w = 0.3
    jitter = np.array([-0.09, 0.0, 0.09])
    for reps, dx, color, name in (
        (reps1, -w / 2 - 0.01, CURVE, "Strain1"),
        (reps2, w / 2 + 0.01, ACCENT, "Strain2"),
    ):
        ax.bar(x + dx, reps.mean(axis=1), w, color=color, label=name, zorder=3)
        ax.errorbar(
            x + dx,
            reps.mean(axis=1),
            yerr=reps.std(axis=1, ddof=1),
            fmt="none",
            ecolor=INK,
            elinewidth=1.6,
            capsize=6,
            capthick=1.6,
            zorder=4,
        )
        for xi, row in zip(x + dx, reps):
            ax.plot(
                xi + jitter, row, "o", ms=6, color="black", mec="white", mew=1, zorder=5
            )
    ax.set_xticks(x, conds, fontsize=FS_LABEL, color=INK)
    ax.set_yticks([])
    ax.set_ylabel("Growth metric", fontsize=FS_LABEL + 1, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 1.12)
    ax.legend(
        frameon=False,
        fontsize=FS_LABEL,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        handlelength=1.0,
        handleheight=1.0,
        labelcolor=INK,
    )
    save(fig, "naming_convention")


def blank_group():
    group_colors = [
        "#ccebc5",
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
    ]  # app blank-group colours
    blanks = {("A", 3), ("A", 6), ("A", 9), ("A", 12)}
    rows = "ABCDEFGH"
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.set_xlim(-0.8, 12.2)
    ax.set_ylim(-0.75, 8.9)
    ax.invert_yaxis()
    ax.axis("off")
    for r, row in enumerate(rows):
        ax.text(
            -0.35,
            r + 0.5,
            row,
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color=INK,
        )
        for c in range(1, 13):
            base = np.array(to_rgb(group_colors[(c - 1) // 3]))
            is_blank = (row, c) in blanks
            fc = base * 0.72 if is_blank else base
            ax.add_patch(
                FancyBboxPatch(
                    (c - 1 + 0.06, r + 0.06),
                    0.88,
                    0.88,
                    boxstyle="round,pad=0,rounding_size=0.12",
                    fc=fc,
                    ec="white",
                    lw=2,
                )
            )
            ax.text(
                c - 0.5,
                r + 0.5,
                f"{row}{c}",
                ha="center",
                va="center",
                fontsize=11,
                color=INK,
                weight="bold" if is_blank else "normal",
            )
    for c in range(1, 13):
        ax.text(
            c - 0.5,
            -0.3,
            str(c),
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color=INK,
        )
    # Legend
    for g, color in enumerate(group_colors):
        x = 0.1 + g * 2.6
        ax.add_patch(
            FancyBboxPatch(
                (x, 8.35),
                0.38,
                0.38,
                boxstyle="round,pad=0,rounding_size=0.08",
                fc=color,
                ec="none",
            )
        )
        ax.text(x + 0.55, 8.54, f"Group {g + 1}", va="center", fontsize=12.5, color=INK)
    ax.add_patch(
        FancyBboxPatch(
            (10.45, 8.35),
            0.38,
            0.38,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc=np.array(to_rgb(group_colors[0])) * 0.72,
            ec="none",
        )
    )
    ax.text(11.0, 8.54, "Blank", va="center", fontsize=12.5, color=INK)
    save(fig, "blank_group")


if __name__ == "__main__":
    sliding_window()
    spline()
    mech_models()
    phenom_models()
    annotations()
    data_upload()
    naming_upload()
    naming_convention()
    blank_group()
