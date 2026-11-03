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
        bbox=dict(boxstyle="round,pad=0.4,rounding_size=0.5", fc="white", ec=GRID, lw=1),
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


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.2)
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
# Growth descriptor methods
# =============================================================================
def sliding_window():
    t, n = noisy_data()
    fit = fit_sliding_window(t, n, window_points=5)["params"]
    fig, ax = schematic_axes()
    ax.plot(T, LN_CURVE, color=CURVE, lw=1.4, alpha=0.35, zorder=1)
    ax.plot(t, np.log(n), "o", ms=6, color=POINTS, zorder=3)
    ax.set_xlim(-0.5, 25)
    lo, hi = fit["fit_t_min"], fit["fit_t_max"]
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.09, lw=0, zorder=0)
    tt = np.linspace(lo - 0.6, hi + 0.6, 10)
    ax.plot(tt, fit["slope"] * tt + fit["intercept"], color=ACCENT, lw=3, zorder=5)
    xm = 0.5 * (lo + hi)
    label(
        ax,
        f"{MU} = slope of the window with\nthe steepest reliable fit",
        xy=(xm, fit["slope"] * xm + fit["intercept"]),
        xytext=(4.2, np.log(N0) + 3.3),
        rad=-0.3,
    )
    save(fig, "sliding_window")


def spline():
    t, n = noisy_data()
    p = fit_spline(t, n)["params"]
    s = models.spline_from_params(p)
    fig, ax = schematic_axes()
    ax.plot(t, np.log(n), "o", ms=6, color=POINTS, zorder=3)
    ts = np.linspace(p["fit_t_min"], p["fit_t_max"], 400)
    ax.plot(ts, s(ts), color=CURVE, lw=3, zorder=4)
    ax.set_xlim(-0.5, 25)
    tm = p["time_at_umax"]
    ym = float(s(tm))
    tt = np.linspace(tm - 2.2, tm + 2.2, 10)
    ax.plot(tt, ym + p["mu_max"] * (tt - tm), color=ACCENT, lw=2.5, zorder=5)
    dot(ax, tm, ym)
    label(
        ax,
        "Smoothing spline fitted to ln(N)",
        xy=(16, float(s(16))),
        xytext=(17.5, np.log(N0) + 2.4),
        rad=0.25,
    )
    label(
        ax,
        f"{MU} = steepest slope\nof the spline",
        xy=(tm, ym),
        xytext=(3.6, np.log(N0) + 3.0),
        rad=-0.3,
    )
    save(fig, "spline")


# =============================================================================
# Parametric models
# =============================================================================
def _mechanistic(name, y, t, K, N0_, extra=None, lag=None, n0_text=None):
    fig, ax = schematic_axes(ylabel="N")
    ax.plot(t, y, color=CURVE, lw=3, zorder=4)
    ax.set_xlim(-0.5, t[-1] + 1)
    ax.set_ylim(-0.04 * K, 1.12 * K)
    hline(ax, K)
    label(ax, "K  (carrying capacity)", xy=None or (t[-1] * 0.8, K), xytext=(t[-1] * 0.8, 0.85 * K), rad=0)
    dot(ax, t[0], N0_)
    label(ax, r"$N_0$  (initial population)", xy=(t[0], N0_),
          xytext=n0_text or (t[-1] * 0.2, 0.3 * K), rad=0.25)
    i = max_slope(t, y)
    label(
        ax,
        r"$\mu$  (intrinsic growth rate)" + "\nsets the steepness",
        xy=(t[i], y[i]),
        xytext=(t[-1] * 0.78, 0.4 * K),
        rad=-0.25,
    )
    if lag is not None:
        ax.plot([lag, lag], [0, 0.5 * K], color=REF, lw=1.4, ls=(0, (5, 4)), zorder=2)
        ax.annotate("", xy=(0, 0.46 * K), xytext=(lag, 0.46 * K),
                    arrowprops=dict(arrowstyle="<|-|>,head_length=0.5,head_width=0.22",
                                    color=ACCENT, lw=1.8, shrinkA=0, shrinkB=0))
        ax.text(lag / 2, 0.5 * K, r"lag  $\lambda = h_0/\mu$", ha="center", va="bottom",
                fontsize=FS_LABEL, color=INK)
    if extra:
        extra(ax)
    save(fig, name)


def mech_models():
    t = np.linspace(0, 24, 400)
    K, n0, mu = 1.0, 0.02, 0.55
    _mechanistic("mech_logistic", models.mech_logistic_model(t, mu, K, n0), t, K, n0)
    _mechanistic("mech_gompertz", models.mech_gompertz_model(t, 0.3, K, n0), t, K, n0)

    def richards_variants(ax):
        for beta, ls in ((0.3, (0, (2, 2))), (3.0, (0, (6, 3)))):
            yv = models.mech_richards_model(t, mu, K, n0, beta)
            ax.plot(t, yv, color=CURVE, lw=1.6, ls=ls, alpha=0.55, zorder=3)
        y03 = float(np.interp(21.0, t, models.mech_richards_model(t, mu, K, n0, 0.3)))
        ax.text(21.0, y03 - 0.07 * K, r"$\beta = 0.3$", ha="center", va="top",
                fontsize=FS_LABEL - 1, color=INK_2)
        ax.text(5.3, 0.62 * K, r"$\beta = 3$", fontsize=FS_LABEL - 1, color=INK_2)
        ax.text(9.6, 0.72 * K, r"$\beta = 1$", fontsize=FS_LABEL - 1, color=INK)
        label(ax, r"shape parameter $\beta$" + "\nchanges the curve's symmetry",
              xy=(0, 0), xytext=(19.3, 0.1 * K), arrow=False)

    _mechanistic(
        "mech_richards",
        models.mech_richards_model(t, mu, K, n0, 1.0),
        t,
        K,
        n0,
        extra=richards_variants,
    )
    h0 = 4.0
    _mechanistic(
        "mech_baranyi",
        models.mech_baranyi_model(t, 0.8, K, n0, h0),
        t,
        K,
        n0,
        lag=h0 / 0.8,
        n0_text=(3.6, 0.27 * K),
    )


def _phenomenological(name, y, t, A, lam, mu_max, extra=None, ymax=None, first_phase=None):
    fig, ax = schematic_axes(ylabel=r"$\ln(N_t / N_0)$")
    ax.plot(t, y, color=CURVE, lw=3, zorder=4)
    ax.set_xlim(-0.5, t[-1] + 1)
    top = ymax or 1.12 * A
    ax.set_ylim(-0.06 * top, top)
    hline(ax, A)
    hline(ax, 0)
    label(ax, "A  (maximum ln increase)", xy=(t[-1] * 0.8, A), xytext=(t[-1] * 0.8, 0.83 * A), rad=0)

    # Tangent at the inflection point: slope μmax, crosses 0 at λ by construction
    m = t <= first_phase if first_phase else slice(None)
    i = max_slope(t[m], y[m])
    t_i, y_i = t[i], y[i]
    t_end = t_i + (A - y_i) / mu_max
    tt = np.linspace(lam, t_end, 10)
    ax.plot(tt, y_i + mu_max * (tt - t_i), color=ACCENT, lw=2.5, zorder=5)
    dot(ax, t_i, y_i)
    dot(ax, lam, 0, color=INK_2, size=9)
    label(ax, f"slope = {MU}", xy=(t_i, y_i), xytext=(t_i + 6.5, 0.32 * A), rad=-0.25)
    label(ax, r"$\lambda$  (lag time)", xy=(lam, 0), xytext=(lam + 4.5, 0.08 * A), rad=0.2)
    if extra:
        extra(ax)
    save(fig, name)


def phenom_models():
    t = np.linspace(0, 24, 400)
    A, mu, lam = 4.0, 0.7, 4.0
    _phenomenological(
        "phenom_logistic", models.phenom_logistic_model_ln(t, A, mu, lam), t, A, lam, mu
    )
    _phenomenological(
        "phenom_gompertz", models.phenom_gompertz_model_ln(t, A, mu, lam), t, A, lam, mu
    )
    _phenomenological(
        "phenom_richards",
        models.phenom_richards_model_ln(t, A, mu, lam, 2.0),
        t,
        A,
        lam,
        mu,
    )

    tm = np.linspace(0, 32, 500)
    alpha, t_shift = 0.3, 32.0
    y = models.phenom_gompertz_modified_model_ln(tm, A, mu, lam, alpha, t_shift)

    def second_phase(ax):
        i = np.searchsorted(tm, 28.5)
        label(
            ax,
            "second growth phase\n" + r"(rate $\alpha$, onset $t_{\mathrm{shift}}$)",
            xy=(tm[i], y[i]),
            xytext=(22.5, 1.75 * A),
            rad=-0.25,
        )

    _phenomenological(
        "phenom_gompertz_modified", y, tm, A, lam, mu, extra=second_phase, ymax=2.15 * A,
        first_phase=14.0,
    )


# =============================================================================
# Phase boundary methods
# =============================================================================
def threshold_demo():
    mu_t = np.gradient(LN_CURVE, T)
    i = int(np.argmax(mu_t))
    mu_max = mu_t[i]
    frac = 0.5
    above = np.where(mu_t >= frac * mu_max)[0]
    t0, t1 = T[above[0]], T[above[-1]]
    fig, ax = schematic_axes(ylabel=r"Specific growth rate  $\mu$")
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(-0.04, 1.2 * mu_max)
    phase_wash(ax, t0, t1)
    hline(ax, mu_max)
    hline(ax, frac * mu_max)
    ax.plot(T, mu_t, color=CURVE, lw=3, zorder=4)
    dot(ax, T[i], mu_max)
    dot(ax, t0, frac * mu_max, color=INK_2, size=9)
    dot(ax, t1, frac * mu_max, color=INK_2, size=9)
    label(ax, MU, xy=(T[i], mu_max), xytext=(T[i] + 6, 1.1 * mu_max), rad=0.2)
    label(
        ax,
        "cutoff × " + MU + "\n(lag / exponential phase cutoff)",
        xy=(21.5, frac * mu_max),
        xytext=(19, 0.34 * mu_max),
        rad=0.0,
    )
    ax.text(0.5 * (t0 + t1), 0.12 * mu_max, "Exponential phase", ha="center",
            fontsize=FS_LABEL, color=INK)
    label(
        ax,
        "Phase boundaries: where μ\ncrosses the cutoff",
        xy=(t1, frac * mu_max),
        xytext=(18.3, 0.78 * mu_max),
        rad=0.15,
        arrow=True,
    )
    save(fig, "threshold_demo")


def tangent_demo():
    y = LN_CURVE
    i = max_slope(T, y)
    mu_max = np.gradient(y, T)[i]
    base, top = y.min(), y.max()
    t0 = T[i] + (base - y[i]) / mu_max
    t1 = T[i] + (top - y[i]) / mu_max
    fig, ax = schematic_axes()
    ax.set_xlim(-0.5, 25)
    ax.set_ylim(base - 0.35, top + 0.5)
    phase_wash(ax, t0, t1)
    hline(ax, base)
    hline(ax, top)
    tt = np.linspace(t0, t1, 10)
    ax.plot(tt, y[i] + mu_max * (tt - T[i]), color=ACCENT, lw=2.5, zorder=5)
    ax.plot(T, y, color=CURVE, lw=3, zorder=4)
    dot(ax, T[i], y[i])
    dot(ax, t0, base, color=INK_2, size=9)
    dot(ax, t1, top, color=INK_2, size=9)
    label(ax, "Max OD", xy=(21.5, top), xytext=(21.5, top - 0.6), rad=0)
    label(ax, "Baseline OD", xy=(21.5, base), xytext=(21.5, base + 0.6), rad=0)
    label(ax, f"Tangent at {MU}", xy=(T[i], y[i]), xytext=(T[i] + 5.5, y[i] - 0.9), rad=-0.25)
    label(
        ax,
        "Phase boundaries: where the\ntangent meets baseline and max OD",
        xy=(t1, top),
        xytext=(16.8, top - 1.9),
        rad=0.25,
    )
    ax.text(0.5 * (t0 + t1), top + 0.22, "Exponential phase", ha="center",
            fontsize=FS_LABEL, color=INK)
    save(fig, "tangent_demo")


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
    ax.plot(tt, nm * np.exp(mu_max * (tt - tm)), color=ACCENT, lw=2.2, ls=(0, (5, 3)), zorder=5)
    ax.plot(T, y, color=CURVE, lw=3, zorder=4)
    dot(ax, tm, nm)
    label(ax, "Max OD", xy=(21, top), xytext=(21, 0.82 * top), rad=0)
    label(ax, "Baseline OD", xy=(21, base), xytext=(21, 0.2 * top), rad=0)
    label(ax, "Max growth\nrate point", xy=(tm, nm), xytext=(2.6, 0.5 * top), rad=0.25)
    label(ax, "Tangent line\nat max growth", xy=(t1 - 0.25, 0.87 * top),
          xytext=(2.9, 0.86 * top), rad=-0.2)
    label(ax, "Fitted model curve", xy=(14, float(np.interp(14, T, y))),
          xytext=(17.6, 0.55 * top), rad=0.25)
    ax.text(0.5 * (t0 + t1), 1.07 * top, "Phase boundaries", ha="center",
            fontsize=FS_LABEL, color=INK)
    for tb in (t0, t1):
        ax.plot([tb, tb], [0, 1.03 * top], color=PHASE, lw=1.6, zorder=1)
    save(fig, "annotations")


# =============================================================================
# Upload / naming figures (tables and plate)
# =============================================================================
def draw_table(ax, x, y, headers, rows, col_w=1.0, row_h=0.42, more_rows=True, more_cols=False):
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
            ax.text(x + (c + 0.5) * col_w, bottom - 0.2, "⋮", ha="center", va="center",
                    fontsize=14, color=INK_2, fontfamily="DejaVu Sans")
    if more_cols:
        for r in range(len(rows) + 1):
            ax.text(x + n_cols * col_w + 0.22, y - (r + 0.5) * row_h, "…", ha="center",
                    va="center", fontsize=14, color=INK_2)


def _two_option_figure(name, left, right, size=(12, 4.0)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.55, 4.6)
    ax.axis("off")
    for (title, subtitle, table_kw), x0 in ((left, 0.3), (right, 6.4)):
        width = len(table_kw["headers"]) * table_kw.get("col_w", 1.0)
        cx = x0 + width / 2
        ax.text(cx, 4.45, title, ha="center", va="top", fontsize=16, weight="bold", color=INK)
        ax.text(cx, 3.98, subtitle, ha="center", va="top", fontsize=13, color=INK_2,
                linespacing=1.35)
        draw_table(ax, x0, 3.05, **table_kw)
    ax.plot([6.05, 6.05], [0.65, 4.4], color=GRID, lw=1)
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
            dict(headers=["Time", "Sample 1", "Strain1_LB", "1"], rows=vals, col_w=1.3, more_cols=True),
        ),
        (
            "Option 2",
            "Well positions as column headers,\nplus a separate name mapping file",
            dict(headers=["Time", "A1", "A2", "A3"], rows=vals, col_w=1.3, more_cols=True),
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
            dict(headers=["rows", "1", "2", "3"], rows=plate, col_w=1.3, more_cols=True),
        ),
        (
            "Option 2",
            "Long-format table with one\nwell and sample name per row",
            dict(headers=["Well", "Sample"], rows=long, col_w=1.6),
        ),
    )


def naming_convention():
    fig = plt.figure(figsize=(10, 5.5))
    # Name split into its two parts
    top = fig.add_axes([0, 0.74, 1, 0.26])
    top.set_xlim(0, 10)
    top.set_ylim(0, 1)
    top.axis("off")
    parts = [("Strain1", 3.05, CURVE), ("_", 4.32, None), ("Condition1", 5.75, INK_2)]
    for text, x, ec in parts:
        if ec is None:
            top.text(x, 0.62, text, ha="center", va="center", fontsize=24, color=INK)
            continue
        top.text(x, 0.62, text, ha="center", va="center", fontsize=24, color=INK, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.4", fc="white", ec=ec, lw=2.2))
    top.text(3.05, 0.12, "strain → bar colour (legend)", ha="center", fontsize=FS_LABEL, color=INK_2)
    top.text(5.75, 0.12, "condition → x-axis group", ha="center", fontsize=FS_LABEL, color=INK_2)

    ax = fig.add_axes([0.12, 0.1, 0.78, 0.58])
    conds = ["Condition1", "Condition2", "Condition3"]
    # Three replicates per sample; bars are their mean, error bars their SD
    reps1 = np.array([[0.47, 0.55, 0.62], [0.99, 0.91, 0.84], [0.72, 0.80, 0.86]])
    reps2 = np.array([[1.02, 0.94, 0.88], [0.53, 0.62, 0.70], [0.53, 0.44, 0.37]])
    x = np.arange(3)
    w = 0.3
    jitter = np.array([-0.09, 0.0, 0.09])
    for reps, dx, color, name in ((reps1, -w / 2 - 0.01, CURVE, "Strain1"),
                                  (reps2, w / 2 + 0.01, ACCENT, "Strain2")):
        ax.bar(x + dx, reps.mean(axis=1), w, color=color, label=name, zorder=3)
        ax.errorbar(x + dx, reps.mean(axis=1), yerr=reps.std(axis=1, ddof=1), fmt="none",
                    ecolor=INK, elinewidth=1.6, capsize=6, capthick=1.6, zorder=4)
        for xi, row in zip(x + dx, reps):
            ax.plot(xi + jitter, row, "o", ms=6, color="black", mec="white", mew=1, zorder=5)
    ax.set_xticks(x, conds, fontsize=FS_LABEL, color=INK)
    ax.set_yticks([])
    ax.set_ylabel("Growth metric", fontsize=FS_LABEL + 1, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=FS_LABEL, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              handlelength=1.0, handleheight=1.0, labelcolor=INK)
    save(fig, "naming_convention")


def blank_group():
    group_colors = ["#ccebc5", "#8dd3c7", "#ffffb3", "#bebada"]  # app blank-group colours
    blanks = {("A", 3), ("A", 6), ("A", 9), ("A", 12)}
    rows = "ABCDEFGH"
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.set_xlim(-0.8, 12.2)
    ax.set_ylim(-0.75, 8.9)
    ax.invert_yaxis()
    ax.axis("off")
    for r, row in enumerate(rows):
        ax.text(-0.35, r + 0.5, row, ha="center", va="center", fontsize=13, weight="bold", color=INK)
        for c in range(1, 13):
            base = np.array(to_rgb(group_colors[(c - 1) // 3]))
            is_blank = (row, c) in blanks
            fc = base * 0.72 if is_blank else base
            ax.add_patch(
                FancyBboxPatch((c - 1 + 0.06, r + 0.06), 0.88, 0.88,
                               boxstyle="round,pad=0,rounding_size=0.12",
                               fc=fc, ec="white", lw=2)
            )
            ax.text(c - 0.5, r + 0.5, f"{row}{c}", ha="center", va="center", fontsize=11,
                    color=INK, weight="bold" if is_blank else "normal")
    for c in range(1, 13):
        ax.text(c - 0.5, -0.3, str(c), ha="center", va="center", fontsize=13, weight="bold",
                color=INK)
    # Legend
    for g, color in enumerate(group_colors):
        x = 0.1 + g * 2.6
        ax.add_patch(FancyBboxPatch((x, 8.35), 0.38, 0.38, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=color, ec="none"))
        ax.text(x + 0.55, 8.54, f"Group {g + 1}", va="center", fontsize=12.5, color=INK)
    ax.add_patch(FancyBboxPatch((10.45, 8.35), 0.38, 0.38, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=np.array(to_rgb(group_colors[0])) * 0.72, ec="none"))
    ax.text(11.0, 8.54, "Blank", va="center", fontsize=12.5, color=INK)
    save(fig, "blank_group")


if __name__ == "__main__":
    sliding_window()
    spline()
    mech_models()
    phenom_models()
    threshold_demo()
    tangent_demo()
    annotations()
    data_upload()
    naming_upload()
    naming_convention()
    blank_group()
