"""Growth analysis for batch and turbidostat data.

Turbidostat data is batch data cut at the dilutions: each reactor is split into
curves (from the end of one drop in OD to the peak before the next) and every
curve is fitted and reviewed like a batch curve. Without dilutions each reactor is one curve.
The pipeline itself lives in ``growthcurve_app.pipeline``.

Session state
-------------
- df_rolling, start_time: input data from the Upload Data page.
- analysis_curves: {curve id: curve dict}, see ``pipeline.analyse``.
- analysis_fits: {curve id: fit result}; refitted on demand if missing.
- analysis_options: options of the last run.
- analysis_dilutions: {reactor: [(peak, end) hours]} or None for batch data.
- an_*__{curve id}: widget state of the review controls of one curve.
"""

from io import BytesIO

import check_fit_plots
import numpy as np
import pandas as pd
import streamlit as st
from buttons import create_download_button
from check_fit_plots import plot_derivative_metric, plot_well_curve
from growthcurves_options import (
    render_parameter_calculation_table_upload_style,
    render_upload_style_analysis_options,
)
from plots import plot_dilution_overview
from ui_components import page_header_with_help, show_warning_to_upload_data

from growthcurve_app import analyze, pipeline

ss = st.session_state

DILUTION_MODES = {
    "none": "None (batch culture)",
    "events": "From uploaded dilution events",
    "auto": "Detect automatically",
}
# No-growth checks per mode. Between dilutions OD only rises ~0.03-0.25 and a
# curve can have few readings, so the batch thresholds would reject most curves.
QC_DEFAULTS = {
    "batch": dict(
        min_data_points=50,
        min_signal_to_noise=1.0,
        min_od_increase=0.5,
        min_growth_rate=0.01,
        min_fold=1.5,
    ),
    "dilutions": dict(
        min_data_points=10,
        min_signal_to_noise=1.0,
        min_od_increase=0.02,
        min_growth_rate=0.01,
        min_fold=1.1,
    ),
}
QC_WIDGET_KEYS = [f"batch_{k}" for k in QC_DEFAULTS["batch"]]
# Options that can be changed per curve in the Re-analyse popover
REFIT_INPUTS = {
    "min_od_increase": (
        "Min OD increase",
        dict(min_value=0.0, step=0.01, format="%.3f"),
    ),
    "min_growth_rate": (
        "Min growth rate (1/h)",
        dict(min_value=0.0, step=0.0001, format="%.4f"),
    ),
    "min_signal_to_noise": (
        "Min signal-to-noise",
        dict(min_value=0.0, step=0.1, format="%.2f"),
    ),
    "min_data_points": ("Min data points", dict(min_value=1, step=1)),
    "window_points": ("Min. window size (points)", dict(min_value=3, step=1)),
    "smooth_mode": (
        "Spline smoothing (0 = automatic)",
        dict(min_value=0.0, step=0.1, format="%g"),
    ),
}
INT_PARAMS = {"min_data_points", "window_points"}

HELP = """
Fit growth curves to the uploaded OD data.

1. **Configure and run.** For turbidostat runs, split each reactor at its
   dilutions, taken from an uploaded events file or detected from the OD reads.
   Each dilution is a drop in OD, from its peak (last read before OD falls) to its
   end (first read after the fall). The reads in between are left out and each
   interval from one drop's end to the next drop's peak is analysed as its own
   curve. Leave at *None* for batch cultures. Then set the curve analysis
   parameters and run.
2. **Review** each curve: adjust phase boundaries, re-analyse with other
   settings, lasso-select points to refit or mark no growth.
   For turbidostat runs, edit or delete the windows.
3. **Download** the summary and the settings used.
"""


# --- state helpers -------------------------------------------------------------
def _key(name: str, cid: str) -> str:
    return f"an_{name}__{cid}"


def _keep(*keys):
    """Keep widget values when navigating between pages."""
    for k in keys:
        if k in ss:
            ss[k] = ss[k]


def _clear_curve_widgets():
    for k in [k for k in ss if str(k).startswith("an_")]:
        del ss[k]


def _on_mode_change():
    """Let the no-growth checks take the defaults of the new mode."""
    if ss.get("analysis_dilution_mode") is None:  # pill clicked off
        ss["analysis_dilution_mode"] = "none"
    for k in QC_WIDGET_KEYS:
        ss.pop(k, None)


def _mode() -> str:
    return (
        "batch" if ss.get("analysis_dilution_mode", "none") == "none" else "dilutions"
    )


def _times(reactor: str) -> np.ndarray:
    return pipeline.reactor_series(ss["df_rolling"], reactor).index.to_numpy(float)


def _drops(reactor: str) -> list[tuple[float, float]]:
    return pipeline.clean_drops(ss["analysis_dilutions"].get(reactor, []))


def _curve_data(cid: str) -> tuple[np.ndarray, np.ndarray]:
    c = ss["analysis_curves"][cid]
    s = pipeline.curve_series(
        pipeline.reactor_series(ss["df_rolling"], c["reactor"]),
        c["start"],
        c["end"],
    )
    return s.index.to_numpy(float), s.to_numpy(float)


def _params(cid: str) -> dict:
    """Options for refitting a curve: global options plus the popover values."""
    params = dict(ss["analysis_options"])
    for name in _refit_inputs():
        v = ss[_key(name, cid)]
        params[name] = analyze.normalize_smooth(v) if name == "smooth_mode" else v
    return params


def _refit_inputs() -> list[str]:
    """REFIT_INPUTS that apply to the model: the method parameter only for its own."""
    method = {"sliding_window": "window_points", "spline": "smooth_mode"}
    own = method.get(ss["analysis_options"]["selected_model"])
    return [n for n in REFIT_INPUTS if n not in method.values() or n == own]


def _reset_refit_inputs(cid: str, params: dict):
    for name in _refit_inputs():
        v = params[name]
        if name == "smooth_mode":
            v = analyze.smooth_to_number(v)
        ss[_key(name, cid)] = int(v) if name in INT_PARAMS else float(v)


def _sync_controls(cid: str, t: np.ndarray, y: np.ndarray):
    """Set the phase and max OD sliders from the curve's stats."""
    stats = ss["analysis_curves"][cid]["stats"]
    t_min, t_max = float(t.min()), float(t.max())

    def clamp(v, default):
        v = float(v) if v is not None and pd.notna(v) else default
        return min(max(v, t_min), t_max)

    ss[_key("phase", cid)] = (
        clamp(stats.get("exp_phase_start"), t_min),
        clamp(stats.get("exp_phase_end"), t_max),
    )
    y_max = float(np.nanmax(y))
    max_od = stats.get("max_od")
    max_od = float(max_od) if max_od is not None and pd.notna(max_od) else y_max
    ss[_key("maxod", cid)] = min(max_od, y_max) if y_max > 0 else 0.0


# --- callbacks -------------------------------------------------------------------
def _refit(cid, t, y, params, used_times=None):
    c = ss["analysis_curves"][cid]
    ss["analysis_fits"][cid], c["stats"] = pipeline.fit_curve(t, y, params)
    options = ss["analysis_options"]
    c["params"] = {k: v for k, v in params.items() if options.get(k) != v}
    c["used_times"] = used_times
    _sync_controls(cid, *_curve_data(cid))


def _subset(t, y, times):
    """Points of (t, y) at the given times, or None if fewer than two match."""
    idx = analyze.match_selected_times(t, np.asarray(times, dtype=float))
    return (t[idx], y[idx]) if idx.size >= 2 else None


def _fit_data(cid):
    """The curve's reads, or its lasso subset (and the selected times) if any."""
    t, y = _curve_data(cid)
    used = ss["analysis_curves"][cid]["used_times"]
    sub = _subset(t, y, used) if used else None
    return (*sub, used) if sub else (t, y, None)


def _on_reanalyse(cid):
    t, y, used = _fit_data(cid)
    _refit(cid, t, y, _params(cid), used)


def _on_defaults(cid):
    _reset_refit_inputs(cid, ss["analysis_options"])
    t, y = _curve_data(cid)
    _refit(cid, t, y, ss["analysis_options"])


def _on_lasso(cid, chart_key):
    t, y = _curve_data(cid)
    xs = analyze.get_selected_times_from_event(ss.get(chart_key))
    sub = _subset(t, y, xs)
    if sub is not None:
        _refit(cid, *sub, _params(cid), sub[0].tolist())


def _on_no_growth(cid):
    c = ss["analysis_curves"][cid]
    c["stats"] = pipeline.no_growth_stats("manually assigned")
    c["stats"]["model_name"] = ss["analysis_options"]["selected_model"]
    ss["analysis_fits"][cid] = None
    _sync_controls(cid, *_curve_data(cid))


def _on_edit(cid):
    """Write the slider values into the curve's stats."""
    lag_end, exp_end = ss[_key("phase", cid)]
    ss["analysis_curves"][cid]["stats"].update(
        exp_phase_start=float(lag_end),
        exp_phase_end=float(exp_end),
        max_od=float(ss[_key("maxod", cid)]),
    )


def _move(step: int):
    ids = list(ss["analysis_curves"])
    ss["analysis_curve"] = ids[(ids.index(ss["analysis_curve"]) + step) % len(ids)]


def _move_reactor(step: int):
    """Select the first curve of the previous or next reactor."""
    curves = ss["analysis_curves"]
    reactors = list(dict.fromkeys(v["reactor"] for v in curves.values()))
    cur = curves[ss["analysis_curve"]]["reactor"]
    nxt = reactors[(reactors.index(cur) + step) % len(reactors)]
    ss["analysis_curve"] = next(k for k, v in curves.items() if v["reactor"] == nxt)


def _apply_dilutions(reactor: str, new: list[tuple[float, float]]):
    """Set the drops of one reactor and refit only the curves that changed."""
    old = _drops(reactor)
    new = pipeline.clean_drops(new)
    if new == old:
        return
    ss["analysis_dilutions"][reactor] = new
    ss["analysis_curves"], ss["analysis_fits"] = pipeline.resplit(
        pipeline.reactor_series(ss["df_rolling"], reactor),
        reactor,
        old,
        new,
        ss["analysis_curves"],
        ss["analysis_fits"],
        ss["analysis_options"],
    )
    if ss.get("analysis_curve") not in ss["analysis_curves"]:
        own = [k for k, v in ss["analysis_curves"].items() if v["reactor"] == reactor]
        ss["analysis_curve"] = (
            own[0] if own else next(iter(ss["analysis_curves"]), None)
        )


def _windows(reactor: str) -> tuple[list[tuple[float, float]], list]:
    """Windows (start, end) of one reactor and the drops between them (drop i
    separates window i and i + 1). Drops outside the reads are left out."""
    wins = pipeline.segments(_times(reactor), _drops(reactor))
    inner = [(a[1], b[0]) for a, b in zip(wins[:-1], wins[1:], strict=True)]
    return wins, inner


def _select_at(reactor: str, time: float):
    """Select the curve of the reactor that contains the time."""
    for k, v in ss["analysis_curves"].items():
        if v["reactor"] == reactor and v["start"] <= time <= v["end"]:
            ss["analysis_curve"] = k
            return


def _after_window_edit(reactor: str, time: float):
    _select_at(reactor, time)
    ss["analysis_window_edits"] = ss.get("analysis_window_edits", 0) + 1


def _on_merge(reactor: str, key: str):
    """Remove the drop between two adjacent windows and reanalyse the merged window
    with the review settings of the windows it replaces (those of the selected
    window if it is one of them, otherwise those of the first)."""
    wins, inner = _windows(reactor)
    a = ss.get(key)
    b = ss.get(f"{key}_with_{a}")
    if a is None or b is None or abs(a - b) != 1:
        return
    i = min(a, b)  # the drop between windows i and i + 1
    curves = ss["analysis_curves"]
    old = [pipeline.curve_id(reactor, *w) for w in wins[i : i + 2]]
    sel = ss.get("analysis_curve")
    src = sel if sel in old else next((k for k in old if k in curves), None)
    params = dict(curves[src]["params"]) if src else {}
    drops = _drops(reactor)
    _apply_dilutions(reactor, [d for d in drops if d != inner[i]])
    cid = pipeline.curve_id(reactor, wins[i][0], wins[i + 1][1])
    if cid in ss["analysis_curves"]:
        params = {**ss["analysis_options"], **params}
        _refit(cid, *_curve_data(cid), params)
        _reset_refit_inputs(cid, params)
    _after_window_edit(reactor, inner[i][0])


def _on_split(reactor: str, key: str):
    """Add a window by cutting the window that contains the time."""
    x = float(ss[key])
    wins, _ = _windows(reactor)
    if not any(a < x < b for a, b in wins):
        st.toast(f"{x:.3f} h is not inside a window: nothing added.")
        return
    drops = _drops(reactor)
    _apply_dilutions(reactor, [*drops, (x, x)])
    _after_window_edit(reactor, x)


def _on_set_window(reactor: str, cid: str, skey: str, ekey: str):
    """Move the start and end of one window by editing the drops next to it.
    Shrinking a window leaves the freed reads out of every fit; extending it takes
    reads from the drop next to it (a cut moves with it). At the first (last) read
    of the run, the reads before the start (after the end) are left out."""
    c = ss["analysis_curves"][cid]
    s0, e0 = c["start"], c["end"]
    xs, xe = float(ss[skey]), float(ss[ekey])
    t = _times(reactor)
    t0, t1 = float(t[0]), float(t[-1])
    if not xs < xe:
        st.toast("The start must be before the end: window not changed.")
        return
    new, before, after = [], False, False
    for p, e in _drops(reactor):
        if e == s0 and not before:  # drop (or cut) before the window
            before = True
            if p == e and xs < e:
                new.append((xs, xs))
            elif xs > p:
                new.append((p, xs))
            elif p > t0:
                new.append((p, p))  # start before the previous peak: overlap
        elif p == e0 and not after:  # drop (or cut) after the window
            after = True
            if p == e and xe > p:
                new.append((xe, xe))
            elif xe < e:
                new.append((xe, e))
            elif e < t1:
                new.append((e, e))  # end after the next drop's end: overlap
        else:
            new.append((p, e))
    if not before and s0 <= t0 and xs > t0:
        new.append((t0, xs))
    if not after and e0 >= t1 and xe < t1:
        new.append((xe, t1))
    if (xs, xe) not in pipeline.segments(t, new):
        st.toast(
            "The window would overlap the window or drop next to it: not changed. "
            "Merge the windows first."
        )
        return
    _apply_dilutions(reactor, new)
    _after_window_edit(reactor, (xs + xe) / 2)


def _on_delete(reactor: str, cid: str):
    """Delete the window: it joins the drops on either side into one drop (or trims
    the start or end of the run), so all its reads are left out of every fit. The
    windows next to it keep their fits."""
    c = ss["analysis_curves"][cid]
    s0, e0 = c["start"], c["end"]
    wins, _ = _windows(reactor)
    if len(wins) < 2:
        st.toast("The only window of a reactor cannot be deleted.")
        return
    t = _times(reactor)
    lo, hi, new = float(t[0]), float(t[-1]), []
    for p, e in _drops(reactor):
        if e == s0:
            lo = p
        elif p == e0:
            hi = e
        else:
            new.append((p, e))
    _apply_dilutions(reactor, [*new, (lo, hi)])
    nxt = [w for w in wins if w[0] >= e0]
    _after_window_edit(reactor, nxt[0][0] if nxt else wins[-2][1])


def _window_controls(reactor: str, cid: str):
    """Set, merge or add windows, above the whole-run plot."""
    wins, inner = _windows(reactor)
    c = ss["analysis_curves"][cid]
    n = ss.get("analysis_window_edits", 0)
    with st.container(border=True, key="card_analysis_windows"):
        t = _times(reactor)
        t0, t1 = float(t[0]), float(t[-1])
        start_col, end_col, set_col = st.columns([3, 3, 2], vertical_alignment="bottom")
        wkey = f"analysis_window_{reactor}_{cid}_{n}"
        for col, name, value in [
            (start_col, "start", c["start"]),
            (end_col, "end", c["end"]),
        ]:
            col.number_input(
                f"Selected window {name} (h)",
                min_value=t0,
                max_value=t1,
                value=float(value),
                step=0.1,
                format="%.3f",
                key=f"{wkey}_{name}",
            )
        set_col.button(
            "Set start and end",
            key=f"{wkey}_btn",
            width="stretch",
            on_click=_on_set_window,
            args=(reactor, cid, f"{wkey}_start", f"{wkey}_end"),
            help=(
                "Move the drop next to each edge of the selected window. At the edges "
                "of the run, the reads outside the window are left out. Only the "
                "changed windows are refitted."
            ),
        )
        merge_col, add_col = st.columns(2, gap="large")
        with merge_col:
            first_col, second_col, btn_col = st.columns(
                [2, 2, 1], vertical_alignment="bottom"
            )
            mkey = f"analysis_merge_{reactor}_{cid}_{n}"
            can_merge = len(wins) > 1

            def _win_label(i):
                return f"Window {i + 1}"

            # Default: the selected window
            sel = next(
                (
                    i
                    for i, w in enumerate(wins)
                    if abs(w[0] - c["start"]) < 1e-9 and abs(w[1] - c["end"]) < 1e-9
                ),
                0,
            )
            first = first_col.selectbox(
                "Merge window",
                range(len(wins)),
                index=sel if can_merge else None,
                format_func=_win_label,
                key=mkey,
                disabled=not can_merge,
                placeholder="Only one window",
            )
            # Options: the windows adjacent to the first (after it by default)
            nbrs = (
                [j for j in (first + 1, first - 1) if 0 <= j < len(wins)]
                if can_merge and first is not None
                else []
            )
            second_col.selectbox(
                "with window",
                nbrs,
                index=0 if nbrs else None,
                format_func=_win_label,
                key=f"{mkey}_with_{first}",
                disabled=not nbrs,
                placeholder="No adjacent window",
            )
            btn_col.button(
                "Merge",
                key=f"{mkey}_btn",
                width="stretch",
                disabled=not nbrs,
                on_click=_on_merge,
                args=(reactor, mkey),
                help=(
                    "Remove the drop between the two windows and fit them as one. "
                    "Only the merged window is refitted."
                ),
            )
        with add_col:
            num_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
            akey = f"analysis_split_{reactor}_{cid}_{n}"
            num_col.number_input(
                "Add window: split at (h)",
                min_value=t0,
                max_value=t1,
                value=float((c["start"] + c["end"]) / 2),
                step=0.1,
                format="%.3f",
                key=akey,
            )
            btn_col.button(
                "Add",
                key=f"{akey}_btn",
                width="stretch",
                on_click=_on_split,
                args=(reactor, akey),
                help=(
                    "Cut the window containing this time in two. Only the two new "
                    "windows are refitted."
                ),
            )


# --- dilutions -----------------------------------------------------------------------
def _dilutions(mode: str, df: pd.DataFrame) -> dict | None:
    """Drops per reactor for the chosen mode, or None for batch data."""
    if mode == "none":
        return None
    if mode == "auto":
        return pipeline.detect_dilutions(df, k=float(ss["analysis_detection_k"]))
    events_bytes = ss.get("turbidostat_meta_upload_bytes")
    if events_bytes is None:
        st.error(
            "No dilution events file uploaded. Upload one on the Upload Data page."
        )
        st.stop()
    try:
        cuts = pipeline.dilutions_from_events(
            df,
            pd.read_csv(BytesIO(events_bytes)),
            ss["start_time"],
            timestamp_col=ss.get("turbidostat_timestamp_col", "timestamp_localtime"),
            reactor_col=ss.get("turbidostat_reactor_col", "pioreactor_unit"),
        )
    except (KeyError, ValueError) as e:
        st.error(
            f"Could not read the dilution events ({e}). Check the timestamp and "
            "reactor columns selected on the Upload Data page."
        )
        st.stop()
    missing = [str(r) for r in df.columns if str(r) not in cuts]
    if missing:
        st.warning(
            f"No dilution events for {', '.join(missing)}: analysed as single curves."
        )
    return cuts


# --- review ----------------------------------------------------------------------------
@st.fragment
def review_and_results(df: pd.DataFrame):
    """Steps 2 and 3. A fragment, so reviewing a curve does not rerun the page."""
    curves, fits = ss["analysis_curves"], ss["analysis_fits"]
    options, dilutions = ss["analysis_options"], ss["analysis_dilutions"]
    if not curves:
        st.warning("No curves to review.")
        return
    if ss.get("analysis_curve") not in curves:
        ss["analysis_curve"] = next(iter(curves))

    # Reactor name, plus the window number (as on the whole-run plot) when split
    labels = {k: v["reactor"] for k, v in curves.items()}
    if dilutions is not None:
        for reactor in set(labels.values()):
            for i, w in enumerate(_windows(reactor)[0], start=1):
                k = pipeline.curve_id(reactor, *w)
                if k in labels:
                    labels[k] = f"{reactor}: Window {i}"
    label = labels.get

    with st.container(border=True, key="card_analysis_2"):
        st.header("Step 2. Review Fits")
        cid = ss["analysis_curve"]
        c = curves[cid]
        t, y = _curve_data(cid)
        if cid not in fits and not analyze.is_bad_fit(c["stats"]):
            # e.g. after restoring a saved session: fits are not saved
            fits[cid], _ = pipeline.fit_curve(
                *_fit_data(cid)[:2], {**options, **c["params"]}
            )
        if _key("phase", cid) not in ss:
            _sync_controls(cid, t, y)
            _reset_refit_inputs(cid, {**options, **c["params"]})

        if dilutions is not None:
            reactor = c["reactor"]
            st.subheader(f"Whole bioreactor run: {reactor}")
            n_reactors = len({v["reactor"] for v in curves.values()})
            left_col, controls_col, right_col = st.columns([1, 14, 1], gap="small")
            left_col.button(
                "",
                icon=":material/chevron_left:",
                width="stretch",
                on_click=_move_reactor,
                args=(-1,),
                key="analysis_prev_reactor",
                type="primary",
                disabled=n_reactors < 2,
                help="Previous bioreactor",
            )
            with controls_col:
                _window_controls(reactor, cid)
            right_col.button(
                "",
                icon=":material/chevron_right:",
                width="stretch",
                on_click=_move_reactor,
                args=(1,),
                key="analysis_next_reactor",
                type="primary",
                disabled=n_reactors < 2,
                help="Next bioreactor",
            )
            st.plotly_chart(
                plot_dilution_overview(
                    pipeline.reactor_series(df, reactor),
                    [
                        (v, fits.get(k))
                        for k, v in curves.items()
                        if v["reactor"] == reactor
                    ],
                    c,
                    _drops(reactor),
                ),
                width="stretch",
            )
            st.divider()
            st.subheader(f"Fit of {label(cid)}")

        # Curve arrows at the far left and right of the card, as for the reactors
        prev_col, card_col, next_col = st.columns([1, 14, 1], gap="small")
        prev_col.button(
            "",
            icon=":material/chevron_left:",
            width="stretch",
            on_click=_move,
            args=(-1,),
            key="analysis_prev",
            shortcut="Left",
            type="primary",
        )
        next_col.button(
            "",
            icon=":material/chevron_right:",
            width="stretch",
            on_click=_move,
            args=(1,),
            key="analysis_next",
            shortcut="Right",
            type="primary",
        )
        with card_col, st.container(border=True, key="card_analysis_curve"):
            control_col, phase_col = st.columns(2, gap="large")
            with control_col:
                st.selectbox(
                    "Curve", list(curves), format_func=label, key="analysis_curve"
                )
                popover_col, log_col = st.columns(
                    2, vertical_alignment="center", gap="small"
                )
                with popover_col, st.popover("Annotations", width="stretch"):
                    show = {
                        name: st.toggle(
                            text, value=default, key=f"analysis_show_{name}"
                        )
                        for name, text, default in [
                            ("phase_boundaries", "Phase boundaries", True),
                            ("umax_point", "Max growth rate point", True),
                            ("max_od", "Max OD", True),
                            ("baseline_od", "Baseline OD", True),
                            ("tangent", "Tangent line at max growth", False),
                            ("fitted_model", "Fitted model curve", True),
                        ]
                    }
                log_scale = log_col.toggle("Log scale", key="analysis_log_scale")
            with phase_col:
                _review_controls(cid, t, y)

        stats = c["stats"]
        status_col, tables_col = st.columns([2, 5])
        with status_col:
            if analyze.is_bad_fit(stats):
                reason = stats.get("no_growth_reason", "No growth detected")
                st.container(border=True).error(f"**No Growth:** {reason}")
            else:
                st.container(border=True).success("**Growth Detected**")
        with tables_col:
            stats_col, params_col = st.columns(2)
            with stats_col.popover(
                f"Growth Statistics - {label(cid)}", width="stretch"
            ):
                st.dataframe(
                    analyze.format_growth_stats_table(stats),
                    width="stretch",
                    hide_index=True,
                )
            with params_col.popover(
                f"Analysis Parameters - {label(cid)}", width="stretch"
            ):
                used = {**c["params"], "smooth": c["params"].get("smooth_mode")}
                st.dataframe(
                    analyze.format_analysis_params_table(
                        stats,
                        options,
                        used,
                        n_total=t.size,
                        n_selected=len(c["used_times"]) if c["used_times"] else t.size,
                    ),
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "💡 **Tip:** Click and drag on the growth curve plot below to select a "
                "subset of data points. The analysis will be automatically rerun using "
                "only the selected points to recalculate growth parameters."
            )

        st.divider()
        _plots(cid, label(cid), t, y, log_scale, show)

    _results(curves, options, dilutions)


def _review_controls(cid: str, t: np.ndarray, y: np.ndarray):
    """Phase and max OD sliders, and No Growth, Re-analyse and Delete window."""
    t_min, t_max = float(t.min()), float(t.max())
    y_max = float(np.nanmax(y))
    slider_col1, slider_col2 = st.columns(2)
    slider_col1.slider(
        "Set phase boundaries (hours)",
        t_min,
        t_max,
        step=float(max((t_max - t_min) / 200.0, 0.01)),
        key=_key("phase", cid),
        on_change=_on_edit,
        args=(cid,),
    )
    if y_max <= 0:
        slider_col2.warning("All OD values are ≤ 0 - no growth detected")
    else:
        slider_col2.slider(
            "Set maximum OD",
            0.0,
            y_max,
            step=float(max(y_max / 120, 1e-6)),
            key=_key("maxod", cid),
            on_change=_on_edit,
            args=(cid,),
        )

    col1, col2, col3 = st.columns(3)
    col1.button(
        "No Growth",
        width="stretch",
        type="primary",
        key=_key("nogrowth", cid),
        on_click=_on_no_growth,
        args=(cid,),
    )
    with col2.popover("Re-analyse", width="stretch"):
        st.markdown("**No-growth thresholds**")
        for name in _refit_inputs():
            text, kwargs = REFIT_INPUTS[name]
            st.number_input(text, key=_key(name, cid), **kwargs)
        btn_col, defaults_col = st.columns(2)
        btn_col.button(
            "Re-analyse",
            type="primary",
            width="stretch",
            key=_key("reanalyse", cid),
            on_click=_on_reanalyse,
            args=(cid,),
        )
        defaults_col.button(
            "Defaults",
            type="primary",
            width="stretch",
            key=_key("defaults", cid),
            on_click=_on_defaults,
            args=(cid,),
        )
    if ss["analysis_dilutions"] is not None:
        reactor = ss["analysis_curves"][cid]["reactor"]
        col3.button(
            "Delete window",
            width="stretch",
            type="tertiary",
            key=_key("delete", cid),
            disabled=len(_windows(reactor)[0]) < 2,
            on_click=_on_delete,
            args=(reactor, cid),
            help=(
                "Leave all reads of the selected window out of every fit. The other "
                "windows are not refitted."
            ),
        )


def _plots(cid, well, t, y, log_scale, show):
    """The three plots of the MicroGrowth Check Growth Fits page."""
    c = ss["analysis_curves"][cid]
    plate = check_fit_plots.as_plate(
        well,
        t,
        y,
        stats=c["stats"],
        fit_result=ss["analysis_fits"].get(cid),
        window_points={**ss["analysis_options"], **c["params"]}["window_points"],
        used_fit_times=c["used_times"],
    )
    gs = plate["growth_stats"][well]
    chart_key = _key("lasso", cid)
    fig_main = plot_well_curve(
        plate,
        well,
        log_scale=log_scale,
        phase_boundaries=show["phase_boundaries"],
        umax_point=show["umax_point"],
        max_od=show["max_od"],
        baseline_od=show["baseline_od"],
        tangent=show["tangent"],
        fitted_model=show["fitted_model"],
    ).update_layout(uirevision="keep", dragmode="lasso")
    if not log_scale:
        # create_base_plot pins the linear y-axis at 0, squashing curves that sit
        # well above 0: fit it to the reads instead
        pad = 0.05 * (y.max() - y.min()) or 0.05 * abs(y.max()) or 0.05
        fig_main.update_yaxes(range=[y.min() - pad, y.max() + pad])

    with st.container(border=True, key="card_analysis_main_plot"):
        st.plotly_chart(
            fig_main,
            key=chart_key,
            selection_mode="lasso",
            on_select=lambda: _on_lasso(cid, chart_key),
            width="stretch",
        )

    for metric in ("dndt", "mu"):
        with st.container(border=True, key=f"card_analysis_{metric}"):
            st.plotly_chart(
                plot_derivative_metric(plate, well, metric=metric, gs=gs),
                width="stretch",
            )


def _results(curves, options, dilutions):
    """Step 3: summary table and downloads."""
    summary = pipeline.summary_table(curves)
    with st.container(border=True, key="card_analysis_3"):
        st.header("Step 3. Summary and Downloads")
        st.write(f"Times are hours since the start at {ss.get('start_time')}.")
        st.dataframe(summary, width="stretch")
        downloads = [
            ("Download summary statistics", summary, "growth_analysis_summary.csv"),
            (
                "Download analysis parameters",
                pipeline.params_table(curves, options).set_index("curve"),
                "growth_analysis_parameters.csv",
            ),
        ]
        if dilutions is not None:
            cuts = pd.DataFrame(
                [
                    (r, p, e)
                    for r, ds in dilutions.items()
                    for p, e in pipeline.clean_drops(ds)
                ],
                columns=["reactor", "peak_h", "end_h"],
            ).set_index("reactor")
            downloads.append(("Download dilutions", cuts, "dilutions.csv"))
        for col, (text, table, name) in zip(
            st.columns(len(downloads)), downloads, strict=True
        ):
            with col:
                create_download_button(
                    label=text,
                    data=table.to_csv(index=True).encode("utf-8"),
                    file_name=name,
                    disabled=False,
                    mime="text/csv",
                )


# --- page --------------------------------------------------------------------------------
page_header_with_help("Growth Analysis", HELP)

df_rolling = ss.get("df_rolling")
if df_rolling is None:
    show_warning_to_upload_data()
    st.stop()

_keep("analysis_dilution_mode", "analysis_detection_k")

with st.container(border=True, key="card_analysis_1"):
    st.header("Step 1. Configure and Run Analysis")
    st.subheader("Set dilution detection")
    has_events = ss.get("turbidostat_meta_upload_bytes") is not None
    ss.setdefault("analysis_dilution_mode", "none")
    mode = st.pills(
        "Split each reactor at its dilutions",
        list(DILUTION_MODES),
        format_func=lambda m: DILUTION_MODES[m]
        + ("" if m != "events" or has_events else " (no file uploaded)"),
        selection_mode="single",
        width="stretch",
        key="analysis_dilution_mode",
        on_change=_on_mode_change,
        help=(
            "For turbidostat runs. Each interval between two dilutions is analysed "
            "as its own curve, without the readings where OD falls. Leave at None "
            "for batch cultures."
        ),
    )
    if mode == "events":
        name = ss.get("turbidostat_meta_upload_name")
        if has_events:
            st.caption(f"Dilution events file: `{name}`")
        else:
            st.page_link(
                "0_upload_data.py",
                label="Upload a dilution events file",
                icon=":material/upload:",
            )
    elif mode == "auto":
        ss.setdefault("analysis_detection_k", 8.0)
        st.number_input(
            "Detection threshold (noise SDs)",
            min_value=1.0,
            step=1.0,
            key="analysis_detection_k",
            help=(
                "A drop is detected where the median OD falls by more than this many "
                "read-noise SDs (on the log scale), and kept if a peak-fall-growth fit "
                "beats a straight line. Its peak and end are then placed on the reads. "
                "Lower finds smaller drops, higher ignores them. 8 was the best "
                "setting on simulated turbidostat runs."
            ),
        )

    st.divider()
    st.subheader("Set curve analysis parameters")
    saved = ss.setdefault("analysis_saved_options", {})
    defaults = {**QC_DEFAULTS[_mode()], **saved.get(_mode(), {})}
    options = render_upload_style_analysis_options(
        min_window_points=3,
        max_window_points=500,
        default_window_points=defaults.get("window_points", 5),
        min_data_points_default=defaults["min_data_points"],
        min_signal_to_noise_default=defaults["min_signal_to_noise"],
        min_od_increase_default=defaults["min_od_increase"],
        min_growth_rate_default=defaults["min_growth_rate"],
        min_fold_default=defaults["min_fold"],
        default_smooth=analyze.smooth_to_number(defaults.get("smooth_mode", "auto")),
    )
    saved[_mode()] = {
        k: options[k] for k in [*QC_DEFAULTS["batch"], "window_points", "smooth_mode"]
    }
    render_parameter_calculation_table_upload_style(options)
    run = st.button("Run Analysis", type="primary", width="stretch")

if run:
    dilutions = _dilutions(mode, df_rolling)
    with st.spinner("Fitting curves...", show_time=True):
        curves, fits = pipeline.analyse(df_rolling, options, dilutions)
    _clear_curve_widgets()
    ss.update(
        analysis_curves=curves,
        analysis_fits=fits,
        analysis_options=options,
        analysis_dilutions=dilutions,
        analysis_curve=None,
    )

if ss.get("analysis_curves") is None:
    st.stop()
ss.setdefault("analysis_fits", {})
review_and_results(df_rolling)
