"""Growth analysis pipeline for batch and turbidostat data.

A *curve* is the readings of one reactor between two dilutions: from the end of
one drop to the peak of the next, so the reads where OD falls are left out.
Without dilutions (batch culture) each reactor is a single curve, so batch
analysis is the special case of no drops:

    drops (optional) -> segments -> fit_curve per segment -> summary

Curves are stored as plain dicts keyed by a curve id (``"reactor | segment"``),
so results survive the session-state export (JSON). No Streamlit code here.
"""

from __future__ import annotations

import time

import growthcurves as gc
import numpy as np
import pandas as pd

from growthcurve_app import dilutions as dil
from growthcurve_app.analyze import build_fit_kwargs

SEP = " | "


# --- dilutions ----------------------------------------------------------------
# A dilution is a drop (peak, end) in hours: the last read before OD falls and the
# first read after the fall. See ``growthcurve_app.dilutions``.
def dilutions_from_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    start_time: pd.Timestamp,
    *,
    timestamp_col: str,
    reactor_col: str,
) -> dict[str, list[tuple[float, float]]]:
    """Drops per reactor at the logged dilution events.

    Rows with an ``event_name`` other than ``DilutionEvent`` are ignored. The event
    is written when the pump has finished, so each drop is fitted in the reads just
    before its event (``dilutions.locate``).
    """
    if "event_name" in events:
        events = events[events["event_name"] == "DilutionEvent"]
    hours = (
        pd.to_datetime(events[timestamp_col]) - start_time
    ).dt.total_seconds() / 3600
    out = {}
    for reactor, h in hours.groupby(events[reactor_col]):
        col = next((c for c in df.columns if str(c) == str(reactor)), None)
        s = df[col] if col is not None else pd.Series(dtype=float)
        out[str(reactor)] = dil.locate(s, h.tolist())
    return out


def detect_dilutions(
    df: pd.DataFrame, *, k: float = 8
) -> dict[str, list[tuple[float, float]]]:
    """Drops per reactor detected from the reads (``dilutions.detect``); ``k`` is
    the detection threshold in noise SDs."""
    return {str(reactor): dil.detect(df[reactor], k=k) for reactor in df.columns}


def clean_drops(drops) -> list[tuple[float, float]]:
    """Drops as sorted (peak, end) pairs with peak <= end; overlapping drops merged.
    A single time (as stored by older sessions) is a cut there (peak = end)."""
    pairs = sorted(
        (
            (float(d), float(d))
            if np.isscalar(d)
            else (min(map(float, d)), max(map(float, d)))
        )
        for d in drops
    )
    out = []
    for p, e in pairs:
        if out and p <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((p, e))
    return out


# --- segments -------------------------------------------------------------------
def segments(t: np.ndarray, drops=()) -> list[tuple[float, float]]:
    """(start, end) of each segment of time points t: from the start (or a drop's
    end) to the next drop's peak (or the last time point). Drops outside the
    reads are ignored. A drop over the first (last) read trims the start (end) of
    the run: its reads belong to no segment."""
    t0, t1 = float(t[0]), float(t[-1])
    bounds, last = [t0], t1
    for p, e in clean_drops(drops):
        if e <= t0 or p >= t1 or (p <= t0 and e >= t1):
            continue
        if p <= t0:
            bounds[0] = e
        elif e >= t1:
            last = p
            break
        else:
            bounds += [p, e]
    bounds.append(last)
    return list(zip(bounds[::2], bounds[1::2], strict=True))


def reactor_series(df: pd.DataFrame, reactor: str) -> pd.Series:
    """One reactor's readings."""
    return df[reactor].dropna()


def curve_series(s: pd.Series, start: float, end: float) -> pd.Series:
    """Readings of one segment. Half-open, (start, end]: the reading at a cut
    ends the previous segment. The first segment includes its start. The reads
    inside a drop (peak, end] belong to no segment."""
    t = s.index
    lower = t >= start if start <= t[0] else t > start
    return s[lower & (t <= end)]


def curve_id(reactor: str, start: float, end: float) -> str:
    return f"{reactor}{SEP}{start:.2f}-{end:.2f}"


# --- fitting --------------------------------------------------------------------
def fit_curve(t: np.ndarray, y: np.ndarray, options: dict) -> tuple[dict | None, dict]:
    """Fit one curve and apply the no-growth checks.

    Returns ``(fit_result, stats)``; ``fit_result`` is None for no growth.
    """
    t_start = time.time()
    model = options["selected_model"]
    fit, stats = gc.fit_model(
        t=t,
        N=y,
        model_name=model,
        lag_threshold=options["lag_cutoff"],
        exp_threshold=options["exp_cutoff"],
        phase_boundary_method=options["phase_boundary_method"],
        **build_fit_kwargs(
            model_name=model,
            window_points=options["window_points"],
            smooth_mode=options.get("smooth_mode", "auto"),
            min_fold=options.get("min_fold", 1.5),
        ),
    )
    check = gc.inference.detect_no_growth(
        t=t,
        N=y,
        growth_stats=stats,
        min_data_points=options["min_data_points"],
        min_signal_to_noise=options["min_signal_to_noise"],
        min_od_increase=options["min_od_increase"],
        min_growth_rate=options["min_growth_rate"],
    )
    if check["is_no_growth"]:
        fit, stats = None, no_growth_stats(check.get("reason", "No growth detected"))
    stats["elapsed_time"] = time.time() - t_start
    stats["model_name"] = model
    return fit, stats


def no_growth_stats(reason: str) -> dict:
    stats = gc.inference.bad_fit_stats()
    stats["no_growth_reason"] = reason
    return stats


def analyse(
    df: pd.DataFrame, options: dict, dilutions: dict | None = None
) -> tuple[dict[str, dict], dict[str, dict | None]]:
    """Fit every curve of every reactor.

    Returns ``(curves, fits)``, both keyed by curve id. A curve is
    ``{"reactor", "start", "end", "stats", "params", "used_times"}``; ``params``
    holds per-curve option overrides and ``used_times`` a lasso subset (or None).
    """
    curves, fits = {}, {}
    for reactor in df.columns:
        s = reactor_series(df, reactor)
        if s.size < 2:
            continue
        drops = (dilutions or {}).get(str(reactor), ())
        for start, end in segments(s.index.to_numpy(float), drops):
            _fit_segment(s, reactor, start, end, options, curves, fits)
    return curves, fits


def _fit_segment(s, reactor, start, end, options, curves, fits):
    """Fit one segment of s into curves and fits (skipped if under two reads)."""
    c = curve_series(s, start, end)
    if c.size < 2:
        return
    cid = curve_id(reactor, start, end)
    fits[cid], stats = fit_curve(c.index.to_numpy(float), c.to_numpy(float), options)
    curves[cid] = {
        "reactor": str(reactor),
        "start": float(start),
        "end": float(end),
        "stats": stats,
        "params": {},
        "used_times": None,
    }


# --- editing dilutions ----------------------------------------------------------
def resplit(
    s: pd.Series,
    reactor: str,
    old_drops,
    new_drops,
    curves: dict[str, dict],
    fits: dict[str, dict | None],
    options: dict,
) -> tuple[dict[str, dict], dict[str, dict | None]]:
    """Re-split one reactor after its drops changed from old_drops to new_drops.

    Unchanged segments keep their curve and review edits (or stay excluded);
    only new segments are fitted. Returns new ``(curves, fits)``.
    """
    reactor = str(reactor)
    t = s.index.to_numpy(float)
    old = {curve_id(reactor, a, b) for a, b in segments(t, old_drops)}
    block, fits = {}, dict(fits)
    for start, end in segments(t, new_drops):
        cid = curve_id(reactor, start, end)
        if cid not in old:
            _fit_segment(s, reactor, start, end, options, block, fits)
        elif cid in curves:
            block[cid] = curves[cid]
    # Put this reactor's curves where its old ones were
    out, placed = {}, False
    for cid, c in curves.items():
        if c["reactor"] != reactor:
            out[cid] = c
        elif not placed:
            out.update(block)
            placed = True
    if not placed:
        out.update(block)
    return out, {k: v for k, v in fits.items() if k in out}


# --- outputs --------------------------------------------------------------------
def summary_table(curves: dict[str, dict]) -> pd.DataFrame:
    """One row per curve, indexed by (reactor, segment)."""
    if not curves:
        return pd.DataFrame()
    index = pd.MultiIndex.from_tuples(
        [tuple(cid.split(SEP, 1)) for cid in curves], names=["reactor", "segment"]
    )
    rows = [
        {"segment_start": c["start"], "segment_end": c["end"], **c["stats"]}
        for c in curves.values()
    ]
    return pd.DataFrame.from_records(rows, index=index)


def params_table(curves: dict[str, dict], options: dict) -> pd.DataFrame:
    """Analysis options used per curve (global options plus overrides)."""
    keys = [
        "selected_model",
        "phase_boundary_method",
        "min_fold",
        "min_od_increase",
        "min_growth_rate",
        "min_signal_to_noise",
        "min_data_points",
        "window_points",
        "smooth_mode",
    ]
    rows = []
    for cid, c in curves.items():
        used = {**options, **c["params"]}
        rows.append(
            {
                "curve": cid,
                **{k: used.get(k) for k in keys},
                "lasso_points": len(c["used_times"]) if c["used_times"] else None,
                "fit_t_min": c["stats"].get("fit_t_min"),
                "fit_t_max": c["stats"].get("fit_t_max"),
            }
        )
    return pd.DataFrame(rows)
