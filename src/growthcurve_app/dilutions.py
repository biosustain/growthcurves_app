"""Turbidostat dilutions from the OD reads alone.

A dilution is a drop ``(peak, end)``: ``peak`` is the last read before OD falls and
``end`` the first read after the fall. The reads strictly between the two are the
drop itself and are left out of every growth fit.

Detection (``detect``) is the step + hinge + snap + penalty detector benchmarked in
growthcurves_dev ``development/turbidostat_dilution_detection`` (``methods.step`` at
its dev-tuned settings: k=8, min_dur=0, persist=True, snap_sd=2, pen=16):

1. Work in ln OD of the reads (non-positive reads dropped).
2. Robust step statistic: median of the next m reads minus median of the previous m
   reads, at 2, 5 and 12 min scales, divided by the local read noise (median
   absolute read-to-read difference within +-30 min). Candidates are local maxima
   above ``k`` noise SDs.
3. Around each candidate (+-25 min), a continuous three-piece linear fit (growth,
   fall, growth), refitted once without outliers.
4. Snap: the peak moves to the last read still on the growth line, the end to the
   first read on the post-drop line (within 2 noise SDs).
5. Keep the drop only if it stays down for 10 min and the hinge beats a straight
   line by a penalised cost, gain / sd^2 > 16 ln n (after cpop's L0 criterion).

``locate`` places the drop before each logged dilution event (the event is written
when the pump has finished) with the same hinge fit and snap.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SPANS = (2, 5, 12)  # step statistic scales (min)
REACH = 25  # half-width of the hinge window around a candidate (min)
SNAP_SD = 2  # snap tolerance (noise SDs)
PEN = 16  # penalty per knot, in units of ln n
PERSIST = 10  # a drop must stay down this long (min)


def _ln(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    s = s.dropna()
    s = s[s > 0].sort_index()
    return s.index.to_numpy(float), np.log(s.to_numpy(float))


def _step_stat(y: np.ndarray, m: int) -> np.ndarray:
    """Median of the next m reads minus median of the previous m reads, at every read.
    Counted in reads, not time, so a gap (reads paused during pumping) never empties
    one side."""
    back = pd.Series(y).rolling(m, min_periods=m).median().shift(1).to_numpy()
    fwd = (
        pd.Series(y[::-1]).rolling(m, min_periods=m).median().shift(1).to_numpy()[::-1]
    )
    return fwd - back


def _noise_sd(t: np.ndarray, y: np.ndarray, window: str = "60min") -> np.ndarray:
    """Per-read noise SD of ln OD from the median absolute read-to-read difference
    within a centred window (noise in ln OD grows at low OD)."""
    dy = np.abs(np.diff(y, prepend=y[0]))
    dy[0] = dy[1] if len(dy) > 1 else 0
    r = pd.Series(dy, index=pd.to_timedelta(t, unit="h")).rolling(
        window, closed="both", center=True
    )
    return 1.4826 * r.median().to_numpy() / np.sqrt(2)


def _hinge_best(tt, y, peaks, ends):
    """Best continuous 3-piece linear fit with knots tt[i] < tt[j] (i in peaks, j in ends,
    j > i + 1) whose middle piece falls. All knot pairs are solved at once from
    sufficient statistics. Returns (sse, i, j, beta)."""
    peaks, ends = np.asarray(peaks), np.asarray(ends)
    ii, jj = np.meshgrid(peaks, ends, indexing="ij")
    ok = jj > ii + 1
    ii, jj = ii[ok], jj[ok]
    if not len(ii):
        return (np.inf, None, None, None)
    K = np.unique(np.r_[peaks, ends])
    pos = dict(zip(K.tolist(), range(len(K)), strict=True))
    a = np.array([pos[v] for v in ii.tolist()])
    c = np.array([pos[v] for v in jj.tolist()])
    H = np.maximum(tt[None, :] - tt[K, None], 0)  # hinge basis for every knot
    h1, ht, hy, HH = H.sum(1), H @ tt, H @ y, H @ H.T
    n, st, stt, sy, sty = len(tt), tt.sum(), tt @ tt, y.sum(), tt @ y
    A = np.empty((len(a), 4, 4))
    A[:, 0, 0], A[:, 0, 1], A[:, 1, 1] = n, st, stt
    A[:, 0, 2], A[:, 1, 2], A[:, 2, 2] = h1[a], ht[a], HH[a, a]
    A[:, 0, 3], A[:, 1, 3], A[:, 3, 3] = h1[c], ht[c], HH[c, c]
    A[:, 2, 3] = HH[a, c]
    lo = np.tril_indices(4, -1)
    A[:, lo[0], lo[1]] = A[:, lo[1], lo[0]]
    A[:, range(4), range(4)] += 1e-9
    rhs = np.column_stack([np.full(len(a), sy), np.full(len(a), sty), hy[a], hy[c]])
    beta = np.linalg.solve(A, rhs[..., None])[..., 0]
    sse = y @ y - (beta * rhs).sum(1)
    sse[beta[:, 1] + beta[:, 2] >= 0] = np.inf
    k = int(np.argmin(sse))
    if not np.isfinite(sse[k]):
        return (np.inf, None, None, None)
    return sse[k], int(ii[k]), int(jj[k]), beta[k]


def _hinge(t, y, coarse=40):
    """Continuous 3-piece linear fit: found on a coarse knot grid, refined at full
    resolution, then refitted once without outliers (|residual| > 4 robust SDs).
    Returns (i_peak, i_end, fit at peak, fit at end, growth line, post-drop line,
    SSE gain over a straight line, reads kept) or None."""
    n = len(t)
    tt = (t - t[0]) * 60  # minutes, for conditioning
    keep = np.ones(n, bool)
    for rnd in range(2):
        idx = np.flatnonzero(keep)
        ti, yi = tt[idx], y[idx]
        m = len(ti)
        if m < 8:
            return None
        stride = max(1, m // coarse)
        grid = np.arange(2, m - 2, stride)
        _, i, j, _ = _hinge_best(ti, yi, grid, grid)
        if i is None:
            return None
        peaks = np.arange(max(1, i - stride), min(m - 3, i + stride + 1))
        ends = np.arange(max(2, j - stride), min(m - 1, j + stride + 1))
        _, i, j, beta = _hinge_best(ti, yi, peaks, ends)
        if i is None:
            return None
        pk, en = idx[i], idx[j]
        fit = (
            beta[0]
            + beta[1] * tt
            + beta[2] * np.maximum(tt - tt[pk], 0)
            + beta[3] * np.maximum(tt - tt[en], 0)
        )
        if rnd == 0:
            r = y - fit
            mad = 1.4826 * np.median(np.abs(r - np.median(r)))
            new = np.abs(r) <= 4 * mad
            if new.all():
                break
            keep = new
    pre = beta[0] + beta[1] * tt
    post = pre + beta[2] * (tt - tt[pk]) + beta[3] * (tt - tt[en])
    kept = np.flatnonzero(keep)
    sse_hinge = float(np.sum((y[kept] - fit[kept]) ** 2))
    line = np.polyval(np.polyfit(tt[kept], y[kept], 1), tt[kept])
    gain = float(np.sum((y[kept] - line) ** 2)) - sse_hinge
    return pk, en, fit[pk], fit[en], pre, post, gain, len(kept)


def _snap(y, i, j, pre, post, tol):
    """Move the peak forward to the last read still on the growth line and the end
    back to the first read already on the post-drop line."""
    while i + 1 < j and y[i + 1] >= pre[i + 1] - tol:
        i += 1
    while j - 1 > i and y[j - 1] <= post[j - 1] + tol:
        j -= 1
    return i, j


def detect(s: pd.Series, k: float = 8) -> list[tuple[float, float]]:
    """Dilutions ``[(peak, end), ...]`` (hours) of one reactor's OD reads, indexed by
    time in hours. ``k``: detection threshold in noise SDs."""
    t, y = _ln(s)
    if len(t) < 20:
        return []
    dt = np.median(np.diff(t)) * 60
    sig = _noise_sd(t, y)
    z = np.zeros(len(t))
    for span in SPANS:
        m = max(3, int(round(span / dt)))
        st = _step_stat(y, m)
        ok = np.isfinite(st) & (sig > 0)
        zs = np.zeros(len(t))
        zs[ok] = -st[ok] / (k * 1.2533 * sig[ok] * np.sqrt(2 / m))
        z = np.maximum(z, zs)
    cand, _ = find_peaks(z, height=1, distance=max(1, int(round(max(SPANS) / dt))))
    if not len(cand):
        return []
    tc = t[cand]
    lo = np.r_[-np.inf, (tc[:-1] + tc[1:]) / 2]
    hi = np.r_[(tc[:-1] + tc[1:]) / 2, np.inf]
    drops = []
    for c, a, b in zip(tc, lo, hi, strict=True):
        w0 = np.searchsorted(t, max(a, c - REACH / 60))
        w1 = np.searchsorted(t, min(b, c + REACH / 60), side="right")
        tw, yw = t[w0:w1], y[w0:w1]
        h = _hinge(tw, yw)
        if h is None:
            continue
        i, j, top, bottom, pre, post, gain, nk = h
        sd = np.median(sig[w0:w1])
        if gain / sd**2 < PEN * np.log(nk):
            continue
        i, j = _snap(yw, i, j, pre, post, SNAP_SD * sd)
        after = yw[(tw > tw[j]) & (tw <= tw[j] + PERSIST / 60)]
        if len(after) and np.median(after) > (top + bottom) / 2:
            continue
        if drops and tw[i] < drops[-1][1]:  # same drop seen twice
            continue
        drops.append((float(tw[i]), float(tw[j])))
    return drops


def locate(
    s: pd.Series, events, before: float = REACH, after: float = 5
) -> list[tuple[float, float]]:
    """Drops ``[(peak, end), ...]`` at logged dilution events (hours). The drop is
    fitted in the ``before`` min before to ``after`` min after each event; where no
    drop can be fitted, the event itself is used (peak = end = event)."""
    t, y = _ln(s)
    if not len(t):
        return [(float(e), float(e)) for e in sorted(events)]
    sig = _noise_sd(t, y) if len(t) > 1 else np.zeros(len(t))
    drops = []
    for e in sorted(float(e) for e in events):
        w0 = np.searchsorted(t, e - before / 60)
        w1 = np.searchsorted(t, e + after / 60, side="right")
        tw, yw = t[w0:w1], y[w0:w1]
        h = _hinge(tw, yw) if len(tw) >= 8 else None
        if h is None:
            drops.append((e, e))
            continue
        i, j, _, _, pre, post, _, _ = h
        i, j = _snap(yw, i, j, pre, post, SNAP_SD * np.median(sig[w0:w1]))
        drops.append((float(tw[i]), float(tw[j])))
    return drops
