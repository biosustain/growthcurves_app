import numpy as np
import pandas as pd

from growthcurve_app import dilutions, pipeline


def test_resplit_fits_only_new_segments(monkeypatch):
    fitted = []

    def fit_curve(t, y, options):
        fitted.append((float(t[0]), float(t[-1])))
        return {}, {}

    monkeypatch.setattr(pipeline, "fit_curve", fit_curve)
    t = np.arange(0, 10.01, 0.5)
    s = pd.Series(np.linspace(0.1, 1.0, t.size), index=t)
    df = pd.DataFrame({"R1": s, "R2": s})
    curves, fits = pipeline.analyse(df, {}, {"R1": [3.0, 6.0]})
    kept = pipeline.curve_id("R1", 6.0, 10.0)
    curves[kept]["params"] = {"edited": True}
    excluded = pipeline.curve_id("R1", 0.0, 3.0)
    del curves[excluded]
    fitted.clear()

    # remove the dilution at 6 h and add one at 8 h
    new, new_fits = pipeline.resplit(s, "R1", [3.0, 6.0], [3.0, 8.0], curves, fits, {})

    assert fitted == [(3.5, 8.0), (8.5, 10.0)]
    assert list(new) == [
        pipeline.curve_id("R1", 3.0, 8.0),
        pipeline.curve_id("R1", 8.0, 10.0),
        pipeline.curve_id("R2", 0.0, 10.0),
    ]
    assert excluded not in new and kept not in new
    assert set(new_fits) == set(new)


def test_segments_skip_the_reads_inside_each_drop():
    t = np.arange(0, 10.01, 0.5)
    assert pipeline.segments(t, [(3.0, 4.0), [6.0, 6.0], 8.0]) == [
        (0.0, 3.0),
        (4.0, 6.0),
        (6.0, 8.0),
        (8.0, 10.0),
    ]
    s = pd.Series(np.ones(t.size), index=t)
    assert pipeline.curve_series(s, 4.0, 6.0).index.tolist() == [4.5, 5.0, 5.5, 6.0]
    assert pipeline.curve_series(s, 0.0, 1.0).index.tolist() == [0.0, 0.5, 1.0]
    # overlapping drops merge, drops outside the reads are ignored
    assert pipeline.clean_drops([(5, 6), (2, 1), (5.5, 7)]) == [(1, 2), (5, 7)]
    assert pipeline.segments(t, [(-1.0, 0.0), (10.0, 11.0)]) == [(0.0, 10.0)]


def _turbidostat(
    seed=0, dt_s=30, mu=0.4, target=0.6, drop_ln=0.5, pump_min=5, noise=0.01
):
    """Exponential growth to the target OD, then a linear fall in ln OD over the
    pump time. Returns the reads and the true (pump start, pump end) times."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 30, dt_s / 3600)
    y, truth, start = np.empty(t.size), [], None
    ln, ln_target = np.log(0.2), np.log(target)
    for i in range(t.size):
        if i:
            dth = t[i] - t[i - 1]
            if start is None:
                ln += mu * dth
                if ln >= ln_target:
                    start = t[i]
            else:
                ln -= drop_ln * dth * 60 / pump_min
                if t[i] - start >= pump_min / 60:
                    truth.append((start, t[i]))
                    start = None
        y[i] = ln
    od = np.exp(y) * (1 + noise * rng.standard_normal(t.size))
    return pd.Series(od, index=t), truth


def test_detect_finds_every_drop_at_its_peak():
    s, truth = _turbidostat()
    drops = dilutions.detect(s)
    assert len(drops) == len(truth) > 5
    for (p, e), (tp, te) in zip(drops, truth, strict=True):
        assert abs(p - tp) * 60 <= 1.5
        assert abs(e - te) * 60 <= 1.5


def test_locate_places_the_drop_before_each_event():
    s, truth = _turbidostat(seed=1)
    events = [te + 0.5 / 60 for _, te in truth]  # logged when the pump has finished
    drops = dilutions.locate(s, events)
    for (p, e), (tp, te) in zip(drops, truth, strict=True):
        assert abs(p - tp) * 60 <= 1.5
        assert abs(e - te) * 60 <= 1.5


def test_no_drops_in_a_batch_curve():
    t = np.arange(0, 20, 30 / 3600)
    rng = np.random.default_rng(2)
    od = 0.05 * np.exp(0.3 * t) * (1 + 0.01 * rng.standard_normal(t.size))
    assert dilutions.detect(pd.Series(od, index=t)) == []
