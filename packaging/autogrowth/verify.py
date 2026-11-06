"""Exercise the relocated bundle with its own interpreter; no test packages needed."""

import io
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
os.chdir(root)
sys.path[:0] = [str(root / "src"), str(root / "AutoGrowth")]
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from growthcurve_app.analyze import fit_single_series
from growthcurve_app.turbistat import detect_peaks

app = AppTest.from_file(str(root / "AutoGrowth/app.py")).run(timeout=90)
assert not app.exception, [e.message for e in app.exception]
print("Upload page rendered with bundled example data and assets")
for page in (
    "0_data_dashboard.py",
    "0_select_data.py",
    "1_batch_analysis.py",
    "2_turbiostat.py",
    "3_comparative_plots.py",
    "0_downloads.py",
):
    app.switch_page(page).run(timeout=30)
    assert not app.exception, [e.message for e in app.exception]
print("All file-based pages rendered without exceptions")
t = np.linspace(0, 24, 100)
n = 0.02 * np.exp(0.2 * t)
result, stats = fit_single_series(
    t,
    n,
    {
        "selected_model": "sliding_window",
        "n_fits": 1,
        "window_points": 10,
        "spline_smoothing_value": 0.1,
        "lag_cutoff": 0.1,
        "exp_cutoff": 0.5,
        "phase_boundary_method": "threshold",
    },
)
assert result is not None and stats, stats
peaks = detect_peaks(pd.Series([0.1, 1.0, 0.1, 1.1, 0.1]), distance=1)
assert len(peaks) == 2
print("Turbidostat peak detection passed")
frame = pd.DataFrame({"time": t, "od": n})
buffer = io.BytesIO()
frame.to_excel(buffer, index=False)
buffer.seek(0)
restored = pd.read_excel(buffer)
np.testing.assert_allclose(restored["od"], n)
print("Representative curve fit and Excel export/import passed")
