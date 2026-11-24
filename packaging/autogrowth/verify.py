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

from growthcurve_app.dilutions import detect
from growthcurve_app.pipeline import fit_curve

app = AppTest.from_file(str(root / "AutoGrowth/app.py")).run(timeout=90)
assert not app.exception, [e.message for e in app.exception]
print("Upload page rendered with bundled example data and assets")
for page in (
    "1_growth_analysis.py",
    "3_comparative_plots.py",
    "0_downloads.py",
):
    app.switch_page(page).run(timeout=30)
    assert not app.exception, [e.message for e in app.exception]
print("All file-based pages rendered without exceptions")
t = np.linspace(0, 24, 100)
n = 0.02 * np.exp(0.2 * t)
result, stats = fit_curve(
    t,
    n,
    {
        "selected_model": "sliding_window",
        "window_points": 10,
        "lag_cutoff": 0.1,
        "exp_cutoff": 0.5,
        "phase_boundary_method": "threshold",
        "min_data_points": 5,
        "min_signal_to_noise": 1.0,
        "min_od_increase": 0.05,
        "min_growth_rate": 0.0,
    },
)
assert result is not None and stats, stats
hours = np.arange(0, 10, 1 / 60)
od = np.where(hours % 5 < 4.9, 0.2 * np.exp(0.3 * (hours % 5)), 0.2)
drops = detect(pd.Series(od, index=hours))
assert drops and abs(drops[0][0] - 4.9) < 0.1, drops
print("Turbidostat dilution detection passed")
frame = pd.DataFrame({"time": t, "od": n})
buffer = io.BytesIO()
frame.to_excel(buffer, index=False)
buffer.seek(0)
restored = pd.read_excel(buffer)
np.testing.assert_allclose(restored["od"], n)
print("Representative curve fit and Excel export/import passed")
