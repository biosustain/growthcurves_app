"""Shared constants for TheGrowthAnalysisApp."""

# Plate layout
ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))
ALL_WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

# Default parameters
DEFAULT_PARAMS = {
    "time_unit": "minutes",
    "pathlength_cm_": 1.0,
    "clip_time_series": (None, None),
    "remove_wells": False,
    "blank": True,
    "outlier_detection": False,
    "outlier_threshold": 3.5,
    "window_points": 10,
    "lag_cutoff": 0.5,
    "exp_cutoff": 0.5,
    "sg_window": 15,
    "sg_poly": 2,
    "min_data_points": 5,
    "min_signal_to_noise": 1.0,
    "min_od_increase": 0.05,
    "min_growth_rate": 0.001,
    "growth_method": "Spline",
    "model_family": "phenomenological",
    "model_type": "phenom_logistic",
    "phase_boundary_method": "tangent",
    "smooth": "fast",
}
