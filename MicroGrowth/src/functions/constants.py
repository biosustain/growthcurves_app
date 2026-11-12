"""Shared constants for MicroGrowth."""

from collections.abc import Iterable
from dataclasses import dataclass, field


# Plate layout
@dataclass(frozen=True)
class PlateFormat:
    """Row/column layout of a microplate."""

    rows: tuple[str, ...]
    cols: tuple[int, ...]
    wells: tuple[str, ...] = field(init=False, repr=False)
    well_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self):
        wells = tuple(f"{r}{c}" for r in self.rows for c in self.cols)
        object.__setattr__(self, "wells", wells)
        object.__setattr__(self, "well_set", frozenset(wells))

    @property
    def n_wells(self) -> int:
        return len(self.wells)

    def well(self, y: int, x: int) -> str:
        """Well ID at grid position (row index y, column index x)."""
        return f"{self.rows[y]}{self.cols[x]}"

    def point(self, well: str) -> tuple[int, int] | None:
        """Grid position (y, x) of a well ID, or None if it is not on this plate."""
        text = str(well).strip().upper()
        if text not in self.well_set:
            return None
        return self.rows.index(text[0]), self.cols.index(int(text[1:]))


PLATE_96 = PlateFormat(tuple("ABCDEFGH"), tuple(range(1, 13)))
PLATE_384 = PlateFormat(tuple("ABCDEFGHIJKLMNOP"), tuple(range(1, 25)))
PLATE_FORMATS = {f.n_wells: f for f in (PLATE_96, PLATE_384)}
# Every well ID accepted on upload (the largest plate contains all smaller ones)
VALID_WELLS = PLATE_384.well_set


def detect_plate_format(wells: Iterable) -> PlateFormat:
    """Smallest supported plate that holds every recognised well ID in `wells`."""
    ids = {str(w).strip().upper() for w in wells} & VALID_WELLS
    return next(f for f in PLATE_FORMATS.values() if ids <= f.well_set)


def plate_format(plate: dict | None) -> PlateFormat:
    """Plate format stored on a plate record (96-well if not recorded)."""
    return PLATE_FORMATS.get((plate or {}).get("n_wells"), PLATE_96)


# Default parameters
DEFAULT_PARAMS = {
    "time_unit": "minutes",
    "pathlength_cm_": 1.0,
    "clip_time_series": (None, None),
    "remove_wells": False,
    "blank": True,
    "outlier_detection": False,
    "outlier_threshold": 3.5,
    "window_points": 5,
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
    "smooth": "auto",
}
