"""Data loading, preprocessing, and growth-curve fitting utilities."""

import io

import numpy as np
import pandas as pd
from growthcurves.preprocessing import blank_subtraction, detect_outliers, path_correct

from .constants import COLS, ROWS
from .fitting_pipeline import fit_growth_series

DEFAULT_BLANK_GROUP = "Group 1"


# ---------- I/O + shaping ----------
def _plate_name_map(plate_bytes):
    """Return (plate_df, well_name_map) from a plate map Excel file."""
    plate = pd.read_excel(io.BytesIO(plate_bytes)).fillna("False").set_index("rows")
    return plate, {f"{r}{c}": str(plate.loc[r, c]).strip() for r in ROWS for c in COLS}


def _read_table(
    data_bytes: bytes, time_unit: str = "hours", filter_to_wells: bool = True
) -> pd.DataFrame:
    """Read plate time series data (rows=timepoints, cols=wells) with Time in hours.

    Args:
        data_bytes: Excel file bytes
        time_unit: Unit of time in the data file ("seconds", "minutes", "hours", "days", or "HH:MM:SS")
        filter_to_wells: If True, keep only valid well columns (A1–H12) and uppercase them.
                         If False, keep all non-Time columns with their original names.

    Returns:
        DataFrame with Time column in hours and sample columns

    Raises:
        ValueError: If no Time column is found in the data file
    """
    df = pd.read_excel(io.BytesIO(data_bytes), header=0)
    df = df.replace(",", ".", regex=True)

    if "Time" not in df.columns:
        raise ValueError(
            "Data file must contain a 'Time' column. "
            "Please add a Time column with numeric values."
        )

    t_raw = df["Time"]
    df = df.drop(columns=["Time"])

    if time_unit == "HH:MM:SS":

        def _hhmmss_to_hours(val):
            parts = str(val).strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) + int(parts[1]) / 60.0 + float(parts[2]) / 3600.0
            return float("nan")

        t_hours = t_raw.map(_hhmmss_to_hours)
    else:
        t = pd.to_numeric(t_raw, errors="coerce")
        if time_unit == "seconds":
            t_hours = t / 3600.0
        elif time_unit == "minutes":
            t_hours = t / 60.0
        elif time_unit == "days":
            t_hours = t * 24.0
        else:
            t_hours = t

    if filter_to_wells:
        valid_wells = {f"{r}{c}" for r in ROWS for c in COLS}
        well_cols = [c for c in df.columns if str(c).strip().upper() in valid_wells]
        df = df[well_cols].copy()
        df.columns = [str(c).strip().upper() for c in df.columns]

    df.insert(0, "Time", t_hours)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _empty_plate():
    """Create an empty plate record structure."""
    return {
        "name": {},
        "raw_data": {},
        "processed_data": {},
        "growth_stats": {},
        "fit_parameters": {},
    }


def load_plate(
    plates: dict,
    plate_id: str,
    *,
    data_bytes: bytes,
    plate_bytes: bytes | None,
    params: dict,
):
    """Store uploads and params for a plate and return the record."""
    rec = plates.setdefault(plate_id, {})
    rec["uploads"] = {"data_bytes": data_bytes, "plate_bytes": plate_bytes}
    rec["params"] = params
    return rec


def _normalize_blank_group_map(group_map) -> dict[str, str]:
    """Return normalized well->group mapping."""
    if not isinstance(group_map, dict):
        return {}
    normalized = {}
    for well, group in group_map.items():
        well_key = str(well).strip().upper()
        group_name = str(group).strip()
        if well_key and group_name:
            normalized[well_key] = group_name
    return normalized


def _analysis_group_for_wells(
    well_series: pd.Series, group_map: dict[str, str]
) -> pd.Series:
    """Return per-row analysis group for each well, defaulting to Group 1."""
    if not group_map:
        return pd.Series(DEFAULT_BLANK_GROUP, index=well_series.index, dtype="object")
    return well_series.map(group_map).fillna(DEFAULT_BLANK_GROUP)


def _apply_outlier_detection(processed: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Optionally remove outlier points from a baseline-corrected well trace."""
    if not bool(params.get("outlier_detection", False)):
        return processed
    if processed.empty:
        return processed

    try:
        factor = float(params.get("outlier_threshold", 3.5))
    except (TypeError, ValueError):
        factor = 3.5
    if not np.isfinite(factor) or factor <= 0:
        factor = 3.5

    y_arr = processed["baseline_corrected"].to_numpy(float)
    if y_arr.size < 5:
        return processed

    try:
        outlier_mask = detect_outliers(y_arr, method="ecod", factor=factor)
    except Exception:
        return processed
    if outlier_mask is None or len(outlier_mask) != len(processed):
        return processed

    keep_mask = ~pd.Series(outlier_mask, index=processed.index).fillna(False)
    return processed.loc[keep_mask].reset_index(drop=True)


def analyse_plate(record: dict):
    """Process a plate record into cleaned, baseline-corrected per-well data."""
    u = (record or {}).get("uploads") or {}
    p = (record or {}).get("params") or {}

    plate_bytes = u.get("plate_bytes")
    if plate_bytes is not None:
        plate_map, name_map = _plate_name_map(plate_bytes)
        df = _read_table(u["data_bytes"], p.get("time_unit", "hours"))
        long = df.melt(id_vars="Time", var_name="well", value_name="value")
        long["well"] = long["well"].astype(str).str.upper()
        long["name"] = long["well"].map(name_map).fillna("False")
        long = long[long["name"] != "False"].copy()
    else:
        plate_map = None
        df = _read_table(
            u["data_bytes"], p.get("time_unit", "hours"), filter_to_wells=False
        )
        long = df.melt(id_vars="Time", var_name="well", value_name="value")
        long["name"] = long["well"]

    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    clip = p.get("clip_time_series", False)
    if clip:
        a, b = clip
        if a is not None and b is not None:
            long = long.query("@a <= Time <= @b").copy()
        elif a is not None:
            long = long[long["Time"] >= a].copy()
        elif b is not None:
            long = long[long["Time"] <= b].copy()

    rm = p.get("remove_wells", False)
    if rm:
        long = long[~long["well"].isin([w.upper() for w in rm])].copy()

    # Path length correction using growthcurves preprocessing function
    long["od_1cm"] = path_correct(long["value"], float(p["pathlength_cm_"]))

    baseline = pd.DataFrame()
    grouped_baseline = pd.DataFrame()
    blank_group_map = _normalize_blank_group_map(
        p.get("blank_group_assignments", False)
    )
    if p.get("blank", True):
        blanks_long = long[long["name"].str.upper().str.startswith("BLANK")].copy()

        # one column per blank well (values are Mean), indexed by Time
        blanks_wide = blanks_long.pivot_table(
            index="Time",
            columns="well",
            values="od_1cm",
            aggfunc="mean",  # in case there are duplicates
        ).sort_index()

        if not blanks_wide.empty:
            # keep the existing mean column name for compatibility
            baseline = blanks_wide.copy()
            baseline["Mean"] = blanks_wide.mean(axis=1)

            blanks_long["analysis_group"] = _analysis_group_for_wells(
                blanks_long["well"], blank_group_map
            )
            grouped_baseline = blanks_long.pivot_table(
                index="Time",
                columns="analysis_group",
                values="od_1cm",
                aggfunc="mean",
            ).sort_index()
            for group_name in grouped_baseline.columns:
                baseline[f"{group_name} Mean"] = grouped_baseline[group_name]

            # Put mean first for compatibility/readability.
            cols = ["Mean"] + [c for c in baseline.columns if c != "Mean"]
            baseline = baseline[cols]

        long = long[~long["name"].str.upper().str.startswith("BLANK")].copy()

    # Blank subtraction using growthcurves preprocessing function
    if not baseline.empty:
        if not grouped_baseline.empty:
            long["analysis_group"] = _analysis_group_for_wells(
                long["well"], blank_group_map
            )
            grouped_long = grouped_baseline.stack().rename("group_mean").reset_index()
            long = long.merge(
                grouped_long,
                how="left",
                on=["Time", "analysis_group"],
                sort=False,
            )
            blank_values = long["group_mean"].fillna(0.0)
            long = long.drop(columns=["group_mean", "analysis_group"], errors="ignore")
        else:
            base = baseline["Mean"].to_dict()
            blank_values = long["Time"].map(base).fillna(0.0)
        long["baseline_corrected"] = blank_subtraction(long["od_1cm"], blank_values)
    else:
        long["baseline_corrected"] = long["od_1cm"]

    plate = _empty_plate()
    plate["baseline"] = baseline
    plate["plate_map"] = plate_map

    for well, g in long.groupby("well", sort=False):
        processed = g[["Time", "baseline_corrected"]].reset_index(drop=True)
        processed = _apply_outlier_detection(processed, p)

        try:
            # Use unified fitting pipeline
            t_arr = processed["Time"].to_numpy(float)
            y_arr = processed["baseline_corrected"].to_numpy(float)
            fit, fit_result = fit_growth_series(t_arr, y_arr, p)
        except Exception:
            from growthcurves.inference import bad_fit_stats

            fit = bad_fit_stats()
            fit_result = None

        plate["name"][well] = str(g["name"].iloc[0])
        plate["raw_data"][well] = g[["Time", "value", "od_1cm"]].reset_index(drop=True)
        plate["processed_data"][well] = processed
        plate["growth_stats"][well] = fit
        plate["fit_parameters"][well] = fit_result

    record.update(plate)
    return record
