"""Data processing helper functions for plate upload and validation."""

from io import BytesIO

import pandas as pd
import streamlit as st

from .constants import (
    DEFAULT_PARAMS,
    PLATE_96,
    PLATE_FORMATS,
    VALID_WELLS,
    detect_plate_format,
)
from .data_processing import is_time_column, time_to_hours


def init_state():
    """Ensure required session state keys exist."""
    st.session_state.setdefault("plates", {})
    return st.session_state


def plate_params(ss, plate_id: str) -> dict:
    """Return stored params for a plate or the defaults."""
    return (ss.plates.get(plate_id, {}) or {}).get("params", DEFAULT_PARAMS)


def read_data_file(file_bytes: bytes) -> pd.DataFrame:
    """Parse a plate reader Excel file once, keeping duplicate headers as they are.

    The Time column (any case) is renamed to "Time".
    """
    raw = pd.read_excel(BytesIO(file_bytes), header=None, dtype=object)
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = ["Time" if is_time_column(c) else str(c).strip() for c in raw.iloc[0]]
    return df.infer_objects()


def plate_map_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide plate map indexed by row letter, with "False" for wells without a name."""
    rows_col = next(c for c in wide.columns if str(c).strip().lower() == "rows")
    return wide.set_index(rows_col).fillna("False").replace("", "False")


def validate_data_file(df: pd.DataFrame):
    """Validate that the data file has the correct format and data types.

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if df.empty:
        return False, "Data file is empty"

    if "Time" not in df.columns:
        return False, "Data file must contain a 'Time' column"

    # Time must be all numbers or all HH:MM:SS (the unit is chosen in Step 2)
    if all(
        parsed.isna().any()
        for parsed in (
            pd.to_numeric(df["Time"], errors="coerce"),
            time_to_hours(df["Time"], "HH:MM:SS"),
        )
    ):
        return (
            False,
            "Time column must contain only numbers (e.g. 0, 0.5) "
            "or only HH:MM:SS times (e.g. 01:30:00)",
        )

    # Check that there are other columns besides Time (well data)
    if len(df.columns) < 2:
        return False, "Data file must contain well columns in addition to Time column"

    return True, None


def validate_plate_map_file(df: pd.DataFrame):
    """Validate that the plate map has the correct format and structure.

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if df.empty:
        return False, "Plate map file is empty"

    # Check for 'rows' column (case-insensitive)
    rows_col = next((c for c in df.columns if str(c).strip().lower() == "rows"), None)
    if rows_col is None:
        return False, "Plate map must contain a 'rows' column"

    # Every supported plate contains at least the 96-well rows A-H and columns 1-12
    expected_rows = set(PLATE_96.rows)
    actual_rows = set(df[rows_col].astype(str).str.strip().str.upper())

    if not expected_rows.issubset(actual_rows):
        missing_rows = expected_rows - actual_rows
        return (
            False,
            "Plate map must contain rows A-H (A-P for a 384-well plate). "
            f"Missing rows: {', '.join(sorted(missing_rows))}",
        )

    # Check that there are column headers for wells (1-12)
    numeric_cols = set()
    for col in df.columns:
        if col != rows_col:
            try:
                numeric_cols.add(int(str(col).strip()))
            except (ValueError, TypeError):
                pass

    if not set(PLATE_96.cols) <= numeric_cols:
        return (
            False,
            "Plate map must contain columns 1-12 (1-24 for a 384-well plate)",
        )

    return True, None


def detect_plate_map_format(df: pd.DataFrame) -> str:
    """Detect whether a plate map is in wide or long format.

    Returns:
        "wide" if the file has a 'rows' column (standard plate layout),
        "long" if it has two columns where the first looks like well IDs,
        "unknown" otherwise.
    """
    if df.empty:
        return "unknown"

    cols = list(df.columns)

    # Wide format: has a 'rows' column
    if any(str(c).strip().lower() == "rows" for c in cols):
        return "wide"

    # Long format: exactly 2 columns, first column contains values that look like well IDs
    if len(cols) >= 2:
        first_col_values = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()
        if first_col_values.isin(VALID_WELLS).any():
            return "long"

    return "unknown"


def validate_long_plate_map_file(df: pd.DataFrame):
    """Validate a plate map in long format (well, sample name per row).

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if df.empty:
        return False, "Plate map file is empty"

    if len(df.columns) < 2:
        return (
            False,
            "Long-format plate map must have at least 2 columns (well ID and sample name)",
        )

    first_col_values = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()

    invalid = first_col_values[~first_col_values.isin(VALID_WELLS)]
    if len(invalid) > 0 and len(first_col_values) > 0:
        sample = ", ".join(invalid.head(3).tolist())
        return (
            False,
            f"First column must contain well IDs (e.g. A1, B2). Invalid values: {sample}",
        )

    if first_col_values.empty:
        return False, "First column contains no valid well IDs"

    return True, None


def long_plate_map_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a long-format plate map to the wide format of an uploaded plate map.

    Long format: first column = well ID (e.g. A1), second column = sample name.
    Wide format: 'rows' column + one column per plate column, with sample names.
    The plate size (96 or 384 wells) is detected from the well IDs.
    """
    wells = df.iloc[:, 0].astype(str).str.strip().str.upper()
    names = df.iloc[:, 1]
    fmt = detect_plate_format(wells)

    wide = pd.DataFrame("", index=list(fmt.rows), columns=list(fmt.cols), dtype=object)
    wide.index.name = "rows"

    for well, name in zip(wells, names):
        pt = fmt.point(well)
        if pt:
            wide.iat[pt] = str(name).strip() if pd.notna(name) else ""

    return wide.reset_index()


def plate_from_column_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Return (data, wide plate map, positions assigned) with the headers as sample names."""
    names = [c for c in df.columns if c != "Time"]
    assigned = not all(n.upper() in VALID_WELLS for n in names)
    if not assigned:
        wells = [n.upper() for n in names]
    else:
        fmt = next(f for f in PLATE_FORMATS.values() if len(names) <= f.n_wells)
        wells = list(fmt.wells[: len(names)])

    long_map = pd.DataFrame({"well": wells, "name": names})
    well_ids = iter(wells)
    df = df.set_axis([c if c == "Time" else next(well_ids) for c in df.columns], axis=1)
    return df, long_plate_map_to_wide(long_map), assigned


def validate_data_columns_are_wells(df: pd.DataFrame):
    """Check that all non-Time columns are valid well IDs (A1–H12, or A1–P24).

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    non_time = [c for c in df.columns if c != "Time"]
    invalid = [c for c in non_time if str(c).strip().upper() not in VALID_WELLS]
    if invalid:
        sample = ", ".join(str(c) for c in invalid[:5])
        return (
            False,
            f"When a plate map is provided, all data columns must be well IDs (A1–H12, or A1–P24 for 384-well plates). Invalid columns: {sample}",
        )
    return True, None
