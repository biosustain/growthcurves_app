"""Data processing helper functions for plate upload and validation."""

from io import BytesIO

import pandas as pd
import streamlit as st

from .constants import DEFAULT_PARAMS


def init_state():
    """Ensure required session state keys exist."""
    st.session_state.setdefault("plates", {})
    return st.session_state


def plate_params(ss, plate_id: str) -> dict:
    """Return stored params for a plate or the defaults."""
    return (ss.plates.get(plate_id, {}) or {}).get("params", DEFAULT_PARAMS)


def validate_data_file(file_bytes):
    """Validate that the data file has the correct format and data types.

    Args:
        file_bytes: Bytes from uploaded file

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        return False, f"Failed to read Excel file: {str(e)}"

    if df.empty:
        return False, "Data file is empty"

    # Check for 'Time' column (case-insensitive)
    time_col = None
    for col in df.columns:
        if str(col).strip().lower() == "time":
            time_col = col
            break

    if time_col is None:
        return False, "Data file must contain a 'Time' column"

    # Check if Time column has numeric values
    try:
        time_values = pd.to_numeric(df[time_col], errors="coerce")
        if time_values.isna().all():
            return (
                False,
                "Time column must contain numeric values (integers or decimals)",
            )
        if time_values.isna().any():
            return False, "Time column contains non-numeric values"
    except Exception:
        return False, "Failed to validate Time column data type"

    # Check that there are other columns besides Time (well data)
    if len(df.columns) < 2:
        return False, "Data file must contain well columns in addition to Time column"

    return True, None


def validate_plate_map_file(file_bytes):
    """Validate that the plate map file has the correct format and structure.

    Args:
        file_bytes: Bytes from uploaded file

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        return False, f"Failed to read Excel file: {str(e)}"

    if df.empty:
        return False, "Plate map file is empty"

    # Check for 'rows' column (case-insensitive)
    rows_col = None
    for col in df.columns:
        if str(col).strip().lower() == "rows":
            rows_col = col
            break

    if rows_col is None:
        return False, "Plate map must contain a 'rows' column"

    # Check that rows column contains expected row labels (A-H)
    expected_rows = set(list("ABCDEFGH"))
    actual_rows = set(df[rows_col].astype(str).str.strip().str.upper())

    if not expected_rows.issubset(actual_rows):
        missing_rows = expected_rows - actual_rows
        return (
            False,
            f"Plate map must contain rows A-H. Missing rows: {', '.join(sorted(missing_rows))}",
        )

    # Check that there are column headers for wells (1-12)
    numeric_cols = []
    for col in df.columns:
        if col != rows_col:
            try:
                col_num = int(str(col).strip())
                if 1 <= col_num <= 12:
                    numeric_cols.append(col_num)
            except (ValueError, TypeError):
                pass

    if len(numeric_cols) < 12:
        return False, "Plate map must contain columns 1-12 for a 96-well plate format"

    return True, None


def detect_plate_map_format(file_bytes: bytes) -> str:
    """Detect whether a plate map file is in wide or long format.

    Returns:
        "wide" if the file has a 'rows' column (standard plate layout),
        "long" if it has two columns where the first looks like well IDs,
        "unknown" otherwise.
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception:
        return "unknown"

    if df.empty:
        return "unknown"

    cols = list(df.columns)

    # Wide format: has a 'rows' column
    if any(str(c).strip().lower() == "rows" for c in cols):
        return "wide"

    # Long format: exactly 2 columns, first column contains values that look like well IDs
    valid_wells = {f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)}
    if len(cols) >= 2:
        first_col_values = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()
        if first_col_values.isin(valid_wells).any():
            return "long"

    return "unknown"


def validate_long_plate_map_file(file_bytes: bytes):
    """Validate a plate map file in long format (well, sample name per row).

    Args:
        file_bytes: Bytes from uploaded file

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        return False, f"Failed to read Excel file: {str(e)}"

    if df.empty:
        return False, "Plate map file is empty"

    if len(df.columns) < 2:
        return (
            False,
            "Long-format plate map must have at least 2 columns (well ID and sample name)",
        )

    valid_wells = {f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)}
    first_col_values = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper()

    invalid = first_col_values[~first_col_values.isin(valid_wells)]
    if len(invalid) > 0 and len(first_col_values) > 0:
        sample = ", ".join(invalid.head(3).tolist())
        return (
            False,
            f"First column must contain well IDs (e.g. A1, B2). Invalid values: {sample}",
        )

    if first_col_values.empty:
        return False, "First column contains no valid well IDs"

    return True, None


def long_plate_map_to_wide_bytes(file_bytes: bytes) -> bytes:
    """Convert a long-format plate map to wide-format Excel bytes.

    Long format: first column = well ID (e.g. A1), second column = sample name.
    Wide format: 'rows' column (A-H) + columns 1-12 with sample names.

    Args:
        file_bytes: Bytes of the long-format Excel file

    Returns:
        Bytes of a wide-format Excel file compatible with the standard plate map format
    """
    df = pd.read_excel(BytesIO(file_bytes), header=0)
    well_col = df.columns[0]
    name_col = df.columns[1]

    wide = pd.DataFrame(
        "", index=list("ABCDEFGH"), columns=list(range(1, 13)), dtype=object
    )
    wide.index.name = "rows"

    for _, row in df.iterrows():
        well = str(row[well_col]).strip().upper()
        name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        if len(well) >= 2:
            r = well[0]
            c_str = well[1:]
            try:
                c = int(c_str)
                if r in "ABCDEFGH" and 1 <= c <= 12:
                    wide.loc[r, c] = name
            except ValueError:
                pass

    buf = BytesIO()
    wide.reset_index().to_excel(buf, index=False)
    return buf.getvalue()


def validate_data_columns_are_wells(file_bytes: bytes):
    """Check that all non-Time columns are valid well IDs (A1–H12).

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        return False, f"Failed to read Excel file: {str(e)}"

    valid_wells = {f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)}
    non_time = [c for c in df.columns if str(c).strip().lower() != "time"]
    invalid = [c for c in non_time if str(c).strip().upper() not in valid_wells]
    if invalid:
        sample = ", ".join(str(c) for c in invalid[:5])
        return (
            False,
            f"When a plate map is provided, all data columns must be well IDs (A1–H12). Invalid columns: {sample}",
        )
    return True, None


@st.cache_data(show_spinner="Loading plate preview...")
def get_plate_preview_data(plate_bytes: bytes | None, data_bytes: bytes):
    """Get plate map and present wells without full analysis.

    Args:
        plate_bytes: Bytes of the plate map Excel file, or None if no plate map
        data_bytes: Bytes of the data Excel file

    Returns:
        tuple: (plate_map DataFrame or None, set of present column names)
    """
    data_df = pd.read_excel(BytesIO(data_bytes))
    time_col = next(
        (c for c in data_df.columns if str(c).strip().lower() == "time"), None
    )
    present = set(data_df.columns) - {time_col} if time_col else set(data_df.columns)

    if plate_bytes is None:
        return None, present

    plate_map = pd.read_excel(BytesIO(plate_bytes), index_col=0).fillna("False")
    return plate_map, present
