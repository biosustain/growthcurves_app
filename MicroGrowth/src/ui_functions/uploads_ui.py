"""UI fragments for the Upload and Analyze page."""

import io
import re
from typing import Any

import pandas as pd
import streamlit as st
from growthcurves.models import MODEL_REGISTRY
from src.functions.constants import COLS, DEFAULT_PARAMS, ROWS
from src.functions.data_processing import analyse_plate, load_plate
from src.functions.upload_functions import (
    detect_plate_map_format,
    get_plate_preview_data,
    long_plate_map_to_wide_bytes,
    plate_params,
    validate_data_columns_are_wells,
    validate_data_file,
    validate_long_plate_map_file,
    validate_plate_map_file,
)
from src.styling import growth_param_table_style
from src.ui_functions.blank_grouping_ui import (
    DEFAULT_GROUP,
    color_for_group,
    darken_hex_color,
    st_selectable_grid,
    ui_blank_group_assigner,
)
from src.ui_functions.ui_components import (
    ui_method_visualization,
    ui_phase_boundary_visualization,
)


def ui_upload_and_analyse_header():
    """Render page title and help popover."""
    title_col, popover_col = st.columns([9, 2])
    with title_col:
        st.title("Upload and Analyze")
    with popover_col:
        st.write("")
        with st.popover("Help", width="stretch"):
            st.markdown("""
**Workflow Overview — Upload & Analyse**

This is your starting point. Follow the 6 steps in order to upload your data and run the growth analysis.

**Step 1 — Upload data file**
Upload your plate reader Excel file. Click "Requirements" to see the expected format and download an example. Your file must have a **Time** column plus one column per well (e.g. A1, A2, ...).

**Step 2 — Upload sample names**
Optionally upload a plate map Excel file that assigns sample names to each well. If provided, data column names must be well IDs (A1–H12) and the plate map is used to label each well. If omitted, the data column names are used directly as sample names. Wells with the same name are treated as replicates. Use **BLANK** (or any name starting with BLANK) for blank wells. Click "Requirements" for the expected format and an example download.

**Step 3 — Match samples with names**
Click the button to load your file(s). If a plate map is provided it will be matched to the data; otherwise column names are used as sample names directly.

**Step 4 — Select preprocessing parameters**
Configure how the data is processed before analysis:
- **Time unit**: Set to match the unit in your data file (seconds, minutes, hours, days, or HH:MM:SS)
- **Pathlength**: Your plate reader's optical path length, used to normalise OD to 1 cm
- **Blank subtraction groups**: Link sample wells to blank wells by analysis group so subtraction uses only matched blank wells (any well named BLANK or starting with BLANK)
- **Time range**: Restrict analysis to a specific window of the experiment
- **Outlier detection**: Optionally detect/remove outliers with a sliding IQR window (window size + threshold)
- **Exclude wells**: Manually remove specific wells (e.g. contaminated or failed wells)

The plate preview updates live — colored wells are included, and gray wells are not included. Hover any well for details.

**Step 5 — Select analysis parameters**
Choose how growth descriptors are calculated:
- **Model family & method**: Select a parametric model (mechanistic or phenomenological) or a non-parametric approach (Sliding Window or Spline). The visualisation below the selector shows the shape of the selected model.
- **Spline fitting mode**: For the Spline method, choose **Fast** (auto-default smoothing with OD weighting) or **Slow** (weighted GCV smoothing).
- **Quality control filters**: Wells not meeting these thresholds are automatically marked as "no growth"
- **Phase boundary method**: Controls how the lag phase end and exponential phase end are determined

The table at the bottom shows exactly how each growth parameter will be calculated for your selected settings.

**Step 6 — Analyse**
Click the button to run the analysis. Once complete, navigate to the other pages using the top navigation bar to review and download your results.
""")

    st.divider()


def _build_plate_preview_cells(
    *,
    plate_map: pd.DataFrame | None,
    present: set[str],
    remove_wells: list[str] | bool,
    blank: bool,
    blank_group_assignments: dict[str, str] | bool,
) -> list[list[dict[str, Any]]]:
    """Build st_selectable_grid cell payload for upload preview."""
    present_wells = {str(well).strip().upper() for well in present}
    removed_wells = {str(well).strip().upper() for well in (remove_wells or [])}
    group_map = (
        {
            str(well).strip().upper(): str(group).strip() or DEFAULT_GROUP
            for well, group in blank_group_assignments.items()
        }
        if isinstance(blank_group_assignments, dict)
        else {}
    )

    cells: list[list[dict[str, Any]]] = []
    for row in ROWS:
        rendered_row: list[dict[str, Any]] = []
        for col in COLS:
            well = f"{row}{col}"
            group_name = group_map.get(well, DEFAULT_GROUP)

            if plate_map is None:
                rendered_row.append(
                    {
                        "label": well,
                        "cell_color": "#e5e7eb",
                        "tooltip": f"{well} · No uploaded files",
                    }
                )
                continue

            sample = _plate_cell_name(plate_map, row, col).strip()
            is_blank_well = sample.upper().startswith("BLANK")
            cell_label = f"<b>{well}</b>" if is_blank_well else well
            has_sample_name = sample not in {
                "",
                "False",
            } and not sample.upper().startswith("BLANK")
            is_not_in_plate_map = sample in {"", "False"}

            included = False
            exclusion_reason = ""
            if well in removed_wells:
                exclusion_reason = "excluded by user"
            elif is_not_in_plate_map:
                exclusion_reason = "not in plate map"
            elif well not in present_wells:
                exclusion_reason = "missing from data file"
            elif is_blank_well and not blank:
                exclusion_reason = "blank subtraction disabled"
            elif is_blank_well and blank:
                included = True
            elif has_sample_name:
                included = True
            else:
                exclusion_reason = "not included"

            sample_suffix = f" · {sample}" if sample and sample != "False" else ""
            if included:
                base_color = color_for_group(group_name)
                cell_data: dict[str, Any] = {
                    "label": cell_label,
                    "cell_color": (
                        darken_hex_color(base_color) if is_blank_well else base_color
                    ),
                    "tooltip": (
                        f"{well}{sample_suffix}"
                        f"{' · BLANK well' if is_blank_well else ''} · {group_name}"
                    ),
                }
                if is_blank_well:
                    cell_data["html"] = True
                if is_blank_well:
                    cell_data["cell_border_width"] = 2
                    cell_data["cell_border_color"] = darken_hex_color(
                        base_color, factor=0.72
                    )
                rendered_row.append(cell_data)
            else:
                rendered_row.append(
                    {
                        "label": cell_label,
                        "cell_color": "#e5e7eb",
                        "tooltip": f"{well}{sample_suffix} · {exclusion_reason}",
                        **({"html": True} if is_blank_well else {}),
                    }
                )
        cells.append(rendered_row)
    return cells


def render_plate_table(
    *,
    key: str,
    plate_map: pd.DataFrame | None,
    present: set[str] | None = None,
    remove_wells: list[str] | bool = False,
    blank: bool = True,
    blank_group_assignments: dict[str, str] | bool = False,
    grid_height: int = 440,
    grid_aspect_ratio: float = 1.0,
):
    """Render plate preview with the blank-linker style table."""
    cells = _build_plate_preview_cells(
        plate_map=plate_map,
        present=present or set(),
        remove_wells=remove_wells,
        blank=blank,
        blank_group_assignments=blank_group_assignments,
    )

    if st_selectable_grid is None:
        # Fallback keeps well labels visible when optional dependency is unavailable.
        fallback_df = pd.DataFrame(
            [
                [re.sub(r"<[^>]+>", "", str(cell["label"])) for cell in row]
                for row in cells
            ],
            index=ROWS,
            columns=COLS,
        )
        st.dataframe(fallback_df, width="stretch", height=grid_height)
        return

    st_selectable_grid(
        cells=cells,
        header=[str(c) for c in COLS],
        index=ROWS,
        aspect_ratio=grid_aspect_ratio,
        allow_secondary_selection=False,
        allow_header_selection=False,
        resize=True,
        height=grid_height,
        primary_selection_color="#6b7280",
        key=f"uploads_preview::{key}",
    )


def _plate_cell_name(plate_map: pd.DataFrame, row: str, col: int) -> str:
    """Return normalized sample name at a plate position."""
    if col in plate_map.columns:
        value = plate_map.loc[row, col]
    elif str(col) in plate_map.columns:
        value = plate_map.loc[row, str(col)]
    else:
        value = "False"
    return str(value).strip()


def _name_by_well_from_plate_map(plate_map: pd.DataFrame) -> dict[str, str]:
    """Return well->sample-name mapping from a preview plate map."""
    return {
        f"{row}{col}": _plate_cell_name(plate_map, row, col)
        for row in ROWS
        for col in COLS
    }


def ui_upload_files(ss):
    """Fragment for file upload controls."""
    u1, u2 = st.columns(2)
    with u1:
        with st.container(border=True):
            # Header row with requirements in top right
            header_col, req_col = st.columns([3, 1])
            with header_col:
                st.header("Step 1. Upload data")
            with req_col:
                with st.popover("Requirements", width="stretch"):
                    st.caption(
                        "Here you can upload your time series data. "
                        "You can choose to name the samples using column headings in "
                        "the data file, or upload a separate plate map to assign "
                        "sample names to wells. If you upload a plate map, column "
                        "names in the data file must be well IDs (e.g. A1, B3)."
                    )
                    st.image("MicroGrowth/info_plots/data_upload.png", width="stretch")

                    st.caption(
                        "The Time column format can be either a float (e.g. 0, 0.5, 1.0) or HH:MM:SS "
                        "(e.g. 00:00:00, 00:30:00, 01:00:00). The units can be selected below in Step 4."
                    )

                    st.markdown("")
                    with open("MicroGrowth/example_data/example_data.xlsx", "rb") as f:
                        st.download_button(
                            "Download example data file",
                            data=f.read(),
                            file_name="example_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                            type="primary",
                            key="download_example_data",
                        )

            data_file = st.file_uploader(
                "Plate reader Excel (.xlsx/.xls)", ["xlsx", "xls"], key="data_up"
            )
    with u2:
        with st.container(border=True):
            # Header row with requirements in top right
            header_col, req_col = st.columns([3, 1])
            with header_col:
                st.header("Step 2. Upload names")
            with req_col:
                with st.popover("Requirements", width="stretch"):
                    st.caption(
                        "The plate map assigns sample names to each well. "
                        "If provided, data column names must be well IDs (A1–H12) and the "
                        "plate map is used to label each well. If omitted, the data "
                        "column names are used directly as sample names. Wells with the same "
                        "name are treated as replicates in the app. Use BLANK (or any name "
                        "starting with BLANK) for blank wells."
                    )
                    st.image(
                        "MicroGrowth/info_plots/naming_upload.png", width="stretch"
                    )

                    st.divider()
                    st.caption(
                        "Sample names can optionally include a strain and a condition, "
                        "separated by the first underscore (e.g. **Strain1_Condition1**). "
                        "The Create Visualizations page uses this to group and color samples "
                        "by strain or condition."
                    )
                    st.image(
                        "MicroGrowth/info_plots/naming_convention.png", width="stretch"
                    )

                    st.markdown("")
                    dl_wide_col, dl_long_col = st.columns(2)
                    with dl_wide_col:
                        with open(
                            "MicroGrowth/example_data/example_plate_map.xls", "rb"
                        ) as f:
                            st.download_button(
                                "Download example plate map (wide)",
                                data=f.read(),
                                file_name="example_plate_map.xls",
                                mime="application/vnd.ms-excel",
                                width="stretch",
                                type="primary",
                                key="download_example_plate_map",
                            )
                    with dl_long_col:
                        with open(
                            "MicroGrowth/example_data/example_long_form_plate_map.xls",
                            "rb",
                        ) as f:
                            st.download_button(
                                "Download example plate map (long)",
                                data=f.read(),
                                file_name="example_long_form_plate_map.xls",
                                mime="application/vnd.ms-excel",
                                width="stretch",
                                type="primary",
                                key="download_example_long_plate_map",
                            )

            map_file = st.file_uploader(
                "Plate map (.xls/.xlsx) — wide or long format (optional)",
                ["xlsx", "xls"],
                key="map_up",
            )

    with st.container(border=True):
        st.header("Step 3. Match samples with names")
        if st.button(
            "Match samples with names",
            type="primary",
            width="stretch",
            disabled=not data_file,
        ):
            is_valid_data, data_error = validate_data_file(data_file.getvalue())
            if not is_valid_data:
                st.toast(
                    f"❌ Data file validation failed: {data_error}",
                    duration="infinite",
                )
                st.stop()

            plate_bytes = None
            if map_file:
                map_format = detect_plate_map_format(map_file.getvalue())
                if map_format == "long":
                    is_valid_map, map_error = validate_long_plate_map_file(
                        map_file.getvalue()
                    )
                    if not is_valid_map:
                        st.toast(
                            f"❌ Plate map validation failed: {map_error}",
                            duration="infinite",
                        )
                        st.stop()
                    plate_bytes = long_plate_map_to_wide_bytes(map_file.getvalue())
                else:
                    is_valid_map, map_error = validate_plate_map_file(
                        map_file.getvalue()
                    )
                    if not is_valid_map:
                        st.toast(
                            f"❌ Plate map validation failed: {map_error}",
                            duration="infinite",
                        )
                        st.stop()
                    plate_bytes = map_file.getvalue()

                is_valid_cols, cols_error = validate_data_columns_are_wells(
                    data_file.getvalue()
                )
                if not is_valid_cols:
                    st.toast(f"❌ {cols_error}", duration="infinite")
                    st.stop()

            plate_id = (
                data_file.name.rsplit(".", 1)[0]
                if getattr(data_file, "name", None)
                else "Plate"
            )
            load_plate(
                ss.plates,
                plate_id,
                data_bytes=data_file.getvalue(),
                plate_bytes=plate_bytes,
                params=DEFAULT_PARAMS,
            )
            st.toast(f"Successfully loaded {plate_id}", duration="infinite")


@st.fragment
def ui_preprocessing_params(ss):
    """Fragment for preprocessing parameters and plate preview."""
    ready = sorted(ss.plates)
    plate_id = None
    params0 = DEFAULT_PARAMS
    blank = False
    blank_group_assignments: dict[str, str] | bool = False
    remove_wells = False
    clip_time_series = (None, None)
    time_unit = "hours"
    pl_cm = float(DEFAULT_PARAMS["pathlength_cm_"])
    outlier_detection = bool(DEFAULT_PARAMS.get("outlier_detection", False))
    outlier_threshold = float(DEFAULT_PARAMS.get("outlier_threshold", 3.5))
    plate_map = None
    present: set[str] = set()
    name_by_well: dict[str, str] = {}
    has_blank_wells = False
    blank_well_count = 0

    with st.container(border=True):
        st.header("Step 4. Select plate and preprocessing parameters")

        selector_col, plate_name_col = st.columns(
            [1.0, 1.35], vertical_alignment="bottom"
        )
        with selector_col:
            plate_id = st.selectbox("Plate to analyse", ready, disabled=not ready)
        with plate_name_col:
            if plate_id:
                st.markdown(
                    f"<h3 style='text-align:center; margin:0; font-size:2rem;'>{plate_id}</h3>",
                    unsafe_allow_html=True,
                )
            else:
                st.write("")
        params0 = plate_params(ss, plate_id) if plate_id else DEFAULT_PARAMS
        outlier_detection = bool(params0.get("outlier_detection", False))
        try:
            outlier_threshold = float(params0.get("outlier_threshold", 3.5))
        except (TypeError, ValueError):
            outlier_threshold = 3.5

        if plate_id:
            rec = ss.plates.get(plate_id, {})
            if rec.get("uploads"):
                uploads = rec["uploads"]
                plate_map, present = get_plate_preview_data(
                    plate_bytes=uploads["plate_bytes"],
                    data_bytes=uploads["data_bytes"],
                )
                if plate_map is not None:
                    name_by_well = _name_by_well_from_plate_map(plate_map)
                    blank_well_count = sum(
                        1
                        for v in name_by_well.values()
                        if v.upper().startswith("BLANK")
                    )
                    has_blank_wells = blank_well_count > 0

        controls_col, plate_col = st.columns([1.0, 1.35], gap="large")
        plate_grid_height = 400
        plate_grid_aspect_ratio = 0.65
        initial_group_map: dict[str, str] | None = None

        with controls_col:
            _outlier_key = f"outlier_cb_{plate_id}"
            _row_cols = st.columns([1.0, 0.8, 1.0, 0.8], vertical_alignment="bottom")
            a, b = _row_cols[0], _row_cols[1]
            _time_unit_options = ["seconds", "minutes", "hours", "days", "HH:MM:SS"]
            _saved_time_unit = params0.get("time_unit", "hours")
            time_unit = a.selectbox(
                "Time unit",
                options=_time_unit_options,
                index=_time_unit_options.index(
                    _saved_time_unit
                    if _saved_time_unit in _time_unit_options
                    else "hours"
                ),
                help="Select the unit of time values in your data file's Time column. Use HH:MM:SS if your Time column contains values like 01:30:00.",
            )
            pl_cm = b.number_input(
                "Path (cm)",
                value=float(params0["pathlength_cm_"]),
                step=0.01,
                format="%.3f",
                help="Optical pathlength of the plate reader (used to normalize OD600 values to 1 cm pathlength)",
            )
            outlier_detection = _row_cols[2].checkbox(
                "Remove outliers",
                value=outlier_detection,
                key=_outlier_key,
            )
            if outlier_detection:
                outlier_threshold = float(
                    _row_cols[3].number_input(
                        "Threshold",
                        min_value=1.0,
                        max_value=5.0,
                        value=outlier_threshold,
                        step=0.1,
                        format="%.2f",
                        help=(
                            "MAD z-score threshold for flagging outliers. Higher values flag "
                            "fewer, more extreme points."
                        ),
                    )
                )

            # Derive min/max time in hours from the uploaded data file
            _min_time_h = 0.0
            _max_time_h = 1e6
            if plate_id:
                _data_bytes = (
                    (ss.plates.get(plate_id) or {}).get("uploads", {}).get("data_bytes")
                )
                if _data_bytes:
                    try:
                        _t_col = pd.read_excel(io.BytesIO(_data_bytes), header=0)[
                            "Time"
                        ]
                        if time_unit == "HH:MM:SS":

                            def _hhmmss_to_hours(val):
                                parts = str(val).strip().split(":")
                                if len(parts) == 3:
                                    return (
                                        int(parts[0])
                                        + int(parts[1]) / 60.0
                                        + float(parts[2]) / 3600.0
                                    )
                                return float("nan")

                            _t_hours = _t_col.map(_hhmmss_to_hours).dropna()
                        else:
                            _t_raw = pd.to_numeric(_t_col, errors="coerce").dropna()
                            _divisor = (
                                3600.0
                                if time_unit == "seconds"
                                else (
                                    60.0
                                    if time_unit == "minutes"
                                    else 1 / 24.0 if time_unit == "days" else 1.0
                                )
                            )
                            _t_hours = _t_raw / _divisor
                        if not _t_hours.empty:
                            _min_time_h = float(_t_hours.min())
                            _max_time_h = float(_t_hours.max())
                    except Exception:
                        pass

            _clip = params0.get("clip_time_series") or (None, None)
            _clip_start = (
                float(max(_min_time_h, min(_clip[0], _max_time_h)))
                if _clip[0] is not None
                else _min_time_h
            )
            _clip_end = (
                float(max(_min_time_h, min(_clip[1], _max_time_h)))
                if _clip[1] is not None
                else _max_time_h
            )
            _clip_start = min(_clip_start, _clip_end)
            clip_time_series = st.slider(
                "Define analysis window (h)",
                min_value=_min_time_h,
                max_value=_max_time_h,
                value=(_clip_start, _clip_end),
                step=0.5,
                help="Time range for analysis — data points outside this window will be excluded",
            )

            # Get default excluded wells from params0
            default_excluded = params0.get("remove_wells", [])
            if default_excluded is False or not default_excluded:
                default_excluded = []

            remove_wells = st.multiselect(
                "Exclude wells",
                options=[f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)],
                default=default_excluded,
                help="Manually exclude specific wells from analysis (e.g., contaminated samples)",
            )

            # Preserve the False sentinel behavior used elsewhere.
            remove_wells = remove_wells if remove_wells else False

            st.divider()

            blank = has_blank_wells
            initial_group_map = params0.get("blank_group_assignments", False)
            if not isinstance(initial_group_map, dict):
                initial_group_map = None

            ui_blank_group_assigner(
                plate_id=plate_id or "placeholder",
                initial_group_map=initial_group_map,
                name_by_well=name_by_well,
                present_wells=present,
                remove_wells=remove_wells,
                blank_enabled=blank,
                show_caption=True,
                show_controls=True,
                controls_disabled=blank_well_count <= 1,
                show_grid=False,
                grid_height=plate_grid_height,
                grid_aspect_ratio=plate_grid_aspect_ratio,
            )

        with plate_col:
            if plate_id and plate_map is not None:
                blank_group_assignments = ui_blank_group_assigner(
                    plate_id=plate_id,
                    initial_group_map=initial_group_map,
                    name_by_well=name_by_well,
                    present_wells=present,
                    remove_wells=remove_wells,
                    blank_enabled=blank,
                    show_caption=False,
                    show_controls=False,
                    show_grid=True,
                    grid_height=plate_grid_height,
                    grid_aspect_ratio=plate_grid_aspect_ratio,
                )
            else:
                render_plate_table(
                    key="empty",
                    plate_map=None,
                    grid_height=plate_grid_height,
                    grid_aspect_ratio=plate_grid_aspect_ratio,
                )
            if st.button(
                "Remove selected plate",
                type="tertiary",
                width="stretch",
                disabled=not plate_id,
            ):
                ss.plates.pop(plate_id, None)
                st.rerun()

    # Store selected values in session state for access by other fragments
    if plate_id:
        ss.setdefault("step3_params", {})
        ss["step3_params"]["plate_id"] = plate_id
        ss["step3_params"]["time_unit"] = time_unit
        ss["step3_params"]["pl_cm"] = pl_cm
        ss["step3_params"]["blank"] = blank
        ss["step3_params"]["blank_group_assignments"] = blank_group_assignments
        ss["step3_params"]["outlier_detection"] = bool(outlier_detection)
        ss["step3_params"]["outlier_threshold"] = float(outlier_threshold)
        ss["step3_params"]["clip_time_series"] = clip_time_series
        ss["step3_params"]["remove_wells"] = remove_wells
        ss["step3_params"]["params0"] = params0


# Helper functions for analysis parameters
def _get_model_display_name(model_code: str) -> str:
    """Convert model code to display name."""
    display_names = {
        "mech_logistic": "Logistic",
        "mech_gompertz": "Gompertz",
        "mech_richards": "Richards",
        "mech_baranyi": "Baranyi-Roberts",
        "phenom_logistic": "Logistic",
        "phenom_gompertz": "Gompertz",
        "phenom_gompertz_modified": "Modified Gompertz",
        "phenom_richards": "Richards",
        "sliding_window": "Sliding Window",
        "spline": "Spline",
    }
    return display_names.get(model_code, model_code)


def _ui_model_selection(params0: dict):
    """Render model family and growth method selection UI."""
    stored_method = params0.get("growth_method", "Spline")
    stored_model_family = params0.get("model_family", "mechanistic")
    stored_model_type = params0.get("model_type", "mech_logistic")

    st.caption("Select the model family and growth descriptor method:")

    family_col, method_col, param_col = st.columns(3)

    # Determine default family index from stored state
    if stored_method in ["Sliding Window", "Spline"]:
        default_family_idx = 2
    elif stored_model_family == "mechanistic":
        default_family_idx = 0
    else:
        default_family_idx = 1

    with family_col:
        model_family = st.selectbox(
            "Model family",
            options=[
                "Mechanistic parametric",
                "Phenomenological parametric",
                "Non-parametric",
            ],
            index=default_family_idx,
            help="Mechanistic models use ODE-based biological principles. Phenomenological models describe growth patterns empirically. Non-parametric methods are data-driven without a fixed curve shape.",
        )

    if model_family == "Mechanistic parametric":
        model_family_internal = "mechanistic"
    elif model_family == "Phenomenological parametric":
        model_family_internal = "phenomenological"
    else:
        model_family_internal = "non_parametric"

    # Build method options from MODEL_REGISTRY
    method_options = []

    if model_family == "Mechanistic parametric":
        for model_code in MODEL_REGISTRY["mechanistic"]:
            method_options.append(
                (_get_model_display_name(model_code), model_code, "Model Fitting")
            )
    elif model_family == "Phenomenological parametric":
        for model_code in MODEL_REGISTRY["phenomenological"]:
            method_options.append(
                (_get_model_display_name(model_code), model_code, "Model Fitting")
            )
    else:  # Non-parametric
        for model_code in MODEL_REGISTRY["non_parametric"]:
            growth_method = (
                "Sliding Window" if model_code == "sliding_window" else "Spline"
            )
            method_options.append(
                (_get_model_display_name(model_code), model_code, growth_method)
            )

    # Determine default index
    default_idx = 0
    for i, (label, code, method) in enumerate(method_options):
        if stored_method in ["Sliding Window", "Spline"]:
            if code == "sliding_window" and stored_method == "Sliding Window":
                default_idx = i
                break
            elif code == "spline" and stored_method == "Spline":
                default_idx = i
                break
        elif stored_method == "Model Fitting":
            if code == stored_model_type:
                default_idx = i
                break

    with method_col:
        selected_method_label = st.selectbox(
            "Growth descriptor method",
            options=[m[0] for m in method_options],
            index=default_idx,
            help="Choose between non-parametric (data-driven) or parametric (model-based) approaches.",
        )

    # Extract internal codes
    growth_method = None
    model_type = None
    for label, code, method in method_options:
        if label == selected_method_label:
            growth_method = method
            if method == "Model Fitting":
                model_type = code
            break

    return model_family_internal, growth_method, model_type, param_col


def ui_model_params(growth_method: str, params0: dict, step4_prev: dict, param_col):
    """Render method-specific parameters (window size or spline mode)."""
    smooth_default = (
        str(step4_prev.get("smooth", params0.get("smooth", "fast"))).strip().lower()
    )
    if smooth_default not in {"fast", "slow"}:
        smooth_default = "fast"

    with param_col:
        if growth_method == "Sliding Window":
            window_points = st.number_input(
                "Window size (points)",
                5,
                200,
                int(params0["window_points"]),
                1,
                help="Number of consecutive data points used for sliding window linear fit to determine maximum growth rate",
            )
            smooth = smooth_default
        elif growth_method == "Spline":
            window_points = int(params0["window_points"])
            smooth = st.radio(
                "Spline fitting mode",
                options=["fast", "slow"],
                index=0 if smooth_default == "fast" else 1,
                horizontal=True,
                format_func=lambda v: v.capitalize(),
                help=(
                    "Fast uses auto-default smoothing with OD weights. "
                    "Slow uses weighted GCV smoothing and is typically slower."
                ),
            )
        else:  # Model Fitting
            window_points = int(params0["window_points"])
            smooth = smooth_default

    return window_points, smooth


def ui_qc_filters(params0: dict):
    """Render quality control filter inputs."""
    st.caption("Wells failing these criteria will be marked as no growth")

    col1, col2, col3, col4 = st.columns(4)
    min_data_points = col1.number_input(
        "Minimum data points",
        1,
        100,
        int(params0.get("min_data_points", 5)),
        1,
        help="Minimum number of valid data points required for growth analysis",
    )
    min_signal_to_noise = col2.number_input(
        "Minimum signal:noise",
        0.1,
        100.0,
        float(params0.get("min_signal_to_noise", 1.0)),
        0.1,
        help="Minimum ratio of maximum to minimum OD600 signal (filters out flat curves)",
    )
    min_od_increase = col3.number_input(
        "Minimum OD increase",
        0.0,
        None,
        float(params0.get("min_od_increase", 0.05)),
        0.001,
        format="%.3f",
        help="Minimum absolute increase in OD600 from baseline to be considered growth",
    )
    min_growth_rate = col4.number_input(
        "Minimum growth rate",
        0.0,
        None,
        float(params0.get("min_growth_rate", 0.001)),
        0.0001,
        format="%.4f",
        help="Minimum specific growth rate to be considered growth (wells with lower rates are marked as no growth)",
    )

    return min_data_points, min_signal_to_noise, min_od_increase, min_growth_rate


def ui_phase_boundaries(params0: dict):
    """Render phase boundary method selection UI."""
    st.caption(
        "Phase boundaries define when the lag phase ends and when the exponential phase ends."
    )
    phase_boundary_method = st.selectbox(
        "Phase boundary calculation",
        options=["threshold", "tangent"],
        index=(
            0 if params0.get("phase_boundary_method", "tangent") == "threshold" else 1
        ),
        format_func=lambda v: v.capitalize(),
        help="Threshold uses fractions of μ_max; tangent uses the tangent at μ_max to estimate exponential phase bounds.",
    )
    lag_cutoff = st.number_input(
        "Lag phase cutoff",
        0.01,
        0.5,
        float(params0.get("lag_cutoff", 0.5)),
        0.01,
        format="%.2f",
        disabled=phase_boundary_method == "tangent",
        help="Fraction of maximum growth rate used to define lag phase end (threshold mode).",
    )
    exp_cutoff = st.number_input(
        "Exponential phase cutoff",
        0.01,
        0.5,
        float(params0.get("exp_cutoff", 0.5)),
        0.01,
        format="%.2f",
        disabled=phase_boundary_method == "tangent",
        help="Fraction of maximum growth rate used to define exponential phase end (threshold mode).",
    )

    return phase_boundary_method, lag_cutoff, exp_cutoff


def ui_calculation_table(
    growth_method: str,
    model_type: str,
    model_family: str,
    phase_boundary_method: str,
    lag_cutoff: float,
    exp_cutoff: float,
    window_points: int,
):
    """Render the growth parameter calculations table."""
    if growth_method == "Model Fitting":
        mu_max_calc = "μ<sub>max</sub>"
        model_rmse_calc = "RMSE over entire curve"
        max_od_calc = "Maximum OD from fitted model"
    else:
        max_od_calc = "Maximum raw OD"
        if growth_method == "Sliding Window":
            mu_max_calc = "b"
            model_rmse_calc = f"RMSE over {window_points} point sliding-window"
        else:
            mu_max_calc = "Max spline derivative"
            model_rmse_calc = "RMSE over spline fit window (log phase)"

    if growth_method == "Model Fitting" and model_family == "mechanistic":
        intrinsic_calc = "Fitted intrinsic μ"
    else:
        intrinsic_calc = "N.a."

    if phase_boundary_method == "threshold":
        boundary_calc = f"Time at instantaneous μ > {lag_cutoff:.0%} μ<sub>max</sub>"
        exp_phase_end_calc = (
            f"Time at instantaneous μ < {exp_cutoff:.0%} μ<sub>max</sub>"
        )
    else:
        boundary_calc = "μ<sub>max</sub> tangent intersect with OD baseline"
        exp_phase_end_calc = "μ<sub>max</sub> tangent intersec with OD(max)"

    if growth_method == "Model Fitting" and model_type in {
        "mech_baranyi",
        "phenom_logistic",
        "phenom_gompertz",
        "phenom_gompertz_modified",
        "phenom_richards",
    }:
        lag_time_calc = "λ"
    else:
        lag_time_calc = boundary_calc

    st.caption("Your selected settings will calculate growth parameters as follows:")
    # Apply styling (moved to styling.py)
    growth_param_table_style()
    st.markdown(
        f"""
<div class="growth-param-table">

| OD(max) | μ<sub>max</sub> | Intrinsic Growth Rate | Doubling Time | Lag Time | μ<sub>max</sub> Time | μ<sub>max</sub> OD | Exponential End Time | RMSE |
|---|---|---|---|---|---|---|---|---|
| {max_od_calc} | {mu_max_calc} | {intrinsic_calc} | ln(2) / μ<sub>max</sub> | {lag_time_calc} | Time at μ<sub>max</sub> | OD at μ<sub>max</sub> | {exp_phase_end_calc} | {model_rmse_calc} |

</div>
""",
        unsafe_allow_html=True,
    )


@st.fragment
def ui_analysis_params(ss):
    """Fragment for analysis parameters and the analyse button."""
    step3_params = ss.get("step3_params", {})
    step4_prev = ss.get("step4_params", {})
    params0 = step3_params.get("params0", DEFAULT_PARAMS)

    # Step 3 preprocessing values needed to build the final params dict
    plate_id = step3_params.get("plate_id")
    time_unit = step3_params.get("time_unit", "hours")
    pl_cm = step3_params.get("pl_cm", 0.42)
    blank = step3_params.get("blank", True)
    blank_group_assignments = step3_params.get("blank_group_assignments", False)
    outlier_detection = bool(
        step3_params.get("outlier_detection", params0.get("outlier_detection", False))
    )
    try:
        outlier_threshold = float(
            step3_params.get("outlier_threshold", params0.get("outlier_threshold", 3.5))
        )
    except (TypeError, ValueError):
        outlier_threshold = 3.5
    clip_time_series = step3_params.get("clip_time_series", (None, None))
    remove_wells = step3_params.get("remove_wells", False)

    with st.container(border=True):
        st.header("Step 5. Select the analysis parameters")

        # Two columns: Model options | Phase boundary options
        model_col, boundary_col = st.columns(2, gap="large")

        with model_col:
            # Model selection
            model_family, growth_method, model_type, param_col = _ui_model_selection(
                params0
            )

            # Method-specific parameters
            window_points, smooth = ui_model_params(
                growth_method, params0, step4_prev, param_col
            )

            st.write("")
            st.write("")

            # Quality control filters
            min_data_points, min_signal_to_noise, min_od_increase, min_growth_rate = (
                ui_qc_filters(params0)
            )

            st.write("")

        with boundary_col:
            # Phase boundary selection
            phase_boundary_method, lag_cutoff, exp_cutoff = ui_phase_boundaries(params0)
            st.write("")

        st.write("")

        # Visualization columns
        help_model_col, help_boundary_col = st.columns(2, gap="large")

        with help_model_col:
            model_fig = ui_method_visualization(growth_method, model_type)

        with help_boundary_col:
            boundary_image = ui_phase_boundary_visualization(phase_boundary_method)

        # Render the visualizations
        graph_col_model, graph_col_boundary = st.columns(2, gap="large")

        with graph_col_model:
            if isinstance(model_fig, str):
                st.image(model_fig, width="stretch")
            elif model_fig is not None:
                st.plotly_chart(model_fig, width="stretch", config={"staticPlot": True})

        with graph_col_boundary:
            if boundary_image is not None:
                st.image(boundary_image, width="stretch")

    # Build final params dict
    params = dict(
        time_unit=str(time_unit),
        pathlength_cm_=float(pl_cm),
        clip_time_series=clip_time_series,
        remove_wells=remove_wells,
        blank=bool(blank),
        blank_group_assignments=blank_group_assignments if blank else False,
        outlier_detection=bool(outlier_detection),
        outlier_threshold=float(outlier_threshold),
        window_points=int(window_points),
        lag_cutoff=float(lag_cutoff),
        exp_cutoff=float(exp_cutoff),
        sg_window=int(params0.get("sg_window", 15)),
        sg_poly=int(params0.get("sg_poly", 2)),
        min_data_points=int(min_data_points),
        min_signal_to_noise=float(min_signal_to_noise),
        min_od_increase=float(min_od_increase),
        min_growth_rate=float(min_growth_rate),
        growth_method=str(growth_method),
        model_family=str(model_family),
        model_type=str(model_type),
        phase_boundary_method=str(phase_boundary_method),
        smooth=str(smooth),
    )

    with st.container(border=True):
        st.header("Step 6. Click analyse")

        ui_calculation_table(
            growth_method,
            model_type,
            model_family,
            phase_boundary_method,
            lag_cutoff,
            exp_cutoff,
            window_points,
        )

        st.write("")
        clicked = st.button(
            "Update parameters and analyse selected plate",
            type="primary",
            width="stretch",
            disabled=not plate_id,
        )
        if clicked:
            rec = ss.plates.get(plate_id, {})
            if not rec.get("uploads"):
                st.error("No uploads found for this plate.")
            else:
                rec["params"] = params
                st.markdown(
                    """
<style>
[data-testid="stSpinner"] > div {
    flex-direction: row;
    align-items: center;
    gap: 0.5rem;
}
</style>
""",
                    unsafe_allow_html=True,
                )
                with st.spinner("Analysing plate...", show_time=True, width="stretch"):
                    ss.plates[plate_id] = analyse_plate(rec)
                st.toast(f"Analysed {plate_id}", duration="infinite")

    # Store analysis parameters in session state
    ss.setdefault("step4_params", {})
    ss["step4_params"]["window_points"] = window_points
    ss["step4_params"]["lag_cutoff"] = lag_cutoff
    ss["step4_params"]["exp_cutoff"] = exp_cutoff
    ss["step4_params"]["min_data_points"] = min_data_points
    ss["step4_params"]["min_signal_to_noise"] = min_signal_to_noise
    ss["step4_params"]["min_od_increase"] = min_od_increase
    ss["step4_params"]["min_growth_rate"] = min_growth_rate
    ss["step4_params"]["growth_method"] = growth_method
    ss["step4_params"]["model_family"] = model_family
    ss["step4_params"]["model_type"] = model_type
    ss["step4_params"]["phase_boundary_method"] = phase_boundary_method
    ss["step4_params"]["smooth"] = smooth
