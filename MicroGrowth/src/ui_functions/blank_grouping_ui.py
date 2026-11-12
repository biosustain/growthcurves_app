"""UI helpers for assigning blank-subtraction analysis groups."""

from __future__ import annotations

import re
from typing import Any

import streamlit as st
from src.functions.constants import PlateFormat
from src.functions.data_processing import group_sort_key
from src.ui_functions.plate_painter import plate_painter

DEFAULT_GROUP = "Group 1"
# Permanent selector entry: painting with it marks wells as blanks (they keep their group)
BLANK_OPTION = "Assign blank wells"
GROUP_PALETTE = [
    "#ccebc5",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ffed6f",
]


def _state_key(prefix: str, suffix: str) -> str:
    return f"{prefix}::{suffix}"


def group_number(name: str) -> int:
    match = re.search(r"(\d+)", str(name))
    return int(match.group(1)) if match else 1


def next_group_name(groups: list[str]) -> str:
    if not groups:
        return DEFAULT_GROUP
    used = {group_number(g) for g in groups}
    n = 1
    while n in used:
        n += 1
    return f"Group {n}"


def color_for_group(group_name: str) -> str:
    return GROUP_PALETTE[(group_number(group_name) - 1) % len(GROUP_PALETTE)]


def darken_hex_color(color: str, factor: float = 0.88) -> str:
    match = re.fullmatch(r"#([0-9a-fA-F]{6})", str(color).strip())
    if not match:
        return str(color).strip() or "#ffffff"
    h = match.group(1)
    r, g, b = (max(0, min(255, int(int(h[i : i + 2], 16) * factor))) for i in (0, 2, 4))
    return f"#{r:02x}{g:02x}{b:02x}"


def fill_rect(
    assignments: list[list[str]], p1: dict[str, int], p2: dict[str, int], value: str
) -> list[list[str]]:
    left, right = sorted((p1["x"], p2["x"]))
    top, bottom = sorted((p1["y"], p2["y"]))
    out = [row[:] for row in assignments]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            out[r][c] = value
    return out


def _assignments_from_map(
    group_map: dict[str, str] | None, fmt: PlateFormat
) -> list[list[str]]:
    grid = [[DEFAULT_GROUP] * len(fmt.cols) for _ in fmt.rows]
    if isinstance(group_map, dict):
        for well, group in group_map.items():
            pt = fmt.point(well)
            if pt:
                grid[pt[0]][pt[1]] = str(group).strip() or DEFAULT_GROUP
    return grid


def _grid_fits(grid, fmt: PlateFormat) -> bool:
    """True if a stored grid has the plate's shape (False after a format change)."""
    return len(grid) == len(fmt.rows) and all(len(r) == len(fmt.cols) for r in grid)


def assignments_to_map(
    assignments: list[list[str]], fmt: PlateFormat
) -> dict[str, str]:
    return {
        fmt.well(y, x): str(group)
        for y, row in enumerate(assignments)
        for x, group in enumerate(row)
    }


def _group_names_from_assignments(assignments: list[list[str]]) -> list[str]:
    groups = sorted(
        {str(cell).strip() or DEFAULT_GROUP for row in assignments for cell in row},
        key=group_sort_key,
    )
    if DEFAULT_GROUP not in groups:
        groups.insert(0, DEFAULT_GROUP)
    return groups


def _normalize_group_labels(
    assignments: list[list[str]], groups: list[str]
) -> tuple[list[list[str]], list[str], dict[str, str]]:
    """Normalize group labels to contiguous names: Group 1..Group n."""
    all_groups = sorted(
        {str(c).strip() or DEFAULT_GROUP for row in assignments for c in row}
        | {str(g).strip() or DEFAULT_GROUP for g in groups},
        key=group_sort_key,
    )
    if DEFAULT_GROUP in all_groups:
        all_groups.remove(DEFAULT_GROUP)
    all_groups.insert(0, DEFAULT_GROUP)
    rename_map = {old: f"Group {i}" for i, old in enumerate(all_groups, 1)}
    normalized = [
        [rename_map.get(str(c).strip() or DEFAULT_GROUP, DEFAULT_GROUP) for c in row]
        for row in assignments
    ]
    return normalized, [rename_map[g] for g in all_groups], rename_map


def _painter_cells(
    assignments: list[list[str]],
    color_map: dict[str, str],
    fmt: PlateFormat,
    name_by_well: dict[str, str] | None = None,
    present_wells: set[str] | None = None,
    remove_wells: list[str] | set[str] | bool = False,
    blank_enabled: bool = True,
    painted_blanks: set[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """Cells for the drag-to-paint plate map; excluded wells are grey."""
    name_by_well = name_by_well or {}
    painted_blanks = painted_blanks or set()
    present_lookup = (
        {str(w).strip().upper() for w in present_wells}
        if present_wells is not None
        else None
    )
    removed_lookup = {str(w).strip().upper() for w in (remove_wells or [])}
    status_mode = (
        present_lookup is not None or bool(removed_lookup) or not blank_enabled
    )

    rows = []
    for y, row in enumerate(assignments):
        out_row = []
        for x, group in enumerate(row):
            well = fmt.well(y, x)
            sample = str(name_by_well.get(well, "")).strip()
            fixed = sample.upper().startswith("BLANK")
            painted = well in painted_blanks
            included = not status_mode or (
                well not in removed_lookup
                and sample not in {"", "False"}
                and (present_lookup is None or well in present_lookup)
                and (blank_enabled or not (fixed or painted))
            )
            blank = included and (fixed or painted)
            base = color_map.get(group, "#ffffff")
            out_row.append(
                {
                    "label": well,
                    "color": (darken_hex_color(base) if blank else base)
                    if included
                    else "#e5e7eb",
                    "border": darken_hex_color(base, factor=0.72) if blank else None,
                    # Hover shows just the sample name
                    "tooltip": "" if sample == "False" else sample,
                    "group": group,
                    "blank": blank,
                    "fixed_blank": fixed,
                    "painted": painted,
                    "included": included,
                }
            )
        rows.append(out_row)
    return rows


def _apply_paint(prefix: str, painter_key: str, fmt: PlateFormat) -> None:
    """Assign the wells of a finished drag stroke (runs before the script body)."""
    ss = st.session_state
    stroke = getattr(ss.get(painter_key), "painted", None)
    if not stroke:
        return
    rows = range(int(stroke["r0"]), int(stroke["r1"]) + 1)
    cols = range(int(stroke["c0"]), int(stroke["c1"]) + 1)
    if ss[_state_key(prefix, "active_group")] == BLANK_OPTION:
        bk = _state_key(prefix, "painted_blanks")
        wells = {fmt.well(y, x) for y in rows for x in cols}
        ss[bk] = sorted(
            set(ss[bk]) - wells if stroke.get("erase") else set(ss[bk]) | wells
        )
        return
    ak = _state_key(prefix, "assignments")
    value = (
        DEFAULT_GROUP if stroke.get("erase") else ss[_state_key(prefix, "active_group")]
    )
    ss[ak] = fill_rect(
        ss[ak],
        {"x": int(stroke["c0"]), "y": int(stroke["r0"])},
        {"x": int(stroke["c1"]), "y": int(stroke["r1"])},
        value,
    )


def painted_blank_wells(plate_id: str, default: list[str] | None = None) -> list[str]:
    """Wells the user painted as blanks for a plate (default before first render)."""
    key = _state_key(f"blank_groups::{plate_id}", "painted_blanks")
    if key in st.session_state:
        return list(st.session_state[key])
    return sorted({str(w).strip().upper() for w in (default or [])})


def _init_state(
    prefix: str,
    fmt: PlateFormat,
    initial_group_map: dict[str, str] | None,
    initial_blank_wells: list[str] | None = None,
):
    ss = st.session_state
    bk = _state_key(prefix, "painted_blanks")
    if bk not in ss:
        ss[bk] = sorted({str(w).strip().upper() for w in (initial_blank_wells or [])})
    ak = _state_key(prefix, "assignments")
    gk = _state_key(prefix, "groups")
    ck = _state_key(prefix, "group_colors")
    actk = _state_key(prefix, "active_group")

    if ak not in ss or not _grid_fits(ss[ak], fmt):
        ss[ak] = _assignments_from_map(initial_group_map, fmt)
    if gk not in ss or not ss[gk]:
        ss[gk] = _group_names_from_assignments(ss[ak]) or [DEFAULT_GROUP]
    if actk not in ss or (ss[actk] not in ss[gk] and ss[actk] != BLANK_OPTION):
        ss[actk] = DEFAULT_GROUP if DEFAULT_GROUP in ss[gk] else ss[gk][0]

    current_active = str(ss[actk]).strip() or DEFAULT_GROUP
    ss[ak], ss[gk], rename_map = _normalize_group_labels(ss[ak], ss[gk])
    ss[ck] = {g: color_for_group(g) for g in ss[gk]}
    ss[actk] = (
        BLANK_OPTION
        if current_active == BLANK_OPTION
        else rename_map.get(current_active, DEFAULT_GROUP)
    )


def _add_group(prefix: str) -> None:
    """Add a group and select it (button callback)."""
    ss = st.session_state
    gk, ck = _state_key(prefix, "groups"), _state_key(prefix, "group_colors")
    new_group = next_group_name(ss[gk])
    ss[gk].append(new_group)
    ss[ck][new_group] = color_for_group(new_group)
    ss[_state_key(prefix, "active_group")] = new_group
    ss[_state_key(prefix, "assigned_group_select")] = new_group


def _remove_group(prefix: str) -> None:
    """Remove the selected group; its wells return to the default group (button callback)."""
    ss = st.session_state
    gk, ck = _state_key(prefix, "groups"), _state_key(prefix, "group_colors")
    ak, actk = _state_key(prefix, "assignments"), _state_key(prefix, "active_group")
    old = ss[actk]
    ss[gk].remove(old)
    ss[ak] = [[DEFAULT_GROUP if c == old else c for c in row] for row in ss[ak]]
    ss[ck].pop(old, None)
    ss[actk] = max(ss[gk], key=group_sort_key) if ss[gk] else DEFAULT_GROUP
    ss[_state_key(prefix, "assigned_group_select")] = ss[actk]


def ui_blank_group_assigner(
    *,
    plate_id: str,
    fmt: PlateFormat,
    initial_group_map: dict[str, str] | None = None,
    name_by_well: dict[str, str] | None = None,
    present_wells: set[str] | None = None,
    remove_wells: list[str] | set[str] | bool = False,
    blank_enabled: bool = True,
    initial_blank_wells: list[str] | None = None,
    show_caption: bool = True,
    show_controls: bool = True,
    controls_disabled: bool = False,
    show_grid: bool = True,
    grid_aspect_ratio: float = 1.0,
) -> dict[str, str]:
    """Render the analysis-group assignment UI and return well->group mapping."""
    prefix = f"blank_groups::{plate_id}"
    _init_state(prefix, fmt, initial_group_map, initial_blank_wells)
    ss = st.session_state

    gk = _state_key(prefix, "groups")
    ck = _state_key(prefix, "group_colors")
    ak = _state_key(prefix, "assignments")
    actk = _state_key(prefix, "active_group")

    help_caption = (
        "When you have more than one blank, you may want to subtract certain blanks from only "
        "certain wells on the plate. Click and drag across the plate map to link blanks and "
        "sample wells by assigning them the same colour group (click a row letter or column "
        "number to select a whole row or column; start on a well already in the selected group "
        "to remove wells from it). "
        "Blanks are then subtracted from samples in the same colour group. "
        "Note: plate groups are independent, meaning blanks and samples "
        "on different plates are never linked. Multiple blanks in the same group will be "
        "averaged at each time point before subtraction."
    )

    if show_controls:
        if show_caption:
            with st.popover("How blank groups work", width="stretch"):
                st.markdown(help_caption)
                st.image("MicroGrowth/info_plots/blank_group.png")
                st.caption(
                    "This example shows a plate with four different blank wells (darker wells) "
                    'identified in the plate map where the well names start with "BLANK_". '
                    "Four coloured blank groups have been created and assigned to wells with "
                    "one blank per group. The blank will be subtracted from "
                    "samples in the same group only - e.g. A3 will be subtracted from green samples"
                    " in columns 1, 2 and 3."
                )

        group_col, color_col, add_col, remove_col = st.columns(
            [2.5, 0.75, 1.5, 1.5], vertical_alignment="bottom"
        )
        select_key = _state_key(prefix, "assigned_group_select")
        options = [*ss[gk], BLANK_OPTION]
        if ss.get(select_key) not in options:
            ss[select_key] = ss[actk]
        selected_group = group_col.selectbox(
            "Blank group",
            options,
            key=select_key,
            help=(
                "Pick a group and drag across the plate map to assign wells to it. "
                f'Pick "{BLANK_OPTION}" to mark wells as blanks; they keep their group.'
            ),
        )
        if selected_group != ss[actk]:
            ss[actk] = selected_group

        if ss[actk] != BLANK_OPTION:
            color_col.markdown(
                "<button type='button' onclick='return false;' "
                "style='display:block;width:35px;height:35px;margin:0 auto;"
                f"background:{ss[ck].get(ss[actk], '#ffffff')};border:1px solid #999;"
                "border-radius:4px;cursor:default;'></button>",
                unsafe_allow_html=True,
            )

        _disabled_help = (
            "When there are fewer than two blanks, there can be only one blank group."
            if controls_disabled
            else None
        )
        add_col.button(
            "Add group",
            type="primary",
            width="stretch",
            key=_state_key(prefix, "add_group"),
            disabled=controls_disabled,
            help=_disabled_help,
            on_click=_add_group,
            args=(prefix,),
        )
        remove_col.button(
            "Remove",
            type="primary",
            width="stretch",
            key=_state_key(prefix, "remove_group"),
            disabled=controls_disabled or ss[actk] in (DEFAULT_GROUP, BLANK_OPTION),
            help=_disabled_help,
            on_click=_remove_group,
            args=(prefix,),
        )

    if show_grid:
        painter_key = _state_key(prefix, "plate_painter")
        plate_painter(
            rows=fmt.rows,
            cols=fmt.cols,
            cells=_painter_cells(
                ss[ak],
                ss[ck],
                fmt,
                name_by_well,
                present_wells=present_wells,
                remove_wells=remove_wells,
                blank_enabled=blank_enabled,
                painted_blanks=set(ss[_state_key(prefix, "painted_blanks")]),
            ),
            group_colors=ss[ck],
            active_group=ss[actk],
            default_group=DEFAULT_GROUP,
            blank_option=BLANK_OPTION,
            key=painter_key,
            on_paint=lambda: _apply_paint(prefix, painter_key, fmt),
            aspect_ratio=grid_aspect_ratio,
        )

    return assignments_to_map(ss[ak], fmt)


# ---------------------------------------------------------------------------
# Single-group well selector (used on the download page)
# ---------------------------------------------------------------------------

_WELL_SEL_INCLUDED = "included"
_WELL_SEL_EXCLUDED = "excluded"
_WELL_SEL_COLORS = {_WELL_SEL_INCLUDED: "#ccebc5", _WELL_SEL_EXCLUDED: "#fb8072"}


def _init_well_selector_state(
    prefix: str, available_set: set[str], fmt: PlateFormat
) -> None:
    ss = st.session_state
    ak = _state_key(prefix, "grid")
    if ak not in ss or not _grid_fits(ss[ak], fmt):
        ss[ak] = [
            [
                (
                    _WELL_SEL_INCLUDED
                    if fmt.well(y, x) in available_set
                    else _WELL_SEL_EXCLUDED
                )
                for x in range(len(fmt.cols))
            ]
            for y in range(len(fmt.rows))
        ]


def _fill_rect_available(
    grid: list[list[str]],
    p1: dict[str, int],
    p2: dict[str, int],
    value: str,
    available_set: set[str],
    fmt: PlateFormat,
) -> list[list[str]]:
    left, right = sorted((p1["x"], p2["x"]))
    top, bottom = sorted((p1["y"], p2["y"]))
    out = [row[:] for row in grid]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            well = fmt.well(r, c)
            if well in available_set:
                out[r][c] = value
    return out


def _apply_well_selector_paint(
    prefix: str, painter_key: str, available_set: set[str], fmt: PlateFormat
):
    """Include/exclude the wells of a finished drag stroke (runs before the script body)."""
    ss = st.session_state
    stroke = getattr(ss.get(painter_key), "painted", None)
    if not stroke:
        return
    ak = _state_key(prefix, "grid")
    ss[ak] = _fill_rect_available(
        ss[ak],
        {"x": int(stroke["c0"]), "y": int(stroke["r0"])},
        {"x": int(stroke["c1"]), "y": int(stroke["r1"])},
        _WELL_SEL_EXCLUDED if stroke.get("erase") else _WELL_SEL_INCLUDED,
        available_set,
        fmt,
    )


def ui_well_selector(
    *,
    plate_id: str,
    fmt: PlateFormat,
    available_wells: list[str],
    grid_aspect_ratio: float = 0.75,
) -> list[str]:
    """Render a drag-to-paint well selector and return the list of included wells.

    Green = included · red = excluded · grey = no data. Dragging from an included
    well excludes the wells covered; dragging from an excluded well includes them.
    """
    prefix = f"well_selector::{plate_id}"
    available_set = {str(w).strip().upper() for w in available_wells}
    _init_well_selector_state(prefix, available_set, fmt)
    ss = st.session_state
    ak = _state_key(prefix, "grid")

    # Wells without data are grey and cannot be changed
    cells = [
        [
            {
                "label": well,
                "color": _WELL_SEL_COLORS[state],
                "tooltip": f"{well} · {state}",
                "group": state,
                "included": True,
            }
            if (well := fmt.well(y, x)) in available_set
            else {"label": well, "color": "#d1d5db", "tooltip": f"{well} · no data"}
            for x, state in enumerate(row)
        ]
        for y, row in enumerate(ss[ak])
    ]
    painter_key = _state_key(prefix, "plate_painter")
    plate_painter(
        rows=fmt.rows,
        cols=fmt.cols,
        cells=cells,
        group_colors=_WELL_SEL_COLORS,
        active_group=_WELL_SEL_INCLUDED,
        default_group=_WELL_SEL_EXCLUDED,
        blank_option="__none__",
        key=painter_key,
        on_paint=lambda: _apply_well_selector_paint(
            prefix, painter_key, available_set, fmt
        ),
        aspect_ratio=grid_aspect_ratio,
    )

    return [
        fmt.well(y, x)
        for y, row in enumerate(ss[ak])
        for x, state in enumerate(row)
        if state == _WELL_SEL_INCLUDED and fmt.well(y, x) in available_set
    ]


def get_well_selector_wells(
    plate_id: str, available_wells: list[str], fmt: PlateFormat
) -> list[str] | None:
    """Return the stored well selection for a plate without rendering the grid.

    Returns the list of included wells if the plate has been visited (i.e. its
    grid state exists in session state), or None if it has never been shown
    (caller should default to all available wells).
    """
    ak = _state_key(f"well_selector::{plate_id}", "grid")
    if ak not in st.session_state or not _grid_fits(st.session_state[ak], fmt):
        return None
    available_set = {str(w).strip().upper() for w in available_wells}
    return [
        fmt.well(y, x)
        for y, row in enumerate(st.session_state[ak])
        for x, state in enumerate(row)
        if state == _WELL_SEL_INCLUDED and fmt.well(y, x) in available_set
    ]
