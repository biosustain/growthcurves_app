"""UI helpers for assigning blank-subtraction analysis groups."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st
from src.functions.constants import COLS, ROWS

try:
    from st_selectable_grid import st_selectable_grid
except ImportError:  # pragma: no cover - optional dependency
    st_selectable_grid = None


DEFAULT_GROUP = "Group 1"
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


def _group_sort_key(name: str) -> tuple[int, str]:
    return group_number(name), str(name)


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


def _well_to_point(well: str) -> dict[str, int] | None:
    text = str(well).strip().upper()
    if len(text) < 2 or text[0] not in ROWS:
        return None
    try:
        col_num = int(text[1:])
    except ValueError:
        return None
    if col_num not in COLS:
        return None
    return {"x": COLS.index(col_num), "y": ROWS.index(text[0])}


def _assignments_from_map(group_map: dict[str, str] | None) -> list[list[str]]:
    grid = [[DEFAULT_GROUP] * len(COLS) for _ in ROWS]
    if isinstance(group_map, dict):
        for well, group in group_map.items():
            pt = _well_to_point(str(well))
            if pt:
                grid[pt["y"]][pt["x"]] = str(group).strip() or DEFAULT_GROUP
    return grid


def assignments_to_map(assignments: list[list[str]]) -> dict[str, str]:
    return {
        f"{row}{col}": str(assignments[y][x])
        for y, row in enumerate(ROWS)
        for x, col in enumerate(COLS)
    }


def _group_names_from_assignments(assignments: list[list[str]]) -> list[str]:
    groups = sorted(
        {str(cell).strip() or DEFAULT_GROUP for row in assignments for cell in row},
        key=_group_sort_key,
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
        key=_group_sort_key,
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


def build_cells(
    assignments: list[list[str]],
    color_map: dict[str, str],
    name_by_well: dict[str, str] | None = None,
    present_wells: set[str] | None = None,
    remove_wells: list[str] | set[str] | bool = False,
    blank_enabled: bool = True,
) -> list[list[dict[str, Any]]]:
    name_by_well = name_by_well or {}
    present_lookup = (
        {str(w).strip().upper() for w in present_wells}
        if present_wells is not None
        else None
    )
    removed_lookup = {str(w).strip().upper() for w in (remove_wells or [])}
    status_mode = (
        present_lookup is not None or bool(removed_lookup) or not blank_enabled
    )

    rows: list[list[dict[str, Any]]] = []
    for y, row in enumerate(assignments):
        rendered_row = []
        for x, group in enumerate(row):
            well = f"{ROWS[y]}{COLS[x]}"
            sample = str(name_by_well.get(well, "")).strip()
            is_blank = sample.upper().startswith("BLANK")
            has_sample = sample not in {"", "False"} and not sample.upper().startswith(
                "BLANK"
            )
            not_in_map = sample in {"", "False"}
            base_color = color_map.get(group, "#ffffff")
            sample_suffix = f" · {sample}" if sample and sample != "False" else ""

            included = True
            exclusion_reason = ""
            if status_mode:
                included = False
                if well in removed_lookup:
                    exclusion_reason = "excluded by user"
                elif not_in_map:
                    exclusion_reason = "not in plate map"
                elif present_lookup is not None and well not in present_lookup:
                    exclusion_reason = "missing from data file"
                elif is_blank and not blank_enabled:
                    exclusion_reason = "blank subtraction disabled"
                elif is_blank or has_sample:
                    included = True
                else:
                    exclusion_reason = "not included"

            label = f"<b>{well}</b>" if is_blank else well
            if included:
                blank_tag = " · BLANK well" if is_blank and status_mode else ""
                cell_data: dict[str, Any] = {
                    "label": label,
                    "cell_color": (
                        darken_hex_color(base_color) if is_blank else base_color
                    ),
                    "tooltip": f"{well}{sample_suffix}{blank_tag} · {group}",
                }
                if is_blank:
                    cell_data.update(
                        {
                            "html": True,
                            "cell_border_width": 2,
                            "cell_border_color": darken_hex_color(
                                base_color, factor=0.72
                            ),
                        }
                    )
            else:
                cell_data = {
                    "label": label,
                    "cell_color": "#e5e7eb",
                    "tooltip": f"{well}{sample_suffix} · {exclusion_reason}",
                    **({"html": True} if is_blank else {}),
                }
            rendered_row.append(cell_data)
        rows.append(rendered_row)
    return rows


def _reset_pending_selection(prefix: str):
    ss = st.session_state
    ss[_state_key(prefix, "first_corner")] = None
    ss[_state_key(prefix, "awaiting_second")] = False
    ss[_state_key(prefix, "pending_clear_mode")] = None


def _init_state(prefix: str, initial_group_map: dict[str, str] | None):
    ss = st.session_state
    ak = _state_key(prefix, "assignments")
    gk = _state_key(prefix, "groups")
    ck = _state_key(prefix, "group_colors")
    actk = _state_key(prefix, "active_group")

    if ak not in ss:
        ss[ak] = _assignments_from_map(initial_group_map)
    if gk not in ss or not ss[gk]:
        ss[gk] = _group_names_from_assignments(ss[ak]) or [DEFAULT_GROUP]
    if actk not in ss or ss[actk] not in ss[gk]:
        ss[actk] = DEFAULT_GROUP if DEFAULT_GROUP in ss[gk] else ss[gk][0]

    current_active = str(ss[actk]).strip() or DEFAULT_GROUP
    ss[ak], ss[gk], rename_map = _normalize_group_labels(ss[ak], ss[gk])
    ss[ck] = {g: color_for_group(g) for g in ss[gk]}
    ss[actk] = rename_map.get(current_active, DEFAULT_GROUP)

    ss.setdefault(_state_key(prefix, "first_corner"), None)
    ss.setdefault(_state_key(prefix, "awaiting_second"), False)
    ss.setdefault(_state_key(prefix, "pending_clear_mode"), None)
    ss.setdefault(_state_key(prefix, "last_processed_click"), None)


def _render_fallback_assigner(prefix: str, grid_height: int = 350):
    ss = st.session_state
    ak = _state_key(prefix, "assignments")
    actk = _state_key(prefix, "active_group")

    st.caption(
        "`st_selectable_grid` is not installed. Using range controls for group assignment."
    )

    all_wells = [f"{r}{c}" for r in ROWS for c in COLS]
    c1, c2 = st.columns(2)
    p1 = _well_to_point(
        c1.selectbox("Start well", all_wells, key=_state_key(prefix, "range_start"))
    ) or {"x": 0, "y": 0}
    p2 = _well_to_point(
        c2.selectbox("End well", all_wells, key=_state_key(prefix, "range_end"))
    ) or {"x": 0, "y": 0}

    a, b = st.columns(2)
    if a.button(
        "Assign range",
        type="primary",
        width="stretch",
        key=_state_key(prefix, "assign_range"),
    ):
        ss[ak] = fill_rect(ss[ak], p1, p2, ss[actk])
        st.rerun()
    if b.button(
        "Clear range",
        type="primary",
        width="stretch",
        key=_state_key(prefix, "clear_range"),
    ):
        ss[ak] = fill_rect(ss[ak], p1, p2, DEFAULT_GROUP)
        st.rerun()

    st.dataframe(
        pd.DataFrame(ss[ak], index=ROWS, columns=COLS),
        width="stretch",
        height=grid_height,
    )


def ui_blank_group_assigner(
    *,
    plate_id: str,
    initial_group_map: dict[str, str] | None = None,
    name_by_well: dict[str, str] | None = None,
    present_wells: set[str] | None = None,
    remove_wells: list[str] | set[str] | bool = False,
    blank_enabled: bool = True,
    show_caption: bool = True,
    show_controls: bool = True,
    controls_disabled: bool = False,
    show_grid: bool = True,
    grid_height: int = 440,
    grid_aspect_ratio: float = 1.0,
) -> dict[str, str]:
    """Render the analysis-group assignment UI and return well->group mapping."""
    prefix = f"blank_groups::{plate_id}"
    _init_state(prefix, initial_group_map)
    ss = st.session_state

    gk = _state_key(prefix, "groups")
    ck = _state_key(prefix, "group_colors")
    ak = _state_key(prefix, "assignments")
    actk = _state_key(prefix, "active_group")
    await_k = _state_key(prefix, "awaiting_second")
    first_k = _state_key(prefix, "first_corner")
    clear_k = _state_key(prefix, "pending_clear_mode")
    last_click_k = _state_key(prefix, "last_processed_click")

    help_caption = (
        "When you have more than one blank, you may want to subtract certain blanks from only "
        "certain wells on the plate. Click the table to link blanks and sample wells by assigning"
        " them the same colour group. "
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
        selected_group = group_col.selectbox(
            "Blank group",
            ss[gk],
            index=ss[gk].index(ss[actk]),
            key=_state_key(prefix, "assigned_group_select"),
            disabled=controls_disabled,
        )
        if selected_group != ss[actk]:
            ss[actk] = selected_group
            _reset_pending_selection(prefix)

        with color_col:
            st.markdown(
                "<button type='button' onclick='return false;' "
                "style='display:block;width:35px;height:35px;margin:0 auto;"
                f"background:{ss[ck][ss[actk]]};border:1px solid #999;"
                "border-radius:4px;cursor:default;'></button>",
                unsafe_allow_html=True,
            )

        _disabled_help = (
            "When there are fewer than two blanks, there can be only one blank group."
            if controls_disabled
            else None
        )
        add_clicked = add_col.button(
            "Add",
            type="primary",
            width="stretch",
            key=_state_key(prefix, "add_group"),
            disabled=controls_disabled,
            help=_disabled_help,
        )
        remove_clicked = remove_col.button(
            "Remove",
            type="primary",
            width="stretch",
            key=_state_key(prefix, "remove_group"),
            disabled=controls_disabled or ss[actk] == DEFAULT_GROUP,
            help=_disabled_help,
        )

        if add_clicked:
            _reset_pending_selection(prefix)
            new_group = next_group_name(ss[gk])
            ss[gk].append(new_group)
            ss[ck][new_group] = color_for_group(new_group)
            ss[actk] = new_group
            st.rerun()

        if remove_clicked:
            _reset_pending_selection(prefix)
            old = ss[actk]
            ss[gk].remove(old)
            ss[ak] = [[DEFAULT_GROUP if c == old else c for c in row] for row in ss[ak]]
            ss[ck].pop(old, None)
            ss[actk] = max(ss[gk], key=_group_sort_key) if ss[gk] else DEFAULT_GROUP
            st.rerun()

    if show_grid:
        if st_selectable_grid is None:
            _render_fallback_assigner(prefix, grid_height=grid_height)
        else:
            selection = st_selectable_grid(
                cells=build_cells(
                    ss[ak],
                    ss[ck],
                    name_by_well,
                    present_wells=present_wells,
                    remove_wells=remove_wells,
                    blank_enabled=blank_enabled,
                ),
                header=[str(c) for c in COLS],
                index=ROWS,
                aspect_ratio=grid_aspect_ratio,
                allow_secondary_selection=False,
                allow_header_selection=False,
                resize=True,
                height=grid_height,
                primary_selection_color="#2563eb",
                key=_state_key(prefix, "well_grid"),
            )

            current_click = (selection or {}).get("primary")
            current_click_key = (
                (current_click["x"], current_click["y"]) if current_click else None
            )

            if current_click_key is None and not ss[await_k]:
                # Allows re-using the same well as the next "first click" after deselection.
                ss[last_click_k] = None

            if not ss[await_k]:
                if current_click and current_click_key != ss[last_click_k]:
                    x, y = current_click["x"], current_click["y"]
                    ss[clear_k] = ss[ak][y][x] == ss[actk]
                    ss[first_k] = current_click
                    ss[await_k] = True
                    ss[last_click_k] = current_click_key
            else:
                first_corner = ss[first_k]
                if current_click is None:
                    # Allow 1x1 rectangles: second click can deselect the first-corner cell.
                    value = DEFAULT_GROUP if ss[clear_k] else ss[actk]
                    ss[ak] = fill_rect(ss[ak], first_corner, first_corner, value)
                    _reset_pending_selection(prefix)
                    st.rerun()
                elif current_click_key != ss[last_click_k]:
                    value = DEFAULT_GROUP if ss[clear_k] else ss[actk]
                    ss[ak] = fill_rect(ss[ak], first_corner, current_click, value)
                    _reset_pending_selection(prefix)
                    ss[last_click_k] = current_click_key
                    st.rerun()
    return assignments_to_map(ss[ak])


# ---------------------------------------------------------------------------
# Single-group well selector (used on the download page)
# ---------------------------------------------------------------------------

_WELL_SEL_INCLUDED = "included"
_WELL_SEL_EXCLUDED = "excluded"


def _init_well_selector_state(prefix: str, available_set: set[str]) -> None:
    ss = st.session_state
    ak = _state_key(prefix, "grid")
    if ak not in ss:
        ss[ak] = [
            [
                (
                    _WELL_SEL_INCLUDED
                    if f"{ROWS[y]}{COLS[x]}" in available_set
                    else _WELL_SEL_EXCLUDED
                )
                for x in range(len(COLS))
            ]
            for y in range(len(ROWS))
        ]
    ss.setdefault(_state_key(prefix, "first_corner"), None)
    ss.setdefault(_state_key(prefix, "awaiting_second"), False)
    ss.setdefault(_state_key(prefix, "pending_clear_mode"), None)
    ss.setdefault(_state_key(prefix, "last_processed_click"), None)


def _build_well_selector_cells(
    grid: list[list[str]],
    available_set: set[str],
    name_by_well: dict[str, str] | None = None,
) -> list[list[dict[str, Any]]]:
    name_by_well = name_by_well or {}
    rows: list[list[dict[str, Any]]] = []
    for y, row in enumerate(grid):
        rendered_row: list[dict[str, Any]] = []
        for x, state in enumerate(row):
            well = f"{ROWS[y]}{COLS[x]}"
            sample = str(name_by_well.get(well, "")).strip()
            is_blank = sample.upper().startswith("BLANK")
            if well not in available_set:
                cell: dict[str, Any] = {
                    "label": well,
                    "cell_color": "#d1d5db",
                    "tooltip": f"{well} · no data",
                }
            elif state == _WELL_SEL_EXCLUDED:
                cell = {
                    "label": well,
                    "cell_color": "#fb8072",
                    "tooltip": f"{well} · excluded",
                }
            elif is_blank:
                cell = {
                    "label": well,
                    "cell_color": "#80b1d3",
                    "tooltip": f"{well} · blank · included",
                }
            else:
                cell = {
                    "label": well,
                    "cell_color": "#ccebc5",
                    "tooltip": f"{well} · included",
                }
            rendered_row.append(cell)
        rows.append(rendered_row)
    return rows


def _fill_rect_available(
    grid: list[list[str]],
    p1: dict[str, int],
    p2: dict[str, int],
    value: str,
    available_set: set[str],
) -> list[list[str]]:
    left, right = sorted((p1["x"], p2["x"]))
    top, bottom = sorted((p1["y"], p2["y"]))
    out = [row[:] for row in grid]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            well = f"{ROWS[r]}{COLS[c]}"
            if well in available_set:
                out[r][c] = value
    return out


def ui_well_selector(
    *,
    plate_id: str,
    available_wells: list[str],
    name_by_well: dict[str, str] | None = None,
    grid_height: int = 400,
    grid_aspect_ratio: float = 0.75,
) -> list[str]:
    """Render a single-group well selector and return the list of included wells.

    Green = included · blue = blank (included) · red = excluded · dark grey = no data.
    Click once to start a selection, click again to finish a rectangle.
    """
    prefix = f"well_selector::{plate_id}"
    available_set = {str(w).strip().upper() for w in available_wells}
    _init_well_selector_state(prefix, available_set)
    ss = st.session_state

    ak = _state_key(prefix, "grid")
    await_k = _state_key(prefix, "awaiting_second")
    first_k = _state_key(prefix, "first_corner")
    clear_k = _state_key(prefix, "pending_clear_mode")
    last_click_k = _state_key(prefix, "last_processed_click")

    if st_selectable_grid is None:
        st.caption(
            "`st_selectable_grid` is not installed – using multiselect fallback."
        )
        return st.multiselect(
            "Wells to include",
            options=sorted(available_set),
            default=sorted(available_set),
            key=_state_key(prefix, "fallback"),
        )

    selection = st_selectable_grid(
        cells=_build_well_selector_cells(ss[ak], available_set, name_by_well),
        header=[str(c) for c in COLS],
        index=ROWS,
        aspect_ratio=grid_aspect_ratio,
        allow_secondary_selection=False,
        allow_header_selection=False,
        resize=True,
        height=grid_height,
        primary_selection_color="#2563eb",
        key=_state_key(prefix, "well_grid"),
    )

    current_click = (selection or {}).get("primary")
    current_click_key = (
        (current_click["x"], current_click["y"]) if current_click else None
    )

    if current_click_key is None and not ss[await_k]:
        ss[last_click_k] = None

    if not ss[await_k]:
        if current_click and current_click_key != ss[last_click_k]:
            x, y = current_click["x"], current_click["y"]
            ss[clear_k] = ss[ak][y][x] == _WELL_SEL_INCLUDED
            ss[first_k] = current_click
            ss[await_k] = True
            ss[last_click_k] = current_click_key
    else:
        first_corner = ss[first_k]
        if current_click is None:
            value = _WELL_SEL_EXCLUDED if ss[clear_k] else _WELL_SEL_INCLUDED
            ss[ak] = _fill_rect_available(
                ss[ak], first_corner, first_corner, value, available_set
            )
            _reset_pending_selection(prefix)
            st.rerun()
        elif current_click_key != ss[last_click_k]:
            value = _WELL_SEL_EXCLUDED if ss[clear_k] else _WELL_SEL_INCLUDED
            ss[ak] = _fill_rect_available(
                ss[ak], first_corner, current_click, value, available_set
            )
            _reset_pending_selection(prefix)
            ss[last_click_k] = current_click_key
            st.rerun()

    return [
        f"{ROWS[y]}{COLS[x]}"
        for y, row in enumerate(ss[ak])
        for x, state in enumerate(row)
        if state == _WELL_SEL_INCLUDED and f"{ROWS[y]}{COLS[x]}" in available_set
    ]


def get_well_selector_wells(
    plate_id: str, available_wells: list[str]
) -> list[str] | None:
    """Return the stored well selection for a plate without rendering the grid.

    Returns the list of included wells if the plate has been visited (i.e. its
    grid state exists in session state), or None if it has never been shown
    (caller should default to all available wells).
    """
    ak = _state_key(f"well_selector::{plate_id}", "grid")
    if ak not in st.session_state:
        return None
    available_set = {str(w).strip().upper() for w in available_wells}
    return [
        f"{ROWS[y]}{COLS[x]}"
        for y, row in enumerate(st.session_state[ak])
        for x, state in enumerate(row)
        if state == _WELL_SEL_INCLUDED and f"{ROWS[y]}{COLS[x]}" in available_set
    ]
