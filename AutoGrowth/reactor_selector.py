"""Include or exclude reactors on a drag-to-paint grid.

Adapted from ``ui_well_selector`` in MicroGrowth/src/ui_functions/blank_grouping_ui.py:
the same plate painter and colours (green = included, red = excluded), with the
reactors laid out in rows instead of plate wells.
"""

import math

import streamlit as st
from plate_painter import plate_painter

INCLUDED = "included"
EXCLUDED = "excluded"
COLORS = {INCLUDED: "#ccebc5", EXCLUDED: "#fb8072"}
NO_REACTOR = "#d1d5db"
MAX_COLS = 8
STATES_KEY = "reactor_selector::states"  # {reactor: INCLUDED or EXCLUDED}
PAINTER_KEY = "reactor_selector::plate_painter"


def _apply_paint(reactors: list[str], n_cols: int):
    """Include/exclude the reactors of a finished drag stroke (runs before the
    script body)."""
    ss = st.session_state
    stroke = getattr(ss.get(PAINTER_KEY), "painted", None)
    if not stroke:
        return
    value = EXCLUDED if stroke.get("erase") else INCLUDED
    for y in range(int(stroke["r0"]), int(stroke["r1"]) + 1):
        for x in range(int(stroke["c0"]), int(stroke["c1"]) + 1):
            i = y * n_cols + x
            if i < len(reactors):
                ss[STATES_KEY][reactors[i]] = value


def reactor_selector(reactors: list[str]) -> list[str]:
    """Render the reactor grid and return the included reactors.

    Dragging from an included reactor excludes the reactors covered; dragging from
    an excluded reactor includes them. New reactors are included.
    """
    states = st.session_state.setdefault(STATES_KEY, {})
    for r in reactors:
        states.setdefault(r, INCLUDED)
    n_cols = min(len(reactors), MAX_COLS)
    n_rows = math.ceil(len(reactors) / n_cols)

    def cell(i: int) -> dict:
        if i >= len(reactors):
            return {"label": "", "color": NO_REACTOR}
        r, state = reactors[i], states[reactors[i]]
        return {
            "label": r,
            "color": COLORS[state],
            "tooltip": f"{r} · {state}",
            "group": state,
            "included": True,
        }

    plate_painter(
        rows=[""] * n_rows,
        cols=[""] * n_cols,
        cells=[[cell(y * n_cols + x) for x in range(n_cols)] for y in range(n_rows)],
        group_colors=COLORS,
        active_group=INCLUDED,
        default_group=EXCLUDED,
        blank_option="__none__",
        key=PAINTER_KEY,
        on_paint=lambda: _apply_paint(reactors, n_cols),
        aspect_ratio=0.45,
    )
    return [r for r in reactors if states[r] == INCLUDED]
