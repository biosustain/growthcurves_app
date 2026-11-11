"""Click-and-drag plate map for painting wells into groups (Streamlit CCv2).

Pressing on a well and dragging previews a rectangle in the browser; releasing
emits one ``painted`` trigger with the rectangle, so Python only reruns once per
stroke. Pressing a row letter or column number (or dragging across several)
selects whole rows or columns; the corner cell selects the whole plate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

_HTML = """
<div class="plate" role="grid" aria-label="Plate map"></div>
"""

_CSS = """
:host { display: block; }
.plate {
    display: grid;
    grid-template-columns: 1.6rem repeat(var(--cols), minmax(0, 1fr));
    gap: 3px;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
    font-family: var(--st-font, sans-serif);
    font-size: 0.82rem;
    color: #262730;
}
.hdr {
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: var(--st-text-color, #262730);
    cursor: pointer;
    border-radius: 4px;
}
.hdr:hover { background: rgba(37, 99, 235, 0.08); }
.corner { cursor: pointer; }
.cell {
    aspect-ratio: 1 / var(--aspect);
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 5px;
    border: 1px solid rgba(0, 0, 0, 0.18);
    cursor: crosshair;
    box-sizing: border-box;
    transition: background-color 80ms linear;
    position: relative;
}
.cell[data-tip]:hover::after {
    content: attr(data-tip);
    position: absolute;
    bottom: calc(100% + 4px);
    left: 50%;
    transform: translateX(-50%);
    padding: 2px 6px;
    border-radius: 4px;
    background: #262730;
    color: #fff;
    font-size: 0.75rem;
    white-space: nowrap;
    pointer-events: none;
    z-index: 1;
}
.cell.blank { font-weight: 700; border-width: 2px; }
.cell.sel {
    outline: 2px solid #2563eb;
    outline-offset: -2px;
    box-shadow: inset 0 0 0 999px rgba(37, 99, 235, 0.16);
}
.cell.sel.erase {
    outline-style: dashed;
    box-shadow: inset 0 0 0 999px rgba(255, 255, 255, 0.45);
}
.plate.locked .cell, .plate.locked .hdr { cursor: default; }
"""

_JS = """
const state = new WeakMap()

function darken(hex, factor) {
  const m = /^#([0-9a-f]{6})$/i.exec(String(hex || "").trim())
  if (!m) return hex
  const v = m[1]
  const ch = (i) => Math.max(0, Math.min(255, Math.floor(parseInt(v.slice(i, i + 2), 16) * factor)))
  return "#" + [0, 2, 4].map((i) => ch(i).toString(16).padStart(2, "0")).join("")
}

function recolour(el, cell, groupColors) {
  if (!cell.included) return
  const base = groupColors[cell.group] || "#ffffff"
  el.style.backgroundColor = cell.blank ? darken(base, 0.88) : base
  el.style.borderColor = cell.blank ? darken(base, 0.72) : ""
  el.classList.toggle("blank", !!cell.blank)
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const grid = parentElement.querySelector(".plate")
  if (!grid || !data) return

  const rows = data.rows, cols = data.cols, cells = data.cells
  const nR = rows.length, nC = cols.length
  grid.style.setProperty("--cols", String(nC))
  grid.style.setProperty("--aspect", String(data.aspect || 0.6))
  grid.classList.toggle("locked", !data.interactive)

  // Rebuild the 13 x 9 grid from the data sent by Python on every run
  grid.textContent = ""
  const els = []
  const corner = document.createElement("div")
  corner.className = "hdr corner"
  corner.dataset.kind = "all"
  corner.title = "Select the whole plate"
  grid.appendChild(corner)
  cols.forEach((c, x) => {
    const h = document.createElement("div")
    h.className = "hdr"
    h.textContent = c
    h.dataset.kind = "col"
    h.dataset.x = x
    h.title = "Select column " + c
    grid.appendChild(h)
  })
  for (let y = 0; y < nR; y++) {
    const rh = document.createElement("div")
    rh.className = "hdr"
    rh.textContent = rows[y]
    rh.dataset.kind = "row"
    rh.dataset.y = y
    rh.title = "Select row " + rows[y]
    grid.appendChild(rh)
    els.push([])
    for (let x = 0; x < nC; x++) {
      const cell = cells[y][x]
      const el = document.createElement("div")
      el.className = "cell" + (cell.blank ? " blank" : "")
      el.textContent = cell.label
      el.dataset.kind = "cell"
      el.dataset.x = x
      el.dataset.y = y
      el.style.backgroundColor = cell.color
      if (cell.border) el.style.borderColor = cell.border
      if (cell.tooltip) el.dataset.tip = cell.tooltip
      grid.appendChild(el)
      els[y].push(el)
    }
  }

  const drag = { active: false }
  state.set(parentElement, drag)

  const hit = (ev) => {
    const root = grid.getRootNode()
    const el = (root.elementFromPoint ? root : document).elementFromPoint(ev.clientX, ev.clientY)
    return el && el.closest ? el.closest("[data-kind]") : null
  }

  // Rectangle (inclusive) covered by a drag from the start target to the current target
  const rectFor = (start, cur) => {
    const sx = start.dataset.x, sy = start.dataset.y
    const cx = cur.dataset.x !== undefined ? +cur.dataset.x : null
    const cy = cur.dataset.y !== undefined ? +cur.dataset.y : null
    if (start.dataset.kind === "all") return { r0: 0, r1: nR - 1, c0: 0, c1: nC - 1 }
    if (start.dataset.kind === "col") {
      const x1 = cx === null ? +sx : cx
      return { r0: 0, r1: nR - 1, c0: Math.min(+sx, x1), c1: Math.max(+sx, x1) }
    }
    if (start.dataset.kind === "row") {
      const y1 = cy === null ? +sy : cy
      return { r0: Math.min(+sy, y1), r1: Math.max(+sy, y1), c0: 0, c1: nC - 1 }
    }
    const x1 = cx === null ? +sx : cx, y1 = cy === null ? +sy : cy
    return { r0: Math.min(+sy, y1), r1: Math.max(+sy, y1), c0: Math.min(+sx, x1), c1: Math.max(+sx, x1) }
  }

  const preview = (rect) => {
    for (let y = 0; y < nR; y++)
      for (let x = 0; x < nC; x++) {
        const inside = rect && y >= rect.r0 && y <= rect.r1 && x >= rect.c0 && x <= rect.c1
        els[y][x].classList.toggle("sel", !!inside)
        els[y][x].classList.toggle("erase", !!inside && drag.erase)
      }
  }

  grid.onpointerdown = (ev) => {
    if (!data.interactive || ev.button !== 0) return
    const t = hit(ev)
    if (!t) return
    ev.preventDefault()
    drag.active = true
    drag.start = t
    // Starting on a well already in the active group (or, for "Blank", a painted
    // blank) removes wells from it instead
    const startCell = t.dataset.kind === "cell" ? cells[+t.dataset.y][+t.dataset.x] : null
    drag.erase = !!startCell && (data.active_group === data.blank_option
      ? !!startCell.painted
      : startCell.group === data.active_group)
    drag.rect = rectFor(t, t)
    grid.setPointerCapture(ev.pointerId)
    preview(drag.rect)
  }

  grid.onpointermove = (ev) => {
    if (!drag.active) return
    const t = hit(ev)
    if (!t) return
    const rect = rectFor(drag.start, t)
    if (JSON.stringify(rect) !== JSON.stringify(drag.rect)) {
      drag.rect = rect
      preview(rect)
    }
  }

  const finish = (ev) => {
    if (!drag.active) return
    drag.active = false
    try { grid.releasePointerCapture(ev.pointerId) } catch (e) {}
    const rect = drag.rect
    preview(null)
    if (!rect) return
    // Recolour immediately; Python stores the change and re-renders
    const blankMode = data.active_group === data.blank_option
    for (let y = rect.r0; y <= rect.r1; y++)
      for (let x = rect.c0; x <= rect.c1; x++) {
        const cell = cells[y][x]
        if (blankMode) {
          cell.painted = !drag.erase
          cell.blank = cell.included && (cell.fixed_blank || cell.painted)
        } else {
          cell.group = drag.erase ? data.default_group : data.active_group
        }
        recolour(els[y][x], cell, data.group_colors)
      }
    setTriggerValue("painted", { ...rect, erase: drag.erase })
  }
  grid.onpointerup = finish
  grid.onpointercancel = finish
}
"""

_PLATE_PAINTER = st.components.v2.component(
    "microgrowth_plate_painter",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def plate_painter(
    *,
    rows: list[str],
    cols: list[int],
    cells: list[list[dict[str, Any]]],
    group_colors: dict[str, str],
    active_group: str,
    default_group: str,
    key: str,
    blank_option: str = "Blank",
    on_paint: Callable[[], None],
    aspect_ratio: float = 0.6,
    interactive: bool = True,
):
    """Render the drag-to-paint plate map.

    cells: per well dict with label, color, border, tooltip, group,
        blank, fixed_blank (blank from the plate map), painted (blank painted by
        the user) and included. When active_group is blank_option, strokes mark
        or unmark blanks instead of changing groups. on_paint runs (before the script body) when a stroke
        finishes; read the rectangle from ``st.session_state[key].painted`` as
        {"r0", "r1", "c0", "c1", "erase"}.
    """
    return _PLATE_PAINTER(
        key=key,
        data={
            "rows": list(rows),
            "cols": [str(c) for c in cols],
            "cells": cells,
            "group_colors": group_colors,
            "active_group": active_group,
            "default_group": default_group,
            "blank_option": blank_option,
            "aspect": float(aspect_ratio),
            "interactive": bool(interactive),
        },
        on_painted_change=on_paint,
    )
