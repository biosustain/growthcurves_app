"""Common utilities used across the application."""

import streamlit as st


def require_plates() -> dict:
    """Return plates from session state if processed data exists, or stop."""
    plates = st.session_state.get("plates") or {}
    has_processed = any(p.get("processed_data") for p in plates.values())
    if not has_processed:
        st.info("Add data first by running **Upload and Analyze**.")
        st.stop()
    return plates


def toast_once(msg: str):
    """Show a permanent toast, once per session, for warnings raised on every rerun."""
    shown = st.session_state.setdefault("_toasts_shown", set())
    if msg not in shown:
        shown.add(msg)
        st.toast(msg, duration="infinite")


def _iter_wells(plates: dict):
    """Yield (plate_id, plate, well, name, processed_df, growth_stats)."""
    for pid, p in plates.items():
        nm_by_well = p.get("name") or {}
        proc = p.get("processed_data") or {}
        gs_all = p.get("growth_stats") or {}
        for well, d in proc.items():
            yield pid, p, well, (nm_by_well.get(well) or ""), d, (
                gs_all.get(well) or {}
            )
