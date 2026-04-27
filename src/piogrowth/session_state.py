"""Session state snapshot helpers for Streamlit apps.

This module centralizes both logic and reusable UI blocks for
exporting/restoring `st.session_state` snapshots.
"""

from __future__ import annotations

import io
import logging
import pickle
import zipfile
from collections.abc import MutableMapping
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

def build_session_state_zip(
    session_state: MutableMapping[str, Any],
    exclude_keys: frozenset[str] | None = None,
) -> bytes:
    """Pickle all serializable session state into a ZIP and return bytes."""
    state_to_save: dict[str, Any] = {}
    _exclude_keys = exclude_keys or frozenset()
    for key, val in session_state.items():
        if key in _exclude_keys:
            continue
        try:
            pickle.dumps(val)
            state_to_save[key] = val
        except Exception:
            logger.debug(
                "Skipping non-serializable session state key %s (type %s)",
                key,
                type(val),
                exc_info=True,
            )
            continue

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session_state.pkl", pickle.dumps(state_to_save))
    buf.seek(0)
    return buf.getvalue()


def restore_session_state_from_zip(
    session_state: MutableMapping[str, Any],
    zip_bytes: bytes,
) -> list[str]:
    """Restore session_state values from a ZIP and return warnings."""
    warnings: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        if "session_state.pkl" not in zf.namelist():
            warnings.append("session_state.pkl not found in ZIP — nothing restored.")
            return warnings
        with zf.open("session_state.pkl") as f:
            state_dict = pickle.loads(f.read())  # noqa: S301

    for key, val in state_dict.items():
        session_state[key] = val
    return warnings


def render_restore_session_state_ui() -> None:
    """Render reusable restore-from-ZIP UI on data-upload pages."""
    with st.expander("Restore Previous Session (Optional)", icon="📁"):
        st.header("Restore Previous Session (Optional)")
        st.caption(
            "Upload a session state ZIP downloaded from the **Downloads** page "
            "to restore the app exactly as it was — including all data, "
            "preprocessing settings, and analysis results."
        )
        session_zip_upload = st.file_uploader(
            "Upload session state ZIP",
            type=["zip"],
            key="session_state_zip_upload",
        )
        if session_zip_upload is not None and st.button(
            "Restore session from ZIP",
            key="restore_session_state_btn",
            type="primary",
        ):
            with st.spinner("Restoring session state...", show_time=True):
                restore_warnings = restore_session_state_from_zip(
                    st.session_state, session_zip_upload.getvalue()
                )
            for warning in restore_warnings:
                st.warning(warning)
            st.success("Session state restored successfully.")
            st.rerun()


def render_export_session_state_ui(
    custom_id: str,
    exclude_keys: frozenset[str] | None = None,
) -> None:
    """Render reusable session snapshot export UI on downloads pages."""
    with st.container(border=True):
        st.header("Export Session State")
        st.caption(
            "Download a ZIP snapshot of the full app state. "
            "Re-upload it on the **Upload Data** page to restore "
            "the session exactly as it was."
        )
        prepare = st.button(
            "Prepare session state download", key="prepare_session_state_zip"
        )

    if prepare:
        with st.spinner("Building session state ZIP...", show_time=True):
            st.session_state["session_state_zip"] = build_session_state_zip(
                st.session_state,
                exclude_keys=exclude_keys,
            )

    if st.session_state.get("session_state_zip") is not None:
        st.download_button(
            label="Download session state ZIP",
            data=st.session_state["session_state_zip"],
            file_name=f"{custom_id}_session_state.zip",
            mime="application/zip",
            key="download_session_state_zip",
        )
