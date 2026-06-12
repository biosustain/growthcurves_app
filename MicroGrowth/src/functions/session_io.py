"""Save / restore the full MicroGrowth session as a downloadable ZIP.

MicroGrowth's session state is deeply nested — the ``plates`` dict holds per-well
DataFrames, fit-result dicts with numpy scalars, plate-map DataFrames and raw
uploaded bytes. To restore a session *exactly* as it was, the snapshot pickles
the session values losslessly. Plots are not stored: they are recomputed from
``plates``, so restoring the data reproduces them.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from collections.abc import MutableMapping
from typing import Any

import streamlit as st

# keys to exclude from session snapshot
_EXCLUDE_EXACT = frozenset(
    {
        "data_up",
        "map_up",
        "session_restore_upload",
        "well_prev",
        "well_next",
        "well_plot_prev",
        "well_plot_next",
        "export_zip_bytes",
    }
)
_EXCLUDE_PREFIXES = ("deletewell__", "reanalyse__", "restore_defaults__", "nogrowth__")


def _is_excluded(key: Any) -> bool:
    key = str(key)
    return key in _EXCLUDE_EXACT or key.startswith(_EXCLUDE_PREFIXES)


def build_session_zip(session_state: MutableMapping[str, Any]) -> bytes:
    """Pickle session state (minus widget keys) into a ZIP archive."""
    payload = {k: v for k, v in session_state.items() if not _is_excluded(k)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session.pkl", pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))
    return buf.getvalue()


def restore_session_zip(
    session_state: MutableMapping[str, Any], zip_bytes: bytes
) -> str | None:
    """Replace session state with the snapshot in a ``build_session_zip`` ZIP.

    The current session is only cleared once the ZIP has parsed successfully, so
    an invalid upload leaves the existing session untouched. Returns ``None`` on
    success, or an error message if the file is not a valid session snapshot.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            payload = pickle.loads(zf.read("session.pkl"))  # noqa: S301
    except (zipfile.BadZipFile, KeyError):
        return "This is not a valid MicroGrowth session ZIP."
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read session snapshot: {exc}"

    session_state.clear()
    session_state.update(payload)
    return None


def render_export_session_ui() -> None:
    """Render the 'Save Session' block (Download Analyzed Data page)."""
    with st.container(border=True):
        st.subheader("Save Session")
        st.caption(
            "Download a ZIP snapshot of the entire session — all data, parameters "
            "and results. Re-upload it on the **Upload and Analyze** page to "
            "restore the session exactly as it is now."
        )
        snapshot = dict(st.session_state)
        st.download_button(
            "Download session ZIP",
            data=lambda: build_session_zip(snapshot),
            file_name="microgrowth_session.zip",
            mime="application/zip",
            width="stretch",
        )


def render_restore_session_ui() -> None:
    """Render the 'Restore session' uploader (top of Upload and Analyze page)."""
    with st.expander("Restore a saved session", icon="📁"):
        st.caption(
            "Drop a session ZIP downloaded from the **Download Analyzed Data** "
            "page to restore that session — all data, parameters and results."
        )
        upload = st.file_uploader(
            "Upload session ZIP", type=["zip"], key="session_restore_upload"
        )
        if upload is not None:
            error = restore_session_zip(st.session_state, upload.getvalue())
            if error:
                st.warning(error)
            else:
                st.toast("Session restored.", icon="✅")
                st.rerun()
