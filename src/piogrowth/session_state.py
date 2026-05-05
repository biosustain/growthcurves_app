"""Session state snapshot helpers for Streamlit apps.

This module centralizes both logic and reusable UI blocks for
exporting/restoring `st.session_state` snapshots.

ZIP structure
-------------
- ``metadata.json``: JSON-serializable scalar/list/dict values.
- ``dataframes/{key}.csv``: one CSV per ``pd.DataFrame`` value.
- ``files/{key}.bin``: raw ``bytes`` values (e.g. uploaded file content).
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections.abc import MutableMapping
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

MAX_LIST_REPR = 5  # max items shown inline for lists


# region: JSON serialization helpers
def _to_json_serializable(val: Any) -> Any:
    """Convert *val* to a JSON-serialisable form, raising ``TypeError`` if
    the value cannot be represented.
    """
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, pd.Timestamp):
        return {"__type__": "pd.Timestamp", "value": str(val)}
    if isinstance(val, list):
        return [_to_json_serializable(item) for item in val]
    if isinstance(val, dict):
        return {k: _to_json_serializable(v) for k, v in val.items()}
    raise TypeError(f"Cannot JSON-serialise {type(val).__name__}")


def _from_json_value(val: Any) -> Any:
    """Reconstruct a value from its JSON-serialisable representation."""
    if isinstance(val, dict):
        if val.get("__type__") == "pd.Timestamp":
            return pd.Timestamp(val["value"])
        return {k: _from_json_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_from_json_value(item) for item in val]
    return val


# endregion


# region: Session state snapshot export/import logic
def build_session_state_zip(
    session_state: MutableMapping[str, Any],
    exclude_keys: frozenset[str] | None = None,
) -> bytes:
    """Serialise session state into a ZIP archive and return the raw bytes.

    DataFrames are stored as CSV files under ``dataframes/``, raw ``bytes``
    values under ``files/``, and all other JSON-serialisable values are
    collected into ``metadata.json``.  Non-serialisable values are silently
    skipped with a debug log entry.
    """
    _exclude_keys = exclude_keys or frozenset()
    metadata: dict[str, Any] = {}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, val in session_state.items():
            if key in _exclude_keys:
                continue
            if isinstance(val, pd.DataFrame):
                zf.writestr(
                    f"dataframes/{key}.csv",
                    val.to_csv().encode("utf-8"),
                )
            elif isinstance(val, bytes):
                zf.writestr(f"files/{key}.bin", val)
            else:
                try:
                    metadata[key] = _to_json_serializable(val)
                except TypeError:
                    logger.debug(
                        "Skipping non-serialisable session state key"
                        " %s (type %s)",
                        key,
                        type(val).__name__,
                    )
        zf.writestr("metadata.json", json.dumps(metadata))
    buf.seek(0)
    return buf.getvalue()


def restore_session_state_from_zip(
    session_state: MutableMapping[str, Any],
    zip_bytes: bytes,
) -> tuple[bool, list[str]]:
    """Restore session state from a ZIP archive.

    Returns a ``(restored, warnings)`` tuple where *restored* is ``True``
    when at least one key was successfully written back and *warnings* is a
    list of human-readable problem descriptions.
    """
    warnings: list[str] = []
    restored = False

    try:
        zf_io = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    except zipfile.BadZipFile:
        warnings.append(
            "The uploaded file is not a valid ZIP archive — nothing restored."
        )
        return False, warnings

    with zf_io as zf:
        names = set(zf.namelist())

        # --- metadata.json ---------------------------------------------------
        if "metadata.json" not in names:
            warnings.append(
                "metadata.json not found in ZIP — no scalar metadata restored."
            )
        else:
            try:
                raw = json.loads(zf.read("metadata.json").decode("utf-8"))
                for key, val in raw.items():
                    session_state[key] = _from_json_value(val)
                restored = True
            except Exception as exc:
                warnings.append(f"Failed to restore metadata: {exc}")

        # --- dataframes/ -----------------------------------------------------
        df_entries = [
            n for n in names
            if n.startswith("dataframes/") and n.endswith(".csv")
        ]
        for entry in df_entries:
            key = entry[len("dataframes/"):-len(".csv")]
            try:
                df = pd.read_csv(
                    io.BytesIO(zf.read(entry)),
                    index_col=0,
                    parse_dates=True,
                )
                session_state[key] = df
                restored = True
            except Exception as exc:
                warnings.append(f"Failed to restore DataFrame '{key}': {exc}")

        # --- files/ ----------------------------------------------------------
        file_entries = [
            n for n in names
            if n.startswith("files/") and n.endswith(".bin")
        ]
        for entry in file_entries:
            key = entry[len("files/"):-len(".bin")]
            try:
                session_state[key] = zf.read(entry)
                restored = True
            except Exception as exc:
                warnings.append(f"Failed to restore file '{key}': {exc}")

    return restored, warnings


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
                restored, restore_warnings = restore_session_state_from_zip(
                    st.session_state, session_zip_upload.getvalue()
                )
            for warning in restore_warnings:
                st.warning(warning)
            if restored:
                st.success("Session state restored successfully.")
                st.rerun()
            elif not restore_warnings:
                st.warning("Nothing was restored from the ZIP.")


# endregion


# region: Session state snapshot export/import UI components
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


# endregion


# region: Session state inspection for debugging
def summarize_value(val, max_list_repr=MAX_LIST_REPR):
    """Return a short, human-readable summary of a session state value."""
    if isinstance(val, pd.DataFrame):
        return f"<DataFrame shape={val.shape} columns={list(val.columns)}>"
    if isinstance(val, list):
        if len(val) <= max_list_repr:
            return repr(val)
        preview = (
            repr(val[:max_list_repr])[:-1] + f", ... +{len(val) - max_list_repr} more]"
        )
        return preview
    return val


def ui_overview_table(max_list_repr=MAX_LIST_REPR):
    """Displays an overview table of all session state keys, their types, and
    summaries.
    """
    st.subheader("Overview")

    # st.write(st.session_state)

    rows = []
    for key, val in st.session_state.items():
        rows.append(
            {
                "key": str(key),
                "type": type(val).__name__,
                "summary": summarize_value(val, max_list_repr=max_list_repr),
            }
        )

    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("key"),
            use_container_width=True,
            height=min(40 + 35 * len(rows), 500),
        )
    else:
        st.info("Session state is empty.")


def ui_key_inspector(max_list_repr=MAX_LIST_REPR):
    """Displays a UI for inspecting the value of a selected session state key in
    detail.
    """
    st.subheader("Inspect key")

    keys = list(st.session_state.keys())
    if keys:
        selected = st.selectbox("Select a key", options=keys, format_func=str)

        val = st.session_state[selected]

        if isinstance(val, pd.DataFrame):
            st.write(
                f"**DataFrame** — shape `{val.shape}`, columns: `{list(val.columns)}`"
            )
            st.dataframe(val, use_container_width=True)
        elif isinstance(val, list):
            st.write(f"**list** — {len(val)} items")
            for i, item in enumerate(val):
                with st.expander(f"[{i}]  {repr(item)[:120]}", expanded=False):
                    if isinstance(item, pd.DataFrame):
                        st.dataframe(item, use_container_width=True)
                    else:
                        st.write(item)
        elif isinstance(val, dict):
            st.write(f"**dict** — {len(val)} keys")
            st.json(
                {
                    k: (
                        summarize_value(v, max_list_repr=max_list_repr)
                        if isinstance(v, (pd.DataFrame, list))
                        else v
                    )
                    for k, v in val.items()
                }
            )
        else:
            st.write(val)
    else:
        st.info("No keys to inspect.")


# endregion
