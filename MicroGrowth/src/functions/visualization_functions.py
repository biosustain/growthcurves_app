"""Non-UI helpers for growth summaries."""

import numpy as np
import pandas as pd

from src.functions.common import _iter_wells


# ---------------- Helpers ----------------
def _build_growth_stats_long_df(
    plates: dict, sel_ids: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Build a long DataFrame for growth stats plots and the sample order."""
    columns = ["plate", "well", "sample_name", "metric", "value"]
    sample_order: list[str] = []
    for sid in sel_ids:
        _, nm = sid.split("||", 1)
        if nm and nm not in sample_order:
            sample_order.append(nm)

    metrics = [
        "mu_max",
        "intrinsic_growth_rate",
        "doubling_time",
        "max_od",
        "exp_phase_start",
        "exp_phase_end",
        "time_at_umax",
        "od_at_umax",
    ]

    rows: list[dict] = []

    for sid in sel_ids:
        pid, nm = sid.split("||", 1)
        p = plates.get(pid) or {}
        name_map = p.get("name") or {}
        gs_map = p.get("growth_stats") or {}

        for well, well_nm in name_map.items():
            if (well_nm or "").strip() != nm:
                continue
            gs = gs_map.get(well) or {}
            if not gs:
                continue

            for m in metrics:
                if m == "mu_max":
                    raw_val = gs.get("mu_max", gs.get("specific_growth_rate", np.nan))
                else:
                    raw_val = gs.get(m, np.nan)
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = np.nan
                rows.append(
                    {
                        "plate": pid,
                        "well": well,
                        "sample_name": nm,
                        "metric": m,
                        "value": val,
                    }
                )

    return pd.DataFrame(rows, columns=columns), sample_order


def _max_time_hours(plates: dict, default: float = 72.0) -> float:
    """Return the maximum time (hours) across all processed data."""
    max_t = float(default)
    for p in plates.values():
        for d in (p.get("processed_data") or {}).values():
            if d is not None and not d.empty and "Time" in d.columns:
                max_t = max(max_t, float(d["Time"].max()))
    return max_t


def _unique_preserve_order(seq):
    """Return unique items in the original order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _build_growth_curves_long_df(plates: dict, sample_names: list[str]) -> pd.DataFrame:
    """Return a long DataFrame for mean/replicate curve plots."""
    if not sample_names:
        return pd.DataFrame(
            columns=[
                "Sample Name",
                "Time",
                "baseline_corrected",
                "plate",
                "well",
                "key",
            ]
        )

    sel = set(sample_names)
    rows = []

    for pid, _, well, nm, d, key in _iter_wells(plates):
        nm = (nm or "").strip()
        if nm in {"", "False"} or nm.upper().startswith("BLANK"):
            continue
        if nm not in sel:
            continue
        if d is None or d.empty:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "Sample Name": nm,
                    "Time": d["Time"],
                    "baseline_corrected": d["baseline_corrected"],
                    "plate": pid,
                    "well": well,
                    "key": key,
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Sample Name",
                "Time",
                "baseline_corrected",
                "plate",
                "well",
                "key",
            ]
        )

    return pd.concat(rows, ignore_index=True)
