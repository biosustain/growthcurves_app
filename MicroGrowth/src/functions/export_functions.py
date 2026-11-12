"""Export functionality for downloadable tables and plots."""

import io
import zipfile

import pandas as pd
from src.functions.check_growth_fits import _sg_params_for_plate
from src.functions.plotting_functions import (
    plot_baseline_by_group,
    plot_derivative_metric,
    plot_replicates_by_sample,
    plot_well_curve,
    plot_window_plate,
)


# ---------------- Export helpers ----------------
def _processed_wide_for_plate(p: dict, *, value_col: str) -> pd.DataFrame:
    """Return a wide, time-indexed DataFrame with one column per well."""
    columns = {
        well: d.set_index("Time")[value_col]
        for well, d in (p.get("processed_data") or {}).items()
        if d is not None and not d.empty and {"Time", value_col} <= set(d.columns)
    }
    if not columns:
        return pd.DataFrame()
    # One outer join on Time for all wells
    return pd.concat(columns, axis=1).sort_index().rename_axis("Time").reset_index()


def _growth_stats_per_well_df(p: dict) -> pd.DataFrame:
    """Return growth stats per well as a tidy DataFrame with sample names."""
    nm_by_well = p.get("name") or {}
    df = (
        pd.DataFrame.from_dict(p.get("growth_stats") or {}, orient="index")
        .rename_axis("well")
        .reset_index()
    )
    if not df.empty:
        # Add Sample Name column after well column
        df.insert(
            1,
            "Sample Name",
            df["well"].map(lambda w: (nm_by_well.get(w) or "").strip()),
        )
        # Remove internal columns not meant for export
        internal_cols = [
            "_lasso_update_time",
            "_used_fit_times",
            "_analysis_params",
        ]
        df = df.drop(
            columns=[c for c in internal_cols if c in df.columns], errors="ignore"
        )
    return df


def _growth_stats_mean_for_sample_df(p: dict) -> pd.DataFrame:
    """Return growth stats averaged per sample name with mean and std columns."""
    gs = _growth_stats_per_well_df(p)
    if gs.empty:
        return gs

    num = [c for c in gs.columns if pd.api.types.is_numeric_dtype(gs[c])]

    # Calculate both mean and std
    result = gs.groupby("Sample Name")[num].agg(["mean", "std"])

    # Flatten the multi-level columns: convert (col, 'mean') to 'col_mean'
    result.columns = ["_".join(col).strip() for col in result.columns.values]

    return result.reset_index()


def _analysis_params_df(p: dict) -> pd.DataFrame:
    """Return analysis parameters as a DataFrame."""
    params = p.get("params") or {}
    if not params:
        return pd.DataFrame()

    # Format parameters for display
    data = {
        "Parameter": [],
        "Value": [],
    }

    # Extract parameters with readable names
    param_mapping = {
        "time_unit": "Time unit (data file)",
        "pathlength_cm_": "Pathlength (cm)",
        "clip_time_series": "Time series clip (hours)",
        "remove_wells": "Excluded wells",
        "blank": "Blank subtraction",
        "blank_wells": "Wells marked as blanks on the plate map",
        "blank_group_assignments": "Blank groups",
        "outlier_detection": "Outlier detection (ECOD)",
        "outlier_threshold": "Outlier threshold (MAD z-score)",
        "growth_method": "Analysis method",
        "model_family": "Parametric model type",
        "model_type": "Growth model",
        "phase_boundary_method": "Phase boundary calculation",
        "window_points": "Minimum window size (points)",
        "smooth": "Spline smoothing (λ)",
        "min_data_points": "Minimum data points",
        "min_signal_to_noise": "Minimum signal-to-noise ratio",
        "min_od_increase": "Minimum OD increase",
        "min_growth_rate": "Minimum growth rate (1/h)",
        "sg_window": "Savitzky-Golay window (derivative plots)",
        "sg_poly": "Savitzky-Golay polynomial order (derivative plots)",
    }

    for key, label in param_mapping.items():
        if key in params:
            value = params[key]
            # Format special cases
            if key == "clip_time_series" and isinstance(value, (tuple, list)):
                value = f"{value[0]} - {value[1]}"
            elif key in ("remove_wells", "blank_wells"):
                value = ", ".join(value) if value else "None"
            elif key == "blank_group_assignments":
                wells_by_group: dict[str, list[str]] = {}
                for well, group in (value or {}).items():
                    wells_by_group.setdefault(group, []).append(well)
                value = (
                    "; ".join(
                        f"{group}: {', '.join(wells)}"
                        for group, wells in sorted(wells_by_group.items())
                    )
                    or "None"
                )
            elif key in ("blank", "outlier_detection"):
                value = "Yes" if value else "No"

            data["Parameter"].append(label)
            data["Value"].append(str(value))

    return pd.DataFrame(data)


# ---------------- ZIP builder ----------------
def build_export_zip(
    plates: dict,
    *,
    include_baseline_corrected: bool,
    include_stats_per_well: bool,
    include_stats_per_sample: bool,
    include_params: bool,
    include_plate_view: bool,
    include_baseline_plots: bool,
    include_replicates: bool,
    include_well_plots: bool,
    well_graphs: (
        list[str] | None
    ) = None,  # e.g. ["Raw OD", "dOD/dt", "Specific Growth Rate"]
    selected_plate_ids: list[str] | None = None,  # plates to include for well plots
    wells_by_plate: dict[str, list[str]] | None = None,  # {plate_id: [well,...]}
    annot_phase: bool = True,
    annot_umax_point: bool = True,
    annot_od_max: bool = True,
    annot_baseline_od: bool = True,
    annot_tangent: bool = False,
    annot_fitted_model: bool = True,
    scale: int = 2,
    baseline_width: int = 1200,
    baseline_height: int = 800,
    plate_width: int = 1200,
    plate_height: int = 800,
    well_width: int = 1200,
    well_height: int = 800,
) -> bytes:
    """Build a ZIP of CSVs and static PNG plots based on selected options."""
    well_graphs = well_graphs or []
    selected_plate_ids = selected_plate_ids or []
    wells_by_plate = wells_by_plate or {}

    def _png(fig, width: int, height: int) -> bytes:
        # White background: the app figures are transparent, for its grey cards
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return fig.to_image(format="png", width=width, height=height, scale=scale)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # ---- Tables (GLOBAL) ----
        # Combine all plates into single tables with a Plate column
        if include_baseline_corrected:
            all_baseline_corrected = []
            for pid, p in plates.items():
                wide = _processed_wide_for_plate(p, value_col="baseline_corrected")
                if not wide.empty:
                    wide.insert(0, "Plate", pid)
                    all_baseline_corrected.append(wide)
            if all_baseline_corrected:
                combined_baseline = pd.concat(all_baseline_corrected, ignore_index=True)
                zf.writestr(
                    "tables/processed_baseline_corrected.csv",
                    combined_baseline.to_csv(index=False),
                )

        if include_stats_per_well:
            all_stats_per_well = []
            for pid, p in plates.items():
                per_well_df = _growth_stats_per_well_df(p)
                if not per_well_df.empty:
                    per_well_df.insert(0, "Plate", pid)
                    all_stats_per_well.append(per_well_df)
            if all_stats_per_well:
                combined_stats_per_well = pd.concat(
                    all_stats_per_well, ignore_index=True
                )
                zf.writestr(
                    "tables/growth_stats_per_well.csv",
                    combined_stats_per_well.to_csv(index=False),
                )

        if include_stats_per_sample:
            all_stats_per_sample = []
            for pid, p in plates.items():
                mean_df = _growth_stats_mean_for_sample_df(p)
                if not mean_df.empty:
                    mean_df.insert(0, "Plate", pid)
                    all_stats_per_sample.append(mean_df)
            if all_stats_per_sample:
                combined_stats_per_sample = pd.concat(
                    all_stats_per_sample, ignore_index=True
                )
                zf.writestr(
                    "tables/growth_stats_mean_for_sample.csv",
                    combined_stats_per_sample.to_csv(index=False),
                )

        # ---- Analysis Parameters ----
        if include_params:
            all_params = []
            for pid, p in plates.items():
                params_df = _analysis_params_df(p)
                if not params_df.empty:
                    # Rename "Value" column to the plate ID
                    params_df = params_df.rename(columns={"Value": pid})
                    all_params.append(params_df)
            if all_params:
                # Merge all plates horizontally on the Parameter column
                combined_params = all_params[0]
                for params_df in all_params[1:]:
                    combined_params = combined_params.merge(
                        params_df, on="Parameter", how="outer"
                    )
                zf.writestr(
                    "tables/default_analysis_parameters.csv",
                    combined_params.to_csv(index=False),
                )

        # ---- Baseline plots (GLOBAL) ----
        if include_baseline_plots:
            for pid, p in plates.items():
                baseline = p.get("baseline")
                if baseline is None or baseline.empty:
                    continue
                # As on Plate Overviews: one panel per blank group
                baseline_fig = plot_baseline_by_group(
                    baseline,
                    blank_group_map=p["params"].get("blank_group_assignments") or {},
                )
                zf.writestr(
                    f"plots/{pid}/baseline.png",
                    _png(baseline_fig, baseline_width, baseline_height),
                )

        # ---- Replicates plot (GLOBAL) ----
        if include_replicates:
            rep_fig = plot_replicates_by_sample(plates)
            if rep_fig is not None:
                zf.writestr(
                    "plots/replicates_by_sample.png",
                    _png(rep_fig, plate_width, plate_height),
                )

        # ---- Plate-view plots ----
        if include_plate_view:
            for pid, p in plates.items():
                fig = plot_window_plate(p)
                if fig is not None:
                    zf.writestr(
                        f"plots/{pid}/window_plate.png",
                        _png(fig, plate_width, plate_height),
                    )

        # ---- Well-level plots ----
        if include_well_plots and selected_plate_ids and well_graphs:
            for pid in selected_plate_ids:
                p = plates.get(pid)
                if not p:
                    continue

                processed = p.get("processed_data") or {}
                if not processed:
                    continue

                sg_w, sg_p = _sg_params_for_plate(plates, pid)

                # use requested wells; default to all available only if plate absent from dict
                wells = wells_by_plate.get(pid, list(processed.keys()))

                # The same figures as on Check Growth Fits
                plate_dir = f"plots/{pid}/wells"
                for well in wells:
                    if well not in processed:
                        continue
                    figs = {}
                    if "Raw OD" in well_graphs:
                        figs["growth_curves"] = plot_well_curve(
                            p,
                            well,
                            phase_boundaries=annot_phase,
                            umax_point=annot_umax_point,
                            max_od=annot_od_max,
                            baseline_od=annot_baseline_od,
                            tangent=annot_tangent,
                            fitted_model=annot_fitted_model,
                        )
                    gs = (p.get("growth_stats") or {}).get(well) or {}
                    for label, metric, folder in (
                        ("dOD/dt", "dndt", "curves_d1"),
                        ("Specific Growth Rate", "mu", "curves_mu"),
                    ):
                        if label in well_graphs:
                            figs[folder] = plot_derivative_metric(
                                p, well, metric, sg_w, sg_p, gs=gs
                            )
                    for folder, fig in figs.items():
                        if fig.data:  # empty when the well has too little data
                            zf.writestr(
                                f"{plate_dir}/{folder}/{well}.png",
                                _png(fig, well_width, well_height),
                            )

    buf.seek(0)
    return buf.getvalue()
