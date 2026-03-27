"""Export functionality for downloadable tables and plots."""

import io
import zipfile

import growthcurves.plot as gc_plot
import pandas as pd
from growthcurves.inference import is_no_growth
from growthcurves.plot import plot_derivative_metric

from src.functions.check_growth_fits import _add_lasso_selected_points
from src.functions.plotting_functions import (
    _finite_sorted_xy,
    plot_baseline,
    plot_replicates_by_sample,
    plot_window_plate,
)


# ---------------- Export helpers ----------------
def _processed_wide_for_plate(p: dict, *, value_col: str) -> pd.DataFrame:
    """Return a wide, time-indexed DataFrame with one column per well."""
    frames = []
    for well, d in (p.get("processed_data") or {}).items():
        if d is None or d.empty:
            continue
        if "Time" not in d.columns or value_col not in d.columns:
            continue
        frames.append(d[["Time", value_col]].rename(columns={value_col: well}))

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Time", how="outer")

    return out.sort_values("Time").reset_index(drop=True)


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
            "lasso_t_min",
            "lasso_t_max",
            "_used_fit_times",
            "_analysis_params",
        ]
        df = df.drop(
            columns=[c for c in internal_cols if c in df.columns], errors="ignore"
        )
    return df


def _growth_stats_mean_for_sample_df(p: dict) -> pd.DataFrame:
    """Return growth stats averaged per sample name with mean and std columns."""
    nm_by_well = p.get("name") or {}
    gs = _growth_stats_per_well_df(p)
    if gs.empty:
        return gs

    gs["Sample Name"] = gs["well"].map(lambda w: (nm_by_well.get(w) or "").strip())
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
        "read_interval_min": "Read interval (minutes)",
        "pathlength_cm_": "Pathlength (cm)",
        "clip_time_series": "Time series clip (hours)",
        "remove_wells": "Excluded wells",
        "blank": "Blank subtraction",
        "outlier_detection": "Outlier detection (ECOD)",
        "outlier_threshold": "Outlier threshold (MAD z-score)",
        "growth_method": "Analysis method",
        "model_family": "Parametric model type",
        "model_type": "Growth model",
        "phase_boundary_method": "Phase boundary calculation",
        "window_points": "Window size (points)",
        "lag_cutoff": "Lag phase cutoff (fraction of μ_max)",
        "exp_cutoff": "Exponential phase cutoff (fraction of μ_max)",
        "min_data_points": "Minimum data points",
        "min_signal_to_noise": "Minimum signal-to-noise ratio",
    }

    for key, label in param_mapping.items():
        if key in params:
            value = params[key]
            # Format special cases
            if key == "clip_time_series" and isinstance(value, (tuple, list)):
                value = f"{value[0]} - {value[1]}"
            elif key == "remove_wells":
                if value is False or not value:
                    value = "None"
                else:
                    value = ", ".join(value)
            elif key == "blank":
                value = "Yes" if value else "No"
            elif key == "outlier_detection":
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
    add_annotations: bool = True,
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
            # baseline source here matches your earlier code; adjust if your structure differs
            test_data = plates.get("test_data", {})
            baseline = test_data.get("baseline")
            name_by_well = test_data.get("name", {})
            if baseline is not None and not baseline.empty:
                baseline_fig = plot_baseline(baseline, name_by_well=name_by_well)
                if baseline_fig is not None:
                    zf.writestr(
                        "plots/baseline.png",
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

                params = p.get("params") or {}
                sg_w = int(params.get("sg_window", 11))
                sg_p = int(params.get("sg_poly", 2))

                # use requested wells; default to all available only if plate absent from dict
                wells = wells_by_plate.get(pid, list(processed.keys()))

                for well in wells:
                    if well not in processed:
                        continue

                    plate_dir = f"plots/{pid}/wells"

                    if "Raw OD" in well_graphs:
                        all_growth_stats = p.get("growth_stats") or {}
                        gs = all_growth_stats.get(well) or {}
                        fit_parameters = p.get("fit_parameters") or {}

                        # Get the processed data for this well
                        d = processed.get(well)

                        if d is not None and not d.empty:
                            # Get time and OD data
                            t_raw, y_raw = _finite_sorted_xy(
                                d["Time"].to_numpy(), d["baseline_corrected"].to_numpy()
                            )

                            if t_raw.size > 0:
                                # Convert time to display unit
                                t_display = t_raw

                                # Create base plot using growthcurves
                                fig = gc_plot.create_base_plot(
                                    t_display, y_raw, scale="linear"
                                )

                                # Highlight data points used in analysis with red overlay
                                selected_times = gs.get("_used_fit_times")
                                if not selected_times:
                                    selected_times = t_raw.tolist()
                                fig = _add_lasso_selected_points(
                                    fig,
                                    t_raw,
                                    y_raw,
                                    selected_times,
                                    scale="linear",
                                )

                                # Annotate plot if requested
                                if add_annotations and not is_no_growth(gs) and gs:
                                    # Get fit result from stored parameters
                                    fit_result = fit_parameters.get(well)

                                    # Pass the stored growth stats and fit result directly
                                    # No need to reconstruct - use the original values from the fit
                                    fig = gc_plot.annotate_plot(
                                        fig,
                                        fit_result=fit_result,
                                        stats=gs,
                                        show_fitted_curve=annot_fitted_model,
                                        show_phase_boundaries=annot_phase,
                                        show_crosshairs=annot_umax_point,
                                        show_od_max_line=annot_od_max,
                                        show_n0_line=annot_baseline_od,
                                        show_umax_marker=annot_umax_point,
                                        show_tangent=annot_tangent,
                                        scale="linear",
                                    )

                                # Update axis labels
                                time_label = "Time (hours)"
                                fig.update_xaxes(title=time_label, showgrid=False)
                                fig.update_yaxes(
                                    title="OD600 (baseline-corrected)", showgrid=False
                                )

                                zf.writestr(
                                    f"{plate_dir}/growth_curves/{well}.png",
                                    _png(fig, well_width, well_height),
                                )

                    # Export derivative and growth rate plots
                    if "dOD/dt" in well_graphs or "Specific Growth Rate" in well_graphs:
                        d = processed.get(well)
                        all_growth_stats = p.get("growth_stats") or {}
                        gs = all_growth_stats.get(well) or {}
                        fit_parameters = p.get("fit_parameters") or {}

                        if d is not None and not d.empty:
                            t_raw, y_raw = _finite_sorted_xy(
                                d["Time"].to_numpy(), d["baseline_corrected"].to_numpy()
                            )

                            if t_raw.size > 0:
                                phase_boundaries = None
                                if not is_no_growth(gs) and gs:
                                    exp_phase_start = gs.get("exp_phase_start")
                                    exp_phase_end = gs.get("exp_phase_end")
                                    if (
                                        exp_phase_start is not None
                                        and exp_phase_end is not None
                                    ):
                                        phase_boundaries = (
                                            exp_phase_start,
                                            exp_phase_end,
                                        )

                                fit_result = fit_parameters.get(well)

                                if "dOD/dt" in well_graphs:
                                    fig = plot_derivative_metric(
                                        t=t_raw,
                                        N=y_raw,
                                        metric="dndt",
                                        fit_result=fit_result,
                                        sg_window=sg_w,
                                        sg_poly=sg_p,
                                        phase_boundaries=phase_boundaries,
                                        title=f"dN/dt – {well}",
                                    )
                                    if fig is not None:
                                        zf.writestr(
                                            f"{plate_dir}/curves_d1/{well}.png",
                                            _png(fig, well_width, well_height),
                                        )

                                if "Specific Growth Rate" in well_graphs:
                                    fig = plot_derivative_metric(
                                        t=t_raw,
                                        N=y_raw,
                                        metric="mu",
                                        fit_result=fit_result,
                                        sg_window=sg_w,
                                        sg_poly=sg_p,
                                        phase_boundaries=phase_boundaries,
                                        title=f"Specific growth rate – {well}",
                                    )
                                    if fig is not None:
                                        zf.writestr(
                                            f"{plate_dir}/curves_mu/{well}.png",
                                            _png(fig, well_width, well_height),
                                        )

    buf.seek(0)
    return buf.getvalue()
