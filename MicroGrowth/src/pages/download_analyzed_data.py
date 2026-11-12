"""Download processed data tables and plot exports."""

import streamlit as st
from src.functions.common import require_plates
from src.functions.export_functions import build_export_zip
from src.ui_functions.download_analyzed_data_ui import (
    _render_global_plots_container,
    _render_tabulated_data_container,
    _render_well_level_plots_container,
)
from src.ui_functions.ui_components import page_header_with_help

page_header_with_help(
    "Download Analyzed Data",
    """
**Workflow Overview - Download Analyzed Data**

Use this page to **export your analysis results** as a ZIP file containing data tables and plots. Check the boxes for the content you want to include, then click the download button at the bottom of the page.

**Data tables**
- **Baseline corrected measurements**: Time series OD data after blank subtraction and pathlength correction, for all included wells
- **Per-well statistics**: Growth descriptors (μ_max, lag time, doubling time, etc.) for each individual well
- **Per-sample statistics**: Growth descriptors averaged across replicates for each sample
- **Analysis parameters**: The parameter settings used for the analysis, useful for record-keeping and reproducibility

**Well-level plots**
- Generates annotated growth curve plots for each well
- Use the **annotation options** to control which overlays are shown on each plot (phase boundaries, μ_max point, Max OD line, baseline OD, tangent line, fitted model)
- Use the plate and well filters to export only a subset of wells rather than the full plate

**Global plots**
- **Plate map**: Colour-coded overview of growth descriptor values across all wells
- **Baseline OD**: Plot of blank well OD measurements used for baseline correction
- **Replicate curves**: Grouped replicate growth curves for each sample

Click **Download Export ZIP** to package all selected content into a single ZIP file.
""",
)

plates = require_plates()
plate_ids = list(plates.keys())

# ---- Tabulated Data + Global Plots in one container ----
with st.container(border=True, key="card_download_page_1"):
    top_left, top_right = st.columns(2)

    with top_left:
        (
            c_baseline_corrected,
            c_stats_per_well,
            c_stats_per_sample,
            c_params,
        ) = _render_tabulated_data_container()

    with top_right:
        (
            c_base,
            c_plate,
            c_replicates,
            global_width,
            global_height,
        ) = _render_global_plots_container()

# ---- Well Level Plots full-width below ----
(
    c_well,
    annot_phase,
    annot_umax_point,
    annot_od_max,
    annot_baseline_od,
    annot_tangent,
    annot_fitted_model,
    well_width,
    well_height,
    well_graphs,
    selected_plate_ids,
    wells_by_plate,
) = _render_well_level_plots_container(plates, plate_ids)

# A callable, so the ZIP is built only when the download is clicked
st.download_button(
    "Download Export ZIP",
    data=lambda: build_export_zip(
        plates,
        include_baseline_corrected=c_baseline_corrected,
        include_stats_per_well=c_stats_per_well,
        include_stats_per_sample=c_stats_per_sample,
        include_params=c_params,
        include_plate_view=c_plate,
        include_baseline_plots=c_base,
        include_replicates=c_replicates,
        include_well_plots=c_well,
        well_graphs=well_graphs,
        selected_plate_ids=selected_plate_ids,
        wells_by_plate=wells_by_plate,
        annot_phase=annot_phase,
        annot_umax_point=annot_umax_point,
        annot_od_max=annot_od_max,
        annot_baseline_od=annot_baseline_od,
        annot_tangent=annot_tangent,
        annot_fitted_model=annot_fitted_model,
        baseline_width=global_width,
        baseline_height=global_height,
        plate_width=global_width,
        plate_height=global_height,
        well_width=well_width,
        well_height=well_height,
    ),
    file_name="export.zip",
    mime="application/zip",
    width="stretch",
    type="primary",
)
