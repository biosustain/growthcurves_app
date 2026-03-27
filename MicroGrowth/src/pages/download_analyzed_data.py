"""Download processed data tables and plot exports."""

import streamlit as st

from src.functions.common import require_plates
from src.ui_functions.download_analyzed_data_ui import (
    _render_global_plots_container,
    _render_tabulated_data_container,
    _render_well_level_plots_container,
    get_export_zip,
)
from src.ui_functions.ui_components import page_header_with_help

page_header_with_help(
    "Download Analyzed Data",
    """
**Workflow Overview — Download Analyzed Data**

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

# Initialize session state for ZIP bytes
if "export_zip_bytes" not in st.session_state:
    st.session_state.export_zip_bytes = None

# ---- Tabulated Data + Global Plots in one container ----
with st.container(border=True):
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
    c_add_annotations,
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


# Convert lists/dicts to tuples for caching
wells_tuple = tuple((k, tuple(v)) for k, v in sorted(wells_by_plate.items()))

# Use data parameter with a callable to build ZIP only when download is clicked
st.download_button(
    "Download Export ZIP",
    data=lambda: get_export_zip(
        plates,
        c_baseline_corrected,
        c_stats_per_well,
        c_stats_per_sample,
        c_params,
        c_plate,
        c_base,
        c_replicates,
        c_well,
        tuple(well_graphs) if well_graphs else (),
        tuple(selected_plate_ids) if selected_plate_ids else (),
        wells_tuple,
        c_add_annotations,
        annot_phase,
        annot_umax_point,
        annot_od_max,
        annot_baseline_od,
        annot_tangent,
        annot_fitted_model,
        global_width,
        global_height,
        well_width,
        well_height,
    ),
    file_name="export.zip",
    mime="application/zip",
    width="stretch",
    type="primary",
)
