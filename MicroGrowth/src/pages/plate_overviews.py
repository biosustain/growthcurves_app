"""Plate overview page: replicates and plate-level fits."""

import streamlit as st

from src.functions.common import require_plates
from src.ui_functions.plate_overviews_ui import (
    ui_replicates,
    ui_window_fits_plate_overview,
)
from src.ui_functions.ui_components import page_header_with_help

page_header_with_help(
    "Plate Overviews",
    """
**Workflow Overview — Plate Overviews**

Use this page as your **first quality check** after running the analysis on the Upload & Analyse page.

**Replicate growth curves (top section)**
Browse through your samples and inspect the grouped replicate curves. Consistent replicates should overlap closely — wide spread may indicate variability or issues with specific wells. Use this view to identify samples that may need attention in the Check Growth Fits page.

**Blank wells**
If blank subtraction is enabled, inspect the OD measurements for all blank wells (any well named BLANK or starting with BLANK). The mean blank value used for baseline correction is shown — verify it looks reasonable. Large variation across blank wells may affect the quality of baseline correction.

**Plate map fit overview (bottom section)**
A colour-coded plate map shows the predicted growth descriptor values across all wells. Use this to quickly spot problematic wells or patterns across the plate (e.g. edge effects, failed wells).

**What to do next**
If you spot inconsistent or poorly fitted wells, go to **Check Growth Fits** to inspect and manually correct individual wells before exporting your data.

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
""",
)

plates = require_plates()

ui_replicates(plates)

st.divider()

ui_window_fits_plate_overview(plates)
