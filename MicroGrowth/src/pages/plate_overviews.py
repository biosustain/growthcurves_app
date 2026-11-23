"""Plate overview page: replicates and plate-level fits."""

from src.functions.common import require_plates
from src.ui_functions.plate_overviews_ui import (
    ui_replicates,
    ui_window_fits_plate_overview,
)
from src.ui_functions.ui_components import page_header_with_help

page_header_with_help(
    "Plate Overviews",
    """
**Workflow Overview - Plate Overviews**

Use this page as your **first quality check** after running the analysis on the Upload & Analyse page.

**Share axes**
The toggle in the top right of each section puts every panel of a plot on the same time and OD scale (on) or scales each panel to its own data (off).

**Replicate growth curves (top section)**
Browse through your samples and inspect the grouped replicate curves. Consistent replicates should overlap closely - wide spread may indicate variability or issues with specific wells. Use this view to identify samples that may need attention in the Check Growth Fits page.

**Plate blanks and fits (bottom section)**
Choose a plate with the plate buttons at the top of the section.

**Blanks**
If blank subtraction is enabled, inspect the OD measurements for all blank wells (any well named BLANK or starting with BLANK). The group mean used for baseline correction is drawn as a line (hover to see its values) - verify it looks reasonable. Large variation across blank wells may affect the quality of baseline correction.

**Fits**
Model fits for every well, laid out in plate order. The wells with the poorest fits (highest RMSE relative to their OD increase) are shaded; choose what percentage of fitted wells with **Highlight poorest fits (%)**. Use this to quickly spot problematic wells or patterns across the plate (e.g. edge effects, failed wells).

**What to do next**
If you spot inconsistent or poorly fitted wells, go to **Check Growth Fits** to inspect and manually correct individual wells before exporting your data.

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
""",
)

plates = require_plates()

ui_replicates(plates)
ui_window_fits_plate_overview(plates)
