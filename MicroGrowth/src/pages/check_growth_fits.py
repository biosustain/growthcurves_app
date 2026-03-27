"""Growth fit review page with well-level editor."""

from src.functions.common import require_plates
from src.ui_functions.check_growth_fits_ui import ui_window_fits_well_editor
from src.ui_functions.ui_components import page_header_with_help

page_header_with_help(
    "Check Growth Fits",
    """
**Workflow Overview — Check Growth Fits**

Use this page to **review and correct individual well fits** before exporting your data. Work through your wells systematically — especially any that looked inconsistent in the Plate Overviews page.

**Navigating between wells**
Use the plate and well selectors to move between wells. The plot shows the raw OD data, the fitted model or growth rate estimate, and the detected phase boundaries.

**Adjusting a fit**
- **Phase boundary sliders**: Drag the lag phase end and exponential phase end markers to reposition the boundaries manually
- **Max OD slider**: Adjust the estimated maximum OD (carrying capacity) if the automatic detection looks incorrect
- **Click and drag on the graph**: Select a subset of data points to re-calculate the maximum growth rate using only those points — useful when a portion of the curve is noisy or atypical

**Well-level actions**
- **Re-analyze**: Re-run the analysis on this well using the settings from the Upload & Analyse page (useful if you changed parameters and want to refit)
- **No growth**: Mark this well as showing no growth — all growth descriptors are set to 0 and the well is excluded from statistics
- **Delete**: Permanently exclude this well from the analysis and all downstream results

**What to do next**
Once you are satisfied with the fits, go to **Create Visualizations** to explore your results or **Download Analyzed Data** to export them.

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
""",
)

plates = require_plates()

ui_window_fits_well_editor(plates)
