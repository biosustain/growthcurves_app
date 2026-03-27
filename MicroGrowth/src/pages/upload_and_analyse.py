"""Upload inputs, configure analysis parameters, and run plate analysis."""

from src.functions.upload_functions import init_state
from src.ui_functions.uploads_ui import (
    ui_analysis_params,
    ui_preprocessing_params,
    ui_upload_and_analyse_header,
    ui_upload_files,
)

# ---------------- App ----------------
ss = init_state()

ui_upload_and_analyse_header()
ui_upload_files(ss)
ui_preprocessing_params(ss)
ui_analysis_params(ss)
