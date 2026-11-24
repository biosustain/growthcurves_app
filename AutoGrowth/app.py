import subprocess
from pathlib import Path

import streamlit as st
from ui_components import render_markdown

import growthcurve_app
from growthcurve_app.styling import (
    card_containers,
    green_gradient,
    green_navbar,
    red_buttons,
)

# set the default plotly chart config to export as svg
if not getattr(st.plotly_chart, "_svg_default", False):
    _orig_plotly_chart = st.plotly_chart

    def _plotly_chart(figure_or_data, *args, config=None, **kwargs):
        config = {"toImageButtonOptions": {"format": "svg"}, **(config or {})}
        return _orig_plotly_chart(figure_or_data, *args, config=config, **kwargs)

    _plotly_chart._svg_default = True
    st.plotly_chart = _plotly_chart


def _git_version(app_dir: Path) -> str | None:
    """Return '<YY.M.D>+<short-hash>' of the checked-out commit, if app_dir is in a repo.

    Used when deployed from a git checkout (e.g. Streamlit Community Cloud). Packaged
    apps have no .git and fall back to the VERSION file written during the build.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(app_dir),
                "log",
                "-1",
                "--format=%cd+%h",
                "--date=format:%y.%-m.%-d",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        top = subprocess.run(
            ["git", "-C", str(app_dir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    # Ignore a repo that merely contains an installed copy of the app.
    return out if out and Path(top).resolve() == app_dir.parent.resolve() else None


_APP_DIR = Path(__file__).resolve().parent
APP_VERSION = (
    _git_version(_APP_DIR)
    or _APP_DIR.joinpath("VERSION").read_text(encoding="utf-8").strip()
)

logo_source = str(Path(__file__).with_name("logo.svg"))

# General configurations
st.set_page_config(
    page_title="AutoGrowth",
    layout="wide",
    page_icon=logo_source,
    initial_sidebar_state="expanded",
)

st.logo(
    logo_source,
    link="https://github.com/biosustain/growthcurves_app/tree/main/AutoGrowth",
)

# Initialize constants
DEFAULT_CUSTOM_ID = "pioreactor_experiment"
st.session_state.setdefault("custom_id", DEFAULT_CUSTOM_ID)
st.session_state.setdefault("df_raw_od_data", None)

st.session_state["DEFAULT_XLABEL_TPS"] = "Timepoints (rounded)"
st.session_state["DEFAULT_XLABEL_REL"] = "Elapsed time (hours)"


# function creating the about page from a markdown file
def render_about():
    render_markdown("AutoGrowth/markdowns/about.md")


# Navigation
raw_data = st.Page("0_upload_data.py", title="Upload Data")
growth_analysis = st.Page("1_growth_analysis.py", title="Growth Analysis")
comparative_plots = st.Page("3_comparative_plots.py", title="Comparative Plots")
downloads_page = st.Page("0_downloads.py", title="Downloads")
about_page = st.Page(render_about, title="About")

# Sidebar
with st.sidebar:
    st.info("To reset the app, reload the page.")
    with st.container(border=True):
        st.markdown("#### Workflow")
        st.markdown(
            "1. Upload, process and review data\n"
            "2. Run the growth analysis\n"
            "3. Compare metrics across groups\n"
            "4. Export downloads"
        )
    st.caption(f"AutoGrowth app v{APP_VERSION}")

    debug_mode = st.checkbox(
        "Debug Mode",
        value=st.session_state.get("debug_mode", False),
        help=(
            "Enable debug mode for more verbose logging and"
            " additional information in the app."
        ),
    )
    st.session_state["debug_mode"] = debug_mode


# build multi-page app
pg = st.navigation(
    [
        raw_data,
        growth_analysis,
        comparative_plots,
        downloads_page,
        about_page,
    ],
    position="top",
)

red_buttons()
green_gradient()
green_navbar()
card_containers()

pg.run()
