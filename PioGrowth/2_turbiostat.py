import functools
import inspect
import itertools
import time
from io import BytesIO

import growthcurves as gc
import pandas as pd
import streamlit as st
from buttons import create_download_button
from growthcurves_options import (
    render_parameter_calculation_table_upload_style,
    render_upload_style_analysis_options,
)
from plots import create_figure_bytes_to_download, plot_growth_data_w_peaks
from ui_components import page_header_with_help, show_warning_to_upload_data

from piogrowth.fit_spline import get_smoothing_range
from piogrowth.turbistat import detect_peaks


## Logic and PLOTTING
def create_summary(maxima: dict[str, pd.Series]) -> pd.DataFrame:
    """Create a summary DataFrame from the maxima dictionary."""
    df_summary = pd.DataFrame(maxima).stack()
    df_summary.index.names = ["timestamp", "pioreactor_unit"]
    df_summary.name = "OD_value"
    df_summary = df_summary.to_frame()
    return df_summary


def get_values_from_df(df_wide: pd.DataFrame, indices: pd.MultiIndex) -> pd.DataFrame:
    """Get values from the wide DataFrame based on the index of the summary DataFrame."""
    return df_wide.loc[indices.get_level_values("timestamp")].stack().loc[indices]


def reset_metadata():
    st.session_state["df_meta"] = None
    st.session_state["turbidostat_meta_upload_bytes"] = None
    st.session_state["turbidostat_meta_upload_name"] = None


def _build_turbidostat_fit_kwargs(
    model_name: str,
    n_fits: int,
    window_points: int,
    spline_s: int,
    smooth_mode: str,
) -> dict:
    """Build model-specific kwargs for growthcurves.fit_model."""
    fit_kwargs = {}
    if model_name == "sliding_window":
        fit_kwargs["n_fits"] = n_fits
        fit_kwargs["window_points"] = window_points
    elif model_name == "spline":
        fit_kwargs["window_points"] = window_points
        if "smooth" in NON_PARAMETRIC_FIT_PARAMS:
            fit_kwargs["smooth"] = smooth_mode
        if "spline_s" in NON_PARAMETRIC_FIT_PARAMS:
            fit_kwargs["spline_s"] = spline_s
    return fit_kwargs


def _run_model_fitting_on_df_with_peaks_compat(
    df: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    model_name: str,
    n_fits: int,
    spline_s: int,
    smooth_mode: str,
    window_points: int,
    phase_boundary_method: str | None,
    lag_threshold: float,
    exp_threshold: float,
) -> pd.DataFrame:
    """Run segmented fitting with growthcurves kwargs compatible across model APIs."""
    stats_dict = {}
    fit_kwargs = _build_turbidostat_fit_kwargs(
        model_name=model_name,
        n_fits=n_fits,
        window_points=window_points,
        spline_s=spline_s,
        smooth_mode=smooth_mode,
    )

    for col in df.columns:
        s = df[col].dropna()
        peaks_col = peaks[col] if col in peaks else pd.Series(dtype="object")
        peak_timepoints = [s.index.min(), *peaks_col.dropna().index, s.index.max()]

        for start_seg, end_seg in itertools.pairwise(peak_timepoints):
            fit_start = time.time()
            s_segment = s.loc[start_seg:end_seg]
            t_segment = s_segment.index.to_numpy()
            n_segment = s_segment.to_numpy()
            key = (col, f"{start_seg:.2f}-{end_seg:.2f}")

            _, stats = gc.fit_model(
                t=t_segment,
                N=n_segment,
                model_name=model_name,
                phase_boundary_method=phase_boundary_method,
                lag_threshold=lag_threshold,
                exp_threshold=exp_threshold,
                **fit_kwargs,
            )

            stats["segment_start"] = start_seg
            stats["segment_end"] = end_seg
            if (
                stats.get("exp_phase_start") is not None
                and stats["exp_phase_start"] < start_seg
            ):
                stats["exp_phase_start"] = start_seg
            if (
                stats.get("exp_phase_end") is not None
                and stats["exp_phase_end"] > end_seg
            ):
                stats["exp_phase_end"] = end_seg
            stats["elapsed_time"] = time.time() - fit_start
            stats["model_name"] = model_name
            stats_dict[key] = stats

    stats_df = pd.DataFrame(stats_dict).T
    stats_df.index.names = ["reactor", "segment"]
    return stats_df


########################################################################################
# state

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_time_map = st.session_state.get("df_time_map")
no_data_uploaded = st.session_state.get("df_rolling") is None
df_rolling = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")
df_meta = st.session_state.get("df_meta")
turbidostat_meta_bytes = st.session_state.get("turbidostat_meta_upload_bytes")
turbidostat_meta_name = st.session_state.get("turbidostat_meta_upload_name")
col_timestamp = st.session_state.get("turbidostat_timestamp_col", "timestamp_localtime")
col_reactors = st.session_state.get("turbidostat_reactor_col", "pioreactor_unit")
col_message = st.session_state.get("turbidostat_message_col", "message")
round_time = st.session_state.get("round_time", 60)

DEFAULT_XLABEL_TPS = st.session_state.get("DEFAULT_XLABEL_TPS", "Timepoints (rounded)")
DEFAULT_XLABEL_REL = st.session_state.get("DEFAULT_XLABEL_REL", "Elapsed time (hours)")
NON_PARAMETRIC_FIT_PARAMS = set(
    inspect.signature(gc.non_parametric.fit_non_parametric).parameters
)
########################################################################################
# UI


TURBIDOSTAT_HELP = """
Analyse OD600 measurements in turbidostat mode and identify high-growth periods.

Workflow:
1. Configure peak detection options
2. Configure growth analysis options
3. Run analysis and inspect peaks/fit visualizations
4. Review and download summary outputs

In turbidostat mode, growth is diluted to maintain microorganisms in a
continuous growth state.
"""

page_header_with_help("Turbidostat Growth Analysis", TURBIDOSTAT_HELP)
if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

### Configuration ######################################################################
has_uploaded_metadata = turbidostat_meta_bytes is not None
if not has_uploaded_metadata:
    st.session_state["turbidostat_use_uploaded_peaks"] = False

with st.container(border=True):
    st.header("Step 1. Configure peak detection")
    checkbox_cols = st.columns(2, gap="large")
    with checkbox_cols[0]:
        use_uploaded_peak_times = st.checkbox(
            "Use uploaded peak times",
            value=has_uploaded_metadata,
            disabled=not has_uploaded_metadata,
            key="turbidostat_use_uploaded_peaks",
            help=(
                "Enable to use peak times from uploaded metadata. Disable to run "
                "automatic peak detection."
            ),
        )
    with checkbox_cols[1]:
        remove_downward_trending = st.checkbox(
            label="Remove downward trending data points (negative OD changes) globally",
            value=True,
            key="remove_downward_trending",
        )

    minimum_peak_height = None
    minimum_distance = int(st.session_state.get("turbiostat_distance", 300))
    if use_uploaded_peak_times:
        meta_label = (
            turbidostat_meta_name if turbidostat_meta_name else "uploaded_metadata.csv"
        )
        st.info(f"Uploaded peak-time file: `{meta_label}`")
    else:
        if not has_uploaded_metadata:
            st.caption(
                "No dilution metadata uploaded. Upload an optional CSV on the Upload Data page (Step 2)."
            )
            st.page_link(
                "0_upload_data.py",
                label="Go to Upload Data",
                icon=":material/upload:",
            )
        st.markdown("Automatic peak detection options")
        minimum_peak_height = st.number_input(
            label=(
                "Minimum peak height (in OD units) - used only if no metadata provided. "
                "No value uses adaptive thresholding based on the maximum of an OD curve."
                " The default is one-fifth of the maximum OD value in a time series."
            ),
            min_value=0.0,
            value=None,
        )
        minimum_distance = st.number_input(
            label="Minimum distance between peaks (in number of measurement timepoints)",
            min_value=3,
            value=300,
            step=1,
            key="turbiostat_distance",
        )

smoothing_range = get_smoothing_range(len(df_rolling))

with st.container(border=True):
    st.header("Step 2. Configure and Run Analysis")
    analysis_options = render_upload_style_analysis_options(
        s_min=smoothing_range.s_min, s_max=smoothing_range.s_max
    )
    render_parameter_calculation_table_upload_style(analysis_options)
    run_analysis = st.button("Run Analysis", type="primary", width="stretch")

with st.sidebar:
    st.button("Reset uploaded metadata", on_click=reset_metadata)

### Error messages
if st.session_state.get("show_error"):
    with st.container(border=True):
        st.error(
            "Could not find column in metadata. Please check the column names."
            " The selection was adjusted to the available columns."
        )

########################################################################################
### On Submission of form parameters
if not run_analysis:
    st.stop()

st.session_state["show_error"] = False

if turbidostat_meta_bytes is not None:
    df_meta = pd.read_csv(
        BytesIO(turbidostat_meta_bytes), parse_dates=["timestamp_localtime"]
    ).convert_dtypes()
    df_meta.insert(
        0,
        "timestamp_rounded",
        df_meta["timestamp_localtime"].dt.round(
            f"{round_time}s",
        ),
    )
    mask_dilution_events = df_meta["event_name"] == "DilutionEvent"
    if not mask_dilution_events.all():
        st.info('Showing only rows with "DilutionEvent" in column "event_name".')
        df_meta = df_meta.loc[mask_dilution_events]
    st.session_state["df_meta"] = df_meta
    df_meta["elapsed_time_in_seconds"] = (
        df_meta["timestamp_localtime"] - start_time
    ).dt.total_seconds()
    df_meta["elapsed_time_in_hours"] = df_meta["elapsed_time_in_seconds"] / 3600.0
else:
    df_meta = None
    st.session_state["df_meta"] = None

# Peak detection: Uploaded peak times or automatic scipy.signal.find_peaks
if use_uploaded_peak_times:
    with st.container(border=True):
        st.subheader("Step 3. Detect Peaks from Uploaded Metadata")
        st.write("Data is rounded to match OD data timepoints.")
        if df_meta is None:
            st.error(
                "Uploaded metadata not available. Disable 'Use uploaded peak times' "
                "or upload metadata on the Upload Data page."
            )
            st.stop()
        # if this fails user needs to pick out names of columns in form
        if not (len(set((col_timestamp, col_reactors, col_message))) == 3):
            st.error(
                "Selected columns from uploaded dilution metadata cannot overlap."
                " Use for each a unique column."
            )
            st.stop()
        try:
            peaks = df_meta.pivot(
                index="elapsed_time_in_hours",
                columns=col_reactors,
                values=col_message,
            )
            st.session_state["turbidostat_timestamp_col"] = col_timestamp
            st.session_state["turbidostat_reactor_col"] = col_reactors
            st.session_state["turbidostat_message_col"] = col_message
        except KeyError:
            st.session_state["show_error"] = True
            st.rerun()
else:
    with st.container(border=True):
        st.subheader("Step 3. Detect Peaks Automatically")
        st.write(
            "Note: Peaks are detected using "
            "[`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)"
        )
        if minimum_peak_height is not None:
            st.write(
                "Minimum distance between peaks: "
                f"{minimum_peak_height} number of measured timepoints"
            )
    _detect_peaks = functools.partial(
        detect_peaks,
        distance=minimum_distance,
        prominence=minimum_peak_height,
    )
    peaks = df_rolling.apply(_detect_peaks)
    st.session_state["peaks"] = peaks

if remove_downward_trending:
    # Remove downward trending data globally on averaged data
    df_rolling = df_rolling.mask(df_rolling.diff().le(0))
    st.info(
        "Downward trending data points (negative OD changes) were removed globally."
    )

selected_model = analysis_options["selected_model"]
spline_smoothing_value = analysis_options["spline_smoothing_value"]
fits_sliding_window = analysis_options["n_fits"]
window_points = analysis_options["window_points"]
phase_boundary_method = analysis_options["phase_boundary_method"]
lag_cutoff = analysis_options["lag_cutoff"]
exp_cutoff = analysis_options["exp_cutoff"]
smooth_mode = analysis_options.get("smooth_mode", "fast")

# views for plotting to allow for elapsed time option
xlabel = DEFAULT_XLABEL_REL


# ? should the one with negative values removed stored globally?
st.session_state["df_rolling_turbidostat"] = df_rolling

with st.spinner(text="Fitting curves...", show_time=True):
    stats_df = _run_model_fitting_on_df_with_peaks_compat(
        df_rolling,
        peaks,
        model_name=selected_model,
        n_fits=fits_sliding_window,
        spline_s=spline_smoothing_value,
        smooth_mode=smooth_mode,
        window_points=window_points,
        phase_boundary_method=phase_boundary_method,
        exp_threshold=exp_cutoff,
        lag_threshold=lag_cutoff,
    )

    fig, axes = plot_growth_data_w_peaks(df_rolling, peaks, is_data_index=False)

    time_at_mu_max = stats_df["time_at_umax"]

    axes = axes.flatten()
    for ax, _col in zip(axes, df_rolling.columns):
        s_maxima = time_at_mu_max.loc[_col]
        for x in s_maxima:
            ax.axvline(x=x, color="red", linestyle="--")
    for ax, col in zip(axes, df_rolling.columns):
        sub_df = stats_df.loc[col]
        range_exp_phase = list(zip(sub_df["exp_phase_start"], sub_df["exp_phase_end"]))
        for _start, _end in range_exp_phase:
            ax.axvspan(_start, _end, color="gray", alpha=0.2)

with st.container(border=True):
    st.subheader("Step 4. Review Fitted Curves and Peaks")
    st.markdown(
        """
        - <span style="color:#1f77b4;"><b>Blue points</b></span>: OD data used for
                analysis (after optional removal of downward trending points)</li>
        - <span style="color:#7f7f7f;"><b>Grey dashed lines</b></span>: Detected
        peaks indicating potential dilution events, either from uploaded metadata
        or automatic detection</li>
        - <span style="color:#d62728;"><b>Red dashed lines</b></span>: Maximum
        growth timepoint for turbiostat window</li>
        - <span style="color:#9e9e9e;"><b>Gray shaded areas</b></span>: Exponential
        growth phases as determined by fitted model</li>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(fig)

    create_download_button(
        label="Download figure for fitted splines as PDF",
        data=create_figure_bytes_to_download(fig, fmt="pdf"),
        file_name="data_with_peaks_and_mu_max.pdf",
        disabled=False,
        mime="application/pdf",
    )


# Summary table
### Summary Table ##################################################################
with st.container(border=True):
    st.subheader("Step 5. Summary of High Growth Periods")
    st.write(
        f"The start time was {start_time}. Timepoints are relative to this start time."
    )
    st.dataframe(stats_df, width="stretch")

st.session_state["batch_analysis_summary_df"] = stats_df

with st.container(border=True):
    st.subheader("Download Tables")
    dl_col1, dl_col2, dl_col3 = st.columns(3, gap="small")
    with dl_col1:
        create_download_button(
            label="Download turbidostat data used",
            data=df_rolling.to_csv(index=True).encode("utf-8"),
            file_name="df_rolling_turbidostat.csv",
            mime="text/csv",
            disabled=False,
        )
    with dl_col2:
        create_download_button(
            label="Download detected peaks",
            data=peaks.to_csv(index=True).encode("utf-8"),
            file_name="peaks.csv",
            mime="text/csv",
            disabled=False,
        )
    with dl_col3:
        create_download_button(
            label="Download summary",
            data=stats_df.to_csv(index=True).encode("utf-8"),
            file_name="batch_analysis_summary_df.csv",
            mime="text/csv",
            disabled=False,
        )

if df_meta is not None:
    create_download_button(
        label="Download uploaded metadata (filtered to dilution events)",
        data=df_meta.to_csv(index=False).encode("utf-8"),
        file_name="turbidostat_uploaded_metadata_filtered.csv",
        mime="text/csv",
        disabled=False,
    )
