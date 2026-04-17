import streamlit as st
from buttons import create_download_button
from plots import create_figure_bytes_to_download, plot_growth_data_w_mask
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth

DATA_DASHBOARD_HELP = """
Review processed upload outputs in one place:

1. Raw and filtered data tables
2. Filtered-data plot with removed-point overlay
3. Rolling-median table and line plot
"""

page_header_with_help("Data Dashboard", DATA_DASHBOARD_HELP)

df_raw_od_data = st.session_state.get("df_raw_od_data")
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")
df_rolling = st.session_state.get("df_rolling")
df_time_map = st.session_state.get("df_time_map")
masked = st.session_state.get("masked")
start_time = st.session_state.get("start_time")
processing_summary = st.session_state.get("upload_processing_summary_msg")
rolling_window = st.session_state.get("rolling_window")
st.session_state.setdefault("yaxis_scale", False)
st.session_state.setdefault("USE_ELAPSED_TIME_FOR_PLOTS", True)
use_same_yaxis_scale = bool(st.session_state.get("yaxis_scale", False))
use_elapsed_time = bool(st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", True))

USE_SAME_YAXIS_SCALE = False
TICKS_X_AXIS_INTERVAL = None
TICKS_X_AXIS_NBINS = 25
TICKS_Y_AXIS_NBINS = 5

if df_raw_od_data is None and df_rolling is None:
    show_warning_to_upload_data()
    st.stop()

with st.container(border=True):
    st.header("Summary Tables")
    raw_col, time_col = st.columns(2, gap="large")
    with raw_col:
        st.subheader("Raw OD data")
        if df_raw_od_data is None:
            st.info("Raw OD data preview appears after data is loaded.")
        else:
            st.dataframe(df_raw_od_data, width="stretch")

    with time_col:
        st.subheader("Timestamp to elapsed-time map")
        if df_time_map is None:
            st.info("Timestamp map is generated after preprocessing.")
        else:
            st.dataframe(df_time_map, width="stretch")

    download_buttons = st.columns(3)
    with download_buttons[0]:
        create_download_button(
            label="Download raw data  \n(long format)",
            data=df_raw_od_data.to_csv(index=True).encode("utf-8"),
            file_name="data_long_rounded_timestamps.csv",
            disabled=False,
            mime="text/csv",
        )
    with download_buttons[1]:
        create_download_button(
            label="Download raw data  \n(wide format)",
            data=df_wide_raw_od_data.to_csv(index=True).encode("utf-8"),
            file_name="data_wide_rounded_timestamps.csv",
            disabled=False,
            mime="text/csv",
        )
    with download_buttons[2]:
        df_wide_raw_od_data_filtered = st.session_state.get(
            "df_wide_raw_od_data_filtered"
        )
        # ! should never be None...
        if df_wide_raw_od_data_filtered is not None:
            create_download_button(
                label="Download filtered data",
                data=df_wide_raw_od_data_filtered.to_csv(index=True).encode("utf-8"),
                file_name="filtered_data_wide_rounded_timestamps.csv",
                disabled=False,
                mime="text/csv",
            )


if df_wide_raw_od_data is not None and masked is not None:
    with st.container(border=True):
        st.header("Filtered Data Plot")
        plot_option_cols = st.columns(2, gap="large")
        with plot_option_cols[0]:
            use_same_yaxis_scale = st.checkbox(
                "Use same y-axis for all reactors?",
                value=st.session_state.get(
                    "use_same_yaxis_scale", USE_SAME_YAXIS_SCALE
                ),
                help="Select plotting behaviour.",
            )
        with plot_option_cols[1]:
            use_elapsed_time = st.checkbox(
                "Use elapsed time (since start) as x-axis on plots?",
                value=use_elapsed_time,
                key="elapsed_time_option",
                help="If checked, elapsed time will be used as x-axis in plots.",
            )
        st.session_state["use_same_yaxis_scale"] = bool(use_same_yaxis_scale)
        st.session_state["USE_ELAPSED_TIME_FOR_PLOTS"] = bool(use_elapsed_time)
        # y-axis and x-axis tick options
        tick_cols = st.columns(3, gap="large")
        with tick_cols[0]:
            ticks_x_axis_interval = st.number_input(
                "X-axis tick interval",
                min_value=1,
                value=st.session_state.get(
                    "ticks_x_axis_interval", TICKS_X_AXIS_INTERVAL
                ),
                step=1,
                help=(
                    "Fixed interval between x-axis ticks. "
                    "Overrides 'X-axis tick count' if set."
                ),
            )
        with tick_cols[1]:
            ticks_x_axis_nbins = st.number_input(
                "X-axis tick count",
                min_value=1,
                value=st.session_state.get("ticks_x_axis_nbins", TICKS_X_AXIS_NBINS),
                step=1,
                disabled=True if ticks_x_axis_interval else False,
                help="Maximum number of x-axis ticks (used when interval is not set).",
            )
        with tick_cols[2]:
            ticks_y_axis_nbins = st.number_input(
                "Y-axis tick count",
                min_value=1,
                value=st.session_state.get("ticks_y_axis_nbins", TICKS_Y_AXIS_NBINS),
                step=1,
                help="Maximum number of y-axis ticks.",
            )
        st.session_state["yaxis_scale"] = bool(use_same_yaxis_scale)
        st.session_state["ticks_x_axis_interval"] = (
            ticks_x_axis_interval if ticks_x_axis_interval else None
        )
        st.session_state["ticks_x_axis_nbins"] = ticks_x_axis_nbins
        st.session_state["ticks_y_axis_nbins"] = ticks_y_axis_nbins

        if not use_same_yaxis_scale:
            st.warning("Using different y-axis scale for each reactor.")

        df_plot = df_wide_raw_od_data
        mask_plot = masked
        if use_elapsed_time:
            df_plot = piogrowth.reindex_w_relative_time(
                df=df_plot,
                start_time=start_time,
            )
            mask_plot = piogrowth.reindex_w_relative_time(
                df=mask_plot,
                start_time=start_time,
            )

        # Time window filtering
        st.divider()
        st.write("#### Time window filtering:")
        st.caption(
            "Select time windows to display. Data outside the selected windows "
            "will not appear in plots. Use the Upload Data page to re-process "
            "with different options."
        )

        # Reset stored ranges when elapsed-time mode changes to avoid type mismatch
        _prev_use_elapsed = st.session_state.get("_dashboard_prev_use_elapsed")
        if _prev_use_elapsed != use_elapsed_time:
            st.session_state.pop("dashboard_min_t", None)
            st.session_state.pop("dashboard_max_t", None)
            st.session_state.pop("dashboard_time_ranges", None)
        st.session_state["_dashboard_prev_use_elapsed"] = use_elapsed_time

        all_timepoints = df_plot.index
        _stored_min = st.session_state.get("dashboard_min_t", all_timepoints.min())
        _stored_max = st.session_state.get("dashboard_max_t", all_timepoints.max())
        if _stored_min not in all_timepoints:
            _stored_min = all_timepoints.min()
        if _stored_max not in all_timepoints:
            _stored_max = all_timepoints.max()

        time_window_cols = st.columns(
            [7, 1], gap="large", vertical_alignment="bottom"
        )
        with time_window_cols[0]:
            min_t, max_t = st.select_slider(
                "Select overall time window (inferred).",
                options=all_timepoints,
                value=(_stored_min, _stored_max),
            )
        with time_window_cols[1]:
            update_zero_timepoint = st.checkbox(
                "Reset T0",
                value=st.session_state.get("update_zero_timepoint", False),
                help=(
                    "If checked, a new zero time is set to the minimum "
                    "timestamp of the overall time window."
                ),
            )

        st.session_state["dashboard_min_t"] = min_t
        st.session_state["dashboard_max_t"] = max_t
        st.session_state["update_zero_timepoint"] = update_zero_timepoint

        df_plot = df_plot.loc[min_t:max_t]
        mask_plot = mask_plot.loc[min_t:max_t]

        with st.expander("Select time window per reactor"):
            st.info("Note: Minimum and maximum for slider are reactor specific!")
            dashboard_time_ranges = st.session_state.get("dashboard_time_ranges", {})
            for reactor in list(df_plot.columns):
                reactor_data = df_plot[reactor].dropna()
                if reactor_data.empty:
                    continue
                _options = reactor_data.index
                _stored_r = dashboard_time_ranges.get(reactor)
                if (
                    _stored_r is not None
                    and _stored_r[0] in _options
                    and _stored_r[1] in _options
                ):
                    _r_min, _r_max = _stored_r
                else:
                    _r_min, _r_max = _options.min(), _options.max()
                r_min_t, r_max_t = st.select_slider(
                    f"Select time window (inferred) for {reactor}."
                    " Bounded by overall time window.",
                    options=_options,
                    value=(_r_min, _r_max),
                )
                dashboard_time_ranges[reactor] = (r_min_t, r_max_t)
                reactor_in_window = df_plot.index.to_series().between(r_min_t, r_max_t)
                df_plot.loc[:, reactor] = df_plot[reactor].where(reactor_in_window)
            st.session_state["dashboard_time_ranges"] = dashboard_time_ranges

        # Figure showing the raw and masked growth data for each reactor
        fig = plot_growth_data_w_mask(
            df_plot,
            mask_plot,
            sharey=use_same_yaxis_scale,
            sharex=False,
            is_data_index=not use_elapsed_time,
            ticks_x_axis_interval=ticks_x_axis_interval,
            ticks_y_axis_nbins=ticks_y_axis_nbins,
            ticks_x_axis_nbins=ticks_x_axis_nbins,
        )
        st.markdown(
            """
            - <span style="color:red">red dots</span> red dots indicate points that
              were masked (removed) during data processing
            - <span style="color:blue">blue dots</span> indicate data that is
              retained before rolling median calculation
            """,
            unsafe_allow_html=True,
        )
        st.write(fig)
        create_download_button(
            label="Download figure as PDF",
            data=create_figure_bytes_to_download(fig, fmt="pdf"),
            file_name="data_overview.pdf",
            disabled=False,
            mime="application/pdf",
        )


# show summary message about data processing
if processing_summary:
    with st.container(border=True):
        st.subheader("Processing summary of OD readings")
        st.markdown(processing_summary)

# show rolling median table
if df_rolling is not None:
    with st.container(border=True):
        st.header("Rolling Median")
        if rolling_window is not None:
            st.subheader(
                f"Rolling median in window of {rolling_window}s using filtered OD data"
            )
        else:
            st.subheader("Rolling median using filtered OD data")
        st.write(df_rolling)
        create_download_button(
            data=df_rolling.to_csv(index=True).encode("utf-8"),
            label="Download rolling median data",
            file_name="rolling_median_on_filtered_wide_data_with_rounded_timestamps.csv",
            disabled=False,
            mime="text/csv",
        )

        # ! removing this plot for now
        # if not use_elapsed_time and start_time is not None:
        #     view = df_rolling.copy()
        #     view.index = start_time + pd.to_timedelta(view.index, unit="h")
        # else:
        #     view = df_rolling

        # ax = view.plot.line(style=".", ms=2)
        # st.write(ax.get_figure())

# ! This was moved to the place in the main page, not sidebar. Can be reverted.
# Download buttons in sidebar
# if st.session_state.get("df_raw_od_data") is not None:
#     download_data_button_in_sidebar(
#         "df_raw_od_data",
#         "Download raw data  \n(long format)",
#         file_name="data_long_rounded_timestamps.csv",
#     )

# if st.session_state.get("df_wide_raw_od_data") is not None:
#     download_data_button_in_sidebar(
#         "df_wide_raw_od_data",
#         "Download raw data  \n(wide format)",
#         file_name="data_wide_rounded_timestamps.csv",
#     )

# if st.session_state.get("df_wide_raw_od_data_filtered") is not None:
#     download_data_button_in_sidebar(
#         "df_wide_raw_od_data_filtered",
#         "Download filtered data",
#         file_name="filtered_data_wide_rounded_timestamps.csv",
#     )

# if df_rolling is not None:
#     download_data_button_in_sidebar(
#         "df_rolling",
#         "Download rolling median data",
#         file_name="rolling_median_on_filtered_wide_data_with_rounded_timestamps.csv",
#     )
