import streamlit as st

from src.functions.export_functions import build_export_zip
from src.ui_functions.blank_grouping_ui import get_well_selector_wells, ui_well_selector


def _render_tabulated_data_container():
    st.header("Tabulated Data")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_baseline_corrected = st.checkbox(
            "Baseline-corrected", value=True, key="table_baseline_corrected"
        )
    with desc_col:
        st.caption("Time series OD600 values for each well")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_stats_per_well = st.checkbox(
            "Stats per well", value=True, key="table_stats_per_well"
        )
    with desc_col:
        st.caption("Max growth rate, lag time, max OD, and phase boundaries per well")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_stats_per_sample = st.checkbox(
            "Stats per sample", value=True, key="table_stats_per_sample"
        )
    with desc_col:
        st.caption("Statistics averaged across replicates")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_params = st.checkbox("Analysis parameters", value=True, key="table_params")
    with desc_col:
        st.caption("All analysis settings used (read interval, pathlength, etc.)")

    return c_baseline_corrected, c_stats_per_well, c_stats_per_sample, c_params


def _render_well_level_plots_container(plates: dict, plate_ids: list[str]):
    with st.container(border=True):
        title_col, cb_col = st.columns(
            [1, 4],
            vertical_alignment="center",
        )

        with title_col:
            st.header("Well Level Plots")
        with cb_col:
            c_well = st.checkbox("Include well plots", value=False, key="well_checkbox")

        if c_well:
            c_add_annotations = True
            wells_by_plate: dict[str, list[str]] = {}

            options_col, map_col = st.columns([1, 2])

            with options_col:
                st.caption(
                    "Individual well growth curves with annotations and derivative plots"
                )
                with st.popover("Choose annotations to include", width="stretch"):
                    st.caption("Choose which annotations to include on well plots:")
                    plot_col, checkbox_col = st.columns([2, 1])
                    with plot_col:
                        st.image("info_plots/annotations.png", width="stretch")
                    with checkbox_col:
                        annot_phase = st.checkbox(
                            "Phase boundaries",
                            value=True,
                            key="annot_phase_boundaries",
                        )
                        annot_umax_point = st.checkbox(
                            "Max growth rate point",
                            value=True,
                            key="annot_umax_point",
                        )
                        annot_od_max = st.checkbox(
                            "Max OD",
                            value=True,
                            key="annot_od_max",
                        )
                        annot_baseline_od = st.checkbox(
                            "Baseline OD",
                            value=True,
                            key="annot_baseline_od",
                        )
                        annot_tangent = st.checkbox(
                            "Tangent line at max growth",
                            value=False,
                            key="annot_tangent",
                        )
                        annot_fitted_model = st.checkbox(
                            "Fitted model curve",
                            value=True,
                            key="annot_fitted_model",
                        )

                well_graphs = st.segmented_control(
                    "Well traces to include",
                    options=["Raw OD", "dOD/dt", "Specific Growth Rate"],
                    default=["Raw OD", "dOD/dt", "Specific Growth Rate"],
                    selection_mode="multi",
                    key="well_graphs",
                    width="stretch",
                )

                ww_col, wh_col = st.columns(2)
                well_width = ww_col.number_input(
                    "Width (px)",
                    min_value=400,
                    max_value=3000,
                    value=1200,
                    step=100,
                    key="well_width",
                )
                well_height = wh_col.number_input(
                    "Height (px)",
                    min_value=300,
                    max_value=2500,
                    value=800,
                    step=100,
                    key="well_height",
                )

                selected_plate_ids = st.multiselect(
                    "Plates to include",
                    options=plate_ids,
                    default=plate_ids,
                    key="selected_plates",
                )

                # Navigation buttons at the bottom of the options column
                if selected_plate_ids:
                    plate_idx = st.session_state.get("well_plot_plate_idx", 0)
                    plate_idx = min(plate_idx, len(selected_plate_ids) - 1)
                    st.session_state["well_plot_plate_idx"] = plate_idx

                    if len(selected_plate_ids) > 1:
                        prev_col, label_col, next_col = st.columns(
                            [1, 4, 1], vertical_alignment="center"
                        )
                        if prev_col.button(
                            "← Prev",
                            disabled=plate_idx == 0,
                            key="well_plot_prev",
                            width="stretch",
                        ):
                            st.session_state["well_plot_plate_idx"] = plate_idx - 1
                            st.rerun()
                        label_col.markdown(
                            f"**{selected_plate_ids[plate_idx]}**  "
                            f"({plate_idx + 1} / {len(selected_plate_ids)})"
                        )
                        if next_col.button(
                            "Next →",
                            disabled=plate_idx == len(selected_plate_ids) - 1,
                            key="well_plot_next",
                            width="stretch",
                        ):
                            st.session_state["well_plot_plate_idx"] = plate_idx + 1
                            st.rerun()
                else:
                    plate_idx = 0

                st.caption(
                    "Click a well to toggle it for including in the zip file; click a second well to select a rectangle. "
                    "Green = included · red = excluded · dark grey = no sample."
                )

            with map_col:
                if selected_plate_ids:
                    pid = selected_plate_ids[plate_idx]
                    plate_data = plates.get(pid) or {}
                    processed = plate_data.get("processed_data") or {}
                    available_wells = sorted(processed.keys())
                    wells_by_plate[pid] = ui_well_selector(
                        plate_id=pid,
                        available_wells=available_wells,
                        name_by_well=plate_data.get("name") or {},
                    )
                    # Plates not currently displayed: use stored selection if available,
                    # otherwise fall back to all wells (plate never visited)
                    for other_pid in selected_plate_ids:
                        if other_pid not in wells_by_plate:
                            other_processed = (plates.get(other_pid) or {}).get(
                                "processed_data"
                            ) or {}
                            available = sorted(other_processed.keys())
                            stored = get_well_selector_wells(other_pid, available)
                            wells_by_plate[other_pid] = (
                                stored if stored is not None else available
                            )
        else:
            st.caption(
                "Individual well growth curves with annotations and derivative plots"
            )
            # Set defaults when well plots are not included
            c_add_annotations = True
            annot_phase = True
            annot_umax_point = True
            annot_od_max = True
            annot_baseline_od = True
            annot_tangent = False
            annot_fitted_model = True
            well_width = 1200
            well_height = 800
            well_graphs = []
            selected_plate_ids = []
            wells_by_plate = {}

    return (
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
    )


def _render_global_plots_container():
    st.header("Global Plots")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_base = st.checkbox("Baseline", value=True, key="baseline_checkbox")
    with desc_col:
        st.caption("Blank well OD measurements and mean baseline over time")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_plate = st.checkbox("Plate view", value=False, key="plate_checkbox")
    with desc_col:
        st.caption("96-well plate overview showing all wells with fitted growth curves")

    cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
    with cb_col:
        c_replicates = st.checkbox("Replicates", value=True, key="replicates_checkbox")
    with desc_col:
        st.caption("Replicate growth curves grouped by sample name")

    if c_base or c_plate or c_replicates:
        col_w, col_h = st.columns(2)
        global_width = col_w.number_input(
            "Width (px)",
            min_value=400,
            max_value=3000,
            value=1200,
            step=100,
            key="global_plot_width",
        )
        global_height = col_h.number_input(
            "Height (px)",
            min_value=300,
            max_value=2500,
            value=800,
            step=100,
            key="global_plot_height",
        )
    else:
        global_width = 1200
        global_height = 800

    return c_base, c_plate, c_replicates, global_width, global_height


# Lazy function that builds ZIP only when called by download button
@st.cache_data(show_spinner="Building ZIP file...")
def get_export_zip(
    _plates,
    include_baseline_corrected,
    include_stats_per_well,
    include_stats_per_sample,
    include_params,
    include_plate_view,
    include_baseline_plots,
    include_replicates,
    include_well_plots,
    well_graphs_tuple,
    selected_plates_tuple,
    wells_tuple,
    add_annotations,
    annot_phase,
    annot_umax_point,
    annot_od_max,
    annot_baseline_od,
    annot_tangent,
    annot_fitted_model,
    global_w,
    global_h,
    well_w,
    well_h,
):
    # Convert tuples back to appropriate types
    wells_dict = {}
    for k, v in wells_tuple:
        wells_dict[k] = list(v)

    return build_export_zip(
        _plates,
        include_baseline_corrected=include_baseline_corrected,
        include_stats_per_well=include_stats_per_well,
        include_stats_per_sample=include_stats_per_sample,
        include_params=include_params,
        include_plate_view=include_plate_view,
        include_baseline_plots=include_baseline_plots,
        include_replicates=include_replicates,
        include_well_plots=include_well_plots,
        well_graphs=list(well_graphs_tuple) if well_graphs_tuple else [],
        selected_plate_ids=list(selected_plates_tuple) if selected_plates_tuple else [],
        wells_by_plate=wells_dict,
        add_annotations=add_annotations,
        annot_phase=annot_phase,
        annot_umax_point=annot_umax_point,
        annot_od_max=annot_od_max,
        annot_baseline_od=annot_baseline_od,
        annot_tangent=annot_tangent,
        annot_fitted_model=annot_fitted_model,
        baseline_width=global_w,
        baseline_height=global_h,
        plate_width=global_w,
        plate_height=global_h,
        well_width=well_w,
        well_height=well_h,
    )
