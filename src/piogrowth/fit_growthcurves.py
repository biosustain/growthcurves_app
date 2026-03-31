import itertools
import time

import growthcurves as gc
import pandas as pd


def datetimeindex_to_elapsed_hours(index):
    return (index - index[0]).total_seconds() / 3_600


def run_model_fitting_on_df(
    df,
    model_name="phenom_richards",
    window_points=500,
    spline_s=1000,
    n_fits=50,
    phase_boundary_method=None,
    lag_frac=0.15,
    exp_frac=0.15,
    **kwargs,
):
    stats_df = {}
    for col in df.columns:
        s = df[col].dropna()
        t = s.index
        start = time.time()
        _, stats_df[col] = gc.fit_model(
            t=t.to_numpy(),
            N=s.to_numpy(),
            model_name=model_name,
            window_points=window_points,
            spline_s=spline_s,
            n_fits=n_fits,
            phase_boundary_method=phase_boundary_method,
            lag_frac=lag_frac,
            exp_frac=exp_frac,
            **kwargs,
        )
        end_time = time.time()
        elapsed = end_time - start
        stats_df[col]["elapsed_time"] = elapsed
        stats_df[col]["model_name"] = model_name
        print(f"Finished fitting {model_name} on {col} in {elapsed:.2f} seconds.")

    stats_df = pd.DataFrame(stats_df).T
    return stats_df


def run_model_fitting_on_df_with_peaks(
    df,
    peaks,
    model_name="phenom_richards",
    window_points=500,
    spline_s=1000,
    n_fits=50,
    phase_boundary_method=None,
    lag_frac=0.15,
    exp_frac=0.15,
    **kwargs,
):
    stats_df = {}
    for col in df.columns:
        s = df[col].dropna()
        peaks_col = peaks[col]
        peak_timepoints = [s.index.min(), *peaks_col.dropna().index, s.index.max()]
        # st.write("Peak timepoints:", peak_timepoints)
        for start_seg, end_seg in itertools.pairwise(peak_timepoints):
            fit_start_time = time.time()
            s_segment = s.loc[start_seg:end_seg]
            t_segment = s_segment.index
            key = (col, f"{start_seg:.2f}-{end_seg:.2f}")
            _, stats_df[key] = gc.fit_model(
                t=t_segment.to_numpy(),
                N=s_segment.to_numpy(),
                model_name=model_name,
                window_points=window_points,
                spline_s=spline_s,
                n_fits=n_fits,
                phase_boundary_method=phase_boundary_method,
                lag_frac=lag_frac,
                exp_frac=exp_frac,
                **kwargs,
            )
            stats_df[key]["segment_start"] = start_seg
            stats_df[key]["segment_end"] = end_seg

            # Respect bounds of segment for exponential phase
            if (
                stats_df[key]["exp_phase_start"] is not None
                and stats_df[key]["exp_phase_start"] < start_seg
            ):
                stats_df[key]["exp_phase_start"] = start_seg
            if (
                stats_df[key]["exp_phase_end"] is not None
                and stats_df[key]["exp_phase_end"] > end_seg
            ):
                stats_df[key]["exp_phase_end"] = end_seg

            fit_end_time = time.time()
            elapsed = fit_end_time - fit_start_time
            stats_df[key]["elapsed_time"] = elapsed
            stats_df[key]["model_name"] = model_name
            print(
                f"Finished fitting {model_name} on {col} for segment "
                f"{start_seg} - {end_seg} in {elapsed:.2f} seconds."
            )

    stats_df = pd.DataFrame(stats_df).T
    stats_df.index.names = ["reactor", "segment"]
    return stats_df
