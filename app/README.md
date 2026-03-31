# PioGrowth Streamlit Application

## Data Formats

- bioscatter measurements in long-data format with timestamps as timeseries:
  timestamp_localtime, pioreactor_unit, od_reading
  > Note that od_reading is not optical density, but the raw bioscatter measurement.
  > A translation to OD can be done using a calibration curve.
- timestamps can be rounded so that it will be easier to align measurements across
  reactors (e.g. 1 measurement every 5 seconds, but not exactly at the same second for
  all reactors)
- wide data means that the timestamps or elapsed time was rounded (for aligning)

| variable in session_state    | description                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| df_wide_raw_od_data_filtered | Raw bioscatter data in wide data format with timestamp_rounded as index                                   |
| df_rolling                   | Median rolling based on window size selected is in wide data format with time elapsed (in hours) as index |

- the analysis uses `df_rolling`, which is in wide data format with elapsed time in
  hours as index, and the median rolling applied (can be potentially other methods
  for smoothing)

### Intermediate wide-data formats

- measurements in wide-data format are in elapsed time in hours
- a combined start-time is used (filtering at zero will let the series start then after
  zero)
