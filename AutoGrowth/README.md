# AutoGrowth Streamlit Application

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
| df_processed                 | Processed data with aggregation, filtering, optional filling, and OD adjustments; elapsed time (hours) as index |
| df_rolling                   | `df_processed` in the time windows of Step 2 (reads outside set to NaN); used by the analysis |

- the analysis uses `df_rolling`, which is in wide data format with elapsed time in
  hours as index. Plotting and fitting use the same processed values, including
  optional forward/backward filling.

### Intermediate wide-data formats

- measurements in wide-data format are in elapsed time in hours
- a combined start-time is used (filtering at zero will let the series start then after
  zero)
