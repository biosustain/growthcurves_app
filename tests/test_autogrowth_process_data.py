import importlib.util
import io
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "AutoGrowth" / "process_data.py"
SPEC = importlib.util.spec_from_file_location("autogrowth_process_data", MODULE_PATH)
process_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(process_data)


def test_maybe_aggregate_high_frequency_raw_data_aggregates_sub_15s():
    df = pd.DataFrame(
        {
            "timestamp_localtime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:05",
                    "2024-01-01 00:00:10",
                    "2024-01-01 00:00:15",
                ]
            ),
            "pioreactor_unit": ["r1", "r1", "r1", "r1"],
            "od_reading": [0.1, 0.2, 0.3, 0.4],
        }
    )

    aggregated, was_aggregated, median_interval = (
        process_data.maybe_aggregate_high_frequency_raw_data(df)
    )

    assert was_aggregated is True
    assert median_interval == 5.0
    assert aggregated.shape[0] == 2
    assert aggregated["od_reading"].tolist() == [0.2, 0.4]


def test_maybe_aggregate_high_frequency_raw_data_skips_15s_or_above():
    df = pd.DataFrame(
        {
            "timestamp_localtime": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:15",
                    "2024-01-01 00:00:30",
                ]
            ),
            "pioreactor_unit": ["r1", "r1", "r1"],
            "od_reading": [0.1, 0.2, 0.3],
        }
    )

    not_aggregated, was_aggregated, median_interval = (
        process_data.maybe_aggregate_high_frequency_raw_data(df)
    )

    assert was_aggregated is False
    assert median_interval == 15.0
    pd.testing.assert_frame_equal(not_aggregated.reset_index(drop=True), df)


def test_read_od_adjustment_table_accepts_comma_delimited_csv():
    uploaded = io.StringIO("reactor,od\nr1,0.1\nr2,0.2\n")
    df = process_data.read_od_adjustment_table(uploaded)

    assert list(df.columns) == ["reactor", "od"]
    assert df["reactor"].tolist() == ["r1", "r2"]
    assert df["od"].tolist() == [0.1, 0.2]


def test_read_od_adjustment_table_accepts_semicolon_delimited_csv():
    uploaded = io.StringIO("reactor;od\nr1;0.1\nr2;0.2\n")
    df = process_data.read_od_adjustment_table(uploaded)

    assert list(df.columns) == ["reactor", "od"]
    assert df["reactor"].tolist() == ["r1", "r2"]
    assert df["od"].tolist() == [0.1, 0.2]


def test_read_od_adjustment_table_accepts_excel_file():
    df_in = pd.DataFrame({"reactor": ["r1", "r2"], "od": [0.1, 0.2]})
    uploaded = io.BytesIO()
    df_in.to_excel(uploaded, index=False)
    uploaded.name = "calibration.xlsx"
    uploaded.seek(0)

    df = process_data.read_od_adjustment_table(uploaded)

    assert list(df.columns) == ["reactor", "od"]
    assert df["reactor"].tolist() == ["r1", "r2"]
    assert df["od"].tolist() == [0.1, 0.2]
