import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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
    expected = pd.DataFrame(
        {
            "timestamp_localtime": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 00:00:15"]
            ),
            "pioreactor_unit": ["r1", "r1"],
            "od_reading": [0.2, 0.4],
        }
    ).convert_dtypes()
    aggregated, was_aggregated, median_interval = (
        process_data.maybe_aggregate_high_frequency_raw_data(df)
    )

    assert was_aggregated is True
    assert median_interval == 5.0
    assert aggregated.shape[0] == 2
    assert aggregated["od_reading"].tolist() == [0.2, 0.4]
    assert pd.testing.assert_frame_equal(aggregated, expected) is None


def test_maybe_aggregate_high_frequency_raw_data_aggregates_sub_15s_non_zero_start():
    df = pd.DataFrame(
        {
            "timestamp_localtime": pd.to_datetime(
                [
                    "2024-01-01 00:00:03",
                    "2024-01-01 00:00:13",
                    "2024-01-01 00:00:18",
                    "2024-01-01 00:00:22",
                ]
            ),
            "pioreactor_unit": ["r1", "r1", "r1", "r1"],
            "od_reading": [0.1, 0.2, 0.3, 0.4],
        }
    )
    expected = pd.DataFrame(
        {
            "timestamp_localtime": pd.to_datetime(
                ["2024-01-01 00:00:00", "2024-01-01 00:00:15"]
            ),
            "pioreactor_unit": ["r1", "r1"],
            "od_reading": [0.15, 0.35],
        }
    ).convert_dtypes()
    aggregated, was_aggregated, median_interval = (
        process_data.maybe_aggregate_high_frequency_raw_data(df)
    )

    assert was_aggregated is True
    assert median_interval == 5.0
    assert aggregated.shape[0] == 2
    assert pd.testing.assert_frame_equal(aggregated, expected) is None


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


@pytest.mark.parametrize("aggregate_high_frequency", [True, False])
def test_process_od_pioreactor_aligns_unaligned_timestamps(
    monkeypatch, aggregate_high_frequency
):
    monkeypatch.setattr(process_data, "st", SimpleNamespace(session_state={}))
    uploaded = io.StringIO(
        "timestamp_localtime,pioreactor_unit,od_reading\n"
        "2024-01-01 00:00:03,r1,0.1\n"
        "2024-01-01 00:00:08,r1,0.2\n"
        "2024-01-01 00:00:13,r1,0.3\n"
        "2024-01-01 00:00:23,r1,0.4\n"
        "2024-01-01 00:00:28,r1,0.5\n"
    )

    raw, wide, _ = process_data.process_od_pioreactor(
        uploaded,
        round_time=5,
        aggregate_high_frequency_raw_data=aggregate_high_frequency,
    )

    start = pd.Timestamp("2024-01-01 00:00:00")
    if aggregate_high_frequency:
        local_seconds = [0, 15]
        rounded_seconds = [0, 15]
        elapsed_seconds = [0, 15]
        expected_od = [0.2, 0.45]
    else:
        local_seconds = [3, 8, 13, 23, 28]
        rounded_seconds = [5, 10, 15, 25, 30]
        elapsed_seconds = [0, 5, 10, 20, 25]
        expected_od = [0.1, 0.2, 0.3, 0.4, 0.5]
    expected_local = start + pd.to_timedelta(local_seconds, unit="s")
    expected_rounded = start + pd.to_timedelta(rounded_seconds, unit="s")

    assert raw["timestamp_localtime"].tolist() == expected_local.tolist()
    assert raw["timestamp_rounded"].tolist() == expected_rounded.tolist()
    assert raw["elapsed_time_in_seconds"].tolist() == elapsed_seconds
    assert raw["od_reading"].tolist() == pytest.approx(expected_od)
    assert wide.index.tolist() == expected_rounded.tolist()
    assert wide["r1"].tolist() == pytest.approx(expected_od)
    assert process_data.st.session_state["start_time"] == expected_rounded[0]


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


# %%
