"""Tests for the main app.

As of now fileuploading cannot be tested.

As long as local module (w.r.t to the app are used, the test script has to be in the
same folder as the app).

https://docs.streamlit.io/develop/concepts/app-testing/cheat-sheet
"""

from streamlit.testing.v1 import AppTest

import piogrowth

fname_data = "data/batch_example/example_batch_data_od_readings.csv"

df_raw_od_data = piogrowth.load.read_csv(fname_data)
msg = (
    f"- Loaded {df_raw_od_data.shape[0]:,d} rows "
    f"and {df_raw_od_data.shape[1]:,d} columns.\n"
)
# round timestamp data
df_raw_od_data.insert(
    0,
    "timestamp_rounded",
    df_raw_od_data["timestamp_localtime"].dt.round(
        f"{5}s",
    ),
)


def test_upload_page():
    fname_app = "app/0_upload_data.py"
    at = AppTest.from_file(fname_app).run()
    assert not at.exception

    assert at.markdown[-1].value == "### Store in QurvE format"
    # cannot test file upload as of now
    at.session_state["df_raw_od_data"] = df_raw_od_data
    at.run()
    assert not at.exception

    # images and download buttons cannot be tested as of now
    # ? Check for subheaders?
