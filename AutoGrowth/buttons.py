import streamlit as st


def convert_data(df):
    return df.to_csv(index=True).encode("utf-8")


@st.fragment
def create_download_button(
    label: str, data: str, file_name: str, disabled: bool, mime: str
):
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        disabled=disabled,
        type="primary",
        icon=":material/download:",
        width="stretch",
    )


def download_data_button_in_sidebar(
    session_key: str,
    label: str = "Download data",
    file_name: str = "filtered_data.csv",
):
    """Create a download button associated with a key in session state
    in the sidebar.

    - nested keys not possible
    - session state must be a DataFrame (which we do not check yet)
    """
    if st.session_state.get(session_key) is not None:
        disabled = False
        data = convert_data(st.session_state[session_key])
    else:
        disabled = True
        data = ""
    with st.sidebar:
        create_download_button(
            label=label,
            data=data,
            file_name=file_name,
            disabled=disabled,
            mime="text/csv",
        )
