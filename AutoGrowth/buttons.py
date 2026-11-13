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

