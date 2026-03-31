import streamlit as st


def is_data_available(key):
    """Check that pioreactor data was uploaded."""
    return st.session_state.get(key) is not None


def page_header_with_help(title: str, help_text: str):
    """Render a standard title row with a right-aligned help popover."""
    title_col, popover_col = st.columns([9, 2])
    with title_col:
        st.title(title)
    with popover_col:
        st.write("")
        with st.popover("Help", width="stretch"):
            st.markdown(help_text)
    st.divider()


def show_warning_to_upload_data():
    """Show a warning message to upload data."""
    with st.container(border=True):
        col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
        with col1:
            st.warning("No data available for analysis. Please upload data first.")
        with col2:
            st.page_link(
                "0_upload_data.py",
                icon=":material/upload:",
                label="Upload Data",
                help="Go to upload data page.",
            )


def render_markdown(fpath: str):
    """Open and write markdown content from file."""
    with open(fpath, "r", encoding="utf-8") as f:
        about_content = f.read()
    st.write(about_content)
