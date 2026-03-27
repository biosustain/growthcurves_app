import streamlit as st


# styles the selected navbar item with green background
def green_navbar():
    return st.markdown(
        """
        <style>
        /* Top navigation - selected/active page */
        button[data-testid="stPageLink-NavLink"][aria-current="page"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #66BB6A !important;
        }

        /* Top navigation - hover effect */
        button[data-testid="stPageLink-NavLink"]:hover {
            background-color: rgba(76, 175, 80, 0.12) !important;
        }

        /* Navigation container styling */
        [data-testid="stSidebarNav"],
        [data-testid="stNavigation"] {
            background-color: transparent !important;
        }

        /* Alternative selectors for active navigation */
        ul[role="tablist"] button[aria-selected="true"],
        div[role="tab"][aria-selected="true"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #66BB6A !important;
        }

        /* Navigation link active state */
        a[aria-current="page"],
        a[data-active="true"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# adds a green gradient background at the top of the page
def green_gradient():
    return st.markdown(
        """
        <style>
        /* Green gradient at the top */
        .stMainBlockContainer {
            background: linear-gradient(to bottom, #66BB6A 0%, #FFFFFF 200px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# hacky solution to make all the tertiary buttons red
def red_buttons():

    return st.markdown(
        """
        <style>
        /* Destructive tertiary buttons */
        button[kind="tertiary"] {
            background-color: #ef5350 !important;
            color: white !important;
            border: 1px solid #ef5350 !important;
            border-radius: 0.5rem;
            font-weight: 600;
        }

        button[kind="tertiary"]:hover {
            background-color: #d32f2f !important;
            border-color: #d32f2f !important;
            color: white !important;
        }

        button[kind="tertiary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(239, 83, 80, 0.3);
            outline: none;
        }

        button[kind="tertiary"]:active {
            background-color: #c62828 !important;
            border-color: #c62828 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# styles plate status grid tables
def plate_table_style():
    return st.markdown(
        """
        <style>
        .plate-wrap { width: 100%; overflow: hidden; }
        table.plate {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            font-size: 18px;
        }
        table.plate th, table.plate td {
            border: 1px solid rgba(49,51,63,0.2);
            text-align: center;
            padding: 6px 0;
            line-height: 1.2;
        }
        table.plate th { font-weight: 600; }
        table.plate th.row { width: 2.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# styles growth parameter calculations table
def growth_param_table_style():
    return st.markdown(
        """
        <style>
        .growth-param-table table {
            width: 100%;
        }
        .growth-param-table th, .growth-param-table td {
            text-align: center !important;
            vertical-align: middle !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# styles data frames and AG Grid tables with larger font sizes
def data_grid_style():
    return st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] table {
            font-size: 16px !important;
        }
        div[data-testid="stDataFrame"] thead th {
            font-size: 17px !important;
            font-weight: bold !important;
        }
        .ag-theme-streamlit .ag-cell {
            font-size: 16px !important;
        }
        .ag-theme-streamlit .ag-header-cell-text {
            font-size: 17px !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
