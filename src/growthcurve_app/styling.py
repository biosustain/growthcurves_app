"""Align this styling of the app with the other app mentioned in the growthcurves
publication.

https://github.com/sambra95/TheGrowthAnalysisApp/blob/main/src/styling.py
"""

import streamlit as st


# styles the top navbar: items spread evenly, green active tab
def green_navbar():
    return st.markdown(
        """
        <style>
        /* The nav is an rc-overflow list; .rc-overflow-item is the flex child */
        .rc-overflow-item {
            flex: 1 !important;
        }

        [data-testid="stTopNavLink"] {
            justify-content: center !important;
        }

        /* Tab labels, 40% larger */
        [data-testid="stTopNavLink"] span {
            font-size: 1.4em !important;
        }

        [data-testid="stTopNavLink"][aria-current="page"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #66BB6A !important;
        }

        [data-testid="stTopNavLink"]:hover {
            background-color: rgba(76, 175, 80, 0.12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# light tint on bordered section containers (st.container(key="card_..."))
def card_containers():
    return st.markdown(
        """
        <style>
        [class*="st-key-card_"] {
            background-color: #F8F9FB;
        }
        /* Inputs inside a tinted container stay white so they stand out */
        [class*="st-key-card_"] [data-baseweb="select"] > div,
        [class*="st-key-card_"] [data-baseweb="input"],
        [class*="st-key-card_"] [data-baseweb="base-input"],
        [class*="st-key-card_"] [data-baseweb="textarea"],
        [class*="st-key-card_"] [data-testid="stNumberInputContainer"],
        [class*="st-key-card_"] [data-testid="stNumberInputStepDown"],
        [class*="st-key-card_"] [data-testid="stNumberInputStepUp"],
        [class*="st-key-card_"] [data-testid="stFileUploaderDropzone"],
        [class*="st-key-card_"] input,
        [class*="st-key-card_"] [data-testid="stSelectbox"] div:has(> input) {
            background-color: #FFFFFF !important;
        }
        /* Reactor and curve prev/next buttons fill the height of the card beside them */
        [class*="st-key-analysis_prev"],
        [class*="st-key-analysis_next"],
        [class*="st-key-analysis_prev"] :has(button),
        [class*="st-key-analysis_next"] :has(button),
        [class*="st-key-analysis_prev"] button,
        [class*="st-key-analysis_next"] button {
            height: 100%;
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
