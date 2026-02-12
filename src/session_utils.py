"""Session state utilities for analysis period management"""
import streamlit as st
from config import ANALYSIS_PERIODS, DEFAULT_PERIOD


def init_session_state():
    """Initialize session state with default values.
    Call this at the top of every page."""
    if 'analysis_period' not in st.session_state:
        st.session_state.analysis_period = DEFAULT_PERIOD


def get_current_period():
    """Get the currently selected period key"""
    init_session_state()
    return st.session_state.analysis_period


def get_current_dates():
    """Get start_date and end_date for current period"""
    period_key = get_current_period()
    period = ANALYSIS_PERIODS[period_key]
    return period["start"], period["end"]


def has_r3000_data():
    """Check if current period has R3000 data available"""
    period_key = get_current_period()
    return ANALYSIS_PERIODS[period_key]["has_r3000_data"]


def render_period_selector():
    """Render the period selector in sidebar.
    Call this from every page to allow period switching from anywhere."""
    init_session_state()

    st.markdown("##### Analysis Period")

    period_options = list(ANALYSIS_PERIODS.keys())

    selected_period = st.pills(
        "Period",
        options=period_options,
        default=st.session_state.analysis_period,
        format_func=lambda x: ANALYSIS_PERIODS[x]["label"],
        label_visibility="collapsed"
    )

    # Update session state if changed
    if selected_period != st.session_state.analysis_period:
        st.session_state.analysis_period = selected_period
        st.rerun()

    return selected_period
