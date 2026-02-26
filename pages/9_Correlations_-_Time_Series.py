"""Correlation Time Series Page - Using Precomputed Data"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS
from precomputed_loader import (
    load_rolling_correlations,
    load_current_weights,
    filter_by_period,
    check_precomputed_exists,
    ARK_PRECOMPUTED_DIR
)
from data_loader import load_etf_prices
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector

st.set_page_config(
    page_title="Correlation Time Series",
    page_icon="📈",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Correlation Time Series

Track how portfolio correlations evolve over time and analyze their relationship with ETF performance.
"""

period_key = get_current_period()
st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data.")
    st.stop()

# Controls
st.subheader("Correlation Time Series")

cols = st.columns([1, 3])

with cols[0]:
    controls_card = st.container(border=True)
    with controls_card:
        st.markdown("##### Select ETF")
        selected_etf = st.pills(
            "ETF",
            options=ARK_ETFS,
            default=ARK_ETFS[0],
            label_visibility="collapsed",
            key="ts_etf_selector"
        )

        ""  # Space

        st.markdown("##### Lookback Period")
        lookback_options = {
            "60 Days": 60,
            "120 Days": 120,
            "250 Days": 250
        }
        selected_lookback = st.pills(
            "Lookback",
            options=list(lookback_options.keys()),
            default="120 Days",
            label_visibility="collapsed",
            key="ts_lookback_selector"
        )
        if selected_lookback is None:
            selected_lookback = "120 Days"
        lookback_days = lookback_options[selected_lookback]

        ""  # Space

        st.markdown("##### Rolling Window")
        rolling_options = {
            "20 Days": 20,
            "30 Days": 30,
            "60 Days": 60,
            "120 Days": 120
        }
        selected_rolling = st.pills(
            "Rolling Window",
            options=list(rolling_options.keys()),
            default="60 Days",
            label_visibility="collapsed",
            key="rolling_window_selector"
        )
        if selected_rolling is None:
            selected_rolling = "60 Days"
        rolling_window = rolling_options[selected_rolling]

        ""  # Space

        st.markdown("##### Correlation Type")
        use_weighted = st.toggle("Weighted Correlation", value=True, key="ts_weighted_toggle")

# Load data
with st.spinner("Loading correlation data..."):
    rolling_corr = load_rolling_correlations(selected_etf, period_key, rolling_window)
    current_weights = load_current_weights(selected_etf, period_key)
    etf_prices = load_etf_prices(selected_etf)

# Filter data by lookback period (last N trading days)
if len(rolling_corr) > 0:
    rolling_corr_filtered = filter_by_period(rolling_corr, start_date, end_date)
    # Apply lookback filter - take last N rows
    if len(rolling_corr_filtered) > lookback_days:
        rolling_corr_filtered = rolling_corr_filtered.tail(lookback_days)
else:
    rolling_corr_filtered = pd.DataFrame()

# Choose which correlation column to use based on toggle
corr_col = 'weighted_mean_corr' if use_weighted else 'mean_corr'
corr_label = 'Weighted' if use_weighted else 'Unweighted'

# Summary Statistics
with cols[0]:
    ""  # Space

    stats_card = st.container(border=True)
    with stats_card:
        st.markdown("##### Summary Statistics")

        if len(rolling_corr_filtered) > 0:
            st.markdown(f"**Data Points:** {len(rolling_corr_filtered)}")

            ""  # Space

            # Current correlation (latest value)
            latest_corr = rolling_corr_filtered.iloc[-1]
            st.markdown("**Latest Correlation**")
            st.markdown(f"{corr_label}: **{latest_corr[corr_col]:.3f}**")

            ""  # Space

            # Period statistics
            st.markdown("**Period Statistics**")
            st.markdown(f"Mean: **{rolling_corr_filtered[corr_col].mean():.3f}**")
            st.markdown(f"Min: **{rolling_corr_filtered[corr_col].min():.3f}**")
            st.markdown(f"Max: **{rolling_corr_filtered[corr_col].max():.3f}**")
        else:
            st.warning("No data for selected period")

# Main chart area
with cols[1]:
    ts_card = st.container(border=True)
    with ts_card:
        if len(rolling_corr_filtered) > 0:
            fig_ts = go.Figure()

            # Main correlation line based on toggle
            fig_ts.add_trace(go.Scatter(
                x=rolling_corr_filtered['Date'],
                y=rolling_corr_filtered[corr_col],
                mode='lines',
                name=f'{corr_label} Mean',
                line=dict(color='steelblue', width=2),
                hovertemplate=f'<b>{corr_label} Mean</b><br>Date: %{{x|%Y-%m-%d}}<br>Correlation: %{{y:.4f}}<extra></extra>'
            ))

            fig_ts.update_layout(
                title=f"{selected_etf} Rolling {rolling_window}-Day {corr_label} Correlation (Last {lookback_days} Days)",
                xaxis_title="Date",
                yaxis_title="Correlation",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )

            st.plotly_chart(fig_ts, width='stretch')
        else:
            st.warning(f"No precomputed rolling correlations available for {selected_etf} with {rolling_window}-day window. Run `python convert_to_parquet.py` to generate.")

""  # Space

# Correlation vs Performance Analysis
st.subheader("Correlation vs Performance Analysis")

perf_card = st.container(border=True)
with perf_card:
    if len(etf_prices) > 0 and len(rolling_corr_filtered) > 0:
        # Calculate ETF returns
        etf_prices_copy = etf_prices.copy()
        etf_prices_copy['Return'] = etf_prices_copy['Close'].pct_change(fill_method=None)

        # Merge correlation data with ETF returns
        corr_perf = pd.merge(
            rolling_corr_filtered,
            etf_prices_copy[['Date', 'Close', 'Return']],
            on='Date',
            how='inner'
        )

        if len(corr_perf) > 0:
            # Create dual-axis chart: Correlation vs Cumulative Return
            fig_perf = make_subplots(specs=[[{"secondary_y": True}]])

            # Cumulative return
            corr_perf['Cum_Return'] = (1 + corr_perf['Return']).cumprod() - 1

            fig_perf.add_trace(
                go.Scatter(
                    x=corr_perf['Date'],
                    y=corr_perf['Cum_Return'] * 100,
                    mode='lines',
                    name=f'{selected_etf} Cumulative Return',
                    line=dict(color='black', width=2),
                    hovertemplate='<b>Cumulative Return</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                secondary_y=False
            )

            # Correlation line based on toggle
            fig_perf.add_trace(
                go.Scatter(
                    x=corr_perf['Date'],
                    y=corr_perf[corr_col],
                    mode='lines',
                    name=f'{corr_label} Correlation',
                    line=dict(color='steelblue', width=2, dash='dot'),
                    hovertemplate=f'<b>{corr_label} Correlation</b><br>Date: %{{x|%Y-%m-%d}}<br>Correlation: %{{y:.4f}}<extra></extra>'
                ),
                secondary_y=True
            )

            fig_perf.update_layout(
                title=f"{selected_etf} Cumulative Return vs Rolling {rolling_window}-Day {corr_label} Correlation",
                height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified'
            )

            fig_perf.update_xaxes(title_text="Date", gridcolor='lightgray')
            fig_perf.update_yaxes(title_text="Cumulative Return (%)", secondary_y=False, gridcolor='lightgray')
            fig_perf.update_yaxes(title_text=f"{corr_label} Correlation", secondary_y=True)

            st.plotly_chart(fig_perf, width='stretch')

            ""  # Space

            # Correlation regime analysis
            st.markdown("#### Correlation Regime Analysis")

            # Split into high/low correlation regimes
            median_corr = corr_perf[corr_col].median()

            high_corr = corr_perf[corr_perf[corr_col] >= median_corr]
            low_corr = corr_perf[corr_perf[corr_col] < median_corr]

            # Calculate statistics for each regime
            regime_cols = st.columns(2)

            with regime_cols[0]:
                st.markdown(f"**High Correlation Regime** (>= {median_corr:.3f})")
                if len(high_corr) > 0:
                    # Compound annualized return
                    high_total_return = (1 + high_corr['Return']).prod() - 1
                    high_n_days = len(high_corr)
                    high_avg_return = ((1 + high_total_return) ** (252 / high_n_days) - 1) * 100
                    high_volatility = high_corr['Return'].std() * np.sqrt(252) * 100
                    high_sharpe = high_avg_return / high_volatility if high_volatility > 0 else 0
                    st.metric("Annualized Return", f"{high_avg_return:.1f}%")
                    st.metric("Annualized Volatility", f"{high_volatility:.1f}%")
                    st.metric("Sharpe Ratio", f"{high_sharpe:.2f}")
                    st.caption(f"Days: {len(high_corr)}")

            with regime_cols[1]:
                st.markdown(f"**Low Correlation Regime** (< {median_corr:.3f})")
                if len(low_corr) > 0:
                    # Compound annualized return
                    low_total_return = (1 + low_corr['Return']).prod() - 1
                    low_n_days = len(low_corr)
                    low_avg_return = ((1 + low_total_return) ** (252 / low_n_days) - 1) * 100
                    low_volatility = low_corr['Return'].std() * np.sqrt(252) * 100
                    low_sharpe = low_avg_return / low_volatility if low_volatility > 0 else 0
                    st.metric("Annualized Return", f"{low_avg_return:.1f}%")
                    st.metric("Annualized Volatility", f"{low_volatility:.1f}%")
                    st.metric("Sharpe Ratio", f"{low_sharpe:.2f}")
                    st.caption(f"Days: {len(low_corr)}")
        else:
            st.warning("Could not merge correlation and price data.")
    else:
        if len(etf_prices) == 0:
            st.warning(f"No price data available for {selected_etf}")
        else:
            st.warning(f"No rolling correlation data available for {selected_etf}")

""  # Space

# Data Download
st.subheader("Data Download")

download_card = st.container(border=True)
with download_card:
    if len(rolling_corr_filtered) > 0:
        st.markdown(f"**Rolling Correlation Data** ({len(rolling_corr_filtered)} records)")

        # Display table
        display_df = rolling_corr_filtered.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df = display_df.rename(columns={
            'mean_corr': 'Unweighted Mean',
            'median_corr': 'Median',
            'weighted_mean_corr': 'Weighted Mean'
        })

        st.dataframe(
            display_df.round(4),
            width='stretch',
            height=300
        )

        ""  # Space

        # Download button
        csv_data = rolling_corr_filtered.to_csv(index=False)
        st.download_button(
            label="Download Rolling Correlations (CSV)",
            data=csv_data,
            file_name=f"{selected_etf}_rolling_correlations_{rolling_window}d.csv",
            mime="text/csv"
        )
    else:
        st.warning("No rolling correlation data available.")
