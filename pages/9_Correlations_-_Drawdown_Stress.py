"""Correlations - Drawdown Stress Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS
from precomputed_loader import (
    load_stress_correlations,
    check_precomputed_exists
)
from data_loader import load_etf_prices, load_ark_holdings, get_ark_files_hash
from session_utils import init_session_state, get_current_dates, render_period_selector


def calculate_recovery_correlation(files_hash, etf: str, stress_corr_df: pd.DataFrame) -> float:
    """Calculate mean correlation during recovery periods.

    Recovery period = from trough of one drawdown to peak of the next drawdown.
    All recovery periods are combined to calculate the overall average correlation.
    """
    if len(stress_corr_df) < 2:
        return 0.0

    # Load full holdings data
    holdings = load_ark_holdings(files_hash, etf)
    if len(holdings) == 0:
        return 0.0

    # Sort drawdowns by trough_date to get chronological order
    sorted_dd = stress_corr_df.sort_values('trough_date').reset_index(drop=True)

    recovery_corrs = []

    for i in range(len(sorted_dd) - 1):
        # Recovery period: from trough of current drawdown to peak of next drawdown
        recovery_start = sorted_dd.iloc[i]['trough_date']
        recovery_end = sorted_dd.iloc[i + 1]['peak_date']

        # Skip if recovery period is invalid (end before start)
        if recovery_end <= recovery_start:
            continue

        # Filter holdings to recovery period
        period_holdings = holdings[
            (holdings['Date'] >= recovery_start) &
            (holdings['Date'] <= recovery_end)
        ].copy()

        if len(period_holdings) < 10:
            continue

        # Get tickers present in this period
        tickers = list(period_holdings['Ticker'].unique())

        # Filter out currency and money market tickers
        if 'Bloomberg Name' in period_holdings.columns:
            currency_tickers = period_holdings[
                period_holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)
            ]['Ticker'].unique()
            tickers = [t for t in tickers if t not in currency_tickers]

        money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
        tickers = [t for t in tickers if not any(t.split()[0].startswith(p) for p in money_market_prefixes)]

        if len(tickers) < 2:
            continue

        # Pivot to get price matrix
        price_matrix = period_holdings[period_holdings['Ticker'].isin(tickers)].pivot_table(
            index='Date',
            columns='Ticker',
            values='Stock_Price',
            aggfunc='first'
        )

        # Drop tickers with too many missing values
        min_data_points = len(price_matrix) * 0.5
        price_matrix = price_matrix.dropna(axis=1, thresh=int(min_data_points))

        if len(price_matrix.columns) < 2:
            continue

        # Calculate daily returns
        returns = price_matrix.pct_change(fill_method=None).iloc[1:].dropna(axis=1)

        if len(returns) < 10 or len(returns.columns) < 2:
            continue

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Extract upper triangle (pairwise correlations)
        n = len(corr_matrix.columns)
        triu_i, triu_j = np.triu_indices(n, k=1)
        corr_values = corr_matrix.values[triu_i, triu_j]
        valid_corrs = corr_values[~np.isnan(corr_values)]

        if len(valid_corrs) > 0:
            recovery_corrs.extend(valid_corrs)

    if len(recovery_corrs) == 0:
        return 0.0

    return np.mean(recovery_corrs)


st.set_page_config(
    page_title="Correlations - Drawdown Stress",
    page_icon="📉",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Correlations - Drawdown Stress

Analyze how portfolio correlations change during drawdown periods vs recovery periods.
"""

st.markdown("**Analysis Period:** Full Historical Data (2021-2026)")
st.caption("Stress correlations are calculated across all top 10 drawdowns in the full historical period. This analysis ignores the period selector.")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")

# Get ETF from session state (default to first)
if 'stress_etf' not in st.session_state:
    st.session_state.stress_etf = ARK_ETFS[0]
selected_etf = st.session_state.stress_etf

# Load data
files_hash = get_ark_files_hash()
stress_corr = load_stress_correlations(selected_etf)

# Calculate recovery correlation
recovery_corr = 0.0
if len(stress_corr) >= 2:
    recovery_corr = calculate_recovery_correlation(files_hash, selected_etf, stress_corr)

if len(stress_corr) > 0:
    # Calculate statistics
    stress_mean = stress_corr['mean_corr'].mean()
    correlation_increase = ((stress_mean / recovery_corr) - 1) * 100 if recovery_corr > 0 else 0

    # === METRICS AT THE TOP (before two-column layout) ===
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Recovery Correlation", f"{recovery_corr:.3f}",
                  help="Average pairwise correlation during recovery periods")

    with metric_cols[1]:
        st.metric("Stress Correlation", f"{stress_mean:.3f}",
                  help="Average correlation during drawdown periods")

    with metric_cols[2]:
        st.metric("Correlation Increase", f"{correlation_increase:+.1f}%",
                  help="How much correlations increase during stress")

    "" # Space

    # Chart title before two-column layout
    st.markdown("#### Correlation by Drawdown Event")

    # === TWO-COLUMN LAYOUT ===
    cols = st.columns([1, 3])

    # Left column: Select ETF + Summary Statistics
    with cols[0]:
        controls_card = st.container(border=True)
        with controls_card:
            st.markdown("##### Select ETF")
            new_etf = st.pills(
                "ETF",
                options=ARK_ETFS,
                default=selected_etf,
                label_visibility="collapsed",
                key="stress_etf_selector"
            )
            if new_etf != selected_etf:
                st.session_state.stress_etf = new_etf
                st.rerun()

        "" # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            st.markdown(f"**Drawdown Events:** {len(stress_corr)}")

            "" # Space

            max_stress = stress_corr['mean_corr'].max()
            st.markdown(f"**Max Stress Corr:** {max_stress:.3f}")

            "" # Space

            min_stress = stress_corr['mean_corr'].min()
            st.markdown(f"**Min Stress Corr:** {min_stress:.3f}")

    # Right column: Chart + Table
    with cols[1]:

        chart_card = st.container(border=True)
        with chart_card:
            # Load ETF prices
            etf_prices = load_etf_prices(selected_etf)

            if len(etf_prices) > 0:
                # Use full price history
                price_df = etf_prices.copy()

                # Create figure
                fig_stress = go.Figure()

                # Drawdown colors
                dd_colors = [
                    'rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                    'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                    'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                    'rgba(255, 228, 181, 0.3)'
                ]

                # Add drawdown shaded regions
                for idx, (_, row) in enumerate(stress_corr.iterrows()):
                    fig_stress.add_vrect(
                        x0=row['peak_date'],
                        x1=row['trough_date'],
                        fillcolor=dd_colors[idx % len(dd_colors)],
                        layer="below",
                        line_width=0
                    )

                # Add ETF price line
                fig_stress.add_trace(go.Scatter(
                    x=price_df['Date'],
                    y=price_df['Close'],
                    mode='lines',
                    name=f'{selected_etf} Price',
                    line=dict(color='black', width=2),
                    hoverinfo='skip'
                ))

                # Calculate midpoints and add correlation points
                midpoint_dates = []
                midpoint_corrs = []
                hover_texts = []

                for _, row in stress_corr.iterrows():
                    peak = row['peak_date']
                    trough = row['trough_date']
                    midpoint = peak + (trough - peak) / 2
                    midpoint_dates.append(midpoint)
                    midpoint_corrs.append(row['mean_corr'])
                    hover_texts.append(
                        f"<b>Drawdown #{int(row['dd_rank'])}</b><br>" +
                        f"Period: {peak.strftime('%Y-%m-%d')} to {trough.strftime('%Y-%m-%d')}<br>" +
                        f"Depth: {row['depth_pct']:.1f}%<br>" +
                        f"Duration: {row['duration_days']} days<br>" +
                        f"Mean Correlation: {row['mean_corr']:.3f}"
                    )

                # Sort by date for proper line connection
                sorted_data = sorted(zip(midpoint_dates, midpoint_corrs, hover_texts), key=lambda x: x[0])
                midpoint_dates = [x[0] for x in sorted_data]
                midpoint_corrs = [x[1] for x in sorted_data]
                hover_texts = [x[2] for x in sorted_data]

                # Add stress correlation line on secondary y-axis
                fig_stress.add_trace(go.Scatter(
                    x=midpoint_dates,
                    y=midpoint_corrs,
                    mode='lines+markers',
                    name='Stress Correlation',
                    line=dict(color='red', width=2),
                    marker=dict(size=12, color='red', symbol='circle'),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_texts,
                    yaxis='y2'
                ))

                # Add recovery correlation reference line as a trace (for legend)
                x_range = [price_df['Date'].min(), price_df['Date'].max()]
                fig_stress.add_trace(go.Scatter(
                    x=x_range,
                    y=[recovery_corr, recovery_corr],
                    mode='lines',
                    name=f'Recovery Correlation ({recovery_corr:.3f})',
                    line=dict(color='steelblue', width=2, dash='dash'),
                    yaxis='y2',
                    hoverinfo='skip'
                ))

                fig_stress.update_layout(
                    title=f"{selected_etf} Price & Stress Correlation by Drawdown",
                    height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    xaxis=dict(title="Date", gridcolor='lightgray'),
                    yaxis=dict(title=f"{selected_etf} Price ($)", gridcolor='lightgray'),
                    yaxis2=dict(title="Mean Correlation", overlaying='y', side='right', range=[0, 1], showgrid=False)
                )

                st.plotly_chart(fig_stress, width='stretch')

                st.markdown("<small>*Colored regions show top 10 drawdown periods. Red line = stress correlation at midpoint of each drawdown. Blue dashed line = recovery correlation baseline.*</small>", unsafe_allow_html=True)
            else:
                st.warning(f"No price data available for {selected_etf}")

    "" # Space

    # Detailed table (full width, outside two-column layout)
    st.markdown("#### Detailed Stress Correlation Data")

    table_card = st.container(border=True)
    with table_card:
        display_df = stress_corr.copy()
        display_df['peak_date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
        display_df['trough_date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
        display_df = display_df.rename(columns={
            'dd_rank': 'Rank',
            'peak_date': 'Peak',
            'trough_date': 'Trough',
            'depth_pct': 'Depth %',
            'duration_days': 'Days',
            'num_tickers': 'Tickers',
            'mean_corr': 'Mean ρ',
            'median_corr': 'Median ρ',
            'max_corr': 'Max ρ',
            'min_corr': 'Min ρ'
        })

        # Format numeric columns
        for col in ['Depth %', 'Mean ρ', 'Median ρ', 'Max ρ', 'Min ρ']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        st.dataframe(
            display_df[['Rank', 'Peak', 'Trough', 'Depth %', 'Days', 'Tickers', 'Mean ρ', 'Median ρ', 'Max ρ', 'Min ρ']],
            hide_index=True,
            width='stretch'
        )

    "" # Space

    # Insight
    if stress_mean > recovery_corr:
        st.warning(f"⚠️ **Correlations increase {correlation_increase:.0f}% during drawdowns.** Diversification benefits are reduced when the portfolio is under stress.")
    else:
        st.success(f"✓ **Correlations remain stable during drawdowns.** The portfolio maintains diversification benefits during stress periods.")

else:
    st.warning(f"No stress correlation data for {selected_etf}. Run `python convert_to_parquet.py` to generate.")
