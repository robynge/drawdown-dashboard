"""Drawdown Distribution Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE, INPUT_DIR, OUTPUT_DIR
from data_loader import load_ark_holdings, load_r3000_holdings, load_industry_info
from drawdown_calculator import calculate_drawdowns_with_filter, calculate_drawdowns

st.set_page_config(
    page_title="Drawdown Distribution",
    page_icon="📊",
    layout="wide"
)

"""
# Drawdown Distribution

Compare drawdown distributions between ARK ETF holdings and Russell 3000 constituents.
"""

# Helper functions for cache
def get_ark_files_hash():
    mtimes = []
    for etf in ARK_ETFS:
        holdings_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'
        if holdings_file.exists():
            mtimes.append(holdings_file.stat().st_mtime)
    return max(mtimes) if mtimes else 0

def get_r3000_files_hash():
    r3000_file = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.xlsx'
    if r3000_file.exists():
        return r3000_file.stat().st_mtime
    return 0

@st.cache_data
def get_cached_ark_holdings(_files_hash, etf):
    return load_ark_holdings(etf)

@st.cache_data
def get_cached_r3000_holdings(_files_hash):
    return load_r3000_holdings()

@st.cache_data
def get_r3000_peer_groups(_files_hash):
    """Get list of GICS Industry Groups from R3000"""
    try:
        industry_dict = load_industry_info(source='r3000')
        peer_groups = sorted(set(industry_dict.values()))
        return peer_groups
    except:
        return []

@st.cache_data
def calculate_ark_holdings_drawdowns(_files_hash, etf, min_depth_pct, min_duration_days, _holdings):
    """Calculate max drawdown for each current holding in ARK ETF

    _holdings: Pre-loaded holdings data (underscore prefix excludes from hashing)
    """
    holdings = _holdings

    # Get current holdings
    latest_date = holdings['Date'].max()
    current_tickers = holdings[holdings['Date'] == latest_date]['Ticker'].unique()

    # Filter out currency and money market funds
    if 'Bloomberg Name' in holdings.columns:
        currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
        current_tickers = [t for t in current_tickers if t not in currency_tickers]

    excluded_tickers = ['FTOXX', 'FIRXX']
    current_tickers = [t for t in current_tickers if t.split()[0] not in excluded_tickers]

    results = []
    for ticker in current_tickers:
        stock_data = holdings[holdings['Ticker'] == ticker].copy()

        if len(stock_data) < 10:
            continue

        # Prepare price data
        price_df = stock_data[['Date', 'Stock_Price']].copy()
        price_df = price_df.rename(columns={'Stock_Price': 'Close'})
        price_df = price_df.dropna(subset=['Close'])

        if len(price_df) < 10:
            continue

        # Calculate drawdowns with filter
        dd_df = calculate_drawdowns_with_filter(price_df, min_depth_pct, min_duration_days)

        if len(dd_df) > 0:
            # Get the maximum (deepest) drawdown
            max_dd = dd_df['depth_pct'].min()
            results.append({
                'ticker': ticker.split()[0],
                'max_drawdown': max_dd,
                'num_drawdowns': len(dd_df)
            })

    return pd.DataFrame(results)

@st.cache_data
def calculate_r3000_drawdowns(_files_hash, min_depth_pct, min_duration_days, peer_group, _holdings):
    """Calculate max drawdown for each stock in R3000 or a peer group

    _holdings: Pre-loaded holdings data (underscore prefix excludes from hashing)
    """
    holdings = _holdings.copy()

    # Filter to peer group if specified
    if peer_group and peer_group != "Russell 3000 (All)":
        industry_dict = load_industry_info(source='r3000')
        # Get tickers in this peer group
        peer_tickers = [ticker for ticker, gics in industry_dict.items() if gics == peer_group]
        # Filter holdings
        holdings = holdings[holdings['Ticker'].isin(peer_tickers)]

    if len(holdings) == 0:
        return pd.DataFrame()

    # Get unique tickers
    all_tickers = holdings['Ticker'].unique()

    results = []
    for ticker in all_tickers:
        stock_data = holdings[holdings['Ticker'] == ticker].copy()

        if len(stock_data) < 10:
            continue

        # R3000 uses 'Price' column
        if 'Price' not in stock_data.columns:
            continue

        price_df = stock_data[['Date', 'Price']].copy()
        price_df = price_df.rename(columns={'Price': 'Close'})
        price_df = price_df.dropna(subset=['Close'])

        if len(price_df) < 10:
            continue

        # Calculate drawdowns with filter
        dd_df = calculate_drawdowns_with_filter(price_df, min_depth_pct, min_duration_days)

        if len(dd_df) > 0:
            max_dd = dd_df['depth_pct'].min()
            results.append({
                'ticker': ticker.split()[0] if isinstance(ticker, str) else ticker,
                'max_drawdown': max_dd,
                'num_drawdowns': len(dd_df)
            })

    return pd.DataFrame(results)

@st.cache_data
def calculate_etf_drawdown(_files_hash, etf):
    """Calculate the ETF's own drawdown (not holdings)"""
    price_file = OUTPUT_DIR / f'{etf}_prices.csv'
    if not price_file.exists():
        return None

    prices = pd.read_csv(price_file)
    prices['Date'] = pd.to_datetime(prices['Date'])

    dd_df = calculate_drawdowns(prices)
    if len(dd_df) == 0:
        return None

    # Get max historical drawdown (excluding current)
    historical = dd_df[dd_df['rank'] != 'Current']
    if len(historical) > 0:
        max_dd = historical['depth_pct'].min()
    else:
        max_dd = dd_df['depth_pct'].min()

    return max_dd

# Main section
st.subheader("Distribution Analysis")

# Layout
cols = st.columns([1, 3])

with cols[0]:
    controls_card = st.container(border=True)
    with controls_card:
        st.markdown("##### Select ARK ETF")
        selected_etf = st.pills(
            "ETF",
            options=ARK_ETFS,
            default=ARK_ETFS[0],
            label_visibility="collapsed"
        )

        ""  # Space

        st.markdown("##### Select R3000 Benchmark")
        r3000_hash = get_r3000_files_hash()
        peer_groups = get_r3000_peer_groups(r3000_hash)
        benchmark_options = ["Russell 3000 (All)"] + peer_groups

        selected_benchmark = st.selectbox(
            "Benchmark",
            benchmark_options,
            label_visibility="collapsed"
        )

        ""  # Space

        st.markdown("##### Drawdown Criteria")
        st.markdown("<small>Valid if depth ≥ 10% OR duration ≥ 7 days</small>", unsafe_allow_html=True)

# Calculate distributions
ark_hash = get_ark_files_hash()
r3000_hash = get_r3000_files_hash()

with st.spinner("Calculating drawdown distributions..."):
    # Load holdings once (cached)
    ark_holdings = get_cached_ark_holdings(ark_hash, selected_etf)
    r3000_holdings = get_cached_r3000_holdings(r3000_hash)

    # ARK holdings drawdowns
    ark_dd = calculate_ark_holdings_drawdowns(ark_hash, selected_etf, min_depth_pct=10, min_duration_days=7, _holdings=ark_holdings)

    # R3000/Peer group drawdowns
    peer_group = None if selected_benchmark == "Russell 3000 (All)" else selected_benchmark
    r3000_dd = calculate_r3000_drawdowns(r3000_hash, min_depth_pct=10, min_duration_days=7, peer_group=peer_group, _holdings=r3000_holdings)

    # ETF's own drawdown
    etf_drawdown = calculate_etf_drawdown(ark_hash, selected_etf)

if len(ark_dd) > 0 and len(r3000_dd) > 0:
    # Calculate statistics
    ark_stats = {
        'count': len(ark_dd),
        'mean': ark_dd['max_drawdown'].mean(),
        'median': ark_dd['max_drawdown'].median(),
        'worst': ark_dd['max_drawdown'].min(),
        'best': ark_dd['max_drawdown'].max()
    }

    r3000_stats = {
        'count': len(r3000_dd),
        'mean': r3000_dd['max_drawdown'].mean(),
        'median': r3000_dd['max_drawdown'].median(),
        'worst': r3000_dd['max_drawdown'].min(),
        'best': r3000_dd['max_drawdown'].max()
    }

    # Calculate ETF percentile in R3000 distribution
    if etf_drawdown is not None:
        etf_percentile = (r3000_dd['max_drawdown'] >= etf_drawdown).mean() * 100
    else:
        etf_percentile = None

    # Display statistics in left panel
    with cols[0]:
        ""  # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            # Table format
            stats_df = pd.DataFrame({
                'Metric': ['Holdings Count', 'Mean Drawdown', 'Median Drawdown', 'Worst Drawdown', 'Best Drawdown'],
                f'{selected_etf}': [
                    f"{ark_stats['count']}",
                    f"{ark_stats['mean']:.1f}%",
                    f"{ark_stats['median']:.1f}%",
                    f"{ark_stats['worst']:.1f}%",
                    f"{ark_stats['best']:.1f}%"
                ],
                selected_benchmark: [
                    f"{r3000_stats['count']}",
                    f"{r3000_stats['mean']:.1f}%",
                    f"{r3000_stats['median']:.1f}%",
                    f"{r3000_stats['worst']:.1f}%",
                    f"{r3000_stats['best']:.1f}%"
                ]
            })

            st.dataframe(stats_df, hide_index=True, use_container_width=True)

            ""  # Space

            if etf_drawdown is not None and etf_percentile is not None:
                st.markdown("##### ETF Overall Drawdown")
                st.markdown(f"**{selected_etf} Max Drawdown:** {etf_drawdown:.1f}%")
                st.markdown(f"**Percentile in {selected_benchmark}:** {etf_percentile:.0f}th")
                st.markdown(f"<small>(Deeper than {100-etf_percentile:.0f}% of stocks)</small>", unsafe_allow_html=True)

    # Right panel: Distribution charts
    with cols[1]:
        chart_card = st.container(border=True)
        with chart_card:
            # Create histogram
            fig = go.Figure()

            # R3000 distribution
            fig.add_trace(go.Histogram(
                x=r3000_dd['max_drawdown'],
                name=selected_benchmark,
                marker_color='steelblue',
                opacity=0.6,
                nbinsx=40
            ))

            # ARK distribution
            fig.add_trace(go.Histogram(
                x=ark_dd['max_drawdown'],
                name=f'{selected_etf} Holdings',
                marker_color='crimson',
                opacity=0.6,
                nbinsx=40
            ))

            # Add vertical line for ETF drawdown
            if etf_drawdown is not None:
                fig.add_vline(
                    x=etf_drawdown,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"{selected_etf} ETF: {etf_drawdown:.1f}%",
                    annotation_position="top"
                )

            # Add vertical lines for means
            fig.add_vline(
                x=r3000_stats['mean'],
                line_dash="dot",
                line_color="steelblue",
                annotation_text=f"R3000 Mean: {r3000_stats['mean']:.1f}%",
                annotation_position="bottom left"
            )

            fig.add_vline(
                x=ark_stats['mean'],
                line_dash="dot",
                line_color="crimson",
                annotation_text=f"ARK Mean: {ark_stats['mean']:.1f}%",
                annotation_position="bottom right"
            )

            fig.update_layout(
                title=f"Max Drawdown Distribution: {selected_etf} Holdings vs {selected_benchmark}",
                xaxis_title="Max Drawdown (%)",
                yaxis_title="Count",
                barmode='overlay',
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )

            st.plotly_chart(fig, use_container_width=True)

    ""  # Space

    # Detailed tables
    st.subheader("Detailed Holdings Drawdowns")

    table_cols = st.columns(2)

    with table_cols[0]:
        st.markdown(f"##### {selected_etf} Holdings")
        ark_display = ark_dd.copy()
        ark_display['max_drawdown'] = ark_display['max_drawdown'].apply(lambda x: f"{x:.1f}%")
        ark_display = ark_display.sort_values('max_drawdown')
        ark_display.columns = ['Ticker', 'Max Drawdown', '# Valid DDs']
        st.dataframe(ark_display, hide_index=True, use_container_width=True, height=400)

    with table_cols[1]:
        st.markdown(f"##### {selected_benchmark} (Top 50 Worst)")
        r3000_display = r3000_dd.nsmallest(50, 'max_drawdown').copy()
        r3000_display['max_drawdown'] = r3000_display['max_drawdown'].apply(lambda x: f"{x:.1f}%")
        r3000_display.columns = ['Ticker', 'Max Drawdown', '# Valid DDs']
        st.dataframe(r3000_display, hide_index=True, use_container_width=True, height=400)

else:
    st.warning("Not enough data to calculate distributions. Please check your data files.")
