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

from config import ARK_ETFS, INPUT_DIR, OUTPUT_DIR
from data_loader import load_ark_holdings, load_r3000_holdings, load_industry_info, get_r3000_drawdowns_cache, save_r3000_drawdowns_cache, get_ark_files_hash, get_r3000_files_hash
from drawdown_calculator import calculate_drawdowns_with_filter, calculate_drawdowns
from session_utils import init_session_state, get_current_dates, has_r3000_data, get_current_period, render_period_selector

st.set_page_config(
    page_title="Drawdown Distribution",
    page_icon="📊",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Drawdown Distribution

Compare drawdown distributions between ARK ETF holdings and Russell 3000 constituents.
"""

@st.cache_data
def get_r3000_peer_groups(_files_hash):
    """Get list of GICS Industry Groups from R3000"""
    try:
        industry_dict = load_industry_info(source='r3000')
        # Filter out invalid entries (Bloomberg error messages)
        peer_groups = sorted([
            pg for pg in set(industry_dict.values())
            if pg and not pg.startswith('#N/A') and 'Unable to retrieve' not in pg
        ])
        return peer_groups
    except:
        return []

@st.cache_data
def calculate_ark_holdings_drawdowns(_files_hash, etf, min_depth_pct, min_duration_days, period_key, _start_date, _end_date, _holdings):
    """Calculate max drawdown for each current holding in ARK ETF

    _holdings: Pre-loaded holdings data (underscore prefix excludes from hashing)
    period_key, _start_date, _end_date: Analysis period for filtering
    """
    holdings = _holdings

    # Filter holdings to analysis period first
    holdings = holdings[
        (holdings['Date'] >= _start_date) &
        (holdings['Date'] <= _end_date)
    ].copy()

    if len(holdings) == 0:
        return pd.DataFrame()

    # Get current holdings (latest date within analysis period)
    latest_date = holdings['Date'].max()
    current_tickers = holdings[holdings['Date'] == latest_date]['Ticker'].unique()

    # Filter out currency and money market funds
    if 'Bloomberg Name' in holdings.columns:
        currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
        current_tickers = [t for t in current_tickers if t not in currency_tickers]

    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
    current_tickers = [t for t in current_tickers if not any(t.split()[0].startswith(p) for p in money_market_prefixes)]

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
def calculate_r3000_drawdowns_full(_files_hash, min_depth_pct, min_duration_days, period_key, _start_date, _end_date, _holdings):
    """Calculate max drawdown for ALL stocks in R3000 (cached to file)

    _holdings: Pre-loaded holdings data (underscore prefix excludes from hashing)
    period_key, _start_date, _end_date: Analysis period for filtering
    """
    # Note: File cache disabled when using period filtering (different periods need different results)
    # Check for precomputed cache only for default period
    # cached = get_r3000_drawdowns_cache()
    # if cached is not None:
    #     return cached

    holdings = _holdings.copy()

    # Filter holdings to analysis period
    holdings = holdings[
        (holdings['Date'] >= _start_date) &
        (holdings['Date'] <= _end_date)
    ].copy()

    if len(holdings) == 0:
        return pd.DataFrame()

    # Get unique tickers
    all_tickers = holdings['Ticker'].unique()

    # Get industry mapping for all tickers
    industry_dict = load_industry_info(source='r3000')

    results = []
    total = len(all_tickers)
    for i, ticker in enumerate(all_tickers):
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
            ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
            # Get industry group for this ticker
            gics = industry_dict.get(ticker, industry_dict.get(ticker_clean, 'Unknown'))
            results.append({
                'ticker': ticker_clean,
                'max_drawdown': max_dd,
                'num_drawdowns': len(dd_df),
                'gics_industry_group': gics
            })

    result_df = pd.DataFrame(results)

    # Save to cache file for future use
    if len(result_df) > 0:
        save_r3000_drawdowns_cache(result_df)

    return result_df


def get_r3000_drawdowns_filtered(full_drawdowns, peer_group):
    """Filter precomputed R3000 drawdowns by peer group"""
    if peer_group and peer_group != "Russell 3000 (All)":
        return full_drawdowns[full_drawdowns['gics_industry_group'] == peer_group].copy()
    return full_drawdowns.copy()

@st.cache_data
def calculate_etf_drawdown(_files_hash, etf, period_key, _start_date, _end_date):
    """Calculate the ETF's own drawdown (not holdings)

    period_key, _start_date, _end_date: Analysis period for filtering
    """
    price_file = OUTPUT_DIR / f'{etf}_prices.csv'
    if not price_file.exists():
        return None

    prices = pd.read_csv(price_file)
    prices['Date'] = pd.to_datetime(prices['Date'])

    # Filter prices to analysis period
    prices = prices[
        (prices['Date'] >= _start_date) &
        (prices['Date'] <= _end_date)
    ].copy()

    if len(prices) == 0:
        return None

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
period_key = get_current_period()

with st.spinner("Calculating drawdown distributions..."):
    # Load holdings once (cached)
    ark_holdings = load_ark_holdings(ark_hash, selected_etf)
    r3000_holdings = load_r3000_holdings(r3000_hash)

    # ARK holdings drawdowns
    ark_dd = calculate_ark_holdings_drawdowns(
        ark_hash, selected_etf, min_depth_pct=10, min_duration_days=7,
        period_key=period_key, _start_date=start_date, _end_date=end_date, _holdings=ark_holdings
    )

    # R3000 drawdowns (compute with period filtering)
    r3000_dd_full = calculate_r3000_drawdowns_full(
        r3000_hash, min_depth_pct=10, min_duration_days=7,
        period_key=period_key, _start_date=start_date, _end_date=end_date, _holdings=r3000_holdings
    )

    # Filter by peer group
    peer_group = None if selected_benchmark == "Russell 3000 (All)" else selected_benchmark
    r3000_dd = get_r3000_drawdowns_filtered(r3000_dd_full, peer_group)

    # ETF's own drawdown
    etf_drawdown = calculate_etf_drawdown(ark_hash, selected_etf, period_key, start_date, end_date)

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

    # Right panel: Distribution charts
    with cols[1]:
        chart_card = st.container(border=True)
        with chart_card:
            # Create subplots: percentage on top, count on bottom
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Percentage Distribution", "Count Distribution"),
                vertical_spacing=0.15
            )

            # === Top chart: Percentage distribution ===
            # R3000 percentage distribution
            fig.add_trace(go.Histogram(
                x=r3000_dd['max_drawdown'],
                name=selected_benchmark,
                marker_color='steelblue',
                opacity=0.6,
                nbinsx=40,
                histnorm='percent',
                hovertemplate=(
                    f"<b>{selected_benchmark}</b><br>"
                    "Drawdown Range: %{x}%<br>"
                    "Percentage: %{y:.1f}%<extra></extra>"
                ),
                legendgroup='benchmark',
                showlegend=True
            ), row=1, col=1)

            # ARK percentage distribution
            fig.add_trace(go.Histogram(
                x=ark_dd['max_drawdown'],
                name=f'{selected_etf} Holdings',
                marker_color='crimson',
                opacity=0.6,
                nbinsx=40,
                histnorm='percent',
                hovertemplate=(
                    f"<b>{selected_etf} Holdings</b><br>"
                    "Drawdown Range: %{x}%<br>"
                    "Percentage: %{y:.1f}%<extra></extra>"
                ),
                legendgroup='ark',
                showlegend=True
            ), row=1, col=1)

            # === Bottom chart: Count distribution ===
            # R3000 count distribution
            fig.add_trace(go.Histogram(
                x=r3000_dd['max_drawdown'],
                name=selected_benchmark,
                marker_color='steelblue',
                opacity=0.6,
                nbinsx=40,
                hovertemplate=(
                    f"<b>{selected_benchmark}</b><br>"
                    "Drawdown Range: %{x}%<br>"
                    "Stock Count: %{y}<extra></extra>"
                ),
                legendgroup='benchmark',
                showlegend=False
            ), row=2, col=1)

            # ARK count distribution
            fig.add_trace(go.Histogram(
                x=ark_dd['max_drawdown'],
                name=f'{selected_etf} Holdings',
                marker_color='crimson',
                opacity=0.6,
                nbinsx=40,
                hovertemplate=(
                    f"<b>{selected_etf} Holdings</b><br>"
                    "Drawdown Range: %{x}%<br>"
                    "Stock Count: %{y}<extra></extra>"
                ),
                legendgroup='ark',
                showlegend=False
            ), row=2, col=1)

            # Add dummy traces for reference lines legend (actual line graphics)
            # Use legend2 to place them separately on the right
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name=f'{selected_benchmark} Mean: {r3000_stats["mean"]:.1f}%',
                line=dict(color='steelblue', width=2, dash='dot'),
                showlegend=True,
                legend='legend2'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name=f'{selected_etf} Holdings Mean: {ark_stats["mean"]:.1f}%',
                line=dict(color='crimson', width=2, dash='dot'),
                showlegend=True,
                legend='legend2'
            ), row=1, col=1)

            if etf_drawdown is not None:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    name=f'{selected_etf} MDD: {etf_drawdown:.1f}%',
                    line=dict(color='darkred', width=2, dash='dash'),
                    showlegend=True,
                    legend='legend2'
                ), row=1, col=1)

            # Add vertical lines for both subplots
            for row in [1, 2]:
                # ETF drawdown line
                if etf_drawdown is not None:
                    fig.add_vline(
                        x=etf_drawdown,
                        line_dash="dash",
                        line_color="darkred",
                        line_width=2,
                        row=row, col=1
                    )

                # Mean lines
                fig.add_vline(
                    x=r3000_stats['mean'],
                    line_dash="dot",
                    line_color="steelblue",
                    line_width=2,
                    row=row, col=1
                )

                fig.add_vline(
                    x=ark_stats['mean'],
                    line_dash="dot",
                    line_color="crimson",
                    line_width=2,
                    row=row, col=1
                )

            fig.update_layout(
                title=f"Max Drawdown Distribution: {selected_etf} Holdings vs {selected_benchmark}",
                barmode='overlay',
                height=700,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                legend2=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            # Update axes
            fig.update_xaxes(title_text="Max Drawdown (%)", gridcolor='lightgray', row=2, col=1)
            fig.update_yaxes(title_text="% of Stocks", gridcolor='lightgray', row=1, col=1)
            fig.update_yaxes(title_text="Number of Stocks", gridcolor='lightgray', row=2, col=1)

            st.plotly_chart(fig, width='stretch')

    ""  # Space

    # Summary Statistics section
    st.subheader("Summary Statistics")

    stats_cols = st.columns(2)

    with stats_cols[0]:
        stats_card = st.container(border=True)
        with stats_card:
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
            st.dataframe(stats_df, hide_index=True, width='stretch')

    with stats_cols[1]:
        if etf_drawdown is not None and etf_percentile is not None:
            etf_card = st.container(border=True)
            with etf_card:
                st.markdown(f"**{selected_etf} MDD:** {etf_drawdown:.1f}%")
                st.markdown(f"**Percentile in {selected_benchmark}:** {etf_percentile:.0f}th")
                st.markdown(f"<small>(Deeper than {100-etf_percentile:.0f}% of stocks)</small>", unsafe_allow_html=True)

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
        st.dataframe(ark_display, hide_index=True, width='stretch', height=400)

    with table_cols[1]:
        st.markdown(f"##### {selected_benchmark} (Top 50 Worst)")
        r3000_display = r3000_dd.nsmallest(50, 'max_drawdown')[['ticker', 'max_drawdown', 'num_drawdowns']].copy()
        r3000_display['max_drawdown'] = r3000_display['max_drawdown'].apply(lambda x: f"{x:.1f}%")
        r3000_display.columns = ['Ticker', 'Max Drawdown', '# Valid DDs']
        st.dataframe(r3000_display, hide_index=True, width='stretch', height=400)

else:
    st.warning("Not enough data to calculate distributions. Please check your data files.")
