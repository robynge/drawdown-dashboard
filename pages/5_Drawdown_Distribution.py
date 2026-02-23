"""Drawdown Distribution Page - Using Precomputed Data"""
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

from config import ARK_ETFS, INPUT_DIR
from precomputed_loader import (
    load_r3000_drawdowns,
    load_r3000_drawdowns_with_dates,
    load_etf_drawdowns,
    load_ark_holdings_max_drawdowns,
    filter_by_period,
    filter_drawdowns_by_period,
    check_precomputed_exists
)
from data_loader import load_ark_holdings, load_industry_info, get_ark_files_hash, get_r3000_files_hash, get_industry_files_hash
from session_utils import init_session_state, get_current_dates, has_r3000_data, render_period_selector

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

st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")


def get_r3000_peer_groups():
    """Get list of GICS Industry Groups from R3000"""
    try:
        industry_dict = load_industry_info(get_industry_files_hash(), source='r3000')
        peer_groups = sorted([
            pg for pg in set(industry_dict.values())
            if pg and not pg.startswith('#N/A') and 'Unable to retrieve' not in pg
        ])
        return peer_groups
    except:
        return []


def get_ark_holdings_drawdowns(etf, _start_date, _end_date, holdings):
    """Get max drawdowns for current ARK holdings from precomputed data

    Args:
        etf: ETF name
        _start_date, _end_date: Analysis period
        holdings: ARK holdings DataFrame

    Returns:
        DataFrame with ticker, max_drawdown, num_drawdowns
    """
    # Load precomputed max drawdowns
    precomputed_dd = load_ark_holdings_max_drawdowns(etf)
    if len(precomputed_dd) == 0:
        return pd.DataFrame()

    # Filter holdings to analysis period to determine current tickers
    holdings_filtered = holdings[
        (holdings['Date'] >= _start_date) &
        (holdings['Date'] <= _end_date)
    ].copy()

    if len(holdings_filtered) == 0:
        return pd.DataFrame()

    # Get current holdings (latest date within analysis period)
    latest_date = holdings_filtered['Date'].max()
    current_tickers = holdings_filtered[holdings_filtered['Date'] == latest_date]['Ticker'].unique()

    # Clean ticker names
    current_tickers_clean = [t.split()[0] for t in current_tickers]

    # Filter precomputed drawdowns to current tickers only
    result = precomputed_dd[precomputed_dd['ticker'].isin(current_tickers_clean)].copy()

    # Return only the columns needed
    if len(result) > 0:
        return result[['ticker', 'max_drawdown', 'num_drawdowns']]

    return pd.DataFrame()


def get_r3000_drawdowns_filtered(detailed_drawdowns, peer_group, start_date, end_date):
    """Filter precomputed R3000 drawdowns by peer group and analysis period

    Args:
        detailed_drawdowns: DataFrame with individual drawdowns including peak_date
        peer_group: GICS industry group to filter (None for all)
        start_date: Analysis period start
        end_date: Analysis period end

    Returns:
        DataFrame with ticker, max_drawdown, num_drawdowns, gics_industry_group
    """
    df = detailed_drawdowns.copy()

    # Filter by peer group
    if peer_group and peer_group != "Russell 3000 (All)":
        df = df[df['gics_industry_group'] == peer_group]

    # Filter by analysis period (peak_date within period)
    df = filter_drawdowns_by_period(df, start_date, end_date)

    if len(df) == 0:
        return pd.DataFrame()

    # Aggregate to get max_drawdown and num_drawdowns per ticker
    result = df.groupby(['ticker', 'gics_industry_group']).agg(
        max_drawdown=('depth_pct', 'min'),
        num_drawdowns=('depth_pct', 'count')
    ).reset_index()

    return result


def get_etf_max_drawdown(etf, _start_date, _end_date):
    """Get ETF's maximum drawdown from precomputed data"""
    dd_df = load_etf_drawdowns(etf)
    if len(dd_df) == 0:
        return None

    # Filter by period (use filter_drawdowns_by_period since dd_df has peak_date, not Date)
    dd_df = filter_drawdowns_by_period(dd_df, _start_date, _end_date)
    if len(dd_df) == 0:
        return None

    # Get historical drawdowns (exclude Current)
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
        peer_groups = get_r3000_peer_groups()
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

with st.spinner("Loading drawdown distributions..."):
    # Load ARK holdings and get precomputed drawdowns for current holdings
    ark_holdings = load_ark_holdings(ark_hash, selected_etf)
    ark_dd = get_ark_holdings_drawdowns(selected_etf, start_date, end_date, ark_holdings)

    # Load precomputed R3000 drawdowns with dates (for period filtering)
    r3000_dd_detailed = load_r3000_drawdowns_with_dates()

    if len(r3000_dd_detailed) == 0:
        # Fallback to old summary data if detailed not available
        r3000_dd_full = load_r3000_drawdowns()
        if len(r3000_dd_full) == 0:
            st.warning("No precomputed R3000 drawdown data found. Run `python convert_to_parquet.py` to generate.")
            st.stop()
        # Use old data without period filtering (show warning)
        st.warning("⚠️ R3000 detailed drawdown data not available. Showing all-time drawdowns without period filtering. Run `python convert_to_parquet.py` to regenerate.")
        peer_group = None if selected_benchmark == "Russell 3000 (All)" else selected_benchmark
        if peer_group:
            r3000_dd = r3000_dd_full[r3000_dd_full['gics_industry_group'] == peer_group].copy()
        else:
            r3000_dd = r3000_dd_full.copy()
    else:
        # Check data date range vs analysis period
        data_min_date = r3000_dd_detailed['peak_date'].min()
        data_max_date = r3000_dd_detailed['peak_date'].max()

        # Check for overlap
        if end_date < data_min_date or start_date > data_max_date:
            st.error(f"⚠️ **No data overlap!** Analysis period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) does not overlap with R3000 data ({data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}).")
            st.info("Please adjust the analysis period in the sidebar to match available data.")
            st.stop()

        # Show info about partial overlap
        if start_date < data_min_date or end_date > data_max_date:
            actual_start = max(start_date, data_min_date)
            actual_end = min(end_date, data_max_date)
            st.info(f"📊 R3000 data available: {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}. Showing drawdowns with peak dates in this range.")

        # Filter by peer group AND analysis period
        peer_group = None if selected_benchmark == "Russell 3000 (All)" else selected_benchmark
        r3000_dd = get_r3000_drawdowns_filtered(r3000_dd_detailed, peer_group, start_date, end_date)

        if len(r3000_dd) == 0:
            st.warning(f"No R3000 drawdowns found in the selected analysis period ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}).")
            st.stop()

    # ETF's own drawdown (from precomputed data)
    etf_drawdown = get_etf_max_drawdown(selected_etf, start_date, end_date)

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

            # Add dummy traces for reference lines legend
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
                if etf_drawdown is not None:
                    fig.add_vline(
                        x=etf_drawdown,
                        line_dash="dash",
                        line_color="darkred",
                        line_width=2,
                        row=row, col=1
                    )

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
