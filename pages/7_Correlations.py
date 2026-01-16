"""Portfolio Correlations Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE, INPUT_DIR
from data_loader import load_ark_holdings, load_company_name

st.set_page_config(
    page_title="Portfolio Correlations",
    page_icon="🔗",
    layout="wide"
)

"""
# Portfolio Correlations

Analyze pairwise correlations across all current holdings in an ARK ETF.
"""

# Helper to get ARK holdings files modification times for cache invalidation
def get_ark_files_hash():
    """Get hash of ARK holdings files for cache invalidation"""
    mtimes = []
    for etf in ARK_ETFS:
        holdings_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'
        if holdings_file.exists():
            mtimes.append(holdings_file.stat().st_mtime)
    return max(mtimes) if mtimes else 0

@st.cache_data
def get_cached_ark_holdings(_files_hash, etf):
    """Load and cache ARK ETF holdings"""
    return load_ark_holdings(etf)

@st.cache_data
def get_current_holdings(_files_hash, etf):
    """Get list of current holdings (stocks held on the latest date)"""
    holdings = get_cached_ark_holdings(_files_hash, etf)

    # Get latest date
    latest_date = holdings['Date'].max()

    # Get tickers held on latest date
    current = holdings[holdings['Date'] == latest_date].copy()

    # Filter out currency tickers
    if 'Bloomberg Name' in current.columns:
        current = current[~current['Bloomberg Name'].str.contains('curncy', case=False, na=False)]

    return current['Ticker'].unique().tolist()

@st.cache_data
def calculate_correlation_matrix(_files_hash, etf, lookback_days, _holdings):
    """Calculate correlation matrix for current holdings

    _holdings: Pre-loaded holdings data (underscore prefix excludes from hashing)
    """
    holdings = _holdings

    # Get current holdings
    latest_date = holdings['Date'].max()
    current_tickers = holdings[holdings['Date'] == latest_date]['Ticker'].unique()

    # Filter out currency tickers
    if 'Bloomberg Name' in holdings.columns:
        currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
        current_tickers = [t for t in current_tickers if t not in currency_tickers]

    # Calculate start date for lookback period
    lookback_start = latest_date - pd.Timedelta(days=lookback_days)

    # Filter holdings to lookback period
    holdings_filtered = holdings[
        (holdings['Date'] >= lookback_start) &
        (holdings['Ticker'].isin(current_tickers))
    ].copy()

    # Pivot to get price matrix (Date x Ticker)
    price_matrix = holdings_filtered.pivot_table(
        index='Date',
        columns='Ticker',
        values='Stock_Price',
        aggfunc='first'
    )

    # Drop tickers with too many missing values (less than 50% data)
    min_data_points = len(price_matrix) * 0.5
    price_matrix = price_matrix.dropna(axis=1, thresh=int(min_data_points))

    # Calculate daily returns
    returns = price_matrix.pct_change().dropna()

    # Calculate correlation matrix
    corr_matrix = returns.corr()

    return corr_matrix, returns

def get_correlation_stats(corr_matrix):
    """Calculate summary statistics for correlation matrix"""
    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)

    # Flatten and remove NaN
    correlations = upper_triangle.values.flatten()
    correlations = correlations[~np.isnan(correlations)]

    if len(correlations) == 0:
        return None

    # Calculate statistics
    stats = {
        'mean': np.mean(correlations),
        'median': np.median(correlations),
        'std': np.std(correlations),
        'min': np.min(correlations),
        'max': np.max(correlations)
    }

    # Find highest and lowest correlation pairs
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            ticker1 = corr_matrix.columns[i]
            ticker2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            # Clean ticker names (remove "US Equity" etc.)
            ticker1_clean = ticker1.split()[0] if isinstance(ticker1, str) else ticker1
            ticker2_clean = ticker2.split()[0] if isinstance(ticker2, str) else ticker2
            pairs.append((ticker1_clean, ticker2_clean, corr))

    # Sort by correlation
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

    stats['highest_pairs'] = pairs_sorted[:5]  # Top 5 highest
    stats['lowest_pairs'] = pairs_sorted[-5:][::-1]  # Top 5 lowest (reversed)

    return stats

# Main section
st.subheader("Correlation Analysis")

# Layout
cols = st.columns([1, 3])

with cols[0]:
    controls_card = st.container(border=True)
    with controls_card:
        st.markdown("##### Select ETF")
        selected_etf = st.selectbox(
            "ETF",
            ARK_ETFS,
            label_visibility="collapsed"
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
            label_visibility="collapsed"
        )
        lookback_days = lookback_options[selected_lookback]

# Calculate correlation matrix
files_hash = get_ark_files_hash()

with st.spinner("Calculating correlations..."):
    # Load holdings once (cached)
    holdings = get_cached_ark_holdings(files_hash, selected_etf)
    corr_matrix, returns = calculate_correlation_matrix(files_hash, selected_etf, lookback_days, holdings)

if corr_matrix is not None and len(corr_matrix) > 0:
    # Get statistics
    stats = get_correlation_stats(corr_matrix)

    # Display statistics in left panel
    with cols[0]:
        ""  # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            st.markdown(f"**Holdings:** {len(corr_matrix)}")
            st.markdown(f"**Avg Correlation:** {stats['mean']:.3f}")
            st.markdown(f"**Median Correlation:** {stats['median']:.3f}")
            st.markdown(f"**Std Dev:** {stats['std']:.3f}")

            ""  # Space

            st.markdown("##### Highest Correlations")
            for t1, t2, corr in stats['highest_pairs']:
                st.markdown(f"<small>{t1} - {t2}: **{corr:.3f}**</small>", unsafe_allow_html=True)

            ""  # Space

            st.markdown("##### Lowest Correlations")
            for t1, t2, corr in stats['lowest_pairs']:
                st.markdown(f"<small>{t1} - {t2}: **{corr:.3f}**</small>", unsafe_allow_html=True)

    # Right panel: Heatmap
    with cols[1]:
        heatmap_card = st.container(border=True)
        with heatmap_card:
            # Clean ticker names for display
            clean_labels = [t.split()[0] if isinstance(t, str) else t for t in corr_matrix.columns]

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=clean_labels,
                y=clean_labels,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                hovertemplate='%{x} - %{y}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(
                    title="Correlation",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                )
            ))

            fig.update_layout(
                title=f"{selected_etf} Holdings Correlation Matrix ({selected_lookback})",
                xaxis_title="",
                yaxis_title="",
                height=700,
                xaxis=dict(tickangle=45, side='bottom'),
                yaxis=dict(autorange='reversed'),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

    ""  # Space

    # Correlation distribution histogram
    st.subheader("Correlation Distribution")

    dist_card = st.container(border=True)
    with dist_card:
        # Get all pairwise correlations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).values.flatten()
        correlations = correlations[~np.isnan(correlations)]

        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(
            x=correlations,
            nbinsx=30,
            marker_color='steelblue',
            opacity=0.7,
            name='Pairwise Correlations'
        ))

        # Add vertical lines for mean and median
        fig_hist.add_vline(x=stats['mean'], line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {stats['mean']:.3f}")
        fig_hist.add_vline(x=stats['median'], line_dash="dash", line_color="green",
                          annotation_text=f"Median: {stats['median']:.3f}")

        fig_hist.update_layout(
            title=f"Distribution of Pairwise Correlations ({len(correlations)} pairs)",
            xaxis_title="Correlation",
            yaxis_title="Count",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', range=[-1, 1]),
            yaxis=dict(gridcolor='lightgray')
        )

        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.warning(f"Not enough data to calculate correlations for {selected_etf}")
