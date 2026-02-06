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
from data_loader import load_ark_holdings, load_company_name, get_ark_files_hash

st.set_page_config(
    page_title="Portfolio Correlations",
    page_icon="🔗",
    layout="wide"
)

"""
# Portfolio Correlations

Analyze pairwise correlations across all current holdings in an ARK ETF.
"""

@st.cache_data
def get_current_holdings(_files_hash, etf):
    """Get list of current holdings (stocks held on the latest date)"""
    holdings = load_ark_holdings(_files_hash, etf)

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

    # Filter out currency tickers and money market funds
    if 'Bloomberg Name' in holdings.columns:
        currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
        current_tickers = [t for t in current_tickers if t not in currency_tickers]

    # Filter out specific tickers (e.g., money market funds)
    excluded_tickers = ['FTOXX', 'FIRXX']
    current_tickers = [t for t in current_tickers if t.split()[0] not in excluded_tickers]

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

    # Get weights for current holdings
    current_weights = holdings_filtered[holdings_filtered['Date'] == latest_date][['Ticker', 'Weight']].copy()
    current_weights = current_weights[current_weights['Ticker'].isin(corr_matrix.columns)]

    return corr_matrix, returns, current_weights

def get_correlation_stats(corr_matrix, weights_df=None):
    """Calculate summary statistics for correlation matrix (vectorized)

    Args:
        corr_matrix: correlation matrix DataFrame
        weights_df: optional DataFrame with Ticker and Weight columns for weighted correlation
    """
    n = len(corr_matrix.columns)
    if n < 2:
        return None

    # Get upper triangle indices (vectorized)
    triu_i, triu_j = np.triu_indices(n, k=1)

    # Extract correlations from upper triangle (vectorized)
    corr_values = corr_matrix.values[triu_i, triu_j]
    valid_mask = ~np.isnan(corr_values)
    correlations = corr_values[valid_mask]

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

    # Calculate weighted correlation if weights provided (vectorized)
    if weights_df is not None and len(weights_df) > 0:
        # Create weight array aligned with correlation matrix columns
        weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
        weights_arr = np.array([weights_dict.get(t, 0) for t in corr_matrix.columns])

        # Compute pair weights using outer product (vectorized)
        weight_matrix = np.outer(weights_arr, weights_arr)
        pair_weights = weight_matrix[triu_i, triu_j]

        # Filter to valid correlations and positive weights
        valid_weight_mask = valid_mask & (pair_weights > 0)
        if valid_weight_mask.any():
            valid_corrs = corr_values[valid_weight_mask]
            valid_weights = pair_weights[valid_weight_mask]
            stats['weighted_mean'] = np.average(valid_corrs, weights=valid_weights)
        else:
            stats['weighted_mean'] = stats['mean']
    else:
        stats['weighted_mean'] = stats['mean']

    # Find highest and lowest correlation pairs (vectorized)
    tickers = corr_matrix.columns.tolist()
    # Clean ticker names
    tickers_clean = [t.split()[0] if isinstance(t, str) else t for t in tickers]

    # Get indices for valid correlations, sorted by value
    valid_indices = np.where(valid_mask)[0]
    sorted_indices = valid_indices[np.argsort(corr_values[valid_mask])]

    # Build pairs list for top 5 highest and lowest
    def get_pair(idx):
        i, j = triu_i[idx], triu_j[idx]
        return (tickers_clean[i], tickers_clean[j], corr_values[idx])

    stats['highest_pairs'] = [get_pair(idx) for idx in sorted_indices[-5:][::-1]]
    stats['lowest_pairs'] = [get_pair(idx) for idx in sorted_indices[:5]]

    return stats

@st.cache_data
def calculate_rolling_correlations(_files_hash, etf, rolling_window, _returns, _holdings):
    """Calculate rolling mean/median pairwise correlation over time (vectorized)

    _returns: Pre-loaded returns data (underscore prefix excludes from hashing)
    _holdings: Pre-loaded holdings data for weight calculation
    """
    returns = _returns
    holdings = _holdings

    if len(returns) < rolling_window:
        return pd.DataFrame()

    results = []
    dates = returns.index
    n_tickers = len(returns.columns)

    # Pre-compute upper triangle indices (reused for all windows)
    triu_i, triu_j = np.triu_indices(n_tickers, k=1)

    # Pre-compute sorted holdings dates for faster lookup
    holdings_dates = np.sort(holdings['Date'].unique())

    for i in range(rolling_window, len(returns) + 1):
        window_returns = returns.iloc[i - rolling_window:i]
        current_date = dates[i - 1]
        corr_matrix = window_returns.corr()

        # Extract upper triangle correlations (vectorized)
        corr_values = corr_matrix.values[triu_i, triu_j]
        valid_mask = ~np.isnan(corr_values)
        valid_corrs = corr_values[valid_mask]

        if len(valid_corrs) == 0:
            continue

        # Unweighted correlations
        mean_corr = np.mean(valid_corrs)
        median_corr = np.median(valid_corrs)

        # Weighted correlation - find closest holdings date using binary search
        valid_dates = holdings_dates[holdings_dates <= current_date]
        if len(valid_dates) == 0:
            weighted_mean = mean_corr
        else:
            closest_date = valid_dates[-1]  # Already sorted, take last
            weights_df = holdings[holdings['Date'] == closest_date][['Ticker', 'Weight']]
            weights_df = weights_df[weights_df['Ticker'].isin(corr_matrix.columns)]

            if len(weights_df) > 0:
                # Vectorized weighted correlation calculation
                weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
                weights_arr = np.array([weights_dict.get(t, 0) for t in corr_matrix.columns])

                # Compute pair weights using outer product
                weight_matrix = np.outer(weights_arr, weights_arr)
                pair_weights = weight_matrix[triu_i, triu_j]

                # Filter to valid correlations and positive weights
                valid_weight_mask = valid_mask & (pair_weights > 0)
                if valid_weight_mask.any():
                    weighted_mean = np.average(
                        corr_values[valid_weight_mask],
                        weights=pair_weights[valid_weight_mask]
                    )
                else:
                    weighted_mean = mean_corr
            else:
                weighted_mean = mean_corr

        results.append({
            'Date': current_date,
            'mean_corr': mean_corr,
            'median_corr': median_corr,
            'weighted_mean_corr': weighted_mean
        })

    return pd.DataFrame(results)

# Main section
st.subheader("Correlation Analysis")

# Layout
cols = st.columns([1, 3])

with cols[0]:
    controls_card = st.container(border=True)
    with controls_card:
        st.markdown("##### Select ETF")
        selected_etf = st.pills(
            "ETF",
            options=ARK_ETFS,
            default=ARK_ETFS[0],
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

        ""  # Space

        st.markdown("##### Rolling Window")
        rolling_options = {
            "20 Days": 20,
            "30 Days": 30
        }
        selected_rolling = st.pills(
            "Rolling",
            options=list(rolling_options.keys()),
            default="20 Days",
            label_visibility="collapsed"
        )
        rolling_window = rolling_options[selected_rolling]

# Calculate correlation matrix
files_hash = get_ark_files_hash()

with st.spinner("Calculating correlations..."):
    # Load holdings once (cached)
    holdings = load_ark_holdings(files_hash, selected_etf)
    corr_matrix, returns, current_weights = calculate_correlation_matrix(files_hash, selected_etf, lookback_days, holdings)

if corr_matrix is not None and len(corr_matrix) > 0:
    # Get statistics (including weighted correlation)
    stats = get_correlation_stats(corr_matrix, current_weights)

    # Display statistics in left panel
    with cols[0]:
        ""  # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            st.markdown(f"**Holdings:** {len(corr_matrix)}")

            ""  # Space

            st.markdown("**Correlation (Unweighted)**")
            st.markdown(f"Mean: **{stats['mean']:.3f}**")
            st.markdown(f"Median: **{stats['median']:.3f}**")

            ""  # Space

            st.markdown("**Correlation (Weighted)**")
            st.markdown(f"Mean: **{stats['weighted_mean']:.3f}**")
            st.caption("Weighted by position size")

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
                xaxis=dict(tickangle=45, side='bottom', dtick=1),
                yaxis=dict(autorange='reversed', dtick=1),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, width='stretch')

            st.markdown("<small>*Holdings with less than 50% price data in the lookback period are excluded from the matrix.</small>", unsafe_allow_html=True)

    ""  # Space

    # Correlation Time Series
    st.subheader("Correlation Time Series")

    ts_card = st.container(border=True)
    with ts_card:
        rolling_corr = calculate_rolling_correlations(files_hash, selected_etf, rolling_window, returns, holdings)

        if len(rolling_corr) > 0:
            fig_ts = go.Figure()

            # Weighted mean correlation (primary)
            fig_ts.add_trace(go.Scatter(
                x=rolling_corr['Date'],
                y=rolling_corr['weighted_mean_corr'],
                mode='lines',
                name='Weighted Mean',
                line=dict(color='steelblue', width=2),
                hovertemplate='<b>Weighted Mean</b><br>Date (x): %{x|%Y-%m-%d}<br>Correlation (y): %{y:.4f}<extra></extra>'
            ))

            # Unweighted mean correlation
            fig_ts.add_trace(go.Scatter(
                x=rolling_corr['Date'],
                y=rolling_corr['mean_corr'],
                mode='lines',
                name='Unweighted Mean',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Unweighted Mean</b><br>Date (x): %{x|%Y-%m-%d}<br>Correlation (y): %{y:.4f}<extra></extra>'
            ))

            fig_ts.update_layout(
                title=f"{selected_etf} Rolling {rolling_window}-Day Pairwise Correlation (Weighted vs Unweighted)",
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

            st.markdown(f"<small>*Solid blue = weighted by position size (large positions matter more). Dashed red = unweighted (equal weight to all pairs). Rising values indicate increasing concentration risk.</small>", unsafe_allow_html=True)
        else:
            st.warning(f"Not enough data for {rolling_window}-day rolling correlation.")

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
            name='Pairwise Correlations',
            hovertemplate='<b>Pairwise Correlations</b><br>Correlation (x): %{x:.4f}<br>Count (y): %{y}<extra></extra>'
        ))

        # Add vertical lines for mean and median
        fig_hist.add_vline(x=stats['mean'], line_dash="dash", line_color="red", line_width=2)
        fig_hist.add_vline(x=stats['median'], line_dash="dash", line_color="green", line_width=2)

        # Add dummy traces for legend
        fig_hist.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=f"Mean: {stats['mean']:.3f}",
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True
        ))
        fig_hist.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name=f"Median: {stats['median']:.3f}",
            line=dict(color='green', width=2, dash='dash'),
            showlegend=True
        ))

        fig_hist.update_layout(
            title=f"Distribution of Pairwise Correlations ({len(correlations)} pairs)",
            xaxis_title="Correlation",
            yaxis_title="Count",
            height=400,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', range=[-1, 1]),
            yaxis=dict(gridcolor='lightgray')
        )

        st.plotly_chart(fig_hist, width='stretch')

else:
    st.warning(f"Not enough data to calculate correlations for {selected_etf}")
