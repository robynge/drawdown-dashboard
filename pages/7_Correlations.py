"""Portfolio Correlations Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
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

@st.cache_data
def calculate_rolling_correlations(_files_hash, etf, rolling_window, _returns):
    """Calculate rolling mean/median pairwise correlation over time

    _returns: Pre-loaded returns data (underscore prefix excludes from hashing)
    """
    returns = _returns

    if len(returns) < rolling_window:
        return pd.DataFrame()

    results = []
    dates = returns.index
    for i in range(rolling_window, len(returns) + 1):
        window_returns = returns.iloc[i - rolling_window:i]
        corr_matrix = window_returns.corr()

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.where(mask).values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]

        if len(corr_values) > 0:
            results.append({
                'Date': dates[i - 1],
                'mean_corr': np.mean(corr_values),
                'median_corr': np.median(corr_values)
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
                xaxis=dict(tickangle=45, side='bottom', dtick=1),
                yaxis=dict(autorange='reversed', dtick=1),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<small>*Holdings with less than 50% price data in the lookback period are excluded from the matrix.</small>", unsafe_allow_html=True)

    ""  # Space

    # Correlation Time Series
    ts_header_cols = st.columns([3, 1])
    with ts_header_cols[0]:
        st.subheader("Correlation Time Series")
    with ts_header_cols[1]:
        rolling_options = {
            "20 Days": 20,
            "30 Days": 30
        }
        rolling_window = st.selectbox(
            "Rolling Window",
            options=list(rolling_options.keys()),
            index=0,
            label_visibility="collapsed"
        )
        rolling_window = rolling_options[rolling_window]

    ts_card = st.container(border=True)
    with ts_card:
        rolling_corr = calculate_rolling_correlations(files_hash, selected_etf, rolling_window, returns)

        if len(rolling_corr) > 0:
            fig_ts = go.Figure()

            fig_ts.add_trace(go.Scatter(
                x=rolling_corr['Date'],
                y=rolling_corr['mean_corr'],
                mode='lines',
                name='Mean Correlation',
                line=dict(color='red', width=2),
                hovertemplate='<b>Mean Correlation</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>'
            ))

            fig_ts.add_trace(go.Scatter(
                x=rolling_corr['Date'],
                y=rolling_corr['median_corr'],
                mode='lines',
                name='Median Correlation',
                line=dict(color='green', width=2),
                hovertemplate='<b>Median Correlation</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>'
            ))

            fig_ts.update_layout(
                title=f"{selected_etf} Rolling {rolling_window}-Day Pairwise Correlation",
                xaxis_title="Date",
                yaxis_title="Correlation",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )

            st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown(f"<small>*Each point shows the mean/median of all pairwise correlations among current holdings, calculated using the past {rolling_window} trading days. Rising values indicate increasing portfolio concentration risk.</small>", unsafe_allow_html=True)
        else:
            st.warning(f"Not enough data for {rolling_window}-day rolling correlation.")

    ""  # Space

    # Statistical Test: Compare correlation matrices at two time points
    st.subheader("Statistical Test: Correlation Change")

    test_card = st.container(border=True)
    with test_card:
        if len(rolling_corr) >= 2:
            # Get correlation matrices at T1 (N days ago) and T2 (today)
            compare_days = min(rolling_window, len(rolling_corr) - 1)

            # T2: most recent
            t2_idx = len(returns) - 1
            t2_returns = returns.iloc[t2_idx - rolling_window + 1:t2_idx + 1]
            corr_t2 = t2_returns.corr()

            # T1: N days before T2
            t1_idx = t2_idx - compare_days
            if t1_idx >= rolling_window - 1:
                t1_returns = returns.iloc[t1_idx - rolling_window + 1:t1_idx + 1]
                corr_t1 = t1_returns.corr()

                # Extract upper triangles
                mask = np.triu(np.ones_like(corr_t1, dtype=bool), k=1)
                corr_t1_vals = corr_t1.where(mask).values.flatten()
                corr_t1_vals = corr_t1_vals[~np.isnan(corr_t1_vals)]
                corr_t2_vals = corr_t2.where(mask).values.flatten()
                corr_t2_vals = corr_t2_vals[~np.isnan(corr_t2_vals)]

                # Statistics
                mean_t1 = np.mean(corr_t1_vals)
                mean_t2 = np.mean(corr_t2_vals)
                delta_mean = mean_t2 - mean_t1

                # Frobenius norm of difference
                diff_matrix = corr_t2.values - corr_t1.values
                frobenius_norm = np.sqrt(np.sum(diff_matrix**2))

                # Paired t-test on correlation values
                t_stat, p_value = stats.ttest_rel(corr_t2_vals, corr_t1_vals)

                # Display results
                t1_date = returns.index[t1_idx].strftime('%Y-%m-%d')
                t2_date = returns.index[t2_idx].strftime('%Y-%m-%d')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Mean Corr ({t1_date})", f"{mean_t1:.3f}")
                with col2:
                    st.metric(f"Mean Corr ({t2_date})", f"{mean_t2:.3f}", delta=f"{delta_mean:+.3f}")
                with col3:
                    sig_text = "Significant" if p_value < 0.05 else "Not Significant"
                    st.metric("Change Significance", sig_text, delta=f"p={p_value:.3f}")

                st.markdown(f"<small>*Paired t-test comparing all pairwise correlations between {t1_date} and {t2_date}. Frobenius norm of difference matrix: {frobenius_norm:.3f}</small>", unsafe_allow_html=True)
            else:
                st.warning("Not enough historical data for statistical comparison.")
        else:
            st.warning("Not enough data for statistical test.")

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

        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.warning(f"Not enough data to calculate correlations for {selected_etf}")
