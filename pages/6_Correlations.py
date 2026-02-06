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

from config import ARK_ETFS, START_DATE, END_DATE, INPUT_DIR, OUTPUT_DIR
from data_loader import load_ark_holdings, load_company_name, get_ark_files_hash, load_etf_prices

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

    return corr_matrix, returns, current_weights, holdings_filtered


@st.cache_data
def calculate_weighted_correlation_matrix(_files_hash, etf, lookback_days, _holdings):
    """Calculate weighted correlation matrix for current holdings

    Weight for each day = w_A,t × w_B,t (product of both stocks' weights)
    Then use weighted mean, weighted covariance, weighted variance to compute correlation.
    """
    holdings = _holdings

    # Get current holdings
    latest_date = holdings['Date'].max()
    current_tickers = holdings[holdings['Date'] == latest_date]['Ticker'].unique()

    # Filter out currency tickers and money market funds
    if 'Bloomberg Name' in holdings.columns:
        currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
        current_tickers = [t for t in current_tickers if t not in currency_tickers]

    excluded_tickers = ['FTOXX', 'FIRXX']
    current_tickers = [t for t in current_tickers if t.split()[0] not in excluded_tickers]

    # Calculate start date for lookback period
    lookback_start = latest_date - pd.Timedelta(days=lookback_days)

    # Filter holdings to lookback period
    holdings_filtered = holdings[
        (holdings['Date'] >= lookback_start) &
        (holdings['Ticker'].isin(current_tickers))
    ].copy()

    # Pivot to get price matrix and weight matrix
    price_matrix = holdings_filtered.pivot_table(
        index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
    )
    weight_matrix = holdings_filtered.pivot_table(
        index='Date', columns='Ticker', values='Weight', aggfunc='first'
    )

    # Drop tickers with too many missing values
    min_data_points = len(price_matrix) * 0.5
    valid_tickers = price_matrix.dropna(axis=1, thresh=int(min_data_points)).columns
    price_matrix = price_matrix[valid_tickers]
    weight_matrix = weight_matrix[valid_tickers]

    # Calculate daily returns
    returns = price_matrix.pct_change().dropna()
    weight_matrix = weight_matrix.loc[returns.index]  # Align weights with returns

    # Fill missing weights with 0
    weight_matrix = weight_matrix.fillna(0)

    tickers = returns.columns.tolist()
    n = len(tickers)

    # Initialize weighted correlation matrix
    weighted_corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)

    # Calculate weighted correlation for each pair
    for i in range(n):
        for j in range(i + 1, n):
            ticker_a, ticker_b = tickers[i], tickers[j]

            R_A = returns[ticker_a].values
            R_B = returns[ticker_b].values
            W_A = weight_matrix[ticker_a].values
            W_B = weight_matrix[ticker_b].values

            # Step 1: Pair weight W_t = w_A,t × w_B,t
            W_t = W_A * W_B

            # Skip if no valid weights
            if W_t.sum() == 0:
                # Fall back to unweighted correlation
                valid_mask = ~(np.isnan(R_A) | np.isnan(R_B))
                if valid_mask.sum() > 1:
                    corr_val = np.corrcoef(R_A[valid_mask], R_B[valid_mask])[0, 1]
                else:
                    corr_val = np.nan
                weighted_corr.iloc[i, j] = corr_val
                weighted_corr.iloc[j, i] = corr_val
                continue

            # Step 2: Normalize weights
            W_norm = W_t / W_t.sum()

            # Step 3: Weighted means
            mu_A = np.sum(W_norm * R_A)
            mu_B = np.sum(W_norm * R_B)

            # Step 4: Weighted covariance and variance
            cov_AB = np.sum(W_norm * (R_A - mu_A) * (R_B - mu_B))
            var_A = np.sum(W_norm * (R_A - mu_A) ** 2)
            var_B = np.sum(W_norm * (R_B - mu_B) ** 2)

            # Step 5: Weighted correlation
            if var_A > 0 and var_B > 0:
                corr_val = cov_AB / np.sqrt(var_A * var_B)
                # Clamp to [-1, 1] for numerical stability
                corr_val = np.clip(corr_val, -1, 1)
            else:
                corr_val = np.nan

            weighted_corr.iloc[i, j] = corr_val
            weighted_corr.iloc[j, i] = corr_val

    return weighted_corr


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

        ""  # Space

        st.markdown("##### Correlation Type")
        use_weighted_corr = st.toggle("Weighted Correlation", value=False)
        if use_weighted_corr:
            st.caption("Weight = w_A × w_B per day")

# Calculate correlation matrix
files_hash = get_ark_files_hash()

with st.spinner("Calculating correlations..."):
    # Load holdings once (cached)
    holdings = load_ark_holdings(files_hash, selected_etf)
    corr_matrix_unweighted, returns, current_weights, holdings_filtered = calculate_correlation_matrix(files_hash, selected_etf, lookback_days, holdings)

    # Use weighted or unweighted correlation matrix based on toggle
    if use_weighted_corr:
        corr_matrix = calculate_weighted_correlation_matrix(files_hash, selected_etf, lookback_days, holdings)
    else:
        corr_matrix = corr_matrix_unweighted

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

            corr_type_label = "Weighted" if use_weighted_corr else "Unweighted"
            fig.update_layout(
                title=f"{selected_etf} Holdings Correlation Matrix ({selected_lookback}, {corr_type_label})",
                xaxis_title="",
                yaxis_title="",
                height=700,
                xaxis=dict(tickangle=45, side='bottom', dtick=1),
                yaxis=dict(autorange='reversed', dtick=1),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, width='stretch')

            if use_weighted_corr:
                st.markdown("<small>*Weighted correlation: each day's weight = w_A × w_B (product of both stocks' portfolio weights). Days when positions are larger contribute more to the correlation.*</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>*Holdings with less than 50% price data in the lookback period are excluded from the matrix.*</small>", unsafe_allow_html=True)

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

    ""  # Space

    # Correlation vs Performance Analysis
    st.subheader("Correlation vs Performance Analysis")

    perf_card = st.container(border=True)
    with perf_card:
        # Load ETF prices
        etf_prices = load_etf_prices(selected_etf)

        if len(etf_prices) > 0 and len(rolling_corr) > 0:
            # Calculate ETF returns
            etf_prices = etf_prices.copy()
            etf_prices['Return'] = etf_prices['Close'].pct_change()

            # Merge correlation data with ETF returns
            corr_perf = pd.merge(
                rolling_corr,
                etf_prices[['Date', 'Close', 'Return']],
                on='Date',
                how='inner'
            )

            if len(corr_perf) > 0:
                # Calculate forward returns (next 5, 10, 20 days)
                corr_perf['Fwd_5d_Return'] = corr_perf['Close'].shift(-5) / corr_perf['Close'] - 1
                corr_perf['Fwd_10d_Return'] = corr_perf['Close'].shift(-10) / corr_perf['Close'] - 1
                corr_perf['Fwd_20d_Return'] = corr_perf['Close'].shift(-20) / corr_perf['Close'] - 1

                # Calculate correlation change
                corr_perf['Corr_Change'] = corr_perf['weighted_mean_corr'].diff()

                # Create dual-axis chart: Correlation vs Cumulative Return
                from plotly.subplots import make_subplots

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

                # Weighted correlation
                fig_perf.add_trace(
                    go.Scatter(
                        x=corr_perf['Date'],
                        y=corr_perf['weighted_mean_corr'],
                        mode='lines',
                        name='Weighted Correlation',
                        line=dict(color='steelblue', width=2, dash='dot'),
                        hovertemplate='<b>Weighted Correlation</b><br>Date: %{x|%Y-%m-%d}<br>Correlation: %{y:.4f}<extra></extra>'
                    ),
                    secondary_y=True
                )

                fig_perf.update_layout(
                    title=f"{selected_etf} Cumulative Return vs Weighted Correlation",
                    height=450,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='x unified'
                )

                fig_perf.update_xaxes(title_text="Date", gridcolor='lightgray')
                fig_perf.update_yaxes(title_text="Cumulative Return (%)", secondary_y=False, gridcolor='lightgray')
                fig_perf.update_yaxes(title_text="Weighted Correlation", secondary_y=True)

                st.plotly_chart(fig_perf, width='stretch')

                ""  # Space

                # Correlation regime analysis
                st.markdown("#### Correlation Regime Analysis")

                # Split into high/low correlation regimes
                median_corr = corr_perf['weighted_mean_corr'].median()

                high_corr = corr_perf[corr_perf['weighted_mean_corr'] >= median_corr]
                low_corr = corr_perf[corr_perf['weighted_mean_corr'] < median_corr]

                # Calculate statistics for each regime
                regime_cols = st.columns(2)

                with regime_cols[0]:
                    st.markdown(f"**High Correlation Regime** (≥ {median_corr:.3f})")
                    if len(high_corr) > 0:
                        high_avg_return = high_corr['Return'].mean() * 252 * 100  # Annualized
                        high_volatility = high_corr['Return'].std() * np.sqrt(252) * 100
                        high_sharpe = high_avg_return / high_volatility if high_volatility > 0 else 0
                        st.metric("Annualized Return", f"{high_avg_return:.1f}%")
                        st.metric("Annualized Volatility", f"{high_volatility:.1f}%")
                        st.metric("Sharpe Ratio", f"{high_sharpe:.2f}")
                        st.caption(f"Days: {len(high_corr)}")

                with regime_cols[1]:
                    st.markdown(f"**Low Correlation Regime** (< {median_corr:.3f})")
                    if len(low_corr) > 0:
                        low_avg_return = low_corr['Return'].mean() * 252 * 100  # Annualized
                        low_volatility = low_corr['Return'].std() * np.sqrt(252) * 100
                        low_sharpe = low_avg_return / low_volatility if low_volatility > 0 else 0
                        st.metric("Annualized Return", f"{low_avg_return:.1f}%")
                        st.metric("Annualized Volatility", f"{low_volatility:.1f}%")
                        st.metric("Sharpe Ratio", f"{low_sharpe:.2f}")
                        st.caption(f"Days: {len(low_corr)}")

                ""  # Space

                # Summary insight
                if len(high_corr) > 0 and len(low_corr) > 0:
                    high_avg = high_corr['Return'].mean() * 252 * 100
                    low_avg = low_corr['Return'].mean() * 252 * 100

                    if low_avg > high_avg:
                        insight = f"📉 **Lower correlation = Better performance**: When correlation is below {median_corr:.3f}, {selected_etf} has higher annualized returns ({low_avg:.1f}% vs {high_avg:.1f}%)."
                    else:
                        insight = f"📈 **Higher correlation = Better performance**: When correlation is above {median_corr:.3f}, {selected_etf} has higher annualized returns ({high_avg:.1f}% vs {low_avg:.1f}%)."

                    st.info(insight)

                ""  # Space

                # Forward return analysis
                st.markdown("#### Correlation Change vs Forward Returns")

                # Create scatter plot: Correlation Change vs Forward Returns
                valid_fwd = corr_perf.dropna(subset=['Corr_Change', 'Fwd_20d_Return'])

                if len(valid_fwd) > 20:
                    fig_scatter = go.Figure()

                    fig_scatter.add_trace(go.Scatter(
                        x=valid_fwd['Corr_Change'],
                        y=valid_fwd['Fwd_20d_Return'] * 100,
                        mode='markers',
                        marker=dict(
                            color='steelblue',
                            size=6,
                            opacity=0.5
                        ),
                        hovertemplate='<b>Correlation Change</b>: %{x:.4f}<br><b>20-Day Forward Return</b>: %{y:.2f}%<extra></extra>'
                    ))

                    # Add trend line
                    z = np.polyfit(valid_fwd['Corr_Change'], valid_fwd['Fwd_20d_Return'] * 100, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid_fwd['Corr_Change'].min(), valid_fwd['Corr_Change'].max(), 100)

                    fig_scatter.add_trace(go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2, dash='dash')
                    ))

                    # Calculate correlation
                    corr_coefficient = valid_fwd['Corr_Change'].corr(valid_fwd['Fwd_20d_Return'])

                    fig_scatter.update_layout(
                        title=f"Correlation Change vs 20-Day Forward Return (r = {corr_coefficient:.3f})",
                        xaxis_title="Correlation Change (daily)",
                        yaxis_title="20-Day Forward Return (%)",
                        height=400,
                        showlegend=False,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis=dict(gridcolor='lightgray', zeroline=True, zerolinecolor='gray'),
                        yaxis=dict(gridcolor='lightgray', zeroline=True, zerolinecolor='gray')
                    )

                    st.plotly_chart(fig_scatter, width='stretch')

                    # Interpretation
                    if corr_coefficient < -0.1:
                        st.success(f"**Negative relationship (r = {corr_coefficient:.3f})**: When correlation decreases, forward returns tend to be higher. Lower correlation may indicate better diversification benefits.")
                    elif corr_coefficient > 0.1:
                        st.warning(f"**Positive relationship (r = {corr_coefficient:.3f})**: When correlation increases, forward returns tend to be higher. This may indicate momentum effects during rallies.")
                    else:
                        st.info(f"**Weak relationship (r = {corr_coefficient:.3f})**: Correlation changes have little predictive power for forward returns.")
                else:
                    st.warning("Not enough data for forward return analysis.")
            else:
                st.warning("Could not merge correlation and price data.")
        else:
            st.warning(f"No price data available for {selected_etf}")

else:
    st.warning(f"Not enough data to calculate correlations for {selected_etf}")
