"""Portfolio Correlations Page - Using Precomputed Data"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, OUTPUT_DIR
from precomputed_loader import (
    load_correlation_matrix,
    load_weighted_correlation_matrix,
    load_rolling_correlations,
    load_correlation_returns,
    load_current_weights,
    load_etf_drawdowns,
    filter_by_period,
    check_precomputed_exists,
    ARK_PRECOMPUTED_DIR
)
from data_loader import load_etf_prices, load_ark_holdings, get_ark_files_hash
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector

st.set_page_config(
    page_title="Portfolio Correlations",
    page_icon="🔗",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Portfolio Correlations

Analyze pairwise correlations across all current holdings in an ARK ETF.
"""

st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")
    # Show diagnostic info
    with st.expander("Diagnostic Info"):
        st.code(f"Precomputed dir: {ARK_PRECOMPUTED_DIR}")
        st.code(f"Dir exists: {ARK_PRECOMPUTED_DIR.exists()}")
        if ARK_PRECOMPUTED_DIR.exists():
            files = list(ARK_PRECOMPUTED_DIR.glob('*.parquet'))
            st.code(f"Parquet files found: {len(files)}")


def get_period_dates(drawdowns_df):
    """Extract drawdown and recovery period dates from drawdown data.

    Drawdown: peak_date → trough_date
    Recovery: trough_date → next peak_date (sorted by date)

    Returns:
        tuple: (drawdown_dates, recovery_dates) - lists of date ranges
    """
    # Filter out 'Current' drawdown and sort by peak_date
    hist_dd = drawdowns_df[drawdowns_df['rank'] != 'Current'].copy()
    if len(hist_dd) == 0:
        return [], []

    hist_dd = hist_dd.sort_values('peak_date').reset_index(drop=True)

    # Drawdown periods: peak → trough
    drawdown_periods = []
    for _, row in hist_dd.iterrows():
        drawdown_periods.append((row['peak_date'], row['trough_date']))

    # Recovery periods: trough → next peak
    # Sort all peaks and troughs chronologically
    events = []
    for _, row in hist_dd.iterrows():
        events.append(('peak', row['peak_date']))
        events.append(('trough', row['trough_date']))
    events.sort(key=lambda x: x[1])

    recovery_periods = []
    for i, (event_type, event_date) in enumerate(events):
        if event_type == 'trough':
            # Find next peak after this trough
            for j in range(i + 1, len(events)):
                if events[j][0] == 'peak':
                    recovery_periods.append((event_date, events[j][1]))
                    break

    return drawdown_periods, recovery_periods


def filter_returns_by_periods(returns_df, periods):
    """Filter returns DataFrame to only include dates within specified periods.

    Args:
        returns_df: DataFrame with Date as index (or column) and ticker columns
        periods: List of (start_date, end_date) tuples

    Returns:
        Filtered DataFrame with Date as column
    """
    if len(periods) == 0 or len(returns_df) == 0:
        return pd.DataFrame()

    # Ensure Date is a column
    if returns_df.index.name == 'Date' or (hasattr(returns_df.index, 'name') and returns_df.index.name is None and isinstance(returns_df.index[0], pd.Timestamp)):
        df = returns_df.reset_index()
        if df.columns[0] != 'Date':
            df = df.rename(columns={df.columns[0]: 'Date'})
    else:
        df = returns_df.copy()

    date_col = 'Date'
    if date_col not in df.columns:
        # Try first column
        date_col = df.columns[0]

    # Ensure dates are datetime
    df[date_col] = pd.to_datetime(df[date_col])

    mask = pd.Series(False, index=df.index)
    for start_date, end_date in periods:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        period_mask = (df[date_col] >= start_ts) & (df[date_col] <= end_ts)
        mask = mask | period_mask

    result = df[mask].copy()
    return result


@st.cache_data
def calculate_holdings_returns(etf):
    """Calculate daily returns for all holdings from price data.

    Returns DataFrame with Date index and ticker columns containing daily returns.
    """
    files_hash = get_ark_files_hash()
    holdings = load_ark_holdings(files_hash, etf)

    if len(holdings) == 0:
        return pd.DataFrame()

    # Filter to stocks only (exclude non-stock tickers)
    holdings = holdings[holdings['Ticker'].notna()].copy()

    # Use Stock_Price column
    if 'Stock_Price' not in holdings.columns:
        return pd.DataFrame()

    price_col = 'Stock_Price'

    # Get prices - pivot to have dates as rows, tickers as columns
    prices = holdings.pivot_table(
        index='Date',
        columns='Ticker',
        values=price_col,
        aggfunc='last'
    )

    # Calculate returns
    returns = prices.pct_change()

    return returns


def calculate_correlation_from_returns(returns_df):
    """Calculate correlation matrix from returns DataFrame.

    Args:
        returns_df: DataFrame with Date column (or index) and ticker columns

    Returns:
        Correlation matrix DataFrame
    """
    if len(returns_df) < 10:  # Need minimum data points
        return pd.DataFrame()

    # Drop Date column if present and calculate correlations
    ticker_cols = [col for col in returns_df.columns if col not in ['Date', 'index']]
    if len(ticker_cols) == 0:
        return pd.DataFrame()

    returns_only = returns_df[ticker_cols]

    # Filter columns that have at least some data (at least 1 non-NaN value)
    valid_cols = returns_only.columns[returns_only.notna().sum() > 0]
    if len(valid_cols) < 2:
        return pd.DataFrame()

    # Calculate correlation
    corr_matrix = returns_only[valid_cols].corr()

    # Remove stocks that have NO valid correlation with any other stock
    # (all off-diagonal values are NaN)
    has_valid_corr = (corr_matrix.notna().sum() > 1)  # >1 because diagonal is always 1.0
    corr_matrix = corr_matrix.loc[has_valid_corr, has_valid_corr]

    return corr_matrix


def get_correlation_stats(corr_matrix, weights_df=None):
    """Calculate summary statistics for correlation matrix (vectorized)"""
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
        weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
        weights_arr = np.array([weights_dict.get(t, 0) for t in corr_matrix.columns])

        weight_matrix = np.outer(weights_arr, weights_arr)
        pair_weights = weight_matrix[triu_i, triu_j]

        valid_weight_mask = valid_mask & (pair_weights > 0)
        if valid_weight_mask.any():
            valid_corrs = corr_values[valid_weight_mask]
            valid_weights = pair_weights[valid_weight_mask]
            stats['weighted_mean'] = np.average(valid_corrs, weights=valid_weights)

            sorted_idx = np.argsort(valid_corrs)
            sorted_corrs = valid_corrs[sorted_idx]
            sorted_weights = valid_weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            median_idx = np.searchsorted(cumsum, 0.5)
            stats['weighted_median'] = sorted_corrs[min(median_idx, len(sorted_corrs) - 1)]
        else:
            stats['weighted_mean'] = stats['mean']
            stats['weighted_median'] = stats['median']
    else:
        stats['weighted_mean'] = stats['mean']
        stats['weighted_median'] = stats['median']

    # Find highest and lowest correlation pairs (vectorized)
    tickers = corr_matrix.columns.tolist()
    tickers_clean = [t.split()[0] if isinstance(t, str) else t for t in tickers]

    valid_indices = np.where(valid_mask)[0]
    sorted_indices = valid_indices[np.argsort(corr_values[valid_mask])]

    def get_pair(idx):
        i, j = triu_i[idx], triu_j[idx]
        return (tickers_clean[i], tickers_clean[j], corr_values[idx])

    stats['highest_pairs'] = [get_pair(idx) for idx in sorted_indices[-5:][::-1]]
    stats['lowest_pairs'] = [get_pair(idx) for idx in sorted_indices[:5]]

    return stats


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

        st.markdown("##### Correlation Mode")
        correlation_mode = st.pills(
            "Mode",
            options=["Overall", "Drawdown", "Recovery"],
            default="Overall",
            label_visibility="collapsed"
        )

        ""  # Space

        st.markdown("##### Correlation Type")
        use_weighted_corr = st.toggle("Weighted Correlation", value=False)

# Load data based on correlation mode
with st.spinner("Loading correlations..."):
    # Load ETF drawdowns for period-based calculations
    etf_drawdowns = load_etf_drawdowns(selected_etf)
    drawdown_periods, recovery_periods = get_period_dates(etf_drawdowns)

    # Load current weights first (needed for filtering)
    current_weights = load_current_weights(selected_etf)
    current_tickers = set(current_weights['Ticker'].tolist()) if len(current_weights) > 0 else set()

    # Load returns - use full historical data for Drawdown/Recovery modes
    if correlation_mode in ["Drawdown", "Recovery"]:
        returns = calculate_holdings_returns(selected_etf)  # Full history from holdings
    else:
        returns = load_correlation_returns(selected_etf, lookback_days)

    # Calculate correlation matrix based on mode
    if correlation_mode == "Overall":
        # Use precomputed correlation matrices
        if use_weighted_corr:
            corr_matrix = load_weighted_correlation_matrix(selected_etf, lookback_days)
        else:
            corr_matrix = load_correlation_matrix(selected_etf, lookback_days)
        corr_matrix_unweighted = load_correlation_matrix(selected_etf, lookback_days)
        period_info = f"Full {lookback_days}-day period"
        num_periods = 1
        total_days = lookback_days

    elif correlation_mode == "Drawdown":
        # Filter returns to drawdown periods only
        filtered_returns = pd.DataFrame()
        if len(returns) > 0 and len(drawdown_periods) > 0:
            filtered_returns = filter_returns_by_periods(returns, drawdown_periods)
            corr_matrix = calculate_correlation_from_returns(filtered_returns)
            corr_matrix_unweighted = corr_matrix  # Same for both
            total_days = len(filtered_returns)
        else:
            corr_matrix = pd.DataFrame()
            corr_matrix_unweighted = pd.DataFrame()
            total_days = 0
        period_info = f"{len(drawdown_periods)} drawdown periods"
        num_periods = len(drawdown_periods)

    else:  # Recovery
        # Filter returns to recovery periods only
        filtered_returns = pd.DataFrame()
        if len(returns) > 0 and len(recovery_periods) > 0:
            filtered_returns = filter_returns_by_periods(returns, recovery_periods)
            corr_matrix = calculate_correlation_from_returns(filtered_returns)
            corr_matrix_unweighted = corr_matrix  # Same for both
            total_days = len(filtered_returns)
        else:
            corr_matrix = pd.DataFrame()
            corr_matrix_unweighted = pd.DataFrame()
            total_days = 0
        period_info = f"{len(recovery_periods)} recovery periods"
        num_periods = len(recovery_periods)

    # Load rolling correlations (only for Overall mode)
    rolling_corr = load_rolling_correlations(selected_etf, lookback_days, rolling_window) if correlation_mode == "Overall" else pd.DataFrame()

if corr_matrix is not None and len(corr_matrix) > 0:
    # Get statistics
    stats = get_correlation_stats(corr_matrix, current_weights)

    # Display statistics in left panel
    with cols[0]:
        ""  # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            st.markdown(f"**Mode:** {correlation_mode}")
            st.markdown(f"**Periods:** {period_info}")
            if correlation_mode != "Overall":
                st.markdown(f"**Trading Days:** {total_days}")

            ""  # Space

            st.markdown(f"**Holdings:** {len(corr_matrix)}")

            ""  # Space

            st.markdown("**Correlation (Unweighted)**")
            st.markdown(f"Mean: **{stats['mean']:.3f}**")
            st.markdown(f"Median: **{stats['median']:.3f}**")

            ""  # Space

            st.markdown("**Correlation (Weighted)**")
            st.markdown(f"Mean: **{stats['weighted_mean']:.3f}**")
            st.markdown(f"Median: **{stats['weighted_median']:.3f}**")

    # Right panel: Heatmap
    with cols[1]:
        heatmap_card = st.container(border=True)
        with heatmap_card:
            # Clean ticker names for display
            clean_labels = [t.split()[0] if isinstance(t, str) else t for t in corr_matrix.columns]
            n_tickers = len(clean_labels)
            is_large_matrix = n_tickers > 50

            # Create heatmap - adjust for matrix size
            heatmap_kwargs = dict(
                z=corr_matrix.values,
                x=clean_labels,
                y=clean_labels,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                hovertemplate='%{x} - %{y}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(
                    title="Correlation",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                )
            )

            # Only show text in cells for small matrices
            if not is_large_matrix:
                heatmap_kwargs['text'] = np.round(corr_matrix.values, 2)
                heatmap_kwargs['texttemplate'] = '%{text}'
                heatmap_kwargs['textfont'] = {"size": 8}

            fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

            corr_type_label = "Weighted" if use_weighted_corr else "Unweighted"
            mode_label = f" - {correlation_mode}" if correlation_mode != "Overall" else ""

            # Adjust layout for matrix size - show ALL tickers, adapt font/height
            if n_tickers <= 30:
                tick_font_size = 10
                chart_height = 700
            elif n_tickers <= 50:
                tick_font_size = 8
                chart_height = 800
            elif n_tickers <= 80:
                tick_font_size = 6
                chart_height = 1000
            elif n_tickers <= 120:
                tick_font_size = 5
                chart_height = 1200
            else:
                tick_font_size = 4
                chart_height = max(1400, n_tickers * 10)

            fig.update_layout(
                title=f"{selected_etf} Holdings Correlation Matrix ({n_tickers} tickers, {corr_type_label}{mode_label})",
                xaxis_title="",
                yaxis_title="",
                height=chart_height,
                xaxis=dict(tickangle=45, side='bottom', dtick=1, tickfont=dict(size=tick_font_size)),
                yaxis=dict(autorange='reversed', dtick=1, tickfont=dict(size=tick_font_size)),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig, use_container_width=True)

            if correlation_mode == "Drawdown":
                st.markdown("<small>*Drawdown correlation: calculated using returns only during drawdown periods (peak → trough). Shows how holdings correlate when ETF is declining.*</small>", unsafe_allow_html=True)
            elif correlation_mode == "Recovery":
                st.markdown("<small>*Recovery correlation: calculated using returns only during recovery periods (trough → next peak). Shows how holdings correlate when ETF is recovering.*</small>", unsafe_allow_html=True)
            elif use_weighted_corr:
                st.markdown("<small>*Weighted correlation: each day's weight = w_A × w_B (product of both stocks' portfolio weights). Days when positions are larger contribute more to the correlation.*</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>*Holdings with less than 50% price data in the lookback period are excluded from the matrix.*</small>", unsafe_allow_html=True)

            ""  # Space

            # Highest and Lowest Correlations
            corr_cols = st.columns(2)
            with corr_cols[0]:
                st.markdown("##### Highest Correlations")
                for t1, t2, corr in stats['highest_pairs']:
                    st.markdown(f"{t1} - {t2}: **{corr:.3f}**")

            with corr_cols[1]:
                st.markdown("##### Lowest Correlations")
                for t1, t2, corr in stats['lowest_pairs']:
                    st.markdown(f"{t1} - {t2}: **{corr:.3f}**")

    ""  # Space

    # Correlation Matrix Data Table & Download
    st.subheader("Correlation Matrix Data")

    data_card = st.container(border=True)
    with data_card:
        # Clean column names for display
        display_corr = corr_matrix.copy()
        display_corr.columns = [c.split()[0] if isinstance(c, str) else c for c in display_corr.columns]
        display_corr.index = [c.split()[0] if isinstance(c, str) else c for c in display_corr.index]

        # Display table
        st.dataframe(
            display_corr.round(3),
            use_container_width=True,
            height=400
        )

        ""  # Space

        # Download button
        csv_data = display_corr.to_csv()
        st.download_button(
            label="Download Correlation Matrix (CSV)",
            data=csv_data,
            file_name=f"{selected_etf}_correlation_matrix_{correlation_mode.lower()}.csv",
            mime="text/csv"
        )

    ""  # Space

    # Correlation Time Series (only for Overall mode)
    if correlation_mode == "Overall":
        st.subheader("Correlation Time Series")

        ts_card = st.container(border=True)
        with ts_card:
            if len(rolling_corr) > 0:
                # Filter by period if needed
                rolling_corr_filtered = filter_by_period(rolling_corr, start_date, end_date)

                if len(rolling_corr_filtered) > 0:
                    fig_ts = go.Figure()

                    # Weighted mean correlation (primary)
                    fig_ts.add_trace(go.Scatter(
                        x=rolling_corr_filtered['Date'],
                        y=rolling_corr_filtered['weighted_mean_corr'],
                        mode='lines',
                        name='Weighted Mean',
                        line=dict(color='steelblue', width=2),
                        hovertemplate='<b>Weighted Mean</b><br>Date (x): %{x|%Y-%m-%d}<br>Correlation (y): %{y:.4f}<extra></extra>'
                    ))

                    # Unweighted mean correlation
                    fig_ts.add_trace(go.Scatter(
                        x=rolling_corr_filtered['Date'],
                        y=rolling_corr_filtered['mean_corr'],
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
                    st.warning(f"No rolling correlation data available for the selected period.")
            else:
                st.warning(f"No precomputed rolling correlations available. Run `python convert_to_parquet.py` to generate.")

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

        mode_suffix = f" - {correlation_mode}" if correlation_mode != "Overall" else ""
        fig_hist.update_layout(
            title=f"Distribution of Pairwise Correlations ({len(correlations)} pairs){mode_suffix}",
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

    # Correlation vs Performance Analysis (only for Overall mode)
    if correlation_mode == "Overall":
        st.subheader("Correlation vs Performance Analysis")

        perf_card = st.container(border=True)
        with perf_card:
            # Load ETF prices
            etf_prices = load_etf_prices(selected_etf)

            if len(etf_prices) > 0 and len(rolling_corr) > 0:
                # Filter rolling correlations to analysis period
                rolling_corr_filtered = filter_by_period(rolling_corr, start_date, end_date)

                # Calculate ETF returns
                etf_prices = etf_prices.copy()
                etf_prices['Return'] = etf_prices['Close'].pct_change()

                # Merge correlation data with ETF returns
                corr_perf = pd.merge(
                    rolling_corr_filtered,
                    etf_prices[['Date', 'Close', 'Return']],
                    on='Date',
                    how='inner'
                )

                if len(corr_perf) > 0:
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
                else:
                    st.warning("Could not merge correlation and price data.")
            else:
                st.warning(f"No price data available for {selected_etf}")

else:
    # More specific error messages for debugging
    if correlation_mode == "Overall":
        st.warning(f"No precomputed correlation data for {selected_etf}. Run `python convert_to_parquet.py` to generate.")
        # Show diagnostic info
        with st.expander("Diagnostic Info"):
            corr_path = ARK_PRECOMPUTED_DIR / f'{selected_etf}_correlation_matrix_{lookback_days}d.parquet'
            st.code(f"Expected file: {corr_path}")
            st.code(f"File exists: {corr_path.exists()}")
            dd_path = ARK_PRECOMPUTED_DIR / f'{selected_etf}_etf_drawdowns.parquet'
            st.code(f"Drawdowns file: {dd_path}")
            st.code(f"Drawdowns exists: {dd_path.exists()}")
    elif correlation_mode == "Drawdown":
        if len(drawdown_periods) == 0:
            st.warning(f"No drawdown periods found for {selected_etf}. Ensure ETF drawdowns are precomputed.")
        else:
            st.warning(f"Could not calculate drawdown correlations for {selected_etf}. Check that holdings have price data.")
        with st.expander("Diagnostic Info"):
            st.code(f"Drawdown periods: {len(drawdown_periods)}")
            st.code(f"Returns shape: {returns.shape if len(returns) > 0 else 'empty'}")
            st.code(f"Filtered returns shape: {filtered_returns.shape if len(filtered_returns) > 0 else 'empty'}")
            if len(returns) > 0:
                st.code(f"Returns columns (first 5): {list(returns.columns[:5])}")
                st.code(f"Returns date range: {returns.index.min()} to {returns.index.max()}")
            if len(drawdown_periods) > 0:
                st.code(f"First drawdown period: {drawdown_periods[0]}")
    else:  # Recovery
        if len(recovery_periods) == 0:
            st.warning(f"No recovery periods found for {selected_etf}. Ensure ETF drawdowns are precomputed.")
        else:
            st.warning(f"Could not calculate recovery correlations for {selected_etf}. Check that holdings have price data.")
        with st.expander("Diagnostic Info"):
            st.code(f"Recovery periods: {len(recovery_periods)}")
            st.code(f"Returns shape: {returns.shape if len(returns) > 0 else 'empty'}")
            st.code(f"Filtered returns shape: {filtered_returns.shape if len(filtered_returns) > 0 else 'empty'}")
            if len(returns) > 0:
                st.code(f"Returns columns (first 5): {list(returns.columns[:5])}")
                st.code(f"Returns date range: {returns.index.min()} to {returns.index.max()}")
            if len(recovery_periods) > 0:
                st.code(f"First recovery period: {recovery_periods[0]}")
