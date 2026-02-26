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

from config import ARK_ETFS
from precomputed_loader import (
    load_correlation_matrix,
    load_weighted_correlation_matrix,
    load_correlation_returns,
    load_current_weights,
    load_etf_drawdowns,
    check_precomputed_exists,
    ARK_PRECOMPUTED_DIR
)
from data_loader import load_ark_holdings, get_ark_files_hash
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

period_key = get_current_period()
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
    returns = prices.pct_change(fill_method=None)

    return returns


def calculate_correlation_from_returns(returns_df, weights_df=None):
    """Calculate correlation matrix from returns DataFrame.

    Args:
        returns_df: DataFrame with Date column (or index) and ticker columns
        weights_df: Not used for matrix calculation (kept for API compatibility)
                   Weighting is now only applied in statistics calculation

    Returns:
        tuple: (correlation matrix DataFrame, list of excluded ticker names)
    """
    if len(returns_df) < 10:  # Need minimum data points
        return pd.DataFrame(), []

    # Drop Date column if present and calculate correlations
    ticker_cols = [col for col in returns_df.columns if col not in ['Date', 'index']]
    if len(ticker_cols) == 0:
        return pd.DataFrame(), []

    returns_only = returns_df[ticker_cols]

    # Filter columns that have at least some data (at least 1 non-NaN value)
    valid_cols = returns_only.columns[returns_only.notna().sum() > 0]
    if len(valid_cols) < 2:
        return pd.DataFrame(), []

    returns_filtered = returns_only[valid_cols]
    all_tickers = set(t.split()[0] for t in returns_filtered.columns)

    # Always calculate unweighted correlation matrix
    # Weighting is applied only in statistics (get_correlation_stats)
    # Use min_periods=20 to require at least 20 overlapping days for each pair
    corr_matrix = returns_filtered.corr(min_periods=20)

    # Remove stocks that have NO valid correlation with any other stock
    # (all off-diagonal values are NaN - less than 20 overlapping days with ALL other stocks)
    has_valid_corr = (corr_matrix.notna().sum() > 1)  # >1 because diagonal is always 1.0
    corr_matrix = corr_matrix.loc[has_valid_corr, has_valid_corr]

    # Track excluded tickers
    included_tickers = set(t.split()[0] for t in corr_matrix.columns)
    excluded_tickers = sorted(all_tickers - included_tickers)

    return corr_matrix, excluded_tickers


def calculate_average_pair_weights(weight_matrix, tickers):
    """Calculate average daily pair weights for weighted mean correlation (vectorized).

    Instead of using single-day weights (e.g., at peak or period end), this calculates
    the average of daily pair weights over all overlapping days. This gives more
    representative weights for stocks with varying presence in the portfolio.

    Args:
        weight_matrix: DataFrame (Date x Ticker) with daily weights
        tickers: List of tickers to calculate weights for (must match correlation matrix columns)

    Returns:
        pair_weights: 1D array of upper triangle pair weights (length = n*(n-1)/2)
    """
    n = len(tickers)
    triu_i, triu_j = np.triu_indices(n, k=1)

    # Align weight_matrix to tickers, fill missing with 0
    wm = weight_matrix.reindex(columns=tickers).fillna(0).values  # Shape: (T, n)

    # Vectorized calculation
    w_i = wm[:, triu_i]  # Shape: (T, num_pairs)
    w_j = wm[:, triu_j]  # Shape: (T, num_pairs)

    pair_products = w_i * w_j
    valid_mask = (w_i > 0) & (w_j > 0)
    valid_counts = valid_mask.sum(axis=0)

    masked_products = np.where(valid_mask, pair_products, 0.0)
    pair_sums = masked_products.sum(axis=0)

    # Average (avoid division by zero warning)
    pair_weights = np.zeros_like(pair_sums)
    nonzero_mask = valid_counts > 0
    pair_weights[nonzero_mask] = pair_sums[nonzero_mask] / valid_counts[nonzero_mask]

    return pair_weights


def get_correlation_stats(corr_matrix, weight_matrix=None):
    """Calculate summary statistics for correlation matrix (vectorized)

    Args:
        corr_matrix: Correlation matrix DataFrame
        weight_matrix: DataFrame (Date x Ticker) with daily weights for weighted stats.
                       If None, weighted stats equal unweighted stats.
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

    # Calculate weighted correlation using average daily pair weights
    if weight_matrix is not None and len(weight_matrix) > 0:
        pair_weights = calculate_average_pair_weights(weight_matrix, corr_matrix.columns.tolist())

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
        t1, t2 = tickers_clean[i], tickers_clean[j]
        # Skip pairs where cleaned ticker names are identical (duplicates with different suffixes)
        if t1 == t2:
            return None
        return (t1, t2, corr_values[idx])

    # Get pairs, filtering out None (same-ticker pairs)
    highest_pairs = [p for p in (get_pair(idx) for idx in sorted_indices[::-1]) if p is not None][:5]
    lowest_pairs = [p for p in (get_pair(idx) for idx in sorted_indices) if p is not None][:5]

    stats['highest_pairs'] = highest_pairs
    stats['lowest_pairs'] = lowest_pairs

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
            label_visibility="collapsed",
            key="etf_selector"
        )

        ""  # Space

        st.markdown("##### Correlation Mode")
        # Use session state to persist selection across reruns (e.g., period switch)
        if "correlation_mode_value" not in st.session_state:
            st.session_state.correlation_mode_value = "Overall"
        correlation_mode = st.pills(
            "Mode",
            options=["Overall", "Drawdown", "Recovery"],
            default=st.session_state.correlation_mode_value,
            label_visibility="collapsed",
            key="correlation_mode_selector"
        )
        if correlation_mode is None:
            correlation_mode = "Overall"
        st.session_state.correlation_mode_value = correlation_mode

        # Lookback Period only shown for Overall mode
        if correlation_mode == "Overall":
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
                key="lookback_selector"
            )
            if selected_lookback is None:
                selected_lookback = "120 Days"
            lookback_days = lookback_options[selected_lookback]
        else:
            lookback_days = 120  # Default for non-Overall modes (not used)

        ""  # Space

        st.markdown("##### Correlation Type")
        use_weighted_corr = st.toggle("Weighted Correlation", value=False)

MONEY_MARKET_PREFIXES = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']


def _filter_non_stocks(holdings):
    """Filter out currency and money market tickers"""
    result = holdings.copy()
    if 'Bloomberg Name' in result.columns:
        result = result[~result['Bloomberg Name'].str.contains('curncy', case=False, na=False)]
    ticker_symbols = result['Ticker'].str.split().str[0]
    is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in MONEY_MARKET_PREFIXES) if pd.notna(x) else False)
    result = result[~is_mm]
    return result


def is_non_stock_ticker(ticker):
    """Check if a ticker is a money market fund or cash instrument"""
    ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
    return any(ticker_clean.startswith(p) for p in MONEY_MARKET_PREFIXES)


@st.cache_data
def load_weight_matrix(etf, period_start, period_end):
    """Load weight matrix (Date x Ticker) for the given period.

    Returns DataFrame with Date as index and Tickers as columns, containing daily weights.
    """
    files_hash = get_ark_files_hash()
    holdings = load_ark_holdings(files_hash, etf)

    if len(holdings) == 0:
        return pd.DataFrame()

    # Filter non-stocks
    holdings_filtered = _filter_non_stocks(holdings)

    # Filter to period
    holdings_period = holdings_filtered[
        (holdings_filtered['Date'] >= period_start) &
        (holdings_filtered['Date'] <= period_end)
    ].copy()

    if len(holdings_period) == 0:
        return pd.DataFrame()

    # Pivot to get weight matrix (Date x Ticker)
    weight_matrix = holdings_period.pivot_table(
        index='Date',
        columns='Ticker',
        values='Weight',
        aggfunc='first'
    )

    # Forward fill weights (holdings may not change daily)
    weight_matrix = weight_matrix.ffill()

    return weight_matrix


# Load data based on correlation mode
with st.spinner("Loading correlations..."):
    # Load ETF drawdowns for period-based calculations
    etf_drawdowns = load_etf_drawdowns(selected_etf)
    drawdown_periods, recovery_periods = get_period_dates(etf_drawdowns)

    # Load current weights first (needed for filtering)
    current_weights = load_current_weights(selected_etf, period_key)
    current_tickers = set(current_weights['Ticker'].tolist()) if len(current_weights) > 0 else set()

    # Load returns - use full historical data for Drawdown/Recovery modes
    if correlation_mode in ["Drawdown", "Recovery"]:
        returns = calculate_holdings_returns(selected_etf)  # Full history from holdings
    else:
        returns = load_correlation_returns(selected_etf, period_key, lookback_days)

    # Load weight matrix for the analysis period (for weighted stats)
    # For Overall mode: use lookback period ending at period_end
    # For Drawdown/Recovery: use full historical data
    if correlation_mode == "Overall":
        period_start_for_weights = end_date - pd.Timedelta(days=lookback_days)
        weight_matrix = load_weight_matrix(selected_etf, period_start_for_weights, end_date)
    else:
        # For Drawdown/Recovery, load full historical weight matrix
        files_hash = get_ark_files_hash()
        holdings = load_ark_holdings(files_hash, selected_etf)
        if len(holdings) > 0:
            weight_matrix = load_weight_matrix(selected_etf, holdings['Date'].min(), holdings['Date'].max())
        else:
            weight_matrix = pd.DataFrame()

    # Calculate correlation matrix based on mode
    excluded_tickers = []  # Track excluded tickers for caption
    if correlation_mode == "Overall":
        # Use precomputed correlation matrices
        if use_weighted_corr:
            corr_matrix = load_weighted_correlation_matrix(selected_etf, period_key, lookback_days)
        else:
            corr_matrix = load_correlation_matrix(selected_etf, period_key, lookback_days)
        corr_matrix_unweighted = load_correlation_matrix(selected_etf, period_key, lookback_days)
        period_info = f"Full {lookback_days}-day period"
        num_periods = 1
        total_days = lookback_days
        # For precomputed data, determine excluded tickers by comparing returns
        # (all tickers before filtering) with correlation matrix columns (filtered)
        if len(corr_matrix) > 0 and len(returns) > 0:
            all_return_tickers = set(t.split()[0] for t in returns.columns)
            included = set(t.split()[0] for t in corr_matrix.columns)
            excluded_raw = all_return_tickers - included
            excluded_tickers = sorted([t for t in excluded_raw if not is_non_stock_ticker(t)])

    elif correlation_mode == "Drawdown":
        # Filter returns to drawdown periods only
        filtered_returns = pd.DataFrame()
        excluded_tickers = []
        if len(returns) > 0 and len(drawdown_periods) > 0:
            filtered_returns = filter_returns_by_periods(returns, drawdown_periods)
            corr_matrix, excluded_tickers = calculate_correlation_from_returns(filtered_returns)
            total_days = len(filtered_returns)
            # Filter weight_matrix to drawdown periods for weighted stats
            if len(weight_matrix) > 0:
                weight_matrix = filter_returns_by_periods(weight_matrix.reset_index(), drawdown_periods)
                if len(weight_matrix) > 0 and 'Date' in weight_matrix.columns:
                    weight_matrix = weight_matrix.set_index('Date')
        else:
            corr_matrix = pd.DataFrame()
            total_days = 0
        period_info = f"{len(drawdown_periods)} drawdown periods"
        num_periods = len(drawdown_periods)

    else:  # Recovery
        # Filter returns to recovery periods only
        filtered_returns = pd.DataFrame()
        excluded_tickers = []
        if len(returns) > 0 and len(recovery_periods) > 0:
            filtered_returns = filter_returns_by_periods(returns, recovery_periods)
            corr_matrix, excluded_tickers = calculate_correlation_from_returns(filtered_returns)
            total_days = len(filtered_returns)
            # Filter weight_matrix to recovery periods for weighted stats
            if len(weight_matrix) > 0:
                weight_matrix = filter_returns_by_periods(weight_matrix.reset_index(), recovery_periods)
                if len(weight_matrix) > 0 and 'Date' in weight_matrix.columns:
                    weight_matrix = weight_matrix.set_index('Date')
        else:
            corr_matrix = pd.DataFrame()
            total_days = 0
        period_info = f"{len(recovery_periods)} recovery periods"
        num_periods = len(recovery_periods)

    # Rolling correlations will be loaded later with user-selected window

if corr_matrix is not None and len(corr_matrix) > 0:
    # Get statistics using weight_matrix for average daily pair weights
    stats = get_correlation_stats(corr_matrix, weight_matrix)

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
            # Filter out tickers with no valid correlations (all NaN off-diagonal)
            has_valid = (corr_matrix.notna().sum() > 1)  # >1 because diagonal may be 1.0
            if not has_valid.all():
                newly_excluded = [t.split()[0] for t in has_valid[~has_valid].index]
                newly_excluded = [t for t in newly_excluded if not is_non_stock_ticker(t)]
                excluded_tickers = sorted(set(excluded_tickers) | set(newly_excluded))
                filtered_tickers = has_valid[has_valid].index
                corr_matrix = corr_matrix.loc[filtered_tickers, filtered_tickers]

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

            st.plotly_chart(fig, width='stretch')

            # Show excluded tickers caption if any were excluded
            if excluded_tickers:
                excluded_str = ', '.join(excluded_tickers)
                st.markdown(f"<small>*Excluded (less than 20 overlapping days with all other stocks): {excluded_str}*</small>", unsafe_allow_html=True)

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

    # Correlation Matrix Data Table & Download (at bottom of page)
    st.subheader("Data Download")

    # Show current selections
    weight_label = "Weighted" if use_weighted_corr else "Unweighted"
    st.markdown(f"**ETF:** {selected_etf} | **Mode:** {correlation_mode} | **Type:** {weight_label}")

    ""  # Space

    # Create tabs for different data views
    data_tabs = st.tabs(["Correlation Matrix", "Pairwise Rankings"])

    # Tab 1: Correlation Matrix
    with data_tabs[0]:
        matrix_card = st.container(border=True)
        with matrix_card:
            # Clean column names for display
            display_corr = corr_matrix.copy()
            display_corr.columns = [c.split()[0] if isinstance(c, str) else c for c in display_corr.columns]
            display_corr.index = [c.split()[0] if isinstance(c, str) else c for c in display_corr.index]

            st.markdown(f"**{len(display_corr)} x {len(display_corr)} correlation matrix**")

            # Display table
            st.dataframe(
                display_corr.round(3),
                width='stretch',
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

    # Tab 2: Pairwise Rankings
    with data_tabs[1]:
        rankings_card = st.container(border=True)
        with rankings_card:
            # Build pairwise rankings DataFrame
            n = len(corr_matrix.columns)
            tickers = corr_matrix.columns.tolist()
            tickers_clean = [t.split()[0] if isinstance(t, str) else t for t in tickers]

            # Get all pairwise correlations (skip pairs where cleaned ticker names are identical)
            pairs_data = []
            for i in range(n):
                for j in range(i + 1, n):
                    t1, t2 = tickers_clean[i], tickers_clean[j]
                    if t1 == t2:  # Skip duplicate tickers with different suffixes
                        continue
                    corr_val = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_val):
                        pairs_data.append({
                            'Ticker 1': t1,
                            'Ticker 2': t2,
                            'Correlation': round(corr_val, 4)
                        })

            pairs_df = pd.DataFrame(pairs_data)

            if len(pairs_df) > 0:
                # Sort by correlation (descending)
                pairs_df = pairs_df.sort_values('Correlation', ascending=False).reset_index(drop=True)
                pairs_df.index = pairs_df.index + 1  # Start index from 1
                pairs_df.index.name = 'Rank'

                st.markdown(f"**{len(pairs_df)} pairwise correlations (sorted by correlation)**")

                # Display table
                st.dataframe(
                    pairs_df,
                    width='stretch',
                    height=400
                )

                ""  # Space

                # Download button
                csv_rankings = pairs_df.to_csv()
                st.download_button(
                    label="Download Pairwise Rankings (CSV)",
                    data=csv_rankings,
                    file_name=f"{selected_etf}_pairwise_rankings_{correlation_mode.lower()}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No pairwise correlations available.")

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
