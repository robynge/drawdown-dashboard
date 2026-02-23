"""Correlation Change Statistical Test Page

Uses precomputed correlation returns data for faster loading.
Bootstrap test runs dynamically since it depends on user-selected parameters.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, INPUT_DIR
from data_loader import load_ark_holdings, get_ark_files_hash
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector
from precomputed_loader import load_correlation_returns, load_current_weights, check_precomputed_exists

st.set_page_config(
    page_title="Correlations - Statistical Test",
    page_icon="📊",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Correlations - Statistical Test

Bootstrap permutation test to determine if portfolio correlation has significantly changed.
"""

@st.cache_data
def prepare_returns_data_precomputed(_files_hash, etf, lookback_days, period_key, _start_date, _end_date):
    """Load precomputed returns data for correlation analysis

    Returns:
        returns: DataFrame of daily returns
        dropped_stocks: empty list (filtering already done during precomputation)
    """
    # Try to load precomputed returns
    returns = load_correlation_returns(etf, lookback_days)

    if len(returns) == 0:
        return pd.DataFrame(), []

    # Filter to analysis period
    returns = returns[
        (returns.index >= _start_date) &
        (returns.index <= _end_date)
    ]

    return returns, []


@st.cache_data
def prepare_returns_data(_files_hash, etf, lookback_days, period_key, _start_date, _end_date, _holdings):
    """Prepare returns data for correlation analysis (fallback when no precomputed data)

    Returns:
        returns: DataFrame of daily returns
        dropped_stocks: list of stocks excluded due to incomplete data
    period_key, _start_date, _end_date: Analysis period for filtering
    """
    holdings = _holdings

    # Filter holdings to analysis period first
    holdings = holdings[
        (holdings['Date'] >= _start_date) &
        (holdings['Date'] <= _end_date)
    ].copy()

    if len(holdings) == 0:
        return pd.DataFrame(), []

    # Get current holdings (latest date within analysis period)
    latest_date = holdings['Date'].max()
    current_tickers = holdings[holdings['Date'] == latest_date]['Ticker'].unique()

    # Filter out currency tickers and money market funds
    if 'Bloomberg Name' in holdings.columns:
        currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
        current_tickers = [t for t in current_tickers if t not in currency_tickers]

    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
    current_tickers = [t for t in current_tickers if not any(t.split()[0].startswith(p) for p in money_market_prefixes)]

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

    # Track stocks before filtering
    stocks_before = set(price_matrix.columns)

    # Drop tickers with too many missing values (less than 50% data)
    min_data_points = len(price_matrix) * 0.5
    price_matrix = price_matrix.dropna(axis=1, thresh=int(min_data_points))

    # Calculate daily returns
    returns = price_matrix.pct_change()

    # Drop first row (NaN from pct_change) and any stocks with missing data
    returns = returns.iloc[1:]  # Remove first row
    returns = returns.dropna(axis=1)  # Remove columns (stocks) with any NaN

    # Track which stocks were dropped
    stocks_after = set(returns.columns)
    dropped_stocks = stocks_before - stocks_after
    dropped_stocks = [t.split()[0] for t in dropped_stocks]  # Clean ticker names

    return returns, sorted(dropped_stocks)

def extract_pairwise_correlations(corr_matrix):
    """Extract upper triangle of correlation matrix as a flat array"""
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_vals = corr_matrix.where(mask).values.flatten()
    return corr_vals[~np.isnan(corr_vals)]

@st.cache_data
def bootstrap_permutation_test(_rho_a_tuple, _rho_b_tuple, n_iterations=10000, _seed=42):
    """
    Bootstrap permutation test for difference in mean correlation.

    H0: rho_A and rho_B come from the same distribution
    H1: rho_B is systematically higher than rho_A

    Returns: observed_delta, p_value, null_distribution

    Note: rho_a and rho_b are passed as tuples for caching.
    """
    rho_a = np.array(_rho_a_tuple)
    rho_b = np.array(_rho_b_tuple)

    # Observed difference
    delta_obs = np.mean(rho_b) - np.mean(rho_a)

    # Combine all correlations
    combined = np.concatenate([rho_a, rho_b])
    n_a = len(rho_a)

    # Set seed for reproducibility (allows caching)
    np.random.seed(_seed)

    # Generate null distribution by permutation
    null_deltas = []
    for _ in range(n_iterations):
        # Randomly shuffle and split
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        null_deltas.append(np.mean(perm_b) - np.mean(perm_a))

    null_deltas = np.array(null_deltas)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_deltas) >= np.abs(delta_obs))

    return delta_obs, p_value, null_deltas

# Main section
st.subheader("Test Configuration")

# Layout
config_cols = st.columns([1, 1, 1])

with config_cols[0]:
    selected_etf = st.selectbox("Select ETF", options=ARK_ETFS, index=0)

with config_cols[1]:
    window_size = st.selectbox("Window Size (days)", options=[20, 30], index=0)

with config_cols[2]:
    n_bootstrap = st.selectbox("Bootstrap Iterations", options=[1000, 5000, 10000], index=2)

""  # Space

# Load data
files_hash = get_ark_files_hash()
period_key = get_current_period()

# Need enough data for two non-overlapping windows
lookback_days = 250  # Use 250d to match precomputed returns

with st.spinner("Loading data..."):
    # Try precomputed data first
    if check_precomputed_exists():
        returns, dropped_stocks = prepare_returns_data_precomputed(
            files_hash, selected_etf, lookback_days, period_key, start_date, end_date
        )
    else:
        returns, dropped_stocks = pd.DataFrame(), []

    # Fallback to dynamic calculation if precomputed is empty
    if len(returns) == 0:
        holdings = load_ark_holdings(files_hash, selected_etf)
        returns, dropped_stocks = prepare_returns_data(
            files_hash, selected_etf, lookback_days, period_key, start_date, end_date, holdings
        )

# Show dropped stocks if any
if dropped_stocks:
    st.caption(f"⚠️ Excluded due to incomplete data: {', '.join(dropped_stocks)}")

if len(returns) >= window_size * 2:
    # Define two non-overlapping windows
    # Window B (recent): last window_size days
    # Window A (past): window_size days before Window B

    window_b_end = len(returns)
    window_b_start = window_b_end - window_size
    window_a_end = window_b_start
    window_a_start = window_a_end - window_size

    if window_a_start >= 0:
        # Extract returns for each window
        returns_a = returns.iloc[window_a_start:window_a_end]
        returns_b = returns.iloc[window_b_start:window_b_end]

        # Calculate correlation matrices
        corr_a = returns_a.corr()
        corr_b = returns_b.corr()

        # Use common columns
        common_cols = corr_a.columns.intersection(corr_b.columns)
        corr_a = corr_a.loc[common_cols, common_cols]
        corr_b = corr_b.loc[common_cols, common_cols]

        # Extract pairwise correlations
        rho_a = extract_pairwise_correlations(corr_a)
        rho_b = extract_pairwise_correlations(corr_b)

        n_pairs = len(rho_a)
        n_stocks = len(common_cols)

        # Display window info
        st.subheader("Time Windows")

        window_cols = st.columns(2)

        date_a_start = returns.index[window_a_start].strftime('%Y-%m-%d')
        date_a_end = returns.index[window_a_end - 1].strftime('%Y-%m-%d')
        date_b_start = returns.index[window_b_start].strftime('%Y-%m-%d')
        date_b_end = returns.index[window_b_end - 1].strftime('%Y-%m-%d')

        with window_cols[0]:
            card_a = st.container(border=True)
            with card_a:
                st.markdown(f"**Window A (Past)**")
                st.markdown(f"{date_a_start} to {date_a_end}")
                st.markdown(f"Mean Correlation: **{np.mean(rho_a):.4f}**")
                st.markdown(f"Median Correlation: **{np.median(rho_a):.4f}**")

        with window_cols[1]:
            card_b = st.container(border=True)
            with card_b:
                st.markdown(f"**Window B (Recent)**")
                st.markdown(f"{date_b_start} to {date_b_end}")
                st.markdown(f"Mean Correlation: **{np.mean(rho_b):.4f}**")
                st.markdown(f"Median Correlation: **{np.median(rho_b):.4f}**")

        st.markdown(f"<small>*{n_stocks} stocks, {n_pairs} pairwise correlations per window</small>", unsafe_allow_html=True)

        ""  # Space

        # Run bootstrap test
        st.subheader("Bootstrap Permutation Test")

        with st.spinner(f"Running {n_bootstrap} bootstrap iterations..."):
            # Convert to tuples for caching
            delta_obs, p_value, null_distribution = bootstrap_permutation_test(
                tuple(rho_a), tuple(rho_b), n_bootstrap
            )

        # Display results
        result_cols = st.columns(3)

        with result_cols[0]:
            st.metric("Observed Δ (Mean)", f"{delta_obs:+.4f}")

        with result_cols[1]:
            st.metric("P-Value", f"{p_value:.4f}")

        with result_cols[2]:
            if p_value < 0.01:
                sig_text = "Highly Significant"
                sig_color = "🔴"
            elif p_value < 0.05:
                sig_text = "Significant"
                sig_color = "🟠"
            else:
                sig_text = "Not Significant"
                sig_color = "🟢"
            st.metric("Result", f"{sig_color} {sig_text}")

        ""  # Space

        # Hypothesis interpretation
        interp_card = st.container(border=True)
        with interp_card:
            st.markdown("**Hypothesis Test**")
            st.markdown("- **H₀**: ρ_A and ρ_B come from the same distribution (no change in correlation structure)")
            st.markdown("- **H₁**: ρ_B is systematically different from ρ_A (correlation structure has changed)")

            if p_value < 0.05:
                if delta_obs > 0:
                    st.markdown(f"**Conclusion**: Reject H₀. Portfolio correlation has **increased** significantly (p={p_value:.4f}).")
                else:
                    st.markdown(f"**Conclusion**: Reject H₀. Portfolio correlation has **decreased** significantly (p={p_value:.4f}).")
            else:
                st.markdown(f"**Conclusion**: Fail to reject H₀. No significant change in correlation structure (p={p_value:.4f}).")

        ""  # Space

        # Visualization - two charts side by side
        st.subheader("Visualizations")

        chart_cols = st.columns(2)

        with chart_cols[0]:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=null_distribution,
                nbinsx=50,
                marker_color='steelblue',
                opacity=0.7,
                name='Null Distribution',
                hovertemplate='<b>Null Distribution</b><br>Δ (x): %{x:.4f}<br>Count (y): %{y}<extra></extra>'
            ))

            fig.add_vline(
                x=delta_obs,
                line_dash="dash",
                line_color="red",
                line_width=2
            )

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                name=f'Observed Δ = {delta_obs:+.4f}',
                line=dict(color='red', width=2, dash='dash')
            ))

            fig.update_layout(
                title=f"Null Distribution (n={n_bootstrap})",
                xaxis_title="Δ (Mean Corr Difference)",
                yaxis_title="Count",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='lightgray'),
                yaxis=dict(gridcolor='lightgray')
            )

            st.plotly_chart(fig, width='stretch')

        with chart_cols[1]:
            fig_dist = go.Figure()

            fig_dist.add_trace(go.Histogram(
                x=rho_a,
                nbinsx=30,
                marker_color='steelblue',
                opacity=0.6,
                name=f'Window A ({date_a_start})',
                hovertemplate='<b>Window A</b><br>Correlation (x): %{x:.4f}<br>Count (y): %{y}<extra></extra>'
            ))

            fig_dist.add_trace(go.Histogram(
                x=rho_b,
                nbinsx=30,
                marker_color='crimson',
                opacity=0.6,
                name=f'Window B ({date_b_start})',
                hovertemplate='<b>Window B</b><br>Correlation (x): %{x:.4f}<br>Count (y): %{y}<extra></extra>'
            ))

            fig_dist.update_layout(
                title="Correlation Distributions: A vs B",
                xaxis_title="Pairwise Correlation",
                yaxis_title="Count",
                barmode='overlay',
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='lightgray', range=[-1, 1]),
                yaxis=dict(gridcolor='lightgray')
            )

            st.plotly_chart(fig_dist, width='stretch')

        st.markdown(f"<small>*Left: Under H₀, the observed Δ should fall within the bulk of the null distribution. Right: Comparison of pairwise correlation distributions between the two windows.</small>", unsafe_allow_html=True)

    else:
        st.warning("Not enough historical data for two non-overlapping windows.")
else:
    st.warning(f"Not enough data. Need at least {window_size * 2} trading days, have {len(returns)}.")
