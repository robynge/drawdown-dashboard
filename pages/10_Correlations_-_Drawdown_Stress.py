"""Correlations - Drawdown Stress Page"""
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
    load_stress_correlations,
    check_precomputed_exists
)
from data_loader import load_etf_prices, load_ark_holdings, get_ark_files_hash
from session_utils import init_session_state, get_current_dates, render_period_selector


def _filter_non_stocks(holdings):
    """Filter out currency and money market tickers"""
    result = holdings.copy()
    if 'Bloomberg Name' in result.columns:
        result = result[~result['Bloomberg Name'].str.contains('curncy', case=False, na=False)]
    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']
    ticker_symbols = result['Ticker'].str.split().str[0]
    is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
    result = result[~is_mm]
    return result


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


def calculate_recovery_correlation(files_hash, etf: str, stress_corr_df: pd.DataFrame) -> dict:
    """Calculate mean correlation during recovery periods.

    Recovery period = from trough of one drawdown to peak of the next drawdown.
    For each recovery period, calculate the mean correlation, then average across all periods.

    Returns:
        dict with keys: unweighted_mean, weighted_mean, excluded_tickers
    """
    result = {
        'unweighted_mean': 0.0,
        'weighted_mean': 0.0,
        'excluded_tickers': []
    }

    if len(stress_corr_df) < 2:
        return result

    # Load full holdings data
    holdings = load_ark_holdings(files_hash, etf)
    if len(holdings) == 0:
        return result

    holdings_filtered = _filter_non_stocks(holdings)

    # Sort drawdowns by trough_date to get chronological order
    sorted_dd = stress_corr_df.sort_values('trough_date').reset_index(drop=True)

    period_unweighted_means = []
    period_weighted_means = []
    all_tickers_seen = set()
    all_tickers_included = set()

    for i in range(len(sorted_dd) - 1):
        # Recovery period: from trough of current drawdown to peak of next drawdown
        recovery_start = sorted_dd.iloc[i]['trough_date']
        recovery_end = sorted_dd.iloc[i + 1]['peak_date']

        # Skip if recovery period is invalid (end before start)
        if recovery_end <= recovery_start:
            continue

        # Filter holdings to recovery period
        period_holdings = holdings_filtered[
            (holdings_filtered['Date'] >= recovery_start) &
            (holdings_filtered['Date'] <= recovery_end)
        ].copy()

        if len(period_holdings) < 10:
            continue

        # Pivot to get price matrix
        price_matrix = period_holdings.pivot_table(
            index='Date',
            columns='Ticker',
            values='Stock_Price',
            aggfunc='first'
        )

        if len(price_matrix.columns) < 2:
            continue

        # Track all tickers seen
        period_tickers = set(t.split()[0] for t in price_matrix.columns)
        all_tickers_seen.update(period_tickers)

        # Calculate daily returns - use iloc[1:] to skip first NaN row, let corr() handle remaining NaNs
        returns = price_matrix.pct_change(fill_method=None).iloc[1:]

        if len(returns) < 10:
            continue

        # Calculate correlation matrix - use min_periods=3 (matching minimum return days filter)
        corr_matrix = returns.corr(min_periods=3)

        # Remove stocks that have NO valid correlation with any other stock
        has_valid_corr = (corr_matrix.notna().sum() > 1)
        valid_tickers = corr_matrix.columns[has_valid_corr]
        corr_matrix = corr_matrix.loc[valid_tickers, valid_tickers]

        # Track included tickers
        included_tickers = set(t.split()[0] for t in corr_matrix.columns)
        all_tickers_included.update(included_tickers)

        # Extract upper triangle (pairwise correlations)
        n = len(corr_matrix.columns)
        if n < 2:
            continue

        triu_i, triu_j = np.triu_indices(n, k=1)
        corr_values = corr_matrix.values[triu_i, triu_j]
        valid_mask = ~np.isnan(corr_values)
        valid_corrs = corr_values[valid_mask]

        if len(valid_corrs) == 0:
            continue

        # Calculate unweighted mean for this period
        period_unweighted_means.append(np.mean(valid_corrs))

        # Calculate weighted mean using average daily pair weights over the recovery period
        weight_matrix = period_holdings.pivot_table(
            index='Date',
            columns='Ticker',
            values='Weight',
            aggfunc='first'
        )
        weight_matrix = weight_matrix.ffill()

        pair_weights = calculate_average_pair_weights(weight_matrix, corr_matrix.columns.tolist())
        valid_weight_mask = valid_mask & (pair_weights > 0)

        if valid_weight_mask.any():
            weighted_mean = np.average(corr_values[valid_weight_mask], weights=pair_weights[valid_weight_mask])
            period_weighted_means.append(weighted_mean)
        else:
            period_weighted_means.append(np.mean(valid_corrs))

    if len(period_unweighted_means) == 0:
        return result

    # Average across all periods (each period weighted equally)
    result['unweighted_mean'] = np.mean(period_unweighted_means)
    result['weighted_mean'] = np.mean(period_weighted_means) if period_weighted_means else result['unweighted_mean']
    result['excluded_tickers'] = sorted(all_tickers_seen - all_tickers_included)

    return result


st.set_page_config(
    page_title="Correlations - Drawdown Stress",
    page_icon="📉",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Correlations - Drawdown Stress

Analyze how portfolio correlations change during drawdown periods vs recovery periods.
"""

st.markdown("**Analysis Period:** Full Historical Data (2021-2026)")
st.caption("Stress correlations are calculated across all top 10 drawdowns in the full historical period. This analysis ignores the period selector.")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")

# Get ETF from session state (default to first)
if 'stress_etf' not in st.session_state:
    st.session_state.stress_etf = ARK_ETFS[0]
selected_etf = st.session_state.stress_etf

# Get weighted toggle from session state (default False)
use_weighted = st.session_state.get('stress_weighted_toggle', False)

# Determine which correlation column to use based on toggle
corr_col = 'weighted_mean_corr' if use_weighted else 'mean_corr'
corr_label = 'Weighted' if use_weighted else 'Unweighted'

# Load data
files_hash = get_ark_files_hash()
stress_corr = load_stress_correlations(selected_etf)

# Calculate recovery correlation (both unweighted and weighted)
recovery_result = {'unweighted_mean': 0.0, 'weighted_mean': 0.0, 'excluded_tickers': []}
if len(stress_corr) >= 2:
    recovery_result = calculate_recovery_correlation(files_hash, selected_etf, stress_corr)

# Select the appropriate recovery correlation based on toggle
recovery_corr = recovery_result['weighted_mean'] if use_weighted else recovery_result['unweighted_mean']
recovery_excluded = recovery_result.get('excluded_tickers', [])

if len(stress_corr) > 0:
    # Check if weighted_mean_corr column exists
    if corr_col not in stress_corr.columns:
        corr_col = 'mean_corr'
        corr_label = 'Unweighted'

    # Calculate statistics using selected correlation column
    stress_mean = stress_corr[corr_col].mean()
    correlation_increase = ((stress_mean / recovery_corr) - 1) * 100 if recovery_corr > 0 else 0

    # === METRICS AT THE TOP (before two-column layout) ===
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Recovery Correlation", f"{recovery_corr:.3f}",
                  help="Average pairwise correlation during recovery periods")

    with metric_cols[1]:
        st.metric("Stress Correlation", f"{stress_mean:.3f}",
                  help=f"Average {corr_label.lower()} correlation during drawdown periods")

    with metric_cols[2]:
        st.metric("Correlation Increase", f"{correlation_increase:+.1f}%",
                  help="How much correlations increase during stress")

    "" # Space

    # Chart title before two-column layout
    st.markdown("#### Correlation by Drawdown Event")

    # === TWO-COLUMN LAYOUT ===
    cols = st.columns([1, 3])

    # Left column: Select ETF + Summary Statistics
    with cols[0]:
        controls_card = st.container(border=True)
        with controls_card:
            st.markdown("##### Select ETF")
            new_etf = st.pills(
                "ETF",
                options=ARK_ETFS,
                default=selected_etf,
                label_visibility="collapsed",
                key="stress_etf_selector"
            )
            if new_etf is None:
                new_etf = selected_etf
            if new_etf != selected_etf:
                st.session_state.stress_etf = new_etf
                st.rerun()

            "" # Space

            st.markdown("##### Correlation Type")
            st.toggle("Weighted Correlation", value=use_weighted, key="stress_weighted_toggle")

        "" # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            st.markdown(f"**Drawdown Events:** {len(stress_corr)}")

            "" # Space

            max_stress = stress_corr[corr_col].max()
            st.markdown(f"**Max Stress Corr:** {max_stress:.3f}")

            "" # Space

            min_stress = stress_corr[corr_col].min()
            st.markdown(f"**Min Stress Corr:** {min_stress:.3f}")

    # Right column: Chart + Table
    with cols[1]:

        chart_card = st.container(border=True)
        with chart_card:
            # Load ETF prices
            etf_prices = load_etf_prices(selected_etf)

            if len(etf_prices) > 0:
                # Use full price history
                price_df = etf_prices.copy()

                # Create figure
                fig_stress = go.Figure()

                # Drawdown colors
                dd_colors = [
                    'rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                    'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                    'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                    'rgba(255, 228, 181, 0.3)'
                ]

                # Add drawdown shaded regions
                for idx, (_, row) in enumerate(stress_corr.iterrows()):
                    fig_stress.add_vrect(
                        x0=row['peak_date'],
                        x1=row['trough_date'],
                        fillcolor=dd_colors[idx % len(dd_colors)],
                        layer="below",
                        line_width=0
                    )

                # Add ETF price line
                fig_stress.add_trace(go.Scatter(
                    x=price_df['Date'],
                    y=price_df['Close'],
                    mode='lines',
                    name=f'{selected_etf} Price',
                    line=dict(color='black', width=2),
                    hoverinfo='skip'
                ))

                # Calculate midpoints and add correlation points
                midpoint_dates = []
                midpoint_corrs = []
                hover_texts = []

                for _, row in stress_corr.iterrows():
                    peak = row['peak_date']
                    trough = row['trough_date']
                    midpoint = peak + (trough - peak) / 2
                    midpoint_dates.append(midpoint)
                    midpoint_corrs.append(row[corr_col])
                    hover_texts.append(
                        f"<b>Drawdown #{int(row['dd_rank'])}</b><br>" +
                        f"Period: {peak.strftime('%Y-%m-%d')} to {trough.strftime('%Y-%m-%d')}<br>" +
                        f"Depth: {row['depth_pct']:.1f}%<br>" +
                        f"Duration: {row['duration_days']} days<br>" +
                        f"{corr_label} Correlation: {row[corr_col]:.3f}"
                    )

                # Sort by date for proper line connection
                sorted_data = sorted(zip(midpoint_dates, midpoint_corrs, hover_texts), key=lambda x: x[0])
                midpoint_dates = [x[0] for x in sorted_data]
                midpoint_corrs = [x[1] for x in sorted_data]
                hover_texts = [x[2] for x in sorted_data]

                # Add stress correlation line on secondary y-axis
                fig_stress.add_trace(go.Scatter(
                    x=midpoint_dates,
                    y=midpoint_corrs,
                    mode='lines+markers',
                    name=f'Stress Correlation ({corr_label})',
                    line=dict(color='red', width=2),
                    marker=dict(size=12, color='red', symbol='circle'),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_texts,
                    yaxis='y2'
                ))

                # Add recovery correlation reference line as a trace (for legend)
                x_range = [price_df['Date'].min(), price_df['Date'].max()]
                fig_stress.add_trace(go.Scatter(
                    x=x_range,
                    y=[recovery_corr, recovery_corr],
                    mode='lines',
                    name=f'Recovery Correlation ({recovery_corr:.3f})',
                    line=dict(color='steelblue', width=2, dash='dash'),
                    yaxis='y2',
                    hoverinfo='skip'
                ))

                fig_stress.update_layout(
                    title=f"{selected_etf} Price & {corr_label} Stress Correlation by Drawdown",
                    height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    xaxis=dict(title="Date", gridcolor='lightgray'),
                    yaxis=dict(title=f"{selected_etf} Price ($)", gridcolor='lightgray'),
                    yaxis2=dict(title=f"{corr_label} Correlation", overlaying='y', side='right', range=[0, 1], showgrid=False)
                )

                st.plotly_chart(fig_stress, width='stretch')

                # Show excluded tickers caption if any were excluded
                if recovery_excluded:
                    excluded_str = ', '.join(recovery_excluded)
                    st.markdown(f"<small>*Recovery correlation excluded (less than 20 overlapping days with all other stocks): {excluded_str}*</small>", unsafe_allow_html=True)
            else:
                st.warning(f"No price data available for {selected_etf}")

    "" # Space

    # Detailed table (full width, outside two-column layout)
    st.markdown("#### Detailed Stress Correlation Data")

    table_card = st.container(border=True)
    with table_card:
        display_df = stress_corr.copy()
        display_df['peak_date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
        display_df['trough_date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')

        # Rename columns based on which correlation type is selected
        rename_map = {
            'dd_rank': 'Rank',
            'peak_date': 'Peak',
            'trough_date': 'Trough',
            'depth_pct': 'Depth %',
            'duration_days': 'Days',
            'num_tickers': 'Tickers',
            'median_corr': 'Median ρ',
            'max_corr': 'Max ρ',
            'min_corr': 'Min ρ'
        }
        # Add the selected correlation column with appropriate name
        rename_map[corr_col] = f'{corr_label} ρ'

        display_df = display_df.rename(columns=rename_map)

        # Format numeric columns
        for col in ['Depth %', f'{corr_label} ρ', 'Median ρ', 'Max ρ', 'Min ρ']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        # Select columns for display
        display_cols = ['Rank', 'Peak', 'Trough', 'Depth %', 'Days', 'Tickers', f'{corr_label} ρ', 'Median ρ', 'Max ρ', 'Min ρ']
        display_cols = [c for c in display_cols if c in display_df.columns]

        st.dataframe(
            display_df[display_cols],
            hide_index=True,
            width='stretch'
        )

else:
    st.warning(f"No stress correlation data for {selected_etf}. Run `python convert_to_parquet.py` to generate.")
