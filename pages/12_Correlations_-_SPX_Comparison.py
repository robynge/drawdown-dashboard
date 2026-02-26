"""Correlations - S&P 500 Comparison Page"""
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
    load_sp500_correlation_matrix,
    load_current_weights,
    check_precomputed_exists
)
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector

st.set_page_config(
    page_title="Correlations - SPX Comparison",
    page_icon="📊",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Correlations - S&P 500 Comparison

Compare ARK ETF correlation structure with S&P 500 Top 50 holdings.
"""

period_key = get_current_period()
st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")


def get_correlation_stats(corr_matrix):
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
        'max': np.max(correlations),
        'n_pairs': len(correlations),
        'n_holdings': n
    }

    # Find highest and lowest correlation pairs (vectorized)
    tickers = corr_matrix.columns.tolist()
    tickers_clean = [t.split()[0] if isinstance(t, str) else t for t in tickers]

    valid_indices = np.where(valid_mask)[0]
    sorted_indices = valid_indices[np.argsort(corr_values[valid_mask])]

    def get_pair(idx):
        i, j = triu_i[idx], triu_j[idx]
        t1, t2 = tickers_clean[i], tickers_clean[j]
        # Skip pairs where cleaned ticker names are identical
        if t1 == t2:
            return None
        return (t1, t2, corr_values[idx])

    # Get pairs, filtering out None (same-ticker pairs)
    highest_pairs = [p for p in (get_pair(idx) for idx in sorted_indices[::-1]) if p is not None][:5]
    lowest_pairs = [p for p in (get_pair(idx) for idx in sorted_indices) if p is not None][:5]

    stats['highest_pairs'] = highest_pairs
    stats['lowest_pairs'] = lowest_pairs

    return stats


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

        "" # Space

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

        "" # Space

        st.markdown("##### Correlation Type")
        st.markdown("Unweighted (Equal)")
        st.caption("All pairs weighted equally")

# Load correlation matrices
ark_corr = load_correlation_matrix(selected_etf, period_key, lookback_days)
sp500_corr = load_sp500_correlation_matrix(lookback_days, period_key=period_key)

# Money market fund prefixes to filter from excluded list
MONEY_MARKET_PREFIXES = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']

def is_non_stock_ticker(ticker):
    """Check if a ticker is a money market fund or cash instrument"""
    ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
    return any(ticker_clean.startswith(p) for p in MONEY_MARKET_PREFIXES)

# Load current weights to determine excluded tickers
current_weights = load_current_weights(selected_etf, period_key)
current_tickers = set(current_weights['Ticker'].tolist()) if len(current_weights) > 0 else set()

# Determine excluded tickers for ARK (filter out money market funds)
ark_excluded_tickers = []
if ark_corr is not None and len(ark_corr) > 0 and len(current_tickers) > 0:
    included = set(t.split()[0] for t in ark_corr.columns)
    current_clean = set(t.split()[0] for t in current_tickers)
    excluded_raw = current_clean - included
    ark_excluded_tickers = sorted([t for t in excluded_raw if not is_non_stock_ticker(t)])

# Get ARK stats
ark_stats = get_correlation_stats(ark_corr) if ark_corr is not None and len(ark_corr) > 0 else None

# Get S&P 500 stats
sp500_stats = None
if len(sp500_corr) > 0:
    sp500_stats = get_correlation_stats(sp500_corr)

# Summary Statistics in left panel
with cols[0]:
    "" # Space

    if sp500_stats and ark_stats:
        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### S&P 500 Top 50")

            st.markdown(f"**Holdings:** {sp500_stats['n_holdings']}")
            st.markdown(f"**Pairs:** {sp500_stats['n_pairs']}")

            "" # Space

            st.markdown("**Correlation**")
            st.markdown(f"Mean: **{sp500_stats['mean']:.3f}**")
            st.markdown(f"Median: **{sp500_stats['median']:.3f}**")
            st.markdown(f"Std: **{sp500_stats['std']:.3f}**")
            st.markdown(f"Range: **{sp500_stats['min']:.3f}** to **{sp500_stats['max']:.3f}**")

            "" # Space

            # Comparison with selected ARK ETF
            st.markdown(f"**vs {selected_etf}**")
            delta_mean = sp500_stats['mean'] - ark_stats['mean']
            delta_median = sp500_stats['median'] - ark_stats['median']
            st.markdown(f"Mean Δ: **{delta_mean:+.3f}**")
            st.markdown(f"Median Δ: **{delta_median:+.3f}**")

# Right panel: Charts
with cols[1]:
    st.markdown("""
    **S&P 500 Top 50** represents a diversified large-cap portfolio for comparison.
    Lower average correlation indicates better diversification potential.
    """)

    "" # Space

    if len(sp500_corr) > 0:
        # Side by side comparison metrics
        if sp500_stats and ark_stats:
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(f"{selected_etf} Mean ρ", f"{ark_stats['mean']:.3f}")
            with metric_cols[1]:
                st.metric("S&P 500 Mean ρ", f"{sp500_stats['mean']:.3f}")
            with metric_cols[2]:
                delta = sp500_stats['mean'] - ark_stats['mean']
                st.metric("Difference", f"{delta:+.3f}",
                          delta="SPX lower" if delta < 0 else "ARK lower" if delta > 0 else "Equal",
                          delta_color="inverse" if delta > 0 else "normal")
            with metric_cols[3]:
                st.metric("Holdings", f"{ark_stats['n_holdings']} vs {sp500_stats['n_holdings']}")

        "" # Space

        # S&P 500 Heatmap
        st.markdown("#### S&P 500 Top 50 Correlation Matrix")

        heatmap_card = st.container(border=True)
        with heatmap_card:
            # Filter out tickers with no valid correlations (all NaN off-diagonal)
            has_valid = (sp500_corr.notna().sum() > 1)
            if not has_valid.all():
                filtered_sp = has_valid[has_valid].index
                sp500_corr = sp500_corr.loc[filtered_sp, filtered_sp]

            fig_sp500 = go.Figure(data=go.Heatmap(
                z=sp500_corr.values,
                x=sp500_corr.columns.tolist(),
                y=sp500_corr.columns.tolist(),
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(sp500_corr.values, 2),
                texttemplate='%{text}',
                textfont={"size": 6},
                hovertemplate='%{x} - %{y}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(
                    title="Correlation",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
                )
            ))

            fig_sp500.update_layout(
                title=f"S&P 500 Top 50 Correlation Matrix ({selected_lookback})",
                height=750,
                xaxis=dict(tickangle=45, side='bottom', dtick=1, tickfont=dict(size=8)),
                yaxis=dict(autorange='reversed', dtick=1, tickfont=dict(size=8)),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            st.plotly_chart(fig_sp500, width='stretch')

            st.markdown("<small>*S&P 500 Top 50 by market cap. Lower average correlation = better diversification potential.*</small>", unsafe_allow_html=True)

            "" # Space

            # Highest and Lowest Correlations
            if sp500_stats:
                corr_pair_cols = st.columns(2)
                with corr_pair_cols[0]:
                    st.markdown("##### Highest Correlations")
                    for t1, t2, corr in sp500_stats['highest_pairs']:
                        st.markdown(f"{t1} - {t2}: **{corr:.3f}**")

                with corr_pair_cols[1]:
                    st.markdown("##### Lowest Correlations")
                    for t1, t2, corr in sp500_stats['lowest_pairs']:
                        st.markdown(f"{t1} - {t2}: **{corr:.3f}**")

        "" # Space

        # Distribution comparison
        st.markdown("#### Correlation Distribution Comparison")

        dist_card = st.container(border=True)
        with dist_card:
            fig_dist = go.Figure()

            # S&P 500 distribution
            n_sp = len(sp500_corr.columns)
            triu_i_sp, triu_j_sp = np.triu_indices(n_sp, k=1)
            sp500_corr_values = sp500_corr.values[triu_i_sp, triu_j_sp]
            sp500_corr_values = sp500_corr_values[~np.isnan(sp500_corr_values)]

            fig_dist.add_trace(go.Histogram(
                x=sp500_corr_values,
                nbinsx=30,
                marker_color='steelblue',
                opacity=0.6,
                name='S&P 500 Top 50',
                hovertemplate='<b>S&P 500</b><br>Correlation (x): %{x:.4f}<br>Count (y): %{y}<extra></extra>'
            ))

            # ARK distribution
            if ark_corr is not None and len(ark_corr) > 0:
                n_ark = len(ark_corr.columns)
                triu_i_ark, triu_j_ark = np.triu_indices(n_ark, k=1)
                ark_corr_values = ark_corr.values[triu_i_ark, triu_j_ark]
                ark_corr_values = ark_corr_values[~np.isnan(ark_corr_values)]

                fig_dist.add_trace(go.Histogram(
                    x=ark_corr_values,
                    nbinsx=30,
                    marker_color='crimson',
                    opacity=0.6,
                    name=selected_etf,
                    hovertemplate=f'<b>{selected_etf}</b><br>Correlation (x): %{{x:.4f}}<br>Count (y): %{{y}}<extra></extra>'
                ))

            fig_dist.update_layout(
                title=f"Correlation Distribution: S&P 500 vs {selected_etf}",
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

            # Show excluded tickers caption if any were excluded
            if ark_excluded_tickers:
                excluded_str = ', '.join(ark_excluded_tickers)
                st.markdown(f"<small>*{selected_etf} excluded (less than 20 overlapping days with all other stocks): {excluded_str}*</small>", unsafe_allow_html=True)

            # Insight
            if sp500_stats and ark_stats:
                if sp500_stats['mean'] < ark_stats['mean']:
                    st.success(f"S&P 500 Top 50 has **lower average correlation** ({sp500_stats['mean']:.3f}) than {selected_etf} ({ark_stats['mean']:.3f}), indicating better diversification.")
                else:
                    st.warning(f"{selected_etf} has **lower average correlation** ({ark_stats['mean']:.3f}) than S&P 500 Top 50 ({sp500_stats['mean']:.3f}), indicating better diversification.")

    else:
        st.warning("S&P 500 Top 50 correlation data not found. Run `python convert_to_parquet.py` to generate.")
