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
    load_sp500_correlation_matrix,
    filter_by_period,
    check_precomputed_exists
)
from data_loader import load_etf_prices
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

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")


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

        st.markdown("##### Correlation Type")
        use_weighted_corr = st.toggle("Weighted Correlation", value=False)

# Load precomputed data
with st.spinner("Loading correlations..."):
    # Load correlation matrices based on toggle
    if use_weighted_corr:
        corr_matrix = load_weighted_correlation_matrix(selected_etf, lookback_days)
    else:
        corr_matrix = load_correlation_matrix(selected_etf, lookback_days)

    # Also load unweighted for stats comparison
    corr_matrix_unweighted = load_correlation_matrix(selected_etf, lookback_days)

    # Load current weights
    current_weights = load_current_weights(selected_etf)

    # Load rolling correlations
    rolling_corr = load_rolling_correlations(selected_etf, lookback_days, rolling_window)

    # Load returns for distribution chart
    returns = load_correlation_returns(selected_etf, lookback_days)

if corr_matrix is not None and len(corr_matrix) > 0:
    # Get statistics
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
            st.markdown(f"Median: **{stats['weighted_median']:.3f}**")

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

    # Correlation Time Series
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

    "" # Space

    # =========================================================================
    # S&P 500 Top 50 Correlation Matrix (Comparison)
    # =========================================================================
    st.subheader("S&P 500 Top 50 Correlation Analysis")

    sp500_cols = st.columns([1, 3])

    with sp500_cols[0]:
        sp500_controls = st.container(border=True)
        with sp500_controls:
            st.markdown("##### Lookback Period")
            sp500_lookback_options = {
                "60 Days": 60,
                "120 Days": 120,
                "250 Days": 250
            }
            sp500_selected_lookback = st.pills(
                "SP500 Lookback",
                options=list(sp500_lookback_options.keys()),
                default="120 Days",
                label_visibility="collapsed",
                key="sp500_lookback"
            )
            sp500_lookback_days = sp500_lookback_options[sp500_selected_lookback]

            "" # Space

            st.markdown("##### Correlation Type")
            st.markdown("Unweighted (Equal)")
            st.caption("All pairs weighted equally")

    # Load S&P 500 correlation matrix
    sp500_corr = load_sp500_correlation_matrix(sp500_lookback_days)

    if len(sp500_corr) > 0:
        # Calculate S&P 500 stats
        n = len(sp500_corr.columns)
        triu_i, triu_j = np.triu_indices(n, k=1)
        sp500_corr_values = sp500_corr.values[triu_i, triu_j]
        valid_sp500_corrs = sp500_corr_values[~np.isnan(sp500_corr_values)]

        sp500_mean = np.mean(valid_sp500_corrs)
        sp500_median = np.median(valid_sp500_corrs)
        sp500_std = np.std(valid_sp500_corrs)
        sp500_min = np.min(valid_sp500_corrs)
        sp500_max = np.max(valid_sp500_corrs)

        # Find highest and lowest pairs
        sorted_indices = np.argsort(valid_sp500_corrs)
        tickers = sp500_corr.columns.tolist()

        def get_sp500_pair(flat_idx):
            # Map back to original indices
            valid_mask = ~np.isnan(sp500_corr_values)
            valid_indices = np.where(valid_mask)[0]
            orig_idx = valid_indices[flat_idx]
            i, j = triu_i[orig_idx], triu_j[orig_idx]
            return (tickers[i], tickers[j], sp500_corr_values[orig_idx])

        sp500_highest = [get_sp500_pair(idx) for idx in sorted_indices[-5:][::-1]]
        sp500_lowest = [get_sp500_pair(idx) for idx in sorted_indices[:5]]

        # Summary Statistics in left panel
        with sp500_cols[0]:
            "" # Space

            sp500_stats_card = st.container(border=True)
            with sp500_stats_card:
                st.markdown("##### Summary Statistics")

                st.markdown(f"**Holdings:** {len(sp500_corr)}")
                st.markdown(f"**Pairs:** {len(valid_sp500_corrs)}")

                "" # Space

                st.markdown("**Correlation**")
                st.markdown(f"Mean: **{sp500_mean:.3f}**")
                st.markdown(f"Median: **{sp500_median:.3f}**")
                st.markdown(f"Std: **{sp500_std:.3f}**")
                st.markdown(f"Range: **{sp500_min:.3f}** to **{sp500_max:.3f}**")

                "" # Space

                # Comparison with selected ARK ETF
                st.markdown(f"**vs {selected_etf}**")
                delta_mean = sp500_mean - stats['mean']
                delta_median = sp500_median - stats['median']
                st.markdown(f"Mean Δ: **{delta_mean:+.3f}**")
                st.markdown(f"Median Δ: **{delta_median:+.3f}**")

    with sp500_cols[1]:
        if len(sp500_corr) > 0:
            sp500_heatmap_card = st.container(border=True)
            with sp500_heatmap_card:
                # S&P 500 Heatmap
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
                    title=f"S&P 500 Top 50 Correlation Matrix ({sp500_selected_lookback})",
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
                corr_pair_cols = st.columns(2)
                with corr_pair_cols[0]:
                    st.markdown("##### Highest Correlations")
                    for t1, t2, corr in sp500_highest:
                        st.markdown(f"{t1} - {t2}: **{corr:.3f}**")

                with corr_pair_cols[1]:
                    st.markdown("##### Lowest Correlations")
                    for t1, t2, corr in sp500_lowest:
                        st.markdown(f"{t1} - {t2}: **{corr:.3f}**")
        else:
            st.warning("S&P 500 Top 50 correlation data not found. Run `python convert_to_parquet.py` to generate.")

else:
    st.warning(f"No precomputed correlation data for {selected_etf}. Run `python convert_to_parquet.py` to generate.")
