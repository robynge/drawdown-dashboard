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
    load_correlation_matrix,
    check_precomputed_exists
)
from data_loader import load_etf_prices
from session_utils import init_session_state, get_current_dates, render_period_selector

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

Analyze how portfolio correlations change during drawdown periods (stress correlations).
"""

st.markdown("**Analysis Period:** 2021-01-01 to 2026-02-21 (Full Data)")
st.caption("Stress correlations are calculated across all top 10 drawdowns in the full historical period.")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")

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

        "" # Space

        st.markdown("##### Lookback Period")
        st.markdown("*For normal correlation baseline*")
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

# Load data
stress_corr = load_stress_correlations(selected_etf)
corr_matrix = load_correlation_matrix(selected_etf, lookback_days)

# Calculate normal correlation baseline
normal_corr = 0
if corr_matrix is not None and len(corr_matrix) > 0:
    n = len(corr_matrix.columns)
    triu_i, triu_j = np.triu_indices(n, k=1)
    corr_values = corr_matrix.values[triu_i, triu_j]
    valid_corrs = corr_values[~np.isnan(corr_values)]
    if len(valid_corrs) > 0:
        normal_corr = np.mean(valid_corrs)

if len(stress_corr) > 0:
    # Summary statistics in left panel
    stress_mean = stress_corr['mean_corr'].mean()
    correlation_increase = ((stress_mean / normal_corr) - 1) * 100 if normal_corr > 0 else 0

    with cols[0]:
        "" # Space

        stats_card = st.container(border=True)
        with stats_card:
            st.markdown("##### Summary Statistics")

            st.markdown(f"**Drawdown Events:** {len(stress_corr)}")

            "" # Space

            st.markdown("**Normal Correlation**")
            st.markdown(f"Mean: **{normal_corr:.3f}**")
            st.caption(f"Based on {selected_lookback}")

            "" # Space

            st.markdown("**Stress Correlation**")
            st.markdown(f"Mean: **{stress_mean:.3f}**")
            delta_color = "🔴" if stress_mean > normal_corr else "🟢"
            st.markdown(f"Δ: {delta_color} **{stress_mean - normal_corr:+.3f}**")

            "" # Space

            st.markdown("**Correlation Increase**")
            if correlation_increase > 0:
                st.markdown(f"**+{correlation_increase:.1f}%** during stress")
            else:
                st.markdown(f"**{correlation_increase:.1f}%** during stress")

    # Right panel: Charts
    with cols[1]:
        st.markdown("""
        **Stress correlations** measure how correlated holdings become during drawdown periods.
        Typically, correlations **increase** during market stress, reducing diversification benefits when they're needed most.
        """)

        "" # Space

        # Key metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Normal Correlation", f"{normal_corr:.3f}",
                      help=f"Average pairwise correlation over {selected_lookback}")

        with metric_cols[1]:
            st.metric("Stress Correlation", f"{stress_mean:.3f}",
                      delta=f"{stress_mean - normal_corr:+.3f}",
                      delta_color="inverse",
                      help="Average correlation during drawdown periods")

        with metric_cols[2]:
            st.metric("Correlation Increase", f"{correlation_increase:+.1f}%",
                      help="How much correlations increase during stress")

        "" # Space

        # Price chart with correlation overlay
        st.markdown("#### Correlation by Drawdown Event")

        chart_card = st.container(border=True)
        with chart_card:
            # Load ETF prices
            etf_prices = load_etf_prices(selected_etf)

            if len(etf_prices) > 0:
                # Use full price history
                price_df = etf_prices.copy()

                # Create figure (go.Figure works with add_vrect, make_subplots doesn't)
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
                    midpoint_corrs.append(row['mean_corr'])
                    hover_texts.append(
                        f"<b>Drawdown #{int(row['dd_rank'])}</b><br>" +
                        f"Period: {peak.strftime('%Y-%m-%d')} to {trough.strftime('%Y-%m-%d')}<br>" +
                        f"Depth: {row['depth_pct']:.1f}%<br>" +
                        f"Duration: {row['duration_days']} days<br>" +
                        f"Mean Correlation: {row['mean_corr']:.3f}"
                    )

                # Sort by date for proper line connection
                sorted_data = sorted(zip(midpoint_dates, midpoint_corrs, hover_texts), key=lambda x: x[0])
                midpoint_dates = [x[0] for x in sorted_data]
                midpoint_corrs = [x[1] for x in sorted_data]
                hover_texts = [x[2] for x in sorted_data]

                # Add correlation line on secondary y-axis
                fig_stress.add_trace(go.Scatter(
                    x=midpoint_dates,
                    y=midpoint_corrs,
                    mode='lines+markers',
                    name='Stress Correlation',
                    line=dict(color='red', width=2),
                    marker=dict(size=12, color='red', symbol='circle'),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hover_texts,
                    yaxis='y2'
                ))

                # Add normal correlation reference line
                fig_stress.add_hline(
                    y=normal_corr, line_dash="dash", line_color="steelblue", line_width=2,
                    annotation_text=f"Normal: {normal_corr:.3f}",
                    annotation_position="right",
                    annotation_font_color="steelblue",
                    yref="y2"
                )

                fig_stress.update_layout(
                    title=f"{selected_etf} Price & Stress Correlation by Drawdown",
                    height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='closest',
                    xaxis=dict(title="Date", gridcolor='lightgray'),
                    yaxis=dict(title=f"{selected_etf} Price ($)", gridcolor='lightgray'),
                    yaxis2=dict(title="Mean Correlation", overlaying='y', side='right', range=[0, 1], showgrid=False)
                )

                st.plotly_chart(fig_stress, width='stretch')

                st.markdown("<small>*Colored regions show top 10 drawdown periods (colors match table order: #1=red, #2=orange, #3=yellow, etc.). Red line connects mean correlation at midpoint of each drawdown. Blue dashed line = normal correlation baseline.*</small>", unsafe_allow_html=True)
            else:
                st.warning(f"No price data available for {selected_etf}")

        "" # Space

        # Detailed table
        st.markdown("#### Detailed Stress Correlation Data")

        table_card = st.container(border=True)
        with table_card:
            display_df = stress_corr.copy()
            display_df['peak_date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
            display_df['trough_date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
            display_df = display_df.rename(columns={
                'dd_rank': 'Rank',
                'peak_date': 'Peak',
                'trough_date': 'Trough',
                'depth_pct': 'Depth %',
                'duration_days': 'Days',
                'num_tickers': 'Tickers',
                'mean_corr': 'Mean ρ',
                'median_corr': 'Median ρ',
                'max_corr': 'Max ρ',
                'min_corr': 'Min ρ'
            })

            # Format numeric columns
            for col in ['Depth %', 'Mean ρ', 'Median ρ', 'Max ρ', 'Min ρ']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

            st.dataframe(
                display_df[['Rank', 'Peak', 'Trough', 'Depth %', 'Days', 'Tickers', 'Mean ρ', 'Median ρ', 'Max ρ', 'Min ρ']],
                hide_index=True,
                width='stretch'
            )

        "" # Space

        # Insight
        if stress_mean > normal_corr:
            st.warning(f"⚠️ **Correlations increase {correlation_increase:.0f}% during drawdowns.** Diversification benefits are reduced when the portfolio is under stress.")
        else:
            st.success(f"✓ **Correlations remain stable during drawdowns.** The portfolio maintains diversification benefits during stress periods.")

else:
    with cols[1]:
        st.warning(f"No stress correlation data for {selected_etf}. Run `python convert_to_parquet.py` to generate.")
