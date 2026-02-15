"""Drawdown Position Changes Analysis Page - Using Precomputed Data"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, OUTPUT_DIR
from precomputed_loader import (
    load_etf_drawdowns,
    load_hhi_timeseries,
    load_position_changes,
    filter_by_period,
    filter_drawdowns_by_period,
    check_precomputed_exists
)
from chart_config import CHART_CONFIG
from session_utils import init_session_state, get_current_dates, render_period_selector

st.set_page_config(
    page_title="Drawdown Position Changes",
    page_icon="",
    layout="wide"
)


@st.cache_data
def load_etf_prices(etf):
    """Load ETF prices from CSV"""
    etf_file = OUTPUT_DIR / f'{etf}_prices.csv'
    if etf_file.exists():
        df = pd.read_csv(etf_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Drawdown Position Changes

Analyze how portfolio positions changed during drawdown periods - who was sold, who was added, and concentration changes.
"""

st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")


# ETF Selection
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Select ETF")
    selected_etf = st.pills(
        "ETF",
        options=ARK_ETFS,
        default=ARK_ETFS[0],
        label_visibility="collapsed"
    )

# Load precomputed data
with st.spinner("Loading data..."):
    # Load precomputed ETF drawdowns
    drawdowns_full = load_etf_drawdowns(selected_etf)
    drawdowns = filter_drawdowns_by_period(drawdowns_full, start_date, end_date)

    # Load precomputed position changes
    position_changes_full = load_position_changes(selected_etf)

    # Load precomputed HHI data and filter to analysis period
    hhi_full = load_hhi_timeseries(selected_etf)
    hhi_period = filter_by_period(hhi_full, start_date, end_date)

    # Load ETF prices and filter to analysis period
    etf_prices_full = load_etf_prices(selected_etf)
    etf_prices = etf_prices_full[
        (etf_prices_full['Date'] >= start_date) &
        (etf_prices_full['Date'] <= end_date)
    ].copy() if len(etf_prices_full) > 0 else pd.DataFrame()

if len(drawdowns) > 0:
    # Filter to top 10 historical drawdowns (exclude current)
    historical_dds = drawdowns[drawdowns['rank'] != 'Current'].head(10)

    if len(historical_dds) > 0:
        with col1:
            "" # Space
            st.subheader("Select Drawdown")

            # Create drawdown options
            dd_options = []
            for _, row in historical_dds.iterrows():
                label = f"#{row['rank']}: {row['depth_pct']:.1f}% ({row['peak_date'].strftime('%Y-%m-%d')})"
                dd_options.append(label)

            selected_dd_label = st.selectbox("Drawdown", dd_options, label_visibility="collapsed")
            selected_dd_idx = dd_options.index(selected_dd_label)
            selected_dd = historical_dds.iloc[selected_dd_idx]

        # Get data for selected drawdown
        peak_date = selected_dd['peak_date']
        trough_date = selected_dd['trough_date']
        dd_rank = selected_dd['rank']

        # Try to get precomputed position changes (match by peak_date, not rank)
        if len(position_changes_full) > 0:
            # Ensure peak_date_actual is datetime for comparison
            if 'peak_date_actual' in position_changes_full.columns:
                position_changes_full['peak_date_actual'] = pd.to_datetime(position_changes_full['peak_date_actual'])
            # Match by peak_date since ranks are period-specific but precomputed data uses global dates
            comparison = position_changes_full[position_changes_full['peak_date_actual'] == peak_date].copy()
            if len(comparison) > 0:
                peak_date_actual = comparison['peak_date_actual'].iloc[0]
                trough_date_actual = comparison['trough_date_actual'].iloc[0]
            else:
                comparison = None
                peak_date_actual = None
                trough_date_actual = None
        else:
            comparison = None
            peak_date_actual = None
            trough_date_actual = None

        # Get HHI data for the drawdown period
        if len(hhi_full) > 0:
            hhi_during_dd = hhi_full[
                (hhi_full['Date'] >= peak_date) &
                (hhi_full['Date'] <= trough_date)
            ].copy()
        else:
            hhi_during_dd = pd.DataFrame()

        with col2:
            # Drawdown summary
            st.subheader(f"Drawdown #{selected_dd['rank']}: {selected_dd['depth_pct']:.2f}%")

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Peak Date", peak_date.strftime('%Y-%m-%d'))
            with metric_cols[1]:
                st.metric("Trough Date", trough_date.strftime('%Y-%m-%d'))
            with metric_cols[2]:
                duration = (trough_date - peak_date).days
                st.metric("Duration", f"{duration} days")
            with metric_cols[3]:
                st.metric("Depth", f"{selected_dd['depth_pct']:.2f}%")

        "" # Space

        if comparison is not None and len(comparison) > 0:
            # HHI Change
            st.subheader("Concentration Change")

            hhi_card = st.container(border=True)
            with hhi_card:
                if len(hhi_during_dd) > 0:
                    hhi_start = hhi_during_dd['HHI'].iloc[0]
                    hhi_end = hhi_during_dd['HHI'].iloc[-1]
                    hhi_change = hhi_end - hhi_start
                    hhi_change_pct = (hhi_change / hhi_start) * 100

                    eff_pos_start = hhi_during_dd['Effective_Positions'].iloc[0]
                    eff_pos_end = hhi_during_dd['Effective_Positions'].iloc[-1]

                    hhi_cols = st.columns(4)
                    with hhi_cols[0]:
                        st.metric("HHI at Peak", f"{hhi_start:.4f}")
                    with hhi_cols[1]:
                        st.metric("HHI at Trough", f"{hhi_end:.4f}", delta=f"{hhi_change:+.4f}")
                    with hhi_cols[2]:
                        st.metric("Effective Positions (Peak)", f"{eff_pos_start:.1f}")
                    with hhi_cols[3]:
                        st.metric("Effective Positions (Trough)", f"{eff_pos_end:.1f}", delta=f"{eff_pos_end - eff_pos_start:+.1f}")

                    if hhi_change > 0:
                        st.warning(f"Portfolio became **more concentrated** during this drawdown (HHI +{hhi_change_pct:.1f}%)")
                    else:
                        st.success(f"Portfolio became **less concentrated** during this drawdown (HHI {hhi_change_pct:.1f}%)")

                    "" # Space

                    # Price & HHI chart for entire analysis period with highlighted drawdown
                    if len(etf_prices) > 0 and len(hhi_period) > 0:
                        from plotly.subplots import make_subplots

                        fig_hhi = make_subplots(specs=[[{"secondary_y": True}]])

                        # Add drawdown region highlight
                        fig_hhi.add_vrect(
                            x0=peak_date,
                            x1=trough_date,
                            fillcolor='rgba(255, 0, 0, 0.15)',
                            layer="below",
                            line_width=0
                        )

                        # ETF Price line
                        fig_hhi.add_trace(
                            go.Scatter(
                                x=etf_prices['Date'],
                                y=etf_prices['Close'],
                                mode='lines',
                                name=f'{selected_etf} Price',
                                line=dict(color='black', width=2),
                                hovertemplate=f'<b>{selected_etf}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>'
                            ),
                            secondary_y=False
                        )

                        # HHI line
                        fig_hhi.add_trace(
                            go.Scatter(
                                x=hhi_period['Date'],
                                y=hhi_period['HHI'],
                                mode='lines',
                                name='HHI',
                                line=dict(color='steelblue', width=2, dash='dot'),
                                hovertemplate='<b>HHI</b><br>Date: %{x|%Y-%m-%d}<br>HHI: %{y:.4f}<extra></extra>'
                            ),
                            secondary_y=True
                        )

                        # Add vertical lines for peak and trough
                        fig_hhi.add_vline(x=peak_date, line_dash="dash", line_color="red", line_width=1)
                        fig_hhi.add_vline(x=trough_date, line_dash="dash", line_color="red", line_width=1)

                        # Add annotations for peak and trough
                        fig_hhi.add_annotation(
                            x=peak_date, y=1.02, yref="paper",
                            text="Peak", showarrow=False,
                            font=dict(size=10, color='red')
                        )
                        fig_hhi.add_annotation(
                            x=trough_date, y=1.02, yref="paper",
                            text="Trough", showarrow=False,
                            font=dict(size=10, color='red')
                        )

                        fig_hhi.update_layout(
                            title=f"{selected_etf} Price & HHI (Drawdown #{dd_rank} Highlighted)",
                            height=400,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                            margin=dict(l=0, r=0, t=60, b=0),
                            hovermode='x unified'
                        )
                        fig_hhi.update_xaxes(title_text="Date", gridcolor='lightgray')
                        fig_hhi.update_yaxes(title_text=f"{selected_etf} Price ($)", gridcolor='lightgray', secondary_y=False)
                        fig_hhi.update_yaxes(title_text="HHI", showgrid=False, secondary_y=True)

                        st.plotly_chart(fig_hhi, width='stretch', config=CHART_CONFIG)

                        st.markdown("<small>*Red shaded area indicates the selected drawdown period.*</small>", unsafe_allow_html=True)
                    else:
                        st.warning("Price or HHI data not available for this period")
                else:
                    st.warning("No HHI data available for this period")

            "" # Space

            # Position changes summary
            st.subheader("Position Changes")

            summary_card = st.container(border=True)
            with summary_card:
                status_counts = comparison['Status'].value_counts()

                sum_cols = st.columns(5)
                with sum_cols[0]:
                    st.metric("Exited", status_counts.get('Exited', 0))
                with sum_cols[1]:
                    st.metric("Reduced", status_counts.get('Reduced', 0))
                with sum_cols[2]:
                    st.metric("Unchanged", status_counts.get('Unchanged', 0))
                with sum_cols[3]:
                    st.metric("Added", status_counts.get('Added', 0))
                with sum_cols[4]:
                    st.metric("New Position", status_counts.get('New Position', 0))

            "" # Space

            # Position Changes Details Table
            st.subheader("Position Changes Details")

            details_card = st.container(border=True)
            with details_card:
                # Prepare display dataframe
                display_df = comparison[comparison['Status'] != 'Unchanged'].copy()
                display_df = display_df.sort_values('Weight_Change')

                # Format columns
                display_df['Ticker'] = display_df['Ticker_Clean']
                display_df['Weight Peak (%)'] = (display_df['Weight_peak'] * 100).round(2)
                display_df['Weight Trough (%)'] = (display_df['Weight_trough'] * 100).round(2)
                display_df['Weight Change (%)'] = (display_df['Weight_Change'] * 100).round(2)
                display_df['Position Peak'] = display_df['Position_peak'].astype(int)
                display_df['Position Trough'] = display_df['Position_trough'].astype(int)
                display_df['Position Change (%)'] = display_df['Position_Change_Pct'].round(2)

                # Select columns to display
                display_cols = ['Ticker', 'Status', 'Weight Peak (%)', 'Weight Trough (%)',
                               'Weight Change (%)', 'Position Peak', 'Position Trough', 'Position Change (%)']
                display_df = display_df[display_cols]

                st.dataframe(display_df, hide_index=True, width='stretch')

            "" # Space

            # Weight change chart
            st.subheader("Weight Changes Visualization")

            chart_card = st.container(border=True)
            with chart_card:
                # Show all tickers with weight changes (exclude unchanged)
                chart_data = comparison[comparison['Weight_Change'] != 0].copy()
                chart_data = chart_data.sort_values('Weight_Change')

                if len(chart_data) == 0:
                    st.info("No weight changes during this drawdown period")
                else:
                    # Prepare data for stacked bar chart
                    chart_data['Weight_peak_pct'] = chart_data['Weight_peak'] * 100
                    chart_data['Weight_trough_pct'] = chart_data['Weight_trough'] * 100
                    chart_data['Weight_Change_pct'] = chart_data['Weight_Change'] * 100

                    # Build data for chart
                    tickers = chart_data['Ticker_Clean'].tolist()
                    peak_weights = chart_data['Weight_peak_pct'].tolist()
                    trough_weights = chart_data['Weight_trough_pct'].tolist()
                    changes = chart_data['Weight_Change_pct'].tolist()

                    fig_changes = go.Figure()

                    bar_width = 0.7

                    for i, ticker in enumerate(tickers):
                        peak_w = peak_weights[i]
                        trough_w = trough_weights[i]
                        change = changes[i]

                        if change < 0:
                            # Decreased: gray=remaining (trough), red=lost portion
                            fig_changes.add_trace(go.Bar(
                                y=[ticker], x=[trough_w], orientation='h',
                                marker=dict(color='lightgray', line=dict(width=0)),
                                width=bar_width,
                                showlegend=False, hoverinfo='skip'
                            ))
                            fig_changes.add_trace(go.Bar(
                                y=[ticker], x=[abs(change)], orientation='h',
                                marker=dict(color='rgba(220,50,50,0.85)', line=dict(width=0)),
                                base=trough_w,
                                width=bar_width,
                                showlegend=False, hoverinfo='skip'
                            ))
                            fig_changes.add_trace(go.Bar(
                                y=[ticker], x=[peak_w], orientation='h',
                                marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=2)),
                                width=bar_width,
                                showlegend=False, hoverinfo='skip'
                            ))
                        else:
                            # Increased: gray=original (peak) with black border, green=added portion OUTSIDE border
                            fig_changes.add_trace(go.Bar(
                                y=[ticker], x=[peak_w], orientation='h',
                                marker=dict(color='lightgray', line=dict(color='black', width=2)),
                                width=bar_width,
                                showlegend=False, hoverinfo='skip'
                            ))
                            fig_changes.add_trace(go.Bar(
                                y=[ticker], x=[change], orientation='h',
                                marker=dict(color='rgba(50,180,50,0.85)', line=dict(width=0)),
                                base=peak_w,
                                width=bar_width,
                                showlegend=False, hoverinfo='skip'
                            ))

                    # Add text annotations for weight change
                    for _, row in chart_data.iterrows():
                        ticker = row['Ticker_Clean']
                        peak_w = row['Weight_peak_pct']
                        trough_w = row['Weight_trough_pct']
                        change = row['Weight_Change_pct']
                        max_w = max(peak_w, trough_w)

                        change_text = f"{change:+.2f}%" if change != 0 else "0%"
                        fig_changes.add_annotation(
                            y=ticker,
                            x=max_w + 0.3,
                            text=change_text,
                            showarrow=False,
                            font=dict(size=10, color='red' if change < 0 else 'green'),
                            xanchor='left'
                        )

                    # Dynamic height based on number of tickers
                    chart_height = max(400, len(chart_data) * 25)

                    fig_changes.update_layout(
                        title="Weight at Peak vs Trough (Gray=Base, Red=Reduced, Green=Added)",
                        xaxis_title="Weight (%)",
                        yaxis_title="",
                        height=chart_height,
                        barmode='overlay',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(l=0, r=50, t=40, b=0)
                    )
                    fig_changes.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='gray')
                    fig_changes.update_yaxes(showgrid=False)

                    st.plotly_chart(fig_changes, width='stretch', config=CHART_CONFIG)
        else:
            st.warning("No precomputed position data available for this drawdown. Run `python convert_to_parquet.py` to generate.")
    else:
        st.warning(f"No historical drawdowns found for {selected_etf} in the selected period.")

else:
    st.warning(f"No drawdown data available for {selected_etf}. Run `python convert_to_parquet.py` to generate precomputed data.")
