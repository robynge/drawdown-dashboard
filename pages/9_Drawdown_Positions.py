"""Drawdown Position Changes Analysis Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE, OUTPUT_DIR
from data_loader import load_ark_holdings, load_etf_prices, get_ark_files_hash
from drawdown_calculator import calculate_drawdowns
from hhi_calculator import calculate_hhi_time_series
from chart_config import CHART_CONFIG

st.set_page_config(
    page_title="Drawdown Position Changes",
    page_icon="",
    layout="wide"
)

"""
# Drawdown Position Changes

Analyze how portfolio positions changed during drawdown periods - who was sold, who was added, and concentration changes.
"""

st.markdown(f"**Analysis Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

"" # Space


@st.cache_data
def get_drawdown_position_changes(_files_hash, etf, _holdings, peak_date, trough_date):
    """Calculate position changes between peak and trough dates"""
    holdings = _holdings

    # Get holdings at peak date (or closest date before)
    peak_holdings = holdings[holdings['Date'] <= peak_date].copy()
    if len(peak_holdings) == 0:
        return None, None, None

    peak_date_actual = peak_holdings['Date'].max()
    peak_holdings = peak_holdings[peak_holdings['Date'] == peak_date_actual].copy()

    # Get holdings at trough date (or closest date before)
    trough_holdings = holdings[holdings['Date'] <= trough_date].copy()
    if len(trough_holdings) == 0:
        return None, None, None

    trough_date_actual = trough_holdings['Date'].max()
    trough_holdings = trough_holdings[trough_holdings['Date'] == trough_date_actual].copy()

    # Filter out currency and money market
    for df in [peak_holdings, trough_holdings]:
        if 'Bloomberg Name' in df.columns:
            mask = ~df['Bloomberg Name'].str.contains('curncy', case=False, na=False)
            df.drop(df[~mask].index, inplace=True)

    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
    for df in [peak_holdings, trough_holdings]:
        ticker_symbols = df['Ticker'].str.split().str[0]
        is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
        df.drop(df[is_mm].index, inplace=True)

    # Create comparison dataframe
    peak_positions = peak_holdings.set_index('Ticker')[['Weight', 'Position']].add_suffix('_peak')
    trough_positions = trough_holdings.set_index('Ticker')[['Weight', 'Position']].add_suffix('_trough')

    comparison = peak_positions.join(trough_positions, how='outer').fillna(0)

    # Calculate changes
    comparison['Weight_Change'] = comparison['Weight_trough'] - comparison['Weight_peak']
    comparison['Position_Change'] = comparison['Position_trough'] - comparison['Position_peak']
    comparison['Position_Change_Pct'] = np.where(
        comparison['Position_peak'] > 0,
        (comparison['Position_trough'] - comparison['Position_peak']) / comparison['Position_peak'] * 100,
        np.where(comparison['Position_trough'] > 0, 100, 0)  # New position = 100% increase
    )

    # Categorize changes
    comparison['Status'] = 'Unchanged'
    comparison.loc[comparison['Position_peak'] == 0, 'Status'] = 'New Position'
    comparison.loc[comparison['Position_trough'] == 0, 'Status'] = 'Exited'
    comparison.loc[(comparison['Position_Change'] > 0) & (comparison['Position_peak'] > 0), 'Status'] = 'Added'
    comparison.loc[(comparison['Position_Change'] < 0) & (comparison['Position_trough'] > 0), 'Status'] = 'Reduced'

    comparison = comparison.reset_index()
    comparison['Ticker_Clean'] = comparison['Ticker'].str.split().str[0]

    return comparison, peak_date_actual, trough_date_actual


@st.cache_data
def get_hhi_during_drawdown(_files_hash, etf, _holdings, peak_date, trough_date):
    """Get HHI time series during drawdown period"""
    holdings = _holdings

    # Filter to drawdown period
    dd_holdings = holdings[
        (holdings['Date'] >= peak_date) &
        (holdings['Date'] <= trough_date)
    ].copy()

    if len(dd_holdings) == 0:
        return pd.DataFrame()

    # Filter out currency and money market
    if 'Bloomberg Name' in dd_holdings.columns:
        dd_holdings = dd_holdings[~dd_holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]

    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
    ticker_symbols = dd_holdings['Ticker'].str.split().str[0]
    is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
    dd_holdings = dd_holdings[~is_mm]

    # Calculate HHI
    hhi_df = calculate_hhi_time_series(dd_holdings[['Date', 'Ticker', 'Weight']])

    return hhi_df


# Load data
files_hash = get_ark_files_hash()

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

# Load data
with st.spinner("Loading data..."):
    holdings = load_ark_holdings(files_hash, selected_etf)
    etf_prices = load_etf_prices(selected_etf)

    if len(etf_prices) > 0:
        drawdowns = calculate_drawdowns(etf_prices)

if len(etf_prices) > 0 and len(drawdowns) > 0:
    # Filter to top 10 historical drawdowns (exclude current)
    historical_dds = drawdowns[drawdowns['rank'] != 'Current'].head(10)

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

    # Get position changes
    peak_date = selected_dd['peak_date']
    trough_date = selected_dd['trough_date']

    comparison, peak_date_actual, trough_date_actual = get_drawdown_position_changes(
        files_hash, selected_etf, holdings, peak_date, trough_date
    )

    hhi_during_dd = get_hhi_during_drawdown(files_hash, selected_etf, holdings, peak_date, trough_date)

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

                # HHI chart during drawdown
                fig_hhi = go.Figure()
                fig_hhi.add_trace(go.Scatter(
                    x=hhi_during_dd['Date'],
                    y=hhi_during_dd['HHI'],
                    mode='lines',
                    name='HHI',
                    line=dict(color='steelblue', width=2)
                ))
                fig_hhi.update_layout(
                    title=f"HHI During Drawdown",
                    xaxis_title="Date",
                    yaxis_title="HHI",
                    height=300,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig_hhi.update_xaxes(gridcolor='lightgray')
                fig_hhi.update_yaxes(gridcolor='lightgray')
                st.plotly_chart(fig_hhi, use_container_width=True, config=CHART_CONFIG)
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

            st.dataframe(display_df, hide_index=True, use_container_width=True)

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

                # For decreased positions: gray from 0 to trough, red from trough to peak
                # For increased positions: gray from 0 to peak, green from peak to trough

                bar_width = 0.7  # Width for base bars
                green_bar_width = 0.82  # Wider to match gray + black border (2px on each side)

                for i, ticker in enumerate(tickers):
                    peak_w = peak_weights[i]
                    trough_w = trough_weights[i]
                    change = changes[i]

                    if change < 0:
                        # Decreased: gray=remaining (trough), red=lost portion
                        # Gray bar (remaining weight)
                        fig_changes.add_trace(go.Bar(
                            y=[ticker], x=[trough_w], orientation='h',
                            marker=dict(color='lightgray', line=dict(width=0)),
                            width=bar_width,
                            showlegend=False, hoverinfo='skip'
                        ))
                        # Red bar (lost portion, starts at trough)
                        fig_changes.add_trace(go.Bar(
                            y=[ticker], x=[abs(change)], orientation='h',
                            marker=dict(color='rgba(220,50,50,0.85)', line=dict(width=0)),
                            base=trough_w,
                            width=bar_width,
                            showlegend=False, hoverinfo='skip'
                        ))
                        # Black border around peak (entire original position)
                        fig_changes.add_trace(go.Bar(
                            y=[ticker], x=[peak_w], orientation='h',
                            marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=2)),
                            width=bar_width,
                            showlegend=False, hoverinfo='skip'
                        ))
                    else:
                        # Increased: gray=original (peak), green=added portion
                        # Gray bar (original weight, no border yet)
                        fig_changes.add_trace(go.Bar(
                            y=[ticker], x=[peak_w], orientation='h',
                            marker=dict(color='lightgray', line=dict(width=0)),
                            width=bar_width,
                            showlegend=False, hoverinfo='skip'
                        ))
                        # Green bar (added portion, starts at peak, wider to match gray+border)
                        fig_changes.add_trace(go.Bar(
                            y=[ticker], x=[change], orientation='h',
                            marker=dict(color='rgba(50,180,50,0.85)', line=dict(width=0)),
                            base=peak_w,
                            width=green_bar_width,
                            showlegend=False, hoverinfo='skip'
                        ))
                        # Black border on top (covers green edge)
                        fig_changes.add_trace(go.Bar(
                            y=[ticker], x=[peak_w], orientation='h',
                            marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=2)),
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

                st.plotly_chart(fig_changes, use_container_width=True, config=CHART_CONFIG)
    else:
        st.warning("No position data available for this drawdown period")

else:
    st.warning(f"No data available for {selected_etf}")
