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

        # Detailed position changes
        detail_cols = st.columns(2)

        with detail_cols[0]:
            st.markdown("#### Positions Reduced/Exited")
            reduced = comparison[comparison['Status'].isin(['Exited', 'Reduced'])].copy()
            reduced = reduced.sort_values('Weight_Change')

            if len(reduced) > 0:
                reduced_card = st.container(border=True)
                with reduced_card:
                    for _, row in reduced.head(15).iterrows():
                        status_icon = "🔴" if row['Status'] == 'Exited' else "🟠"
                        weight_peak = row['Weight_peak'] * 100
                        weight_trough = row['Weight_trough'] * 100
                        weight_change = row['Weight_Change'] * 100
                        st.markdown(f"{status_icon} **{row['Ticker_Clean']}**: {weight_peak:.2f}% → {weight_trough:.2f}% ({weight_change:+.2f}%)")
            else:
                st.info("No positions were reduced or exited")

        with detail_cols[1]:
            st.markdown("#### Positions Added/New")
            added = comparison[comparison['Status'].isin(['Added', 'New Position'])].copy()
            added = added.sort_values('Weight_Change', ascending=False)

            if len(added) > 0:
                added_card = st.container(border=True)
                with added_card:
                    for _, row in added.head(15).iterrows():
                        status_icon = "🟢" if row['Status'] == 'New Position' else "🔵"
                        weight_peak = row['Weight_peak'] * 100
                        weight_trough = row['Weight_trough'] * 100
                        weight_change = row['Weight_Change'] * 100
                        st.markdown(f"{status_icon} **{row['Ticker_Clean']}**: {weight_peak:.2f}% → {weight_trough:.2f}% ({weight_change:+.2f}%)")
            else:
                st.info("No positions were added")

        "" # Space

        # Weight change chart
        st.subheader("Weight Changes Visualization")

        chart_card = st.container(border=True)
        with chart_card:
            # Sort by weight change and take top/bottom
            top_reduced = comparison.nsmallest(10, 'Weight_Change')
            top_added = comparison.nlargest(10, 'Weight_Change')
            chart_data = pd.concat([top_reduced, top_added]).drop_duplicates()
            chart_data = chart_data.sort_values('Weight_Change')

            colors = ['red' if x < 0 else 'green' for x in chart_data['Weight_Change']]

            fig_changes = go.Figure()
            fig_changes.add_trace(go.Bar(
                y=chart_data['Ticker_Clean'],
                x=chart_data['Weight_Change'] * 100,
                orientation='h',
                marker_color=colors,
                hovertemplate='<b>%{y}</b><br>Weight Change: %{x:.2f}%<extra></extra>'
            ))

            fig_changes.update_layout(
                title="Top Weight Changes (Peak → Trough)",
                xaxis_title="Weight Change (%)",
                yaxis_title="",
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig_changes.update_xaxes(gridcolor='lightgray', zeroline=True, zerolinecolor='gray')
            fig_changes.update_yaxes(gridcolor='lightgray')

            st.plotly_chart(fig_changes, use_container_width=True, config=CHART_CONFIG)
    else:
        st.warning("No position data available for this drawdown period")

else:
    st.warning(f"No data available for {selected_etf}")
