"""Stock Analysis Page - Using Precomputed Data"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import OUTPUT_DIR, ARK_ETFS, INPUT_DIR
from data_loader import (
    load_ark_holdings, load_industry_info, load_company_name,
    get_ark_files_hash, get_stocks_for_etf,
    get_industry_files_hash, get_company_name_files_hash
)
from precomputed_loader import (
    load_ark_stock_drawdowns,
    load_peer_group_drawdowns,
    filter_drawdowns_by_period,
    check_precomputed_exists
)
from peer_group import get_peer_group_prices
from drawdown_calculator import calculate_drawdowns
from chart_config import CHART_CONFIG, DD_COLORS, add_reconstitution_vlines
from recovery_probability import get_stock_drawdowns_in_depth_range
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector, is_latest_period

st.set_page_config(
    page_title="Individual Stock vs Peer Group",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

st.title("Individual Stock vs Peer Group")

st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Run `python convert_to_parquet.py` for faster loading.")


def get_price_column(stock_data):
    """Detect which price column has actual data"""
    if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
        return 'YFinance Close Price'
    return 'Stock_Price'


@st.cache_data
def load_stock_data(ticker, etf, _files_hash, _start_date, _end_date):
    """Load stock data from ARK holdings"""
    holdings = load_ark_holdings(_files_hash, etf)

    matching_tickers = holdings[holdings['Ticker'].str.startswith(ticker + ' ', na=False) |
                               (holdings['Ticker'] == ticker)]['Ticker'].unique()

    if len(matching_tickers) == 0:
        return None, None, None, False

    full_ticker = matching_tickers[0]
    stock_data = holdings[holdings['Ticker'] == full_ticker].copy()
    stock_data = stock_data[(stock_data['Date'] >= _start_date) & (stock_data['Date'] <= _end_date)]

    bloomberg_name = stock_data['Bloomberg Name'].iloc[0] if len(stock_data) > 0 else None

    # Check if this stock is in current holdings
    latest_date = holdings['Date'].max()
    current_holdings = holdings[holdings['Date'] == latest_date]['Ticker'].unique()
    is_current = full_ticker in current_holdings

    return stock_data, full_ticker, bloomberg_name, is_current


# Main Analysis Section
if True:
    st.subheader("Individual Stock vs Peer Group Analysis")

    # Layout: left controls and metrics, right chart
    cols = st.columns([1, 3])

    # Initialize variables
    stock_data = None
    dd_data = pd.DataFrame()
    gics = None
    peer_prices = pd.DataFrame()
    peer_dd_data = pd.DataFrame()

    # Left column: two stacked cards
    with cols[0]:
        # Card 1: Selection Controls
        selection_card = st.container(border=True)
        with selection_card:
            st.markdown("##### ETF")
            selected_etf = st.selectbox("Select ETF", ARK_ETFS, label_visibility="collapsed")

            st.markdown("##### Select Stock")
            files_hash = get_ark_files_hash()
            stock_list, stock_ticker_map = get_stocks_for_etf(files_hash, selected_etf, start_date, end_date)
            selected_display_ticker = st.selectbox("Select Stock", stock_list, label_visibility="collapsed")

            # Get the actual ticker from display name
            selected_ticker = stock_ticker_map.get(selected_display_ticker, selected_display_ticker.replace(" (Non-current)", ""))

            st.markdown("##### Peer Group Version")
            version = st.pills(
                "Version",
                ["Market Value", "Weighted Price"],
                default="Market Value",
                label_visibility="collapsed"
            )
            version_param = "mv" if version == "Market Value" else "weighted"

        files_hash = get_ark_files_hash()
        stock_data, full_ticker, bloomberg_name, is_current = load_stock_data(
            selected_ticker, selected_etf, files_hash, start_date, end_date
        )

        if stock_data is None or len(stock_data) == 0:
            st.error(f"No data available for {selected_ticker} in {selected_etf}")
        else:
            # Try to load precomputed stock drawdowns
            dd_precomputed = load_ark_stock_drawdowns(selected_etf, selected_ticker)
            if len(dd_precomputed) > 0:
                dd_data = filter_drawdowns_by_period(dd_precomputed, start_date, end_date)
                # If filtered result is empty, recalculate
                if len(dd_data) == 0:
                    price_col = get_price_column(stock_data)
                    price_df = stock_data[['Date', price_col]].copy()
                    price_df.columns = ['Date', 'Close']
                    dd_data = calculate_drawdowns(price_df, start_date=start_date, end_date=end_date)
            else:
                # Fallback to dynamic calculation
                price_col = get_price_column(stock_data)
                price_df = stock_data[['Date', price_col]].copy()
                price_df.columns = ['Date', 'Close']
                dd_data = calculate_drawdowns(price_df, start_date=start_date, end_date=end_date)

            # Get GICS industry
            industry_dict = load_industry_info(get_industry_files_hash(), source='ark')
            gics = industry_dict.get(bloomberg_name) if bloomberg_name else None

            # Load peer group data and drawdowns
            if gics:
                try:
                    peer_prices = get_peer_group_prices(
                        gics, version=version_param,
                        period_key=get_current_period(), start_date=start_date, end_date=end_date
                    )

                    # Try to load precomputed peer group drawdowns
                    peer_dd_precomputed = load_peer_group_drawdowns(gics, version=version_param)
                    if len(peer_dd_precomputed) > 0:
                        peer_dd_data = filter_drawdowns_by_period(peer_dd_precomputed, start_date, end_date)
                        # If filtered result is empty, recalculate
                        if len(peer_dd_data) == 0 and len(peer_prices) > 0:
                            peer_prices_for_dd = peer_prices.copy()
                            peer_prices_for_dd = peer_prices_for_dd.rename(columns={'Value': 'Close'})
                            peer_dd_data = calculate_drawdowns(peer_prices_for_dd, start_date=start_date, end_date=end_date)
                    elif len(peer_prices) > 0:
                        # Fallback to dynamic calculation
                        peer_prices_for_dd = peer_prices.copy()
                        peer_prices_for_dd = peer_prices_for_dd.rename(columns={'Value': 'Close'})
                        peer_dd_data = calculate_drawdowns(peer_prices_for_dd, start_date=start_date, end_date=end_date)
                except:
                    peer_prices = pd.DataFrame()
                    peer_dd_data = pd.DataFrame()
                    gics = None
            else:
                peer_prices = pd.DataFrame()
                peer_dd_data = pd.DataFrame()

            # Card 2: Key Metrics
            if len(dd_data) > 0:
                metrics_card = st.container(border=True)
                with metrics_card:
                    st.markdown("##### Key Metrics")

                    current_dd_rows = dd_data[dd_data['rank'] == 'Current']
                    top_dd_rows = dd_data[dd_data['rank'] == '1']
                    historical_dd = dd_data[dd_data['rank'] != 'Current']

                    # Get top drawdown
                    if len(top_dd_rows) > 0:
                        top_dd = top_dd_rows.iloc[0]
                    elif len(historical_dd) > 0:
                        top_dd = historical_dd.iloc[0]
                    else:
                        top_dd = None

                    price_col = get_price_column(stock_data)
                    first_price = stock_data[price_col].iloc[0]
                    last_price = stock_data[price_col].iloc[-1]
                    overall_return = ((last_price - first_price) / first_price) * 100
                    max_dd_abs = abs(top_dd['depth_pct']) if top_dd is not None else 0
                    romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

                    # Company Name
                    try:
                        company_names = load_company_name(get_company_name_files_hash(), source='ark')
                        company_name = company_names.get(selected_ticker)
                        if company_name:
                            st.markdown(f"<small>Company Name</small><br><b>{company_name}</b>", unsafe_allow_html=True)
                    except:
                        pass

                    # GICS Industry Group
                    if gics:
                        st.markdown(f"<small>GICS Industry Group</small><br><b>{gics}</b>", unsafe_allow_html=True)

                    if top_dd is not None:
                        st.markdown(f"<small>Max Drawdown</small><br><b>{top_dd['depth_pct']:.2f}%</b>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<small>Max Drawdown</small><br><b>N/A</b>", unsafe_allow_html=True)

                    price_col = get_price_column(stock_data)
                    cols_price = st.columns(2)
                    with cols_price[0]:
                        current_price = stock_data[price_col].iloc[-1]
                        st.markdown(f"<small>Current Price</small><br><b>${current_price:.2f}</b>", unsafe_allow_html=True)
                    with cols_price[1]:
                        peak_price = stock_data[price_col].max()
                        st.markdown(f"<small>Peak Price</small><br><b>${peak_price:.2f}</b>", unsafe_allow_html=True)

                    st.markdown(f"<small>RoMaD</small><br><b>{romad:.2f}</b>", unsafe_allow_html=True)

    # Right column: two charts stacked
    if stock_data is not None and len(stock_data) > 0:
        right_panel = cols[1].container(border=True)

        with right_panel:
            price_col = get_price_column(stock_data)

            # ============ CHART 1: STOCK PRICE ============
            fig1 = go.Figure()

            # Get top 10 drawdowns
            if len(dd_data) > 0:
                top_10_dd = dd_data[dd_data['rank'] != 'Current'].head(10)

                # Add drawdown shaded regions
                for idx, (_, row) in enumerate(top_10_dd.iterrows()):
                    fig1.add_vrect(
                        x0=row['peak_date'],
                        x1=row['trough_date'],
                        fillcolor=DD_COLORS[idx % len(DD_COLORS)],
                        layer="below",
                        line_width=0
                    )

            # Add price line with custom hover template
            price_df_copy = stock_data.copy()
            price_df_copy['DD_Info'] = ''

            if len(dd_data) > 0:
                top_10_dd = dd_data[dd_data['rank'] != 'Current'].head(10)
                for _, row in top_10_dd.iterrows():
                    mask = (price_df_copy['Date'] >= row['peak_date']) & (price_df_copy['Date'] <= row['trough_date'])
                    price_df_copy.loc[mask, 'DD_Info'] = (
                        f"<br><b>Drawdown #{row['rank']}</b><br>" +
                        f"Depth: {row['depth_pct']:.2f}%<br>" +
                        f"Peak: {row['peak_date'].strftime('%Y-%m-%d')} ${row['peak_price']:.2f}<br>" +
                        f"Trough: {row['trough_date'].strftime('%Y-%m-%d')} ${row['trough_price']:.2f}"
                    )

            line_color = 'black'
            hover_format = 'Price: $%{y:.2f}%{customdata}<extra></extra>'

            fig1.add_trace(go.Scatter(
                x=price_df_copy['Date'],
                y=price_df_copy[price_col],
                mode='lines',
                line=dict(color=line_color, width=2),
                customdata=price_df_copy['DD_Info'],
                hovertemplate='%{x|%Y-%m-%d}<br>' + hover_format,
                showlegend=False,
                hoverlabel=dict(bgcolor='white', bordercolor='lightgray'),
                marker=dict(color='rgba(0,0,0,0)')
            ))

            # Add current drawdown line and shaded area (only for latest period, current holdings, and if Current exists)
            current_dd_rows = dd_data[dd_data['rank'] == 'Current'] if len(dd_data) > 0 else pd.DataFrame()
            if is_latest_period() and is_current and len(current_dd_rows) > 0:
                current_dd = current_dd_rows.iloc[0]
                peak_price = current_dd['peak_price']
                peak_date = current_dd['peak_date']
                current_price = current_dd['trough_price']
                current_dd_pct = current_dd['depth_pct']

                fig1.add_shape(
                    type="line",
                    x0=peak_date,
                    x1=stock_data['Date'].max(),
                    y0=peak_price,
                    y1=peak_price,
                    line=dict(color='red', width=2, dash='dash'),
                    layer='above'
                )

                fig1.add_shape(
                    type="rect",
                    x0=peak_date,
                    x1=stock_data['Date'].max(),
                    y0=current_price,
                    y1=peak_price,
                    fillcolor='rgba(128,128,128,0.25)',
                    line=dict(width=0),
                    layer='below'
                )

                annotation_text = (
                    f"<b>Current Drawdown</b><br>" +
                    f"Depth: {current_dd_pct:.2f}%<br>" +
                    f"Peak: {peak_date.strftime('%Y-%m-%d')} ${peak_price:.2f}<br>" +
                    f"Current: {stock_data['Date'].max().strftime('%Y-%m-%d')} ${current_price:.2f}"
                )

                fig1.add_annotation(
                    text=annotation_text,
                    x=stock_data['Date'].max(),
                    y=(peak_price + current_price) / 2,
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    xshift=10,
                    font=dict(size=10, color='black'),
                    align='left',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.3)',
                    borderwidth=1,
                    borderpad=4
                )

            chart_title = f"{selected_display_ticker} Stock Price with Top 10 Drawdowns & Current Drawdown"

            fig1.update_layout(
                title=chart_title,
                xaxis_title="Date",
                yaxis_title="Stock Price ($)",
                hovermode='x unified',
                height=450,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(gridcolor='lightgray', showgrid=True, range=[start_date, end_date]),
                yaxis=dict(gridcolor='lightgray', showgrid=True),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig1, width='stretch', config=CHART_CONFIG)

            # ============ CHART 2: PEER GROUP ============
            if gics and len(peer_prices) > 0 and len(peer_dd_data) > 0:
                fig2 = go.Figure()

                # Get top 10 drawdowns
                peer_dd_data['peak_date'] = pd.to_datetime(peer_dd_data['peak_date'])
                peer_dd_data['trough_date'] = pd.to_datetime(peer_dd_data['trough_date'])
                peer_top_10 = peer_dd_data[peer_dd_data['rank'] != 'Current'].head(10)

                # Add drawdown shaded regions
                for idx, (_, row) in enumerate(peer_top_10.iterrows()):
                    fig2.add_vrect(
                        x0=row['peak_date'],
                        x1=row['trough_date'],
                        fillcolor=DD_COLORS[idx % len(DD_COLORS)],
                        layer="below",
                        line_width=0
                    )

                # Add price line with custom hover template
                peer_df_copy = peer_prices.copy()
                peer_df_copy['DD_Info'] = ''

                for _, row in peer_top_10.iterrows():
                    mask = (peer_df_copy['Date'] >= row['peak_date']) & (peer_df_copy['Date'] <= row['trough_date'])
                    peer_df_copy.loc[mask, 'DD_Info'] = (
                        f"<br><b>Drawdown #{row['rank']}</b><br>" +
                        f"Depth: {row['depth_pct']:.2f}%<br>" +
                        f"Peak: {row['peak_date'].strftime('%Y-%m-%d')} ${row['peak_price']:,.0f}<br>" +
                        f"Trough: {row['trough_date'].strftime('%Y-%m-%d')} ${row['trough_price']:,.0f}"
                    )

                line_color = 'darkblue'
                hover_format = 'Value: $%{y:,.0f}%{customdata}<extra></extra>'

                fig2.add_trace(go.Scatter(
                    x=peer_df_copy['Date'],
                    y=peer_df_copy['Value'],
                    mode='lines',
                    line=dict(color=line_color, width=2),
                    customdata=peer_df_copy['DD_Info'],
                    hovertemplate='%{x|%Y-%m-%d}<br>' + hover_format,
                    showlegend=False,
                    hoverlabel=dict(bgcolor='white', bordercolor='lightgray'),
                    marker=dict(color='rgba(0,0,0,0)')
                ))

                # Add current drawdown line and shaded area (only for latest period)
                peer_current_dd = peer_dd_data[peer_dd_data['rank'] == 'Current']
                if is_latest_period() and len(peer_current_dd) > 0:
                    peer_current_dd = peer_current_dd.iloc[0]
                    peer_peak_price = peer_current_dd['peak_price']
                    peer_peak_date = peer_current_dd['peak_date']
                    peer_current_price = peer_current_dd['trough_price']
                    peer_current_dd_pct = peer_current_dd['depth_pct']

                    fig2.add_shape(
                        type="line",
                        x0=peer_peak_date,
                        x1=peer_prices['Date'].max(),
                        y0=peer_peak_price,
                        y1=peer_peak_price,
                        line=dict(color='red', width=2, dash='dash'),
                        layer='above'
                    )

                    fig2.add_shape(
                        type="rect",
                        x0=peer_peak_date,
                        x1=peer_prices['Date'].max(),
                        y0=peer_current_price,
                        y1=peer_peak_price,
                        fillcolor='rgba(128,128,128,0.25)',
                        line=dict(width=0),
                        layer='below'
                    )

                    peer_annotation_text = (
                        f"<b>Current Drawdown</b><br>" +
                        f"Depth: {peer_current_dd_pct:.2f}%<br>" +
                        f"Peak: {peer_peak_date.strftime('%Y-%m-%d')} ${peer_peak_price:,.0f}<br>" +
                        f"Current: {peer_prices['Date'].max().strftime('%Y-%m-%d')} ${peer_current_price:,.0f}"
                    )

                    fig2.add_annotation(
                        text=peer_annotation_text,
                        x=peer_prices['Date'].max(),
                        y=(peer_peak_price + peer_current_price) / 2,
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        xshift=10,
                        font=dict(size=10, color='black'),
                        align='left',
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='rgba(0,0,0,0.3)',
                        borderwidth=1,
                        borderpad=4
                    )

                peer_chart_title = f"{gics} - {version} with Top 10 Drawdowns & Current Drawdown"

                fig2.update_layout(
                    title=peer_chart_title,
                    xaxis_title="Date",
                    yaxis_title="Peer Group Value ($)",
                    hovermode='x unified',
                    height=450,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(gridcolor='lightgray', showgrid=True, range=[start_date, end_date]),
                    yaxis=dict(gridcolor='lightgray', showgrid=True),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                add_reconstitution_vlines(fig2, price_line_name=version)
                st.plotly_chart(fig2, width='stretch', config=CHART_CONFIG)

    ""  # Space

    # Current Drawdown Analysis (only for latest period, current holdings, and if Current drawdown exists)
    current_dd_for_analysis = dd_data[dd_data['rank'] == 'Current'] if len(dd_data) > 0 else pd.DataFrame()
    if is_latest_period() and stock_data is not None and len(stock_data) > 0 and len(current_dd_for_analysis) > 0 and is_current:
        st.markdown("### Current Drawdown Analysis")

        current_dd_container = st.container(border=True)
        with current_dd_container:
            st.markdown("#### Current Drawdown Information")

            current_dd = current_dd_for_analysis.iloc[0]
            price_col = get_price_column(stock_data)

            current_price = stock_data[price_col].iloc[-1]
            current_date = stock_data['Date'].iloc[-1]
            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_dd_pct = current_dd['depth_pct']

            drawdown_period = stock_data[stock_data['Date'] >= peak_date]
            actual_trough_price = drawdown_period[price_col].min()
            actual_trough_date = drawdown_period[drawdown_period[price_col] == actual_trough_price]['Date'].iloc[0]

            duration_days = (current_date - peak_date).days

            if peak_price != actual_trough_price:
                recovery_rate = (current_price - actual_trough_price) / (peak_price - actual_trough_price)
            else:
                recovery_rate = 0.0

            current_dd_info = pd.DataFrame([{
                'Peak Date': peak_date.strftime('%Y-%m-%d'),
                'Peak Price': f'${peak_price:.2f}',
                'Trough Date': actual_trough_date.strftime('%Y-%m-%d'),
                'Trough Price': f'${actual_trough_price:.2f}',
                'Current Date': current_date.strftime('%Y-%m-%d'),
                'Current Price': f'${current_price:.2f}',
                'Duration (Days)': duration_days,
                'Drawdown Depth': f'{current_dd_pct:.2f}%',
                'Recovery Rate': f'{recovery_rate * 100:.1f}%'
            }])

            st.dataframe(current_dd_info, hide_index=True, width='stretch')

            ""  # Space

            st.markdown("#### Historical Drawdown Analysis for This Stock")

            st.markdown(f"""
            <small>View historical drawdown records for <b>{selected_ticker}</b> across different depth ranges to understand its performance and recovery patterns at various levels.</small>
            """, unsafe_allow_html=True)

            ""  # Space

            depth_ranges = ['0% to -10%', '-10% to -20%', '-20% to -30%', '-30% to -40%',
                          '-40% to -50%', '-50% to -60%', '-60% to -70%', '-70% to -80%', '< -80%']

            bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
            current_range_idx = 0
            for i in range(len(bins) - 1):
                if bins[i] < current_dd_pct <= bins[i+1]:
                    current_range_idx = len(bins) - 2 - i
                    break

            st.markdown("**Select Drawdown Depth Range:**")
            selected_range = st.pills(
                "Depth Range",
                depth_ranges,
                default=depth_ranges[current_range_idx],
                label_visibility="collapsed"
            )

            with st.spinner(f"Loading {selected_ticker} historical drawdowns for {selected_range}..."):
                range_drawdowns = get_stock_drawdowns_in_depth_range(
                    selected_ticker, selected_etf, selected_range,
                    period_key=get_current_period(), _start_date=start_date, _end_date=end_date
                )

            if len(range_drawdowns) > 0:
                total_events = len(range_drawdowns)
                recovered_events = range_drawdowns['recovered'].sum()
                recovery_probability = recovered_events / total_events if total_events > 0 else 0

                st.markdown(f"""
                **{selected_ticker} - {selected_range} Historical Statistics:**
                - Total Drawdowns: {total_events}
                - Recovered: {recovered_events}
                - **Recovery Rate: {recovery_probability * 100:.1f}%**
                """)

                ""  # Space

                st.markdown(f"**{selected_ticker} - All Historical Drawdowns in {selected_range}:**")

                display_range_dd = range_drawdowns.copy()
                display_range_dd['Peak Date'] = display_range_dd['peak_date'].dt.strftime('%Y-%m-%d')
                display_range_dd['Trough Date'] = display_range_dd['trough_date'].dt.strftime('%Y-%m-%d')
                display_range_dd['Recovery Date'] = display_range_dd['recovery_date'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'Not Recovered'
                )
                display_range_dd['Depth %'] = display_range_dd['depth_pct'].apply(lambda x: f'{x:.2f}%')
                display_range_dd['Peak Price'] = display_range_dd['peak_price'].apply(lambda x: f'${x:.2f}')
                display_range_dd['Trough Price'] = display_range_dd['trough_price'].apply(lambda x: f'${x:.2f}')
                display_range_dd['Recovery Rate'] = display_range_dd['recovery_rate'].apply(lambda x: f'{x * 100:.1f}%')
                display_range_dd['Recovered'] = display_range_dd['recovered'].apply(lambda x: 'Yes' if x else 'No')
                display_range_dd['Days to Recover'] = display_range_dd['days_to_recover'].apply(
                    lambda x: f'{int(x)}' if pd.notna(x) else 'N/A'
                )

                display_cols = ['Peak Date', 'Trough Date', 'duration_days',
                              'Depth %', 'Peak Price', 'Trough Price',
                              'Recovered', 'Recovery Date', 'Days to Recover', 'Recovery Rate']

                display_range_dd = display_range_dd[display_cols]
                display_range_dd = display_range_dd.rename(columns={
                    'duration_days': 'Duration (Days)'
                })

                st.dataframe(
                    display_range_dd,
                    hide_index=True,
                    width='stretch',
                    height=400
                )
            else:
                st.info(f"{selected_ticker} has no historical drawdowns in {selected_range} range.")

    ""  # Space

    # Drawdown Details
    if stock_data is not None and len(stock_data) > 0 and len(dd_data) > 0:
        st.markdown("### Drawdown Details")

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
            text-align: left !important;
        }
        </style>
        """, unsafe_allow_html=True)

        historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

        display_cols = ['rank', 'depth_pct', 'peak_date', 'trough_date', 'peak_price', 'trough_price',
                       'PeerGroup_DD_%', 'Cosine_Similarity']

        display_cols = [col for col in display_cols if col in historical_dd.columns]
        display_df = historical_dd[display_cols].copy()

        display_df['rank'] = display_df['rank'].astype(str)

        if 'peak_date' in display_df.columns:
            display_df['peak_date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
        if 'trough_date' in display_df.columns:
            display_df['trough_date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')

        column_config = {
            "rank": st.column_config.TextColumn("Rank"),
            "depth_pct": st.column_config.NumberColumn("Stock DD %", format="%.2f%%"),
            "peak_date": st.column_config.TextColumn("Peak Date"),
            "trough_date": st.column_config.TextColumn("Trough Date"),
            "peak_price": st.column_config.NumberColumn("Peak Price", format="$%.2f"),
            "trough_price": st.column_config.NumberColumn("Trough Price", format="$%.2f"),
            "PeerGroup_DD_%": st.column_config.NumberColumn("Peer DD %", format="%.2f%%"),
            "Cosine_Similarity": st.column_config.NumberColumn("Similarity", format="%.4f")
        }

        st.dataframe(
            display_df,
            column_config=column_config,
            hide_index=True,
            width='stretch'
        )
