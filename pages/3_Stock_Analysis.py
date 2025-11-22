"""Stock Analysis Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import START_DATE, END_DATE, OUTPUT_DIR, ARK_ETFS
from data_loader import load_ark_holdings, load_industry_info, load_company_name
from peer_group import get_peer_group_prices
from drawdown_calculator import calculate_drawdowns
from chart_config import CHART_CONFIG
from recovery_probability import get_recovery_probability_for_depth, get_drawdowns_in_depth_range

st.set_page_config(
    page_title="Individual Stock vs Peer Group",
    layout="wide"
)

st.title("Individual Stock vs Peer Group")

@st.cache_data
def get_stock_etf_mapping():
    """Get mapping of stocks to their ETFs"""
    from data_loader import load_ark_holdings
    stock_map = {}
    for etf in ARK_ETFS:
        try:
            holdings = load_ark_holdings(etf)
            for ticker in holdings['Ticker'].unique():
                # Skip currency tickers - check Bloomberg Name
                ticker_holdings = holdings[holdings['Ticker'] == ticker]
                if 'Bloomberg Name' in ticker_holdings.columns:
                    bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                    if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                        continue

                if ticker not in stock_map:
                    stock_map[ticker] = []
                full_ticker = holdings[holdings['Ticker'] == ticker]['Ticker'].iloc[0]
                stock_map[ticker].append((etf, full_ticker))
        except:
            continue
    return stock_map

@st.cache_data
def load_all_stocks():
    """Load all unique stocks across ARK ETFs that have data in analysis period"""
    from data_loader import load_ark_holdings

    stock_map = get_stock_etf_mapping()
    valid_tickers = set()

    # Filter stocks that have data in analysis period
    for ticker, etf_list in stock_map.items():
        for etf, full_ticker in etf_list:
            try:
                holdings = load_ark_holdings(etf)
                stock_data = holdings[holdings['Ticker'] == full_ticker].copy()
                stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

                if len(stock_data) >= 30:  # Need at least 30 data points
                    valid_tickers.add(ticker)
                    break
            except:
                continue

    return {k: v for k, v in stock_map.items() if k in valid_tickers}

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

    # Left column: two stacked cards
    with cols[0]:
        # Card 1: Selection Controls
        selection_card = st.container(border=True)
        with selection_card:
            st.markdown("##### ETF")
            # Let user select ETF first
            selected_etf = st.selectbox("Select ETF", ARK_ETFS, label_visibility="collapsed")

            st.markdown("##### Select Stock")
            # Get stocks that have data in the selected ETF during analysis period
            @st.cache_data
            def get_stocks_for_etf(etf):
                from data_loader import load_ark_holdings
                holdings = load_ark_holdings(etf)

                # Get latest date holdings to identify current positions
                latest_date = holdings['Date'].max()
                current_holdings = holdings[holdings['Date'] == latest_date]['Ticker'].unique()

                valid_stocks = []
                stock_ticker_map = {}  # Maps display name to actual ticker

                for ticker in holdings['Ticker'].unique():
                    # Skip currency tickers - check Bloomberg Name
                    ticker_holdings = holdings[holdings['Ticker'] == ticker]
                    if 'Bloomberg Name' in ticker_holdings.columns:
                        bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                        if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                            continue

                    stock_data = holdings[holdings['Ticker'] == ticker].copy()
                    stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]
                    if len(stock_data) >= 30:
                        ticker_simple = ticker.split()[0] if isinstance(ticker, str) else ticker

                        # Check if in current holdings
                        is_current = ticker in current_holdings

                        if is_current:
                            display_name = ticker_simple
                        else:
                            display_name = f"{ticker_simple} (Non-current)"

                        valid_stocks.append((ticker_simple, display_name))
                        stock_ticker_map[display_name] = ticker_simple

                # Sort by ticker_simple (first element) alphabetically
                valid_stocks.sort(key=lambda x: x[0])

                # Extract just the display names
                valid_stocks = [display_name for _, display_name in valid_stocks]

                return valid_stocks, stock_ticker_map

            stock_list, stock_ticker_map = get_stocks_for_etf(selected_etf)
            selected_display_ticker = st.selectbox("Select Stock", stock_list, label_visibility="collapsed")

            # Get the actual ticker from display name
            selected_ticker = stock_ticker_map.get(selected_display_ticker, selected_display_ticker.replace(" (Non-current)", ""))

            st.markdown("##### Peer Group Version")
            # Peer group version pills
            version = st.pills(
                "Version",
                ["Market Value", "Weighted Price"],
                default="Market Value",
                label_visibility="collapsed"
            )
            version_param = "mv" if version == "Market Value" else "weighted"

        @st.cache_data
        def load_stock_data(ticker, etf):
            """Load stock data from ARK holdings"""
            from data_loader import load_ark_holdings

            holdings = load_ark_holdings(etf)

            # Find the full ticker (e.g., "PD" or "PD US Equity")
            matching_tickers = holdings[holdings['Ticker'].str.startswith(ticker + ' ', na=False) |
                                       (holdings['Ticker'] == ticker)]['Ticker'].unique()

            if len(matching_tickers) == 0:
                return None, None, None, False

            full_ticker = matching_tickers[0]
            stock_data = holdings[holdings['Ticker'] == full_ticker].copy()
            stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

            bloomberg_name = stock_data['Bloomberg Name'].iloc[0] if len(stock_data) > 0 else None

            # Check if this stock is in current holdings
            latest_date = holdings['Date'].max()
            current_holdings = holdings[holdings['Date'] == latest_date]['Ticker'].unique()
            is_current = full_ticker in current_holdings

            return stock_data, full_ticker, bloomberg_name, is_current

        stock_data, full_ticker, bloomberg_name, is_current = load_stock_data(selected_ticker, selected_etf)

        if stock_data is None or len(stock_data) == 0:
            st.error(f"No data available for {selected_ticker} in {selected_etf}")
        else:
            # Calculate stock drawdowns directly from price data
            # Determine which price column has actual data
            if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                price_col = 'YFinance Close Price'
            else:
                price_col = 'Stock_Price'

            # Prepare price dataframe for drawdown calculation
            price_df = stock_data[['Date', price_col]].copy()
            price_df.columns = ['Date', 'Close']

            # Calculate drawdowns
            dd_data = calculate_drawdowns(price_df)

            # Get GICS industry
            industry_dict = load_industry_info(source='ark')
            gics = industry_dict.get(bloomberg_name) if bloomberg_name else None

            # Load peer group data
            if gics:
                try:
                    peer_prices = get_peer_group_prices(gics, version=version_param)
                    peer_prices = peer_prices[(peer_prices['Date'] >= START_DATE) & (peer_prices['Date'] <= END_DATE)]
                except:
                    peer_prices = pd.DataFrame()
                    gics = None
            else:
                peer_prices = pd.DataFrame()

            # Card 2: Key Metrics
            if len(dd_data) > 0:
                metrics_card = st.container(border=True)
                with metrics_card:
                    st.markdown("##### Key Metrics")

                    current_dd = dd_data[dd_data['rank'] == 'Current'].iloc[0]
                    top_dd = dd_data[dd_data['rank'] == '1'].iloc[0]

                    # Calculate RoMaD
                    # Check which price column has actual data
                    if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                        price_col = 'YFinance Close Price'
                    else:
                        price_col = 'Stock_Price'

                    first_price = stock_data[price_col].iloc[0]
                    last_price = stock_data[price_col].iloc[-1]
                    overall_return = ((last_price - first_price) / first_price) * 100
                    max_dd_abs = abs(top_dd['depth_pct'])
                    romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

                    # Company Name (at top)
                    try:
                        company_names = load_company_name(source='ark')
                        company_name = company_names.get(selected_ticker)
                        if company_name:
                            st.markdown(f"<small>Company Name</small><br><b>{company_name}</b>", unsafe_allow_html=True)
                    except:
                        pass

                    # GICS Industry Group (at top)
                    if gics:
                        st.markdown(f"<small>GICS Industry Group</small><br><b>{gics}</b>", unsafe_allow_html=True)

                    # Max Drawdown
                    st.markdown(f"<small>Max Drawdown</small><br><b>{top_dd['depth_pct']:.2f}%</b>", unsafe_allow_html=True)

                    cols_price = st.columns(2)
                    with cols_price[0]:
                        # Check which price column has actual data
                        if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                            current_price = stock_data['YFinance Close Price'].iloc[-1]
                        else:
                            current_price = stock_data['Stock_Price'].iloc[-1]
                        st.markdown(f"<small>Current Price</small><br><b>${current_price:.2f}</b>", unsafe_allow_html=True)
                    with cols_price[1]:
                        # Check which price column has actual data
                        if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                            peak_price = stock_data['YFinance Close Price'].max()
                        else:
                            peak_price = stock_data['Stock_Price'].max()
                        st.markdown(f"<small>Peak Price</small><br><b>${peak_price:.2f}</b>", unsafe_allow_html=True)

                    # RoMaD
                    st.markdown(f"<small>RoMaD</small><br><b>{romad:.2f}</b>", unsafe_allow_html=True)

    # Right column: two charts stacked
    if stock_data is not None and len(stock_data) > 0:
        right_panel = cols[1].container(border=True)

        with right_panel:
            # Determine which price column has actual data
            if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                price_col = 'YFinance Close Price'
            else:
                price_col = 'Stock_Price'

            # ============ CHART 1: STOCK PRICE ============
            # Create figure with drawdown regions (COPIED FROM RUSSELL 3000)
            fig1 = go.Figure()

            # Get top 10 drawdowns
            if len(dd_data) > 0:
                top_10_dd = dd_data[dd_data['rank'] != 'Current'].head(10)

                # Color palette for drawdowns
                dd_colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                             'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                             'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                             'rgba(255, 228, 181, 0.3)']

                # Add drawdown shaded regions
                for idx, (_, row) in enumerate(top_10_dd.iterrows()):
                    fig1.add_vrect(
                        x0=row['peak_date'],
                        x1=row['trough_date'],
                        fillcolor=dd_colors[idx % len(dd_colors)],
                        layer="below",
                        line_width=0
                    )

            # Add price line with custom hover template
            price_df_copy = stock_data.copy()
            price_df_copy['DD_Info'] = ''

            # Add drawdown info for each date
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

            # Line color is black for stock
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

            # Add current drawdown line and shaded area (only for current holdings)
            if is_current and len(dd_data) > 0:
                current_dd = dd_data[dd_data['rank'] == 'Current'].iloc[0]
                peak_price = current_dd['peak_price']
                peak_date = current_dd['peak_date']
                current_price = current_dd['trough_price']
                current_dd_pct = current_dd['depth_pct']

                # Add horizontal line from peak date to the end of the chart
                fig1.add_shape(
                    type="line",
                    x0=peak_date,
                    x1=stock_data['Date'].max(),
                    y0=peak_price,
                    y1=peak_price,
                    line=dict(color='red', width=2, dash='dash'),
                    layer='above'
                )

                # Add shaded rectangle from peak date to current date
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

                # Add text annotation on the right side showing current drawdown
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
                xaxis=dict(gridcolor='lightgray', showgrid=True),
                yaxis=dict(gridcolor='lightgray', showgrid=True),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            st.plotly_chart(fig1, width='stretch', config=CHART_CONFIG)

            # ============ CHART 2: PEER GROUP ============
            if gics and len(peer_prices) > 0:
                # Calculate peer group drawdowns dynamically (same as Russell 3000 page)
                from drawdown_calculator import calculate_drawdowns
                peer_prices_for_dd = peer_prices.copy()
                peer_prices_for_dd = peer_prices_for_dd.rename(columns={'Value': 'Close'})
                peer_dd_data = calculate_drawdowns(peer_prices_for_dd)

                # Create figure with drawdown regions (COPIED FROM RUSSELL 3000)
                fig2 = go.Figure()

                # Get top 10 drawdowns
                if len(peer_dd_data) > 0:
                    peer_dd_data['peak_date'] = pd.to_datetime(peer_dd_data['peak_date'])
                    peer_dd_data['trough_date'] = pd.to_datetime(peer_dd_data['trough_date'])
                    peer_top_10 = peer_dd_data[peer_dd_data['rank'] != 'Current'].head(10)

                    # Color palette for drawdowns
                    dd_colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                                 'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                                 'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                                 'rgba(255, 228, 181, 0.3)']

                    # Add drawdown shaded regions
                    for idx, (_, row) in enumerate(peer_top_10.iterrows()):
                        fig2.add_vrect(
                            x0=row['peak_date'],
                            x1=row['trough_date'],
                            fillcolor=dd_colors[idx % len(dd_colors)],
                            layer="below",
                            line_width=0
                        )

                # Add price line with custom hover template
                peer_df_copy = peer_prices.copy()
                peer_df_copy['DD_Info'] = ''

                # Add drawdown info for each date
                if len(peer_dd_data) > 0:
                    peer_top_10 = peer_dd_data[peer_dd_data['rank'] != 'Current'].head(10)
                    for _, row in peer_top_10.iterrows():
                        mask = (peer_df_copy['Date'] >= row['peak_date']) & (peer_df_copy['Date'] <= row['trough_date'])
                        peer_df_copy.loc[mask, 'DD_Info'] = (
                            f"<br><b>Drawdown #{row['rank']}</b><br>" +
                            f"Depth: {row['depth_pct']:.2f}%<br>" +
                            f"Peak: {row['peak_date'].strftime('%Y-%m-%d')} ${row['peak_price']:,.0f}<br>" +
                            f"Trough: {row['trough_date'].strftime('%Y-%m-%d')} ${row['trough_price']:,.0f}"
                        )

                # Line color is darkblue for peer group
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

                # Add current drawdown line and shaded area
                if len(peer_dd_data) > 0:
                    peer_current_dd = peer_dd_data[peer_dd_data['rank'] == 'Current'].iloc[0]
                    peer_peak_price = peer_current_dd['peak_price']
                    peer_peak_date = peer_current_dd['peak_date']
                    peer_current_price = peer_current_dd['trough_price']
                    peer_current_dd_pct = peer_current_dd['depth_pct']

                    # Add horizontal line from peak date to the end of the chart
                    fig2.add_shape(
                        type="line",
                        x0=peer_peak_date,
                        x1=peer_prices['Date'].max(),
                        y0=peer_peak_price,
                        y1=peer_peak_price,
                        line=dict(color='red', width=2, dash='dash'),
                        layer='above'
                    )

                    # Add shaded rectangle from peak date to current date
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

                    # Add text annotation on the right side showing current drawdown
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
                    xaxis=dict(gridcolor='lightgray', showgrid=True),
                    yaxis=dict(gridcolor='lightgray', showgrid=True),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig2, width='stretch', config=CHART_CONFIG)

    ""  # Space

    # Current Drawdown Analysis (only for current holdings)
    if stock_data is not None and len(stock_data) > 0 and len(dd_data) > 0 and is_current:
        st.markdown("### Current Drawdown Analysis")

        current_dd_container = st.container(border=True)
        with current_dd_container:
            st.markdown("#### Current Drawdown Information")

            current_dd = dd_data[dd_data['rank'] == 'Current'].iloc[0]

            # Determine which price column to use
            if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                price_col = 'YFinance Close Price'
            else:
                price_col = 'Stock_Price'

            current_price = stock_data[price_col].iloc[-1]
            current_date = stock_data['Date'].iloc[-1]
            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_dd_pct = current_dd['depth_pct']

            # For Current drawdown, find the actual trough (lowest price) from peak to now
            drawdown_period = stock_data[stock_data['Date'] >= peak_date]
            actual_trough_price = drawdown_period[price_col].min()
            actual_trough_date = drawdown_period[drawdown_period[price_col] == actual_trough_price]['Date'].iloc[0]

            # Calculate duration
            duration_days = (current_date - peak_date).days

            # Calculate Recovery Rate
            if peak_price != actual_trough_price:
                recovery_rate = (current_price - actual_trough_price) / (peak_price - actual_trough_price)
            else:
                recovery_rate = 0.0

            # Create current drawdown info DataFrame
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

            st.dataframe(current_dd_info, hide_index=True, use_container_width=True)

            ""  # Space

            st.markdown("#### Recovery Probability Analysis")

            st.markdown("""
            <small>Recovery Probability是基于历史统计数据，表示在相同跌幅区间内的股票最终回到前高的概率。
            选择一个跌幅区间来查看该区间内所有历史drawdown的详细信息。</small>
            """, unsafe_allow_html=True)

            ""  # Space

            # Depth range selector
            depth_ranges = ['0% to -10%', '-10% to -20%', '-20% to -30%', '-30% to -40%',
                          '-40% to -50%', '-50% to -60%', '-60% to -70%', '-70% to -80%', '< -80%']

            # Determine default selection based on current drawdown depth
            bins = [0, -10, -20, -30, -40, -50, -60, -70, -80, -float('inf')]
            current_range_idx = 0
            for i in range(len(bins) - 1):
                if bins[i] >= current_dd_pct > bins[i+1]:
                    current_range_idx = i
                    break

            selected_range = st.selectbox(
                "Select Drawdown Depth Range",
                depth_ranges,
                index=current_range_idx
            )

            # Get drawdowns in selected range
            with st.spinner(f"Loading historical drawdowns for {selected_range}..."):
                range_drawdowns = get_drawdowns_in_depth_range(selected_range)

            if len(range_drawdowns) > 0:
                # Calculate recovery probability for this range
                total_events = len(range_drawdowns)
                recovered_events = range_drawdowns['recovered'].sum()
                recovery_probability = recovered_events / total_events if total_events > 0 else 0

                # Display recovery probability stats
                st.markdown(f"""
                **{selected_range} Recovery Statistics:**
                - Total Events: {total_events}
                - Recovered Events: {recovered_events}
                - **Recovery Probability: {recovery_probability * 100:.1f}%**
                """)

                ""  # Space

                # Display detailed table
                st.markdown(f"**All Historical Drawdowns in {selected_range}:**")

                # Format the dataframe for display
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

                # Select and rename columns
                display_cols = ['ticker', 'etf', 'Peak Date', 'Trough Date', 'duration_days',
                              'Depth %', 'Peak Price', 'Trough Price',
                              'Recovered', 'Recovery Date', 'Days to Recover', 'Recovery Rate']

                display_range_dd = display_range_dd[display_cols]
                display_range_dd = display_range_dd.rename(columns={
                    'ticker': 'Ticker',
                    'etf': 'ETF',
                    'duration_days': 'Duration (Days)'
                })

                st.dataframe(
                    display_range_dd,
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info(f"No historical drawdowns found in {selected_range} range.")

    ""  # Space

    # Drawdown Details
    if stock_data is not None and len(stock_data) > 0 and len(dd_data) > 0:
        st.markdown("### Drawdown Details")

        # Add CSS for left alignment
        st.markdown("""
        <style>
        [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
            text-align: left !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Filter out Current drawdown (shown in separate module above)
        historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

        # Select columns to display
        display_cols = ['rank', 'depth_pct', 'peak_date', 'trough_date', 'peak_price', 'trough_price',
                       'PeerGroup_DD_%', 'Cosine_Similarity']

        # Filter to only columns that exist
        display_cols = [col for col in display_cols if col in historical_dd.columns]

        display_df = historical_dd[display_cols].copy()

        # Convert rank to string to avoid mixed type issues with PyArrow
        display_df['rank'] = display_df['rank'].astype(str)

        # Format dates
        if 'peak_date' in display_df.columns:
            display_df['peak_date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
        if 'trough_date' in display_df.columns:
            display_df['trough_date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')

        # Configure column display
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
