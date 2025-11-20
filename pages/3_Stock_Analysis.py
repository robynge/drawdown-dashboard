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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_recovery_probability(price_series_hash, current_price, target_price, years=2, n_simulations=10000):
    """
    Estimate probability of recovering to target price within T years using GBM and Monte Carlo simulation.

    Parameters:
    - price_series_hash: tuple of (price values, length) for caching
    - current_price: current price S0
    - target_price: target peak price B
    - years: time horizon in years (default 2)
    - n_simulations: number of Monte Carlo paths (default 10000)

    Returns:
    - probability: fraction of paths that hit target
    """
    # Reconstruct price series from hash
    price_values = price_series_hash[0]
    price_series = pd.Series(price_values)

    if len(price_series) < 2 or current_price >= target_price:
        return 1.0 if current_price >= target_price else 0.0

    # Calculate log returns
    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    if len(log_returns) < 2:
        return 0.0

    # Estimate μ (drift) and σ (volatility) from historical data
    mu = log_returns.mean() * 252  # Annualized drift
    sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility

    # Time parameters
    dt = 1/252  # Daily time step
    n_steps = int(years * 252)  # Number of trading days

    # Vectorized Monte Carlo simulation (much faster!)
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal((n_simulations, n_steps))

    # Simulate all paths at once using vectorization
    price_paths = current_price * np.exp(
        np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    )

    # Check if any point in each path hits target
    recovery_count = np.sum(np.any(price_paths >= target_price, axis=1))

    probability = recovery_count / n_simulations
    return probability

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
                    top_dd = dd_data[dd_data['rank'] == 1].iloc[0]

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

                    cols_metrics = st.columns(2)
                    with cols_metrics[0]:
                        st.markdown(f"<small>Current Drawdown</small><br><b>{current_dd['depth_pct']:.2f}%</b>", unsafe_allow_html=True)
                    with cols_metrics[1]:
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
                xaxis=dict(gridcolor='lightgray', showgrid=True, range=[START_DATE, END_DATE]),
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
                    xaxis=dict(gridcolor='lightgray', showgrid=True, range=[START_DATE, END_DATE]),
                    yaxis=dict(gridcolor='lightgray', showgrid=True),
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig2, width='stretch', config=CHART_CONFIG)

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

        # Select columns to display
        display_cols = ['rank', 'depth_pct', 'peak_date', 'trough_date', 'peak_price', 'trough_price',
                       'PeerGroup_DD_%', 'Cosine_Similarity']

        # Filter to only columns that exist
        display_cols = [col for col in display_cols if col in dd_data.columns]

        display_df = dd_data[display_cols].copy()

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

# ============ CURRENT DRAWDOWN & RECOVERY RATE ============
st.markdown("---")
st.subheader("Current Drawdown & Recovery Rate")

# ETF Selection and Time Horizon for recovery table
cols_selection = st.columns([2, 1])

with cols_selection[0]:
    st.markdown("##### Select ETF")
    recovery_etf = st.selectbox(
        "Select ETF for portfolio analysis",
        ARK_ETFS,
        index=0,
        label_visibility="collapsed",
        key="recovery_etf_selector"
    )

with cols_selection[1]:
    st.markdown("##### Recovery Time Horizon (Months)")
    recovery_months = st.selectbox(
        "Select months for recovery probability",
        options=list(range(1, 25)),
        index=23,  # Default to 24 months
        label_visibility="collapsed",
        key="recovery_months_selector"
    )

@st.cache_data
def calculate_all_stocks_recovery(etf, months):
    """Calculate current drawdown and recovery rate for all stocks in selected ETF"""
    holdings = load_ark_holdings(etf)

    # Load company name mapping
    company_names = load_company_name(source='ark')

    # Get latest date holdings for weight
    latest_date = holdings['Date'].max()
    latest_holdings = holdings[holdings['Date'] == latest_date]

    recovery_data = []

    for ticker in holdings['Ticker'].unique():
        try:
            # Skip currency tickers - check Bloomberg Name
            ticker_holdings = holdings[holdings['Ticker'] == ticker]
            if 'Bloomberg Name' in ticker_holdings.columns:
                bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                    continue

            # Get stock data
            stock_data = holdings[holdings['Ticker'] == ticker].copy()
            stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

            if len(stock_data) < 30:  # Skip stocks with insufficient data
                continue

            # Get Bloomberg Name and Weight from latest holdings
            latest_ticker_data = latest_holdings[latest_holdings['Ticker'] == ticker]
            if len(latest_ticker_data) == 0:
                continue  # Stock not in latest holdings

            bloomberg_name = latest_ticker_data['Bloomberg Name'].iloc[0] if 'Bloomberg Name' in latest_ticker_data.columns else ticker
            weight = latest_ticker_data['Weight'].iloc[0] if 'Weight' in latest_ticker_data.columns else 0

            # Calculate price (use YFinance Close Price if available, otherwise use Market Value / Position)
            if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().sum() > 0:
                stock_data['Price'] = stock_data['YFinance Close Price']
            else:
                stock_data['Price'] = stock_data['Market Value'] / stock_data['Position']

            stock_data = stock_data.sort_values('Date')

            # Calculate drawdowns
            stock_for_dd = stock_data[['Date', 'Price']].copy()
            stock_for_dd = stock_for_dd.rename(columns={'Price': 'Close'})
            dd_df = calculate_drawdowns(stock_for_dd)

            if len(dd_df) == 0:
                continue

            # Get current drawdown
            current_dd = dd_df[dd_df['rank'] == 'Current'].iloc[0]

            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_price = stock_data['Price'].iloc[-1]  # Last available price
            current_dd_pct = current_dd['depth_pct']

            # Find the minimum price during the current drawdown period (from peak to now)
            drawdown_period = stock_data[stock_data['Date'] >= peak_date]
            if len(drawdown_period) > 0:
                min_price_in_period = drawdown_period['Price'].min()
            else:
                min_price_in_period = current_price

            # Calculate Recovery Rate = (Current Price - Min Price) / (Peak Price - Min Price)
            if peak_price > min_price_in_period:
                recovery_rate = ((current_price - min_price_in_period) / (peak_price - min_price_in_period)) * 100
            else:
                recovery_rate = 0

            trough_price = min_price_in_period  # Use the actual minimum price in the period

            # Calculate Recovery Probability using Monte Carlo simulation
            years_horizon = months / 12.0  # Convert months to years
            # Create hashable version of price series for caching
            price_series_hash = (tuple(stock_data['Price'].values), len(stock_data['Price']))
            recovery_prob = calculate_recovery_probability(
                price_series_hash=price_series_hash,
                current_price=current_price,
                target_price=peak_price,
                years=years_horizon
            ) * 100  # Convert to percentage

            # Get ticker symbol for company name lookup
            ticker_symbol = ticker.split()[0] if isinstance(ticker, str) else ticker
            company_name = company_names.get(ticker_symbol, '')

            recovery_data.append({
                'Ticker': bloomberg_name,
                'Company Name': company_name,
                'Current Weight %': weight * 100,  # Convert to percentage
                'Current Price': current_price,
                'Peak Price': peak_price,
                'Trough Price': trough_price,
                'Current DD %': current_dd_pct,
                'Recovery Rate %': recovery_rate,
                'Recovery Probability %': recovery_prob
            })

        except Exception as e:
            continue

    if len(recovery_data) == 0:
        return pd.DataFrame(), latest_date

    recovery_df = pd.DataFrame(recovery_data)
    # Sort by Current DD % - smallest value first (most negative to least: -50%, -20%, -10%, -5%)
    recovery_df = recovery_df.sort_values('Current DD %', ascending=True)

    return recovery_df, latest_date

# Load recovery data for selected ETF
with st.spinner(f"Calculating recovery rates for all stocks in {recovery_etf}..."):
    recovery_df, latest_date = calculate_all_stocks_recovery(recovery_etf, recovery_months)

""  # Add space

if len(recovery_df) > 0:
    st.markdown(f"**ETF:** {recovery_etf} | **Total Stocks:** {len(recovery_df)} | **Holdings Date:** {latest_date.strftime('%Y-%m-%d')} | **Time Horizon:** {recovery_months} months")

    # Add CSS for left alignment
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        recovery_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Stock"),
            "Company Name": st.column_config.TextColumn("Company Name"),
            "Current Weight %": st.column_config.NumberColumn("Current Weight %", format="%.2f%%"),
            "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "Peak Price": st.column_config.NumberColumn("Peak Price", format="$%.2f"),
            "Trough Price": st.column_config.NumberColumn("Trough Price", format="$%.2f"),
            "Current DD %": st.column_config.NumberColumn("Current DD %", format="%.2f%%"),
            "Recovery Rate %": st.column_config.NumberColumn("Recovery Rate %", format="%.2f%%"),
            "Recovery Probability %": st.column_config.NumberColumn(f"Recovery Prob ({recovery_months}M) %", format="%.2f%%")
        },
        hide_index=True,
        width='stretch',
        height=600
    )
else:
    st.info(f"No stock data available for {recovery_etf}")
