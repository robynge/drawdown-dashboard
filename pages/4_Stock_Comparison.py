"""ARK vs Russell 3000 Stock Comparison"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE, INPUT_DIR
from data_loader import load_ark_holdings, load_r3000_holdings, load_company_name, get_r3000_ticker_list, get_ark_ticker_list, get_ark_files_hash, get_r3000_files_hash
from drawdown_calculator import calculate_drawdowns
from chart_config import CHART_CONFIG, DD_COLORS

st.set_page_config(
    page_title="Stock Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
# ARK vs Russell 3000 Stock Comparison

Compare drawdown patterns between ARK ETF holdings and Russell 3000 constituents.
"""

st.markdown(f"**Analysis Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

""  # Add space

@st.cache_data
def get_ark_stock_list(_files_hash, etf):
    """Get list of stocks from specified ARK ETF (fast - uses precomputed ticker list)"""
    try:
        # Use fast ticker list instead of loading full holdings
        tickers = get_ark_ticker_list(etf)

        # Load company name mapping once
        company_name_dict = load_company_name('ark')

        # Get unique tickers with company names
        stock_info = []
        for ticker in tickers:
            company_name = company_name_dict.get(ticker, ticker)
            if company_name and company_name != ticker:
                display_name = f"{ticker} - {company_name}"
            else:
                display_name = ticker
            stock_info.append((ticker, display_name))

        # Sort by ticker
        stock_info.sort(key=lambda x: x[0])
        return stock_info
    except Exception as e:
        st.error(f"Error loading ARK stock list: {e}")
        return []

@st.cache_data
def get_r3000_stock_list(_files_hash):
    """Get list of stocks from Russell 3000 (fast - uses precomputed ticker list)"""
    try:
        # Use fast ticker list instead of loading full holdings
        tickers = get_r3000_ticker_list()

        # Load company name mapping once
        company_name_dict = load_company_name('r3000')

        # Get unique tickers with company names
        stock_info = []
        for ticker in tickers:
            company_name = company_name_dict.get(ticker, ticker)
            if company_name and company_name != ticker:
                display_name = f"{ticker} - {company_name}"
            else:
                display_name = ticker
            stock_info.append((ticker, display_name))

        # Sort by ticker
        stock_info.sort(key=lambda x: x[0])
        return stock_info
    except Exception as e:
        st.error(f"Error loading Russell 3000 stock list: {e}")
        return []

def load_ark_stock_prices(holdings, ticker):
    """Load price data for ARK stock from cached holdings"""
    try:
        stock_data = holdings[holdings['Ticker'] == ticker].copy()
        stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

        if len(stock_data) == 0:
            return pd.DataFrame()

        # Rename columns to match expected format
        stock_data = stock_data.rename(columns={'Stock_Price': 'Close'})
        stock_data = stock_data[['Date', 'Close']].sort_values('Date')

        return stock_data
    except Exception as e:
        st.error(f"Error loading ARK stock prices: {e}")
        return pd.DataFrame()

def load_r3000_stock_prices(holdings, ticker):
    """Load price data for Russell 3000 stock from cached holdings"""
    try:
        stock_data = holdings[holdings['Ticker'] == ticker].copy()
        stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

        if len(stock_data) == 0:
            return pd.DataFrame()

        # Rename columns to match expected format (Russell 3000 uses 'Price', not 'Stock_Price')
        stock_data = stock_data.rename(columns={'Price': 'Close'})
        stock_data = stock_data[['Date', 'Close']].sort_values('Date')

        return stock_data
    except Exception as e:
        st.error(f"Error loading Russell 3000 stock prices: {e}")
        return pd.DataFrame()

def create_stock_chart(price_df, dd_data, stock_name):
    """Create price chart with drawdown regions (matching ETF_Analysis.py style)"""
    fig = go.Figure()

    # Get top 10 drawdowns
    if len(dd_data) > 0:
        top_10_dd = dd_data[dd_data['rank'] != 'Current'].head(10)

        # Add drawdown shaded regions
        for idx, (_, row) in enumerate(top_10_dd.iterrows()):
            fig.add_vrect(
                x0=row['peak_date'],
                x1=row['trough_date'],
                fillcolor=DD_COLORS[idx % len(DD_COLORS)],
                layer="below",
                line_width=0
            )

    # Add price line with custom hover template
    price_df_copy = price_df.copy()
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

    fig.add_trace(go.Scatter(
        x=price_df_copy['Date'],
        y=price_df_copy['Close'],
        mode='lines',
        line=dict(color='black', width=2),
        customdata=price_df_copy['DD_Info'],
        hovertemplate='%{x|%Y-%m-%d}<br>' +
                      'Price: $%{y:.2f}%{customdata}<extra></extra>',
        showlegend=False,
        hoverlabel=dict(bgcolor='white', bordercolor='lightgray'),
        marker=dict(color='rgba(0,0,0,0)')
    ))

    # Add current drawdown line and shaded area
    if len(dd_data) > 0:
        current_dd = dd_data[dd_data['rank'] == 'Current'].iloc[0]
        peak_price = current_dd['peak_price']
        peak_date = current_dd['peak_date']
        current_price = current_dd['trough_price']
        current_dd_pct = current_dd['depth_pct']

        # Add horizontal line from peak date to the end of the chart
        fig.add_shape(
            type="line",
            x0=peak_date,
            x1=price_df['Date'].max(),
            y0=peak_price,
            y1=peak_price,
            line=dict(color='red', width=2, dash='dash'),
            layer='above'
        )

        # Add shaded rectangle from peak date to current date
        fig.add_shape(
            type="rect",
            x0=peak_date,
            x1=price_df['Date'].max(),
            y0=current_price,
            y1=peak_price,
            fillcolor='rgba(128,128,128,0.25)',
            line=dict(width=0),
            layer='below'
        )

        # Add text annotation on the right side showing current drawdown
        fig.add_annotation(
            text=f"<b>Current Drawdown</b><br>" +
                 f"Depth: {current_dd_pct:.2f}%<br>" +
                 f"Peak: {peak_date.strftime('%Y-%m-%d')} ${peak_price:.2f}<br>" +
                 f"Current: {price_df['Date'].max().strftime('%Y-%m-%d')} ${current_price:.2f}",
            x=price_df['Date'].max(),
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

    fig.update_layout(
        title=f"{stock_name} Price with Top 10 Drawdowns & Current Drawdown",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=650,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='lightgray', showgrid=True),
        yaxis=dict(gridcolor='lightgray', showgrid=True),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

# Main section
st.subheader("Stock Selection")

# Layout: left column for controls, right column for charts
cols = st.columns([1, 3])

left_panel = cols[0].container(border=True)
right_panel = cols[1].container(border=True)

ark_ticker = None
r3000_ticker = None
ark_price_df = None
r3000_price_df = None
ark_dd = None
r3000_dd = None

with left_panel:
    st.markdown("### Select Stocks")

    # Step 1: Select ARK ETF
    selected_etf = st.pills(
        "ETF",
        options=ARK_ETFS,
        default=ARK_ETFS[0],
        label_visibility="visible"
    )

    ""  # Space

    # Step 2: Select ARK stock
    ark_files_hash = get_ark_files_hash()
    ark_stocks = get_ark_stock_list(ark_files_hash, selected_etf)

    if len(ark_stocks) > 0:
        ark_stock_options = {display: ticker for ticker, display in ark_stocks}
        selected_ark_display = st.selectbox(
            f"{selected_etf} Stock",
            options=list(ark_stock_options.keys()),
            key="ark_stock_selector"
        )
        ark_ticker = ark_stock_options[selected_ark_display]
    else:
        st.warning(f"No stocks found in {selected_etf}")

    ""  # Space

    # Step 3: Select Russell 3000 stock
    r3000_files_hash = get_r3000_files_hash()
    r3000_stocks = get_r3000_stock_list(r3000_files_hash)

    if len(r3000_stocks) > 0:
        r3000_stock_options = {display: ticker for ticker, display in r3000_stocks}
        selected_r3000_display = st.selectbox(
            "Russell 3000 Stock",
            options=list(r3000_stock_options.keys()),
            key="r3000_stock_selector"
        )
        r3000_ticker = r3000_stock_options[selected_r3000_display]
    else:
        st.warning("No stocks found in Russell 3000")

    ""  # Space

    # Load data and calculate drawdowns if both stocks selected
    if ark_ticker and r3000_ticker:
        with st.spinner("Loading stock data..."):
            # Use cached ARK holdings for fast ticker lookup
            ark_holdings = load_ark_holdings(ark_files_hash, selected_etf)
            ark_price_df = load_ark_stock_prices(ark_holdings, ark_ticker)
            # Use cached R3000 holdings for fast ticker lookup
            r3000_holdings = load_r3000_holdings(r3000_files_hash)
            r3000_price_df = load_r3000_stock_prices(r3000_holdings, r3000_ticker)

            if len(ark_price_df) > 0:
                ark_dd = calculate_drawdowns(ark_price_df, ticker=ark_ticker)

            if len(r3000_price_df) > 0:
                r3000_dd = calculate_drawdowns(r3000_price_df, ticker=r3000_ticker)

    ""  # Space

    # Display metrics
    if ark_price_df is not None and len(ark_price_df) > 0 and ark_dd is not None and len(ark_dd) > 0:
        st.markdown(f"### {ark_ticker} Metrics")

        max_dd = ark_dd[ark_dd['rank'] != 'Current'].iloc[0] if len(ark_dd[ark_dd['rank'] != 'Current']) > 0 else None

        if max_dd is not None:
            # Calculate RoMaD
            first_price = ark_price_df['Close'].iloc[0]
            last_price = ark_price_df['Close'].iloc[-1]
            overall_return = ((last_price - first_price) / first_price) * 100
            max_dd_abs = abs(max_dd['depth_pct'])
            romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

            st.metric("Max Drawdown", f"{max_dd['depth_pct']:.2f}%")
            st.metric("RoMaD", f"{romad:.2f}")
            st.metric("Current Price", f"${ark_price_df['Close'].iloc[-1]:.2f}")
            st.metric("Peak Price", f"${ark_price_df['Close'].max():.2f}")

    ""  # Space

    if r3000_price_df is not None and len(r3000_price_df) > 0 and r3000_dd is not None and len(r3000_dd) > 0:
        st.markdown(f"### {r3000_ticker} Metrics")

        max_dd = r3000_dd[r3000_dd['rank'] != 'Current'].iloc[0] if len(r3000_dd[r3000_dd['rank'] != 'Current']) > 0 else None

        if max_dd is not None:
            # Calculate RoMaD
            first_price = r3000_price_df['Close'].iloc[0]
            last_price = r3000_price_df['Close'].iloc[-1]
            overall_return = ((last_price - first_price) / first_price) * 100
            max_dd_abs = abs(max_dd['depth_pct'])
            romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

            st.metric("Max Drawdown", f"{max_dd['depth_pct']:.2f}%")
            st.metric("RoMaD", f"{romad:.2f}")
            st.metric("Current Price", f"${r3000_price_df['Close'].iloc[-1]:.2f}")
            st.metric("Peak Price", f"${r3000_price_df['Close'].max():.2f}")

with right_panel:
    if ark_ticker and r3000_ticker:
        if ark_price_df is not None and len(ark_price_df) > 0 and ark_dd is not None and len(ark_dd) > 0:
            # ARK stock chart
            ark_company_name_dict = load_company_name('ark')
            ark_company_name = ark_company_name_dict.get(ark_ticker, ark_ticker)
            ark_display_name = f"{ark_ticker}" if not ark_company_name or ark_company_name == ark_ticker else f"{ark_ticker} - {ark_company_name}"

            fig_ark = create_stock_chart(ark_price_df, ark_dd, f"{selected_etf}: {ark_display_name}")
            st.plotly_chart(fig_ark, use_container_width=True, config=CHART_CONFIG)
        else:
            st.warning(f"No price data available for {ark_ticker}")

        ""  # Space

        if r3000_price_df is not None and len(r3000_price_df) > 0 and r3000_dd is not None and len(r3000_dd) > 0:
            # Russell 3000 stock chart
            r3000_company_name_dict = load_company_name('r3000')
            r3000_company_name = r3000_company_name_dict.get(r3000_ticker, r3000_ticker)
            r3000_display_name = f"{r3000_ticker}" if not r3000_company_name or r3000_company_name == r3000_ticker else f"{r3000_ticker} - {r3000_company_name}"

            fig_r3000 = create_stock_chart(r3000_price_df, r3000_dd, f"Russell 3000: {r3000_display_name}")
            st.plotly_chart(fig_r3000, use_container_width=True, config=CHART_CONFIG)
        else:
            st.warning(f"No price data available for {r3000_ticker}")
    else:
        st.info("Please select both ARK and Russell 3000 stocks to view comparison")

""  # Add space

# Section: Drawdown Tables
if ark_ticker and r3000_ticker:
    if ark_dd is not None and len(ark_dd) > 0:
        st.subheader(f"{ark_ticker} Top 10 Drawdown")

        table_container = st.container(border=True)
        with table_container:
            display_df = ark_dd[ark_dd['rank'] != 'Current'].head(10).copy()

            if len(display_df) > 0:
                # Format columns
                display_df['Depth %'] = display_df['depth_pct'].apply(lambda x: f"{x:.2f}%")
                display_df['Peak Date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
                display_df['Trough Date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
                display_df['Peak Price'] = display_df['peak_price'].apply(lambda x: f"${x:,.2f}")
                display_df['Trough Price'] = display_df['trough_price'].apply(lambda x: f"${x:,.2f}")

                # Select and rename columns
                display_df = display_df[['rank', 'Depth %', 'Peak Date', 'Trough Date', 'Peak Price', 'Trough Price']]
                display_df = display_df.rename(columns={'rank': 'Rank'})

                st.dataframe(display_df, hide_index=True, use_container_width=True)

    ""  # Space

    if r3000_dd is not None and len(r3000_dd) > 0:
        st.subheader(f"{r3000_ticker} Top 10 Drawdown")

        table_container = st.container(border=True)
        with table_container:
            display_df = r3000_dd[r3000_dd['rank'] != 'Current'].head(10).copy()

            if len(display_df) > 0:
                # Format columns
                display_df['Depth %'] = display_df['depth_pct'].apply(lambda x: f"{x:.2f}%")
                display_df['Peak Date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
                display_df['Trough Date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
                display_df['Peak Price'] = display_df['peak_price'].apply(lambda x: f"${x:,.2f}")
                display_df['Trough Price'] = display_df['trough_price'].apply(lambda x: f"${x:,.2f}")

                # Select and rename columns
                display_df = display_df[['rank', 'Depth %', 'Peak Date', 'Trough Date', 'Peak Price', 'Trough Price']]
                display_df = display_df.rename(columns={'rank': 'Rank'})

                st.dataframe(display_df, hide_index=True, use_container_width=True)
