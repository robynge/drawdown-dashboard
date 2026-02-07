"""HHI Concentration Analysis Page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE, INPUT_DIR, OUTPUT_DIR
from data_loader import load_ark_holdings, load_etf_prices, get_ark_files_hash
from hhi_calculator import calculate_hhi_time_series
from drawdown_calculator import calculate_drawdowns
from chart_config import CHART_CONFIG, DD_COLORS

st.set_page_config(
    page_title="HHI Analysis",
    page_icon="",
    layout="wide"
)

"""
# HHI Concentration Analysis

Analyze portfolio concentration using Herfindahl-Hirschman Index (HHI) and its relationship with drawdowns.
"""

st.markdown(f"**Analysis Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

"" # Space

@st.cache_data
def get_cached_qqq_prices(_files_hash):
    """Load and cache QQQ prices"""
    qqq_file = OUTPUT_DIR / 'QQQ_prices.csv'
    if qqq_file.exists():
        df = pd.read_csv(qqq_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()

@st.cache_data
def get_cached_etf_prices(_files_hash, etf):
    """Load and cache ETF prices"""
    etf_file = OUTPUT_DIR / f'{etf}_prices.csv'
    if etf_file.exists():
        df = pd.read_csv(etf_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return pd.DataFrame()

@st.cache_data
def calculate_hhi_data(_files_hash, etf, _holdings):
    """Calculate HHI time series for an ETF"""
    holdings = _holdings

    # Filter to analysis period and exclude currency/money market
    holdings_filtered = holdings[
        (holdings['Date'] >= START_DATE) &
        (holdings['Date'] <= END_DATE)
    ].copy()

    # Filter out currency tickers
    if 'Bloomberg Name' in holdings_filtered.columns:
        holdings_filtered = holdings_filtered[
            ~holdings_filtered['Bloomberg Name'].str.contains('curncy', case=False, na=False)
        ]

    # Filter out money market funds (prefix matching)
    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
    ticker_symbols = holdings_filtered['Ticker'].str.split().str[0]
    is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
    holdings_filtered = holdings_filtered[~is_mm]

    # Calculate HHI time series
    hhi_df = calculate_hhi_time_series(holdings_filtered[['Date', 'Ticker', 'Weight']])

    return hhi_df

@st.cache_data
def calculate_returns_data(_files_hash, etf, _etf_prices, _qqq_prices):
    """Calculate absolute and relative returns"""
    etf_prices = _etf_prices.copy()
    qqq_prices = _qqq_prices.copy()

    if len(etf_prices) == 0 or len(qqq_prices) == 0:
        return pd.DataFrame()

    # Merge on date
    merged = pd.merge(
        etf_prices[['Date', 'Close']].rename(columns={'Close': 'ETF_Price'}),
        qqq_prices[['Date', 'Close']].rename(columns={'Close': 'QQQ_Price'}),
        on='Date',
        how='inner'
    )

    if len(merged) == 0:
        return pd.DataFrame()

    # Drop rows with NaN prices
    merged = merged.dropna(subset=['ETF_Price', 'QQQ_Price'])

    if len(merged) == 0:
        return pd.DataFrame()

    # Calculate cumulative returns from start
    first_etf = merged['ETF_Price'].iloc[0]
    first_qqq = merged['QQQ_Price'].iloc[0]

    # Check for valid first prices
    if pd.isna(first_etf) or pd.isna(first_qqq) or first_etf == 0 or first_qqq == 0:
        return pd.DataFrame()

    merged['ETF_Return'] = (merged['ETF_Price'] / first_etf - 1) * 100
    merged['QQQ_Return'] = (merged['QQQ_Price'] / first_qqq - 1) * 100
    merged['Relative_Return'] = merged['ETF_Return'] - merged['QQQ_Return']

    return merged

# Load data
files_hash = get_ark_files_hash()

# ETF Selection
st.subheader("Select ETF")

selected_etf = st.pills(
    "ETF",
    options=ARK_ETFS,
    default=ARK_ETFS[0],
    label_visibility="collapsed"
)

"" # Space

# Load data for selected ETF
with st.spinner("Loading data..."):
    holdings = load_ark_holdings(files_hash, selected_etf)
    etf_prices = get_cached_etf_prices(files_hash, selected_etf)
    qqq_prices = get_cached_qqq_prices(files_hash)
    hhi_data = calculate_hhi_data(files_hash, selected_etf, holdings)
    returns_data = calculate_returns_data(files_hash, selected_etf, etf_prices, qqq_prices)

if len(hhi_data) > 0 and len(etf_prices) > 0:
    # Section 1: Key Metrics
    st.subheader("Current Concentration Metrics")

    latest_hhi = hhi_data.iloc[-1]

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric(
            "HHI",
            f"{latest_hhi['HHI']:.4f}",
            help="Herfindahl-Hirschman Index: Sum of squared weights. Higher = more concentrated."
        )

    with metric_cols[1]:
        st.metric(
            "Effective Positions",
            f"{latest_hhi['Effective_Positions']:.1f}",
            help="1/HHI: Equivalent number of equal-weighted positions."
        )

    with metric_cols[2]:
        st.metric(
            "Top 5 Concentration",
            f"{latest_hhi['Top5_Concentration']*100:.1f}%",
            help="Combined weight of top 5 positions."
        )

    with metric_cols[3]:
        st.metric(
            "Total Positions",
            f"{latest_hhi['Num_Positions']:.0f}",
            help="Number of positions in the portfolio."
        )

    "" # Space

    # Section 2: HHI Time Series
    st.subheader("Concentration Over Time")

    # Toggle between HHI and Effective Positions
    conc_metric = st.pills("Metric", options=["HHI", "Effective Positions"], default="HHI", label_visibility="collapsed")

    hhi_ts_card = st.container(border=True)
    with hhi_ts_card:
        fig_conc = go.Figure()

        if conc_metric == "HHI":
            fig_conc.add_trace(
                go.Scatter(
                    x=hhi_data['Date'],
                    y=hhi_data['HHI'],
                    mode='lines',
                    name='HHI',
                    line=dict(color='steelblue', width=2),
                    hovertemplate='<b>HHI</b><br>Date: %{x|%Y-%m-%d}<br>HHI: %{y:.4f}<extra></extra>'
                )
            )
            fig_conc.update_layout(
                title=f"{selected_etf} HHI Over Time",
                yaxis_title="HHI"
            )
        else:
            fig_conc.add_trace(
                go.Scatter(
                    x=hhi_data['Date'],
                    y=hhi_data['Effective_Positions'],
                    mode='lines',
                    name='Effective Positions',
                    line=dict(color='green', width=2),
                    hovertemplate='<b>Effective Positions</b><br>Date: %{x|%Y-%m-%d}<br>Positions: %{y:.1f}<extra></extra>'
                )
            )
            fig_conc.update_layout(
                title=f"{selected_etf} Effective Positions Over Time",
                yaxis_title="Effective Positions"
            )

        fig_conc.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            showlegend=False
        )
        fig_conc.update_xaxes(title_text="Date", gridcolor='lightgray')
        fig_conc.update_yaxes(gridcolor='lightgray', autorange=True)

        st.plotly_chart(fig_conc, width='stretch', config=CHART_CONFIG)

        if conc_metric == "Effective Positions":
            st.markdown("<small>*Effective Positions = 1/HHI. It represents the equivalent number of equal-weighted positions that would produce the same concentration level. For example, if Effective Positions = 20, the portfolio's concentration is equivalent to holding 20 equally-weighted stocks.*</small>", unsafe_allow_html=True)

    "" # Space

    # Section 3: Price + Drawdown + HHI
    st.subheader("Price & Concentration")

    # Toggle for showing drawdowns
    show_drawdowns = st.toggle("Show Drawdowns", value=True)

    dd_hhi_card = st.container(border=True)
    with dd_hhi_card:
        price_df = etf_prices
        etf_dd_data = calculate_drawdowns(price_df)

        # Create figure
        fig2 = go.Figure()

        # Add drawdown shaded regions (if enabled)
        if show_drawdowns and len(etf_dd_data) > 0:
            top_10_dd = etf_dd_data[etf_dd_data['rank'] != 'Current'].head(10)

            # Color palette for drawdowns
            dd_colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                         'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                         'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                         'rgba(255, 228, 181, 0.3)']

            for idx, (_, row) in enumerate(top_10_dd.iterrows()):
                fig2.add_vrect(
                    x0=row['peak_date'],
                    x1=row['trough_date'],
                    fillcolor=dd_colors[idx % len(dd_colors)],
                    layer="below",
                    line_width=0
                )

        # ETF price line
        fig2.add_trace(go.Scatter(
            x=price_df['Date'],
            y=price_df['Close'],
            mode='lines',
            name=f'{selected_etf} Price',
            line=dict(color='black', width=2),
            hovertemplate=f'<b>{selected_etf}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>'
        ))

        # QQQ price line (actual prices, Y-axis aligned so first day overlaps)
        qqq_axis_range = None
        if len(qqq_prices) > 0:
            # Filter QQQ to match ETF date range
            qqq_filtered = qqq_prices[
                (qqq_prices['Date'] >= price_df['Date'].min()) &
                (qqq_prices['Date'] <= price_df['Date'].max())
            ].copy()

            if len(qqq_filtered) > 0:
                # Get first day prices and ETF price range
                first_etf_price = price_df['Close'].iloc[0]
                first_qqq_price = qqq_filtered['Close'].iloc[0]
                etf_min = price_df['Close'].min()
                etf_max = price_df['Close'].max()

                # Calculate ETF percentage range from first price
                etf_min_pct = etf_min / first_etf_price
                etf_max_pct = etf_max / first_etf_price

                # Apply same percentage range to QQQ
                qqq_axis_range = [first_qqq_price * etf_min_pct, first_qqq_price * etf_max_pct]

                fig2.add_trace(go.Scatter(
                    x=qqq_filtered['Date'],
                    y=qqq_filtered['Close'],
                    mode='lines',
                    name='QQQ Price',
                    line=dict(color='orange', width=2),
                    hovertemplate='<b>QQQ</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>',
                    yaxis='y3'
                ))

        # Add current drawdown line and shaded area (if enabled)
        if show_drawdowns and len(etf_dd_data) > 0:
            current_dd = etf_dd_data[etf_dd_data['rank'] == 'Current'].iloc[0]
            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_price = current_dd['trough_price']
            current_dd_pct = current_dd['depth_pct']

            fig2.add_shape(
                type="line",
                x0=peak_date,
                x1=price_df['Date'].max(),
                y0=peak_price,
                y1=peak_price,
                line=dict(color='red', width=2, dash='dash'),
                layer='above'
            )

            fig2.add_shape(
                type="rect",
                x0=peak_date,
                x1=price_df['Date'].max(),
                y0=current_price,
                y1=peak_price,
                fillcolor='rgba(128,128,128,0.25)',
                line=dict(width=0),
                layer='below'
            )

            fig2.add_annotation(
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

        # HHI line on secondary y-axis
        fig2.add_trace(go.Scatter(
            x=hhi_data['Date'],
            y=hhi_data['HHI'],
            mode='lines',
            name=f'{selected_etf} HHI',
            line=dict(color='steelblue', width=2, dash='dot'),
            hovertemplate=f'<b>{selected_etf} HHI</b><br>Date: %{{x|%Y-%m-%d}}<br>HHI: %{{y:.4f}}<extra></extra>',
            yaxis='y2'
        ))

        chart_title = f"{selected_etf} vs QQQ with HHI"
        if show_drawdowns:
            chart_title = f"{selected_etf} vs QQQ with Drawdowns & HHI"

        fig2.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title=f"{selected_etf} Price ($)",
            hovermode='x unified',
            height=650,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', showgrid=True, domain=[0, 0.86], rangeslider=dict(visible=True)),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis2=dict(title=f'{selected_etf} HHI', overlaying='y', side='right', position=0.97, showgrid=False),
            yaxis3=dict(title='QQQ Price', overlaying='y', side='right', position=0.91, showgrid=False, range=qqq_axis_range),
            margin=dict(l=0, r=70, t=40, b=0)
        )

        st.plotly_chart(fig2, width='stretch', config=CHART_CONFIG)

        st.markdown("<small>*Colored regions show top 10 historical drawdowns. Gray region shows current drawdown. QQQ Y-axis scaled so first day aligns with ETF.*</small>", unsafe_allow_html=True)

    "" # Space

    # Section 4: Performance & Concentration
    if len(returns_data) > 0:
        st.subheader("Returns & Concentration")

        returns_card = st.container(border=True)
        with returns_card:
            # Merge HHI with returns data
            returns_hhi = pd.merge(
                returns_data,
                hhi_data[['Date', 'HHI']],
                on='Date',
                how='inner'
            )

            if len(returns_hhi) > 0:
                fig_returns = make_subplots(specs=[[{"secondary_y": True}]])

                # ETF return
                fig_returns.add_trace(
                    go.Scatter(
                        x=returns_hhi['Date'],
                        y=returns_hhi['ETF_Return'],
                        mode='lines',
                        name=f'{selected_etf} Return',
                        line=dict(color='black', width=2),
                        hovertemplate='<b>%s Return</b><br>Date: %%{x|%%Y-%%m-%%d}<br>Return: %%{y:.2f}%%<extra></extra>' % selected_etf
                    ),
                    secondary_y=False
                )

                # QQQ return
                fig_returns.add_trace(
                    go.Scatter(
                        x=returns_hhi['Date'],
                        y=returns_hhi['QQQ_Return'],
                        mode='lines',
                        name='QQQ Return',
                        line=dict(color='orange', width=2),
                        hovertemplate='<b>QQQ Return</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                    ),
                    secondary_y=False
                )

                # Relative return
                fig_returns.add_trace(
                    go.Scatter(
                        x=returns_hhi['Date'],
                        y=returns_hhi['Relative_Return'],
                        mode='lines',
                        name='Relative (vs QQQ)',
                        line=dict(color='green', width=2, dash='dash'),
                        hovertemplate='<b>Relative Return</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                    ),
                    secondary_y=False
                )

                # HHI on secondary axis
                fig_returns.add_trace(
                    go.Scatter(
                        x=returns_hhi['Date'],
                        y=returns_hhi['HHI'],
                        mode='lines',
                        name=f'{selected_etf} HHI',
                        line=dict(color='steelblue', width=2, dash='dot'),
                        hovertemplate=f'<b>{selected_etf} HHI</b><br>Date: %{{x|%Y-%m-%d}}<br>HHI: %{{y:.4f}}<extra></extra>'
                    ),
                    secondary_y=True
                )

                # Add zero line for reference
                fig_returns.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, secondary_y=False)

                fig_returns.update_layout(
                    title=f"{selected_etf} Cumulative Returns vs QQQ & HHI",
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='x unified'
                )

                fig_returns.update_xaxes(title_text="Date", gridcolor='lightgray', rangeslider=dict(visible=True))
                fig_returns.update_yaxes(title_text="Cumulative Return (%)", secondary_y=False, gridcolor='lightgray')
                fig_returns.update_yaxes(title_text="HHI", secondary_y=True)

                st.plotly_chart(fig_returns, width='stretch', config=CHART_CONFIG)

                # Summary statistics
                st.markdown("#### Performance Summary")

                summary_cols = st.columns(3)

                with summary_cols[0]:
                    etf_total_return = returns_hhi['ETF_Return'].iloc[-1]
                    st.metric(f"{selected_etf} Total Return", f"{etf_total_return:+.2f}%")

                with summary_cols[1]:
                    qqq_total_return = returns_hhi['QQQ_Return'].iloc[-1]
                    st.metric("QQQ Total Return", f"{qqq_total_return:+.2f}%")

                with summary_cols[2]:
                    relative_return = returns_hhi['Relative_Return'].iloc[-1]
                    delta_color = "normal" if relative_return >= 0 else "inverse"
                    st.metric(
                        "Relative Performance",
                        f"{relative_return:+.2f}%",
                        delta="Outperforming" if relative_return >= 0 else "Underperforming",
                        delta_color=delta_color
                    )

else:
    st.warning(f"Not enough data for {selected_etf}")
