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

    # Filter out money market funds
    excluded_tickers = ['FTOXX', 'FIRXX']
    holdings_filtered = holdings_filtered[
        ~holdings_filtered['Ticker'].str.split().str[0].isin(excluded_tickers)
    ]

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

    hhi_ts_card = st.container(border=True)
    with hhi_ts_card:
        fig_hhi = make_subplots(specs=[[{"secondary_y": True}]])

        # HHI line (left axis)
        fig_hhi.add_trace(
            go.Scatter(
                x=hhi_data['Date'],
                y=hhi_data['HHI'],
                mode='lines',
                name='HHI',
                line=dict(color='steelblue', width=2),
                hovertemplate='<b>HHI</b><br>Date (x): %{x|%Y-%m-%d}<br>HHI (y): %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )

        # Effective Positions line (right axis)
        fig_hhi.add_trace(
            go.Scatter(
                x=hhi_data['Date'],
                y=hhi_data['Effective_Positions'],
                mode='lines',
                name='Effective Positions',
                line=dict(color='green', width=2),
                hovertemplate='<b>Effective Positions</b><br>Date (x): %{x|%Y-%m-%d}<br>Positions (y): %{y:.1f}<extra></extra>'
            ),
            secondary_y=True
        )

        fig_hhi.update_layout(
            title=f"{selected_etf} HHI & Effective Positions",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )

        fig_hhi.update_xaxes(title_text="Date", gridcolor='lightgray')
        fig_hhi.update_yaxes(title_text="HHI", secondary_y=False, gridcolor='lightgray', autorange=True)
        fig_hhi.update_yaxes(title_text="Effective Positions", secondary_y=True, autorange=True)

        st.plotly_chart(fig_hhi, width='stretch', config=CHART_CONFIG)

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

        # QQQ price line
        if len(qqq_prices) > 0:
            fig2.add_trace(go.Scatter(
                x=qqq_prices['Date'],
                y=qqq_prices['Close'],
                mode='lines',
                name='QQQ Price',
                line=dict(color='orange', width=2),
                hovertemplate='<b>QQQ</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
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

        chart_title = f"{selected_etf} & QQQ Price with HHI"
        if show_drawdowns:
            chart_title = f"{selected_etf} Price with Drawdowns & HHI"

        fig2.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=650,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis2=dict(title='HHI', overlaying='y', side='right'),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig2, width='stretch', config=CHART_CONFIG)

        st.markdown("<small>*Colored regions show top 10 historical drawdowns. Gray region shows current drawdown. Dotted blue line shows HHI.*</small>", unsafe_allow_html=True)

    "" # Space

    # Section 4: Performance & Concentration
    if len(returns_data) > 0:
        st.subheader("Performance & Concentration")

        # Toggle between Return and Price view
        view_mode = st.pills("View", options=["Return", "Price"], default="Return", label_visibility="collapsed")

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
                fig_perf = make_subplots(specs=[[{"secondary_y": True}]])

                if view_mode == "Return":
                    # Absolute return
                    fig_perf.add_trace(
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
                    fig_perf.add_trace(
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
                    fig_perf.add_trace(
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

                    # Add zero line for reference
                    fig_perf.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, secondary_y=False)

                    y_axis_title = "Cumulative Return (%)"
                    chart_title = f"{selected_etf} Cumulative Returns vs QQQ & HHI"

                else:  # Price view
                    # ETF price
                    fig_perf.add_trace(
                        go.Scatter(
                            x=returns_hhi['Date'],
                            y=returns_hhi['ETF_Price'],
                            mode='lines',
                            name=f'{selected_etf} Price',
                            line=dict(color='black', width=2),
                            hovertemplate=f'<b>{selected_etf} Price</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>'
                        ),
                        secondary_y=False
                    )

                    # QQQ price
                    fig_perf.add_trace(
                        go.Scatter(
                            x=returns_hhi['Date'],
                            y=returns_hhi['QQQ_Price'],
                            mode='lines',
                            name='QQQ Price',
                            line=dict(color='orange', width=2),
                            hovertemplate='<b>QQQ Price</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
                        ),
                        secondary_y=False
                    )

                    y_axis_title = "Price ($)"
                    chart_title = f"{selected_etf} Price vs QQQ & HHI"

                # HHI on secondary axis (both views)
                fig_perf.add_trace(
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

                fig_perf.update_layout(
                    title=chart_title,
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    hovermode='x unified'
                )

                fig_perf.update_xaxes(title_text="Date", gridcolor='lightgray', rangeslider=dict(visible=True))
                fig_perf.update_yaxes(title_text=y_axis_title, secondary_y=False, gridcolor='lightgray')
                fig_perf.update_yaxes(title_text="HHI", secondary_y=True)

                st.plotly_chart(fig_perf, width='stretch', config=CHART_CONFIG)

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

    "" # Space

    # Section 5: HHI vs Drawdown Scatter (Optional Analysis)
    st.subheader("Concentration vs Drawdown Analysis")

    analysis_card = st.container(border=True)
    with analysis_card:
        # Merge HHI with price data to calculate daily drawdown
        price_hhi = pd.merge(
            etf_prices[['Date', 'Close']],
            hhi_data[['Date', 'HHI', 'Effective_Positions']],
            on='Date',
            how='inner'
        )

        if len(price_hhi) > 0:
            # Calculate running max and drawdown
            price_hhi['Peak'] = price_hhi['Close'].cummax()
            price_hhi['Drawdown'] = (price_hhi['Close'] / price_hhi['Peak'] - 1) * 100

            # Create dual-axis chart: Drawdown (inverted) vs HHI
            fig_dd_hhi = make_subplots(specs=[[{"secondary_y": True}]])

            # Drawdown (inverted so more negative is higher on chart)
            fig_dd_hhi.add_trace(
                go.Scatter(
                    x=price_hhi['Date'],
                    y=price_hhi['Drawdown'],
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    hovertemplate='<b>Drawdown</b><br>Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                secondary_y=False
            )

            # HHI
            fig_dd_hhi.add_trace(
                go.Scatter(
                    x=price_hhi['Date'],
                    y=price_hhi['HHI'],
                    mode='lines',
                    name='HHI',
                    line=dict(color='steelblue', width=2),
                    hovertemplate='<b>HHI</b><br>Date: %{x|%Y-%m-%d}<br>HHI: %{y:.4f}<extra></extra>'
                ),
                secondary_y=True
            )

            fig_dd_hhi.update_layout(
                title=f"{selected_etf} Drawdown vs HHI",
                height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified'
            )

            fig_dd_hhi.update_xaxes(title_text="Date", gridcolor='lightgray')
            fig_dd_hhi.update_yaxes(
                title_text="Drawdown (%)",
                secondary_y=False,
                gridcolor='lightgray',
                autorange='reversed'  # Invert so deeper drawdowns are lower
            )
            fig_dd_hhi.update_yaxes(title_text="HHI", secondary_y=True)

            st.plotly_chart(fig_dd_hhi, width='stretch', config=CHART_CONFIG)

            st.markdown("<small>*Red area shows drawdown (lower = deeper drawdown). Blue line shows HHI concentration. Look for patterns: does HHI increase during drawdowns?*</small>", unsafe_allow_html=True)

else:
    st.warning(f"Not enough data for {selected_etf}")
