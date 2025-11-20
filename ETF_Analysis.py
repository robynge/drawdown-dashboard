"""ETF Analysis Dashboard"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE
from data_loader import load_etf_prices, get_stock_etf_mapping
from drawdown_calculator import calculate_drawdowns
from chart_config import CHART_CONFIG

st.set_page_config(
    page_title="ETF Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
# ETF Drawdown Analysis

Analyze drawdown patterns across ARK ETFs.
"""

st.markdown(f"**Analysis Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

""  # Add space

# Load and cache ETF price and drawdown data
@st.cache_data
def load_all_etf_data():
    """Load price and drawdown data for all ETFs"""
    price_data = {}
    dd_data = []

    for etf in ARK_ETFS:
        etf_prices = load_etf_prices(etf)

        if len(etf_prices) > 0:
            price_data[etf] = etf_prices

            dd_df = calculate_drawdowns(etf_prices)
            if len(dd_df) > 0:
                dd_df.insert(0, 'ETF', etf)
                dd_data.append(dd_df)

    all_dd = pd.concat(dd_data, ignore_index=True) if dd_data else pd.DataFrame()
    return price_data, all_dd

# Load data
with st.spinner("Loading ETF drawdown data..."):
    etf_prices, etf_dd = load_all_etf_data()

# Section 1: ARK ETF Price Overview
st.subheader("ARK ETF Price Trends")

if len(etf_prices) > 0:
    chart_container = st.container(border=True)

    with chart_container:
        # Create multi-line chart with actual prices
        fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for idx, (etf, price_df) in enumerate(etf_prices.items()):
            fig.add_trace(go.Scatter(
                x=price_df['Date'],
                y=price_df['Close'],
                mode='lines',
                name=etf,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            margin=dict(l=0, r=0, t=30, b=60)
        )

        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
else:
    st.error("No ETF price data available")

""  # Add space

# Section 2: Individual ETF Analysis
st.subheader("Individual ETF Drawdown Analysis")

# Layout: left column for controls and metrics, right column for chart
cols = st.columns([1, 3])

left_panel = cols[0].container(border=True, height=700)
right_panel = cols[1].container(border=True, height=700)

with left_panel:
    st.markdown("### Select ETF")
    selected_etf = st.pills(
        "ETF",
        options=ARK_ETFS,
        default=ARK_ETFS[0],
        label_visibility="collapsed"
    )

    ""  # Space

    # Show metrics for selected ETF
    if selected_etf in etf_prices and len(etf_dd[etf_dd['ETF'] == selected_etf]) > 0:
        etf_dd_data = etf_dd[etf_dd['ETF'] == selected_etf]
        current_dd = etf_dd_data[etf_dd_data['rank'] == 'Current'].iloc[0]
        max_dd = etf_dd_data[etf_dd_data['rank'] == 1].iloc[0]

        # Calculate RoMaD (Return over Maximum Drawdown)
        price_df = etf_prices[selected_etf]
        first_price = price_df['Close'].iloc[0]
        last_price = price_df['Close'].iloc[-1]
        overall_return = ((last_price - first_price) / first_price) * 100
        max_dd_abs = abs(max_dd['depth_pct'])
        romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

        st.markdown("### Key Metrics")

        st.metric(
            "Current Drawdown",
            f"{current_dd['depth_pct']:.2f}%",
            delta=None
        )

        st.metric(
            "Max Drawdown",
            f"{max_dd['depth_pct']:.2f}%",
            delta=None
        )

        st.metric(
            "Current Price",
            f"${price_df['Close'].iloc[-1]:.2f}",
            delta=None
        )

        st.metric(
            "Peak Price",
            f"${price_df['Close'].max():.2f}",
            delta=None
        )

        st.metric(
            "RoMaD",
            f"{romad:.2f}",
            delta=None
        )

        ""  # Space

        # Drawdown count
        st.markdown("### Drawdown Statistics")
        total_dd = len(etf_dd_data[etf_dd_data['rank'] != 'Current'])
        st.metric("Total Drawdowns", total_dd)

with right_panel:
    if selected_etf in etf_prices:
        price_df = etf_prices[selected_etf]
        etf_dd_data = etf_dd[etf_dd['ETF'] == selected_etf]

        # Create figure with drawdown regions
        fig2 = go.Figure()

        # Get top 10 drawdowns
        if len(etf_dd_data) > 0:
            top_10_dd = etf_dd_data[etf_dd_data['rank'] != 'Current'].head(10)

            # Color palette for drawdowns
            dd_colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                         'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                         'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                         'rgba(255, 228, 181, 0.3)']

            # Add drawdown shaded regions
            for idx, (_, row) in enumerate(top_10_dd.iterrows()):
                fig2.add_vrect(
                    x0=row['peak_date'],
                    x1=row['trough_date'],
                    fillcolor=dd_colors[idx % len(dd_colors)],
                    layer="below",
                    line_width=0
                )

        # Add price line with custom hover template
        price_df_copy = price_df.copy()
        price_df_copy['DD_Info'] = ''

        # Add drawdown info for each date
        if len(etf_dd_data) > 0:
            top_10_dd = etf_dd_data[etf_dd_data['rank'] != 'Current'].head(10)
            for _, row in top_10_dd.iterrows():
                mask = (price_df_copy['Date'] >= row['peak_date']) & (price_df_copy['Date'] <= row['trough_date'])
                price_df_copy.loc[mask, 'DD_Info'] = (
                    f"<br><b>Drawdown #{row['rank']}</b><br>" +
                    f"Depth: {row['depth_pct']:.2f}%<br>" +
                    f"Peak: {row['peak_date'].strftime('%Y-%m-%d')} ${row['peak_price']:.2f}<br>" +
                    f"Trough: {row['trough_date'].strftime('%Y-%m-%d')} ${row['trough_price']:.2f}"
                )

        fig2.add_trace(go.Scatter(
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
        if len(etf_dd_data) > 0:
            current_dd = etf_dd_data[etf_dd_data['rank'] == 'Current'].iloc[0]
            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_price = current_dd['trough_price']
            current_dd_pct = current_dd['depth_pct']

            # Add horizontal line from peak date to the end of the chart
            fig2.add_shape(
                type="line",
                x0=peak_date,
                x1=price_df['Date'].max(),
                y0=peak_price,
                y1=peak_price,
                line=dict(color='red', width=2, dash='dash'),
                layer='above'
            )

            # Add shaded rectangle from peak date to current date
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

            # Add text annotation on the right side showing current drawdown
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

        fig2.update_layout(
            title=f"{selected_etf} Price with Top 10 Drawdowns & Current Drawdown",
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

        st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)
    else:
        st.error(f"No data available for {selected_etf}")

""  # Add space

# Section 3: Drawdown Details
st.subheader("Drawdown Details")

# Add CSS for left alignment
st.markdown("""
<style>
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)

if selected_etf in etf_prices and len(etf_dd[etf_dd['ETF'] == selected_etf]) > 0:
    details_container = st.container(border=True)

    with details_container:
        etf_dd_data = etf_dd[etf_dd['ETF'] == selected_etf]
        display_df = etf_dd_data.copy()

        # Format numeric columns as strings
        display_df['Depth %'] = display_df['depth_pct'].apply(lambda x: f"{x:.2f}%")
        display_df['Peak Date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
        display_df['Trough Date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
        display_df['Peak Price'] = display_df['peak_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Trough Price'] = display_df['trough_price'].apply(lambda x: f"${x:,.2f}")

        # Reorder to put Current first
        current_row = display_df[display_df['rank'] == 'Current']
        historical_rows = display_df[display_df['rank'] != 'Current']
        display_df = pd.concat([current_row, historical_rows], ignore_index=True)

        # Select and rename columns for display
        display_df = display_df[['ETF', 'rank', 'Depth %', 'Peak Date', 'Trough Date', 'Peak Price', 'Trough Price']]
        display_df = display_df.rename(columns={'rank': 'Rank'})

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

""  # Add space

# Section 4: ETF Comparison
st.subheader("ETF Drawdown Comparison")

if len(etf_dd) > 0:
    comparison_container = st.container(border=True)

    with comparison_container:
        # Show only max drawdown per ETF
        etf_summary = etf_dd[etf_dd['rank'] == 1][['ETF', 'depth_pct', 'peak_date', 'trough_date']].copy()

        # Format columns as strings
        etf_summary['Max Drawdown %'] = etf_summary['depth_pct'].apply(lambda x: f"{x:.2f}%")
        etf_summary['Peak Date'] = etf_summary['peak_date'].dt.strftime('%Y-%m-%d')
        etf_summary['Trough Date'] = etf_summary['trough_date'].dt.strftime('%Y-%m-%d')

        # Select final columns
        etf_summary = etf_summary[['ETF', 'Max Drawdown %', 'Peak Date', 'Trough Date']]

        st.dataframe(
            etf_summary,
            use_container_width=True,
            hide_index=True
        )
else:
    st.info("No ETF drawdown data available")
