"""Russell 3000 Analysis Page"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import START_DATE, END_DATE, OUTPUT_DIR
from peer_group import get_peer_group_prices
from chart_config import CHART_CONFIG

st.set_page_config(page_title="Russell 3000 Analysis", page_icon="", layout="wide")

"""
# Russell 3000 Analysis

Analyze Russell 3000 Index and GICS Industry Peer Group drawdowns.
"""

st.markdown(f"**Analysis Period:** {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")

""  # Add space

# Load IWV index data
@st.cache_data
def load_iwv_data():
    """Load Russell 3000 Index (IWV) price and drawdown data"""
    price_file = OUTPUT_DIR / 'IWV_prices.csv'
    dd_file = OUTPUT_DIR / 'IWV_drawdown_2024-2025.xlsx'

    if not price_file.exists() or not dd_file.exists():
        return None, None

    prices = pd.read_csv(price_file)
    prices['Date'] = pd.to_datetime(prices['Date'])

    drawdowns = pd.read_excel(dd_file, sheet_name='Drawdowns')
    drawdowns['peak_date'] = pd.to_datetime(drawdowns['peak_date'])
    drawdowns['trough_date'] = pd.to_datetime(drawdowns['trough_date'])

    # Convert rank to string for consistency with dynamically calculated drawdowns
    if 'rank' in drawdowns.columns:
        drawdowns['rank'] = drawdowns['rank'].astype(str)

    return prices, drawdowns

# Load peer group data
@st.cache_data
def load_peer_groups():
    """Load GICS industry peer group drawdowns"""
    peer_file = OUTPUT_DIR / 'R3000_peer_groups_drawdown_2024-2025.xlsx'

    if not peer_file.exists():
        return []

    xl_file = pd.ExcelFile(peer_file)
    return xl_file.sheet_names

# Main Analysis Section
st.subheader("Drawdown Analysis")

iwv_prices, iwv_dd = load_iwv_data()
peer_groups = load_peer_groups()

if iwv_prices is not None and iwv_dd is not None:
    # Layout: left controls and metrics, right chart
    cols = st.columns([1, 3])

    # Left column: two stacked cards
    with cols[0]:
        # Card 1: Selection controls
        controls_card = st.container(border=True)
        with controls_card:
            st.markdown("<small><b>Select Analysis Target</b></small>", unsafe_allow_html=True)

            # Create options list: IWV + all peer groups
            analysis_options = ["IWV (Russell 3000)"] + peer_groups
            selected_target = st.selectbox(
                "Analysis Target",
                analysis_options,
                index=0,
                label_visibility="collapsed"
            )

            # Show version selector only for peer groups
            is_peer_group = selected_target != "IWV (Russell 3000)"

            if is_peer_group:
                ""  # Space
                st.markdown("<small><b>Version</b></small>", unsafe_allow_html=True)
                version = st.pills(
                    "Version",
                    options=["Market Value", "Weighted Price"],
                    default="Market Value",
                    label_visibility="collapsed"
                )
                version_param = 'mv' if version == "Market Value" else 'weighted'
            else:
                version = None
                version_param = None

        ""  # Space

        # Load data based on selection
        if is_peer_group:
            # Load peer group data
            with st.spinner(f"Loading {selected_target} data..."):
                prices = get_peer_group_prices(selected_target, version=version_param)

            # Filter to analysis period
            prices = prices[(prices['Date'] >= START_DATE) & (prices['Date'] <= END_DATE)]

            # Calculate drawdowns dynamically for the selected version
            from drawdown_calculator import calculate_drawdowns
            prices_for_dd = prices.copy()
            prices_for_dd = prices_for_dd.rename(columns={'Value': 'Close'})
            dd_data = calculate_drawdowns(prices_for_dd)

            if len(dd_data) > 0:
                dd_data['peak_date'] = pd.to_datetime(dd_data['peak_date'])
                dd_data['trough_date'] = pd.to_datetime(dd_data['trough_date'])

            price_column = 'Value'
            y_axis_title = version
        else:
            # Use IWV data
            prices = iwv_prices
            dd_data = iwv_dd
            price_column = 'Close'
            y_axis_title = "Price ($)"

        # Card 2: Key Metrics
        metrics_card = st.container(border=True)
        with metrics_card:
            st.markdown("<small><b>Key Metrics</b></small>", unsafe_allow_html=True)

            # Check if drawdown data is valid
            if len(dd_data) == 0 or 'rank' not in dd_data.columns:
                st.error("Insufficient data to calculate drawdowns for this selection")
            else:
                current_dd = dd_data[dd_data['rank'] == 'Current'].iloc[0]
                top_dd = dd_data[dd_data['rank'] == '1'].iloc[0]

                # Calculate RoMaD (Return over Maximum Drawdown)
                first_price = prices[price_column].iloc[0]
                last_price = prices[price_column].iloc[-1]
                overall_return = ((last_price - first_price) / first_price) * 100
                max_dd_abs = abs(top_dd['depth_pct'])
                romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

                cols_metrics = st.columns(2)
                with cols_metrics[0]:
                    st.markdown(f"<small>Current Drawdown</small><br><b>{current_dd['depth_pct']:.2f}%</b>", unsafe_allow_html=True)
                with cols_metrics[1]:
                    st.markdown(f"<small>Max Drawdown</small><br><b>{top_dd['depth_pct']:.2f}%</b>", unsafe_allow_html=True)

                if not is_peer_group:
                    cols_price = st.columns(2)
                    with cols_price[0]:
                        st.markdown(f"<small>Current</small><br><b>${prices['Close'].iloc[-1]:,.2f}</b>", unsafe_allow_html=True)
                    with cols_price[1]:
                        st.markdown(f"<small>Peak</small><br><b>${prices['Close'].max():,.2f}</b>", unsafe_allow_html=True)

                # RoMaD row
                st.markdown(f"<small>RoMaD</small><br><b>{romad:.2f}</b>", unsafe_allow_html=True)

    # Right column: chart
    right_panel = cols[1].container(border=True, height=700)

    with right_panel:
        # Create figure with drawdown regions
        fig = go.Figure()

        # Get top 10 drawdowns
        if len(dd_data) > 0 and 'rank' in dd_data.columns:
            top_10_dd = dd_data[dd_data['rank'] != 'Current'].head(10)

            # Color palette for drawdowns
            dd_colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                         'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                         'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                         'rgba(255, 228, 181, 0.3)']

            # Add drawdown shaded regions
            for idx, (_, row) in enumerate(top_10_dd.iterrows()):
                fig.add_vrect(
                    x0=row['peak_date'],
                    x1=row['trough_date'],
                    fillcolor=dd_colors[idx % len(dd_colors)],
                    layer="below",
                    line_width=0
                )

        # Add price line with custom hover template
        price_df_copy = prices.copy()
        price_df_copy['DD_Info'] = ''

        # Add drawdown info for each date
        if len(dd_data) > 0 and 'rank' in dd_data.columns:
            top_10_dd = dd_data[dd_data['rank'] != 'Current'].head(10)
            for _, row in top_10_dd.iterrows():
                mask = (price_df_copy['Date'] >= row['peak_date']) & (price_df_copy['Date'] <= row['trough_date'])
                price_df_copy.loc[mask, 'DD_Info'] = (
                    f"<br><b>Drawdown #{row['rank']}</b><br>" +
                    f"Depth: {row['depth_pct']:.2f}%<br>" +
                    f"Peak: {row['peak_date'].strftime('%Y-%m-%d')} ${row['peak_price']:,.2f}<br>" +
                    f"Trough: {row['trough_date'].strftime('%Y-%m-%d')} ${row['trough_price']:,.2f}"
                )

        # Choose line color based on selection
        line_color = 'darkblue' if is_peer_group else 'black'

        # Format hover value based on whether it's peer group
        if is_peer_group:
            hover_format = 'Value: $%{y:,.0f}%{customdata}<extra></extra>'
        else:
            hover_format = 'Price: $%{y:,.2f}%{customdata}<extra></extra>'

        fig.add_trace(go.Scatter(
            x=price_df_copy['Date'],
            y=price_df_copy[price_column],
            mode='lines',
            line=dict(color=line_color, width=2),
            customdata=price_df_copy['DD_Info'],
            hovertemplate='%{x|%Y-%m-%d}<br>' + hover_format,
            showlegend=False,
            hoverlabel=dict(bgcolor='white', bordercolor='lightgray'),
            marker=dict(color='rgba(0,0,0,0)')
        ))

        # Add current drawdown line and shaded area
        if len(dd_data) > 0 and 'rank' in dd_data.columns:
            current_dd = dd_data[dd_data['rank'] == 'Current'].iloc[0]
            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_price = current_dd['trough_price']
            current_dd_pct = current_dd['depth_pct']

            # Add horizontal line from peak date to the end of the chart
            fig.add_shape(
                type="line",
                x0=peak_date,
                x1=prices['Date'].max(),
                y0=peak_price,
                y1=peak_price,
                line=dict(color='red', width=2, dash='dash'),
                layer='above'
            )

            # Add shaded rectangle from peak date to current date
            fig.add_shape(
                type="rect",
                x0=peak_date,
                x1=prices['Date'].max(),
                y0=current_price,
                y1=peak_price,
                fillcolor='rgba(128,128,128,0.25)',
                line=dict(width=0),
                layer='below'
            )

            # Format annotation based on whether it's peer group
            if is_peer_group:
                annotation_text = (
                    f"<b>Current Drawdown</b><br>" +
                    f"Depth: {current_dd_pct:.2f}%<br>" +
                    f"Peak: {peak_date.strftime('%Y-%m-%d')} ${peak_price:,.0f}<br>" +
                    f"Current: {prices['Date'].max().strftime('%Y-%m-%d')} ${current_price:,.0f}"
                )
            else:
                annotation_text = (
                    f"<b>Current Drawdown</b><br>" +
                    f"Depth: {current_dd_pct:.2f}%<br>" +
                    f"Peak: {peak_date.strftime('%Y-%m-%d')} ${peak_price:,.2f}<br>" +
                    f"Current: {prices['Date'].max().strftime('%Y-%m-%d')} ${current_price:,.2f}"
                )

            # Add text annotation on the right side showing current drawdown
            fig.add_annotation(
                text=annotation_text,
                x=prices['Date'].max(),
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

        # Set chart title based on selection
        if is_peer_group:
            chart_title = f"{selected_target} - {version} with Top 10 Drawdowns & Current Drawdown"
        else:
            chart_title = "Russell 3000 (IWV) Price with Top 10 Drawdowns & Current Drawdown"

        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            hovermode='x unified',
            height=650,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray', showgrid=True),
            yaxis=dict(gridcolor='lightgray', showgrid=True),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, width='stretch', config=CHART_CONFIG)

    ""  # Space

    # Drawdown Details
    st.markdown("### Drawdown Details")

    # Add CSS for left alignment
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    details_container = st.container(border=True)

    with details_container:
        if len(dd_data) == 0 or 'rank' not in dd_data.columns:
            st.info("No drawdown data available for display")
        else:
            display_df = dd_data.copy()

            # Convert rank to string to avoid mixed type issues
            display_df['rank'] = display_df['rank'].astype(str)

            # Format numeric columns as strings
            display_df['Depth %'] = display_df['depth_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['Peak Date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
            display_df['Trough Date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')

            # Label depends on whether it's peer group
            if is_peer_group:
                display_df['Peak Value'] = display_df['peak_price'].apply(lambda x: f"${x:,.2f}")
                display_df['Trough Value'] = display_df['trough_price'].apply(lambda x: f"${x:,.2f}")
                cols_to_show = ['rank', 'Depth %', 'Peak Date', 'Trough Date', 'Peak Value', 'Trough Value']
            else:
                display_df['Peak Price'] = display_df['peak_price'].apply(lambda x: f"${x:,.2f}")
                display_df['Trough Price'] = display_df['trough_price'].apply(lambda x: f"${x:,.2f}")
                cols_to_show = ['rank', 'Depth %', 'Peak Date', 'Trough Date', 'Peak Price', 'Trough Price']

            # Reorder to put Current first
            current_row = display_df[display_df['rank'] == 'Current']
            historical_rows = display_df[display_df['rank'] != 'Current']
            display_df = pd.concat([current_row, historical_rows], ignore_index=True)

            # Select columns and rename
            display_df = display_df[cols_to_show]
            display_df = display_df.rename(columns={'rank': 'Rank'})

            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True
            )
else:
    st.error("Russell 3000 Index data not available")
