"""ETF Analysis Dashboard"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import ARK_ETFS, START_DATE, END_DATE, OUTPUT_DIR
from data_loader import load_etf_prices, get_stock_etf_mapping
from drawdown_calculator import calculate_drawdowns
from chart_config import CHART_CONFIG
from recovery_probability import get_etf_drawdowns_in_depth_range

# Precomputed data directory
PRECOMP_DIR = Path(__file__).parent / 'data' / 'precomputed'
PRECOMP_FILE = PRECOMP_DIR / 'etf_data.pkl'

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

# Helper function to get modification times of input files
def get_input_files_hash():
    """Get a hash of all input file modification times for cache invalidation"""
    mtimes = []

    # ETF price files
    for etf in ARK_ETFS:
        price_file = OUTPUT_DIR / f'{etf}_prices.csv'
        if price_file.exists():
            mtimes.append(price_file.stat().st_mtime)

    # If precomputed file exists and is newer than all source files, use it
    if PRECOMP_FILE.exists():
        precomp_mtime = PRECOMP_FILE.stat().st_mtime
        # Only use precomputed if it's newer than all price files
        if mtimes and precomp_mtime >= max(mtimes):
            mtimes.append(precomp_mtime)

    return max(mtimes) if mtimes else 0

# Load and cache ETF price and drawdown data
@st.cache_data
def load_all_etf_data(_files_hash):
    """Load price and drawdown data for all ETFs

    Note: _files_hash parameter (with underscore prefix) is not hashed by Streamlit,
    but changing it will invalidate the cache. This allows cache invalidation when
    input files change.
    """
    # Try to load from precomputed data if it exists and is fresh
    use_precomputed = False
    if PRECOMP_FILE.exists():
        precomp_mtime = PRECOMP_FILE.stat().st_mtime
        # Check if precomputed is newer than all price files
        price_mtimes = []
        for etf in ARK_ETFS:
            price_file = OUTPUT_DIR / f'{etf}_prices.csv'
            if price_file.exists():
                price_mtimes.append(price_file.stat().st_mtime)

        if price_mtimes and precomp_mtime >= max(price_mtimes):
            use_precomputed = True

    if use_precomputed:
        with open(PRECOMP_FILE, 'rb') as f:
            all_data = pickle.load(f)
        price_data = {etf: data['prices'] for etf, data in all_data.items()}
        dd_data = [data['drawdowns'] for etf, data in all_data.items()]
        all_dd = pd.concat(dd_data, ignore_index=True) if dd_data else pd.DataFrame()
        return price_data, all_dd

    # Calculate on the fly (always fresh)
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

# Load data with automatic cache invalidation when input files change
with st.spinner("Loading ETF drawdown data..."):
    files_hash = get_input_files_hash()
    etf_prices, etf_dd = load_all_etf_data(files_hash)

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

        st.plotly_chart(fig, width='stretch', config=CHART_CONFIG)
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
        max_dd = etf_dd_data[etf_dd_data['rank'] == '1'].iloc[0]

        # Calculate RoMaD (Return over Maximum Drawdown)
        price_df = etf_prices[selected_etf]
        first_price = price_df['Close'].iloc[0]
        last_price = price_df['Close'].iloc[-1]
        overall_return = ((last_price - first_price) / first_price) * 100
        max_dd_abs = abs(max_dd['depth_pct'])
        romad = overall_return / max_dd_abs if max_dd_abs > 0 else 0

        st.markdown("### Key Metrics")

        st.metric(
            "Max Drawdown",
            f"{max_dd['depth_pct']:.2f}%",
            delta=None
        )

        st.metric(
            "RoMaD",
            f"{romad:.2f}",
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

        ""  # Space

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

        st.plotly_chart(fig2, width='stretch', config=CHART_CONFIG)
    else:
        st.error(f"No data available for {selected_etf}")

""  # Add space

# Section 3: Current Drawdown Analysis
if selected_etf in etf_prices and len(etf_dd[etf_dd['ETF'] == selected_etf]) > 0:
    etf_dd_data = etf_dd[etf_dd['ETF'] == selected_etf]

    # Check if there is a current drawdown
    if len(etf_dd_data[etf_dd_data['rank'] == 'Current']) > 0:
        current_dd_container = st.container(border=True)
        with current_dd_container:
            st.markdown("#### Current Drawdown Information")

            current_dd = etf_dd_data[etf_dd_data['rank'] == 'Current'].iloc[0]
            price_df = etf_prices[selected_etf]

            current_price = price_df['Close'].iloc[-1]
            current_date = price_df['Date'].iloc[-1]
            peak_price = current_dd['peak_price']
            peak_date = current_dd['peak_date']
            current_dd_pct = current_dd['depth_pct']

            # For Current drawdown, find the actual trough (lowest price) from peak to now
            drawdown_period = price_df[price_df['Date'] >= peak_date]
            actual_trough_price = drawdown_period['Close'].min()
            actual_trough_date = drawdown_period[drawdown_period['Close'] == actual_trough_price]['Date'].iloc[0]

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

            st.markdown("#### Historical Drawdown Analysis for ETF Constituent Stocks")

            st.markdown(f"""
            <small>View historical drawdown records for all constituent stocks in <b>{selected_etf}</b> across different depth ranges to understand their performance and recovery patterns at various levels.</small>
            """, unsafe_allow_html=True)

            ""  # Space

            # Depth range selector
            depth_ranges = ['0% to -10%', '-10% to -20%', '-20% to -30%', '-30% to -40%',
                          '-40% to -50%', '-50% to -60%', '-60% to -70%', '-70% to -80%', '< -80%']

            # Determine default selection based on current drawdown depth
            bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
            current_range_idx = 0  # Default to '0% to -10%'
            for i in range(len(bins) - 1):
                if bins[i] < current_dd_pct <= bins[i+1]:
                    # Reverse index since our depth_ranges list is reversed from bins
                    current_range_idx = len(bins) - 2 - i
                    break

            st.markdown("**Select Drawdown Depth Range:**")
            selected_range = st.pills(
                "Depth Range",
                depth_ranges,
                default=depth_ranges[current_range_idx],
                label_visibility="collapsed",
                key=f"etf_depth_range_{selected_etf}"
            )

            # Get drawdowns in selected range for THIS ETF only
            with st.spinner(f"Loading {selected_etf} historical drawdowns for {selected_range}..."):
                range_drawdowns = get_etf_drawdowns_in_depth_range(selected_etf, selected_range)

            if len(range_drawdowns) > 0:
                # Calculate recovery statistics for constituent stocks
                total_events = len(range_drawdowns)
                recovered_events = range_drawdowns['recovered'].sum()
                recovery_probability = recovered_events / total_events if total_events > 0 else 0

                # Display recovery statistics
                st.markdown(f"""
                **{selected_etf} Constituent Stocks - {selected_range} Historical Statistics:**
                - Total Drawdowns (all stocks): {total_events}
                - Recovered: {recovered_events}
                - **Recovery Rate: {recovery_probability * 100:.1f}%**
                """)

                ""  # Space

                # Display detailed table
                st.markdown(f"**{selected_etf} - All Constituent Stock Drawdowns in {selected_range}:**")

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

                # Select and rename columns (include ticker since showing all constituent stocks)
                display_cols = ['ticker', 'Peak Date', 'Trough Date', 'duration_days',
                              'Depth %', 'Peak Price', 'Trough Price',
                              'Recovered', 'Recovery Date', 'Days to Recover', 'Recovery Rate']

                display_range_dd = display_range_dd[display_cols]
                display_range_dd = display_range_dd.rename(columns={
                    'ticker': 'Ticker',
                    'duration_days': 'Duration (Days)'
                })

                st.dataframe(
                    display_range_dd,
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info(f"{selected_etf} constituent stocks have no historical drawdowns in {selected_range} range.")

""  # Add space

# Section 4: ETF Top 10 Drawdown
st.subheader("ETF Top 10 Drawdown")

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

        # Filter out Current drawdown (shown in separate section above) and show only top 10
        display_df = etf_dd_data[etf_dd_data['rank'] != 'Current'].head(10).copy()

        # Convert rank to string to avoid mixed type issues
        display_df['rank'] = display_df['rank'].astype(str)

        # Format numeric columns as strings
        display_df['Depth %'] = display_df['depth_pct'].apply(lambda x: f"{x:.2f}%")
        display_df['Peak Date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
        display_df['Trough Date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
        display_df['Peak Price'] = display_df['peak_price'].apply(lambda x: f"${x:,.2f}")
        display_df['Trough Price'] = display_df['trough_price'].apply(lambda x: f"${x:,.2f}")

        # Select and rename columns for display
        display_df = display_df[['ETF', 'rank', 'Depth %', 'Peak Date', 'Trough Date', 'Peak Price', 'Trough Price']]
        display_df = display_df.rename(columns={'rank': 'Rank'})

        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True
        )

