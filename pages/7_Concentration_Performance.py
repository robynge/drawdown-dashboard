"""Concentration & Performance Analysis Page - Using Precomputed Data

Analyze portfolio concentration (HHI) and its relationship with ARK's relative performance vs QQQ.
Includes HHI metrics, drawdown visualization, regime analysis, and regression tests.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS, OUTPUT_DIR
from precomputed_loader import (
    load_concentration_performance,
    load_hhi_timeseries,
    filter_by_period,
    check_precomputed_exists
)
from data_loader import load_etf_prices
from chart_config import CHART_CONFIG
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector, is_latest_period

st.set_page_config(
    page_title="Concentration & Performance",
    page_icon="📈",
    layout="wide"
)

# Initialize session state and render period selector
init_session_state()
with st.sidebar:
    render_period_selector()
start_date, end_date = get_current_dates()

"""
# Concentration & Performance Analysis

Analyze portfolio concentration (HHI) and its relationship with ARK's relative performance vs QQQ.
"""

st.markdown(f"**Analysis Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

"" # Space

# Check for precomputed data
if not check_precomputed_exists():
    st.warning("Precomputed data not found. Please run `python convert_to_parquet.py` to generate precomputed data for faster loading.")


@st.cache_data
def calculate_regime_stats(spread_data, regime_col='HHI_Regime'):
    """Calculate statistics for each regime"""
    if len(spread_data) == 0 or regime_col not in spread_data.columns:
        return pd.DataFrame()

    results = []
    for regime in spread_data[regime_col].unique():
        regime_data = spread_data[spread_data[regime_col] == regime]

        if len(regime_data) < 5:
            continue

        # Daily spread statistics
        mean_spread = regime_data['Spread'].mean()
        std_spread = regime_data['Spread'].std()

        # Annualized metrics (252 trading days)
        annualized_spread = mean_spread * 252
        annualized_vol = std_spread * np.sqrt(252)
        sharpe = annualized_spread / annualized_vol if annualized_vol > 0 else 0

        # Win rate (days ARK outperforms QQQ)
        win_rate = (regime_data['Spread'] > 0).mean() * 100

        # Average HHI in regime
        avg_hhi = regime_data['HHI'].mean()

        results.append({
            'Regime': regime,
            'Observations': len(regime_data),
            'Avg HHI': avg_hhi,
            'Mean Daily Spread (%)': mean_spread,
            'Annualized Spread (%)': annualized_spread,
            'Annualized Volatility (%)': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Win Rate (%)': win_rate
        })

    return pd.DataFrame(results)


@st.cache_data
def run_regression_analysis(spread_data):
    """Run regression: Spread ~ HHI"""
    if len(spread_data) < 10:
        return None

    # Clean data
    reg_data = spread_data[['Spread', 'HHI', 'HHI_Change']].dropna()

    if len(reg_data) < 10:
        return None

    results = {}

    # Regression 1: Spread ~ HHI (contemporaneous)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(reg_data['HHI'], reg_data['Spread'])
    t_stat1 = slope1 / std_err1 if std_err1 > 0 else 0
    results['contemporaneous'] = {
        'slope': slope1,
        'intercept': intercept1,
        'r_squared': r_value1 ** 2,
        'p_value': p_value1,
        't_stat': t_stat1,
        'std_err': std_err1,
        'n_obs': len(reg_data)
    }

    # Regression 2: Spread ~ HHI_Change
    reg_data_change = reg_data[reg_data['HHI_Change'].notna()]
    if len(reg_data_change) >= 10:
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            reg_data_change['HHI_Change'], reg_data_change['Spread']
        )
        t_stat2 = slope2 / std_err2 if std_err2 > 0 else 0
        results['hhi_change'] = {
            'slope': slope2,
            'intercept': intercept2,
            'r_squared': r_value2 ** 2,
            'p_value': p_value2,
            't_stat': t_stat2,
            'std_err': std_err2,
            'n_obs': len(reg_data_change)
        }

    # Regression 3: Next day Spread ~ HHI (predictive)
    reg_data['Spread_Next'] = reg_data['Spread'].shift(-1)
    reg_data_pred = reg_data[['HHI', 'Spread_Next']].dropna()
    if len(reg_data_pred) >= 10:
        slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(
            reg_data_pred['HHI'], reg_data_pred['Spread_Next']
        )
        t_stat3 = slope3 / std_err3 if std_err3 > 0 else 0
        results['predictive'] = {
            'slope': slope3,
            'intercept': intercept3,
            'r_squared': r_value3 ** 2,
            'p_value': p_value3,
            't_stat': t_stat3,
            'std_err': std_err3,
            'n_obs': len(reg_data_pred)
        }

    return results


# ETF Selection
st.subheader("Select ETF")

selected_etf = st.pills(
    "ETF",
    options=ARK_ETFS,
    default=ARK_ETFS[0],
    label_visibility="collapsed"
)

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


# Load precomputed data
with st.spinner("Loading data..."):
    # Load precomputed concentration performance data (fast)
    spread_data_full = load_concentration_performance(selected_etf)

    if len(spread_data_full) == 0:
        st.warning(f"No precomputed concentration performance data for {selected_etf}. Run `python convert_to_parquet.py` to generate.")
        st.stop()

    # Filter by analysis period
    spread_data = filter_by_period(spread_data_full, start_date, end_date)

    # Load HHI data for additional metrics
    hhi_data_full = load_hhi_timeseries(selected_etf)
    hhi_data = filter_by_period(hhi_data_full, start_date, end_date) if len(hhi_data_full) > 0 else pd.DataFrame()

    # Load ETF and QQQ prices
    etf_prices = get_cached_etf_prices(0, selected_etf)
    qqq_prices = get_cached_qqq_prices(0)

if len(spread_data) > 0:
    # =========================================================================
    # Section 0: Current Concentration Metrics
    # =========================================================================
    if len(hhi_data) > 0:
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

    # =========================================================================
    # Section 0b: Price & Concentration with Drawdowns
    # =========================================================================
    if len(etf_prices) > 0 and len(hhi_data) > 0:
        st.subheader("Price & Concentration")

        # Filter ETF prices to analysis period
        price_df = etf_prices[
            (etf_prices['Date'] >= start_date) &
            (etf_prices['Date'] <= end_date)
        ].copy()

        # Toggle for showing drawdowns
        show_drawdowns = st.toggle("Show Drawdowns", value=True)

        price_card = st.container(border=True)
        with price_card:
            # Calculate drawdowns within analysis period
            from drawdown_calculator import calculate_drawdowns
            _etf_prices = load_etf_prices(selected_etf)
            if len(_etf_prices) > 0:
                etf_dd_data = calculate_drawdowns(_etf_prices[['Date', 'Close']].copy(), start_date=start_date, end_date=end_date)
            else:
                etf_dd_data = pd.DataFrame()

            # Create figure
            fig_price = go.Figure()

            # Add drawdown shaded regions (if enabled)
            if show_drawdowns and len(etf_dd_data) > 0:
                top_10_dd = etf_dd_data[etf_dd_data['rank'] != 'Current'].head(10)

                # Color palette for drawdowns
                dd_colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
                             'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)', 'rgba(221, 160, 221, 0.3)',
                             'rgba(255, 192, 203, 0.3)', 'rgba(176, 224, 230, 0.3)', 'rgba(240, 230, 140, 0.3)',
                             'rgba(255, 228, 181, 0.3)']

                for idx, (_, row) in enumerate(top_10_dd.iterrows()):
                    fig_price.add_vrect(
                        x0=row['peak_date'],
                        x1=row['trough_date'],
                        fillcolor=dd_colors[idx % len(dd_colors)],
                        layer="below",
                        line_width=0
                    )

            # ETF price line
            fig_price.add_trace(go.Scatter(
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

                    fig_price.add_trace(go.Scatter(
                        x=qqq_filtered['Date'],
                        y=qqq_filtered['Close'],
                        mode='lines',
                        name='QQQ Price',
                        line=dict(color='orange', width=2),
                        hovertemplate='<b>QQQ</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>',
                        yaxis='y3'
                    ))

            # Add current drawdown line and shaded area (only for latest period, if enabled and current drawdown exists)
            current_dd_rows = etf_dd_data[etf_dd_data['rank'] == 'Current'] if len(etf_dd_data) > 0 else pd.DataFrame()
            if is_latest_period() and show_drawdowns and len(current_dd_rows) > 0:
                current_dd = current_dd_rows.iloc[0]
                peak_price = current_dd['peak_price']
                peak_date = current_dd['peak_date']
                current_price = current_dd['trough_price']
                current_dd_pct = current_dd['depth_pct']

                fig_price.add_shape(
                    type="line",
                    x0=peak_date,
                    x1=price_df['Date'].max(),
                    y0=peak_price,
                    y1=peak_price,
                    line=dict(color='red', width=2, dash='dash'),
                    layer='above'
                )

                fig_price.add_shape(
                    type="rect",
                    x0=peak_date,
                    x1=price_df['Date'].max(),
                    y0=current_price,
                    y1=peak_price,
                    fillcolor='rgba(128,128,128,0.25)',
                    line=dict(width=0),
                    layer='below'
                )

                fig_price.add_annotation(
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
            fig_price.add_trace(go.Scatter(
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

            fig_price.update_layout(
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

            st.plotly_chart(fig_price, width='stretch', config=CHART_CONFIG)

            st.markdown("<small>*Colored regions show top 10 historical drawdowns. Gray region shows current drawdown. QQQ Y-axis scaled so first day aligns with ETF.*</small>", unsafe_allow_html=True)

        "" # Space

    # =========================================================================
    # Section 1: Relative Performance (Spread)
    # =========================================================================
    st.subheader("Relative Performance (Spread)")

    spread_card = st.container(border=True)
    with spread_card:
        # Returns & Concentration chart
        fig_returns = make_subplots(specs=[[{"secondary_y": True}]])

        # ETF cumulative return
        fig_returns.add_trace(
            go.Scatter(
                x=spread_data['Date'],
                y=spread_data['ETF_Cumulative'],
                mode='lines',
                name=f'{selected_etf} Return',
                line=dict(color='black', width=2),
                hovertemplate=f'<b>{selected_etf} Return</b><br>Date: %{{x|%Y-%m-%d}}<br>Return: %{{y:.2f}}%<extra></extra>'
            ),
            secondary_y=False
        )

        # QQQ cumulative return
        fig_returns.add_trace(
            go.Scatter(
                x=spread_data['Date'],
                y=spread_data['QQQ_Cumulative'],
                mode='lines',
                name='QQQ Return',
                line=dict(color='orange', width=2),
                hovertemplate='<b>QQQ Return</b><br>Date: %{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            secondary_y=False
        )

        # Cumulative spread (ARK - QQQ)
        fig_returns.add_trace(
            go.Scatter(
                x=spread_data['Date'],
                y=spread_data['Cumulative_Spread'],
                mode='lines',
                name='Spread (ARK - QQQ)',
                line=dict(color='green', width=2, dash='dash'),
                hovertemplate='<b>Cumulative Spread</b><br>Date: %{x|%Y-%m-%d}<br>Spread: %{y:.2f}%<extra></extra>'
            ),
            secondary_y=False
        )

        # HHI on secondary axis
        fig_returns.add_trace(
            go.Scatter(
                x=spread_data['Date'],
                y=spread_data['HHI'],
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

        # Summary stats
        total_spread = spread_data['Cumulative_Spread'].iloc[-1]
        avg_daily_spread = spread_data['Spread'].mean()
        win_rate = (spread_data['Spread'] > 0).mean() * 100

        stat_cols = st.columns(3)
        with stat_cols[0]:
            delta_color = "normal" if total_spread >= 0 else "inverse"
            st.metric(
                "Total Relative Return",
                f"{total_spread:+.2f}%",
                delta="Outperforming" if total_spread >= 0 else "Underperforming",
                delta_color=delta_color
            )
        with stat_cols[1]:
            st.metric("Avg Daily Spread", f"{avg_daily_spread:+.4f}%")
        with stat_cols[2]:
            st.metric("Win Rate", f"{win_rate:.1f}%", help="% of days ARK outperformed QQQ")

    "" # Space

    # =========================================================================
    # Section 2: HHI Regime Analysis
    # =========================================================================
    st.subheader("HHI Regime Analysis")

    regime_card = st.container(border=True)
    with regime_card:
        st.markdown("""
        Split HHI history into 3 regimes based on percentiles:
        - **Concentrated** (Top 1/3): High HHI periods
        - **Moderate** (Middle 1/3): Medium HHI periods
        - **Diversified** (Bottom 1/3): Low HHI periods
        """)

        # Calculate HHI percentiles and assign regimes
        hhi_33 = spread_data['HHI'].quantile(0.33)
        hhi_67 = spread_data['HHI'].quantile(0.67)

        def assign_regime(hhi):
            if hhi <= hhi_33:
                return 'Diversified (Bottom 1/3)'
            elif hhi <= hhi_67:
                return 'Moderate (Middle 1/3)'
            else:
                return 'Concentrated (Top 1/3)'

        spread_data['HHI_Regime'] = spread_data['HHI'].apply(assign_regime)

        # Display thresholds
        st.markdown(f"**HHI Thresholds:** Diversified ≤ {hhi_33:.4f} < Moderate ≤ {hhi_67:.4f} < Concentrated")

        "" # Space

        # Calculate regime statistics
        regime_stats = calculate_regime_stats(spread_data, 'HHI_Regime')

        if len(regime_stats) > 0:
            # Order regimes logically
            regime_order = ['Diversified (Bottom 1/3)', 'Moderate (Middle 1/3)', 'Concentrated (Top 1/3)']
            regime_stats['Regime'] = pd.Categorical(regime_stats['Regime'], categories=regime_order, ordered=True)
            regime_stats = regime_stats.sort_values('Regime')

            # Display stats table
            st.markdown("#### Regime Performance Statistics")

            display_stats = regime_stats.copy()
            display_stats['Avg HHI'] = display_stats['Avg HHI'].apply(lambda x: f"{x:.4f}")
            display_stats['Mean Daily Spread (%)'] = display_stats['Mean Daily Spread (%)'].apply(lambda x: f"{x:+.4f}")
            display_stats['Annualized Spread (%)'] = display_stats['Annualized Spread (%)'].apply(lambda x: f"{x:+.2f}")
            display_stats['Annualized Volatility (%)'] = display_stats['Annualized Volatility (%)'].apply(lambda x: f"{x:.2f}")
            display_stats['Sharpe Ratio'] = display_stats['Sharpe Ratio'].apply(lambda x: f"{x:+.3f}")
            display_stats['Win Rate (%)'] = display_stats['Win Rate (%)'].apply(lambda x: f"{x:.1f}")

            st.dataframe(display_stats, hide_index=True, width='stretch')

            "" # Space

            # Visualize regime performance
            st.markdown("#### Regime Performance Comparison")

            fig_regime = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Annualized Spread (%)", "Sharpe Ratio", "Win Rate (%)")
            )

            regime_colors = {'Diversified (Bottom 1/3)': 'green', 'Moderate (Middle 1/3)': 'gray', 'Concentrated (Top 1/3)': 'red'}

            for col_idx, metric in enumerate(['Annualized Spread (%)', 'Sharpe Ratio', 'Win Rate (%)'], 1):
                metric_vals = regime_stats[metric.replace('(%)', '(%)').replace('Ratio', 'Ratio')].astype(float).values
                fig_regime.add_trace(
                    go.Bar(
                        x=regime_stats['Regime'].astype(str).values,
                        y=metric_vals,
                        marker_color=[regime_colors.get(r, 'gray') for r in regime_stats['Regime'].astype(str).values],
                        showlegend=False,
                        hovertemplate='%{x}<br>' + metric + ': %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=col_idx
                )

            fig_regime.update_layout(
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            fig_regime.update_yaxes(gridcolor='lightgray')

            st.plotly_chart(fig_regime, width='stretch', config=CHART_CONFIG)

            "" # Space

            # Cumulative return by regime
            st.markdown("#### Cumulative Spread by Regime")

            fig_regime_cum = go.Figure()

            for regime in regime_order:
                regime_mask = spread_data['HHI_Regime'] == regime
                regime_spreads = spread_data.loc[regime_mask, 'Spread'].values

                # Calculate cumulative return for this regime only
                cumulative = np.cumprod(1 + regime_spreads / 100) - 1
                cumulative_pct = cumulative * 100

                fig_regime_cum.add_trace(
                    go.Scatter(
                        x=list(range(len(cumulative_pct))),
                        y=cumulative_pct,
                        mode='lines',
                        name=regime,
                        line=dict(color=regime_colors.get(regime, 'gray'), width=2),
                        hovertemplate=regime + '<br>Day: %{x}<br>Cumulative: %{y:.2f}%<extra></extra>'
                    )
                )

            fig_regime_cum.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

            fig_regime_cum.update_layout(
                title="Cumulative Spread if Only Invested During Each Regime",
                xaxis_title="Trading Days in Regime",
                yaxis_title="Cumulative Spread (%)",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            fig_regime_cum.update_xaxes(gridcolor='lightgray')
            fig_regime_cum.update_yaxes(gridcolor='lightgray')

            st.plotly_chart(fig_regime_cum, width='stretch', config=CHART_CONFIG)

    "" # Space

    # =========================================================================
    # Section 3: Regression Analysis
    # =========================================================================
    st.subheader("Regression Analysis")

    reg_card = st.container(border=True)
    with reg_card:
        regression_results = run_regression_analysis(spread_data)

        if regression_results:
            st.markdown("""
            Test the statistical relationship between concentration and relative performance:
            - **Contemporaneous**: Does HHI level explain same-day spread?
            - **HHI Change**: Does change in HHI explain spread?
            - **Predictive**: Does today's HHI predict tomorrow's spread?
            """)

            "" # Space

            # Display regression results
            reg_df_rows = []

            if 'contemporaneous' in regression_results:
                r = regression_results['contemporaneous']
                reg_df_rows.append({
                    'Model': 'Spread ~ HHI',
                    'Coefficient': f"{r['slope']:.4f}",
                    't-statistic': f"{r['t_stat']:.2f}",
                    'p-value': f"{r['p_value']:.4f}" if r['p_value'] >= 0.0001 else "<0.0001",
                    'R²': f"{r['r_squared']:.4f}",
                    'N': r['n_obs'],
                    'Significant': '✓' if r['p_value'] < 0.05 else ''
                })

            if 'hhi_change' in regression_results:
                r = regression_results['hhi_change']
                reg_df_rows.append({
                    'Model': 'Spread ~ ΔHHI',
                    'Coefficient': f"{r['slope']:.4f}",
                    't-statistic': f"{r['t_stat']:.2f}",
                    'p-value': f"{r['p_value']:.4f}" if r['p_value'] >= 0.0001 else "<0.0001",
                    'R²': f"{r['r_squared']:.4f}",
                    'N': r['n_obs'],
                    'Significant': '✓' if r['p_value'] < 0.05 else ''
                })

            if 'predictive' in regression_results:
                r = regression_results['predictive']
                reg_df_rows.append({
                    'Model': 'Spread(t+1) ~ HHI(t)',
                    'Coefficient': f"{r['slope']:.4f}",
                    't-statistic': f"{r['t_stat']:.2f}",
                    'p-value': f"{r['p_value']:.4f}" if r['p_value'] >= 0.0001 else "<0.0001",
                    'R²': f"{r['r_squared']:.4f}",
                    'N': r['n_obs'],
                    'Significant': '✓' if r['p_value'] < 0.05 else ''
                })

            reg_df = pd.DataFrame(reg_df_rows)
            st.dataframe(reg_df, hide_index=True, width='stretch')

            "" # Space

            # Scatter plot: HHI vs Spread
            st.markdown("#### HHI vs Daily Spread")

            fig_scatter = go.Figure()

            fig_scatter.add_trace(
                go.Scatter(
                    x=spread_data['HHI'],
                    y=spread_data['Spread'],
                    mode='markers',
                    marker=dict(
                        color=spread_data['Spread'],
                        colorscale='RdYlGn',
                        cmin=-2,
                        cmax=2,
                        size=5,
                        opacity=0.6
                    ),
                    hovertemplate='HHI: %{x:.4f}<br>Spread: %{y:.2f}%<extra></extra>'
                )
            )

            # Add regression line
            if 'contemporaneous' in regression_results:
                r = regression_results['contemporaneous']
                x_range = np.linspace(spread_data['HHI'].min(), spread_data['HHI'].max(), 100)
                y_pred = r['intercept'] + r['slope'] * x_range

                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f"Regression (β={r['slope']:.4f})",
                        line=dict(color='black', width=2, dash='dash')
                    )
                )

            fig_scatter.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

            fig_scatter.update_layout(
                xaxis_title="HHI",
                yaxis_title="Daily Spread (%)",
                height=450,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            fig_scatter.update_xaxes(gridcolor='lightgray')
            fig_scatter.update_yaxes(gridcolor='lightgray')

            st.plotly_chart(fig_scatter, width='stretch', config=CHART_CONFIG)

            # Interpretation
            if 'contemporaneous' in regression_results:
                r = regression_results['contemporaneous']
                if r['p_value'] < 0.05:
                    if r['slope'] > 0:
                        st.success(f"**Significant positive relationship**: Higher HHI is associated with higher spread (ARK outperforming QQQ). Coefficient = {r['slope']:.4f}, p = {r['p_value']:.4f}")
                    else:
                        st.warning(f"**Significant negative relationship**: Higher HHI is associated with lower spread (ARK underperforming QQQ). Coefficient = {r['slope']:.4f}, p = {r['p_value']:.4f}")
                else:
                    st.info(f"**No significant relationship** between HHI and daily spread. p = {r['p_value']:.4f}")

        else:
            st.warning("Not enough data for regression analysis")

    "" # Space

    # =========================================================================
    # Section 4: Download Data
    # =========================================================================
    st.subheader("Download Data")

    download_card = st.container(border=True)
    with download_card:
        st.markdown("Download the analysis data for external use (e.g., further statistical analysis in R/Python).")

        # Prepare download data
        download_df = spread_data[[
            'Date', 'ETF_Price', 'QQQ_Price',
            'ETF_Daily_Return', 'QQQ_Daily_Return', 'Spread',
            'ETF_Cumulative', 'QQQ_Cumulative', 'Cumulative_Spread',
            'HHI', 'Effective_Positions', 'HHI_Change', 'HHI_Regime'
        ]].copy()

        csv = download_df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_etf}_concentration_performance.csv",
            mime="text/csv"
        )

        st.markdown(f"**Data shape:** {len(download_df)} rows × {len(download_df.columns)} columns")

else:
    st.warning(f"Not enough data for {selected_etf}")
