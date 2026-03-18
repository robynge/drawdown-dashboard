"""Conviction vs Drawdown Analysis Page

Compares drawdown depth and recovery speed between high-conviction (high weight)
and low-conviction (low weight) holdings in ARK ETFs.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ARK_ETFS
from precomputed_loader import (
    load_conviction_drawdowns,
    check_precomputed_exists,
)
from session_utils import init_session_state, get_current_dates, get_current_period, render_period_selector

st.set_page_config(page_title="Conviction vs Drawdown", layout="wide")

# Display labels for conviction levels
WEIGHT_LABELS = {'High': '≥5%', 'Mid': '1%-5%', 'Low': '<1%'}
WEIGHT_ORDER = ['≥5%', '1%-5%', '<1%']
COLORS = {'≥5%': '#D62728', '1%-5%': '#1F77B4', '<1%': '#F5A623'}
CONVICTION_ORDER = ['High', 'Mid', 'Low']

# Shared layout: remove top/right spines, light grid, clean font
AXIS_STYLE = dict(
    showline=True, linewidth=2, linecolor='#272727',
    showgrid=True, gridcolor='rgba(0,0,0,0.07)',
)
LAYOUT_COMMON = dict(
    font_family='Arial, Helvetica, sans-serif',
    font_size=14,
    plot_bgcolor='white',
    legend=dict(
        bgcolor='rgba(0,0,0,0)', borderwidth=0,
        font_size=13,
    ),
    margin=dict(t=40, b=60),
)

# Initialize session state
init_session_state()

# Sidebar
with st.sidebar:
    selected_period = render_period_selector()

start_date, end_date = get_current_dates()

st.title("Conviction vs Drawdown Analysis")
st.markdown(f"**Analysis Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# ETF selector
selected_etf = st.pills(
    "ETF",
    options=ARK_ETFS,
    default=ARK_ETFS[0],
    label_visibility="collapsed",
    key="conviction_etf_selector"
)
if selected_etf is None:
    selected_etf = ARK_ETFS[0]

# Check precomputed data
if not check_precomputed_exists():
    st.error("Precomputed data not found. Run `python convert_to_parquet.py` first.")
    st.stop()

# Load data
period_key = get_current_period()

with st.spinner(f"Loading conviction drawdown data for {selected_etf}..."):
    df_period = load_conviction_drawdowns(selected_etf, period_key=period_key)

if len(df_period) == 0:
    st.warning(
        f"No conviction drawdown data for {selected_etf} ({period_key}). "
        "Run `python convert_to_parquet.py --step 24`."
    )
    st.stop()

df_period = df_period.copy()

# Map conviction to weight labels for display
df_period['weight'] = df_period['conviction'].map(WEIGHT_LABELS)

# Compute net PnL per event (drawdown + recovery, recovery is 0 if not recovered)
df_period['net_pnl'] = df_period['adj_pnl'] + df_period['recovery_adj_pnl'].fillna(0)

# ============================================================================
# Section 1: Summary Statistics
# ============================================================================
st.header("Summary Statistics")

summary_rows = []
for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        summary_rows.append({
            'Weight': label, 'Holdings': 0, 'Drawdown Events': 0,
            'Avg Depth (%)': None, 'Median Depth (%)': None,
            'Recovery Rate (%)': None,
            'Avg Recovery Days': None, 'Median Recovery Days': None,
            'Total Drawdown PnL ($)': None, 'Total Net PnL ($)': None,
        })
        continue

    n_tickers = group['ticker'].nunique()
    n_events = len(group)
    avg_depth = group['depth_pct'].mean()
    med_depth = group['depth_pct'].median()
    recovery_rate = group['recovered'].mean() * 100
    recovered_group = group[group['recovered'] & group['days_to_recover'].notna()]
    avg_recovery = recovered_group['days_to_recover'].mean() if len(recovered_group) > 0 else None
    med_recovery = recovered_group['days_to_recover'].median() if len(recovered_group) > 0 else None

    summary_rows.append({
        'Weight': label,
        'Holdings': n_tickers,
        'Drawdown Events': n_events,
        'Avg Depth (%)': round(avg_depth, 2),
        'Median Depth (%)': round(med_depth, 2),
        'Recovery Rate (%)': round(recovery_rate, 1),
        'Avg Recovery Days': round(avg_recovery) if avg_recovery is not None else None,
        'Median Recovery Days': round(med_recovery) if med_recovery is not None else None,
        'Total Drawdown PnL ($)': f"${group['adj_pnl'].sum():,.0f}",
        'Total Net PnL ($)': f"${group['net_pnl'].sum():,.0f}",
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, width='stretch', hide_index=True)

# ============================================================================
# Section 2: Drawdown Depth Distribution — Violin + Strip
# ============================================================================
st.header("Drawdown Depth Distribution by Weight")

fig_violin = go.Figure()

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    fig_violin.add_trace(go.Violin(
        y=group['depth_pct'],
        name=f"{label} (n={len(group)})",
        line_color=color,
        fillcolor=color,
        opacity=0.4,
        meanline_visible=True,
        box_visible=True,
        points=False,
        scalemode='width',
        width=0.7,
        hoverinfo='y',
    ))

fig_violin.update_layout(
    **LAYOUT_COMMON,
    height=520,
    yaxis=dict(title='Drawdown Depth (%)', **AXIS_STYLE),
    xaxis=dict(title='Weight', **AXIS_STYLE),
    showlegend=True,
    violingap=0.35,
    violinmode='group',
)
st.plotly_chart(fig_violin, width='stretch')

# ============================================================================
# Section 3: Recovery Curve (Kaplan-Meier style)
# ============================================================================
st.header("Recovery Curve by Weight")
st.caption("Cumulative % of drawdowns that recovered within N days after trough")

fig_km = go.Figure()

max_days = 500  # x-axis limit
day_range = np.arange(0, max_days + 1, 1)

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue

    n_total = len(group)
    recovery_days = group['days_to_recover'].copy()

    cumulative_pct = []
    for d in day_range:
        recovered_by_d = (recovery_days <= d).sum()
        cumulative_pct.append(recovered_by_d / n_total * 100)

    color = COLORS[label]
    fig_km.add_trace(go.Scatter(
        x=day_range,
        y=cumulative_pct,
        mode='lines',
        name=f"{label} (n={n_total})",
        line=dict(color=color, width=2.5),
        hovertemplate='Day %{x}: %{y:.1f}% recovered<extra></extra>',
    ))

# Reference line at 50%
fig_km.add_hline(y=50, line_dash='dot', line_color='#767676', line_width=1,
                 annotation_text='50%', annotation_position='left')

fig_km.update_layout(
    **LAYOUT_COMMON,
    height=480,
    xaxis=dict(title='Days After Trough', **AXIS_STYLE, range=[0, max_days]),
    yaxis=dict(title='Cumulative Recovery Rate (%)', **AXIS_STYLE, range=[0, 105]),
)
st.plotly_chart(fig_km, width='stretch')

# ============================================================================
# Section 4: Drawdown Depth vs Recovery Days (scatter, color=weight)
# ============================================================================
st.header("Drawdown Depth vs Recovery Days")

recovered_df = df_period[df_period['recovered'] & df_period['days_to_recover'].notna()].copy()

if len(recovered_df) > 0:
    recovered_df['weight'] = recovered_df['conviction'].map(WEIGHT_LABELS)
    fig_depth_rec = go.Figure()

    for conv, label in WEIGHT_LABELS.items():
        group = recovered_df[recovered_df['conviction'] == conv]
        if len(group) == 0:
            continue
        color = COLORS[label]

        hover_texts = [
            f"{row['ticker']} (DD #{row['rank']})" for _, row in group.iterrows()
        ]

        fig_depth_rec.add_trace(go.Scatter(
            x=group['depth_pct'].abs(),
            y=group['days_to_recover'],
            mode='markers',
            name=f"{label} (n={len(group)})",
            marker=dict(color=color, size=6, opacity=0.6,
                        line=dict(width=0.5, color='#272727')),
            hovertext=hover_texts,
            hovertemplate='%{hovertext}<br>Depth: -%{x:.1f}%<br>Recovery: %{y:.0f} days<extra></extra>',
        ))

        # Trend line
        if len(group) >= 5:
            x = group['depth_pct'].abs().values
            y = group['days_to_recover'].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 5:
                coeffs = np.polyfit(x[mask], y[mask], 1)
                x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
                y_line = np.polyval(coeffs, x_line)
                fig_depth_rec.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode='lines',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=False, hoverinfo='skip',
                ))

    fig_depth_rec.update_layout(
        **LAYOUT_COMMON,
        height=520,
        xaxis=dict(title='Drawdown Depth (%, absolute)', **AXIS_STYLE),
        yaxis=dict(title='Days to Recovery', **AXIS_STYLE),
    )
    st.plotly_chart(fig_depth_rec, width='stretch')
else:
    st.info("No recovered drawdowns in this period.")

# ============================================================================
# Section 5: Weight vs Drawdown Depth (full width)
# ============================================================================
st.header("Weight vs Drawdown Depth")

fig_depth = go.Figure()

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    hover_texts = [
        f"{row['ticker']} (DD #{row['rank']})" for _, row in group.iterrows()
    ]

    fig_depth.add_trace(go.Scatter(
        x=group['weight_at_peak'],
        y=group['depth_pct'],
        mode='markers',
        name=f"{label} (n={len(group)})",
        marker=dict(color=color, size=6, opacity=0.6,
                    line=dict(width=0.5, color='#272727')),
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<br>Weight: %{x:.4f}%<br>Depth: %{y:.1f}%<extra></extra>',
    ))

    if len(group) >= 5:
        x = group['weight_at_peak'].values
        y = group['depth_pct'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
            y_line = np.polyval(coeffs, x_line)
            fig_depth.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
            ))

fig_depth.update_layout(
    **LAYOUT_COMMON,
    height=520,
    xaxis=dict(title='Weight (%)', **AXIS_STYLE),
    yaxis=dict(title='Drawdown Depth (%)', **AXIS_STYLE),
)
st.plotly_chart(fig_depth, width='stretch')

# ============================================================================
# Section 6: Aggregate PnL Analysis (replaces individual PnL scatters)
# ============================================================================
st.header("Aggregate PnL Impact by Conviction")

col_agg_pnl, col_eff = st.columns(2)

# --- Left: Total Drawdown PnL vs Recovery PnL vs Net PnL ---
agg_data = []
for conv in CONVICTION_ORDER:
    label = WEIGHT_LABELS[conv]
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    total_dd_pnl = group['adj_pnl'].sum()
    total_rec_pnl = group['recovery_adj_pnl'].fillna(0).sum()
    total_net = total_dd_pnl + total_rec_pnl
    agg_data.append({
        'conviction': conv, 'label': label,
        'Drawdown PnL': total_dd_pnl,
        'Recovery PnL': total_rec_pnl,
        'Net PnL': total_net,
    })

agg_df = pd.DataFrame(agg_data)

fig_agg = go.Figure()

# Drawdown PnL bars (expect negative)
fig_agg.add_trace(go.Bar(
    x=agg_df['label'],
    y=agg_df['Drawdown PnL'],
    name='Drawdown PnL',
    marker_color='#D62728',
    opacity=0.8,
    hovertemplate='%{x}<br>Drawdown PnL: $%{y:,.0f}<extra></extra>',
))

# Recovery PnL bars (expect positive)
fig_agg.add_trace(go.Bar(
    x=agg_df['label'],
    y=agg_df['Recovery PnL'],
    name='Recovery PnL',
    marker_color='#2CA02C',
    opacity=0.8,
    hovertemplate='%{x}<br>Recovery PnL: $%{y:,.0f}<extra></extra>',
))

# Net PnL as small circles with horizontal dashed line from y-axis
for i, row in agg_df.iterrows():
    fig_agg.add_shape(
        type='line',
        x0=0, x1=row['label'],
        y0=row['Net PnL'], y1=row['Net PnL'],
        xref='paper', yref='y',
        line=dict(color='#272727', width=1.5, dash='dash'),
    )

fig_agg.add_trace(go.Scatter(
    x=agg_df['label'],
    y=agg_df['Net PnL'],
    mode='markers+text',
    name='Net PnL',
    marker=dict(color='#272727', size=8, symbol='circle'),
    text=[f"${v:,.0f}" for v in agg_df['Net PnL']],
    textposition='top center',
    textfont=dict(size=12, color='#272727'),
    hovertemplate='%{x}<br>Net PnL: $%{y:,.0f}<extra></extra>',
))

fig_agg.update_layout(
    **LAYOUT_COMMON,
    height=520,
    barmode='group',
    xaxis=dict(title='Conviction', **AXIS_STYLE),
    yaxis=dict(title='Total PnL ($)', **AXIS_STYLE),
)

with col_agg_pnl:
    st.subheader("Total PnL by Conviction")
    st.plotly_chart(fig_agg, width='stretch')

# --- Right: Recovery Efficiency ---
eff_data = []
for conv in CONVICTION_ORDER:
    label = WEIGHT_LABELS[conv]
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue

    total_dd_pnl = group['adj_pnl'].sum()
    total_rec_pnl = group['recovery_adj_pnl'].fillna(0).sum()
    # Recovery efficiency: how much of the drawdown loss was recovered
    # (drawdown PnL is typically negative, so we use abs)
    efficiency = (total_rec_pnl / abs(total_dd_pnl) * 100) if total_dd_pnl != 0 else 0

    # Net negative rate: % of events with net negative PnL
    net_negative_rate = (group['net_pnl'] < 0).mean() * 100

    eff_data.append({
        'label': label,
        'Recovery Efficiency (%)': efficiency,
        'Net Negative Rate (%)': net_negative_rate,
    })

eff_df = pd.DataFrame(eff_data)

fig_eff = go.Figure()

fig_eff.add_trace(go.Bar(
    x=eff_df['label'],
    y=eff_df['Recovery Efficiency (%)'],
    name='PnL Recovery Efficiency',
    marker_color='#2CA02C',
    opacity=0.8,
    hovertemplate='%{x}<br>Recovery Efficiency: %{y:.1f}%<extra></extra>',
))

fig_eff.add_trace(go.Bar(
    x=eff_df['label'],
    y=eff_df['Net Negative Rate (%)'],
    name='Net Negative Event Rate',
    marker_color='#D62728',
    opacity=0.8,
    hovertemplate='%{x}<br>Net Negative: %{y:.1f}%<extra></extra>',
))

fig_eff.update_layout(
    **LAYOUT_COMMON,
    height=520,
    barmode='group',
    xaxis=dict(title='Conviction', **AXIS_STYLE),
    yaxis=dict(title='%', **AXIS_STYLE, range=[0, 105]),
)

with col_eff:
    st.subheader("Recovery Efficiency")
    st.plotly_chart(fig_eff, width='stretch')

# ============================================================================
# Section 7: "Death by a Thousand Cuts" — Per-Stock Net PnL for Low Conviction
# ============================================================================
st.header("Small Position Net Drawdown Impact by Stock")
st.caption(
    "Net PnL (Drawdown + Recovery) summed per stock for **<1% weight** positions. "
    "Red = net loss, green = net gain."
)

low_df = df_period[df_period['conviction'] == 'Low'].copy()

if len(low_df) > 0:
    # Sum net PnL per stock
    stock_net = low_df.groupby('ticker').agg(
        net_pnl=('net_pnl', 'sum'),
        n_events=('net_pnl', 'count'),
        avg_weight=('weight_at_peak', 'mean'),
    ).reset_index()
    stock_net = stock_net.sort_values('net_pnl', ascending=True)

    # Color by sign
    bar_colors = ['#D62728' if v < 0 else '#2CA02C' for v in stock_net['net_pnl']]

    fig_cuts = go.Figure()
    fig_cuts.add_trace(go.Bar(
        y=stock_net['ticker'],
        x=stock_net['net_pnl'],
        orientation='h',
        marker_color=bar_colors,
        opacity=0.85,
        customdata=np.stack([stock_net['n_events'], stock_net['avg_weight']], axis=-1),
        hovertemplate=(
            '%{y}<br>'
            'Net PnL: $%{x:,.0f}<br>'
            'Events: %{customdata[0]:.0f}<br>'
            'Avg Weight: %{customdata[1]:.4f}%'
            '<extra></extra>'
        ),
    ))

    # Summary annotation
    total_loss = stock_net[stock_net['net_pnl'] < 0]['net_pnl'].sum()
    total_gain = stock_net[stock_net['net_pnl'] >= 0]['net_pnl'].sum()
    n_negative = (stock_net['net_pnl'] < 0).sum()
    n_positive = (stock_net['net_pnl'] >= 0).sum()

    fig_cuts.update_layout(
        **LAYOUT_COMMON,
        height=max(400, len(stock_net) * 22 + 80),
        xaxis=dict(title='Net PnL ($)', **AXIS_STYLE),
        yaxis=dict(
            title='', **AXIS_STYLE,
            tickfont=dict(size=11),
            autorange='reversed',  # most negative at top
        ),
        showlegend=False,
    )

    # Add vertical line at 0
    fig_cuts.add_vline(x=0, line_color='#272727', line_width=1.5)

    st.plotly_chart(fig_cuts, width='stretch')

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Stocks with Net Loss", f"{n_negative}")
    mc2.metric("Stocks with Net Gain", f"{n_positive}")
    mc3.metric("Total Net Loss", f"${total_loss:,.0f}")
    mc4.metric("Total Net Gain", f"${total_gain:,.0f}")
else:
    st.info("No Low conviction drawdown events in this period.")

# ============================================================================
# Section 8: Duration vs Depth Scatter (color=weight)
# ============================================================================
st.header("Drawdown Duration vs Depth")

fig_dur = go.Figure()

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    hover_texts = [
        f"{row['ticker']} (DD #{row['rank']})" for _, row in group.iterrows()
    ]

    fig_dur.add_trace(go.Scatter(
        x=group['duration_days'],
        y=group['depth_pct'],
        mode='markers',
        name=f"{label} (n={len(group)})",
        marker=dict(color=color, size=6, opacity=0.6,
                    line=dict(width=0.5, color='#272727')),
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<br>Duration: %{x} days<br>Depth: %{y:.1f}%<extra></extra>',
    ))

    # Trend line
    if len(group) >= 5:
        x = group['duration_days'].values.astype(float)
        y = group['depth_pct'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
            y_line = np.polyval(coeffs, x_line)
            fig_dur.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False, hoverinfo='skip',
            ))

fig_dur.update_layout(
    **LAYOUT_COMMON,
    height=520,
    xaxis=dict(title='Duration (days, peak to trough)', **AXIS_STYLE),
    yaxis=dict(title='Drawdown Depth (%)', **AXIS_STYLE),
)
st.plotly_chart(fig_dur, width='stretch')

# ============================================================================
# Section 9: Detailed Table
# ============================================================================
with st.expander("Detailed Data Table", expanded=False):
    display_df = df_period[[
        'ticker', 'rank', 'weight', 'weight_at_peak', 'peak_date', 'trough_date',
        'peak_price', 'trough_price', 'depth_pct', 'adj_pnl', 'recovery_adj_pnl',
        'net_pnl', 'duration_days', 'recovered', 'recovery_date', 'days_to_recover'
    ]].copy()
    display_df['peak_date'] = display_df['peak_date'].dt.strftime('%Y-%m-%d')
    display_df['trough_date'] = display_df['trough_date'].dt.strftime('%Y-%m-%d')
    display_df['recovery_date'] = display_df['recovery_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
    )
    display_df = display_df.sort_values('depth_pct', ascending=True)
    st.dataframe(display_df, width='stretch', hide_index=True)

    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{selected_etf}_conviction_drawdowns.csv",
        mime="text/csv",
    )
