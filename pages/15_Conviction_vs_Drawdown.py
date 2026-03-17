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
    filter_drawdowns_by_period
)
from session_utils import init_session_state, get_current_dates, render_period_selector

st.set_page_config(page_title="Conviction vs Drawdown", layout="wide")

# Display labels for conviction levels
WEIGHT_LABELS = {'High': '≥5%', 'Mid': '1%-5%', 'Low': '<1%'}
WEIGHT_ORDER = ['≥5%', '1%-5%', '<1%']
# Semantic colors: blue=key group, teal=secondary, grey=baseline
COLORS = {'≥5%': '#0F4D92', '1%-5%': '#42949E', '<1%': '#CFCECE'}

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
with st.spinner(f"Loading conviction drawdown data for {selected_etf}..."):
    df = load_conviction_drawdowns(selected_etf)

if len(df) == 0:
    st.warning(
        f"No conviction drawdown data for {selected_etf}. "
        "Run `python convert_to_parquet.py --step 24`."
    )
    st.stop()

# Filter by period
df_period = df[
    (df['peak_date'] >= start_date) & (df['peak_date'] <= end_date)
].copy()

if len(df_period) == 0:
    st.info("No drawdown events in this analysis period.")
    st.stop()

# Map conviction to weight labels for display
df_period['weight'] = df_period['conviction'].map(WEIGHT_LABELS)

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
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, width='stretch', hide_index=True)

# ============================================================================
# Section 2: Drawdown Depth — Violin + Strip
# ============================================================================
st.header("Drawdown Depth Distribution by Weight")

fig_violin = go.Figure()

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    # Violin (distribution shape)
    fig_violin.add_trace(go.Violin(
        y=group['depth_pct'],
        name=f"{label} (n={len(group)})",
        legendgroup=label,
        line_color=color,
        fillcolor=color,
        opacity=0.35,
        meanline_visible=True,
        box_visible=True,
        points=False,
        side='both',
        scalemode='width',
        width=0.8,
    ))

    # Strip (individual data points with jitter)
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(group))
    x_pos = list(WEIGHT_LABELS.values()).index(label)
    fig_violin.add_trace(go.Scatter(
        x=x_pos + jitter,
        y=group['depth_pct'],
        mode='markers',
        name=label,
        legendgroup=label,
        showlegend=False,
        marker=dict(
            color=color, size=4, opacity=0.5,
            line=dict(width=0),
        ),
        hovertext=group['ticker'],
        hovertemplate='%{hovertext}<br>Depth: %{y:.1f}%<extra></extra>',
    ))

fig_violin.update_layout(
    **LAYOUT_COMMON,
    height=520,
    yaxis=dict(title='Drawdown Depth (%)', **AXIS_STYLE),
    xaxis=dict(
        title='Weight', **AXIS_STYLE,
        tickmode='array',
        tickvals=list(range(len(WEIGHT_ORDER))),
        ticktext=[f"{l}" for l in WEIGHT_ORDER],
    ),
    showlegend=True,
    violingap=0.3,
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
    # Not recovered → treated as censored (not counted as recovered)
    not_recovered_count = recovery_days.isna().sum()

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
# Section 4: Recovery Days Distribution by Weight
# ============================================================================
st.header("Recovery Speed by Weight")

recovered_df = df_period[df_period['recovered'] & df_period['days_to_recover'].notna()].copy()

if len(recovered_df) > 0:
    fig_rec = go.Figure()

    for conv, label in WEIGHT_LABELS.items():
        group = recovered_df[recovered_df['conviction'] == conv]
        if len(group) == 0:
            continue
        color = COLORS[label]
        fig_rec.add_trace(go.Box(
            y=group['days_to_recover'],
            name=f"{label} (n={len(group)})",
            marker_color=color,
            line_color=color,
            boxpoints='outliers',
            marker=dict(outliercolor=color, size=3, opacity=0.5),
            boxmean=True,
        ))

    fig_rec.update_layout(
        **LAYOUT_COMMON,
        height=480,
        yaxis=dict(title='Days to Recovery', **AXIS_STYLE),
        xaxis=dict(title='Weight', **AXIS_STYLE),
        showlegend=False,
    )
    st.plotly_chart(fig_rec, width='stretch')
else:
    st.info("No recovered drawdowns in this period.")

# ============================================================================
# Section 5: Weight vs Depth Scatter + Trend Lines
# ============================================================================
st.header("Weight at Peak vs Drawdown Depth")

fig_scatter = go.Figure()

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    fig_scatter.add_trace(go.Scatter(
        x=group['weight_at_peak'],
        y=group['depth_pct'],
        mode='markers',
        name=f"{label} (n={len(group)})",
        marker=dict(color=color, size=6, opacity=0.6,
                    line=dict(width=0.5, color='#272727')),
        hovertext=group['ticker'],
        hovertemplate='%{hovertext}<br>Weight: %{x:.1f}%<br>Depth: %{y:.1f}%<extra></extra>',
    ))

    # OLS trend line per group
    if len(group) >= 5:
        x = group['weight_at_peak'].values
        y = group['depth_pct'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
            y_line = np.polyval(coeffs, x_line)
            fig_scatter.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                name=f"{label} trend",
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
            ))

fig_scatter.update_layout(
    **LAYOUT_COMMON,
    height=520,
    xaxis=dict(title='Weight at Peak (%)', **AXIS_STYLE),
    yaxis=dict(title='Drawdown Depth (%)', **AXIS_STYLE),
)
st.plotly_chart(fig_scatter, width='stretch')

# ============================================================================
# Section 6: Detailed Table
# ============================================================================
with st.expander("Detailed Data Table", expanded=False):
    display_df = df_period[[
        'ticker', 'weight', 'weight_at_peak', 'peak_date', 'trough_date',
        'peak_price', 'trough_price', 'depth_pct', 'duration_days',
        'recovered', 'recovery_date', 'days_to_recover'
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
