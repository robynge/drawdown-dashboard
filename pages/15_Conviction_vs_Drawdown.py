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
COLORS = {'≥5%': '#D62728', '1%-5%': '#1F77B4', '<1%': '#F5A623'}

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

    fig_violin.add_trace(go.Violin(
        y=group['depth_pct'],
        name=f"{label} (n={len(group)})",
        line_color=color,
        fillcolor=color,
        opacity=0.4,
        meanline_visible=True,
        box_visible=True,
        points='all',
        jitter=0.35,
        pointpos=0,
        marker=dict(color=color, size=3, opacity=0.45),
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
# Section 5b: Drawdown PnL vs Recovery PnL (side by side)
# ============================================================================
st.header("Adjusted PnL: Drawdown vs Recovery")

col_dd_pnl, col_rec_pnl = st.columns(2)

# --- Left: Drawdown PnL (peak → trough) ---
fig_dd_pnl = go.Figure()

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    hover_texts = [
        f"{row['ticker']} (DD #{row['rank']})" for _, row in group.iterrows()
    ]

    fig_dd_pnl.add_trace(go.Scatter(
        x=group['weight_at_peak'],
        y=group['adj_pnl'],
        mode='markers',
        name=f"{label} (n={len(group)})",
        marker=dict(color=color, size=6, opacity=0.6,
                    line=dict(width=0.5, color='#272727')),
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<br>Weight: %{x:.4f}%<br>Adj PnL: $%{y:,.0f}<extra></extra>',
    ))

    if len(group) >= 5:
        x = group['weight_at_peak'].values
        y = group['adj_pnl'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
            y_line = np.polyval(coeffs, x_line)
            fig_dd_pnl.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
            ))

fig_dd_pnl.update_layout(
    **LAYOUT_COMMON,
    height=520,
    xaxis=dict(title='Weight (%)', **AXIS_STYLE),
    yaxis=dict(title='Drawdown Adj PnL ($)', **AXIS_STYLE),
)

with col_dd_pnl:
    st.subheader("Drawdown (Peak → Trough)")
    st.plotly_chart(fig_dd_pnl, width='stretch')

# --- Right: Recovery PnL (trough → recovery_date) ---
fig_rec_pnl = go.Figure()

rec_df = df_period[df_period['recovered'] & df_period['recovery_adj_pnl'].notna()].copy()

for conv, label in WEIGHT_LABELS.items():
    group = rec_df[rec_df['conviction'] == conv]
    if len(group) == 0:
        continue
    color = COLORS[label]

    hover_texts = [
        f"{row['ticker']} (DD #{row['rank']})" for _, row in group.iterrows()
    ]

    fig_rec_pnl.add_trace(go.Scatter(
        x=group['weight_at_peak'],
        y=group['recovery_adj_pnl'],
        mode='markers',
        name=f"{label} (n={len(group)})",
        marker=dict(color=color, size=6, opacity=0.6,
                    line=dict(width=0.5, color='#272727')),
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<br>Weight: %{x:.4f}%<br>Recovery PnL: $%{y:,.0f}<extra></extra>',
    ))

    if len(group) >= 5:
        x = group['weight_at_peak'].values
        y = group['recovery_adj_pnl'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 5:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(x[mask].min(), x[mask].max(), 50)
            y_line = np.polyval(coeffs, x_line)
            fig_rec_pnl.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
            ))

fig_rec_pnl.update_layout(
    **LAYOUT_COMMON,
    height=520,
    xaxis=dict(title='Weight (%)', **AXIS_STYLE),
    yaxis=dict(title='Recovery Adj PnL ($)', **AXIS_STYLE),
)

with col_rec_pnl:
    st.subheader("Recovery (Trough → Recovery)")
    st.plotly_chart(fig_rec_pnl, width='stretch')

# ============================================================================
# Section 6: Duration vs Depth Scatter (color=weight)
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
# Section 7: Detailed Table
# ============================================================================
with st.expander("Detailed Data Table", expanded=False):
    display_df = df_period[[
        'ticker', 'rank', 'weight', 'weight_at_peak', 'peak_date', 'trough_date',
        'peak_price', 'trough_price', 'depth_pct', 'adj_pnl', 'recovery_adj_pnl',
        'duration_days', 'recovered', 'recovery_date', 'days_to_recover'
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
