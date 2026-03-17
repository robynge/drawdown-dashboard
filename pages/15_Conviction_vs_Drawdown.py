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

# Section 2: Summary Statistics
st.header("Summary Statistics")

# Build summary table
summary_rows = []
for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        summary_rows.append({
            'Weight': label,
            'Holdings': 0,
            'Drawdown Events': 0,
            'Avg Depth (%)': None,
            'Median Depth (%)': None,
            'Recovery Rate (%)': None,
            'Avg Recovery Days': None,
            'Median Recovery Days': None,
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

# Section 3: Drawdown Depth Box Plot
st.header("Drawdown Depth by Weight")

fig_box = go.Figure()
colors = {'≥5%': '#EF553B', '1%-5%': '#FFA15A', '<1%': '#636EFA'}

for conv, label in WEIGHT_LABELS.items():
    group = df_period[df_period['conviction'] == conv]
    if len(group) == 0:
        continue
    fig_box.add_trace(go.Box(
        y=group['depth_pct'],
        name=f"{label} ({len(group)})",
        marker_color=colors[label],
        boxpoints='outliers',
    ))

fig_box.update_layout(
    yaxis_title="Drawdown Depth (%)",
    xaxis_title="Weight",
    height=500,
    showlegend=False,
)
st.plotly_chart(fig_box, width='stretch')

# Section 4: Recovery Rate by Depth Bucket
st.header("Recovery Rate by Depth Bucket")

df_period['depth_bucket'] = pd.cut(
    df_period['depth_pct'].abs(),
    bins=[0, 10, 20, 30, 50, 100],
    labels=['0-10%', '10-20%', '20-30%', '30-50%', '50%+'],
    right=True
)

# Group by bucket and weight
recovery_data = []
for bucket in ['0-10%', '10-20%', '20-30%', '30-50%', '50%+']:
    for conv, label in WEIGHT_LABELS.items():
        group = df_period[
            (df_period['depth_bucket'] == bucket) &
            (df_period['conviction'] == conv)
        ]
        if len(group) == 0:
            continue
        recovery_data.append({
            'Depth Bucket': bucket,
            'Weight': label,
            'Recovery Rate (%)': round(group['recovered'].mean() * 100, 1),
            'Count': len(group),
        })

if recovery_data:
    recovery_df = pd.DataFrame(recovery_data)
    fig_bar = px.bar(
        recovery_df,
        x='Depth Bucket',
        y='Recovery Rate (%)',
        color='Weight',
        barmode='group',
        color_discrete_map=colors,
        category_orders={'Weight': WEIGHT_ORDER},
        text='Count',
        height=500,
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(yaxis_range=[0, 110])
    st.plotly_chart(fig_bar, width='stretch')
else:
    st.info("Not enough data for recovery rate comparison.")

# Section 5: Weight vs Depth Scatter
st.header("Weight at Peak vs Drawdown Depth")

fig_scatter = px.scatter(
    df_period,
    x='weight_at_peak',
    y='depth_pct',
    color='weight',
    color_discrete_map=colors,
    category_orders={'weight': WEIGHT_ORDER},
    hover_data=['ticker', 'peak_date', 'duration_days', 'recovered'],
    labels={
        'weight_at_peak': 'Weight at Peak (%)',
        'depth_pct': 'Drawdown Depth (%)',
        'weight': 'Weight',
    },
    height=500,
)
fig_scatter.update_layout(
    xaxis_title="Weight at Peak (%)",
    yaxis_title="Drawdown Depth (%)",
)
st.plotly_chart(fig_scatter, width='stretch')

# Section 6: Detailed Table
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

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{selected_etf}_conviction_drawdowns.csv",
        mime="text/csv",
    )
