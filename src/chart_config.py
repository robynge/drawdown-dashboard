"""Shared chart configuration for high-quality exports"""
from config import RUSSELL_RECONSTITUTION_DATES, START_DATE, END_DATE

# High-quality PNG export - Streamlit + Plotly global default
CHART_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'height': 1080,
        'width': 1920,
        'scale': 3  # 3x resolution for crisp images
    },
    'displayModeBar': True,
    'displaylogo': False
}

# Color palette for drawdown shaded regions (top 10)
DD_COLORS = [
    'rgba(255, 99, 71, 0.3)',   # Tomato
    'rgba(255, 165, 0, 0.3)',   # Orange
    'rgba(255, 215, 0, 0.3)',   # Gold
    'rgba(144, 238, 144, 0.3)', # Light Green
    'rgba(173, 216, 230, 0.3)', # Light Blue
    'rgba(221, 160, 221, 0.3)', # Plum
    'rgba(255, 192, 203, 0.3)', # Pink
    'rgba(176, 224, 230, 0.3)', # Powder Blue
    'rgba(240, 230, 140, 0.3)', # Khaki
    'rgba(255, 228, 181, 0.3)'  # Moccasin
]


def add_reconstitution_vlines(fig, price_line_name="Price"):
    """Add vertical red lines for Russell Index reconstitution dates with legend

    Args:
        fig: Plotly figure object
        price_line_name: Name for the main price line in legend

    Returns:
        The modified figure object
    """
    import plotly.graph_objects as go

    # Update existing price trace to show in legend
    if len(fig.data) > 0:
        fig.data[0].name = price_line_name
        fig.data[0].showlegend = True

    # Check if any reconstitution dates are in range
    recon_dates_in_range = [d for d in RUSSELL_RECONSTITUTION_DATES if START_DATE <= d <= END_DATE]

    if recon_dates_in_range:
        # Add a dummy trace for legend entry
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='red', width=2, dash='dot'),
            name='Russell Reconstitution',
            showlegend=True
        ))

        # Add vertical lines
        for recon_date in recon_dates_in_range:
            fig.add_vline(
                x=recon_date,
                line=dict(color='red', width=2, dash='dot'),
                layer='above'
            )

    # Show legend with white background (not transparent, so it's on top)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='white',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1
        )
    )

    return fig
