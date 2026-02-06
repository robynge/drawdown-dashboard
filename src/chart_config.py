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


def add_reconstitution_lines(fig, y_min=None, y_max=None):
    """Add vertical red lines for Russell Index reconstitution dates

    Args:
        fig: Plotly figure object
        y_min: Optional minimum y value for the line (defaults to chart min)
        y_max: Optional maximum y value for the line (defaults to chart max)

    Returns:
        The modified figure object
    """
    for recon_date in RUSSELL_RECONSTITUTION_DATES:
        # Only add lines within analysis period
        if START_DATE <= recon_date <= END_DATE:
            # Add vertical line
            fig.add_vline(
                x=recon_date,
                line=dict(color='red', width=1.5, dash='dot'),
                layer='above'
            )

            # Add annotation at the top
            fig.add_annotation(
                x=recon_date,
                y=1.02,  # Position above chart
                yref='paper',
                text=f"Recon {recon_date.strftime('%Y-%m')}",
                showarrow=False,
                font=dict(size=9, color='red'),
                textangle=-90,
                xanchor='left',
                yanchor='bottom'
            )

    return fig
