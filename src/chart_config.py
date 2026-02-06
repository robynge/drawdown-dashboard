"""Shared chart configuration for high-quality exports"""

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
