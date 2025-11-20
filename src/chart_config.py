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
