"""Configuration settings"""
from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / 'input'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
PROCESSED_DIR = DATA_DIR / 'processed'

# Analysis settings
START_DATE = pd.to_datetime('2024-01-02')
END_DATE = pd.to_datetime('2026-02-06')

# ETFs
ARK_ETFS = ['ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'ARKX']

# Russell Index Reconstitution Dates (annual in June, semi-annual starting 2026)
# Reference: https://www.lseg.com/en/ftse-russell/russell-reconstitution
RUSSELL_RECONSTITUTION_DATES = [
    pd.to_datetime('2024-06-28'),  # 2024 annual reconstitution
    pd.to_datetime('2025-06-27'),  # 2025 annual reconstitution (last annual-only)
    # 2026: Semi-annual schedule begins (June + November)
    pd.to_datetime('2026-06-26'),  # 2026 June reconstitution (4th Friday)
    pd.to_datetime('2026-11-13'),  # 2026 November reconstitution (2nd Friday)
]

# Note: Manual caching removed - all caching now handled by Streamlit's @st.cache_data
