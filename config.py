"""Configuration settings"""
from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / 'input'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
PROCESSED_DIR = DATA_DIR / 'processed'

# Analysis Periods
ANALYSIS_PERIODS = {
    "2024-2026": {
        "start": pd.to_datetime('2024-01-02'),
        "end": pd.to_datetime('2026-06-26'),
        "label": "2024-2026",
        "has_r3000_data": True
    },
    "2021-2023": {
        "start": pd.to_datetime('2021-01-01'),
        "end": pd.to_datetime('2023-01-01'),
        "label": "2021-2023",
        "has_r3000_data": False
    }
}
DEFAULT_PERIOD = "2024-2026"

# Backward compatible defaults (used when session state not available)
START_DATE = ANALYSIS_PERIODS[DEFAULT_PERIOD]["start"]
END_DATE = ANALYSIS_PERIODS[DEFAULT_PERIOD]["end"]

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
