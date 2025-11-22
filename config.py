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
END_DATE = pd.to_datetime('2025-11-21')

# ETFs
ARK_ETFS = ['ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'ARKX']

# Note: Manual caching removed - all caching now handled by Streamlit's @st.cache_data
