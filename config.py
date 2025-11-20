"""Configuration settings"""
from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / 'input'
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
CACHE_DIR = DATA_DIR / 'cache'
PROCESSED_DIR = DATA_DIR / 'processed'

# Analysis settings
START_DATE = pd.to_datetime('2024-04-01')
END_DATE = pd.to_datetime('2025-11-18')

# ETFs
ARK_ETFS = ['ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'ARKX']

# Cache settings
CACHE_ENABLED = True
