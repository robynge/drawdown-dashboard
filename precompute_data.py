"""Precompute all data for faster dashboard loading"""
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import ARK_ETFS, OUTPUT_DIR, CACHE_DIR
from data_loader import load_etf_prices
from drawdown_calculator import calculate_drawdowns

print("Precomputing all ETF data...")

# Create precomputed directory
PRECOMP_DIR = Path(__file__).parent / 'data' / 'precomputed'
PRECOMP_DIR.mkdir(parents=True, exist_ok=True)

all_data = {}

# Compute for each ETF
for etf in ARK_ETFS:
    print(f"Processing {etf}...")

    # Load prices
    prices = load_etf_prices(etf)

    if len(prices) > 0:
        # Calculate drawdowns
        dd_df = calculate_drawdowns(prices)

        if len(dd_df) > 0:
            dd_df.insert(0, 'ETF', etf)
            all_data[etf] = {
                'prices': prices,
                'drawdowns': dd_df
            }

# Save combined data
print("\nSaving precomputed data...")

# Save as pickle for fast loading
with open(PRECOMP_DIR / 'etf_data.pkl', 'wb') as f:
    pickle.dump(all_data, f)

# Also save as CSV for transparency
for etf, data in all_data.items():
    data['drawdowns'].to_csv(PRECOMP_DIR / f'{etf}_drawdowns.csv', index=False)

print(f"\n✓ Precomputed data saved to {PRECOMP_DIR}")
print(f"  - {len(all_data)} ETFs processed")
print(f"  - Total size: {sum((PRECOMP_DIR / f).stat().st_size for f in PRECOMP_DIR.glob('*')) / 1024:.1f} KB")
