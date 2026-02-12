"""Fetch ETF price data from Yahoo Finance for all analysis periods"""
import yfinance as yf
from pathlib import Path
import pandas as pd
from config import ARK_ETFS, ANALYSIS_PERIODS, OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fetch price data covering ALL analysis periods
# Find the earliest start and latest end across all periods
all_starts = [p["start"] for p in ANALYSIS_PERIODS.values()]
all_ends = [p["end"] for p in ANALYSIS_PERIODS.values()]
start_date = min(all_starts)
end_date = max(all_ends)

# Yahoo Finance's end parameter is exclusive, so add 1 day to include end_date
end_date_inclusive = end_date + pd.Timedelta(days=1)

print(f"Fetching price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Covering periods: {', '.join(ANALYSIS_PERIODS.keys())}")
print()

# Fetch and save price data for ARK ETFs
for etf in ARK_ETFS:
    print(f"Fetching {etf}...")
    ticker = yf.Ticker(etf)
    df = ticker.history(start=start_date, end=end_date_inclusive, interval="1d")
    if not df.empty:
        df = df.reset_index()
        df = df[['Date', 'Close']]
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['Close'] = df['Close'].round(2)
        df.to_csv(OUTPUT_DIR / f'{etf}_prices.csv', index=False, float_format='%.2f')
        print(f"  Saved {etf}_prices.csv ({len(df)} rows)")
    else:
        print(f"  No data for {etf}")

print()
print("Done!")
