"""Fetch all ARK ETF holdings stock prices from Yahoo Finance"""
import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import ANALYSIS_PERIODS, OUTPUT_DIR, ARK_ETFS
from data_loader import load_ark_holdings, get_ark_files_hash

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / 'ark_holdings_prices.parquet'
FAILED_FILE = OUTPUT_DIR / 'ark_holdings_prices_failed.txt'

# Check cache
if OUTPUT_FILE.exists():
    print(f"Cache exists: {OUTPUT_FILE}")
    print("Delete the file to re-download.")
    sys.exit(0)

# Collect unique tickers from all 6 ETFs
files_hash = get_ark_files_hash()
all_tickers_raw = set()

for etf in ARK_ETFS:
    holdings = load_ark_holdings(files_hash, etf)
    if len(holdings) == 0:
        continue

    for t in holdings['Ticker'].unique():
        clean = t.split()[0] if pd.notna(t) and ' ' in t else t
        if pd.notna(clean):
            all_tickers_raw.add(clean)

# Sort for reproducibility
tickers = sorted(all_tickers_raw)

# Date range covering all analysis periods
all_starts = [p["start"] for p in ANALYSIS_PERIODS.values()]
all_ends = [p["end"] for p in ANALYSIS_PERIODS.values()]
start_date = min(all_starts)
end_date = max(all_ends)
end_date_inclusive = end_date + pd.Timedelta(days=1)

print(f"Fetching ARK holdings prices from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Total unique tickers: {len(tickers)}")
print()

# Fetch all prices
all_prices = {}
failed_tickers = []

for ticker in tickers:
    print(f"Fetching {ticker}...", end=" ")
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date_inclusive, interval="1d")
        if not df.empty:
            df = df.reset_index()
            df = df[['Date', 'Close']]
            df['Date'] = df['Date'].dt.tz_localize(None)  # Remove timezone
            df['Ticker'] = ticker
            all_prices[ticker] = df
            print(f"OK ({len(df)} rows)")
        else:
            print("NO DATA")
            failed_tickers.append(ticker)
    except Exception as e:
        print(f"ERROR: {e}")
        failed_tickers.append(ticker)

print()

if failed_tickers:
    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")
    print()

# Save failed tickers
with open(FAILED_FILE, 'w') as f:
    for t in failed_tickers:
        f.write(f"{t}\n")
print(f"Saved {FAILED_FILE}")

# Combine and save
if all_prices:
    combined = pd.concat(all_prices.values(), ignore_index=True)

    # Pivot to wide format (Date as index, tickers as columns)
    prices_wide = combined.pivot(index='Date', columns='Ticker', values='Close')
    prices_wide = prices_wide.reset_index()

    prices_wide.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} ({len(prices_wide)} rows, {len(all_prices)} tickers)")
else:
    print("No prices fetched!")

print()
print("Done!")
