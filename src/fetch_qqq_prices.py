"""Fetch QQQ (Nasdaq-100) stock prices from Yahoo Finance

Reads holdings from input/qqq/QQQ holdings.xlsx and fetches prices
for top 100 stocks by market value. Saves both top50 and top100 price files.
"""
import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANALYSIS_PERIODS, OUTPUT_DIR, INPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_qqq_tickers(max_n=100):
    """Load QQQ tickers from holdings Excel file, sorted by market value"""
    holdings_file = INPUT_DIR / 'qqq' / 'QQQ holdings.xlsx'
    if not holdings_file.exists():
        raise FileNotFoundError(f"QQQ holdings file not found: {holdings_file}")

    df = pd.read_excel(holdings_file)
    # Skip header row (row 0 contains column descriptions)
    df = df.iloc[1:].copy()
    df.columns = ['Rank', 'Ticker_Raw', 'Company', 'MarketValue']

    # Filter out non-stock entries (Index, Curncy)
    df = df[df['Ticker_Raw'].str.contains('US Equity', na=False)]

    # Extract clean ticker symbols
    df['Ticker'] = df['Ticker_Raw'].str.replace(' US Equity', '', regex=False).str.strip()

    # Take top N
    tickers = df['Ticker'].head(max_n).tolist()
    return tickers


# Load tickers from holdings file
QQQ_TICKERS = load_qqq_tickers(max_n=100)

# Fetch price data covering ALL analysis periods
all_starts = [p["start"] for p in ANALYSIS_PERIODS.values()]
all_ends = [p["end"] for p in ANALYSIS_PERIODS.values()]
start_date = min(all_starts)
end_date = max(all_ends)

# Yahoo Finance's end parameter is exclusive, so add 1 day
end_date_inclusive = end_date + pd.Timedelta(days=1)

print(f"Fetching QQQ Top {len(QQQ_TICKERS)} prices from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Total tickers: {len(QQQ_TICKERS)}")
print()

# Fetch all prices
all_prices = {}
failed_tickers = []

for ticker in QQQ_TICKERS:
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
    print(f"Failed tickers: {failed_tickers}")
    print()

# Combine all prices into a single DataFrame
if all_prices:
    combined = pd.concat(all_prices.values(), ignore_index=True)

    # Pivot to wide format (Date as index, tickers as columns)
    prices_wide = combined.pivot(index='Date', columns='Ticker', values='Close')
    prices_wide = prices_wide.reset_index()
    prices_wide['Date'] = prices_wide['Date'].dt.strftime('%Y-%m-%d')

    # Determine which tickers were successfully fetched, in original order
    fetched_tickers = [t for t in QQQ_TICKERS if t in all_prices]

    # Save top 50 (backward compatible)
    top50_tickers = fetched_tickers[:50]
    top50_wide = prices_wide[['Date'] + [t for t in top50_tickers if t in prices_wide.columns]]
    output_file_50 = OUTPUT_DIR / 'QQQ_top50_prices.csv'
    top50_wide.to_csv(output_file_50, index=False, float_format='%.2f')
    print(f"Saved {output_file_50} ({len(top50_wide)} rows, {len(top50_tickers)} tickers)")

    # Save top 100
    top100_tickers = fetched_tickers[:100]
    top100_wide = prices_wide[['Date'] + [t for t in top100_tickers if t in prices_wide.columns]]
    output_file_100 = OUTPUT_DIR / 'QQQ_top100_prices.csv'
    top100_wide.to_csv(output_file_100, index=False, float_format='%.2f')
    print(f"Saved {output_file_100} ({len(top100_wide)} rows, {len(top100_tickers)} tickers)")

    # Save ticker lists
    for label, tickers in [('top50', top50_tickers), ('top100', top100_tickers)]:
        ticker_file = OUTPUT_DIR / f'QQQ_{label}_tickers.txt'
        with open(ticker_file, 'w') as f:
            for ticker in tickers:
                f.write(f"{ticker}\n")
        print(f"Saved {ticker_file}")

print()
print("Done!")
