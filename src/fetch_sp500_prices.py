"""Fetch S&P 500 Top 50 stock prices from Yahoo Finance"""
import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ANALYSIS_PERIODS, OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# S&P 500 Top 50 tickers by market cap (as of Jan 2026, verified Apr 2026)
SP500_TOP50 = [
    # Rank 1-10
    'NVDA', 'AAPL', 'GOOG', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA', 'BRK-B',
    # Rank 11-20
    'LLY', 'WMT', 'JPM', 'V', 'ORCL', 'XOM', 'MA', 'JNJ', 'BAC', 'ABBV',
    # Rank 21-30
    'NFLX', 'COST', 'AMD', 'MU', 'HD', 'GE', 'PG', 'CVX', 'WFC', 'UNH',
    # Rank 31-40
    'CSCO', 'KO', 'MS', 'CAT', 'GS', 'IBM', 'MRK', 'AXP', 'RTX', 'PM',
    # Rank 41-50
    'CRM', 'LRCX', 'TMUS', 'TMO', 'C', 'MCD', 'ABT', 'AMAT', 'ISRG', 'LIN',
]

# Fetch price data covering ALL analysis periods
all_starts = [p["start"] for p in ANALYSIS_PERIODS.values()]
all_ends = [p["end"] for p in ANALYSIS_PERIODS.values()]
start_date = min(all_starts)
end_date = max(all_ends)

# Yahoo Finance's end parameter is exclusive, so add 1 day
end_date_inclusive = end_date + pd.Timedelta(days=1)

print(f"Fetching S&P 500 Top 50 prices from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print(f"Total tickers: {len(SP500_TOP50)}")
print()

# Fetch all prices
all_prices = {}
failed_tickers = []

for ticker in SP500_TOP50:
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

    # Save to CSV
    output_file = OUTPUT_DIR / 'SP500_top50_prices.csv'
    prices_wide.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Saved {output_file} ({len(prices_wide)} rows, {len(all_prices)} tickers)")

    # Also save the ticker list
    ticker_file = OUTPUT_DIR / 'SP500_top50_tickers.txt'
    with open(ticker_file, 'w') as f:
        for ticker in SP500_TOP50:
            if ticker in all_prices:
                f.write(f"{ticker}\n")
    print(f"Saved {ticker_file}")

print()
print("Done!")
