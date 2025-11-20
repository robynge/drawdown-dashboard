"""Fetch ETF price data from Yahoo Finance"""
import yfinance as yf
from pathlib import Path
from config import ARK_ETFS, START_DATE, END_DATE, OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fetch and save price data for ARK ETFs
for etf in ARK_ETFS:
    print(f"Fetching {etf}...")
    ticker = yf.Ticker(etf)
    df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")
    if not df.empty:
        df = df.reset_index()
        df = df[['Date', 'Close']]
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['Close'] = df['Close'].round(2)
        df.to_csv(OUTPUT_DIR / f'{etf}_prices.csv', index=False, float_format='%.2f')
        print(f"  Saved {etf}_prices.csv")
    else:
        print(f"  No data for {etf}")

print("Done!")
