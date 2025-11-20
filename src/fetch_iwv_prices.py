"""Fetch IWV (iShares Russell 3000 ETF) price data from Yahoo Finance"""
import yfinance as yf
from pathlib import Path
from config import START_DATE, END_DATE, OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Fetching IWV...")
ticker = yf.Ticker("IWV")
df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")

if not df.empty:
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Close'] = df['Close'].round(2)
    df.to_csv(OUTPUT_DIR / 'IWV_prices.csv', index=False, float_format='%.2f')
    print(f"  Saved IWV_prices.csv with {len(df)} rows")
else:
    print("  No data for IWV")

print("Done!")
