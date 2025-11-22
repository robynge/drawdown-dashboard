"""Fetch IWV (iShares Russell 3000 ETF) price data from Yahoo Finance"""
import yfinance as yf
from pathlib import Path
import pandas as pd
from config import START_DATE, END_DATE, OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo Finance's end parameter is exclusive, so add 1 day to include END_DATE
end_date_inclusive = END_DATE + pd.Timedelta(days=1)

print("Fetching IWV...")
ticker = yf.Ticker("IWV")
df = ticker.history(start=START_DATE, end=end_date_inclusive, interval="1d")

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
