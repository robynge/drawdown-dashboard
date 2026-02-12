"""Convert Excel files to Parquet format for faster loading"""
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(__file__).parent / 'input'

def convert_ark_etfs():
    """Convert ARK ETF Excel files to Parquet"""
    ark_dir = INPUT_DIR / 'ark_etfs'

    for xlsx_file in ark_dir.glob('*_Transformed_Data.xlsx'):
        print(f"Converting {xlsx_file.name}...")

        df = pd.read_excel(xlsx_file)
        df['Date'] = pd.to_datetime(df['Date'])

        # Fix CUSIP column type
        if 'CUSIP' in df.columns:
            df['CUSIP'] = df['CUSIP'].astype(str)

        # Save as Parquet
        parquet_file = xlsx_file.with_suffix('.parquet')
        df.to_parquet(parquet_file, index=False)

        # Show size comparison
        xlsx_size = xlsx_file.stat().st_size / (1024 * 1024)
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)
        print(f"  {xlsx_size:.1f} MB -> {parquet_size:.1f} MB ({parquet_size/xlsx_size*100:.0f}%)")

def convert_russell_3000():
    """Convert Russell 3000 Excel file to Parquet"""
    r3000_dir = INPUT_DIR / 'russell_3000'
    xlsx_file = r3000_dir / 'IWV_Transformed_Data.xlsx'

    if not xlsx_file.exists():
        print(f"File not found: {xlsx_file}")
        return

    print(f"Converting {xlsx_file.name}...")

    # Read all sheets and combine
    all_data = []
    xl = pd.ExcelFile(xlsx_file)
    for sheet in xl.sheet_names:
        print(f"  Reading sheet: {sheet}")
        df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet)
        all_data.append(df_sheet)

    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Fix CUSIP column type
    if 'CUSIP' in df.columns:
        df['CUSIP'] = df['CUSIP'].astype(str)

    # Save as Parquet
    parquet_file = xlsx_file.with_suffix('.parquet')
    df.to_parquet(parquet_file, index=False)

    # Show size comparison
    xlsx_size = xlsx_file.stat().st_size / (1024 * 1024)
    parquet_size = parquet_file.stat().st_size / (1024 * 1024)
    print(f"  {xlsx_size:.1f} MB -> {parquet_size:.1f} MB ({parquet_size/xlsx_size*100:.0f}%)")

def precompute_peer_group_cache():
    """Precompute peer group cache after data conversion"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import get_r3000_files_hash
    from peer_group import calculate_peer_group_prices_mv, calculate_peer_group_prices_weighted, calculate_iwv_total_market_value

    files_hash = get_r3000_files_hash()

    print("Precomputing IWV Total Market Value cache...")
    calculate_iwv_total_market_value(files_hash)

    print("Precomputing Market Value cache...")
    calculate_peer_group_prices_mv(files_hash)

    print("Precomputing Weighted Price cache...")
    calculate_peer_group_prices_weighted(files_hash)


def precompute_r3000_drawdowns():
    """Precompute R3000 drawdowns for all stocks (takes several minutes)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import load_r3000_holdings, load_industry_info, get_r3000_files_hash, save_r3000_drawdowns_cache
    from drawdown_calculator import calculate_drawdowns_with_filter

    files_hash = get_r3000_files_hash()
    holdings = load_r3000_holdings(files_hash)

    if len(holdings) == 0:
        print("  No R3000 holdings data found")
        return

    # Get unique tickers
    all_tickers = holdings['Ticker'].unique()
    industry_dict = load_industry_info(source='r3000')

    print(f"  Processing {len(all_tickers)} tickers...")

    results = []
    for i, ticker in enumerate(all_tickers):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{len(all_tickers)}")

        stock_data = holdings[holdings['Ticker'] == ticker].copy()

        if len(stock_data) < 10:
            continue

        if 'Price' not in stock_data.columns:
            continue

        price_df = stock_data[['Date', 'Price']].copy()
        price_df = price_df.rename(columns={'Price': 'Close'})
        price_df = price_df.dropna(subset=['Close'])

        if len(price_df) < 10:
            continue

        dd_df = calculate_drawdowns_with_filter(price_df, min_depth_pct=10, min_duration_days=7)

        if len(dd_df) > 0:
            max_dd = dd_df['depth_pct'].min()
            ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
            gics = industry_dict.get(ticker, industry_dict.get(ticker_clean, 'Unknown'))
            results.append({
                'ticker': ticker_clean,
                'max_drawdown': max_dd,
                'num_drawdowns': len(dd_df),
                'gics_industry_group': gics
            })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        save_r3000_drawdowns_cache(result_df)
        print(f"  Saved {len(result_df)} ticker drawdowns to cache")


def precompute_ark_drawdowns():
    """Precompute ARK stock drawdowns for all stocks (takes several minutes)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS, START_DATE, END_DATE
    from data_loader import load_ark_holdings, get_ark_files_hash
    from drawdown_calculator import calculate_drawdowns
    from recovery_probability import save_ark_drawdowns_cache

    all_drawdowns = []

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        try:
            holdings = load_ark_holdings(get_ark_files_hash(), etf)
            holdings = holdings[(holdings['Date'] >= START_DATE) & (holdings['Date'] <= END_DATE)]

            tickers = holdings['Ticker'].unique()
            for ticker in tickers:
                ticker_holdings = holdings[holdings['Ticker'] == ticker]
                if 'Bloomberg Name' in ticker_holdings.columns:
                    bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                    if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                        continue

                stock_data = holdings[holdings['Ticker'] == ticker].copy()
                if len(stock_data) < 30:
                    continue

                if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                    price_col = 'YFinance Close Price'
                else:
                    price_col = 'Stock_Price'

                price_df = stock_data[['Date', price_col]].copy()
                price_df.columns = ['Date', 'Close']
                price_df = price_df.dropna()

                if len(price_df) < 30:
                    continue

                # Use actual data date range for drawdown calculation
                data_start = price_df['Date'].min()
                data_end = price_df['Date'].max()
                dd_data = calculate_drawdowns(price_df, start_date=data_start, end_date=data_end)
                if len(dd_data) == 0 or 'rank' not in dd_data.columns:
                    continue
                historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

                for _, dd in historical_dd.iterrows():
                    peak_date = dd['peak_date']
                    trough_date = dd['trough_date']
                    peak_price = dd['peak_price']
                    trough_price = dd['trough_price']

                    future_prices = price_df[price_df['Date'] > trough_date]
                    recovery_dates = future_prices[future_prices['Close'] >= peak_price]

                    if len(recovery_dates) > 0:
                        recovery_date = recovery_dates.iloc[0]['Date']
                        recovered = True
                        days_to_recover = (recovery_date - trough_date).days
                    else:
                        recovery_date = None
                        recovered = False
                        days_to_recover = None

                    all_drawdowns.append({
                        'ticker': ticker,
                        'etf': etf,
                        'peak_date': peak_date,
                        'trough_date': trough_date,
                        'peak_price': peak_price,
                        'trough_price': trough_price,
                        'depth_pct': dd['depth_pct'],
                        'recovery_date': recovery_date,
                        'recovered': recovered,
                        'days_to_recover': days_to_recover
                    })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    df = pd.DataFrame(all_drawdowns)
    if len(df) > 0:
        save_ark_drawdowns_cache(df)
        print(f"  Saved {len(df)} drawdown records to cache")


if __name__ == '__main__':
    print("=" * 50)
    print("Converting Excel files to Parquet & Precomputing Caches")
    print("=" * 50)
    print()

    print("Step 1/5: ARK ETFs")
    convert_ark_etfs()
    print()

    print("Step 2/5: Russell 3000")
    convert_russell_3000()
    print()

    print("Step 3/5: Peer Group Cache")
    precompute_peer_group_cache()
    print()

    print("Step 4/5: R3000 Drawdowns (this may take several minutes)")
    precompute_r3000_drawdowns()
    print()

    print("Step 5/5: ARK Stock Drawdowns (this may take several minutes)")
    precompute_ark_drawdowns()
    print()

    print("=" * 50)
    print("Done! All caches precomputed.")
    print("=" * 50)
