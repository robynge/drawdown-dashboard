"""Data loading without manual caching - caching handled by Streamlit"""
import pandas as pd
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, ARK_ETFS


def load_ark_holdings(etf):
    """Load ARK ETF holdings from Parquet (fast) or Excel (fallback)"""
    parquet_path = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.parquet'
    xlsx_path = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_excel(xlsx_path)
        df['Date'] = pd.to_datetime(df['Date'])
        if 'CUSIP' in df.columns:
            df['CUSIP'] = df['CUSIP'].astype(str)

    return df


def load_r3000_holdings():
    """Load Russell 3000 holdings from Parquet (fast) or Excel (fallback)"""
    parquet_path = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.parquet'
    xlsx_path = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.xlsx'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        all_data = []
        xl = pd.ExcelFile(xlsx_path)
        for sheet in xl.sheet_names:
            df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet)
            all_data.append(df_sheet)

        df = pd.concat(all_data, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'])
        if 'CUSIP' in df.columns:
            df['CUSIP'] = df['CUSIP'].astype(str)

    return df


def load_industry_info(source='ark'):
    """Load industry mapping from 'value' sheet

    Maps tickers by their symbol only (e.g., 'AAPL' from 'AAPL US Equity')
    to handle different exchange codes (US/UW/UN/etc)
    """
    if source == 'ark':
        file_path = INPUT_DIR / 'industry_mappings' / 'ARK ETFs industry info.xlsx'
    else:  # r3000
        file_path = INPUT_DIR / 'industry_mappings' / 'R3000 industry info.xlsx'

    # Read 'value' sheet
    df = pd.read_excel(file_path, sheet_name='value')

    # Find the GICS Industry Group column
    gics_col = None
    for col in df.columns:
        if 'GICS Ind Grp' in col or 'GICS Industry Group' in col:
            gics_col = col
            break

    if gics_col is None:
        # Fallback to looking for exact column name
        if 'GICS Industry Group' in df.columns:
            gics_col = 'GICS Industry Group'
        else:
            raise ValueError(f"Cannot find GICS Industry Group column in {file_path}")

    # Find the ticker/name column (different column names for ARK vs R3000)
    ticker_col = None
    if 'Bloomberg Name' in df.columns:
        ticker_col = 'Bloomberg Name'
    elif 'Ticker' in df.columns:
        ticker_col = 'Ticker'
    else:
        raise ValueError(f"Cannot find ticker column (Bloomberg Name or Ticker) in {file_path}")

    # Extract ticker and GICS Industry Group
    df_clean = df[[ticker_col, gics_col]].copy()
    df_clean.columns = ['Bloomberg_Name', 'GICS']

    # Remove rows where GICS is NaN
    df_valid = df_clean[df_clean['GICS'].notna()].copy()

    # For R3000: Create symbol-based mapping (match by ticker symbol only)
    if source == 'r3000':
        # Extract symbol from Bloomberg Name (first part before space)
        df_valid['Symbol'] = df_valid['Bloomberg_Name'].str.split().str[0]

        # Create mapping for both symbol and full Bloomberg Name
        industry_dict = {}
        for _, row in df_valid.iterrows():
            symbol = row['Symbol']
            gics = row['GICS']
            # Map both "AAPL" and "AAPL US/UW/UN Equity" formats
            industry_dict[symbol] = gics
            industry_dict[row['Bloomberg_Name']] = gics
    else:
        # For ARK: Use full Bloomberg Name
        industry_dict = dict(zip(df_valid['Bloomberg_Name'], df_valid['GICS']))

    return industry_dict


def load_company_name(source='ark'):
    """Load company name mapping from 'value' sheet

    Maps tickers by their symbol only (e.g., 'AAPL' from 'AAPL US Equity')
    to handle different exchange codes (US/UW/UN/etc)
    """
    if source == 'ark':
        file_path = INPUT_DIR / 'companyname_mappings' / 'ARK ETFs company name.xlsx'
    else:  # r3000
        file_path = INPUT_DIR / 'companyname_mappings' / 'R3000 company name.xlsx'

    # Read 'value' sheet
    df = pd.read_excel(file_path, sheet_name='value')

    # Extract Bloomberg Name and Company Name
    df_clean = df[['Bloomberg Name', 'Company Name']].copy()
    df_clean.columns = ['Bloomberg_Name', 'Company_Name']

    # Remove rows where Company Name is NaN
    df_valid = df_clean[df_clean['Company_Name'].notna()].copy()

    # Extract symbol from Bloomberg Name (first part before space)
    df_valid['Symbol'] = df_valid['Bloomberg_Name'].str.split().str[0]

    # Create mapping for both symbol and full Bloomberg Name
    company_dict = {}
    for _, row in df_valid.iterrows():
        symbol = row['Symbol']
        company_name = row['Company_Name']
        # Map both "AAPL" and "AAPL US/UW/UN Equity" formats
        company_dict[symbol] = company_name
        company_dict[row['Bloomberg_Name']] = company_name

    return company_dict


def load_all_ark_stock_tickers():
    """Get list of all unique stocks across ARK ETFs"""
    all_tickers = set()
    for etf in ARK_ETFS:
        holdings = load_ark_holdings(etf)

        # Filter out currency tickers (vectorized)
        if 'Bloomberg Name' in holdings.columns:
            holdings = holdings[
                ~holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)
            ]

        all_tickers.update(holdings['Ticker'].unique())

    return sorted(all_tickers)


def get_stock_etf_mapping():
    """Map each stock to the ETFs it appears in"""
    stock_map = {}
    for etf in ARK_ETFS:
        holdings = load_ark_holdings(etf)

        # Filter out currency tickers (vectorized)
        if 'Bloomberg Name' in holdings.columns:
            holdings = holdings[
                ~holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)
            ]

        # Get unique tickers and process
        for ticker in holdings['Ticker'].unique():
            ticker_clean = str(ticker).split()[0] if pd.notna(ticker) else ticker
            if ticker_clean not in stock_map:
                stock_map[ticker_clean] = []
            stock_map[ticker_clean].append((etf, ticker))

    return stock_map


def load_etf_prices(etf):
    """Load ETF price data from CSV - no caching, let Streamlit handle it"""
    file_path = OUTPUT_DIR / f'{etf}_prices.csv'
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    return df


def get_r3000_ticker_list():
    """Get list of unique R3000 tickers from precomputed file (fast)"""
    cache_file = INPUT_DIR / 'russell_3000' / 'ticker_list.csv'
    parquet_file = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.parquet'
    xlsx_file = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.xlsx'

    # Use parquet file modification time if available
    source_file = parquet_file if parquet_file.exists() else xlsx_file

    # Check if cache exists and is newer than source
    if cache_file.exists():
        if cache_file.stat().st_mtime >= source_file.stat().st_mtime:
            return pd.read_csv(cache_file)['Ticker'].tolist()

    # Generate ticker list from source
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file, columns=['Ticker'])
        all_tickers = set(df['Ticker'].unique())
    else:
        xl = pd.ExcelFile(xlsx_file)
        all_tickers = set()
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet, usecols=['Ticker'])
            all_tickers.update(df['Ticker'].unique())

    # Save to cache
    ticker_df = pd.DataFrame({'Ticker': sorted(all_tickers)})
    ticker_df.to_csv(cache_file, index=False)

    return sorted(all_tickers)


def get_r3000_drawdowns_cache():
    """Get precomputed R3000 drawdowns from cache file (fast)"""
    cache_file = INPUT_DIR / 'russell_3000' / 'drawdowns_cache.csv'

    if cache_file.exists():
        return pd.read_csv(cache_file)

    return None


def save_r3000_drawdowns_cache(df):
    """Save R3000 drawdowns to cache file"""
    cache_file = INPUT_DIR / 'russell_3000' / 'drawdowns_cache.csv'
    df.to_csv(cache_file, index=False)


def get_ark_ticker_list(etf):
    """Get list of unique ARK ETF tickers from precomputed file (fast)"""
    cache_file = INPUT_DIR / 'ark_etfs' / f'{etf}_ticker_list.csv'
    parquet_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.parquet'
    xlsx_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'

    # Use parquet file modification time if available
    source_file = parquet_file if parquet_file.exists() else xlsx_file

    # Check if cache exists and is newer than source
    if cache_file.exists():
        if cache_file.stat().st_mtime >= source_file.stat().st_mtime:
            return pd.read_csv(cache_file)['Ticker'].tolist()

    # Generate ticker list from source
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file, columns=['Ticker', 'Bloomberg Name'])
    else:
        df = pd.read_excel(xlsx_file, usecols=['Ticker', 'Bloomberg Name'])

    # Filter out currency tickers
    if 'Bloomberg Name' in df.columns:
        df = df[~df['Bloomberg Name'].str.contains('curncy', case=False, na=False)]

    all_tickers = sorted(df['Ticker'].unique().tolist())

    # Save to cache
    ticker_df = pd.DataFrame({'Ticker': all_tickers})
    ticker_df.to_csv(cache_file, index=False)

    return all_tickers
