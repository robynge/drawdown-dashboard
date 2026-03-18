"""Data loading with centralized Streamlit caching"""
import pandas as pd
import streamlit as st
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, ARK_ETFS


def get_ark_files_hash():
    """Get hash of ARK holdings files for cache invalidation"""
    mtimes = []
    for etf in ARK_ETFS:
        parquet_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.parquet'
        xlsx_file = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'
        if parquet_file.exists():
            mtimes.append(parquet_file.stat().st_mtime)
        elif xlsx_file.exists():
            mtimes.append(xlsx_file.stat().st_mtime)
    return max(mtimes) if mtimes else 0


def get_r3000_files_hash():
    """Get hash of R3000 holdings file for cache invalidation"""
    parquet_file = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.parquet'
    xlsx_file = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.xlsx'
    if parquet_file.exists():
        return parquet_file.stat().st_mtime
    elif xlsx_file.exists():
        return xlsx_file.stat().st_mtime
    return 0


MONEY_MARKET_PREFIXES = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']


def filter_non_stocks(holdings):
    """Filter out currency tickers and money market funds from holdings DataFrame

    Use this to get only actual stock holdings from ARK ETF data.
    """
    result = holdings.copy()

    # Filter out currency tickers
    if 'Bloomberg Name' in result.columns:
        result = result[
            ~result['Bloomberg Name'].str.contains('curncy', case=False, na=False)
        ]

    # Filter out money market funds (match prefix)
    if 'Ticker' in result.columns:
        ticker_symbols = result['Ticker'].str.split().str[0]
        is_money_market = ticker_symbols.apply(
            lambda x: any(x.startswith(prefix) for prefix in MONEY_MARKET_PREFIXES) if pd.notna(x) else False
        )
        result = result[~is_money_market]

    return result


def is_non_stock_ticker(ticker):
    """Check if a single ticker is a money market fund or cash instrument"""
    ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
    return any(ticker_clean.startswith(p) for p in MONEY_MARKET_PREFIXES)


@st.cache_data
def load_ark_holdings(_files_hash, etf):
    """Load ARK ETF holdings from Parquet (fast) or Excel (fallback)

    Cached centrally so all pages share the same cache.
    _files_hash: for cache invalidation when files change
    """
    parquet_path = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.parquet'
    xlsx_path = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_excel(xlsx_path)
        df['Date'] = pd.to_datetime(df['Date'])
        if 'CUSIP' in df.columns:
            df['CUSIP'] = df['CUSIP'].astype(str)

    return filter_non_stocks(df)


@st.cache_data
def load_r3000_holdings(_files_hash):
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


def get_industry_files_hash():
    """Get hash of industry mapping files for cache invalidation"""
    ark_file = INPUT_DIR / 'industry_mappings' / 'ARK ETFs industry info.xlsx'
    r3000_file = INPUT_DIR / 'industry_mappings' / 'IWV_industry group.xlsx'
    mtimes = []
    if ark_file.exists():
        mtimes.append(ark_file.stat().st_mtime)
    if r3000_file.exists():
        mtimes.append(r3000_file.stat().st_mtime)
    return max(mtimes) if mtimes else 0


def get_company_name_files_hash():
    """Get hash of company name mapping files for cache invalidation"""
    ark_file = INPUT_DIR / 'companyname_mappings' / 'ARK ETFs company name.xlsx'
    r3000_file = INPUT_DIR / 'companyname_mappings' / 'R3000 company name.xlsx'
    mtimes = []
    if ark_file.exists():
        mtimes.append(ark_file.stat().st_mtime)
    if r3000_file.exists():
        mtimes.append(r3000_file.stat().st_mtime)
    return max(mtimes) if mtimes else 0


def _load_industry_info_impl(source='ark'):
    """Internal implementation - Load industry mapping from 'value' sheet

    Maps tickers by their symbol only (e.g., 'AAPL' from 'AAPL US Equity')
    to handle different exchange codes (US/UW/UN/etc)

    Use this for non-Streamlit contexts (e.g., precompute scripts).
    """
    if source == 'ark':
        file_path = INPUT_DIR / 'industry_mappings' / 'ARK ETFs industry info.xlsx'
    else:  # r3000
        file_path = INPUT_DIR / 'industry_mappings' / 'IWV_industry group.xlsx'

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


@st.cache_data
def load_industry_info(_files_hash, source='ark'):
    """Load industry mapping from 'value' sheet (Streamlit cached version)

    _files_hash: for cache invalidation when files change
    """
    return _load_industry_info_impl(source)


@st.cache_data
def load_company_name(_files_hash, source='ark'):
    """Load company name mapping from 'value' sheet

    Maps tickers by their symbol only (e.g., 'AAPL' from 'AAPL US Equity')
    to handle different exchange codes (US/UW/UN/etc)

    _files_hash: for cache invalidation when files change
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


def load_etf_prices(etf):
    """Load ETF price data from CSV

    Not cached because CSV files are small and this function is also used
    by precompute_data.py which runs outside Streamlit.
    """
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

    # Priority 1: Return cached data if it exists (for Streamlit Cloud)
    if cache_file.exists():
        return pd.read_csv(cache_file)['Ticker'].tolist()

    # Priority 2: Generate from source if available (local development)
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file, columns=['Ticker'])
        all_tickers = set(df['Ticker'].unique())
    elif xlsx_file.exists():
        xl = pd.ExcelFile(xlsx_file)
        all_tickers = set()
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet, usecols=['Ticker'])
            all_tickers.update(df['Ticker'].unique())
    else:
        # No source available
        return []

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

    # Priority 1: Return cached data if it exists (for Streamlit Cloud)
    if cache_file.exists():
        return pd.read_csv(cache_file)['Ticker'].tolist()

    # Priority 2: Generate from source if available (local development)
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file, columns=['Ticker', 'Bloomberg Name'])
    elif xlsx_file.exists():
        df = pd.read_excel(xlsx_file, usecols=['Ticker', 'Bloomberg Name'])
    else:
        # No source available
        return []

    # Filter out currency tickers
    if 'Bloomberg Name' in df.columns:
        df = df[~df['Bloomberg Name'].str.contains('curncy', case=False, na=False)]

    all_tickers = sorted(df['Ticker'].unique().tolist())

    # Save to cache
    ticker_df = pd.DataFrame({'Ticker': all_tickers})
    ticker_df.to_csv(cache_file, index=False)

    return all_tickers


@st.cache_data
def get_stocks_for_etf(_files_hash, etf, start_date, end_date):
    """Get list of valid stocks for an ETF with current/non-current status

    Only includes stocks with at least 30 data points in the analysis period.
    Cached centrally so all pages share the same cache.
    """
    holdings = load_ark_holdings(_files_hash, etf)

    # Get latest date holdings to identify current positions
    latest_date = holdings['Date'].max()
    current_holdings = set(holdings[holdings['Date'] == latest_date]['Ticker'].unique())

    # Filter by date range first (once, not per ticker)
    holdings_filtered = holdings[
        (holdings['Date'] >= start_date) &
        (holdings['Date'] <= end_date)
    ].copy()

    # Count rows per ticker using groupby (vectorized)
    ticker_counts = holdings_filtered.groupby('Ticker').size()
    valid_tickers = ticker_counts[ticker_counts >= 30].index.tolist()

    # Build the stock list
    valid_stocks = []
    stock_ticker_map = {}

    for ticker in valid_tickers:
        ticker_simple = ticker.split()[0] if isinstance(ticker, str) else ticker
        is_current = ticker in current_holdings

        if is_current:
            display_name = ticker_simple
        else:
            display_name = f"{ticker_simple} (Non-current)"

        valid_stocks.append((ticker_simple, display_name))
        stock_ticker_map[display_name] = ticker_simple

    # Sort alphabetically
    valid_stocks.sort(key=lambda x: x[0])
    valid_stocks = [display_name for _, display_name in valid_stocks]

    return valid_stocks, stock_ticker_map
