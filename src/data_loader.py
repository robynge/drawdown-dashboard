"""Data loading without manual caching - caching handled by Streamlit"""
import pandas as pd
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, ARK_ETFS


def load_ark_holdings(etf):
    """Load ARK ETF holdings - no caching, let Streamlit handle it"""
    file_path = INPUT_DIR / 'ark_etfs' / f'{etf}_Transformed_Data.xlsx'
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Fix CUSIP column type for PyArrow compatibility
    if 'CUSIP' in df.columns:
        df['CUSIP'] = df['CUSIP'].astype(str)

    return df


def load_r3000_holdings():
    """Load Russell 3000 holdings - no caching, let Streamlit handle it"""
    file_path = INPUT_DIR / 'russell_3000' / 'IWV_Transformed_Data.xlsx'
    all_data = []
    for sheet in ['2024', '2025']:
        df_sheet = pd.read_excel(file_path, sheet_name=sheet)
        all_data.append(df_sheet)

    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Fix CUSIP column type for PyArrow compatibility
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
        for ticker in holdings['Ticker'].unique():
            # Skip currency tickers - check Bloomberg Name
            ticker_holdings = holdings[holdings['Ticker'] == ticker]
            if 'Bloomberg Name' in ticker_holdings.columns:
                bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                    continue
            all_tickers.add(ticker)
    return sorted(all_tickers)


def get_stock_etf_mapping():
    """Map each stock to the ETFs it appears in"""
    stock_map = {}
    for etf in ARK_ETFS:
        holdings = load_ark_holdings(etf)
        for ticker in holdings['Ticker'].unique():
            # Skip currency tickers - check Bloomberg Name
            ticker_holdings = holdings[holdings['Ticker'] == ticker]
            if 'Bloomberg Name' in ticker_holdings.columns:
                bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                    continue

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
