"""Peer group price calculation logic"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from data_loader import load_r3000_holdings, load_industry_info, get_r3000_files_hash
from config import START_DATE, END_DATE, INPUT_DIR


def _get_peer_group_cache_path(version='mv'):
    """Get cache file path for peer group prices"""
    return INPUT_DIR / 'russell_3000' / f'peer_group_{version}_cache.parquet'


def _is_cache_valid(cache_path, source_mtime):
    """Check if cache file exists and is newer than source"""
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_mtime


@st.cache_data
def calculate_iwv_total_market_value(_files_hash):
    """Calculate IWV total market value (sum of all holdings)

    Returns:
        DataFrame with columns: Date, Value
    """
    cache_path = INPUT_DIR / 'russell_3000' / 'iwv_total_mv_cache.parquet'
    if _is_cache_valid(cache_path, _files_hash):
        return pd.read_parquet(cache_path)

    holdings = load_r3000_holdings(_files_hash)

    # Filter out dates where less than 50% of stocks have valid prices
    holdings['_valid_price'] = (holdings['Price'] > 0).astype(int)
    date_stats = holdings.groupby('Date').agg(
        valid_count=('_valid_price', 'sum'),
        total_count=('_valid_price', 'size')
    )
    date_stats['valid_pct'] = date_stats['valid_count'] / date_stats['total_count']
    valid_dates = date_stats[date_stats['valid_pct'] > 0.5].index
    holdings = holdings[holdings['Date'].isin(valid_dates)].copy()

    # Calculate Market Value
    holdings['Market_Value'] = holdings['Position'] * holdings['Price']

    # Sum all market values by Date
    total_mv = holdings.groupby('Date')['Market_Value'].sum().reset_index()
    total_mv.columns = ['Date', 'Value']

    total_mv.to_parquet(cache_path, index=False)
    return total_mv


@st.cache_data
def calculate_peer_group_prices_mv(_files_hash):
    """Calculate peer group total market values (sum of market values by GICS)

    Uses file-based caching for faster startup after data updates.

    Returns:
        DataFrame with columns: Date, GICS, Value
    """
    # Check file cache first
    cache_path = _get_peer_group_cache_path('mv')
    if _is_cache_valid(cache_path, _files_hash):
        return pd.read_parquet(cache_path)

    holdings = load_r3000_holdings(_files_hash)
    industry_dict = load_industry_info(source='r3000')

    # Filter out dates where less than 50% of stocks have valid prices (Price > 0)
    # Vectorized: count valid prices and total per date, then filter
    holdings['_valid_price'] = (holdings['Price'] > 0).astype(int)
    date_stats = holdings.groupby('Date').agg(
        valid_count=('_valid_price', 'sum'),
        total_count=('_valid_price', 'size')
    )
    date_stats['valid_pct'] = date_stats['valid_count'] / date_stats['total_count']
    valid_dates = date_stats[date_stats['valid_pct'] > 0.5].index
    holdings = holdings[holdings['Date'].isin(valid_dates)].copy()
    holdings.drop(columns=['_valid_price'], inplace=True)

    # Calculate Market Value if not present
    if 'Market_Value' not in holdings.columns:
        if 'Position' in holdings.columns and 'Price' in holdings.columns:
            holdings['Market_Value'] = holdings['Position'] * holdings['Price']
        else:
            raise ValueError("Cannot calculate Market_Value: missing Position or Price columns")

    # Map GICS to tickers - vectorized approach
    # Extract symbol (first part before space) for all tickers at once
    holdings['Symbol'] = holdings['Ticker'].str.split().str[0]

    # Try mapping by symbol first (most common case), then by full ticker
    holdings['GICS'] = holdings['Symbol'].map(industry_dict)
    still_unmatched = holdings['GICS'].isna()
    if still_unmatched.sum() > 0:
        holdings.loc[still_unmatched, 'GICS'] = holdings.loc[still_unmatched, 'Ticker'].map(industry_dict)

    # Filter holdings with valid GICS info
    holdings_with_gics = holdings[holdings['GICS'].notna()].copy()

    # Sum market values by Date and GICS
    peer_prices = holdings_with_gics.groupby(['Date', 'GICS'])['Market_Value'].sum().reset_index()
    peer_prices.columns = ['Date', 'GICS', 'Value']

    # Save to file cache
    peer_prices.to_parquet(cache_path, index=False)

    return peer_prices


@st.cache_data
def calculate_peer_group_prices_weighted(_files_hash):
    """Calculate peer group weighted prices

    For each stock:
    1. Calculate weight = stock's Market_Value / total R3000 Market_Value on that day
    2. Calculate weighted_price = weight × stock's Price
    3. Sum weighted_prices by GICS group

    Uses file-based caching for faster startup after data updates.

    Returns:
        DataFrame with columns: Date, GICS, Value
    """
    # Check file cache first
    cache_path = _get_peer_group_cache_path('weighted')
    if _is_cache_valid(cache_path, _files_hash):
        return pd.read_parquet(cache_path)

    holdings = load_r3000_holdings(_files_hash)
    industry_dict = load_industry_info(source='r3000')

    # Filter out dates where less than 50% of stocks have valid prices (Price > 0)
    # Vectorized: count valid prices and total per date, then filter
    holdings['_valid_price'] = (holdings['Price'] > 0).astype(int)
    date_stats = holdings.groupby('Date').agg(
        valid_count=('_valid_price', 'sum'),
        total_count=('_valid_price', 'size')
    )
    date_stats['valid_pct'] = date_stats['valid_count'] / date_stats['total_count']
    valid_dates = date_stats[date_stats['valid_pct'] > 0.5].index
    holdings = holdings[holdings['Date'].isin(valid_dates)].copy()
    holdings.drop(columns=['_valid_price'], inplace=True)

    # Calculate Market Value if not present
    if 'Market_Value' not in holdings.columns:
        if 'Position' in holdings.columns and 'Price' in holdings.columns:
            holdings['Market_Value'] = holdings['Position'] * holdings['Price']
        else:
            raise ValueError("Cannot calculate Market_Value: missing Position or Price columns")

    # Map GICS to tickers - vectorized approach
    holdings['Symbol'] = holdings['Ticker'].str.split().str[0]
    holdings['GICS'] = holdings['Symbol'].map(industry_dict)
    still_unmatched = holdings['GICS'].isna()
    if still_unmatched.sum() > 0:
        holdings.loc[still_unmatched, 'GICS'] = holdings.loc[still_unmatched, 'Ticker'].map(industry_dict)

    # Filter holdings with valid GICS info
    holdings_with_gics = holdings[holdings['GICS'].notna()].copy()

    # Calculate total R3000 market value for each date using transform (faster than merge)
    holdings_with_gics['Total_MV'] = holdings_with_gics.groupby('Date')['Market_Value'].transform('sum')

    # Calculate weight = stock's MV / total R3000 MV
    holdings_with_gics['Weight'] = holdings_with_gics['Market_Value'] / holdings_with_gics['Total_MV']

    # Calculate weighted price = weight × stock price
    holdings_with_gics['Weighted_Price'] = holdings_with_gics['Weight'] * holdings_with_gics['Price']

    # Sum weighted prices by Date and GICS
    peer_prices = holdings_with_gics.groupby(['Date', 'GICS'])['Weighted_Price'].sum().reset_index()
    peer_prices.columns = ['Date', 'GICS', 'Value']

    # Save to file cache
    peer_prices.to_parquet(cache_path, index=False)

    return peer_prices


def get_peer_group_prices(industry, version='mv'):
    """Get price data for a specific industry peer group

    Args:
        industry: GICS industry name (may be truncated Excel sheet name)
        version: 'mv' for Market Value or 'weighted' for Weighted Price

    Returns:
        DataFrame with columns: Date, Value
    """
    # Map truncated Excel sheet names to full GICS names
    # Excel sheet names are limited to 31 characters
    name_mapping = {
        'Commercial & Professional Serv': 'Commercial & Professional Services',
        'Consumer Discretionary Distrib': 'Consumer Discretionary Distribution & Retail',
        'Consumer Staples Distribution': 'Consumer Staples Distribution & Retail',
        'Equity Real Estate Investment': 'Equity Real Estate Investment Trusts (REITs)',
        'Health Care Equipment & Servic': 'Health Care Equipment & Services',
        'Pharmaceuticals, Biotechnology': 'Pharmaceuticals, Biotechnology & Life Sciences',
        'Real Estate Management & Devel': 'Real Estate Management & Development',
        'Semiconductors & Semiconductor': 'Semiconductors & Semiconductor Equipment',
        'Technology Hardware & Equipmen': 'Technology Hardware & Equipment',
    }

    # Use mapped name if available, otherwise use original
    full_industry_name = name_mapping.get(industry, industry)

    files_hash = get_r3000_files_hash()
    if version == 'mv':
        all_prices = calculate_peer_group_prices_mv(files_hash)
    else:
        all_prices = calculate_peer_group_prices_weighted(files_hash)

    industry_prices = all_prices[all_prices['GICS'] == full_industry_name].copy()
    industry_prices = industry_prices[['Date', 'Value']].sort_values('Date')

    return industry_prices
