"""Peer group price calculation logic"""
import pandas as pd
import numpy as np
from data_loader import load_r3000_holdings, load_industry_info
from config import START_DATE, END_DATE

_cache = {}

def calculate_peer_group_prices_mv():
    """Calculate peer group total market values (sum of market values by GICS)

    Returns:
        DataFrame with columns: Date, GICS, Value
    """
    cache_key = f'peer_group_prices_mv_{START_DATE}_{END_DATE}'
    if cache_key in _cache:
        return _cache[cache_key]

    holdings = load_r3000_holdings()
    industry_dict = load_industry_info(source='r3000')

    # Filter out dates where less than 50% of stocks have valid prices (Price > 0)
    # This removes US holidays where only a few Canadian stocks have prices
    daily_valid_pct = holdings.groupby('Date')['Price'].apply(lambda x: (x > 0).sum() / len(x))
    valid_dates = daily_valid_pct[daily_valid_pct > 0.5].index
    holdings = holdings[holdings['Date'].isin(valid_dates)].copy()

    # Calculate Market Value if not present
    if 'Market_Value' not in holdings.columns:
        if 'Position' in holdings.columns and 'Price' in holdings.columns:
            holdings['Market_Value'] = holdings['Position'] * holdings['Price']
        else:
            raise ValueError("Cannot calculate Market_Value: missing Position or Price columns")

    # Map GICS to tickers
    # First try exact match on full ticker
    holdings['GICS'] = holdings['Ticker'].map(industry_dict)

    # For unmatched tickers, try matching by symbol only (first part before space)
    unmatched_mask = holdings['GICS'].isna()
    if unmatched_mask.sum() > 0:
        holdings.loc[unmatched_mask, 'Symbol'] = holdings.loc[unmatched_mask, 'Ticker'].str.split().str[0]
        holdings.loc[unmatched_mask, 'GICS'] = holdings.loc[unmatched_mask, 'Symbol'].map(industry_dict)

    # Filter holdings with valid GICS info
    holdings_with_gics = holdings[holdings['GICS'].notna()].copy()

    # Sum market values by Date and GICS
    peer_prices = holdings_with_gics.groupby(['Date', 'GICS'])['Market_Value'].sum().reset_index()
    peer_prices.columns = ['Date', 'GICS', 'Value']

    _cache[cache_key] = peer_prices
    return peer_prices


def calculate_peer_group_prices_weighted():
    """Calculate peer group weighted prices

    For each stock:
    1. Calculate weight = stock's Market_Value / total R3000 Market_Value on that day
    2. Calculate weighted_price = weight × stock's Price
    3. Sum weighted_prices by GICS group

    Returns:
        DataFrame with columns: Date, GICS, Value
    """
    cache_key = f'peer_group_prices_weighted_{START_DATE}_{END_DATE}'
    if cache_key in _cache:
        return _cache[cache_key]

    holdings = load_r3000_holdings()
    industry_dict = load_industry_info(source='r3000')

    # Filter out dates where less than 50% of stocks have valid prices (Price > 0)
    # This removes US holidays where only a few Canadian stocks have prices
    daily_valid_pct = holdings.groupby('Date')['Price'].apply(lambda x: (x > 0).sum() / len(x))
    valid_dates = daily_valid_pct[daily_valid_pct > 0.5].index
    holdings = holdings[holdings['Date'].isin(valid_dates)].copy()

    # Calculate Market Value if not present
    if 'Market_Value' not in holdings.columns:
        if 'Position' in holdings.columns and 'Price' in holdings.columns:
            holdings['Market_Value'] = holdings['Position'] * holdings['Price']
        else:
            raise ValueError("Cannot calculate Market_Value: missing Position or Price columns")

    # Map GICS to tickers
    # First try exact match on full ticker
    holdings['GICS'] = holdings['Ticker'].map(industry_dict)

    # For unmatched tickers, try matching by symbol only
    unmatched_mask = holdings['GICS'].isna()
    if unmatched_mask.sum() > 0:
        holdings.loc[unmatched_mask, 'Symbol'] = holdings.loc[unmatched_mask, 'Ticker'].str.split().str[0]
        holdings.loc[unmatched_mask, 'GICS'] = holdings.loc[unmatched_mask, 'Symbol'].map(industry_dict)

    # Filter holdings with valid GICS info
    holdings_with_gics = holdings[holdings['GICS'].notna()].copy()

    # Calculate total R3000 market value for each date
    daily_total_mv = holdings_with_gics.groupby('Date')['Market_Value'].sum().reset_index()
    daily_total_mv.columns = ['Date', 'Total_MV']

    # Merge to get daily total MV for each stock
    holdings_with_gics = holdings_with_gics.merge(daily_total_mv, on='Date', how='left')

    # Calculate weight = stock's MV / total R3000 MV
    holdings_with_gics['Weight'] = holdings_with_gics['Market_Value'] / holdings_with_gics['Total_MV']

    # Calculate weighted price = weight × stock price
    holdings_with_gics['Weighted_Price'] = holdings_with_gics['Weight'] * holdings_with_gics['Price']

    # Sum weighted prices by Date and GICS
    peer_prices = holdings_with_gics.groupby(['Date', 'GICS'])['Weighted_Price'].sum().reset_index()
    peer_prices.columns = ['Date', 'GICS', 'Value']

    _cache[cache_key] = peer_prices
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

    if version == 'mv':
        all_prices = calculate_peer_group_prices_mv()
    else:
        all_prices = calculate_peer_group_prices_weighted()

    industry_prices = all_prices[all_prices['GICS'] == full_industry_name].copy()
    industry_prices = industry_prices[['Date', 'Value']].sort_values('Date')

    return industry_prices
