"""Recovery probability calculator based on historical drawdown data"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from data_loader import load_ark_holdings, get_ark_files_hash, ARK_ETFS
from drawdown_calculator import calculate_drawdowns
from config import INPUT_DIR, ANALYSIS_PERIODS, DEFAULT_PERIOD


def _get_ark_drawdowns_cache_path():
    """Get cache file path for ARK stock drawdowns"""
    return INPUT_DIR / 'ark_etfs' / 'all_stock_drawdowns_cache.parquet'


def _is_cache_valid(cache_path, source_mtime):
    """Check if cache file exists and is newer than source"""
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_mtime


def get_ark_drawdowns_cache():
    """Get precomputed ARK drawdowns from cache file (fast)"""
    cache_file = _get_ark_drawdowns_cache_path()
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Convert date columns back to datetime
        for col in ['peak_date', 'trough_date', 'recovery_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    return None


def save_ark_drawdowns_cache(df):
    """Save ARK drawdowns to cache file"""
    cache_file = _get_ark_drawdowns_cache_path()
    df.to_parquet(cache_file, index=False)


@st.cache_data
def _calculate_all_stock_drawdowns_full(_files_hash):
    """Calculate all historical drawdowns for all stocks across ARK ETFs for ALL dates (internal, cached to file)

    Returns:
        DataFrame with columns: ticker, etf, peak_date, trough_date, depth_pct,
                                recovery_date, recovered, days_to_recover (unfiltered)
    """
    cache_path = _get_ark_drawdowns_cache_path()
    if _is_cache_valid(cache_path, _files_hash):
        df = pd.read_parquet(cache_path)
        # Convert date columns back to datetime
        for col in ['peak_date', 'trough_date', 'recovery_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    all_drawdowns = []

    for etf in ARK_ETFS:
        try:
            holdings = load_ark_holdings(_files_hash, etf)

            # Get unique tickers
            for ticker in holdings['Ticker'].unique():
                # Skip currency tickers
                ticker_holdings = holdings[holdings['Ticker'] == ticker]
                if 'Bloomberg Name' in ticker_holdings.columns:
                    bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                    if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                        continue

                stock_data = holdings[holdings['Ticker'] == ticker].copy()

                if len(stock_data) < 30:  # Need at least 30 data points
                    continue

                # Determine which price column to use
                if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                    price_col = 'YFinance Close Price'
                else:
                    price_col = 'Stock_Price'

                # Prepare price dataframe
                price_df = stock_data[['Date', price_col]].copy()
                price_df.columns = ['Date', 'Close']
                price_df = price_df.dropna()

                if len(price_df) < 30:
                    continue

                # Calculate drawdowns for ALL dates (no filtering)
                # Use the actual date range from the data
                data_start = price_df['Date'].min()
                data_end = price_df['Date'].max()
                dd_data = calculate_drawdowns(price_df, start_date=data_start, end_date=data_end)
                if len(dd_data) == 0 or 'rank' not in dd_data.columns:
                    continue
                historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

                for _, dd in historical_dd.iterrows():
                    # Check if this drawdown recovered (price returned to peak)
                    peak_date = dd['peak_date']
                    trough_date = dd['trough_date']
                    peak_price = dd['peak_price']
                    trough_price = dd['trough_price']

                    # Get all prices after trough
                    future_prices = price_df[price_df['Date'] > trough_date]

                    # Find first date when price >= peak_price
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
            print(f"Error processing {etf}: {e}")
            continue

    df = pd.DataFrame(all_drawdowns)

    # Save to file cache for future use
    if len(df) > 0:
        df.to_parquet(cache_path, index=False)

    return df


@st.cache_data
def calculate_all_stock_drawdowns(_files_hash, period_key, _start_date, _end_date):
    """Calculate all historical drawdowns for all stocks across ARK ETFs, filtered by period

    Args:
        period_key, _start_date, _end_date: Analysis period for filtering

    Returns:
        DataFrame with columns: ticker, etf, peak_date, trough_date, depth_pct,
                                recovery_date, recovered, days_to_recover
    """
    # Load full data (from file cache if available)
    full_data = _calculate_all_stock_drawdowns_full(_files_hash)

    if len(full_data) == 0:
        return full_data

    # Filter to analysis period based on peak_date
    filtered = full_data[
        (full_data['peak_date'] >= _start_date) &
        (full_data['peak_date'] <= _end_date)
    ].copy()

    return filtered


@st.cache_data
def calculate_recovery_probabilities(_files_hash, period_key, _start_date, _end_date):
    """Calculate recovery probabilities for different drawdown depth ranges

    Args:
        period_key, _start_date, _end_date: Analysis period for filtering

    Returns:
        DataFrame with columns: depth_range, total_events, recovered_events, recovery_probability
    """
    all_dd = calculate_all_stock_drawdowns(_files_hash, period_key, _start_date, _end_date)

    if len(all_dd) == 0:
        return pd.DataFrame(columns=['depth_range', 'total_events', 'recovered_events', 'recovery_probability'])

    # Define depth ranges (bins)
    # Bins must be monotonically increasing: from -inf to 0
    # Using right=True (default), intervals are (left, right], i.e., left exclusive, right inclusive
    bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
    labels = ['< -80%', '-70% to -80%', '-60% to -70%', '-50% to -60%',
              '-40% to -50%', '-30% to -40%', '-20% to -30%', '-10% to -20%', '0% to -10%']

    # Assign each drawdown to a depth range
    all_dd['depth_range'] = pd.cut(all_dd['depth_pct'], bins=bins, labels=labels, ordered=False)

    # Calculate recovery statistics for each range
    recovery_stats = []
    for depth_range in labels:
        range_dd = all_dd[all_dd['depth_range'] == depth_range]
        total_events = len(range_dd)

        if total_events > 0:
            recovered_events = range_dd['recovered'].sum()
            recovery_probability = recovered_events / total_events
        else:
            recovered_events = 0
            recovery_probability = 0.0

        recovery_stats.append({
            'depth_range': depth_range,
            'total_events': total_events,
            'recovered_events': recovered_events,
            'recovery_probability': recovery_probability
        })

    df = pd.DataFrame(recovery_stats)
    return df


def get_recovery_probability_for_depth(depth_pct, period_key=None, start_date=None, end_date=None):
    """Get recovery probability for a specific drawdown depth

    Args:
        depth_pct: Drawdown depth percentage (e.g., -25.5)
        period_key, start_date, end_date: Analysis period for filtering

    Returns:
        Recovery probability (0-1) or None if no data available
    """
    # Use default period if not specified
    if period_key is None:
        period_key = DEFAULT_PERIOD
        period = ANALYSIS_PERIODS[period_key]
        start_date = period["start"]
        end_date = period["end"]

    recovery_probs = calculate_recovery_probabilities(get_ark_files_hash(), period_key, start_date, end_date)

    if len(recovery_probs) == 0:
        return None

    # Determine which depth range this drawdown falls into
    bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
    labels = ['< -80%', '-70% to -80%', '-60% to -70%', '-50% to -60%',
              '-40% to -50%', '-30% to -40%', '-20% to -30%', '-10% to -20%', '0% to -10%']

    depth_range = pd.cut([depth_pct], bins=bins, labels=labels, ordered=False)[0]

    # Look up recovery probability for this range
    matching_rows = recovery_probs[recovery_probs['depth_range'] == depth_range]

    if len(matching_rows) > 0:
        return matching_rows.iloc[0]['recovery_probability']
    else:
        return None


def get_drawdowns_in_depth_range(depth_range_label, period_key=None, start_date=None, end_date=None):
    """Get all historical drawdowns within a specific depth range

    Args:
        depth_range_label: String like '0% to -10%', '-10% to -20%', etc.
        period_key, start_date, end_date: Analysis period for filtering

    Returns:
        DataFrame with columns: ticker, etf, peak_date, trough_date, depth_pct, duration_days,
                                peak_price, trough_price, recovery_date, recovered,
                                days_to_recover, recovery_rate
        Or empty DataFrame if no data
    """
    # Use default period if not specified
    if period_key is None:
        period_key = DEFAULT_PERIOD
        period = ANALYSIS_PERIODS[period_key]
        start_date = period["start"]
        end_date = period["end"]

    files_hash = get_ark_files_hash()
    all_dd = calculate_all_stock_drawdowns(files_hash, period_key, start_date, end_date)

    if len(all_dd) == 0:
        return pd.DataFrame()

    # Define depth ranges
    bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
    labels = ['< -80%', '-70% to -80%', '-60% to -70%', '-50% to -60%',
              '-40% to -50%', '-30% to -40%', '-20% to -30%', '-10% to -20%', '0% to -10%']

    # Assign each drawdown to a depth range
    all_dd['depth_range'] = pd.cut(all_dd['depth_pct'], bins=bins, labels=labels, ordered=False)

    # Filter to requested range
    range_dd = all_dd[all_dd['depth_range'] == depth_range_label].copy()

    if len(range_dd) == 0:
        return pd.DataFrame()

    # Calculate duration (peak to trough)
    range_dd['duration_days'] = (range_dd['trough_date'] - range_dd['peak_date']).dt.days

    # Calculate recovery rate for each drawdown
    # For drawdowns that haven't recovered, we need to get the latest price after trough
    detailed_dd = []

    for _, dd in range_dd.iterrows():
        ticker = dd['ticker']
        etf = dd['etf']
        peak_date = dd['peak_date']
        trough_date = dd['trough_date']
        peak_price = dd['peak_price']
        trough_price = dd['trough_price']

        try:
            # If already recovered, recovery_rate = 100%
            if dd['recovered']:
                recovery_rate = 1.0
            else:
                # Need to get latest price after trough to calculate current recovery rate
                holdings = load_ark_holdings(files_hash, etf)
                holdings = holdings[(holdings['Date'] >= start_date) & (holdings['Date'] <= end_date)]
                stock_data = holdings[holdings['Ticker'] == ticker].copy()

                if len(stock_data) == 0:
                    recovery_rate = 0.0
                else:
                    # Determine price column
                    if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                        price_col = 'YFinance Close Price'
                    else:
                        price_col = 'Stock_Price'

                    # Get latest price after trough
                    after_trough = stock_data[stock_data['Date'] > trough_date]
                    if len(after_trough) > 0:
                        latest_price = after_trough[price_col].iloc[-1]
                        if peak_price != trough_price:
                            recovery_rate = (latest_price - trough_price) / (peak_price - trough_price)
                        else:
                            recovery_rate = 0.0
                    else:
                        recovery_rate = 0.0

            detailed_dd.append({
                'ticker': ticker,
                'etf': etf,
                'peak_date': peak_date,
                'trough_date': trough_date,
                'depth_pct': dd['depth_pct'],
                'duration_days': dd['duration_days'],
                'peak_price': peak_price,
                'trough_price': trough_price,
                'recovery_date': dd['recovery_date'],
                'recovered': dd['recovered'],
                'days_to_recover': dd['days_to_recover'],
                'recovery_rate': recovery_rate
            })

        except Exception as e:
            print(f"Error processing {ticker} from {etf}: {e}")
            continue

    return pd.DataFrame(detailed_dd)


def get_stock_drawdowns_in_depth_range(ticker, etf, depth_range_label, period_key=None, start_date=None, end_date=None):
    """Get all historical drawdowns for a specific stock within a depth range

    Args:
        ticker: Stock ticker (e.g., 'TSLA')
        etf: ETF name (e.g., 'ARKK')
        depth_range_label: String like '0% to -10%', '-10% to -20%', etc.
        period_key, start_date, end_date: Analysis period for filtering

    Returns:
        DataFrame with columns: peak_date, trough_date, depth_pct, duration_days,
                                peak_price, trough_price, recovery_date, recovered,
                                days_to_recover, recovery_rate
        Or empty DataFrame if no data
    """
    # Use default period if not specified
    if period_key is None:
        period_key = DEFAULT_PERIOD
        period = ANALYSIS_PERIODS[period_key]
        start_date = period["start"]
        end_date = period["end"]

    try:
        # Load stock data
        holdings = load_ark_holdings(get_ark_files_hash(), etf)
        holdings = holdings[(holdings['Date'] >= start_date) & (holdings['Date'] <= end_date)]

        # Find matching ticker
        stock_data = holdings[holdings['Ticker'].str.startswith(ticker + ' ', na=False) |
                             (holdings['Ticker'] == ticker)].copy()

        if len(stock_data) < 30:
            return pd.DataFrame()

        # Determine price column
        if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
            price_col = 'YFinance Close Price'
        else:
            price_col = 'Stock_Price'

        # Prepare price dataframe
        price_df = stock_data[['Date', price_col]].copy()
        price_df.columns = ['Date', 'Close']
        price_df = price_df.dropna()

        if len(price_df) < 30:
            return pd.DataFrame()

        # Calculate all drawdowns (excluding current)
        dd_data = calculate_drawdowns(price_df)
        historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

        if len(historical_dd) == 0:
            return pd.DataFrame()

        # Define depth ranges
        bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
        labels = ['< -80%', '-70% to -80%', '-60% to -70%', '-50% to -60%',
                  '-40% to -50%', '-30% to -40%', '-20% to -30%', '-10% to -20%', '0% to -10%']

        # Assign each drawdown to a depth range
        historical_dd['depth_range'] = pd.cut(historical_dd['depth_pct'], bins=bins, labels=labels, ordered=False)

        # Filter to requested range
        range_dd = historical_dd[historical_dd['depth_range'] == depth_range_label].copy()

        if len(range_dd) == 0:
            return pd.DataFrame()

        # Calculate duration and recovery info
        detailed_dd = []

        for _, dd in range_dd.iterrows():
            peak_date = dd['peak_date']
            trough_date = dd['trough_date']
            peak_price = dd['peak_price']
            trough_price = dd['trough_price']

            # Calculate duration
            duration_days = (trough_date - peak_date).days

            # Check if recovered
            future_prices = price_df[price_df['Date'] > trough_date]
            recovery_dates = future_prices[future_prices['Close'] >= peak_price]

            if len(recovery_dates) > 0:
                recovery_date = recovery_dates.iloc[0]['Date']
                recovered = True
                days_to_recover = (recovery_date - trough_date).days
                recovery_rate = 1.0
            else:
                recovery_date = None
                recovered = False
                days_to_recover = None

                # Calculate current recovery rate
                if len(future_prices) > 0:
                    latest_price = future_prices['Close'].iloc[-1]
                    if peak_price != trough_price:
                        recovery_rate = (latest_price - trough_price) / (peak_price - trough_price)
                    else:
                        recovery_rate = 0.0
                else:
                    recovery_rate = 0.0

            detailed_dd.append({
                'peak_date': peak_date,
                'trough_date': trough_date,
                'depth_pct': dd['depth_pct'],
                'duration_days': duration_days,
                'peak_price': peak_price,
                'trough_price': trough_price,
                'recovery_date': recovery_date,
                'recovered': recovered,
                'days_to_recover': days_to_recover,
                'recovery_rate': recovery_rate
            })

        return pd.DataFrame(detailed_dd)

    except Exception as e:
        print(f"Error processing {ticker} from {etf}: {e}")
        return pd.DataFrame()


def get_etf_drawdowns_in_depth_range(etf, depth_range_label, period_key=None, start_date=None, end_date=None):
    """Get all historical drawdowns for all constituent stocks in an ETF within a depth range

    Args:
        etf: ETF name (e.g., 'ARKK')
        depth_range_label: String like '0% to -10%', '-10% to -20%', etc.
        period_key, start_date, end_date: Analysis period for filtering

    Returns:
        DataFrame with columns: ticker, peak_date, trough_date, depth_pct, duration_days,
                                peak_price, trough_price, recovery_date, recovered,
                                days_to_recover, recovery_rate
        Or empty DataFrame if no data
    """
    # Use default period if not specified
    if period_key is None:
        period_key = DEFAULT_PERIOD
        period = ANALYSIS_PERIODS[period_key]
        start_date = period["start"]
        end_date = period["end"]

    files_hash = get_ark_files_hash()

    # Use cached all_stock_drawdowns and filter by ETF
    all_dd = calculate_all_stock_drawdowns(files_hash, period_key, start_date, end_date)

    if len(all_dd) == 0:
        return pd.DataFrame()

    # Filter to this ETF only
    etf_dd = all_dd[all_dd['etf'] == etf].copy()

    if len(etf_dd) == 0:
        return pd.DataFrame()

    # Define depth ranges
    bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
    labels = ['< -80%', '-70% to -80%', '-60% to -70%', '-50% to -60%',
              '-40% to -50%', '-30% to -40%', '-20% to -30%', '-10% to -20%', '0% to -10%']

    # Assign each drawdown to a depth range
    etf_dd['depth_range'] = pd.cut(etf_dd['depth_pct'], bins=bins, labels=labels, ordered=False)

    # Filter to requested range
    range_dd = etf_dd[etf_dd['depth_range'] == depth_range_label].copy()

    if len(range_dd) == 0:
        return pd.DataFrame()

    # Calculate duration (peak to trough)
    range_dd['duration_days'] = (range_dd['trough_date'] - range_dd['peak_date']).dt.days

    # Calculate recovery rate for each drawdown
    detailed_dd = []

    for _, dd in range_dd.iterrows():
        ticker = dd['ticker']
        peak_price = dd['peak_price']
        trough_price = dd['trough_price']
        trough_date = dd['trough_date']

        # If already recovered, recovery_rate = 100%
        if dd['recovered']:
            recovery_rate = 1.0
        else:
            # Need to get latest price after trough to calculate current recovery rate
            try:
                holdings = load_ark_holdings(files_hash, etf)
                holdings = holdings[(holdings['Date'] >= start_date) & (holdings['Date'] <= end_date)]
                stock_data = holdings[holdings['Ticker'] == ticker].copy()

                if len(stock_data) == 0:
                    # Try matching by ticker prefix
                    stock_data = holdings[holdings['Ticker'].str.startswith(ticker + ' ', na=False)].copy()

                if len(stock_data) == 0:
                    recovery_rate = 0.0
                else:
                    # Determine price column
                    if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                        price_col = 'YFinance Close Price'
                    else:
                        price_col = 'Stock_Price'

                    # Get latest price after trough
                    after_trough = stock_data[stock_data['Date'] > trough_date]
                    if len(after_trough) > 0:
                        latest_price = after_trough[price_col].iloc[-1]
                        if peak_price != trough_price:
                            recovery_rate = (latest_price - trough_price) / (peak_price - trough_price)
                        else:
                            recovery_rate = 0.0
                    else:
                        recovery_rate = 0.0
            except Exception:
                recovery_rate = 0.0

        # Get simple ticker symbol
        ticker_simple = ticker.split()[0] if isinstance(ticker, str) else ticker

        detailed_dd.append({
            'ticker': ticker_simple,
            'peak_date': dd['peak_date'],
            'trough_date': dd['trough_date'],
            'depth_pct': dd['depth_pct'],
            'duration_days': dd['duration_days'],
            'peak_price': dd['peak_price'],
            'trough_price': dd['trough_price'],
            'recovery_date': dd['recovery_date'],
            'recovered': dd['recovered'],
            'days_to_recover': dd['days_to_recover'],
            'recovery_rate': recovery_rate
        })

    return pd.DataFrame(detailed_dd)
