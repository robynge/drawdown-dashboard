"""Recovery probability calculator based on historical drawdown data"""
import pandas as pd
import numpy as np
from data_loader import load_ark_holdings, ARK_ETFS
from drawdown_calculator import calculate_drawdowns
from config import START_DATE, END_DATE
import pickle
from pathlib import Path

_cache = {}
CACHE_FILE = Path(__file__).parent.parent / 'data' / 'cache' / 'recovery_probabilities.pkl'

def calculate_all_stock_drawdowns():
    """Calculate all historical drawdowns for all stocks across ARK ETFs

    Returns:
        DataFrame with columns: ticker, etf, peak_date, trough_date, depth_pct,
                                recovery_date, recovered, days_to_recover
    """
    cache_key = f'all_stock_drawdowns_{START_DATE}_{END_DATE}'
    if cache_key in _cache:
        return _cache[cache_key]

    all_drawdowns = []

    for etf in ARK_ETFS:
        try:
            holdings = load_ark_holdings(etf)

            # Filter to analysis period
            holdings = holdings[(holdings['Date'] >= START_DATE) & (holdings['Date'] <= END_DATE)]

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

                # Calculate drawdowns (excluding current)
                dd_data = calculate_drawdowns(price_df)
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
    _cache[cache_key] = df
    return df


def calculate_recovery_probabilities():
    """Calculate recovery probabilities for different drawdown depth ranges

    Returns:
        DataFrame with columns: depth_range, total_events, recovered_events, recovery_probability
    """
    cache_key = f'recovery_probabilities_{START_DATE}_{END_DATE}'
    if cache_key in _cache:
        return _cache[cache_key]

    # Try to load from disk cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data['cache_key'] == cache_key:
                    _cache[cache_key] = cached_data['data']
                    return cached_data['data']
        except:
            pass

    # Calculate from scratch
    all_dd = calculate_all_stock_drawdowns()

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
    _cache[cache_key] = df

    # Save to disk cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'cache_key': cache_key, 'data': df}, f)
    except Exception as e:
        print(f"Failed to save recovery probability cache: {e}")

    return df


def get_recovery_probability_for_depth(depth_pct):
    """Get recovery probability for a specific drawdown depth

    Args:
        depth_pct: Drawdown depth percentage (e.g., -25.5)

    Returns:
        Recovery probability (0-1) or None if no data available
    """
    recovery_probs = calculate_recovery_probabilities()

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


def get_drawdowns_in_depth_range(depth_range_label):
    """Get all historical drawdowns within a specific depth range

    Args:
        depth_range_label: String like '0% to -10%', '-10% to -20%', etc.

    Returns:
        DataFrame with columns: ticker, etf, peak_date, trough_date, depth_pct, duration_days,
                                peak_price, trough_price, recovery_date, recovered,
                                days_to_recover, recovery_rate
        Or empty DataFrame if no data
    """
    all_dd = calculate_all_stock_drawdowns()

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
                holdings = load_ark_holdings(etf)
                holdings = holdings[(holdings['Date'] >= START_DATE) & (holdings['Date'] <= END_DATE)]
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


def get_stock_drawdowns_in_depth_range(ticker, etf, depth_range_label):
    """Get all historical drawdowns for a specific stock within a depth range

    Args:
        ticker: Stock ticker (e.g., 'TSLA')
        etf: ETF name (e.g., 'ARKK')
        depth_range_label: String like '0% to -10%', '-10% to -20%', etc.

    Returns:
        DataFrame with columns: peak_date, trough_date, depth_pct, duration_days,
                                peak_price, trough_price, recovery_date, recovered,
                                days_to_recover, recovery_rate
        Or empty DataFrame if no data
    """
    try:
        # Load stock data
        holdings = load_ark_holdings(etf)
        holdings = holdings[(holdings['Date'] >= START_DATE) & (holdings['Date'] <= END_DATE)]

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


def get_etf_drawdowns_in_depth_range(etf, depth_range_label):
    """Get all historical drawdowns for all constituent stocks in an ETF within a depth range

    Args:
        etf: ETF name (e.g., 'ARKK')
        depth_range_label: String like '0% to -10%', '-10% to -20%', etc.

    Returns:
        DataFrame with columns: ticker, peak_date, trough_date, depth_pct, duration_days,
                                peak_price, trough_price, recovery_date, recovered,
                                days_to_recover, recovery_rate
        Or empty DataFrame if no data
    """
    try:
        # Load ETF holdings
        holdings = load_ark_holdings(etf)
        holdings = holdings[(holdings['Date'] >= START_DATE) & (holdings['Date'] <= END_DATE)]

        # Define depth ranges
        bins = [-float('inf'), -80, -70, -60, -50, -40, -30, -20, -10, 0]
        labels = ['< -80%', '-70% to -80%', '-60% to -70%', '-50% to -60%',
                  '-40% to -50%', '-30% to -40%', '-20% to -30%', '-10% to -20%', '0% to -10%']

        all_constituent_dd = []

        # Get unique tickers in this ETF
        for ticker in holdings['Ticker'].unique():
            # Skip currency tickers
            ticker_holdings = holdings[holdings['Ticker'] == ticker]
            if 'Bloomberg Name' in ticker_holdings.columns:
                bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                    continue

            stock_data = holdings[holdings['Ticker'] == ticker].copy()

            if len(stock_data) < 30:
                continue

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
                continue

            # Calculate drawdowns (excluding current)
            dd_data = calculate_drawdowns(price_df)
            historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

            if len(historical_dd) == 0:
                continue

            # Assign each drawdown to a depth range
            historical_dd['depth_range'] = pd.cut(historical_dd['depth_pct'], bins=bins, labels=labels, ordered=False)

            # Filter to requested range
            range_dd = historical_dd[historical_dd['depth_range'] == depth_range_label].copy()

            if len(range_dd) == 0:
                continue

            # Process each drawdown for this stock
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

                # Get simple ticker symbol
                ticker_simple = ticker.split()[0] if isinstance(ticker, str) else ticker

                all_constituent_dd.append({
                    'ticker': ticker_simple,
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

        return pd.DataFrame(all_constituent_dd)

    except Exception as e:
        print(f"Error processing ETF {etf}: {e}")
        return pd.DataFrame()
