"""Drawdown calculation logic"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure src directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import START_DATE, END_DATE

def find_max_drawdown_in_period(df, value_col='Close'):
    """Find the maximum drawdown in a given period"""
    if df.empty or len(df) < 2:
        return None

    df = df.copy()
    df['Running_Peak'] = df[value_col].cummax()
    df['Drawdown'] = (df[value_col] - df['Running_Peak']) / df['Running_Peak'] * 100

    min_dd = df['Drawdown'].min()
    if pd.isna(min_dd) or min_dd >= 0:
        return None

    min_idx = df['Drawdown'].idxmin()
    if pd.isna(min_idx):
        return None

    df_before = df.loc[:min_idx]
    peak_val = df_before[value_col].max()
    peak_idx = df_before[value_col].idxmax()

    if pd.isna(peak_idx):
        return None

    return {
        'peak_date': peak_idx,
        'trough_date': min_idx,
        'peak_price': peak_val,
        'trough_price': df.loc[min_idx, value_col],
        'depth_pct': min_dd
    }

def find_top_n_drawdowns(df, n=10, value_col='Close'):
    """Find top N drawdowns in non-overlapping periods using iterative global search"""
    drawdowns = []
    remaining_periods = [(df.index[0], df.index[-1])]

    for rank in range(1, n + 1):
        best_dd = None
        best_dd_value = 0
        best_period_idx = -1
        best_split = None

        # Find the deepest drawdown across ALL remaining periods (global search)
        for i, (start, end) in enumerate(remaining_periods):
            period_df = df.loc[start:end]
            dd = find_max_drawdown_in_period(period_df, value_col)

            if dd and dd['depth_pct'] < best_dd_value:  # depth_pct is negative
                best_dd = dd
                best_dd_value = dd['depth_pct']
                best_period_idx = i
                best_split = (start, end)

        if best_dd is None:
            break

        best_dd['rank'] = rank
        best_dd['Name'] = f'Drawdown_{rank}'
        drawdowns.append(best_dd)

        # Remove the used period and add split periods
        start, end = best_split
        peak_date = best_dd['peak_date']
        trough_date = best_dd['trough_date']

        remaining_periods.pop(best_period_idx)

        # Add period before peak
        if start < peak_date and (peak_date - start).days > 1:
            day_before_peak = peak_date - pd.Timedelta(days=1)
            if day_before_peak >= start:
                remaining_periods.append((start, day_before_peak))

        # Add period after trough
        if trough_date < end and (end - trough_date).days > 1:
            day_after_trough = trough_date + pd.Timedelta(days=1)
            if day_after_trough <= end:
                remaining_periods.append((day_after_trough, end))

    return drawdowns

def calculate_drawdowns(prices_df, ticker=None, start_date=None, end_date=None):
    """Calculate top drawdowns and current drawdown

    Args:
        prices_df: DataFrame with 'Date' and 'Close' columns
        ticker: Optional ticker name for output
        start_date: Start of analysis period (defaults to config START_DATE)
        end_date: End of analysis period (defaults to config END_DATE)
    """
    start = start_date if start_date is not None else START_DATE
    end = end_date if end_date is not None else END_DATE

    prices = prices_df.copy()
    prices = prices[(prices['Date'] >= start) & (prices['Date'] <= end)].copy()

    if len(prices) == 0 or prices['Close'].isna().all():
        return pd.DataFrame()

    # Set Date as index for the drawdown calculation
    prices_indexed = prices.set_index('Date')

    # Find top 10 drawdowns using the correct algorithm
    top_dds = find_top_n_drawdowns(prices_indexed, n=10, value_col='Close')

    # Current drawdown
    peak_price = prices_indexed['Close'].max()
    peak_date = prices_indexed['Close'].idxmax()
    current_price = prices_indexed['Close'].iloc[-1]
    current_date = prices_indexed.index[-1]
    actual_current_dd = ((current_price - peak_price) / peak_price) * 100

    results = []
    for dd in top_dds:
        results.append({
            'rank': dd['rank'],
            'peak_date': dd['peak_date'],
            'trough_date': dd['trough_date'],
            'peak_price': round(dd['peak_price'], 2),
            'trough_price': round(dd['trough_price'], 2),
            'depth_pct': round(dd['depth_pct'], 2)
        })

    # Add current drawdown
    results.append({
        'rank': 'Current',
        'peak_date': peak_date,
        'trough_date': current_date,
        'peak_price': round(peak_price, 2),
        'trough_price': round(current_price, 2),
        'depth_pct': round(actual_current_dd, 2)
    })

    df = pd.DataFrame(results)

    # Convert rank to string to avoid PyArrow serialization issues
    if 'rank' in df.columns:
        df['rank'] = df['rank'].astype(str)

    if ticker:
        df.insert(0, 'ticker', ticker)

    return df

def find_all_valid_drawdowns(df, value_col='Close', min_depth_pct=10, min_duration_days=7):
    """Find all drawdowns that meet criteria: depth >= min_depth_pct OR duration >= min_duration_days

    Args:
        df: DataFrame with DatetimeIndex and value_col
        value_col: Column name for price values
        min_depth_pct: Minimum depth percentage (positive number, e.g., 10 for -10%)
        min_duration_days: Minimum duration in trading days

    Returns:
        List of drawdown dictionaries
    """
    drawdowns = []
    remaining_periods = [(df.index[0], df.index[-1])]
    rank = 0

    while remaining_periods:
        best_dd = None
        best_dd_value = 0
        best_period_idx = -1
        best_split = None

        # Find the deepest drawdown across ALL remaining periods
        for i, (start, end) in enumerate(remaining_periods):
            period_df = df.loc[start:end]
            dd = find_max_drawdown_in_period(period_df, value_col)

            if dd and dd['depth_pct'] < best_dd_value:
                best_dd = dd
                best_dd_value = dd['depth_pct']
                best_period_idx = i
                best_split = (start, end)

        if best_dd is None:
            break

        # Calculate duration in trading days
        peak_date = best_dd['peak_date']
        trough_date = best_dd['trough_date']
        duration_days = len(df.loc[peak_date:trough_date])

        # Check if this drawdown meets our criteria
        depth_qualifies = abs(best_dd['depth_pct']) >= min_depth_pct
        duration_qualifies = duration_days >= min_duration_days

        if depth_qualifies or duration_qualifies:
            rank += 1
            best_dd['rank'] = rank
            best_dd['duration_days'] = duration_days
            drawdowns.append(best_dd)

            # Remove the used period and add split periods
            start, end = best_split
            remaining_periods.pop(best_period_idx)

            # Add period before peak
            if start < peak_date and (peak_date - start).days > 1:
                day_before_peak = peak_date - pd.Timedelta(days=1)
                if day_before_peak >= start:
                    remaining_periods.append((start, day_before_peak))

            # Add period after trough
            if trough_date < end and (end - trough_date).days > 1:
                day_after_trough = trough_date + pd.Timedelta(days=1)
                if day_after_trough <= end:
                    remaining_periods.append((day_after_trough, end))
        else:
            # This drawdown doesn't qualify, stop searching
            break

    return drawdowns


def calculate_drawdowns_with_filter(prices_df, min_depth_pct=10, min_duration_days=7, start_date=None, end_date=None):
    """Calculate all valid drawdowns meeting criteria for distribution analysis

    Args:
        prices_df: DataFrame with 'Date' and 'Close' columns
        min_depth_pct: Minimum depth percentage (positive number, e.g., 10 for -10%)
        min_duration_days: Minimum duration in trading days
        start_date: Start of analysis period (defaults to config START_DATE)
        end_date: End of analysis period (defaults to config END_DATE)

    Returns:
        DataFrame with all valid drawdowns
    """
    start = start_date if start_date is not None else START_DATE
    end = end_date if end_date is not None else END_DATE

    prices = prices_df.copy()
    prices = prices[(prices['Date'] >= start) & (prices['Date'] <= end)].copy()

    if len(prices) == 0 or prices['Close'].isna().all():
        return pd.DataFrame()

    prices_indexed = prices.set_index('Date')

    # Find all valid drawdowns
    all_dds = find_all_valid_drawdowns(
        prices_indexed,
        value_col='Close',
        min_depth_pct=min_depth_pct,
        min_duration_days=min_duration_days
    )

    if not all_dds:
        return pd.DataFrame()

    results = []
    for dd in all_dds:
        results.append({
            'rank': dd['rank'],
            'peak_date': dd['peak_date'],
            'trough_date': dd['trough_date'],
            'peak_price': round(dd['peak_price'], 2),
            'trough_price': round(dd['trough_price'], 2),
            'depth_pct': round(dd['depth_pct'], 2),
            'duration_days': dd['duration_days']
        })

    df = pd.DataFrame(results)
    df['rank'] = df['rank'].astype(str)

    return df


def calculate_stock_drawdowns_by_etf(holdings_df, prices_df, ticker, start_date=None, end_date=None):
    """Calculate drawdowns for a stock grouped by ETF holdings

    Args:
        holdings_df: DataFrame with holdings data
        prices_df: DataFrame with price data
        ticker: Stock ticker
        start_date: Start of analysis period (defaults to config START_DATE)
        end_date: End of analysis period (defaults to config END_DATE)
    """
    start = start_date if start_date is not None else START_DATE
    end = end_date if end_date is not None else END_DATE

    results = {}

    for etf in holdings_df['ETF'].unique():
        etf_holdings = holdings_df[holdings_df['ETF'] == etf].copy()
        etf_holdings = etf_holdings[(etf_holdings['Date'] >= start) &
                                     (etf_holdings['Date'] <= end)]

        if len(etf_holdings) == 0:
            continue

        # Merge holdings with prices
        merged = pd.merge(etf_holdings[['Date', 'Shares', 'Market Value']],
                         prices_df[['Date', 'Close']], on='Date', how='inner')

        if len(merged) == 0:
            continue

        dd_df = calculate_drawdowns(merged.rename(columns={'Close': 'Close'}), ticker=None, start_date=start, end_date=end)
        if len(dd_df) > 0:
            results[etf] = dd_df

    return results
