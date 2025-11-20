"""Drawdown calculation logic"""
import pandas as pd
import numpy as np
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

def calculate_drawdowns(prices_df, ticker=None):
    """Calculate top drawdowns and current drawdown"""
    prices = prices_df.copy()
    prices = prices[(prices['Date'] >= START_DATE) & (prices['Date'] <= END_DATE)].copy()

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
    if ticker:
        df.insert(0, 'ticker', ticker)

    return df

def calculate_stock_drawdowns_by_etf(holdings_df, prices_df, ticker):
    """Calculate drawdowns for a stock grouped by ETF holdings"""
    results = {}

    for etf in holdings_df['ETF'].unique():
        etf_holdings = holdings_df[holdings_df['ETF'] == etf].copy()
        etf_holdings = etf_holdings[(etf_holdings['Date'] >= START_DATE) &
                                     (etf_holdings['Date'] <= END_DATE)]

        if len(etf_holdings) == 0:
            continue

        # Merge holdings with prices
        merged = pd.merge(etf_holdings[['Date', 'Shares', 'Market Value']],
                         prices_df[['Date', 'Close']], on='Date', how='inner')

        if len(merged) == 0:
            continue

        dd_df = calculate_drawdowns(merged.rename(columns={'Close': 'Close'}), ticker=None)
        if len(dd_df) > 0:
            results[etf] = dd_df

    return results
