"""HHI (Herfindahl-Hirschman Index) and Concentration Metrics Calculator"""
import pandas as pd
import numpy as np


def calculate_hhi(weights):
    """
    Calculate Herfindahl-Hirschman Index (HHI)

    HHI = Σ(weight_i²)
    Range: 1/n (perfectly diversified) to 1 (single position)

    Args:
        weights: array-like of portfolio weights (should sum to 1)

    Returns:
        float: HHI value
    """
    weights = np.array(weights)
    # Normalize weights to sum to 1 if they don't
    weights = weights / weights.sum()
    return np.sum(weights ** 2)


def calculate_effective_positions(hhi):
    """
    Calculate Effective Number of Positions (ENP)

    ENP = 1 / HHI
    Interpretation: equivalent number of equal-weighted positions

    Args:
        hhi: HHI value

    Returns:
        float: Effective number of positions
    """
    if hhi == 0:
        return np.inf
    return 1 / hhi


def calculate_top_n_concentration(weights, n=5):
    """
    Calculate concentration of top N positions

    Args:
        weights: array-like of portfolio weights
        n: number of top positions to consider

    Returns:
        float: sum of top N weights (as percentage)
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    sorted_weights = np.sort(weights)[::-1]  # Sort descending
    return np.sum(sorted_weights[:n])


def calculate_hhi_time_series(holdings_df):
    """
    Calculate HHI for each date in holdings data

    Args:
        holdings_df: DataFrame with columns ['Date', 'Ticker', 'Weight']

    Returns:
        DataFrame with columns ['Date', 'HHI', 'Effective_Positions', 'Top5_Concentration', 'Num_Positions']
    """
    results = []

    for date, group in holdings_df.groupby('Date'):
        weights = group['Weight'].values

        # Filter out zero/nan weights
        weights = weights[~np.isnan(weights)]
        weights = weights[weights > 0]

        if len(weights) == 0:
            continue

        hhi = calculate_hhi(weights)
        enp = calculate_effective_positions(hhi)
        top5 = calculate_top_n_concentration(weights, n=5)

        results.append({
            'Date': date,
            'HHI': hhi,
            'Effective_Positions': enp,
            'Top5_Concentration': top5,
            'Num_Positions': len(weights)
        })

    return pd.DataFrame(results)


def calculate_weighted_correlation(returns_df, weights_df, date):
    """
    Calculate weighted pairwise correlation for a given date

    The weighted correlation gives more weight to pairs involving
    high-conviction (large weight) positions.

    Args:
        returns_df: DataFrame of returns with tickers as columns
        weights_df: DataFrame with ['Date', 'Ticker', 'Weight']
        date: the date for which to get weights

    Returns:
        tuple: (weighted_mean_corr, unweighted_mean_corr)
    """
    # Get weights for the date
    date_weights = weights_df[weights_df['Date'] == date].set_index('Ticker')['Weight']

    # Get common tickers
    common_tickers = returns_df.columns.intersection(date_weights.index)

    if len(common_tickers) < 2:
        return np.nan, np.nan

    # Filter to common tickers
    returns_subset = returns_df[common_tickers]
    weights_subset = date_weights.loc[common_tickers]

    # Normalize weights
    weights_subset = weights_subset / weights_subset.sum()

    # Calculate correlation matrix
    corr_matrix = returns_subset.corr()

    # Extract upper triangle (pairwise correlations)
    n = len(common_tickers)
    unweighted_corrs = []
    weighted_corrs = []
    pair_weights = []

    for i in range(n):
        for j in range(i + 1, n):
            ticker_i = common_tickers[i]
            ticker_j = common_tickers[j]

            corr_val = corr_matrix.iloc[i, j]
            if np.isnan(corr_val):
                continue

            unweighted_corrs.append(corr_val)

            # Weight for this pair = product of individual weights
            # (pairs with two large positions get highest weight)
            w_i = weights_subset.iloc[i]
            w_j = weights_subset.iloc[j]
            pair_weight = w_i * w_j

            weighted_corrs.append(corr_val)
            pair_weights.append(pair_weight)

    if len(unweighted_corrs) == 0:
        return np.nan, np.nan

    unweighted_mean = np.mean(unweighted_corrs)

    # Weighted mean
    pair_weights = np.array(pair_weights)
    weighted_corrs = np.array(weighted_corrs)
    weighted_mean = np.average(weighted_corrs, weights=pair_weights)

    return weighted_mean, unweighted_mean


def calculate_rolling_weighted_correlation(returns_df, holdings_df, window=20):
    """
    Calculate rolling weighted and unweighted correlation over time

    Args:
        returns_df: DataFrame of daily returns with Date index and Ticker columns
        holdings_df: DataFrame with ['Date', 'Ticker', 'Weight']
        window: rolling window size in days

    Returns:
        DataFrame with ['Date', 'Weighted_Corr', 'Unweighted_Corr']
    """
    results = []

    # Get unique dates from returns
    dates = returns_df.index.unique()

    for i in range(window, len(dates)):
        current_date = dates[i]
        window_start = dates[i - window]

        # Get returns for window
        window_returns = returns_df.loc[window_start:current_date]

        if len(window_returns) < window * 0.5:  # Need at least 50% of window
            continue

        # Find closest holdings date
        holdings_dates = holdings_df['Date'].unique()
        closest_holdings_date = holdings_dates[holdings_dates <= current_date]
        if len(closest_holdings_date) == 0:
            continue
        closest_holdings_date = closest_holdings_date.max()

        # Calculate weighted correlation
        weighted_corr, unweighted_corr = calculate_weighted_correlation(
            window_returns, holdings_df, closest_holdings_date
        )

        if not np.isnan(weighted_corr):
            results.append({
                'Date': current_date,
                'Weighted_Corr': weighted_corr,
                'Unweighted_Corr': unweighted_corr
            })

    return pd.DataFrame(results)
