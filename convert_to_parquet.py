"""Convert Excel files to Parquet format and precompute all data for Streamlit dashboard"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

INPUT_DIR = Path(__file__).parent / 'input'
OUTPUT_DIR = Path(__file__).parent / 'output'
ARK_PRECOMPUTED_DIR = INPUT_DIR / 'ark_etfs' / 'precomputed'
R3000_PRECOMPUTED_DIR = INPUT_DIR / 'russell_3000' / 'precomputed'
METADATA_DIR = INPUT_DIR / 'precomputed'


def ensure_dirs():
    """Create precomputed directories if they don't exist"""
    ARK_PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    R3000_PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def convert_ark_etfs():
    """Convert ARK ETF Excel files to Parquet"""
    ark_dir = INPUT_DIR / 'ark_etfs'

    for xlsx_file in ark_dir.glob('*_Transformed_Data.xlsx'):
        print(f"Converting {xlsx_file.name}...")

        df = pd.read_excel(xlsx_file)
        df['Date'] = pd.to_datetime(df['Date'])

        # Fix CUSIP column type
        if 'CUSIP' in df.columns:
            df['CUSIP'] = df['CUSIP'].astype(str)

        # Save as Parquet
        parquet_file = xlsx_file.with_suffix('.parquet')
        df.to_parquet(parquet_file, index=False)

        # Show size comparison
        xlsx_size = xlsx_file.stat().st_size / (1024 * 1024)
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)
        print(f"  {xlsx_size:.1f} MB -> {parquet_size:.1f} MB ({parquet_size/xlsx_size*100:.0f}%)")


def convert_russell_3000():
    """Convert Russell 3000 Excel file to Parquet"""
    r3000_dir = INPUT_DIR / 'russell_3000'
    xlsx_file = r3000_dir / 'IWV_Transformed_Data.xlsx'

    if not xlsx_file.exists():
        print(f"File not found: {xlsx_file}")
        return

    print(f"Converting {xlsx_file.name}...")

    # Read all sheets and combine
    all_data = []
    xl = pd.ExcelFile(xlsx_file)
    for sheet in xl.sheet_names:
        print(f"  Reading sheet: {sheet}")
        df_sheet = pd.read_excel(xlsx_file, sheet_name=sheet)
        all_data.append(df_sheet)

    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Fix CUSIP column type
    if 'CUSIP' in df.columns:
        df['CUSIP'] = df['CUSIP'].astype(str)

    # Save as Parquet
    parquet_file = xlsx_file.with_suffix('.parquet')
    df.to_parquet(parquet_file, index=False)

    # Show size comparison
    xlsx_size = xlsx_file.stat().st_size / (1024 * 1024)
    parquet_size = parquet_file.stat().st_size / (1024 * 1024)
    print(f"  {xlsx_size:.1f} MB -> {parquet_size:.1f} MB ({parquet_size/xlsx_size*100:.0f}%)")


def precompute_peer_group_cache():
    """Precompute peer group cache files (full time series, no period filter)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import get_r3000_files_hash

    files_hash = get_r3000_files_hash()

    # Import the _full versions that compute for ALL dates
    from peer_group import (
        _calculate_iwv_total_market_value_full,
        _calculate_peer_group_prices_mv_full,
        _calculate_peer_group_prices_weighted_full
    )

    print("  Computing IWV Total Market Value (full)...")
    iwv_mv = _calculate_iwv_total_market_value_full(files_hash)
    print(f"    Saved {len(iwv_mv)} records")

    print("  Computing Peer Group Market Values (full)...")
    pg_mv = _calculate_peer_group_prices_mv_full(files_hash)
    print(f"    Saved {len(pg_mv)} records")

    print("  Computing Peer Group Weighted Prices (full)...")
    pg_weighted = _calculate_peer_group_prices_weighted_full(files_hash)
    print(f"    Saved {len(pg_weighted)} records")


def precompute_ark_holdings_max_drawdowns():
    """Precompute max drawdown for ALL stocks that have ever been in ARK holdings"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from data_loader import load_ark_holdings, get_ark_files_hash
    from drawdown_calculator import calculate_drawdowns_with_filter

    all_results = []

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(get_ark_files_hash(), etf)

        if len(holdings) == 0:
            continue

        # Filter out currency and money market funds
        if 'Bloomberg Name' in holdings.columns:
            currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
            holdings = holdings[~holdings['Ticker'].isin(currency_tickers)]

        money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']
        holdings = holdings[~holdings['Ticker'].str.split().str[0].apply(
            lambda t: any(t.startswith(p) for p in money_market_prefixes)
        )]

        # Get all unique tickers
        all_tickers = holdings['Ticker'].unique()

        for ticker in all_tickers:
            stock_data = holdings[holdings['Ticker'] == ticker].copy()

            if len(stock_data) < 10:
                continue

            price_df = stock_data[['Date', 'Stock_Price']].copy()
            price_df = price_df.rename(columns={'Stock_Price': 'Close'})
            price_df = price_df.dropna(subset=['Close'])

            if len(price_df) < 10:
                continue

            # Calculate drawdowns with standard filter
            dd_df = calculate_drawdowns_with_filter(price_df, min_depth_pct=10, min_duration_days=7)

            if len(dd_df) > 0:
                max_dd = dd_df['depth_pct'].min()
                ticker_clean = ticker.split()[0]
                all_results.append({
                    'etf': etf,
                    'ticker': ticker_clean,
                    'max_drawdown': max_dd,
                    'num_drawdowns': len(dd_df),
                    'first_date': stock_data['Date'].min(),
                    'last_date': stock_data['Date'].max()
                })

    # Save to precomputed directory
    if all_results:
        result_df = pd.DataFrame(all_results)
        output_path = ARK_PRECOMPUTED_DIR / 'ark_holdings_max_drawdowns.parquet'
        result_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(result_df)} records to {output_path.name}")


def precompute_r3000_drawdowns():
    """Precompute R3000 drawdowns for all stocks (takes several minutes)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import load_r3000_holdings, _load_industry_info_impl, get_r3000_files_hash, save_r3000_drawdowns_cache
    from drawdown_calculator import calculate_drawdowns_with_filter

    files_hash = get_r3000_files_hash()
    holdings = load_r3000_holdings(files_hash)

    if len(holdings) == 0:
        print("  No R3000 holdings data found")
        return

    # Get unique tickers
    all_tickers = holdings['Ticker'].unique()
    industry_dict = _load_industry_info_impl(source='r3000')

    print(f"  Processing {len(all_tickers)} tickers...")

    results = []
    for i, ticker in enumerate(all_tickers):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{len(all_tickers)}")

        stock_data = holdings[holdings['Ticker'] == ticker].copy()

        if len(stock_data) < 10:
            continue

        if 'Price' not in stock_data.columns:
            continue

        price_df = stock_data[['Date', 'Price']].copy()
        price_df = price_df.rename(columns={'Price': 'Close'})
        price_df = price_df.dropna(subset=['Close'])

        if len(price_df) < 10:
            continue

        dd_df = calculate_drawdowns_with_filter(price_df, min_depth_pct=10, min_duration_days=7)

        if len(dd_df) > 0:
            max_dd = dd_df['depth_pct'].min()
            ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
            gics = industry_dict.get(ticker, industry_dict.get(ticker_clean, 'Unknown'))
            results.append({
                'ticker': ticker_clean,
                'max_drawdown': max_dd,
                'num_drawdowns': len(dd_df),
                'gics_industry_group': gics
            })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        save_r3000_drawdowns_cache(result_df)
        print(f"  Saved {len(result_df)} ticker drawdowns to cache")


def precompute_ark_drawdowns():
    """Precompute ARK stock drawdowns for all stocks (takes several minutes)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS, START_DATE, END_DATE
    from data_loader import load_ark_holdings, get_ark_files_hash
    from drawdown_calculator import calculate_drawdowns
    from recovery_probability import save_ark_drawdowns_cache

    all_drawdowns = []

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        try:
            holdings = load_ark_holdings(get_ark_files_hash(), etf)
            holdings = holdings[(holdings['Date'] >= START_DATE) & (holdings['Date'] <= END_DATE)]

            tickers = holdings['Ticker'].unique()
            for ticker in tickers:
                ticker_holdings = holdings[holdings['Ticker'] == ticker]
                if 'Bloomberg Name' in ticker_holdings.columns:
                    bloomberg_name = ticker_holdings['Bloomberg Name'].iloc[0]
                    if isinstance(bloomberg_name, str) and 'curncy' in bloomberg_name.lower():
                        continue

                stock_data = holdings[holdings['Ticker'] == ticker].copy()
                if len(stock_data) < 30:
                    continue

                if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                    price_col = 'YFinance Close Price'
                else:
                    price_col = 'Stock_Price'

                price_df = stock_data[['Date', price_col]].copy()
                price_df.columns = ['Date', 'Close']
                price_df = price_df.dropna()

                if len(price_df) < 30:
                    continue

                # Use actual data date range for drawdown calculation
                data_start = price_df['Date'].min()
                data_end = price_df['Date'].max()
                dd_data = calculate_drawdowns(price_df, start_date=data_start, end_date=data_end)
                if len(dd_data) == 0 or 'rank' not in dd_data.columns:
                    continue
                historical_dd = dd_data[dd_data['rank'] != 'Current'].copy()

                for _, dd in historical_dd.iterrows():
                    peak_date = dd['peak_date']
                    trough_date = dd['trough_date']
                    peak_price = dd['peak_price']
                    trough_price = dd['trough_price']

                    future_prices = price_df[price_df['Date'] > trough_date]
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
            print(f"    Error: {e}")
            continue

    df = pd.DataFrame(all_drawdowns)
    if len(df) > 0:
        save_ark_drawdowns_cache(df)
        print(f"  Saved {len(df)} drawdown records to cache")


# ============================================================================
# NEW PRECOMPUTATION STEPS (Steps 6-14)
# ============================================================================

def _filter_non_stocks(holdings):
    """Filter out currency and money market tickers"""
    result = holdings.copy()

    # Filter out currency tickers
    if 'Bloomberg Name' in result.columns:
        result = result[~result['Bloomberg Name'].str.contains('curncy', case=False, na=False)]

    # Filter out money market funds
    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']
    ticker_symbols = result['Ticker'].str.split().str[0]
    is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
    result = result[~is_mm]

    return result


def calculate_average_pair_weights(weight_matrix, tickers):
    """Calculate average daily pair weights for weighted mean correlation (vectorized).

    Instead of using single-day weights (e.g., at peak or period end), this calculates
    the average of daily pair weights over all overlapping days. This gives more
    representative weights for stocks with varying presence in the portfolio.

    Args:
        weight_matrix: DataFrame (Date x Ticker) with daily weights
        tickers: List of tickers to calculate weights for (must match correlation matrix columns)

    Returns:
        pair_weights: 1D array of upper triangle pair weights (length = n*(n-1)/2)
    """
    n = len(tickers)
    triu_i, triu_j = np.triu_indices(n, k=1)

    # Align weight_matrix to tickers, fill missing with 0
    wm = weight_matrix.reindex(columns=tickers).fillna(0).values  # Shape: (T, n)

    # Vectorized calculation: for each pair (i, j), compute mean of w_i * w_j where both > 0
    # Extract columns for all pairs at once
    w_i = wm[:, triu_i]  # Shape: (T, num_pairs)
    w_j = wm[:, triu_j]  # Shape: (T, num_pairs)

    # Pair products for all days and all pairs
    pair_products = w_i * w_j  # Shape: (T, num_pairs)

    # Valid mask: both weights > 0
    valid_mask = (w_i > 0) & (w_j > 0)  # Shape: (T, num_pairs)

    # Count valid days per pair
    valid_counts = valid_mask.sum(axis=0)  # Shape: (num_pairs,)

    # Sum of valid pair products
    masked_products = np.where(valid_mask, pair_products, 0.0)
    pair_sums = masked_products.sum(axis=0)  # Shape: (num_pairs,)

    # Average (avoid division by zero warning)
    pair_weights = np.zeros_like(pair_sums)
    nonzero_mask = valid_counts > 0
    pair_weights[nonzero_mask] = pair_sums[nonzero_mask] / valid_counts[nonzero_mask]

    return pair_weights


def precompute_etf_drawdowns():
    """Step 6: Precompute ETF-level drawdowns for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        price_file = OUTPUT_DIR / f'{etf}_prices.csv'

        if not price_file.exists():
            print(f"    Price file not found: {price_file}")
            continue

        prices = pd.read_csv(price_file)
        prices['Date'] = pd.to_datetime(prices['Date'])

        # Calculate drawdowns for full date range
        dd_df = calculate_drawdowns(prices, start_date=prices['Date'].min(), end_date=prices['Date'].max())

        if len(dd_df) > 0:
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_etf_drawdowns.parquet'
            dd_df.to_parquet(output_path, index=False)
            print(f"    Saved {len(dd_df)} drawdowns")


def precompute_hhi_timeseries():
    """Step 7: Precompute HHI time series for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from data_loader import load_ark_holdings, get_ark_files_hash
    from hhi_calculator import calculate_hhi_time_series

    ensure_dirs()
    files_hash = get_ark_files_hash()

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)

        # Filter non-stocks
        holdings_filtered = _filter_non_stocks(holdings)

        # Calculate HHI time series
        hhi_df = calculate_hhi_time_series(holdings_filtered[['Date', 'Ticker', 'Weight']])

        if len(hhi_df) > 0:
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_hhi_timeseries.parquet'
            hhi_df.to_parquet(output_path, index=False)
            print(f"    Saved {len(hhi_df)} HHI records")


def precompute_correlation_matrices():
    """Step 8: Precompute correlation matrices for each ARK ETF and analysis period"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS, ANALYSIS_PERIODS
    from data_loader import load_ark_holdings, get_ark_files_hash

    ensure_dirs()
    files_hash = get_ark_files_hash()

    lookback_periods = [60, 120, 250]

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)
        holdings_filtered = _filter_non_stocks(holdings)

        if len(holdings_filtered) == 0:
            continue

        for period_key, period_config in ANALYSIS_PERIODS.items():
            period_end = period_config["end"]

            # Get holdings up to the period end date
            holdings_period = holdings_filtered[holdings_filtered['Date'] <= period_end].copy()
            if len(holdings_period) == 0:
                continue

            # Get the latest date within this period
            period_latest_date = holdings_period['Date'].max()
            current_tickers = holdings_period[holdings_period['Date'] == period_latest_date]['Ticker'].unique()

            # Save current weights for all current holdings (before correlation filtering)
            # This allows pages to detect which tickers were excluded due to insufficient data
            current_weights = holdings_period[holdings_period['Date'] == period_latest_date][['Ticker', 'Weight']].copy()
            weights_path = ARK_PRECOMPUTED_DIR / f'{etf}_current_weights_{period_key}.parquet'
            current_weights.to_parquet(weights_path, index=False)

            for lookback_days in lookback_periods:
                # Calculate lookback start from period end date
                lookback_start = period_end - pd.Timedelta(days=lookback_days)

                # Filter to lookback period and current tickers
                holdings_lookback = holdings_period[
                    (holdings_period['Date'] >= lookback_start) &
                    (holdings_period['Ticker'].isin(current_tickers))
                ].copy()

                if len(holdings_lookback) == 0:
                    continue

                # Pivot to get price matrix
                price_matrix = holdings_lookback.pivot_table(
                    index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
                )

                if len(price_matrix.columns) < 2:
                    continue

                # Calculate returns and correlation
                # Use min_periods=20 to require at least 20 overlapping days for each pair
                returns = price_matrix.pct_change(fill_method=None).iloc[1:]
                corr_matrix = returns.corr(min_periods=20)

                # Remove tickers with no valid correlations (all NaN off-diagonal)
                has_valid = (corr_matrix.notna().sum() > 1)  # >1 because diagonal is 1.0
                corr_matrix = corr_matrix.loc[has_valid, has_valid]

                # Save correlation matrix with period key in filename
                output_path = ARK_PRECOMPUTED_DIR / f'{etf}_correlation_matrix_{period_key}_{lookback_days}d.parquet'
                corr_matrix.to_parquet(output_path)

                # Save returns for later use (rolling correlations need them)
                returns_reset = returns.reset_index()
                returns_path = ARK_PRECOMPUTED_DIR / f'{etf}_returns_{period_key}_{lookback_days}d.parquet'
                returns_reset.to_parquet(returns_path, index=False)

                print(f"    Saved {period_key} {lookback_days}d correlation matrix ({len(corr_matrix)}x{len(corr_matrix)})")


def precompute_weighted_correlations():
    """Step 9: Precompute weighted correlation matrices for each ARK ETF and analysis period"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS, ANALYSIS_PERIODS
    from data_loader import load_ark_holdings, get_ark_files_hash

    ensure_dirs()
    files_hash = get_ark_files_hash()

    lookback_periods = [60, 120, 250]

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)
        holdings_filtered = _filter_non_stocks(holdings)

        if len(holdings_filtered) == 0:
            continue

        for period_key, period_config in ANALYSIS_PERIODS.items():
            period_end = period_config["end"]

            # Get holdings up to the period end date
            holdings_period = holdings_filtered[holdings_filtered['Date'] <= period_end].copy()
            if len(holdings_period) == 0:
                continue

            # Get the latest date within this period
            period_latest_date = holdings_period['Date'].max()
            current_tickers = holdings_period[holdings_period['Date'] == period_latest_date]['Ticker'].unique()

            for lookback_days in lookback_periods:
                # Calculate lookback start from period end date
                lookback_start = period_end - pd.Timedelta(days=lookback_days)

                holdings_lookback = holdings_period[
                    (holdings_period['Date'] >= lookback_start) &
                    (holdings_period['Ticker'].isin(current_tickers))
                ].copy()

                if len(holdings_lookback) == 0:
                    continue

                # Pivot for price and weight matrices
                price_matrix = holdings_lookback.pivot_table(
                    index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
                )
                weight_matrix = holdings_lookback.pivot_table(
                    index='Date', columns='Ticker', values='Weight', aggfunc='first'
                )

                if len(price_matrix.columns) < 2:
                    continue

                # Calculate returns
                returns = price_matrix.pct_change(fill_method=None).iloc[1:]
                weight_matrix = weight_matrix.ffill()
                weight_matrix = weight_matrix.loc[returns.index]

                tickers = returns.columns.tolist()
                n = len(tickers)

                # Initialize weighted correlation matrix
                weighted_corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)

                # Vectorized weighted correlation calculation
                R = returns.values  # Shape: (T, n)
                W = weight_matrix.reindex(columns=tickers).fillna(0).values  # Shape: (T, n)

                triu_i, triu_j = np.triu_indices(n, k=1)

                for pair_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
                    R_A, R_B = R[:, i], R[:, j]
                    W_A, W_B = W[:, i], W[:, j]

                    W_t = W_A * W_B
                    valid_mask = (~np.isnan(R_A)) & (~np.isnan(R_B))

                    # Require at least 20 overlapping days
                    if valid_mask.sum() < 20:
                        corr_val = np.nan
                    else:
                        mask = valid_mask & (W_t > 0)
                        if mask.sum() < 20:
                            # Fall back to unweighted if not enough weighted data
                            corr_val = np.corrcoef(R_A[valid_mask], R_B[valid_mask])[0, 1]
                        else:
                            w = W_t[mask]
                            w = w / w.sum()
                            a, b = R_A[mask], R_B[mask]

                            mu_a, mu_b = np.dot(w, a), np.dot(w, b)
                            da, db = a - mu_a, b - mu_b
                            cov_ab = np.dot(w, da * db)
                            var_a, var_b = np.dot(w, da * da), np.dot(w, db * db)

                            if var_a > 0 and var_b > 0:
                                corr_val = cov_ab / np.sqrt(var_a * var_b)
                                corr_val = np.clip(corr_val, -1, 1)
                            else:
                                corr_val = np.nan

                    weighted_corr.iloc[i, j] = corr_val
                    weighted_corr.iloc[j, i] = corr_val

                # Remove tickers with no valid correlations (all NaN off-diagonal)
                has_valid = (weighted_corr.notna().sum() > 1)  # >1 because diagonal is 1.0
                weighted_corr = weighted_corr.loc[has_valid, has_valid]

                output_path = ARK_PRECOMPUTED_DIR / f'{etf}_weighted_correlation_matrix_{period_key}_{lookback_days}d.parquet'
                weighted_corr.to_parquet(output_path)
                print(f"    Saved {period_key} {lookback_days}d weighted correlation matrix ({len(weighted_corr)}x{len(weighted_corr)})")


def precompute_rolling_correlations():
    """Step 10: Precompute rolling correlations for each ARK ETF and analysis period

    Rolling window is now independent from lookback period.
    Generates files for each (period_key, rolling_window) combination.
    Rolling windows: 20, 30, 60, 120 days
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS, ANALYSIS_PERIODS
    from data_loader import load_ark_holdings, get_ark_files_hash

    ensure_dirs()
    files_hash = get_ark_files_hash()

    # Independent rolling windows for correlation time series
    rolling_windows = [20, 30, 60, 120]

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)
        holdings_filtered = _filter_non_stocks(holdings)

        if len(holdings_filtered) == 0:
            continue

        for period_key, period_config in ANALYSIS_PERIODS.items():
            period_start = period_config["start"]
            period_end = period_config["end"]

            # Get holdings within the period
            holdings_period = holdings_filtered[
                (holdings_filtered['Date'] >= period_start) &
                (holdings_filtered['Date'] <= period_end)
            ].copy()
            if len(holdings_period) == 0:
                continue

            # Get the latest date within this period
            period_latest_date = holdings_period['Date'].max()
            current_tickers = holdings_period[holdings_period['Date'] == period_latest_date]['Ticker'].unique()

            # Use full historical data for current tickers (up to period end)
            # Need to include data before period_start for rolling window calculation
            holdings_full = holdings_filtered[
                (holdings_filtered['Ticker'].isin(current_tickers)) &
                (holdings_filtered['Date'] <= period_end)
            ].copy()

            price_matrix = holdings_full.pivot_table(
                index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
            )
            weight_matrix = holdings_full.pivot_table(
                index='Date', columns='Ticker', values='Weight', aggfunc='first'
            )

            returns = price_matrix.pct_change(fill_method=None).iloc[1:]
            weight_matrix = weight_matrix.ffill()

            n_tickers = len(returns.columns)
            if n_tickers < 2:
                continue

            triu_i, triu_j = np.triu_indices(n_tickers, k=1)
            holdings_dates = np.sort(holdings_full['Date'].unique())

            for rolling_window in rolling_windows:
                if len(returns) < rolling_window:
                    print(f"    Skipping {period_key} {rolling_window}d (not enough data)")
                    continue

                results = []
                dates = returns.index

                for i in range(rolling_window, len(returns) + 1):
                    window_returns = returns.iloc[i - rolling_window:i]
                    current_date = dates[i - 1]

                    # Only include dates within the analysis period
                    if current_date < period_start:
                        continue

                    # Use min_periods=20 to require at least 20 overlapping days
                    corr_matrix = window_returns.corr(min_periods=20)

                    corr_values = corr_matrix.values[triu_i, triu_j]
                    valid_mask = ~np.isnan(corr_values)
                    valid_corrs = corr_values[valid_mask]

                    if len(valid_corrs) == 0:
                        continue

                    mean_corr = np.mean(valid_corrs)
                    median_corr = np.median(valid_corrs)

                    # Weighted correlation using average daily pair weights over the window
                    window_weight_matrix = weight_matrix.iloc[i - rolling_window:i]
                    pair_weights = calculate_average_pair_weights(window_weight_matrix, corr_matrix.columns.tolist())

                    valid_weight_mask = valid_mask & (pair_weights > 0)
                    if valid_weight_mask.any():
                        weighted_mean = np.average(corr_values[valid_weight_mask], weights=pair_weights[valid_weight_mask])
                    else:
                        weighted_mean = mean_corr

                    results.append({
                        'Date': current_date,
                        'mean_corr': mean_corr,
                        'median_corr': median_corr,
                        'weighted_mean_corr': weighted_mean
                    })

                if len(results) > 0:
                    rolling_df = pd.DataFrame(results)
                    output_path = ARK_PRECOMPUTED_DIR / f'{etf}_rolling_correlations_{period_key}_{rolling_window}d.parquet'
                    rolling_df.to_parquet(output_path, index=False)
                    print(f"    Saved {period_key} {rolling_window}d rolling correlations ({len(rolling_df)} records)")


def precompute_holdings_drawdowns():
    """Step 11: Precompute holdings drawdowns for each ARK ETF (already done by precompute_ark_drawdowns)"""
    print("  (Using existing all_stock_drawdowns_cache.parquet)")


def precompute_concentration_performance():
    """Step 12: Precompute concentration vs performance data for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS

    ensure_dirs()

    # Load QQQ prices
    qqq_file = OUTPUT_DIR / 'QQQ_prices.csv'
    if not qqq_file.exists():
        print("  QQQ prices not found, skipping")
        return

    qqq_prices = pd.read_csv(qqq_file)
    qqq_prices['Date'] = pd.to_datetime(qqq_prices['Date'])

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")

        # Load ETF prices
        etf_file = OUTPUT_DIR / f'{etf}_prices.csv'
        if not etf_file.exists():
            print(f"    ETF prices not found: {etf_file}")
            continue

        etf_prices = pd.read_csv(etf_file)
        etf_prices['Date'] = pd.to_datetime(etf_prices['Date'])

        # Load HHI data
        hhi_file = ARK_PRECOMPUTED_DIR / f'{etf}_hhi_timeseries.parquet'
        if not hhi_file.exists():
            print(f"    HHI data not found: {hhi_file}")
            continue

        hhi_data = pd.read_parquet(hhi_file)

        # Merge ETF and QQQ prices
        merged = pd.merge(
            etf_prices[['Date', 'Close']].rename(columns={'Close': 'ETF_Price'}),
            qqq_prices[['Date', 'Close']].rename(columns={'Close': 'QQQ_Price'}),
            on='Date', how='inner'
        )

        if len(merged) < 2:
            continue

        merged = merged.dropna(subset=['ETF_Price', 'QQQ_Price'])

        # Calculate returns
        merged['ETF_Daily_Return'] = merged['ETF_Price'].pct_change() * 100
        merged['QQQ_Daily_Return'] = merged['QQQ_Price'].pct_change() * 100
        merged['Spread'] = merged['ETF_Daily_Return'] - merged['QQQ_Daily_Return']

        # Cumulative returns
        first_etf = merged['ETF_Price'].iloc[0]
        first_qqq = merged['QQQ_Price'].iloc[0]
        merged['ETF_Cumulative'] = (merged['ETF_Price'] / first_etf - 1) * 100
        merged['QQQ_Cumulative'] = (merged['QQQ_Price'] / first_qqq - 1) * 100
        merged['Cumulative_Spread'] = merged['ETF_Cumulative'] - merged['QQQ_Cumulative']

        # Merge with HHI
        merged = pd.merge(merged, hhi_data[['Date', 'HHI', 'Effective_Positions']], on='Date', how='inner')
        merged['HHI_Change'] = merged['HHI'].diff()

        # Drop NaN rows
        merged = merged.dropna(subset=['ETF_Daily_Return', 'QQQ_Daily_Return', 'Spread'])

        if len(merged) > 0:
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_concentration_performance.parquet'
            merged.to_parquet(output_path, index=False)
            print(f"    Saved {len(merged)} records")


def precompute_r3000_drawdowns_full():
    """Step 13: Precompute R3000 drawdowns to parquet in precomputed dir

    Saves two files:
    - r3000_drawdowns_full.parquet: Summary with max_drawdown per ticker (for backward compatibility)
    - r3000_drawdowns_detailed.parquet: All individual drawdowns with dates (for period filtering)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import load_r3000_holdings, _load_industry_info_impl, get_r3000_files_hash
    from drawdown_calculator import calculate_drawdowns_with_filter

    ensure_dirs()
    files_hash = get_r3000_files_hash()
    holdings = load_r3000_holdings(files_hash)

    if len(holdings) == 0:
        print("  No R3000 holdings data found")
        return

    all_tickers = holdings['Ticker'].unique()
    industry_dict = _load_industry_info_impl(source='r3000')

    # Get data date range for reference
    data_min_date = holdings['Date'].min()
    data_max_date = holdings['Date'].max()
    print(f"  Data date range: {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}")
    print(f"  Processing {len(all_tickers)} tickers...")

    summary_results = []
    detailed_results = []

    for i, ticker in enumerate(all_tickers):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{len(all_tickers)}")

        stock_data = holdings[holdings['Ticker'] == ticker].copy()

        if len(stock_data) < 10 or 'Price' not in stock_data.columns:
            continue

        price_df = stock_data[['Date', 'Price']].copy()
        price_df = price_df.rename(columns={'Price': 'Close'})
        price_df = price_df.dropna(subset=['Close'])

        if len(price_df) < 10:
            continue

        # Use full date range (no filtering) to get all drawdowns
        dd_df = calculate_drawdowns_with_filter(
            price_df,
            min_depth_pct=10,
            min_duration_days=7,
            start_date=data_min_date,
            end_date=data_max_date
        )

        if len(dd_df) > 0:
            ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
            gics = industry_dict.get(ticker, industry_dict.get(ticker_clean, 'Unknown'))

            # Summary result (backward compatible)
            max_dd = dd_df['depth_pct'].min()
            summary_results.append({
                'ticker': ticker_clean,
                'max_drawdown': max_dd,
                'num_drawdowns': len(dd_df),
                'gics_industry_group': gics
            })

            # Detailed results (each drawdown with dates)
            for _, row in dd_df.iterrows():
                detailed_results.append({
                    'ticker': ticker_clean,
                    'ticker_full': ticker,
                    'gics_industry_group': gics,
                    'rank': row['rank'],
                    'peak_date': row['peak_date'],
                    'trough_date': row['trough_date'],
                    'depth_pct': row['depth_pct'],
                    'duration_days': row['duration_days']
                })

    # Save summary file (backward compatible)
    summary_df = pd.DataFrame(summary_results)
    if len(summary_df) > 0:
        output_path = R3000_PRECOMPUTED_DIR / 'r3000_drawdowns_full.parquet'
        summary_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(summary_df)} ticker summaries to {output_path}")

    # Save detailed file (new - for period filtering)
    detailed_df = pd.DataFrame(detailed_results)
    if len(detailed_df) > 0:
        output_path = R3000_PRECOMPUTED_DIR / 'r3000_drawdowns_with_dates.parquet'
        detailed_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(detailed_df)} individual drawdowns to {output_path}")


def precompute_peer_group_drawdowns():
    """Step 19: Precompute peer group drawdowns for each GICS industry (MV and weighted versions)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from peer_group import (
        _calculate_peer_group_prices_mv_full,
        _calculate_peer_group_prices_weighted_full
    )
    from data_loader import get_r3000_files_hash
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()
    files_hash = get_r3000_files_hash()

    # Load all peer group prices (already grouped by GICS)
    print("  Loading MV peer group prices...")
    try:
        all_mv_prices = _calculate_peer_group_prices_mv_full(files_hash)
        unique_gics_mv = all_mv_prices['GICS'].unique()
        print(f"    Found {len(unique_gics_mv)} GICS groups")
    except Exception as e:
        print(f"    Error loading MV prices: {e}")
        all_mv_prices = pd.DataFrame()
        unique_gics_mv = []

    print("  Loading Weighted peer group prices...")
    try:
        all_weighted_prices = _calculate_peer_group_prices_weighted_full(files_hash)
        unique_gics_weighted = all_weighted_prices['GICS'].unique()
        print(f"    Found {len(unique_gics_weighted)} GICS groups")
    except Exception as e:
        print(f"    Error loading weighted prices: {e}")
        all_weighted_prices = pd.DataFrame()
        unique_gics_weighted = []

    # Calculate drawdowns for each GICS group - MV version
    all_mv_drawdowns = []
    if len(all_mv_prices) > 0:
        for gics in unique_gics_mv:
            try:
                gics_prices = all_mv_prices[all_mv_prices['GICS'] == gics].copy()
                if len(gics_prices) >= 30:
                    gics_prices_dd = gics_prices[['Date', 'Value']].copy()
                    gics_prices_dd = gics_prices_dd.rename(columns={'Value': 'Close'})
                    dd_data = calculate_drawdowns(gics_prices_dd, start_date=gics_prices_dd['Date'].min(), end_date=gics_prices_dd['Date'].max())
                    if len(dd_data) > 0:
                        dd_data['gics'] = gics
                        dd_data['version'] = 'mv'
                        all_mv_drawdowns.append(dd_data)
            except Exception as e:
                continue

    # Calculate drawdowns for each GICS group - Weighted version
    all_weighted_drawdowns = []
    if len(all_weighted_prices) > 0:
        for gics in unique_gics_weighted:
            try:
                gics_prices = all_weighted_prices[all_weighted_prices['GICS'] == gics].copy()
                if len(gics_prices) >= 30:
                    gics_prices_dd = gics_prices[['Date', 'Value']].copy()
                    gics_prices_dd = gics_prices_dd.rename(columns={'Value': 'Close'})
                    dd_data = calculate_drawdowns(gics_prices_dd, start_date=gics_prices_dd['Date'].min(), end_date=gics_prices_dd['Date'].max())
                    if len(dd_data) > 0:
                        dd_data['gics'] = gics
                        dd_data['version'] = 'weighted'
                        all_weighted_drawdowns.append(dd_data)
            except Exception as e:
                continue

    # Save MV drawdowns
    if all_mv_drawdowns:
        mv_df = pd.concat(all_mv_drawdowns, ignore_index=True)
        output_path = R3000_PRECOMPUTED_DIR / 'peer_group_drawdowns_mv.parquet'
        mv_df.to_parquet(output_path, index=False)
        print(f"    Saved {len(mv_df)} MV drawdown records for {len(all_mv_drawdowns)} GICS groups")

    # Save weighted drawdowns
    if all_weighted_drawdowns:
        weighted_df = pd.concat(all_weighted_drawdowns, ignore_index=True)
        output_path = R3000_PRECOMPUTED_DIR / 'peer_group_drawdowns_weighted.parquet'
        weighted_df.to_parquet(output_path, index=False)
        print(f"    Saved {len(weighted_df)} weighted drawdown records for {len(all_weighted_drawdowns)} GICS groups")


def precompute_iwv_total_mv_drawdowns():
    """Step 20: Precompute IWV Total Market Value drawdowns"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from peer_group import _calculate_iwv_total_market_value_full
    from data_loader import get_r3000_files_hash
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()
    files_hash = get_r3000_files_hash()

    # Get IWV total market value
    iwv_mv = _calculate_iwv_total_market_value_full(files_hash)

    if len(iwv_mv) >= 30:
        iwv_mv_dd = iwv_mv.copy()
        iwv_mv_dd = iwv_mv_dd.rename(columns={'Value': 'Close'})
        dd_data = calculate_drawdowns(iwv_mv_dd, start_date=iwv_mv_dd['Date'].min(), end_date=iwv_mv_dd['Date'].max())

        if len(dd_data) > 0:
            output_path = R3000_PRECOMPUTED_DIR / 'iwv_total_mv_drawdowns.parquet'
            dd_data.to_parquet(output_path, index=False)
            print(f"    Saved {len(dd_data)} IWV Total MV drawdowns")
    else:
        print("    Not enough IWV Total MV data")


def precompute_iwv_etf_drawdowns():
    """Step 21: Precompute IWV ETF price drawdowns"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()

    # Load IWV prices
    price_file = OUTPUT_DIR / 'IWV_prices.csv'
    if not price_file.exists():
        print(f"    IWV prices not found: {price_file}")
        return

    prices = pd.read_csv(price_file)
    prices['Date'] = pd.to_datetime(prices['Date'])

    if len(prices) >= 30:
        dd_data = calculate_drawdowns(prices, start_date=prices['Date'].min(), end_date=prices['Date'].max())

        if len(dd_data) > 0:
            output_path = R3000_PRECOMPUTED_DIR / 'iwv_etf_drawdowns.parquet'
            dd_data.to_parquet(output_path, index=False)
            print(f"    Saved {len(dd_data)} IWV ETF drawdowns")
    else:
        print("    Not enough IWV price data")


def precompute_ark_stock_drawdowns_full():
    """Step 17: Precompute full top 10 drawdowns for each stock in ARK ETFs"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from data_loader import load_ark_holdings, get_ark_files_hash
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()
    files_hash = get_ark_files_hash()

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)

        if len(holdings) == 0:
            continue

        # Filter out currency and money market funds
        if 'Bloomberg Name' in holdings.columns:
            currency_tickers = holdings[holdings['Bloomberg Name'].str.contains('curncy', case=False, na=False)]['Ticker'].unique()
            holdings = holdings[~holdings['Ticker'].isin(currency_tickers)]

        money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']
        holdings = holdings[~holdings['Ticker'].str.split().str[0].apply(
            lambda t: any(t.startswith(p) for p in money_market_prefixes)
        )]

        all_tickers = holdings['Ticker'].unique()
        all_drawdowns = []

        for ticker in all_tickers:
            stock_data = holdings[holdings['Ticker'] == ticker].copy()

            if len(stock_data) < 30:
                continue

            # Determine which price column to use
            if 'YFinance Close Price' in stock_data.columns and stock_data['YFinance Close Price'].notna().any():
                price_col = 'YFinance Close Price'
            else:
                price_col = 'Stock_Price'

            price_df = stock_data[['Date', price_col]].copy()
            price_df.columns = ['Date', 'Close']
            price_df = price_df.dropna()

            if len(price_df) < 30:
                continue

            # Calculate drawdowns for full date range
            dd_data = calculate_drawdowns(price_df, start_date=price_df['Date'].min(), end_date=price_df['Date'].max())

            if len(dd_data) == 0:
                continue

            # Add ticker info
            dd_data['ticker'] = ticker.split()[0] if ' ' in ticker else ticker
            dd_data['ticker_full'] = ticker
            all_drawdowns.append(dd_data)

        if all_drawdowns:
            result_df = pd.concat(all_drawdowns, ignore_index=True)
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_stock_drawdowns.parquet'
            result_df.to_parquet(output_path, index=False)
            print(f"    Saved {len(result_df)} drawdown records for {len(all_drawdowns)} stocks")


def precompute_r3000_stock_drawdowns_full_detailed():
    """Step 18: Precompute full top 10 drawdowns for each R3000 stock"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import load_r3000_holdings, get_r3000_files_hash
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()
    files_hash = get_r3000_files_hash()
    holdings = load_r3000_holdings(files_hash)

    if len(holdings) == 0:
        print("  No R3000 holdings data found")
        return

    all_tickers = holdings['Ticker'].unique()
    print(f"  Processing {len(all_tickers)} tickers...")

    all_drawdowns = []
    for i, ticker in enumerate(all_tickers):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{len(all_tickers)}")

        stock_data = holdings[holdings['Ticker'] == ticker].copy()

        if len(stock_data) < 30 or 'Price' not in stock_data.columns:
            continue

        price_df = stock_data[['Date', 'Price']].copy()
        price_df = price_df.rename(columns={'Price': 'Close'})
        price_df = price_df.dropna(subset=['Close'])

        if len(price_df) < 30:
            continue

        # Calculate drawdowns for full date range
        dd_data = calculate_drawdowns(price_df, start_date=price_df['Date'].min(), end_date=price_df['Date'].max())

        if len(dd_data) == 0:
            continue

        # Add ticker info
        ticker_clean = ticker.split()[0] if isinstance(ticker, str) else ticker
        dd_data['ticker'] = ticker_clean
        dd_data['ticker_full'] = ticker
        all_drawdowns.append(dd_data)

    if all_drawdowns:
        result_df = pd.concat(all_drawdowns, ignore_index=True)
        output_path = R3000_PRECOMPUTED_DIR / 'r3000_stock_drawdowns_detailed.parquet'
        result_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(result_df)} drawdown records for {len(all_drawdowns)} stocks")


def precompute_position_changes():
    """Step 16: Precompute position changes during top 10 drawdowns for each ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from data_loader import load_ark_holdings, get_ark_files_hash
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()
    files_hash = get_ark_files_hash()

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")

        # Load ETF prices to get drawdowns
        price_file = OUTPUT_DIR / f'{etf}_prices.csv'
        if not price_file.exists():
            print(f"    Price file not found: {price_file}")
            continue

        prices = pd.read_csv(price_file)
        prices['Date'] = pd.to_datetime(prices['Date'])

        # Calculate drawdowns for full date range
        dd_df = calculate_drawdowns(prices, start_date=prices['Date'].min(), end_date=prices['Date'].max())
        if len(dd_df) == 0:
            continue

        # Get top 10 historical drawdowns
        historical_dds = dd_df[dd_df['rank'] != 'Current'].head(10)

        # Load holdings
        holdings = load_ark_holdings(files_hash, etf)

        all_changes = []

        for _, dd_row in historical_dds.iterrows():
            peak_date = dd_row['peak_date']
            trough_date = dd_row['trough_date']
            dd_rank = dd_row['rank']

            # Get holdings at peak date (or closest date before)
            peak_holdings = holdings[holdings['Date'] <= peak_date].copy()
            if len(peak_holdings) == 0:
                continue
            peak_date_actual = peak_holdings['Date'].max()
            peak_holdings = peak_holdings[peak_holdings['Date'] == peak_date_actual].copy()

            # Get holdings at trough date (or closest date before)
            trough_holdings = holdings[holdings['Date'] <= trough_date].copy()
            if len(trough_holdings) == 0:
                continue
            trough_date_actual = trough_holdings['Date'].max()
            trough_holdings = trough_holdings[trough_holdings['Date'] == trough_date_actual].copy()

            # Filter out currency and money market
            for df in [peak_holdings, trough_holdings]:
                if 'Bloomberg Name' in df.columns:
                    mask = ~df['Bloomberg Name'].str.contains('curncy', case=False, na=False)
                    df.drop(df[~mask].index, inplace=True)

            money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX', 'DGCXX', 'MVRXX']
            for df in [peak_holdings, trough_holdings]:
                ticker_symbols = df['Ticker'].str.split().str[0]
                is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
                df.drop(df[is_mm].index, inplace=True)

            # Create comparison
            peak_positions = peak_holdings.set_index('Ticker')[['Weight', 'Position']].add_suffix('_peak')
            trough_positions = trough_holdings.set_index('Ticker')[['Weight', 'Position']].add_suffix('_trough')

            comparison = peak_positions.join(trough_positions, how='outer').fillna(0)
            comparison['Weight_Change'] = comparison['Weight_trough'] - comparison['Weight_peak']
            comparison['Position_Change'] = comparison['Position_trough'] - comparison['Position_peak']
            comparison['Position_Change_Pct'] = np.where(
                comparison['Position_peak'] > 0,
                (comparison['Position_trough'] - comparison['Position_peak']) / comparison['Position_peak'] * 100,
                np.where(comparison['Position_trough'] > 0, 100, 0)
            )

            # Categorize changes
            comparison['Status'] = 'Unchanged'
            comparison.loc[comparison['Position_peak'] == 0, 'Status'] = 'New Position'
            comparison.loc[comparison['Position_trough'] == 0, 'Status'] = 'Exited'
            comparison.loc[(comparison['Position_Change'] > 0) & (comparison['Position_peak'] > 0), 'Status'] = 'Added'
            comparison.loc[(comparison['Position_Change'] < 0) & (comparison['Position_trough'] > 0), 'Status'] = 'Reduced'

            comparison = comparison.reset_index()
            comparison['Ticker_Clean'] = comparison['Ticker'].str.split().str[0]
            comparison['dd_rank'] = dd_rank
            comparison['peak_date'] = peak_date
            comparison['trough_date'] = trough_date
            comparison['peak_date_actual'] = peak_date_actual
            comparison['trough_date_actual'] = trough_date_actual
            comparison['depth_pct'] = dd_row['depth_pct']

            all_changes.append(comparison)

        if all_changes:
            result_df = pd.concat(all_changes, ignore_index=True)
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_position_changes.parquet'
            result_df.to_parquet(output_path, index=False)
            print(f"    Saved {len(result_df)} position change records")


def precompute_sp500_correlations():
    """Precompute S&P 500 Top 50 correlation matrices per analysis period"""
    from config import ANALYSIS_PERIODS

    ensure_dirs()

    # Load S&P 500 Top 50 prices
    sp500_file = OUTPUT_DIR / 'SP500_top50_prices.csv'
    if not sp500_file.exists():
        print("  S&P 500 Top 50 prices not found. Run fetch_sp500_prices.py first.")
        return

    prices = pd.read_csv(sp500_file)
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.set_index('Date')

    lookback_periods = [60, 120, 250]

    for period_key, period_config in ANALYSIS_PERIODS.items():
        period_end = period_config["end"]
        # Use prices up to the period end date
        prices_period = prices[prices.index <= period_end]
        if len(prices_period) == 0:
            print(f"  No price data for period {period_key}")
            continue

        for lookback_days in lookback_periods:
            latest_date = prices_period.index.max()
            lookback_start = latest_date - pd.Timedelta(days=lookback_days)

            # Filter to lookback period
            prices_lookback = prices_period[prices_period.index >= lookback_start].copy()

            if len(prices_lookback.columns) < 2:
                continue

            # Calculate returns and correlation
            # Use min_periods=20 to require at least 20 overlapping days
            returns = prices_lookback.pct_change().iloc[1:]
            corr_matrix = returns.corr(min_periods=20)

            # Remove tickers with no valid correlations (all NaN off-diagonal)
            has_valid = (corr_matrix.notna().sum() > 1)
            corr_matrix = corr_matrix.loc[has_valid, has_valid]

            # Save correlation matrix with period key in filename
            output_path = ARK_PRECOMPUTED_DIR / f'SP500_top50_correlation_matrix_{period_key}_{lookback_days}d.parquet'
            corr_matrix.to_parquet(output_path)
            print(f"    Saved {output_path.name} ({len(corr_matrix)} tickers)")


def precompute_stress_correlations():
    """Precompute stress correlations (correlations during drawdowns) for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from data_loader import load_ark_holdings, get_ark_files_hash
    from precomputed_loader import load_etf_drawdowns

    ensure_dirs()
    files_hash = get_ark_files_hash()

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")

        # Load drawdowns
        drawdowns = load_etf_drawdowns(etf)
        if len(drawdowns) == 0:
            print(f"    No drawdowns found for {etf}")
            continue

        # Filter to historical drawdowns only (exclude Current)
        historical_dds = drawdowns[drawdowns['rank'] != 'Current'].head(10)

        if len(historical_dds) == 0:
            print(f"    No historical drawdowns for {etf}")
            continue

        # Load holdings
        holdings = load_ark_holdings(files_hash, etf)
        holdings_filtered = _filter_non_stocks(holdings)

        if len(holdings_filtered) == 0:
            continue

        stress_results = []

        for _, dd_row in historical_dds.iterrows():
            dd_rank = dd_row['rank']
            peak_date = dd_row['peak_date']
            trough_date = dd_row['trough_date']
            depth_pct = dd_row['depth_pct']

            # Filter holdings to drawdown period
            dd_holdings = holdings_filtered[
                (holdings_filtered['Date'] >= peak_date) &
                (holdings_filtered['Date'] <= trough_date)
            ].copy()

            if len(dd_holdings) == 0:
                continue

            # Pivot to get price and weight matrices
            price_matrix = dd_holdings.pivot_table(
                index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
            )
            weight_matrix = dd_holdings.pivot_table(
                index='Date', columns='Ticker', values='Weight', aggfunc='first'
            )

            # Need at least 5 days of data
            if len(price_matrix) < 5:
                continue

            if len(price_matrix.columns) < 2:
                continue

            # Calculate returns and correlation
            # Use iloc[1:] to skip first row (NaN from pct_change)
            returns = price_matrix.pct_change(fill_method=None).iloc[1:]
            if len(returns) < 3:
                continue

            # corr(min_periods=20) requires at least 20 overlapping days for each pair
            corr_matrix = returns.corr(min_periods=20)

            # Get upper triangle correlations
            n = len(corr_matrix.columns)
            triu_i, triu_j = np.triu_indices(n, k=1)
            corr_values = corr_matrix.values[triu_i, triu_j]
            valid_mask = ~np.isnan(corr_values)
            valid_corrs = corr_values[valid_mask]

            if len(valid_corrs) == 0:
                continue

            # Calculate weighted correlation using average daily pair weights over the drawdown period
            pair_weights = calculate_average_pair_weights(weight_matrix, corr_matrix.columns.tolist())

            valid_weight_mask = valid_mask & (pair_weights > 0)
            if valid_weight_mask.any():
                weighted_mean = np.average(corr_values[valid_weight_mask], weights=pair_weights[valid_weight_mask])
            else:
                weighted_mean = np.mean(valid_corrs)

            # Calculate stress correlation stats
            stress_results.append({
                'dd_rank': dd_rank,
                'peak_date': peak_date,
                'trough_date': trough_date,
                'depth_pct': depth_pct,
                'duration_days': (trough_date - peak_date).days,
                'num_tickers': len(corr_matrix.columns),
                'num_pairs': len(valid_corrs),
                'mean_corr': np.mean(valid_corrs),
                'weighted_mean_corr': weighted_mean,
                'median_corr': np.median(valid_corrs),
                'min_corr': np.min(valid_corrs),
                'max_corr': np.max(valid_corrs),
                'std_corr': np.std(valid_corrs)
            })

        if stress_results:
            stress_df = pd.DataFrame(stress_results)
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_stress_correlations.parquet'
            stress_df.to_parquet(output_path, index=False)
            print(f"    Saved {len(stress_df)} drawdown periods")
        else:
            print(f"    No stress correlations computed for {etf}")


def generate_metadata():
    """Generate metadata.json with version, timestamps, and hashes"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from data_loader import get_ark_files_hash, get_r3000_files_hash
    from config import ANALYSIS_PERIODS

    ensure_dirs()

    metadata = {
        'version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'source_hashes': {
            'ark_files_hash': get_ark_files_hash(),
            'r3000_files_hash': get_r3000_files_hash()
        },
        'analysis_periods': list(ANALYSIS_PERIODS.keys()),
        'parameters': {
            'lookback_days': [60, 120, 250],
            'rolling_windows': [20, 30, 60, 120],  # Independent rolling windows for correlation time series
            'min_depth_pct': 10,
            'min_duration_days': 7
        }
    }

    output_path = METADATA_DIR / 'metadata.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata to {output_path}")


def precompute_conviction_drawdowns():
    """Step 24: Precompute conviction vs drawdown data for each ARK ETF

    Uses yfinance-downloaded prices (from fetch_ark_holdings_prices.py)
    and holdings weights to classify drawdowns by conviction level.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS, ANALYSIS_PERIODS
    from data_loader import load_ark_holdings, get_ark_files_hash
    from drawdown_calculator import calculate_drawdowns

    ensure_dirs()

    # Load yfinance prices
    prices_path = OUTPUT_DIR / 'ark_holdings_prices.parquet'
    if not prices_path.exists():
        print(f"    Prices file not found: {prices_path}")
        print("    Run: python src/fetch_ark_holdings_prices.py")
        return

    prices_wide = pd.read_parquet(prices_path)
    prices_wide['Date'] = pd.to_datetime(prices_wide['Date'])

    # Get full date range across all periods
    all_starts = [p["start"] for p in ANALYSIS_PERIODS.values()]
    all_ends = [p["end"] for p in ANALYSIS_PERIODS.values()]
    global_start = min(all_starts)
    global_end = max(all_ends)

    files_hash = get_ark_files_hash()

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)
        if len(holdings) == 0:
            print(f"    No holdings data for {etf}")
            continue

        # Filter non-stocks
        holdings_filtered = _filter_non_stocks(holdings)

        # Get unique tickers (clean names)
        ticker_map = {}  # clean -> full
        for t in holdings_filtered['Ticker'].unique():
            clean = t.split()[0] if pd.notna(t) and ' ' in t else t
            if pd.notna(clean):
                ticker_map[clean] = t

        all_rows = []

        for clean_ticker, full_ticker in ticker_map.items():
            # Check if we have price data for this ticker
            if clean_ticker not in prices_wide.columns:
                continue

            # Build price DataFrame
            price_df = prices_wide[['Date', clean_ticker]].copy()
            price_df.columns = ['Date', 'Close']
            price_df = price_df.dropna()

            if len(price_df) < 30:
                continue

            # Calculate drawdowns for full date range
            dd_data = calculate_drawdowns(
                price_df, start_date=global_start, end_date=global_end
            )

            if len(dd_data) == 0:
                continue

            # Get holdings weight data for this ticker
            ticker_holdings = holdings_filtered[
                holdings_filtered['Ticker'] == full_ticker
            ][['Date', 'Weight']].copy()
            ticker_holdings = ticker_holdings.sort_values('Date')

            if len(ticker_holdings) == 0:
                continue

            # Process each drawdown (exclude Current, require >= 7 calendar days)
            for _, dd_row in dd_data.iterrows():
                if dd_row['rank'] == 'Current':
                    continue

                peak_date = dd_row['peak_date']
                trough_date = dd_row['trough_date']

                # Skip short drawdowns (less than 1 week)
                if (trough_date - peak_date).days < 7:
                    continue
                peak_price = dd_row['peak_price']
                depth_pct = dd_row['depth_pct']

                # Find weight at peak_date: closest holdings date <= peak_date
                holdings_before_peak = ticker_holdings[
                    ticker_holdings['Date'] <= peak_date
                ]
                if len(holdings_before_peak) == 0:
                    weight_at_peak = 0.0
                else:
                    weight_at_peak = holdings_before_peak.iloc[-1]['Weight']

                # Classify conviction (weights are in decimal form, e.g., 0.05 = 5%)
                if weight_at_peak >= 0.05:
                    conviction = 'High'
                elif weight_at_peak >= 0.01:
                    conviction = 'Mid'
                else:
                    conviction = 'Low'

                # Duration in calendar days
                duration_days = (trough_date - peak_date).days

                # Check recovery: did price reach peak_price after trough?
                future_prices = price_df[price_df['Date'] > trough_date]
                recovered = False
                recovery_date = None
                days_to_recover = None

                if len(future_prices) > 0:
                    recovery_prices = future_prices[future_prices['Close'] >= peak_price]
                    if len(recovery_prices) > 0:
                        recovered = True
                        recovery_date = recovery_prices.iloc[0]['Date']
                        days_to_recover = (recovery_date - trough_date).days

                all_rows.append({
                    'etf': etf,
                    'ticker': clean_ticker,
                    'conviction': conviction,
                    'weight_at_peak': round(weight_at_peak * 100, 2),
                    'peak_date': peak_date,
                    'trough_date': trough_date,
                    'peak_price': peak_price,
                    'trough_price': dd_row['trough_price'],
                    'depth_pct': depth_pct,
                    'duration_days': duration_days,
                    'recovered': recovered,
                    'recovery_date': recovery_date,
                    'days_to_recover': days_to_recover,
                })

        if all_rows:
            result_df = pd.DataFrame(all_rows)
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_conviction_drawdowns.parquet'
            result_df.to_parquet(output_path, index=False)
            print(f"    Saved {len(result_df)} conviction drawdown records "
                  f"({len(result_df[result_df['conviction']=='High'])} High, "
                  f"{len(result_df[result_df['conviction']=='Mid'])} Mid, "
                  f"{len(result_df[result_df['conviction']=='Low'])} Low)")
        else:
            print(f"    No conviction drawdown data for {etf}")


STEPS = {
    1: ("ARK ETFs - Excel to Parquet", convert_ark_etfs),
    2: ("Russell 3000 - Excel to Parquet", convert_russell_3000),
    3: ("Peer Group Cache", precompute_peer_group_cache),
    4: ("R3000 Drawdowns (legacy)", precompute_r3000_drawdowns),
    5: ("ARK Stock Drawdowns", precompute_ark_drawdowns),
    6: ("ARK Holdings Max Drawdowns", precompute_ark_holdings_max_drawdowns),
    7: ("ETF-level Drawdowns", precompute_etf_drawdowns),
    8: ("HHI Time Series", precompute_hhi_timeseries),
    9: ("Correlation Matrices", precompute_correlation_matrices),
    10: ("Weighted Correlation Matrices", precompute_weighted_correlations),
    11: ("Rolling Correlations", precompute_rolling_correlations),
    12: ("Holdings Drawdowns", precompute_holdings_drawdowns),
    13: ("Concentration Performance", precompute_concentration_performance),
    14: ("R3000 Drawdowns Full", precompute_r3000_drawdowns_full),
    15: ("Position Changes", precompute_position_changes),
    16: ("ARK Stock Drawdowns (full)", precompute_ark_stock_drawdowns_full),
    17: ("R3000 Stock Drawdowns (full)", precompute_r3000_stock_drawdowns_full_detailed),
    18: ("Peer Group Drawdowns", precompute_peer_group_drawdowns),
    19: ("IWV Total MV Drawdowns", precompute_iwv_total_mv_drawdowns),
    20: ("IWV ETF Drawdowns", precompute_iwv_etf_drawdowns),
    21: ("S&P 500 Top 50 Correlations", precompute_sp500_correlations),
    22: ("Stress Correlations", precompute_stress_correlations),
    23: ("Generate Metadata", generate_metadata),
    24: ("Conviction vs Drawdown", precompute_conviction_drawdowns),
}

# Step groups for convenience
STEP_GROUPS = {
    'convert': [1, 2],
    'correlations': [9, 10, 11, 21, 22],
    'drawdowns': [4, 5, 6, 7, 14, 16, 17, 18, 19, 20, 24],
    'ark': [1, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 22, 24],
    'r3000': [2, 3, 4, 14, 17, 18, 19, 20],
}


def run_step(step_num):
    """Run a single step by number"""
    if step_num not in STEPS:
        print(f"Invalid step: {step_num}")
        return
    name, func = STEPS[step_num]
    print(f"Step {step_num}/{len(STEPS)}: {name}")
    func()
    print()


def run_steps(step_nums):
    """Run multiple steps"""
    for step_num in step_nums:
        run_step(step_num)


def run_all():
    """Run all steps"""
    print("=" * 60)
    print("Converting Excel files to Parquet & Precomputing All Data")
    print("=" * 60)
    print()

    for step_num in sorted(STEPS.keys()):
        run_step(step_num)

    print("=" * 60)
    print("Done! All data precomputed.")
    print("=" * 60)


def print_help():
    """Print usage help"""
    print("Usage: python convert_to_parquet.py [options]")
    print()
    print("Options:")
    print("  (no args)           Run all steps")
    print("  --step N            Run step N only (e.g., --step 11)")
    print("  --steps N,M,O       Run steps N, M, O (e.g., --steps 9,10,11)")
    print("  --group NAME        Run a predefined group of steps")
    print("  --list              List all steps")
    print("  --help              Show this help")
    print()
    print("Step Groups:")
    for group, steps in STEP_GROUPS.items():
        print(f"  {group:15} Steps {steps}")
    print()
    print("Examples:")
    print("  python convert_to_parquet.py --step 22           # Run stress correlations only")
    print("  python convert_to_parquet.py --steps 9,10,11,22  # Run correlation-related steps")
    print("  python convert_to_parquet.py --group correlations")


def list_steps():
    """List all available steps"""
    print("Available steps:")
    print()
    for step_num, (name, _) in sorted(STEPS.items()):
        print(f"  {step_num:2}: {name}")


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if not args:
        run_all()
    elif '--help' in args or '-h' in args:
        print_help()
    elif '--list' in args:
        list_steps()
    elif '--step' in args:
        idx = args.index('--step')
        if idx + 1 < len(args):
            step_num = int(args[idx + 1])
            ensure_dirs()
            run_step(step_num)
        else:
            print("Error: --step requires a step number")
    elif '--steps' in args:
        idx = args.index('--steps')
        if idx + 1 < len(args):
            step_nums = [int(x.strip()) for x in args[idx + 1].split(',')]
            ensure_dirs()
            run_steps(step_nums)
        else:
            print("Error: --steps requires step numbers (comma-separated)")
    elif '--group' in args:
        idx = args.index('--group')
        if idx + 1 < len(args):
            group_name = args[idx + 1]
            if group_name in STEP_GROUPS:
                ensure_dirs()
                print(f"Running group '{group_name}': steps {STEP_GROUPS[group_name]}")
                print()
                run_steps(STEP_GROUPS[group_name])
            else:
                print(f"Unknown group: {group_name}")
                print(f"Available groups: {list(STEP_GROUPS.keys())}")
        else:
            print("Error: --group requires a group name")
    else:
        print(f"Unknown argument: {args[0]}")
        print("Use --help for usage information")
