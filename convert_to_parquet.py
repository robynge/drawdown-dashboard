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

        money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
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
    money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
    ticker_symbols = result['Ticker'].str.split().str[0]
    is_mm = ticker_symbols.apply(lambda x: any(x.startswith(p) for p in money_market_prefixes) if pd.notna(x) else False)
    result = result[~is_mm]

    return result


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
    """Step 8: Precompute correlation matrices for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
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

        latest_date = holdings_filtered['Date'].max()
        current_tickers = holdings_filtered[holdings_filtered['Date'] == latest_date]['Ticker'].unique()

        for lookback_days in lookback_periods:
            lookback_start = latest_date - pd.Timedelta(days=lookback_days)

            # Filter to lookback period and current tickers
            holdings_lookback = holdings_filtered[
                (holdings_filtered['Date'] >= lookback_start) &
                (holdings_filtered['Ticker'].isin(current_tickers))
            ].copy()

            # Pivot to get price matrix
            price_matrix = holdings_lookback.pivot_table(
                index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
            )

            # Drop tickers with too many missing values
            min_data_points = len(price_matrix) * 0.5
            price_matrix = price_matrix.dropna(axis=1, thresh=int(min_data_points))

            if len(price_matrix.columns) < 2:
                continue

            # Calculate returns and correlation
            returns = price_matrix.pct_change().dropna()
            corr_matrix = returns.corr()

            # Save correlation matrix
            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_correlation_matrix_{lookback_days}d.parquet'
            corr_matrix.to_parquet(output_path)

            # Save returns for later use (rolling correlations need them)
            returns_reset = returns.reset_index()
            returns_path = ARK_PRECOMPUTED_DIR / f'{etf}_returns_{lookback_days}d.parquet'
            returns_reset.to_parquet(returns_path, index=False)

            # Save current weights
            current_weights = holdings_lookback[holdings_lookback['Date'] == latest_date][['Ticker', 'Weight']].copy()
            current_weights = current_weights[current_weights['Ticker'].isin(corr_matrix.columns)]
            weights_path = ARK_PRECOMPUTED_DIR / f'{etf}_current_weights.parquet'
            current_weights.to_parquet(weights_path, index=False)

            print(f"    Saved {lookback_days}d correlation matrix ({len(corr_matrix)}x{len(corr_matrix)})")


def precompute_weighted_correlations():
    """Step 9: Precompute weighted correlation matrices for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
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

        latest_date = holdings_filtered['Date'].max()
        current_tickers = holdings_filtered[holdings_filtered['Date'] == latest_date]['Ticker'].unique()

        for lookback_days in lookback_periods:
            lookback_start = latest_date - pd.Timedelta(days=lookback_days)

            holdings_lookback = holdings_filtered[
                (holdings_filtered['Date'] >= lookback_start) &
                (holdings_filtered['Ticker'].isin(current_tickers))
            ].copy()

            # Pivot for price and weight matrices
            price_matrix = holdings_lookback.pivot_table(
                index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
            )
            weight_matrix = holdings_lookback.pivot_table(
                index='Date', columns='Ticker', values='Weight', aggfunc='first'
            )

            # Drop tickers with too many missing values
            min_data_points = len(price_matrix) * 0.5
            valid_tickers = price_matrix.dropna(axis=1, thresh=int(min_data_points)).columns
            price_matrix = price_matrix[valid_tickers]
            weight_matrix = weight_matrix[valid_tickers]

            if len(valid_tickers) < 2:
                continue

            # Calculate returns
            returns = price_matrix.pct_change().dropna(how='all')
            weight_matrix = weight_matrix.ffill()
            weight_matrix = weight_matrix.loc[returns.index]

            tickers = returns.columns.tolist()
            n = len(tickers)

            # Initialize weighted correlation matrix
            weighted_corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)

            # Calculate weighted correlation for each pair
            for i in range(n):
                for j in range(i + 1, n):
                    ticker_a, ticker_b = tickers[i], tickers[j]

                    R_A = returns[ticker_a].values
                    R_B = returns[ticker_b].values
                    W_A = weight_matrix[ticker_a].values if ticker_a in weight_matrix.columns else np.zeros(len(R_A))
                    W_B = weight_matrix[ticker_b].values if ticker_b in weight_matrix.columns else np.zeros(len(R_B))

                    W_t = W_A * W_B
                    mask = (~np.isnan(R_A)) & (~np.isnan(R_B)) & (W_t > 0)

                    if mask.sum() < 2:
                        valid_mask = (~np.isnan(R_A)) & (~np.isnan(R_B))
                        if valid_mask.sum() > 1:
                            corr_val = np.corrcoef(R_A[valid_mask], R_B[valid_mask])[0, 1]
                        else:
                            corr_val = np.nan
                    else:
                        w = W_t[mask]
                        w = w / w.sum()
                        a = R_A[mask]
                        b = R_B[mask]

                        mu_a = np.sum(w * a)
                        mu_b = np.sum(w * b)
                        da = a - mu_a
                        db = b - mu_b
                        cov_ab = np.sum(w * da * db)
                        var_a = np.sum(w * da * da)
                        var_b = np.sum(w * db * db)

                        if var_a > 0 and var_b > 0:
                            corr_val = cov_ab / np.sqrt(var_a * var_b)
                            corr_val = np.clip(corr_val, -1, 1)
                        else:
                            corr_val = np.nan

                    weighted_corr.iloc[i, j] = corr_val
                    weighted_corr.iloc[j, i] = corr_val

            output_path = ARK_PRECOMPUTED_DIR / f'{etf}_weighted_correlation_matrix_{lookback_days}d.parquet'
            weighted_corr.to_parquet(output_path)
            print(f"    Saved {lookback_days}d weighted correlation matrix ({n}x{n})")


def precompute_rolling_correlations():
    """Step 10: Precompute rolling correlations for each ARK ETF"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    from config import ARK_ETFS
    from data_loader import load_ark_holdings, get_ark_files_hash

    ensure_dirs()
    files_hash = get_ark_files_hash()

    lookback_periods = [60, 120, 250]
    rolling_windows = [20, 30]

    for etf in ARK_ETFS:
        print(f"  Processing {etf}...")
        holdings = load_ark_holdings(files_hash, etf)
        holdings_filtered = _filter_non_stocks(holdings)

        if len(holdings_filtered) == 0:
            continue

        latest_date = holdings_filtered['Date'].max()
        current_tickers = holdings_filtered[holdings_filtered['Date'] == latest_date]['Ticker'].unique()

        for lookback_days in lookback_periods:
            lookback_start = latest_date - pd.Timedelta(days=lookback_days)

            holdings_lookback = holdings_filtered[
                (holdings_filtered['Date'] >= lookback_start) &
                (holdings_filtered['Ticker'].isin(current_tickers))
            ].copy()

            price_matrix = holdings_lookback.pivot_table(
                index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
            )
            weight_matrix = holdings_lookback.pivot_table(
                index='Date', columns='Ticker', values='Weight', aggfunc='first'
            )

            min_data_points = len(price_matrix) * 0.5
            valid_tickers = price_matrix.dropna(axis=1, thresh=int(min_data_points)).columns
            price_matrix = price_matrix[valid_tickers]
            weight_matrix = weight_matrix[valid_tickers]

            returns = price_matrix.pct_change().dropna()
            weight_matrix = weight_matrix.ffill()

            if len(returns) < 20:
                continue

            n_tickers = len(returns.columns)
            triu_i, triu_j = np.triu_indices(n_tickers, k=1)
            holdings_dates = np.sort(holdings_lookback['Date'].unique())

            for rolling_window in rolling_windows:
                if len(returns) < rolling_window:
                    continue

                results = []
                dates = returns.index

                for i in range(rolling_window, len(returns) + 1):
                    window_returns = returns.iloc[i - rolling_window:i]
                    current_date = dates[i - 1]
                    corr_matrix = window_returns.corr()

                    corr_values = corr_matrix.values[triu_i, triu_j]
                    valid_mask = ~np.isnan(corr_values)
                    valid_corrs = corr_values[valid_mask]

                    if len(valid_corrs) == 0:
                        continue

                    mean_corr = np.mean(valid_corrs)
                    median_corr = np.median(valid_corrs)

                    # Weighted correlation
                    valid_dates = holdings_dates[holdings_dates <= current_date]
                    if len(valid_dates) == 0:
                        weighted_mean = mean_corr
                    else:
                        closest_date = valid_dates[-1]
                        weights_df = holdings_lookback[holdings_lookback['Date'] == closest_date][['Ticker', 'Weight']]
                        weights_df = weights_df[weights_df['Ticker'].isin(corr_matrix.columns)]

                        if len(weights_df) > 0:
                            weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))
                            weights_arr = np.array([weights_dict.get(t, 0) for t in corr_matrix.columns])
                            weight_mat = np.outer(weights_arr, weights_arr)
                            pair_weights = weight_mat[triu_i, triu_j]
                            valid_weight_mask = valid_mask & (pair_weights > 0)
                            if valid_weight_mask.any():
                                weighted_mean = np.average(corr_values[valid_weight_mask], weights=pair_weights[valid_weight_mask])
                            else:
                                weighted_mean = mean_corr
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
                    output_path = ARK_PRECOMPUTED_DIR / f'{etf}_rolling_correlations_{lookback_days}d_{rolling_window}w.parquet'
                    rolling_df.to_parquet(output_path, index=False)
                    print(f"    Saved {lookback_days}d/{rolling_window}w rolling correlations ({len(rolling_df)} records)")


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

        money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
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

            money_market_prefixes = ['FTOXX', 'FIRXX', 'FEDXX', 'FDRXX', 'SPRXX']
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
    """Precompute S&P 500 Top 50 correlation matrices"""
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

    for lookback_days in lookback_periods:
        print(f"  Processing {lookback_days}d lookback...")

        latest_date = prices.index.max()
        lookback_start = latest_date - pd.Timedelta(days=lookback_days)

        # Filter to lookback period
        prices_lookback = prices[prices.index >= lookback_start].copy()

        # Drop tickers with too many missing values
        min_data_points = len(prices_lookback) * 0.5
        prices_lookback = prices_lookback.dropna(axis=1, thresh=int(min_data_points))

        if len(prices_lookback.columns) < 2:
            continue

        # Calculate returns and correlation
        returns = prices_lookback.pct_change().dropna()
        corr_matrix = returns.corr()

        # Save correlation matrix
        output_path = ARK_PRECOMPUTED_DIR / f'SP500_top50_correlation_matrix_{lookback_days}d.parquet'
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

            # Pivot to get price matrix
            price_matrix = dd_holdings.pivot_table(
                index='Date', columns='Ticker', values='Stock_Price', aggfunc='first'
            )

            # Need at least 5 days of data
            if len(price_matrix) < 5:
                continue

            # Drop tickers with too many missing values (>50% missing)
            min_data_points = len(price_matrix) * 0.5
            price_matrix = price_matrix.dropna(axis=1, thresh=int(min_data_points))

            if len(price_matrix.columns) < 2:
                continue

            # Calculate returns and correlation
            returns = price_matrix.pct_change().dropna()
            if len(returns) < 3:
                continue

            corr_matrix = returns.corr()

            # Get upper triangle correlations
            n = len(corr_matrix.columns)
            triu_i, triu_j = np.triu_indices(n, k=1)
            corr_values = corr_matrix.values[triu_i, triu_j]
            valid_corrs = corr_values[~np.isnan(corr_values)]

            if len(valid_corrs) == 0:
                continue

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
            'rolling_windows': [20, 30],
            'min_depth_pct': 10,
            'min_duration_days': 7
        }
    }

    output_path = METADATA_DIR / 'metadata.json'
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata to {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Converting Excel files to Parquet & Precomputing All Data")
    print("=" * 60)
    print()

    print("Step 1/20: ARK ETFs - Excel to Parquet")
    convert_ark_etfs()
    print()

    print("Step 2/20: Russell 3000 - Excel to Parquet")
    convert_russell_3000()
    print()

    print("Step 3/20: Peer Group Cache (full time series)")
    precompute_peer_group_cache()
    print()

    print("Step 4/20: R3000 Drawdowns (legacy cache)")
    precompute_r3000_drawdowns()
    print()

    print("Step 5/20: ARK Stock Drawdowns")
    precompute_ark_drawdowns()
    print()

    print("Step 6/20: ARK Holdings Max Drawdowns")
    precompute_ark_holdings_max_drawdowns()
    print()

    print("Step 7/20: ETF-level Drawdowns")
    precompute_etf_drawdowns()
    print()

    print("Step 8/20: HHI Time Series")
    precompute_hhi_timeseries()
    print()

    print("Step 9/20: Correlation Matrices")
    precompute_correlation_matrices()
    print()

    print("Step 10/20: Weighted Correlation Matrices")
    precompute_weighted_correlations()
    print()

    print("Step 11/20: Rolling Correlations")
    precompute_rolling_correlations()
    print()

    print("Step 12/20: Holdings Drawdowns")
    precompute_holdings_drawdowns()
    print()

    print("Step 13/20: Concentration Performance")
    precompute_concentration_performance()
    print()

    print("Step 14/20: R3000 Drawdowns Full (precomputed)")
    precompute_r3000_drawdowns_full()
    print()

    print("Step 15/20: Position Changes")
    precompute_position_changes()
    print()

    print("Step 16/20: ARK Stock Drawdowns (full details)")
    precompute_ark_stock_drawdowns_full()
    print()

    print("Step 17/20: R3000 Stock Drawdowns (full details)")
    precompute_r3000_stock_drawdowns_full_detailed()
    print()

    print("Step 18/20: Peer Group Drawdowns")
    precompute_peer_group_drawdowns()
    print()

    print("Step 19/22: IWV Total MV Drawdowns")
    precompute_iwv_total_mv_drawdowns()
    print()

    print("Step 20/22: S&P 500 Top 50 Correlations")
    precompute_sp500_correlations()
    print()

    print("Step 21/22: Stress Correlations")
    precompute_stress_correlations()
    print()

    print("Step 22/22: Generate Metadata")
    generate_metadata()
    print()

    print("=" * 60)
    print("Done! All data precomputed.")
    print("=" * 60)
