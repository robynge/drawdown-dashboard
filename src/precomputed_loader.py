"""Precomputed data loader module

Provides unified interface to load precomputed data for Streamlit pages.
All heavy computation is moved to convert_to_parquet.py, this module
only handles reading parquet files and filtering by period.
"""
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

# Ensure src directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / 'input'
ARK_PRECOMPUTED_DIR = INPUT_DIR / 'ark_etfs' / 'precomputed'
R3000_PRECOMPUTED_DIR = INPUT_DIR / 'russell_3000' / 'precomputed'
METADATA_PATH = INPUT_DIR / 'precomputed' / 'metadata.json'


def _ensure_datetime(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """Ensure date columns are datetime type"""
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def load_etf_drawdowns(etf: str) -> pd.DataFrame:
    """Load precomputed ETF-level drawdowns

    Args:
        etf: ETF name (e.g., 'ARKK')

    Returns:
        DataFrame with drawdown data (peak_date, trough_date, depth_pct, etc.)
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_etf_drawdowns.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date', 'recovery_date'])


def load_hhi_timeseries(etf: str) -> pd.DataFrame:
    """Load precomputed HHI time series for an ETF

    Args:
        etf: ETF name (e.g., 'ARKK')

    Returns:
        DataFrame with columns: Date, HHI, Effective_Positions, Top5_Concentration, Num_Positions
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_hhi_timeseries.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['Date'])


def load_correlation_matrix(etf: str, period_key: str, lookback_days: int = 250) -> pd.DataFrame:
    """Load precomputed correlation matrix for an ETF and analysis period

    Args:
        etf: ETF name (e.g., 'ARKK')
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')
        lookback_days: Lookback period in days (60, 120, or 250)

    Returns:
        DataFrame with correlation matrix (tickers as index and columns)
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_correlation_matrix_{period_key}_{lookback_days}d.parquet'
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_weighted_correlation_matrix(etf: str, period_key: str, lookback_days: int = 250) -> pd.DataFrame:
    """Load precomputed weighted correlation matrix for an ETF and analysis period

    Args:
        etf: ETF name (e.g., 'ARKK')
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')
        lookback_days: Lookback period in days (60, 120, or 250)

    Returns:
        DataFrame with weighted correlation matrix (tickers as index and columns)
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_weighted_correlation_matrix_{period_key}_{lookback_days}d.parquet'
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_rolling_correlations(etf: str, period_key: str, rolling_window: int = 60) -> pd.DataFrame:
    """Load precomputed rolling correlations for an ETF and analysis period

    Args:
        etf: ETF name (e.g., 'ARKK')
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')
        rolling_window: Rolling window in days (20, 30, 60, or 120)

    Returns:
        DataFrame with columns: Date, mean_corr, median_corr, weighted_mean_corr
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_rolling_correlations_{period_key}_{rolling_window}d.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['Date'])


def load_holdings_drawdowns(etf: str) -> pd.DataFrame:
    """Load precomputed holdings drawdowns for an ETF

    Args:
        etf: ETF name (e.g., 'ARKK')

    Returns:
        DataFrame with drawdown data for all constituent stocks
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_holdings_drawdowns.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date', 'recovery_date'])


def load_concentration_performance(etf: str) -> pd.DataFrame:
    """Load precomputed concentration vs performance data for an ETF

    Args:
        etf: ETF name (e.g., 'ARKK')

    Returns:
        DataFrame with merged price, HHI, and spread data
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_concentration_performance.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['Date'])


def load_r3000_drawdowns() -> pd.DataFrame:
    """Load precomputed R3000 drawdowns summary for all stocks

    Returns:
        DataFrame with columns: ticker, max_drawdown, num_drawdowns, gics_industry_group
    """
    path = R3000_PRECOMPUTED_DIR / 'r3000_drawdowns_full.parquet'
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_r3000_drawdowns_with_dates() -> pd.DataFrame:
    """Load precomputed R3000 drawdowns with date information for period filtering

    Returns:
        DataFrame with columns: ticker, ticker_full, gics_industry_group, rank,
                               peak_date, trough_date, depth_pct, duration_days
    """
    path = R3000_PRECOMPUTED_DIR / 'r3000_drawdowns_with_dates.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date'])


def load_iwv_etf_drawdowns() -> pd.DataFrame:
    """Load precomputed IWV ETF drawdowns

    Returns:
        DataFrame with IWV ETF drawdown data
    """
    path = R3000_PRECOMPUTED_DIR / 'iwv_etf_drawdowns.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date', 'recovery_date'])


def load_correlation_returns(etf: str, period_key: str, lookback_days: int = 250) -> pd.DataFrame:
    """Load precomputed returns data used for correlation calculation

    Args:
        etf: ETF name (e.g., 'ARKK')
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')
        lookback_days: Lookback period in days

    Returns:
        DataFrame with daily returns for each holding
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_returns_{period_key}_{lookback_days}d.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    return df


def load_current_weights(etf: str, period_key: str) -> pd.DataFrame:
    """Load precomputed current weights for an ETF at the end of an analysis period

    Args:
        etf: ETF name (e.g., 'ARKK')
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')

    Returns:
        DataFrame with columns: Ticker, Weight
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_current_weights_{period_key}.parquet'
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_peer_group_drawdowns(gics: str, version: str = 'mv') -> pd.DataFrame:
    """Load precomputed peer group drawdowns for a GICS industry

    Args:
        gics: GICS industry group name
        version: 'mv' for Market Value or 'weighted' for Weighted Price

    Returns:
        DataFrame with drawdown data for the peer group
    """
    filename = f'peer_group_drawdowns_{version}.parquet'
    path = R3000_PRECOMPUTED_DIR / filename
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = _ensure_datetime(df, ['peak_date', 'trough_date'])

    # Filter by GICS
    df = df[df['gics'] == gics].copy()

    return df


def load_iwv_total_mv_drawdowns() -> pd.DataFrame:
    """Load precomputed IWV Total Market Value drawdowns

    Returns:
        DataFrame with drawdown data for IWV total market value
    """
    path = R3000_PRECOMPUTED_DIR / 'iwv_total_mv_drawdowns.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date'])


def load_ark_stock_drawdowns(etf: str, ticker: str = None) -> pd.DataFrame:
    """Load precomputed stock drawdowns for ARK ETF holdings

    Args:
        etf: ETF name (e.g., 'ARKK')
        ticker: Optional ticker to filter (e.g., 'TSLA')

    Returns:
        DataFrame with full drawdown data for each stock
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_stock_drawdowns.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = _ensure_datetime(df, ['peak_date', 'trough_date'])

    if ticker is not None:
        # Match either clean ticker or full ticker
        df = df[(df['ticker'] == ticker) | (df['ticker_full'] == ticker)].copy()

    return df


def load_r3000_stock_drawdowns_detailed(ticker: str = None) -> pd.DataFrame:
    """Load precomputed detailed stock drawdowns for R3000

    Args:
        ticker: Optional ticker to filter (e.g., 'AAPL')

    Returns:
        DataFrame with full drawdown data for each stock
    """
    path = R3000_PRECOMPUTED_DIR / 'r3000_stock_drawdowns_detailed.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = _ensure_datetime(df, ['peak_date', 'trough_date'])

    if ticker is not None:
        # Match either clean ticker or full ticker
        df = df[(df['ticker'] == ticker) | (df['ticker_full'] == ticker)].copy()

    return df


def load_position_changes(etf: str, dd_rank: str = None) -> pd.DataFrame:
    """Load precomputed position changes during drawdowns for an ETF

    Args:
        etf: ETF name (e.g., 'ARKK')
        dd_rank: Optional drawdown rank to filter (e.g., '1', '2')

    Returns:
        DataFrame with position change data for each drawdown period
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_position_changes.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = _ensure_datetime(df, ['peak_date', 'trough_date', 'peak_date_actual', 'trough_date_actual'])

    if dd_rank is not None:
        df = df[df['dd_rank'] == dd_rank].copy()

    return df


def load_ark_holdings_max_drawdowns(etf: str = None) -> pd.DataFrame:
    """Load precomputed max drawdowns for ARK holdings

    Args:
        etf: ETF name (e.g., 'ARKK'). If None, returns all ETFs.

    Returns:
        DataFrame with columns: etf, ticker, max_drawdown, num_drawdowns, first_date, last_date
    """
    path = ARK_PRECOMPUTED_DIR / 'ark_holdings_max_drawdowns.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Ensure date columns are datetime
    df = _ensure_datetime(df, ['first_date', 'last_date'])

    if etf is not None:
        df = df[df['etf'] == etf].copy()

    return df


def filter_by_period(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
                     date_column: str = 'Date') -> pd.DataFrame:
    """Filter DataFrame by analysis period

    Args:
        df: DataFrame to filter
        start_date: Period start date
        end_date: Period end date
        date_column: Name of date column to filter on

    Returns:
        Filtered DataFrame
    """
    if len(df) == 0 or date_column not in df.columns:
        return df

    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df[mask].copy()


def filter_drawdowns_by_period(df: pd.DataFrame, start_date: pd.Timestamp,
                                end_date: pd.Timestamp) -> pd.DataFrame:
    """Filter drawdowns DataFrame by peak_date within analysis period and re-rank

    Args:
        df: DataFrame with peak_date column
        start_date: Period start date
        end_date: Period end date

    Returns:
        Filtered DataFrame with re-ranked drawdowns (1 = deepest in period)
    """
    if len(df) == 0 or 'peak_date' not in df.columns:
        return df

    mask = (df['peak_date'] >= start_date) & (df['peak_date'] <= end_date)
    filtered = df[mask].copy()

    if len(filtered) == 0:
        return filtered

    # Re-rank drawdowns within the filtered period (excluding 'Current')
    # Rank by depth_pct (most negative = rank 1)
    historical = filtered[filtered['rank'] != 'Current'].copy()
    current = filtered[filtered['rank'] == 'Current'].copy()

    if len(historical) > 0:
        historical = historical.sort_values('depth_pct', ascending=True)
        historical['rank'] = range(1, len(historical) + 1)

    # Combine back
    filtered = pd.concat([historical, current], ignore_index=True)
    return filtered


def get_metadata() -> dict:
    """Load precomputed metadata

    Returns:
        Dictionary with metadata (version, generated_at, source_hashes, parameters)
    """
    if not METADATA_PATH.exists():
        return {}

    with open(METADATA_PATH, 'r') as f:
        return json.load(f)


def check_precomputed_validity() -> Tuple[bool, str]:
    """Check if precomputed data is valid and up-to-date

    Returns:
        Tuple of (is_valid, message)
    """
    # Check if precomputed directories exist
    if not ARK_PRECOMPUTED_DIR.exists():
        return False, "Precomputed data directory not found. Run `python convert_to_parquet.py` to generate."

    # Check if metadata exists
    if not METADATA_PATH.exists():
        return False, "Metadata file not found. Run `python convert_to_parquet.py` to generate."

    # Load metadata
    metadata = get_metadata()
    if not metadata:
        return False, "Metadata file is empty. Run `python convert_to_parquet.py` to generate."

    # Check source file hashes
    from data_loader import get_ark_files_hash, get_r3000_files_hash

    current_ark_hash = get_ark_files_hash()
    current_r3000_hash = get_r3000_files_hash()

    stored_ark_hash = metadata.get('source_hashes', {}).get('ark_files_hash', 0)
    stored_r3000_hash = metadata.get('source_hashes', {}).get('r3000_files_hash', 0)

    if current_ark_hash != stored_ark_hash:
        return False, "ARK source data has been updated. Run `python convert_to_parquet.py` to regenerate precomputed data."

    if current_r3000_hash != stored_r3000_hash:
        return False, "Russell 3000 source data has been updated. Run `python convert_to_parquet.py` to regenerate precomputed data."

    return True, f"Precomputed data is valid. Generated at: {metadata.get('generated_at', 'unknown')}"


def check_precomputed_exists() -> bool:
    """Quick check if any precomputed data exists (for page initialization)

    Returns:
        True if precomputed data exists, False otherwise
    """
    return ARK_PRECOMPUTED_DIR.exists() and any(ARK_PRECOMPUTED_DIR.glob('*.parquet'))


def load_sp500_correlation_matrix(lookback_days: int = 120, period_key: str = None) -> pd.DataFrame:
    """Load precomputed S&P 500 Top 50 correlation matrix

    Args:
        lookback_days: Number of days used for correlation calculation (60, 120, or 250)
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')

    Returns:
        DataFrame with correlation matrix (tickers as both index and columns)
    """
    # Try period-specific file first
    if period_key is not None:
        path = ARK_PRECOMPUTED_DIR / f'SP500_top50_correlation_matrix_{period_key}_{lookback_days}d.parquet'
        if path.exists():
            return pd.read_parquet(path)

    # Fallback to legacy file without period key
    path = ARK_PRECOMPUTED_DIR / f'SP500_top50_correlation_matrix_{lookback_days}d.parquet'
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_qqq_correlation_matrix(lookback_days: int = 120, period_key: str = None) -> pd.DataFrame:
    """Load precomputed QQQ Top 50 correlation matrix

    Args:
        lookback_days: Number of days used for correlation calculation (60, 120, or 250)
        period_key: Analysis period key (e.g., '2024-2026' or '2021-2023')

    Returns:
        DataFrame with correlation matrix (tickers as both index and columns)
    """
    if period_key is not None:
        path = ARK_PRECOMPUTED_DIR / f'QQQ_top50_correlation_matrix_{period_key}_{lookback_days}d.parquet'
        if path.exists():
            return pd.read_parquet(path)

    path = ARK_PRECOMPUTED_DIR / f'QQQ_top50_correlation_matrix_{lookback_days}d.parquet'
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_conviction_drawdowns(etf: str, period_key: str = None) -> pd.DataFrame:
    """Load precomputed conviction vs drawdown data for an ETF and period

    Args:
        etf: ETF name (e.g., 'ARKK')
        period_key: Analysis period key (e.g., '2024-2026'). If None, uses default.

    Returns:
        DataFrame with conviction drawdown data
    """
    if period_key is None:
        from config import DEFAULT_PERIOD
        period_key = DEFAULT_PERIOD

    path = ARK_PRECOMPUTED_DIR / f'{etf}_conviction_drawdowns_{period_key}.parquet'
    if not path.exists():
        # Fallback to old format without period
        path = ARK_PRECOMPUTED_DIR / f'{etf}_conviction_drawdowns.parquet'
        if not path.exists():
            return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date', 'recovery_date'])


def load_stress_correlations(etf: str) -> pd.DataFrame:
    """Load precomputed stress correlations for an ETF

    Args:
        etf: ETF name (e.g., 'ARKK')

    Returns:
        DataFrame with stress correlation data for each drawdown period
        Columns: dd_rank, peak_date, trough_date, depth_pct, duration_days,
                 num_tickers, num_pairs, mean_corr, median_corr, min_corr, max_corr, std_corr
    """
    path = ARK_PRECOMPUTED_DIR / f'{etf}_stress_correlations.parquet'
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    return _ensure_datetime(df, ['peak_date', 'trough_date'])
