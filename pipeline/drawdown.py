"""Drawdown math. Pure functions over a price Series (DatetimeIndex, float)."""
import pandas as pd


def compute_drawdown(close: pd.Series) -> pd.Series:
    """dd_t = close_t / running_max_t - 1, in [-1, 0]."""
    return close / close.cummax() - 1.0


def _peak_before(close: pd.Series, dd: pd.Series, ref_pos: int) -> pd.Index:
    """Find the peak date that "owns" the drawdown episode ending at ref_pos.

    Walks back from ref_pos through the contiguous run of at-a-running-high
    points (dd == 0) that immediately precedes it, and returns the *first*
    index of that run -- i.e. the point where price most recently recovered
    to a running high right before its final, unbroken descent toward
    ref_pos. This is what an episode's "peak" means: not just any tied
    all-time-high, but the one that starts the decline the episode is made
    of.
    """
    if ref_pos == 0:
        return close.index[0]

    is_at_high = dd.to_numpy() == 0.0
    p = ref_pos - 1
    if not is_at_high[p]:
        # No at-high point immediately precedes ref_pos (shouldn't happen
        # since dd always starts at 0 for the first observation).
        return close.index[0]

    while p - 1 >= 0 and is_at_high[p - 1]:
        p -= 1
    return close.index[p]


def summarize_drawdown(close: pd.Series) -> dict:
    """Summarize the deepest drawdown episode and the current (as-of-last) one."""
    dd = compute_drawdown(close)
    trough_i = dd.idxmin()
    trough_pos = close.index.get_loc(trough_i)
    peak_i = _peak_before(close, dd, trough_pos)

    return {
        "last_close": round(float(close.iloc[-1]), 4),
        "current_dd_pct": round(float(dd.iloc[-1] * 100), 4),
        "current_peak_date": peak_i.strftime("%Y-%m-%d"),
        "max_dd_pct": round(float(dd.min() * 100), 4),
        "peak_date": peak_i.strftime("%Y-%m-%d"),
        "trough_date": trough_i.strftime("%Y-%m-%d"),
    }
