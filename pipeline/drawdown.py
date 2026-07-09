"""Drawdown math. Pure functions over a price Series (DatetimeIndex, float)."""
import pandas as pd


def compute_drawdown(close: pd.Series) -> pd.Series:
    """dd_t = close_t / running_max_t - 1, in [-1, 0]."""
    return close / close.cummax() - 1.0


def summarize_drawdown(close: pd.Series) -> dict:
    """Summarize the deepest drawdown episode and the current (as-of-last) one.

    Peak-date semantics match the legacy implementation (real-data parity):
    a peak is the date the relevant max close is FIRST touched -- pandas
    ``idxmax`` default tie-breaking -- not the last tied high before the
    decline.

    - ``peak_date``: first occurrence of the max close up to (and including)
      the max-drawdown trough.
    - ``current_peak_date``: first occurrence of the max close over the whole
      series (the running high the current drawdown is measured from).
    """
    dd = compute_drawdown(close)
    trough_i = dd.idxmin()
    peak_before_trough = close.loc[:trough_i].idxmax()
    overall_peak = close.idxmax()
    return {
        "last_close": round(float(close.iloc[-1]), 4),
        "current_dd_pct": round(float(dd.iloc[-1] * 100), 4),
        "current_peak_date": overall_peak.strftime("%Y-%m-%d"),
        "max_dd_pct": round(float(dd.min() * 100), 4),
        "peak_date": peak_before_trough.strftime("%Y-%m-%d"),
        "trough_date": trough_i.strftime("%Y-%m-%d"),
    }
