import pandas as pd
import pytest
from pipeline.drawdown import compute_drawdown, summarize_drawdown


def _series(vals, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="B")
    return pd.Series(vals, index=idx, dtype=float)


def test_drawdown_series_basic():
    s = _series([100, 110, 99, 121, 60.5])
    dd = compute_drawdown(s)
    assert dd.iloc[0] == 0.0
    assert dd.iloc[1] == 0.0
    assert dd.iloc[2] == pytest.approx(99 / 110 - 1)
    assert dd.iloc[3] == 0.0
    assert dd.iloc[4] == pytest.approx(60.5 / 121 - 1)


def test_summarize_picks_deepest_episode():
    s = _series([100, 80, 100, 100, 50, 75])
    out = summarize_drawdown(s)
    assert out["max_dd_pct"] == pytest.approx(-50.0)
    assert out["peak_date"] == "2024-01-03"
    assert out["trough_date"] == "2024-01-05"
    assert out["current_dd_pct"] == pytest.approx(-25.0)
    assert out["current_peak_date"] == "2024-01-03"
    assert out["last_close"] == pytest.approx(75.0)


def test_monotonic_rise_has_zero_drawdown():
    s = _series([1, 2, 3, 4])
    out = summarize_drawdown(s)
    assert out["max_dd_pct"] == 0.0
    assert out["current_dd_pct"] == 0.0
