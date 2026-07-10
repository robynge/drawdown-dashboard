from pathlib import Path
import pandas as pd
import pytest
from pipeline.drawdown import summarize_drawdown

FIX = Path(__file__).parent / "fixtures"

# Frozen legacy-pipeline outputs (data as of 2026-06-26). Do NOT edit these to
# make a failing test green — a mismatch means the new code regressed, not that
# the judge is stale. Values derive from output/<ETF>_prices.csv via the legacy
# drawdown semantics (first occurrence of episode max).
EXPECTED = {
    "ARKK": {"current_dd_pct": -49.0812, "max_dd_pct": -80.9168, "peak_date": "2021-02-12", "trough_date": "2022-12-28"},
    "ARKQ": {"current_dd_pct": -13.3570, "max_dd_pct": -59.8859, "peak_date": "2021-02-12", "trough_date": "2022-12-28"},
    "ARKW": {"current_dd_pct": -22.4183, "max_dd_pct": -80.0145, "peak_date": "2021-02-12", "trough_date": "2022-12-28"},
    "ARKG": {"current_dd_pct": -62.8165, "max_dd_pct": -83.5913, "peak_date": "2021-01-20", "trough_date": "2025-04-08"},
    "ARKF": {"current_dd_pct": -38.2594, "max_dd_pct": -78.6276, "peak_date": "2021-02-16", "trough_date": "2022-12-28"},
    "ARKX": {"current_dd_pct": -14.1759, "max_dd_pct": -43.6235, "peak_date": "2021-09-02", "trough_date": "2022-10-14"},
}


@pytest.mark.parametrize("etf", sorted(EXPECTED))
def test_parity_with_legacy_outputs(etf):
    df = pd.read_csv(FIX / f"{etf}_prices.csv", parse_dates=["Date"], index_col="Date")
    out = summarize_drawdown(df["Close"])
    exp = EXPECTED[etf]
    assert out["current_dd_pct"] == pytest.approx(exp["current_dd_pct"], abs=0.01)
    assert out["max_dd_pct"] == pytest.approx(exp["max_dd_pct"], abs=0.01)
    assert out["peak_date"] == exp["peak_date"]
    assert out["trough_date"] == exp["trough_date"]
