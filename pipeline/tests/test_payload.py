import json
import pandas as pd
from pipeline.payload import build_etf_drawdowns_payload, build_etf_drawdowns_series


def _closes(vals, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="B")
    return pd.Series(vals, index=idx, dtype=float)


def test_payload_shape_and_json_safety():
    closes = {"ARKK": _closes([100, 50, 75]), "ARKQ": _closes([10, 12, 9])}
    p = build_etf_drawdowns_payload(closes, as_of="2026-07-09")
    json.dumps(p)
    assert p["as_of"] == "2026-07-09"
    assert p["etfs"] == ["ARKK", "ARKQ"]
    row = {r["etf"]: r for r in p["table"]}["ARKK"]
    assert row["max_dd_pct"] == -50.0
    assert p["chart_svg"].startswith("<svg")
    assert p["heatmap_svg"].startswith("<svg")
    assert p["schema_version"] == 1


def test_series_payload_shape():
    closes = {"ARKK": _closes([100, 50, 75])}
    p = build_etf_drawdowns_series(closes, as_of="2026-07-09")
    json.dumps(p)
    assert p["schema_version"] == 1
    assert p["as_of"] == "2026-07-09"
    arkk = p["etfs"]["ARKK"]
    assert arkk["dates"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert arkk["close"] == [100.0, 50.0, 75.0]
    assert arkk["dd_pct"] == [0.0, -50.0, -25.0]
