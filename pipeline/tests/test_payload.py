import json
import pandas as pd
from pipeline.payload import build_etf_drawdowns_payload


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
