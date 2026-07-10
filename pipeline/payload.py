"""Assemble the KV payload for research:risk:etf-drawdowns."""
from datetime import datetime, timezone
from pipeline.drawdown import compute_drawdown, summarize_drawdown
from pipeline.svg import drawdown_chart_svg, heatmap_strip_svg


def build_etf_drawdowns_payload(closes_by_etf: dict, as_of: str) -> dict:
    dd_series = {etf: compute_drawdown(s) for etf, s in closes_by_etf.items()}
    table = [{"etf": etf, **summarize_drawdown(s)} for etf, s in closes_by_etf.items()]
    return {
        "schema_version": 1,
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etfs": list(closes_by_etf),
        "table": table,
        "chart_svg": drawdown_chart_svg(dd_series),
        "heatmap_svg": heatmap_strip_svg(dd_series),
    }


def build_etf_drawdowns_series(closes_by_etf: dict, as_of: str) -> dict:
    """Raw closes + per-day drawdown, for the xlsx export route."""
    etfs = {}
    for etf, s in closes_by_etf.items():
        dd = compute_drawdown(s)
        etfs[etf] = {
            "dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "close": [round(float(v), 4) for v in s],
            "dd_pct": [round(float(v) * 100, 4) for v in dd],
        }
    return {"schema_version": 1, "as_of": as_of, "etfs": etfs}
