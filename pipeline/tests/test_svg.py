import pandas as pd
from pipeline.svg import drawdown_chart_svg, heatmap_strip_svg


def _dd(vals, start="2024-01-05"):
    idx = pd.date_range(start, periods=len(vals), freq="W-FRI")
    return pd.Series(vals, index=idx, dtype=float)


SERIES = {"ARKK": _dd([0, -0.1, -0.5]), "ARKQ": _dd([0, -0.05, -0.2])}


def test_chart_svg_structure():
    svg = drawdown_chart_svg(SERIES)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'viewBox="0 0 720 300"' in svg
    assert svg.count("<polyline") == 2
    assert "ARKK" in svg and "ARKQ" in svg
    assert "-50%" in svg
    assert not any("一" <= ch <= "鿿" for ch in svg)


def test_chart_svg_deterministic():
    assert drawdown_chart_svg(SERIES) == drawdown_chart_svg(SERIES)


def test_heatmap_has_one_row_per_etf():
    svg = heatmap_strip_svg(SERIES)
    assert svg.count('class="hm-row"') == 2
    assert "<svg" in svg and "</svg>" in svg
