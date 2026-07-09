"""Hand-rolled inline SVG builders. No matplotlib -- tiny, theme-aware output."""
from xml.sax.saxutils import escape

import pandas as pd

PALETTE = {"ARKK": "#d6604d", "ARKQ": "#4393c3", "ARKW": "#f4a582",
           "ARKG": "#5aae61", "ARKF": "#9970ab", "ARKX": "#b8860b"}
W, H, PAD = 720, 300, 36
HEAT_BUCKETS = [(-0.10, "#fddbc7"), (-0.25, "#f4a582"), (-0.50, "#d6604d"), (-1.01, "#b2182b")]


def _weekly(s: pd.Series) -> pd.Series:
    return s.resample("W-FRI").last().dropna()


def _x(i, n):
    return PAD + i * (W - 2 * PAD) / max(n - 1, 1)


def _y(dd, dd_min):
    return PAD + (dd / dd_min) * (H - 2 * PAD) if dd_min < 0 else PAD


def drawdown_chart_svg(series_by_etf: dict) -> str:
    weekly = {k: _weekly(v) for k, v in series_by_etf.items()}
    dd_min = min((s.min() for s in weekly.values() if len(s)), default=-1.0) or -1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="ARK ETF drawdown curves">']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        dd = dd_min * frac
        y = _y(dd, dd_min)
        parts.append(f'<line x1="{PAD}" y1="{y:.1f}" x2="{W-PAD}" y2="{y:.1f}" stroke="var(--line-soft,#ddd)" stroke-width="1"/>')
        parts.append(f'<text x="{PAD-6}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="var(--ink,#555)">{dd*100:.0f}%</text>')
    for etf, s in weekly.items():
        pts = " ".join(f"{_x(i, len(s)):.1f},{_y(v, dd_min):.1f}" for i, v in enumerate(s))
        parts.append(f'<polyline fill="none" stroke="{PALETTE.get(etf, "#888")}" stroke-width="1.6" points="{pts}"/>')
    for j, etf in enumerate(weekly):
        x = PAD + j * 92
        parts.append(f'<rect x="{x}" y="8" width="10" height="10" fill="{PALETTE.get(etf, "#888")}"/>')
        parts.append(f'<text x="{x+14}" y="17" font-size="11" fill="var(--ink,#555)">{escape(etf)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap_strip_svg(series_by_etf: dict) -> str:
    weekly = {k: _weekly(v) for k, v in series_by_etf.items()}
    row_h, label_w = 22, 52
    h = row_h * len(weekly) + 20
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="Drawdown heatmap by ETF">']
    for r, (etf, s) in enumerate(weekly.items()):
        y = r * row_h + 14
        parts.append(f'<g class="hm-row"><text x="0" y="{y+12}" font-size="11" fill="var(--ink,#555)">{escape(etf)}</text>')
        cw = (W - label_w) / max(len(s), 1)
        for i, v in enumerate(s):
            color = "#e8f0e8" if v > -0.02 else next((c for lim, c in HEAT_BUCKETS if v > lim), "#b2182b")
            parts.append(f'<rect x="{label_w+i*cw:.1f}" y="{y}" width="{cw+0.5:.1f}" height="{row_h-4}" fill="{color}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)
