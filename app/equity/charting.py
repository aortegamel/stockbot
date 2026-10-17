"""Charting data math + render fallbacks (FinRobot ReportChartUtils port).

Ports ``get_share_performance`` (ticker vs S&P 500 rebased percent change)
and ``get_pe_eps_performance`` (price / EPS with zero-safe narrowing) to
deterministic stdlib-only pure functions. Callers supply already-fetched
close/EPS lists; this module never fetches, never writes files.

SVG output is a minimal hand-built ``<polyline>`` string (no matplotlib);
Markdown tables are the no-graphics fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from xml.sax.saxutils import escape

_WIDTH = 640
_HEIGHT = 360
_PAD_LEFT = 48
_PAD_RIGHT = 12
_PAD_TOP = 24
_PAD_BOTTOM = 28
_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def _as_list(value: object) -> list[object]:
    """Narrow untrusted input to a list ([] when absent/mistyped)."""
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_dict(value: object) -> dict[str, object]:
    """Narrow untrusted input to a str-keyed mapping ({} when absent/mistyped)."""
    if not isinstance(value, Mapping):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str)}


def _num(value: object) -> float | None:
    """Finite float for a numeric entry (None when missing/non-numeric/bool/non-finite)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _cell(value: object) -> str:
    """Compact scalar text ("" for None/dict/list)."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def _rebased(closes: list[float]) -> list[float]:
    """Percent change rebased to the first close (first point is 0.0)."""
    base = closes[0]
    return [(close / base - 1.0) * 100.0 for close in closes]


def sp500_relative_changes(
    ticker_closes: Sequence[object] | None, sp500_closes: Sequence[object] | None
) -> dict[str, list[float]]:
    """Rebased pct-change series for ticker vs S&P 500 (FinRobot share-performance math).

    Each leg is ``(close / first_close - 1) * 100`` over pairwise-valid
    points (index kept only where both closes are numeric, so one bad
    tick never misaligns the rest). Empty input or a non-positive base
    close yields ``[]`` legs.
    """
    ticker_rows = _as_list(ticker_closes)
    index_rows = _as_list(sp500_closes)
    pairs = [
        (t, s)
        for t, s in ((_num(ticker_rows[i]), _num(index_rows[i])) for i in range(min(len(ticker_rows), len(index_rows))))
        if t is not None and s is not None
    ]
    if not pairs or pairs[0][0] <= 0 or pairs[0][1] <= 0:
        return {"ticker_pct": [], "sp500_pct": []}
    ticker = [p[0] for p in pairs]
    index = [p[1] for p in pairs]
    return {"ticker_pct": _rebased(ticker), "sp500_pct": _rebased(index)}


def pe_eps_points(closes: Sequence[object] | None, eps: Sequence[object] | None) -> dict[str, list[float | None]]:
    """Per-point P/E ratios (FinRobot pe/eps-performance math).

    Zero, missing, or non-numeric EPS yields ``None`` (never ``inf``);
    lengths truncate to the shorter leg, non-numeric closes yield ``None``.
    """
    close_rows = _as_list(closes)
    eps_rows = _as_list(eps)
    count = min(len(close_rows), len(eps_rows))
    points: list[float | None] = []
    for i in range(count):
        price = _num(close_rows[i])
        earning = _num(eps_rows[i])
        if price is None or earning is None or earning == 0.0:
            points.append(None)
            continue
        ratio = price / earning
        points.append(ratio if math.isfinite(ratio) else None)
    return {"pe": points}


def _plot_values(series: Mapping[str, object]) -> dict[str, list[object]]:
    """Series narrowed to list values, sorted by name (empty/unusable dropped later)."""
    out: dict[str, list[object]] = {}
    for name in sorted(series):
        values = series[name]
        if isinstance(values, (list, tuple)) and values:
            out[name] = list(values)
    return out


def _plot_bounds(data: dict[str, list[object]]) -> tuple[float, float] | None:
    """(min, max) over numeric entries (None when no plottable point)."""
    lo = hi = None
    for values in data.values():
        for entry in values:
            num = _num(entry)
            if num is None:
                continue
            lo = num if lo is None or num < lo else lo
            hi = num if hi is None or num > hi else hi
    return (lo, hi) if lo is not None and hi is not None else None


def _x_at(i: int, count: int) -> float:
    """X for point index (left edge when a single point)."""
    width = _WIDTH - _PAD_LEFT - _PAD_RIGHT
    return float(_PAD_LEFT) if count <= 1 else _PAD_LEFT + i * width / (count - 1)


def _y_at(value: float, lo: float, hi: float) -> float:
    """Y for a value (vertical midline when flat)."""
    height = _HEIGHT - _PAD_TOP - _PAD_BOTTOM
    if hi <= lo:
        return _PAD_TOP + height / 2.0
    return _PAD_TOP + (hi - value) / (hi - lo) * height


def _segments(values: list[object], count: int, lo: float, hi: float) -> list[list[str]]:
    """Plottable `"x,y"` segments (gaps split on None/non-numeric)."""
    runs: list[list[str]] = []
    current: list[str] = []
    for i, entry in enumerate(values):
        num = _num(entry)
        if num is None:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(f"{_x_at(i, count):.1f},{_y_at(num, lo, hi):.1f}")
    if current:
        runs.append(current)
    return runs


def render_svg_series(
    series: dict[str, list[float | None]] | None,
    labels: Sequence[object] | None,
    title: str | None,
) -> str:
    """Minimal ``<svg>`` polyline chart ("" when nothing plottable, never raises)."""
    del labels  # x geometry comes from series positions; labels serve the markdown fallback
    data = _plot_values(_as_dict(series))
    bounds = _plot_bounds(data)
    if bounds is None:
        return ""
    lo, hi = bounds
    count = max(len(values) for values in data.values())
    shapes: list[str] = []
    for idx, (name, values) in enumerate(data.items()):
        color = _COLORS[idx % len(_COLORS)]
        for run in _segments(values, count, lo, hi):
            if len(run) == 1:
                x, y = run[0].split(",")
                shapes.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{color}"/>')
            else:
                shapes.append(
                    f'<polyline points="{" ".join(run)}" fill="none" '
                    f'stroke="{color}" stroke-width="2"><title>{escape(name)}</title></polyline>'
                )
    legend = "".join(
        f'<text x="{_WIDTH - _PAD_RIGHT - 8}" y="{_PAD_TOP + 14 * i}" '
        f'text-anchor="end" font-size="11" fill="{_COLORS[i % len(_COLORS)]}">{escape(name)}</text>'
        for i, name in enumerate(data)
    )
    head = _cell(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" role="img">'
        f"<title>{escape(head) if head else 'chart'}</title>{''.join(shapes)}{legend}</svg>"
    )


def _fmt_value(value: object) -> str:
    """Markdown cell: 2-decimal numbers, "n/a" for gaps."""
    num = _num(value)
    return f"{num:.2f}" if num is not None else "n/a"


def _md_escape(text: str) -> str:
    """Cell text with pipes escaped so tables never break."""
    return text.replace("|", "\\|")


def _series_rows(data: dict[str, list[object]], names: list[str], labels: list[object]) -> list[str]:
    """Format aligned series rows with positional labels and ``n/a`` gaps."""
    count = max(len(values) for values in data.values())
    rows: list[str] = []
    for i in range(count):
        label = _md_escape(_cell(labels[i])) if i < len(labels) else str(i)
        cells = [_fmt_value(values[i]) if i < len(values) else "n/a" for values in (data[n] for n in names)]
        rows.append("| " + label + " | " + " | ".join(cells) + " |")
    return rows


def _series_table(series: Mapping[str, object], labels: object, title: object) -> str:
    """Markdown table over named series (one row per index, gaps are "n/a")."""
    data = _plot_values(series)
    if not data:
        return ""
    names = sorted(data)
    head = f"## {_cell(title).strip()}\n\n" if _cell(title).strip() else ""
    header = "| label | " + " | ".join(_md_escape(n) for n in names) + " |"
    divider = "| --- | " + " | ".join("---" for _ in names) + " |"
    rows = _series_rows(data, names, _as_list(labels))
    return head + "\n".join([header, divider, *rows])


def _generic_table(headers: object, rows: object, title: object) -> str:
    """Markdown table from a headers list + rows matrix."""
    cols = [_md_escape(_cell(h)) or f"col{i}" for i, h in enumerate(_as_list(headers))]
    if not cols:
        return ""
    head = f"## {_cell(title).strip()}\n\n" if _cell(title).strip() else ""
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    body: list[str] = []
    for row in _as_list(rows):
        cells_in = _as_list(row)
        cells = [_md_escape(_cell(c)) if _num(c) is None else _fmt_value(c) for c in cells_in]
        cells += ["n/a"] * (len(cols) - len(cells))
        body.append("| " + " | ".join(cells[: len(cols)]) + " |")
    return head + "\n".join([header, divider, *body])


def render_markdown_table(
    series: dict[str, list[float | None]] | list[str] | None,
    labels: Sequence[object] | Sequence[Sequence[object]] | None = None,
    title: str | None = "",
) -> str:
    """Markdown fallback for charts (never raises, "" when nothing tabular).

    Series mode (dict, mirrors :func:`render_svg_series`) renders one row
    per index with ``labels`` as the first column; headers mode (list)
    renders ``labels`` as the rows matrix under ``series`` as headers.
    """
    if isinstance(series, Mapping):
        return _series_table(series, labels, title)
    if isinstance(series, (list, tuple)):
        return _generic_table(series, labels, title)
    return ""
