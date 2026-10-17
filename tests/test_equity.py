"""Equity render regressions: upside percent units + pairwise SP500 alignment."""

import math
from xml.etree import ElementTree

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.equity import (
    ValuationEngine,
    build_equity_report,
    financial_data_from_evidence,
    from_evidence,
    pe_eps_points,
    render_markdown_table,
    render_report_markdown,
    render_svg_series,
    sp500_relative_changes,
)
from app.equity.charting import _as_dict, _generic_table, _plot_values


def _valuation_body(ticker: str, synthesis: dict[str, object]) -> str:
    report = build_equity_report(ticker, {"synthesis": synthesis})
    sections = report.get("sections")
    assert isinstance(sections, list)
    section = next(s for s in sections if isinstance(s, dict) and s.get("id") == "valuation")
    assert isinstance(section, dict)
    body = section.get("body_markdown")
    assert isinstance(body, str)
    return body


def test_upside_ratio_renders_as_percent() -> None:
    body = _valuation_body("AAPL", {"target_price": 185.7, "upside": 0.857})
    assert "- Upside: 85.70%." in body
    assert "0.86%" not in body


def test_sp500_skips_bad_tick_pairwise() -> None:
    result = sp500_relative_changes([100.0, "bad", 110.0], [100.0, 105.0, 115.0])
    ticker_pct = result.get("ticker_pct")
    index_pct = result.get("sp500_pct")
    assert isinstance(ticker_pct, list) and isinstance(index_pct, list)
    assert len(ticker_pct) == 2
    assert ticker_pct[1] == pytest.approx(10.0)
    assert index_pct[1] == pytest.approx(15.0)


def test_from_evidence_maps_snapshot_keys() -> None:
    engine = from_evidence(
        {"price": {"last": 213.05}, "shares_outstanding": 1000, "ttm_gaap_eps": 5.0},
        peer_multiples={"AAA": 10.0, "BAD": -3.0},
        ev_ebitda_history=[8.0, 10.0],
    )
    assert engine.financial_data["current_price"] == 213.05
    assert engine.financial_data["shares_outstanding"] == 1000
    assert engine.financial_data["ev_ebitda_history"] == [8.0, 10.0]
    assert set(engine.peer_data) == {"AAA"}
    assert from_evidence(None).financial_data["current_price"] == 0.0


def test_financial_data_from_evidence_maps_fcf() -> None:
    data = financial_data_from_evidence(
        {"price": {"last": 213.05}, "shares_outstanding": 1000},
        {"safety": {"ttm_fcf": 3200.0}},
    )
    assert data["current_price"] == 213.05
    assert data["shares_outstanding"] == 1000
    assert data["free_cash_flow"] == 3200.0
    assert data["ebitda"] is None


def test_explicit_ebitda_fcf_overrides_win() -> None:
    engine = from_evidence(
        {"price": {"last": 100.0}, "shares_outstanding": 1000},
        dividend_fundamentals={"safety": {"ttm_fcf": 3200.0}},
        ebitda=500.0,
        free_cash_flow=100.0,
        ev_ebitda_history=[10.0],
    )
    assert engine.financial_data["ebitda"] == 500.0
    assert engine.financial_data["free_cash_flow"] == 100.0
    assert engine.synthesize_valuation()["methods_used"] != []


def test_zero_confidence_bands_render_not_available() -> None:
    engine = from_evidence({"price": {"last": 100.0}, "shares_outstanding": 1000})
    synthesis = engine.synthesize_valuation()
    assert synthesis["methods_used"] == []
    report = build_equity_report(
        "AAPL", {"synthesis": synthesis, "football_field": engine.generate_football_field_data()}
    )
    sections = report.get("sections")
    assert isinstance(sections, list)
    section = next(s for s in sections if isinstance(s, dict) and s.get("id") == "valuation")
    assert isinstance(section, dict)
    assert section.get("body_markdown") == "Valuation not available (no evidence supplied)."


def test_report_markdown_carries_upside_percent() -> None:
    report = build_equity_report("AAPL", {"synthesis": {"target_price": 185.7, "upside": 0.857}})
    assert "85.70%" in render_report_markdown(report)


@pytest.mark.parametrize(
    ("ticker", "index"),
    [
        (None, None),
        ([], [100.0]),
        ([100.0], []),
        ([0.0, 100.0], [100.0, 110.0]),
        ([-1.0, 100.0], [100.0, 110.0]),
        ([100.0, 110.0], [0.0, 100.0]),
        ([None, True, float("nan")], [100.0, 110.0, 120.0]),
    ],
)
def test_sp500_empty_or_nonpositive_base_returns_empty(ticker: list[object] | None, index: list[object] | None) -> None:
    assert sp500_relative_changes(ticker, index) == {"ticker_pct": [], "sp500_pct": []}


@settings(derandomize=True, max_examples=50)
@given(
    st.lists(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1),
    st.lists(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1),
)
def test_sp500_relative_changes_rebase_and_preserve_pair_count(ticker: list[float], index: list[float]) -> None:
    result = sp500_relative_changes(ticker, index)
    count = min(len(ticker), len(index))
    assert len(result["ticker_pct"]) == len(result["sp500_pct"]) == count
    assert result["ticker_pct"][0] == result["sp500_pct"][0] == 0.0
    assert result["ticker_pct"][-1] == pytest.approx((ticker[count - 1] / ticker[0] - 1.0) * 100)
    assert result["sp500_pct"][-1] == pytest.approx((index[count - 1] / index[0] - 1.0) * 100)


def test_pe_eps_points_keep_gaps_and_truncate() -> None:
    assert pe_eps_points(
        [100, 90, "bad", 20, True, 30, 1e308, 40, 50],
        [5, -3, 2, 0, 1, None, 1e-308, float("inf")],
    ) == {"pe": [20.0, -30.0, None, None, None, None, None, None]}
    assert pe_eps_points(None, [1]) == {"pe": []}
    assert pe_eps_points([1], []) == {"pe": []}


@pytest.mark.parametrize("series", [None, {}, {"empty": []}, {"missing": [None, float("nan"), float("inf")]}])
def test_svg_without_numeric_values_is_empty(series: dict[str, list[float | None]] | None) -> None:
    assert render_svg_series(series, None, None) == ""


def test_svg_untrusted_series_drops_invalid_names_and_values() -> None:
    untrusted: dict[object, object] = {1: [100], "empty": [], "scalar": 10, "valid": (4.0,)}
    assert _plot_values(_as_dict(untrusted)) == {"valid": [4.0]}
    root = ElementTree.fromstring(render_svg_series({"empty": [], "valid": [4.0]}, [], None))
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert root.findtext("svg:title", namespaces=ns) == "chart"
    circle = root.find("svg:circle", ns)
    assert circle is not None
    assert circle.attrib == {"cx": "48.0", "cy": "178.0", "r": "2.5", "fill": "#1f77b4"}
    assert [entry.text for entry in root.findall("svg:text", ns)] == ["valid"]


def test_svg_splits_gaps_and_escapes_title_and_name() -> None:
    svg = render_svg_series({"A&B<": [None, 1, 3, None, None, 2, None]}, [], " <Price & value> ")
    root = ElementTree.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert root.findtext("svg:title", namespaces=ns) == "<Price & value>"
    line = root.find("svg:polyline", ns)
    assert line is not None
    assert line.attrib["points"] == "144.7,332.0 241.3,24.0"
    assert line.findtext("svg:title", namespaces=ns) == "A&B<"
    circle = root.find("svg:circle", ns)
    assert circle is not None
    assert circle.attrib["cx"] == "531.3"
    assert circle.attrib["cy"] == "178.0"
    assert root.findtext("svg:text", namespaces=ns) == "A&B<"


def test_svg_flat_series_uses_vertical_midline() -> None:
    root = ElementTree.fromstring(render_svg_series({"flat": [7.0, 7.0]}, None, "Flat"))
    line = root.find("{http://www.w3.org/2000/svg}polyline")
    assert line is not None
    assert line.attrib["points"] == "48.0,178.0 628.0,178.0"


@settings(derandomize=True, max_examples=50)
@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=20)
)
def test_svg_coordinates_contain_all_finite_values(values: list[float]) -> None:
    root = ElementTree.fromstring(render_svg_series({"values": list(values)}, None, "Bounds"))
    line = root.find("{http://www.w3.org/2000/svg}polyline")
    assert line is not None
    points = [tuple(map(float, point.split(","))) for point in line.attrib["points"].split()]
    assert len(points) == len(values)
    assert points[0][0] == 48.0
    assert points[-1][0] == 628.0
    assert all(48.0 <= x <= 628.0 and 24.0 <= y <= 332.0 for x, y in points)
    if min(values) < max(values):
        assert min(y for _, y in points) == 24.0
        assert max(y for _, y in points) == 332.0
    else:
        assert all(y == 178.0 for _, y in points)


def test_series_markdown_table_preserves_labels_gaps_and_heading() -> None:
    assert render_markdown_table({"B": [None], "A|value": [1, 2]}, ["first|label"], "  Price  ") == (
        "## Price\n\n| label | A\\|value | B |\n| --- | --- | --- |\n| first\\|label | 1.00 | n/a |\n| 1 | 2.00 | n/a |"
    )
    assert render_markdown_table({"A": [2.0]}, None, None) == "| label | A |\n| --- | --- |\n| 0 | 2.00 |"
    assert render_markdown_table({"A": []}) == ""


def test_generic_markdown_table_formats_and_normalizes_rows() -> None:
    rows = [[" a|b ", True, None, 1.234, "extra"], ["short"], None]
    expected = (
        "## Data\n\n"
        "| text\\|value | col1 | col2 | number |\n"
        "| --- | --- | --- | --- |\n"
        "| a\\|b | True |  | 1.23 |\n"
        "| short | n/a | n/a | n/a |\n"
        "| n/a | n/a | n/a | n/a |"
    )
    assert _generic_table(["text|value", None, {}, "number"], rows, "  Data  ") == expected
    assert render_markdown_table(["text|value", "", " ", "number"], rows, "  Data  ") == expected
    assert render_markdown_table(["Value"], [[2]], "") == "| Value |\n| --- |\n| 2.00 |"
    assert render_markdown_table([], [[1]]) == ""
    assert render_markdown_table(None) == ""


def test_report_overview_renders_all_available_evidence() -> None:
    report = build_equity_report(
        " abc ",
        {
            "valuation": {"price": {"last": 123.45}, "ttm_gaap_eps": 4.5, "trailing_pe": 27.43},
            "estimates": {"as_of": " 2026-01-02 "},
            "sp500_weight": {"weight_pct": 1.23, "rank": 12},
        },
    )
    text = render_report_markdown(report)
    assert "# ABC Equity Report" in text
    assert (
        "## Overview\n"
        "- Ticker: ABC.\n"
        "- Last price: 123.45.\n"
        "- ttm_gaap_eps: 4.50.\n"
        "- trailing_pe: 27.43.\n"
        "- Analyst estimates as of 2026-01-02.\n"
        "- S&P 500 weight: 1.23% (rank 12)."
    ) in text


def test_report_overview_omits_missing_rank_and_unusable_evidence() -> None:
    text = render_report_markdown(
        build_equity_report("", {"sp500_weight": {"weight_pct": 0}, "estimates": {"as_of": " "}})
    )
    assert "- Ticker: unknown.\n- S&P 500 weight: 0.00%." in text
    assert "(rank" not in text
    empty = render_report_markdown(
        build_equity_report("ABC", {"valuation": {"price": {"last": True}}, "estimates": {"as_of": None}})
    )
    assert "## Overview\nOverview not available (no evidence supplied)." in empty


def test_ev_ebitda_history_band_preserves_sample_stdev_and_net_debt() -> None:
    engine = ValuationEngine(
        {"ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [8, 10, 0, -2, None, True, float("nan")]}
    )
    result = engine.calculate_ev_ebitda_valuation()
    assert result.target_price == 81
    assert result.low_estimate == pytest.approx((9 - math.sqrt(2)) * 9)
    assert result.high_estimate == pytest.approx((9 + math.sqrt(2)) * 9)
    assert result.confidence == 0.7
    assert result.assumptions["history_count"] == 2
    assert result.assumptions["target_multiple"] == 9
    assert result.assumptions["history_stdev"] == pytest.approx(math.sqrt(2))
    assert result.assumptions["net_debt_pct_of_ev"] == 0.1
    override = engine.calculate_ev_ebitda_valuation(0.5)
    assert override.target_price == 4.5
    assert override.low_estimate == 0
    assert override.high_estimate == pytest.approx((0.5 + math.sqrt(2)) * 9)
    assert engine.calculate_ev_ebitda_valuation(-1).target_price == result.target_price
    assert engine.calculate_ev_ebitda_valuation(0).target_price == result.target_price


def test_ev_ebitda_override_without_history_and_single_history_point() -> None:
    engine = ValuationEngine({"ebitda": 1000, "shares_outstanding": 100})
    result = engine.calculate_ev_ebitda_valuation(10)
    assert (result.low_estimate, result.target_price, result.high_estimate) == (90, 90, 90)
    assert result.confidence == 0.5
    assert result.assumptions["history_stdev"] == 0
    assert result.assumptions["history_count"] == 0
    engine = ValuationEngine({"ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [10]})
    single = engine.calculate_ev_ebitda_valuation()
    assert (single.low_estimate, single.target_price, single.high_estimate) == (90, 90, 90)
    assert single.confidence == 0.7
    fractional = ValuationEngine(
        {"ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [0.5]}
    ).calculate_ev_ebitda_valuation()
    assert fractional.target_price == 4.5
    single_point = ValuationEngine(
        {"ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [10, 10]}
    ).calculate_ev_ebitda_valuation()
    assert single_point.assumptions["history_stdev"] == 0
    assert (single_point.low_estimate, single_point.target_price, single_point.high_estimate) == (90, 90, 90)


def test_ev_ebitda_zero_history_stdev_keeps_band_at_target() -> None:
    """A zero-variance history keeps the band pinned at the target multiple."""
    result = ValuationEngine(
        {"ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [9, 9]}
    ).calculate_ev_ebitda_valuation()
    assert result.assumptions["history_stdev"] == 0
    assert (result.low_estimate, result.target_price, result.high_estimate) == (81, 81, 81)


def test_ev_ebitda_missing_inputs_keep_full_assumption_keys() -> None:
    """Missing inputs return zero estimates with the net-debt and target-multiple keys intact."""
    result = ValuationEngine({"ebitda": 1000, "shares_outstanding": 100}).calculate_ev_ebitda_valuation(10)
    assert str(result.assumptions["net_debt_note"]).startswith("net debt assumed")
    assert result.assumptions["target_multiple"] == 10
    assert result.assumptions["net_debt_pct_of_ev"] == 0.1


@pytest.mark.parametrize(
    "financial_data",
    [
        {"ebitda": 1000, "shares_outstanding": 100},
        {"ebitda": 0, "shares_outstanding": 100, "ev_ebitda_history": [10]},
        {"ebitda": 1000, "shares_outstanding": 0, "ev_ebitda_history": [10]},
    ],
)
def test_ev_ebitda_missing_inputs_return_zero_confidence(financial_data: dict[str, object]) -> None:
    result = ValuationEngine(financial_data).calculate_ev_ebitda_valuation()
    assert (result.low_estimate, result.target_price, result.high_estimate, result.confidence) == (0, 0, 0, 0)
    assert result.description == "Missing EBITDA, shares outstanding, or EV/EBITDA multiple."
    assert ValuationEngine(financial_data).generate_football_field_data()["EV/EBITDA"] == {
        "low": 0,
        "mid": 0,
        "high": 0,
    }


def test_peer_valuation_uses_only_positive_finite_multiples() -> None:
    result = ValuationEngine(
        {"ebitda": 1000, "shares_outstanding": 100},
        {
            "LOW": {"ev_ebitda": 8},
            "HIGH": {"ev_ebitda": 12},
            "BAD": None,
            "MISSING": {},
            "ZERO": {"ev_ebitda": 0},
            "NEGATIVE": {"ev_ebitda": -2},
            "INFINITE": {"ev_ebitda": float("inf")},
        },
    ).calculate_peer_comparison_valuation()
    assert (result.low_estimate, result.target_price, result.high_estimate) == (72, 90, 108)
    assert result.confidence == 0.6
    assert result.assumptions["peer_count"] == 2
    assert result.assumptions["avg_multiple"] == 10
    assert result.assumptions["min_multiple"] == 8
    assert result.assumptions["max_multiple"] == 12
    assert result.description == "EBITDA x 10.0x average peer EV/EBITDA (2 peers)."


def test_peer_valuation_single_peer_and_missing_inputs() -> None:
    peers: dict[str, object] = {"ONE": {"ev_ebitda": 10}}
    result = ValuationEngine({"ebitda": 1000, "shares_outstanding": 100}, peers).calculate_peer_comparison_valuation()
    assert (result.low_estimate, result.target_price, result.high_estimate, result.confidence) == (90, 90, 90, 0.5)
    cases: list[tuple[dict[str, object], dict[str, object]]] = [
        ({"ebitda": 1000, "shares_outstanding": 100}, {}),
        ({"ebitda": 0, "shares_outstanding": 100}, peers),
        ({"ebitda": 1000, "shares_outstanding": 0}, peers),
    ]
    for financial, peer_data in cases:
        missing = ValuationEngine(financial, peer_data).calculate_peer_comparison_valuation()
        assert missing.confidence == missing.target_price == missing.low_estimate == missing.high_estimate == 0
        assert missing.description == "Missing EBITDA, shares outstanding, or peer EV/EBITDA multiples."


def test_valuation_explanation_compares_methods_and_current_price() -> None:
    text = ValuationEngine(
        {"current_price": 60, "ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [10]},
        {"PEER": {"ev_ebitda": 20}},
    ).explain_valuation_differences()
    assert text.startswith("EV/EBITDA $90.00 vs Peer Comparison $180.00. Spread $90.00 (50% of the highest target).")
    assert "vs current price: EV/EBITDA implies +50%, Peer Comparison implies +200%." in text
    assert "DCF reflects cash generation" in text


def test_valuation_explanation_handles_missing_price_and_no_methods() -> None:
    text = ValuationEngine(
        {"ebitda": 1000, "shares_outstanding": 100, "ev_ebitda_history": [10]}
    ).explain_valuation_differences()
    assert text.startswith("EV/EBITDA $90.00. Spread $0.00 (0% of the highest target).")
    assert "vs current price" not in text
    assert ValuationEngine({}).explain_valuation_differences() == (
        "Insufficient inputs for valuation: no method produced a usable target."
    )
