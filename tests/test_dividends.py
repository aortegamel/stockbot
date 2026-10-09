"""Live dividend fundamentals tests (app/services/sec_facts.py metric "dividends").

Seed an in-memory gateway stub through the real normalizers, then exercise the
live point-in-time path: full-year-duration annual history, contiguous
TTM windows, historical price isolation, current-date Yahoo valuation,
coverage-uncertainty status, future-known exclusion, and live fallback.
The edgar_client / valuation seams stay monkeypatched.

Warehouse-removal seam: live providers (SourceGateway + normalization + raw_archive) serve reads, nothing is persisted; a future warehouse slots in behind the gateway.
"""

import datetime as _dt
from collections.abc import Iterable

import pytest

from app import edgar_client, valuation
from app.domain.market.securities import TickerAlias
from app.normalization import (
    COMPANY_FACTS_PARSER_VERSION,
    DIVIDEND_PER_SHARE_CONCEPT,
    normalize_sec_company_facts,
    normalize_sec_tickers,
)
from app.policy import LOCAL_CONTEXT
from app.services import sec_facts
from app.tool_render import render_tool_result
from app.tools import TOOLS, execute_tool

KO_CIK = 21344
RETRIEVED_AT = "2026-08-01T00:00:00Z"
AS_OF = "2026-08-10"
STALE_AS_OF = "2026-09-05"
QUOTE_RETRIEVED_AT = "2026-08-10T12:00:00Z"


class _Gateway:
    """SourceGateway double: normalized rows in, PIT-filtered company_facts out."""

    def __init__(self) -> None:
        self.alias_rows: list[dict[str, object]] = []
        self.facts_by_cik: dict[int, dict[str, list[dict[str, object]]]] = {}

    def ticker_candidates(self, ticker: str, as_of: object = None) -> list[TickerAlias]:
        """All aliases for the ticker, unfiltered (PIT stays in resolve_ticker_aliases)."""
        del as_of
        want = str(ticker).strip().upper()
        out: list[TickerAlias] = []
        for row in self.alias_rows:
            if str(row.get("alias_value") or "").strip().upper() != want:
                continue
            security_id = row.get("security_id")
            out.append(
                TickerAlias(
                    alias_type=str(row.get("alias_type")),
                    alias_value=str(row.get("alias_value")),
                    entity_id=str(row.get("entity_id")),
                    security_id=str(security_id) if security_id else None,
                    source=str(row.get("source")),
                    valid_from=str(row.get("valid_from")) if row.get("valid_from") else None,
                    valid_to=str(row.get("valid_to")) if row.get("valid_to") else None,
                    known_at=str(row.get("known_at")) if row.get("known_at") else None,
                    retrieved_at=str(row.get("retrieved_at")) if row.get("retrieved_at") else None,
                )
            )
        return out

    def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
        """Full normalized dict, PIT-filtered to as_of like the live gateway."""
        facts = self.facts_by_cik.get(int(cik))
        if facts is None:
            return {"documents": [], "financial_facts": [], "securities": [], "dividend_events": []}
        out: dict[str, object] = {name: list(rows) for name, rows in facts.items()}
        if as_of is None:
            return out
        for name, rows in out.items():
            if isinstance(rows, list):
                out[name] = [
                    row for row in rows if isinstance(row, dict) and str(row.get("known_at") or "")[:10] <= as_of
                ]
        return out


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> _Gateway:
    """SourceGateway double behind sec_facts._gateway (no warehouse)."""
    gw = _Gateway()
    monkeypatch.setattr(sec_facts, "_gateway", lambda: gw)
    return gw


def _merge(gw: _Gateway, cik: int, datasets: dict[str, list[dict[str, object]]]) -> None:
    merged = dict(gw.facts_by_cik.get(cik, {}))
    for name, rows in datasets.items():
        merged[name] = list(merged.get(name, [])) + list(rows)
    gw.facts_by_cik[cik] = merged


def _seed_ticker(gw: _Gateway, cik: int, ticker: str) -> None:
    datasets = normalize_sec_tickers(
        {"0": {"cik_str": cik, "ticker": ticker, "title": f"{ticker} Corp"}},
        retrieved_at=RETRIEVED_AT,
        content_hash=f"tickers-{cik}",
    )
    gw.alias_rows.extend(datasets.get("entity_aliases", []))


def _div_fact(val: float, start: str, end: str, fy: int, fp: str, filed: str, accn: str) -> dict[str, object]:
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "filed": filed}


def _seed_dividends(
    gw: _Gateway,
    cik: int,
    facts: list[dict[str, object]],
    distractors: Iterable[tuple[str, list[dict[str, object]]]] = (),
) -> None:
    units = {"USD/shares": list(facts)}
    for unit, extra in distractors:
        units.setdefault(unit, []).extend(extra)
    payload = {
        "cik": cik,
        "entityName": f"CIK{cik}",
        "facts": {
            "us-gaap": {DIVIDEND_PER_SHARE_CONCEPT: {"units": units}},
        },
    }
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash=f"div-facts-{cik}",
        source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        source_record_id=f"cik{cik:010d}",
    )
    _merge(gw, cik, datasets)


# KO-style calendar fixture: restated Q1, YTD distractors, explicit Q4,
# one full-year fact, wrong-unit and paid-concept distractors, one next-year quarter.
_KO_FACTS = [
    _div_fact(0.50, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-04-29", "k1"),
    _div_fact(0.51, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-05-06", "k2"),
    _div_fact(1.02, "2025-01-01", "2025-06-30", 2025, "Q2", "2025-07-29", "k3"),  # 6-mo YTD
    _div_fact(0.51, "2025-04-01", "2025-06-30", 2025, "Q2", "2025-07-29", "k4"),
    _div_fact(1.53, "2025-01-01", "2025-09-30", 2025, "Q3", "2025-10-28", "k5"),  # 9-mo YTD
    _div_fact(0.51, "2025-07-01", "2025-09-30", 2025, "Q3", "2025-10-28", "k6"),
    _div_fact(0.51, "2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10", "k7"),
    _div_fact(2.04, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "kfy"),
    _div_fact(0.54, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "k8"),
]
_KO_DISTRACTORS = [
    ("USD", [_div_fact(999.0, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-04-29", "x1")]),
]
_KO_PAID = {
    "cik": KO_CIK,
    "entityName": "KO",
    "facts": {
        "us-gaap": {
            "CommonStockDividendsPaid": {
                "units": {
                    "USD": [
                        _div_fact(999.0, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-04-29", "x2"),
                    ]
                }
            },
        }
    },
}


def _seed_ko(gw: _Gateway) -> None:
    _seed_ticker(gw, KO_CIK, "KO")
    _seed_dividends(gw, KO_CIK, _KO_FACTS, _KO_DISTRACTORS)


# NVDA-style dates: Q4 exists only as an FY total, so the trailing quarter
# must be derived as FY_total - YTD_through_Q3 (mirrors test_sec_facts NVDA).
# Annual history comes from the FY fact alone (period_end year 2026).
_NVDA_DIV_FACTS = [
    _div_fact(0.50, "2025-01-27", "2025-04-27", 2026, "Q1", "2025-05-28", "d1"),
    _div_fact(0.51, "2025-01-27", "2025-04-27", 2026, "Q1", "2025-05-28", "d2"),
    _div_fact(1.02, "2025-01-27", "2025-07-27", 2026, "Q2", "2025-08-27", "d3"),  # 6-mo YTD
    _div_fact(0.51, "2025-04-28", "2025-07-27", 2026, "Q2", "2025-08-27", "d4"),
    _div_fact(1.53, "2025-01-27", "2025-10-26", 2026, "Q3", "2025-11-19", "d5"),  # 9-mo YTD
    _div_fact(0.51, "2025-07-28", "2025-10-26", 2026, "Q3", "2025-11-19", "d6"),
    _div_fact(2.04, "2025-01-27", "2026-01-25", 2026, "FY", "2026-02-25", "d7"),  # Q4 only as FY
    _div_fact(0.54, "2026-01-26", "2026-04-26", 2027, "Q1", "2026-05-27", "d8"),
]


def _fy_fact(year: int, total: float) -> dict[str, object]:
    """One full-year-duration fact per year."""
    return _div_fact(total, f"{year}-01-01", f"{year}-12-31", year, "FY", f"{year + 1}-02-10", f"y{year}FY")


def _quarter_facts(year: int, total: float) -> list[dict[str, object]]:
    """Four contiguous quarterly facts splitting an annual total."""
    ends = [
        (f"{year}-01-01", f"{year}-03-31", "Q1"),
        (f"{year}-04-01", f"{year}-06-30", "Q2"),
        (f"{year}-07-01", f"{year}-09-30", "Q3"),
        (f"{year}-10-01", f"{year}-12-31", "Q4"),
    ]
    return [
        _div_fact(round(total / 4, 4), start, end, year, fp, f"{year + 1}-02-10", f"q{year}{fp}")
        for start, end, fp in ends
    ]


_ANNUAL_TOTALS = {
    2015: 1.00,
    2016: 1.05,
    2017: 1.10,
    2018: 1.16,
    2019: 1.22,
    2020: 1.25,
    2021: 1.32,
    2022: 1.50,
    2023: 1.70,
    2024: 1.90,
    2025: 2.00,
}


def _seed_growth(gw: _Gateway, skip_years: Iterable[int] = ()) -> None:
    _seed_ticker(gw, KO_CIK, "KO")
    facts = [_fy_fact(year, total) for year, total in _ANNUAL_TOTALS.items() if year not in skip_years]
    if 2025 not in skip_years:
        facts.extend(_quarter_facts(2025, _ANNUAL_TOTALS[2025]))
        # Keep the TTM window fresh for an August 2026 query: trailing
        # Q3/Q4 2025 + Q1/Q2 2026 stays contiguous at 2.00.
        facts.append(_div_fact(0.50, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "q2026Q1"))
        facts.append(_div_fact(0.50, "2026-04-01", "2026-06-30", 2026, "Q2", "2026-07-29", "q2026Q2"))
    _seed_dividends(gw, KO_CIK, facts)


def _fail_on_price(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ticker: str) -> dict[str, object]:
        raise AssertionError("historical dividend query must not call Yahoo")

    monkeypatch.setattr(valuation, "get_live_quote", _boom)


def test_concept_parity_and_parser_bump() -> None:
    assert DIVIDEND_PER_SHARE_CONCEPT == "CommonStockDividendsPerShareDeclared"
    assert sec_facts.DIVIDEND_PER_SHARE_CONCEPT == DIVIDEND_PER_SHARE_CONCEPT
    assert edgar_client._DIVIDEND_CONCEPT == DIVIDEND_PER_SHARE_CONCEPT
    assert COMPANY_FACTS_PARSER_VERSION == "sec-companyfacts-v6"


def test_wrong_unit_and_paid_concept_rejected() -> None:
    out = normalize_sec_company_facts(
        _KO_PAID, retrieved_at=RETRIEVED_AT, content_hash="paid", source_url="u", source_record_id="r"
    )
    assert out["financial_facts"] == []


def test_historical_never_calls_yahoo(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ko(gateway)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["data_source"] == "live"
    # Restated Q1 wins; YTD rows never double-count the TTM sum.
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["price_source"] is None
    assert result["price_retrieved_at"] is None
    assert result["dividend_status"] == "paying"
    assert result["annual_history"] == [{"fiscal_year": 2025, "dividend_per_share": 2.04}]


def test_derived_q4_uses_fy_minus_ytd(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(gateway, KO_CIK, _NVDA_DIV_FACTS)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["dividend_status"] == "paying"
    assert result["annual_history"] == [{"fiscal_year": 2026, "dividend_per_share": 2.04}]


def test_exact_gap_growth_and_cagr(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_growth(gateway)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] == 2.00
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["dividend_status"] == "paying"
    assert result["growth_1y"] == 0.0526
    assert result["growth_3y_cagr"] == 0.1006
    assert result["growth_5y_cagr"] == 0.0986
    assert result["growth_10y_cagr"] == 0.0718
    annual_history = result["annual_history"]
    assert isinstance(annual_history, list)
    years = [row["fiscal_year"] for row in annual_history]
    assert years == sorted(years, reverse=True) == list(range(2025, 2014, -1))


def test_missing_comparison_year_yields_null(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_growth(gateway, skip_years=(2022,))
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["growth_1y"] == 0.0526
    assert result["growth_3y_cagr"] is None
    assert result["growth_5y_cagr"] == 0.0986
    assert result["growth_10y_cagr"] == 0.0718


def test_current_date_valuation(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ko(gateway)
    monkeypatch.setattr(sec_facts, "_today", lambda: _dt.date(2026, 8, 10))

    def _quote_65(ticker: str) -> dict[str, object]:
        return {"price": 65.0, "retrieved_at": QUOTE_RETRIEVED_AT}

    monkeypatch.setattr(
        valuation,
        "get_live_quote",
        _quote_65,
    )
    result = sec_facts.get_fundamentals("KO", "dividends")
    assert result["data_source"] == "live"
    assert result["dividend_status"] == "paying"
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["ttm_dividend_yield"] == 0.0318
    assert result["price"] == 65.0
    assert result["price_source"] == "yahoo"
    assert result["price_retrieved_at"] == QUOTE_RETRIEVED_AT


@pytest.mark.parametrize("price", [None, 0, -3.0])
def test_current_date_unusable_price_yields_null(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch, price: float | None
) -> None:
    _seed_ko(gateway)
    monkeypatch.setattr(sec_facts, "_today", lambda: _dt.date(2026, 8, 10))

    def _quote_price(ticker: str) -> dict[str, object]:
        return {"price": price, "retrieved_at": QUOTE_RETRIEVED_AT}

    monkeypatch.setattr(
        valuation,
        "get_live_quote",
        _quote_price,
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["price_source"] is None
    assert result["price_retrieved_at"] is None
    assert result["annual_history"] == [{"fiscal_year": 2025, "dividend_per_share": 2.04}]


def test_shifted_fy_metadata_uses_period_end_year(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(2.04, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "s1"),
            _div_fact(2.10, "2025-01-01", "2025-12-31", 2026, "FY", "2026-03-01", "s2"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["dividend_status"] == "unknown"
    assert result["ttm_dividend_per_share"] is None
    assert result["annual_history"] == [{"fiscal_year": 2025, "dividend_per_share": 2.10}]


def test_non_contiguous_ttm_reports_unknown_with_null_ttm(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(0.50, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-04-29", "g1"),
            _div_fact(0.50, "2025-04-01", "2025-06-30", 2025, "Q2", "2025-07-29", "g2"),
            _div_fact(0.50, "2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10", "g3"),
            _div_fact(0.50, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "g4"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["dividend_status"] == "unknown"
    assert result["ttm_dividend_per_share"] is None
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["price_source"] is None
    assert result["price_retrieved_at"] is None


def test_future_known_restatements_excluded(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            # Quarterly Q1 restated after the query date.
            _div_fact(0.51, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-05-06", "f1"),
            _div_fact(9.99, "2025-01-01", "2025-03-31", 2025, "Q1", "2026-09-01", "f2"),
            _div_fact(0.51, "2025-04-01", "2025-06-30", 2025, "Q2", "2025-07-29", "f3"),
            _div_fact(0.51, "2025-07-01", "2025-09-30", 2025, "Q3", "2025-10-28", "f4"),
            _div_fact(0.51, "2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10", "f5"),
            _div_fact(0.54, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "f6"),
            # Full-year restated after the query date.
            _div_fact(2.04, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "f7"),
            _div_fact(9.99, "2025-01-01", "2025-12-31", 2025, "FY", "2026-09-01", "f8"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["annual_history"] == [{"fiscal_year": 2025, "dividend_per_share": 2.04}]


def test_live_rows_used_never_calls_yahoo(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(0.50, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-04-29", "p1"),
        ],
    )

    def _boom(ticker: str, metric: str, **kwargs: object) -> dict[str, object]:
        raise AssertionError("live rows must not call edgar_client fallback")

    monkeypatch.setattr(sec_facts.edgar_client, "get_fundamentals", _boom)
    _fail_on_price(monkeypatch)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["data_source"] == "live"
    assert result["dividend_status"] == "unknown"
    assert result["ttm_dividend_per_share"] is None
    assert result["ttm_dividend_yield"] is None
    assert result["annual_history"] == []
    assert result["growth_1y"] is None


def test_stale_2021_quarters_report_unknown_without_yahoo(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _boom(ticker: str) -> dict[str, object]:
        calls.append(ticker)
        raise AssertionError("stale dividend query must not call Yahoo")

    monkeypatch.setattr(valuation, "get_live_quote", _boom)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [*_quarter_facts(2021, 2.00), _fy_fact(2021, 2.00)],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=STALE_AS_OF)
    assert result["data_source"] == "live"
    assert result["dividend_status"] == "unknown"
    assert result["ttm_dividend_per_share"] is None
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["price_source"] is None
    assert result["price_retrieved_at"] is None
    assert result["annual_history"] == [{"fiscal_year": 2021, "dividend_per_share": 2.00}]
    assert calls == []


def _seed_chain_ending(gw: _Gateway, latest_end: _dt.date) -> None:
    ends = [latest_end - _dt.timedelta(days=91 * i) for i in (3, 2, 1, 0)]
    facts = []
    for i, end in enumerate(ends):
        start = end - _dt.timedelta(days=90)
        filed = end + _dt.timedelta(days=10)
        facts.append(
            _div_fact(
                0.50,
                start.isoformat(),
                end.isoformat(),
                end.year,
                f"Q{(end.month - 1) // 3 + 1}",
                filed.isoformat(),
                f"b{i}",
            )
        )
    _seed_ticker(gw, KO_CIK, "KO")
    _seed_dividends(gw, KO_CIK, facts)


@pytest.mark.parametrize(
    ("age_days", "expected_status", "expected_ttm"),
    [(180, "paying", 2.00), (181, "unknown", None)],
)
def test_dividend_recency_boundary_days(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch, age_days: int, expected_status: str, expected_ttm: float | None
) -> None:
    _fail_on_price(monkeypatch)
    requested = _dt.date.fromisoformat(AS_OF)
    _seed_chain_ending(gateway, requested - _dt.timedelta(days=age_days))
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["dividend_status"] == expected_status
    assert result["ttm_dividend_per_share"] == expected_ttm
    assert result["price_retrieved_at"] is None


def test_live_cached_candidate_finalization_removes_private_and_nulls_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    fake_store: dict[str, object] = {}

    class _FakeCache:
        def get(self, key: str, ttl: float | None = None) -> object | None:
            return fake_store.get(key)

        def set(self, key: str, value: object) -> None:
            fake_store[key] = value

    monkeypatch.setattr(edgar_client, "cache", _FakeCache())
    monkeypatch.setattr(edgar_client, "_ensure_init", lambda: None)

    def _boom(ticker: str) -> dict[str, object]:
        raise AssertionError("stale live candidate must not call Yahoo")

    monkeypatch.setattr(valuation, "get_live_quote", _boom)
    starts = ["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"]
    ends = ["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31"]
    rows = [
        {
            "concept": "us-gaap:" + edgar_client._DIVIDEND_CONCEPT,
            "period_start": start,
            "period_end": end,
            "value": 0.50,
            "fiscal_year": 2021,
            "fiscal_period": f"Q{q}",
        }
        for q, (start, end) in enumerate(zip(starts, ends), start=1)
    ]
    rows.append(
        {
            "concept": "us-gaap:" + edgar_client._DIVIDEND_CONCEPT,
            "period_start": "2021-01-01",
            "period_end": "2021-12-31",
            "value": 2.00,
            "fiscal_year": 2021,
            "fiscal_period": "FY",
        }
    )

    class _Facts:
        def to_dataframe(self):
            return pd.DataFrame(rows)

    class _Company:
        ticker: str

        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def get_facts(self):
            return _Facts()

    monkeypatch.setattr(edgar_client, "Company", _Company)
    result = edgar_client.get_fundamentals("KO", "dividends", include_dividend_price=True)
    assert "_latest_dividend_period_end" not in result
    assert result["dividend_status"] == "unknown"
    assert result["ttm_dividend_per_share"] is None
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["price_source"] is None
    assert result["price_retrieved_at"] is None
    # Cached facts-only candidate retains the private period end for next call.
    cached = fake_store["fundamentals:KO:dividends"]
    assert isinstance(cached, dict)
    assert cached["_latest_dividend_period_end"] == "2021-12-31"
    assert cached["ttm_dividend_per_share"] == 2.00


def test_live_fallback_when_gateway_empty(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, KO_CIK, "KO")  # resolved entity, no dividend facts
    calls: list[object] = []

    def _live(ticker: str, metric: str, include_dividend_price: bool = True) -> dict[str, object]:
        calls.append((ticker, metric, include_dividend_price))
        return {
            "ticker": ticker,
            "dividend_status": "paying",
            "ttm_dividend_per_share": 2.07,
            "ttm_dividend_yield": None,
            "price": None,
            "price_source": None,
            "price_retrieved_at": None,
            "growth_1y": None,
            "growth_3y_cagr": None,
            "growth_5y_cagr": None,
            "growth_10y_cagr": None,
            "annual_history": [{"fiscal_year": 2025, "dividend_per_share": 2.04}],
            "source": edgar_client._DIVIDEND_SOURCE,
        }

    monkeypatch.setattr(sec_facts.edgar_client, "get_fundamentals", _live)
    result = sec_facts.get_fundamentals("KO", "dividends")
    assert result["data_source"] == "live"
    assert "requested_as_of" not in result
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["ttm_dividend_yield"] is None
    assert result["price"] is None
    assert result["dividend_status"] == "paying"
    assert result["safety"] is None
    assert calls == [("KO", "dividends", True)]


def test_explicit_as_of_empty_gateway_is_pit_unavailable(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, KO_CIK, "KO")
    calls: list[object] = []

    def _live(ticker: str, metric: str, include_dividend_price: bool = True) -> dict[str, object]:
        calls.append((ticker, metric))
        raise AssertionError("live must not be called with explicit as_of")

    monkeypatch.setattr(sec_facts.edgar_client, "get_fundamentals", _live)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["error_type"] == "pit_data_unavailable"
    assert calls == []


def test_gateway_empty_without_edgar_rows_is_pit_unavailable(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_ticker(gateway, KO_CIK, "KO")

    def _boom(ticker: str, metric: str, include_dividend_price: bool = True) -> dict[str, object]:
        raise AssertionError("gateway rows decide emptiness; edgar_client must not be consulted")

    monkeypatch.setattr(sec_facts.edgar_client, "get_fundamentals", _boom)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["error_type"] == "pit_data_unavailable"


def test_unknown_metric_still_errors(gateway: _Gateway) -> None:
    result = sec_facts.get_fundamentals("KO", "bogus", as_of=AS_OF)
    assert result["error"] == "Unknown metric 'bogus'"


def test_tool_schema_and_dispatch(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    schema = None
    for item in TOOLS:
        fn = item.get("function")
        assert isinstance(fn, dict)
        if fn.get("name") == "get_fundamentals":
            schema = fn
            break
    assert schema is not None
    params = schema.get("parameters")
    assert isinstance(params, dict)
    props = params.get("properties")
    assert isinstance(props, dict)
    metric = props.get("metric")
    assert isinstance(metric, dict)
    enum_vals = metric.get("enum")
    assert isinstance(enum_vals, list)
    assert "dividends" in enum_vals
    _seed_ko(gateway)
    result = execute_tool(
        "get_fundamentals",
        {"ticker": "KO", "metric": "dividends", "as_of": AS_OF},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert result["source"] == "sec"
    assert result["metric"] == "dividends"
    assert result["data_source"] == "live"
    assert result["ticker"] == "KO"
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["ttm_dividend_yield"] is None
    assert result["dividend_status"] == "paying"
    assert set(result) >= {
        "ticker",
        "ttm_dividend_per_share",
        "ttm_dividend_yield",
        "growth_1y",
        "growth_3y_cagr",
        "growth_5y_cagr",
        "growth_10y_cagr",
        "annual_history",
        "dividend_status",
        "price",
        "price_source",
        "price_retrieved_at",
    }


def test_render_annual_history(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ko(gateway)
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    text = render_tool_result(result)
    assert text.startswith("KO dividends [live] as of 2026-08-10")
    assert "ttm_dividend_per_share: 2.07" in text
    assert "- 2025: dividend 2.04" in text


def test_semiannual_payer_ttm(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(0.90, "2025-07-01", "2025-12-31", 2025, "Q2", "2026-01-15", "s1"),
            _div_fact(0.90, "2026-01-01", "2026-06-30", 2026, "Q2", "2026-07-15", "s2"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] is None
    assert result["dividend_status"] == "unknown"


def test_annual_payer_ttm(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(2.50, "2025-07-01", "2026-06-30", 2026, "FY", "2026-07-15", "a1"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] is None
    assert result["dividend_status"] == "unknown"


def test_monthly_payer_ttm(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    months = [
        ("2025-08-01", "2025-08-31"),
        ("2025-09-01", "2025-09-30"),
        ("2025-10-01", "2025-10-31"),
        ("2025-11-01", "2025-11-30"),
        ("2025-12-01", "2025-12-31"),
        ("2026-01-01", "2026-01-31"),
        ("2026-02-01", "2026-02-28"),
        ("2026-03-01", "2026-03-31"),
        ("2026-04-01", "2026-04-30"),
        ("2026-05-01", "2026-05-31"),
        ("2026-06-01", "2026-06-30"),
        ("2026-07-01", "2026-07-31"),
    ]
    _seed_dividends(
        gateway, KO_CIK, [_div_fact(0.20, s, e, 2026, "M", "2026-08-01", f"m{i}") for i, (s, e) in enumerate(months)]
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] is None
    assert result["dividend_status"] == "unknown"


def test_stale_annual_reports_unknown(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(2.50, "2024-01-01", "2024-12-31", 2024, "FY", "2025-02-10", "old1"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] is None
    assert result["dividend_status"] == "unknown"


def test_fy_aggregate_plus_incomplete_quarters_is_not_ttm(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(2.04, "2025-01-01", "2025-12-31", 2025, "FY", "2026-02-10", "y2025FY"),
            _div_fact(0.51, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "q1"),
            _div_fact(0.51, "2026-04-01", "2026-06-30", 2026, "Q2", "2026-07-28", "q2"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] is None
    assert result["dividend_status"] == "unknown"
    assert result["annual_history"] == [{"fiscal_year": 2025, "dividend_per_share": 2.04}]
