"""Offline tests for the live SEC fact service (app/services/sec_facts.py).

Normalized fixtures are built through the real normalizers and served
through a SourceGateway double (ticker_candidates/company_facts), then
get_fundamentals/get_xbrl_facts envelopes are exercised: the NVDA
live parity (ttm 6.53), true-filed_at restatement ordering, as_of gating,
alias and dispatch behavior.  Live edgar_client fallbacks are monkeypatched.
"""

from collections.abc import Iterable, Mapping
from datetime import date

import pytest

from app.domain.market.securities import TickerAlias
from app.normalization import normalize_sec_company_facts, normalize_sec_tickers
from app.policy import LOCAL_CONTEXT
from app.services import sec_facts
from app.tool_render import render_tool_result
from app.tools import TOOLS, execute_tool

NVDA_CIK = 1045810
RETRIEVED_AT = "2026-08-01T00:00:00Z"


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


def _seed_ticker(gw: _Gateway, cik: int, ticker: str, retrieved_at: str = RETRIEVED_AT) -> None:
    datasets = normalize_sec_tickers(
        {"0": {"cik_str": cik, "ticker": ticker, "title": f"{ticker} Corp"}},
        retrieved_at=retrieved_at,
        content_hash=f"tickers-{cik}",
    )
    gw.alias_rows.extend(datasets.get("entity_aliases", []))


def _seed_facts(
    gw: _Gateway,
    cik: int,
    # SEC companyfacts JSON shapes (external boundary); normalized inward below.
    diluted: Iterable[Mapping[str, object]] = (),
    basic: Iterable[Mapping[str, object]] = (),
    shares: Iterable[Mapping[str, object]] = (),
) -> None:
    facts: dict[str, object] = {}
    payload: dict[str, object] = {"cik": cik, "entityName": f"CIK{cik}", "facts": facts}
    if shares:
        facts["dei"] = {
            "EntityCommonStockSharesOutstanding": {"units": {"shares": list(shares)}},
        }
    us_gaap: dict[str, object] = {}
    if diluted:
        us_gaap["EarningsPerShareDiluted"] = {"units": {"USD/shares": list(diluted)}}
    if basic:
        us_gaap["EarningsPerShareBasic"] = {"units": {"USD/shares": list(basic)}}
    if us_gaap:
        facts["us-gaap"] = us_gaap
    datasets = normalize_sec_company_facts(
        payload,
        retrieved_at=RETRIEVED_AT,
        content_hash=f"facts-{cik}",
        source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        source_record_id=f"cik{cik:010d}",
    )
    merged = dict(gw.facts_by_cik.get(cik, {}))
    for name, rows in datasets.items():
        merged[name] = list(merged.get(name, [])) + list(rows)
    gw.facts_by_cik[cik] = merged


def _eps_fact(val: float, start: str, end: str, fy: int, fp: str, filed: str, accn: str):
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "filed": filed}


def _shares_fact(val: int, end: str, filed: str, accn: str):
    return {"end": end, "val": val, "accn": accn, "filed": filed}


# NVDA fact set mirroring tests/test_edgar_metrics.py::_EPS_ROWS (the live
# algorithm's fixture): quarterly, YTD, FY-only Q4, restated Q3.
_NVDA_FACTS = [
    _eps_fact(0.76, "2025-01-27", "2025-04-27", 2026, "Q1", "2025-05-28", "a1"),
    _eps_fact(0.77, "2025-01-27", "2025-04-27", 2026, "Q1", "2025-05-28", "a2"),
    _eps_fact(1.84, "2025-01-27", "2025-07-27", 2026, "Q2", "2025-08-27", "a3"),  # 6-month YTD
    _eps_fact(1.08, "2025-04-28", "2025-07-27", 2026, "Q2", "2025-08-27", "a4"),
    _eps_fact(3.14, "2025-01-27", "2025-10-26", 2026, "Q3", "2025-11-19", "a5"),  # 9-month YTD
    _eps_fact(1.30, "2025-07-28", "2025-10-26", 2026, "Q3", "2025-11-19", "a6"),
    _eps_fact(4.90, "2025-01-27", "2026-01-25", 2026, "FY", "2026-02-25", "a7"),  # Q4 only as FY
    _eps_fact(2.39, "2026-01-26", "2026-04-26", 2027, "Q1", "2026-05-27", "a8"),
    _eps_fact(2.40, "2026-01-26", "2026-04-26", 2027, "Q1", "2026-05-27", "a9"),
]
_NVDA_BASIC = (0.77, 2.40)


def _seed_nvda(gw: _Gateway) -> None:
    _seed_ticker(gw, NVDA_CIK, "NVDA")
    diluted = [f for f in _NVDA_FACTS if f["val"] not in _NVDA_BASIC]
    basic = [f for f in _NVDA_FACTS if f["val"] in _NVDA_BASIC]
    _seed_facts(gw, NVDA_CIK, diluted=diluted, basic=basic)


def _live_eps(ticker: str = "NVDA"):
    return {
        "ticker": ticker,
        "quarterly_eps": [
            {"fiscal_year": "2026", "fiscal_period": "Q1", "eps_diluted": 0.76, "period_end": "2025-04-27"},
        ],
        "ttm_eps_diluted": 0.76,
        "source": "SEC EDGAR company facts (Basic & Diluted EPS)",
    }


def _live_shares(ticker: str = "NVDA"):
    return {
        "ticker": ticker,
        "shares_outstanding": 999.0,
        "as_of": "2026-01-01",
        "source": "SEC EDGAR company facts",
        "note": "SEC-reported shares outstanding, not public float",
    }


def test_eps_store_parity_with_live_algorithm(gateway: _Gateway) -> None:
    _seed_nvda(gateway)
    result = sec_facts.get_fundamentals("NVDA", "eps", as_of="2026-08-10")
    assert result["data_source"] == "live"
    assert result["as_of_date"] == "2026-08-10"
    assert "requested_as_of" not in result
    assert result["row_count"] == 4  # tail(4): Q1'26 drops, mirroring live
    quarterly_eps = result["quarterly_eps"]
    assert isinstance(quarterly_eps, list)
    by_end = {q["period_end"]: q for q in quarterly_eps}
    assert set(by_end) == {"2025-07-27", "2025-10-26", "2026-01-25", "2026-04-26"}
    assert by_end["2025-07-27"]["eps_diluted"] == 1.08
    assert by_end["2025-10-26"]["eps_diluted"] == 1.30
    assert by_end["2026-01-25"]["eps_diluted"] == 1.76  # derived: 4.90 - 3.14
    assert by_end["2026-04-26"]["eps_diluted"] == 2.39
    assert by_end["2026-04-26"]["eps_basic"] == 2.40
    assert result["ttm_eps_diluted"] == 6.53  # matches the live fixture exactly
    assert "ttm_eps_basic" not in result
    assert result["source"] == "sec"
    assert result["source_label"] == "SEC EDGAR company facts (Basic & Diluted EPS)"


def test_eps_store_restatement_uses_true_filed_at_not_fiscal_year_proxy(gateway: _Gateway) -> None:
    """The fiscal-year proxy (live _dedup_latest) would pick the fy2027
    version; the store must pick the true latest filed_at (fy2026, 1.30)."""
    _seed_ticker(gateway, NVDA_CIK, "NVDA")
    diluted = [f for f in _NVDA_FACTS if f["val"] not in _NVDA_BASIC and f["val"] != 1.30]
    diluted.append(_eps_fact(1.35, "2025-07-28", "2025-10-26", 2027, "Q3", "2025-11-19", "a6"))
    diluted.append(_eps_fact(1.30, "2025-07-28", "2025-10-26", 2026, "Q3", "2026-03-10", "a10"))
    basic = [f for f in _NVDA_FACTS if f["val"] in _NVDA_BASIC]
    _seed_facts(gateway, NVDA_CIK, diluted=diluted, basic=basic)

    result = sec_facts.get_fundamentals("NVDA", "eps", as_of="2026-08-10")
    quarterly_eps = result["quarterly_eps"]
    assert isinstance(quarterly_eps, list)
    by_end = {q["period_end"]: q for q in quarterly_eps}
    assert by_end["2025-10-26"]["eps_diluted"] == 1.30  # latest filed_at wins
    assert result["ttm_eps_diluted"] == 6.53


def test_eps_store_as_of_gating_excludes_later_restatement(gateway: _Gateway) -> None:
    """A restatement filed after as_of must not change an earlier as_of."""
    _seed_ticker(gateway, NVDA_CIK, "NVDA")
    diluted = [
        _eps_fact(1.00, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-05-01", "b1"),
        _eps_fact(1.10, "2026-04-01", "2026-06-30", 2026, "Q2", "2026-08-01", "b2"),
        _eps_fact(1.20, "2026-07-01", "2026-09-30", 2026, "Q3", "2026-11-01", "b3"),
        _eps_fact(1.30, "2026-10-01", "2026-12-31", 2026, "Q4", "2026-12-15", "b4"),
        _eps_fact(1.25, "2026-07-01", "2026-09-30", 2026, "Q3", "2027-01-10", "b5"),
    ]
    _seed_facts(gateway, NVDA_CIK, diluted=diluted)

    earlier = sec_facts.get_fundamentals("NVDA", "eps", as_of="2026-12-20")
    quarterly_eps = earlier["quarterly_eps"]
    assert isinstance(quarterly_eps, list)
    by_end = {q["period_end"]: q for q in quarterly_eps}
    assert by_end["2026-09-30"]["eps_diluted"] == 1.20
    assert earlier["ttm_eps_diluted"] == 4.60

    later = sec_facts.get_fundamentals("NVDA", "eps", as_of="2027-02-01")
    quarterly_eps = later["quarterly_eps"]
    assert isinstance(quarterly_eps, list)
    by_end = {q["period_end"]: q for q in quarterly_eps}
    assert by_end["2026-09-30"]["eps_diluted"] == 1.25
    assert later["ttm_eps_diluted"] == 4.65


def test_derive_q4_uses_restated_fy_total():
    """A restated FY total (later filed_at, accession tiebreak) drives derived Q4."""
    from datetime import date

    concept = sec_facts.DILUTED_EPS_CONCEPT
    fy_end = date(2026, 1, 25)

    def _row(start: str, end: str, val: float, filed: str, accn: str) -> sec_facts.FinancialFactRow:
        return {
            "concept": concept,
            "period_start": start,
            "period_end": end,
            "value": val,
            "filed_at": filed,
            "accession": accn,
            "known_at": "",
            "source_url": None,
            "fiscal_year": None,
            "fiscal_period": None,
        }

    rows = [
        _row("2025-01-27", "2025-10-26", 3.14, "2025-11-19", "a5"),
        _row("2025-01-27", "2026-01-25", 4.90, "2026-02-25", "a7"),
        _row("2025-01-27", "2026-01-25", 4.95, "2026-03-10", "b1"),
        _row("2025-01-27", "2026-01-25", 5.00, "2026-03-10", "b2"),
    ]
    derived = sec_facts._derive_q4_from_facts(rows, concept, fy_end)
    assert derived is not None
    assert derived["value"] == pytest.approx(5.00 - 3.14, abs=1e-9)
    assert derived["accession"] == "b2"
    assert derived["fiscal_period"] == "Q4"


# ---------------------------------------------------------------------------
# Live fallback path
# ---------------------------------------------------------------------------


def test_eps_live_fallback_when_store_empty(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, NVDA_CIK, "NVDA")  # resolved entity, no facts
    calls: list[tuple[str, str]] = []

    def _fake_get(ticker: str, metric: str):
        calls.append((ticker, metric))
        return _live_eps()

    monkeypatch.setattr(
        sec_facts.edgar_client,
        "get_fundamentals",
        _fake_get,
    )
    result = sec_facts.get_fundamentals("NVDA", "eps", as_of="2025-01-15")
    assert result["error_type"] == "pit_data_unavailable"
    assert calls == []


def test_eps_omitted_as_of_defaults_to_today_live(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, NVDA_CIK, "NVDA")

    def _fake_get(ticker: str, metric: str):
        return _live_eps(ticker)

    monkeypatch.setattr(
        sec_facts.edgar_client,
        "get_fundamentals",
        _fake_get,
    )
    result = sec_facts.get_fundamentals("NVDA", "eps")
    assert result["data_source"] == "live"
    assert result["as_of_date"] == date.today().isoformat()  # noqa: DTZ011 - trading-calendar local date has no tz meaning
    assert "requested_as_of" not in result


def test_ambiguous_ticker_skips_store_never_guesses(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, 1, "NVDA")
    _seed_ticker(gateway, 2, "NVDA")
    _seed_facts(gateway, 1, diluted=[_eps_fact(1.0, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-05-01", "c1")])
    calls: list[str] = []

    def _fake_get(ticker: str, metric: str):
        calls.append(metric)
        return _live_eps()

    monkeypatch.setattr(
        sec_facts.edgar_client,
        "get_fundamentals",
        _fake_get,
    )
    result = sec_facts.get_fundamentals("NVDA", "eps", as_of="2026-08-10")
    assert result["error_type"] == "pit_data_unavailable"
    assert calls == []


def test_invalid_as_of_returns_tool_argument_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(ticker: str, metric: str):
        return _live_eps(ticker)

    monkeypatch.setattr(
        sec_facts.edgar_client,
        "get_fundamentals",
        _fake_get,
    )
    result = sec_facts.get_fundamentals("NVDA", "eps", as_of="2025/01/15")
    assert result["error"] == "as_of must be a date in YYYY-MM-DD format"
    assert result["error_type"] == "invalid_tool_arguments"


# ---------------------------------------------------------------------------
# Shares outstanding live path
# ---------------------------------------------------------------------------


def test_shares_outstanding_store_latest_period_wins(gateway: _Gateway) -> None:
    _seed_ticker(gateway, NVDA_CIK, "NVDA")
    _seed_facts(
        gateway,
        NVDA_CIK,
        shares=[
            _shares_fact(1000, "2025-10-26", "2025-11-19", "s1"),
            _shares_fact(1050, "2025-10-26", "2026-03-10", "s2"),  # restated
            _shares_fact(1100, "2026-01-25", "2026-02-25", "s3"),
        ],
    )
    result = sec_facts.get_fundamentals("NVDA", "shares_outstanding", as_of="2026-08-10")
    assert result["data_source"] == "live"
    assert result["shares_outstanding"] == 1100  # newest period_end; filed_at breaks restatement ties
    assert result["as_of"] == "2026-01-25"
    assert result["accession"] == "s3"
    assert result["filed_at"] == "2026-02-25"
    assert result["known_at"] == "2026-02-25"
    note = result["note"]
    assert isinstance(note, str)
    assert "not public float" in note
    assert result["source_label"] == "SEC EDGAR company facts"


def test_shares_float_alias_preserved(gateway: _Gateway) -> None:
    _seed_ticker(gateway, NVDA_CIK, "NVDA")
    _seed_facts(gateway, NVDA_CIK, shares=[_shares_fact(1000, "2025-10-26", "2025-11-19", "s1")])
    result = sec_facts.get_fundamentals("NVDA", "shares_float", as_of="2026-08-10")
    assert result["metric"] == "shares_outstanding"
    assert result["shares_outstanding"] == 1000
    assert result["data_source"] == "live"


def test_shares_live_fallback_empty_store(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_ticker(gateway, NVDA_CIK, "NVDA")

    def _fake_get(ticker: str, metric: str):
        return _live_shares(ticker)

    monkeypatch.setattr(
        sec_facts.edgar_client,
        "get_fundamentals",
        _fake_get,
    )
    result = sec_facts.get_fundamentals("NVDA", "shares_outstanding", as_of="2026-08-10")
    assert result["error_type"] == "pit_data_unavailable"


def test_company_facts_failure_returns_pit_unavailable(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    """A down companyfacts provider with an explicit as_of keeps the pit_data_unavailable envelope."""
    from app import data_sources as _ds

    _seed_ticker(gateway, NVDA_CIK, "NVDA")

    def _down(cik: int) -> object:
        del cik
        raise RuntimeError("SEC down")

    monkeypatch.setattr("edgar.entity.entity_facts.download_company_facts_from_sec", _down)
    monkeypatch.setattr(_ds._sec, "ensure_identity", lambda: None)
    with pytest.raises(RuntimeError, match="SEC down"):
        _ds.SourceGateway().company_facts(NVDA_CIK)
    result = sec_facts.get_fundamentals("NVDA", "shares_outstanding", as_of="2026-08-10")
    assert result["error_type"] == "pit_data_unavailable"


# ---------------------------------------------------------------------------
# Always-live metrics + XBRL envelope
# ---------------------------------------------------------------------------


def test_balance_sheet_and_overview_are_live_with_requested_as_of_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fake(ticker: str, metric: str):
        calls.append(metric)
        return (
            {
                "ticker": ticker,
                "balance_sheet": {"totalAssets": 1.0},
                "source": "SEC EDGAR financials",
            }
            if metric == "balance_sheet"
            else {
                "ticker": ticker,
                "name": "NVDA",
                "cik": "0001045810",
                "industry": "Semiconductors",
                "source": "SEC EDGAR",
            }
        )

    monkeypatch.setattr(sec_facts.edgar_client, "get_fundamentals", _fake)
    bs = sec_facts.get_fundamentals("NVDA", "balance_sheet", as_of="2025-01-15")
    assert bs["error_type"] == "pit_data_unavailable"
    ov = sec_facts.get_fundamentals("NVDA", "overview", as_of="2025-01-15")
    assert ov["error_type"] == "pit_data_unavailable"
    assert calls == []
    live_bs = sec_facts.get_fundamentals("NVDA", "balance_sheet")
    assert live_bs["data_source"] == "live"
    assert live_bs["balance_sheet"] == {"totalAssets": 1.0}
    assert live_bs["source_label"] == "SEC EDGAR financials"
    assert "requested_as_of" not in live_bs


def test_get_xbrl_facts_always_live_enveloped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_facts(ticker: str, concept: str):
        return {
            "ticker": ticker,
            "concept_searched": concept,
            "matching_concepts": [
                {"concept": "Revenue", "value": 1.0, "period_end": "2026-01-25", "fiscal_period": "FY"}
            ],
            "count": 1,
            "source": "SEC EDGAR XBRL facts",
        }

    monkeypatch.setattr(sec_facts.edgar_client, "get_xbrl_facts", _fake_facts)
    result = sec_facts.get_xbrl_facts("NVDA", "Revenue")
    assert result["source"] == "sec"
    assert result["metric"] == "concept"
    assert result["data_source"] == "live"
    assert result["row_count"] == 1
    assert result["returned_count"] == 1
    assert result["truncated"] is False
    matching_concepts = result["matching_concepts"]
    assert isinstance(matching_concepts, list)
    first_concept = matching_concepts[0]
    assert isinstance(first_concept, dict)
    assert first_concept["concept"] == "Revenue"
    assert result["source_label"] == "SEC EDGAR XBRL facts"
    assert "source" not in first_concept


# ---------------------------------------------------------------------------
# Tool schema + dispatch + render
# ---------------------------------------------------------------------------


def _is_get_fundamentals_schema(item: object) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("function"), dict)
        and item["function"].get("name") == "get_fundamentals"
    )


def test_tool_schema_and_dispatch(gateway: _Gateway) -> None:
    # TOOLS is untyped nested tool-schema data (app/tools.py); validate the
    # get_fundamentals schema shape at this boundary before asserting on it.
    candidates = [item for item in TOOLS if _is_get_fundamentals_schema(item)]
    assert len(candidates) == 1
    function = candidates[0]["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    as_of_schema = properties["as_of"]
    assert isinstance(as_of_schema, dict)
    assert as_of_schema["type"] == "string"
    description = as_of_schema["description"]
    assert isinstance(description, str)
    assert "store-backed for eps/shares_outstanding" in description

    _seed_nvda(gateway)
    result = execute_tool(
        "get_fundamentals",
        {"ticker": "NVDA", "metric": "eps", "as_of": "2026-08-10"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert result["source"] == "sec"
    assert result["data_source"] == "live"
    assert result["ttm_eps_diluted"] == 6.53


def test_render_sec_facts_envelope(gateway: _Gateway) -> None:
    _seed_nvda(gateway)
    result = sec_facts.get_fundamentals("NVDA", "eps", as_of="2026-08-10")
    text = render_tool_result(result)
    assert text.startswith("NVDA eps [live] as of 2026-08-10")
    assert "rows: 4" in text
    assert "ttm_eps_diluted: 6.53" in text
    assert "2026 Q2 (period end 2025-07-27): diluted 1.08" in text
    assert "2027 Q1 (period end 2026-04-26): diluted 2.39 | basic 2.4" in text
    assert "source_label: SEC EDGAR company facts (Basic & Diluted EPS)" in text


def test_now_stamped_alias_unresolved_at_past_as_of(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    """Now-known ticker identity stays unknown at a past as_of; no archive-all fetch."""
    _seed_ticker(gateway, NVDA_CIK, "NVDA", retrieved_at="2026-08-20T12:00:00Z")
    _seed_facts(gateway, NVDA_CIK, shares=[_shares_fact(1000, "2026-08-01", "2026-08-02", "s1")])

    def _boom(ticker: str, metric: str):
        raise AssertionError(f"live fallback must not run for unknown identity: {ticker} {metric}")

    monkeypatch.setattr(sec_facts.edgar_client, "get_fundamentals", _boom)
    result = sec_facts.get_fundamentals("NVDA", "shares_outstanding", as_of="2026-08-10")
    assert result["error_type"] == "pit_data_unavailable"
