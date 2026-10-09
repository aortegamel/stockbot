"""Past/present/future-declared dividend events (app/services/sec_facts.py).

Isolation reuses the gateway-stub seeding pattern from tests/test_dividends.py:
every test seeds one ticker plus dividend facts and dividend_events rows into
an in-memory SourceGateway double, then queries get_fundamentals as_of a
fixed date with Yahoo stubbed out.

Warehouse-removal seam: live providers (SourceGateway + normalization + raw_archive) serve reads, nothing is persisted; a future warehouse slots in behind the gateway.
"""

import pytest

from app import valuation
from app.domain.market import ids
from app.domain.market.securities import TickerAlias
from app.normalization import (
    DIVIDEND_PER_SHARE_CONCEPT,
    normalize_sec_company_facts,
    normalize_sec_tickers,
)
from app.services import sec_facts

KO_CIK = 21344
RETRIEVED_AT = "2026-08-01T00:00:00Z"
AS_OF = "2026-08-10"


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


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> _Gateway:
    """SourceGateway double behind sec_facts._gateway (no warehouse)."""
    gw = _Gateway()
    monkeypatch.setattr(sec_facts, "_gateway", lambda: gw)
    return gw


def _div_fact(val: float, start: str, end: str, fy: int, fp: str, filed: str, accn: str) -> dict[str, object]:
    return {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "filed": filed}


def _seed_dividends(gw: _Gateway, cik: int, facts: list[dict[str, object]]) -> None:
    payload = {
        "cik": cik,
        "entityName": f"CIK{cik}",
        "facts": {
            "us-gaap": {DIVIDEND_PER_SHARE_CONCEPT: {"units": {"USD/shares": list(facts)}}},
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


def _seed_quarters(gw: _Gateway) -> None:
    """Four contiguous quarters (TTM 2.10) so the live path serves events."""
    _seed_ticker(gw, KO_CIK, "KO")
    _seed_dividends(
        gw,
        KO_CIK,
        [
            _div_fact(0.51, "2025-07-01", "2025-09-30", 2025, "Q3", "2025-10-28", "q3"),
            _div_fact(0.51, "2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10", "q4"),
            _div_fact(0.54, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "q1"),
            _div_fact(0.54, "2026-04-01", "2026-06-30", 2026, "Q2", "2026-07-28", "q2"),
        ],
    )


def _event(
    cik: int,
    event_id: str,
    amount: float | None,
    *,
    decl: str | None = None,
    record: str | None = None,
    pay: str | None = None,
    known: str = "2026-08-01T00:00:00Z",
    filed: str | None = None,
    accn: str = "0000123456",
    source_type: str = "structured_xbrl",
    dtype: str = "regular",
) -> dict[str, object]:
    return {
        "dividend_event_id": event_id,
        "entity_id": ids.sec_entity_id(cik),
        "security_id": ids.sec_security_id(cik),
        "ticker": "KO",
        "amount_per_share": amount,
        "currency": "USD",
        "dividend_type": dtype,
        "declaration_date": decl,
        "record_date": record,
        "payment_date": pay,
        "ex_dividend_date": None,
        "ex_dividend_date_source": "unknown",
        "status": "unknown",
        "source_form": "10-Q",
        "accession": accn,
        "filed_at": filed or known,
        "known_at": known,
        "source_url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/doc.htm",
        "source_concept": "DividendsPayableAmountPerShare",
        "source_type": source_type,
        "evidence_excerpt": None,
        "content_hash": f"h-{event_id}",
        "parser_version": "sec-companyfacts-v5",
    }


def _seed_events(gw: _Gateway, rows: list[dict[str, object]]) -> None:
    merged = dict(gw.facts_by_cik.get(KO_CIK, {}))
    merged["dividend_events"] = list(merged.get("dividend_events", [])) + list(rows)
    gw.facts_by_cik[KO_CIK] = merged


def _fail_on_price(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ticker: str) -> dict[str, object]:
        raise AssertionError("historical dividend query must not call Yahoo")

    monkeypatch.setattr(valuation, "get_live_quote", _boom)


def test_upcoming_vs_paid_split(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK, "ev-paid", 0.51, decl="2026-06-15", record="2026-06-30", pay="2026-07-01", accn="0000000001"
            ),
            _event(
                KO_CIK, "ev-next", 0.54, decl="2026-08-01", record="2026-08-29", pay="2026-09-15", accn="0000000002"
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    next_declared = result["next_declared_dividend"]
    assert isinstance(next_declared, dict)
    past = result["past_events"]
    assert isinstance(past, list)
    annual = result["annual_history"]
    assert isinstance(annual, list)
    assert result["last_dividend"] == {"amount_per_share": 0.51, "payment_date": "2026-07-01", "type": "regular"}
    assert result["next_declared_dividend"] == {
        "amount_per_share": 0.54,
        "declaration_date": "2026-08-01",
        "record_date": "2026-08-29",
        "payment_date": "2026-09-15",
        "status": "upcoming",
        "source_url": next_declared["source_url"],
        "accession": "0000000002",
    }
    assert [e["payment_date"] for e in past] == ["2026-07-01"]
    assert result["events_coverage"] == "structured_only"
    assert result["row_count"] == len(annual)


def test_future_declaration_invisible_at_earlier_as_of(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                "ev-future",
                0.54,
                decl="2026-08-20",
                record="2026-09-15",
                pay="2026-10-01",
                known="2026-09-01T00:00:00Z",
                accn="0000000003",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of="2026-08-15")
    assert result["next_declared_dividend"] is None
    assert result["last_dividend"] is None
    assert result["past_events"] == []
    assert result["events_coverage"] == "no_structured_events"


def test_duplicate_accessions_dedup_to_one(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    event_id = ids.sec_dividend_event_id(KO_CIK, 0.54, "2026-08-29", "2026-09-15", "regular")
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                event_id,
                0.54,
                decl="2026-08-01",
                record="2026-08-29",
                pay="2026-09-15",
                known="2026-08-01T00:00:00Z",
                accn="0000000004",
            ),
            _event(
                KO_CIK,
                event_id,
                0.54,
                decl="2026-08-01",
                record="2026-08-29",
                pay="2026-09-15",
                known="2026-08-02T00:00:00Z",
                accn="0000000005",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["past_events"] == []
    next_declared = result["next_declared_dividend"]
    assert isinstance(next_declared, dict)
    assert next_declared["amount_per_share"] == 0.54


def test_amended_amount_supersedes_via_latest_known_at(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    old_id = ids.sec_dividend_event_id(KO_CIK, 0.50, "2026-08-29", "2026-09-15", "regular")
    new_id = ids.sec_dividend_event_id(KO_CIK, 0.54, "2026-08-29", "2026-09-15", "regular")
    assert old_id != new_id
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                old_id,
                0.50,
                decl="2026-07-15",
                record="2026-08-29",
                pay="2026-09-15",
                known="2026-07-20T00:00:00Z",
                accn="0000000006",
            ),
            _event(
                KO_CIK,
                new_id,
                0.54,
                decl="2026-08-01",
                record="2026-08-29",
                pay="2026-09-15",
                known="2026-08-02T00:00:00Z",
                accn="0000000007",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    next_declared = result["next_declared_dividend"]
    assert isinstance(next_declared, dict)
    assert next_declared["amount_per_share"] == 0.54
    assert next_declared["accession"] == "0000000007"


def test_incomplete_event_excluded_from_last_next(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK, "ev-full", 0.54, decl="2026-08-01", record="2026-08-29", pay="2026-09-15", accn="0000000008"
            ),
            _event(KO_CIK, "ev-partial", 0.54, decl="2026-08-01", record=None, pay=None, accn="0000000009"),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    next_declared = result["next_declared_dividend"]
    assert isinstance(next_declared, dict)
    assert next_declared["payment_date"] == "2026-09-15"
    assert result["last_dividend"] is None
    assert result["past_events"] == []
    past = result["past_events"]
    assert isinstance(past, list)
    assert all(e["accession"] != "0000000009" for e in past)


def test_restated_q1_still_wins_ttm_while_events_classify(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_dividends(
        gateway,
        KO_CIK,
        [
            _div_fact(0.50, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-04-29", "k1"),
            _div_fact(0.51, "2025-01-01", "2025-03-31", 2025, "Q1", "2025-05-06", "k2"),
            _div_fact(0.51, "2025-04-01", "2025-06-30", 2025, "Q2", "2025-07-29", "k4"),
            _div_fact(0.51, "2025-07-01", "2025-09-30", 2025, "Q3", "2025-10-28", "k6"),
            _div_fact(0.51, "2025-10-01", "2025-12-31", 2025, "Q4", "2026-02-10", "k7"),
            _div_fact(0.54, "2026-01-01", "2026-03-31", 2026, "Q1", "2026-04-28", "k8"),
        ],
    )
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK, "ev-ko-next", 0.53, decl="2026-07-20", record="2026-08-29", pay="2026-09-15", accn="0000000010"
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["ttm_dividend_per_share"] == 2.07
    assert result["dividend_status"] == "paying"
    next_declared = result["next_declared_dividend"]
    assert isinstance(next_declared, dict)
    assert next_declared["amount_per_share"] == 0.53
    assert result["last_dividend"] is None


def test_upcoming_excluded_from_past(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK, "ev-paid-2", 0.51, decl="2026-06-15", record="2026-06-30", pay="2026-07-01", accn="0000000011"
            ),
            _event(
                KO_CIK, "ev-up-2", 0.54, decl="2026-08-01", record="2026-08-29", pay="2026-09-15", accn="0000000012"
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    past = result["past_events"]
    assert isinstance(past, list)
    assert len(past) == 1
    assert past[0]["payment_date"] == "2026-07-01"


def test_amendment_canonicalizes_to_latest(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    old_id = ids.sec_dividend_event_id(KO_CIK, 0.50, "2026-06-30", "2026-07-01", "regular")
    new_id = ids.sec_dividend_event_id(KO_CIK, 0.54, "2026-06-30", "2026-07-01", "regular")
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                old_id,
                0.50,
                decl="2026-06-15",
                record="2026-06-30",
                pay="2026-07-01",
                known="2026-08-01T00:00:00Z",
                accn="0000000013",
            ),
            _event(
                KO_CIK,
                new_id,
                0.54,
                decl="2026-06-15",
                record="2026-06-30",
                pay="2026-07-01",
                known="2026-08-05T00:00:00Z",
                accn="0000000014",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    last = result["last_dividend"]
    assert isinstance(last, dict)
    assert last["amount_per_share"] == 0.54
    past = result["past_events"]
    assert isinstance(past, list)
    assert len(past) == 1
    assert past[0]["amount_per_share"] == 0.54


def test_event_only_surfaces_without_aggregate(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_ticker(gateway, KO_CIK, "KO")
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK, "ev-only", 0.51, decl="2026-06-15", record="2026-06-30", pay="2026-07-01", accn="0000000015"
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["data_source"] == "live"
    assert result["dividend_status"] == "unknown"
    assert result["ttm_dividend_per_share"] is None
    last = result["last_dividend"]
    assert isinstance(last, dict)
    assert last["amount_per_share"] == 0.51
    assert result["row_count"] == 0


def test_analysis_wired_through_live(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK, f"ev-q{i}", 0.50, record=f"2025-0{i + 1}-15" if i < 9 else None, pay=pay, accn=f"000000002{i}"
            )
            for i, pay in enumerate(["2025-04-15", "2025-07-15", "2025-10-15", "2026-01-15", "2026-04-15"])
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)
    assert result["payment_cadence"] == "quarterly"
    assert result["growth_basis"] == "total_aggregates"
    assert "growth_trend" in result


def test_coverage_matrix(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                "ev-text",
                0.51,
                decl="2026-06-15",
                record="2026-06-30",
                pay="2026-07-01",
                accn="0000000030",
                source_type="filing_text",
            ),
        ],
    )
    assert sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)["events_coverage"] == "text_only"
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                "ev-xbrl",
                0.54,
                decl="2026-06-15",
                record="2026-06-29",
                pay="2026-07-01",
                accn="0000000031",
                source_type="structured_xbrl",
            ),
        ],
    )
    assert sec_facts.get_fundamentals("KO", "dividends", as_of=AS_OF)["events_coverage"] == "structured_and_text"


def test_cross_source_duplicate_merges_to_one_payment(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                "ev-xbrl-dup",
                0.54,
                decl="2026-08-01",
                record="2026-08-29",
                pay="2026-09-15",
                accn="0000000040",
                source_type="structured_xbrl",
                dtype="unknown",
            ),
            _event(
                KO_CIK,
                "ev-text-dup",
                0.54,
                decl="2026-08-01",
                record="2026-08-29",
                pay="2026-09-15",
                accn="0000000041",
                source_type="filing_text",
                dtype="regular",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of="2026-09-20")
    past = result["past_events"]
    assert isinstance(past, list)
    assert len(past) == 1
    assert past[0]["amount_per_share"] == 0.54
    assert result["total_paid_per_share"] == 0.54
    assert result["regular_paid_per_share"] == 0.54
    assert result["events_coverage"] == "structured_and_text"


def test_unknown_matches_special_by_amount_not_first_bucket(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                "ev-reg",
                0.50,
                decl="2026-08-01",
                record="2026-09-01",
                pay="2026-09-15",
                accn="0000000050",
                source_type="filing_text",
                dtype="regular",
            ),
            _event(
                KO_CIK,
                "ev-spec",
                2.00,
                decl="2026-08-01",
                record="2026-09-01",
                pay="2026-09-15",
                accn="0000000051",
                source_type="filing_text",
                dtype="special",
            ),
            _event(
                KO_CIK,
                "ev-xbrl-unk",
                2.00,
                decl="2026-08-01",
                record="2026-09-01",
                pay="2026-09-15",
                accn="0000000052",
                source_type="structured_xbrl",
                dtype="unknown",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of="2026-09-20")
    past = result["past_events"]
    assert isinstance(past, list)
    assert len(past) == 2
    assert result["regular_paid_per_share"] == 0.50
    assert result["special_paid_per_share"] == 2.00
    assert result["total_paid_per_share"] == 2.50


def test_unknown_amount_match_ignores_row_order(gateway: _Gateway, monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_on_price(monkeypatch)
    _seed_quarters(gateway)
    _seed_events(
        gateway,
        [
            _event(
                KO_CIK,
                "ev-xbrl-unk",
                2.00,
                decl="2026-08-01",
                record="2026-09-01",
                pay="2026-09-15",
                known="2026-08-06T00:00:00Z",
                accn="0000000052",
                source_type="structured_xbrl",
                dtype="unknown",
            ),
            _event(
                KO_CIK,
                "ev-spec",
                2.00,
                decl="2026-08-01",
                record="2026-09-01",
                pay="2026-09-15",
                known="2026-08-02T00:00:00Z",
                accn="0000000051",
                source_type="filing_text",
                dtype="special",
            ),
            _event(
                KO_CIK,
                "ev-reg",
                0.50,
                decl="2026-08-01",
                record="2026-09-01",
                pay="2026-09-15",
                known="2026-08-04T00:00:00Z",
                accn="0000000050",
                source_type="filing_text",
                dtype="regular",
            ),
        ],
    )
    result = sec_facts.get_fundamentals("KO", "dividends", as_of="2026-09-20")
    past = result["past_events"]
    assert isinstance(past, list)
    assert len(past) == 2
    assert result["regular_paid_per_share"] == 0.50
    assert result["special_paid_per_share"] == 2.00
    assert result["total_paid_per_share"] == 2.50


# --- Phase 4: pure lifecycle analysis (app/services/dividend_analysis.py) ---
# These tests call the pure module directly with plain dicts: no gateway fixture,
# no network, no Yahoo. Running fixture-free IS the purity proof.

import datetime as _dt

from app.services import dividend_analysis as _lifecycle


def _paid(payment_date: str, amount: float, dtype: str = "regular") -> dict[str, object]:
    return {"payment_date": payment_date, "amount_per_share": amount, "dividend_type": dtype}


def _series(amounts: list[float], start: str, step_days: int) -> list[dict[str, object]]:
    day = _dt.date.fromisoformat(start)
    return [_paid((day + _dt.timedelta(days=i * step_days)).isoformat(), amount) for i, amount in enumerate(amounts)]


def test_lifecycle_quarterly_cadence_high_confidence() -> None:
    result = _lifecycle.analyze_dividends(paid_events=_series([0.50] * 5, "2025-01-15", 91), as_of="2026-02-01")
    assert result["payment_cadence"] == "quarterly"
    assert result["cadence_confidence"] == "high"
    assert result["cadence_basis"] == "payment_dates"


def test_lifecycle_monthly_cadence_high_confidence() -> None:
    result = _lifecycle.analyze_dividends(paid_events=_series([0.10] * 5, "2026-01-15", 30), as_of="2026-06-01")
    assert result["payment_cadence"] == "monthly"
    assert result["cadence_confidence"] == "high"


def test_lifecycle_two_intervals_is_medium_confidence() -> None:
    result = _lifecycle.analyze_dividends(paid_events=_series([0.50] * 3, "2025-01-15", 91), as_of="2025-08-01")
    assert result["payment_cadence"] == "quarterly"
    assert result["cadence_confidence"] == "medium"


def test_lifecycle_sparse_payments_are_unknown_cadence() -> None:
    one = _lifecycle.analyze_dividends(paid_events=[_paid("2025-01-15", 0.50)])
    assert one["payment_cadence"] == "unknown"
    assert one["cadence_confidence"] is None
    two = _lifecycle.analyze_dividends(paid_events=[_paid("2025-01-15", 0.50), _paid("2025-04-16", 0.50)])
    assert two["payment_cadence"] == "unknown"
    assert two["cadence_confidence"] is None


def test_lifecycle_increase() -> None:
    paid = _series([0.50, 0.50, 0.50, 0.54], "2025-01-15", 91)
    result = _lifecycle.analyze_dividends(paid_events=paid, as_of="2025-11-01")
    assert result["increase"] == {"pct": 0.08, "amount": 0.54, "date": paid[-1]["payment_date"]}
    assert result["cut"] is None
    assert result["freeze"] is None


def test_lifecycle_cut() -> None:
    paid = _series([0.54, 0.54, 0.54, 0.40], "2025-01-15", 91)
    result = _lifecycle.analyze_dividends(paid_events=paid, as_of="2025-11-01")
    assert result["cut"] == {
        "pct": round((0.40 - 0.54) / 0.54, 4),
        "prior": 0.54,
        "new": 0.40,
        "date": paid[-1]["payment_date"],
    }
    assert result["increase"] is None


def test_lifecycle_freeze_quarterly() -> None:
    result = _lifecycle.analyze_dividends(paid_events=_series([0.54] * 4, "2025-01-15", 91), as_of="2025-11-01")
    assert result["freeze"] == {"amount": 0.54, "count": 4}
    assert result["increase"] is None and result["cut"] is None


def test_lifecycle_freeze_semiannual_needs_two() -> None:
    result = _lifecycle.analyze_dividends(paid_events=_series([0.80] * 3, "2025-01-15", 182), as_of="2026-07-20")
    assert result["payment_cadence"] == "semiannual"
    assert result["cadence_confidence"] == "medium"
    assert result["freeze"] == {"amount": 0.80, "count": 3}


def test_lifecycle_specials_listed_separately_with_splits() -> None:
    paid = _series([0.50] * 4, "2025-01-15", 91)
    paid += [_paid("2025-12-15", 2.00, "special"), _paid("2025-12-15", 0.25, "supplemental")]
    result = _lifecycle.analyze_dividends(paid_events=paid, as_of="2026-01-10")
    assert result["specials"] == [
        {"amount": 2.00, "payment_date": "2025-12-15"},
        {"amount": 0.25, "payment_date": "2025-12-15"},
    ]
    assert result["regular_paid_per_share"] == 2.00
    assert result["special_paid_per_share"] == 2.25
    assert result["total_paid_per_share"] == 4.25
    assert result["increase"] is None and result["cut"] is None


def test_lifecycle_possible_suspension_is_flag_not_status() -> None:
    paid = _series([0.50] * 4, "2025-01-15", 91)
    stale = _lifecycle.analyze_dividends(paid_events=paid, as_of="2026-08-10")
    assert stale["possible_suspension"] is True
    assert "dividend_status" not in stale
    fresh = _lifecycle.analyze_dividends(paid_events=paid, as_of="2025-12-01")
    assert fresh["possible_suspension"] is False


def test_lifecycle_reinstatement_after_gap() -> None:
    paid = _series([0.50] * 4, "2023-01-15", 91)
    paid.append(_paid("2025-06-15", 0.50))
    result = _lifecycle.analyze_dividends(paid_events=paid, as_of="2025-07-01")
    assert result["reinstatement"] == {"date": "2025-06-15"}
    assert result["possible_suspension"] is False


def test_lifecycle_growth_trend() -> None:
    paid = [_paid("2025-01-15", 0.50)]
    assert (
        _lifecycle.analyze_dividends(paid_events=paid, growth={"growth_1y": 0.02, "growth_5y_cagr": 0.08})[
            "growth_trend"
        ]
        == "decelerating"
    )
    assert (
        _lifecycle.analyze_dividends(paid_events=paid, growth={"growth_1y": 0.12, "growth_5y_cagr": 0.08})[
            "growth_trend"
        ]
        == "accelerating"
    )
    assert (
        _lifecycle.analyze_dividends(paid_events=paid, growth={"growth_1y": 0.08, "growth_5y_cagr": 0.08})[
            "growth_trend"
        ]
        == "stable_or_unknown"
    )
    assert _lifecycle.analyze_dividends(paid_events=paid)["growth_trend"] == "stable_or_unknown"
    assert _lifecycle.analyze_dividends(paid_events=paid)["growth_basis"] == "total_aggregates"


def test_lifecycle_regular_basis_growth_needs_five_consecutive_years() -> None:
    five_years = [p for year in range(2021, 2026) for p in _series([0.50] * 4, f"{year}-01-15", 91)]
    full = _lifecycle.analyze_dividends(paid_events=five_years, as_of="2026-02-01")
    assert full["growth_basis"] == "total_aggregates"
    assert full["regular_basis_growth"] is not None
    growth = full["regular_basis_growth"]
    assert isinstance(growth, dict)
    assert growth["growth_1y"] == 0.0
    four_years = [p for year in range(2022, 2026) for p in _series([0.50] * 4, f"{year}-01-15", 91)]
    short = _lifecycle.analyze_dividends(paid_events=four_years, as_of="2026-02-01")
    assert short["regular_basis_growth"] is None
    assert short["growth_basis"] == "total_aggregates"
