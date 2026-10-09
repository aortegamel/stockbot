"""Offline tests for the research data refresh path.

Fetch is stubbed at the provider-transport layer (``research_data._edgar_get``
for SEC payloads, ``finra_client.ingestion_post_query`` for paged FINRA rows,
and a null ``research_data._gateway`` for enrichment); archive, normalize, and
the inline typed rows run for real against a tmp data root. Screen assertions
stub the screens seams (``_fetch_settlement_rows`` + ``_gateway``).
"""

import json
from pathlib import Path

import pytest

import cli
from app.analytics import screens
from app.domain.market.securities import TickerAlias
from app.normalization import (
    normalize_finra_short_interest,
    normalize_sec_company_facts,
    normalize_sec_tickers,
)
from app.services import research_data
from app.services.research_data import prepare_short_interest_data
from app.storage import raw_archive

TICKERS_PAYLOAD: dict[str, object] = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"},
    "1": {"cik_str": 2488, "ticker": "AMD", "title": "Advanced Micro Devices"},
}

SETTLEMENT = "2026-08-14"


def _no_sleep(seconds: float) -> None:
    return None


def _facts_payload(cik: int, val: int) -> dict[str, object]:
    return {
        "cik": cik,
        "entityName": f"CIK{cik}",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2026-08-01", "val": val, "accn": f"a{cik}", "filed": "2026-08-02"},
                        ]
                    }
                },
            }
        },
    }


def _finra_row(symbol: str, pos: int) -> dict[str, object]:
    return {
        "symbolCode": symbol,
        "issueName": symbol,
        "settlementDate": SETTLEMENT,
        "currentShortPositionQuantity": pos,
    }


def _page(
    rows: list[dict[str, object]], total: int, offset: int
) -> tuple[bytes, list[dict[str, object]], dict[str, str]]:
    return (
        json.dumps(rows).encode(),
        rows,
        {"record-total": str(total), "record-offset": str(offset), "record-limit": "1000"},
    )


class _NullGateway:
    """Enrichment-confirmation double: facts already arrived via ``_edgar_get``."""

    def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
        del cik, as_of
        return {}


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    get_script: list[bytes | Exception],
    page_script: list[tuple[bytes, list[dict[str, object]], dict[str, str]]],
) -> list[str]:
    """Scripted SEC payloads plus paged FINRA rows; records SEC URLs."""
    get_calls: list[str] = []

    def fake_get(url: str) -> bytes:
        get_calls.append(url)
        item = get_script.pop(0)  # IndexError when the script is exhausted
        if isinstance(item, Exception):
            raise item
        return item

    def fake_ingestion_post_query(
        group: str, dataset_name: str, payload: dict[str, object]
    ) -> tuple[bytes, list[dict[str, object]], dict[str, str]]:
        return page_script.pop(0)

    monkeypatch.setattr(research_data, "_edgar_get", fake_get)
    monkeypatch.setattr(
        research_data.finra_client,
        "ingestion_post_query",
        fake_ingestion_post_query,
    )
    monkeypatch.setattr(research_data, "_gateway", lambda: _NullGateway())
    monkeypatch.setattr(research_data.time, "sleep", _no_sleep)
    return get_calls


class _ScreenGateway:
    """Screens gateway double built from normalizer output (PIT-filtered)."""

    def __init__(
        self,
        tickers_payload: dict[str, object],
        facts_by_cik: dict[int, dict[str, object]],
    ) -> None:
        alias_rows = normalize_sec_tickers(
            tickers_payload,
            retrieved_at="2026-08-10T12:00:00Z",
            content_hash="tickers",
        )["entity_aliases"]
        self._aliases = [
            TickerAlias(
                alias_type=str(row.get("alias_type")),
                alias_value=str(row.get("alias_value")),
                entity_id=str(row.get("entity_id")),
                security_id=str(row.get("security_id")) if row.get("security_id") else None,
                source=str(row.get("source")),
                valid_from=str(row.get("valid_from")) if row.get("valid_from") else None,
                valid_to=str(row.get("valid_to")) if row.get("valid_to") else None,
                known_at=str(row.get("known_at")) if row.get("known_at") else None,
                retrieved_at=str(row.get("retrieved_at")) if row.get("retrieved_at") else None,
            )
            for row in alias_rows
            if isinstance(row, dict)
        ]
        self._facts: dict[int, dict[str, list[dict[str, object]]]] = {
            cik: normalize_sec_company_facts(
                payload,
                retrieved_at="2026-08-10T12:00:00Z",
                content_hash=f"facts-{cik}",
                source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                source_record_id=f"cik{cik:010d}",
            )
            for cik, payload in facts_by_cik.items()
        }

    def ticker_candidates(self, ticker: str, as_of: object) -> list[TickerAlias]:
        del as_of  # PIT stays in resolve_ticker_aliases; return all aliases unfiltered.
        return [a for a in self._aliases if a.alias_value == ticker.strip().upper()]

    def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
        datasets = self._facts.get(cik, {})
        out: dict[str, object] = {name: list(rows) for name, rows in datasets.items()}
        if as_of is not None:
            for name in ("financial_facts", "dividend_events"):
                rows = out.get(name)
                if isinstance(rows, list):
                    out[name] = [r for r in rows if str(r.get("known_at") or "")[:10] <= as_of]
        return out


def _install_screen(
    monkeypatch: pytest.MonkeyPatch,
    facts_by_cik: dict[int, dict[str, object]],
    finra_pairs: list[tuple[str, int]],
    tickers_payload: dict[str, object] | None = None,
) -> None:
    """Stub the screens seams: normalized FINRA rows plus a gateway double."""
    raw: list[dict[str, object]] = [
        {
            "symbolCode": symbol,
            "issueName": symbol,
            "settlementDate": SETTLEMENT,
            "currentShortPositionQuantity": pos,
        }
        for symbol, pos in finra_pairs
    ]
    typed = normalize_finra_short_interest(
        raw,
        settlement_date=SETTLEMENT,
        retrieved_at="2026-08-30T12:00:00Z",
        content_hash="screen-snapshot",
        source_url="u",
        source_record_id=f"otcMarket/consolidatedShortInterest:{SETTLEMENT}",
    )["short_interest"]
    rows = [row for row in typed if isinstance(row, dict)]

    def _settlement_rows(settlement: str) -> list[dict[str, object]]:
        return rows

    monkeypatch.setattr(screens, "_fetch_settlement_rows", _settlement_rows)
    monkeypatch.setattr(screens, "_gateway", lambda: _ScreenGateway(tickers_payload or TICKERS_PAYLOAD, facts_by_cik))
    monkeypatch.setattr(screens.time, "sleep", _no_sleep)


def test_refresh_data_offline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fetch -> archive -> normalize -> inline rows; screen reads live doubles."""
    get_script: list[bytes | Exception] = [
        json.dumps(TICKERS_PAYLOAD).encode(),
        json.dumps(_facts_payload(320193, 100)).encode(),
        json.dumps(_facts_payload(2488, 200)).encode(),
    ]
    page_script = [
        _page([_finra_row("AAPL", 20), _finra_row("AMD", 20)], 3, 0),
        _page([_finra_row("XOM", 5)], 3, 2),  # full snapshot: a non-universe symbol
    ]
    get_calls = _install_mocks(monkeypatch, get_script, page_script)

    summary = prepare_short_interest_data(SETTLEMENT, tickers=["AAPL", "AMD"], data_root=tmp_path)

    assert summary["unresolved_tickers"] == []
    sec_facts = summary["sec_facts"]
    assert isinstance(sec_facts, list)
    assert len(sec_facts) == 2
    assert {f["cik"] for f in sec_facts if isinstance(f, dict)} == {320193, 2488}
    assert all(f["normalized_rows"] == 3 for f in sec_facts if isinstance(f, dict))
    sec_tickers = summary["sec_tickers"]
    assert isinstance(sec_tickers, dict)
    assert "ticker_ciks" not in sec_tickers  # full map stays internal
    assert sec_tickers["ticker_count"] == 2
    finra = summary["finra"]
    assert isinstance(finra, dict)
    assert finra["rows"] == 3
    assert finra["normalized_rows"] == 3
    assert finra["written"] == 0
    assert get_calls == [
        "https://www.sec.gov/files/company_tickers.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000002488.json",
    ]

    assert raw_archive.find("sec", "company_tickers", "company_tickers", root=tmp_path / "raw") is not None
    assert raw_archive.find("sec", "cik0000320193", "companyfacts", root=tmp_path / "raw") is not None
    assert (
        raw_archive.find(
            "finra",
            "data_page",
            f"otcMarket/consolidatedShortInterest:{SETTLEMENT}:offset0",
            root=tmp_path / "raw",
        )
        is not None
    )
    assert (
        raw_archive.find(
            "finra",
            "data_page",
            f"otcMarket/consolidatedShortInterest:{SETTLEMENT}:offset2",
            root=tmp_path / "raw",
        )
        is not None
    )

    # Market-wide screen (P1 promise): the leaderboard is not universe-bound.
    _install_screen(
        monkeypatch,
        {320193: _facts_payload(320193, 100), 2488: _facts_payload(2488, 200)},
        [("AAPL", 20), ("AMD", 20), ("XOM", 5)],
    )
    result = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-30")
    entries = result["entries"]
    assert isinstance(entries, list)
    assert [e["ticker"] for e in entries if isinstance(e, dict)] == ["AAPL", "AMD"]  # 20/100 > 20/200
    coverage = result["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["finra_rows"] == 3
    assert coverage["eligible_rows"] == 2
    exclusions = coverage["exclusions"]
    assert isinstance(exclusions, dict)
    assert exclusions["unmapped_symbol"] == 1  # XOM


def test_cli_refresh_data_coverage_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    _install_screen(
        monkeypatch,
        {320193: _facts_payload(320193, 100), 2488: _facts_payload(2488, 200)},
        [("AAPL", 20), ("AMD", 20), ("XOM", 5)],
    )

    cli._cmd_refresh_data(SETTLEMENT, ["AAPL", "AMD"], [], data_root=str(tmp_path))
    out = capsys.readouterr().out

    assert "FINRA securities:             3" in out
    assert "Ticker mappings:              2" in out
    assert "Shares-outstanding coverage:  2" in out
    assert "Eligible screen universe:     2" in out
    assert "Coverage: 66.7%" in out
    assert "Leaderboard entries: ['AAPL', 'AMD']" in out


def test_unresolved_ticker_is_reported_not_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_script: list[bytes | Exception] = [
        json.dumps(TICKERS_PAYLOAD).encode(),
        json.dumps(_facts_payload(320193, 100)).encode(),
    ]
    page_script = [_page([_finra_row("AAPL", 20)], 1, 0)]
    get_calls = _install_mocks(monkeypatch, get_script, page_script)

    summary = prepare_short_interest_data(SETTLEMENT, tickers=["AAPL", "ZZZZ"], data_root=tmp_path)

    assert summary["unresolved_tickers"] == ["ZZZZ"]
    sec_facts = summary["sec_facts"]
    assert isinstance(sec_facts, list)
    assert len(sec_facts) == 1
    assert len(get_calls) == 2  # tickers + AAPL facts; no facts request for ZZZZ


def test_prepare_without_universe_skips_sec_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_script: list[bytes | Exception] = [json.dumps(TICKERS_PAYLOAD).encode()]
    page_script = [_page([_finra_row("AAPL", 20)], 1, 0)]
    get_calls = _install_mocks(monkeypatch, get_script, page_script)

    summary = prepare_short_interest_data(SETTLEMENT, data_root=tmp_path)

    assert summary["sec_facts"] == []
    assert len(get_calls) == 1  # SEC ticker universe only
    assert summary["unresolved_tickers"] == []
    sec_tickers = summary["sec_tickers"]
    assert isinstance(sec_tickers, dict)
    assert "ticker_ciks" not in sec_tickers
    assert sec_tickers["ticker_count"] == 2


def test_finra_missing_record_total_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_script: list[bytes | Exception] = [
        json.dumps(TICKERS_PAYLOAD).encode(),
        json.dumps(_facts_payload(320193, 100)).encode(),
    ]
    page_script: list[tuple[bytes, list[dict[str, object]], dict[str, str]]] = [
        (b"[]", [], {})
    ]  # no record-total header
    _install_mocks(monkeypatch, get_script, page_script)

    with pytest.raises(ValueError, match="Record-Total"):
        prepare_short_interest_data(SETTLEMENT, tickers=["AAPL"], data_root=tmp_path)


def test_enrichment_failure_does_not_block_finra_or_siblings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """P1: a failed facts request must never prevent the FINRA snapshot."""
    get_script: list[bytes | Exception] = [
        json.dumps(TICKERS_PAYLOAD).encode(),
        json.dumps(_facts_payload(320193, 100)).encode(),
        RuntimeError("HTTP 500"),
    ]
    page_script = [_page([_finra_row("AAPL", 20)], 1, 0)]
    get_calls = _install_mocks(monkeypatch, get_script, page_script)

    summary = prepare_short_interest_data(SETTLEMENT, tickers=["AAPL", "AMD"], data_root=tmp_path)

    finra = summary["finra"]
    assert isinstance(finra, dict)
    assert finra["rows"] == 1  # market-wide snapshot still landed
    sec_facts = summary["sec_facts"]
    assert isinstance(sec_facts, list)
    assert len(sec_facts) == 1  # AAPL enrichment succeeded
    assert summary["failed_enrichments"] == [
        {"ticker": "AMD", "cik": 2488, "error": "RuntimeError: HTTP 500"},
    ]
    assert get_calls == [
        "https://www.sec.gov/files/company_tickers.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000002488.json",
    ]


def test_cik_only_enrichment_failure_reports_null_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_script: list[bytes | Exception] = [
        json.dumps(TICKERS_PAYLOAD).encode(),
        RuntimeError("HTTP 500"),
    ]
    page_script = [_page([_finra_row("AAPL", 20)], 1, 0)]
    _install_mocks(monkeypatch, get_script, page_script)

    summary = prepare_short_interest_data(SETTLEMENT, ciks=[999999], data_root=tmp_path)

    assert summary["sec_facts"] == []
    assert summary["failed_enrichments"] == [
        {"ticker": None, "cik": 999999, "error": "RuntimeError: HTTP 500"},
    ]


def test_coverage_counters_truthful_with_invalid_short_interest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """P2 regression: invalid rows never reach mapping/shares checks, so the
    CLI must print the screen's stage counters, not derived complements."""
    _install_screen(
        monkeypatch,
        {320193: _facts_payload(320193, 100), 2488: _facts_payload(2488, 200)},
        [("AAPL", 20), ("AMD", 20), ("BAD", -1), ("XOM", 5)],
    )

    # One CLI run drives the live leaderboard; a replay of the screen then
    # recomputes the same ephemeral result deterministically.
    cli._cmd_refresh_data(SETTLEMENT, ["AAPL", "AMD"], [], data_root=str(tmp_path))
    out = capsys.readouterr().out

    result = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-30")
    coverage = result["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["finra_rows"] == 4
    assert coverage["valid_short_interest_rows"] == 3  # BAD excluded here
    assert coverage["mapped_rows"] == 2
    assert coverage["unambiguous_rows"] == 2
    assert coverage["common_equity_rows"] == 2
    assert coverage["shares_outstanding_rows"] == 2
    assert coverage["eligible_rows"] == 2
    assert coverage["exclusions"] == {
        "unmapped_symbol": 1,  # XOM
        "ambiguous_ticker_mapping": 0,
        "not_classified_common_equity": 0,
        "missing_shares_outstanding": 0,
        "invalid_short_interest": 1,  # BAD
        "conflicting_versions": 0,
    }
    entries = result["entries"]
    assert isinstance(entries, list)
    assert [e["ticker"] for e in entries if isinstance(e, dict)] == ["AAPL", "AMD"]

    # CLI prints the counters: BAD never inflated mapping/shares coverage
    # (the old derived formula would have printed "Ticker mappings: 3").
    assert "FINRA securities:             4" in out
    assert "Ticker mappings:              2" in out
    assert "Shares-outstanding coverage:  2" in out
    assert "Eligible screen universe:     2" in out
    assert "Coverage: 50.0%" in out
    assert "Leaderboard entries: ['AAPL', 'AMD']" in out


def test_refresh_sec_tickers_archives_and_returns_inline_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _tickers_get(url: str) -> bytes:
        return json.dumps(TICKERS_PAYLOAD).encode()

    monkeypatch.setattr(research_data, "_edgar_get", _tickers_get)

    result = research_data.refresh_sec_tickers(data_root=tmp_path)

    assert result["ticker_ciks"] == {"AAPL": 320193, "AMD": 2488}
    assert result["written"] == 0
    assert result["normalized_rows"] == 4  # entities + aliases
    assert raw_archive.find("sec", "company_tickers", "company_tickers", root=tmp_path / "raw") is not None


def test_ticker_candidates_reads_snapshot_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical ticker views resolve from archived company_tickers snapshots, not the live universe."""
    from datetime import UTC, datetime

    from app import data_sources as _ds
    from app.domain.market.identity import resolve_ticker_aliases

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    old = {"0": {"cik_str": 111, "ticker": "AAA", "title": "Old AAA"}}
    new = {"0": {"cik_str": 222, "ticker": "AAA", "title": "New AAA"}}
    raw_archive.archive(
        "sec",
        "company_tickers",
        "company_tickers",
        json.dumps(old).encode(),
        url="https://www.sec.gov/files/company_tickers.json",
        retrieved_at="2024-05-01T00:00:00Z",
        root=tmp_path / "raw",
    )
    raw_archive.archive(
        "sec",
        "company_tickers",
        "company_tickers",
        json.dumps(new).encode(),
        url="https://www.sec.gov/files/company_tickers.json",
        retrieved_at="2026-05-01T00:00:00Z",
        root=tmp_path / "raw",
    )
    mid = datetime(2025, 6, 1, tzinfo=UTC)
    aliases = _ds.SourceGateway().ticker_candidates("AAA", mid)
    assert [a.entity_id for a in aliases] == ["sec:cik:0000000111"]
    assert resolve_ticker_aliases("AAA", aliases, as_of=mid).entity_id == "sec:cik:0000000111"
    assert resolve_ticker_aliases("AAA", aliases, as_of=datetime(2023, 1, 1, tzinfo=UTC)).resolved is False
    late = datetime(2026, 6, 1, tzinfo=UTC)
    assert [a.entity_id for a in _ds.SourceGateway().ticker_candidates("AAA", late)] == ["sec:cik:0000000222"]


def test_refresh_sec_company_facts_archives_and_returns_inline_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = json.dumps(_facts_payload(320193, 100)).encode()

    def _facts_get(url: str) -> bytes:
        return payload

    monkeypatch.setattr(research_data, "_edgar_get", _facts_get)
    monkeypatch.setattr(research_data, "_gateway", lambda: _NullGateway())

    result = research_data.refresh_sec_company_facts(320193, data_root=tmp_path)

    assert result["cik"] == 320193
    assert result["written"] == 0
    assert result["normalized_rows"] == 3  # documents + financial_facts + securities
    assert raw_archive.find("sec", "cik0000320193", "companyfacts", root=tmp_path / "raw") is not None


def test_refresh_finra_short_interest_archives_pages_and_returns_inline_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = [_finra_row("AAPL", 20), _finra_row("AMD", 20)]
    content = json.dumps(rows).encode()

    def _fake_query(
        group: str, name: str, req: dict[str, object]
    ) -> tuple[bytes, list[dict[str, object]], dict[str, str]]:
        return (content, rows, {"record-total": "2"})

    monkeypatch.setattr(research_data.finra_client, "ingestion_post_query", _fake_query)
    monkeypatch.setattr(research_data.time, "sleep", _no_sleep)

    result = research_data.refresh_finra_short_interest(SETTLEMENT, data_root=tmp_path)

    assert result["settlement_date"] == SETTLEMENT
    assert result["rows"] == 2
    assert result["normalized_rows"] == 2
    assert result["written"] == 0
    assert (
        raw_archive.find(
            "finra",
            "data_page",
            f"otcMarket/consolidatedShortInterest:{SETTLEMENT}:offset0",
            root=tmp_path / "raw",
        )
        is not None
    )


def _replay_facts_payload(cik: int) -> dict[str, object]:
    """Companyfacts payload with a shares fact plus EPS facts (pre/post-EPS)."""
    return {
        "cik": cik,
        "entityName": f"CIK{cik}",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {"end": "2026-08-01", "val": 100, "accn": f"a{cik}", "filed": "2026-08-02"},
                        ]
                    }
                }
            },
            "us-gaap": {
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2026-08-01",
                                "val": 6.5,
                                "accn": f"a{cik}",
                                "filed": "2026-08-02",
                                "fy": 2026,
                                "fp": "Q3",
                            },
                        ]
                    }
                },
                "EarningsPerShareBasic": {
                    "units": {
                        "USD/shares": [
                            {
                                "end": "2026-08-01",
                                "val": 6.6,
                                "accn": f"a{cik}",
                                "filed": "2026-08-02",
                                "fy": 2026,
                                "fp": "Q3",
                            },
                        ]
                    }
                },
            },
        },
    }


def test_replay_sec_facts_normalizes_archive_deterministically(tmp_path: Path):
    """Offline replay: archived payloads normalize inline; rerun is identical,
    retrieved_at comes from the manifest not the clock."""
    payload = json.dumps(_replay_facts_payload(320193)).encode()
    retrieved_at = "2026-08-10T12:00:00Z"
    raw_archive.archive(
        "sec",
        "cik0000320193",
        "companyfacts",
        payload,
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        retrieved_at=retrieved_at,
        root=tmp_path / "raw",
    )

    first = research_data.replay_sec_facts_from_archive(data_root=tmp_path)
    assert first["archived_payloads"] == 1
    assert first["written_rows"] == 0
    assert first["normalized_rows"] == 5  # 3 facts + documents + securities
    assert first["failed"] == []

    # deterministic: retrieved_at from the archive manifest, not the wall clock
    rows = research_data.iter_archive_company_facts(320193, data_root=tmp_path)
    assert len(rows) == 3
    concepts = sorted(str(r.get("concept")) for r in rows)
    assert concepts == [
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "EntityCommonStockSharesOutstanding",
    ]
    eps_rows = [r for r in rows if str(r.get("concept")).startswith("EarningsPerShare")]
    assert all(r["unit"] == "USD/shares" for r in eps_rows)
    assert all(r["fiscal_year"] == 2026 and r["fiscal_period"] == "Q3" for r in eps_rows)
    assert all(r["retrieved_at"] == retrieved_at for r in rows)

    second = research_data.replay_sec_facts_from_archive(data_root=tmp_path)
    assert second["normalized_rows"] == first["normalized_rows"]


def test_replay_sec_facts_isolates_corrupt_payloads(tmp_path: Path):
    good = json.dumps(_replay_facts_payload(320193)).encode()
    raw_archive.archive(
        "sec",
        "cik0000320193",
        "companyfacts",
        good,
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        retrieved_at="2026-08-10T12:00:00Z",
        root=tmp_path / "raw",
    )
    raw_archive.archive(
        "sec",
        "cik0000000007",
        "companyfacts",
        b"{not valid json",
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000007.json",
        retrieved_at="2026-08-10T12:00:00Z",
        root=tmp_path / "raw",
    )
    summary = research_data.replay_sec_facts_from_archive(data_root=tmp_path)
    assert summary["archived_payloads"] == 2
    failed = summary["failed"]
    assert isinstance(failed, list)
    assert len(failed) == 1
    first_failure = failed[0]
    assert isinstance(first_failure, dict)
    assert first_failure["cik"] == "cik0000000007"
    failure_error = first_failure["error"]
    assert isinstance(failure_error, str)
    assert "JSONDecodeError" in failure_error
    normalized_rows = summary["normalized_rows"]
    assert isinstance(normalized_rows, (int, float))
    assert normalized_rows > 0  # the valid payload still processed


def test_iter_archive_company_facts_propagates_corrupt_payload(tmp_path: Path):
    raw_archive.archive(
        "sec",
        "cik0000320193",
        "companyfacts",
        json.dumps(_replay_facts_payload(320193)).encode(),
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        retrieved_at="2026-08-10T12:00:00Z",
        root=tmp_path / "raw",
    )
    raw_archive.archive(
        "sec",
        "cik0000320193",
        "companyfacts",
        b"{not valid json",
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        retrieved_at="2026-08-11T12:00:00Z",
        root=tmp_path / "raw",
    )

    with pytest.raises(ValueError):
        research_data.iter_archive_company_facts(320193, data_root=tmp_path)
