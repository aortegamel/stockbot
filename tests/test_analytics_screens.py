"""Tests for the live-provider short-interest screen.

The acceptance criteria under test:

- a screen result exposes settlement date, as-of timestamp, source records,
  coverage/exclusions, and calculation version;
- changing the requested as_of cannot use facts with a later known_at (the
  as-of regression test: a later filing cannot affect an earlier ranking);
- rerunning the same screen is deterministic;
- only eligible, classified equity securities are ranked.

Seeding builds provider data through the production normalizers into an
in-memory store; the screen's live seams (``screens._gateway``,
``screens._fetch_settlement_rows``, ``screens._probe_published_rows``) are
stubbed to serve it.  Nothing is persisted.

Warehouse-removal seam: live providers (SourceGateway + normalization + raw_archive) serve reads, nothing is persisted; a future warehouse slots in behind the gateway.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from app.analytics import screens
from app.data_sources import TickerAlias
from app.normalization import (
    normalize_finra_short_interest,
    normalize_sec_company_facts,
    normalize_sec_tickers,
)

SETTLEMENT = "2026-08-14"
# Prior cycle on the FINRA calendar canvas: the change slice discovers cycles
# by probing calendar candidates, so the prior seed must be a candidate for
# the as_of values under test (08-14 through 08-30).
PRIOR_SETTLEMENT = "2026-08-12"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


class _Seeds:
    """In-memory provider data built through production normalizers."""

    def __init__(self) -> None:
        self.aliases: list[TickerAlias] = []
        self.facts: dict[int, list[dict[str, object]]] = {}
        self.securities: dict[int, list[dict[str, object]]] = {}
        # settlement -> versions (each version is one normalized row list);
        # the live fetch serves the latest version per settlement.
        self.short_interest: dict[str, list[list[dict[str, object]]]] = {}


@pytest.fixture
def seeds() -> _Seeds:
    return _Seeds()


def _seed_tickers(
    seeds: _Seeds,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    retrieved_at: str = "2026-08-10T12:00:00Z",
    cik_start: int = 1,
) -> None:
    payload = {
        str(i): {"cik_str": cik, "ticker": ticker, "title": f"{ticker} Corp"}
        for i, (ticker, cik) in enumerate(zip(tickers, range(cik_start, cik_start + len(tickers))), start=0)
    }
    datasets = normalize_sec_tickers(
        payload,
        retrieved_at=retrieved_at,
        content_hash="tickers-hash",
    )
    for row in datasets.get("entity_aliases", []):
        if isinstance(row, dict):
            seeds.aliases.append(
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
            )


def _seed_facts(
    seeds: _Seeds, facts_by_cik: dict[int, list[dict[str, object]]], retrieved_at: str = "2026-08-10T12:00:00Z"
) -> None:
    for cik, facts in facts_by_cik.items():
        payload = {
            "cik": cik,
            "entityName": f"CIK{cik}",
            "facts": {
                "dei": {
                    "EntityCommonStockSharesOutstanding": {"units": {"shares": facts}},
                }
            },
        }
        datasets = normalize_sec_company_facts(
            payload,
            retrieved_at=retrieved_at,
            content_hash=f"facts-{cik}",
            source_url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
            source_record_id=f"cik{cik:010d}",
        )
        for row in datasets.get("financial_facts", []):
            if isinstance(row, dict):
                seeds.facts.setdefault(cik, []).append(row)
        for row in datasets.get("securities", []):
            if isinstance(row, dict):
                seeds.securities.setdefault(cik, []).append(row)


def _seed_short_interest(
    seeds: _Seeds,
    rows: list[dict[str, object]],
    retrieved_at: str = "2026-08-10T12:00:00Z",
    content_hash: str = "snapshot-hash",
    settlement: str = SETTLEMENT,
) -> None:
    datasets = normalize_finra_short_interest(
        rows,
        settlement_date=settlement,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        source_url="https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest",
        source_record_id=f"otcMarket/consolidatedShortInterest:{settlement}",
    )
    normalized = datasets.get("short_interest", [])
    seeds.short_interest.setdefault(settlement, []).append([row for row in normalized if isinstance(row, dict)])


def _seed_cycle(
    seeds: _Seeds, settlement_date: str, rows: list[dict[str, object]], retrieved_at: str = "2026-08-10T12:00:00Z"
) -> None:
    _seed_short_interest(
        seeds,
        rows,
        retrieved_at=retrieved_at,
        content_hash=f"snapshot-{settlement_date}",
        settlement=settlement_date,
    )


def _reclassify(seeds: _Seeds, cik: int, security_type: str, known_at: str, retrieved_at: str) -> None:
    """Later classification row for one entity's security (PIT-filtered by the fake)."""
    seeds.securities.setdefault(cik, []).append(
        {
            "security_id": f"sec:equity:{cik:010d}",
            "entity_id": f"sec:cik:{cik:010d}",
            "security_type": security_type,
            "ticker": None,
            "exchange": None,
            "source": "provider-test",
            "known_at": known_at,
            "retrieved_at": retrieved_at,
            "content_hash": "x",
            "parser_version": "t",
        }
    )


def _add_alias(seeds: _Seeds, alias_value: str, cik: int, known_at: str, retrieved_at: str) -> None:
    """Extra ticker alias row (e.g. a second CIK claiming the same ticker)."""
    seeds.aliases.append(
        TickerAlias(
            alias_type="ticker",
            alias_value=alias_value,
            entity_id=f"sec:cik:{cik:010d}",
            security_id=f"sec:equity:{cik:010d}",
            source="sec:company_tickers",
            valid_from=None,
            valid_to=None,
            known_at=known_at,
            retrieved_at=retrieved_at,
        )
    )


def _instant(value: object) -> float:
    moment = datetime.fromisoformat(str(value or ""))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


class _FakeGateway:
    """Live-provider double: aliases unfiltered (PIT stays in the resolver),
    facts and classifications PIT-filtered to ``as_of``."""

    def __init__(self, seeds: _Seeds) -> None:
        self._seeds = seeds

    def ticker_candidates(self, ticker: str, as_of: datetime) -> list[TickerAlias]:
        del as_of
        want = str(ticker).strip().upper()
        return [alias for alias in self._seeds.aliases if alias.alias_value == want]

    def company_facts(self, cik: int, as_of: str | None = None) -> dict[str, object]:
        cik = int(cik)
        facts = [
            fact
            for fact in self._seeds.facts.get(cik, [])
            if as_of is None or str(fact.get("known_at") or "")[:10] <= as_of
        ]
        securities = [
            sec
            for sec in self._seeds.securities.get(cik, [])
            if as_of is None or str(sec.get("known_at") or "")[:10] <= as_of
        ]

        # Oldest first: the production join lets the newest knowable row win.
        def _known_at(sec: dict[str, object]) -> float:
            return _instant(sec.get("known_at"))

        securities.sort(key=_known_at)
        return {"securities": securities, "financial_facts": facts, "dividend_events": []}


def _install(monkeypatch: pytest.MonkeyPatch, seeds: _Seeds) -> None:
    """Serve seeded provider data through the screen's live seams."""
    gateway = _FakeGateway(seeds)
    monkeypatch.setattr(screens, "_gateway", lambda: gateway)

    def _fetch_settlement_rows(settlement: str) -> list[dict[str, object]]:
        return [dict(row) for row in seeds.short_interest.get(settlement, [[]])[-1]]

    def _probe_published_rows(candidate: str) -> int:
        return 1 if candidate in seeds.short_interest else 0

    monkeypatch.setattr(screens, "_fetch_settlement_rows", _fetch_settlement_rows)
    monkeypatch.setattr(screens, "_probe_published_rows", _probe_published_rows)


def _install_gateway(monkeypatch: pytest.MonkeyPatch, seeds: _Seeds) -> None:
    """Serve seeded SEC data while leaving the FINRA fetch path live (for client fakes)."""
    gateway = _FakeGateway(seeds)
    monkeypatch.setattr(screens, "_gateway", lambda: gateway)


def _default_rows() -> list[dict[str, object]]:
    return [
        {"symbolCode": "AAA", "issueName": "Alpha", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 20},
        {"symbolCode": "BBB", "issueName": "Beta", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 20},
        {"symbolCode": "CCC", "issueName": "Gamma", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 5},
    ]


def _prior_rows() -> list[dict[str, object]]:
    return [
        {
            "symbolCode": "AAA",
            "issueName": "Alpha",
            "settlementDate": PRIOR_SETTLEMENT,
            "currentShortPositionQuantity": 10,
        },
        {
            "symbolCode": "BBB",
            "issueName": "Beta",
            "settlementDate": PRIOR_SETTLEMENT,
            "currentShortPositionQuantity": 10,
        },
        {
            "symbolCode": "CCC",
            "issueName": "Gamma",
            "settlementDate": PRIOR_SETTLEMENT,
            "currentShortPositionQuantity": 5,
        },
    ]


def _default_facts() -> dict[int, list[dict[str, object]]]:
    return {
        1: [{"end": "2026-08-01", "val": 100, "accn": "a1", "filed": "2026-08-02"}],
        2: [{"end": "2026-08-01", "val": 200, "accn": "b1", "filed": "2026-08-02"}],
        3: [{"end": "2026-08-01", "val": 10, "accn": "c1", "filed": "2026-08-02"}],
    }


def _seed_default(seeds: _Seeds) -> None:
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows())


# ---------------------------------------------------------------------------
# Ranking, provenance, determinism
# ---------------------------------------------------------------------------


def test_materialize_ranks_complete_snapshot(data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_default(seeds)
    _install(monkeypatch, seeds)

    result = screens.materialize_short_interest_screen(SETTLEMENT, data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)

    assert [entry["ticker"] for entry in entries] == ["CCC", "AAA", "BBB"]
    assert entries[0]["short_interest_percent"] == 50
    assert result["coverage"] == {
        "finra_rows": 3,
        "eligible_rows": 3,
        "valid_short_interest_rows": 3,
        "mapped_rows": 3,
        "unambiguous_rows": 3,
        "common_equity_rows": 3,
        "shares_outstanding_rows": 3,
        "exclusions": {
            "unmapped_symbol": 0,
            "ambiguous_ticker_mapping": 0,
            "not_classified_common_equity": 0,
            "missing_shares_outstanding": 0,
            "invalid_short_interest": 0,
            "conflicting_versions": 0,
        },
    }
    assert result["calculation_version"] == screens.SCREEN_CALC_VERSION
    # Default as_of is the live horizon (UTC today), not the settlement date.
    assert result["as_of_date"] == datetime.now(UTC).date().isoformat()
    assert result["source_records"]
    assert entries[0]["sec_accession"] == "c1"
    assert entries[0]["sec_source_url"].endswith("CIK0000000003.json")


def test_rerun_is_deterministic(data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_default(seeds)
    _install(monkeypatch, seeds)
    first = screens.materialize_short_interest_screen(SETTLEMENT, data_root=data_root)
    second = screens.materialize_short_interest_screen(SETTLEMENT, data_root=data_root)
    first_entries = first["entries"]
    assert isinstance(first_entries, list)
    second_entries = second["entries"]
    assert isinstance(second_entries, list)
    assert [e["ticker"] for e in first_entries] == [e["ticker"] for e in second_entries]
    assert second == first


def test_enrichment_adds_newcomer_and_rerun_is_deterministic(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-day targeted enrichment ranks the newcomer; rerunning is a no-op."""
    _seed_tickers(seeds, tickers=("AAA", "BBB", "CCC", "DDD"))
    extra_ddd: list[dict[str, object]] = [
        {"symbolCode": "DDD", "issueName": "Delta", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 20},
    ]
    _seed_short_interest(seeds, _default_rows() + extra_ddd)
    _seed_facts(seeds, _default_facts())  # DDD's SEC facts arrive later
    _install(monkeypatch, seeds)
    first = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    first_entries = first["entries"]
    assert isinstance(first_entries, list)
    assert [e["ticker"] for e in first_entries] == ["CCC", "AAA", "BBB"]
    first_coverage = first["coverage"]
    assert isinstance(first_coverage, dict)
    assert first_coverage["exclusions"]["not_classified_common_equity"] == 1
    # Mid-day enrichment: DDD facts (filed 2026-08-05 -> known_at, visible at as_of 08-14)
    _seed_facts(seeds, {4: [{"end": "2026-08-01", "val": 50, "accn": "d1", "filed": "2026-08-05"}]})
    second = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    second_entries = second["entries"]
    assert isinstance(second_entries, list)
    assert [e["ticker"] for e in second_entries] == ["CCC", "DDD", "AAA", "BBB"]
    # Deterministic no-op on identical inputs serves the latest version.
    third = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    assert third == second
    third_entries = third["entries"]
    assert isinstance(third_entries, list)
    assert [e["ticker"] for e in third_entries] == ["CCC", "DDD", "AAA", "BBB"]


def test_read_is_bounded_by_limit(data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_default(seeds)
    _install(monkeypatch, seeds)
    result = screens.get_short_interest_leaderboard(limit=2, settlement_date=SETTLEMENT, data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)
    assert [entry["ticker"] for entry in entries] == ["CCC", "AAA"]
    result = screens.get_short_interest_leaderboard(limit=999, settlement_date=SETTLEMENT, data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)
    assert len(entries) == 3  # cap is a maximum, not a target
    assert len(entries) <= screens.MAX_LIMIT


def test_missing_settlement_date_is_honest_error(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_default(seeds)
    _install(monkeypatch, seeds)
    # Historical reproduction never fetches: a missing cycle stays an error.
    result = screens.get_short_interest_leaderboard(
        settlement_date="2025-01-15", as_of="2026-08-14", data_root=data_root
    )
    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "knowable on or before 2026-08-14" in error


# ---------------------------------------------------------------------------
# As-of regression: a later filing cannot affect an earlier ranking
# ---------------------------------------------------------------------------


def test_as_of_regression_later_filing_does_not_change_earlier_ranking(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_tickers(seeds)
    _seed_facts(
        seeds,
        {
            1: [{"end": "2026-08-01", "val": 100, "accn": "a1", "filed": "2026-08-02"}],
            2: [{"end": "2026-08-01", "val": 200, "accn": "b1", "filed": "2026-08-02"}],
            3: [{"end": "2026-08-01", "val": 10, "accn": "c1", "filed": "2026-08-02"}],
        },
    )
    _seed_short_interest(seeds, _default_rows())
    _install(monkeypatch, seeds)

    early = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    early_entries = early["entries"]
    assert isinstance(early_entries, list)
    assert [e["ticker"] for e in early_entries] == ["CCC", "AAA", "BBB"]
    assert early_entries[1]["short_interest_percent"] == 20  # AAA: 20/100

    # A later filing (known_at after 2026-08-14) restates AAA's shares to 400.
    _seed_facts(
        seeds,
        {
            1: [{"end": "2026-08-01", "val": 400, "accn": "a2", "filed": "2026-08-20"}],
        },
    )

    # The earlier as-of ranking must be byte-identical after the later filing.
    rerun = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    assert rerun["entries"] == early["entries"]
    rerun_entries = rerun["entries"]
    assert isinstance(rerun_entries, list)
    assert rerun_entries[1]["sec_accession"] == "a1"
    assert rerun_entries[1]["short_interest_percent"] == 20

    # A later as-of sees the restatement: AAA falls from 20% to 5%.
    later = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-21", data_root=data_root)
    later_entries = later["entries"]
    assert isinstance(later_entries, list)
    by_ticker = {e["ticker"]: e for e in later_entries}
    assert [e["ticker"] for e in later_entries] == ["CCC", "BBB", "AAA"]
    assert by_ticker["AAA"]["sec_accession"] == "a2"
    assert by_ticker["AAA"]["short_interest_percent"] == 5


def test_fact_with_period_after_settlement_is_never_used(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shares-outstanding fact must be as of (or before) the settlement
    date; a fact with a later period end is not eligible — even when it is
    already knowable at the as_of."""
    _seed_tickers(seeds)
    _seed_facts(
        seeds,
        {
            1: [{"end": "2026-09-01", "val": 100, "accn": "a1", "filed": "2026-08-20"}],
            2: [{"end": "2026-06-30", "val": 200, "accn": "b1", "filed": "2026-08-02"}],
            3: [{"end": "2026-08-01", "val": 10, "accn": "c1", "filed": "2026-08-02"}],
        },
    )
    _seed_short_interest(seeds, _default_rows())
    _install(monkeypatch, seeds)

    result = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-30", data_root=data_root)
    coverage = result["coverage"]
    assert isinstance(coverage, dict)
    entries = result["entries"]
    assert isinstance(entries, list)

    assert coverage["exclusions"]["missing_shares_outstanding"] == 1
    assert [e["ticker"] for e in entries] == ["CCC", "BBB"]


def test_e2e_fixtures_to_leaderboard_uses_production_only(
    tmp_path: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh seeds built via production normalizers only, read through the
    live seams — no normalized rows hand-constructed."""
    data_root = tmp_path / "data"
    _seed_default(seeds)
    _install(monkeypatch, seeds)

    result = screens.get_short_interest_leaderboard(settlement_date=SETTLEMENT, data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)

    assert [e["ticker"] for e in entries] == ["CCC", "AAA", "BBB"]
    assert [e["short_interest_percent"] for e in entries] == [50.0, 20.0, 10.0]
    assert result["source_records"]


# ---------------------------------------------------------------------------
# Universe and exclusions
# ---------------------------------------------------------------------------


def test_unmapped_ambiguous_and_unclassified_rows_are_excluded(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra_unmapped: list[dict[str, object]] = [
        {"symbolCode": "DDD", "issueName": "Delta", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 10},
        {"symbolCode": "EEE", "issueName": "Epsilon", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 10},
        {"symbolCode": "FFF", "issueName": "Phi", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": None},
    ]
    rows: list[dict[str, object]] = _default_rows() + extra_unmapped
    _seed_tickers(seeds, tickers=("AAA", "BBB", "CCC", "EEE"))
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, rows)
    # EEE also appears under a second CIK -> ambiguous.
    _add_alias(seeds, "EEE", 99, known_at="2026-08-21T12:00:00Z", retrieved_at="2026-08-21T12:00:00Z")
    _install(monkeypatch, seeds)

    result = screens.materialize_short_interest_screen(SETTLEMENT, data_root=data_root)
    coverage = result["coverage"]
    assert isinstance(coverage, dict)
    entries = result["entries"]
    assert isinstance(entries, list)

    assert coverage["finra_rows"] == 6
    assert coverage["eligible_rows"] == 3
    assert coverage["exclusions"] == {
        "unmapped_symbol": 1,  # DDD
        "ambiguous_ticker_mapping": 1,  # EEE
        "not_classified_common_equity": 0,
        "missing_shares_outstanding": 0,
        "invalid_short_interest": 1,  # FFF
        "conflicting_versions": 0,
    }
    assert [e["ticker"] for e in entries] == ["CCC", "AAA", "BBB"]
    assert result["truncated"] is False
    assert result["unresolved"] == ["DDD", "EEE", "FFF"]


def test_stale_settlement_is_surfaced(data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_facts(seeds, _default_facts())
    stale_date = "2025-01-15"
    _seed_short_interest(
        seeds,
        _default_rows(),
        retrieved_at="2025-01-20T12:00:00Z",
        content_hash="snapshot-hash-2",
        settlement=stale_date,
    )
    _install(monkeypatch, seeds)

    stale = screens.materialize_short_interest_screen(stale_date, as_of="2025-01-20", data_root=data_root)
    assert stale["data_freshness"] == "stale"
    assert stale["as_of_date"] == "2025-01-20"


def test_short_interest_rows_gate_on_known_at_not_settlement(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A past-settlement cycle retrieved today is invisible at a past as_of."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows(), retrieved_at="2026-08-20T12:00:00Z")
    _install(monkeypatch, seeds)

    early = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    assert "error" in early
    early_error = early["error"]
    assert isinstance(early_error, str) and "knowable on or before 2026-08-14" in early_error
    later = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-20", data_root=data_root)
    assert isinstance(later.get("coverage"), dict) and later["coverage"]["finra_rows"] == 3


def test_snapshot_later_settlement_invisible_at_as_of(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settlement after as_of is invisible to that as_of."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_cycle(seeds, "2026-08-29", _default_rows(), retrieved_at="2026-08-30T12:00:00Z")
    _install(monkeypatch, seeds)

    result = screens.materialize_short_interest_screen("2026-08-29", as_of="2026-08-14", data_root=data_root)

    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "knowable on or before 2026-08-14" in error


def test_snapshot_fetched_late_but_public_early_is_visible(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieval-gated: settlement 08-14 retrieved 08-30 is invisible at
    as_of 08-20, visible once as_of reaches retrieval."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows(), retrieved_at="2026-08-30T12:00:00Z")
    _install(monkeypatch, seeds)

    early = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-20", data_root=data_root)
    assert "error" in early

    visible = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-30", data_root=data_root)
    entries = visible["entries"]
    assert isinstance(entries, list)
    assert [e["ticker"] for e in entries] == ["CCC", "AAA", "BBB"]

    pre = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-10", data_root=data_root)
    assert "error" in pre


def test_finra_dec15_cycle_hidden_before_publication(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FINRA Dec-15-2025 settlement published Dec-24: invisible at 12-20, visible at 12-24."""
    _seed_tickers(seeds, retrieved_at="2025-12-01T12:00:00Z")
    _seed_facts(
        seeds,
        {
            1: [{"end": "2025-09-30", "val": 100, "accn": "a1", "filed": "2025-11-01"}],
            2: [{"end": "2025-09-30", "val": 200, "accn": "b1", "filed": "2025-11-01"}],
            3: [{"end": "2025-09-30", "val": 10, "accn": "c1", "filed": "2025-11-01"}],
        },
        retrieved_at="2025-12-01T12:00:00Z",
    )
    _seed_cycle(seeds, "2025-12-15", _default_rows(), retrieved_at="2025-12-24T12:00:00Z")
    _install(monkeypatch, seeds)

    hidden = screens.materialize_short_interest_screen("2025-12-15", as_of="2025-12-20", data_root=data_root)
    assert "error" in hidden

    shown = screens.materialize_short_interest_screen("2025-12-15", as_of="2025-12-24", data_root=data_root)
    shown_entries = shown["entries"]
    assert isinstance(shown_entries, list)
    assert [e["ticker"] for e in shown_entries] == ["CCC", "AAA", "BBB"]


def test_ticker_alias_acquired_after_as_of_is_unusable(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ticker mapping acquired after as_of cannot be used by an earlier
    screen: CCC is unmapped at 2026-08-14 and mapped at 2026-08-21."""
    _seed_tickers(seeds, tickers=("AAA", "BBB"), retrieved_at="2026-08-10T12:00:00Z")
    _seed_tickers(seeds, tickers=("CCC",), retrieved_at="2026-08-20T12:00:00Z", cik_start=3)
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows())
    _install(monkeypatch, seeds)

    early = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    early_coverage = early["coverage"]
    assert isinstance(early_coverage, dict)
    early_entries = early["entries"]
    assert isinstance(early_entries, list)
    assert early_coverage["exclusions"]["unmapped_symbol"] == 1
    assert [e["ticker"] for e in early_entries] == ["AAA", "BBB"]

    later = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-21", data_root=data_root)
    later_coverage = later["coverage"]
    assert isinstance(later_coverage, dict)
    later_entries = later["entries"]
    assert isinstance(later_entries, list)
    assert later_coverage["exclusions"]["unmapped_symbol"] == 0
    assert [e["ticker"] for e in later_entries] == ["CCC", "AAA", "BBB"]


def test_corrected_snapshot_newest_retrieved_wins_at_both_as_of(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two source versions: only versions retrieved on/before as_of are
    knowable; the live snapshot serves the newest knowable version per symbol."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows(), retrieved_at="2026-08-10T12:00:00Z")
    _install(monkeypatch, seeds)

    early = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    early_entries = early["entries"]
    assert isinstance(early_entries, list)
    assert [e["ticker"] for e in early_entries] == ["CCC", "AAA", "BBB"]
    assert early_entries[1]["short_shares"] == 20  # correction not yet knowable at 08-14

    # The provider publishes a correction retrieved 08-20; the live snapshot
    # serves it once as_of reaches retrieval.
    corrected: list[dict[str, object]] = [
        {"symbolCode": "AAA", "issueName": "Alpha", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 25},
        {"symbolCode": "BBB", "issueName": "Beta", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 20},
        {"symbolCode": "CCC", "issueName": "Gamma", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 5},
    ]
    _seed_short_interest(seeds, corrected, retrieved_at="2026-08-20T12:00:00Z", content_hash="v2-snapshot-hash")

    later = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-21", data_root=data_root)
    later_entries = later["entries"]
    assert isinstance(later_entries, list)
    later_coverage = later["coverage"]
    assert isinstance(later_coverage, dict)
    assert later_entries[1]["short_shares"] == 25  # corrected version
    assert later_coverage["finra_rows"] == 3  # one version per symbol, not both


def test_security_classification_is_consulted(data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eligibility comes from the securities classification, not a
    fact-presence proxy: reclassifying ETF (unknown type) excludes it even
    though a shares-outstanding fact exists."""
    _seed_tickers(seeds, tickers=("AAA", "BBB", "CCC", "ETF"))
    _seed_facts(
        seeds,
        {
            **{cik: facts for cik, facts in _default_facts().items()},
            4: [{"end": "2026-08-01", "val": 50, "accn": "e1", "filed": "2026-08-02"}],
        },
    )
    extra_etf: list[dict[str, object]] = [
        {
            "symbolCode": "ETF",
            "issueName": "Index Fund",
            "settlementDate": SETTLEMENT,
            "currentShortPositionQuantity": 5,
        },
    ]
    rows: list[dict[str, object]] = _default_rows() + extra_etf
    _seed_short_interest(seeds, rows)
    # A later classification row reclassifies the ETF as not common equity.
    _reclassify(seeds, 4, "unknown", known_at="2026-08-25T12:00:00Z", retrieved_at="2026-08-25T12:00:00Z")
    _install(monkeypatch, seeds)

    early = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-21", data_root=data_root)
    early_entries = early["entries"]
    assert isinstance(early_entries, list)
    assert "ETF" in [e["ticker"] for e in early_entries]
    assert early["unresolved"] == []

    later = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-30", data_root=data_root)
    later_coverage = later["coverage"]
    assert isinstance(later_coverage, dict)
    later_entries = later["entries"]
    assert isinstance(later_entries, list)
    assert later_coverage["exclusions"]["not_classified_common_equity"] == 1
    assert "ETF" not in [e["ticker"] for e in later_entries]
    assert later["unresolved"] == []


def test_corrected_snapshot_mixed_offsets_newest_wins(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live snapshot serves one version per symbol: the correction
    retrieved 12:30Z supersedes the 13:00+01:00 (=12:00Z) version."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows(), retrieved_at="2026-08-10T13:00:00+01:00")
    corrected: list[dict[str, object]] = [
        {"symbolCode": "AAA", "issueName": "Alpha", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 25},
        {"symbolCode": "BBB", "issueName": "Beta", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 20},
        {"symbolCode": "CCC", "issueName": "Gamma", "settlementDate": SETTLEMENT, "currentShortPositionQuantity": 5},
    ]
    _seed_short_interest(seeds, corrected, retrieved_at="2026-08-10T12:30:00Z", content_hash="v2-mixed-offset-hash")
    _install(monkeypatch, seeds)

    result = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)
    assert [e["ticker"] for e in entries] == ["CCC", "AAA", "BBB"]
    assert entries[1]["short_shares"] == 25  # the 12:30Z correction wins


def test_security_type_map_mixed_offsets_newest_wins(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A classification revision with mixed offsets: the lexically-larger
    but chronologically older 13:00+01:00 'unknown' row must not beat the
    12:30Z 'equity-common' correction."""
    _seed_tickers(seeds, tickers=("AAA", "BBB", "CCC"))
    _seed_facts(seeds, _default_facts())
    _seed_short_interest(seeds, _default_rows())
    _reclassify(seeds, 1, "unknown", known_at="2026-08-10T13:00:00+01:00", retrieved_at="2026-08-10T13:00:00+01:00")
    _reclassify(seeds, 1, "equity-common", known_at="2026-08-10T12:30:00Z", retrieved_at="2026-08-10T12:30:00Z")
    _install(monkeypatch, seeds)

    result = screens.materialize_short_interest_screen(SETTLEMENT, as_of="2026-08-14", data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)
    assert [e["ticker"] for e in entries] == ["CCC", "AAA", "BBB"]  # AAA stays classified


# ---------------------------------------------------------------------------
# Research slice: short-interest change + shares-outstanding change
# ---------------------------------------------------------------------------


def test_change_slice_computes_changes_with_evidence(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_cycle(seeds, PRIOR_SETTLEMENT, _prior_rows())
    _seed_cycle(seeds, SETTLEMENT, _default_rows())
    _install(monkeypatch, seeds)

    result = screens.short_interest_change_screen("2026-08-21", data_root=data_root)
    entries = result["entries"]
    assert isinstance(entries, list)

    assert result["settlement_current"] == SETTLEMENT
    assert result["settlement_prior"] == PRIOR_SETTLEMENT
    assert result["calculation_version"] == screens.SLICE_CALC_VERSION
    by_ticker = {e["ticker"]: e for e in entries}
    assert by_ticker["AAA"]["short_shares_current"] == 20
    assert by_ticker["AAA"]["short_shares_prior"] == 10
    assert by_ticker["AAA"]["short_change_pct"] == 100.0
    assert by_ticker["AAA"]["si_pp_change"] == 10.0  # 20% - 10%
    assert by_ticker["AAA"]["shares_change_abs"] == 0
    assert by_ticker["AAA"]["sec_accession_current"] == "a1"
    assert by_ticker["AAA"]["sec_accession_prior"] == "a1"
    assert by_ticker["AAA"]["finra_source_url"].startswith("https://api.finra.org")
    # Sorted by signed short-interest pp change: AAA moved most.
    assert [e["ticker"] for e in entries] == ["AAA", "BBB", "CCC"]
    assert result["unresolved"] == []


def _unmapped_row() -> dict[str, object]:
    return {
        "symbolCode": "DDD",
        "issueName": "Delta",
        "settlementDate": SETTLEMENT,
        "currentShortPositionQuantity": 10,
    }


def test_change_slice_surfaces_unresolved_symbols(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unresolved symbols surface explicitly, never vanish (matches the leaderboard key)."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_cycle(seeds, PRIOR_SETTLEMENT, _prior_rows())
    _seed_cycle(seeds, SETTLEMENT, _default_rows() + [_unmapped_row()])
    _install(monkeypatch, seeds)

    result = screens.short_interest_change_screen("2026-08-21", data_root=data_root)
    assert result["unresolved"] == ["DDD"]


def test_change_slice_reports_missing_prior_cycle_as_none_not_zero(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_cycle(seeds, SETTLEMENT, _default_rows())
    _install(monkeypatch, seeds)

    result = screens.short_interest_change_screen("2026-08-21", data_root=data_root)

    assert result["settlement_prior"] is None
    entries = result["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    assert entry["short_shares_prior"] is None
    assert entry["short_change_pct"] is None
    assert entry["si_pp_change"] is None


def test_change_slice_as_of_regression(data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch) -> None:
    """A later filing cannot alter a slice computed at an earlier as_of."""
    _seed_tickers(seeds)
    _seed_facts(
        seeds,
        {
            1: [{"end": "2026-08-01", "val": 100, "accn": "a1", "filed": "2026-08-02"}],
            2: [{"end": "2026-08-01", "val": 200, "accn": "b1", "filed": "2026-08-02"}],
            3: [{"end": "2026-08-01", "val": 10, "accn": "c1", "filed": "2026-08-02"}],
        },
    )
    _seed_cycle(
        seeds,
        PRIOR_SETTLEMENT,
        _prior_rows(),
        retrieved_at="2026-08-10T12:00:00Z",
    )
    _seed_cycle(seeds, SETTLEMENT, _default_rows(), retrieved_at="2026-08-10T12:00:00Z")
    _install(monkeypatch, seeds)

    early = screens.short_interest_change_screen("2026-08-14", data_root=data_root)
    early_entries = early["entries"]
    assert isinstance(early_entries, list)
    assert early_entries[0]["ticker"] == "AAA"
    assert early_entries[0]["shares_outstanding_current"] == 100.0

    # A filing known only after 2026-08-14 restates AAA's shares for a
    # period between the two settlements (end 2026-08-13, filed 2026-08-20).
    _seed_facts(
        seeds,
        {
            1: [{"end": "2026-08-13", "val": 400, "accn": "a2", "filed": "2026-08-20"}],
        },
    )

    rerun = screens.short_interest_change_screen("2026-08-14", data_root=data_root)
    assert rerun["entries"] == early["entries"]

    later = screens.short_interest_change_screen("2026-08-21", data_root=data_root)
    later_entries = later["entries"]
    assert isinstance(later_entries, list)
    aaa = next(e for e in later_entries if e["ticker"] == "AAA")
    assert aaa["sec_accession_current"] == "a2"
    assert aaa["shares_outstanding_current"] == 400.0
    assert aaa["shares_change_abs"] == 300.0


def test_change_slice_later_settlement_invisible_at_as_of(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settlement after as_of is not knowable at that as_of."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_cycle(seeds, "2026-08-29", _default_rows(), retrieved_at="2026-08-10T12:00:00Z")
    _install(monkeypatch, seeds)

    result = screens.short_interest_change_screen("2026-08-14", data_root=data_root)
    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "knowable" in error


def test_change_slice_fetched_late_but_public_early_is_visible(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieval-gated: the change slice sees a 08-14 cycle retrieved 08-30
    once as_of reaches retrieval."""
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _seed_cycle(
        seeds,
        PRIOR_SETTLEMENT,
        _prior_rows(),
    )
    _seed_cycle(seeds, SETTLEMENT, _default_rows(), retrieved_at="2026-08-30T12:00:00Z")
    _install(monkeypatch, seeds)

    result = screens.short_interest_change_screen("2026-08-30", data_root=data_root)
    assert result["settlement_current"] == SETTLEMENT


# ---------------------------------------------------------------------------
# Live discovery: screens fetch from FINRA via the client
# ---------------------------------------------------------------------------


def _install_finra_fetch_fake(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, object]]) -> None:
    """Serve discovery probes (limit 1) and full snapshots from one fake.

    The first probe reports no published rows so discovery must skip it;
    later probes report rows.  Full fetches return AAA/BBB/CCC rows for the
    requested settlement date.
    """
    probed = {"count": 0}

    def fake(
        group: str, dataset_name: str, payload: dict[str, object]
    ) -> tuple[bytes, list[dict[str, object]], dict[str, str]]:
        calls.append(payload)
        raw_filters = payload.get("compareFilters", [])
        assert isinstance(raw_filters, list)
        filters = {f.get("fieldName"): f.get("fieldValue") for f in raw_filters if isinstance(f, dict)}
        settlement = str(filters.get("settlementDate"))
        if payload.get("limit") == 1:  # discovery probe: existence only
            probed["count"] += 1
            total = 3 if probed["count"] > 1 else 0
            return b"[]", [], {"record-total": str(total)}
        rows: list[dict[str, object]] = [
            {
                "symbolCode": symbol,
                "issueName": symbol,
                "settlementDate": settlement,
                "currentShortPositionQuantity": position,
            }
            for symbol, position in (("AAA", 20), ("BBB", 20), ("CCC", 5))
        ]
        import json as _json

        return (
            _json.dumps(rows).encode(),
            rows,
            {"record-total": str(len(rows))},
        )

    monkeypatch.setattr(screens.finra_client, "ingestion_post_query", fake)


def test_live_leaderboard_empty_store_discovers_and_fetches_once(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _install_gateway(monkeypatch, seeds)
    calls: list[dict[str, object]] = []
    _install_finra_fetch_fake(monkeypatch, calls)

    result = screens.get_short_interest_leaderboard(data_root=data_root)

    assert "error" not in result
    entries = result["entries"]
    assert isinstance(entries, list)
    assert [e["ticker"] for e in entries] == ["CCC", "AAA", "BBB"]
    probes = [c for c in calls if c.get("limit") == 1]
    full = [c for c in calls if c.get("limit") != 1]
    assert len(probes) == 2  # newest candidate empty, next one hits
    assert len(full) == 1  # exactly one full fetch
    full_filters = full[0]["compareFilters"]
    assert isinstance(full_filters, list) and full_filters
    full_first = full_filters[0]
    assert isinstance(full_first, dict)
    probe_filters = probes[1]["compareFilters"]
    assert isinstance(probe_filters, list) and probe_filters
    probe_first = probe_filters[0]
    assert isinstance(probe_first, dict)
    assert full_first["fieldValue"] == probe_first["fieldValue"]
    assert result["settlement_date"] == full_first["fieldValue"]


def test_live_leaderboard_explicit_date_fetches_exactly_that_date(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _install_gateway(monkeypatch, seeds)
    calls: list[dict[str, object]] = []
    _install_finra_fetch_fake(monkeypatch, calls)

    result = screens.get_short_interest_leaderboard(settlement_date=SETTLEMENT, data_root=data_root)

    assert "error" not in result
    assert result["settlement_date"] == SETTLEMENT
    assert len(calls) == 1  # no discovery probes, one exact-date fetch
    filters = calls[0]["compareFilters"]
    assert isinstance(filters, list) and filters
    first = filters[0]
    assert isinstance(first, dict)
    assert first["fieldValue"] == SETTLEMENT


def test_historical_leaderboard_live_rows_stay_excluded_by_pit(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery still probes live, but rows fetched now are not knowable at
    a past as_of, so a historical screen stays an honest error."""
    _install_gateway(monkeypatch, seeds)
    calls: list[dict[str, object]] = []
    _install_finra_fetch_fake(monkeypatch, calls)

    result = screens.get_short_interest_leaderboard(as_of="2026-08-14", data_root=data_root)

    assert calls != []  # discovery probed the live client
    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "knowable on or before 2026-08-14" in error


def test_discovery_probe_uses_mock_dataset_in_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRA_USE_MOCK", "1")
    names: list[str] = []

    def fake(
        group: str, dataset_name: str, payload: dict[str, object]
    ) -> tuple[bytes, list[dict[str, object]], dict[str, str]]:
        names.append(dataset_name)
        return b"[]", [], {"record-total": "0"}

    monkeypatch.setattr(screens.finra_client, "ingestion_post_query", fake)

    assert screens._discover_latest_published_settlement_date(date(2026, 9, 9)) is None
    assert len(names) == screens._FETCH_DISCOVERY_CYCLES
    assert all(name == "consolidatedShortInterestMock" for name in names)


def test_live_fetch_failure_returns_error_not_raise(
    data_root: Path, seeds: _Seeds, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_tickers(seeds)
    _seed_facts(seeds, _default_facts())
    _install_gateway(monkeypatch, seeds)

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("network down")

    monkeypatch.setattr(screens.finra_client, "ingestion_post_query", boom)

    result = screens.get_short_interest_leaderboard(settlement_date=SETTLEMENT, data_root=data_root)

    assert "error" in result
    error = result["error"]
    assert isinstance(error, str)
    assert "network down" in error
