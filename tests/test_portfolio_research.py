"""Tests for the portfolio research view (SEC + FINRA enrichment).

Live-seam by construction: a stub gateway hands SEC facts and FINRA rows to
``enrich_portfolio_research`` via the ``gateway=`` kwarg; point-in-time
filtering stays in the service under test.

Warehouse-removal seam: live providers (SourceGateway + normalization + raw_archive) serve reads, nothing is persisted; a future warehouse slots in behind the gateway.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import override

import pytest

from app.data_sources import SourceGateway
from app.domain.portfolio import PortfolioSnapshot, Position
from app.services.portfolio_research import (
    SEC_CONCEPTS,
    PortfolioResearchPosition,
    enrich_portfolio_research,
)

ENTITY_ID = "sec:cik:0000320193"
SECURITY_ID = "sec:equity:0000320193"
RETRIEVED_AT = "2026-08-25T12:00:00Z"


def _fact_source_url(entity_id: str = ENTITY_ID) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/{entity_id.split(':')[-1]}.json"


class _Gateway(SourceGateway):
    """Seeded SourceGateway double: company_facts + short_interest only."""

    def __init__(self) -> None:
        super().__init__()
        self.facts: list[dict[str, object]] = []
        self.shorts: list[dict[str, object]] = []

    @override
    def company_facts(self, cik: int, *, as_of: str | None = None) -> dict[str, object]:
        assert cik == 320193
        rows = [dict(row) for row in self.facts]
        if as_of is not None:
            rows = [row for row in rows if str(row.get("known_at") or "")[:10] <= as_of]
        return {"financial_facts": rows}

    @override
    def short_interest(self, symbol: str, *, as_of: str | None = None) -> list[dict[str, object]]:
        rows = [row for row in self.shorts if row.get("symbol_code") == symbol.strip().upper()]
        if as_of is not None:
            rows = [row for row in rows if str(row.get("settlement_date") or "")[:10] <= as_of]
        return [dict(row) for row in rows]


@pytest.fixture
def gateway() -> _Gateway:
    return _Gateway()


def _seed_fact(
    gateway: _Gateway,
    concept: str,
    value: float,
    period_end: str,
    filed_at: str,
    accession: str,
    entity_id: str = ENTITY_ID,
) -> None:
    gateway.facts.append(
        {
            "fact_id": f"sec:fact:{entity_id}:{concept}:{accession}",
            "entity_id": entity_id,
            "security_id": SECURITY_ID,
            "concept": concept,
            "value": value,
            "unit": "shares" if concept == "EntityCommonStockSharesOutstanding" else "USD",
            "period_end": period_end,
            "filed_at": filed_at,
            "accession": accession,
            "known_at": filed_at,
            "retrieved_at": RETRIEVED_AT,
            "source_url": _fact_source_url(entity_id),
        }
    )


def _seed_short_interest(
    gateway: _Gateway,
    settlement_date: str,
    short_position: float,
    retrieved_at: str,
    prev_position: float | None = None,
    avg_daily_volume: float | None = None,
    days_to_cover: float | None = None,
    symbol: str = "AMD",
) -> None:
    gateway.shorts.append(
        {
            "symbol_code": symbol,
            "issue_name": "Advanced Micro Devices, Inc.",
            "settlement_date": settlement_date,
            "short_position": short_position,
            "prev_position": prev_position,
            "avg_daily_volume": avg_daily_volume,
            "days_to_cover": days_to_cover,
            "known_at": settlement_date,
            "retrieved_at": retrieved_at,
        }
    )


def _position(
    position_id: str = "pos-1",
    entity_id: str | None = ENTITY_ID,
    security_id: str | None = SECURITY_ID,
    ticker: str = "AMD",
) -> Position:
    return Position(
        position_id=position_id,
        account_id="acc-1",
        security_id=security_id,
        entity_id=entity_id,
        ticker=ticker,
        quantity=Decimal(10),
        average_cost=Decimal(100),
        market_price=Decimal(110),
        market_value=Decimal(1100),
        unrealized_gain=Decimal(100),
        unrealized_gain_pct=Decimal(10),
        portfolio_weight=Decimal("0.05"),
        source="test",
        retrieved_at=datetime(2026, 8, 25, 15, 0, tzinfo=UTC),
    )


def _snapshot(positions: list[Position]) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        snapshot_id="snap-1",
        created_at=datetime(2026, 8, 25, 15, 0, tzinfo=UTC),
        broker="test",
        account_ids=("acc-1",),
        cash=Decimal(0),
        invested_value=None,
        total_value=None,
        positions=tuple(positions),
    )


# ---------------------------------------------------------------------------
# Cross-source integration (spec §30)
# ---------------------------------------------------------------------------


def test_cross_source_integration_enriches_resolved_position(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 5_860_000_000.0, "2026-06-30", "2026-08-05", "accn-rev-1")
    _seed_fact(gateway, "Revenue", 5_890_000_000.0, "2026-06-30", "2026-08-20", "accn-rev-2")
    _seed_fact(gateway, "NetIncomeLoss", 265_000_000.0, "2026-06-30", "2026-08-05", "accn-ni-1")
    _seed_fact(gateway, "CashAndCashEquivalents", 4_100_000_000.0, "2026-06-30", "2026-08-05", "accn-cash-1")
    _seed_fact(gateway, "LongTermDebt", 2_300_000_000.0, "2026-06-30", "2026-08-05", "accn-debt-1")
    _seed_fact(
        gateway, "EntityCommonStockSharesOutstanding", 1_610_000_000.0, "2026-07-01", "2026-08-06", "accn-shares-1"
    )
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-07",
        short_position=1_000_000,
        prev_position=950_000,
        retrieved_at="2026-08-10T12:00:00Z",
    )
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        prev_position=1_000_000,
        avg_daily_volume=38_000_000,
        days_to_cover=2.3,
        retrieved_at="2026-08-20T12:00:00Z",
    )

    position = _position()
    results = enrich_portfolio_research(_snapshot([position]), gateway=gateway)

    assert len(results) == 1
    research = results[0]
    assert isinstance(research, PortfolioResearchPosition)
    assert research.position is position

    sec = research.latest_sec_metrics
    assert set(sec) == set(SEC_CONCEPTS)
    assert sec["Revenue"] == {
        "value": Decimal(5890000000),
        "period_end": "2026-06-30",
        "filed_at": "2026-08-20",
        "accession": "accn-rev-2",
        "source_url": _fact_source_url(),
    }
    net_income = sec["NetIncomeLoss"]
    assert isinstance(net_income, dict)
    assert net_income["value"] == Decimal(265000000)
    cash = sec["CashAndCashEquivalents"]
    assert isinstance(cash, dict)
    assert cash["value"] == Decimal(4100000000)
    debt = sec["LongTermDebt"]
    assert isinstance(debt, dict)
    assert debt["value"] == Decimal(2300000000)
    shares = sec["EntityCommonStockSharesOutstanding"]
    assert isinstance(shares, dict)
    assert shares["value"] == Decimal(1610000000)

    finra = research.latest_finra_metrics
    assert finra == {
        "short_position": Decimal(1150000),
        "prev_position": Decimal(1000000),
        "short_interest_change": Decimal(150000),
        "short_interest_change_pct": Decimal(15),
        "days_to_cover": Decimal("2.3"),
        "settlement_date": "2026-08-14",
        "avg_daily_volume": Decimal(38000000),
        "known_at": "2026-08-14",
        "retrieved_at": "2026-08-20T12:00:00Z",
    }

    freshness = research.research_data_freshness
    assert freshness == {
        "as_of": datetime.now(UTC).date().isoformat(),
        "sec_latest_filed_at": date(2026, 8, 20),
        "finra_settlement_date": date(2026, 8, 14),
        "finra_retrieved_at": "2026-08-20T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# Unresolved positions
# ---------------------------------------------------------------------------


def test_unresolved_position_gets_empty_sec_and_symbol_based_finra(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 5_860_000_000.0, "2026-06-30", "2026-08-05", "accn-rev-1")
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        prev_position=1_000_000,
        retrieved_at="2026-08-20T12:00:00Z",
    )

    resolved = _position()
    unresolved = _position(position_id="pos-2", entity_id=None, security_id=None)
    results = enrich_portfolio_research(_snapshot([resolved, unresolved]), gateway=gateway)

    assert [r.position.position_id for r in results] == ["pos-1", "pos-2"]
    assert set(results[0].latest_sec_metrics) == {"Revenue"}
    assert results[1].latest_sec_metrics == {}
    assert results[1].latest_finra_metrics["short_position"] == Decimal(1150000)
    assert results[1].research_data_freshness["sec_latest_filed_at"] is None
    assert results[1].research_data_freshness["finra_settlement_date"] == date(2026, 8, 14)
    assert results[1].research_data_freshness["finra_retrieved_at"] == "2026-08-20T12:00:00Z"


# ---------------------------------------------------------------------------
# As-of regression (spec §29/§30)
# ---------------------------------------------------------------------------


def test_as_of_regression_facts_after_as_of_are_excluded(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 5_860_000_000.0, "2026-06-30", "2026-08-05", "accn-rev-1")
    _seed_fact(gateway, "Revenue", 5_890_000_000.0, "2026-06-30", "2026-08-20", "accn-rev-2")
    _seed_fact(gateway, "LongTermDebt", 2_300_000_000.0, "2026-06-30", "2026-08-30", "accn-debt-1")

    position = _position()
    early = enrich_portfolio_research(_snapshot([position]), as_of=date(2026, 8, 14), gateway=gateway)[0]
    assert early.latest_sec_metrics["Revenue"] == {
        "value": Decimal(5860000000),
        "period_end": "2026-06-30",
        "filed_at": "2026-08-05",
        "accession": "accn-rev-1",
        "source_url": _fact_source_url(),
    }
    assert "LongTermDebt" not in early.latest_sec_metrics
    assert early.research_data_freshness == {
        "as_of": "2026-08-14",
        "sec_latest_filed_at": date(2026, 8, 5),
        "finra_settlement_date": None,
        "finra_retrieved_at": None,
    }

    later = enrich_portfolio_research(_snapshot([position]), as_of=date(2026, 8, 25), gateway=gateway)[0]
    later_revenue = later.latest_sec_metrics["Revenue"]
    assert isinstance(later_revenue, dict)
    assert later_revenue["value"] == Decimal(5890000000)
    assert later_revenue["accession"] == "accn-rev-2"
    assert "LongTermDebt" not in later.latest_sec_metrics
    assert later.research_data_freshness["sec_latest_filed_at"] == date(2026, 8, 20)

    future = enrich_portfolio_research(_snapshot([position]), as_of=date(2026, 9, 5), gateway=gateway)[0]
    future_debt = future.latest_sec_metrics["LongTermDebt"]
    assert isinstance(future_debt, dict)
    assert future_debt["value"] == Decimal(2300000000)


# ---------------------------------------------------------------------------
# FINRA newest-version semantics and missing values
# ---------------------------------------------------------------------------


def test_finra_newest_version_wins_per_symbol(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        prev_position=1_000_000,
        retrieved_at="2026-08-20T12:00:00Z",
    )

    research = enrich_portfolio_research(_snapshot([_position(entity_id=None)]), gateway=gateway)[0]
    assert research.latest_finra_metrics["short_position"] == Decimal(1150000)
    assert research.latest_finra_metrics["settlement_date"] == "2026-08-14"
    assert research.latest_finra_metrics["known_at"] == "2026-08-14"
    assert research.latest_finra_metrics["retrieved_at"] == "2026-08-20T12:00:00Z"


def test_finra_newest_revision_visible(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        prev_position=1_000_000,
        retrieved_at="2026-08-17T12:30:00Z",
    )

    research = enrich_portfolio_research(_snapshot([_position(entity_id=None)]), gateway=gateway)[0]

    assert research.latest_finra_metrics["short_position"] == Decimal(1150000)
    assert research.latest_finra_metrics["known_at"] == "2026-08-14"
    assert research.latest_finra_metrics["retrieved_at"] == "2026-08-17T12:30:00Z"


def test_finra_same_instant_conflicting_versions_empty(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_200_000,
        prev_position=1_000_000,
        retrieved_at="2026-08-17T12:00:00Z",
    )
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        prev_position=1_000_000,
        retrieved_at="2026-08-17T12:00:00Z",
    )
    research = enrich_portfolio_research(_snapshot([_position(entity_id=None)]), gateway=gateway)[0]
    assert research.latest_finra_metrics == {}


def test_finra_older_settlement_correction_does_not_beat_newer_settlement(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_200_000,
        retrieved_at="2026-08-20T12:00:00Z",
    )
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-29",
        short_position=900_000,
        retrieved_at="2026-09-02T12:00:00Z",
    )
    # Correction to the OLDER settlement, learned after the Aug 29 cycle:
    # must not replace the newer settlement's metrics.
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        retrieved_at="2026-09-03T12:00:00Z",
    )
    research = enrich_portfolio_research(
        _snapshot([_position(entity_id=None)]), as_of=date(2026, 9, 5), gateway=gateway
    )[0]
    assert research.latest_finra_metrics["settlement_date"] == "2026-08-29"
    assert research.latest_finra_metrics["short_position"] == Decimal(900000)
    assert research.latest_finra_metrics["known_at"] == "2026-08-29"


def test_finra_same_instant_ingestion_across_settlements_is_not_conflict(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_200_000,
        retrieved_at="2026-09-01T12:00:00Z",
    )
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-29",
        short_position=900_000,
        retrieved_at="2026-09-01T12:00:00Z",
    )
    research = enrich_portfolio_research(
        _snapshot([_position(entity_id=None)]), as_of=date(2026, 9, 5), gateway=gateway
    )[0]
    # Same instant, two different settlements: NOT a conflict — the newer
    # settlement wins with real metrics.
    assert research.latest_finra_metrics["settlement_date"] == "2026-08-29"
    assert research.latest_finra_metrics["short_position"] == Decimal(900000)


def test_no_data_reports_empty_metrics_without_raising(gateway: _Gateway) -> None:
    position = _position(position_id="pos-1", entity_id=None, security_id=None, ticker="NODATA")
    research = enrich_portfolio_research(_snapshot([position]), gateway=gateway)[0]

    assert research.latest_sec_metrics == {}
    assert research.latest_finra_metrics == {}
    assert research.research_data_freshness == {
        "as_of": datetime.now(UTC).date().isoformat(),
        "sec_latest_filed_at": None,
        "finra_settlement_date": None,
        "finra_retrieved_at": None,
    }
    assert enrich_portfolio_research(_snapshot([]), gateway=gateway) == []


def test_missing_values_are_none_never_zero(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=1_150_000,
        prev_position=None,
        avg_daily_volume=None,
        days_to_cover=None,
        retrieved_at="2026-08-20T12:00:00Z",
    )

    finra = enrich_portfolio_research(_snapshot([_position(entity_id=None)]), gateway=gateway)[0].latest_finra_metrics

    assert finra["short_position"] == Decimal(1150000)
    assert finra["prev_position"] is None
    assert finra["avg_daily_volume"] is None
    assert finra["days_to_cover"] is None
    assert finra["short_interest_change"] is None
    assert finra["short_interest_change_pct"] is None


def test_change_pct_is_none_when_prev_is_zero(gateway: _Gateway) -> None:
    _seed_short_interest(
        gateway,
        settlement_date="2026-08-14",
        short_position=100,
        prev_position=0,
        retrieved_at="2026-08-20T12:00:00Z",
    )

    finra = enrich_portfolio_research(_snapshot([_position(entity_id=None)]), gateway=gateway)[0].latest_finra_metrics

    assert finra["short_interest_change"] == Decimal(100)
    assert finra["short_interest_change_pct"] is None


def _metric(research: PortfolioResearchPosition, concept: str) -> dict[str, object]:
    """Typed accessor: latest_sec_metrics[concept] is always a fact dict."""
    fact = research.latest_sec_metrics[concept]
    assert isinstance(fact, dict)
    return fact


# ---------------------------------------------------------------------------
# Date-resolution: no-date latest, as-of cutoff, interval, last-quarter,
# latest-doc respects cutoff. PIT home; deterministic seeds only.
# ---------------------------------------------------------------------------


def test_no_date_resolves_to_latest_available(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 5_860_000_000.0, "2026-06-30", "2026-08-05", "accn-rev-1")
    _seed_fact(gateway, "Revenue", 5_890_000_000.0, "2026-06-30", "2026-08-20", "accn-rev-2")
    latest = enrich_portfolio_research(_snapshot([_position()]), as_of=None, gateway=gateway)[0]
    assert _metric(latest, "Revenue")["accession"] == "accn-rev-2"


def test_as_of_2025_01_01_excludes_later_filings(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 5_860_000_000.0, "2024-12-31", "2024-12-31", "accn-old")
    _seed_fact(gateway, "Revenue", 9_999_000_000.0, "2025-06-30", "2025-06-30", "accn-new")
    early = enrich_portfolio_research(_snapshot([_position()]), as_of=date(2025, 1, 1), gateway=gateway)[0]
    assert _metric(early, "Revenue")["accession"] == "accn-old"
    assert _metric(early, "Revenue")["filed_at"] == "2024-12-31"


def test_interval_start_end_bounds_facts(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 1_000_000_000.0, "2025-03-31", "2025-04-30", "accn-q1")
    _seed_fact(gateway, "Revenue", 2_000_000_000.0, "2025-06-30", "2025-07-30", "accn-q2")
    mid = enrich_portfolio_research(_snapshot([_position()]), as_of=date(2025, 5, 15), gateway=gateway)[0]
    assert _metric(mid, "Revenue")["accession"] == "accn-q1"
    later = enrich_portfolio_research(_snapshot([_position()]), as_of=date(2025, 8, 1), gateway=gateway)[0]
    assert _metric(later, "Revenue")["accession"] == "accn-q2"


def test_last_quarter_range_picks_quarter_doc(gateway: _Gateway) -> None:
    _seed_fact(gateway, "Revenue", 5_860_000_000.0, "2026-03-31", "2026-05-05", "accn-q1")
    _seed_fact(gateway, "Revenue", 5_890_000_000.0, "2026-06-30", "2026-08-05", "accn-q2")
    end_q1 = enrich_portfolio_research(_snapshot([_position()]), as_of=date(2026, 6, 29), gateway=gateway)[0]
    assert _metric(end_q1, "Revenue")["accession"] == "accn-q1"
    end_q2 = enrich_portfolio_research(_snapshot([_position()]), as_of=date(2026, 8, 25), gateway=gateway)[0]
    assert _metric(end_q2, "Revenue")["accession"] == "accn-q2"


def test_latest_doc_respects_cutoff_not_newest_ingested(gateway: _Gateway) -> None:
    _seed_fact(gateway, "LongTermDebt", 2_100_000_000.0, "2026-03-31", "2026-05-05", "accn-cutoff")
    _seed_fact(gateway, "LongTermDebt", 2_300_000_000.0, "2026-06-30", "2026-08-30", "accn-future")
    at_cutoff = enrich_portfolio_research(_snapshot([_position()]), as_of=date(2026, 8, 14), gateway=gateway)[0]
    assert _metric(at_cutoff, "LongTermDebt")["accession"] == "accn-cutoff"
    assert at_cutoff.research_data_freshness["sec_latest_filed_at"] == date(2026, 5, 5)
