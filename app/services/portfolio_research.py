"""Portfolio research view: SEC + FINRA enrichment per position.

Deterministic, point-in-time enrichment over live providers (SEC EDGAR
company facts, FINRA short interest), scoped by ``as_of`` through the
``SourceGateway`` seam. Missing data is reported as absent (empty dicts /
None), never estimated or zeroed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from ..analytics.screens import _resolve_as_of
from ..domain.portfolio import PortfolioSnapshot, Position

if TYPE_CHECKING:
    from ..data_sources import SourceGateway

SEC_CONCEPTS: tuple[str, ...] = (
    "Revenue",
    "NetIncomeLoss",
    "CashAndCashEquivalents",
    "LongTermDebt",
    "EntityCommonStockSharesOutstanding",
)


@dataclass(frozen=True)
class PortfolioResearchPosition:
    position: Position
    latest_sec_metrics: dict[str, object]
    latest_finra_metrics: dict[str, object]
    research_data_freshness: dict[str, object]


def enrich_portfolio_research(
    snapshot: PortfolioSnapshot,
    *,
    as_of: date | None = None,
    data_root: Path | None = None,
    gateway: SourceGateway | None = None,
) -> list[PortfolioResearchPosition]:
    """Enrich every snapshot position with its latest SEC facts and FINRA
    short-interest metrics, each scoped to ``as_of`` via live providers."""
    as_of_str = _resolve_as_of(as_of.isoformat() if as_of is not None else None)
    del data_root
    if gateway is None:
        from app.data_sources import SourceGateway as _Gateway

        gateway = _Gateway()
    results: list[PortfolioResearchPosition] = []
    for position in snapshot.positions:
        if position.entity_id is not None:
            sec_metrics = _sec_metrics(gateway, position.entity_id, as_of_str)
        else:
            sec_metrics: dict[str, object] = {}
        ticker = (position.ticker or "").strip()
        if ticker:
            finra_metrics = _finra_metrics(gateway, ticker, as_of_str)
        else:
            finra_metrics: dict[str, object] = {}
        results.append(
            PortfolioResearchPosition(
                position=position,
                latest_sec_metrics=sec_metrics,
                latest_finra_metrics=finra_metrics,
                research_data_freshness=_freshness(as_of_str, sec_metrics, finra_metrics),
            )
        )
    return results


def _cik_of(entity_id: str) -> int | None:
    """CIK int for SEC entity ids (None when not an SEC identity)."""
    if not entity_id.startswith("sec:cik:"):
        return None
    try:
        return int(entity_id.split(":")[-1])
    except ValueError:
        return None


def _sec_metrics(gateway: SourceGateway, entity_id: str, as_of: str) -> dict[str, object]:
    """Latest fact per SEC concept for an entity; missing concepts are
    simply absent from the result."""
    cik = _cik_of(entity_id)
    if cik is None:
        return {}
    try:
        facts = gateway.company_facts(cik, as_of=as_of)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {}
    rows = facts.get("financial_facts") if isinstance(facts, dict) else None
    if not isinstance(rows, list):
        return {}
    metrics: dict[str, object] = {}
    for concept in SEC_CONCEPTS:
        fact = _latest_concept_fact(rows, concept)
        if fact is not None:
            metrics[concept] = fact
    return metrics


def _latest_fact_key(fact: dict[str, object]) -> tuple[str, str, str]:
    """Sort key: true latest filing first by filed_at, period end, accession."""
    return (
        str(fact.get("filed_at") or ""),
        str(fact.get("period_end") or ""),
        str(fact.get("accession") or ""),
    )


def _latest_concept_fact(rows: list[object], concept: str) -> dict[str, object] | None:
    """Latest fact of one concept by (filed_at, period_end, accession)."""
    cands = [row for row in rows if isinstance(row, dict) and row.get("concept") == concept]
    if not cands:
        return None
    row = max(cands, key=_latest_fact_key)
    return {
        "value": _decimal(row.get("value")),
        "period_end": _iso_date(row.get("period_end")),
        "filed_at": _iso_date(row.get("filed_at")),
        "accession": row.get("accession"),
        "source_url": row.get("source_url"),
    }


def _iso_date(value: object) -> str:
    """Provider date/datetime to ISO date text (empty when missing)."""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return "" if value is None else str(value)


def _iso_instant(value: object) -> str:
    """Provider datetime to ISO instant text (empty when missing)."""
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return "" if value is None else str(value)


def _finra_day_key(fact: dict[str, object]) -> tuple[str, str]:
    """Sort key: newest source revision wins within one settlement day."""
    return (str(fact.get("known_at") or ""), str(fact.get("retrieved_at") or ""))


def _finra_variants(rows: list[dict[str, object]]) -> set[tuple[str, ...]]:
    """Distinct material value tuples (same-day conflict detection)."""
    return {
        tuple(
            str(row.get(key))
            for key in ("short_position", "prev_position", "avg_daily_volume", "days_to_cover", "issue_name")
        )
        for row in rows
    }


def _finra_metrics(gateway: SourceGateway, ticker: str, as_of: str) -> dict[str, object]:
    """Latest short-interest metrics for a ticker's newest eligible
    settlement cycle (``settlement_date <= as_of``).

    Same-day conflicting versions yield no metrics (empty dict ->
    freshness finra fields None).
    """
    try:
        rows = gateway.short_interest(ticker, as_of=as_of)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {}
    if not isinstance(rows, list):
        return {}
    row = _latest_finra_row(rows, as_of)
    return {} if row is None else _finra_row_metrics(row)


def _eligible_finra_rows(rows: Sequence[object], as_of: str) -> list[dict[str, object]]:
    """Retain provider rows with settlement dates on or before the cutoff."""
    dated = [row for row in rows if isinstance(row, dict) and str(row.get("settlement_date") or "")]
    return [row for row in dated if str(row.get("settlement_date") or "")[:10] <= as_of]


def _latest_finra_row(rows: Sequence[object], as_of: str) -> dict[str, object] | None:
    """Select the newest revision unless its settlement cycle has conflicting values."""
    eligible = _eligible_finra_rows(rows, as_of)
    if not eligible:
        return None
    latest_day = max(str(row.get("settlement_date") or "")[:10] for row in eligible)
    same_day = [row for row in eligible if str(row.get("settlement_date") or "")[:10] == latest_day]
    if len(_finra_variants(same_day)) > 1:
        return None
    return max(same_day, key=_finra_day_key)


def _finra_row_metrics(row: dict[str, object]) -> dict[str, object]:
    """Calculate changes and normalize provenance for one selected revision."""
    short_position = _decimal(row.get("short_position"))
    prev_position = _decimal(row.get("prev_position"))
    change: Decimal | None = None
    if short_position is not None and prev_position is not None:
        change = short_position - prev_position
    change_pct: Decimal | None = None
    if change is not None and prev_position is not None and prev_position != 0:
        change_pct = Decimal(100) * change / prev_position
    return {
        "short_position": short_position,
        "prev_position": prev_position,
        "short_interest_change": change,
        "short_interest_change_pct": change_pct,
        "days_to_cover": _decimal(row.get("days_to_cover")),
        "settlement_date": _iso_date(row.get("settlement_date")),
        "avg_daily_volume": _decimal(row.get("avg_daily_volume")),
        "known_at": _iso_date(row.get("known_at")),
        "retrieved_at": _iso_instant(row.get("retrieved_at")),
    }


def _freshness(as_of: str, sec_metrics: dict[str, object], finra_metrics: dict[str, object]) -> dict[str, object]:
    filed_dates = [
        d
        for fact in sec_metrics.values()
        if isinstance(fact, dict)
        for d in (_parse_date(fact.get("filed_at")),)
        if d is not None
    ]
    return {
        "as_of": as_of,
        "sec_latest_filed_at": max(filed_dates, default=None),
        "finra_settlement_date": _parse_date(finra_metrics.get("settlement_date")),
        "finra_retrieved_at": finra_metrics.get("retrieved_at") or None,
    }


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except TypeError, ValueError:
        return None


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except TypeError, ValueError, ArithmeticError:
        return None
