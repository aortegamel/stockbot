"""Live SEC fact read path behind get_fundamentals/get_xbrl_facts.
Live normalized companyfacts via the SourceGateway, PIT-gated

``known_at <= as_of`` with restatements resolved by the true latest
``filed_at``. Live fallback is the only path: explicit as_of with no
knowable rows returns pit_data_unavailable; otherwise the provider payload
is enveloped with a truthful live data_source.

The live EPS payload carries a human label under the key ``source``; the
envelope needs ``source`` for the provider code, so the payload label key is
renamed ``source`` -> ``source_label`` in every envelope alike.
All other payload keys stay byte-identical.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from .. import edgar_client
from ..data_sources import SourceGateway
from ..domain.market.identity import resolve_ticker_aliases
from ..edgar_client import (
    _DERIVED_Q4_OFFSET_DAYS,
    _DIVIDEND_SOURCE,
    _FY_DAYS,
    _MISSING_QUARTER_GAP_DAYS,
    _QUARTER_DAYS,
    _YTD_DAYS,
    _dividend_annual_history,
    _dividend_growth,
    _dividend_valuation,
    _has_contiguous_quarters,
    _is_recent_dividend_period,
)
from .dividend_analysis import analyze_dividends

DEFAULT_DATA_ROOT = None

DILUTED_EPS_CONCEPT = "EarningsPerShareDiluted"
BASIC_EPS_CONCEPT = "EarningsPerShareBasic"
SHARES_OUTSTANDING_CONCEPT = "EntityCommonStockSharesOutstanding"
DIVIDEND_PER_SHARE_CONCEPT = "CommonStockDividendsPerShareDeclared"

_EPS_CONCEPTS = (DILUTED_EPS_CONCEPT, BASIC_EPS_CONCEPT)
_DIVIDEND_CONCEPTS = (DIVIDEND_PER_SHARE_CONCEPT,)
_METRICS = ("eps", "shares_outstanding", "balance_sheet", "overview", "dividends")


class FinancialFactRow(TypedDict):
    """One normalized financial-fact row (float value)."""

    concept: str
    value: float
    period_end: str
    filed_at: str
    accession: str
    known_at: str
    period_start: str | None
    fiscal_year: int | None
    fiscal_period: str | None
    source_url: str | None


class DividendEventRow(TypedDict):
    """One normalized ``dividend_events`` row; None amounts stay None."""

    dividend_event_id: str
    entity_id: str | None
    security_id: str | None
    ticker: str | None
    amount_per_share: float | None
    currency: str | None
    dividend_type: str | None
    declaration_date: str | None
    record_date: str | None
    payment_date: str | None
    ex_dividend_date: str | None
    ex_dividend_date_source: str | None
    status: str | None
    source_form: str | None
    accession: str | None
    filed_at: str | None
    known_at: str | None
    source_url: str | None
    source_concept: str | None
    source_type: str | None
    evidence_excerpt: str | None
    content_hash: str | None
    parser_version: str | None


class CanonicalDividendEventRow(DividendEventRow, total=False):
    """Amended-duplicate winner plus its merged ``source_types``."""

    source_types: list[str]


StoredFactKey = tuple[str, str, str]
RevisionKey = tuple[str, str, str]


def _today() -> _dt.date:
    return _dt.datetime.now(_dt.UTC).date()


def _validated_as_of(as_of: str | None) -> _dt.date | None:
    """None -> today; otherwise strict YYYY-MM-DD or an error marker."""
    if as_of is None:
        return _today()
    try:
        return _dt.datetime.strptime(as_of, "%Y-%m-%d").replace(tzinfo=_dt.UTC).date()
    except TypeError, ValueError:
        return None


def _stored_text(value: object) -> str:
    """Coerce a live text field: str as-is, date/datetime to ISO, None -> "", else str."""
    from datetime import date, datetime

    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _stored_opt_text(value: object) -> str | None:
    """Coerce an optional live text field: str as-is, date/datetime to ISO, else None."""
    from datetime import date, datetime

    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return None


def _stored_opt_int(value: object) -> int | None:
    """Coerce an optional live int field: int as-is (bool excluded), else None."""
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _validated_fact_row(row: Mapping[str, object]) -> FinancialFactRow | None:
    """Narrow one live normalized fact row to a FinancialFactRow; None when unusable."""
    concept = row.get("concept")
    period_end = _stored_opt_text(row.get("period_end"))
    value = row.get("value")
    if not isinstance(concept, str) or not concept:
        return None
    if not period_end:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return {
        "concept": concept,
        "value": float(value),
        "period_end": period_end,
        "filed_at": _stored_text(row.get("filed_at")),
        "accession": _stored_text(row.get("accession")),
        "known_at": _stored_text(row.get("known_at")),
        "period_start": _stored_opt_text(row.get("period_start")),
        "fiscal_year": _stored_opt_int(row.get("fiscal_year")),
        "fiscal_period": _stored_opt_text(row.get("fiscal_period")),
        "source_url": _stored_opt_text(row.get("source_url")),
    }


def _gateway() -> SourceGateway:
    """Per-call gateway: fetch-once-per-run cache never outlives the call."""
    return SourceGateway()


def _resolve_entity(ticker: str, as_of: _dt.date, data_root: Path | None) -> str | None:
    """Resolve a ticker to its entity id through live company-tickers aliases.

    The alias horizon is end-of-day UTC on the as-of date so an alias
    knowable on the as-of day itself is visible (day-granularity semantics,
    matching the ``known_at <= as_of`` facts gate). Ambiguous or
    unresolved tickers resolve to None: no store path, never a guess.
    No archive-all from fetch: historically-unknown identity stays unknown.
    """
    del data_root
    horizon = _dt.datetime.combine(as_of, _dt.time.max, tzinfo=_dt.UTC)
    aliases = _gateway().ticker_candidates(ticker, horizon)
    if not aliases:
        return None
    resolution = resolve_ticker_aliases(ticker, aliases, as_of=horizon)
    if not resolution.resolved:
        return None
    return resolution.entity_id


def _resolve_cik(entity_id: str) -> int | None:
    """Trailing-digit CIK from a sec:cik entity id or a bare CIK string."""
    try:
        digits = "".join(ch for ch in str(entity_id or "") if ch.isdigit())
        return int(digits) if digits else None
    except TypeError, ValueError:
        return None


def _fact_order_key(fact: FinancialFactRow) -> tuple[str, str, str]:
    return (fact["period_end"], fact.get("filed_at") or "", fact.get("accession") or "")


def _live_fact_rows(entity_id: str, concepts: tuple[str, ...], as_of: _dt.date) -> list[FinancialFactRow]:
    """Live normalized companyfacts for one entity, narrowed to validated rows."""
    cik = _resolve_cik(entity_id)
    if cik is None:
        return []
    try:
        facts = _gateway().company_facts(cik, as_of=as_of.isoformat())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"company facts unavailable for {entity_id}: {exc}") from exc
    rows = facts.get("financial_facts")
    if not isinstance(rows, list):
        return []
    wanted = set(concepts)
    validated: list[FinancialFactRow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("concept") not in wanted:
            continue
        fact = _validated_fact_row(row)
        if fact is not None:
            validated.append(fact)
    validated.sort(key=_fact_order_key)
    return validated


def _store_rows(
    entity_id: str, concepts: tuple[str, ...], as_of: _dt.date, data_root: Path | None
) -> list[FinancialFactRow]:
    """Live fallback is the only path: normalized companyfacts via the gateway."""
    del data_root
    return _live_fact_rows(entity_id, concepts, as_of)


def _envelope(
    ticker: str,
    metric: str,
    payload: Mapping[str, object],
    *,
    data_source: str,
    as_of_date: str,
    requested_as_of: str | None = None,
    row_count: int | None = None,
    returned_count: int | None = None,
    truncated: bool = False,
) -> dict[str, object]:
    """Wrap a payload in the exact public envelope shape."""
    env: dict[str, object] = {
        "source": "sec",
        "metric": metric,
        "data_source": data_source,
        "as_of_date": as_of_date,
        "row_count": row_count,
        "returned_count": returned_count if returned_count is not None else row_count,
        "truncated": truncated,
    }
    if requested_as_of is not None and requested_as_of != as_of_date:
        env["requested_as_of"] = requested_as_of
    for key, value in payload.items():
        env["source_label" if key == "source" else key] = value
    return env


# ---------------------------------------------------------------------------
# EPS: live assembly mirroring edgar_client semantics
# ---------------------------------------------------------------------------


def _duration_days(row: FinancialFactRow) -> int | None:
    start, end = row.get("period_start"), row.get("period_end")
    if not start or not end:
        return None
    try:
        start_date = _dt.date.fromisoformat(start[:10])
        end_date = _dt.date.fromisoformat(end[:10])
    except ValueError:
        return None
    return (end_date - start_date).days


def _row_period_end(row: FinancialFactRow) -> str:
    """Sort key: fact period end as text (ISO dates order lexically)."""
    return row["period_end"]


def _row_payment_date(row: CanonicalDividendEventRow) -> str:
    """Sort key: dividend payment date as text."""
    return str(row["payment_date"])


def _row_known_at(row: Mapping[str, object]) -> str:
    """Sort key: revision recency as text (missing sorts first)."""
    return str(row.get("known_at") or "")


def _row_filed_accession(row: FinancialFactRow) -> tuple[str, str]:
    """Sort key: true latest filing first by filed_at, accession tie-break."""
    return ((row.get("filed_at") or ""), (row.get("accession") or ""))


def _row_stored_order(row: FinancialFactRow) -> tuple[str, str, str]:
    """Sort key: newest stored fact by period end, filed_at, accession."""
    return (row["period_end"], (row.get("filed_at") or ""), (row.get("accession") or ""))


def _row_revision_order(row: Mapping[str, object]) -> tuple[str, str, str]:
    """Sort key: latest revision by known_at, filed_at, accession."""
    return (str(row.get("known_at") or ""), str(row.get("filed_at") or ""), str(row.get("accession") or ""))


def _concept_duration_match(row: FinancialFactRow, concept: str, day_range: tuple[int, int]) -> bool:
    """One fact of the concept whose duration falls inside day_range."""
    if row.get("concept") != concept:
        return False
    duration = _duration_days(row)
    return duration is not None and day_range[0] <= duration <= day_range[1]


def _newer_filed_revision(row: FinancialFactRow, prev: FinancialFactRow) -> bool:
    """True when row supersedes prev by true latest filed_at (accession tie-break)."""
    return ((row["filed_at"] or ""), (row["accession"] or "")) > ((prev["filed_at"] or ""), (prev["accession"] or ""))


def _dedup_latest_by_period_end(q: Sequence[FinancialFactRow]) -> list[FinancialFactRow]:
    """Keep the newest filed revision per period end, newest period first."""
    by_end: dict[str, FinancialFactRow] = {}
    for row in q:
        key = row["period_end"]
        prev = by_end.get(key)
        if prev is None or _newer_filed_revision(row, prev):
            by_end[key] = row
    return sorted(by_end.values(), key=_row_period_end)


def _duration_rows(
    rows: Sequence[FinancialFactRow], concept: str, day_range: tuple[int, int]
) -> list[FinancialFactRow]:
    """Facts of a given duration for a concept, restatements resolved
    by the true latest filed_at (accession DESC tie-break) — replacing
    edgar_client._dedup_latest's fiscal-year proxy — newest period first."""
    return _dedup_latest_by_period_end([row for row in rows if _concept_duration_match(row, concept, day_range)])


def _fy_total_for_year(rows: Sequence[FinancialFactRow], concept: str, fy_end: _dt.date) -> FinancialFactRow | None:
    """Latest FY-duration fact for the fiscal year ending fy_end."""
    fy = [
        row
        for row in rows
        if row.get("concept") == concept
        and row["period_end"] == fy_end.isoformat()
        and _concept_duration_match(row, concept, _FY_DAYS)
    ]
    return max(fy, key=_row_filed_accession) if fy else None


def _ytd_through_q3(rows: Sequence[FinancialFactRow], concept: str, fy_end: _dt.date) -> FinancialFactRow | None:
    """Latest YTD-duration fact strictly before fy_end but within one quarter gap."""
    ytd = []
    for row in rows:
        if row.get("concept") != concept:
            continue
        duration = _duration_days(row)
        if duration is None or not (_YTD_DAYS[0] <= duration <= _YTD_DAYS[1]):
            continue
        try:
            row_end = _dt.date.fromisoformat(row["period_end"][:10])
        except ValueError:
            continue
        if fy_end - _dt.timedelta(days=_MISSING_QUARTER_GAP_DAYS) <= row_end < fy_end:
            ytd.append(row)
    return max(ytd, key=_row_period_end) if ytd else None


def _q4_from_totals(latest_fy: FinancialFactRow, ytd_q3: FinancialFactRow, fy_end: _dt.date) -> FinancialFactRow:
    """Q4 = FY_total - YTD_through_Q3 anchored on the FY fact's identity."""
    return {
        **latest_fy,
        "value": latest_fy["value"] - ytd_q3["value"],
        "period_end": fy_end.isoformat(),
        "fiscal_period": "Q4",
    }


def _derive_q4_from_facts(rows: Sequence[FinancialFactRow], concept: str, fy_end: _dt.date) -> FinancialFactRow | None:
    """Q4 = FY_total - YTD_through_Q3 for the fiscal year ending fy_end."""
    latest_fy = _fy_total_for_year(rows, concept, fy_end)
    if latest_fy is None:
        return None
    ytd_q3 = _ytd_through_q3(rows, concept, fy_end)
    if ytd_q3 is None:
        return None
    return _q4_from_totals(latest_fy, ytd_q3, fy_end)


def _quarters_with_derived_q4(
    quarter_rows: Sequence[FinancialFactRow], all_rows: Sequence[FinancialFactRow], concept: str
) -> list[FinancialFactRow]:
    """Last 4 quarterly facts, deriving a missing final quarter (NVDA reports
    Q4 diluted EPS only as a full-year fact) — mirrors
    edgar_client._quarters_with_derived_q4."""
    quarter = sorted(quarter_rows, key=_row_period_end)
    if len(quarter) < 2:
        return quarter[-4:]
    ends = [_dt.date.fromisoformat(row["period_end"][:10]) for row in quarter]
    last_gap = (ends[-1] - ends[-2]).days
    if last_gap > _MISSING_QUARTER_GAP_DAYS:
        missing_end = ends[-1] - _dt.timedelta(days=_DERIVED_Q4_OFFSET_DAYS)
        derived = _derive_q4_from_facts(all_rows, concept, missing_end)
        if derived is not None:
            quarter = sorted([*quarter, derived], key=_row_period_end)
    return quarter[-4:]


def _recent_concept_quarters(rows: Sequence[FinancialFactRow], concept: str) -> list[FinancialFactRow]:
    """Last-4 quarterly facts for a concept with the derived-Q4 fallback."""
    return _quarters_with_derived_q4(_duration_rows(rows, concept, _QUARTER_DAYS), rows, concept)


def _basic_match_for(period_end: str, recent_basic: Sequence[FinancialFactRow] | None) -> FinancialFactRow | None:
    """Basic-EPS row sharing a diluted row's period end (None when absent)."""
    if not recent_basic:
        return None
    return next((b for b in recent_basic if b["period_end"] == period_end), None)


def _eps_entry(diluted: FinancialFactRow, recent_basic: Sequence[FinancialFactRow] | None) -> dict[str, object]:
    """One quarterly entry: diluted value plus basic when the period matches."""
    entry: dict[str, object] = {
        "fiscal_year": str(diluted["fiscal_year"]) if diluted.get("fiscal_year") is not None else "",
        "fiscal_period": (diluted.get("fiscal_period") or ""),
        "eps_diluted": round(diluted["value"], 2),
        "period_end": diluted["period_end"],
    }
    matching = _basic_match_for(diluted["period_end"], recent_basic)
    if matching is not None:
        entry["eps_basic"] = round(matching["value"], 2)
    return entry


def _eps_ttm(quarters: Sequence[FinancialFactRow] | None) -> float | None:
    """TTM total once 4 quarters are present; None while the series builds."""
    if quarters is None or len(quarters) != 4:
        return None
    return round(sum(r["value"] for r in quarters), 2)


def _assemble_eps_payload(ticker: str, rows: Sequence[FinancialFactRow]) -> dict[str, object] | None:
    """Deterministic live assembly over feed rows (pure; no storage)."""
    recent_diluted = _recent_concept_quarters(rows, DILUTED_EPS_CONCEPT)
    if not recent_diluted:
        return None
    basic_quarters = _duration_rows(rows, BASIC_EPS_CONCEPT, _QUARTER_DAYS)
    recent_basic = _recent_concept_quarters(rows, BASIC_EPS_CONCEPT) if basic_quarters else None
    quarterly_eps = [_eps_entry(r, recent_basic) for r in recent_diluted]
    result: dict[str, object] = {
        "ticker": ticker,
        "quarterly_eps": quarterly_eps,
        "source": "SEC EDGAR company facts (Basic & Diluted EPS)",
    }
    ttm_diluted = _eps_ttm(recent_diluted)
    if ttm_diluted is not None:
        result["ttm_eps_diluted"] = ttm_diluted
    ttm_basic = _eps_ttm(recent_basic)
    if ttm_basic is not None:
        result["ttm_eps_basic"] = ttm_basic
    return result


def _assemble_dividend_payload(
    ticker: str, rows: Sequence[FinancialFactRow], as_of: _dt.date
) -> dict[str, object] | None:
    """Deterministic live assembly over feed rows (pure; no storage)."""
    if not any(r.get("concept") == DIVIDEND_PER_SHARE_CONCEPT for r in rows):
        return None
    quarters = _duration_rows(rows, DIVIDEND_PER_SHARE_CONCEPT, _QUARTER_DAYS)
    recent = _quarters_with_derived_q4(quarters, rows, DIVIDEND_PER_SHARE_CONCEPT)
    if (
        len(recent) == 4
        and _has_contiguous_quarters([r["period_end"] for r in recent])
        and _is_recent_dividend_period(recent[-1]["period_end"], as_of)
    ):
        ttm = round(sum(r["value"] for r in recent), 4)
    else:
        ttm = None
    fy_rows = _duration_rows(rows, DIVIDEND_PER_SHARE_CONCEPT, _FY_DAYS)
    history, annual = _dividend_annual_history(fy_rows)
    return {
        "ticker": ticker,
        "dividend_status": "paying" if ttm is not None else "unknown",
        "ttm_dividend_per_share": ttm,
        **_dividend_growth(annual),
        "annual_history": history,
        "source": _DIVIDEND_SOURCE,
    }


# ---------------------------------------------------------------------------
# Dividend events: past / present / future-declared (no projections)
# ---------------------------------------------------------------------------


def _validated_dividend_event(row: Mapping[str, object]) -> DividendEventRow | None:
    """Narrow one raw ``dividend_events`` row; None amounts stay None, never zero."""
    event_id = row.get("dividend_event_id")
    if not isinstance(event_id, str) or not event_id:
        return None

    def _text(key: str) -> str | None:
        from datetime import date, datetime

        value = row.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return None

    amount = row.get("amount_per_share")
    return {
        "dividend_event_id": event_id,
        "entity_id": _text("entity_id"),
        "security_id": _text("security_id"),
        "ticker": _text("ticker"),
        "amount_per_share": float(amount) if isinstance(amount, (int, float)) else None,
        "currency": _text("currency"),
        "dividend_type": _text("dividend_type"),
        "declaration_date": _text("declaration_date"),
        "record_date": _text("record_date"),
        "payment_date": _text("payment_date"),
        "ex_dividend_date": _text("ex_dividend_date"),
        "ex_dividend_date_source": _text("ex_dividend_date_source"),
        "status": _text("status"),
        "source_form": _text("source_form"),
        "accession": _text("accession"),
        "filed_at": _text("filed_at"),
        "known_at": _text("known_at"),
        "source_url": _text("source_url"),
        "source_concept": _text("source_concept"),
        "source_type": _text("source_type"),
        "evidence_excerpt": _text("evidence_excerpt"),
        "content_hash": _text("content_hash"),
        "parser_version": _text("parser_version"),
    }


def _store_dividend_events(entity_id: str, as_of: _dt.date, data_root: Path | None) -> list[DividendEventRow]:
    """Live PIT-gated dividend events, deduped by id keeping max known_at.

    Amended filings produce different ids (amount is part of the id) so both
    revisions stay visible; duplicate 8-K/10-Q disclosures share an id and
    collapse here.
    """
    del data_root
    cik = _resolve_cik(entity_id)
    if cik is None:
        return []
    try:
        facts = _gateway().company_facts(cik, as_of=as_of.isoformat())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"dividend events unavailable for {entity_id}: {exc}") from exc
    rows = facts.get("dividend_events")
    if not isinstance(rows, list):
        return []
    by_id: dict[str, DividendEventRow] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        event = _validated_dividend_event(row)
        if event is None:
            continue
        key = event["dividend_event_id"]
        prev = by_id.get(key)
        if prev is None or (event.get("known_at") or "") >= (prev.get("known_at") or ""):
            by_id[key] = event
    return list(by_id.values())


def _classify_dividend_event(row: CanonicalDividendEventRow, as_of: _dt.date) -> str:
    """payment_date < as_of -> paid; >= as_of -> upcoming; null -> unknown."""
    pay = row.get("payment_date")
    if not pay:
        return "unknown"
    return "paid" if pay[:10] < as_of.isoformat() else "upcoming"


def _extreme_event(cands: Sequence[CanonicalDividendEventRow], *, earliest: bool) -> CanonicalDividendEventRow | None:
    """Edge payment date wins; same-date revisions prefer latest known_at."""
    edge = (min if earliest else max)(str(r["payment_date"]) for r in cands)
    tied = [r for r in cands if str(r["payment_date"]) == edge]
    return max(tied, key=_row_known_at)


def _is_typed_dividend_row(row: DividendEventRow) -> bool:
    """Concrete dividend_type present (not blank/unknown)."""
    return (row.get("dividend_type") or "") not in ("", "unknown")


def _cluster_typed_rows(group: Sequence[DividendEventRow]) -> list[list[DividendEventRow]]:
    """Group typed rows by exact dividend_type; regular never merges into special."""
    clusters: list[list[DividendEventRow]] = []
    for row in group:
        if not _is_typed_dividend_row(row):
            continue
        for cluster in clusters:
            if all((m.get("dividend_type") or "") == (row.get("dividend_type") or "") for m in cluster):
                cluster.append(row)
                break
        else:
            clusters.append([row])
    return clusters


def _amount_matched_clusters(
    amount: float | None, clusters: list[list[DividendEventRow]]
) -> list[list[DividendEventRow]]:
    """Clusters holding a row with the untyped row's amount (empty when None)."""
    return [c for c in clusters if amount is not None and any(m.get("amount_per_share") == amount for m in c)]


def _place_untyped_row(
    row: DividendEventRow, clusters: list[list[DividendEventRow]], stray: list[DividendEventRow]
) -> None:
    """Attach one unknown row: unique amount match, single-identity amendment, else stray."""
    matched = _amount_matched_clusters(row.get("amount_per_share"), clusters)
    if len(matched) == 1:
        matched[0].append(row)
    elif matched:
        stray.append(row)
    elif len(clusters) <= 1:
        (clusters[0].append(row) if clusters else stray.append(row))
    else:
        stray.append(row)


def _backfill_mate_amount(out: CanonicalDividendEventRow, mate: DividendEventRow) -> None:
    """Winner keeps its amount; a missing amount takes the mate's numeric value."""
    mate_amount = mate.get("amount_per_share")
    if out.get("amount_per_share") is None and isinstance(mate_amount, (int, float)):
        out["amount_per_share"] = float(mate_amount)


def _backfill_mate_type(out: CanonicalDividendEventRow, mate: DividendEventRow) -> None:
    """Unknown winner type takes the mate's concrete type."""
    mate_type = mate.get("dividend_type")
    if (
        (out.get("dividend_type") or "") in ("", "unknown")
        and isinstance(mate_type, str)
        and mate_type not in ("", "unknown")
    ):
        out["dividend_type"] = mate_type


def _backfill_mate_text(out: CanonicalDividendEventRow, mate: DividendEventRow) -> None:
    """Winner keeps its excerpt/concept; missing fields take the mate's."""
    mate_excerpt = mate.get("evidence_excerpt")
    if not out.get("evidence_excerpt") and isinstance(mate_excerpt, str) and mate_excerpt:
        out["evidence_excerpt"] = mate_excerpt
    mate_concept = mate.get("source_concept")
    if not out.get("source_concept") and isinstance(mate_concept, str) and mate_concept:
        out["source_concept"] = mate_concept


def _attach_untyped_rows(
    group: Sequence[DividendEventRow], clusters: list[list[DividendEventRow]]
) -> list[DividendEventRow]:
    """Attach unknown rows to a unique amount-matched cluster (or the single
    typed identity as an amendment); leftovers stay unresolved."""
    stray: list[DividendEventRow] = []
    for row in group:
        if _is_typed_dividend_row(row):
            continue
        _place_untyped_row(row, clusters, stray)
    return stray


def _merge_bucket_mate(out: CanonicalDividendEventRow, mate: DividendEventRow) -> None:
    """Fold one amendment mate into the winner: amount, type, excerpt, concept."""
    _backfill_mate_amount(out, mate)
    _backfill_mate_type(out, mate)
    _backfill_mate_text(out, mate)


def _canonicalize_bucket(bucket: Sequence[DividendEventRow]) -> CanonicalDividendEventRow:
    """Newest revision wins; mates backfill amount/type/excerpt/concept."""
    winner = max(bucket, key=_row_revision_order)
    out: CanonicalDividendEventRow = {
        **winner,
        "source_types": sorted({m.get("source_type") for m in bucket if m.get("source_type")}),
    }
    for mate in bucket:
        if mate is winner:
            continue
        _merge_bucket_mate(out, mate)
    return out


def _canonical_group_events(group: Sequence[DividendEventRow]) -> list[CanonicalDividendEventRow]:
    """One (record_date, payment_date) bucket -> canonical rows."""
    clusters = _cluster_typed_rows(group)
    stray = _attach_untyped_rows(group, clusters)
    if stray:
        clusters.append(stray)
    return [_canonicalize_bucket(bucket) for bucket in clusters]


def _bucket_dated_events(
    events: Sequence[DividendEventRow],
) -> tuple[list[DividendEventRow], dict[tuple[str, str], list[DividendEventRow]]]:
    """Split undated rows (never merge) from (record_date, payment_date) buckets."""
    undated = [r for r in events if not r.get("payment_date")]
    groups: dict[tuple[str, str], list[DividendEventRow]] = {}
    for row in events:
        if not row.get("payment_date"):
            continue
        groups.setdefault(((row.get("record_date") or ""), row.get("payment_date")), []).append(row)
    return undated, groups


def _canonical_dividend_events(events: Sequence[DividendEventRow]) -> list[CanonicalDividendEventRow]:
    """Collapse amended duplicates; undated rows never merge.

    Buckets share (record_date, payment_date). Concrete types cluster exactly
    (regular never merges into special). An unknown row joins a typed cluster
    only on a unique amount match — or when the bucket holds a single typed
    identity (amendment); otherwise it stays unresolved instead of attaching
    to the first compatible bucket.
    """
    undated, groups = _bucket_dated_events(events)
    canonical: list[CanonicalDividendEventRow] = []
    for group in groups.values():
        canonical.extend(_canonical_group_events(group))
    undated_canonical: list[CanonicalDividendEventRow] = [
        {**r, "source_types": [r["source_type"]] if r["source_type"] else []} for r in undated
    ]
    return undated_canonical + canonical


def _dividend_event_source_types(row: CanonicalDividendEventRow) -> list[str]:
    sts = row.get("source_types")
    if isinstance(sts, (list, tuple, set)) and sts:
        return [s for s in sts if s]
    st = row.get("source_type")
    return [st] if st else []


def _classed_events(
    canonical: Sequence[CanonicalDividendEventRow], as_of: _dt.date, status: str
) -> list[CanonicalDividendEventRow]:
    """Canonical events with a classification, amount, and payment date."""
    return [
        r
        for r in canonical
        if _classify_dividend_event(r, as_of) == status
        and r.get("amount_per_share") is not None
        and r.get("payment_date")
    ]


def _past_event_rows(paid: Sequence[CanonicalDividendEventRow], as_of: _dt.date) -> list[dict[str, object]]:
    """Newest-12 paid events as public past-event dicts."""
    return [
        {
            "amount_per_share": r.get("amount_per_share"),
            "payment_date": r.get("payment_date"),
            "declaration_date": r.get("declaration_date"),
            "record_date": r.get("record_date"),
            "dividend_type": r.get("dividend_type"),
            "status": _classify_dividend_event(r, as_of),
            "accession": r.get("accession"),
            "source_url": r.get("source_url"),
        }
        for r in sorted(paid, key=_row_payment_date, reverse=True)[:12]
    ]


def _events_coverage(canonical: Sequence[CanonicalDividendEventRow]) -> str:
    """structured/text coverage over canonical source types."""
    has_xbrl = any("structured_xbrl" in _dividend_event_source_types(r) for r in canonical)
    has_text = any("filing_text" in _dividend_event_source_types(r) for r in canonical)
    if has_xbrl and has_text:
        return "structured_and_text"
    if has_xbrl:
        return "structured_only"
    if has_text:
        return "text_only"
    return "no_structured_events"


def _last_dividend_dict(last: CanonicalDividendEventRow | None) -> dict[str, object] | None:
    """Latest paid event as the public last_dividend shape (None when absent)."""
    if last is None:
        return None
    return {
        "amount_per_share": last["amount_per_share"],
        "payment_date": last["payment_date"],
        "type": last.get("dividend_type"),
    }


def _next_dividend_dict(nxt: CanonicalDividendEventRow | None) -> dict[str, object] | None:
    """Earliest upcoming event as the public next_declared_dividend shape."""
    if nxt is None:
        return None
    return {
        "amount_per_share": nxt["amount_per_share"],
        "declaration_date": nxt.get("declaration_date"),
        "record_date": nxt.get("record_date"),
        "payment_date": nxt["payment_date"],
        "status": "upcoming",
        "source_url": nxt.get("source_url"),
        "accession": nxt.get("accession"),
    }


def _dividend_event_payload(
    events: Sequence[DividendEventRow],
    as_of: _dt.date,
    *,
    growth: Mapping[str, object] | None = None,
    ttm_dps: float | None = None,
    annual_history: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """last/next/past/coverage over classified events (pure; no storage)."""
    canonical = _canonical_dividend_events(events)
    upcoming = _classed_events(canonical, as_of, "upcoming")
    paid = _classed_events(canonical, as_of, "paid")
    nxt = _extreme_event(upcoming, earliest=True) if upcoming else None
    last = _extreme_event(paid, earliest=False) if paid else None
    analysis: dict[str, object] = analyze_dividends(
        paid_events=paid, as_of=as_of, ttm_dps=ttm_dps, growth=growth, annual_history=annual_history
    )
    return {
        "last_dividend": _last_dividend_dict(last),
        "next_declared_dividend": _next_dividend_dict(nxt),
        "past_events": _past_event_rows(paid, as_of),
        "events_coverage": _events_coverage(canonical),
        **analysis,
    }


# ---------------------------------------------------------------------------
# Dividend safety: FCF-based coverage on SEC inputs (no new providers)
# ---------------------------------------------------------------------------

_SAFETY_METHODOLOGY = "common-stock EPS/FCF basis; not AFFO/FFO"

_OCF_CONCEPT = "OperatingCashFlow"
_CAPEX_CONCEPT = "CapEx"
_DIV_PAID_CONCEPT = "DividendsPaid"
_CASH_CONCEPT = "CashAndCashEquivalents"
_DEBT_CONCEPT = "LongTermDebt"
_NET_INCOME_CONCEPT = "NetIncomeLoss"

_SAFETY_CONCEPTS = (
    _OCF_CONCEPT,
    _CAPEX_CONCEPT,
    _DIV_PAID_CONCEPT,
    _CASH_CONCEPT,
    _DEBT_CONCEPT,
    _NET_INCOME_CONCEPT,
    DILUTED_EPS_CONCEPT,
    BASIC_EPS_CONCEPT,
)


def _ttm_cash_total(rows: Sequence[FinancialFactRow], concept: str, *, outflow: bool = False) -> float | None:
    """Trailing-4-quarter total via the shared quarterly + derived-Q4 machinery.

    Outflow concepts (CapEx, dividends paid) are cash-flow debits, usually
    filed as negative values; coverage math needs their magnitude.
    """
    quarters = _quarters_with_derived_q4(_duration_rows(rows, concept, _QUARTER_DAYS), rows, concept)
    if len(quarters) != 4 or not _has_contiguous_quarters([r["period_end"] for r in quarters]):
        return None
    values = [abs(r["value"]) if outflow else r["value"] for r in quarters]
    return round(sum(values), 2)


def _fy_annual_totals(rows: Sequence[FinancialFactRow], concept: str, *, outflow: bool = False) -> dict[int, float]:
    """FY-duration facts keyed by calendar year of period_end."""
    annual: dict[int, float] = {}
    for r in _duration_rows(rows, concept, _FY_DAYS):
        try:
            year = _dt.date.fromisoformat(r["period_end"][:10]).year
            val = abs(r["value"]) if outflow else r["value"]
        except TypeError, ValueError:
            continue
        annual[year] = round(val, 2)
    return annual


def _latest_concept_value(rows: Sequence[FinancialFactRow], concept: str) -> FinancialFactRow | None:
    """Latest row for a concept by (period_end, filed_at, accession)."""
    cands = [r for r in rows if r.get("concept") == concept and r.get("period_end")]
    if not cands:
        return None
    return max(cands, key=_row_stored_order)


def _debt_base_value(latest: FinancialFactRow) -> tuple[_dt.date, float] | None:
    """(period end, value) of the latest debt row; None when unparsable."""
    try:
        return _dt.date.fromisoformat(latest["period_end"][:10]), latest["value"]
    except TypeError, ValueError:
        return None


def _debt_yoy_base(rows: Sequence[FinancialFactRow], cutoff: str) -> FinancialFactRow | None:
    """Latest debt row at least ~10 months older than the newest (None when absent)."""
    older = [r for r in rows if r.get("concept") == _DEBT_CONCEPT and (r.get("period_end") or "") <= cutoff]
    return _latest_concept_value(older, _DEBT_CONCEPT) if older else None


def _debt_up_yoy(rows: Sequence[FinancialFactRow]) -> bool | None:
    """Latest LongTermDebt vs the most recent row at least ~10 months older."""
    latest = _latest_concept_value(rows, _DEBT_CONCEPT)
    if latest is None:
        return None
    parsed = _debt_base_value(latest)
    if parsed is None:
        return None
    end, now_val = parsed
    base = _debt_yoy_base(rows, (end - _dt.timedelta(days=300)).isoformat())
    if base is None:
        return None
    return now_val > base["value"]


def _safety_ttm_inputs(
    rows: Sequence[FinancialFactRow], dividend: Mapping[str, object]
) -> tuple[float | None, float | None, float | None, float | None]:
    """(ttm dps, ocf, capex magnitude, dividends-paid magnitude)."""
    ttm_dps_raw = dividend.get("ttm_dividend_per_share")
    ttm_dps = float(ttm_dps_raw) if isinstance(ttm_dps_raw, (int, float)) else None
    ttm_ocf = _ttm_cash_total(rows, _OCF_CONCEPT)
    ttm_capx = _ttm_cash_total(rows, _CAPEX_CONCEPT, outflow=True)
    ttm_div_paid = _ttm_cash_total(rows, _DIV_PAID_CONCEPT, outflow=True)
    return ttm_dps, ttm_ocf, ttm_capx, ttm_div_paid


def _safety_fcf(ttm_ocf: float | None, ttm_capx: float | None) -> float | None:
    """FCF = OCF - capex once both TTM legs exist."""
    if ttm_ocf is None or ttm_capx is None:
        return None
    return round(ttm_ocf - ttm_capx, 2)


def _safety_set(safety: dict[str, object], key: str, value: float | None, reason: str | None = None) -> None:
    """Nullable ratio assignment with its reason key (never zero-filled)."""
    safety[key] = value
    if value is None and reason:
        safety[f"{key}_reason"] = reason


def _safety_earnings_ratio(
    safety: dict[str, object], flags: list[dict[str, object]], ttm_dps: float | None, ttm_eps_diluted: float | None
) -> None:
    """Earnings payout ratio + negative-EPS flag."""
    if ttm_eps_diluted is not None and ttm_eps_diluted <= 0:
        _safety_set(safety, "earnings_payout_ratio", None, f"ttm_eps_diluted {ttm_eps_diluted} <= 0")
        flags.append(
            {
                "flag": "negative_eps",
                "status": True,
                "basis": f"ttm_eps_diluted {ttm_eps_diluted} <= 0; payout meaningless",
            }
        )
    elif ttm_dps is None or ttm_eps_diluted is None:
        _safety_set(safety, "earnings_payout_ratio", None, "missing ttm dps or diluted eps")
    else:
        _safety_set(safety, "earnings_payout_ratio", round(ttm_dps / ttm_eps_diluted, 4))


def _safety_fcf_ratios(
    safety: dict[str, object], flags: list[dict[str, object]], ttm_fcf: float | None, ttm_div_paid: float | None
) -> None:
    """FCF payout/coverage ratios + nonpositive/zero FCF or dividend flags."""
    if ttm_fcf is not None and ttm_fcf <= 0:
        _safety_set(safety, "fcf_payout_ratio", None, f"ttm_fcf {ttm_fcf} <= 0")
        _safety_set(safety, "fcf_coverage", None, f"ttm_fcf {ttm_fcf} <= 0")
        flags.append(
            {
                "flag": "negative_or_zero_fcf",
                "status": True,
                "basis": f"ttm_fcf {ttm_fcf} <= 0; payout/coverage meaningless",
            }
        )
    elif ttm_div_paid is None or ttm_fcf is None:
        _safety_set(safety, "fcf_payout_ratio", None, "missing ttm dividends-paid or fcf")
        _safety_set(safety, "fcf_coverage", None, "missing ttm dividends-paid or fcf")
    elif ttm_div_paid == 0:
        _safety_set(safety, "fcf_payout_ratio", 0.0)
        _safety_set(safety, "fcf_coverage", None, "zero ttm dividends-paid")
        flags.append(
            {"flag": "zero_dividend", "status": True, "basis": "ttm_dividends_paid is zero; coverage undefined"}
        )
    else:
        _safety_set(safety, "fcf_payout_ratio", round(ttm_div_paid / ttm_fcf, 4))
        _safety_set(safety, "fcf_coverage", round(ttm_fcf / ttm_div_paid, 4))


def _safety_cash_ratio(safety: dict[str, object], rows: Sequence[FinancialFactRow], ttm_div_paid: float | None) -> None:
    """Cash-to-annual-dividend multiple from the latest cash balance."""
    cash_row = _latest_concept_value(rows, _CASH_CONCEPT)
    cash = cash_row["value"] if cash_row is not None else None
    if cash is None or ttm_div_paid is None:
        _safety_set(safety, "cash_to_annual_dividend", None, "missing cash balance or ttm dividends-paid")
    elif ttm_div_paid == 0:
        _safety_set(safety, "cash_to_annual_dividend", None, "zero ttm dividends-paid")
    else:
        _safety_set(safety, "cash_to_annual_dividend", round(cash / ttm_div_paid, 4))


def _safety_annual_tables(
    rows: Sequence[FinancialFactRow],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """(annual fcf, annual dividends-paid, annual earnings-preferred) tables."""
    ocf_ann = _fy_annual_totals(rows, _OCF_CONCEPT)
    capx_ann = _fy_annual_totals(rows, _CAPEX_CONCEPT, outflow=True)
    fcf_ann = {y: round(ocf_ann[y] - capx_ann[y], 2) for y in ocf_ann if y in capx_ann}
    paid_ann = _fy_annual_totals(rows, _DIV_PAID_CONCEPT, outflow=True)
    return fcf_ann, paid_ann, ocf_ann


def _safety_growth_verdict(
    dividend: Mapping[str, object], fcf_ann: dict[int, float]
) -> tuple[float | None, float | None, str]:
    """Dividend-vs-FCF 5y CAGR verdict (insufficient_data unless both exist)."""
    div_cagr_raw = dividend.get("growth_5y_cagr")
    div_cagr = float(div_cagr_raw) if isinstance(div_cagr_raw, (int, float)) else None
    fcf_cagr = _dividend_growth(fcf_ann)["growth_5y_cagr"]
    if div_cagr is None or fcf_cagr is None:
        return div_cagr, fcf_cagr, "insufficient_data"
    if div_cagr - fcf_cagr > 0.02:
        return div_cagr, fcf_cagr, "payout_expanding"
    return div_cagr, fcf_cagr, "runway_supported"


def _safety_yield_flag(flags: list[dict[str, object]], ttm_yield: float | None) -> None:
    """High-absolute-yield flag (unknown without a current yield)."""
    if ttm_yield is None:
        flags.append(
            {
                "flag": "high_absolute_yield",
                "status": None,
                "basis": "no current ttm yield (historical as_of or missing price)",
            }
        )
    else:
        flags.append(
            {"flag": "high_absolute_yield", "status": ttm_yield >= 0.06, "basis": f"ttm_dividend_yield {ttm_yield}"}
        )


def _safety_fcf_declined_flag(flags: list[dict[str, object]], fcf_ann: dict[int, float]) -> None:
    """Year-over-year FCF decline flag over consecutive annual totals."""
    if fcf_ann and max(fcf_ann) - 1 in fcf_ann:
        latest_y = max(fcf_ann)
        flags.append(
            {
                "flag": "fcf_declined_yoy",
                "status": fcf_ann[latest_y] < fcf_ann[latest_y - 1],
                "basis": f"annual fcf {latest_y - 1} {fcf_ann[latest_y - 1]} -> {latest_y} {fcf_ann[latest_y]}",
            }
        )
    else:
        flags.append({"flag": "fcf_declined_yoy", "status": None, "basis": "missing consecutive annual fcf totals"})


def _safety_payout_expanded_flag(
    flags: list[dict[str, object]], fcf_ann: dict[int, float], paid_ann: dict[int, float]
) -> None:
    """FCF payout expansion flag (>10pp YoY) over consecutive payout bases."""
    if fcf_ann and paid_ann and max(fcf_ann) - 1 in fcf_ann and max(fcf_ann) - 1 in paid_ann:
        latest_y = max(fcf_ann)
        prior_y = latest_y - 1
        if fcf_ann[latest_y] > 0 and fcf_ann[prior_y] > 0:
            cur, prev = paid_ann[latest_y] / fcf_ann[latest_y], paid_ann[prior_y] / fcf_ann[prior_y]
            flags.append(
                {
                    "flag": "fcf_payout_expanded",
                    "status": cur - prev > 0.10,
                    "basis": f"annual fcf payout {prev:.4f} ({prior_y}) -> {cur:.4f} ({latest_y})",
                }
            )
        else:
            flags.append({"flag": "fcf_payout_expanded", "status": None, "basis": "non-positive annual fcf base"})
    else:
        flags.append(
            {"flag": "fcf_payout_expanded", "status": None, "basis": "missing consecutive annual payout bases"}
        )


def _safety_eps_declined_flag(flags: list[dict[str, object]], rows: Sequence[FinancialFactRow]) -> None:
    """Earnings decline flag (diluted EPS, net-income fallback) over consecutive years."""
    eps_ann = _fy_annual_totals(rows, DILUTED_EPS_CONCEPT)
    eps_basis_name = "diluted eps"
    if not eps_ann:
        eps_ann = _fy_annual_totals(rows, _NET_INCOME_CONCEPT)
        eps_basis_name = "net income"
    if eps_ann and max(eps_ann) - 1 in eps_ann:
        latest_y = max(eps_ann)
        flags.append(
            {
                "flag": "eps_declined_yoy",
                "status": eps_ann[latest_y] < eps_ann[latest_y - 1],
                "basis": f"annual {eps_basis_name} {latest_y - 1} {eps_ann[latest_y - 1]} -> "
                f"{latest_y} {eps_ann[latest_y]}",
            }
        )
    else:
        flags.append(
            {"flag": "eps_declined_yoy", "status": None, "basis": "missing consecutive annual earnings totals"}
        )


def _safety_leverage_flag(flags: list[dict[str, object]], debt_up: bool | None) -> None:
    """Leverage-rising flag from the YoY debt comparison."""
    if debt_up is None:
        flags.append({"flag": "leverage_rising", "status": None, "basis": "missing current or year-ago long-term debt"})
    else:
        flags.append(
            {
                "flag": "leverage_rising",
                "status": debt_up,
                "basis": f"long-term debt {'up' if debt_up else 'not up'} year-over-year",
            }
        )


def _safety_growth_flag(flags: list[dict[str, object]], dividend: Mapping[str, object]) -> None:
    """Growth-deceleration flag (1y vs 5y CAGR)."""
    g1_raw, g5_raw = dividend.get("growth_1y"), dividend.get("growth_5y_cagr")
    g1 = float(g1_raw) if isinstance(g1_raw, (int, float)) else None
    g5 = float(g5_raw) if isinstance(g5_raw, (int, float)) else None
    if g1 is None or g5 is None:
        flags.append({"flag": "growth_decelerating", "status": None, "basis": "missing growth_1y or growth_5y_cagr"})
    else:
        flags.append(
            {"flag": "growth_decelerating", "status": g1 < g5, "basis": f"growth_1y {g1} vs growth_5y_cagr {g5}"}
        )


def _safety_trend_flags(
    flags: list[dict[str, object]],
    rows: Sequence[FinancialFactRow],
    dividend: Mapping[str, object],
    debt_up: bool | None,
    ttm_yield: float | None,
    fcf_ann: dict[int, float],
    paid_ann: dict[int, float],
) -> None:
    """Yield, FCF, payout, earnings, leverage, and growth risk flags."""
    _safety_yield_flag(flags, ttm_yield)
    _safety_fcf_declined_flag(flags, fcf_ann)
    _safety_payout_expanded_flag(flags, fcf_ann, paid_ann)
    _safety_eps_declined_flag(flags, rows)
    _safety_leverage_flag(flags, debt_up)
    _safety_growth_flag(flags, dividend)


def _assemble_dividend_safety(
    rows: Sequence[FinancialFactRow],
    dividend: Mapping[str, object],
    *,
    ttm_eps_diluted: float | None = None,
    ttm_yield: float | None = None,
) -> dict[str, object]:
    """FCF/EPS safety on SEC inputs only (pure; no storage).

    Absent concepts yield nulls with reasons, never zero-filled or borrowed.
    """
    ttm_dps, ttm_ocf, ttm_capx, ttm_div_paid = _safety_ttm_inputs(rows, dividend)
    ttm_fcf = _safety_fcf(ttm_ocf, ttm_capx)
    safety: dict[str, object] = {
        "ttm_fcf": ttm_fcf,
        "ttm_dividends_paid": ttm_div_paid,
        "methodology": _SAFETY_METHODOLOGY,
    }
    flags: list[dict[str, object]] = []
    _safety_earnings_ratio(safety, flags, ttm_dps, ttm_eps_diluted)
    _safety_fcf_ratios(safety, flags, ttm_fcf, ttm_div_paid)
    _safety_cash_ratio(safety, rows, ttm_div_paid)
    _safety_set(safety, "interest_coverage", None, "operating-income/interest-expense concepts not in store")
    debt_up = _debt_up_yoy(rows)
    safety["debt_up_yoy"] = debt_up
    if debt_up is None:
        safety["debt_up_yoy_reason"] = "missing current or year-ago long-term debt"
    fcf_ann, paid_ann, _ = _safety_annual_tables(rows)
    div_cagr, fcf_cagr, verdict = _safety_growth_verdict(dividend, fcf_ann)
    safety["dividend_vs_fcf_growth_5y"] = {
        "dividend_cagr": div_cagr,
        "fcf_cagr": fcf_cagr,
        "verdict": verdict,
    }
    _safety_trend_flags(flags, rows, dividend, debt_up, ttm_yield, fcf_ann, paid_ann)
    safety["risk_flags"] = flags
    return safety


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def get_fundamentals(ticker: str, metric: str, as_of: str | None = None) -> dict[str, object]:
    """Live fundamentals with a truthful data_source envelope."""
    explicit_as_of = as_of is not None
    requested = _validated_as_of(as_of)
    if requested is None:
        return {
            "error": "as_of must be a date in YYYY-MM-DD format",
            "error_type": "invalid_tool_arguments",
        }
    if metric == "shares_float":
        metric = "shares_outstanding"
    if metric == "eps":
        return _eps_fundamental(ticker, requested, explicit_as_of)
    if metric == "dividends":
        return _dividend_fundamental(ticker, requested, explicit_as_of)
    if metric == "shares_outstanding":
        return _shares_outstanding_fundamental(ticker, requested, explicit_as_of)
    if metric in ("balance_sheet", "overview"):
        return _live_only_fundamental(ticker, metric, requested, explicit_as_of)
    return {"error": f"Unknown metric '{metric}'", "error_type": "invalid_tool_arguments"}


def _pit_unavailable(ticker: str, metric: str, requested: _dt.date) -> dict[str, object]:
    return {
        "error": f"No {metric} data knowable as of {requested.isoformat()} for {ticker}",
        "error_type": "pit_data_unavailable",
    }


def _store_dividend_inputs(ticker: str, requested: _dt.date) -> tuple[list[FinancialFactRow], list[DividendEventRow]]:
    """Live rows + dividend events for one ticker at the requested date."""
    entity_id = _resolve_entity(ticker, requested, None)
    store_rows: list[FinancialFactRow] = (
        _store_rows(entity_id, _DIVIDEND_CONCEPTS + _SAFETY_CONCEPTS, requested, None) if entity_id else []
    )
    events: list[DividendEventRow] = _store_dividend_events(entity_id, requested, None) if entity_id else []
    return store_rows, events


def _unknown_status_payload(
    ticker: str, payload: dict[str, object] | None, events: Sequence[DividendEventRow]
) -> dict[str, object] | None:
    """Events-only fallback payload when XBRL facts are absent."""
    if payload is None and events:
        return {
            "ticker": ticker,
            "dividend_status": "unknown",
            "ttm_dividend_per_share": None,
            **_dividend_growth({}),
            "annual_history": [],
            "source": _DIVIDEND_SOURCE,
        }
    return payload


def _eps_ttm_for_safety(ticker: str, store_rows: Sequence[FinancialFactRow]) -> float | None:
    """Diluted TTM EPS for the safety section (None while the series builds)."""
    eps_payload = _assemble_eps_payload(ticker, store_rows)
    eps_ttm_raw = eps_payload.get("ttm_eps_diluted") if eps_payload else None
    return eps_ttm_raw if isinstance(eps_ttm_raw, (int, float)) else None


def _events_analysis_inputs(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], float | None, list[Mapping[str, object]] | None]:
    """Growth/TTM/history inputs the event payload derives from the XBRL payload."""
    ttm_raw = payload.get("ttm_dividend_per_share")
    ttm_dps = float(ttm_raw) if isinstance(ttm_raw, (int, float)) else None
    history_raw = payload.get("annual_history")
    annual_history = [h for h in history_raw if isinstance(h, Mapping)] if isinstance(history_raw, list) else None
    growth = {"growth_1y": payload.get("growth_1y"), "growth_5y_cagr": payload.get("growth_5y_cagr")}
    return growth, ttm_dps, annual_history


def _store_dividend_payload(
    ticker: str,
    payload: Mapping[str, object],
    events: Sequence[DividendEventRow],
    store_rows: Sequence[FinancialFactRow],
    requested: _dt.date,
    *,
    current: bool,
) -> dict[str, object]:
    """Live-path merged payload: events analysis + valuation + safety."""
    valuation = _dividend_valuation(ticker, payload.get("ttm_dividend_per_share"), include_price=current)
    growth, ttm_dps, annual_history = _events_analysis_inputs(payload)
    safety = _assemble_dividend_safety(
        store_rows,
        payload,
        ttm_eps_diluted=_eps_ttm_for_safety(ticker, store_rows),
    )
    return {
        **payload,
        **_dividend_event_payload(events, requested, growth=growth, ttm_dps=ttm_dps, annual_history=annual_history),
        **valuation,
        "safety": safety,
    }


def _live_dividend_payload(ticker: str, requested: _dt.date, *, current: bool) -> dict[str, object]:
    """Live fallback payload with empty events and no safety section."""
    payload = edgar_client.get_fundamentals(ticker, "dividends", include_dividend_price=current)
    if "error" in payload:
        return payload
    live_history = payload.get("annual_history")
    merged = {**payload, **_dividend_event_payload([], requested), "safety": None}
    return _envelope(
        ticker,
        "dividends",
        merged,
        data_source="live",
        as_of_date=_today().isoformat(),
        requested_as_of=requested.isoformat(),
        row_count=len(live_history) if isinstance(live_history, list) else 0,
    )


def _dividend_fundamental(ticker: str, requested: _dt.date, explicit_as_of: bool = False) -> dict[str, object]:
    store_rows, events = _store_dividend_inputs(ticker, requested)
    payload = _assemble_dividend_payload(ticker, store_rows, requested) if store_rows else None
    payload = _unknown_status_payload(ticker, payload, events)
    current = (requested == _today()) and not explicit_as_of
    if payload is not None:
        merged = _store_dividend_payload(ticker, payload, events, store_rows, requested, current=current)
        history_count = merged.get("annual_history")
        return _envelope(
            ticker,
            "dividends",
            merged,
            data_source="live",
            as_of_date=requested.isoformat(),
            row_count=len(history_count) if isinstance(history_count, list) else 0,
        )
    if explicit_as_of:
        return _pit_unavailable(ticker, "dividends", requested)
    return _live_dividend_payload(ticker, requested, current=current)


def _eps_fundamental(ticker: str, requested: _dt.date, explicit_as_of: bool = False) -> dict[str, object]:
    entity_id = _resolve_entity(ticker, requested, None)
    store_rows: list[FinancialFactRow] = _store_rows(entity_id, _EPS_CONCEPTS, requested, None) if entity_id else []
    payload: dict[str, object] | None = _assemble_eps_payload(ticker, store_rows) if store_rows else None
    if payload is not None:
        quarters_count = payload.get("quarterly_eps")
        return _envelope(
            ticker,
            "eps",
            payload,
            data_source="live",
            as_of_date=requested.isoformat(),
            row_count=len(quarters_count) if isinstance(quarters_count, list) else 0,
        )
    if explicit_as_of:
        return _pit_unavailable(ticker, "eps", requested)
    payload = edgar_client.get_fundamentals(ticker, "eps")
    if "error" in payload:
        return payload
    quarters_raw = payload.get("quarterly_eps")
    quarters: list[object] = quarters_raw if isinstance(quarters_raw, list) else []
    return _envelope(
        ticker,
        "eps",
        payload,
        data_source="live",
        as_of_date=_today().isoformat(),
        requested_as_of=requested.isoformat(),
        row_count=len(quarters),
    )


def _envelope_for_live(
    ticker: str,
    metric: str,
    payload: Mapping[str, object],
    requested: _dt.date | None,
    row_count: int,
    *,
    returned_count: int | None = None,
    truncated: bool = False,
) -> dict[str, object]:
    """Live-source envelope: today as_of_date stamped with the requested date."""
    return _envelope(
        ticker,
        metric,
        payload,
        data_source="live",
        as_of_date=_today().isoformat(),
        requested_as_of=requested.isoformat() if requested is not None else None,
        row_count=row_count,
        returned_count=returned_count,
        truncated=truncated,
    )


def _latest_shares_row(
    requested: _dt.date, data_root: Path | None, entity_id: str
) -> tuple[dict[str, object], float] | None:
    """Newest (row, shares value) for shares-outstanding, else None."""
    del data_root
    rows = _live_fact_rows(entity_id, (SHARES_OUTSTANDING_CONCEPT,), requested)
    if not rows:
        return None
    latest = max(rows, key=_fact_order_key)
    if not isinstance(latest["value"], (int, float)):
        return None
    return dict(latest), float(latest["value"])


def _shares_store_payload(ticker: str, row: Mapping[str, object], shares_value: float | None) -> dict[str, object]:
    """Live-path shares-outstanding payload anchored on the latest row."""
    return {
        "ticker": ticker,
        "shares_outstanding": shares_value,
        "as_of": _stored_opt_text(row.get("period_end")) or "",
        "source": "SEC EDGAR company facts",
        "note": "SEC-reported shares outstanding, not public float",
        "filed_at": _stored_text(row.get("filed_at")),
        "accession": row.get("accession"),
        "source_url": row.get("source_url"),
        "known_at": _stored_text(row.get("known_at")),
    }


def _shares_outstanding_fundamental(
    ticker: str, requested: _dt.date, explicit_as_of: bool = False
) -> dict[str, object]:
    entity_id = _resolve_entity(ticker, requested, None)
    row: dict[str, object] | None = None
    shares_value: float | None = None
    if entity_id:
        latest = _latest_shares_row(requested, None, entity_id)
        if latest is not None:
            row, shares_value = latest
    if row is not None:
        return _envelope(
            ticker,
            "shares_outstanding",
            _shares_store_payload(ticker, row, shares_value),
            data_source="live",
            as_of_date=requested.isoformat(),
            row_count=1,
        )
    if explicit_as_of:
        return _pit_unavailable(ticker, "shares_outstanding", requested)
    payload = edgar_client.get_fundamentals(ticker, "shares_outstanding")
    if "error" in payload:
        return payload
    return _envelope_for_live(ticker, "shares_outstanding", payload, requested, 1)


def _live_row_count(metric: str, payload: Mapping[str, object]) -> int:
    """1, or the balance-sheet entry count for balance_sheet payloads."""
    if metric == "balance_sheet":
        sheet = payload.get("balance_sheet")
        if isinstance(sheet, dict):
            return len(sheet)
    return 1


def _live_only_fundamental(
    ticker: str, metric: str, requested: _dt.date, explicit_as_of: bool = False
) -> dict[str, object]:
    """balance_sheet/overview are live-only: explicit as_of never gets current data."""
    if explicit_as_of:
        return _pit_unavailable(ticker, metric, requested)
    payload = edgar_client.get_fundamentals(ticker, metric)
    if "error" in payload:
        return payload
    row_count = _live_row_count(metric, payload)
    return _envelope_for_live(ticker, metric, payload, requested, row_count)


def _xbrl_counts(payload: Mapping[str, object]) -> tuple[list[object], int]:
    """(matching concepts, count) with the count defaulting to len(matching)."""
    matching_raw = payload.get("matching_concepts")
    matching: list[object] = matching_raw if isinstance(matching_raw, list) else []
    count_raw = payload.get("count")
    count: int = count_raw if isinstance(count_raw, int) and count_raw else len(matching)
    return matching, count


def get_xbrl_facts(ticker: str, concept: str) -> dict[str, object]:
    """Always-live XBRL fact search, enveloped (label key source_label)."""
    payload = edgar_client.get_xbrl_facts(ticker, concept)
    if "error" in payload:
        return payload
    matching, count = _xbrl_counts(payload)
    return _envelope_for_live(
        ticker,
        "concept",
        payload,
        None,
        count,
        returned_count=len(matching),
        truncated=count > len(matching),
    )
