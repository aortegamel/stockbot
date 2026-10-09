"""Deterministic, point-in-time screens over live providers.

The canonical short-interest leaderboard lives here: it reads FINRA short
interest (full settlement snapshot, normalized in memory) and SEC facts
through the SourceGateway seam (providers are authoritative),
enforces ``known_at <= as_of`` on every fact join, classifies eligible
equities, and returns a bounded result to the agent tool. Derived market
tables are ephemeral per invocation; the model-visible output is logged to
the run bundle by the caller.
"""

from __future__ import annotations

import hashlib
import json
import time
from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from .. import finra_client
from ..config import finra_use_mock
from ..domain.market.identity import resolve_ticker_aliases
from ..normalization import normalize_finra_short_interest

if TYPE_CHECKING:
    from ..data_sources import SourceGateway

SCREEN_CALC_VERSION = "short-interest-leaderboard-v2"
SLICE_CALC_VERSION = "short-interest-change-slice-v1"
DEFAULT_LIMIT = 10
MAX_LIMIT = 25
SCREEN_NAME = "short_interest_leaderboard"
SLICE_NAME = "short_interest_change"
_SHARES_CONCEPT = "EntityCommonStockSharesOutstanding"

_COMMON_EQUITY = "equity-common"
_SHORT_FIELDS = (
    "symbolCode",
    "issueName",
    "settlementDate",
    "currentShortPositionQuantity",
    "previousShortPositionQuantity",
    "averageDailyVolumeQuantity",
    "daysToCoverQuantity",
)


def _resolve_as_of(as_of: str | None) -> str:
    """Knowledge horizon for a screen request.

    When as_of is omitted, the horizon is today's UTC date: the live screen
    sees everything fetched so far.  Historical reproduction must pass an
    explicit as_of, which then gates every FINRA row, ticker alias, security
    classification, and SEC fact via ``known_at <= as_of``.
    """
    if as_of:
        return as_of
    return datetime.now(UTC).date().isoformat()


def _as_of_datetime(as_of: str) -> datetime:
    """Aware as_of instant; date-only values mean end of that UTC day.

    Provider rows carry instant known_at timestamps, so a date horizon must
    cover the whole day to preserve the historical DATE-granularity PIT
    rule (anything knowable on that date is visible).  Timestamp-precise
    horizons pass through normalized to UTC.
    """
    moment = datetime.fromisoformat(as_of)
    if moment.tzinfo is None:
        if len(as_of.strip()) <= 10:
            moment = datetime.combine(date.fromisoformat(as_of.strip()), dtime.max, tzinfo=UTC)
        else:
            moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _date_str(value: object) -> str:
    """ISO date string for TEXT or TIMESTAMPTZ column values (existing boundary)."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    return str(value)


def _clamp_limit(limit: int | None) -> int:
    try:
        return max(1, min((limit if limit is not None else DEFAULT_LIMIT), MAX_LIMIT))
    except TypeError, ValueError:
        return DEFAULT_LIMIT


def _gateway() -> SourceGateway:
    from ..data_sources import SourceGateway

    return SourceGateway()


def _cik_of(entity_id: str) -> int | None:
    """CIK int for SEC entity ids (None when not an SEC identity)."""
    if not entity_id.startswith("sec:cik:"):
        return None
    try:
        return int(entity_id.split(":")[-1])
    except ValueError:
        return None


def _fetch_settlement_rows(settlement_date: str) -> list[dict[str, object]]:
    """Live FINRA snapshot for one settlement date, normalized in memory.

    Pages the authoritative FINRA query API with Record-Total completeness
    proof (same envelope as the ingestion path) and normalizes through the
    shared normalizer; rows are never persisted.  Transport failures raise
    to the caller's best-effort boundary.
    """
    name = "consolidatedShortInterest" + ("Mock" if finra_use_mock() else "")
    url = f"{finra_client.FINRA_API_BASE}/data/group/otcMarket/name/{name}"
    all_rows: list[dict[str, object]] = []
    total: int | None = None
    offset = 0
    while True:
        time.sleep(0.2)  # politeness pacing, same interval as the ingestion path
        payload: dict[str, object] = {
            "limit": finra_client.MAX_LIMIT,
            "offset": offset,
            "fields": list(_SHORT_FIELDS),
            "compareFilters": [
                {
                    "compareType": "EQUAL",
                    "fieldName": "settlementDate",
                    "fieldValue": settlement_date,
                }
            ],
        }
        _content, rows, headers = finra_client.ingestion_post_query("otcMarket", name, payload)
        raw_total = headers.get("record-total")
        if raw_total is None:
            raise ValueError("FINRA omitted Record-Total; cannot prove the short-interest snapshot is complete.")
        page_total = int(str(raw_total))
        if total is not None and page_total != total:
            raise ValueError("FINRA Record-Total changed while paging the snapshot.")
        total = page_total
        page_rows: list[dict[str, object]] = (
            [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        )
        if not page_rows and len(all_rows) < total:
            raise ValueError("FINRA pagination ended before the complete short-interest snapshot was retrieved.")
        all_rows.extend(page_rows)
        offset += len(page_rows)
        if len(all_rows) >= total:
            break
    if len(all_rows) != total:
        raise ValueError("FINRA pagination returned an incomplete short-interest snapshot.")
    retrieved_at = datetime.now(UTC).isoformat()
    snapshot_hash = hashlib.sha256(
        json.dumps(all_rows, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    normalized = normalize_finra_short_interest(
        all_rows,
        settlement_date=settlement_date,
        retrieved_at=retrieved_at,
        content_hash=snapshot_hash,
        source_url=url,
        source_record_id=f"otcMarket/consolidatedShortInterest:{settlement_date}",
    )
    short_rows = normalized.get("short_interest")
    return [row for row in short_rows if isinstance(row, dict)] if isinstance(short_rows, list) else []


def _snapshot_rows(settlement_date: str, as_of: str) -> list[dict[str, object]]:
    """Short-interest rows for one settlement cycle, point-in-time.

    Only rows knowable on/before ``as_of`` are visible (date granularity).
    A single live fetch carries one source version per symbol, so there are
    no conflicting versions to exclude.
    """
    rows = [row for row in _fetch_settlement_rows(settlement_date) if str(row.get("known_at") or "")[:10] <= as_of]
    rows.sort(key=_symbol_code_key)
    return rows


def _symbol_code_key(row: dict[str, object]) -> str:
    return str(row.get("symbol_code") or "")


def _shares_fact_key(fact: dict[str, object]) -> tuple[str, str, str]:
    return (str(fact.get("filed_at") or ""), str(fact.get("period_end") or ""), str(fact.get("accession") or ""))


def latest_settlement_date(as_of: str | None = None, data_root: Path | None = None) -> str:
    """Latest published settlement cycle, optionally restricted to cycles
    knowable on or before ``as_of``."""
    del data_root
    resolved = _resolve_as_of(as_of)
    for candidate in _candidate_settlement_dates(date.fromisoformat(resolved)):
        try:
            if _probe_published_rows(candidate) > 0:
                return candidate
        except OSError, ValueError, RuntimeError:
            continue
    horizon = f" knowable on or before {resolved}" if as_of else ""
    raise ValueError(f"No FINRA short interest cycle is published{horizon}.")


class _ScreenInputs:
    """Resolved point-in-time maps for one screen build (existing boundary)."""

    def __init__(
        self,
        resolutions: dict[str, object],
        security_types: dict[str, str],
        facts_by_entity: dict[str, list[dict[str, object]]],
    ) -> None:
        self.resolutions = resolutions
        self.security_types = security_types
        self.facts_by_entity = facts_by_entity


def _screen_resolutions(symbols: list[str], as_of: str, gateway: SourceGateway) -> dict[str, object]:
    """Per-symbol ticker resolutions over live company-tickers aliases."""
    moment = _as_of_datetime(as_of)
    resolutions: dict[str, object] = {}
    for symbol in symbols:
        aliases = gateway.ticker_candidates(symbol, moment)
        resolutions[symbol] = resolve_ticker_aliases(symbol, aliases, as_of=moment)
    return resolutions


def _screen_inputs(symbols: list[str], as_of: str, gateway: SourceGateway) -> _ScreenInputs:
    """Point-in-time resolution/classification/fact maps (existing boundary).

    Facts and classifications come from live company-facts payloads via the
    gateway (PIT-filtered to ``as_of``); a per-CIK fetch failure reads as
    absent facts for that entity, never blocks sibling entities.
    """
    from ..domain.market.securities import SecurityResolution

    resolutions = _screen_resolutions(symbols, as_of, gateway)
    resolved_ids = {
        res.entity_id
        for res in resolutions.values()
        if isinstance(res, SecurityResolution) and res.resolved and res.entity_id
    }
    security_types: dict[str, str] = {}
    facts_by_entity: dict[str, list[dict[str, object]]] = {}
    for entity_id in sorted(resolved_ids):
        cik = _cik_of(entity_id)
        if cik is None:
            security_types[entity_id] = "unknown"
            facts_by_entity[entity_id] = []
            continue
        try:
            facts: dict[str, object] = gateway.company_facts(cik, as_of=as_of)
        except OSError, ValueError, RuntimeError:
            facts = {}
        securities = facts.get("securities") if isinstance(facts, dict) else None
        if isinstance(securities, list):
            for sec in securities:
                if isinstance(sec, dict) and str(sec.get("entity_id") or "") == entity_id:
                    security_types[entity_id] = str(sec.get("security_type") or "unknown")
        raw_facts = facts.get("financial_facts") if isinstance(facts, dict) else None
        entity_facts = [
            fact
            for fact in (raw_facts if isinstance(raw_facts, list) else [])
            if isinstance(fact, dict) and fact.get("concept") == _SHARES_CONCEPT
        ]
        entity_facts.sort(key=_shares_fact_key, reverse=True)
        facts_by_entity[entity_id] = entity_facts
    return _ScreenInputs(resolutions, security_types, facts_by_entity)


class _ScreenAccum:
    """Exclusions, stage counters, and candidates (existing boundary)."""

    def __init__(self) -> None:
        self.exclusions = {
            "unmapped_symbol": 0,
            "ambiguous_ticker_mapping": 0,
            "not_classified_common_equity": 0,
            "missing_shares_outstanding": 0,
            "invalid_short_interest": 0,
            "conflicting_versions": 0,
        }
        # Stage counters are cumulative complements of the exclusions: a row
        # excluded at an earlier stage never reached the later checks, so the
        # CLI reports these directly instead of deriving them from exclusions.
        self.counters = {
            "valid_short_interest_rows": 0,
            "mapped_rows": 0,
            "unambiguous_rows": 0,
            "common_equity_rows": 0,
            "shares_outstanding_rows": 0,
        }
        self.candidates: list[dict[str, object]] = []


def _empty_screen_error(settlement_date: str, as_of: str) -> dict[str, object]:
    """Empty-snapshot error envelope (existing boundary)."""
    return {
        "error": (
            f"No FINRA short interest for settlement date "
            f"{settlement_date} is knowable on or before {as_of} "
            f"(live screens fetch the current cycle; omit as_of)."
        )
    }


def _short_shares(row: dict[str, object], accum: _ScreenAccum) -> float | None:
    """Validated short shares, None when invalid (existing boundary)."""
    short_shares_raw = row.get("short_position")
    if short_shares_raw is None or float(str(short_shares_raw)) < 0:
        accum.exclusions["invalid_short_interest"] += 1
        return None
    accum.counters["valid_short_interest_rows"] += 1
    return float(str(short_shares_raw))


def _screen_entity(symbol: str, inputs: _ScreenInputs, accum: _ScreenAccum) -> str | None:
    """Mapped unambiguous entity, None when excluded (existing boundary)."""
    from ..domain.market.securities import SecurityResolution

    resolution = inputs.resolutions.get(symbol)
    if not isinstance(resolution, SecurityResolution) or (
        not resolution.resolved and resolution.resolution_method == "unresolved"
    ):
        accum.exclusions["unmapped_symbol"] += 1
        return None
    accum.counters["mapped_rows"] += 1
    if not resolution.resolved or resolution.entity_id is None:
        accum.exclusions["ambiguous_ticker_mapping"] += 1
        return None
    accum.counters["unambiguous_rows"] += 1
    return resolution.entity_id


def _screen_fact(
    entity_id: str, settlement_date: str, inputs: _ScreenInputs, accum: _ScreenAccum
) -> dict[str, object] | None:
    """Eligible shares-outstanding fact, None when excluded (existing boundary)."""
    # Eligibility is the security classification, not a fact-
    # presence proxy: only entities classified as common equity rank.
    if inputs.security_types.get(entity_id) != _COMMON_EQUITY:
        accum.exclusions["not_classified_common_equity"] += 1
        return None
    accum.counters["common_equity_rows"] += 1
    fact = _select_fact_for_period(inputs.facts_by_entity.get(entity_id) or [], settlement_date)
    if fact is None:
        # Classified common equity but no shares-outstanding fact
        # knowable on/before as_of with period end <= settlement: a data
        # gap, not proof of non-common-equity.
        accum.exclusions["missing_shares_outstanding"] += 1
        return None
    accum.counters["shares_outstanding_rows"] += 1
    return fact


def _screen_candidate(
    symbol: str, row: dict[str, object], entity_id: str, short_shares: float, fact: dict[str, object]
) -> dict[str, object]:
    """One ranked candidate row (existing boundary)."""
    shares = float(str(fact["value"]))
    return {
        "entity_id": entity_id,
        "security_id": f"sec:equity:{entity_id.rsplit(':', 1)[1]}",
        "ticker": symbol,
        "issue_name": row.get("issue_name"),
        "short_shares": short_shares,
        "shares_outstanding": shares,
        "short_interest_percent": 100 * short_shares / shares,
        "sec_shares_as_of": _date_str(fact["period_end"]),
        "sec_filed_at": _date_str(fact["filed_at"]),
        "sec_accession": fact.get("accession"),
        "sec_source_url": fact.get("source_url"),
    }


def _accumulate_screen_row(
    row: dict[str, object], settlement_date: str, inputs: _ScreenInputs, accum: _ScreenAccum
) -> None:
    """One snapshot row through the eligibility pipeline (existing boundary)."""
    symbol = str(row["symbol_code"])
    short_shares = _short_shares(row, accum)
    if short_shares is None:
        return
    entity_id = _screen_entity(symbol, inputs, accum)
    if entity_id is None:
        return
    fact = _screen_fact(entity_id, settlement_date, inputs, accum)
    if fact is None:
        return
    accum.candidates.append(_screen_candidate(symbol, row, entity_id, short_shares, fact))


def _leaderboard_key(item: dict[str, object]) -> tuple[float, str]:
    """Short-interest percent descending, ticker ascending."""
    return (-float(str(item["short_interest_percent"])), str(item["ticker"]))


def _unresolved_symbols(inputs: _ScreenInputs) -> list[str]:
    """Symbols with no resolved entity: explicit unknown, never silent."""
    from ..domain.market.securities import SecurityResolution

    out: list[str] = []
    for symbol, res in inputs.resolutions.items():
        if not isinstance(res, SecurityResolution) or not res.resolved:
            out.append(symbol)
    return out


def _compute_leaderboard(settlement_date: str, as_of: str, limit: int | None) -> dict[str, object]:
    """Build one complete settlement-date leaderboard from live providers."""
    limit = _clamp_limit(limit)
    rows = _snapshot_rows(settlement_date, as_of)
    if not rows:
        return _empty_screen_error(settlement_date, as_of)
    gateway = _gateway()
    symbols = [str(row["symbol_code"]) for row in rows]
    inputs = _screen_inputs(symbols, as_of, gateway)
    accum = _ScreenAccum()
    for row in rows:
        _accumulate_screen_row(row, settlement_date, inputs, accum)
    ranked = sorted(accum.candidates, key=_leaderboard_key)
    entries = ranked[:limit]
    try:
        days = (datetime.now(UTC).date() - date.fromisoformat(settlement_date)).days
        freshness = "stale" if days > finra_client.STALE_AFTER_DAYS else "current"
    except TypeError, ValueError:
        freshness = "unknown"
    return {
        "source": "FINRA consolidated short interest + SEC EDGAR company facts (live providers)",
        "metric": "short shares divided by SEC-reported shares outstanding (not public float)",
        "settlement_date": settlement_date,
        "as_of_date": as_of,
        "data_freshness": freshness,
        "calculation_version": SCREEN_CALC_VERSION,
        "environment": finra_client._environment(),
        "row_count": len(ranked),
        "returned_count": len(entries),
        "truncated": len(entries) < len(ranked),
        "coverage": {
            "finra_rows": len(rows),
            "eligible_rows": len(ranked),
            "valid_short_interest_rows": accum.counters["valid_short_interest_rows"],
            "mapped_rows": accum.counters["mapped_rows"],
            "unambiguous_rows": accum.counters["unambiguous_rows"],
            "common_equity_rows": accum.counters["common_equity_rows"],
            "shares_outstanding_rows": accum.counters["shares_outstanding_rows"],
            "exclusions": accum.exclusions,
        },
        "unresolved": sorted(_unresolved_symbols(inputs)),
        "source_records": [
            f"FINRA otcMarket/consolidatedShortInterest (settlement {settlement_date})",
            "SEC company_tickers.json",
            "SEC companyfacts (EntityCommonStockSharesOutstanding)",
        ],
        "entries": [
            {
                "rank": index,
                "ticker": entry["ticker"],
                "issue_name": entry["issue_name"],
                "short_shares": entry["short_shares"],
                "shares_outstanding": entry["shares_outstanding"],
                "short_interest_percent": entry["short_interest_percent"],
                "sec_shares_as_of": entry["sec_shares_as_of"],
                "sec_filed_at": entry["sec_filed_at"],
                "sec_accession": entry["sec_accession"],
                "sec_source_url": entry["sec_source_url"],
            }
            for index, entry in enumerate(entries, 1)
        ],
    }


def materialize_short_interest_screen(
    settlement_date: str,
    as_of: str | None = None,
    data_root: Path | None = None,
) -> dict[str, object]:
    """Build one complete settlement-date leaderboard from live providers.

    ``as_of`` is the knowledge horizon: FINRA rows, ticker aliases, security
    classifications, and SEC facts are all restricted to ``known_at <=
    as_of``.  When omitted it defaults to today (the live screen);
    historical reproduction passes an explicit as_of.

    The ranking is deterministic: same settlement date, same ``as_of``, same
    provider data -> identical ranking.  The result is ephemeral (never
    persisted); the caller logs the model-visible output to the run bundle.
    """
    del data_root
    return _compute_leaderboard(settlement_date, _resolve_as_of(as_of), DEFAULT_LIMIT)


_FETCH_DISCOVERY_CYCLES = 6


def _raw_cycle_dates(year: int, month: int) -> list[date]:
    """Month-end then mid-month raw settlement dates (existing boundary)."""
    return [date(year, month, monthrange(year, month)[1]), date(year, month, 15)]


def _shift_candidates(raw: date, today: date, candidates: list[str]) -> None:
    """Up-to-3 preceding weekdays for one raw date (existing boundary)."""
    for offset in range(4):
        candidate = raw - timedelta(days=offset)
        if offset and candidate.weekday() >= 5:
            continue
        if candidate <= today and str(candidate) not in candidates:
            candidates.append(str(candidate))


def _prev_month(year: int, month: int) -> tuple[int, int]:
    """One month back with year rollover (existing boundary)."""
    month -= 1
    if month == 0:
        return year - 1, 12
    return year, month


def _candidate_settlement_dates(today: date, count: int = _FETCH_DISCOVERY_CYCLES) -> list[str]:
    """Newest-first FINRA settlement calendar dates on/before ``today``.

    FINRA publishes mid-month (15th) and month-end cycles, shifted to a
    business day when the calendar date hits a weekend or holiday. The
    shift is resolved by the 1-row probe in
    ``_discover_latest_published_settlement_date``, not by weekday
    arithmetic here: each raw date is emitted with up to 3 preceding
    weekdays (covers weekend + single-holiday shifts), newest-first
    deduped, and the first candidate with published rows wins.
    """
    candidates: list[str] = []
    year, month = today.year, today.month
    while len(candidates) < count:
        for raw in _raw_cycle_dates(year, month):
            _shift_candidates(raw, today, candidates)
            if len(candidates) >= count:
                break
        year, month = _prev_month(year, month)
    return candidates


def _probe_published_rows(candidate: str) -> int:
    """1-row FINRA probe: Record-Total tells whether ``candidate`` published."""
    name = "consolidatedShortInterest" + ("Mock" if finra_use_mock() else "")
    _, _, headers = finra_client.ingestion_post_query(
        "otcMarket",
        name,
        {
            "limit": 1,
            "offset": 0,
            "fields": ["settlementDate"],
            "compareFilters": [
                {
                    "compareType": "EQUAL",
                    "fieldName": "settlementDate",
                    "fieldValue": candidate,
                }
            ],
        },
    )
    try:
        return int(str(headers.get("record-total", 0)))
    except TypeError, ValueError:
        return 0


def _discover_latest_published_settlement_date(today: date) -> str | None:
    """Newest FINRA-published settlement date, newest candidate first.

    Probes are best-effort: a failed probe skips that candidate.  Returns
    None when no candidate has published rows.
    """
    for candidate in _candidate_settlement_dates(today):
        try:
            if _probe_published_rows(candidate) > 0:
                return candidate
        except OSError, ValueError, RuntimeError:
            continue
    return None


def _discover_target(resolved: str) -> str:
    """Newest published cycle on/before the horizon (existing boundary)."""
    discovered = _discover_latest_published_settlement_date(date.fromisoformat(resolved))
    if discovered is None:
        raise ValueError("no published settlement cycle")
    return discovered


def get_short_interest_leaderboard(
    limit: int | None = None,
    settlement_date: str | None = None,
    as_of: str | None = None,
    data_root: Path | None = None,
) -> dict[str, object]:
    """Return a bounded leaderboard, computed from live providers for the
    requested cycle (recomputed per invocation, never persisted).

    ``as_of`` defaults to today; pass an explicit as_of for a historical
    screen (only data knowable on/before as_of is used).  With no
    ``settlement_date`` the newest published cycle is discovered and read.
    Fetch failures surface as ``{"error": ...}``, never raise.
    """
    del data_root
    try:
        resolved = _resolve_as_of(as_of)
        target = settlement_date if settlement_date is not None else _discover_target(resolved)
        return _compute_leaderboard(target, resolved, limit)
    except (OSError, ValueError, RuntimeError) as exc:
        return {"error": f"Short-interest leaderboard is unavailable: {exc}"}


# ---------------------------------------------------------------------------
# Research slice: short-interest change + shares-outstanding change
# ---------------------------------------------------------------------------


def _cycle_settlement_dates(as_of: str) -> list[str]:
    """Latest two published settlement cycles on or before ``as_of``."""
    found: list[str] = []
    for candidate in _candidate_settlement_dates(date.fromisoformat(as_of)):
        try:
            if _probe_published_rows(candidate) > 0:
                found.append(candidate)
        except OSError, ValueError, RuntimeError:
            continue
        if len(found) == 2:
            break
    return found


class _CycleItem(TypedDict):
    row: dict[str, object]
    entity_id: str
    fact: dict[str, object]


def _cycle_entities(
    rows: list[dict[str, object]],
    inputs: _ScreenInputs,
) -> dict[str, _CycleItem]:
    """Eligible entities for one settlement cycle: symbol -> row + fact.

    Same point-in-time rules as the leaderboard: only source versions and
    classifications knowable on/before ``as_of`` are used, and only entities
    classified as common equity rank.
    """
    from ..domain.market.securities import SecurityResolution

    result: dict[str, _CycleItem] = {}
    for row in rows:
        symbol = str(row["symbol_code"])
        short_shares = row.get("short_position")
        if short_shares is None:
            continue
        resolution = inputs.resolutions.get(symbol)
        if not isinstance(resolution, SecurityResolution) or not resolution.resolved or not resolution.entity_id:
            continue
        entity_id = resolution.entity_id
        if inputs.security_types.get(entity_id) != _COMMON_EQUITY:
            continue
        fact = _select_fact_for_period(
            inputs.facts_by_entity.get(entity_id) or [], str(row.get("settlement_date") or "")
        )
        if fact is None:
            continue
        result[symbol] = {"row": row, "entity_id": entity_id, "fact": fact}
    return result


def _select_fact_for_period(facts: list[dict[str, object]], settlement_date: str) -> dict[str, object] | None:
    """Latest fact whose period end is on/before the settlement date; facts
    are pre-sorted newest first and already restricted by known_at <= as_of."""
    for fact in facts:
        period_end = _date_str(fact.get("period_end") or "")
        if not period_end or period_end[:10] > settlement_date:
            continue
        value = fact.get("value")
        if value is None or float(str(value)) <= 0:
            continue
        return fact
    return None


def _change_key(e: dict[str, object]) -> tuple[float, str]:
    """Largest percentage-point change first, ticker ascending."""
    change = e["si_pp_change"]
    return (-(float(str(change)) if change is not None else 0.0), str(e["ticker"]))


def short_interest_change_screen(
    as_of: str,
    limit: int | None = None,
    data_root: Path | None = None,
) -> dict[str, object]:
    """Dated research slice: short-interest change + shares-outstanding change
    between the two most recent settlement cycles knowable on/before as_of.

    Every input is filtered by ``known_at <= as_of``; a later filing can
    never alter a slice computed at an earlier ``as_of``.  Missing prior
    cycles or facts are reported as None, never as zero.  Computed per
    invocation from live providers; never persisted.
    """
    del data_root
    limit = _clamp_limit(limit)
    dates = _cycle_settlement_dates(as_of)
    if not dates:
        return {"error": f"No FINRA short interest cycles knowable on or before {as_of}."}
    current_date, prior_date = dates[0], dates[1] if len(dates) > 1 else None
    current_rows = _snapshot_rows(current_date, as_of)
    prior_rows: list[dict[str, object]] = _snapshot_rows(prior_date, as_of) if prior_date else []
    gateway = _gateway()
    symbols = sorted({str(row["symbol_code"]) for row in current_rows + prior_rows})
    inputs = _screen_inputs(symbols, as_of, gateway)
    current = _cycle_entities(current_rows, inputs)
    prior: dict[str, _CycleItem] = _cycle_entities(prior_rows, inputs) if prior_date else {}
    entries: list[dict[str, object]] = []
    for symbol, item in sorted(current.items()):
        row, fact = item["row"], item["fact"]
        short_current = float(str(row["short_position"]))
        si_pct_current = 100 * short_current / float(str(fact["value"]))
        entry: dict[str, object] = {
            "ticker": symbol,
            "issue_name": row.get("issue_name"),
            "settlement_current": current_date,
            "settlement_prior": prior_date,
            "short_shares_current": short_current,
            "short_interest_percent_current": si_pct_current,
            "shares_outstanding_current": float(str(fact["value"])),
            "sec_shares_as_of_current": _date_str(fact["period_end"]),
            "sec_filed_at_current": _date_str(fact["filed_at"]),
            "sec_accession_current": fact.get("accession"),
            "sec_source_url_current": fact.get("source_url"),
            "short_shares_prior": None,
            "short_interest_percent_prior": None,
            "shares_outstanding_prior": None,
            "sec_shares_as_of_prior": None,
            "sec_filed_at_prior": None,
            "sec_accession_prior": None,
            "sec_source_url_prior": None,
            "short_change_abs": None,
            "short_change_pct": None,
            "shares_change_abs": None,
            "shares_change_pct": None,
            "si_pp_change": None,
            "finra_source_url": row.get("source_url"),
        }
        prior_item = prior.get(symbol)
        if prior_item is not None:
            prior_row, prior_fact = prior_item["row"], prior_item["fact"]
            short_prior = float(str(prior_row["short_position"]))
            si_pct_prior = 100 * short_prior / float(str(prior_fact["value"]))
            entry.update(
                {
                    "short_shares_prior": short_prior,
                    "short_interest_percent_prior": si_pct_prior,
                    "shares_outstanding_prior": float(str(prior_fact["value"])),
                    "sec_shares_as_of_prior": _date_str(prior_fact["period_end"]),
                    "sec_filed_at_prior": _date_str(prior_fact["filed_at"]),
                    "sec_accession_prior": prior_fact.get("accession"),
                    "sec_source_url_prior": prior_fact.get("source_url"),
                    "short_change_abs": short_current - short_prior,
                    "short_change_pct": 100 * (short_current - short_prior) / short_prior if short_prior else None,
                    "shares_change_abs": float(str(fact["value"])) - float(str(prior_fact["value"])),
                    "shares_change_pct": 100
                    * (float(str(fact["value"])) - float(str(prior_fact["value"])))
                    / float(str(prior_fact["value"])),
                    "si_pp_change": si_pct_current - si_pct_prior,
                }
            )
        entries.append(entry)
    entries.sort(key=_change_key)
    for index, entry in enumerate(entries, 1):
        entry["rank"] = index
    return {
        "source": "FINRA consolidated short interest + SEC EDGAR company facts (live providers)",
        "metric": "cycle-over-cycle short-interest change and shares-outstanding change; short interest is a settlement-date position, not Reg SHO volume",
        "as_of": as_of,
        "settlement_current": current_date,
        "settlement_prior": prior_date,
        "calculation_version": SLICE_CALC_VERSION,
        "coverage": {"current_finra_rows": len(current), "eligible_rows": len(entries)},
        "unresolved": sorted(_unresolved_symbols(inputs)),
        "entries": entries[:limit],
    }
