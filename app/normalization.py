"""Deterministic SEC/FINRA normalizers for the research data store.

Network-free and agent-free: raw payloads in, normalized rows out.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

from .domain.market import ids

COMPANY_TICKERS_PARSER_VERSION = "sec-company-tickers-v1"
COMPANY_FACTS_PARSER_VERSION = "sec-companyfacts-v6"
FILING_TEXT_PARSER_VERSION = "sec-filing-text-v1"

SHARES_OUTSTANDING_CONCEPT = "EntityCommonStockSharesOutstanding"
_ORIGINAL_CONCEPT = "dei:EntityCommonStockSharesOutstanding"
EPS_UNIT = "USD/shares"
EPS_CONCEPT_NAMES: tuple[str, ...] = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
DIVIDEND_PER_SHARE_CONCEPT = "CommonStockDividendsPerShareDeclared"
DIVIDEND_EVENT_AMOUNT_CONCEPT = "DividendsPayableAmountPerShare"
DIVIDEND_EVENT_DECLARED_CONCEPT = "DividendsPayableDateDeclaredDayMonthAndYear"
DIVIDEND_EVENT_RECORD_CONCEPT = "DividendsPayableDateOfRecordDayMonthAndYear"
DIVIDEND_EVENT_PAYMENT_CONCEPT = "DividendPayableDateToBePaidDayMonthAndYear"
_DIVIDEND_EVENT_DATE_ROLES: dict[str, str] = {
    DIVIDEND_EVENT_DECLARED_CONCEPT: "declaration_date",
    DIVIDEND_EVENT_RECORD_CONCEPT: "record_date",
    DIVIDEND_EVENT_PAYMENT_CONCEPT: "payment_date",
}

CANONICAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "Revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"),
    "NetIncomeLoss": ("NetIncomeLoss",),
    "CashAndCashEquivalents": (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
    ),
    "LongTermDebt": ("LongTermDebtCurrentAndNoncurrent", "LongTermDebtNoncurrent", "LongTermDebt"),
    "OperatingCashFlow": ("NetCashProvidedByUsedInOperatingActivities",),
    "CapEx": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "DividendsPaid": (
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ),
}


def _is_non_blank(value: object) -> bool:
    """True for non-blank strings (row values that must carry an ID/key)."""
    return isinstance(value, str) and bool(value.strip())


def _is_parseable_timestamp(value: object) -> bool:
    """True when a value parses as an ISO-8601 instant (``Z`` accepted)."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def _validate(rows: list[dict[str, object]], model: type[pa.DataFrameModel]) -> None:
    """Validate one normalized dataset's rows; empty is valid, else raises."""
    if rows:
        model.validate(pl.from_dicts(rows, strict=False))


class _LenientConfig(pa.DataFrameModel.Config):
    strict = False
    coerce = True


class _LenientFrame(pa.DataFrameModel):
    """Shared non-strict, coercing config; pandera names each subclass schema after the class."""

    Config = _LenientConfig


class FinancialFactsFrame(_LenientFrame):
    """Boundary shape for ``financial_facts`` rows."""

    fact_id: Series[str] = pa.Field(nullable=False)
    known_at: Series[str] = pa.Field(nullable=False)
    retrieved_at: Series[str] = pa.Field(nullable=False)
    content_hash: Series[str] = pa.Field(nullable=False)
    source_url: Series[str] = pa.Field(nullable=False)
    source_record_id: Series[str] = pa.Field(nullable=False)

    @pa.check(
        "fact_id", "known_at", "retrieved_at", "content_hash", "source_url", "source_record_id", element_wise=True
    )
    def _non_blank(cls, value: object) -> bool:
        return _is_non_blank(value)

    @pa.check("known_at", "retrieved_at", element_wise=True)
    def _timestamp(cls, value: object) -> bool:
        return _is_parseable_timestamp(value)


class ShortInterestFrame(_LenientFrame):
    """Boundary shape for ``short_interest`` rows."""

    row_id: Series[str] = pa.Field(nullable=False)
    known_at: Series[str] = pa.Field(nullable=False)
    retrieved_at: Series[str] = pa.Field(nullable=False)
    content_hash: Series[str] = pa.Field(nullable=False)
    source_url: Series[str] = pa.Field(nullable=False)
    source_record_id: Series[str] = pa.Field(nullable=False)

    @pa.check("row_id", "known_at", "retrieved_at", "content_hash", "source_url", "source_record_id", element_wise=True)
    def _non_blank(cls, value: object) -> bool:
        return _is_non_blank(value)

    @pa.check("known_at", "retrieved_at", element_wise=True)
    def _timestamp(cls, value: object) -> bool:
        return _is_parseable_timestamp(value)


class DividendEventsFrame(_LenientFrame):
    """Boundary shape for ``dividend_events`` rows."""

    dividend_event_id: Series[str] = pa.Field(nullable=False)
    known_at: Series[str] = pa.Field(nullable=False)
    content_hash: Series[str] = pa.Field(nullable=False)
    source_url: Series[str] = pa.Field(nullable=False)

    @pa.check("dividend_event_id", "known_at", "content_hash", "source_url", element_wise=True)
    def _non_blank(cls, value: object) -> bool:
        return _is_non_blank(value)

    @pa.check("known_at", element_wise=True)
    def _timestamp(cls, value: object) -> bool:
        return _is_parseable_timestamp(value)


class EntityFrame(_LenientFrame):
    """Boundary shape for ``entities`` rows."""

    entity_id: Series[str] = pa.Field(nullable=False)
    known_at: Series[str] = pa.Field(nullable=False)
    retrieved_at: Series[str] = pa.Field(nullable=False)
    content_hash: Series[str] = pa.Field(nullable=False)

    @pa.check("entity_id", "known_at", "retrieved_at", "content_hash", element_wise=True)
    def _non_blank(cls, value: object) -> bool:
        return _is_non_blank(value)

    @pa.check("known_at", "retrieved_at", element_wise=True)
    def _timestamp(cls, value: object) -> bool:
        return _is_parseable_timestamp(value)


class AliasFrame(_LenientFrame):
    """Boundary shape for ``entity_aliases`` rows."""

    alias_type: Series[str] = pa.Field(nullable=False)
    alias_value: Series[str] = pa.Field(nullable=False)
    entity_id: Series[str] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(nullable=False)
    valid_from: Series[str] = pa.Field(nullable=True)
    known_at: Series[str] = pa.Field(nullable=False)
    retrieved_at: Series[str] = pa.Field(nullable=False)
    content_hash: Series[str] = pa.Field(nullable=False)

    @pa.check(
        "alias_type",
        "alias_value",
        "entity_id",
        "source",
        "known_at",
        "retrieved_at",
        "content_hash",
        element_wise=True,
    )
    def _non_blank(cls, value: object) -> bool:
        return _is_non_blank(value)

    @pa.check("known_at", "retrieved_at", element_wise=True)
    def _timestamp(cls, value: object) -> bool:
        return _is_parseable_timestamp(value)


class SecurityFrame(_LenientFrame):
    """Boundary shape for ``securities`` rows."""

    security_id: Series[str] = pa.Field(nullable=False)
    known_at: Series[str] = pa.Field(nullable=False)
    retrieved_at: Series[str] = pa.Field(nullable=False)
    content_hash: Series[str] = pa.Field(nullable=False)

    @pa.check("security_id", "known_at", "retrieved_at", "content_hash", element_wise=True)
    def _non_blank(cls, value: object) -> bool:
        return _is_non_blank(value)

    @pa.check("known_at", "retrieved_at", element_wise=True)
    def _timestamp(cls, value: object) -> bool:
        return _is_parseable_timestamp(value)


def _ticker_values(raw: object) -> list[object]:
    """Ticker rows from a dict- or list-shaped SEC tickers payload."""
    if isinstance(raw, dict):
        return list(raw.values())
    return list(raw) if isinstance(raw, (list, tuple)) else []


def _coerce_number_text(value: object) -> str | int | float | bytes:
    """int/float/str/bytes passthrough narrowed for int()/float()."""
    if isinstance(value, (int, float, str, bytes)):
        return value
    raise TypeError(f"non-numeric CIK/amount: {value!r}")


def _ticker_cik(item: dict[str, object]) -> tuple[str, int | None]:
    """Upper-cased ticker + int CIK from one tickers row (None CIK when bad)."""
    ticker = str(item.get("ticker") or "").strip().upper()
    cik_raw: object = item.get("cik_str")
    if not ticker or cik_raw is None:
        return "", None
    try:
        cik = int(_coerce_number_text(cik_raw))
    except TypeError, ValueError:
        return "", None
    return ticker, cik


def _ticker_rows(
    item: dict[str, object], ticker: str, cik: int, retrieved_at: str, content_hash: str
) -> tuple[dict[str, object], dict[str, object]]:
    """Entity + alias rows for one validated ticker/CIK pair."""
    entity_id = ids.sec_entity_id(cik)
    return {
        "entity_id": entity_id,
        "name": str(item.get("title") or "").strip() or None,
        "entity_type": "unknown",
        "sic": None,
        "source": "sec:company_tickers",
        "known_at": retrieved_at,
        "retrieved_at": retrieved_at,
        "content_hash": content_hash,
        "parser_version": COMPANY_TICKERS_PARSER_VERSION,
    }, {
        "alias_type": "ticker",
        "alias_value": ticker,
        "entity_id": entity_id,
        "security_id": ids.sec_security_id(cik),
        "source": "sec:company_tickers",
        "valid_from": None,
        "valid_to": None,
        "known_at": retrieved_at,
        "retrieved_at": retrieved_at,
        "content_hash": content_hash,
        "parser_version": COMPANY_TICKERS_PARSER_VERSION,
    }


def normalize_sec_tickers(raw: object, *, retrieved_at: str, content_hash: str) -> dict[str, list[dict[str, object]]]:
    entities: list[dict[str, object]] = []
    aliases: list[dict[str, object]] = []
    for item in _ticker_values(raw):
        if not isinstance(item, dict):
            continue
        ticker, cik = _ticker_cik(item)
        if not ticker or cik is None:
            continue
        entity_row, alias_row = _ticker_rows(item, ticker, cik, retrieved_at, content_hash)
        entities.append(entity_row)
        aliases.append(alias_row)
    _validate(entities, EntityFrame)
    _validate(aliases, AliasFrame)
    return {"entities": entities, "entity_aliases": aliases}


def _extract_shares_facts(raw: object) -> list[dict[str, object]]:
    if not isinstance(raw, dict):
        return []
    facts_obj: object = raw.get("facts")
    if not isinstance(facts_obj, dict):
        return []
    dei_obj: object = facts_obj.get("dei")
    if not isinstance(dei_obj, dict):
        return []
    concept_obj: object = dei_obj.get(SHARES_OUTSTANDING_CONCEPT)
    if not isinstance(concept_obj, dict):
        return []
    units_obj: object = concept_obj.get("units")
    if not isinstance(units_obj, dict):
        return []
    shares_obj: object = units_obj.get("shares") or []
    if not isinstance(shares_obj, list):
        return []
    return [fact for fact in shares_obj if isinstance(fact, dict)]


def _facts_namespaces(raw: object) -> dict[object, object]:
    """facts namespaces mapping (non-dict at any level -> {})."""
    if not isinstance(raw, dict):
        return {}
    namespaces: object = raw.get("facts") or {}
    return namespaces if isinstance(namespaces, dict) else {}


def _concepts_of(namespace_value: object) -> dict[object, object]:
    """Concept mapping for one namespace (non-dict -> {})."""
    return namespace_value if isinstance(namespace_value, dict) else {}


def _unit_facts(payload: object, unit: str) -> list[object]:
    """Fact list for one unit of a concept payload (missing/non-list -> [])."""
    if not isinstance(payload, dict):
        return []
    units_obj: object = payload.get("units") or {}
    if not isinstance(units_obj, dict):
        return []
    unit_obj: object = units_obj.get(unit) or []
    return unit_obj if isinstance(unit_obj, list) else []


def _canonical_name(tag: object) -> str | None:
    """Canonical concept name for a tag (None when the tag is unknown)."""
    return next((name for name, aliases in CANONICAL_CONCEPTS.items() if tag in aliases), None)


def _collect_entries(
    entries: list[tuple[str, str, str, dict[str, object]]],
    name: str,
    namespace: object,
    tag: object,
    unit: str,
    facts: list[object],
) -> None:
    """Append (name, original, unit, fact) rows for dict facts only."""
    for fact in facts:
        if isinstance(fact, dict):
            entries.append((name, f"{namespace}:{tag}", unit, fact))


def _extract_canonical_facts(raw: object) -> list[tuple[str, str, str, dict[str, object]]]:
    entries: list[tuple[str, str, str, dict[str, object]]] = []
    for namespace, concepts in _facts_namespaces(raw).items():
        for tag, payload in _concepts_of(concepts).items():
            canonical = _canonical_name(tag)
            if canonical is None:
                continue
            _collect_entries(entries, canonical, namespace, tag, "USD", _unit_facts(payload, "USD"))
    return entries


def _extract_eps_facts(raw: object) -> list[tuple[str, str, str, dict[str, object]]]:
    """Per-share earnings facts, accepted only under the ``USD/shares`` unit."""
    entries: list[tuple[str, str, str, dict[str, object]]] = []
    for namespace, concepts in _facts_namespaces(raw).items():
        for tag, payload in _concepts_of(concepts).items():
            if tag not in EPS_CONCEPT_NAMES:
                continue
            if not isinstance(tag, str):
                continue
            _collect_entries(entries, tag, namespace, tag, EPS_UNIT, _unit_facts(payload, EPS_UNIT))
    return entries


def _extract_dividend_facts(raw: object) -> list[tuple[str, str, str, dict[str, object]]]:
    """Declared dividend-per-share facts, accepted only under ``USD/shares``."""
    entries: list[tuple[str, str, str, dict[str, object]]] = []
    for namespace, concepts in _facts_namespaces(raw).items():
        for tag, payload in _concepts_of(concepts).items():
            if tag != DIVIDEND_PER_SHARE_CONCEPT:
                continue
            if not isinstance(tag, str):
                continue
            _collect_entries(entries, tag, namespace, tag, EPS_UNIT, _unit_facts(payload, EPS_UNIT))
    return entries


def _event_namespaces(raw: object) -> dict[object, object]:
    """facts namespaces for dividend events (non-dict -> {})."""
    namespaces_obj: object = (raw.get("facts") if isinstance(raw, dict) else None) or {}
    return namespaces_obj if isinstance(namespaces_obj, dict) else {}


def _amount_unit_choice(units: dict[object, object]) -> list[str]:
    """Chosen amount unit: preferred USD/shares, else the sole observed unit."""
    unit_facts: list[tuple[str, list[object]]] = [
        (unit, facts) for unit, facts in units.items() if isinstance(unit, str) and isinstance(facts, list) and facts
    ]
    preferred = [u for u, _ in unit_facts if u == EPS_UNIT]
    # Contingency A: fall back to the sole observed unit for this concept.
    return preferred[:1] if preferred else ([unit_facts[0][0]] if len(unit_facts) == 1 else [])


def _event_amount(fact: dict[str, object]) -> float | None:
    """Numeric dividend amount from a fact (non-numeric -> None)."""
    val_raw: object = fact.get("val")
    if val_raw is None:
        return None
    try:
        return float(_coerce_number_text(val_raw))
    except TypeError, ValueError:
        return None


def _event_key(fact: dict[str, object]) -> tuple[str, str] | None:
    """(accession, filed) key for a fact (blank part -> None)."""
    accession, filed = str(fact.get("accn") or ""), str(fact.get("filed") or "")
    return (accession, filed) if accession and filed else None


def _collect_event_amounts(
    amounts: list[tuple[str, str, float, str, str]], units: dict[object, object], namespace: object, tag: object
) -> None:
    """Append validated (accession, filed, amount, unit, concept) rows."""
    units_map: dict[object, object] = units
    for unit in _amount_unit_choice(units_map):
        facts_obj: object = units_map.get(unit) or []
        if not isinstance(facts_obj, list):
            continue
        for fact in facts_obj:
            if not isinstance(fact, dict):
                continue
            amount = _event_amount(fact)
            if amount is None:
                continue
            key = _event_key(fact)
            if key is None:
                continue
            amounts.append((key[0], key[1], amount, unit, f"{namespace}:{tag}"))


def _collect_event_dates(
    dates: dict[tuple[str, str], dict[str, set[str]]], units: dict[object, object], role: str
) -> None:
    """Merge date-role values keyed by (accession, filed)."""
    for facts in units.values():
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            value = str(fact.get("val") or "").strip()
            key = _event_key(fact)
            if not value or key is None:
                continue
            dates.setdefault(key, {}).setdefault(role, set()).add(value)


def _scan_event_concept(
    amounts: list[tuple[str, str, float, str, str]],
    dates: dict[tuple[str, str], dict[str, set[str]]],
    namespace: object,
    tag: object,
    payload: object,
) -> None:
    """Scan one concept as an amount or date-role contributor (others skipped)."""
    is_amount = tag == DIVIDEND_EVENT_AMOUNT_CONCEPT
    role = _DIVIDEND_EVENT_DATE_ROLES.get(tag) if isinstance(tag, str) else None
    if not is_amount and role is None:
        return
    if not isinstance(payload, dict):
        return
    units_obj: object = payload.get("units") or {}
    if not isinstance(units_obj, dict):
        return
    if is_amount:
        _collect_event_amounts(amounts, units_obj, namespace, tag)
    else:
        assert role is not None
        _collect_event_dates(dates, units_obj, role)


def _group_event_amounts(
    amounts: list[tuple[str, str, float, str, str]],
) -> dict[tuple[str, str], list[tuple[float, str, str]]]:
    """Amounts grouped by (accession, filed)."""
    groups: dict[tuple[str, str], list[tuple[float, str, str]]] = {}
    for accession, filed, amount, unit, source_concept in amounts:
        groups.setdefault((accession, filed), []).append((amount, unit, source_concept))
    return groups


def _deduped_amounts(entries: list[tuple[float, str, str]]) -> dict[float, tuple[str, str]]:
    """First (unit, concept) per distinct amount."""
    seen: dict[float, tuple[str, str]] = {}
    for amount, unit, source_concept in entries:
        seen.setdefault(amount, (unit, source_concept))
    return seen


def _role_is_single(group_dates: dict[str, set[str]], role: str) -> bool:
    """True when a date role has at most one non-blank value."""
    return len([d for d in sorted(group_dates.get(role) or ()) if d]) <= 1


def _group_is_paired(seen: dict[float, tuple[str, str]], group_dates: dict[str, set[str]]) -> bool:
    """True when one amount pairs with at most one date per role."""
    payments = sorted(group_dates.get("payment_date") or ())
    return (
        len(seen) == 1
        and _role_is_single(group_dates, "declaration_date")
        and _role_is_single(group_dates, "record_date")
        and len([p for p in payments if p]) <= 1
    )


def _paired_dates(group_dates: dict[str, set[str]]) -> tuple[str | None, str | None, list[str | None]]:
    """(declaration, record, [payment]) for a paired group."""
    declaration = min(group_dates.get("declaration_date") or (), default=None)
    record = min(group_dates.get("record_date") or (), default=None)
    payments = sorted(group_dates.get("payment_date") or ())
    payments_nonnull: list[str | None] = [p for p in payments if p]
    return declaration, record, payments_nonnull[:1] if payments_nonnull else [None]


def _currency_of(unit: str) -> str:
    """Currency prefix of a per-share unit (USD/shares -> USD)."""
    return unit.split("/")[0] if "/" in unit else unit


def _paired_event(
    cik: int,
    entity_id: str,
    security_id: str,
    amount: float,
    unit: str,
    source_concept: str,
    declaration: str | None,
    record: str | None,
    payment: str | None,
    accession: str,
    filed: str,
    source_url: str,
    content_hash: str,
) -> dict[str, object]:
    """One date-paired dividend event."""
    return {
        "dividend_event_id": ids.sec_dividend_event_id(cik, amount, record, payment, "unknown", accession, declaration),
        "entity_id": entity_id,
        "security_id": security_id,
        "ticker": None,
        "amount_per_share": amount,
        "currency": _currency_of(unit),
        "dividend_type": "unknown",
        "declaration_date": declaration,
        "record_date": record,
        "payment_date": payment,
        "ex_dividend_date": None,
        "ex_dividend_date_source": "unknown",
        "status": "unknown",
        "source_form": None,
        "accession": accession,
        "filed_at": filed,
        "known_at": filed,
        "source_url": source_url,
        "source_concept": source_concept,
        "source_type": "structured_xbrl",
        "evidence_excerpt": None,
        "content_hash": content_hash,
        "parser_version": COMPANY_FACTS_PARSER_VERSION,
    }


def _unpaired_event(
    cik: int,
    entity_id: str,
    security_id: str,
    amount: float,
    unit: str,
    source_concept: str,
    accession: str,
    filed: str,
    source_url: str,
    content_hash: str,
) -> dict[str, object]:
    """One undated dividend event for an ambiguous group."""
    return {
        "dividend_event_id": ids.sec_dividend_event_id(cik, amount, None, None, "unknown", accession, None),
        "entity_id": entity_id,
        "security_id": security_id,
        "ticker": None,
        "amount_per_share": amount,
        "currency": _currency_of(unit),
        "dividend_type": "unknown",
        "declaration_date": None,
        "record_date": None,
        "payment_date": None,
        "ex_dividend_date": None,
        "ex_dividend_date_source": "unknown",
        "status": "unknown",
        "source_form": None,
        "accession": accession,
        "filed_at": filed,
        "known_at": filed,
        "source_url": source_url,
        "source_concept": source_concept,
        "source_type": "structured_xbrl",
        "evidence_excerpt": None,
        "content_hash": content_hash,
        "parser_version": COMPANY_FACTS_PARSER_VERSION,
    }


def _emit_group_events(
    events: list[dict[str, object]],
    cik: int,
    entity_id: str,
    security_id: str,
    seen: dict[float, tuple[str, str]],
    group_dates: dict[str, set[str]],
    accession: str,
    filed: str,
    source_url: str,
    content_hash: str,
) -> None:
    """Append paired events when dates pin down, else undated events."""
    if _group_is_paired(seen, group_dates):
        declaration, record, paired = _paired_dates(group_dates)
        for amount in sorted(seen):
            unit, source_concept = seen[amount]
            for payment in paired:
                events.append(
                    _paired_event(
                        cik,
                        entity_id,
                        security_id,
                        amount,
                        unit,
                        source_concept,
                        declaration,
                        record,
                        payment,
                        accession,
                        filed,
                        source_url,
                        content_hash,
                    )
                )
        return
    for amount in sorted(seen):
        unit, source_concept = seen[amount]
        events.append(
            _unpaired_event(
                cik, entity_id, security_id, amount, unit, source_concept, accession, filed, source_url, content_hash
            )
        )


def _extract_dividend_event_facts(
    raw: object,
    *,
    cik: int,
    entity_id: str,
    security_id: str,
    source_url: str,
    retrieved_at: str,
    content_hash: str,
) -> list[dict[str, object]]:
    """Declared-dividend events from structured XBRL facts; never infers dates."""
    del retrieved_at  # known_at is filed_at, never extraction time.
    amounts: list[tuple[str, str, float, str, str]] = []
    dates: dict[tuple[str, str], dict[str, set[str]]] = {}
    for namespace, concepts in _event_namespaces(raw).items():
        if not isinstance(concepts, dict):
            continue
        for tag, payload in concepts.items():
            _scan_event_concept(amounts, dates, namespace, tag, payload)
    groups = _group_event_amounts(amounts)
    events: list[dict[str, object]] = []
    for accession, filed in sorted(groups):
        seen = _deduped_amounts(groups[(accession, filed)])
        _emit_group_events(
            events,
            cik,
            entity_id,
            security_id,
            seen,
            dates.get((accession, filed), {}),
            accession,
            filed,
            source_url,
            content_hash,
        )
    return events


_DIVIDEND_TEXT_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}
_DIVIDEND_DECLARE_RE = re.compile(
    r"declared\s+an?\s+"
    r"(?:(quarterly|monthly|semiannual|annual|special|supplemental|extraordinary)\s+)?"
    r"(?:cash\s+)?dividend\s+of\s+\$(?P<amount>\d[\d,]*(?:\.\d+)?)\s+per\s+share",
    re.IGNORECASE,
)
_DIVIDEND_DATE_RES = {
    "payment_date": re.compile(r"payable\s+(?:on\s+)?(?P<date>[A-Za-z]+\.?\s+\d{1,2},\s*\d{4})", re.IGNORECASE),
    "record_date": re.compile(
        r"(?:shareholders|stockholders)\s+of\s+record\s+(?:on\s+|as\s+of\s+)?"
        r"(?P<date>[A-Za-z]+\.?\s+\d{1,2},\s*\d{4})",
        re.IGNORECASE,
    ),
    "ex_dividend_date": re.compile(
        r"ex[-\s]?dividend\s+(?:date\s+)?(?:of\s+|on\s+|is\s+)?"
        r"(?P<date>[A-Za-z]+\.?\s+\d{1,2},\s*\d{4})",
        re.IGNORECASE,
    ),
}


def _parse_dividend_text_date(value: str) -> str | None:
    match = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", value.strip())
    if not match:
        return None
    month = _DIVIDEND_TEXT_MONTHS.get(match.group(1)[:3].lower())
    day = int(match.group(2))
    if month is None or not 1 <= day <= 31:
        return None
    return f"{match.group(3)}-{month}-{day:02d}"


def _text_event_amount(match: re.Match[str]) -> float | None:
    """Dollar amount from a declare-match (unparseable -> None)."""
    try:
        return float(match.group("amount").replace(",", ""))
    except TypeError, ValueError:
        return None


def _text_dividend_type(sentence: str) -> str:
    """Dividend type from prose keywords (supplemental/special else regular)."""
    lowered = sentence.lower()
    if "supplemental" in lowered:
        return "supplemental"
    return "special" if "special" in lowered or "extraordinary" in lowered else "regular"


def _text_event_dates(sentence: str) -> dict[str, str | None]:
    """Parsed record/payment/ex-dividend dates from one sentence."""
    return {
        key: _parse_dividend_text_date(m.group("date"))
        for key, m in ((key, rx.search(sentence)) for key, rx in _DIVIDEND_DATE_RES.items())
        if m
    }


def _text_event(
    cik: int,
    entity_id: str,
    security_id: str,
    amount: float,
    dividend_type: str,
    found: dict[str, str | None],
    sentence: str,
    accession: str,
    filed_at: str,
    source_url: str,
    content_hash: str,
) -> dict[str, object]:
    """One proposed dividend event from parsed sentence parts."""
    record = found.get("record_date")
    payment = found.get("payment_date")
    ex_date = found.get("ex_dividend_date")
    return {
        "dividend_event_id": ids.sec_dividend_event_id(cik, amount, record, payment, dividend_type, accession, None),
        "entity_id": entity_id,
        "security_id": security_id,
        "ticker": None,
        "amount_per_share": amount,
        "currency": "USD",
        "dividend_type": dividend_type,
        "declaration_date": None,
        "record_date": record,
        "payment_date": payment,
        "ex_dividend_date": ex_date,
        "ex_dividend_date_source": "explicit" if ex_date else "unknown",
        "status": "unknown",
        "source_form": None,
        "accession": accession,
        "filed_at": filed_at,
        "known_at": filed_at,
        "source_url": source_url,
        "source_concept": None,
        "source_type": "filing_text",
        "evidence_excerpt": sentence,
        "content_hash": content_hash,
        "parser_version": FILING_TEXT_PARSER_VERSION,
    }


def _text_event_for_sentence(
    cik: int,
    entity_id: str,
    security_id: str,
    sentence: str,
    accession: str,
    filed_at: str,
    source_url: str,
    content_hash: str,
) -> dict[str, object] | None:
    """Proposed event for one sentence (None when no declare-amount)."""
    match = _DIVIDEND_DECLARE_RE.search(sentence)
    if not match:
        return None
    amount = _text_event_amount(match)
    if amount is None:
        return None
    return _text_event(
        cik,
        entity_id,
        security_id,
        amount,
        _text_dividend_type(sentence),
        _text_event_dates(sentence),
        sentence,
        accession,
        filed_at,
        source_url,
        content_hash,
    )


def _extract_dividend_events_from_text(
    text: str,
    *,
    cik: int,
    entity_id: str,
    security_id: str,
    accession: str,
    filed_at: str,
    source_url: str,
    content_hash: str,
) -> list[dict[str, object]]:
    """Proposed dividend events from filing prose; amount mandatory, dates optional."""
    events: list[dict[str, object]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        sentence = sentence.strip()
        event = _text_event_for_sentence(
            cik, entity_id, security_id, sentence, accession, filed_at, source_url, content_hash
        )
        if event is not None:
            events.append(event)
    return events


def _extract_facts(raw: object) -> list[tuple[str, str, str, dict[str, object]]]:
    entries = [(SHARES_OUTSTANDING_CONCEPT, _ORIGINAL_CONCEPT, "shares", fact) for fact in _extract_shares_facts(raw)]
    entries.extend(_extract_canonical_facts(raw))
    entries.extend(_extract_eps_facts(raw))
    entries.extend(_extract_dividend_facts(raw))
    return entries


def _parse_company_cik(raw: object) -> int:
    """CIK int from a companyfacts envelope (unparseable -> 0)."""
    cik_raw: object = raw.get("cik") if isinstance(raw, dict) else None
    try:
        return (
            int(cik_raw or 0)
            if isinstance(cik_raw, (int, float, str, bytes))
            else (int(str(cik_raw) or 0) if cik_raw is not None else 0)
        )
    except TypeError, ValueError:
        return 0


def _envelope_known(extracted_facts: list[tuple[str, str, str, dict[str, object]]], retrieved_at: str) -> str:
    """Latest filed date across facts (no dates -> retrieval time)."""
    filed_dates = sorted(
        str(fact.get("filed") or "") for _, _, _, fact in extracted_facts if str(fact.get("filed") or "")
    )
    return filed_dates[-1] if filed_dates else retrieved_at


def _envelope_document(
    source_record_id: str, source_url: str, retrieved_at: str, envelope_known: str, content_hash: str
) -> dict[str, object]:
    """Companyfacts envelope document row."""
    return {
        "doc_id": ids.sec_doc_id("companyfacts", source_record_id, content_hash),
        "source": "sec",
        "kind": "companyfacts",
        "key": source_record_id,
        "source_url": source_url,
        "accession": None,
        "sha256": content_hash,
        "retrieved_at": retrieved_at,
        "published_at": None,
        "known_at": envelope_known,
        "content_hash": content_hash,
        "parser_version": COMPANY_FACTS_PARSER_VERSION,
    }


def _fact_float(fact: dict[str, object]) -> float | None:
    """Numeric val for a fact (missing/unparseable -> None)."""
    val_raw: object = fact.get("val")
    try:
        if val_raw is None:
            return None
        if isinstance(val_raw, (int, float, str, bytes)):
            return float(_coerce_number_text(val_raw))
        return float(str(val_raw))
    except TypeError, ValueError:
        return None


def _fact_key(fact: dict[str, object]) -> tuple[str, str, str] | None:
    """(period_end, filed_at, accession) for a fact (blank part -> None)."""
    period_end = str(fact.get("end") or "")
    filed_at = str(fact.get("filed") or "")
    accession = str(fact.get("accn") or "")
    return (period_end, filed_at, accession) if period_end and filed_at and accession else None


def _fact_fiscal_year(fact: dict[str, object]) -> int | None:
    """Fiscal year int for a fact (missing/unparseable -> None)."""
    fy_raw: object = fact.get("fy")
    try:
        if fy_raw is None:
            return None
        if isinstance(fy_raw, (int, float, str, bytes)):
            return int(_coerce_number_text(fy_raw))
        return int(str(fy_raw))
    except TypeError, ValueError:
        return None


def _fact_row(
    cik: int,
    entity_id: str,
    security_id: str,
    concept: str,
    original_concept: str,
    unit: str,
    fact: dict[str, object],
    value: float,
    period_end: str,
    filed_at: str,
    accession: str,
    retrieved_at: str,
    source_url: str,
    source_record_id: str,
    content_hash: str,
) -> dict[str, object]:
    """One normalized financial-fact row."""
    start_raw = fact.get("start")
    return {
        "fact_id": ids.sec_fact_id(cik, accession, concept, period_end, value),
        "entity_id": entity_id,
        "security_id": security_id,
        "concept": concept,
        "original_concept": original_concept,
        "value": value,
        "unit": unit,
        "duration_type": "duration" if fact.get("start") else "instant",
        "period_end": period_end,
        "period_start": str(start_raw) if start_raw else None,
        "fiscal_year": _fact_fiscal_year(fact),
        "fiscal_period": str(fact.get("fp") or "") or None,
        "filed_at": filed_at,
        "accession": accession,
        "frame": fact.get("frame"),
        "known_at": filed_at,
        "retrieved_at": retrieved_at,
        "source_url": source_url,
        "source_record_id": source_record_id,
        "content_hash": content_hash,
        "parser_version": COMPANY_FACTS_PARSER_VERSION,
    }


def _fact_row_or_none(
    cik: int,
    entity_id: str,
    security_id: str,
    concept: str,
    original_concept: str,
    unit: str,
    fact: dict[str, object],
    retrieved_at: str,
    source_url: str,
    source_record_id: str,
    content_hash: str,
) -> dict[str, object] | None:
    """Normalized fact row (None when value or key parts are missing)."""
    value = _fact_float(fact)
    if value is None:
        return None
    key = _fact_key(fact)
    if key is None:
        return None
    period_end, filed_at, accession = key
    return _fact_row(
        cik,
        entity_id,
        security_id,
        concept,
        original_concept,
        unit,
        fact,
        value,
        period_end,
        filed_at,
        accession,
        retrieved_at,
        source_url,
        source_record_id,
        content_hash,
    )


def _financial_fact_rows(
    extracted_facts: list[tuple[str, str, str, dict[str, object]]],
    cik: int,
    entity_id: str,
    security_id: str,
    retrieved_at: str,
    source_url: str,
    source_record_id: str,
    content_hash: str,
) -> list[dict[str, object]]:
    """Normalized financial-fact rows (unparseable facts skipped)."""
    financial_facts: list[dict[str, object]] = []
    for concept, original_concept, unit, fact in extracted_facts:
        row = _fact_row_or_none(
            cik,
            entity_id,
            security_id,
            concept,
            original_concept,
            unit,
            fact,
            retrieved_at,
            source_url,
            source_record_id,
            content_hash,
        )
        if row is not None:
            financial_facts.append(row)
    return financial_facts


def _company_security(
    security_id: str,
    entity_id: str,
    financial_facts: list[dict[str, object]],
    envelope_known: str,
    retrieved_at: str,
    content_hash: str,
) -> dict[str, object]:
    """Company security row (equity-common only when facts exist)."""
    return {
        "security_id": security_id,
        "entity_id": entity_id,
        "security_type": "equity-common" if financial_facts else "unknown",
        "ticker": None,
        "exchange": None,
        "source": "sec:companyfacts",
        "known_at": envelope_known,
        "retrieved_at": retrieved_at,
        "content_hash": content_hash,
        "parser_version": COMPANY_FACTS_PARSER_VERSION,
    }


def normalize_sec_company_facts(
    raw: object,
    *,
    retrieved_at: str,
    content_hash: str,
    source_url: str,
    source_record_id: str,
) -> dict[str, list[dict[str, object]]]:
    cik = _parse_company_cik(raw)
    entity_id = ids.sec_entity_id(cik)
    security_id = ids.sec_security_id(cik)
    extracted_facts = _extract_facts(raw)
    known = _envelope_known(extracted_facts, retrieved_at)
    documents = [_envelope_document(source_record_id, source_url, retrieved_at, known, content_hash)]
    financial_facts = _financial_fact_rows(
        extracted_facts, cik, entity_id, security_id, retrieved_at, source_url, source_record_id, content_hash
    )
    dividend_events = _extract_dividend_event_facts(
        raw,
        cik=cik,
        entity_id=entity_id,
        security_id=security_id,
        source_url=source_url,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
    )
    securities = [_company_security(security_id, entity_id, financial_facts, known, retrieved_at, content_hash)]
    _validate(financial_facts, FinancialFactsFrame)
    _validate(dividend_events, DividendEventsFrame)
    _validate(securities, SecurityFrame)
    return {
        "documents": documents,
        "financial_facts": financial_facts,
        "securities": securities,
        "dividend_events": dividend_events,
    }


SHORT_INTEREST_PARSER_VERSION = "finra-short-interest-v2"


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes)):
        try:
            text = value.decode() if isinstance(value, bytes) else value
            if text.strip() == "":
                return None
            return float(text)
        except TypeError, ValueError:
            return None
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except TypeError, ValueError:
        return None


def _parse_iso_instant(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} {value!r} is not a parseable ISO-8601 timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _check_finra_known_at(settlement_date: str, retrieved_at: str, known_at: str) -> None:
    """Validate explicit known_at against settlement/retrieved instants."""
    try:
        settlement_day = date.fromisoformat(settlement_date)
    except ValueError:
        raise ValueError(f"settlement_date {settlement_date!r} is not a parseable date")
    known_instant = _parse_iso_instant(known_at, "known_at")
    retrieved_instant = _parse_iso_instant(retrieved_at, "retrieved_at")
    if known_instant.date() < settlement_day:
        raise ValueError(f"known_at {known_at} precedes settlement_date {settlement_date}")
    if known_instant > retrieved_instant:
        raise ValueError(f"known_at {known_at} exceeds retrieved_at {retrieved_at}")


def _finra_symbol(row: dict[str, object]) -> str | None:
    """Upper-cased symbol code (None when blank)."""
    symbol = str(row.get("symbolCode") or "").strip().upper()
    return symbol or None


def _finra_short_position(row: dict[str, object]) -> float | None:
    """Non-negative short position (negative source values -> None)."""
    short_position = _to_float(row.get("currentShortPositionQuantity"))
    return None if short_position is not None and short_position < 0 else short_position


def _finra_row(
    settlement_date: str,
    retrieved_at: str,
    known_at: str | None,
    content_hash: str,
    source_url: str,
    source_record_id: str,
    row: dict[str, object],
    symbol: str,
) -> dict[str, object]:
    """One normalized short-interest row."""
    # The row ID includes the snapshot content hash so a corrected source
    # payload becomes a NEW source version (new retrieved_at) instead of
    # colliding with the original row in the dedupe.
    return {
        "row_id": f"finra:row:{settlement_date}:{symbol}:{content_hash[:12]}",
        "entity_id": ids.finra_entity_id(symbol),
        "security_id": None,
        "symbol_code": symbol,
        "issue_name": str(row.get("issueName") or "").strip() or None,
        "settlement_date": settlement_date,
        "short_position": _finra_short_position(row),
        "prev_position": _to_float(row.get("previousShortPositionQuantity")),
        "avg_daily_volume": _to_float(row.get("averageDailyVolumeQuantity")),
        "days_to_cover": _to_float(row.get("daysToCoverQuantity")),
        "source_url": source_url,
        "source_record_id": source_record_id,
        # Explicit publication date wins when provided; otherwise retrieved_at
        # is the conservative known_at (no FINRA calendar lookup in-repo).
        "known_at": known_at if known_at else retrieved_at,
        "retrieved_at": retrieved_at,
        "content_hash": content_hash,
        "parser_version": SHORT_INTEREST_PARSER_VERSION,
    }


def normalize_finra_short_interest(
    rows: list[dict[str, object]],
    *,
    settlement_date: str,
    retrieved_at: str,
    known_at: str | None = None,
    content_hash: str,
    source_url: str,
    source_record_id: str,
) -> dict[str, list[dict[str, object]]]:
    if known_at:
        _check_finra_known_at(settlement_date, retrieved_at, known_at)
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return {"short_interest": []}
    # Object holder: source cells mix ints, floats, strings, and bytes, which
    # have no common supertype for pl.from_dicts. Row-wise helpers stay the
    # single home of per-cell semantics; Polars owns the projection only.
    frame = pl.DataFrame({"_row": pl.Series("_row", dict_rows, dtype=pl.Object)})
    short_interest: list[dict[str, object]] = (
        frame.with_columns(
            pl.col("_row").map_elements(_finra_symbol, return_dtype=pl.String).alias("symbol_code"),
            pl.col("_row")
            .map_elements(lambda row: str(row.get("issueName") or "").strip() or None, return_dtype=pl.String)
            .alias("issue_name"),
            pl.col("_row").map_elements(_finra_short_position, return_dtype=pl.Float64).alias("short_position"),
            pl.col("_row")
            .map_elements(lambda row: _to_float(row.get("previousShortPositionQuantity")), return_dtype=pl.Float64)
            .alias("prev_position"),
            pl.col("_row")
            .map_elements(lambda row: _to_float(row.get("averageDailyVolumeQuantity")), return_dtype=pl.Float64)
            .alias("avg_daily_volume"),
            pl.col("_row")
            .map_elements(lambda row: _to_float(row.get("daysToCoverQuantity")), return_dtype=pl.Float64)
            .alias("days_to_cover"),
        )
        .filter(pl.col("symbol_code").is_not_null())
        .with_columns(
            pl.concat_str(
                [pl.lit(f"finra:row:{settlement_date}:"), pl.col("symbol_code"), pl.lit(f":{content_hash[:12]}")]
            ).alias("row_id"),
            pl.concat_str([pl.lit("finra:symbol:"), pl.col("symbol_code")]).alias("entity_id"),
            pl.lit(None, dtype=pl.String).alias("security_id"),
            pl.lit(settlement_date).alias("settlement_date"),
            pl.lit(source_url).alias("source_url"),
            pl.lit(source_record_id).alias("source_record_id"),
            # Explicit publication date wins when provided; otherwise retrieved_at
            # is the conservative known_at (no FINRA calendar lookup in-repo).
            pl.lit(known_at if known_at else retrieved_at).alias("known_at"),
            pl.lit(retrieved_at).alias("retrieved_at"),
            pl.lit(content_hash).alias("content_hash"),
            pl.lit(SHORT_INTEREST_PARSER_VERSION).alias("parser_version"),
        )
        .select(
            "row_id",
            "entity_id",
            "security_id",
            "symbol_code",
            "issue_name",
            "settlement_date",
            "short_position",
            "prev_position",
            "avg_daily_volume",
            "days_to_cover",
            "source_url",
            "source_record_id",
            "known_at",
            "retrieved_at",
            "content_hash",
            "parser_version",
        )
        .to_dicts()
    )
    _validate(short_interest, ShortInterestFrame)
    return {"short_interest": short_interest}
