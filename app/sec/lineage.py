"""XBRL fact lineage: fact -> accession -> filing -> period -> concept."""

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

_KEYS = (
    "concept",
    "value",
    "period_start",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "filed_at",
    "accession",
    "source_url",
    "known_at",
)

_AS_OF_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _iso_cell(value: object) -> object:
    """Live row date/datetime to ISO text so lineage groups and projects."""
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def fact_lineage(row: Mapping[str, object]) -> dict[str, object]:
    """Pure projection; missing keys -> None.

    Rows arrive as plain dicts from the live normalized companyfacts
    payload, so the boundary takes a Mapping and projects only the known
    lineage keys. Timestamp columns arrive as ISO text and pass through;
    datetime values are ISO-coerced so grouping and display stay text-based.
    """
    try:
        items: dict[str, object] = dict(row) if isinstance(row, dict) else {}
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        items = {}
    return {key: _iso_cell(items.get(key)) for key in _KEYS}


def period_lineage(
    rows: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Group by period_end; earliest filed_at is originally_reported.

    Rows with a missing or non-text period_end cannot be grouped, so they
    are skipped like the missing-key rows the original code already dropped.
    """
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows or []:
        try:
            end = _iso_cell(row.get("period_end"))
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
        if not isinstance(end, str) or not end:
            continue
        groups.setdefault(end, []).append(row)
    out: list[dict[str, object]] = []
    for end in sorted(groups):
        group = sorted(groups[end], key=_filed_known)
        original = fact_lineage(group[0])
        latest = fact_lineage(group[-1])
        out.append(
            {"period_end": end, "originally_reported": original, "latest": latest, "restated": latest != original}
        )
    return out


def _filed_known(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("filed_at") or ""), str(row.get("known_at") or ""))


def _cik_from_entity(entity_id: str) -> int | None:
    """Trailing-digit CIK from a sec:cik entity id or a bare CIK string."""
    try:
        digits = "".join(ch for ch in str(entity_id or "") if ch.isdigit())
        return int(digits) if digits else None
    except TypeError, ValueError:
        return None


def _check_as_of(as_of: str | None) -> str | None:
    if as_of is None:
        return None
    if not isinstance(as_of, str) or not _AS_OF_RE.match(as_of):
        raise ValueError(f"invalid as_of date: {as_of!r} (expected YYYY-MM-DD)")
    try:
        date.fromisoformat(as_of)
    except ValueError:
        raise ValueError(f"invalid as_of date: {as_of!r} (expected YYYY-MM-DD)") from None
    return as_of


def _concept_rows(facts: Mapping[str, object], concept: str) -> list[Mapping[str, object]]:
    rows = facts.get("financial_facts")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("concept") == concept]


def _lineage_order(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (str(row.get("period_end") or ""), str(row.get("filed_at") or ""), str(row.get("accession") or ""))


def xbrl_lineage(
    entity_id: str,
    concept: str,
    *,
    as_of: str | None = None,
    root: Path | None = None,
) -> list[dict[str, object]]:
    """Newest period_end first; as_of gates on known_at (strict YYYY-MM-DD)."""
    del root
    bound = _check_as_of(as_of)
    cik = _cik_from_entity(entity_id)
    if cik is None:
        return []
    from ..data_sources import SourceGateway

    facts = SourceGateway().company_facts(cik, as_of=bound)
    rows = _concept_rows(facts, concept)
    rows.sort(key=_lineage_order, reverse=True)
    return [fact_lineage(row) for row in rows]
