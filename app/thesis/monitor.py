"""Deterministic monitoring + duplicate-safe ticks (stdlib + PyYAML only).

A tick queries canonical Stockbot sources from persisted checkpoints, normalizes
hits to stable canonical events, persists one pending trigger per new meaningful
event, then runs each pending trigger oldest-first via ``run_trigger`` (one
kernel scheduler run each). Ticks with nothing new make zero kernel runs.

No broker, price, Greeks, or options monitoring exists here on purpose: rules
without a reliable canonical backing stay ``enabled: false``/``unsupported``
and are never queried. Never fake data: production services with no resolvable
targets return no events instead of inventing any.

Feed contract: new feed builders MUST build summaries from bounded typed fields
plus canonical refs with origin ``"deterministic"``; raw post/article/feed text
MUST NEVER enter ``summary`` or ``metadata`` — only a canonical ref — and raw
text stays retrievable solely through a guarded Stockbot tool (built with the
feeds, out of scope here).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict
from urllib.parse import urlsplit

from app.thesis.models import Checkpoint, JSONValue, Thesis, Trigger, WatchRule
from app.thesis.repository import ThesisRepository
from app.thesis.runner import RunOutcome, run_trigger
from app.thesis.yaml import load_raw_yaml, thesis_lock

if TYPE_CHECKING:
    from app.sec.models import Filing, RegulatoryEvent

logger = logging.getLogger(__name__)

_MAX_DUP_MARKERS = 1_000
# ponytail: recent-hash markers capped at newest 1_000; stable per-source
# cursors are the primary resume path, hashes only cover cursor-less sources.
_DEEP_REVIEW_DAYS = 30
# ponytail: FINRA "material" = metrics-payload change between settlement cycles
# (magnitude-blind); add a size bar here when Stockbot analytics grows one.


@dataclass(frozen=True)
class CanonicalEvent:
    """One normalized source hit: stable IDs, PIT timestamp, deterministic summary."""

    event_id: str
    canonical_ref: str
    source: str
    known_at: str
    entity: str = ""
    summary: str = ""
    content_hash: str = ""
    cycle: str | None = None
    cursor: str | None = None
    file_id: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


class SourceService(Protocol):
    """Injectable canonical query surface (tests use fakes; CLI wires real ones)."""

    name: str

    def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]: ...


@dataclass
class TickResult:
    thesis_id: str
    triggers_created: list[str] = field(default_factory=list)  # trigger_id strings
    runs: list[RunOutcome] = field(default_factory=list)  # RunOutcome objects
    no_op: bool = False
    no_op_reason: str = ""
    checkpoint: dict[str, JSONValue] = field(default_factory=dict)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _digest(canonical_ref: str, summary: str, known_at: str) -> str:
    return hashlib.sha256(f"{canonical_ref}\n{summary}\n{known_at}".encode()).hexdigest()


def _as_dt(value: str) -> datetime | None:
    try:
        out = datetime.fromisoformat(value)
    except ValueError, TypeError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)  # bare dates are UTC days
    return out


def _le(a: str, b: str) -> bool:
    """Point-in-time ``a <= b``; unparseable values compare as raw strings."""
    da, db = _as_dt(a), _as_dt(b)
    if da is not None and db is not None:
        return da <= db
    return a <= b


def _day(ts: str) -> str:
    ts = ts or ""
    return ts[:10] if len(ts) >= 10 else ts


def targets_for_thesis(thesis: Thesis) -> tuple[str, ...]:
    """Explicit scope only: a single 1-5 letter token, else no targets."""
    scope = str(getattr(thesis, "scope", "") or "").strip()
    if re.fullmatch(r"[A-Za-z]{1,5}", scope):
        return (scope.upper(),)
    return ()


def _blob_texts(thesis: Thesis, key: str) -> list[str]:
    """Raw texts for one free-text thesis list field."""
    return [str(p) for p in (getattr(thesis, key, None) or []) if p is not None]


def _blob_claim_texts(thesis: Thesis) -> list[str]:
    """Claim statements across the thesis."""
    return [str(getattr(c, "statement", "") or "") for c in (getattr(thesis, "claims", None) or [])]


def _blob_expression_texts(thesis: Thesis) -> list[str]:
    """Intent/instrument/structure texts across expressions."""
    out: list[str] = []
    for e in getattr(thesis, "expressions", None) or []:
        out.append(str(getattr(e, "intent", "") or ""))
        out.append(str(getattr(e, "instrument", "") or ""))
        out.append(str(getattr(e, "structure", "") or ""))
    return out


def _thesis_blob(thesis: Thesis) -> str:
    parts = [str(getattr(thesis, "user_thesis", "") or ""), str(getattr(thesis, "scope", "") or "")]
    parts.extend(_blob_texts(thesis, "assumptions"))
    parts.extend(_blob_texts(thesis, "invalidators"))
    parts.extend(_blob_texts(thesis, "unknowns"))
    parts.extend(_blob_claim_texts(thesis))
    parts.extend(_blob_expression_texts(thesis))
    return "\n".join(parts).lower()


def _relevant(event: CanonicalEvent, targets: tuple[str, ...], blob: str) -> bool:
    """Keep events that name the thesis; drop clearly irrelevant entities.

    An empty entity carries no signal either way, so it stays (the scheduler decides).
    """
    ent = (event.entity or "").strip()
    if not ent:
        return True
    if ent.upper() in {t.upper() for t in targets}:
        return True
    toks = [t for t in re.split(r"[^a-z0-9]+", ent.lower()) if len(t) >= 3]
    return any(t in blob for t in toks)


def _importance(rule_type: str) -> str:
    return {"explicit_thesis_invalidator": "high", "scheduled_deep_review": "low"}.get(rule_type, "medium")


# -- source query phases: window/fetch/map -------------------------------------
# Each production query_since splits into a checkpoint-window helper plus a
# per-hit mapper so the service method stays a thin fetch loop.


def _filings_window(checkpoint: Mapping[str, object], since_default: str | None) -> str | None:
    """Checkpoint cursor (day) for SEC filing discovery; None when unbounded."""
    return _day(str(checkpoint.get("cursor") or "")) or _day(since_default or "") or None


def _filing_known_at(filing: Filing, fallback: str) -> str:
    """PIT timestamp for one filing (known_at, else filed_at, else cutoff)."""
    return filing.known_at or filing.filed_at or fallback


def _filing_is_amendment(filing: Filing) -> bool:
    """True for amended filings (flag or /A form suffix)."""
    return bool(filing.is_amendment or (filing.form or "").endswith("/A"))


def _filing_summary(target: str, filing: Filing) -> tuple[str, str]:
    """Stable (ref, summary) for one filing."""
    ref = f"sec:{filing.accession_no}"
    return ref, f"{filing.form} filed by {filing.filer_name or target} (filed {filing.filed_at or 'unknown date'})"


def _filing_to_event(target: str, filing: Filing, *, known_at: str, source: str) -> CanonicalEvent | None:
    """Map one filing to a PIT-gated event; None when future-known at cutoff."""
    ka = _filing_known_at(filing, known_at)
    if ka and known_at and not _le(ka, known_at):
        return None
    ref, summary = _filing_summary(target, filing)
    return CanonicalEvent(
        event_id=ref,
        canonical_ref=ref,
        source=source,
        known_at=ka,
        entity=target,
        summary=summary,
        content_hash=_digest(ref, summary, ka),
        cursor=_day(ka) or None,
        file_id=filing.accession_no,
        metadata={"amendment": _filing_is_amendment(filing), "form": filing.form, "link": filing.source},
    )


def _material_window(checkpoint: Mapping[str, object], since_default: str | None) -> str:
    """Checkpoint cursor (day) for material events; floored at 2000-01-01."""
    return _day(str(checkpoint.get("cursor") or "")) or _day(since_default or "") or "2000-01-01"


def _material_summary(hit: RegulatoryEvent) -> tuple[str, str]:
    """Stable (event_id, summary) for one material event."""
    return hit.event_id, (
        f"{hit.event_type} ({hit.issuer}) effective {hit.effective_date or 'unknown date'} severity {hit.severity}"
    )


def _material_to_event(target: str, hit: RegulatoryEvent, *, known_at: str, source: str) -> CanonicalEvent | None:
    """Map one material event to a PIT-gated event; None when future-known."""
    ka = hit.known_at or known_at
    if ka and known_at and not _le(ka, known_at):
        return None
    eid, summary = _material_summary(hit)
    accs = list(hit.source_accessions or [])
    return CanonicalEvent(
        event_id=eid,
        canonical_ref=eid,
        source=source,
        known_at=ka,
        entity=target,
        summary=summary,
        content_hash=_digest(eid, summary, ka),
        cursor=_day(ka) or None,
        file_id=accs[0] if accs else None,
        metadata={"event_type": hit.event_type, "issuer": hit.issuer},
    )


def _finra_cycle(res: dict[str, object]) -> str:
    """Settlement-cycle identity; empty when the payload names none."""
    return str(res.get("as_of_date") or "")


def _finra_trends(res: dict[str, object]) -> list[str]:
    """Trend texts riding along in the FINRA payload."""
    raw = res.get("trends")
    return [str(t) for t in raw] if isinstance(raw, list) else []


def _finra_summary(target: str, cycle: str, trends: list[str]) -> tuple[str, str]:
    """Stable (ref, summary) for one FINRA settlement cycle."""
    ref = f"finra-si:{target}:{cycle}"
    summary = f"short-interest settlement {cycle} for {target}" + (f": {'; '.join(trends)}" if trends else "")
    return ref, summary


def _finra_to_event(target: str, res: dict[str, object], *, known_at: str, source: str) -> CanonicalEvent | None:
    """Map one FINRA payload to its cycle event; None when unorderable/future."""
    cycle = _finra_cycle(res)
    if not cycle or not _le(cycle, _day(known_at)):
        return None  # no orderable cycle: never invent one
    trends = _finra_trends(res)
    ref, summary = _finra_summary(target, cycle, trends)
    return CanonicalEvent(
        event_id=ref,
        canonical_ref=ref,
        source=source,
        known_at=cycle,
        entity=target,
        summary=summary,
        content_hash=_digest(ref, summary, "|".join(trends)),
        cycle=cycle,
        cursor=cycle,
        metadata={"ticker": target, "cycle": cycle, "trends": [t for t in trends]},
    )


FinraFetch = Callable[[str], object]


def _finra_failure(target: str, res: object) -> str:
    """One-line failure for an error-dict/empty FINRA payload."""
    err = res.get("error", "unknown FINRA error") if isinstance(res, dict) else "unknown FINRA error"
    return f"{target}: {err}"


def _finra_query_target(
    target: str, fetch: FinraFetch, *, known_at: str, source: str
) -> tuple[CanonicalEvent | None, str | None]:
    """Fetch one FINRA target: (event, None), (None, error), or (None, None) on skip."""
    try:
        res = fetch(target)
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None, f"{target}: {type(exc).__name__}: {exc}"
    if not isinstance(res, dict) or res.get("error"):
        return None, _finra_failure(target, res)
    ev = _finra_to_event(target, res, known_at=known_at, source=source)
    return (ev, None) if ev is not None else (None, None)


def _rule_order(rule: WatchRule) -> tuple[str, str]:
    return (rule.rule_type, rule.rule_id)


# -- production source services (thin, PIT-gated, empty-safe) -----------------


class SecFilingsService:
    """New/amended filings via SEC discovery, ``as_of``-gated to ``known_at``."""

    name = "sec_filings"

    def __init__(self, targets: tuple[str, ...] = (), *, since_default: str | None = None) -> None:
        self.targets = tuple(targets)
        self.since_default = since_default

    def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]:
        if not self.targets:
            return []
        from app.sec.filings import list_sec_filings

        since = _filings_window(checkpoint, self.since_default)
        out: list[CanonicalEvent] = []
        for target in self.targets:
            for f in list_sec_filings(target, start_date=since, as_of=_day(known_at), limit=50):
                ev = _filing_to_event(target, f, known_at=known_at, source=self.name)
                if ev is not None:
                    out.append(ev)
        return out


class MaterialEventsService:
    """8-K material events via the SEC material-event path, PIT-gated."""

    name = "material_events"

    def __init__(self, targets: tuple[str, ...] = (), *, since_default: str | None = None) -> None:
        self.targets = tuple(targets)
        self.since_default = since_default

    def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]:
        if not self.targets:
            return []
        from app.sec.material import get_material_events

        since = _material_window(checkpoint, self.since_default)
        out: list[CanonicalEvent] = []
        for target in self.targets:
            for e in get_material_events(target, since, as_of=_day(known_at), limit=50):
                ev = _material_to_event(target, e, known_at=known_at, source=self.name)
                if ev is not None:
                    out.append(ev)
        return out


class FinraShortInterestService:
    """Latest FINRA short-interest settlement cycle per target (no backfill).

    Cycle identity is the briefing ``as_of_date``; trend text rides along in
    metadata so handlers can tell a new cycle from a changed one.
    """

    name = "finra_short_interest"

    def __init__(self, targets: tuple[str, ...] = ()) -> None:
        self.targets = tuple(targets)

    def query_since(self, checkpoint: Mapping[str, object], *, known_at: str) -> list[CanonicalEvent]:
        if not self.targets:
            return []
        from app.finra_client import get_short_interest

        out: list[CanonicalEvent] = []
        errors: list[str] = []
        for target in self.targets:
            ev, err = _finra_query_target(target, get_short_interest, known_at=known_at, source=self.name)
            if ev is not None:
                out.append(ev)
            elif err is not None:
                errors.append(err)
        if not out and errors:
            raise RuntimeError(f"<monitor>: FINRA short-interest query failed: {errors[0]}")
        return out


# -- supported handlers: rule_type -> fn --------------------------------------
# Each handler maps already-queried source events (or local checkpoint state)
# to candidate CanonicalEvents plus its new per-source cursor/detail state.


def _cursor_days(events: list[CanonicalEvent]) -> list[str]:
    """Non-empty cursor/known_at days across events."""
    return [d for d in (_day(e.cursor or e.known_at or "") for e in events) if d]


def _cursor_state(events: list[CanonicalEvent], known_at: str) -> dict[str, JSONValue]:
    days = _cursor_days(events)
    return {"cursor": max(days) if days else _day(known_at)}


def _is_new_filing(event: CanonicalEvent) -> bool:
    """True for non-amendment SEC filings."""
    return not (event.metadata or {}).get("amendment")


def _is_amendment(event: CanonicalEvent) -> bool:
    """True for amendment SEC filings."""
    return bool((event.metadata or {}).get("amendment"))


def _handle_new_filing(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    filings = source_events.get("sec_filings", [])
    return [e for e in filings if _is_new_filing(e)], _cursor_state(filings, known_at)


def _handle_filing_change(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    filings = source_events.get("sec_filings", [])
    return [e for e in filings if _is_amendment(e)], _cursor_state(filings, known_at)


def _handle_material_event(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    evs = list(source_events.get("material_events", []))
    return evs, _cursor_state(evs, known_at)


def _finra_ticker_entry(event: CanonicalEvent) -> tuple[str, dict[str, JSONValue]] | None:
    """One per-ticker FINRA detail row; None when the event carries no cycle."""
    meta = event.metadata or {}
    if not meta.get("cycle"):
        return None
    raw = meta.get("trends") or []
    rows = raw if isinstance(raw, (list, dict, str)) else []
    return str(meta.get("ticker") or event.entity), {"cycle": meta["cycle"], "trends": list[JSONValue](rows)}


def _finra_prior_cursor(prev: Mapping[str, JSONValue], cursor: str) -> str:
    """Max of the stored FINRA cursor and the newest event day."""
    prior = prev.get("cursor")
    return max(str(prior), cursor) if prior else cursor


def _finra_cursor(all_evs: list[CanonicalEvent], prev: Mapping[str, JSONValue], known_at: str) -> str:
    """Monotone FINRA cursor over prior state plus this tick's events."""
    return _finra_prior_cursor(prev, str(_cursor_state(all_evs, known_at).get("cursor") or ""))


def _finra_state(all_evs: list[CanonicalEvent], prev: Mapping[str, JSONValue], known_at: str) -> dict[str, JSONValue]:
    tickers: dict[str, JSONValue] = {}
    for e in all_evs:
        row = _finra_ticker_entry(e)
        if row is not None:
            tickers[row[0]] = row[1]
    return {"cursor": _finra_cursor(all_evs, prev, known_at), "tickers": tickers}


def _si_prev(sources: dict[str, JSONValue]) -> tuple[dict[str, JSONValue], str, dict[str, JSONValue]]:
    """Prior FINRA checkpoint as (prev dict, cursor, seen-tickers dict)."""
    raw = sources.get("finra_short_interest")
    prev: dict[str, JSONValue] = dict(raw) if isinstance(raw, dict) else {}
    seen_raw = prev.get("tickers")
    seen: dict[str, JSONValue] = seen_raw if isinstance(seen_raw, dict) else {}
    return prev, str(prev.get("cursor") or ""), seen


def _si_cycle_fresh(event: CanonicalEvent, cursor: str) -> bool:
    """True when the FINRA event carries a cycle newer than the cursor."""
    cycle = (event.metadata or {}).get("cycle")
    return bool(cycle) and (not cursor or str(cycle) > cursor)


def _si_cycle_new(evs: list[CanonicalEvent], cursor: str) -> list[CanonicalEvent]:
    """FINRA events with a cycle newer than the stored cursor."""
    return [e for e in evs if _si_cycle_fresh(e, cursor)]


def _si_trends(event: CanonicalEvent) -> list[JSONValue]:
    """Trends payload for a FINRA event (empty when absent/invalid)."""
    raw = (event.metadata or {}).get("trends")
    return list(raw) if isinstance(raw, list) else []


def _si_candidate(event: CanonicalEvent, cursor: str) -> tuple[str, str, list[JSONValue]] | None:
    """(cycle, ticker, trends) for a FINRA event; None when stale/unorderable."""
    meta = event.metadata or {}
    cycle = str(meta.get("cycle") or "")
    if not cycle or (cursor and cycle <= cursor):
        return None
    return cycle, str(meta.get("ticker") or event.entity), _si_trends(event)


def _si_stored_trends(seen: dict[str, JSONValue], ticker: str) -> object | None:
    """Stored trends payload for a ticker; None when never seen."""
    prior_raw = seen.get(ticker)
    prior: dict[str, JSONValue] = prior_raw if isinstance(prior_raw, dict) else {}
    return prior.get("trends")


def _si_trend_changed(event: CanonicalEvent, cursor: str, seen: dict[str, JSONValue]) -> bool:
    """True when the event's trends payload changed vs stored tickers."""
    cand = _si_candidate(event, cursor)
    if cand is None:
        return False
    _cycle, ticker, trends = cand
    prior = _si_stored_trends(seen, ticker)
    return prior is not None and _trends_of(prior) != _trends_of(trends)


def _trends_of(value: object) -> list[object]:
    """Normalize a trends payload to a list for comparison."""
    return list(value) if isinstance(value, (list, dict, str)) else []


def _si_changed_trends(evs: list[CanonicalEvent], cursor: str, seen: dict[str, JSONValue]) -> list[CanonicalEvent]:
    """FINRA cycle events whose trends payload changed vs stored tickers."""
    return [e for e in evs if _si_trend_changed(e, cursor, seen)]


def _handle_si_cycle(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    prev, cursor, _seen = _si_prev(sources)
    evs = _si_cycle_new(source_events.get("finra_short_interest", []), cursor)
    return evs, _finra_state(source_events.get("finra_short_interest", []), prev, known_at)


def _handle_si_material(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    prev, cursor, seen = _si_prev(sources)
    out = _si_changed_trends(source_events.get("finra_short_interest", []), cursor, seen)
    return out, _finra_state(source_events.get("finra_short_interest", []), prev, known_at)


def _handle_external(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    # Dead: no production source exists; kept only so old imports don't break.
    # Never referenced from SUPPORTED_HANDLERS/_SOURCE_FOR_RULE/_STATE_KEY.
    evs = list(source_events.get("external_evidence", []))
    return evs, _cursor_state(evs, known_at)


def _invalidator_tokens(thesis: Thesis) -> set[str]:
    """Keyword tokens (>=5 chars) from the thesis invalidators."""
    return {
        w
        for inv in (getattr(thesis, "invalidators", None) or [])
        for w in re.split(r"[^a-z0-9]+", str(inv).lower())
        if len(w) >= 5
    }


def _invalidator_pool(
    source_events: dict[str, list[CanonicalEvent]], stored: tuple[CanonicalEvent, ...]
) -> list[CanonicalEvent]:
    """This tick's events plus already-stored refs as the match pool."""
    return [e for evs in source_events.values() for e in evs] + list(stored)


def _invalidator_match(event: CanonicalEvent, toks: set[str]) -> bool:
    """True when any invalidator token names the event summary/ref/entity."""
    return any(t in f"{event.summary} {event.canonical_ref} {event.entity}".lower() for t in toks)


def _invalidator_event(rule: WatchRule, event: CanonicalEvent, known_at: str) -> CanonicalEvent:
    """Wrap a matched event as a stable invalidator candidate."""
    ref = f"invalidator:{rule.rule_id}:{hashlib.sha256(event.canonical_ref.encode()).hexdigest()[:8]}"
    summary = f"possible invalidator ({rule.rule_type}): {event.summary}"
    return CanonicalEvent(
        event_id=ref,
        canonical_ref=ref,
        source="watch",
        known_at=event.known_at or known_at,
        entity=event.entity,
        summary=summary,
        content_hash=_digest(ref, summary, event.known_at or known_at),
    )


def _scan_invalidator_pool(
    rule: WatchRule, pool: list[CanonicalEvent], toks: set[str], known_at: str
) -> list[CanonicalEvent]:
    """First match per canonical ref, in pool order."""
    out: list[CanonicalEvent] = []
    matched: set[str] = set()
    for e in pool:
        if e.canonical_ref in matched or not _invalidator_match(e, toks):
            continue
        matched.add(e.canonical_ref)
        out.append(_invalidator_event(rule, e, known_at))
    return out


def _handle_invalidator(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    # Deterministic semantic match, no source query: invalidator keywords from
    # the thesis against this tick's events plus already-stored canonical refs
    # (evidence files, earlier triggers). Refs stay stable, so a repeat tick
    # dedups via the persisted trigger instead of refiring.
    toks = _invalidator_tokens(thesis)
    if not toks:
        return [], {}
    return _scan_invalidator_pool(rule, _invalidator_pool(source_events, stored), toks, known_at), {}


def _review_last(sources: dict[str, JSONValue]) -> str:
    """Stored deep-review cursor; empty when never reviewed."""
    raw = sources.get("scheduled")
    sched: dict[str, JSONValue] = raw if isinstance(raw, dict) else {}
    return str(sched.get("cursor") or "")


def _review_due(last: str, today: str) -> bool:
    """True when the scheduled deep review is due (first run always due)."""
    if not last:
        return True
    dl, dt = _as_dt(last), _as_dt(today)
    # ponytail: date subtraction covers it; unparseable cursors fall back
    # to lexicographic compare (ISO dates order lexically).
    return (dt - dl).days >= _DEEP_REVIEW_DAYS if dl is not None and dt is not None else today > last


def _review_event(thesis: Thesis, last: str, today: str, known_at: str) -> CanonicalEvent:
    """One scheduled deep-review candidate for today."""
    ref = f"review:{thesis.thesis_id}:{today}"
    summary = f"scheduled deep review ({thesis.slug}); last reviewed {last or 'never'}"
    return CanonicalEvent(
        event_id=ref,
        canonical_ref=ref,
        source="watch",
        known_at=known_at,
        entity=thesis.slug,
        summary=summary,
        content_hash=_digest(ref, summary, known_at),
        cursor=today,
    )


def _handle_deep_review(
    *,
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...] = (),
) -> tuple[list[CanonicalEvent], dict[str, JSONValue]]:
    last = _review_last(sources)
    today = _day(known_at)
    if not _review_due(last, today):
        return [], {}
    return [_review_event(thesis, last, today, known_at)], {"cursor": today}


SUPPORTED_HANDLERS = {
    "new_filing": _handle_new_filing,
    "filing_change": _handle_filing_change,
    "new_material_event": _handle_material_event,
    "new_short_interest_cycle": _handle_si_cycle,
    "material_short_interest_change": _handle_si_material,
    "explicit_thesis_invalidator": _handle_invalidator,
    "scheduled_deep_review": _handle_deep_review,
}

_SOURCE_FOR_RULE = {
    "new_filing": "sec_filings",
    "filing_change": "sec_filings",
    "new_material_event": "material_events",
    "new_short_interest_cycle": "finra_short_interest",
    "material_short_interest_change": "finra_short_interest",
    "explicit_thesis_invalidator": None,
    "scheduled_deep_review": None,
}

_STATE_KEY = {
    "new_filing": "sec_filings",
    "filing_change": "sec_filings",
    "new_material_event": "material_events",
    "new_short_interest_cycle": "finra_short_interest",
    "material_short_interest_change": "finra_short_interest",
    "explicit_thesis_invalidator": None,
    "scheduled_deep_review": "scheduled",
}


class _Grouped(TypedDict):
    rule: WatchRule
    event: CanonicalEvent
    digest: str
    claims: set[str]
    exprs: set[str]
    digests: set[str]


@dataclass
class _TickScope:
    """Live per-tick view: thesis, rules, targets, and relevance blob."""

    thesis: Thesis
    tid: str
    live: list[WatchRule]
    targets: tuple[str, ...]
    blob: str


def _tick_gate(repository: ThesisRepository, thesis_id: str) -> tuple[Thesis, str] | TickResult:
    """Load thesis + status gate: closed raises, paused returns its TickResult."""
    thesis = repository.load_thesis(thesis_id)
    tid = thesis.thesis_id
    with thesis_lock(repository.root / thesis.slug):
        status = repository.load_thesis(tid).status
    if status == "closed":
        raise ValueError(f"{repository.root / thesis.slug}: thesis {tid!r} is closed; cannot tick")
    if status == "paused":
        cp = repository.load_checkpoint(tid)
        return TickResult(thesis_id=tid, no_op=True, no_op_reason="paused", checkpoint=cp.to_dict())
    repository.normalize_watch(tid)
    return thesis, tid


def _active_targets(thesis: Thesis) -> tuple[set[str], set[str]]:
    """Claim/expression IDs still eligible for monitoring."""
    claims = {c.claim_id for c in thesis.claims if c.status != "invalidated"}
    exprs = {e.expression_id for e in thesis.expressions if e.status != "closed"}
    return claims, exprs


def _rule_live(rule: WatchRule, claims: set[str], exprs: set[str]) -> bool:
    """True for supported, enabled, actively-targeted watches."""
    if not (rule.enabled and rule.support_status == "supported"):
        return False
    if rule.rule_type not in SUPPORTED_HANDLERS:
        return False
    return bool((set(rule.claim_ids) & claims) or (set(rule.expression_ids) & exprs))


def _tick_scope(repository: ThesisRepository, thesis: Thesis, tid: str) -> _TickScope:
    """Live rules filtered to supported, enabled, actively-targeted watches."""
    rules: list[WatchRule] = repository.load_watch_rules(tid)
    claims, exprs = _active_targets(thesis)
    live = [r for r in sorted(rules, key=_rule_order) if _rule_live(r, claims, exprs)]
    return _TickScope(thesis=thesis, tid=tid, live=live, targets=targets_for_thesis(thesis), blob=_thesis_blob(thesis))


def _call_source(
    svc: SourceService, sources: dict[str, JSONValue], key: str, known_at: str
) -> list[CanonicalEvent] | None:
    """Call one wired source; None when it fails."""
    try:
        seen = svc.query_since(_source_checkpoint(sources, key), known_at=known_at)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None  # failed source: checkpoint never advances
    return [e for e in (seen or []) if isinstance(e, CanonicalEvent)]


def _query_source(
    services: dict[str, SourceService], key: str, sources: dict[str, JSONValue], known_at: str
) -> tuple[str, list[CanonicalEvent] | None]:
    """Query one source from its checkpoint; None events on unwired/failed."""
    svc = services.get(key)
    if svc is None:
        return key, None  # unwired source: never queried, checkpoint untouched
    return key, _call_source(svc, sources, key, known_at)


def _source_checkpoint(sources: dict[str, JSONValue], key: str) -> dict[str, JSONValue]:
    """Checkpoint slice for one source (empty when absent/corrupt)."""
    raw = sources.get(key)
    return dict(raw) if isinstance(raw, dict) else {}


def _query_sources(
    services: dict[str, SourceService], live: list[WatchRule], sources: dict[str, JSONValue], known_at: str
) -> tuple[dict[str, list[CanonicalEvent]], set[str]]:
    """Query each needed source once from its checkpoint up to known_at."""
    needed = _needed_sources(live)
    source_events: dict[str, list[CanonicalEvent]] = {}
    queried_ok: set[str] = set()
    for key in needed:
        _key, evs = _query_source(services, key, sources, known_at)
        if evs is None:
            continue
        source_events[_key] = evs
        queried_ok.add(_key)
    return source_events, queried_ok


def _needed_sources(live: list[WatchRule]) -> list[str]:
    """Sorted source keys needed by the live rules."""
    return sorted({v for v in (_SOURCE_FOR_RULE[r.rule_type] for r in live) if v is not None})


def _seen_markers(
    repository: ThesisRepository, tid: str, checkpoint: Checkpoint
) -> tuple[list[Trigger], set[str], set[str]]:
    """Existing triggers plus content-hash markers for the dedup filters."""
    existing = repository.load_triggers(tid)  # created-time order already
    # ponytail: inbox scan is O(n) per tick; index by canonical ref past thousands.
    seen_refs = {ref for t in existing for ref in t.canonical_refs}
    seen_hashes = set(checkpoint.recent_hashes or {})
    for t in existing:
        _collect_trigger_hash(t, seen_hashes)
    return existing, seen_refs, seen_hashes


def _collect_trigger_hash(t: Trigger, seen_hashes: set[str]) -> None:
    """Add a trigger's content-hash marker when present."""
    ch = t.metadata.get("content_hash") if isinstance(t.metadata, dict) else None
    if isinstance(ch, str) and ch:
        seen_hashes.add(ch)


def _stored_file_event(raw: dict[str, JSONValue], tid: str, known_at: str) -> CanonicalEvent | None:
    """One evidence-file row as a stored event; None when missing/future."""
    if raw.get("thesis_id") != tid:
        return None
    ref, ka, summary = _stored_file_fields(raw)
    if not ref or not ka or not _le(ka, known_at):
        return None  # missing or future-known evidence is never visible
    return CanonicalEvent(
        event_id=ref,
        canonical_ref=ref,
        source="stored",
        known_at=ka,
        summary=summary,
        content_hash=_digest(ref, summary, ka),
    )


def _stored_file_fields(raw: dict[str, JSONValue]) -> tuple[str, str, str]:
    """(canonical_ref, known_at, summary) for one evidence-file row."""
    return (str(raw.get("canonical_ref") or ""), str(raw.get("known_at") or ""), str(raw.get("summary") or ""))


def _stored_trigger_event(t: Trigger, known_at: str) -> list[CanonicalEvent]:
    """Earlier-trigger refs as stored events; empty when future-known."""
    if t.created_at and not _le(t.created_at, known_at):
        return []  # future-known trigger: never visible at this tick
    return [_trigger_stored_event(t, ref) for ref in t.canonical_refs]


def _trigger_stored_event(t: Trigger, ref: str) -> CanonicalEvent:
    """One earlier-trigger ref as a stored event."""
    return CanonicalEvent(
        event_id=ref,
        canonical_ref=ref,
        source="stored",
        known_at=t.created_at,
        summary=t.summary or "",
        content_hash=_digest(ref, t.summary or "", t.created_at),
    )


def _stored_file_events(repository: ThesisRepository, tid: str, slug: str, known_at: str) -> list[CanonicalEvent]:
    """PIT-visible evidence-file events for the handler pool."""
    stored: list[CanonicalEvent] = []
    evdir = repository.root / slug / "evidence"
    if not evdir.is_dir():
        return stored
    for f in sorted(evdir.glob("*.yaml")):
        ev = _load_stored_file(f, tid, known_at)
        if ev is not None:
            stored.append(ev)
    return stored


def _load_stored_file(f: object, tid: str, known_at: str) -> CanonicalEvent | None:
    """Load one evidence file as a stored event; None when unreadable."""
    if not isinstance(f, (str, Path)):
        return None
    try:
        raw = load_raw_yaml(f)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if not isinstance(raw, dict):
        return None
    return _stored_file_event(raw, tid, known_at)


def _stored_evidence(
    repository: ThesisRepository, tid: str, slug: str, existing: list[Trigger], known_at: str
) -> tuple[CanonicalEvent, ...]:
    """Evidence-file + earlier-trigger pool for the handler (PIT-gated)."""
    stored = _stored_file_events(repository, tid, slug, known_at)
    for t in existing:
        stored.extend(_stored_trigger_event(t, known_at))
    return tuple(stored)


def _candidate_digest(event: CanonicalEvent) -> str:
    """Content hash for a candidate event (stored or recomputed)."""
    return event.content_hash or _digest(event.canonical_ref, event.summary, event.known_at)


def _pit_visible_event(event: CanonicalEvent, known_at: str) -> bool:
    """False for ref-less or future-known candidates."""
    if not event.canonical_ref:
        return False
    return not (event.known_at and known_at and not _le(event.known_at, known_at))


def _dup_seen(
    event: CanonicalEvent, digest: str, seen_refs: set[str], seen_hashes: set[str], grouped: dict[str, _Grouped]
) -> bool:
    """True when already processed (checkpoint/trigger) or identical in-tick."""
    if digest in seen_hashes or event.canonical_ref in seen_refs:
        return True  # checkpoint / processed-marker / existing-trigger dup
    return event.canonical_ref in grouped and digest in grouped[event.canonical_ref]["digests"]


def _should_keep(
    event: CanonicalEvent,
    digest: str,
    known_at: str,
    seen_refs: set[str],
    seen_hashes: set[str],
    grouped: dict[str, _Grouped],
    targets: tuple[str, ...],
    blob: str,
) -> bool:
    """Global dedup + PIT + relevance gate for one handler candidate."""
    if not _pit_visible_event(event, known_at):
        return False
    if _dup_seen(event, digest, seen_refs, seen_hashes, grouped):
        return False
    return _relevant(event, targets, blob)


def _fold_candidate(grouped: dict[str, _Grouped], rule: WatchRule, event: CanonicalEvent, digest: str) -> None:
    """Fold one kept candidate into its canonical-ref group."""
    g = grouped.setdefault(
        event.canonical_ref,
        {
            "rule": rule,
            "event": event,
            "digest": digest,
            "claims": set[str](),
            "exprs": set[str](),
            "digests": set[str](),
        },
    )
    g["claims"] |= set(rule.claim_ids)
    g["exprs"] |= set(rule.expression_ids)
    g["digests"].add(digest)


def _apply_rule(
    rule: WatchRule,
    thesis: Thesis,
    source_events: dict[str, list[CanonicalEvent]],
    sources: dict[str, JSONValue],
    known_at: str,
    stored: tuple[CanonicalEvent, ...],
    seen_refs: set[str],
    seen_hashes: set[str],
    targets: tuple[str, ...],
    blob: str,
    grouped: dict[str, _Grouped],
    pending_states: dict[str, dict[str, JSONValue]],
    queried_ok: set[str],
) -> None:
    """Run one rule handler and fold candidates through the dedup filters."""
    evs, state = SUPPORTED_HANDLERS[rule.rule_type](
        rule=rule, thesis=thesis, source_events=source_events, sources=sources, known_at=known_at, stored=stored
    )
    _record_state(rule, state, pending_states, queried_ok)
    for e in evs:
        digest = _candidate_digest(e)
        if not _should_keep(e, digest, known_at, seen_refs, seen_hashes, grouped, targets, blob):
            continue
        _fold_candidate(grouped, rule, e, digest)


def _record_state(
    rule: WatchRule, state: dict[str, JSONValue], pending_states: dict[str, dict[str, JSONValue]], queried_ok: set[str]
) -> None:
    """Record a rule's state when its source queried OK (scheduled always)."""
    skey = _STATE_KEY[rule.rule_type]
    if skey is not None and (skey == "scheduled" or skey in queried_ok):
        pending_states[skey] = state


def _trigger_metadata(event: CanonicalEvent, digest: str) -> dict[str, JSONValue]:
    """Trigger metadata: identity + optional locators + event payload."""
    meta: dict[str, JSONValue] = {
        "event_id": event.event_id,
        "source": event.source,
        "content_hash": digest,
        "entity": event.entity,
        "event_known_at": event.known_at,
    }
    for k, v in {"cycle": event.cycle, "cursor": event.cursor, "file_id": event.file_id}.items():
        if v is not None:
            meta[k] = v
    meta.update(event.metadata or {})
    return meta


def _summary_origin(event: CanonicalEvent, ref: str) -> str:
    """Recycled for stored/invalidator hits, else deterministic."""
    if event.source == "stored" or ref.startswith("invalidator:"):
        return "recycled"
    return "deterministic"


def _persist_group(
    repository: ThesisRepository,
    tid: str,
    ref: str,
    g: _Grouped,
    seen_refs: set[str],
    seen_hashes: set[str],
    new_hashes: list[str],
) -> Trigger:
    """Persist one grouped canonical event as a pending trigger."""
    trig = repository.create_trigger(
        tid,
        trigger_type=g["rule"].rule_type,
        importance=_importance(g["rule"].rule_type),
        claim_ids=sorted(g["claims"]),
        expression_ids=sorted(g["exprs"]),
        canonical_refs=[ref],
        summary=g["event"].summary,
        summary_origin=_summary_origin(g["event"], ref),
        metadata=_trigger_metadata(g["event"], g["digest"]),
    )
    seen_refs.add(ref)
    seen_hashes.add(g["digest"])
    new_hashes.append(g["digest"])
    return trig


def _persist_triggers(
    repository: ThesisRepository, tid: str, grouped: dict[str, _Grouped], seen_refs: set[str], seen_hashes: set[str]
) -> tuple[list[Trigger], list[str]]:
    """One trigger per grouped canonical event; returns (created, new_hashes)."""
    created: list[Trigger] = []
    new_hashes: list[str] = []
    for ref in sorted(grouped):
        created.append(_persist_group(repository, tid, ref, grouped[ref], seen_refs, seen_hashes, new_hashes))
    return created, new_hashes


def _run_pending(repository: ThesisRepository, tid: str, known_at: str) -> list[RunOutcome]:
    """Run each pending trigger oldest-first; a failure halts the tick."""
    pending = [t for t in repository.load_triggers(tid) if t.status == "pending"]
    runs: list[RunOutcome] = []
    for t in pending:
        try:
            runs.append(run_trigger(repository, tid, t.trigger_id, known_at=known_at))
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            break
    return runs


def _merge_source_state(sources: dict[str, JSONValue], key: str, state: dict[str, JSONValue], known_at: str) -> None:
    """Merge one queried source's state (FINRA tickers merge, rest replace)."""
    old = dict(sources.get(key) or {})
    rest = _merge_tickers(old, state)
    old.update(rest)
    old["updated_at"] = known_at
    sources[key] = old


def _merge_tickers(old: dict[str, JSONValue], state: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Merge FINRA tickers into the stored state; return the remaining fields."""
    tickers = state.get("tickers")
    if isinstance(tickers, dict):
        stored = old.get("tickers")
        base: dict[str, JSONValue] = dict(stored) if isinstance(stored, dict) else {}
        old["tickers"] = {**base, **tickers}
        return {k: v for k, v in state.items() if k != "tickers"}
    return dict(state)


def _advance_checkpoint(
    repository: ThesisRepository,
    tid: str,
    checkpoint: Checkpoint,
    sources: dict[str, JSONValue],
    pending_states: dict[str, dict[str, JSONValue]],
    new_hashes: list[str],
    known_at: str,
) -> Checkpoint:
    """Merge per-source state (never failed ones) and bound the dup markers."""
    for key, state in pending_states.items():
        _merge_source_state(sources, key, state, known_at)
    recent = (list(checkpoint.recent_hashes or []) + new_hashes)[-_MAX_DUP_MARKERS:]
    fresh = Checkpoint(thesis_id=tid, sources=sources, recent_hashes=recent)
    if fresh.to_dict() != checkpoint.to_dict():
        return repository.save_checkpoint(tid, fresh)
    return checkpoint


def _ntfy(thesis: Thesis, trig: Trigger) -> None:
    """Best-effort phone push of one new trigger to $NTFY_TOPIC_URL (http/https); never raises."""
    url = os.environ.get("NTFY_TOPIC_URL", "").strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return
    link = trig.metadata.get("link")
    headers = {"Title": f"{thesis.slug}: {trig.trigger_type}"}
    if isinstance(link, str) and link.startswith(("http://", "https://")):
        headers["Click"] = link
    body = f"{trig.summary}\n{headers.get('Click') or ', '.join(trig.canonical_refs)}"
    req = urllib.request.Request(url, data=body.encode(), headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:  # intentional best-effort boundary: a failed push never aborts a tick
        logger.warning("ntfy push failed for %s", trig.trigger_id, exc_info=True)


def tick(
    repository: ThesisRepository,
    thesis_id: str,
    source_services: Mapping[str, SourceService] | None = None,
    *,
    known_at: str | None = None,
) -> TickResult:
    """Run one live monitor tick with a PIT data cutoff.

    known_at bounds source, event, evidence, and trigger queries only; it never
    selects historical thesis, watch, claim, expression, checkpoint, or
    trigger-timestamp state, which are always read live.
    """
    cutoff = known_at or _utcnow()
    services = dict(source_services or {})
    gated = _tick_gate(repository, thesis_id)
    if isinstance(gated, TickResult):
        return gated
    thesis, tid = gated
    scope = _tick_scope(repository, thesis, tid)
    checkpoint = repository.load_checkpoint(tid)
    sources = dict(checkpoint.sources or {})
    source_events, queried_ok = _query_sources(services, scope.live, sources, cutoff)
    existing, seen_refs, seen_hashes = _seen_markers(repository, tid, checkpoint)
    pending_states: dict[str, dict[str, JSONValue]] = {}
    grouped: dict[str, _Grouped] = {}
    stored = _stored_evidence(repository, tid, thesis.slug, existing, cutoff)
    for rule in scope.live:
        _apply_rule(
            rule,
            thesis,
            source_events,
            sources,
            cutoff,
            stored,
            seen_refs,
            seen_hashes,
            scope.targets,
            scope.blob,
            grouped,
            pending_states,
            queried_ok,
        )
    created, new_hashes = _persist_triggers(repository, tid, grouped, seen_refs, seen_hashes)
    for trig in created:
        _ntfy(thesis, trig)
    runs = _run_pending(repository, tid, cutoff)
    checkpoint = _advance_checkpoint(repository, tid, checkpoint, sources, pending_states, new_hashes, cutoff)
    no_op = not created and not runs
    return TickResult(
        thesis_id=tid,
        triggers_created=[t.trigger_id for t in created],
        runs=runs,
        no_op=no_op,
        no_op_reason="",
        checkpoint=checkpoint.to_dict(),
    )
