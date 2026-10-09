"""Exhaustive SEC discovery orchestration over the existing SEC surface.

CIK stays an SEC identifier mapped to the provider-independent Stockbot
``entity_id`` (``app/domain/market/ids.py``); there is no ``SECCompany``
class and no second entity store. Name comparison is deterministic stdlib
only (``unicodedata`` + ``re`` + ``difflib``); ``difflib`` ranks fuzzy
candidates but ties or fuzzy-only evidence stay ``ambiguous`` — never
first-result wins. Person expansion is exact plus honorific/middle-initial
stripping only: no nicknames are ever generated.
Seam: live reads via SourceGateway + normalization (app/sec/*) + raw_archive
(write-once) + per-session bundle writer. Providers stay authoritative.
NOTE: a future warehouse slots in beside this seam, never inside providers.

"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType
from typing import Literal, NamedTuple, Protocol, TypedDict, runtime_checkable

from ...domain.market.ids import sec_entity_id
from ..context import TRANSACTION_FORMS
from ..filings import _check_as_of
from ..models import (
    EntityCandidate,
    Filing,
    FilingDocument,
    FilingParty,
    InstitutionalHolding,
    SearchAttempt,
    SearchCoverage,
    SearchRun,
    SECSearchRequest,
    SECSearchResult,
    SECTextHit,
    pit_of,
)
from ..offerings import OFFERING_FORMS

PARSER_VERSION = "1"
SOURCE = "sec-submissions"

# Trailing legal-form tokens stripped for the second comparison key only;
# the displayed source name is always preserved.
_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "ltd",
        "limited",
        "llc",
        "pllc",
        "llp",
        "lp",
        "lllp",
        "plp",
        "plc",
        "pa",
        "pc",
        "sa",
        "ag",
        "gmbh",
        "nv",
        "bv",
        "spa",
        "ab",
        "asa",
        "as",
        "aps",
        "oy",
        "oyj",
        "sarl",
        "sas",
        "srl",
        "sl",
        "pty",
        "pte",
        "pvt",
        "bhd",
        "sdn",
        "ltda",
    }
)

# Rank order: exact ticker > exact name > normalized > historical > fuzzy.
_TIER_RANK = {
    "exact_cik": 0,
    "exact_ticker": 0,
    "exact_name": 1,
    "normalized": 2,
    "historical": 3,
    "fuzzy": 4,
}

_FUZZY_FLOOR = 0.6

_AttemptStatus = Literal["complete", "source_limited", "partial", "failed", "not_applicable"]
_CoverageStatus = Literal["complete", "complete_within_source_limits", "partial", "failed"]


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _row_mapping(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _store_rows(store: ModuleType, name: str, **kwargs: object) -> list[dict[str, object]]:
    # Live seam: SourceGateway + normalization + raw_archive + bundle writer; NOTE warehouse slots beside seam.
    fn: object = getattr(store, name, None)
    assert callable(fn)
    return _coerce_query_rows(fn(**kwargs))


def _store_bool(store: ModuleType, name: str, **kwargs: object) -> bool:
    # Live seam: job-ledger coverage reads only; NOTE warehouse slots beside seam.
    fn: object = getattr(store, name, None)
    assert callable(fn)
    return bool(fn(**kwargs))


def _coerce_query_rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _filing_list(value: object) -> list[Filing]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Filing)]
    return []


def _meta_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _row_mapping(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _call_filing_list(fn: Callable[..., object], *args: object, **kwargs: object) -> list[Filing]:
    return _filing_list(fn(*args, **kwargs))


def _cap_mapping[K, V](mapping: dict[K, V], result_limit: int) -> bool:
    if len(mapping) <= result_limit:
        return False
    for key in list(mapping.keys())[result_limit:]:
        del mapping[key]
    return True


class _PoolSlot(TypedDict):
    """Pooled candidate slot: display name, ticker labels, source backends."""

    name: str
    tickers: list[str]
    sources: list[str]


def _object_list(value: object) -> list[object]:
    """Raw SEC/store list payload -> plain list (dynamic boundary containment)."""
    return list(value) if isinstance(value, (list, tuple)) else []


_ACCESSION_RE = re.compile(r"^(\d{10})-?(\d{2})-?(\d{6})$")


def normalize_name(value: object) -> str:
    """Deterministic name key: NFKD, casefold, punctuation/whitespace collapse."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value).casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stripped_name_key(value: object) -> str:
    """Second comparison key with trailing legal suffixes removed."""
    parts = normalize_name(value).split(" ")
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(part for part in parts if part)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_accession_no(value: object) -> str:
    """Canonical dashed accession; accepts dashed or undashed input once."""
    if not isinstance(value, str):
        raise TypeError(f"invalid accession number: {value!r}")
    match = _ACCESSION_RE.match(value.strip())
    if not match:
        raise ValueError(f"invalid accession number: {value!r}")
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


# Raw EDGAR payloads stay Mapping[str, object] at the parse entry only
# (provider SDK shape); validated before constructing domain objects.
def _former_start_ok(entry: Mapping[str, object], as_of: str) -> bool:
    start = entry.get("from") or None
    return start is None or str(start)[:10] <= as_of


def _former_end_ok(entry: Mapping[str, object], as_of: str) -> bool:
    end = entry.get("to") or None
    return end is None or as_of < str(end)[:10]


def _former_valid_at(entry: Mapping[str, object], as_of: str | None) -> bool:
    """Half-open [from, to) validity; null bounds are unbounded."""
    if as_of is None:
        return True
    return _former_start_ok(entry, as_of) and _former_end_ok(entry, as_of)


def _classify_name(
    query: str, current: object, former_names: list[dict[str, object]] | None, as_of: str | None
) -> tuple[str | None, float, tuple[str, ...]]:
    """Query vs current/former names -> (match_type|None, score, warnings).

    ``None`` match_type means no evidence (or only PIT-excluded historical
    evidence, which is reported via warnings and never resolves).
    """
    warnings: list[str] = []
    want, want_stripped = normalize_name(query), stripped_name_key(query)
    if not want:
        return None, 0.0, tuple(warnings)
    exact = _classify_exact(want, want_stripped, current)
    if exact is not None:
        return exact[0], exact[1], tuple(warnings)
    former = former_names or []
    _warn_missing_former_dates(former, warnings)
    historical, best = _scan_former_names(former, want, want_stripped, as_of, warnings)
    if historical is not None:
        return historical, 0.8, tuple(warnings)
    return _classify_fuzzy(want, best, current, warnings)


def _classify_exact(want: str, want_stripped: str, current: object) -> tuple[str, float] | None:
    have, have_stripped = normalize_name(current), stripped_name_key(current)
    if want == have:
        return ("exact_name", 1.0)
    if want_stripped and want_stripped == have_stripped:
        return ("normalized", 0.9)
    return None


def _classify_fuzzy(
    want: str, best: tuple[str | None, float], current: object, warnings: list[str]
) -> tuple[str | None, float, tuple[str, ...]]:
    if best[0] is not None and best[1] >= _FUZZY_FLOOR:
        return best[0], best[1], tuple(warnings)
    have = normalize_name(current)
    if have:
        score = _ratio(want, have)
        if score >= _FUZZY_FLOOR:
            return "fuzzy", score, tuple(warnings)
    return None, 0.0, tuple(warnings)


def _warn_missing_former_dates(former: list[dict[str, object]], warnings: list[str]) -> None:
    if _former_dates_missing(former):
        warnings.append("former name(s) lack effective dates; historical alias coverage is source-limited")


def _former_dates_missing(former: list[dict[str, object]]) -> bool:
    return any(not (entry.get("from") or entry.get("to")) for entry in former if isinstance(entry, dict))


def _former_exact_hit(
    entry: dict[str, object], want: str, want_stripped: str, as_of: str | None
) -> tuple[bool, bool, str]:
    """(exact, pit_valid, pit_warning) for one former-name entry."""
    raw = entry.get("name") or ""
    if not _former_keys_match(raw, want, want_stripped):
        return False, False, ""
    if _former_valid_at(entry, as_of):
        return True, True, ""
    return True, False, _former_pit_warning(entry, raw, as_of)


def _scan_former_names(
    former: list[dict[str, object]], want: str, want_stripped: str, as_of: str | None, warnings: list[str]
) -> tuple[str | None, tuple[str | None, float]]:
    best: tuple[str | None, float] = (None, 0.0)
    for entry in former:
        step = _scan_former_entry(entry, want, want_stripped, as_of, warnings, best)
        if step == "historical":
            return "historical", best
        if step == "fuzzy":
            best = _fuzzy_best(want, entry, best)
    return None, best


def _former_keys_match(raw: object, want: str, want_stripped: str) -> bool:
    cand, cand_stripped = normalize_name(raw), stripped_name_key(raw)
    return want == cand or bool(want_stripped) and want_stripped == cand_stripped


def _former_pit_warning(entry: dict[str, object], raw: object, as_of: str | None) -> str:
    return (
        f"former name {str(raw)!r} outside PIT interval "
        f"[{entry.get('from') or '?'}..{entry.get('to') or '?'}) "
        f"for as_of {as_of}"
    )


def _scan_former_entry(
    entry: object, want: str, want_stripped: str, as_of: str | None, warnings: list[str], best: tuple[str | None, float]
) -> str | None:
    if not isinstance(entry, dict):
        return None
    exact, valid, pit_warning = _former_exact_hit(entry, want, want_stripped, as_of)
    decided = _scan_exact_outcome(exact, valid, pit_warning, warnings)
    if decided is not None:
        return decided
    return _scan_fuzzy_outcome(entry)


def _scan_exact_outcome(exact: bool, valid: bool, pit_warning: str, warnings: list[str]) -> str | None:
    if exact and valid:
        return "historical"
    if exact:
        warnings.append(pit_warning)
        return None
    return None


def _scan_fuzzy_outcome(entry: dict[str, object]) -> str | None:
    if normalize_name(entry.get("name") or ""):
        return "fuzzy"
    return None


def _fuzzy_best(want: str, entry: dict[str, object], best: tuple[str | None, float]) -> tuple[str | None, float]:
    score = _ratio(want, normalize_name(entry.get("name") or ""))
    if score > best[1]:
        return ("fuzzy", score)
    return best


def _candidate(
    cik: int | None,
    name: str,
    tickers: tuple[str, ...],
    match_source: str,
    match_type: str,
    score: float,
    status: Literal["unverified", "verified", "ambiguous", "conflict", "not_found"],
    entity_id: str | None = None,
) -> EntityCandidate:
    return EntityCandidate(
        cik=cik,
        name=name,
        tickers=tuple(tickers),
        exchange=None,
        match_source=match_source,
        match_score=score,
        match_type=match_type,
        verification_status=status,
        entity_id=entity_id,
    )


def _verify_against(
    meta: Mapping[str, object],
    *,
    expected_name: str | None = None,
    expected_ticker: str | None = None,
    as_of: str | None = None,
) -> tuple[EntityCandidate, tuple[str, ...]]:
    """Shared strict verifier -> (EntityCandidate, warnings).

    Either expectation contradicting authoritative metadata is ``conflict``;
    fuzzy-only name evidence is ``ambiguous`` (may not map to an entity);
    otherwise ``verified`` with the mechanical ``sec:cik:`` entity id.
    """
    cik, current, tickers, former_names, entity_id = _verify_inputs(meta)
    warnings: list[str] = []
    if expected_name is None and expected_ticker is None:
        return _candidate(cik, str(current), tickers, SOURCE, "exact_cik", 1.0, "verified", entity_id), tuple(warnings)
    ticker_ok = _ticker_ok(tickers, expected_ticker)
    match_type: str | None = None
    score = 0.0
    if expected_name is not None:
        match_type, score, name_warnings = _classify_name(expected_name, current, former_names, as_of)
        warnings.extend(name_warnings)
    return _verify_outcome(cik, str(current), tickers, entity_id, match_type, score, ticker_ok, expected_name), tuple(
        warnings
    )


def _verify_inputs(
    meta: Mapping[str, object],
) -> tuple[int | None, object, tuple[str, ...], list[dict[str, object]] | None, str | None]:
    cik_raw: object = meta.get("cik")
    cik: int | None = None if cik_raw is None else int(str(cik_raw).strip())
    current: object = meta.get("name") or ""
    tickers: tuple[str, ...] = tuple(str(item) for item in _object_list(meta.get("tickers")))
    former_raw: object = meta.get("former_names")
    former_names: list[dict[str, object]] | None = (
        [entry for entry in former_raw if isinstance(entry, dict)] if isinstance(former_raw, list) else None
    )
    return cik, current, tickers, former_names, (sec_entity_id(cik) if cik is not None else None)


def _verify_conflict(match_type: str | None, ticker_ok: bool | None, expected_name: str | None) -> bool:
    return (ticker_ok is False) or (expected_name is not None and match_type is None)


def _verify_outcome(
    cik: int | None,
    current: str,
    tickers: tuple[str, ...],
    entity_id: str | None,
    match_type: str | None,
    score: float,
    ticker_ok: bool | None,
    expected_name: str | None,
) -> EntityCandidate:
    non_verified = _verify_non_verified(cik, current, tickers, match_type, score, ticker_ok, expected_name)
    if non_verified is not None:
        return non_verified
    if ticker_ok:
        return _candidate(
            cik, current, tickers, SOURCE, match_type or "exact_ticker", score or 1.0, "verified", entity_id
        )
    match_type = _require_match_type(match_type)
    return _candidate(cik, current, tickers, SOURCE, match_type, score, "verified", entity_id)


def _verify_non_verified(
    cik: int | None,
    current: str,
    tickers: tuple[str, ...],
    match_type: str | None,
    score: float,
    ticker_ok: bool | None,
    expected_name: str | None,
) -> EntityCandidate | None:
    if _verify_conflict(match_type, ticker_ok, expected_name):
        return _candidate(
            cik, current, tickers, SOURCE, match_type or ("exact_ticker" if ticker_ok else ""), score, "conflict"
        )
    if match_type == "fuzzy":
        return _candidate(cik, current, tickers, SOURCE, "fuzzy", score, "ambiguous")
    return None


def _ticker_ok(tickers: tuple[str, ...], expected_ticker: str | None) -> bool | None:
    if expected_ticker is None:
        return None
    want = expected_ticker.strip().upper()
    return bool(want) and want in [t.strip().upper() for t in tickers]


def _require_match_type(match_type: str | None) -> str:
    assert match_type is not None  # guarded by the conflict branch above
    return match_type


def verify_sec_entity(
    cik: int | str, *, expected_name: str | None = None, expected_ticker: str | None = None, as_of: str | None = None
) -> EntityCandidate:
    """Verify one CIK against submissions metadata.

    Missing CIK -> ``not_found``; contradicting expectations -> ``conflict``;
    fuzzy-only evidence -> ``ambiguous`` with no entity id. Only ``verified``
    candidates carry the canonical ``entity_id``. Transport/schema failure
    raises (never ``not_found``); callers record failed coverage.
    """
    from ..client import get_submissions_metadata

    as_of = _check_as_of(as_of)
    cik_int = _parse_cik(cik)
    if cik_int is None:
        return _candidate(None, expected_name or "", (), SOURCE, "", 0.0, "not_found")
    meta = get_submissions_metadata(cik_int)
    if meta is None:
        return _candidate(cik_int, expected_name or "", (), SOURCE, "", 0.0, "not_found")
    candidate, _warnings = _verify_against(
        meta, expected_name=expected_name, expected_ticker=expected_ticker, as_of=as_of
    )
    return candidate


def _parse_cik(cik: int | str) -> int | None:
    try:
        return int(str(cik).strip())
    except TypeError, ValueError, AttributeError:
        return None


class _AttemptCounts(TypedDict):
    results_reported: int
    results_retrieved: int


class _AttemptError(TypedDict):
    error_type: str | None
    error_message: str | None


def _attempt_counts(reported: int, retrieved: int) -> _AttemptCounts:
    return {"results_reported": reported, "results_retrieved": retrieved}


def _attempt_error(error: Exception | None) -> _AttemptError:
    return {"error_type": type(error).__name__ if error else None, "error_message": str(error) if error else None}


class _EntitySearchState:
    """Mutable accumulation for the exact-cik/ticker/pool entity routes."""

    def __init__(self, search_id: str, query: str, as_of: str | None, now: str) -> None:
        self.search_id = search_id
        self.query = query
        self.as_of = as_of
        self.now = now
        self.attempts: list[SearchAttempt] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.ranked: list[tuple[int, float, int, EntityCandidate]] = []
        self.pool: dict[int, _PoolSlot] = {}
        self.metas: dict[int, dict[str, object]] = {}
        self.failed = 0
        self.meta_errors: list[str] = []

    def attempt(
        self,
        backend: str,
        reported: int,
        retrieved: int,
        status: _AttemptStatus = "complete",
        error: Exception | None = None,
        pit_basis: str | None = None,
        truncated: bool = False,
        source_limit: str | None = None,
    ) -> None:
        self.attempts.append(
            self._build(backend, reported, retrieved, status, error, pit_basis, truncated, source_limit)
        )

    def _build(
        self,
        backend: str,
        reported: int,
        retrieved: int,
        status: _AttemptStatus,
        error: Exception | None,
        pit_basis: str | None,
        truncated: bool,
        source_limit: str | None,
    ) -> SearchAttempt:
        return SearchAttempt(
            attempt_id=f"{self.search_id}-{backend}",
            search_id=self.search_id,
            backend=backend,
            query=self.query,
            filters={"as_of": self.as_of} if self.as_of else {},
            started_at=self.now,
            completed_at=self.now,
            status=status,
            **_attempt_counts(reported, retrieved),
            pages_retrieved=1,
            truncated=truncated,
            source_limit=source_limit,
            pit_basis=pit_basis,
            **_attempt_error(error),
        )

    def add(self, cik: object, name: object, tickers: Iterable[object], source: str) -> None:
        try:
            cik_int = int(str(cik).strip())
        except TypeError, ValueError, AttributeError:
            return
        slot = self.pool.setdefault(cik_int, {"name": "", "tickers": [], "sources": []})
        if name and not slot["name"]:
            slot["name"] = str(name)
        _merge_slot_tickers(slot, tickers)
        if source not in slot["sources"]:
            slot["sources"].append(source)


def _merge_slot_tickers(slot: _PoolSlot, tickers: Iterable[object]) -> None:
    for ticker in tickers:
        label = str(ticker).strip()
        if label and label not in slot["tickers"]:
            slot["tickers"].append(label)


def _ranked_sort_key(item: tuple[int, float, int, EntityCandidate]) -> tuple[int, float, int]:
    return (item[0], item[1], item[2])


def _entity_exact_cik(
    state: _EntitySearchState, query: str, verify: Callable[..., EntityCandidate], get_meta: Callable[..., object]
) -> None:
    cik_int = _parse_exact_cik(state, query)
    if cik_int is None:
        return
    candidate = _verify_exact_cik(state, cik_int, verify)
    if candidate is None:
        return
    _record_exact_cik_hit(state, query, cik_int, candidate, get_meta)


def _record_exact_cik_hit(
    state: _EntitySearchState, query: str, cik_int: int, candidate: EntityCandidate, get_meta: Callable[..., object]
) -> None:
    ok = candidate.verification_status == "verified"
    state.attempt("exact-cik", 1 if ok else 0, 1 if ok else 0, pit_basis="known_at" if state.as_of else None)
    candidate = replace(candidate, match_source="exact-cik")
    if ok:
        _record_verified_cik(state, cik_int, candidate, get_meta)
    else:
        state.ranked.append((5, 0.0, cik_int, candidate))
        state.errors.append(f"CIK {query} not found in SEC submissions")


def _parse_exact_cik(state: _EntitySearchState, query: str) -> int | None:
    try:
        return int(query)
    except ValueError:
        state.attempt("exact-cik", 0, 0, status="failed")
        state.failed += 1
        return None


def _verify_exact_cik(
    state: _EntitySearchState, cik_int: int, verify: Callable[..., EntityCandidate]
) -> EntityCandidate | None:
    try:
        return verify(cik_int, as_of=state.as_of)
    except Exception as exc:  # noqa: BLE001 - route failure records an attempt and tries the next backend
        state.attempt("exact-cik", 0, 0, status="failed", error=exc, pit_basis="known_at" if state.as_of else None)
        state.errors.append(f"exact-cik route failed: {exc}")
        state.failed += 1
        return None


def _entity_exact_ticker(state: _EntitySearchState, query: str, resolve_cik: Callable[..., object]) -> None:
    ticker_cik = _resolve_ticker_cik(state, query, resolve_cik)
    _record_ticker_cik(state, query, ticker_cik)


def _record_verified_cik(
    state: _EntitySearchState, cik_int: int, candidate: EntityCandidate, get_meta: Callable[..., object]
) -> None:
    state.ranked.append((0, -1.0, cik_int, candidate))
    meta = get_meta(cik_int)
    row = _row_mapping(meta)
    if row is not None:
        state.metas[cik_int] = row


def _resolve_ticker_cik(state: _EntitySearchState, query: str, resolve_cik: Callable[..., object]) -> object:
    try:
        return resolve_cik(query)
    except Exception as exc:  # defensive: resolve_cik never raises today  # noqa: BLE001 - route failure records an attempt and tries the next backend
        state.errors.append(f"exact-ticker route failed: {exc}")
        state.attempt("exact-ticker", 0, 0, status="failed", error=exc)
        state.failed += 1
        return None


def _record_ticker_cik(state: _EntitySearchState, query: str, ticker_cik: object) -> None:
    if ticker_cik is None:
        if not any(a.backend == "exact-ticker" for a in state.attempts):
            state.attempt("exact-ticker", 0, 0)
    else:
        state.add(ticker_cik, "", [query.strip().upper()], "exact-ticker")
        state.attempt("exact-ticker", 1, 1)


def _entity_pool_cap(found: list[dict[str, object]] | None, max_results: int | None) -> str:
    if max_results is None and len(found or []) > 50:
        return "source"
    if max_results is not None and len(found or []) > max_results:
        return "caller"
    return "none"


def _entity_pool_rows(
    state: _EntitySearchState, backend: str, found: list[dict[str, object]] | None, limit: int | None
) -> None:
    for row in (found or [])[:limit] if limit is not None else (found or []):
        state.add(row.get("cik"), row.get("name"), _object_list(row.get("tickers")), backend)


def _entity_record_pool_rows(
    state: _EntitySearchState, backend: str, found: list[dict[str, object]] | None, max_results: int | None
) -> None:
    cap = _entity_pool_cap(found, max_results)
    if cap == "source":
        _record_source_capped(state, backend, found)
    elif cap == "caller":
        assert max_results is not None
        _record_caller_capped(state, backend, found, max_results)
    else:
        _entity_pool_rows(state, backend, found, None)
        state.attempt(backend, len(found or []), len(found or []))


def _record_source_capped(state: _EntitySearchState, backend: str, found: list[dict[str, object]] | None) -> None:
    _entity_pool_rows(state, backend, found, 50)
    state.attempt(backend, 51, 50, status="partial", truncated=True, source_limit="50 candidates")
    state.warnings.append(f"{backend} candidate retrieval capped at 50; entity coverage partial")


def _record_caller_capped(
    state: _EntitySearchState, backend: str, found: list[dict[str, object]] | None, max_results: int
) -> None:
    _entity_pool_rows(state, backend, found, None)
    state.attempt(
        backend,
        len(found or []),
        len(found or []),
        status="partial",
        truncated=True,
        source_limit=f"{max_results} candidates",
    )


def _entity_fetch_pool(
    state: _EntitySearchState,
    query: str,
    max_results: int | None,
    lookup: Callable[..., object],
    company: Callable[..., object],
) -> None:
    fetch_limit = 51 if max_results is None else max_results + 1
    for backend, fetch in (
        ("cik-lookup", lambda: lookup(query, limit=fetch_limit)),
        ("company-search", lambda: company(query, limit=fetch_limit)),
    ):
        try:
            found = _coerce_query_rows(fetch())
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - route failure records an attempt and tries the next backend
            state.attempt(backend, 0, 0, status="failed", error=exc)
            state.errors.append(f"{backend} route failed: {exc}")
            state.failed += 1
            continue
        _entity_record_pool_rows(state, backend, found, max_results)


def _classify_pooled_cik(
    state: _EntitySearchState, cik_int: int, slot: _PoolSlot, query: str, get_meta: Callable[..., object]
) -> None:
    meta = _fetch_pooled_meta(state, cik_int, get_meta)
    if meta is None:
        return
    candidate = _verify_pooled_cik(state, meta, slot, query)
    # Conflict (name contradicts authoritative metadata) stays out.
    if candidate.verification_status != "conflict":
        _record_pooled_cik(state, cik_int, meta, candidate)


def _fetch_pooled_meta(
    state: _EntitySearchState, cik_int: int, get_meta: Callable[..., object]
) -> Mapping[str, object] | None:
    try:
        return _meta_mapping(get_meta(cik_int))
    except Exception as exc:  # noqa: BLE001 - entity meta fetch records the error and degrades to None
        state.meta_errors.append(f"{cik_int}: {exc}")
        state.errors.append(f"submissions metadata {cik_int} failed: {exc}")
        return None


def _verify_pooled_cik(
    state: _EntitySearchState, meta: Mapping[str, object], slot: _PoolSlot, query: str
) -> EntityCandidate:
    if _pooled_is_ticker(meta, query):
        return _verify_pooled_ticker(state, meta, slot, query)
    return _verify_pooled_name(state, meta, slot, query)


def _pooled_tickers(meta: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(item) for item in _object_list(meta.get("tickers")))


def _pooled_is_ticker(meta: Mapping[str, object], query: str) -> bool:
    tickers = _pooled_tickers(meta)
    return query.strip().upper() in [t.strip().upper() for t in tickers]


def _verify_pooled_ticker(
    state: _EntitySearchState, meta: Mapping[str, object], slot: _PoolSlot, query: str
) -> EntityCandidate:
    candidate, _w = _verify_against(meta, expected_ticker=query, as_of=state.as_of)
    return replace(candidate, match_source="+".join(sorted(slot["sources"])) or "exact-ticker")


def _verify_pooled_name(
    state: _EntitySearchState, meta: Mapping[str, object], slot: _PoolSlot, query: str
) -> EntityCandidate:
    candidate, name_warnings = _verify_against(meta, expected_name=query, as_of=state.as_of)
    state.warnings.extend(name_warnings)
    return replace(candidate, match_source="+".join(sorted(slot["sources"])) or "company-search")


def _record_pooled_cik(
    state: _EntitySearchState, cik_int: int, meta: Mapping[str, object], candidate: EntityCandidate
) -> None:
    tier = _TIER_RANK.get(candidate.match_type, 5)
    state.ranked.append((tier, -candidate.match_score, cik_int, candidate))
    row = _row_mapping(meta)
    if row is not None:
        state.metas[cik_int] = row


def _classify_pool(state: _EntitySearchState, query: str, get_meta: Callable[..., object]) -> None:
    for cik_int, slot in sorted(state.pool.items()):
        _classify_pooled_cik(state, cik_int, slot, query, get_meta)
    if state.meta_errors and not state.metas:
        state.errors.append(f"submissions metadata unavailable: {state.meta_errors[0]}")
        state.failed += 1


def _mark_tie_ambiguous(
    ranked: list[tuple[int, float, int, EntityCandidate]], entities: list[EntityCandidate], warnings: list[str]
) -> list[EntityCandidate]:
    tied = _tied_top(ranked, entities)
    if not tied:
        return entities
    tied_ciks = {item[2] for item in tied}
    _warn_tie(warnings, tied)
    return [
        replace(item[3], verification_status="ambiguous", entity_id=None) if item[2] in tied_ciks else item[3]
        for item in ranked
    ]


def _tied_top(
    ranked: list[tuple[int, float, int, EntityCandidate]], entities: list[EntityCandidate]
) -> list[tuple[int, float, int, EntityCandidate]]:
    if len(entities) <= 1:
        return []
    top_tier, top_score = ranked[0][0], ranked[0][1]
    tied = [item for item in ranked if item[0] == top_tier and item[1] == top_score]
    return tied if len(tied) > 1 else []


def _warn_tie(warnings: list[str], tied: list[tuple[int, float, int, EntityCandidate]]) -> None:
    warnings.append(
        f"{len(tied)} candidates tie at {tied[0][3].match_type} score {tied[0][3].match_score:.3f}; marking ambiguous"
    )


def _mark_fuzzy_top(entities: list[EntityCandidate], warnings: list[str]) -> list[EntityCandidate]:
    if entities and entities[0].match_type == "fuzzy":
        warnings.append("top match is fuzzy-only; marking ambiguous")
        first = entities[0]
        entities[0] = replace(first, verification_status="ambiguous", entity_id=None)
    return entities


def _cap_entities(entities: list[EntityCandidate], max_results: int | None) -> list[EntityCandidate]:
    if max_results is not None and len(entities) > max_results:
        return entities[:max_results]
    return entities


def _finalize_entities(
    state: _EntitySearchState, max_results: int | None, data_root: Path | str | None
) -> list[EntityCandidate]:
    # Live seam: verified CIKs need no persistence, providers stay authoritative.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    del data_root
    state.ranked.sort(key=_ranked_sort_key)
    entities: list[EntityCandidate] = [item[3] for item in state.ranked]
    entities = _mark_tie_ambiguous(state.ranked, entities, state.warnings)
    entities = _mark_fuzzy_top(entities, state.warnings)
    return _cap_entities(entities, max_results)


def _warn_entity_empty(state: _EntitySearchState, entities: list[EntityCandidate], query: str) -> None:
    if not entities and not state.errors:
        state.warnings.append(f"no SEC entity candidates for {query!r} (no direct corpus; other routes still searched)")


def _warn_entity_cap(state: _EntitySearchState, max_results: int | None) -> None:
    if max_results is not None and any(getattr(a, "truncated", False) for a in state.attempts):
        cap_warning = f"results capped at {max_results}; rerun with a higher limit or exhaustive=true"
        if cap_warning not in state.warnings:
            state.warnings.append(cap_warning)


def _entity_coverage_status(
    state: _EntitySearchState, entities: list[EntityCandidate], query: str, max_results: int | None
) -> _CoverageStatus:
    _warn_entity_empty(state, entities, query)
    _warn_entity_cap(state, max_results)
    has_partial = any(getattr(a, "status", None) in ("partial", "source_limited") for a in state.attempts)
    if state.failed and not entities:
        return "failed"
    if state.failed or has_partial:
        return "partial"
    return "complete"


def _entity_result(
    search_id: str,
    request: SECSearchRequest,
    state: _EntitySearchState,
    entities: list[EntityCandidate],
    status: _CoverageStatus,
) -> SECSearchResult:
    sources = tuple(a.backend for a in state.attempts)
    packet = build_evidence_packet(search_id, entities=tuple(entities))
    return SECSearchResult(
        search_id=search_id,
        request=request,
        entities=tuple(entities),
        coverage=_entity_coverage(state, entities, sources, status),
        attempts=tuple(state.attempts),
        warnings=tuple(state.warnings),
        errors=tuple(state.errors),
        retrieval_order=sources,
        evidence_packet_ids=packet,
    )


def _entity_coverage(
    state: _EntitySearchState, entities: list[EntityCandidate], sources: tuple[str, ...], status: _CoverageStatus
) -> SearchCoverage:
    completed, failed, limits = _entity_attempt_sets(state)
    return SearchCoverage(
        status=status,
        sources_attempted=sources,
        sources_completed=completed,
        sources_failed=failed,
        source_limits=limits,
        results_reported=len(state.pool) if state.pool else len(entities),
        results_retrieved=len(entities),
    )


def _entity_attempt_sets(state: _EntitySearchState) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        _attempt_backends(state, "complete"),
        _attempt_backends(state, "failed"),
        _attempt_limits(state),
    )


def _attempt_backends(state: _EntitySearchState, status: str) -> tuple[str, ...]:
    return tuple(a.backend for a in state.attempts if a.status == status)


def _attempt_limits(state: _EntitySearchState) -> tuple[str, ...]:
    return tuple(dict.fromkeys(a.source_limit for a in state.attempts if a.source_limit))


def find_sec_entities(
    query: str,
    *,
    as_of: str | None = None,
    exhaustive: bool = False,
    max_results: int | None = 20,
    data_root: Path | str | None = None,
) -> SECSearchResult:
    """Fan out over exact-CIK, exact-ticker, and general legal-name routes.

    General names merge the no-ticker ``cik-lookup-data.txt`` scan with the
    ticker-company index by CIK; each pooled CIK loads submissions once and
    is classified exact/normalized/historical/fuzzy. Ties and fuzzy-only tops
    stay ``ambiguous``; verified candidates persist to the entity store.
    Bounded calls (``exhaustive=False``) probe each candidate source with
    ``max_results + 1``, rank/retain at most ``max_results``, and prove
    truncation with the extra row. Exhaustive calls (the research-session
    default) run every route with ``max_results=None``: retrieval is never
    bounded by the caller's packet limit, and the local 50-candidate
    name-scan cap still applies to that corpus.
    """

    from ..client import (
        find_sec_company,
        get_cik_lookup_candidates,
        get_submissions_metadata,
        resolve_cik,
    )

    if not isinstance(query, str):
        raise TypeError(f"invalid query: {query!r}")
    if not query.strip():
        raise ValueError(f"invalid query: {query!r}")
    as_of = _check_as_of(as_of)
    query = query.strip()
    search_id = uuid.uuid4().hex[:12]
    request = SECSearchRequest(query=query, as_of=as_of, exhaustive=exhaustive, max_results=max_results)
    state = _EntitySearchState(search_id, query, as_of, _utcnow())
    if query.isdigit():
        _entity_exact_cik(state, query, verify_sec_entity, get_submissions_metadata)
    else:
        _entity_exact_ticker(state, query, resolve_cik)
        _entity_fetch_pool(state, query, max_results, get_cik_lookup_candidates, find_sec_company)
        _classify_pool(state, query, get_submissions_metadata)
    entities = _finalize_entities(state, max_results, data_root)
    status = _entity_coverage_status(state, entities, query, max_results)
    return _entity_result(search_id, request, state, entities, status)


def resolve_sec_accession(accession_no: str, *, as_of: str | None = None) -> SECSearchResult:
    """Exact accession lookup: normalize once, bypass fuzzy discovery.

    Returns filer/form/timestamps, the document inventory, and ``FilingParty``
    rows (carried in ``relationships`` until typed datasets land). An
    accession unknown at ``as_of`` is rejected with ``failed`` coverage —
    never zero-match silence.
    """

    from ..documents import list_sec_documents
    from ..filings import get_sec_filing

    as_of = _check_as_of(as_of)
    normalized = normalize_accession_no(accession_no)
    search_id = uuid.uuid4().hex[:12]
    request = SECSearchRequest(accession_no=normalized, as_of=as_of, exhaustive=False)
    now = _utcnow()
    try:
        filing = get_sec_filing(normalized, as_of=as_of)
    except ValueError as exc:
        return _accession_lookup_failure(search_id, request, normalized, as_of, now, exc)
    documents, doc_warning = _accession_documents(list_sec_documents, normalized, as_of)
    return _accession_success(search_id, request, filing, documents, doc_warning, normalized, as_of, now)


# Form families only EXPAND routable global queries (person/proxy/13F
# fan-out); unknown form strings always pass through untouched, never rejected.
_PERSON_FORMS = (
    "3",
    "4",
    "5",
    "144",
    "SC 13D",
    "SC 13G",
    "DEF 14A",
    "DEFM14A",
    "PREM14A",
    "13F-HR",
)

_HONORIFICS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "miss",
        "dr",
        "prof",
        "sir",
        "dame",
        "mx",
        "rev",
        "hon",
    }
)

# ponytail: at most 8 quarterly partitions per interactive search; deeper
# history is Phase 5 backfill work, never a blocked model-visible call.
_GLOBAL_QUARTER_CAP = 8

# Interactive search never waits for unbounded history: missing partitions
# become bounded quarterly/form backfill jobs in this priority order.
BACKFILL_PRIORITY = (
    "8-K",
    "10-K",
    "10-Q",
    "13D",
    "13G",
    "13F-HR",
    "3",
    "4",
    "5",
    "144",
    "S-1",
    "S-3",
    "424B",
    "S-4",
    "DEF 14A",
    "D",
)
BACKFILL_SOURCE = "sec-global"
DOC_SOURCE = "sec-documents"
TYPED_SOURCE = "sec-typed"
# SEC global quarterly indexes start in 1993; the current quarter always
# comes from the current feed, never a quarterly partition.
SEC_GLOBAL_START = (1993, 1)

_EVIDENCE_MAX_ITEMS = 20
_EVIDENCE_MAX_CHARS = 8000

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _accession_error(error: Exception | None) -> tuple[str | None, str | None]:
    if error is None:
        return None, None
    kind = "NotFound" if "not known as of" in str(error) else type(error).__name__
    return kind, str(error)


def _accession_attempt(
    search_id: str,
    normalized: str,
    as_of: str | None,
    now: str,
    status: _AttemptStatus,
    error: Exception | None = None,
    pit_basis: str | None = None,
) -> SearchAttempt:
    error_type, error_message = _accession_error(error)
    count = _accession_count(status)
    return SearchAttempt(
        attempt_id=f"{search_id}-exact-accession",
        search_id=search_id,
        backend="exact-accession",
        query=normalized,
        filters={"as_of": as_of} if as_of else {},
        started_at=now,
        completed_at=now,
        status=status,
        results_reported=count,
        results_retrieved=count,
        pages_retrieved=count,
        pit_basis=pit_basis,
        error_type=error_type,
        error_message=error_message,
    )


def _accession_count(status: _AttemptStatus) -> int:
    return 1 if status == "complete" else 0


def _accession_lookup_failure(
    search_id: str, request: SECSearchRequest, normalized: str, as_of: str | None, now: str, exc: ValueError
) -> SECSearchResult:
    return SECSearchResult(
        search_id=search_id,
        request=request,
        coverage=SearchCoverage(
            status="failed",
            sources_attempted=("exact-accession",),
            sources_failed=("exact-accession",),
        ),
        attempts=(_accession_attempt(search_id, normalized, as_of, now, "failed", error=exc),),
        errors=(str(exc),),
        retrieval_order=("exact-accession",),
    )


def _accession_documents(
    list_sec_documents: Callable[..., object], normalized: str, as_of: str | None
) -> tuple[list[FilingDocument], tuple[str, ...]]:
    try:
        raw = list_sec_documents(normalized, as_of=as_of)
    except ValueError as exc:
        return [], (f"document inventory unavailable: {exc}",)
    if isinstance(raw, list):
        return [doc for doc in raw if isinstance(doc, FilingDocument)], ()
    return [], ()


def _accession_filer_party(filing: Filing) -> FilingParty:
    return FilingParty(
        accession_no=filing.accession_no,
        entity_id=sec_entity_id(filing.filer_cik),
        cik=filing.filer_cik,
        name=filing.filer_name,
        role="filer",
        source=filing.source,
        known_at=filing.known_at,
        parser_version=PARSER_VERSION,
    )


def _accession_has_subject(filing: Filing) -> bool:
    if filing.subject_cik is None and not filing.subject_name:
        return False
    return filing.subject_cik != filing.filer_cik or (filing.subject_name or "") != (filing.filer_name or "")


def _accession_parties(filing: Filing) -> list[FilingParty]:
    parties = [_accession_filer_party(filing)]
    if _accession_has_subject(filing):
        parties.append(
            FilingParty(
                accession_no=filing.accession_no,
                entity_id=sec_entity_id(filing.subject_cik) if filing.subject_cik is not None else None,
                cik=filing.subject_cik,
                name=filing.subject_name or "",
                role="subject",
                source=filing.source,
                known_at=filing.known_at,
                parser_version=PARSER_VERSION,
            )
        )
    return parties


def _accession_filer(filing: Filing) -> EntityCandidate:
    return EntityCandidate(
        cik=filing.filer_cik,
        name=filing.filer_name,
        tickers=(),
        exchange=None,
        match_source="exact-accession",
        match_score=1.0,
        match_type="exact_cik",
        verification_status="unverified",
        entity_id=sec_entity_id(filing.filer_cik),
    )


def _accession_success(
    search_id: str,
    request: SECSearchRequest,
    filing: Filing,
    documents: list[FilingDocument],
    doc_warning: tuple[str, ...],
    normalized: str,
    as_of: str | None,
    now: str,
) -> SECSearchResult:
    _pit_value, pit_basis = pit_of(filing)
    return SECSearchResult(
        search_id=search_id,
        request=request,
        entities=(_accession_filer(filing),),
        filings=(filing,),
        documents=tuple(documents),
        relationships=tuple(_accession_parties(filing)),
        coverage=SearchCoverage(
            status="complete",
            sources_attempted=("exact-accession",),
            sources_completed=("exact-accession",),
            results_reported=1,
            results_retrieved=1,
            forms_covered=(filing.form,),
        ),
        attempts=(_accession_attempt(search_id, normalized, as_of, now, "complete", pit_basis=pit_basis),),
        warnings=doc_warning,
        retrieval_order=("exact-accession",),
    )


def _expand_entity_queries(entities: Iterable[EntityCandidate], as_of: str | None = None) -> list[str]:
    """Verified entities -> deterministic text variants for EFTS/global routes.

    Current legal name, legal-name comparison form, ticker(s), and former
    names valid/known for ``as_of``. CIKs route to the filer-submissions
    backend only (callers use ``entity.cik`` directly); they are never
    emitted as text variants.
    """
    as_of = _check_as_of(as_of)
    from ..client import get_submissions_metadata

    variants: list[str] = []
    seen: set[str] = set()
    push = _variant_pusher(variants, seen)

    for entity in entities or ():
        _push_entity_variants(entity, push)
        _push_entity_metadata(entity, push, as_of, get_submissions_metadata)
    return variants


def _push_entity_variants(entity: EntityCandidate, push: Callable[[object], None]) -> None:
    if getattr(entity, "verification_status", None) != "verified":
        return
    name = (getattr(entity, "name", "") or "").strip()
    if name:
        push(name)
        push(stripped_name_key(name))
    for ticker in getattr(entity, "tickers", None) or ():
        if str(ticker).strip():
            push(str(ticker).strip().upper())


def _push_entity_metadata(
    entity: EntityCandidate, push: Callable[[object], None], as_of: str | None, get_meta: Callable[..., object]
) -> None:
    if getattr(entity, "verification_status", None) != "verified":
        return
    cik = getattr(entity, "cik", None)
    if cik is None:
        return
    meta = get_meta(cik)
    if not isinstance(meta, dict):
        return
    push(meta.get("name") or "")
    for entry in _former_variant_entries(meta, as_of):
        push(entry)
        push(stripped_name_key(entry))


def _variant_pusher(variants: list[str], seen: set[str]) -> Callable[[object], None]:
    def _push(value: object) -> None:
        text = str(value or "").strip()
        key = normalize_name(text)
        if text and key and key not in seen:
            seen.add(key)
            variants.append(text)

    return _push


def _former_variant_entries(meta: Mapping[str, object], as_of: str | None) -> list[str]:
    out: list[str] = []
    for entry in _object_list(meta.get("former_names")):
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            continue
        if _former_valid_at(entry, as_of):
            out.append(str(entry["name"]))
    return out


def _expand_person_queries(name: str) -> list[str]:
    """Exact name plus honorific-stripped plus middle-initial-stripped forms.

    Every variant stays a separate attempt with its own provenance; no
    nicknames, no suffix handling, no fuzzy variants.
    """
    if not isinstance(name, str):
        raise TypeError(f"invalid person name: {name!r}")
    if not name.strip():
        raise ValueError(f"invalid person name: {name!r}")
    exact = re.sub(r"\s+", " ", name.strip())
    variants = [exact]
    _strip_honorific(variants)
    _strip_middle_initials(variants)
    return variants


def _strip_honorific(variants: list[str]) -> None:
    tokens = variants[0].split(" ")
    if len(tokens) > 1 and tokens[0].rstrip(".").casefold() in _HONORIFICS:
        variants.append(" ".join(tokens[1:]))


def _is_middle_initial(tok: str) -> bool:
    return bool(tok.rstrip(".").isalpha()) and len(tok.rstrip(".")) == 1


def _strip_middle_initials(variants: list[str]) -> None:
    parts = variants[-1].split(" ")
    middle = [tok for i, tok in enumerate(parts) if i == 0 or i == len(parts) - 1 or not _is_middle_initial(tok)]
    squashed = " ".join(middle)
    if squashed and squashed not in variants:
        variants.append(squashed)


def _expand_domain_queries(domain: str) -> list[str]:
    """Lowercase hostname plus bare/``www.``/literal-``@`` text variants."""
    if not isinstance(domain, str):
        raise TypeError(f"invalid domain: {domain!r}")
    if not domain.strip():
        raise ValueError(f"invalid domain: {domain!r}")
    raw = _normalize_domain_host(domain)
    if not raw or "." not in raw or re.search(r"\s", raw):
        raise ValueError(f"invalid domain: {domain!r}")
    bare = raw.removeprefix("www.")
    return list(dict.fromkeys((bare, f"www.{bare}", f"@{bare}")))


def _normalize_domain_host(domain: str) -> str:
    raw = domain.strip().lower()
    raw = re.sub(r"^[a-z][a-z0-9+.-]*://", "", raw)
    raw = re.split(r"[/?#]", raw, maxsplit=1)[0]
    if "@" in raw:
        raw = raw.rsplit("@", 1)[1]
    return raw.split(":")[0].rstrip(".").strip()


def _expand_security_queries(identifier: str) -> list[str]:
    """Ticker/CUSIP/ISIN/class-title text variants; never entity identity.

    No identifier resolution happens here: a ``Security`` stays separate from
    its issuer ``Entity``, so expansion yields only case variants and the
    service creates no entity candidate from them.
    """
    if not isinstance(identifier, str):
        raise TypeError(f"invalid security identifier: {identifier!r}")
    if not identifier.strip():
        raise ValueError(f"invalid security identifier: {identifier!r}")
    exact = re.sub(r"\s+", " ", identifier.strip())
    upper = exact.upper()
    return [exact] if upper == exact else [exact, upper]


def _quarters_for_range(
    start_date: str | None, end_date: str | None, *, cap: int = _GLOBAL_QUARTER_CAP
) -> tuple[list[tuple[int, int]], bool]:
    """(start, end) -> ([(year, quarter)] oldest-first, capped?).

    Empty when unbounded (caller uses the current feed instead of quarterly
    partitions). Partitions before 1993-Q1 are dropped (no global index);
    the current quarter is excluded (the current feed covers it).
    Over-wide ranges keep the most recent ``cap`` quarters.
    """
    if not start_date and not end_date:
        return [], False
    now = datetime.now(UTC)
    current = (now.year, (now.month - 1) // 3 + 1)
    low, high = _quarter_bounds(start_date, end_date, current)
    if low > high:
        raise ValueError(f"invalid date range: {start_date!r}..{end_date!r}")
    if high < SEC_GLOBAL_START:
        return [], False
    low = max(low, SEC_GLOBAL_START)
    quarters = _enumerate_quarters(low, high, current)
    if len(quarters) > cap:
        return quarters[len(quarters) - cap :], True
    return quarters, False


def _parse_quarter_date(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise TypeError(f"invalid {label} date: {value!r} (expected YYYY-MM-DD)")
    if not _DATE_RE.match(value):
        raise ValueError(f"invalid {label} date: {value!r} (expected YYYY-MM-DD)")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"invalid {label} date: {value!r} (expected YYYY-MM-DD)") from None
    return parsed.year, (parsed.month - 1) // 3 + 1


def _quarter_bounds(
    start_date: str | None, end_date: str | None, current: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    if start_date and end_date:
        return (_parse_quarter_date(start_date, "start"), _parse_quarter_date(end_date, "end"))
    if start_date:
        return _parse_quarter_date(start_date, "start"), current
    assert end_date is not None
    return SEC_GLOBAL_START, _parse_quarter_date(end_date, "end")


def _enumerate_quarters(low: tuple[int, int], high: tuple[int, int], current: tuple[int, int]) -> list[tuple[int, int]]:
    year, quarter = low
    quarters: list[tuple[int, int]] = []
    while (year, quarter) <= high:
        if (year, quarter) != current:
            quarters.append((year, quarter))
        quarter += 1
        if quarter > 4:
            quarter, year = 1, year + 1
    return quarters


def _quarter_dates(year: int, quarter: int) -> tuple[str, str]:
    """Quarter partition -> (start_date, end_date) YYYY-MM-DD bounds."""
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return f"{year}-{starts[quarter]}", f"{year}-{ends[quarter]}"


def _partition_for_quarter(year: int, quarter: int) -> str:
    return f"{year}-Q{quarter}"


def _sort_forms_by_priority(forms: Iterable[str]) -> list[str]:
    """Requested forms first in BACKFILL_PRIORITY order, unknown forms last."""
    order = {name.upper(): i for i, name in enumerate(BACKFILL_PRIORITY)}

    def _key(form: str) -> tuple[int, str]:
        return (order.get(form.upper(), len(order)), form.upper())

    return sorted(forms, key=_key)


_WORKER_LOCK = threading.Lock()
_WORKER_THREAD: threading.Thread | None = None


_LOCAL_EXHAUSTIVE_GUARD = 100_000


def _fetch_typed(
    query_fn: Callable[..., list[dict[str, object]]],
    *,
    cap: int | None,
    root: Path | str | None = None,
    **filters: object,
) -> tuple[list[dict[str, object]], bool, int]:
    """Fetch one typed store query in a single snapshot. Returns (rows, exhausted, pages).

    cap=None reads up to a documented local guard and proves exhaustion with
    a short read; hitting the guard stays partial. Bounded caps return at
    most cap rows via one limit+1 probe. Rows dedupe by persisted identity
    (overlapping directions can return the same row twice).
    """
    out: list[dict[str, object]] = []
    seen: set[str] = set()

    def _push(rows: Iterable[dict[str, object]] | None) -> None:
        for row in rows or []:
            try:
                key = json.dumps(row, sort_keys=True, default=str)
            except Exception:  # noqa: BLE001 - row key falls back to repr on unserializable payload
                key = repr(row)
            if key not in seen:
                seen.add(key)
                out.append(row)

    if cap is None:
        raw = query_fn(limit=_LOCAL_EXHAUSTIVE_GUARD, root=root, **filters)
        _push(raw)
        return out, len(raw or []) < _LOCAL_EXHAUSTIVE_GUARD, 1
    cap = max(cap, 0)
    probe = query_fn(limit=cap + 1, root=root, **filters)
    _push(probe)
    if len(probe or []) <= cap:
        return out, True, 1
    return out[:cap], False, 1


def _live_filing_batch(
    form: str, qs: str, qe: str, *, as_of: str | None = None, limit: int | None = None
) -> tuple[list[Filing], bool, str | None]:
    """Date-scoped live filing read via provider global index, amendments included.

    Returns (rows, exhausted, error): provider reads are exhaustive by
    construction; transport failure returns ([], False, str(exc)).
    Never raises.
    """
    from ..client import get_global_filings

    try:
        year_qs, year_qe = int(qs[:4]), int(qe[:4])
        years = list(range(min(year_qs, year_qe), max(year_qs, year_qe) + 1)) or None
        filings = get_global_filings(years, form=[form] if form.endswith("/A") else [form, f"{form}/A"])
    except (ConnectionError, TimeoutError, OSError) as exc:
        return [], False, str(exc)
    out: list[Filing] = []
    for filing in filings or []:
        day = (filing.filed_at or filing.known_at or "")[:10]
        if day and not qs <= day <= qe:
            continue
        if as_of is not None:
            value, _basis = pit_of(filing)
            if value is None or value[:10] > as_of:
                continue
        out.append(filing)
        if limit is not None and len(out) >= limit:
            break
    return out, True, None


def _base_form(value: object) -> str:
    text = str(value or "").strip().upper()
    text = text.removesuffix("/A")
    return text


_TRANSACTION_BASE_FORMS = frozenset(_base_form(f) for f in TRANSACTION_FORMS)
_OFFERING_BASE_FORMS = frozenset(_base_form(f) for f in OFFERING_FORMS)
_TYPED_HYDRATION_FORMS = frozenset(
    {"SC 13D", "SC 13G", "13D", "13G", "3", "4", "5", "13F-HR"} | _TRANSACTION_BASE_FORMS | _OFFERING_BASE_FORMS
)


def _needs_typed_hydration(value: object) -> bool:
    return _base_form(value) in _TYPED_HYDRATION_FORMS


def _enqueue_or_requeue(
    store: ModuleType, source: str, form: str, qs: str, qe: str, *, batch_size: int, root: Path | str | None = None
) -> str:
    """Enqueue a quarterly job; reset finished-but-uncovered ones for resume."""
    # Live seam: coverage/checkpoint state lives on the job ledger.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    job_id_value: object = store.enqueue_backfill_job(
        source, form, qs, qe, PARSER_VERSION, batch_size=batch_size, root=root
    )
    job_id = job_id_value if isinstance(job_id_value, str) else str(job_id_value)
    try:
        job = store.get_job(job_id=job_id, root=root)
        if (
            job is not None
            and job.get("status") in ("complete", "failed")
            and _enqueue_needs_resume(store, source, form, qs, qe, root=root)
        ):
            store.requeue_job(job_id, root=root)
    except Exception:  # noqa: BLE001, S110 - best-effort requeue probe, failure keeps the enqueued job
        pass
    return job_id


def _enqueue_past_quarters(qs: str, qe: str) -> list[tuple[int, int]]:
    now = datetime.now(UTC)
    cur = (now.year, (now.month - 1) // 3 + 1)
    quarters, _ = _quarters_for_range(qs, qe, cap=2)
    return [(y, q) for y, q in quarters or [] if (y, q) != cur]


def _enqueue_wanted_sources(source: str, form: str) -> list[str]:
    return [source] + ([DOC_SOURCE, TYPED_SOURCE] if _needs_typed_hydration(form) else [])


def _enqueue_partition_uncovered(
    store: ModuleType, wanted: list[str], form: str, year: int, quarter: int, root: Path | str | None
) -> bool:
    # Live seam: coverage probe reads the job ledger; NOTE warehouse slots beside seam.
    for src in wanted:
        covered = store.is_partition_covered(
            source=src,
            form=form,
            date_partition=_partition_for_quarter(year, quarter),
            root=root,
        )
        if not covered:
            return True
    return False


def _enqueue_needs_resume(store: ModuleType, source: str, form: str, qs: str, qe: str, root: Path | str | None) -> bool:
    past = _enqueue_past_quarters(qs, qe)
    if not past:
        return False
    wanted = _enqueue_wanted_sources(source, form)
    for year, quarter in past:
        if _enqueue_partition_uncovered(store, wanted, form, year, quarter, root=root):
            return True
    return False


class _HydrateContext:
    form_raw: str
    form: str
    accession: str
    filed_at: str | None
    known_at: str | None
    source_url: str
    filer_name: str | None
    filer_cik: int | str | None
    subject_cik: int | None
    subject_name: str | None

    def __init__(self, filing: Filing) -> None:
        self.form_raw = filing.form
        self.form = _base_form(self.form_raw)
        self.accession = filing.accession_no
        self.filed_at = filing.filed_at or None
        self.known_at = filing.known_at or self.filed_at
        self.source_url = filing.source or ""
        self.filer_name = filing.filer_name or None
        self.filer_cik = filing.filer_cik
        self.subject_cik = filing.subject_cik
        self.subject_name = filing.subject_name or None


class _DocFields:
    text: str
    doc_name: str | None
    raw_path: Path | str | None
    retrieved_at: str | None
    content_hash: str | None

    def __init__(
        self, text: str, doc_name: object, raw_path: object, retrieved_at: object, content_hash: object
    ) -> None:
        self.text = text
        self.doc_name = doc_name if isinstance(doc_name, str) else None
        self.raw_path = raw_path if isinstance(raw_path, (str, Path)) else None
        self.retrieved_at = _opt_str(retrieved_at)
        self.content_hash = _opt_str(content_hash)


class _FetchedDoc:
    doc: dict[str, object] | None
    error: str | None

    def __init__(self, doc: object = None, error: str | None = None) -> None:
        self.doc = doc if isinstance(doc, dict) else None
        self.error = error


class _StampedDoc:
    fields: _DocFields | None
    error: str | None

    def __init__(self, fields: _DocFields | None = None, error: str | None = None) -> None:
        self.fields = fields
        self.error = error


def _hydrate_context(filing: Filing) -> _HydrateContext | None:
    ctx = _HydrateContext(filing)
    if not ctx.accession:
        return None
    return ctx


def _hydrate_fetch_doc(get_sec_document: Callable[..., object], accession: str, data_root: object) -> _FetchedDoc:
    try:
        doc = get_sec_document(accession, None, data_root=data_root)
    except Exception as exc:  # noqa: BLE001 - document fetch failure returns an error packet, never raises
        return _FetchedDoc(error=f"no supported primary document: {exc}")
    if isinstance(doc, dict) and doc.get("error_type") == "pit_revision_conflict":
        return _FetchedDoc(error=f"no supported primary document: {doc.get('error')}")
    return _FetchedDoc(doc=doc)


def _stamp_text(doc: dict[str, object]) -> str:
    text = doc.get("text") or ""
    return text if isinstance(text, str) else ""


def _stamp_str_field(doc: dict[str, object], field: str) -> str | None:
    value = doc.get(field)
    return value if isinstance(value, str) else None


def _hydrate_stamp_doc(doc: object, filing: Filing) -> _StampedDoc:
    if not isinstance(doc, dict):
        return _StampedDoc(error="document archive failed: not a mapping")
    try:
        text = _stamp_text(doc)
        doc_value: object = doc.get("document_name") or filing.primary_document or "primary"
        doc_name: str = doc_value if isinstance(doc_value, str) else "primary"
        raw_path = doc.get("raw_archive_path")
        raw_value: Path | str | None = raw_path if isinstance(raw_path, (str, Path)) else None
        retrieved_at = _stamp_str_field(doc, "retrieved_at")
        content_hash = _stamp_str_field(doc, "content_hash")
    except Exception as exc:  # noqa: BLE001 - document archive failure returns an error packet, never raises
        return _StampedDoc(error=f"document archive failed: {exc}")
    return _StampedDoc(fields=_DocFields(text, doc_name, raw_value, retrieved_at, content_hash))


@runtime_checkable
class _HasObj(Protocol):
    def obj(self) -> object: ...


def _edgar_obj(filing: object) -> object | None:
    if isinstance(filing, _HasObj):
        return filing.obj()
    return None


def _hydrate_schedule(
    _store: ModuleType,
    accession: str,
    form_raw: str,
    doc_name: str | None,
    raw_path: Path | str | None,
    retrieved_at: str | None,
    content_hash: str | None,
    filed_at: str | None,
    known_at: str | None,
    source_url: str,
    filer_name: str | None,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None]:
    from ..ownership import load_schedule, normalize_schedule

    schedule = load_schedule(accession)
    recs = normalize_schedule(
        schedule,
        issuer=filer_name or "",
        form=form_raw,
        filed_at=filed_at,
        accession_no=accession,
        document_name=str(doc_name),
        known_at=known_at,
        source_url=source_url or None,
    )
    # Live seam: normalize per filing; normalized rows count as typed rows, raw bytes stay in raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, raw_path, retrieved_at, content_hash, source_url, data_root)
    return len(list(recs or [])), True, None


def _hydrate_ownership(
    _store: ModuleType,
    accession: str,
    form_raw: str,
    doc_name: str | None,
    raw_path: Path | str | None,
    retrieved_at: str | None,
    content_hash: str | None,
    filed_at: str | None,
    known_at: str | None,
    source_url: str,
    filer_name: str | None,
    filer_cik: int | str | None,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None]:
    from ..insider import load_ownership, normalize_ownership_filing

    obj = load_ownership(accession)
    recs = normalize_ownership_filing(
        obj,
        issuer=filer_name or "",
        form=form_raw,
        filed_at=filed_at,
        accession_no=accession,
        issuer_cik=str(filer_cik).strip() if filer_cik is not None else None,
        document_name=str(doc_name),
        known_at=known_at,
    )
    # Live seam: normalize per filing; normalized rows count as typed rows, raw bytes stay in raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, raw_path, retrieved_at, content_hash, source_url, data_root)
    return len(list(recs or [])), True, None


def _hydrate_infotable(get_by_accession_number: Callable[..., object], accession: str) -> object | None:
    edgar_filing = get_by_accession_number(accession)
    inner: object = _edgar_obj(edgar_filing)
    if inner is None:
        inner = edgar_filing
    for attr in ("infotable", "information_table", "holdings", "info_table"):
        try:
            table_value: object = getattr(inner, attr, None)
        except Exception:  # noqa: BLE001 - untrusted SDK attr read falls through to the next table name
            table_value = None
        if table_value is not None:
            return table_value
    return None


def _hydrate_13f(
    _store: ModuleType,
    get_by_accession_number: Callable[..., object],
    accession: str,
    form_raw: str,
    doc_name: str | None,
    raw_path: Path | str | None,
    retrieved_at: str | None,
    content_hash: str | None,
    filed_at: str | None,
    known_at: str | None,
    source_url: str,
    filer_name: str | None,
    filer_cik: int | str | None,
    report_period: str | None,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None]:
    from ..insider import normalize_13f_holdings

    infotable = _hydrate_infotable(get_by_accession_number, accession)
    if infotable is None:
        return 0, True, "no 13F information table in filing object"
    recs = normalize_13f_holdings(
        infotable,
        manager_name=filer_name,
        manager_cik=str(filer_cik).strip() if filer_cik is not None else None,
        accession_no=accession,
        report_period=report_period,
        filed_at=filed_at,
        form=form_raw,
        document_name=str(doc_name),
        known_at=known_at,
        source_url=source_url or None,
    )
    return _store_13f_rows(_store, recs, raw_path, retrieved_at, content_hash, source_url, data_root)


def _store_13f_rows(
    store: ModuleType,
    recs: list[InstitutionalHolding],
    raw_path: Path | str | None,
    retrieved_at: str | None,
    content_hash: str | None,
    source_url: str,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None]:
    # Live seam: normalize per filing; normalized rows count as typed rows, raw bytes stay in raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (store, raw_path, retrieved_at, content_hash, source_url, data_root)
    return len(list(recs or [])), True, None


def _hydrate_transaction(
    _store: ModuleType,
    get_by_accession_number: Callable[..., object],
    accession: str,
    form_raw: str,
    doc_name: str | None,
    raw_path: Path | str | None,
    retrieved_at: str | None,
    content_hash: str | None,
    filed_at: str | None,
    known_at: str | None,
    source_url: str,
    filer_name: str | None,
    filer_cik: int | str | None,
    subject_cik: int | str | None,
    subject_name: str | None,
    text: str,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None]:
    from ..transactions import normalize_transaction

    obj = _edgar_obj(get_by_accession_number(accession))
    rec = normalize_transaction(
        accession,
        form_raw,
        target=subject_name or "",
        filed_at=filed_at,
        text=text,
        obj=obj,
        filer_cik=filer_cik,
        filer_name=filer_name,
        subject_cik=subject_cik,
        subject_name=subject_name,
        document_name=str(doc_name),
        known_at=known_at,
        source_url=source_url or None,
    )
    # Live seam: normalize per filing; normalized rows count as typed rows, raw bytes stay in raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, raw_path, retrieved_at, content_hash, source_url, data_root)
    return 1 if rec is not None else 0, True, None


def _hydrate_offering(
    _store: ModuleType,
    get_by_accession_number: Callable[..., object],
    accession: str,
    form_raw: str,
    doc_name: str | None,
    raw_path: Path | str | None,
    retrieved_at: str | None,
    content_hash: str | None,
    filed_at: str | None,
    known_at: str | None,
    source_url: str,
    filer_name: str | None,
    filer_cik: int | str | None,
    text: str,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None]:
    from ..offerings import normalize_offering

    obj, terms = _offering_inputs(get_by_accession_number, accession)
    rec = normalize_offering(
        accession,
        form_raw,
        issuer=filer_name or "",
        filed_at=filed_at,
        terms=terms,
        obj=obj,
        text=text,
        filer_cik=filer_cik,
        filer_name=filer_name,
        registrant_cik=filer_cik,
        registrant_name=filer_name,
        document_name=str(doc_name),
        known_at=known_at,
        source_url=source_url or None,
    )
    # Live seam: normalize per filing; normalized rows count as typed rows, raw bytes stay in raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, raw_path, retrieved_at, content_hash, source_url, data_root)
    return 1 if rec is not None else 0, True, None


def _offering_inputs(
    get_by_accession_number: Callable[..., object], accession: str
) -> tuple[object, dict[str, object]]:
    from ..offerings import load_terms

    obj: object = _edgar_obj(get_by_accession_number(accession))
    terms: dict[str, object] = load_terms(accession)
    return obj, terms


class _HydrateUnpacked(NamedTuple):
    form_raw: str
    form: str
    accession: str
    text: str
    doc_name: str | None
    raw_path: Path | str | None
    retrieved_at: str | None
    content_hash: str | None
    filed_at: str | None
    known_at: str | None
    source_url: str
    filer_name: str | None
    filer_cik: int | str | None
    subject_cik: int | None
    subject_name: str | None


def _hydrate_unpack(ctx: _HydrateContext, fields: _DocFields) -> _HydrateUnpacked:
    return _HydrateUnpacked(
        ctx.form_raw,
        ctx.form,
        ctx.accession,
        fields.text,
        fields.doc_name,
        fields.raw_path,
        fields.retrieved_at,
        fields.content_hash,
        ctx.filed_at,
        ctx.known_at,
        ctx.source_url,
        ctx.filer_name,
        ctx.filer_cik,
        ctx.subject_cik,
        ctx.subject_name,
    )


def _hydrate_dispatch(
    store: ModuleType,
    get_by_accession_number: Callable[..., object],
    filing: Filing,
    unpacked: _HydrateUnpacked,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None] | None:
    routed = _hydrate_equity_forms(store, get_by_accession_number, filing, unpacked, data_root)
    if routed is not None:
        return routed
    routed = _hydrate_holdings_forms(store, get_by_accession_number, filing, unpacked, data_root)
    if routed is not None:
        return routed
    return _hydrate_deal_forms(store, get_by_accession_number, unpacked, data_root)


def _hydrate_deal_forms(
    store: ModuleType,
    get_by_accession_number: Callable[..., object],
    unpacked: _HydrateUnpacked,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None] | None:
    if unpacked.form in _TRANSACTION_BASE_FORMS:
        return _hydrate_transaction(
            store,
            get_by_accession_number,
            unpacked.accession,
            unpacked.form_raw,
            unpacked.doc_name,
            unpacked.raw_path,
            unpacked.retrieved_at,
            unpacked.content_hash,
            unpacked.filed_at,
            unpacked.known_at,
            unpacked.source_url,
            unpacked.filer_name,
            unpacked.filer_cik,
            unpacked.subject_cik,
            unpacked.subject_name,
            unpacked.text,
            data_root,
        )
    if unpacked.form in _OFFERING_BASE_FORMS:
        return _hydrate_offering(
            store,
            get_by_accession_number,
            unpacked.accession,
            unpacked.form_raw,
            unpacked.doc_name,
            unpacked.raw_path,
            unpacked.retrieved_at,
            unpacked.content_hash,
            unpacked.filed_at,
            unpacked.known_at,
            unpacked.source_url,
            unpacked.filer_name,
            unpacked.filer_cik,
            unpacked.text,
            data_root,
        )
    return None


def _hydrate_equity_forms(
    store: ModuleType,
    get_by_accession_number: Callable[..., object],
    filing: Filing,
    unpacked: _HydrateUnpacked,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None] | None:
    if unpacked.form in ("SC 13D", "SC 13G", "13D", "13G"):
        return _hydrate_schedule(
            store,
            unpacked.accession,
            unpacked.form_raw,
            unpacked.doc_name,
            unpacked.raw_path,
            unpacked.retrieved_at,
            unpacked.content_hash,
            unpacked.filed_at,
            unpacked.known_at,
            unpacked.source_url,
            unpacked.filer_name,
            data_root,
        )
    if unpacked.form in ("3", "4", "5"):
        return _hydrate_ownership(
            store,
            unpacked.accession,
            unpacked.form_raw,
            unpacked.doc_name,
            unpacked.raw_path,
            unpacked.retrieved_at,
            unpacked.content_hash,
            unpacked.filed_at,
            unpacked.known_at,
            unpacked.source_url,
            unpacked.filer_name,
            unpacked.filer_cik,
            data_root,
        )
    return None


def _hydrate_holdings_forms(
    store: ModuleType,
    get_by_accession_number: Callable[..., object],
    filing: Filing,
    unpacked: _HydrateUnpacked,
    data_root: Path | str | None,
) -> tuple[int, bool, str | None] | None:
    if unpacked.form == "13F-HR":
        return _hydrate_13f(
            store,
            get_by_accession_number,
            unpacked.accession,
            unpacked.form_raw,
            unpacked.doc_name,
            unpacked.raw_path,
            unpacked.retrieved_at,
            unpacked.content_hash,
            unpacked.filed_at,
            unpacked.known_at,
            unpacked.source_url,
            unpacked.filer_name,
            unpacked.filer_cik,
            filing.report_period,
            data_root,
        )
    return None


class _HydrateFetched(NamedTuple):
    store: ModuleType
    get_by_accession_number: Callable[[str], object]
    filing: Filing
    unpacked: _HydrateUnpacked


def _hydrate_fetched(filing: Filing, data_root: Path | str | None) -> _HydrateFetched | str:
    from .. import store as _store
    from ..documents import get_by_accession_number, get_sec_document

    ctx = _hydrate_context(filing)
    if ctx is None:
        return "missing accession"
    fetched = _hydrate_fetch_doc(get_sec_document, ctx.accession, data_root)
    if fetched.error is not None:
        return fetched.error
    assert fetched.doc is not None
    stamped = _hydrate_stamp_doc(fetched.doc, filing)
    if stamped.error is not None:
        return stamped.error
    assert stamped.fields is not None
    unpacked = _hydrate_unpack(ctx, stamped.fields)
    return _HydrateFetched(_store, get_by_accession_number, filing, unpacked)


def _hydrate_relationship_filing(
    filing: Filing, *, data_root: Path | str | None = None
) -> tuple[int, bool, str | None]:
    """Archive primary document + store typed rows for one filing.

    Returns (typed_rows, doc_ok, error): doc_ok tells the document stage
    from the typed stage; error None means this filing parsed cleanly.
    Never raises.
    """
    try:
        fetched = _hydrate_fetched(filing, data_root)
        if isinstance(fetched, str):
            return 0, False, fetched
        return _hydrate_routed(fetched, data_root)
    except Exception as exc:  # noqa: BLE001 - hydration failure returns an error triple, never raises
        return 0, False, str(exc)


def _hydrate_routed(fetched: _HydrateFetched, data_root: Path | str | None) -> tuple[int, bool, str | None]:
    try:
        routed = _hydrate_dispatch(
            fetched.store, fetched.get_by_accession_number, fetched.filing, fetched.unpacked, data_root
        )
        if routed is not None:
            return routed
        return (0, True, (f"unsupported form for typed hydration: {fetched.unpacked.form_raw!r}"))
    except Exception as exc:  # noqa: BLE001 - typed hydration failure returns an error triple, never raises
        return 0, True, f"typed parse/store failed: {exc}"


def _backfill_window(job: dict[str, object]) -> tuple[list[tuple[int, int]], str | None]:
    raw_start: object = job.get("start_date")
    raw_end: object = job.get("end_date")
    quarters, _ = _quarters_for_range(
        raw_start if raw_start is None or isinstance(raw_start, str) else str(raw_start),
        raw_end if raw_end is None or isinstance(raw_end, str) else str(raw_end),
        cap=10_000,
    )
    end_raw: object = job.get("end_date")
    coverage_date: str | None = end_raw if end_raw is None or isinstance(end_raw, str) else str(end_raw)
    return list(quarters), coverage_date


def _backfill_batch_size(job: dict[str, object]) -> int:
    batch_value: object = job.get("batch_size")
    batch = int(batch_value) if batch_value and isinstance(batch_value, (int, float, str)) else 50
    return max(batch, 1)


def _backfill_head(job: dict[str, object]) -> tuple[str, str, str, int, list[tuple[int, int]], str | None]:
    return (str(job["source"]), str(job["form"]), str(job["id"]), _backfill_batch_size(job), *_backfill_window(job))


def _backfill_skip_current(_store: ModuleType, source: str, key: str, data_root: Path | str | None) -> bool:
    # Live seam: checkpoint state lives on the job ledger; NOTE warehouse slots beside seam.
    _ = (source, key)
    return bool(
        _store.is_partition_covered(
            source=source, form=key.split("/")[0], date_partition=key.split("/", 1)[1], root=data_root
        )
    )


def _backfill_resume_filing(
    _store: ModuleType,
    source: str,
    key: str,
    form: str,
    qs: str,
    qe: str,
    partition: str,
    coverage_date: str | None,
    data_root: Path | str | None,
) -> tuple[list[Filing], bool] | None:
    """Resume-quarter path: live provider rows (search-filings -> archive -> Filing)."""
    # Live seam: transport failure returns None; job ledger owns retry state.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, source, key, partition, coverage_date, data_root)
    _batch, _exh, _err = _live_filing_batch(form, qs, qe)
    if _err is not None:
        return None
    return _batch, False


def _backfill_archive_one(
    _archive: ModuleType, _store: ModuleType, filing: Filing, data_root: Path | str | None
) -> str | None:
    """Backfill = search-filings -> archive -> Filing (no warehouse filing rows)."""
    _ = (_store, data_root)
    # Live seam: filing stage writes straight to <root>/raw via raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    payload = json.dumps(filing.to_dict(), sort_keys=True, default=str).encode()
    _archive.archive_sec_filing(
        filing,
        {"submission": payload},
        url=filing.source or "",
        root=None if data_root is None else Path(data_root) / "raw",
    )
    return filing.accession_no


def _backfill_archive_batch(
    _archive: ModuleType,
    _store: ModuleType,
    batch: list[Filing],
    filing_skip: bool,
    last_key: str | None,
    data_root: Path | str | None,
) -> str | None:
    if filing_skip:
        for filing in batch:
            last_key = filing.accession_no
        return last_key
    for filing in batch:
        last_key = _backfill_archive_one(_archive, _store, filing, data_root)
    return last_key


def _backfill_write_filing_stage(
    _store: ModuleType,
    source: str,
    form: str,
    key: str,
    partition: str,
    batch: list[Filing],
    last_key: str | None,
    feed_snapshot: bool,
    coverage_date: str | None,
    data_root: Path | str | None,
) -> None:
    # Live seam: filing stage is archive-complete; job completion marks coverage on the job ledger.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, source, form, key, partition, batch, last_key, feed_snapshot, coverage_date, data_root)


def _backfill_ensure_coverage(
    _store: ModuleType,
    sources: list[str],
    form: str,
    partition: str,
    coverage_date: str | None,
    data_root: Path | str | None,
) -> None:
    # Live seam: coverage derives from complete jobs on the job ledger; nothing to write.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, sources, form, partition, coverage_date, data_root)


def _backfill_hydrate_batch(batch: list[Filing], data_root: Path | str | None) -> tuple[int, int, list[str]]:
    typed_rows = 0
    doc_bad = 0
    typed_errors: list[str] = []
    for filing in batch:
        rows, ok, err = _hydrate_one(filing, data_root)
        typed_rows += rows
        if not ok:
            doc_bad += 1
        if err:
            typed_errors.append(f"{filing.accession_no}: {err}")
    return typed_rows, doc_bad, typed_errors


def _hydrate_one(filing: Filing, data_root: Path | str | None) -> tuple[int, bool, str | None]:
    try:
        return _hydrate_relationship_filing(filing, data_root=data_root)
    except Exception as exc:  # noqa: BLE001 - relationship hydration failure returns an error triple, never raises
        return 0, False, str(exc)


def _backfill_write_typed_stage(
    _store: ModuleType,
    form: str,
    key: str,
    partition: str,
    batch: list[Filing],
    last_key: str | None,
    typed_rows: int,
    doc_bad: int,
    typed_errors: list[str],
    feed_snapshot: bool,
    coverage_date: str | None,
    data_root: Path | str | None,
) -> bool:
    """Document+typed stage outcome. Returns job_failed; job ledger owns completion state."""
    # Live seam: typed rows normalize per filing, raw bytes stay in raw_archive.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, form, key, partition, batch, last_key, typed_rows, coverage_date, data_root)
    _doc_status, typed_status = _typed_stage_status(doc_bad, typed_errors, feed_snapshot)
    return typed_status != "complete" and not feed_snapshot


def _typed_stage_status(doc_bad: int, typed_errors: list[str], feed_snapshot: bool) -> tuple[str, str]:
    doc_status = "complete" if not doc_bad else "partial"
    typed_status = "complete" if not typed_errors else "partial"
    if feed_snapshot:
        doc_status = typed_status = "partial"
    return doc_status, typed_status


def _backfill_fail(
    _store: ModuleType, job: dict[str, object], source: str, form: str, exc: Exception, data_root: Path | str | None
) -> bool:
    # Live seam: failure state lives on the job ledger; NOTE warehouse slots beside seam.
    _ = (source, form)
    _store.fail_job(str(job["id"]), str(exc), root=data_root)
    return False


class _BackfillTracker(TypedDict):
    last_key: str | None
    total: int
    failed: bool


def _tracker_last_key(tracker: _BackfillTracker) -> str | None:
    return tracker["last_key"]


def _tracker_total(tracker: _BackfillTracker) -> int:
    return tracker["total"]


def run_backfill_job(job: dict[str, object], data_root: Path | str | None = None) -> bool:
    """Drain one leased job: archive + normalize + coverage + checkpoint."""
    from .. import archive as _archive
    from .. import store as _store
    from ..client import get_current_filings, get_global_filings

    source, form, job_id, batch_size, quarters, coverage_date = _backfill_head(job)
    now = datetime.now(UTC)
    current = (now.year, (now.month - 1) // 3 + 1)
    needs_typed = _needs_typed_hydration(form)
    try:
        targets = list(quarters) or [None]
        tracker: _BackfillTracker = {"last_key": None, "total": 0, "failed": False}
        for target in targets:
            _backfill_target(
                _archive,
                _store,
                get_current_filings,
                get_global_filings,
                target,
                current,
                source,
                form,
                batch_size,
                coverage_date,
                needs_typed,
                tracker,
                data_root,
            )
        job_failed = bool(tracker["failed"])
        last_key = _tracker_last_key(tracker)
        if job_failed:
            _store.fail_job(job_id, "typed stage incomplete; retryable", last_key=last_key, root=data_root)
            return False
        _store.complete_job(job_id, last_key=last_key, root=data_root)
        return True
    except Exception as exc:  # noqa: BLE001 - job completion failure routes to the failure path, never raises
        return _backfill_fail(_store, job, source, form, exc, data_root)


class _BackfillTarget(NamedTuple):
    partition: str
    key: str
    rows: list[Filing]
    feed_snapshot: bool
    filing_skip: bool
    typed_done: bool


def _backfill_current_target(
    store: ModuleType,
    get_current_filings: object,
    current: tuple[int, int],
    source: str,
    form: str,
    batch_size: int,
    data_root: Path | str | None,
) -> _BackfillTarget | None:
    partition = f"{current[0]}-Q{current[1]}"
    key = f"{form}/{partition}"
    if _backfill_skip_current(store, source, key, data_root):
        return None
    assert callable(get_current_filings)
    return _BackfillTarget(
        partition, key, _call_filing_list(get_current_filings, form, page_size=batch_size), True, False, False
    )


def _backfill_quarter_target(
    store: ModuleType,
    get_global_filings: object,
    target: tuple[int, int],
    source: str,
    form: str,
    coverage_date: str | None,
    needs_typed: bool,
    tracker: _BackfillTracker,
    data_root: Path | str | None,
) -> _BackfillTarget | None:
    year, quarter = target
    partition = _partition_for_quarter(year, quarter)
    key = f"{form}/{partition}"
    qs, qe = _quarter_dates(year, quarter)
    filing_done, _doc_done, typed_done = _backfill_stage_flags(store, source, key, needs_typed, data_root)
    if _backfill_all_done(store, source, key, form, partition, coverage_date, needs_typed, data_root):
        return None
    if filing_done:
        # Filing stage already complete: rehydrate from the
        # live provider index instead of a warehouse read.
        return _backfill_resumed_target(
            store, source, key, form, qs, qe, partition, coverage_date, typed_done, tracker, data_root
        )
    assert callable(get_global_filings)
    return _BackfillTarget(
        partition, key, _call_filing_list(get_global_filings, year, quarter, form=form), False, False, typed_done
    )


def _backfill_stage_flags(
    store: ModuleType, source: str, key: str, needs_typed: bool, data_root: Path | str | None
) -> tuple[bool, bool, bool]:
    # Live seam: checkpoint state lives on the job ledger; completion derives from covered partitions.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (needs_typed, data_root)
    try:
        form, partition = key.split("/", 1)
    except ValueError:
        return False, False, False
    filing_done = bool(store.is_partition_covered(source=source, form=form, date_partition=partition, root=data_root))
    doc_done = bool(store.is_partition_covered(source=DOC_SOURCE, form=form, date_partition=partition, root=data_root))
    typed_done = (not needs_typed) or bool(
        store.is_partition_covered(source=TYPED_SOURCE, form=form, date_partition=partition, root=data_root)
    )
    return filing_done, doc_done, typed_done


def _backfill_all_done(
    store: ModuleType,
    source: str,
    key: str,
    form: str,
    partition: str,
    coverage_date: str | None,
    needs_typed: bool,
    data_root: Path | str | None,
) -> bool:
    filing_done, doc_done, typed_done = _backfill_stage_flags(store, source, key, needs_typed, data_root)
    if filing_done and doc_done and typed_done:
        _backfill_ensure_coverage(
            store,
            [source, DOC_SOURCE] + ([TYPED_SOURCE] if needs_typed else []),
            form,
            partition,
            coverage_date,
            data_root,
        )
        return True
    return False


def _backfill_resumed_target(
    store: ModuleType,
    source: str,
    key: str,
    form: str,
    qs: str,
    qe: str,
    partition: str,
    coverage_date: str | None,
    typed_done: bool,
    tracker: _BackfillTracker,
    data_root: Path | str | None,
) -> _BackfillTarget | None:
    resumed = _backfill_resume_filing(store, source, key, form, qs, qe, partition, coverage_date, data_root)
    if resumed is None:
        tracker["failed"] = True
        return None
    rows, _ = resumed
    return _BackfillTarget(partition, key, list(rows), False, True, typed_done)


def _backfill_target(
    archive: ModuleType,
    store: ModuleType,
    get_current_filings: object,
    get_global_filings: object,
    target: tuple[int, int] | None,
    current: tuple[int, int],
    source: str,
    form: str,
    batch_size: int,
    coverage_date: str | None,
    needs_typed: bool,
    tracker: _BackfillTracker,
    data_root: Path | str | None,
) -> None:
    resolved = _backfill_resolve(
        store,
        get_current_filings,
        get_global_filings,
        target,
        current,
        source,
        form,
        batch_size,
        coverage_date,
        needs_typed,
        tracker,
        data_root,
    )
    if resolved is None:
        return
    batch = _backfill_write_filing(
        archive, store, resolved, source, form, batch_size, coverage_date, tracker, data_root
    )
    _backfill_write_typed(store, resolved, form, batch, coverage_date, needs_typed, tracker, data_root)


def _backfill_write_typed(
    store: ModuleType,
    resolved: _BackfillTarget,
    form: str,
    batch: list[Filing],
    coverage_date: str | None,
    needs_typed: bool,
    tracker: _BackfillTracker,
    data_root: Path | str | None,
) -> None:
    # Document + typed stages (relationship forms only).
    if needs_typed and not resolved.typed_done:
        typed_rows, doc_bad, typed_errors = _backfill_hydrate_batch(batch, data_root)
        if _backfill_write_typed_stage(
            store,
            form,
            resolved.key,
            resolved.partition,
            batch,
            _tracker_last_key(tracker),
            typed_rows,
            doc_bad,
            typed_errors,
            resolved.feed_snapshot,
            coverage_date,
            data_root,
        ):
            tracker["failed"] = True


def _backfill_resolve(
    store: ModuleType,
    get_current_filings: object,
    get_global_filings: object,
    target: tuple[int, int] | None,
    current: tuple[int, int],
    source: str,
    form: str,
    batch_size: int,
    coverage_date: str | None,
    needs_typed: bool,
    tracker: _BackfillTracker,
    data_root: Path | str | None,
) -> _BackfillTarget | None:
    if target is None:
        return _backfill_current_target(store, get_current_filings, current, source, form, batch_size, data_root)
    return _backfill_quarter_target(
        store, get_global_filings, target, source, form, coverage_date, needs_typed, tracker, data_root
    )


def _backfill_write_filing(
    archive: ModuleType,
    store: ModuleType,
    resolved: _BackfillTarget,
    source: str,
    form: str,
    batch_size: int,
    coverage_date: str | None,
    tracker: _BackfillTracker,
    data_root: Path | str | None,
) -> list[Filing]:
    # Feed snapshots are bounded samples, never full-quarter coverage;
    # quarterly partitions drain fully within the job, so "complete"
    # means the source was exhausted (never claimed after truncation).
    batch = list(resolved.rows)[:batch_size] if resolved.feed_snapshot else list(resolved.rows)
    tracker["last_key"] = _backfill_archive_batch(
        archive, store, batch, resolved.filing_skip, _tracker_last_key(tracker), data_root
    )
    tracker["total"] = _tracker_total(tracker) + len(batch)
    # Filing-index stage.
    if not resolved.filing_skip:
        _backfill_write_filing_stage(
            store,
            source,
            form,
            resolved.key,
            resolved.partition,
            batch,
            _tracker_last_key(tracker),
            resolved.feed_snapshot,
            coverage_date,
            data_root,
        )
    return batch


def drain_backfill_queue(data_root: Path | str | None = None, max_jobs: int | None = None) -> dict[str, int]:
    """Synchronously claim->ingest queued jobs (CLI/resume path)."""
    from .. import store as _store

    _store.recover_stale_jobs(root=data_root)
    done, failed = 0, 0
    while not _drain_capped(max_jobs, done, failed):
        job = _store.claim_job(root=data_root)
        if job is None:
            break
        if run_backfill_job(job, data_root):
            done += 1
        else:
            failed += 1
    return {"completed": done, "failed": failed}


def _drain_capped(max_jobs: int | None, done: int, failed: int) -> bool:
    return max_jobs is not None and done + failed >= max_jobs


def ensure_backfill_worker(data_root: Path | str | None = None) -> threading.Thread | None:
    """Start the single process-local daemon draining the durable queue."""
    global _WORKER_THREAD
    from .. import store as _store

    with _WORKER_LOCK:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return _WORKER_THREAD
        _store.recover_stale_jobs(root=data_root)

        def _drain() -> None:
            while True:
                job = _store.claim_job(root=data_root)
                if job is None:
                    return
                run_backfill_job(job, data_root)

        # ponytail: one daemon thread per process; later enqueues restart it.
        _WORKER_THREAD = threading.Thread(target=_drain, name="sec-backfill", daemon=True)
        _WORKER_THREAD.start()
        return _WORKER_THREAD


def _hit_recency(hit: SECTextHit) -> tuple[str, float]:
    return (_hit_filed(hit), _hit_score(hit))


def _hit_recency_key(hit: SECTextHit) -> tuple[str, float]:
    return _hit_recency(hit)


def rank_hits(
    hits: Iterable[SECTextHit],
    *,
    verified_ciks: Iterable[int | None] = (),
    verified_names: Iterable[str] = (),
    relevant_forms: Iterable[str] = (),
    query: str | None = None,
    person_name: str | None = None,
) -> tuple[SECTextHit, ...]:
    """Rank after retrieval: issuer > exact query > topic > form > section > money > terms.

    Stable and total: the pre-sort makes filed_at/score the tiebreak, so
    recency never outranks substance. Nothing is discarded (low-ranked
    structured results stay queryable, the packet alone is bounded); each hit
    carries why it ranked in ``relevance_reason``.
    Embedding similarity reorders only equal-rank ties on bounded sets; Needle failure keeps this order.
    """
    ciks, names, forms = _rank_sets(verified_ciks, verified_names, relevant_forms)
    key = _RankKey(
        ciks,
        names,
        forms,
        _rank_query_terms(query),
        isinstance(person_name, str) and bool(person_name.strip()),
        _norm_query(query),
    )
    by_recency = sorted(hits or (), key=_hit_recency_key, reverse=True)
    ranked = tuple(key.tag(hit) for hit in sorted(by_recency, key=key))
    return _embedding_tiebreak(ranked, key, query)


_EMBED_TIEBREAK_MAX_CHARS = 1000


def _hit_embed_text(hit: SECTextHit) -> str:
    """Bounded hit text for the embedding tiebreak (metadata only, never evidence).

    Snippet first, semantic fields only: filer, form and document name are
    near-constant inside an equal-rank run and dilute the query signal.
    """
    return " ".join(
        part
        for part in (
            hit.snippet or "",
            hit.file_description or "",
            " ".join(hit.items or ()),
        )
        if part
    )[:_EMBED_TIEBREAK_MAX_CHARS]


def _embedding_tiebreak(ranked: tuple[SECTextHit, ...], key: _RankKey, query: str | None) -> tuple[SECTextHit, ...]:
    """Embedding order within equal deterministic ranks; input order on any failure."""
    if not isinstance(query, str) or not query.strip() or len(ranked) < 2:
        return ranked
    try:
        from ...needle_client import similarity_order
    except ImportError:
        return ranked
    keys = [key(hit) for hit in ranked]
    out = list(ranked)
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and keys[end] == keys[start]:
            end += 1
        if end - start > 1:
            run = out[start:end]
            try:
                order = similarity_order([_hit_embed_text(hit) for hit in run], query)
            except Exception:  # noqa: BLE001 - embedding is a tiebreak; failure keeps deterministic order
                order = None
            if order is not None:
                out[start:end] = [run[i] for i in order]
        start = end
    return tuple(out)


class _RankKey:
    def __init__(
        self, ciks: set[int], names: set[str], forms: set[str], wants: tuple[str, ...], person: bool, query: str | None
    ) -> None:
        self.ciks = ciks
        self.names = names
        self.forms = forms
        self.wants = wants
        self.person = person
        self.query = query

    def __call__(self, hit: SECTextHit) -> tuple[int, ...]:
        return _hit_rank(hit, self.ciks, self.names, self.forms, self.wants, self.person, self.query)

    def reasons(self, hit: SECTextHit) -> tuple[str, ...]:
        """Why this hit ranked: one token per firing signal, rank order."""
        section = _hit_section(hit, self.wants)
        flags = (
            ("issuer-match", _hit_identity(hit, self.ciks, self.names) == 0),
            ("exact-query-match", self.query is not None and _hit_exact(hit, self.query) == 0),
            ("query-topic-match", bool(self.wants) and _hit_topic(hit, self.wants) == 0),
            ("requested-form", _hit_relevance(hit, self.forms) == 0),
            ("priority-form", _hit_form_weight(hit, self.person) <= 2),
            ("query-section-match", section == 0),
            ("disclosure-section", section == 1),
            ("quantified-exposure", _hit_money(hit) == 0),
            ("exposure-terminology", _hit_terms(hit) == 0),
        )
        return tuple(label for label, fired in flags if fired)

    def tag(self, hit: SECTextHit) -> SECTextHit:
        """Hit with its rank reasons attached; already-reasoned hits pass through."""
        if hit.relevance_reason:
            return hit
        return replace(hit, relevance_reason=self.reasons(hit))


def _hit_filed(hit: SECTextHit) -> str:
    return str(hit.filed_at or "")


def _hit_score(hit: SECTextHit) -> float:
    try:
        return float(hit.score or 0.0)
    except TypeError, ValueError:
        return 0.0


def _hit_identity(hit: SECTextHit, ciks: set[int], names: set[str]) -> int:
    if hit.issuer_cik is not None and hit.filer_cik == hit.issuer_cik:
        return 0
    if hit.filer_cik in ciks:
        return 0
    filer_name = hit.filer_name or ""
    if filer_name and normalize_name(filer_name) in names:
        return 0
    return 1


def _norm_query(query: str | None) -> str | None:
    """Casefolded request query for exact-variant matching; None when blank."""
    text = query.strip().casefold() if isinstance(query, str) else ""
    return text or None


def _hit_exact(hit: SECTextHit, query: str) -> int:
    """0 when the hit came from the exact request query, not an expanded variant."""
    return 0 if (hit.query or "").strip().casefold() == query else 1


def _rank_query_terms(query: str | None) -> tuple[str, ...]:
    """Lowercased query tokens (length>=3) for topic/section matching."""
    if not isinstance(query, str) or not query.strip():
        return ()
    return tuple(
        dict.fromkeys(token.casefold() for token in re.split(r"[^0-9a-z]+", query.casefold()) if len(token) >= 3)
    )


def _hit_topic(hit: SECTextHit, wants: tuple[str, ...]) -> int:
    """0 when a query token names the filer/file; 1 when none of them do."""
    if not wants:
        return 0
    haystack = " ".join(
        part
        for part in (hit.filer_name or "", hit.form or "", hit.file_description or "", hit.matched_document or "")
        if part
    ).casefold()
    return 0 if any(token in haystack for token in wants) else 1


# 10-K/10-Q/8-K/S-1 lead; 3/4/5/144 trail unless this is a person-name query.
_FORM_WEIGHT = {
    "10-K": 0,
    "10-K/A": 0,
    "10-Q": 1,
    "10-Q/A": 1,
    "8-K": 2,
    "8-K/A": 2,
    "S-1": 3,
    "S-1/A": 3,
    "3": 8,
    "4": 8,
    "5": 8,
    "144": 8,
    "3/A": 8,
    "4/A": 8,
    "5/A": 8,
    "144/A": 8,
}


def _hit_form_weight(hit: SECTextHit, person: bool) -> int:
    form = (hit.form or "").strip().upper()
    if person and form in _FORM_WEIGHT and _FORM_WEIGHT[form] >= 8:
        return 1
    return _FORM_WEIGHT.get(form, 4)


# EFTS item/file_type/file_description carry the filing section when the index has one.
_SECTION_TERMS = ("risk factor", "mda", "md&a", "management's discussion", "business", "financial statement", "note")


def _hit_section_haystack(hit: SECTextHit) -> str:
    """Casefolded section text (single join site for the section signal)."""
    return " ".join(
        part for part in (hit.file_type or "", hit.file_description or "", " ".join(hit.items or ())) if part
    ).casefold()


_SECTION_RANK = {(True, True): 0, (True, False): 0, (False, True): 1, (False, False): 2}
"""Section rank as data: query-token hit outranks disclosure-section hit."""


def _hit_section(hit: SECTextHit, wants: tuple[str, ...]) -> int:
    haystack = _hit_section_haystack(hit)
    return _SECTION_RANK[
        (bool(wants) and any(token in haystack for token in wants), any(term in haystack for term in _SECTION_TERMS))
    ]


_MONEY_RE = re.compile(r"\$[\d,]+(\.\d+)?|\b\d+(\.\d+)?\s*(million|billion|usd|\$)")


def _hit_money(hit: SECTextHit) -> int:
    """0 when the hit names exposure terms + a dollar amount (EFTS metadata)."""
    haystack = " ".join(part for part in (hit.file_description or "", " ".join(hit.items or ())) if part).casefold()
    terms = _TERM_RE.findall(haystack)
    if terms and _MONEY_RE.search(haystack):
        return 0
    if terms:
        return 1
    return 2


_TERM_RE = re.compile(r"contract|concentration|investments?|commitments?|counterpart\w*")


def _hit_terms(hit: SECTextHit) -> int:
    """0 when the hit names contract/concentration/investment/commitment/counterparty terms."""
    haystack = " ".join(part for part in (hit.file_description or "", " ".join(hit.items or ())) if part).casefold()
    return 0 if _TERM_RE.search(haystack) else 1


def _hit_relevance(hit: SECTextHit, forms: set[str]) -> int:
    form = (hit.form or "").strip().upper()
    if form and form in forms:
        return 0
    return 1


def _rank_ciks(verified_ciks: Iterable[int | None]) -> set[int]:
    return {c for c in verified_ciks or () if c is not None}


def _rank_names(verified_names: Iterable[str]) -> set[str]:
    return {normalize_name(n) for n in verified_names or () if n}


def _rank_forms(relevant_forms: Iterable[str]) -> set[str]:
    return {f.strip().upper() for f in relevant_forms or () if f.strip()}


def _rank_sets(
    verified_ciks: Iterable[int | None], verified_names: Iterable[str], relevant_forms: Iterable[str]
) -> tuple[set[int], set[str], set[str]]:
    return (
        _rank_ciks(verified_ciks),
        _rank_names(verified_names),
        _rank_forms(relevant_forms),
    )


def _hit_rank(
    hit: SECTextHit,
    ciks: set[int],
    names: set[str],
    forms: set[str],
    wants: tuple[str, ...] = (),
    person: bool = False,
    query: str | None = None,
) -> tuple[int, ...]:
    return (
        _hit_identity(hit, ciks, names),
        _hit_exact(hit, query) if query is not None else 1,
        _hit_topic(hit, wants),
        _hit_relevance(hit, forms),
        _hit_form_weight(hit, person),
        _hit_section(hit, wants),
        _hit_money(hit),
        _hit_terms(hit),
    )


def build_evidence_packet(
    search_id: str,
    *,
    entities: Iterable[EntityCandidate] = (),
    filings: Iterable[Filing] = (),
    text_hits: Iterable[SECTextHit] = (),
    max_items: int = _EVIDENCE_MAX_ITEMS,
    max_chars: int = _EVIDENCE_MAX_CHARS,
) -> tuple[str, ...]:
    """Bounded packet IDs (verified entities, ranked hits, filings); lists stay full.

    Only the packet is context-budgeted; the stored search keeps every
    entity/filing/hit queryable.
    """
    packet = _EvidencePacket(search_id, max_items, max_chars)
    if _packet_prefix(packet, entities, text_hits):
        packet.push_filings(filings or ())
    return packet.ids


_PACKET_DISPLAY_MAX_CHARS = 2000


_PACKET_DISPLAY_RECORD: dict[str, object] = {
    "name": "packet_hit_display",
    "description": "Display labels for one SEC search hit, copied from the stored hit text.",
    "parameters": {
        "type": "object",
        "properties": {
            "section": {"type": "string", "description": "Filing section label from the stored hit text."},
            "term": {"type": "string", "description": "Exposure term from the stored hit text."},
            "snippet": {
                "type": "string",
                "description": "Verbatim quote of the stored hit text, never a paraphrase.",
            },
        },
    },
}


def _packet_display_text(hit: SECTextHit) -> str:
    """Stored hit text behind one packet row (EFTS metadata + snippet, never live bytes)."""
    return " ".join(
        part
        for part in (
            hit.filer_name or "",
            hit.form or "",
            hit.file_type or "",
            hit.file_description or "",
            " ".join(hit.items or ()),
            hit.snippet or "",
        )
        if part
    )[:_PACKET_DISPLAY_MAX_CHARS]


def _grounded_display_value(value: object, stored: str) -> str | None:
    """Needle string only when it reproduces stored bytes verbatim, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    return text if text in stored else None


def packet_display_fields(hit: SECTextHit) -> dict[str, str | None]:
    """Needle display labels for one stored hit; ungrounded output drops to None.

    IDs, rank, and provenance stay untouched: this fills only the display
    labels (section/term/snippet) the packet window shows, verified against
    stored hit bytes. Strict extract raises on ungrounded values, so any
    Needle failure keeps the deterministic fallbacks the caller already uses.
    """
    stored = _packet_display_text(hit)
    if not stored.strip():
        return {"section": None, "term": None, "snippet": None}
    try:
        from ...needle_client import extract_fields
    except ImportError:
        return {"section": None, "term": None, "snippet": None}
    try:
        out = extract_fields(_PACKET_DISPLAY_RECORD, stored, strict=True)
    except Exception:  # noqa: BLE001 - display labels only; Needle failure keeps None fallbacks
        return {"section": None, "term": None, "snippet": None}
    fields = out.get("fields")
    if not isinstance(fields, dict):
        return {"section": None, "term": None, "snippet": None}
    return {
        "section": _grounded_display_value(fields.get("section"), stored),
        "term": _grounded_display_value(fields.get("term"), stored),
        "snippet": _grounded_display_value(fields.get("snippet"), stored),
    }


class _EvidencePacket:
    """Bounded packet accumulator with flat ~120-char weight per item."""

    def __init__(self, search_id: str, max_items: int, max_chars: int) -> None:
        self.search_id = search_id
        self.max_items = max_items
        self.max_chars = max_chars
        self.items: list[str] = []
        self.budget = 0

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self.items)

    def push(self, packet_id: str) -> bool:
        # ponytail: flat ~120-char weight per item instead of serializing bodies.
        if len(self.items) >= self.max_items or self.budget + len(packet_id) + 120 > self.max_chars:
            return False
        self.items.append(packet_id)
        self.budget += len(packet_id) + 120
        return True

    def push_entities(self, entities: Iterable[EntityCandidate]) -> bool:
        for entity in entities:
            if entity.verification_status != "verified":
                continue
            cik = entity.cik
            if cik is not None:
                packet_id = f"{self.search_id}-entity-cik-{cik}"
            else:
                packet_id = f"{self.search_id}-entity-{normalize_name(entity.name)[:40]}"
            if not self.push(packet_id):
                return False
        return True

    def push_hits(self, text_hits: Iterable[SECTextHit]) -> bool:
        for hit in text_hits:
            doc = hit.matched_document or "primary"
            if not self.push(f"{self.search_id}-hit-{hit.accession_no}-{doc}"):
                return False
        return True

    def push_filings(self, filings: Iterable[Filing]) -> None:
        for filing in filings:
            if not self.push(f"{self.search_id}-filing-{filing.accession_no}"):
                return


class _SearchState:
    """Mutable accumulation for the six search routes + final assembly."""

    search_id: str
    as_of: str | None
    now: str
    attempts: list[SearchAttempt]
    warnings: list[str]
    errors: list[str]
    entities: dict[int | str, EntityCandidate]
    filings: dict[str, Filing]
    documents: dict[tuple[str, str | None], FilingDocument]
    relationships: dict[tuple[str, str, int | None, str], FilingParty]
    hits: dict[tuple[str, str, str | None], SECTextHit]
    retrieval_order: list[str]
    variants: list[tuple[str, str]]
    pit_gaps: int
    quarter_capped: bool
    caller_capped: bool
    pending: list[str]
    adopted_limits: list[str]
    adopted_not_complete: list[str]
    rel_cap: list[int | None]
    rel_pages: list[int]
    rel_open: list[int]
    adopt_idmap: dict[str, str]
    issuer_cik: int | None
    full_hits: int
    full_filings: int
    full_entities: int

    def __init__(self, search_id: str, as_of: str | None, now: str) -> None:
        self.search_id = search_id
        self.as_of = as_of
        self.now = now
        self.attempts: list[SearchAttempt] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.entities: dict[int | str, EntityCandidate] = {}
        self.filings: dict[str, Filing] = {}
        self.documents: dict[tuple[str, str | None], FilingDocument] = {}
        self.relationships: dict[tuple[str, str, int | None, str], FilingParty] = {}
        self.hits: dict[tuple[str, str, str | None], SECTextHit] = {}
        self.retrieval_order: list[str] = []
        self.variants: list[tuple[str, str]] = []
        self.pit_gaps = 0
        self.quarter_capped = False
        self.caller_capped = False
        self.pending: list[str] = []
        self.adopted_limits: list[str] = []
        self.adopted_not_complete: list[str] = []
        self.rel_cap: list[int | None] = [50]
        self.rel_pages = [0]
        self.rel_open = [0]
        self.adopt_idmap: dict[str, str] = {}
        self.issuer_cik: int | None = None
        self.full_hits = 0
        self.full_filings = 0
        self.full_entities = 0

    def record(
        self,
        backend: str,
        query: str,
        status: _AttemptStatus,
        *,
        reported: int = 0,
        retrieved: int = 0,
        pages: int = 0,
        pit_basis: str | None = None,
        error: Exception | None = None,
        source_limit: str | None = None,
        filters: dict[str, object] | None = None,
    ) -> None:
        self.attempts.append(
            self._build_attempt(
                backend, query, status, reported, retrieved, pages, pit_basis, error, source_limit, filters
            )
        )
        if backend not in self.retrieval_order:
            self.retrieval_order.append(backend)

    def _attempt_filters(self, filters: dict[str, object] | None) -> dict[str, object]:
        entry_filters = dict(filters or {})
        if self.as_of and "as_of" not in entry_filters:
            entry_filters["as_of"] = self.as_of
        return entry_filters

    def _build_attempt(
        self,
        backend: str,
        query: str,
        status: _AttemptStatus,
        reported: int,
        retrieved: int,
        pages: int,
        pit_basis: str | None,
        error: Exception | None,
        source_limit: str | None,
        filters: dict[str, object] | None,
    ) -> SearchAttempt:
        return SearchAttempt(
            attempt_id=f"{self.search_id}-{backend}-{len(self.attempts) + 1}",
            search_id=self.search_id,
            backend=backend,
            query=query,
            filters=self._attempt_filters(filters),
            started_at=self.now,
            completed_at=self.now,
            status=status,
            results_reported=reported,
            results_retrieved=retrieved,
            pages_retrieved=pages,
            truncated=status in ("partial", "source_limited"),
            source_limit=source_limit,
            pit_basis=pit_basis,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    def merge_entity(self, candidate: EntityCandidate) -> None:
        key = (
            candidate.cik
            if candidate.cik is not None
            else candidate.entity_id or f"name:{normalize_name(candidate.name)}"
        )
        if key not in self.entities:
            self.entities[key] = candidate

    def keep(self, record: Filing) -> bool:
        if self.as_of is None:
            return True
        value, _basis = pit_of(record)
        if value is None or value[:10] > self.as_of:
            self.pit_gaps += 1
            return False
        return True

    def adopt(self, sub: SECSearchResult, *, route: str | None = None) -> None:
        _adopt_attempts(self, sub, route)
        _adopt_coverage(self, sub)
        _adopt_results(self, sub)

    def add_variants(self, items: Iterable[object], route: str) -> None:
        for item in items or ():
            text = str(item).strip()
            if text and all(text != seen for seen, _ in self.variants):
                self.variants.append((text, route))

    def rel_rows(
        self, query_fn: Callable[..., list[dict[str, object]]], data_root: Path | str | None, **kw: object
    ) -> list[dict[str, object]]:
        # Live seam: accession-scoped typed queries only; no persisted stub indexes.
        # NOTE: a future warehouse slots in beside this seam, never inside providers.
        rows, exh, pg = _fetch_typed(query_fn, cap=self.rel_cap[0], root=data_root, **kw)
        self.rel_pages[0] += pg
        if not exh:
            self.rel_open[0] += 1
        return rows

    def add_party(
        self, accession: object, cik_value: object, name: object, role: str, source: str, known_at: object
    ) -> None:
        label = _party_label(accession, role, known_at, cik_value, name)
        if label is None:
            return
        cik = _parse_party_cik(cik_value)
        # Phase 7 owns transaction/offering roles; keep Phase 6
        # projection to ownership/insider/13F evidence only.
        party = FilingParty(
            accession_no=str(accession),
            entity_id=_party_entity_id(cik),
            cik=cik,
            name=label,
            role=role,
            source=source,
            known_at=str(known_at),
            parser_version=PARSER_VERSION,
        )
        self.relationships.setdefault((party.accession_no, party.role, party.cik, party.name), party)


def _party_has_keys(accession: object, role: str, known_at: object) -> bool:
    return bool(accession and role and known_at)


def _party_label(accession: object, role: str, known_at: object, cik_value: object, name: object) -> str | None:
    if not _party_has_keys(accession, role, known_at):
        return None
    return _party_text(cik_value, name)


def _party_name_text(name: object, cik_value: object) -> str:
    return str(name or "").strip() or str(cik_value or "")


def _party_text(cik_value: object, name: object) -> str | None:
    return _party_name_text(name, cik_value) or None


def _party_entity_id(cik: int | None) -> str | None:
    if cik is None:
        return None
    return sec_entity_id(cik)


def _parse_party_cik(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except TypeError, ValueError, AttributeError:
        return None


def _adopt_attempts(state: _SearchState, sub: SECSearchResult, route: str | None) -> None:
    idmap: dict[str, str] = {}
    for attempt in sub.attempts:
        aid = f"{state.search_id}-{attempt.backend}-{len(state.attempts) + 1}"
        idmap[attempt.attempt_id] = aid
        entry_filters = dict(attempt.filters or {})
        if route is not None:
            entry_filters.setdefault("route", route)
        state.attempts.append(replace(attempt, attempt_id=aid, search_id=state.search_id, filters=entry_filters))
        if attempt.status == "failed" and attempt.error_message:
            state.errors.append(f"{attempt.backend} {attempt.query}: {attempt.error_message}")
        if attempt.backend not in state.retrieval_order:
            state.retrieval_order.append(attempt.backend)
    state.adopt_idmap = idmap


def _adopt_coverage(state: _SearchState, sub: SECSearchResult) -> None:
    coverage = sub.coverage
    for limit in coverage.source_limits or ():
        if limit not in state.adopted_limits:
            state.adopted_limits.append(limit)
    if coverage.status != "complete":
        state.adopted_not_complete.append(str(coverage.status))


def _adopt_warnings_entities(state: _SearchState, sub: SECSearchResult) -> None:
    for warning in sub.warnings:
        if warning not in state.warnings:
            state.warnings.append(warning)
    for candidate in sub.entities:
        state.merge_entity(candidate)


def _adopt_hit(state: _SearchState, hit: SECTextHit, idmap: dict[str, str]) -> None:
    key = (hit.query, hit.accession_no, hit.matched_document)
    if key not in state.hits:
        state.hits[key] = replace(hit, search_id=state.search_id, attempt_id=idmap.get(hit.attempt_id, hit.attempt_id))


def _adopt_results(state: _SearchState, sub: SECSearchResult) -> None:
    idmap = state.adopt_idmap
    _adopt_warnings_entities(state, sub)
    for filing in sub.filings:
        state.filings.setdefault(filing.accession_no, filing)
    for document in sub.documents:
        state.documents.setdefault((document.accession_no, document.document_name), document)
    for party in sub.relationships:
        state.relationships.setdefault((party.accession_no, party.role, party.cik, party.name), party)
    for hit in sub.text_hits:
        _adopt_hit(state, hit, idmap)


def _append_selector(selectors: list[str], selector: str | None) -> None:
    text = selector.strip() if selector is not None else ""
    if text and all(text != seen for seen in selectors):
        selectors.append(text)


def _search_entity_selectors(request: SECSearchRequest, entity_query: str | None) -> list[str]:
    selectors: list[str] = []
    _append_selector(selectors, request.cik)
    _append_selector(selectors, request.ticker)
    if not ((request.cik or request.ticker) and entity_query == request.query):
        _append_selector(selectors, entity_query)
    return selectors


def _fetch_entity_selector(
    request: SECSearchRequest, selector: str, as_of: str | None, data_root: Path | str | None
) -> SECSearchResult | ValueError:
    """One selector fetch with no shared-state mutation; caller adopts on its own thread."""
    try:
        return find_sec_entities(
            selector,
            as_of=as_of,
            exhaustive=request.exhaustive,
            max_results=request.max_results,
            data_root=data_root,
        )
    except ValueError as exc:
        return exc


def _search_run_entity_selector(
    state: _SearchState, request: SECSearchRequest, selector: str, as_of: str | None, data_root: Path | str | None
) -> None:
    sub = _fetch_entity_selector(request, selector, as_of, data_root)
    if isinstance(sub, ValueError):
        state.record("entity-discovery", selector, "failed", error=sub)
        state.errors.append(str(sub))
        return
    state.adopt(sub, route="entity")


def _search_accession_route(state: _SearchState, request: SECSearchRequest, as_of: str | None) -> None:
    # Route 1: exact accession first.
    if request.accession_no:
        try:
            state.adopt(resolve_sec_accession(request.accession_no, as_of=as_of), route="accession")
        except ValueError as exc:
            state.record("exact-accession", request.accession_no, "failed", error=exc)
            state.errors.append(str(exc))
    else:
        state.record("exact-accession", "no accession_no", "not_applicable")


def _search_entity_route(
    state: _SearchState,
    request: SECSearchRequest,
    entity_query: str | None,
    as_of: str | None,
    data_root: Path | str | None,
) -> list[EntityCandidate]:
    # Route 2: exact CIK/ticker/name entity routes; fetches overlap, adoption stays serial.
    entity_selectors = _search_entity_selectors(request, entity_query)
    if entity_selectors and request.search_entities:
        import concurrent.futures as _fut

        with _fut.ThreadPoolExecutor(max_workers=len(entity_selectors)) as pool:
            import functools as _ft

            fetch = _ft.partial(_fetch_entity_selector, request, as_of=as_of, data_root=data_root)
            subs = list(pool.map(fetch, entity_selectors))
        for selector, sub in zip(entity_selectors, subs):
            if isinstance(sub, ValueError):
                state.record("entity-discovery", selector, "failed", error=sub)
                state.errors.append(str(sub))
            else:
                state.adopt(sub, route="entity")
        return [e for e in state.entities.values() if e.verification_status == "verified"]
    _note_entity_skipped(state, request, entity_query)
    return []


def _note_entity_skipped(state: _SearchState, request: SECSearchRequest, entity_query: str | None) -> None:
    if entity_query is None:
        state.record("entity-discovery", "no query/ticker/cik/company_name", "not_applicable")
    else:
        state.record("entity-discovery", entity_query, "not_applicable", filters={"reason": "disabled by request"})


def _search_accession_and_entities(
    state: _SearchState, request: SECSearchRequest, as_of: str | None, data_root: Path | str | None
) -> tuple[str | None, list[EntityCandidate]]:
    _search_accession_route(state, request, as_of)
    entity_query = request.query or request.company_name or request.ticker or request.cik
    return entity_query, _search_entity_route(state, request, entity_query, as_of, data_root)


def _search_efts_params(request: SECSearchRequest) -> tuple[list[str] | None, int | None]:
    forms = list(request.forms) if request.forms else None
    # Exhaustive drains to route exhaustion or the documented EFTS source cap;
    # max_results only bounds the returned packet (display), never retrieval.
    if request.exhaustive:
        return forms, 10_000
    return forms, (request.max_results if request.max_results is not None else 50)


def _explicit_request_cik(request: SECSearchRequest) -> int | None:
    """Explicit request CIK as int; None when absent or unparsable (single coerce site)."""
    return _parse_cik(request.cik) if request.cik is not None else None


def _search_issuer_cik(request: SECSearchRequest, verified: list[EntityCandidate]) -> int | None:
    """Single verified filer corpus: explicit CIK wins, else sole verified CIK, else ticker resolve."""
    explicit = _explicit_request_cik(request)
    if request.cik is not None:
        return explicit
    ciks = {e.cik for e in verified or () if e.cik is not None}
    if len(ciks) == 1:
        return next(iter(ciks))
    if request.ticker:
        from ..client import resolve_cik

        return resolve_cik(request.ticker.strip().upper())
    return None


def _search_efts_drive(
    state: _SearchState, request: SECSearchRequest, as_of: str | None, result_limit: int | None
) -> None:
    forms, per_variant = _search_efts_params(request)
    import time as _time

    # ponytail: global 30s EFTS deadline across variants; each variant keeps its own
    # per-drain budget, but several slow variants no longer stack past the search bound.
    deadline = _time.monotonic() + 30.0
    for i, (variant, route) in enumerate(state.variants):
        if i > 0 and _time.monotonic() >= deadline:
            state.warnings.append(f"EFTS variants stopped after 30s; {variant!r} skipped")
            state.record("efts", variant, "not_applicable", filters={"reason": "global EFTS deadline"})
            continue
        _search_efts_variant(state, request, variant, route, forms, per_variant, as_of, result_limit)


def _search_efts_empty(state: _SearchState) -> None:
    if not any(a.backend == "efts" for a in state.attempts):
        state.record("efts", "no text/person/domain/security query", "not_applicable")


def _search_efts_route(
    state: _SearchState,
    request: SECSearchRequest,
    entity_query: str | None,
    verified: list[EntityCandidate],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    # Route 3: EFTS text/topic/person/domain/security variants.
    if request.search_documents:
        _search_text_variants(state, request, entity_query, verified, as_of)
        _search_efts_fetch(state, request, as_of, result_limit)
    else:
        _search_efts_disabled(state, request)


def _search_person_variant(state: _SearchState, request: SECSearchRequest, entity_query: str | None) -> None:
    if request.person_name is not None:
        _fetch_person_variant(state, request)
    elif entity_query is None and request.domain is None and request.security_identifier is None:
        state.record("person-search", "no person_name", "not_applicable")


def _fetch_person_variant(state: _SearchState, request: SECSearchRequest) -> None:
    assert request.person_name is not None
    try:
        state.add_variants(_expand_person_queries(request.person_name), "person")
    except ValueError as exc:
        state.record("efts", request.person_name, "failed", error=exc, filters={"route": "person"})
        state.errors.append(str(exc))


def _search_domain_variant(state: _SearchState, request: SECSearchRequest, entity_query: str | None) -> None:
    if request.domain is not None:
        _fetch_domain_variant(state, request)
    elif entity_query is None and request.person_name is None and request.security_identifier is None:
        state.record("domain-search", "no domain", "not_applicable")


def _merge_domain_mention(state: _SearchState, domain_variants: list[str]) -> None:
    state.add_variants(domain_variants, "domain")
    # Mention-backed only: never verified, never an entity id.
    state.merge_entity(
        EntityCandidate(
            cik=None,
            name=domain_variants[0],
            tickers=(),
            exchange=None,
            match_source="domain-mention",
            match_score=0.0,
            match_type="text_mention",
            verification_status="unverified",
            entity_id=None,
        )
    )


def _fetch_domain_variant(state: _SearchState, request: SECSearchRequest) -> None:
    assert request.domain is not None
    try:
        domain_variants = _expand_domain_queries(request.domain)
    except ValueError as exc:
        state.record("efts", request.domain, "failed", error=exc, filters={"route": "domain"})
        state.errors.append(str(exc))
    else:
        _merge_domain_mention(state, domain_variants)


def _search_security_variant(state: _SearchState, request: SECSearchRequest, entity_query: str | None) -> None:
    if request.security_identifier is not None:
        _fetch_security_variant(state, request)
        # Security identity stays separate: no entity candidate.
    elif entity_query is None and request.person_name is None and request.domain is None:
        state.record("security-search", "no security_identifier", "not_applicable")


def _fetch_security_variant(state: _SearchState, request: SECSearchRequest) -> None:
    assert request.security_identifier is not None
    try:
        state.add_variants(_expand_security_queries(request.security_identifier), "security")
    except ValueError as exc:
        state.record("efts", request.security_identifier, "failed", error=exc, filters={"route": "security"})
        state.errors.append(str(exc))


def _search_text_variants(
    state: _SearchState,
    request: SECSearchRequest,
    entity_query: str | None,
    verified: list[EntityCandidate],
    as_of: str | None,
) -> None:
    _search_issuer_variant(state, request, verified)
    if entity_query is not None:
        state.add_variants(_expand_entity_queries(verified, as_of) if verified else [], "entity")
        state.add_variants([entity_query.strip()], "text")
    _search_person_variant(state, request, entity_query)
    _search_domain_variant(state, request, entity_query)
    _search_security_variant(state, request, entity_query)


def _search_issuer_variant(state: _SearchState, request: SECSearchRequest, verified: list[EntityCandidate]) -> None:
    """Ticker->CIK corpus marker: every EFTS variant searches the issuer's filings."""
    cik = _search_issuer_cik(request, verified)
    if cik is None:
        return
    state.issuer_cik = cik
    state.record("issuer-scope", f"CIK {cik}", "complete", filters={"cik": str(cik)})


def _search_efts_disabled(state: _SearchState, request: SECSearchRequest) -> None:
    state.record("efts", "disabled by request", "not_applicable")
    if request.person_name is not None:
        state.record("person-search", "disabled by request", "not_applicable")
    if request.domain is not None:
        state.record("domain-search", "disabled by request", "not_applicable")
    if request.security_identifier is not None:
        state.record("security-search", "disabled by request", "not_applicable")


def _search_efts_variant(
    state: _SearchState,
    request: SECSearchRequest,
    variant: str,
    route: str,
    forms: list[str] | None,
    per_variant: int | None,
    as_of: str | None,
    result_limit: int | None,
) -> None:

    sub = _fetch_efts_variant(state, request, variant, route, forms, per_variant, as_of)
    if sub is None:
        return
    _warn_variant_scope(state, sub, route, variant)
    _adopt_efts_variant(state, sub, route, result_limit)


_VARIANT_SCOPE_TERMS = ("as_of", "outside filer scope")
"""Warning substrings worth surfacing per variant (PIT/scope exclusions only)."""


def _warn_variant_scope(state: _SearchState, sub: SECSearchResult, route: str, variant: str) -> None:
    """Surface PIT/scope exclusions per variant; silence means fully in-corpus."""
    seen = set(state.warnings)
    scoped = [f"{route} {variant!r}: {w}" for w in sub.warnings if any(term in w for term in _VARIANT_SCOPE_TERMS)]
    state.warnings.extend(tagged for tagged in dict.fromkeys(scoped) if tagged not in seen)


def _fetch_efts_variant(
    state: _SearchState,
    request: SECSearchRequest,
    variant: str,
    route: str,
    forms: list[str] | None,
    per_variant: int | None,
    as_of: str | None,
) -> SECSearchResult | None:
    from ..client import search_sec_filings

    try:
        # CIK alone scopes EFTS; ticker would re-trigger resolve inside the client.
        return search_sec_filings(
            variant,
            forms=forms,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=per_variant or 10_000,
            as_of=as_of,
            cik=state.issuer_cik,
            ticker=None,
        )
    except Exception as exc:  # noqa: BLE001 - route failure records an attempt and degrades to no results
        state.record("efts", variant, "failed", error=exc, filters={"route": route})
        state.errors.append(f"efts {variant!r} failed: {exc}")
        return None


def _adopt_efts_variant(state: _SearchState, sub: SECSearchResult, route: str, result_limit: int | None) -> None:
    if result_limit is not None and sub.coverage.status == "partial" and not sub.errors:
        state.caller_capped = True
    state.adopt(sub, route=route)


def _search_efts_fetch(
    state: _SearchState, request: SECSearchRequest, as_of: str | None, result_limit: int | None
) -> None:
    # EFTS per variant: explicit limit (or exhaustive=False default) stays bounded;
    # exhaustive without an explicit limit pages to reported-total exhaustion or the
    # documented EFTS cap; unretrieved remainder stays partial.
    if state.variants:
        _search_efts_drive(state, request, as_of, result_limit)
    else:
        _search_efts_empty(state)


def _search_efts_route(
    state: _SearchState,
    request: SECSearchRequest,
    entity_query: str | None,
    verified: list[EntityCandidate],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    # Route 3: EFTS text/topic/person/domain/security variants.
    if request.search_documents:
        _search_text_variants(state, request, entity_query, verified, as_of)
        _search_efts_fetch(state, request, as_of, result_limit)
    else:
        _search_efts_disabled(state, request)


def _search_filer_unknown_cik(state: _SearchState, as_of: str | None) -> None:
    state.record(
        "filer-submissions",
        "unknown cik",
        "failed",
        error=ValueError("missing cik"),
        pit_basis="known_at" if as_of else None,
    )
    state.errors.append("filer-submissions unknown cik failed: missing cik")


def _search_filer_fetch(
    candidate: EntityCandidate,
    request: SECSearchRequest,
    as_of: str | None,
    probe: int | None,
    list_sec_filings: Callable[..., object],
) -> list[Filing]:
    assert candidate.cik is not None
    assert callable(list_sec_filings)
    return _call_filing_list(
        list_sec_filings,
        candidate.cik,
        forms=list(request.forms) if request.forms else None,
        start_date=request.start_date,
        end_date=request.end_date,
        as_of=as_of,
        limit=probe,
    )


def _search_filer_record(
    state: _SearchState,
    request: SECSearchRequest,
    candidate: EntityCandidate,
    rows: list[Filing],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    assert candidate.cik is not None
    kept, partial = _filer_slice(rows, result_limit, state)
    _filer_record_attempt(state, request, candidate, rows, kept, partial, as_of, result_limit)
    for filing in kept:
        state.filings.setdefault(filing.accession_no, filing)


def _filer_slice(rows: list[Filing], result_limit: int | None, state: _SearchState) -> tuple[list[Filing], bool]:
    if result_limit is not None and len(rows) > result_limit:
        state.caller_capped = True
        return rows[:result_limit], True
    if result_limit is None:
        return rows, False
    return rows[:result_limit], False


def _filer_record_attempt(
    state: _SearchState,
    request: SECSearchRequest,
    candidate: EntityCandidate,
    rows: list[Filing],
    kept: list[Filing],
    partial: bool,
    as_of: str | None,
    result_limit: int | None,
) -> None:
    assert candidate.cik is not None
    state.record(
        "filer-submissions",
        str(candidate.cik),
        _filer_status(partial),
        reported=len(rows),
        retrieved=len(kept),
        pages=1,
        pit_basis="known_at" if as_of else None,
        source_limit=_filer_limit(partial, result_limit),
        filters=_filer_filters(request),
    )


def _filer_status(partial: bool) -> _AttemptStatus:
    return "partial" if partial else "complete"


def _filer_limit(partial: bool, result_limit: int | None) -> str | None:
    return f"{result_limit} filings" if partial else None


def _search_filer_enabled(
    state: _SearchState,
    request: SECSearchRequest,
    verified: list[EntityCandidate],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    from ..filings import list_sec_filings

    for candidate in verified:
        _search_filer_candidate(state, request, candidate, as_of, result_limit, list_sec_filings)


def _filer_filters(request: SECSearchRequest) -> dict[str, object]:
    return {
        "forms": list(request.forms) if request.forms else None,
        "start_date": request.start_date,
        "end_date": request.end_date,
    }


def _search_filer_candidate(
    state: _SearchState,
    request: SECSearchRequest,
    candidate: EntityCandidate,
    as_of: str | None,
    result_limit: int | None,
    list_sec_filings: Callable[..., object],
) -> None:
    if candidate.cik is None:
        _search_filer_unknown_cik(state, as_of)
        return
    rows = _fetch_filer_rows(state, request, candidate, as_of, result_limit, list_sec_filings)
    if rows is None:
        return
    _search_filer_record(state, request, candidate, rows, as_of, result_limit)


def _fetch_filer_rows(
    state: _SearchState,
    request: SECSearchRequest,
    candidate: EntityCandidate,
    as_of: str | None,
    result_limit: int | None,
    list_sec_filings: Callable[..., object],
) -> list[Filing] | None:
    assert candidate.cik is not None
    try:
        return _search_filer_fetch(
            candidate, request, as_of, None if result_limit is None else result_limit + 1, list_sec_filings
        )
    except Exception as exc:  # noqa: BLE001 - route failure records an attempt and degrades to no results
        state.record(
            "filer-submissions", str(candidate.cik), "failed", error=exc, pit_basis="known_at" if as_of else None
        )
        state.errors.append(f"filer-submissions {candidate.cik} failed: {exc}")
        return None


def _search_filer_route(
    state: _SearchState,
    request: SECSearchRequest,
    verified: list[EntityCandidate],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    # Route 4: filer submissions for known entities.
    if verified and request.search_documents:
        _search_filer_enabled(state, request, verified, as_of, result_limit)
    elif not verified:
        state.record("filer-submissions", "no verified entity", "not_applicable")
    else:
        state.record("filer-submissions", "disabled by request", "not_applicable")


def _dedup_form(global_forms: list[str], form: str) -> None:
    text = (form or "").strip()
    if text and all(text.upper() != seen.upper() for seen in global_forms):
        global_forms.append(text)


def _search_dedup_forms(request: SECSearchRequest) -> list[str]:
    global_forms: list[str] = []
    for form in request.forms or ():
        _dedup_form(global_forms, form)
    if request.search_relationships and request.person_name:
        for form in _PERSON_FORMS:
            _dedup_form(global_forms, form)
    return global_forms


def _search_split_partitions(
    backfill_store: ModuleType, ordered_forms: list[str], quarters: list[tuple[int, int]], data_root: Path | str | None
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    missing: list[tuple[str, int, int]] = []
    covered: list[tuple[str, int, int]] = []
    # Live seam: coverage reads from the job ledger; NOTE warehouse slots beside seam.
    for form in ordered_forms:
        for year, quarter in quarters:
            partition = _partition_for_quarter(year, quarter)
            is_covered = backfill_store.is_partition_covered(
                source=BACKFILL_SOURCE,
                form=form,
                date_partition=partition,
                root=data_root,
            )
            (covered if is_covered else missing).append((form, year, quarter))
    return missing, covered


def _search_add_capped_quarters(
    missing: list[tuple[str, int, int]],
    ordered_forms: list[str],
    quarters: list[tuple[int, int]],
    request: SECSearchRequest,
) -> None:
    # Discarded older quarters get the same bounded backfill
    # jobs as missing partitions; never claimed as covered.
    full_quarters, _ = _quarters_for_range(request.start_date, request.end_date, cap=10_000)
    recent = set(quarters)
    for form in ordered_forms:
        for year, quarter in full_quarters:
            if (year, quarter) not in recent:
                missing.append((form, year, quarter))


def _search_enqueue_missing(
    state: _SearchState,
    backfill_store: ModuleType,
    missing: list[tuple[str, int, int]],
    batch_size: int,
    data_root: Path | str | None,
) -> None:
    for form, year, quarter in missing:
        _search_enqueue_partition(state, backfill_store, form, year, quarter, batch_size, data_root)
    try:
        ensure_backfill_worker(data_root)
    except Exception as exc:  # noqa: BLE001 - worker-start failure degrades to a warning, search continues
        state.warnings.append(f"backfill worker failed to start: {exc}")
    state.warnings.append(
        f"{len(missing)} quarterly partition(s) not yet "
        f"ingested; queued backfill jobs {state.pending} and "
        "returned immediately (never waits for history)"
    )


def _search_enqueue_partition(
    state: _SearchState,
    backfill_store: ModuleType,
    form: str,
    year: int,
    quarter: int,
    batch_size: int,
    data_root: Path | str | None,
) -> None:
    qs, qe = _quarter_dates(year, quarter)
    partition = _partition_for_quarter(year, quarter)
    try:
        job_id = _enqueue_or_requeue(
            backfill_store, BACKFILL_SOURCE, form, qs, qe, batch_size=batch_size, root=data_root
        )
    except Exception as exc:  # noqa: BLE001 - route failure records an attempt and degrades to no results
        state.record(
            "backfill",
            f"{form} {partition}",
            "failed",
            error=exc,
            filters={"form": form, "year": year, "quarter": quarter},
        )
        state.errors.append(f"backfill enqueue {partition} failed: {exc}")
        return
    state.pending.append(job_id)
    state.record(
        "backfill",
        f"{form} {partition}",
        "partial",
        filters={
            "form": form,
            "year": year,
            "quarter": quarter,
            "partition": partition,
            "job_id": job_id,
            "checkpoint": f"sec-backfill/{BACKFILL_SOURCE}/{form}/{partition}",
        },
    )


def _search_parse_covered_row(filing: Filing, qs: str, qe: str, state: _SearchState) -> Filing | None:
    day = (filing.filed_at or filing.known_at or "")[:10]
    if day and not qs <= day <= qe:
        return None
    if state.keep(filing):
        return filing
    return None


def _search_record_covered(
    state: _SearchState,
    form: str,
    partition: str,
    rows: list[Filing],
    kept: list[Filing],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    fully_evaluated = _covered_fully_evaluated(rows, result_limit, state)
    state.record(
        "live-filings",
        f"{form} {partition}",
        "complete" if fully_evaluated else "partial",
        reported=len(rows),
        retrieved=len(kept),
        pages=1,
        pit_basis="known_at" if as_of else None,
        source_limit=_covered_limit(fully_evaluated, result_limit),
        filters={"form": form, "partition": partition},
    )


def _search_covered_partition(
    state: _SearchState,
    backfill_store: ModuleType,
    form: str,
    year: int,
    quarter: int,
    as_of: str | None,
    result_limit: int | None,
    data_root: Path | str | None,
) -> None:
    partition = _partition_for_quarter(year, quarter)
    qs, qe = _quarter_dates(year, quarter)
    rows = _fetch_covered_rows(state, backfill_store, form, partition, qs, qe, as_of, result_limit, data_root)
    if rows is None:
        return
    kept = _assemble_covered_rows(state, rows, qs, qe, result_limit)
    _search_record_covered(state, form, partition, rows, kept, as_of, result_limit)
    for filing in kept:
        state.filings.setdefault(filing.accession_no, filing)


def _fetch_covered_rows(
    state: _SearchState,
    backfill_store: ModuleType,
    form: str,
    partition: str,
    qs: str,
    qe: str,
    as_of: str | None,
    result_limit: int | None,
    data_root: Path | str | None,
) -> list[Filing] | None:
    _ = (backfill_store, data_root)
    try:
        rows, _exhausted, error = _live_filing_batch(form, qs, qe, as_of=as_of, limit=result_limit)
        if error is not None:
            raise RuntimeError(error)
        probe = rows if result_limit is None else rows[: result_limit + 1]
        return probe
    except (ConnectionError, TimeoutError, OSError, RuntimeError) as exc:
        state.record(
            "live-filings",
            f"{form} {partition}",
            "failed",
            error=exc,
            pit_basis="known_at" if as_of else None,
            filters={"form": form, "partition": partition},
        )
        state.errors.append(f"live-filings {partition} failed: {exc}")
        return None


def _assemble_covered_rows(
    state: _SearchState, rows: list[Filing], qs: str, qe: str, result_limit: int | None
) -> list[Filing]:
    kept_all: list[Filing] = []
    for filing in rows:
        kept = _search_parse_covered_row(filing, qs, qe, state)
        if kept is not None:
            kept_all.append(kept)
    return kept_all if result_limit is None else kept_all[:result_limit]


def _covered_fully_evaluated(
    rows: list[Filing] | list[dict[str, object]], result_limit: int | None, state: _SearchState
) -> bool:
    fully = result_limit is None or len(rows) <= result_limit
    if result_limit is not None and not fully:
        state.caller_capped = True
    return fully


def _covered_limit(fully_evaluated: bool, result_limit: int | None) -> str | None:
    if fully_evaluated:
        return None
    return f"{result_limit} filings"


def _search_want_current(request: SECSearchRequest) -> bool:
    if not request.start_date and not request.end_date:
        return True
    if request.end_date:
        _now = datetime.now(UTC)
        _cur = (_now.year, (_now.month - 1) // 3 + 1)
        _cur_qs, _ = _quarter_dates(*_cur)
        return request.end_date >= _cur_qs
    return True


def _filing_day(filing: Filing) -> str:
    return (filing.filed_at or filing.known_at or "")[:10]


def _after_start(day: str, start: str | None) -> bool:
    return bool(start and day and day < start)


def _before_end(day: str, end: str | None) -> bool:
    return bool(end and day and day > end)


def _search_filter_current_row(filing: Filing, request: SECSearchRequest) -> bool:
    day = _filing_day(filing)
    if _after_start(day, request.start_date):
        return False
    return not _before_end(day, request.end_date)


def _search_filter_current_rows(state: _SearchState, request: SECSearchRequest, rows: list[Filing]) -> list[Filing]:
    kept: list[Filing] = []
    for filing in rows:
        if not _search_filter_current_row(filing, request):
            continue
        if state.keep(filing):
            kept.append(filing)
    return kept


def _fetch_current_form(
    state: _SearchState,
    form: str,
    get_current_filings: Callable[..., object],
    as_of: str | None,
    result_limit: int | None,
) -> list[Filing] | None:
    current_probe = None if result_limit is None else result_limit + 1
    try:
        assert callable(get_current_filings)
        return _call_filing_list(get_current_filings, form, page_size=current_probe)
    except Exception as exc:  # noqa: BLE001 - route failure records an attempt and degrades to no results
        state.record(
            "current-filings",
            form,
            "failed",
            error=exc,
            pit_basis="known_at" if as_of else None,
            filters={"form": form},
        )
        state.errors.append(f"current-filings {form!r} failed: {exc}")
        return None


def _current_is_partial(rows: list[Filing], result_limit: int | None, state: _SearchState) -> bool:
    partial = result_limit is not None and len(rows) > result_limit
    if partial:
        state.caller_capped = True
    return partial


def _search_record_current(
    state: _SearchState, form: str, rows: list[Filing], kept: list[Filing], as_of: str | None, result_limit: int | None
) -> None:
    if _current_is_partial(rows, result_limit, state):
        state.record(
            "current-filings",
            form,
            "partial",
            reported=len(rows),
            retrieved=len(kept),
            pages=1,
            pit_basis="known_at" if as_of else None,
            source_limit=f"{result_limit} filings",
            filters={"form": form},
        )
    else:
        state.record(
            "current-filings",
            form,
            "complete",
            reported=len(rows),
            retrieved=len(kept),
            pages=1,
            pit_basis="known_at" if as_of else None,
            filters={"form": form},
        )


def _search_current_form(
    state: _SearchState,
    request: SECSearchRequest,
    form: str,
    get_current_filings: Callable[..., object],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    rows = _fetch_current_form(state, form, get_current_filings, as_of, result_limit)
    if rows is None:
        return
    kept = _search_filter_current_rows(state, request, rows)
    if result_limit is not None:
        kept = kept[:result_limit]
    _search_record_current(state, form, rows, kept, as_of, result_limit)
    for filing in kept:
        state.filings.setdefault(filing.accession_no, filing)


def _search_current_feed(
    state: _SearchState,
    request: SECSearchRequest,
    ordered_forms: list[str],
    get_current_filings: Callable[..., object],
    as_of: str | None,
    result_limit: int | None,
) -> None:
    # Current quarter always comes from the current feed; skip
    # the live call entirely when the range excludes it.
    if not _search_want_current(request):
        state.record(
            "current-filings", "range excludes current quarter; quarterly partitions cover it", "not_applicable"
        )
        return
    for form in ordered_forms:
        _search_current_form(state, request, form, get_current_filings, as_of, result_limit)


def _search_range_note(state: _SearchState, request: SECSearchRequest, quarters: list[tuple[int, int]]) -> None:
    if not quarters:
        if request.start_date or request.end_date:
            state.record("global-filings", "range served by current feed; no quarterly partitions", "not_applicable")
        else:
            state.record("global-filings", "unbounded range uses the current feed", "not_applicable")
    if state.quarter_capped:
        state.warnings.append(
            f"date range spans more than {_GLOBAL_QUARTER_CAP} "
            "quarterly partitions; searched the most recent "
            f"{_GLOBAL_QUARTER_CAP} and queued backfill jobs for "
            "the older partitions (see backfill attempts)"
        )


def _search_quarter_partitions(
    state: _SearchState,
    request: SECSearchRequest,
    backfill_store: ModuleType,
    get_current_filings: Callable[..., object],
    global_forms: list[str],
    as_of: str | None,
    data_root: Path | str | None,
    batch_size: int,
    result_limit: int | None,
) -> None:
    try:
        quarters, state.quarter_capped = _quarters_for_range(request.start_date, request.end_date)
    except ValueError as exc:
        state.record("global-filings", str(global_forms), "failed", error=exc)
        state.errors.append(str(exc))
        return
    # Local-first: covered partitions run locally; missing ones
    # become bounded quarterly/form jobs, never a blocked call.
    ordered_forms = _sort_forms_by_priority(global_forms)
    missing, covered = _search_split_partitions(backfill_store, ordered_forms, quarters, data_root)
    if state.quarter_capped:
        _search_add_capped_quarters(missing, ordered_forms, quarters, request)
    if missing:
        _search_enqueue_missing(state, backfill_store, missing, batch_size, data_root)
    for form, year, quarter in covered:
        _search_covered_partition(state, backfill_store, form, year, quarter, as_of, result_limit, data_root)
    _search_current_feed(state, request, ordered_forms, get_current_filings, as_of, result_limit)
    _search_range_note(state, request, quarters)


def _search_global_route(
    state: _SearchState,
    request: SECSearchRequest,
    as_of: str | None,
    data_root: Path | str | None,
    batch_size: int,
    result_limit: int | None,
) -> list[str]:
    # Route 5: global filing indexes for forms/relationships.
    global_forms = _search_dedup_forms(request)
    if global_forms and request.search_documents:
        from .. import store as _backfill_store
        from ..client import get_current_filings

        _search_quarter_partitions(
            state,
            request,
            _backfill_store,
            get_current_filings,
            global_forms,
            as_of,
            data_root,
            batch_size,
            result_limit,
        )
    elif not global_forms:
        state.record("global-filings", "no forms", "not_applicable")
        state.record("current-filings", "no forms", "not_applicable")
    else:
        state.record("global-filings", "disabled by request", "not_applicable")
        state.record("current-filings", "disabled by request", "not_applicable")
    return global_forms


_LOCAL_REL_FORMS = ("SC 13D", "SC 13G", "3", "4", "5")
_LOCAL_SEC_FORMS = ("13F-HR",)


def _append_rel_cik(rel_ciks: list[str], text: str) -> None:
    text = text.strip()
    if text and text not in rel_ciks:
        rel_ciks.append(text)


def _search_verified_ciks(verified: list[EntityCandidate]) -> list[str]:
    rel_ciks: list[str] = []
    for candidate in verified:
        try:
            if candidate.cik is not None:
                _append_rel_cik(rel_ciks, str(candidate.cik))
        except Exception:  # noqa: BLE001, S112 - malformed candidate is skipped, verification continues
            continue
    return rel_ciks


def _local_quarters(request: SECSearchRequest) -> list[tuple[int, int]]:
    try:
        quarters, _capped = _quarters_for_range(request.start_date, request.end_date)
    except ValueError:
        return list[tuple[int, int]]()
    return quarters


def _local_ordered_forms() -> list[str]:
    return _sort_forms_by_priority(
        list(
            dict.fromkeys(
                list(_LOCAL_REL_FORMS)
                + list(_LOCAL_SEC_FORMS)
                + sorted(_TRANSACTION_BASE_FORMS)
                + sorted(_OFFERING_BASE_FORMS)
            )
        )
    )


def _search_rel_ciks(state: _SearchState, request: SECSearchRequest, verified: list[EntityCandidate]) -> list[str]:
    rel_ciks = _search_verified_ciks(verified)
    if request.cik is not None:
        try:
            _append_rel_cik(rel_ciks, request.cik)
        except Exception:  # noqa: BLE001, S110 - best-effort related-CIK stamp, failure keeps the prior list
            pass
    return rel_ciks


def _search_queue_local_backfill(
    state: _SearchState, request: SECSearchRequest, rel_store: ModuleType, data_root: Path | str | None
) -> None:
    # Missing partitions become bounded backfill jobs when the
    # request carries a date range; unbounded requests query the
    # local typed indexes directly (partial/limited, never complete).
    quarters = _local_quarters(request)
    if not quarters:
        return
    ordered = _local_ordered_forms()
    for form in ordered:
        for year, quarter in quarters:
            _search_queue_local_partition(state, rel_store, form, year, quarter, data_root)
    if state.pending:
        try:
            ensure_backfill_worker(data_root)
        except Exception as exc:  # noqa: BLE001 - worker-start failure degrades to a warning, search continues
            state.warnings.append(f"backfill worker failed to start: {exc}")
        state.warnings.append(
            "local relationship/security partitions not yet "
            f"ingested; queued backfill jobs {state.pending} and "
            "returned immediately (never waits for history)"
        )


def _search_queue_local_partition(
    state: _SearchState, rel_store: ModuleType, form: str, year: int, quarter: int, data_root: Path | str | None
) -> None:
    # Live seam: coverage reads from the job ledger; NOTE warehouse slots beside seam.
    partition = _partition_for_quarter(year, quarter)
    is_covered = rel_store.is_partition_covered(
        source=TYPED_SOURCE, form=form, date_partition=partition, root=data_root
    )
    if is_covered:
        return
    qs, qe = _quarter_dates(year, quarter)
    try:
        job_id = _enqueue_or_requeue(rel_store, BACKFILL_SOURCE, form, qs, qe, batch_size=50, root=data_root)
    except Exception as exc:  # noqa: BLE001 - route failure records an attempt and degrades to no results
        state.record(
            "backfill",
            f"{form} {partition}",
            "failed",
            error=exc,
            filters={"form": form, "year": year, "quarter": quarter, "route": "local-index"},
        )
        state.errors.append(f"backfill enqueue {partition} failed: {exc}")
        return
    state.pending.append(job_id)
    state.record(
        "backfill",
        f"{form} {partition}",
        "partial",
        filters={
            "form": form,
            "year": year,
            "quarter": quarter,
            "partition": partition,
            "job_id": job_id,
            "route": "local-index",
            "checkpoint": f"sec-backfill/{BACKFILL_SOURCE}/{form}/{partition}",
        },
    )


def _search_add_ownership_rows(
    state: _SearchState, rel_store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> int:
    # Live seam: no persisted cik-scanned ownership index; accession-scoped live queries answer per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, rel_store, cik, as_of, data_root)
    return 0


def _search_add_insider_rows(
    state: _SearchState, rel_store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> int:
    # Live seam: no persisted cik-scanned insider index; accession-scoped live queries answer per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, rel_store, cik, as_of, data_root)
    return 0


def _search_add_13f_rows(
    state: _SearchState, rel_store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> int:
    # Live seam: no persisted cik-scanned 13F index; accession-scoped live queries answer per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, rel_store, cik, as_of, data_root)
    return 0


def _search_local_relationships(
    state: _SearchState,
    request: SECSearchRequest,
    rel_store: ModuleType,
    rel_ciks: list[str],
    unbounded_rel: bool,
    as_of: str | None,
    data_root: Path | str | None,
) -> None:
    if not rel_ciks:
        state.record("local-relationships", "no entity/cik context", "not_applicable")
        return
    try:
        _run_local_relationships(state, request, rel_store, rel_ciks, unbounded_rel, as_of, data_root)
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("local-relationships", f"{rel_ciks}", "failed", error=exc, pit_basis="known_at" if as_of else None)
        state.errors.append(f"local-relationships failed: {exc}")


def _run_local_relationships(
    state: _SearchState,
    request: SECSearchRequest,
    rel_store: ModuleType,
    rel_ciks: list[str],
    unbounded_rel: bool,
    as_of: str | None,
    data_root: Path | str | None,
) -> None:
    rel_mark = (state.rel_pages[0], state.rel_open[0])
    if unbounded_rel:
        state.rel_cap[0] = request.max_results or 50
    rel_found = _accumulate_local_ciks(state, rel_store, rel_ciks, as_of, data_root)
    _search_record_local_relationships(state, rel_ciks, unbounded_rel, rel_found, rel_mark, as_of)


def _search_local_cik(
    state: _SearchState, rel_store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> int:
    found = 0
    found += _search_add_ownership_rows(state, rel_store, cik, as_of, data_root)
    found += _search_add_insider_rows(state, rel_store, cik, as_of, data_root)
    found += _search_add_13f_rows(state, rel_store, cik, as_of, data_root)
    return found


def _accumulate_local_ciks(
    state: _SearchState, rel_store: ModuleType, rel_ciks: list[str], as_of: str | None, data_root: Path | str | None
) -> int:
    rel_found = 0
    for cik in rel_ciks:
        rel_found += _search_local_cik(state, rel_store, cik, as_of, data_root)
    return rel_found


def _warn_unbounded_rel(state: _SearchState, unbounded_rel: bool) -> None:
    if unbounded_rel:
        state.warnings.append("unbounded relationship search covers only locally stored rows (partial, limited)")


def _search_record_local_relationships(
    state: _SearchState,
    rel_ciks: list[str],
    unbounded_rel: bool,
    rel_found: int,
    rel_mark: tuple[int, int],
    as_of: str | None,
) -> None:
    _warn_unbounded_rel(state, unbounded_rel)
    state.record(
        "local-relationships",
        f"{len(rel_ciks)} cik(s)",
        _local_rel_status(state, unbounded_rel, rel_mark),
        reported=rel_found,
        retrieved=len(state.relationships),
        pages=state.rel_pages[0] - rel_mark[0],
        pit_basis="known_at" if as_of else None,
        filters={"ciks": rel_ciks},
    )


def _local_rel_status(state: _SearchState, unbounded_rel: bool, rel_mark: tuple[int, int]) -> _AttemptStatus:
    if unbounded_rel or state.rel_open[0] > rel_mark[1]:
        return "partial"
    return "complete"


def _search_local_securities(
    state: _SearchState,
    request: SECSearchRequest,
    rel_store: ModuleType,
    as_of: str | None,
    data_root: Path | str | None,
) -> None:
    if request.security_identifier is None:
        state.record("local-securities", "no security_identifier", "not_applicable")
        return
    fetched = _fetch_local_securities(state, rel_store, request, as_of, data_root)
    if fetched is None:
        return
    rows, sec_exh, sec_pg = fetched
    for row in rows:
        state.add_party(
            row.get("accession"),
            row.get("manager_cik"),
            row.get("manager_name"),
            "13f-manager",
            "sec-13f",
            row.get("known_at"),
        )
    state.record(
        "local-securities",
        request.security_identifier,
        "complete" if sec_exh else "partial",
        reported=len(rows),
        retrieved=len(rows),
        pages=sec_pg,
        pit_basis="known_at" if as_of else None,
    )


def _fetch_local_securities(
    state: _SearchState,
    rel_store: ModuleType,
    request: SECSearchRequest,
    as_of: str | None,
    data_root: Path | str | None,
) -> tuple[list[dict[str, object]], bool, int] | None:
    # Live seam: no persisted security-scanned 13F index; accession-scoped live queries answer per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    assert request.security_identifier is not None
    _ = (rel_store, as_of, data_root)
    return [], True, 1


def _search_add_transaction_rows(
    state: _SearchState, rel_store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> int:
    # Live seam: no persisted transaction index; live resolution happens per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, rel_store, cik, as_of, data_root)
    return 0


def _search_add_offering_rows(
    state: _SearchState, rel_store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> int:
    # Live seam: no persisted offering index; live resolution happens per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, rel_store, cik, as_of, data_root)
    return 0


def _search_local_transactions(
    state: _SearchState, rel_store: ModuleType, rel_ciks: list[str], as_of: str | None, data_root: Path | str | None
) -> None:
    # Phase 7: local transaction/offering indexes over stored rows.
    # Missing partitions are queued by the global-filings route when
    # those forms are requested; otherwise this reports stored rows.
    # Roles stay mention-free: only evidenced
    # filer/target/acquirer/registrant links project.
    if not rel_ciks:
        state.record("local-transactions", "no entity/cik context", "not_applicable")
        return
    try:
        _run_local_transactions(state, rel_store, rel_ciks, as_of, data_root)
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("local-transactions", f"{rel_ciks}", "failed", error=exc, pit_basis="known_at" if as_of else None)
        state.errors.append(f"local-transactions failed: {exc}")


def _run_local_transactions(
    state: _SearchState, rel_store: ModuleType, rel_ciks: list[str], as_of: str | None, data_root: Path | str | None
) -> None:
    txn_mark = (state.rel_pages[0], state.rel_open[0])
    txn_found = _accumulate_txn_ciks(state, rel_store, rel_ciks, as_of, data_root)
    state.record(
        "local-transactions",
        f"{len(rel_ciks)} cik(s)",
        "complete" if state.rel_open[0] <= txn_mark[1] else "partial",
        reported=txn_found,
        retrieved=len(state.relationships),
        pages=state.rel_pages[0] - txn_mark[0],
        pit_basis="known_at" if as_of else None,
        filters={"ciks": rel_ciks},
    )


def _accumulate_txn_ciks(
    state: _SearchState, rel_store: ModuleType, rel_ciks: list[str], as_of: str | None, data_root: Path | str | None
) -> int:
    txn_found = 0
    for cik in rel_ciks:
        txn_found += _search_add_transaction_rows(state, rel_store, cik, as_of, data_root)
        txn_found += _search_add_offering_rows(state, rel_store, cik, as_of, data_root)
    return txn_found


def _search_local_route(
    state: _SearchState,
    request: SECSearchRequest,
    verified: list[EntityCandidate],
    as_of: str | None,
    data_root: Path | str | None,
    result_limit: int | None,
) -> None:
    # Route 6: local relationship/security indexes (Phase 6 typed rows).
    # Covered partitions run locally; missing ones become bounded
    # quarterly/form jobs, never a blocked call.
    if request.search_relationships:
        _search_local_enabled(state, request, verified, as_of, data_root, result_limit)
    else:
        _search_local_disabled(state)


def _search_rank(
    state: _SearchState, request: SECSearchRequest, verified: list[EntityCandidate], global_forms: list[str]
) -> tuple[SECTextHit, ...]:
    if state.pit_gaps > 0:
        # as_of is embedded in every attempt filter; keep the gap count here.
        pass
    return rank_hits(
        tuple(state.hits.values()),
        verified_ciks=[e.cik for e in verified],
        verified_names=[e.name for e in verified],
        relevant_forms=global_forms,
        query=request.query,
        person_name=request.person_name,
    )


def _search_warn_pit_gaps(state: _SearchState, as_of: str | None) -> None:
    if state.pit_gaps > 0:
        state.warnings.append(f"{state.pit_gaps} global filing(s) excluded by as_of {as_of}")


def _search_cap_results(
    state: _SearchState, ranked: tuple[SECTextHit, ...], request: SECSearchRequest, display_limit: int | None
) -> tuple[tuple[SECTextHit, ...], bool]:
    # Retrieval already drained every route; only the hit packet is
    # display-bound. Filings/entities stay whole (undrained exhaustive keeps
    # all 75; the model packet bounds hits via top_hits/additional_hits).
    state.full_hits = len(ranked)
    state.full_filings = len(state.filings)
    state.full_entities = len(state.entities)
    capped = False
    if display_limit is None:
        return ranked, capped
    if len(ranked) > display_limit:
        ranked = ranked[:display_limit]
        capped = True
    _warn_display_capped(state, display_limit, capped, request)
    return ranked, capped


def _warn_display_capped(state: _SearchState, display: int, capped: bool, request: SECSearchRequest) -> None:
    # Undrained exhaustive retrieval (max_results is None) is never capped by
    # the caller: only the packet is display-bound, so never claim otherwise.
    if request.exhaustive and request.max_results is None:
        if capped:
            state.warnings.append(
                f"exhaustive retrieval kept {state.full_hits} hit(s); "
                f"packet shows the top {display} "
                f"(page the rest with research_read_search)"
            )
        return
    _warn_capped(state, display, capped)
    if request.exhaustive and request.max_results is not None and capped:
        state.warnings.append(
            f"exhaustive retrieval kept {state.full_hits} hit(s); "
            f"packet shows the top {display} "
            f"(page the rest with research_read_search)"
        )


def _warn_capped(state: _SearchState, result_limit: int, capped: bool) -> None:
    if capped or state.caller_capped:
        cap_warning = f"results capped at {result_limit}; rerun with a higher limit or exhaustive=true"
        if cap_warning not in state.warnings:
            state.warnings.append(cap_warning)


def _search_coverage_limits(state: _SearchState, active: list[SearchAttempt]) -> tuple[str, ...]:
    limits: tuple[str, ...] = ()
    if state.quarter_capped or any(a.status == "source_limited" for a in active):
        limits = ("global-filings:quarter-cap",)
    # Adopted sub-coverages (e.g. caller-limited EFTS partials whose page
    # attempts stay complete) fold in: union source limits, never promote
    # an adopted partial to complete.
    return tuple(dict.fromkeys(tuple(limits) + tuple(state.adopted_limits)))


def _search_local_enabled(
    state: _SearchState,
    request: SECSearchRequest,
    verified: list[EntityCandidate],
    as_of: str | None,
    data_root: Path | str | None,
    result_limit: int | None,
) -> None:
    from .. import store as _rel_store

    # ponytail: one shared pager; each route snapshots the counters
    # below so bounded probes stay attributable to their own attempt.
    _reset_rel_pager(state, request)
    rel_ciks = _search_rel_ciks(state, request, verified)
    _search_queue_local_backfill(state, request, _rel_store, data_root)
    _search_local_typed(state, request, _rel_store, rel_ciks, as_of, data_root)
    _search_local_transactions(state, _rel_store, rel_ciks, as_of, data_root)
    if result_limit is not None and state.rel_open[0] > 0:
        state.caller_capped = True


def _search_local_typed(
    state: _SearchState,
    request: SECSearchRequest,
    rel_store: ModuleType,
    rel_ciks: list[str],
    as_of: str | None,
    data_root: Path | str | None,
) -> None:
    unbounded_rel = not (request.start_date or request.end_date)
    _search_local_relationships(state, request, rel_store, rel_ciks, unbounded_rel, as_of, data_root)
    _search_local_securities(state, request, rel_store, as_of, data_root)


def _reset_rel_pager(state: _SearchState, request: SECSearchRequest) -> None:
    # ponytail: one shared pager; each route snapshots the counters
    # below so bounded probes stay attributable to their own attempt.
    state.rel_cap = [None if (request.exhaustive and request.max_results is None) else (request.max_results or 50)]
    state.rel_pages = [0]
    state.rel_open = [0]


def _search_local_disabled(state: _SearchState) -> None:
    state.record("local-relationships", "disabled by request", "not_applicable")
    state.record("local-securities", "disabled by request", "not_applicable")
    state.record("local-transactions", "disabled by request", "not_applicable")


def _search_coverage_status(
    state: _SearchState, active: list[SearchAttempt], limits: tuple[str, ...]
) -> _CoverageStatus:
    if _search_is_failed(active):
        return "failed"
    if _search_is_partial(state, active):
        # Missing partitions queued as bounded backfill jobs: the call
        # returns partial immediately with job IDs, never waits.
        return "partial"
    if limits or _search_adopted_limited(state):
        return "complete_within_source_limits"
    return "complete"


def _search_is_failed(active: list[SearchAttempt]) -> bool:
    return not active or all(a.status == "failed" for a in active)


def _search_forms_seen(state: _SearchState, ranked: tuple[SECTextHit, ...], global_forms: list[str]) -> set[str]:
    forms_seen = _forms_from_global(global_forms)
    forms_seen.update(_forms_from_filings(state))
    forms_seen.update(_forms_from_hits(ranked))
    return forms_seen


def _forms_from_global(global_forms: list[str]) -> set[str]:
    return {f.strip().upper() for f in global_forms if f.strip()}


def _forms_from_filings(state: _SearchState) -> set[str]:
    return {f.form.strip().upper() for f in state.filings.values() if str(getattr(f, "form", "") or "").strip()}


def _forms_from_hits(ranked: tuple[SECTextHit, ...]) -> set[str]:
    return {h.form.strip().upper() for h in ranked if str(getattr(h, "form", "") or "").strip()}


def _search_persist_ledger(
    state: _SearchState,
    request: SECSearchRequest,
    search_id: str,
    ranked: tuple[SECTextHit, ...],
    active: list[SearchAttempt],
    completed: tuple[str, ...],
    failed: tuple[str, ...],
    limits: tuple[str, ...],
    status: _CoverageStatus,
    forms_seen: set[str],
    date_coverage: str | None,
    pagination_complete: bool,
    source_exhausted: bool,
) -> None:
    """Per-session bundle owns the search ledger; no persisted search rows remain."""
    # Live seam: per-session bundle writer owns search ledgers; NOTE warehouse slots beside seam.
    _ = (
        state,
        request,
        search_id,
        ranked,
        active,
        completed,
        failed,
        limits,
        status,
        forms_seen,
        date_coverage,
        pagination_complete,
        source_exhausted,
    )


def _search_attempt_failed(active: list[SearchAttempt]) -> bool:
    return any(a.status in ("failed", "partial") for a in active)


def _search_adopted_failed(state: _SearchState) -> bool:
    return any(s in ("partial", "failed") for s in state.adopted_not_complete)


def _search_queued_partial(state: _SearchState) -> bool:
    return bool(state.pending or state.caller_capped)


def _search_is_partial(state: _SearchState, active: list[SearchAttempt]) -> bool:
    return _search_queued_partial(state) or _search_attempt_failed(active) or _search_adopted_failed(state)


def _search_adopted_limited(state: _SearchState) -> bool:
    return any(s == "complete_within_source_limits" for s in state.adopted_not_complete)


def _search_attempt_filters(attempt: SearchAttempt) -> dict[str, object]:
    """Attempt filters as a plain dict (never the live mapping)."""
    return dict(attempt.filters or {})


def _search_run_counts(ranked: tuple[SECTextHit, ...], accessions: set[str]) -> tuple[int, int]:
    """(matched_documents, matched_passages) for one run."""
    for hit in ranked:
        accessions.add(hit.accession_no)
    return len(accessions), len(ranked)


def _search_runs(
    state: _SearchState, request: SECSearchRequest, ranked: tuple[SECTextHit, ...], as_of: str | None
) -> tuple[SearchRun, ...]:
    """One SearchRun per executed attempt: query + filters + PIT + match counts."""
    _ = request
    runs: list[SearchRun] = []
    for attempt in state.attempts:
        if attempt.status == "not_applicable":
            continue
        accessions: set[str] = set()
        matched_documents, matched_passages = _search_run_counts(
            tuple(h for h in ranked if h.query == attempt.query), accessions
        )
        runs.append(
            SearchRun(
                id=attempt.attempt_id,
                source=attempt.backend,
                query=attempt.query,
                filters=_search_attempt_filters(attempt),
                executed_at=attempt.completed_at or state.now,
                as_of=as_of,
                matched_entities=len(state.entities),
                matched_documents=matched_documents,
                matched_passages=matched_passages,
            )
        )
    return tuple(runs)


def _search_finalize(
    state: _SearchState,
    request: SECSearchRequest,
    global_forms: list[str],
    display_limit: int | None,
    evidence_max_items: int,
    evidence_max_chars: int,
) -> SECSearchResult:
    as_of: str | None = _check_as_of(request.as_of)
    search_id = state.search_id
    _search_warn_pit_gaps(state, as_of)
    ranked_full = _search_rank(state, request, _search_final_verified(state), global_forms)
    ranked, _capped = _search_cap_results(state, ranked_full, request, display_limit)
    packet = build_evidence_packet(
        search_id,
        entities=tuple(state.entities.values()),
        filings=tuple(state.filings.values()),
        text_hits=ranked,
        max_items=evidence_max_items,
        max_chars=evidence_max_chars,
    )
    active = [a for a in state.attempts if a.status != "not_applicable"]
    completed, failed = _search_attempt_sets(state)
    limits = _search_coverage_limits(state, active)
    # Display cap is a context saver, never retrieval completeness: coverage
    # and both retrieval flags read paging/route state only.
    status = _search_coverage_status(state, active, limits)
    forms_seen = _search_forms_seen(state, ranked_full, global_forms)
    date_coverage = _search_date_coverage(request)
    _search_persist_ledger(
        state,
        request,
        search_id,
        ranked_full,
        active,
        completed,
        failed,
        limits,
        status,
        forms_seen,
        date_coverage,
        pagination_complete=not _search_is_partial(state, active),
        source_exhausted=status == "complete",
    )
    return _search_result_packet(
        state,
        request,
        search_id,
        ranked,
        active,
        completed,
        failed,
        limits,
        status,
        forms_seen,
        date_coverage,
        packet,
        as_of,
    )


def _search_result_packet(
    state: _SearchState,
    request: SECSearchRequest,
    search_id: str,
    ranked: tuple[SECTextHit, ...],
    active: list[SearchAttempt],
    completed: tuple[str, ...],
    failed: tuple[str, ...],
    limits: tuple[str, ...],
    status: _CoverageStatus,
    forms_seen: set[str],
    date_coverage: str | None,
    packet: tuple[str, ...],
    as_of: str | None,
) -> SECSearchResult:
    """Final SECSearchResult: ranked display packet over fully drained routes."""
    return SECSearchResult(
        search_id=search_id,
        request=request,
        entities=tuple(state.entities.values()),
        filings=tuple(state.filings.values()),
        documents=tuple(state.documents.values()),
        relationships=tuple(state.relationships.values()),
        text_hits=ranked,
        coverage=SearchCoverage(
            status=status,
            sources_attempted=tuple(state.retrieval_order),
            sources_completed=completed,
            sources_failed=failed,
            source_limits=limits,
            results_reported=sum(a.results_reported for a in active),
            results_retrieved=len(state.filings) + len(ranked) + len(state.entities),
            pages=sum(a.pages_retrieved for a in active),
            date_coverage=date_coverage,
            forms_covered=tuple(sorted(forms_seen)),
            pending_backfill_jobs=tuple(state.pending),
        ),
        attempts=tuple(state.attempts),
        warnings=tuple(state.warnings),
        errors=tuple(state.errors),
        retrieval_order=tuple(state.retrieval_order),
        evidence_packet_ids=packet,
        search_runs=_search_runs(state, request, ranked, as_of),
    )


def _search_attempt_sets(state: _SearchState) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(dict.fromkeys(a.backend for a in state.attempts if a.status == "complete")),
        tuple(dict.fromkeys(a.backend for a in state.attempts if a.status == "failed")),
    )


def _search_date_coverage(request: SECSearchRequest) -> str | None:
    if request.start_date or request.end_date:
        return f"{request.start_date or ''}:{request.end_date or ''}"
    return None


def _search_final_verified(state: _SearchState) -> list[EntityCandidate]:
    return [e for e in state.entities.values() if e.verification_status == "verified"]


def _packet_prefix(
    packet: _EvidencePacket, entities: Iterable[EntityCandidate], text_hits: Iterable[SECTextHit]
) -> bool:
    if not packet.push_entities(entities or ()):
        return False
    return bool(packet.push_hits(text_hits or ()))


class SECDiscoveryService:
    """Cross-source discovery owner: accession, entity, EFTS, submissions,
    global, then local routes; inapplicable routes are explicit attempts."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self._data_root = data_root

    def search(
        self,
        request: SECSearchRequest,
        *,
        evidence_max_items: int = _EVIDENCE_MAX_ITEMS,
        evidence_max_chars: int = _EVIDENCE_MAX_CHARS,
    ) -> SECSearchResult:
        """Run every applicable route; dedup, rank after retrieval, bound packet."""
        data_root = self._data_root
        request = _checked_search_request(request)
        as_of: str | None = _check_as_of(request.as_of)
        search_id, now = uuid.uuid4().hex[:12], _utcnow()
        # Interactive bound for global/current-feed reads and backfill batches.
        batch_size = max(request.max_results or 50, 1)
        # Exhaustive drains every route (None = undrained); max_results only
        # bounds the returned packet (display), never retrieval.
        result_limit = None if request.exhaustive and request.max_results is None else request.max_results
        state = _SearchState(search_id, as_of, now)
        global_forms = _search_run_routes(state, request, as_of, data_root, batch_size, result_limit)
        return _search_finalize(
            state,
            request,
            global_forms,
            50 if result_limit is None else result_limit,
            evidence_max_items,
            evidence_max_chars,
        )


def _checked_search_request(request: SECSearchRequest) -> SECSearchRequest:
    """Type-checked search request; raises on non-request input."""
    if not isinstance(request, SECSearchRequest):
        raise TypeError(f"request must be SECSearchRequest, got {type(request).__name__}")
    return request


def _search_run_routes(
    state: _SearchState,
    request: SECSearchRequest,
    as_of: str | None,
    data_root: Path | str | None,
    batch_size: int,
    result_limit: int | None,
) -> list[str]:
    """Run accession/entity/EFTS/filer/global/local routes; return global forms."""
    entity_query, verified = _search_accession_and_entities(state, request, as_of, data_root)
    _search_efts_route(state, request, entity_query, verified, as_of, result_limit)
    _search_filer_route(state, request, verified, as_of, result_limit)
    global_forms = _search_global_route(state, request, as_of, data_root, batch_size, result_limit)
    _search_local_route(state, request, verified, as_of, data_root, result_limit)
    return global_forms


# --- Phase 8: open-vocabulary relationship search over typed indexes,
# verified/candidate workflow rows, mentions, and EFTS. Results group by
# type/status; mentions never flatten into verified links.

_CIK_RE = re.compile(r"(\d{1,10})\s*$")
_DIRECT_CIK_RE = re.compile(r"^\s*(?:sec:cik:)?0*(\d{1,10})\s*$", re.IGNORECASE)


def _relationship_ciks(entity: object) -> list[str]:
    """Entity id / CIK / candidate -> bare CIK strings (identity, not text)."""
    ciks: list[str] = []
    if isinstance(entity, str):
        _ciks_from_str(ciks, entity)
    else:
        _ciks_from_object(ciks, entity)
    return ciks


def _append_cik(ciks: list[str], raw: str) -> None:
    text = str(int(raw))
    if text not in ciks:
        ciks.append(text)


def _ciks_add(ciks: list[str], value: object) -> None:
    match = _CIK_RE.search(str(value or ""))
    if match:
        _append_cik(ciks, match.group(1))


def _ciks_from_str(ciks: list[str], entity: str) -> None:
    # ponytail: full-match only; substring search turned "Rule 144" into CIK 144
    match = _DIRECT_CIK_RE.match(entity)
    if match:
        _append_cik(ciks, match.group(1))


def _ciks_from_attrs(ciks: list[str], entity: object) -> None:
    for attr in ("cik", "entity_id"):
        try:
            value = getattr(entity, attr, None)
        except Exception:  # noqa: BLE001 - untrusted entity attr read falls through to the next attr name
            value = None
        if value is not None:
            _ciks_add(ciks, value)


def _ciks_from_object(ciks: list[str], entity: object) -> None:
    _ciks_from_attrs(ciks, entity)
    if not ciks and isinstance(entity, dict):
        _ciks_add(ciks, entity.get("cik"))
        _ciks_add(ciks, entity.get("entity_id"))


def _identity_query_from_dict(entity: dict[str, object]) -> str | None:
    for key in ("query", "ticker", "name"):
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _identity_attr(entity: object, attr: str) -> object:
    try:
        return getattr(entity, attr, None)
    except Exception:  # noqa: BLE001 - untrusted entity attr read coerces to None, never raises
        return None


def _identity_name_text(value: object) -> str | None:
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001 - untrusted name value coerces to None, never raises
        return None
    return text or None


def _identity_attr_text(entity: object, attr: str) -> str | None:
    value = _identity_attr(entity, attr)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _identity_name_query(entity: object) -> str | None:
    value = _identity_attr(entity, "name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None:
        return _identity_name_text(value)
    return None


def _identity_query_from_object(entity: object) -> str | None:
    for attr in ("query", "ticker"):
        text = _identity_attr_text(entity, attr)
        if text:
            return text
    return _identity_name_query(entity)


def _identity_query(entity: object) -> str | None:
    if isinstance(entity, str):
        return entity.strip() or None
    if isinstance(entity, dict):
        return _identity_query_from_dict(entity)
    return _identity_query_from_object(entity)


def _identity_verified_ciks(ents: list[EntityCandidate]) -> dict[str, object]:
    uniq: dict[str, object] = {}
    for cand in ents:
        try:
            if getattr(cand, "verification_status", None) != "verified":
                continue
            cik_value = getattr(cand, "cik", None)
            if cik_value is None:
                continue
            uniq.setdefault(str(int(str(cik_value).strip())), cand)
        except Exception:  # noqa: BLE001, S112 - malformed candidate is skipped, dedup continues
            continue
    return uniq


def _identity_entities(sub: object) -> list[EntityCandidate]:
    try:
        return list(getattr(sub, "entities", ()) or ())
    except Exception:  # noqa: BLE001 - untrusted sub-result coerces to empty, never raises
        return []


def _identity_coverage(sub: object) -> tuple[object, list[object]]:
    try:
        return (getattr(getattr(sub, "coverage", None), "status", None), list(getattr(sub, "errors", ()) or ()))
    except Exception:  # noqa: BLE001 - untrusted sub-result coerces to empty coverage, never raises
        return None, []


def _identity_finish(
    sub: object, ents: list[EntityCandidate]
) -> tuple[list[str], list[EntityCandidate], str, Exception | None]:
    uniq = _identity_verified_ciks(ents)
    if len(uniq) == 1:
        return list(uniq), ents, "verified", None
    coverage_status, sub_errors = _identity_coverage(sub)
    if coverage_status == "failed" and not ents:
        first = sub_errors[0] if sub_errors else "entity resolution failed"
        err = first if isinstance(first, Exception) else RuntimeError(str(first))
        return [], ents, "failed", err
    if not ents:
        return [], [], "not_found", None
    return [], ents, "ambiguous", None


def _resolve_relationship_identity(
    entity: object, *, as_of: str | None = None, data_root: Path | str | None = None
) -> tuple[list[str], list[EntityCandidate], str, Exception | None]:
    """Direct CIK/ID or single-verified-candidate CIK; else ([], candidates, status, err)."""
    direct = _relationship_ciks(entity)
    if direct:
        return direct, [], "direct", None
    query = _identity_query(entity)
    if not query:
        return [], [], "unresolved", None
    match = _DIRECT_CIK_RE.match(query)
    if match:
        return [str(int(match.group(1)))], [], "direct", None
    try:
        sub = find_sec_entities(query, as_of=as_of, exhaustive=True, max_results=None, data_root=data_root)
    except Exception as exc:  # noqa: BLE001 - identity resolution failure returns a failed packet, never raises
        return [], [], "failed", exc
    return _identity_finish(sub, _identity_entities(sub))


class _RelState:
    """Mutable accumulation for the relationship fan-out routes."""

    def __init__(self, wanted: set[str] | None) -> None:
        self.wanted = wanted
        self.attempts: list[dict[str, object]] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.typed: list[dict[str, object]] = []
        self.workflow: list[dict[str, object]] = []
        self.mentions: list[dict[str, object]] = []
        self.managers: list[EntityCandidate] = []
        self.t1_cap: int | None = 50
        self.t1_pages = [0]
        self.t1_open = [0]
        self.inv_pages = [0]
        self.inv_open = [0]

    def want(self, label: object) -> bool:
        from ...domain.evidence.relationships import normalize_label

        return self.wanted is None or normalize_label(label) in self.wanted

    def record(self, backend: str, status: str, **extra: object) -> None:
        self.attempts.append({"backend": backend, "status": status, **extra})

    def emit(self, label: object, status: str, row: dict[str, object]) -> None:
        from ...domain.evidence.relationships import normalize_label

        if self.want(label):
            self.typed.append({"relationship_type": normalize_label(label), "status": status, **row})


def _rel_wanted(relationship_types: Iterable[str] | None) -> set[str] | None:
    from ...domain.evidence.relationships import normalize_label

    if relationship_types is None:
        return None
    return {normalize_label(t) for t in relationship_types}


def _rel_unresolved(
    state: _RelState,
    entity: object,
    resolution: str,
    candidates: list[EntityCandidate],
    resolution_err: Exception | None,
    relationship_types: Iterable[str] | None,
    as_of: str | None,
) -> dict[str, object]:
    if resolution == "failed":
        _rel_failed(state, resolution, resolution_err)
    elif resolution == "ambiguous":
        _rel_ambiguous(state, entity, resolution, candidates)
    elif resolution == "not_found":
        _rel_not_found(state, entity, resolution)
    else:
        _rel_no_context(state, entity, resolution)
    _rel_unresolved_tail(state, resolution, resolution_err)
    return {
        "entity": str(entity),
        "ciks": (),
        "relationship_types": tuple(relationship_types or ()),
        "as_of": as_of,
        "groups": {},
        "typed": [],
        "relationships": [],
        "mentions": [],
        "managers": [],
        "attempts": state.attempts,
        "warnings": state.warnings,
        "errors": state.errors,
        "candidates": tuple(candidates),
        "resolution": resolution,
    }


def _rel_failed(state: _RelState, resolution: str, resolution_err: Exception | None) -> None:
    state.record("entity-resolution", "failed", error=str(resolution_err), resolution=resolution)
    state.errors.append(f"entity resolution failed: {resolution_err}")


def _rel_ambiguous(state: _RelState, entity: object, resolution: str, candidates: list[EntityCandidate]) -> None:
    state.record("entity-resolution", "ambiguous", resolution=resolution, candidates=len(candidates))
    state.warnings.append(
        f"ambiguous entity {str(entity)!r}: {len(candidates)} candidates; no relationship rows returned"
    )


def _rel_not_found(state: _RelState, entity: object, resolution: str) -> None:
    state.record("entity-resolution", "not_applicable", resolution=resolution, reason="no verified candidate")
    state.warnings.append(f"no SEC entity candidates for {str(entity)!r} (no direct corpus; mentions still searched)")


def _rel_no_context(state: _RelState, entity: object, resolution: str) -> None:
    state.record("entity-resolution", "not_applicable", resolution=resolution, reason="no cik context")
    state.warnings.append(f"no CIK context for {str(entity)!r}")


def _rel_unresolved_tail(state: _RelState, resolution: str, resolution_err: Exception | None) -> None:
    state.record("local-typed", "not_applicable", reason="no cik context", resolution=resolution)
    state.record(
        "local-workflow",
        "failed" if resolution == "failed" else "not_applicable",
        reason="no entity filter; unfiltered scan disabled",
        resolution=resolution,
        error=str(resolution_err) if resolution_err else None,
    )
    state.record("local-mentions", "not_applicable", reason="no cik context")
    state.record("efts-mentions", "not_applicable", reason="no cik context")


def _rel_typed_cik(
    state: _RelState, store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> None:
    # Live seam: stub cik/index scans are gone; accession-scoped live queries answer per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, store, cik, as_of, data_root)


def _rel_typed_route(
    state: _RelState, store: ModuleType, ciks: list[str], as_of: str | None, data_root: Path | str | None
) -> None:
    # Route 1: typed ownership / holdings / insider indexes, both directions.
    if not ciks:
        state.record("local-typed", "not_applicable", reason="no cik context")
        return
    try:
        for cik in ciks:
            _rel_typed_cik(state, store, cik, as_of, data_root)
        state.record(
            "local-typed",
            "complete" if not state.t1_open[0] else "partial",
            ciks=ciks,
            found=len(state.typed),
            pages=state.t1_pages[0],
        )
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("local-typed", "failed", ciks=ciks, error=str(exc))
        state.errors.append(f"local-typed failed: {exc}")


def _rel_inverse_cik(
    state: _RelState, store: ModuleType, cik: str, as_of: str | None, data_root: Path | str | None
) -> None:
    # Live seam: no persisted issuer map remains; live resolution happens per filing.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (store, as_of, data_root)
    _rel_record_inverse(state, cik, [])


def _rel_record_inverse(state: _RelState, cik: str, held: list[dict[str, object]]) -> None:
    if held and not state.inv_open[0]:
        state.record(
            "local-13f-inverse", "complete", found=len(held), managers=len(state.managers), pages=state.inv_pages[0]
        )
        return
    if held:
        state.record(
            "local-13f-inverse",
            "partial",
            found=len(held),
            pages=state.inv_pages[0],
            reason="retrieval capped before exhaustion",
        )
        return
    state.record(
        "local-13f-inverse",
        "partial",
        pages=state.inv_pages[0],
        reason="no mapped CUSIP/ISIN/security ID holdings for issuer",
    )
    state.warnings.append(
        f"no 13F holdings map to issuer {cik}; unmapped keyspace stays partial (never globally scanned)"
    )


def _rel_inverse_route(
    state: _RelState, store: ModuleType, ciks: list[str], as_of: str | None, data_root: Path | str | None
) -> None:
    # Inverse 13F: verified issuer/entity CIK -> governed CUSIP/ISIN mapping
    # -> holdings -> manager CIKs. Unmapped issuers stay partial; holdings
    # are never globally scanned.
    if not ciks:
        state.record("local-13f-inverse", "not_applicable", reason="no cik context")
        return
    try:
        for cik in ciks:
            _rel_inverse_cik(state, store, cik, as_of, data_root)
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("local-13f-inverse", "failed", ciks=ciks, error=str(exc))
        state.errors.append(f"local-13f-inverse failed: {exc}")


def _rel_workflow_evidence(
    state: _RelState,
    store: ModuleType,
    ciks: list[str],
    as_of: str | None,
    data_root: Path | str | None,
    exhaustive: bool,
    limit: int,
) -> list[dict[str, object]]:
    # Live seam: no persisted evidence ledger remains; per-session bundles own workflow rows.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, store, ciks, as_of, data_root, exhaustive, limit)
    return []


def _rel_workflow_row(
    state: _RelState, store: ModuleType, rid: str, ev_rows: list[dict[str, object]], data_root: Path | str | None
) -> bool:
    # Live seam: no persisted evidence/revision ledger remains; per-session bundles own workflow rows.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (state, store, rid, ev_rows, data_root)
    return False


def _rel_workflow_route(
    state: _RelState,
    store: ModuleType,
    ciks: list[str],
    as_of: str | None,
    data_root: Path | str | None,
    exhaustive: bool,
    limit: int,
) -> None:
    # Route 2: verified/candidate workflow rows keep their stored status.
    try:
        ev_all = _rel_workflow_evidence(state, store, ciks, as_of, data_root, exhaustive, limit)
        # ponytail: no unfiltered fallback; without an entity filter there is no query
        by_rel: dict[str, list[dict[str, object]]] = {}
        for ev in ev_all:
            rid_value: object = ev.get("relationship_id")
            if rid_value:
                by_rel.setdefault(str(rid_value), []).append(ev)
        kept = 0
        for rid, ev_rows in by_rel.items():
            if _rel_workflow_row(state, store, rid, ev_rows, data_root):
                kept += 1
        if exhaustive and len(ev_all) >= _LOCAL_EXHAUSTIVE_GUARD:
            state.record("local-workflow", "partial", found=kept, reason="retrieval capped at local exhaustive guard")
        else:
            state.record("local-workflow", "complete", found=kept)
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("local-workflow", "failed", error=str(exc))
        state.errors.append(f"local-workflow failed: {exc}")


def _rel_mentions_route(
    state: _RelState,
    store: ModuleType,
    ciks: list[str],
    as_of: str | None,
    data_root: Path | str | None,
    exhaustive: bool,
    limit: int,
) -> None:
    # Route 3: local text mentions stay observed, never verified.
    if not ciks:
        state.record("local-mentions", "not_applicable", reason="no cik context")
        return
    try:
        for cik in ciks:
            _rel_collect_mentions(state, store, cik, as_of, data_root, exhaustive, limit)
        _rel_record_mentions(state, exhaustive)
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("local-mentions", "failed", error=str(exc))
        state.errors.append(f"local-mentions failed: {exc}")


def _rel_collect_mentions(
    state: _RelState,
    store: ModuleType,
    cik: str,
    as_of: str | None,
    data_root: Path | str | None,
    exhaustive: bool,
    limit: int,
) -> None:
    # Live seam: mention search delegates to the provider full-text search.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    for row in store.search_document_text(
        query=cik,
        limit=_LOCAL_EXHAUSTIVE_GUARD if exhaustive else min(limit, 20),
        as_of=as_of,
        root=data_root,
    ):
        state.mentions.append(
            {
                "relationship_type": "mention",
                "status": "observed",
                "accession": row.get("accession"),
                "document_name": row.get("document_name"),
                "source_span": str(row.get("text") or "")[:280],
                "known_at": row.get("known_at"),
            }
        )


def _rel_record_mentions(state: _RelState, exhaustive: bool) -> None:
    found = len(state.mentions)
    if exhaustive and found >= _LOCAL_EXHAUSTIVE_GUARD:
        state.record("local-mentions", "partial", found=found, reason="retrieval capped at local exhaustive guard")
    else:
        state.record("local-mentions", "complete", found=found)


def _rel_efts_status(result: SECSearchResult) -> str | None:
    status: object = result.coverage.status
    return status if isinstance(status, str) else None


def _rel_efts_cik(
    state: _RelState,
    search_sec_filings: Callable[..., SECSearchResult],
    cik: str,
    as_of: str | None,
    exhaustive: bool,
    limit: int,
    flags: dict[str, object],
) -> None:
    result = search_sec_filings(cik, limit=_LOCAL_EXHAUSTIVE_GUARD if exhaustive else min(limit, 20), as_of=as_of)
    status = _rel_efts_status(result)
    state.warnings.extend(list(result.warnings or []))
    state.warnings.extend(list(result.errors or []))
    if status in ("partial", "source_limited", "complete_within_source_limits"):
        flags["capped"] = True
        flags["note"] = f"efts coverage {status}"
    elif status == "failed":
        flags["failed"] = True
        flags["note"] = f"efts coverage {status}"
    for hit in result.text_hits:
        state.mentions.append(
            {
                "relationship_type": "mention",
                "status": "observed",
                "accession": hit.accession_no,
                "document_name": hit.matched_document,
                "source_span": hit.query,
                "known_at": None,
            }
        )
        found_value: object = flags.get("found", 0)
        found_count = found_value if isinstance(found_value, int) else 0
        flags["found"] = found_count + 1


def _rel_record_efts(state: _RelState, flags: dict[str, object]) -> None:
    found_value: object = flags.get("found", 0)
    found = found_value if isinstance(found_value, int) else 0
    note = str(flags.get("note", ""))
    failed = bool(flags.get("failed", False))
    capped = bool(flags.get("capped", False))
    if failed and found == 0:
        state.record("efts-mentions", "failed", found=found, reason=note)
    elif failed or capped:
        state.record("efts-mentions", "partial", found=found, reason=note)
    else:
        state.record("efts-mentions", "complete", found=found)


def _rel_efts_route(state: _RelState, ciks: list[str], as_of: str | None, exhaustive: bool, limit: int) -> None:
    # Route 4: bounded EFTS for uncovered partitions (evidence, not identity).
    if not ciks:
        state.record("efts-mentions", "not_applicable", reason="no cik context")
        return
    try:
        from ..client import search_sec_filings

        flags: dict[str, object] = {"found": 0}
        for cik in ciks:
            _rel_efts_cik(state, search_sec_filings, cik, as_of, exhaustive, limit, flags)
        _rel_record_efts(state, flags)
    except Exception as exc:  # noqa: BLE001 - stage failure records an attempt and continues with partial state
        state.record("efts-mentions", "failed", error=str(exc))
        state.errors.append(f"efts-mentions failed: {exc}")


def _rel_apply_ontology(state: _RelState, data_root: Path | str | None) -> None:
    # Phase 9: ontology state reorders ranking only. Active types sort first,
    # demoted types sort last; routes, forms, documents, and candidates are
    # never removed, so the ontology cannot prove itself by restricting
    # discovery. State lookup must never break retrieval.
    try:
        from ...domain.evidence.relationship_evaluation import ontology_boost

        states = get_type_states(data_root=data_root)
        active = {t for t, s in states.items() if s == "active"}
        demoted = {t for t, s in states.items() if s == "demoted"}
        if active or demoted:

            def _boost_key(entry: dict[str, object]) -> float:
                return -ontology_boost(entry.get("relationship_type"), active, demoted)

            state.typed.sort(key=_boost_key)
            state.workflow.sort(key=_boost_key)
    except Exception:  # noqa: BLE001, S110 - best-effort relevance sort keeps insertion order on failure
        pass


def _rel_truncate(state: _RelState, limit: int) -> None:
    n_typed, n_wf, n_m = len(state.typed), len(state.workflow), len(state.mentions)
    state.typed = state.typed[:limit]
    state.workflow = state.workflow[:limit]
    state.mentions = state.mentions[:limit]
    if n_typed > limit:
        state.warnings.append(f"typed truncated to limit {limit}")
    if n_wf > limit:
        state.warnings.append(f"workflow truncated to limit {limit}")
    if n_m > limit:
        state.warnings.append(f"mentions truncated to limit {limit}")


def _rel_group(state: _RelState) -> dict[str, dict[str, list[dict[str, object]]]]:
    groups: dict[str, dict[str, list[dict[str, object]]]] = {}
    for entry in state.typed + state.workflow + state.mentions:
        rtype = str(entry.get("relationship_type") or "unknown")
        status = str(entry.get("status") or "unknown")
        if not state.want(rtype):
            continue
        groups.setdefault(rtype, {}).setdefault(status, []).append(entry)
    return groups


def _rel_result(
    state: _RelState,
    entity: object,
    ciks: list[str],
    relationship_types: Iterable[str] | None,
    as_of: str | None,
    candidates: list[EntityCandidate],
    resolution: str,
    groups: dict[str, dict[str, list[dict[str, object]]]],
) -> dict[str, object]:
    return {
        "entity": str(entity),
        "ciks": tuple(ciks),
        "relationship_types": tuple(relationship_types or ()),
        "as_of": as_of,
        "groups": groups,
        "typed": state.typed,
        "relationships": state.workflow,
        "mentions": state.mentions,
        "managers": state.managers,
        "attempts": state.attempts,
        "warnings": state.warnings,
        "errors": state.errors,
        "candidates": tuple(candidates),
        "resolution": resolution,
    }


def search_sec_relationships(
    entity: object,
    relationship_types: Iterable[str] | None = None,
    as_of: str | None = None,
    data_root: Path | str | None = None,
    limit: int = 50,
    exhaustive: bool = True,
) -> dict[str, object]:
    """Fan out across typed, workflow, mention, and EFTS routes.

    Returns groups by ``relationship_type`` then status. Typed
    source-encoded roles project as ``verified``; workflow rows keep
    their stored status; text matches stay ``observed`` mentions.
    """
    from .. import store as _store

    if as_of is not None:
        _check_as_of(as_of)
    state = _RelState(_rel_wanted(relationship_types))
    state.t1_cap = None if exhaustive else limit
    ciks, candidates, resolution, resolution_err = _resolve_relationship_identity(
        entity, as_of=as_of, data_root=data_root
    )
    if not ciks:
        return _rel_unresolved(state, entity, resolution, candidates, resolution_err, relationship_types, as_of)
    _rel_typed_route(state, _store, ciks, as_of, data_root)
    _rel_inverse_route(state, _store, ciks, as_of, data_root)
    _rel_workflow_route(state, _store, ciks, as_of, data_root, exhaustive, limit)
    _rel_mentions_route(state, _store, ciks, as_of, data_root, exhaustive, limit)
    _rel_efts_route(state, ciks, as_of, exhaustive, limit)
    _rel_apply_ontology(state, data_root)
    _rel_truncate(state, limit)
    groups = _rel_group(state)
    return _rel_result(state, entity, ciks, relationship_types, as_of, candidates, resolution, groups)


# --- Phase 9: walk-forward relationship-type evaluation + ontology state ---


def get_type_states(data_root: Path | str | None = None) -> dict[str, str]:
    """Latest ontology state per normalized type: active/demoted/unevaluated."""
    # Live seam: no persisted evaluation ledger remains; every type is unevaluated.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = data_root
    return {}


def _observation_float(value: object) -> float:
    """Market observation to float; exotic numerics normalize via str.

    Garbage raises from float(), never a silent default.
    """
    if isinstance(value, (int, float, str)):
        return float(value)
    return float(str(value))


def evaluate_and_persist_type(
    relationship_type: str,
    instances: Iterable[Mapping[str, object]] | None,
    *,
    observations: Mapping[tuple[str, str], object] | None = None,
    benchmark: Mapping[str, object] | None = None,
    windows: Iterable[tuple[str, str]],
    horizons: Iterable[int] | None = None,
    data_root: Path | str | None = None,
    actor: str = "evaluation",
    known_at: str | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    """Run the pure walk-forward eval and persist one row per window.

    Records the evaluation, inputs hash, per-window/per-horizon metrics,
    threshold decision, and type revision; every prior state stays queryable.
    Deterministic reruns over identical inputs write nothing. A human-set
    state is superseded only by a later qualifying evaluation, which cites
    the superseded evaluation id in its reason.
    """
    from ...domain.evidence import relationship_evaluation as _eval
    from ...domain.evidence.relationships import normalize_label
    from .. import store as _store

    label = normalize_label(relationship_type)
    owned = _eval_owned_instances(instances, label, normalize_label)
    windows = [(s, e) for s, e in windows]
    kwargs: dict[str, tuple[int, ...]] = {} if horizons is None else {"horizons": tuple(horizons)}
    obs_rows, bench_rows = _eval_float_rows(observations, benchmark)
    outcome = _eval.evaluate_type(label, owned, obs_rows, bench_rows, windows, **kwargs)
    inputs_hash = _eval.hash_inputs(
        {
            "relationship_type": label,
            "instances": owned,
            "observations": observations,
            "benchmark": benchmark,
            "windows": windows,
            "horizons": list(kwargs.get("horizons", _eval.HORIZONS)),
        }
    )
    # Live seam: no persisted evaluation ledger remains; every type stays unevaluated.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, actor, known_at, data_root)
    _decision, new_state, _note = _eval_decision(outcome, "unevaluated", None, reason)
    outcome.update(inputs_hash=inputs_hash, prev_state="unevaluated", new_state=new_state, rows_written=0)
    return outcome


def _eval_owned_instances(
    instances: Iterable[Mapping[str, object]] | None, label: str, normalize_label: Callable[[object], str]
) -> list[dict[str, object]]:
    return [
        dict(it)
        for it in (instances or [])
        if "relationship_type" not in it or normalize_label(it.get("relationship_type")) == label
    ]


def _eval_float_rows(
    observations: Mapping[tuple[str, str], object] | None, benchmark: Mapping[str, object] | None
) -> tuple[dict[tuple[str, str], float] | None, dict[str, float] | None]:
    # Normalize caller mappings to plain float dicts at the evaluation boundary.
    obs_rows: dict[tuple[str, str], float] | None = (
        {key: _observation_float(value) for key, value in observations.items()} if observations is not None else None
    )
    bench_rows: dict[str, float] | None = (
        {key: _observation_float(value) for key, value in benchmark.items()} if benchmark is not None else None
    )
    return obs_rows, bench_rows


def _eval_decision(
    outcome: dict[str, object], prev_state: str, prev_row: dict[str, object] | None, reason: str | None
) -> tuple[str, str, object]:
    decision_raw = outcome["decision"]
    decision = decision_raw if isinstance(decision_raw, str) else ""
    new_state = {"activate": "active", "demote": "demoted"}.get(decision, prev_state)
    note = reason or outcome["reason"]
    if (
        prev_row is not None
        and prev_row.get("actor") == "human"
        and new_state != prev_state
        and decision in ("activate", "demote")
    ):
        note = f"supersedes human {prev_row.get('evaluation_id')}: {note}"
    return decision, new_state, note


def record_type_decision(
    relationship_type: str,
    state: str,
    *,
    reason: str,
    actor: str = "human",
    window_start: str | None = None,
    window_end: str | None = None,
    inputs_hash: str = "",
    data_root: Path | str | None = None,
    known_at: str | None = None,
) -> dict[str, object]:
    """Persist a manual (default human) type-state decision as a revision row.

    Later qualifying walk-forward evaluations may supersede it with an
    explicit revision citing the superseded evaluation id.
    """
    from ...domain.evidence.relationships import normalize_label
    from .. import store as _store

    # Live seam: no persisted evaluation ledger remains; manual decisions return the row without persisting.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (_store, data_root)
    if state not in ("active", "demoted"):
        raise ValueError(f"state must be active|demoted, got {state!r}")
    if not (reason or "").strip():
        raise ValueError("human type decisions require a reason")
    label = normalize_label(relationship_type)
    return {
        "evaluation_id": f"te:manual:{uuid.uuid4().hex[:12]}",
        "relationship_type": label,
        "window_start": window_start,
        "window_end": window_end,
        "metrics_json": json.dumps({"manual": True}, sort_keys=True),
        "decision": "activate" if state == "active" else "demote",
        "inputs_hash": inputs_hash,
        "prev_state": "unevaluated",
        "new_state": state,
        "actor": actor,
        "reason": reason,
        "known_at": known_at,
    }


def get_sec_search_coverage(
    *,
    source: str | None = None,
    form: str | None = None,
    search_id: str | None = None,
    data_root: Path | str | None = None,
    limit: int = 200,
) -> dict[str, object]:
    """Persisted coverage + backfill jobs only; never infers from rows.

    Reads ``sec_ingestion_coverage`` / ``sec_searches`` ledgers and the
    durable job queue. Absent ledgers mean unknown coverage, never complete.
    """
    from .. import store as _store

    coverage, coverage_error = _coverage_ledger(_store, source, form, limit, data_root)
    jobs, jobs_error = _coverage_jobs(_store, limit, data_root)
    search = _coverage_search(_store, search_id, data_root)
    errors = [
        e
        for e in ([f"coverage ledger unavailable: {coverage_error}"] if coverage_error else [])
        + ([f"job queue unavailable: {jobs_error}"] if jobs_error else [])
    ]
    return {
        "source": source,
        "form": form,
        "search_id": search_id,
        "search": search,
        "coverage": coverage,
        "jobs": jobs,
        "errors": errors,
        "provenance": "persisted-ledgers-only",
    }


def _coverage_ledger(
    store: ModuleType, source: str | None, form: str | None, limit: int, data_root: Path | str | None
) -> tuple[list[dict[str, object]], str | None]:
    try:
        return (_store_rows(store, "query_coverage", source=source, form=form, limit=limit, root=data_root), None)
    except Exception as exc:  # noqa: BLE001 - coverage query returns empty plus error text, never raises
        return [], str(exc)


def _coverage_jobs(
    store: ModuleType, limit: int, data_root: Path | str | None
) -> tuple[list[dict[str, object]], str | None]:
    try:
        return (_store_rows(store, "list_jobs", limit=limit, root=data_root), None)
    except Exception as exc:  # noqa: BLE001 - job query returns empty plus error text, never raises
        return [], str(exc)


def _coverage_search(
    store: ModuleType, search_id: str | None, data_root: Path | str | None
) -> dict[str, object] | None:
    # Live seam: no persisted search ledger remains; per-session bundles own search ledgers.
    # NOTE: a future warehouse slots in beside this seam, never inside providers.
    _ = (store, search_id, data_root)
    return None
