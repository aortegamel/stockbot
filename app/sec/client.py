"""edgartools-only access. `edgar` is imported lazily so module import
has no side effects and never touches the network."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Provider SDK types at the boundary only; `edgar` stays lazily imported.
    from edgar import Company
    from edgar.search.efts import EFTSResult

    from .models import (
        Filing,
        SearchAttempt,
        SearchCoverage,
        SearchRun,
        SECSearchRequest,
        SECSearchResult,
        SECTextHit,
    )


@runtime_checkable
class _FrameRows(Protocol):
    """Structural frame seam: pandas frame or test double with itertuples."""

    def itertuples(self) -> Iterable[object]: ...


class SECClientError(Exception):
    """SEC EDGAR transport/parse failure; never a no-data answer."""


_initialized = False


def ensure_identity() -> None:
    """Set the SEC identity once; later calls are no-ops."""
    global _initialized
    if _initialized:
        return
    from edgar import set_identity

    from ..config import get_sec_edgar_identity

    set_identity(get_sec_edgar_identity())
    _initialized = True


def get_company(ticker_or_cik: str | int) -> Company:
    """Company handle: digits/int go by CIK, anything else by ticker."""
    ensure_identity()
    from edgar import Company

    if isinstance(ticker_or_cik, int) or (isinstance(ticker_or_cik, str) and ticker_or_cik.strip().isdigit()):
        return Company(int(str(ticker_or_cik).strip()))
    return Company(ticker_or_cik)


def resolve_cik(ticker_or_cik: str | int) -> int | None:
    """Ticker/CIK to int CIK; None on failure, never raises."""
    try:
        if isinstance(ticker_or_cik, int):
            return ticker_or_cik
        if isinstance(ticker_or_cik, str) and ticker_or_cik.strip().isdigit():
            return int(ticker_or_cik.strip())
        return get_company(ticker_or_cik).cik
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _company_tickers(ticker: object) -> list[str]:
    """Company-index ticker surface to a list; blanks and NaN become []."""
    ticker_s = "" if ticker is None else str(ticker).strip()
    if ticker_s.lower() == "nan":
        ticker_s = ""
    return [ticker_s] if ticker_s else []


def _row_to_company_candidate(row: object) -> dict[str, object] | None:
    """Index row to candidate dict; None when the CIK is non-numeric."""
    try:
        cik = int(str(getattr(row, "cik")).strip())  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    except TypeError, ValueError:
        return None
    return {
        "name": str(getattr(row, "company", "")),
        "cik": cik,
        "tickers": _company_tickers(getattr(row, "ticker", None)),
        "exchange": None,
    }


def _fetch_company_frame(query: str, limit: int) -> _FrameRows | None:
    """Company-index frame; None when the backend reports no results."""
    ensure_identity()
    from edgar.entity.search import find_company

    results = find_company(query, top_n=limit)
    frame: object = getattr(results, "results", None)
    if frame is None or getattr(frame, "empty", False):
        return None
    if not isinstance(frame, _FrameRows):
        return None
    return frame


def _check_find_params(query: str, limit: int) -> str:
    """Validated issuer query; raises on blank query or non-positive limit."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"invalid query: {query!r}")
    if limit < 1:
        raise ValueError(f"invalid limit: {limit!r}")
    return query


def _collect_company_candidates(frame: _FrameRows, limit: int) -> list[dict[str, object]]:
    """Index frame to bounded candidate list; skips non-numeric CIK rows."""
    out: list[dict[str, object]] = []
    for row in frame.itertuples():
        candidate = _row_to_company_candidate(row)
        if candidate is None:
            continue
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


def find_sec_company(query: str, limit: int = 10) -> list[dict[str, object]]:
    """Issuer name to candidate CIKs via the edgartools company index."""
    _check_find_params(query, limit)
    frame = _fetch_company_frame(query, limit)
    if frame is None:
        return []
    return _collect_company_candidates(frame, limit)


def _utcnow() -> str:
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_search_query(query: str) -> str:
    """Stripped query; raises on blank or non-string input."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"invalid query: {query!r}")
    return query.strip()


def _check_search_limit(limit: int) -> int:
    """Caller result bound; raises unless a positive int."""
    if limit < 1:
        raise ValueError(f"invalid limit: {limit!r}")
    return limit


def _check_search_cik(cik: int | str | None) -> int | None:
    """Issuer-scope CIK as int; None passes through, garbage raises."""
    if cik is None:
        return None
    if isinstance(cik, int):
        return cik
    text = str(cik).strip()
    if text.isdigit():
        return int(text)
    raise ValueError(f"invalid cik: {cik!r}")


def _check_search_ticker(ticker: str | None) -> str | None:
    """Issuer-scope ticker uppercased; None/blank passes through as None."""
    if ticker is None:
        return None
    return str(ticker).strip().upper() or None


def _check_search_as_of(as_of: str | None) -> str | None:
    """Normalized as_of via the filings PIT checker; None passes through."""
    if as_of is None:
        return None
    from .filings import _check_as_of

    return _check_as_of(as_of)


def _search_filters(
    forms: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    as_of: str | None,
    cik: int | str | None = None,
    ticker: str | None = None,
) -> dict[str, object]:
    """Non-null search filters for the attempt ledger (issuer scope included)."""
    filters: dict[str, object] = {
        key: value
        for key, value in (
            ("forms", list(forms) if forms else None),
            ("start_date", start_date),
            ("end_date", end_date),
            ("as_of", as_of),
            ("cik", str(cik) if cik is not None else None),
            ("ticker", ticker),
        )
        if value is not None
    }
    return filters


def _normalize_search_params(
    query: str,
    forms: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    as_of: str | None,
    cik: int | None = None,
    ticker: str | None = None,
) -> tuple[str, str, SECSearchRequest, dict[str, object]]:
    """Validate + normalize search inputs; returns (query, search_id, request, filters)."""
    from .models import SECSearchRequest

    query = _check_search_query(query)
    limit = _check_search_limit(limit)
    as_of = _check_search_as_of(as_of)
    import uuid

    search_id = uuid.uuid4().hex[:12]
    request = SECSearchRequest(
        query=query,
        ticker=ticker,
        cik=str(cik) if cik is not None else None,
        forms=tuple(forms) if forms else None,
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
        max_results=limit,
    )
    return query, search_id, request, _search_filters(forms, start_date, end_date, as_of, cik, ticker)


def _record_attempt(
    attempts: list[SearchAttempt],
    search_id: str,
    query: str,
    filters: dict[str, object],
    page_num: int,
    status: Literal["complete", "source_limited", "partial", "failed", "not_applicable"],
    reported: int,
    retrieved: int,
    **extra: str,
) -> None:
    """Append one EFTS page attempt (one timestamp, one page)."""
    from .models import SearchAttempt

    now = _utcnow()
    attempts.append(
        SearchAttempt(
            attempt_id=f"{search_id}-p{page_num}",
            search_id=search_id,
            backend="efts",
            query=query,
            filters={**filters, "page": page_num},
            started_at=now,
            completed_at=now,
            status=status,
            results_reported=reported,
            results_retrieved=retrieved,
            pages_retrieved=1,
            truncated=status in ("partial", "source_limited"),
            pit_basis="filed_at",
            **extra,
        )
    )


def _coerce_reported_total(page: object) -> int:
    """EFTS reported total as int; falls back to page length, then zero."""
    reported = getattr(page, "total", None)
    if reported is None:
        reported = len(getattr(page, "results", None) or [])
    try:
        return int(reported)
    except TypeError, ValueError:
        return 0


def _fetch_first_page(
    query: str,
    forms: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    cik: int | None = None,
    ticker: str | None = None,
) -> object:
    """First EFTS page, optionally scoped to one filer corpus; transport failure propagates."""
    ensure_identity()
    from edgar.search.efts import search_filings

    return search_filings(
        query,
        forms=forms,
        start_date=start_date,
        end_date=end_date,
        limit=min(limit, 100),
        cik=cik,
        ticker=(ticker if cik is None else None),
    )


def _failed_search_result(
    search_id: str,
    request: SECSearchRequest,
    attempts: list[SearchAttempt],
    warnings: list[str],
    errors: list[str],
    start_date: str | None,
    end_date: str | None,
    forms: list[str] | None,
) -> SECSearchResult:
    """Failed-coverage packet; transport failure never becomes zero matches."""
    from .models import SearchCoverage, SECSearchResult

    return SECSearchResult(
        search_id=search_id,
        request=request,
        coverage=SearchCoverage(
            status="failed",
            sources_attempted=("efts",),
            sources_failed=("efts",),
            date_coverage=f"{start_date or ''}:{end_date or ''}",
            forms_covered=tuple(forms) if forms else (),
        ),
        attempts=tuple(attempts),
        warnings=tuple(warnings),
        errors=tuple(errors),
        retrieval_order=("efts",),
        search_runs=(
            _search_run(
                search_id,
                request.query or "",
                _search_filters(forms, start_date, end_date, request.as_of, request.cik, request.ticker),
                request.as_of,
                [],
            ),
        ),
    )


def _pit_excluded(filed: str, as_of: str | None) -> bool:
    """Point-in-time exclusion: missing filed date or filed after as_of."""
    return as_of is not None and (not filed or filed > as_of)


def _hit_resource_uri(accession_no: str, document: object) -> str:
    """Stable document address mirroring documents._source_uri_for (no import)."""
    doc = document if isinstance(document, str) and document else "primary"
    return f"source://sec/{accession_no}/{doc}"


def _scope_excluded(filer_cik: int | None, issuer_cik: int | None) -> bool:
    """Out-of-corpus hit: scope set but the filer CIK is missing or different."""
    return issuer_cik is not None and filer_cik != issuer_cik


def _warn_scope_gaps(
    warnings: list[str],
    scope_gaps: int,
    issuer_cik: int | None,
) -> None:
    """Issuer-scope exclusion warning; silent when everything matched scope."""
    if scope_gaps > 0:
        warnings.append(f"{scope_gaps} EFTS hit(s) outside filer scope CIK {issuer_cik} excluded")


def _accept_hit(
    query: str,
    text_hit: SECTextHit,
    seen: set[tuple[str, str, str | None]],
    hits: list[SECTextHit],
) -> bool:
    """Append unseen (query, accession, document) hits; True when accepted."""
    key = (query, text_hit.accession_no, text_hit.matched_document)
    if key in seen:
        return False
    seen.add(key)
    hits.append(text_hit)
    return True


def _map_one_hit(
    hit: EFTSResult,
    search_id: str,
    page_num: int,
    query: str,
    as_of: str | None,
    hits: list[SECTextHit],
    seen: set[tuple[str, str, str | None]],
    issuer_cik: int | None = None,
) -> tuple[int, int, int]:
    """Map one EFTS hit; returns (retrieved, pit_gaps, scope_gaps) increments."""
    text_hit = _hit_to_text_hit(search_id, f"{search_id}-p{page_num}", query, hit, page_num, issuer_cik=issuer_cik)
    if _pit_excluded((text_hit.filed_at or "")[:10], as_of):
        return 0, 1, 0
    if _scope_excluded(text_hit.filer_cik, issuer_cik):
        return 0, 0, 1
    return (1, 0, 0) if _accept_hit(query, text_hit, seen, hits) else (0, 0, 0)


def _map_page_hits(
    results: list[EFTSResult],
    search_id: str,
    page_num: int,
    query: str,
    as_of: str | None,
    limit: int,
    hits: list[SECTextHit],
    seen: set[tuple[str, str, str | None]],
    issuer_cik: int | None = None,
) -> tuple[int, int, int]:
    """Map one EFTS page to text hits; returns (retrieved, pit_gaps, scope_gaps)."""
    retrieved = 0
    pit_gaps = 0
    scope_gaps = 0
    for hit in results:
        if len(hits) >= limit:
            break
        got, pit, scope = _map_one_hit(hit, search_id, page_num, query, as_of, hits, seen, issuer_cik)
        retrieved += got
        pit_gaps += pit
        scope_gaps += scope
    return retrieved, pit_gaps, scope_gaps


def _drain_one_page(
    page: object,
    page_num: int,
    query: str,
    search_id: str,
    filters: dict[str, object],
    as_of: str | None,
    limit: int,
    reported: int,
    attempts: list[SearchAttempt],
    hits: list[SECTextHit],
    seen: set[tuple[str, str, str | None]],
    errors: list[str],
    issuer_cik: int | None = None,
) -> tuple[object | None, int, bool, bool, int, int]:
    """Drain one EFTS page; returns (page, retrieved, failed, capped, pit_gaps, scope_gaps)."""
    results = getattr(page, "results", None) or []
    if not results:
        return None, 0, False, False, 0, 0
    retrieved, pit_gaps, scope_gaps = _map_page_hits(
        results, search_id, page_num, query, as_of, limit, hits, seen, issuer_cik
    )
    if len(hits) >= limit or len(hits) + pit_gaps + scope_gaps >= reported:
        return None, retrieved, False, False, pit_gaps, scope_gaps
    nxt, failed_tail, source_capped = _advance_page(
        page, page_num, reported, attempts, search_id, query, filters, errors
    )
    return nxt, retrieved, failed_tail, source_capped, pit_gaps, scope_gaps


def _drain_recorder(
    attempts: list[SearchAttempt],
    search_id: str,
    query: str,
    filters: dict[str, object],
    reported: int,
) -> Callable[..., None]:
    """Attempt recorder bound to one drain's ledger context."""

    def _attempt(
        page_no: int,
        status: Literal["complete", "source_limited", "partial", "failed", "not_applicable"],
        retrieved: int,
        **extra: str,
    ) -> None:
        _record_attempt(attempts, search_id, query, filters, page_no, status, reported, retrieved, **extra)

    return _attempt


def _apply_drain_step(
    page: object,
    page_num: int,
    query: str,
    search_id: str,
    filters: dict[str, object],
    as_of: str | None,
    limit: int,
    reported: int,
    attempts: list[SearchAttempt],
    hits: list[SECTextHit],
    seen: set[tuple[str, str, str | None]],
    errors: list[str],
    record: Callable[..., None],
    issuer_cik: int | None = None,
) -> tuple[object | None, int, int, bool, bool]:
    """One drain iteration: step the page, record the attempt, sum gaps."""
    page, retrieved, failed_tail, source_capped, pit_gaps, scope_gaps = _drain_one_page(
        page, page_num, query, search_id, filters, as_of, limit, reported, attempts, hits, seen, errors, issuer_cik
    )
    record(page_num, "complete", retrieved)
    return page, pit_gaps, scope_gaps, failed_tail, source_capped


_DRAIN_BUDGET_S = 30.0


def _drain_pages(
    page: object,
    query: str,
    search_id: str,
    filters: dict[str, object],
    as_of: str | None,
    limit: int,
    reported: int,
    attempts: list[SearchAttempt],
    hits: list[SECTextHit],
    seen: set[tuple[str, str, str | None]],
    warnings: list[str],
    errors: list[str],
    issuer_cik: int | None = None,
) -> tuple[int, int, int, bool, bool]:
    """Drain EFTS pages within a 30s between-page budget; returns (page_num, pit_gaps, scope_gaps, failed_tail, source_capped)."""
    page_num = 0
    pit_gaps = 0
    scope_gaps = 0
    source_capped = False
    failed_tail = False

    _attempt = _drain_recorder(attempts, search_id, query, filters, reported)
    start = time.monotonic()

    while page is not None and len(hits) < limit:
        if page_num > 0 and time.monotonic() - start >= _DRAIN_BUDGET_S:
            warnings.append(
                f"EFTS drain stopped after {_DRAIN_BUDGET_S:.0f}s; {len(hits)} of {reported} hits retrieved"
            )
            break
        page_num += 1
        page, pit, scope, failed_tail, source_capped = _apply_drain_step(
            page,
            page_num,
            query,
            search_id,
            filters,
            as_of,
            limit,
            reported,
            attempts,
            hits,
            seen,
            errors,
            _attempt,
            issuer_cik,
        )
        pit_gaps += pit
        scope_gaps += scope
        page = None if (failed_tail or source_capped) else page
    _warn_pit_gaps(warnings, pit_gaps, as_of)
    _warn_scope_gaps(warnings, scope_gaps, issuer_cik)
    return page_num, pit_gaps, scope_gaps, failed_tail, source_capped


def _advance_page(
    page: object,
    page_num: int,
    reported: int,
    attempts: list[SearchAttempt],
    search_id: str,
    query: str,
    filters: dict[str, object],
    errors: list[str],
) -> tuple[object | None, bool, bool]:
    """Next EFTS page; returns (page, failed_tail, source_capped).

    A page without a callable ``next`` drains as terminal: returns
    ``(None, False, False)`` so the ``while page is not None`` loop exits
    after the current page is recorded (never reprocesses it).
    """
    nxt = getattr(page, "next", None)
    if not callable(nxt):
        return None, False, False
    try:
        nxt_page = nxt()
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _record_attempt(
            attempts,
            search_id,
            query,
            filters,
            page_num + 1,
            "failed",
            reported,
            0,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        errors.append(f"efts page {page_num + 1} failed: {exc}")
        return page, True, False
    if nxt_page is None:
        return None, False, True
    return nxt_page, False, False


def _search_failed(
    hits: list[SECTextHit], pit_gaps: int, scope_gaps: int, reported: int, failed_tail: bool, limit: int
) -> bool:
    """Caller-bound failure: transport tail loss or reporting over the bound."""
    return failed_tail or (len(hits) + pit_gaps + scope_gaps < reported and len(hits) >= limit)


def _search_drained(hits: list[SECTextHit], pit_gaps: int, scope_gaps: int, reported: int, source_capped: bool) -> bool:
    """Drain completeness without a source cap; empty/empty counts as drained."""
    return (len(hits) + pit_gaps + scope_gaps >= reported or (not hits and not reported)) and not source_capped


def _resolve_search_status(
    hits: list[SECTextHit],
    pit_gaps: int,
    scope_gaps: int,
    reported: int,
    source_capped: bool,
    failed_tail: bool,
    limit: int,
) -> Literal["complete", "complete_within_source_limits", "partial", "failed"]:
    """Coverage status from drain outcome (caller bound vs source cap vs failure)."""
    if _search_failed(hits, pit_gaps, scope_gaps, reported, failed_tail, limit):
        return "partial"
    if _search_drained(hits, pit_gaps, scope_gaps, reported, source_capped):
        return "complete"
    if source_capped or reported > 10_000:
        return "complete_within_source_limits"
    return "partial"


def _hit_filer_cik(hit: EFTSResult) -> int | None:
    """EFTS hit CIK as int; None when missing or non-numeric."""
    try:
        return int(str(getattr(hit, "cik", None)).strip())
    except TypeError, ValueError:
        return None


def _hit_score(hit: EFTSResult) -> float:
    """EFTS hit score as float; zero when missing or non-numeric."""
    try:
        return float(getattr(hit, "score", 0.0) or 0.0)
    except TypeError, ValueError:
        return 0.0


def _hit_items(hit: EFTSResult) -> tuple[str, ...]:
    """EFTS hit items as tuple of str; empty when missing or non-iterable."""
    try:
        items = tuple(getattr(hit, "items", None) or ())
    except TypeError:
        return ()
    return tuple(str(i) for i in items)


def _hit_to_text_hit(
    search_id: str,
    attempt_id: str,
    query: str,
    hit: EFTSResult,
    page_num: int,
    issuer_cik: int | None = None,
) -> SECTextHit:
    """EFTS hit -> SECTextHit; never infers identity beyond the filer."""
    from .models import SECTextHit

    company = getattr(hit, "company", None)
    accession_no = str(getattr(hit, "accession_number", ""))
    matched_document = getattr(hit, "document_id", None)
    return SECTextHit(
        search_id=search_id,
        attempt_id=attempt_id,
        query=query,
        accession_no=accession_no,
        form=str(getattr(hit, "form", "")),
        filed_at=str(getattr(hit, "filed", "") or ""),
        filer_cik=_hit_filer_cik(hit),
        filer_name=None if company is None else str(company),
        matched_document=matched_document,
        issuer_cik=issuer_cik,
        resource_uri=_hit_resource_uri(accession_no, matched_document),
        file_type=getattr(hit, "file_type", None),
        file_description=getattr(hit, "file_description", None),
        items=_hit_items(hit),
        sic=getattr(hit, "sic", None),
        location=getattr(hit, "location", None),
        state=getattr(hit, "state", None),
        inc_state=getattr(hit, "inc_state", None),
        score=_hit_score(hit),
        source_url=None,
        page=page_num,
    )


def _warn_pit_gaps(
    warnings: list[str],
    pit_gaps: int,
    as_of: str | None,
) -> None:
    """Point-in-time exclusion warning; silent when nothing excluded."""
    if pit_gaps > 0:
        warnings.append(f"{pit_gaps} EFTS hit(s) excluded by as_of {as_of} (no usable filed date or filed after as_of)")


def _coverage_limits(
    status: Literal["complete", "complete_within_source_limits", "partial", "failed"],
    warnings: list[str],
    reported: int,
    retrieved: int,
) -> tuple[str, ...]:
    """Source-limit marker for capped coverage; warns with reported/retrieved."""
    if status != "complete_within_source_limits":
        return ()
    warnings.append(f"EFTS reports {reported} hits; retrieved {retrieved} within source limits")
    return ("efts:deep-pagination-cap",)


def _coverage_status_block(
    status: Literal["complete", "complete_within_source_limits", "partial", "failed"],
    failed_tail: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(completed, failed) source tuples for the coverage block."""
    completed = ("efts",) if status != "failed" else ()
    return completed, ("efts",) if failed_tail else ()


def _coverage_counts(
    start_date: str | None,
    end_date: str | None,
    forms: list[str] | None,
) -> tuple[str | None, tuple[str, ...]]:
    """(date_coverage, forms_covered) projections for the coverage block."""
    return (
        f"{start_date or ''}:{end_date or ''}",
        tuple(forms) if forms else (),
    )


def _search_coverage(
    status: Literal["complete", "complete_within_source_limits", "partial", "failed"],
    failed_tail: bool,
    reported: int,
    retrieved: int,
    page_num: int,
    start_date: str | None,
    end_date: str | None,
    forms: list[str] | None,
    limits: tuple[str, ...],
) -> SearchCoverage:
    """Coverage block for the final search packet."""
    from .models import SearchCoverage

    completed, failed = _coverage_status_block(status, failed_tail)
    date_coverage, forms_covered = _coverage_counts(start_date, end_date, forms)
    return SearchCoverage(
        status=status,
        sources_attempted=("efts",),
        sources_completed=completed,
        sources_failed=failed,
        source_limits=limits,
        results_reported=reported,
        results_retrieved=retrieved,
        pages=page_num,
        date_coverage=date_coverage,
        forms_covered=forms_covered,
    )


def _search_run(
    search_id: str,
    query: str,
    filters: dict[str, object],
    as_of: str | None,
    hits: list[SECTextHit] | tuple[SECTextHit, ...],
) -> SearchRun:
    """One SearchRun row: executed-at now, PIT basis, match counts."""
    from .models import SearchRun

    return SearchRun(
        id=search_id,
        source="efts",
        query=query,
        filters=dict(filters),
        executed_at=_utcnow(),
        as_of=as_of,
        matched_entities=0,
        matched_documents=len({h.accession_no for h in hits}),
        matched_passages=len(hits),
    )


def _build_search_result(
    search_id: str,
    request: SECSearchRequest,
    hits: tuple[SECTextHit, ...] | list[SECTextHit],
    attempts: tuple[SearchAttempt, ...] | list[SearchAttempt],
    warnings: list[str],
    errors: list[str],
    reported: int,
    page_num: int,
    failed_tail: bool,
    start_date: str | None,
    end_date: str | None,
    forms: list[str] | None,
    status: Literal["complete", "complete_within_source_limits", "partial", "failed"],
) -> SECSearchResult:
    """Assemble the final search packet with coverage and attempt ledger."""
    from .models import SECSearchResult

    limits = _coverage_limits(status, warnings, reported, len(hits))
    runs = (
        _search_run(
            search_id,
            request.query or "",
            _search_filters(forms, start_date, end_date, request.as_of, request.cik, request.ticker),
            request.as_of,
            list(hits),
        ),
    )
    return SECSearchResult(
        search_id=search_id,
        request=request,
        text_hits=tuple(hits),
        coverage=_search_coverage(
            status, failed_tail, reported, len(hits), page_num, start_date, end_date, forms, limits
        ),
        attempts=tuple(attempts),
        warnings=tuple(warnings),
        errors=tuple(errors),
        retrieval_order=("efts",),
        search_runs=runs,
    )


def search_sec_filings(
    query: str,
    forms: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
    as_of: str | None = None,
    *,
    cik: int | str | None = None,
    ticker: str | None = None,
) -> SECSearchResult:
    """EDGAR full-text search over one filer corpus when scoped, else global.

    Text hits, never inferred identity. Paginates EFTS until the reported
    total, an empty page, the caller limit, a documented source cap, or
    failure. A caller bound yields ``partial``; a documented EFTS cap yields
    ``complete_within_source_limits``; transport failure never becomes zero
    matches. Point-in-time and filer-scope exclusions warn, never silently drop.
    """
    cik = _check_search_cik(cik)
    ticker = _check_search_ticker(ticker)
    query, search_id, request, filters = _normalize_search_params(
        query, forms, start_date, end_date, limit, as_of, cik, ticker
    )
    as_of = request.as_of
    issuer_cik = cik if cik is not None else (resolve_cik(ticker) if ticker else None)
    attempts: list[SearchAttempt] = []
    hits: list[SECTextHit] = []
    warnings: list[str] = []
    errors: list[str] = []
    seen: set[tuple[str, str, str | None]] = set()
    try:
        page = _fetch_first_page(query, forms, start_date, end_date, limit, cik, ticker)
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _record_attempt(
            attempts,
            search_id,
            query,
            filters,
            0,
            "failed",
            0,
            0,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        errors.append(f"efts page 1 failed: {exc}")
        return _failed_search_result(search_id, request, attempts, warnings, errors, start_date, end_date, forms)
    reported = _coerce_reported_total(page)
    page_num, pit_gaps, scope_gaps, failed_tail, source_capped = _drain_pages(
        page, query, search_id, filters, as_of, limit, reported, attempts, hits, seen, warnings, errors, issuer_cik
    )
    status = _resolve_search_status(hits, pit_gaps, scope_gaps, reported, source_capped, failed_tail, limit)
    return _build_search_result(
        search_id,
        request,
        hits,
        attempts,
        warnings,
        errors,
        reported,
        page_num,
        failed_tail,
        start_date,
        end_date,
        forms,
        status,
    )


def _normalize_lookup_text(value: object) -> str:
    """NFKD-casefold alphanumeric normalization for legal-name matching."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[\W_]+", " ", text)).strip()


_LOOKUP_TTL_S = 86400.0
_lookup_lock = threading.Lock()
_lookup_cached_at = 0.0
_lookup_cached_frame: _FrameRows | None = None
_lookup_cached_index: list[tuple[str, str, int]] | None = None


def _cached_lookup_index() -> list[tuple[str, str, int]] | None:
    """Cached (normalized name, raw name, cik) rows when fresh; None when stale."""
    with _lookup_lock:
        if _lookup_cached_index is None:
            return None
        if time.monotonic() - _lookup_cached_at >= _LOOKUP_TTL_S:
            return None
        return _lookup_cached_index


def _store_lookup_index(rows: list[tuple[str, str, int]]) -> None:
    """Cache one normalized lookup index with a fresh timestamp."""
    global _lookup_cached_at, _lookup_cached_index
    with _lookup_lock:
        _lookup_cached_index = rows
        _lookup_cached_at = time.monotonic()


def _cached_lookup_frame() -> _FrameRows | None:
    """Cached lookup frame when fresh; None when stale or empty."""
    with _lookup_lock:
        if _lookup_cached_frame is None:
            return None
        if time.monotonic() - _lookup_cached_at >= _LOOKUP_TTL_S:
            return None
        return _lookup_cached_frame


def _store_lookup_frame(frame: _FrameRows) -> None:
    """Cache one lookup frame with a fresh timestamp."""
    global _lookup_cached_at, _lookup_cached_frame
    with _lookup_lock:
        _lookup_cached_frame = frame
        _lookup_cached_at = time.monotonic()


def _fetch_lookup_frame(query: str) -> _FrameRows:
    """CIK lookup dataset, one-day process cache; fetch failure raises (never zero-result)."""
    hit = _cached_lookup_frame()
    if hit is not None:
        return hit
    try:
        ensure_identity()
        from edgar.entity.tickers import get_cik_lookup_data

        frame: object = get_cik_lookup_data()
        if not isinstance(frame, _FrameRows):
            raise SECClientError(f"cik lookup parse failed for {query!r}: bad frame")
        _store_lookup_frame(frame)
        return frame
    except SECClientError:
        raise
    except Exception as exc:
        raise SECClientError(f"cik lookup fetch failed for {query!r}: {exc}") from exc


def _rank_lookup_name(normed: str, want: str) -> int | None:
    """Match rank: exact 0, prefix 1, substring 2, else None."""
    if normed == want:
        return 0
    if normed.startswith(want):
        return 1
    if want in normed:
        return 2
    return None


def _build_lookup_index(frame: _FrameRows) -> list[tuple[str, str, int]]:
    """Normalized (normed, raw, cik) rows built once per cached frame."""
    rows: list[tuple[str, str, int]] = []
    for row in frame.itertuples():
        try:
            cik = int(str(getattr(row, "cik")).strip())  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
        except TypeError, ValueError:
            continue
        name = str(getattr(row, "name", ""))
        try:
            rows.append((_normalize_lookup_text(name), name, cik))
        except Exception:  # noqa: BLE001, S112 - one bad name never breaks the index
            continue
    return rows


def _scan_lookup_index(index: list[tuple[str, str, int]], want: str) -> list[tuple[int, str, int]]:
    """Deterministic rank over precomputed normalized names."""
    rows: list[tuple[int, str, int]] = []
    for normed, name, cik in index:
        rank = _rank_lookup_name(normed, want)
        if rank is not None:
            rows.append((rank, name, cik))
    return rows


def _check_lookup_params(query: str, limit: int) -> str:
    """Validated lookup query; raises on blank query or non-positive limit."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"invalid query: {query!r}")
    if limit < 1:
        raise ValueError(f"invalid limit: {limit!r}")
    return query


def _scan_lookup_rows(frame: _FrameRows, query: str, want: str) -> list[tuple[int, str, int]]:
    """Normalized scan over the precomputed index; parse failure raises."""
    try:
        if not want:
            return []
        cached = _cached_lookup_index()
        if cached is not None:
            return _scan_lookup_index(cached, want)
        index = _build_lookup_index(frame)
        _store_lookup_index(index)
        return _scan_lookup_index(index, want)
    except Exception as exc:
        raise SECClientError(f"cik lookup parse failed for {query!r}: {exc}") from exc


def _format_lookup_rows(
    rows: list[tuple[int, str, int]],
    limit: int,
) -> list[dict[str, object]]:
    """Ranked rows to bounded candidate dicts."""
    rows.sort()
    out: list[dict[str, object]] = []
    for _, name, cik in rows[:limit]:
        out.append({"name": name, "cik": cik, "tickers": [], "exchange": None})
    return out


def get_cik_lookup_candidates(query: str, limit: int = 10) -> list[dict[str, object]]:
    """General legal-name to CIK candidates via SEC ``cik-lookup-data.txt``.

    Unlike the ticker-company index behind :func:`find_sec_company`, this
    dataset includes no-ticker registrants. Matching is a deterministic
    normalized substring scan (exact, then prefix, then substring).
    Fetch failure raises ``SECClientError`` (never a zero-result assertion);
    ``limit``/blank violations raise ``ValueError`` like the other adapters.
    """
    _check_lookup_params(query, limit)
    frame = _fetch_lookup_frame(query)
    rows = _scan_lookup_rows(frame, query, _normalize_lookup_text(query))
    return _format_lookup_rows(rows, limit)


def _maybe_str(value: object) -> str | None:
    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    return text or None


def _address_dict(address: object) -> dict[str, object | None]:
    """Best-effort address projection over dict or attribute surfaces."""

    def _get(name: str) -> object:
        if isinstance(address, dict):
            return address.get(name)
        return getattr(address, name, None)

    out: dict[str, object | None] = {}
    for key in ("street1", "street2", "city", "stateOrCountry", "stateOrCountryDescription", "zipCode"):
        try:
            out[key] = _maybe_str(_get(key))
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            out[key] = None
    return out


def _parse_submissions_cik(cik: int | str) -> int | None:
    """CIK text to int; None when not a plain integer."""
    try:
        return int(str(cik).strip())
    except TypeError, ValueError, AttributeError:
        return None


def _fetch_submissions_data(cik_int: int, cik: int | str) -> object | None:
    """SEC submissions payload; None only for unknown CIK, else raises."""
    try:
        ensure_identity()
        from edgar.entity.submissions import get_entity_submissions

        return get_entity_submissions(cik_int)
    except Exception as exc:
        if "404" in str(exc):
            return None
        raise SECClientError(f"submissions fetch failed for CIK {cik!r}: {exc}") from exc


def _clean_str_list(raw: object) -> list[str]:
    """Strip-nonempty string list; empty when non-iterable."""
    if raw is None:
        return []
    items: list[object] = list(raw) if isinstance(raw, (list, tuple)) else []
    return [str(t).strip() for t in items if str(t).strip()]


def _parse_former_names(data: object) -> list[dict[str, object | None]]:
    """Former-name rows with SEC from/to dates; skips non-dict entries."""
    former: list[dict[str, object | None]] = []
    for entry in getattr(data, "former_names", None) or []:
        if isinstance(entry, dict):
            get = entry.get
            former.append(
                {
                    "name": _maybe_str(get("name")),
                    "from": _maybe_str(get("from")),
                    "to": _maybe_str(get("to")),
                    "type": _maybe_str(get("type")),
                }
            )
    return former


def _parse_filing_history(data: object) -> list[dict[str, object | None]]:
    """Light filing history (first 5); empty when the surface is unusable."""
    history: list[dict[str, object | None]] = []
    try:
        filings = getattr(data, "filings", None)
        candidates: list[object] | None = getattr(filings, "data", filings)
        if candidates is None:
            candidates = []
        for item in list(candidates)[:5]:
            history.append(
                {
                    "form": _maybe_str(getattr(item, "form", None)),
                    "filed_at": _maybe_str(getattr(item, "filing_date", None)),
                    "accession_no": _maybe_str(getattr(item, "accession_number", None)),
                }
            )
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        history = []
    return history


def _parse_sic(data: object) -> str | None:
    """SIC as stripped text; None when missing, blank, or non-stringable."""
    sic = getattr(data, "sic", None)
    try:
        return None if sic is None else str(sic).strip() or None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _assemble_submissions_metadata(
    cik: int | str,
    cik_int: int,
    data: object,
) -> dict[str, object]:
    """Map the submissions payload to the metadata contract."""
    return {
        "cik": cik_int,
        "name": _maybe_str(getattr(data, "name", None)),
        "tickers": _clean_str_list(getattr(data, "tickers", None)),
        "exchanges": _clean_str_list(getattr(data, "exchanges", None)),
        "sic": _parse_sic(data),
        "sic_description": _maybe_str(getattr(data, "sic_description", None)),
        "entity_type": _maybe_str(getattr(data, "entity_type", None)),
        "state_of_incorporation": _maybe_str(getattr(data, "state_of_incorporation", None)),
        "business_address": _address_dict(getattr(data, "business_address", None)),
        "mailing_address": _address_dict(getattr(data, "mailing_address", None)),
        "former_names": _parse_former_names(data),
        "filing_history": _parse_filing_history(data),
    }


def get_submissions_metadata(cik: int | str) -> dict[str, object] | None:
    """Authoritative per-CIK metadata from SEC submissions.

    Returns current name, ``formerNames`` (with SEC ``from``/``to`` dates),
    tickers, SIC, state of incorporation, addresses, and a light filing
    history. Unknown CIK (definitive SEC not-found/empty) returns ``None``;
    transport or schema failure raises ``SECClientError`` and never becomes
    a zero-data assertion.
    """
    cik_int = _parse_submissions_cik(cik)
    if cik_int is None:
        return None
    data = _fetch_submissions_data(cik_int, cik)
    if data is None:
        return None
    try:
        return _assemble_submissions_metadata(cik, cik_int, data)
    except Exception as exc:
        raise SECClientError(f"submissions parse failed for CIK {cik!r}: {exc}") from exc


def _normalize_feed_items(feed: object) -> list[Filing]:
    """Edgar feed to normalized filings; None/non-iterable becomes []."""
    from .normalization import filing_from_edgar

    if feed is None:
        return []
    if not isinstance(feed, (list, tuple)):
        return []
    out: list[Filing] = []
    for item in feed:
        try:
            out.append(filing_from_edgar(item))
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
    return out


def get_global_filings(
    year: int | list[int] | range | None = None,
    quarter: int | list[int] | range | None = None,
    form: str | list[str | int] | None = None,
    filing_date: str | None = None,
    *,
    amendments: bool = True,
) -> list[Filing]:
    """Global quarterly filing index -> normalized ``Filing`` list.

    Thin wrapper over installed ``edgar.get_filings`` for 1993+ quarterly
    partitions. Arbitrary form strings pass straight through (never an
    allowlist); ``None`` from edgar becomes ``[]``. Transport/parse failure
    raises so the caller records a failed attempt (never silent zero
    matches); per-row normalization failures are skipped.
    """
    ensure_identity()
    import edgar

    return _normalize_feed_items(
        edgar.get_filings(year, quarter, form=form, amendments=amendments, filing_date=filing_date)
    )


def get_current_filings(
    form: str = "",
    *,
    page_size: int | None = 40,
    owner: str = "include",
) -> list[Filing]:
    """Current-quarter SEC feed -> normalized ``Filing`` list.

    Thin wrapper over installed ``edgar.get_current_filings`` (near
    real-time, ~last 24h). Same failure contract as :func:`get_global_filings`.
    """
    ensure_identity()
    import edgar

    return _normalize_feed_items(edgar.get_current_filings(form=form or "", page_size=page_size, owner=owner))
