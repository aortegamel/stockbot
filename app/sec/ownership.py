"""Deterministic SC 13D/G beneficial-ownership normalization (no network).

Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind these live readers, never inside normalization.
"""

from __future__ import annotations

import itertools
from datetime import date, datetime
from typing import TYPE_CHECKING

from .models import BeneficialOwnership, Filing, OwnershipChangeEvent

if TYPE_CHECKING:
    # Provider SDK schedules share no common base, and tests exercise this
    # boundary with SimpleNamespace doubles, so both shapes are named.
    from types import SimpleNamespace

    from edgar import Filing as EdgarFiling
    from edgar.beneficial_ownership import Schedule13D, Schedule13G

_FORMS_13D = ("SC 13D", "SC 13D/A")
_FORMS_13G = ("SC 13G", "SC 13G/A")
_DEFAULT_FORMS = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A")


def list_sec_filings(
    ticker_or_cik: str | int,
    forms: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    as_of: str | date | datetime | None = None,
    limit: int | None = 50,
) -> list[Filing]:
    """Lazy seam: tests monkeypatch this name; real path imports on call."""
    from .filings import list_sec_filings as _real

    return _real(ticker_or_cik, forms=forms, start_date=start_date, end_date=end_date, as_of=as_of, limit=limit)


def _safe_int(value: object) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in ("none", "nan", "na", "n/a", "--"):
            return None
        return int(float(text)) if "." in text else int(text)
    except ValueError, TypeError:
        return None


def _safe_float(value: object) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip().replace(",", "").rstrip("%")
        if not text or text.lower() in ("none", "nan", "na", "n/a", "--"):
            return None
        return float(text)
    except ValueError, TypeError:
        return None


def _first(obj: object, *names: str) -> object:
    """First present attribute; the SDK/test-double schedules probed here are
    dynamically shaped, so callers validate the result (`_safe_int`,
    `_safe_float`, `str`)."""
    for name in names:
        try:
            value: object = getattr(obj, name)
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
        if value is not None:
            return value
    return None


def _named_purpose_of(items: object) -> str | None:
    for attr in ("purpose_of_transaction", "purpose"):
        try:
            value = getattr(items, attr)
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
        if value:
            try:
                return str(value)
            except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
                continue
    return None


def _flat_purpose_of(items: object) -> str | None:
    try:
        text = str(items)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if text and not text.startswith("<"):
        return text
    return None


def _purpose_of(items: object) -> str | None:
    if items is None:
        return None
    return _named_purpose_of(items) or _flat_purpose_of(items)


def _issuer_info_of(schedule: Schedule13D | Schedule13G | SimpleNamespace) -> object:
    try:
        return getattr(schedule, "issuer_info", None)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _subject_of(schedule: Schedule13D | Schedule13G | SimpleNamespace) -> tuple[str | None, str | None]:
    """Authoritative subject from structured issuer info; never the filer."""
    info = _issuer_info_of(schedule)
    if info is None:
        return None, None
    cik = _first(info, "cik", "issuerCIK", "issuer_cik")
    name = _first(info, "name", "issuerName", "issuer_name")
    try:
        cik = str(cik).strip() if cik is not None else None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        cik = None
    try:
        name = str(name).strip() if name is not None else None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        name = None
    return (cik or None), (name or None)


def _reporting_persons_of(schedule: Schedule13D | Schedule13G | SimpleNamespace) -> list[object]:
    try:
        persons = getattr(schedule, "reporting_persons", []) or []
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return []
    if isinstance(persons, (str, bytes)) or not isinstance(persons, (list, tuple)):
        return []
    return list(persons)


def _explicit_subject(subject_cik: str | int | None, subject_name: str | None) -> tuple[str | None, str | None]:
    try:
        explicit_cik = str(subject_cik).strip() if subject_cik is not None else None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        explicit_cik = None
    try:
        explicit_name = subject_name.strip() if subject_name is not None else None
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        explicit_name = None
    return explicit_cik, explicit_name


def _resolved_subject(
    schedule: Schedule13D | Schedule13G | SimpleNamespace, subject_cik: str | int | None, subject_name: str | None
) -> tuple[str | None, str | None]:
    info_cik, info_name = _subject_of(schedule)
    explicit_cik, explicit_name = _explicit_subject(subject_cik, subject_name)
    return explicit_cik or info_cik, explicit_name or info_name


def _is_amendment_form(form: str) -> bool:
    try:
        return (form or "").strip().upper().endswith("/A")
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return False


def _person_record(
    person: object,
    *,
    issuer: str,
    form: str,
    filed_at: str | None,
    accession_no: str,
    purpose: str | None,
    resolved_cik: str | None,
    resolved_name: str | None,
    is_amend: bool,
    known: str | None,
    document_name: str | None,
    source_url: str | None,
) -> BeneficialOwnership:
    name = _first(person, "name", "filer_name", "reporting_person_name")
    cik = _first(person, "cik", "filer_cik", "reporting_person_cik")
    return BeneficialOwnership(
        filer_name=str(name) if name is not None else "",
        filer_cik=str(cik) if cik is not None else None,
        issuer=issuer,
        form=form,
        filed_at=filed_at,
        accession_no=accession_no,
        shares=_safe_int(_first(person, "aggregate_amount", "shares", "beneficially_owned", "aggregate_shares")),
        percent=_safe_float(_first(person, "percent_of_class", "percent", "ownership_percent")),
        sole_voting=_safe_int(_first(person, "sole_voting_power", "sole_voting")),
        shared_voting=_safe_int(_first(person, "shared_voting_power", "shared_voting")),
        sole_dispositive=_safe_int(_first(person, "sole_dispositive_power", "sole_dispositive")),
        shared_dispositive=_safe_int(_first(person, "shared_dispositive_power", "shared_dispositive")),
        is_amendment=is_amend,
        purpose_text=purpose,
        subject_cik=resolved_cik,
        subject_name=resolved_name,
        document_name=document_name,
        known_at=known,
        source_url=source_url,
    )


def _schedule_context_of(
    schedule: Schedule13D | Schedule13G | SimpleNamespace,
    subject_cik: str | int | None,
    subject_name: str | None,
    form: str,
    known_at: str | None,
    filed_at: str | None,
) -> tuple[str | None, str | None, str | None, bool, str | None]:
    try:
        items = getattr(schedule, "items", None)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        items = None
    resolved_cik, resolved_name = _resolved_subject(schedule, subject_cik, subject_name)
    try:
        known = known_at if known_at is not None else filed_at
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        known = filed_at
    return _purpose_of(items), resolved_cik, resolved_name, _is_amendment_form(form), known


def normalize_schedule(
    schedule: Schedule13D | Schedule13G | SimpleNamespace,
    *,
    issuer: str,
    form: str,
    filed_at: str | None,
    accession_no: str,
    subject_cik: str | int | None = None,
    subject_name: str | None = None,
    document_name: str | None = None,
    known_at: str | None = None,
    source_url: str | None = None,
) -> list[BeneficialOwnership]:
    """One BeneficialOwnership per reporting person; never raises.

    Subject identity comes only from explicit args or the schedule's
    structured ``issuer_info``. The filer is never copied into subject.
    """
    persons = _reporting_persons_of(schedule)
    if not persons:
        return []
    purpose, resolved_cik, resolved_name, is_amend, known = _schedule_context_of(
        schedule, subject_cik, subject_name, form, known_at, filed_at
    )
    out: list[BeneficialOwnership] = []
    for person in persons:
        try:
            out.append(
                _person_record(
                    person,
                    issuer=issuer,
                    form=form,
                    filed_at=filed_at,
                    accession_no=accession_no,
                    purpose=purpose,
                    resolved_cik=resolved_cik,
                    resolved_name=resolved_name,
                    is_amend=is_amend,
                    known=known,
                    document_name=document_name,
                    source_url=source_url,
                )
            )
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
    return out


def _schedule_from_filing(filing: EdgarFiling, form: str, accession_no: str) -> Schedule13D | Schedule13G:
    if form in _FORMS_13D:
        from edgar.beneficial_ownership import Schedule13D

        schedule = Schedule13D.from_filing(filing)
    else:
        from edgar.beneficial_ownership import Schedule13G

        schedule = Schedule13G.from_filing(filing)
    if schedule is None:
        raise ValueError(f"no schedule for accession {accession_no!r}")
    return schedule


def load_schedule(accession_no: str) -> Schedule13D | Schedule13G:
    """Live seam: edgar import stays here; raises on failure."""
    from .documents import get_by_accession_number

    filing = get_by_accession_number(accession_no)
    form = getattr(filing, "form", "") or ""
    return _schedule_from_filing(filing, form, accession_no)


def _records_from_filings(filings: list[Filing], ticker_or_cik: str | int) -> list[BeneficialOwnership]:
    out: list[BeneficialOwnership] = []
    for filing in filings:
        try:
            accession = getattr(filing, "accession_no", "")
            form = getattr(filing, "form", "")
            filed_at = getattr(filing, "filed_at", None)
            issuer = getattr(filing, "filer_name", None) or str(ticker_or_cik)
            schedule = load_schedule(accession)
            try:
                info = getattr(schedule, "issuer_info", None)
                subject_name = getattr(info, "name", None) if info is not None else None
                issuer = subject_name or issuer
            except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
                pass
            out.extend(
                normalize_schedule(
                    schedule,
                    issuer=issuer,
                    form=form,
                    filed_at=filed_at,
                    accession_no=accession,
                    document_name=getattr(filing, "primary_document", None),
                    known_at=getattr(filing, "known_at", None) or filed_at,
                    source_url=getattr(filing, "source", None),
                )
            )
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
    return out


def get_beneficial_ownership(
    ticker_or_cik: str | int,
    *,
    as_of: str | None = None,
    limit: int | None = 20,
    forms: tuple[str, ...] | list[str] = _DEFAULT_FORMS,
) -> list[BeneficialOwnership]:
    filings = list_sec_filings(ticker_or_cik, forms=list(forms), as_of=as_of, limit=limit)
    out = _records_from_filings(filings, ticker_or_cik)
    if limit is not None:
        out = out[:limit]
    return out


# Seam: subject/owner-filtered reads normalize live per filing via SourceGateway + normalization + raw_archive + write_bundle; NOTE: warehouse slots behind live readers.


def _filer_key(record: BeneficialOwnership) -> str:
    return record.filer_cik or record.filer_name


def _record_order(record: BeneficialOwnership) -> tuple[str, str]:
    return (record.filed_at or "", record.accession_no)


def _event_order(event: OwnershipChangeEvent) -> tuple[str, str]:
    return (event.filed_at or "", event.current_accession)


def _share_delta_of(previous: int | None, current: int | None) -> int | None:
    if previous is None or current is None:
        return None
    return current - previous


def _percent_delta_of(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return current - previous


_VOTING_PAIRS = ("sole_voting", "shared_voting", "sole_dispositive", "shared_dispositive")


def _voting_changed_of(previous: BeneficialOwnership, current: BeneficialOwnership) -> bool:
    return any(
        getattr(previous, name) is not None
        and getattr(current, name) is not None
        and getattr(previous, name) != getattr(current, name)
        for name in _VOTING_PAIRS
    )


def diff_ownership(previous: BeneficialOwnership, current: BeneficialOwnership) -> OwnershipChangeEvent:
    text_changed = bool(
        previous.purpose_text and current.purpose_text and previous.purpose_text != current.purpose_text
    )
    return OwnershipChangeEvent(
        filer_name=current.filer_name,
        filer_cik=current.filer_cik,
        issuer=current.issuer,
        previous_accession=previous.accession_no,
        current_accession=current.accession_no,
        filed_at=current.filed_at,
        prev_shares=previous.shares,
        curr_shares=current.shares,
        share_change=_share_delta_of(previous.shares, current.shares),
        prev_percent=previous.percent,
        curr_percent=current.percent,
        percent_change=_percent_delta_of(previous.percent, current.percent),
        voting_changed=_voting_changed_of(previous, current),
        text_changed=text_changed,
    )


def _window_changes(
    ticker_or_cik: str | int,
    *,
    as_of: str | None,
    limit: int | None,
) -> tuple[list[OwnershipChangeEvent], list[Filing], list[dict[str, object]]]:
    """Newest-window pairwise diffs; single-record owners become notices, never deltas."""
    scan_limit = None if limit is None else max(2, limit + 1)
    filings = list_sec_filings(ticker_or_cik, forms=list(_DEFAULT_FORMS), as_of=as_of, limit=scan_limit)
    records = _records_from_filings(filings, ticker_or_cik)
    groups: dict[str, list[BeneficialOwnership]] = {}
    for record in records:
        groups.setdefault(_filer_key(record), []).append(record)
    events: list[OwnershipChangeEvent] = []
    missing: list[dict[str, object]] = []
    for owner_records in groups.values():
        owner_records.sort(key=_record_order)
        if len(owner_records) < 2:
            only = owner_records[0]
            missing.append(
                {
                    "filer_name": only.filer_name,
                    "filer_cik": only.filer_cik,
                    "accession_no": only.accession_no,
                    "filed_at": only.filed_at,
                    "note": "single filing in scanned window; no prior baseline, no change inferred",
                }
            )
            continue
        for prev, curr in itertools.pairwise(owner_records):
            try:
                events.append(diff_ownership(prev, curr))
            except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
                continue
    events.sort(key=_event_order, reverse=True)
    if limit is not None:
        events = events[:limit]
    return events, filings, missing


def get_ownership_changes_with_coverage(
    ticker_or_cik: str | int,
    *,
    as_of: str | None = None,
    limit: int | None = 20,
) -> dict[str, object]:
    """Bounded diffs plus retrieval coverage for the tool envelope.

    Integer limits parse at most max(2, limit + 1) newest filings; explicit
    None stays exhaustive. The list result stays under ``events``.
    """
    events, filings, missing = _window_changes(ticker_or_cik, as_of=as_of, limit=limit)
    scan_limit = None if limit is None else max(2, limit + 1)
    return {
        "events": events,
        "scanned_accessions": [getattr(filing, "accession_no", "") for filing in filings],
        "scanned_count": len(filings),
        "incomplete_history": scan_limit is not None and len(filings) == scan_limit,
        "missing_baselines": missing,
    }


def get_ownership_changes(
    ticker_or_cik: str | int,
    *,
    as_of: str | None = None,
    limit: int | None = 20,
) -> list[OwnershipChangeEvent]:
    events, _, _ = _window_changes(ticker_or_cik, as_of=as_of, limit=limit)
    return events
