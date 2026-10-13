"""edgartools objects -> domain models. Every optional metadata access is
best-effort: any failure yields None, never an invented value."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .models import Filing, FilingDocument

if TYPE_CHECKING:
    # Provider SDK types at the boundary only; never constructed here.
    from edgar import Attachment as EdgarAttachment
    from edgar import Filing as EdgarFiling


def _best[T](fn: Callable[[], T], default: T | None = None) -> T | None:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return default


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    return text or None


def _first_present_attr(obj: object, attrs: tuple[str, ...]) -> str | None:
    for attr in attrs:
        value = _best(lambda a=attr: getattr(obj, a, None))
        if value is not None:
            text = _str_or_none(value)
            if text is not None:
                return text
    return None


def _moment_or_none(raw: str) -> datetime | None:
    """Parse SEC submissions timestamp text; None leaves normalization silent."""
    try:
        moment = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment


def _accepted_at(filing: EdgarFiling) -> str | None:
    """Normalize the submissions clock. Accept list and direct submissions rows."""
    listed = _first_present_attr(filing, ("acceptance_datetime", "accepted_at", "acceptanceDateTime"))
    moment = _moment_or_none(listed) if listed is not None else None
    if moment is None:
        return None
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


_ISSUER_SUBJECT_FORMS = frozenset({"10-K", "10-Q", "8-K", "S-1", "S-3", "DEF 14A"})


def _subject_of(form: str, filer_cik: int, filer_name: str) -> tuple[int | None, str | None]:
    """Subject equals filer only for issuer periodic/current/registration/proxy
    forms; third-party filings (13D/G, 3/4/5/144, 13F, tender/merger, …) leave
    subject unknown until a structured parser supplies it."""
    base = (form or "").split("/")[0].strip().upper()
    if base in _ISSUER_SUBJECT_FORMS:
        return filer_cik, filer_name
    return None, None


def _accession_of(filing: EdgarFiling) -> str:
    accession_no = _best(lambda: filing.accession_no)
    if accession_no is None:
        accession_no = _best(lambda: filing.accession_number, "") or ""
    return accession_no


def _primary_document_of(filing: EdgarFiling) -> str | None:
    doc = _best(lambda: filing.document)
    if isinstance(doc, str):
        return doc or None
    if doc is None:
        return None
    name = _best(lambda: getattr(doc, "document", None))
    return name if isinstance(name, str) and name else None


def filing_from_edgar(filing: EdgarFiling) -> Filing:
    form = _best(lambda: filing.form, "") or ""
    filed_at = _str_or_none(_best(lambda: filing.filing_date, "") or "") or ""
    accepted_at = _accepted_at(filing)
    accession_no = _accession_of(filing)
    filer_cik = _best(lambda: filing.cik, 0) or 0
    filer_name = _best(lambda: filing.company, "") or ""
    subject_cik, subject_name = _subject_of(form, filer_cik, filer_name)
    return Filing(
        accession_no=accession_no,
        form=form,
        filer_cik=filer_cik,
        filer_name=filer_name,
        filed_at=filed_at,
        accepted_at=accepted_at,
        known_at=accepted_at or filed_at,
        report_period=_str_or_none(_best(lambda: filing.period_of_report)),
        primary_document=_primary_document_of(filing),
        is_amendment=form.endswith("/A"),
        amendment_of=None,
        source=_best(lambda: filing.homepage_url, "") or "",
        subject_cik=subject_cik,
        subject_name=subject_name,
        accepted_at_missing=accepted_at is None,
    )


def document_from_attachment(accession_no: str, attachment: EdgarAttachment) -> FilingDocument:
    return FilingDocument(
        accession_no=accession_no,
        document_name=_best(lambda: attachment.document),
        description=_best(lambda: attachment.description),
        size=_best(lambda: attachment.size),
        url=_best(lambda: attachment.url, "") or "",
        document_type=_best(lambda: attachment.document_type),
    )
