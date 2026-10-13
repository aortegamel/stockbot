"""Deterministic filing diffs (stdlib difflib only; never an LLM)."""

from __future__ import annotations

import difflib

_MAX_LINES = 500


def _is_periodic_set(forms: set[str]) -> bool:
    return forms <= {"10K", "10Q", "10KA", "10QA"}


def _specialization_of_blob(blob: str) -> str:
    for token, label in (("13D", "13D/A"), ("13G", "13G/A"), ("S1", "S-1/A"), ("S3", "S-3/A")):
        if token in blob:
            return label
    if any(x in blob for x in ("14A", "14C", "PX14A")):
        return "proxy"
    if any(x in blob for x in ("SCTO", "14D9", "13E3", "S4")):
        return "tender/merger"
    return "generic"


def _specialization(forms: list[str]) -> str:
    f = {x.upper().replace("-", "").replace(" ", "") for x in forms}
    if _is_periodic_set(f):
        return "10-K/10-Q"
    return _specialization_of_blob(" ".join(f))


def _wants_primary(section: str | None) -> str | None:
    if section is None or not section.strip() or section.strip().lower() == "full":
        return None
    return section.strip()


def _list_document_names(accession_no: str) -> list[str]:
    from . import documents

    try:
        return [d.document_name for d in documents.list_sec_documents(accession_no) if d.document_name]
    except Exception as exc:
        raise ValueError(f"cannot list documents for {accession_no!r}: {exc}") from exc


def _text_for_section(accession_no: str, section: str | None) -> str:
    """Read an exact attachment or a rendered heading in the primary document."""
    from . import documents

    want = _wants_primary(section)
    if want is None:
        return documents.get_sec_filing_text(accession_no)
    for name in _list_document_names(accession_no):
        if name.lower() == want.lower():
            return documents.get_sec_filing_text(accession_no, name)
    text = documents.get_sec_document(accession_no, section=want)["text"]
    if not isinstance(text, str):
        raise TypeError(f"no text for accession: {accession_no!r}")
    return text


def diff_filings(
    current_accession: str,
    previous_accession: str,
    section: str | None = None,
) -> dict[str, object]:
    from . import filings

    try:
        cur = _text_for_section(current_accession, section)
        prev = _text_for_section(previous_accession, section)
        forms = [filings.get_sec_filing(a).form for a in (current_accession, previous_accession)]
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {"error": str(exc)}
    lines = list(difflib.unified_diff(prev.splitlines(), cur.splitlines(), lineterm="", n=3))
    truncated = len(lines) > _MAX_LINES
    body = lines[:_MAX_LINES]
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return {
        "current_accession": current_accession,
        "previous_accession": previous_accession,
        "section": section,
        "specialization": _specialization(forms),
        "diff_lines": body,
        "added": added,
        "removed": removed,
        "truncated": truncated,
    }
