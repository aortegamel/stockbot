"""Document-level retrieval off a filing accession.

Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind these live readers, never inside normalization.
"""

import hashlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    # Provider SDK filing type at the boundary only; never constructed here.
    from edgar import Filing as EdgarFiling
from .models import Filing, FilingDocument, pit_of
from .normalization import document_from_attachment, filing_from_edgar

_MAX_CHARS = 32_000
_VIEW_MAX_SECTIONS = 50

_IXBLR_NOISE_RE = re.compile(
    r"<[A-Za-z][\w.-]*:[^>]*>.*?</[A-Za-z][\w.-]*:[^>]*>|<[A-Za-z][\w.-]*:[^>]*/>", re.IGNORECASE | re.DOTALL
)
_STYLE_SCRIPT_RE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\x0b\x0c\r]+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HEADING_RE = re.compile(r"^\s*(item\s+\d+[a-z]?(?:\([^)]*\))?\.?.*|part\s+[ivx]+\.?|signatures?)\s*$", re.IGNORECASE)
_BLOCK_TAGS = frozenset({"br", "p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li", "table"})
_CELL_TAGS = frozenset({"td", "th"})
_END_BLOCK_TAGS = frozenset({"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "table"})


def _tag_name(tag: str) -> str:
    """Parser tag without namespace prefix, lowercased (filings mix `ix:`/`xbrli:` prefixes)."""
    return tag.lower().split(":")[-1]


class _ViewTextParser(HTMLParser):
    """Minimal stdlib renderer: block structure + tables, inline text otherwise."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._cell: list[str] | None = None
        self._row: list[str] | None = None

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = _tag_name(tag)
        if name in _BLOCK_TAGS:
            self._chunks.append("\n")
        if name == "tr":
            self._row = []
        elif name in _CELL_TAGS:
            self._cell = []

    def _close_cell(self) -> None:
        if self._cell is not None and self._row is not None:
            self._row.append(_WS_RE.sub(" ", "".join(self._cell)).strip())
            self._cell = None

    def _close_row(self) -> None:
        cells = [c for c in (self._row or []) if c]
        if cells:
            self._chunks.append(" | ".join(cells) + "\n")
        self._row = None

    @override
    def handle_endtag(self, tag: str) -> None:
        name = _tag_name(tag)
        if name in _CELL_TAGS:
            self._close_cell()
        elif name == "tr":
            self._close_row()
        elif name in _END_BLOCK_TAGS:
            self._chunks.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def _strip_noise(source: str) -> str:
    """Drop CSS/layout/XBRL noise before parsing; keep textual markup."""
    text = _STYLE_SCRIPT_RE.sub("\n", source)
    text = _COMMENT_RE.sub("", text)
    return _IXBLR_NOISE_RE.sub("", text)


def _normalize_plain(cleaned: str) -> str:
    lines = [_WS_RE.sub(" ", ln).strip() for ln in cleaned.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _parse_view(cleaned: str) -> str:
    parser = _ViewTextParser()
    try:
        parser.feed(cleaned)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed filing HTML degrades to tag-strip, never raises
        return _WS_RE.sub(" ", _TAG_RE.sub(" ", cleaned)).strip()
    lines = [_WS_RE.sub(" ", ln).strip(" |") for ln in parser.text().splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _section_names(view: str) -> list[str]:
    return [name for name, _ in _split_sections(view) if name][:_VIEW_MAX_SECTIONS]


def _render_text(source: str) -> str:
    """Model-readable text: tables as pipes, headings/paragraphs preserved."""
    cleaned = _strip_noise(source)
    if "<" not in cleaned or ">" not in cleaned:
        return _normalize_plain(cleaned)
    return _parse_view(cleaned)


def _split_sections(view: str) -> list[tuple[str | None, int]]:
    """(heading, char_offset) scan: first block unheaded, then ITEM/PART/SIGNATURES."""
    sections: list[tuple[str | None, int]] = []
    offset = 0
    current: str | None = None
    for line in view.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and _HEADING_RE.match(stripped) and len(stripped) < 160:
            current = stripped
            sections.append((current, offset))
        elif not sections:
            sections.append((None, 0))
        offset += len(line)
    if not sections:
        sections.append((None, 0))
    return sections[: _VIEW_MAX_SECTIONS + 1]


def _section_spans(view: str) -> list[tuple[str, int, int]]:
    """Named (heading, start, end) spans over the rendered view."""
    spans = _split_sections(view)
    ends = [start for _, start in spans[1:]] + [len(view)]
    return [(name, start, end) for (name, start), end in zip(spans, ends) if name is not None]


def _resolve_section(view: str, bounds: list[tuple[str, int, int]], section: str) -> tuple[str, str, int] | None:
    want = section.strip().lower()
    best: tuple[str, str, int] | None = None
    for name, start, end in bounds:
        lowered = name.lower()
        # TOC entries match first but carry ~30 chars; the body match wins on length.
        if (want in lowered or lowered in want) and (best is None or (end - start) > len(best[0])):
            best = (view[start:end], name, start)
    return best


def _select_section(view: str, section: str | None) -> tuple[str, str | None, int]:
    """Narrow a rendered view to one heading; returns (text, resolved, base_offset)."""
    if section is None or not section.strip():
        return view, None, 0
    bounds = _section_spans(view)
    hit = _resolve_section(view, bounds, section)
    if hit is not None:
        return hit
    available = "; ".join(name for name, _, _ in bounds[:12]) or "none"
    raise ValueError(f"section not found: {section!r}; available sections: {available}")


def _select_query(view: str, query: str | None) -> tuple[str, int]:
    """Narrow a rendered view to the first query hit with context; (text, base_offset)."""
    if query is None or not query.strip():
        return view, 0
    at = view.lower().find(query.strip().lower())
    if at < 0:
        raise ValueError(f"query not found in document: {query!r}")
    start = max(0, at - 2000)
    return view[start:], start


def _source_uri_for(accession_no: str, document_name: object) -> str:
    doc = document_name if isinstance(document_name, str) and document_name else "primary"
    return f"source://sec/{accession_no}/{doc}"


def render_sec_view(source: str) -> str:
    """Public seam for the derived model-readable view (raw stays in the archive)."""
    return _render_text(source)


def _normalize_accession(accession_no: str) -> str:
    """Tolerate pasting variants of the same identifier (never invent one):
    surrounding whitespace is stripped and a bare 18-digit run takes the
    canonical 10-2-6 dashes. Anything else passes through to live lookup."""
    text = accession_no.strip() if isinstance(accession_no, str) else accession_no
    digits = "".join(ch for ch in text if ch.isdigit())
    if isinstance(text, str) and "-" not in text and len(digits) == 18 and text.strip().isdigit():
        return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
    return text


def get_by_accession_number(accession_no: str) -> EdgarFiling:
    """Seam for tests: monkeypatch this name, never `edgar` itself."""
    from edgar import get_by_accession_number as _get

    from .client import ensure_identity

    ensure_identity()
    filing: EdgarFiling = _get(_normalize_accession(accession_no))
    return filing


def _filing(accession_no: str) -> EdgarFiling:
    try:
        filing = get_by_accession_number(accession_no)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid accession number: {accession_no!r}") from exc
    if filing is None:
        raise ValueError(f"invalid accession number: {accession_no!r}")
    return filing


def _meta(accession_no: str) -> Filing:
    """Hydrate filing metadata first so exact lookups cannot bypass PIT."""
    return filing_from_edgar(_filing(accession_no))


def _require_known(accession_no: str, as_of: str | None) -> Filing:
    """Validate as_of and reject accessions unknown at that date."""
    from .filings import _check_as_of

    as_of = _check_as_of(as_of)
    if as_of is None:
        return _meta(accession_no)
    meta = _meta(accession_no)
    value, _basis = pit_of(meta)
    if value is None or value[:10] > as_of:
        raise ValueError(f"filing {accession_no!r} not known as of {as_of!r}")
    return meta


def list_sec_documents(accession_no: str, as_of: str | None = None) -> list[FilingDocument]:
    meta = _require_known(accession_no, as_of)
    filing = _filing(accession_no)
    out: list[FilingDocument] = []
    for a in filing.attachments:
        doc = document_from_attachment(accession_no, a)
        out.append(
            replace(
                doc,
                filed_at=meta.filed_at or None,
                accepted_at=meta.accepted_at,
                known_at=meta.known_at or None,
                source_url=meta.source or None,
                is_primary=(doc.document_name is not None and doc.document_name == meta.primary_document),
            )
        )
    return out


def _primary_attachment_of(filing: EdgarFiling, accession_no: str) -> object:
    try:
        attachment = filing.document
    except Exception as exc:
        raise ValueError(f"no primary document for accession: {accession_no!r}") from exc
    if attachment is None:
        raise ValueError(f"no primary document for accession: {accession_no!r}")
    return attachment


def _attachment_document_name(attachment: object) -> str | None:
    """Best-effort document filename; None when unreadable or blank (one shared probe)."""
    try:
        name = getattr(attachment, "document")  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    return name if isinstance(name, str) and name else None


def _missing_document_error(accession_no: str, document_name: str, names: list[str]) -> ValueError:
    """Not-found error with the 12-name sample (single message construction site)."""
    sample = ", ".join(names[:12])
    return ValueError(
        f"document not found: {document_name!r} for accession {accession_no!r}; "
        f"available documents: {sample or 'none'}. Call list_sec_documents "
        f"for the full list, or omit the document name for the primary document."
    )


def _named_attachment_of(filing: EdgarFiling, accession_no: str, document_name: str) -> object:
    try:
        attachments = filing.attachments
    except Exception as exc:
        raise ValueError(f"no documents for accession: {accession_no!r}") from exc
    names: list[str] = []
    want_exhibit = _normalize_exhibit(document_name)
    for attachment in attachments:
        name = _attachment_document_name(attachment)
        if name is not None:
            names.append(name)
        if name == document_name:
            return attachment
    if want_exhibit is not None:
        for attachment in attachments:
            if _attachment_exhibit_of(attachment) == want_exhibit:
                return attachment
    raise _missing_document_error(accession_no, document_name, names)


def _resolve_in(filing: EdgarFiling, accession_no: str, document_name: str | None = None) -> object:
    if document_name is None:
        return _primary_attachment_of(filing, accession_no)
    return _named_attachment_of(filing, accession_no, document_name)


def _attr_text_of(attachment: object, attr: str) -> str | None:
    try:
        value = getattr(attachment, attr)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if callable(value):
        try:
            value = value()
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    return None


def _text_of(attachment: object) -> str:
    for attr in ("content", "text"):
        found = _attr_text_of(attachment, attr)
        if found is not None:
            return found
    return ""


def _raw_root_for(data_root: Path | str | None) -> Path | None:
    """Raw archive root straight under a data root."""
    # Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in here.
    if data_root is None:
        return None
    return Path(data_root) / "raw"


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_window(offset: str | float, max_chars: str | float | None) -> tuple[int, int | None]:
    try:
        offset = int(offset)
    except TypeError, ValueError:
        raise ValueError(f"offset must be an integer, got {offset!r}") from None
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if max_chars is None:
        return offset, None
    try:
        max_chars = int(max_chars)
    except TypeError, ValueError:
        raise ValueError(f"max_chars must be an integer, got {max_chars!r}") from None
    if max_chars < 1 or max_chars > _MAX_CHARS:
        raise ValueError(f"max_chars must be 1..{_MAX_CHARS}, got {max_chars}")
    return offset, max_chars


def _iso_stamp(value: object) -> object:
    """Warehouse date/datetime to ISO text; non-dates pass through."""
    from datetime import date, datetime

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _iso_instant(value: object) -> object:
    """Warehouse datetime to full UTC instant text; strings pass through."""
    from datetime import UTC, date, datetime

    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _local_stamp(value: object) -> str | None:
    """Host-local rendering of the same instant; None when no time info."""
    from datetime import UTC, datetime

    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone().isoformat()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone().isoformat()
    return None


def _bounded_response(
    *,
    accession_no: str,
    document_name: object,
    description: object,
    url: object,
    full_text: str,
    content_hash: object,
    source_content_hash: object,
    source_representation: object,
    raw_archive_path: object,
    source_url: object,
    filed_at: object,
    known_at: object,
    retrieved_at: object,
    offset: int,
    max_chars: int | None,
    cache_hit: bool,
    cache_type: str,
    warnings: Iterable[str] | None = None,
    view_text: str | None = None,
    view_base: int = 0,
    resolved_section: str | None = None,
    raw_view: bool = False,
    query: str | None = None,
    available_sections: list[str] | None = None,
) -> dict[str, object]:
    effective = view_text if view_text is not None else full_text
    total = len(effective)
    if offset > total:
        raise ValueError(f"offset {offset} beyond document length {total}")
    end = total if max_chars is None else min(offset + max_chars, total)
    out = {
        "accession_no": accession_no,
        "document_name": document_name,
        "description": description,
        "url": url,
        "text": effective[offset:end],
        "content_hash": content_hash,
        "source_content_hash": source_content_hash,
        "source_representation": source_representation,
        "raw_archive_path": raw_archive_path,
        "offset": offset,
        "end_offset": end,
        "total_chars": total,
        "more_available": end < total,
        "source_url": source_url,
        "filed_at": _iso_stamp(filed_at),
        "known_at": _iso_instant(known_at),
        "retrieved_at": _iso_instant(retrieved_at),
        "known_at_local": _local_stamp(known_at),
        "retrieved_at_local": _local_stamp(retrieved_at),
        "cache_hit": cache_hit,
        "cache_type": cache_type,
    }
    out["source_handle"] = _source_handle(
        accession_no=accession_no,
        document_name=document_name,
        view_text=view_text,
        full_text=full_text,
        offset=offset,
        end=end,
        max_chars=max_chars,
        raw_view=raw_view,
        resolved_section=resolved_section,
        query=query,
        content_hash=content_hash,
        source_content_hash=source_content_hash,
    )
    if warnings:
        out["warnings"] = list(warnings)
    _attach_view(
        out,
        accession_no=accession_no,
        document_name=document_name,
        view_text=view_text,
        view_base=view_base,
        resolved_section=resolved_section,
        raw_view=raw_view,
        available_sections=available_sections,
        source_url=source_url,
        filed_at=_iso_stamp(filed_at),
        known_at=_iso_instant(known_at),
        retrieved_at=_iso_instant(retrieved_at),
        known_at_local=_local_stamp(known_at),
        retrieved_at_local=_local_stamp(retrieved_at),
        content_hash=content_hash,
        source_content_hash=source_content_hash,
        offset=offset,
        end=end,
        total=total,
    )
    return out


def _attach_view(
    out: dict[str, object],
    *,
    accession_no: str,
    document_name: object,
    view_text: str | None,
    view_base: int,
    resolved_section: str | None,
    raw_view: bool,
    available_sections: list[str] | None,
    source_url: object,
    filed_at: object,
    known_at: object,
    retrieved_at: object,
    known_at_local: object,
    retrieved_at_local: object,
    content_hash: object,
    source_content_hash: object,
    offset: int,
    end: int,
    total: int,
) -> None:
    """Raw-addressable cursor + span pointers for the derived view (raw stays in the archive)."""
    if view_text is None:
        out["cursor"] = offset
        out["next_cursor"] = end if end < total else None
        return
    uri = _source_uri_for(accession_no, document_name)
    out["view"] = "raw" if raw_view else "rendered"
    out["section"] = resolved_section
    out["available_sections"] = (
        list(available_sections) if available_sections is not None else _section_names(view_text)
    )
    out["metadata"] = {
        "accession_no": accession_no,
        "document_name": document_name,
        "section": resolved_section,
        "source_uri": uri,
        "source_url": source_url,
        "filed_at": filed_at,
        "known_at": known_at,
        "retrieved_at": retrieved_at,
        "known_at_local": known_at_local,
        "retrieved_at_local": retrieved_at_local,
        "content_hash": content_hash,
        "source_content_hash": source_content_hash,
    }
    out["source_uri"] = uri
    out["source_refs"] = [
        {
            "accession": accession_no,
            "document": document_name if isinstance(document_name, str) else None,
            "offset": view_base + offset,
            "source_uri": uri,
        }
    ]
    out["cursor"] = offset
    out["next_cursor"] = end if end < len(view_text) else None


def _source_handle(
    *,
    accession_no: str,
    document_name: object,
    view_text: str | None,
    full_text: str,
    offset: int,
    end: int,
    max_chars: int | None,
    raw_view: bool,
    resolved_section: str | None,
    query: str | None,
    source_content_hash: object,
    content_hash: object = None,
) -> dict[str, object]:
    """Canonical source handle for the window just returned.

    It carries the exact read coordinates and a hash of the returned text, so a
    caller (the research kernel) can re-read the same window and verify it before
    admitting a passage as evidence: the archive, never a model string, is
    authoritative for source text.
    """
    effective = view_text if view_text is not None else full_text
    window = effective[offset:end]
    return {
        "accession_no": accession_no,
        "document_name": document_name if isinstance(document_name, str) else None,
        "basis": "raw" if raw_view else "rendered",
        "section": resolved_section if isinstance(resolved_section, str) and resolved_section.strip() else None,
        "query": query if isinstance(query, str) and query.strip() else None,
        "offset": offset,
        "max_chars": max_chars,
        "length": len(window),
        "text_hash": hashlib.sha256(window.encode("utf-8")).hexdigest(),
        "source_content_hash": source_content_hash,
        "source_uri": _source_uri_for(accession_no, document_name),
    }


def _binary_source_kind(text: str) -> str | None:
    """Binary representation marker for retrieved text: "pdf" for the %PDF magic, else None."""
    return "pdf" if text[:16].lstrip("\ufeff \t\r\n\v\f")[:4] == "%PDF" else None


def _pdftotext_output(extractor: str, path: str) -> str | None:
    """One ``pdftotext -layout`` run: cleaned stdout text, or None on nonzero exit or no text."""
    completed = subprocess.run([extractor, "-q", "-layout", path, "-"], capture_output=True, timeout=30, check=False)
    if completed.returncode != 0:
        return None
    text = _CONTROL_CHARS_RE.sub("", completed.stdout.decode("utf-8", "replace"))
    return text if text.strip() else None


def _extract_pdf_text(payload: bytes | Path) -> str | None:
    """Text of a PDF payload (exact source bytes, or the archived file holding them).

    None when no ``pdftotext`` is installed, the source is unreadable, or extraction
    fails or yields no text; the caller reports that as ``binary_unsupported``.
    """
    extractor = shutil.which("pdftotext")
    if extractor is None:
        return None
    try:
        if isinstance(payload, Path):
            return _pdftotext_output(extractor, str(payload))
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(payload)
            handle.flush()
            return _pdftotext_output(extractor, handle.name)
    except OSError, subprocess.SubprocessError:
        return None


def _readable_document_text(
    text: str, representation: object, payload: bytes | Path | None
) -> tuple[str, object, list[str]]:
    """Caller-facing document text: a binary payload never leaves here as ``text``.

    Non-binary text passes through untouched with its representation. A binary payload
    becomes extracted PDF text (``pdf_extracted``) or empty text plus an explicit
    ``binary_unsupported`` marker and a warning naming the reason. ``payload`` is the
    exact source (bytes, or the archived file path) and is only read for binary text.
    """
    kind = _binary_source_kind(text)
    if kind is None and "\x00" not in text:
        return text, representation, []
    extracted = _extract_pdf_text(payload) if kind == "pdf" and payload is not None else None
    if extracted is not None:
        # cleaned once more: no NUL may reach a caller, and extraction is a monkeypatch seam
        return _CONTROL_CHARS_RE.sub("", extracted), "pdf_extracted", ["pdf text extracted via pdftotext"]
    if kind != "pdf":
        reason = f"binary {representation} payload has no text representation"
    elif payload is None:
        reason = "no raw pdf bytes available for extraction"
    elif shutil.which("pdftotext") is None:
        reason = "pdftotext unavailable"
    else:
        reason = "pdf text extraction failed"
    return "", "binary_unsupported", [f"{reason}; raw bytes remain at raw_archive_path"]


def _build_view(
    full: str, *, section: str | None, query: str | None, raw: bool
) -> tuple[str, str | None, int, bool, list[str]]:
    """Derived window over stored text; returns (view, resolved, base, is_raw, sections)."""
    if raw:
        return full, section.strip() if isinstance(section, str) and section.strip() else None, 0, True, []
    rendered = _render_text(full)
    sections = _section_names(rendered)
    narrowed, resolved, base = _select_section(rendered, section)
    queried, qbase = _select_query(narrowed, query)
    return queried, resolved, base + qbase, False, sections


def _download_bytes_of(attachment: object) -> bytes | None:
    download = getattr(attachment, "download", None)
    if not callable(download):
        return None
    try:
        result = download()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if isinstance(result, bytes):
        return result
    if isinstance(result, str):
        return result.encode("utf-8")
    return None


def _attr_bytes_of(attachment: object, attr: str) -> bytes | None:
    try:
        value = getattr(attachment, attr)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if callable(value):
        try:
            value = value()
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return None


def _source_bytes_of(attachment: object) -> tuple[bytes | None, str | None]:
    """Exact source bytes when EdgarTools exposes them.

    ``download()`` bytes win, then ``content``/``text`` (bytes as-is, str
    encoded as UTF-8). Returns (None, None) only when no payload is exposed.
    """
    downloaded = _download_bytes_of(attachment)
    if downloaded is not None:
        return downloaded, "source_bytes"
    for attr in ("content", "text"):
        found = _attr_bytes_of(attachment, attr)
        if found is not None:
            return found, "source_bytes"
    return None, None


def _filing_row_of(accession_no: str, as_of: str | None, data_root: Path | str | None) -> Filing | None:
    from . import store as _store

    try:
        filings = _store.query_filings(accession=accession_no, as_of=as_of, limit=5, root=data_root)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    return filings[0] if filings else None


def _effective_document_name(document_name: str | None, filing: Filing | None) -> str | None:
    filing_name = filing.primary_document if filing else None
    if filing_name is not None and not isinstance(filing_name, str):
        filing_name = str(filing_name)
    return document_name or filing_name


def _archived_row_of(record: object, *, accession_no: str, document_name: str) -> dict[str, object]:
    """One raw-archive record as an archived-response row (text + provenance)."""
    from ..storage import raw_archive as _raw

    payload_path = getattr(record, "payload_path", None)
    text = ""
    if isinstance(payload_path, Path):
        try:
            text = payload_path.read_bytes().decode("utf-8", "replace")
        except OSError:
            text = ""
    metadata = getattr(record, "metadata", None)
    meta = metadata if isinstance(metadata, dict) else {}
    content_hash = getattr(record, "sha256", None)
    return {
        "document_name": document_name,
        "text": text,
        "source_url": getattr(record, "url", "") or "",
        "content_hash": content_hash if isinstance(content_hash, str) else _raw.content_hash(text.encode("utf-8")),
        "source_content_hash": content_hash if isinstance(content_hash, str) else None,
        "source_representation": meta.get("representation", "source_bytes"),
        "raw_archive_path": str(payload_path) if payload_path is not None else None,
        "filed_at": meta.get("filed_at"),
        "known_at": meta.get("known_at"),
        "retrieved_at": getattr(record, "retrieved_at", None),
    }


def _known_at_or_before(row: dict[str, object], as_of: str) -> bool:
    """A row passes a historical read only with a real known_at on or before ``as_of``."""
    known = str(row.get("known_at") or "").strip()
    return bool(known) and known[:10] <= as_of


def _archived_rows_of(
    accession_no: str,
    document_name: str | None,
    as_of: str | None,
    data_root: Path | str | None,
    filing: Filing | None,
) -> list[dict[str, object]]:
    """Raw-archive read-back: previously archived source bytes, PIT-filtered."""
    # Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in here.
    from . import archive as _archive

    effective = _effective_document_name(document_name, filing)
    if effective is None:
        return []
    try:
        records = list(_archive.iter_archived_documents(accession_no, effective, root=_raw_root_for(data_root)))
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return []
    rows = [_archived_row_of(record, accession_no=accession_no, document_name=effective) for record in records]
    if as_of is not None:
        rows = [row for row in rows if _known_at_or_before(row, as_of)]
    rows.sort(
        key=lambda row: (
            str(row.get("known_at") or ""),
            str(row.get("retrieved_at") or ""),
            str(row.get("content_hash") or ""),
        ),
        reverse=True,
    )
    return rows


def _stored_candidates(
    accession_no: str, document_name: str | None, as_of: str | None, data_root: Path | str | None
) -> tuple[Filing | None, list[dict[str, object]]]:
    """Local archive rows for one accession/document; returns (filing, rows).

    Filing lookup resolves the primary name when document_name is None;
    without a filing row and without a name, an unambiguous single-document
    accession still resolves. Storage failures read as a local miss.
    """
    filing = _filing_row_of(accession_no, as_of, data_root)
    return filing, _archived_rows_of(accession_no, document_name, as_of, data_root, filing)


def _revision_conflict(accession_no: str, document_name: object, top_known: object, as_of: str) -> dict[str, object]:
    return {
        "error": f"conflicting revisions for {accession_no}/{document_name} "
        f"known at {top_known}; cannot resolve as of {as_of}",
        "error_type": "pit_revision_conflict",
        "accession_no": accession_no,
        "document_name": document_name,
        "known_at": top_known,
        "as_of": as_of,
    }


def _revision_warning(accession_no: str, document_name: object, top_known: object, count: int) -> str:
    return (
        f"conflicting revisions for {accession_no}/{document_name} "
        f"known at {top_known}; chose latest retrieved_at/content_hash, "
        f"{count} revision(s) retained"
    )


def _pick_revision(
    rows: list[dict[str, object]], *, as_of: str | None, accession_no: str, document_name: object
) -> tuple[dict[str, object] | None, list[str] | None, dict[str, object] | None]:
    """Deterministic revision choice; returns (row, warnings) or an error dict.

    Rows arrive ordered known_at DESC, retrieved_at DESC, content_hash DESC.
    Conflicting hashes at the winning known_at warn on latest reads but are
    a PIT conflict on historical reads: retrieval time must not decide what
    was knowable.
    """
    top_known = rows[0].get("known_at")
    winners = [r for r in rows if r.get("known_at") == top_known]
    if len({r.get("content_hash") for r in winners}) > 1:
        if as_of is not None:
            return None, None, _revision_conflict(accession_no, document_name, top_known, as_of)
        return winners[0], [_revision_warning(accession_no, document_name, top_known, len(winners) - 1)], None
    return rows[0], None, None


def _winning_revision(
    accession_no: str, document_name: str | None, rows: list[dict[str, object]], *, as_of: str | None
) -> tuple[dict[str, object], list[str] | None]:
    display_name = document_name or rows[0].get("document_name")
    row, rev_warnings, conflict = _pick_revision(
        rows, as_of=as_of, accession_no=accession_no, document_name=display_name
    )
    if conflict is not None:
        return conflict, rev_warnings
    if row is None:  # Unreachable: a winning revision always yields a row.
        raise ValueError(f"no document revision for {accession_no}/{display_name}")
    return row, rev_warnings


def _archived_body_of(row: dict[str, object]) -> tuple[str, object]:
    raw_text = row.get("text")
    return (raw_text if isinstance(raw_text, str) else "", row.get("source_url"))


def _archived_bounded(
    accession_no: str,
    document_name: str | None,
    row: dict[str, object],
    full: str,
    source_url: object,
    offset: int,
    max_chars: int | None,
    rev_warnings: list[str] | None,
    *,
    section: str | None = None,
    query: str | None = None,
    raw: bool = False,
) -> dict[str, object]:
    name_raw = row.get("document_name")
    doc_name = (name_raw if isinstance(name_raw, str) else None) or document_name
    raw_path = row.get("raw_archive_path")
    full, representation, binary_warnings = _readable_document_text(
        full,
        row.get("source_representation"),
        Path(raw_path) if isinstance(raw_path, str) and raw_path else None,
    )
    view, resolved, base, is_raw, sections = _build_view(full, section=section, query=query, raw=raw)
    return _bounded_response(
        accession_no=accession_no,
        document_name=doc_name,
        description=None,
        url=source_url or "",
        full_text=full,
        content_hash=row.get("content_hash"),
        source_content_hash=row.get("source_content_hash"),
        source_representation=representation,
        raw_archive_path=row.get("raw_archive_path"),
        source_url=source_url,
        filed_at=row.get("filed_at"),
        known_at=row.get("known_at"),
        retrieved_at=row.get("retrieved_at"),
        offset=offset,
        max_chars=max_chars,
        cache_hit=True,
        cache_type="stockbot_archive",
        warnings=((rev_warnings or []) + binary_warnings) or None,
        view_text=view,
        view_base=base,
        resolved_section=resolved,
        raw_view=is_raw,
        query=query,
        available_sections=sections,
    )


def _archived_response(
    accession_no: str,
    document_name: str | None,
    rows: list[dict[str, object]],
    *,
    as_of: str | None,
    offset: int,
    max_chars: int | None,
    section: str | None = None,
    query: str | None = None,
    raw: bool = False,
) -> dict[str, object]:
    row, rev_warnings = _winning_revision(accession_no, document_name, rows, as_of=as_of)
    if isinstance(row, dict) and row.get("error_type") == "pit_revision_conflict":
        return row
    full, source_url = _archived_body_of(row)
    return _archived_bounded(
        accession_no,
        document_name,
        row,
        full,
        source_url,
        offset,
        max_chars,
        rev_warnings,
        section=section,
        query=query,
        raw=raw,
    )


def _attachment_attr_of(attachment: object, name: str) -> object:
    try:
        found: object = getattr(attachment, name)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    narrowed: object = found
    return narrowed


def _attachment_meta_of(
    attachment: object, document_name: str | None, primary_document: str | None
) -> tuple[object, object, object, str]:
    name = _attachment_attr_of(attachment, "document")
    description = _attachment_attr_of(attachment, "description")
    url = _attachment_attr_of(attachment, "url") or ""
    doc_name = (name if isinstance(name, str) else None) or document_name or primary_document or "primary"
    return name, description, url, doc_name


def _normalized_text_of(attachment: object) -> tuple[bytes, str, str]:
    source_bytes, representation = _source_bytes_of(attachment)
    if source_bytes is None:
        normalized = _text_of(attachment)
        return normalized.encode("utf-8"), normalized, "normalized_text"
    return source_bytes, source_bytes.decode("utf-8", "replace"), representation or "normalized_text"


def _persist_live_document(
    *,
    accession_no: str,
    doc_name: str,
    source_bytes: bytes,
    representation: str,
    normalized: str,
    source_content_hash: str,
    content_hash: str,
    meta: Filing,
    source_url: str | None,
    data_root: Path | str | None,
) -> tuple[str | None, str | None, list[str]]:
    """Fetch-time lookup only: no archiving here (selective retention)."""
    _ = (accession_no, doc_name, source_bytes, representation, meta, source_url, data_root)
    _ = (normalized, content_hash, source_content_hash)
    return None, None, []


def _live_hashes(source_bytes: bytes, normalized: str) -> tuple[str, str]:
    source_content_hash = hashlib.sha256(source_bytes).hexdigest()
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return source_content_hash, content_hash


def _live_source_url(url: object, meta: Filing) -> str | None:
    source_url = url or meta.source or None
    if not isinstance(source_url, str) and source_url is not None:
        return str(source_url)
    return source_url


def _live_response(
    accession_no: str,
    document_name: str | None,
    *,
    attachment: object,
    meta: Filing,
    offset: int,
    max_chars: int | None,
    data_root: Path | str | None,
    section: str | None = None,
    query: str | None = None,
    raw: bool = False,
) -> dict[str, object]:
    _name, description, url, doc_name = _attachment_meta_of(attachment, document_name, meta.primary_document)
    source_bytes, normalized, representation = _normalized_text_of(attachment)
    source_content_hash, content_hash = _live_hashes(source_bytes, normalized)
    source_url = _live_source_url(url, meta)
    raw_path, retrieved_at, warnings = _persist_live_document(
        accession_no=accession_no,
        doc_name=doc_name,
        source_bytes=source_bytes,
        representation=representation,
        normalized=normalized,
        source_content_hash=source_content_hash,
        content_hash=content_hash,
        meta=meta,
        source_url=source_url,
        data_root=data_root,
    )
    payload = source_bytes if representation == "source_bytes" else None
    full, representation, binary_warnings = _readable_document_text(normalized, representation, payload)
    view, resolved, base, is_raw, sections = _build_view(full, section=section, query=query, raw=raw)
    return _bounded_response(
        accession_no=accession_no,
        document_name=doc_name,
        description=description,
        url=url,
        full_text=full,
        content_hash=content_hash,
        source_content_hash=source_content_hash,
        source_representation=representation,
        raw_archive_path=raw_path,
        source_url=source_url,
        filed_at=meta.filed_at,
        known_at=meta.known_at or meta.filed_at,
        retrieved_at=retrieved_at,
        offset=offset,
        max_chars=max_chars,
        cache_hit=False,
        cache_type="live_or_edgartools_http",
        warnings=(warnings + binary_warnings) or None,
        view_text=view,
        view_base=base,
        resolved_section=resolved,
        raw_view=is_raw,
        query=query,
        available_sections=sections,
    )


def _checked_live_meta(accession_no: str, as_of: str | None) -> tuple[EdgarFiling, Filing]:
    filing = _filing(accession_no)
    meta = filing_from_edgar(filing)
    if as_of is not None:
        value, _basis = pit_of(meta)
        if value is None or value[:10] > as_of:
            raise ValueError(f"filing {accession_no!r} not known as of {as_of!r}")
    return filing, meta


def _coerce_window_alias(
    offset: int, max_chars: int | None, cursor: str | float | None, limit: str | float | None
) -> tuple[int, int | None]:
    """cursor/limit aliases win over offset/max_chars; None falls back to the legacy value."""
    return _check_window(cursor if cursor is not None else offset, limit if limit is not None else max_chars)


def get_sec_document(
    accession_no: str,
    document_name: str | None = None,
    as_of: str | None = None,
    *,
    offset: int = 0,
    max_chars: int | None = None,
    data_root: Path | str | None = None,
    section: str | None = None,
    query: str | None = None,
    cursor: str | float | None = None,
    limit: str | float | None = None,
    raw: bool = False,
) -> dict[str, object]:
    """Exact document retrieval; EFTS callers pass the matched document name.
    Primary-document fallback applies only when document_name is None.

    Archive-first: stored revisions under the selected root win with
    ``known_at <= as_of``; a local miss fetches live once and archives
    nothing (selective retention: only evidence acceptance archives exact
    bytes). Raw filing bytes stay addressable via
    ``raw_archive_path``/``source://sec/<accession>/<document>`` once
    accepted; the bounded ``text`` is a derived rendered view
    (section/query narrow it, cursor/limit paginate it) with per-span
    ``source_refs``. Pass ``raw=True`` for the bounded raw-source window
    instead. Omitted ``max_chars``/``limit`` returns the full stored text
    for internal callers; model callers pass a bound.
    """
    from .filings import _check_as_of

    offset, max_chars = _coerce_window_alias(offset, max_chars, cursor, limit)
    as_of = _check_as_of(as_of)
    accession_no = accession_no if isinstance(accession_no, str) else str(accession_no)
    _filing_row, rows = _stored_candidates(accession_no, document_name, as_of, data_root)
    if rows:
        return _archived_response(
            accession_no,
            document_name,
            rows,
            as_of=as_of,
            offset=offset,
            max_chars=max_chars,
            section=section,
            query=query,
            raw=raw,
        )
    filing, meta = _checked_live_meta(accession_no, as_of)
    attachment = _resolve_in(filing, accession_no, document_name)
    if isinstance(attachment, str):
        return {
            "accession_no": accession_no,
            "document_name": attachment,
            "description": None,
            "url": "",
            "text": attachment,
        }
    return _live_response(
        accession_no,
        document_name,
        attachment=attachment,
        meta=meta,
        offset=offset,
        max_chars=max_chars,
        data_root=data_root,
        section=section,
        query=query,
        raw=raw,
    )


def get_sec_filing_text(accession_no: str, document_name: str | None = None, as_of: str | None = None) -> str:
    text = get_sec_document(accession_no, document_name, as_of=as_of, raw=True)["text"]
    if not isinstance(text, str):
        raise ValueError(f"no text for accession: {accession_no!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return text


_EXHIBIT_NORM_RE = re.compile(r"^EX-\d{1,3}(?:\.\d{1,3}[A-Z]?)?$")
_BARE_EXHIBIT_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}[A-Z]?)?)$")
_FILENAME_EXHIBIT_RE = re.compile(r"ex[\s_\-]*(\d{1,3})(?:[\s_\-.]+(\d{1,3}[a-z]?))?", re.IGNORECASE)
_WARRANT_RE = re.compile(r"\bwarrants?\b")
_SOW_RE = re.compile(r"\bsow\b")
_MSA_RE = re.compile(r"\bmsa\b")

_ORDER_FORM_PHRASES = ("order form", "order schedule", "work order", "statement of work")
_PRESENTATION_PHRASES = (
    "investor presentation",
    "corporate presentation",
    "investor deck",
    "earnings presentation",
    "analyst presentation",
    "investor day",
)
_CREDIT_PHRASES = (
    "credit agreement",
    "credit facility",
    "loan agreement",
    "loan and security",
    "term loan",
    "revolving credit",
    "credit and guarant",
)
_SERVICE_PHRASES = (
    "service agreement",
    "services agreement",
    "master service",
    "professional service",
    "managed service",
)

_DOCUMENT_KEYWORD_RULES = (
    ("amendment", ("amend",), ()),
    ("order_form", _ORDER_FORM_PHRASES, (_SOW_RE,)),
    ("investor_presentation", _PRESENTATION_PHRASES, ()),
    ("warrant", (), (_WARRANT_RE,)),
    ("credit_agreement", _CREDIT_PHRASES, ()),
    ("service_agreement", _SERVICE_PHRASES, (_MSA_RE,)),
)
"""Keyword label, phrase hits, regex hits: one table in priority order."""


def _normalize_exhibit(value: object) -> str | None:
    """Canonical exhibit number (``EX-10.1``) from ``EX 10.1``/``EX10.1``/bare ``10.1``; None otherwise."""
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    if text.startswith("EXHIBIT"):
        text = text[len("EXHIBIT") :].strip()
    text = text.replace(" ", "").replace("_", "")
    if text.startswith("EX") and not text.startswith("EX-"):
        text = "EX-" + text[2:]
    if _EXHIBIT_NORM_RE.match(text):
        return text
    bare = _BARE_EXHIBIT_RE.match(text)
    return f"EX-{bare.group(1)}" if bare else None


def _exhibit_from_filename(value: object) -> str | None:
    """Exhibit number encoded in a filing filename (``ex991.htm`` -> ``EX-99.1``); None when absent."""
    if not isinstance(value, str):
        return None
    match = _FILENAME_EXHIBIT_RE.search(value)
    if match is None:
        return None
    base, sub = match.group(1), match.group(2)
    if sub is None and len(base) == 3:
        base, sub = base[:2], base[2:]  # run-together EX-10.x/EX-99.x convention
    number = (f"EX-{base}" if sub is None else f"EX-{base}.{sub}").upper()
    return number if _EXHIBIT_NORM_RE.match(number) else None


def _attachment_exhibit_of(attachment: object) -> str | None:
    """Canonical exhibit number for one live attachment (document_type wins, filename backstops)."""
    doc_type = attachment.document_type if hasattr(attachment, "document_type") else None
    number = _normalize_exhibit(doc_type)
    if number is not None:
        return number
    doc_name = attachment.document if hasattr(attachment, "document") else None
    return _exhibit_from_filename(doc_name)


def _document_haystack(document_name: object, description: object) -> str | None:
    """Lowercased filename+description text; None when untrusted input will not coerce."""
    try:
        return f"{document_name or ''} {description or ''}".lower()
    except Exception:  # noqa: BLE001 - untrusted display strings coerce to other, never raise
        return None


def _keyword_document_kind(hay: str) -> str | None:
    """First keyword-table hit in priority order; None when only exhibit rules can fire."""
    for label, phrases, patterns in _DOCUMENT_KEYWORD_RULES:
        if any(p in hay for p in phrases) or any(rx.search(hay) for rx in patterns):
            return label
    return None


def _exhibit_document_kind(hay: str, number: str | None) -> str:
    """Exhibit-number defaults after keywords (EX-99.1 press release, EX-10.* contract)."""
    press = "press release" in hay or "pressrelease" in hay or number == "EX-99.1"
    material = (
        (number is not None and number.startswith("EX-10"))
        or "material contract" in hay
        or "definitive agreement" in hay
    )
    return "press_release" if press else ("material_contract" if material else "other")


def classify_document(exhibit: object = None, document_name: object = None, description: object = None) -> str:
    """Deterministic document kind from exhibit number, filename, and description title.

    Labels: ``amendment`` | ``order_form`` | ``investor_presentation`` | ``warrant``
    | ``credit_agreement`` | ``service_agreement`` | ``press_release`` | ``material_contract``
    | ``other``. Keyword hits precede the exhibit-number defaults (bare ``EX-99.1`` reads as a
    press release, bare ``EX-10.*`` as a material contract); filenames like ``ex991.htm``
    backstop a missing exhibit number. Never raises: unparsable input reads as ``other``.

    Chain: ``list_sec_documents`` -> ``classify_document(d.document_type, d.document_name,
    d.description)`` -> ``get_sec_document`` for the exhibit text.
    """
    hay = _document_haystack(document_name, description)
    if hay is None:
        return "other"
    kind = _keyword_document_kind(hay)
    if kind is not None:
        return kind
    return _exhibit_document_kind(hay, _normalize_exhibit(exhibit) or _exhibit_from_filename(document_name))


def _exhibit_dict(accession_no: str, attachment: object) -> dict[str, object]:
    def _get(name: str) -> object:
        try:
            value: object = getattr(attachment, name)
            return value
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            return None

    url = _get("url") or ""
    return {
        "accession_no": accession_no,
        "exhibit": _get("document_type"),
        "description": _get("description"),
        "document": _get("document"),
        "url": url if isinstance(url, str) else "",
    }


def get_filing_exhibits(accession_no: str) -> list[dict[str, object]]:
    filing = _filing(accession_no)
    exhibits = getattr(filing, "exhibits", None)
    if exhibits is None:
        exhibits = filing.attachments
    return [_exhibit_dict(accession_no, a) for a in exhibits]


def get_filing_exhibit(accession_no: str, exhibit: str) -> dict[str, object]:
    want = _normalize_exhibit(exhibit) or exhibit.upper()
    for row in get_filing_exhibits(accession_no):
        if _normalize_exhibit(row.get("exhibit")) == want or str(row.get("exhibit") or "").upper() == want:
            return row
    raise ValueError(f"exhibit not found: {exhibit!r}")
