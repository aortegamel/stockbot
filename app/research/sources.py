"""Read immutable source captures. Viewer reads never retrieve live URLs.

Single-shot request scope: the bridge wraps one tool request in
:func:`capture_scope`. Live SEC retrieval retains its exact bytes through
:func:`retain_sec_bytes`; :func:`capture_result` archives them only when the
runtime accepted the result. Normal kernel fetch behavior stays unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeGuard
from urllib.parse import quote, urlencode, urlparse

from app.config import get_data_root
from app.research.repository import ResearchRepository
from app.storage import raw_archive

SOURCE_ID_RE = re.compile(r"(?:capture:[a-f0-9]{64}|evidence:[A-Za-z0-9:_-]{1,256})\Z")
_HASH_RE = re.compile(r"[a-f0-9]{64}\Z")
# ponytail: fixed capture bound, per-row cap when total bytes matter.
_MAX_CAPTURE_BYTES = 256 * 1024
_SOURCE_PREFIX_CHARS = 120


@dataclass
class _Capture:
    root: Path
    sec_bytes: dict[tuple[str, str], tuple[bytes, dict[str, object]]] = field(default_factory=dict)
    references: list[dict[str, str]] = field(default_factory=list)


_capture: ContextVar[_Capture | None] = ContextVar("source_capture", default=None)


@contextmanager
def capture_scope(data_root: str | None = None) -> Generator[list[dict[str, str]]]:
    """Capture only the current bridge request. Normal kernel retention stays unchanged."""
    root = Path(data_root) if data_root and Path(data_root).is_absolute() else get_data_root()
    capture = _Capture(root=root)
    token = _capture.set(capture)
    try:
        yield capture.references
    finally:
        _capture.reset(token)


def retain_sec_bytes(accession: str, document: str, payload: bytes, metadata: dict[str, object]) -> None:
    """Retain bytes from the existing SEC retrieval until the runtime accepts its result."""
    capture = _capture.get()
    if capture is not None:
        capture.sec_bytes[(accession, document)] = (payload, metadata)


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _url(value: str) -> str:
    parsed = urlparse(value)
    # Any userinfo, including a password with an empty username, carries credentials.
    return value if parsed.scheme in ("http", "https") and parsed.hostname and "@" not in parsed.netloc else ""


_REPLAYABLE_KINDS = frozenset({"sec_source", "web_source", "finra_record", "sec_record"})


def _reference_provenance(record: Mapping[str, object]) -> tuple[str, Mapping[str, object]] | None:
    evidence_id = _text(record, "evidence_id")
    source_id = f"evidence:{evidence_id}"
    provenance = record.get("provenance")
    if not evidence_id or not SOURCE_ID_RE.fullmatch(source_id) or not isinstance(provenance, Mapping):
        return None
    if provenance.get("kind") not in _REPLAYABLE_KINDS:
        return None
    return source_id, provenance


def source_references(record: Mapping[str, object]) -> list[dict[str, str]]:
    """Return server-created references for durable, replayable evidence rows."""
    validated = _reference_provenance(record)
    if validated is None:
        return []
    source_id, provenance = validated
    title = _text(provenance, "title") or _text(provenance, "document_name") or _text(record, "source_name")
    url = _url(_text(provenance, "url") or _text(record, "source_uri"))
    return [{"id": source_id, **({"title": title} if title else {}), **({"url": url} if url else {})}]


def _bounded(text: str) -> str:
    return text[:_MAX_CAPTURE_BYTES]


def _archive_capture(capture: _Capture, document: dict[str, object]) -> None:
    payload = json.dumps({"schema": "source_capture", **document}, sort_keys=True, ensure_ascii=False).encode()
    digest = raw_archive.content_hash(payload)
    raw_archive.archive("viewer", "capture", digest, payload, url=_text(document, "url"), root=capture.root / "raw")
    capture.references.append(
        {
            "id": f"capture:{digest}",
            "title": _text(document, "title"),
            **({"url": _text(document, "url")} if _text(document, "url") else {}),
        }
    )


def _cached_highlights(row: Mapping[str, object]) -> list[str]:
    highlights = row.get("highlights")
    return [text for text in highlights if isinstance(text, str) and text] if isinstance(highlights, list) else []


def _web_document(row: Mapping[str, object], quoted: str) -> dict[str, object]:
    content = _bounded(_text(row, "source_content"))
    highlights = [_bounded(text) for text in ([*_cached_highlights(row), _text(row, "highlight")])]
    excerpt = highlights[0] if highlights else ""
    body = content or excerpt
    representation = _text(row, "source_representation") or "text"
    if representation not in ("html", "text"):
        representation = "text"
    return {
        "title": _text(row, "title"),
        "url": _url(_text(row, "url")),
        "publisher": _text(row, "source_domain"),
        "publishedAt": _text(row, "published_at"),
        "capturedAt": _text(row, "retrieved_at"),
        "representation": representation,
        "captureKind": "provider_extraction" if content else "search_excerpt",
        "content": body,
        "contentHash": raw_archive.content_hash(body.encode()),
        "quote": quoted,
    }


def _capture_web_rows(capture: _Capture, result: Mapping[str, object]) -> None:
    rows = result.get("evidence")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping) and _text(row, "highlight") and _url(_text(row, "url")):
            _archive_capture(capture, _web_document(row, _text(row, "highlight")))


def _retain_pending_sec(capture: _Capture, accession: str, document: str, result: Mapping[str, object]) -> None:
    from app.sec.archive import archive_sec_document

    pending = capture.sec_bytes.get((accession, document))
    if pending is None:
        return
    payload, metadata = pending
    archive_sec_document(
        accession,
        document,
        payload,
        url=_text(result, "source_url") or _text(result, "url"),
        metadata=metadata,
        root=capture.root / "raw",
    )


def _sole_sec_ref(result: Mapping[str, object]) -> Mapping[str, object]:
    """Tool's own full-document reference: exactly one mapping ref, else unverified."""
    refs = result.get("source_refs")
    if not isinstance(refs, list) or len(refs) != 1:
        return {}
    sole = refs[0]
    return sole if isinstance(sole, Mapping) else {}


def _verified_handle_basis(handle: object, accession: str, document: str, digest: str) -> str | None:
    """Handle names this exact archived revision on a readable basis, else unverified."""
    if not isinstance(handle, Mapping):
        return None
    if (handle.get("accession_no"), handle.get("document_name"), handle.get("source_content_hash")) != (
        accession,
        document,
        digest,
    ):
        return None
    basis = handle.get("basis")
    if basis not in ("raw", "rendered"):
        return None
    assert isinstance(basis, str)
    return basis


def _reference_addresses_capture(
    ref: Mapping[str, object], result: Mapping[str, object], basis: str, accession: str, document: str
) -> bool:
    """View and reference address the same basis and document as the verified handle."""
    if result.get("view", basis) != basis:
        return False
    if ref.get("basis", basis) != basis:
        return False
    return (ref.get("accession"), ref.get("document")) == (accession, document)


def _full_document_span(start: object, cursor: object, cursor_end: object, text_len: int) -> dict[str, int] | None:
    """View window maps onto the full-document offset; an inconsistent window yields no span."""
    if type(start) is not int or type(cursor) is not int or type(cursor_end) is not int:
        return None
    if not 0 <= cursor <= start:
        return None
    if cursor_end - cursor != text_len:
        return None
    return {"offset": start, "end": start + cursor_end - cursor}


def _sec_span(result: Mapping[str, object], accession: str, document: str, digest: str) -> dict[str, int]:
    """Full-document span from the tool's own reference. An unverified reference yields no span."""
    ref = _sole_sec_ref(result)
    basis = _verified_handle_basis(result.get("source_handle"), accession, document, digest)
    if basis is None:
        return {}
    if not _reference_addresses_capture(ref, result, basis, accession, document):
        return {}
    span = _full_document_span(
        ref.get("offset"), result.get("offset"), result.get("end_offset"), len(_text(result, "text"))
    )
    return span if span is not None else {}


def _capture_sec_document(capture: _Capture, result: Mapping[str, object]) -> None:
    from app.sec.archive import find_archived_document

    accession, document = _text(result, "accession_no"), _text(result, "document_name")
    digest = _text(result, "source_content_hash")
    if not _HASH_RE.fullmatch(digest):
        raise ValueError("SEC source hash is missing or invalid")
    _retain_pending_sec(capture, accession, document, result)
    archived = find_archived_document(accession, document, sha256=digest, root=capture.root / "raw")
    if archived is None:
        raise FileNotFoundError("Pinned SEC original is missing")
    _checked_bytes(archived, capture.root, digest)
    handle = result.get("source_handle")
    basis = handle.get("basis") if isinstance(handle, Mapping) else None
    _archive_capture(
        capture,
        {
            "title": document,
            "url": _url(archived.url),
            "publisher": "SEC",
            "publishedAt": _text(result, "filed_at"),
            "capturedAt": archived.retrieved_at,
            "quote": _bounded(_text(result, "text")),
            "location": {
                "basis": "raw" if basis == "raw" else "rendered",
                **_sec_span(result, accession, document, digest),
            },
            "sec": {"accession_no": accession, "document_name": document, "sha256": digest},
        },
    )


def _capture_structured_record(capture: _Capture, name: str, result: Mapping[str, object]) -> None:
    from app.research.agents.source_agent import is_finra_tool
    from app.research.service import _SEC_REPLAY_TOOLS

    if not (is_finra_tool(name) or name in _SEC_REPLAY_TOOLS):
        return
    content = _bounded(json.dumps(dict(result), ensure_ascii=False, sort_keys=True, indent=2))
    _archive_capture(
        capture,
        {
            "title": name,
            "publisher": "FINRA" if is_finra_tool(name) else "SEC",
            "captureKind": "structured_record",
            "representation": "json",
            "content": content,
            "contentHash": raw_archive.content_hash(content.encode()),
            "quote": "",
        },
    )


def capture_result(name: str, result: Mapping[str, object]) -> None:
    """Archive successful bridge sources after all runtime security gates accept the result."""
    capture = _capture.get()
    if capture is None or result.get("error"):
        return
    if name == "search_web":
        _capture_web_rows(capture, result)
    elif name == "get_sec_document":
        _capture_sec_document(capture, result)
    else:
        _capture_structured_record(capture, name, result)


def _archive_root(root: Path) -> Path:
    return (root / "raw").resolve()


def _checked_bytes(record: raw_archive.ArchiveRecord, root: Path, digest: str) -> bytes:
    _check_digest(record, digest)
    raw_root = _archive_root(root)
    for path in (record.payload_path, record.manifest_path):
        if not path.resolve().is_relative_to(raw_root):
            raise ValueError("Source archive path is invalid")
    payload = _read_payload(record)
    if raw_archive.content_hash(payload) != digest or len(payload) != record.size:
        raise ValueError("Source hash mismatch")
    return payload


def _check_digest(record: raw_archive.ArchiveRecord, digest: str) -> None:
    if not _HASH_RE.fullmatch(digest) or record.sha256 != digest:
        raise ValueError("Source hash mismatch")


def _read_payload(record: raw_archive.ArchiveRecord) -> bytes:
    try:
        return record.payload_path.read_bytes()
    except OSError as exc:
        raise OSError("Source capture read failed") from exc


def _sec_identity(identity: Mapping[str, object]) -> tuple[str, str, str]:
    accession, document, digest = (_text(identity, key) for key in ("accession_no", "document_name", "sha256"))
    if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession) or not document or not _HASH_RE.fullmatch(digest):
        raise ValueError("Source identity is invalid")
    return accession, document, digest


def _sec_record(identity: Mapping[str, object], root: Path) -> raw_archive.ArchiveRecord:
    from app.sec.archive import find_archived_document

    accession, document, digest = _sec_identity(identity)
    record = find_archived_document(accession, document, sha256=digest, root=root / "raw")
    if record is None:
        raise FileNotFoundError("Source capture is missing")
    return record


class _SourceStore(Protocol):
    """Structural read seam: ResearchRepository or a test double."""

    def get_evidence(self, evidence_id: str, /) -> Mapping[str, object]: ...

    def get_tool_result(self, tool_result_id: str, /) -> object: ...


def _has_reads(candidate: object) -> TypeGuard[_SourceStore]:
    get_evidence = getattr(candidate, "get_evidence", None)
    get_tool_result = getattr(candidate, "get_tool_result", None)
    return callable(get_evidence) and callable(get_tool_result)


def _resolve_capture(source_id: str, root: Path) -> dict[str, object]:
    digest = source_id.removeprefix("capture:")
    record = raw_archive.find("viewer", "capture", digest, sha256=digest, root=root / "raw")
    if record is None:
        raise FileNotFoundError("Source capture is missing")
    try:
        document = json.loads(_checked_bytes(record, root, digest))
    except OSError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Source capture is invalid") from exc
    if not isinstance(document, dict) or document.pop("schema", None) != "source_capture":
        raise ValueError("Source capture is invalid")
    return document


def _evidence_provenance(evidence: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    provenance = evidence.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FileNotFoundError("Source provenance is missing")
    how = {
        "toolName": _text(provenance, "tool_name"),
        "toolResultId": _text(provenance, "tool_result_id"),
        "dataset": _text(provenance, "dataset"),
        "knownAt": _text(provenance, "known_at") or _text(evidence, "known_at"),
    }
    return str(provenance.get("kind") or ""), {
        "title": _text(provenance, "title") or _text(provenance, "document_name") or _text(evidence, "source_name"),
        "url": _url(_text(provenance, "url") or _text(evidence, "source_uri")),
        "publisher": _text(evidence, "source_name"),
        "publishedAt": _text(evidence, "published_at"),
        "capturedAt": _text(evidence, "retrieved_at"),
        **{key: value for key, value in how.items() if value},
    }


def _tool_args(tool_result: Mapping[str, object], *echo_keys: str) -> dict[str, object]:
    """Persisted call arguments, else the request the payload echoes (rows persisted before arguments)."""
    args = tool_result.get("arguments")
    if isinstance(args, Mapping):
        return {"toolArgs": dict(args)}
    payload = tool_result.get("result")
    payload = payload if isinstance(payload, Mapping) else {}
    echo = {key: payload[key] for key in echo_keys if payload.get(key) is not None}
    query = echo.get("query")
    if len(echo) == 1 and isinstance(query, Mapping):
        echo = dict(query)
    return {"toolArgs": echo} if echo else {}


def _pinned_sec_digest(evidence: Mapping[str, object], provenance: Mapping[str, object], root: Path) -> str:
    from app.sec.archive import iter_archived_documents

    metadata = evidence.get("metadata")
    digest = _text(metadata, "source_content_hash") if isinstance(metadata, Mapping) else ""
    if digest:
        return digest
    records = iter_archived_documents(
        _text(provenance, "accession_no"), _text(provenance, "document_name"), root=root / "raw"
    )
    return next((r.sha256 for r in records if r.metadata.get("evidence_id") == evidence.get("evidence_id")), "")


def _resolve_sec_source(
    evidence: Mapping[str, object], provenance: Mapping[str, object], common: dict[str, object], root: Path
) -> dict[str, object]:
    digest = _pinned_sec_digest(evidence, provenance, root)
    if not digest:
        raise FileNotFoundError("Pinned source capture is missing")
    return {
        **common,
        "quote": _text(provenance, "passage"),
        "location": {key: provenance[key] for key in ("offset", "end", "basis") if key in provenance},
        # The kernel persists no tool result for document reads; the archived identity names the call.
        "toolName": "get_sec_document",
        "toolArgs": {key: _text(provenance, key) for key in ("accession_no", "document_name")},
        "sec": {
            "accession_no": provenance.get("accession_no"),
            "document_name": provenance.get("document_name"),
            "sha256": digest,
        },
    }


def _replayed_result(store: object, result_id: str) -> Mapping[str, object]:
    get_tool_result = getattr(store, "get_tool_result", None)
    if not callable(get_tool_result):
        raise FileNotFoundError("Source replay is missing")
    try:
        tool_result = get_tool_result(result_id)
    except KeyError as exc:
        raise FileNotFoundError("Source replay is missing") from exc
    if not isinstance(tool_result, Mapping):
        raise FileNotFoundError("Source replay is missing")
    return tool_result


def _checked_replay(
    store: object, evidence: Mapping[str, object], provenance: Mapping[str, object]
) -> Mapping[str, object]:
    result_id = _text(provenance, "tool_result_id")
    if not result_id:
        raise FileNotFoundError("Source replay is missing")
    tool_result = _replayed_result(store, result_id)
    if tool_result.get("session_id") != evidence.get("session_id"):
        raise ValueError("Source provenance mismatch")
    payload = tool_result.get("result")
    if not isinstance(payload, Mapping):
        raise FileNotFoundError("Source replay is missing")
    return tool_result


def _pinned_web_document(common: dict[str, object], provenance: Mapping[str, object]) -> dict[str, object] | None:
    quoted = _text(provenance, "excerpt")
    content = _text(provenance, "source_content")
    if not content:
        return None
    return {
        **common,
        **_web_document(
            {
                "url": _text(provenance, "url"),
                "title": _text(provenance, "title"),
                "source_domain": _text(provenance, "domain"),
                "published_at": _text(provenance, "published_at"),
                "retrieved_at": _text(provenance, "retrieved_at"),
                "highlight": quoted,
                "source_content": content,
                "source_representation": _text(provenance, "source_representation") or "text",
            },
            quoted,
        ),
    }


def _excerpt_row(rows: object, provenance: Mapping[str, object], quoted: str) -> Mapping[str, object] | None:
    for row in rows if isinstance(rows, list) else []:
        if (
            isinstance(row, Mapping)
            and _text(row, "url") == _text(provenance, "url")
            and " ".join(quoted.split()) in " ".join(_text(row, "highlight").split())
        ):
            return row
    return None


def _replayed_web_document(
    common: dict[str, object], provenance: Mapping[str, object], tool_result: Mapping[str, object]
) -> dict[str, object]:
    quoted = _text(provenance, "excerpt")
    payload = tool_result.get("result")
    rows = payload.get("evidence") if isinstance(payload, Mapping) else None
    row = _excerpt_row(rows, provenance, quoted)
    if row is None:
        raise FileNotFoundError("Source excerpt is missing")
    return {**common, **_web_document(row, quoted)}


def _replay_tool_result(
    store: object, evidence: Mapping[str, object], provenance: Mapping[str, object]
) -> Mapping[str, object]:
    return _checked_replay(store, evidence, provenance)


def _resolve_web_source(
    common: dict[str, object], provenance: Mapping[str, object], tool_result: Mapping[str, object]
) -> dict[str, object]:
    if tool_result.get("tool_name") != "search_web":
        raise ValueError("Source provenance mismatch")
    document = _pinned_web_document(common, provenance) or _replayed_web_document(common, provenance, tool_result)
    return {
        **document,
        "toolName": "search_web",
        **_tool_args(tool_result, "query", "search_type"),
        **({"publicUrl": document["url"]} if document.get("url") else {}),
    }


def _record_row(provenance: Mapping[str, object]) -> Mapping[str, object]:
    """The cited row: record_identity holds the row as JSON when the kernel cited one row."""
    try:
        row = json.loads(_text(provenance, "record_identity"))
    except ValueError:
        return {}
    return row if isinstance(row, Mapping) else {}


def _finra_urls(common: Mapping[str, object], payload: Mapping[str, object]) -> dict[str, object]:
    from app.config import FINRA_API_BASE

    group, _, name = (_text(common, "dataset") or _text(payload, "dataset_id")).partition("/")
    if not (group and name):
        return {}
    return {"upstreamUrl": f"{FINRA_API_BASE}/data/group/{quote(group, safe='')}/name/{quote(name, safe='')}"}


_ACCESSION_RE = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}")


def _filing_index_url(row: Mapping[str, object]) -> str:
    cik = row.get("filer_cik") or row.get("subject_cik")
    accession = _text(row, "accession_no")
    if not isinstance(cik, (int, str)) or not str(cik).isdigit():
        return ""
    if not _ACCESSION_RE.fullmatch(accession):
        return ""
    from edgar.urls import build_archive_url

    return _url(build_archive_url(f"data/{int(cik)}/{accession.replace('-', '')}/"))


# Endpoint whose response holds a cited row's values; tools whose rows mix endpoints stay out.
# get_fundamentals keys on the metric its envelope reports.
_SEC_API_FAMILIES: dict[str, str] = {
    "list_sec_filings": "submissions",
    "get_governance_events": "submissions",  # rows are listing metadata; the proxy text goes unused
    "get_fundamentals:overview": "submissions",
    "get_fundamentals:eps": "companyfacts",
    "get_fundamentals:shares_outstanding": "companyfacts",
    "get_xbrl_facts": "companyfacts",
    "search_sec_filings": "efts",
    # Parsed from the filing's full-submission .txt (edgartools Filing.sgml()).
    "get_insider_activity": "archive",
    "get_planned_insider_sales": "archive",
    "get_beneficial_ownership": "archive",
    "get_offering_history": "archive",
    "get_transaction_status": "archive",
    "get_recent_ownership_filings": "archive",
}
_CIK_KEYS = ("subject_cik", "issuer_cik", "filer_cik", "cik")
_FACTS_URL_RE = re.compile(r".*/api/xbrl/companyfacts/CIK([0-9]{10})\.json")


def _cik(*records: Mapping[str, object]) -> int | None:
    """First CIK a record names directly or inside its persisted companyfacts URL."""
    for record in records:
        for key in _CIK_KEYS:
            value = record.get(key)
            if type(value) is int or (isinstance(value, str) and value.isascii() and value.isdigit()):
                return int(value)
        match = _FACTS_URL_RE.fullmatch(_text(record, "source_url"))
        if match:
            return int(match[1])
    return None


def _efts_url(row: Mapping[str, object], payload: Mapping[str, object]) -> str:
    """The EFTS request behind one text hit, from its attempt's query and filters (page offset unrecorded)."""
    attempt_id, attempts = _text(row, "attempt_id"), payload.get("attempts")
    attempt = next(
        (
            a
            for a in (attempts if attempt_id and isinstance(attempts, list) else [])
            if isinstance(a, Mapping) and a.get("backend") == "efts" and a.get("attempt_id") == attempt_id
        ),
        None,
    )
    filters = attempt.get("filters") if attempt is not None else None
    # A ticker scope makes edgartools resolve the CIK itself, so the sent `ciks` is unknown.
    if attempt is None or not isinstance(filters, Mapping) or filters.get("ticker") is not None:
        return ""
    from edgar.search.efts import EFTS_BASE_URL

    # Same parameters, same order as edgar.search.efts.search_filings.
    forms, start, end, cik = (filters.get(key) for key in ("forms", "start_date", "end_date", "cik"))
    params = [("q", _text(attempt, "query"))]
    params += [("forms", ",".join(map(str, forms)))] if isinstance(forms, list) and forms else []
    params += [("dateRange", "custom")] if start or end else []
    params += [(name, str(value)) for name, value in (("startdt", start), ("enddt", end)) if value]
    params += [("ciks", str(cik).strip().zfill(10))] if cik is not None else []
    return f"{EFTS_BASE_URL}?{urlencode(params)}"


def _sec_api_url(tool: str, row: Mapping[str, object], payload: Mapping[str, object]) -> str:
    """The SEC API URL the tool called for this row; blank when unverified or the CIK is unknown."""
    from edgar.urls import build_archive_url, build_company_facts_url, build_submissions_url

    family = _SEC_API_FAMILIES.get(tool) or _SEC_API_FAMILIES.get(f"{tool}:{_text(payload, 'metric')}")
    if family == "efts":
        return _url(_efts_url(row, payload))
    cik = _cik(row, payload)
    if cik is None:
        return ""
    accession = _text(row, "accession_no")
    if family == "submissions":
        return _url(build_submissions_url(cik))
    if family == "companyfacts":
        return _url(build_company_facts_url(cik))
    if family == "archive" and _ACCESSION_RE.fullmatch(accession):
        return _url(build_archive_url(f"data/{cik}/{accession.replace('-', '')}/{accession}.txt"))
    return ""


def _sec_record_urls(provenance: Mapping[str, object], payload: Mapping[str, object]) -> dict[str, object]:
    row = _record_row(provenance)
    candidates = (
        _text(row, "source_url"),
        _text(row, "source"),
        _text(provenance, "source_uri"),
        _text(payload, "source_url"),
        _text(payload, "source"),
    )
    declared = next(filter(None, map(_url, candidates)), "")
    # A filing row's `source` is its sec.gov filing homepage; else build the filing index.
    public = _url(_text(row, "source")) or _filing_index_url(row) or declared
    upstream = _sec_api_url(_text(provenance, "tool_name"), row, payload)
    return {
        **({"upstreamUrl": upstream} if upstream else {}),
        **({"publicUrl": public} if public else {}),
    }


def _resolve_structured_record(
    kind: str, common: dict[str, object], provenance: Mapping[str, object], tool_result: Mapping[str, object]
) -> dict[str, object]:
    if tool_result.get("tool_name") != provenance.get("tool_name"):
        raise ValueError("Source provenance mismatch")
    raw = tool_result.get("result")
    payload: Mapping[str, object] = raw if isinstance(raw, Mapping) else {}
    content = (
        _bounded(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2))
        if isinstance(raw, Mapping)
        else ""
    )
    finra = kind == "finra_record"
    return {
        **common,
        "content": content,
        "contentHash": raw_archive.content_hash(content.encode()),
        "representation": "json",
        "captureKind": "structured_record",
        "quote": _text(provenance, "record_identity"),
        **_tool_args(tool_result, "query" if finra else "subject"),
        **(_finra_urls(common, payload) if finra else _sec_record_urls(provenance, payload)),
    }


def _resolve(source_id: str, root: Path, repo: object = None) -> dict[str, object]:
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise ValueError("Source id is invalid")
    if source_id.startswith("capture:"):
        return _resolve_capture(source_id, root)
    store = repo if _has_reads(repo) else ResearchRepository(data_root=root)
    evidence = store.get_evidence(source_id.removeprefix("evidence:"))
    kind, common = _evidence_provenance(evidence)
    provenance = evidence.get("provenance")
    assert isinstance(provenance, Mapping)
    if kind == "sec_source":
        return _resolve_sec_source(evidence, provenance, common, root)
    tool_result = _replay_tool_result(store, evidence, provenance)
    if kind == "web_source":
        return _resolve_web_source(common, provenance, tool_result)
    if kind in ("finra_record", "sec_record"):
        return _resolve_structured_record(kind, common, provenance, tool_result)
    raise FileNotFoundError("Source capture is missing")


def _rendered_basis(payload: bytes) -> tuple[str, bool]:
    from app.sec.documents import _render_text

    text = payload.decode("utf-8", "replace")
    html = bool(re.search(r"<(?:html|body|div|p|table|tr|td)\b", text, re.IGNORECASE))
    return (_render_text(text) if html else text), html


def _attach_sec_context(document: dict[str, object], location: object, basis: str) -> None:
    start, end = (location.get(key) if isinstance(location, Mapping) else None for key in ("offset", "end"))
    if (
        isinstance(location, dict)
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(basis)
        and basis[start:end] == document.get("quote")
    ):
        location.update(
            {
                "prefix": basis[max(0, start - _SOURCE_PREFIX_CHARS) : start],
                "suffix": basis[end : end + _SOURCE_PREFIX_CHARS],
            }
        )


def _render_sec_text(document: dict[str, object], payload: bytes) -> None:
    location = document.get("location")
    original = payload.decode("utf-8", "replace")
    # Raw-basis offsets address the decoded original bytes, so show those bytes as text.
    if isinstance(location, Mapping) and location.get("basis") == "raw":
        basis, html = original, False
    else:
        basis, html = _rendered_basis(payload)
    document.update({"representation": "html" if html else "text", "content": original if html else basis})
    if html:
        # The viewer maps HTML text nodes onto this SEC renderer output before it trusts offsets.
        document["matchingText"] = basis
    _attach_sec_context(document, location, basis)


def _render_sec_document(
    document: dict[str, object], identity: Mapping[str, object], root: Path, source_id: str
) -> None:
    record = _sec_record(identity, root)
    payload = _checked_bytes(record, root, _text(identity, "sha256"))
    url = _url(record.url)
    document.update({"contentHash": record.sha256, "captureKind": "original", "url": url})
    if url:
        # The archived sec.gov document URL is both the fetched address and its public page.
        document.update({"upstreamUrl": url, "publicUrl": url})
    if payload.startswith(b"%PDF"):
        document.update({"representation": "pdf", "pdfUrl": f"/api/research/sources/{quote(source_id, safe='')}/pdf"})
    else:
        _render_sec_text(document, payload)


def read_source_document(source_id: str, *, data_root: Path | None = None, repo: object = None) -> dict[str, object]:
    """Resolve one immutable source capture into the shared viewer document."""
    root = data_root if data_root is not None else get_data_root()
    document = _resolve(source_id, root, repo)
    identity = document.pop("sec", None)
    if isinstance(identity, Mapping):
        _render_sec_document(document, identity, root, source_id)
    document["id"] = source_id
    return document


def read_source_pdf(source_id: str, *, data_root: Path | None = None, repo: object = None) -> bytes:
    """Read only original, pinned SEC PDF bytes. No provider extraction qualifies."""
    root = data_root if data_root is not None else get_data_root()
    document = _resolve(source_id, root, repo)
    identity = document.get("sec")
    if not isinstance(identity, Mapping):
        raise FileNotFoundError("Original SEC PDF capture is missing")
    record = _sec_record(identity, root)
    payload = _checked_bytes(record, root, _text(identity, "sha256"))
    if not payload.startswith(b"%PDF"):
        raise FileNotFoundError("Original SEC PDF capture is missing")
    return payload
