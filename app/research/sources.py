"""Read immutable source captures and link each one to its official human page.

Single-shot request scope: the bridge wraps one tool request in
:func:`capture_scope`. Live SEC retrieval retains its exact bytes through
:func:`retain_sec_bytes`; :func:`capture_result` archives them only when the
runtime accepted the result. Normal kernel fetch behavior stays unchanged.

Viewer reads of SEC/FINRA record sources fetch the official page once on first
view (``resolve_sec_official`` / ``resolve_finra_official``). The resolver
archives those exact bytes, and every later read uses the archived bytes.

Web highlights use unique quotes in held provider content after visible-text
extraction. Snippets and PDF URLs keep their original links without a new
highlight. Extraction never changes the captured content or its hash.

Citation labels (:func:`read_source_label`) read persisted metadata only: no
replay, no official page, no network. :func:`read_source_view` is the viewer
entry: ordinary failures become a controlled error state with one internal
failure class (:data:`FAILURE_REASONS`) in the ERROR log. Reads never write
evidence or verification.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeGuard
from urllib.parse import quote, unquote, urlparse

from app.config import get_data_root
from app.research.repository import ResearchRepository
from app.storage import raw_archive

logger = logging.getLogger(__name__)

SOURCE_ID_RE = re.compile(r"(?:capture:[a-f0-9]{64}|evidence:[A-Za-z0-9:_-]{1,256})\Z")
_HASH_RE = re.compile(r"[a-f0-9]{64}\Z")
# ponytail: fixed capture bound, per-row cap when total bytes matter.
_MAX_CAPTURE_BYTES = 256 * 1024
_SOURCE_PREFIX_CHARS = 120
# Official human pages only; API hosts (data.sec.gov, efts.sec.gov, api.finra.org) never qualify.
_OFFICIAL_HOSTS = frozenset({"www.sec.gov", "cdn.finra.org"})
_OFFICIAL_MEDIA = frozenset({"html", "text", "pdf"})
# app.sec.documents._render_text puts each block on its own line and joins table cells with " | ".
_BLOCK_SEP = re.compile(r"\n| \| ")
_SEP_CHARS = " \t\n|"
_TEXT_DIRECTIVE = ":~:text="


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
    """Return server-created references for durable, replayable evidence rows and calculations."""
    from app.research.synthesis.final import _is_calculation

    evidence_id = _text(record, "evidence_id")
    if evidence_id and SOURCE_ID_RE.fullmatch(f"evidence:{evidence_id}") and _is_calculation(record):
        # The calculation's Stockbot view links each input record; it never borrows one input's original.
        return [{"id": f"evidence:{evidence_id}", "title": "Stockbot calculation"}]
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
    finra = is_finra_tool(name)
    _archive_capture(
        capture,
        {
            "title": name,
            "publisher": "FINRA" if finra else "SEC",
            "toolName": name,
            # A bridge result cites no single row: the resolver reads the page from the whole payload.
            "record": {
                "kind": "finra_record" if finra else "sec_record",
                "tool": name,
                "row": {},
                "payload": dict(result),
            },
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
    digests = {r.sha256 for r in records if r.metadata.get("evidence_id") == evidence.get("evidence_id")}
    # Two archived revisions claiming one evidence row are incompatible candidates: never pick the first.
    if len(digests) > 1:
        raise ValueError("Source capture is ambiguous")
    return next(iter(digests), "")


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
        raise FileNotFoundError("Source replay identity is missing")
    tool_result = _replayed_result(store, result_id)
    if tool_result.get("session_id") != evidence.get("session_id"):
        raise ValueError("Source provenance mismatch")
    payload = tool_result.get("result")
    if not isinstance(payload, Mapping):
        raise FileNotFoundError("Source replay is missing")
    return tool_result


def _excerpt_row(rows: object, provenance: Mapping[str, object], quoted: str) -> Mapping[str, object] | None:
    for row in rows if isinstance(rows, list) else []:
        if (
            isinstance(row, Mapping)
            and _text(row, "url") == _text(provenance, "url")
            and " ".join(quoted.split()) in " ".join(_text(row, "highlight").split())
        ):
            return row
    return None


def _pinned_web_document(
    common: dict[str, object], provenance: Mapping[str, object], tool_result: Mapping[str, object] | None = None
) -> dict[str, object] | None:
    quoted = _text(provenance, "excerpt")
    content = _text(provenance, "source_content")
    if not content:
        return None
    if tool_result is not None:
        payload = tool_result.get("result")
        rows = payload.get("evidence") if isinstance(payload, Mapping) else None
        if _excerpt_row(rows, provenance, quoted) is None:
            raise FileNotFoundError("Source excerpt is missing")
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
    document = _pinned_web_document(common, provenance, tool_result) or _replayed_web_document(
        common, provenance, tool_result
    )
    return {**document, "toolName": "search_web", **_tool_args(tool_result, "query", "search_type")}


def _record_row(provenance: Mapping[str, object]) -> Mapping[str, object]:
    """The cited row: record_identity holds the row as JSON when the kernel cited one row."""
    try:
        row = json.loads(_text(provenance, "record_identity"))
    except ValueError:
        return {}
    return row if isinstance(row, Mapping) else {}


def _resolve_structured_record(
    kind: str, common: dict[str, object], provenance: Mapping[str, object], tool_result: Mapping[str, object]
) -> dict[str, object]:
    if tool_result.get("tool_name") != provenance.get("tool_name"):
        raise ValueError("Source provenance mismatch")
    raw = tool_result.get("result")
    return {
        **common,
        **_tool_args(tool_result, "query" if kind == "finra_record" else "subject"),
        "record": {
            "kind": kind,
            "tool": _text(provenance, "tool_name"),
            "row": dict(_record_row(provenance)),
            "payload": dict(raw) if isinstance(raw, Mapping) else {},
        },
    }


@dataclass
class _Resolved:
    """One source before rendering: the viewer document plus what renders its original and audit."""

    document: dict[str, object]
    kind: str
    sec: Mapping[str, object] | None = None
    record: Mapping[str, object] | None = None
    evidence: Mapping[str, object] | None = None
    store: object = None


def _resolved(kind: str, document: dict[str, object]) -> _Resolved:
    sec, record = document.pop("sec", None), document.pop("record", None)
    identity = sec if isinstance(sec, Mapping) else None
    cited = record if isinstance(record, Mapping) else None
    kind = kind or (_text(cited, "kind") if cited else "") or ("sec_source" if identity else "web_source")
    return _Resolved(document, kind, identity, cited)


def _calculation_document(evidence: Mapping[str, object]) -> dict[str, object]:
    """A calculation has no single original: the viewer shows its formula and audits each input."""
    metadata = evidence.get("metadata")
    metric = _text(metadata, "metric") if isinstance(metadata, Mapping) else ""
    content = _text(evidence, "content")
    return {
        "title": (metric or _text(evidence, "claim_text") or _text(evidence, "subject") or "Calculation")[:120],
        "publisher": "Stockbot",
        "capturedAt": _text(evidence, "retrieved_at"),
        "knownAt": _text(evidence, "known_at"),
        "representation": "text",
        "captureKind": "calculation",
        "content": content,
        "contentHash": _text(evidence, "content_hash") or raw_archive.content_hash(content.encode()),
        "quote": "",
    }


def _evidence_document(store: object, evidence: Mapping[str, object], root: Path) -> tuple[str, dict[str, object]]:
    from app.research.synthesis.final import _is_calculation

    if _is_calculation(evidence):
        return "calculation", _calculation_document(evidence)
    kind, common = _evidence_provenance(evidence)
    provenance = evidence.get("provenance")
    assert isinstance(provenance, Mapping)
    if kind not in _REPLAYABLE_KINDS:
        raise FileNotFoundError("Source identity is missing")
    if kind == "sec_source":
        return kind, _resolve_sec_source(evidence, provenance, common, root)
    tool_result = _replay_tool_result(store, evidence, provenance)
    if kind == "web_source":
        return kind, _resolve_web_source(common, provenance, tool_result)
    return kind, _resolve_structured_record(kind, common, provenance, tool_result)


def _resolve(source_id: str, root: Path, repo: object = None) -> _Resolved:
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise ValueError("Source id is invalid")
    if source_id.startswith("capture:"):
        return _resolved("", _resolve_capture(source_id, root))
    store = repo if _has_reads(repo) else ResearchRepository(data_root=root)
    evidence = store.get_evidence(source_id.removeprefix("evidence:"))
    resolved = _resolved(*_evidence_document(store, evidence, root))
    resolved.evidence, resolved.store = evidence, store
    return resolved


def _official_url(value: str) -> str:
    """The URL when it is a clean https official human page, else blank."""
    parsed = urlparse(value)
    official = parsed.scheme == "https" and parsed.netloc in _OFFICIAL_HOSTS
    return value if official and not parsed.fragment else ""


@dataclass(frozen=True)
class _Official:
    """A resolver's official page: exact archived bytes plus the cited passage in them."""

    url: str
    payload: bytes
    sha256: str
    media: str
    quote: str
    anchor: str


def _checked_official(found: Mapping[str, object]) -> _Official:
    """Trust boundary: official https host, bytes that match their hash, a known media."""
    url, payload, quoted, anchor = (found.get(key) for key in ("url", "payload", "quote", "anchor"))
    if (
        not isinstance(url, str)
        or not url
        or _official_url(url) != url
        or not isinstance(payload, bytes)
        or found.get("sha256") != raw_archive.content_hash(payload)
        or found.get("media") not in _OFFICIAL_MEDIA
        or not isinstance(quoted, str)
        or not isinstance(anchor, str)
    ):
        raise ValueError("Official source page is invalid")
    return _Official(url, payload, raw_archive.content_hash(payload), _text(found, "media"), quoted, anchor)


def _record_dataset(document: Mapping[str, object], payload: Mapping[str, object]) -> str:
    group, name = _text(payload, "group"), _text(payload, "dataset")
    return _text(document, "dataset") or _text(payload, "dataset_id") or (f"{group}/{name}" if group and name else "")


def _official_page(document: Mapping[str, object], record: Mapping[str, object], root: Path) -> _Official | None:
    """The record's official page. The resolver fetches it on first view, then reads its archive."""
    row, payload = record.get("row"), record.get("payload")
    row = row if isinstance(row, Mapping) else {}
    payload = payload if isinstance(payload, Mapping) else {}
    if record.get("kind") == "finra_record":
        from app.finra_client import resolve_finra_official

        found = resolve_finra_official(_record_dataset(document, payload), row, payload, root=root)
    else:
        from app.sec.documents import resolve_sec_official

        found = resolve_sec_official(_text(record, "tool"), row, payload, root=root)
    return None if found is None else _checked_official(found)


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


def _render_sec_text(document: dict[str, object], payload: bytes) -> tuple[str, bool]:
    """Show the original bytes. Return the basis that location offsets address and whether it is HTML."""
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
    return basis, html


def _word_char(text: str, at: int) -> bool:
    return 0 <= at < len(text) and text[at].isalnum()


def _occurrences(text: str, term: str) -> list[tuple[int, int]]:
    """Every case-insensitive occurrence of term, overlapping ones included."""
    pattern = re.compile(f"(?=({re.escape(term)}))", re.IGNORECASE)
    return [(match.start(), match.start() + len(match[1])) for match in pattern.finditer(text)]


def _term_before(text: str, term: str, at: int) -> bool:
    """term ends at `at`, apart from whitespace and block separators."""
    while at and text[at - 1] in _SEP_CHARS:
        at -= 1
    return at >= len(term) and text[at - len(term) : at].lower() == term.lower()


def _term_after(text: str, term: str, at: int) -> bool:
    """term starts at `at`, apart from whitespace and block separators."""
    while at < len(text) and text[at] in _SEP_CHARS:
        at += 1
    return text[at : at + len(term)].lower() == term.lower()


def _ranges(text: str, prefix: str, start: str, end: str, suffix: str) -> list[tuple[int, int]]:
    """Every (start, end) range a browser could highlight for these text-directive terms.

    Terms match case-insensitively at any position. A browser also needs word boundaries,
    so it can only drop ranges from this list. Like a browser, each start takes the first
    later end that the suffix follows.
    """
    ends: list[tuple[int, int]] = _occurrences(text, end) if end else []
    found: list[tuple[int, int]] = []
    for at, start_end in _occurrences(text, start):
        if prefix and not _term_before(text, prefix, at):
            continue
        candidates = [stop for begin, stop in ends if begin >= start_end] if end else [start_end]
        hit = next((stop for stop in candidates if not suffix or _term_after(text, suffix, stop)), None)
        if hit is not None:
            found.append((at, hit))
    return found


def _fragment_ranges(text: str, fragment: str) -> list[tuple[int, int]]:
    """Parse one `:~:text=` directive and list every range it could highlight in text."""
    parts = fragment.partition(_TEXT_DIRECTIVE)[2].split(",")
    prefix = unquote(parts.pop(0)[:-1]) if len(parts) > 1 and parts[0].endswith("-") else ""
    suffix = unquote(parts.pop()[1:]) if len(parts) > 1 and parts[-1].startswith("-") else ""
    terms = [unquote(part) for part in parts]
    return _ranges(text, prefix, terms[0], terms[1] if len(terms) > 1 else "", suffix)


def _term(text: str) -> str:
    # The directive syntax uses "-", "," and "&": percent-encode them along with everything else.
    return quote(text, safe="").replace("-", "%2D")


def _text_fragment(text: str, start: int, end: int, anchor: str) -> str | None:
    """URL fragment that highlights exactly text[start:end] once, else None.

    text is the official page as app.sec.documents._render_text renders it. Each term
    stays inside one block (table cell or line): start is the first quote block, end the
    last. When start..end is not unique, the adjacent blocks become prefix, then suffix,
    then both.
    """
    while start < end and text[start] in _SEP_CHARS:
        start += 1
    while end > start and text[end - 1] in _SEP_CHARS:
        end -= 1
    # A browser matches whole words only: a quote cut mid-word never highlights.
    if start == end or _word_char(text, start - 1) or _word_char(text, end):
        return None
    blocks = [block for block in (part.strip() for part in _BLOCK_SEP.split(text[start:end])) if block]
    first, last = blocks[0], blocks[-1] if len(blocks) > 1 else ""
    before = _BLOCK_SEP.split(text[:start].rstrip(_SEP_CHARS))[-1].strip()
    after = _BLOCK_SEP.split(text[end:].lstrip(_SEP_CHARS), maxsplit=1)[0].strip()
    head = f"#{quote(anchor, safe='')}{_TEXT_DIRECTIVE}"
    for prefix, suffix in dict.fromkeys((("", ""), (before, ""), ("", after), (before, after))):
        terms = [
            *([f"{_term(prefix)}-"] if prefix else []),
            _term(first),
            *([_term(last)] if last else []),
            *([f"-{_term(suffix)}"] if suffix else []),
        ]
        fragment = head + ",".join(terms)
        if _fragment_ranges(text, fragment) == [(start, end)]:
            return fragment
    return None


def _quote_fragment(document: Mapping[str, object], basis: str, anchor: str) -> str | None:
    """Text fragment for the document quote: at its verified offset, else its first occurrence."""
    quoted = _text(document, "quote")
    location = document.get("location")
    offset = location.get("offset") if isinstance(location, Mapping) else None
    at = offset if type(offset) is int and 0 <= offset and basis[offset : offset + len(quoted)] == quoted else -1
    at = at if at >= 0 else basis.find(quoted)
    return _text_fragment(basis, at, at + len(quoted), anchor) if quoted and at >= 0 else None


def _web_fragment(document: Mapping[str, object], original: str) -> str | None:
    """Highlight a unique quote in held provider text. Preserve snippet and PDF links without a highlight."""
    from app.sec.documents import _render_text

    parsed = urlparse(original)
    if (
        not original
        or document.get("captureKind") != "provider_extraction"
        or unquote(parsed.path).lower().endswith(".pdf")
        or ":~:" in parsed.fragment
    ):
        return None
    # Mark non-rendered web containers hidden for the existing parser. Keep the captured bytes unchanged.
    content = re.sub(
        r"<(head|title|template|noscript)\b", r"<\1 hidden", _text(document, "content"), flags=re.IGNORECASE
    )
    basis = _render_text(content)
    quoted = _text(document, "quote")
    if not quoted or len(_occurrences(basis, quoted)) != 1:
        return None
    return _quote_fragment(document, basis, unquote(parsed.fragment))


def _pdf_url(source_id: str) -> str:
    return f"/api/research/sources/{quote(source_id, safe='')}/pdf"


def _render_sec_document(
    document: dict[str, object], identity: Mapping[str, object], root: Path, source_id: str
) -> tuple[str, str | None]:
    """Render pinned SEC bytes. Return (official page URL, text fragment)."""
    record = _sec_record(identity, root)
    payload = _checked_bytes(record, root, _text(identity, "sha256"))
    url = _url(record.url)
    document.update({"contentHash": record.sha256, "captureKind": "original", "url": url})
    if payload.startswith(b"%PDF"):
        document.update({"representation": "pdf", "pdfUrl": _pdf_url(source_id)})
        return _official_url(url), None
    basis, html = _render_sec_text(document, payload)
    return _official_url(url), _quote_fragment(document, basis, "") if html else None


def _render_record(
    document: dict[str, object], record: Mapping[str, object], root: Path, source_id: str
) -> tuple[str, str | None, tuple[str, str]]:
    """Render a record's official page like an SEC original.

    Return (official page URL, text fragment, (failure, reason) for a missing page). A failed
    official fetch drops only the public link: the held record still renders, nothing persisted changes.
    A published FINRA file that disagrees with the cited row or holds ambiguous lines also drops the
    public link. The held record renders, and ``sourceConflict`` reports the file conflict apart from
    the API proof verdict in ``audit``. Archive corruption and other errors still raise.
    """
    from app.finra_client import OfficialFileConflict

    try:
        official = _official_page(document, record, root)
        missing = ("PUBLIC_PAGE_UNAVAILABLE", "resolver found no official page for the cited record")
    except OSError as exc:
        official, missing = None, ("SOURCE_UNAVAILABLE", f"official page fetch failed: {exc}")
    except OfficialFileConflict as exc:
        failure = failure_reason(exc)
        official, missing = None, (failure, f"published file conflicts with the cited record: {exc}")
        document["sourceConflict"] = {"failure": failure, "reason": str(exc)}
    if official is None:
        # No official page: the Stockbot view shows the cited row (the whole payload when none was cited).
        row, payload = record.get("row"), record.get("payload")
        shown = row if isinstance(row, Mapping) and row else payload
        content = _bounded(json.dumps(shown, sort_keys=True, ensure_ascii=False, indent=2, default=str))
        document.update(
            {
                "representation": "text",
                "captureKind": "structured_record",
                "content": content,
                "contentHash": raw_archive.content_hash(content.encode()),
                "quote": "",
            }
        )
        return "", None, missing
    document.update(
        {"url": official.url, "captureKind": "original", "contentHash": official.sha256, "quote": official.quote}
    )
    if official.media == "pdf":
        document.update({"representation": "pdf", "pdfUrl": _pdf_url(source_id)})
        return official.url, None, missing
    location: dict[str, object] = {"basis": "rendered"}
    document["location"] = location
    basis, html = _render_sec_text(document, official.payload)
    at = basis.find(official.quote) if official.quote else -1
    if at >= 0:
        location.update({"offset": at, "end": at + len(official.quote)})
        _attach_sec_context(document, location, basis)
    return official.url, _quote_fragment(document, basis, official.anchor) if html else None, missing


def _evidence_id_of(source_id: str) -> str:
    return source_id.removeprefix("evidence:") if source_id.startswith("evidence:") else ""


def _log_missing(source_id: str, resolved: _Resolved, failure: str, reason: str) -> None:
    record = resolved.record or {}
    logger.error(
        "source official page missing: source_id=%s evidence_id=%s kind=%s tool=%s tool_args=%s row=%s reason=%s"
        " failure=%s",
        source_id,
        _evidence_id_of(source_id),
        resolved.kind,
        _text(resolved.document, "toolName") or _text(record, "tool"),
        json.dumps(resolved.document.get("toolArgs", {}), sort_keys=True, default=str),
        json.dumps(record.get("row", {}), sort_keys=True, default=str),
        reason,
        failure,
    )


_AUDIT_TEXT_KEYS = (
    ("value", "value"),
    ("entity", "entity"),
    ("period", "period"),
    ("unit", "unit"),
    ("known_at", "knownAt"),
    ("formula", "formula"),
    ("formulaId", "formulaId"),
    ("formulaVersion", "formulaVersion"),
    ("recomputedValue", "recomputedValue"),
)


def _audit_node(view: Mapping[str, object], rows: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Viewer shape of one audit view: typed value, verification, source identity, inputs. No API URLs.

    A verified node also carries its kernel-built canonical ``datapoint`` descriptor (with its id).
    The replay verdict rebuilds and compares that descriptor, so an unverified node never shows one.
    """
    evidence_id = _text(view, "evidence_id")
    node: dict[str, object] = {
        "evidenceId": evidence_id,
        "claimType": _text(view, "claim_type"),
        "verification": _text(view, "verification") or "unverified",
        "verificationReason": _text(view, "verification_reason"),
        **{name: _text(view, key) for key, name in _AUDIT_TEXT_KEYS if _text(view, key)},
    }
    if evidence_id and SOURCE_ID_RE.fullmatch(f"evidence:{evidence_id}"):
        node["sourceId"] = f"evidence:{evidence_id}"
    raw_source = view.get("source")
    source: Mapping[str, object] = raw_source if isinstance(raw_source, Mapping) else {}
    fields = (("name", "source_name"), ("document", "document"), ("accession", "accession_no"), ("hash", "sourceId"))
    identity = {name: _text(source, key) for name, key in fields if _text(source, key)}
    if identity:
        node["source"] = identity
    proof = rows.get(evidence_id, {}).get("proof")
    if node["verification"] == "verified" and isinstance(proof, Mapping):
        locator = proof.get("locator")
        # Observations hold the descriptor in their locator; calculations hold it on the proof.
        datapoint = proof.get("datapoint") or (locator.get("datapoint") if isinstance(locator, Mapping) else None)
        if isinstance(datapoint, Mapping):
            node["datapoint"] = dict(datapoint)
    inputs = view.get("inputs")
    if isinstance(inputs, list):
        node["inputs"] = [_audit_node(item, rows) for item in inputs if isinstance(item, Mapping)]
    return node


def _repair_summary(store: object, evidence_id: str, session_id: str | None) -> list[dict[str, object]]:
    """Immutable repair history of one legacy row: status and identity, never a verified citation.

    Repairs stay artifact-only (the publication gates never read them), so the
    viewer shows them as audit history: repair id, status, reason, category,
    datapoint, and value. A read failure yields no repairs rather than an error.
    """
    if not evidence_id or not isinstance(session_id, str) or not session_id:
        return []
    try:
        from app.research.repository import ResearchRepository

        repairs = store.list_evidence_repairs(session_id, evidence_id) if isinstance(store, ResearchRepository) else []
    except Exception:  # noqa: BLE001 - best-effort audit history, never blocks the view
        return []
    out: list[dict[str, object]] = []
    for repair in repairs:
        if not isinstance(repair, Mapping):
            continue
        summary = {key: repair.get(key) for key in ("repair_id", "status", "reason", "category", "datapoint", "value")}
        if isinstance(summary.get("repair_id"), str) and summary["repair_id"]:
            out.append({key: value for key, value in summary.items() if value not in (None, "")})
    return out


def _audit(store: object, evidence: Mapping[str, object], source_id: str) -> dict[str, object]:
    """Replay the observation or calculation proof against this session's ledger. Read-only: nothing persists."""
    from app.research.service import _session_rows
    from app.research.synthesis.final import _calculation_view, _is_calculation

    session_id = evidence.get("session_id")
    rows: dict[str, Mapping[str, object]] = {}
    if isinstance(store, ResearchRepository) and isinstance(session_id, str):
        rows.update(_session_rows(store, session_id))
    rows.setdefault(_text(evidence, "evidence_id"), evidence)
    audit = _audit_node(_calculation_view(evidence, rows, ""), rows)
    repairs = _repair_summary(
        store, _text(evidence, "evidence_id"), session_id if isinstance(session_id, str) else None
    )
    if repairs:
        audit["repairs"] = repairs
    if _is_calculation(evidence) and audit["verification"] != "verified":
        logger.error(
            "source calculation invalid: failure=CALCULATION_INVALID source_id=%s evidence_id=%s reason=%s",
            source_id,
            _evidence_id_of(source_id),
            audit["verificationReason"],
        )
    return audit


def read_source_document(source_id: str, *, data_root: Path | None = None, repo: object = None) -> dict[str, object]:
    """Resolve one source into the shared viewer document with its official-page links and proof audit."""
    root = data_root if data_root is not None else get_data_root()
    resolved = _resolve(source_id, root, repo)
    document = resolved.document
    missing = ("PUBLIC_PAGE_UNAVAILABLE", "")
    if resolved.sec is not None:
        original, fragment = _render_sec_document(document, resolved.sec, root, source_id)
        missing = ("PUBLIC_PAGE_UNAVAILABLE", "archived SEC document URL is not an official page")
    elif resolved.record is not None:
        original, fragment, missing = _render_record(document, resolved.record, root, source_id)
    elif resolved.kind == "calculation":
        # Never one arbitrary original for a calculation: its audit links each input record instead.
        original, fragment = "", None
    else:
        original = _url(_text(document, "url"))
        fragment = _web_fragment(document, original)
        missing = ("PUBLIC_PAGE_UNAVAILABLE", "web source has no page URL")
    if original:
        document["originalUrl"] = original
        document["linkStatus"] = "highlighted_original" if fragment else "stockbot_render"
        if fragment:
            document["highlightUrl"] = original.partition("#")[0] + fragment
    else:
        document["linkStatus"] = "missing"
        if missing[1]:
            _log_missing(source_id, resolved, *missing)
    if resolved.evidence is not None:
        document["audit"] = _audit(resolved.store, resolved.evidence, source_id)
    document["id"] = source_id
    return document


FAILURE_REASONS = (
    "PUBLIC_PAGE_UNAVAILABLE",
    "SOURCE_IDENTITY_MISSING",
    "SOURCE_MISMATCH",
    "SOURCE_AMBIGUOUS",
    "SOURCE_UNAVAILABLE",
    "SOURCE_CORRUPT",
    "CALCULATION_INVALID",
)
# First matching message fragment wins. These are this module's and the official resolvers' own messages.
_FAILURE_MESSAGES = (
    ("mbiguous", "SOURCE_AMBIGUOUS"),
    ("hash", "SOURCE_CORRUPT"),
    ("archive path is invalid", "SOURCE_CORRUPT"),
    ("capture is invalid", "SOURCE_CORRUPT"),
    ("page is invalid", "SOURCE_CORRUPT"),
    ("id is invalid", "SOURCE_IDENTITY_MISSING"),
    ("identity", "SOURCE_IDENTITY_MISSING"),
    ("provenance is missing", "SOURCE_IDENTITY_MISSING"),
    ("Pinned source capture is missing", "SOURCE_IDENTITY_MISSING"),
    ("mismatch", "SOURCE_MISMATCH"),
    ("excerpt is missing", "SOURCE_MISMATCH"),
    ("disagrees", "SOURCE_MISMATCH"),
    ("no line for", "SOURCE_MISMATCH"),
)
_SOURCE_ERRORS = {404: "source capture not found", 409: "invalid source capture", 502: "source read failed"}


def failure_reason(exc: BaseException) -> str:
    """Internal failure class of one source read error (never shown to the reader)."""
    if isinstance(exc, KeyError):
        return "SOURCE_UNAVAILABLE"
    message = str(exc)
    reason = next((reason for fragment, reason in _FAILURE_MESSAGES if fragment in message), "")
    return reason or ("SOURCE_CORRUPT" if isinstance(exc, ValueError) else "SOURCE_UNAVAILABLE")


def read_source_view(source_id: str, *, data_root: Path | None = None, repo: object = None) -> dict[str, object]:
    """Viewer read: the document, or a controlled error state. Unexpected failures still raise.

    A failed read never writes evidence or verification; it logs the internal failure class.
    """
    try:
        return read_source_document(source_id, data_root=data_root, repo=repo)
    except (KeyError, OSError, ValueError) as exc:
        missing = isinstance(exc, (KeyError, FileNotFoundError))
        status = 404 if missing else 409 if isinstance(exc, ValueError) else 502
        failure = failure_reason(exc)
        logger.error(
            "source read failed: failure=%s source_id=%s evidence_id=%s error=%s: %s",
            failure,
            source_id,
            _evidence_id_of(source_id) if isinstance(source_id, str) else "",
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        return {"id": source_id, "error": _SOURCE_ERRORS[status], "status": status, "failure": failure}


def read_source_label(source_id: str, *, data_root: Path | None = None, repo: object = None) -> dict[str, object]:
    """Citation label from persisted metadata only: no tool replay, no official page, no network.

    The link is the already pinned human page (archived SEC URL or the web page). Record sources
    get their official page only when the viewer resolves them.
    """
    root = data_root if data_root is not None else get_data_root()
    if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
        raise ValueError("Source id is invalid")
    if source_id.startswith("capture:"):
        document = _resolve_capture(source_id, root)
        url = _text(document, "url")
        original = _official_url(url) if "sec" in document else "" if "record" in document else _url(url)
    else:
        store = repo if _has_reads(repo) else ResearchRepository(data_root=root)
        document, original = _evidence_label(store.get_evidence(source_id.removeprefix("evidence:")), root)
    label = {"id": source_id, "title": _text(document, "title")}
    label.update({key: _text(document, key) for key in ("publisher", "publishedAt", "knownAt") if _text(document, key)})
    if not original:
        return {**label, "linkStatus": "missing"}
    return {**label, "linkStatus": "stockbot_render", "originalUrl": original}


def _evidence_label(evidence: Mapping[str, object], root: Path) -> tuple[Mapping[str, object], str]:
    from app.research.synthesis.final import _is_calculation

    if _is_calculation(evidence):
        return _calculation_document(evidence), ""
    kind, common = _evidence_provenance(evidence)
    provenance = evidence.get("provenance")
    assert isinstance(provenance, Mapping)
    if kind == "sec_source":
        identity = _resolve_sec_source(evidence, provenance, common, root)["sec"]
        assert isinstance(identity, Mapping)
        return common, _official_url(_sec_record(identity, root).url)
    return common, _url(_text(common, "url")) if kind == "web_source" else ""


def read_source_pdf(source_id: str, *, data_root: Path | None = None, repo: object = None) -> bytes:
    """Read only original PDF bytes: a pinned SEC document or a record's archived official page."""
    root = data_root if data_root is not None else get_data_root()
    resolved = _resolve(source_id, root, repo)
    payload = b""
    if resolved.sec is not None:
        payload = _checked_bytes(_sec_record(resolved.sec, root), root, _text(resolved.sec, "sha256"))
    elif resolved.record is not None:
        official = _official_page(resolved.document, resolved.record, root)
        payload = official.payload if official is not None and official.media == "pdf" else b""
    if not payload.startswith(b"%PDF"):
        raise FileNotFoundError("Original PDF capture is missing")
    return payload
