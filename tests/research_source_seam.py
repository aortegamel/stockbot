"""Shared test seam for evidence admission: a fake SEC archive behind the kernel reload.

`record_evidence` admits an observed fact only by reloading the document window a
canonical ``source_handle`` names and slicing the cited passage out of those bytes.
Tests mint handles here and route the reload to this fake archive, so evidence
admission is exercised end to end without touching the network.
"""

from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest

DEFAULT_ACCESSION = "0000320193-25-000079"
DEFAULT_DOCUMENT = "nvda-20250331.htm"

_DOC_WINDOWS: dict[tuple[str, str], str] = {}
_SOURCE_TIMING: dict[tuple[str, str], tuple[str, str, str]] = {}
_NO_SOURCE_BYTES: set[tuple[str, str]] = set()
_SOURCE_URL: dict[tuple[str, str], str] = {}
_SOURCE_ARCHIVE_PATH: dict[tuple[str, str], str] = {}


def _default_source_url(accession: str, document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{accession.replace('-', '')}/{document}"


def register_document(
    accession: str,
    document: str,
    full: str,
    *,
    known_at: str | None = None,
    filed_at: str | None = None,
    retrieved_at: str | None = None,
    source_url: str | None = None,
) -> None:
    """Pin one exact document revision verbatim (no accumulation).

    ``handle_for`` accumulates, so two handles minted from one key would
    otherwise pin different fulls; the two-window test must use this, never two
    accumulating mints.
    """
    _DOC_WINDOWS[(accession, document)] = full
    _SOURCE_TIMING[(accession, document)] = (
        known_at or "2025-05-01T00:00:00Z",
        filed_at or "2025-04-30",
        retrieved_at or "2025-05-01T00:00:00Z",
    )
    _SOURCE_URL[(accession, document)] = source_url or _default_source_url(accession, document)


def handle_for(
    passage: str,
    *,
    accession: str = DEFAULT_ACCESSION,
    document: str = DEFAULT_DOCUMENT,
    known_at: str | None = None,
    filed_at: str | None = None,
    retrieved_at: str | None = None,
    no_source_bytes: bool = False,
) -> dict[str, object]:
    """Canonical source_handle for one cited passage.

    Passages accumulate under the document key and the window is the span this
    passage occupies, so reloading the same coordinates reproduces exactly these
    bytes and a different passage at the same coordinates fails the hash check.
    """
    window = _DOC_WINDOWS.get((accession, document), "")
    offset = len(window) + (1 if window else 0)
    full = f"{window}\n{passage}" if window else passage
    _DOC_WINDOWS[(accession, document)] = full
    _SOURCE_TIMING[(accession, document)] = (
        known_at or "2025-05-01T00:00:00Z",
        filed_at or "2025-04-30",
        retrieved_at or "2025-05-01T00:00:00Z",
    )
    if no_source_bytes:
        _NO_SOURCE_BYTES.add((accession, document))
    return {
        "accession_no": accession,
        "document_name": document,
        "basis": "rendered",
        "section": None,
        "query": None,
        "offset": offset,
        "max_chars": len(passage),
        "length": len(passage),
        "text_hash": sha256(passage.encode("utf-8")).hexdigest(),
        "source_content_hash": sha256(full.encode("utf-8")).hexdigest(),
        "source_url": _SOURCE_URL.get((accession, document)) or _default_source_url(accession, document),
        "source_uri": f"source://sec/{accession}/{document}",
    }


def handle_for_registered(
    passage: str,
    *,
    accession: str = DEFAULT_ACCESSION,
    document: str = DEFAULT_DOCUMENT,
) -> dict[str, object]:
    """Canonical source_handle for a passage of the registered revision."""
    full = _DOC_WINDOWS.get((accession, document))
    if full is None:
        raise ValueError(f"test seam: no registered document for {(accession, document)!r}")
    offset = full.index(passage)
    return {
        "accession_no": accession,
        "document_name": document,
        "basis": "rendered",
        "section": None,
        "query": None,
        "offset": offset,
        "max_chars": len(passage),
        "length": len(passage),
        "text_hash": sha256(passage.encode("utf-8")).hexdigest(),
        "source_content_hash": sha256(full.encode("utf-8")).hexdigest(),
        "source_url": _SOURCE_URL.get((accession, document)) or _default_source_url(accession, document),
        "source_uri": f"source://sec/{accession}/{document}",
    }


def fake_sec_document(
    accession_no: str,
    document_name: str | None = None,
    as_of: str | datetime | None = None,
    *,
    offset: int = 0,
    max_chars: int | None = None,
    data_root: object = None,
    section: str | None = None,
    query: str | None = None,
    cursor: object = None,
    limit: object = None,
    raw: bool = False,
) -> dict[str, object]:
    """The archive reload: the registered window, sliced to the requested coordinates."""
    window = _DOC_WINDOWS.get((accession_no, document_name or ""))
    if window is None:
        raise ValueError(f"test seam: no registered document for {(accession_no, document_name)!r}")
    known_at, filed_at, retrieved_at = _SOURCE_TIMING.get(
        (accession_no, document_name or ""),
        ("2025-05-01T00:00:00Z", "2025-04-30", "2025-05-01T00:00:00Z"),
    )
    bound = as_of.date().isoformat() if isinstance(as_of, datetime) else as_of
    if bound is not None and known_at[:10] > bound[:10]:
        raise ValueError(f"filing {accession_no!r} not known as of {as_of!r}")
    end = len(window) if max_chars is None else min(offset + max_chars, len(window))
    key = (accession_no, document_name or "")
    return {
        "accession_no": accession_no,
        "document_name": document_name,
        "text": window[offset:end],
        "content_hash": "0" * 64,
        "source_content_hash": sha256(window.encode("utf-8")).hexdigest(),
        "source_representation": "source_bytes",
        "known_at": known_at,
        "filed_at": filed_at,
        "retrieved_at": retrieved_at,
        "source_url": _SOURCE_URL.get(key) or _default_source_url(accession_no, document_name or ""),
        "raw_archive_path": _SOURCE_ARCHIVE_PATH.get(key),
        "source_uri": f"source://sec/{accession_no}/{document_name}",
    }


def _fake_exact_source_bytes(accession: str, document: str) -> tuple[bytes, str]:
    """Full accumulated document bytes; unknown or byte-less keys fail closed."""
    window = _DOC_WINDOWS.get((accession, document))
    if window is None or (accession, document) in _NO_SOURCE_BYTES:
        raise ValueError("record_evidence: ERR_NO_SOURCE_BYTES (no exact document bytes)")
    return window.encode("utf-8"), "source_bytes"


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the fake archive and route evidence admission reloads to it."""
    _DOC_WINDOWS.clear()
    _SOURCE_TIMING.clear()
    _NO_SOURCE_BYTES.clear()
    _SOURCE_URL.clear()
    _SOURCE_ARCHIVE_PATH.clear()
    monkeypatch.setattr("app.sec.documents.get_sec_document", fake_sec_document)
    monkeypatch.setattr("app.research.service._exact_source_bytes", _fake_exact_source_bytes)


def proven_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passages: dict[str, str]) -> str:
    """A research session whose ledger holds one archived SEC observation per evidence id.

    Final gates verify claims against this ledger (content hash + archive bytes),
    so a gate test cites these ids instead of ids the ledger never admitted.
    """
    from app.research import service
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "data"))
    install(monkeypatch)
    register_document(DEFAULT_ACCESSION, DEFAULT_DOCUMENT, "\n".join(passages.values()))
    sid = service.create_research("support?", "support?", as_of="2025-06-30T00:00:00+00:00")
    job = ResearchRepository().list_jobs(sid)[0].job_id
    for eid, passage in passages.items():
        item: dict[str, object] = {
            "evidence_id": eid,
            "content": passage,
            "claim_text": passage,
            "subject": "NVDA",
            "source_name": "SEC",
            "source_record_id": DEFAULT_ACCESSION,
            "document_name": DEFAULT_DOCUMENT,
            "matching_passage": passage,
            "known_at": "2025-05-01T00:00:00+00:00",
            "source_handle": handle_for_registered(passage),
        }
        service.record_evidence(sid, job, item)
    return sid
