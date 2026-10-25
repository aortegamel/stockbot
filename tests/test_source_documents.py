"""Immutable source captures: single-shot archives plus durable evidence reads."""

from __future__ import annotations

import json
import re
import tempfile
import warnings
from collections.abc import Callable, Mapping
from html import escape
from pathlib import Path
from typing import override
from urllib.parse import unquote

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from app.research import sources as _sources
from app.research.models import JSONValue
from app.research.repository import ResearchRepository
from app.research.sources import capture_result, capture_scope, read_source_document, read_source_pdf, source_references
from app.sec import archive as sec_archive
from app.storage import raw_archive

ACC = "0000000000-26-000042"
DOC = "primary.htm"


def _metadata(evidence_id: str) -> dict[str, object]:
    return {"source_content_hash": "x", "evidence_id": evidence_id, "representation": "source_bytes"}


def _sec_result(text: str, digest: str, start: int, *, basis: str = "rendered", **extra: object) -> dict[str, object]:
    """get_sec_document result for one whole-document window, with its own full-document reference."""
    return {
        "text": text,
        "accession_no": ACC,
        "document_name": DOC,
        "source_content_hash": digest,
        "source_handle": {"accession_no": ACC, "document_name": DOC, "basis": basis, "source_content_hash": digest},
        "source_refs": [{"accession": ACC, "document": DOC, "offset": start}],
        "offset": start,
        "end_offset": start + len(text),
        **extra,
    }


class _Repo:
    """Persisted evidence rows plus the tool results they replay."""

    def __init__(
        self, evidence: Mapping[str, Mapping[str, object]], results: Mapping[str, object] | None = None
    ) -> None:
        self._evidence = evidence
        self._results = results or {}

    def get_evidence(self, evidence_id: str) -> Mapping[str, object]:
        return self._evidence[evidence_id]

    def get_tool_result(self, tool_result_id: str) -> object:
        return self._results[tool_result_id]


_Calls = list[tuple[str, dict[str, object], dict[str, object]]]


def _official(monkeypatch: pytest.MonkeyPatch, page: Mapping[str, object] | None, *, finra: bool) -> _Calls:
    """Fake the SEC or FINRA official-page resolver: record each call, return one fixed page (or None)."""
    from app import finra_client
    from app.sec import documents

    calls: _Calls = []

    def resolve(name: str, row: Mapping[str, object], payload: Mapping[str, object], *, root: Path) -> object:
        calls.append((name, dict(row), dict(payload)))
        return None if page is None else dict(page)

    target = (finra_client, "resolve_finra_official") if finra else (documents, "resolve_sec_official")
    monkeypatch.setattr(*target, resolve, raising=False)
    return calls


def _page(url: str, body: bytes, media: str, quoted: str, anchor: str = "") -> dict[str, object]:
    return {
        "url": url,
        "payload": body,
        "sha256": raw_archive.content_hash(body),
        "media": media,
        "quote": quoted,
        "anchor": anchor,
    }


def test_capture_web_result_archives_only_success_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    with capture_scope() as sources:
        capture_result("search_web", {"evidence": [{"url": "bogus", "highlight": "kept"}]})
        capture_result("search_web", {"error": "soft", "soft": True})
        capture_result("search_web", {"evidence": None})
        capture_result(
            "search_web", {"error": "upstream", "evidence": [{"url": "https://example.com/e", "highlight": "x"}]}
        )
        capture_result(
            "search_web",
            {
                "evidence": [
                    {
                        "title": "T",
                        "url": "https://example.com/a",
                        "source_domain": "example.com",
                        "published_at": "2026-01-01",
                        "retrieved_at": "2026-01-02",
                        "highlight": "quoted passage",
                        "source_content": "<p>quoted passage with context</p>",
                        "source_representation": "html",
                    },
                    {"url": "notaurl", "highlight": "dropped"},
                    {"url": "https://example.com/b", "highlight": ""},
                    {"url": "ftp://example.com/a", "highlight": "dropped"},
                    {"url": "https://user@example.com/a", "highlight": "dropped"},
                    {"url": "https://:secret@example.com/a", "highlight": "dropped"},
                ]
            },
        )
    assert len(sources) == 1
    assert sources[0]["id"].startswith("capture:")
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["captureKind"] == "provider_extraction"
    assert document["representation"] == "html"
    assert document["quote"] == "quoted passage"
    assert "quoted passage with context" in str(document["content"])
    assert sources == [{"id": sources[0]["id"], "title": "T", "url": "https://example.com/a"}]
    assert {key: document[key] for key in ("title", "url", "publisher", "publishedAt", "capturedAt")} == {
        "title": "T",
        "url": "https://example.com/a",
        "publisher": "example.com",
        "publishedAt": "2026-01-01",
        "capturedAt": "2026-01-02",
    }


def test_web_provenance_pins_provider_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    from app.research import service as _svc
    from app.research.repository import ResearchRepository

    result = {
        "result_type": "web_search",
        "evidence": [
            {
                "url": "https://example.com/a",
                "title": "T",
                "source_domain": "example.com",
                "published_at": "2026-01-01",
                "retrieved_at": "2026-01-02",
                "highlight": "quoted passage",
                "source_content": "longer provider extraction quoted passage tail",
                "source_representation": "text",
            }
        ],
    }
    store = ResearchRepository(tmp_path / "research.sqlite")
    store.save_tool_result(
        {"tool_result_id": "tr-1", "session_id": "s", "job_id": "j", "tool_name": "search_web", "result": result}
    )
    provenance = _svc._web_provenance(
        {"tool_result_id": "tr-1", "matching_passage": "quoted passage"}, store=store, session_id="s"
    )
    assert provenance["source_content"] == "longer provider extraction quoted passage tail"
    assert provenance["source_representation"] == "text"


def test_capture_search_excerpt_preserved_when_no_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    with capture_scope() as sources:
        capture_result("search_web", {"evidence": [{"url": "https://example.com/a", "highlight": "only"}]})
        capture_result(
            "search_web",
            {
                "evidence": [
                    {
                        "url": "http://example.com/b",
                        "highlight": "plain",
                        "source_content": "plain body",
                        "source_representation": "pdf",
                    }
                ]
            },
        )
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["captureKind"] == "search_excerpt"
    assert document["quote"] == "only"
    assert document["content"] == "only"
    plain = read_source_document(sources[1]["id"], data_root=tmp_path)
    assert plain["url"] == "http://example.com/b"
    assert plain["representation"] == "text"


def test_capture_structured_record_without_official_page_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bridge record capture cites no row: the resolver reads the whole payload, and None is case 3."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    calls = _official(monkeypatch, None, finra=True)
    result = {"dataset_id": "otcMarket/thresholdList", "content": "short interest rows", "row": 1}
    with capture_scope() as sources:
        capture_result("browse_tools", {"content": "ignored navigation"})
        capture_result("query_finra", result)
    assert len(sources) == 1
    with caplog.at_level("ERROR", logger="app.research.sources"):
        document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert calls == [("otcMarket/thresholdList", {}, result)]
    assert document["captureKind"] == "structured_record"
    assert document["representation"] == "text"
    assert json.loads(str(document["content"])) == result
    assert document["publisher"] == "FINRA"
    assert document["linkStatus"] == "missing"
    assert "originalUrl" not in document and "highlightUrl" not in document
    assert [record.levelname for record in caplog.records] == ["ERROR"]
    assert f"source_id={sources[0]['id']}" in caplog.records[0].getMessage()
    assert "kind=finra_record tool=query_finra" in caplog.records[0].getMessage()


def test_capture_sec_result_archives_live_bytes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    payload = b"<html><body>quoted</body></html>"
    digest = sec_archive.archive_sec_document(
        ACC, DOC, payload, url="https://www.sec.gov/x", metadata=_metadata("ignored"), root=tmp_path / "raw"
    ).sha256
    with capture_scope() as sources:
        capture_result("get_sec_document", _sec_result("quoted", digest, 0))
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["captureKind"] == "original"
    assert document["representation"] == "html"
    assert document["contentHash"] == digest
    assert document["content"] == payload.decode()
    assert document["matchingText"] == "quoted"
    assert document["location"] == {"offset": 0, "end": 6, "basis": "rendered", "prefix": "", "suffix": ""}


def test_sec_pdf_capture_resolves_to_original_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    payload = b"%PDF-1.4\nfake pdf bytes\n%%EOF\n"
    digest = sec_archive.archive_sec_document(
        ACC, DOC, payload, url="https://www.sec.gov/y", metadata=_metadata("ignored"), root=tmp_path / "raw"
    ).sha256
    with capture_scope() as sources:
        capture_result("get_sec_document", _sec_result("extracted", digest, 0))
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["representation"] == "pdf"
    assert document["pdfUrl"] == f"/api/research/sources/{sources[0]['id'].replace(':', '%3A')}/pdf"
    assert read_source_pdf(sources[0]["id"], data_root=tmp_path) == payload
    assert document["location"] == {"offset": 0, "end": 9, "basis": "rendered"}
    assert read_source_document(sources[0]["id"])["contentHash"] == digest
    assert read_source_pdf(sources[0]["id"]) == payload
    evidence = _Repo({"ev-pdf": _sec_evidence("ev-pdf", digest, "fake pdf", 9, "rendered")})
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "elsewhere"))
    assert read_source_pdf(sources[0]["id"], data_root=tmp_path) == payload
    assert read_source_pdf("evidence:ev-pdf", data_root=tmp_path, repo=evidence) == payload


def test_sec_scope_retains_live_bytes_without_kernel_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retained bytes archive even when the result pins other bytes, but that result yields no source."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    with capture_scope() as sources:
        _sources.retain_sec_bytes(ACC, DOC, b"live exact bytes", _metadata("bridge-only"))
        with pytest.raises(FileNotFoundError):
            capture_result("get_sec_document", _sec_result("live exact bytes", "0" * 64, 0))
        assert sources == []
    assert sec_archive.find_archived_document(ACC, DOC, root=tmp_path / "raw") is not None


@pytest.mark.parametrize(
    ("digest", "error"),
    [
        (None, ValueError),
        ("", ValueError),
        ("A" * 64, ValueError),
        ("a" * 63, ValueError),
        ("a" * 64, FileNotFoundError),
    ],
)
@pytest.mark.parametrize("text", ["quoted", ""])
def test_sec_capture_without_pinned_original_is_rejected(
    digest: str | None, error: type[Exception], text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or malformed hash, or a hash with no archived original, never becomes a capture."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    result = _sec_result(text, digest or "", 0)
    if digest is None:
        del result["source_content_hash"]
    with capture_scope() as sources, pytest.raises(error):
        capture_result("get_sec_document", result)
    assert sources == []


@pytest.mark.parametrize("digest", ["", "invalid", "a" * 64])
def test_runtime_rejects_sec_result_without_original(
    digest: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import tool_runtime as runtime

    result = _sec_result("quoted", digest, 0)

    def handler(*args: object, **kwargs: object) -> dict[str, object]:
        return result

    monkeypatch.setattr(runtime, "execute_tool", handler)
    with capture_scope(str(tmp_path)) as sources:
        response = runtime.execute_agent_tool(
            "get_sec_document",
            {"accession_no": ACC, "document_name": DOC},
            runtime.RuntimeToolSession(session_id="missing-original"),
            data_root=tmp_path,
        )
    assert response["error_type"] == "source_capture_failed"
    assert "content" not in response and "meta" not in response
    assert sources == []


def test_sec_capture_rejects_missing_archived_payload(tmp_path: Path) -> None:
    record = sec_archive.archive_sec_document(
        ACC, DOC, b"quoted", url="https://www.sec.gov/x", metadata={}, root=tmp_path / "raw"
    )
    record.payload_path.unlink()
    with capture_scope(str(tmp_path)) as sources, pytest.raises(OSError):
        capture_result("get_sec_document", _sec_result("quoted", record.sha256, 0))
    assert sources == []


def test_reader_rejects_paths_urls_and_wrong_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    for bad in ("https://example.com/x", "../raw/x", "evidence:", "bogus:1", "capture:xyz", "", "evidence:a/b"):
        with pytest.raises(ValueError):
            read_source_document(bad, data_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        read_source_document("capture:" + "a" * 64, data_root=tmp_path)


def test_reader_rejects_corrupt_capture_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    for payload in (b"not json", b'{"schema": "wrong"}', b"{}", b"[]"):
        digest = raw_archive.content_hash(payload)
        record = raw_archive.archive("viewer", "capture", digest, payload, url="", root=tmp_path / "raw")
        assert record.sha256 == digest
        with pytest.raises(ValueError):
            read_source_document(f"capture:{digest}", data_root=tmp_path)


def test_sec_evidence_resolves_through_pinned_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    body = "<html><body><p>before quoted passage after</p></body></html>"
    digest = sec_archive.archive_sec_document(
        ACC, DOC, body.encode(), url="https://www.sec.gov/z", metadata=_metadata("ev-1"), root=tmp_path / "raw"
    ).sha256

    class _SecRepo:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            assert evidence_id == "ev-1"
            return {
                "evidence_id": "ev-1",
                "session_id": "s",
                "source_name": "SEC",
                "published_at": "2026-01-01",
                "retrieved_at": "2026-01-02",
                "metadata": {"source_content_hash": digest},
                "provenance": {
                    "kind": "sec_source",
                    "accession_no": ACC,
                    "document_name": DOC,
                    "passage": "quoted passage",
                    "offset": 7,
                    "end": 21,
                    "basis": "rendered",
                },
            }

        def get_tool_result(self, tool_result_id: str) -> dict[str, object]:
            raise AssertionError(tool_result_id)

    assert source_references(
        {
            "evidence_id": "ev-1",
            "source_name": "SEC",
            "provenance": {"kind": "sec_source", "document_name": DOC, "url": "https://www.sec.gov/z"},
        }
    ) == [{"id": "evidence:ev-1", "title": DOC, "url": "https://www.sec.gov/z"}]
    assert source_references({"evidence_id": "", "provenance": {}}) == []
    assert source_references({"provenance": {"kind": "search_run"}}) == []
    document = read_source_document("evidence:ev-1", data_root=tmp_path, repo=_SecRepo())
    location = document["location"]
    assert isinstance(location, Mapping)
    assert document["captureKind"] == "original"
    assert location["prefix"] == "before "
    assert location["suffix"] == " after"
    # Case 1: exact HTML bytes and a directive that matches the cited passage once.
    keys = ("toolName", "toolArgs", "originalUrl", "highlightUrl", "linkStatus")
    assert {key: document.get(key) for key in keys} == {
        "toolName": "get_sec_document",
        "toolArgs": {"accession_no": ACC, "document_name": DOC},
        "originalUrl": "https://www.sec.gov/z",
        "highlightUrl": "https://www.sec.gov/z#:~:text=quoted%20passage",
        "linkStatus": "highlighted_original",
    }
    assert "upstreamUrl" not in document and "publicUrl" not in document
    assert "sec" not in document and "toolResultId" not in document


def test_sec_evidence_without_metadata_hash_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))

    class _MissingRepo:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            return {
                "evidence_id": evidence_id,
                "session_id": "s",
                "provenance": {
                    "kind": "sec_source",
                    "accession_no": ACC,
                    "document_name": DOC,
                    "passage": "quoted",
                },
                "metadata": {},
            }

        def get_tool_result(self, tool_result_id: str) -> dict[str, object]:
            raise AssertionError(tool_result_id)

    with pytest.raises(FileNotFoundError):
        read_source_document("evidence:gone", data_root=tmp_path, repo=_MissingRepo())


def _sec_evidence(evidence_id: str, digest: str, passage: str, offset: int, basis: str) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "session_id": "s",
        "source_name": "SEC",
        "metadata": {"source_content_hash": digest},
        "provenance": {
            "kind": "sec_source",
            "accession_no": ACC,
            "document_name": DOC,
            "passage": passage,
            "offset": offset,
            "end": offset + len(passage),
            "basis": basis,
        },
    }


def test_evidence_reader_uses_configured_research_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.repository import ResearchRepository

    data_root = tmp_path / "data"
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(data_root))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "elsewhere" / "research.sqlite"))
    body = b"<html><body><p>before quoted passage after</p></body></html>"
    digest = sec_archive.archive_sec_document(
        ACC, DOC, body, url="https://www.sec.gov/z", metadata=_metadata("ev-db"), root=data_root / "raw"
    ).sha256
    ResearchRepository().save_evidence(_sec_evidence("ev-db", digest, "quoted passage", 7, "rendered"))
    assert not (data_root / "research.sqlite").exists()
    document = read_source_document("evidence:ev-db", data_root=data_root)
    assert document["contentHash"] == digest
    assert document["quote"] == "quoted passage"


def test_sec_raw_basis_offsets_address_original_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    body = "<html><body><p>before quoted passage after</p></body></html>"
    digest = sec_archive.archive_sec_document(
        ACC, DOC, body.encode(), url="https://www.sec.gov/z", metadata=_metadata("ev-raw"), root=tmp_path / "raw"
    ).sha256
    record = _sec_evidence("ev-raw", digest, "quoted passage", body.index("quoted passage"), "raw")

    class _RawRepo:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            assert evidence_id == "ev-raw"
            return record

        def get_tool_result(self, tool_result_id: str) -> dict[str, object]:
            raise AssertionError(tool_result_id)

    document = read_source_document("evidence:ev-raw", data_root=tmp_path, repo=_RawRepo())
    location = document["location"]
    assert isinstance(location, Mapping)
    assert document["representation"] == "text"
    assert document["content"] == body
    assert location["prefix"] == "<html><body><p>before "
    assert location["suffix"] == " after</p></body></html>"


def test_finra_web_evidence_replays_persisted_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    # A row persisted before call arguments: the Exa payload echo stands in for them.
    payload = {
        "query": "acme news",
        "search_type": "auto",
        "evidence": [{"url": "https://example.com/a", "highlight": "the quoted excerpt here"}],
    }

    class _WebRepo:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            assert evidence_id == "ev-web"
            return {
                "evidence_id": evidence_id,
                "session_id": "s",
                "provenance": {
                    "kind": "web_source",
                    "url": "https://example.com/a",
                    "excerpt": "quoted excerpt",
                    "title": "T",
                    "tool_result_id": "tr-1",
                },
            }

        def get_tool_result(self, tool_result_id: str) -> dict[str, object]:
            assert tool_result_id == "tr-1"
            return {"tool_name": "search_web", "session_id": "s", "result": payload}

    assert (
        source_references(
            {
                "evidence_id": "ev-web",
                "provenance": {"kind": "web_source", "url": "https://example.com/a", "excerpt": "quoted excerpt"},
            }
        )
        != []
    )
    document = read_source_document("evidence:ev-web", data_root=tmp_path, repo=_WebRepo())
    assert document["captureKind"] == "search_excerpt"
    assert document["quote"] == "quoted excerpt"
    # Case 2: a web source holds the provider extraction, not the page bytes.
    keys = ("toolName", "toolResultId", "toolArgs", "originalUrl", "linkStatus")
    assert {key: document.get(key) for key in keys} == {
        "toolName": "search_web",
        "toolResultId": "tr-1",
        "toolArgs": {"query": "acme news", "search_type": "auto"},
        "originalUrl": "https://example.com/a",
        "linkStatus": "stockbot_render",
    }
    assert "highlightUrl" not in document and "upstreamUrl" not in document and "publicUrl" not in document


def test_sec_capture_rejects_archived_bytes_changed_after_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture cites archived SEC bytes only after their hash verifies; edited bytes create no source."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    payload = b"<html><body>quoted</body></html>"
    record = sec_archive.archive_sec_document(
        ACC, DOC, payload, url="https://www.sec.gov/x", metadata={}, root=tmp_path / "raw"
    )
    result = {
        "text": "quoted",
        "accession_no": ACC,
        "document_name": DOC,
        "source_content_hash": record.sha256,
        "offset": 0,
        "end_offset": 6,
    }
    with capture_scope() as accepted:
        capture_result("get_sec_document", result)
    assert len(accepted) == 1
    record.payload_path.write_bytes(payload.replace(b"quoted", b"forged"))
    with capture_scope() as sources, pytest.raises(ValueError):
        capture_result("get_sec_document", result)
    assert sources == []


@pytest.mark.parametrize(("section", "query", "cursor"), [("Item 2", None, 0), ("Item 2", "Revenue", 6)])
def test_sec_section_capture_stores_full_document_coordinates(
    section: str, query: str | None, cursor: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A section or query window cites full-document offsets into the SEC matching basis, never view offsets."""
    from app.sec.documents import get_sec_document

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    body = (
        "<html><head><script>hidden script</script><style>hidden style</style></head><body>"
        "<!--hidden comment--><ix:hidden><x:context>hidden context</x:context></ix:hidden>"
        "<p>Cover page</p><p>Item 1. Business</p><p>We sell widgets.</p><p>Item 2. Properties</p><p>"
        + "intro " * 450
        + "</p><table><tr><td>Revenue</td><td>5</td></tr></table><p>We lease offices.</p></body></html>"
    )
    sec_archive.archive_sec_document(
        ACC, DOC, body.encode(), url="https://www.sec.gov/s", metadata={}, root=tmp_path / "raw"
    )
    result = get_sec_document(ACC, DOC, data_root=tmp_path, section=section, query=query, cursor=cursor, limit=20)
    unchanged = dict(result)
    view_offset = result["offset"]
    with capture_scope(str(tmp_path)) as sources:
        capture_result("get_sec_document", result)
    assert result == unchanged
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    location = document["location"]
    basis = document["matchingText"]
    assert isinstance(location, dict) and isinstance(basis, str) and isinstance(view_offset, int)
    start, end = location["offset"], location["end"]
    assert isinstance(start, int) and isinstance(end, int)
    assert start > view_offset
    assert basis[start:end] == result["text"]
    assert "Revenue | 5" in basis
    assert "hidden" not in basis
    if query is not None:
        assert start > basis.index("Item 2. Properties") + cursor
    assert (location["prefix"], location["suffix"]) == (basis[max(0, start - 120) : start], basis[end : end + 120])
    assert (document["representation"], document["content"]) == ("html", body)


_HTML = b"<html><body>quoted</body></html>"
_HTML_HASH = raw_archive.content_hash(_HTML)


@pytest.mark.parametrize(
    "change",
    [
        {"source_refs": None},
        {"source_refs": []},
        {"source_refs": ["not-a-mapping"]},
        {"source_refs": [{"accession": ACC, "document": "other.htm", "offset": 0}]},
        {"source_refs": [{"accession": "0000000000-26-000043", "document": DOC, "offset": 0}]},
        {"source_refs": [{"accession": ACC, "document": DOC, "offset": True}]},
        {"source_refs": [{"accession": ACC, "document": DOC, "offset": 0, "basis": "raw"}]},
        {"view": "raw"},
        {"source_handle": None},
        {
            "source_handle": {
                "accession_no": ACC,
                "document_name": DOC,
                "basis": "view",
                "source_content_hash": _HTML_HASH,
            }
        },
        {
            "source_handle": {
                "accession_no": ACC,
                "document_name": DOC,
                "basis": "rendered",
                "source_content_hash": "b" * 64,
            }
        },
        {
            "source_handle": {
                "accession_no": ACC,
                "document_name": "x.htm",
                "basis": "rendered",
                "source_content_hash": _HTML_HASH,
            }
        },
        {"offset": 1, "end_offset": 7},
        {"end_offset": 5},
        {"offset": None},
    ],
)
def test_sec_capture_keeps_unverified_reference_unselected(
    change: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference that disagrees with the result identity, basis, or window stores no coordinates."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    sec_archive.archive_sec_document(ACC, DOC, _HTML, url="https://www.sec.gov/x", metadata={}, root=tmp_path / "raw")
    with capture_scope() as sources:
        capture_result("get_sec_document", {**_sec_result("quoted", _HTML_HASH, 0), **change})
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["location"] == {"basis": "rendered"}
    assert document["matchingText"] == "quoted"


def test_sec_reader_rejects_valid_hex_digest_for_other_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed digest that fails the manifest check rejects before payload bytes matter."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    digest = sec_archive.archive_sec_document(
        ACC, DOC, b"quoted", url="https://www.sec.gov/x", metadata={}, root=tmp_path / "raw"
    ).sha256
    record = sec_archive.find_archived_document(ACC, DOC, sha256=digest, root=tmp_path / "raw")
    assert record is not None
    wrong = "0" * 63 + ("1" if digest[-1] != "1" else "2")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        _sources._check_digest(record, wrong)


def test_sec_scope_archives_retained_bytes_as_original_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted live SEC bytes become the original: archive URL and time, retained metadata, raw-basis context."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    body = "<html><body><p>before quoted passage after</p></body></html>"
    digest = raw_archive.content_hash(body.encode())
    start = body.index("quoted passage")
    end = start + len("quoted passage")
    with capture_scope() as sources:
        _sources.retain_sec_bytes(ACC, DOC, body.encode(), {"source_content_hash": digest})
        capture_result(
            "get_sec_document",
            _sec_result(
                "quoted passage",
                digest,
                start,
                basis="raw",
                source_url="https://www.sec.gov/live",
                url="https://www.sec.gov/other",
                filed_at="2026-01-03",
            ),
        )
    record = sec_archive.find_archived_document(ACC, DOC, sha256=digest, root=tmp_path / "raw")
    assert record is not None
    assert record.url == "https://www.sec.gov/live"
    assert record.metadata.get("source_content_hash") == digest
    assert sources == [{"id": sources[0]["id"], "title": DOC, "url": "https://www.sec.gov/live"}]
    assert read_source_document(sources[0]["id"], data_root=tmp_path) == {
        "id": sources[0]["id"],
        "title": DOC,
        "url": "https://www.sec.gov/live",
        "originalUrl": "https://www.sec.gov/live",
        "linkStatus": "stockbot_render",
        "publisher": "SEC",
        "publishedAt": "2026-01-03",
        "capturedAt": record.retrieved_at,
        "quote": "quoted passage",
        "location": {"offset": start, "end": end, "basis": "raw", "prefix": body[:start], "suffix": body[end:]},
        "contentHash": digest,
        "captureKind": "original",
        "representation": "text",
        "content": body,
    }


def test_sec_evidence_reads_only_its_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two immutable revisions of one document: each evidence row reads exactly the bytes it pinned."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    bodies = {"ev-a": "alpha quoted passage", "ev-b": "omega quoted passage"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # A second revision warns by design.
        digests = {
            evidence_id: sec_archive.archive_sec_document(
                ACC,
                DOC,
                body.encode(),
                url="https://www.sec.gov/r",
                metadata={"evidence_id": evidence_id},
                root=tmp_path / "raw",
            ).sha256
            for evidence_id, body in bodies.items()
        }
    # An unpinned lookup returns the lowest-sorted revision, so pin the highest one.
    pinned = max(digests.values())
    no_metadata = _sec_evidence("ev-b", "", "quoted passage", 6, "rendered")
    del no_metadata["metadata"]
    repo = _Repo(
        {
            "ev-pin": _sec_evidence("ev-pin", pinned, "quoted passage", 6, "rendered"),
            "ev-a": _sec_evidence("ev-a", "", "quoted passage", 6, "rendered"),
            "ev-b": no_metadata,
        }
    )
    content = {digest: bodies[evidence_id] for evidence_id, digest in digests.items()}
    for evidence_id, digest in {"ev-pin": pinned, **digests}.items():
        document = read_source_document(f"evidence:{evidence_id}", data_root=tmp_path, repo=repo)
        assert (document["contentHash"], document["content"], document["title"]) == (digest, content[digest], DOC)


def test_sec_evidence_with_malformed_identity_is_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    bad_accession = _sec_evidence("ev-acc", "a" * 64, "quoted", 0, "rendered")
    provenance = bad_accession["provenance"]
    assert isinstance(provenance, dict)
    provenance["accession_no"] = "../x"
    repo = _Repo({"ev-acc": bad_accession, "ev-hash": _sec_evidence("ev-hash", "not-a-hash", "quoted", 0, "rendered")})
    for evidence_id in ("ev-acc", "ev-hash"):
        with pytest.raises(ValueError):
            read_source_document(f"evidence:{evidence_id}", data_root=tmp_path, repo=repo)


@pytest.mark.parametrize(
    ("body", "passage", "offset", "end", "representation", "context"),
    [
        (b"<HTML><BODY>quoted passage</BODY></HTML>", "quoted passage", 0, 14, "html", ("", "")),
        (b"before  quoted passage\n\n  after", "quoted passage", 8, 22, "text", ("before  ", "\n\n  after")),
        (b"caf\xe9 quoted passage", "quoted passage", 5, 19, "text", ("caf\ufffd ", "")),
        (b"quoted passage", "quoted passage", 0, 14, "text", ("", "")),
        (b"before quoted passage after", "quoted passage", 0, 14, "text", None),
        (b"quoted passage", "quoted passage", None, 14, "text", None),
        (b"quoted passage", "", 3, 3, "text", None),
    ],
)
def test_sec_original_reads_verbatim_with_context_only_at_exact_offsets(
    body: bytes,
    passage: str,
    offset: int | None,
    end: int,
    representation: str,
    context: tuple[str, str] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any tag case or byte encoding reads as the original; context needs offsets that address the quote."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    digest = sec_archive.archive_sec_document(
        ACC, DOC, body, url="https://www.sec.gov/c", metadata={}, root=tmp_path / "raw"
    ).sha256
    row = _sec_evidence("ev-ctx", digest, passage, 0, "rendered")
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance.update(offset=offset, end=end)
    document = read_source_document("evidence:ev-ctx", data_root=tmp_path, repo=_Repo({"ev-ctx": row}))
    location: dict[str, object] = {"offset": offset, "end": end, "basis": "rendered"}
    if context is not None:
        location.update(prefix=context[0], suffix=context[1])
    assert document["representation"] == representation
    assert document["content"] == body.decode("utf-8", "replace")
    assert document["location"] == location


def test_evidence_reader_uses_requested_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research.repository import ResearchRepository

    data_root = tmp_path / "data"
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    digest = sec_archive.archive_sec_document(
        ACC, DOC, b"plain quoted passage", url="https://www.sec.gov/d", metadata={}, root=data_root / "raw"
    ).sha256
    ResearchRepository(data_root=data_root).save_evidence(_sec_evidence("ev-root", digest, "quoted passage", 6, "text"))
    assert read_source_document("evidence:ev-root", data_root=data_root)["contentHash"] == digest


_OFFICIAL_HTML = (
    b"<html><body><p>Intro</p><table><tr><td>ACME</td><td>2026-01-02</td><td>5</td></tr></table></body></html>"
)


@pytest.mark.parametrize(
    ("tool", "kind", "publisher", "url"),
    [
        ("query_finra", "finra_record", "FINRA", "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20260102.csv"),
        ("get_fundamentals", "sec_record", "SEC", "https://www.sec.gov/Archives/edgar/data/42/x.htm"),
    ],
)
def test_structured_capture_matches_persisted_record_evidence(
    tool: str, kind: str, publisher: str, url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-shot record capture and its persisted evidence replay render the same official page."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    calls = _official(
        monkeypatch, _page(url, _OFFICIAL_HTML, "html", "ACME | 2026-01-02 | 5"), finra=kind == "finra_record"
    )
    result = {"symbol": "ACME", "text": "row summary", "rows": [{"name": "Société", "value": 5}], "as_of": "2026-01-02"}
    with capture_scope() as sources:
        capture_result(tool, {**result, "error": "upstream"})
        capture_result(tool, result)
    assert sources == [{"id": sources[0]["id"], "title": tool}]
    captured = read_source_document(sources[0]["id"], data_root=tmp_path)

    def evidence(evidence_id: str, tool_result_id: str) -> dict[str, object]:
        return {
            "evidence_id": evidence_id,
            "session_id": "s",
            "source_name": publisher,
            "source_uri": "https://example.com/record",
            "published_at": "2026-01-01",
            "retrieved_at": "2026-01-02",
            "provenance": {
                "kind": kind,
                "title": "Record",
                "tool_name": tool,
                "tool_result_id": tool_result_id,
                "record_identity": "ACME 2026-01-02",
            },
        }

    repo = _Repo(
        {"ev-rec": evidence("ev-rec", "tr-rec"), "ev-swap": evidence("ev-swap", "tr-swap")},
        {
            "tr-rec": {"tool_name": tool, "session_id": "s", "result": result},
            "tr-swap": {"tool_name": "search_web", "session_id": "s", "result": result},
        },
    )
    persisted = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    # Neither cites a JSON row, so the resolver reads both pages from the same whole payload.
    name = "" if kind == "finra_record" else tool
    assert calls == [(name, {}, result), (name, {}, result)]
    keys = ("url", "content", "contentHash", "captureKind", "representation", "quote", "location", "linkStatus")
    assert {key: persisted[key] for key in keys} == {key: captured[key] for key in keys}
    assert {key: captured[key] for key in ("url", "captureKind", "representation", "quote", "originalUrl")} == {
        "url": url,
        "captureKind": "original",
        "representation": "html",
        "quote": "ACME | 2026-01-02 | 5",
        "originalUrl": url,
    }
    assert captured["contentHash"] == raw_archive.content_hash(_OFFICIAL_HTML)
    assert captured["highlightUrl"] == f"{url}#:~:text=ACME,5"
    assert (captured["title"], captured["publisher"]) == (tool, publisher)
    assert {key: persisted[key] for key in ("title", "publisher", "publishedAt", "capturedAt")} == {
        "title": "Record",
        "publisher": publisher,
        "publishedAt": "2026-01-01",
        "capturedAt": "2026-01-02",
    }
    with pytest.raises(ValueError):
        read_source_document("evidence:ev-swap", data_root=tmp_path, repo=repo)


def test_web_evidence_reads_pinned_provider_content(tmp_path: Path) -> None:
    """Persisted provider content wins over a replayed tool result that drifted."""
    content = "<p>pinned quoted excerpt body</p>"
    provenance = {
        "kind": "web_source",
        "url": "https://example.com/a",
        "excerpt": "quoted excerpt",
        "title": "T",
        "domain": "example.com",
        "published_at": "2026-01-01",
        "retrieved_at": "2026-01-02",
        "tool_result_id": "tr-1",
        "source_content": content,
        "source_representation": "html",
    }
    drifted = {
        "query": "echoed query",
        "evidence": [{"url": "https://example.com/a", "highlight": "quoted excerpt", "source_content": "drift"}],
    }
    # Persisted call arguments win over the payload echo.
    arguments = {"query": "acme news", "num_results": 5}
    repo = _Repo(
        {"ev-pin": {"evidence_id": "ev-pin", "session_id": "s", "source_name": "Exa", "provenance": provenance}},
        {"tr-1": {"tool_name": "search_web", "session_id": "s", "result": drifted, "arguments": arguments}},
    )
    document = read_source_document("evidence:ev-pin", data_root=tmp_path, repo=repo)
    # A legacy row without a proof opens with an unverified audit; the viewer never upgrades it.
    audit = document.pop("audit")
    assert isinstance(audit, dict)
    assert {key: audit[key] for key in ("evidenceId", "sourceId", "claimType", "verification")} == {
        "evidenceId": "ev-pin",
        "sourceId": "evidence:ev-pin",
        "claimType": "observed_fact",
        "verification": "unverified",
    }
    assert document == {
        "id": "evidence:ev-pin",
        "title": "T",
        "url": "https://example.com/a",
        "publisher": "example.com",
        "publishedAt": "2026-01-01",
        "capturedAt": "2026-01-02",
        "representation": "html",
        "captureKind": "provider_extraction",
        "content": content,
        "contentHash": raw_archive.content_hash(content.encode()),
        "quote": "quoted excerpt",
        "toolName": "search_web",
        "toolResultId": "tr-1",
        "toolArgs": arguments,
        "originalUrl": "https://example.com/a",
        "linkStatus": "highlighted_original",
        "highlightUrl": "https://example.com/a#:~:text=quoted%20excerpt",
    }


@pytest.mark.parametrize(
    ("content", "quoted", "representation", "url", "fragment"),
    [
        ("Intro\nRevenue rose.\nEnd", "Revenue rose.", "text", "https://example.com/a", "#:~:text=Revenue%20rose."),
        (
            "<p>Revenue <strong>rose</strong>.</p>",
            "Revenue rose.",
            "html",
            "https://example.com/a",
            "#:~:text=Revenue%20rose.",
        ),
        (
            "<span>café &amp; costs, a-b</span>",
            "café & costs, a-b",
            "html",
            "https://example.com/a",
            "#:~:text=caf%C3%A9%20%26%20costs%2C%20a%2Db",
        ),
        (
            "<p>Revenue rose.</p>",
            "Revenue rose.",
            "text",
            "https://example.com/a#quarter%202",
            "#quarter%202:~:text=Revenue%20rose.",
        ),
        ("<p>Revenue rose.</p>", "<p>Revenue rose.</p>", "html", "https://example.com/a", None),
        ("<p>Revenue &amp; costs</p>", "Revenue &amp; costs", "html", "https://example.com/a", None),
        ('<p title="Revenue rose.">Other text</p>', "Revenue rose.", "html", "https://example.com/a", None),
        ("<script>Revenue rose.</script><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        ("<style>Revenue rose.</style><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        ("<head>Revenue rose.</head><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        ("<title>Revenue rose.</title><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        ("<template>Revenue rose.</template><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        ("<noscript>Revenue rose.</noscript><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        (
            "<template><template>Other</template>Revenue rose.</template><p>Other text</p>",
            "Revenue rose.",
            "html",
            "https://example.com/a",
            None,
        ),
        ("<p hidden>Revenue rose.</p><p>Other text</p>", "Revenue rose.", "html", "https://example.com/a", None),
        (
            '<p style="display:none">Revenue rose.</p><p>Other text</p>',
            "Revenue rose.",
            "html",
            "https://example.com/a",
            None,
        ),
        ("<p>Other text</p>", "Missing quote", "html", "https://example.com/a", None),
        ("Revenue", "", "text", "https://example.com/a", None),
        ("Revenue rose. Revenue rose.", "Revenue rose.", "text", "https://example.com/a", None),
        ("Revenue rose. REVENUE ROSE.", "Revenue rose.", "text", "https://example.com/a", None),
        ("<p>Item 1</p><p>Revenue</p><p>Item 2</p><p>Revenue</p>", "Revenue", "html", "https://example.com/a", None),
        ("Revenues rose.", "Revenue", "text", "https://example.com/a", None),
        ("Revenue rose.", "Revenue rose.", "text", "https://example.com/a.pdf", None),
        ("Revenue rose.", "Revenue rose.", "text", "https://example.com/a.PDF?download=1#page=4", None),
        ("Revenue rose.", "Revenue rose.", "text", "https://example.com/a%2Epdf#page=4", None),
        ("Revenue rose.", "Revenue rose.", "text", "https://example.com/a#:~:text=Other", None),
    ],
)
def test_web_original_highlight_uses_only_unique_visible_provider_text(
    content: str, quoted: str, representation: str, url: str, fragment: str | None, tmp_path: Path
) -> None:
    """A web highlight preserves the original anchor and exact capture without markup or invented PDF locations."""
    provenance = {
        "kind": "web_source",
        "url": url,
        "excerpt": quoted,
        "source_content": content,
        "source_representation": representation,
        "tool_result_id": "tr-web",
    }
    repo = _Repo(
        {"ev-web": {"evidence_id": "ev-web", "session_id": "s", "provenance": provenance}},
        {
            "tr-web": {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {"evidence": [{"url": url, "highlight": quoted}]},
            }
        },
    )
    document = read_source_document("evidence:ev-web", data_root=tmp_path, repo=repo)
    assert document["originalUrl"] == url
    assert document["content"] == content
    assert document["contentHash"] == raw_archive.content_hash(content.encode())
    assert document["quote"] == quoted
    assert document["representation"] == representation
    assert repo.get_evidence("ev-web")["provenance"] == provenance
    if fragment is None:
        assert document["linkStatus"] == "stockbot_render"
        assert "highlightUrl" not in document
    else:
        assert document["linkStatus"] == "highlighted_original"
        assert document["highlightUrl"] == url.partition("#")[0] + fragment


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/source", "https://user:secret@example.com/a"])
def test_web_unsafe_original_has_no_highlight(url: str, tmp_path: Path) -> None:
    provenance = {
        "kind": "web_source",
        "url": url,
        "excerpt": "Revenue rose.",
        "source_content": "Revenue rose.",
        "tool_result_id": "tr-web",
    }
    repo = _Repo(
        {"ev-web": {"evidence_id": "ev-web", "session_id": "s", "provenance": provenance}},
        {
            "tr-web": {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {"evidence": [{"url": url, "highlight": "Revenue rose."}]},
            }
        },
    )
    document = read_source_document("evidence:ev-web", data_root=tmp_path, repo=repo)
    assert document["linkStatus"] == "missing"
    assert not {"originalUrl", "highlightUrl"} & document.keys()


@seed(2119)
@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.sampled_from(["Revenue", "81,615", "a-b", "x&y", "café", "東京", "100%"]), min_size=1, max_size=8),
    st.sampled_from(["", "quarter%202", "caf%C3%A9"]),
)
def test_prop_web_highlights_encode_visible_text_without_changing_capture(words: list[str], anchor: str) -> None:
    """Readable HTML quotes produce encoded directives while captures retain their exact bytes and identity."""
    quoted = " ".join(words)
    content = "<p>" + " ".join(f"<strong>{escape(word)}</strong>" for word in words) + "</p>"
    url = "https://example.com/a" + ("#" + anchor if anchor else "")
    row = {"url": url, "highlight": quoted, "source_content": content, "source_representation": "html"}
    with tempfile.TemporaryDirectory() as tmp:
        with capture_scope(tmp) as first:
            capture_result("search_web", {"evidence": [row]})
        document = read_source_document(first[0]["id"], data_root=Path(tmp))
        with capture_scope(tmp) as repeated:
            capture_result("search_web", {"evidence": [row]})
    assert first == repeated
    assert document["content"] == content
    assert document["contentHash"] == raw_archive.content_hash(content.encode())
    assert document["originalUrl"] == url
    assert document["linkStatus"] == "highlighted_original"
    fragment = "#" + anchor + ":~:text=" + _sources._term(quoted)
    assert document["highlightUrl"] == "https://example.com/a" + fragment
    assert _sources._fragment_ranges(quoted, fragment) == [(0, len(quoted))]


def _record_evidence(kind: str, tool: str, row: Mapping[str, object], **extra: object) -> dict[str, object]:
    return {
        "evidence_id": "ev-rec",
        "session_id": "s",
        "source_name": "FINRA" if kind == "finra_record" else "SEC",
        "provenance": {
            "kind": kind,
            "tool_name": tool,
            "tool_result_id": "tr-rec",
            "record_identity": json.dumps(row, sort_keys=True),
            **extra,
        },
    }


def _record_repo(kind: str, tool: str, row: Mapping[str, object], payload: object, **extra: object) -> _Repo:
    return _Repo(
        {"ev-rec": _record_evidence(kind, tool, row, **extra)},
        {"tr-rec": {"tool_name": tool, "session_id": "s", "result": payload}},
    )


_SHVOL_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260102.txt"
_SHVOL = b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20260102|ACME|5|0|9|B,Q,N\n"


def test_finra_record_reports_how_it_was_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FINRA: the cited row's daily file is the original. Plain text never highlights (case 2)."""
    line = "20260102|ACME|5|0|9|B,Q,N"
    calls = _official(monkeypatch, _page(_SHVOL_URL, _SHVOL, "text", line), finra=True)
    row = {"securitiesInformationProcessorSymbolIdentifier": "ACME", "tradeReportDate": "2026-01-02"}
    payload = {
        "dataset_id": "otcMarket/regShoDaily",
        "query": {"ticker": "ACME", "start_date": "2026-01-01", "limit": 100},
        "records": [row],
    }
    repo = _record_repo(
        "finra_record", "get_reg_sho_volume", row, payload, dataset="otcMarket/regShoDaily", known_at="2026-01-02"
    )
    document = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert calls == [("otcMarket/regShoDaily", row, payload)]
    how = ("toolName", "toolResultId", "dataset", "knownAt", "toolArgs")
    assert {key: document[key] for key in how} == {
        "toolName": "get_reg_sho_volume",
        "toolResultId": "tr-rec",
        "dataset": "otcMarket/regShoDaily",
        "knownAt": "2026-01-02",
        "toolArgs": {"ticker": "ACME", "start_date": "2026-01-01", "limit": 100},
    }
    start = _SHVOL.decode().index(line)
    assert {key: document[key] for key in ("url", "captureKind", "representation", "content", "quote")} == {
        "url": _SHVOL_URL,
        "captureKind": "original",
        "representation": "text",
        "content": _SHVOL.decode(),
        "quote": line,
    }
    assert document["location"] == {
        "basis": "rendered",
        "offset": start,
        "end": start + len(line),
        "prefix": _SHVOL.decode()[:start],
        "suffix": "\n",
    }
    assert (document["originalUrl"], document["linkStatus"]) == (_SHVOL_URL, "stockbot_render")
    assert not {"highlightUrl", "upstreamUrl", "publicUrl", "matchingText"} & set(document)


_TEN_Q = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000042/nvda-20260426.htm"


@pytest.mark.parametrize(
    ("body", "quoted", "anchor", "fragment"),
    [
        pytest.param(
            "<table><tr><td>Revenue</td><td>81,615</td><td>60,922</td></tr>"
            "<tr><td>Cost of revenue</td><td>81,615</td><td>17,394</td></tr></table>",
            "Revenue | 81,615",
            "f-30",
            # "Revenue" also starts "revenue" in the next row, so the next cell becomes the suffix.
            "#f-30:~:text=Revenue,81%2C615,-60%2C922",
            id="table-row-cells-with-suffix",
        ),
        pytest.param(
            "<p>Item 1</p><p>Net income</p><p>Item 2</p><p>Net income</p>",
            "Net income",
            "",
            "#:~:text=Item%201-,Net%20income",
            id="prefix-block",
        ),
        pytest.param("<p>Total</p><p>Total</p><p>Total</p>", "Total", "", None, id="ambiguous"),
        pytest.param("<p>Revenues grew</p>", "Revenue", "", None, id="mid-word"),
        pytest.param("<p>Revenue</p>", "Absent quote", "", None, id="quote-not-found"),
    ],
)
def test_sec_record_link_status_follows_verified_directive(
    body: str, quoted: str, anchor: str, fragment: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case 1 only when the directive matches the quote exactly once in our rendering; else case 2."""
    page = f"<html><body>{body}</body></html>".encode()
    calls = _official(monkeypatch, _page(_TEN_Q, page, "html", quoted, anchor), finra=False)
    row = {"concept": "Revenues", "value": 81615, "accession": ACC}
    payload = {"cik": 1045810, "matching_concepts": [row]}
    document = read_source_document(
        "evidence:ev-rec", data_root=tmp_path, repo=_record_repo("sec_record", "get_xbrl_facts", row, payload)
    )
    assert calls == [("get_xbrl_facts", row, payload)]
    assert (document["captureKind"], document["representation"], document["content"]) == (
        "original",
        "html",
        page.decode(),
    )
    assert document["originalUrl"] == _TEN_Q
    if fragment is None:
        assert document["linkStatus"] == "stockbot_render"
        assert "highlightUrl" not in document
    else:
        assert document["linkStatus"] == "highlighted_original"
        assert document["highlightUrl"] == _TEN_Q + fragment
        matching = str(document["matchingText"])
        assert _sources._fragment_ranges(matching, fragment) == [
            (matching.index(quoted), matching.index(quoted) + len(quoted))
        ]


@pytest.mark.parametrize("element_id", ["f-hidden", None])
def test_sec_fact_case_two_replays_exact_official_bytes(
    element_id: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hidden and equivalent facts retain an official page without an invented highlighted passage."""
    from app.sec import documents
    from tests.test_sec_documents_archive import OFF_INDEX, OFF_PRIMARY, OFF_PRIMARY_HTML, _FakeSec, _revenue_row

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    network = _FakeSec()
    monkeypatch.setattr(documents, "_sec_get", network)
    row = _revenue_row(**({"element_id": element_id} if element_id is not None else {}))
    result = dict(row)
    with capture_scope() as captures:
        capture_result("get_xbrl_facts", result)
    assert network.calls == []
    source_id = captures[0]["id"]
    first = read_source_document(source_id, data_root=tmp_path)
    assert (first["linkStatus"], first["originalUrl"], first["quote"]) == ("stockbot_render", OFF_PRIMARY, "")
    assert "highlightUrl" not in first
    assert first["content"] == OFF_PRIMARY_HTML
    assert first["contentHash"] == raw_archive.content_hash(OFF_PRIMARY_HTML.encode())
    assert first["location"] == {"basis": "rendered"}
    assert network.calls == [OFF_INDEX, OFF_PRIMARY]
    network.pages.clear()
    second = read_source_document(source_id, data_root=tmp_path)
    assert second == first
    assert network.calls == [OFF_INDEX, OFF_PRIMARY]


@pytest.mark.parametrize("metadata", [False, True])
def test_sec_metadata_index_does_not_become_a_financial_fact_citation(
    metadata: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same filing index proves filing metadata but cannot prove an absent financial value."""
    from app.sec import documents
    from tests.test_sec_documents_archive import OFF_ACC, OFF_CIK, OFF_INDEX, OFF_PRIMARY, _FakeSec, _revenue_row

    network = _FakeSec()
    monkeypatch.setattr(documents, "_sec_get", network)
    row = (
        {"accession_no": OFF_ACC, "form": "10-Q", "filer_cik": OFF_CIK}
        if metadata
        else _revenue_row(value=81615000001.0)
    )
    tool = "list_sec_filings" if metadata else "get_xbrl_facts"
    payload = {"filings" if metadata else "matching_concepts": [row], "cik": OFF_CIK}
    repo = _record_repo("sec_record", tool, row, payload)
    document = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    if metadata:
        assert document["originalUrl"] == OFF_INDEX
        assert document["quote"] == f"SEC Accession No. {OFF_ACC}"
        assert document["captureKind"] == "original"
        assert network.calls == [OFF_INDEX]
    else:
        assert document["linkStatus"] == "missing"
        assert document["captureKind"] == "structured_record"
        assert document["quote"] == ""
        assert "originalUrl" not in document
        assert "highlightUrl" not in document
        assert network.calls == [OFF_INDEX, OFF_PRIMARY]


def test_sec_record_reports_how_it_was_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC: persisted arguments, else the subject echo. A PDF original never highlights (case 2)."""
    pdf = b"%PDF-1.4\nofficial filing\n%%EOF\n"
    url = f"https://www.sec.gov/Archives/edgar/data/42/000000000026000042/{ACC}.pdf"
    _official(monkeypatch, _page(url, pdf, "pdf", "official filing"), finra=False)
    row = {"accession_no": ACC, "form": "10-Q", "filer_cik": 42}
    tool_result: dict[str, object] = {
        "tool_name": "list_sec_filings",
        "session_id": "s",
        "result": {"subject": "ACME", "filings": [row]},
        "arguments": {"identifier": "ACME", "form": "10-Q"},
    }
    evidence = _record_evidence("sec_record", "list_sec_filings", row, source_uri="https://data.sec.gov/x.json")
    repo = _Repo({"ev-rec": evidence}, {"tr-rec": tool_result})
    document = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert document["toolArgs"] == {"identifier": "ACME", "form": "10-Q"}
    assert {key: document.get(key) for key in ("url", "representation", "pdfUrl", "originalUrl", "linkStatus")} == {
        "url": url,
        "representation": "pdf",
        "pdfUrl": "/api/research/sources/evidence%3Aev-rec/pdf",
        "originalUrl": url,
        "linkStatus": "stockbot_render",
    }
    assert read_source_pdf("evidence:ev-rec", data_root=tmp_path, repo=repo) == pdf
    del tool_result["arguments"]  # A row persisted before call arguments: the subject echo stands in.
    assert read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)["toolArgs"] == {"subject": "ACME"}


def test_record_without_official_page_logs_full_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Case 3: no official page. Only the Stockbot view remains, and an ERROR names every input."""
    calls = _official(monkeypatch, None, finra=False)
    row = {"source_accessions": [ACC], "issuer": "Acme"}
    payload = {"subject": "ACME", "events": [row]}
    tool_result = {"tool_name": "get_material_events", "session_id": "s", "result": payload, "arguments": {"x": 1}}
    repo = _Repo(
        {"ev-rec": _record_evidence("sec_record", "get_material_events", row, source_uri="https://data.sec.gov/a")},
        {"tr-rec": tool_result},
    )
    with caplog.at_level("ERROR", logger="app.research.sources"):
        document = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert calls == [("get_material_events", row, payload)]
    assert document["linkStatus"] == "missing"
    assert not {"originalUrl", "highlightUrl"} & set(document)
    assert (document["captureKind"], document["representation"], document["quote"]) == ("structured_record", "text", "")
    assert json.loads(str(document["content"])) == row
    with pytest.raises(FileNotFoundError):
        read_source_pdf("evidence:ev-rec", data_root=tmp_path, repo=repo)
    [record] = [r for r in caplog.records if r.levelname == "ERROR"]
    assert record.getMessage() == (
        "source official page missing: source_id=evidence:ev-rec evidence_id=ev-rec kind=sec_record "
        f'tool=get_material_events tool_args={{"x": 1}} row={json.dumps(row, sort_keys=True)} '
        "reason=resolver found no official page for the cited record failure=PUBLIC_PAGE_UNAVAILABLE"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"url": "https://data.sec.gov/submissions/CIK0000000042.json"},
        {"url": "https://efts.sec.gov/LATEST/search-index?q=x"},
        {"url": "https://api.finra.org/data/group/otcMarket/name/regShoDaily"},
        {"url": "http://www.sec.gov/x.htm"},
        {"url": "https://www.sec.gov/x.htm#:~:text=x"},
        {"url": ""},
        {"sha256": "0" * 64},
        {"media": "json"},
        {"payload": "not bytes"},
        {"quote": None},
    ],
)
def test_resolver_page_outside_contract_is_rejected(
    change: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No API host, plain-http, fragment, or unhashed page ever becomes an original link."""
    _official(monkeypatch, {**_page(_TEN_Q, b"<p>x</p>", "html", "x"), **change}, finra=False)
    repo = _record_repo("sec_record", "get_xbrl_facts", {"concept": "x"}, {"cik": 42})
    with pytest.raises(ValueError, match="Official source page is invalid"):
        read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)


def test_finra_official_page_is_fetched_once_on_first_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first view fetches the official file and archives it. Later views read the archive only."""
    from unittest.mock import MagicMock

    url = "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt"
    body = b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20261015|NVDA|274457.043010|1022|934921.209763|B\n2\n"
    get = MagicMock(return_value=MagicMock(content=body))
    monkeypatch.setattr("app.finra_client.requests.get", get)
    row = {"securitiesInformationProcessorSymbolIdentifier": "NVDA", "tradeReportDate": "2026-10-15", "marketCode": "B"}
    repo = _record_repo("finra_record", "get_reg_sho_volume", row, {}, dataset="otcMarket/regShoDaily")
    first = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    second = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert get.call_count == 1
    assert get.call_args.args[0] == url
    assert first == second
    assert (first["originalUrl"], first["linkStatus"], first["content"]) == (url, "stockbot_render", body.decode())
    assert first["quote"] == "20261015|NVDA|274457.043010|1022|934921.209763|B"
    assert first["contentHash"] == raw_archive.content_hash(body)


@pytest.mark.parametrize(
    "rows",
    [
        [{"url": "https://example.com/other", "highlight": "the quoted excerpt here"}],
        [{"url": "https://example.com/a", "highlight": "an unrelated passage"}],
        None,
    ],
)
def test_web_evidence_replay_requires_matching_url_and_excerpt(rows: object, tmp_path: Path) -> None:
    provenance = {
        "kind": "web_source",
        "url": "https://example.com/a",
        "excerpt": "quoted excerpt",
        "tool_result_id": "tr",
    }
    repo = _Repo(
        {"ev-web": {"evidence_id": "ev-web", "session_id": "s", "provenance": provenance}},
        {"tr": {"tool_name": "search_web", "session_id": "s", "result": {"evidence": rows}}},
    )
    with pytest.raises(FileNotFoundError):
        read_source_document("evidence:ev-web", data_root=tmp_path, repo=repo)


def test_source_references_accept_only_valid_replayable_rows() -> None:
    web = {"kind": "web_source"}
    assert source_references({"evidence_id": "a/b", "provenance": web}) == []
    assert source_references({"evidence_id": "ev", "provenance": "web_source"}) == []
    assert source_references(
        {"evidence_id": "ev", "source_name": "Exa", "source_uri": "https://example.com/s", "provenance": web}
    ) == [{"id": "evidence:ev", "title": "Exa", "url": "https://example.com/s"}]
    assert source_references(
        {"evidence_id": "ev", "source_name": "Exa", "source_uri": "https://:pw@example.com/s", "provenance": web}
    ) == [{"id": "evidence:ev", "title": "Exa"}]
    assert source_references(
        {
            "evidence_id": "ev",
            "source_uri": "javascript:alert(1)",
            "provenance": {**web, "title": "T", "document_name": "D"},
        }
    ) == [{"id": "evidence:ev", "title": "T"}]
    assert source_references({"evidence_id": "ev", "provenance": web}) == [{"id": "evidence:ev"}]


# --- property invariants: immutable capture identity + verified reads ---
# Hypothesis rejects function-scoped tmp_path, so each example owns a fresh TemporaryDirectory.
# deadline=None: per-example filesystem I/O timing must not make these deterministic properties flaky.


@settings(max_examples=100, derandomize=True, deadline=None)
@given(st.text(min_size=1, max_size=200), st.text(min_size=1, max_size=40), st.sampled_from(["html", "text"]))
def test_prop_web_capture_identity_is_stable_and_reads_verbatim(content: str, quote: str, representation: str) -> None:
    """Any accepted web row keeps one identity per exact capture and reads back exactly what was captured."""
    row = {
        "url": "https://example.com/a",
        "highlight": quote,
        "source_content": content,
        "source_representation": representation,
    }
    with tempfile.TemporaryDirectory() as tmp:
        with capture_scope(tmp) as first:
            capture_result("search_web", {"evidence": [row, {**row, "source_content": content + "!"}]})
        with capture_scope(tmp) as again:
            capture_result("search_web", {"evidence": [row]})
        source_id = first[0]["id"]
        assert again[0]["id"] == source_id
        assert first[1]["id"] != source_id
        document = read_source_document(source_id, data_root=Path(tmp))
    assert document["id"] == source_id
    assert document["content"] == content
    assert document["contentHash"] == raw_archive.content_hash(content.encode())
    assert document["quote"] == quote
    assert document["representation"] == representation
    assert document["captureKind"] == "provider_extraction"


@settings(max_examples=100, derandomize=True, deadline=None)
@given(st.text(min_size=1, max_size=200), st.text(min_size=1, max_size=40))
def test_prop_reader_rejects_capture_bytes_changed_after_archive(content: str, quote: str) -> None:
    """A same-length, still-valid capture edited in place fails the hash check instead of reading as the original."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with capture_scope(tmp) as sources:
            capture_result(
                "search_web",
                {"evidence": [{"url": "https://example.com/a", "highlight": quote, "source_content": content}]},
            )
        source_id = sources[0]["id"]
        read_source_document(source_id, data_root=root)
        digest = source_id.removeprefix("capture:")
        record = raw_archive.find("viewer", "capture", digest, sha256=digest, root=root / "raw")
        assert record is not None
        payload = record.payload_path.read_bytes()
        at = payload.index(b'"contentHash": "') + len(b'"contentHash": "')
        swapped = b"1" if payload[at : at + 1] == b"0" else b"0"
        record.payload_path.write_bytes(payload[:at] + swapped + payload[at + 1 :])
        with pytest.raises(ValueError):
            read_source_document(source_id, data_root=root)


def test_sec_capture_uses_exact_full_document_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cursor movement keeps only the addressed window. The quote cannot include earlier bytes."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    sec_archive.archive_sec_document(ACC, DOC, _HTML, url="https://www.sec.gov/x", metadata={}, root=tmp_path / "raw")
    result = _sec_result("quoted", _HTML_HASH, 4, offset=2, end_offset=8)
    with capture_scope() as sources:
        capture_result("get_sec_document", result)
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["location"] == {"offset": 4, "end": 10, "basis": "rendered"}


def test_reader_rejects_store_missing_tool_result_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both store reads are required: a double with only get_evidence falls back to the configured database."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    digest = sec_archive.archive_sec_document(
        ACC, DOC, b"plain quoted passage", url="https://www.sec.gov/d", metadata={}, root=tmp_path / "raw"
    ).sha256
    record = _sec_evidence("ev-half", digest, "quoted passage", 6, "text")

    class _OnlyEvidence:
        def get_evidence(self, evidence_id: str) -> dict[str, object]:
            assert evidence_id == "ev-half"
            return record

    with pytest.raises(KeyError):
        read_source_document("evidence:ev-half", data_root=tmp_path, repo=_OnlyEvidence())


_CELLS = st.sampled_from(["Revenue", "revenue", "81,615", "Net income", "a-b", "x&y", "Item 1", "Total", "5"])


@settings(max_examples=200, derandomize=True, deadline=None)
@given(st.lists(st.lists(_CELLS, min_size=1, max_size=3), min_size=1, max_size=6), st.data())
def test_prop_text_directive_matches_exactly_once(lines: list[list[str]], data: st.DataObject) -> None:
    """Any directive the builder returns matches exactly the quoted blocks once, with one block per term."""
    rendered = "\n".join(" | ".join(cells) for cells in lines)
    spans = [(match.start(), match.end()) for match in re.finditer(r"[^\n|]+", rendered)]
    spans = [(start + (rendered[start] == " "), end - (rendered[end - 1] == " ")) for start, end in spans]
    first = data.draw(st.integers(0, len(spans) - 1))
    last = data.draw(st.integers(first, len(spans) - 1))
    start, end = spans[first][0], spans[last][1]
    fragment = _sources._text_fragment(rendered, start, end, data.draw(st.sampled_from(["", "f-30"])))
    if fragment is None:
        return
    assert _sources._fragment_ranges(rendered, fragment) == [(start, end)]
    terms = fragment.partition(":~:text=")[2].split(",")
    assert len(terms) <= 4
    assert all("\n" not in unquote(term) and " | " not in unquote(term) for term in terms)


# ---------------------------------------------------------------------------
# Change 4 regression: FINRA multi-record captures, web snippet/article, absence
# ---------------------------------------------------------------------------
_REG4_SHVOL2_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260102.txt"
_REG4_SHVOL2 = (
    b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    b"20260102|ACME|5|0|9|B,Q,N\n"
    b"20260102|ACME|7|1|11|Q\n"
    b"20260102|BETA|3|0|4|B,Q,N\n"
)


def test_reg4_finra_aggregated_leaderboard_cites_contributing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-record leaderboard capture cites every contributing record, never one first row."""
    from app.research import service as _svc

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    rows = [
        {"symbolCode": "ACME", "settlementDate": "2026-09-30", "currentShortPositionQuantity": 10},
        {"symbolCode": "BETA", "settlementDate": "2026-09-30", "currentShortPositionQuantity": 20},
    ]
    texts = _svc._finra_record_texts({"records": rows})
    assert len(texts) == 2 and all("currentShortPositionQuantity" in text for text in texts)
    assert '"ACME"' in texts[0] and '"BETA"' in texts[1]


def test_reg4_finra_record_without_venue_cites_consolidated_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A venue-less trimmed envelope cites the consolidated CNMS line with an empty column check."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    line = "20260102|ACME|5|0|9|B,Q,N"
    calls = _official(monkeypatch, _page(_REG4_SHVOL2_URL, _REG4_SHVOL2, "text", line), finra=True)
    row: dict[str, object] = {"ticker": "ACME"}
    payload = {"dataset_id": "otcMarket/regShoDaily", "as_of_date": "2026-01-02", "query": row}
    repo = _record_repo("finra_record", "query_finra", row, payload, dataset="otcMarket/regShoDaily")
    document = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert calls == [("otcMarket/regShoDaily", row, payload)]
    assert document["quote"] == line and document["captureKind"] == "original"
    assert document["linkStatus"] == "stockbot_render"  # plain text never highlights


def test_reg4_finra_duplicate_rows_replay_second_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay against a two-row payload finds the cited second row, not the first."""
    from app.research import service as _svc
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    repo = ResearchRepository()
    sid = _svc.create_research("q?", "o", as_of="2026-06-30T00:00:00+00:00", repo=repo)
    src = repo.list_jobs(sid)[0].job_id
    repo.save_job(__import__("dataclasses").replace(repo.get_job(src), source_domain="FINRA", status="running"))
    _svc.persist_tool_result(
        sid,
        src,
        "query_finra",
        "tr-dup",
        {
            "records": [{"symbolCode": "ACME", "v": 1}, {"symbolCode": "ACME", "v": 2}],
            "dataset_id": "otcMarket/regShoDaily",
        },
        repo=repo,
    )
    prov = _svc._finra_provenance(
        {
            "tool_result_id": "tr-dup",
            "matching_passage": json.dumps({"symbolCode": "ACME", "v": 2}, sort_keys=True),
        },
        store=repo,
        session_id=sid,
    )
    assert str(prov["record_identity"]) == json.dumps({"symbolCode": "ACME", "v": 2}, sort_keys=True)


def test_reg4_web_snippet_vs_full_article_distinguished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider extraction and a bare excerpt are different kinds with different content."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    with capture_scope() as sources:
        capture_result("search_web", {"evidence": [{"url": "https://example.com/a", "highlight": "only"}]})
        capture_result(
            "search_web",
            {
                "evidence": [
                    {
                        "url": "https://example.com/a",
                        "highlight": "quoted passage",
                        "source_content": "longer article quoted passage tail",
                        "source_representation": "text",
                    }
                ]
            },
        )
    excerpt = read_source_document(sources[0]["id"], data_root=tmp_path)
    article = read_source_document(sources[1]["id"], data_root=tmp_path)
    assert (excerpt["captureKind"], excerpt["content"]) == ("search_excerpt", "only")
    assert article["captureKind"] == "provider_extraction"
    assert article["content"] == "longer article quoted passage tail"
    assert excerpt["contentHash"] != article["contentHash"]


def test_reg4_web_evidence_replay_requires_matching_url_and_excerpt(tmp_path: Path) -> None:
    """A replayed web row must match both URL and excerpt; drift raises instead of citing."""
    provenance = {
        "kind": "web_source",
        "url": "https://example.com/a",
        "excerpt": "quoted excerpt",
        "tool_result_id": "tr-1",
    }
    repo = _Repo(
        {"ev-w": {"evidence_id": "ev-w", "session_id": "s", "source_name": "Exa", "provenance": provenance}},
        {
            "tr-1": {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {"evidence": [{"url": "https://example.com/a", "highlight": "drifted text"}]},
            }
        },
    )
    import pytest as _pytest

    with _pytest.raises(FileNotFoundError, match="[Ee]xcerpt is missing"):
        read_source_document("evidence:ev-w", data_root=tmp_path, repo=repo)


def test_reg4_absence_claim_keeps_query_scope_and_pagination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An absence artifact pins query + scope + pagination; incomplete paging stays visible."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    from app.research import service as _svc
    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    sid = _svc.create_research("q?", "o", as_of="2026-06-30T00:00:00+00:00", repo=repo)
    src = repo.list_jobs(sid)[0].job_id
    out = _svc.record_evidence(
        sid,
        src,
        {
            "wave_id": 1,
            "content": "no NVDA threshold listing",
            "claim_text": "no listing in scope",
            "claim_kind": "absence_observation",
            "subject": "NVDA",
            "source_name": "FINRA",
            "known_at": "2026-06-29T00:00:00+00:00",
            "search_id": "s1",
            "query": "NVDA thresholdList 2026-10",
            "coverage": {
                "forms": [],
                "dates": ["2026-10-01"],
                "partitions": ["tradeDate=2026-10-01"],
                "entities": ["NVDA"],
                "docs": [],
                "gaps": [],
                "pagination_complete": False,
                "complete": False,
            },
        },
        repo=repo,
    )
    assert out["recorded_as"] == "coverage_artifact" and out["citable"] is False
    stored = repo.list_coverage_artifacts(sid)[0]
    assert isinstance(stored, dict)
    coverage = stored["coverage"]
    assert isinstance(coverage, dict) and coverage["pagination_complete"] is False
    assert stored["query"] == "NVDA thresholdList 2026-10"
    assert repo.list_evidence(sid) == []


def test_regweb_capture_pins_exact_provider_bytes_with_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Web capture archives the exact provider excerpt/source bytes first, with a content hash."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    row = {
        "title": "NVDA demand",
        "url": "https://example.com/nvda",
        "source_domain": "example.com",
        "published_at": "2025-06-20",
        "retrieved_at": "2025-06-21",
        "highlight": "NVDA demand rises",
        "source_content": "NVDA demand rises across the quarter.",
        "source_representation": "text",
    }
    with capture_scope() as sources:
        capture_result("search_web", {"evidence": [row]})
    assert len(sources) == 1
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["captureKind"] == "provider_extraction"
    assert document["quote"] == "NVDA demand rises"
    assert document["content"] == "NVDA demand rises across the quarter."
    assert document["contentHash"] == raw_archive.content_hash(document["content"].encode())
    assert document["linkStatus"] == "highlighted_original"
    assert document["highlightUrl"] == "https://example.com/nvda#:~:text=NVDA%20demand%20rises"


def test_regweb_pinned_content_never_substitutes_today_text(tmp_path: Path) -> None:
    """A pinned provider extraction replays only while the replay still holds the excerpt."""
    content = "<p>NVDA demand rises across the quarter.</p>"
    provenance = {
        "kind": "web_source",
        "url": "https://example.com/nvda",
        "excerpt": "NVDA demand rises",
        "title": "NVDA demand",
        "domain": "example.com",
        "published_at": "2025-06-20",
        "retrieved_at": "2025-06-21",
        "tool_result_id": "tr-pin",
        "source_content": content,
        "source_representation": "html",
    }
    repo = _Repo(
        {"ev-pin": {"evidence_id": "ev-pin", "session_id": "s", "source_name": "Exa", "provenance": provenance}},
        {
            "tr-pin": {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {
                    "query": "NVDA demand",
                    "evidence": [{"url": "https://example.com/nvda", "highlight": "NVDA demand rises"}],
                },
            }
        },
    )
    document = read_source_document("evidence:ev-pin", data_root=tmp_path, repo=repo)
    assert document["content"] == content
    assert document["quote"] == "NVDA demand rises"
    drifted = _Repo(
        {"ev-pin": {"evidence_id": "ev-pin", "session_id": "s", "source_name": "Exa", "provenance": provenance}},
        {
            "tr-pin": {
                "tool_name": "search_web",
                "session_id": "s",
                "result": {
                    "query": "NVDA demand",
                    "evidence": [{"url": "https://example.com/nvda", "highlight": "rewritten today"}],
                },
            }
        },
    )
    with pytest.raises(FileNotFoundError, match="excerpt"):
        read_source_document("evidence:ev-pin", data_root=tmp_path, repo=drifted)


# ---------------------------------------------------------------------------
# Change 5: lazy citation labels, controlled viewer errors, failure classes, audit view
# ---------------------------------------------------------------------------

_FNQC_URL = "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt"
_FNQC = b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20261015|NVDA|274457.043010|1022|934921.209763|B\n2\n"
_FNQC_ROW = {
    "securitiesInformationProcessorSymbolIdentifier": "NVDA",
    "tradeReportDate": "2026-10-15",
    "marketCode": "B",
}


class _CountingRepo(_Repo):
    """_Repo that records every tool-result replay."""

    def __init__(
        self, evidence: Mapping[str, Mapping[str, object]], results: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(evidence, results)
        self.replays: list[str] = []

    @override
    def get_tool_result(self, tool_result_id: str) -> object:
        self.replays.append(tool_result_id)
        return super().get_tool_result(tool_result_id)


def test_label_reads_persisted_metadata_and_only_the_open_viewer_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A citation label never replays the tool result, calls the official resolver, archives, or logs."""
    line = "20260102|ACME|5|0|9|B,Q,N"
    calls = _official(monkeypatch, _page(_SHVOL_URL, _SHVOL, "text", line), finra=True)
    row = {"securitiesInformationProcessorSymbolIdentifier": "ACME", "tradeReportDate": "2026-01-02"}
    payload = {"dataset_id": "otcMarket/regShoDaily", "records": [row]}
    evidence = {
        **_record_evidence("finra_record", "get_reg_sho_daily", row, dataset="otcMarket/regShoDaily"),
        "published_at": "2026-01-02",
        "source_uri": "https://api.finra.org/data/group/otcMarket/name/regShoDaily",
    }
    repo = _CountingRepo(
        {"ev-rec": evidence}, {"tr-rec": {"tool_name": "get_reg_sho_daily", "session_id": "s", "result": payload}}
    )
    with caplog.at_level("DEBUG", logger="app.research.sources"):
        label = _sources.read_source_label("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert label == {
        "id": "evidence:ev-rec",
        "title": "FINRA",
        "publisher": "FINRA",
        "publishedAt": "2026-01-02",
        "linkStatus": "missing",
    }
    assert (calls, repo.replays, caplog.records) == ([], [], [])
    assert not (tmp_path / "raw").exists()
    document = _sources.read_source_view("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert calls == [("otcMarket/regShoDaily", row, payload)]
    assert repo.replays == ["tr-rec"]
    assert (document["originalUrl"], document["linkStatus"], document["quote"]) == (_SHVOL_URL, "stockbot_render", line)


def test_label_links_only_already_pinned_human_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Labels carry the archived www.sec.gov page or the web page; API hosts and highlights never appear."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    page = "https://www.sec.gov/Archives/edgar/data/42/000000000026000042/primary.htm"
    official = sec_archive.archive_sec_document(
        ACC, DOC, b"<p>quoted passage</p>", url=page, metadata={}, root=tmp_path / "raw"
    ).sha256
    api = sec_archive.archive_sec_document(
        ACC, "api.json", b"{}", url="https://data.sec.gov/api/xbrl/x.json", metadata={}, root=tmp_path / "raw"
    ).sha256
    api_row = _sec_evidence("ev-api", api, "x", 0, "text")
    api_provenance = api_row["provenance"]
    assert isinstance(api_provenance, dict)
    api_provenance["document_name"] = "api.json"
    web = {"kind": "web_source", "url": "https://example.com/a", "excerpt": "x", "tool_result_id": "tr"}
    repo = _CountingRepo(
        {
            "ev-sec": _sec_evidence("ev-sec", official, "quoted passage", 3, "rendered"),
            "ev-api": api_row,
            "ev-web": {"evidence_id": "ev-web", "session_id": "s", "source_name": "Exa", "provenance": web},
            "ev-rec": _record_evidence("sec_record", "get_xbrl_facts", {"x": 1}, source_uri="https://data.sec.gov/a"),
        }
    )
    labels = {
        evidence_id: _sources.read_source_label(f"evidence:{evidence_id}", data_root=tmp_path, repo=repo)
        for evidence_id in ("ev-sec", "ev-api", "ev-web", "ev-rec")
    }
    assert {key: (value["linkStatus"], value.get("originalUrl")) for key, value in labels.items()} == {
        "ev-sec": ("stockbot_render", page),
        "ev-api": ("missing", None),
        "ev-web": ("stockbot_render", "https://example.com/a"),
        "ev-rec": ("missing", None),
    }
    assert all("highlightUrl" not in label and "content" not in label for label in labels.values())
    assert repo.replays == []


class _FinraResponse:
    def __init__(self, url: str, history: tuple[_FinraResponse, ...] = ()) -> None:
        self.url, self.history, self.content = url, list(history), _FNQC

    def raise_for_status(self) -> None:
        return None


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("offline", id="network-down"),
        pytest.param(_FinraResponse("https://evil.example/FNQCshvol20261015.txt"), id="final-host-off-official"),
        pytest.param(_FinraResponse(_FNQC_URL, (_FinraResponse("https://evil.example/hop"),)), id="redirect-hop"),
        pytest.param(_FinraResponse(_FNQC_URL.replace("https:", "http:")), id="https-downgrade"),
    ],
)
def test_later_official_fetch_failure_is_controlled_and_never_mutates_evidence(
    failure: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed or redirected official fetch drops only the public link; evidence and its audit stay unchanged."""
    import copy

    import requests

    from app import finra_client

    def get(url: str, timeout: int) -> object:
        if failure == "offline":
            raise requests.ConnectionError("finra offline")
        return failure

    monkeypatch.setattr("app.finra_client.requests.get", get)
    repo = _record_repo("finra_record", "get_reg_sho_volume", _FNQC_ROW, {}, dataset="otcMarket/regShoDaily")
    stored = copy.deepcopy(repo.get_evidence("ev-rec"))
    with caplog.at_level("ERROR", logger="app.research.sources"):
        failed = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert (failed["captureKind"], failed["linkStatus"]) == ("structured_record", "missing")
    assert not {"originalUrl", "highlightUrl"} & set(failed)
    assert json.loads(str(failed["content"])) == _FNQC_ROW
    [record] = [r for r in caplog.records if r.levelname == "ERROR"]
    message = record.getMessage()
    assert message.startswith("source official page missing: source_id=evidence:ev-rec evidence_id=ev-rec")
    assert "reason=official page fetch failed:" in message
    assert message.endswith(" failure=SOURCE_UNAVAILABLE")
    assert raw_archive.find("finra", finra_client._OFFICIAL_FILE_KIND, _FNQC_URL, root=tmp_path / "raw") is None

    def healthy_get(url: str, timeout: int) -> _FinraResponse:
        return _FinraResponse(url)

    monkeypatch.setattr("app.finra_client.requests.get", healthy_get)
    healthy = read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert (healthy["originalUrl"], healthy["linkStatus"], healthy["captureKind"]) == (
        _FNQC_URL,
        "stockbot_render",
        "original",
    )
    assert repo.get_evidence("ev-rec") == stored
    assert failed["audit"] == healthy["audit"]


@pytest.mark.parametrize(
    ("lines", "failure", "reason"),
    [
        pytest.param(["20261015|NVDA|1|0|2|Q"], "SOURCE_MISMATCH", "disagrees", id="file-disagrees"),
        pytest.param(
            ["20261015|NVDA|1|0|2|B", "20261015|NVDA|3|0|4|B"], "SOURCE_AMBIGUOUS", "ambiguous", id="ambiguous"
        ),
    ],
)
def test_published_file_conflict_keeps_api_record_and_reports_conflict_apart(
    lines: list[str],
    failure: str,
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A conflicting or ambiguous FINRA file drops only the public link. The API record and its audit stay."""
    import copy

    body = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n" + "\n".join(lines) + "\n").encode()

    def healthy_get(url: str, timeout: int) -> _FinraResponse:
        return _FinraResponse(url)

    def conflicting_get(url: str, timeout: int) -> _FinraResponse:
        response = _FinraResponse(url)
        response.content = body
        return response

    monkeypatch.setattr("app.finra_client.requests.get", healthy_get)
    healthy = read_source_document(
        "evidence:ev-rec",
        data_root=tmp_path / "healthy",
        repo=_record_repo("finra_record", "get_reg_sho_volume", _FNQC_ROW, {}, dataset="otcMarket/regShoDaily"),
    )
    monkeypatch.setattr("app.finra_client.requests.get", conflicting_get)
    repo = _record_repo("finra_record", "get_reg_sho_volume", _FNQC_ROW, {}, dataset="otcMarket/regShoDaily")
    stored = copy.deepcopy(repo.get_evidence("ev-rec"))
    with caplog.at_level("ERROR", logger="app.research.sources"):
        document = _sources.read_source_view("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert "error" not in document
    assert (document["captureKind"], document["linkStatus"], document["quote"]) == ("structured_record", "missing", "")
    assert not {"originalUrl", "highlightUrl"} & set(document)
    assert json.loads(str(document["content"])) == _FNQC_ROW
    conflict = document["sourceConflict"]
    assert isinstance(conflict, dict) and conflict["failure"] == failure
    assert reason in str(conflict["reason"]) and _FNQC_URL in str(conflict["reason"])
    # The API proof verdict is unchanged by the file conflict; the conflict lives beside it.
    assert document["audit"] == healthy["audit"] and "sourceConflict" not in healthy
    [record] = [r for r in caplog.records if r.levelname == "ERROR"]
    assert record.getMessage().endswith(f" failure={failure}")
    assert repo.get_evidence("ev-rec") == stored


def test_corrupt_finra_archive_never_renders_as_conflict_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Edited archived file bytes stay a controlled 409 SOURCE_CORRUPT read error, never a structured fallback."""

    def healthy_get(url: str, timeout: int) -> _FinraResponse:
        return _FinraResponse(url)

    monkeypatch.setattr("app.finra_client.requests.get", healthy_get)
    repo = _record_repo("finra_record", "get_reg_sho_volume", _FNQC_ROW, {}, dataset="otcMarket/regShoDaily")
    assert read_source_document("evidence:ev-rec", data_root=tmp_path, repo=repo)["captureKind"] == "original"
    archived = raw_archive.find("finra", "official_file", _FNQC_URL, root=tmp_path / "raw")
    assert archived is not None
    archived.payload_path.write_bytes(_FNQC.replace(b"274457.043010", b"274457.043011"))
    view = _sources.read_source_view("evidence:ev-rec", data_root=tmp_path, repo=repo)
    assert (view["status"], view["failure"]) == (409, "SOURCE_CORRUPT")
    assert "sourceConflict" not in view and "content" not in view


def test_sec_official_fetch_rejects_off_host_redirects_and_transport_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SEC page fetch keeps only a final https://www.sec.gov/ response; failures surface as OSError."""
    import httpx
    from edgar import httprequests

    from app.sec import client, documents

    class _Response:
        def __init__(self, url: str) -> None:
            self.url, self.content = url, b"<p>official</p>"

    final: dict[str, str] = {}

    def get_with_retry(url: str) -> _Response:
        if final["url"] == "transport":
            raise httpx.ConnectError("sec offline")
        return _Response(final["url"])

    def no_identity() -> None:
        return None

    def accept(response: object) -> None:
        return None

    monkeypatch.setattr(client, "ensure_identity", no_identity)
    monkeypatch.setattr(httprequests, "get_with_retry", get_with_retry)
    monkeypatch.setattr(httprequests, "inspect_response", accept)
    page = "https://www.sec.gov/Archives/edgar/data/42/x.htm"
    final["url"] = page
    assert documents._sec_get(page) == b"<p>official</p>"
    for url, message in (
        ("https://evil.example/x.htm", "redirect left the official SEC host"),
        ("https://data.sec.gov/api/xbrl/x.json", "redirect left the official SEC host"),
        ("http://www.sec.gov/Archives/edgar/data/42/x.htm", "redirect left the official SEC host"),
        ("transport", "official SEC fetch failed"),
    ):
        final["url"] = url
        with pytest.raises(OSError, match=message):
            documents._sec_get(page)


def _identity_missing(root: Path) -> tuple[str, _Repo]:
    row = {"evidence_id": "ev-x", "session_id": "s", "source_name": "SEC", "provenance": {"kind": "none"}}
    return "evidence:ev-x", _Repo({"ev-x": row})


def _replay_identity_missing(root: Path) -> tuple[str, _Repo]:
    row = _record_evidence("finra_record", "query_finra", {"symbolCode": "ACME"})
    provenance = row["provenance"]
    assert isinstance(provenance, dict)
    provenance["tool_result_id"] = ""
    return "evidence:ev-rec", _Repo({"ev-rec": row})


def _session_mismatch(root: Path) -> tuple[str, _Repo]:
    row = _record_evidence("finra_record", "query_finra", {"symbolCode": "ACME"})
    return "evidence:ev-rec", _Repo({"ev-rec": row}, {"tr-rec": {"tool_name": "query_finra", "session_id": "other"}})


def _two_revisions(root: Path) -> tuple[str, _Repo]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # A second revision warns by design.
        for body in (b"alpha quoted passage", b"omega quoted passage"):
            sec_archive.archive_sec_document(
                ACC, DOC, body, url="https://www.sec.gov/x", metadata=_metadata("ev-amb"), root=root / "raw"
            )
    return "evidence:ev-amb", _Repo({"ev-amb": _sec_evidence("ev-amb", "", "quoted passage", 6, "text")})


def _replay_missing(root: Path) -> tuple[str, _Repo]:
    return "evidence:ev-rec", _Repo({"ev-rec": _record_evidence("finra_record", "query_finra", {"symbolCode": "A"})})


def _evidence_missing(root: Path) -> tuple[str, _Repo]:
    return "evidence:ev-gone", _Repo({})


def _web_capture(root: Path) -> tuple[str, raw_archive.ArchiveRecord]:
    with capture_scope(str(root)) as sources:
        capture_result("search_web", {"evidence": [{"url": "https://example.com/a", "highlight": "quoted"}]})
    source_id = sources[0]["id"]
    digest = source_id.removeprefix("capture:")
    record = raw_archive.find("viewer", "capture", digest, sha256=digest, root=root / "raw")
    assert record is not None
    return source_id, record


def _unreadable_capture(root: Path) -> tuple[str, _Repo]:
    source_id, record = _web_capture(root)
    record.payload_path.unlink()
    record.payload_path.mkdir()  # Present but unreadable: the read raises OSError, not a missing capture.
    return source_id, _Repo({})


def _edited_capture(root: Path) -> tuple[str, _Repo]:
    source_id, record = _web_capture(root)
    payload = record.payload_path.read_bytes()
    record.payload_path.write_bytes(payload.replace(b"quoted", b"QUOTED"))
    return source_id, _Repo({})


@pytest.mark.parametrize(
    ("build", "status", "failure"),
    [
        pytest.param(_identity_missing, 404, "SOURCE_IDENTITY_MISSING", id="identity-never-retained"),
        pytest.param(_replay_identity_missing, 404, "SOURCE_IDENTITY_MISSING", id="replay-id-never-retained"),
        pytest.param(_session_mismatch, 409, "SOURCE_MISMATCH", id="replay-from-other-session"),
        pytest.param(_two_revisions, 409, "SOURCE_AMBIGUOUS", id="two-archived-revisions"),
        pytest.param(_replay_missing, 404, "SOURCE_UNAVAILABLE", id="tool-result-gone"),
        pytest.param(_evidence_missing, 404, "SOURCE_UNAVAILABLE", id="evidence-gone"),
        pytest.param(_unreadable_capture, 502, "SOURCE_UNAVAILABLE", id="capture-unreadable"),
        pytest.param(_edited_capture, 409, "SOURCE_CORRUPT", id="capture-hash-mismatch"),
    ],
)
def test_viewer_read_failure_returns_controlled_error_with_internal_failure_class(
    build: Callable[[Path], tuple[str, _Repo]],
    status: int,
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each ordinary failure becomes one public error state; only the ERROR log names the failure class."""
    import copy

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    source_id, repo = build(tmp_path)
    stored = copy.deepcopy(repo._evidence)
    with caplog.at_level("ERROR", logger="app.research.sources"):
        state = _sources.read_source_view(source_id, data_root=tmp_path, repo=repo)
    public = {404: "source capture not found", 409: "invalid source capture", 502: "source read failed"}
    assert state == {"id": source_id, "error": public[status], "status": status, "failure": failure}
    [record] = [r for r in caplog.records if r.levelname == "ERROR"]
    evidence_id = source_id.removeprefix("evidence:") if source_id.startswith("evidence:") else ""
    assert record.getMessage().startswith(
        f"source read failed: failure={failure} source_id={source_id} evidence_id={evidence_id} error="
    )
    assert record.exc_info is not None
    assert repo._evidence == stored


def test_failure_classes_are_the_seven_internal_reasons() -> None:
    assert _sources.FAILURE_REASONS == (
        "PUBLIC_PAGE_UNAVAILABLE",
        "SOURCE_IDENTITY_MISSING",
        "SOURCE_MISMATCH",
        "SOURCE_AMBIGUOUS",
        "SOURCE_UNAVAILABLE",
        "SOURCE_CORRUPT",
        "CALCULATION_INVALID",
    )
    assert (
        _sources.failure_reason(ValueError("Archived FINRA file does not match its content hash")) == "SOURCE_CORRUPT"
    )
    assert _sources.failure_reason(ValueError("x.txt: ACME line disagrees with the cited row")) == "SOURCE_MISMATCH"
    assert _sources.failure_reason(OSError("x.txt: redirect left the official FINRA host")) == "SOURCE_UNAVAILABLE"
    assert _sources.failure_reason(KeyError("unknown evidence_id")) == "SOURCE_UNAVAILABLE"


_PROOF_ACC = "0001045810-24-000124"
_PROOF_DOC_NAME = "nvda-20240428.htm"
_PROOF_PAGE = f"https://www.sec.gov/Archives/edgar/data/1045810/000104581024000124/{_PROOF_DOC_NAME}"


def _proof_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ResearchRepository, str]:
    """Real ledger: two archived SEC observations, a verified difference, and a calculation missing an input."""
    import dataclasses

    from app.research.evidence import evidence_to_dict
    from tests.test_research_harness import _PROOF_DC, _PROOF_DOC, _PROOF_TOT, _calculation_row, _observation_row

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    digest = sec_archive.archive_sec_document(
        _PROOF_ACC, _PROOF_DOC_NAME, _PROOF_DOC, url=_PROOF_PAGE, metadata={}, root=tmp_path / "raw"
    ).sha256
    repo = ResearchRepository()
    pinned: dict[str, JSONValue] = {"source_content_hash": digest}
    total = dataclasses.replace(_observation_row("EV-TOT", _PROOF_TOT, "26,044"), metadata=pinned)
    dc = dataclasses.replace(_observation_row("EV-DC", _PROOF_DC, "22,563"), metadata=pinned)
    for row in (
        total,
        dc,
        _calculation_row("EV-NONDC", "3481", ["EV-TOT", "EV-DC"], rows=[total, dc]),
        _calculation_row("EV-BAD", "3481", ["EV-TOT", "EV-GONE"]),
    ):
        repo.save_evidence(evidence_to_dict(row))
    return repo, digest


def test_calculation_view_audits_each_input_record_and_never_borrows_an_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A verified calculation shows formula, recomputed value, and verified inputs that open their own views."""
    repo, digest = _proof_ledger(tmp_path, monkeypatch)
    with caplog.at_level("ERROR", logger="app.research.sources"):
        document = read_source_document("evidence:EV-NONDC", data_root=tmp_path, repo=repo)
    assert [r for r in caplog.records if r.levelname == "ERROR"] == []
    assert (document["captureKind"], document["representation"], document["linkStatus"], document["publisher"]) == (
        "calculation",
        "text",
        "missing",
        "Stockbot",
    )
    assert not {"originalUrl", "highlightUrl", "pdfUrl"} & set(document)
    audit = document["audit"]
    assert isinstance(audit, dict)
    assert {key: audit.get(key) for key in ("sourceId", "claimType", "verification", "value", "formulaId")} == {
        "sourceId": "evidence:EV-NONDC",
        "claimType": "calculation",
        "verification": "verified",
        "value": "3481",
        "formulaId": "difference",
    }
    assert (audit["formulaVersion"], audit["recomputedValue"], audit["entity"], audit["period"]) == (
        "1",
        "3481",
        "NVDA",
        "Q1 FY2025",
    )
    inputs = audit["inputs"]
    assert isinstance(inputs, list)
    assert [(item["sourceId"], item["verification"], item["value"]) for item in inputs] == [
        ("evidence:EV-TOT", "verified", "26,044"),
        ("evidence:EV-DC", "verified", "22,563"),
    ]
    assert [(item["source"]["hash"], item["source"]["accession"]) for item in inputs] == [
        (f"sha256:{digest}", _PROOF_ACC)
    ] * 2
    assert "data.sec.gov" not in json.dumps(document)
    # Each input opens its own record: an SEC original with its own official page and observation audit.
    observation = read_source_document(str(inputs[1]["sourceId"]), data_root=tmp_path, repo=repo)
    assert (observation["captureKind"], observation["originalUrl"], observation["contentHash"]) == (
        "original",
        _PROOF_PAGE,
        digest,
    )
    assert observation["quote"] == "NVDA Q1 FY2025 Data Center revenue was $22,563 million"
    observed = observation["audit"]
    assert isinstance(observed, dict)
    assert (observed["claimType"], observed["verification"], observed["value"]) == (
        "observed_fact",
        "verified",
        "22,563",
    )
    assert source_references(repo.get_evidence("EV-NONDC")) == [
        {"id": "evidence:EV-NONDC", "title": "Stockbot calculation"}
    ]
    assert _sources.read_source_label("evidence:EV-NONDC", data_root=tmp_path, repo=repo) == {
        "id": "evidence:EV-NONDC",
        "title": "c",  # The row's claim text: _calculation_row stores no metric.
        "publisher": "Stockbot",
        "knownAt": "2025-05-01T00:00:00+00:00",
        "linkStatus": "missing",
    }


def test_viewer_audit_shows_repair_history_beside_unverified_legacy_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repaired legacy row still fails verification, but the viewer shows the repair artifact."""
    import json

    from app.research import repair
    from app.storage import raw_archive
    from tests.test_research_harness import _REP_CIK, _REP_SID, _rep_record, _rep_result, _rep_row

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    repo = ResearchRepository()
    q1 = {"period_end": "2025-04-27", "period_start": "2025-01-27", "fiscal_period": "Q1", "eps_diluted": 0.76}
    _rep_result(repo, "TR-LEGACY", "get_fundamentals", {"ticker": "NVDA", "cik": _REP_CIK, "quarterly_eps": [q1]})
    _rep_row(repo, "EV-LEGACY", "2025-09-01T00:00:00+00:00", _rep_record("get_fundamentals", q1, "TR-LEGACY"))
    raw_archive.store_evidence_artifact(
        json.dumps(repo.get_tool_result("TR-LEGACY"), sort_keys=True, default=str).encode(),
        metadata={"session_id": _REP_SID, "tool_result_id": "TR-LEGACY"},
    )
    [artifact] = repair.repair_evidence(repo, "EV-LEGACY")
    assert artifact["status"] == "verified"
    stored = repo.get_evidence("EV-LEGACY")
    assert stored["proof"] == {}
    document = read_source_document("evidence:EV-LEGACY", data_root=tmp_path, repo=repo)
    assert isinstance(document.get("audit"), dict)
    assert (document["audit"]["verification"], document["audit"].get("value")) == ("unverified", None)
    assert "datapoint" not in document["audit"]
    [shown] = document["audit"].get("repairs", [])
    assert isinstance(shown, dict)
    assert (shown["repair_id"], shown["status"], shown["category"]) == (
        artifact["repair_id"],
        "verified",
        "eps_without_accession",
    )
    assert shown["value"] == "0.76"
    assert set(shown) <= {"repair_id", "status", "reason", "category", "datapoint", "value"}
    assert repo.get_evidence("EV-LEGACY") == stored
    assert "data.sec.gov" not in json.dumps(document)


def test_calculation_with_absent_input_is_invalid_logged_and_left_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CALCULATION_INVALID: the view still opens, the audit fails closed, the ERROR log names it, nothing persists."""
    repo, _ = _proof_ledger(tmp_path, monkeypatch)
    stored = repo.get_evidence("EV-BAD")
    with caplog.at_level("ERROR", logger="app.research.sources"):
        document = _sources.read_source_view("evidence:EV-BAD", data_root=tmp_path, repo=repo)
    audit = document["audit"]
    assert isinstance(audit, dict)
    assert (document["captureKind"], document["linkStatus"], audit["verification"]) == (
        "calculation",
        "missing",
        "unverified",
    )
    inputs = audit["inputs"]
    assert isinstance(inputs, list)
    assert [(item["evidenceId"], item["verification"]) for item in inputs] == [
        ("EV-TOT", "verified"),
        ("EV-GONE", "unverified"),
    ]
    [record] = [r for r in caplog.records if r.levelname == "ERROR"]
    assert record.getMessage().startswith(
        "source calculation invalid: failure=CALCULATION_INVALID source_id=evidence:EV-BAD evidence_id=EV-BAD reason="
    )
    assert repo.get_evidence("EV-BAD") == stored
    assert repo.list_evidence_repairs("S-proof") == []


@pytest.mark.parametrize("public_file", [True, False], ids=["finra-captured-record", "finra-api-only"])
def test_e2e_citations_open_verified_views_with_exact_replay(
    public_file: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing public file changes the link, never the proof, frozen bytes, or publication result."""
    import tests.research_source_seam as seam
    from app.research import service
    from app.research.freeze import freeze_from_dict, verify_freeze
    from tests.test_research_harness import (
        _E2E_FINRA_ROWS,
        _E2E_FULL,
        _E2E_WEB,
        _e2e_finra_line,
        _e2e_publish,
        _e2e_rows,
        _e2e_session,
    )

    seam.install(monkeypatch)
    e2e = _e2e_session(tmp_path, monkeypatch)
    line = _e2e_finra_line(_E2E_FINRA_ROWS[0])
    url = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20250613.csv"
    body = line.encode()
    calls = _official(monkeypatch, _page(url, body, "text", line) if public_file else None, finra=True)
    rows = _e2e_rows(e2e)
    before = e2e.repo.list_evidence(e2e.sid)
    frozen = freeze_from_dict(e2e.repo.get_freeze(f"{e2e.sid}:1:freeze"))
    final = _e2e_publish(e2e)["final_result"]
    assert isinstance(final, dict)
    # Citation labels must read only metadata. Neither tool replay nor official resolution is permitted.
    with monkeypatch.context() as labels:

        def no_replay(*args: object, **kwargs: object) -> object:
            pytest.fail("A citation label tried to replay a tool result")

        labels.setattr(e2e.repo, "get_tool_result", no_replay)
        for key, evidence in rows.items():
            [ref] = source_references(e2e.repo.get_evidence(evidence.evidence_id))
            assert ref["id"] == f"evidence:{e2e.ids[key]}"
            label = _sources.read_source_label(ref["id"], data_root=e2e.data, repo=e2e.repo)
            assert label["id"] == ref["id"] and "audit" not in label
    assert calls == []

    documents = {}
    for key, evidence in rows.items():
        [ref] = source_references(e2e.repo.get_evidence(evidence.evidence_id))
        source_id = ref["id"]
        view = read_source_document(source_id, data_root=e2e.data, repo=e2e.repo)
        assert view["id"] == source_id
        audit = view["audit"]
        assert isinstance(audit, dict)
        assert (audit["sourceId"], audit["verification"], audit["value"]) == (source_id, "verified", evidence.value)
        proof = e2e.repo.get_evidence(evidence.evidence_id)["proof"]
        assert isinstance(proof, dict)
        locator = proof.get("locator")
        descriptor = proof.get("datapoint") or (locator.get("datapoint") if isinstance(locator, dict) else None)
        # A verified audit carries the kernel-built canonical datapoint, id included.
        assert isinstance(descriptor, dict) and audit["datapoint"] == descriptor
        assert read_source_document(source_id, data_root=e2e.data, repo=ResearchRepository()) == view
        documents[key] = view

    assert documents["tot"]["content"] == _E2E_FULL
    assert documents["tot"]["contentHash"] == raw_archive.content_hash(_E2E_FULL.encode())
    assert documents["web"]["captureKind"] == "search_excerpt"
    assert documents["web"]["content"] == _E2E_WEB
    finra = documents["finra"]
    assert finra["linkStatus"] == ("stockbot_render" if public_file else "missing")
    assert finra.get("originalUrl") == (url if public_file else None)
    if public_file:
        assert finra["content"] == line and finra["contentHash"] == raw_archive.content_hash(body)
    calc = documents["nested"]
    assert calc["captureKind"] == "calculation" and "originalUrl" not in calc
    nested_audit = calc["audit"]
    assert isinstance(nested_audit, dict) and nested_audit["recomputedValue"] == "26044"
    inputs = nested_audit["inputs"]
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    assert inputs[0]["recomputedValue"] == "3481"
    assert e2e.repo.list_evidence(e2e.sid) == before
    assert e2e.repo.list_evidence_repairs(e2e.sid) == []
    verify_freeze(frozen, list(_e2e_rows(e2e).values()))
    rows_by_id = service._session_rows(e2e.repo, e2e.sid)
    cutoff = e2e.repo.get_session(e2e.sid).as_of
    assert service._proof_verdict(rows_by_id, rows["finra"], "12345", as_of=cutoff).status == "verified"


@pytest.mark.parametrize("element_id", ["f-hidden", None], ids=["hidden-fact", "canonical-fact"])
def test_e2e_sec_case_two_publishes_api_proof_and_replays_official_bytes(
    element_id: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified API fact publishes without a fabricated passage or highlighted official link."""
    from dataclasses import replace

    from app.research import service
    from app.research.evidence import datapoint_descriptor, evidence_from_dict
    from app.research.freeze import freeze_from_dict, verify_freeze
    from app.sec import documents
    from tests.test_research_harness import _committee_analysis
    from tests.test_sec_documents_archive import (
        OFF_CIK,
        OFF_INDEX,
        OFF_PRIMARY,
        OFF_PRIMARY_HTML,
        _FakeSec,
        _revenue_row,
    )

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    network = _FakeSec()
    monkeypatch.setattr(documents, "_sec_get", network)
    repo = ResearchRepository()
    sid = service.create_research("Revenue?", as_of="2026-08-01", repo=repo)
    job = repo.list_jobs(sid)[0].job_id
    repo.save_job(replace(repo.get_job(job), source_domain="SEC"))
    # SEC availability comes from the cited record's own filing metadata, never the caller.
    fact = _revenue_row(filed_at="2026-07-20", **({"element_id": element_id} if element_id is not None else {}))
    service.persist_tool_result(sid, job, "get_xbrl_facts", "TR-CASE-TWO", fact, repo=repo)
    service.record_evidence(
        sid,
        job,
        {
            "evidence_id": "EV-CASE-TWO",
            "tool_result_id": "TR-CASE-TWO",
            "matching_passage": json.dumps(fact, sort_keys=True),
            "known_at": "2026-07-20",
            "value": str(fact["value"]),
            "metric": "us-gaap:Revenues",
            "entity": str(OFF_CIK),
            "period": "2026-06-30",
            "unit": "USD",
        },
        repo=repo,
    )
    evidence = evidence_from_dict(repo.get_evidence("EV-CASE-TWO"))
    source_bytes = service._proof_bytes(evidence)
    assert source_bytes is not None
    assert source_bytes == json.dumps(repo.get_tool_result("TR-CASE-TWO"), sort_keys=True, default=str).encode()
    assert evidence.proof["sourceId"] == f"sha256:{raw_archive.content_hash(source_bytes)}"
    service.complete_job(job, {}, repo=repo)
    service.freeze_session(sid, repo=repo)
    frozen = freeze_from_dict(repo.get_freeze(f"{sid}:1:freeze"))
    point = datapoint_descriptor(evidence)
    assert point is not None
    claim = {
        "text": f"{evidence.entity} Revenues for {evidence.period} was {evidence.value} USD",
        "claim_type": "observed_fact",
        "evidence_ids": ["EV-CASE-TWO"],
        "datapoint_ids": [point["id"]],
    }
    for role in ("stockbot", "bullbot", "bearbot"):
        role_id = str(service.start_job(sid, role, repo=repo)["job_id"])
        service.record_committee_analysis(sid, role_id, role, _committee_analysis("EV-CASE-TWO"), repo=repo)
    published = service.finalize_session(sid, claim["text"], [claim], repo=repo)
    assert published["status"] == "completed"
    [ref] = source_references(repo.get_evidence("EV-CASE-TWO"))
    first = read_source_document(ref["id"], data_root=tmp_path, repo=repo)
    assert (first["linkStatus"], first["originalUrl"], first["quote"]) == ("stockbot_render", OFF_PRIMARY, "")
    assert "highlightUrl" not in first
    assert first["content"] == OFF_PRIMARY_HTML
    audit = first["audit"]
    assert isinstance(audit, dict)
    assert (audit["verification"], audit["value"]) == ("verified", evidence.value)
    assert network.calls == [OFF_INDEX, OFF_PRIMARY]
    network.pages.clear()
    assert read_source_document(ref["id"], data_root=tmp_path, repo=ResearchRepository()) == first
    assert network.calls == [OFF_INDEX, OFF_PRIMARY]
    assert service._proof_bytes(evidence) == source_bytes
    verify_freeze(frozen, [evidence_from_dict(repo.get_evidence("EV-CASE-TWO"))])
