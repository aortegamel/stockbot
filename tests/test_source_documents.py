"""Immutable source captures: single-shot archives plus durable evidence reads."""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.research import sources as _sources
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


def test_capture_structured_record_archives_finra_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    with capture_scope() as sources:
        capture_result("browse_tools", {"content": "ignored navigation"})
        capture_result("query_finra", {"content": "short interest rows", "row": 1})
    assert len(sources) == 1
    document = read_source_document(sources[0]["id"], data_root=tmp_path)
    assert document["captureKind"] == "structured_record"
    assert document["representation"] == "json"
    assert document["publisher"] == "FINRA"


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
    payload = {"evidence": [{"url": "https://example.com/a", "highlight": "the quoted excerpt here"}]}

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


@pytest.mark.parametrize(
    ("tool", "kind", "publisher"),
    [("query_finra", "finra_record", "FINRA"), ("get_fundamentals", "sec_record", "SEC")],
)
def test_structured_capture_matches_persisted_record_evidence(
    tool: str, kind: str, publisher: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-shot record capture and its persisted evidence replay show the same exact content."""
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
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
    keys = ("content", "contentHash", "captureKind", "representation")
    assert {key: persisted[key] for key in keys} == {key: captured[key] for key in keys}
    assert captured["contentHash"] == raw_archive.content_hash(str(captured["content"]).encode())
    assert (captured["title"], captured["publisher"], captured["quote"]) == (tool, publisher, "")
    assert {key: persisted[key] for key in ("title", "url", "publisher", "publishedAt", "capturedAt", "quote")} == {
        "title": "Record",
        "url": "https://example.com/record",
        "publisher": publisher,
        "publishedAt": "2026-01-01",
        "capturedAt": "2026-01-02",
        "quote": "ACME 2026-01-02",
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
    drifted = {"evidence": [{"url": "https://example.com/a", "highlight": "quoted excerpt", "source_content": "drift"}]}
    repo = _Repo(
        {"ev-pin": {"evidence_id": "ev-pin", "session_id": "s", "source_name": "Exa", "provenance": provenance}},
        {"tr-1": {"tool_name": "search_web", "session_id": "s", "result": drifted}},
    )
    document = read_source_document("evidence:ev-pin", data_root=tmp_path, repo=repo)
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
    }


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
