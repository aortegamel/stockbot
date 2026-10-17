from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

import app.tool_runtime as rt
from app.research.sources import read_source_document, retain_sec_bytes
from app.storage import raw_archive
from app.storage.runs import RunRecorder, reset_current_recorder, set_current_recorder
from app.tool_runtime import AgentToolSession
from scripts import tool_bridge

_ACC = "0000000000-26-000042"
_DOC = "primary.htm"
_SEC_BYTES = b"<html><body><p>before quoted after</p></body></html>"


def _web_row(highlight: str) -> dict[str, object]:
    return {
        "title": "T",
        "url": "https://example.com/a",
        "source_domain": "example.com",
        "highlight": highlight,
        "source_content": f"full page with {highlight}",
    }


def _handler(highlight: str) -> Callable[..., dict[str, object]]:
    """Stub only the tool handler; every runtime gate and capture call stays real."""

    def _execute(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        if name == "search_web":
            return {"query": arguments.get("query"), "evidence": [_web_row(highlight)], "source": "exa"}
        assert name == "get_sec_document"
        # The real SEC retrieval retains its exact bytes the same way (app/sec/documents.py).
        retain_sec_bytes(_ACC, _DOC, _SEC_BYTES, {"representation": "source_bytes"})
        return {
            "text": "quoted",
            "accession_no": _ACC,
            "document_name": _DOC,
            "source_content_hash": hashlib.sha256(_SEC_BYTES).hexdigest(),
            "source_url": "https://www.sec.gov/x",
            "source_handle": {"basis": "rendered"},
            "offset": 7,
            "end_offset": 13,
        }

    return _execute


def _invoke(name: str, arguments: dict[str, object], data_root: Path) -> dict[str, object]:
    tool_bridge._sessions.pop("capture-1", None)
    try:
        response = tool_bridge.handle(
            {
                "id": "c1",
                "op": "tool.invoke",
                "name": name,
                "arguments": arguments,
                "session_id": "capture-1",
                "data_root": str(data_root),
            }
        )
    finally:
        tool_bridge._sessions.pop("capture-1", None)
    result = response["result"]
    assert isinstance(result, dict)
    return result


def _viewer_documents(result: dict[str, object], data_root: Path) -> list[dict[str, object]]:
    meta = result["meta"]
    assert isinstance(meta, dict)
    sources = meta["sources"]
    assert isinstance(sources, list) and sources
    return [read_source_document(source["id"], data_root=data_root) for source in sources]


def test_accepted_web_result_captures_raw_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _handler("quoted passage"))
    result = _invoke("search_web", {"query": "NVDA demand"}, tmp_path)
    assert "error" not in result, result
    [document] = _viewer_documents(result, tmp_path)
    assert document["captureKind"] == "provider_extraction"
    assert document["content"] == "full page with quoted passage"
    assert document["quote"] == "quoted passage"
    assert document["url"] == "https://example.com/a"


def test_accepted_sec_result_captures_original_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _handler(""))
    result = _invoke("get_sec_document", {"accession_no": _ACC, "document_name": _DOC}, tmp_path)
    assert "error" not in result, result
    [document] = _viewer_documents(result, tmp_path)
    assert document["captureKind"] == "original"
    assert document["contentHash"] == hashlib.sha256(_SEC_BYTES).hexdigest()
    assert document["content"] == _SEC_BYTES.decode()
    assert document["quote"] == "quoted"


def test_rejected_result_creates_no_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _handler("leaked Bearer abc123def456"))
    result = _invoke("search_web", {"query": "NVDA demand"}, tmp_path)
    assert result.get("error_type") == "ingress_blocked"
    assert "meta" not in result
    assert not (tmp_path / "raw").exists()


def test_capture_failure_rejects_result_without_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _two_rows(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        return {"evidence": [_web_row("first passage"), {**_web_row("second passage"), "url": "https://example.com/b"}]}

    real_archive = raw_archive.archive
    calls: list[str] = []

    def _fail_second(
        source: str, kind: str, key: str, payload: bytes, *, url: str, root: Path | None = None
    ) -> raw_archive.ArchiveRecord:
        calls.append(key)
        if len(calls) == 2:
            raise OSError(f"disk full at {tmp_path}/raw")
        return real_archive(source, kind, key, payload, url=url, root=root)

    monkeypatch.setattr(rt, "execute_tool", _two_rows)
    monkeypatch.setattr(raw_archive, "archive", _fail_second)
    monkeypatch.setenv("RUNS_DB_PATH", str(tmp_path / "runs.sqlite"))
    recorder = RunRecorder(
        run_id="run-capture-fail",
        request_id="req",
        question="q",
        as_of=None,
        model="t",
        provider="p",
        model_parameters={},
        agent_version="0",
        prompt_version="0",
        tool_registry_version="0",
        git_sha="0",
    )
    with recorder:
        token = set_current_recorder(recorder)
        try:
            result = _invoke("search_web", {"query": "NVDA demand"}, tmp_path)
        finally:
            reset_current_recorder(token)
    assert len(calls) == 2  # the first row archived, so a partial source reference existed
    assert result.get("error_type") == "source_capture_failed"
    assert "content" not in result and "meta" not in result
    assert str(tmp_path) not in json.dumps(result) and "disk full" not in json.dumps(result)
    conn = sqlite3.connect(str(tmp_path / "runs.sqlite"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
    finally:
        conn.close()


def test_describe_exposes_canonical_research_tools() -> None:
    response = tool_bridge.handle({"id": "d1", "op": "describe"})
    tools = response.get("tools")
    assert isinstance(tools, list)
    names = {
        fn["name"]
        for item in tools
        if isinstance(item, dict) and isinstance((fn := item.get("function")), dict) and isinstance(fn.get("name"), str)
    }
    assert "search_web" in names
    assert "search_sec_filings" in names


def test_invoke_reuses_generic_session_and_end_drops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[AgentToolSession] = []

    def fake_execute(
        name: str,
        arguments: dict[str, object],
        session: AgentToolSession,
        **kwargs: object,
    ) -> dict[str, object]:
        seen.append(session)
        return {"content": f"{name}:ok"}

    monkeypatch.setattr(tool_bridge, "execute_agent_tool", fake_execute)
    tool_bridge._sessions.pop("generic-1", None)
    try:
        first = tool_bridge.handle(
            {
                "id": "i1",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA Anthropic"},
                "session_id": "generic-1",
            }
        )
        second = tool_bridge.handle(
            {
                "id": "i2",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA exposure"},
                "session_id": "generic-1",
            }
        )
        assert first["result"] == {"content": "search_web:ok"}
        assert second["result"] == {"content": "search_web:ok"}
        assert len(seen) == 2
        assert seen[0] is seen[1]

        ended = tool_bridge.handle({"id": "e1", "op": "tool.session.end", "session_id": "generic-1"})
        assert ended == {"id": "e1", "result": {"ended": True}}

        tool_bridge.handle(
            {
                "id": "i3",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA"},
                "session_id": "generic-1",
            }
        )
        assert seen[2] is not seen[0]
    finally:
        tool_bridge._sessions.pop("generic-1", None)


def test_invoke_requires_explicit_session() -> None:
    assert tool_bridge.handle({"id": "bad", "op": "tool.invoke", "name": "search_web", "arguments": {}}) == {
        "id": "bad",
        "error": "missing_arg",
    }


def test_invoke_failure_reports_bridge_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(
        name: str,
        arguments: dict[str, object],
        session: AgentToolSession,
        **kwargs: object,
    ) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(tool_bridge, "execute_agent_tool", _boom)
    tool_bridge._sessions.pop("boom-1", None)
    try:
        assert tool_bridge.handle(
            {
                "id": "f1",
                "op": "tool.invoke",
                "name": "search_web",
                "arguments": {"query": "NVDA"},
                "session_id": "boom-1",
            }
        ) == {"id": "f1", "error": "bridge_failed"}
    finally:
        tool_bridge._sessions.pop("boom-1", None)


def test_direct_bridge_search_supports_session_scoped_read_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct bridge search persists rows; its session reads them after restart."""
    from app import tools as _tools
    from app.policy import Capability
    from app.policy import RequestContext as _Ctx
    from app.research.repository import ResearchRepository
    from app.sec.models import SearchCoverage, SECSearchRequest, SECSearchResult, SECTextHit

    hits = tuple(
        SECTextHit(
            search_id="sb-1",
            attempt_id="sb-1-efts-1",
            query="Acme Labs",
            accession_no=f"0000000001-26-00000{i}",
            form="10-K",
            filed_at="2026-01-01",
            filer_cik=1234567,
            filer_name="Acme Labs Inc",
            matched_document=f"0000000001-26-00000{i}.htm",
            file_type="10-K",
            score=float(10 - i),
            snippet="supply agreement",
        )
        for i in range(1, 4)
    )
    result = SECSearchResult(
        search_id="sb-1",
        request=SECSearchRequest(query="Acme Labs"),
        text_hits=hits,
        coverage=SearchCoverage(status="partial"),
        attempts=(),
        retrieval_order=(),
    )

    class _FakeService:
        def __init__(self, data_root: Path | None = None) -> None:
            pass

        def search(self, request: SECSearchRequest) -> SECSearchResult:
            del request
            return result

    monkeypatch.setattr(_tools.sec, "SECDiscoveryService", _FakeService)
    tool_bridge._sessions.pop("bridge-search-1", None)
    try:
        response = tool_bridge.handle(
            {
                "id": "b1",
                "op": "tool.invoke",
                "name": "search_sec_filings",
                "arguments": {"query": "Acme Labs", "limit": 2},
                "session_id": "bridge-search-1",
                "data_root": str(tmp_path),
            }
        )
    finally:
        tool_bridge._sessions.pop("bridge-search-1", None)
    assert "result" in response, response
    fresh = ResearchRepository(data_root=tmp_path)
    stored = fresh.get_search("sb-1")
    assert stored["query"] == "Acme Labs"
    with fresh._connect() as conn:
        rows = conn.execute("SELECT session_id FROM sessions").fetchall()
    assert len(rows) == 1
    session_id = str(rows[0]["session_id"])
    context = _Ctx("research", frozenset({Capability.RESEARCH}), data_root=tmp_path)
    page = _tools.execute_tool(
        "research_read_search",
        {"session_id": session_id, "search_id": "sb-1", "offset": 2},
        "test",
        context=context,
    )
    assert page["session_id"] == session_id
    assert page["total"] == 3
    raw_hits = page["hits"]
    assert isinstance(raw_hits, list)
    assert [hit["accession_no"] for hit in raw_hits if isinstance(hit, dict)] == ["0000000001-26-000003"]


def test_bridge_main_preserves_jsonl_envelopes(monkeypatch: pytest.MonkeyPatch) -> None:
    from io import StringIO

    stdin = StringIO('\nnot json\n[]\n{}\n{"id":"x","op":"unknown"}\n{"id":"end","op":"tool.session.end"}\n')
    stdout = StringIO()
    monkeypatch.setattr(tool_bridge.sys, "stdin", stdin)
    monkeypatch.setattr(tool_bridge.sys, "stdout", stdout)
    tool_bridge.main()
    assert [json.loads(line) for line in stdout.getvalue().splitlines()] == [
        {"error": "bad_request"},
        {"error": "bad_request"},
        {"error": "missing_arg"},
        {"id": "x", "error": "unknown_op"},
        {"id": "end", "error": "missing_arg"},
    ]


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), (7, None), (" value ", " value ")])
def test_invoke_optional_strings_keep_transport_values(
    value: object, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, object]] = []

    def execute(
        name: str, arguments: dict[str, object], session: AgentToolSession, **kwargs: object
    ) -> dict[str, object]:
        seen.append(kwargs)
        return {"content": "ok"}

    monkeypatch.setattr(tool_bridge, "execute_agent_tool", execute)
    try:
        response = tool_bridge.handle(
            {
                "id": "optional",
                "op": "tool.invoke",
                "name": "search_web",
                "session_id": "optional-strings",
                "tool_call_id": value,
                "data_root": value,
                "as_of": value,
            }
        )
        assert response == {"id": "optional", "result": {"content": "ok"}}
        assert seen == [{"tool_call_id": expected, "protocol_id": "optional", "data_root": expected, "as_of": expected}]
    finally:
        tool_bridge._sessions.pop("optional-strings", None)
