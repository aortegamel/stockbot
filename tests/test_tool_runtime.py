"""Runtime tool gateway parity: gate chain, budgets, never-raises, outcome."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.tool_runtime as rt
from app.tool_runtime import (
    AgentToolSession,
    RuntimeToolSession,
    execute_agent_tool,
    outcome_from_result,
)


def test_session_is_gateway_subclass() -> None:
    assert issubclass(RuntimeToolSession, RuntimeToolSession)
    assert AgentToolSession is RuntimeToolSession
    s = RuntimeToolSession(session_id="s1")
    assert s.budget.max_tool_calls is None


def test_happy_path_search_tools() -> None:
    out = execute_agent_tool("search_tools", {"query": "revenue probe"}, RuntimeToolSession(session_id="s1"))
    assert "error" not in out, out
    assert isinstance(out.get("content"), str) and out["content"]


def test_permit_blocked_tool() -> None:
    out = execute_agent_tool("definitely_not_a_tool", {}, RuntimeToolSession(session_id="s1"))
    assert "not permitted" in str(out.get("error", ""))


def test_invalid_args() -> None:
    out = execute_agent_tool("call_tool", {"name": "", "arguments": {}}, RuntimeToolSession(session_id="s1"))
    assert "error" in out


def test_budget_exhaustion() -> None:
    session = RuntimeToolSession(session_id="s1")
    session.budget.max_tool_calls = 0
    out = execute_agent_tool("search_tools", {"query": "probe"}, session)
    assert out.get("error_type") == "run_budget_exceeded"


def test_never_raises_on_internal_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(rt, "_run_pre_gates", _boom)
    out = execute_agent_tool("search_tools", {"query": "probe"}, RuntimeToolSession(session_id="s1"))
    assert "search_tools" in str(out.get("error", ""))
    assert "boom" in str(out.get("error", ""))


def test_outcome_success() -> None:
    result = {
        "content": "x",
        "source_handle": {"a": 1},
        "accession_no": "0001",
        "url": "https://sec.gov/b",
        "known_at": "2025-02-01",
    }
    oc = outcome_from_result("search_sec_filings", dict(result))
    assert oc.tool_name == "search_sec_filings"
    assert oc.content
    assert oc.error is None and oc.error_type is None
    assert oc.retryable is False
    assert oc.source_handle == {"a": 1}
    assert oc.source_refs is not None and oc.source_refs.get("record_id") == "0001"
    assert oc.meta.row_count >= 0


def test_outcome_retryable_only_transient() -> None:
    assert outcome_from_result("t", {"error": "x", "error_type": "tool_error"}).retryable is True
    assert outcome_from_result("t", {"error": "x", "error_type": "intent_denied"}).retryable is False
    assert outcome_from_result("t", {"error": "x", "error_type": "run_budget_exceeded"}).retryable is False


def test_accepted_result_without_capture_scope_keeps_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _web(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        row = {"title": "T", "url": "https://example.com/a", "highlight": "quoted passage"}
        return {"query": arguments.get("query"), "evidence": [row], "source": "exa"}

    monkeypatch.setattr(rt, "execute_tool", _web)
    out = execute_agent_tool(
        "search_web", {"query": "NVDA demand"}, RuntimeToolSession(session_id="s1"), data_root=tmp_path
    )
    assert set(out) == {"content", "meta"}, out
    meta = out["meta"]
    assert isinstance(meta, dict) and "sources" not in meta
    assert not (tmp_path / "raw").exists()
