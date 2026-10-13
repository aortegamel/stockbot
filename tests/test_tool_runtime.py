"""Runtime tool gateway parity: gate chain, budgets, never-raises, outcome."""

from __future__ import annotations

from copy import deepcopy
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
    nested = outcome_from_result(
        "get_sec_filing",
        {"content": "x", "meta": {"source_refs": {"record_id": "0001", "known_at": "2025-02-01"}}},
    )
    assert nested.source_refs is not None and nested.source_refs.get("record_id") == "0001"


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


def test_outcome_nested_source_refs_keep_primary_and_breadth_first_order() -> None:
    primary = {
        "record_id": "0000320193-25-000079",
        "uri": "https://www.sec.gov/Archives/filing.htm",
        "known_at": "2025-02-01T18:00:00Z",
    }
    result: dict[str, object] = {
        "accession_no": primary["record_id"],
        "url": primary["uri"],
        "known_at": primary["known_at"],
        "date": "2024",
        "shallow": {"filed_at": "2025-01-31"},
        "deep": {"level": {"record_id": "nested-record", "known_at": "2025-01-30"}},
        "other": [["not a reference", {"published_at": "2025-01-29"}]],
        "duplicate": {
            "accession_no": primary["record_id"],
            "url": primary["uri"],
            "known_at": primary["known_at"],
        },
    }

    outcome = outcome_from_result("get_sec_filing", result)

    assert outcome.source_refs == {
        **primary,
        "all": [
            primary,
            {"known_at": "2025-01-31"},
            {"record_id": "nested-record", "known_at": "2025-01-30"},
            {"known_at": "2025-01-29"},
        ],
    }


@pytest.mark.parametrize("date", ["2020", "2020-05", "2020-05-31"])
def test_outcome_observation_dates_do_not_become_known_at(date: str) -> None:
    result: dict[str, object] = {"series": [{"observations": [{"date": date, "value": 42}]}]}

    assert outcome_from_result("get_macro_context", result).source_refs is None
    series = result["series"]
    assert isinstance(series, list)
    first = series[0]
    assert isinstance(first, dict)
    observations = first["observations"]
    assert isinstance(observations, list)
    observation = observations[0]
    assert isinstance(observation, dict)
    assert observation["date"] == date


@pytest.mark.parametrize("date", ["2020", "2020-05", "2020-05-31"])
def test_execution_macro_nested_known_at_preserves_original_observation_dates(
    date: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result: dict[str, object] = {
        "source": "datacommons",
        "known_at": "2026-09-01T12:00:00Z",
        "series": [
            {
                "geo": "country/USA",
                "variable": "Count_Person",
                "facet": "census",
                "unit": "Count",
                "provider": "Census",
                "provenanceUrl": "https://datacommons.org/browser/census",
                "known_at": "2026-09-01T12:00:00Z",
                "observations": [
                    {
                        "date": date,
                        "value": 42,
                        "provenance": {"releases": [[{"known_at": "2026-08-31T10:00:00Z"}]]},
                    }
                ],
            }
        ],
        "continuation": True,
    }
    original = deepcopy(result)

    def _macro(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
        assert name == "get_macro_context"
        assert arguments == {"geos": ["country/USA"], "variables": ["Count_Person"]}
        return result

    monkeypatch.setattr(rt, "execute_tool", _macro)
    out = execute_agent_tool(
        "get_macro_context",
        {"geos": ["country/USA"], "variables": ["Count_Person"]},
        RuntimeToolSession(session_id="macro-dates"),
        data_root=tmp_path,
    )

    assert set(out) == {"content", "meta"}, out
    meta = out["meta"]
    assert isinstance(meta, dict)
    assert meta["source_refs"] == {
        "known_at": "2026-09-01T12:00:00Z",
        "all": [
            {"known_at": "2026-09-01T12:00:00Z"},
            {"known_at": "2026-08-31T10:00:00Z"},
        ],
    }
    assert meta["continuation"] is True
    content = out["content"]
    assert isinstance(content, str)
    assert f"  - date: {date}, value: 42, unit: Count" in content
    for field in (
        "geo: country/USA",
        "variable: Count_Person",
        "facet: census",
        "unit: Count",
        "provider: Census",
        "provenanceUrl: https://datacommons.org/browser/census",
    ):
        assert field in content
    assert "continuation: True" in content
    assert result == original
