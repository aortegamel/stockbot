"""Runtime tool gateway parity: gate chain, budgets, never-raises, outcome."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

import app.tool_runtime as rt
from app.tool_runtime import (
    AgentToolSession,
    RuntimeToolSession,
    execute_agent_tool,
    outcome_from_result,
)

if TYPE_CHECKING:
    from app.research.repository import ResearchRepository


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


# -- outcome_from_result classification --------------------------------------


def test_outcome_source_handle_prefers_direct_then_envelope() -> None:
    nested = {"basis": "raw", "text_hash": "b"}
    direct_result: dict[str, object] = {"source_handle": {"a": 1}, "meta": {"source_handle": nested}}
    assert outcome_from_result("t", direct_result).source_handle == {"a": 1}
    probes: tuple[object, ...] = ("x", {}, None, [1])
    for direct in probes:
        result: dict[str, object] = {"meta": {"source_handle": nested}}
        if direct is not None:
            result["source_handle"] = direct
        assert outcome_from_result("t", result).source_handle == nested
    assert outcome_from_result("t", {"meta": {"source_handle": {}}}).source_handle is None
    assert outcome_from_result("t", {"meta": {"source_handle": "x"}}).source_handle is None
    assert outcome_from_result("t", {"meta": 7}).source_handle is None
    assert outcome_from_result("t", {}).source_handle is None


def test_outcome_source_handle_is_a_copy() -> None:
    handle = {"a": 1}
    outcome = outcome_from_result("t", {"source_handle": handle})
    assert outcome.source_handle == handle
    assert outcome.source_handle is not handle


def test_outcome_error_and_retryable_classification() -> None:
    ok = outcome_from_result("t", {})
    assert ok.error is None and ok.error_type is None and ok.retryable is False
    retry = outcome_from_result("t", {"error": "x", "error_type": "tool_error"})
    assert retry.error == "x" and retry.error_type == "tool_error" and retry.retryable is True
    typed = outcome_from_result("t", {"error": "x", "error_type": "intent_denied"})
    assert typed.error_type == "intent_denied" and typed.retryable is False
    untyped = outcome_from_result("t", {"error": "x", "error_type": 5})
    assert untyped.error == "x" and untyped.error_type is None and untyped.retryable is False
    # A falsy non-None error keeps the classification the scheduler sees.
    falsy = outcome_from_result("t", {"error": 0, "error_type": "tool_error"})
    assert falsy.error == "0" and falsy.retryable is True
    assert outcome_from_result("t", {"error": "x", "error_type": True}).error_type is None


def test_outcome_tool_result_id_requires_string() -> None:
    assert outcome_from_result("t", {"tool_result_id": "s:tr:1"}).tool_result_id == "s:tr:1"
    assert outcome_from_result("t", {"tool_result_id": 5}).tool_result_id is None
    assert outcome_from_result("t", {}).tool_result_id is None


_OUTCOME_KEYS = ("content", "error", "error_type", "tool_result_id", "source_handle", "meta")
_OUTCOME_VALUES = st.one_of(
    st.none(),
    st.text(max_size=12),
    st.integers(),
    st.booleans(),
    st.lists(st.text(max_size=4), max_size=2),
    st.dictionaries(
        st.sampled_from(["source_handle", "basis", "row_count", "accession_no", "url"]),
        st.one_of(st.none(), st.text(max_size=6), st.integers()),
        max_size=3,
    ),
)


@seed(20261014)
@settings(max_examples=120, derandomize=True, deadline=None)
@given(st.dictionaries(st.sampled_from(_OUTCOME_KEYS), _OUTCOME_VALUES, max_size=6))
def test_prop_outcome_classification_matches_result_shape(result: dict[str, object]) -> None:
    """Every result shape classifies as its own fields say: no invented error, handle, or id."""
    outcome = outcome_from_result("t", dict(result))
    raw_error = result.get("error")
    assert (outcome.error is None) is (raw_error is None)
    assert outcome.error == (str(raw_error) if raw_error is not None else None)
    raw_type = result.get("error_type")
    assert outcome.error_type == (raw_type if isinstance(raw_type, str) else None)
    assert outcome.retryable is (outcome.error is not None and outcome.error_type == "tool_error")
    raw_id = result.get("tool_result_id")
    assert outcome.tool_result_id == (raw_id if isinstance(raw_id, str) else None)
    direct = result.get("source_handle")
    meta = result.get("meta")
    nested = meta.get("source_handle") if isinstance(meta, dict) else None
    expected = direct if isinstance(direct, dict) and direct else nested
    assert outcome.source_handle == (dict(expected) if isinstance(expected, dict) and expected else None)


# -- staged persistence ------------------------------------------------------


def _staged_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ResearchRepository, str, str]:
    from app.research import service
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "research.sqlite"))
    repo = ResearchRepository()
    sid = service.create_research("NVDA demand?", "o", as_of="2025-06-30T00:00:00+00:00", repo=repo)
    return repo, sid, repo.list_jobs(sid)[0].job_id


def test_persistable_staged_result_requires_replayable_tool_and_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.tool_runtime import _StagedContext

    repo, sid, jid = _staged_session(tmp_path, monkeypatch)
    assert rt._can_persist_staged_result("search_web", _StagedContext(sid, jid, False, None)) is False
    assert rt._can_persist_staged_result("search_tools", _StagedContext(sid, jid, False, repo)) is False
    assert rt._can_persist_staged_result("search_web", _StagedContext(None, jid, False, repo)) is False
    assert rt._can_persist_staged_result("search_web", _StagedContext("", jid, False, repo)) is False
    assert rt._can_persist_staged_result("search_web", _StagedContext(sid, None, False, repo)) is False
    assert rt._can_persist_staged_result("search_web", _StagedContext(sid, "", False, repo)) is False
    assert rt._can_persist_staged_result("search_web", _StagedContext(sid, jid, False, repo)) is True
    assert (
        rt._persist_staged_tool_result("search_web", {"content": "x"}, _StagedContext(sid, jid, False, None), "tc-0")
        is None
    )
    assert (
        rt._persist_staged_tool_result("search_tools", {"content": "x"}, _StagedContext(sid, jid, False, repo), "tc-0")
        is None
    )


def test_persist_staged_result_records_id_and_absorbs_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service
    from app.tool_runtime import _StagedContext

    repo, sid, jid = _staged_session(tmp_path, monkeypatch)
    staged = _StagedContext(sid, jid, False, repo)
    calls: list[tuple[object, ...]] = []

    def _persist(
        session_id: str,
        job_id: str,
        tool_name: str,
        tool_result_id: str | None,
        result: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        calls.append((session_id, job_id, tool_name, tool_result_id, result))
        return {"tool_result_id": f"{session_id}:tr:fixed"}

    monkeypatch.setattr(service, "persist_tool_result", _persist)
    stored: dict[str, object] = {"content": "x"}
    assert rt._persist_staged_tool_result("search_web", stored, staged, "tc-1") == f"{sid}:tr:fixed"
    assert stored["tool_result_id"] == f"{sid}:tr:fixed"
    # The persisted argument is the caller's dict, and the stored id lands in it after the call.
    assert calls == [(sid, jid, "search_web", f"{sid}:tr:tc-1", {"content": "x", "tool_result_id": f"{sid}:tr:fixed"})]

    def _empty_persist(*args: object, **kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(service, "persist_tool_result", _empty_persist)
    unlabeled: dict[str, object] = {"content": "x"}
    assert rt._persist_staged_tool_result("search_web", unlabeled, staged, "") is None
    assert "tool_result_id" not in unlabeled

    def _boom(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("store down")

    monkeypatch.setattr(service, "persist_tool_result", _boom)
    assert rt._persist_staged_tool_result("search_web", {"content": "x"}, staged, "tc-2") is None


# -- gateway result handling -------------------------------------------------


def _web_ok(name: str, arguments: dict[str, object], model: str, context: object = None) -> dict[str, object]:
    return {"query": arguments.get("query"), "evidence": [], "source": "exa"}


def _handler_returning(value: object):
    def _handler(name: str, arguments: dict[str, object], model: str, context: object = None) -> object:
        return value

    return _handler


def _ignore(*args: object, **kwargs: object) -> dict[str, object]:
    return {}


def _fixed_persist(tool_result_id: str):
    def _persist(*args: object, **kwargs: object) -> dict[str, object]:
        return {"tool_result_id": tool_result_id}

    return _persist


def _fail_if_called(*args: object, **kwargs: object) -> object:
    pytest.fail("the handler must not run for a denied call")


def test_success_response_carries_the_persisted_tool_result_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.research import service

    _repo, sid, jid = _staged_session(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "persist_tool_result", _fixed_persist("s:tr:gateway"))
    monkeypatch.setattr(service, "authorize_and_consume_dispatch", _ignore)
    monkeypatch.setattr(rt, "execute_tool", _web_ok)

    session = RuntimeToolSession(session_id="gw-persist-id")
    session.active_research_session_id = sid
    session.active_research_job_id = jid
    out = execute_agent_tool("search_web", {"query": "NVDA demand"}, session)
    assert set(out) == {"content", "meta", "tool_result_id"}, out
    assert out["tool_result_id"] == "s:tr:gateway"


def test_ingress_block_withholds_content_even_when_an_id_was_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.research import service

    _repo, sid, jid = _staged_session(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "persist_tool_result", _fixed_persist("s:tr:blocked"))
    monkeypatch.setattr(service, "authorize_and_consume_dispatch", _ignore)
    credential = "Authorization: Bearer " + "a" * 32
    monkeypatch.setattr(rt, "execute_tool", _handler_returning({"text": credential + " " + "context " * 60}))

    session = RuntimeToolSession(session_id="gw-ingress")
    session.active_research_session_id = sid
    session.active_research_job_id = jid
    out = execute_agent_tool("search_web", {"query": "NVDA demand"}, session)
    assert out["error_type"] == "ingress_blocked" and out["soft"] is True, out
    assert "content" not in out and "tool_result_id" not in out
    assert session.run_security.quarantined_items == 1


def test_handler_error_shapes_are_reported_without_the_success_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _handler_returning({"error": "upstream failed"}))
    failed = execute_agent_tool("search_tools", {"query": "probe"}, RuntimeToolSession(session_id="gw-failed"))
    assert set(failed) == {"error", "error_type"}
    assert failed["error_type"] == "tool_error"
    assert "upstream failed" in str(failed["error"])

    monkeypatch.setattr(rt, "execute_tool", _handler_returning({"error": "denied by policy", "soft": True}))
    soft = execute_agent_tool("search_tools", {"query": "probe"}, RuntimeToolSession(session_id="gw-soft"))
    assert soft == {"error": "denied by policy", "soft": True}

    monkeypatch.setattr(rt, "execute_tool", _handler_returning("not a mapping"))
    shaped = execute_agent_tool("search_tools", {"query": "probe"}, RuntimeToolSession(session_id="gw-nondict"))
    assert shaped["error_type"] == "tool_error"
    assert "empty tool result" in str(shaped["error"])


def test_evidence_budget_exhaustion_withholds_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _handler_returning({"content": "x" * 40}))
    session = RuntimeToolSession(session_id="gw-evidence")
    session.budget.max_evidence_tokens = 0
    out = execute_agent_tool("search_tools", {"query": "revenue probe"}, session)
    assert out == {
        "error": "Runtime evidence-token budget exceeded (used 0/0 tokens); narrow the question or window.",
        "error_type": "evidence_budget_exceeded",
    }


# -- gate decisions and budgets ----------------------------------------------


def test_run_budget_refusal_names_search_tool_and_runtime_exhaustion() -> None:
    search = RuntimeToolSession(session_id="gw-search-budget")
    search.budget.max_search_calls = 0
    out = execute_agent_tool("search_web", {"query": "NVDA demand"}, search)
    assert out == {
        "error": (
            "Runtime run search budget exceeded (used 0/0 search calls); "
            "retry with a narrower question or fewer tool calls."
        ),
        "error_type": "run_budget_exceeded",
    }
    tool = RuntimeToolSession(session_id="gw-tool-budget")
    tool.budget.max_tool_calls = 0
    out2 = execute_agent_tool("search_tools", {"query": "probe"}, tool)
    assert out2["error_type"] == "run_budget_exceeded"
    assert "tool budget exceeded (used 0/0 tool calls)" in str(out2["error"])
    exhausted = RuntimeToolSession(session_id="gw-runtime-budget")
    exhausted.budget.max_runtime = 0
    out3 = execute_agent_tool("search_web", {"query": "NVDA demand"}, exhausted)
    assert out3["error_type"] == "deadline_exceeded"
    assert str(out3["error"]).startswith("Runtime run runtime budget exceeded (remaining 0.0s)")


def test_dispatch_budget_requires_both_staged_ids_then_consumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.research import service
    from app.tool_runtime import _StagedContext

    repo, sid, jid = _staged_session(tmp_path, monkeypatch)
    assert rt._consume_dispatch_budget("search_web", _StagedContext(sid, jid, False, None), {}) == (False, None)
    assert rt._consume_dispatch_budget("search_tools", _StagedContext(sid, jid, False, repo), {}) == (False, None)
    assert rt._consume_dispatch_budget("search_web", _StagedContext(sid, None, False, repo), {}) == (
        False,
        {"error": "Active research job is required for tool 'search_web'", "error_type": "invalid_research_context"},
    )
    assert rt._consume_dispatch_budget("search_web", _StagedContext(None, jid, False, repo), {}) == (
        False,
        {
            "error": "Active research session is required for tool 'search_web'",
            "error_type": "invalid_research_context",
        },
    )

    def _key_error(*args: object, **kwargs: object) -> dict[str, object]:
        raise KeyError("no such job")

    monkeypatch.setattr(service, "authorize_and_consume_dispatch", _key_error)
    assert rt._consume_dispatch_budget("search_web", _StagedContext(sid, jid, False, repo), {}) == (
        False,
        {"error": "'no such job'", "error_type": "invalid_research_context"},
    )

    def _value_error(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("tool_budget exhausted")

    monkeypatch.setattr(service, "authorize_and_consume_dispatch", _value_error)
    assert rt._consume_dispatch_budget("search_web", _StagedContext(sid, jid, False, repo), {}) == (
        False,
        {"error": "tool_budget exhausted"},
    )

    monkeypatch.setattr(service, "authorize_and_consume_dispatch", _ignore)
    assert rt._consume_dispatch_budget("search_web", _StagedContext(sid, jid, False, repo), {"query": "x"}) == (
        True,
        None,
    )


def test_permit_and_schema_gate_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _fail_if_called)
    invalid = execute_agent_tool("list_sec_filings", {}, RuntimeToolSession(session_id="gw-schema"))
    assert invalid["error_type"] == "invalid_tool_arguments"
    assert invalid["tool"] == "list_sec_filings" and invalid["required"] == ["identifier"]
    assert "Missing required argument(s)" in str(invalid["error"])

    oversized = execute_agent_tool("search_tools", {"query": "x" * 9000}, RuntimeToolSession(session_id="gw-bytes"))
    assert oversized["error_type"] == "invalid_tool_arguments"
    assert str(oversized["error"]) == "Tool arguments exceed the maximum size (8192 bytes): search_tools"


def test_intent_denial_through_gateway_is_soft_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    session = RuntimeToolSession(session_id="gw-intent")
    session.run_security.original_intent = replace(session.run_security.original_intent, permitted_domains=frozenset())
    monkeypatch.setattr(rt, "execute_tool", _fail_if_called)
    out = execute_agent_tool("search_web", {"query": "NVDA demand"}, session)
    assert out == {"error": "Tool call exceeds original user intent", "error_type": "intent_denied", "soft": True}
    assert session.run_security.security_events == [
        {"source": "search_web", "decision": "action_blocked", "reason": "tool call exceeds original user intent"}
    ]


def test_portfolio_gate_denial_shape_is_recorded() -> None:
    session = RuntimeToolSession(session_id="gw-portfolio")
    session.run_security.original_intent = replace(session.run_security.original_intent, permitted_domains=frozenset())
    out = rt._check_intent_and_egress("get_portfolio_snapshot", {}, session, "{}")
    assert out == {
        "error": "Portfolio access is not authorized for this session",
        "error_type": "authorization_denied",
        "soft": True,
    }
    assert session.run_security.security_events == [
        {
            "source": "get_portfolio_snapshot",
            "decision": "action_blocked",
            "reason": "portfolio access is not authorized for this session",
        }
    ]


def test_private_arguments_block_local_tool_before_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _fail_if_called)
    session = RuntimeToolSession(session_id="gw-private")
    out = execute_agent_tool(
        "get_fundamentals",
        {"ticker": "AAPL", "metric": "overview", "note": "my holdings 123456"},
        session,
    )
    assert out == {
        "error": "Tool arguments contain private data that must not be transmitted",
        "error_type": "private_args_denied",
        "soft": True,
    }
    assert session.run_security.security_events == [
        {"source": "get_fundamentals", "decision": "action_blocked", "reason": "PRIVATE -> EXTERNAL egress"}
    ]


def test_search_egress_denial_is_soft_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "execute_tool", _fail_if_called)
    session = RuntimeToolSession(session_id="gw-egress")
    out = execute_agent_tool("search_web", {"query": "my holdings 123456"}, session)
    assert out == {
        "error": "Egress blocked: private data must not leave Stockbot",
        "error_type": "egress_denied",
        "soft": True,
    }
    assert session.run_security.security_events == [
        {"source": "search_web", "decision": "egress_blocked", "reason": "PRIVATE -> EXTERNAL egress"}
    ]


# -- success meta shaping ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("set_chart_symbol", {"ticker": "  aapl \t"}),
        ("call_tool", {"name": "set_chart_symbol", "arguments": {"ticker": "  aapl \t"}}),
    ],
)
def test_execute_chart_symbol_returns_normalized_selection_meta(
    name: str, arguments: dict[str, object], tmp_path: Path
) -> None:
    out = execute_agent_tool(name, arguments, RuntimeToolSession(session_id="gw-chart"), data_root=tmp_path)

    assert set(out) == {"content", "meta"}, out
    assert isinstance(out["content"], str) and out["content"]
    meta = out["meta"]
    assert isinstance(meta, dict)
    assert meta["status"] == "completed"
    assert meta["chart_selection"] == {"result_type": "chart_selection", "ticker": "AAPL"}


def test_success_meta_copies_only_normalized_chart_selection_fields() -> None:
    from app.security.context_gateway import envelope_for_tool

    result: dict[str, object] = {
        "result_type": "chart_selection",
        "ticker": "  brk.b \t",
        "unknown_field": "dropped",
        "chart_selection": {"result_type": "chart_selection", "ticker": "MSFT"},
    }
    envelope = envelope_for_tool("set_chart_symbol", result)
    out = rt._success_meta("set_chart_symbol", result, rt._tool_result_meta(result), envelope, "completed")

    assert out["chart_selection"] == {"result_type": "chart_selection", "ticker": "BRK.B"}
    assert "unknown_field" not in out


@pytest.mark.parametrize(
    ("name", "status", "result"),
    [
        pytest.param(
            "get_current_time", "completed", {"result_type": "chart_selection", "ticker": "AAPL"}, id="other-tool"
        ),
        pytest.param("set_chart_symbol", "failed", {"result_type": "chart_selection", "ticker": "AAPL"}, id="failed"),
        pytest.param("set_chart_symbol", "partial", {"result_type": "chart_selection", "ticker": "AAPL"}, id="partial"),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "other_result", "ticker": "AAPL"}, id="wrong-result-type"
        ),
        pytest.param("set_chart_symbol", "completed", {"ticker": "AAPL"}, id="missing-result-type"),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": ""}, id="empty-ticker"
        ),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": " \t"}, id="blank-ticker"
        ),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": None}, id="null-ticker"
        ),
        pytest.param("set_chart_symbol", "completed", {"result_type": "chart_selection"}, id="missing-ticker"),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": 7}, id="numeric-ticker"
        ),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": True}, id="boolean-ticker"
        ),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": []}, id="list-ticker"
        ),
        pytest.param(
            "set_chart_symbol", "completed", {"result_type": "chart_selection", "ticker": "AAPL!"}, id="invalid-ticker"
        ),
        pytest.param(
            "set_chart_symbol",
            "completed",
            {"result_type": "chart_selection", "ticker": " sec "},
            id="organization-word",
        ),
        pytest.param(
            "set_chart_symbol",
            "completed",
            {"result_type": "chart_selection", "ticker": "AAPL", "error": None},
            id="error-key",
        ),
        pytest.param(
            "set_chart_symbol",
            "completed",
            {"result_type": "chart_selection", "ticker": "AAPL", "error_type": ""},
            id="error-type-key",
        ),
    ],
)
def test_success_meta_withholds_invalid_chart_selection(name: str, status: str, result: dict[str, object]) -> None:
    from app.security.context_gateway import envelope_for_tool

    envelope = envelope_for_tool(name, result)
    out = rt._success_meta(name, result, rt._tool_result_meta(result), envelope, status)

    assert "chart_selection" not in out


def test_execute_watchlist_returns_normalized_receipt_meta(tmp_path: Path) -> None:
    out = execute_agent_tool(
        "update_watchlist",
        {"action": "add", "ticker": " amd "},
        RuntimeToolSession(session_id="gw-watch"),
        data_root=tmp_path,
    )

    meta = out["meta"]
    assert isinstance(meta, dict) and meta["status"] == "completed"
    assert meta["watchlist_update"] == {"result_type": "watchlist_update", "action": "add", "ticker": "AMD"}
    assert "chart_selection" not in meta


@pytest.mark.parametrize(
    ("name", "status", "result"),
    [
        pytest.param("get_current_time", "completed", {"result_type": "watchlist_update"}, id="other-tool"),
        pytest.param("update_watchlist", "failed", {}, id="failed"),
        pytest.param("update_watchlist", "completed", {"result_type": "chart_selection"}, id="wrong-result-type"),
        pytest.param(
            "update_watchlist", "completed", {"result_type": "watchlist_update", "action": "clear"}, id="action"
        ),
        pytest.param(
            "update_watchlist", "completed", {"result_type": "watchlist_update", "ticker": "AMD!"}, id="ticker"
        ),
        pytest.param("update_watchlist", "completed", {"result_type": "watchlist_update", "ticker": ""}, id="blank"),
        pytest.param("update_watchlist", "completed", {"result_type": "watchlist_update", "error": None}, id="error"),
    ],
)
def test_success_meta_withholds_invalid_watchlist_receipt(name: str, status: str, result: dict[str, object]) -> None:
    from app.security.context_gateway import envelope_for_tool

    receipt: dict[str, object] = {"result_type": "watchlist_update", "action": "add", "ticker": "AMD", **result}
    out = rt._success_meta(name, receipt, rt._tool_result_meta(receipt), envelope_for_tool(name, receipt), status)

    assert "watchlist_update" not in out


def test_execute_add_widget_returns_normalized_receipt_meta(tmp_path: Path) -> None:
    out = execute_agent_tool(
        "add_widget",
        {"type": "insider", "ticker": " nvda "},
        RuntimeToolSession(session_id="gw-widget"),
        data_root=tmp_path,
    )

    meta = out["meta"]
    assert isinstance(meta, dict) and meta["status"] == "completed"
    assert meta["widget_selection"] == {"result_type": "widget_selection", "widget_type": "insider", "ticker": "NVDA"}
    assert "watchlist_update" not in meta


@pytest.mark.parametrize(
    ("name", "status", "result"),
    [
        pytest.param("update_watchlist", "completed", {}, id="other-tool"),
        pytest.param("add_widget", "failed", {}, id="failed"),
        pytest.param("add_widget", "completed", {"result_type": "watchlist_update"}, id="wrong-result-type"),
        pytest.param("add_widget", "completed", {"widget_type": "heatmap"}, id="widget-type"),
        pytest.param("add_widget", "completed", {"ticker": "NVDA!"}, id="ticker"),
        pytest.param("add_widget", "completed", {"ticker": ""}, id="blank"),
        pytest.param("add_widget", "completed", {"error": None}, id="error"),
        pytest.param("add_widget", "completed", {"error_type": "x"}, id="error-type"),
    ],
)
def test_success_meta_withholds_invalid_widget_receipt(name: str, status: str, result: dict[str, object]) -> None:
    from app.security.context_gateway import envelope_for_tool

    receipt: dict[str, object] = {"result_type": "widget_selection", "widget_type": "news", "ticker": "NVDA", **result}
    out = rt._success_meta(name, receipt, rt._tool_result_meta(receipt), envelope_for_tool(name, receipt), status)

    assert "widget_selection" not in out


def test_success_meta_copies_discovery_names_and_paging_fields_only() -> None:
    from app.security.context_gateway import envelope_for_tool

    result: dict[str, object] = {
        "tools": [{"name": "get_sec_filing"}, {"name": 7}, "ignored"],
        "matches": [{"name": "get_xbrl_facts"}],
        "continuation": True,
        "total": 3,
        "unknown_field": "dropped",
    }
    envelope = envelope_for_tool("browse_tools", result)
    out = rt._success_meta("browse_tools", result, rt._tool_result_meta(result), envelope, "completed")
    assert out["tools"] == ["get_sec_filing"] and out["matches"] == ["get_xbrl_facts"]
    assert out["continuation"] is True and out["total"] == 3
    assert out["status"] == "completed" and out["source"] == envelope.source
    assert out["sensitivity"] == envelope.sensitivity.value and out["integrity"] == envelope.integrity.value
    assert "unknown_field" not in out


def test_success_meta_tolerates_a_non_mapping_result() -> None:
    from app.security.context_gateway import envelope_for_tool

    out = rt._success_meta(
        "get_sec_filing",
        "not a mapping",
        rt._tool_result_meta(None),
        envelope_for_tool("get_sec_filing", {}),
        "completed",
    )
    assert "source_refs" not in out and "source_handle" not in out and "matches" not in out


_NAVIGATION_SAMPLE_KEYS = ("cursor", "continuation", "offset", "total", "more")
_UNKNOWN_KEY_PREFIX = "zz_unknown_"


@seed(20261014)
@settings(max_examples=60, derandomize=True)
@given(
    st.dictionaries(
        st.one_of(
            st.sampled_from(_NAVIGATION_SAMPLE_KEYS),
            st.text(max_size=8).map(lambda suffix: _UNKNOWN_KEY_PREFIX + suffix),
        ),
        st.one_of(st.none(), st.integers(), st.text(max_size=6), st.booleans()),
        max_size=8,
    )
)
def test_prop_navigation_meta_copies_known_values_and_never_invents_keys(extra: dict[str, object]) -> None:
    """Paging fields copy through unchanged; unknown keys never reach the success envelope."""
    safe_meta: dict[str, object] = {}
    rt._copy_navigation_meta(extra, safe_meta)
    for key in _NAVIGATION_SAMPLE_KEYS:
        if key in extra:
            assert safe_meta.get(key) == extra[key]
    assert not [key for key in safe_meta if key.startswith(_UNKNOWN_KEY_PREFIX)]


def test_copy_navigation_meta_ignores_non_mapping_results() -> None:
    safe_meta: dict[str, object] = {}
    rt._copy_navigation_meta(["continuation"], safe_meta)
    rt._copy_navigation_meta(None, safe_meta)
    assert safe_meta == {}
