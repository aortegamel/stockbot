"""Fail-closed authority guards: registry abort + JEV-outage objective-only.

Pinned to HEAD ``_jev_admit(sid, objective, proposals, jev=None)``: outage is
simulated by passing a failing ``jev`` directly. If further injection params
land, extend — do not replace — these tests.
"""

from unittest import mock

import pytest

from app.research import kernel_worker as kw


def test_registry_failure_raises_not_empty() -> None:
    with mock.patch("app.research.scheduler.build_registry", side_effect=RuntimeError("boom")):
        try:
            kw._registry_portfolio_hit()
        except RuntimeError as exc:
            assert "registry guard forbids" in str(exc)
        else:
            raise AssertionError("registry failure must raise")


def _props() -> list[dict[str, object]]:
    return [
        {"id": "s-q1", "objectiveId": "s", "question": "Other angle?", "dependsOn": [], "whyItMatters": "w"},
        {"id": "s-q2", "objectiveId": "s", "question": "Exact user objective?", "dependsOn": [], "whyItMatters": "w"},
    ]


class _JevDown:
    def decide(self, *a: object, **k: object) -> object:
        raise RuntimeError("jev down")


def _run_outage(fn: object, *args: object) -> object:
    # Inject the failing client explicitly: _shared_jev() caches process-wide,
    # so patching the JevClient constructor is order-dependent (a cached
    # success bypasses the patch) and caching the double leaks into later tests.
    assert callable(fn)
    return fn(*args, jev=_JevDown())  # type: ignore[operator]


def test_jev_outage_returns_objective_only() -> None:
    props = _props()
    out = _run_outage(kw._jev_admit, "s", "Exact user objective?", props)
    assert out == [props[1]]


def test_jev_outage_synthesizes_single_objective_node() -> None:
    out = _run_outage(kw._jev_admit, "s", "Missing objective?", _props())
    assert isinstance(out, list) and len(out) == 1
    first = out[0]
    assert isinstance(first, dict) and first["question"] == "Missing objective?"


class _JevAdmitTwoOfThree:
    async def decide(self, *a: object, **k: object) -> dict[str, dict[str, str]]:
        return {"s-q1": {"choice": "analyze"}, "s-q2": {"choice": "gather_evidence"}, "s-q3": {"choice": "reject"}}


def test_jev_success_preserves_all_admitted_proposals() -> None:
    props: list[dict[str, object]] = [
        {"id": "s-q1", "objectiveId": "s", "question": "Other angle?", "dependsOn": [], "whyItMatters": "w"},
        {"id": "s-q2", "objectiveId": "s", "question": "Exact user objective?", "dependsOn": [], "whyItMatters": "w"},
        {"id": "s-q3", "objectiveId": "s", "question": "Tangent?", "dependsOn": [], "whyItMatters": "w"},
    ]
    out = kw._jev_admit("s", "Exact user objective?", props, jev=_JevAdmitTwoOfThree())  # type: ignore[arg-type]
    assert out == props[:2]


def test_single_proposal_passthrough() -> None:
    props = _props()[:1]
    assert kw._jev_admit("s", "q", props) == props


def test_graph_prompt_registry_failure_is_terminal() -> None:
    with mock.patch("app.research.scheduler.build_registry", side_effect=RuntimeError("boom")):
        resp = kw._run({"id": "r1", "op": "run", "prompt": "Is XYZ solvent?"})
    assert resp["id"] == "r1"
    terminal = resp.get("terminal")
    assert isinstance(terminal, dict) and "registry guard forbids" in str(terminal.get("message"))


class _JevRoute:
    def __init__(self, choice: str | None = None, fail: bool = False) -> None:
        self.choice = choice
        self.fail = fail
        self.seen: list[object] = []

    async def route_entry(self, prompt: str, registry: object | None = None) -> str:
        if self.fail:
            raise RuntimeError("jev down")
        self.seen.append((prompt, registry))
        return self.choice or "research_required"


def test_route_reasoning_choice_returns_fast_path() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "hello"}, jev=_JevRoute("reasoning_required"))  # type: ignore[arg-type]
    assert out == {"id": "r1", "route": "reasoning_required"}


def test_route_outage_fails_open_to_research() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "hello"}, jev=_JevRoute(fail=True))  # type: ignore[arg-type]
    assert out == {"id": "r1", "route": "research_required"}


def test_route_blank_prompt_needs_no_jev() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "  "}, jev=_JevRoute("reasoning_required"))  # type: ignore[arg-type]
    assert out == {"id": "r1", "route": "research_required"}


def test_route_tool_winner_returns_exact_tool() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "what time is it?"}, jev=_JevRoute("get_current_time"))  # type: ignore[arg-type]
    assert out == {"id": "r1", "route": "get_current_time"}


class _FailGenerate:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.seen: list[object] = []

    def __call__(self, **kwargs: object) -> object:
        self.seen.append(kwargs)
        return self.payload


def test_arguments_uses_shared_needle_and_schema_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _FailGenerate({"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "r"})
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)
    monkeypatch.setattr(
        "app.research.scheduler.build_registry", lambda: [{"name": "query_finra", "parameters": {"type": "object"}}]
    )
    monkeypatch.setattr("app.research.scheduler._schema_for", lambda name, reg: {"type": "object"})
    out = kw._arguments({"id": "a1", "op": "arguments", "tool": "query_finra", "objective": "short interest?"})
    assert out == {
        "id": "a1",
        "tool": "query_finra",
        "arguments": {"ticker": "NVDA"},
        "confidence": 0.9,
        "reasoning": "r",
    }
    seen = gen.seen[0]
    assert isinstance(seen, dict) and seen["schema"] == {"type": "object"}
    assert isinstance(seen.get("objective"), str) and seen["objective"] == "short interest?"


def test_kernel_worker_stamps_route_assess_and_node() -> None:
    # Clock asks short-circuit in programmatic_route, so JEV never sees them;
    # the stamp check uses an unroutable prompt that still reaches JEV.
    jev = _JevRoute("get_current_time")
    out = kw._route({"id": "r1", "op": "route", "prompt": "what time is it?"}, jev=jev)  # type: ignore[arg-type]
    assert out == {"id": "r1", "route": "get_current_time"}
    assert jev.seen == []
    jev2 = _JevRoute("research_required")
    out2 = kw._route({"id": "r2", "op": "route", "prompt": "halp money stuff?"}, jev=jev2)  # type: ignore[arg-type]
    assert out2 == {"id": "r2", "route": "research_required"}
    assert isinstance(jev2.seen[0][0], str) and jev2.seen[0][0].startswith("[Today UTC ")
    assess = _JevAssess("node_resolved")
    kw._assess_entry({"id": "s1", "prompt": "risk?", "tool": "t", "result": {}}, jev=assess)  # type: ignore[arg-type]
    assert isinstance(assess.seen[0][0], str) and assess.seen[0][0].startswith("[Today UTC ")
    prompt = kw._intake_reasoner_prompt("rs:test", "objective?", None, "")
    assert "Today is" in prompt and "UTC" in prompt


def test_arguments_mismatch_is_error_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _FailGenerate({"tool": "other", "arguments": {}}))
    out = kw._arguments({"id": "a2", "op": "arguments", "tool": "query_finra"})
    assert out["id"] == "a2" and "error" in out
    assert "error" in kw._arguments({"id": "a3"})


def test_arguments_withhold_seeds_sec_and_sho(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle withhold seeds SEC identifier / SHO ticker+company; unseedable stays error."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA", "Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("needle tool mismatch: jev selected 'x', needle emitted None")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    sho = kw._arguments(
        {
            "id": "s1",
            "tool": "get_reg_sho_volume",
            "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
        }
    )
    assert sho["arguments"] == {"ticker": "AAPL", "company_name": "Apple"}
    sec = kw._arguments({"id": "s2", "tool": "list_sec_filings", "objective": "What drove NVDA revenue last quarter?"})
    assert sec["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    none = kw._arguments({"id": "s3", "tool": "get_reg_sho_volume", "objective": "Which filings mention Elon Musk?"})
    assert none["id"] == "s3" and "error" in none


def test_arguments_repairs_needle_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle placeholder forms / org ticker repair through the shared scheduler path."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _gen(**kwargs: object) -> object:
        tool = kwargs.get("tool")
        if tool == "list_sec_filings":
            return {
                "tool": tool,
                "arguments": {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]},
                "confidence": 0.8,
                "reasoning": "r",
            }
        return {
            "tool": tool,
            "arguments": {"ticker": "FINRA", "company_name": "Apple"},
            "confidence": 0.8,
            "reasoning": "r",
        }

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)
    sec = kw._arguments(
        {
            "id": "r1",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
            "schema": {"type": "object"},
        }
    )
    assert sec["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    sho = kw._arguments(
        {
            "id": "r2",
            "tool": "get_reg_sho_volume",
            "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
            "schema": {"type": "object"},
        }
    )
    assert sho["arguments"] == {"ticker": "AAPL", "company_name": "Apple"}


class _JevAssess:
    def __init__(self, verdict: str | None = None, fail: bool = False) -> None:
        self.verdict = verdict
        self.fail = fail
        self.seen: list[object] = []

    async def assess_entry_tool(
        self, prompt: str, tool: str, arguments: object = None, result: object = None
    ) -> str | None:
        if self.fail:
            raise RuntimeError("jev down")
        self.seen.append((prompt, tool, arguments, result))
        return self.verdict or "node_resolved"


def test_assess_entry_returns_verdict() -> None:
    jev = _JevAssess("get_sec_document")
    out = kw._assess_entry(
        {"id": "s1", "prompt": "risk?", "tool": "search_sec_filings", "arguments": {}, "result": {"ok": True}},
        jev=jev,  # type: ignore[arg-type]
    )
    assert out == {"id": "s1", "verdict": "get_sec_document"}


def test_assess_entry_outage_and_blank_fail_open_to_research() -> None:
    out = kw._assess_entry({"id": "s2", "prompt": "p", "tool": "t", "result": {}}, jev=_JevAssess(fail=True))  # type: ignore[arg-type]
    assert out == {"id": "s2", "verdict": "research_required"}
    out = kw._assess_entry({"id": "s3", "prompt": "  ", "tool": "t"}, jev=_JevAssess("node_resolved"))  # type: ignore[arg-type]
    assert out == {"id": "s3", "verdict": "research_required"}


def test_entry_prompts_carry_temporal_decoding() -> None:
    import asyncio

    from app.decision_client import JevClient

    seen: dict[str, object] = {}

    async def _stub(state: object, questions: object) -> object:
        seen["questions"] = questions
        assert isinstance(questions, dict)
        qid = next(iter(questions))
        opts = questions[qid]["criteria"] if isinstance(questions[qid], dict) else {}
        assert isinstance(opts, dict)
        winner = "reasoning_required" if "reasoning_required" in opts else next(iter(opts))
        return {
            "answers": {
                qid: {
                    "type": "choice",
                    "choice": winner,
                    "probabilities": dict.fromkeys(opts, 0.0) | {winner: 1.0},
                    "confidence": 1.0,
                }
            }
        }

    client = JevClient(transport=_stub, data_root=__import__("pathlib").Path("/tmp"))
    client._persist = lambda **kwargs: None  # type: ignore[method-assign]
    out = asyncio.run(client.route_entry("hello"))
    assert out == "reasoning_required"
    entry_q = seen["questions"]
    assert isinstance(entry_q, dict)
    entry_text = str(next(iter(entry_q.values()))["instructions"])
    assert "Today UTC is" in entry_text and "decode relative dates before choosing" in entry_text
    seen.clear()
    verdict = asyncio.run(client.assess_entry_tool("risk?", "search_sec_filings", {}, {"ok": True}))
    assert verdict in ("reasoning_required", "research_required", "node_resolved") or isinstance(verdict, str)
    assess_q = seen["questions"]
    assert isinstance(assess_q, dict)
    assess_text = str(next(iter(assess_q.values()))["instructions"])
    assert "Today UTC is" in assess_text and "decode relative dates before choosing" in assess_text


def test_run_scheduler_deadline_honors_shared_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """_run passes run_deadline to the scheduler; setup gets _SETUP_S of the wall."""
    import time as _time

    seen: dict[str, object] = {}

    def _fake_graph(prompt: str, as_of: object = None, **k: object) -> str:
        deadline = k.get("setup_deadline")
        assert isinstance(deadline, float)
        seen["setup_deadline"] = deadline
        _time.sleep(0.05)
        return "s1"

    async def _fake_sched(sid: str, **hooks: object) -> dict[str, object]:
        seen["deadline_at"] = hooks.get("deadline_at")
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", lambda sid: None)
    t0 = _time.perf_counter()
    out = kw._run({"id": "r1", "op": "run", "prompt": "hello?"}, jev=object())  # type: ignore[arg-type]
    assert out["id"] == "r1"
    assert isinstance(seen["setup_deadline"], float)
    assert isinstance(seen["deadline_at"], float)
    assert abs(seen["setup_deadline"] - (t0 + kw._SETUP_S)) < 5.0
    assert abs(seen["deadline_at"] - (t0 + kw._RUN_WALL_S - kw._FINALIZE_S)) < 5.0


def test_round2_skipped_for_corrected_query_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 2 runs only for new tickers; a corrected query alone never earns it."""
    import asyncio

    rounds = {"n": 0}

    async def _one_round(*a: object, **k: object) -> tuple[list[object], list[object], dict[str, object]]:
        rounds["n"] += 1
        return ([], [], {"calls": 1, "admitted": 0})

    def _decompose(*a: object, **k: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [{"id": "s-q1", "objectiveId": "s", "question": "q?", "dependsOn": [], "whyItMatters": "w"}],
            {"corrected_query": "fixed query"},
        )

    monkeypatch.setattr(kw, "_intake_round", _one_round)
    monkeypatch.setattr(kw, "_reasoner_decompose_with_retry", _decompose)
    out = asyncio.run(kw._graph_intake("q?", None, "s1", {}, [], None, object(), None))
    assert len(out) == 1 and rounds["n"] == 1


def test_jev_admit_timeout_falls_back_to_objective() -> None:
    """A hanging JEV decide returns the objective fallback before the setup deadline."""
    import asyncio
    import time as _time

    class _Hang:
        async def decide(self, *a: object, **k: object) -> object:
            await asyncio.sleep(30.0)
            raise AssertionError("must time out")

    t0 = _time.perf_counter()
    out = kw._jev_admit(
        "s",
        "Exact user objective?",
        _props(),
        jev=_Hang(),  # type: ignore[arg-type]
        setup_deadline=t0 + 0.2,
    )
    assert _time.perf_counter() - t0 < 5.0
    assert isinstance(out, list) and len(out) == 1
    first = out[0]
    assert isinstance(first, dict) and first["question"] == "Exact user objective?"


def test_decompose_retry_skipped_when_setup_budget_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient decompose failure with <15s left returns setup_budget, no retry."""
    import time as _time

    calls = {"n": 0}

    def _boom(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _boom)
    proposals, hints = kw._reasoner_decompose_with_retry("q?", None, "s", "", None, None, _time.perf_counter() + 5.0)
    assert calls["n"] == 1 and hints.get("fallback") == "setup_budget"
    assert len(proposals) == 1


def test_followup_reuses_session_skips_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """sessionId+gap creates one node in the live session; no new session, no intake."""
    import asyncio
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    sid = service.create_research("q?", "q?")
    node = service.create_node(sid, "q1?", "w")
    service.block_node(sid, node.node_id, "incomplete: pass 1 gap")

    created: list[str] = []
    orig_graph = kw.run_graph_prompt

    def _boom_graph(*a: object, **k: object) -> str:
        raise AssertionError("follow-up must not run setup")

    async def _fake_sched(run_sid: str, **hooks: object) -> dict[str, object]:
        assert run_sid == sid
        nodes = service.ready_nodes(sid)
        assert [n.node_id for n in nodes] == created
        service.resolve_node(sid, nodes[0].node_id)
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    orig_create = service.create_node

    def _count(run_sid: str, *a: object, **k: object) -> object:
        out = orig_create(run_sid, *a, **k)
        if isinstance(out, SimpleNamespace) or hasattr(out, "node_id"):
            created.append(out.node_id)  # type: ignore[attr-defined]
        return out

    monkeypatch.setattr(kw, "run_graph_prompt", _boom_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr("app.research.service.create_node", _count)
    monkeypatch.setattr(kw, "_close_bootstrap_job", lambda s: None)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r2", "prompt": "q?", "sessionId": sid, "gap": "need doc X"}))
    assert out["sessionId"] == sid
    assert out["unresolved"] == []
    assert len(created) == 1
    nodes = out["nodes"]
    assert isinstance(nodes, list) and len(nodes) == 2
    kw.run_graph_prompt = orig_graph  # type: ignore[method-assign]


def test_followup_unknown_session_is_invalid_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown sessionId returns invalid_params, never a new session."""
    import tempfile
    from pathlib import Path

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    out = kw._run({"id": "r9", "prompt": "q?", "sessionId": "rs:nope", "gap": "need doc X"})
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal.get("category") == "invalid_params"


def test_followup_blocks_leftover_proposed_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leftover pass-1 proposed node is blocked; the scheduler sees only the gap node."""
    import asyncio
    import tempfile
    from pathlib import Path

    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    sid = service.create_research("q?", "q?")
    leftover = service.create_node(sid, "pass-1 leftover?", "w")

    gap_ids: list[str] = []

    async def _fake_sched(run_sid: str, **hooks: object) -> dict[str, object]:
        assert run_sid == sid
        assert [n.node_id for n in service.ready_nodes(run_sid)] == gap_ids
        service.resolve_node(run_sid, gap_ids[0])
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    orig_create = service.create_node

    def _count(run_sid: str, *a: object, **k: object) -> object:
        out = orig_create(run_sid, *a, **k)
        node_id = getattr(out, "node_id", "")
        if isinstance(node_id, str) and node_id:
            gap_ids.append(node_id)
        return out

    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr("app.research.service.create_node", _count)
    monkeypatch.setattr(kw, "_close_bootstrap_job", lambda s: None)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r3", "prompt": "q?", "sessionId": sid, "gap": "need doc X"}))
    assert out["sessionId"] == sid
    assert len(gap_ids) == 1
    from app.research.repository import ResearchRepository

    assert ResearchRepository().get_node(leftover.node_id).status == "blocked"


def test_followup_links_new_evidence_not_pass1(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass-2 attempt without evidence_id links the new id, never a pass-1 id."""
    import asyncio
    import tempfile
    from pathlib import Path

    from app.research import service
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    sid = service.create_research("q?", "q?")
    service.create_node(sid, "q1?", "w")
    store = ResearchRepository()
    store.save_evidence({"evidence_id": "ev-pass1", "session_id": sid, "content": "pass-1 fact"})

    new_eid = "ev-pass2"

    async def _fake_sched(run_sid: str, **hooks: object) -> dict[str, object]:
        assert run_sid == sid
        for n in service.ready_nodes(run_sid):
            service.resolve_node(run_sid, n.node_id)
        ResearchRepository().save_evidence({"evidence_id": new_eid, "session_id": run_sid, "content": "pass-2 fact"})
        return {
            "status": "complete",
            "nodes": [{"node_id": "n", "attempts": [{"tool": "search_sec_filings", "arguments": {}}]}],
            "incomplete_guard": False,
        }

    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", lambda s: None)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r4", "prompt": "q?", "sessionId": sid, "gap": "need doc Y"}))
    calls = out["toolCalls"]
    assert isinstance(calls, list) and len(calls) == 1
    first = calls[0]
    assert isinstance(first, dict) and first.get("evidenceId") == new_eid
    ids = [e["id"] for e in out["evidence"] if isinstance(e, dict)]
    assert "ev-pass1" in ids and new_eid in ids


def test_setup_keyerror_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyError inside run_graph_prompt reports provider_error, not invalid_params."""
    import tempfile
    from pathlib import Path

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))

    def _boom(*a: object, **k: object) -> str:
        raise KeyError("missing-key")

    monkeypatch.setattr(kw, "run_graph_prompt", _boom)
    monkeypatch.setattr(kw, "_close_bootstrap_job", lambda s: None)
    out = kw._run({"id": "r5", "op": "run", "prompt": "hello?"})
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal.get("category") == "provider_error"
