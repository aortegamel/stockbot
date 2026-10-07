"""Scheduler lifecycle: domain wiring, admit-then-complete, stall, expansion."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest import mock

from app.research import scheduler as sched


def _node() -> SimpleNamespace:
    return SimpleNamespace(node_id="n1", session_id="s1", question="q?", why_it_matters="w")


class _Kernel:
    """Fake kernel recording lifecycle order; evidence store is in-memory."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.evidence: list[str] = []

    def start_job(self, sid: str, **kw: Any) -> dict[str, str]:
        jid = f"job-{len(self.calls)}"
        self.calls.append(("start", kw.get("source")))
        return {"job_id": jid}

    def heartbeat_job(self, jid: str) -> None:
        pass

    def admit_evidence(self, sid: str, jid: str, cand: dict[str, Any]) -> dict[str, str]:
        assert jid.startswith("job-"), jid
        self.calls.append(("admit", jid))
        eid = f"ev-{len(self.evidence)}"
        self.evidence.append(eid)
        return {"evidence_id": eid}

    def complete_job(self, jid: str, out: Any = None) -> None:
        self.calls.append(("complete", jid))

    def fail_job(self, jid: str, cat: str, msg: str) -> None:
        self.calls.append(("fail", jid))

    def record_decision(self, sid: str, dtype: str, **kw: Any) -> None:
        pass

    def resolve_node(self, sid: str, nid: str) -> None:
        pass

    def block_node(self, sid: str, nid: str, reason: str = "") -> None:
        pass


def _outcome(content: str = "record values here") -> SimpleNamespace:
    return SimpleNamespace(
        tool_name="query_finra",
        content=content,
        source_handle=None,
        source_refs=None,
        error=None,
        error_type=None,
        retryable=False,
        meta=None,
    )


class _JevAdmit:
    async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
        return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)

    async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {
            "probabilities": {},
            "confidence": 1.0,
            "continuation": "resolve_node",
            "continue": "resolve_node",
            "action": "resolve_node",
            "evidence": None,
            "candidate": None,
            "admit": None,
            "evidence_state": "sufficient_support",
            "decision": "sufficient_support",
        }

    async def adjudicate(self, *a: Any, **k: Any) -> SimpleNamespace:
        return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)


def _run_node(kernel: _Kernel, jev: Any) -> dict[str, Any]:
    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {
            "tool_result_id": "s1:tr:abc",
            "records": [{"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123}],
            "briefing": "B",
        }

    return asyncio.run(
        sched._run_node(
            _node(),
            session_id="s1",
            kernel=kernel,
            jev=jev,
            repo=None,
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=lambda name, result: _outcome(),
            registry=[{"name": "query_finra", "parameters": {}}],
            tool_session=SimpleNamespace(),
        )
    )


def test_admit_before_complete_with_finra_domain() -> None:
    import json

    from app.research.service import _finra_record_texts

    kernel = _Kernel()
    res = _run_node(kernel, _JevAdmit())
    assert kernel.calls[0] == ("start", "FINRA")
    seq = [c[0] for c in kernel.calls]
    assert seq.index("admit") < seq.index("complete"), seq
    assert res["status"] == "resolved" and res["admitted"] == 1
    candidate = sched._evidence_candidate(
        "query_finra",
        "FINRA",
        {
            "tool_result_id": "s1:tr:abc",
            "records": [{"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123}],
        },
        _outcome(),
    )
    assert candidate is not None
    row_text = " ".join(
        json.dumps(
            {"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123},
            sort_keys=True,
            default=str,
        ).split()
    )
    assert candidate["record_identity"] == row_text
    replay = _finra_record_texts(
        {
            "result": {
                "records": [{"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123}]
            },
            "tool_name": "query_finra",
        }
    )
    assert any(str(candidate["record_identity"]) in t for t in replay)


def test_sec_candidate_needs_handle_and_text() -> None:
    """SEC boundary pins: missing handle or empty window/content stays uncitable."""
    good_handle = {"tool_result_id": "s1:tr:sec"}
    good_result = {"source_handle": good_handle, "text": "window words"}
    assert sched._sec_locator(sched._persisted_shapes(good_result)) == "window words"
    assert sched._outcome_summary(_outcome("words")) == "words"
    assert sched._sec_evidence_candidate(good_result, _outcome("words")) is not None
    assert sched._sec_evidence_candidate({"text": "window words"}, _outcome("words")) is None
    assert sched._sec_evidence_candidate({"source_handle": good_handle}, _outcome("")) is None


def test_insufficient_evidence_state_skips_admission() -> None:
    class _JevWeak(_JevAdmit):
        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            base = await super().assess_result(*a, **k)
            return {
                **base,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
                "evidence_state": "insufficient",
                "decision": "insufficient",
            }

    kernel = _Kernel()
    res = _run_node(kernel, _JevWeak())
    assert res["admitted"] == 0
    assert not any(c[0] == "admit" for c in kernel.calls)
    assert any(c[0] == "complete" for c in kernel.calls)


def test_pure_stall_is_guard_false_with_stall_reason() -> None:
    class _KS(_Kernel):
        def ready_nodes(self, sid: str) -> list[SimpleNamespace]:
            return [_node()]

    async def stall(n: Any, sid: str, **kw: Any) -> dict[str, Any]:
        return {"node_id": "n1", "status": "gathering", "admitted": 0, "incomplete_guard": False}

    with mock.patch.object(sched, "run_node", stall):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == "stalled"
    assert out["incomplete_guard"] is False
    assert str(out["reason"]).startswith("stalled:")


def test_node_guard_trip_yields_incomplete_guard_not_stalled() -> None:
    class _KS(_Kernel):
        def ready_nodes(self, sid: str) -> list[SimpleNamespace]:
            return [_node()]

    async def trip(n: Any, sid: str, **kw: Any) -> dict[str, Any]:
        return {"node_id": "n1", "status": "blocked", "admitted": 0, "incomplete_guard": True}

    with mock.patch.object(sched, "run_node", trip):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == "incomplete_guard"
    assert out["incomplete_guard"] is True
    assert "guard trip stalled" in str(out["reason"])


def test_session_ceiling_yields_incomplete_guard() -> None:
    class _KS(_Kernel):
        def ready_nodes(self, sid: str) -> list[SimpleNamespace]:
            return [_node()]

    async def trip(n: Any, sid: str, **kw: Any) -> dict[str, Any]:
        return {"node_id": "n1", "status": "blocked", "admitted": 0, "incomplete_guard": True}

    with mock.patch.object(sched, "run_node", trip), mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 0):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert "runtime guard" in str(out["reason"])


def test_expansion_creates_jev_admitted_nodes_with_dep() -> None:
    created: list[tuple[str, str, tuple[str, ...]]] = []
    seen: list[tuple[str, dict[str, Any]]] = []
    orig = sched._record
    try:
        sched._record = lambda k, sid, dtype, **kw: seen.append((dtype, kw.get("selected")))  # type: ignore[assignment]

        class _K:
            def record_decision(self, sid: str, t: str, **kw: Any) -> None:
                pass

            def create_node(self, sid: str, q: str, w: str, depends_on: Any = None) -> SimpleNamespace:
                created.append((q, w, tuple(depends_on or ())))
                return SimpleNamespace(node_id=f"n-{len(created)}")

        class _J:
            async def decide(self, state: Any, questions: Any, decision_type: Any = None, **kw: Any) -> dict[str, Any]:
                assert decision_type == "graph_expansion"
                return {qid: {"choice": "admit"} for qid in questions}

        proposal = {
            "proposals": [
                {"question": "Follow-up A?", "whyItMatters": "Why A"},
                {"question": "Follow-up B?", "whyItMatters": ""},
            ]
        }
        n = asyncio.run(sched._expand_graph(_J(), _K(), "s1", "Objective?", "n0", proposal))
    finally:
        sched._record = orig  # type: ignore[assignment]
    assert n == 2
    assert created[0] == ("Follow-up A?", "Why A", ("n0",))
    assert created[1] == ("Follow-up B?", "Route question.", ("n0",))
    assert ("graph_expansion", {"created": 2, "proposed": 2}) in seen


def test_expansion_is_fail_closed_on_jev_outage() -> None:
    created: list[str] = []

    class _K:
        def record_decision(self, sid: str, t: str, **kw: Any) -> None:
            pass

        def create_node(self, sid: str, q: str, w: str, depends_on: Any = None) -> SimpleNamespace:
            created.append(q)
            return SimpleNamespace(node_id="n-x")

    class _JDown:
        async def decide(self, *a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("jev down")

    proposal = {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}
    n = asyncio.run(sched._expand_graph(_JDown(), _K(), "s1", "Objective?", "n0", proposal))
    assert n == 0 and created == []


def test_reason_path_calls_analyze_then_expand() -> None:
    """Production reason path: analyze shape (no proposals) -> expand supplies proposals."""
    created: list[str] = []
    calls: list[str] = []

    class _K:
        def create_node(self, sid: str, q: str, w: str, depends_on: Any = None) -> SimpleNamespace:
            created.append(q)
            return SimpleNamespace(node_id=f"n-{len(created)}")

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": "Objective?", "query": "", "as_of": None}

        def record_decision(self, sid: str, t: str, **kw: Any) -> None:
            pass

        def start_job(self, sid: str, **kw: Any) -> dict[str, str]:
            return {"job_id": "job-0"}

        def heartbeat_job(self, jid: str) -> None:
            pass

        def complete_job(self, jid: str, out: Any = None) -> None:
            pass

        def block_node(self, sid: str, nid: str, reason: str = "") -> None:
            pass

    class _J:
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            return SimpleNamespace(tool_name="reasoning_required", probabilities={}, confidence=1.0)

        async def adjudicate(self, proposal: Any, *a: Any, **k: Any) -> SimpleNamespace:
            assert isinstance(proposal, dict) and "analyses" in proposal and "proposals" not in proposal
            calls.append("adjudicate")
            return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)

        async def decide(self, state: Any, questions: Any, decision_type: Any = None, **kw: Any) -> dict[str, Any]:
            return {qid: {"choice": "admit"} for qid in questions}

        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            return {
                "probabilities": {},
                "confidence": 1.0,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
                "evidence_state": "insufficient",
                "decision": "insufficient",
            }

    class _R:
        async def analyze(self, prompt: Any) -> dict[str, Any]:
            calls.append("analyze")
            assert isinstance(prompt, str) and "CONTEXT" in prompt
            return {"analyses": [{"nodeId": "n1"}], "evidenceRequests": []}

        async def expand(self, prompt: Any, objective_id: str, prior_ids: Any) -> dict[str, Any]:
            calls.append("expand")
            assert isinstance(prompt, str) and "n1" in prompt
            assert objective_id == "s1" and "n1" in prior_ids
            return {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}

    orig = sched._record
    orig_rounds = sched._MAX_TOOL_ROUNDS
    try:
        sched._record = lambda k, sid, dtype, **kw: None  # type: ignore[assignment]
        sched._MAX_TOOL_ROUNDS = 1
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_K(),
                jev=_J(),
                reasoner=_R(),
                repo=None,
                needle_generate=lambda **kw: {"tool": "query_finra", "arguments": {}, "reasoning": "r"},
                invoke=lambda *a, **k: {"tool_result_id": None},
                to_outcome=lambda name, result: _outcome(),
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=SimpleNamespace(),
            )
        )
    finally:
        sched._record = orig  # type: ignore[assignment]
        sched._MAX_TOOL_ROUNDS = orig_rounds
    assert calls[0] == "analyze" and "expand" in calls
    assert calls.index("analyze") < calls.index("adjudicate") < calls.index("expand")
    assert created == ["Follow-up?"]
    assert out["admitted"] == 0


def test_other_no_resolve_without_admitted_evidence() -> None:
    """OTHER tool success + sufficient_support but no admitted evidence MUST NOT resolve."""
    resolved: list[str] = []
    assess_calls: list[str] = []

    class _K(_Kernel):
        def resolve_node(self, sid: str, nid: str) -> None:
            resolved.append(nid)

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": "q?", "query": "", "as_of": None}

    class _J(_JevAdmit):
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            return SimpleNamespace(tool_name="mystery_tool", probabilities={}, confidence=1.0)

        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            assess_calls.append("assess")
            return await super().assess_result(*a, **k)

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_K(),
                jev=_J(),
                repo=None,
                needle_generate=lambda **kw: {"tool": "mystery_tool", "arguments": {}, "reasoning": "r"},
                invoke=lambda *a, **k: {"note": "uncitable bytes"},
                to_outcome=lambda name, result: _outcome(),
                registry=[{"name": "mystery_tool", "parameters": {}}],
                tool_session=SimpleNamespace(),
            )
        )
    assert resolved == []
    assert out["status"] == "blocked" and out["admitted"] == 0


def test_source_fallback_returns_other() -> None:
    with mock.patch.dict("sys.modules", {"app.research.agents.source_agent": None}):
        import builtins

        real_import = builtins.__import__

        def _boom(name: str, *a: Any, **k: Any) -> Any:
            if name == "app.research.agents.source_agent":
                raise ImportError("no source_agent")
            return real_import(name, *a, **k)

        with mock.patch.object(builtins, "__import__", _boom):
            assert sched._source_for_tool("mystery_tool") == "OTHER"


def test_failed_outcome_never_reaches_assess() -> None:
    assess_calls: list[str] = []
    failed: list[str] = []

    class _K(_Kernel):
        def fail_job(self, jid: str, cat: str, msg: str) -> None:
            failed.append(jid)
            super().fail_job(jid, cat, msg)

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": "q?", "query": "", "as_of": None}

    class _J(_JevAdmit):
        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            assess_calls.append("assess")
            return await super().assess_result(*a, **k)

    def _bad_outcome(name: str, result: Any) -> SimpleNamespace:
        out = _outcome()
        out.error = "provider blew up"
        return out

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_K(),
                jev=_J(),
                repo=None,
                needle_generate=lambda **kw: {"tool": "query_finra", "arguments": {}, "reasoning": "r"},
                invoke=lambda *a, **k: {"tool_result_id": "s1:tr:x"},
                to_outcome=_bad_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=SimpleNamespace(),
            )
        )
    assert assess_calls == []
    assert failed != []
    assert out["admitted"] == 0


def test_resume_winner_filtered_from_in_node_registry() -> None:
    """research_resume re-enters its own node so it can never advance it; filtered in-node."""
    ctx = sched._node_context(
        _node(),
        "s1",
        _Kernel(),
        None,
        {
            "registry": [
                {"name": "research_resume", "parameters": {}},
                {"name": "research_start", "parameters": {}},
                {"name": "research_read_search", "parameters": {}},
                {"name": "query_finra", "parameters": {}},
            ]
        },
    )
    names = [e.get("name") for e in ctx["registry"]]
    assert "research_resume" not in names
    assert "research_start" not in names
    assert "query_finra" in names


def test_repeated_winner_drop_triggers_on_4_of_5() -> None:
    """reg_sho tool_error loop with an assess continue_research interleaved still drops."""
    seen: list[list[str]] = []
    calls = {"n": 0}
    real_select = sched._select_round

    async def spy_select(
        jev: Any, kernel: Any, sid: str, nid: str, session: Any, node: Any, registry: Any, ctx_ev: Any, attempts: Any
    ) -> Any:
        seen.append([str(e.get("name")) for e in registry])
        return await real_select(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts)

    class _JLoop:
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            return SimpleNamespace(tool_name="get_reg_sho_volume", probabilities={}, confidence=1.0)

        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            return {
                "probabilities": {},
                "confidence": 1.0,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
                "evidence": None,
                "candidate": None,
                "admit": None,
                "evidence_state": "insufficient",
                "decision": "insufficient",
            }

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_reg_sho_volume", "arguments": {}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {"tool_result_id": "s1:tr:x"}

    def fake_outcome(name: str, result: Any) -> SimpleNamespace:
        calls["n"] += 1
        out = _outcome("window words here")
        if calls["n"] != 3:
            out.error = "finra downstream blew up"
        return out

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 7):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_Kernel(),
                jev=_JLoop(),
                repo=None,
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=fake_outcome,
                registry=[
                    {"name": "get_reg_sho_volume", "parameters": {}},
                    {"name": "query_finra", "parameters": {}},
                ],
                tool_session=SimpleNamespace(),
                select_round=spy_select,
            )
        )
    # Round 4 selects on a [fail, fail, success] tail: same tool but no 3 straight errors -> no drop.
    assert "get_reg_sho_volume" in seen[3]
    # Round 6 selects on [fail, fail, success, fail, fail]: 4-of-5 repeated winner -> dropped.
    assert "get_reg_sho_volume" not in seen[5]
    assert out["status"] == "blocked" and out["incomplete_guard"] is True


def test_select_raises_twice_then_succeeds_continues() -> None:
    """Select 400s must not escape to node_failure; the node continues once select recovers."""
    calls = {"n": 0}
    failures: list[str] = []

    class _K(_Kernel):
        def record_decision(self, sid: str, dtype: str, **kw: Any) -> None:
            if dtype == "node_failure":
                failures.append(dtype)

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": "q?", "query": "", "as_of": None}

    class _J(_JevAdmit):
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("400 max_tokens_exceeded: decide sidecar payload too large")
            return await super().select_tool(*a, **k)

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 5):
        out = asyncio.run(
            sched.run_node(
                _node(),
                "s1",
                kernel=_K(),
                jev=_J(),
                repo=None,
                needle_generate=lambda **kw: {"tool": "query_finra", "arguments": {}, "reasoning": "r"},
                invoke=lambda *a, **k: {
                    "tool_result_id": "s1:tr:abc",
                    "records": [
                        {"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123}
                    ],
                    "briefing": "B",
                },
                to_outcome=lambda name, result: _outcome(),
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=SimpleNamespace(),
            )
        )
    assert out["status"] == "resolved" and out["admitted"] == 1
    assert failures == []
    notes = [a.get("error") for a in out["attempts"] if isinstance(a, dict) and a.get("error")]
    assert any(str(n).startswith("select failed (400 max_tokens") for n in notes)


def test_select_raises_thrice_yields_blocked_terminal() -> None:
    """3 consecutive select failures block visibly incomplete, never failed."""
    blocked: list[str] = []
    failures: list[str] = []

    class _K(_Kernel):
        def block_node(self, sid: str, nid: str, reason: str = "") -> None:
            blocked.append(reason)

        def record_decision(self, sid: str, dtype: str, **kw: Any) -> None:
            if dtype == "node_failure":
                failures.append(dtype)

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": "q?", "query": "", "as_of": None}

    class _JDown(_JevAdmit):
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            raise RuntimeError("400 max_tokens_exceeded: decide sidecar payload too large")

    out = asyncio.run(sched.run_node(_node(), "s1", kernel=_K(), jev=_JDown(), repo=None))
    assert out["status"] == "blocked" and failures == []
    assert out.get("incomplete_guard") is True
    assert len(out["attempts"]) == 3 and blocked != []


def test_select_fail_trim_keeps_ctx_bounded() -> None:
    """A select retry after 400 sends a smaller state: last ~5 ctx items."""
    many = [
        {"tool": "get_sec_filing", "job_id": f"job-{i}", "outcome_summary": f"obs {i}", "error": None} for i in range(8)
    ]
    assert len(sched._context_evidence([], many)) == 8
    assert len(sched._context_evidence([], many, cap_last=5)) == 5
    seen: list[int] = []

    class _K(_Kernel):
        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": "q?", "query": "", "as_of": None}

    class _J(_JevAdmit):
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            ctx = a[3] if len(a) > 3 else k.get("ctx_evidence", [])
            seen.append(len(ctx) if isinstance(ctx, list) else -1)
            if len(seen) == 1:
                raise RuntimeError("400 max_tokens_exceeded: decide sidecar payload too large")
            return await super().select_tool(*a, **k)

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {"tool_result_id": "s1:tr:abc", "records": [], "briefing": "B"}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 3):
        out = asyncio.run(
            sched.run_node(
                _node(),
                "s1",
                kernel=_K(),
                jev=_J(),
                repo=None,
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=lambda name, result: _outcome(),
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=SimpleNamespace(),
            )
        )
    assert out["status"] == "resolved"
    assert len(seen) >= 2 and seen[1] <= 5


def test_accession_carry_fills_from_packet_context() -> None:
    """Open-step {} args + packet accession in context -> args carry the accession."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_filing", "arguments": {}, "reasoning": "r"}

    async def fake_doc_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_document", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    filing_evidence = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), filing_evidence, [], None)
    )
    assert args.get("accession_no") == acc
    doc_evidence = [
        {"source_handle": {"accession_no": acc, "document_name": "nvda-10k.htm"}, "outcome_summary": f"read {acc}"}
    ]
    doc_args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_doc_gen, "get_sec_document", [], session, _node(), doc_evidence, [], None)
    )
    assert doc_args.get("accession_no") == acc
    assert doc_args.get("document_name") == "nvda-10k.htm"


def test_accession_carry_withheld_none_still_fills_from_packet() -> None:
    """Needle withholds (None) on ungrounded open-step + packet accession -> carry fills, no mismatch."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: no grounded accession"}

    session = {"session_id": "s1", "objective": "q?"}
    evidence = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), evidence, [], None)
    )
    assert args.get("accession_no") == acc

    try:
        asyncio.run(sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), [], [], None))
    except ValueError as exc:
        assert "needle tool mismatch" in str(exc)
    else:
        raise AssertionError("withhold without packet must still raise mismatch")


def test_accession_carry_leaves_args_without_packet() -> None:
    """No packet accession anywhere -> args stay as Needle emitted them."""

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_filing", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), [], [], None)
    )
    assert "accession_no" not in args


def test_runtime_mismatch_carry_returns_packet_accession() -> None:
    """Server-side RuntimeError mismatch + packet accession -> carried args, no raise."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        raise RuntimeError("needle arguments.generate failed (needle tool mismatch: JEV=x Needle=y)")

    session = {"session_id": "s1", "objective": "q?"}
    evidence = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), evidence, [], None)
    )
    assert args.get("accession_no") == acc


def test_runtime_mismatch_no_packet_reraises() -> None:
    """Server-side RuntimeError mismatch + no packet -> same raise as today."""

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        raise RuntimeError("needle arguments.generate failed (needle tool mismatch: JEV=x Needle=y)")

    session = {"session_id": "s1", "objective": "q?"}
    try:
        asyncio.run(sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), [], [], None))
    except RuntimeError as exc:
        assert "needle tool mismatch" in str(exc)
    else:
        raise AssertionError("mismatch without packet must re-raise")


def test_grounding_hint_present_in_context() -> None:
    """Accession/ticker/dataset tools get one compact grounding hint in context."""
    seen: dict[str, Any] = {}

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    asyncio.run(sched._generate_tool_arguments(fake_gen, "list_sec_filings", [], session, _node(), [], [], None))
    hint = str(seen.get("context", {}).get("grounding_hint", ""))
    assert "accession_no only from packet" in hint and "never org words" in hint and "singular YYYY-MM-DD" in hint


def test_context_carries_scope_and_today() -> None:
    """Needle context carries resolved temporal_scope + today_utc."""
    seen: dict[str, Any] = {}

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "reasoning": "r"}

    scope = {"mode": "range", "start": "2025-04-01", "end": "2025-06-30"}
    session = {"session_id": "s1", "objective": "q?", "temporal_scope": scope}
    asyncio.run(sched._generate_tool_arguments(fake_gen, "list_sec_filings", [], session, _node(), [], [], None))
    ctx = seen.get("context", {})
    assert ctx.get("temporal_scope") == scope and isinstance(ctx.get("today_utc"), str) and len(ctx["today_utc"]) == 10
    assert str(seen.get("objective", "")) == "q?"


def test_generate_tool_arguments_objective_raw_verbatim() -> None:
    """Needle-path objective passes verbatim; no [Today UTC ...] stamp (prompt-only date)."""
    seen: dict[str, Any] = {}

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    asyncio.run(sched._generate_tool_arguments(fake_gen, "list_sec_filings", [], session, _node(), [], [], None))
    assert str(seen.get("objective", "")) == "q?"


def test_analyze_expand_prompts_stamp_objective() -> None:
    """Analyze/expand CONTEXT objective carries the [Today UTC ...] prefix."""
    import json as _json

    session = {"session_id": "s1", "objective": "q?"}
    ctx_a = _json.loads(sched._analyze_prompt(session, _node(), [], []).split("CONTEXT: ", 1)[1])
    assert str(ctx_a["objective"]["prompt"]).startswith("[Today UTC ")
    ctx_e = _json.loads(
        sched._expand_prompt(session, _node(), {"analyses": [], "evidenceRequests": []}, "s1", "n1").split(
            "CONTEXT: ", 1
        )[1]
    )
    assert str(ctx_e["objective"]["prompt"]).startswith("[Today UTC ")


def test_scan_finds_accession_in_full_json() -> None:
    """Accession nested where repr hides it is still found via json.dumps fallback."""

    class _Hidden(dict):  # type: ignore[type-arg]
        def __repr__(self) -> str:
            return "packet(hidden)"

    acc = "0001628280-26-044069"
    packet = _Hidden({"nested": [{"accession_no": acc}]})
    assert repr(packet) == "packet(hidden)"
    found, _ = sched._scan_packet_accession(packet)
    assert found == acc


def test_identical_failures_break_with_guided_message() -> None:
    """3 identical failing calls block with guidance instead of burning 10 rounds."""

    class _JLoop:
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "query_finra", "arguments": {"ticker": "X"}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {"tool_result_id": "s1:tr:x"}

    def bad_outcome(name: str, result: Any) -> SimpleNamespace:
        out = _outcome("x")
        out.error = "boom provider down detail identical"
        return out

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 10):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_Kernel(),
                jev=_JLoop(),
                repo=None,
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=bad_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=SimpleNamespace(),
            )
        )
    assert out["status"] == "blocked" and out["incomplete_guard"] is True
    assert len(out["attempts"]) == 3
    assert "3x" in str(out["reason"]) and "different ticker/dataset/accession" in str(out["reason"])


def test_force_open_registry_carry_tools_on_nav_packet() -> None:
    """Nav-only packet (accession, zero admissions) restricts selection to carry tools."""
    acc = "0001628280-26-044069"
    registry: list[Any] = [
        {"name": "search_sec_filings", "parameters": {}},
        {"name": "get_sec_filing", "parameters": {}},
        {"name": "query_finra", "parameters": {}},
    ]
    attempts: list[Any] = [{"tool": "search_sec_filings", "outcome_summary": f"filing {acc} 10-K"}]
    forced = sched._force_open_registry(registry, attempts, [], 0)
    assert forced is not None and [e.get("name") for e in forced] == ["get_sec_filing"]
    assert sched._force_open_registry(registry, attempts, [], 1) is None
    assert (
        sched._force_open_registry(registry, [{"tool": "search_sec_filings", "outcome_summary": "no hits"}], [], 0)
        is None
    )


def test_relative_tradedate_scrubbed_to_wtd_omit() -> None:
    """Phrase tradeDate on get_reg_sho_volume scrubs to omit (WTD path); valid dates pass through."""

    async def fake_phrase(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "AAPL", "tradeDate": "this week"},
            "reasoning": "r",
        }

    async def fake_valid(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "AAPL", "tradeDate": "2026-09-22"},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "Apple this week?"}
    scrubbed, _ = asyncio.run(
        sched._generate_tool_arguments(fake_phrase, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert scrubbed == {"ticker": "AAPL"}
    kept, _ = asyncio.run(
        sched._generate_tool_arguments(fake_valid, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert kept == {"ticker": "AAPL", "tradeDate": "2026-09-22"}


def test_sec_withhold_falls_back_to_objective_ticker_latest() -> None:
    """Needle withhold (None) on SEC tools seeds explicit objective ticker, latest (no dates)."""

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    listed, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert listed == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    searched, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "search_sec_filings", [], session, _node(), [], [], None)
    )
    assert searched["ticker"] == "NVDA" and "start_date" not in searched and "end_date" not in searched


def test_sec_withhold_without_ticker_still_raises() -> None:
    """No explicit ticker token (first-word fallback forbidden) -> mismatch still raises."""

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "Growth slowed last quarter — which segment drove it?"}
    try:
        asyncio.run(sched._generate_tool_arguments(fake_none, "list_sec_filings", [], session, _node(), [], [], None))
    except ValueError as exc:
        assert "needle tool" in str(exc)
    else:
        raise AssertionError("withhold without explicit ticker must still raise mismatch")


def test_sho_withhold_falls_back_to_both_fields() -> None:
    """Needle withhold (None) on SHO seeds ticker+company_name from for-<Company>."""

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?"}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert args == {"ticker": "AAPL", "company_name": "Apple"}


def test_with_company_seeds_ticker_and_name() -> None:
    """'What is happening with Oracle now' resolves via the EDGAR index (live oracle-query defect)."""
    assert sched._objective_company("What is happening with Oracle now") == ("Oracle", "ORCL")
    assert sched._objective_subject_ticker("What is happening with Oracle now") == "ORCL"
    repaired = sched._repair_tool_arguments("get_material_events", {}, "What is happening with Oracle now", "s", "n")
    assert repaired == {"company_name": "Oracle", "ticker": "ORCL"}
    assert sched._objective_company("What is happening with oracle now") == ("oracle", "ORCL")
    assert sched._objective_subject_ticker("What is happening with oracle now") == "ORCL"
    assert sched._objective_company("What drove growth this week?") is None
    assert sched._objective_company("news for growth this week") is None
    assert sched._objective_subject_ticker("news for growth this week") is None


def test_sec_placeholder_forms_reseed_latest() -> None:
    """Needle forms ['YYYY-MM-DD'] on quarterly revenue re-lists latest 10-Q/10-K/8-K."""

    async def fake_placeholder(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_placeholder, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}


def test_sho_org_ticker_remaps_from_objective() -> None:
    """Needle ticker FINRA + company Apple remaps to AAPL from the objective."""

    async def fake_org(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "FINRA", "company_name": "Apple"},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?"}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_org, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert args == {"ticker": "AAPL", "company_name": "Apple"}


def test_repeat_8k_swaps_to_packet_10q() -> None:
    """Same 8-K accession twice on quarterly revenue swaps to the packet 10-Q."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_8k(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts = [
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence = [{"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}]
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_8k, "get_sec_filing", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq


def test_withhold_carry_repairs_query_and_swaps_10q() -> None:
    """Needle withhold with a carried 8-K repeat gets query seed + 10-Q swap."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence = [{"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}]
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "get_sec_document", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq
    assert args.get("query") == "revenue increased"


def test_repeat_8k_error_swaps_to_packet_10q() -> None:
    """Same 8-K twice with query errors on revenue swaps to the packet 10-Q."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_8k(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": "query not found"},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": "query not found"},
    ]
    evidence = [{"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}]
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_8k, "get_sec_document", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq


def test_revenue_document_seeds_query() -> None:
    """get_sec_document without query on quarterly revenue seeds a revenue query."""

    async def fake_doc(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_document", "arguments": {"accession_no": "0001045810-26-000075"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_doc, "get_sec_document", [], session, _node(), [], [], None)
    )
    assert args.get("query") == "revenue increased"


def test_stale_as_of_dropped_dateless_objective() -> None:
    """Q4 shape: invented as_of on a dateless accession query scrubs to omit."""

    async def fake_stale(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079, give me the filing metadata (form, dates, filer)",
    }
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_stale, "get_sec_filing", [], session, _node(), [], [], None)
    )
    assert args == {"accession_no": "0000320193-25-000079"}


def test_explicit_as_of_kept_when_in_objective() -> None:
    """Explicit YYYY-MM-DD in the objective keeps as_of (latest-available otherwise)."""

    async def fake_kept(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079 as of 2025-09-27, give me the filing metadata",
    }
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_kept, "get_sec_filing", [], session, _node(), [], [], None)
    )
    assert args == {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"}


def test_session_as_of_kept_when_datetime() -> None:
    """Session as_of as datetime keeps the matching cutoff (not scrubbed)."""
    from datetime import datetime

    async def fake_kept(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079, give me the filing metadata",
        "as_of": datetime(2025, 9, 27, 12, 0, 0),
    }
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_kept, "get_sec_filing", [], session, _node(), [], [], "2025-09-27")
    )
    assert args == {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"}


def test_garbage_identifier_reseeded_from_objective() -> None:
    """Needle identifier F1/TODAY/SEC on an Apple query reseeds to AAPL."""

    async def fake_garbage(**kw: Any) -> dict[str, Any]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "F1"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "List Apple's most recent 10-K and 10-Q filings."}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_garbage, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "AAPL"


def test_identifier_repair_keeps_valid_cik_and_case() -> None:
    """CIK digits and lowercase ticker pass through; only EDGAR-unresolvable garbage reseeds."""

    async def fake_cik(**kw: Any) -> dict[str, Any]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "0000320193"}, "reasoning": "r"}

    async def fake_lower(**kw: Any) -> dict[str, Any]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "aapl"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "List Apple's most recent 10-K and 10-Q filings."}
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_cik, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "0000320193"
    args, _ = asyncio.run(
        sched._generate_tool_arguments(fake_lower, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "aapl"


def test_empty_search_withhold_reselects() -> None:
    """Empty search_sec_filings args on a withhold-like objective re-select, never invoke empty."""
    import pytest

    async def fake_empty(**kw: Any) -> dict[str, Any]:
        return {"tool": "search_sec_filings", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove Growth this week?"}
    with pytest.raises(sched._ReselectRequest, match="re-selecting"):
        asyncio.run(
            sched._generate_tool_arguments(fake_empty, "search_sec_filings", [], session, _node(), [], [], None)
        )


def test_ungrounded_search_attempt_reselects() -> None:
    """A reselect attempt settles as bookkeeping, never as a handler error."""

    async def fake_empty(**kw: Any) -> dict[str, Any]:
        return {"tool": "search_sec_filings", "arguments": {}, "reasoning": "r"}

    kernel = _Kernel()
    session = {"session_id": "s1", "objective": "What drove Growth this week?"}
    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_sec_filings",
            node=_node(),
            session=session,
            registry=[{"name": "search_sec_filings", "parameters": {}}],
            evidence=[],
            attempts=[],
            kernel=kernel,
            needle_generate=fake_empty,
            invoke=lambda *a, **k: (_ for _ in ()).throw(AssertionError("ungrounded search must not invoke")),
            to_outcome=lambda name, result: _outcome(),
            tool_session=SimpleNamespace(),
            as_of=None,
        )
    )
    assert out["tool"] == "search_sec_filings" and out["error_type"] == "reselect"
    settled, done = sched._settle_attempt(kernel, {**out, "outcome": None})
    assert done is True and settled is not None and settled["tool"] == "search_sec_filings"


def test_needle_carry_fails_loud_after_two() -> None:
    """Two carried-accession fallbacks then a third needle outage raises, never carries again (D)."""
    carried = {"reasoning": "carried accession from packet after needle failure"}

    async def boom(**kw: Any) -> dict[str, Any]:
        raise RuntimeError("needle worker closed")

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    with __import__("pytest").raises(RuntimeError, match="carried fallbacks used"):
        asyncio.run(
            sched._generate_tool_arguments(
                boom, "get_sec_filing", [], session, _node(), [], [{**carried}, {**carried}], None
            )
        )


def test_accession_family_break_across_tools() -> None:
    """Same accession failing across get_sec_filing/get_sec_document breaks as one family (C)."""
    acc = "0001193125-09-214859"
    attempts = [
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc}, "error": "boom-a"},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc}, "error": "boom-b"},
    ]
    guided = sched._accession_family_break(attempts)
    assert guided is not None and guided["error_type"] == "invalid_tool_arguments"
    assert acc in guided["error"]


def test_find_sec_entities_seeded_from_objective() -> None:
    """find_sec_entities seeds a grounded query (ticker-first) instead of erroring on empty args."""
    from app.research import scheduler as _sched

    seeded = _sched._fallback_sec_args(
        "find_sec_entities",
        "Did Tesla insiders actually sell shares last quarter, or only file planned-sale notices?",
    )
    assert seeded == {"query": "TSLA"}


def test_sync_invoke_overlaps_not_serial() -> None:
    """Two 0.3s sync SEC tools finish near the slower time, never the sum (A1)."""
    import time as _time

    def slow(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        _time.sleep(0.3)
        return {"ok": True, "tool_result_id": "s1:tr:x"}

    async def _both() -> None:
        await asyncio.gather(
            *(
                sched._invoke_attempt_tool(
                    slow,
                    lambda n, r: SimpleNamespace(error=None),
                    "list_sec_filings",
                    {},
                    SimpleNamespace(),
                    "n1",
                    "s1",
                    None,
                    "j",
                )
                for _ in range(2)
            )
        )

    start = _time.perf_counter()
    asyncio.run(_both())
    assert _time.perf_counter() - start < 0.55


def test_needle_error_streak_breaks_at_three() -> None:
    """3 consecutive generation-shaped errors streak 3; args-produced breaks it (D5)."""
    bad = {"tool": "search_web", "error": "needle down", "arguments": {}}
    assert sched._needle_error_streak([dict(bad), dict(bad)]) == 2
    assert sched._needle_error_streak([dict(bad), dict(bad), dict(bad)]) == 3
    grounded = dict(bad, arguments={"query": "q"})
    assert sched._needle_error_streak([dict(bad), dict(bad), grounded]) == 0


def test_intake_round_admits_citable_without_jev_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """_intake_round admits citable bytes by code; uncitable/error close jobs (E1)."""
    from types import SimpleNamespace as _NS

    from app.research import kernel_worker as kw

    kernel = _Kernel()

    async def fake_attempt(tool: str, args: dict[str, Any], sid: str, *a: Any, **k: Any) -> dict[str, Any]:
        if tool == "search_web":
            return {
                "tool": tool,
                "arguments": args,
                "job_id": "job-web",
                "result": {
                    "tool_result_id": "s1:tr:w",
                    "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
                },
                "outcome": _NS(content="web bytes here", error=None),
                "outcome_summary": "web bytes here",
                "error": None,
            }
        if tool == "get_sec_document":
            return {
                "tool": tool,
                "arguments": args,
                "job_id": "job-doc",
                "result": {"tool_result_id": "s1:tr:d"},
                "outcome": _NS(content="", error=None),
                "outcome_summary": "",
                "error": None,
            }
        return {"tool": tool, "arguments": args, "job_id": "job-bad", "error": "boom", "outcome": None}

    async def fake_chain(*a: Any, **k: Any) -> list[tuple[str, dict[str, Any], dict[str, Any], float]]:
        return [
            ("list_sec_filings", {"ticker": "ORCL"}, await fake_attempt("list_sec_filings", {}, "s1"), 5.0),
            ("get_sec_document", {"accession_no": "x"}, await fake_attempt("get_sec_document", {}, "s1"), 5.0),
        ]

    monkeypatch.setattr(kw, "_intake_attempt", fake_attempt)
    monkeypatch.setattr(kw, "_intake_ticker_chain", fake_chain)
    monkeypatch.setattr(sched, "_load_evidence", lambda sid, kernel, repo: [{"evidence_id": "ev-0"}])
    admitted, raw, stats = asyncio.run(
        kw._intake_round("orcl manjure", ["ORCL"], "s1", {}, [], None, kernel, None, None)
    )
    assert stats["outcomes"] == {"admitted": 1, "not_citable": 1, "error": 1, "timeout": 0}
    assert stats["admitted_ids"] and stats["admitted"] == 1 and admitted == [{"evidence_id": "ev-0"}]


def test_intake_round_budget_timeout_records_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pending budget tasks synthesize timeout records with wall_ms (A2)."""
    from app.research import kernel_worker as kw

    kernel = _Kernel()

    async def slow_chain(*a: Any, **k: Any) -> list[tuple[str, dict[str, Any], dict[str, Any], float]]:
        await asyncio.sleep(60)
        return []

    monkeypatch.setattr(kw, "_intake_ticker_chain", slow_chain)
    monkeypatch.setattr(kw, "_INTAKE_BUDGET_S", 0.05)
    monkeypatch.setattr(sched, "_load_evidence", lambda sid, kernel, repo: [])


def test_sec_thread_cap_survives_four_concurrent_calls() -> None:
    """4 concurrent SEC thread calls complete; double-acquire would deadlock (bug 1)."""
    import time

    async def _main() -> list[str]:
        async def _call(i: int) -> str:
            return await asyncio.wait_for(sched._sec_thread_call(lambda: (time.sleep(0.1), f"r{i}")[1]), timeout=10)

        return await asyncio.gather(*(_call(i) for i in range(5)))

    assert sorted(asyncio.run(_main())) == ["r0", "r1", "r2", "r3", "r4"]


def test_session_run_survives_second_event_loop() -> None:
    """Two asyncio.run sessions with 3 contending nodes; loop-bound cap would fail run 2 (bug 2)."""
    from types import SimpleNamespace
    from unittest import mock

    class _KS:
        def __init__(self) -> None:
            self.seen: set[str] = set()

        def ready_nodes(self, sid: str) -> list[SimpleNamespace]:
            if sid in self.seen:
                return []
            self.seen.add(sid)
            return [
                SimpleNamespace(node_id=f"n{i}", session_id=sid, question="q?", why_it_matters="w") for i in range(3)
            ]

    async def done(n: Any, sid: str, **kw: Any) -> dict[str, Any]:
        await asyncio.sleep(0)  # yield so the 3rd node actually waits on the 2-lane cap
        return {"node_id": n.node_id, "status": "resolved", "admitted": 1, "incomplete_guard": False}

    with mock.patch.object(sched, "run_node", done):
        first = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
        second = asyncio.run(sched.run("s2", kernel=_KS(), repo=None))
    assert first["status"] == "complete" and second["status"] == "complete"
    assert len(first["nodes"]) == 3 and len(second["nodes"]) == 3
