"""Scheduler lifecycle: domain wiring, admit-then-complete, stall, expansion."""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import override
from unittest import mock

import pytest

from app.decision_client import JevClient
from app.reasoner_client import ReasonerClient
from app.research import scheduler as sched
from app.research.models import DecisionRecord, JSONValue, ResearchNode, ToolDecision, new_decision_id
from app.runtime import ToolResultMeta
from app.tool_runtime import RuntimeToolSession, ToolOutcome

_FINRA_ROW: dict[str, JSONValue] = {
    "settlementDate": "2024-01-01",
    "symbolCode": "XYZ",
    "currentShortPositionQuantity": 123,
}
_FINRA_RESULT: dict[str, JSONValue] = {"tool_result_id": "s1:tr:abc", "records": [_FINRA_ROW], "briefing": "B"}


def _node(node_id: str = "n1", session_id: str = "s1") -> ResearchNode:
    return ResearchNode(node_id=node_id, session_id=session_id, question="q?", why_it_matters="w")


def _tool_session() -> RuntimeToolSession:
    return RuntimeToolSession(session_id="scheduler:s1")


class _Kernel(sched._Kernel):
    """Fake kernel recording lifecycle order; evidence and tool-result stores are in-memory."""

    def __init__(self, tool_results: Mapping[str, dict[str, JSONValue]] | None = None, objective: str = "") -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []
        self.evidence: list[str] = []
        self.decisions: list[tuple[str, object]] = []
        self.tool_results: dict[str, dict[str, JSONValue]] = dict(tool_results or {})
        self.objective = objective

    @override
    def start_job(
        self,
        session_id: str,
        type: str = "source_agent",
        source: str | None = None,
        *,
        owner: str = "kernel",
        request_id: str | None = None,
    ) -> dict[str, JSONValue]:
        jid = f"job-{len(self.calls)}"
        self.calls.append(("start", source))
        return {"job_id": jid}

    @override
    def heartbeat_job(self, job_id: str) -> None:
        pass

    @override
    def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
        assert job_id.startswith("job-"), job_id
        self.calls.append(("admit", job_id))
        eid = f"ev-{len(self.evidence)}"
        self.evidence.append(eid)
        return {"evidence_id": eid}

    @override
    def complete_job(self, job_id: str, outcome: Mapping[str, object] | None = None) -> None:
        self.calls.append(("complete", job_id))

    @override
    def fail_job(self, job_id: str, category: str, message: str) -> None:
        self.calls.append(("fail", job_id, category))

    @override
    def record_decision(
        self,
        session_id: str,
        decision_type: str,
        candidates: Mapping[str, object],
        probabilities: Mapping[str, object],
        selected: object,
        *,
        node_id: str | None = None,
        job_id: str | None = None,
        confidence: float | None = None,
        request: object = None,
        response: object = None,
    ) -> DecisionRecord:
        self.decisions.append((decision_type, selected))
        return DecisionRecord(
            decision_id=new_decision_id(),
            session_id=session_id,
            node_id=node_id,
            job_id=job_id,
            decision_type=decision_type,
            candidates={},
            probabilities={},
            selected=None,
            confidence=confidence,
        )

    @override
    def resolve_node(self, session_id: str, node_id: str) -> ResearchNode:
        return _node(node_id, session_id)

    @override
    def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
        return _node(node_id, session_id)

    @override
    def create_node(
        self, session_id: str, question: str, why_it_matters: str, depends_on: Sequence[str] | None = None
    ) -> ResearchNode:
        return _node("n-new", session_id)

    @override
    def ready_nodes(self, session_id: str) -> list[ResearchNode]:
        return []

    @override
    def get_session(self, session_id: str) -> dict[str, JSONValue]:
        return {"session_id": session_id, "objective": self.objective, "query": "", "as_of": None}

    @override
    def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
        return []

    @override
    def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
        return self.tool_results[tool_result_id]


class _DupKernel(_Kernel):
    """Store already holds the identity: admission returns the stored winner, accepted False."""

    @override
    def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
        self.calls.append(("admit", job_id))
        return {"evidence_id": "ev-winner", "accepted": False, "duplicate_of": "ev-winner"}


def _outcome(content: str = "record values here", error: str | None = None) -> ToolOutcome:
    return ToolOutcome(
        tool_name="query_finra",
        content=content,
        source_handle=None,
        source_refs=None,
        error=error,
        error_type=None,
        retryable=False,
        meta=ToolResultMeta(0, None, False, None, [], {}),
    )


def _to_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
    return _outcome()


def _invoke(tool: str) -> ToolDecision:
    return ToolDecision(action="invoke", tool_name=tool, tool_names=(tool,), probabilities={}, confidence=1.0)


class _JevAdmit(JevClient):
    @override
    async def select_tool(self, *a: object, **k: object) -> ToolDecision:
        return _invoke("query_finra")

    @override
    async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
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

    @override
    async def adjudicate(self, *a: object, **k: object) -> ToolDecision:
        return _invoke("query_finra")


def _run_node(kernel: _Kernel, jev: JevClient) -> dict[str, JSONValue]:
    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        return dict(_FINRA_RESULT)

    kernel.tool_results["s1:tr:abc"] = dict(_FINRA_RESULT)
    return asyncio.run(
        sched._run_node(
            _node(),
            session_id="s1",
            kernel=kernel,
            jev=jev,
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=_to_outcome,
            registry=[{"name": "query_finra", "parameters": {}}],
            tool_session=_tool_session(),
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
    finra_result: dict[str, JSONValue] = {"tool_result_id": "s1:tr:abc", "records": [_FINRA_ROW]}
    candidate = sched._evidence_candidate(
        "query_finra", "FINRA", finra_result, _outcome(), _Kernel({"s1:tr:abc": finra_result})
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
    assert sched._sec_evidence_candidate(good_result, _outcome("words"), _Kernel()) is not None
    assert sched._sec_evidence_candidate({"text": "window words"}, _outcome("words"), _Kernel()) is None
    assert sched._sec_evidence_candidate({"source_handle": good_handle}, _outcome(""), _Kernel()) is None


def test_insufficient_evidence_state_skips_admission() -> None:
    class _JevWeak(_JevAdmit):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
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


def test_duplicate_admission_links_winner_without_counting() -> None:
    """A duplicate cites the stored winner on its attempt but never counts as admitted."""
    res = _run_node(_DupKernel(), _JevAdmit())
    assert res["status"] == "resolved" and res["admitted"] == 0
    attempts = res["attempts"]
    assert isinstance(attempts, list)
    ids: list[JSONValue] = []
    for a in attempts:
        if isinstance(a, dict) and a.get("tool") == "query_finra":
            ids.append(a.get("evidence_id"))
    assert ids == ["ev-winner"]


def test_duplicate_admission_signals_no_progress() -> None:
    """Only a fresh admission (no accepted key, or accepted True) is new-evidence progress."""

    class _JevContinue(_JevAdmit):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            base = await super().assess_result(*a, **k)
            return {**base, "continuation": "continue_research", "continue": "continue_research"}

    def settle(kernel: _Kernel) -> dict[str, object]:
        result: dict[str, JSONValue] = {
            "tool_result_id": "s1:tr:abc",
            "records": [{"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 1}],
        }
        kernel.tool_results["s1:tr:abc"] = result
        record: dict[str, object] = {
            "tool": "query_finra",
            "arguments": {},
            "outcome": _outcome(),
            "outcome_summary": "record values here",
            "error": None,
            "result": result,
            "job_id": "job-1",
        }
        attempts: list[dict[str, JSONValue]] = []
        out = asyncio.run(
            sched._settle_round(
                [record], _JevContinue(), kernel, _node(), [], attempts, "s1", "n1", _invoke("query_finra"), 0
            )
        )
        return {**out, "evidence_id": attempts[-1].get("evidence_id")}

    fresh = settle(_Kernel())
    dup = settle(_DupKernel())
    assert (fresh["admitted"], fresh["progressed"], fresh["evidence_id"]) == (1, True, "ev-0")
    assert (dup["admitted"], dup["progressed"], dup["evidence_id"]) == (0, False, "ev-winner")


def test_pure_stall_is_guard_false_with_stall_reason() -> None:
    class _KS(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node()]

    async def stall(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {"node_id": "n1", "status": "gathering", "admitted": 0, "incomplete_guard": False}

    with mock.patch.object(sched, "run_node", stall):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == "stalled"
    assert out["incomplete_guard"] is False
    assert str(out["reason"]).startswith("stalled:")


def test_node_guard_trip_yields_incomplete_guard_not_stalled() -> None:
    class _KS(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node()]

    async def trip(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {"node_id": "n1", "status": "blocked", "admitted": 0, "incomplete_guard": True}

    with mock.patch.object(sched, "run_node", trip):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == "incomplete_guard"
    assert out["incomplete_guard"] is True
    assert "guard trip stalled" in str(out["reason"])


def test_session_ceiling_yields_incomplete_guard() -> None:
    class _KS(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node()]

    async def trip(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {"node_id": "n1", "status": "blocked", "admitted": 0, "incomplete_guard": True}

    with mock.patch.object(sched, "run_node", trip), mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 0):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert "runtime guard" in str(out["reason"])


def test_expansion_creates_jev_admitted_nodes_with_dep() -> None:
    created: list[tuple[str, str, tuple[str, ...]]] = []

    class _K(_Kernel):
        @override
        def create_node(
            self, session_id: str, question: str, why_it_matters: str, depends_on: Sequence[str] | None = None
        ) -> ResearchNode:
            created.append((question, why_it_matters, tuple(depends_on or ())))
            return _node(f"n-{len(created)}", session_id)

    class _J(JevClient):
        @override
        async def decide(
            self, state: object, questions: Mapping[str, JSONValue], decision_type: object = None, **kw: object
        ) -> dict[str, dict[str, JSONValue]]:
            assert decision_type == "graph_expansion"
            return {qid: {"choice": "admit"} for qid in questions}

    proposal = {
        "proposals": [
            {"question": "Follow-up A?", "whyItMatters": "Why A"},
            {"question": "Follow-up B?", "whyItMatters": ""},
        ]
    }
    kernel = _K()
    n = asyncio.run(sched._expand_graph(_J(), kernel, "s1", "Objective?", "n0", proposal))
    assert n == 2
    assert created[0] == ("Follow-up A?", "Why A", ("n0",))
    assert created[1] == ("Follow-up B?", "Route question.", ("n0",))
    assert ("graph_expansion", {"created": 2, "proposed": 2}) in kernel.decisions


def test_expansion_is_fail_closed_on_jev_outage() -> None:
    created: list[str] = []

    class _K(_Kernel):
        @override
        def create_node(
            self, session_id: str, question: str, why_it_matters: str, depends_on: Sequence[str] | None = None
        ) -> ResearchNode:
            created.append(question)
            return _node("n-x", session_id)

    class _JDown(JevClient):
        @override
        async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
            raise RuntimeError("jev down")

    proposal = {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}
    n = asyncio.run(sched._expand_graph(_JDown(), _K(), "s1", "Objective?", "n0", proposal))
    assert n == 0 and created == []


def test_reason_path_calls_analyze_then_expand() -> None:
    """Production reason path: analyze shape (no proposals) -> expand supplies proposals."""
    created: list[str] = []
    calls: list[str] = []

    class _K(_Kernel):
        @override
        def create_node(
            self, session_id: str, question: str, why_it_matters: str, depends_on: Sequence[str] | None = None
        ) -> ResearchNode:
            created.append(question)
            return _node(f"n-{len(created)}", session_id)

    class _J(JevClient):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return ToolDecision(action="reason", probabilities={}, confidence=1.0)

        @override
        async def adjudicate(self, proposal: object, *a: object, **k: object) -> ToolDecision:
            assert isinstance(proposal, dict) and "analyses" in proposal and "proposals" not in proposal
            calls.append("adjudicate")
            return _invoke("query_finra")

        @override
        async def decide(
            self, state: object, questions: Mapping[str, JSONValue], decision_type: object = None, **kw: object
        ) -> dict[str, dict[str, JSONValue]]:
            return {qid: {"choice": "admit"} for qid in questions}

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {
                "probabilities": {},
                "confidence": 1.0,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
                "evidence_state": "insufficient",
                "decision": "insufficient",
            }

    class _R(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            calls.append("analyze")
            assert isinstance(prompt, str) and "CONTEXT" in prompt
            return {"analyses": [{"nodeId": "n1"}], "evidenceRequests": []}

        @override
        def expand(self, prompt: str, objective_id: str, prior_ids: set[str] | None = None) -> dict[str, list[object]]:
            calls.append("expand")
            assert isinstance(prompt, str) and "n1" in prompt
            assert objective_id == "s1" and prior_ids is not None and "n1" in prior_ids
            return {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}

    def gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        return {"tool_result_id": None}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_K(objective="Objective?"),
                jev=_J(),
                reasoner=_R(model="m", url="u"),
                needle_generate=gen,
                invoke=invoke,
                to_outcome=_to_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=_tool_session(),
            )
        )
    assert calls[0] == "analyze" and "expand" in calls
    assert calls.index("analyze") < calls.index("adjudicate") < calls.index("expand")
    assert created == ["Follow-up?"]
    assert out["admitted"] == 0


def test_other_no_resolve_without_admitted_evidence() -> None:
    """OTHER tool success + sufficient_support but no admitted evidence MUST NOT resolve."""
    resolved: list[str] = []
    assess_calls: list[str] = []

    class _K(_Kernel):
        @override
        def resolve_node(self, session_id: str, node_id: str) -> ResearchNode:
            resolved.append(node_id)
            return super().resolve_node(session_id, node_id)

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return _invoke("mystery_tool")

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            assess_calls.append("assess")
            return await super().assess_result(*a, **k)

    def gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "mystery_tool", "arguments": {}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        return {"note": "uncitable bytes"}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_K(objective="q?"),
                jev=_J(),
                needle_generate=gen,
                invoke=invoke,
                to_outcome=_to_outcome,
                registry=[{"name": "mystery_tool", "parameters": {}}],
                tool_session=_tool_session(),
            )
        )
    assert resolved == []
    assert out["status"] == "blocked" and out["admitted"] == 0


def test_source_domain_maps_unknown_tool_to_other() -> None:
    assert sched.source_domain_for_tool("mystery_tool") == "OTHER"


def test_failed_outcome_never_reaches_assess() -> None:
    assess_calls: list[str] = []
    failed: list[str] = []

    class _K(_Kernel):
        @override
        def fail_job(self, job_id: str, category: str, message: str) -> None:
            failed.append(job_id)
            super().fail_job(job_id, category, message)

    class _J(_JevAdmit):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            assess_calls.append("assess")
            return await super().assess_result(*a, **k)

    def _bad_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        return _outcome(error="provider blew up")

    def gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        return {"tool_result_id": "s1:tr:x"}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_K(objective="q?"),
                jev=_J(),
                needle_generate=gen,
                invoke=invoke,
                to_outcome=_bad_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=_tool_session(),
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
        jev: JevClient,
        kernel: sched._Kernel,
        sid: str,
        nid: str,
        session: Mapping[str, JSONValue],
        node: object,
        registry: Sequence[Mapping[str, JSONValue]],
        ctx_ev: Sequence[JSONValue],
        attempts: Sequence[Mapping[str, JSONValue]],
    ) -> tuple[str, ToolDecision]:
        seen.append([str(e.get("name")) for e in registry])
        return await real_select(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts)

    class _JLoop(JevClient):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return _invoke("get_reg_sho_volume")

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
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

    _gen_n = {"n": 0}

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        _gen_n["n"] += 1
        return {"tool": "get_reg_sho_volume", "arguments": {"round": _gen_n["n"]}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        return {"tool_result_id": "s1:tr:x"}

    def fake_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        calls["n"] += 1
        return _outcome("window words here", error=None if calls["n"] == 3 else "finra downstream blew up")

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 7):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_Kernel(),
                jev=_JLoop(),
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=fake_outcome,
                registry=[
                    {"name": "get_reg_sho_volume", "parameters": {}},
                    {"name": "query_finra", "parameters": {}},
                ],
                tool_session=_tool_session(),
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
    kernel = _Kernel({"s1:tr:abc": dict(_FINRA_RESULT)}, objective="q?")

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("400 max_tokens_exceeded: decide sidecar payload too large")
            return await super().select_tool(*a, **k)

    def gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        return dict(_FINRA_RESULT)

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 5):
        out = asyncio.run(
            sched.run_node(
                _node(),
                "s1",
                kernel=kernel,
                jev=_J(),
                needle_generate=gen,
                invoke=invoke,
                to_outcome=_to_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=_tool_session(),
            )
        )
    assert out["status"] == "resolved" and out["admitted"] == 1
    assert "node_failure" not in [dtype for dtype, _ in kernel.decisions]
    attempts = out["attempts"]
    assert isinstance(attempts, list)
    notes = [a.get("error") for a in attempts if isinstance(a, dict) and a.get("error")]
    assert any(str(n).startswith("select failed (400 max_tokens") for n in notes)


def test_select_raises_thrice_yields_blocked_terminal() -> None:
    """3 consecutive select failures block visibly incomplete, never failed."""
    blocked: list[str] = []

    class _K(_Kernel):
        @override
        def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
            blocked.append(reason)
            return super().block_node(session_id, node_id, reason)

    class _JDown(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            raise RuntimeError("400 max_tokens_exceeded: decide sidecar payload too large")

    kernel = _K(objective="q?")
    out = asyncio.run(sched.run_node(_node(), "s1", kernel=kernel, jev=_JDown()))
    assert out["status"] == "blocked" and "node_failure" not in [dtype for dtype, _ in kernel.decisions]
    assert out.get("incomplete_guard") is True
    attempts = out["attempts"]
    assert isinstance(attempts, list) and len(attempts) == 3 and blocked != []


def test_select_fail_trim_keeps_ctx_bounded() -> None:
    """A select retry after 400 sends a smaller state: last ~5 ctx items."""
    many = [
        {"tool": "get_sec_filing", "job_id": f"job-{i}", "outcome_summary": f"obs {i}", "error": None} for i in range(8)
    ]
    assert len(sched._context_evidence([], many)) == 8
    assert len(sched._context_evidence([], many, cap_last=5)) == 5
    seen: list[int] = []

    kernel = _Kernel({"s1:tr:abc": {"tool_result_id": "s1:tr:abc", "records": [], "briefing": "B"}}, objective="q?")

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            ctx = a[3] if len(a) > 3 else k.get("ctx_evidence", [])
            seen.append(len(ctx) if isinstance(ctx, list) else -1)
            if len(seen) == 1:
                raise RuntimeError("400 max_tokens_exceeded: decide sidecar payload too large")
            return await super().select_tool(*a, **k)

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        return {"tool_result_id": "s1:tr:abc", "records": [], "briefing": "B"}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 3):
        out = asyncio.run(
            sched.run_node(
                _node(),
                "s1",
                kernel=kernel,
                jev=_J(),
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=_to_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=_tool_session(),
            )
        )
    assert out["status"] == "resolved"
    assert len(seen) >= 2 and seen[1] <= 5


def test_accession_carry_fills_from_packet_context() -> None:
    """Open-step {} args + packet accession in context -> args carry the accession."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_filing", "arguments": {}, "reasoning": "r"}

    async def fake_doc_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_document", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    filing_evidence: list[JSONValue] = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), filing_evidence, [], None)
    )
    assert args.get("accession_no") == acc
    doc_evidence: list[JSONValue] = [
        {"source_handle": {"accession_no": acc, "document_name": "nvda-10k.htm"}, "outcome_summary": f"read {acc}"}
    ]
    doc_args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_doc_gen, "get_sec_document", [], session, _node(), doc_evidence, [], None)
    )
    assert doc_args.get("accession_no") == acc
    assert doc_args.get("document_name") == "nvda-10k.htm"


def test_accession_carry_withheld_none_still_fills_from_packet() -> None:
    """Needle withholds (None) on ungrounded open-step + packet accession -> carry fills, no mismatch."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: no grounded accession"}

    session = {"session_id": "s1", "objective": "q?"}
    evidence: list[JSONValue] = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _, _ = asyncio.run(
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

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_filing", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), [], [], None)
    )
    assert "accession_no" not in args


def test_runtime_mismatch_carry_returns_packet_accession() -> None:
    """Server-side RuntimeError mismatch + packet accession -> carried args, no raise."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle arguments.generate failed (needle tool mismatch: JEV=x Needle=y)")

    session = {"session_id": "s1", "objective": "q?"}
    evidence: list[JSONValue] = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), evidence, [], None)
    )
    assert args.get("accession_no") == acc


def test_runtime_mismatch_no_packet_reraises() -> None:
    """Server-side RuntimeError mismatch + no packet -> same raise as today."""

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
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
    seen: dict[str, object] = {}

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        seen.update(kw)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    asyncio.run(sched._generate_tool_arguments(fake_gen, "list_sec_filings", [], session, _node(), [], [], None))
    ctx = seen.get("context")
    assert isinstance(ctx, dict)
    hint = str(ctx.get("grounding_hint", ""))
    assert "accession_no only from packet" in hint and "never org words" in hint and "singular YYYY-MM-DD" in hint


def test_context_carries_scope_and_today() -> None:
    """Needle context carries resolved temporal_scope + today_utc."""
    seen: dict[str, object] = {}

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        seen.update(kw)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "reasoning": "r"}

    scope: dict[str, JSONValue] = {"mode": "range", "start": "2025-04-01", "end": "2025-06-30"}
    session: dict[str, JSONValue] = {"session_id": "s1", "objective": "q?", "temporal_scope": scope}
    asyncio.run(sched._generate_tool_arguments(fake_gen, "list_sec_filings", [], session, _node(), [], [], None))
    ctx = seen.get("context")
    assert isinstance(ctx, dict)
    assert ctx.get("temporal_scope") == scope and isinstance(ctx.get("today_utc"), str) and len(ctx["today_utc"]) == 10
    assert str(seen.get("objective", "")) == "q?"


def test_generate_tool_arguments_objective_raw_verbatim() -> None:
    """Needle-path objective passes verbatim; no [Today UTC ...] stamp (prompt-only date)."""
    seen: dict[str, object] = {}

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
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

    class _Hidden(dict[str, JSONValue]):
        @override
        def __repr__(self) -> str:
            return "packet(hidden)"

    acc = "0001628280-26-044069"
    packet = _Hidden({"nested": [{"accession_no": acc}]})
    assert repr(packet) == "packet(hidden)"
    found, _ = sched._scan_packet_accession(packet)
    assert found == acc


def test_identical_failures_break_with_guided_message() -> None:
    """3 identical failing calls short-circuit duplicates instead of burning 10 rounds."""

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {"ticker": "X"}, "reasoning": "r"}

    class _JLoop(JevClient):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return _invoke("query_finra")

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        return {"tool_result_id": "s1:tr:x"}

    def bad_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        return _outcome("x", error="boom provider down detail identical")

    out = asyncio.run(
        sched._run_node(
            _node(),
            session_id="s1",
            kernel=_Kernel(),
            jev=_JLoop(),
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=bad_outcome,
            registry=[{"name": "query_finra", "parameters": {}}],
            tool_session=_tool_session(),
        )
    )
    attempts = out["attempts"]
    assert out["status"] == "blocked" and out["incomplete_guard"] is True and isinstance(attempts, list)
    assert any(isinstance(a, dict) and "duplicate" in str(a.get("error")) for a in attempts)
    assert (
        sched._identical_failure_break(
            [
                {"tool": "t", "arguments": {"a": 1}, "error": "boom x"},
                {"tool": "t", "arguments": {"a": 1}, "error": "boom x"},
                {"tool": "t", "arguments": {"a": 1}, "error": "boom x"},
            ]
        )
        is not None
    )


def test_force_open_registry_carry_tools_on_nav_packet() -> None:
    """Nav-only packet (accession, zero admissions) restricts selection to carry tools."""
    acc = "0001628280-26-044069"
    registry: list[dict[str, JSONValue]] = [
        {"name": "search_sec_filings", "parameters": {}},
        {"name": "get_sec_filing", "parameters": {}},
        {"name": "query_finra", "parameters": {}},
    ]
    attempts: list[dict[str, JSONValue]] = [{"tool": "search_sec_filings", "outcome_summary": f"filing {acc} 10-K"}]
    forced = sched._force_open_registry(registry, attempts, [], 0)
    assert forced is not None and [e.get("name") for e in forced] == ["get_sec_filing"]
    assert sched._force_open_registry(registry, attempts, [], 1) is None
    assert (
        sched._force_open_registry(registry, [{"tool": "search_sec_filings", "outcome_summary": "no hits"}], [], 0)
        is None
    )


def test_relative_tradedate_scrubbed_to_wtd_omit() -> None:
    """Phrase tradeDate on get_reg_sho_volume scrubs to omit (WTD path); valid dates pass through."""

    async def fake_phrase(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "AAPL", "tradeDate": "this week"},
            "reasoning": "r",
        }

    async def fake_valid(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "AAPL", "tradeDate": "2026-09-22"},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "Apple this week?"}
    scrubbed, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_phrase, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert scrubbed == {"ticker": "AAPL"}
    kept, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_valid, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert kept == {"ticker": "AAPL", "tradeDate": "2026-09-22"}


def test_sec_withhold_falls_back_to_objective_ticker_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle withhold (None) on SEC tools seeds explicit objective ticker, latest (no dates)."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    listed, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert listed == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    searched, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "search_sec_filings", [], session, _node(), [], [], None)
    )
    assert searched["ticker"] == "NVDA" and "start_date" not in searched and "end_date" not in searched


def test_sec_withhold_without_ticker_still_raises() -> None:
    """No explicit ticker token (first-word fallback forbidden) -> mismatch still raises."""

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "Growth slowed last quarter — which segment drove it?"}
    try:
        asyncio.run(sched._generate_tool_arguments(fake_none, "list_sec_filings", [], session, _node(), [], [], None))
    except ValueError as exc:
        assert "needle tool" in str(exc)
    else:
        raise AssertionError("withhold without explicit ticker must still raise mismatch")


def test_sho_withhold_falls_back_to_both_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle withhold (None) on SHO seeds ticker+company_name from for-<Company>."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert args == {"ticker": "AAPL", "company_name": "Apple"}


def test_with_company_seeds_ticker_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """'What is happening with Oracle now' resolves via the EDGAR index (live oracle-query defect)."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Oracle": "ORCL", "oracle": "ORCL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
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

    async def fake_placeholder(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_placeholder, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}


def test_sho_org_ticker_remaps_from_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle ticker FINRA + company Apple remaps to AAPL from the objective."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_org(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "FINRA", "company_name": "Apple"},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_org, "get_reg_sho_volume", [], session, _node(), [], [], None)
    )
    assert args == {"ticker": "AAPL", "company_name": "Apple"}


def test_repeat_8k_swaps_to_packet_10q() -> None:
    """Same 8-K accession twice on quarterly revenue swaps to the packet 10-Q."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_8k(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_8k, "get_sec_filing", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq


def test_withhold_carry_repairs_query_and_swaps_10q() -> None:
    """Needle withhold with a carried 8-K repeat gets query seed + 10-Q swap."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "get_sec_document", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq
    assert args.get("query") == "revenue increased"


def test_repeat_8k_error_swaps_to_packet_10q() -> None:
    """Same 8-K twice with query errors on revenue swaps to the packet 10-Q."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_8k(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": "query not found"},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": "query not found"},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_8k, "get_sec_document", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq


def test_revenue_document_seeds_query() -> None:
    """get_sec_document without query on quarterly revenue seeds a revenue query."""

    async def fake_doc(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_document", "arguments": {"accession_no": "0001045810-26-000075"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_doc, "get_sec_document", [], session, _node(), [], [], None)
    )
    assert args.get("query") == "revenue increased"


def test_stale_as_of_dropped_dateless_objective() -> None:
    """Q4 shape: invented as_of on a dateless accession query scrubs to omit."""

    async def fake_stale(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079, give me the filing metadata (form, dates, filer)",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_stale, "get_sec_filing", [], session, _node(), [], [], None)
    )
    assert args == {"accession_no": "0000320193-25-000079"}


def test_explicit_as_of_kept_when_in_objective() -> None:
    """Explicit YYYY-MM-DD in the objective keeps as_of (latest-available otherwise)."""

    async def fake_kept(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079 as of 2025-09-27, give me the filing metadata",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_kept, "get_sec_filing", [], session, _node(), [], [], None)
    )
    assert args == {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"}


def test_session_as_of_kept_when_datetime() -> None:
    """Session as_of from a stored datetime (ISO-serialized by the store) keeps the matching cutoff."""
    from datetime import UTC, datetime

    async def fake_kept(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"},
            "reasoning": "r",
        }

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079, give me the filing metadata",
        "as_of": datetime(2025, 9, 27, 12, 0, 0, tzinfo=UTC).isoformat(),
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_kept, "get_sec_filing", [], session, _node(), [], [], "2025-09-27")
    )
    assert args == {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"}


def test_garbage_identifier_reseeded_from_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle identifier F1/TODAY/SEC on an Apple query reseeds to AAPL."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_garbage(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "F1"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "List Apple's most recent 10-K and 10-Q filings."}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_garbage, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "AAPL"


def test_identifier_repair_keeps_valid_cik_and_case() -> None:
    """CIK digits and lowercase ticker pass through; only EDGAR-unresolvable garbage reseeds."""

    async def fake_cik(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "0000320193"}, "reasoning": "r"}

    async def fake_lower(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "aapl"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "List Apple's most recent 10-K and 10-Q filings."}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_cik, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "0000320193"
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_lower, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "aapl"


def test_session_control_carries_sid_without_needle() -> None:
    """Session-control tools carry caller sid and never call Needle (sid-carry tuple fix)."""

    def boom_gen(**kw: object) -> dict[str, JSONValue]:
        raise AssertionError("sid-carry must not call Needle")

    session = {"session_id": "s1", "objective": "q?"}
    for tool in ("research_resume", "research_status", "research_cancel"):
        args, reasoning, withheld = asyncio.run(
            sched._generate_tool_arguments(boom_gen, tool, [], session, _node(), [], [], None)
        )
        assert args == {"session_id": "s1"}
        assert withheld is False
        assert "carried" in reasoning

    invoked: list[dict[str, JSONValue]] = []

    def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(dict(args))
        return {"ok": True}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="research_status",
            node=_node(),
            session={"session_id": "s1", "objective": "q?"},
            registry=[{"name": "research_status", "parameters": {}}],
            evidence=[],
            attempts=[],
            kernel=_Kernel(),
            needle_generate=boom_gen,
            invoke=invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert invoked == [{"session_id": "s1"}]
    assert out["arguments"] == {"session_id": "s1"} and out["withheld"] is False


def test_empty_search_withhold_reselects() -> None:
    """Empty search_sec_filings args on a withhold-like objective re-select, never invoke empty."""

    async def fake_empty(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_sec_filings", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove Growth this week?"}
    with pytest.raises(sched._ReselectRequest, match="re-selecting"):
        asyncio.run(
            sched._generate_tool_arguments(fake_empty, "search_sec_filings", [], session, _node(), [], [], None)
        )


def test_ungrounded_search_attempt_reselects() -> None:
    """A reselect attempt settles as bookkeeping, never as a handler error."""

    async def fake_empty(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_sec_filings", "arguments": {}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        raise AssertionError("ungrounded search must not invoke")

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
            invoke=invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert out["tool"] == "search_sec_filings" and out["error_type"] == "reselect"
    settled, done = sched._settle_attempt(kernel, {**out, "outcome": None})
    assert done is True and settled is not None and settled["tool"] == "search_sec_filings"


def test_needle_carry_fails_loud_after_two() -> None:
    """Two carried-accession fallbacks then a third needle outage raises, never carries again (D)."""
    carried = {"reasoning": "carried accession from packet after needle failure"}

    async def boom(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle worker closed")

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    with pytest.raises(RuntimeError, match="carried fallbacks used"):
        asyncio.run(
            sched._generate_tool_arguments(
                boom, "get_sec_filing", [], session, _node(), [], [{**carried}, {**carried}], None
            )
        )


def test_accession_family_break_across_tools() -> None:
    """Same accession failing across get_sec_filing/get_sec_document breaks as one family (C)."""
    acc = "0001193125-09-214859"
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc}, "error": "boom-a"},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc}, "error": "boom-b"},
    ]
    guided = sched._accession_family_break(attempts)
    assert guided is not None and guided["error_type"] == "invalid_tool_arguments"
    error = guided["error"]
    assert isinstance(error, str) and acc in error


def test_find_sec_entities_seeded_from_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_sec_entities seeds a grounded query (ticker-first) instead of erroring on empty args."""
    import app.tools as _tools
    from app.research import scheduler as _sched

    def _resolve(name: str) -> str | None:
        return {"Tesla": "TSLA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    seeded = _sched._fallback_sec_args(
        "find_sec_entities",
        "Did Tesla insiders actually sell shares last quarter, or only file planned-sale notices?",
    )
    assert seeded == {"query": "TSLA"}


def test_sync_invoke_overlaps_not_serial() -> None:
    """Two 0.3s sync SEC tools finish near the slower time, never the sum (A1)."""
    import time as _time

    def slow(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        _time.sleep(0.3)
        return {"ok": True, "tool_result_id": "s1:tr:x"}

    def to_ok(name: str, result: Mapping[str, object]) -> ToolOutcome:
        return _outcome()

    async def _both() -> None:
        await asyncio.gather(
            *(
                sched._invoke_attempt_tool(
                    slow,
                    to_ok,
                    "list_sec_filings",
                    {},
                    _tool_session(),
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
    bad: dict[str, JSONValue] = {"tool": "search_web", "error": "needle down", "arguments": {}}
    assert sched._needle_error_streak([dict(bad), dict(bad)]) == 2
    assert sched._needle_error_streak([dict(bad), dict(bad), dict(bad)]) == 3
    grounded = dict(bad, arguments={"query": "q"})
    assert sched._needle_error_streak([dict(bad), dict(bad), grounded]) == 0


@pytest.mark.parametrize(
    ("kernel_cls", "outcomes", "admitted_ids"),
    [
        (_Kernel, {"admitted": 1, "duplicate": 0, "not_citable": 1, "error": 1, "timeout": 0}, ["ev-0"]),
        (_DupKernel, {"admitted": 0, "duplicate": 1, "not_citable": 1, "error": 1, "timeout": 0}, []),
    ],
)
def test_intake_round_admits_citable_without_jev_verdict(
    monkeypatch: pytest.MonkeyPatch, kernel_cls: type[_Kernel], outcomes: dict[str, int], admitted_ids: list[str]
) -> None:
    """_intake_round admits citable bytes by code; uncitable/error close jobs (E1); duplicates never count."""
    from app.research import kernel_worker as kw

    kernel = kernel_cls()
    kernel.tool_results["s1:tr:w"] = {
        "tool_result_id": "s1:tr:w",
        "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
    }

    async def fake_attempt(
        tool: str, args: dict[str, JSONValue], sid: str, *a: object, **k: object
    ) -> dict[str, object]:
        if tool == "search_web":
            return {
                "tool": tool,
                "arguments": args,
                "job_id": "job-web",
                "result": {
                    "tool_result_id": "s1:tr:w",
                    "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
                },
                "outcome": _outcome("web bytes here"),
                "outcome_summary": "web bytes here",
                "error": None,
            }
        if tool == "get_sec_document":
            return {
                "tool": tool,
                "arguments": args,
                "job_id": "job-doc",
                "result": {"tool_result_id": "s1:tr:d"},
                "outcome": _outcome(""),
                "outcome_summary": "",
                "error": None,
            }
        return {"tool": tool, "arguments": args, "job_id": "job-bad", "error": "boom", "outcome": None}

    async def fake_chain(*a: object, **k: object) -> list[tuple[str, dict[str, JSONValue], dict[str, object], float]]:
        return [
            ("list_sec_filings", {"ticker": "ORCL"}, await fake_attempt("list_sec_filings", {}, "s1"), 5.0),
            ("get_sec_document", {"accession_no": "x"}, await fake_attempt("get_sec_document", {}, "s1"), 5.0),
        ]

    monkeypatch.setattr(kw, "_intake_attempt", fake_attempt)
    monkeypatch.setattr(kw, "_intake_ticker_chain", fake_chain)

    def one_evidence(session_id: str, kernel: sched._Kernel) -> list[dict[str, JSONValue]]:
        return [{"evidence_id": "ev-0"}]

    monkeypatch.setattr(sched, "_load_evidence", one_evidence)
    admitted, _raw, stats = asyncio.run(
        kw._intake_round("orcl manjure", ["ORCL"], "s1", {}, [], None, kernel, None, None)
    )
    assert stats["outcomes"] == outcomes
    assert stats["admitted_ids"] == admitted_ids and stats["admitted"] == 1 and admitted == [{"evidence_id": "ev-0"}]


def test_intake_round_budget_timeout_records_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pending budget tasks synthesize timeout records with wall_ms (A2)."""
    from app.research import kernel_worker as kw

    async def slow_chain(*a: object, **k: object) -> list[tuple[str, dict[str, JSONValue], dict[str, object], float]]:
        await asyncio.sleep(60)
        return []

    monkeypatch.setattr(kw, "_intake_ticker_chain", slow_chain)
    monkeypatch.setattr(kw, "_INTAKE_BUDGET_S", 0.05)

    def no_evidence(session_id: str, kernel: sched._Kernel) -> list[dict[str, JSONValue]]:
        return []

    monkeypatch.setattr(sched, "_load_evidence", no_evidence)


def test_sec_thread_cap_survives_four_concurrent_calls() -> None:
    """4 concurrent SEC thread calls complete; double-acquire would deadlock (bug 1)."""
    import time

    async def _main() -> list[str]:
        async def _call(i: int) -> str:
            out = await asyncio.wait_for(sched._sec_thread_call(lambda: (time.sleep(0.1), f"r{i}")[1]), timeout=10)
            assert isinstance(out, str)
            return out

        return await asyncio.gather(*(_call(i) for i in range(5)))

    assert sorted(asyncio.run(_main())) == ["r0", "r1", "r2", "r3", "r4"]


def test_sec_thread_call_timeout_returns_promptly() -> None:
    """A 4s SEC call under a 0.5s timeout raises at ~0.5s, not 4s (executor fix)."""
    import time

    async def _main() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sched._sec_thread_call(lambda: time.sleep(4)), timeout=0.5)

    start = time.perf_counter()
    asyncio.run(_main())
    assert time.perf_counter() - start < 2.0


def test_sec_pool_caps_concurrency_at_four() -> None:
    """12 submitted SEC calls peak at exactly 4 concurrent workers."""
    import threading
    import time

    peak = {"n": 0, "live": 0}
    lock = threading.Lock()
    full = threading.Event()

    def _work(i: int) -> int:
        with lock:
            peak["live"] += 1
            peak["n"] = max(peak["n"], peak["live"])
            if peak["live"] >= 4:
                full.set()
        try:
            assert full.wait(timeout=10)
            time.sleep(0.05)
            return i
        finally:
            with lock:
                peak["live"] -= 1

    async def _main() -> list[int]:
        results = await asyncio.gather(*[sched._sec_thread_call(lambda i=i: _work(i)) for i in range(12)])
        return [r for r in results if isinstance(r, int)]

    assert sorted(asyncio.run(_main())) == list(range(12))
    assert peak["n"] == 4


def test_session_run_survives_second_event_loop() -> None:
    """Two asyncio.run sessions with 3 contending nodes; loop-bound cap would fail run 2 (bug 2)."""
    from unittest import mock

    class _KS(_Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.seen: set[str] = set()

        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            if session_id in self.seen:
                return []
            self.seen.add(session_id)
            return [_node(f"n{i}", session_id) for i in range(3)]

    async def done(n: ResearchNode, sid: str, **kw: object) -> dict[str, JSONValue]:
        await asyncio.sleep(0)  # yield so the 3rd node actually waits on the 2-lane cap
        return {"node_id": n.node_id, "status": "resolved", "admitted": 1, "incomplete_guard": False}

    with mock.patch.object(sched, "run_node", done):
        first = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
        second = asyncio.run(sched.run("s2", kernel=_KS(), repo=None))
    first_nodes, second_nodes = first["nodes"], second["nodes"]
    assert first["status"] == "complete" and second["status"] == "complete"
    assert isinstance(first_nodes, list) and isinstance(second_nodes, list)
    assert len(first_nodes) == 3 and len(second_nodes) == 3


def test_needle_streak_ignores_reselect_attempts() -> None:
    """3 routine reselects never trip needle fail-fast (medium 7)."""
    reselect: dict[str, JSONValue] = {
        "tool": "search_sec_filings",
        "arguments": {},
        "error": "ungrounded; re-selecting",
        "error_type": "reselect",
    }
    assert sched._needle_error_streak([dict(reselect) for _ in range(3)]) == 0
    gen_fail: dict[str, JSONValue] = {"tool": "query_finra", "arguments": {}, "error": "needle blew up"}
    assert sched._needle_error_streak([dict(gen_fail) for _ in range(3)]) == 3


def test_no_new_evidence_counts_error_free_stale_rounds() -> None:
    """Stale rule fires on progress-free, error-free rounds; error rounds reset it (loop-breaker lane)."""
    assert sched._NO_EVIDENCE_ROUNDS == 2


def test_cancelled_attempt_fails_job() -> None:
    """wait_for cancel during invoke fails the kernel job, then re-raises."""
    import asyncio

    kernel = _Kernel()

    async def _slow(*a: object, **k: object) -> object:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    async def _main() -> None:
        await asyncio.wait_for(
            sched._attempt_tool(
                tool_name="list_sec_filings",
                node=_node(),
                session={"session_id": "s1", "objective": "q"},
                registry=[],
                evidence=[],
                attempts=[],
                kernel=kernel,
                needle_generate=None,
                invoke=_slow,
                to_outcome=_to_outcome,
                tool_session=_tool_session(),
                as_of=None,
                fixed_arguments={"identifier": "320193"},
            ),
            timeout=0.05,
        )

    with pytest.raises(TimeoutError):
        asyncio.run(_main())
    assert ("fail", "job-0", "timeout") in kernel.calls  # cancelled job fails as timeout, never running


def test_cancelled_generation_fails_job() -> None:
    """wait_for cancel during Needle generation fails the kernel job, then re-raises."""
    import asyncio

    kernel = _Kernel()

    async def _slow(*a: object, **k: object) -> object:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    async def _main() -> None:
        def _never_outcome(n: str, r: Mapping[str, object]) -> ToolOutcome:
            return _outcome()

        await asyncio.wait_for(
            sched._attempt_tool(
                tool_name="list_sec_filings",
                node=_node(),
                session={"session_id": "s1", "objective": "q"},
                registry=[],
                evidence=[],
                attempts=[],
                kernel=kernel,
                needle_generate=_slow,
                invoke=_slow,
                to_outcome=_never_outcome,
                tool_session=_tool_session(),
                as_of=None,
            ),
            timeout=0.05,
        )

    with pytest.raises(TimeoutError):
        asyncio.run(_main())
    assert ("fail", "job-0", "timeout") in kernel.calls  # generation cancel fails as timeout, never running


def test_drive_rounds_emits_parseable_timing_per_round(caplog: pytest.LogCaptureFixture) -> None:
    """One toolflow round_timing line per round with 4 phase ms + ev before/after."""
    import logging

    kernel = _Kernel()
    session = {"session_id": "s1", "objective": "o"}

    async def select_round(jev: object, kernel: object, sid: str, nid: str, *a: object, **k: object) -> object:
        return ("invoke", _invoke("search_web"))

    async def gen(tool: object, schema: object = None, **k: object) -> dict[str, JSONValue]:
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "", "confidence": 1.0}

    async def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, JSONValue]:
        return {"status": "ok", "rows": [{"url": "https://e.com", "title": "t", "passage": "p" * 50}]}

    class _J(JevClient):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {"evidence_state": "insufficient", "continuation": "resolve_node"}

    import app.research.scheduler as _s

    with mock.patch.object(_s, "_MAX_TOOL_ROUNDS", 1), caplog.at_level(logging.INFO):
        asyncio.run(
            _s._drive_rounds(
                _node(),
                session,
                [{"name": "search_web"}],
                kernel,
                _J(),
                "s1",
                "n1",
                {
                    "select_round": select_round,
                    "reasoner": None,
                    "needle_generate": gen,
                    "invoke": invoke,
                    "to_outcome": _to_outcome,
                },
                _tool_session(),
                None,
            )
        )
    lines = [r.getMessage() for r in caplog.records if "round_timing" in r.getMessage()]
    assert len(lines) >= 1
    line = lines[0]
    for key in ("round=", "tool=", "select_ms=", "generate_ms=", "tool_ms=", "assess_ms=", "ev_before=", "ev_after="):
        assert key in line


def test_terminal_select_failure_still_logs_timing(caplog: pytest.LogCaptureFixture) -> None:
    """Third straight select failure emits a round_timing row before the terminal return."""
    import asyncio
    import logging

    import app.research.scheduler as _s

    async def down(*a: object, **k: object) -> object:
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        out = asyncio.run(
            _s._drive_rounds(
                _node(),
                {"session_id": "s1", "objective": "o"},
                [{"name": "search_web"}],
                _Kernel(),
                JevClient(),
                "s1",
                "n1",
                {"select_round": down, "reasoner": None, "needle_generate": None, "invoke": None, "to_outcome": None},
                _tool_session(),
                None,
            )
        )
    reason = out["reason"]
    assert out["status"] == "blocked" and isinstance(reason, str) and "select failed 3x" in reason
    assert sum("round_timing" in r.getMessage() for r in caplog.records) == 3


def test_intake_round_stats_carry_phase_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """_intake_round stats carry sec_ms + web_ms + settle_ms + wall_ms."""
    from app.research import kernel_worker as kw

    kernel = _Kernel()

    async def fake_attempt(
        tool: str, args: dict[str, JSONValue], sid: str, *a: object, **k: object
    ) -> dict[str, object]:
        return {
            "tool": tool,
            "arguments": dict(args),
            "outcome": _outcome(),
            "result": {"bytes": "x" * 100},
            "job_id": "j",
        }

    monkeypatch.setattr(kw, "_intake_attempt", fake_attempt)

    async def _empty_chain(*a: object, **k: object) -> list[tuple[str, dict[str, object], dict[str, object], float]]:
        return []

    monkeypatch.setattr(kw, "_intake_ticker_chain", _empty_chain)
    import app.research.scheduler as _s2

    def no_evidence(session_id: str, kernel: sched._Kernel) -> list[dict[str, JSONValue]]:
        return []

    def no_candidate(*a: object, **k: object) -> None:
        return None

    def web_domain(*a: object, **k: object) -> str:
        return "web"

    monkeypatch.setattr(_s2, "_load_evidence", no_evidence)
    monkeypatch.setattr(_s2, "_evidence_candidate", no_candidate)
    monkeypatch.setattr(_s2, "_attempt_domain", web_domain)

    _admitted, _raw, stats = asyncio.run(
        kw._intake_round("q", [], "s1", {"objective": "o"}, [], object(), kernel, None)
    )
    for key in ("sec_ms", "web_ms", "settle_ms", "wall_ms", "ms", "per_source"):
        assert key in stats


def test_selection_guidance_carries_intake_digest() -> None:
    """Seeded guidance holds intake ids plus summaries and the do-not-repeat line."""
    ev: list[JSONValue] = [{"evidence_id": "ev-1", "content": "Apple 10-K revenue rose"}]
    g = sched._selection_guidance(ev, [{"tool": "search_web", "arguments": {"query": "q"}}])
    instruction = g["instruction"]
    assert g["intake_evidence"] == ["ev-1: Apple 10-K revenue rose"]
    assert isinstance(instruction, str) and "identical arguments" in instruction


def test_duplicate_call_reselects() -> None:
    """Same tool plus equal args returns a duplicate reselect, never invokes."""
    kernel = _Kernel()
    seen: list[str] = []

    def gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        seen.append("invoked")
        return {}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_web",
            node=_node(),
            session={"session_id": "s1", "objective": "q"},
            registry=[],
            evidence=[],
            attempts=[{"tool": "search_web", "arguments": {"query": "q"}}],
            kernel=kernel,
            needle_generate=gen,
            invoke=invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    error = out["error"]
    assert out["error_type"] == "reselect" and isinstance(error, str) and "duplicate" in error
    assert seen == []
    reselect: dict[str, JSONValue] = {"tool": "search_web", "arguments": {}, "error": error, "error_type": "reselect"}
    assert sched._needle_error_streak([reselect]) == 0


def test_no_new_evidence_stops_after_two() -> None:
    """Two rounds without progress block with reason no new evidence."""
    kernel = _Kernel()
    calls = {"n": 0}

    async def select_round(
        jev: object,
        kernel: object,
        sid: str,
        nid: str,
        session: object,
        node: object,
        registry: object,
        ctx_ev: Sequence[JSONValue],
        attempts: object,
    ) -> object:
        calls["n"] += 1
        assert any(isinstance(e, dict) and e.get("type") == "selection_guidance" for e in ctx_ev)
        return ("invoke", _invoke("search_web"))

    async def gen(tool: object, schema: object = None, **k: object) -> dict[str, JSONValue]:
        return {"tool": "search_web", "arguments": {"query": f"q{calls['n']}"}, "reasoning": "", "confidence": 1.0}

    def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, JSONValue]:
        return {}

    class _J(JevClient):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {"evidence_state": "insufficient", "continuation": "continue_research"}

    out = asyncio.run(
        sched._drive_rounds(
            _node(),
            {"session_id": "s1", "objective": "o"},
            [{"name": "search_web"}],
            kernel,
            _J(),
            "s1",
            "n1",
            {
                "select_round": select_round,
                "reasoner": None,
                "needle_generate": gen,
                "invoke": invoke,
                "to_outcome": _to_outcome,
            },
            _tool_session(),
            None,
            None,
        )
    )
    assert out["status"] == "blocked" and out["reason"] == "incomplete: no new evidence"
    assert calls["n"] == 2


def test_round_cap_is_five() -> None:
    """Cap constant is 5."""
    assert sched._MAX_TOOL_ROUNDS == 5
    assert sched._NO_EVIDENCE_ROUNDS == 2


def test_expired_deadline_blocks_before_first_round() -> None:
    """A past deadline_at stops the node with a run-deadline reason, never selecting."""
    import time as _t

    selected: list[str] = []

    async def spy(jev: object, kernel: object, sid: str, nid: str, *a: object, **k: object) -> object:
        selected.append(nid)
        return ("invoke", _invoke("search_web"))

    out = asyncio.run(
        sched._drive_rounds(
            _node(),
            {"session_id": "s1", "objective": "o"},
            [{"name": "search_web"}],
            _Kernel(),
            JevClient(),
            "s1",
            "n1",
            {"select_round": spy, "reasoner": None, "deadline_at": _t.perf_counter() - 1.0},
            _tool_session(),
            None,
        )
    )
    reason = out["reason"]
    assert out["status"] == "blocked" and isinstance(reason, str) and "run deadline" in reason
    assert selected == []


def test_run_honors_caller_deadline_at() -> None:
    """A caller-supplied past deadline_at stops run() before selecting, and reaches nodes."""
    import time as _t

    selected: list[str] = []

    class _K(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node("n1", session_id)]

        @override
        def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
            selected.append(node_id)
            return super().block_node(session_id, node_id, reason)

    out = asyncio.run(sched.run("s1", kernel=_K(), deadline_at=_t.perf_counter() - 1.0))
    assert out["status"] == "incomplete_guard" and "run deadline" in str(out.get("reason"))
    assert out["nodes"] == [] and selected == []


def test_mid_round_timeout_blocks_each_node() -> None:
    """A deadline during the session round records a blocked result per in-flight node."""
    import time as _t

    blocked: list[str] = []

    class _K(_Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.seen = False

        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            if not self.seen:
                self.seen = True
                return [_node("n1", session_id), _node("n2", session_id)]
            return []

        @override
        def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
            blocked.append(node_id)
            return super().block_node(session_id, node_id, reason)

    async def slow(
        node: object, session_id: str, hooks: Mapping[str, object], kernel: sched._Kernel, cap: asyncio.Semaphore
    ) -> dict[str, JSONValue]:
        await asyncio.sleep(60)
        return {"node_id": "?", "status": "resolved"}

    with mock.patch.object(sched, "_run_one_node", slow):
        out = asyncio.run(sched.run("s1", kernel=_K(), deadline_at=_t.perf_counter() + 0.05))
    nodes = out["nodes"]
    assert out["status"] == "incomplete_guard" and "during session round" in str(out.get("reason"))
    assert isinstance(nodes, list)
    rows = [r for r in nodes if isinstance(r, dict)]
    assert len(rows) == len(nodes)
    assert sorted(str(r["node_id"]) for r in rows) == ["n1", "n2"]
    assert all(r["status"] == "blocked" and r.get("incomplete_guard") for r in rows)
    assert sorted(blocked) == ["n1", "n2"]


def test_max_rounds_hook_caps_rounds() -> None:
    """max_rounds=2 with progressing rounds reaches the guard naming the cap."""
    calls = {"n": 0}

    async def select_round(jev: object, kernel: object, sid: str, nid: str, *a: object, **k: object) -> object:
        calls["n"] += 1
        return ("invoke", _invoke("query_finra"))

    async def gen(tool: object, schema: object = None, **k: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    async def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, JSONValue]:
        return {**_FINRA_RESULT, "tool_result_id": f"s1:tr:{calls['n']}"}

    class _J(JevClient):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {"evidence_state": "sufficient_support", "continuation": "continue_research"}

    out = asyncio.run(
        sched._drive_rounds(
            _node(),
            {"session_id": "s1", "objective": "o"},
            [{"name": "query_finra", "parameters": {}}],
            _Kernel({f"s1:tr:{n}": {**_FINRA_RESULT, "tool_result_id": f"s1:tr:{n}"} for n in (1, 2)}),
            _J(),
            "s1",
            "n1",
            {
                "select_round": select_round,
                "reasoner": None,
                "needle_generate": gen,
                "invoke": invoke,
                "to_outcome": _to_outcome,
                "max_rounds": 2,
            },
            _tool_session(),
            None,
        )
    )
    reason = out["reason"]
    assert calls["n"] == 2 and isinstance(reason, str) and "(2 rounds without resolution)" in reason


def test_blocked_terminal_names_non_default_cap() -> None:
    """_blocked_terminal carries the caller's max_rounds, not the module default."""
    out = sched._blocked_terminal("n1", 0, [], 2)
    assert out["reason"] == "incomplete: runtime guard (2 rounds without resolution)"


def test_reasoner_analyze_times_out_slow_call() -> None:
    """A 2s sync analyze with a 0.2s deadline raises TimeoutError in <1s."""
    import time as _time

    class _Slow(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            _time.sleep(2.0)
            return {"analyses": [], "evidenceRequests": []}

    attempts: list[dict[str, JSONValue]] = []
    t0 = _time.perf_counter()
    with pytest.raises(TimeoutError):
        asyncio.run(
            sched._reasoner_analyze(
                _Slow(model="m", url="u"),
                {"session_id": "s1", "objective": "o"},
                _node(),
                [],
                attempts,
                _time.perf_counter() + 0.2,
            )
        )
    assert _time.perf_counter() - t0 < 1.0


def test_reasoner_uses_own_pool_and_deadline() -> None:
    """Sync reasoner calls run on reasoner-* threads and honor the per-call timeout."""
    import threading
    import time as _time

    seen: list[str] = []

    def _slow() -> str:
        seen.append(threading.current_thread().name)
        _time.sleep(2.0)
        return "late"

    t0 = _time.perf_counter()
    with pytest.raises(TimeoutError):
        asyncio.run(sched._call_reasoner_blocking("analyze", _slow, 0.2))
    assert _time.perf_counter() - t0 < 1.0
    assert seen and seen[0].startswith("reasoner")
    # Fast path still works inline on the same pool.
    out = asyncio.run(sched._call_reasoner_blocking("analyze", lambda: "ok", None))
    assert out == "ok"


def test_reason_phase_skips_when_too_little_time() -> None:
    """A deadline 5s away never calls analyze; it adds the 'too little time' attempt."""
    import time as _time

    class _Boom(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            raise AssertionError("analyze must not run")

    attempts: list[dict[str, JSONValue]] = []
    out = asyncio.run(
        sched._reason_phase(
            _Boom(model="m", url="u"),
            {"session_id": "s1", "objective": "o"},
            _node(),
            [],
            attempts,
            JevClient(),
            _Kernel(),
            "s1",
            "n1",
            ToolDecision(action="reason"),
            0,
            _time.perf_counter() + 5.0,
        )
    )
    assert out["done"] is True and out["terminal"] is None
    assert len(attempts) == 1 and "too little time" in str(attempts[0].get("error"))


def test_bounded_reasoner_clamps_real_client_only() -> None:
    """Clients get timeout_s=min(config, remaining) on a copy; no deadline returns the shared client."""
    from app.reasoner_client import ReasonerClient

    client = ReasonerClient(model="m", url="u", api_key="k", timeout_s=120.0)
    bounded = sched._bounded_reasoner(client, 3.0)
    assert isinstance(bounded, ReasonerClient) and bounded.timeout_s == 3.0
    assert client.timeout_s == 120.0  # the shared client is never mutated
    assert sched._bounded_reasoner(client, None) is client


def test_bounded_reasoner_preserves_fake_hooks() -> None:
    """A ReasonerClient fake with per-test hooks stays usable after the timeout clamp."""

    class _FakeReasoner(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            assert prompt
            return {"analyses": [{"nodeId": "n1"}], "evidenceRequests": []}

        @override
        def expand(self, prompt: str, objective_id: str, prior_ids: set[str] | None = None) -> dict[str, list[object]]:
            assert prompt and objective_id
            return {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}

    fake = _FakeReasoner(model="m", url="u", api_key="k", timeout_s=120.0)
    bounded = sched._bounded_reasoner(fake, 3.0)
    assert isinstance(bounded, ReasonerClient) and bounded.timeout_s == 3.0
    assert bounded.analyze("CONTEXT x")["analyses"] == [{"nodeId": "n1"}]
    assert bounded.expand("CONTEXT n1", "s1", {"n1"})["proposals"] == [
        {"question": "Follow-up?", "whyItMatters": "Why"}
    ]


def test_force_open_ignores_intake_evidence() -> None:
    """Intake 8-K accession with no node attempts never forces the carry registry."""
    intake: list[JSONValue] = [{"id": "ev:1", "content": "filed 8-K accession 0000320193-25-000079, see filing"}]
    reg: list[dict[str, JSONValue]] = [
        {"name": "get_sec_filing"},
        {"name": "get_sec_document"},
        {"name": "get_short_interest"},
    ]
    assert sched._force_open_registry(reg, [], intake, 0) is None


def test_force_open_fires_on_node_attempt_accession() -> None:
    """This node's own attempt carrying an accession narrows to the carry tools."""
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "list_sec_filings",
            "arguments": {},
            "outcome_summary": "saw accession 0000320193-25-000079",
            "error": None,
        }
    ]
    reg: list[dict[str, JSONValue]] = [
        {"name": "get_sec_filing"},
        {"name": "get_sec_document"},
        {"name": "get_short_interest"},
    ]
    out = sched._force_open_registry(reg, attempts, [], 0)
    assert out is not None and {e["name"] for e in out} == {"get_sec_filing", "get_sec_document"}
    assert sched._force_open_registry(reg, attempts, [], 1) is None


def test_needle_carry_streak_counts_trailing_carried_fallbacks() -> None:
    """Only the trailing run of carried-accession fallbacks counts; the first other entry stops it."""
    carried: dict[str, JSONValue] = {"reasoning": "carried accession from packet after needle failure"}
    other: dict[str, JSONValue] = {"reasoning": "needle generated"}
    assert sched._needle_carry_streak([]) == 0
    assert sched._needle_carry_streak([carried, other]) == 0
    assert sched._needle_carry_streak([carried]) == 1
    assert sched._needle_carry_streak([carried, carried]) == 2
    assert sched._needle_carry_streak([carried, carried, carried]) == 3
    assert sched._needle_carry_streak([carried, carried, other, carried, carried]) == 2


def test_run_one_node_passes_session_id_and_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The node failure lands on the given kernel under the given session id; the node lacks one."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    kernel = _Kernel()
    node: dict[str, JSONValue] = {"node_id": "n9"}

    async def go() -> dict[str, JSONValue]:
        return await sched._run_one_node(node, "s-given", {"jev": "not-a-jev"}, kernel, asyncio.Semaphore(1))

    out = asyncio.run(go())
    assert out["node_id"] == "n9" and out["status"] == "failed"
    assert [kind for kind, _ in kernel.decisions] == ["node_failure"]
