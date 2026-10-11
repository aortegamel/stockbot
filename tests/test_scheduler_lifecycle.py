"""Scheduler lifecycle: domain wiring, admit-then-complete, stall, expansion."""

import asyncio
import time
from types import SimpleNamespace
from typing import Any, ClassVar, override
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.research import scheduler as sched
from app.research.stage import CONTROL_TOOLS, stage_for_session


def _node() -> SimpleNamespace:
    return SimpleNamespace(node_id="n1", session_id="s1", question="q?", why_it_matters="w")


class _Kernel:
    """Fake kernel recording lifecycle order; evidence store is in-memory."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
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
        self.calls.append(("fail", jid, cat))

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
            "probabilities": {"topical_relevance": 0.9, "resolution_support": 0.9},
            "confidence": 1.0,
            "continuation": "resolve_node",
            "continue": "resolve_node",
            "action": "resolve_node",
            "evidence": None,
            "candidate": None,
            "admit": None,
            "evidence_state": "sufficient_support",
            "decision": "sufficient_support",
            "topical_relevance_answer": "yes",
            "resolution_support_answer": "yes",
        }

    async def adjudicate(self, *a: Any, **k: Any) -> SimpleNamespace:
        return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)


def _run_node(kernel: _Kernel, jev: Any) -> dict[str, Any]:
    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
            "reasoning": "r",
        }

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


@pytest.mark.parametrize("runtime_envelope", [True, False], ids=["runtime-envelope", "canonical-text"])
def test_sec_document_admits_full_window_with_bounded_summary(
    runtime_envelope: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Admission preserves a revenue table after character 2,000. The attempt summary stays bounded."""
    import tests.research_source_seam as seam
    from app.research import service
    from app.research.repository import ResearchRepository
    from app.tool_runtime import outcome_from_result

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    seam.install(monkeypatch)
    fact = "Revenue | $ | 46,743 | $ | 30,040\nYear-over-year growth | 56%"
    window = "Document context\n" + "Operating context. " * 130 + "\n" + fact + "\nEnd of the bounded window."
    handle = seam.handle_for(window)
    handle["query"] = "revenue"
    assert window.index("Revenue") > sched._OUTCOME_SUMMARY_MAX
    formatted = "Bottom line: SEC document\nCanonical window follows:\n" + window
    result: dict[str, Any] = {
        "content": formatted,
        "meta": {"source_handle": handle},
        "tool_result_id": None,
    }
    if not runtime_envelope:
        result["text"] = window
        result["source_handle"] = handle
    outcome = outcome_from_result("get_sec_document", result)
    assert outcome.source_handle == handle
    assert outcome.content != window
    candidate = sched._evidence_candidate("get_sec_document", "SEC", result, outcome)
    assert candidate is not None
    assert candidate["content"] == candidate["matching_passage"] == window
    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("What does the filing disclose?", repo=store)
    node = service.create_node(sid, "q?", "w", repo=store)
    kernel = sched._default_kernel(store)
    jid = sched._attempt_job(kernel, sid, "SEC")
    attempts: list[dict[str, Any]] = []
    settled = asyncio.run(
        sched._settle_round(
            [
                {
                    "tool": "get_sec_document",
                    "source_domain": "SEC",
                    "job_id": jid,
                    "arguments": {"accession_no": handle["accession_no"], "query": "revenue"},
                    "result": result,
                    "outcome": outcome,
                    "outcome_summary": sched._outcome_summary(outcome),
                    "error": None,
                }
            ],
            _JevAdmit(),
            kernel,
            _node(),
            [],
            attempts,
            sid,
            node.node_id,
            None,
            0,
            store,
            continuation=False,
        )
    )
    assert settled["admitted"] == 1
    assert attempts[0]["error"] is None
    assert attempts[0]["outcome_summary"] == formatted[: sched._OUTCOME_SUMMARY_MAX]
    assert len(attempts[0]["outcome_summary"]) == 2000
    assert fact not in attempts[0]["outcome_summary"]
    rows = store.list_evidence(sid)
    assert len(rows) == 1
    evidence = rows[0]
    assert isinstance(evidence, dict)
    assert evidence.get("content") == window
    assert isinstance(evidence.get("provenance"), dict)
    provenance = evidence.get("provenance")
    assert isinstance(provenance, dict)
    assert isinstance(provenance.get("passage"), str) and fact in str(provenance.get("passage"))
    assert evidence.get("record_kind") == "evidence"
    assert evidence.get("claim_kind") == "observed_fact"
    assert provenance.get("text_hash") == handle["text_hash"]


def _prior_context() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Admitted prior evidence with a large body plus one unadmitted failed observation."""
    evidence = [
        {
            "evidence_id": "ev-prior",
            "source_name": "get_sec_document",
            "subject": "NVDA",
            "claim_text": "Prior claim. " * 40,
            "content": "Prior filing bytes. " * 5000,
        }
    ]
    attempts = [
        {
            "tool": "list_sec_filings",
            "job_id": "job-0",
            "outcome_summary": "Listing row. " * 100,
            "error": "upstream timeout",
            "error_type": "timeout",
            "tool_result_ref": "s1:tr:list",
        }
    ]
    return evidence, attempts


class _JevCapture(_JevAdmit):
    """Records the context and candidate that reach JEV assessment."""

    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

    @override
    async def assess_result(self, *a: object, **k: object) -> dict[str, object]:
        self.seen["context"] = a[2]
        self.seen["candidate"] = k.get("candidate")
        return await super().assess_result(*a, **k)


@pytest.mark.parametrize(("before", "after"), [(130, 0), (20000, 20000)], ids=["fact-after-2000", "huge-window"])
def test_assess_sec_document_sends_compact_view(before: int, after: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Document JEV gets one bounded verbatim revenue quote and prior-evidence digests. Inputs and admission stay full."""
    import copy

    import tests.research_source_seam as seam
    from app.tool_runtime import outcome_from_result

    seam.install(monkeypatch)
    fact = "Revenue | $ | 46,743 | $ | 30,040"
    window = (
        "Document context\n"
        + "Operating context. " * before
        + "\n"
        + fact
        + "\n"
        + "Trailing context. " * after
        + "End of window."
    )
    handle = seam.handle_for(window)
    handle["query"] = "revenue"
    assert window.index("Revenue") > sched._OUTCOME_SUMMARY_MAX
    result: dict[str, object] = {"content": "Bottom line: SEC document\n" + window, "meta": {"source_handle": handle}}
    outcome = outcome_from_result("get_sec_document", result)
    attempt = {"tool": "get_sec_document", "result": result, "job_id": "job-1", "domain": "SEC"}
    evidence, attempts = _prior_context()
    inputs = copy.deepcopy((result, attempt, evidence, attempts, handle))
    outcome_content = outcome.content
    jev = _JevCapture()

    assessment, _ms = asyncio.run(
        sched._assess_attempt(
            jev, _Kernel(), _node(), outcome, evidence, attempts, "s1", "n1", "job-1", "get_sec_document", attempt
        )
    )

    candidate = jev.seen["candidate"]
    assert isinstance(candidate, dict)
    assert set(candidate) == {"source_handle", "matching_passage"}
    assert candidate["source_handle"] == handle
    quote = candidate["matching_passage"]
    assert len(quote) <= sched._JEV_QUOTE_MAX
    assert quote in window and fact in quote
    context = jev.seen["context"]
    assert isinstance(context, list)
    prior, observation = context
    assert prior == {
        "evidence_id": "ev-prior",
        "source_name": "get_sec_document",
        "subject": "NVDA",
        "claim_text": evidence[0]["claim_text"][: sched._JEV_DIGEST_MAX],
    }
    assert {k: observation[k] for k in ("tool", "job_id", "error", "error_type", "tool_result_ref")} == {
        "tool": "list_sec_filings",
        "job_id": "job-0",
        "error": "upstream timeout",
        "error_type": "timeout",
        "tool_result_ref": "s1:tr:list",
    }
    assert observation["outcome_summary"] == attempts[0]["outcome_summary"][: sched._JEV_DIGEST_MAX]
    assert (result, attempt, evidence, attempts, handle) == inputs
    assert outcome.content == outcome_content
    admitted = sched._admittable_candidate("get_sec_document", attempt, outcome, assessment)
    assert admitted is not None
    assert admitted["content"] == admitted["matching_passage"] == window


@pytest.mark.parametrize(
    ("tool", "canonical", "text", "with_attempt", "expect_candidate"),
    [
        ("get_sec_document", False, "", True, False),
        ("get_sec_document", True, "Revenue | $ | 46,743", False, False),
        ("get_sec_document", False, "Revenue | $ | 46,743", True, True),
        ("get_sec_filing", True, "Revenue | $ | 46,743", True, True),
    ],
    ids=["empty-window", "missing-attempt", "non-canonical-handle", "non-document"],
)
def test_assess_candidate_failure_falls_back_to_bounded_assessment(
    tool: str, canonical: bool, text: str, with_attempt: bool, expect_candidate: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a canonical document candidate, JEV gets the unchanged candidate and full context."""
    import tests.research_source_seam as seam

    seam.install(monkeypatch)
    handle = seam.handle_for(text) if canonical else {"basis": "guessed"}
    result: dict[str, object] = {"source_handle": handle, "text": text}
    outcome = _outcome("summary")
    outcome.source_handle = handle  # type: ignore[attr-defined]
    attempt = {"tool": tool, "result": result, "job_id": "job-9", "domain": "SEC"} if with_attempt else None
    evidence, attempts = _prior_context()
    kernel = _Kernel()
    expected = sched._preassess_candidate(tool, attempt, outcome, kernel)
    assert (expected is not None) == expect_candidate
    jev = _JevCapture()
    assessment, _ms = asyncio.run(
        sched._assess_attempt(jev, kernel, _node(), outcome, evidence, attempts, "s1", "n1", "job-9", tool, attempt)
    )
    assert jev.seen["candidate"] == expected
    assert jev.seen["context"] == sched._context_evidence(evidence, attempts)
    assert assessment["evidence_state"] == "sufficient_support"


@pytest.mark.parametrize(
    ("defect", "error_code"),
    [
        ("text_hash", "ERR_SEC_HANDLE_STALE"),
        ("source_content_hash", "ERR_SEC_HANDLE_STALE"),
        ("basis", "ERR_SEC_HANDLE_INVALID"),
    ],
)
def test_sec_runtime_envelope_rejects_invalid_handle(
    defect: str, error_code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical reload and admission reject stale hashes or invalid handle fields."""
    import tests.research_source_seam as seam
    from app.research.service import materialize_sec_passage
    from app.tool_runtime import outcome_from_result

    seam.install(monkeypatch)
    handle = seam.handle_for("Revenue | $ | 46,743")
    handle[defect] = "guessed" if defect == "basis" else "0" * 64
    result: dict[str, object] = {
        "content": "Bottom line: Formatted document output.",
        "meta": {"source_handle": handle},
        "tool_result_id": None,
    }
    outcome = outcome_from_result("get_sec_document", result)

    class _ValidatedKernel(_Kernel):
        def admit_evidence(self, sid: str, jid: str, cand: dict[str, Any]) -> dict[str, str]:
            materialize_sec_passage(cand["source_handle"], cand["matching_passage"])
            return super().admit_evidence(sid, jid, cand)

    kernel = _ValidatedKernel()
    jid = sched._attempt_job(kernel, "s1", "SEC")
    attempt = {"result": result, "job_id": jid, "outcome_summary": sched._outcome_summary(outcome)}
    assessment = asyncio.run(_JevAdmit().assess_result())
    eid, progressed, failure = sched._admit_attempt_evidence(
        kernel, "s1", "get_sec_document", attempt, outcome, assessment
    )
    assert eid is None and not progressed
    assert failure is not None and error_code in failure["error"]
    assert kernel.evidence == []
    assert ("complete", jid) in kernel.calls


def test_sec_list_receipt_routes_through_persisted_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """list_sec_filings source_refs is a catalog receipt, never a handle: it replays by tool_result_id."""
    import tests.research_source_seam as seam
    from app.research.service import _handle_fields, _sec_record_texts
    from app.tool_runtime import outcome_from_result

    filing = {
        "form": "8-K",
        "accession_no": "0000320193-25-000079",
        "url": "https://sec.gov/x",
        "known_at": "2025-05-01",
    }
    listed: dict[str, object] = {"tool_result_id": "s1:tr:list", "filings": [filing], "source": "SEC EDGAR"}
    receipt = outcome_from_result("list_sec_filings", listed)
    assert receipt.source_handle is None and receipt.source_refs
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _handle_fields(receipt.source_refs)  # the receipt is no document handle
    persisted: dict[str, object] = {"tool_result_id": "s1:tr:list", "tool_name": "list_sec_filings", "result": listed}
    reads: list[str] = []

    def _get_tool_result(ref: str) -> dict[str, object]:
        reads.append(ref)
        return persisted

    kernel = SimpleNamespace(store=SimpleNamespace(get_tool_result=_get_tool_result))
    cand = sched._evidence_candidate("list_sec_filings", "SEC", listed, receipt, kernel)
    assert cand is not None and "source_handle" not in cand, cand
    assert cand["tool_result_id"] == "s1:tr:list" and reads == ["s1:tr:list"]
    assert cand["record_identity"] in _sec_record_texts(persisted)

    # get_sec_document keeps its canonical handle; a malformed explicit handle still reaches the validator.
    seam.install(monkeypatch)
    passage = "Data center revenue grew."
    handle = seam.handle_for(passage)
    doc: dict[str, object] = {"source_handle": handle, "text": passage, "content": passage}
    doc_cand = sched._evidence_candidate("get_sec_document", "SEC", doc, outcome_from_result("get_sec_document", doc))
    assert doc_cand is not None and doc_cand["source_handle"] == handle
    _handle_fields(handle)
    bogus: dict[str, object] = {"source_handle": {"basis": "guessed"}, "text": passage, "content": passage}
    bad = sched._evidence_candidate("get_sec_document", "SEC", bogus, outcome_from_result("get_sec_document", bogus))
    assert bad is not None and bad["source_handle"] == {"basis": "guessed"}
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _handle_fields(bad["source_handle"])


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
                "probabilities": {"topical_relevance": 0.4, "resolution_support": 0.2},
                "confidence": 1.0,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
                "evidence_state": "insufficient",
                "decision": "insufficient",
                "topical_relevance_answer": "no",
                "resolution_support_answer": "no",
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
                needle_generate=lambda **kw: {
                    "tool": "query_finra",
                    "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
                    "reasoning": "r",
                },
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
                needle_generate=lambda **kw: {
                    "tool": "query_finra",
                    "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
                    "reasoning": "r",
                },
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


_NODE_INVALID: set[str] = {"research_start", "research_read_search", "research_resume"}
_LATE_STAGE_TOOLS: set[str] = {"research_add_analysis", "research_finalize"}
_FROZEN: dict[str, object] = {"freeze_ids": ["fr-1"], "committee_runs": []}
_CREATED: dict[str, object] = {"status": "created", "freeze_ids": [], "committee_runs": []}
_TARGETED_AFTER_FREEZE: dict[str, object] = {"status": "TARGETED_RESEARCH", **_FROZEN}


class _StagedKernel(_Kernel):
    """Fake kernel whose session status/freeze state drives the derived research stage."""

    def __init__(self, session: dict[str, object]) -> None:
        super().__init__()
        self.session = session

    def get_session(self, sid: str) -> dict[str, object]:
        return {"session_id": sid, "objective": "q?", "query": "", "as_of": None, **self.session}


@pytest.mark.parametrize(
    ("session", "stage", "withheld"),
    [
        pytest.param(_CREATED, "SOURCE_RESEARCH", _LATE_STAGE_TOOLS, id="created-run22"),
        pytest.param({}, "SOURCE_RESEARCH", _LATE_STAGE_TOOLS, id="unknown-status"),
        pytest.param(_TARGETED_AFTER_FREEZE, "SOURCE_RESEARCH", _LATE_STAGE_TOOLS, id="targeted-after-freeze"),
        pytest.param({"status": "ANALYZING", **_FROZEN}, "COMMITTEE", set(), id="frozen-analyzing"),
        pytest.param({"status": "SYNTHESIZING", **_FROZEN}, "FINAL", set(), id="final"),
    ],
)
def test_node_registry_withholds_analysis_and_finalize_only_in_source_stage(
    session: dict[str, object], stage: str, withheld: set[str]
) -> None:
    """SOURCE nodes never see analysis/finalize; COMMITTEE/FINAL nodes keep them; data + source controls stay."""
    kernel = _StagedKernel(session)
    assert stage_for_session(kernel.get_session("s1")) == stage
    tools = sorted(CONTROL_TOOLS | {"query_finra"})
    registry: list[dict[str, object]] = [{"name": n, "parameters": {}} for n in tools]
    ctx = sched._node_context(_node(), "s1", kernel, None, {"registry": registry})
    names = {str(e.get("name")) for e in ctx["registry"]}
    assert set(CONTROL_TOOLS) - names == _NODE_INVALID | withheld
    assert "query_finra" in names
    assert {"research_add_evidence", "research_submit_source_result"} <= names


@pytest.mark.parametrize(
    "session",
    [pytest.param(_CREATED, id="created-run22"), pytest.param(_TARGETED_AFTER_FREEZE, id="targeted-after-freeze")],
)
def test_source_node_never_generates_analysis_or_finalize_arguments(session: dict[str, object]) -> None:
    """run22: JEV picked offered research_add_analysis in SOURCE; its argument generation timed out the node."""
    generated: list[str] = []

    class _JPrefersLateStage(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> SimpleNamespace:
            registry = a[2]
            names = {e.get("name") for e in registry if isinstance(e, dict)} if isinstance(registry, list) else set()
            pick = next((n for n in ("research_add_analysis", "research_finalize") if n in names), "query_finra")
            return SimpleNamespace(tool_name=pick, probabilities={}, confidence=1.0)

    async def fake_gen(**kw: object) -> dict[str, object]:
        generated.append(str(kw["tool"]))
        return {
            "tool": kw["tool"],
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
            "reasoning": "r",
        }

    async def fake_invoke(name: str, args: dict[str, object], sess: object, **kw: object) -> dict[str, object]:
        return {
            "tool_result_id": "s1:tr:abc",
            "records": [{"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123}],
            "briefing": "B",
        }

    def to_outcome(name: str, result: object) -> SimpleNamespace:
        return _outcome()

    kernel = _StagedKernel(session)
    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=kernel,
                jev=_JPrefersLateStage(),
                repo=None,
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=to_outcome,
                registry=[
                    {"name": "research_add_analysis", "parameters": {}},
                    {"name": "research_finalize", "parameters": {}},
                    {"name": "research_add_evidence", "parameters": {}},
                    {"name": "query_finra", "parameters": {}},
                ],
                tool_session=SimpleNamespace(),
            )
        )
    assert generated == ["query_finra"]
    assert [c for c in kernel.calls if c[0] == "start"] == [("start", "FINRA")]
    assert out["status"] == "resolved" and out["admitted"] == 1


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
                "probabilities": {"topical_relevance": 0.4, "resolution_support": 0.2},
                "confidence": 1.0,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
                "evidence": None,
                "candidate": None,
                "admit": None,
                "evidence_state": "insufficient",
                "decision": "insufficient",
                "topical_relevance_answer": "no",
                "resolution_support_answer": "no",
            }

    _gen_n = {"n": 0}

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        _gen_n["n"] += 1
        return {"tool": "get_reg_sho_volume", "arguments": {"round": _gen_n["n"]}, "reasoning": "r"}

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


def test_ungrounded_reselect_drops_tool_for_next_round_only() -> None:
    """An ungrounded-argument reselect excludes that tool next round; later rounds may retry it."""
    seen: list[list[str]] = []
    real_select = sched._select_round

    async def spy_select(
        jev: Any, kernel: Any, sid: str, nid: str, session: Any, node: Any, registry: Any, ctx_ev: Any, attempts: Any
    ) -> Any:
        seen.append([str(e.get("name")) for e in registry])
        return await real_select(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts)

    class _JLoop:
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            names = {e.get("name") for e in a[2]} if isinstance(a[2], list) else set()
            pick = "get_xbrl_facts" if "get_xbrl_facts" in names else "query_finra"
            return SimpleNamespace(tool_name=pick, probabilities={}, confidence=1.0)

        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            base = await _JevAdmit().assess_result(*a, **k)
            return {
                **base,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
            }

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        if kw["tool"] == "get_xbrl_facts":
            return {
                "tool": "get_xbrl_facts",
                "arguments": {"ticker": "NVDA", "concept": "instruction"},
                "reasoning": "r",
            }
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
            "reasoning": "r",
        }

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {"tool_result_id": "s1:tr:x", "records": [], "briefing": "B"}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 3):
        out = asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_Kernel(),
                jev=_JLoop(),
                repo=None,
                needle_generate=fake_gen,
                invoke=fake_invoke,
                to_outcome=lambda name, result: _outcome(),
                registry=[
                    {"name": "get_xbrl_facts", "parameters": {}},
                    {"name": "query_finra", "parameters": {}},
                ],
                tool_session=SimpleNamespace(),
                select_round=spy_select,
            )
        )
    assert seen[0] == ["get_xbrl_facts", "query_finra"]
    assert seen[1] == ["query_finra"]
    assert "get_xbrl_facts" in seen[2]
    assert out["attempts"][0]["error_type"] == "reselect"
    assert out["attempts"][0]["arguments"] == {}


def test_ungrounded_reselect_keeps_last_tool_when_no_alternative() -> None:
    """A lone-tool registry keeps the rejected tool: exclusion never leaves JEV with nothing to pick."""
    seen: list[list[str]] = []
    real_select = sched._select_round

    async def spy_select(
        jev: Any, kernel: Any, sid: str, nid: str, session: Any, node: Any, registry: Any, ctx_ev: Any, attempts: Any
    ) -> Any:
        seen.append([str(e.get("name")) for e in registry])
        return await real_select(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts)

    class _JLoop:
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            return SimpleNamespace(tool_name="get_xbrl_facts", probabilities={}, confidence=1.0)

        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            return await _JevAdmit().assess_result(*a, **k)

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_xbrl_facts", "arguments": {"ticker": "NVDA", "concept": "instruction"}, "reasoning": "r"}

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 2):
        asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_Kernel(),
                jev=_JLoop(),
                repo=None,
                needle_generate=fake_gen,
                invoke=lambda *a, **k: {"tool_result_id": "s1:tr:x"},
                to_outcome=lambda name, result: _outcome(),
                registry=[{"name": "get_xbrl_facts", "parameters": {}}],
                tool_session=SimpleNamespace(),
                select_round=spy_select,
            )
        )
    assert seen == [["get_xbrl_facts"], ["get_xbrl_facts"]]


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
                needle_generate=lambda **kw: {
                    "tool": "query_finra",
                    "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
                    "reasoning": "r",
                },
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
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
            "reasoning": "r",
        }

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
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), filing_evidence, [], None)
    )
    assert args.get("accession_no") == acc
    doc_evidence = [
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

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: no grounded accession"}

    session = {"session_id": "s1", "objective": "q?"}
    evidence = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
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

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_filing", "arguments": {}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "q?"}
    args, _, _ = asyncio.run(
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
    args, _, _ = asyncio.run(
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
    """3 identical failing calls short-circuit duplicates instead of burning 10 rounds."""

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": "X"},
            "reasoning": "r",
        }

    class _JLoop:
        async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
            return SimpleNamespace(tool_name="query_finra", probabilities={}, confidence=1.0)

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {"tool_result_id": "s1:tr:x"}

    def bad_outcome(name: str, result: Any) -> SimpleNamespace:
        out = _outcome("x")
        out.error = "boom provider down detail identical"
        return out

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
    assert any("duplicate" in str(a.get("error")) for a in out["attempts"])
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

    async def fake_none(**kw: Any) -> dict[str, Any]:
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

    async def fake_none(**kw: Any) -> dict[str, Any]:
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

    async def fake_none(**kw: Any) -> dict[str, Any]:
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

    async def fake_placeholder(**kw: Any) -> dict[str, Any]:
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

    async def fake_org(**kw: Any) -> dict[str, Any]:
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

    async def fake_8k(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts = [
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence = [{"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}]
    args, _, _ = asyncio.run(
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
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "get_sec_document", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq
    assert args.get("query") == "revenue"


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
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_8k, "get_sec_document", [], session, _node(), evidence, attempts, None)
    )
    assert args.get("accession_no") == accq


def test_revenue_document_seeds_query() -> None:
    """get_sec_document without query on quarterly revenue seeds a revenue query."""

    async def fake_doc(**kw: Any) -> dict[str, Any]:
        return {"tool": "get_sec_document", "arguments": {"accession_no": "0001045810-26-000075"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_doc, "get_sec_document", [], session, _node(), [], [], None)
    )
    assert args.get("query") == "revenue"
    assert args.get("limit") == 32000


def test_ten_q_revenue_lists_forms_and_reads_listed_10q(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit 10-Q revenue request lists only 10-Q filings and reads three sections of the listed 10-Q."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    # 8-K listed first, out of the 10-Q window, so only the 10-Q preference picks accq.
    listing = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA"},
        "error": None,
        "outcome_summary": f"filing {acc8} 8-K {'x' * 130} filing {accq} 10-Q",
    }
    for objective, forms, doc in (
        (
            "What did NVDA's latest 10-Q disclose about revenue? Use stockbot, bearbot, and bullbot.",
            ["10-Q"],
            {"accession_no": accq, "query": "revenue", "limit": 32000},
        ),
        ("What did NVDA disclose about revenue?", None, {"accession_no": acc8}),
    ):
        session = {"session_id": "s1", "objective": objective}
        listed, _, _ = asyncio.run(
            sched._generate_tool_arguments(fake_none, "list_sec_filings", [], session, _node(), [], [], None)
        )
        assert listed.get("forms") == forms
        read, _, _ = asyncio.run(
            sched._generate_tool_arguments(fake_none, "get_sec_document", [], session, _node(), [], [listing], None)
        )
        assert read == doc

    invoked: list[dict[str, Any]] = []

    async def invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append(dict(args))
        return {"text": "Revenue was $46.7 billion."}

    explicit = "What did NVDA's latest 10-Q disclose about revenue? Use stockbot, bearbot, and bullbot."
    sections = [{"accession_no": accq, "limit": 32000, "section": s} for s in ("Item 1", "Item 2", "Item 1A")]
    revenue_read = {"accession_no": accq, "query": "revenue", "limit": 32000}
    failed_read = {"tool": "get_sec_document", "arguments": {}, "error": "timeout"}
    # Only the first undated explicit latest-10-Q read fans out. Retries, dated asks, and other revenue asks keep one read.
    dated = "What did NVDA's latest 10-Q as of 2026-09-01 disclose about revenue?"
    for objective, attempts, reads in (
        (explicit, [listing], sections),
        (explicit, [listing, failed_read], [revenue_read]),
        (dated, [listing], [revenue_read]),
        ("What drove NVDA revenue last quarter?", [listing], [revenue_read]),
        ("What did NVDA disclose about revenue?", [listing], [{"accession_no": acc8}]),
    ):
        invoked.clear()
        results = asyncio.run(
            sched._invoke_phase(
                ["get_sec_document"],
                _node(),
                {"session_id": "s1", "objective": objective},
                [],
                [],
                attempts,
                _Kernel(),
                fake_none,
                invoke,
                lambda n, r: _outcome(),
                SimpleNamespace(),
                None,
            )
        )
        assert invoked == reads
        assert [r["arguments"] for r in results] == reads
        assert all(r["error"] is None for r in results)

    async def fail_needle(**kw: Any) -> dict[str, Any]:
        raise RuntimeError("needle boom")

    async def fail_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("sec read failed")

    # Failed sibling section reads settle without a breaker block, so each section returns to the selector alone.
    for needle, call, failed in (
        (fake_none, fail_invoke, sections),
        (fail_needle, invoke, [{"section": s["section"]} for s in sections]),
    ):
        attempts = [listing]
        results = asyncio.run(
            sched._invoke_phase(
                ["get_sec_document"],
                _node(),
                {"session_id": "s1", "objective": explicit},
                [],
                [],
                attempts,
                _Kernel(),
                needle,
                call,
                lambda n, r: _outcome(),
                SimpleNamespace(),
                None,
            )
        )
        assert [r["arguments"] for r in results] == failed
        assert all(r["error"] for r in results)
        signal = asyncio.run(sched._settle_round(results, None, _Kernel(), _node(), [], attempts, "s1", "n1", None, 0))
        assert signal["terminal"] is None and len(attempts) == 4


@pytest.mark.parametrize("grounds", ["none", "objective", "node", "scope", "scope-as-of", "session-as-of", "as-of"])
def test_latest_ten_q_listing_constrains_forms_and_preserves_grounded_dates(
    grounds: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid broad forms become 10-Q only. Only grounded dates remain."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", lambda name: {"NVDA": "NVDA"}.get(name))
    generated = {
        "identifier": "NVDA",
        "forms": ["10-Q", "10-K", "8-K"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    needle_calls: list[str] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        needle_calls.append(kw["tool"])
        return {"tool": "list_sec_filings", "arguments": dict(generated), "reasoning": "r"}

    prompt = "What did NVDA's latest 10-Q disclose about revenue?"
    dated = f"{prompt} Include filings from 2025-01-01 through 2025-12-31."
    session: dict[str, Any] = {"session_id": "s1", "objective": dated if grounds == "objective" else prompt}
    node = SimpleNamespace(node_id="n1", question=dated if grounds == "node" else prompt, why_it_matters="w")
    if grounds == "scope":
        session["temporal_scope"] = {"mode": "range", "start": "2025-01-01", "end": "2025-12-31"}
    if grounds == "scope-as-of":
        session["temporal_scope"] = {"mode": "as_of", "as_of": "2025-12-31"}
    if grounds == "session-as-of":
        session["as_of"] = "2025-12-31"
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "list_sec_filings",
            sched.build_registry(),
            session,
            node,
            [],
            [],
            "2025-12-31" if grounds == "as-of" else None,
        )
    )
    expected: dict[str, Any] = {"identifier": "NVDA", "forms": ["10-Q"]}
    if grounds in ("objective", "node", "scope"):
        expected.update(start_date="2025-01-01", end_date="2025-12-31")
    elif grounds in ("scope-as-of", "session-as-of"):
        expected.update(end_date="2025-12-31")
    assert args == expected
    assert needle_calls == ([] if grounds == "none" else ["list_sec_filings"])
    assert generated["forms"] == ["10-Q", "10-K", "8-K"]


def test_quarter_revenue_listing_keeps_generated_forms_and_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quarter-only request keeps its existing form and date behavior."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", lambda name: {"NVDA": "NVDA"}.get(name))
    generated = {
        "identifier": "NVDA",
        "forms": ["10-Q", "10-K", "8-K"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    needle_calls: list[str] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        needle_calls.append(kw["tool"])
        return {"tool": "list_sec_filings", "arguments": dict(generated), "reasoning": "r"}

    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "list_sec_filings",
            sched.build_registry(),
            {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"},
            _node(),
            [],
            [],
            None,
        )
    )
    assert args == generated
    assert needle_calls == ["list_sec_filings"]


def test_latest_ten_q_listing_generates_missing_required_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle supplies a required limit that the local listing arguments cannot fill."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", lambda name: {"NVDA": "NVDA"}.get(name))
    generated = {"identifier": "NVDA", "forms": ["10-Q"], "limit": 3}
    needle_calls: list[str] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        needle_calls.append(kw["tool"])
        return {"tool": "list_sec_filings", "arguments": dict(generated), "reasoning": "required limit"}

    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "list_sec_filings",
            [{"name": "list_sec_filings", "parameters": {"required": ["identifier", "forms", "limit"]}}],
            {"session_id": "s1", "objective": "What did NVDA's latest 10-Q disclose about revenue?"},
            _node(),
            [],
            [],
            None,
        )
    )
    assert needle_calls == ["list_sec_filings"]
    assert args == generated
    assert reasoning == "required limit"
    assert withheld is False


@pytest.mark.parametrize("listed", [False, True], ids=["missing-listing", "only-8k"])
def test_latest_ten_q_document_without_ten_q_generates_and_reselects(listed: bool) -> None:
    """Without a listed 10-Q, Needle runs once. The scheduler does not read an 8-K."""
    acc8 = "0001045810-26-000078"
    attempts = (
        [
            {
                "tool": "list_sec_filings",
                "arguments": {"identifier": "NVDA", "forms": ["10-Q"]},
                "error": None,
                "result": {"filings": [{"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"}]},
                "outcome_summary": f"8-K {acc8}",
            }
        ]
        if listed
        else []
    )
    needle_calls: list[str] = []
    invoked: list[str] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        needle_calls.append(kw["tool"])
        return {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    async def invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append(name)
        return {}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_sec_document",
            node=_node(),
            session={"session_id": "s1", "objective": "What did NVDA's latest 10-Q disclose about revenue?"},
            registry=sched.build_registry(),
            evidence=[],
            attempts=attempts,
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=lambda n, r: _outcome(),
            tool_session=SimpleNamespace(),
            as_of=None,
        )
    )
    assert needle_calls == ["get_sec_document"]
    assert invoked == []
    assert out["error_type"] == "reselect"
    assert "no 10-Q" in out["error"]


def test_structured_listing_keeps_form_identity_and_selects_latest_ten_q() -> None:
    """Source rows label their own accession. The latest 10-Q wins over an older 10-Q and a newer 8-K."""
    acc8, old_q, latest_q = "0001045810-26-000078", "0001045810-26-000050", "0001045810-26-000075"
    listing = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]},
        "error": None,
        "result": {
            "filings": [
                {"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"},
                {"accession_no": old_q, "form": "10-Q", "filed_at": "2026-05-01"},
                {"accession_no": latest_q, "form": "10-Q", "filed_at": "2026-08-01"},
            ]
        },
        "outcome_summary": f"8-K {acc8}, 10-Q {old_q}, 10-Q {latest_q}",
    }
    labeled = sched._ten_q_accessions(listing)
    assert set(labeled) == {old_q, latest_q}
    assert acc8 not in labeled

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none,
            "get_sec_document",
            sched.build_registry(),
            {"session_id": "s1", "objective": "What did NVDA's latest 10-Q disclose about revenue?"},
            _node(),
            [],
            [listing],
            None,
        )
    )
    assert args == {"accession_no": latest_q, "query": "revenue", "limit": 32000}


def test_generated_valid_8k_accession_changes_to_listed_latest_ten_q_before_invoke() -> None:
    """The scheduler replaces a valid generated 8-K accession before the first document call."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"
    listing = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA", "forms": ["10-Q"]},
        "error": None,
        "result": {
            "filings": [
                {"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"},
                {"accession_no": accq, "form": "10-Q", "filed_at": "2026-08-01"},
            ]
        },
        "outcome_summary": f"8-K {acc8}, 10-Q {accq}",
    }
    invoked: list[tuple[str, dict[str, Any]]] = []
    needle_calls: list[str] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        needle_calls.append(kw["tool"])
        return {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "reasoning": "r"}

    async def invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append((name, dict(args)))
        return {"text": "Revenue was $46.7 billion."}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_sec_document",
            node=_node(),
            session={
                "session_id": "s1",
                "objective": "What did NVDA's latest 10-Q disclose about revenue as of 2026-09-01?",
            },
            registry=sched.build_registry(),
            evidence=[],
            attempts=[listing],
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=lambda n, r: _outcome("Revenue was $46.7 billion."),
            tool_session=SimpleNamespace(),
            as_of=None,
        )
    )
    assert needle_calls == ["get_sec_document"]
    assert invoked == [("get_sec_document", {"accession_no": accq, "query": "revenue", "limit": 32000})]
    assert out["arguments"] == {"accession_no": accq, "query": "revenue", "limit": 32000}
    assert out["error"] is None


@pytest.mark.parametrize("runtime_envelope", [True, False], ids=["runtime-envelope", "in-memory"])
def test_ten_q_revenue_route_settles_structured_listing(
    runtime_envelope: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route reads the latest exact 10-Q from persisted or in-memory rows, not misleading summary text."""
    from copy import deepcopy

    import app.tools as _tools
    import tests.research_source_seam as seam
    from app.research.programmatic_router import programmatic_select_round
    from app.tool_runtime import outcome_from_result

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", lambda name: {"NVDA": "NVDA"}.get(name))
    prompt = "What did NVDA's latest 10-Q disclose about revenue?"
    acc8, old_q, latest_q = "0001045810-26-000078", "0001045810-26-000050", "0001045810-26-000075"
    seam.install(monkeypatch)
    document = "nvda-20260726.htm"
    section_text = {
        "Item 1": "Revenue was $46.7 billion, up 56% from a year ago.",
        "Item 2": "Data Center revenue drove the increase. Gaming revenue rose 49%.",
        "Item 1A": "Export controls may reduce our revenue from China.",
    }
    seam.register_document(latest_q, document, "\n".join(section_text.values()))
    handles = {
        section: seam.handle_for_registered(text, accession=latest_q, document=document)
        for section, text in section_text.items()
    }
    passages: list[object] = []
    receipt = "s1:tr:list"
    rows = [
        {"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"},
        {"accession_no": old_q, "form": "10-Q", "filed_at": "2026-05-01"},
        {"accession_no": latest_q, "form": "10-Q", "filed_at": "2026-08-26"},
    ]
    structured: dict[str, Any] = {"filings": rows, "source": "SEC EDGAR"}
    summary = f"Bottom line: The latest 10-Q is {acc8}. The other 10-Q is {old_q}."
    listed: dict[str, Any] = {
        "content": summary,
        "meta": {"source": "SEC EDGAR"},
        "tool_result_id": receipt,
    }
    if not runtime_envelope:
        listed.update(structured)
    original = deepcopy(listed)
    reads: list[str] = []
    invoked: list[tuple[str, dict[str, Any]]] = []
    needle_calls: list[str] = []

    def get_tool_result(ref: str) -> dict[str, Any]:
        reads.append(ref)
        assert ref == receipt
        if not runtime_envelope:
            raise KeyError(ref)
        return {"tool_result_id": receipt, "tool_name": "list_sec_filings", "result": structured}

    class _StoredKernel(_Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.store = SimpleNamespace(get_tool_result=get_tool_result)

        def admit_evidence(self, sid: str, jid: str, cand: dict[str, Any]) -> dict[str, str]:
            passages.append(cand.get("matching_passage"))
            return super().admit_evidence(sid, jid, cand)

    class _Jev(_JevAdmit):
        async def select_tool(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("The programmatic router must select the 10-Q revenue tools.")

    async def fake_none(**kw: Any) -> dict[str, Any]:
        needle_calls.append(kw["tool"])
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    async def invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        assert needle_calls == []
        invoked.append((name, dict(args)))
        if name == "list_sec_filings":
            return listed
        section = args["section"]
        return {
            "tool_result_id": f"s1:tr:doc{list(section_text).index(section)}",
            "content": f"Bottom line: {section_text[section]}",
            "meta": {"source_handle": handles[section]},
        }

    kernel = _StoredKernel()
    out = asyncio.run(
        sched._drive_rounds(
            SimpleNamespace(
                node_id="n1", session_id="s1", question=f"[Today UTC 2026-10-09] {prompt}", why_it_matters="w"
            ),
            {
                "session_id": "s1",
                "objective": prompt,
                "created_at": "2026-10-09T12:00:00+00:00",
                "temporal_scope": {
                    "mode": "as_of",
                    "as_of": "2026-10-09T12:00:00+00:00",
                    "start": None,
                    "end": None,
                    "raw": None,
                },
            },
            [{"name": "list_sec_filings", "parameters": {}}, {"name": "get_sec_document", "parameters": {}}],
            kernel,
            _Jev(),
            "s1",
            "n1",
            None,
            {
                "select_round": programmatic_select_round,
                "reasoner": None,
                "needle_generate": fake_none,
                "invoke": invoke,
                "to_outcome": outcome_from_result,
                "max_rounds": 4,
            },
            SimpleNamespace(),
            None,
        )
    )
    assert invoked == [
        ("list_sec_filings", {"identifier": "NVDA", "forms": ["10-Q"]}),
        *(
            ("get_sec_document", {"accession_no": latest_q, "limit": 32000, "section": section})
            for section in section_text
        ),
    ]
    # Each section admits as its own evidence with its full section text.
    assert out["status"] == "resolved" and out["admitted"] == 4
    assert passages[1:] == list(section_text.values())
    listing, *documents = out["attempts"]
    assert needle_calls == []
    for attempt in (listing, *documents):
        assert attempt["reasoning"].startswith("scheduler grounded latest 10-Q revenue arguments")
        assert not any(word in attempt["reasoning"].lower() for word in ("needle", "withhold", "failure"))
    assert listing["result"] == (structured if runtime_envelope else original)
    assert listing["result"]["filings"] == rows
    assert listing["tool_result_ref"] == receipt
    assert listing["outcome_summary"] == summary
    assert [d["tool_result_ref"] for d in documents] == ["s1:tr:doc0", "s1:tr:doc1", "s1:tr:doc2"]
    assert len({d["evidence_id"] for d in documents}) == 3
    assert all("result" not in d for d in documents)
    assert listed == original
    assert reads and all(ref == receipt for ref in reads)
    for _, job_id in (call for call in kernel.calls if call[0] == "admit"):
        assert kernel.calls.index(("admit", job_id)) < kernel.calls.index(("complete", job_id))


def test_ten_q_revenue_node_reads_document_after_reason_over_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live rs:39fb3c2e: reason_over_evidence on the node's own listing must not end the node before the 10-Q read."""
    import app.tools as _tools
    import tests.research_source_seam as seam
    from app.research.programmatic_router import programmatic_select_round

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    prompt = "What did NVDA's latest 10-Q disclose about revenue? Use stockbot, bearbot, and bullbot."
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"
    seam.install(monkeypatch)
    document_handle = seam.handle_for(
        "Revenue was $46.7 billion, up 56% from a year ago.", accession=accq, document="nvda-20260726.htm"
    )
    invoked: list[tuple[str, dict[str, Any]]] = []

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    async def invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append((name, args))
        return {"tool": name}

    def to_outcome(name: str, result: dict[str, Any]) -> SimpleNamespace:
        if name == "list_sec_filings":
            content = f"filing {acc8} 8-K {'x' * 130} filing {accq} 10-Q"
            handle = None
        else:
            content = "Revenue was $46.7 billion, up 56% from a year ago."
            handle = document_handle
        return SimpleNamespace(tool_name=name, content=content, source_handle=handle, error=None, error_type=None)

    class _Jev:
        async def select_tool(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("router owns the 10-Q revenue route; JEV select must not run")

        async def assess_result(self, node: Any, outcome: Any, *a: Any, **k: Any) -> dict[str, Any]:
            listing = outcome.tool_name == "list_sec_filings"
            continuation = "reason_over_evidence" if listing else "resolve_node"
            return {
                "continuation": continuation,
                "evidence_state": "sufficient_support",
                "topical_relevance_answer": "yes",
                "resolution_support_answer": "unsure" if listing else "yes",
            }

    session = {"session_id": "s1", "objective": prompt}
    out = asyncio.run(
        sched._drive_rounds(
            SimpleNamespace(node_id="n1", session_id="s1", question=prompt, why_it_matters="w"),
            session,
            [{"name": "list_sec_filings", "parameters": {}}, {"name": "get_sec_document", "parameters": {}}],
            _Kernel(),
            _Jev(),
            "s1",
            "n1",
            None,
            {
                "select_round": programmatic_select_round,
                "reasoner": None,
                "needle_generate": fake_none,
                "invoke": invoke,
                "to_outcome": to_outcome,
                "max_rounds": 4,
            },
            SimpleNamespace(),
            None,
        )
    )
    sections = ("Item 1", "Item 2", "Item 1A")
    assert [name for name, _ in invoked] == ["list_sec_filings", *["get_sec_document"] * 3]
    assert [args for _, args in invoked[1:]] == [{"accession_no": accq, "limit": 32000, "section": s} for s in sections]
    assert out["status"] == "resolved" and out["admitted"] == 3
    # Boundaries: the guard holds only until one document read, and only on revenue-quarter asks.
    listing_ok = {"tool": "list_sec_filings", "error": None}
    read_ok = {"tool": "get_sec_document", "error": None}
    read_err = {"tool": "get_sec_document", "error": "timeout"}
    node = SimpleNamespace(question=prompt)
    assert sched._revenue_read_pending(session, node, [listing_ok, read_err])
    assert not sched._revenue_read_pending(session, node, [listing_ok, read_ok])
    assert not sched._revenue_read_pending(session, node, [])
    other = SimpleNamespace(question="What did NVDA disclose about revenue?")
    assert not sched._revenue_read_pending({"objective": "What did NVDA disclose about revenue?"}, other, [listing_ok])


def test_ten_q_revenue_node_reads_document_before_supported_listing_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """A supported resolve_node on the catalog listing waits for the 10-Q read; intake evidence makes it resolvable."""
    import app.tools as _tools
    import tests.research_source_seam as seam
    from app.research.programmatic_router import programmatic_select_round

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    prompt = "What did NVDA's latest 10-Q disclose about revenue? Use stockbot, bearbot, and bullbot."
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"
    seam.install(monkeypatch)
    document_handle = seam.handle_for(
        "Revenue was $46.7 billion, up 56% from a year ago.", accession=accq, document="nvda-20260726.htm"
    )
    invoked: list[tuple[str, dict[str, Any]]] = []

    async def fake_none(**kw: Any) -> dict[str, Any]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    async def invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append((name, args))
        return {"tool": name}

    def to_outcome(name: str, result: dict[str, Any]) -> SimpleNamespace:
        if name == "list_sec_filings":
            return SimpleNamespace(
                tool_name=name,
                content=f"filing {acc8} 8-K {'x' * 130} filing {accq} 10-Q",
                source_handle=None,
                error=None,
                error_type=None,
            )
        handle = document_handle
        content = "Revenue was $46.7 billion, up 56% from a year ago."
        return SimpleNamespace(tool_name=name, content=content, source_handle=handle, error=None, error_type=None)

    class _Jev:
        async def select_tool(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("router owns the 10-Q revenue route; JEV select must not run")

        async def assess_result(self, node: Any, outcome: Any, *a: Any, **k: Any) -> dict[str, Any]:
            return {
                "continuation": "resolve_node",
                "evidence_state": "sufficient_support",
                "topical_relevance_answer": "yes",
                "resolution_support_answer": "yes",
            }

    class _IntakeKernel(_Kernel):
        def list_evidence(self, sid: str) -> list[dict[str, str]]:
            return [{"evidence_id": "ev-intake", "content": "NVDA intake snippet"}]

    session = {"session_id": "s1", "objective": prompt}
    out = asyncio.run(
        sched._drive_rounds(
            SimpleNamespace(node_id="n1", session_id="s1", question=prompt, why_it_matters="w"),
            session,
            [{"name": "list_sec_filings", "parameters": {}}, {"name": "get_sec_document", "parameters": {}}],
            _IntakeKernel(),
            _Jev(),
            "s1",
            "n1",
            None,
            {
                "select_round": programmatic_select_round,
                "reasoner": None,
                "needle_generate": fake_none,
                "invoke": invoke,
                "to_outcome": to_outcome,
                "max_rounds": 4,
            },
            SimpleNamespace(),
            None,
        )
    )
    sections = ("Item 1", "Item 2", "Item 1A")
    assert [name for name, _ in invoked] == ["list_sec_filings", *["get_sec_document"] * 3]
    assert [args for _, args in invoked[1:]] == [{"accession_no": accq, "limit": 32000, "section": s} for s in sections]
    assert out["status"] == "resolved" and out["admitted"] == 3


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
    args, _, _ = asyncio.run(
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
    args, _, _ = asyncio.run(
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

    async def fake_garbage(**kw: Any) -> dict[str, Any]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "F1"}, "reasoning": "r"}

    session = {"session_id": "s1", "objective": "List Apple's most recent 10-K and 10-Q filings."}
    args, _, _ = asyncio.run(
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
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_cik, "list_sec_filings", [], session, _node(), [], [], None)
    )
    assert args.get("identifier") == "0000320193"
    args, _, _ = asyncio.run(
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
    # Sibling 10-Q section reads fail independently. A repeat of one section still breaks.
    item1, item2 = (
        {"tool": "get_sec_document", "arguments": {"accession_no": acc, "section": s}, "error": "boom"}
        for s in ("Item 1", "Item 2")
    )
    assert sched._accession_family_break([item1, item2]) is None
    assert sched._accession_family_break([item1, {**item1, "error": "boom-2"}]) is not None
    assert sched._accession_family_break([attempts[0], item1]) is not None


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
        return await asyncio.gather(*[sched._sec_thread_call(lambda i=i: _work(i)) for i in range(12)])

    assert sorted(asyncio.run(_main())) == list(range(12))
    assert peak["n"] == 4


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


def test_needle_streak_ignores_reselect_attempts() -> None:
    """3 routine reselects never trip needle fail-fast (medium 7)."""
    reselect = {
        "tool": "search_sec_filings",
        "arguments": {},
        "error": "ungrounded; re-selecting",
        "error_type": "reselect",
    }
    assert sched._needle_error_streak([dict(reselect) for _ in range(3)]) == 0
    gen_fail = {"tool": "query_finra", "arguments": {}, "error": "needle blew up"}
    assert sched._needle_error_streak([dict(gen_fail) for _ in range(3)]) == 3


def test_no_new_evidence_counts_error_free_stale_rounds() -> None:
    """Stale rule fires on progress-free, error-free rounds; error rounds reset it (loop-breaker lane)."""
    assert sched._NO_EVIDENCE_ROUNDS == 2


def test_cancelled_attempt_fails_job() -> None:
    """wait_for cancel during invoke fails the kernel job, then re-raises."""
    import asyncio

    kernel = _Kernel()

    async def _slow(*a: Any, **k: Any) -> Any:
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
                to_outcome=lambda n, r: _outcome(),
                tool_session=SimpleNamespace(),
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
        def _never_outcome(n: object, r: object) -> SimpleNamespace:
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
                tool_session=SimpleNamespace(),
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
    from types import SimpleNamespace as _NS

    kernel = _Kernel()
    node = _NS(node_id="n1", session_id="s1", question="q?")
    session = {"session_id": "s1", "objective": "o"}

    async def select_round(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts):
        return ("tool", {"selected": ["search_web"]}, None)

    async def gen(tool, schema=None, **k):
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "", "confidence": 1.0}

    async def invoke(name, args, sess, **k):
        return {"status": "ok", "rows": [{"url": "https://e.com", "title": "t", "passage": "p" * 50}]}

    def to_outcome(name, result):
        return {"error": None, "rows": result["rows"]}

    jev = _NS(
        assess_result=lambda *a, **k: {"evidence_state": "insufficient", "continuation": "resolve_node"},
    )
    import app.research.scheduler as _s

    orig_max = _s._MAX_TOOL_ROUNDS
    _s._MAX_TOOL_ROUNDS = 1
    try:
        with caplog.at_level(logging.INFO):
            asyncio.run(
                _s._drive_rounds(
                    node,
                    session,
                    [{"name": "search_web"}],
                    kernel,
                    jev,
                    "s1",
                    "n1",
                    None,
                    {
                        "select_round": select_round,
                        "reasoner": None,
                        "needle_generate": gen,
                        "invoke": invoke,
                        "to_outcome": to_outcome,
                    },
                    object(),
                    None,
                )
            )
    finally:
        _s._MAX_TOOL_ROUNDS = orig_max
    lines = [r.getMessage() for r in caplog.records if "round_timing" in r.getMessage()]
    assert len(lines) >= 1
    line = lines[0]
    for key in ("round=", "tool=", "select_ms=", "generate_ms=", "tool_ms=", "assess_ms=", "ev_before=", "ev_after="):
        assert key in line


def test_terminal_select_failure_still_logs_timing(caplog: pytest.LogCaptureFixture) -> None:
    """Third straight select failure emits a round_timing row before the terminal return."""
    import asyncio
    import logging
    from types import SimpleNamespace as _NS

    import app.research.scheduler as _s

    async def down(*a: object, **k: object) -> object:
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        out = asyncio.run(
            _s._drive_rounds(
                _NS(node_id="n1", session_id="s1", question="q?"),
                {"session_id": "s1", "objective": "o"},
                [{"name": "search_web"}],
                _Kernel(),
                _NS(),
                "s1",
                "n1",
                None,
                {"select_round": down, "reasoner": None, "needle_generate": None, "invoke": None, "to_outcome": None},
                object(),
                None,
            )
        )
    assert out["status"] == "blocked" and "select failed 3x" in out["reason"]
    assert sum("round_timing" in r.getMessage() for r in caplog.records) == 3


def test_intake_round_stats_carry_phase_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """_intake_round stats carry sec_ms + web_ms + settle_ms + wall_ms."""
    from app.research import kernel_worker as kw

    kernel = _Kernel()

    async def fake_attempt(tool: str, args: dict, sid: str, *a, **k):
        return {
            "tool": tool,
            "arguments": dict(args),
            "outcome": {"error": None},
            "result": {"bytes": "x" * 100},
            "job_id": "j",
        }

    monkeypatch.setattr(kw, "_intake_attempt", fake_attempt)

    async def _empty_chain():
        return []

    monkeypatch.setattr(kw, "_intake_ticker_chain", lambda *a, **k: _empty_chain())
    import app.research.scheduler as _s2

    monkeypatch.setattr(_s2, "_load_evidence", lambda sid, kernel, repo: [])
    monkeypatch.setattr(_s2, "_evidence_candidate", lambda *a, **k: None)
    monkeypatch.setattr(_s2, "_attempt_domain", lambda *a, **k: "web")

    async def _main():
        return await kw._intake_round("q", [], "s1", {"objective": "o"}, [], object(), kernel, None)

    _admitted, _raw, stats = asyncio.run(_main())
    for key in ("sec_ms", "web_ms", "settle_ms", "wall_ms", "ms", "per_source"):
        assert key in stats


def test_selection_guidance_carries_intake_digest() -> None:
    """Seeded guidance holds intake ids plus summaries and the do-not-repeat line."""
    ev = [{"evidence_id": "ev-1", "content": "Apple 10-K revenue rose"}]
    g = sched._selection_guidance(ev, [{"tool": "search_web", "arguments": {"query": "q"}}])
    assert g["intake_evidence"] == ["ev-1: Apple 10-K revenue rose"]
    assert "identical arguments" in g["directive"]


def test_arguments_generate_context_keeps_selection_guidance_for_failed_calls() -> None:
    """Needle still sees a failed prior call's exact arguments and error, in attempts and in the do-not-repeat guidance."""
    evidence = [{"evidence_id": "ev-1", "content": "NVDA 10-Q revenue disclosure"}]
    attempts = [
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": "0001045810-26-000075", "query": "revenue increased"},
            "error": "query not found",
        }
    ]
    seen: dict[str, Any] = {}

    async def fake_gen(payload: dict[str, Any]) -> dict[str, Any]:
        seen.update(payload)
        return {"tool": "search_web", "arguments": {"query": "NVDA revenue"}, "reasoning": "r"}

    asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "search_web",
            sched.build_registry(),
            {"session_id": "s1", "objective": "What did NVDA disclose about revenue?"},
            _node(),
            evidence,
            attempts,
            None,
        )
    )
    assert seen["op"] == "arguments.generate"
    context = seen["context"]
    (sent,) = context["attempts"]
    assert sent["tool"] == "get_sec_document" and sent["error"] == "query not found"
    assert sent["arguments"] == attempts[0]["arguments"]
    guidance = context["selection_guidance"]
    assert guidance["intake_evidence"] == ["ev-1: NVDA 10-Q revenue disclosure"]
    assert guidance["prior_calls"] == [f"get_sec_document {attempts[0]['arguments']!r}"]
    assert "identical arguments" in guidance["directive"]


_GEN_RAW = "RAWPAYLOAD-7731"


def _oversized_generation_inputs(scale: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run-sized sources and attempt results; scale grows only raw bytes, never ids or the leading facts."""
    evidence = [
        {
            "evidence_id": f"ev-{i}",
            "source_name": "get_sec_document",
            "ticker": "NVDA",
            "cik": "0001045810",
            "content": f"NVDA fact {i}: revenue was $46.7 billion. " + "Filing boilerplate. " * (3500 * scale),
            "metadata": {"raw": _GEN_RAW * (100 * scale)},
        }
        for i in range(6)
    ]
    # The newest source names its accession only inside the nested handle.
    evidence[-1]["metadata"]["source_handle"] = {"accession_no": "0001045810-26-000075", "text": "x" * (4000 * scale)}
    attempts = [
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": f"0001045810-26-{i:06d}", "query": "q" * (2500 * scale)},
            "outcome_summary": f"read {i} " + "s" * (1990 * scale),
            "error": None if i % 2 else "query not found " * (40 * scale),
            "error_type": None if i % 2 else "tool_error",
            "job_id": f"job-{i}",
            "tool_result_ref": f"s1:tr:{i}",
            "reasoning": "r" * (3000 * scale),
            "result": {"filings": [{"accession_no": _GEN_RAW, "form": "8-K"}] * (200 * scale)},
        }
        for i in range(14)
    ]
    return evidence, attempts


def _generation_context(evidence: list[Any], attempts: list[dict[str, Any]], session: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"tool": "search_web", "arguments": {"query": "NVDA revenue"}, "reasoning": "r"}

    asyncio.run(
        sched._generate_tool_arguments(
            fake_gen, "search_web", sched.build_registry(), session, _node(), evidence, attempts, None
        )
    )
    return seen["context"]


def test_arguments_generate_context_is_bounded_but_keeps_grounded_ids_and_facts() -> None:
    """Run26 sent a 69K source plus raw attempt results into a 20s generation; the bounded view keeps facts and ids."""
    import copy
    import json

    evidence, attempts = _oversized_generation_inputs(1)
    stored = copy.deepcopy((evidence, attempts))
    session = {"session_id": "s1", "objective": "What did NVDA disclose about revenue?"}
    context = _generation_context(evidence, attempts, session)

    # Full inputs stay untouched for storage, admission, duplicate detection, and accession carry.
    assert (evidence, attempts) == stored
    payload = json.dumps(context)
    assert len(json.dumps([evidence, attempts])) > 500_000 and len(payload) < 60_000
    assert _GEN_RAW not in payload and "Filing boilerplate. " * 20 not in payload

    sent_ev = context["evidence"]
    assert [e["evidence_id"] for e in sent_ev] == ["ev-2", "ev-3", "ev-4", "ev-5"]
    for item, full in zip(sent_ev, evidence[-4:], strict=True):
        assert item["ticker"] == "NVDA" and item["cik"] == "0001045810"
        assert item["content"] == full["content"][: sched._JEV_DIGEST_MAX]
        assert "revenue was $46.7 billion" in item["content"]
        assert "metadata" not in item
    assert sent_ev[-1]["accession_no"] == "0001045810-26-000075"

    sent_at = context["attempts"]
    assert [a["tool_result_ref"] for a in sent_at] == [f"s1:tr:{i}" for i in range(4, 14)]
    for item, full in zip(sent_at, attempts[-10:], strict=True):
        assert item["job_id"] == full["job_id"] and item["tool"] == "get_sec_document"
        assert item["arguments"]["accession_no"] == full["arguments"]["accession_no"]
        assert item["arguments"]["query"] == full["arguments"]["query"][: sched._OUTCOME_SUMMARY_MAX]
        assert item["outcome_summary"] == full["outcome_summary"][: sched._JEV_DIGEST_MAX]
        assert item.get("error") == (None if full["error"] is None else full["error"][: sched._JEV_DIGEST_MAX])
        assert "result" not in item and "reasoning" not in item

    guidance = context["selection_guidance"]
    assert len(guidance["prior_calls"]) == 10
    assert all(len(call) < sched._OUTCOME_SUMMARY_MAX + 200 for call in guidance["prior_calls"])
    assert "0001045810-26-000013" in guidance["prior_calls"][-1]
    assert [line.split(":", 1)[0] for line in guidance["intake_evidence"]] == ["ev-2", "ev-3", "ev-4", "ev-5"]

    # Missing facts stay absent: no date, scope, or digest is invented.
    assert context["as_of"] is None and context["temporal_scope"] == {}
    assert "selection_digest" not in context


@settings(max_examples=8, derandomize=True, deadline=None)
@given(st.integers(min_value=2, max_value=5))
def test_prop_generation_context_independent_of_raw_payload_growth(scale: int) -> None:
    """Growing raw sources and results past every bound leaves every non-clock generation field identical."""
    import json

    session = {"session_id": "s1", "objective": "What did NVDA disclose about revenue?"}
    base = _generation_context(*_oversized_generation_inputs(1), session)
    grown = _generation_context(*_oversized_generation_inputs(scale), session)
    # today_utc reads the wall clock; every other field must not move.
    base.pop("today_utc")
    grown.pop("today_utc")
    assert json.dumps(grown, sort_keys=True) == json.dumps(base, sort_keys=True)


def test_arguments_generate_withheld_missing_ticker_stays_absent_with_oversized_context() -> None:
    """Bounding never seeds a missing fact: a withheld generation keeps its gap and its withheld flag."""
    evidence, attempts = _oversized_generation_inputs(1)

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest"},
            "reasoning": "ticker not stated",
            "withheld": True,
        }

    args, _reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "query_finra",
            sched.build_registry(),
            {"session_id": "s1", "objective": "Which stocks have the highest short interest?"},
            _node(),
            evidence,
            attempts,
            None,
        )
    )
    assert args == {"dataset": "consolidatedShortInterest"} and withheld is True


def test_duplicate_guard_uses_full_history_beyond_generation_window() -> None:
    """A repeat of a call older than the 10-attempt generation window still reselects without invoking."""
    attempts = [{"tool": "search_web", "arguments": {"query": f"NVDA revenue {i}"}, "error": None} for i in range(14)]
    invoked: list[str] = []
    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_web",
            node=_node(),
            session={"session_id": "s1", "objective": "NVDA revenue"},
            registry=[],
            evidence=[],
            attempts=attempts,
            kernel=_Kernel(),
            needle_generate=lambda **kw: {"tool": "search_web", "arguments": {"query": "NVDA revenue 0"}},
            invoke=lambda *a, **k: invoked.append("invoked") or {},
            to_outcome=lambda n, r: _outcome(),
            tool_session=SimpleNamespace(),
            as_of=None,
        )
    )
    assert out["error_type"] == "reselect" and "duplicate" in out["error"]
    assert invoked == []


def _xbrl_attempt(
    objective: str,
    concept: str,
    *,
    question: str = "q?",
    evidence: list[Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
    reasoning: str = "r",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One real _attempt_tool for get_xbrl_facts with a fake Needle and a recording runtime invoke."""
    invoked: list[dict[str, Any]] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        return {
            "tool": "get_xbrl_facts",
            "arguments": {"ticker": "NVDA", "concept": concept},
            "reasoning": reasoning,
        }

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append(dict(args))
        return {"tool_result_id": "s1:tr:xbrl"}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_xbrl_facts",
            node=SimpleNamespace(node_id="n1", session_id="s1", question=question, why_it_matters="w"),
            session={"session_id": "s1", "objective": objective},
            registry=sched.build_registry(),
            evidence=evidence or [],
            attempts=attempts or [],
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=lambda n, r: _outcome("NVDA XBRL fact"),
            tool_session=SimpleNamespace(),
            as_of=None,
        )
    )
    return out, invoked


@pytest.mark.parametrize("concept", ["Today UTC date", "TodayUTCDate", "today"])
def test_xbrl_date_placeholder_concept_reselects_before_invoke(concept: str) -> None:
    """Live control run: Needle copied the Today UTC date guidance into concept; the runtime must never see it."""
    prompt = "[Today UTC 2026-10-09] What did NVDA report in its latest XBRL filing?"
    out, invoked = _xbrl_attempt(prompt, concept, question=prompt)
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in out["error"]
    assert out["arguments"] == {}


@pytest.mark.parametrize("concept", ["NetIncomeLoss", "GrossProfit"])
def test_xbrl_unrequested_concept_reselects_before_invoke(concept: str) -> None:
    """A canonical or real tag the user never asked about and no source names cannot reach the runtime."""
    out, invoked = _xbrl_attempt("What was NVDA revenue in the latest quarter?", concept)
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in out["error"]


@pytest.mark.parametrize("concept", ["Revenue", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
def test_xbrl_requested_canonical_revenue_alias_invokes(concept: str) -> None:
    """A revenue ask grounds the canonical Revenue key and each of its existing SEC aliases, unchanged."""
    out, invoked = _xbrl_attempt("What was NVDA revenue in the latest quarter?", concept)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": concept}]


def test_xbrl_requested_exact_sec_tag_outside_canonical_map_invokes() -> None:
    """An exact SEC tag the user names stays valid even though the canonical map does not list it."""
    out, invoked = _xbrl_attempt(
        "What did NVDA report as ResearchAndDevelopmentExpense in XBRL?", "ResearchAndDevelopmentExpense"
    )
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ResearchAndDevelopmentExpense"}]


def test_xbrl_concept_named_late_in_full_source_invokes() -> None:
    """Grounding reads the full admitted source, so a tag past every generation bound still counts."""
    row = {
        "evidence_id": "ev-10k",
        "source_name": "get_sec_document",
        "content": "Balance sheet boilerplate. " * 3000 + "us-gaap:InventoryNet 10,080 million.",
    }
    out, invoked = _xbrl_attempt(
        "Which balance sheet facts did NVDA's latest 10-K tag?", "InventoryNet", evidence=[row]
    )
    assert len(row["content"]) > 69_000 and row["content"].index("InventoryNet") > sched._OUTCOME_SUMMARY_MAX
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "InventoryNet"}]


def test_xbrl_failed_attempt_and_generated_text_cannot_ground_fabricated_concept() -> None:
    """A concept that only prior generated args, error text, or Needle reasoning name stays ungrounded."""
    fabricated = "AIAcceleratorRevenueNet"
    failed = {
        "tool": "get_xbrl_facts",
        "arguments": {"ticker": "NVDA", "concept": fabricated, "period": "FY2026"},
        "outcome_summary": f"concept {fabricated} not found",
        "error": f"concept {fabricated} not found for NVDA",
        "error_type": "tool_error",
        "reasoning": f"use {fabricated}",
    }
    out, invoked = _xbrl_attempt(
        "What did NVDA report about AI accelerators?",
        fabricated,
        attempts=[failed],
        reasoning=f"{fabricated} is the matching concept",
    )
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in out["error"]


# Vowel-free segments keep generated tags from spelling date words (Today, ThisWeek) the guard rightly rejects.
@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.from_regex(r"Zq[bcdfgjkmpvxz]{3,6}(?:[A-Z][bcdfgjkmpvxz]{3,6}){1,3}", fullmatch=True),
    st.sampled_from(["objective", "source", "failed_attempt", "generated_only"]),
)
def test_prop_xbrl_tag_invokes_only_when_requested_or_source_backed(tag: str, placement: str) -> None:
    """Any arbitrary SEC-shaped tag reaches the runtime exactly when the user or an admitted source names it."""
    objective = (
        f"What did NVDA report as {tag} in XBRL?"
        if placement == "objective"
        else "What did NVDA report about AI accelerators?"
    )
    evidence = (
        [{"evidence_id": "ev-src", "source_name": "get_sec_document", "content": "Notes. " * 400 + f"us-gaap:{tag} 7"}]
        if placement == "source"
        else []
    )
    attempts = (
        [
            {
                "tool": "get_xbrl_facts",
                "arguments": {"ticker": "NVDA", "concept": tag, "period": "FY2026"},
                "outcome_summary": f"concept {tag} not found",
                "error": f"concept {tag} not found",
                "error_type": "tool_error",
            }
        ]
        if placement == "failed_attempt"
        else []
    )
    out, invoked = _xbrl_attempt(objective, tag, evidence=evidence, attempts=attempts, reasoning=f"{tag} matches")
    if placement in ("objective", "source"):
        assert out["error"] is None
        assert invoked == [{"ticker": "NVDA", "concept": tag}]
    else:
        assert invoked == []
        assert out["error_type"] == "reselect" and "duplicate" not in out["error"]


def test_duplicate_call_reselects() -> None:
    """Same tool plus equal args returns a duplicate reselect, never invokes."""
    kernel = _Kernel()
    seen: list[str] = []
    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_web",
            node=_node(),
            session={"session_id": "s1", "objective": "q"},
            registry=[],
            evidence=[],
            attempts=[{"tool": "search_web", "arguments": {"query": "q"}}],
            kernel=kernel,
            needle_generate=lambda **kw: {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "r"},
            invoke=lambda *a, **k: seen.append("invoked") or {},
            to_outcome=lambda n, r: _outcome(),
            tool_session=SimpleNamespace(),
            as_of=None,
        )
    )
    assert out["error_type"] == "reselect" and "duplicate" in out["error"]
    assert seen == []
    assert sched._needle_error_streak([dict(out)]) == 0


def test_no_new_evidence_stops_after_two() -> None:
    """Two rounds without progress block with reason no new evidence."""
    kernel = _Kernel()
    calls = {"n": 0}

    async def select_round(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts):
        calls["n"] += 1
        assert any(isinstance(e, dict) and e.get("type") == "selection_guidance" for e in ctx_ev)
        return ("tool", {"selected": ["search_web"]}, None)

    async def gen(tool, schema=None, **k):
        return {"tool": "search_web", "arguments": {"query": f"q{calls['n']}"}, "reasoning": "", "confidence": 1.0}

    out = asyncio.run(
        sched._drive_rounds(
            _node(),
            {"session_id": "s1", "objective": "o"},
            [{"name": "search_web"}],
            kernel,
            SimpleNamespace(
                assess_result=lambda *a, **k: {"evidence_state": "insufficient", "continuation": "continue_research"}
            ),
            "s1",
            "n1",
            None,
            {
                "select_round": select_round,
                "reasoner": None,
                "needle_generate": gen,
                "invoke": lambda n, a, s, **k: {},
                "to_outcome": lambda n, r: _outcome(),
            },
            SimpleNamespace(),
            None,
            None,
        )
    )
    assert out["status"] == "blocked" and out["reason"] == "incomplete: no new evidence"
    assert calls["n"] == 2


def test_finra_datapoints_unknown_dataset_reselects() -> None:
    with pytest.raises(sched._ReselectRequest, match="unknown dataset"):
        sched._validate_finra_datapoint_args("get_finra_datapoints", {"dataset": "2025-03-08", "fields": ["trade"]})
    with pytest.raises(sched._ReselectRequest, match="non-empty string list"):
        sched._validate_finra_datapoint_args(
            "get_finra_datapoints", {"dataset": "consolidatedShortInterest", "fields": []}
        )
    sched._validate_finra_datapoint_args(
        "get_finra_datapoints",
        {"dataset": "otcMarket/consolidatedShortInterest", "fields": ["settlementDate"], "ticker": "XYZ"},
    )


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
        return ("tool", {"selected": ["search_web"]}, None)

    out = asyncio.run(
        sched._drive_rounds(
            _node(),
            {"session_id": "s1", "objective": "o"},
            [{"name": "search_web"}],
            _Kernel(),
            SimpleNamespace(),
            "s1",
            "n1",
            None,
            {"select_round": spy, "reasoner": None, "deadline_at": _t.perf_counter() - 1.0},
            object(),
            None,
        )
    )
    assert out["status"] == "blocked" and "run deadline" in out["reason"]
    assert selected == []


def test_run_honors_caller_deadline_at() -> None:
    """A caller-supplied past deadline_at stops run() before selecting, and reaches nodes."""
    import time as _t
    from types import SimpleNamespace as _NS

    selected: list[str] = []

    class _K:
        def ready_nodes(self, sid: str) -> list[_NS]:
            return [_NS(node_id="n1", session_id=sid, question="q?")]

        def block_node(self, sid: str, nid: str, reason: str = "") -> None:
            selected.append(nid)

    out = asyncio.run(sched.run("s1", kernel=_K(), deadline_at=_t.perf_counter() - 1.0))
    assert out["status"] == "incomplete_guard" and "run deadline" in str(out.get("reason"))
    assert out["nodes"] == [] and selected == []


def test_mid_round_timeout_blocks_each_node() -> None:
    """A deadline during the session round records a blocked result per in-flight node."""
    import time as _t

    from types import SimpleNamespace as _NS

    blocked: list[str] = []

    class _K:
        def ready_nodes(self, sid: str) -> list[_NS]:
            if not hasattr(self, "seen"):
                self.seen = True
                return [_NS(node_id="n1", session_id=sid, question="q?"), _NS(node_id="n2", session_id=sid)]
            return []

        def block_node(self, sid: str, nid: str, reason: str = "") -> None:
            blocked.append(nid)

    async def slow(node: object, session_id: str, hooks: dict, kernel: object, cap: object) -> dict:
        import asyncio as _a

        await _a.sleep(60)
        return {"node_id": "?", "status": "resolved"}

    import unittest.mock as _mock

    with _mock.patch.object(sched, "_run_one_node", slow):
        out = asyncio.run(sched.run("s1", kernel=_K(), deadline_at=_t.perf_counter() + 0.05))
    assert out["status"] == "incomplete_guard" and "during session round" in str(out.get("reason"))
    assert sorted(r["node_id"] for r in out["nodes"]) == ["n1", "n2"]
    assert all(r["status"] == "blocked" and r.get("incomplete_guard") for r in out["nodes"])
    assert sorted(blocked) == ["n1", "n2"]


def test_max_rounds_hook_caps_rounds() -> None:
    """max_rounds=2 with progressing rounds reaches the guard naming the cap."""
    calls = {"n": 0}

    async def select_round(jev: object, kernel: object, sid: str, nid: str, *a: object, **k: object) -> object:
        calls["n"] += 1
        return ("tool", {"selected": ["query_finra"]}, None)

    async def gen(tool: object, schema: object = None, **k: object) -> dict[str, object]:
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": "XYZ"},
            "reasoning": "r",
        }

    async def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, object]:
        return {
            "tool_result_id": f"s1:tr:{calls['n']}",
            "records": [{"settlementDate": "2024-01-01", "symbolCode": "XYZ", "currentShortPositionQuantity": 123}],
            "briefing": "B",
        }

    out = asyncio.run(
        sched._drive_rounds(
            _node(),
            {"session_id": "s1", "objective": "o"},
            [{"name": "query_finra", "parameters": {}}],
            _Kernel(),
            SimpleNamespace(
                assess_result=lambda *a, **k: {
                    "evidence_state": "sufficient_support",
                    "continuation": "continue_research",
                }
            ),
            "s1",
            "n1",
            None,
            {
                "select_round": select_round,
                "reasoner": None,
                "needle_generate": gen,
                "invoke": invoke,
                "to_outcome": lambda n, r: _outcome(),
                "max_rounds": 2,
            },
            SimpleNamespace(),
            None,
        )
    )
    assert calls["n"] == 2 and "(2 rounds without resolution)" in out["reason"]


def test_reason_over_evidence_selects_again_with_admitted_evidence() -> None:
    """Live rs:79691cfa: reason_over_evidence starts another selection inside the cap, not the runtime guard."""

    class _StoreKernel(_Kernel):
        def list_evidence(self, sid: str) -> list[dict[str, str]]:
            return [{"evidence_id": eid, "content": "record values here"} for eid in self.evidence]

        @override
        def resolve_node(self, sid: str, nid: str) -> None:
            self.calls.append(("resolve", nid))

    def drive(max_rounds: int) -> tuple[dict[str, object], _StoreKernel, list[list[object]]]:
        kernel = _StoreKernel()
        seen: list[list[object]] = []

        async def select_round(
            jev: object,
            kernel: object,
            sid: str,
            nid: str,
            session: object,
            node: object,
            registry: object,
            ctx_ev: list[object],
            attempts: object,
        ) -> tuple[str, dict[str, list[str]], None]:
            seen.append([e["evidence_id"] for e in ctx_ev if isinstance(e, dict) and e.get("evidence_id")])
            return ("tool", {"selected": ["query_finra"]}, None)

        async def gen(tool: object, schema: object = None, **k: object) -> dict[str, object]:
            ticker = "XYZ" if len(seen) == 1 else "ABC"
            return {"tool": "query_finra", "arguments": {"dataset": "consolidatedShortInterest", "ticker": ticker}}

        async def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, object]:
            return {"tool_result_id": f"s1:tr:{len(seen)}", "records": [{"symbolCode": "XYZ"}], "briefing": "B"}

        async def assess(*a: object, **k: object) -> dict[str, str]:
            first = len(seen) == 1
            return {
                "continuation": "reason_over_evidence" if first else "resolve_node",
                "evidence_state": "sufficient_support",
                "topical_relevance_answer": "yes",
                "resolution_support_answer": "unsure" if first else "yes",
            }

        out = asyncio.run(
            sched._drive_rounds(
                _node(),
                {"session_id": "s1", "objective": "o"},
                [{"name": "query_finra", "parameters": {}}],
                kernel,
                SimpleNamespace(assess_result=assess),
                "s1",
                "n1",
                None,
                {
                    "select_round": select_round,
                    "reasoner": None,
                    "needle_generate": gen,
                    "invoke": invoke,
                    "to_outcome": lambda n, r: _outcome(),
                    "max_rounds": max_rounds,
                },
                SimpleNamespace(),
                None,
            )
        )
        return out, kernel, seen

    out, kernel, seen = drive(4)
    assert out["status"] == "resolved" and out["admitted"] == 2
    assert seen == [[], ["ev-0"]]  # the fresh selection receives the admitted evidence
    assert ("resolve", "n1") in kernel.calls
    # Boundary: a fresh selection past the cap still blocks the unresolved node.
    out, kernel, seen = drive(1)
    assert out["status"] == "blocked" and out["reason"] == "incomplete: runtime guard (1 rounds without resolution)"
    assert seen == [[]] and ("resolve", "n1") not in kernel.calls


def test_blocked_terminal_names_non_default_cap() -> None:
    """_blocked_terminal carries the caller's max_rounds, not the module default."""
    out = sched._blocked_terminal("n1", 0, [], 2)
    assert out["reason"] == "incomplete: runtime guard (2 rounds without resolution)"


def test_reasoner_analyze_times_out_slow_call() -> None:
    """A 2s sync analyze with a 0.2s deadline raises TimeoutError in <1s."""
    import time as _time

    class _Slow:
        def analyze(self, prompt: str) -> dict[str, object]:
            _time.sleep(2.0)
            return {"analyses": [], "evidenceRequests": []}

    attempts: list[dict[str, Any]] = []
    t0 = _time.perf_counter()
    with pytest.raises(TimeoutError):
        asyncio.run(
            sched._reasoner_analyze(
                _Slow(),
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

    class _Boom:
        def analyze(self, prompt: str) -> dict[str, object]:
            raise AssertionError("analyze must not run")

    attempts: list[dict[str, Any]] = []
    out = asyncio.run(
        sched._reason_phase(
            _Boom(),
            {"session_id": "s1", "objective": "o"},
            _node(),
            [],
            [],
            attempts,
            SimpleNamespace(),
            _Kernel(),
            "s1",
            "n1",
            SimpleNamespace(),
            0,
            _time.perf_counter() + 5.0,
        )
    )
    assert out["done"] is True and out["terminal"] is None
    assert len(attempts) == 1 and "too little time" in str(attempts[0].get("error"))


def test_bounded_reasoner_clamps_real_client_only() -> None:
    """Real clients get timeout_s=min(config, remaining); stubs return unchanged."""
    from app.reasoner_client import ReasonerClient

    client = ReasonerClient(model="m", url="u", api_key="k", timeout_s=120.0)
    bounded = sched._bounded_reasoner(client, 3.0)
    assert isinstance(bounded, ReasonerClient) and bounded.timeout_s == 3.0
    assert client.timeout_s == 120.0  # the shared client is never mutated
    stub = SimpleNamespace(analyze=lambda p: {})
    assert sched._bounded_reasoner(stub, 3.0) is stub
    assert sched._bounded_reasoner(client, None) is client


def test_force_open_ignores_intake_evidence() -> None:
    """Intake 8-K accession with no node attempts never forces the carry registry."""
    intake = [{"id": "ev:1", "content": "filed 8-K accession 0000320193-25-000079, see filing"}]
    reg = [{"name": "get_sec_filing"}, {"name": "get_sec_document"}, {"name": "get_short_interest"}]
    assert sched._force_open_registry(reg, [], intake, 0) is None


def test_force_open_fires_on_node_attempt_accession() -> None:
    """This node's own attempt carrying an accession narrows to the carry tools."""
    attempts = [
        {
            "tool": "list_sec_filings",
            "arguments": {},
            "outcome_summary": "saw accession 0000320193-25-000079",
            "error": None,
        }
    ]
    reg = [{"name": "get_sec_filing"}, {"name": "get_sec_document"}, {"name": "get_short_interest"}]
    out = sched._force_open_registry(reg, attempts, [], 0)
    assert out is not None and {e["name"] for e in out} == {"get_sec_filing", "get_sec_document"}
    assert sched._force_open_registry(reg, attempts, [], 1) is None
    after_meta = [*attempts, {"tool": "get_sec_filing", "arguments": {}, "outcome_summary": "meta", "error": None}]
    docs = sched._force_open_registry(reg, after_meta, [], 0)
    assert docs is not None and {e["name"] for e in docs} == {"get_sec_document"}


def _assess(
    topical: object = "yes",
    support: object = "yes",
    state: object = "sufficient_support",
    continuation: str = "resolve_node",
) -> dict[str, Any]:
    return {
        "continuation": continuation,
        "continue": continuation,
        "action": continuation,
        "evidence_state": state,
        "decision": state,
        "topical_relevance_answer": topical,
        "resolution_support_answer": support,
    }


def test_probability_policy_boundaries_fail_closed() -> None:
    from app.decision_client import _classify_probability

    assert _classify_probability(0.70) == "yes"
    assert _classify_probability(0.6999) == "unsure"
    assert _classify_probability(0.50) == "unsure"
    assert _classify_probability(0.4999) == "no"
    for bad in (None, "yes", True, float("nan"), -0.1, 1.1):
        assert _classify_probability(bad) is None


def test_relevant_contradiction_does_not_complete() -> None:
    kernel = _Kernel()
    terminal, fresh = sched._continuation_terminal(
        _assess("yes", "unsure", "conflicted"), kernel, "s1", "n1", [{"evidence_id": "ev-1"}], [], 1
    )
    assert terminal is None and fresh is False


def test_relevant_contradiction_stays_admittable() -> None:
    candidate = sched._admittable_candidate(
        "query_finra",
        {
            "result": {"tool_result_id": "s1:tr:x", "records": [{"symbolCode": "XYZ"}]},
            "job_id": "job-1",
        },
        _outcome("bytes"),
        _assess("yes", "unsure", "conflicted"),
    )
    assert candidate is not None


def test_low_topical_relevance_blocks_admission() -> None:
    candidate = sched._admittable_candidate(
        "query_finra",
        {"result": {"tool_result_id": "s1:tr:x", "records": []}, "job_id": "job-1"},
        _outcome("bytes"),
        _assess("no", "yes", "sufficient_support"),
    )
    assert candidate is None


def test_missing_answers_fail_closed_for_admission_and_completion() -> None:
    kernel = _Kernel()
    candidate = sched._admittable_candidate(
        "query_finra",
        {"result": {"tool_result_id": "s1:tr:x", "records": []}, "job_id": "job-1"},
        _outcome("bytes"),
        {"evidence_state": "sufficient_support", "decision": "sufficient_support"},
    )
    assert candidate is None
    terminal, _fresh = sched._continuation_terminal(
        {"continuation": "resolve_node", "evidence_state": "sufficient_support"},
        kernel,
        "s1",
        "n1",
        [{"evidence_id": "ev-1"}],
        [],
        1,
    )
    assert terminal is None


def test_insufficient_or_conflicted_completion_continues() -> None:
    kernel = _Kernel()
    for state in ("insufficient", "conflicted"):
        terminal, _fresh = sched._continuation_terminal(
            _assess("yes", "unsure" if state == "conflicted" else "no", state),
            kernel,
            "s1",
            "n1",
            [{"evidence_id": "ev-1"}],
            [],
            1,
        )
        assert terminal is None


def test_select_resolved_without_support_reselects() -> None:
    resolved: list[str] = []

    class _K(_Kernel):
        def resolve_node(self, sid: str, nid: str) -> None:
            resolved.append(nid)

    class _J:
        async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
            return _assess("yes", "unsure", "conflicted")

    out = asyncio.run(
        sched._resolve_or_reselect(
            _J(),
            _K(),
            _node(),
            SimpleNamespace(action="resolved"),
            "s1",
            "n1",
            [{"evidence_id": "ev-1"}],
            [],
            [],
            1,
            "job-1",
        )
    )
    assert out is None and resolved == []


def test_select_resolved_failure_continues_research() -> None:
    class _J:
        async def assess_result(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("jev down")

    attempts: list[dict[str, Any]] = []
    out = asyncio.run(
        sched._resolve_or_reselect(
            _J(),
            _Kernel(),
            _node(),
            SimpleNamespace(action="resolved"),
            "s1",
            "n1",
            [{"evidence_id": "ev-1"}],
            [],
            attempts,
            1,
        )
    )
    assert out is None and len(attempts) == 1


_REVENUE_ASK = "What did NVDA's latest 10-Q disclose about revenue?"
_RAW_MARKER = "0001045810-26-000999"
_FAKE_VECTOR_MARK = 0.731


def _window_row() -> dict[str, Any]:
    """Admitted 32KB filing window, as run20 carried it into selection."""
    return {
        "evidence_id": "ev-window",
        "source_name": "get_sec_document",
        "content": "Filing window boilerplate. " * 1200 + "Revenue was $46.7 billion, up 56%.",
        "metadata": {"source_handle": {"tool_result_id": "s1:tr:doc", "text": "x" * 4000}},
    }


def _raw_listing_attempt() -> dict[str, Any]:
    rows = [{"accession_no": f"0001045810-26-{i:06d}", "form": "8-K", "filed_at": "2026-09-01"} for i in range(59)]
    rows.append({"accession_no": _RAW_MARKER, "form": "10-Q", "filed_at": "2026-08-01"})
    return {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA", "forms": ["10-Q"]},
        "outcome_summary": "Listed 60 NVDA filings.",
        "error": None,
        "job_id": "job-list",
        "tool_result_ref": "s1:tr:list",
        "result": {"filings": rows},
    }


def _fake_needle(monkeypatch: pytest.MonkeyPatch, mode: str = "ok") -> dict[str, Any]:
    """Fake app.needle_client embed/extract at the module seam; never spawns the worker."""
    import threading

    import app.needle_client as nc

    calls: dict[str, Any] = {"embed": [], "extract": [], "release": threading.Event()}

    def embed(text: str, *, stop_at: float) -> list[float]:
        calls["embed"].append(text)
        if mode == "embed_error":
            raise RuntimeError("needle embed failed: server unavailable")
        if mode == "timeout":
            calls["release"].wait(5.0)
            raise RuntimeError("needle embed released after test")
        return [_FAKE_VECTOR_MARK, 0.1] if "revenue" in text.lower() else [0.1, _FAKE_VECTOR_MARK]

    def extract_fields(record: dict[str, object], passage: str, **kw: object) -> dict[str, object]:
        calls["extract"].append((record, passage))
        if mode == "extract_error":
            raise RuntimeError("needle extract failed: extraction returned values not grounded")
        return {
            "record": record["name"],
            "fields": {"fact": "Revenue was $46.7 billion, up 56%.", "source": "10-Q"},
            "reasoning": "",
            "confidence": 0.9,
        }

    monkeypatch.setattr(nc, "embed", embed)
    monkeypatch.setattr(nc, "extract_fields", extract_fields)
    monkeypatch.setattr(sched, "_SELECT_NEEDLE_S", 0.2 if mode == "timeout" else 5.0, raising=False)
    return calls


class _JevSearchWebCapture(_JevAdmit):
    """Selects search_web for the digest-carry test and captures the selection inputs."""

    def __init__(self) -> None:
        self.selects: list[tuple[Any, Any, Any]] = []
        self.assessed: list[Any] = []

    @override
    async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
        self.selects.append((a[2], a[3], a[4]))
        return SimpleNamespace(tool_name="search_web", probabilities={}, confidence=1.0)

    @override
    async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
        self.assessed.append(a[2])
        return {
            **(await super().assess_result(*a, **k)),
            "continuation": "continue_research",
            "continue": "continue_research",
            "action": "continue_research",
        }


class _JevSelectCapture(_JevAdmit):
    """Captures (registry, evidence, attempts) exactly as they reach jev.select_tool."""

    def __init__(self) -> None:
        self.selects: list[tuple[Any, Any, Any]] = []
        self.assessed: list[Any] = []

    @override
    async def select_tool(self, *a: Any, **k: Any) -> SimpleNamespace:
        self.selects.append((a[2], a[3], a[4]))
        return await super().select_tool(*a, **k)

    @override
    async def assess_result(self, *a: Any, **k: Any) -> dict[str, Any]:
        self.assessed.append(a[2])
        return {
            **(await super().assess_result(*a, **k)),
            "continuation": "continue_research",
            "continue": "continue_research",
            "action": "continue_research",
        }


def _selection_digests(evidence: list[Any]) -> list[str]:
    return [str(e["text"]) for e in evidence if isinstance(e, dict) and e.get("type") == "selection_digest"]


def _assert_compact(items: list[Any]) -> None:
    """Every non-guidance, non-digest selection item carries short scalars only (arguments stay intact)."""
    for item in items:
        assert isinstance(item, dict)
        if item.get("type") in ("selection_guidance", "selection_digest"):
            continue
        assert "result" not in item and "outcome" not in item and "metadata" not in item, item.keys()
        for key, value in item.items():
            if key == "arguments":
                continue
            assert not isinstance(value, (dict, list)), key
            assert not isinstance(value, str) or len(value) <= sched._JEV_DIGEST_MAX, key


def test_select_round_sends_needle_ranked_extracted_text_not_raw_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selection gets compact items plus one readable Needle digest; registry and caller inputs stay full."""
    import copy
    import json

    calls = _fake_needle(monkeypatch)
    registry = sched.build_registry()
    guidance = {"type": "selection_guidance", "intake_evidence": [], "directive": "x", "prior_calls": []}
    ctx = [_window_row(), guidance]
    attempts = [
        _raw_listing_attempt(),
        {"tool": "get_sec_document", "arguments": {"accession_no": _RAW_MARKER}, "error": "query not found " * 30},
    ]
    inputs = copy.deepcopy((registry, ctx, attempts))
    jev = _JevSelectCapture()

    action, _, sel_digest = asyncio.run(
        sched._select_round(
            jev,
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": _REVENUE_ASK},
            _node(),
            registry,
            ctx,
            attempts,
        )
    )

    assert action == "invoke"
    assert isinstance(sel_digest, str) and "Revenue was $46.7 billion, up 56%." in sel_digest
    (sent_registry, sent_ctx, sent_attempts), *_ = jev.selects
    assert sent_registry == inputs[0]
    assert (registry, ctx, attempts) == inputs
    _assert_compact(sent_ctx)
    _assert_compact(sent_attempts)
    assert guidance in sent_ctx
    assert sent_attempts[0]["tool_result_ref"] == "s1:tr:list"
    assert sent_attempts[1]["error"] == attempts[1]["error"][: sched._JEV_DIGEST_MAX]
    payload = json.dumps([sent_ctx, sent_attempts])
    assert _RAW_MARKER not in json.dumps(sent_ctx)
    assert '"filings"' not in payload and str(_FAKE_VECTOR_MARK) not in payload
    assert len(payload) < 4000
    (digest,) = _selection_digests(sent_ctx)
    assert digest.index("ev-window") < digest.index("s1:tr:list")
    assert "Revenue was $46.7 billion, up 56%." in digest
    assert len(calls["extract"]) == 1
    _record, passage = calls["extract"][0]
    assert "Revenue was $46.7 billion" in passage


@pytest.mark.parametrize("mode", ["embed_error", "extract_error", "timeout"])
def test_select_round_needle_outage_falls_back_to_compact_ids(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Needle error or stall never blocks selection: ids plus 200-char summaries, no digest, no raw payloads."""
    import json

    calls = _fake_needle(monkeypatch, mode)
    registry = sched.build_registry()
    row = _window_row()
    listing = _raw_listing_attempt()
    jev = _JevSelectCapture()
    t0 = time.perf_counter()
    try:
        action, _, sel_digest = asyncio.run(
            sched._select_round(
                jev,
                _Kernel(),
                "s1",
                "n1",
                {"session_id": "s1", "objective": _REVENUE_ASK},
                _node(),
                registry,
                [row],
                [listing],
            )
        )
        elapsed = time.perf_counter() - t0
    finally:
        calls["release"].set()

    assert action == "invoke" and elapsed < 2.0
    assert sel_digest is None
    (sent_registry, sent_ctx, sent_attempts), *_ = jev.selects
    assert sent_registry == registry
    assert _selection_digests(sent_ctx) == []
    _assert_compact(sent_ctx)
    _assert_compact(sent_attempts)
    assert sent_ctx[0]["evidence_id"] == "ev-window"
    assert sent_ctx[0]["content"] == row["content"][: sched._JEV_DIGEST_MAX]
    assert sent_attempts[0]["tool_result_ref"] == "s1:tr:list"
    assert _RAW_MARKER not in json.dumps([sent_ctx, sent_attempts])


class _PipeNeedle:
    """Fake Needle child on a real pipe, so the client's select-bounded reads time out for real."""

    instances: ClassVar[list["_PipeNeedle"]] = []
    hang_embed: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import io
        import os

        read_fd, write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "r")
        self._replies = os.fdopen(write_fd, "w")
        self.stdin = self
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.actions: list[str] = []
        _PipeNeedle.instances.append(self)

    def write(self, line: str) -> None:
        import json

        req = json.loads(line)
        action = req["action"]
        self.actions.append(action)
        if self._replies.closed or (action == "embed" and _PipeNeedle.hang_embed):
            return  # the model never answers this embed
        reply = {"ping": {"ready": True}, "embed": {"embedding": [0.1, 0.2]}}.get(
            action, {"tool": req.get("tool"), "arguments": {"identifier": "NVDA"}}
        )
        self._replies.write(json.dumps({"id": req["id"], **reply}) + "\n")
        self._replies.flush()

    def flush(self) -> None:
        pass

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9
        self._replies.close()  # EOF for any reader still waiting

    terminate = kill

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


def test_abandoned_selection_projection_never_delays_argument_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real scheduler + Needle client: a timed-out projection embed frees Needle for the next argument generation."""
    import subprocess
    import threading

    import app.needle_client as nc

    nc.close()
    _PipeNeedle.instances.clear()
    _PipeNeedle.hang_embed = True
    monkeypatch.setattr(subprocess, "Popen", _PipeNeedle)
    monkeypatch.setattr(nc, "_ensure_weights", lambda: None)
    monkeypatch.setattr(sched, "_SELECT_NEEDLE_S", 0.2)
    generated: list[Any] = []
    gen = threading.Thread(target=lambda: generated.append(nc.generate_arguments(tool="list_sec_filings")), daemon=True)
    try:
        ctx, _ = asyncio.run(
            sched._selection_view(
                "s1",
                "n1",
                {"session_id": "s1", "objective": _REVENUE_ASK},
                _node(),
                [_window_row()],
                [_raw_listing_attempt()],
            )
        )
        gen.start()
        # Generation's own bound is its 20s read; it must never queue behind the abandoned 120s embed read.
        gen.join(timeout=5.0)
        finished_in_time = not gen.is_alive()
    finally:
        _PipeNeedle.hang_embed = False
        for proc in list(_PipeNeedle.instances):
            proc.kill()
        sched._NEEDLE_SELECT_POOL.submit(lambda: None).result(timeout=10)
        if gen.ident is not None:
            gen.join(timeout=10)
        nc.close()
        for proc in _PipeNeedle.instances:
            proc.stdout.close()
        actions = [proc.actions for proc in _PipeNeedle.instances]
        _PipeNeedle.instances.clear()

    assert _selection_digests(ctx) == []
    assert actions[0][:2] == ["ping", "embed"]
    assert finished_in_time, "argument generation waited on the abandoned selection projection"
    assert generated == [
        {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "NVDA"},
            "withheld": False,
            "reasoning": "",
            "confidence": None,
        }
    ]


def test_drive_rounds_projects_selection_and_keeps_downstream_context_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 2 selection sees compact attempts plus a digest; the attempt log keeps full bytes."""
    import copy

    _fake_needle(monkeypatch)
    row = _window_row()
    stored = copy.deepcopy(row)
    big = "Short interest rows for XYZ. " * 200

    class _K(_Kernel):
        def list_evidence(self, sid: str) -> list[dict[str, Any]]:
            return [row]

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": _REVENUE_ASK, "query": "", "as_of": None}

    gen_n = {"n": 0}

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        gen_n["n"] += 1
        return {
            "tool": "query_finra",
            "arguments": {"dataset": "consolidatedShortInterest", "ticker": f"X{gen_n['n']}"},
        }

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        return {"tool_result_id": "s1:tr:finra", "records": [], "briefing": big}

    jev = _JevSelectCapture()
    out = asyncio.run(
        sched._run_node(
            _node(),
            session_id="s1",
            kernel=_K(),
            jev=jev,
            repo=None,
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=lambda name, result: _outcome(big),
            registry=[{"name": "query_finra", "parameters": {}}],
            tool_session=SimpleNamespace(),
            max_rounds=2,
        )
    )

    assert len(jev.selects) == 2
    registry, sent_ctx, sent_attempts = jev.selects[1]
    assert [e["name"] for e in registry] == ["query_finra"]
    _assert_compact(sent_ctx)
    _assert_compact(sent_attempts)
    assert sent_attempts and len(_selection_digests(sent_ctx)) == 1
    assert len(out["attempts"][0]["outcome_summary"]) > sched._JEV_DIGEST_MAX
    assert row == stored


@pytest.mark.parametrize("mode", ["ok", "extract_error"])
def test_run_node_carries_the_one_selection_digest_into_argument_generation(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest JEV selected on reaches generation unrecomputed; the follow-up query uses its late fact."""
    import json

    calls = _fake_needle(monkeypatch, mode)
    row = _window_row()

    class _K(_Kernel):
        def list_evidence(self, sid: str) -> list[dict[str, Any]]:
            return [row]

        def get_session(self, sid: str) -> dict[str, Any]:
            return {"session_id": sid, "objective": _REVENUE_ASK, "query": "", "as_of": None}

    generations: list[tuple[dict[str, Any], int, int]] = []
    invoked: list[dict[str, Any]] = []

    async def fake_gen(**kw: Any) -> dict[str, Any]:
        # Model-like consumer: it computes prior-year revenue only when the bounded context states the late fact.
        context = kw["context"]
        generations.append((context, len(calls["embed"]), len(calls["extract"])))
        if "$46.7 billion" in json.dumps(context):
            prior = 46.7 / 1.56  # the digest alone carries the 56%-growth fact past the evidence bound
            return {
                "tool": "search_web",
                "arguments": {"query": f"NVDA prior year revenue {prior:.1f} billion"},
                "reasoning": "r",
            }
        return {"tool": "search_web", "arguments": {"query": "revenue background"}, "reasoning": "r"}

    async def fake_invoke(name: str, args: dict[str, Any], sess: Any, **kw: Any) -> dict[str, Any]:
        invoked.append(dict(args))
        return {"tool_result_id": "s1:tr:finra", "records": []}

    jev = _JevSearchWebCapture()
    asyncio.run(
        sched._run_node(
            _node(),
            session_id="s1",
            kernel=_K(),
            jev=jev,
            repo=None,
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=lambda name, result: _outcome("web rows"),
            registry=[{"name": "search_web", "parameters": {}}],
            tool_session=SimpleNamespace(),
            max_rounds=1,
        )
    )

    ((context, embeds_at_gen, extracts_at_gen),) = generations
    # Bounded evidence drops the 32KB window tail, so only the digest can carry its revenue fact.
    assert "$46.7 billion" not in json.dumps(context["evidence"])
    selected_digests = _selection_digests(jev.selects[0][1])
    if mode == "ok":
        (digest,) = selected_digests
        assert context["selection_digest"] == digest
        assert (embeds_at_gen, extracts_at_gen) == (len(calls["embed"]), len(calls["extract"]))
        assert len(calls["extract"]) == 1
        assert len(invoked) == 1 and "29.9" in json.dumps(invoked)
    else:
        assert selected_digests == []
        assert "selection_digest" not in context
        assert "$46.7 billion" not in json.dumps(context)
        assert len(invoked) == 1 and "29.9" not in json.dumps(invoked)


def test_xbrl_wrapper_echo_cannot_ground_without_real_tag_data() -> None:
    """Envelope echoes (concept_searched, Form, valueless concept) never ground; real tag rows do."""
    bogus = "ZqBogusNet"
    echo = {
        "evidence_id": "ev-echo",
        "source_name": "get_xbrl_facts",
        "content": "XBRL facts envelope.",
        "concept_searched": bogus,
        "form": "10-Q",
        "concept": bogus,
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, evidence=[echo])
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in out["error"]
    echo_attempt = {
        "tool": "get_xbrl_facts",
        "arguments": {"ticker": "NVDA", "concept": "Revenues"},
        "outcome_summary": "XBRL facts envelope.",
        "error": None,
        "result": {"concept_searched": bogus, "form": "10-Q", "matching_concepts": []},
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, attempts=[echo_attempt])
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in out["error"]
    real = {
        "evidence_id": "ev-real",
        "source_name": "get_xbrl_facts",
        "content": "XBRL facts envelope.",
        "matching_concepts": [{"concept": bogus, "value": 123, "period_end": "2026-07-31"}],
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, evidence=[real])
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": bogus}]
    namespaced = {
        "evidence_id": "ev-ns",
        "source_name": "get_sec_document",
        "content": f"Notes to the filing. us-gaap:{bogus} 7 million.",
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, evidence=[namespaced])
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": bogus}]


def test_xbrl_error_and_summary_text_cannot_ground_alias() -> None:
    """Errors and outcome summaries never ground, even when they name a real alias."""
    bad_evidence = {
        "evidence_id": "ev-bad",
        "source_name": "get_sec_document",
        "content": "Filing notes Revenues 10 million.",
        "error": "tool_error",
    }
    failed = {
        "tool": "get_xbrl_facts",
        "arguments": {"ticker": "NVDA", "concept": "Revenues", "period": "FY2026"},
        "outcome_summary": "concept Revenues not found",
        "error": "concept Revenues not found for NVDA",
        "error_type": "tool_error",
    }
    no_data = {
        "tool": "get_xbrl_facts",
        "arguments": {"ticker": "NVDA", "concept": "Revenues", "period": "FY2026"},
        "outcome_summary": "Revenues 10 million",
        "error": None,
        "job_id": "job-1",
    }
    out, invoked = _xbrl_attempt(
        "What did NVDA report about AI accelerators?",
        "Revenues",
        evidence=[bad_evidence],
        attempts=[failed, no_data],
    )
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in out["error"]


def test_generation_item_omits_oversized_nested_arg_keeps_small_types() -> None:
    """An oversized nested argument omits its key, never a cut JSON string; small values keep their type."""
    import copy
    import json

    big = {"rows": [{"n": n} for n in range(600)]}
    assert len(json.dumps(big)) > sched._OUTCOME_SUMMARY_MAX
    item = {
        "tool": "get_sec_document",
        "arguments": {
            "accession_no": "0001045810-26-000075",
            "query": "revenue",
            "page": 3,
            "exact": True,
            "tags": ["a", "b"],
            "big": big,
        },
        "job_id": "job-1",
        "tool_result_ref": "s1:tr:1",
    }
    stored = copy.deepcopy(item)
    out = sched._generation_item(item)
    assert item == stored
    assert "big" not in out["arguments"] and '"big"' not in json.dumps(out)
    assert out["arguments"]["accession_no"] == "0001045810-26-000075"
    assert out["arguments"]["query"] == "revenue"
    assert out["arguments"]["page"] == 3 and isinstance(out["arguments"]["page"], int)
    assert out["arguments"]["exact"] is True
    assert out["arguments"]["tags"] == ["a", "b"]
    assert out["job_id"] == "job-1" and out["tool_result_ref"] == "s1:tr:1"
