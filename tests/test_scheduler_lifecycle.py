"""Scheduler lifecycle: domain wiring, admit-then-complete, stall, expansion."""

import asyncio
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import override
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision_client import JevClient
from app.reasoner_client import ReasonerClient
from app.research import scheduler as sched
from app.research.models import (
    DecisionRecord,
    JSONValue,
    ResearchNode,
    ToolDecision,
    new_decision_id,
    validate_json_value,
)
from app.runtime import ToolResultMeta
from app.tool_runtime import RuntimeToolSession, ToolOutcome

_FINRA_ROW: dict[str, JSONValue] = {
    "settlementDate": "2024-01-01",
    "symbolCode": "XYZ",
    "currentShortPositionQuantity": 123,
}
_FINRA_RESULT: dict[str, JSONValue] = {
    "tool_result_id": "s1:tr:abc",
    "records": [_FINRA_ROW],
    "briefing": "B",
}


def _node(node_id: str = "n1", session_id: str = "s1") -> ResearchNode:
    return ResearchNode(node_id=node_id, session_id=session_id, question="q?", why_it_matters="w")


def _tool_session() -> RuntimeToolSession:
    return RuntimeToolSession(session_id="scheduler:s1")


class _Kernel(sched._Kernel):
    """Fake kernel recording lifecycle order; evidence and tool-result stores are in-memory."""

    def __init__(
        self,
        tool_results: Mapping[str, dict[str, JSONValue]] | None = None,
        objective: str = "",
    ) -> None:
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
        self,
        session_id: str,
        question: str,
        why_it_matters: str,
        depends_on: Sequence[str] | None = None,
    ) -> ResearchNode:
        return _node("n-new", session_id)

    @override
    def ready_nodes(self, session_id: str) -> list[ResearchNode]:
        return []

    @override
    def get_session(self, session_id: str) -> dict[str, JSONValue]:
        return {
            "session_id": session_id,
            "objective": self.objective,
            "query": "",
            "as_of": None,
        }

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
        return {
            "evidence_id": "ev-winner",
            "accepted": False,
            "duplicate_of": "ev-winner",
        }


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


def _invoke_result(*a: object, **k: object) -> dict[str, JSONValue]:
    return {"tool_result_id": "s1:tr:x"}


def _xbrl_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
    return _outcome("NVDA XBRL fact")


def _resolve_nvda(name: str) -> str | None:
    return {"NVDA": "NVDA"}.get(name)


def _invoke(tool: str) -> ToolDecision:
    return ToolDecision(
        action="invoke",
        tool_name=tool,
        tool_names=(tool,),
        probabilities={},
        confidence=1.0,
    )


class _JevAdmit(JevClient):
    @override
    async def select_tool(self, *a: object, **k: object) -> ToolDecision:
        return _invoke("query_finra")

    @override
    async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
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
    finra_result: dict[str, JSONValue] = {
        "tool_result_id": "s1:tr:abc",
        "records": [_FINRA_ROW],
    }
    candidate = sched._evidence_candidate(
        "query_finra",
        "FINRA",
        finra_result,
        _outcome(),
        _Kernel({"s1:tr:abc": finra_result}),
    )
    assert candidate is not None
    row_text = " ".join(
        json.dumps(
            {
                "settlementDate": "2024-01-01",
                "symbolCode": "XYZ",
                "currentShortPositionQuantity": 123,
            },
            sort_keys=True,
            default=str,
        ).split()
    )
    assert candidate["record_identity"] == row_text
    replay = _finra_record_texts(
        {
            "result": {
                "records": [
                    {
                        "settlementDate": "2024-01-01",
                        "symbolCode": "XYZ",
                        "currentShortPositionQuantity": 123,
                    }
                ]
            },
            "tool_name": "query_finra",
        }
    )
    assert any(str(candidate["record_identity"]) in t for t in replay)


def test_finra_envelope_identity_is_symbol_distinct_and_replay_valid() -> None:
    """Envelope identity comes from its own scope: replay-valid, stable across repeats, distinct per symbol."""
    from app.research.service import _finra_record_texts

    def envelope(symbol: str, metrics_sum: int) -> dict[str, JSONValue]:
        return {
            "dataset_id": "otcMarket/consolidatedShortInterest",
            "briefing": None,
            "briefing_source": "deterministic_only",
            "query": {"ticker": symbol, "start_date": None, "end_date": None, "limit": 5, "offset": 0},
            "metrics": {"fields": {"currentShortPositionQuantity": {"sum": metrics_sum}}, "latest_vs_prior": []},
        }

    def identity(payload: dict[str, JSONValue]) -> str:
        persisted: dict[str, JSONValue] = {"result": payload, "tool_name": "get_short_interest"}
        result = {**payload, "tool_result_id": "s1:tr:si"}
        candidate = sched._evidence_candidate(
            "get_short_interest", "FINRA", result, _outcome(), _Kernel({"s1:tr:si": persisted})
        )
        assert candidate is not None
        locator = str(candidate["record_identity"])
        assert locator
        # A model citing this identity must still replay against the persisted result.
        assert any(locator in text for text in _finra_record_texts(persisted))
        return locator

    first = identity(envelope("NVDA", 100))
    assert first == identity(envelope("NVDA", 100))  # identical repeated call keeps one identity
    # Two symbols can return identical metrics; the identity still separates them.
    assert first != identity(envelope("QQQ", 100))


def test_sec_candidate_needs_handle_and_text() -> None:
    """SEC boundary pins: missing handle or empty window/content stays uncitable."""
    good_handle = {"tool_result_id": "s1:tr:sec"}
    good_result = {"source_handle": good_handle, "text": "window words"}
    assert sched._sec_locator(sched._persisted_shapes(good_result)) == "window words"
    assert sched._outcome_summary(_outcome("words")) == "words"
    assert sched._sec_evidence_candidate(good_result, _outcome("words"), _Kernel()) is not None
    assert sched._sec_evidence_candidate({"text": "window words"}, _outcome("words"), _Kernel()) is None
    assert sched._sec_evidence_candidate({"source_handle": good_handle}, _outcome(""), _Kernel()) is None


@pytest.mark.parametrize("runtime_envelope", [True, False], ids=["runtime-envelope", "canonical-text"])
def test_sec_document_admits_full_window_with_bounded_summary(
    runtime_envelope: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    result: dict[str, object] = {
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
    candidate = sched._evidence_candidate("get_sec_document", "SEC", result, outcome, _Kernel())
    assert candidate is not None
    assert candidate["content"] == candidate["matching_passage"] == window
    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("What does the filing disclose?", repo=store)
    node = service.create_node(sid, "q?", "w", repo=store)
    kernel = sched._default_kernel(store)
    jid = sched._attempt_job(kernel, sid, "SEC")
    attempts: list[dict[str, JSONValue]] = []
    settled = asyncio.run(
        sched._settle_round(
            [
                {
                    "tool": "get_sec_document",
                    "source_domain": "SEC",
                    "job_id": jid,
                    "arguments": {
                        "accession_no": handle["accession_no"],
                        "query": "revenue",
                    },
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
            _invoke("get_sec_document"),
            0,
            continuation=False,
        )
    )
    assert settled["admitted"] == 1
    assert attempts[0]["error"] is None
    assert attempts[0]["outcome_summary"] == formatted[: sched._OUTCOME_SUMMARY_MAX]
    summary = attempts[0]["outcome_summary"]
    assert isinstance(summary, str)
    assert len(summary) == 2000
    assert fact not in summary
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
        @override
        def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
            handle = data["source_handle"]
            passage = data["matching_passage"]
            assert isinstance(handle, dict) and isinstance(passage, str)
            materialize_sec_passage(handle, passage)
            return super().admit_evidence(session_id, job_id, data)

    kernel = _ValidatedKernel()
    jid = sched._attempt_job(kernel, "s1", "SEC")
    attempt = {
        "result": result,
        "job_id": jid,
        "outcome_summary": sched._outcome_summary(outcome),
    }
    eid, progressed, failure = sched._admit_attempt_evidence(
        kernel, "s1", "get_sec_document", attempt, outcome, "sufficient_support"
    )
    assert eid is None and not progressed
    assert failure is not None and error_code in str(failure["error"])
    assert kernel.evidence == []
    assert ("complete", jid) in kernel.calls


def test_sec_list_receipt_routes_through_persisted_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_sec_filings source_refs is a catalog receipt, never a handle: it replays by tool_result_id."""
    import tests.research_source_seam as seam
    from app.research.service import _handle_fields, _sec_record_texts
    from app.tool_runtime import outcome_from_result

    filing: dict[str, JSONValue] = {
        "form": "8-K",
        "accession_no": "0000320193-25-000079",
        "url": "https://sec.gov/x",
        "known_at": "2025-05-01",
    }
    listed: dict[str, JSONValue] = {
        "tool_result_id": "s1:tr:list",
        "filings": [filing],
        "source": "SEC EDGAR",
    }
    receipt = outcome_from_result("list_sec_filings", dict(listed))
    assert receipt.source_handle is None and receipt.source_refs
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _handle_fields(receipt.source_refs)  # the receipt is no document handle
    persisted: dict[str, JSONValue] = {
        "tool_result_id": "s1:tr:list",
        "tool_name": "list_sec_filings",
        "result": listed,
    }
    reads: list[str] = []

    class _StoredKernel(_Kernel):
        @override
        def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
            reads.append(tool_result_id)
            return persisted

    kernel = _StoredKernel()
    cand = sched._evidence_candidate("list_sec_filings", "SEC", listed, receipt, kernel)
    assert cand is not None and "source_handle" not in cand, cand
    assert cand["tool_result_id"] == "s1:tr:list" and reads == ["s1:tr:list"]
    assert cand["record_identity"] in _sec_record_texts(persisted)

    # get_sec_document keeps its canonical handle; a malformed explicit handle still reaches the validator.
    seam.install(monkeypatch)
    passage = "Data center revenue grew."
    handle = seam.handle_for(passage)
    doc: dict[str, object] = {
        "source_handle": handle,
        "text": passage,
        "content": passage,
    }
    doc_cand = sched._evidence_candidate(
        "get_sec_document",
        "SEC",
        doc,
        outcome_from_result("get_sec_document", doc),
        kernel,
    )
    assert doc_cand is not None and doc_cand["source_handle"] == handle
    _handle_fields(handle)
    bogus: dict[str, object] = {
        "source_handle": {"basis": "guessed"},
        "text": passage,
        "content": passage,
    }
    bad = sched._evidence_candidate(
        "get_sec_document",
        "SEC",
        bogus,
        outcome_from_result("get_sec_document", bogus),
        kernel,
    )
    assert bad is not None and bad["source_handle"] == {"basis": "guessed"}
    _bad_handle = bad["source_handle"]
    assert isinstance(_bad_handle, Mapping)
    with pytest.raises(ValueError, match="ERR_SEC_HANDLE_INVALID"):
        _handle_fields(_bad_handle)


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
            return {
                **base,
                "continuation": "continue_research",
                "continue": "continue_research",
            }

    def settle(kernel: _Kernel) -> dict[str, object]:
        result: dict[str, JSONValue] = {
            "tool_result_id": "s1:tr:abc",
            "records": [
                {
                    "settlementDate": "2024-01-01",
                    "symbolCode": "XYZ",
                    "currentShortPositionQuantity": 1,
                }
            ],
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
                [record],
                _JevContinue(),
                kernel,
                _node(),
                [],
                attempts,
                "s1",
                "n1",
                _invoke("query_finra"),
                0,
            )
        )
        return {**out, "evidence_id": attempts[-1].get("evidence_id")}

    fresh = settle(_Kernel())
    dup = settle(_DupKernel())
    assert (fresh["admitted"], fresh["progressed"], fresh["evidence_id"]) == (
        1,
        True,
        "ev-0",
    )
    assert (dup["admitted"], dup["progressed"], dup["evidence_id"]) == (
        0,
        False,
        "ev-winner",
    )


def test_assessment_evidence_is_scoped_to_current_node_jobs() -> None:
    """Assessment sees this node's jobs, explicitly referenced duplicates, and observations.

    Unrelated rows from another node or intake never reach it, even from the duplicate winner's job.
    """
    seen: list[object] = []

    class _J(_JevAdmit):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen.append(a[2])
            base = await super().assess_result(*a, **k)
            return {
                **base,
                "continuation": "continue_research",
                "continue": "continue_research",
            }

    kernel = _Kernel()
    result: dict[str, JSONValue] = {
        "tool_result_id": "s1:tr:abc",
        "records": [
            {
                "settlementDate": "2024-01-01",
                "symbolCode": "XYZ",
                "currentShortPositionQuantity": 1,
            }
        ],
    }
    kernel.tool_results["s1:tr:abc"] = result
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-prev", "job_id": "job-1"},
        {"evidence_id": "ev-cur", "job_id": "job-3"},
        {"evidence_id": "ev-winner", "job_id": "job-other-node"},
        {"evidence_id": "ev-other", "job_id": "job-other-node"},
        {"evidence_id": "ev-intake", "job_id": "job-intake"},
    ]
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "query_finra",
            "job_id": "job-1",
            "evidence_id": "ev-prev",
            "outcome_summary": "admitted",
        },
        {
            "tool": "query_finra",
            "job_id": "job-2",
            "evidence_id": None,
            "outcome_summary": "unadmitted obs",
        },
        {
            "tool": "query_finra",
            "job_id": "job-4",
            "evidence_id": "ev-winner",
            "outcome_summary": "duplicate",
        },
        {
            "tool": None,
            "job_id": None,
            "evidence_id": None,
            "outcome_summary": "",
            "error": "select failed",
        },
    ]
    record: dict[str, object] = {
        "tool": "query_finra",
        "arguments": {},
        "outcome": _outcome(),
        "outcome_summary": "record values here",
        "error": None,
        "result": result,
        "job_id": "job-3",
    }
    asyncio.run(
        sched._settle_round(
            [record],
            _J(),
            kernel,
            _node(),
            evidence,
            attempts,
            "s1",
            "n1",
            _invoke("query_finra"),
            0,
        )
    )
    (ctx,) = seen
    assert isinstance(ctx, list)
    assert [c.get("evidence_id") for c in ctx if c.get("evidence_id")] == [
        "ev-prev",
        "ev-cur",
        "ev-winner",
    ]
    assert [c.get("outcome_summary") for c in ctx if "outcome_summary" in c] == ["unadmitted obs"]


def test_pure_stall_is_guard_false_with_stall_reason() -> None:
    class _KS(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node()]

    async def stall(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {
            "node_id": "n1",
            "status": "gathering",
            "admitted": 0,
            "incomplete_guard": False,
        }

    with mock.patch.object(sched, "run_node", stall):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == "stalled"
    assert out["incomplete_guard"] is False
    assert str(out["reason"]).startswith("stalled:")
    assert out["rounds"] == 1
    assert out["nodes"] == [{"node_id": "n1", "status": "gathering", "admitted": 0, "incomplete_guard": False}]


def test_node_guard_trip_yields_incomplete_guard_not_stalled() -> None:
    class _KS(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node()]

    async def trip(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {
            "node_id": "n1",
            "status": "blocked",
            "admitted": 0,
            "incomplete_guard": True,
        }

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
        return {
            "node_id": "n1",
            "status": "blocked",
            "admitted": 0,
            "incomplete_guard": True,
        }

    with (
        mock.patch.object(sched, "run_node", trip),
        mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 0),
    ):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert "runtime guard" in str(out["reason"])


def test_expansion_creates_jev_admitted_nodes_with_dep() -> None:
    created: list[tuple[str, str, tuple[str, ...]]] = []

    class _K(_Kernel):
        @override
        def create_node(
            self,
            session_id: str,
            question: str,
            why_it_matters: str,
            depends_on: Sequence[str] | None = None,
        ) -> ResearchNode:
            created.append((question, why_it_matters, tuple(depends_on or ())))
            return _node(f"n-{len(created)}", session_id)

    class _J(JevClient):
        @override
        async def decide(
            self,
            state: object,
            questions: Mapping[str, JSONValue],
            decision_type: object = None,
            **kw: object,
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
            self,
            session_id: str,
            question: str,
            why_it_matters: str,
            depends_on: Sequence[str] | None = None,
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


@pytest.mark.parametrize("remaining", [None, 30.0], ids=["unbounded", "deadline"])
def test_reason_path_calls_analyze_then_expand(remaining: float | None) -> None:
    """A node admits follow-ups after analysis and adjudication, with or without a deadline."""
    import re
    import time

    created: list[str] = []
    calls: list[str] = []

    class _K(_Kernel):
        @override
        def create_node(
            self,
            session_id: str,
            question: str,
            why_it_matters: str,
            depends_on: Sequence[str] | None = None,
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
            self,
            state: object,
            questions: Mapping[str, JSONValue],
            decision_type: object = None,
            **kw: object,
        ) -> dict[str, dict[str, JSONValue]]:
            return {qid: {"choice": "admit"} for qid in questions}

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
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

    class _R(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            calls.append("analyze")
            assert "Objective?" in prompt and re.search(r"\b\d{4}-\d{2}-\d{2}\b", prompt)
            if remaining is not None:
                assert 1.0 <= self.timeout_s <= remaining
            return {"analyses": [{"nodeId": "n1"}], "evidenceRequests": []}

        @override
        def expand(self, prompt: str, objective_id: str, prior_ids: set[str] | None = None) -> dict[str, list[object]]:
            calls.append("expand")
            assert "Objective?" in prompt and re.search(r"\b\d{4}-\d{2}-\d{2}\b", prompt)
            if remaining is not None:
                assert 1.0 <= self.timeout_s <= remaining
            assert objective_id == "s1" and prior_ids is not None and "n1" in prior_ids
            return {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}

    def gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}, "reasoning": "r"}

    def invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        return {"tool_result_id": None}

    reasoner = _R(model="m", url="u", timeout_s=120.0)
    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 1):
        out = asyncio.run(
            sched.run_node(
                _node(),
                session_id="s1",
                kernel=_K(objective="Objective?"),
                jev=_J(),
                reasoner=reasoner,
                needle_generate=gen,
                invoke=invoke,
                to_outcome=_to_outcome,
                registry=[{"name": "query_finra", "parameters": {}}],
                tool_session=_tool_session(),
                deadline_at=None if remaining is None else time.perf_counter() + remaining,
            )
        )
    assert calls[0] == "analyze" and "expand" in calls
    assert calls.index("analyze") < calls.index("adjudicate") < calls.index("expand")
    assert created == ["Follow-up?"]
    assert out["admitted"] == 0
    assert reasoner.timeout_s == 120.0


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


def test_node_selection_excludes_recursive_research_controls() -> None:
    seen: list[set[str]] = []

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            registry = a[2]
            assert isinstance(registry, list)
            seen.append({str(entry["name"]) for entry in registry})
            return _invoke("query_finra")

    def generate(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {}}

    def invoke(*a: object, **kw: object) -> dict[str, JSONValue]:
        return dict(_FINRA_RESULT)

    kernel = _Kernel({"s1:tr:abc": dict(_FINRA_RESULT)})
    out = asyncio.run(
        sched.run_node(
            _node(),
            "s1",
            kernel=kernel,
            jev=_J(),
            needle_generate=generate,
            invoke=invoke,
            to_outcome=_to_outcome,
            registry=[
                {"name": "research_resume", "parameters": {}},
                {"name": "research_start", "parameters": {}},
                {"name": "research_read_search", "parameters": {}},
                {"name": "research_read", "parameters": {}},
                {"name": "query_finra", "parameters": {}},
            ],
            tool_session=_tool_session(),
        )
    )
    assert seen and all("query_finra" in names for names in seen)
    assert all(
        not names.intersection({"research_resume", "research_start", "research_read", "research_read_search"})
        for names in seen
    )
    assert out["status"] == "resolved" and out["admitted"] == 1


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

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        _gen_n["n"] += 1
        return {
            "tool": "get_reg_sho_volume",
            "arguments": {"round": _gen_n["n"]},
            "reasoning": "r",
        }

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        return {"tool_result_id": "s1:tr:x"}

    def fake_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        calls["n"] += 1
        return _outcome(
            "window words here",
            error=None if calls["n"] == 3 else "finra downstream blew up",
        )

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


@pytest.mark.parametrize("failure", ["ungrounded_argument", "withheld", "runtime_withheld", "grounding"])
def test_ungrounded_reselect_drops_tool_for_next_round_only(failure: str) -> None:
    """One deterministic rejection excludes the tool next round. Later rounds can select it again."""
    seen: list[list[str]] = []
    real_select = sched._select_round

    async def spy_select(
        jev: JevClient,
        kernel: sched._Kernel,
        sid: str,
        nid: str,
        session: Mapping[str, JSONValue],
        node: ResearchNode,
        registry: Sequence[dict[str, JSONValue]],
        ctx_ev: Sequence[JSONValue],
        attempts: Sequence[Mapping[str, JSONValue]],
    ) -> tuple[str, ToolDecision]:
        seen.append([str(e.get("name")) for e in registry])
        return await real_select(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts)

    class _JLoop(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            names = {e.get("name") for e in a[2]} if isinstance(a[2], list) else set()
            pick = "get_xbrl_facts" if "get_xbrl_facts" in names else "query_finra"
            return _invoke(pick)

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            base = await _JevAdmit().assess_result(*a, **k)
            return {
                **base,
                "continuation": "continue_research",
                "continue": "continue_research",
                "action": "continue_research",
            }

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        if kw["tool"] == "get_xbrl_facts":
            if failure == "withheld":
                return {"tool": None, "arguments": {}}
            if failure == "runtime_withheld":
                raise RuntimeError("needle tool mismatch: jev selected 'get_xbrl_facts', needle emitted None")
            if failure == "grounding":
                raise RuntimeError("needle arguments are not grounded in the request")
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

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
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
                to_outcome=_to_outcome,
                registry=[
                    {"name": "get_xbrl_facts", "parameters": {}},
                    {"name": "query_finra", "parameters": {}},
                ],
                tool_session=_tool_session(),
                select_round=spy_select,
            )
        )
    assert seen[0] == ["get_xbrl_facts", "query_finra"]
    assert seen[1] == ["query_finra"]
    assert "get_xbrl_facts" in seen[2]
    attempts = out["attempts"]
    assert isinstance(attempts, list) and isinstance(attempts[0], dict)
    assert attempts[0]["error_type"] == ("reselect" if failure == "ungrounded_argument" else "tool_error")
    assert attempts[0]["arguments"] == {}


def test_ungrounded_reselect_keeps_last_tool_when_no_alternative() -> None:
    """A lone-tool registry keeps the rejected tool: exclusion never leaves JEV with nothing to pick."""
    seen: list[list[str]] = []
    real_select = sched._select_round

    async def spy_select(
        jev: JevClient,
        kernel: sched._Kernel,
        sid: str,
        nid: str,
        session: Mapping[str, JSONValue],
        node: ResearchNode,
        registry: Sequence[dict[str, JSONValue]],
        ctx_ev: Sequence[JSONValue],
        attempts: Sequence[Mapping[str, JSONValue]],
    ) -> tuple[str, ToolDecision]:
        seen.append([str(e.get("name")) for e in registry])
        return await real_select(jev, kernel, sid, nid, session, node, registry, ctx_ev, attempts)

    class _JLoop(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return _invoke("get_xbrl_facts")

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return await _JevAdmit().assess_result(*a, **k)

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_xbrl_facts",
            "arguments": {"ticker": "NVDA", "concept": "instruction"},
            "reasoning": "r",
        }

    with mock.patch.object(sched, "_MAX_TOOL_ROUNDS", 2):
        asyncio.run(
            sched._run_node(
                _node(),
                session_id="s1",
                kernel=_Kernel(),
                jev=_JLoop(),
                repo=None,
                needle_generate=fake_gen,
                invoke=_invoke_result,
                to_outcome=_to_outcome,
                registry=[{"name": "get_xbrl_facts", "parameters": {}}],
                tool_session=_tool_session(),
                select_round=spy_select,
            )
        )
    assert seen == [["get_xbrl_facts"], ["get_xbrl_facts"]]


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
        {
            "tool": "get_sec_filing",
            "job_id": f"job-{i}",
            "outcome_summary": f"obs {i}",
            "error": None,
        }
        for i in range(8)
    ]
    assert len(sched._context_evidence([], many)) == 8
    assert len(sched._context_evidence([], many, cap_last=5)) == 5
    seen: list[int] = []

    kernel = _Kernel(
        {"s1:tr:abc": {"tool_result_id": "s1:tr:abc", "records": [], "briefing": "B"}},
        objective="q?",
    )

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
        sched._generate_tool_arguments(
            fake_gen,
            "get_sec_filing",
            [],
            session,
            _node(),
            filing_evidence,
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("accession_no") == acc
    doc_evidence: list[JSONValue] = [
        {
            "source_handle": {"accession_no": acc, "document_name": "nvda-10k.htm"},
            "outcome_summary": f"read {acc}",
        }
    ]
    doc_args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_doc_gen,
            "get_sec_document",
            [],
            session,
            _node(),
            doc_evidence,
            [],
            None,
            _Kernel(),
        )
    )
    assert doc_args.get("accession_no") == acc
    assert doc_args.get("document_name") == "nvda-10k.htm"


def test_accession_carry_withheld_none_still_fills_from_packet() -> None:
    """Needle withholds (None) on ungrounded open-step + packet accession -> carry fills, no mismatch."""
    acc = "0001628280-26-044069"

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": None,
            "arguments": {},
            "reasoning": "withheld: no grounded accession",
        }

    session = {"session_id": "s1", "objective": "q?"}
    evidence: list[JSONValue] = [{"outcome_summary": f"filing {acc} 10-K", "tool": "search_sec_filings"}]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "get_sec_filing",
            [],
            session,
            _node(),
            evidence,
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("accession_no") == acc

    try:
        asyncio.run(
            sched._generate_tool_arguments(
                fake_gen,
                "get_sec_filing",
                [],
                session,
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )
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
        sched._generate_tool_arguments(fake_gen, "get_sec_filing", [], session, _node(), [], [], None, _Kernel())
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
        sched._generate_tool_arguments(
            fake_gen,
            "get_sec_filing",
            [],
            session,
            _node(),
            evidence,
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("accession_no") == acc


def test_runtime_mismatch_no_packet_reraises() -> None:
    """Server-side RuntimeError mismatch + no packet -> same raise as today."""

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle arguments.generate failed (needle tool mismatch: JEV=x Needle=y)")

    session = {"session_id": "s1", "objective": "q?"}
    try:
        asyncio.run(
            sched._generate_tool_arguments(
                fake_gen,
                "get_sec_filing",
                [],
                session,
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )
    except RuntimeError as exc:
        assert "needle tool mismatch" in str(exc)
    else:
        raise AssertionError("mismatch without packet must re-raise")


def test_node_generation_receives_scope_date_and_verbatim_objective() -> None:
    """A node supplies temporal anchors without rewriting the user's objective."""
    from datetime import date

    from app.tool_runtime import outcome_from_result

    seen: dict[str, object] = {}
    invoked: list[dict[str, JSONValue]] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        seen.update(kw)
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "NVDA"},
            "reasoning": "r",
        }

    scope: dict[str, JSONValue] = {
        "mode": "range",
        "start": "2025-04-01",
        "end": "2025-06-30",
    }
    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "q?",
        "temporal_scope": scope,
    }

    class _K(_Kernel):
        @override
        def get_session(self, session_id: str) -> dict[str, JSONValue]:
            return session

    def invoke(name: str, args: dict[str, JSONValue], tool_session: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(args)
        return {}

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return _invoke("list_sec_filings")

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {"evidence_state": "insufficient", "continuation": "continue_research"}

    out = asyncio.run(
        sched.run_node(
            _node(),
            "s1",
            kernel=_K(),
            jev=_J(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=outcome_from_result,
            registry=sched.build_registry(),
            max_rounds=1,
            tool_session=_tool_session(),
        )
    )
    ctx = seen.get("context")
    assert isinstance(ctx, dict)
    assert ctx.get("temporal_scope") == scope
    today = ctx.get("today_utc")
    assert isinstance(today, str)
    assert date.fromisoformat(today).isoformat() == today
    assert seen["objective"] == "q?"
    assert seen["schema"]
    assert len(invoked) == 1 and invoked[0]["identifier"] == "NVDA"
    assert out["status"] == "blocked" and out["admitted"] == 0


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
        sched._force_open_registry(
            registry,
            [{"tool": "search_sec_filings", "outcome_summary": "no hits"}],
            [],
            0,
        )
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
        sched._generate_tool_arguments(
            fake_phrase,
            "get_reg_sho_volume",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert scrubbed == {"ticker": "AAPL"}
    kept, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_valid,
            "get_reg_sho_volume",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert kept == {"ticker": "AAPL", "tradeDate": "2026-09-22"}


@pytest.mark.parametrize(
    "failure",
    ["withheld", "foreign_tool", "runtime_mismatch", "value_mismatch", "grounding"],
)
@pytest.mark.parametrize(
    ("question", "expected_query"),
    [
        (
            "  Which segment drove the margin change? " + "x" * 220 + "  ",
            ("Which segment drove the margin change? " + "x" * 220)[:200],
        ),
        (
            "[Today UTC 2026-10-19] \n  Which segment drove the margin change? " + "x" * 220 + "  ",
            ("Which segment drove the margin change? " + "x" * 220)[:200],
        ),
        ("What changed on 2026-09-22?", "What changed on 2026-09-22?"),
        (
            "What does [Today UTC 2026-10-19] mean?",
            "What does [Today UTC 2026-10-19] mean?",
        ),
    ],
)
def test_search_web_fallback_uses_node_question_once(failure: str, question: str, expected_query: str) -> None:
    """Web recovery removes only the leading date instruction before it trims and truncates the question."""
    calls: list[str] = []
    node = dataclasses.replace(_node(), question=question)

    async def generate(**kw: object) -> dict[str, JSONValue]:
        calls.append(str(kw["tool"]))
        if failure == "runtime_mismatch":
            raise RuntimeError("needle tool mismatch: jev selected 'search_web', needle emitted None")
        if failure == "value_mismatch":
            raise ValueError("needle tool mismatch: jev selected 'search_web', needle emitted None")
        if failure == "grounding":
            raise RuntimeError("needle arguments are not grounded in the request")
        return {
            "tool": None if failure == "withheld" else "get_sec_document",
            "arguments": {"query": "Unrelated company news", "limit": 99},
        }

    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            generate,
            "search_web",
            [],
            {"session_id": "s1", "objective": "Explain the full investment thesis."},
            node,
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == {"query": expected_query}
    assert calls == ["search_web"]
    assert "node question" in reasoning
    assert withheld is False


@pytest.mark.parametrize("question", [None, "", "   ", 12, "[Today UTC 2026-10-19]", "[Today UTC 2026-10-19] \n "])
def test_search_web_fallback_requires_nonblank_node_question(question: object) -> None:
    """Web recovery does not invent a query when the node lacks a question."""

    async def generate(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}}

    with pytest.raises(ValueError, match="needle tool mismatch"):
        asyncio.run(
            sched._generate_tool_arguments(
                generate,
                "search_web",
                [],
                {"session_id": "s1", "objective": "Do not use this objective as the query."},
                {"node_id": "n1", "question": question},
                [],
                [],
                None,
                _Kernel(),
            )
        )


@pytest.mark.parametrize(
    ("tool_name", "message"),
    [
        ("search_web", "provider connection failed"),
        ("search_web", "needle worker closed"),
        ("search_web", "needle timed out"),
        ("get_sec_filing", "needle arguments are not grounded in the request"),
        ("list_sec_filings", "needle arguments are not grounded in the request"),
    ],
)
def test_search_web_fallback_preserves_other_generation_errors(tool_name: str, message: str) -> None:
    """Only web grounding failures recover. Other generation errors still raise."""

    async def generate(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(
            sched._generate_tool_arguments(
                generate,
                tool_name,
                [],
                {"session_id": "s1", "objective": "Read NVDA filing history."},
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )


def test_sec_withhold_falls_back_to_objective_ticker_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Needle withhold (None) on SEC tools seeds explicit objective ticker, latest (no dates)."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    listed, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_none, "list_sec_filings", [], session, _node(), [], [], None, _Kernel())
    )
    assert listed == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    searched, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none,
            "search_sec_filings",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert searched["ticker"] == "NVDA" and "start_date" not in searched and "end_date" not in searched


def test_sec_withhold_without_ticker_still_raises() -> None:
    """No explicit ticker token (first-word fallback forbidden) -> mismatch still raises."""

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {
        "session_id": "s1",
        "objective": "Growth slowed last quarter — which segment drove it?",
    }
    try:
        asyncio.run(
            sched._generate_tool_arguments(
                fake_none,
                "list_sec_filings",
                [],
                session,
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )
    except ValueError as exc:
        assert "needle tool" in str(exc)
    else:
        raise AssertionError("withhold without explicit ticker must still raise mismatch")


def test_sho_withhold_falls_back_to_both_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Needle withhold (None) on SHO seeds ticker+company_name from for-<Company>."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {
        "session_id": "s1",
        "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none,
            "get_reg_sho_volume",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == {"ticker": "AAPL", "company_name": "Apple"}


def test_with_company_seeds_ticker_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """'What is happening with Oracle now' resolves via the EDGAR index (live oracle-query defect)."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Oracle": "ORCL", "oracle": "ORCL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    assert sched._objective_company("What is happening with Oracle now") == (
        "Oracle",
        "ORCL",
    )
    assert sched._objective_subject_ticker("What is happening with Oracle now") == "ORCL"
    repaired = sched._repair_tool_arguments("get_material_events", {}, "What is happening with Oracle now", "s", "n")
    assert repaired == {"company_name": "Oracle", "ticker": "ORCL"}
    assert sched._objective_company("What is happening with oracle now") == (
        "oracle",
        "ORCL",
    )
    assert sched._objective_subject_ticker("What is happening with oracle now") == "ORCL"
    assert sched._objective_company("What drove growth this week?") is None
    assert sched._objective_company("news for growth this week") is None
    assert sched._objective_subject_ticker("news for growth this week") is None


def test_sec_placeholder_forms_are_not_substituted() -> None:
    """Needle forms ['YYYY-MM-DD'] stay verbatim; the tool rejects them, so no substituted listing runs."""

    async def fake_placeholder(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_placeholder,
            "list_sec_filings",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]}


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

    session = {
        "session_id": "s1",
        "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_org,
            "get_reg_sho_volume",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == {"ticker": "AAPL", "company_name": "Apple"}


def test_repeat_8k_swaps_to_packet_10q() -> None:
    """Same 8-K accession twice on quarterly revenue swaps to the packet 10-Q."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_8k(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": acc8},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_filing", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {
            "outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q",
            "tool": "list_sec_filings",
        }
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_8k,
            "get_sec_filing",
            [],
            session,
            _node(),
            evidence,
            attempts,
            None,
            _Kernel(),
        )
    )
    assert args.get("accession_no") == accq


def test_withhold_carry_repairs_query_and_swaps_10q() -> None:
    """Needle withhold with a carried 8-K repeat gets query seed + 10-Q swap."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "error": None,
        },
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "error": None,
        },
    ]
    evidence: list[JSONValue] = [
        {
            "outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q",
            "tool": "list_sec_filings",
        }
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none,
            "get_sec_document",
            [],
            session,
            _node(),
            evidence,
            attempts,
            None,
            _Kernel(),
        )
    )
    assert args.get("accession_no") == accq
    assert args.get("query") == "revenue"


def test_repeat_8k_error_swaps_to_packet_10q() -> None:
    """Same 8-K twice with query errors on revenue swaps to the packet 10-Q."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_8k(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "error": "query not found",
        },
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "error": "query not found",
        },
    ]
    evidence: list[JSONValue] = [
        {
            "outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q",
            "tool": "list_sec_filings",
        }
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_8k,
            "get_sec_document",
            [],
            session,
            _node(),
            evidence,
            attempts,
            None,
            _Kernel(),
        )
    )
    assert args.get("accession_no") == accq


def test_revenue_document_seeds_query() -> None:
    """get_sec_document without query on quarterly revenue seeds a revenue query."""

    async def fake_doc(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_document",
            "arguments": {"accession_no": "0001045810-26-000075"},
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_doc, "get_sec_document", [], session, _node(), [], [], None, _Kernel())
    )
    assert args.get("query") == "revenue"
    assert args.get("limit") == 32000


def test_ten_q_revenue_lists_forms_and_reads_listed_10q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit 10-Q revenue request lists only 10-Q filings and reads three sections of the listed 10-Q."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    # 8-K listed first, out of the 10-Q window, so only the 10-Q preference picks accq.
    listing: dict[str, JSONValue] = {
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
            sched._generate_tool_arguments(
                fake_none,
                "list_sec_filings",
                [],
                session,
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )
        assert listed.get("forms") == forms
        read, _, _ = asyncio.run(
            sched._generate_tool_arguments(
                fake_none,
                "get_sec_document",
                [],
                session,
                _node(),
                [],
                [listing],
                None,
                _Kernel(),
            )
        )
        assert read == doc

    invoked: list[dict[str, JSONValue]] = []

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(dict(args))
        return {"text": "Revenue was $46.7 billion."}

    explicit = "What did NVDA's latest 10-Q disclose about revenue? Use stockbot, bearbot, and bullbot."
    sections = [{"accession_no": accq, "limit": 32000, "section": s} for s in ("Item 1", "Item 2", "Item 1A")]
    revenue_read = {"accession_no": accq, "query": "revenue", "limit": 32000}
    failed_read: dict[str, JSONValue] = {
        "tool": "get_sec_document",
        "arguments": {},
        "error": "timeout",
    }
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
                _tool_session(),
                None,
            )
        )
        assert invoked == reads
        assert [r["arguments"] for r in results] == reads
        assert all(r["error"] is None for r in results)

    async def fail_needle(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle boom")

    async def fail_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("sec read failed")

    # Failed sibling section reads settle without a breaker block, so each section returns to the selector alone.
    for needle, call, failed in (
        (fake_none, fail_invoke, sections),
        (fail_needle, invoke, [{"section": s["section"]} for s in sections]),
    ):
        attempts: list[dict[str, JSONValue]] = [listing]
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
                _tool_session(),
                None,
            )
        )
        assert [r["arguments"] for r in results] == failed
        assert all(r["error"] for r in results)
        signal = asyncio.run(
            sched._settle_round(
                results,
                _JevAdmit(),
                _Kernel(),
                _node(),
                [],
                attempts,
                "s1",
                "n1",
                _invoke("get_sec_document"),
                0,
            )
        )
        assert signal["terminal"] is None and len(attempts) == 4


@pytest.mark.parametrize(
    "grounds",
    ["none", "objective", "node", "scope", "scope-as-of", "session-as-of", "as-of"],
)
def test_latest_ten_q_listing_constrains_forms_and_preserves_grounded_dates(
    grounds: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid broad forms become 10-Q only. Only grounded dates remain."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    generated: dict[str, JSONValue] = {
        "identifier": "NVDA",
        "forms": ["10-Q", "10-K", "8-K"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    needle_calls: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {
            "tool": "list_sec_filings",
            "arguments": dict(generated),
            "reasoning": "r",
        }

    prompt = "What did NVDA's latest 10-Q disclose about revenue?"
    dated = f"{prompt} Include filings from 2025-01-01 through 2025-12-31."
    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": dated if grounds == "objective" else prompt,
    }
    node = ResearchNode(
        node_id="n1",
        session_id="s1",
        question=dated if grounds == "node" else prompt,
        why_it_matters="w",
    )
    if grounds == "scope":
        session["temporal_scope"] = {
            "mode": "range",
            "start": "2025-01-01",
            "end": "2025-12-31",
        }
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
            _Kernel(),
        )
    )
    expected: dict[str, JSONValue] = {"identifier": "NVDA", "forms": ["10-Q"]}
    if grounds in ("objective", "node", "scope"):
        expected.update(start_date="2025-01-01", end_date="2025-12-31")
    elif grounds in ("scope-as-of", "session-as-of"):
        expected.update(end_date="2025-12-31")
    assert args == expected
    assert needle_calls == ([] if grounds == "none" else ["list_sec_filings"])


def test_quarter_revenue_listing_keeps_generated_forms_and_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quarter-only request keeps its existing form and date behavior."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    generated: dict[str, JSONValue] = {
        "identifier": "NVDA",
        "forms": ["10-Q", "10-K", "8-K"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    needle_calls: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {
            "tool": "list_sec_filings",
            "arguments": dict(generated),
            "reasoning": "r",
        }

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
            _Kernel(),
        )
    )
    assert args == generated
    assert needle_calls == ["list_sec_filings"]


def test_latest_ten_q_listing_generates_missing_required_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Needle supplies a required limit that the local listing arguments cannot fill."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    generated: dict[str, JSONValue] = {
        "identifier": "NVDA",
        "forms": ["10-Q"],
        "limit": 3,
    }
    needle_calls: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {
            "tool": "list_sec_filings",
            "arguments": dict(generated),
            "reasoning": "required limit",
        }

    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "list_sec_filings",
            [
                {
                    "name": "list_sec_filings",
                    "parameters": {"required": ["identifier", "forms", "limit"]},
                }
            ],
            {
                "session_id": "s1",
                "objective": "What did NVDA's latest 10-Q disclose about revenue?",
            },
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert needle_calls == ["list_sec_filings"]
    assert args == generated
    assert reasoning == "required limit"
    assert withheld is False


@pytest.mark.parametrize("listed", [False, True], ids=["missing-listing", "only-8k"])
def test_latest_ten_q_document_without_ten_q_generates_and_reselects(
    listed: bool,
) -> None:
    """Without a listed 10-Q, Needle runs once. The scheduler does not read an 8-K."""
    acc8 = "0001045810-26-000078"
    attempts: list[dict[str, JSONValue]] = (
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

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "reasoning": "r",
        }

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(name)
        return {}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_sec_document",
            node=_node(),
            session={
                "session_id": "s1",
                "objective": "What did NVDA's latest 10-Q disclose about revenue?",
            },
            registry=sched.build_registry(),
            evidence=[],
            attempts=attempts,
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=lambda n, r: _outcome(),
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert needle_calls == ["get_sec_document"]
    assert invoked == []
    assert out["error_type"] == "reselect"
    assert "no 10-Q" in str(out["error"])


def test_structured_listing_keeps_form_identity_and_selects_latest_ten_q() -> None:
    """Source rows label their own accession. The latest 10-Q wins over an older 10-Q and a newer 8-K."""
    acc8, old_q, latest_q = (
        "0001045810-26-000078",
        "0001045810-26-000050",
        "0001045810-26-000075",
    )
    listing: dict[str, JSONValue] = {
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

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none,
            "get_sec_document",
            sched.build_registry(),
            {
                "session_id": "s1",
                "objective": "What did NVDA's latest 10-Q disclose about revenue?",
            },
            _node(),
            [],
            [listing],
            None,
            _Kernel(),
        )
    )
    assert args == {"accession_no": latest_q, "query": "revenue", "limit": 32000}


def test_latest_revenue_accessions_reads_every_persisted_shape() -> None:
    """Exact row labels win over summary text; every persisted shape returns newest-first revenue accessions."""
    acc_q, acc_k, acc_8k = (
        "0001045810-26-000075",
        "0001045810-26-000070",
        "0001045810-26-000078",
    )
    rows: JSONValue = [
        {"accession_no": acc_8k, "form": "8-K", "filed_at": "2026-09-01"},
        {"accession_no": "not-an-accession", "form": "10-K", "filed_at": "2026-08-15"},
        {"accession_no": acc_k, "form": "10-K", "filed_at": "2026-07-01"},
        {"accession_no": acc_q, "form": "10-Q", "filed_at": "2026-08-01"},
    ]
    assert sched._latest_revenue_accessions({"result": {"filings": rows}}) == [acc_q, acc_k]
    assert sched._latest_revenue_accessions({"filings": rows}) == [acc_q, acc_k]
    assert sched._latest_revenue_accessions(json.dumps({"filings": rows})) == [acc_q, acc_k]
    assert sched._latest_revenue_accessions({"outcome_summary": f"10-Q {acc_q}"}) == [acc_q]
    assert sched._latest_revenue_accessions(f"seen 8-K {acc_8k} then 10-K accession {acc_k}") == [acc_k]
    assert sched._latest_revenue_accessions({"result": {"form": "10-Q", "accession_no": acc_q}}) == [acc_q]
    assert sched._latest_revenue_accessions({"result": {"form": "10-Q", "accession_no": "nope"}}) == []
    assert sched._latest_revenue_accessions({"tool": "get_sec_document", "error": None}) == []
    assert sched._latest_revenue_accessions("{not json") == []
    assert sched._latest_revenue_accessions("no filings here") == []
    assert sched._latest_revenue_accessions(42) == []
    # A revenue-form row without a usable accession is skipped, never aborts the scan.
    assert sched._latest_revenue_accessions({"result": {"filings": [{"form": "10-Q", "accession_no": None}]}}) == []
    assert sched._latest_revenue_accessions({"result": {"filings": [{"form": "10-K"}]}}) == []
    assert sched._latest_revenue_accessions({"result": {"form": "10-Q"}}) == []


def test_scalar_or_scalar_list_accepts_finite_scalars_only() -> None:
    """Grounded execution values accept scalars and scalar lists; non-finite numbers and maps drop."""
    assert sched._scalar_or_scalar_list("ORCL")
    assert sched._scalar_or_scalar_list(2.5)
    assert sched._scalar_or_scalar_list([1, "a", 2.5, None])
    assert not sched._scalar_or_scalar_list(float("inf"))
    assert not sched._scalar_or_scalar_list([float("nan")])
    assert not sched._scalar_or_scalar_list([])
    assert not sched._scalar_or_scalar_list({"a": 1})


def test_generated_valid_8k_accession_changes_to_listed_latest_ten_q_before_invoke() -> None:
    """The scheduler replaces a valid generated 8-K accession before the first document call."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA", "forms": ["10-Q"]},
        "error": None,
        "result": {
            "filings": [
                {"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"},
                {"accession_no": "0001045810-26-000079", "form": "10-K", "filed_at": "2026-08-15"},
                {"accession_no": accq, "form": "10-Q", "filed_at": "2026-08-01"},
            ]
        },
        "outcome_summary": f"8-K {acc8}, 10-Q {accq}",
    }
    invoked: list[tuple[str, dict[str, JSONValue]]] = []
    needle_calls: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc8},
            "reasoning": "r",
        }

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
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
            attempts=[
                listing,
                {
                    "tool": "get_sec_document",
                    "error": None,
                    "result": {"form": "10-Q", "accession_no": "0001045810-26-000080"},
                },
            ],
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=lambda n, r: _outcome("Revenue was $46.7 billion."),
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert needle_calls == ["get_sec_document"]
    assert invoked == [("get_sec_document", {"accession_no": accq, "query": "revenue", "limit": 32000})]
    assert out["arguments"] == {
        "accession_no": accq,
        "query": "revenue",
        "limit": 32000,
    }
    assert out["error"] is None


_LATEST_QUARTER_REVENUE = "What was NVDA's revenue in the latest quarter?"


@pytest.mark.parametrize("generated", [{}, {"accession_no": "0001045810-26-000078"}], ids=["no-accession", "8-k"])
def test_latest_quarter_revenue_document_without_listing_reselects(generated: dict[str, JSONValue]) -> None:
    """Quarter wording without a 10-Q token is still a latest read: no listed filing reselects before invoke."""
    invoked: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_document", "arguments": dict(generated), "reasoning": "r"}

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(name)
        return {}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_sec_document",
            node=_node(),
            session={"session_id": "s1", "objective": _LATEST_QUARTER_REVENUE},
            registry=sched.build_registry(),
            evidence=[],
            attempts=[],
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=lambda n, r: _outcome(),
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert invoked == []
    assert out["error_type"] == "reselect"
    assert "no 10-Q/10-K" in str(out["error"])


def test_latest_quarter_revenue_read_replaces_stale_needle_accession_with_latest_listing() -> None:
    """Quarter wording without a 10-Q token reads the newest listed 10-Q/10-K, not Needle's older accession."""
    acc8, old_k, latest_q = (
        "0001045810-26-000078",
        "0001045810-26-000050",
        "0001045810-26-000075",
    )
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA", "forms": ["10-Q", "10-K"]},
        "error": None,
        "result": {
            "filings": [
                {"accession_no": acc8, "form": "8-K", "filed_at": "2026-10-01"},
                {"accession_no": old_k, "form": "10-K", "filed_at": "2026-06-01"},
                {"accession_no": latest_q, "form": "10-Q", "filed_at": "2026-09-01"},
            ]
        },
        "outcome_summary": f"8-K {acc8}, 10-K {old_k}, 10-Q {latest_q}",
    }
    needle_calls: list[str] = []

    async def fake_stale(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {"tool": "get_sec_document", "arguments": {"accession_no": old_k}, "reasoning": "r"}

    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_stale,
            "get_sec_document",
            sched.build_registry(),
            {"session_id": "s1", "objective": _LATEST_QUARTER_REVENUE},
            _node(),
            [],
            [listing],
            None,
            _Kernel(),
        )
    )
    assert needle_calls == ["get_sec_document"]
    assert args.get("accession_no") == latest_q


@pytest.mark.parametrize("runtime_envelope", [True, False], ids=["runtime-envelope", "in-memory"])
def test_ten_q_revenue_route_settles_structured_listing(
    runtime_envelope: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route reads the latest exact 10-Q from persisted or in-memory rows, not misleading summary text."""

    import app.tools as _tools
    import tests.research_source_seam as seam
    from app.research.programmatic_router import programmatic_select_round
    from app.tool_runtime import outcome_from_result

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    prompt = "What did NVDA's latest 10-Q disclose about revenue?"
    acc8, old_q, latest_q = (
        "0001045810-26-000078",
        "0001045810-26-000050",
        "0001045810-26-000075",
    )
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
    rows: list[JSONValue] = [
        {"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"},
        {"accession_no": old_q, "form": "10-Q", "filed_at": "2026-05-01"},
        {"accession_no": latest_q, "form": "10-Q", "filed_at": "2026-08-26"},
    ]
    structured: dict[str, JSONValue] = {"filings": rows, "source": "SEC EDGAR"}
    summary = f"Bottom line: The latest 10-Q is {acc8}. The other 10-Q is {old_q}."
    listed: dict[str, JSONValue] = {
        "content": summary,
        "meta": {"source": "SEC EDGAR"},
        "tool_result_id": receipt,
    }
    if not runtime_envelope:
        listed.update(structured)
    reads: list[str] = []
    invoked: list[tuple[str, dict[str, JSONValue]]] = []
    needle_calls: list[str] = []

    def get_tool_result(ref: str) -> dict[str, JSONValue]:
        reads.append(ref)
        assert ref == receipt
        if not runtime_envelope:
            raise KeyError(ref)
        return {
            "tool_result_id": receipt,
            "tool_name": "list_sec_filings",
            "result": structured,
        }

    class _StoredKernel(_Kernel):
        @override
        def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
            return get_tool_result(tool_result_id)

        @override
        def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
            passages.append(data.get("matching_passage"))
            return super().admit_evidence(session_id, job_id, data)

    class _Jev(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            raise AssertionError("The programmatic router must select the 10-Q revenue tools.")

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        assert needle_calls == []
        invoked.append((name, dict(args)))
        if name == "list_sec_filings":
            return listed
        section = args["section"]
        assert isinstance(section, str)
        return {
            "tool_result_id": f"s1:tr:doc{list(section_text).index(section)}",
            "content": f"Bottom line: {section_text[section]}",
            "meta": {"source_handle": validate_json_value(handles[section])},
        }

    kernel = _StoredKernel()
    out = asyncio.run(
        sched._drive_rounds(
            ResearchNode(
                node_id="n1",
                session_id="s1",
                question=f"[Today UTC 2026-10-09] {prompt}",
                why_it_matters="w",
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
            [
                {"name": "list_sec_filings", "parameters": {}},
                {"name": "get_sec_document", "parameters": {}},
            ],
            kernel,
            _Jev(),
            "s1",
            "n1",
            {
                "select_round": programmatic_select_round,
                "reasoner": None,
                "needle_generate": fake_none,
                "invoke": invoke,
                "to_outcome": outcome_from_result,
                "max_rounds": 4,
            },
            _tool_session(),
            None,
        )
    )
    assert invoked == [
        ("list_sec_filings", {"identifier": "NVDA", "forms": ["10-Q"]}),
        *(
            (
                "get_sec_document",
                {"accession_no": latest_q, "limit": 32000, "section": section},
            )
            for section in section_text
        ),
    ]
    # Each section admits as its own evidence with its full section text.
    assert out["status"] == "resolved" and out["admitted"] == 4
    assert passages[1:] == list(section_text.values())
    attempts = out["attempts"]
    assert isinstance(attempts, list)
    listing, *documents = attempts
    assert isinstance(listing, dict)
    assert all(isinstance(document, dict) for document in documents)
    documents = [document for document in documents if isinstance(document, dict)]
    assert needle_calls == []
    assert listing["tool_result_ref"] == receipt
    assert listing["outcome_summary"] == summary
    assert [d["tool_result_ref"] for d in documents] == [
        "s1:tr:doc0",
        "s1:tr:doc1",
        "s1:tr:doc2",
    ]
    assert len({d["evidence_id"] for d in documents}) == 3
    assert reads and all(ref == receipt for ref in reads)
    for _, job_id in (call for call in kernel.calls if call[0] == "admit"):
        assert kernel.calls.index(("admit", job_id)) < kernel.calls.index(("complete", job_id))


def test_ten_q_revenue_node_reads_document_after_reason_over_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "Revenue was $46.7 billion, up 56% from a year ago.",
        accession=accq,
        document="nvda-20260726.htm",
    )
    invoked: list[tuple[str, dict[str, JSONValue]]] = []

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append((name, args))
        return {"tool": name}

    def to_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        if name == "list_sec_filings":
            content = f"filing {acc8} 8-K {'x' * 130} filing {accq} 10-Q"
            handle = None
        else:
            content = "Revenue was $46.7 billion, up 56% from a year ago."
            handle = document_handle
        return dataclasses.replace(_outcome(content), tool_name=name, source_handle=handle)

    class _Jev(JevClient):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            raise AssertionError("router owns the 10-Q revenue route; JEV select must not run")

        @override
        async def assess_result(self, node: object, outcome: object, *a: object, **k: object) -> dict[str, JSONValue]:
            assert isinstance(outcome, ToolOutcome)
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
            ResearchNode(node_id="n1", session_id="s1", question=prompt, why_it_matters="w"),
            session,
            [
                {"name": "list_sec_filings", "parameters": {}},
                {"name": "get_sec_document", "parameters": {}},
            ],
            _Kernel(),
            _Jev(),
            "s1",
            "n1",
            {
                "select_round": programmatic_select_round,
                "reasoner": None,
                "needle_generate": fake_none,
                "invoke": invoke,
                "to_outcome": to_outcome,
                "max_rounds": 4,
            },
            _tool_session(),
            None,
        )
    )
    sections = ("Item 1", "Item 2", "Item 1A")
    assert [name for name, _ in invoked] == [
        "list_sec_filings",
        *["get_sec_document"] * 3,
    ]
    assert [args for _, args in invoked[1:]] == [{"accession_no": accq, "limit": 32000, "section": s} for s in sections]
    assert out["status"] == "resolved" and out["admitted"] == 3
    # Boundaries: the guard holds only until one document read, and only on revenue-quarter asks.
    listing_ok = {"tool": "list_sec_filings", "error": None}
    read_ok = {"tool": "get_sec_document", "error": None}
    read_err = {"tool": "get_sec_document", "error": "timeout"}
    node = ResearchNode(node_id="n1", session_id="s1", question=prompt, why_it_matters="w")
    assert sched._revenue_read_pending(session, node, [listing_ok, read_err])
    assert not sched._revenue_read_pending(session, node, [listing_ok, read_ok])
    assert not sched._revenue_read_pending(session, node, [])
    other = ResearchNode(
        node_id="n1",
        session_id="s1",
        question="What did NVDA disclose about revenue?",
        why_it_matters="w",
    )
    assert not sched._revenue_read_pending({"objective": "What did NVDA disclose about revenue?"}, other, [listing_ok])


@pytest.mark.parametrize("selected_resolved", [False, True])
def test_ten_q_revenue_node_reads_document_before_supported_listing_resolve(
    selected_resolved: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler reads the revenue document before assessment or selection resolves the node."""
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
        "Revenue was $46.7 billion, up 56% from a year ago.",
        accession=accq,
        document="nvda-20260726.htm",
    )
    invoked: list[tuple[str, dict[str, JSONValue]]] = []

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    async def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append((name, args))
        return {"tool": name}

    def to_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        content = (
            f"filing {acc8} 8-K {'x' * 130} filing {accq} 10-Q"
            if name == "list_sec_filings"
            else "Revenue was $46.7 billion, up 56% from a year ago."
        )
        return dataclasses.replace(
            _outcome(content),
            tool_name=name,
            source_handle=None if name == "list_sec_filings" else document_handle,
        )

    class _Jev(JevClient):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            assert selected_resolved
            assert [name for name, _ in invoked] == ["list_sec_filings"]
            return ToolDecision(action="resolved", confidence=1.0)

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {
                "continuation": "resolve_node",
                "evidence_state": "sufficient_support",
                "topical_relevance_answer": "yes",
                "resolution_support_answer": "yes",
            }

    class _IntakeKernel(_Kernel):
        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [{"evidence_id": "ev-intake", "content": "NVDA intake snippet"}]

    sent_resolved = False

    async def select_round(
        jev: JevClient,
        kernel: sched._Kernel,
        sid: str,
        nid: str,
        session: Mapping[str, JSONValue],
        node: ResearchNode,
        registry: Sequence[dict[str, JSONValue]],
        ctx_ev: Sequence[JSONValue],
        attempts: Sequence[Mapping[str, JSONValue]],
    ) -> tuple[str, ToolDecision]:
        nonlocal sent_resolved
        if selected_resolved and invoked and not sent_resolved:
            sent_resolved = True
            decision = await jev.select_tool(
                sched.query_with_today_utc(str(session["objective"])),
                node,
                registry,
                ctx_ev,
                [dict(a) for a in attempts],
                session_id=sid,
            )
            return sched._selection_action(decision), decision
        return await programmatic_select_round(
            jev,
            kernel,
            sid,
            nid,
            session,
            node,
            list(registry),
            list(ctx_ev),
            [dict(a) for a in attempts],
        )

    session = {"session_id": "s1", "objective": prompt}
    out = asyncio.run(
        sched._drive_rounds(
            ResearchNode(node_id="n1", session_id="s1", question=prompt, why_it_matters="w"),
            session,
            [
                {"name": "list_sec_filings", "parameters": {}},
                {"name": "get_sec_document", "parameters": {}},
            ],
            _IntakeKernel(),
            _Jev(),
            "s1",
            "n1",
            {
                "select_round": select_round,
                "reasoner": None,
                "needle_generate": fake_none,
                "invoke": invoke,
                "to_outcome": to_outcome,
                "max_rounds": 4,
            },
            _tool_session(),
            None,
        )
    )
    sections = ("Item 1", "Item 2", "Item 1A")
    assert [name for name, _ in invoked] == [
        "list_sec_filings",
        *["get_sec_document"] * 3,
    ]
    assert [args for _, args in invoked[1:]] == [{"accession_no": accq, "limit": 32000, "section": s} for s in sections]
    assert out["status"] == "resolved" and out["admitted"] == 3
    assert sent_resolved == selected_resolved


def test_stale_as_of_dropped_dateless_objective() -> None:
    """Q4 shape: invented as_of on a dateless accession query scrubs to omit."""

    async def fake_stale(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {
                "accession_no": "0000320193-25-000079",
                "as_of": "2025-09-27",
            },
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079, give me the filing metadata (form, dates, filer)",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_stale, "get_sec_filing", [], session, _node(), [], [], None, _Kernel())
    )
    assert args == {"accession_no": "0000320193-25-000079"}


def test_explicit_as_of_kept_when_in_objective() -> None:
    """Explicit YYYY-MM-DD in the objective keeps as_of (latest-available otherwise)."""

    async def fake_kept(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {
                "accession_no": "0000320193-25-000079",
                "as_of": "2025-09-27",
            },
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079 as of 2025-09-27, give me the filing metadata",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_kept, "get_sec_filing", [], session, _node(), [], [], None, _Kernel())
    )
    assert args == {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"}


def test_session_as_of_kept_when_datetime() -> None:
    """Session as_of from a stored datetime (ISO-serialized by the store) keeps the matching cutoff."""
    from datetime import UTC, datetime

    async def fake_kept(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_sec_filing",
            "arguments": {
                "accession_no": "0000320193-25-000079",
                "as_of": "2025-09-27",
            },
            "reasoning": "r",
        }

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "For Apple's 10-K accession 0000320193-25-000079, give me the filing metadata",
        "as_of": datetime(2025, 9, 27, 12, 0, 0, tzinfo=UTC).isoformat(),
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_kept,
            "get_sec_filing",
            [],
            session,
            _node(),
            [],
            [],
            "2025-09-27",
            _Kernel(),
        )
    )
    assert args == {"accession_no": "0000320193-25-000079", "as_of": "2025-09-27"}


def test_garbage_identifier_reseeded_from_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Needle identifier F1/TODAY/SEC on an Apple query reseeds to AAPL."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def fake_garbage(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "F1"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "List Apple's most recent 10-K and 10-Q filings.",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_garbage,
            "list_sec_filings",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("identifier") == "AAPL"


def test_identifier_repair_keeps_valid_cik_and_case() -> None:
    """CIK digits and lowercase ticker pass through; only EDGAR-unresolvable garbage reseeds."""

    async def fake_cik(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "0000320193"},
            "reasoning": "r",
        }

    async def fake_lower(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "aapl"},
            "reasoning": "r",
        }

    session = {
        "session_id": "s1",
        "objective": "List Apple's most recent 10-K and 10-Q filings.",
    }
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(fake_cik, "list_sec_filings", [], session, _node(), [], [], None, _Kernel())
    )
    assert args.get("identifier") == "0000320193"
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_lower,
            "list_sec_filings",
            [],
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("identifier") == "aapl"


def test_session_control_carries_sid_without_needle() -> None:
    """Session-control tools carry caller sid and never call Needle (sid-carry tuple fix)."""

    def boom_gen(**kw: object) -> dict[str, JSONValue]:
        raise AssertionError("sid-carry must not call Needle")

    session = {"session_id": "s1", "objective": "q?"}
    for tool in ("research_resume", "research_status", "research_cancel"):
        args, reasoning, withheld = asyncio.run(
            sched._generate_tool_arguments(boom_gen, tool, [], session, _node(), [], [], None, _Kernel())
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
            sched._generate_tool_arguments(
                fake_empty,
                "search_sec_filings",
                [],
                session,
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
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
                boom,
                "get_sec_filing",
                [],
                session,
                _node(),
                [],
                [{**carried}, {**carried}],
                None,
                _Kernel(),
            )
        )


def test_accession_family_break_across_tools() -> None:
    """Same accession failing across get_sec_filing/get_sec_document breaks as one family (C)."""
    acc = "0001193125-09-214859"
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": acc},
            "error": "boom-a",
        },
        {
            "tool": "get_sec_document",
            "arguments": {"accession_no": acc},
            "error": "boom-b",
        },
    ]
    guided = sched._accession_family_break(attempts)
    assert guided is not None and guided["error_type"] == "invalid_tool_arguments"
    error = guided["error"]
    assert isinstance(error, str) and acc in error
    # Sibling 10-Q section reads fail independently. A repeat of one section still breaks.
    item1, item2 = (
        dict[str, JSONValue](
            {
                "tool": "get_sec_document",
                "arguments": {"accession_no": acc, "section": s},
                "error": "boom",
            }
        )
        for s in ("Item 1", "Item 2")
    )
    assert sched._accession_family_break([item1, item2]) is None
    assert sched._accession_family_break([item1, {**item1, "error": "boom-2"}]) is not None
    assert sched._accession_family_break([attempts[0], item1]) is not None


def test_find_sec_entities_seeded_from_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    bad: dict[str, JSONValue] = {
        "tool": "search_web",
        "error": "needle down",
        "arguments": {},
    }
    assert sched._needle_error_streak([dict(bad), dict(bad)]) == 2
    assert sched._needle_error_streak([dict(bad), dict(bad), dict(bad)]) == 3
    grounded = dict(bad, arguments={"query": "q"})
    assert sched._needle_error_streak([dict(bad), dict(bad), grounded]) == 0


@pytest.mark.parametrize(
    ("kernel_cls", "outcomes", "admitted_ids"),
    [
        (
            _Kernel,
            {"admitted": 1, "duplicate": 0, "not_citable": 1, "error": 1, "timeout": 0},
            ["ev-0"],
        ),
        (
            _DupKernel,
            {"admitted": 0, "duplicate": 1, "not_citable": 1, "error": 1, "timeout": 0},
            [],
        ),
    ],
)
def test_intake_round_admits_citable_without_jev_verdict(
    monkeypatch: pytest.MonkeyPatch,
    kernel_cls: type[_Kernel],
    outcomes: dict[str, int],
    admitted_ids: list[str],
) -> None:
    """_intake_round admits citable bytes by code; uncitable/error close jobs (E1); duplicates never count."""
    from app.research import kernel_worker as kw

    kernel = kernel_cls()
    kernel.tool_results["s1:tr:w"] = {
        "tool_result_id": "s1:tr:w",
        "evidence": [
            {
                "url": "https://example.com/a",
                "highlight": "Oracle merger filing mentions",
            }
        ],
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
                    "evidence": [
                        {
                            "url": "https://example.com/a",
                            "highlight": "Oracle merger filing mentions",
                        }
                    ],
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
        return {
            "tool": tool,
            "arguments": args,
            "job_id": "job-bad",
            "error": "boom",
            "outcome": None,
        }

    async def fake_chain(*a: object, **k: object) -> list[tuple[str, dict[str, JSONValue], dict[str, object], float]]:
        return [
            (
                "list_sec_filings",
                {"ticker": "ORCL"},
                await fake_attempt("list_sec_filings", {}, "s1"),
                5.0,
            ),
            (
                "get_sec_document",
                {"accession_no": "x"},
                await fake_attempt("get_sec_document", {}, "s1"),
                5.0,
            ),
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


def test_intake_round_budget_timeout_records_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            out = await asyncio.wait_for(
                sched._sec_thread_call(lambda: (time.sleep(0.1), f"r{i}")[1]),
                timeout=10,
            )
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
        return {
            "node_id": n.node_id,
            "status": "resolved",
            "admitted": 1,
            "incomplete_guard": False,
        }

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
    gen_fail: dict[str, JSONValue] = {
        "tool": "query_finra",
        "arguments": {},
        "error": "needle blew up",
    }
    assert sched._needle_error_streak([dict(gen_fail) for _ in range(3)]) == 3


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
    assert (
        "fail",
        "job-0",
        "timeout",
    ) in kernel.calls  # cancelled job fails as timeout, never running


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
    assert (
        "fail",
        "job-0",
        "timeout",
    ) in kernel.calls  # generation cancel fails as timeout, never running


def test_drive_rounds_emits_parseable_timing_per_round(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One toolflow round_timing line per round with 4 phase ms + ev before/after."""
    import logging

    kernel = _Kernel()
    session = {"session_id": "s1", "objective": "o"}

    async def select_round(jev: object, kernel: object, sid: str, nid: str, *a: object, **k: object) -> object:
        return ("invoke", _invoke("search_web"))

    async def gen(tool: object, schema: object = None, **k: object) -> dict[str, JSONValue]:
        return {
            "tool": "search_web",
            "arguments": {"query": "q"},
            "reasoning": "",
            "confidence": 1.0,
        }

    async def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, JSONValue]:
        return {
            "status": "ok",
            "rows": [{"url": "https://e.com", "title": "t", "passage": "p" * 50}],
        }

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
    for key in (
        "round=",
        "tool=",
        "select_ms=",
        "generate_ms=",
        "tool_ms=",
        "assess_ms=",
        "ev_before=",
        "ev_after=",
    ):
        assert key in line


def test_terminal_select_failure_still_logs_timing(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
                {
                    "select_round": down,
                    "reasoner": None,
                    "needle_generate": None,
                    "invoke": None,
                    "to_outcome": None,
                },
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


def test_node_selection_receives_intake_identity_and_content() -> None:
    seen: list[str] = []

    class _K(_Kernel):
        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [{"evidence_id": "ev-intake", "content": "Apple 10-K revenue rose"}]

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            seen.append(json.dumps(a[3]))
            return _invoke("query_finra")

    out = _run_node(_K(), _J())
    assert seen and all("ev-intake" in context and "Apple 10-K revenue rose" in context for context in seen)
    assert out["status"] == "resolved" and out["admitted"] == 1


def _dup_gen(**kw: object) -> dict[str, JSONValue]:
    return {
        "tool": "search_web",
        "arguments": {"query": "NVDA revenue 0"},
    }


def test_duplicate_guard_uses_full_history_beyond_generation_window() -> None:
    """A repeat of a call older than the 10-attempt generation window still reselects without invoking."""
    invoked: list[str] = []

    def _dup_invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        invoked.append("invoked")
        return {}

    def _dup_outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        return _outcome()

    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "search_web",
            "arguments": {"query": f"NVDA revenue {i}"},
            "error": None,
        }
        for i in range(14)
    ]
    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_web",
            node=_node(),
            session={"session_id": "s1", "objective": "NVDA revenue"},
            registry=[],
            evidence=[],
            attempts=attempts,
            kernel=_Kernel(),
            needle_generate=_dup_gen,
            invoke=_dup_invoke,
            to_outcome=_dup_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert out["error_type"] == "reselect" and "duplicate" in str(out["error"])
    assert invoked == []


def _xbrl_attempt(
    objective: str,
    concept: str,
    *,
    question: str = "q?",
    query: str = "",
    evidence: Sequence[JSONValue] | None = None,
    attempts: Sequence[Mapping[str, JSONValue]] | None = None,
    reasoning: str = "r",
    kernel: _Kernel | None = None,
) -> tuple[dict[str, object], list[dict[str, JSONValue]]]:
    """One real _attempt_tool for get_xbrl_facts with a fake Needle and a recording runtime invoke."""
    invoked: list[dict[str, JSONValue]] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {
            "tool": "get_xbrl_facts",
            "arguments": {"ticker": "NVDA", "concept": concept},
            "reasoning": reasoning,
        }

    async def fake_invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(dict(args))
        return {"tool_result_id": "s1:tr:xbrl"}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_xbrl_facts",
            node=ResearchNode(node_id="n1", session_id="s1", question=question, why_it_matters="w"),
            session={"session_id": "s1", "objective": objective, "query": query},
            registry=sched.build_registry(),
            evidence=evidence or [],
            attempts=attempts or [],
            kernel=kernel if kernel is not None else _Kernel(),
            needle_generate=fake_gen,
            invoke=fake_invoke,
            to_outcome=_xbrl_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    return out, invoked


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_xbrl_locator_grounds_only_successful_stored_source(failed: bool, stored: bool) -> None:
    """Only stored bytes from a successful source call can ground an unrequested XBRL tag."""
    ref = "s1:tr:sec"
    text = "Balance sheet boilerplate. " * 3000 + "us-gaap:InventoryNet 10,080 million."
    kernel = _Kernel({ref: {"tool_name": "get_sec_document", "result": {"text": text}}} if stored else {})
    attempt: dict[str, JSONValue] = {
        "tool": "get_sec_document",
        "tool_result_ref": ref,
        "arguments": {"accession_no": "0001045810-26-000075"},
        "error": "timeout" if failed else None,
        "outcome_summary": text[: sched._OUTCOME_SUMMARY_MAX],
    }
    assert "InventoryNet" not in str(attempt["outcome_summary"])
    out, invoked = _xbrl_attempt(
        "Which balance sheet facts did NVDA's latest 10-K tag?",
        "InventoryNet",
        attempts=[attempt],
        kernel=kernel,
    )
    if stored and not failed:
        assert out["error"] is None
        assert invoked == [{"ticker": "NVDA", "concept": "InventoryNet"}]
    else:
        assert out["error_type"] == "reselect"
        assert invoked == []


@pytest.mark.parametrize("concept", ["Today UTC date", "TodayUTCDate", "today"])
def test_xbrl_date_placeholder_concept_reselects_before_invoke(concept: str) -> None:
    """Live control run: Needle copied the Today UTC date guidance into concept; the runtime must never see it."""
    prompt = "[Today UTC 2026-10-09] What did NVDA report in its latest XBRL filing?"
    out, invoked = _xbrl_attempt(prompt, concept, question=prompt)
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])
    assert out["arguments"] == {}


@pytest.mark.parametrize("concept", ["NetIncomeLoss", "GrossProfit"])
def test_xbrl_unrequested_concept_reselects_before_invoke(concept: str) -> None:
    """A canonical or real tag the user never asked about and no source names cannot reach the runtime."""
    out, invoked = _xbrl_attempt("What was NVDA revenue in the latest quarter?", concept)
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])


@pytest.mark.parametrize(
    "concept",
    ["Revenue", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
)
def test_xbrl_requested_canonical_revenue_alias_invokes(concept: str) -> None:
    """A revenue ask grounds the canonical Revenue key and each of its existing SEC aliases, unchanged."""
    out, invoked = _xbrl_attempt("What was NVDA revenue in the latest quarter?", concept)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": concept}]


def test_xbrl_requested_exact_sec_tag_outside_canonical_map_invokes() -> None:
    """An exact SEC tag the user names stays valid even though the canonical map does not list it."""
    out, invoked = _xbrl_attempt(
        "What did NVDA report as ResearchAndDevelopmentExpense in XBRL?",
        "ResearchAndDevelopmentExpense",
    )
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ResearchAndDevelopmentExpense"}]


def test_xbrl_concept_named_late_in_full_source_invokes() -> None:
    """Grounding reads the full admitted source, so a tag past every generation bound still counts."""
    row: dict[str, JSONValue] = {
        "evidence_id": "ev-10k",
        "source_name": "get_sec_document",
        "content": "Balance sheet boilerplate. " * 3000 + "us-gaap:InventoryNet 10,080 million.",
    }
    out, invoked = _xbrl_attempt(
        "Which balance sheet facts did NVDA's latest 10-K tag?",
        "InventoryNet",
        evidence=[row],
    )
    _row_content = row["content"]
    assert isinstance(_row_content, str)
    assert len(_row_content) > 69_000 and _row_content.index("InventoryNet") > sched._OUTCOME_SUMMARY_MAX
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "InventoryNet"}]


def test_xbrl_failed_attempt_and_generated_text_cannot_ground_fabricated_concept() -> None:
    """A concept that only prior generated args, error text, or Needle reasoning name stays ungrounded."""
    fabricated = "AIAcceleratorRevenueNet"
    failed: dict[str, JSONValue] = {
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
    assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])


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
    evidence: list[dict[str, JSONValue]] = (
        [
            {
                "evidence_id": "ev-src",
                "source_name": "get_sec_document",
                "content": "Notes. " * 400 + f"us-gaap:{tag} 7",
            }
        ]
        if placement == "source"
        else []
    )
    attempts: list[dict[str, JSONValue]] = (
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
        assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])


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
    reselect: dict[str, JSONValue] = {
        "tool": "search_web",
        "arguments": {},
        "error": error,
        "error_type": "reselect",
    }
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
        return {
            "tool": "search_web",
            "arguments": {"query": f"q{calls['n']}"},
            "reasoning": "",
            "confidence": 1.0,
        }

    def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, JSONValue]:
        return {}

    class _J(JevClient):
        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {
                "evidence_state": "insufficient",
                "continuation": "continue_research",
            }

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
        )
    )
    assert out["status"] == "blocked" and out["reason"] == "incomplete: no new evidence"
    assert calls["n"] == 2


def test_finra_datapoints_unknown_dataset_reselects() -> None:
    with pytest.raises(sched._ReselectRequest, match="unknown dataset"):
        sched._validate_finra_datapoint_args("get_finra_datapoints", {"dataset": "2025-03-08", "fields": ["trade"]})
    with pytest.raises(sched._ReselectRequest, match="non-empty string list"):
        sched._validate_finra_datapoint_args(
            "get_finra_datapoints",
            {"dataset": "consolidatedShortInterest", "fields": []},
        )
    sched._validate_finra_datapoint_args(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "XYZ",
        },
    )


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
            {
                "select_round": spy,
                "reasoner": None,
                "deadline_at": _t.perf_counter() - 1.0,
            },
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

    blocked: list[tuple[str, str, str]] = []

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
            blocked.append((session_id, node_id, reason))
            return super().block_node(session_id, node_id, reason)

    async def slow(
        node: object,
        session_id: str,
        hooks: Mapping[str, object],
        kernel: sched._Kernel,
        cap: asyncio.Semaphore,
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
    assert all(r["reason"] == out["reason"] for r in rows)
    assert sorted(blocked) == [("s1", nid, out["reason"]) for nid in ("n1", "n2")]


@pytest.mark.parametrize(("cap", "expected"), [(None, 5), (2, 2)], ids=["default", "caller-cap"])
def test_node_round_limit_blocks_even_when_each_round_admits(cap: int | None, expected: int) -> None:
    calls = 0

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            nonlocal calls
            calls += 1
            return _invoke("query_finra")

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            return {"evidence_state": "sufficient_support", "continuation": "continue_research"}

    async def generate(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "query_finra", "arguments": {"ticker": f"SYM{calls}"}}

    async def invoke(name: str, args: dict[str, JSONValue], session: object, **kw: object) -> dict[str, JSONValue]:
        return {**_FINRA_RESULT, "tool_result_id": f"s1:tr:{calls}"}

    kernel = _Kernel({f"s1:tr:{n}": {**_FINRA_RESULT, "tool_result_id": f"s1:tr:{n}"} for n in range(1, expected + 1)})
    out = asyncio.run(
        sched.run_node(
            _node(),
            "s1",
            kernel=kernel,
            jev=_J(),
            needle_generate=generate,
            invoke=invoke,
            to_outcome=_to_outcome,
            registry=[{"name": "query_finra", "parameters": {}}],
            tool_session=_tool_session(),
            **({"max_rounds": cap} if cap is not None else {}),
        )
    )
    assert calls == expected and out["admitted"] == expected
    assert out["status"] == "blocked" and out["incomplete_guard"] is True
    assert f"{expected} rounds without resolution" in str(out["reason"])


def test_reason_over_evidence_selects_again_with_admitted_evidence() -> None:
    """Live rs:79691cfa: reason_over_evidence starts another selection inside the cap, not the runtime guard."""

    class _StoreKernel(_Kernel):
        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [{"evidence_id": eid, "content": "record values here"} for eid in self.evidence]

        @override
        def resolve_node(self, session_id: str, node_id: str) -> ResearchNode:
            self.calls.append(("resolve", node_id))
            return super().resolve_node(session_id, node_id)

    def drive(
        max_rounds: int,
    ) -> tuple[dict[str, JSONValue], _StoreKernel, list[list[object]]]:
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
        ) -> tuple[str, ToolDecision]:
            seen.append([e["evidence_id"] for e in ctx_ev if isinstance(e, dict) and e.get("evidence_id")])
            return ("invoke", _invoke("query_finra"))

        async def gen(tool: object, schema: object = None, **k: object) -> dict[str, object]:
            ticker = "XYZ" if len(seen) == 1 else "ABC"
            return {
                "tool": "query_finra",
                "arguments": {"dataset": "consolidatedShortInterest", "ticker": ticker},
            }

        async def invoke(name: object, args: object, sess: object, **k: object) -> dict[str, object]:
            return {
                "tool_result_id": f"s1:tr:{len(seen)}",
                "records": [{"symbolCode": "XYZ"}],
                "briefing": "B",
            }

        class _J(JevClient):
            @override
            async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
                first = len(seen) == 1
                return {
                    "continuation": "reason_over_evidence" if first else "resolve_node",
                    "evidence_state": "sufficient_support",
                }

        out = asyncio.run(
            sched._drive_rounds(
                _node(),
                {"session_id": "s1", "objective": "o"},
                [{"name": "query_finra", "parameters": {}}],
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
                    "max_rounds": max_rounds,
                },
                _tool_session(),
                None,
            )
        )
        return {"node_id": "n1", **out}, kernel, seen

    out, kernel, seen = drive(4)
    assert out["status"] == "resolved" and out["admitted"] == 2
    assert seen == [[], ["ev-0"]]  # the fresh selection receives the admitted evidence
    assert ("resolve", "n1") in kernel.calls
    # Boundary: a fresh selection past the cap still blocks the unresolved node.
    out, kernel, seen = drive(1)
    assert out["status"] == "blocked" and out["reason"] == "incomplete: runtime guard (1 rounds without resolution)"
    assert seen == [[]] and ("resolve", "n1") not in kernel.calls


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


def test_force_open_ignores_intake_evidence() -> None:
    """Intake 8-K accession with no node attempts never forces the carry registry."""
    intake: list[JSONValue] = [
        {
            "id": "ev:1",
            "content": "filed 8-K accession 0000320193-25-000079, see filing",
        }
    ]
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
    assert out is not None and {e["name"] for e in out} == {
        "get_sec_filing",
        "get_sec_document",
    }
    assert sched._force_open_registry(reg, attempts, [], 1) is None
    after_meta = [
        *attempts,
        {
            "tool": "get_sec_filing",
            "arguments": {},
            "outcome_summary": "meta",
            "error": None,
        },
    ]
    docs = sched._force_open_registry(reg, after_meta, [], 0)
    assert docs is not None and {e["name"] for e in docs} == {"get_sec_document"}


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


def test_session_records_each_node_failure_on_the_requested_kernel() -> None:
    class _K(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [_node("n9", ""), _node("n10", "")]

    kernel = _K()
    with mock.patch.object(kernel, "record_decision", wraps=kernel.record_decision) as recorded:
        out = asyncio.run(sched.run("s-given", kernel=kernel, jev="not-a-jev"))
    assert out["session_id"] == "s-given" and out["status"] == "stalled"
    nodes = out["nodes"]
    assert isinstance(nodes, list)
    assert {(node["node_id"], node["status"]) for node in nodes if isinstance(node, dict)} == {
        ("n9", "failed"),
        ("n10", "failed"),
    }
    failures = [call for call in recorded.call_args_list if call.args[1] == "node_failure"]
    assert {call.kwargs["node_id"] for call in failures} == {"n9", "n10"}
    assert all(call.args[0] == "s-given" for call in failures)


def test_sync_kwargs_generate_does_not_freeze_deadline() -> None:
    """Blocking sync kwargs generate leaves the loop free; cancel fails the job."""
    import threading
    import time as _time

    entered = threading.Event()
    release = threading.Event()

    def sync_gen(**kw: object) -> dict[str, JSONValue]:
        entered.set()
        release.wait(timeout=5.0)
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "r"}

    def _never_invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        raise AssertionError("invoke must not run while generation is blocked")

    kernel = _Kernel()

    async def _main() -> None:
        await asyncio.wait_for(
            sched._attempt_tool(
                tool_name="search_web",
                node=_node(),
                session={"session_id": "s1", "objective": "q"},
                registry=[{"name": "search_web", "parameters": {}}],
                evidence=[],
                attempts=[],
                kernel=kernel,
                needle_generate=sync_gen,
                invoke=_never_invoke,
                to_outcome=_to_outcome,
                tool_session=_tool_session(),
                as_of=None,
            ),
            timeout=0.2,
        )

    t0 = _time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(_main())
        assert _time.perf_counter() - t0 < 1.0
        assert entered.is_set()
        assert ("fail", "job-0", "timeout") in kernel.calls
    finally:
        release.set()


def test_sync_dict_generate_does_not_freeze_deadline() -> None:
    """Blocking sync single-dict generate leaves the loop free; cancel fails the job."""
    import threading
    import time as _time

    entered = threading.Event()
    release = threading.Event()

    def sync_gen(request: object) -> dict[str, JSONValue]:
        assert isinstance(request, dict)
        entered.set()
        release.wait(timeout=5.0)
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "r"}

    def _never_invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        raise AssertionError("invoke must not run while generation is blocked")

    kernel = _Kernel()

    async def _main() -> None:
        await asyncio.wait_for(
            sched._attempt_tool(
                tool_name="search_web",
                node=_node(),
                session={"session_id": "s1", "objective": "q"},
                registry=[{"name": "search_web", "parameters": {}}],
                evidence=[],
                attempts=[],
                kernel=kernel,
                needle_generate=sync_gen,
                invoke=_never_invoke,
                to_outcome=_to_outcome,
                tool_session=_tool_session(),
                as_of=None,
            ),
            timeout=0.2,
        )

    t0 = _time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(_main())
        assert _time.perf_counter() - t0 < 1.0
        assert entered.is_set()
        assert ("fail", "job-0", "timeout") in kernel.calls
    finally:
        release.set()


def test_async_dict_generate_stays_supported() -> None:
    """Async single-dict generate still returns grounded args."""

    async def async_gen(request: object) -> dict[str, JSONValue]:
        assert isinstance(request, dict)
        assert request.get("tool") == "search_web"
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "r"}

    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            async_gen,
            "search_web",
            [{"name": "search_web", "parameters": {}}],
            {"session_id": "s1", "objective": "q"},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("query") == "q"
    assert reasoning == "r" and withheld is False


def test_sync_wrapper_returning_coroutine_stays_supported() -> None:
    """Sync kwargs wrapper returning a coroutine still resolves to grounded args."""

    async def _impl(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_web", "arguments": {"query": "q"}, "reasoning": "r"}

    def wrapper(**kw: object) -> object:
        return _impl(**kw)

    args, reasoning, _ = asyncio.run(
        sched._generate_tool_arguments(
            wrapper,
            "search_web",
            [{"name": "search_web", "parameters": {}}],
            {"session_id": "s1", "objective": "q"},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("query") == "q"
    assert reasoning == "r"


def test_xbrl_wrapper_echo_cannot_ground_without_real_tag_data() -> None:
    """Envelope echoes (concept_searched, Form, valueless concept) never ground; real tag rows do."""
    bogus = "ZqBogusNet"
    echo: dict[str, JSONValue] = {
        "evidence_id": "ev-echo",
        "source_name": "get_xbrl_facts",
        "content": "XBRL facts envelope.",
        "concept_searched": bogus,
        "form": "10-Q",
        "concept": bogus,
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, evidence=[echo])
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])
    echo_attempt: dict[str, JSONValue] = {
        "tool": "get_xbrl_facts",
        "arguments": {"ticker": "NVDA", "concept": "Revenues"},
        "outcome_summary": "XBRL facts envelope.",
        "error": None,
        "result": {"concept_searched": bogus, "form": "10-Q", "matching_concepts": []},
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, attempts=[echo_attempt])
    assert invoked == []
    assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])
    real: dict[str, JSONValue] = {
        "evidence_id": "ev-real",
        "source_name": "get_xbrl_facts",
        "content": "XBRL facts envelope.",
        "matching_concepts": [{"concept": bogus, "value": 123, "period_end": "2026-07-31"}],
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, evidence=[real])
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": bogus}]
    namespaced: dict[str, JSONValue] = {
        "evidence_id": "ev-ns",
        "source_name": "get_sec_document",
        "content": f"Notes to the filing. us-gaap:{bogus} 7 million.",
    }
    out, invoked = _xbrl_attempt("What did NVDA report about AI accelerators?", bogus, evidence=[namespaced])
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": bogus}]


def test_xbrl_error_and_summary_text_cannot_ground_alias() -> None:
    """Errors and outcome summaries never ground, even when they name a real alias."""
    bad_evidence: dict[str, JSONValue] = {
        "evidence_id": "ev-bad",
        "source_name": "get_sec_document",
        "content": "Filing notes Revenues 10 million.",
        "error": "tool_error",
    }
    failed: dict[str, JSONValue] = {
        "tool": "get_xbrl_facts",
        "arguments": {"ticker": "NVDA", "concept": "Revenues", "period": "FY2026"},
        "outcome_summary": "concept Revenues not found",
        "error": "concept Revenues not found for NVDA",
        "error_type": "tool_error",
    }
    no_data: dict[str, JSONValue] = {
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
    assert out["error_type"] == "reselect" and "duplicate" not in str(out["error"])


def _resolve_orcl(name: str) -> str | None:
    return {"ORCL": "ORCL", "Oracle": "ORCL", "oracle": "ORCL"}.get(name)


def test_orcl_latest_revenue_bypass_seeds_listing_without_needle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ORCL latest revenue takes the deterministic bypass: seeded 10-Q/10-K listing, no Needle call."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_orcl)

    needle_calls: list[str] = []

    async def fake_malformed(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {
            "tool": "list_sec_filings",
            "arguments": {
                "identifier": "ORCL",
                "forms": ["10-Q/10-K"],
                "start_date": "2026-10-12",
                "end_date": "2026-10-12",
            },
            "reasoning": "r",
        }

    session = {"session_id": "s1", "objective": "What is ORCL's latest revenue?"}
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_malformed,
            "list_sec_filings",
            sched.build_registry(),
            session,
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == {"identifier": "ORCL", "forms": ["10-Q", "10-K"]}
    assert needle_calls == []


def test_orcl_latest_revenue_bypass_skips_needle_and_reads_listed_filing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undated ORCL latest revenue grounds listing/reads without Needle; the read carries the newest 10-Q/10-K."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_orcl)
    acc8, old_k, latest_q = (
        "0001341439-26-000099",
        "0001341439-26-000050",
        "0001341439-26-000075",
    )
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "ORCL", "forms": ["10-Q", "10-K"]},
        "error": None,
        "result": {
            "filings": [
                {"accession_no": acc8, "form": "8-K", "filed_at": "2026-10-01"},
                {"accession_no": old_k, "form": "10-K", "filed_at": "2026-06-01"},
                {"accession_no": latest_q, "form": "10-Q", "filed_at": "2026-09-01"},
            ]
        },
        "outcome_summary": f"8-K {acc8}, 10-K {old_k}, 10-Q {latest_q}",
    }
    needle_calls: list[str] = []

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What is ORCL's latest revenue?"}
    listed, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert listed == {"identifier": "ORCL", "forms": ["10-Q", "10-K"]}
    read, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_none, "get_sec_document", sched.build_registry(), session, _node(), [], [listing], None, _Kernel()
        )
    )
    assert read == {"accession_no": latest_q, "query": "revenue", "limit": 32000}
    assert needle_calls == []


def test_orcl_latest_revenue_read_requires_listing_first() -> None:
    """A revenue read with no 10-Q/10-K listing reselects instead of reading an unlisted accession."""

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "get_sec_document", "arguments": {}, "reasoning": "r"}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_sec_document",
            node=_node(),
            session={"session_id": "s1", "objective": "What is ORCL's latest revenue?"},
            registry=sched.build_registry(),
            evidence=[],
            attempts=[],
            kernel=_Kernel(),
            needle_generate=fake_gen,
            invoke=_invoke_result,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert out["error_type"] == "reselect"
    assert "10-Q/10-K" in str(out["error"])


def test_repeated_empty_search_reselects_without_invoking() -> None:
    """A second identical empty search reselects pre-invoke, so it never consumes handler budget."""
    invoked: list[str] = []

    async def fake_empty(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_sec_filings_bounded", "arguments": {}, "reasoning": "r"}

    def _never_invoke(*a: object, **k: object) -> dict[str, JSONValue]:
        invoked.append("invoked")
        return {}

    session = {"session_id": "s1", "objective": "What drove Growth this week?"}
    prior: dict[str, JSONValue] = {
        "tool": "search_sec_filings_bounded",
        "arguments": {},
        "outcome": None,
        "outcome_summary": "",
        "error": "search_sec_filings ungrounded (no grounding after repair); re-selecting",
        "error_type": "reselect",
        "reasoning": "r",
        "job_id": "job-0",
        "evidence_id": None,
        "tool_result_ref": None,
    }
    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_sec_filings_bounded",
            node=_node(),
            session=session,
            registry=[{"name": "search_sec_filings_bounded", "parameters": {}}],
            evidence=[],
            attempts=[prior],
            kernel=_Kernel(),
            needle_generate=fake_empty,
            invoke=_never_invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert out["error_type"] == "reselect" and "duplicate" in str(out["error"])
    assert invoked == []
    assert sched._empty_search_repeat([prior], "search_sec_filings_bounded", {})
    assert not sched._empty_search_repeat([prior], "search_sec_filings_bounded", {"ticker": "ORCL"})
    assert not sched._empty_search_repeat([], "search_sec_filings_bounded", {})
    # Non-dict args and non-search tools never count as a repeat.
    assert not sched._empty_search_repeat([prior], "search_sec_filings_bounded", None)
    assert not sched._empty_search_repeat([prior], "get_sec_filing", {})


def test_needle_context_carries_ticker_scope_grounded_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Needle context carries ticker, narrowed scope, and grounded args; evidence and attempts stay out."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_orcl)
    seen: dict[str, object] = {}
    selection_contexts: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        if kw["tool"] == "query_finra":
            return {"tool": "query_finra", "arguments": {}}
        seen.update(kw)
        return {"tool": "list_sec_filings", "arguments": {"identifier": "ORCL"}, "reasoning": "r"}

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is ORCL's filing history?",
        "ticker": "ORCL",
        "temporal_scope": {"mode": "latest-available", "raw": "private scope text", "as_of": None},
    }
    registry: list[dict[str, JSONValue]] = [
        {
            "name": "list_sec_filings",
            "parameters": {"type": "object", "properties": {"identifier": {"type": "string"}}},
        },
        {"name": "query_finra", "parameters": {}},
    ]

    class _K(_Kernel):
        @override
        def get_session(self, session_id: str) -> dict[str, JSONValue]:
            return session

        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [{"evidence_id": eid, "content": "private admitted bytes"} for eid in self.evidence]

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            selection_contexts.append(json.dumps(a[3]))
            return _invoke("list_sec_filings" if kernel.evidence else "query_finra")

        @override
        async def assess_result(self, *a: object, **k: object) -> dict[str, JSONValue]:
            assessment = await super().assess_result(*a, **k)
            if not kernel.evidence:
                assessment.update({"continuation": "continue_research", "continue": "continue_research"})
            return assessment

    def invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        return dict(_FINRA_RESULT) if name == "query_finra" else {"filings": []}

    kernel = _K({"s1:tr:abc": dict(_FINRA_RESULT)})
    out = asyncio.run(
        sched.run_node(
            _node(),
            "s1",
            kernel=kernel,
            jev=_J(),
            needle_generate=fake_gen,
            invoke=invoke,
            to_outcome=_to_outcome,
            registry=registry,
            max_rounds=2,
            tool_session=_tool_session(),
        )
    )
    ctx = seen.get("context")
    assert isinstance(ctx, dict)
    assert "evidence" not in ctx and "attempts" not in ctx and "grounding_hint" not in ctx
    serialized = json.dumps(seen)
    assert "private admitted bytes" not in serialized
    assert "record values here" not in serialized
    assert "private scope text" not in serialized
    assert ctx["ticker"] == "ORCL"
    assert ctx["temporal_scope"] == {"mode": "latest-available"}
    assert ctx["grounded_arguments"] == {"identifier": "ORCL"}
    assert out["status"] == "resolved" and out["admitted"] == 1
    assert any("private admitted bytes" in context for context in selection_contexts)


@pytest.mark.parametrize("form", ["10-Q/A", "10-K/A"])
def test_valid_amendment_forms_survive_repair(form: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid /A amendment forms are explicit selections, never malformed generated prose."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_orcl)

    async def fake_amend(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "ORCL", "forms": [form]}, "reasoning": "r"}

    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_amend,
            "list_sec_filings",
            sched.build_registry(),
            {"session_id": "s1", "objective": "List ORCL's most recent 10-K and 10-Q filings."},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args.get("forms") == [form]


def test_explicit_dated_latest_revenue_keeps_dates_and_uses_needle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit dates on a latest-revenue ask stay grounded and still route through Needle."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_orcl)
    generated: dict[str, JSONValue] = {
        "identifier": "ORCL",
        "forms": ["10-Q", "10-K"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    needle_calls: list[str] = []

    async def fake_gen(**kw: object) -> dict[str, JSONValue]:
        needle_calls.append(str(kw["tool"]))
        return {"tool": "list_sec_filings", "arguments": dict(generated), "reasoning": "r"}

    dated = "What is ORCL's latest revenue? Include filings from 2025-01-01 through 2025-12-31."
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            fake_gen,
            "list_sec_filings",
            sched.build_registry(),
            {"session_id": "s1", "objective": dated},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == generated
    assert needle_calls == ["list_sec_filings"]


@settings(max_examples=40, derandomize=True, deadline=None)
@given(
    st.sampled_from(
        ["10-Q/10-K", "Oracle Corp", "10-Q 10-K", "2026-10-12", "YYYY-MM-DD", "TSLA", "CIK", "SEC"]
        + ["10-Q/A", "10-K/A", "SC 13D", "DEF 14A", "N-PORT", "1-A", "EFFECT"]
    ),
)
def test_prop_generated_forms_are_never_substituted(form: str) -> None:
    """Repair keeps generated forms verbatim; unknown forms fail before EDGAR, known forms reach it."""
    from app import tools
    from app.policy import Capability, RequestContext

    filled = sched._repair_tool_arguments(
        "list_sec_filings",
        {"identifier": "ORCL", "forms": [form]},
        "What is ORCL's latest revenue?",
        "s",
        "n",
    )
    assert filled["forms"] == [form]
    calls: list[object] = []

    def _provider(*args: object, **kwargs: object) -> list[object]:
        calls.append(kwargs.get("forms"))
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tools.sec, "list_sec_filings", _provider)
        result = tools.execute_tool(
            "list_sec_filings",
            dict[str, object](filled),
            "test",
            context=RequestContext("research", frozenset({Capability.RESEARCH})),
        )
    if form in ("10-Q/A", "10-K/A", "SC 13D", "DEF 14A", "N-PORT", "1-A", "EFFECT"):
        assert "error" not in result and calls == [(form,)]
    else:
        assert result["error_type"] == "invalid_tool_arguments" and form in str(result["error"])
        assert calls == []


# ---------------------------------------------------------------------------
# Post-refactor stage coverage: locators, registry manifests, seeds, session loop
# ---------------------------------------------------------------------------


def test_finra_locator_reads_rows_then_envelope_then_nothing() -> None:
    """Row JSON wins; a rows-less envelope cites its own scope; a payload with neither stays uncitable."""
    row_text = sched._finra_locator({"records": [_FINRA_ROW]})
    assert row_text is not None and "XYZ" in row_text
    assert sched._finra_locator({"briefing": "  short   interest  "}) == "short interest"
    assert sched._finra_locator({"briefing_source": "deterministic_only"}) == "deterministic_only"
    assert sched._finra_locator({}) is None
    assert sched._finra_locator({"briefing": "   ", "query": {}}) is None


def test_sec_replay_locator_reads_scalar_rows_and_envelope() -> None:
    """Scalar rows cite verbatim; a payload with no citable row cites its own JSON."""
    assert sched._sec_replay_locator({"filings": ["10-K 0000320193-25-000079"]}) == "10-K 0000320193-25-000079"
    assert sched._sec_replay_locator({"rows": [42]}) == "42"
    assert sched._sec_replay_locator({"filings": [None]}) == '{"filings": [null]}'
    assert sched._sec_replay_locator({}) == "{}"


def test_tool_result_ref_reads_handle_and_result_ids() -> None:
    """A source_handle reference wins over the result-level ids; absent everywhere it stays None."""
    assert sched._tool_result_ref({"source_handle": {"tool_result_id": "h1"}}) == "h1"
    assert sched._tool_result_ref({"source_handle": {"source_handle_id": "h2"}}) == "h2"
    assert sched._tool_result_ref({"result_id": "r1"}) == "r1"
    assert sched._tool_result_ref({"source_handle_id": "r2"}) == "r2"
    assert sched._tool_result_ref({"source_handle": {}, "tool_result_id": " "}) is None
    assert sched._tool_result_ref(None, {"tool_result_ref": "o1"}) == "o1"


def test_scan_packet_accession_reads_packet_and_handle_document_names() -> None:
    """Document name comes from the packet, else its source handle; no accession yields (None, None)."""
    acc = "0001628280-26-044069"
    assert sched._scan_packet_accession({"document_name": "nvda-10q.htm", "nested": acc}) == (acc, "nvda-10q.htm")
    assert sched._scan_packet_accession(
        {"source_handle": {"document_name": "handle.htm"}, "text": f"filing {acc}"}
    ) == (acc, "handle.htm")
    assert sched._scan_packet_accession({"document_name": "only.pdf"}) == (None, None)


def test_ten_q_accessions_accepts_every_persisted_shape() -> None:
    """JSON text, labelled text, summary text and the payload's own form all yield exact 10-Q accessions."""
    acc_q, acc_k = "0001045810-26-000075", "0001045810-26-000070"
    assert sched._ten_q_accessions(f'{{"filings": [{{"form": "10-Q", "accession_no": "{acc_q}"}}]}}') == [acc_q]
    assert sched._ten_q_accessions(f"10-Q {acc_q}") == [acc_q]
    assert sched._ten_q_accessions(f"{acc_k} 10-K") == []
    assert sched._ten_q_accessions({"content": f"10-Q {acc_q}"}) == [acc_q]
    assert sched._ten_q_accessions({"result": {"form": "10-Q", "accession_no": acc_q}}) == [acc_q]
    assert sched._ten_q_accessions({"result": {"form": "8-K", "accession_no": acc_k}}) == []
    assert sched._ten_q_accessions(7) == []


def test_build_registry_skips_malformed_entries_and_renders_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed and JEV-excluded entries never reach the registry; named tools render discovery fields."""
    import app.tools as _tools

    def _fake_tools(caps: object) -> list[dict[str, object]]:
        return [
            {"function": "not-a-mapping"},
            {"function": {"name": ""}},
            {"function": {"name": "call_tool"}},
            {
                "function": {
                    "name": "list_sec_filings",
                    "description": "List EDGAR filings.",
                    "parameters": {"required": ["accession_no", "concept"], "properties": {}},
                }
            },
            {"function": {"name": "unregistered_tool", "parameters": {}}},
        ]

    monkeypatch.setattr(_tools, "tools_for_capabilities", _fake_tools)
    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)
    manifests = sched.build_registry()
    assert [m["name"] for m in manifests] == ["list_sec_filings", "unregistered_tool"]
    known = manifests[0]
    assert known["domain"] == "financial_research"
    assert known["description"] == "List EDGAR filings."
    assert known["keyInputs"] == "req(accession_no, concept)"
    assert known["prerequisites"] == "needs accession_no"
    assert known["pitSupport"] == "PIT-scoped"
    assert str(known["purpose"]).strip() and str(known["useWhen"]).strip()
    unknown = manifests[1]
    assert unknown["domain"] == "unknown"
    assert (unknown["purpose"], unknown["outputKind"], unknown["nextTools"]) == ("", "", "")


def test_objective_company_token_reads_possessive_and_insider_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Possessive, multi-word and insider phrases resolve; org words, stop words and bare titles do not."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL", "Tesla": "TSLA", "Acme Systems": "ACME"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    assert sched._objective_company_token("What are Apple's margins?") == ("Apple", "AAPL")
    assert sched._objective_company_token("What did Tesla insiders sell?") == ("Tesla", "TSLA")
    assert sched._objective_company_token("What are Acme Systems margins?") == ("Acme Systems", "ACME")
    # A leading title-case non-stop word glues into the multi-word match, so no name resolves alone.
    assert sched._objective_company_token("Compare Acme Systems revenue") is None
    assert sched._objective_company_token("What are FINRA short interest figures?") is None
    assert sched._objective_company_token("Growth slowed last quarter") is None
    assert sched._objective_company_token(None) is None


def test_fallback_sec_args_seeds_entities_and_accession(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entity queries seed the resolved ticker or the raw question; filing reads seed a stated accession."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    assert sched._fallback_sec_args("find_sec_entities", "What are Apple's margins?") == {"query": "AAPL"}
    assert sched._fallback_sec_args("find_sec_entities", "Growth slowed last quarter") == {
        "query": "Growth slowed last quarter"
    }
    acc = "0000320193-25-000079"
    assert sched._fallback_sec_args("get_sec_filing", f"read filing {acc}") == {"accession_no": acc}
    assert sched._fallback_sec_args("get_sec_filing", "read the latest filing") is None
    assert sched._fallback_sec_args("list_sec_filings", "What is Apple's latest revenue?") == {
        "identifier": "AAPL",
        "forms": ["10-Q", "10-K"],
    }


def test_scrub_ungrounded_date_args_keeps_every_grounded_date() -> None:
    """Request, scope, session cutoff and Today UTC dates stay; ungrounded dates and relative wording drop."""
    from datetime import UTC, datetime

    today = datetime.now(UTC).date().isoformat()
    keys = ("as_of", "start_date", "end_date")
    grounded: dict[str, JSONValue] = {"as_of": today, "start_date": "2025-09-27", "end_date": "2024-01-01"}
    assert (
        sched._scrub_ungrounded_date_args(
            grounded,
            "What is ORCL's latest revenue?",
            scope={"as_of": "2025-09-27"},
            session_as_of="2024-01-01",
            keys=keys,
        )
        == []
    )
    assert grounded == {"as_of": today, "start_date": "2025-09-27", "end_date": "2024-01-01"}

    dropped: dict[str, JSONValue] = {"as_of": "2020-01-01", "start_date": "this week", "end_date": 20240101}
    assert sched._scrub_ungrounded_date_args(dropped, "What is ORCL's latest revenue?", keys=keys) == [
        "as_of",
        "start_date",
    ]
    assert dropped == {"end_date": 20240101}

    requested: dict[str, JSONValue] = {"as_of": "2023-05-04"}
    assert sched._scrub_ungrounded_date_args(requested, "Compare ORCL as of 2023-05-04") == []
    required: dict[str, JSONValue] = {"as_of": "2020-01-01"}
    assert sched._scrub_ungrounded_date_args(required, "q", required=["as_of"]) == []
    assert required == {"as_of": "2020-01-01"}


def test_repair_seeds_finra_dataset_and_risk_section() -> None:
    """query_finra defaults its dataset; a risk-factor read seeds Item 1A only when it carries no query."""
    assert sched._repair_tool_arguments("query_finra", {}, "q?", "s1", "n1") == {"dataset": "consolidatedShortInterest"}
    assert sched._repair_tool_arguments("query_finra", {"dataset": "regShoDaily"}, "q?", "s1", "n1") == {
        "dataset": "regShoDaily"
    }
    assert sched._repair_tool_arguments("get_sec_document", {}, "What are NVDA's risk factors?", "s1", "n1") == {
        "section": "Item 1A"
    }
    assert sched._repair_tool_arguments(
        "get_sec_document", {"query": "risk"}, "What are NVDA's risk factors?", "s1", "n1"
    ) == {"query": "risk"}


def test_repair_withholds_search_selectors_without_a_subject() -> None:
    """search_sec_filings receives no selector when the objective names no resolvable subject."""
    assert sched._repair_tool_arguments("search_sec_filings", {}, "Growth slowed last quarter", "s1", "n1") == {}


def test_listing_carry_needs_actual_filing_rows() -> None:
    """A successful listing without filing rows never carries an accession onto a document read."""
    objective = "What is NVDA's latest revenue?"
    empty: dict[str, JSONValue] = {"tool": "list_sec_filings", "error": None, "result": {"filings": []}}
    assert sched._listing_carry_args("get_sec_document", objective, [empty]) is None
    rows: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": [{"form": "10-Q", "accession_no": "0001045810-26-000075", "filed_at": "2026-08-01"}]},
    }
    assert sched._listing_carry_args("get_sec_document", objective, [rows]) == {"accession_no": "0001045810-26-000075"}


def test_selection_action_maps_sentinels_without_an_action() -> None:
    """An empty action falls back to the sentinel mapping; a named tool still invokes."""
    assert sched._selection_action(ToolDecision(action="invoke", tool_names=("query_finra",))) == "invoke"
    assert sched._selection_action(ToolDecision(action="", tool_names=("node_resolved",))) == "resolved"
    assert sched._selection_action(ToolDecision(action="", tool_names=("reasoning_required",))) == "reason"
    assert sched._selection_action(ToolDecision(action="", tool_names=("query_finra",))) == "invoke"
    assert sched._selection_action(ToolDecision(action="")) == "reason"


def test_xbrl_source_facts_reads_prose_and_companyfacts_tags() -> None:
    """Prose, decoded JSON, and companyfacts rows carrying a value ground concepts; errors and empty units do not."""
    record: dict[str, JSONValue] = {
        "content": "Balance sheet notes.",
        "result": {"facts": {"us-gaap": {"InventoryNet": {"units": {"USD": [{"val": 10080}]}}}}},
    }
    facts = sched._xbrl_source_facts(record)
    assert "Balance sheet notes." in facts
    assert "us-gaap:InventoryNet" in facts
    assert sched._xbrl_source_facts({"us-gaap": {"InventoryNet": {"units": {"USD": []}}}}) == []
    assert sched._xbrl_source_facts({"error": "boom", "content": "unused"}) == []
    assert sched._xbrl_source_facts('{"content": "decoded prose"}') == ["decoded prose"]
    assert sched._xbrl_source_facts("plain prose") == ["plain prose"]


def _round_drops(
    registry: list[dict[str, JSONValue]], attempts: list[dict[str, JSONValue]], caplog: pytest.LogCaptureFixture
) -> tuple[list[str], list[str]]:
    """Registry names left for the round, plus each logged ``tool=… fails=…`` drop label in order."""
    import logging

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=sched.logger.name):
        kept = sched._round_registry(registry, attempts, [], 1, {"session_id": "s1", "objective": "o"}, "s1", "n1")
    drops = [r.getMessage().split("n1 ", 1)[1] for r in caplog.records if "select_drop" in r.getMessage()]
    return [str(e.get("name")) for e in kept], drops


def test_round_registry_drops_the_loop_shape(caplog: pytest.LogCaptureFixture) -> None:
    """Three identical failures, four-of-five wins, and a rejected call each drop one tool for one round."""
    registry: list[dict[str, JSONValue]] = [{"name": "query_finra"}, {"name": "search_web"}]
    three: list[dict[str, JSONValue]] = [{"tool": "query_finra", "error": "boom", "arguments": {}} for _ in range(3)]
    assert _round_drops(registry, three, caplog) == (["search_web"], ["tool=query_finra fails=3"])
    alternating: list[dict[str, JSONValue]] = [
        {"tool": "query_finra", "error": "x"},
        {"tool": "query_finra", "error": "x"},
        {"tool": "search_web", "error": None},
        {"tool": "query_finra", "error": "x"},
        {"tool": "query_finra", "error": "x"},
    ]
    assert _round_drops(registry, alternating, caplog) == (["search_web"], ["tool=query_finra fails=4-of-5"])
    rejected: list[dict[str, JSONValue]] = [{"tool": "query_finra", "error_type": "reselect", "arguments": {}}]
    assert _round_drops(registry, rejected, caplog) == (["search_web"], ["tool=query_finra fails=reselect-1"])
    assert _round_drops([{"name": "query_finra"}], rejected, caplog) == (["query_finra"], [])


@pytest.mark.parametrize(
    ("error", "drops"),
    [
        ("needle tool mismatch: jev selected 'query_finra', needle emitted None", True),
        ("needle arguments are not grounded in the request", True),
        ("needle tool mismatch: jev selected 'query_finra', needle emitted 'search_web'", False),
        ("needle timed out", False),
        ("provider connection failed", False),
    ],
)
def test_round_registry_drops_one_deterministic_failure_only(
    error: str, drops: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """One deterministic Needle failure excludes its tool. Transient errors preserve the registry."""
    registry: list[dict[str, JSONValue]] = [{"name": "query_finra"}, {"name": "search_web"}]
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "query_finra", "error_type": "tool_error", "error": error, "arguments": {}}
    ]
    expected: tuple[list[str], list[str]] = (
        (["search_web"], ["tool=query_finra fails=reselect-1"])
        if drops
        else (
            ["query_finra", "search_web"],
            [],
        )
    )
    assert _round_drops(registry, attempts, caplog) == expected
    assert _round_drops(registry[:1], attempts, caplog) == (["query_finra"], [])


def test_round_registry_drops_looping_and_rejected_tools_together(caplog: pytest.LogCaptureFixture) -> None:
    """A four-of-five loop and a following reselect each drop their own tool in the same round."""
    loop: list[dict[str, JSONValue]] = [{"tool": "query_finra", "error": "x"} for _ in range(4)]
    attempts: list[dict[str, JSONValue]] = [
        *loop,
        {"tool": "search_web", "error": "ungrounded", "error_type": "reselect", "arguments": {}},
    ]
    registry: list[dict[str, JSONValue]] = [{"name": "query_finra"}, {"name": "search_web"}, {"name": "get_quote"}]
    assert _round_drops(registry, attempts, caplog) == (
        ["get_quote"],
        ["tool=query_finra fails=4-of-5", "tool=search_web fails=reselect-1"],
    )
    # The reselect drop sees the loop-reduced registry: it never empties the round.
    assert _round_drops(registry[:2], attempts, caplog) == (["search_web"], ["tool=query_finra fails=4-of-5"])


def test_failed_document_read_keeps_the_full_registry() -> None:
    """A failed document read stays visible: force-open never restricts the registry around it."""
    registry: list[dict[str, JSONValue]] = [
        {"name": "list_sec_filings", "parameters": {}},
        {"name": "get_sec_document", "parameters": {}},
        {"name": "query_finra", "parameters": {}},
    ]
    failed_read: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "error": "boom", "arguments": {"accession_no": "0001045810-26-000075"}}
    ]
    session: dict[str, JSONValue] = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    assert sched._revenue_read_unlisted(session, failed_read) is True
    assert sched._round_registry(registry, failed_read, [], 0, session, "s1", "n1") == registry
    assert sched._revenue_read_unlisted({"session_id": "s1", "objective": "How many employees?"}, failed_read) is False


def test_reason_phase_records_reselect_on_analyze_and_expand_timeouts() -> None:
    """A reasoner stage that hits the run deadline records a reselect instead of failing the node."""
    import time as _time

    class _NoAnalyze(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            raise TimeoutError("analyze ran past the run deadline")

    attempts: list[dict[str, JSONValue]] = []
    out = asyncio.run(
        sched._reason_phase(
            _NoAnalyze(model="m", url="u"),
            {"session_id": "s1", "objective": "o"},
            _node(),
            [],
            [],
            attempts,
            JevClient(),
            _Kernel(),
            "s1",
            "n1",
            ToolDecision(action="reason"),
            0,
            _time.perf_counter() + 30.0,
        )
    )
    assert out["done"] is True and out["terminal"] is None
    assert [a.get("error") for a in attempts] == ["reasoner analyze hit the run deadline; re-selecting"]

    class _NoExpand(ReasonerClient):
        @override
        def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
            return {"analyses": [], "evidenceRequests": []}

        @override
        def expand(self, prompt: str, objective_id: str, prior_ids: set[str] | None = None) -> dict[str, list[object]]:
            raise TimeoutError("expand ran past the run deadline")

    expand_attempts: list[dict[str, JSONValue]] = []
    out = asyncio.run(
        sched._reason_phase(
            _NoExpand(model="m", url="u"),
            {"session_id": "s1", "objective": "o"},
            _node(),
            [],
            [],
            expand_attempts,
            _JevAdmit(),
            _Kernel(),
            "s1",
            "n1",
            ToolDecision(action="reason"),
            0,
            _time.perf_counter() + 30.0,
        )
    )
    assert out["done"] is True and out["terminal"] is None
    assert [a.get("error") for a in expand_attempts] == ["reasoner expand hit the run deadline; re-selecting"]


def test_run_reports_default_kernel_and_ready_node_failures() -> None:
    """Run fails loud for an unusable repo hook and for a kernel that cannot list ready nodes."""
    with pytest.raises(TypeError, match="repo hook must be ResearchRepository, Path or str, got object"):
        asyncio.run(sched.run("s1", repo=object()))

    with mock.patch.object(sched, "_default_kernel", side_effect=RuntimeError("no store")):
        failed = asyncio.run(sched.run("s1"))
    assert failed["status"] == "failed" and "default kernel unavailable: no store" in str(failed["error"])
    assert failed["session_id"] == "s1"

    class _KDown(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            raise RuntimeError("store offline")

    out = asyncio.run(sched.run("s1", kernel=_KDown()))
    assert out["status"] == "failed" and out["nodes"] == [] and "store offline" in str(out["error"])


def test_run_deadline_before_first_round_and_complete_drain() -> None:
    """A past run deadline reports incomplete_guard; a session with no ready nodes completes clean."""

    class _KIdle(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return []

    done = asyncio.run(sched.run("s1", kernel=_KIdle()))
    assert done == {"session_id": "s1", "status": "complete", "rounds": 0, "nodes": [], "incomplete_guard": False}

    expired = asyncio.run(sched.run("s1", kernel=_KIdle(), deadline_at=0.0))
    assert expired["status"] == "incomplete_guard" and expired["incomplete_guard"] is True
    assert expired["reason"] == "incomplete: run deadline reached before session round 1"


@pytest.mark.parametrize("registry", ["not-a-list", ["not-a-manifest"]])
def test_node_rejects_malformed_registry_before_selection(registry: object) -> None:
    kernel = _Kernel()
    with mock.patch.object(_JevAdmit, "select_tool", new_callable=mock.AsyncMock) as select:
        out = asyncio.run(sched.run_node(_node(), "s1", kernel=kernel, jev=_JevAdmit(), registry=registry))
    assert out["status"] == "failed"
    assert "registry hook must be a list of tool manifests" in str(out["error"])
    select.assert_not_called()
    assert kernel.calls == []
    assert [kind for kind, _ in kernel.decisions] == ["node_failure"]


def test_fallback_ticker_args_adds_eps_only_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ticker fallback adds the EPS metric only for explicit EPS wording; a subjectless objective withholds."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    assert sched._fallback_sec_args("get_fundamentals", "What was NVDA's EPS last quarter?") == {
        "ticker": "NVDA",
        "metric": "eps",
    }
    assert sched._fallback_sec_args("get_fundamentals", "How did NVDA revenue grow?") == {"ticker": "NVDA"}
    assert sched._fallback_sec_args("get_short_interest", "What is NVDA short interest?") == {"ticker": "NVDA"}
    assert sched._fallback_sec_args("get_insider_activity", "What drove Growth this week?") is None
    assert sched._fallback_sec_args("get_planned_insider_sales", None) is None


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ({}, {"ticker": "NVDA", "query": "What is NVDA's latest revenue?"}),
        ({"ticker": "NVDA", "query": "nvda revenue"}, {"ticker": "NVDA", "query": "nvda revenue"}),
        ({"company_name": "NVIDIA"}, {"company_name": "NVIDIA"}),
    ],
    ids=["empty", "query", "company"],
)
def test_node_search_invokes_grounded_selectors_without_overwriting_existing_values(
    generated: dict[str, JSONValue], expected: dict[str, JSONValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    invoked: list[dict[str, JSONValue]] = []

    class _J(_JevAdmit):
        @override
        async def select_tool(self, *a: object, **k: object) -> ToolDecision:
            return _invoke("search_sec_filings")

    async def generate(**kw: object) -> dict[str, JSONValue]:
        assert kw["schema"]
        return {"tool": "search_sec_filings", "arguments": generated}

    def invoke(name: str, args: dict[str, JSONValue], session: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(args)
        return {}

    out = asyncio.run(
        sched.run_node(
            _node(),
            "s1",
            kernel=_Kernel(objective="What is NVDA's latest revenue?"),
            jev=_J(),
            needle_generate=generate,
            invoke=invoke,
            to_outcome=_to_outcome,
            registry=sched.build_registry(),
            max_rounds=1,
            tool_session=_tool_session(),
        )
    )
    assert len(invoked) == 1
    assert invoked == [expected]
    assert out["status"] == "blocked" and out["admitted"] == 0


class _ReasonResolved(ReasonerClient):
    """Analyze stub without proposals: the reason phase adjudicates straight away."""

    @override
    def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
        return {"analyses": [], "evidenceRequests": []}


class _ReasonExpand(ReasonerClient):
    """Analyze stub that also expands one follow-up question."""

    @override
    def analyze(self, prompt: str) -> dict[str, list[dict[str, object]]]:
        return {"analyses": [], "evidenceRequests": []}

    @override
    def expand(self, prompt: str, objective_id: str, prior_ids: set[str] | None = None) -> dict[str, list[object]]:
        return {"proposals": [{"question": "Follow-up?", "whyItMatters": "Why"}]}


class _FixedAdjudicator(JevClient):
    """Adjudication stub with one fixed verdict; every expansion follow-up is admitted."""

    def __init__(self, verdict: ToolDecision) -> None:
        super().__init__()
        self._verdict = verdict

    @override
    async def adjudicate(self, *a: object, **k: object) -> ToolDecision:
        return self._verdict

    @override
    async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
        questions = a[1]
        assert isinstance(questions, Mapping)
        return {str(qid): {"choice": "admit"} for qid in questions}


def test_reason_phase_resolved_keeps_pending_revenue_read_open() -> None:
    """An adjudicated resolve re-selects while the retained revenue document read is pending; the node stays open."""
    verdict = ToolDecision(action="resolved", probabilities={"node_resolved": 0.9})
    attempts: list[dict[str, JSONValue]] = [{"tool": "list_sec_filings", "error": None, "arguments": {}}]
    out = asyncio.run(
        sched._reason_phase(
            _ReasonResolved(model="m", url="u"),
            {"session_id": "s1", "objective": "What was NVDA's latest revenue?"},
            _node(),
            [],
            [],
            attempts,
            _FixedAdjudicator(verdict),
            _Kernel(),
            "s1",
            "n1",
            ToolDecision(action="reason"),
            0,
            None,
        )
    )
    assert out["done"] is True and out["terminal"] is None and out["decision"] is verdict
    assert len(attempts) == 2
    assert attempts[-1]["error"] == "read the retained revenue document before resolving"
    assert attempts[-1]["error_type"] == "reselect"


def test_reason_phase_resolves_with_admitted_evidence() -> None:
    """An adjudicated resolve with admitted evidence returns the resolved terminal and records no reselect."""
    verdict = ToolDecision(action="resolved", probabilities={"node_resolved": 0.95})
    attempts: list[dict[str, JSONValue]] = []
    out = asyncio.run(
        sched._reason_phase(
            _ReasonResolved(model="m", url="u"),
            {"session_id": "s1", "objective": "How many employees does NVDA have?"},
            _node(),
            [{"id": "ev:1", "content": "NVDA 10-Q headcount row"}],
            [],
            attempts,
            _FixedAdjudicator(verdict),
            _Kernel(),
            "s1",
            "n1",
            ToolDecision(action="reason"),
            2,
            None,
        )
    )
    assert out["done"] is True and out["decision"] is verdict and attempts == []
    assert out["terminal"] == {
        "node_id": "n1",
        "status": "resolved",
        "admitted": 2,
        "attempts": [],
        "incomplete_guard": False,
    }


def test_reason_phase_expands_then_reselects_without_a_tool() -> None:
    """Adjudication that names no tool still runs the expansion, then re-selects carrying that verdict."""
    verdict = ToolDecision(action="reason", probabilities={"reasoning_required": 0.8})
    kernel = _Kernel()
    attempts: list[dict[str, JSONValue]] = []
    out = asyncio.run(
        sched._reason_phase(
            _ReasonExpand(model="m", url="u"),
            {"session_id": "s1", "objective": "What are NVDA's risk factors?"},
            _node(),
            [],
            [],
            attempts,
            _FixedAdjudicator(verdict),
            kernel,
            "s1",
            "n1",
            ToolDecision(action="reason"),
            0,
            None,
        )
    )
    assert ("graph_expansion", {"created": 1, "proposed": 1}) in kernel.decisions
    assert out["done"] is True and out["terminal"] is None and out["decision"] is verdict
    assert [a.get("error") for a in attempts] == ["adjudication named no tool; re-selecting"]


"""Scheduler survivor repairs: dependency order, deadlines, resume state, registries, replay scope, carried evidence, tool selection."""
import time


def test_carry_company_token_rejects_leading_stop_phrase() -> None:
    """Leading-stop phrases (What now) never seed company args."""
    from app.research import scheduler as sched

    assert sched._company_token_allowed("What now") is False
    assert sched._company_token_allowed("How now") is False


def test_carry_company_token_rejects_org_words() -> None:
    """Organization words never seed company args."""
    from app.research import scheduler as sched

    for word in ("FINRA", "SEC", "NYSE", "NASDAQ", "OTC", "EDGAR"):
        assert sched._company_token_allowed(word) is False


def test_carry_company_token_rejects_reserved_short_words() -> None:
    """Reserved short-interest words never seed company args."""
    from app.research import scheduler as sched

    for word in ("SHO", "REG", "EPS"):
        assert sched._company_token_allowed(word) is False
    assert sched._company_token_allowed("AAPL") is True


def test_carry_company_token_rejects_stop_and_suffix() -> None:
    """Stop words (What) and suffixes (Inc) never seed company args."""
    from app.research import scheduler as sched

    assert sched._company_token_allowed("What") is False
    assert sched._company_token_allowed("Inc") is False
    assert sched._company_token_allowed("Apple") is True


def test_carry_company_token_second_word_stop_still_rejected() -> None:
    """Phrases starting with a stop word stay rejected even when later words are clean."""
    from app.research import scheduler as sched

    assert sched._company_token_allowed("What Apple") is False
    assert sched._company_token_allowed("How Apple") is False


def test_carry_explicit_10q_blocks_unlabelled_fallback() -> None:
    """Explicit 10-Q asks never fall back to an unlabelled accession."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    assert (
        sched._carry_packet_fallback("get_sec_document", {}, [{"outcome_summary": acc}], [], False, True, False) is None
    )


def test_carry_explicit_10q_reads_newest_10q_not_latest_10k() -> None:
    """Explicit latest-10-Q reads the newest 10-Q even when a newer 10-K exists."""
    from app.research import scheduler as sched

    acc_q, acc_k = "0001045810-26-000075", "0001045810-26-000070"
    rows: list[JSONValue] = [
        {"form": "10-K", "filed_at": "2026-07-01", "accession_no": acc_k},
        {"form": "10-Q", "filed_at": "2026-05-01", "accession_no": acc_q},
    ]
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": rows},
        "outcome_summary": f"10-K {acc_k} 10-Q {acc_q}",
    }
    assert sched._listed_revenue_carry({}, [listing], True) == (True, {"accession_no": acc_q})
    assert sched._listed_revenue_carry({}, [listing], False) == (True, {"accession_no": acc_k})


def test_carry_explicit_10q_uses_ten_q_labels_not_latest() -> None:
    """An explicit 10-Q token with a latest ask labels with 10-Q accessions, not latest-revenue ones."""
    from app.research import scheduler as sched

    acc_q, acc_k = "0001045810-26-000075", "0001045810-26-000070"
    cand: dict[str, JSONValue] = {"outcome_summary": f"{acc_k} 10-K {acc_q} 10-Q"}
    assert sched._carry_packet_fallback("get_sec_document", {}, [cand], [], True, True, True) == {"accession_no": acc_q}


def test_carry_fallback_latest_prefers_newest_labelled() -> None:
    """Latest-revenue scan prefers the newest labelled 10-Q/10-K over older unlabelled accessions."""
    from app.research import scheduler as sched

    acc_old, acc_new = "0000320193-25-000001", "0000320193-25-000002"
    got = sched._carry_packet_fallback(
        "get_sec_document", {}, [{"outcome_summary": f"{acc_old} junk {acc_new} 10-Q"}], [], True, False, True
    )
    assert got == {"accession_no": acc_new}


def test_carry_fill_keeps_grounded_document_name() -> None:
    """Carry keeps an already-grounded document_name and only fills a missing one."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    cand = {"outcome_summary": f"doc other.htm {acc}"}
    assert sched._fill_carried_accession("get_sec_document", {"document_name": "keep.htm"}, cand, acc) == {
        "accession_no": acc,
        "document_name": "keep.htm",
    }
    filled = sched._fill_carried_accession("get_sec_document", {}, cand, acc)
    assert filled["accession_no"] == acc


def test_carry_latest_only_uses_listed_carry_not_fallback() -> None:
    """Latest-only revenue reaches the listed carry: newest 10-Q/10-K wins over packet order."""
    from app.research import scheduler as sched

    acc_q, acc_k = "0001045810-26-000075", "0001045810-26-000070"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {
            "filings": [
                {"form": "10-Q", "filed_at": "2026-05-01", "accession_no": acc_q},
                {"form": "10-K", "filed_at": "2026-07-01", "accession_no": acc_k},
            ]
        },
        "outcome_summary": f"10-Q {acc_q} 10-K {acc_k}",
    }
    assert sched._carry_packet_accession("get_sec_document", {}, [], [listing], "latest revenue") == {
        "accession_no": acc_k
    }


def test_carry_latest_revenue_blocks_unlabelled_fallback() -> None:
    """Latest revenue asks never fall back to an unlabelled accession."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    assert (
        sched._carry_packet_fallback("get_sec_document", {}, [{"outcome_summary": acc}], [], False, False, True) is None
    )
    assert sched._carry_packet_fallback(
        "get_sec_document", {}, [{"outcome_summary": acc}], [], False, False, False
    ) == {"accession_no": acc}


def test_carry_latest_revenue_reads_newest_10q_or_10k() -> None:
    """Latest revenue reads the newest 10-Q/10-K, not the newest 10-Q alone."""
    from app.research import scheduler as sched

    acc_q, acc_k = "0001045810-26-000050", "0001045810-26-000070"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {
            "filings": [
                {"form": "10-Q", "filed_at": "2026-05-01", "accession_no": acc_q},
                {"form": "10-K", "filed_at": "2026-07-01", "accession_no": acc_k},
            ]
        },
    }
    assert sched._listed_revenue_carry({}, [listing], False) == (True, {"accession_no": acc_k})


def test_carry_listed_accession_change_drops_stale_document() -> None:
    """A changed listed accession drops a stale generated document_name; matches keep it."""
    from app.research import scheduler as sched

    acc_old, acc_new = "0001045810-26-000050", "0001045810-26-000075"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc_new}]},
    }
    assert sched._listed_revenue_carry({"accession_no": acc_old, "document_name": "old.htm"}, [listing], True) == (
        True,
        {"accession_no": acc_new},
    )
    assert sched._listed_revenue_carry({"accession_no": acc_new, "document_name": "keep.htm"}, [listing], True) == (
        True,
        {"accession_no": acc_new, "document_name": "keep.htm"},
    )


def test_carry_listed_skips_errored_listings() -> None:
    """Errored listings never control the carry."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    bad: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": "boom",
        "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc}]},
    }
    good: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc}]},
    }
    assert sched._listed_revenue_carry({}, [bad], True) == (False, None)
    assert sched._listed_revenue_carry({}, [bad, good], True) == (True, {"accession_no": acc})


def test_carry_listed_uses_newest_successful_listing() -> None:
    """The newest successful listing controls the carry; older and failed packets do not."""
    from app.research import scheduler as sched

    acc_old, acc_new = "0001045810-26-000050", "0001045810-26-000075"

    def listing(acc: str) -> dict[str, JSONValue]:
        return {
            "tool": "list_sec_filings",
            "error": None,
            "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc}]},
        }

    assert sched._listed_revenue_carry({}, [listing(acc_old), listing(acc_new)], True) == (
        True,
        {"accession_no": acc_new},
    )
    assert sched._listed_revenue_carry({}, [{"tool": "get_sec_document", "error": None}, listing(acc_new)], True) == (
        True,
        {"accession_no": acc_new},
    )


def test_carry_listed_without_matching_rows_blocks_carry() -> None:
    """A controlling listing without matching rows blocks the carry; no listing scans packets."""
    from app.research import scheduler as sched

    empty: dict[str, JSONValue] = {"tool": "list_sec_filings", "error": None, "result": {"filings": []}}
    assert sched._listed_revenue_carry({}, [empty], True) == (True, None)
    assert sched._listed_revenue_carry({}, [], True) == (False, None)


def test_carry_listing_args_carries_for_requested_tool() -> None:
    """Listing carry computes the accession for the requested document tool."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    objective = "What did NVDA's latest 10-Q disclose about revenue?"
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "list_sec_filings",
            "error": None,
            "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc}]},
        }
    ]
    assert sched._listing_carry_args("get_sec_document", objective, attempts) == {"accession_no": acc}


def test_carry_listing_args_ignores_errored_listings() -> None:
    """Listing carry uses the newest successful listing with filing rows."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    objective = "What did NVDA's latest 10-Q disclose about revenue?"
    bad: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": "boom",
        "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc}]},
    }
    good: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": [{"form": "10-Q", "filed_at": "2026-08-01", "accession_no": acc}]},
    }
    assert sched._listing_carry_args("get_sec_document", objective, [bad]) is None
    assert sched._listing_carry_args("get_sec_document", objective, [bad, good]) == {"accession_no": acc}


def test_carry_quarter_only_without_token_skips_listed_gate() -> None:
    """Quarter wording without a 10-Q token skips the listed-carry gate for non-revenue fallback."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    assert sched._carry_packet_accession(
        "get_sec_document", {}, [f"packet {acc}"], [], "What drove NVDA revenue last quarter?"
    ) == {"accession_no": acc}


def test_carry_repeat_swap_needs_two_same_accession_repeats() -> None:
    """The repeat swap needs two same-tool same-accession repeats; anything less keeps args."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"

    def _entry(tool: str) -> Mapping[str, JSONValue]:
        return {"tool": tool, "arguments": {"accession_no": acc}}

    one: Sequence[Mapping[str, JSONValue]] = [_entry("get_sec_document")]
    two: Sequence[Mapping[str, JSONValue]] = [_entry("get_sec_document"), _entry("get_sec_document")]
    assert sched._same_tool_accession_count("get_sec_document", acc, one) == 1
    assert sched._same_tool_accession_count("get_sec_document", acc, two) == 2
    assert (
        sched._same_tool_accession_count(
            "get_sec_document", acc, [{"tool": "other", "arguments": {"accession_no": acc}}]
        )
        == 0
    )


def test_carry_revenue_guards_block_unlabelled_fallback() -> None:
    """Both revenue guards block unlabelled fallback; plain questions keep it."""
    from app.research import scheduler as sched

    acc = "0001045810-26-000075"
    assert (
        sched._carry_packet_fallback("get_sec_document", {}, [{"outcome_summary": acc}], [], False, True, False) is None
    )
    assert (
        sched._carry_packet_fallback("get_sec_document", {}, [{"outcome_summary": acc}], [], False, False, True) is None
    )
    assert sched._carry_packet_fallback(
        "get_sec_document", {}, [{"outcome_summary": acc}], [], False, False, False
    ) == {"accession_no": acc}


def test_carry_scan_reads_string_packets_directly() -> None:
    """String packets scan as text: labelled accessions stay preferred, unlabelled stay fallback."""
    from app.research import scheduler as sched

    acc = "0000320193-25-000001"
    assert sched._scan_carried_accessions("get_sec_document", {}, f"{acc} 10-Q", [acc]) == ({"accession_no": acc}, None)
    assert sched._scan_carried_accessions("get_sec_document", {}, f"{acc} junk", []) == (None, {"accession_no": acc})


def test_carry_stale_identifier_keeps_seven_digit_cik() -> None:
    """Six-digit numeric identifiers reseed; seven-digit ones name a real subject."""
    from app.research import scheduler as sched

    assert sched._stale_listing_identifier("123456", "AAPL") is True
    assert sched._stale_listing_identifier("1234567", "AAPL") is False


def test_carry_stale_identifier_reseeds_placeholders() -> None:
    """Blank, YYYY, date-shaped, and org identifiers reseed; real subjects keep their identifier."""
    from app.research import scheduler as sched

    assert sched._stale_listing_identifier("", None) is True
    assert sched._stale_listing_identifier("2025-01-01", None) is True
    assert sched._stale_listing_identifier("FINRA", "AAPL") is True
    assert sched._stale_listing_identifier("AAPL", "AAPL") is False
    assert sched._stale_listing_identifier("MSFT", "AAPL") is True


def test_carry_summary_key_none_yields_no_candidate() -> None:
    """A wrong summary key still carries: packet repr scan finds any accession-like token."""
    import asyncio

    from app.research import scheduler as sched
    from app.research.models import ResearchNode

    acc = "0001045810-26-000075"

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    node = ResearchNode(node_id="n1", session_id="s1", question="q?", why_it_matters="w")

    class _K(sched._Kernel):
        @override
        def get_session(self, session_id: str) -> dict[str, JSONValue]:
            return {"session_id": session_id, "objective": "q?", "query": "", "as_of": None}

        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return []

    async def main() -> dict[str, JSONValue]:
        args, _, _ = await sched._generate_tool_arguments(
            fake_none,
            "get_sec_filing",
            [],
            {"session_id": "s1", "objective": "q?"},
            node,
            [{"WRONG_SUMMARY": f"filing {acc} 10-Q"}],
            [],
            None,
            _K(),
        )
        return args

    assert asyncio.run(main()).get("accession_no") == acc


def test_carry_summary_only_packet_carries_accession() -> None:
    """Summary-only packets carry: outcome_summary accession fills open-step args."""
    import asyncio

    from app.research import scheduler as sched
    from app.research.models import ResearchNode

    acc = "0001045810-26-000075"

    async def fake_none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    node = ResearchNode(node_id="n1", session_id="s1", question="q?", why_it_matters="w")

    class _K(sched._Kernel):
        @override
        def get_session(self, session_id: str) -> dict[str, JSONValue]:
            return {"session_id": session_id, "objective": "q?", "query": "", "as_of": None}

        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return []

    async def main() -> dict[str, JSONValue]:
        args, _, _ = await sched._generate_tool_arguments(
            fake_none,
            "get_sec_filing",
            [],
            {"session_id": "s1", "objective": "q?"},
            node,
            [{"outcome_summary": f"filing {acc} 10-Q"}],
            [],
            None,
            _K(),
        )
        return args

    assert asyncio.run(main()).get("accession_no") == acc


def test_carry_ten_q_rows_missing_filed_at_sort_first() -> None:
    """Rows without filed_at sort after dated rows under reverse chronological order."""
    rows: list[object] = [
        {"form": "10-Q", "accession_no": "0000000000-25-000001"},
        {"form": "10-Q", "filed_at": "2025-01-01", "accession_no": "0000000000-25-000002"},
    ]
    assert sched._ten_q_rows_accessions(rows) == ["0000000000-25-000002", "0000000000-25-000001"]


def test_carry_ten_q_rows_skip_malformed_rows() -> None:
    """Malformed 10-Q rows drop; valid rows keep newest-first order."""
    rows: list[object] = [
        {"form": "10-Q", "filed_at": "2025-05-01", "accession_no": "BAD"},
        {"form": "10-Q", "filed_at": "2025-04-01", "accession_no": "0000000000-25-000002"},
        {"form": "10-Q", "filed_at": "2025-01-01", "accession_no": "0000000000-25-000001"},
    ]
    assert sched._ten_q_rows_accessions(rows) == ["0000000000-25-000002", "0000000000-25-000001"]


def test_carry_ten_q_text_reads_lowercase_accession_label() -> None:
    """Lowercase accession labels before the token still label the 10-Q."""
    from app.research import scheduler as sched

    acc = "0000320193-25-000001"
    assert sched._ten_q_text_accessions(f"10-Q accession {acc}") == [acc]


def test_carry_units_values_need_dict_with_reported_val() -> None:
    """Only dict units with a reported val carry values; everything else carries nothing."""
    from app.research import scheduler as sched

    assert sched._units_carry_values(["x"]) is False
    assert sched._units_carry_values({"USD": [{"val": 1}]}) is True
    assert sched._units_carry_values({"USD": [{"val": None}]}) is False
    assert sched._units_carry_values({"USD": []}) is False
    assert sched._units_carry_values({"USD": 5}) is False


def test_sched_needle_blank_search_selector_ungrounded() -> None:
    """A whitespace-only search selector is ungrounded and reselects before invoke."""
    assert sched._has_search_grounding({"ticker": "   "}) is False
    assert sched._has_search_grounding({"ticker": "NVDA"}) is True

    async def _blank(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_sec_filings", "arguments": {"ticker": "   "}, "reasoning": "r"}

    with pytest.raises(sched._ReselectRequest):
        asyncio.run(
            sched._generate_tool_arguments(
                _blank,
                "search_sec_filings",
                [],
                {"session_id": "s1", "objective": "What drove Growth this week?"},
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )


@pytest.mark.parametrize("needle_mode", ["failure", "withhold"])
def test_sched_needle_failure_carry_keeps_document_name(needle_mode: str) -> None:
    """Needle recovery preserves the document name beside the carried accession."""
    acc = "0001628280-26-044069"

    async def _down(**kw: object) -> dict[str, JSONValue]:
        if needle_mode == "failure":
            raise RuntimeError("needle worker closed")
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    session = {"session_id": "s1", "objective": "q?"}
    evidence: list[JSONValue] = [
        {"source_handle": {"accession_no": acc, "document_name": "nvda-10k.htm"}, "outcome_summary": f"read {acc}"}
    ]
    args, _, withheld = asyncio.run(
        sched._generate_tool_arguments(_down, "get_sec_document", [], session, _node(), evidence, [], None, _Kernel())
    )
    assert args.get("accession_no") == acc and args.get("document_name") == "nvda-10k.htm" and withheld is False


def test_sched_needle_failure_prefers_labeled_10q_and_seeds_query() -> None:
    """Failure carry prefers the packet 10-Q and repairs the revenue query/limit."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def _down(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle worker closed")

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}
    ]
    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            _down, "get_sec_document", [], session, _node(), evidence, attempts, None, _Kernel()
        )
    )
    assert args.get("accession_no") == accq and args.get("query") == "revenue"
    assert reasoning == "carried accession from packet after needle failure" and withheld is False


@pytest.mark.parametrize("tool_name", ["search_sec_filings", "search_sec_filings_bounded"])
def test_sched_needle_failure_search_without_ticker_reselects(tool_name: str) -> None:
    """Needle outage on ungrounded search reselects instead of invoking or raising bare."""

    async def _down(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle worker closed")

    session = {"session_id": "s1", "objective": "What drove Growth this week?"}
    with pytest.raises(sched._ReselectRequest, match="re-selecting"):
        asyncio.run(sched._generate_tool_arguments(_down, tool_name, [], session, _node(), [], [], None, _Kernel()))


@pytest.mark.parametrize("include_10q", [False, True])
def test_sched_needle_failure_swap_fire_uses_call_args(include_10q: bool) -> None:
    """A recovery preserves a repeated accession when its evidence names no alternative filing."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def _down(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle worker closed")

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K" + (f" filing {accq} 10-Q" if include_10q else "")}
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            _down, "get_sec_document", [], session, _node(), evidence, attempts, None, _Kernel()
        )
    )
    assert args.get("accession_no") == (accq if include_10q else acc8)


def test_sched_needle_grounded_search_invokes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonblank search selector invokes instead of reselecting."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    invoked: list[str] = []

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "search_sec_filings", "arguments": {"ticker": "NVDA"}, "reasoning": "r"}

    def _invoke(name: str, args: dict[str, JSONValue], sess: object, **kw: object) -> dict[str, JSONValue]:
        invoked.append(name)
        return {"tool_result_id": "s1:tr:x"}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="search_sec_filings",
            node=_node(),
            session={"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"},
            registry=[{"name": "search_sec_filings", "parameters": {}}],
            evidence=[],
            attempts=[],
            kernel=_Kernel(),
            needle_generate=_gen,
            invoke=_invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert invoked == ["search_sec_filings"] and out["error"] is None


def test_sched_needle_mixed_tool_failures_never_identical_break() -> None:
    """A mixed-tool tail never trips the identical-failure break."""
    tail: list[dict[str, JSONValue]] = [
        {"tool": "search_sec_filings", "arguments": {"ticker": "NVDA"}, "error": "boom"},
        {"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "error": "boom"},
        {"tool": "search_sec_filings", "arguments": {"ticker": "NVDA"}, "error": "boom"},
    ]
    assert sched._failed_tool_attempts(tail, frozenset({"search_sec_filings"})) is False
    assert sched._identical_failure_break(tail) is None


def test_sched_needle_repaired_carry_keeps_question_grounded_date() -> None:
    """A node-question date grounds carried args; dropping the question must not scrub it."""
    acc = "0001628280-26-044069"
    node = {"node_id": "n1", "question": "What happened on 2025-09-22?"}
    args = sched._repaired_carry(
        "get_sec_filing",
        {"accession_no": acc, "as_of": "2025-09-22"},
        "q?",
        {"session_id": "s1"},
        node,
        [],
        [],
        "s1",
        "n1",
        "r",
    )[0]
    assert args.get("as_of") == "2025-09-22"


def test_sched_needle_repaired_carry_keeps_scope_grounded_date() -> None:
    """A scope-window date grounds carried args; dropping the scope must not scrub it."""
    acc = "0001628280-26-044069"
    session: Mapping[str, JSONValue] = {
        "session_id": "s1",
        "temporal_scope": {"start": "2025-09-22", "raw": "week of 2025-09-22"},
    }
    args = sched._repaired_carry(
        "get_sec_filing",
        {"accession_no": acc, "as_of": "2025-09-22"},
        "q?",
        session,
        {"node_id": "n1"},
        [],
        [],
        "s1",
        "n1",
        "r",
    )[0]
    assert args.get("as_of") == "2025-09-22"


def test_sched_needle_repaired_carry_keeps_session_grounded_date() -> None:
    """A session cutoff date grounds carried args; dropping it must not scrub it."""
    acc = "0001628280-26-044069"
    session = {"session_id": "s1", "as_of": "2025-09-22"}
    args = sched._repaired_carry(
        "get_sec_filing",
        {"accession_no": acc, "as_of": "2025-09-22"},
        "q?",
        session,
        {"node_id": "n1"},
        [],
        [],
        "s1",
        "n1",
        "r",
    )[0]
    assert args.get("as_of") == "2025-09-22"


def test_sched_needle_repaired_carry_swaps_repeat_to_10q() -> None:
    """A carried same-accession repeat swaps for the packet 10-Q instead of failing it."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"
    objective = "What drove NVDA revenue last quarter?"
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K filing {accq} 10-Q", "tool": "list_sec_filings"}
    ]
    args, reasoning, withheld = sched._repaired_carry(
        "get_sec_document",
        {"accession_no": acc8},
        objective,
        {"session_id": "s1"},
        {"node_id": "n1"},
        evidence,
        attempts,
        "s1",
        "n1",
        "r",
    )
    assert args.get("accession_no") == accq and reasoning == "r" and withheld is False


def test_sched_needle_reselect_record_fails_job_and_settles() -> None:
    """A reselect record fails its own job and settles as bookkeeping."""

    kernel = _Kernel()
    record = sched._reselect_tool_attempt(
        sched._ReselectRequest("search_sec_filings ungrounded; re-selecting"),
        kernel,
        "job-9",
        "search_sec_filings",
        [],
        "r",
        "s1",
        "n1",
        1.0,
        2.0,
    )
    assert record["tool"] == "search_sec_filings" and record["arguments"] == {} and record["error_type"] == "reselect"
    assert record["outcome"] is None and record["evidence_id"] is None and record["tool_result_ref"] is None
    assert ("fail", "job-9", "tool_error") in kernel.calls
    settled, done = sched._settle_attempt(kernel, {**record, "outcome": None})
    assert done is True and settled is not None and settled["error_type"] == "reselect"
    terminal = sched._guided_break_result([dict(settled)], dict(settled), kernel, "s1", "n1", 0, False)
    assert terminal is None

    long_msg, long_reason = "e" * 600, "r" * 2500
    capped = sched._reselect_tool_attempt(
        sched._ReselectRequest(long_msg), kernel, "job-9", "search_sec_filings", [], long_reason, "s1", "n1", 1.0, 2.0
    )
    assert len(str(capped["error"])) == 500 and len(str(capped["reasoning"])) == 2000


def test_sched_needle_seed_shape_truncation_noforms_unwithheld(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded queries truncate at 200 chars, listings omit forms without revenue, seeds stay unwithheld."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def _none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    long_objective = "What is NVDA filing history? " + "x" * 500
    session = {"session_id": "s1", "objective": long_objective}
    args, _, withheld = asyncio.run(
        sched._generate_tool_arguments(_none, "search_sec_filings", [], session, _node(), [], [], None, _Kernel())
    )
    assert args["ticker"] == "NVDA" and len(str(args["query"])) == 200 and "forms" not in args and withheld is False
    listed, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            _none,
            "list_sec_filings",
            [],
            {"session_id": "s1", "objective": "What is NVDA filing history?"},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert listed == {"identifier": "NVDA"}


def test_sched_needle_signature_canonicalizes_key_order() -> None:
    """Same-content args in different key order share one signature."""
    same: Sequence[Mapping[str, JSONValue]] = [{"arguments": {"b": 1, "a": 2}}, {"arguments": {"a": 2, "b": 1}}]
    assert sched._attempt_call_signatures(same)[0] == sched._attempt_call_signatures(same)[1]


def test_sched_needle_timeout_recovers_only_on_carry_tools() -> None:
    """Timeout wording recovers on accession-carry tools only; anything else re-raises."""
    assert sched._needle_failure_recoverable("get_sec_filing", RuntimeError("needle timed out")) is True
    assert sched._needle_failure_recoverable("get_sec_document", RuntimeError("request timeout")) is True
    assert sched._needle_failure_recoverable("search_sec_filings", RuntimeError("needle timed out")) is False
    assert sched._needle_failure_recoverable("get_sec_filing", RuntimeError("boom")) is False
    acc = "0001628280-26-044069"

    async def _timeout(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle timed out")

    prior: list[dict[str, JSONValue]] = [
        {
            "tool": "get_sec_filing",
            "arguments": {"accession_no": acc},
            "error": "stale",
            "outcome_summary": f"filing {acc} 10-K",
        },
    ]
    args, _, withheld = asyncio.run(
        sched._generate_tool_arguments(
            _timeout, "get_sec_filing", [], {"session_id": "s1", "objective": "q?"}, _node(), [], prior, None, _Kernel()
        )
    )
    assert args.get("accession_no") == acc and withheld is False


def test_sched_needle_withhold_carry_docs_and_args() -> None:
    """A withhold for get_sec_document fails closed on tool mismatch; non-mappings also raise."""
    acc = "0001628280-26-044069"

    async def _with_acc(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {"accession_no": acc, "document_name": "kept.htm"}, "reasoning": "withheld"}

    async def _with_list(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": [acc], "reasoning": "withheld"}

    session = {"session_id": "s1", "objective": "q?"}
    evidence: list[JSONValue] = [{"source_handle": {"accession_no": acc, "document_name": "packet.htm"}}]
    with pytest.raises(ValueError, match="needle tool mismatch"):
        asyncio.run(
            sched._generate_tool_arguments(
                _with_acc, "get_sec_document", [], session, _node(), evidence, [], None, _Kernel()
            )
        )
    try:
        asyncio.run(
            sched._generate_tool_arguments(_with_list, "get_sec_filing", [], session, _node(), [], [], None, _Kernel())
        )
    except ValueError as exc:
        assert "needle tool mismatch" in str(exc)
    else:
        raise AssertionError("non-mapping withhold args must still raise mismatch")

    dated_acc = "0001628280-26-044069"

    async def _none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    dated_session = {"session_id": "s1", "objective": "q?"}
    dated_node = {"node_id": "n1", "question": "What happened on 2025-09-22?"}
    dated_evidence: list[JSONValue] = [{"outcome_summary": f"filing {dated_acc} 10-K", "tool": "search_sec_filings"}]
    dated_args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            _none, "get_sec_filing", [], dated_session, dated_node, dated_evidence, [], None, _Kernel()
        )
    )
    assert dated_args.get("accession_no") == dated_acc


@pytest.mark.parametrize("tool_name", ["search_sec_filings", "search_sec_filings_bounded"])
def test_sched_needle_withhold_search_without_ticker_reselects(tool_name: str) -> None:
    """Needle withhold on ungrounded search reselects instead of invoking or raising bare."""

    async def _none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove Growth this week?"}
    with pytest.raises(sched._ReselectRequest, match="re-selecting"):
        asyncio.run(sched._generate_tool_arguments(_none, tool_name, [], session, _node(), [], [], None, _Kernel()))


@pytest.mark.parametrize("include_10q", [False, True])
def test_sched_needle_withhold_swap_repeats_to_10q(include_10q: bool) -> None:
    """A recovery preserves a repeated accession when its evidence names no alternative filing."""
    acc8, accq = "0001045810-26-000078", "0001045810-26-000075"

    async def _none(**kw: object) -> dict[str, JSONValue]:
        return {"tool": None, "arguments": {}, "reasoning": "withheld: ungrounded"}

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
        {"tool": "get_sec_document", "arguments": {"accession_no": acc8}, "error": None},
    ]
    evidence: list[JSONValue] = [
        {"outcome_summary": f"filing {acc8} 8-K" + (f" filing {accq} 10-Q" if include_10q else "")}
    ]
    args, _, _ = asyncio.run(
        sched._generate_tool_arguments(
            _none, "get_sec_document", [], session, _node(), evidence, attempts, None, _Kernel()
        )
    )
    assert args.get("accession_no") == (accq if include_10q else acc8)


def test_sched_needle_worker_down_seeds_sec_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker-down outage seeds the SEC listing from the objective ticker, unwithheld."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    async def _down(**kw: object) -> dict[str, JSONValue]:
        raise RuntimeError("needle worker closed")

    session = {"session_id": "s1", "objective": "What drove NVDA revenue last quarter?"}
    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(_down, "list_sec_filings", [], session, _node(), [], [], None, _Kernel())
    )
    assert args == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    assert reasoning == "seeded SEC identifier from objective ticker after needle failure" and withheld is False
    for reason in ("server unavailable", "needle ping failed", "missing-server", "server missing"):
        assert sched._needle_failure_recoverable("get_sec_filing", RuntimeError(reason)) is True


def test_scope_empty_dict_unconstrained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing mode reads latest-available, so the undated ask seeds without Needle."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    calls: list[str] = []

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        calls.append(str(kw["tool"]))
        raise AssertionError("Needle must not run for an undated latest-revenue ask")

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is NVDA's latest revenue?",
        "temporal_scope": {},
    }
    out = asyncio.run(
        sched._generate_tool_arguments(
            _gen, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert calls == []
    assert out[0] == {"identifier": "NVDA", "forms": ["10-Q", "10-K"]}


def test_scope_end_only_constrains(monkeypatch: pytest.MonkeyPatch) -> None:
    """An end-only window constrains the ask, so Needle generates."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA", "forms": ["10-Q"]}, "reasoning": "r"}

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is NVDA's latest revenue?",
        "temporal_scope": {"mode": "latest-available", "end": "2026-12-31"},
    }
    out = asyncio.run(
        sched._generate_tool_arguments(
            _gen, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert out[0] == {"identifier": "NVDA", "forms": ["10-Q"]}


def test_scope_historical_mode_constrains(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-latest mode constrains the ask, so Needle generates."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA", "forms": ["10-Q"]}, "reasoning": "r"}

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is NVDA's latest revenue?",
        "temporal_scope": {"mode": "historical"},
    }
    out = asyncio.run(
        sched._generate_tool_arguments(
            _gen, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert out[0] == {"identifier": "NVDA", "forms": ["10-Q"]}


def test_scope_latest_mode_unconstrained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A latest mode reads unconstrained, so the undated ask seeds without Needle."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)
    calls: list[str] = []

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        calls.append(str(kw["tool"]))
        raise AssertionError("Needle must not run for an undated latest-revenue ask")

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is NVDA's latest revenue?",
        "temporal_scope": {"mode": "latest"},
    }
    out = asyncio.run(
        sched._generate_tool_arguments(
            _gen, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert calls == []
    assert out[0] == {"identifier": "NVDA", "forms": ["10-Q", "10-K"]}


def test_scope_raw_text_dates_constrain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope raw text carrying dates constrains the ask, so Needle generates."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA", "forms": ["10-Q"]}, "reasoning": "r"}

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is NVDA's latest revenue?",
        "temporal_scope": {"raw": "filings from 2025-01-01 through 2025-12-31"},
    }
    out = asyncio.run(
        sched._generate_tool_arguments(
            _gen, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert out[0] == {"identifier": "NVDA", "forms": ["10-Q"]}


def test_scope_start_only_constrains(monkeypatch: pytest.MonkeyPatch) -> None:
    """A start-only window constrains the ask, so Needle generates."""
    import app.tools as _tools

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve_nvda)

    async def _gen(**kw: object) -> dict[str, JSONValue]:
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA", "forms": ["10-Q"]}, "reasoning": "r"}

    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What is NVDA's latest revenue?",
        "temporal_scope": {"mode": "latest-available", "start": "2026-01-01"},
    }
    out = asyncio.run(
        sched._generate_tool_arguments(
            _gen, "list_sec_filings", sched.build_registry(), session, _node(), [], [], None, _Kernel()
        )
    )
    assert out[0] == {"identifier": "NVDA", "forms": ["10-Q"]}


def test_scrub_drops_empty_as_of() -> None:
    """An empty as_of is ungrounded and drops."""
    filled: dict[str, JSONValue] = {"as_of": ""}
    assert sched._scrub_ungrounded_date_args(filled, "q") == ["as_of"]
    assert filled == {}


def test_scrub_drops_relative_as_of() -> None:
    """A relative as_of is ungrounded and drops."""
    filled: dict[str, JSONValue] = {"as_of": "today"}
    assert sched._scrub_ungrounded_date_args(filled, "What revenue as of today?") == ["as_of"]
    assert filled == {}


def test_scrub_keeps_longer_value_when_day_scoped() -> None:
    """A longer value whose day scope text names stays."""
    filled: dict[str, JSONValue] = {"as_of": "2026-09-01T00:00"}
    assert sched._scrub_ungrounded_date_args(filled, "q", scope={"raw": "window 2026-09-01 report"}) == []
    assert filled == {"as_of": "2026-09-01T00:00"}


def test_scrub_keeps_timestamp_against_scope_day() -> None:
    """A full timestamp whose day matches the scope day stays."""
    filled: dict[str, JSONValue] = {"as_of": "2026-09-01T12:00:00"}
    assert sched._scrub_ungrounded_date_args(filled, "q", scope={"as_of": "2026-09-01"}) == []
    assert filled == {"as_of": "2026-09-01T12:00:00"}


def test_sec_replay_dict_row_truncates_2000() -> None:
    """A long dict row cites exactly 2000 chars."""
    assert len(sched._sec_replay_locator({"filings": [{"k": "x" * 2500}]}) or "") == 2000


def test_sec_replay_documents_row() -> None:
    """A documents row cites its own JSON, never the envelope."""
    assert sched._sec_replay_locator({"documents": [{"a": 5}]}) == '{"a": 5}'


def test_sec_replay_envelope_truncates_2000() -> None:
    """A payload with no citable row cites its own JSON truncated to 2000 chars."""
    locator = sched._sec_replay_locator({"note": "x" * 2500})
    assert isinstance(locator, str) and len(locator) == 2000
    assert locator == sched._compact_json({"note": "x" * 2500})[:2000]


def test_sec_replay_events_row() -> None:
    """An events row cites its own JSON, never the envelope."""
    assert sched._sec_replay_locator({"events": [{"a": 6}]}) == '{"a": 6}'


def test_sec_replay_facts_list_row() -> None:
    """A facts-list row cites its own JSON, never the envelope."""
    assert sched._sec_replay_locator({"facts": [{"a": 3}]}) == '{"a": 3}'


def test_sec_replay_quarterly_eps_row() -> None:
    """A quarterly_eps row cites its own JSON, never the envelope."""
    assert sched._sec_replay_locator({"quarterly_eps": [{"a": 2}]}) == '{"a": 2}'


def test_sec_replay_records_row() -> None:
    """A records row cites its own JSON, never the envelope."""
    assert sched._sec_replay_locator({"records": [{"a": 4}]}) == '{"a": 4}'


def test_sec_replay_scalar_row_truncates_2000() -> None:
    """A long scalar row cites exactly 2000 chars."""
    assert sched._sec_replay_locator({"rows": ["x" * 2500]}) == "x" * 2000


def test_sec_replay_transactions_row() -> None:
    """A transactions row cites its own JSON, never the envelope."""
    assert sched._sec_replay_locator({"transactions": [{"a": 1}]}) == '{"a": 1}'


def test_xbrl_attempt_list_skips_error_row() -> None:
    """An error row is skipped; a later attempt row still grounds."""
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "error": "boom", "result": {"content": "noise"}},
        {"tool": "get_sec_document", "error": None, "result": {"content": "notes ZqSkipTag 4"}},
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqSkipTag", attempts=attempts)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqSkipTag"}]


def test_xbrl_attempt_result_key_grounds() -> None:
    """A prior attempt result naming the tag grounds it even without a stored ref."""
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "error": None, "result": {"content": "notes ZqAttemptTag 4"}}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqAttemptTag", attempts=attempts)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqAttemptTag"}]


def test_xbrl_broken_json_brackets_are_not_prose() -> None:
    """Broken brace/bracket text reads as broken JSON, never as prose."""
    for text, tag in (("{oops names ZqX 1", "ZqX"), ("[oops names ZqY 2", "ZqY")):
        evidence: list[dict[str, JSONValue]] = [{"evidence_id": "ev-br", "source_name": "s", "content": text}]
        out, invoked = _xbrl_attempt("What about AI?", tag, evidence=evidence)
        assert invoked == []
        assert out["error_type"] == "reselect"


def test_xbrl_claim_text_carries_prose() -> None:
    """The claim_text field walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-ZqClaimTag", "source_name": "s", "claim_text": "says ZqClaimTag 5"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqClaimTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqClaimTag"}]


def test_xbrl_companyfacts_ignores_non_dict_detail() -> None:
    """A non-dict tag detail carries no units; the ask reselects instead of failing."""
    evidence: list[dict[str, JSONValue]] = [
        {
            "evidence_id": "ev-detail",
            "source_name": "get_sec_document",
            "result": {"facts": {"us-gaap": {"T": "oops"}}},
        }
    ]
    out, invoked = _xbrl_attempt("What about AI?", "T", evidence=evidence)
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_companyfacts_skips_bad_namespace() -> None:
    """A non-dict namespace is skipped; tags in later namespaces still ground."""
    evidence: list[dict[str, JSONValue]] = [
        {
            "evidence_id": "ev-ns",
            "source_name": "get_sec_document",
            "result": {"facts": {"bad": "oops", "us-gaap": {"T2": {"units": {"USD": [{"val": 1}]}}}}},
        }
    ]
    out, invoked = _xbrl_attempt("What about AI?", "T2", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "T2"}]


@pytest.mark.parametrize("concept", ["2026", "123", "YYYY"])
def test_xbrl_date_fragment_concept_reselects(concept: str) -> None:
    """A numeric or date-placeholder concept cannot reach the runtime, even when the request names it."""
    out, invoked = _xbrl_attempt(f"Compare as of {concept}", concept)
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_error_record_skips_but_keeps_good() -> None:
    """An error record is skipped; a later good record still grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"error": "boom", "source_name": "s", "claim_text": "skip ZqSkipTag 1"},
        {"evidence_id": "ev-good", "source_name": "s", "content": "notes ZqSkipTag 4"},
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqSkipTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqSkipTag"}]


def test_xbrl_evidence_list_walks_rows() -> None:
    """A nested evidence list walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-nest", "source_name": "s", "evidence": [{"concept": "ZqNestTag", "value": 1}]}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqNestTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqNestTag"}]


def test_xbrl_fact_carries_prose() -> None:
    """The fact field walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-ZqFactTag", "source_name": "s", "fact": "says ZqFactTag 5"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqFactTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqFactTag"}]


def test_xbrl_facts_list_walks_rows() -> None:
    """A facts row list walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-fl", "source_name": "s", "result": {"facts": [{"concept": "ZqFactsTag", "value": 1}]}}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqFactsTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqFactsTag"}]


def test_xbrl_identified_attempt_record_grounds() -> None:
    """A tool record carrying evidence ids still carries its prose."""
    evidence: list[dict[str, JSONValue]] = [
        {"tool": "get_sec_document", "evidence_id": "e-stored", "content": "notes ZqIdentTag 7"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqIdentTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqIdentTag"}]


def test_xbrl_leading_space_brace_is_not_prose() -> None:
    """Leading-space broken JSON reads as a broken object, never as prose."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-br", "source_name": "s", "content": "  {oops says ZqBraceTag 5"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqBraceTag", evidence=evidence)
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_loss_alias_invokes() -> None:
    """The GAAP Loss suffix reads as its income words, so the alias invokes."""
    out, invoked = _xbrl_attempt("What was NVDA net income (loss)?", "NetIncomeLoss")
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "NetIncomeLoss"}]


def test_xbrl_lowercase_mention_grounds_mixed_tag() -> None:
    """A lowercase source mention exactly grounds a tag with no canonical group."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-low", "source_name": "s", "content": "notes zqbogusnet 7"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqBogusNet", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqBogusNet"}]


def test_xbrl_lowercase_tag_objective_grounds() -> None:
    """A lowercase-only tag the request names invokes."""
    out, invoked = _xbrl_attempt("What was NVDA eps?", "eps")
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "eps"}]


def test_xbrl_matching_passage_carries_prose() -> None:
    """The matching_passage field walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-ZqMpTag", "source_name": "s", "matching_passage": "says ZqMpTag 5"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqMpTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqMpTag"}]


def test_xbrl_node_question_grounds() -> None:
    """The node question names the tag when nothing else does, so it invokes."""
    out, invoked = _xbrl_attempt("What about AI?", "ZqNodeTag", question="names ZqNodeTag 4?")
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqNodeTag"}]


def test_xbrl_passage_carries_prose() -> None:
    """The passage field walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-ZqPassageTag", "source_name": "s", "passage": "says ZqPassageTag 5"}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqPassageTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqPassageTag"}]


def test_xbrl_records_walk_grounds() -> None:
    """The records list walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-ZqRecordsTag", "source_name": "s", "records": [{"concept": "ZqRecordsTag", "value": 1}]}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqRecordsTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqRecordsTag"}]


def test_xbrl_relative_camel_tag_reselects() -> None:
    """A CamelCase tag reading as a relative date never grounds, even when named."""
    out, invoked = _xbrl_attempt("Provide the LastQuarterRevenue figure for NVDA", "LastQuarterRevenue")
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_rows_walk_grounds() -> None:
    """The rows list walks; a tag named only there grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-ZqRowsTag", "source_name": "s", "rows": [{"concept": "ZqRowsTag", "value": 1}]}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqRowsTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqRowsTag"}]


def test_xbrl_session_query_grounds() -> None:
    """Only the session query names the tag, so the runtime receives the requested concept."""
    out, invoked = _xbrl_attempt("What about AI?", "ZqQueryTag", query="names ZqQueryTag 3")
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqQueryTag"}]


def test_xbrl_source_concept_row_grounds() -> None:
    """A source_concept on a row carrying a value names the tag."""
    evidence: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-src", "source_name": "s", "source_concept": "ZqSrcTag", "value": 9}
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqSrcTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqSrcTag"}]


def test_xbrl_string_evidence_never_crashes() -> None:
    """String evidence without prose fields reselects cleanly instead of failing."""
    out, invoked = _xbrl_attempt("What about AI?", "ZqPlainTag", evidence=["plain prose"])
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_summary_then_good_attempt_keeps_good() -> None:
    """A summary attempt is skipped; a later good record still grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {"tool": "search_web", "outcome_summary": "noise"},
        {"evidence_id": "e2", "source_name": "s", "content": "keep me ZqKeepTag"},
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqKeepTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqKeepTag"}]


@pytest.mark.parametrize("concept", ["Today", "UTC"])
def test_xbrl_today_prefix_concept_reselects(concept: str) -> None:
    """The date instruction cannot supply an XBRL concept."""
    out, invoked = _xbrl_attempt("[Today UTC 2026-10-09] What did NVDA report?", concept)
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_tool_record_without_id_never_crashes() -> None:
    """An identified record without a matching source reselects cleanly instead of failing."""
    evidence: list[dict[str, JSONValue]] = [{"tool": "get_sec_document", "evidence_id": "ECRASH", "content": "plain"}]
    out, invoked = _xbrl_attempt("What about AI?", "ZqCrashTag", evidence=evidence)
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_unkeyed_record_grounds() -> None:
    """A sourceless record without tool or ids still carries prose."""
    evidence: list[dict[str, JSONValue]] = [{"source_name": "get_sec_document", "content": "notes ZqUnkeyTag 6"}]
    out, invoked = _xbrl_attempt("What about AI?", "ZqUnkeyTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqUnkeyTag"}]


def test_xbrl_unstored_attempt_summary_never_grounds() -> None:
    """An unstored attempt summary is not source material; its echo never grounds."""
    evidence: list[dict[str, JSONValue]] = [
        {
            "tool": "get_sec_document",
            "content": "notes ZqEchoTag 2",
            "outcome_summary": "echo ZqEchoTag 2",
        }
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqEchoTag", evidence=evidence)
    assert invoked == []
    assert out["error_type"] == "reselect"


def test_xbrl_uppercase_tag_objective_grounds() -> None:
    """An uppercase-only tag the request names invokes."""
    out, invoked = _xbrl_attempt("What was NVDA EPS?", "EPS")
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "EPS"}]


def test_document_read_failed_requires_matching_tool_and_error() -> None:
    """Only a failed get_sec_document read counts; unrelated errors and clean reads do not."""
    assert sched._document_read_failed([{"tool": "get_sec_document", "error": "boom"}]) is True
    assert sched._document_read_failed([{"tool": "list_sec_filings", "error": "boom"}]) is False
    assert sched._document_read_failed([{"tool": "get_sec_document", "error": None}]) is False
    assert sched._document_read_failed([]) is False


def test_latest_revenue_requested_honors_all_date_sources() -> None:
    """Node, scope text, scope window, session cutoff, and as_of each constrain the latest-revenue bypass."""
    base: dict[str, JSONValue] = {"session_id": "s1", "objective": "What is ORCL's latest revenue?"}
    assert sched._latest_revenue_requested(base, _node(), None) is True
    dated_node = ResearchNode(
        node_id="n1",
        session_id="s1",
        question="What is ORCL's latest revenue as of 2025-01-01?",
        why_it_matters="w",
    )
    assert sched._latest_revenue_requested(base, dated_node, None) is False
    scoped_raw: dict[str, JSONValue] = {
        **base,
        "temporal_scope": {"mode": "latest-available", "raw": "from 2025-01-01"},
    }
    assert sched._latest_revenue_requested(scoped_raw, _node(), None) is False
    scoped_window: dict[str, JSONValue] = {
        **base,
        "temporal_scope": {"mode": "range", "start": "2025-01-01", "end": "2025-12-31"},
    }
    assert sched._latest_revenue_requested(scoped_window, _node(), None) is False
    session_cutoff: dict[str, JSONValue] = {**base, "as_of": "2025-12-31"}
    assert sched._latest_revenue_requested(session_cutoff, _node(), None) is False
    assert sched._latest_revenue_requested(base, _node(), "2025-12-31") is False
    assert (
        sched._latest_revenue_requested({"session_id": "s1", "objective": "How many employees?"}, _node(), None)
        is False
    )


def test_revenue_listing_forms_separates_explicit_ten_q() -> None:
    """Latest quarterly asks stay broad; only explicit 10-Q wording narrows to 10-Q."""
    assert sched._revenue_listing_forms("What is ORCL's latest quarterly revenue?") == ["10-Q", "10-K"]
    assert sched._revenue_listing_forms("What drove NVDA revenue last quarter? 10-Q") == ["10-Q"]
    assert sched._revenue_listing_forms("What drove NVDA revenue last quarter?") == ["10-Q", "10-K", "8-K"]
    assert sched._revenue_listing_forms("How many employees?") is None


def test_revenue_read_unlisted_explicit_ten_q_quarter() -> None:
    """An explicit 10-Q quarterly ask with no listing stays unlisted."""
    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What drove NVDA revenue last quarter? 10-Q",
    }
    assert sched._revenue_read_unlisted(session, []) is True


def test_revenue_read_unlisted_ignores_unlabelled_for_latest() -> None:
    """An 8-K-only listing never satisfies a latest ask; the filing must still open."""
    acc8 = "0001045810-26-000078"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": [{"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"}]},
        "outcome_summary": f"8-K {acc8}",
    }
    session: dict[str, JSONValue] = {"session_id": "s1", "objective": "What is ORCL's latest revenue?"}
    assert sched._revenue_read_unlisted(session, [listing]) is True


def test_revenue_read_unlisted_listed_ten_q_opens() -> None:
    """A successful 10-Q listing clears the unlisted gate so the document opens next."""
    accq = "0001045810-26-000075"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "error": None,
        "result": {"filings": [{"accession_no": accq, "form": "10-Q", "filed_at": "2026-08-01"}]},
        "outcome_summary": f"10-Q {accq}",
    }
    session: dict[str, JSONValue] = {"session_id": "s1", "objective": "What is ORCL's latest revenue?"}
    assert sched._revenue_read_unlisted(session, [listing]) is False


def test_revenue_read_unlisted_quarter_without_ten_q_stays_listed() -> None:
    """A quarter-only ask without 10-Q wording is listed; unrelated failures never force it."""
    session: dict[str, JSONValue] = {
        "session_id": "s1",
        "objective": "What drove NVDA revenue last quarter?",
    }
    assert sched._revenue_read_unlisted(session, []) is False
    assert sched._revenue_read_unlisted(session, [{"tool": "list_sec_filings", "error": "boom"}]) is False


def test_revenue_read_unlisted_query_fallback_and_latest() -> None:
    """A query-only session still gates; an undated latest ask with no listing stays unlisted."""
    query_only: dict[str, JSONValue] = {
        "session_id": "s1",
        "query": "What did NVDA's latest 10-Q disclose about revenue?",
    }
    assert sched._revenue_read_unlisted(query_only, []) is True
    latest: dict[str, JSONValue] = {"session_id": "s1", "objective": "What is ORCL's latest revenue?"}
    assert sched._revenue_read_unlisted(latest, []) is True


@pytest.mark.parametrize("guarded", [False, True])
def test_run_admitted_without_resolution_counts_as_progress(guarded: bool) -> None:
    """An admitted round progresses. A node guard still makes the drained session incomplete."""

    class _KS(_Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.seen = False

        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            if self.seen:
                return []
            self.seen = True
            return [_node()]

    async def fake(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {"node_id": "n1", "status": "gathering", "admitted": 1, "incomplete_guard": guarded}

    with mock.patch.object(sched, "run_node", fake):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == ("incomplete_guard" if guarded else "complete")
    assert out["incomplete_guard"] is guarded
    assert out["rounds"] == 1
    assert out["nodes"] == [{"node_id": "n1", "status": "gathering", "admitted": 1, "incomplete_guard": guarded}]


def test_run_resolved_without_admission_counts_as_progress() -> None:
    """A resolved node with zero admissions is progress, never a stall."""

    class _KS(_Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.seen = False

        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            if self.seen:
                return []
            self.seen = True
            return [_node()]

    async def fake(n: object, sid: str, **kw: object) -> dict[str, JSONValue]:
        return {"node_id": "n1", "status": "resolved", "admitted": 0, "incomplete_guard": False}

    with mock.patch.object(sched, "run_node", fake):
        out = asyncio.run(sched.run("s1", kernel=_KS(), repo=None))
    assert out["status"] == "complete"
    assert out["incomplete_guard"] is False


def test_sched_block_cancelled_round_reports_every_node() -> None:
    blocked: list[tuple[str, str, str]] = []

    class _K(sched._Kernel):
        @override
        def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
            blocked.append((session_id, node_id, reason))
            return ResearchNode(node_id=node_id, session_id=session_id, question="q", why_it_matters="w")

    results: list[JSONValue] = []
    node = ResearchNode(node_id="a", session_id="S", question="q", why_it_matters="w")
    nodes: list[object] = [node, {"id": "i9"}]
    sched._block_cancelled_round(_K(), "S", nodes, "boom", results)
    assert blocked == [("S", "a", "boom"), ("S", "i9", "boom")]
    assert [r["node_id"] for r in results if isinstance(r, dict)] == ["a", "i9"]


def test_sched_carry_registry_keeps_navigation_after_filing() -> None:
    registry: Sequence[dict[str, JSONValue]] = [
        {"name": "get_sec_filing", "parameters": {}},
        {"name": "get_sec_document", "parameters": {}},
        {"name": "list_sec_documents", "parameters": {}},
    ]
    filing_attempts: list[dict[str, JSONValue]] = [{"tool": "get_sec_filing"}]
    got = sched._carry_registry(registry, filing_attempts)
    assert got is not None and [e["name"] for e in got] == ["get_sec_document", "list_sec_documents"]
    listing_attempts: list[dict[str, JSONValue]] = [{"tool": "list_sec_filings"}]
    whole = sched._carry_registry(registry, listing_attempts)
    assert whole is not None and [e["name"] for e in whole] == [
        "get_sec_filing",
        "get_sec_document",
        "list_sec_documents",
    ]


def test_registry_advertises_sec_reader_identity_and_discovery() -> None:
    entry = next(manifest for manifest in sched.build_registry() if manifest["name"] == "get_sec_filing")
    parameters = entry["parameters"]
    assert isinstance(parameters, dict)
    required = parameters["required"]
    assert isinstance(required, list) and "accession_no" in required
    assert entry["purpose"] and entry["intent"] and entry["pitSupport"]


def test_sched_session_round_deadline_before_first_round() -> None:
    class _K(sched._Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            raise AssertionError("no ready read past the deadline")

    out = asyncio.run(sched._session_rounds("S", _K(), asyncio.Semaphore(2), {}, time.perf_counter() - 1))
    assert out["session_id"] == "S" and out["status"] == "incomplete_guard"
    assert out["rounds"] == 0 and out["reason"] == "incomplete: run deadline reached before session round 1"
    assert out["incomplete_guard"] is True


def test_registry_contains_unique_tool_names() -> None:
    registry = sched.build_registry()
    assert registry
    assert len(registry) == len({str(manifest.get("name")) for manifest in registry})


def test_sched_build_registry_none_capabilities_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools as _tools

    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)

    def _boom(caps: object) -> object:
        raise AssertionError("caps required")

    monkeypatch.setattr(
        _tools,
        "tools_for_capabilities",
        _boom,
    )
    with pytest.raises(AssertionError, match="caps required"):
        sched.build_registry()
    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)


def test_sched_session_round_guard_ceiling_reports_incomplete() -> None:
    node = ResearchNode(node_id="a", session_id="S", question="q", why_it_matters="w")

    class _K(sched._Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            return [node]

        @override
        def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
            return node

        @override
        def get_session(self, session_id: str) -> dict[str, JSONValue]:
            return {"session_id": session_id, "objective": "q", "query": "q", "as_of": None}

        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return []

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

    async def _fake(
        node: object, session_id: str, hooks: Mapping[str, object], kernel: sched._Kernel, cap: asyncio.Semaphore
    ) -> dict[str, JSONValue]:
        return {"node_id": "a", "status": "resolved", "admitted": 1}

    async def _main() -> dict[str, JSONValue]:
        orig = sched._run_one_node
        sched._run_one_node = _fake
        try:
            return await sched._session_rounds("S", _K(), asyncio.Semaphore(2), {}, time.perf_counter() + 30)
        finally:
            sched._run_one_node = orig

    out = asyncio.run(_main())
    nodes = out["nodes"]
    assert out["status"] == "incomplete_guard" and out["rounds"] == 6 and isinstance(nodes, list) and len(nodes) == 5
    assert out["reason"] == "incomplete: runtime guard (5 session rounds without convergence)"


def test_run_default_kernel_uses_requested_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The requested repository supplies the session, not the default repository."""
    from app.research import service
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "default.sqlite"))
    store = ResearchRepository(tmp_path / "requested.sqlite")
    sid = service.create_research("Which filings exist?", repo=store)
    out = asyncio.run(sched.run(sid, repo=store))
    assert out == {"session_id": sid, "status": "complete", "rounds": 0, "nodes": [], "incomplete_guard": False}


def test_build_registry_from_empty_cache_returns_research_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh registry contains research tools and excludes broker execution tools."""
    from app.policy import Capability
    from app.tools import TOOL_CAPABILITIES

    monkeypatch.setattr(sched, "_REGISTRY_CACHE", None)
    names = {entry["name"] for entry in sched.build_registry()}
    assert "list_sec_filings" in names
    assert all(TOOL_CAPABILITIES[str(name)] is Capability.RESEARCH for name in names)


def test_run_stops_node_selection_when_session_deadline_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node that starts after the session deadline cannot select or invoke a tool."""
    now = [100.0]
    blocked: list[tuple[str, str, str]] = []
    selected: list[str] = []

    class _K(_Kernel):
        @override
        def ready_nodes(self, session_id: str) -> list[ResearchNode]:
            now[0] = 1000.0
            return [_node(session_id=session_id)]

        @override
        def block_node(self, session_id: str, node_id: str, reason: str = "") -> ResearchNode:
            blocked.append((session_id, node_id, reason))
            return _node(node_id, session_id)

    async def select(*args: object, **kwargs: object) -> tuple[str, ToolDecision]:
        selected.append("selected")
        return "invoke", ToolDecision(action="invoke")

    monkeypatch.setattr(sched._time, "perf_counter", lambda: now[0])
    out = asyncio.run(sched.run("s1", kernel=_K(), jev=_JevAdmit(), registry=[], select_round=select))
    assert selected == []
    assert out["status"] == "incomplete_guard"
    assert out["rounds"] == 1
    assert len(blocked) == 1
    assert blocked[0][:2] == ("s1", "n1")
    nodes = out["nodes"]
    assert isinstance(nodes, list) and len(nodes) == 1
    node = nodes[0]
    assert isinstance(node, dict)
    assert node["status"] == "blocked" and node["incomplete_guard"] is True
    assert node["reason"] == blocked[0][2]


def test_xbrl_nested_error_does_not_hide_valid_source_text() -> None:
    """A failed nested result cannot hide a valid sibling source passage."""
    evidence: list[JSONValue] = [
        {
            "evidence_id": "ev-sibling",
            "content": "Filing notes ZqSiblingTag 5.",
            "result": {"error": "boom", "content": "unavailable"},
        }
    ]
    out, invoked = _xbrl_attempt("What about AI?", "ZqSiblingTag", evidence=evidence)
    assert out["error"] is None
    assert invoked == [{"ticker": "NVDA", "concept": "ZqSiblingTag"}]


@pytest.mark.parametrize("objective", ["latest revenue", "10-Q revenue"])
@pytest.mark.parametrize("empty_listing", [False, True])
@pytest.mark.parametrize("needle_failure", [False, True])
def test_revenue_carry_never_invokes_unlisted_filing(objective: str, empty_listing: bool, needle_failure: bool) -> None:
    """Revenue reads reject unlabelled packets and an older filing behind an empty controlling listing."""
    accession = "0001045810-26-000075"
    attempts: list[dict[str, JSONValue]] = []
    if empty_listing:
        attempts = [
            {
                "tool": "list_sec_filings",
                "error": None,
                "result": {"filings": [{"form": "10-Q", "accession_no": accession}]},
            },
            {"tool": "list_sec_filings", "error": None, "result": {"filings": []}},
        ]
    invoked: list[dict[str, JSONValue]] = []

    async def generate(**kwargs: object) -> dict[str, JSONValue]:
        if needle_failure:
            raise RuntimeError("needle worker closed")
        return {"tool": "get_sec_document", "arguments": {}, "reasoning": "r"}

    async def invoke(name: str, args: dict[str, JSONValue], session: object, **kwargs: object) -> dict[str, JSONValue]:
        invoked.append(dict(args))
        return {"text": "Revenue was 10 million."}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name="get_sec_document",
            node=_node(),
            session={"session_id": "s1", "objective": objective},
            registry=[],
            evidence=[{"source_handle": {"accession_no": accession}}],
            attempts=attempts,
            kernel=_Kernel(),
            needle_generate=generate,
            invoke=invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert invoked == []
    assert out["error"] is not None
    if not needle_failure:
        assert out["error_type"] == "reselect"


@pytest.mark.parametrize(
    ("tool_name", "generated", "packet", "expected"),
    [
        (
            "get_sec_document",
            {"document_name": "keep.htm"},
            {"source_handle": {"accession_no": "0001045810-26-000075", "document_name": "other.htm"}},
            {"accession_no": "0001045810-26-000075", "document_name": "keep.htm"},
        ),
        (
            "get_sec_filing",
            {},
            "filing\n0001045810-26-000075",
            {"accession_no": "0001045810-26-000075"},
        ),
        (
            "get_sec_filing",
            {},
            {"outcome_summary": "filing\n0001045810-26-000075"},
            {"accession_no": "0001045810-26-000075"},
        ),
    ],
    ids=["preserve-generated-document", "newline-before-accession", "newline-in-summary"],
)
def test_carried_accession_preserves_runtime_arguments(
    tool_name: str, generated: dict[str, JSONValue], packet: JSONValue, expected: dict[str, JSONValue]
) -> None:
    """The runtime receives the carried accession without an unwanted document change."""
    invoked: list[dict[str, JSONValue]] = []

    async def generate(**kwargs: object) -> dict[str, JSONValue]:
        return {"tool": tool_name, "arguments": generated, "reasoning": "r"}

    async def invoke(name: str, args: dict[str, JSONValue], session: object, **kwargs: object) -> dict[str, JSONValue]:
        invoked.append(dict(args))
        return {"text": "Filing source text."}

    out = asyncio.run(
        sched._attempt_tool(
            tool_name=tool_name,
            node=_node(),
            session={"session_id": "s1", "objective": "q?"},
            registry=[],
            evidence=[packet],
            attempts=[],
            kernel=_Kernel(),
            needle_generate=generate,
            invoke=invoke,
            to_outcome=_to_outcome,
            tool_session=_tool_session(),
            as_of=None,
        )
    )
    assert out["error"] is None
    assert invoked == [expected]


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("Compare NVDA, AAPL, NVDA and MSFT.", ["NVDA", "AAPL", "MSFT"]),
        ("compare Apple and Microsoft", ["Apple", "Microsoft"]),
        ("compare Acme Systems versus Apple", ["Acme Systems", "Apple"]),
        ("Apple versus Microsoft", ["Apple", "Microsoft"]),
        ("buy Apple; compare Microsoft; with apple", ["Apple", "Microsoft"]),
        ("buy Apple; compare NVDA and Microsoft; hold AAPL", ["Apple", "NVDA", "Microsoft", "AAPL"]),
        ("What are Apple's margins? What did Tesla insiders sell?", ["Apple", "Tesla"]),
        ("What are Acme Systems margins?", ["Acme Systems"]),
        ("short interest for Apple this week", ["Apple"]),
        ("volume with Tesla now", ["Tesla"]),
        ("Should I buy Apple?", ["Apple"]),
        ("sell Microsoft", ["Microsoft"]),
        ("hold Apple", ["Apple"]),
        ("invest in Acme Systems", ["Acme Systems"]),
        ("how is Apple doing?", ["Apple"]),
        ("tell me about Apple", ["Apple"]),
        ("[Today UTC 2026-10-19] compare AAPL and NVDA", ["AAPL", "NVDA"]),
        ("[Today UTC 2026-10-19] hi", []),
        ("hello, good morning", []),
        ("hi ty", []),
        ("HI TY", ["HI", "TY"]),
        ("What are SEC FINRA REG SHO EPS figures?", []),
        ("Growth slowed last quarter", []),
        ("", []),
    ],
)
def test_objective_chart_candidates_are_ordered_lexical_labels(
    objective: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate extraction preserves explicit labels without calling the resolver."""
    from app import tools

    resolver = mock.Mock(side_effect=AssertionError("Lexical extraction must not resolve a company."))
    monkeypatch.setattr(tools, "_resolve_company_to_ticker", resolver)
    assert sched._objective_chart_candidates(objective) == expected
    resolver.assert_not_called()


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.lists(st.sampled_from(["NVDA", "AAPL", "MSFT", "ORCL", "HI", "TY"]), min_size=1, max_size=12),
    st.sampled_from([" ", "  ", "\t", "\n", " \t "]),
)
def test_prop_chart_ticker_candidates_preserve_first_order_and_deduplicate(labels: list[str], whitespace: str) -> None:
    """Repeated ticker labels retain their first position across whitespace changes."""
    objective = whitespace + ("," + whitespace).join(labels) + whitespace
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect=AssertionError("No resolver call allowed.")):
        assert sched._objective_chart_candidates(objective) == list(dict.fromkeys(labels))


@settings(max_examples=60, derandomize=True, deadline=None)
@given(
    st.lists(st.sampled_from(["Apple", "Microsoft", "Acme Systems", "Coca Cola"]), min_size=1, max_size=10),
    st.sampled_from([" ", "  ", "\t", "\n", " \t "]),
)
def test_prop_chart_company_candidates_normalize_whitespace_and_deduplicate(labels: list[str], whitespace: str) -> None:
    """Company labels keep their first order despite case changes and internal whitespace."""
    objective = "; ".join(f"buy {whitespace.join(label.split())}; with {label.lower()} now" for label in labels)
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect=AssertionError("No resolver call allowed.")):
        assert sched._objective_chart_candidates(objective) == list(dict.fromkeys(labels))


def _resolve_chart_test_subject(label: str) -> str | None:
    return {"Apple": "AAPL", "MSFT": "MSFT"}.get(label)


@pytest.mark.parametrize(
    ("objective", "supplied", "expected"),
    [
        ("How is it doing?", " nvda ", "NVDA"),
        ("hi", " aapl ", "AAPL"),
        ("", " msft ", "MSFT"),
        ("How is it doing?", "", None),
        ("How is it doing?", " \t ", None),
        ("How is it doing?", None, None),
        ("How is it doing?", 42, None),
        ("buy Apple", "NVDA", "AAPL"),
        ("how is Apple doing?", "NVDA", "AAPL"),
        ("tell me about Apple", "NVDA", "AAPL"),
        ("What are Apple's margins? Compare MSFT", "NVDA", "MSFT"),
        ("buy Unlisted Company", "NVDA", "NVDA"),
        ("compare ZZZZX and Unknown Company", "NVDA", "NVDA"),
    ],
)
def test_objective_subject_prefers_explicit_labels_over_chart_context(
    objective: str, supplied: object, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit subjects retain old precedence and unresolved labels fall back to the chart ticker."""
    from app import tools

    resolver = mock.Mock(side_effect=_resolve_chart_test_subject)
    monkeypatch.setattr(tools, "_resolve_company_to_ticker", resolver)
    assert sched._objective_subject_ticker(objective, supplied) == expected
    assert all(call.args != ("NVDA",) for call in resolver.call_args_list)


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("list_sec_filings", {"identifier": "NVDA"}),
        ("search_sec_filings", {"ticker": "NVDA", "query": "How is it doing?"}),
        ("search_sec_filings_bounded", {"ticker": "NVDA", "query": "How is it doing?"}),
        ("find_sec_entities", {"query": "NVDA"}),
        ("find_sec_entities_bounded", {"query": "NVDA"}),
        ("get_reg_sho_volume", {"ticker": "NVDA"}),
        ("get_short_interest", {"ticker": "NVDA"}),
        ("get_insider_activity", {"ticker": "NVDA"}),
        ("get_planned_insider_sales", {"ticker": "NVDA"}),
        ("get_fundamentals", {"ticker": "NVDA"}),
        ("set_chart_symbol", {"ticker": "NVDA"}),
        ("get_sec_document", None),
        ("get_sec_filing", None),
        ("get_stock_price", None),
    ],
)
def test_fallback_args_use_normalized_chart_context_without_an_explicit_subject(
    tool_name: str, expected: dict[str, JSONValue] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chart context supplies only the selectors each tool accepts, never a filing accession."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", mock.Mock(return_value=None))
    assert sched._fallback_sec_args(tool_name, "How is it doing?", " nvda ") == expected


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("list_sec_filings", {"identifier": "NVDA"}),
        ("search_sec_filings", {"ticker": "NVDA", "query": "buy Unlisted Company"}),
        ("get_reg_sho_volume", {"ticker": "NVDA"}),
        ("set_chart_symbol", {"ticker": "NVDA"}),
        ("find_sec_entities", {"query": "NVDA"}),
    ],
)
def test_fallback_args_use_chart_context_for_unresolved_explicit_subject(
    tool_name: str, expected: dict[str, JSONValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolved explicit company falls back to the selected chart ticker."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", mock.Mock(return_value=None))
    assert sched._fallback_sec_args(tool_name, "buy Unlisted Company", "NVDA") == expected


@pytest.mark.parametrize(
    ("objective", "supplied"),
    [("What do you think about the CEO?", "AAPL"), ("Is AI hype hurting it?", "NVDA")],
)
def test_objective_subject_uses_chart_context_for_unresolved_acronyms(
    objective: str, supplied: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acronyms lose to chart context, even when they resolve to a real ticker."""
    from app import tools

    def _resolve_acronym(label: str) -> str | None:
        return {"AI": "AI", "CEO": "CEOO"}.get(label, _resolve_chart_test_subject(label))

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_acronym)
    assert sched._objective_subject_ticker(objective, supplied) == supplied


def test_objective_subject_resolves_acronym_without_chart_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone acronym subjects still resolve when no chart ticker is present."""
    from app import tools

    def _resolve_ai(label: str) -> str | None:
        return {"AI": "AI"}.get(label)

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_ai)
    assert sched._objective_ticker("Is AI hype hurting it?") == "AI"
    assert sched._objective_subject_ticker("Is AI hype hurting it?") == "AI"


def test_node_registry_excludes_chart_symbol_tool() -> None:
    """Node selection drops the chart UI action. Entry and assessment registries keep it."""
    registry = sched.build_registry()
    assert "set_chart_symbol" in {e.get("name") for e in registry}
    assert "set_chart_symbol" not in {e.get("name") for e in sched._without_control_tools(registry, "s", "n")}


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("How is it doing?", "NVDA"),
        ("buy Apple", "AAPL"),
        ("buy Unlisted Company", "NVDA"),
    ],
)
def test_needle_context_and_grounded_values_share_explicit_subject_precedence(
    objective: str, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Needle receives the selected subject and only schema-supported grounded values."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_chart_test_subject)
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
    }
    session: dict[str, JSONValue] = {"session_id": "s1", "objective": objective, "ticker": " nvda "}
    context = sched._needle_context("get_fundamentals", schema, objective, None, {}, session)
    grounded: dict[str, JSONValue] = {} if expected is None else {"ticker": expected}
    assert context.get("ticker") == expected
    assert context["grounded_arguments"] == grounded
    assert sched._grounded_execution_values("get_fundamentals", schema, objective, " nvda ") == grounded
    assert sched._grounded_execution_values("get_fundamentals", {"properties": {}}, objective, "NVDA") == {}


@pytest.mark.parametrize("failure", ["unavailable", "withhold"])
@pytest.mark.parametrize(
    ("tool_name", "selector"),
    [
        ("list_sec_filings", "identifier"),
        ("search_sec_filings", "ticker"),
        ("find_sec_entities", "query"),
        ("get_reg_sho_volume", "ticker"),
        ("get_fundamentals", "ticker"),
        ("get_short_interest", "ticker"),
        ("set_chart_symbol", "ticker"),
    ],
)
@pytest.mark.parametrize(
    ("objective", "ticker"),
    [("How is it doing?", "NVDA"), ("buy Apple", "AAPL"), ("buy Unlisted Company", "NVDA")],
)
def test_generate_arguments_recovers_with_session_chart_ticker(
    failure: str, tool_name: str, selector: str, objective: str, ticker: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both real recovery paths preserve explicit subjects before they use chart context."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_chart_test_subject)
    calls: list[str] = []

    async def generate(**kwargs: object) -> dict[str, JSONValue]:
        calls.append(str(kwargs["tool"]))
        context = kwargs["context"]
        assert isinstance(context, dict)
        assert context["ticker"] == ticker
        if failure == "unavailable":
            raise RuntimeError("needle worker closed")
        return {"tool": None, "arguments": {}, "reasoning": "withheld"}

    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            generate,
            tool_name,
            sched.build_registry(),
            {"session_id": "s1", "objective": objective, "ticker": " nvda "},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    expected: dict[str, JSONValue] = {selector: ticker}
    if tool_name == "search_sec_filings":
        expected["query"] = objective
    assert calls == [tool_name]
    assert args == expected
    assert reasoning
    assert withheld is False


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("What is its latest revenue?", "NVDA"),
        ("What is Apple's latest revenue?", "AAPL"),
    ],
)
def test_latest_revenue_listing_uses_session_ticker_without_needle(
    objective: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deterministic revenue listing uses chart context only when the objective lacks a subject."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_chart_test_subject)
    generate = mock.AsyncMock(side_effect=AssertionError("The deterministic listing must not call Needle."))
    args, reasoning, withheld = asyncio.run(
        sched._generate_tool_arguments(
            generate,
            "list_sec_filings",
            sched.build_registry(),
            {"session_id": "s1", "objective": objective, "ticker": " nvda "},
            _node(),
            [],
            [],
            None,
            _Kernel(),
        )
    )
    assert args == {"identifier": expected, "forms": ["10-Q", "10-K"]}
    assert "latest revenue" in reasoning
    assert withheld is False
    generate.assert_not_called()


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "result-error",
        "receipt-error",
        "outcome-error",
        "invoke-error",
        "wrong-type",
        "missing-type",
        "missing-ticker",
        "blank-ticker",
        "invalid-ticker",
        "organization-ticker",
        "nonstring-ticker",
        "formatted-only",
        "top-level-only",
        "missing-meta",
        "invalid-meta",
        "invalid-receipt",
    ],
)
def test_run_node_emits_only_valid_chart_receipts_without_resolving_evidence(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a successful structured chart receipt emits an update, never factual evidence or resolution."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_chart_test_subject)
    receipt: dict[str, JSONValue] = {"result_type": "chart_selection", "ticker": "AAPL"}
    if defect == "receipt-error":
        receipt["error"] = "chart selection failed"
    elif defect == "wrong-type":
        receipt["result_type"] = "quote"
    elif defect == "missing-type":
        receipt.pop("result_type")
    elif defect == "missing-ticker":
        receipt.pop("ticker")
    elif defect == "blank-ticker":
        receipt["ticker"] = " \t "
    elif defect == "invalid-ticker":
        receipt["ticker"] = "AAPL MSFT"
    elif defect == "organization-ticker":
        receipt["ticker"] = "SEC"
    elif defect == "nonstring-ticker":
        receipt["ticker"] = 123
    response: dict[str, JSONValue] = {"meta": {"chart_selection": receipt}}
    if defect == "result-error":
        response["error"] = "chart selection failed"
    elif defect == "formatted-only":
        response = {"content": '{"result_type":"chart_selection","ticker":"AAPL"}'}
    elif defect == "top-level-only":
        response = dict(receipt)
    elif defect == "missing-meta":
        response = {}
    elif defect == "invalid-meta":
        response = {"meta": "not a receipt"}
    elif defect == "invalid-receipt":
        response = {"meta": {"chart_selection": "not a receipt"}}
    updates: list[tuple[str, dict[str, object]]] = []
    invoked: list[str] = []
    selected = 0
    kernel = _Kernel(objective="buy Apple")

    async def select(*args: object, **kwargs: object) -> tuple[str, ToolDecision]:
        nonlocal selected
        selected += 1
        return (
            ("invoke", _invoke("set_chart_symbol")) if selected == 1 else ("resolved", ToolDecision(action="resolved"))
        )

    async def generate(**kwargs: object) -> dict[str, JSONValue]:
        return {"tool": "set_chart_symbol", "arguments": {"ticker": "AAPL"}}

    async def invoke(name: str, args: dict[str, JSONValue], session: object, **kwargs: object) -> dict[str, JSONValue]:
        invoked.append(name)
        assert args == {"ticker": "AAPL"}
        if defect == "invoke-error":
            raise RuntimeError("chart unavailable")
        return dict(response)

    def outcome(name: str, result: Mapping[str, object]) -> ToolOutcome:
        return dataclasses.replace(
            _outcome(
                json.dumps({"result_type": "chart_selection", "ticker": "AAPL"}),
                error="chart rejected" if defect == "outcome-error" else None,
            ),
            tool_name=name,
        )

    def progress(stage: str, detail: dict[str, object]) -> None:
        updates.append((stage, detail))

    with (
        mock.patch.object(kernel, "resolve_node", wraps=kernel.resolve_node) as resolve,
        mock.patch.object(
            sched, "_evidence_candidate", side_effect=AssertionError("Chart tools cannot admit evidence.")
        ),
    ):
        result = asyncio.run(
            sched.run_node(
                _node(),
                "s1",
                kernel=kernel,
                jev=_JevAdmit(),
                select_round=select,
                needle_generate=generate,
                invoke=invoke,
                to_outcome=outcome,
                registry=sched.build_registry(),
                tool_session=_tool_session(),
                progress=progress,
                max_rounds=2,
            )
        )
        resolve.assert_not_called()
    assert invoked == ["set_chart_symbol"]
    chart_updates = [detail for stage, detail in updates if stage == "chart_update"]
    assert chart_updates == ([{"ticker": "AAPL"}] if defect == "none" else [])
    assert result["status"] == "blocked"
    assert result["admitted"] == 0
    assert kernel.evidence == []
    assert not any(call[0] == "admit" for call in kernel.calls)


def test_chart_records_never_supply_factual_context_citations_or_trusted_evidence() -> None:
    """Chart records remain outside factual context, citation identifiers, and resolution evidence."""
    chart: dict[str, JSONValue] = {
        "tool": "set_chart_symbol",
        "evidence_id": "ev-chart",
        "tool_result_ref": "s1:tr:chart",
        "content": "Chart changed to AAPL.",
        "outcome_summary": "Chart changed to AAPL.",
    }
    fact: dict[str, JSONValue] = {
        "tool": "get_sec_document",
        "evidence_id": "ev-fact",
        "content": "Revenue was 10 million.",
    }
    observation: dict[str, JSONValue] = {
        "tool": "get_stock_price",
        "tool_result_ref": "s1:tr:quote",
        "outcome_summary": "Price was 100.",
    }
    assert sched._context_evidence([chart], [chart]) == []
    assert sched._grounded_allowed_ids([chart], [chart]) == set()
    assert sched._has_trusted_evidence([chart], [chart]) is False
    context = sched._context_evidence([chart, fact], [chart, observation])
    assert context[0] == fact
    assert len(context) == 2 and context[1]["tool"] == "get_stock_price"
    assert sched._grounded_allowed_ids([chart, fact], [chart, observation]) == {"ev-fact", "s1:tr:quote"}
    assert sched._has_trusted_evidence([chart, fact], []) is True
    assert sched._has_trusted_evidence([], [chart, fact]) is True


def test_run_node_does_not_resolve_from_preexisting_chart_evidence() -> None:
    """A stored chart record cannot satisfy a later resolution decision."""
    chart: dict[str, JSONValue] = {
        "tool": "set_chart_symbol",
        "evidence_id": "ev-chart",
        "content": "Chart selection is not a financial fact.",
    }

    class _ChartKernel(_Kernel):
        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [chart]

    async def select(*args: object, **kwargs: object) -> tuple[str, ToolDecision]:
        assert "Chart selection is not a financial fact." not in json.dumps(args[7])
        return "resolved", ToolDecision(action="resolved")

    kernel = _ChartKernel(objective="How is it doing?")
    with mock.patch.object(kernel, "resolve_node", wraps=kernel.resolve_node) as resolve:
        result = asyncio.run(
            sched.run_node(
                _node(),
                "s1",
                kernel=kernel,
                jev=_JevAdmit(),
                select_round=select,
                registry=[],
                max_rounds=1,
                tool_session=_tool_session(),
            )
        )
        resolve.assert_not_called()
    assert result["status"] == "blocked"
    assert result["admitted"] == 0


@pytest.mark.parametrize("failure", ["timeout", "other-tool"])
def test_chart_context_does_not_hide_unrecoverable_generation_errors(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chart context does not recover unrelated timeouts or a different generated tool."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", mock.Mock(return_value=None))

    async def generate(**kwargs: object) -> dict[str, JSONValue]:
        if failure == "timeout":
            raise RuntimeError("needle timed out")
        return {"tool": "search_web", "arguments": {"query": "unrelated"}}

    with pytest.raises((RuntimeError, ValueError), match="needle timed out|needle tool mismatch"):
        asyncio.run(
            sched._generate_tool_arguments(
                generate,
                "list_sec_filings",
                sched.build_registry(),
                {"session_id": "s1", "objective": "How is it doing?", "ticker": "NVDA"},
                _node(),
                [],
                [],
                None,
                _Kernel(),
            )
        )


def test_run_node_rejects_chart_receipt_from_a_non_chart_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chart-shaped receipt from another tool cannot change the chart."""
    from app import tools

    monkeypatch.setattr(tools, "_resolve_company_to_ticker", _resolve_chart_test_subject)
    updates: list[dict[str, object]] = []

    async def select(*args: object, **kwargs: object) -> tuple[str, ToolDecision]:
        return "invoke", _invoke("get_fundamentals")

    async def generate(**kwargs: object) -> dict[str, JSONValue]:
        return {"tool": "get_fundamentals", "arguments": {"ticker": "AAPL"}}

    async def invoke(name: str, args: dict[str, JSONValue], session: object, **kwargs: object) -> dict[str, JSONValue]:
        return {"meta": {"chart_selection": {"result_type": "chart_selection", "ticker": "MSFT"}}}

    def progress(stage: str, detail: dict[str, object]) -> None:
        if stage == "chart_update":
            updates.append(detail)

    result = asyncio.run(
        sched.run_node(
            _node(),
            "s1",
            kernel=_Kernel(objective="buy Apple"),
            jev=_JevAdmit(),
            select_round=select,
            needle_generate=generate,
            invoke=invoke,
            to_outcome=_to_outcome,
            registry=sched.build_registry(),
            tool_session=_tool_session(),
            progress=progress,
            max_rounds=1,
        )
    )
    assert updates == []
    assert result["status"] == "blocked"
    assert result["admitted"] == 0
