"""Tool-catalog select: default two-step JEV path keeps payloads small."""

import asyncio
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import override

from app.decision_client import JevClient
from app.research import scheduler
from app.research.models import DecisionRecord, JSONValue, ToolDecision
from app.research.tool_catalogs import build_tool_catalogs, catalog_options, catalog_select_round


class _Kernel(scheduler._Kernel):
    """Decision log only; catalog select persists through record_decision."""

    def __init__(self) -> None:
        super().__init__()
        self.recorded: list[tuple[str, object]] = []

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
        self.recorded.append((decision_type, selected))
        return DecisionRecord(
            decision_id="dec-test",
            session_id=session_id,
            node_id=node_id,
            job_id=job_id,
            decision_type=decision_type,
            candidates={},
            probabilities={},
            selected={},
        )


def _node(question: str) -> SimpleNamespace:
    return SimpleNamespace(node_id="n1", session_id="s1", question=question, why_it_matters="w")


class _J(JevClient):
    """Catalog pick -> sec; subset select -> list_sec_filings."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int]] = []

    @override
    async def decide(
        self,
        state: object,
        questions: Mapping[str, JSONValue],
        *,
        decision_type: str,
        session_id: str,
        node_id: str | None = None,
        job_id: str | None = None,
        choice_options: Mapping[str, Mapping[str, str]] | None = None,
    ) -> dict[str, dict[str, JSONValue]]:
        qid = next(iter(questions))
        opts = (choice_options or {}).get(qid, {})
        self.calls.append((decision_type, len(opts)))
        assert decision_type == "catalog_selection"
        assert "sec" in opts
        return {
            qid: {
                "choice": "sec",
                "probabilities": {k: (1.0 if k == "sec" else 0.0) for k in opts},
                "confidence": 0.9,
            }
        }

    @override
    async def select_tool(
        self,
        objective: str,
        node: object,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
        evidence: Sequence[JSONValue] | Mapping[str, JSONValue] | None = None,
        attempts: Sequence[JSONValue] | Mapping[str, JSONValue] | None = None,
        *,
        session_id: str,
        job_id: str | None = None,
    ) -> ToolDecision:
        reg = list(registry or [])
        self.calls.append(("tool_selection", len(reg)))
        assert len(reg) < len(scheduler.build_registry())
        assert any(e.get("name") == "list_sec_filings" for e in reg)
        return ToolDecision(
            action="invoke",
            tool_name="list_sec_filings",
            tool_names=("list_sec_filings",),
            probabilities={"list_sec_filings": 1.0},
            confidence=0.9,
        )


def test_catalog_pick_then_subset_select() -> None:
    reg = scheduler.build_registry()
    kernel = _Kernel()
    act, dec = asyncio.run(
        catalog_select_round(
            _J(),
            kernel,
            "s1",
            "n1",
            {"session_id": "s1", "objective": "List Apple's most recent 10-K and 10-Q filings."},
            _node("List Apple's most recent 10-K and 10-Q filings."),
            reg,
            [],
            [],
        )
    )
    assert (act, dec.tool_name) == ("invoke", "list_sec_filings")
    assert kernel.recorded == [("tool_selection", "list_sec_filings")]


def test_catalog_sentinel_short_circuits_subset() -> None:
    class _JSentinel(_J):
        @override
        async def decide(
            self,
            state: object,
            questions: Mapping[str, JSONValue],
            *,
            decision_type: str,
            session_id: str,
            node_id: str | None = None,
            job_id: str | None = None,
            choice_options: Mapping[str, Mapping[str, str]] | None = None,
        ) -> dict[str, dict[str, JSONValue]]:
            qid = next(iter(questions))
            opts = (choice_options or {}).get(qid, {})
            return {
                qid: {
                    "choice": "reasoning_required",
                    "probabilities": {k: (1.0 if k == "reasoning_required" else 0.0) for k in opts},
                    "confidence": 0.9,
                }
            }

    reg = scheduler.build_registry()
    kernel = _Kernel()
    act, dec = asyncio.run(
        catalog_select_round(
            _JSentinel(),
            kernel,
            "s1",
            "n1",
            {"session_id": "s1", "objective": "q?"},
            _node("q?"),
            reg,
            [],
            [],
        )
    )
    assert act == "reason"
    assert dec == ToolDecision(action="reason")
    assert kernel.recorded == []


def test_pick_state_trims_blobs_and_caps_items() -> None:
    """Pick state keeps last-5 tool/error/outcome rows, blobs truncated, non-dicts dropped."""
    from app.research.tool_catalogs import _pick_state

    items = [{"tool": "t", "outcome_summary": "x" * 5000, "error": "e" * 500}] * 20 + ["junk", 7]
    rows = _pick_state(items)
    assert len(rows) == 5
    assert all(isinstance(r, dict) for r in rows)
    assert all(isinstance(r, dict) and len(str(r.get("outcome_summary", ""))) <= 200 for r in rows)
    assert _pick_state("nope") == []


def test_catalog_covers_registry_and_stays_small() -> None:
    reg = scheduler.build_registry()
    cats = build_tool_catalogs(reg)
    covered: list[str] = []
    for c in cats:
        tools = c["tools"]
        assert isinstance(tools, list)
        covered.extend(str(t) for t in tools)
    assert sorted(covered) == sorted(str(e["name"]) for e in reg)
    assert len(catalog_options(cats)) < len(reg)
