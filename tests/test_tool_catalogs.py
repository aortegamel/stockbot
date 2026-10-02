"""Tool-catalog select: default two-step JEV path keeps payloads small."""

import asyncio
from typing import Any

from app.research import scheduler
from app.research.models import ToolDecision
from app.research.tool_catalogs import build_tool_catalogs, catalog_options, catalog_select_round


class _Kernel:
    def record_decision(self, sid: Any, dtype: Any, **kw: Any) -> None:
        pass


def _node(question: Any) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(node_id="n1", session_id="s1", question=question, why_it_matters="w")


class _J:
    """Catalog pick -> sec; subset select -> list_sec_filings."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def decide(
        self,
        state: Any,
        questions: Any,
        decision_type: Any = "",
        session_id: Any = "",
        node_id: Any = None,
        job_id: Any = None,
        choice_options: Any = None,
    ) -> Any:
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

    async def select_tool(
        self,
        objective: Any,
        node: Any,
        registry: Any = None,
        evidence: Any = None,
        attempts: Any = None,
        session_id: Any = "",
        job_id: Any = None,
    ) -> Any:
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
    act, dec = asyncio.run(
        catalog_select_round(
            _J(),
            _Kernel(),
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


def test_catalog_sentinel_short_circuits_subset() -> None:
    class _JSentinel(_J):
        async def decide(  # type: ignore[override]
            self,
            state: Any,
            questions: Any,
            decision_type: Any = "",
            session_id: Any = "",
            node_id: Any = None,
            job_id: Any = None,
            choice_options: Any = None,
        ) -> Any:
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
    act, dec = asyncio.run(
        catalog_select_round(
            _JSentinel(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": "q?"},
            _node("q?"),
            reg,
            [],
            [],
        )
    )
    assert act == "reason" and dec["tool_name"] is None


def test_pick_state_trims_blobs_and_caps_items() -> None:
    """Pick state keeps last-5 tool/error/outcome rows, blobs truncated, non-dicts dropped."""
    from app.research.tool_catalogs import _pick_state

    items = [{"tool": "t", "outcome_summary": "x" * 5000, "error": "e" * 500}] * 20 + ["junk", 7]
    rows = _pick_state(items)
    assert len(rows) == 5
    assert all(isinstance(r, dict) for r in rows)
    assert all(len(str(r.get("outcome_summary", ""))) <= 200 for r in rows)
    assert _pick_state("nope") == []


def test_catalog_covers_registry_and_stays_small() -> None:
    reg = scheduler.build_registry()
    cats = build_tool_catalogs(reg)
    covered = sorted(t for c in cats for t in c["tools"])
    assert covered == sorted(e["name"] for e in reg)
    assert len(catalog_options(cats)) < len(reg)
