"""Tool-catalog select: default two-step JEV path keeps payloads small."""

import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision_client import JevClient
from app.research import scheduler, service
from app.research.models import DecisionRecord, JSONValue, ToolDecision
from app.research.repository import ResearchRepository
from app.research.tool_catalogs import (
    _catalog_subset,
    _catalog_tools,
    build_tool_catalogs,
    catalog_options,
    catalog_select_round,
)


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
        evidence: Sequence[JSONValue] | None = None,
        attempts: Sequence[JSONValue] | None = None,
        *,
        session_id: str,
        job_id: str | None = None,
        ticker: str | None = None,
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
            {
                "session_id": "s1",
                "objective": "List Apple's most recent 10-K and 10-Q filings.",
            },
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

    items = [{"tool": "t", "outcome_summary": "x" * 5000, "error": "e" * 500}] * 20 + [
        "junk",
        7,
    ]
    rows = _pick_state(items)
    assert len(rows) == 5
    assert all(isinstance(r, dict) for r in rows)
    assert all(isinstance(r, dict) and len(str(r.get("outcome_summary", ""))) <= 200 for r in rows)
    assert _pick_state("nope") == []


def test_pick_state_keeps_only_string_fields_in_order() -> None:
    """Empty rows survive, non-string or empty fields drop, and each kept string field slices to 200."""
    from app.research.tool_catalogs import _pick_state

    picked = _pick_state(
        [
            {"tool": "first", "error": "", "error_type": None},
            {"tool": 7, "error": "x" * 300, "outcome_summary": " spaced \t text "},
            {},
        ]
    )
    rows = [row for row in picked if isinstance(row, dict)]
    assert len(rows) == len(picked)
    assert [row.get("tool") for row in rows] == ["first", None, None]
    assert rows[1]["error"] == "x" * 200
    assert rows[1]["outcome_summary"] == "spaced text"
    assert rows[2] == {}
    assert "error_type" not in rows[0]


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


# --- Boundary reads, fallback paths, and the winner subset ---


class _JCapture(JevClient):
    """Catalog pick with a caller-chosen answer; records the payload JEV received."""

    def __init__(self, decision: JSONValue | None) -> None:
        super().__init__()
        self.decision = decision
        self.states: list[dict[str, JSONValue]] = []
        self.questions: list[Mapping[str, JSONValue]] = []
        self.selected_registry: list[Mapping[str, JSONValue]] = []

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
        assert decision_type == "catalog_selection"
        assert isinstance(state, dict)
        self.states.append(state)
        self.questions.append(questions)
        if self.decision is None:
            return {}
        qid = next(iter(questions))
        return {qid: {"choice": self.decision, "probabilities": {}, "confidence": 0.9}}

    @override
    async def select_tool(
        self,
        objective: str,
        node: object,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
        evidence: Sequence[JSONValue] | None = None,
        attempts: Sequence[JSONValue] | None = None,
        *,
        session_id: str,
        job_id: str | None = None,
        ticker: str | None = None,
    ) -> ToolDecision:
        self.selected_registry = list(registry or [])
        return ToolDecision(
            action="invoke",
            tool_name="list_sec_filings",
            tool_names=("list_sec_filings",),
            probabilities={"list_sec_filings": 1.0},
            confidence=0.9,
        )


def _instructions(jev: _JCapture) -> object:
    """Instructions handed to JEV for the catalog choice."""
    inner = jev.questions[0]["catalog_selection"]
    assert isinstance(inner, dict)
    return inner["instructions"]


def test_catalog_instructions_name_the_node_question() -> None:
    """A dict node's question leads the JEV instructions and reaches the state payload."""
    jev = _JCapture("sec")
    act, dec = asyncio.run(
        catalog_select_round(
            jev,
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": "objective text"},
            {"node_id": "n1", "question": "Which filings mention X?"},
            scheduler.build_registry(),
            [],
            [],
        )
    )
    assert (act, dec.tool_name) == ("invoke", "list_sec_filings")
    assert jev.states[0]["question"] == "Which filings mention X?"
    assert jev.states[0]["objective"] == "objective text"
    assert _instructions(jev) == (
        "Select the one tool catalog (domain) most likely to advance this research question: Which filings mention X?"
    )


def test_catalog_instructions_generic_without_text() -> None:
    """Absent or non-text objective and question keep the generic instruction and blank state."""
    jev = _JCapture("sec")
    act, dec = asyncio.run(
        catalog_select_round(
            jev,
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1"},
            {"node_id": "n1", "question": 7},
            scheduler.build_registry(),
            [],
            [],
        )
    )
    assert (act, dec.tool_name) == ("invoke", "list_sec_filings")
    assert jev.states[0]["question"] == ""
    assert jev.states[0]["objective"] == ""
    assert _instructions(jev) == (
        "Select the one tool catalog (domain) most likely to advance the current research question."
    )


def test_catalog_answer_outside_options_falls_back_to_whole_registry() -> None:
    """An unknown or missing catalog choice runs the normal select over the whole registry."""
    reg = scheduler.build_registry()
    for decision in ("zzz-not-a-catalog", None):
        jev = _JCapture(decision)
        kernel = _Kernel()
        act, dec = asyncio.run(
            catalog_select_round(
                jev,
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
        assert (act, dec.tool_name) == ("invoke", "list_sec_filings")
        assert [str(entry["name"]) for entry in jev.selected_registry] == [str(entry["name"]) for entry in reg]
        assert kernel.recorded == [("tool_selection", "list_sec_filings")]


def test_catalog_winner_subset_keeps_registry_order() -> None:
    """The winning catalog's tools reach the subset select in registry order, without other domains."""
    reg = scheduler.build_registry()
    catalogs = build_tool_catalogs(reg)
    sec_tools = next(catalog["tools"] for catalog in catalogs if catalog.get("name") == "sec")
    assert isinstance(sec_tools, list)
    jev = _JCapture("sec")
    act, dec = asyncio.run(
        catalog_select_round(
            jev,
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
    names = [str(entry["name"]) for entry in jev.selected_registry]
    sec_names = {str(tool) for tool in sec_tools}
    assert names == [str(entry["name"]) for entry in reg if str(entry["name"]) in sec_names]
    assert "search_web" not in names


def test_catalog_tools_and_subset_skip_malformed_rows() -> None:
    """A catalog without a tool list contributes nothing, and unnamed registry rows never pass."""
    assert _catalog_tools([{"name": "sec"}], "sec") == set()
    assert _catalog_tools([{"name": "sec", "tools": ["list_sec_filings", 7]}], "sec") == {"list_sec_filings"}
    assert _catalog_subset([{"name": 7}, {"name": "x"}], {"x"}) == [{"name": "x"}]
    assert _catalog_subset([{"name": "x"}], set()) == []
    catalogs = build_tool_catalogs([{"name": ""}, {"name": 7}, {"name": "list_sec_filings"}])
    covered: list[str] = []
    for catalog in catalogs:
        tools = catalog["tools"]
        assert isinstance(tools, list)
        covered.extend(str(tool) for tool in tools)
    assert covered == ["list_sec_filings"]
    options = catalog_options(
        [
            {"name": "", "description": "blank name"},
            {"name": "sec", "description": ""},
            {"name": "sec", "description": "SEC filings and filing text."},
        ]
    )
    assert options["sec"] == "SEC filings and filing text."
    assert "" not in options


@settings(derandomize=True, max_examples=25, deadline=None)
@given(st.lists(st.text(alphabet="abcdefgh_", min_size=1, max_size=8), max_size=10))
def test_prop_catalog_grouping_covers_every_entry_in_sorted_order(names: list[str]) -> None:
    """Any registry: catalogs cover exactly the input names, with sorted catalog and tool names."""
    catalogs = build_tool_catalogs([{"name": name} for name in names])
    covered: list[str] = []
    for catalog in catalogs:
        tools = catalog["tools"]
        assert isinstance(tools, list)
        assert [str(tool) for tool in tools] == sorted(str(tool) for tool in tools)
        covered.extend(str(tool) for tool in tools)
    assert sorted(covered) == sorted(names)
    catalog_names = [str(catalog["name"]) for catalog in catalogs]
    assert catalog_names == sorted(catalog_names)


# --- Real persistence: stored decisions stay linked to the session and node ---


class _CatalogDecideJev(JevClient):
    """Real catalog decide over the transport; the downstream select always picks list_sec_filings."""

    @override
    async def select_tool(
        self,
        objective: str,
        node: object,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
        evidence: Sequence[JSONValue] | None = None,
        attempts: Sequence[JSONValue] | None = None,
        *,
        session_id: str,
        job_id: str | None = None,
        ticker: str | None = None,
    ) -> ToolDecision:
        return ToolDecision(
            action="invoke",
            tool_name="list_sec_filings",
            tool_names=("list_sec_filings",),
            probabilities={"list_sec_filings": 1.0},
            confidence=0.9,
        )


class _OffMenuCatalogJev(_CatalogDecideJev):
    """The catalog answer falls outside its options.

    The real decide rejects an off-menu choice, so only a client override reaches the fallback.
    """

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
        return {"catalog_selection": {"kind": "choice", "choice": "zzz-not-a-catalog"}}


@pytest.mark.parametrize(
    ("jev_cls", "persisted"),
    [
        (_CatalogDecideJev, {"catalog_selection": 1, "tool_selection": 1}),
        (_OffMenuCatalogJev, {"tool_selection": 1}),
    ],
    ids=["catalog-winner", "off-menu-fallback"],
)
def test_catalog_round_persists_decisions_linked_to_session_and_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jev_cls: type[JevClient], persisted: dict[str, int]
) -> None:
    """Winner and fallback rounds keep the invoke result; every stored decision links to the session and node."""
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    store = ResearchRepository(data_root=tmp_path)
    sid = service.create_research("List Apple's most recent 10-K and 10-Q filings.", repo=store)
    node = service.create_node(sid, "Which recent Apple 10-K and 10-Q filings exist?", "w", repo=store)
    kernel = scheduler._default_kernel(store)
    reg = scheduler.build_registry()
    probabilities: dict[str, JSONValue] = {
        name: (1.0 if name == "sec" else 0.0) for name in catalog_options(build_tool_catalogs(reg))
    }
    answer: dict[str, JSONValue] = {
        "answers": {
            "catalog_selection": {"type": "choice", "choice": "sec", "probabilities": probabilities, "confidence": 0.9}
        }
    }

    def _transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        return answer

    jev = jev_cls(transport=_transport, data_root=tmp_path)
    act, dec = asyncio.run(
        catalog_select_round(jev, kernel, sid, node.node_id, kernel.get_session(sid), node, reg, [], [])
    )
    assert (act, dec.tool_name) == ("invoke", "list_sec_filings")
    decisions = store.list_decisions(sid)
    assert Counter(d.decision_type for d in decisions) == persisted
    assert {(d.session_id, d.node_id) for d in decisions} == {(sid, node.node_id)}
    by_id = {d.decision_id: d for d in decisions}
    journaled = [
        by_id[str(event.payload["decision_id"])]
        for event in store.list_events(sid)
        if event.event_type == "decision.recorded" and event.payload.get("type") == "tool_selection"
    ]
    assert [(d.node_id, d.selected) for d in journaled] == [(node.node_id, "list_sec_filings")]
