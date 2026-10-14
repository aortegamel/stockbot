"""Fail-closed guard contracts: failures propagate, envelopes preserved."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import cast, override
from unittest import mock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.decision_client import _PERSONA_OPTIONS, JevClient
from app.research import kernel_worker as kw
from app.research import scheduler as sched
from app.research.models import DecisionRecord, JSONValue, ResearchNode
from app.research.repository import ResearchRepository


def _no_bootstrap(sid: str) -> None:
    return None


def _evidence_none(self: ResearchRepository, sid: str) -> list[dict[str, JSONValue]]:
    return []


def _nodes_none(self: ResearchRepository, sid: str) -> list[ResearchNode]:
    return []


def _decisions_none(self: ResearchRepository, sid: str) -> list[DecisionRecord]:
    return []


def _empty_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ResearchRepository, "list_evidence", _evidence_none)
    monkeypatch.setattr(ResearchRepository, "list_nodes", _nodes_none)
    monkeypatch.setattr(ResearchRepository, "list_decisions", _decisions_none)


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
        {
            "id": "s-q1",
            "objectiveId": "s",
            "question": "Other angle?",
            "dependsOn": [],
            "whyItMatters": "w",
        },
        {
            "id": "s-q2",
            "objectiveId": "s",
            "question": "Exact user objective?",
            "dependsOn": [],
            "whyItMatters": "w",
        },
    ]


class _JevDown(JevClient):
    @override
    async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
        raise RuntimeError("jev down")


def test_jev_outage_propagates() -> None:
    with pytest.raises(RuntimeError, match="jev down"):
        kw._jev_admit("s", "Exact user objective?", _props(), jev=_JevDown())


def test_jev_outage_missing_objective_propagates() -> None:
    with pytest.raises(RuntimeError, match="jev down"):
        kw._jev_admit("s", "Missing objective?", _props(), jev=_JevDown())


class _JevAdmitTwoOfThree(JevClient):
    @override
    async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
        return {
            "s-q1": {"choice": "analyze"},
            "s-q2": {"choice": "gather_evidence"},
            "s-q3": {"choice": "reject"},
        }


def test_jev_success_preserves_all_admitted_proposals() -> None:
    props: list[dict[str, object]] = [
        {
            "id": "s-q1",
            "objectiveId": "s",
            "question": "Other angle?",
            "dependsOn": [],
            "whyItMatters": "w",
        },
        {
            "id": "s-q2",
            "objectiveId": "s",
            "question": "Exact user objective?",
            "dependsOn": [],
            "whyItMatters": "w",
        },
        {
            "id": "s-q3",
            "objectiveId": "s",
            "question": "Tangent?",
            "dependsOn": [],
            "whyItMatters": "w",
        },
    ]
    out = kw._jev_admit("s", "Exact user objective?", props, jev=_JevAdmitTwoOfThree())
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


class _JevRoute(JevClient):
    def __init__(
        self,
        choice: str | None = None,
        personas: list[str] | None = None,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self.choice = choice
        self.personas = personas
        self.fail = fail
        self.seen: list[tuple[str, str | None]] = []
        self.watchlists: list[Sequence[str] | None] = []

    @override
    async def route_entry(
        self,
        prompt: str,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
        route_hint: str | None = None,
        watchlist: Sequence[str] | None = None,
    ) -> dict[str, JSONValue]:
        if self.fail:
            raise RuntimeError("jev down")
        self.seen.append((prompt, route_hint))
        self.watchlists.append(watchlist)
        return {
            "route": route_hint or self.choice or "research_required",
            "personas": list[JSONValue](self.personas) if self.personas is not None else None,
        }


@pytest.mark.parametrize("rid", [None, 7, False, [], {}])
def test_route_invalid_request_id_uses_protocol_placeholder(rid: JSONValue) -> None:
    out = kw._route({"id": rid, "prompt": "hello"}, jev=_JevRoute("reasoning_required"))
    assert out == {"id": "?", "route": "reasoning_required", "personas": None}


def test_route_reasoning_choice_returns_fast_path() -> None:
    out = kw._route(
        {"id": "r1", "op": "route", "prompt": "hello"},
        jev=_JevRoute("reasoning_required"),
    )
    assert out == {"id": "r1", "route": "reasoning_required", "personas": None}


def test_route_selector_outage_is_error_not_assumed_personas() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "hello"}, jev=_JevRoute(fail=True))
    assert out == {"id": "r1", "error": "jev down"}


def test_route_blank_prompt_needs_no_jev() -> None:
    jev = _JevRoute("reasoning_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "  "}, jev=jev)
    assert out == {"id": "r1", "route": "research_required", "personas": None}
    assert jev.seen == []


def test_route_tool_winner_returns_exact_tool() -> None:
    out = kw._route(
        {"id": "r1", "op": "route", "prompt": "what time is it?"},
        jev=_JevRoute("get_current_time"),
    )
    assert out == {"id": "r1", "route": "get_current_time", "personas": None}


def test_route_fast_path_keeps_programmatic_route_and_jev_personas() -> None:
    jev = _JevRoute("research_required", personas=["bearbot"])
    out = kw._route({"id": "r1", "op": "route", "prompt": "what time is it?"}, jev=jev)
    assert out == {"id": "r1", "route": "get_current_time", "personas": ["bearbot"]}
    assert [hint for _, hint in jev.seen] == ["get_current_time"]


def test_route_prefixed_clock_keeps_prompt_and_returns_clock() -> None:
    """Exact NVDA clock regression: the [NVDA] prefix never diverts the clock ask."""
    jev = _JevRoute("research_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "What time is it?", "ticker": "NVDA"}, jev=jev)
    assert out == {"id": "r1", "route": "get_current_time", "personas": None}
    assert jev.seen[0][0].startswith("[Today UTC ")
    assert "What time is it?" in jev.seen[0][0]
    assert "[NVDA]" not in jev.seen[0][0]


def test_route_bad_ticker_is_error() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "hello", "ticker": "FINRA"}, jev=_JevRoute("x"))
    assert out["id"] == "r1" and "error" in out


def test_route_forwards_watchlist_as_context_and_keeps_prompt_verbatim() -> None:
    """The list rides beside the prompt; nothing is appended to the text JEV and Needle read."""
    jev = _JevRoute("research_required")
    out = kw._route(
        {"id": "r1", "op": "route", "prompt": "Latest filings for my watchlist?", "watchlist": ["nvda", "QQQ", "nvda"]},
        jev=jev,
    )
    assert out == {"id": "r1", "route": "research_required", "personas": None}
    assert jev.watchlists == [["NVDA", "QQQ"]]
    assert jev.seen[0][0].startswith("[Today UTC ")
    assert "Latest filings for my watchlist?" in jev.seen[0][0]
    for sym in ("NVDA", "QQQ"):
        assert sym not in jev.seen[0][0]


@pytest.mark.parametrize("bad", ["NVDA", [], [""], [7], [f"T{i}" for i in range(26)]])
def test_route_bad_watchlist_is_error_without_jev(bad: JSONValue) -> None:
    jev = _JevRoute("research_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "my watchlist?", "watchlist": bad}, jev=jev)
    assert out["id"] == "r1" and "watchlist" in str(out["error"])
    assert jev.seen == []


def test_route_entry_states_watchlist_scope_only_in_entry_instructions() -> None:
    """Entry instructions carry the scope fact; the personas question and the prompt text stay scope-free."""
    out, seen = _route_entry(
        "Latest filings for my watchlist?",
        {"entry": "research_required"},
        watchlist=["NVDA", "QQQ"],
    )
    assert out == {"route": "research_required", "personas": None}
    questions = seen["questions"]
    assert isinstance(questions, dict)
    entry_text = str(questions["entry"]["instructions"])
    assert "The user's watchlist is NVDA, QQQ" in entry_text
    assert "single-shot tool never fits" in entry_text
    assert entry_text.startswith("Route this entry prompt with exactly one winner: [Today UTC ")
    assert "Latest filings for my watchlist?" in entry_text
    assert "watchlist is" not in str(questions["personas"]["instructions"])


def test_route_entry_without_watchlist_keeps_entry_instructions_unchanged() -> None:
    _, seen = _route_entry("Latest filings for my watchlist?", {"entry": "research_required"})
    questions = seen["questions"]
    assert isinstance(questions, dict)
    assert "watchlist is" not in str(questions["entry"]["instructions"])


@pytest.mark.parametrize("bad", ["NVDA", [], [""], [7], [object()]])
def test_route_entry_rejects_blank_watchlist(bad: object) -> None:
    with pytest.raises(ValueError, match="blank watchlist"):
        _route_entry("my watchlist?", {"entry": "research_required"}, watchlist=cast(Sequence[str], bad))


def test_arguments_context_carries_ticker_without_raw_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw evidence/attempts never cross; the ticker rides beside grounded scalar args."""
    gen = _FailGenerate(
        {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "confidence": 0.8, "reasoning": "r"}
    )
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)
    out = kw._arguments(
        {
            "id": "a1",
            "op": "arguments",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
            "ticker": "NVDA",
            "context": {
                "evidence": [{"evidence_id": "ev-1", "content": "raw bytes"}],
                "attempts": [{"tool": "list_sec_filings"}],
                "grounding_hint": "do not send",
                "as_of": None,
                "temporal_scope": {"mode": "latest-available"},
                "today_utc": "2026-10-12",
                "ticker": "NVDA",
            },
            "schema": {"type": "object"},
        }
    )
    assert out["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    seen = gen.seen[0]
    assert isinstance(seen, dict)
    context = seen.get("context")
    assert isinstance(context, dict)
    assert context.get("ticker") == "NVDA"
    assert "evidence" not in context and "attempts" not in context and "grounding_hint" not in context
    grounded = context.get("grounded_arguments")
    assert isinstance(grounded, dict) and grounded.get("identifier") == "NVDA"


class _FailGenerate:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.seen: list[object] = []

    def __call__(self, **kwargs: object) -> object:
        self.seen.append(kwargs)
        return self.payload


def test_arguments_uses_shared_needle_and_schema_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _FailGenerate(
        {
            "tool": "query_finra",
            "arguments": {"ticker": "NVDA"},
            "confidence": 0.9,
            "reasoning": "r",
        }
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)
    monkeypatch.setattr(
        "app.research.scheduler.build_registry",
        lambda: [{"name": "query_finra", "parameters": {"type": "object"}}],
    )

    def _schema(name: str, reg: list[dict[str, object]]) -> dict[str, object]:
        return {"type": "object"}

    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    out = kw._arguments(
        {
            "id": "a1",
            "op": "arguments",
            "tool": "query_finra",
            "objective": "short interest?",
        }
    )
    assert out == {
        "id": "a1",
        "tool": "query_finra",
        "arguments": {"ticker": "NVDA", "dataset": "consolidatedShortInterest"},
        "confidence": 0.9,
        "reasoning": "r",
    }
    seen = gen.seen[0]
    assert isinstance(seen, dict) and seen["schema"] == {"type": "object"}
    assert isinstance(seen.get("objective"), str) and seen["objective"] == "short interest?"


def test_kernel_worker_stamps_route_assess_and_node() -> None:
    # Clock asks short-circuit in programmatic_route; JEV only selects personas for them.
    jev = _JevRoute("research_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "what time is it?"}, jev=jev)
    assert out == {"id": "r1", "route": "get_current_time", "personas": None}
    assert jev.seen[0][1] == "get_current_time"
    jev2 = _JevRoute("research_required")
    out2 = kw._route({"id": "r2", "op": "route", "prompt": "halp money stuff?"}, jev=jev2)
    assert out2 == {"id": "r2", "route": "research_required", "personas": None}
    assert jev2.seen[0][1] is None
    assert isinstance(jev2.seen[0][0], str) and jev2.seen[0][0].startswith("[Today UTC ")
    assess = _JevAssess("node_resolved")
    kw._assess_entry({"id": "s1", "prompt": "risk?", "tool": "t", "result": {}}, jev=assess)
    assert isinstance(assess.seen[0][0], str) and assess.seen[0][0].startswith("[Today UTC ")
    prompt = kw._intake_reasoner_prompt("rs:test", "objective?", None, "")
    assert "Today is" in prompt and "UTC" in prompt


def test_arguments_mismatch_is_error_never_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kw,
        "_shared_needle_generate",
        lambda: _FailGenerate({"tool": "other", "arguments": {}}),
    )
    out = kw._arguments({"id": "a2", "op": "arguments", "tool": "query_finra"})
    assert out["id"] == "a2" and "error" in out
    assert "error" in kw._arguments({"id": "a3"})


def test_arguments_withhold_seeds_sec_and_sho(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine Needle withhold (returns None) seeds SEC/SHO; unseedable stays error."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA", "Apple": "AAPL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _withhold(**kwargs: object) -> object:
        return None

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _withhold)
    sho = kw._arguments(
        {
            "id": "s1",
            "tool": "get_reg_sho_volume",
            "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
        }
    )
    assert sho["arguments"] == {"ticker": "AAPL", "company_name": "Apple"}
    sec = kw._arguments(
        {
            "id": "s2",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
        }
    )
    assert sec["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}
    none = kw._arguments(
        {
            "id": "s3",
            "tool": "get_reg_sho_volume",
            "objective": "Which filings mention Elon Musk?",
        }
    )
    assert none["id"] == "s3" and "error" in none


def test_arguments_generation_failure_is_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising Needle generate reaches the outer error envelope, never seeded success."""

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("needle boom")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments(
        {
            "id": "s9",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
        }
    )
    assert out["id"] == "s9" and "error" in out
    assert "needle boom" in str(out["error"])
    assert "arguments" not in out


def test_arguments_schema_failure_is_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _FailGenerate(
        {
            "tool": "query_finra",
            "arguments": {"ticker": "NVDA"},
            "confidence": 0.9,
            "reasoning": "r",
        }
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)

    def _bad_schema(name: str, reg: object) -> object:
        raise RuntimeError("schema boom")

    monkeypatch.setattr("app.research.scheduler._schema_for", _bad_schema)
    out = kw._arguments({"id": "a9", "op": "arguments", "tool": "query_finra"})
    assert out["id"] == "a9" and "error" in out
    assert "schema boom" in str(out["error"])


def test_arguments_repair_failure_is_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = _FailGenerate(
        {
            "tool": "query_finra",
            "arguments": {"ticker": "NVDA"},
            "confidence": 0.9,
            "reasoning": "r",
        }
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: gen)
    monkeypatch.setattr(
        "app.research.scheduler.build_registry",
        lambda: [{"name": "query_finra", "parameters": {"type": "object"}}],
    )

    def _bad_repair(*a: object, **k: object) -> object:
        raise RuntimeError("repair boom")

    monkeypatch.setattr("app.research.scheduler._repair_tool_arguments", _bad_repair)
    out = kw._arguments(
        {
            "id": "a10",
            "tool": "query_finra",
            "objective": "q?",
            "schema": {"type": "object"},
        }
    )
    assert out["id"] == "a10" and "error" in out
    assert "repair boom" in str(out["error"])


def test_arguments_repairs_needle_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Org ticker repairs through the shared scheduler path; generated forms stay verbatim for tool rejection."""
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
    assert sec["arguments"] == {"identifier": "NVDA", "forms": ["YYYY-MM-DD"]}
    sho = kw._arguments(
        {
            "id": "r2",
            "tool": "get_reg_sho_volume",
            "objective": "What does FINRA Reg SHO daily short volume show for Apple this week?",
            "schema": {"type": "object"},
        }
    )
    assert sho["arguments"] == {"ticker": "AAPL", "company_name": "Apple"}


class _JevAssess(JevClient):
    def __init__(self, verdict: str | None = None, fail: bool = False) -> None:
        super().__init__()
        self.verdict = verdict
        self.fail = fail
        self.seen: list[tuple[str, str, Mapping[str, JSONValue] | None, Mapping[str, JSONValue] | None]] = []

    @override
    async def assess_entry_tool(
        self,
        prompt: str,
        tool: str,
        arguments: Mapping[str, JSONValue] | None = None,
        result: Mapping[str, JSONValue] | None = None,
    ) -> str:
        if self.fail:
            raise RuntimeError("jev down")
        self.seen.append((prompt, tool, arguments, result))
        return self.verdict or "node_resolved"


def test_assess_entry_returns_verdict() -> None:
    jev = _JevAssess("get_sec_document")
    out = kw._assess_entry(
        {
            "id": "s1",
            "prompt": "risk?",
            "tool": "search_sec_filings",
            "arguments": {},
            "result": {"ok": True},
        },
        jev=jev,
    )
    assert out == {"id": "s1", "verdict": "get_sec_document"}


def test_assess_entry_outage_propagates() -> None:
    with pytest.raises(RuntimeError, match="jev down"):
        kw._assess_entry(
            {"id": "s2", "prompt": "p", "tool": "t", "result": {}},
            jev=_JevAssess(fail=True),
        )


def test_assess_entry_blank_still_research_required() -> None:
    out = kw._assess_entry({"id": "s3", "prompt": "  ", "tool": "t"}, jev=_JevAssess("node_resolved"))
    assert out == {"id": "s3", "verdict": "research_required"}


def _choice_stub(
    picks: dict[str, str], seen: dict[str, object]
) -> Callable[[object, Mapping[str, JSONValue]], Awaitable[dict[str, JSONValue]]]:
    """The transport answers each choice question from picks or its first option."""

    async def _stub(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        seen["questions"] = questions
        assert isinstance(questions, dict)
        answers: dict[str, JSONValue] = {}
        for qid, q in questions.items():
            assert isinstance(q, dict)
            opts = q["criteria"]
            assert isinstance(opts, dict)
            pick = picks.get(qid) or next(iter(opts))
            probs: dict[str, JSONValue] = dict.fromkeys(opts, 0.0)
            if pick in opts:
                probs[pick] = 1.0
            answers[qid] = {
                "type": "choice",
                "choice": pick,
                "probabilities": probs,
                "confidence": 1.0,
            }
        return {"answers": answers}

    return _stub


_ROUTE_REG: list[dict[str, JSONValue]] = [{"name": "get_current_time", "description": "Current time."}]


def _route_entry(
    prompt: str,
    picks: dict[str, str],
    route_hint: str | None = None,
    watchlist: Sequence[str] | None = None,
) -> tuple[dict[str, JSONValue], dict[str, object]]:
    seen: dict[str, object] = {}
    client = JevClient(transport=_choice_stub(picks, seen), data_root=Path("/tmp"))
    return asyncio.run(client.route_entry(prompt, _ROUTE_REG, route_hint=route_hint, watchlist=watchlist)), seen


@pytest.mark.parametrize("prompt", [None, 7, "", " \t\n"])
def test_route_entry_rejects_invalid_prompt(prompt: JSONValue) -> None:
    with pytest.raises(ValueError, match="blank prompt"):
        _route_entry(cast(str, prompt), {"entry": "research_required"})


@pytest.mark.parametrize("hint", ["", " \t\n"])
def test_route_entry_rejects_blank_hint(hint: str) -> None:
    with pytest.raises(ValueError, match="blank route_hint"):
        _route_entry("NVDA?", {"personas": "bearbot"}, route_hint=hint)


@pytest.mark.parametrize(
    ("registry", "error"),
    [
        ([{}], "registry entry needs a name"),
        ([{"name": ""}], "registry entry needs a name"),
        ([{"name": 7}], "registry entry needs a name"),
        ([{"name": "research_required"}], "duplicate tool"),
        (_ROUTE_REG * 2, "duplicate tool"),
    ],
)
def test_route_entry_rejects_defective_registry(registry: list[dict[str, JSONValue]], error: str) -> None:
    client = JevClient(
        transport=_choice_stub({"entry": "research_required"}, {}),
        data_root=Path("/tmp"),
    )
    with pytest.raises(ValueError, match=error):
        asyncio.run(client.route_entry("NVDA?", registry))


def test_route_entry_rejects_empty_installed_registry() -> None:
    client = JevClient(
        transport=_choice_stub({"entry": "research_required"}, {}),
        data_root=Path("/tmp"),
    )
    with (
        mock.patch("app.decision_client._auto_registry", return_value=[]),
        pytest.raises(ValueError, match="empty registry"),
    ):
        asyncio.run(client.route_entry("NVDA?"))


@pytest.mark.parametrize("mode", ["catalog", "full", "whole", "direct"])
def test_route_catalog_modes_use_selector_instead_of_programmatic_hint(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("STOCKBOT_TOOLFLOW", mode)
    client = JevClient(
        transport=_choice_stub({"entry": "reasoning_required", "personas": "bearbot"}, {}),
        data_root=Path("/tmp"),
    )
    monkeypatch.setattr("app.decision_client._auto_registry", lambda: _ROUTE_REG)
    monkeypatch.setattr(kw, "_shared_jev", lambda: client)
    out = kw._route({"id": "catalog-request", "prompt": "what time is it?"})
    assert out == {
        "id": "catalog-request",
        "route": "reasoning_required",
        "personas": ["bearbot"],
    }


@pytest.mark.parametrize(("message", "error"), [("", "route selection failed"), ("x" * 501, "x" * 500)])
def test_route_selector_error_is_nonempty_and_bounded(message: str, error: str) -> None:
    async def _down(state: object, questions: object) -> object:
        raise RuntimeError(message)

    client = JevClient(transport=_down, data_root=Path("/tmp"))
    assert kw._route({"id": "failed-request", "prompt": "NVDA?"}, jev=client) == {
        "id": "failed-request",
        "error": error,
    }


def test_route_entry_defaults_to_no_explicit_personas() -> None:
    out, _ = _route_entry("Build the NVDA thesis", {"entry": "research_required"})
    assert out == {"route": "research_required", "personas": None}


def test_route_entry_explicit_subset_is_canonical() -> None:
    out, _ = _route_entry(
        "Stockbot and Bullbot: NVDA?",
        {"entry": "research_required", "personas": "stockbot_bullbot"},
    )
    assert out == {"route": "research_required", "personas": ["stockbot", "bullbot"]}


@pytest.mark.parametrize("pick", ["bullbot_stockbot", "stockbot_stockbot", "all"])
def test_route_entry_rejects_malformed_persona_choice(pick: str) -> None:
    with pytest.raises(ValueError, match="malformed_decision_answer for personas"):
        _route_entry("NVDA?", {"entry": "research_required", "personas": pick})


def test_route_entry_no_personas_remaining_raises() -> None:
    with pytest.raises(ValueError, match="no personas remain"):
        _route_entry("NVDA?", {"entry": "research_required", "personas": "no_personas_remaining"})


@settings(max_examples=50, derandomize=True)
@given(st.text(max_size=40))
def test_route_entry_rejects_arbitrary_invalid_persona_label(label: str) -> None:
    assume(label)
    assume(label not in _PERSONA_OPTIONS)
    with pytest.raises(ValueError, match="malformed_decision_answer for personas"):
        _route_entry("NVDA?", {"entry": "research_required", "personas": label})


@pytest.mark.parametrize(
    "decisions",
    [
        {"entry": {"choice": "research_required"}, "personas": {"choice": "everyone"}},
        {"entry": {"choice": "research_required"}},
    ],
)
def test_route_entry_rejects_sidecar_persona_outside_options(
    monkeypatch: pytest.MonkeyPatch, decisions: dict[str, JSONValue]
) -> None:
    client = JevClient(data_root=Path("/tmp"))

    def _sidecar(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return {"id": payload["id"], "decisions": decisions, "raw": {}}

    monkeypatch.setattr(client, "_sidecar_roundtrip", _sidecar)
    with pytest.raises(ValueError, match="persona choice"):
        asyncio.run(client.route_entry("NVDA?", _ROUTE_REG))


@pytest.mark.parametrize("entry", [None, {}, {"choice": "unknown_tool"}, {"choice": []}])
def test_route_entry_rejects_missing_or_invalid_sidecar_route(
    monkeypatch: pytest.MonkeyPatch, entry: JSONValue
) -> None:
    client = JevClient(data_root=Path("/tmp"))
    decisions: dict[str, JSONValue] = {"personas": {"choice": "bearbot"}}
    if entry is not None:
        decisions["entry"] = entry

    def _sidecar(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return {"id": payload["id"], "decisions": decisions, "raw": {}}

    monkeypatch.setattr(client, "_sidecar_roundtrip", _sidecar)
    with pytest.raises(ValueError, match="winner .* not in options"):
        asyncio.run(client.route_entry("NVDA?", _ROUTE_REG))


def test_route_entry_hint_keeps_route_and_asks_only_personas() -> None:
    out, seen = _route_entry(
        "what time is it?",
        {"personas": "bearbot_bullbot"},
        route_hint="get_current_time",
    )
    assert out == {"route": "get_current_time", "personas": ["bearbot", "bullbot"]}
    questions = seen["questions"]
    assert isinstance(questions, dict) and list(questions) == ["personas"]


def test_route_entry_selector_unavailable_raises() -> None:
    async def _down(state: object, questions: object) -> object:
        raise RuntimeError("jev down")

    client = JevClient(transport=_down, data_root=Path("/tmp"))
    with pytest.raises(RuntimeError, match="jev down"):
        asyncio.run(client.route_entry("NVDA?", _ROUTE_REG))


def test_entry_prompts_carry_temporal_decoding() -> None:
    out, seen = _route_entry("hello", {"entry": "reasoning_required"})
    assert out == {"route": "reasoning_required", "personas": None}
    questions = seen["questions"]
    assert isinstance(questions, dict)
    entry_text = str(questions["entry"]["instructions"])
    assert "Today UTC is" in entry_text and "decode relative dates before choosing" in entry_text

    class _NoPersistJev(JevClient):
        @override
        def _persist(self, **kwargs: object) -> None:
            return None

    seen.clear()
    client = _NoPersistJev(
        transport=_choice_stub({"assess": "reasoning_required"}, seen),
        data_root=Path("/tmp"),
    )
    verdict = asyncio.run(client.assess_entry_tool("risk?", "search_sec_filings", {}, {"ok": True}))
    assert verdict == "reasoning_required"
    questions = seen["questions"]
    assert isinstance(questions, dict)
    assess_text = str(questions["assess"]["instructions"])
    assert "Today UTC is" in assess_text and "decode relative dates before choosing" in assess_text


def test_run_scheduler_deadline_honors_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_run passes run_deadline to the scheduler; setup gets _SETUP_S of the wall."""
    import time as _time

    seen: dict[str, object] = {}
    closed: list[str] = []

    def _fake_graph(prompt: str, as_of: object = None, **k: object) -> str:
        deadline = k.get("setup_deadline")
        assert isinstance(deadline, float)
        seen["setup_deadline"] = deadline
        _time.sleep(0.05)
        return "s1"

    async def _fake_sched(sid: str, **hooks: object) -> dict[str, object]:
        seen["deadline_at"] = hooks.get("deadline_at")
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    def _close(sid: str) -> None:
        closed.append(sid)

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _close)
    _empty_repo(monkeypatch)
    t0 = _time.perf_counter()
    out = kw._run({"id": "r1", "op": "run", "prompt": "hello?"}, jev=JevClient())
    assert out["id"] == "r1"
    assert closed == ["s1"]
    assert isinstance(seen["setup_deadline"], float)
    assert isinstance(seen["deadline_at"], float)
    assert abs(seen["setup_deadline"] - (t0 + kw._SETUP_S)) < 5.0
    assert abs(seen["deadline_at"] - (t0 + kw._RUN_WALL_S - kw._FINALIZE_S)) < 5.0


def test_run_watchlist_replaces_ticker_scope_and_returns_persisted_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A watchlist request drops the session ticker; coverage rows come only from the persisted intake_digest."""
    seen: list[dict[str, object]] = []
    rows: list[JSONValue] = [
        {"symbol": "NVDA", "source": "sec_filings", "tool": "list_sec_filings", "status": "ok", "evidence_id": "ev:1"},
        {"symbol": "QQQ", "source": "finra_threshold_list", "tool": "get_threshold_securities", "status": "timeout"},
    ]

    def _fake_graph(prompt: str, as_of: object = None, **k: object) -> str:
        seen.append({"prompt": prompt, "ticker": k.get("ticker"), "watchlist": k.get("watchlist")})
        return "s1"

    async def _fake_sched(sid: str, **hooks: object) -> dict[str, object]:
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    def _decisions(self: ResearchRepository, sid: str) -> list[DecisionRecord]:
        digest = DecisionRecord(
            decision_id="dec-digest",
            session_id=sid,
            node_id=None,
            job_id=None,
            decision_type="intake_digest",
            candidates={},
            probabilities={},
            selected={"stats": {"watchlist": rows}},
        )
        # A non-digest decision carrying the same key never contributes coverage.
        other = DecisionRecord(
            decision_id="dec-other",
            session_id=sid,
            node_id=None,
            job_id=None,
            decision_type="route",
            candidates={},
            probabilities={},
            selected={"stats": {"watchlist": [{"symbol": "FAKE"}]}},
        )
        return [digest, other]

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    _empty_repo(monkeypatch)
    monkeypatch.setattr(ResearchRepository, "list_decisions", _decisions)
    prompt = "Latest filings for my watchlist?"
    out = kw._run(
        {"id": "rw", "prompt": prompt, "ticker": "aapl", "watchlist": ["nvda", "QQQ", "nvda"]}, jev=JevClient()
    )
    assert seen == [{"prompt": prompt, "ticker": None, "watchlist": ["NVDA", "QQQ"]}]
    assert out["watchlist"] == rows

    seen.clear()
    _empty_repo(monkeypatch)
    out = kw._run({"id": "rt", "prompt": "NVDA risk factors?", "ticker": "nvda"}, jev=JevClient())
    assert seen == [{"prompt": "NVDA risk factors?", "ticker": "NVDA", "watchlist": None}]
    assert "watchlist" not in out


@pytest.mark.parametrize("bad", ["NVDA", [], ["NVDA", "SEC"], ["NVDA", 7], ["  "], [f"T{i}" for i in range(26)]])
def test_run_malformed_watchlist_is_invalid_params_before_setup(
    monkeypatch: pytest.MonkeyPatch, bad: JSONValue
) -> None:
    def _boom(*a: object, **k: object) -> str:
        raise AssertionError("setup must not run for a malformed watchlist")

    monkeypatch.setattr(kw, "run_graph_prompt", _boom)
    out = kw._run({"id": "rb", "prompt": "my watchlist?", "watchlist": bad}, jev=JevClient())
    terminal = out["terminal"]
    assert isinstance(terminal, dict) and terminal["category"] == "invalid_params"
    assert "watchlist" in str(terminal["message"])
    assert kw._watchlist({"id": "wb", "watchlist": bad}) == {"id": "wb", "error": terminal["message"]}


def test_run_scheduler_deadline_honors_absolute_deadline_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deadlineAt (epoch ms) shrinks the run; bool/str values fall back to _RUN_WALL_S."""
    import time as _time

    seen: dict[str, object] = {}
    closed: list[str] = []

    def _fake_graph(prompt: str, as_of: object = None, **k: object) -> str:
        return "s1"

    async def _fake_sched(sid: str, **hooks: object) -> dict[str, object]:
        seen["deadline_at"] = hooks.get("deadline_at")
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    def _close(sid: str) -> None:
        closed.append(sid)

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _close)
    _empty_repo(monkeypatch)
    t0 = _time.perf_counter()
    kw._run(
        {
            "id": "r2",
            "op": "run",
            "prompt": "hello?",
            "deadlineAt": _time.time() * 1000 + 60_000,
        },
        jev=JevClient(),
    )
    assert isinstance(seen["deadline_at"], float)
    assert abs(float(seen["deadline_at"]) - (t0 + 60 - kw._FINALIZE_S)) < 5.0
    assert closed == ["s1"]
    for bad in (True, "soon"):
        kw._run(
            {"id": "r3", "op": "run", "prompt": "hello?", "deadlineAt": bad},
            jev=JevClient(),
        )
        assert abs(float(seen["deadline_at"]) - (t0 + kw._RUN_WALL_S - kw._FINALIZE_S)) < 5.0


def test_run_repository_read_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_graph(prompt: str, as_of: object = None, **k: object) -> str:
        return "s1"

    async def _fake_sched(sid: str, **hooks: object) -> dict[str, object]:
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    def _boom(self: ResearchRepository, sid: str) -> list[dict[str, JSONValue]]:
        raise RuntimeError("store down")

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    monkeypatch.setattr(ResearchRepository, "list_evidence", _boom)
    with pytest.raises(RuntimeError, match="store down"):
        kw._run({"id": "r9", "op": "run", "prompt": "hello?"}, jev=JevClient())


def _kernel_with_decision_log(recorded: list[dict[str, object]]) -> sched._Kernel:
    class _K(sched._Kernel):
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
            recorded.append({"dtype": decision_type, "selected": selected})
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

    return _K()


def test_round2_skipped_for_corrected_query_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 2 runs only for new tickers; a corrected query alone never earns it."""
    import asyncio

    rounds = {"n": 0}
    recorded: list[dict[str, object]] = []

    async def _one_round(*a: object, **k: object) -> tuple[list[object], list[object], dict[str, object]]:
        rounds["n"] += 1
        return ([], [], {"calls": 1, "admitted": 0})

    def _decompose(*a: object, **k: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [
                {
                    "id": "s-q1",
                    "objectiveId": "s",
                    "question": "q?",
                    "dependsOn": [],
                    "whyItMatters": "w",
                }
            ],
            {"corrected_query": "fixed query"},
        )

    kernel = _kernel_with_decision_log(recorded)
    monkeypatch.setattr(kw, "_intake_round", _one_round)
    monkeypatch.setattr(kw, "_reasoner_decompose_with_retry", _decompose)
    out = asyncio.run(kw._graph_intake("q?", None, "s1", {}, [], JevClient(), kernel, None))
    assert len(out) == 1 and rounds["n"] == 1
    assert recorded and recorded[0].get("dtype") == "intake_digest"


def test_jev_admit_timeout_propagates() -> None:
    """A hanging JEV decide raises TimeoutError before the setup deadline."""
    import asyncio
    import time as _time

    class _Hang(JevClient):
        @override
        async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
            await asyncio.sleep(30.0)
            raise AssertionError("must time out")

    t0 = _time.perf_counter()
    with pytest.raises(TimeoutError):
        kw._jev_admit(
            "s",
            "Exact user objective?",
            _props(),
            jev=_Hang(),
            setup_deadline=t0 + 0.2,
        )
    assert _time.perf_counter() - t0 < 5.0


def test_decompose_retry_budget_propagates_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient decompose failure with <15s left propagates, no synthesized proposal."""
    import time as _time

    calls = {"n": 0}

    def _boom(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _boom)
    with pytest.raises(RuntimeError, match="connection reset by peer"):
        kw._reasoner_decompose_with_retry("q?", None, "s", "", None, None, _time.perf_counter() + 5.0)
    assert calls["n"] == 1


def test_notify_progress_failure_propagates_with_setup_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listener errors propagate directly and setup reports the real provider_error boundary."""
    failure = RuntimeError("listener down")

    def _bad(stage: str, detail: dict[str, object]) -> None:
        raise failure

    with pytest.raises(RuntimeError, match="listener down") as raised:
        kw._notify(_bad, "intake_start", {"calls": 1})
    assert raised.value is failure

    def _fake_graph(
        prompt: str,
        as_of: object = None,
        *,
        progress: kw.ProgressFn | None = None,
        **k: object,
    ) -> str:
        kw._notify(progress, "intake_start", {"calls": 1})
        return "s1"

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    out = kw._run({"id": "rp", "op": "run", "prompt": "hello?"}, jev=JevClient(), progress=_bad)
    assert out["id"] == "rp"
    terminal = out["terminal"]
    assert isinstance(terminal, dict)
    assert terminal["category"] == "provider_error"
    assert out["error"] == "session setup failed: listener down"
    assert terminal["message"] == out["error"]
    assert out["failures"] == {"provider_error": 1}
    assert out["escalated"] is True
    assert out["evidence"] == []


def test_normalize_proposals_malformed_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="proposal must be an object"):
        kw._normalize_proposals([object()], "s")
    with pytest.raises(TypeError, match="empty proposals"):
        kw._normalize_proposals("not-a-list", "s")
    with pytest.raises(TypeError, match="empty proposals"):
        kw._normalize_proposals([], "s")


def test_normalize_proposals_ignores_non_list_dependencies() -> None:
    proposal = {
        "id": "s-q1",
        "objectiveId": "s",
        "question": "q?",
        "dependsOn": "s-q2",
        "whyItMatters": "w",
    }
    assert kw._normalize_proposals([proposal], "s") == [{**proposal, "dependsOn": []}]


def test_create_nodes_topological_cycle_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cyclic dependsOn rejects; no nodes persist for the cyclic batch."""
    import tempfile

    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    sid = service.create_research("q?", "q?")
    cyclic: list[dict[str, object]] = [
        {
            "id": "s-q1",
            "objectiveId": "s",
            "question": "q1?",
            "dependsOn": ["s-q2"],
            "whyItMatters": "w",
        },
        {
            "id": "s-q2",
            "objectiveId": "s",
            "question": "q2?",
            "dependsOn": ["s-q1"],
            "whyItMatters": "w",
        },
    ]
    with pytest.raises(ValueError, match="cyclic"):
        kw._create_nodes_topological(sid, "q?", cyclic)
    assert ResearchRepository().list_nodes(sid) == []


def test_startup_jev_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadJev(JevClient):
        @override
        def start(self) -> None:
            raise RuntimeError("jev start down")

    monkeypatch.setattr(kw, "_shared_jev", lambda: _BadJev())
    monkeypatch.setattr(kw, "_prewarm", lambda: None)
    with pytest.raises(RuntimeError, match="jev start down"):
        kw._startup()


def test_startup_needle_failure_stays_live(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    jev = JevClient()

    def _noop_start(self: JevClient) -> None:
        return None

    monkeypatch.setattr(kw, "_shared_jev", lambda: jev)
    monkeypatch.setattr(JevClient, "start", _noop_start)
    monkeypatch.setattr(kw, "_prewarm", lambda: None)
    real_import = importlib.import_module

    def _bad_needle(name: str, package: str | None = None) -> ModuleType:
        if name == "app.needle_client":
            raise RuntimeError("needle down")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _bad_needle)
    assert kw._startup() is jev


@pytest.mark.parametrize("stage", ["repository", "list", "cancel"])
def test_close_bootstrap_job_failure_logs(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.research import service
    from app.research.models import Job

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    failure = RuntimeError(f"{stage} down")
    job = Job(
        job_id="j1",
        session_id="rs:x",
        wave_id=1,
        parent_job_id=None,
        job_type="source_agent",
        owner="k",
    )

    def _one_job(self: ResearchRepository, sid: str) -> list[Job]:
        return [job]

    if stage == "repository":
        target = "app.research.repository.ResearchRepository"
    elif stage == "list":
        target = "app.research.repository.ResearchRepository.list_jobs"
    else:
        monkeypatch.setattr(ResearchRepository, "list_jobs", _one_job)
        target = "app.research.service.cancel_job"
    with mock.patch(target, side_effect=failure) as cleanup:
        kw._close_bootstrap_job("rs:x")
    assert cleanup.call_count == 1
    records = [record for record in caplog.records if "bootstrap cleanup failed" in record.message]
    assert len(records) == 1
    assert "sid=rs:x" in records[0].message
    assert records[0].exc_info is not None and records[0].exc_info[1] is failure

    if stage == "cancel":
        done_source = replace(job, job_id="j0", status="completed")
        open_other = replace(job, job_id="j2", job_type="stockbot")
        with (
            mock.patch.object(
                ResearchRepository,
                "list_jobs",
                return_value=[open_other, done_source, job],
            ),
            mock.patch.object(service, "cancel_job") as cancel,
        ):
            kw._close_bootstrap_job("rs:x")
        cancel.assert_called_once_with("j1")


@pytest.mark.parametrize("scheduler_error", [False, True])
def test_run_bootstrap_cleanup_failure_preserves_outcome(
    scheduler_error: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The worker keeps the scheduler outcome when bootstrap cleanup fails."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    failure = RuntimeError("jobs down")
    jev = JevClient()

    def _fake_graph(prompt: str, as_of: object = None, **k: object) -> str:
        return "s1"

    async def _fake_sched(sid: str, **hooks: object) -> dict[str, object]:
        if scheduler_error:
            raise RuntimeError("scheduler down")
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    def _boom_jobs(self: ResearchRepository, sid: str) -> list[object]:
        raise failure

    def _needle_stub(**kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(kw, "run_graph_prompt", _fake_graph)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _needle_stub)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(ResearchRepository, "list_jobs", _boom_jobs)
    _empty_repo(monkeypatch)
    request: dict[str, JSONValue] = {"id": "rc", "op": "run", "prompt": "hello?"}
    out = kw._run(request, jev=jev)
    assert out["id"] == "rc"
    if scheduler_error:
        assert out["error"] == "kernel run failed: scheduler down"
        assert out["terminal"] == {
            "category": "provider_error",
            "message": out["error"],
        }
        assert out["failures"] == {"provider_error": 1}
    else:
        assert out["sessionId"] == "s1"
        assert out["objective"] == "hello?"
        assert out["failures"] == {}
        assert out["escalated"] is False
        assert "error" not in out and "terminal" not in out
    cleanup_records = [record for record in caplog.records if "bootstrap cleanup failed" in record.message]
    assert len(cleanup_records) == 1
    assert cleanup_records[0].exc_info is not None and cleanup_records[0].exc_info[1] is failure

    stdout = StringIO()
    monkeypatch.setattr(kw, "_startup", lambda: jev)
    monkeypatch.setattr(kw, "_shutdown", lambda: None)
    monkeypatch.setattr(kw.sys, "stdin", StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr(kw.sys, "stdout", stdout)
    with mock.patch.object(kw.signal, "signal"):
        kw.main()

    responses: list[object] = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses == [{"type": "ready"}, out]
    assert out["evidence"] == []


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_intake_closes_bootstrap_without_replacing_setup_error(
    cleanup_fails: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))

    def _no_ticker(objective: str) -> None:
        return None

    monkeypatch.setattr(sched, "build_registry", list)
    monkeypatch.setattr(sched, "_objective_subject_ticker", _no_ticker)
    repo = ResearchRepository()
    failure = RuntimeError("intake down")
    cleanup_failure = RuntimeError("cancel down")
    sessions: list[str] = []
    cancellations: list[str] = []
    original_cancel = service.cancel_job

    def _session(stage: str, detail: dict[str, object]) -> None:
        if stage == "session":
            sid = detail["session_id"]
            assert isinstance(sid, str)
            sessions.append(sid)
            assert [job.status for job in repo.list_jobs(sid)] == ["running"]

    async def _failed_intake(*args: object, **kwargs: object) -> object:
        raise failure

    def _cancel(job_id: str) -> dict[str, JSONValue]:
        cancellations.append(job_id)
        if cleanup_fails:
            raise cleanup_failure
        return original_cancel(job_id)

    monkeypatch.setattr(kw, "_intake_round", _failed_intake)
    monkeypatch.setattr(service, "cancel_job", _cancel)
    with pytest.raises(RuntimeError, match="intake down") as raised:
        kw.run_graph_prompt("q?", jev=JevClient(), progress=_session)
    assert raised.value is failure
    assert len(sessions) == 1
    sid = sessions[0]
    session = repo.get_session(sid)
    assert session.query == "q?"
    jobs = repo.list_jobs(sid)
    assert len(jobs) == 1
    assert cancellations == [jobs[0].job_id]
    assert jobs[0].status == ("running" if cleanup_fails else "cancelled")
    if not cleanup_fails:
        assert not [job for job in jobs if job.status in ("queued", "running")]
    cleanup_records = [record for record in caplog.records if "bootstrap cleanup failed" in record.message]
    assert len(cleanup_records) == int(cleanup_fails)
    if cleanup_fails:
        assert cleanup_records[0].exc_info is not None
        assert cleanup_records[0].exc_info[1] is cleanup_failure


def test_graph_intake_reasoner_crash_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    recorded: list[dict[str, object]] = []

    async def _one_round(*a: object, **k: object) -> tuple[list[object], list[object], dict[str, object]]:
        return ([], [{"tool": "list_sec_filings"}], {"calls": 1, "admitted": 0})

    def _crash(*a: object, **k: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        raise RuntimeError("reasoner crash")

    monkeypatch.setattr(kw, "_intake_round", _one_round)
    monkeypatch.setattr(kw, "_reasoner_decompose_with_retry", _crash)
    with pytest.raises(RuntimeError, match="reasoner crash"):
        asyncio.run(
            kw._graph_intake(
                "q?",
                None,
                "s1",
                {},
                [],
                JevClient(),
                _kernel_with_decision_log(recorded),
                None,
            )
        )


def test_graph_intake_round_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    recorded: list[dict[str, object]] = []

    async def _boom_round(*a: object, **k: object) -> tuple[list[object], list[object], dict[str, object]]:
        raise RuntimeError("intake round down")

    monkeypatch.setattr(kw, "_intake_round", _boom_round)
    with pytest.raises(RuntimeError, match="intake round down"):
        asyncio.run(
            kw._graph_intake(
                "q?",
                None,
                "s1",
                {},
                [],
                JevClient(),
                _kernel_with_decision_log(recorded),
                None,
            )
        )


def test_followup_reuses_session_skips_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """sessionId+gap creates one node in the live session; no new session, no intake."""
    import asyncio
    import tempfile

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

    def _count(
        run_sid: str,
        question: str,
        why_it_matters: str,
        depends_on: Sequence[str] | None = None,
        *,
        repo: ResearchRepository | Path | str | None = None,
    ) -> ResearchNode:
        out = orig_create(run_sid, question, why_it_matters, depends_on, repo=repo)
        created.append(out.node_id)
        return out

    monkeypatch.setattr(kw, "run_graph_prompt", _boom_graph)
    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr("app.research.service.create_node", _count)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r2", "prompt": "q?", "sessionId": sid, "gap": "need doc X"}))
    assert out["sessionId"] == sid
    assert out["unresolved"] == []
    assert len(created) == 1
    nodes = out["nodes"]
    assert isinstance(nodes, list) and len(nodes) == 2
    kw.run_graph_prompt = orig_graph


def test_followup_unknown_session_is_invalid_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown sessionId returns invalid_params, never a new session."""
    import tempfile

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    out = kw._run({"id": "r9", "prompt": "q?", "sessionId": "rs:nope", "gap": "need doc X"})
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal.get("category") == "invalid_params"


def test_followup_blocks_leftover_proposed_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leftover pass-1 proposed node is blocked; the scheduler sees only the gap node."""
    import asyncio
    import tempfile

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

    def _count(
        run_sid: str,
        question: str,
        why_it_matters: str,
        depends_on: Sequence[str] | None = None,
        *,
        repo: ResearchRepository | Path | str | None = None,
    ) -> ResearchNode:
        out = orig_create(run_sid, question, why_it_matters, depends_on, repo=repo)
        if out.node_id:
            gap_ids.append(out.node_id)
        return out

    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr("app.research.service.create_node", _count)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r3", "prompt": "q?", "sessionId": sid, "gap": "need doc X"}))
    assert out["sessionId"] == sid
    assert len(gap_ids) == 1
    assert ResearchRepository().get_node(leftover.node_id).status == "blocked"


def test_followup_links_only_explicit_known_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass-2 calls cite their own known ids (pass-1 duplicate winner, repeats); others get no evidenceId."""
    import asyncio
    import tempfile

    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    sid = service.create_research("q?", "q?")
    service.create_node(sid, "q1?", "w")
    store = ResearchRepository()
    store.save_evidence({"evidence_id": "ev-pass1", "session_id": sid, "content": "pass-1 fact"})

    def _refs(record: Mapping[str, object]) -> list[dict[str, str]]:
        assert record.get("session_id") == sid
        if record.get("evidence_id") == "ev-pass1":
            return [
                {
                    "id": "evidence:ev-pass1",
                    "title": "Prior capture",
                    "url": "https://www.sec.gov/prior",
                }
            ]
        return []

    monkeypatch.setattr("app.research.sources.source_references", _refs)

    async def _fake_sched(run_sid: str, **hooks: object) -> dict[str, object]:
        assert run_sid == sid
        for n in service.ready_nodes(run_sid):
            service.resolve_node(run_sid, n.node_id)
        ResearchRepository().save_evidence({"evidence_id": "ev-pass2", "session_id": run_sid, "content": "pass-2 fact"})
        attempts: list[dict[str, JSONValue]] = [
            {"tool": "search_sec_filings", "arguments": {}},
            {"tool": "get_sec_filing", "arguments": {}, "evidence_id": "ev-pass1"},
            {"tool": "get_sec_document", "arguments": {}, "evidence_id": "ev-pass2"},
            {"tool": "get_sec_document", "arguments": {}, "evidence_id": "ev-pass2"},
            {"tool": "search_web", "arguments": {}, "evidence_id": "ev-ghost"},
            {
                "tool": "query_finra",
                "arguments": {},
                "error": "boom",
                "evidence_id": "ev-pass2",
            },
            {
                "tool": "get_sec_filing",
                "arguments": {},
                "evidence_id": None,
                "evidenceId": "ev-pass1",
            },
            {
                "tool": "get_sec_filing",
                "arguments": {},
                "evidence_id": "",
                "evidenceId": "ev-pass2",
            },
        ]
        return {
            "status": "complete",
            "nodes": [{"node_id": "n", "attempts": attempts}],
            "incomplete_guard": False,
        }

    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r4", "prompt": "q?", "sessionId": sid, "gap": "need doc Y"}))
    calls = out["toolCalls"]
    assert isinstance(calls, list)
    assert [c.get("evidenceId") for c in calls if isinstance(c, dict)] == [
        None,
        "ev-pass1",
        "ev-pass2",
        "ev-pass2",
        None,
        None,
        "ev-pass1",
        "ev-pass2",
    ]
    evidence = out["evidence"]
    assert isinstance(evidence, list)
    ids = [e["id"] for e in evidence if isinstance(e, dict)]
    assert "ev-pass1" in ids and "ev-pass2" in ids
    by_id = {e["id"]: e for e in evidence if isinstance(e, dict)}
    assert by_id["ev-pass1"]["sources"] == [
        {
            "id": "evidence:ev-pass1",
            "title": "Prior capture",
            "url": "https://www.sec.gov/prior",
        }
    ]
    assert "sources" not in by_id["ev-pass2"]


def test_wire_evidence_sends_full_sec_document_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored SEC document text reaches personas whole; other rows and textless rows keep the 2000 cap."""
    import asyncio
    import tempfile
    from pathlib import Path

    from app.research import service
    from app.research.repository import ResearchRepository

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    sid = service.create_research("q?", "q?")
    service.create_node(sid, "q1?", "w")
    fact = "Total revenue $46,743 million"
    window = "Item 2. MD&A | 25\n" * 112 + fact + "x" * 9000  # fact starts at char 2016 of an ~11k window
    assert window.index(fact) > 2000
    sec_prov = {
        "kind": "sec_source",
        "accession_no": "0001045810-26-000075",
        "document_name": "nvda.htm",
    }
    store = ResearchRepository()
    store.save_evidence(
        {
            "evidence_id": "ev-sec",
            "session_id": sid,
            "content": window,
            "provenance": sec_prov,
        }
    )
    store.save_evidence(
        {
            "evidence_id": "ev-web",
            "session_id": sid,
            "content": window,
            "provenance": {"kind": "web_source"},
        }
    )
    store.save_evidence(
        {
            "evidence_id": "ev-rec",
            "session_id": sid,
            "content": window,
            "provenance": {"kind": "sec_record"},
        }
    )
    store.save_evidence(
        {
            "evidence_id": "ev-sec-blank",
            "session_id": sid,
            "content": "  ",
            "provenance": {**sec_prov, "passage": window},
        }
    )

    async def _fake_sched(run_sid: str, **hooks: object) -> dict[str, object]:
        for n in service.ready_nodes(run_sid):
            service.resolve_node(run_sid, n.node_id)
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr("app.research.scheduler.run", _fake_sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    out = asyncio.run(asyncio.to_thread(kw._run, {"id": "r6", "prompt": "q?", "sessionId": sid, "gap": "revenue"}))
    evidence = out["evidence"]
    assert isinstance(evidence, list)
    rows = {e["id"]: e["content"] for e in evidence if isinstance(e, dict)}
    assert rows["ev-sec"] == window
    assert rows["ev-web"] == window[:2000] and rows["ev-rec"] == window[:2000]
    # No stored text: the row never gets the full SEC projection (clipped record JSON, no fact).
    blank = rows["ev-sec-blank"]
    assert isinstance(blank, str) and len(blank) == 2000 and fact not in blank


def test_setup_keyerror_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyError inside run_graph_prompt reports provider_error, not invalid_params."""
    import tempfile

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))

    def _boom(*a: object, **k: object) -> str:
        raise KeyError("missing-key")

    monkeypatch.setattr(kw, "run_graph_prompt", _boom)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    out = kw._run({"id": "r5", "op": "run", "prompt": "hello?"})
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal.get("category") == "provider_error"
