"""Fail-closed guard contracts: failures propagate, envelopes preserved."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import override
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


@pytest.mark.parametrize("bad", ["NVDA", [], [""]])
def test_route_entry_rejects_blank_watchlist(bad: Sequence[str]) -> None:
    with pytest.raises(ValueError, match="blank watchlist"):
        _route_entry("my watchlist?", {"entry": "research_required"}, watchlist=bad)


@pytest.mark.parametrize("bad", [[7], [object()]])
def test_watchlist_scope_rejects_non_string_symbols(bad: list[object]) -> None:
    """A non-string symbol raises the blank-watchlist error before route_entry asks JEV."""
    from app.decision_client import _watchlist_scope

    with pytest.raises(ValueError, match="route_entry: blank watchlist"):
        _watchlist_scope(bad)


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


@pytest.mark.parametrize("prompt", ["", " \t\n"])
def test_route_entry_rejects_invalid_prompt(prompt: str) -> None:
    with pytest.raises(ValueError, match="blank prompt"):
        _route_entry(prompt, {"entry": "research_required"})


@pytest.mark.parametrize("prompt", [None, 7])
def test_route_text_rejects_non_text_prompt(prompt: object) -> None:
    from app.decision_client import _route_text

    with pytest.raises(ValueError, match="blank prompt"):
        _route_text(prompt)


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
    assert out["id"] == "r5"
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal.get("category") == "provider_error"


def _graph_session(prompt: str, as_of: object = None, **rest: object) -> str:
    return "s1"


def _run_with(monkeypatch: pytest.MonkeyPatch, sched_result: object) -> dict[str, JSONValue]:
    """Run one request through a stubbed session + scheduler; the caller owns the repository stubs."""

    async def _sched(sid: str, **hooks: object) -> object:
        return sched_result

    monkeypatch.setattr(kw, "run_graph_prompt", _graph_session)
    monkeypatch.setattr("app.research.scheduler.run", _sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    return kw._run({"id": "r1", "prompt": "hello?"}, jev=JevClient())


@pytest.mark.parametrize(
    ("flow", "selected"), [("catalog", True), ("full", False), ("whole", False), ("direct", False)]
)
def test_run_selects_round_by_toolflow(monkeypatch: pytest.MonkeyPatch, flow: str, selected: bool) -> None:
    """catalog keeps the two-step selector; full/whole/direct keep the whole-registry escape hatch."""
    from app.research.tool_catalogs import catalog_select_round

    monkeypatch.setenv("STOCKBOT_TOOLFLOW", flow)
    chosen = kw._run_select_round()
    assert (chosen is catalog_select_round) is selected


def test_run_selects_programmatic_round_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The programmatic fast route is the default round selector."""
    from app.research.programmatic_router import programmatic_select_round

    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    assert kw._run_select_round() is programmatic_select_round


@pytest.mark.parametrize(("flow", "passed"), [("catalog", True), ("full", False)])
def test_run_passes_select_round_only_when_selected(monkeypatch: pytest.MonkeyPatch, flow: str, passed: bool) -> None:
    """The scheduler receives select_round only when the flow selected one."""
    seen: dict[str, object] = {}

    async def _sched(sid: str, **hooks: object) -> dict[str, object]:
        seen.update(hooks)
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setenv("STOCKBOT_TOOLFLOW", flow)
    monkeypatch.setattr(kw, "run_graph_prompt", _graph_session)
    monkeypatch.setattr("app.research.scheduler.run", _sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    _empty_repo(monkeypatch)
    out = kw._run({"id": "rf", "prompt": "hello?"}, jev=JevClient())
    assert out["sessionId"] == "s1"
    assert ("select_round" in seen) is passed


def test_run_malformed_scheduler_result_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-mapping scheduler result reports provider_error instead of being read as nodes."""
    out = _run_with(monkeypatch, ["not", "a", "mapping"])
    assert out["id"] == "r1"
    assert out["error"] == "kernel run failed: malformed result"
    assert out["terminal"] == {"category": "provider_error", "message": out["error"]}
    assert out["failures"] == {"provider_error": 1}


def test_run_failed_status_keeps_scheduler_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed scheduler result reports its own error text as provider_error."""
    out = _run_with(monkeypatch, {"status": "failed", "error": "boom"})
    assert out["id"] == "r1"
    assert out["error"] == "kernel failed: boom"
    assert out["failures"] == {"provider_error": 1}


def test_run_projects_nodes_calls_failures_and_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attempts project to tool rows, failures count once per category, and an unresolved node escalates."""
    evidence_rows: list[dict[str, JSONValue]] = [
        {"evidence_id": "ev-1", "content": "stored fact", "provenance": {"kind": "web_source"}},
        {"evidence_id": "ev-blank", "content": "   "},
    ]
    node_rows = [
        ResearchNode(
            node_id="n1",
            session_id="s1",
            question="q1?",
            why_it_matters="w",
            depends_on=("n0",),
            status="gathering",
        ),
        ResearchNode(node_id="", session_id="s1", question="q?", why_it_matters="w"),
        ResearchNode(node_id="n2", session_id="s1", question="q2?", why_it_matters="w", status="resolved"),
    ]
    decision_rows = [
        DecisionRecord(
            decision_id="d1",
            session_id="s1",
            node_id=None,
            job_id=None,
            decision_type="intake_digest",
            candidates={},
            probabilities={},
            selected={"stats": {"watchlist": [{"symbol": "AAPL"}]}},
        ),
        DecisionRecord(
            decision_id="d2",
            session_id="s1",
            node_id=None,
            job_id=None,
            decision_type="tool_select",
            candidates={},
            probabilities={},
            selected={"stats": {"watchlist": [{"symbol": "MSFT"}]}},
        ),
    ]
    sched_result: dict[str, object] = {
        "status": "complete",
        "incomplete_guard": False,
        "nodes": [
            {
                "node_id": "n1",
                "incomplete_guard": True,
                "attempts": [
                    {
                        "tool": "get_sec_document",
                        "arguments": {"accession_no": "a"},
                        "confidence": 0.5,
                        "evidence_id": "ev-1",
                    },
                    {"tool": "search_web", "arguments": "not-a-mapping", "confidence": True, "evidenceId": "ev-ghost"},
                    {"tool": 7, "arguments": {}, "error": "tool blew up"},
                    {"tool": "search_sec_filings", "arguments": {}, "error": "reselect", "category": "reselect"},
                ],
            },
            "not-a-mapping",
            {"attempts": [{"tool": "query_finra", "arguments": {}, "error": {"code": 500}}]},
        ],
    }

    def _refs(record: Mapping[str, object]) -> list[dict[str, str]]:
        if record.get("evidence_id") == "ev-1":
            return [{"id": "evidence:ev-1", "title": "Capture", "url": "https://example.test/a"}]
        return []

    def _list_evidence(self: ResearchRepository, sid: str) -> list[dict[str, JSONValue]]:
        return list(evidence_rows)

    def _list_nodes(self: ResearchRepository, sid: str) -> list[ResearchNode]:
        return list(node_rows)

    def _list_decisions(self: ResearchRepository, sid: str) -> list[DecisionRecord]:
        return list(decision_rows)

    monkeypatch.setattr("app.research.sources.source_references", _refs)
    monkeypatch.setattr(ResearchRepository, "list_evidence", _list_evidence)
    monkeypatch.setattr(ResearchRepository, "list_nodes", _list_nodes)
    monkeypatch.setattr(ResearchRepository, "list_decisions", _list_decisions)
    out = _run_with(monkeypatch, sched_result)

    assert out["evidence"] == [
        {
            "id": "ev-1",
            "content": "stored fact",
            "sources": [{"id": "evidence:ev-1", "title": "Capture", "url": "https://example.test/a"}],
        },
        # A row with no stored text still ships as its clipped record JSON
        # (HEAD kernel_worker.py:186-195 fallback, :1951-1961 append).
        {"id": "ev-blank", "content": '{"evidence_id": "ev-blank", "content": "   "}'},
    ]
    assert out["incomplete_guard"] is True  # the node-level guard reaches the response
    assert out["stalled"] is False and "stalled_reason" not in out
    assert out["unresolved"] == ["n1"] and out["escalated"] is True and out["escalations"] == 1
    assert out["failures"] == {"tool_error": 2}  # reselect never fails a run
    executions = out["toolExecutions"]
    assert isinstance(executions, list) and out["needleDecisions"] == executions
    assert executions == [
        {
            "step": 0,
            "tool": "get_sec_document",
            "arguments": {"accession_no": "a"},
            "confidence": 0.5,
            "reasoning": "",
        },
        {"step": 1, "tool": "search_web", "arguments": {}, "confidence": None, "reasoning": ""},
        {"step": 2, "tool": "unknown", "arguments": {}, "confidence": None, "reasoning": ""},
        {"step": 3, "tool": "search_sec_filings", "arguments": {}, "confidence": None, "reasoning": ""},
        {"step": 4, "tool": "query_finra", "arguments": {}, "confidence": None, "reasoning": ""},
    ]
    assert out["toolCalls"] == [
        {"tool": "get_sec_document", "ok": True, "step": 0, "evidenceId": "ev-1"},
        {"tool": "search_web", "ok": True, "step": 1},
        {"tool": "unknown", "ok": False, "step": 2, "error": "tool blew up", "category": "tool_error"},
        {"tool": "search_sec_filings", "ok": False, "step": 3, "error": "reselect", "category": "reselect"},
        {"tool": "query_finra", "ok": False, "step": 4, "error": "tool failed", "category": "tool_error"},
    ]
    assert out["nodes"] == [
        {"node_id": "n1", "question": "q1?", "status": "gathering", "depends_on": ["n0"]},
        {"node_id": "n2", "question": "q2?", "status": "resolved", "depends_on": []},
    ]
    assert out["decisions"] == [d.to_dict() for d in decision_rows]
    assert out["watchlist"] == [{"symbol": "AAPL"}]  # only the intake_digest carries coverage


def test_run_scope_rejects_malformed_ticker_before_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ticker channel is invalid_params, and no session is created."""

    def _boom(*a: object, **k: object) -> str:
        raise AssertionError("setup must not run for a malformed ticker")

    monkeypatch.setattr(kw, "run_graph_prompt", _boom)
    out = kw._run({"id": "rt", "prompt": "q?", "ticker": 7}, jev=JevClient())
    terminal = out["terminal"]
    assert isinstance(terminal, dict) and terminal["category"] == "invalid_params"
    assert "ticker" in str(terminal["message"])


@pytest.mark.parametrize("prompt", [None, "", "   ", 7])
def test_run_requires_a_prompt_before_any_setup(monkeypatch: pytest.MonkeyPatch, prompt: JSONValue) -> None:
    """A missing or blank prompt is invalid_params, before scope validation and setup."""

    def _boom(*a: object, **k: object) -> str:
        raise AssertionError("setup must not run without a prompt")

    monkeypatch.setattr(kw, "run_graph_prompt", _boom)
    out = kw._run({"id": "rp", "prompt": prompt}, jev=JevClient())
    assert out["id"] == "rp"
    assert out["error"] == "prompt required"
    assert out["terminal"] == {"category": "invalid_params", "message": "prompt required"}


def test_run_sessionid_without_gap_starts_a_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sessionId without a gap is not a pass-2 request: setup still runs and no node is created."""
    seen: list[str] = []

    def _graph(prompt: str, as_of: object = None, **rest: object) -> str:
        seen.append(prompt)
        return "s1"

    async def _sched(sid: str, **hooks: object) -> dict[str, object]:
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(kw, "run_graph_prompt", _graph)
    monkeypatch.setattr("app.research.scheduler.run", _sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    _empty_repo(monkeypatch)
    out = kw._run({"id": "rz", "prompt": "q?", "sessionId": "rs:ignored"}, jev=JevClient())
    assert seen == ["q?"]
    assert out["sessionId"] == "s1"
    assert out["unresolved"] == [] and out["escalated"] is False and out["escalations"] == 0


@pytest.mark.parametrize(
    ("line", "expected_message"),
    [
        ("{oops", "invalid JSON"),
        ("[1, 2]", f"op must be one of {', '.join(kw._KERNEL_OPS)}"),
        ('{"op": "nope"}', f"op must be one of {', '.join(kw._KERNEL_OPS)}"),
    ],
)
def test_request_line_reports_bad_input(line: str, expected_message: str) -> None:
    """Invalid JSON, a non-mapping request, and an unknown op all answer invalid_params."""
    req, response = kw._request_or_response(line)
    assert req is None
    assert response is not None and response["id"] == "?"
    assert response["terminal"] == {"category": "invalid_params", "message": expected_message}


def test_request_line_keeps_known_id_and_blank_reports_nothing() -> None:
    """A blank line is answered with nothing; an unknown op keeps the caller's own id."""
    assert kw._request_or_response("  \n") == (None, None)
    req, response = kw._request_or_response('{"op": "nope", "id": "keep-me"}')
    assert req is None
    assert response is not None and response["id"] == "keep-me"
    valid, response = kw._request_or_response('{"op": "route", "id": "r1", "prompt": "hi"}')
    assert response is None
    assert valid == {"op": "route", "id": "r1", "prompt": "hi"}


@pytest.mark.parametrize(
    "op",
    ["route", "arguments", "assess_entry", "assess_final_response", "assess_final_claim"],
)
def test_direct_op_dispatches_to_its_handler(monkeypatch: pytest.MonkeyPatch, op: str) -> None:
    """Each direct op reaches its own handler; only route and the assess gates get the shared client."""
    calls: list[tuple[str, object]] = []

    def _stub(req: Mapping[str, JSONValue], jev: object = None) -> dict[str, JSONValue]:
        calls.append((kw._request_id(req), jev))
        return {"id": kw._request_id(req), "handled": op}

    monkeypatch.setattr(kw, f"_{op}", _stub)
    jev = JevClient()
    out = kw._direct_op({"id": "d1", "op": op}, jev)
    assert out == {"id": "d1", "handled": op}
    assert calls == [("d1", None if op == "arguments" else jev)]


def test_direct_op_failure_is_worker_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising direct op answers with the worker failure terminal, never a raise."""

    def _boom(*a: object, **k: object) -> dict[str, JSONValue]:
        raise RuntimeError("op blew up")

    monkeypatch.setattr(kw, "_assess_entry", _boom)
    out = kw._direct_op({"id": "d2", "op": "assess_entry"}, None)
    assert out["id"] == "d2"
    assert out["error"] == "worker failed: op blew up"
    assert out["terminal"] == {"category": "provider_error", "message": out["error"]}


def test_configure_logging_maps_level_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """STOCKBOT_LOG_LEVEL selects the level; an unknown name falls back to WARNING."""
    seen: list[dict[str, object]] = []

    def _basic_config(**kwargs: object) -> None:
        seen.append(dict(kwargs))

    monkeypatch.setattr(kw.logging, "basicConfig", _basic_config)
    monkeypatch.setenv("STOCKBOT_LOG_LEVEL", "debug")
    kw._configure_logging()
    monkeypatch.setenv("STOCKBOT_LOG_LEVEL", "chatty")
    kw._configure_logging()
    assert [row["level"] for row in seen] == [kw.logging.DEBUG, kw.logging.WARNING]
    assert seen[0]["format"] == "%(asctime)s %(levelname)s %(name)s: %(message)s"


def test_prewarm_caches_warms_then_only_logs_failures(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Both warm steps run in order; either failure logs and leaves the worker live."""
    import logging

    import app.sec.client as sec_client
    from app.research import scheduler as _sched

    warmed: list[str] = []

    def _frame(key: str) -> None:
        warmed.append(key)

    monkeypatch.setattr(_sched, "build_registry", lambda: warmed.append("registry"))
    monkeypatch.setattr(sec_client, "ensure_identity", lambda: warmed.append("identity"))
    monkeypatch.setattr(sec_client, "_fetch_lookup_frame", _frame)
    kw._prewarm_caches()
    assert warmed == ["registry", "identity", "warmup"]

    def _boom(*a: object, **k: object) -> None:
        raise RuntimeError("warm down")

    monkeypatch.setattr(_sched, "build_registry", _boom)
    monkeypatch.setattr(sec_client, "_fetch_lookup_frame", _boom)
    with caplog.at_level(logging.DEBUG, logger=kw.__name__):
        kw._prewarm_caches()
    assert [r.getMessage() for r in caplog.records] == [
        "prewarm registry failed: warm down",
        "prewarm CIK lookup failed: warm down",
    ]


def test_prewarm_starts_named_daemon_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prewarm hands the warm work to a named daemon thread, so startup never waits for it."""
    started: dict[str, object] = {}

    class _Thread:
        def __init__(self, target: object = None, name: str = "", daemon: bool = False) -> None:
            started.update(target=target, name=name, daemon=daemon)

        def start(self) -> None:
            started["started"] = True

    monkeypatch.setattr(kw.threading, "Thread", _Thread)
    kw._prewarm()
    assert started == {"target": kw._prewarm_caches, "name": "kernel-prewarm", "daemon": True, "started": True}


def test_shutdown_closes_jev_needle_and_pools_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown is idempotent; a JEV close that raises never blocks the remaining closes."""
    import sys as _sys
    from types import SimpleNamespace

    closed: list[str] = []

    class _Jev:
        def close(self) -> None:
            closed.append("jev")
            raise OSError("already gone")

    def _pools() -> None:
        closed.append("pools")

    def _needle_close() -> None:
        closed.append("needle")

    monkeypatch.setattr(kw, "_JEV", _Jev())
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    monkeypatch.setattr(kw, "_shutdown_pools", _pools)
    monkeypatch.setitem(_sys.modules, "app.needle_client", SimpleNamespace(close=_needle_close))
    kw._shutdown()
    kw._shutdown()
    assert closed == ["jev", "needle", "pools"]


def test_shutdown_survives_missing_needle_and_dead_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Needle module or a pool without shutdown never breaks worker exit."""
    import sys as _sys

    from app.research import scheduler as _sched

    calls: list[str] = []
    real_pools = kw._shutdown_pools

    def _counting_pools() -> None:
        calls.append("pools")
        real_pools()

    monkeypatch.setattr(kw, "_JEV", None)
    monkeypatch.setattr(kw, "_SHUTDOWN_DONE", False)
    monkeypatch.setattr(kw, "_shutdown_pools", _counting_pools)
    monkeypatch.setitem(_sys.modules, "app.needle_client", None)
    monkeypatch.setattr(_sched, "_SEC_POOL", object())
    kw._shutdown()
    kw._shutdown()
    assert calls == ["pools"]  # the second call returns before touching anything


def test_arguments_requires_tool_before_anything_else() -> None:
    """A blank or missing tool is denied without touching Needle."""
    for req in ({"id": "a0"}, {"id": "a0", "tool": ""}, {"id": "a0", "tool": 7}, {"id": "a0", "objective": "q?"}):
        assert kw._arguments(req) == {"id": "a0", "error": "tool required"}


@pytest.mark.parametrize("tool", ["research_resume", "research_status", "research_cancel"])
def test_arguments_session_carry_skips_needle(monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    """resume/status/cancel carry the caller's session id without grounding or invoking the tool."""

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run for a sid carry")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments({"id": "s1", "tool": tool, "arguments": {"session_id": " rs:1 "}})
    assert out == {
        "id": "s1",
        "tool": tool,
        "arguments": {"session_id": "rs:1"},
        "confidence": 1.0,
        "reasoning": "session_id carried from caller; no Needle grounding needed",
    }


def test_arguments_read_search_carries_ids_or_denies() -> None:
    """read_search carries session_id+search_id from node and arguments, and denies a missing search_id."""
    carried = kw._arguments(
        {
            "id": "r1",
            "tool": "research_read_search",
            "node": {"session_id": "rs:9"},
            "arguments": {"search_id": " s1 "},
        }
    )
    assert carried == {
        "id": "r1",
        "tool": "research_read_search",
        "arguments": {"session_id": "rs:9", "search_id": "s1"},
        "confidence": 1.0,
        "reasoning": "session_id+search_id carried from caller; no Needle grounding needed",
    }
    denied = kw._arguments({"id": "r2", "tool": "research_read_search", "node": {"session_id": "rs:9"}})
    assert denied == {
        "id": "r2",
        "error": "research_read_search needs search_id from a prior search_sec_filings result",
    }


def test_arguments_session_id_falls_back_to_node() -> None:
    """A blank arguments.session_id falls back to node.session_id."""
    carried = kw._arguments(
        {"id": "r3", "tool": "research_status", "arguments": {"session_id": " "}, "node": {"session_id": "rs:5"}}
    )
    assert carried["arguments"] == {"session_id": "rs:5"}


def test_arguments_bad_ticker_is_answered_without_fail_open(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed ticker channel is its own envelope: the call never reaches the fail-open boundary."""
    import logging

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run with a malformed ticker")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    with caplog.at_level(logging.WARNING, logger=kw.__name__):
        out = kw._arguments({"id": "t1", "tool": "list_sec_filings", "objective": "q?", "ticker": 7})
    assert out["id"] == "t1" and "ticker" in str(out["error"])
    assert caplog.records == []


def test_arguments_schema_falls_back_to_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty inline schema reads the registry schema for the tool and passes it to Needle."""
    seen: list[dict[str, object]] = []

    def _gen(**kwargs: object) -> dict[str, object]:
        seen.append(dict(kwargs))
        return {"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "confidence": 0.25, "reasoning": "r"}

    def _schema(tool: str, reg: object) -> dict[str, object]:
        return {"type": "object", "title": tool}

    def _grounded(tool: str, schema: object, objective: object) -> dict[str, JSONValue]:
        return {}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)
    monkeypatch.setattr("app.research.scheduler.build_registry", lambda: [{"name": "query_finra"}])
    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    monkeypatch.setattr("app.research.scheduler._grounded_execution_values", _grounded)
    out = kw._arguments(
        {
            "id": "a2",
            "tool": "query_finra",
            "objective": "NVDA revenue?",
            "schema": {},
            "node": {"session_id": "rs:1"},
            "context": {},
        }
    )
    assert out == {
        "id": "a2",
        "tool": "query_finra",
        # The scheduler repair seeds the FINRA dataset when the call names none
        # (HEAD kernel_worker.py:1627 repair; scheduler.py:2221-2229 seed).
        "arguments": {"ticker": "NVDA", "dataset": "consolidatedShortInterest"},
        "confidence": 0.25,
        "reasoning": "r",
    }
    sent = seen[0]
    assert sent["schema"] == {"type": "object", "title": "query_finra"}
    assert sent["objective"] == "NVDA revenue?" and sent["node"] == {"session_id": "rs:1"}
    assert isinstance(sent["context"], dict)


def _needle_withhold(**kwargs: object) -> None:
    """A genuine Needle withhold: generate() answers None instead of a tool call."""


def test_arguments_withhold_seeds_or_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Needle withhold seeds grounded SEC arguments; an unseedable objective fails open."""

    def _fallback(tool: str, prompt: str) -> dict[str, object]:
        return {"identifier": "NVDA"}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _needle_withhold)
    monkeypatch.setattr("app.research.scheduler._fallback_sec_args", _fallback)
    seeded = kw._arguments({"id": "w1", "tool": "list_sec_filings", "objective": "NVDA revenue?"})
    assert seeded == {
        "id": "w1",
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA"},
        "confidence": None,
        "reasoning": "seeded after Needle withhold",
    }

    def _no_fallback(tool: str, prompt: str) -> None:
        return None

    monkeypatch.setattr("app.research.scheduler._fallback_sec_args", _no_fallback)
    failed = kw._arguments({"id": "w2", "tool": "list_sec_filings", "objective": "what happened?"})
    assert failed == {"id": "w2", "error": "needle arguments failed"}


def test_arguments_result_drops_non_numeric_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-mapping generate result carries no confidence or reasoning; a boolean confidence reads as None."""

    def _repair(*a: object, **k: object) -> dict[str, object]:
        return {"ticker": "NVDA"}

    def _not_a_mapping(**kwargs: object) -> object:
        return "not-a-mapping"

    monkeypatch.setattr("app.research.scheduler._repair_tool_arguments", _repair)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _not_a_mapping)
    assert kw._arguments({"id": "n1", "tool": "query_finra", "objective": "NVDA?"}) == {
        "id": "n1",
        "tool": "query_finra",
        "arguments": {"ticker": "NVDA"},
    }
    generated: dict[str, object] = {"tool": "query_finra", "arguments": {}, "confidence": True, "reasoning": 7}

    def _boolean_confidence(**kwargs: object) -> dict[str, object]:
        return generated

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boolean_confidence)
    out = kw._arguments({"id": "n2", "tool": "query_finra", "objective": "NVDA?"})
    assert out["confidence"] is None and out["reasoning"] == ""


class _JevBlankVerdict(JevClient):
    @override
    async def assess_entry_tool(
        self,
        prompt: str,
        tool: str,
        arguments: Mapping[str, JSONValue] | None = None,
        result: Mapping[str, JSONValue] | None = None,
    ) -> str:
        return ""


def test_assess_entry_denies_before_jev_and_narrows_payloads() -> None:
    """Tool and result cross as mappings only; a tool-less request and a blank verdict stay research_required."""
    jev = _JevAssess("get_sec_document")
    assert kw._assess_entry({"id": "e1", "prompt": "risk?"}, jev=jev) == {"id": "e1", "verdict": "research_required"}
    assert kw._assess_entry({"id": "e2", "prompt": "risk?", "tool": ""}, jev=jev) == {
        "id": "e2",
        "verdict": "research_required",
    }
    assert jev.seen == []
    out = kw._assess_entry(
        {"id": "e3", "prompt": " risk? ", "tool": "t", "arguments": "nope", "result": ["nope"]},
        jev=jev,
    )
    assert out == {"id": "e3", "verdict": "get_sec_document"}
    # The prompt crosses with the temporal prefix and stripped (HEAD kernel_worker.py:1676).
    seen_prompt, seen_tool, seen_args, seen_result = jev.seen[0]
    assert seen_prompt.startswith("[Today UTC ") and seen_prompt.endswith("] risk?")
    assert (seen_tool, seen_args, seen_result) == ("t", {}, {})
    assert kw._assess_entry({"id": "e4", "prompt": "risk?", "tool": "t"}, jev=_JevBlankVerdict()) == {
        "id": "e4",
        "verdict": "research_required",
    }


async def _list_result(*a: object, **k: object) -> object:
    return ["not", "a", "mapping"]


async def _text_result(*a: object, **k: object) -> object:
    return "not-a-mapping"


def test_assess_final_malformed_results_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-mapping JEV result is a fail-closed verdict carrying the malformed-result text."""
    jev = JevClient()
    monkeypatch.setattr(jev, "assess_final_response", _list_result)
    monkeypatch.setattr(jev, "assess_final_claim", _text_result)
    response = kw._assess_final_response({"id": "f1", "text": "claim text"}, jev=jev)
    assert response == {
        "id": "f1",
        "error": "assess_final_response: malformed result",
        "support": None,
        "support_answer": None,
        "evidence_state": "insufficient",
    }
    claim = kw._assess_final_claim({"id": "c1", "claim": {"text": "claim text", "claim_type": "inference"}}, jev=jev)
    assert claim == {
        "id": "c1",
        "error": "assess_final_claim: malformed result",
        "support": None,
        "support_answer": None,
        "evidence_state": "insufficient",
    }


def test_assess_final_response_narrows_claims_and_confidence() -> None:
    """Claims keep only complete rows, cap text at 2000 chars, and pass numeric confidence through."""
    seen: dict[str, object] = {}

    class _Recorder(JevClient):
        @override
        async def assess_final_response(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen["args"] = a
            seen["kwargs"] = k
            return {"support_answer": "yes", "evidence_state": "sufficient_support"}

    out = kw._assess_final_response(
        {
            "id": "f2",
            "text": " WSJ reported X. ",
            "confidence": True,
            "claims": [
                {
                    "text": f"  {'c' * 2100}  ",
                    "claim_type": " inference ",
                    "confidence": 0.5,
                    "evidence_ids": [" ev-1 ", 7, ""],
                },
                {"text": "  ", "claim_type": "inference"},
                {"text": "x", "claim_type": ""},
                "not-a-mapping",
                {"text": "y", "claim_type": "fact", "confidence": True},
                {"text": "z", "claim_type": "fact", "evidence_ids": "ev-1"},
            ],
        },
        jev=_Recorder(),
    )
    assert out == {"id": "f2", "support_answer": "yes", "evidence_state": "sufficient_support"}
    assert seen["args"] == ("WSJ reported X.", [])
    assert seen["kwargs"] == {
        "session_id": "f2",
        "confidence": None,
        "claims": [
            {
                "text": "c" * 2000,
                "claim_type": "inference",
                "evidence_ids": [" ev-1 "],
                "confidence": 0.5,
            },
            {"text": "y", "claim_type": "fact", "evidence_ids": [], "confidence": None},
            {"text": "z", "claim_type": "fact", "evidence_ids": [], "confidence": None},
        ],
    }


def test_assess_final_claim_carries_own_ids_as_empty_rows() -> None:
    """Without evidence rows the claim's own ids cross as empty-content rows; claim_type is required."""
    seen: dict[str, object] = {}

    class _Recorder(JevClient):
        @override
        async def assess_final_claim(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen["args"] = a
            seen["kwargs"] = k
            return {"support_answer": "no", "evidence_state": "insufficient_support"}

    out = kw._assess_final_claim(
        {
            "id": "c2",
            "claim": {"text": "X.", "claim_type": "inference", "evidence_ids": ["ev-1", 7], "confidence": 0.5},
        },
        jev=_Recorder(),
    )
    assert out == {"id": "c2", "support_answer": "no", "evidence_state": "insufficient_support"}
    assert seen["args"] == ("X.", "inference", [{"id": "ev-1", "content": ""}])
    assert seen["kwargs"] == {"session_id": "c2", "confidence": 0.5}

    missing = kw._assess_final_claim({"id": "c3", "claim": {"text": "X."}})
    assert missing == {
        "id": "c3",
        "error": "claim_type required",
        "support": None,
        "support_answer": None,
        "evidence_state": "insufficient",
    }


def test_jev_disposition_asks_one_choice_per_proposal() -> None:
    """Every proposal gets one choice question with the shared criteria and its own question text."""
    proposals = _props()
    questions, choice_options = kw._jev_disposition(proposals)
    assert list(questions) == ["s-q1", "s-q2"] and list(choice_options) == ["s-q1", "s-q2"]
    first = questions["s-q1"]
    assert first == {
        "type": "choice",
        "instructions": (
            "What should happen to this proposed question relative to the user's objective? Question: Other angle?"
        ),
        "criteria": kw._DISPOSITION_OPTIONS,
    }
    assert choice_options["s-q1"] == kw._DISPOSITION_OPTIONS


def test_jev_kept_keeps_everything_but_an_explicit_reject() -> None:
    """A reject drops the proposal; a missing or malformed disposition keeps it."""
    proposals = _props()
    assert kw._jev_kept(proposals, {"s-q1": {"choice": "reject"}, "s-q2": {"choice": "gather_evidence"}}) == [
        proposals[1]
    ]
    assert kw._jev_kept(proposals, {}) == proposals
    assert kw._jev_kept(proposals, {"s-q1": "gather_evidence"}) == proposals


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, None), (-5, None), (True, None), ("soon", None), (None, None), (1_500.5, 1_500.5)],
)
def test_deadline_field_accepts_only_positive_numbers(raw: object, expected: float | None) -> None:
    """An epoch-ms deadline is a positive number that is not a bool; everything else reads as absent."""
    assert kw._deadline_field(raw) == expected


def test_intake_pending_skips_fetched_scope() -> None:
    """Only unseen tickers and an unseen query are fetched again."""
    assert kw._intake_pending(["ORCL", "NVDA"], ["ORCL"], "q?", []) == (["NVDA"], "q?")
    assert kw._intake_pending(["ORCL"], ["ORCL"], "q?", ["q?"]) == ([], None)
    assert kw._intake_pending([], [], None, []) == ([], None)


def test_intake_hint_tickers_normalizes_and_dedupes() -> None:
    """Reasoner hints become new uppercase tickers: junk drops, known ones never repeat."""
    hints: dict[str, object] = {"tickers": ["nvda", "NVDA", 7, "  ", "orcl", "orcl"]}
    assert kw._intake_hint_tickers(hints, ["NVDA"]) == ["ORCL"]
    assert kw._intake_hint_tickers({}, ["NVDA"]) == []
    assert kw._intake_hint_tickers({"tickers": "NVDA"}, []) == []


def test_intake_corrected_query_only_when_new() -> None:
    """A corrected query is used once, and never when blank or already fetched."""
    assert kw._intake_corrected_query({"corrected_query": " fixed "}, ["q?"]) == "fixed"
    assert kw._intake_corrected_query({"corrected_query": "fixed"}, ["fixed"]) is None
    assert kw._intake_corrected_query({"corrected_query": "  "}, []) is None
    assert kw._intake_corrected_query({}, []) is None


def test_intake_digest_unchanged_needs_a_second_round_and_no_correction() -> None:
    """Round 2 with the same digest and no corrected query reuses round-1 proposals."""
    assert kw._intake_digest_unchanged(0, "d", "d", {}) is False
    assert kw._intake_digest_unchanged(1, "d", "e", {}) is False
    assert kw._intake_digest_unchanged(1, "d", "d", {"corrected_query": "fixed"}) is False
    assert kw._intake_digest_unchanged(1, "d", "d", {}) is True


def test_intake_round2_spent_needs_a_deadline_and_a_first_round() -> None:
    """Round 1 always runs; round 2 needs at least the round-2 minimum left in the setup budget."""
    import time as _time

    now = _time.perf_counter()
    assert kw._intake_round2_spent(None, 5) is False
    assert kw._intake_round2_spent(now + 3600, 0) is False
    assert kw._intake_round2_spent(now + 1.0, 1) is True
    assert kw._intake_round2_spent(now + 3600, 1) is False


def test_intake_decompose_timeout_keeps_a_floor() -> None:
    """The decompose wait follows the setup deadline, never below one second."""
    import time as _time

    now = _time.perf_counter()
    assert kw._intake_decompose_timeout(None) is None
    assert kw._intake_decompose_timeout(now - 10.0) == 1.0
    wait = kw._intake_decompose_timeout(now + 7.0)
    assert wait is not None and 1.0 <= wait <= 7.0


def test_intake_start_tickers_reads_the_objective_subject() -> None:
    """The objective's own company scopes round 1; an objective without one starts empty."""
    assert kw._intake_start_tickers("What is happening with Oracle now") == ["ORCL"]
    assert kw._intake_start_tickers("news for growth this week") == []


def test_intake_timed_reports_its_own_timeout_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call past the per-call timeout reports a timeout record, never an exception."""

    async def _slow(*a: object, **k: object) -> dict[str, object]:
        await asyncio.sleep(60)
        return {}

    monkeypatch.setattr(kw, "_intake_attempt", _slow)
    monkeypatch.setattr(kw, "_INTAKE_CALL_TIMEOUT_S", 0.05)
    tool, args, record, taken = asyncio.run(
        kw._intake_timed("search_web", {"query": "q?"}, "s1", {}, [], None, sched._Kernel(), None)
    )
    assert (tool, args) == ("search_web", {"query": "q?"})
    assert record == {
        "tool": "search_web",
        "arguments": {"query": "q?"},
        "error": "intake_timeout",
        "error_type": "timeout",
    }
    assert taken >= 0.0


def test_intake_task_meta_labels_every_call() -> None:
    """Chains, web, and watchlist calls each carry the tool and arguments a timeout would report."""

    async def _scenario() -> None:
        async def _chain() -> kw._IntakeChainRows:
            return []

        async def _call() -> kw._IntakeCallRow:
            return ("search_web", {}, {}, 0.0)

        chain = asyncio.ensure_future(_chain())
        web = asyncio.ensure_future(_call())
        watch = asyncio.ensure_future(_call())
        meta = kw._intake_task_meta([chain], ["ORCL"], [web], None, [watch], [("AAPL", "list_sec_filings")])
        assert meta == {
            id(chain): ("list_sec_filings", {"ticker": "ORCL"}),
            id(web): ("search_web", {"query": ""}),
            id(watch): ("list_sec_filings", {"identifier": "AAPL", "limit": kw._INTAKE_SEC_LIMIT}),
        }
        for task in (chain, web, watch):
            task.cancel()
        await asyncio.gather(chain, web, watch, return_exceptions=True)

    asyncio.run(_scenario())


def test_intake_outcome_rows_cover_chain_call_timeout_and_straggler() -> None:
    """A chain yields its rows, a call yields one row plus its record, and a straggler is cancelled."""

    async def _scenario() -> None:
        async def _chain() -> kw._IntakeChainRows:
            return [("list_sec_filings", {"ticker": "ORCL"}, {"tool": "list_sec_filings"}, 5.0)]

        async def _call() -> kw._IntakeCallRow:
            return ("search_web", {"query": "q?"}, {"tool": "search_web"}, 7.0)

        async def _hang() -> kw._IntakeChainRows:
            await asyncio.Event().wait()
            return []

        async def _expired() -> kw._IntakeChainRows:
            raise TimeoutError

        chain = asyncio.ensure_future(_chain())
        call = asyncio.ensure_future(_call())
        hang = asyncio.ensure_future(_hang())
        expired = asyncio.ensure_future(_expired())
        await asyncio.sleep(0.05)
        assert chain.done() and call.done() and expired.done() and not hang.done()

        chain_rows, chain_watch = kw._intake_outcome_rows(chain, ("list_sec_filings", {"ticker": "ORCL"}), True)
        assert chain_rows == [("list_sec_filings", {"ticker": "ORCL"}, {"tool": "list_sec_filings"}, 5.0)]
        assert chain_watch is None
        call_rows, call_watch = kw._intake_outcome_rows(call, ("search_web", {"query": "q?"}), True)
        assert call_rows == [("search_web", {"query": "q?"}, {"tool": "search_web"}, 7.0)]
        assert call_watch == {"tool": "search_web"}
        expired_rows, expired_watch = kw._intake_outcome_rows(expired, ("list_sec_filings", {}), True)
        assert expired_rows == [
            (
                "list_sec_filings",
                {},
                {"tool": "list_sec_filings", "arguments": {}, "error": "intake_timeout", "error_type": "timeout"},
                0.0,
            )
        ]
        assert expired_watch is None
        late = {"tool": "list_sec_filings", "arguments": {"ticker": "X"}, "error": "intake_timeout"}
        late["error_type"] = "timeout"
        pending_rows, pending_watch = kw._intake_outcome_rows(hang, ("list_sec_filings", {"ticker": "X"}), False)
        # cancel() requests cancellation; the task reports cancelled only once the loop delivers it.
        assert hang.cancelling() == 1 and pending_watch == late
        assert pending_rows == [("list_sec_filings", {"ticker": "X"}, late, 0.0)]
        await asyncio.gather(chain, call, hang, expired, return_exceptions=True)
        assert hang.cancelled()

    asyncio.run(_scenario())


def test_intake_settle_reports_rows_watch_records_and_wall_time() -> None:
    """One wait settles every call and keeps the watch record its row came from."""

    async def _scenario() -> None:
        async def _call() -> kw._IntakeCallRow:
            return ("search_web", {"query": "q?"}, {"tool": "search_web"}, 1.0)

        task = asyncio.ensure_future(_call())
        flat, watch_records, wall_ms = await kw._intake_settle([task], {id(task): ("search_web", {"query": "q?"})})
        assert flat == [("search_web", {"query": "q?"}, {"tool": "search_web"}, 1.0)]
        assert watch_records == {id(task): {"tool": "search_web"}}
        assert isinstance(wall_ms, float) and wall_ms >= 0.0

    asyncio.run(_scenario())


def test_intake_source_timing_splits_sec_and_web() -> None:
    """Per-source counts and milliseconds add up; search_web lands in the web bucket only."""
    calls: list[tuple[str, dict[str, object]]] = [
        ("list_sec_filings", {}),
        ("search_web", {}),
        ("list_sec_filings", {}),
    ]
    timed: list[tuple[dict[str, object], float]] = [({"tool": "a"}, 10.0), ({"tool": "b"}, 4.0), ({"tool": "c"}, 1.0)]
    per_source, sec_ms, web_ms = kw._intake_source_timing(calls, timed)
    assert per_source == {"list_sec_filings": {"count": 2, "ms": 11.0}, "search_web": {"count": 1, "ms": 4.0}}
    assert (sec_ms, web_ms) == (11.0, 4.0)


def test_intake_admit_all_settles_failed_calls_and_counts_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed call settles its job; a jobless timeout only counts; an uncitable answer counts as such."""
    kernel = sched._Kernel()
    settled: list[object] = []

    def _settle(kernel_: object, record: Mapping[str, object], sid: str, node_id: str) -> None:
        settled.append(record.get("job_id"))

    def _complete(kernel_: object, record: object, tool: str) -> None:
        settled.append(f"complete:{tool}")

    def _no_candidate(*a: object, **k: object) -> None:
        return None

    monkeypatch.setattr(sched, "_settle_attempt", _settle)
    monkeypatch.setattr(sched, "_complete_attempt_job", _complete)
    monkeypatch.setattr(sched, "_evidence_candidate", _no_candidate)
    calls: list[tuple[str, dict[str, object]]] = [
        ("list_sec_filings", {"identifier": "ORCL"}),
        ("search_web", {"query": "q?"}),
        ("query_finra", {"ticker": "NVDA"}),
    ]
    timed: list[tuple[dict[str, object], float]] = [
        ({"tool": "list_sec_filings", "arguments": {}, "error": "boom", "job_id": "j1"}, 1.0),
        ({"tool": "search_web", "arguments": {}, "error": "intake_timeout", "error_type": "timeout"}, 0.0),
        ({"tool": "query_finra", "arguments": {}, "outcome": {"ok": True}, "result": {"x": 1}}, 2.0),
    ]
    raw, outcomes, admitted_ids, settle_ms = kw._intake_admit_all(kernel, "s1", calls, timed)
    assert settled == ["j1", "complete:query_finra"]
    assert outcomes == {"admitted": 0, "duplicate": 0, "not_citable": 1, "error": 1, "timeout": 1}
    assert admitted_ids == [] and settle_ms >= 0.0
    assert raw == [
        {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "ORCL"},
            # The log view carries every key, absent values as None
            # (HEAD kernel_worker.py:850-859 _jsonable_record).
            "record": {
                "tool": "list_sec_filings",
                "arguments": {},
                "outcome_summary": None,
                "error": "boom",
                "error_type": None,
                "evidence_id": None,
                "job_id": "j1",
            },
        },
        {
            "tool": "search_web",
            "arguments": {"query": "q?"},
            "record": {
                "tool": "search_web",
                "arguments": {},
                "outcome_summary": None,
                "error": "intake_timeout",
                "error_type": "timeout",
                "evidence_id": None,
                "job_id": None,
            },
        },
        {
            "tool": "query_finra",
            "arguments": {"ticker": "NVDA"},
            "record": {
                "tool": "query_finra",
                "arguments": {},
                "outcome_summary": None,
                "error": None,
                "error_type": None,
                "evidence_id": None,
                "job_id": None,
            },
        },
    ]


def test_intake_admit_all_counts_admitted_and_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An accepted candidate is persisted and counted; a duplicate never counts twice."""
    kernel = sched._Kernel()

    def _noop(*a: object, **k: object) -> None:
        return None

    def _candidate(*a: object, **k: object) -> dict[str, bool]:
        return {"candidate": True}

    results: Iterator[tuple[str, bool]] = iter([("ev-1", True), ("ev-1", False)])

    def _persist(*a: object, **k: object) -> tuple[str, bool]:
        return next(results)

    monkeypatch.setattr(sched, "_complete_attempt_job", _noop)
    monkeypatch.setattr(sched, "_evidence_candidate", _candidate)
    monkeypatch.setattr(sched, "_persist_admitted_evidence", _persist)
    calls: list[tuple[str, dict[str, object]]] = [("list_sec_filings", {}), ("list_sec_filings", {})]
    timed: list[tuple[dict[str, object], float]] = [
        ({"arguments": {}, "outcome": {"ok": True}}, 1.0),
        ({"arguments": {}, "outcome": {"ok": True}}, 1.0),
    ]
    raw, outcomes, admitted_ids, _ms = kw._intake_admit_all(kernel, "s1", calls, timed)
    assert admitted_ids == ["ev-1"]
    assert outcomes["admitted"] == 1 and outcomes["duplicate"] == 1
    records = [row["record"] for row in raw]
    # The log row is snapshotted before admission (HEAD kernel_worker.py:967 append,
    # :989 evidence_id write), so the raw view never carries the admitted id.
    assert [rec.get("evidence_id") for rec in records if isinstance(rec, dict)] == [None, None]


def test_run_followup_setup_failure_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising pass-2 setup reports provider_error with the session-setup text."""

    def _boom(session_id: str, gap: str) -> tuple[str, list[str]]:
        raise RuntimeError("setup blew up")

    monkeypatch.setattr(kw, "_followup_gap_nodes", _boom)
    out = kw._run({"id": "rf", "prompt": "q?", "sessionId": "rs:1", "gap": "need doc X"}, jev=JevClient())
    assert out["error"] == "session setup failed: setup blew up"
    assert out["terminal"] == {"category": "provider_error", "message": out["error"]}


def test_decompose_empty_proposals_twice_is_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two empty proposal sets exhaust the retry budget and surface the empty-proposals rejection."""
    calls = {"n": 0}

    # ReasonerClient.decompose is synchronous (app/reasoner_client.py:424); an async stub never runs.
    def _empty(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        return {}

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _empty)
    with pytest.raises(TypeError, match="decompose: empty proposals"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
    assert calls["n"] == 2


def test_decompose_spent_budget_before_retry_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty proposal set with a spent setup budget fails before a second call."""
    import time as _time

    calls = {"n": 0}

    # ReasonerClient.decompose is synchronous (app/reasoner_client.py:424); an async stub never runs.
    def _empty(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        return {}

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _empty)
    with pytest.raises(RuntimeError, match="reasoner setup budget spent before retry"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "", None, None, _time.perf_counter() - 1.0)
    assert calls["n"] == 1


def test_decompose_missing_config_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing-config failure is terminal after the first call, never a second wait."""
    calls = {"n": 0}

    # ReasonerClient.decompose is synchronous (app/reasoner_client.py:424); an async stub never runs.
    def _unavailable(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        raise RuntimeError("OPENCODE_UNAVAILABLE: no reasoner configured")

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _unavailable)
    with pytest.raises(RuntimeError, match="OPENCODE_UNAVAILABLE"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
    assert calls["n"] == 1


def test_attempt_category_prefers_error_type_and_rejects_non_strings() -> None:
    """error_type wins over category; a non-string category never crosses."""
    assert kw._attempt_category({"error_type": "timeout", "category": "other"}) == "timeout"
    assert kw._attempt_category({"category": "timeout"}) == "timeout"
    assert kw._attempt_category({"error_type": 7, "category": "timeout"}) == "tool_error"
    assert kw._attempt_category({"error_type": "", "category": "timeout"}) == "tool_error"
    assert kw._attempt_category({}) == "tool_error"


def test_attempt_reasoning_survives_projection() -> None:
    """A string reasoning reaches the tool execution row; a boolean confidence reads as None."""
    row = kw._run_attempt_row(0, {"tool": "t", "arguments": {}, "confidence": True, "reasoning": "kept"})
    assert row["reasoning"] == "kept"
    assert row["confidence"] is None


def test_run_level_guard_alone_escalates_response() -> None:
    """A run-level guard admits nothing but still escalates; only True counts."""
    assert kw._run_attempts({"incomplete_guard": True}, []) == ([], True)
    assert kw._run_attempts({"incomplete_guard": 1}, []) == ([], False)
    assert kw._run_attempts({}, [{"incomplete_guard": False}]) == ([], False)


def test_call_row_keeps_camel_evidence_id_only_when_known() -> None:
    """The camelCase claim links only when the id is already stored."""
    call, category = kw._run_call_row(0, {"tool": "t", "evidenceId": "known-id"}, {"known-id"})
    assert call.get("evidenceId") == "known-id" and category == ""
    ghost, _ = kw._run_call_row(1, {"tool": "t", "evidenceId": "ghost"}, {"known-id"})
    assert "evidenceId" not in ghost


def test_call_row_error_text_bounded_at_500() -> None:
    """A long tool error is clipped to exactly 500 chars in the call row."""
    call, category = kw._run_call_row(0, {"tool": "t", "error": "x" * 600}, set())
    assert call["error"] == "x" * 500 and category == "tool_error"


def test_run_deadlines_keep_exact_wall_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 100s wall splits setup and scheduler deadlines exactly; the default wall keeps its floor."""
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 1_000.0)
    monkeypatch.setattr(_time, "perf_counter", lambda: 500.0)
    started, setup, run = kw._run_deadlines({"deadlineAt": 1_100_000.0})
    assert (started, setup, run) == (500.0, 530.0, 580.0)
    started, setup, run = kw._run_deadlines({})
    assert (started, setup, run) == (500.0, 550.0, 600.0)


def test_run_kwargs_pins_round_cap_deadline_and_selector() -> None:
    """The scheduler receives the 4-round cap, the run deadline, the JEV client, and the exact round."""
    from app.research.programmatic_router import programmatic_select_round

    jev = JevClient()
    kwargs = kw._run_kwargs(jev, None, programmatic_select_round, 123.0)
    assert kwargs["max_rounds"] == 4 and kwargs["deadline_at"] == 123.0
    assert kwargs["select_round"] is programmatic_select_round and kwargs["jev"] is jev
    plain = kw._run_kwargs(jev, None, None, 123.0)
    assert "select_round" not in plain and plain["max_rounds"] == 4


def test_node_results_drop_non_list_schedulers() -> None:
    """A non-list scheduler nodes field reads as no node results, never iterated."""
    assert kw._run_node_results({"nodes": {"a": 1}}) == []
    assert kw._run_node_results({"nodes": None}) == []
    assert kw._run_node_results({}) == []


def test_evidence_row_without_id_gets_its_index() -> None:
    """A stored row without an evidence id is wired by its ev-index, and the id is known after."""
    rows = kw._run_evidence([{"content": "x"}, {"content": "y"}])
    assert [row["id"] for row in rows[0] if isinstance(row, dict)] == ["ev-0", "ev-1"]
    assert rows[1] == {"ev-0", "ev-1"}


def test_run_incomplete_guard_escalates_without_unresolved_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guard escalates a run even when no stored node remains unresolved."""
    _empty_repo(monkeypatch)
    out = _run_with(monkeypatch, {"status": "incomplete_guard", "nodes": [], "incomplete_guard": True})
    assert out["unresolved"] == []
    assert out["incomplete_guard"] is True
    assert out["escalated"] is True and out["escalations"] == 1


def test_run_stall_adds_to_the_tool_failure_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session stall adds one failure without discarding the tool's own stalled outcome."""
    _empty_repo(monkeypatch)
    out = _run_with(
        monkeypatch,
        {
            "status": "stalled",
            "nodes": [{"attempts": [{"tool": "search_web", "error": "no progress", "error_type": "stalled"}]}],
        },
    )
    assert out["failures"] == {"stalled": 2}
    assert out["stalled"] is True and out["escalated"] is True


def test_pass2_denial_keeps_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown pass-2 session answers invalid_params under the caller's own id."""
    import tempfile

    monkeypatch.setenv("RESEARCH_DB_PATH", str(Path(tempfile.mkdtemp()) / "r.sqlite"))
    out = kw._run({"id": "r9", "prompt": "q?", "sessionId": "rs:nope", "gap": "need doc X"}, jev=JevClient())
    assert out["id"] == "r9"
    terminal = out["terminal"]
    assert isinstance(terminal, dict) and terminal["category"] == "invalid_params"
    assert "unknown session" in str(out["error"])


def test_scope_denial_keeps_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed scope answers invalid_params under the caller's own id before any setup."""

    def _boom(*a: object, **k: object) -> str:
        raise AssertionError("setup must not run for a malformed scope")

    monkeypatch.setattr(kw, "run_graph_prompt", _boom)
    out = kw._run({"id": "rx", "prompt": "q?", "ticker": 7}, jev=JevClient())
    assert out["id"] == "rx"
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal["category"] == "invalid_params"
    assert "ticker" in str(out.get("error"))


def test_setup_failure_keeps_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising pass-2 setup reports provider_error under the caller's own id."""

    def _boom(session_id: str, gap: str) -> tuple[str, list[str]]:
        raise RuntimeError("setup blew up")

    monkeypatch.setattr(kw, "_followup_gap_nodes", _boom)
    out = kw._run({"id": "rf", "prompt": "q?", "sessionId": "rs:1", "gap": "need doc X"}, jev=JevClient())
    assert out["id"] == "rf"
    terminal = out.get("terminal")
    assert isinstance(terminal, dict) and terminal["category"] == "provider_error"
    assert terminal["message"] == out.get("error")


def test_registry_guard_error_keeps_exact_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guard refusal crosses verbatim instead of the session-setup prefix."""

    def _guard(*a: object, **k: object) -> str:
        raise RuntimeError("registry guard forbids portfolio tools: x")

    monkeypatch.setattr(kw, "run_graph_prompt", _guard)
    sid, denied = kw._run_graph_session("rg", "q?", None, JevClient(), None, 1.0, None, None)
    assert sid == "" and denied is not None
    denied_error = denied.get("error")
    assert denied["id"] == "rg" and denied_error == "registry guard forbids portfolio tools: x"
    assert denied.get("terminal") == {"category": "provider_error", "message": denied_error}


def test_setup_failure_carries_session_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-guard setup failure carries the session-setup prefix under the caller's id."""

    def _other(*a: object, **k: object) -> str:
        raise RuntimeError("store down")

    monkeypatch.setattr(kw, "run_graph_prompt", _other)
    _, denied2 = kw._run_graph_session("rg", "q?", None, JevClient(), None, 1.0, None, None)
    denied2_error = denied2.get("error") if isinstance(denied2, dict) else None
    assert denied2_error == "session setup failed: store down"
    assert isinstance(denied2, dict) and denied2.get("terminal") == {
        "category": "provider_error",
        "message": denied2_error,
    }


def test_graph_session_forwards_scope_client_and_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """as_of, the JEV client, the setup deadline, and the ticker/watchlist scope all reach the session."""
    seen: dict[str, object] = {}

    def _fake(prompt: str, as_of: object = None, **k: object) -> str:
        seen.update(prompt=prompt, as_of=as_of, **k)
        return "s9"

    monkeypatch.setattr(kw, "run_graph_prompt", _fake)
    jev = JevClient()
    sid, denied = kw._run_graph_session("rf", "q?", "2024-01-01", jev, None, 7.0, "NVDA", None)
    assert (sid, denied) == ("s9", None)
    assert seen["as_of"] == "2024-01-01" and seen["jev"] is jev and seen["setup_deadline"] == 7.0
    assert seen["ticker"] == "NVDA" and seen["watchlist"] is None
    _, _ = kw._run_graph_session("rf", "q?", None, jev, None, 7.0, "NVDA", ["NVDA"])
    assert seen["ticker"] is None and seen["watchlist"] == ["NVDA"]


def test_run_session_forwards_scope_to_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new request carries as_of, the JEV client, and the setup deadline into the graph session."""
    seen: dict[str, object] = {}

    def _fake_graph(
        rid: str,
        objective: str,
        as_of: object,
        client: object,
        progress: object,
        setup_deadline: float,
        run_ticker: object,
        watchlist: object,
    ) -> tuple[str, None]:
        seen.update(as_of=as_of, client=client, setup_deadline=setup_deadline)
        return "s2", None

    monkeypatch.setattr(kw, "_run_graph_session", _fake_graph)
    jev = JevClient()
    out = kw._run_session("rs", {"prompt": "q?"}, "q?", "2024-05-01", jev, None, 9.0, "NVDA", None)
    assert out == ("s2", [], None)
    assert seen == {"as_of": "2024-05-01", "client": jev, "setup_deadline": 9.0}


def test_run_forwards_asof_and_injected_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    """asOf reaches setup verbatim (a non-string reads as absent) and the injected client drives the run."""
    seen: dict[str, object] = {}

    def _fake_session(
        rid: str,
        req: object,
        objective: str,
        as_of: object,
        client: object,
        progress: object,
        setup_deadline: float,
        run_ticker: object,
        watchlist: object,
    ) -> tuple[str, list[str], None]:
        seen["as_of"] = as_of
        seen["client"] = client
        return "s3", [], None

    async def _sched(sid: str, **hooks: object) -> dict[str, object]:
        seen.update(hooks)
        seen["sid"] = sid
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(kw, "_run_session", _fake_session)
    monkeypatch.setattr("app.research.scheduler.run", _sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    _empty_repo(monkeypatch)
    jev = JevClient()
    out = kw._run({"id": "ra", "prompt": "q?", "asOf": "2024-05-01"}, jev=jev)
    assert seen["as_of"] == "2024-05-01" and seen["client"] is jev and seen["jev"] is jev
    assert out["sessionId"] == "s3"
    kw._run({"id": "rb", "prompt": "q?", "asOf": 7}, jev=jev)
    assert seen["as_of"] is None and seen["client"] is jev


def test_stalled_reason_contract() -> None:
    """A stalled run keeps its bounded reason; anything else reports the fixed stall text."""
    assert kw._run_stalled({"status": "stalled", "reason": "waiting"}) == (True, "waiting")
    assert kw._run_stalled({"status": "stalled"}) == (True, "stalled: no progress")
    assert kw._run_stalled({"status": "stalled", "reason": 7}) == (True, "stalled: no progress")
    assert kw._run_stalled({"status": "complete", "reason": "waiting"}) == (False, None)
    stalled, reason = kw._run_stalled({"status": "stalled", "reason": "x" * 600})
    assert stalled is True and reason == "x" * 500


def test_pass2_response_gates_only_gap_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """A follow-up escalates only on its own gap node; pass-1 states never gate the writer again."""

    async def _sched(sid: str, **hooks: object) -> dict[str, object]:
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    def _ev(self: ResearchRepository, sid: str) -> list[dict[str, JSONValue]]:
        return []

    def _nodes(self: ResearchRepository, sid: str) -> list[ResearchNode]:
        return [
            ResearchNode(node_id="n1", session_id="s1", question="q1?", why_it_matters="w", status="gathering"),
            ResearchNode(node_id="gap1", session_id="s1", question="q2?", why_it_matters="w", status="gathering"),
        ]

    def _dec(self: ResearchRepository, sid: str) -> list[DecisionRecord]:
        return []

    monkeypatch.setattr("app.research.scheduler.run", _sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)

    def _session(*a: object) -> tuple[str, list[str], None]:
        return ("s1", ["gap1"], None)

    monkeypatch.setattr(kw, "_run_session", _session)
    monkeypatch.setattr(ResearchRepository, "list_evidence", _ev)
    monkeypatch.setattr(ResearchRepository, "list_nodes", _nodes)
    monkeypatch.setattr(ResearchRepository, "list_decisions", _dec)
    out = kw._run({"id": "rg", "prompt": "q?", "sessionId": "s1", "gap": "x"}, jev=JevClient())
    assert out["unresolved"] == ["gap1"] and out["escalated"] is True and out["escalations"] == 1


def test_escalation_needs_any_unresolved_guard_or_stall() -> None:
    """One unresolved node, one guard, or one stall escalates; a clean run stays quiet."""
    assert kw._run_escalation({}, ["n"], False, False)[1:] == (1, True)
    assert kw._run_escalation({}, [], True, False)[1:] == (1, True)
    assert kw._run_escalation({}, [], False, True) == ({"stalled": 1}, 1, True)
    assert kw._run_escalation({}, [], False, False) == ({}, 0, False)


def test_run_result_reads_session_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evidence, nodes, and decisions are read under the run's own session id."""
    seen: dict[str, list[str]] = {"evidence": [], "nodes": [], "decisions": []}

    def _ev(self: ResearchRepository, sid: str) -> list[dict[str, JSONValue]]:
        seen["evidence"].append(sid)
        return [{"content": "x", "evidence_id": "e1"}]

    def _nodes(self: ResearchRepository, sid: str) -> list[ResearchNode]:
        seen["nodes"].append(sid)
        return []

    def _dec(self: ResearchRepository, sid: str) -> list[DecisionRecord]:
        seen["decisions"].append(sid)
        return []

    async def _sched(sid: str, **hooks: object) -> dict[str, object]:
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(ResearchRepository, "list_evidence", _ev)
    monkeypatch.setattr(ResearchRepository, "list_nodes", _nodes)
    monkeypatch.setattr(ResearchRepository, "list_decisions", _dec)
    out = _run_with(monkeypatch, {"status": "complete", "nodes": [], "incomplete_guard": False})
    assert seen == {"evidence": ["s1"], "nodes": ["s1"], "decisions": ["s1"]}
    assert out["evidence"] == [{"id": "e1", "content": "x"}]


def test_select_round_flows_return_no_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """full, whole, and direct keep the whole-registry escape hatch: no round selector at all."""
    for flow in ("full", "whole", "direct"):
        monkeypatch.setenv("STOCKBOT_TOOLFLOW", flow)
        assert kw._run_select_round() is None


def test_shutdown_pools_drop_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown drops queued pool work without blocking and still cancels queued futures."""
    from app.research import scheduler as _sched

    seen: list[dict[str, object]] = []

    class _Pool:
        def shutdown(self, **kwargs: object) -> None:
            seen.append(dict(kwargs))

    monkeypatch.setattr(_sched, "_SEC_POOL", _Pool())
    monkeypatch.setattr(_sched, "_TOOL_POOL", _Pool())
    kw._shutdown_pools()
    assert seen == [{"wait": False, "cancel_futures": True}] * 2


def test_graph_prompt_forwards_intake_scope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The graph entry carries the objective, as_of, session, registry, JEV client, and kernel into intake."""
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    seen: dict[str, object] = {}

    async def _fake_intake(
        objective: str,
        as_of: str | None,
        sid: str,
        session: dict[str, JSONValue],
        registry: list[dict[str, JSONValue]],
        client: object,
        kernel: object,
        progress: object,
        setup_deadline: float | None,
        watchlist: list[str] | None,
    ) -> list[dict[str, object]]:
        seen.update(
            objective=objective,
            as_of=as_of,
            sid=sid,
            session=session,
            registry=registry,
            client=client,
            kernel=kernel,
            setup_deadline=setup_deadline,
            watchlist=watchlist,
        )
        return []

    monkeypatch.setattr(kw, "_graph_intake", _fake_intake)
    monkeypatch.setattr("app.research.scheduler.build_registry", lambda: [{"name": "t"}])
    jev = JevClient()
    sid = kw.run_graph_prompt("What is NVDA?", as_of="2024-01-01", jev=jev, setup_deadline=9.0, ticker="NVDA")
    assert seen["objective"] == "What is NVDA?" and seen["as_of"] == "2024-01-01"
    assert seen["sid"] == sid and seen["client"] is jev and seen["kernel"] is not None
    assert isinstance(seen["session"], dict) and seen["session"].get("session_id") == sid
    assert seen["registry"] == [{"name": "t"}] and seen["setup_deadline"] == 9.0


# --- Staged worker guard blocks: intake, protocol, reasoner (41 tests) ---


def test_graph_intake_round_trip_returns_reasoner_proposals_and_logs_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The graph path fetches pending scope once, returns Reasoner proposals, and logs one digest."""
    import asyncio

    from app.research import scheduler as _sched
    from app.research.models import DecisionRecord, new_decision_id
    from app.runtime import ToolResultMeta
    from app.tool_runtime import ToolOutcome

    class _GraphKernel(_sched._Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.jobs = 0
            self.store: dict[str, list[dict[str, JSONValue]]] = {}
            self.decisions: list[str] = []
            self.tool_results: dict[str, dict[str, JSONValue]] = {
                "s1:tr:w": {
                    "tool_result_id": "s1:tr:w",
                    "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
                },
            }

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
            self.jobs += 1
            return {"job_id": f"job-{self.jobs}"}

        @override
        def heartbeat_job(self, job_id: str) -> None:
            return None

        @override
        def complete_job(self, job_id: str, outcome: object = None) -> None:
            return None

        @override
        def fail_job(self, job_id: str, category: str, message: str) -> None:
            return None

        @override
        def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
            rows = self.store.setdefault(session_id, [])
            eid = f"ev-{len(rows)}"
            rows.append({"evidence_id": eid})
            return {"evidence_id": eid}

        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [dict(row) for row in self.store.get(session_id, [])]

        @override
        def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
            return self.tool_results[tool_result_id]

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
            self.decisions.append(decision_type)
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

    def _outcome(content: str) -> ToolOutcome:
        return ToolOutcome(
            tool_name="t",
            content=content,
            source_handle=None,
            source_refs=None,
            error=None,
            error_type=None,
            retryable=False,
            meta=ToolResultMeta(0, None, False, None, [], {}),
        )

    async def _attempt(
        tool: str,
        args: dict[str, object],
        sid: str,
        session: dict[str, object],
        registry: object,
        jev: object,
        kernel: _sched._Kernel,
        as_of: str | None,
    ) -> dict[str, object]:
        if sid != "s1":
            raise KeyError(sid)
        objective = session.get("objective")
        start = kernel.start_job(sid, source="WEB")
        job_id = start["job_id"]
        assert isinstance(job_id, str)
        if tool == "search_web":
            return {
                "tool": tool,
                "arguments": dict(args),
                "job_id": job_id,
                "result": {
                    "tool_result_id": "s1:tr:w",
                    "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
                },
                "outcome": _outcome(f"web bytes for {objective}"),
                "outcome_summary": f"web bytes for {objective}",
                "error": None,
            }
        return {"tool": tool, "arguments": dict(args), "job_id": job_id, "error": "boom", "outcome": None}

    proposals: list[dict[str, JSONValue]] = [
        {"id": "s-q1", "objectiveId": "s", "question": "q?", "dependsOn": [], "whyItMatters": "w"}
    ]

    async def _scenario() -> None:
        kernel = _GraphKernel()
        monkeypatch.setattr(kw, "_intake_attempt", _attempt)

        def _stub(*a: object, **k: object) -> tuple[list[dict[str, JSONValue]], dict[str, JSONValue]]:
            return (proposals, {})

        monkeypatch.setattr(kw, "_reasoner_decompose_with_retry", _stub)
        objective = "What is happening with Oracle now"
        out = await kw._graph_intake(
            objective, None, "s1", {"session_id": "s1", "objective": objective}, [], None, kernel, None, None, ["NVDA"]
        )
        assert out == proposals
        assert kernel.decisions == ["intake_digest"]
        assert [row["evidence_id"] for row in kernel.store.get("s1", [])] == ["ev-0"]

    asyncio.run(_scenario())


def test_intake_round_trip_admits_scoped_evidence_with_full_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    """One real intake round admits session-scoped SEC + web + FINRA bytes with full stats."""
    import asyncio

    from app.research import scheduler as _sched
    from app.runtime import ToolResultMeta
    from app.tool_runtime import ToolOutcome

    class _IntakeRoundKernel(_sched._Kernel):
        def __init__(self) -> None:
            super().__init__()
            self.jobs = 0
            self.completed: list[str] = []
            self.store: dict[str, list[dict[str, JSONValue]]] = {}
            self.tool_results: dict[str, dict[str, JSONValue]] = {
                "s1:tr:w": {
                    "tool_result_id": "s1:tr:w",
                    "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
                },
                "s1:tr:f": {
                    "tool_result_id": "s1:tr:f",
                    "records": [
                        {
                            "settlementDate": "2024-01-01",
                            "symbolCode": "NVDA",
                            "currentShortPositionQuantity": 123,
                        }
                    ],
                    "briefing": "B",
                },
                "s1:tr:l": {
                    "tool_result_id": "s1:tr:l",
                    "filings": [{"form": "10-Q", "accession_no": "0000000000-26-000001", "title": "Q3"}],
                },
            }

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
            self.jobs += 1
            return {"job_id": f"job-{self.jobs}"}

        @override
        def heartbeat_job(self, job_id: str) -> None:
            return None

        @override
        def complete_job(self, job_id: str, outcome: object = None) -> None:
            self.completed.append(job_id)

        @override
        def fail_job(self, job_id: str, category: str, message: str) -> None:
            return None

        @override
        def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
            rows = self.store.setdefault(session_id, [])
            eid = f"ev-{len(rows)}"
            content = data.get("content") if isinstance(data, dict) else None
            rows.append({"evidence_id": eid, "content": content if isinstance(content, str) else None})
            return {"evidence_id": eid}

        @override
        def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
            return [dict(row) for row in self.store.get(session_id, [])]

        @override
        def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
            return self.tool_results[tool_result_id]

    def _outcome(content: str) -> ToolOutcome:
        return ToolOutcome(
            tool_name="t",
            content=content,
            source_handle=None,
            source_refs=None,
            error=None,
            error_type=None,
            retryable=False,
            meta=ToolResultMeta(0, None, False, None, [], {}),
        )

    async def _attempt(
        tool: str,
        args: dict[str, object],
        sid: str,
        session: dict[str, object],
        registry: object,
        jev: object,
        kernel: _sched._Kernel,
        as_of: str | None,
    ) -> dict[str, object]:
        if sid != "s1":
            raise KeyError(sid)
        objective = session.get("objective")
        start = kernel.start_job(sid, source="WEB")
        job_id = start["job_id"]
        assert isinstance(job_id, str)
        if tool == "search_web":
            return {
                "tool": tool,
                "arguments": dict(args),
                "job_id": job_id,
                "result": {
                    "tool_result_id": "s1:tr:w",
                    "evidence": [{"url": "https://example.com/a", "highlight": "Oracle merger filing mentions"}],
                },
                "outcome": _outcome(f"web bytes for {objective} asof={as_of}"),
                "outcome_summary": f"web bytes for {objective} asof={as_of}",
                "error": None,
            }
        if tool == "get_reg_sho_volume":
            return {
                "tool": tool,
                "arguments": dict(args),
                "job_id": job_id,
                "result": {
                    "tool_result_id": "s1:tr:f",
                    "records": [
                        {"settlementDate": "2024-01-01", "symbolCode": "NVDA", "currentShortPositionQuantity": 123}
                    ],
                    "briefing": "B",
                },
                "outcome": _outcome(f"sho bytes asof={as_of}"),
                "outcome_summary": f"sho bytes asof={as_of}",
                "error": None,
            }
        if tool == "list_sec_filings":
            return {
                "tool": tool,
                "arguments": dict(args),
                "job_id": job_id,
                "result": {
                    "tool_result_id": "s1:tr:l",
                    "filings": [{"form": "10-Q", "accession_no": "0000000000-26-000001", "title": "Q3"}],
                },
                "outcome": _outcome(f"list bytes asof={as_of}"),
                "outcome_summary": f"list bytes asof={as_of}",
                "error": None,
            }
        return {"tool": tool, "arguments": dict(args), "job_id": job_id, "error": "boom", "outcome": None}

    async def _scenario() -> None:
        kernel = _IntakeRoundKernel()
        monkeypatch.setattr(kw, "_intake_attempt", _attempt)
        objective = "What is happening with Oracle now"
        session: dict[str, JSONValue] = {"session_id": "s1", "objective": objective}
        admitted, _raw, stats = await kw._intake_round(
            objective, ["ORCL"], "s1", session, [], None, kernel, "2026-10-01", None, ["NVDA"]
        )
        assert stats["outcomes"] == {"admitted": 4, "duplicate": 0, "not_citable": 0, "error": 2, "timeout": 0}
        assert stats["admitted_ids"] == ["ev-0", "ev-1", "ev-2", "ev-3"] and stats["admitted"] == 4
        assert {row["evidence_id"] for row in admitted} == {"ev-0", "ev-1", "ev-2", "ev-3"}
        by_id = {str(row["evidence_id"]): str(row["content"]) for row in admitted}
        assert "2026-10-01" in by_id["ev-0"]
        assert objective in by_id["ev-1"] and "2026-10-01" in by_id["ev-1"]
        assert "2026-10-01" in by_id["ev-2"] and "2026-10-01" in by_id["ev-3"]
        assert set(stats) >= {
            "calls",
            "admitted",
            "admitted_ids",
            "outcomes",
            "ms",
            "wall_ms",
            "settle_ms",
            "per_source",
            "sec_ms",
            "web_ms",
        }
        assert isinstance(stats["ms"], float) and isinstance(stats["settle_ms"], float)
        assert isinstance(stats["wall_ms"], float) and stats["wall_ms"] >= 0.0
        assert isinstance(stats["sec_ms"], float) and isinstance(stats["web_ms"], float)
        watch = stats["watchlist"]
        assert isinstance(watch, list) and len(watch) == 4
        assert sum(1 for row in watch if isinstance(row, dict) and row.get("status") == "ok") == 2
        assert len(kernel.completed) == 4

    asyncio.run(_scenario())


def test_intake_settle_keeps_unsettled_watch_row_and_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsettled call keeps its own tool/args timeout row plus wall milliseconds."""
    import asyncio
    import time as _time

    async def _scenario() -> None:
        async def _call() -> kw._IntakeCallRow:
            return ("search_web", {"query": "q?"}, {"tool": "search_web"}, 1.0)

        async def _hang() -> kw._IntakeChainRows:
            await asyncio.Event().wait()
            return []

        monkeypatch.setattr(kw, "_INTAKE_BUDGET_S", 0.05)
        call = asyncio.ensure_future(_call())
        hang = asyncio.ensure_future(_hang())
        await asyncio.sleep(0.1)
        meta: dict[int, tuple[str, dict[str, object]]] = {
            id(call): ("search_web", {"query": "q?"}),
            id(hang): ("get_reg_sho_volume", {"ticker": "NVDA"}),
        }
        ticks = iter([100.0, 100.2])
        monkeypatch.setattr(_time, "perf_counter", lambda: next(ticks))
        try:
            flat, watch_records, wall_ms = await asyncio.wait_for(kw._intake_settle([call, hang], meta), timeout=1.0)
        finally:
            call.cancel()
            hang.cancel()
            await asyncio.gather(call, hang, return_exceptions=True)
        assert wall_ms == pytest.approx(200.0)
        assert ("search_web", {"query": "q?"}, {"tool": "search_web"}, 1.0) in flat
        late = [row for row in flat if row[0] == "get_reg_sho_volume"]
        assert len(late) == 1 and late[0][1] == {"ticker": "NVDA"}
        assert late[0][2] == {
            "tool": "get_reg_sho_volume",
            "arguments": {"ticker": "NVDA"},
            "error": "intake_timeout",
            "error_type": "timeout",
        }
        assert late[0][3] == 0.0
        assert watch_records[id(call)] == {"tool": "search_web"}
        assert watch_records[id(hang)] == late[0][2]
        assert hang.cancelled()

    asyncio.run(_scenario())


def test_intake_admit_all_accumulates_settle_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every admit adds its own cost: two calls settle near 400ms on a stepped clock."""
    import time as _time

    kernel = sched._Kernel()

    def _settle(
        kernel: sched._Kernel, attempt: Mapping[str, object], sid: str = "-", nid: str = "-"
    ) -> tuple[dict[str, JSONValue] | None, bool]:
        return None, False

    def _complete(kernel: sched._Kernel, attempt: Mapping[str, object], tool: str) -> None:
        return None

    def _cand(
        tool_name: str, domain: str, result: object, outcome: object, kernel: sched._Kernel
    ) -> dict[str, object] | None:
        return None

    monkeypatch.setattr(sched, "_settle_attempt", _settle)
    monkeypatch.setattr(sched, "_complete_attempt_job", _complete)
    monkeypatch.setattr(sched, "_evidence_candidate", _cand)
    ticks = iter([10.0, 10.1, 20.0, 20.3])
    monkeypatch.setattr(_time, "perf_counter", lambda: next(ticks))
    calls: list[tuple[str, dict[str, object]]] = [
        ("list_sec_filings", {"identifier": "ORCL"}),
        ("search_web", {"query": "q?"}),
    ]
    timed: list[tuple[dict[str, object], float]] = [
        ({"tool": "list_sec_filings", "arguments": {}, "error": "boom", "job_id": "j1"}, 1.0),
        ({"tool": "search_web", "arguments": {}, "error": "x", "error_type": "timeout"}, 0.0),
    ]
    _raw, outcomes, admitted_ids, settle_ms = kw._intake_admit_all(kernel, "s1", calls, timed)
    assert outcomes == {"admitted": 0, "duplicate": 0, "not_citable": 0, "error": 1, "timeout": 1}
    assert admitted_ids == []
    assert settle_ms == pytest.approx(400.0)


def test_intake_timed_reports_elapsed_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0.5s call reports 500ms taken alongside its tool, args, and record."""
    import asyncio
    import time as _time

    async def _fast(*a: object, **k: object) -> dict[str, object]:
        return {"tool": "search_web", "arguments": {}, "job_id": "j"}

    async def _scenario() -> tuple[str, dict[str, object], dict[str, object], float]:
        monkeypatch.setattr(kw, "_intake_attempt", _fast)
        ticks = iter([50.0, 50.5])
        monkeypatch.setattr(_time, "perf_counter", lambda: next(ticks))
        return await kw._intake_timed("search_web", {"query": "q?"}, "s1", {}, [], None, sched._Kernel(), None)

    tool, args, record, taken = asyncio.run(_scenario())
    assert (tool, args) == ("search_web", {"query": "q?"})
    assert record["job_id"] == "j"
    assert taken == pytest.approx(500.0)


def test_intake_round_timeout_row_keeps_request_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A web call past the budget reports a timeout row that still names the request query."""
    import asyncio

    from app.research import scheduler as _sched

    async def _hang(*a: object, **k: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def _scenario() -> tuple[list[dict[str, JSONValue]], list[dict[str, object]], dict[str, object]]:
        monkeypatch.setattr(kw, "_intake_attempt", _hang)
        monkeypatch.setattr(kw, "_INTAKE_BUDGET_S", 0.05)

        def _load(sid: str, kernel: _sched._Kernel) -> list[dict[str, JSONValue]]:
            return []

        monkeypatch.setattr(_sched, "_load_evidence", _load)
        session: dict[str, JSONValue] = {"session_id": "s1"}
        registry: list[dict[str, JSONValue]] = []
        tickers: list[str] = []
        return await kw._intake_round("my question?", tickers, "s1", session, registry, None, _sched._Kernel(), None)

    _admitted, raw, stats = asyncio.run(_scenario())
    assert stats["outcomes"] == {"admitted": 0, "duplicate": 0, "not_citable": 0, "error": 0, "timeout": 1}
    assert [(row["tool"], row["arguments"]) for row in raw] == [("search_web", {"query": "my question?"})]
    assert raw[0]["record"] == {
        "tool": "search_web",
        "arguments": {"query": "my question?"},
        "outcome_summary": None,
        "error": "intake_timeout",
        "error_type": "timeout",
        "evidence_id": None,
        "job_id": None,
    }


def test_intake_round_chain_timeout_fails_fast_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung ticker chain trips the per-call ceiling long before the budget wait ends."""
    import asyncio

    from app.research import scheduler as _sched

    async def _hang_chain(*a: object, **k: object) -> kw._IntakeChainRows:
        await asyncio.sleep(60)
        return []

    async def _scenario() -> dict[str, object]:
        monkeypatch.setattr(kw, "_intake_ticker_chain", _hang_chain)
        monkeypatch.setattr(kw, "_INTAKE_CALL_TIMEOUT_S", 0.05)
        monkeypatch.setattr(kw, "_INTAKE_BUDGET_S", 0.3)

        def _load(sid: str, kernel: _sched._Kernel) -> list[dict[str, JSONValue]]:
            return []

        monkeypatch.setattr(_sched, "_load_evidence", _load)
        session: dict[str, JSONValue] = {"session_id": "s1"}
        registry: list[dict[str, JSONValue]] = []
        _admitted, _raw, stats = await kw._intake_round(
            None, ["ORCL"], "s1", session, registry, None, _sched._Kernel(), None
        )
        return stats

    stats = asyncio.run(_scenario())
    assert stats["outcomes"] == {"admitted": 0, "duplicate": 0, "not_citable": 0, "error": 0, "timeout": 1}
    assert isinstance(stats["wall_ms"], float) and stats["wall_ms"] < 200.0


def test_intake_admit_all_counts_every_uncitable_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two uncitable answers count not_citable twice, never collapse to one."""
    kernel = sched._Kernel()

    def _settle(
        kernel: sched._Kernel, attempt: Mapping[str, object], sid: str = "-", nid: str = "-"
    ) -> tuple[dict[str, JSONValue] | None, bool]:
        return None, False

    def _complete(kernel: sched._Kernel, attempt: Mapping[str, object], tool: str) -> None:
        return None

    def _cand(
        tool_name: str, domain: str, result: object, outcome: object, kernel: sched._Kernel
    ) -> dict[str, object] | None:
        return None

    monkeypatch.setattr(sched, "_settle_attempt", _settle)
    monkeypatch.setattr(sched, "_complete_attempt_job", _complete)
    monkeypatch.setattr(sched, "_evidence_candidate", _cand)
    calls: list[tuple[str, dict[str, object]]] = [
        ("list_sec_filings", {"identifier": "ORCL"}),
        ("search_web", {"query": "q?"}),
    ]
    timed: list[tuple[dict[str, object], float]] = [
        ({"tool": "list_sec_filings", "arguments": {}, "outcome": {"ok": True}, "result": {"x": 1}}, 1.0),
        ({"tool": "search_web", "arguments": {}, "outcome": {"ok": True}, "result": {"y": 2}}, 2.0),
    ]
    _raw, outcomes, admitted_ids, _ms = kw._intake_admit_all(kernel, "s1", calls, timed)
    assert outcomes == {"admitted": 0, "duplicate": 0, "not_citable": 2, "error": 0, "timeout": 0}
    assert admitted_ids == []


def test_intake_admit_call_completes_attempt_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """An admitted call completes its kernel job; the job never stays open."""
    completed: list[str] = []

    class _CompletingKernel(sched._Kernel):
        @override
        def complete_job(self, job_id: str, outcome: object = None) -> None:
            completed.append(job_id)

    def _candidate(*a: object, **k: object) -> dict[str, bool]:
        return {"candidate": True}

    monkeypatch.setattr(sched, "_evidence_candidate", _candidate)

    def _persist(
        kernel: sched._Kernel, sid: str, attempt: Mapping[str, object], candidate: Mapping[str, object]
    ) -> tuple[str, bool]:
        return ("ev-1", True)

    monkeypatch.setattr(sched, "_persist_admitted_evidence", _persist)
    outcomes = {"admitted": 0, "duplicate": 0, "not_citable": 0, "error": 0, "timeout": 0}
    kw._intake_admit_call(
        _CompletingKernel(),
        "s1",
        "search_web",
        {"tool": "search_web", "arguments": {}, "outcome": {"ok": True}, "result": {"x": 1}, "job_id": "job-7"},
        outcomes,
        [],
    )
    assert outcomes["admitted"] == 1
    assert completed == ["job-7"]


def test_arguments_caller_context_scope_reaches_needle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller temporal scope and as_of reach Needle; raw evidence never does."""
    seen: list[dict[str, object]] = []

    def _gen(**kwargs: object) -> dict[str, object]:
        seen.append(dict(kwargs))
        return {"tool": "list_sec_filings", "arguments": {"identifier": "NVDA"}, "confidence": 0.9, "reasoning": "r"}

    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)
    out = kw._arguments(
        {
            "id": "c1",
            "op": "arguments",
            "tool": "list_sec_filings",
            "objective": "What drove NVDA revenue last quarter?",
            "context": {
                "temporal_scope": {"mode": "as_of", "as_of": "2026-06-30"},
                "as_of": "2026-06-30",
                "evidence": [{"evidence_id": "ev-1", "content": "raw"}],
            },
            "schema": {"type": "object"},
        }
    )
    assert out["id"] == "c1" and out["tool"] == "list_sec_filings"
    context = seen[0]["context"]
    assert isinstance(context, dict)
    assert context.get("as_of") == "2026-06-30"
    assert context.get("temporal_scope") == {"mode": "as_of", "as_of": "2026-06-30"}
    assert "evidence" not in context


def test_arguments_error_envelopes_keep_500_char_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both error envelopes truncate at 500 chars, never 501."""

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("x" * 600)

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    outer = kw._arguments({"id": "e1", "tool": "list_sec_filings", "objective": "NVDA revenue?"})
    assert isinstance(outer["error"], str) and len(outer["error"]) == 500
    ticker = kw._arguments({"id": "e2", "tool": "list_sec_filings", "objective": "q?", "ticker": "X" * 600})
    assert isinstance(ticker["error"], str) and len(ticker["error"]) == 500
    assert "ticker" in str(ticker["error"])


def test_arguments_needle_receives_exact_jev_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needle generate is called with the exact JEV-selected tool."""
    seen: list[dict[str, object]] = []

    def _gen(**kwargs: object) -> dict[str, object]:
        seen.append(dict(kwargs))
        return {"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "r"}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)

    def _schema(tool_name: str, registry: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
        return {"type": "object"}

    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    out = kw._arguments({"id": "t1", "tool": "query_finra", "objective": "short interest?"})
    assert out["tool"] == "query_finra"
    assert seen and seen[0].get("tool") == "query_finra"


def test_arguments_prompt_grounds_when_objective_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt-only request grounds like its objective; without either, the objective is empty."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)

    def _withhold(**kwargs: object) -> object:
        return None

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _withhold)
    seeded = kw._arguments({"id": "p1", "tool": "list_sec_filings", "prompt": "What drove NVDA revenue last quarter?"})
    assert seeded["arguments"] == {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]}


def test_arguments_non_string_objective_grounds_empty() -> None:
    """A non-string objective never leaks into grounding; it reads as empty."""
    assert kw._arguments_objective({"objective": 7}) == ""
    assert kw._arguments_objective({"objective": None, "prompt": 7}) == ""


def test_arguments_missing_needle_tool_defaults_to_jev_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Needle payload without a tool key still grounds the JEV-selected tool."""

    def _gen(**kwargs: object) -> dict[str, object]:
        return {"arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "r"}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)

    def _schema(tool_name: str, registry: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
        return {"type": "object"}

    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    out = kw._arguments({"id": "m1", "tool": "query_finra", "objective": "short interest?"})
    assert out["tool"] == "query_finra" and "error" not in out


def test_arguments_non_mapping_needle_arguments_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dict Needle arguments repair from empty instead of raising into the envelope."""

    def _gen(**kwargs: object) -> dict[str, object]:
        return {"tool": "query_finra", "arguments": ["ticker"], "confidence": 0.9, "reasoning": "r"}

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)

    def _schema(tool_name: str, registry: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
        return {"type": "object"}

    monkeypatch.setattr("app.research.scheduler._schema_for", _schema)
    out = kw._arguments({"id": "n1", "tool": "query_finra", "objective": "short interest?"})
    assert "error" not in out and isinstance(out["arguments"], dict)


def test_arguments_inline_schema_reaches_needle_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty inline schema reaches Needle verbatim; the inline helper never consults the registry."""
    seen: list[dict[str, object]] = []
    inline: dict[str, JSONValue] = {"type": "object", "properties": {"ticker": {"type": "string"}}}

    def _gen(**kwargs: object) -> dict[str, object]:
        seen.append(dict(kwargs))
        return {"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "r"}

    def _boom_schema(name: str, reg: object) -> dict[str, object]:
        raise AssertionError("registry must not substitute for an inline schema")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _gen)

    def _ctx(context: JSONValue, tool: str, objective: str, ticker: str | None) -> dict[str, JSONValue]:
        return {}

    monkeypatch.setattr(kw, "_needle_request_context", _ctx)
    monkeypatch.setattr("app.research.scheduler._schema_for", _boom_schema)
    out = kw._arguments({"id": "s1", "tool": "query_finra", "objective": "short interest?", "schema": inline})
    assert out["tool"] == "query_finra" and seen and seen[0]["schema"] == inline


def test_arguments_node_session_id_carries_without_arguments_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node-carried session_id shortcuts without Needle when arguments holds no mapping."""

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run for a sid carry")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments({"id": "n1", "tool": "research_status", "node": {"session_id": "rs:5"}})
    assert out["arguments"] == {"session_id": "rs:5"}


def test_assess_entry_forwards_caller_mappings_to_jev() -> None:
    """Caller arguments and result mappings reach JEV; non-mappings narrow to empty."""
    from typing import override

    from app.decision_client import JevClient

    class _Rec(JevClient):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[object, object]] = []

        @override
        async def assess_entry_tool(
            self, prompt: str, tool: str, arguments: object = None, result: object = None
        ) -> str:
            self.seen.append((arguments, result))
            return "node_resolved"

    jev = _Rec()
    out = kw._assess_entry(
        {"id": "j1", "prompt": "risk?", "tool": "t", "arguments": {"a": 1}, "result": {"ok": True}}, jev=jev
    )
    assert out == {"id": "j1", "verdict": "node_resolved"}
    assert jev.seen == [({"a": 1}, {"ok": True})]


class _ArgsResultJev(JevClient):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[object, object]] = []

    @override
    async def assess_entry_tool(
        self,
        prompt: str,
        tool: str,
        arguments: Mapping[str, JSONValue] | None = None,
        result: Mapping[str, JSONValue] | None = None,
    ) -> str:
        self.seen.append((arguments, result))
        return "node_resolved"


def test_assess_entry_forwards_arguments_and_result_mappings() -> None:
    jev = _ArgsResultJev()
    out = kw._assess_entry(
        {"id": "ea", "prompt": "risk?", "tool": "t", "arguments": {"ticker": "NVDA"}, "result": {"ok": True}}, jev=jev
    )
    assert out == {"id": "ea", "verdict": "node_resolved"}
    assert jev.seen == [({"ticker": "NVDA"}, {"ok": True})]


def test_assess_final_response_blank_text_keeps_request_id() -> None:
    out = kw._assess_final_response({"id": "fb", "text": "  "})
    assert out["id"] == "fb"
    assert out["error"] == "text required"
    assert out["evidence_state"] == "insufficient"


def test_assess_final_response_carries_numeric_confidence() -> None:
    seen: dict[str, object] = {}

    class _Recorder(JevClient):
        @override
        async def assess_final_response(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen["kwargs"] = k
            return {"support_answer": "yes", "evidence_state": "sufficient_support"}

    out = kw._assess_final_response({"id": "fc", "text": "WSJ reported X.", "confidence": 0.7}, jev=_Recorder())
    assert out["id"] == "fc"
    assert seen["kwargs"] == {"session_id": "fc", "confidence": 0.7, "claims": []}


def test_assess_final_claim_blank_text_keeps_request_id() -> None:
    blank = kw._assess_final_claim({"id": "cb", "claim": {"text": "  ", "claim_type": "inference"}})
    assert blank["id"] == "cb"
    assert blank["error"] == "text required"
    assert blank["support"] is None and blank["evidence_state"] == "insufficient"
    missing_arg = kw._assess_final_claim({"id": "cm"})
    assert missing_arg["id"] == "cm"
    assert missing_arg["error"] == "text required"


def test_assess_final_claim_nontext_fails_closed() -> None:
    raws: list[JSONValue] = [7, None, ["x"]]
    for raw in raws:
        out = kw._assess_final_claim({"id": "cn", "claim": {"text": raw, "claim_type": "inference"}})
        assert out["id"] == "cn"
        assert out["error"] == "text required"
        assert out["support"] is None and out["evidence_state"] == "insufficient"
    top = kw._assess_final_claim({"id": "ct", "text": 7, "claim_type": "inference"})
    assert top == {
        "id": "ct",
        "error": "text required",
        "support": None,
        "support_answer": None,
        "evidence_state": "insufficient",
    }


def test_assess_final_claim_falls_back_to_request_confidence() -> None:
    seen: dict[str, object] = {}

    class _Recorder(JevClient):
        @override
        async def assess_final_claim(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen["kwargs"] = k
            return {"support_answer": "yes", "evidence_state": "sufficient_support"}

    out = kw._assess_final_claim(
        {"id": "cc", "claim": {"text": "X.", "claim_type": "inference"}, "confidence": 0.6}, jev=_Recorder()
    )
    assert out["id"] == "cc"
    assert seen["kwargs"] == {"session_id": "cc", "confidence": 0.6}


def test_assess_final_claim_prefers_request_evidence_rows() -> None:
    seen: dict[str, object] = {}

    class _Recorder(JevClient):
        @override
        async def assess_final_claim(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen["args"] = a
            return {"support_answer": "no", "evidence_state": "insufficient_support"}

    out = kw._assess_final_claim(
        {
            "id": "ce",
            "claim": {"text": "X.", "claim_type": "inference", "evidence_ids": ["ev-1"]},
            "evidence": [{"id": "ev-9", "content": "filing text", "other": 1}],
        },
        jev=_Recorder(),
    )
    assert out["id"] == "ce"
    assert seen["args"] == ("X.", "inference", [{"id": "ev-9", "content": "filing text"}])


def test_assess_final_claim_reads_top_level_text_and_type() -> None:
    seen: dict[str, object] = {}

    class _Recorder(JevClient):
        @override
        async def assess_final_claim(self, *a: object, **k: object) -> dict[str, JSONValue]:
            seen["args"] = a
            seen["kwargs"] = k
            return {"support_answer": "yes", "evidence_state": "sufficient_support"}

    out = kw._assess_final_claim(
        {"id": "ch", "text": "WSJ reported X.", "claim_type": "attributed_report"}, jev=_Recorder()
    )
    assert out["id"] == "ch"
    assert seen["args"] == ("WSJ reported X.", "attributed_report", [])
    assert seen["kwargs"] == {"session_id": "ch", "confidence": None}


def test_digest_dedupes_shared_provenance_source_uri() -> None:
    shared = "https://example.com/shared-doc"
    old: dict[str, object] = {
        "evidence_id": "ev-old",
        "session_id": "s1",
        "content": "old content here",
        "source_name": "SEC",
        "source_uri": "",
        "published_at": "2026-08-01T00:00:00+00:00",
        "known_at": "2026-08-01T00:00:00+00:00",
        "retrieved_at": "2026-10-06T00:00:00+00:00",
        "metadata": {},
        "provenance": {"source_uri": shared},
    }
    new: dict[str, object] = {
        "evidence_id": "ev-new",
        "session_id": "s1",
        "content": "new content here",
        "source_name": "SEC",
        "source_uri": "",
        "published_at": "2026-09-01T00:00:00+00:00",
        "known_at": "2026-09-01T00:00:00+00:00",
        "retrieved_at": "2026-10-06T00:00:00+00:00",
        "metadata": "not-a-mapping",
        "provenance": {"source_uri": shared},
    }
    digest = kw._build_digest([old, new])
    assert "ev-new" in digest and "ev-old" not in digest
    dup: dict[str, object] = {
        "evidence_id": "ev-dup",
        "session_id": "s1",
        "content": "new content here",
        "source_name": "SEC",
        "source_uri": "",
        "published_at": "2026-09-02T00:00:00+00:00",
        "known_at": "2026-09-02T00:00:00+00:00",
        "retrieved_at": "2026-10-06T00:00:00+00:00",
        "metadata": {},
        "provenance": {"source_uri": "https://example.com/other-doc"},
    }
    digest2 = kw._build_digest([new, dup])
    assert "ev-dup" in digest2 and "ev-new" in digest2


def test_route_caller_ticker_reaches_programmatic_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller ticker grounds subject routing: hint wins over the JEV fallback."""
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    jev = _JevRoute("reasoning_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "hello", "ticker": "NVDA"}, jev=jev)
    assert out == {"id": "r1", "route": "research_required", "personas": None}


def test_jev_admit_forwards_exact_disposition_contract() -> None:
    """JEV owns admission: the disposition call carries the exact session, objective, proposals, and contract."""
    seen: dict[str, object] = {}

    class _Contract(JevClient):
        @override
        async def decide(
            self,
            state: object,
            questions: object,
            *,
            decision_type: str,
            session_id: str,
            choice_options: object = None,
            **rest: object,
        ) -> dict[str, dict[str, JSONValue]]:
            seen["state"] = state
            seen["questions"] = questions
            seen["decision_type"] = decision_type
            seen["session_id"] = session_id
            seen["choice_options"] = choice_options
            return {
                "s-q1": {"choice": "analyze", "probabilities": {"analyze": 1.0}, "confidence": 0.9},
                "s-q2": {"choice": "gather_evidence", "probabilities": {"gather_evidence": 1.0}, "confidence": 0.9},
            }

    props = _props()
    out = kw._jev_admit("s", "Exact user objective?", props, jev=_Contract())
    assert out == props
    assert seen["decision_type"] == "proposal_disposition"
    assert seen["session_id"] == "s"
    assert seen["state"] == {"objective": {"prompt": "Exact user objective?"}, "proposals": props}
    expected_questions, expected_options = kw._jev_disposition(props)
    assert seen["questions"] == expected_questions
    assert seen["choice_options"] == expected_options


def test_jev_admit_all_reject_falls_back_to_exact_objective() -> None:
    """Total JEV rejection keeps the exact-objective proposal, never a synthesized question."""

    class _RejectAll(JevClient):
        @override
        async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
            return {
                "s-q1": {"choice": "reject", "probabilities": {"reject": 1.0}, "confidence": 0.9},
                "s-q2": {"choice": "reject", "probabilities": {"reject": 1.0}, "confidence": 0.9},
            }

    out = kw._jev_admit("s", "Exact user objective?", _props(), jev=_RejectAll())
    assert out == [_props()[1]]


def test_jev_admit_all_reject_synthesizes_session_route_question() -> None:
    """No exact-objective proposal under total rejection synthesizes the session route question."""

    class _RejectAll(JevClient):
        @override
        async def decide(self, *a: object, **k: object) -> dict[str, dict[str, JSONValue]]:
            return {
                "s-q1": {"choice": "reject", "probabilities": {"reject": 1.0}, "confidence": 0.9},
                "s-q2": {"choice": "reject", "probabilities": {"reject": 1.0}, "confidence": 0.9},
            }

    out = kw._jev_admit("s", "Missing objective?", _props(), jev=_RejectAll())
    assert out == [
        {
            "id": "s-q1",
            "objectiveId": "s",
            "question": "Missing objective?",
            "dependsOn": [],
            "whyItMatters": "Route question.",
        }
    ]


def test_decompose_config_wiring_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset OpenCode settings fail as missing config after the first call, never a silent retry."""
    calls = {"n": 0}

    def _boom(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        raise RuntimeError("opencode_unavailable: missing OPENCODE_API_KEY")

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _boom)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_URL", raising=False)
    monkeypatch.delenv("OPENCODE_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="opencode_unavailable"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
    assert calls["n"] == 1


@pytest.mark.parametrize("missing", ["OPENCODE_API_KEY", "OPENCODE_URL", "OPENCODE_MODEL"])
def test_decompose_missing_setting_denies_transport(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """A missing setting denies the provider request before the transport receives any call."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_URL", "https://reasoner.invalid/responses")
    monkeypatch.setenv("OPENCODE_MODEL", "test-model")
    monkeypatch.delenv(missing)
    with mock.patch(
        "app.reasoner_client._default_post", side_effect=AssertionError("missing config must deny transport")
    ) as post:
        with pytest.raises(RuntimeError, match="opencode_unavailable"):
            kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
        post.assert_not_called()


def test_decompose_explicit_timeout_bounds_a_stuck_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit timeout bounds a stuck decompose; ignoring it exceeds the bound."""
    import time as _t

    calls = {"n": 0}

    def _hang(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        _t.sleep(5)
        return {"proposals": []}

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _hang)
    t0 = _t.perf_counter()
    with pytest.raises(TimeoutError, match="timed out"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "", timeout_s=0.2)
    assert calls["n"] == 1
    assert _t.perf_counter() - t0 < 5.0


def test_decompose_request_carries_exact_objective_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decompose call and its proposals carry the exact objective text and session id."""
    seen: dict[str, object] = {}
    good: dict[str, object] = {
        "id": "rs:t-q1",
        "objectiveId": "rs:t",
        "question": "Q?",
        "dependsOn": [],
        "whyItMatters": "W",
    }

    def _good(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        seen["prompt"] = prompt
        seen["objective_id"] = objective_id
        return {"proposals": [dict(good)]}

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _good)
    proposals, _ = kw._reasoner_decompose_with_retry("Exact user objective?", None, "rs:t", "DIGEST")
    assert proposals == [good]
    assert seen["objective_id"] == "rs:t"
    assert isinstance(seen["prompt"], str) and "Exact user objective?" in seen["prompt"] and "DIGEST" in seen["prompt"]


def test_decompose_non_dict_reply_retries_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-mapping reply reads as empty proposals and exhausts the two-attempt budget as TypeError."""
    calls = {"n": 0}

    def _weird(self: object, prompt: str, objective_id: str) -> object:
        calls["n"] += 1
        return ["not-a-mapping"]

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _weird)
    with pytest.raises(TypeError, match="decompose: empty proposals"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
    assert calls["n"] == 2


def test_decompose_invalid_reply_with_spent_budget_raises_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed proposal set with a spent setup budget raises at once, never a second call."""
    import time as _t

    calls = {"n": 0}

    def _malformed(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        return {"proposals": [42]}

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _malformed)
    with pytest.raises(TypeError, match="proposal must be an object"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "", None, None, _t.perf_counter() - 1.0)
    assert calls["n"] == 1


def test_decompose_empty_feedback_text_reaches_retry_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The attempt-2 prompt carries the exact empty-proposals validator error for repair."""
    prompts: list[str] = []
    good: dict[str, object] = {
        "proposals": [{"id": "rs:t-q1", "objectiveId": "rs:t", "question": "Q?", "dependsOn": [], "whyItMatters": "W"}]
    }

    def _empty_then_good(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        prompts.append(prompt)
        return {"proposals": []} if len(prompts) == 1 else dict(good)

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _empty_then_good)
    proposals, _ = kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
    assert [p["id"] for p in proposals] == ["rs:t-q1"]
    assert len(prompts) == 2 and "decompose: empty proposals" in prompts[1]


def test_decompose_quota_typed_error_never_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typed quota error never resends: one call, then the original error propagates."""
    calls = {"n": 0}

    class FreeUsageLimitError(RuntimeError):
        pass

    failure = FreeUsageLimitError("account unavailable")

    def _quota(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        raise failure

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _quota)
    with pytest.raises(FreeUsageLimitError) as caught:
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "")
    assert calls["n"] == 1
    assert caught.value is failure


def test_decompose_usage_window_typed_error_marks_rate_limited_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typed usage-window error resends once with the rate-limited marker, then succeeds."""
    calls = {"n": 0}
    good: dict[str, object] = {
        "id": "rs:t-q1",
        "objectiveId": "rs:t",
        "question": "Q?",
        "dependsOn": [],
        "whyItMatters": "W",
    }
    events: list[tuple[str, dict[str, object] | None]] = []

    def _window(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise type("GoUsageLimitError", (RuntimeError,), {})("Go usage window exceeded")
        return {"proposals": [dict(good)]}

    def _progress(stage: str, detail: dict[str, object] | None = None) -> None:
        events.append((stage, detail))

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _window)
    proposals, _ = kw._reasoner_decompose_with_retry("q?", None, "rs:t", "", progress=_progress)
    assert [p["id"] for p in proposals] == ["rs:t-q1"]
    assert calls["n"] == 2
    assert ("reasoner_retry", {"attempt": 2, "reason": "rate_limited"}) in events


def test_decompose_plain_transient_twice_raises_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain transient error resends once without the rate-limited marker, then raises itself."""
    calls = {"n": 0}
    events: list[tuple[str, dict[str, object] | None]] = []

    def _transient(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        calls["n"] += 1
        raise RuntimeError("connection reset by peer")

    def _progress(stage: str, detail: dict[str, object] | None = None) -> None:
        events.append((stage, detail))

    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _transient)
    with pytest.raises(RuntimeError, match="connection reset by peer"):
        kw._reasoner_decompose_with_retry("q?", None, "rs:t", "", progress=_progress)
    assert calls["n"] == 2
    assert ("reasoner_retry", {"attempt": 2}) in events
    assert ("reasoner_retry", {"attempt": 2, "reason": "rate_limited"}) not in events
