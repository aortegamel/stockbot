"""Fail-closed guard contracts: failures propagate, envelopes preserved."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    assert kw._jev_admit("s", "q", props, jev=_JevDown()) == props


def test_graph_prompt_registry_failure_is_terminal() -> None:
    with mock.patch("app.research.scheduler.build_registry", side_effect=RuntimeError("boom")):
        resp = kw._run({"id": "r1", "op": "run", "prompt": "Is XYZ solvent?"})
    assert resp["id"] == "r1"
    terminal = resp.get("terminal")
    assert isinstance(terminal, dict) and "registry guard forbids" in str(terminal.get("message"))


def test_research_registry_excludes_portfolio_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live research registry exposes no portfolio tool, even with the broker enabled."""
    from app.tools import PORTFOLIO_AUTHORIZED_TOOLS

    monkeypatch.setenv("BROKER_ENABLED", "true")
    names = {entry["name"] for entry in sched.build_registry()}
    assert PORTFOLIO_AUTHORIZED_TOOLS
    assert not names & PORTFOLIO_AUTHORIZED_TOOLS
    assert kw._registry_portfolio_hit() == []
    leaked = [*sched.build_registry(), {"name": "get_portfolio_snapshot"}]
    with mock.patch("app.research.scheduler.build_registry", return_value=leaked):
        assert kw._registry_portfolio_hit() == ["get_portfolio_snapshot"]


class _JevRoute(JevClient):
    def __init__(
        self,
        choice: str | None = None,
        personas: list[str] | None = None,
        fail: bool = False,
        chart_metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        super().__init__()
        self.choice = choice
        self.personas = personas
        self.fail = fail
        self.chart_metadata = chart_metadata or {}
        self.seen: list[tuple[str, str | None]] = []
        self.watchlists: list[Sequence[str] | None] = []
        self.current_tickers: list[str | None] = []

    @override
    async def route_entry(
        self,
        prompt: str,
        registry: Sequence[Mapping[str, JSONValue]] | None = None,
        route_hint: str | None = None,
        watchlist: Sequence[str] | None = None,
        current_ticker: str | None = None,
    ) -> dict[str, JSONValue]:
        if self.fail:
            raise RuntimeError("jev down")
        self.seen.append((prompt, route_hint))
        self.watchlists.append(watchlist)
        self.current_tickers.append(current_ticker)
        return {
            "route": route_hint or self.choice or "research_required",
            "personas": list[JSONValue](self.personas) if self.personas is not None else None,
            **self.chart_metadata,
        }


@pytest.mark.parametrize("rid", [None, 7, False, [], {}])
def test_route_invalid_request_id_uses_protocol_placeholder(rid: JSONValue) -> None:
    out = kw._route({"id": rid, "prompt": "explain diversification"}, jev=_JevRoute("reasoning_required"))
    assert out == {"id": "?", "route": "reasoning_required", "personas": None}


def test_route_reasoning_choice_returns_fast_path() -> None:
    out = kw._route(
        {"id": "r1", "op": "route", "prompt": "explain diversification"},
        jev=_JevRoute("reasoning_required"),
    )
    assert out == {"id": "r1", "route": "reasoning_required", "personas": None}


def test_route_selector_outage_is_error_not_assumed_personas() -> None:
    out = kw._route({"id": "r1", "op": "route", "prompt": "explain diversification"}, jev=_JevRoute(fail=True))
    assert out == {"id": "r1", "error": "jev down"}


@pytest.mark.parametrize(
    ("req", "rid"),
    [
        ({"prompt": "hello"}, "?"),
        ({"id": "r1", "prompt": "hello"}, "r1"),
        ({"id": None, "prompt": "hello"}, "?"),
        ({"id": 7, "prompt": "hello"}, "?"),
        ({"id": False, "prompt": "hello"}, "?"),
        ({"id": [], "prompt": "hello"}, "?"),
        ({"id": {}, "prompt": "hello"}, "?"),
    ],
)
def test_route_small_talk_needs_no_jev(monkeypatch: pytest.MonkeyPatch, req: dict[str, JSONValue], rid: str) -> None:
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    jev = _JevRoute(fail=True)
    out = kw._route(req, jev=jev)
    assert out == {"id": rid, "route": "no_session", "personas": None}
    assert jev.seen == []


def test_route_missing_advice_subject_needs_no_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    jev = _JevRoute(fail=True)
    with mock.patch.object(kw, "_jev_client", side_effect=AssertionError("clarification must not access JEV")):
        out = kw._route({"id": "advice", "op": "route", "prompt": "can you give me buying advice"}, jev=jev)
    assert out == {"id": "advice", "route": "clarify_subject", "personas": None}
    assert jev.seen == []


@pytest.mark.parametrize(
    "context",
    [
        {"ticker": "NVDA"},
        {"currentChartTicker": "nvda"},
        {"ticker": "AAPL", "currentChartTicker": "NVDA"},
    ],
)
def test_route_current_context_supplies_advice_subject(
    context: dict[str, JSONValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    jev = _JevRoute("research_required", personas=["bearbot"])
    out = kw._route({"id": "advice", "prompt": "can you give me buying advice", **context}, jev=jev)
    assert out == {"id": "advice", "route": "research_required", "personas": ["bearbot"]}
    assert jev.seen[0][1] == "research_required"
    assert jev.current_tickers == (["NVDA"] if "currentChartTicker" in context else [None])


@pytest.mark.parametrize("prompt", ["hello", "thanks", "good morning"])
def test_route_greeting_ignores_current_chart(prompt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    jev = _JevRoute(fail=True)
    assert kw._route({"id": "greeting", "prompt": prompt, "currentChartTicker": "NVDA"}, jev=jev) == {
        "id": "greeting",
        "route": "no_session",
        "personas": None,
    }
    assert jev.seen == []


def test_route_current_chart_does_not_override_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    jev = _JevRoute(personas=["bearbot"])
    out = kw._route({"id": "clock", "prompt": "What time is it?", "currentChartTicker": "nvda"}, jev=jev)
    assert out == {"id": "clock", "route": "get_current_time", "personas": ["bearbot"]}
    assert jev.seen[0][1] == "get_current_time"
    assert jev.current_tickers == ["NVDA"]


@pytest.mark.parametrize("bad", [True, 7, [], {}, "FINRA", "NVDA AAPL"])
def test_route_rejects_invalid_current_chart(bad: JSONValue) -> None:
    jev = _JevRoute(fail=True)
    out = kw._route({"id": "bad-chart", "prompt": "advice", "currentChartTicker": bad}, jev=jev)
    assert out["id"] == "bad-chart"
    assert "currentChartTicker" in str(out["error"])
    assert jev.seen == []


@pytest.mark.parametrize("metadata", [{"chartTicker": "AAPL"}, {"chartError": "EDGAR unavailable"}])
def test_route_preserves_chart_metadata(metadata: dict[str, JSONValue]) -> None:
    jev = _JevRoute("research_required", personas=["bullbot"], chart_metadata=metadata)
    assert kw._route({"id": "chart", "prompt": "explain diversification"}, jev=jev) == {
        "id": "chart",
        "route": "research_required",
        "personas": ["bullbot"],
        **metadata,
    }


def test_route_current_chart_stays_separate_from_watchlist() -> None:
    jev = _JevRoute("research_required")
    out = kw._route(
        {
            "id": "watchlist",
            "prompt": "Latest filings for my watchlist?",
            "currentChartTicker": "aapl",
            "watchlist": ["nvda", "QQQ"],
        },
        jev=jev,
    )
    assert out == {"id": "watchlist", "route": "research_required", "personas": None}
    assert jev.watchlists == [["NVDA", "QQQ"]]
    assert jev.current_tickers == ["AAPL"]


@pytest.mark.parametrize(
    ("prompt", "ticker", "hint"),
    [
        ("can you give me buying advice on NVDA", None, "research_required"),
        ("can you give me buying advice on NVDA", "NVDA", "research_required"),
        ("can you give me buying advice for Nvidia", None, "research_required"),
        ("can you give me buying advice for Nvidia", "NVDA", "research_required"),
        ("can you give me buying advice on bonds", None, None),
        ("can you give me buying advice after CPI", None, None),
    ],
)
def test_route_advice_with_content_retains_jev(
    prompt: str, ticker: str | None, hint: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA", "Nvidia": "NVDA"}.get(name)

    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _resolve)
    jev = _JevRoute("research_required")
    out = kw._route({"id": "advice", "prompt": prompt, "ticker": ticker}, jev=jev)
    assert out == {"id": "advice", "route": "research_required", "personas": None}
    assert len(jev.seen) == 1
    assert jev.seen[0][1] == hint


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


def test_route_ticker_does_not_override_clock_request() -> None:
    jev = _JevRoute("research_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "What time is it?", "ticker": "NVDA"}, jev=jev)
    assert out == {"id": "r1", "route": "get_current_time", "personas": None}
    assert "What time is it?" in jev.seen[0][0]
    assert "NVDA" not in jev.seen[0][0]


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
    assert "Latest filings for my watchlist?" in jev.seen[0][0]
    for sym in ("NVDA", "QQQ"):
        assert sym not in jev.seen[0][0]


@pytest.mark.parametrize("bad", ["NVDA", [], [""], [7], [f"T{i}" for i in range(26)]])
def test_route_bad_watchlist_is_error_without_jev(bad: JSONValue) -> None:
    jev = _JevRoute("research_required")
    out = kw._route({"id": "r1", "op": "route", "prompt": "my watchlist?", "watchlist": bad}, jev=jev)
    assert out["id"] == "r1" and "watchlist" in str(out["error"])
    assert jev.seen == []


def test_route_watchlist_scope_does_not_override_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_TOOLFLOW", "catalog")
    monkeypatch.setattr("app.decision_client._auto_registry", lambda: _ROUTE_REG)
    seen: dict[str, object] = {}
    client = JevClient(transport=_choice_stub({"entry": "reasoning_required", "personas": "bearbot"}, seen))
    out = kw._route(
        {"id": "rw", "prompt": "Latest filings for my watchlist?", "watchlist": ["nvda", "QQQ", "nvda"]},
        jev=client,
    )
    assert out == {"id": "rw", "route": "reasoning_required", "personas": ["bearbot"]}
    questions = seen["questions"]
    assert isinstance(questions, dict)
    entry = str(questions["entry"])
    personas = str(questions["personas"])
    assert "Latest filings for my watchlist?" in entry
    assert all(symbol in entry and symbol not in personas for symbol in ("NVDA", "QQQ"))


@pytest.mark.parametrize("bad", ["NVDA", [], [""]])
def test_route_entry_rejects_blank_watchlist(bad: Sequence[str]) -> None:
    with pytest.raises(ValueError, match="blank watchlist"):
        _route_entry("my watchlist?", {"entry": "research_required"}, watchlist=bad)


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


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("can you give me buying advice", "NVDA"),
        ("Should I buy Apple?", "AAPL"),
        ("Should I buy MissingCo?", "NVDA"),
    ],
)
def test_arguments_context_and_withhold_preserve_subject_precedence(
    monkeypatch: pytest.MonkeyPatch, objective: str, expected: str
) -> None:
    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL", "NVDA": "NVDA"}.get(name)

    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _resolve)
    generated = _FailGenerate(None)
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: generated)
    out = kw._arguments({"id": "context", "tool": "list_sec_filings", "objective": objective, "ticker": "NVDA"})
    payload = generated.seen[0]
    assert isinstance(payload, dict)
    context = payload["context"]
    assert isinstance(context, dict)
    assert context.get("ticker") == expected
    grounded = context["grounded_arguments"]
    assert isinstance(grounded, dict)
    assert grounded["identifier"] == expected
    arguments = out["arguments"]
    assert isinstance(arguments, dict) and arguments["identifier"] == expected


class _FailGenerate:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.seen: list[object] = []

    def __call__(self, **kwargs: object) -> object:
        self.seen.append(kwargs)
        return self.payload


@pytest.mark.parametrize("schema_source", ["registry", "empty", "inline"])
def test_arguments_uses_the_caller_schema_or_registered_schema(
    monkeypatch: pytest.MonkeyPatch, schema_source: str
) -> None:
    registered: dict[str, JSONValue] = {
        "type": "object",
        "title": "registered",
        "properties": {"ticker": {"type": "string"}},
    }
    inline: dict[str, JSONValue] = {"type": "object", "title": "caller", "properties": {"ticker": {"type": "string"}}}
    generated = _FailGenerate(
        {"tool": "query_finra", "arguments": {"ticker": "NVDA"}, "confidence": 0.9, "reasoning": "grounded"}
    )
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: generated)
    monkeypatch.setattr(sched, "build_registry", lambda: [{"name": "query_finra", "parameters": registered}])
    request: dict[str, JSONValue] = {"id": "schema", "tool": "query_finra", "objective": "short interest?"}
    if schema_source != "registry":
        request["schema"] = inline if schema_source == "inline" else dict[str, JSONValue]()
    out = kw._direct_op({"op": "arguments", **request}, None)
    assert out is not None
    assert out["id"] == "schema" and out["tool"] == "query_finra"
    assert out["arguments"] == {"ticker": "NVDA", "dataset": "consolidatedShortInterest"}
    assert out["confidence"] == 0.9 and out["reasoning"] == "grounded"
    assert len(generated.seen) == 1
    payload = generated.seen[0]
    assert isinstance(payload, dict)
    assert payload["tool"] == "query_finra"
    assert payload["schema"] == (inline if schema_source == "inline" else registered)
    assert payload["objective"] == "short interest?"


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


def test_chart_tool_stays_in_research_selection_but_not_direct_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    session_id = service.create_research("show a chart", "Show a chart.")
    node = service.create_node(session_id, "Show a chart.", "why chart tooling matters")
    registry = sched.build_registry()
    assert any(entry.get("name") == "set_chart_symbol" for entry in registry)
    seen: dict[str, object] = {}
    client = JevClient(
        transport=_choice_stub(
            {"entry": "research_required", "assess": "node_resolved", "tool_selection": "set_chart_symbol"}, seen
        ),
        data_root=tmp_path,
    )
    route = asyncio.run(client.route_entry("explain diversification", registry))
    assert route == {"route": "research_required", "personas": None}
    route_questions = seen["questions"]
    assert isinstance(route_questions, dict)
    entry = route_questions["entry"]
    assert isinstance(entry, dict)
    entry_options = entry["criteria"]
    assert isinstance(entry_options, dict) and "set_chart_symbol" not in entry_options

    assessment = asyncio.run(client.assess_entry_tool("What time is it?", "get_current_time"))
    assert assessment == "node_resolved"
    assess_questions = seen["questions"]
    assert isinstance(assess_questions, dict)
    assess = assess_questions["assess"]
    assert isinstance(assess, dict)
    assess_options = assess["criteria"]
    assert isinstance(assess_options, dict) and "set_chart_symbol" not in assess_options

    selected = asyncio.run(
        client.select_tool(
            "show a chart", {"node_id": node.node_id, "question": "Show a chart."}, registry, session_id=session_id
        )
    )
    assert selected.action == "invoke" and selected.tool_name == "set_chart_symbol"
    tool_questions = seen["questions"]
    assert isinstance(tool_questions, dict)
    tool_selection = tool_questions["tool_selection"]
    assert isinstance(tool_selection, dict)
    tool_options = tool_selection["criteria"]
    assert isinstance(tool_options, dict) and "set_chart_symbol" in tool_options


def _route_entry(
    prompt: str,
    picks: dict[str, str],
    route_hint: str | None = None,
    watchlist: Sequence[str] | None = None,
    current_ticker: str | None = None,
) -> tuple[dict[str, JSONValue], dict[str, object]]:
    seen: dict[str, object] = {}
    client = JevClient(transport=_choice_stub(picks, seen), data_root=Path("/tmp"))
    return (
        asyncio.run(
            client.route_entry(
                prompt, _ROUTE_REG, route_hint=route_hint, watchlist=watchlist, current_ticker=current_ticker
            )
        ),
        seen,
    )


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


def test_route_entry_hint_keeps_route_and_asks_personas_and_chart() -> None:
    out, seen = _route_entry(
        "what time is it?",
        {"personas": "bearbot_bullbot"},
        route_hint="get_current_time",
    )
    assert out == {"route": "get_current_time", "personas": ["bearbot", "bullbot"]}
    questions = seen["questions"]
    assert isinstance(questions, dict) and list(questions) == ["personas", "chart"]


def test_route_entry_no_change_never_resolves_candidates() -> None:
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect=AssertionError("no resolution")):
        out, seen = _route_entry(
            "Compare Nvidia and Apple",
            {"entry": "research_required", "personas": "bearbot", "chart": "no_change"},
            current_ticker="MSFT",
        )
    assert out == {"route": "research_required", "personas": ["bearbot"]}
    questions = seen["questions"]
    assert isinstance(questions, dict)
    chart = questions["chart"]
    assert isinstance(chart, dict)
    assert chart["type"] == "choice"
    criteria = chart["criteria"]
    assert isinstance(criteria, dict) and set(criteria) == {"no_change", "Nvidia", "Apple"}


@pytest.mark.parametrize(("selected", "ticker"), [("Nvidia", "NVDA"), ("Apple", "AAPL")])
def test_route_entry_resolves_only_selected_candidate(selected: str, ticker: str) -> None:
    with mock.patch("app.tools._resolve_company_to_ticker", return_value=ticker) as resolve:
        out, _ = _route_entry(
            "Compare Nvidia and Apple",
            {"entry": "research_required", "personas": "stockbot_bullbot", "chart": selected},
            current_ticker="MSFT",
        )
    resolve.assert_called_once_with(selected)
    assert out == {"route": "research_required", "personas": ["stockbot", "bullbot"], "chartTicker": ticker}


def test_route_entry_explicit_stock_excludes_displayed_chart_candidate() -> None:
    with mock.patch("app.tools._resolve_company_to_ticker", return_value="NVDA") as resolve:
        out, seen = _route_entry(
            "Should I buy NVDA?",
            {"entry": "reasoning_required", "chart": "NVDA"},
            current_ticker="AAPL",
        )
    resolve.assert_called_once_with("NVDA")
    assert out == {"route": "reasoning_required", "personas": None, "chartTicker": "NVDA"}
    questions = seen["questions"]
    assert isinstance(questions, dict)
    chart = questions["chart"]
    assert isinstance(chart, dict)
    criteria = chart["criteria"]
    assert isinstance(criteria, dict) and set(criteria) == {"no_change", "NVDA"}


def test_route_entry_selected_current_chart_has_no_update() -> None:
    with mock.patch("app.tools._resolve_company_to_ticker", return_value="NVDA") as resolve:
        out, _ = _route_entry(
            "Should I buy it?",
            {"entry": "research_required", "personas": "bearbot", "chart": "NVDA"},
            current_ticker="NVDA",
        )
    resolve.assert_called_once_with("NVDA")
    assert out == {"route": "research_required", "personas": ["bearbot"]}


@pytest.mark.parametrize("resolution", [None, "", "FINRA", 7])
def test_route_entry_chart_resolution_failure_keeps_route_and_personas(resolution: object) -> None:
    with mock.patch("app.tools._resolve_company_to_ticker", return_value=resolution) as resolve:
        out, _ = _route_entry(
            "Should I buy Nvidia?",
            {"entry": "reasoning_required", "personas": "bullbot", "chart": "Nvidia"},
            current_ticker="AAPL",
        )
    resolve.assert_called_once_with("Nvidia")
    assert out["route"] == "reasoning_required"
    assert out["personas"] == ["bullbot"]
    assert "Nvidia" in str(out["chartError"])
    assert "chartTicker" not in out


def test_route_entry_chart_resolver_outage_keeps_route_and_personas() -> None:
    with mock.patch("app.tools._resolve_company_to_ticker", side_effect=RuntimeError("EDGAR unavailable")):
        out, _ = _route_entry(
            "Should I buy Nvidia?",
            {"entry": "research_required", "personas": "bearbot", "chart": "Nvidia"},
        )
    assert out["route"] == "research_required"
    assert out["personas"] == ["bearbot"]
    assert "EDGAR unavailable" in str(out["chartError"])
    assert "chartTicker" not in out


@settings(max_examples=50, derandomize=True)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text(max_size=40),
        st.lists(st.text(max_size=10), max_size=3),
    )
)
def test_route_entry_malformed_chart_choice_never_resolves(label: JSONValue) -> None:
    assume(label not in ("no_change", "NVDA"))
    client = JevClient(data_root=Path("/tmp"))

    def _sidecar(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return {
            "id": payload["id"],
            "decisions": {
                "entry": {"choice": "research_required"},
                "personas": {"choice": "bearbot"},
                "chart": {"choice": label},
            },
            "raw": {},
        }

    with (
        mock.patch.object(client, "_sidecar_roundtrip", _sidecar),
        mock.patch("app.tools._resolve_company_to_ticker", side_effect=AssertionError("invalid chart choice")),
    ):
        out = asyncio.run(client.route_entry("NVDA?", _ROUTE_REG))
    assert out["route"] == "research_required"
    assert out["personas"] == ["bearbot"]
    assert isinstance(out.get("chartError"), str) and out["chartError"]
    assert "chartTicker" not in out


def test_route_entry_selector_unavailable_raises() -> None:
    async def _down(state: object, questions: object) -> object:
        raise RuntimeError("jev down")

    client = JevClient(transport=_down, data_root=Path("/tmp"))
    with pytest.raises(RuntimeError, match="jev down"):
        asyncio.run(client.route_entry("NVDA?", _ROUTE_REG))


@pytest.mark.parametrize(
    ("deadline", "setup", "scheduler"),
    [
        (None, 550.0, 600.0),
        (True, 550.0, 600.0),
        ("soon", 550.0, 600.0),
        (0, 550.0, 600.0),
        (-5, 550.0, 600.0),
        (1_060_000.0, 490.0, 540.0),
        (1_100_000.0, 530.0, 580.0),
        (1_500_000.0, 550.0, 600.0),
    ],
)
def test_run_shares_bounded_deadline_across_setup_and_scheduler(
    monkeypatch: pytest.MonkeyPatch, deadline: JSONValue, setup: float, scheduler: float
) -> None:
    monkeypatch.setattr(kw.time, "time", lambda: 1000.0)
    monkeypatch.setattr(kw.time, "perf_counter", lambda: 500.0)
    closed: list[str] = []

    def _graph(prompt: str, as_of: object = None, **hooks: object) -> str:
        assert hooks["setup_deadline"] == setup
        return "s1"

    async def _schedule(sid: str, **hooks: object) -> dict[str, object]:
        assert hooks["deadline_at"] == scheduler
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(kw, "run_graph_prompt", _graph)
    monkeypatch.setattr(sched, "run", _schedule)
    monkeypatch.setattr(kw, "_close_bootstrap_job", closed.append)
    _empty_repo(monkeypatch)
    out = kw._run({"id": "deadline", "prompt": "hello?", "deadlineAt": deadline}, jev=JevClient())
    assert out["sessionId"] == "s1"
    assert closed == ["s1"]


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
    assert responses[0] == {"type": "ready"} and responses[-1] == out
    if scheduler_error:
        assert responses[1:-1] == []
    else:
        assert responses[1:-1] == [{"type": "progress", "id": "rc", "stage": "nodes_done", "detail": {"ms": mock.ANY}}]
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

    def _no_ticker(objective: str, supplied_ticker: object = None) -> None:
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


@pytest.mark.parametrize("flow", [None, "catalog", "full", "whole", "direct"])
def test_run_toolflow_selects_scheduler_policy(monkeypatch: pytest.MonkeyPatch, flow: str | None) -> None:
    from app.research.programmatic_router import programmatic_select_round
    from app.research.tool_catalogs import catalog_select_round

    async def _sched(sid: str, **hooks: object) -> dict[str, object]:
        expected = programmatic_select_round if flow is None else catalog_select_round if flow == "catalog" else None
        assert hooks.get("select_round") is expected
        assert hooks["max_rounds"] == 4
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    if flow is None:
        monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)
    else:
        monkeypatch.setenv("STOCKBOT_TOOLFLOW", flow)
    monkeypatch.setattr(kw, "run_graph_prompt", _graph_session)
    monkeypatch.setattr("app.research.scheduler.run", _sched)
    monkeypatch.setattr(kw, "_close_bootstrap_job", _no_bootstrap)
    _empty_repo(monkeypatch)
    out = kw._run({"id": "rf", "prompt": "hello?"}, jev=JevClient())
    assert out["sessionId"] == "s1"


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


@pytest.mark.parametrize("guard", [True, 1, False, None])
def test_run_guard_uses_only_the_boolean_signal(monkeypatch: pytest.MonkeyPatch, guard: JSONValue) -> None:
    _empty_repo(monkeypatch)
    out = _run_with(monkeypatch, {"status": "complete", "nodes": {}, "incomplete_guard": guard})
    assert out["toolCalls"] == [] and out["toolExecutions"] == []
    assert out["incomplete_guard"] is (guard is True)
    assert out["escalated"] is (guard is True)
    assert out["unresolved"] == []


@pytest.mark.parametrize("nodes", [None, {}, "not-a-list"])
def test_run_ignores_malformed_scheduler_nodes(monkeypatch: pytest.MonkeyPatch, nodes: JSONValue) -> None:
    _empty_repo(monkeypatch)
    out = _run_with(monkeypatch, {"status": "complete", "nodes": nodes})
    assert out["toolCalls"] == [] and out["toolExecutions"] == []
    assert out["failures"] == {} and out["escalated"] is False


def test_run_sanitizes_attempts_and_links_only_known_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_repo(monkeypatch)

    def _evidence(self: ResearchRepository, sid: str) -> list[dict[str, JSONValue]]:
        return [{"content": "first fact"}, {"content": "second fact"}]

    monkeypatch.setattr(ResearchRepository, "list_evidence", _evidence)
    attempts: list[dict[str, JSONValue]] = [
        {"tool": "t", "arguments": {}, "confidence": True, "reasoning": "kept", "evidenceId": "ev-0"},
        {"tool": "t", "evidenceId": "ghost"},
        {"tool": "t", "error": "x" * 600, "error_type": "timeout", "category": "other"},
        {"tool": "t", "error": "failed", "category": "timeout"},
        {"tool": "t", "error": "failed", "error_type": 7, "category": "timeout"},
        {"tool": "t", "error": "failed", "error_type": "", "category": "timeout"},
        {"tool": "t", "error": "failed"},
    ]
    out = _run_with(monkeypatch, {"status": "complete", "nodes": [{"attempts": attempts}]})
    assert out["evidence"] == [{"id": "ev-0", "content": "first fact"}, {"id": "ev-1", "content": "second fact"}]
    calls = out["toolCalls"]
    executions = out["toolExecutions"]
    assert isinstance(calls, list) and isinstance(executions, list)
    assert isinstance(calls[0], dict) and calls[0]["evidenceId"] == "ev-0"
    assert isinstance(calls[1], dict) and "evidenceId" not in calls[1]
    assert isinstance(calls[2], dict) and calls[2]["error"] == "x" * 500
    assert isinstance(executions[0], dict)
    assert executions[0]["reasoning"] == "kept" and executions[0]["confidence"] is None
    assert out["failures"] == {"timeout": 2, "tool_error": 3}


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


def test_prewarm_does_not_block_requests_when_sources_fail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    import threading

    import app.sec.client as sec_client

    monkeypatch.delenv("STOCKBOT_TOOLFLOW", raising=False)

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _registry() -> list[dict[str, JSONValue]]:
        entered.set()
        if not release.wait(2):
            raise AssertionError("test did not release prewarm")
        raise RuntimeError("registry down")

    class _WarmThread(threading.Thread):
        @override
        def run(self) -> None:
            try:
                assert self.daemon
                super().run()
            finally:
                finished.set()

    def _lookup(key: str) -> None:
        raise RuntimeError("lookup down")

    monkeypatch.setattr(kw.threading, "Thread", _WarmThread)

    monkeypatch.setattr(sched, "build_registry", _registry)
    monkeypatch.setattr(sec_client, "ensure_identity", lambda: None)
    monkeypatch.setattr(sec_client, "_fetch_lookup_frame", _lookup)
    with caplog.at_level(logging.DEBUG, logger=kw.__name__):
        kw._prewarm()
        try:
            assert entered.wait(1)
            assert not release.is_set()
            assert kw._route({"id": "warm", "prompt": "what time is it?"}, jev=_JevRoute("research_required")) == {
                "id": "warm",
                "route": "get_current_time",
                "personas": None,
            }
        finally:
            release.set()
            assert finished.wait(1)
    errors = [record.exc_info[1] for record in caplog.records if record.exc_info is not None]
    assert any(isinstance(error, RuntimeError) and str(error) == "registry down" for error in errors)
    assert any(isinstance(error, RuntimeError) and str(error) == "lookup down" for error in errors)


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
    assert sorted(closed) == ["jev", "needle", "pools"]


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
    assert out["id"] == "s1" and out["tool"] == tool
    assert out["arguments"] == {"session_id": "rs:1"}
    assert out["confidence"] == 1.0


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
    assert carried["id"] == "r1" and carried["tool"] == "research_read_search"
    assert carried["arguments"] == {"session_id": "rs:9", "search_id": "s1"}
    assert carried["confidence"] == 1.0
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


def _needle_withhold(**kwargs: object) -> None:
    """A genuine Needle withhold: generate() answers None instead of a tool call."""


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


def test_entry_gates_use_dated_context_and_valid_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from app.research import models

    monkeypatch.setenv("STOCKBOT_TOOLFLOW", "catalog")
    monkeypatch.setattr("app.decision_client._auto_registry", lambda: _ROUTE_REG)
    monkeypatch.setattr(models, "utcnow", lambda: datetime(2026, 10, 17, tzinfo=UTC))
    states: list[dict[str, JSONValue]] = []

    async def _transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        assert isinstance(state, dict)
        states.append(state)
        assert "2026-10-17" in str(state["prompt"]) and "risk tomorrow?" in str(state["prompt"])
        answer = "node_resolved" if "filing fact" in str(state.get("outcome")) else "research_required"
        return await _choice_stub({"entry": "reasoning_required", "assess": answer}, {})(state, questions)

    jev = JevClient(transport=_transport)
    assert kw._direct_op({"id": "route", "op": "route", "prompt": "risk tomorrow?"}, jev) == {
        "id": "route",
        "route": "reasoning_required",
        "personas": None,
    }
    assert kw._assess_entry({"id": "e1", "prompt": "risk tomorrow?"}, jev=jev) == {
        "id": "e1",
        "verdict": "research_required",
    }
    assert kw._assess_entry({"id": "e2", "prompt": "risk tomorrow?", "tool": ""}, jev=jev) == {
        "id": "e2",
        "verdict": "research_required",
    }
    assert len(states) == 1
    assert kw._direct_op(
        {
            "id": "e3",
            "op": "assess_entry",
            "prompt": "risk tomorrow?",
            "tool": "get_current_time",
            "arguments": {"ticker": "NVDA"},
            "result": {"ok": True, "content": "filing fact"},
        },
        jev,
    ) == {"id": "e3", "verdict": "node_resolved"}
    assert states[-1]["arguments"] == {"ticker": "NVDA"}
    assert kw._assess_entry(
        {"id": "e4", "prompt": "risk tomorrow?", "tool": "get_current_time", "arguments": "nope", "result": ["nope"]},
        jev=jev,
    ) == {"id": "e4", "verdict": "research_required"}
    assert states[-1]["arguments"] == {}
    assert "nope" not in str(states[-1]["outcome"])
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


@pytest.mark.parametrize("confidence", [True, 0.7])
def test_final_response_gate_validates_inventory_before_model_assessment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, confidence: JSONValue
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    sid = service.create_research("support?", "support?")
    seen: list[dict[str, JSONValue]] = []

    async def _transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        assert isinstance(state, dict)
        seen.append(state)
        return {
            "answers": {
                "support": {"type": "noul", "noul": 0.9},
                "evidence_state": {
                    "type": "choice",
                    "choice": "sufficient_support",
                    "confidence": 0.9,
                    "probabilities": {
                        "sufficient_support": 0.9,
                        "sufficient_contradiction": 0.0,
                        "conflicted": 0.0,
                        "insufficient": 0.1,
                    },
                },
            }
        }

    out = kw._direct_op(
        {
            "id": "f2",
            "op": "assess_final_response",
            "sessionId": sid,
            "text": " WSJ reported X. ",
            "confidence": confidence,
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
        JevClient(transport=_transport, data_root=tmp_path),
    )
    assert out is not None
    expected_confidence = None if isinstance(confidence, bool) else confidence
    assert out["id"] == "f2"
    assert out["support_answer"] == "yes" and out["evidence_state"] == "sufficient_support"
    assert out["confidence"] == expected_confidence
    assert len(seen) == 1
    state = seen[0]
    assert state["text"] == "WSJ reported X." and state["evidence"] == []
    assert state["confidence"] == expected_confidence
    assert state["claims"] == [
        {"text": "c" * 2000, "claim_type": "inference", "evidence_ids": [" ev-1 "], "confidence": 0.5},
        {"text": "y", "claim_type": "fact", "evidence_ids": [], "confidence": None},
        {"text": "z", "claim_type": "fact", "evidence_ids": [], "confidence": None},
    ]
    decisions = ResearchRepository().list_decisions(sid)
    assert len(decisions) == 1 and decisions[0].decision_type == "final_response_assessment"


@pytest.mark.parametrize(
    ("payload", "evidence", "confidence", "claim_type"),
    [
        (
            {"claim": {"text": "X.", "claim_type": "inference", "evidence_ids": ["ev-1", 7], "confidence": 0.5}},
            [{"id": "ev-1", "content": ""}],
            0.5,
            "inference",
        ),
        (
            {"claim": {"text": "X.", "claim_type": "inference"}, "confidence": 0.6},
            [],
            0.6,
            "inference",
        ),
        (
            {
                "claim": {"text": "X.", "claim_type": "inference", "evidence_ids": ["ev-1"], "confidence": 0.5},
                "evidence": [{"id": "ev-9", "content": "filing text", "other": 1}],
                "confidence": 0.8,
            },
            [{"id": "ev-9", "content": "filing text"}],
            0.5,
            "inference",
        ),
        ({"text": "X.", "claim_type": "attributed_report"}, [], None, "attributed_report"),
    ],
)
def test_final_claim_gate_assesses_available_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, JSONValue],
    evidence: list[JSONValue],
    confidence: float | None,
    claim_type: str,
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    sid = service.create_research("support?", "support?")
    seen: list[object] = []
    supported = any(isinstance(row, dict) and row.get("content") for row in evidence)

    async def _transport(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        seen.append(state)
        choice = "sufficient_support" if supported else "insufficient"
        question = questions["evidence_state"]
        assert isinstance(question, dict)
        options = question["criteria"]
        assert isinstance(options, dict)
        return {
            "answers": {
                "support": {"type": "noul", "noul": 0.9 if supported else 0.1},
                "evidence_state": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == choice) for key in options},
                },
            }
        }

    out = kw._direct_op(
        {"id": "claim", "op": "assess_final_claim", "sessionId": sid, **payload},
        JevClient(transport=_transport, data_root=tmp_path),
    )
    assert out is not None
    assert out["id"] == "claim"
    assert out["support_answer"] == ("yes" if supported else "no")
    assert out["evidence_state"] == ("sufficient_support" if supported else "insufficient")
    assert out["confidence"] == confidence
    assert seen == [{"text": "X.", "claim_type": claim_type, "evidence": evidence, "confidence": confidence}]
    decisions = ResearchRepository().list_decisions(sid)
    assert len(decisions) == 1 and decisions[0].decision_type == "final_claim_assessment"

    missing = kw._assess_final_claim({"id": "c3", "claim": {"text": "X."}})
    assert missing == {
        "id": "c3",
        "error": "claim_type required",
        "support": None,
        "support_answer": None,
        "evidence_state": "insufficient",
    }


def test_jev_kept_keeps_everything_but_an_explicit_reject() -> None:
    """A reject drops the proposal; a missing or malformed disposition keeps it."""
    proposals = _props()
    assert kw._jev_kept(proposals, {"s-q1": {"choice": "reject"}, "s-q2": {"choice": "gather_evidence"}}) == [
        proposals[1]
    ]
    assert kw._jev_kept(proposals, {}) == proposals
    assert kw._jev_kept(proposals, {"s-q1": "gather_evidence"}) == proposals


@pytest.mark.parametrize(
    ("hints", "queries", "identifiers", "reasoner_calls"),
    [
        ({}, ["ORACLE ORCL What is happening with Oracle now"], ["ORCL"], 1),
        ({"corrected_query": "fixed"}, ["ORACLE ORCL What is happening with Oracle now"], ["ORCL"], 1),
        ({"tickers": "NVDA"}, ["ORACLE ORCL What is happening with Oracle now"], ["ORCL"], 1),
        ({"tickers": ["NVDA"]}, ["ORACLE ORCL What is happening with Oracle now"], ["ORCL", "NVDA"], 1),
        (
            {"tickers": ["orcl", "ORCL", 7, " ", "nvda", "nvda"], "corrected_query": " fixed "},
            ["ORACLE ORCL What is happening with Oracle now", "fixed"],
            ["ORCL", "NVDA"],
            2,
        ),
        (
            {"tickers": ["NVDA"], "corrected_query": "What is happening with Oracle now"},
            ["ORACLE ORCL What is happening with Oracle now", "What is happening with Oracle now"],
            ["ORCL", "NVDA"],
            2,
        ),
        (
            {"tickers": ["NVDA"], "corrected_query": " "},
            ["ORACLE ORCL What is happening with Oracle now"],
            ["ORCL", "NVDA"],
            2,
        ),
    ],
)
def test_graph_intake_fetches_only_new_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hints: dict[str, object],
    queries: list[str],
    identifiers: list[str],
    reasoner_calls: int,
) -> None:
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"Oracle": "ORCL", "oracle": "ORCL"}.get(name)

    monkeypatch.setattr(_tools, "_resolve_company_to_ticker", _resolve)
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)

    def _no_cik(symbol: str) -> None:
        return None

    monkeypatch.setattr("app.sec.client.resolve_cik", _no_cik)
    calls: list[tuple[str, dict[str, object]]] = []
    prompts: list[str] = []

    async def _attempt(tool: str, args: dict[str, object], *rest: object) -> dict[str, object]:
        calls.append((tool, args))
        return {"tool": tool, "arguments": args, "error": "source unavailable"}

    def _decompose(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        prompts.append(prompt)
        return {
            "proposals": [
                {
                    "id": f"{objective_id}-q1",
                    "objectiveId": objective_id,
                    "question": "Filing risk?",
                    "dependsOn": [],
                    "whyItMatters": "w",
                }
            ],
            **(hints if len(prompts) == 1 else {"tickers": ["MSFT"], "corrected_query": "third query"}),
        }

    monkeypatch.setattr(kw, "_intake_attempt", _attempt)
    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _decompose)
    objective = "What is happening with Oracle now"
    sid = kw.run_graph_prompt(objective, jev=_JevDown())
    assert [args["query"] for tool, args in calls if tool == "search_web"] == queries
    assert [args["identifier"] for tool, args in calls if tool == "list_sec_filings"] == identifiers
    assert len(prompts) == reasoner_calls
    assert [node.question for node in ResearchRepository().list_nodes(sid)] == ["Filing risk?"]
    assert ResearchRepository().get_session(sid).objective == objective
    assert [decision.decision_type for decision in ResearchRepository().list_decisions(sid)] == ["intake_digest"] * len(
        identifiers
    )

    calls.clear()
    prompts.clear()
    sid = kw.run_graph_prompt("news for growth this week", jev=_JevDown(), setup_deadline=kw.time.perf_counter())
    assert [tool for tool, _args in calls] == ["search_web"]
    assert len(prompts) == 1
    assert ResearchRepository().list_nodes(sid)
    assert service.ready_nodes(sid)


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("can you give me buying advice", ["NVDA"]),
        ("Should I buy Apple?", ["AAPL"]),
        ("Should I buy MissingCo?", ["NVDA"]),
    ],
)
def test_graph_intake_uses_current_context_only_without_resolved_stock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, objective: str, expected: list[str]
) -> None:
    def _resolve(name: str) -> str | None:
        return {"Apple": "AAPL", "NVDA": "NVDA"}.get(name)

    def _no_cik(symbol: str) -> None:
        return None

    monkeypatch.setattr("app.tools._resolve_company_to_ticker", _resolve)
    monkeypatch.setattr("app.sec.client.resolve_cik", _no_cik)
    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)
    identifiers: list[object] = []

    async def _attempt(tool: str, args: dict[str, object], *rest: object) -> dict[str, object]:
        if tool == "list_sec_filings":
            identifiers.append(args["identifier"])
        return {"tool": tool, "arguments": args, "error": "source unavailable"}

    def _decompose(
        prompt: str, as_of: object, sid: str, digest: str, *rest: object
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return (
            [{"id": "q1", "objectiveId": sid, "question": prompt, "dependsOn": [], "whyItMatters": "w"}],
            {},
        )

    monkeypatch.setattr(kw, "_intake_attempt", _attempt)
    monkeypatch.setattr(kw, "_reasoner_decompose_with_retry", _decompose)
    sid = kw.run_graph_prompt(objective, ticker="NVDA", jev=_JevDown(), setup_deadline=kw.time.perf_counter())
    assert identifiers == expected
    assert ResearchRepository().get_session(sid).objective == objective
    assert [node.question for node in ResearchRepository().list_nodes(sid)] == [objective]


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
    denied = kw._run({"id": "rg", "prompt": "q?"}, jev=JevClient())
    assert denied["id"] == "rg"
    assert denied["error"] == "registry guard forbids portfolio tools: x"
    assert denied["terminal"] == {"category": "provider_error", "message": denied["error"]}


def test_setup_failure_carries_session_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-guard setup failure carries the session-setup prefix under the caller's id."""

    def _other(*a: object, **k: object) -> str:
        raise RuntimeError("store down")

    monkeypatch.setattr(kw, "run_graph_prompt", _other)
    denied = kw._run({"id": "rg", "prompt": "q?"}, jev=JevClient())
    assert denied["id"] == "rg"
    assert denied["error"] == "session setup failed: store down"
    assert denied["terminal"] == {"category": "provider_error", "message": denied["error"]}


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        ("stalled", "waiting", "waiting"),
        ("stalled", None, "stalled: no progress"),
        ("stalled", 7, "stalled: no progress"),
        ("stalled", "x" * 600, "x" * 500),
        ("complete", "waiting", None),
    ],
)
def test_run_reports_bounded_stall_reason(
    monkeypatch: pytest.MonkeyPatch, status: str, reason: JSONValue, expected: str | None
) -> None:
    _empty_repo(monkeypatch)
    out = _run_with(monkeypatch, {"status": status, "reason": reason})
    assert out["stalled"] is (status == "stalled")
    assert out.get("stalled_reason") == expected
    assert out["escalated"] is (status == "stalled")


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


def test_run_response_cannot_include_another_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    own = service.create_research("own question?", "own objective")
    other = service.create_research("other question?", "other objective")
    repo = ResearchRepository()
    for sid, label in ((own, "own"), (other, "other")):
        repo.save_evidence({"evidence_id": f"ev-{label}", "session_id": sid, "content": f"{label} fact"})
        service.create_node(sid, f"{label} node?", "w")
        service.record_decision(sid, "scope", {}, {}, {"owner": label})

    def _graph(prompt: str, as_of: object = None, **hooks: object) -> str:
        return own

    async def _schedule(sid: str, **hooks: object) -> dict[str, object]:
        assert sid == own
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(kw, "run_graph_prompt", _graph)
    monkeypatch.setattr(sched, "run", _schedule)
    out = kw._run({"id": "scoped", "prompt": "own question?"}, jev=JevClient())
    assert out["sessionId"] == own
    assert out["evidence"] == [{"id": "ev-own", "content": "own fact"}]
    nodes = out["nodes"]
    assert isinstance(nodes, list)
    assert [node["question"] for node in nodes if isinstance(node, dict)] == ["own node?"]
    decisions = out["decisions"]
    assert isinstance(decisions, list)
    assert [decision["selected"] for decision in decisions if isinstance(decision, dict)] == [{"owner": "own"}]


def test_shutdown_cancels_queued_work_without_waiting_for_running_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release = threading.Event()
    entered = [threading.Event(), threading.Event()]
    pools = [ThreadPoolExecutor(max_workers=1), ThreadPoolExecutor(max_workers=1)]

    def _running(started: threading.Event) -> None:
        started.set()
        release.wait(2)

    running = [pool.submit(_running, started) for pool, started in zip(pools, entered)]
    queued = [pool.submit(lambda: None) for pool in pools]
    monkeypatch.setattr(sched, "_SEC_POOL", pools[0])
    monkeypatch.setattr(sched, "_TOOL_POOL", pools[1])
    try:
        assert all(started.wait(1) for started in entered)
        kw._shutdown_pools()
        assert all(future.cancelled() for future in queued)
        assert all(not future.done() for future in running)
    finally:
        release.set()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("as_of", ["2024-05-01", 7])
def test_run_persists_scope_and_uses_injected_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, as_of: JSONValue
) -> None:
    from datetime import UTC, datetime

    from app.research import models, service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)

    def _no_subject(objective: str, supplied_ticker: object = None) -> None:
        return None

    monkeypatch.setattr(sched, "_objective_subject_ticker", _no_subject)
    now = datetime(2026, 10, 17, tzinfo=UTC)
    monkeypatch.setattr(models, "utcnow", lambda: now)
    monkeypatch.setattr(kw, "utcnow", lambda: now)
    seen_dates: list[object] = []
    prompts: list[str] = []

    async def _authority(state: object, questions: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        assert isinstance(state, dict)
        assert state["objective"] == {"prompt": "Exact user objective?"}
        picks = {qid: "reject" if qid.endswith("-q3") else "analyze" for qid in questions}
        return await _choice_stub(picks, {})(state, questions)

    jev = JevClient(transport=_authority, data_root=tmp_path)

    async def _attempt(
        tool: str,
        args: dict[str, object],
        sid: str,
        session: object,
        registry: object,
        client: object,
        kernel: object,
        cutoff: object,
    ) -> dict[str, object]:
        assert client is jev
        seen_dates.append(cutoff)
        return {"tool": tool, "arguments": args, "error": "source unavailable"}

    def _decompose(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        prompts.append(prompt)
        return {
            "proposals": [
                {
                    "id": f"{objective_id}-q{index}",
                    "objectiveId": objective_id,
                    "question": question,
                    "dependsOn": [],
                    "whyItMatters": "w",
                }
                for index, question in enumerate(["Other angle?", "Exact user objective?", "Tangent?"], 1)
            ]
        }

    async def _schedule(sid: str, **hooks: object) -> dict[str, object]:
        assert hooks["jev"] is jev
        for node in service.ready_nodes(sid):
            service.resolve_node(sid, node.node_id)
        return {"status": "complete", "nodes": [], "incomplete_guard": False}

    monkeypatch.setattr(kw, "_intake_attempt", _attempt)
    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _decompose)
    monkeypatch.setattr(sched, "run", _schedule)
    out = kw._run({"id": "scope", "prompt": "Exact user objective?", "asOf": as_of, "ticker": "NVDA"}, jev=jev)
    sid = out["sessionId"]
    assert isinstance(sid, str)
    session = ResearchRepository().get_session(sid)
    assert session.ticker == "NVDA"
    assert session.objective == "Exact user objective?"
    assert seen_dates == [as_of if isinstance(as_of, str) else None]
    if isinstance(as_of, str):
        assert session.as_of is not None and session.as_of.date().isoformat() == as_of
    else:
        assert session.temporal_scope["mode"] == "latest-available"
    assert len(prompts) == 1
    assert "Exact user objective?" in prompts[0] and "2026-10-17" in prompts[0]
    assert [node.question for node in ResearchRepository().list_nodes(sid)] == ["Other angle?", "Exact user objective?"]
    assert out["unresolved"] == [] and out["escalated"] is False
    assert [decision.decision_type for decision in ResearchRepository().list_decisions(sid)] == [
        "intake_digest",
        "proposal_disposition",
    ]


def test_graph_intake_persists_citable_sources_dedupes_and_settles_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.research import service
    from app.tool_runtime import outcome_from_result

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    monkeypatch.setattr(sched, "build_registry", list)

    def _no_subject(objective: str, supplied_ticker: object = None) -> None:
        return None

    monkeypatch.setattr(sched, "_objective_subject_ticker", _no_subject)
    monkeypatch.setattr("app.sec.client.resolve_cik", _no_subject)
    attempts: dict[str, str] = {}
    prompts: list[str] = []
    payloads: dict[str, dict[str, object]] = {
        "list_sec_filings": {
            "filings": [{"form": "10-Q", "accession_no": "0001045810-26-000001", "title": "Revenue filing"}]
        },
        "get_short_interest": {"records": [{"ticker": "NVDA", "v": 123}], "query": {"ticker": "NVDA"}},
        "search_web": {"evidence": [{"url": "https://example.test/report", "highlight": "Independent filing report"}]},
    }

    async def _attempt(
        tool: str,
        args: dict[str, object],
        sid: str,
        session: object,
        registry: object,
        jev: object,
        kernel: sched._Kernel,
        as_of: str | None,
    ) -> dict[str, object]:
        domain = "SEC" if tool == "list_sec_filings" else "WEB" if tool == "search_web" else "FINRA"
        job_id = sched._attempt_job(kernel, sid, domain)
        attempts[job_id] = tool
        result: dict[str, object]
        if tool in payloads:
            result = dict(payloads[tool])
            if tool == "list_sec_filings" and args.get("identifier") == "MSFT":
                result["filings"] = [{"form": "10-Q", "accession_no": "0000789019-26-000001", "title": "MSFT filing"}]
            ref = f"{sid}:tr:{tool}:{args.get('identifier', '')}"
            service.persist_tool_result(sid, job_id, tool, ref, result)
            result["tool_result_id"] = ref
        elif tool == "get_reg_sho_volume":
            result = {"ok": True, "note": "no citable rows"}
        else:
            result = {"ok": False, "error": "source failed", "error_type": "tool_error"}
        outcome = outcome_from_result(tool, result)
        return sched._success_attempt(tool, args, outcome, result, domain, "fixed intake", job_id)

    def _decompose(self: object, prompt: str, objective_id: str) -> dict[str, object]:
        prompts.append(prompt)
        return {
            "proposals": [
                {
                    "id": f"{objective_id}-q1",
                    "objectiveId": objective_id,
                    "question": "What does the filing establish?",
                    "dependsOn": [],
                    "whyItMatters": "Check the admitted sources.",
                }
            ],
            "tickers": ["MSFT"],
        }

    monkeypatch.setattr(kw, "_intake_attempt", _attempt)
    monkeypatch.setattr("app.reasoner_client.ReasonerClient.decompose", _decompose)
    sid = kw.run_graph_prompt("What happened?", jev=_JevDown(), watchlist=["NVDA"])
    repo = ResearchRepository()
    evidence = repo.list_evidence(sid)
    assert len(evidence) == 4
    ids = {row["evidence_id"] for row in evidence}
    assert len(prompts) == 2 and all(str(eid) in prompts[-1] for eid in ids)
    assert [node.question for node in repo.list_nodes(sid)] == ["What does the filing establish?"]
    decisions = repo.list_decisions(sid)
    assert [decision.decision_type for decision in decisions] == ["intake_digest", "intake_digest"]
    selected = decisions[0].selected
    assert isinstance(selected, dict)
    stats = selected["stats"]
    assert isinstance(stats, dict)
    assert stats["outcomes"] == {"admitted": 3, "duplicate": 0, "not_citable": 1, "error": 1, "timeout": 0}
    admitted_ids = stats["admitted_ids"]
    assert isinstance(admitted_ids, list) and len(admitted_ids) == 3 and set(admitted_ids) < ids
    last_selected = decisions[-1].selected
    assert isinstance(last_selected, dict)
    last_stats = last_selected["stats"]
    assert isinstance(last_stats, dict)
    assert last_stats["outcomes"] == {"admitted": 1, "duplicate": 0, "not_citable": 0, "error": 0, "timeout": 0}
    assert {str(row["provenance"]["kind"]) for row in evidence if isinstance(row.get("provenance"), dict)} == {
        "sec_record",
        "finra_record",
        "web_source",
    }
    for job_id, tool in attempts.items():
        assert repo.get_job(job_id).status == ("failed" if tool == "get_threshold_securities" else "completed")

    admitted, raw, repeated = asyncio.run(
        kw._intake_round(
            "What happened?",
            [],
            sid,
            repo.get_session(sid).to_dict(),
            [],
            None,
            sched._Kernel(),
            None,
            watchlist=["NVDA"],
        )
    )
    assert {row["evidence_id"] for row in admitted} == ids
    assert {row["evidence_id"] for row in repo.list_evidence(sid)} == ids
    assert repeated["admitted_ids"] == []
    assert repeated["outcomes"] == {"admitted": 0, "duplicate": 3, "not_citable": 1, "error": 1, "timeout": 0}
    assert len(raw) == 5
    coverage = repeated["watchlist"]
    assert isinstance(coverage, list) and len(coverage) == 4
    by_tool = {row["tool"]: row for row in coverage if isinstance(row, dict)}
    assert by_tool["list_sec_filings"]["evidence_id"] in ids
    assert by_tool["get_short_interest"]["evidence_id"] in ids
    assert by_tool["get_threshold_securities"]["status"] == "error"
    assert "evidence_id" not in by_tool["get_threshold_securities"]
    for job_id, tool in attempts.items():
        assert repo.get_job(job_id).status == ("failed" if tool == "get_threshold_securities" else "completed")


@pytest.mark.parametrize(("call_timeout", "round_timeout"), [(60.0, 120.0), (120.0, 60.0)])
def test_intake_round_cancels_overdue_sources_without_losing_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, call_timeout: float, round_timeout: float
) -> None:
    from app.research import service

    monkeypatch.setenv("RESEARCH_DB_PATH", str(tmp_path / "r.sqlite"))
    sid = service.create_research("my question?", "my question?")
    repo = ResearchRepository()
    (bootstrap,) = repo.list_jobs(sid)
    service.cancel_job(bootstrap.job_id)
    sources: list[asyncio.Task[None]] = []
    started: list[tuple[str, dict[str, object]]] = []
    cancelled: list[tuple[str, dict[str, object]]] = []
    all_started = asyncio.Event()
    all_cancelled = asyncio.Event()

    def _no_cik(symbol: str) -> None:
        return None

    async def _hang(invoke: object, to_outcome: object, tool: str, args: Mapping[str, object], *rest: object) -> None:
        async def _source() -> None:
            started.append((tool, dict(args)))
            if len(started) == 6:
                all_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append((tool, dict(args)))
                if len(cancelled) == 6:
                    all_cancelled.set()
                raise

        source = asyncio.create_task(_source())
        sources.append(source)
        await source
        raise AssertionError("unreachable")

    monkeypatch.setattr("app.sec.client.resolve_cik", _no_cik)
    monkeypatch.setattr(sched, "_invoke_attempt_tool", _hang)
    monkeypatch.setattr(kw, "_INTAKE_CALL_TIMEOUT_S", call_timeout)
    monkeypatch.setattr(kw, "_INTAKE_BUDGET_S", round_timeout)

    async def _scenario() -> tuple[list[dict[str, JSONValue]], list[dict[str, object]], dict[str, object]]:
        loop = asyncio.get_running_loop()
        actual_time = loop.time
        clock_shift = 0.0
        with monkeypatch.context() as clock:
            clock.setattr(loop, "time", lambda: actual_time() + clock_shift)
            intake = asyncio.create_task(
                kw._intake_round(
                    "my question?",
                    ["ORCL"],
                    sid,
                    {"session_id": sid},
                    [],
                    None,
                    sched._Kernel(),
                    None,
                    watchlist=["NVDA"],
                )
            )
            try:
                await asyncio.wait_for(all_started.wait(), timeout=5.0)
                jobs = [job for job in repo.list_jobs(sid) if job.job_id != bootstrap.job_id]
                assert len(jobs) == 6 and all(job.status == "running" for job in jobs)
                clock_shift = min(call_timeout, round_timeout) + 1.0
                result = await asyncio.wait_for(intake, timeout=5.0)
                await asyncio.wait_for(all_cancelled.wait(), timeout=5.0)
                await asyncio.gather(*sources, return_exceptions=True)
                return result
            finally:
                intake.cancel()
                for source in sources:
                    source.cancel()
                await asyncio.gather(intake, *sources, return_exceptions=True)

    admitted, raw, stats = asyncio.run(_scenario())
    assert admitted == [] and repo.list_evidence(sid) == []
    assert len(sources) == len(started) == len(cancelled) == 6
    assert all(call in cancelled for call in started)
    jobs = [job for job in repo.list_jobs(sid) if job.job_id != bootstrap.job_id]
    assert len(jobs) == 6 and all(job.status == "failed" for job in jobs)
    assert all(job.failure is not None and job.failure.category == "timeout" for job in jobs)
    assert repo.get_job(bootstrap.job_id).status == "cancelled"
    assert stats["outcomes"] == {"admitted": 0, "duplicate": 0, "not_citable": 0, "error": 0, "timeout": 6}
    assert stats["admitted_ids"] == []
    assert any(row["tool"] == "search_web" and row["arguments"] == {"query": "my question?"} for row in raw)
    assert any(row["tool"] == "list_sec_filings" and "ORCL" in str(row["arguments"]) for row in raw)
    assert all(isinstance(row["record"], dict) and row["record"]["error_type"] == "timeout" for row in raw)
    coverage = stats["watchlist"]
    assert isinstance(coverage, list) and len(coverage) == 4
    assert all(
        isinstance(row, dict) and row["symbol"] == "NVDA" and row["status"] == "timeout" and "evidence_id" not in row
        for row in coverage
    )


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


@pytest.mark.parametrize("payload", [{"objective": 7}, {"objective": None, "prompt": 7}])
def test_arguments_non_string_objective_cannot_seed_a_call(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, JSONValue]
) -> None:
    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _needle_withhold)
    out = kw._arguments({"id": "bad-objective", "tool": "list_sec_filings", **payload})
    assert out == {"id": "bad-objective", "error": "needle arguments failed"}


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


def test_arguments_node_session_id_carries_without_arguments_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node-carried session_id shortcuts without Needle when arguments holds no mapping."""

    def _boom(**kwargs: object) -> object:
        raise AssertionError("Needle must not run for a sid carry")

    monkeypatch.setattr(kw, "_shared_needle_generate", lambda: _boom)
    out = kw._arguments({"id": "n1", "tool": "research_status", "node": {"session_id": "rs:5"}})
    assert out["arguments"] == {"session_id": "rs:5"}


def test_assess_final_response_blank_text_keeps_request_id() -> None:
    out = kw._assess_final_response({"id": "fb", "text": "  "})
    assert out["id"] == "fb"
    assert out["error"] == "text required"
    assert out["evidence_state"] == "insufficient"


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
    out = kw._route({"id": "r1", "op": "route", "prompt": "explain diversification", "ticker": "NVDA"}, jev=jev)
    assert out == {"id": "r1", "route": "research_required", "personas": None}


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


def test_decompose_repairs_an_empty_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reasoner retries an empty response once and returns the repaired proposals."""
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
    assert len(prompts) == 2


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
