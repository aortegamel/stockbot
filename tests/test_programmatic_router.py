"""Programmatic router: 5 live questions route deterministically; JEV only on ambiguity."""

import asyncio
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import override

import pytest

from app.decision_client import JevClient
from app.research import scheduler
from app.research.models import DecisionRecord, JSONValue, ToolDecision
from app.research.programmatic_router import (
    _ranked,
    _registry_names,
    _router_pick,
    _signals,
    programmatic_route,
    programmatic_select_round,
)


class _Kernel(scheduler._Kernel):
    """Decision log only; every select path persists through record_decision."""

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


class _BoomJev(JevClient):
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
        raise AssertionError("router must not call JEV")


def _pick(objective: str) -> tuple[str, str | None]:
    reg = scheduler.build_registry()
    kernel = _Kernel()
    act, dec = asyncio.run(
        programmatic_select_round(
            _BoomJev(),
            kernel,
            "s1",
            "n1",
            {"session_id": "s1", "objective": objective},
            _node(objective),
            reg,
            [],
            [],
        )
    )
    assert kernel.recorded == [("tool_selection", dec.tool_name)]
    return act, dec.tool_name


def test_route_fast_path_marks_research():
    assert programmatic_route("What drove NVDA revenue last quarter?") == "research_required"
    assert programmatic_route("hello") is None


def test_route_clock_and_advice_fast_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clock asks single-shot; pure advice answers direct; mixed/insider asks research."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(
        _tools,
        "_resolve_company_to_ticker",
        _resolve,
    )
    assert programmatic_route("what time is it") == "get_current_time"
    assert programmatic_route("should I buy NVDA") == "reasoning_required"
    assert programmatic_route("should I buy NVDA after its Q3 revenue") == "research_required"
    assert programmatic_route("did NVDA insiders sell") == "research_required"


def test_small_talk_never_creates_session() -> None:
    """thanks / never mind short-circuit to no_session (plan entry route)."""
    from app.research.programmatic_router import programmatic_route

    assert programmatic_route("thanks") == "no_session"
    assert programmatic_route("never mind") == "no_session"
    assert programmatic_route("What drove NVDA revenue last quarter?") == "research_required"


def test_five_live_questions_first_hop():
    assert _pick("What drove NVDA revenue last quarter?") == (
        "invoke",
        "list_sec_filings",
    )
    assert _pick("Which filings mention Elon Musk, and who actually filed them?") == (
        "invoke",
        "search_sec_filings",
    )
    assert _pick("Has anyone on Apple's executive team traded company stock recently?") == (
        "invoke",
        "get_insider_activity",
    )
    assert _pick("How heavily are traders betting against Apple at the moment?") == (
        "invoke",
        "get_short_interest",
    )
    assert _pick("What does FINRA Reg SHO daily short volume show for Apple this week?") == (
        "invoke",
        "get_reg_sho_volume",
    )


def test_ambiguous_falls_back_to_jev_whole_registry():
    seen: dict[str, int] = {}

    class _J(JevClient):
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
            seen["n"] = len(registry or [])
            return ToolDecision(action="reason", probabilities={}, confidence=None)

    reg = scheduler.build_registry()
    act, _ = asyncio.run(
        programmatic_select_round(
            _J(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": "halp money stuff?"},
            _node("halp money stuff?"),
            reg,
            [],
            [],
        )
    )
    assert act == "reason" and seen["n"] == len(reg)


def test_mention_check_selects_bounded_without_jev() -> None:
    """One-mention ask resolves by rule 9; a JEV stub that fails must never run."""

    reg = scheduler.build_registry()
    names = _registry_names(reg)
    objective = "Which 10-K mentions Jensen Huang?"
    action, dec = asyncio.run(
        programmatic_select_round(
            _BoomJev(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": objective},
            _node(objective),
            reg,
            [],
            [],
        )
    )
    assert isinstance(dec, ToolDecision)
    assert (action, dec.tool_name) == ("invoke", "search_sec_filings_bounded")
    ranked = [n for _, n in _ranked(" ".join(objective.split()), {"10k", "mention"}, names)]
    assert not ({"search_sec_filings", "search_sec_filings_bounded"} <= set(ranked))


def test_coverage_ask_stays_exhaustive() -> None:
    """Coverage asks keep the exhaustive variant."""
    assert _pick("Who filed all filings that mention Jensen Huang?") == (
        "invoke",
        "search_sec_filings",
    )


def test_variant_falls_back_to_registered_member() -> None:
    """A missing wanted variant falls back to the registered pair member."""
    from app.research.programmatic_router import _pick_variant

    assert (
        _pick_variant(
            "search_sec_filings",
            "which 10k mention",
            {"mention"},
            {"search_sec_filings"},
        )
        == "search_sec_filings"
    )
    assert (
        _pick_variant(
            "search_sec_filings_bounded",
            "who filed all",
            {"all"},
            {"search_sec_filings_bounded"},
        )
        == "search_sec_filings_bounded"
    )
    # Rule 5 passes the exhaustive key; bounded-only registry must not return it.
    assert (
        _pick_variant(
            "search_sec_filings",
            "who filed all",
            {"all"},
            {"search_sec_filings_bounded"},
        )
        == "search_sec_filings_bounded"
    )


def test_jev_fallback_rewritten_to_coverage_variant() -> None:
    """JEV's bounded pick becomes exhaustive on a coverage ask."""

    class _J(JevClient):
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
            return ToolDecision(
                action="invoke",
                tool_name="search_sec_filings_bounded",
                tool_names=("search_sec_filings_bounded",),
                probabilities={"search_sec_filings_bounded": 0.9},
                confidence=0.9,
            )

    reg = scheduler.build_registry()
    # "all companies" carries coverage but no mention/filing trigger, so no rule
    # fires and the JEV fallback runs; the bounded pick then rewrites to exhaustive.
    ask = "zqxj vbnm all companies qqqq"

    norm, toks = _signals(f"{ask} {ask}")
    names = _registry_names(reg)
    ranked = _ranked(norm, toks, names)
    assert _router_pick(f"{ask} {ask}", norm, toks, names, ranked, [], []) is None
    action, dec = asyncio.run(
        programmatic_select_round(
            _J(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": ask},
            _node(ask),
            reg,
            [],
            [],
        )
    )
    assert isinstance(dec, ToolDecision)
    assert (action, dec.tool_name) == ("invoke", "search_sec_filings")
    assert dec.tool_names == ("search_sec_filings",)


def test_short_interest_beats_intake_accession() -> None:
    """Intake evidence with an 8-K accession never diverts a short ask to get_sec_filing."""
    intake: list[JSONValue] = [
        {
            "id": "ev:1",
            "content": "filed 8-K accession 0000320193-25-000079, see filing",
        }
    ]
    reg = scheduler.build_registry()
    objective = "What's AAPL's short interest?"
    action, dec = asyncio.run(
        programmatic_select_round(
            _BoomJev(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": objective},
            _node(objective),
            reg,
            intake,
            [],
        )
    )
    assert isinstance(dec, ToolDecision)
    assert (action, dec.tool_name) == ("invoke", "get_short_interest")


def test_form_mentions_route_person_mention() -> None:
    """10-Q and 8-K mentions resolve by rule 5 without JEV."""

    reg = scheduler.build_registry()
    for objective in (
        "Which 10-Q mentions Jensen Huang?",
        "Which 8-K mentions Elon Musk?",
    ):
        action, dec = asyncio.run(
            programmatic_select_round(
                _BoomJev(),
                _Kernel(),
                "s1",
                "n1",
                {"session_id": "s1", "objective": objective},
                _node(objective),
                reg,
                [],
                [],
            )
        )
        assert isinstance(dec, ToolDecision)
        assert (action, dec.tool_name) == ("invoke", "search_sec_filings_bounded")


_TEN_Q_REVENUE = "What did NVDA's latest 10-Q disclose about revenue? Use stockbot, bearbot, and bullbot."
_LISTING: dict[str, JSONValue] = {
    "tool": "list_sec_filings",
    "arguments": {"identifier": "NVDA"},
    "error": None,
    "outcome_summary": "filing 0001045810-26-000075 10-Q",
}


def test_ten_q_revenue_lists_then_reads_document_without_jev() -> None:
    """Live 10-Q revenue prompt: list, then read the listed document; JEV (node_resolved) never runs first."""

    class _BoomJev(JevClient):
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
        ) -> ToolDecision:
            raise AssertionError("router must not call JEV")

    reg = scheduler.build_registry()
    # Intake accessions never skip the node's own listing (same as force-open).
    intake: list[JSONValue] = [{"id": "ev:1", "content": "NVDA 8-K accession 0001045810-26-000078"}]

    def _select(attempts: list[dict[str, JSONValue]]) -> tuple[str, str | None]:
        action, dec = asyncio.run(
            programmatic_select_round(
                _BoomJev(),
                _Kernel(),
                "s1",
                "n1",
                {"session_id": "s1", "objective": _TEN_Q_REVENUE},
                _node(_TEN_Q_REVENUE),
                reg,
                intake,
                attempts,
            )
        )
        assert isinstance(dec, ToolDecision)
        return action, dec.tool_name

    assert _select([]) == ("invoke", "list_sec_filings")
    assert _select([_LISTING]) == ("invoke", "get_sec_document")


def test_revenue_route_hands_off_after_document_read() -> None:
    """One successful document read ends the route; quarter wording reads too; plain revenue never routes."""
    reg = scheduler.build_registry()
    names = _registry_names(reg)
    read: dict[str, JSONValue] = {
        "tool": "get_sec_document",
        "arguments": {"accession_no": "0001045810-26-000075"},
        "error": None,
    }

    def _rule(ask: str, attempts: list[dict[str, JSONValue]]) -> tuple[str, str] | None:
        text = f"{ask} {ask}"
        norm, toks = _signals(text)
        return _router_pick(text, norm, toks, names, _ranked(norm, toks, names), attempts, [])

    assert _rule(_TEN_Q_REVENUE, [_LISTING, read]) is None
    assert _rule("What drove NVDA revenue last quarter?", [_LISTING]) == (
        "get_sec_document",
        "revenue-quarter:doc",
    )
    assert _rule("What did NVDA disclose about revenue?", [_LISTING]) is None


def test_ten_q_revenue_listing_without_source_ten_q_lists_again() -> None:
    """Request forms and nearby 10-Q text cannot label an 8-K source row."""
    acc8 = "0001045810-26-000078"
    listing: dict[str, JSONValue] = {
        "tool": "list_sec_filings",
        "arguments": {"identifier": "NVDA", "forms": ["10-Q", "10-K", "8-K"]},
        "error": None,
        "result": {"filings": [{"accession_no": acc8, "form": "8-K", "filed_at": "2026-09-01"}]},
        "outcome_summary": f"Requested latest 10-Q. Returned 8-K {acc8}.",
    }

    class _BoomJev(JevClient):
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
        ) -> ToolDecision:
            raise AssertionError("router must list filings without calling JEV")

    action, dec = asyncio.run(
        programmatic_select_round(
            _BoomJev(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": _TEN_Q_REVENUE},
            _node(_TEN_Q_REVENUE),
            scheduler.build_registry(),
            [],
            [listing],
        )
    )
    assert isinstance(dec, ToolDecision)
    assert (action, dec.tool_name) == ("invoke", "list_sec_filings")


@pytest.mark.parametrize("carry_only", [False, True])
@pytest.mark.parametrize("error", ["query not found", "duplicate tool call with identical arguments"])
def test_failed_ten_q_document_read_uses_jev_without_direct_retry(carry_only: bool, error: str) -> None:
    """A failed document call leaves direct routing, even when the registry contains only carry tools."""
    reg = scheduler.build_registry()
    if carry_only:
        reg = [entry for entry in reg if entry["name"] in {"get_sec_document", "get_sec_filing", "list_sec_documents"}]
    rejected: dict[str, JSONValue] = {
        "tool": "get_sec_document",
        "arguments": {"accession_no": "0001045810-26-000075", "query": "revenue"},
        "error": error,
    }
    attempts = [_LISTING, rejected]
    seen: list[tuple[list[Mapping[str, JSONValue]], list[JSONValue]]] = []

    class _Jev(JevClient):
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
        ) -> ToolDecision:
            seen.append((list(registry or []), list(attempts or [])))
            return ToolDecision(action="reason", probabilities={}, confidence=None)

    names = _registry_names(reg)
    text = f"{_TEN_Q_REVENUE} {_TEN_Q_REVENUE}"
    norm, toks = _signals(text)
    assert _router_pick(text, norm, toks, names, _ranked(norm, toks, names), attempts, []) is None
    action, dec = asyncio.run(
        programmatic_select_round(
            _Jev(),
            _Kernel(),
            "s1",
            "n1",
            {"session_id": "s1", "objective": _TEN_Q_REVENUE},
            _node(_TEN_Q_REVENUE),
            reg,
            [],
            attempts,
        )
    )
    assert isinstance(dec, ToolDecision)
    assert action == "reason" and dec.tool_name is None
    assert len(seen) == 1
    selected_registry, prior = seen[0]
    assert prior == attempts
    assert _registry_names(selected_registry) == names
    assert "get_sec_document" in names
    if carry_only:
        assert "list_sec_filings" not in names


def test_insider_verbs_stay_research_and_advice_stays_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare sell/hold tokens route research; should+verb and good-investment route direct."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"TSLA": "TSLA", "NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(
        _tools,
        "_resolve_company_to_ticker",
        _resolve,
    )
    assert programmatic_route("Did Musk sell TSLA shares") == "research_required"
    assert programmatic_route("Who holds the most NVDA shares") == "research_required"
    assert programmatic_route("should I buy NVDA") == "reasoning_required"
    assert programmatic_route("is NVDA a good investment") == "reasoning_required"


def test_tokenizer_merges_form_pairs_only() -> None:
    """(10,k/q) and (8,k) merge; other 10/8 pairs never merge."""
    from app.tools import _normalize_discovery_text as _norm

    assert "10q" in _norm("Which 10-Q mentions X?")
    assert "8k" in _norm("filed 8-K accession here")
    assert "10k" in _norm("Which 10-K mentions X?")
    assert "10key" not in _norm("10 key facts") and "10" in _norm("10 key facts")


def test_prefixed_clock_routes_clock_with_ticker_channel() -> None:
    """Exact NVDA clock regression: ticker channel never diverts the original clock ask."""
    assert programmatic_route("What time is it?", ticker="NVDA") == "get_current_time"
    assert programmatic_route("should I buy NVDA", ticker="NVDA") == "reasoning_required"


def test_explicit_persona_market_research_keeps_ticker_grounding() -> None:
    """Explicit-persona market asks still resolve the ticker subject for research."""
    assert programmatic_route("Bullbot view on ORCL revenue", ticker="ORCL") == "research_required"
