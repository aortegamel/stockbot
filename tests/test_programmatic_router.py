"""Programmatic router: 5 live questions route deterministically; JEV only on ambiguity."""

import asyncio

from app.research import scheduler
from app.research.programmatic_router import (
    _ranked,
    _registry_names,
    _router_pick,
    _signals,
    programmatic_route,
    programmatic_select_round,
)


class _Kernel:
    def record_decision(self, sid, dtype, **kw):
        pass


def _node(question):
    from types import SimpleNamespace

    return SimpleNamespace(node_id="n1", session_id="s1", question=question, why_it_matters="w")


def _pick(objective):
    reg = scheduler.build_registry()
    act, dec = asyncio.run(
        programmatic_select_round(
            None,
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
    return act, dec.tool_name


def test_route_fast_path_marks_research():
    assert programmatic_route("What drove NVDA revenue last quarter?") == "research_required"
    assert programmatic_route("hello") is None


def test_route_clock_and_advice_fast_paths() -> None:
    """Clock asks single-shot; pure advice answers direct; mixed/insider asks research."""
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
    assert _pick("What drove NVDA revenue last quarter?") == ("invoke", "list_sec_filings")
    assert _pick("Which filings mention Elon Musk, and who actually filed them?") == ("invoke", "search_sec_filings")
    assert _pick("Has anyone on Apple's executive team traded company stock recently?") == (
        "invoke",
        "get_insider_activity",
    )
    assert _pick("How heavily are traders betting against Apple at the moment?") == ("invoke", "get_short_interest")
    assert _pick("What does FINRA Reg SHO daily short volume show for Apple this week?") == (
        "invoke",
        "get_reg_sho_volume",
    )


def test_ambiguous_falls_back_to_jev_subset():
    from app.research.models import ToolDecision

    seen = {}

    class _J:
        async def select_tool(self, objective, node, registry, evidence, attempts, session_id=None, job_id=None):
            seen["n"] = len(registry)
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

    class _BoomJev:
        async def select_tool(self, *a: object, **k: object) -> object:
            raise AssertionError("router must not call JEV")

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
    assert (action, dec.tool_name) == ("invoke", "search_sec_filings_bounded")
    ranked = [n for _, n in _ranked(" ".join(objective.split()), {"10k", "mention"}, names)]
    assert not ({"search_sec_filings", "search_sec_filings_bounded"} <= set(ranked))


def test_coverage_ask_stays_exhaustive() -> None:
    """Coverage asks keep the exhaustive variant."""
    assert _pick("Who filed all filings that mention Jensen Huang?") == ("invoke", "search_sec_filings")


def test_variant_falls_back_to_registered_member() -> None:
    """A missing wanted variant falls back to the registered pair member."""
    from app.research.programmatic_router import _pick_variant

    assert (
        _pick_variant("search_sec_filings", "which 10k mention", {"mention"}, {"search_sec_filings"})
        == "search_sec_filings"
    )
    assert (
        _pick_variant("search_sec_filings_bounded", "who filed all", {"all"}, {"search_sec_filings_bounded"})
        == "search_sec_filings_bounded"
    )
    # Rule 5 passes the exhaustive key; bounded-only registry must not return it.
    assert (
        _pick_variant("search_sec_filings", "who filed all", {"all"}, {"search_sec_filings_bounded"})
        == "search_sec_filings_bounded"
    )


def test_jev_fallback_rewritten_to_coverage_variant() -> None:
    """JEV's bounded pick becomes exhaustive on a coverage ask."""
    from app.research.models import ToolDecision

    class _J:
        async def select_tool(self, objective, node, registry, evidence, attempts, session_id=None, job_id=None):
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
    assert (action, dec.tool_name) == ("invoke", "search_sec_filings")
    assert dec.tool_names == ("search_sec_filings",)


def test_short_interest_beats_intake_accession() -> None:
    """Intake evidence with an 8-K accession never diverts a short ask to get_sec_filing."""
    intake = [{"id": "ev:1", "content": "filed 8-K accession 0000320193-25-000079, see filing"}]
    reg = scheduler.build_registry()
    objective = "What's AAPL's short interest?"
    action, dec = asyncio.run(
        programmatic_select_round(
            None,
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
    assert (action, dec.tool_name) == ("invoke", "get_short_interest")


def test_form_mentions_route_person_mention() -> None:
    """10-Q and 8-K mentions resolve by rule 5 without JEV."""

    class _BoomJev:
        async def select_tool(self, *a: object, **k: object) -> object:
            raise AssertionError("router must not call JEV")

    reg = scheduler.build_registry()
    for objective in ("Which 10-Q mentions Jensen Huang?", "Which 8-K mentions Elon Musk?"):
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
        assert (action, dec.tool_name) == ("invoke", "search_sec_filings_bounded")


def test_insider_verbs_stay_research_and_advice_stays_direct(monkeypatch: object) -> None:
    """Bare sell/hold tokens route research; should+verb and good-investment route direct."""
    import app.tools as _tools

    def _resolve(name: str) -> str | None:
        return {"TSLA": "TSLA", "NVDA": "NVDA"}.get(name)

    monkeypatch.setattr(  # type: ignore[union-attr]
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
