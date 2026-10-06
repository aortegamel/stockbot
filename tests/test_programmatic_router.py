"""Programmatic router: 5 live questions route deterministically; JEV only on ambiguity."""

import asyncio

from app.research import scheduler
from app.research.programmatic_router import programmatic_route, programmatic_select_round


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
