"""Behavioral dispatch checks over the discovery catalog (no rank asserts)."""

import asyncio
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import override

import pytest

from app import tools as tools_mod
from app.decision_client import JevClient
from app.policy import Capability, RequestContext
from app.research import scheduler
from app.research.models import DecisionRecord, JSONValue, ToolDecision
from app.research.programmatic_router import programmatic_select_round
from app.tools import (
    TOOL_DISCOVERY_REGISTRY,
    TOOLS,
    _tool_function,
    execute_tool,
)

_CTX = RequestContext("test", frozenset({Capability.RESEARCH}))


def _browse(args: dict[str, object]) -> dict[str, object]:
    result = execute_tool("browse_tools", args, "test", context=_CTX)
    assert "error" not in result, result
    return result


def test_browse_root_lists_domains_only() -> None:
    result = _browse({})
    assert result["path"] == "/"
    domains = result["domains"]
    assert isinstance(domains, list)
    names = {d["name"] for d in domains if isinstance(d, dict)}
    assert names == {m.domain for m in TOOL_DISCOVERY_REGISTRY.values()}
    assert all(isinstance(d, dict) and d["path"] == f"/{d['name']}" for d in domains)
    assert "tools" not in result and "matches" not in result


def test_browse_domain_lists_families() -> None:
    result = _browse({"domain": "finra"})
    assert result["path"] == "/finra"
    fams_raw = result["families"]
    assert isinstance(fams_raw, list)
    fams = {f["name"]: f for f in fams_raw if isinstance(f, dict)}
    assert fams["short-interest"]["tool_count"] == 4
    assert fams["short-interest"]["path"] == "/finra/short-interest"


def test_browse_family_returns_four_tools_plus_contrast() -> None:
    result = _browse({"domain": "finra", "family": "short-interest"})
    assert result["path"] == "/finra/short-interest"
    assert result["count"] == 4
    tools_raw = result["tools"]
    assert isinstance(tools_raw, list)
    assert [t["name"] for t in tools_raw if isinstance(t, dict)] == [
        "get_finra_datapoints",
        "get_short_interest",
        "get_short_pressure_profile",
        "query_finra",
    ]
    contrast_raw = result["contrast_table"]
    assert isinstance(contrast_raw, list)
    assert len(contrast_raw) == 4
    by_tool = {r["tool"]: r for r in contrast_raw if isinstance(r, dict)}
    assert by_tool["get_short_interest"]["use_it_for"]
    assert "get_finra_datapoints" in str(by_tool["get_short_interest"]["do_not_use_it_for"])
    assert "query_finra" in str(by_tool["get_short_interest"]["do_not_use_it_for"])


def test_browse_name_returns_full_contract() -> None:
    result = _browse({"name": "get_short_interest"})
    assert result["summary"] == TOOL_DISCOVERY_REGISTRY["get_short_interest"].summary
    assert result["family"] == "short-interest"
    assert result["intent"] == "current_reported_short_position"
    canonical = next(t for t in TOOLS if _tool_function(t).get("name") == "get_short_interest")
    assert result["parameters"] == _tool_function(canonical)["parameters"]
    assert isinstance(result["required_arguments"], list) and "ticker" in result["required_arguments"]


def test_browse_rejects_bad_paths() -> None:
    bad_family = execute_tool("browse_tools", {"family": "short-interest"}, "test", context=_CTX)
    assert "error" in bad_family
    unknown_domain = execute_tool("browse_tools", {"domain": "nope"}, "test", context=_CTX)
    assert unknown_domain.get("error") == "unknown_domain"
    unknown_family = execute_tool("browse_tools", {"domain": "finra", "family": "nope"}, "test", context=_CTX)
    assert "error" in unknown_family
    both = execute_tool("browse_tools", {"domain": "finra", "name": "get_short_interest"}, "test", context=_CTX)
    assert "error" not in both and both["name"] == "get_short_interest"


def test_browse_name_overrides_conflicting_coordinates() -> None:
    conflicted = execute_tool(
        "browse_tools",
        {"domain": "sec", "family": "filing-catalog", "name": "get_short_interest"},
        "test",
        context=_CTX,
    )
    assert "error" not in conflicted
    assert conflicted["name"] == "get_short_interest"
    assert conflicted["domain"] == "finra"
    assert conflicted["family"] == "short-interest"


def test_invalid_arguments_return_repairable_shape_and_execute_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _canary(args: dict[str, object], model: str) -> dict[str, object]:
        calls.append(dict(args))
        return {"unexpected": True}

    for handlers in (
        tools_mod._MODEL_HANDLERS,
        tools_mod._FINRA_HANDLERS,
        tools_mod._ROBINHOOD_HANDLERS,
        tools_mod._THESIS_HANDLERS,
    ):
        if "get_short_interest" in handlers:
            monkeypatch.setitem(handlers, "get_short_interest", _canary)
    result = execute_tool("get_short_interest", {}, "test", context=_CTX)
    assert result["error_type"] == "invalid_tool_arguments"
    assert result["tool"] == "get_short_interest"
    required = result["required"]
    assert isinstance(required, list) and "ticker" in required
    assert result["parameters"] and result["optional"] is not None


def test_invalid_published_dates_fail_closed_with_tool_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid dates name the tool and block the provider. Valid dates reach the provider."""
    calls: list[dict[str, object]] = []

    def _canary(query: str, **kwargs: object) -> dict[str, object]:
        calls.append({"query": query, **kwargs})
        return {"ok": True}

    monkeypatch.setattr(tools_mod.exa_client, "search", _canary)
    bad = execute_tool(
        "search_web",
        {"query": "NVDA", "start_published_date": "not-a-date", "end_published_date": "2026-13-99"},
        "test",
        context=_CTX,
    )
    assert bad["error_type"] == "invalid_tool_arguments"
    assert bad["tool"] == "search_web"
    assert "start_published_date" in str(bad["error"])
    assert calls == []
    for dates in (
        {"start_published_date": "2026-02-30", "end_published_date": "2026-03-01"},
        {"start_published_date": "2026-01-01", "end_published_date": "2026-13-99"},
    ):
        invalid_calendar_date = execute_tool(
            "search_web",
            {"query": "NVDA", **dates},
            "test",
            context=_CTX,
        )
        assert invalid_calendar_date["error_type"] == "invalid_tool_arguments"
        assert invalid_calendar_date["tool"] == "search_web"
        assert calls == []
    good = execute_tool(
        "search_web",
        {"query": "NVDA", "start_published_date": "2026-01-01", "end_published_date": "2026-02-01"},
        "test",
        context=_CTX,
    )
    assert good == {"ok": True}
    assert calls and calls[0]["start_published_date"] == "2026-01-01"
    assert calls == [
        {
            "query": "NVDA",
            "category": None,
            "include_domains": None,
            "exclude_domains": None,
            "start_published_date": "2026-01-01",
            "end_published_date": "2026-02-01",
            "search_type": "auto",
            "limit": tools_mod.exa_client.EXA_DEFAULT_LIMIT,
        }
    ]


def test_unknown_tool_executes_nothing() -> None:
    result = execute_tool("no_such_tool", {}, "test", context=_CTX)
    assert "error" in result
    assert "no_such_tool" in str(result["error"])


# ---------------------------------------------------------------------------
# SEC-only routing denial: FINRA/Web/Market/Analyst tools denied at the
# kernel gates even when registered and reachable via browse_tools/call_tool.
# ---------------------------------------------------------------------------


def test_browse_lists_non_sec_but_kernel_denies_them() -> None:
    from app.research.agents.source_agent import is_sec_tool
    from app.research.models import Job
    from app.research.service import _dispatch_check_domain

    sec_job = Job(
        job_id="job:t",
        session_id="s",
        wave_id=1,
        parent_job_id=None,
        job_type="source_agent",
        owner="t",
        source_domain="SEC",
    )
    # Registered and discoverable via the catalog (browse does not filter);
    # broker tools live outside the discovery catalog — the kernel still denies.
    for name in ("query_finra", "get_short_interest", "search_web", "get_analyst_estimates"):
        found = _browse({"name": name})
        assert found["name"] == name, name
        assert is_sec_tool(name) is False, name
        try:
            _dispatch_check_domain(sec_job, "job:t", name)
        except ValueError as exc:
            assert "outside SEC domain" in str(exc), (name, exc)
        else:
            raise AssertionError(f"expected kernel denial for {name}")
    for name in ("get_market_snapshot", "get_option_chain"):
        assert is_sec_tool(name) is False, name
        try:
            _dispatch_check_domain(sec_job, "job:t", name)
        except ValueError as exc:
            assert "outside SEC domain" in str(exc), (name, exc)
        else:
            raise AssertionError(f"expected kernel denial for {name}")


def test_runner_guarded_call_denies_non_sec() -> None:
    import inspect as _inspect

    from app.research.runner import _LiveRun

    src = _inspect.getsource(_LiveRun._guarded_tool_call)
    assert "POLICY_DENIED" in src
    assert "is_sec_tool" in src


def test_call_tool_inner_name_still_gated() -> None:
    # call_tool is the SEC envelope, but a non-SEC inner name is denied by
    # is_sec_tool at the kernel boundary (runner + service agree).
    from app.research.agents.source_agent import is_sec_tool

    assert is_sec_tool("call_tool") is True
    for inner in ("query_finra", "search_web", "get_market_snapshot", "get_analyst_estimates"):
        assert is_sec_tool(inner) is False, inner


def test_filter_allowed_tools_never_bypassed_via_browse() -> None:
    from app.security.action_policy import filter_allowed_tools

    sec_only: dict[str, object] = {"allowed": ["SEC"], "denied": [], "mode": "allowlist"}
    names = [
        "search_sec_filings",
        "search_web",
        "query_finra",
        "get_market_snapshot",
        "get_analyst_estimates",
        "list_sec_filings",
    ]
    assert filter_allowed_tools(names, sec_only) == ["search_sec_filings", "list_sec_filings"]
    assert filter_allowed_tools(names, None) == names


# ---------------------------------------------------------------------------
# Recorded programmatic-router survivors: deterministic picks persist unit
# probabilities, session/node identity, and variant rewrites. Short-family
# single-token signals route to their own member. Entry cues need their exact
# wording. JEV fallback keeps full context and remaps coverage variants.
# ---------------------------------------------------------------------------


def _capturing_kernel() -> tuple[scheduler._Kernel, list[tuple[object, ...]]]:
    seen: list[tuple[object, ...]] = []

    class _Capture(scheduler._Kernel):
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
            seen.append(
                (
                    session_id,
                    decision_type,
                    dict(candidates),
                    dict(probabilities),
                    selected,
                    node_id,
                    job_id,
                    confidence,
                )
            )
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

    return _Capture(), seen


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


def _round(
    kernel: scheduler._Kernel,
    objective: JSONValue,
    question: str,
    registry: list[dict[str, JSONValue]],
    attempts: list[dict[str, JSONValue]] | None = None,
    evidence: list[JSONValue] | None = None,
    jev: JevClient | None = None,
) -> tuple[str, ToolDecision]:
    node = SimpleNamespace(node_id="n1", session_id="s1", question=question, why_it_matters="w")
    action, decision = asyncio.run(
        programmatic_select_round(
            jev if jev is not None else _BoomJev(),
            kernel,
            "s1",
            "n1",
            {"session_id": "s1", "objective": objective},
            node,
            registry,
            evidence if evidence is not None else [],
            attempts if attempts is not None else [],
        )
    )
    return action, decision


def test_record_pick_persists_unit_probabilities_and_identity() -> None:
    from app.research.programmatic_router import _record_pick

    kernel, seen = _capturing_kernel()
    action, decision = _record_pick(kernel, "s1", "n1", "get_short_interest", "short-family")
    assert action == "invoke"
    assert decision.tool_name == "get_short_interest"
    assert decision.tool_names == ("get_short_interest",)
    assert decision.probabilities == {"get_short_interest": 1.0}
    assert seen == [
        (
            "s1",
            "tool_selection",
            {"get_short_interest": 1.0},
            {"get_short_interest": 1.0},
            "get_short_interest",
            "n1",
            None,
            None,
        )
    ]


def test_clock_and_deterministic_picks_record_session_and_node() -> None:
    registry = scheduler.build_registry()
    kernel, seen = _capturing_kernel()
    action, decision = _round(kernel, "", "what time is it", registry)
    assert (action, decision.tool_name) == ("invoke", "get_current_time")
    kernel2, seen2 = _capturing_kernel()
    action2, decision2 = _round(kernel2, "", "What's AAPL's short interest?", registry)
    assert (action2, decision2.tool_name) == ("invoke", "get_short_interest")
    for recorded in (*seen, *seen2):
        assert recorded[0] == "s1" and recorded[5] == "n1"
        assert recorded[2] == recorded[3] == {recorded[4]: 1.0}


def test_node_question_clock_wins_over_revenue_objective() -> None:
    registry = scheduler.build_registry()
    kernel, seen = _capturing_kernel()
    action, decision = _round(kernel, "NVDA revenue filing", "what time is it", registry)
    assert (action, decision.tool_name) == ("invoke", "get_current_time")
    assert seen and seen[0][0] == "s1" and seen[0][5] == "n1"


def test_node_question_carries_short_ask_when_objective_empty() -> None:
    registry = scheduler.build_registry()
    kernel, _ = _capturing_kernel()
    action, decision = _round(kernel, "", "What's AAPL's short interest?", registry)
    assert (action, decision.tool_name) == ("invoke", "get_short_interest")
    assert decision.probabilities == {"get_short_interest": 1.0}


def test_objective_clock_answers_with_blank_node_question() -> None:
    registry = scheduler.build_registry()
    kernel, seen = _capturing_kernel()
    action, decision = _round(kernel, "what time is it", "", registry)
    assert (action, decision.tool_name) == ("invoke", "get_current_time")
    assert seen and seen[0][4] == "get_current_time"


@pytest.mark.parametrize(
    ("ask", "tool"),
    [
        ("venue short interest AAPL", "get_reg_sho_volume"),
        ("daily short interest AAPL", "get_reg_sho_volume"),
        ("volume short interest AAPL", "get_reg_sho_volume"),
        ("sho short interest AAPL", "get_reg_sho_volume"),
        ("reg short interest AAPL", "get_reg_sho_volume"),
        ("outstanding short pressure", "get_short_pressure_profile"),
        ("relative short pressure", "get_short_pressure_profile"),
        ("ratio short pressure", "get_short_pressure_profile"),
        ("pressure short profile", "get_short_pressure_profile"),
        ("trend short interest query", "query_finra"),
        ("distribution short interest", "query_finra"),
        ("briefing short interest", "query_finra"),
        ("change short interest", "query_finra"),
        ("coverage short interest", "query_finra"),
        ("screen short interest", "get_short_interest_leaderboard"),
        ("leaderboard short interest", "get_short_interest_leaderboard"),
        ("most shorted", "get_short_interest_leaderboard"),
        ("Reg SHO daily short volume for AAPL", "get_reg_sho_volume"),
        ("short pressure outstanding shares ratio", "get_short_pressure_profile"),
        ("short interest trend coverage", "query_finra"),
        ("most shorted stocks screen", "get_short_interest_leaderboard"),
        ("What's AAPL's short interest?", "get_short_interest"),
    ],
)
def test_short_family_single_token_signals_route_each_member(ask: str, tool: str) -> None:
    from app.research.programmatic_router import _ranked, _registry_names, _router_pick, _signals

    registry = scheduler.build_registry()
    names = _registry_names(registry)
    doubled = f"{ask} {ask}"
    norm, toks = _signals(doubled)
    ranked = _ranked(norm, toks, names)
    rule_pick = _router_pick(doubled, norm, toks, names, ranked, [], [])
    assert rule_pick is not None and rule_pick[0] == tool
    kernel, _ = _capturing_kernel()
    action, decision = _round(kernel, ask, ask, registry)
    assert (action, decision.tool_name) == ("invoke", tool)
    assert decision.probabilities == {tool: 1.0}


@pytest.mark.parametrize(
    ("ask", "tool"),
    [
        ("Reg data for AAPL short interest", "get_reg_sho_volume"),
    ],
)
def test_short_family_bare_reg_signal_routes_reg_sho(ask: str, tool: str) -> None:
    kernel, seen = _capturing_kernel()
    action, decision = _round(kernel, ask, ask, scheduler.build_registry())
    assert (action, decision.tool_name) == ("invoke", tool)
    assert seen and seen[0][4] == tool


def test_reg_sho_substring_collision_routes_explicit_sho_volume() -> None:
    """A cross-word 'reg sho' collision routes explicit SHO volume, not JEV."""
    kernel, seen = _capturing_kernel()
    action, decision = _round(
        kernel, "Greg shows AAPL short interest", "Greg shows AAPL short interest", scheduler.build_registry()
    )
    assert (action, decision.tool_name) == ("invoke", "get_reg_sho_volume")
    assert seen and seen[0][4] == "get_reg_sho_volume"


@pytest.mark.parametrize(
    ("ask", "tool"),
    [
        ("reg short interest for AAPL", "get_reg_sho_volume"),
        ("AAPL short interest versus shares outstanding", "get_short_pressure_profile"),
        ("AAPL short positioning relative to float", "get_short_pressure_profile"),
        ("AAPL short ratio", "get_short_pressure_profile"),
    ],
)
def test_short_family_distinguishes_position_volume_and_float_context(ask: str, tool: str) -> None:
    kernel, seen = _capturing_kernel()
    action, decision = _round(kernel, ask, ask, scheduler.build_registry())
    assert (action, decision.tool_name) == ("invoke", tool)
    assert seen and seen[0][4] == tool


def test_advice_cue_needs_an_advice_verb() -> None:
    from app.research.programmatic_router import programmatic_route

    assert programmatic_route("should AAPL halt trading", ticker="AAPL") == "research_required"
    assert programmatic_route("should I buy NVDA", ticker="NVDA") == "reasoning_required"


def test_research_cue_single_signals_route_research() -> None:
    from app.research.programmatic_router import programmatic_route

    assert programmatic_route("revenue outlook") == "research_required"
    assert programmatic_route("funds are betting against it") == "research_required"


def test_advice_requires_prompt_subject_but_other_routes_use_caller_ticker() -> None:
    from app.research.programmatic_router import programmatic_route

    assert programmatic_route("should I buy it") == "clarify_subject"
    assert programmatic_route("should I buy it", ticker="GME") == "reasoning_required"
    assert programmatic_route("how is it doing", ticker="GME") == "research_required"
    assert programmatic_route("how is it doing") is None


class _BoundedPick(JevClient):
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


class _FullPick(JevClient):
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
            tool_name="search_sec_filings",
            tool_names=("search_sec_filings",),
            probabilities={"search_sec_filings": 0.9},
            confidence=0.9,
        )


def test_jev_fallback_needs_coverage_to_rewrite_variants() -> None:
    registry = scheduler.build_registry()
    kernel, _ = _capturing_kernel()
    action, decision = _round(
        kernel, "zqxj vbnm all companies qqqq", "zqxj vbnm all companies qqqq", registry, jev=_BoundedPick()
    )
    assert (action, decision.tool_name) == ("invoke", "search_sec_filings")
    assert decision.tool_names == ("search_sec_filings",)
    assert decision.probabilities == {"search_sec_filings": 0.9}
    assert decision.confidence == 0.9


def test_jev_pair_pick_on_plain_ask_passes_through() -> None:
    registry = scheduler.build_registry()
    kernel, _ = _capturing_kernel()
    action, decision = _round(kernel, "halp money stuff", "halp money stuff", registry, jev=_BoundedPick())
    assert (action, decision.tool_name) == ("invoke", "search_sec_filings_bounded")
    assert decision.probabilities == {"search_sec_filings_bounded": 0.9}
    assert decision.confidence == 0.9


def test_jev_fallback_records_selected_tool_for_node() -> None:
    kernel, seen = _capturing_kernel()
    action, decision = _round(
        kernel, "halp money stuff", "halp money stuff", scheduler.build_registry(), jev=_BoundedPick()
    )
    assert (action, decision.tool_name) == ("invoke", "search_sec_filings_bounded")
    assert seen == [
        (
            "s1",
            "tool_selection",
            {"search_sec_filings_bounded": 0.9},
            {"search_sec_filings_bounded": 0.9},
            "search_sec_filings_bounded",
            "n1",
            None,
            0.9,
        )
    ]


def test_jev_exhaustive_pick_survives_without_a_registered_variant() -> None:
    registry = [entry for entry in scheduler.build_registry() if entry.get("name") == "get_short_interest"]
    kernel, _ = _capturing_kernel()
    action, decision = _round(kernel, "who filed them qqqq", "who filed them qqqq", registry, jev=_FullPick())
    assert (action, decision.tool_name) == ("invoke", "search_sec_filings")
    assert decision.tool_names == ("search_sec_filings",)


def test_jev_bounded_pick_rewrites_to_registered_exhaustive() -> None:
    registry = [entry for entry in scheduler.build_registry() if entry.get("name") == "search_sec_filings"]
    kernel, _ = _capturing_kernel()
    action, decision = _round(kernel, "halp money stuff", "halp money stuff", registry, jev=_BoundedPick())
    assert (action, decision.tool_name) == ("invoke", "search_sec_filings")
    assert decision.probabilities == {"search_sec_filings": 0.9}


def test_carry_revenue_uses_listed_packet() -> None:
    carry_only = {"get_sec_document", "get_sec_filing", "list_sec_documents"}
    attempts: list[dict[str, JSONValue]] = [
        {
            "tool": "list_sec_filings",
            "arguments": {"identifier": "NVDA"},
            "error": None,
            "result": {
                "subject": "NVDA",
                "filings": [{"accession_no": "0001045810-26-000075", "form": "10-Q", "filed_at": "2026-08-01"}],
            },
        }
    ]
    kernel, _ = _capturing_kernel()
    action, decision = _round(
        kernel,
        "NVDA revenue 10q read the document",
        "NVDA revenue 10q read the document",
        scheduler.build_registry(),
        attempts=attempts,
    )
    assert action == "invoke" and decision.tool_name in carry_only | {"list_sec_filings"}
    kernel2, _ = _capturing_kernel()
    action2, decision2 = _round(
        kernel2,
        "NVDA revenue 10q read the document",
        "NVDA revenue 10q read the document",
        [{"name": name} for name in sorted(carry_only)],
        attempts=attempts,
    )
    assert (action2, decision2.tool_name) == ("invoke", "get_sec_document")
    kernel3, seen = _capturing_kernel()
    action3, decision3 = _round(
        kernel3,
        "NVDA revenue 10q read the document",
        "NVDA revenue 10q read the document",
        [{"name": "get_sec_filing"}],
        attempts=attempts,
    )
    assert (action3, decision3.tool_name) == ("invoke", "get_sec_filing")
    assert seen and seen[0][4] == "get_sec_filing"


def test_context_evidence_counts_for_carry() -> None:
    carry_only = {"get_sec_document", "get_sec_filing", "list_sec_documents"}
    evidence: list[JSONValue] = [
        {
            "id": "ev:1",
            "content": "filing",
            "accession_no": "0001045810-26-000075",
            "form": "10-Q",
            "filed_at": "2026-08-01",
            "subject": "1045810",
            "subject_cik": "1045810",
            "filer_cik": "1045810",
        }
    ]
    kernel, _ = _capturing_kernel()
    action, decision = _round(
        kernel,
        "CIK 1045810 read the filing document section",
        "read the filing document section",
        [{"name": name} for name in sorted(carry_only)],
        evidence=evidence,
    )
    assert (action, decision.tool_name) == ("invoke", "get_sec_document")


def test_scorer_margin_pick_routes_transaction_status() -> None:
    ask = "checking merger acquisition or tender offer filing context for a ticker proxy vote meeting governance board compensation records"
    registry = scheduler.build_registry()
    kernel, _ = _capturing_kernel()
    action, decision = _round(kernel, ask, ask, registry)
    assert (action, decision.tool_name) == ("invoke", "get_transaction_status")


class _ContextCapture(JevClient):
    def __init__(self) -> None:
        self.seen: dict[str, object] = {}

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
        self.seen["objective"] = objective
        self.seen["node"] = node
        self.seen["registry"] = registry
        self.seen["evidence"] = evidence
        self.seen["attempts"] = attempts
        self.seen["session_id"] = session_id
        return ToolDecision(action="reason", probabilities={}, confidence=None)


def test_select_round_forwards_full_context_to_jev() -> None:
    registry = scheduler.build_registry()
    attempts: list[dict[str, JSONValue]] = [{"tool": "list_sec_filings", "error": None}]
    evidence: list[JSONValue] = [{"id": "ev:1"}]
    jev = _ContextCapture()
    kernel, _ = _capturing_kernel()
    action, decision = _round(
        kernel, "halp money stuff", "halp money stuff", registry, attempts=attempts, evidence=evidence, jev=jev
    )
    assert action == "reason" and decision.tool_name is None
    assert jev.seen["session_id"] == "s1"
    assert jev.seen["registry"] == registry
    seen_attempts = jev.seen["attempts"]
    seen_evidence = jev.seen["evidence"]
    assert isinstance(seen_attempts, Sequence)
    assert isinstance(seen_evidence, Sequence)
    assert list(seen_attempts) == attempts
    assert list(seen_evidence) == evidence
    assert "halp money stuff" in str(jev.seen["objective"])
    assert jev.seen["node"] is not None
