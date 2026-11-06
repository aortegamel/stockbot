"""Behavioral dispatch checks over the discovery catalog (no rank asserts)."""

import pytest

from app import tools as tools_mod
from app.policy import Capability, RequestContext
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


# ---------------------------------------------------------------------------
# Explicit thesis objectives (judge thesis_create / thesis_contradict /
# watch_vs_journal) select their lifecycle tool before JEV; read-only thesis
# questions keep JEV.
# ---------------------------------------------------------------------------
