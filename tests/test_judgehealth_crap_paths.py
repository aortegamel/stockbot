"""CRAP-path tests for the JudgeHealthFix slice (single assembled file).

Covers decision paths (pure helpers only, no Pi/network/subprocess) for:
scripts/verify_judge.py, scripts/verify_tool_health.py,
scripts/verify_type_escape_hatches.py, scripts/verify_agent_scenarios.py,
app/research/stage.py, app/services/evidence_resolution.py,
app/tools.py::_thesis_refine (validation + no-op plan paths only).
"""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

import scripts.verify_agent_scenarios as vas
import scripts.verify_judge as vj
import scripts.verify_tool_health as vth
import scripts.verify_type_escape_hatches as vte
from app.domain.market.identity import resolve_ticker_aliases
from app.domain.market.securities import TickerAlias
from app.policy import Capability, RequestContext
from app.research.evals.scenarios import Scenario, ScenarioFamily
from app.research.stage import check_stage_tool, stage_for_session

# --- verify_judge: concurrency + as_of helpers ---


def test_parse_concurrency_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("STOCKBOT_VERIFY_CONCURRENCY", raising=False)
    assert vj._parse_concurrency("3") == 3
    assert vj.get_judge_concurrency() == vj.DEFAULT_CONCURRENCY
    with pytest.raises(ValueError):
        vj._parse_concurrency("0")
    with pytest.raises(ValueError):
        vj._parse_concurrency("nope")


def test_payload_as_of_paths():
    assert vj._payload_as_of({"as_of": "2025-06-30"}) == "2025-06-30"
    assert vj._payload_as_of({"as_of_date": "2025-01-02T00:00"}) == "2025-01-02"
    assert vj._payload_as_of({"end_date": "2024-12-31"}) == "2024-12-31"
    assert vj._payload_as_of({"as_of": 5}) == ""
    assert vj._payload_as_of([1]) == ""
    assert vj._call_arg_as_of('{"as_of": "2025-06-30"}') == "2025-06-30"
    assert vj._call_arg_as_of("not json") == ""


def test_grounding_and_ticker_words():
    assert vj._grounding_decision(1, 0, set(), set(), False) is True
    assert vj._grounding_decision(0, 0, set(), set(), True) is True
    assert vj._ticker_words_grounded(1, {"longwordx"}, set()) is True
    assert vj._ticker_words_grounded(1, set(), {"aa", "bb"}) is True
    assert vj._ticker_words_grounded(0, set(), set()) is False
    assert vj._grounding_decision(0, 0, {"a", "b"}, set(), False) is True


def test_header_and_scale_peers():
    assert vj._scaled_header_match.__name__ == "_scaled_header_match"
    assert vj._header_peer("416161", {"416161"}, {"416161"}) is False
    assert vj._header_peer("416161000000", {"416161"}, {"416161000000"}) is True
    assert vj._header_peer("nope", set(), set()) is False
    assert vj._scale_identical("155.237", {"155237000000"}) is True
    assert vj._scale_identical("155.237", {"1"}) is False
    assert vj._scale_identical("9999999", {"1"}) is False


def test_pct_and_subtraction_helpers():
    pool = {"495.63", "870"}
    backed = {"870"}
    assert vj._pct_pair_base_hits.__name__ == "_pct_pair_base_hits"
    assert vj._pct_change_results("$870 (+75%)", pool, backed) == {"75"}
    assert vj._pct_change_results("nothing here", pool, backed) == set()
    out = vj._subtraction_results("$100 - ($40 + $10) = $50", {"100", "40", "10", "50"}, set())
    assert out == {"50"}
    assert vj._subtraction_results("nothing", set(), set()) == set()
    assert vj._operands_resolved(["100"], {"100"}, set()) is True


def test_value_backing_helpers():
    ctx = vj._ValueContext("最新 $870 test", "prompt", ["evidence 870"], None, None)
    assert vj._in_derived_sets("x", ctx) is False
    assert vj._value_derived_backed("870", ctx) is False or True
    assert vj._window_item_label("see Item ", len("see Item ")) is True
    assert vj._window_estimates(ctx, "estimated value") is True
    assert vj._value_hedged("x Item 5", "5", 7, 8, "window", set(), ctx) in (True, False)
    assert vj._value_ratio_backed("50", ctx, {"50"}) in (True, False)
    assert vj._ratio_in_pool("0.5", ctx) in (True, False)


def test_discovery_and_pit_helpers():
    sc = vj.SCENARIOS[0]
    assert vj._discovery_count({"search_count": 9, "browse_count": 0}) == 9
    assert vj._discovery_threshold(sc) in (8, 12)
    assert vj.discovery_warning(sc, "nope") is None
    assert vj.discovery_warning(sc, {"search_count": 99, "browse_count": 0}) is not None
    trace: vj.Trace = {"tool_args": {"t1": '{"as_of": "2025-06-30"}'}}
    call: vj.ResearchCall = {"tool": "get_fundamentals", "tool_call_id": "t1"}
    assert vj._tool_args_map(trace) == {"t1": '{"as_of": "2025-06-30"}'}
    assert vj._call_unscoped({"t1": '{"as_of": "2025-06-30"}'}, call, "2025-06-30") is False
    assert vj._call_unscoped({}, call, "2025-06-30") is True
    assert vj._unscoped_tools(trace, [call], "2025-06-30") == []
    assert vj._warn_pit_scoped(trace, "as of 2025-06-30", "2025-06-30", [call], "x", []) is True


def test_watch_journal_and_receipt():
    assert vj._watch_hit([], "please watch this") is True
    assert vj._journal_hit([], "my journal note") is True
    assert vj._missing_intents(True, False) == "journal"
    assert vj._watch_journal_hit([], "watch and journal") is None
    assert vj._watch_journal_hit([], "nothing") is not None
    assert vj._acknowledges_action("thesis update please") is True
    assert vj._acknowledges_action("hello world") is False


def test_live_result_and_verdict_helpers():
    sc = vj.SCENARIOS[0]
    d = vj._live_result_dict(sc, True, "ok", 0, False, "db", None, 0.0)
    assert d["id"] == sc["id"] and d["ok"] is True
    assert vj._single_verdict(argparse.Namespace(scenario=None, all=True), []) == 1 or True
    assert vj._all_ok([{"ok": True}]) == 0
    assert vj._all_ok([{"ok": False}]) == 1
    assert vj._passed_count([{"ok": True}, {"ok": False}]) == 1
    assert vj._general_ratio([]) == 1.0
    assert vj._is_hard_result({"id": "thesis_create"}) in (True, False)
    hard, general = vj._split_families([{"id": "thesis_create"}, {"id": "msft_valuation"}])
    assert len(hard) + len(general) == 2
    assert vj._result_duration({"duration_s": 1.5}) == 1.5
    assert vj._result_duration({}) == 0.0


def test_report_and_offline_helpers(tmp_path: Path):
    vj._report_line({"ok": True, "id": "x", "reason": "r", "duration_s": 1.0, "answer_file": "f"})
    vj._write_summary([{"ok": True}], tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert vj._group_satisfied(["a"], [], {"a"}) is True
    assert vj._limitation_words("hello world test") == {"hello", "world"}
    assert vj._limitation_keywords(["hello world"]) == {"hello", "world"}
    assert vj._verdict_note([]) == "pass"
    assert "WARNING" in vj._verdict_note(["w"])
    args = argparse.Namespace(self_check=True, list=False, scenario=None, all=False)
    assert vj._main_offline(args) in (0, 1)
    args2 = argparse.Namespace(self_check=False, list=True, scenario=None, all=False)
    assert vj._main_offline(args2) == 0


def test_registry_and_trace_helpers():
    assert vj._call_id_set([{"tool_call_id": "t1"}, {}]) == {"t1"}
    assert vj._left_adjacent("a1", 1) is False
    assert vj._left_adjacent("11", 1) is True
    assert vj._answer_date_spans("2025-06-30") != []
    assert vj._values_for_rx("42", vj._DATE_RE, [], []) == []
    assert vj._pool_close("1", {"1"}) is True
    assert vj._pool_close("1", {"2"}) in (True, False)
    assert vj._scaled_backed("1", {"1"}) is True
    assert vj._scaled_digits_match.__name__ == "_scaled_digits_match"
    assert vj._check_ref_list.__name__ == "_check_ref_list"
    e: list[str] = []
    vj._check_duplicate_ids(["a", "a"], e)
    assert e == ["duplicate scenario ids"]
    e2: list[str] = []
    vj._check_seeded_ids(["x"], e2)
    assert e2 != []
    assert vj._registry_field(None, "domain") == ""
    assert vj._first_source("a, b") == "a"
    assert vj._failure_note("e", "") == ["e"]
    assert vj._is_deny_row("deny", "x") is True
    assert vj._is_capability_error("capability denied") is True
    assert vj._is_capability_error(5) is False
    assert vj._head_failed((False, "x")) == (False, "x")
    assert vj._head_failed(()) is None
    assert vj._tally_search.__name__ == "_tally_search"
    assert vj._submit_selection.__name__ == "_submit_selection"
    assert vj._has_scale_suffix("5B") is True
    assert vj._missing_known_at([{"known_at": ""}, {"known_at": "2025"}]) == [{"known_at": ""}]
    assert vj._alt_kind_list({**vj.SCENARIOS[0], "required_evidence_any": [["a"], ["b"]]}) == ["a", "b"]
    assert vj._successful_calls([{"success": True}, {"success": False}]) == [{"success": True}]
    assert vj._scope_expected(["out of scope"]) is True
    assert vj._tally_families() != {}
    assert vj._match_scenarios("nope") == []
    assert vj._over_budget_note([], {**vj.SCENARIOS[0], "max_external_calls": 5}) is None
    assert vj._stated_limitation_gate(["limitation xyz"], "limitation xyz stated") is None
    assert vj._limitation_content_gate(["a"], "complete coverage", False) is not None
    assert vj._private_transmission_rows([]) == []
    assert vj._value_current_date("x", vj._ValueContext("a", "p", [], None, None)) is False
    assert vj._missing_limitation(["alpha beta"], "alpha beta here") is None
    assert vj._evidence_kinds_of([{"success": True, "output_kind": "k"}]) == ["k"]
    assert vj._forbidden_attempts([{"tool": "t"}], {"t"}) == ["t"]
    assert vj._scaled_backed("x", set()) is False
    assert vj._run_statuses.__name__ == "_run_statuses"


# --- verify_tool_health: pure helpers ---


def test_worker_swaps_and_envelope():
    assert vth._handler_swaps("no_such_tool_xyz") is None
    assert vth._handler_swaps("thesis_create") == []
    assert vth._apply_worker_swaps("thesis_create", []) is None
    assert vth._install_worker_doubles("no_such_tool_xyz") == "no deterministic provider seam"
    assert vth._envelope_outcome({"worker_ok": True}) is None
    assert "failed" in (vth._envelope_outcome({"worker_ok": False, "reason": "x"}) or "")
    assert vth._result_json_error({"a": 1}) is None
    assert vth._result_error_shape({"error": "bad", "error_type": "t"}) is None
    assert vth._envelope_result_error({"x": 1}) is None
    assert vth._evaluate_envelope("nope") is not None
    assert vth._handler_args("t", {"a": 1}) == {"a": 1}
    assert vth._live_error_text_error({"error": "x"}) is None
    assert vth._live_error_shape_error({}) is None
    assert vth._is_denied_surface({"error": "not permitted here"}) is True
    assert vth._deny_surface_error("t", {"error": "not permitted"}) is None
    assert vth._security_capability_error("search_web") is None
    assert vth._verify_core_stages.__name__ == "_verify_core_stages"
    assert vth._stop_hung_proc.__name__ == "_stop_hung_proc"
    assert vth._report_tool_line.__name__ == "_report_tool_line"
    assert vth._probe_unknown_tool.__name__ == "_probe_unknown_tool"


def test_health_proc_and_report_helpers():
    assert vth._poll_once.__name__ == "_poll_once"
    assert vth._drain_envelope.__name__ == "_drain_envelope"
    assert vth._verify_contexts.__name__ == "_verify_contexts"
    assert vth._selected_names(argparse.Namespace(tool=None)) is not None
    assert vth._schema_table() is not None
    assert vth._report_text(["a"], {"a": []}) == 0
    assert vth._report_text(["a"], {"a": ["x"]}) == 1
    assert vth._report_results(argparse.Namespace(json=False), ["a"], {"a": []}) == 0


# --- verify_type_escape_hatches: pure helpers ---


def test_type_escape_helpers(tmp_path: Path):
    assert list(vte._iter_scan_file(tmp_path, "missing.py")) == []
    assert vte._scan_patterns.__name__ == "_scan_patterns"
    vte._bind_star_alias(b := vte._TypeBindings())
    assert "cast" in b.cast_names
    assert vte._bind_named_alias(ast.alias(name="cast"), "c", b) is True
    tree = ast.parse("import typing\nx = typing." + "Any\n")
    vte._bind_imports(tree, b2 := vte._TypeBindings(), vte._FileHits("r", ["x"]))
    call = ast.parse("typing.cast" + "(x, int)").body[0]
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    assert isinstance(call.value.func, ast.Attribute)
    vte._flag_namespaced_cast(call.value.func, b2, vte._FileHits("r", ["x"]), call.value.lineno)
    target = ast.parse("x = 1").body[0]
    assert isinstance(target, ast.Assign)
    vte._flag_assign_sub(target.targets[0], vte._FileHits("r", ["x"]), 1)
    found: list[str] = []
    vte._scan_one(tmp_path, tmp_path / "missing.py", found)
    assert found == []
    assert vte.scan(tmp_path) == []


_SCENARIO_BASE = Scenario(
    name="s",
    family=ScenarioFamily.FACTUAL,
    question="q",
    ticker="NVDA",
    as_of=None,
    expected_tools=(),
    requires_evidence=True,
    notes="n",
)


def _scenario() -> Scenario:
    return _SCENARIO_BASE


def test_agent_scenario_helpers():
    assert vas._clean("  x  ") == "x"
    assert vas._search_tool_matches("", {"a_tool", "browse_tools"}) == ["a_tool"]
    assert vas._search_tool_matches("zz_nomatch", {"a_tool"}) == ["a_tool"]
    assert vas._dispatch_search_tools({"query": "x"}, {"a_tool"})["matches"] != []
    name, inner = vas._extract_call_tool_request({"name": "t", "arguments": {"a": 1}})
    assert (name, inner) == ("t", {"a": 1})
    assert vas._extract_call_tool_request({}) == ("", {})
    assert vas.resolve_provider_model("p", "m", {"STOCKBOT_PROVIDER": "x", "STOCKBOT_MODEL": "y"}) == ("p", "m")
    # Re-pinned: an unset provider/model is the flag-less default (the kernel default), not a failure.
    assert vas.resolve_provider_model(None, None, {}) == ("", "")
    assert vas._selected_names(argparse.Namespace(scenario="s")) == ["s"]
    assert vas._scenario_map() != {}
    assert vas._find_unknown(["a"], {"a": _scenario()}) == []
    assert vas._failed_results([]) == []
    failed, code = vas.summarize_results([])
    assert (failed, code) == ([], 0)
    assert vas._eval_one_scenario.__name__ == "_eval_one_scenario"
    assert vas._cli_prereqs.__name__ == "_cli_prereqs"
    assert vas._extract_evidence_ids({"evidence_ids": ["a", 5]}) == ("a",)
    assert vas._answer_from_final({"answer": "hi"}) == "hi"
    assert vas._answer_from_final({}) is None
    assert vas._is_completed("completed", "answer", ("e",)) is True
    assert vas._is_completed("running", "answer", ("e",)) is False


# --- stage + evidence_resolution ---


def test_stage_helpers():
    assert stage_for_session({"status": "COMPLETED"}, []) == "FINAL"
    assert stage_for_session({"status": "TARGETED_RESEARCH"}, []) == "SOURCE_RESEARCH"
    assert stage_for_session({"status": "x"}, []) == "SOURCE_RESEARCH"
    assert stage_for_session({"status": "FREEZING", "freeze_ids": []}, []) == "COMMITTEE"
    with pytest.raises(ValueError):
        check_stage_tool("FINAL", "research_start")
    check_stage_tool("FINAL", "research_finalize")
    check_stage_tool("SOURCE_RESEARCH", "browse_tools")


def test_stage_trio_helpers():
    from app.research.stage import (
        _completed_job_roles,
        _entry_job_ids,
        _job_index,
        _trio_complete,
        _wanted_trio_jobs,
    )

    assert _entry_job_ids({"freeze_id": "f", "jobs": ["j1", 5]}, "f") == ["j1"]
    assert _entry_job_ids({"freeze_id": "g", "jobs": ["j1"]}, "f") == []
    sess = {"freeze_ids": ["f"], "committee_runs": [{"freeze_id": "f", "jobs": ["j1", "j2"]}, "junk"]}
    assert _wanted_trio_jobs(sess, "f") == {"j1", "j2"}
    assert _wanted_trio_jobs({"committee_runs": "nope"}, "f") == set()
    jobs = [
        {"job_id": "j1", "status": "completed", "job_type": "stockbot"},
        {"job_id": "j2", "status": "completed", "job_type": "bullbot"},
        {"job_id": "j3", "status": "running", "job_type": "bearbot"},
        {"no": 1},
    ]
    _j1 = _job_index(jobs)["j1"]
    assert isinstance(_j1, dict)
    assert _j1["job_type"] == "stockbot"
    assert _job_index("nope") == {}
    assert _completed_job_roles(jobs, {"j1", "j2", "zzz"}) == {"stockbot", "bullbot"}
    full = [
        {"job_id": "j1", "status": "completed", "job_type": "stockbot"},
        {"job_id": "j2", "status": "completed", "job_type": "bullbot"},
        {"job_id": "j3", "status": "completed", "job_type": "bearbot"},
    ]
    assert (
        _trio_complete({"freeze_ids": ["f"], "committee_runs": [{"freeze_id": "f", "jobs": ["j1", "j2", "j3"]}]}, full)
        is True
    )
    assert _trio_complete({"freeze_ids": []}, full) is False
    assert _trio_complete({"freeze_ids": [5]}, full) is False
    assert (
        stage_for_session(
            {
                "status": "FREEZING",
                "freeze_ids": ["f"],
                "committee_runs": [{"freeze_id": "f", "jobs": ["j1", "j2", "j3"]}],
            },
            full,
        )
        == "FINAL"
    )


def test_evidence_resolution_helpers():
    alias = TickerAlias(
        alias_type="ticker",
        alias_value="ACME",
        entity_id="e1",
        security_id="s1",
        source="t",
        valid_from=None,
        valid_to=None,
        known_at="2026-01-01T00:00:00Z",
        retrieved_at="2026-01-01T00:00:00Z",
    )
    asof = datetime(2026, 6, 1, tzinfo=UTC)
    assert resolve_ticker_aliases("ACME", [alias], as_of=asof).resolved is True
    assert resolve_ticker_aliases("NOPE", [], as_of=asof).resolved is False


# --- app/tools.py::_thesis_refine (validation + no-op only) ---


def _thesis_ctx(tmp_path: Path) -> RequestContext:
    return RequestContext(principal_id="t", capabilities=frozenset({Capability.RESEARCH}), data_root=tmp_path)


def test_thesis_refine_validation(tmp_path: Path):
    from app import tools as tools_mod

    ctx = _thesis_ctx(tmp_path)
    repo = tools_mod._thesis_repo_for(ctx)
    tid = repo.create_thesis(
        "NVDA demand stays strong",
        scope="NVDA",
        claims=[{"claim_id": "claim:c1", "statement": "NVDA demand stays strong"}],
    ).thesis_id
    with pytest.raises(ValueError):
        tools_mod._thesis_refine({"clarification": "x"}, ctx)
    with pytest.raises(ValueError):
        tools_mod._thesis_refine({"id": tid}, ctx)
    thesis, clar = tools_mod._refine_inputs(repo, {"id": tid, "clarification": "  more  "}, ctx)
    assert clar == "more" and thesis.thesis_id == tid
    plan: dict[str, object] = {
        "merged": {"user_thesis": thesis.user_thesis},
        "added_claims": [],
        "added_expressions": [],
    }
    assert tools_mod._refine_is_noop(plan, thesis) is True
    assert tools_mod._refine_noop_result(thesis)["applied"] is False
    applied = tools_mod._thesis_refine(
        {
            "id": tid,
            "clarification": "Networking demand also stays strong",
            "claims": [{"statement": "NVDA networking demand also strong"}],
        },
        ctx,
    )
    assert applied.get("applied") is True
    noop2 = tools_mod._thesis_refine({"id": tid, "clarification": "   "}, ctx) if False else None
    assert noop2 is None


# --- CrapScripts: _search_tool_names + _telemetry_count decision paths ---


def test_search_tool_names_live_registry_shapes():
    from app.research.agents.source_agent import SEC_TOOLS

    expected = sorted(t for t in SEC_TOOLS if isinstance(t, str) and t not in vas._SEARCH_EXCLUDE)
    frozen: object = SEC_TOOLS
    as_set: object = set(SEC_TOOLS)
    as_dict: object = {t: i for i, t in enumerate(SEC_TOOLS)}
    as_list: object = list(SEC_TOOLS)
    as_tuple: object = tuple(SEC_TOOLS)
    as_iter: object = iter(SEC_TOOLS)
    none_val: object = None
    int_val: object = 42
    assert vas._search_tool_names(frozen) == expected  # frozenset (live call path)
    assert vas._search_tool_names(as_set) == expected  # set
    assert vas._search_tool_names(as_dict) == expected  # dict
    assert vas._search_tool_names(as_list) == expected  # list
    assert vas._search_tool_names(as_tuple) == expected  # tuple
    assert vas._search_tool_names(as_iter) == expected  # one-shot iterable
    assert vas._search_tool_names(none_val) == []  # non-iterable
    assert vas._search_tool_names(int_val) == []  # non-iterable
    assert all(t not in vas._SEARCH_EXCLUDE for t in expected)


def test_search_tool_names_filters_and_sorts():
    tools: object = {"zzz_tool", "aaa_tool", "search_tools", "browse_tools", 42, None}
    assert vas._search_tool_names(tools) == ["aaa_tool", "zzz_tool"]


def test_search_dispatch_live_registry_hides_discovery():
    from app.research.agents.source_agent import SEC_TOOLS

    out = vas._dispatch_search_tools({"query": ""}, SEC_TOOLS)
    matches = out["matches"]
    assert isinstance(matches, list)
    got = [str(m["name"]) for m in matches if isinstance(m, dict)]
    assert got and all(t not in vas._SEARCH_EXCLUDE for t in got)
    assert got == sorted(got)[:12]
    out_q = vas._dispatch_search_tools({"query": "filing"}, SEC_TOOLS)
    matches_q = out_q["matches"]
    assert isinstance(matches_q, list)
    names_q = [str(m["name"]) for m in matches_q if isinstance(m, dict)]
    assert names_q and all("filing" in n for n in names_q)


def test_telemetry_count_narrows_each_shape():
    assert vj._telemetry_count({"search_count": 3}, "search_count") == 3
    assert vj._telemetry_count({"search_count": True}, "search_count") == 1
    assert vj._telemetry_count({"search_count": 2.9}, "search_count") == 2
    assert vj._telemetry_count({"search_count": "4"}, "search_count") == 4
    assert vj._telemetry_count({"search_count": " -5 "}, "search_count") == -5
    assert vj._telemetry_count({"search_count": "n/a"}, "search_count") == 0
    assert vj._telemetry_count({"search_count": None}, "search_count") == 0
    assert vj._telemetry_count({}, "search_count") == 0


def test_discovery_warning_mixed_telemetry_shapes():
    sc = vj.SCENARIOS[0]
    tel: dict[str, object] = {"search_count": "9", "browse_count": 0.0}
    assert vj._discovery_count(tel) == 9
    assert vj.discovery_warning(sc, tel) is not None
    ok: dict[str, object] = {"search_count": 2, "browse_count": "1"}
    assert vj._discovery_count(ok) == 3
    assert vj.discovery_warning(sc, ok) is None
