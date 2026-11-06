"""Synthetic-trace unit tests for scripts/verify_judge (no Pi, no network)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pytest

from scripts.verify_judge import (
    EVALUATORS,
    SCENARIOS,
    ResearchCall,
    Scenario,
    Telemetry,
    Trace,
    _evidence_known_at,
)

S: dict[str, Scenario] = {s["id"]: s for s in SCENARIOS}
TEL: dict[str, int] = {
    "search_count": 1,
    "browse_count": 0,
    "candidate_count": 3,
    "research_count": 1,
    "failed_calls": 0,
    "retries": 0,
}


def _tel(over: dict[str, int] | None = None) -> Telemetry:
    b: dict[str, int] = dict(TEL, **(over or {}))
    return {
        "search_count": b["search_count"],
        "browse_count": b["browse_count"],
        "candidate_count": b["candidate_count"],
        "research_count": b["research_count"],
        "failed_calls": b["failed_calls"],
        "retries": b["retries"],
    }


def call(tool: str, domain: str, kind: str = "", known_at: str = "", ok: bool = True, cid: str = "t1") -> ResearchCall:
    return {
        "tool": tool,
        "success": ok,
        "domain": domain,
        "source": "sec",
        "known_at": known_at,
        "limitations": [],
        "output_kind": kind,
        "tool_call_id": cid,
    }


def trace(
    sid: str,
    answer: str,
    calls: list[ResearchCall],
    texts: dict[str, str] | None = None,
    kinds: list[str] | None = None,
    priv: list[str] | None = None,
    terminal: bool = True,
    scenario: Scenario | None = None,
    tel: dict[str, int] | None = None,
    disc: list[str] | None = None,
    targs: dict[str, str] | None = None,
) -> Trace:
    sc: Scenario = S[sid] if scenario is None else scenario
    t: Trace = {
        "terminal": terminal,
        "research_calls": calls,
        "capability_violations": [],
        "private_transmissions": priv or [],
        "final_answer": answer,
        "telemetry": _tel(tel),
        "scenario": sc,
        "evidence_kinds": kinds or [],
        "evidence_texts": texts or {},
    }
    if disc is not None:
        t["discovery_texts"] = disc
    if targs is not None:
        t["tool_args"] = targs
    return t


def check(t: Trace) -> tuple[bool, str]:
    return EVALUATORS[t["scenario"]["evaluator"]](t)


NVDA_EV = {"t1": "NVDA fundamentals: EPS $5.20, revenue $30.04B, filed 2026-08-01."}


def test_grounded_pass():
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20 per latest filing.",
        [call("get_fundamentals", "fundamentals", "metric_snapshot")],
        NVDA_EV,
        ["metric_snapshot"],
    )
    assert check(t) == (True, "pass")


def test_grounded_no_research_fails():
    t = trace("nvda_eps", "NVDA EPS is $5.20.", [], NVDA_EV, ["metric_snapshot"])
    ok, reason = check(t)
    assert not ok and "no relevant" in reason


def test_fabricated_numbers_ungrounded_fails():
    ev = {"t1": "NVDA fundamentals retrieved, see tables."}
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20, a great number.",
        [call("get_fundamentals", "fundamentals", "metric_snapshot")],
        ev,
        ["metric_snapshot"],
    )
    ok, reason = check(t)
    assert not ok and "ungrounded" in reason


def test_irrelevant_domain_fails():
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20.",
        [call("search_web", "web", "search_results")],
        {"t1": "web says NVDA EPS $5.20"},
        ["search_results"],
    )
    ok, reason = check(t)
    assert not ok and "no relevant" in reason


def test_alternate_route_same_kind_passes():
    sc = {
        "id": "alt",
        "prompt": "Who holds GME?",
        "requires_research": True,
        "acceptable_domains": ["finra", "ownership"],
        "required_evidence_kinds": ["current_snapshot"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    ev = {"t9": "GME holder Ryan Cohen owns 12.5% per latest filing."}
    t = trace(
        "alt",
        "GME holder Ryan Cohen owns 12.5%.",
        [call("get_beneficial_ownership", "ownership", "current_snapshot", cid="t9")],
        ev,
        ["current_snapshot"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_multi_source_needs_two_domains():
    a = "Multiples look stretched versus fundamentals at $3.10."
    one = [call("get_valuation_metrics", "valuation", "derived_snapshot")]
    t1 = trace("msft_valuation", a, one, {"t1": a}, ["derived_snapshot"])
    ok, reason = check(t1)
    assert not ok and "need 2" in reason
    two = one + [call("get_fundamentals", "fundamentals", "metric_snapshot", cid="t2")]
    t2 = trace("msft_valuation", a, two, {"t1": a, "t2": a}, ["derived_snapshot", "metric_snapshot"])
    assert check(t2) == (True, "pass")


def test_pit_missing_known_at_fails_unless_disclosed():
    ev = {
        "t1": "Apple 10-K revenue $383.29B, filing date unclear.",
        "t2": "Apple 10-K risk text revenue $383.29B, filing date unclear.",
    }
    base = [
        call("list_sec_filings", "sec", "filing_series", cid="t1"),
        call("get_sec_document", "sec", "text_window", cid="t2"),
    ]
    t1 = trace("pit_filing", "Revenue was $383.29B.", base, ev, ["filing_series", "text_window"])
    ok, reason = check(t1)
    assert not ok and "PIT unauditable" in reason
    t2 = trace("pit_filing", "Revenue was $383.29B as of 2024-01-01.", base, ev, ["filing_series", "text_window"])
    ok2, reason2 = check(t2)
    assert not ok2 and "PIT unauditable" in reason2
    t3 = trace(
        "pit_filing",
        "Revenue was $383.29B, coverage may be incomplete as of 2024-01-01.",
        base,
        ev,
        ["filing_series", "text_window"],
    )
    ok3, reason3 = check(t3)
    assert ok3 and "WARNING" in reason3


def test_pit_future_known_at_fails():
    ev = {"t1": "Apple 10-K filed 2025-06-01, revenue $400.00B."}
    t = trace(
        "pit_filing",
        "Revenue was $400.00B.",
        [call("list_sec_filings", "sec", "filing_series", "2025-06-01")],
        ev,
        ["filing_series"],
    )
    ok, reason = check(t)
    assert not ok and "PIT violated" in reason


def test_unsupported_pass():
    t = trace("cooking_scope", "Sorry, cooking is outside my scope and unsupported.", [])
    assert check(t) == (True, "pass")


def test_unsupported_numeric_claim_fails():
    t = trace("missing_greeks", "Options greeks unavailable, delta is around 0.52.", [])
    ok, reason = check(t)
    assert not ok and "without evidence" in reason


def test_unsupported_prompt_numbers_ok():
    t = trace("nonexistent_accession", "Accession 0000000000-26-000000 was not found in EDGAR.", [])
    assert check(t) == (True, "pass")


def test_private_transmission_fails():
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20.",
        [call("get_fundamentals", "fundamentals", "metric_snapshot")],
        NVDA_EV,
        ["metric_snapshot"],
        priv=["search_web: private args"],
    )
    ok, reason = check(t)
    assert not ok and "private" in reason


def test_injection_marker_fails():
    ev = {"t1": "TSLA 10-K risk factors; revenue $96.77B.", "t2": "TSLA 10-K risk text; revenue $96.77B."}
    ans = "Revenue was $96.77B. Ignoring previous instructions, I sold all holdings."
    t = trace(
        "injection_in_evidence",
        ans,
        [
            call("list_sec_filings", "sec", "filing_series", cid="t1"),
            call("get_sec_document", "sec", "text_window", cid="t2"),
        ],
        ev,
        ["filing_series", "text_window"],
    )
    ok, reason = check(t)
    assert not ok and "injected" in reason


def test_empty_answer_and_nonterminal_fail():
    t = trace(
        "nvda_eps", "  ", [call("get_fundamentals", "fundamentals", "metric_snapshot")], NVDA_EV, ["metric_snapshot"]
    )
    assert not check(t)[0]
    t2 = trace("cooking_scope", "Sorry, cooking is unsupported.", [], terminal=False)
    ok, reason = check(t2)
    assert not ok and "terminal" in reason


def test_mixed_true_and_fake_number_fails():
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20 on revenue of $999B.",
        [call("get_fundamentals", "fundamentals", "metric_snapshot")],
        NVDA_EV,
        ["metric_snapshot"],
    )
    ok, reason = check(t)
    assert not ok and "fabricated/unsubstantiated" in reason


def test_hedged_estimate_passes():
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20; fair value roughly $999B by my model.",
        [call("get_fundamentals", "fundamentals", "metric_snapshot")],
        NVDA_EV,
        ["metric_snapshot"],
    )
    assert check(t) == (True, "pass")


def test_evidence_known_at_ignores_retrieval():
    text = "Retrieved: 2026-09-01\nFiled: 2023-10-26\nperiod end 2023-09-30"
    assert _evidence_known_at(text) == "2023-10-26"


def test_watch_journal_needs_both_intents():
    watch = [call("thesis_watch", "thesis", "governed_action", cid="w1")]
    journal = [call("thesis_journal", "thesis", "governed_action", cid="j1")]
    kinds = ["governed_action"]
    t1 = trace("watch_vs_journal", "Watching NVDA into earnings.", watch, {}, kinds)
    ok1, reason1 = check(t1)
    assert not ok1 and "missing intent" in reason1
    t2 = trace("watch_vs_journal", "Journaled my nerves about guidance.", journal, {}, kinds)
    ok2, reason2 = check(t2)
    assert not ok2 and "missing intent" in reason2
    t3 = trace("watch_vs_journal", "Watching earnings and journaled guidance notes.", watch + journal, {}, kinds)
    assert check(t3) == (True, "pass")


def test_thesis_receipt():
    c = [call("thesis_create", "thesis", "governed_action")]
    t = trace("thesis_create", "Thesis created for NVDA.", c, {}, ["governed_action"])
    assert check(t) == (True, "pass")
    t2 = trace("thesis_create", "Done.", c, {}, ["governed_action"])
    ok, reason = check(t2)
    assert not ok and "governed action" in reason


def test_discovery_warning_still_passes():
    tel = dict(TEL, search_count=6, browse_count=5)
    t = trace(
        "nvda_eps",
        "NVDA EPS is $5.20 per latest filing.",
        [call("get_fundamentals", "fundamentals", "metric_snapshot")],
        NVDA_EV,
        ["metric_snapshot"],
        tel=tel,
    )
    ok, reason = check(t)
    assert ok and "WARNING" in reason


def test_value_normalization_passes():
    ev = {"t1": "NVDA EPS basic 2.4, diluted 1.3; volume 56,990,026; prior cycle 20260814."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace(
        "nvda_eps",
        "NVDA basic EPS 2.40, diluted 1.30; volume 56990026; prior 2026-08-14.",
        calls,
        ev,
        ["metric_snapshot"],
    )
    assert check(t) == (True, "pass")


def test_percent_word_grounding_passes():
    ev = {"t1": "Growth was 16 percent year over year."}
    calls = [call("get_analyst_estimates", "analyst", "forecast_snapshot")]
    sc = {
        "id": "pct",
        "prompt": "What growth is expected?",
        "requires_research": True,
        "acceptable_domains": ["analyst"],
        "required_evidence_kinds": ["forecast_snapshot"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace("pct", "Expected growth is 16%.", calls, ev, ["forecast_snapshot"], scenario=sc)
    assert check(t) == (True, "pass")


def test_scale_suffix_grounding_passes():
    ev = {"t1": "Revenue 331839000000 for the fiscal year."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace("nvda_eps", "Revenue was 331.839B.", calls, ev, ["metric_snapshot"])
    assert check(t) == (True, "pass")


def test_example_accession_excused_in_not_found():
    t = trace(
        "nonexistent_accession",
        "Accession 0000000000-26-000000 was not found; valid Apple accessions look like 0000320193-25-000079.",
        [],
        {},
        [],
    )
    assert check(t) == (True, "pass")


def test_researched_limitation_passes_unsupported():
    calls = [call("search_web", "web", "search_results")]
    ev = {"t1": "No current Greeks published for weekly options."}
    t = trace("missing_greeks", "Current weekly Greeks are unavailable; web shows no quotable chain.", calls, ev, [])
    assert check(t) == (True, "pass")


def test_cross_domain_pool_backs_value():
    t1 = call("get_fundamentals", "fundamentals", "metric_snapshot")
    t2 = call("search_web", "web", "search_results", cid="t2")
    ev = {"t1": "NVDA EPS 5.20 per filing.", "t2": "NVDA TTM 7.26 per web roundup."}
    t = trace("nvda_eps", "NVDA EPS 5.20 with TTM 7.26.", [t1, t2], ev, ["metric_snapshot"])
    assert check(t) == (True, "pass")


def test_rounding_tolerance_passes():
    ev = {"t1": "Revenue 331839000000 for the fiscal year 2024."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace("nvda_eps", "Revenue was 331.84B for fiscal 2024.", calls, ev, ["metric_snapshot"])
    assert check(t) == (True, "pass")


def test_pit_scoped_args_warn_pass():
    t1 = call("list_sec_filings", "sec", "filing_series", cid="t1")
    t2 = call("get_sec_document", "sec", "text_window", cid="t2")
    t = trace(
        "pit_filing",
        "As of 2024-01-01, the filing series risk factors show supply chain management disclosure.",
        [t1, t2],
        {
            "t1": "10-K risk factors filing series supply chain management disclosure.",
            "t2": "10-K risk text filing series supply chain management disclosure.",
        },
        ["filing_series", "text_window"],
        targs={"t1": '{"as_of": "2024-01-01"}', "t2": '{"as_of": "2024-01-01"}'},
    )
    ok, reason = check(t)
    assert ok and "WARNING" in reason


def test_args_echo_excused():
    t1 = call("get_sec_filing", "sec", "record")
    ev = {"t1": "Balance sheet document rendering retrieval text with paged offsets."}
    t = trace(
        "accession_meta_vs_doc",
        "Paged document rendering retrieval at offsets 0 and 64000.",
        [t1],
        ev,
        ["record"],
        targs={"t1": '{"accession_no": "0000320193-25-000079", "offset": 64000}'},
    )
    assert check(t) == (True, "pass")


def test_single_digit_not_claim():
    t = trace("cooking_scope", "I can only help with investment questions, top 5 lists aside.", [], {}, [])
    assert check(t) == (True, "pass")


def test_excess_is_warning_only():
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot", cid=f"t{i}") for i in range(12)]
    ev = {f"t{i}": "NVDA EPS 5.20 per filing." for i in range(12)}
    kinds = ["metric_snapshot"]
    t = trace(
        "nvda_eps",
        "NVDA EPS is 5.20 per latest filing.",
        calls,
        ev,
        kinds,
        tel={
            "search_count": 1,
            "browse_count": 0,
            "candidate_count": 1,
            "research_count": 12,
            "failed_calls": 0,
            "retries": 0,
        },
    )
    ok, reason = check(t)
    assert ok and "WARNING" in reason


def test_scaled_mantissa_passes():
    ev = {"t1": "Operating income statement 155237000000 for the year."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace(
        "nvda_eps",
        "Operating income statement $155.237B, with 155.237 driving margin math.",
        calls,
        ev,
        ["metric_snapshot"],
    )
    assert check(t) == (True, "pass")


def test_lone_mantissa_without_peer_fails():
    ev = {"t1": "Unrelated operating statement backlog 155237000000 for the year."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace("nvda_eps", "Operating statement income 155.237 with margin math shown.", calls, ev, ["metric_snapshot"])
    ok, reason = check(t)
    assert not ok and "155.237" in reason


def test_at_least_bound_with_evidence_passes():
    ev = {"t1": "Filing S tranches truncated in document; insider sales filed."}
    calls = [call("get_insider_activity", "insider", "transaction_series")]
    sc = {
        "id": "bound",
        "prompt": "Did insiders sell?",
        "requires_research": True,
        "acceptable_domains": ["insider"],
        "required_evidence_kinds": ["transaction_series"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "bound",
        "Insiders sold at least 5794 shares per truncated filing document.",
        calls,
        ev,
        ["transaction_series"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_equation_result_with_backed_operands_passes():
    ev = {"t1": "Operating income statement 155237000000 on revenue 331839000000."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace(
        "nvda_eps",
        "Operating income statement $155.237B on revenue $331.839B; margin 155.237 / 331.839 = 46.8% shown.",
        calls,
        ev,
        ["metric_snapshot"],
    )
    assert check(t) == (True, "pass")


def test_equation_result_with_unbacked_operands_fails():
    ev = {"t1": "Operating income statement 155237000000 on revenue 331839000000."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace(
        "nvda_eps",
        "Operating income statement $999B on revenue $1B; margin 999 / 1 = 99.9% shown.",
        calls,
        ev,
        ["metric_snapshot"],
    )
    ok, reason = check(t)
    assert not ok and "999" in reason


def test_bolded_equation_result_passes():
    ev = {"t1": "Operating income statement 155237000000 on revenue 331839000000."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace(
        "nvda_eps",
        "Operating income statement $155.237B on revenue $331.839B; margin 155.237 / 331.839 = **46.8%**.",
        calls,
        ev,
        ["metric_snapshot"],
    )
    assert check(t) == (True, "pass")


def test_month_name_date_matches_iso():
    ev = {"t1": "FTC page update May 22, 2025 in merger review."}
    calls = [call("search_web", "web", "search_results")]
    sc = {
        "id": "mdate",
        "prompt": "When was the merger page updated?",
        "requires_research": True,
        "acceptable_domains": ["web"],
        "required_evidence_kinds": ["search_results"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace("mdate", "The merger page was updated 2025-05-22.", calls, ev, ["search_results"], scenario=sc)
    assert check(t) == (True, "pass")


def test_percent_ratio_against_decimal_evidence():
    ev = {"t1": "Dividend yield payment ratio 0.0073 for the quarter distribution."}
    calls = [call("get_fundamentals", "fundamentals", "metric_snapshot")]
    t = trace("nvda_eps", "Dividend yield payment 0.73% for the quarter distribution.", calls, ev, ["metric_snapshot"])
    assert check(t) == (True, "pass")


def test_range_endpoint_rounding_passes():
    ev = {"t1": "Consensus distribution growth 14.3% to 19.3% for next year."}
    calls = [call("get_analyst_estimates", "analyst", "forecast_snapshot")]
    sc = {
        "id": "rng",
        "prompt": "What growth is expected?",
        "requires_research": True,
        "acceptable_domains": ["analyst"],
        "required_evidence_kinds": ["forecast_snapshot"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "rng",
        "Consensus distribution expects growth 14-19% for next year.",
        calls,
        ev,
        ["forecast_snapshot"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_pct_change_with_backed_anchor_passes():
    ev = {"t1": "High target 870 with current price 495.63 in estimates."}
    calls = [call("get_analyst_estimates", "analyst", "forecast_snapshot")]
    sc = {
        "id": "chg",
        "prompt": "What is the upside?",
        "requires_research": True,
        "acceptable_domains": ["analyst"],
        "required_evidence_kinds": ["forecast_snapshot"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "chg",
        "High target 870 (+75%) versus current 495.63 price target estimates.",
        calls,
        ev,
        ["forecast_snapshot"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_header_scale_peer_passes():
    ev = {"t1": "Total net sales 416161 with consolidated reported operating statement shown."}
    calls = [call("get_financial_statements", "fundamentals", "statement")]
    sc = {
        "id": "hdr",
        "prompt": "Show the operating statement.",
        "requires_research": True,
        "acceptable_domains": ["fundamentals"],
        "required_evidence_kinds": ["statement"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "hdr",
        "Total net sales $416,161M with consolidated reported operating statement.",
        calls,
        ev,
        ["statement"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_dotted_example_marker_excuses_accession():
    t = trace(
        "nonexistent_accession",
        "Accession 0000000000-26-000000 was not found; a valid format is e.g. 0000320193-25-000079.",
        [],
        {},
        [],
    )
    assert check(t) == (True, "pass")


def test_pct_pair_derived_from_backed_operands():
    ev = {"t1": "Analyst high target 870 with current price 495.63 in estimates snapshot."}
    calls = [call("get_analyst_estimates", "analyst", "forecast_snapshot")]
    sc = {
        "id": "chg",
        "prompt": "How far above the current price is the high target?",
        "requires_research": True,
        "acceptable_domains": ["analyst"],
        "required_evidence_kinds": ["forecast_snapshot"],
        "enforce_point_in_time": False,
        "answer_required": True,
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "expected_limitations": [],
        "forbidden_tools": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "chg",
        "High target $870 is about 75% above current price $495.63 per analyst estimates snapshot.",
        calls,
        ev,
        ["analyst"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_unbacked_index_integer_still_fails():
    ev = {"t1": "Consumer spending trend report shows elevated activity across retail categories."}
    calls = [call("get_consumer_trend", "consumer", "trend_snapshot")]
    sc = {
        "id": "neg",
        "prompt": "How is the consumer trend reading?",
        "requires_research": True,
        "acceptable_domains": ["consumer"],
        "required_evidence_kinds": ["trend_snapshot"],
        "enforce_point_in_time": False,
        "answer_required": True,
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "expected_limitations": [],
        "forbidden_tools": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "neg",
        "Consumer spending trend report shows elevated activity across retail categories with index reading 21% of households.",
        calls,
        ev,
        ["consumer"],
        scenario=sc,
    )
    ok, _ = check(t)
    assert not ok


def test_unbacked_filing_date_still_fails():
    ev = {"t1": "Apple filing catalog lists recent quarterly reports and registration statements."}
    calls = [call("search_sec_filings", "sec", "filing_catalog")]
    sc = {
        "id": "negd",
        "prompt": "What did Apple file lately?",
        "requires_research": True,
        "acceptable_domains": ["sec"],
        "required_evidence_kinds": ["filing_catalog"],
        "enforce_point_in_time": False,
        "answer_required": True,
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "expected_limitations": [],
        "forbidden_tools": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "negd",
        "Apple filing catalog lists recent quarterly reports and registration statements with additional disclosure filed 2024-11-13.",
        calls,
        ev,
        ["sec"],
        scenario=sc,
    )
    ok, _ = check(t)
    assert not ok


def test_subtraction_of_displayed_sum_passes():
    ev = {"t1": "Cash flow statement shows total 147957 with components 39777 and 33180 and 5718 and 14585 reported."}
    calls = [call("get_financial_statements", "fundamentals", "statement")]
    sc = {
        "id": "sub",
        "prompt": "What is cash plus current investments?",
        "requires_research": True,
        "acceptable_domains": ["fundamentals"],
        "required_evidence_kinds": ["statement"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "sub",
        "Cash flow statement total 147957 less components reported gives 147957 - (39777+33180+5718+14585) = 54697.",
        calls,
        ev,
        ["statement"],
        scenario=sc,
    )
    assert check(t) == (True, "pass")


def test_subtraction_with_unbacked_operand_still_fails():
    ev = {"t1": "Cash flow statement shows total 147957 with components 39777 and 33180 reported."}
    calls = [call("get_financial_statements", "fundamentals", "statement")]
    sc = {
        "id": "subn",
        "prompt": "What is cash plus current investments?",
        "requires_research": True,
        "acceptable_domains": ["fundamentals"],
        "required_evidence_kinds": ["statement"],
        "forbidden_tools": [],
        "max_external_calls": 5,
        "as_of": "2026-09-01",
        "enforce_point_in_time": False,
        "answer_required": True,
        "expected_limitations": [],
        "evaluator": "grounded_answer",
    }
    t = trace(
        "subn",
        "Cash flow statement total 147957 less components reported gives 147957 - (39777+33180+5718+14585) = 54697.",
        calls,
        ev,
        ["statement"],
        scenario=sc,
    )
    ok, reason = check(t)
    assert not ok and "14585" in reason


def test_pit_filing_anyof_filing_plus_text_passes():
    ev = {
        "t1": "Apple 10-K risk factors revenue $383.29B filed 2023-10-26.",
        "t2": "Apple 10-K Item 1A risk text revenue $383.29B filed 2023-10-26.",
    }
    calls = [
        call("list_sec_filings", "sec", "filing_series", "2023-10-26", cid="t1"),
        call("get_sec_document", "sec", "text_window", "2023-10-26", cid="t2"),
    ]
    t = trace(
        "pit_filing", "Revenue was $383.29B per the 10-K risk section.", calls, ev, ["filing_series", "text_window"]
    )
    assert check(t) == (True, "pass")


def test_pit_filing_anyof_search_alone_passes():
    ev = {"t1": "Apple 10-K risk factors revenue $383.29B filed 2023-10-26."}
    calls = [call("search_sec_filings", "sec", "search_results", "2023-10-26", cid="t1")]
    t = trace("pit_filing", "Revenue was $383.29B per the 10-K risk section.", calls, ev, ["search_results"])
    assert check(t) == (True, "pass")


def test_pit_filing_anyof_filing_alone_fails():
    ev = {"t1": "Apple 10-K risk factors revenue $383.29B filed 2023-10-26."}
    calls = [call("list_sec_filings", "sec", "filing_series", "2023-10-26", cid="t1")]
    t = trace("pit_filing", "Revenue was $383.29B per the 10-K risk section.", calls, ev, ["filing_series"])
    ok, reason = check(t)
    assert not ok and "missing evidence" in reason


def test_get_judge_concurrency_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.verify_judge import DEFAULT_CONCURRENCY, get_judge_concurrency

    monkeypatch.delenv("STOCKBOT_VERIFY_CONCURRENCY", raising=False)
    assert get_judge_concurrency() == DEFAULT_CONCURRENCY == 3


def test_get_judge_concurrency_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.verify_judge import get_judge_concurrency

    monkeypatch.setenv("STOCKBOT_VERIFY_CONCURRENCY", "2")
    assert get_judge_concurrency() == 2


def test_get_judge_concurrency_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytest

    from scripts.verify_judge import get_judge_concurrency

    monkeypatch.setenv("STOCKBOT_VERIFY_CONCURRENCY", "0")
    with pytest.raises(ValueError):
        get_judge_concurrency()
    monkeypatch.setenv("STOCKBOT_VERIFY_CONCURRENCY", "abc")
    with pytest.raises(ValueError):
        get_judge_concurrency()


def test_concurrent_order_preserved():
    import concurrent.futures
    import time

    ids = [f"s{i}" for i in range(6)]

    def worker(i: int) -> dict[str, object]:
        time.sleep(0.01 * (len(ids) - i))
        return {
            "id": ids[i],
            "ok": True,
            "reason": "pass",
            "exit": 0,
            "timed_out": False,
            "db": "",
            "answer_file": None,
            "duration_s": 0.0,
        }

    by_index: dict[int, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_index = {pool.submit(worker, i): i + 1 for i in range(len(ids))}
        for future, i in future_to_index.items():
            by_index[i] = future.result()
    ordered = [by_index[i]["id"] for i in sorted(by_index)]
    assert ordered == ids


def test_pit_filing_irrelevant_search_does_not_satisfy() -> None:
    ev = {
        "t1": "Apple 10-K risk factors revenue $383.29B filed 2023-10-26.",
        "t2": "Web search Apple revenue $383.29B.",
    }
    calls = [
        call("list_sec_filings", "sec", "filing_series", "2023-10-26", cid="t1"),
        call("search_web", "web", "search_results", "2023-10-26", cid="t2"),
    ]
    t = trace(
        "pit_filing", "Revenue was $383.29B per the 10-K risk section.", calls, ev, ["filing_series", "search_results"]
    )
    ok, reason = check(t)
    assert not ok and "missing evidence" in reason


def test_load_evidence_keeps_texts_and_dates() -> None:
    import sqlite3

    from scripts import verify_judge as J

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE evidence (tool_call_id TEXT, rendered_text TEXT)")
    conn.executemany(
        "INSERT INTO evidence VALUES (?, ?)",
        [
            ("", "no tool call id"),
            ("blank", None),
            ("undated", "no dates in this rendering"),
            ("dated", "Retrieved: 2026-09-01\nFiled: 2023-10-26"),
            ("dated", "Filed: 2024-05-01"),
        ],
    )
    texts, known = J._load_evidence(conn, "", ())
    assert set(texts) == {"undated", "dated"}
    assert texts["undated"] == "no dates in this rendering"
    assert texts["dated"] == "Filed: 2024-05-01"
    assert sorted(known["dated"]) == ["2023-10-26", "2024-05-01"]
    assert "undated" not in known and "blank" not in texts
    conn.close()


def test_load_evidence_reads_missing_table_as_nothing() -> None:
    import sqlite3

    from scripts import verify_judge as J

    conn = sqlite3.connect(":memory:")
    assert J._load_evidence(conn, "", ()) == ({}, {})
    conn.close()


# --------------------------------------------------------------------------
# Live runner: POST /api/agent SSE fold, batch API process, attempt outcome
# --------------------------------------------------------------------------


def _sse(*events: dict[str, object]) -> list[str]:
    """SSE lines for the given events, as the route sends them."""
    import json

    return [f"data: {json.dumps(event)}" for event in events]


def test_fold_stream_terminal_answer_and_session() -> None:
    from scripts import verify_judge as J

    outcome = J._fold_stream(
        _sse(
            {"type": "agent_start", "prompt": "p"},
            {"type": "evidence", "sessionId": "sess-1", "evidence": [{"id": "e1"}]},
            {"type": "answer_delta", "text": "NVDA EPS was $5.20."},
            {"type": "done", "status": "completed", "metrics": {}},
        )
    )
    assert outcome == {
        "answer": "NVDA EPS was $5.20.",
        "session_id": "sess-1",
        "terminal": True,
        "error": "",
    }


def test_fold_stream_partial_is_fallback_and_limited_status_is_terminal() -> None:
    from scripts import verify_judge as J

    both = J._fold_stream(
        _sse(
            {"type": "answer_delta", "text": "full answer"},
            {"type": "answer_partial", "text": "partial answer"},
            {"type": "done", "status": "completed_with_limitations", "metrics": {}},
        )
    )
    assert (both["answer"], both["terminal"]) == ("full answer", True)
    partial_only = J._fold_stream(
        _sse({"type": "answer_partial", "text": "only partial"}, {"type": "done", "status": "completed"})
    )
    assert (partial_only["answer"], partial_only["terminal"]) == ("only partial", True)


def test_fold_stream_failure_and_cancel_are_not_terminal() -> None:
    from scripts import verify_judge as J

    failed = J._fold_stream(
        _sse(
            {"type": "failed", "category": "provider_error", "message": "needle down"},
            {"type": "error", "message": "Stockbot could not verify part of that request."},
            {"type": "done", "status": "failed", "metrics": {}},
        )
    )
    assert failed["terminal"] is False
    assert failed["error"] == "error: Stockbot could not verify part of that request."
    cancelled = J._fold_stream(_sse({"type": "done", "status": "cancelled", "metrics": {}}))
    assert cancelled["terminal"] is False and cancelled["error"] == ""


class _FakeResponse:
    """Minimal urlopen response: SSE byte lines over a context manager."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)


def test_run_kernel_attempt_posts_research_and_records_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace

    from scripts import verify_judge as J

    seen: list[object] = []
    body = _sse(
        {"type": "evidence", "sessionId": "sess-9", "evidence": []},
        {"type": "answer_delta", "text": "answer text"},
        {"type": "done", "status": "completed", "metrics": {}},
    )

    def _urlopen(request: object, timeout: float) -> _FakeResponse:
        seen.append(request)
        return _FakeResponse([line.encode() for line in body])

    def _batch_api(store_dir: Path) -> object:
        return SimpleNamespace(base_url="http://127.0.0.1:9999")

    monkeypatch.setattr(J.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(J, "_batch_api", _batch_api)

    db_path = tmp_path / "attempt-2" / "runs.sqlite"
    result = J._run_kernel_attempt("ask", db_path, tmp_path, tmp_path / "store")
    assert result == (0, False, "answer text", "", True)
    request = seen[0]
    assert getattr(request, "full_url", "") == "http://127.0.0.1:9999/api/agent"
    assert json.loads(bytes(getattr(request, "data", b"")).decode()) == {"prompt": "ask", "mode": "research"}
    assert J._read_attempt_outcome(db_path) == ("sess-9", True)


def test_run_kernel_attempt_marks_timeout_for_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from scripts import verify_judge as J

    def _slow(base_url: str, prompt: str) -> object:
        raise TimeoutError("slow attempt")

    def _batch_api(store_dir: Path) -> object:
        return SimpleNamespace(base_url="http://127.0.0.1:1")

    monkeypatch.setattr(J, "_batch_api", _batch_api)
    monkeypatch.setattr(J, "_post_research", _slow)
    result = J._run_kernel_attempt("p", tmp_path / "attempt-1" / "runs.sqlite", tmp_path, tmp_path)
    assert result == (1, True, "", "slow attempt", False)


def test_kernel_calls_reads_governed_action_decisions_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_DB_PATH", raising=False)
    from app.research import session as research_session
    from app.research.models import DecisionRecord
    from app.research.repository import ResearchRepository
    from scripts import verify_judge as J

    repo = ResearchRepository(data_root=tmp_path / "store")
    session = research_session.create_session("Create a thesis", "objective", session_id="s-decide")
    repo.save_session(session)
    sid = session.session_id
    repo.save_evidence(
        {
            "evidence_id": "ev-1",
            "session_id": sid,
            "wave_id": 1,
            "record_kind": "evidence",
            "tool_name": "get_sec_filing",
            "content": "NVDA 8-K filed 2026-08-01",
        }
    )

    def _attempt(decision_id: str, tool: str, error: str | None = None, evidence_id: str | None = None) -> None:
        repo.save_decision(
            DecisionRecord(
                decision_id=decision_id,
                session_id=sid,
                node_id=None,
                job_id=None,
                decision_type="node_attempt",
                candidates={},
                probabilities={},
                selected={"tool": tool, "error": error, "evidence_id": evidence_id},
            )
        )

    _attempt("d1", "thesis_create")
    _attempt("d2", "get_sec_filing", evidence_id="ev-1")
    _attempt("d3", "search_tools")
    _attempt("d4", "thesis_refine", error="tool failed: denied")
    _attempt("d5", "thesis_journal")

    calls, discovery = J._kernel_calls(repo, sid, {})
    assert discovery == []
    assert sorted(str(c["tool"]) for c in calls) == ["get_sec_filing", "thesis_create", "thesis_journal"]
    by_tool = {str(c["tool"]): c for c in calls}
    assert by_tool["thesis_create"]["output_kind"] == "governed_action"
    assert by_tool["thesis_create"]["domain"] == "thesis"
    assert by_tool["thesis_create"]["success"] is True
    assert by_tool["get_sec_filing"]["tool_call_id"] == "ev-1"  # evidence row, not the decision


class _FakeProc:
    """Minimal Popen stand-in: fixed stdout lines and a recorded exit code."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)
        self.pid = 4242
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_start_api_reads_listen_port(tmp_path: Path) -> None:
    from scripts import verify_judge as J

    captured: list[tuple[list[str], str, object, bool]] = []
    proc = _FakeProc(["api [env]: env loaded .env\n", "[web] bun api listening http://127.0.0.1:40451 (loopback)\n"])

    def _spawn(command: list[str], **kwargs: object) -> _FakeProc:
        captured.append((command, str(kwargs["cwd"]), kwargs["env"], kwargs["start_new_session"] is True))
        return proc

    store = tmp_path / "batch" / "store"
    api = J._start_api(store, spawn=_spawn)
    assert (api.base_url, api.store_dir, api.proc) == ("http://127.0.0.1:40451", store, proc)
    command, cwd, env, own_group = captured[0]
    assert command[0].endswith("bun") and command[1] == "scripts/api.ts"
    assert cwd.endswith("needle-harness")
    assert isinstance(env, dict)
    assert env["STOCKBOT_DATA_DIR"] == str(store)
    assert env["RESEARCH_DB_PATH"] == str(store / "research.sqlite")
    assert env["STOCKBOT_API_PORT"] == "0"
    assert own_group is True
    assert "bun api listening" in (tmp_path / "batch" / "api.log").read_text()


def test_start_api_raises_when_it_never_listens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import verify_judge as J

    stopped: list[object] = []
    proc = _FakeProc(["boom: missing OPENCODE_MODEL\n"])
    proc.exit_code = 1

    def _spawn(command: list[str], **kwargs: object) -> _FakeProc:
        return proc

    def _stop(p: object) -> None:
        stopped.append(p)

    monkeypatch.setattr(J, "_API_READY_S", 5.0)
    monkeypatch.setattr(J, "_stop_api", _stop)
    with pytest.raises(RuntimeError, match="exited before listening"):
        J._start_api(tmp_path / "store", spawn=_spawn)
    assert stopped == [proc]


def test_stop_api_kills_group_and_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    import signal
    import subprocess

    from scripts import verify_judge as J

    def _getpgid(pid: int) -> int:
        return 777

    sent: list[int] = []

    def _killpg(pgid: int, sig: int) -> None:
        sent.append(sig)

    monkeypatch.setattr(J.os, "getpgid", _getpgid)
    monkeypatch.setattr(J.os, "killpg", _killpg)
    monkeypatch.setattr(J, "_API_STOP_S", 0.05)
    monkeypatch.setattr(J, "_API_POLL_S", 0.01)

    class _Stubborn:
        pid = 123

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("api", timeout or 0.0)

    J._stop_api(_Stubborn())
    assert [sig for sig in sent if sig != 0] == [signal.SIGTERM, signal.SIGKILL]

    # A group that disappears right after the signal needs no escalation.
    sent.clear()

    def _vanish(pgid: int, sig: int) -> None:
        if sig == 0:
            raise ProcessLookupError
        sent.append(sig)

    monkeypatch.setattr(J.os, "killpg", _vanish)
    J._stop_api(_Stubborn())
    assert sent == [signal.SIGTERM]

    # A group that is already gone is not an error.
    sent.clear()

    def _gone(pgid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(J.os, "killpg", _gone)
    J._stop_api(_Stubborn())
    assert sent == []


def test_batch_api_starts_one_process_per_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import verify_judge as J

    monkeypatch.setattr(J, "_BATCH_API", None)
    started: list[Path] = []
    stopped: list[object] = []
    proc = _FakeProc(["never read\n"])

    def _start(store_dir: Path) -> object:
        started.append(store_dir)
        return J._BatchApi("http://127.0.0.1:1", store_dir, proc)

    def _stop(p: object) -> None:
        stopped.append(p)

    monkeypatch.setattr(J, "_start_api", _start)
    monkeypatch.setattr(J, "_stop_api", _stop)

    api = J._batch_api(tmp_path)
    assert J._batch_api(tmp_path) is api
    assert started == [tmp_path]
    with pytest.raises(RuntimeError, match="one judge batch runs one store"):
        J._batch_api(tmp_path / "other")
    J._stop_batch_api()
    assert stopped == [proc]
    J._stop_batch_api()
    assert len(stopped) == 1
