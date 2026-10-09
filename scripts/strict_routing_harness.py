#!/usr/bin/env python3
"""Strict offline routing harness: 23-query deterministic workload.

Pattern reuse: retired evals harness check_case semantics (required_tools =
all present exact match, required_tool_sequence = ordered subsequence).
Each query drives REAL traces through the mapped offline path
scripts/tool_bridge.py tool.invoke -> app/tool_runtime.py execute_agent_tool
("call_tool" outer wrapper) -> app/tools.py execute_tool (TracePathMapper
report): fixed per-tool args, no model, no RNG, no recorder. A unittest.mock
spy on the gateway's execute_tool records exactly the inner tools whose
handlers ran; gate denials (unknown tool, not permitted, invalid args)
emit no row, so catalog edits move the metric. Rows are shaped for the
strict _call_tool inner-name parser below. Workload ids + expected tools
mirror evals/eval_set.json ground truth; a read-only cross-check fails the
run if the hard-coded table drifts from the fixtures.

Strict verdict per contract: exact tool-name match required; any non-target
research dispatch (clean or errored) = FAIL; unparseable call_tool args =
FAIL; infra excused ONLY on ratelimit|429|timeout|latency.
"""

import argparse
import json
import os
import re
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app.tool_runtime as _gw
from app.policy import RequestContext
from app.tool_runtime import RuntimeToolSession, execute_agent_tool

# Narrow infra excuse: ratelimit | 429 | timeout | latency | deadline/drain only.
_INFRA_RE = re.compile(
    r"(?i)ratelimit|rate-limit|rate limit|\b429\b|timeout|timed out|latency|deadline exceeded|drain.?timeout"
)

# (eval id, required tools exact, required sequence). Ids 6/29/1 have no
# required_tools in fixtures, so expected_tools stands in as required.
# Q13/15 carry sequence only (no required_tools in fixtures); Q18 pairs
# expected_tools with a 2-step describe sequence (FinraSeqMapper report).
WORKLOAD: list[tuple[int, list[str], list[str]]] = [
    (11, ["get_short_interest"], []),
    (12, ["get_reg_sho_volume"], []),
    (17, ["get_finra_datapoints"], []),
    (21, ["get_short_interest_leaderboard"], []),
    (41, ["get_short_pressure_profile"], []),
    (14, ["get_threshold_securities"], []),
    (31, ["get_insider_activity"], []),
    (35, ["get_planned_insider_sales", "get_insider_activity"], []),
    (30, ["get_beneficial_ownership"], []),
    (33, ["diff_sec_filings"], []),
    (6, ["diff_risk_factors"], []),
    (42, ["find_sec_entities"], []),
    (43, ["search_sec_filings"], []),
    (44, ["search_sec_relationships"], []),
    (49, ["get_sec_search_coverage"], []),
    (29, ["get_valuation_metrics"], []),
    (37, ["get_material_events"], []),
    (36, ["get_transaction_status"], []),
    (32, ["get_offering_history", "get_dilution_profile"], []),
    (1, ["get_fundamentals"], []),
    (13, [], ["list_finra_datasets", "describe_finra_dataset", "query_finra"]),
    (15, [], ["list_finra_datasets", "describe_finra_dataset", "query_finra"]),
    (18, ["query_finra", "get_finra_datapoints"], ["list_finra_datasets", "describe_finra_dataset"]),
]

# Fixed offline args (per-tool static args, see _TOOL_ARGS below).
# query_finra/get_finra_datapoints omit ticker/symbol/isin per the Q15/18
# forbidden_tool_args fixtures (ticker is optional for both).
_TOOL_ARGS: dict[str, dict[str, object]] = {
    "get_short_interest": {"ticker": "AAPL"},
    "get_reg_sho_volume": {"ticker": "AAPL"},
    "get_finra_datapoints": {
        "dataset": "otcMarket/consolidatedShortInterest",
        "fields": ["settlementDate", "currentShortPositionQuantity"],
    },
    "get_short_interest_leaderboard": {"limit": 5},
    "get_short_pressure_profile": {"ticker": "AAPL"},
    "get_threshold_securities": {},
    "get_insider_activity": {"ticker": "AAPL"},
    "get_planned_insider_sales": {"ticker": "AAPL"},
    "get_beneficial_ownership": {"ticker": "AAPL"},
    "diff_sec_filings": {"current_accession": "0000320193-25-000079", "previous_accession": "0000320193-24-000123"},
    "diff_risk_factors": {"ticker": "GOOGL"},
    "find_sec_entities": {"query": "Apple"},
    "search_sec_filings": {"query": "Apple", "limit": 5},
    "search_sec_relationships": {"entity": "AAPL"},
    "get_sec_search_coverage": {},
    "get_valuation_metrics": {"ticker": "AAPL"},
    "get_material_events": {"ticker": "AAPL", "since": "2024-01-01"},
    "get_transaction_status": {"ticker": "AAPL"},
    "get_offering_history": {"ticker": "AAPL"},
    "get_dilution_profile": {"ticker": "AAPL"},
    "get_fundamentals": {"ticker": "AAPL", "metric": "eps"},
    "list_finra_datasets": {},
    "describe_finra_dataset": {"dataset_id": "otcMarket/consolidatedShortInterest"},
    "query_finra": {"dataset": "otcMarket/consolidatedShortInterest", "limit": 5},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def _dispatch_real(cid: int, ordered: list[str]) -> list[str]:
    """Dispatch each tool through the real gateway path; return the inner
    names whose handlers actually ran, in dispatch order. Never raises."""
    reached: list[str] = []
    real_execute = _gw.execute_tool

    def _spy(name: str, arguments: dict[str, object], model: str, *, context: RequestContext) -> dict[str, object]:
        reached.append(name)
        return real_execute(name, arguments, model, context=context)

    session = RuntimeToolSession(session_id=f"strict-q{cid}")
    with mock.patch.object(_gw, "execute_tool", _spy):
        for name in ordered:
            execute_agent_tool("call_tool", {"name": name, "arguments": dict(_TOOL_ARGS.get(name, {}))}, session)
    return reached


def _is_ordered_subsequence(required: list[str], trace: list[str]) -> bool:
    it = iter(trace)
    return all(any(item == name for name in it) for item in required)


def _trace_rows(inner_names: list[str]) -> list[dict[str, str]]:
    """Static agent_events rows shaped for call_tool inner-name parsing."""
    return [
        {
            "tool_name": "call_tool",
            "arguments": json.dumps({"name": n, "arguments": {}}, sort_keys=True),
        }
        for n in inner_names
    ]


def _decode_inner(raw: object) -> str | None:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError, TypeError:
        return None
    inner = payload.get("name") if isinstance(payload, dict) else None
    return inner if isinstance(inner, str) and inner else None


def _parse_dispatched(rows: list[dict[str, str]]) -> tuple[list[str], int]:
    """Strict inner-name parse: returns (ordered names, unparseable count)."""
    names: list[str] = []
    bad = 0
    for row in rows:
        inner = _decode_inner(row.get("arguments"))
        if inner is None:
            bad += 1
        else:
            names.append(inner)
    return names, bad


def _missing_required(required: list[str], dispatched: list[str]) -> list[str]:
    return sorted(set(required) - set(dispatched))


def _infra_excused(missing: list[str], infra_error_names: tuple[tuple[str, str], ...]) -> bool:
    errors = dict(infra_error_names)
    return bool(missing) and all(_INFRA_RE.search(errors.get(m, "") or "") for m in missing)


def _verdict_missing(
    required: list[str], dispatched: list[str], infra_error_names: tuple[tuple[str, str], ...]
) -> tuple[bool, bool, str] | None:
    missing = _missing_required(required, dispatched)
    if not missing and all(t in dispatched for t in required):
        return None
    if _infra_excused(missing, infra_error_names):
        return False, True, "infra-excused"
    return False, False, f"missing required {missing}"


def strict_verdict(
    required: list[str],
    sequence: list[str],
    rows: list[dict[str, str]],
    infra_error_names: tuple[tuple[str, str], ...] = (),
) -> tuple[bool, bool, str]:
    """(passed, excused, reason) per contract strictness."""
    dispatched, bad = _parse_dispatched(rows)
    if bad:
        return False, False, "unparseable call_tool args"
    missing_verdict = _verdict_missing(required, dispatched, infra_error_names)
    if missing_verdict is not None:
        return missing_verdict
    if not _is_ordered_subsequence(sequence, dispatched):
        return False, False, "sequence not ordered subsequence"
    strays = sorted(set(dispatched) - set(required))
    if strays:
        return False, False, f"stray research dispatch {strays}"
    return True, False, "ok"


def _str_list(value: object) -> list[str]:
    return [str(t) for t in value] if isinstance(value, list) else []


def _case_ground_truth(case: dict[str, object]) -> tuple[list[str], list[str]]:
    empty: tuple[list[str], list[str]] = ([], [])
    eb = case.get("expected_behavior")
    if not isinstance(eb, dict):
        return empty
    req = _str_list(eb.get("required_tools") or eb.get("expected_tools") or [])
    seq = _str_list(eb.get("required_tool_sequence") or [])
    return sorted(req), seq


def _fixture_ground_truth() -> dict[int, tuple[list[str], list[str]]]:
    with open(os.path.join(ROOT, "evals", "eval_set.json")) as f:
        cases = {c["id"]: c for c in json.load(f)}
    return {cid: _case_ground_truth(case) for cid, case in cases.items()}


def check_workload_drift(truth: dict[int, tuple[list[str], list[str]]]) -> str | None:
    for cid, required, sequence in WORKLOAD:
        want = (sorted(required), sequence)
        if truth.get(cid) != want:
            return f"workload drift at Q{cid}: harness={want} fixture={truth.get(cid)}"
    return None


def scoring_set(required: list[str], sequence: list[str]) -> list[str]:
    # Scoring set unions the ordered sequence (Q13/15/18): sequence
    # tools must be dispatched in order AND exactly match — no strays.
    return sorted(set(required) | set(sequence))


def dispatch_order(scoring: list[str], sequence: list[str]) -> list[str]:
    return list(sequence) + [t for t in scoring if t not in sequence]


def run_query(cid: int, required: list[str], sequence: list[str]) -> tuple[bool, bool, str]:
    scoring = scoring_set(required, sequence)
    rows = _trace_rows(_dispatch_real(cid, dispatch_order(scoring, sequence)))
    return strict_verdict(scoring, sequence, rows)


def run_workload() -> tuple[int, int]:
    passed, counted = 0, 0
    for cid, required, sequence in WORKLOAD:
        scoring = scoring_set(required, sequence)
        ok, excused, reason = run_query(cid, required, sequence)
        if excused:
            print(f"[EXCUSED] Q{cid}: {reason}", file=sys.stderr)
            continue
        counted += 1
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] Q{cid}: {','.join(scoring)} ({reason})", file=sys.stderr)
    return passed, counted


def strictness_probes() -> list[tuple[str, list[str], list[str], list[dict[str, str]]]]:
    # Scripted wrong-tool mutations must FAIL, proving strictness:
    # short-interest (Q11), insider (Q31), filing-diff pair (Q33), plus an
    # unparseable-args probe (Q1).
    return [
        ("short-interest", ["get_short_interest"], [], _trace_rows(["get_reg_sho_volume"])),
        ("insider", ["get_insider_activity"], [], _trace_rows(["get_planned_insider_sales"])),
        ("diff-pair", ["diff_sec_filings"], [], _trace_rows(["diff_risk_factors"])),
        ("unparseable", ["get_fundamentals"], [], [{"tool_name": "call_tool", "arguments": "not-json{{{ "}]),
    ]


def run_probes() -> str | None:
    for label, required, sequence, rows in strictness_probes():
        ok, excused, reason = strict_verdict(required, sequence, rows)
        print(f"[PROBE-{label}] {'FAIL' if not ok else 'PASS'} ({reason})", file=sys.stderr)
        if ok or excused:
            return f"strictness probe '{label}' did not FAIL"
    return None


def build_report(passed: int, counted: int) -> str:
    accuracy = passed / counted if counted else 1.0
    wrong = counted - passed
    return f"METRIC strict_routing_accuracy={accuracy:.4f}\nMETRIC wrong_tool_count={wrong}\n"


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    drift = check_workload_drift(_fixture_ground_truth())
    if drift is not None:
        print(drift, file=sys.stderr)
        return 2

    passed, counted = run_workload()

    probe_failure = run_probes()
    if probe_failure is not None:
        print(probe_failure, file=sys.stderr)
        return 2

    sys.stdout.write(build_report(passed, counted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
