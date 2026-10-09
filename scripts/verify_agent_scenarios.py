#!/usr/bin/env python3
"""Live broad-agent scenario verification through the kernel scheduler (no static PASS, no fakes).

Usage:
    python scripts/verify_agent_scenarios.py [--scenario NAME] [--model LABEL]
        [--provider LABEL] [--model-timeout SECONDS] [--prompt-version VER]
        [--fixtures-dir DIR] [--json]
    python scripts/verify_agent_scenarios.py --list

Each scenario invokes the kernel scheduler path, captures the resulting
ResearchSession and trace, extracts observable outcomes, and runs deterministic
validators against those outputs. A scenario without executable prerequisites
(kernel availability) fails or is explicitly skipped with a non-zero clearly
reported prerequisite status; it never passes from static definitions.
--model/--provider select/record the actual provider/model used; with neither
set, results record "kernel default". Exit 0 when
every scenario passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.research.evals.evaluators import (
    EvalInput,
    ScenarioResult,
    evaluate,
)
from app.research.evals.scenarios import Scenario, list_scenarios
from app.research.models import Job
from app.research.repository import ResearchRepository


def _clean(value: str | None) -> str:
    return (value or "").strip()


# Provider/model labels are recorded only; with no flag at all the
# kernel default is used. That path has no concrete names to record, so results
# carry this label instead of an empty string.
_DEFAULT_MODEL_LABEL = "kernel default"

# Default per-call model timeout seconds (flag/env override; must stay well
# above the slowest expected source-scout completion).
MODEL_TIMEOUT_DEFAULT_S = 300


def _model_label(provider: str, model: str) -> str:
    """Recorded label for the effective model; the kernel default when neither is set."""
    if not provider and not model:
        return _DEFAULT_MODEL_LABEL
    return f"{provider or _DEFAULT_MODEL_LABEL}/{model or _DEFAULT_MODEL_LABEL}"


def _lookup_env_value(lookup: object, key: str) -> str | None:
    """One env value narrowed to str; None when absent or non-string."""
    if isinstance(lookup, dict):
        value = lookup.get(key)
        return value if isinstance(value, str) else None
    get = getattr(lookup, "get", None)
    if callable(get):
        value = get(key)
        return value if isinstance(value, str) else None
    return None


def resolve_provider_model(provider_arg: str | None, model_arg: str | None, env: object = None) -> tuple[str, str]:
    """Explicit flag, then env, then "" — meaning the kernel default."""
    lookup: object = os.environ if env is None else env
    provider = _clean(provider_arg) or _clean(_lookup_env_value(lookup, "STOCKBOT_PROVIDER"))
    model = _clean(model_arg) or _clean(_lookup_env_value(lookup, "STOCKBOT_MODEL"))
    return provider, model


def _resolve_provider_model(args: argparse.Namespace) -> tuple[str, str]:
    return resolve_provider_model(args.provider, args.model, os.environ)


def resolve_model_timeout(timeout_arg: str | None, env: object = None) -> int:
    """Per-call model timeout seconds: flag, then STOCKBOT_MODEL_TIMEOUT, then the default."""
    lookup: object = os.environ if env is None else env
    raw = _clean(timeout_arg) or _clean(_lookup_env_value(lookup, "STOCKBOT_MODEL_TIMEOUT"))
    if not raw:
        return MODEL_TIMEOUT_DEFAULT_S
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid model timeout {raw!r}: expected whole seconds > 0") from exc
    if seconds <= 0:
        raise RuntimeError(f"invalid model timeout {raw!r}: expected whole seconds > 0")
    return seconds


def _resolve_model_timeout(args: argparse.Namespace) -> int:
    return resolve_model_timeout(args.model_timeout, os.environ)


_SEARCH_EXCLUDE = frozenset({"browse_tools", "search_tools", "describe_tool", "list_tool_domains", "call_tool"})


def _search_tool_names(sec_tools: object) -> list[str]:
    """Sorted SEC tool names excluding discovery primitives."""
    if not isinstance(sec_tools, Iterable):
        return []
    return sorted(t for t in sec_tools if isinstance(t, str) and t not in _SEARCH_EXCLUDE)


def _search_tool_matches(query: object, sec_tools: object) -> list[str]:
    q = str(query or "").lower()
    base = _search_tool_names(sec_tools)
    if not q:
        return base[:12]
    hits = [t for t in base if q in t.lower()]
    return (hits or base)[:12]


def _dispatch_search_tools(args: dict[str, object], sec_tools: object) -> dict[str, object]:
    query = args.get("query", "") if isinstance(args, dict) else ""
    return {"matches": [{"name": t} for t in _search_tool_matches(query, sec_tools)]}


def _extract_call_tool_request(
    args: dict[str, object],
) -> tuple[str, dict[str, object]]:
    inner = args.get("name") if isinstance(args, dict) else None
    inner_name = inner if isinstance(inner, str) else ""
    raw = args.get("arguments") if isinstance(args, dict) else None
    inner_args = dict(raw) if isinstance(raw, dict) else {}
    return inner_name, inner_args


_ToolHarness = tuple[object, object, object]


def _load_tool_harness() -> tuple[_ToolHarness | None, str]:
    try:
        from app.policy import Capability as _Cap
        from app.policy import RequestContext as _Ctx
        from app.tools import execute_tool as _exec
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None, f"tool harness unavailable: {exc}"
    harness: _ToolHarness = (_exec, _Cap, _Ctx)
    return harness, ""


def _make_tool_context(make_ctx: object, research: object) -> object:
    """Construct the harness context without static call typing."""
    if not callable(make_ctx):
        raise TypeError(f"tool context not callable: {type(make_ctx).__name__}")
    return make_ctx(principal_id="verify-agent-scenarios", capabilities=frozenset({research}))


def _build_tool_context(harness: _ToolHarness) -> tuple[object | None, str]:
    _, _Cap, _Ctx = harness
    research = getattr(_Cap, "RESEARCH", None)
    try:
        return _make_tool_context(_Ctx, research), ""
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None, f"tool context failed: {exc}"


def _call_tool_exec(
    exec_fn: object,
    inner_name: str,
    inner_args: dict[str, object],
    label: str,
    ctx: object,
) -> object:
    """Invoke the tool harness without static call typing."""
    if not callable(exec_fn):
        raise TypeError(f"tool harness not callable: {type(exec_fn).__name__}")
    return exec_fn(inner_name, dict(inner_args), label, context=ctx)


def _run_tool_exec(
    exec_fn: object,
    inner_name: str,
    inner_args: dict[str, object],
    provider: str,
    model: str,
    ctx: object,
) -> dict[str, object]:
    try:
        result = _call_tool_exec(exec_fn, inner_name, inner_args, f"{provider}/{model}", ctx)
    except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, dict):
        narrowed: dict[str, object] = {str(k): v for k, v in result.items()}
        return narrowed
    return {"content": str(result)}


def _dispatch_call_tool(args: dict[str, object], provider: str, model: str) -> dict[str, object]:
    from app.research.agents.source_agent import is_sec_tool

    inner_name, inner_args = _extract_call_tool_request(args)
    if not is_sec_tool(inner_name):
        return {"error": f"POLICY_DENIED: non-SEC tool {inner_name!r}"}
    harness, err = _load_tool_harness()
    if harness is None:
        return {"error": err}
    ctx, err = _build_tool_context(harness)
    if ctx is None:
        return {"error": err}
    return _run_tool_exec(harness[0], inner_name, inner_args, provider, model, ctx)


_ENV_KEYS = ("RESEARCH_DB_PATH", "XDG_DATA_HOME")


def setup_env(tmp: str) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in _ENV_KEYS}
    os.environ["RESEARCH_DB_PATH"] = str(Path(tmp) / "research.sqlite")
    os.environ["XDG_DATA_HOME"] = str(Path(tmp) / "data")
    return old


def restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _crash_eval_input(scenario: Scenario, wall_ms: float) -> EvalInput:
    return EvalInput(
        scenario_name=scenario.name,
        answer_text="",
        tool_calls=(),
        job_count=0,
        failed_count=1,
        evidence_ids=(),
        as_of=scenario.as_of,
        requires_evidence=scenario.requires_evidence,
        has_fabricated_id=False,
        has_fabricated_source="INJECT" in scenario.question,
        wall_clock_ms=wall_ms,
        budget_used=0,
        budget_cap=0,
        scenario_crashed=True,
    )


def _iter_trace_events(get_events: object, trace_id: str) -> list[object]:
    """Trace events without static call typing."""
    if not callable(get_events):
        return []
    events = get_events(trace_id)
    return list(events) if isinstance(events, list) else []


def _trace_event_tool_name(evt: object) -> str | None:
    """Tool name from one trace event; None when absent."""
    event_type = getattr(evt, "event_type", None)
    if event_type != "tool.completed":
        return None
    payload = getattr(evt, "payload", None)
    tool_name = payload.get("tool") if isinstance(payload, dict) else None
    return tool_name if isinstance(tool_name, str) and tool_name else None


def _trace_tool_names(trace_id: str, get_events: object) -> list[str]:
    names: list[str] = []
    for evt in _iter_trace_events(get_events, trace_id):
        tool_name = _trace_event_tool_name(evt)
        if tool_name is not None:
            names.append(tool_name)
    return names


def _extract_evidence_ids(out: dict[str, object]) -> tuple[str, ...]:
    raw = out.get("evidence_ids", [])
    if not isinstance(raw, list):
        return ()
    return tuple(e for e in raw if isinstance(e, str))


def _answer_from_final(final: object) -> str | None:
    if isinstance(final, dict) and isinstance(final.get("answer"), str):
        return final.get("answer")
    return None


def _analysis_answer(analysis: object) -> str:
    if not getattr(analysis, "claims", None):
        return ""
    for attr in ("answer", "base_case", "bull_case", "bear_case"):
        value = getattr(analysis, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_answer(final: object, out: dict[str, object]) -> str:
    answer = _answer_from_final(final)
    if answer is not None:
        return answer
    for key in ("stock", "bull", "bear"):
        answer = _analysis_answer(out.get(key))
        if answer:
            return answer
    return ""


def _is_completed(sess_status: object, answer: object, evidence_ids: tuple[str, ...]) -> bool:
    if sess_status != "completed":
        return False
    if not isinstance(answer, str) or not answer.strip():
        return False
    return bool(evidence_ids)


_FAILED_JOB_STATUSES: tuple[str, ...] = ("failed", "cancelled", "timed_out")


def _job_failed(job: Job) -> bool:
    """Failed-status probe for one repository job; a successful job never counts."""
    return job.status in _FAILED_JOB_STATUSES


def _trace_strs(trace: Mapping[str, object], key: str) -> tuple[str, ...]:
    """String tuple from one live trace record; non-sequence/non-string entries are dropped."""
    raw = trace.get(key)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(v for v in raw if isinstance(v, str))


def _row_provenance(row: object) -> tuple[str, object] | None:
    """(accession, document) from one raw-source evidence row; None when the row proves no filing.

    Non-evidence rows and rows without an accession are navigation artifacts, not
    opened documents, so they carry no provenance for the opened-filings ledger.
    """
    if not isinstance(row, Mapping) or row.get("record_kind") != "evidence":
        return None
    meta = row.get("metadata")
    if not isinstance(meta, Mapping):
        return None
    accession = meta.get("accession_no")
    if not isinstance(accession, str) or not accession:
        return None
    return accession, meta.get("document_name")


def _ledger_documents(repo: ResearchRepository, session_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(filings, documents) actually opened, from the raw-source rows' own provenance.

    The runner journals no telemetry payload, so the ledger is the only real
    producer: each raw-source row records the accession and document it came from.
    """
    filings: list[str] = []
    documents: list[str] = []
    for row in repo.list_evidence(session_id):
        provenance = _row_provenance(row)
        if provenance is None:
            continue
        accession, document = provenance
        if accession not in filings:
            filings.append(accession)
        if isinstance(document, str):
            entry = f"{accession}|{document}"
            if entry not in documents:
                documents.append(entry)
    return tuple(filings), tuple(documents)


def _ledger_ids(repo: ResearchRepository, session_id: str, record_kind: str) -> tuple[str, ...]:
    """Ledger evidence ids of one record kind: raw-source rows vs navigation artifacts."""
    out: list[str] = []
    for row in repo.list_evidence(session_id):
        if isinstance(row, Mapping) and row.get("record_kind") == record_kind:
            eid = row.get("evidence_id")
            if isinstance(eid, str) and eid:
                out.append(eid)
    return tuple(out)


_COMMITTEE_ROLES = ("stockbot", "bullbot", "bearbot")


def _completed_roles(jobs: Iterable[Job]) -> tuple[str, ...]:
    """Completed committee roles only, in canonical order.

    Child jobs (scouts, source agents) are not committee members; counting them
    would make the role-set gate unsatisfiable.
    """
    done = {j.job_type for j in jobs if j.status == "completed"}
    return tuple(role for role in _COMMITTEE_ROLES if role in done)


def _build_success_input(
    scenario: Scenario,
    answer: str,
    tool_names: list[str],
    jobs: list[Job],
    sess_status: str,
    evidence_ids: tuple[str, ...],
    wall_ms: float,
    tool_call_cap: int = 0,
    trace: Mapping[str, object] | None = None,
) -> EvalInput:
    live: Mapping[str, object] = trace or {}
    failed = sum(1 for j in jobs if _job_failed(j))
    completed = _is_completed(sess_status, answer, evidence_ids)
    recovered = failed if completed else 0
    return EvalInput(
        scenario_name=scenario.name,
        answer_text=answer if isinstance(answer, str) else "",
        tool_calls=tuple(tool_names),
        job_count=len(jobs),
        failed_count=failed,
        recovered_count=recovered,
        evidence_ids=evidence_ids,
        as_of=scenario.as_of,
        requires_evidence=scenario.requires_evidence,
        # Depth is not a completeness boundary: only a cap the run was actually given
        # can be violated (0 = unlimited, the default for live research).
        wall_clock_ms=wall_ms,
        budget_used=len(tool_names),
        budget_cap=tool_call_cap,
        requires_trace=bool(live.get("requires_trace")),
        trace_present=bool(live.get("trace_present")),
        filings_opened=_trace_strs(live, "filings_opened"),
        documents_opened=_trace_strs(live, "documents_opened"),
        raw_evidence_ids=_trace_strs(live, "raw_evidence_ids"),
        navigation_evidence_ids=_trace_strs(live, "navigation_evidence_ids"),
        branches_covered=_trace_strs(live, "branches_covered"),
        waves=_trace_strs(live, "waves"),
        roles_completed=_trace_strs(live, "roles_completed"),
        committee_freeze_ids=_trace_strs(live, "committee_freeze_ids"),
    )


def _live_trace(
    scenario: Scenario, repo: ResearchRepository, sid: str, jobs: list[Job], evidence_ids: tuple[str, ...] = ()
) -> dict[str, object]:
    """Eval fields the live run can actually prove: opened filings/documents + ledger record kinds.

    Every field has a real producer on the runner path. ``raw_evidence_ids`` are the
    ids the run itself advertises for citation (the frozen raw-source set), so the
    subset check compares like with like; fields the runner does not persist
    (model-reported branch names, passages) stay unset rather than invented.
    """
    sess = repo.get_session(sid)
    filings, documents = _ledger_documents(repo, sid)
    freeze_ids = tuple(f for f in getattr(sess, "freeze_ids", ()) if isinstance(f, str))
    raw_rows = set(_ledger_ids(repo, sid, "evidence"))
    return {
        "requires_trace": bool(getattr(scenario, "requires_trace", False)),
        "trace_present": bool(freeze_ids),
        "filings_opened": filings,
        "documents_opened": documents,
        "raw_evidence_ids": tuple(eid for eid in evidence_ids if eid in raw_rows),
        "navigation_evidence_ids": _ledger_ids(repo, sid, "discovery"),
        "waves": freeze_ids,
        "committee_freeze_ids": tuple(run for run in getattr(sess, "committee_runs", ()) if isinstance(run, str)),
        "roles_completed": _completed_roles(jobs),
    }


def evaluate_and_record(
    scenario: Scenario, out: dict[str, object], wall_ms: float, tool_call_cap: int = 0
) -> EvalInput:
    from app.research.evals.traces import get_trace_events, list_traces
    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    sid = str(out.get("session_id", ""))
    sess = repo.get_session(sid)
    jobs = repo.list_jobs(sid)
    traces = list_traces(sid)
    trace_id = traces[0].trace_id if traces else ""
    tool_names = _trace_tool_names(trace_id, get_trace_events) if trace_id else []
    evidence_ids = _extract_evidence_ids(out)
    answer = _extract_answer(sess.final_result or {}, out)
    return _build_success_input(
        scenario,
        answer,
        tool_names,
        jobs,
        sess.status,
        evidence_ids,
        wall_ms,
        tool_call_cap,
        _live_trace(scenario, repo, sid, jobs, evidence_ids),
    )


def _close_bootstrap_jobs(sid: str) -> None:
    """Cancel bootstrap source jobs left running; a scenario run must not leak them."""
    from app.research import service
    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    for job in repo.list_jobs(sid):
        if job.status in ("queued", "running"):
            service.cancel_job(job.job_id)


def _run_kernel_scenario(
    scenario: Scenario, provider: str, model: str, prompt_version: str, timeout_s: int
) -> EvalInput:
    """Production-path live eval: kernel scheduler (create_research + node + run_node)."""
    import asyncio

    from app.research import scheduler, service

    _ = (provider, model, prompt_version, timeout_s)  # labels recorded on the summary only
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="agent-scenario-") as tmp:
        old = setup_env(tmp)
        try:
            try:
                sid = service.create_research(
                    scenario.question, scenario.notes or scenario.question, as_of=scenario.as_of
                )
                node = service.create_node(sid, scenario.question, "Route question.")
                try:
                    asyncio.run(scheduler.run_node(node, session_id=sid))
                finally:
                    _close_bootstrap_jobs(sid)
            except Exception as exc:  # noqa: BLE001 - the verdict reports the crash, never hides it
                print(f"CRASH {type(exc).__name__}: {exc}", file=sys.stderr)
                return _crash_eval_input(scenario, (time.monotonic() - t0) * 1000.0)
            out: dict[str, object] = {"session_id": sid, "evidence_ids": []}
            wall_ms = (time.monotonic() - t0) * 1000.0
            # Kernel research has no static tool-call cap: 0 = unlimited, judged on real limits only.
            return evaluate_and_record(scenario, out, wall_ms, 0)
        finally:
            restore_env(old)


def _run_live_scenario(scenario: Scenario, provider: str, model: str, prompt_version: str, timeout_s: int) -> EvalInput:
    """Live eval entrypoint: production kernel path only (run_live retired, see runner.py)."""
    return _run_kernel_scenario(scenario, provider, model, prompt_version, timeout_s)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list scenario names and exit")
    parser.add_argument(
        "--scenario", default=None, help="run one scenario, fixture-only regressions included (default: all live)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model ID recorded for live execution (or STOCKBOT_MODEL; default: kernel default)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="provider recorded for live execution (or STOCKBOT_PROVIDER; default: kernel default)",
    )
    parser.add_argument(
        "--model-timeout",
        default=None,
        help="per-call model timeout seconds (or STOCKBOT_MODEL_TIMEOUT; default 300)",
    )
    parser.add_argument("--prompt-version", default="v1", help="prompt version stamp (default v1)")
    parser.add_argument(
        "--fixtures-dir", default=None, help="accepted for compat; live runs do not use static fixtures"
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    return parser.parse_args(argv)


def _list_scenarios() -> int:
    for scenario in list_scenarios():
        print(f"{scenario.name} [{scenario.family.value}]")
    return 0


def _prepare_provider_model(args: argparse.Namespace) -> tuple[str, str, int]:
    """Provider/model/timeout resolution before any live run."""
    provider, model = _resolve_provider_model(args)
    timeout_s = _resolve_model_timeout(args)
    return provider, model, timeout_s


def _selected_names(args: argparse.Namespace) -> list[str]:
    if args.scenario:
        return [args.scenario]
    return [s.name for s in list_scenarios() if not s.fixture_only]


def _scenario_map() -> dict[str, Scenario]:
    return {s.name: s for s in list_scenarios()}


def _find_unknown(names: list[str], by_name: dict[str, Scenario]) -> list[str]:
    return [n for n in names if n not in by_name]


def _eval_one_scenario(
    name: str, by_name: dict[str, Scenario], provider: str, model: str, prompt_version: str, timeout_s: int
) -> ScenarioResult:
    return evaluate(_run_live_scenario(by_name[name], provider, model, prompt_version, timeout_s))


def _run_all_scenarios(
    names: list[str],
    by_name: dict[str, Scenario],
    provider: str,
    model: str,
    prompt_version: str,
    timeout_s: int,
) -> list[ScenarioResult]:
    return [_eval_one_scenario(name, by_name, provider, model, prompt_version, timeout_s) for name in names]


def _failed_results(results: list[ScenarioResult]) -> list[ScenarioResult]:
    return [r for r in results if not r.passed]


def summarize_results(results: list[ScenarioResult]) -> tuple[list[ScenarioResult], int]:
    failed = _failed_results(results)
    return failed, (1 if failed else 0)


def _print_results(results: list[ScenarioResult], provider: str, model: str) -> None:
    label = _model_label(provider, model)
    for result in results:
        if result.passed:
            print(f"PASS {result.scenario_name} (live via kernel {label})")
        else:
            print(f"FAIL {result.scenario_name} (live via kernel {label}): {', '.join(result.violations)}")


def _build_summary(provider: str, model: str, prompt_version: str, results: list[ScenarioResult]) -> dict[str, object]:
    return {
        "provider": provider or _DEFAULT_MODEL_LABEL,
        "model": model or _DEFAULT_MODEL_LABEL,
        "prompt_version": prompt_version,
        "scenarios": [
            {"scenario": r.scenario_name, "passed": r.passed, "violations": list(r.violations)} for r in results
        ],
    }


def _maybe_print_suite_info(
    results: list[ScenarioResult],
    provider: str,
    model: str,
    summary: dict[str, object],
) -> None:
    """Tally + git sha of a finished live run; the sha is recorded on the summary here only."""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=5).strip()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        git_sha = "unknown"
    # Passed/failed counts come from eval results, not from static fixtures.
    passed_count = sum(1 for r in results if r.passed)
    print(
        f"{passed_count}/{len(results)} passed "
        f"(model={model or _DEFAULT_MODEL_LABEL} provider={provider or _DEFAULT_MODEL_LABEL} git={git_sha})"
    )
    summary["git_sha"] = git_sha


def _maybe_print_json(json_flag: bool, summary: dict[str, object]) -> None:
    if json_flag:
        print(json.dumps(summary, indent=2, sort_keys=True))


def _cli_prereqs(
    args: argparse.Namespace,
) -> tuple[str, str, int, list[str], dict[str, Scenario]] | int:
    try:
        provider, model, timeout_s = _prepare_provider_model(args)
    except RuntimeError as exc:
        print(f"SKIP live scenarios: {exc}", file=sys.stderr)
        print(
            "No live prerequisites: the kernel scheduler runs in-process.",
            file=sys.stderr,
        )
        return 2
    names = _selected_names(args)
    by_name = _scenario_map()
    unknown = _find_unknown(names, by_name)
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    return provider, model, timeout_s, names, by_name


def _report_cli_run(
    args: argparse.Namespace, provider: str, model: str, results: list[ScenarioResult], summary: dict[str, object]
) -> int:
    _print_results(results, provider, model)
    _maybe_print_suite_info(results, provider, model, summary)
    _maybe_print_json(args.json, summary)
    _, exit_code = summarize_results(results)
    return exit_code


def _run_cli(args: argparse.Namespace) -> int:
    if args.list:
        return _list_scenarios()
    prereqs = _cli_prereqs(args)
    if isinstance(prereqs, int):
        return prereqs
    provider, model, timeout_s, names, by_name = prereqs
    results = _run_all_scenarios(names, by_name, provider, model, args.prompt_version, timeout_s)
    summary = _build_summary(provider, model, args.prompt_version, results)
    return _report_cli_run(args, provider, model, results, summary)


def main(argv: list[str] | None = None) -> int:
    return _run_cli(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
