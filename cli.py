"""Stockbot admin CLI — research state, runs, data refresh, log server, login (no chat; Pi is the harness)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Protocol, runtime_checkable


@runtime_checkable
class _HasRunId(Protocol):
    """Structural seam: RunOutcome or test double with run_id."""

    run_id: str


if TYPE_CHECKING:
    from app.domain.risk.evaluation import RiskEvaluation
    from app.thesis.models import JSONValue, Thesis
    from app.thesis.monitor import TickResult
    from app.thesis.repository import ThesisRepository

from app.config import configure_logging
from app.log_server import DEFAULT_LOG_SERVER_PORT, run_log_server
from app.robinhood.auth import DEFAULT_TOKEN_PATH
from app.services.mandate import load_mandate_file
from app.services.risk import evaluate_latest_mandate
from app.storage.runs import (
    get_events,
    get_evidence,
    get_model_calls,
    get_run,
    get_security_events,
    get_security_summary,
    get_tool_calls,
    list_runs,
)
from app.tool_render import issue_to_prose
from app.tools import authorize_robinhood_browser

_LOG_SERVER_DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_LOG_SERVER_PORT}"
_MAX_LEADERBOARD_PRINT = 25
_SUBCOMMANDS = (
    "runs",
    "inspect",
    "refresh-data",
    "log-server",
    "robinhood-login",
    "backfill-sec",
    "resume-sec-backfill",
    "sec-coverage",
    "thesis",
    "google-data",
    "research",
    "trace",
    "eval",
)


def _as_data_root(data_root: str | Path | None) -> Path | None:
    """Shared --data-root normalization: str -> Path, otherwise passthrough."""
    if isinstance(data_root, str):
        return Path(data_root)
    return data_root


def _coerce_coverage_int(value: object) -> int:
    """Coverage counts are ints; anything else (None, str) reads as 0."""
    if isinstance(value, int):
        return value
    return 0


def _coverage_date_str(row: dict[str, object]) -> str:
    seen = row.get("coverage_date")
    if isinstance(seen, str):
        return seen[:10]
    return ""


def _coerce_run_number(value: object) -> float:
    """Run-table numerics are ints/floats; anything else reads as 0."""
    if isinstance(value, (int, float)):
        return value
    return 0.0


def _format_run_started(value: object) -> str:
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value).astimezone().isoformat()
    return ""


def _format_run_row(row: dict[str, object]) -> str:
    duration = _coerce_run_number(row["duration_ms"])
    cost = _coerce_run_number(row["estimated_total_cost"])
    question_raw = row["question"]
    question = (question_raw if isinstance(question_raw, str) else "")[:60]
    started_local = _format_run_started(row["started_at"])
    return (
        f"{row['run_id']:<38} {started_local[:26]:<26} "
        f"{(row['status'] or ''):<16} {duration:>10.0f} {cost:>9.6f}  {question}"
    )


def _cmd_runs(limit: int) -> None:
    rows = list_runs(limit=limit)
    if not rows:
        print("No runs recorded.")
        return
    print(f"{'run_id':<38} {'started_at':<26} {'status':<16} {'duration_ms':>10} {'cost':>9}  question")
    for row in rows:
        print(_format_run_row(row))


def _refresh_coverage_counts(result: dict[str, object]) -> tuple[int, int, int, int, float]:
    """Normalize the materialized coverage block into printable counts."""
    coverage_raw = result["coverage"]
    coverage: dict[str, object] = coverage_raw if isinstance(coverage_raw, dict) else {}
    finra_rows = _coerce_coverage_int(coverage.get("finra_rows"))
    mapped = _coerce_coverage_int(coverage.get("mapped_rows"))
    shares_covered = _coerce_coverage_int(coverage.get("shares_outstanding_rows"))
    eligible = _coerce_coverage_int(coverage.get("eligible_rows"))
    pct = 100.0 * eligible / finra_rows if finra_rows else 0.0
    return finra_rows, mapped, shares_covered, eligible, pct


def _print_coverage_report(finra_rows: int, mapped: int, shares_covered: int, eligible: int, pct: float) -> None:
    print(f"FINRA securities:             {finra_rows:,}")
    print(f"Ticker mappings:              {mapped:,}")
    print(f"Shares-outstanding coverage:  {shares_covered:,}")
    print(f"Eligible screen universe:     {eligible:,}")
    print()
    print(f"Coverage: {pct:.1f}%")


def _leaderboard_tickers(entries: list[object]) -> list[object]:
    return [e["ticker"] for e in entries if isinstance(e, dict)]


def _print_leaderboard_entries(result: dict[str, object]) -> None:
    entries_raw = result.get("entries", [])
    entries: list[object] = entries_raw if isinstance(entries_raw, list) else []
    print(f"Leaderboard entries: {_leaderboard_tickers(entries)}")


def _cmd_refresh_data(
    settlement_date: str, tickers: list[str], ciks: list[int], data_root: str | Path | None = None
) -> None:
    """Live leaderboard read for one settlement cycle (ephemeral, never persisted)."""
    del tickers, ciks, data_root
    from app.analytics.screens import get_short_interest_leaderboard

    result = get_short_interest_leaderboard(limit=_MAX_LEADERBOARD_PRINT, settlement_date=settlement_date)
    if result.get("error"):
        print(f"Leaderboard error: {result['error']}")
        return
    finra_rows, mapped, shares_covered, eligible, pct = _refresh_coverage_counts(result)
    _print_coverage_report(finra_rows, mapped, shares_covered, eligible, pct)
    _print_leaderboard_entries(result)


def _cmd_replay_sec_facts() -> None:
    print(json.dumps({"status": "removed", "reason": "SEC facts read live via providers; no replay"}))


def _cmd_refresh_obligations(ticker: str) -> None:
    """Evaluate obligations in memory (never persisted); caller logs output to the bundle."""
    from app import obligations

    result = obligations.get_obligations(ticker)
    print(json.dumps(result, indent=2))


def _cmd_robinhood_login() -> None:
    print("Starting Robinhood authorization...")
    if authorize_robinhood_browser():
        print(f"Robinhood authorized. Tokens stored at {DEFAULT_TOKEN_PATH}")
    else:
        print("Robinhood authorization failed or was declined.")
        raise SystemExit(1)


def _cmd_log_server(port: int) -> None:
    try:
        run_log_server(port)
    except OSError as exc:
        print(f"error: cannot bind port {port}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _print_run_record(run: dict[str, object]) -> None:
    for key, value in run.items():
        if key in ("started_at", "completed_at") and isinstance(value, str) and value:
            value = datetime.fromisoformat(value).astimezone().isoformat()
        print(f"{key}: {value}")


def _print_inspect_events(run_id: str) -> None:
    print("events (seq type round tool duration_ms summary):")
    for ev in get_events(run_id):
        result_raw = ev.get("result_summary")
        summary = (result_raw if isinstance(result_raw, str) else "").replace("\n", " ")[:80]
        duration = ev["duration_ms"] if ev.get("duration_ms") is not None else ""
        print(
            f"{ev['sequence']:>4} {ev['event_type']:<20} {ev.get('round')!s:<6} "
            f"{(ev.get('tool_name') or ''):<24} {duration!s:<10} {summary}"
        )


def _print_inspect_tool_calls(run_id: str) -> None:
    print("tool calls:")
    for tc in get_tool_calls(run_id):
        print(
            f"  {tc['tool_call_id']} {tc['tool_name']} {tc['status']} "
            f"rows={tc['result_row_count']} bytes={tc['result_bytes']} "
            f"err={tc['error_type']} {tc['error_message'] or ''}"
        )


def _print_inspect_evidence(run_id: str) -> None:
    print("search_web evidence:")
    for ev in get_evidence(run_id):
        if ev["tool_name"] == "search_web":
            rendered_raw = ev.get("rendered_text")
            snippet = (rendered_raw if isinstance(rendered_raw, str) else "").replace("\n", " ")[:200]
            print(f"  {ev['evidence_id']} {ev['tool_call_id']} {snippet}")


def _print_inspect_model_calls(run_id: str) -> None:
    print("model calls:")
    for mc in get_model_calls(run_id):
        print(
            f"  {mc['model_call_id']} {mc['provider']}/{mc['model']} "
            f"in={mc['input_tokens']} out={mc['output_tokens']} "
            f"cost={mc['estimated_cost']} finish={mc['finish_reason']} "
            f"req={mc['provider_request_id']}"
        )


def _format_security_event(event: dict[str, object]) -> str:
    return (
        f"  {event.get('source') or ''} | score={event.get('score')} | "
        f"{event.get('verdict') or ''} | rules={event.get('rule_ids') or ''} | "
        f"{event.get('decision')} | {event.get('reason') or ''} | "
        f"{event.get('created_at') or ''}"
    )


def _print_inspect_security(run_id: str) -> None:
    print("SECURITY:")
    summary = get_security_summary(run_id)
    print(
        f"  allowed={summary['allowed']} quarantined={summary['quarantined']} "
        f"blocked={summary['blocked']} action_blocked={summary['action_blocked']} "
        f"egress_blocked={summary['egress_blocked']} "
        f"response_stripped={summary['response_stripped']}"
    )
    for event in get_security_events(run_id):
        line = _format_security_event(event)
        if event.get("span_length") is not None:
            line += f" | stripped_span={event['span_length']} chars"
        print(line)


def _cmd_inspect(run_id: str) -> None:
    run = get_run(run_id)
    if run is None:
        print(f"error: no run found for {run_id}", file=sys.stderr)
        sys.exit(1)
    _print_run_record(run)
    print()
    _print_inspect_events(run_id)
    print()
    _print_inspect_tool_calls(run_id)
    print()
    _print_inspect_evidence(run_id)
    print()
    _print_inspect_model_calls(run_id)
    print()
    _print_inspect_security(run_id)


def _load_mandate_evaluation(mandate_path: Path, data_root: str | Path | None):
    """Load evaluation + mandate file; exit 1 on missing/invalid input."""
    try:
        root = _as_data_root(data_root)
        evaluation = evaluate_latest_mandate(mandate_path, data_root=root)
        mandate = load_mandate_file(mandate_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    return evaluation, mandate


_UnitKey = tuple[str, "str | None"]
_MandateValue = Decimal | str | None


def _format_mandate_value(value: _MandateValue, metric: str, target: str | None, units: dict[_UnitKey, str]) -> str:
    if value is None:
        return ""
    if metric == "prohibited_assets":
        return str(value)
    unit = units.get((metric, target))
    if unit == "dollars":
        return str(value)
    return f"{float(value) * 100:.1f}%"


def _print_mandate_sector_exposures(evaluation: RiskEvaluation) -> None:
    if evaluation.sector_exposures:
        print(
            "Sector exposures: "
            + ", ".join(
                f"{sector} {float(weight) * 100:.1f}%" for sector, weight in evaluation.sector_exposures.items()
            )
        )


def _print_mandate_breaches(evaluation: RiskEvaluation, units: dict[_UnitKey, str]) -> None:
    if not evaluation.breaches:
        print("No breaches.")
        return
    print("Breaches:")
    for breach in evaluation.breaches:
        target = f" {breach.target}" if breach.target else ""
        line = (
            f"    [{breach.severity}] {breach.metric}{target}: "
            f"actual {_format_mandate_value(breach.actual, breach.metric, breach.target, units)}, "
            f"limit {_format_mandate_value(breach.limit, breach.metric, breach.target, units)}"
        )
        if breach.excess is not None:
            line += f", excess {_format_mandate_value(breach.excess, breach.metric, breach.target, units)}"
        print(line)


def _print_mandate_issues(evaluation: RiskEvaluation) -> None:
    if evaluation.issues:
        print("Not evaluable:")
        for issue in evaluation.issues:
            print(f"    - {issue_to_prose(issue)}")


def _cmd_evaluate_mandate(mandate_path: Path, data_root: str | Path | None) -> None:
    """Evaluate a mandate against the latest persisted snapshot; report or exit 1."""
    evaluation, mandate = _load_mandate_evaluation(mandate_path, data_root)
    units = {(limit.metric, limit.target): limit.unit for limit in mandate.limits}
    print(f"Mandate: {mandate_path}")
    print(f"Snapshot: {evaluation.snapshot_id} created {evaluation.created_at.astimezone().isoformat()}")
    _print_mandate_sector_exposures(evaluation)
    _print_mandate_breaches(evaluation, units)
    _print_mandate_issues(evaluation)


def _resolve_backfill_quarters(from_date: str, to_date: str) -> list[tuple[int, int]]:
    """Parse the backfill range into quarterly partitions; exit 2 on bad dates."""
    from app.sec.discovery.service import _quarters_for_range

    try:
        quarters, _ = _quarters_for_range(from_date, to_date, cap=10_000)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    return quarters


def _enqueue_backfill_jobs(
    sec_store: ModuleType,
    source: str | None,
    forms: list[str],
    quarters: list[tuple[int, int]],
    batch_size: int,
    data_root: str | None,
) -> list[str]:
    """Enqueue one job per form/quarter; exit 2 when the store rejects input."""
    from app.sec.discovery.service import BACKFILL_SOURCE, _quarter_dates

    ids: list[str] = []
    for form in forms:
        for year, quarter in quarters:
            qs, qe = _quarter_dates(year, quarter)
            try:
                ids.append(
                    sec_store.enqueue_backfill_job(
                        source or BACKFILL_SOURCE, form, qs, qe, batch_size=batch_size, root=data_root
                    )
                )
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                raise SystemExit(2)
    return ids


def _validate_backfill_forms(forms: list[str]) -> None:
    if not forms:
        print("error: --form is required (e.g. --form 10-K)", file=sys.stderr)
        raise SystemExit(2)


def _drain_backfill_inline(ids: list[str], data_root: str | None) -> None:
    from app.sec.discovery.service import drain_backfill_queue

    print(f"queued {len(ids)} job(s): {ids}")
    print(json.dumps({"jobs": ids, **drain_backfill_queue(data_root)}, indent=2))


def _cmd_backfill_sec(
    source: str | None, forms: list[str], from_date: str, to_date: str, batch_size: int, data_root: str | None
) -> None:
    """Enqueue bounded quarterly/form jobs for the range, then drain inline."""
    from app.sec import store as sec_store

    _validate_backfill_forms(forms)
    quarters = _resolve_backfill_quarters(from_date, to_date)
    if not quarters:
        print(
            "no quarterly partitions in range (before 1993 global indexes or current quarter only); nothing to backfill"
        )
        return
    _drain_backfill_inline(_enqueue_backfill_jobs(sec_store, source, forms, quarters, batch_size, data_root), data_root)


def _requeue_one_backfill_job(sec_store: ModuleType, job_id: str, data_root: str | None) -> None:
    if sec_store.get_job(job_id, root=data_root) is None:
        print(f"error: no backfill job {job_id!r}", file=sys.stderr)
        raise SystemExit(1)
    sec_store.requeue_job(job_id, root=data_root)
    print(f"requeued {job_id}")


def _requeue_failed_backfill_jobs(sec_store: ModuleType, data_root: str | None) -> None:
    failed = sec_store.list_jobs(status="failed", root=data_root)
    for job in failed:
        job_id_raw = job.get("id")
        if isinstance(job_id_raw, str):
            sec_store.requeue_job(job_id_raw, root=data_root)
    print(f"requeued {len(failed)} failed job(s)")


def _cmd_resume_sec_backfill(job_id: str | None, data_root: str | None) -> None:
    """Requeue one (or all) interrupted jobs and drain the queue inline."""
    from app.sec import store as sec_store
    from app.sec.discovery.service import drain_backfill_queue

    if job_id:
        _requeue_one_backfill_job(sec_store, job_id, data_root)
    else:
        _requeue_failed_backfill_jobs(sec_store, data_root)
    print(json.dumps(drain_backfill_queue(data_root), indent=2))


def _validate_coverage_date(label: str, value: str | None) -> None:
    if value is None:
        return
    try:
        datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        print(f"error: --{label} must be YYYY-MM-DD, got {value!r}", file=sys.stderr)
        raise SystemExit(2)


def _validate_coverage_range(from_date: str | None, to_date: str | None) -> None:
    for label, value in (("from", from_date), ("to", to_date)):
        _validate_coverage_date(label, value)


def _filter_coverage_rows(
    rows: list[dict[str, object]], from_date: str | None, to_date: str | None
) -> list[dict[str, object]]:
    if from_date:
        rows = [r for r in rows if _coverage_date_str(r) >= from_date]
    if to_date:
        rows = [r for r in rows if _coverage_date_str(r) <= to_date]
    return rows


def _print_coverage_rows(rows: list[dict[str, object]]) -> None:
    for row in rows:
        print(
            f"{row.get('source')} {row.get('form')} "
            f"{row.get('date_partition')} {row.get('status')} "
            f"count={row.get('accession_count')} last={row.get('last_key')}"
        )
    if not rows:
        print("no coverage rows")


def _print_pending_backfill_jobs(sec_store: ModuleType, data_root: str | None) -> None:
    pending = [j for j in sec_store.list_jobs(root=data_root) if j["status"] in ("queued", "running", "failed")]
    if pending:
        print(f"pending jobs: {[j['id'] for j in pending]}")
    else:
        print("no pending backfill jobs")


def _cmd_sec_coverage(
    source: str | None, form: str | None, from_date: str | None, to_date: str | None, data_root: str | None
) -> None:
    """Show ingestion coverage rows plus pending backfill jobs."""
    from app.sec import store as sec_store

    _validate_coverage_range(from_date, to_date)
    rows = sec_store.query_coverage(source=source, form=form, root=data_root)
    rows = _filter_coverage_rows(rows, from_date, to_date)
    _print_coverage_rows(rows)
    _print_pending_backfill_jobs(sec_store, data_root)


def _thesis_repo(args: argparse.Namespace) -> ThesisRepository:
    """Thesis root via the existing data-root mechanism (<root>/thesis)."""
    from app.config import get_data_root
    from app.thesis.repository import ThesisRepository

    raw_root = getattr(args, "data_root", None)
    override = raw_root if isinstance(raw_root, str) else None
    base = Path(override) if override else get_data_root()
    return ThesisRepository(base / "thesis")


def _thesis_load(repo: ThesisRepository, id_or_slug: str) -> Thesis:
    try:
        return repo.load_thesis(id_or_slug)
    except KeyError as exc:
        raise SystemExit(f"thesis: {exc}") from exc


def _thesis_list(args: argparse.Namespace) -> None:
    repo = _thesis_repo(args)
    rows = repo.list_theses()
    if not rows:
        print("No theses.")
        return
    for t in rows:
        one = t.user_thesis.splitlines()[0][:100] if t.user_thesis else ""
        print(f"{t.thesis_id} {t.slug} [{t.status}] {t.updated_at} {one}")


def _thesis_show(args: argparse.Namespace) -> None:
    repo = _thesis_repo(args)
    thesis = _thesis_load(repo, str(args.id))
    state = repo.load_state(thesis.thesis_id)
    print(f"{thesis.thesis_id} ({thesis.slug}) [{thesis.status}] updated {thesis.updated_at}")
    print(f"Thesis: {thesis.user_thesis}")
    print(f"Scope: {thesis.scope}")
    for c in thesis.claims:
        print(f"- claim [{c.status}]: {c.statement}")
    for key in ("assumptions", "invalidators", "unknowns"):
        vals = getattr(thesis, key)
        if vals:
            print(f"{key.capitalize()}: {'; '.join(vals)}")
    print(f"Assessment: {state.assessment}")
    print("Expressions:")
    if not thesis.expressions:
        print("  (none)")
    for e in thesis.expressions:
        print(f"  - {e.intent} {e.instrument}/{e.direction} ({e.structure}, {e.horizon}) [{e.status}]")


def _thesis_status(args: argparse.Namespace) -> None:
    repo = _thesis_repo(args)
    op = {"pause": repo.pause_thesis, "resume": repo.resume_thesis, "close": repo.close_thesis}[args.thesis_command]
    try:
        updated = op(str(args.id))
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"thesis: {exc}") from exc
    print(f"{updated.thesis_id} ({updated.slug}) [{updated.status}]")


def _check_inspect_file(name: str, fn: Callable[[], object], problems: list[str]) -> None:
    """Run one inspect validator; record failures without aborting the rest."""
    try:
        fn()
        print(f"{name}: ok")
    except Exception as exc:  # noqa: BLE001 - inspect records any validator failure, never aborts
        print(f"{name}: INVALID ({exc})")
        problems.append(name)


def _owned_inspect_doc(d: Path, thesis_id: str, name: str) -> dict[str, JSONValue]:
    """Load a thesis-owned YAML doc and verify its thesis_id matches."""
    from app.thesis.yaml import load_raw_yaml

    raw = load_raw_yaml(d / name)
    if raw.get("thesis_id") != thesis_id:
        raise ValueError(f"{d / name}: thesis_id mismatch")
    return raw


def _check_inspect_entry_list(d: Path, thesis_id: str, name: str, key: str, from_dict: Callable[..., object]) -> None:
    """Validate every entry of a thesis-owned YAML list doc."""
    items = _owned_inspect_doc(d, thesis_id, name).get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"{d / name}: '{key}' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    for entry in items:
        if not isinstance(entry, dict):
            raise ValueError(f"{d / name}: {key.rstrip('s')} entry must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        from_dict(entry, str(d / name))


def _thesis_inspect(args: argparse.Namespace) -> None:
    from app.thesis.models import (
        Checkpoint,
        Thesis,
        ThesisMemory,
        ThesisQuestion,
        ThesisState,
        WatchRule,
    )
    from app.thesis.yaml import load_yaml

    repo = _thesis_repo(args)
    thesis = _thesis_load(repo, str(args.id))
    d = repo.root / thesis.slug
    problems: list[str] = []

    def _questions() -> None:
        _check_inspect_entry_list(d, thesis.thesis_id, "questions.yaml", "questions", ThesisQuestion.from_dict)

    def _watch() -> None:
        _check_inspect_entry_list(d, thesis.thesis_id, "watch.yaml", "rules", WatchRule.from_dict)

    def _memory() -> None:
        _check_inspect_entry_list(d, thesis.thesis_id, "memory.yaml", "memories", ThesisMemory.from_dict)

    _check_inspect_file("thesis.yaml", lambda: load_yaml(d / "thesis.yaml", Thesis), problems)
    _check_inspect_file("state.yaml", lambda: load_yaml(d / "state.yaml", ThesisState), problems)
    _check_inspect_file("questions.yaml", _questions, problems)
    _check_inspect_file("watch.yaml", _watch, problems)
    _check_inspect_file("memory.yaml", _memory, problems)
    _check_inspect_file("checkpoint.yaml", lambda: load_yaml(d / "checkpoint.yaml", Checkpoint), problems)
    triggers = repo.load_triggers(thesis.thesis_id)
    pending = sum(1 for t in triggers if t.status == "pending")
    print(f"triggers: {pending} pending / {len(triggers)} total")
    for name in ("evidence", "journal"):
        sub = d / name
        n = len([p for p in sub.glob("*.md" if name == "journal" else "*") if p.is_file()]) if sub.is_dir() else 0
        print(f"{name} files: {n}")
    if problems:
        raise SystemExit(f"thesis inspect: invalid files: {', '.join(problems)}")


def _thesis_inbox(args: argparse.Namespace) -> None:
    repo = _thesis_repo(args)
    thesis = _thesis_load(repo, str(args.id))
    triggers = repo.load_triggers(thesis.thesis_id)
    if not triggers:
        print("No triggers.")
        return
    for t in triggers:
        print(f"{t.trigger_id} [{t.status}] {t.trigger_type} ({t.importance}) {t.created_at}")
        if t.summary:
            print(f"  {t.summary.splitlines()[0][:120]}")


def _journal_head(path: Path) -> tuple[str, str, str]:
    entry_id, created, title = path.stem, "", ""
    try:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i > 15:
                    break
                s = line.strip()
                if s.startswith("entry_id:"):
                    entry_id = s.split(":", 1)[1].strip()
                elif s.startswith("created_at:"):
                    created = s.split(":", 1)[1].strip()
                elif s.startswith("# "):
                    title = s[2:].strip()
    except OSError:
        pass
    return entry_id, created, title


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def _list_journal_files(jdir: Path) -> list[Path]:
    if not jdir.is_dir():
        return []
    return sorted((p for p in jdir.glob("*.md") if p.is_file()), key=_mtime, reverse=True)


def _journal_entry_matches(path: Path, norm: str, sel: str) -> bool:
    if path.stem == norm:
        return True
    if norm in path.stem:
        return True
    return sel in path.stem


def _match_journal_entry(files: list[Path], sel: str) -> Path | None:
    norm = sel.replace(":", "_")  # filenames store entry IDs with ':' -> '_'
    for path in files:
        if _journal_entry_matches(path, norm, sel):
            return path
    return None


def _print_journal_index(files: list[Path]) -> None:
    for path in files:
        entry_id, created, title = _journal_head(path)
        print(f"{entry_id} {created} {title}".rstrip())


def _thesis_journal(args: argparse.Namespace) -> None:
    repo = _thesis_repo(args)
    thesis = _thesis_load(repo, str(args.id))
    jdir = repo.root / thesis.slug / "journal"
    files = _list_journal_files(jdir)
    sel_raw = getattr(args, "entry", None)
    sel = sel_raw if isinstance(sel_raw, str) else None
    if sel:
        match = _match_journal_entry(files, sel)
        if match is None:
            raise SystemExit(f"thesis journal: no entry matching {sel!r} ({len(files)} entries)")
        print(match.read_text(encoding="utf-8"), end="")
        return
    if not files:
        print("No journal entries.")
        return
    _print_journal_index(files)


def _thesis_runtime(args: argparse.Namespace, what: str = "thesis tick"):
    """Shared tick/monitor wiring: repo, thesis ID, source services."""
    from app.thesis import monitor

    repo = _thesis_repo(args)
    thesis = _thesis_load(repo, str(args.id))
    targets = monitor.targets_for_thesis(thesis)
    services = {
        "sec_filings": monitor.SecFilingsService(targets, since_default=thesis.created_at),
        "material_events": monitor.MaterialEventsService(targets, since_default=thesis.created_at),
        "finra_short_interest": monitor.FinraShortInterestService(targets),
    }
    return repo, thesis.thesis_id, services


def _format_tick_no_op(no_op_reason: str | None) -> str:
    return "no meaningful change" + (f": {no_op_reason}" if no_op_reason else "")


def _print_tick_outcome(triggers_created: list[str], runs: object, *, flush: bool = False) -> None:
    items: list[object] = list(runs) if isinstance(runs, list) else []
    run_ids = [r.run_id for r in items if isinstance(r, _HasRunId)]
    print(
        f"triggers created: {len(triggers_created)}"
        + (f" ({', '.join(triggers_created)})" if triggers_created else ""),
        flush=flush,
    )
    print(f"runs: {len(run_ids)}" + (f" ({', '.join(run_ids)})" if run_ids else ""), flush=flush)


def _resolve_tick_known_at(args: argparse.Namespace) -> str:
    from datetime import datetime

    return getattr(args, "known_at", None) or datetime.now(UTC).isoformat(timespec="seconds")


def _thesis_tick(args: argparse.Namespace) -> None:
    """One deterministic monitor tick; exit 0 always (pauses/no-ops print a message)."""
    from app.thesis import monitor

    repo, thesis_id, services = _thesis_runtime(args)
    result = monitor.tick(repo, thesis_id, services, known_at=_resolve_tick_known_at(args))
    if result.no_op:
        print(_format_tick_no_op(result.no_op_reason))
        return
    _print_tick_outcome(result.triggers_created, result.runs)


def _validate_monitor_interval(interval: int) -> None:
    if interval <= 0:
        print("thesis monitor: --interval-seconds must be > 0", file=sys.stderr)
        raise SystemExit(2)


def _report_tick_outcome(outcome: TickResult) -> None:
    if outcome.no_op:
        print(_format_tick_no_op(outcome.no_op_reason), flush=True)
        return
    _print_tick_outcome(outcome.triggers_created, outcome.runs, flush=True)


def _thesis_monitor(args: argparse.Namespace) -> None:
    """Loop ticks until stopped or closed; paused sleeps without querying, closed exits 0."""
    import signal
    import threading

    from app.thesis.worker import monitor_loop

    _validate_monitor_interval(args.interval_seconds)
    repo, thesis_id, services = _thesis_runtime(args, what="thesis monitor")
    fixed_known_at = getattr(args, "known_at", None)
    stop = threading.Event()

    def _stop(signum: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    monitor_loop(
        repository=repo,
        thesis_id=thesis_id,
        interval_seconds=args.interval_seconds,
        source_services=services,
        known_at_fn=(lambda: fixed_known_at) if fixed_known_at else None,
        stop_event=stop,
        on_tick=_report_tick_outcome,
    )


def _cmd_thesis(args: argparse.Namespace) -> None:
    cmd = getattr(args, "thesis_command", None)
    if cmd == "list":
        _thesis_list(args)
    elif cmd == "show":
        _thesis_show(args)
    elif cmd in ("pause", "resume", "close"):
        _thesis_status(args)
    elif cmd == "inspect":
        _thesis_inspect(args)
    elif cmd == "inbox":
        _thesis_inbox(args)
    elif cmd == "journal":
        _thesis_journal(args)
    elif cmd == "tick":
        _thesis_tick(args)
    elif cmd == "monitor":
        _thesis_monitor(args)
    else:
        raise SystemExit("thesis: choose from list, show, pause, resume, close, inspect, inbox, journal, tick, monitor")


def _google_geos(args: argparse.Namespace) -> list[str]:
    if args.geo:
        return list(args.geo)
    return ["US"]


def _google_limit(args: argparse.Namespace) -> int:
    raw = args.limit or 25
    return max(1, min(raw, 1000))


def _google_sources(args: argparse.Namespace) -> tuple[str, ...]:
    if args.source == "all":
        return ("trends", "patents", "macro", "geo", "stackoverflow")
    return (args.source,)


def _collect_google_trends(args: argparse.Namespace, geos: list[str], limit: int):
    from app.google_data import trends as _trends

    return _trends.collect_trends(
        start_date=args.start_date,
        end_date=args.end_date,
        geos=list(geos),
        limit=limit,
        data_root=args.data_root or None,
    )


def _collect_google_patents(args: argparse.Namespace, limit: int):
    if not args.company:
        return {
            "status": "error",
            "source": "patents",
            "error": "company id required (--company) for patents collection",
        }
    from app.google_data import patents as _patents

    return _patents.search_company_patents(
        args.company, start_date=args.start_date, end_date=args.end_date, limit=min(limit, 20)
    )


def _collect_google_macro(args: argparse.Namespace, geos: list[str], limit: int):
    from app.google_data import datacommons as _dc

    variables = list(args.variable or [])
    return _dc.get_macro_context(
        list(geos), variables, start_date=args.start_date, end_date=args.end_date, limit=min(limit, 100)
    )


def _collect_google_geo(args: argparse.Namespace, geos: list[str], limit: int):
    from app.google_data import geo_context as _geo

    return _geo.get_geo_context(
        list(geos),
        variables=list(args.variable or []),
        start_date=args.start_date,
        end_date=args.end_date,
        limit=min(limit, 100),
    )


def _collect_google_stackoverflow(args: argparse.Namespace, limit: int):
    from app.google_data import stackoverflow as _so

    return _so.get_tag_activity(
        list(args.tag or []), start_date=args.start_date, end_date=args.end_date, limit=min(limit, 100)
    )


def _collect_google_source(src: str, args: argparse.Namespace, geos: list[str], limit: int):
    if src == "trends":
        return _collect_google_trends(args, geos, limit)
    elif src == "patents":
        return _collect_google_patents(args, limit)
    elif src == "macro":
        return _collect_google_macro(args, geos, limit)
    elif src == "geo":
        return _collect_google_geo(args, geos, limit)
    else:
        return _collect_google_stackoverflow(args, limit)


def _cmd_google_data(args: argparse.Namespace) -> None:
    """Manual Google public-data collection; per-source status, never raises."""
    if getattr(args, "google_data_command", None) != "collect":
        raise SystemExit("google-data: choose 'collect' (e.g. google-data collect --source trends ...)")
    geos = _google_geos(args)
    limit = _google_limit(args)
    out: dict[str, object] = {}
    for src in _google_sources(args):
        try:
            out[src] = _collect_google_source(src, args, geos, limit)
        except Exception as exc:  # noqa: BLE001 - per-source status capture, never raises
            out[src] = {"status": "error", "source": src, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps({"sources": out}, indent=2, default=str))


def _research_create(args: argparse.Namespace, service: ModuleType) -> None:
    from app.research.models import default_policy

    policy = default_policy()
    if args.interrupt_after is not None:
        # ponytail: test hook recorded on the session; the wave runner honors it when it lands.
        policy["interrupt_after"] = args.interrupt_after
    try:
        session_id = service.create_research(
            args.question, args.objective or args.question, as_of=args.as_of, policy=policy
        )
        state = service.inspect_research(session_id)
    except ValueError as exc:
        raise SystemExit(f"research: {exc}") from None
    session = state["session"]
    assert isinstance(session, dict)
    jobs = state["jobs"]
    assert isinstance(jobs, list)
    first = jobs[0] if jobs else None
    job_id = first.get("job_id") if isinstance(first, dict) else None
    print(f"{session_id} job={job_id} status={session.get('status')}")


def _format_research_inspect(state: dict[str, object], resumed: dict[str, object]) -> list[str]:
    session = state["session"]
    assert isinstance(session, dict)
    jobs = state["jobs"]
    assert isinstance(jobs, list)
    evidence_ids = session.get("evidence_ids")
    freeze_ids = session.get("freeze_ids")
    dossier_ids = session.get("dossier_ids")
    budgets = resumed.get("budgets")
    open_ids = resumed.get("open_job_ids")
    assert isinstance(open_ids, list)
    evidence_count = len(evidence_ids) if isinstance(evidence_ids, list) else 0
    freeze_count = len(freeze_ids) if isinstance(freeze_ids, list) else 0
    dossier_count = len(dossier_ids) if isinstance(dossier_ids, list) else 0
    budgets_text = json.dumps(budgets, sort_keys=True) if isinstance(budgets, dict) else "{}"
    open_text = ", ".join(str(i) for i in open_ids) if open_ids else "-"
    return [
        f"{session.get('session_id')} status={session.get('status')} wave={session.get('current_wave')}",
        f"query: {session.get('query')}",
        f"jobs={len(jobs)} evidence={evidence_count} freezes={freeze_count} dossiers={dossier_count}",
        f"pending: {state.get('pending_next_action')}",
        f"budgets: {budgets_text}",
        f"open jobs: {open_text}",
    ]


def _research_inspect(args: argparse.Namespace, service: ModuleType) -> None:
    try:
        state = service.inspect_research(args.session_id)
        resumed = service.resume_research(args.session_id)
    except service.ResearchNotFound:
        raise SystemExit(f"research: unknown session {args.session_id}") from None
    for line in _format_research_inspect(state, resumed):
        print(line)


def _research_list(args: argparse.Namespace, service: ModuleType) -> None:
    rows = service.list_research(limit=args.limit)
    if not rows:
        print("No research sessions recorded.")
        return
    for row in rows:
        print(f"{row['session_id']} status={row['status']} {str(row['updated_at'])[:26]} {row['query']}")


def _research_cancel(args: argparse.Namespace, service: ModuleType) -> None:
    try:
        out = service.cancel_research(args.session_id)
    except service.ResearchNotFound:
        raise SystemExit(f"research: unknown session {args.session_id}") from None
    print(f"{out.get('session_id')} status={out.get('status')}")


def _research_retry(args: argparse.Namespace, service: ModuleType) -> None:
    try:
        job = service.retry_job(args.job_id)
    except service.ResearchNotFound:
        raise SystemExit(f"research: unknown job {args.job_id}") from None
    print(f"{job.get('job_id')} status={job.get('status')} session={job.get('session_id')}")


def _cmd_research(args: argparse.Namespace) -> None:
    """Research sessions: state-only admin over the kernel service (no models here)."""
    from app.research import service as _service

    sub = getattr(args, "research_command", None)
    if sub == "create":
        _research_create(args, _service)
    elif sub == "inspect":
        _research_inspect(args, _service)
    elif sub == "list":
        _research_list(args, _service)
    elif sub == "cancel":
        _research_cancel(args, _service)
    elif sub == "retry":
        _research_retry(args, _service)
    else:
        raise SystemExit("research: choose from create, inspect, list, cancel, retry")


def _print_trace_header(header: object) -> None:
    trace_id = getattr(header, "trace_id", None)
    wave_id = getattr(header, "wave_id", None)
    model = getattr(header, "model", None)
    status = getattr(header, "status", None)
    if not isinstance(trace_id, str) or not isinstance(wave_id, int):
        raise TypeError(f"trace header must have trace_id str and wave_id int, got {header!r}")
    if not isinstance(model, str) or not isinstance(status, str):
        raise TypeError(f"trace header must have model str and status str, got {header!r}")
    print(f"{trace_id} wave={wave_id} model={model} status={status}")


def _trace_events_of(header: object, get_trace_events: Callable[..., object]) -> list[object] | None:
    """Fetch events for one trace header; None when the fetch fails."""
    trace_id = getattr(header, "trace_id", None)
    if not isinstance(trace_id, str):
        raise TypeError(f"trace header must have trace_id str, got {header!r}")
    try:
        events = get_trace_events(trace_id)
    except Exception as exc:  # noqa: BLE001 - trace fetch is best-effort, prints unavailable
        print(f"  events: unavailable ({exc})")
        return None
    return events if isinstance(events, list) else None


def _print_trace_event(evt: object) -> None:
    """One trace event line; non-conforming events print repr."""
    seq = getattr(evt, "seq", None)
    event_type = getattr(evt, "event_type", None)
    duration_ms = getattr(evt, "duration_ms", None)
    if (
        isinstance(seq, int)
        and isinstance(event_type, str)
        and (duration_ms is None or isinstance(duration_ms, (int, float)))
    ):
        print(f"    {seq} {event_type} {duration_ms}")
    else:
        print(f"    {evt!r}")


def _print_trace_events(header: object, get_trace_events: Callable[..., object]) -> None:
    events = _trace_events_of(header, get_trace_events)
    if events is None:
        print("  events: unavailable (invalid response shape)")
        return
    print(f"  events={len(events)}")
    for evt in events[:50]:
        _print_trace_event(evt)
    if len(events) > 50:
        print(f"    ... {len(events) - 50} more")


def _cmd_trace(session_id: str) -> None:
    """Show persisted traces for a research session (read-only)."""
    from app.research.evals.traces import get_trace_events, list_traces

    headers = list_traces(session_id)
    if not headers:
        print(f"no traces for session {session_id}")
        return
    for header in headers:
        _print_trace_header(header)
        _print_trace_events(header, get_trace_events)


def _herdr_raw_logs(args: argparse.Namespace) -> None:
    from app.services.herdr_client import HerdrClient

    try:
        print(HerdrClient().pane_read(args.pane, source=args.source, lines=args.lines), end="")
    except (ConnectionError, FileNotFoundError) as exc:
        raise SystemExit(f"herdr: cannot reach Herdr socket ({exc})") from None
    except KeyError as exc:
        raise SystemExit(f"herdr: unexpected response shape (missing {exc})") from None


def _cmd_herdr(args: argparse.Namespace) -> None:
    """Live Herdr reads (read-only; WorkerId->pane map stays local)."""
    sub = getattr(args, "herdr_command", None)
    if sub == "raw-logs":
        _herdr_raw_logs(args)
    else:
        raise SystemExit("herdr: choose from raw-logs")


def _eval_run_suite(args: argparse.Namespace) -> None:
    from app.research.evals.evaluators import outcomes_from_fixtures, run_eval_suite

    names = [args.scenario] if args.scenario else None
    outcomes = outcomes_from_fixtures(names)
    summary = run_eval_suite(
        model=args.model, provider=args.provider, prompt_version=args.prompt_version, outcomes=outcomes
    )
    print(
        f"eval run {summary.eval_run_id}:"
        f" {summary.passed_count}/{summary.scenario_count} passed (model={summary.model})"
    )
    if summary.failed_count:
        raise SystemExit(1)


def _eval_inspect(args: argparse.Namespace) -> None:
    from app.research.evals.evaluators import get_eval_results, get_eval_run

    header = get_eval_run(args.eval_run_id)
    if header is None:
        raise SystemExit(f"eval: unknown run {args.eval_run_id}")
    print(
        f"eval run {header.eval_run_id} model={header.model} provider={header.provider}"
        f" harness={header.harness_version} prompt={header.prompt_version}"
        f" git={header.git_sha} scenarios={header.scenario_version} at={header.started_at}"
    )
    for row in get_eval_results(args.eval_run_id):
        verdict = "PASS" if row.passed else f"FAIL {','.join(row.violations)}"
        print(f"  {verdict} {row.scenario_name}")


def _eval_promote(args: argparse.Namespace) -> None:
    from app.research.evals.regression import promote_to_fixture

    path = promote_to_fixture(
        session_id=args.session_id, scenario_name=args.scenario_name, question=args.question, as_of=args.as_of
    )
    print(str(path))


def _cmd_eval(args: argparse.Namespace) -> None:
    """Deterministic agent-scenario evals (own store: data/eval_runs.sqlite)."""
    sub = getattr(args, "eval_command", None)
    if sub in ("run", "model"):
        _eval_run_suite(args)
    elif sub == "inspect":
        _eval_inspect(args)
    elif sub == "promote":
        _eval_promote(args)
    else:
        raise SystemExit("eval: choose from run, model, inspect, promote")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stockbot — AI investment research assistant")
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--log-server",
        nargs="?",
        const=_LOG_SERVER_DEFAULT_URL,
        help="stream all logs to this log server URL (default: http://127.0.0.1:8765)",
    )

    runs_parser = subparsers.add_parser("runs", help="list recent runs")
    runs_parser.add_argument("--limit", type=int, default=20, help="max rows (default 20)")
    inspect_parser = subparsers.add_parser("inspect", help="show one run's record")
    inspect_parser.add_argument("run_id", help="run id, e.g. run:20260829T123456789012")
    refresh_parser = subparsers.add_parser(
        "refresh-data", help="read the live FINRA/SEC leaderboard for one settlement cycle (ephemeral, never persisted)"
    )
    refresh_parser.add_argument("--settlement-date", required=True, help="FINRA settlement date YYYY-MM-DD")
    refresh_parser.add_argument(
        "--ticker", action="append", default=[], help="accepted and ignored (live reads need no enrichment)"
    )
    refresh_parser.add_argument(
        "--cik", type=int, action="append", default=[], help="accepted and ignored (live reads need no enrichment)"
    )
    refresh_parser.add_argument("--data-root", default=None, help="accepted and ignored (live reads need no data root)")
    subparsers.add_parser("replay-sec-facts", help="removed: SEC facts read live via providers; no replay")
    obligations_parser = subparsers.add_parser(
        "refresh-obligations", help="evaluate obligations for a ticker in memory (never persisted)"
    )
    obligations_parser.add_argument("ticker", help="ticker, e.g. NVDA")
    mandate_parser = subparsers.add_parser(
        "evaluate-mandate", help="evaluate the mandate JSON against the latest portfolio snapshot"
    )
    mandate_parser.add_argument("--data-root", default=None, help="data root directory (default: repo data/)")
    log_server_parser = subparsers.add_parser(
        "log-server", help="receive and print log lines from CLI/Pi-bridge clients (Ctrl-C to stop)"
    )
    log_server_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_LOG_SERVER_PORT,
        help=f"port to listen on (default {DEFAULT_LOG_SERVER_PORT})",
    )
    subparsers.add_parser("robinhood-login", help="authorize Robinhood OAuth deliberately (opens browser)")
    backfill_parser = subparsers.add_parser(
        "backfill-sec",
        help="enqueue bounded SEC quarterly/form backfill jobs, then drain inline (dates required; no all-history default)",
    )
    backfill_parser.add_argument("--source", default="sec-global", help="coverage source (default sec-global)")
    backfill_parser.add_argument(
        "--form", action="append", default=[], help="SEC form, e.g. 10-K (repeatable; required)"
    )
    backfill_parser.add_argument("--from", dest="from_date", required=True, help="range start YYYY-MM-DD (required)")
    backfill_parser.add_argument("--to", dest="to_date", required=True, help="range end YYYY-MM-DD (required)")
    backfill_parser.add_argument("--batch-size", type=int, default=50, help="filings per job batch (default 50)")
    backfill_parser.add_argument("--data-root", default=None, help="data root directory (default: repo data/)")
    resume_parser = subparsers.add_parser(
        "resume-sec-backfill", help="requeue interrupted SEC backfill jobs and drain the queue inline"
    )
    resume_parser.add_argument(
        "job_id", nargs="?", default=None, help="one job ID to resume (default: all queued/failed)"
    )
    resume_parser.add_argument("--data-root", default=None, help="data root directory (default: repo data/)")
    coverage_parser = subparsers.add_parser(
        "sec-coverage", help="show SEC ingestion coverage plus pending backfill jobs"
    )
    coverage_parser.add_argument("--source", default=None, help="filter by coverage source")
    coverage_parser.add_argument("--form", default=None, help="filter by SEC form")
    coverage_parser.add_argument("--from", dest="from_date", default=None, help="coverage on/after YYYY-MM-DD")
    coverage_parser.add_argument("--to", dest="to_date", default=None, help="coverage on/before YYYY-MM-DD")
    coverage_parser.add_argument("--data-root", default=None, help="data root directory (default: repo data/)")
    gd_parser = subparsers.add_parser(
        "google-data", help="manual Google public-data collection (optional; never affects SEC/FINRA)"
    )
    gd_sub = gd_parser.add_subparsers(dest="google_data_command")
    gd_collect = gd_sub.add_parser("collect", help="collect one Google source (manual only)")
    gd_collect.add_argument(
        "--source",
        required=True,
        choices=["trends", "patents", "macro", "geo", "stackoverflow", "all"],
        help="source to collect",
    )
    gd_collect.add_argument("--start-date", default=None, help="range start YYYY-MM-DD")
    gd_collect.add_argument("--end-date", default=None, help="range end YYYY-MM-DD")
    gd_collect.add_argument("--geo", action="append", default=None, help="geography, e.g. US (repeatable; default US)")
    gd_collect.add_argument("--company", default=None, help="documented assignee for --source patents")
    gd_collect.add_argument(
        "--variable",
        action="append",
        default=[],
        help="Data Commons variable ID for --source macro, census column or weather hint for --source geo (repeatable)",
    )
    gd_collect.add_argument(
        "--tag", action="append", default=[], help="Stack Overflow tag for --source stackoverflow (repeatable)"
    )
    gd_collect.add_argument("--limit", type=int, default=25, help="max rows (default 25, capped at 1000)")
    gd_collect.add_argument("--data-root", default=None, help="data root directory (default: repo data/)")
    thesis_common = argparse.ArgumentParser(add_help=False)
    thesis_common.add_argument(
        "--data-root", default=argparse.SUPPRESS, help="data root directory (default: $STOCKBOT_DATA_DIR or repo data/)"
    )
    thesis_parser = subparsers.add_parser("thesis", parents=[thesis_common], help="persistent thesis management")
    thesis_sub = thesis_parser.add_subparsers(dest="thesis_command")
    thesis_sub.add_parser("list", parents=[thesis_common], help="list theses")
    show_parser = thesis_sub.add_parser("show", parents=[thesis_common], help="show thesis and assessment")
    show_parser.add_argument("id", help="thesis ID or slug")
    for _name in ("pause", "resume", "close"):
        _p = thesis_sub.add_parser(_name, parents=[thesis_common], help=f"{_name} a thesis")
        _p.add_argument("id", help="thesis ID or slug")
    inspect_parser = thesis_sub.add_parser(
        "inspect", parents=[thesis_common], help="validate thesis files and show counts"
    )
    inspect_parser.add_argument("id", help="thesis ID or slug")
    inbox_parser = thesis_sub.add_parser("inbox", parents=[thesis_common], help="list triggers and their state")
    inbox_parser.add_argument("id", help="thesis ID or slug")
    journal_parser = thesis_sub.add_parser(
        "journal", parents=[thesis_common], help="list journal entries (newest first)"
    )
    journal_parser.add_argument("id", help="thesis ID or slug")
    journal_parser.add_argument("entry", nargs="?", default=None, help="print one entry without loading all")
    tick_parser = thesis_sub.add_parser("tick", parents=[thesis_common], help="run one deterministic monitor tick")
    tick_parser.add_argument("id", help="thesis ID or slug")
    tick_parser.add_argument("--known-at", default=None, help="PIT upper bound ISO timestamp (default: now UTC)")
    monitor_parser = thesis_sub.add_parser(
        "monitor",
        parents=[thesis_common],
        help="loop ticks until stopped; paused sleeps without querying, closed exits 0",
    )
    monitor_parser.add_argument("id", help="thesis ID or slug")
    monitor_parser.add_argument(
        "--interval-seconds", type=int, default=900, help="seconds between ticks (default 900; must be > 0)"
    )
    monitor_parser.add_argument(
        "--known-at", default=None, help="PIT upper bound ISO timestamp (default: now UTC per tick)"
    )
    research_parser = subparsers.add_parser(
        "research", help="create/inspect/list/cancel/retry a research session (state only)"
    )
    research_sub = research_parser.add_subparsers(dest="research_command")
    research_create = research_sub.add_parser("create", help="create a session and enqueue its first job")
    research_create.add_argument("--question", required=True, help="research question")
    research_create.add_argument("--objective", default=None, help="objective (default: question)")
    research_create.add_argument("--as-of", default=None, help="PIT upper bound ISO timestamp")
    research_create.add_argument(
        "--interrupt-after",
        default=None,
        choices=["source", "freeze", "one-committee"],
        help="test hook recorded on the session so resume can be exercised",
    )
    research_inspect = research_sub.add_parser("inspect", help="show a research session")
    research_inspect.add_argument("session_id", help="session id")
    research_list = research_sub.add_parser("list", help="list recent research sessions")
    research_list.add_argument("--limit", type=int, default=20, help="max rows (default 20)")
    research_cancel = research_sub.add_parser("cancel", help="cancel a research session")
    research_cancel.add_argument("session_id", help="session id")
    research_retry = research_sub.add_parser("retry", help="enqueue a replacement for a failed job")
    research_retry.add_argument("job_id", help="job id")

    trace_parser = subparsers.add_parser("trace", help="show persisted traces for a research session")
    trace_parser.add_argument("session_id", help="research session id")
    herdr_parser = subparsers.add_parser("herdr", help="live Herdr pane output (read-only)")
    herdr_sub = herdr_parser.add_subparsers(dest="herdr_command")
    raw_logs = herdr_sub.add_parser("raw-logs", help="print a pane's recent output (RAW LOGS)")
    raw_logs.add_argument("pane", help="pane id, e.g. wC:p1")
    raw_logs.add_argument("--lines", type=int, default=200, help="max lines (default 200)")
    raw_logs.add_argument(
        "--source",
        default="recent",
        choices=["visible", "recent", "recent-unwrapped"],
        help="terminal snapshot source (default: recent)",
    )

    eval_common = argparse.ArgumentParser(add_help=False)
    eval_common.add_argument("--scenario", default=None, help="one scenario (default: all)")
    eval_common.add_argument("--provider", default="unknown", help="provider label")
    eval_common.add_argument("--prompt-version", default="v1", help="prompt version stamp")
    eval_parser = subparsers.add_parser("eval", help="deterministic agent-scenario evals")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")
    eval_run = eval_sub.add_parser("run", parents=[eval_common], help="run evals")
    eval_run.add_argument("--model", default="deterministic", help="model label")
    eval_model = eval_sub.add_parser("model", parents=[eval_common], help="labelled multi-model eval run")
    eval_model.add_argument("--model", required=True, help="model label, e.g. granite-4.1-8b")
    eval_inspect = eval_sub.add_parser("inspect", help="show one eval run")
    eval_inspect.add_argument("eval_run_id", help="eval run id, e.g. eval:abc123")
    eval_promote = eval_sub.add_parser("promote", help="promote a session to a regression fixture")
    eval_promote.add_argument("session_id", help="session id")
    eval_promote.add_argument("--scenario-name", required=True, help="scenario, e.g. factual-nvda-datacenter-growth")
    eval_promote.add_argument("--question", default=None, help="override the scenario question")
    eval_promote.add_argument("--as-of", default=None, help="override the scenario as_of")
    return parser


def _rewrite_bare_log_server(argv: list[str]) -> list[str]:
    """A bare --log-server directly before the subcommand (cli.py --log-server runs)
    would be consumed by nargs='?' as its value; rewrite it to the explicit default
    URL so the subcommand still parses and dispatches."""
    for i, arg in enumerate(argv[:-1]):
        if arg == "--log-server" and argv[i + 1] in _SUBCOMMANDS:
            return argv[:i] + [f"--log-server={_LOG_SERVER_DEFAULT_URL}"] + argv[i + 1 :]
    return argv


def _resolve_stream_url(args: argparse.Namespace) -> str | None:
    if args.command == "log-server":
        return None
    return args.log_server or os.getenv("STOCKBOT_LOG_SERVER") or None


def _run_runs(args: argparse.Namespace) -> None:
    _cmd_runs(args.limit)


def _run_robinhood_login(args: argparse.Namespace) -> None:
    _cmd_robinhood_login()


def _run_inspect(args: argparse.Namespace) -> None:
    _cmd_inspect(args.run_id)


def _run_refresh_data(args: argparse.Namespace) -> None:
    _cmd_refresh_data(args.settlement_date, args.ticker, args.cik, args.data_root or None)


def _run_replay_sec_facts(args: argparse.Namespace) -> None:
    _cmd_replay_sec_facts()


def _run_refresh_obligations(args: argparse.Namespace) -> None:
    _cmd_refresh_obligations(args.ticker)


def _run_evaluate_mandate(args: argparse.Namespace) -> None:
    data_root = args.data_root or None
    from app.config import get_data_root

    mandate_path = Path(args.mandate) if args.mandate else get_data_root() / "mandate.json"
    _cmd_evaluate_mandate(mandate_path, data_root)


def _run_log_server(args: argparse.Namespace) -> None:
    _cmd_log_server(args.port)


def _run_backfill_sec(args: argparse.Namespace) -> None:
    _cmd_backfill_sec(args.source, args.form, args.from_date, args.to_date, args.batch_size, args.data_root or None)


def _run_resume_sec_backfill(args: argparse.Namespace) -> None:
    _cmd_resume_sec_backfill(args.job_id, args.data_root or None)


def _run_sec_coverage(args: argparse.Namespace) -> None:
    _cmd_sec_coverage(args.source, args.form, args.from_date, args.to_date, args.data_root or None)


def _run_trace(args: argparse.Namespace) -> None:
    _cmd_trace(args.session_id)


_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "runs": _run_runs,
    "robinhood-login": _run_robinhood_login,
    "inspect": _run_inspect,
    "refresh-data": _run_refresh_data,
    "replay-sec-facts": _run_replay_sec_facts,
    "refresh-obligations": _run_refresh_obligations,
    "evaluate-mandate": _run_evaluate_mandate,
    "log-server": _run_log_server,
    "backfill-sec": _run_backfill_sec,
    "resume-sec-backfill": _run_resume_sec_backfill,
    "sec-coverage": _run_sec_coverage,
    "thesis": _cmd_thesis,
    "google-data": _cmd_google_data,
    "research": _cmd_research,
    "trace": _run_trace,
    "herdr": _cmd_herdr,
    "eval": _cmd_eval,
}


def _dispatch_command(args: argparse.Namespace) -> bool:
    """Run the subcommand handler; False when the command is unknown."""
    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        return False
    handler(args)
    return True


def _unknown_command_error() -> str:
    return (
        "unknown command (choose from runs, inspect, refresh-data, replay-sec-facts, "
        "refresh-obligations, evaluate-mandate, log-server, robinhood-login, "
        "backfill-sec, resume-sec-backfill, sec-coverage, thesis, google-data, research, trace, herdr, eval)"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args(_rewrite_bare_log_server(sys.argv[1:]))
    configure_logging(stream_url=_resolve_stream_url(args))
    if not _dispatch_command(args):
        parser.error(_unknown_command_error())


if __name__ == "__main__":
    main()
