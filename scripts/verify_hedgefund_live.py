#!/usr/bin/env python3
"""Opt-in live golden scenario: real SEC + FINRA + Exa only (never CI).

Runs one golden question through the production kernel path and exports the
read-only web projection. Requires reachable credentials (same prerequisites
as scripts/verify_agent_scenarios.py):

This script performs real network calls and is NEVER run by CI. Invoke only:
  HEDGEFUND_LIVE=1 venv/bin/python scripts/verify_hedgefund_live.py [--scenario NAME] [--json]
Without HEDGEFUND_LIVE=1 it exits 2 with an explicit opt-in message.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.research.models import Job, ResearchSession
    from app.research.repository import ResearchRepository

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.research.evidence import evidence_domain

GOLDEN_SCENARIO = "hedgefund-growth-thesis-multisource"

REQUIRED_ENV = ("SEC_EDGAR_IDENTITY", "FINRA_CLIENT_ID", "FINRA_CLIENT_SECRET", "EXA_API_KEY")

GOLDEN_DOMAINS = ("SEC", "FINRA", "WEB")
GOLDEN_COMMITTEE_ROLES = ("stockbot", "bullbot", "bearbot")


def _missing_env() -> list[str]:
    missing = [name for name in REQUIRED_ENV if not (os.getenv(name) or "").strip()]
    if (os.getenv("EXA_ENABLED") or "").strip().lower() not in ("1", "true", "yes"):
        missing.append("EXA_ENABLED=1")
    return missing


def _golden_db_path() -> Path:
    from app.config import get_data_root

    override = (os.getenv("RESEARCH_DB_PATH") or "").strip()
    if override:
        return Path(override)
    return get_data_root() / "research.sqlite"


def _str_list(raw: object) -> list[str]:
    return [v for v in raw if isinstance(v, str)] if isinstance(raw, (list, tuple)) else []


def _evidence_domain(row: object) -> str:
    get = row.get if isinstance(row, dict) else getattr(row, "get", None)
    raw = get("domain", get("source_domain", None)) if callable(get) else None
    if isinstance(raw, str) and raw.strip().upper() in GOLDEN_DOMAINS:
        return raw.strip().upper()
    raw_prov: object = get("provenance", {}) if callable(get) else {}
    prov: dict[str, object] = raw_prov if isinstance(raw_prov, dict) else {}
    domain = evidence_domain(prov if isinstance(prov, dict) else None)
    return domain if domain != "SOURCE" else "SEC"


def _frozen_rows(rows: object, freeze_ids: set[str]) -> Iterator[tuple[object, Callable[..., object]]]:
    """Yield each frozen evidence record with its field reader; skip every other row."""
    seq = rows if isinstance(rows, (list, tuple)) else []
    for row in seq:
        get = row.get if isinstance(row, dict) else getattr(row, "get", None)
        if not callable(get):
            continue
        if get("evidence_id") in freeze_ids and get("record_kind") != "discovery":
            yield row, get


def _freeze_domains(rows: object, freeze_ids: set[str]) -> set[str]:
    return {_evidence_domain(row) for row, _get in _frozen_rows(rows, freeze_ids)}


def _job_role_map(jobs: object) -> dict[str, str]:
    by_id: dict[str, str] = {}
    seq = jobs if isinstance(jobs, (list, tuple)) else []
    for job in seq:
        if isinstance(job, dict):
            jid, role = job.get("job_id"), job.get("job_type")
        else:
            jid, role = getattr(job, "job_id", None), getattr(job, "job_type", None)
        if isinstance(jid, str) and isinstance(role, str):
            by_id[jid] = role
    return by_id


def _committee_roles_match(job_ids: list[object], by_id: dict[str, str]) -> bool:
    """True when the run holds exactly the golden committee roles, ids all strings."""
    ids = [j for j in job_ids if isinstance(j, str)]
    return len(ids) == len(job_ids) and {by_id.get(j) for j in ids} == set(GOLDEN_COMMITTEE_ROLES)


def _trio_job_ids(entry: object) -> list[object] | None:
    """The three job ids of one committee run; None for any other shape."""
    if not isinstance(entry, dict):
        return None
    job_ids = entry.get("jobs")
    return job_ids if isinstance(job_ids, list) and len(job_ids) == 3 else None


def _trio_freezes(runs: object, jobs: object = None) -> list[str]:
    out: list[str] = []
    seq = runs if isinstance(runs, (list, tuple)) else []
    by_id = None if jobs is None else _job_role_map(jobs)
    for entry in seq:
        fid = entry.get("freeze_id") if isinstance(entry, dict) else None
        job_ids = _trio_job_ids(entry)
        if job_ids is None or not isinstance(fid, str):
            continue
        if by_id is None or _committee_roles_match(job_ids, by_id):
            out.append(fid)
    return out


def _frozen_ids(repo: ResearchRepository, freeze_ids: list[str]) -> set[str]:
    """Read the union of session freezes, ignoring missing freeze records."""
    frozen: set[str] = set()
    for fid in freeze_ids:
        try:
            frozen |= set(_str_list(repo.get_freeze(fid).get("evidence_ids")))
        except KeyError:
            continue
    return frozen


def _domain_checks(jobs: list[Job], freeze_domains: set[str], failures: list[str]) -> set[str]:
    """Check source execution and frozen source diversity in that order."""
    job_domains = {str(j.source_domain).upper() for j in jobs if j.source_domain}
    domains = job_domains | freeze_domains if job_domains != {"SEC"} else freeze_domains or {"SEC"}
    if not set(GOLDEN_DOMAINS) <= domains:
        failures.append(f"source domains ran {sorted(domains)}; want {list(GOLDEN_DOMAINS)}")
    if not set(GOLDEN_DOMAINS) <= freeze_domains:
        failures.append(f"freeze holds {sorted(freeze_domains)}; want {list(GOLDEN_DOMAINS)}")
    return domains


def _trio_roles(runs: Sequence[object], jobs: object) -> set[str | None]:
    """Report the roles attached to each three-job committee run."""
    by_id = _job_role_map(jobs)
    seen: set[str | None] = set()
    for entry in runs:
        job_ids = _trio_job_ids(entry)
        if job_ids is None:
            continue
        seen.update(by_id.get(j) if isinstance(j, str) else None for j in job_ids)
    return seen


def _committee_checks(sess: ResearchSession, jobs: list[Job], freeze_ids: list[str], failures: list[str]) -> list[str]:
    """Check the role trio and its membership in the session freezes."""
    trio = _trio_freezes(sess.committee_runs, jobs)
    if not trio or len(set(trio)) != 1:
        msg = f"trio shares no single freeze (runs={trio})"
        if not trio and _trio_freezes(sess.committee_runs):
            seen = _trio_roles(sess.committee_runs, jobs)
            msg += f"; trio roles {seen} want {list(GOLDEN_COMMITTEE_ROLES)}"
        failures.append(msg)
    elif trio[0] not in set(freeze_ids):
        failures.append(f"trio freeze {trio[0]!r} not in session freezes")
    return trio


def _source_job_counts(jobs: object, failures: list[str]) -> dict[str, int]:
    """Count valid source labels, then report missing source jobs."""
    job_seq = jobs if isinstance(jobs, (list, tuple)) else []
    counts: dict[str, int] = {d: 0 for d in GOLDEN_DOMAINS}
    for job in job_seq:
        dom = getattr(job, "source_domain", None)
        if isinstance(dom, str) and dom.strip().upper() in counts:
            counts[dom.strip().upper()] += 1
    for dom in GOLDEN_DOMAINS:
        if counts[dom] < 1:
            failures.append(f"source domain {dom} has no jobs")
    return counts


def _check_frozen_provenance(get: Callable[..., object], failures: list[str]) -> None:
    """Reject frozen records without an allowed source provenance kind."""
    prov = get("provenance", {})
    kind = prov.get("kind") if isinstance(prov, dict) else None
    if kind not in {"sec_source", "sec_record", "finra_record", "web_source"}:
        failures.append(f"frozen evidence {get('evidence_id')!r} has provenance {kind!r}")


def _frozen_evidence_counts(evidence: object, frozen: set[str], failures: list[str]) -> dict[str, int]:
    """Count frozen evidence, report each invalid provenance, then report empty domains."""
    counts: dict[str, int] = {d: 0 for d in GOLDEN_DOMAINS}
    for row, get in _frozen_rows(evidence, frozen):
        fdom = _evidence_domain(row)
        if fdom in counts:
            counts[fdom] += 1
        _check_frozen_provenance(get, failures)
    for dom in GOLDEN_DOMAINS:
        if counts[dom] < 1:
            failures.append(f"freeze holds no {dom} evidence")
    return counts


def _claimed_ids(final: dict[str, object]) -> set[str]:
    """Read evidence citations from both persisted claim representations."""
    claimed: set[str] = set()
    for key in ("claims", "grounded_claims"):
        grows = final.get(key)
        for grow in grows if isinstance(grows, list) else []:
            ids = grow.get("evidence_ids") if isinstance(grow, dict) else None
            claimed |= set(_str_list(ids))
    return claimed


def _answer_checks(
    sess: ResearchSession, final: dict[str, object], frozen: set[str], answer: str, failures: list[str]
) -> None:
    """Check citations, coverage, disagreements, and substance in that order."""
    invented = sorted(_claimed_ids(final) - frozen)
    if invented:
        failures.append(f"claims cite unfrozen evidence {invented}")
    if not (getattr(sess, "coverage", None) or final.get("coverage")):
        failures.append("coverage empty")
    if not any(final.get(k) for k in ("critical_disagreements", "disagreements", "uncertainties")):
        failures.append("disagreements/uncertainties empty")
    # Targeted follow-up and post-freeze search stay covered by evaluators/stage gate; not reimplemented here.
    if len(answer.strip()) < 50:
        failures.append("final answer not substantive")


def golden_structure(session_id: str, db_path: Path | None = None) -> tuple[bool, list[str], dict[str, object]]:
    """Offline structural checks over the persisted golden session (no network)."""
    from app.research.repository import ResearchRepository

    path = db_path if db_path is not None else _golden_db_path()
    repo = ResearchRepository(path)
    sess = repo.get_session(session_id)
    jobs = repo.list_jobs(session_id)
    evidence = repo.list_evidence(session_id)
    freeze_ids = _str_list(sess.freeze_ids)
    frozen = _frozen_ids(repo, freeze_ids)
    raw_final: object = sess.final_result
    final: dict[str, object] = dict(raw_final) if isinstance(raw_final, dict) else {}
    answer = str(final.get("content") or final.get("answer") or "")
    failures: list[str] = []
    freeze_domains = _freeze_domains(evidence, frozen)
    domains = _domain_checks(jobs, freeze_domains, failures)
    trio = _committee_checks(sess, jobs, freeze_ids, failures)
    if sess.status != "completed":
        failures.append(f"session status {sess.status!r}; want 'completed'")
    job_counts = _source_job_counts(jobs, failures)
    frozen_counts = _frozen_evidence_counts(evidence, frozen, failures)
    _answer_checks(sess, final, frozen, answer, failures)
    summary: dict[str, object] = {
        "session_id": session_id,
        "status": sess.status,
        "source_domains": sorted(domains),
        "freeze_domains": sorted(freeze_domains),
        "source_job_counts": job_counts,
        "frozen_evidence_counts": frozen_counts,
        "trio_freeze": trio[0] if len(set(trio)) == 1 else None,
        "evidence_count": len(evidence),
        "answer_chars": len(answer.strip()),
    }
    return (not failures, failures, summary)


def _live_structural(scenario_name: str) -> list[str]:
    """Structural pass criteria for the golden session, read offline from the live DB."""
    from app.research.repository import ResearchRepository

    if scenario_name != GOLDEN_SCENARIO:
        return []
    try:
        sessions = ResearchRepository().list_sessions(limit=1)
    except Exception:  # noqa: BLE001 - a missing DB is a structural failure, reported below
        return ["golden-no-session-persisted"]
    if not sessions:
        return ["golden-no-session-persisted"]
    first = sessions[0]
    sid = first.get("session_id") if isinstance(first, dict) else None
    if not isinstance(sid, str) or not sid:
        return ["golden-no-session-persisted"]
    _ok, failures, summary = golden_structure(sid)
    print(f"golden structure: {json.dumps(summary, sort_keys=True)}")
    return [f"golden-{failure}" for failure in failures]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=GOLDEN_SCENARIO, help="live golden scenario name")
    parser.add_argument("--model", default=None, help="model ID recorded (or STOCKBOT_MODEL)")
    parser.add_argument("--provider", default=None, help="provider recorded (or STOCKBOT_PROVIDER)")
    parser.add_argument("--model-timeout", default=None, help="per-call model timeout seconds")
    parser.add_argument("--prompt-version", default="v1", help="prompt version stamp")
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    return parser.parse_args(argv)


def _live_skip_reason() -> str | None:
    """Keep the opt-in check ahead of the credential check."""
    if (os.getenv("HEDGEFUND_LIVE") or "").strip() != "1":
        return "SKIP hedgefund live golden: set HEDGEFUND_LIVE=1 to opt in (never CI)."
    missing = _missing_env()
    if missing:
        return f"SKIP hedgefund live golden: missing {', '.join(missing)}."
    return None


def _export_viewer() -> None:
    """Refresh the web projection; export never changes the verdict."""
    try:
        import scripts.export_harness_viewer as viewer

        viewer.main([])
    except Exception as exc:  # noqa: BLE001 - export is best-effort after the verdict
        print(f"viewer export failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skip = _live_skip_reason()
    if skip:
        print(skip, file=sys.stderr)
        return 2
    import scripts.verify_agent_scenarios as live

    provider, model = live.resolve_provider_model(args.provider, args.model, os.environ)
    timeout_s = live.resolve_model_timeout(args.model_timeout, os.environ)
    by_name = live._scenario_map()
    if args.scenario not in by_name:
        print(f"unknown scenario: {args.scenario!r}", file=sys.stderr)
        return 2
    result = live._eval_one_scenario(args.scenario, by_name, provider, model, args.prompt_version, timeout_s)
    structural = _live_structural(args.scenario)
    violations = [*result.violations, *structural]
    passed = result.passed and not structural
    print(f"{'PASS' if passed else 'FAIL'} {result.scenario_name} (live golden via real SEC/FINRA/Exa)")
    if violations:
        print(f"violations: {', '.join(violations)}", file=sys.stderr)
    _export_viewer()
    if args.json:
        print(
            json.dumps(
                {"scenario": result.scenario_name, "passed": passed, "violations": violations},
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
