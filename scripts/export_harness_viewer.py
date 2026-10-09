#!/usr/bin/env python3
"""Export trace/eval/research SQLite projections for the read-only harness viewer (stdlib only).

Reads the authoritative ledgers (research.sqlite, research_traces.sqlite,
eval_runs.sqlite) and writes apps/harness-viewer/convex/projection.ts exporting
PROJECTION. Re-run after live runs; the viewer never writes and never migrates
the ledgers. Missing/in-progress/failed/empty states are preserved explicitly.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_data_root
from app.research.evals.traces import (
    TraceHeader,
    get_trace_events,
    list_traces,
)
from app.research.evidence import evidence_domain
from app.research.models import Job, ResearchSession
from app.research.repository import (
    ResearchRepository,
    get_research_db_path,
)


def _research_db() -> Path:
    try:
        return get_research_db_path()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return get_data_root() / "research.sqlite"


def _all_session_ids(research_db: Path) -> list[str]:
    if not research_db.exists():
        return []
    with sqlite3.connect(research_db) as conn:
        try:
            rows = conn.execute("SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT 50").fetchall()
        except sqlite3.Error:
            return []
    return [str(r[0]) for r in rows if r and r[0]]


def _ts(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _evidence_domain(rec: object) -> str:
    """Source domain of one evidence record: explicit field wins, else the kernel mapping."""
    get = rec.get if isinstance(rec, dict) else getattr(rec, "get", None)
    raw = get("domain", get("source_domain", None)) if callable(get) else None
    if isinstance(raw, str) and raw.strip().upper() in ("SEC", "FINRA", "WEB"):
        return raw.strip().upper()
    raw_prov: object = get("provenance", {}) if callable(get) else {}
    prov: dict[str, object] = raw_prov if isinstance(raw_prov, dict) else {}
    domain = evidence_domain(prov if isinstance(prov, dict) else None)
    return domain


def _event_seq(event: dict[str, object]) -> int:
    seq = event.get("seq")
    return seq if isinstance(seq, int) else 0


def _safe_get_session(repo: ResearchRepository, sid: str) -> ResearchSession | None:
    try:
        return repo.get_session(sid)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _safe_list_jobs(repo: ResearchRepository, sid: str) -> list[Job]:
    try:
        return repo.list_jobs(sid)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return []


def _safe_list_traces(sid: str) -> list[TraceHeader]:
    try:
        return list_traces(sid)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return []


def _collect_trace(traces: list[TraceHeader]) -> dict[str, object]:
    if not traces:
        return {"trace_id": None, "provider": None, "model": None, "conclusion": None, "status": None, "events": []}
    trace_id = traces[0].trace_id
    if not trace_id:
        return {"trace_id": trace_id, "provider": None, "model": None, "conclusion": None, "status": None, "events": []}
    events: list[dict[str, object]] = []
    try:
        for evt in get_trace_events(trace_id):
            events.append({"seq": evt.seq, "eventType": evt.event_type, "payload": dict(evt.payload)})
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        events = []
    try:
        header = traces[0]
        conclusion = header.conclusion
        status = header.status
        provider = header.provider
        model = header.model
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        conclusion = None
        status = None
        provider = None
        model = None
    return {
        "trace_id": trace_id,
        "provider": provider,
        "model": model,
        "conclusion": conclusion,
        "status": status,
        "events": events,
    }


def _collect_evidence(repo: ResearchRepository, sid: str) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    try:
        for rec in repo.list_evidence(sid):
            evidence.append(
                {
                    "evidenceId": str(rec.get("evidence_id", "")),
                    "subject": str(rec.get("subject", "")),
                    "knownAt": _ts(rec.get("known_at")),
                    "sourceName": str(rec.get("source_name", "")),
                    "sourceUri": _ts(rec.get("source_uri")),
                    "domain": _evidence_domain(rec),
                }
            )
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        evidence = []
    return evidence


def _collect_freezes(repo: ResearchRepository, sess: object) -> list[dict[str, object]]:
    freezes: list[dict[str, object]] = []
    freeze_ids = getattr(sess, "freeze_ids", None)
    if not isinstance(freeze_ids, list):
        return freezes
    try:
        for fid in freeze_ids:
            if not isinstance(fid, str):
                continue
            try:
                fr = repo.get_freeze(fid)
            except KeyError:
                continue
            raw_eids: object = fr.get("evidence_ids")
            if isinstance(raw_eids, list):
                eid_list: list[str] = [e for e in raw_eids if isinstance(e, str)]
            else:
                eid_list = []
            freezes.append({"freezeId": fid, "evidenceIds": eid_list})
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        freezes = []
    return freezes


def _clean_text_items(raw: object) -> list[dict[str, object]]:
    clean: list[dict[str, object]] = []
    if not isinstance(raw, list):
        return clean
    for item in raw:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("text"), str):
            continue
        raw_ids: object = item.get("evidence_ids")
        if isinstance(raw_ids, list):
            id_list: list[str] = [e for e in raw_ids if isinstance(e, str)]
        else:
            id_list = []
        clean.append({"text": str(item.get("text")), "evidenceIds": id_list})
    return clean


def _collect_dossiers(repo: ResearchRepository, sid: str) -> list[dict[str, object]]:
    dossiers: list[dict[str, object]] = []
    try:
        for d in repo.list_dossiers(sid):
            dossiers.append(
                {
                    "dossierId": str(d.get("dossier_id", "")),
                    "findings": _clean_text_items(d.get("findings")),
                }
            )
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        dossiers = []
    return dossiers


def _collect_claims(sess: object) -> list[dict[str, object]]:
    try:
        final = getattr(sess, "final_result", None) or {}
        claims = final.get("claims") if isinstance(final, dict) else None
        return _clean_text_items(claims)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return []


def _clean_str_list(raw: object) -> list[str]:
    """Non-empty strings from a list payload ([] for anything else)."""
    return [v.strip() for v in raw if isinstance(v, str) and v.strip()] if isinstance(raw, list) else []


def _collect_coverage(repo: ResearchRepository, sid: str) -> list[dict[str, object]]:
    """Coverage artifacts: search scope + absence text per artifact (never evidence)."""
    artifacts: list[dict[str, object]] = []
    try:
        for row in repo.list_coverage_artifacts(sid):
            artifacts.append(
                {
                    "artifactId": str(row.get("artifact_id", "")),
                    "waveId": row.get("wave_id") if isinstance(row.get("wave_id"), int) else 0,
                    "claimText": str(row.get("claim_text", "")),
                    "searchId": str(row.get("search_id", "")),
                    "query": str(row.get("query", "")),
                }
            )
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        artifacts = []
    return artifacts


def _collect_final(sess: object) -> dict[str, object] | None:
    """Persisted final_result in viewer shape (None when absent/mistyped)."""
    try:
        final = getattr(sess, "final_result", None)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if not isinstance(final, dict):
        return None
    claims = _clean_text_items(final.get("claims") or final.get("grounded_claims"))
    scope = final.get("research_scope")
    allowed: list[str] = _clean_str_list(scope.get("allowed_sources")) if isinstance(scope, dict) else []
    coverage = final.get("coverage")
    coverage_out: dict[str, object] = dict(coverage) if isinstance(coverage, dict) else {}
    return {
        "answer": str(final.get("content") or final.get("answer") or final.get("executive_summary") or ""),
        "executiveSummary": str(final.get("executive_summary") or final.get("answer") or ""),
        "consensus": str(final.get("consensus") or ""),
        "baseCase": str(final.get("base_case") or ""),
        "bullCase": str(
            (final.get("bull_case") or {}).get("summary")
            if isinstance(final.get("bull_case"), dict)
            else final.get("bull_case") or ""
        ),
        "bearCase": str(
            (final.get("bear_case") or {}).get("summary")
            if isinstance(final.get("bear_case"), dict)
            else final.get("bear_case") or ""
        ),
        "disagreements": _clean_str_list(final.get("critical_disagreements") or final.get("disagreements")),
        "positioning": _clean_str_list(final.get("positioning")),
        "catalysts": _clean_str_list(final.get("catalysts")),
        "uncertainties": _clean_str_list(final.get("uncertainties")),
        "whatChangesTheView": _clean_str_list(final.get("what_would_change") or final.get("what_changes_the_view")),
        "limitations": _clean_str_list(final.get("evidence_limitations") or final.get("limitations")),
        "claims": claims,
        "sources": [dict(s) for s in final["sources"]] if isinstance(final.get("sources"), list) else [],
        "coverage": coverage_out,
        "allowedSources": allowed,
        "freezeId": str(final.get("freeze_id") or ""),
        "asOf": str(final.get("as_of") or ""),
    }


def _job_row(job: Job) -> dict[str, object]:
    diag = job.diagnostics or {}
    if job.failure is None:
        failure_category = None
        failure_message = None
    else:
        failure_category = job.failure.category
        failure_message = job.failure.message
    assignment = diag.get("assignment_id")
    if not isinstance(assignment, str):
        assignment = None
    role = diag.get("role")
    if not isinstance(role, str):
        role = None
    return {
        "jobId": job.job_id,
        "parentJobId": job.parent_job_id,
        "jobType": job.job_type,
        "owner": job.owner,
        "waveId": job.wave_id,
        "status": job.status,
        "failureCategory": failure_category,
        "failureMessage": failure_message,
        "assignmentId": assignment,
        "role": role,
        "sourceDomain": job.source_domain,
    }


def _job_rows(jobs: list[Job]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in jobs:
        rows.append(_job_row(job))
    return rows


def _session_as_of_str(sess: object) -> str | None:
    """ISO as-of string or None when absent/non-datetime."""
    from datetime import datetime

    as_of = getattr(sess, "as_of", None)
    return as_of.isoformat() if isinstance(as_of, datetime) else None


def _session_time_str(sess: object, name: str) -> str:
    """ISO datetime session field; missing/non-datetime reads as empty."""
    from datetime import datetime

    value = getattr(sess, name, None)
    return value.isoformat() if isinstance(value, datetime) else ""


def _session_str_list(sess: object, name: str) -> list[object]:
    """Session list field; non-list reads as empty."""
    value = getattr(sess, name, None)
    return value if isinstance(value, list) else []


def _session_row(
    sid: str,
    sess: ResearchSession,
    jobs: list[Job],
    trace: dict[str, object],
    evidence: list[dict[str, object]],
    freezes: list[dict[str, object]],
    dossiers: list[dict[str, object]],
    claims: list[dict[str, object]],
    coverage_artifacts: list[dict[str, object]],
    final_result: dict[str, object] | None,
) -> dict[str, object]:
    events = trace["events"]
    assert isinstance(events, list)
    job_rows = _job_rows(jobs)
    waves = sorted({row["waveId"] for row in job_rows if isinstance(row.get("waveId"), int)})
    return {
        "sessionId": sid,
        "waveId": sess.current_wave,
        "question": sess.query,
        "status": sess.status,
        "asOf": _session_as_of_str(sess),
        "updatedAt": _session_time_str(sess, "updated_at"),
        "traceId": trace["trace_id"],
        "conclusion": trace["conclusion"],
        "traceStatus": trace["status"],
        "provider": trace["provider"],
        "model": trace["model"],
        "jobs": job_rows,
        "waves": waves,
        "events": sorted(events, key=_event_seq),
        "claims": claims,
        "evidence": evidence,
        "freezes": freezes,
        "dossiers": dossiers,
        "coverageArtifacts": coverage_artifacts,
        "finalResult": final_result,
        "committeeRuns": list(_session_str_list(sess, "committee_runs")),
    }


def build_session_run(sid: str, repo: ResearchRepository | None = None) -> dict[str, object] | None:
    if repo is None:
        repo = ResearchRepository()
    sess = _safe_get_session(repo, sid)
    if sess is None:
        return None
    jobs = _safe_list_jobs(repo, sid)
    traces = _safe_list_traces(sid)
    trace = _collect_trace(traces)
    evidence = _collect_evidence(repo, sid)
    freezes = _collect_freezes(repo, sess)
    dossiers = _collect_dossiers(repo, sid)
    claims = _collect_claims(sess)
    coverage_artifacts = _collect_coverage(repo, sid)
    final_result = _collect_final(sess)
    return _session_row(sid, sess, jobs, trace, evidence, freezes, dossiers, claims, coverage_artifacts, final_result)


def _parse_violations(raw: object) -> list[object]:
    try:
        violations = json.loads(str(raw))
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return [str(raw)]
    if isinstance(violations, list):
        return violations
    return [str(violations)]


def _read_eval_runs(conn: sqlite3.Connection) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for row in conn.execute(
        "SELECT eval_run_id, model, provider, harness_version, prompt_version, git_sha, started_at, scenario_version FROM eval_runs ORDER BY started_at DESC LIMIT 20"
    ):
        runs.append(
            {
                "evalRunId": str(row[0]),
                "model": str(row[1]),
                "provider": str(row[2]),
                "harnessVersion": str(row[3]),
                "promptVersion": str(row[4]),
                "gitSha": str(row[5]),
                "startedAt": str(row[6]),
                "scenarioVersion": str(row[7]),
                "passed": 0,
                "failed": 0,
            }
        )
    return runs


def _read_scenario_results(conn: sqlite3.Connection) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in conn.execute(
        "SELECT eval_run_id, scenario_name, passed, violations_json FROM eval_scenario_results ORDER BY eval_run_id, scenario_name LIMIT 200"
    ):
        violations = _parse_violations(row[3])
        results.append(
            {"evalRunId": str(row[0]), "scenarioName": str(row[1]), "passed": bool(row[2]), "violations": violations}
        )
    return results


def _read_failure_records(conn: sqlite3.Connection) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in conn.execute(
        "SELECT failure_id, eval_run_id, scenario_name, violation FROM failure_records ORDER BY eval_run_id, scenario_name LIMIT 200"
    ):
        records.append(
            {"failureId": str(row[0]), "evalRunId": str(row[1]), "scenarioName": str(row[2]), "violation": str(row[3])}
        )
    return records


def read_eval_db(eval_db: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    eval_runs: list[dict[str, object]] = []
    scenario_results: list[dict[str, object]] = []
    failure_records: list[dict[str, object]] = []
    if not eval_db.exists():
        return (eval_runs, scenario_results, failure_records)
    try:
        with sqlite3.connect(eval_db) as conn:
            eval_runs.extend(_read_eval_runs(conn))
            scenario_results.extend(_read_scenario_results(conn))
            failure_records.extend(_read_failure_records(conn))
    except sqlite3.Error:
        pass
    return (eval_runs, scenario_results, failure_records)


def attach_pass_fail(eval_runs: list[dict[str, object]], scenario_results: list[dict[str, object]]) -> None:
    for run in eval_runs:
        run_id = run.get("evalRunId")
        passed = 0
        failed = 0
        for r in scenario_results:
            if r.get("evalRunId") != run_id:
                continue
            if r.get("passed"):
                passed += 1
            else:
                failed += 1
        run["passed"] = passed
        run["failed"] = failed


def build_projection(
    research_runs: list[dict[str, object]],
    eval_runs: list[dict[str, object]],
    scenario_results: list[dict[str, object]],
    experiments: list[dict[str, object]],
    failure_records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "researchRuns": research_runs,
        "evalRuns": eval_runs,
        "evalScenarioResults": scenario_results,
        "experiments": experiments,
        "failureRecords": failure_records,
    }


def render_projection_ts(projection: dict[str, object]) -> str:
    return (
        'import type { EvalRun, EvalScenarioResult, Experiment, FailureRecord, ResearchRun } from "./schema";\n'
        "export const PROJECTION: { researchRuns: ResearchRun[]; evalRuns: EvalRun[]; "
        "evalScenarioResults: EvalScenarioResult[]; experiments: Experiment[]; "
        "failureRecords: FailureRecord[] } = " + json.dumps(projection, indent=2, sort_keys=True) + ";\n"
    )


def _resolve_eval_db() -> Path:
    try:
        root = get_data_root()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        root = REPO_ROOT / "data"
    return root / "eval_runs.sqlite"


def _write_projection(projection: dict[str, object]) -> Path:
    out_path = REPO_ROOT / "apps" / "harness-viewer" / "convex" / "projection.ts"
    out_path.write_text(render_projection_ts(projection), encoding="utf-8")
    return out_path


def main() -> int:
    repo = ResearchRepository()
    session_ids = _all_session_ids(_research_db())
    research_runs: list[dict[str, object]] = []
    for sid in session_ids:
        run = build_session_run(sid, repo)
        if run is not None:
            research_runs.append(run)
    eval_runs, scenario_results, failure_records = read_eval_db(_resolve_eval_db())
    attach_pass_fail(eval_runs, scenario_results)
    projection = build_projection(research_runs, eval_runs, scenario_results, [], failure_records)
    out_path = _write_projection(projection)
    print(f"wrote {out_path} researchRuns={len(research_runs)} evalRuns={len(eval_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
