"""Authoritative kernel service API: deterministic persistence/policy/PIT/evidence/jobs.

Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind these live readers, never inside normalization.

Pi/CLI/IPC call these functions; nothing here invokes a model, spawns a
subprocess, or synthesizes outcomes. stdlib + kernel modules only.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from . import jobs as _jobs
from . import session as _session
from .evidence import (
    CLAIM_KINDS,
    Evidence,
    evidence_content_hash,
    evidence_domain,
    evidence_to_dict,
    finra_record_ref,
    ingest_evidence,
    normalize_accession,
    search_run_ref,
    sec_record_ref,
    sec_source_ref,
    web_source_ref,
)
from .freeze import EvidenceFreeze
from .models import (
    DecisionRecord,
    FailureCategory,
    JSONValue,
    ResearchNode,
    ResearchSession,
    default_policy,
    new_decision_id,
    new_node_id,
    utcnow,
    validate_json_mapping,
    validate_json_value,
)
from .repository import ResearchRepository, pending_next_action

if TYPE_CHECKING:
    from .agents import GroundedClaim
    from .agents.bearbot import BearAnalysis
    from .agents.bullbot import BullAnalysis
    from .agents.stockbot import StockbotAnalysis
    from .director import Wave1Result, WaveDecision
    from .models import Job
    from .synthesis.committee import CommitteeDisagreement
    from .synthesis.final import FinalSynthesis

__all__ = [
    "ResearchNotFound",
    "attach_job_runtime",
    "authorize_and_consume_dispatch",
    "block_node",
    "cancel_job",
    "cancel_research",
    "complete_job",
    "create_committee_jobs",
    "create_node",
    "create_research",
    "decide_next_wave",
    "fail_job",
    "finalize_session",
    "freeze_session",
    "get_tool_result",
    "heartbeat_job",
    "inspect_research",
    "job_diagnostics",
    "list_research",
    "persist_tool_result",
    "ready_nodes",
    "record_committee_analysis",
    "record_decision",
    "record_evidence",
    "reject_node",
    "research_events",
    "resolve_node",
    "resume_research",
    "retry_job",
    "run_research",
    "start_job",
    "submit_source_result",
    "transition_job_completed",
]


class ResearchNotFound(KeyError):
    """Unknown session/job/evidence id. Subclasses KeyError for existing handlers."""


def transition_job_completed(
    session_id: str, job_id: str, *, repo: ResearchRepository | Path | str | None = None
) -> dict[str, object]:
    """Single authoritative wave/counter transition: job wave + session current_wave together."""
    store = _repo(repo)
    job = _require_job(store, job_id)
    found = _require_session(store, session_id)
    if job.session_id != found.session_id:
        raise ValueError("transition_job_completed: session/job mismatch")
    wave = job.wave_id if isinstance(job.wave_id, int) and job.wave_id >= 1 else 1
    cur = store.get_session(session_id)
    if wave > cur.current_wave:
        from dataclasses import replace as _replace

        store.save_session(_replace(cur, current_wave=wave, updated_at=utcnow()))
        cur = store.get_session(session_id)
    return {"session_id": session_id, "job_id": job_id, "current_wave": cur.current_wave, "job_wave": wave}


def research_events(
    session_id: str,
    job_id: str | None = None,
    limit: int = 100,
    cursor: int = 0,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Bounded structured events from the journal (read-only)."""
    store = _repo(repo)
    _require_session(store, session_id)
    limit = max(1, min(200, limit if isinstance(limit, int) else 100))
    cursor = max(0, cursor if isinstance(cursor, int) else 0)
    events = store.list_events(session_id)
    if job_id:
        events = [e for e in events if isinstance(e.payload, dict) and e.payload.get("job_id") == job_id]
    page = events[cursor : cursor + limit]
    return {
        "session_id": session_id,
        "job_id": job_id,
        "cursor": cursor,
        "limit": limit,
        "events": [
            {
                "ts": e.timestamp.isoformat(),
                "kind": e.event_type,
                "job_id": (e.payload.get("job_id") if isinstance(e.payload, dict) else None),
                "detail": dict(e.payload),
            }
            for e in page
        ],
    }


_TERMINAL_JOBS = frozenset({"completed", "failed", "cancelled", "timed_out"})


def _repo(repo: ResearchRepository | Path | str | None = None) -> ResearchRepository:
    if isinstance(repo, ResearchRepository):
        return repo
    if repo is None:
        return ResearchRepository()
    return ResearchRepository(path=repo)


def _require_session(repo: ResearchRepository, session_id: str):
    try:
        return repo.get_session(session_id)
    except KeyError:
        raise ResearchNotFound(f"unknown session_id: {session_id!r}") from None


def _require_job(repo: ResearchRepository, job_id: str):
    try:
        return repo.get_job(job_id)
    except KeyError:
        raise ResearchNotFound(f"unknown job_id: {job_id!r}") from None


def _emit(store: ResearchRepository, session_id: str, event_type: str, payload: dict[str, object]) -> None:
    """Append one journal event (best-effort; never breaks the mutation)."""
    try:
        from .journal import append_event, hydrate

        hydrate(session_id, store.list_events(session_id))
        store.save_event(append_event(session_id, event_type, "service", "service", dict(payload)))
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
        pass


def _enforce_live_job(store: ResearchRepository, job_id: str):
    """Fail-closed deadline guard shared by every governed mutation."""
    from dataclasses import replace as _replace

    from .models import Failure, normalize_time

    job = _require_job(store, job_id)
    if job.deadline is not None:
        deadline = normalize_time(job.deadline)
        if utcnow() > deadline:
            timed = _replace(
                job,
                status="timed_out",
                completed_at=utcnow(),
                failure=Failure(category="timeout", message="deadline expired"),
            )
            store.save_job(timed)
            raise ValueError(f"ERR_JOB_TIMED_OUT: job {job_id!r} deadline expired")
    if job.status == "timed_out":
        raise ValueError(f"ERR_JOB_TIMED_OUT: job {job_id!r} timed out")
    return job


def heartbeat_job(job_id: str, *, repo: ResearchRepository | Path | str | None = None) -> dict[str, JSONValue]:
    """Refresh last_heartbeat_at; read path surfaces staleness + remaining."""
    from dataclasses import replace as _replace

    store = _repo(repo)
    job = _require_job(store, job_id)
    beat = _replace(job, last_heartbeat_at=utcnow())
    store.save_job(beat)
    return beat.to_dict()


# Closed runtime-identity vocabulary the OMP session writes into Job.diagnostics.
_JOB_RUNTIME_KEYS: tuple[str, ...] = (
    "runtime",
    "runtime_agent_id",
    "runtime_parent_agent_id",
    "runtime_task_call_id",
    "runtime_agent_type",
    "runtime_session_file",
)


def attach_job_runtime(
    job_id: str, runtime: Mapping[str, object] | None = None, *, repo: ResearchRepository | Path | str | None = None
) -> dict[str, JSONValue]:
    """Merge OMP runtime identity into one job's diagnostics and persist it.

    Only accepted keys with non-empty string values are written; everything
    else (other keys, blanks, non-strings, prior values) is left untouched so
    partial calls accumulate across the job's life.
    """
    store = _repo(repo)
    job = _require_job(store, job_id)
    merged = dict(job.diagnostics)
    merged.update(
        {
            key: value
            for key, value in (runtime or {}).items()
            if key in _JOB_RUNTIME_KEYS and isinstance(value, str) and value
        }
    )
    updated = replace(job, diagnostics=merged)
    store.save_job(updated)
    return updated.to_dict()


def _jobs_by_status(job_list: list[Job]) -> dict[str, list[dict[str, JSONValue]]]:
    """Jobs split by terminal state; successful completions never render under failure."""
    out: dict[str, list[dict[str, JSONValue]]] = {"completed": [], "failed": [], "cancelled": [], "timed_out": []}
    for job in job_list:
        dumped = job.to_dict()
        if job.status in out:
            out[job.status].append(dumped)
    return out


def job_diagnostics(
    session_id: str, job_id: str, *, repo: ResearchRepository | Path | str | None = None
) -> dict[str, JSONValue]:
    """Non-empty diagnostics: staleness, deadline remaining, per-status counts, evidence count."""
    from .models import HEARTBEAT_STALE_S

    store = _repo(repo)
    job = _require_job(store, job_id)
    now = utcnow()
    hb = job.last_heartbeat_at or job.started_at or job.created_at
    staleness = max(0.0, (now - hb).total_seconds()) if hb is not None else 0.0
    remaining = (job.deadline - now).total_seconds() if job.deadline is not None else None
    count = sum(1 for r in store.list_evidence(session_id) if r.get("job_id") == job_id)
    by_status = _jobs_by_status(store.list_jobs(session_id))
    return {
        "job_id": job_id,
        "job_status": job.status,
        "stale": staleness > HEARTBEAT_STALE_S,
        "staleness_s": staleness,
        "deadline_remaining_s": remaining,
        "evidence_count": count,
        "jobs_by_status": {k: [j.get("job_id") for j in v if isinstance(j, dict)] for k, v in by_status.items()},
        "last_event": job.status,
    }


def create_research(
    question: str,
    objective: str | None = None,
    *,
    as_of: str | None = None,
    policy: dict[str, JSONValue] | None = None,
    repo: ResearchRepository | Path | str | None = None,
) -> str:
    """Create a session plus its first source_agent job; returns session_id."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("create_research: 'question' must be a non-empty string")
    store = _repo(repo)
    new_session = _session.create_session(
        question.strip(),
        (objective or question).strip(),
        as_of=as_of,
        policy=policy if policy is not None else default_policy(),
    )
    updated, job = _jobs.create_job(new_session, [], job_type="source_agent", owner="kernel", source_domain=None)
    running = _jobs.start_job(job)
    store.save_session_and_job(updated, running)
    _emit(store, updated.session_id, "job.started", {"job_id": running.job_id})
    return updated.session_id


def run_research(
    session_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Idempotent first-job ensure: returns the open/first job for a session."""
    store = _repo(repo)
    found = _require_session(store, session_id)
    existing = store.list_jobs(session_id)
    if existing:
        for job in existing:
            if job.status == "running":
                return job.to_dict()
        for job in existing:
            if job.status == "queued":
                running = _jobs.start_job(job)
                store.save_job(running)
                return running.to_dict()
        return existing[0].to_dict()
    updated, job = _jobs.create_job(found, [], job_type="source_agent", owner="kernel", source_domain=None)
    running = _jobs.start_job(job)
    store.save_session_and_job(updated, running)
    return running.to_dict()


def resume_research(
    session_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Read-only resume snapshot; never duplicates jobs/evidence."""
    store = _repo(repo)
    try:
        state = store.resume(session_id)
    except KeyError:
        raise ResearchNotFound(f"unknown session_id: {session_id!r}") from None
    return {
        "session": state.session.to_dict(),
        "wave": state.wave,
        "budgets": dict(state.budgets),
        "open_job_ids": list(state.open_job_ids),
        "pending_next_action": state.pending_next_action,
    }


def inspect_research(
    session_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Session + jobs (split completed vs failed vs cancelled vs timed_out) + deterministic next action (read-only)."""
    store = _repo(repo)
    found = _require_session(store, session_id)
    job_list = store.list_jobs(session_id)
    latest_freeze: dict[str, JSONValue] | None = None
    if found.freeze_ids:
        fid = found.freeze_ids[-1]
        try:
            latest_freeze = store.get_freeze(fid)
        except KeyError:
            raise ValueError(f"inspect: unknown freeze_id: {fid!r}") from None
    return {
        "session": found.to_dict(),
        "jobs": [j.to_dict() for j in job_list],
        "jobs_by_status": _jobs_by_status(job_list),
        "pending_next_action": pending_next_action(found, job_list),
        "latest_freeze": latest_freeze,
    }


def cancel_research(
    session_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Cancel a session; terminal sessions return current state (no-op)."""
    from .models import SessionStatus

    store = _repo(repo)
    found = _require_session(store, session_id)
    if found.status in _session.TERMINAL_STATUSES:
        return found.to_dict()
    try:
        out = _session.transition_session(found, SessionStatus.CANCELLED)
    except ValueError:
        return found.to_dict()
    store.save_session(out)
    return out.to_dict()


def start_job(
    session_id: str,
    type: str = "source_agent",
    source: str | None = None,
    parent: str | None = None,
    budget: dict[str, object] | None = None,
    *,
    repo: ResearchRepository | Path | str | None = None,
    owner: str = "kernel",
    wave_id: int = 1,
    model: str | None = None,
) -> dict[str, JSONValue]:
    """Create a queued job for the kernel scheduler to run, then mark it running. Returns the job."""
    store = _repo(repo)
    found = _require_session(store, session_id)
    existing = store.list_jobs(session_id)
    details: dict[str, object] = dict(budget or {})
    token_budget = details.get("token_budget")
    tool_budget = details.get("tool_budget")
    child_budget = details.get("child_budget")
    wave_raw: object = details.get("wave_id", wave_id)
    if isinstance(wave_raw, bool) or not isinstance(wave_raw, int):
        raise ValueError(f"start_job: 'wave_id' must be an int, got {wave_raw!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    wave_id_arg: int = wave_raw
    model_raw: object = details.get("model")
    model_arg: str | None = model if model is not None else (model_raw if isinstance(model_raw, str) else None)
    updated, job = _jobs.create_job(
        found,
        existing,
        job_type=type,
        owner=str(details.get("owner", owner)),
        wave_id=wave_id_arg,
        parent_job_id=parent,
        source_domain=source,
        model=model_arg,
        token_budget=token_budget if isinstance(token_budget, int) else None,
        tool_budget=tool_budget if isinstance(tool_budget, int) else None,
        child_budget=child_budget if isinstance(child_budget, int) else None,
    )
    store.save_session(updated)
    store.save_job(job)
    running = _jobs.start_job(job)
    store.save_job(running)
    return running.to_dict()


def complete_job(
    job_id: str,
    outcome: Mapping[str, object] | None = None,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Mark a Pi-run job completed; terminal jobs return current state (no-op)."""
    store = _repo(repo)
    job = _require_job(store, job_id)
    if job.status in _TERMINAL_JOBS:
        return job.to_dict()
    if job.status == "queued":
        job = _jobs.start_job(job)
        store.save_job(job)
    result = dict(outcome) if isinstance(outcome, Mapping) else {}
    try:
        done = _jobs.complete_job(job, result=result)
    except ValueError:
        return job.to_dict()
    store.save_job(done)
    transition_job_completed(job.session_id, job.job_id, repo=store)
    return store.get_job(job.job_id).to_dict()


def fail_job(
    job_id: str,
    category: str,
    message: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Mark a Pi-run job failed; terminal jobs return current state (no-op)."""
    store = _repo(repo)
    job = _require_job(store, job_id)
    if job.status in _TERMINAL_JOBS:
        return job.to_dict()
    done = _jobs.fail_job(job, category, message)
    store.save_job(done)
    return store.get_job(job.job_id).to_dict()


def cancel_job(
    job_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Mark a Pi-run job cancelled; terminal jobs return current state (no-op)."""
    store = _repo(repo)
    job = _require_job(store, job_id)
    if job.status in _TERMINAL_JOBS:
        return job.to_dict()
    done = _jobs.cancel_job(job)
    store.save_job(done)
    return store.get_job(job.job_id).to_dict()


def _exact_source_bytes(accession: str, document: str) -> tuple[bytes, str]:
    """Exact source bytes that produced the verified window, re-resolved live."""
    from app.sec.documents import _filing, _resolve_in, _source_bytes_of

    attachment = _resolve_in(_filing(accession), accession, document or None)
    payload, rep = _source_bytes_of(attachment)
    if payload is None:
        raise ValueError(
            f"record_evidence: ERR_NO_SOURCE_BYTES (EdgarTools exposed no exact bytes for {accession}/{document})"
        )
    return payload, rep or "source_bytes"


def _coerce_dt(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.strip())
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _evidence_content(data: Mapping[str, object]) -> tuple[str, object]:
    """Resolve content with claim/JSON fallbacks. Returns (content, claim)."""
    content_raw = data.get("content")
    claim = data.get("claim_text", data.get("claim", ""))
    if isinstance(content_raw, str) and content_raw.strip():
        return content_raw.strip(), claim
    if isinstance(claim, str) and claim.strip():
        return claim.strip(), claim
    dumped = json.dumps(dict(data), sort_keys=True, default=str)
    return dumped if dumped else "{}", claim


def _evidence_wave(data: Mapping[str, object], job_wave: int) -> int:
    """Coerce item wave to job wave; non-int waves fall back to the job wave."""
    wave_raw = data.get("wave_id", job_wave)
    if isinstance(wave_raw, bool) or not isinstance(wave_raw, int):
        return job_wave
    if wave_raw != job_wave:
        raise ValueError(f"record_evidence: item wave {wave_raw!r} != job wave {job_wave!r}")
    return job_wave


def _norm_lower(value: object) -> str:
    """Lowercase-normalize one identity-key component."""
    return str(value or "").strip().lower()


def _evidence_identity(
    data: Mapping[str, object],
    claim: object,
    subject: str,
    provenance: Mapping[str, object],
) -> str:
    """Identity key for dedupe: the source-backed identity of one observed fact."""
    tail: object = data.get("observation_type", claim[:80] if isinstance(claim, str) else "")
    kind = str(provenance.get("kind") or "")
    if kind == "finra_record":
        return "|".join(
            (
                "finra",
                "record",
                _norm_lower(provenance.get("tool_name")),
                _norm_lower(provenance.get("record_identity")),
                _norm_lower(subject),
                _norm_lower(tail),
            )
        )
    if kind == "web_source":
        return "|".join(
            (
                "web",
                "source",
                _norm_lower(provenance.get("url")),
                _norm_lower(provenance.get("excerpt"))[:200],
                _norm_lower(subject),
                _norm_lower(tail),
            )
        )
    return "|".join(
        (
            "sec",
            "filing",
            _norm_lower(provenance.get("accession_no")),
            _norm_lower(provenance.get("document_name")),
            _norm_lower(subject),
            _norm_lower(tail),
        )
    )


# Closed claim vocabulary; legacy field names/values are aliases, wording is never the signal.
_CLAIM_KIND_ALIASES: dict[str, str] = {
    "filing_observation": "observed_fact",
    "search_coverage": "absence_observation",
}
_CLAIM_KIND_KEYS: tuple[str, ...] = ("claim_kind", "evidence_type", "ev_type", "type")
_ACCESSION_KEYS: tuple[str, ...] = ("source_record_id", "accession_no", "accession")
_DOCUMENT_KEYS: tuple[str, ...] = ("document_name", "document")
_PASSAGE_KEYS: tuple[str, ...] = ("matching_passage", "passage", "section", "fact")
_ABSENCE_COVERAGE_LISTS: tuple[str, ...] = ("forms", "dates", "partitions", "entities", "docs", "gaps")
_ABSENCE_COVERAGE_BOOLS: tuple[str, ...] = ("pagination_complete", "complete")
_RAW_SOURCE_RULE = (
    "SEC search results are navigation artifacts. Open the underlying "
    "filing/document and cite a raw passage before recording evidence."
)


def _first_text(data: Mapping[str, object], keys: Sequence[str]) -> str | None:
    """First non-blank string under any of the keys, in key order."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _claim_kind(data: Mapping[str, object]) -> str:
    """Closed claim kind from claim_kind/evidence_type/ev_type/type; absent means observed_fact.

    Legacy values are aliases; anything else, or disagreeing fields, fail closed.
    ``absence_observation`` is the search-coverage route: it is recorded as a session
    coverage artifact, never as ledger evidence.
    """
    kinds: set[str] = set()
    for key in _CLAIM_KIND_KEYS:
        raw = data.get(key)
        if raw is None:
            continue
        name = str(raw)
        kind = name if name in (*CLAIM_KINDS, "absence_observation") else _CLAIM_KIND_ALIASES.get(name)
        if kind is None:
            raise ValueError(f"record_evidence: ERR_UNKNOWN_EVIDENCE_TYPE ({key}={raw!r})")
        kinds.add(kind)
    if len(kinds) > 1:
        raise ValueError(f"record_evidence: ERR_UNKNOWN_EVIDENCE_TYPE (claim kind fields disagree: {sorted(kinds)})")
    return next(iter(kinds)) if kinds else "observed_fact"


def _navigational_ids(data: Mapping[str, object]) -> set[str]:
    """search_id/query values of a search hit: navigation, never an accession."""
    return {v.strip() for v in (data.get("search_id"), data.get("query")) if isinstance(v, str) and v.strip()}


def _required_text(data: Mapping[str, object], keys: Sequence[str], message: str) -> str:
    """First non-blank text under ``keys``, else the caller's pinned error (code included)."""
    value = _first_text(data, keys)
    if value is None:
        raise ValueError(message)
    return value


def _declared_accession(data: Mapping[str, object]) -> str | None:
    """Canonical accession a caller declared, or None when it declared none.

    The source_handle identifies the filing, so a declared accession is an
    optional cross-check: it must be well-formed and must not be a search id.
    """
    raw = next((data[k] for k in _ACCESSION_KEYS if data.get(k)), None)
    navigational = _navigational_ids(data)
    if raw is None:
        if navigational:
            # A search hit carries no raw document: it stays navigation, never evidence.
            raise ValueError(f"record_evidence: ERR_RAW_SOURCE_REQUIRED ({_RAW_SOURCE_RULE})")
        return None
    if isinstance(raw, str) and raw.strip() in navigational:
        raise ValueError("record_evidence: ERR_PROVENANCE_MISMATCH (a search id is navigation, not an accession)")
    try:
        return normalize_accession(raw)
    except ValueError:
        raise ValueError(
            "record_evidence: ERR_ACCESSION_FORMAT "
            f"(accession {raw!r} must be the NNNNNNNNNN-NN-NNNNNN of the filing you read)"
        ) from None


@dataclass(frozen=True)
class MaterializedEvidenceSource:
    """Verified SEC passage plus the exact source bytes and canonical timing."""

    provenance: dict[str, JSONValue]
    source_bytes: bytes
    representation: str
    source_content_hash: str
    source_url: str | None
    known_at: datetime | None
    filed_at: datetime | None
    retrieved_at: datetime | None


def _observed_provenance(
    data: Mapping[str, object],
    job: Job | None = None,
    *,
    store: ResearchRepository | None = None,
    session_id: str | None = None,
    as_of: datetime | str | None = None,
) -> MaterializedEvidenceSource | dict[str, JSONValue]:
    """Kernel-materialized provenance for an observed fact, routed by the owning job's domain.

    SEC jobs reload a get_sec_document handle from the archive, else replay a
    persisted SEC structured tool result. FINRA/WEB jobs replay a persisted
    staged tool result the kernel stored at dispatch time; a citation naming
    no persisted result fails ERR_RAW_SOURCE_REQUIRED, and one naming bytes
    the persisted result does not reproduce fails ERR_PASSAGE_NOT_IN_SOURCE.
    A bare row/URL with no persisted result is navigation at best, never
    evidence.
    """
    domain = (job.source_domain or "SEC").upper() if job is not None else "SEC"
    if domain == "FINRA":
        return _finra_provenance(data, job, store=store, session_id=session_id)
    if domain == "WEB":
        return _web_provenance(data, job, store=store, session_id=session_id)
    if isinstance(data.get("tool_result_id"), str) and data.get("tool_result_id"):
        return _sec_record_provenance(data, job, store=store, session_id=session_id)
    return _sec_provenance(data, as_of=as_of)


def _sec_provenance(data: Mapping[str, object], *, as_of: datetime | str | None) -> MaterializedEvidenceSource:
    """SECSourceRef for an observed fact: the KERNEL-materialized passage of a canonical handle."""
    locator = _required_text(
        data,
        _PASSAGE_KEYS,
        f"record_evidence: ERR_RAW_SOURCE_REQUIRED ({_RAW_SOURCE_RULE})",
    )
    handle = data.get("source_handle")
    if not isinstance(handle, Mapping):
        raise ValueError(  # noqa: TRY004 - the public error contract pins ValueError, tests are oracle
            "record_evidence: ERR_RAW_SOURCE_REQUIRED "
            f"({_RAW_SOURCE_RULE} Open the document with get_sec_document and pass its "
            "source_handle with the passage you are citing.)"
        )
    declared_accession = _declared_accession(data)
    declared_document = _first_text(data, _DOCUMENT_KEYS)
    materialized = materialize_sec_passage(handle, locator, as_of=as_of)
    _check_declared_ref(materialized.provenance, accession=declared_accession, document=declared_document)
    return materialized


_FINRA_EVIDENCE_TOOLS: frozenset[str] = frozenset(
    {
        "get_finra_datapoints",
        "query_finra",
        "get_short_interest",
        "get_short_pressure_profile",
        "get_reg_sho_volume",
        "get_threshold_securities",
        "get_short_interest_leaderboard",
    }
)
"""FINRA tools whose persisted results can ground a finra_record (catalog reads excluded)."""

_SEC_REPLAY_TOOLS: frozenset[str] = frozenset(
    {
        "list_sec_filings",
        "get_sec_filing",
        "get_insider_activity",
        "get_planned_insider_sales",
        "get_fundamentals",
        "get_beneficial_ownership",
        "get_ownership_changes",
        "get_offering_history",
        "get_dilution_profile",
        "get_governance_events",
        "get_transaction_status",
        "get_financial_statements",
        "get_xbrl_facts",
        "get_obligations",
        "get_valuation_metrics",
        "get_recent_ownership_filings",
        "get_material_events",
        "search_sec_filings",
    }
)
"""SEC structured tools whose persisted results can ground a sec_record (document reads excluded)."""


def _tool_result_ref(data: Mapping[str, object], domain: str) -> str:
    """Persisted tool_result_id a FINRA/WEB citation names; bare rows fail closed."""
    for key in ("tool_result_id", "source_handle_id", "result_id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"record_evidence: ERR_RAW_SOURCE_REQUIRED ({domain} rows are navigation artifacts. "
        "Cite the persisted tool result id of the tool response you read, not a copied number.)"
    )


def _normalize_record_text(value: object) -> str:
    """Whitespace-normalized comparison text for persisted-result replay."""
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _finra_record_texts(result: Mapping[str, object]) -> list[str]:
    """Citable texts of one persisted FINRA result: raw rows + briefing + metrics JSON."""
    texts: list[str] = []
    payload = result.get("result")
    payload = payload if isinstance(payload, Mapping) else result
    records = payload.get("records")
    if isinstance(records, list):
        for row in records:
            if isinstance(row, Mapping):
                texts.append(_normalize_record_text(json.dumps(row, sort_keys=True, default=str)))
    for key in ("briefing", "briefing_source"):
        texts.append(_normalize_record_text(payload.get(key)))
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        texts.append(_normalize_record_text(json.dumps(metrics, sort_keys=True, default=str)))
    return [text for text in texts if text]


def _finra_known_at(result: Mapping[str, object]) -> str | None:
    """Persisted publication/availability time of one FINRA result; None when unknown."""
    payload = result.get("result")
    payload = payload if isinstance(payload, Mapping) else result
    for key in ("published_at", "publication_date", "available_at"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _finra_provenance(
    data: Mapping[str, object],
    job: Job | None = None,
    *,
    store: ResearchRepository | None = None,
    session_id: str | None = None,
) -> dict[str, JSONValue]:
    """FinraRecordRef replayed against the persisted FINRA tool result."""
    from .repository import ResearchRepository as _RR

    ref_id = _tool_result_ref(data, "FINRA")
    locator = _normalize_record_text(_first_text(data, (*_PASSAGE_KEYS, "record_identity")))
    if not locator:
        raise ValueError(
            "record_evidence: ERR_RAW_SOURCE_REQUIRED (a FINRA citation names the persisted "
            "tool result id plus the record values you are citing.)"
        )
    try:
        repo = store if store is not None else _RR()
        result = repo.get_tool_result(ref_id)
    except KeyError:
        raise ValueError(
            "record_evidence: ERR_RAW_SOURCE_REQUIRED (unknown persisted FINRA tool result; "
            "cite the tool_result_id the kernel returned for the response you read.)"
        ) from None
    other = result.get("session_id")
    if isinstance(session_id, str) and session_id and isinstance(other, str) and other and other != session_id:
        raise ValueError(
            f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} belongs to session {other!r}, not {session_id!r})"
        )
    tool_name = str(result.get("tool_name") or "")
    if tool_name not in _FINRA_EVIDENCE_TOOLS:
        raise ValueError(
            f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} is {tool_name!r}, not FINRA records)"
        )
    if job is not None:
        allowed = _submit_subtree_job_ids(repo, job)
        if str(result.get("job_id")) not in allowed:
            raise ValueError(
                f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} not in wave {job.wave_id} {(job.source_domain or 'SEC').upper()} lane)"
            )
    texts = _finra_record_texts(result)
    if not any(locator in text for text in texts):
        raise ValueError(
            "record_evidence: ERR_PASSAGE_NOT_IN_SOURCE (the cited record values do not appear "
            "in the persisted FINRA tool result)"
        )
    payload = result.get("result")
    payload = payload if isinstance(payload, Mapping) else result
    dataset = payload.get("dataset_id") or payload.get("dataset")
    return finra_record_ref(
        tool_name=tool_name,
        record_identity=locator[:2000],
        dataset=dataset if isinstance(dataset, str) else None,
        source_uri=_svc_opt_str(data, "source_uri"),
        known_at=_finra_known_at(result),
        tool_result_id=ref_id,
    )


def _sec_record_texts(result: Mapping[str, object]) -> list[str]:
    """Citable texts of one persisted SEC structured result: each row JSON + envelope JSON."""
    texts: list[str] = []
    payload = result.get("result")
    payload = payload if isinstance(payload, Mapping) else result
    for key in ("filings", "transactions", "quarterly_eps", "facts", "rows", "records", "documents", "events"):
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    texts.append(_normalize_record_text(json.dumps(dict(row), sort_keys=True, default=str)))
                elif isinstance(row, (str, int, float)):
                    texts.append(_normalize_record_text(str(row)))
    texts.append(_normalize_record_text(json.dumps(dict(payload), sort_keys=True, default=str)))
    return [text for text in texts if text]


def _sec_record_known_at(data: Mapping[str, object], payload: Mapping[str, object]) -> str | None:
    """Caller-known_at wins (structured tools report their own filing dates); else payload scan."""
    for key in ("known_at", "filed_at", "accepted_at", "published_at", "as_of_date"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("known_at", "filed_at", "accepted_at", "published_at", "as_of_date"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sec_record_provenance(
    data: Mapping[str, object],
    job: Job | None = None,
    *,
    store: ResearchRepository | None = None,
    session_id: str | None = None,
) -> dict[str, JSONValue]:
    """SecRecordRef replayed against the persisted SEC structured tool result."""
    from .repository import ResearchRepository as _RR

    ref_id = _tool_result_ref(data, "SEC")
    locator = _normalize_record_text(_first_text(data, (*_PASSAGE_KEYS, "record_identity")))
    if not locator:
        raise ValueError(
            "record_evidence: ERR_RAW_SOURCE_REQUIRED (an SEC citation names the persisted "
            "tool result id plus the record values you are citing.)"
        )
    try:
        repo = store if store is not None else _RR()
        result = repo.get_tool_result(ref_id)
    except KeyError:
        raise ValueError(
            "record_evidence: ERR_RAW_SOURCE_REQUIRED (unknown persisted SEC tool result; "
            "cite the tool_result_id the kernel returned for the response you read.)"
        ) from None
    other = result.get("session_id")
    if isinstance(session_id, str) and session_id and isinstance(other, str) and other and other != session_id:
        raise ValueError(
            f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} belongs to session {other!r}, not {session_id!r})"
        )
    tool_name = str(result.get("tool_name") or "")
    if tool_name not in _SEC_REPLAY_TOOLS:
        raise ValueError(
            f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} is {tool_name!r}, not SEC records)"
        )
    if job is not None:
        allowed = _submit_subtree_job_ids(repo, job)
        if str(result.get("job_id")) not in allowed:
            raise ValueError(
                f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} not in wave {job.wave_id} {(job.source_domain or 'SEC').upper()} lane)"
            )
    texts = _sec_record_texts(result)
    if not any(locator in text for text in texts):
        raise ValueError(
            "record_evidence: ERR_PASSAGE_NOT_IN_SOURCE (the cited record values do not appear "
            "in the persisted SEC tool result)"
        )
    payload = result.get("result")
    payload = payload if isinstance(payload, Mapping) else result
    source_uri = _svc_opt_str(data, "source_uri")
    if source_uri is None and isinstance(payload, Mapping):
        raw_uri = payload.get("source") or payload.get("source_url")
        if isinstance(raw_uri, str) and raw_uri.strip():
            source_uri = raw_uri.strip()
    return sec_record_ref(
        tool_name=tool_name,
        record_identity=locator[:2000],
        dataset=str(payload.get("subject") or payload.get("ticker") or "") or None,
        source_uri=source_uri,
        known_at=_sec_record_known_at(data, payload if isinstance(payload, Mapping) else {}),
        tool_result_id=ref_id,
    )


def _web_result_texts(result: Mapping[str, object]) -> list[tuple[str, str, str, str, str, str]]:
    """(url, domain, title, published_at, retrieved_at, highlight) of persisted web results."""
    payload = result.get("result")
    payload = payload if isinstance(payload, Mapping) else result
    rows = payload.get("evidence")
    out: list[tuple[str, str, str, str, str, str]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            url = row.get("url")
            highlight = row.get("highlight")
            if not (isinstance(url, str) and url.strip() and isinstance(highlight, str) and highlight.strip()):
                continue
            out.append(
                (
                    url.strip(),
                    str(row.get("source_domain") or "").strip(),
                    str(row.get("title") or "").strip(),
                    str(row.get("published_at") or "").strip(),
                    str(row.get("retrieved_at") or "").strip(),
                    highlight.strip(),
                )
            )
    return out


def _web_provenance(
    data: Mapping[str, object],
    job: Job | None = None,
    *,
    store: ResearchRepository | None = None,
    session_id: str | None = None,
) -> dict[str, JSONValue]:
    """WebSourceRef replayed against the persisted search_web result."""
    from .repository import ResearchRepository as _RR

    ref_id = _tool_result_ref(data, "WEB")
    locator = _normalize_record_text(_first_text(data, (*_PASSAGE_KEYS, "excerpt")))
    if not locator:
        raise ValueError(
            "record_evidence: ERR_RAW_SOURCE_REQUIRED (a web citation names the persisted "
            "search_web result id plus the highlight text you are citing.)"
        )
    try:
        repo = store if store is not None else _RR()
        result = repo.get_tool_result(ref_id)
    except KeyError:
        raise ValueError(
            "record_evidence: ERR_RAW_SOURCE_REQUIRED (unknown persisted web tool result; "
            "cite the tool_result_id the kernel returned for the response you read.)"
        ) from None
    other = result.get("session_id")
    if isinstance(session_id, str) and session_id and isinstance(other, str) and other and other != session_id:
        raise ValueError(
            f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} belongs to session {other!r}, not {session_id!r})"
        )
    if str(result.get("tool_name") or "") != "search_web":
        raise ValueError(
            f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} is not a search_web result)"
        )
    if job is not None:
        allowed = _submit_subtree_job_ids(repo, job)
        if str(result.get("job_id")) not in allowed:
            raise ValueError(
                f"record_evidence: ERR_PROVENANCE_MISMATCH (tool result {ref_id!r} not in wave {job.wave_id} {(job.source_domain or 'SEC').upper()} lane)"
            )
    rows = _web_result_texts(result)
    url = data.get("source_record_id") or data.get("url") or data.get("source_uri")
    for row_url, domain, title, published_at, retrieved_at, highlight in rows:
        if url is not None and url != row_url:
            continue
        norm = _normalize_record_text(highlight)
        if locator and locator in norm:
            return web_source_ref(
                url=row_url,
                excerpt=highlight[:2000],
                title=title or None,
                domain=domain or None,
                published_at=published_at or None,
                retrieved_at=retrieved_at or None,
                tool_result_id=ref_id,
            )
    raise ValueError(
        "record_evidence: ERR_PASSAGE_NOT_IN_SOURCE (the cited excerpt does not appear "
        "in the persisted search_web result)"
    )


def _check_declared_ref(provenance: Mapping[str, object], *, accession: str | None, document: str | None) -> None:
    """A declared accession/document must be the one the handle materialized; disagreement fails closed."""
    if accession is not None and accession != provenance.get("accession_no"):
        raise ValueError(
            "record_evidence: ERR_PROVENANCE_MISMATCH (declared accession differs from the source_handle's filing)"
        )
    if document is not None and document != provenance.get("document_name"):
        raise ValueError(
            "record_evidence: ERR_PROVENANCE_MISMATCH "
            "(declared document_name differs from the source_handle's document)"
        )


def _handle_str(handle: Mapping[str, object], key: str, code: str) -> str:
    """Required non-blank string of one handle field."""
    value = handle.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record_evidence: {code} (source_handle[{key!r}] must be a non-empty string)")
    return value.strip()


def _handle_int(handle: Mapping[str, object], key: str, code: str, *, minimum: int = 0) -> int:
    """Required int field of one handle field (bools never count)."""
    value = handle.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"record_evidence: {code} (source_handle[{key!r}] must be an int >= {minimum})")
    return value


def _handle_window(handle: Mapping[str, object], code: str) -> tuple[int, int | None]:
    """(offset, max_chars) of one handle; max_chars is optional (whole-document reads)."""
    offset = _handle_int(handle, "offset", code)
    raw_max = handle.get("max_chars")
    if raw_max is None:
        return offset, None
    return offset, _handle_int(handle, "max_chars", code, minimum=1)


def _normalized_with_index(text: str) -> tuple[str, list[int]]:
    """Whitespace-normalized text plus the original index of every kept character."""
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for position, char in enumerate(text):
        if char.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            index.append(position)
            pending_space = False
        out.append(char)
        index.append(position)
    return "".join(out), index


def _locate_passage(window: str, locator: str) -> tuple[int, int]:
    """(start, end) of the locator inside the window by whitespace-normalized match; fails closed."""
    normalized_window, index = _normalized_with_index(window)
    normalized_locator = " ".join(locator.split())
    if not normalized_locator:
        raise ValueError("record_evidence: ERR_PASSAGE_NOT_IN_SOURCE (the cited passage is blank)")
    found = normalized_window.find(normalized_locator)
    if found < 0:
        raise ValueError(
            "record_evidence: ERR_PASSAGE_NOT_IN_SOURCE "
            "(the cited passage does not appear in the document window the handle names)"
        )
    return index[found], index[found + len(normalized_locator) - 1] + 1


@dataclass(frozen=True)
class _SecHandle:
    """Validated coordinates of one canonical source_handle (the reload recipe)."""

    basis: str
    accession: str
    document: str | None
    text_hash: str
    source_content_hash: str
    offset: int
    max_chars: int | None
    section: str | None
    query: str | None


def _handle_fields(handle: Mapping[str, object]) -> _SecHandle:
    """Validated identity + coordinates of one canonical handle; ERR_SEC_HANDLE_INVALID on any defect."""
    basis = _handle_str(handle, "basis", "ERR_SEC_HANDLE_INVALID")
    if basis not in ("raw", "rendered"):
        raise ValueError(
            f"record_evidence: ERR_SEC_HANDLE_INVALID (source_handle['basis'] must be raw|rendered, got {basis!r})"
        )
    accession_raw = _handle_str(handle, "accession_no", "ERR_SEC_HANDLE_INVALID")
    try:
        accession = normalize_accession(accession_raw)
    except ValueError:
        raise ValueError(
            "record_evidence: ERR_SEC_HANDLE_INVALID "
            f"(source_handle['accession_no'] {accession_raw!r} is not a SEC accession)"
        ) from None
    document_raw = handle.get("document_name")
    offset, max_chars = _handle_window(handle, "ERR_SEC_HANDLE_INVALID")
    return _SecHandle(
        basis=basis,
        accession=accession,
        document=document_raw.strip() if isinstance(document_raw, str) and document_raw.strip() else None,
        text_hash=_handle_str(handle, "text_hash", "ERR_SEC_HANDLE_INVALID"),
        source_content_hash=_handle_hex64(handle, "source_content_hash"),
        offset=offset,
        max_chars=max_chars,
        section=_handle_opt_str(handle, "section", "ERR_SEC_HANDLE_INVALID"),
        query=_handle_opt_str(handle, "query", "ERR_SEC_HANDLE_INVALID"),
    )


def _handle_opt_str(handle: Mapping[str, object], key: str, code: str) -> str | None:
    """Optional string field of one handle; absent/None stays None, any other type is invalid."""
    if handle.get(key) is None:
        return None
    return _handle_str(handle, key, code)


def _reload_handle_window(fields: _SecHandle, as_of: datetime | str | None) -> dict[str, object]:
    """Reload the handle's window from the archive; any failure is ERR_SEC_HANDLE_UNREADABLE."""
    from app.config import get_data_root
    from app.sec.documents import get_sec_document

    try:
        return get_sec_document(
            fields.accession,
            fields.document,
            as_of=as_of,  # type: ignore[arg-type] - filings._check_as_of accepts datetime
            offset=fields.offset,
            max_chars=fields.max_chars,
            section=fields.section,
            query=fields.query,
            raw=fields.basis == "raw",
            data_root=get_data_root(),
        )
    except Exception as exc:
        raise ValueError(
            f"record_evidence: ERR_SEC_HANDLE_UNREADABLE ({type(exc).__name__}: {str(exc)[:200]})"
        ) from exc


def _is_hex64(value: object) -> bool:
    """64-hex sha256 digest check for the source-bytes cross-check."""
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _handle_hex64(handle: Mapping[str, object], key: str) -> str:
    """Required 64-hex digest of one handle field; missing/malformed is ERR_SEC_HANDLE_INVALID."""
    value = handle.get(key)
    if not _is_hex64(value):
        raise ValueError(
            f"record_evidence: ERR_SEC_HANDLE_INVALID (source_handle[{key!r}] must be a sha256 hex digest)"
        )
    assert isinstance(value, str)
    return value


def _resolve_source_bytes(accession: str, document: str, reloaded: Mapping[str, object]) -> tuple[bytes, str]:
    """Exact source bytes: the immutable archive wins when present, else a live re-resolve."""
    raw_path = reloaded.get("raw_archive_path")
    if isinstance(raw_path, (str, Path)):
        try:
            payload = Path(raw_path).read_bytes()
        except OSError:
            raise ValueError(
                f"record_evidence: ERR_NO_SOURCE_BYTES (archived source bytes unreadable at {raw_path})"
            ) from None
        rep = reloaded.get("source_representation")
        return payload, rep if isinstance(rep, str) and rep else "source_bytes"
    return _exact_source_bytes(accession, document)


def materialize_sec_passage(
    handle: Mapping[str, object], locator: object, *, as_of: datetime | str | None = None
) -> MaterializedEvidenceSource:
    """Reload one canonical handle from the SEC archive and slice the cited passage out of it.

    Returns the materialized source (kernel passage + coordinates + window hash,
    full exact bytes, representation, source hash, canonical timing) for an
    observed fact. The caller's locator only says WHICH passage is meant; the
    stored text is always the archive's. Fail-closed codes: ERR_SEC_HANDLE_INVALID
    (malformed handle), ERR_SEC_HANDLE_UNREADABLE (archive cannot reload it),
    ERR_SEC_HANDLE_STALE (the reloaded window no longer hashes to the handle),
    ERR_PASSAGE_NOT_IN_SOURCE (the locator is not in that window),
    ERR_NO_SOURCE_BYTES (no exact document bytes to archive).
    """
    if not isinstance(handle, Mapping):
        raise ValueError(  # noqa: TRY004 - the public error contract pins ValueError, tests are oracle
            "record_evidence: ERR_SEC_HANDLE_INVALID (source_handle must be an object)"
        )
    fields = _handle_fields(handle)
    reloaded = _reload_handle_window(fields, as_of)
    window = reloaded.get("text")
    if not isinstance(window, str):
        raise ValueError(  # noqa: TRY004 - the public error contract pins ValueError, tests are oracle
            "record_evidence: ERR_SEC_HANDLE_UNREADABLE (the archive returned no document text)"
        )
    if sha256(window.encode("utf-8")).hexdigest() != fields.text_hash:
        raise ValueError(
            "record_evidence: ERR_SEC_HANDLE_STALE "
            "(the document window changed since the handle was issued; re-read the document "
            "with get_sec_document and cite the new handle)"
        )
    start, end = _locate_passage(window, locator if isinstance(locator, str) else "")
    provenance = sec_source_ref(
        accession_no=fields.accession,
        document_name=fields.document
        or _required_text(
            reloaded,
            ("document_name",),
            "record_evidence: ERR_SEC_HANDLE_UNREADABLE (no document identity)",
        ),
        passage=window[start:end],
        source_uri=reloaded.get("source_uri")
        if isinstance(reloaded.get("source_uri"), str)
        else handle.get("source_uri"),
        offset=fields.offset + start,
        end=fields.offset + end,
        basis=fields.basis,
        text_hash=fields.text_hash,
    )
    document = fields.document or str(reloaded.get("document_name") or "")
    source_bytes, byte_rep = _resolve_source_bytes(fields.accession, document, reloaded)
    if sha256(source_bytes).hexdigest() != fields.source_content_hash:
        raise ValueError(
            "record_evidence: ERR_SEC_HANDLE_STALE "
            "(the document revision changed since the handle was issued; re-read the document)"
        )
    return MaterializedEvidenceSource(
        provenance=provenance,
        source_bytes=source_bytes,
        representation=byte_rep,
        source_content_hash=fields.source_content_hash,
        source_url=reloaded.get("source_url") if isinstance(reloaded.get("source_url"), str) else None,
        known_at=_coerce_dt(reloaded.get("known_at")),
        filed_at=_coerce_dt(reloaded.get("filed_at")),
        retrieved_at=_coerce_dt(reloaded.get("retrieved_at")),
    )


# Coverage envelope schema: every key with the kind of value it must carry.
_COVERAGE_FIELD_KINDS: tuple[tuple[str, str], ...] = (
    *((key, "list of strings") for key in _ABSENCE_COVERAGE_LISTS),
    *((key, "bool") for key in _ABSENCE_COVERAGE_BOOLS),
)


def _coverage_field_ok(kind: str, value: object) -> bool:
    """Type predicate per coverage field kind (one table-driven check)."""
    if kind == "bool":
        return isinstance(value, bool)
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _require_coverage_envelope(cov: Mapping[str, object]) -> None:
    """Fail closed unless every coverage key is present and typed; the code stays ERR_COVERAGE_REQUIRED."""
    missing = [key for key, _ in _COVERAGE_FIELD_KINDS if key not in cov]
    if missing:
        raise ValueError(f"record_evidence: ERR_COVERAGE_REQUIRED (coverage missing {missing})")
    for key, kind in _COVERAGE_FIELD_KINDS:
        if not _coverage_field_ok(kind, cov[key]):
            raise ValueError(f"record_evidence: ERR_COVERAGE_REQUIRED (coverage[{key!r}] must be a {kind})")


def _absence_coverage(data: Mapping[str, object]) -> dict[str, JSONValue]:
    """Validated coverage envelope: what was searched, where, and whether the paging was exhausted."""
    raw = data.get("coverage", data.get("search_coverage"))
    if not isinstance(raw, Mapping):
        raise ValueError(  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
            "record_evidence: ERR_COVERAGE_REQUIRED (absence observation needs coverage "
            "{forms,dates,partitions,entities,docs,gaps,pagination_complete,complete})"
        )
    cov = dict(raw)
    _require_coverage_envelope(cov)
    return validate_json_mapping(cov, "<record_evidence>: 'coverage'")


def _absence_provenance(data: Mapping[str, object]) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    """SearchRunRef + coverage for an absence observation; an accession disproves nothing (fail closed)."""
    search_id = _first_text(data, ("search_id",))
    query = _first_text(data, ("query",))
    if search_id is None or query is None:
        raise ValueError(
            "record_evidence: ERR_COVERAGE_REQUIRED "
            "(absence observation needs search_id + query of the executed SearchRun)"
        )
    coverage = _absence_coverage(data)
    refs = [data.get(k) for k in _ACCESSION_KEYS]
    if any(isinstance(ref, str) and ref.strip() for ref in refs):
        raise ValueError(
            "record_evidence: ERR_PROVENANCE_MISMATCH "
            "(absence observation cites the SearchRun only; a filing accession proves no absence)"
        )
    return search_run_ref(search_id=search_id, query=query), coverage


def _live_evidence_job(store: ResearchRepository, session_id: str, job_id: str):
    """Load session + live running source/scout job, or raise the owning guard."""
    found = _require_session(store, session_id)
    job = _enforce_live_job(store, job_id)
    if job.session_id != found.session_id:
        raise ValueError(f"record_evidence: job {job_id!r} belongs to {job.session_id!r}")
    if job.status != "running":
        raise ValueError(f"record_evidence: job {job_id!r} status is {job.status!r} (running required)")
    if job.job_type not in ("source_agent", "scout"):
        raise ValueError(f"record_evidence: job {job_id!r} job_type {job.job_type!r} (source_agent|scout required)")
    if found.status in _session.TERMINAL_STATUSES:
        raise ValueError(f"record_evidence: session {session_id!r} status is {found.status!r} (terminal)")
    return found, job


def _evidence_id_name(data: Mapping[str, object], session_id: str, job: Job) -> tuple[str, str]:
    """Coerce evidence_id + source_name with job fallbacks."""
    evidence_id_raw = data.get("evidence_id")
    evidence_id = (
        evidence_id_raw
        if isinstance(evidence_id_raw, str) and evidence_id_raw.strip()
        else f"{session_id}:ev:{uuid.uuid4().hex[:8]}"
    )
    source_name_raw = data.get("source_name", data.get("source", job.source_domain or "pi"))
    source_name = source_name_raw if isinstance(source_name_raw, str) and source_name_raw.strip() else "pi"
    return evidence_id, source_name


def _str_tuple(raw: object) -> tuple[str, ...]:
    """Filter one supports/contradicts payload down to plain strings."""
    return tuple(s for s in raw if isinstance(s, str)) if isinstance(raw, (list, tuple)) else ()


def _evidence_ids_names(
    data: Mapping[str, object], session_id: str, job: Job
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Coerce evidence_id/source_name/supports/contradicts with job fallbacks."""
    evidence_id, source_name = _evidence_id_name(data, session_id, job)
    return evidence_id, source_name, _str_tuple(data.get("supports", ())), _str_tuple(data.get("contradicts", ()))


def _svc_explicit_kind(data: Mapping[str, object]) -> object:
    """Explicit record_kind from the item or its metadata; None when absent."""
    kind_raw: object = data.get("record_kind")
    if kind_raw is not None:
        return kind_raw
    meta: object = data.get("metadata")
    return meta.get("record_kind") if isinstance(meta, dict) else None


def _svc_record_kind(data: Mapping[str, object]) -> str:
    """Explicit record_kind (item or metadata) wins; everything recorded here is evidence."""
    kind_raw = _svc_explicit_kind(data)
    return kind_raw if isinstance(kind_raw, str) and kind_raw in ("discovery", "evidence") else "evidence"


def _svc_claim_text(claim: object, kind: str, claim_kind: str, evidence_id: str, content: str) -> str:
    """Claim text: the claim when non-blank, else a discovery label or content lead."""
    if isinstance(claim, str) and claim.strip():
        return claim[:2000]
    if kind == "discovery":
        return f"{claim_kind} {evidence_id}"
    return content[:500]


def _svc_opt_str(data: Mapping[str, object], key: str) -> str | None:
    """One optional string field; non-strings coerce to None."""
    raw = data.get(key)
    return raw if isinstance(raw, str) else None


def _svc_opt_float(data: Mapping[str, object], key: str) -> float | None:
    """One optional numeric field; non-numerics coerce to None."""
    raw = data.get(key)
    return float(raw) if isinstance(raw, (int, float)) else None


def _provenance_uri(provenance: Mapping[str, object], data: Mapping[str, object]) -> str | None:
    """Ingest source_ref: the kernel-materialized URI (SEC archive URL, FINRA source, web URL)."""
    if str(provenance.get("kind") or "") == "sec_source":
        # record_evidence replaced the model string with the kernel canonical URL.
        candidate = _svc_opt_str(data, "source_uri")
        if candidate:
            return candidate
    for key in ("source_uri", "url"):
        value = provenance.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _svc_opt_str(data, "source_uri")


def _provenance_record_id(provenance: Mapping[str, object], data: Mapping[str, object]) -> str | None:
    """Ingest source_ref: the kernel-materialized record id (accession, FINRA identity, web URL)."""
    for key in ("accession_no", "record_identity", "url"):
        value = provenance.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _svc_opt_str(data, "source_record_id")


def _provenance_time(caller: object, authoritative: object, *, kind: str = "") -> datetime | None:
    """Source time: FINRA/WEB rows use the kernel-replayed result time only; SEC uses caller time.

    A model-asserted known_at must never backdate a persisted result past PIT:
    FINRA known_at is the persisted publication/availability time, WEB known_at
    is the persisted published_at (never retrieval time, never caller time).
    Missing result time stays None so bounded sessions fail PIT_UNVERIFIED
    """
    if kind in ("finra_record", "web_source"):
        return _coerce_dt(authoritative)
    return _coerce_dt(authoritative) if authoritative is not None else _coerce_dt(caller)


def _build_evidence_record(
    data: Mapping[str, object],
    *,
    session_id: str,
    job_id: str,
    wave: int,
    evidence_id: str,
    source_name: str,
    subject: str,
    claim: object,
    content: str,
    supports: tuple[str, ...],
    contradicts: tuple[str, ...],
    metadata: dict[str, JSONValue],
    identity_key: str,
    claim_kind: str,
    provenance: dict[str, JSONValue],
) -> Evidence:
    """Assemble the Evidence row from coerced fields (no I/O)."""
    kind = _svc_record_kind(data)
    return Evidence(
        evidence_id=evidence_id,
        session_id=session_id,
        wave_id=wave,
        source_type=str(data.get("source_type", "pi")),
        source_name=source_name,
        subject=subject,
        claim_text=_svc_claim_text(claim, kind, claim_kind, evidence_id, content),
        content=content,
        content_hash=evidence_content_hash(content),
        retrieved_at=_coerce_dt(data.get("retrieved_at")) or utcnow(),
        source_uri=_provenance_uri(provenance, data),
        source_record_id=_provenance_record_id(provenance, data),
        published_at=_provenance_time(
            data.get("published_at"), provenance.get("published_at"), kind=str(provenance.get("kind") or "")
        ),
        known_at=_provenance_time(
            data.get("known_at"),
            provenance.get("published_at")
            if str(provenance.get("kind") or "") == "web_source"
            else provenance.get("known_at"),
            kind=str(provenance.get("kind") or ""),
        ),
        effective_at=_coerce_dt(data.get("effective_at")),
        job_id=job_id,
        agent_id=str(data.get("agent_id", "pi")),
        supports=supports,
        contradicts=contradicts,
        confidence=_svc_opt_float(data, "confidence"),
        quality=_svc_opt_str(data, "quality"),
        metadata={**metadata, "identity_key": identity_key, "claim_kind": claim_kind},
        superseded_by=_svc_opt_str(data, "superseded_by"),
        record_kind=kind,
        claim_kind=claim_kind,
        provenance=provenance,
    )


def _duplicate_response(store: ResearchRepository, session_id: str, record: Evidence) -> dict[str, JSONValue]:
    """Winner row for a lost identity race; raises when the winner is gone."""
    hit = store.find_evidence_by_identity(session_id, str(record.metadata.get("identity_key") or ""))
    if hit is None:
        raise ValueError(f"<research.sqlite>: evidence: duplicate identity_key {record.evidence_id!r}")
    return {"evidence_id": hit.get("evidence_id"), "accepted": False, "duplicate_of": hit.get("evidence_id")}


def _archive_materialized_source(
    session_id: str, evidence_id: str, retrieved_at: datetime, m: MaterializedEvidenceSource
) -> None:
    """Archive the full exact source bytes; raises on failure, never best-effort."""
    from app.sec.archive import archive_sec_document

    accession = m.provenance.get("accession_no")
    document = m.provenance.get("document_name")
    assert isinstance(accession, str) and accession
    url = (
        m.source_url
        or (m.provenance.get("source_uri") if isinstance(m.provenance.get("source_uri"), str) else "")
        or ""
    )
    archive_sec_document(
        accession,
        document if isinstance(document, str) and document else "primary",
        m.source_bytes,
        url=url,
        retrieved_at=retrieved_at.isoformat(),
        metadata={
            "evidence_id": evidence_id,
            "session_id": session_id,
            "source_content_hash": m.source_content_hash,
            "known_at": m.known_at.isoformat() if m.known_at is not None else None,
            "filed_at": m.filed_at.isoformat() if m.filed_at is not None else None,
            "representation": m.representation,
        },
    )


def _persist_evidence_record(
    store: ResearchRepository,
    session_id: str,
    job_id: str,
    found: ResearchSession,
    record: Evidence,
    *,
    source: MaterializedEvidenceSource,
) -> dict[str, JSONValue]:
    """PIT-gate + archive the full exact bytes before commit; missing bytes cannot arrive."""
    from .evidence import EvidenceLedger

    ledger = EvidenceLedger()
    for existing in store.list_evidence(session_id):
        try:
            from .evidence import evidence_from_dict

            ledger.append(evidence_from_dict(existing))
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
    ingest_evidence(
        ledger, record, as_of=found.as_of, on_reject=lambda event, payload: _emit(store, session_id, event, payload)
    )
    _archive_materialized_source(session_id, record.evidence_id, record.retrieved_at, source)
    stored = evidence_to_dict(record)
    stored_meta = stored.get("metadata")
    stored["metadata"] = {**(stored_meta if isinstance(stored_meta, dict) else {}), "source_bytes": "archived"}
    try:
        store.save_evidence(stored)
    except ValueError as exc:
        if "identity_key" not in str(exc):
            raise
        return _duplicate_response(store, session_id, record)
    _emit(
        store,
        session_id,
        "evidence.accepted",
        {
            "job_id": job_id,
            "evidence_id": record.evidence_id,
            "wave_id": record.wave_id,
            "source_domain": evidence_domain(record.provenance),
            "tool_name": record.provenance.get("tool_name"),
            "tool_result_id": record.provenance.get("tool_result_id"),
        },
    )
    try:
        from app.storage import raw_archive as _artifacts

        _artifacts.store_evidence_artifact(
            record.content.encode("utf-8"),
            url=record.source_uri or "",
            metadata={
                "evidence_id": record.evidence_id,
                "session_id": session_id,
                "source_bytes": "archived",
            },
        )
    except Exception:  # noqa: BLE001, S110 - best-effort artifact mirror, never blocks acceptance
        pass
    if record.evidence_id not in found.evidence_ids:
        store.save_session(
            replace(
                found,
                evidence_ids=[*found.evidence_ids, record.evidence_id],
                updated_at=utcnow(),
            )
        )
    return stored


def _persist_finra_web_record(
    store: ResearchRepository,
    session_id: str,
    job_id: str,
    found: ResearchSession,
    record: Evidence,
) -> dict[str, JSONValue]:
    """PIT-gate + commit a FINRA/WEB replay row; no SEC archive bytes exist to store."""
    from .evidence import EvidenceLedger

    ledger = EvidenceLedger()
    for existing in store.list_evidence(session_id):
        try:
            from .evidence import evidence_from_dict

            ledger.append(evidence_from_dict(existing))
        except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
            continue
    ingest_evidence(
        ledger, record, as_of=found.as_of, on_reject=lambda event, payload: _emit(store, session_id, event, payload)
    )
    stored = evidence_to_dict(record)
    try:
        store.save_evidence(stored)
    except ValueError as exc:
        if "identity_key" not in str(exc):
            raise
        return _duplicate_response(store, session_id, record)
    _emit(
        store,
        session_id,
        "evidence.accepted",
        {
            "job_id": job_id,
            "evidence_id": record.evidence_id,
            "wave_id": record.wave_id,
            "source_domain": evidence_domain(record.provenance),
            "tool_name": record.provenance.get("tool_name"),
            "tool_result_id": record.provenance.get("tool_result_id"),
        },
    )
    if record.evidence_id not in found.evidence_ids:
        store.save_session(
            replace(
                found,
                evidence_ids=[*found.evidence_ids, record.evidence_id],
                updated_at=utcnow(),
            )
        )
    return stored


def persist_tool_result(
    session_id: str,
    job_id: str,
    tool_name: str,
    tool_result_id: str | None,
    result: Mapping[str, object],
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Persist one staged tool result for FINRA/WEB/SEC-replay evidence; returns its id.

    Only staged source/scout jobs persist results, and only the tool the job's
    domain owns (FINRA tools on FINRA jobs, search_web on WEB jobs, SEC replay
    tools on SEC jobs). Document reads never persist: their evidence replays
    the archive instead. Resume is idempotent: a repeated persist of the same
    id keeps the first bytes and returns the same id.
    """
    from .agents.source_agent import is_finra_tool, is_sec_tool, is_web_tool

    store = _repo(repo)
    _found, job = _live_evidence_job(store, session_id, job_id)
    domain = (job.source_domain or "").upper()
    if domain == "FINRA":
        if not is_finra_tool(tool_name) or tool_name not in _FINRA_EVIDENCE_TOOLS:
            raise ValueError(f"persist_tool_result: tool {tool_name!r} cannot ground FINRA evidence for job {job_id!r}")
    elif domain == "WEB":
        if not is_web_tool(tool_name):
            raise ValueError(f"persist_tool_result: tool {tool_name!r} cannot ground WEB evidence for job {job_id!r}")
    elif domain == "SEC":
        if not is_sec_tool(tool_name) or tool_name not in _SEC_REPLAY_TOOLS:
            raise ValueError(f"persist_tool_result: tool {tool_name!r} cannot ground SEC replay for job {job_id!r}")
    else:
        raise ValueError(f"persist_tool_result: job {job_id!r} domain {job.source_domain!r} persists no tool results")
    rid = (
        tool_result_id.strip()
        if isinstance(tool_result_id, str) and tool_result_id.strip()
        else f"{session_id}:tr:{uuid.uuid4().hex[:8]}"
    )
    record: dict[str, JSONValue] = {
        "tool_result_id": rid,
        "session_id": session_id,
        "job_id": job_id,
        "tool_name": tool_name,
        "created_at": utcnow().isoformat(),
        "result": validate_json_mapping(dict(result), "<persist_tool_result>: 'result'"),
    }
    try:
        store.save_tool_result(record)
    except ValueError as exc:
        if "duplicate tool_result_id" not in str(exc):
            raise
    _emit(store, session_id, "tool_result.persisted", {"job_id": job_id, "tool_result_id": rid, "tool": tool_name})
    return {"tool_result_id": rid, "session_id": session_id, "job_id": job_id, "tool_name": tool_name}


def get_tool_result(
    tool_result_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Load one persisted staged tool result; raises ResearchNotFound when absent."""
    store = _repo(repo)
    try:
        return store.get_tool_result(tool_result_id)
    except KeyError:
        raise ResearchNotFound(f"unknown tool_result_id: {tool_result_id!r}") from None


def _evidence_live_wave(found: ResearchSession, job: Job, session_id: str) -> None:
    """Enforce the job wave is the live wave and not already frozen."""
    # ponytail: current_wave stays 0 on the Pi path until wave 2; floor at 1 via freeze count.
    live_wave = max(found.current_wave, len(found.freeze_ids) + 1, 1)
    if job.wave_id != live_wave:
        raise ValueError(f"record_evidence: job wave {job.wave_id!r} != current wave {live_wave!r}")
    if f"{session_id}:{job.wave_id}:freeze" in found.freeze_ids:
        raise ValueError(f"record_evidence: wave {job.wave_id} already frozen for {session_id!r}")


def _evidence_typed(data: dict[str, object], found: ResearchSession) -> tuple[dict[str, JSONValue], str]:
    """Validate metadata for one observed fact; return (metadata, subject).

    The search-coverage route never reaches here: ``record_evidence`` records an
    absence observation as a session coverage artifact before this point.
    """
    metadata: dict[str, JSONValue] = validate_json_mapping(data.get("metadata", {}), "record_evidence: 'metadata'")
    subject = str(data.get("subject", found.query[:120]))
    return metadata, subject


def _coverage_artifact_id(session_id: str, wave: int, search_id: str, query: str) -> str:
    """Deterministic coverage-artifact id: one scope, one artifact (a repeat is idempotent)."""
    digest = sha256("|".join((session_id, str(wave), search_id, query)).encode("utf-8")).hexdigest()[:16]
    return f"{session_id}:cov:{digest}"


def _record_absence_artifact(
    store: ResearchRepository,
    session_id: str,
    job_id: str,
    job: Job,
    data: Mapping[str, object],
) -> dict[str, JSONValue]:
    """Record a search-derived absence observation as a session coverage artifact.

    Not evidence: it carries no raw document, mints no evidence id, joins no
    freeze, and no citation path resolves it. The Director and the finalizer read
    it as coverage state (``research_read`` kind ``coverage``).
    """
    provenance, coverage = _absence_provenance(data)
    search_id = str(provenance["search_id"])
    query = str(provenance["query"])
    artifact_id = _coverage_artifact_id(session_id, job.wave_id, search_id, query)
    existing = {str(row.get("artifact_id")) for row in store.list_coverage_artifacts(session_id)}
    content, claim = _evidence_content(data)
    claim_text = _svc_claim_text(claim, "evidence", "absence_observation", artifact_id, content)
    record: dict[str, JSONValue] = {
        "artifact_id": artifact_id,
        "session_id": session_id,
        "job_id": job_id,
        "wave_id": job.wave_id,
        "claim_kind": "absence_observation",
        "claim_text": claim_text,
        "search_id": search_id,
        "query": query,
        "coverage": coverage,
        "recorded_at": utcnow().isoformat(),
    }
    if artifact_id not in existing:
        store.save_coverage_artifact(record)
        _emit(
            store,
            session_id,
            "coverage.recorded",
            {
                "job_id": job_id,
                "artifact_id": artifact_id,
                "search_id": search_id,
                "query": query,
            },
        )
    return {
        "recorded_as": "coverage_artifact",
        "artifact_id": artifact_id,
        "citable": False,
        "accepted": artifact_id not in existing,
        "duplicate_of": artifact_id if artifact_id in existing else None,
        "session_id": session_id,
        "claim_kind": "absence_observation",
        "claim_text": claim_text,
        "search_id": search_id,
        "query": query,
        "coverage": coverage,
        "evidence_ids": [],
    }


def record_evidence(
    session_id: str,
    job_id: str,
    item: Mapping[str, object],
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Validate (PIT/claim-kind/provenance/IDs) + persist one finding.

    An ``absence_observation`` item is a search-scope record: it is validated the
    same way and then persisted as a session coverage artifact (never a ledger
    row, never a citable evidence id).
    """
    if not isinstance(item, Mapping):
        raise ValueError("record_evidence: 'item' must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    store = _repo(repo)
    found, job = _live_evidence_job(store, session_id, job_id)
    data = dict(item)
    if _claim_kind(data) == "absence_observation":
        _evidence_live_wave(found, job, session_id)
        return _record_absence_artifact(store, session_id, job_id, job, data)
    content, claim = _evidence_content(data)
    wave = _evidence_wave(data, job.wave_id)
    _evidence_live_wave(found, job, session_id)
    evidence_id, source_name, supports, contradicts = _evidence_ids_names(data, session_id, job)
    metadata, subject = _evidence_typed(data, found)
    observed = _observed_provenance(data, job, store=store, session_id=session_id, as_of=found.as_of)
    if isinstance(observed, MaterializedEvidenceSource):
        materialized = observed
        provenance = materialized.provenance
        identity_key = _evidence_identity(data, claim, subject, provenance)
        hit = store.find_evidence_by_identity(session_id, identity_key)
        if hit is not None:
            return {"evidence_id": hit.get("evidence_id"), "accepted": False, "duplicate_of": hit.get("evidence_id")}
        data["known_at"] = materialized.known_at.isoformat() if materialized.known_at else None
        data["published_at"] = materialized.filed_at.isoformat() if materialized.filed_at else None
        data["retrieved_at"] = materialized.retrieved_at.isoformat() if materialized.retrieved_at else None
        data["source_uri"] = materialized.source_url or (
            materialized.provenance.get("source_uri")
            if isinstance(materialized.provenance.get("source_uri"), str)
            else None
        )
        record = _build_evidence_record(
            data,
            session_id=session_id,
            job_id=job_id,
            wave=wave,
            evidence_id=evidence_id,
            source_name=source_name,
            subject=subject,
            claim=claim,
            content=content,
            supports=supports,
            contradicts=contradicts,
            metadata=metadata,
            identity_key=identity_key,
            claim_kind="observed_fact",
            provenance=provenance,
        )
        return _persist_evidence_record(store, session_id, job_id, found, record, source=materialized)
    provenance = observed
    identity_key = _evidence_identity(data, claim, subject, provenance)
    hit = store.find_evidence_by_identity(session_id, identity_key)
    if hit is not None:
        return {"evidence_id": hit.get("evidence_id"), "accepted": False, "duplicate_of": hit.get("evidence_id")}
    record = _build_evidence_record(
        data,
        session_id=session_id,
        job_id=job_id,
        wave=wave,
        evidence_id=evidence_id,
        source_name=source_name,
        subject=subject,
        claim=claim,
        content=content,
        supports=supports,
        contradicts=contradicts,
        metadata=metadata,
        identity_key=identity_key,
        claim_kind="observed_fact",
        provenance=provenance,
    )
    return _persist_finra_web_record(store, session_id, job_id, found, record)


def _submit_str_list_field(cov: dict[str, object], key: str, *, required: bool = False) -> list[str]:
    """Validated string list for one coverage key (missing optional key means [])."""
    raw = cov.get(key)
    if raw is None:
        if required:
            raise ValueError(f"submit_source_result: ERR_COVERAGE_REQUIRED (coverage[{key!r}] required)")
        return []
    if not isinstance(raw, list) or any(not isinstance(v, str) for v in raw):
        raise ValueError(f"submit_source_result: ERR_COVERAGE_REQUIRED (coverage[{key!r}] must be a list of strings)")
    return list(raw)


_SUFFICIENCY_REQUIRED_KEYS: tuple[str, ...] = (
    "major_entities_investigated",
    "relationship_types_checked",
    "forms_examined",
    "exhibits_examined",
    "material_open_questions",
)
_SUFFICIENCY_RESIDUAL_KEYS: tuple[str, ...] = (
    "material_open_questions",
    "major_entities_missing",
    "remaining_branches",
    "routes_unsearched",
)
_SUFFICIENCY_BRANCH_KEYS: tuple[str, ...] = ("search_runs", "covered_branches")
# Per-domain work keys: the FINRA/WEB desks prove scope with their own
# artifacts (datasets/tickers/windows, queries/results), not SEC filings.
_SUFFICIENCY_FINRA_KEYS: tuple[str, ...] = (
    "datasets_queried",
    "tickers_covered",
    "settlement_windows_covered",
    "dataset_reads",
    "covered_branches",
)
_SUFFICIENCY_WEB_KEYS: tuple[str, ...] = (
    "semantic_branches_covered",
    "queries_executed",
    "results_inspected",
    "covered_branches",
)


def _sufficient_residuals(cov: dict[str, object], unresolved: list[str] | None) -> list[str]:
    """Non-blank residual questions/branches/routes across the sufficiency keys + the argument."""
    remaining = [q for key in _SUFFICIENCY_RESIDUAL_KEYS for q in _submit_str_list_field(cov, key) if q.strip()]
    remaining.extend(q for q in (unresolved or []) if isinstance(q, str) and q.strip())
    return remaining


def _sufficient_required(cov: dict[str, object], domain: str = "SEC") -> None:
    """The full structured envelope for one source domain's slice."""
    work = (
        _SUFFICIENCY_FINRA_KEYS
        if domain == "FINRA"
        else _SUFFICIENCY_WEB_KEYS
        if domain == "WEB"
        else (*_SUFFICIENCY_REQUIRED_KEYS, *_SUFFICIENCY_BRANCH_KEYS)
    )
    missing = [key for key in work if not isinstance(cov.get(key), list)]
    if missing:
        raise ValueError(f"submit_source_result: ERR_COVERAGE_REQUIRED (sufficient coverage missing {missing})")
    for key in work if domain in ("FINRA", "WEB") else _SUFFICIENCY_REQUIRED_KEYS:
        _submit_str_list_field(cov, key, required=True)
    if domain in ("FINRA", "WEB"):
        if not [b for b in _submit_str_list_field(cov, "covered_branches", required=True) if b.strip()]:
            raise ValueError(
                "submit_source_result: ERR_COVERAGE_REQUIRED (sufficient needs non-empty covered_branches)"
            )
        return
    if not _submit_str_list_field(cov, "search_runs", required=True):
        raise ValueError("submit_source_result: ERR_COVERAGE_REQUIRED (sufficient needs non-empty search_runs)")
    if not [b for b in _submit_str_list_field(cov, "covered_branches", required=True) if b.strip()]:
        raise ValueError("submit_source_result: ERR_COVERAGE_REQUIRED (sufficient needs non-empty covered_branches)")


def _submit_sufficient_gate(cov: dict[str, object], unresolved: list[str] | None, domain: str = "SEC") -> None:
    """sufficient claims a fully covered slice of its own source domain or it fails closed.

    SEC needs the filing envelope (entities/relationships/forms/exhibits/search
    runs/covered branches); FINRA needs datasets/tickers/windows/reads/branches;
    WEB needs semantic branches/queries/results/branches — each with every
    residual empty. There is no envelope-shape bypass: evidence row count is
    never sufficiency. The Director coverage challenge stays the second layer
    for incomplete dossiers.
    """
    if cov.get("useful_for_question") != "sufficient":
        return
    _sufficient_required(cov, domain)
    remaining = _sufficient_residuals(cov, unresolved)
    if remaining:
        raise ValueError(
            f"submit_source_result: ERR_COVERAGE_INCOMPLETE (sufficient with remaining branches/questions: {remaining[:5]})"
        )


def _submit_coverage_ids(
    coverage: dict[str, object] | None, evidence_ids: list[str] | None
) -> tuple[dict[str, object], object, list[str]]:
    """Validate the coverage envelope + evidence list; return (cov, useful, ids).

    Envelope only; the sufficiency gate runs in submit_source_result after the
    dangling-ref check so citing unknown ids still reports ERR_EVIDENCE_NOT_FOUND.
    """
    cov = dict(coverage or {})
    if "useful_for_question" not in cov:
        raise ValueError(
            "submit_source_result: ERR_COVERAGE_REQUIRED (coverage.useful_for_question required: sufficient|insufficient)"
        )
    useful = cov.get("useful_for_question")
    if useful not in ("sufficient", "insufficient"):
        raise ValueError("submit_source_result: coverage useful_for_question must be sufficient|insufficient")
    ids = list(evidence_ids or [])
    if not ids and useful != "insufficient":
        raise ValueError("submit_source_result: ERR_EMPTY_RESULT (no evidence with sufficient coverage)")
    return cov, useful, ids


def _submit_live_job(store: ResearchRepository, job_id: str):
    """Load the open running source job + session, or raise the owning guard."""
    job = _enforce_live_job(store, job_id)
    if job.status != "running":
        raise ValueError(f"submit_source_result: ERR_JOB_CLOSED (job {job_id!r} status is {job.status!r})")
    if job.job_type not in ("source_agent", "scout"):
        raise ValueError(
            f"submit_source_result: job {job_id!r} job_type {job.job_type!r} (source_agent|scout required)"
        )
    found = _require_session(store, job.session_id)
    if found.status in _session.TERMINAL_STATUSES:
        raise ValueError(f"submit_source_result: session {found.session_id!r} status is {found.status!r} (terminal)")
    return job, found


_SUBMIT_COVERAGE_KEYS = (
    "resolved",
    "partially_resolved",
    "unresolved",
    "source_limitations",
    "dates",
    "partitions",
    "docs",
    "gaps",
    "major_entities_investigated",
    "major_entities_missing",
    "relationship_types_checked",
    "forms_examined",
    "exhibits_examined",
    "material_open_questions",
    "remaining_branches",
    "routes_unsearched",
    "search_runs",
    "covered_branches",
    "datasets_queried",
    "tickers_covered",
    "settlement_windows_covered",
    "dataset_reads",
    "semantic_branches_covered",
    "queries_executed",
    "results_inspected",
)


def _submit_subtree_job_ids(store: ResearchRepository, job: Job) -> set[str]:
    """Job ids in the submitting wave-domain subtree: same wave + same source domain.

    OMP scouts run as sibling source jobs under the Director's context block
    (no parent threading), so the ledger must span the wave/domain lane, not a
    parent chain. Cross-domain isolation holds because the domain must match.
    """
    domain = (job.source_domain or "SEC").upper()
    return {
        other.job_id
        for other in store.list_jobs(job.session_id)
        if other.wave_id == job.wave_id
        and (other.source_domain or "SEC").upper() == domain
        and other.job_type in ("source_agent", "scout")
    }


def _submit_ledger_ids(store: ResearchRepository, job: Job) -> set[str]:
    """Substantive evidence ids in this wave-domain subtree (discovery rows are never citable)."""
    allowed = _submit_subtree_job_ids(store, job)
    return {
        str(r.get("evidence_id"))
        for r in store.list_evidence(job.session_id)
        if _freeze_row_kind(r) == "evidence" and r.get("job_id") in allowed
    }


def _submit_require_refs(ledger_ids: set[str], ids: list[str]) -> None:
    """Every cited evidence id must already be persisted in this source subtree."""
    missing = [e for e in ids if e not in ledger_ids]
    if missing:
        raise ValueError(f"submit_source_result: ERR_EVIDENCE_NOT_FOUND {missing[:3]}")


def _submit_coverage_merge(coverage: dict[str, object], cov: dict[str, object] | None, key: str) -> None:
    """Copy one caller-supplied string list into the coverage envelope."""
    values = (cov or {}).get(key)
    if isinstance(values, list) and all(isinstance(v, str) for v in values):
        coverage[key] = list(values)


def _sec_search_ledger(search_ids: Sequence[str]) -> tuple[dict[str, str], bool]:
    """No persisted SEC search ledger remains; always empty but readable."""
    # Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind live readers, never here.
    del search_ids
    return {}, True


def _submit_search_run_warnings(cov: dict[str, object]) -> list[str]:
    """Unknown search_run ids against the persisted SEC ledger; advisory only, never a failure."""
    runs = _submit_str_list_field(cov, "search_runs")
    known, readable = _sec_search_ledger(runs)
    if not readable:
        return []
    return [f"search_runs id not found in the persisted SEC search ledger: {sid!r}" for sid in runs if sid not in known]


def _submit_coverage_dict(cov: dict[str, object] | None, useful: object, job: Job | None = None) -> dict[str, object]:
    """Coverage envelope: default coverage plus caller-supplied string lists.

    New sufficiency keys (major_entities_investigated, relationship_types_checked,
    forms_examined, exhibits_examined, material_open_questions, search_runs,
    covered_branches + remaining-branch lists) ride alongside the existing
    resolution/negative-scope keys. ``source_domain``/``source_sufficiency``
    scope the verdict to this source (global completion belongs to the Director);
    ``useful_for_question`` persists so the director challenge + freeze telemetry
    can read it back from the stored dossier.
    """
    from .dossiers import default_coverage

    coverage = {
        **default_coverage(),
        "complete": useful == "sufficient",
        "useful_for_question": useful,
        "source_domain": (job.source_domain or "SEC").upper() if job is not None else "SEC",
        "source_sufficiency": useful,
    }
    for key in _SUBMIT_COVERAGE_KEYS:
        _submit_coverage_merge(coverage, cov, key)
    return coverage


def _distinct_values(rows: Sequence[object], key: str) -> list[str]:
    """Sorted distinct non-blank ``key`` values across each row's top level, metadata, and provenance."""
    found: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for source in (row, row.get("metadata"), row.get("provenance")):
            value = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
    return sorted(found)


def _coverage_strs(cov: Mapping[str, object], *keys: str) -> list[str]:
    """Sorted distinct strings across several persisted coverage list keys."""
    values: set[str] = set()
    for key in keys:
        raw = cov.get(key)
        if isinstance(raw, list):
            values.update(v.strip() for v in raw if isinstance(v, str) and v.strip())
    return sorted(values)


def _telemetry_coverage(
    store: ResearchRepository, session_id: str, extra: Mapping[str, object] | None, unresolved: Sequence[str]
) -> dict[str, object]:
    """Persisted dossier coverage overlaid with an in-flight submission (+ its unresolved argument)."""
    raw = _latest_dossier(store, session_id).get("coverage")
    cov = dict(raw) if isinstance(raw, Mapping) else {}
    if extra:
        cov.update({k: v for k, v in extra.items() if v is not None})
    pending = [q.strip() for q in unresolved if isinstance(q, str) and q.strip()]
    if pending:
        prior = cov.get("unresolved")
        cov["unresolved"] = [*(prior if isinstance(prior, list) else []), *pending]
    return cov


def _journal_search_activity(store: ResearchRepository, session_id: str) -> tuple[int, list[str]]:
    """(search-tool calls, query strings) from persisted journal tool events."""
    calls = 0
    queries: list[str] = []
    for event in store.list_events(session_id):
        if event.event_type not in ("tool.completed", "tool.skipped"):
            continue
        raw_payload = event.payload
        payload: Mapping[str, object] = raw_payload if isinstance(raw_payload, Mapping) else {}
        if payload.get("tool") != "search_sec_filings":
            continue
        calls += 1
        raw_args = payload.get("args")
        args: Mapping[str, object] = raw_args if isinstance(raw_args, Mapping) else {}
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return calls, queries


def _journal_blocked_novelty(store: ResearchRepository, session_id: str) -> tuple[int, int]:
    """(duplicate_actions_blocked, zero_novelty_actions) from persisted loop/novelty journal events."""
    blocked = 0
    zero_novelty = 0
    for event in store.list_events(session_id):
        if event.event_type == "research_loop_detected":
            blocked += 1
            continue
        if event.event_type != "wave.novelty":
            continue
        action = event.payload.get("zero_novelty_actions") if isinstance(event.payload, Mapping) else None
        zero_novelty += action if isinstance(action, int) and not isinstance(action, bool) else 0
    return blocked, zero_novelty


# Coverage keys that still hold open branches; everything here must be empty for a sufficient slice.
_TELEMETRY_RESIDUAL_KEYS: tuple[str, ...] = (*_SUFFICIENCY_RESIDUAL_KEYS, "unresolved")


def _ledger_queries(ledger: Mapping[str, str], search_ids: Sequence[str], readable: bool) -> set[str]:
    """Persisted query text for the coverage's search ids (empty when the ledger is unreadable)."""
    if not (search_ids and readable):
        return set()
    return {ledger[sid].strip() for sid in search_ids if ledger.get(sid, "").strip()}


def _telemetry_search_signals(
    store: ResearchRepository, session_id: str, cov: Mapping[str, object]
) -> tuple[object, object, list[str]]:
    """(searches_count, queries_attempted, gaps) from the persisted SEC ledger + journal.

    Journal search events are persisted activity too: they fold into the query
    set whenever they exist, and back the count when the coverage carries no
    search ids. A key with no persisted source anywhere is ``None`` + a gap.
    """
    search_ids = _coverage_strs(cov, "search_runs")
    ledger, readable = _sec_search_ledger(search_ids)
    journal_calls, journal_queries = _journal_search_activity(store, session_id)
    resolved_queries = _ledger_queries(ledger, search_ids, readable) | set(journal_queries)
    queries: object = sorted(resolved_queries) if resolved_queries else None
    searches: object = len(search_ids) if search_ids else (journal_calls if journal_calls else None)
    gaps = [name for name, value in (("queries_attempted", queries), ("searches_count", searches)) if value is None]
    return searches, queries, gaps


def _sec_source_documents(rows: Sequence[Mapping[str, object]]) -> set[tuple[str, str]]:
    """(accession, document) pairs held by the rows' raw SEC provenance."""
    return {
        (str(prov.get("accession_no")), str(prov.get("document_name")))
        for prov in (row.get("provenance") for row in rows)
        if isinstance(prov, Mapping) and prov.get("kind") == "sec_source"
    }


def _telemetry_document_signals(rows: Sequence[Mapping[str, object]]) -> tuple[object, object, list[str]]:
    """(filings_opened, documents_opened, gaps) from the raw-source rows.

    Zero rows is a true zero; rows that carry no usable identity are a gap, and
    ``raw_documents_used`` shares the document count (one opened document, one
    raw document used).
    """
    accessions = _distinct_values(rows, "accession_no") or _distinct_values(rows, "source_record_id")
    documents = _sec_source_documents(rows)
    gaps: list[str] = []
    filings: object = len(accessions)
    if rows and not accessions:
        filings = None
        gaps.append("filings_opened")
    opened: object = len(documents)
    if rows and not documents:
        opened = None
        gaps.extend(("documents_opened", "raw_documents_used"))
    return filings, opened, gaps


def _telemetry_branch_signals(cov: Mapping[str, object]) -> tuple[object, object, list[str]]:
    """(covered_branches, branches_remaining, gaps); no persisted coverage is a gap, never a zero."""
    if not cov:
        return None, None, ["branches_covered", "branches_remaining"]
    return _coverage_strs(cov, "covered_branches"), _coverage_strs(cov, *_TELEMETRY_RESIDUAL_KEYS), []


def _telemetry_relationship_count(store: ResearchRepository, session_id: str) -> int:
    """Persisted dossier relationships; unreadable storage reads as none, never a guessed count."""
    relationships: list[object] = []
    try:
        for dossier in store.list_dossiers(session_id):
            rels = dossier.get("relationships")
            if isinstance(rels, list):
                relationships.extend(r for r in rels if isinstance(r, Mapping))
    except Exception:  # noqa: BLE001 - storage degradation omits the key, never fabricates a count
        return 0
    return len(relationships)


def _derive_telemetry(
    store: ResearchRepository,
    session_id: str,
    extra: Mapping[str, object] | None = None,
    unresolved: Sequence[str] = (),
) -> dict[str, object]:
    """Coverage telemetry derived only from persisted rows/journal; nothing model-reported.

    Every key is counted from the evidence rows, the persisted coverage, the
    session row, or the journal. A key whose source is missing (no provenance
    on any row, an unresolvable SEC search ledger entry, no persisted coverage)
    is reported as ``null`` and named in ``telemetry_gaps`` — never invented as
    a zero.
    """
    found = store.get_session(session_id)
    rows = [r for r in store.list_evidence(session_id) if _freeze_row_kind(r) != "discovery"]
    cov = _telemetry_coverage(store, session_id, extra, unresolved)
    searches, queries, search_gaps = _telemetry_search_signals(store, session_id, cov)
    filings, opened, document_gaps = _telemetry_document_signals(rows)
    covered, remaining, branch_gaps = _telemetry_branch_signals(cov)
    gaps: list[str] = [*search_gaps, *document_gaps, *branch_gaps]

    blocked, zero_novelty = _journal_blocked_novelty(store, session_id)
    telemetry: dict[str, object] = {
        "searches_count": searches,
        "queries_attempted": queries,
        "forms_examined": sorted(
            set(_distinct_values(rows, "form")) | set(_coverage_strs(cov, "forms_examined", "forms"))
        ),
        "entities_investigated": sorted(
            set(_distinct_values(rows, "subject")) | set(_coverage_strs(cov, "major_entities_investigated", "entities"))
        ),
        "filings_opened": filings,
        "documents_opened": opened,
        "raw_documents_used": opened,
        "evidence_records": len(rows),
        "branches_covered": covered,
        "branches_remaining": remaining,
        "waves": len(found.freeze_ids),
        "committee_rounds": len(found.committee_runs),
        "duplicate_actions_blocked": blocked,
        "zero_novelty_actions": zero_novelty,
    }
    material_relationships = _telemetry_relationship_count(store, session_id)
    if material_relationships:
        telemetry["material_relationships_found"] = material_relationships
    telemetry["telemetry_gaps"] = sorted(set(gaps))
    return telemetry


def _submit_telemetry(
    store: ResearchRepository, session_id: str, cov: dict[str, object] | None, unresolved: list[str] | None
) -> dict[str, JSONValue]:
    """Telemetry merged into the completed job result: this submission's envelope + persisted rows."""
    return validate_json_mapping(_derive_telemetry(store, session_id, cov, unresolved or ()), "<submit:telemetry>")


def _submit_persist_dossier(
    store: ResearchRepository,
    job: Job,
    found: ResearchSession,
    dossier_id: str,
    coverage: dict[str, object],
    ids: list[str],
    unresolved: list[str] | None,
) -> None:
    """Persist the SEC dossier + link it to the session; duplicate saves are no-ops."""
    from .dossiers import create_dossier
    from .dossiers.sec import dossier_to_dict

    dossier = create_dossier(
        dossier_id=dossier_id,
        session_id=job.session_id,
        wave_id=job.wave_id,
        subject=found.query[:120],
        coverage=coverage,
        findings=[{"text": "source result", "evidence_ids": ids}] if ids else [],
        open_questions=list(unresolved or []),
    )
    try:
        store.save_dossier(dossier_to_dict(dossier))
    except ValueError:
        pass
    cur = store.get_session(job.session_id)
    if dossier_id not in cur.dossier_ids:
        store.save_session(replace(cur, dossier_ids=[*cur.dossier_ids, dossier_id], updated_at=utcnow()))


def _submit_dossier(
    store: ResearchRepository,
    job: Job,
    found: ResearchSession,
    useful: object,
    ids: list[str],
    unresolved: list[str] | None,
    cov: dict[str, object] | None = None,
) -> tuple[str, set[str]]:
    """Validate refs + persist the source dossier (idempotent); return (dossier_id, ledger_ids)."""
    ledger_ids = _submit_ledger_ids(store, job)
    _submit_require_refs(ledger_ids, ids)
    domain = (job.source_domain or "SEC").lower()
    dossier_id = f"{job.session_id}:{job.wave_id}:{domain}"
    _submit_persist_dossier(store, job, found, dossier_id, _submit_coverage_dict(cov, useful, job), ids, unresolved)
    return dossier_id, ledger_ids


def submit_source_result(
    job_id: str,
    coverage: dict[str, object] | None = None,
    evidence_ids: list[str] | None = None,
    unresolved_questions: list[str] | None = None,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Complete one running source job with validated coverage; evidence stays mutation-only.

    sufficient requires its own domain's full envelope (SEC filings, FINRA
    datasets, WEB queries) and no residual branches or questions — never just
    N evidence rows. Validates job open + deadline live, every evidence id
    exists in this session, then completes the job with a coverage/result
    payload. Returns {job_status: completed, dossier-ish refs, telemetry,
    warnings}. Terminal reuse fails closed.
    """
    store = _repo(repo)
    job, found = _submit_live_job(store, job_id)
    cov, _, ids = _submit_coverage_ids(coverage, evidence_ids)
    _submit_require_refs(_submit_ledger_ids(store, job), ids)
    _submit_sufficient_gate(cov, unresolved_questions, (job.source_domain or "SEC").upper())
    warnings = _submit_search_run_warnings(cov)
    dossier_id, _ = _submit_dossier(store, job, found, cov.get("useful_for_question"), ids, unresolved_questions, cov)
    telemetry = _submit_telemetry(store, job.session_id, cov, unresolved_questions)
    done = _jobs.complete_job(job, result={"coverage": cov, "evidence_ids": ids, "telemetry": telemetry})
    store.save_job(done)
    _emit(
        store,
        job.session_id,
        "job.completed",
        {"job_id": job.job_id, "telemetry": dict(telemetry), "warnings": warnings},
    )
    if warnings:
        _emit(store, job.session_id, "coverage.warning", {"job_id": job.job_id, "warnings": warnings})
    transition_job_completed(job.session_id, job.job_id, repo=store)
    out: dict[str, JSONValue] = {
        "job_status": done.status,
        "job_id": done.job_id,
        "dossier_id": dossier_id,
        "evidence_ids": validate_json_value(ids, "<submit>"),
        "telemetry": telemetry,
        "warnings": validate_json_value(warnings, "<submit>"),
    }
    return out


def list_research(
    limit: int = 20,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> list[dict[str, object]]:
    """Newest-first session summaries (read-only; empty when no DB yet)."""
    return _repo(repo).list_sessions(limit=limit)


def retry_job(
    job_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
    owner: str = "cli-retry",
) -> dict[str, JSONValue]:
    """Enqueue a replacement for a failed/cancelled/timed-out job; else no-op."""
    store = _repo(repo)
    job = _require_job(store, job_id)
    if job.status not in ("failed", "cancelled", "timed_out"):
        return job.to_dict()
    found = _require_session(store, job.session_id)
    existing = store.list_jobs(job.session_id)
    updated, replacement = _jobs.create_job(
        found,
        existing,
        job_type=job.job_type,
        owner=owner,
        wave_id=job.wave_id,
        parent_job_id=job.parent_job_id,
        source_domain=job.source_domain,
        model=job.model,
        token_budget=job.token_budget,
        tool_budget=job.tool_budget,
        child_budget=None,  # re-derive from current policy; the stored echo is not an override
    )
    store.save_session(updated)
    store.save_job(replacement)
    return replacement.to_dict()


def _freeze_context(store: ResearchRepository, found: ResearchSession) -> tuple[str, str, list[str], int, str]:
    """Resolve (sid, fid, evidence ids, wave, as_of) for the latest freeze."""
    sid = found.session_id
    if not found.freeze_ids:
        raise ValueError(f"committee: session {sid!r} has no freeze")
    fid = found.freeze_ids[-1]
    try:
        frozen = store.get_freeze(fid)
    except KeyError:
        raise ValueError(f"committee: unknown freeze_id: {fid!r}") from None
    raw_ids = frozen.get("evidence_ids")
    ids = [e for e in raw_ids if isinstance(e, str)] if isinstance(raw_ids, list) else []
    wave_raw = frozen.get("wave_id")
    wave = wave_raw if isinstance(wave_raw, int) and not isinstance(wave_raw, bool) and wave_raw >= 1 else 1
    as_of = found.as_of.isoformat() if isinstance(found.as_of, datetime) else "unbounded"
    return sid, fid, ids, wave, as_of


def _str_list(raw: object) -> list[str]:
    """Filter one unknowns/changes payload down to plain strings."""
    return [u for u in raw if isinstance(u, str)] if isinstance(raw, list) else []


def _freeze_run_jobs(found: ResearchSession, fid: str) -> list[str]:
    """Collect committee-run job ids recorded for this freeze."""
    run_jobs: list[str] = []
    for entry in found.committee_runs:
        if isinstance(entry, dict) and entry.get("freeze_id") == fid:
            got = entry.get("jobs", [])
            if isinstance(got, list):
                run_jobs.extend(j for j in got if isinstance(j, str))
    return run_jobs


def _committee_results_by_role(
    store: ResearchRepository, found: ResearchSession, fid: str
) -> dict[str, dict[str, JSONValue]]:
    """Latest-freeze completed trio results keyed by role; skips dangling/open/uncited."""
    by_role: dict[str, dict[str, JSONValue]] = {}
    for jid in _freeze_run_jobs(found, fid):
        try:
            job = store.get_job(jid)
        except KeyError:
            continue
        res = job.result
        if job.status != "completed" or not isinstance(res, dict):
            continue
        if not isinstance(res.get("claims"), list):
            continue
        by_role.setdefault(job.job_type, res)
    return by_role


def _build_stock(
    role_res: dict[str, JSONValue], *, sid: str, wave: int, fid: str, ids: list[str], as_of: str, question: str
) -> StockbotAnalysis:
    """Parse the stockbot envelope into its typed analysis."""
    from .agents import parse_committee_envelope
    from .agents.stockbot import StockbotAnalysis

    env = parse_committee_envelope(json.dumps(dict(role_res)), frozen=ids, agent="stockbot")
    prose = "\n".join(c.text for c in env.claims).strip() or "No grounded claims in freeze."
    view = env.executive_view or prose
    unknowns = list(env.uncertainties) or _str_list(role_res.get("uncertainties", role_res.get("unknowns")))
    return StockbotAnalysis(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=list(ids),
        as_of=as_of,
        question=question,
        answer=view,
        base_case=view,
        unknowns=unknowns,
        what_would_change=_str_list(role_res.get("what_would_change")),
        claims=env.claims,
        research_requests=env.follow_ups,
        executive_view=view,
        impact_channels=list(env.impact_channels),
        materiality=env.materiality,
        uncertainties=list(unknowns),
    )


def _build_bull(
    role_res: dict[str, JSONValue], *, sid: str, wave: int, fid: str, ids: list[str], as_of: str, question: str
) -> BullAnalysis:
    """Parse the bullbot envelope into its typed analysis."""
    from .agents import parse_committee_envelope
    from .agents.bullbot import BullAnalysis

    env = parse_committee_envelope(json.dumps(dict(role_res)), frozen=ids, agent="bullbot")
    prose = "\n".join(c.text for c in env.claims).strip() or "No grounded claims in freeze."
    view = env.executive_view or prose
    unknowns = list(env.uncertainties) or _str_list(role_res.get("uncertainties", role_res.get("unknowns")))
    return BullAnalysis(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=list(ids),
        as_of=as_of,
        question=question,
        stance="bullish",
        bull_case=view,
        unknowns=unknowns,
        what_would_change=_str_list(role_res.get("what_would_change")),
        claims=env.claims,
        research_requests=env.follow_ups,
        executive_view=view,
        impact_channels=list(env.impact_channels),
        materiality=env.materiality,
        uncertainties=list(unknowns),
    )


def _build_bear(
    role_res: dict[str, JSONValue], *, sid: str, wave: int, fid: str, ids: list[str], as_of: str, question: str
) -> BearAnalysis:
    """Parse the bearbot envelope into its typed analysis."""
    from .agents import parse_committee_envelope
    from .agents.bearbot import BearAnalysis

    env = parse_committee_envelope(json.dumps(dict(role_res)), frozen=ids, agent="bearbot")
    prose = "\n".join(c.text for c in env.claims).strip() or "No grounded claims in freeze."
    view = env.executive_view or prose
    unknowns = list(env.uncertainties) or _str_list(role_res.get("uncertainties", role_res.get("unknowns")))
    return BearAnalysis(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=list(ids),
        as_of=as_of,
        question=question,
        stance="bearish",
        bear_case=view,
        unknowns=unknowns,
        what_would_change=_str_list(role_res.get("what_would_change")),
        claims=env.claims,
        research_requests=env.follow_ups,
        executive_view=view,
        impact_channels=list(env.impact_channels),
        materiality=env.materiality,
        uncertainties=list(unknowns),
    )


def _latest_dossier(store: ResearchRepository, session_id: str) -> dict[str, object]:
    """Latest dossier mapping ({} when none stored)."""
    dossiers = store.list_dossiers(session_id)
    latest = dossiers[-1] if dossiers else None
    return dict(latest) if isinstance(latest, dict) else {}


def _wave_dossiers(store: ResearchRepository, session_id: str) -> list[dict[str, object]]:
    """Dossier mappings for the latest frozen wave, across every source domain.

    Dossiers persist per domain ({sid}:{wave}:{domain}), so the coverage gate
    must read the whole wave's set: a FINRA residual is invisible when only
    the latest single dossier is consulted. Falls back to the latest dossier
    when no freeze exists yet (pre-freeze submits still gate on their own row).
    """
    dossiers = [d for d in store.list_dossiers(session_id) if isinstance(d, dict)]
    if not dossiers:
        return []
    rows: list[dict[str, object]] = [dict(d) for d in dossiers]
    try:
        found = store.get_session(session_id)
        wave = max(len(found.freeze_ids), found.current_wave, 1)
    except KeyError:
        return [rows[-1]]
    wave_rows = [dict(d) for d in rows if d.get("wave_id") == wave]
    return wave_rows or [rows[-1]]


def _wave1_coverage(
    store: ResearchRepository, session_id: str
) -> tuple[dict[str, object] | None, list[dict[str, object]], list[str]]:
    """Wave dossiers' merged coverage + relationships + open questions for the director challenge.

    Residuals union across the wave's per-domain dossiers (FINRA/WEB rows are
    first-class); each dossier's source_domain rides alongside its coverage so
    the challenge can route the follow-up to the residual's own domain.
    """
    rows = _wave_dossiers(store, session_id)
    if not rows:
        return None, [], []
    merged: dict[str, object] = {}
    rels: list[dict[str, object]] = []
    open_q: list[str] = []
    for row in rows:
        coverage = row.get("coverage")
        if isinstance(coverage, dict):
            domain = str(coverage.get("source_domain") or "").upper()
            for key, value in coverage.items():
                if not isinstance(value, list) or not value:
                    continue
                prior = merged.get(key)
                tagged = [f"[{domain}] {v}" if domain and isinstance(v, str) else v for v in value]
                merged[key] = [*prior, *tagged] if isinstance(prior, list) else list(tagged)
            verdict = coverage.get("useful_for_question")
            if (
                isinstance(verdict, str)
                and verdict in ("sufficient", "insufficient")
                and merged.get("useful_for_question") != "insufficient"
            ):
                merged["useful_for_question"] = verdict
    for row in rows:
        raw_rels = row.get("relationships")
        if isinstance(raw_rels, list):
            for r in raw_rels:
                if isinstance(r, dict):
                    rels.append(dict(r))
        raw_open = row.get("open_questions")
        if isinstance(raw_open, list):
            for q in raw_open:
                if isinstance(q, str) and q:
                    open_q.append(q)
    return (merged or None), rels, open_q


def _wave1_state(
    store: ResearchRepository,
    found: ResearchSession,
) -> tuple[Wave1Result, dict[str, object]]:
    """Rebuild the trio Wave1Result from the latest freeze's committee-run jobs.

    Carries the original question + latest dossier coverage/relationships/open
    questions so the director coverage challenge can route incomplete dossiers
    to targeted follow-up without redefining the objective.
    """
    from .director import Wave1Result
    from .synthesis.committee import compute_disagreement

    sid, fid, ids, wave, as_of = _freeze_context(store, found)
    by_role = _committee_results_by_role(store, found, fid)
    kw = {"sid": sid, "wave": wave, "fid": fid, "ids": ids, "as_of": as_of, "question": found.query}
    stock = _build_stock(by_role["stockbot"], **kw) if "stockbot" in by_role else None
    bull = _build_bull(by_role["bullbot"], **kw) if "bullbot" in by_role else None
    bear = _build_bear(by_role["bearbot"], **kw) if "bearbot" in by_role else None
    disagreement = (
        compute_disagreement(stock, bull, bear) if stock is not None and bull is not None and bear is not None else None
    )
    coverage, relationships, open_questions = _wave1_coverage(store, sid)
    wave1 = Wave1Result(
        session_id=sid,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=list(ids),
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
        coverage=coverage,
        relationships=relationships,
        open_questions=open_questions,
        question=found.query,
    )
    return wave1, {"freeze_id": fid, "evidence_ids": ids, "wave_id": wave, "as_of": as_of}


def _freeze_wave_cap(found: ResearchSession, session_id: str, wave_id: int) -> None:
    """Enforce the int>=1 wave id; a ceiling applies only when policy sets a finite int."""
    if isinstance(wave_id, bool) or not isinstance(wave_id, int) or wave_id < 1:
        raise ValueError(f"freeze_session: 'wave_id' must be an int >= 1, got {wave_id!r}")
    raw_section: object = found.policy.get("research", {})
    research: dict[str, object] = raw_section if isinstance(raw_section, dict) else {}
    max_waves = research.get("max_waves")
    if isinstance(max_waves, int) and not isinstance(max_waves, bool) and wave_id > max_waves:
        raise ValueError(f"freeze_session: wave {wave_id} exceeds max_waves {max_waves}")


def _freeze_ready_session(store: ResearchRepository, session_id: str, wave_id: int):
    """Validate the wave cap, then walk CREATED->PLANNING->RESEARCHING (+T for wave 2+)."""
    from .models import SessionStatus

    found = _require_session(store, session_id)
    _freeze_wave_cap(found, session_id, wave_id)
    # Pi bootstrap mirrors runner C->P->R; service.create_research leaves CREATED.
    if found.status == SessionStatus.CREATED.value:
        found = _session.transition_session(found, SessionStatus.PLANNING)
        store.save_session(found)
    if found.status == SessionStatus.PLANNING.value:
        found = _session.transition_session(found, SessionStatus.RESEARCHING)
        store.save_session(found)
    if wave_id >= 2 and found.status == SessionStatus.TARGETED_RESEARCH.value:
        pass  # T->FREEZING via the transition below (runner R|T->F mirror).
    elif found.status != SessionStatus.RESEARCHING.value:
        raise ValueError(f"freeze_session: session {session_id!r} status is {found.status!r} (RESEARCHING required)")
    return found


def _freeze_row_kind(record: Mapping[str, object]) -> str:
    """Stored row kind; metadata.record_kind fallback; absent means evidence."""
    raw = record.get("record_kind")
    if raw is None:
        meta = record.get("metadata")
        if isinstance(meta, Mapping):
            return str(meta.get("record_kind", "evidence"))
        return "evidence"
    return str(raw)


def _freeze_open_source_jobs(store: ResearchRepository, session_id: str, wave_id: int) -> list[str]:
    """Open source/scout job ids blocking the wave freeze."""
    return [
        j.job_id
        for j in store.list_jobs(session_id)
        if j.wave_id == wave_id and j.job_type in ("source_agent", "scout") and j.status in ("queued", "running")
    ]


def _freeze_wave_records(store: ResearchRepository, session_id: str, wave_id: int):
    from .evidence import evidence_from_dict

    rows = [r for r in store.list_evidence(session_id) if _freeze_row_kind(r) != "discovery"]
    wave_recs = [evidence_from_dict(r) for r in rows]
    wave_recs = [e for e in wave_recs if e.wave_id <= wave_id]
    open_src = _freeze_open_source_jobs(store, session_id, wave_id)
    if open_src:
        raise ValueError(f"freeze_session: {len(open_src)} source jobs still open for wave {wave_id}: {open_src}")
    return wave_recs


def _freeze_coverage_note(store: ResearchRepository, session_id: str) -> str:
    """Insufficiency is carried forward as a limitation AND evaluated independently by the coverage gate.

    The note never blocks the committee; the director's coverage challenge
    decides continuation from the dossier's residuals (remaining branches /
    unsearched routes / material open questions) with or without a committee
    request.
    """
    dossiers = store.list_dossiers(session_id)
    if not dossiers:
        return ""
    completes = [d.get("coverage", {}) for d in dossiers if isinstance(d.get("coverage"), dict)]
    if completes and all(isinstance(c, dict) and c.get("complete") is False for c in completes):
        return "no sufficient source coverage; proceeding with limitations"
    return ""


def _freeze_wave_telemetry(store: ResearchRepository, session_id: str) -> dict[str, object]:
    """Persisted wave telemetry, derived from evidence rows/coverage/journal — never model counts.

    Coverage state + remaining branches persist here (freeze payload) and on the
    submit job telemetry; the stop reason persists via the wave.stopped /
    wave.authorized journal events. Underivable keys are null + telemetry_gaps.
    """
    return _derive_telemetry(store, session_id)


def _freeze_save_idempotent(store: ResearchRepository, frozen: EvidenceFreeze) -> None:
    """Persist the freeze; a duplicate save is a no-op (idempotent retry)."""
    from . import freeze as _freeze

    if not isinstance(frozen, EvidenceFreeze):
        raise ValueError(f"freeze: expected EvidenceFreeze, got {type(frozen).__name__}")  # noqa: TRY004 - the public error contract pins ValueError, tests are oracle
    try:
        store.save_freeze(_freeze.freeze_to_dict(frozen))
    except ValueError:
        pass


def _freeze_track_session(store: ResearchRepository, found: ResearchSession, fid: str, wave_id: int) -> ResearchSession:
    """Walk RESEARCHING->FREEZING, attach the freeze id, advance the wave."""
    from .models import SessionStatus

    out = _session.transition_session(found, SessionStatus.FREEZING)
    if fid not in out.freeze_ids:
        out = replace(out, freeze_ids=[*out.freeze_ids, fid], updated_at=utcnow())
    store.save_session(out)
    if wave_id > out.current_wave:
        out = replace(out, current_wave=wave_id, updated_at=utcnow())
        store.save_session(out)
    return out


def _freeze_empty_note(wave_recs: Sequence[object], note: str) -> str:
    """Empty waves freeze with a limitations note; non-empty waves keep the gate note."""
    if note or wave_recs:
        return note
    return "no PIT-eligible evidence for this wave; proceeding with limitations"


def _freeze_committee_jobs(store: ResearchRepository, session_id: str, wave_id: int) -> list[str]:
    """Pre-existing committee job ids for this wave; empty when none exist."""
    jobs = store.list_jobs(session_id)
    return [j.job_id for j in jobs if j.wave_id == wave_id and j.job_type in ("stockbot", "bullbot", "bearbot")]


def _freeze_result_payload(
    frozen: EvidenceFreeze,
    committee: list[str],
    wave_id: int,
    fid: str,
    note: str,
    telemetry: dict[str, object] | None = None,
) -> dict[str, object]:
    """Freeze payload: freeze dict plus the caller-driven committee verb (+ wave telemetry)."""
    from . import freeze as _freeze

    if not isinstance(frozen, EvidenceFreeze):
        raise ValueError(f"freeze: expected EvidenceFreeze, got {type(frozen).__name__}")  # noqa: TRY004 - the public error contract pins ValueError, tests are oracle
    out_dict = _freeze.freeze_to_dict(frozen)
    out_dict["pending_next_action"] = {
        "verb": "EXECUTE_COMMITTEE",
        "jobs": committee or ["stockbot", "bullbot", "bearbot"],
        "wave_id": wave_id,
        "freeze_id": fid,
    }
    if telemetry is not None:
        out_dict["telemetry"] = telemetry
    if not note:
        return out_dict
    out_dict["coverage_gate"] = "insufficient"
    out_dict["limitations"] = note
    return out_dict


def freeze_session(
    session_id: str,
    wave_id: int = 1,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Freeze wave evidence (PIT-checked) and move RESEARCHING -> FREEZING."""
    from . import freeze as _freeze

    store = _repo(repo)
    found = _freeze_ready_session(store, session_id, wave_id)
    wave_recs = _freeze_wave_records(store, session_id, wave_id)
    fid = f"{session_id}:{wave_id}:freeze"
    frozen = _freeze.create_freeze(
        freeze_id=fid, session_id=session_id, wave_id=wave_id, records=wave_recs, as_of=found.as_of
    )
    _freeze.verify_freeze(frozen, wave_recs)
    _freeze_save_idempotent(store, frozen)
    _freeze_track_session(store, found, fid, wave_id)
    _emit(store, session_id, "freeze.created", {"freeze_id": fid, "wave_id": wave_id})
    note = _freeze_empty_note(wave_recs, _freeze_coverage_note(store, session_id))
    telemetry = _freeze_wave_telemetry(store, session_id)
    _emit(store, session_id, "wave.telemetry", {"freeze_id": fid, **telemetry})
    # Committee creation stays caller-driven: freeze returns the verb + roles and
    # reuses pre-existing committee job ids when present, never auto-creating.
    return _freeze_result_payload(
        frozen, _freeze_committee_jobs(store, session_id, wave_id), wave_id, fid, note, telemetry
    )


def _committee_roles_by_wave(store: ResearchRepository, session_id: str, wave_id: int) -> dict[str, str]:
    """Existing committee job id per role for this wave (first row per role wins)."""
    by_role: dict[str, str] = {}
    for j in store.list_jobs(session_id):
        if j.wave_id == wave_id and j.job_type in ("stockbot", "bullbot", "bearbot"):
            by_role.setdefault(j.job_type, j.job_id)
    return by_role


def _fail_partial_trio(store: ResearchRepository, started_ids: Sequence[str], exc: Exception) -> None:
    """Fail the jobs this call started so a failed trio never stays RUNNING."""
    reason = f"partial trio: {type(exc).__name__}"[:2000]
    for jid in started_ids:
        try:
            leftover = store.get_job(jid)
            if leftover.status in ("queued", "running"):
                store.save_job(_jobs.fail_job(leftover, FailureCategory.COMMITTEE_DEADLOCK, reason))
        except KeyError:
            continue


def _ensure_committee_roles(
    store: ResearchRepository, session_id: str, wave_id: int, by_role: Mapping[str, str]
) -> list[str]:
    """Reuse or create+start one job per committee role, in order; rolls the started ones back on failure."""
    created: list[str] = []
    started_ids: list[str] = []
    cur = store.get_session(session_id)
    try:
        for role in ("stockbot", "bullbot", "bearbot"):
            if role in by_role:
                created.append(by_role[role])
                continue
            cur, job = _jobs.create_job(
                cur, store.list_jobs(session_id), job_type=role, owner="kernel", wave_id=wave_id
            )
            store.save_session(cur)
            started = _jobs.start_job(job)
            store.save_job(started)
            created.append(started.job_id)
            started_ids.append(started.job_id)
            _emit(store, session_id, "committee.created", {"job_id": started.job_id, "role": role})
    except Exception as exc:
        # Never leave a half-created trio RUNNING: fail the jobs this call started.
        _fail_partial_trio(store, started_ids, exc)
        raise
    return created


def create_committee_jobs(
    session_id: str, wave_id: int = 1, *, repo: ResearchRepository | Path | str | None = None
) -> dict[str, object]:
    """Opt-in atomic trio creation: 3 distinct pi-owned committee jobs, ids returned together."""
    store = _repo(repo)
    found = _require_session(store, session_id)
    fid = f"{session_id}:{wave_id}:freeze"
    created = _ensure_committee_roles(store, session_id, wave_id, _committee_roles_by_wave(store, session_id, wave_id))
    pending: dict[str, object] = {"verb": "EXECUTE_COMMITTEE", "jobs": created, "wave_id": wave_id, "freeze_id": fid}
    if fid not in found.freeze_ids:
        pending["freeze_pending"] = fid
    return {
        "session_id": session_id,
        "wave_id": wave_id,
        "jobs": created,
        "freeze_id": fid,
        "pending_next_action": pending,
    }


def _committee_live_job(store: ResearchRepository, session_id: str, job_id: str, role: str, analysis: object):
    """Validate role/shape, then load the running job owned by this session."""
    if role not in ("stockbot", "bullbot", "bearbot"):
        raise ValueError(f"record_committee_analysis: role must be stockbot|bullbot|bearbot, got {role!r}")
    if not isinstance(analysis, Mapping):
        raise ValueError(f"record_committee_analysis: 'analysis' must be a mapping, got {type(analysis).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    found = _require_session(store, session_id)
    job = _require_job(store, job_id)
    if job.session_id != found.session_id:
        raise ValueError(f"record_committee_analysis: job {job_id!r} belongs to {job.session_id!r}")
    if job.status != "running":
        raise ValueError(
            f"record_committee_analysis: ERR_JOB_CLOSED job {job_id!r} status is {job.status!r} (running required)"
        )
    if job.job_type != role:
        raise ValueError(f"record_committee_analysis: job {job_id!r} job_type {job.job_type!r} != role {role!r}")
    return found, job


def _committee_freeze_ids(
    store: ResearchRepository, session_id: str, found: ResearchSession, job_wave: int
) -> tuple[str, list[str]]:
    """Resolve the latest freeze id + evidence ids; enforce the freeze-wave match."""
    if not found.freeze_ids:
        raise ValueError(f"record_committee_analysis: session {session_id!r} has no freeze")
    fid = found.freeze_ids[-1]
    try:
        frozen = store.get_freeze(fid)
    except KeyError:
        raise ValueError(f"record_committee_analysis: unknown freeze_id: {fid!r}") from None
    raw_ids = frozen.get("evidence_ids")
    freeze_ids = [e for e in raw_ids if isinstance(e, str)] if isinstance(raw_ids, list) else []
    frozen_wave = frozen.get("wave_id")
    if isinstance(frozen_wave, int) and not isinstance(frozen_wave, bool) and job_wave != frozen_wave:
        raise ValueError(f"record_committee_analysis: job wave {job_wave!r} != freeze wave {frozen_wave!r}")
    return fid, freeze_ids


def _committee_verify_write(store: ResearchRepository, session_id: str, fid: str) -> None:
    """Recompute the freeze hash from stored evidence; raise on drift (fail-closed write)."""
    from . import freeze as _freeze
    from .evidence import evidence_from_dict

    frozen = _freeze.freeze_from_dict(store.get_freeze(fid))
    recs = [
        evidence_from_dict(r)
        for r in store.list_evidence(session_id)
        if isinstance(r.get("evidence_id"), str) and r.get("evidence_id") in set(frozen.evidence_ids)
    ]
    _freeze.verify_freeze(frozen, recs)


def _committee_envelope(analysis: Mapping[str, object], role: str, freeze_ids: list[str]) -> dict[str, object]:
    """JSON-shape the analysis and ground it against the freeze; return the dict form."""
    from .agents import parse_committee_envelope

    try:
        envelope = json.dumps(dict(analysis))
    except TypeError as exc:
        raise ValueError(f"record_committee_analysis: 'analysis' must be JSON-able: {exc}") from exc
    env = parse_committee_envelope(envelope, frozen=freeze_ids, agent=role)
    out = dict(analysis)
    out.setdefault("role", role)
    out.setdefault("executive_view", env.executive_view)
    return out


def _committee_track_run(cur: ResearchSession, fid: str, job_id: str, job_wave: int):
    """Merge one completed job into the session's committee_runs for this freeze."""
    from .models import validate_json_value

    runs: list[JSONValue] = list(cur.committee_runs)
    for i, entry in enumerate(runs):
        if isinstance(entry, dict) and entry.get("freeze_id") == fid:
            got = entry.get("jobs", [])
            known: list[str] = [j for j in got if isinstance(j, str)] if isinstance(got, list) else []
            if job_id not in known:
                known.append(job_id)
            merged: dict[str, JSONValue] = {
                "freeze_id": fid,
                "wave_id": entry.get("wave_id", job_wave),
                "jobs": validate_json_value(known, "<service>"),
            }
            runs[i] = merged
            break
    else:
        fresh: dict[str, JSONValue] = {
            "freeze_id": fid,
            "wave_id": job_wave,
            "jobs": validate_json_value([job_id], "<service>"),
        }
        runs.append(fresh)
    return replace(cur, committee_runs=runs, updated_at=utcnow())


def record_committee_analysis(
    session_id: str,
    job_id: str,
    role: str,
    analysis: dict[str, object],
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Validate one trio analysis against the freeze, persist it, complete the job."""
    from .models import SessionStatus

    store = _repo(repo)
    found, job = _committee_live_job(store, session_id, job_id, role, analysis)
    fid, freeze_ids = _committee_freeze_ids(store, session_id, found, job.wave_id)
    _committee_verify_write(store, session_id, fid)
    result = _committee_envelope(analysis, role, freeze_ids)
    done = _jobs.complete_job(job, result=result)
    store.save_job(done)
    cur = found
    if cur.status == SessionStatus.FREEZING.value:
        cur = _session.transition_session(cur, SessionStatus.ANALYZING)
    cur = _committee_track_run(cur, fid, job_id, job.wave_id)
    store.save_session(cur)
    _emit(store, session_id, "committee.created", {"job_id": job_id, "role": role, "freeze_id": fid})
    return done.to_dict()


def _no_create_session(_question: str, _as_of: str) -> str:
    raise AssertionError("decide_next_wave: create_session is not sequenced in service")


def _no_fetch(_sid: str) -> list[str]:
    raise AssertionError("decide_next_wave: fetch_wave_evidence is not sequenced in service")


def _no_freeze(_sid: str) -> str:
    raise AssertionError("decide_next_wave: create_freeze is not sequenced in service")


def _no_committee(_sid: str):
    raise AssertionError("decide_next_wave: run_committee is not sequenced in service")


def _text(value: object) -> str:
    """Stripped string of one optional value ("" when absent)."""
    return str(value or "").strip()


def _raw_doc_identity(row: Mapping[str, object]) -> tuple[str, str] | None:
    """(accession, document) identity of one raw-source evidence row; None when absent."""
    prov = row.get("provenance")
    if isinstance(prov, Mapping) and prov.get("kind") == "sec_source":
        accession = _text(prov.get("accession_no"))
        if accession:
            return (accession, _text(prov.get("document_name")))
    record = _text(row.get("source_record_id"))
    return (record, _text(row.get("source_name"))) if record else None


def _novelty_wave(row: Mapping[str, object]) -> int:
    """Persisted wave of one evidence row (0 when absent or mistyped)."""
    value = row.get("wave_id")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _row_novelty_key(row: Mapping[str, object]) -> tuple[object, str]:
    """(raw-document identity, content hash): a re-read of a held document is not new evidence."""
    return (_raw_doc_identity(row), str(row.get("content_hash") or ""))


def _split_wave_rows(
    rows: Sequence[Mapping[str, object]], wave_id: int
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """(rows from earlier waves, rows of ``wave_id``)."""
    return ([r for r in rows if _novelty_wave(r) != wave_id], [r for r in rows if _novelty_wave(r) == wave_id])


def _rows_new_versus_prior(
    prior: Sequence[Mapping[str, object]], current: Sequence[Mapping[str, object]]
) -> list[Mapping[str, object]]:
    """Current rows whose (document, content-hash) key the prior waves did not already hold."""
    prior_keys = {_row_novelty_key(r) for r in prior}
    return [r for r in current if _row_novelty_key(r) not in prior_keys]


def _doc_identities(items: Iterable[Mapping[str, object]]) -> set[tuple[str, str]]:
    """Distinct raw-document identities across a row set (unnamed rows contribute none)."""
    return {d for d in (_raw_doc_identity(r) for r in items) if d}


def _novelty_subjects(items: Iterable[Mapping[str, object]]) -> set[str]:
    """Lower-cased non-blank subjects of a row set (one entity counts once)."""
    return {str(r.get("subject") or "").strip().lower() for r in items if str(r.get("subject") or "").strip()}


def _novelty_row_deltas(rows: Sequence[Mapping[str, object]], wave_id: int) -> dict[str, int]:
    """New raw documents / evidence records / entities in the current wave versus everything prior."""
    prior, current = _split_wave_rows(rows, wave_id)
    new_rows = _rows_new_versus_prior(prior, current)
    return {
        "new_raw_documents": len(_doc_identities(new_rows) - _doc_identities(prior)),
        "new_evidence_records": len(new_rows),
        "new_entities": len(_novelty_subjects(new_rows) - _novelty_subjects(prior)),
    }


def _productive_waves(rows: Iterable[Mapping[str, object]]) -> dict[int, bool]:
    """Per-wave productivity: productive means the wave contributed a row the run did not already hold."""
    productive: dict[int, bool] = {}
    seen_rows: set[tuple[object, str]] = set()
    for row in rows:
        wave = _novelty_wave(row)
        productive.setdefault(wave, False)
        key = _row_novelty_key(row)
        if key in seen_rows:
            continue
        seen_rows.add(key)
        productive[wave] = True
    return productive


def _relationship_keys_by_wave(store: ResearchRepository, session_id: str) -> dict[int, set[str]]:
    """Persisted dossier relationships per wave as JSON keys; {} when storage is unreadable."""
    out: dict[int, set[str]] = {}
    try:
        dossiers = store.list_dossiers(session_id)
    except Exception:  # noqa: BLE001 - storage degradation means no relationship signal, never a guess
        return out
    for dossier in dossiers:
        wave = dossier.get("wave_id")
        wave_int = wave if isinstance(wave, int) and not isinstance(wave, bool) else 0
        rels = dossier.get("relationships")
        if isinstance(rels, list):
            out.setdefault(wave_int, set()).update(
                json.dumps(r, sort_keys=True, default=str) for r in rels if isinstance(r, Mapping)
            )
    return out


def _relationship_novelty(store: ResearchRepository, session_id: str, wave_id: int, productive: dict[int, bool]) -> int:
    """Mark relationship-bearing waves productive; return the current wave's fresh relationships."""
    rels_by_wave = _relationship_keys_by_wave(store, session_id)
    prior_rels: set[str] = set()
    for wave in sorted(rels_by_wave):
        fresh = rels_by_wave[wave] - prior_rels
        prior_rels |= fresh
        if fresh:
            productive[wave] = True
    elsewhere = {r for wave, rels in rels_by_wave.items() if wave != wave_id for r in rels}
    return len(rels_by_wave.get(wave_id, set()) - elsewhere)


def _zero_novelty_streak(productive: Mapping[int, bool], wave_id: int) -> int:
    """Consecutive unproductive waves counting back from ``wave_id``."""
    streak = 0
    for wave in range(wave_id, 0, -1):
        if productive.get(wave, False):
            break
        streak += 1
    return streak


def _gate_novelty(store: ResearchRepository, found: ResearchSession) -> dict[str, object]:
    """Persisted novelty for the director gate: doc/evidence/entity deltas + loop counters.

    Derived from evidence rows and journal events only; never model-reported.
    A re-read of a document the run already holds is a record, not new evidence.
    """
    rows = [r for r in store.list_evidence(found.session_id) if _freeze_row_kind(r) != "discovery"]
    wave_id = found.current_wave if isinstance(found.current_wave, int) else 1
    productive = _productive_waves(rows)
    new_relationships = _relationship_novelty(store, found.session_id, wave_id, productive)
    blocked, zero_actions = _journal_blocked_novelty(store, found.session_id)
    raw_unresolved: object = found.unresolved_questions
    unresolved: list[object] = list(raw_unresolved) if isinstance(raw_unresolved, list) else []
    return {
        **_novelty_row_deltas(rows, wave_id),
        "new_relationships": new_relationships,
        "new_material_claims": 0,
        "resolved_questions": 0,
        "new_questions": len([q for q in unresolved if isinstance(q, str) and q.strip()]),
        "zero_novelty_waves": _zero_novelty_streak(productive, wave_id),
        "duplicate_actions_blocked": blocked,
        "zero_novelty_actions": zero_actions,
    }


def _decide_inputs(store: ResearchRepository, session_id: str, found: ResearchSession):
    """Build the director inputs: wave1 (empty when no freeze), budgets, counters."""
    from .director import DirectorDeps, Wave1Result

    jobs = store.list_jobs(session_id)
    try:
        wave1, _ = _wave1_state(store, found)
    except ValueError:
        wave1 = None
    if wave1 is None:
        wave1 = Wave1Result(
            session_id=session_id,
            wave_id=1,
            freeze_id="",
            evidence_ids=[],
            stock=None,
            bull=None,
            bear=None,
            disagreement=None,
        )
        waves_used = 1
    else:
        waves_used = max(1, len(found.freeze_ids))
    raw_used = found.budget.get("tool_calls_used")
    tool_used = raw_used if isinstance(raw_used, int) and not isinstance(raw_used, bool) and raw_used >= 0 else 0
    elapsed = (utcnow() - found.created_at).total_seconds() if isinstance(found.created_at, datetime) else 0.0

    def _record_stop(sid: str, reason: str) -> None:
        from .journal import append_event, hydrate

        hydrate(sid, store.list_events(sid))
        store.save_event(append_event(sid, "wave.stopped", "service", "service", {"reason": reason}))

    deps = DirectorDeps(
        create_session=_no_create_session,
        fetch_wave_evidence=_no_fetch,
        create_freeze=_no_freeze,
        run_committee=_no_committee,
        record_stop=_record_stop,
    )
    return wave1, deps, _configured_budgets(found), waves_used, len(jobs), tool_used, elapsed


def _configured_budgets(found: ResearchSession):
    """Explicitly configured policy budgets (None everywhere when unlimited).

    ``research.zero_novelty_limit`` is an explicit operator runaway guard on a
    branch's zero-novelty streak, never research semantics: left unset, the
    director's exhaustion gate decides termination.
    """
    from .director import DirectorBudgets

    research = found.policy.get("research") if isinstance(found.policy, Mapping) else None
    section: Mapping[str, object] = research if isinstance(research, Mapping) else {}

    def _int_cfg(key: str) -> int | None:
        value = section.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    def _float_cfg(key: str) -> float | None:
        value = section.get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None

    return DirectorBudgets(
        max_waves=_int_cfg("max_waves"),
        max_jobs=_int_cfg("max_total_jobs"),
        max_tool_calls=_int_cfg("max_tool_calls"),
        runtime_budget_s=_float_cfg("max_runtime"),
        zero_novelty_limit=_int_cfg("zero_novelty_limit"),
    )


def _decide_settle(store: ResearchRepository, session_id: str, decision: WaveDecision) -> None:
    """Persist the gate outcome: authorized->TARGETED_RESEARCH(+next wave), else->SYNTHESIZING."""
    from .journal import append_event
    from .journal import hydrate as _hydrate
    from .models import SessionStatus

    if decision.authorized:
        cur = store.get_session(session_id)
        if cur.status == SessionStatus.ANALYZING.value:
            cur = _session.transition_session(cur, SessionStatus.TARGETED_RESEARCH)
            nxt = max(cur.current_wave, len(cur.freeze_ids)) + 1
            cur = replace(
                cur,
                current_wave=nxt,
                targeted_question=decision.targeted_question or None,
                targeted_domain=decision.targeted_domain or None,
                updated_at=utcnow(),
            )
            store.save_session(cur)
        _hydrate(session_id, store.list_events(session_id))
        store.save_event(
            append_event(
                session_id,
                "wave.authorized",
                "service",
                "service",
                {
                    "question": decision.targeted_question,
                    "domain": decision.targeted_domain,
                },
            )
        )
    else:
        cur = store.get_session(session_id)
        if cur.status == SessionStatus.ANALYZING.value:
            store.save_session(_session.transition_session(cur, SessionStatus.SYNTHESIZING))


def decide_next_wave(
    session_id: str,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Gate one targeted wave (coverage challenge first); persists the stop reason either way.

    Targeted follow-up resolves one uncertainty then a new freeze + rerun
    analysis; it never redefines the objective (original question + covered vs
    remaining branches ride the decision detail). Coverage state + remaining
    branches persist via the wave.stopped/wave.authorized journal events and
    the telemetry, not the return shape (stable contract).
    """
    from .director import WaveDecision
    from .director import decide_next_wave as _decide
    from .models import source_domain_allowed

    store = _repo(repo)
    found = _require_session(store, session_id)
    wave1, deps, budgets, waves_used, jobs_used, tool_used, elapsed = _decide_inputs(store, session_id, found)
    novelty = _gate_novelty(store, found)
    decision = _decide(
        wave1,
        deps=deps,
        budgets=budgets,
        waves_used=waves_used,
        jobs_used=jobs_used,
        tool_calls_used=tool_used,
        elapsed_s=elapsed,
        novelty=novelty,
    )
    if (
        decision.authorized
        and decision.targeted_domain.strip()
        and not source_domain_allowed(found.source_policy, decision.targeted_domain, "<wave>")
    ):
        decision = WaveDecision(
            False,
            "not_actionable",
            f"targeted domain {decision.targeted_domain!r} denied by session source_policy",
            targeted_question=decision.targeted_question,
            targeted_domain=decision.targeted_domain,
        )
    _decide_settle(store, session_id, decision)
    return {
        "authorized": decision.authorized,
        "stop_reason": decision.stop_reason,
        "reason_detail": decision.reason_detail,
        "targeted_question": decision.targeted_question,
        "targeted_domain": decision.targeted_domain,
    }


def _finalize_trio(store: ResearchRepository, session_id: str, found: ResearchSession):
    """Load wave1 + require the full trio and disagreement for this freeze."""
    wave1, meta = _wave1_state(store, found)
    fid = meta["freeze_id"]
    assert isinstance(fid, str)
    missing = [
        name
        for name, present in (("stockbot", wave1.stock), ("bullbot", wave1.bull), ("bearbot", wave1.bear))
        if present is None
    ]
    if missing:
        raise ValueError(f"finalize_session: missing committee analyses for {missing} on freeze {fid!r}")
    assert wave1.stock is not None and wave1.bull is not None and wave1.bear is not None
    disagreement = wave1.disagreement
    if disagreement is None:
        raise ValueError(f"finalize_session: missing disagreement on freeze {fid!r}")
    return wave1, meta, fid, disagreement


def _first_present(data: Mapping[str, object], *keys: str) -> object:
    """First present value under any of the keys, in order; absent keys fall back to ``""``."""
    for key in keys:
        if key in data:
            return data[key]
    return ""


def _claim_mappings(claims: Sequence[object]) -> list[Mapping[str, object]]:
    """Every claim must be a mapping (the public error contract pins ValueError)."""
    out: list[Mapping[str, object]] = []
    for item in claims:
        if not isinstance(item, Mapping):
            raise ValueError(f"finalize_session: each claim must be a mapping, got {type(item).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        out.append(item)
    return out


def _shaped_claim(item: Mapping[str, object]) -> dict[str, object]:
    """One caller claim narrowed to the grounded envelope: text + evidence_ids (+ explicit claim_type)."""
    refs = item.get("evidence_ids", [])
    shaped: dict[str, object] = {
        "text": _first_present(item, "text", "claim_text", "claim"),
        "evidence_ids": list(refs) if isinstance(refs, (list, tuple)) else [],
    }
    if item.get("claim_type") is not None:
        # Explicit claim_type survives; an absent one stays absent (parsers default to inference).
        shaped["claim_type"] = item.get("claim_type")
    return shaped


def _finalize_claims(session_id: str, claims: object, freeze_ids: list[str]):
    """Shape caller claims and ground every citation against the freeze."""
    from .agents import parse_grounded_claims

    if not isinstance(claims, list):
        raise ValueError(f"finalize_session: 'claims' must be a list, got {type(claims).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    if not claims:
        raise ValueError("finalize_session: 'claims' must be a non-empty grounded list")
    narrowed = [_shaped_claim(item) for item in _claim_mappings(claims)]
    try:
        envelope = json.dumps(narrowed)
    except TypeError as exc:
        raise ValueError(f"finalize_session: 'claims' must be JSON-able: {exc}") from exc
    return parse_grounded_claims(envelope, frozen=[e for e in freeze_ids if isinstance(e, str)])


def _finalize_answer(answer: object) -> str:
    """Require a non-blank synthesis answer."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("finalize_session: 'answer' must be a non-empty string")
    return answer


def _append_unique(lims: list[str], value: object) -> None:
    """Append one stripped limitation once (ignores blanks/mistyped)."""
    text = value.strip() if isinstance(value, str) else ""
    if text and text not in lims:
        lims.append(text)


def _dossier_limitations(lims: list[str], dossier: Mapping[str, object]) -> None:
    """Fold one dossier coverage (limitations/gaps + incomplete flag) into lims."""
    coverage = dossier.get("coverage")
    if not isinstance(coverage, dict):
        return
    for key in ("source_limitations", "gaps"):
        raw = coverage.get(key)
        if isinstance(raw, list):
            for item in raw:
                _append_unique(lims, item)
    if coverage.get("complete") is False:
        domain = str(coverage.get("source_domain") or "").upper()
        _append_unique(lims, f"{domain or 'Source'} coverage incomplete for this session")


def _finalize_limitations(store: ResearchRepository, session_id: str) -> list[str]:
    """Evidence limitations from dossier coverage + unresolved questions."""
    lims: list[str] = []
    dossiers: list[Mapping[str, object]]
    try:
        dossiers = [d for d in store.list_dossiers(session_id) if isinstance(d, dict)]
    except Exception:  # noqa: BLE001 - best-effort read; synthesis never fails on limitations
        dossiers = []
    for dossier in dossiers:
        _dossier_limitations(lims, dossier)
    try:
        found = store.get_session(session_id)
    except KeyError:
        return lims
    for item in found.unresolved_questions:
        _append_unique(lims, item)
    return lims


def _finalize_scope(found: ResearchSession) -> dict[str, object]:
    """Research scope from the session source policy (SEC-only by default)."""
    policy = getattr(found, "source_policy", {}) or {}
    raw = policy.get("allowed", ["SEC"]) if isinstance(policy, dict) else ["SEC"]
    allowed: list[str] = [s.strip() for s in raw if isinstance(s, str) and s.strip()] if isinstance(raw, list) else []
    return {"allowed_sources": allowed or ["SEC"]}


def _finalize_content(synth: FinalSynthesis, answer: str) -> str:
    """Substantive rendered answer from the rich result (never a bare status line)."""
    from app.tool_render import render_final_result

    text = render_final_result(synth.to_dict())
    return text.strip() or answer.strip()


def _finalize_persist(
    store: ResearchRepository,
    session_id: str,
    fid: str,
    answer: str,
    grounded: list[GroundedClaim],
    synth: FinalSynthesis,
) -> dict[str, object]:
    """Persist the synthesis result and walk ANALYZING->SYNTHESIZING->COMPLETED."""
    from .models import SessionStatus

    final = dict(synth.to_dict())
    final.setdefault("answer", answer)
    final.setdefault("freeze_id", fid)
    raw_claims = final.get("claims")
    claims_json: list[object] = list(raw_claims) if isinstance(raw_claims, list) else []
    if not claims_json:
        from .models import validate_json_value

        for claim in grounded:
            row: dict[str, object] = {
                "text": claim.text,
                "claim_type": claim.claim_type,
                "evidence_ids": validate_json_value(list(claim.evidence_ids), "<service>"),
            }
            claims_json.append(row)
        final["claims"] = claims_json
        final["grounded_claims"] = list(claims_json)
    content = _finalize_content(synth, answer)
    final["content"] = content
    from .models import validate_json_mapping

    validated = validate_json_mapping(final, "<service>: 'final_result'")
    cur = store.get_session(session_id)
    if cur.status == SessionStatus.ANALYZING.value:
        cur = _session.transition_session(cur, SessionStatus.SYNTHESIZING)
        store.save_session(cur)
    cur = replace(store.get_session(session_id), final_result=validated, updated_at=utcnow())
    store.save_session(cur)
    if cur.status == SessionStatus.SYNTHESIZING.value:
        cur = _session.transition_session(cur, SessionStatus.COMPLETED)
        store.save_session(cur)
    return {
        "session_id": session_id,
        "freeze_id": fid,
        "status": cur.status,
        "content": content,
        "final_result": dict(validated),
    }


def _finalize_observations(
    store: ResearchRepository, session_id: str, freeze_ids: Sequence[object]
) -> list[dict[str, JSONValue]]:
    """Canonical ledger rows of this freeze: the direct-evidence material for synthesis.

    Coverage artifacts are not evidence and never appear here; a row that is not a
    substantive ledger record (or is not in this freeze) is left out.
    """
    wanted = {e for e in freeze_ids if isinstance(e, str)}
    return [
        row
        for row in store.list_evidence(session_id)
        if row.get("record_kind") != "discovery" and str(row.get("evidence_id")) in wanted
    ]


def _finalize_absence_texts(store: ResearchRepository, session_id: str) -> list[str]:
    """Claim texts of the session's coverage artifacts (search-scope absence records)."""
    texts: list[str] = []
    for artifact in store.list_coverage_artifacts(session_id):
        text = artifact.get("claim_text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _finalize_coverage(store: ResearchRepository, session_id: str) -> dict[str, object]:
    """Per-source searched scope for the final render: one section per dossier domain.

    Each completed source job persists its own dossier coverage ({sid}:{wave}:
    {domain}); the fold carries submit-shaped keys forward verbatim and
    final.py keeps every non-empty string-list field except verdict keys, so
    desk keys (docs, datasets_queried, queries_executed, ...) survive under
    their own names to the rendered Coverage block.
    """
    out: dict[str, dict[str, list[str]]] = {"sec": {}, "finra": {}, "web": {}}
    try:
        dossiers = [d for d in store.list_dossiers(session_id) if isinstance(d, dict)]
    except Exception:  # noqa: BLE001 - best-effort read; synthesis never fails on coverage
        return {}
    for dossier in dossiers:
        coverage = dossier.get("coverage")
        if not isinstance(coverage, dict):
            continue
        domain = str(coverage.get("source_domain") or "").upper()
        section = out.get(domain.lower()) if domain.lower() in out else None
        if section is None:
            continue
        for key, value in coverage.items():
            if not isinstance(value, list) or not value:
                continue
            items = [v.strip() for v in value if isinstance(v, str) and v.strip()]
            if items:
                section[key] = sorted(set(section.get(key, []) + items))
    return {k: v for k, v in out.items() if v}


def _finalize_synth(
    found: ResearchSession,
    wave1: Wave1Result,
    meta: dict[str, object],
    fid: str,
    disagreement: CommitteeDisagreement,
    model: str,
    claims: list[object],
    session_id: str,
    store: ResearchRepository,
):
    """Ground claims + run the trio synthesis; return (synth, grounded)."""
    from .synthesis.final import synthesize_final

    ids_raw = meta["evidence_ids"]
    assert isinstance(ids_raw, list)
    grounded = _finalize_claims(session_id, claims, [e for e in ids_raw if isinstance(e, str)])
    assert wave1.stock is not None and wave1.bull is not None and wave1.bear is not None
    wave_raw = meta["wave_id"]
    as_of_raw = meta["as_of"]
    assert isinstance(wave_raw, int) and isinstance(as_of_raw, str)
    synth = synthesize_final(
        found.query,
        session_id=session_id,
        wave_id=wave_raw,
        freeze_id=fid,
        as_of=as_of_raw,
        stock=wave1.stock,
        bull=wave1.bull,
        bear=wave1.bear,
        disagreement=disagreement,
        model=model,
        extra_claims=[
            {
                "text": c.text,
                "claim_type": c.claim_type,
                "evidence_ids": list(c.evidence_ids),
            }
            for c in grounded
        ],
        evidence_limitations=_finalize_limitations(store, session_id),
        research_scope=_finalize_scope(found),
        observations=_finalize_observations(store, session_id, ids_raw),
        absence_observations=_finalize_absence_texts(store, session_id),
        coverage=_finalize_coverage(store, session_id),
    )
    if not synth.answer.strip():
        raise ValueError("finalize_session: synthesis produced an empty answer")
    return synth, grounded


def finalize_session(
    session_id: str,
    answer: str,
    claims: list[object],
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Persist the trio-joined synthesis and move to COMPLETED."""
    store = _repo(repo)
    found = _require_session(store, session_id)
    if found.status in _session.TERMINAL_STATUSES:
        raise ValueError(f"finalize_session: session {session_id!r} status is {found.status!r} (terminal)")
    model = _finalize_answer(answer)
    wave1, meta, fid, disagreement = _finalize_trio(store, session_id, found)
    open_all = [j.job_id for j in store.list_jobs(session_id) if j.status in ("queued", "running")]
    if open_all:
        raise ValueError(f"finalize_session: {len(open_all)} jobs still open: {open_all}")
    synth, grounded = _finalize_synth(found, wave1, meta, fid, disagreement, model, claims, session_id, store)
    return _finalize_persist(store, session_id, fid, synth.answer, grounded, synth)


def _dispatch_unknown_job_detail(store: ResearchRepository, session_id: str, detail: str) -> str:
    """Actionable unknown-job detail: live ids + likely never-persisted attempt job."""
    import sqlite3

    try:
        live: list[str] = [j.job_id for j in store.list_jobs(session_id) if j.status in ("queued", "running")][:8]
    except (sqlite3.Error, ValueError, KeyError, OSError):
        live = []
    return f"{detail} hint: attempt job not persisted — check source_policy/source_domain on start_job live_jobs={live}"


def _dispatch_live_job(store: ResearchRepository, session_id: str, job_id: str):
    """Load session + job with cross-session ownership enforced."""
    found = _require_session(store, session_id)
    try:
        job = _require_job(store, job_id)
    except ResearchNotFound as exc:
        detail = str(exc.args[0]) if exc.args else str(exc)
        raise ResearchNotFound(_dispatch_unknown_job_detail(store, session_id, detail)) from None
    if job.session_id != found.session_id:
        raise ValueError(f"dispatch: job {job_id!r} belongs to {job.session_id!r}")
    return found, job


def _dispatch_check_domain(job: Job, job_id: str, tool_name: str) -> None:
    """Committee jobs never fetch; source jobs accept only their domain's tools."""
    if job.job_type in ("stockbot", "bullbot", "bearbot"):
        raise ValueError(
            f"dispatch: committee job {job_id!r} ({job.job_type}) cannot dispatch tools (frozen evidence only)"
        )
    if job.source_domain is None or tool_name.startswith("research"):
        return
    from .agents.source_agent import is_finra_tool, is_sec_tool, is_web_tool

    domain = job.source_domain.upper()
    allowed = (
        is_sec_tool(tool_name)
        if domain == "SEC"
        else is_finra_tool(tool_name)
        if domain == "FINRA"
        else is_web_tool(tool_name)
        if domain == "WEB"
        else False
    )
    if not allowed:
        scope = "SEC" if domain == "SEC" else domain
        raise ValueError(f"dispatch: tool {tool_name!r} outside {scope} domain for job {job_id!r}")


# Job-diagnostics slot: normalized dispatch action key -> session evidence count when it ran.
_DISPATCH_ACTIONS = "dispatch_actions"


def _dispatch_action(
    job: Job, found: ResearchSession, tool_name: str, arguments: Mapping[str, object]
) -> tuple[str, str, str, str, tuple[str, ...], str, str, str, str]:
    """Deterministic identity of one dispatch: normalized action fields + canonical arguments.

    Semantic fields mirror the runner's loop key through
    ``normalize_research_action`` (source/tool/query/ticker/forms/as_of/
    accession/objective); the call's own arguments ride along verbatim so two
    distinct calls — two documents of one accession, two queries — are never
    one action. Staging ids are routing metadata, not action identity.
    """
    from .director import normalize_research_action

    args = {key: value for key, value in arguments.items() if key not in ("session_id", "job_id")}
    forms = args.get("forms")
    entity = args.get("ticker") or args.get("identifier") or args.get("entity") or args.get("query")
    action = normalize_research_action(
        job.source_domain or "",
        tool_name,
        str(args.get("query") or ""),
        str(entity or ""),
        forms if isinstance(forms, (list, tuple, str)) else (),
        str(args.get("as_of") or ""),
        str(args.get("accession_no") or ""),
        found.objective,
    )
    return (*action, json.dumps(args, sort_keys=True, default=str))


def _dispatch_loop_gate(
    store: ResearchRepository, found: ResearchSession, job: Job, tool_name: str, arguments: Mapping[str, object] | None
) -> None:
    """Refuse an exact repeat of an action already dispatched for this job with no new evidence.

    State rides in the job's persisted ``diagnostics`` (action key -> session
    evidence count when the action ran), so a resumed session keeps detecting
    loops. Only exact repeats with zero evidence growth are refused: a
    materially different action, or the same action after new evidence landed,
    always runs — no numeric cap on searches, filings, documents, waves, or
    jobs. A refused repeat raises ``research_loop_detected`` (the gateway's
    error channel) and journals the same event, which ``decide_next_wave``
    counts as ``duplicate_actions_blocked``.

    A dispatch without arguments carries no action identity and passes through
    untracked: the model path always supplies the validated arguments, so this
    only shields direct kernel callers.
    """
    if arguments is None:
        return
    action = _dispatch_action(job, found, tool_name, arguments)
    key = json.dumps(list(action))
    raw_seen = job.diagnostics.get(_DISPATCH_ACTIONS)
    seen: dict[str, JSONValue] = dict(raw_seen) if isinstance(raw_seen, Mapping) else {}
    count = len(store.list_evidence(found.session_id))
    prior = seen.get(key)
    if isinstance(prior, int) and not isinstance(prior, bool) and count <= prior:
        _emit(
            store,
            found.session_id,
            "research_loop_detected",
            {
                "job_id": job.job_id,
                "tool": tool_name,
                "action": list(action),
                "query": action[2],
                "evidence_count": count,
                "reason": "research_loop_detected",
            },
        )
        raise ValueError(
            f"research_loop_detected: {tool_name!r} repeats an action that already ran with no new evidence "
            f"(evidence_count={count}); vary the arguments or record the findings it produced before retrying"
        )
    seen[key] = count
    # ponytail: read-modify-write on the job row; a lost concurrent update can
    # only re-admit a repeat (never block a fresh action), so it needs no lock.
    store.save_job(replace(job, diagnostics={**job.diagnostics, _DISPATCH_ACTIONS: seen}))


def authorize_and_consume_dispatch(
    session_id: str,
    job_id: str,
    tool_name: str,
    *,
    arguments: Mapping[str, object] | None = None,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, object]:
    """Authorize one tool dispatch, then atomically consume job + global budget slots.

    ``arguments`` is the validated tool call's argument mapping; it keys the
    no-progress repeat gate (a refused repeat raises ``research_loop_detected``).
    """
    from .stage import check_stage_tool, stage_for_session

    store = _repo(repo)
    found, job = _dispatch_live_job(store, session_id, job_id)
    check_stage_tool(stage_for_session(found, store.list_jobs(session_id)), tool_name)
    _dispatch_check_domain(job, job_id, tool_name)
    try:
        billed, spent = store.consume_dispatch_budget(session_id, job_id)
    except KeyError as exc:
        detail = str(exc.args[0]) if exc.args else str(exc)
        if "unknown job_id" in detail:
            detail = _dispatch_unknown_job_detail(store, session_id, detail)
        raise ResearchNotFound(detail) from None
    # Budget first, repeat gate second (the runner's order), so an exhausted
    # budget keeps its own refusal verbatim.
    _dispatch_loop_gate(store, found, spent, tool_name, arguments)
    raw_used: object = billed.budget.get("tool_calls_used", 0)
    used = raw_used if isinstance(raw_used, int) and not isinstance(raw_used, bool) else 0
    return {
        "session_id": session_id,
        "job_id": job_id,
        "tool_name": tool_name,
        "tool_budget": spent.tool_budget,
        "tool_calls_used": used,
    }


def _require_node(store: ResearchRepository, node_id: str) -> ResearchNode:
    try:
        return store.get_node(node_id)
    except KeyError:
        raise ResearchNotFound(f"unknown node_id: {node_id!r}") from None


def create_node(
    session_id: str,
    question: str,
    why_it_matters: str,
    depends_on: Sequence[str] | None = None,
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> ResearchNode:
    """Create one ResearchNode (what needs knowing); fail-closed on unknown session."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("create_node: 'question' must be a non-empty string")
    if not isinstance(why_it_matters, str) or not why_it_matters.strip():
        raise ValueError("create_node: 'why_it_matters' must be a non-empty string")
    deps = tuple(depends_on or ())
    if any(not isinstance(d, str) or not d for d in deps):
        raise ValueError("create_node: 'depends_on' must be a list of non-empty strings")
    store = _repo(repo)
    _require_session(store, session_id)
    for dep in deps:
        dep_node = _require_node(store, dep)
        if dep_node.session_id != session_id:
            raise ValueError(f"create_node: depends_on {dep!r} belongs to another session")
    node = ResearchNode(
        node_id=new_node_id(),
        session_id=session_id,
        question=question.strip(),
        why_it_matters=why_it_matters.strip(),
        depends_on=deps,
    )
    node.validate("<service>")
    try:
        store.save_node(node)
    except KeyError:
        raise ResearchNotFound(f"unknown session_id: {session_id!r}") from None
    _emit(store, session_id, "node.created", {"node_id": node.node_id})
    return node


def _node_resolved(nodes: list[ResearchNode], node_id: str) -> bool:
    found = next((n for n in nodes if n.node_id == node_id), None)
    return found is not None and found.status == "resolved"


def ready_nodes(session_id: str, *, repo: ResearchRepository | Path | str | None = None) -> list[ResearchNode]:
    """Nodes whose deps resolved and whose own status is proposed/gathering (read-only)."""
    store = _repo(repo)
    _require_session(store, session_id)
    nodes = [n for n in store.list_nodes(session_id) if n.session_id == session_id]
    return [
        n
        for n in nodes
        if n.status in ("proposed", "gathering") and all(_node_resolved(nodes, d) for d in n.depends_on)
    ]


def _transition_node(store: ResearchRepository, session_id: str, node_id: str, status: str, event: str) -> ResearchNode:
    node = _require_node(store, node_id)
    if node.session_id != session_id:
        raise ResearchNotFound(f"unknown node_id: {node_id!r}")
    try:
        updated = store.update_node_status(node_id, status)
    except KeyError:
        raise ResearchNotFound(f"unknown node_id: {node_id!r}") from None
    _emit(store, session_id, event, {"node_id": node_id})
    return updated


def resolve_node(session_id: str, node_id: str, *, repo: ResearchRepository | Path | str | None = None) -> ResearchNode:
    """Mark one node resolved."""
    return _transition_node(_repo(repo), session_id, node_id, "resolved", "node.resolved")


def block_node(
    session_id: str, node_id: str, reason: str = "", *, repo: ResearchRepository | Path | str | None = None
) -> ResearchNode:
    """Mark one node blocked (reason carried on the journal event)."""
    store = _repo(repo)
    updated = _transition_node(store, session_id, node_id, "blocked", "node.blocked")
    if reason:
        _emit(store, session_id, "node.blocked", {"node_id": node_id, "reason": str(reason)[:500]})
    return updated


def reject_node(session_id: str, node_id: str, *, repo: ResearchRepository | Path | str | None = None) -> ResearchNode:
    """Mark one node rejected."""
    return _transition_node(_repo(repo), session_id, node_id, "rejected", "node.rejected")


def record_decision(
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
    provider: str | None = None,
    latency_ms: float | None = None,
    repo: ResearchRepository | Path | str | None = None,
) -> DecisionRecord:
    """Persist one JEV disposition: full candidate registry + probabilities + selected tool set."""
    if not isinstance(decision_type, str) or not decision_type.strip():
        raise ValueError("record_decision: 'decision_type' must be a non-empty string")
    store = _repo(repo)
    _require_session(store, session_id)
    if node_id is not None:
        node = _require_node(store, node_id)
        if node.session_id != session_id:
            raise ResearchNotFound(f"unknown node_id: {node_id!r}")
    if job_id is not None:
        job = _require_job(store, job_id)
        if job.session_id != session_id:
            raise ResearchNotFound(f"unknown job_id: {job_id!r}")
    decision = DecisionRecord(
        decision_id=new_decision_id(),
        session_id=session_id,
        node_id=node_id,
        job_id=job_id,
        decision_type=decision_type.strip(),
        candidates=validate_json_mapping(dict(candidates), "<service>: 'candidates'"),
        probabilities=validate_json_mapping(dict(probabilities), "<service>: 'probabilities'"),
        selected=validate_json_value(selected, "<service>: 'selected'"),
        confidence=confidence,
        created_at=utcnow(),
    )
    decision.validate("<service>")
    try:
        store.save_decision(decision, request=request, response=response, provider=provider, latency_ms=latency_ms)
    except KeyError as exc:
        raise ResearchNotFound(exc.args[0] if exc.args else str(exc)) from None
    _emit(store, session_id, "decision.recorded", {"decision_id": decision.decision_id, "type": decision.decision_type})
    return decision


def admit_evidence(
    session_id: str,
    job_id: str,
    data: Mapping[str, object],
    *,
    repo: ResearchRepository | Path | str | None = None,
) -> dict[str, JSONValue]:
    """Kernel evidence admission: research_session_id/job_id/as_of carriage into record_evidence."""
    if not isinstance(data, Mapping):
        raise ValueError("admit_evidence: 'data' must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return record_evidence(session_id, job_id, dict(data), repo=repo)
