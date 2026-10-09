"""Retired live-research loop (deterministic test helper only).

Production research runs the kernel scheduler path: needle harness +
runKernelAgent + kernel_worker + scheduler.run_node (see
scripts/verify_agent_scenarios.py). This module's sequential
in-process loop (source job -> fetch -> freeze -> trio -> gate) stays only
because deterministic unit tests pin its step behavior; it must not gain new
production callers.
Records built here with record_kind=='evidence' are candidate/materialized
evidence for deterministic tests only, not production citable evidence:
only service acceptance persists to the raw archive (runner fetch/materialize
never archives).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from app.research import freeze as _freeze
from app.research import jobs as _jobs
from app.research import session as _session
from app.research.agents import ResearchRequest
from app.research.agents.bearbot import BearAnalysis, run_bearbot
from app.research.agents.bullbot import BullAnalysis, run_bullbot
from app.research.agents.scout import ScoutAssignment, ScoutResult
from app.research.agents.sec_agent import run_sec_assignment
from app.research.agents.source_agent import is_sec_tool
from app.research.agents.stockbot import StockbotAnalysis, run_stockbot
from app.research.director import (
    NOVELTY_ZERO_KEYS,
    DirectorBudgets,
    DirectorDeps,
    LoopDetector,
    Wave1Result,
    decide_next_wave,
    normalize_research_action,
    run_wave1,
    synthesize_wave1,
)
from app.research.dossiers.sec import SECDossier, dossier_to_dict, validate_dossier
from app.research.evals.traces import (
    TraceRecorder,
    create_trace,
    get_trace_events,
    list_traces,
)
from app.research.evidence import (
    Evidence,
    EvidenceLedger,
    EvidenceRejectedError,
    evidence_content_hash,
    evidence_from_dict,
    evidence_to_dict,
    ingest_evidence,
    normalize_accession,
    search_run_ref,
    sec_source_ref,
)
from app.research.journal import append_event, hydrate, rejection_payload
from app.research.models import (
    Failure,
    FailureCategory,
    Job,
    JobType,
    JournalEvent,
    JSONValue,
    ResearchSession,
    SessionStatus,
    default_policy,
    utcnow,
)
from app.research.repository import ResearchRepository
from app.research.synthesis.committee import CommitteeDisagreement, compute_disagreement

if TYPE_CHECKING:
    from app.research.service import MaterializedEvidenceSource


# Tool-*reported* errors (a result dict carrying "error") are agent-visible by
# default: the payload - reason text and schema hint included - goes back to the
# caller, which keeps working. Only an explicit fatal ``error_type`` ends the
# wave, because no retry repairs a dead provider, missing auth, broken storage,
# or a passed deadline. Escaping exceptions (a dispatch that raises) stay fatal.
FATAL_TOOL_ERRORS: frozenset[str] = frozenset(
    {
        "auth_required",
        "source_unavailable",
        "storage_error",
        "provider_error",
        "timeout",
        "deadline_exceeded",
    }
)
# Scout failures that degrade one assignment instead of ending the run: the
# provider was slow, or the model call failed/misbehaved, so this assignment
# returned nothing. Infrastructure faults (a tool dispatch that raises) and
# run-level denials (policy/budget) keep their fatal behavior; cancellation
# propagates on its own.
SCOUT_ASSIGNMENT_ERRORS: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.TIMEOUT,
        FailureCategory.MODEL_ERROR,
        FailureCategory.PROVIDER_ERROR,
        FailureCategory.MODEL_OUTPUT_FAILURE,
    }
)


_KNOWN_AT_KEYS = (
    "known_at",
    "accepted_at",
    "acceptanceDatetime",
    "acceptedDate",
    "published_at",
    "publishedAt",
    "published",
    "filingDate",
    "filedAt",
    "filed",
    "date",
    "timestamp",
)


def _known_at_scopes(raw: Mapping[str, object]) -> list[object]:
    """Every known_at candidate in priority order: top, record, meta, source_refs."""
    scopes: list[object] = [raw.get(k) for k in _KNOWN_AT_KEYS]
    record = raw.get("record")
    if isinstance(record, dict):
        scopes.extend(record.get(k) for k in _KNOWN_AT_KEYS)
    meta = raw.get("meta")
    if isinstance(meta, dict):
        scopes.extend(meta.get(k) for k in _KNOWN_AT_KEYS)
        refs = meta.get("source_refs")
        if isinstance(refs, dict):
            scopes.append(refs.get("known_at"))
    return scopes


def _coerce_known_at(value: object) -> datetime | None:
    """One candidate: datetime UTC-converted, ISO string parsed; anything else skipped."""
    from datetime import datetime as _dt

    if isinstance(value, _dt):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = _dt.fromisoformat(value.strip())
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _extract_known_at(raw: Mapping[str, object]) -> datetime | None:
    """Source-provided timestamp or None; never invented, never as_of."""
    for value in _known_at_scopes(raw):
        parsed = _coerce_known_at(value)
        if parsed is not None:
            return parsed
    return None


_SOURCE_URI_KEYS = (
    "source_uri",
    "uri",
    "url",
    "source_url",
    "source",
    "document_url",
    "filing_url",
    "link",
    "path",
)
_SOURCE_ID_KEYS = (
    "source_record_id",
    "record_id",
    "accession_no",
    "accession",
    "accessionNumber",
    "accession_number",
    "document",
    "document_id",
    "filing_id",
    "id",
)
# Only these open a raw source document; every other SEC tool result is a
# navigation artifact and can never ground evidence.
_DOCUMENT_TOOLS = frozenset({"get_sec_document", "get_sec_filing"})
# Display packet carried back to the source agent for navigation results: the
# hits are what the agent opens next (never evidence themselves).
_NAVIGATION_PACKET_KEYS = ("search_id", "top_hits", "count")


def _json_dumps(value: object, _json: object) -> str:
    dumps = getattr(_json, "dumps", None)
    if not callable(dumps):
        return str(value)
    out: object = dumps(value, sort_keys=True, default=str)
    return out if isinstance(out, str) else str(value)


def _flatten_trace_value(value: object, _rj: object, _json: object) -> str | int | float | bool | None:
    """One complex trace value: redacted JSON, str() fallback on encoder failure."""
    try:
        redact = _rj if callable(_rj) else None
        rendered = _json_dumps(value, _json)
        cleaned: object = redact(rendered) if redact is not None else rendered
        text = cleaned if isinstance(cleaned, str) else rendered
        return text[:2000]
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return str(value)[:2000]


def _is_uri_like(value: str) -> bool:
    text = value.strip()
    return "://" in text or "/" in text or "." in text


def _source_ref_scopes(raw: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Every reference scope in priority order: top, record, meta, source_refs."""
    scopes: list[Mapping[str, object]] = [raw]
    record = raw.get("record")
    if isinstance(record, dict):
        scopes.append(record)
    meta = raw.get("meta")
    if isinstance(meta, dict):
        scopes.append(meta)
        refs = meta.get("source_refs")
        if isinstance(refs, dict):
            scopes.append(refs)
    return scopes


def _scope_uri(scope: Mapping[str, object]) -> str | None:
    """First URI-like string in one scope; None when absent."""
    for key in _SOURCE_URI_KEYS:
        value = scope.get(key)
        if isinstance(value, str) and value.strip() and _is_uri_like(value):
            return value.strip()
    return None


def _scope_ref(scope: Mapping[str, object]) -> str | None:
    """First record id in one scope; None when absent."""
    for key in _SOURCE_ID_KEYS:
        value = scope.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return None


def _extract_source_ref(raw: Mapping[str, object]) -> tuple[str | None, str | None]:
    """Real SEC reference from the tool result; (None, None) when absent."""
    uri: str | None = None
    ref: str | None = None
    for scope in _source_ref_scopes(raw):
        if uri is None:
            uri = _scope_uri(scope)
        if ref is None:
            ref = _scope_ref(scope)
        if uri is not None and ref is not None:
            break
    return uri, ref


def _inner_args(args: Mapping[str, object]) -> Mapping[str, object]:
    """Inner tool arguments: a nested 'arguments' mapping wins, else the call args."""
    inner = args.get("arguments")
    return inner if isinstance(inner, Mapping) else args


def _first_str(scope: Mapping[str, object], keys: Sequence[str]) -> str:
    """First stripped non-empty string among the keys; empty when none is present."""
    for key in keys:
        value = scope.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _str_items(value: object) -> set[str]:
    """Stripped non-empty strings from a list value; empty set otherwise."""
    return {v.strip() for v in value if isinstance(v, str) and v.strip()} if isinstance(value, list) else set()


def _row_mappings(row: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    """Mapping items of one list-valued row field; anything else yields nothing."""
    value = row.get(key)
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _keyed_strs(rows: Sequence[Mapping[str, object]], key: str) -> set[str]:
    """Stripped non-empty string values of one key across rows."""
    found: set[str] = set()
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            found.add(value.strip())
    return found


def _scope_questions(scope: Mapping[str, object], keys: Sequence[str]) -> set[str]:
    """Open-question strings carried under any of one scope's question keys."""
    found: set[str] = set()
    for key in keys:
        found |= _str_items(scope.get(key))
    return found


def _dossier_items(row: Mapping[str, object]) -> tuple[set[str], set[str], set[str]]:
    """(relationship ids, claim texts, open questions) from one persisted dossier row."""
    rels = _keyed_strs(_row_mappings(row, "relationships"), "relationship_id")
    claims = _keyed_strs(_row_mappings(row, "findings"), "text")
    questions = _scope_questions(row, ("open_questions", "unknowns"))
    coverage = row.get("coverage")
    if isinstance(coverage, Mapping):
        questions |= _scope_questions(coverage, ("material_open_questions", "unresolved", "open_questions"))
    return rels, claims, questions


def _document_fields(raw: Mapping[str, object], inner_args: Mapping[str, object]) -> tuple[str, str, str]:
    """(normalized accession, document name, raw passage) from a tool result and its call args."""
    raw_accession = _first_str(raw, ("accession_no", "accession")) or _first_str(
        inner_args, ("accession_no", "accession")
    )
    try:
        accession = normalize_accession(raw_accession)
    except ValueError:
        accession = ""
    document_name = _first_str(raw, ("document_name", "document", "primary_document")) or _first_str(
        inner_args, ("document_name",)
    )
    passage = _first_str(raw, ("text", "content", "passage"))
    return accession, document_name, passage


def _record_kind(inner: str, accession: str, document_name: str, passage: str) -> str:
    """'evidence' only for a document-opening tool result carrying accession, document, and passage."""
    if inner in _DOCUMENT_TOOLS and accession and document_name and passage:
        return "evidence"
    return "discovery"


def _record_content(raw: Mapping[str, object], passage: str, is_evidence: bool) -> str:
    """Record body: the raw passage for evidence, else the result's text or its JSON form."""
    if is_evidence:
        return passage
    content_obj: object = raw.get("content")
    if isinstance(content_obj, str) and content_obj.strip():
        return content_obj.strip()
    return json.dumps(raw, sort_keys=True, default=str)  # a mapping always renders, at least "{}"


def _record_labels(
    q: str, is_evidence: bool, inner: str, accession: str, document_name: str, scoped: Sequence[str]
) -> tuple[str, str]:
    """(subject, claim text): the question, then the document identity or the navigation note."""
    subject = q.strip()[:120] if q.strip() else "sec evidence"
    if is_evidence:
        return subject, f"{document_name} {accession}"[:500]
    return subject, f"{inner} finding for {', '.join(scoped) if scoped else 'universe'}"[:500]


def _record_ref(is_evidence: bool, ref: str | None, search_id: str, accession: str) -> str | None:
    """Record id: the document ref for evidence; a navigation row falls back to search run, then accession."""
    if is_evidence:
        return ref
    return ref or search_id or accession or None


def _record_provenance(
    is_evidence: bool,
    *,
    accession: str,
    document_name: str,
    passage: str,
    uri: str | None,
    search_id: str,
    query: str,
) -> dict[str, JSONValue]:
    """Record provenance: the SEC source ref for evidence, the search run (or none) for navigation."""
    if is_evidence:
        return sec_source_ref(accession_no=accession, document_name=document_name, passage=passage, source_uri=uri)
    if search_id and query:
        return search_run_ref(search_id=search_id, query=query)
    return {"kind": "none"}


def _row_identity(rec: Evidence) -> tuple[str, str]:
    """(accession, document name) of one ledger row; blanks when the record carries neither."""
    accession = str(rec.metadata.get("accession_no", "") or rec.source_record_id or "")
    document = str(rec.metadata.get("document_name", "") or "")
    return accession, document


def _ledger_evidence(ledger: EvidenceLedger, session_id: str, wave: int | None = None) -> list[Evidence]:
    """Substantive ledger rows of a session (evidence records only), optionally one wave."""
    return [
        e
        for e in ledger.list_session(session_id)
        if e.record_kind == "evidence" and (wave is None or e.wave_id == wave)
    ]


def _merge_disagreement(
    first: CommitteeDisagreement,
    second: CommitteeDisagreement,
) -> CommitteeDisagreement:
    """Fold two waves' disagreements; the newer freeze owns the merged record.

    Routing follows the newest committee read: ``requested_research`` is the
    second (latest) wave's set, unioned with the same question's requesting
    agents inside that wave. Carrying older requests forward would pin the gate
    to an already-executed question - its exact repeat is then blocked as a
    research loop - and starve every newer material question, so the loop could
    never advance past the first requested wave. Agreement, uncertainties, and
    critical disagreements stay cumulative for the record.
    """
    seen: dict[str, ResearchRequest] = {}
    for request in second.requested_research:
        prior: ResearchRequest | None = seen.get(request.question)
        if prior is None:
            seen[request.question] = request
        else:
            for name in request.requesting_agents:
                if name not in prior.requesting_agents:
                    prior.requesting_agents.append(name)
    return CommitteeDisagreement(
        session_id=second.session_id,
        wave_id=second.wave_id,
        freeze_id=second.freeze_id,
        agreement=list(dict.fromkeys([*first.agreement, *second.agreement]))[:20],
        disagreement=list(dict.fromkeys([*first.disagreement, *second.disagreement]))[:20],
        critical_uncertainties=list(dict.fromkeys([*first.critical_uncertainties, *second.critical_uncertainties]))[
            :20
        ],
        requested_research=list(seen.values()),
        consensus=list(dict.fromkeys([*getattr(first, "consensus", []), *getattr(second, "consensus", [])]))[:20],
        critical_disagreements=[
            *getattr(first, "critical_disagreements", []),
            *getattr(second, "critical_disagreements", []),
        ][:10],
    )


class BudgetLedger:
    """One authoritative run budget: atomic consume, read-only total (None = unlimited)."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._used = 0
        self._lock = threading.Lock()

    def consume_research_dispatch(self) -> bool:
        """Increment once before a real research dispatch; False only on an explicit int limit."""
        with self._lock:
            if self._limit is not None and self._used >= self._limit:
                return False
            self._used += 1
            return True

    def hydrate(self, used: int) -> None:
        """Seed cumulative total on resume; never rewinds below current."""
        with self._lock:
            if isinstance(used, bool):
                return
            if isinstance(used, int) and used > self._used:
                self._used = used

    @property
    def used(self) -> int:
        with self._lock:
            return self._used


# Keyword -> category for committee failures the shared ladder does not claim.
# Ordered: the first matching group wins.
_COMMITTEE_KEYWORDS: tuple[tuple[tuple[str, ...], FailureCategory], ...] = (
    (("model_output", "unknown evidence", "uncited"), FailureCategory.MODEL_OUTPUT_FAILURE),
    (("tool_error", "tool failed"), FailureCategory.TOOL_ERROR),
)


class LiveModelError(RuntimeError):
    """A live Pi model call failed after the failure was persisted.

    Carries the session/stage so the CLI can exit nonzero with the same
    message plus the session id. The job is already FAILED (TIMEOUT),
    the journal holds ``model.failed`` + ``wave.stopped``, and the
    session carries the failure — resume stays safe, never duplicated.
    """

    def __init__(
        self, session_id: str, stage: str, message: str, category: FailureCategory = FailureCategory.MODEL_ERROR
    ) -> None:
        super().__init__(message)
        self.session_id: str = session_id
        self.stage: str = stage
        self.message: str = message
        # Carried so downstream classifiers report the real category, never a fake timeout.
        self._failure_category: FailureCategory = category


class _LiveRun:
    """Shared sequential-wave machinery for run_live (new) and resume_live (existing).

    Owns the store, in-memory ledger, dossier/job bookkeeping, and every
    stage step. run_live drives it from session creation; resume_live preloads
    it from persisted state and skips already-done fetch/freeze stages.
    """

    def __init__(
        self,
        store: ResearchRepository,
        question: str,
        objective: str,
        as_of: str | None,
        as_of_str: str,
        scoped: list[str],
        dispatch: Callable[[str, dict[str, object]], dict[str, object]],
        model: Callable[[str], str],
        limits: DirectorBudgets,
        wave_id: int = 1,
        actor: str = "run_live",
        provider: str = "fake",
        model_name: str | None = None,
    ) -> None:
        self.store = store
        self.question = question
        self.objective = objective
        self.as_of = as_of
        self.as_of_str = as_of_str
        self.scoped = scoped
        self.dispatch = dispatch
        self.model = model
        self.limits = limits
        self.wave_id = wave_id
        self._actor = actor
        self.ledger: EvidenceLedger = EvidenceLedger()
        self.dossier_ids: list[str] = []
        self.source_jobs: list[str] = []
        self.budget = BudgetLedger(limits.max_tool_calls)
        self.loops = LoopDetector()
        self.branch = ""
        self.zero_novelty_waves = 0
        self.wave_actions: dict[int, dict[str, int]] = {}
        self.wave_items: dict[int, dict[str, set[str]]] = {}
        self.t0: float = monotonic()
        self._lock = threading.Lock()
        self.trace: TraceRecorder | None = None
        self.trace_id: str | None = None
        self._scout_deadline: datetime | None = None
        self.provider = provider or "fake"
        _mn = (
            model_name.strip()
            if isinstance(model_name, str) and model_name.strip()
            else getattr(model, "__name__", "live") or "live"
        )
        self.model_name = str(_mn)[:120]

    def _emit(self, session_id: str, event_type: str, payload: Mapping[str, object]) -> None:
        with self._lock:
            prior: list[JournalEvent] = self.store.list_events(session_id)
            hydrate(session_id, prior)
            event: JournalEvent = append_event(session_id, event_type, "runner", self._actor, dict(payload))
            self.store.save_event(event)
        self._trace_record(event_type, payload)

    def _save_budget_used(self, session_id: str, *, strict: bool = False) -> None:
        """Persist cumulative research dispatches so resume hydrates the same total."""
        if strict:
            with self._lock:
                used = self.budget.used
                sess = self.store.get_session(session_id)
                budget = dict(sess.budget)
                budget["tool_calls_used"] = used
                self.store.save_session(replace(sess, budget=budget, updated_at=utcnow()))
            return
        try:
            with self._lock:
                used = self.budget.used
                sess = self.store.get_session(session_id)
                budget = dict(sess.budget)
                budget["tool_calls_used"] = used
                self.store.save_session(replace(sess, budget=budget, updated_at=utcnow()))
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _flatten_trace_payload(
        self, payload: Mapping[str, object] | None
    ) -> dict[str, str | int | float | bool | None]:
        """Redacted flat payload for the trace recorder; complex values JSON-encoded."""
        import json as _json

        from app.redact import redact_json as _rj

        flat: dict[str, str | int | float | bool | None] = {}
        for k, v in (payload or {}).items():
            if v is None or isinstance(v, (str, int, float, bool)):
                flat[k] = _rj(str(v)) if isinstance(v, str) else v
            else:
                flat[k] = _flatten_trace_value(v, _rj, _json)
        return flat

    def _trace_record(
        self,
        event_type: str,
        payload: Mapping[str, object] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        tr = self.trace
        if tr is None:
            return
        try:
            flat = self._flatten_trace_payload(payload)
            with self._lock:
                tr.record(event_type, flat, duration_ms)
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _fail_open_job(
        self,
        session_id: str,
        job_id: str,
        stage: str,
        message: str,
        exc: Exception,
        category: FailureCategory,
    ) -> None:
        """Fail one queued/running job with its real category; silent when already closed or missing."""
        with self._lock:
            try:
                open_job = self.store.get_job(job_id)
            except KeyError:
                return
        if open_job is not None and open_job.status in ("queued", "running"):
            with self._lock:
                self.store.save_job(_jobs.fail_job(open_job, category, message))
            self._emit(
                session_id,
                "job.failed",
                {
                    "job_id": job_id,
                    "stage": stage,
                    "failure_category": category.value,
                    "error": str(exc)[:2000],
                },
            )

    def _fail_session(self, session_id: str, message: str, category: FailureCategory) -> None:
        """Stamp the session failure and move it to FAILED; never reopens terminals."""
        with self._lock:
            sess = self.store.get_session(session_id)
            sess = replace(
                sess,
                failure=Failure(category=category.value, message=message),
                updated_at=utcnow(),
            )
            if sess.status not in ("failed", "completed", "cancelled"):
                sess = _session.transition_session(sess, SessionStatus.FAILED)
            self.store.save_session(sess)

    def _persist_model_failure(
        self, session_id: str, job_id: str, stage: str, exc: Exception
    ) -> tuple[str, FailureCategory]:
        """Fail one RUNNING job + journal + session failure; return (message, category).

        The category is the real one: only an actual timeout is a TIMEOUT, every
        other model-call exception journals as MODEL_ERROR (never a fake timeout).
        """
        import subprocess

        detail: str = f"{type(exc).__name__}: {exc}"
        message: str = f"{stage}: {detail}"[:2000]
        timed_out = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
        low = f"{type(exc).__name__} {exc}".lower()
        if timed_out:
            category = FailureCategory.TIMEOUT
        elif "429" in low or "usage limit" in low or "rate limit" in low:
            # Provider quota/rate exhaustion: the run is blocked externally, not by its own logic.
            category = FailureCategory.PROVIDER_ERROR
        else:
            category = FailureCategory.MODEL_ERROR
        reason = (
            "timeout:model-call"
            if timed_out
            else "provider-error:model-call"
            if category == FailureCategory.PROVIDER_ERROR
            else f"model-error:{type(exc).__name__}"
        )
        self._fail_open_job(session_id, job_id, stage, message, exc, category)
        self._emit(
            session_id,
            "model.failed",
            {
                "stage": stage,
                "job_id": job_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            },
        )
        self._fail_session(session_id, message, category)
        self._emit(
            session_id,
            "research.failed",
            {
                "stage": stage,
                "failure_category": category.value,
                "reason": reason,
            },
        )
        self._emit(session_id, "wave.stopped", {"reason": reason, "stage": stage})
        try:
            if self.trace is not None:
                self.trace.finish(message[:2000], "failed")
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        return detail, category

    def _model_at_stage(self, stage: str, session_id: str, job_id: str) -> Callable[[str], str]:
        """Wrap the live model so a failure persists before it propagates."""

        def _call(prompt: str) -> str:
            from time import perf_counter as _pc

            from app.redact import redact_text as _rt

            _t0 = _pc()
            try:
                out = self.model(prompt)
                _dur = (_pc() - _t0) * 1000.0
                self._trace_record(
                    "model.completed",
                    {
                        "provider": self.provider,
                        "model": self.model_name,
                        "stage": stage,
                        "job_id": job_id,
                        "prompt": _rt(prompt)[:2000],
                        "output": _rt(out)[:2000],
                    },
                    _dur,
                )
                return out
            except Exception as exc:
                _dur2 = (_pc() - _t0) * 1000.0
                self._trace_record(
                    "model.failed",
                    {
                        "provider": self.provider,
                        "model": self.model_name,
                        "stage": stage,
                        "job_id": job_id,
                        "error": str(exc)[:2000],
                    },
                    _dur2,
                )
                detail, category = self._persist_model_failure(session_id, job_id, stage, exc)
                raise LiveModelError(session_id, stage, detail, category) from exc

        return _call

    def _attach_trace(self, session_id: str) -> None:
        """Create the eval trace and stamp its id on the session budget; raises on failure."""
        import subprocess as _sp

        try:
            _sha = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=2).strip() or "unknown"
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            _sha = "unknown"
        tr = create_trace(
            session_id=session_id,
            wave_id=self.wave_id,
            provider=self.provider,
            model=self.model_name,
            prompt_version="v1",
            git_sha=_sha,
        )
        self.trace, self.trace_id = tr, tr.trace_id
        try:
            with self._lock:
                cur = self.store.get_session(session_id)
                b = dict(cur.budget)
                b["trace_id"] = tr.trace_id
                self.store.save_session(replace(cur, budget=b, updated_at=utcnow()))
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        self._trace_record("trace.opened", {"trace_id": tr.trace_id, "session_id": session_id})

    def _create_session(self, q: str, aof: str, interrupt_after: str | None = None) -> str:
        _ = aof
        policy: dict[str, JSONValue] = default_policy()
        if interrupt_after is not None:
            policy["interrupt_after"] = interrupt_after
        section: JSONValue | None = policy.get("research")
        if isinstance(section, dict):
            section["max_total_jobs"] = self.limits.max_jobs
            section["max_waves"] = self.limits.max_waves
        eff_as_of: str | None = self.as_of if self.as_of and self.as_of.strip() else None
        sess = _session.create_session(q, self.objective or q, as_of=eff_as_of, policy=policy)
        sess = replace(sess, current_wave=self.wave_id)
        self.store.save_session(sess)
        try:
            self._attach_trace(sess.session_id)
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        self._emit(sess.session_id, "session.created", {"question": q, "wave_id": self.wave_id})
        sess = _session.transition_session(sess, SessionStatus.PLANNING)
        self.store.save_session(sess)
        sess = _session.transition_session(sess, SessionStatus.RESEARCHING)
        self.store.save_session(sess)
        self._open_source_job(sess.session_id, self.wave_id, self.question)
        return sess.session_id

    def _open_source_job(self, session_id: str, wave: int, question: str | None = None) -> str:
        """Create + start one SEC source_agent job for a wave; persist both sides."""
        sess = self.store.get_session(session_id)
        existing: list[Job] = self.store.list_jobs(session_id)
        updated, job = _jobs.create_job(
            sess,
            existing,
            job_type=JobType.SOURCE_AGENT,
            owner="runner",
            wave_id=wave,
            source_domain="SEC",
        )
        q = question if isinstance(question, str) and question.strip() else self.question
        tickers_json: list[JSONValue] = [t for t in self.scoped]
        job = replace(
            job,
            diagnostics={
                "tickers": tickers_json,
                "question": q[:500],
                "as_of": self.as_of_str,
            },
        )
        self.store.save_session(updated)
        self.store.save_job(job)
        self._emit(session_id, "job.created", {"job_id": job.job_id, "job_type": job.job_type})
        self.store.save_job(_jobs.start_job(job))
        self.source_jobs.append(job.job_id)
        return job.job_id

    @staticmethod
    def _match_names(items: object) -> list[dict[str, object]]:
        """Names from one match list: bare strings and {"name"} dicts only."""
        names: list[dict[str, object]] = []
        if not isinstance(items, list):
            return names
        for item in items:
            if isinstance(item, str) and item:
                names.append({"name": item})
            elif isinstance(item, dict):
                cand: object = item.get("name")
                if isinstance(cand, str) and cand:
                    names.append({"name": cand})
        return names

    def _catalog_dispatch(self, name: str, args: dict[str, object]) -> dict[str, object]:
        """Discovery passthrough: timed dispatch, normalized match names for catalog tools."""
        from time import perf_counter as _pc2

        _t1 = _pc2()
        raw: dict[str, object] = self.dispatch(name, args)
        _d1 = (_pc2() - _t1) * 1000.0
        self._trace_record(
            "discovery.completed",
            {"tool": name, "args": args, "matches": raw.get("matches")},
            _d1,
        )
        if name == "search_tools" or name == "browse_tools":
            found: object = raw.get("matches")
            if isinstance(found, list):
                return {"matches": self._match_names(found)}
            meta: object = raw.get("meta")
            inner: object = meta.get("matches") if isinstance(meta, dict) else None
            return {"matches": self._match_names(inner)}
        return raw

    def _action_key(
        self, inner: str, args: Mapping[str, object]
    ) -> tuple[str, str, str, str, tuple[str, ...], str, str, str]:
        """Semantic action key for the loop detector (source, tool, query, entity, forms, as_of, accession, objective)."""
        inner_args = _inner_args(args)
        forms: object = inner_args.get("forms")
        entity: object = (
            inner_args.get("ticker")
            or inner_args.get("identifier")
            or inner_args.get("entity")
            or inner_args.get("query")
        )
        return normalize_research_action(
            "sec",
            inner,
            str(inner_args.get("query") or ""),
            str(entity or ""),
            forms if isinstance(forms, (list, tuple, str)) else (),
            str(inner_args.get("as_of") or ""),
            str(inner_args.get("accession_no") or ""),
            self.objective,
        )

    def _guarded_tool_call(
        self, sid: str, inner: str, name: str, args: dict[str, object]
    ) -> tuple[dict[str, object], float]:
        """Policy + loop gates, timed dispatch, deadline and error checks; returns (raw, duration_ms).

        Tool volume is unlimited by default; an explicit int BudgetLedger limit
        still rejects via policy_rejection. An exact repeat of an action whose
        prior run produced no evidence is not re-executed: it journals
        ``research_loop_detected`` and returns an empty result. Materially
        different actions always run. A returned ``error`` payload is a result
        for the caller unless its ``error_type`` is in ``FATAL_TOOL_ERRORS``;
        only an exception or a fatal category ends the wave.
        """
        if not is_sec_tool(inner):
            self._emit(sid, "policy.denied", {"tool": inner, "reason": "POLICY_DENIED"})
            raise ValueError(f"POLICY_DENIED: non-SEC tool {inner!r}")
        if not self.budget.consume_research_dispatch():
            self._emit(sid, "budget.exhausted", {"tool": inner, "reason": "policy_rejection"})
            raise ValueError("policy_rejection: explicit tool limit reached")
        action = self._action_key(inner, args)
        if self.loops.precheck(action)["duplicate"]:
            self._emit(
                sid,
                "research_loop_detected",
                {
                    "tool": inner,
                    "action": list(action),
                    "query": action[2],
                    "reason": "research_loop_detected",
                },
            )
            return {"evidence_ids": []}, 0.0
        from time import perf_counter as _pc

        _t0 = _pc()
        try:
            raw: dict[str, object] = self.dispatch(name, args)
        except Exception as exc:
            _dur_e = (_pc() - _t0) * 1000.0
            self._trace_record(
                "tool.failed",
                {"tool": inner, "args": args, "error": str(exc)[:2000]},
                _dur_e,
            )
            raise
        _dur = (_pc() - _t0) * 1000.0
        self._raise_if_past_deadline(sid, inner, args, _dur)
        if "error" in raw:
            # Tool-reported errors go back to the caller as results: the model
            # sees the reason text and any schema hint and can re-call correctly.
            # Only an explicit fatal error_type ends the wave; the reason text is
            # journaled either way, so a live failure is never opaque.
            error_type = str(raw.get("error_type") or "")
            reason = str(raw.get("error"))[:2000]
            if error_type in FATAL_TOOL_ERRORS:
                self._emit(
                    sid,
                    "tool.failed",
                    {"tool": inner, "error_type": error_type, "error": reason},
                )
                self._trace_record(
                    "tool.failed",
                    {
                        "tool": inner,
                        "args": args,
                        "error_type": error_type,
                        "error": reason,
                    },
                    _dur,
                )
                raise ValueError(f"TOOL_ERROR: {inner} failed: {raw.get('error')}")
            self._emit(
                sid,
                "tool.rejected",
                {"tool": inner, "error_type": error_type, "error": reason},
            )
            self._trace_record(
                "tool.rejected",
                {
                    "tool": inner,
                    "args": args,
                    "error_type": error_type,
                    "error": reason,
                },
                _dur,
            )
            return raw, _dur
        return raw, _dur

    def _raise_if_past_deadline(self, sid: str, inner: str, args: dict[str, object], _dur: float) -> None:
        """Timeout when the scout deadline passed during dispatch; tolerant of naive datetimes."""
        try:
            _sdl = self._scout_deadline
            if _sdl is not None:
                from datetime import datetime as _dts

                _nows = _dts.now(UTC)
                _sdlc = _sdl if _sdl.tzinfo is not None else _sdl.replace(tzinfo=UTC)
                if _nows >= _sdlc:
                    self._emit(
                        sid,
                        "tool.failed",
                        {"tool": inner, "error": "scout deadline exceeded"},
                    )
                    self._trace_record(
                        "tool.failed",
                        {
                            "tool": inner,
                            "args": args,
                            "error": "scout deadline exceeded",
                        },
                        _dur,
                    )
                    raise TimeoutError("scout deadline exceeded after dispatch")
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _register_rejected_action(
        self, inner: str, args: Mapping[str, object], raw: dict[str, object]
    ) -> dict[str, object]:
        """Hand an agent-visible tool error back and register it as a zero-progress action.

        A rejected result never reaches ``_build_evidence_record`` /
        ``_ingest_tool_evidence``: persisting a discovery row per malformed call
        would claim coverage no tool produced and would count as evidence
        progress. It is registered with the loop detector at a zero evidence
        delta instead, so the exact repeat is blocked as ``research_loop_detected``
        by the pre-dispatch gate - tool budgets are unlimited by default, and
        every re-issued action is a real network call.
        """
        self.loops.check(
            self._action_key(inner, args),
            evidence_content_hash(json.dumps(raw, sort_keys=True, default=str)),
            0,
        )
        return raw

    @staticmethod
    def _copy_navigation_packets(out: dict[str, object], raw: Mapping[str, object]) -> dict[str, object]:
        """Carry the navigation packet keys a raw result held into its ingest output."""
        for key in _NAVIGATION_PACKET_KEYS:
            if key in raw and key not in out:
                out[key] = raw[key]
        return out

    def _build_evidence_record(
        self,
        sid: str,
        wave: int,
        q: str,
        src_job_id: str,
        inner: str,
        args: Mapping[str, object],
        raw: dict[str, object],
        eid: str,
    ) -> Evidence:
        """One ledger record from a successful tool result; timestamps never invented.

        Only document-opening tools (``_DOCUMENT_TOOLS``) can become evidence, and
        only when the result yields an accession, a document name, and a raw
        passage (SECSourceRef). Search/navigation results - and document results
        missing any of those - persist as discovery records: navigation
        artifacts with search-run provenance, never substantive coverage.
        Binary content (a PDF handed back as raw bytes) is not text: it is
        demoted to a NUL-free discovery record and its rejection is journaled.
        """
        inner_args = _inner_args(args)
        uri, ref = _extract_source_ref(raw)
        accession, document_name, passage = _document_fields(raw, inner_args)
        kind = _record_kind(inner, accession, document_name, passage)
        is_evidence = kind == "evidence"
        query = _first_str(inner_args, ("query",))
        search_id = _first_str(raw, ("search_id",))
        known_at = _extract_known_at(raw)
        materialized: MaterializedEvidenceSource | None = None
        provenance: dict[str, JSONValue] | None = None
        if is_evidence:
            # The archive, never the model or the tool payload, is authoritative for
            # the admitted text: reload the handle the document read returned and
            # slice the passage out of that. A result without a canonical handle, or
            # one the archive no longer reproduces, stays a navigation artifact.
            materialized, kind, is_evidence, passage = self._materialize_document_handle(
                sid, eid, raw.get("source_handle"), passage, known_at
            )
            provenance = dict(materialized.provenance) if materialized is not None else None
        content = _record_content(raw, passage, is_evidence)
        if "\x00" in content:
            # NULs mean the bytes are not readable text, and POSIX argv cannot carry
            # them at all, so this can never ground a claim. Keep a stripped navigation
            # record and say why the evidence candidacy was dropped.
            content = content.replace("\x00", "")
            kind, is_evidence, provenance, materialized = "discovery", False, None, None
            self._emit(
                sid,
                "evidence.rejected",
                rejection_payload(
                    eid,
                    "binary_source_unreadable",
                    known_at.isoformat() if known_at is not None else None,
                    self.as_of,
                ),
            )
        subject, claim_text = _record_labels(q, is_evidence, inner, accession, document_name, self.scoped)
        metadata: dict[str, JSONValue] = {
            "tool": inner,
            "tickers": ", ".join(self.scoped),
            "record_kind": kind,
        }
        scope = _first_str(inner_args, ("ticker", "cik"))
        for key, value in (
            ("accession_no", accession),
            ("document_name", document_name),
            ("query", query),
            ("search_id", search_id),
            ("search_scope", scope),
        ):
            if value:
                metadata[key] = value
        return Evidence(
            evidence_id=eid,
            session_id=sid,
            wave_id=wave,
            source_type="sec",
            source_name=inner,
            subject=subject,
            claim_text=claim_text,
            content=content,
            content_hash=evidence_content_hash(content),
            retrieved_at=materialized.retrieved_at
            if materialized is not None and materialized.retrieved_at
            else utcnow(),
            source_uri=(materialized.source_url if materialized is not None else None) or uri,
            source_record_id=_record_ref(is_evidence, ref, search_id, accession),
            published_at=materialized.filed_at if materialized is not None else None,
            known_at=materialized.known_at if materialized is not None else known_at,
            job_id=src_job_id or None,
            agent_id="sec_scout",
            supports=(),
            contradicts=(),
            confidence=None,
            quality=None,
            metadata=metadata,
            superseded_by=None,
            record_kind=kind,
            provenance=(
                provenance
                if provenance is not None
                else _record_provenance(
                    False,
                    accession=accession,
                    document_name=document_name,
                    passage=passage,
                    uri=uri,
                    search_id=search_id,
                    query=query,
                )
            ),
        )

    def _materialize_document_handle(
        self,
        sid: str,
        eid: str,
        handle: object,
        locator: str,
        known_at: datetime | None,
    ) -> tuple[MaterializedEvidenceSource | None, str, bool, str]:
        """(materialized, record kind, is_evidence, passage) for one document result.

        The kernel reloads the handle's window from the SEC archive against the
        session as_of and slices the passage out of it; a missing, stale, or
        unreadable handle demotes the result to a navigation record (journaled),
        so a claim can never ground on text the archive does not reproduce.
        """
        from app.research.service import materialize_sec_passage

        if not isinstance(handle, Mapping):
            self._emit(
                sid,
                "evidence.rejected",
                rejection_payload(
                    eid,
                    "no_canonical_source_handle",
                    known_at.isoformat() if known_at is not None else None,
                    self.as_of,
                ),
            )
            return None, "discovery", False, locator
        try:
            materialized = materialize_sec_passage(handle, locator, as_of=self.as_of)
        except ValueError as exc:
            self._emit(
                sid,
                "evidence.rejected",
                rejection_payload(
                    eid,
                    str(exc)[:300],
                    known_at.isoformat() if known_at is not None else None,
                    self.as_of,
                ),
            )
            return None, "discovery", False, locator
        passage = materialized.provenance.get("passage")
        return (
            materialized,
            "evidence",
            True,
            passage if isinstance(passage, str) else locator,
        )

    def _ingest_tool_evidence(
        self,
        sid: str,
        inner: str,
        args: dict[str, object],
        record: Evidence,
        eid: str,
        _dur: float,
    ) -> dict[str, object]:
        """Ledger + store persist for one tool record; rejected records yield empty ids.

        Discovery records persist for provenance but are never advertised as
        citable evidence ids, so no claim can ground on a navigation artifact.
        """
        if record.record_kind == "discovery":
            self._emit(
                sid,
                "discovery.recorded",
                {
                    "record_id": eid,
                    "tool": inner,
                    "query": record.metadata.get("query"),
                    "search_id": record.metadata.get("search_id"),
                },
            )
        try:
            ingest_evidence(
                self.ledger,
                record,
                as_of=self.as_of,
                on_reject=lambda t, p: self._emit(sid, t, p),
            )
        except EvidenceRejectedError as exc:
            self._trace_record(
                "tool.failed",
                {
                    "tool": inner,
                    "args": args,
                    "error": f"evidence.rejected:{exc.reason if hasattr(exc, 'reason') else exc}"[:2000],
                },
                _dur,
            )
            return {"evidence_ids": []}
        try:
            with self._lock:
                self.store.save_evidence(evidence_to_dict(record))
        except ValueError:
            pass
        if record.record_kind == "discovery":
            self._emit(sid, "discovery.ingested", {"record_id": eid, "tool": inner})
            self._trace_record(
                "tool.completed",
                {"tool": inner, "args": args, "record_kind": "discovery"},
                _dur,
            )
            return {"evidence_ids": []}
        self._emit(sid, "evidence.ingested", {"evidence_id": eid, "tool": inner})
        self._trace_record("tool.completed", {"tool": inner, "args": args, "evidence_id": eid}, _dur)
        known_at = record.known_at
        return {
            "evidence_ids": [
                {
                    "evidence_id": eid,
                    "known_at": known_at.isoformat() if known_at else None,
                    "claim_text": record.claim_text[:500],
                    "content_snippet": record.content[:500],
                }
            ]
        }

    @staticmethod
    def _finding_text(item: Mapping[str, object]) -> str | None:
        """Stripped finding text or None when blank."""
        txt = item.get("text")
        return txt.strip()[:500] if isinstance(txt, str) and txt.strip() else None

    @staticmethod
    def _finding_ids(item: Mapping[str, object]) -> list[str] | None:
        """String evidence ids or None when absent."""
        ids = item.get("evidence_ids")
        if not isinstance(ids, list) or not ids:
            return None
        eids = [e for e in ids if isinstance(e, str) and e]
        return eids or None

    @staticmethod
    def _valid_finding(item: object) -> object | None:
        """One valid finding dict or None: non-empty text plus a string evidence id."""
        from app.research.agents import GroundedClaim as _GC

        if not isinstance(item, Mapping):
            return None
        text = _LiveRun._finding_text(item)
        eids = _LiveRun._finding_ids(item)
        if text is None or eids is None:
            return None
        declared = item.get("claim_type")
        return _GC(
            text=text,
            claim_type=declared.strip() if isinstance(declared, str) and declared.strip() else "inference",
            evidence_ids=eids,
        )

    @staticmethod
    def _coerce_reused_findings(raw_findings: object) -> list[object]:
        """Valid finding dicts only: non-empty text plus at least one string evidence id."""
        if not isinstance(raw_findings, list):
            return []
        out: list[object] = []
        for item in raw_findings:
            if (found := _LiveRun._valid_finding(item)) is not None:
                out.append(found)
        return out

    @staticmethod
    def _valid_request(item: object) -> ResearchRequest | None:
        """One follow-up request or None; malformed rows never raise."""
        if not isinstance(item, Mapping):
            return None
        try:
            raw_agents: object = item.get("requesting_agents")
            agents: list[str] = [a for a in raw_agents if isinstance(a, str)] if isinstance(raw_agents, list) else []
            question_raw: object = item.get("question", "")
            material_raw: object = item.get("why_material", "")
            domain_raw: object = item.get("requested_source_domain", "SEC")
            gain_raw: object = item.get("expected_gain", "medium")
            return ResearchRequest(
                question=str(question_raw),
                why_material=str(material_raw),
                requested_source_domain=str(domain_raw),
                expected_gain=str(gain_raw),
                requesting_agents=agents,
            )
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            return None

    @staticmethod
    def _coerce_reused_requests(req_raw: object) -> list[ResearchRequest]:
        """Follow-up requests with string agents; malformed rows skipped."""
        if not isinstance(req_raw, list):
            return []
        out: list[ResearchRequest] = []
        for item in req_raw:
            req = _LiveRun._valid_request(item)
            if req is not None:
                out.append(req)
        return out

    def _reused_scout_result(
        self,
        sid: str,
        assignment: ScoutAssignment,
        child: Job,
        res: Mapping[str, object],
    ) -> ScoutResult:
        """Rebuild a ScoutResult from a completed child job; emits scout.reused."""
        from app.research.agents import GroundedClaim as _GC2

        unk_raw = res.get("unknowns")
        lim_raw = res.get("limitations")
        self._emit(
            sid,
            "scout.reused",
            {"job_id": child.job_id, "assignment_id": assignment.assignment_id},
        )
        self._trace_record(
            "scout.reused",
            {"job_id": child.job_id, "assignment_id": assignment.assignment_id},
        )
        raw_findings: list[object] = _LiveRun._coerce_reused_findings(res.get("findings"))
        findings: list[_GC2] = [f for f in raw_findings if isinstance(f, _GC2)]
        return ScoutResult(
            assignment_id=assignment.assignment_id,
            session_id=sid,
            coverage=str(res.get("coverage", "reused")),
            findings=findings,
            unknowns=[e for e in unk_raw if isinstance(e, str)] if isinstance(unk_raw, list) else [],
            limitations=[e for e in lim_raw if isinstance(e, str)] if isinstance(lim_raw, list) else [],
            follow_up_requests=_LiveRun._coerce_reused_requests(res.get("follow_up_requests")),
        )

    def _find_completed_scout(
        self,
        sid: str,
        src_job_id: str,
        assignment: ScoutAssignment,
        existing_jobs: list[Job],
    ) -> ScoutResult | None:
        """Completed child with the same assignment id; None when the scout must run."""
        for child in existing_jobs:
            if child.parent_job_id != src_job_id or child.job_type != JobType.SCOUT.value:
                continue
            res = child.result or {}
            if (
                res.get("assignment_id") == assignment.assignment_id
                and child.status == "completed"
                and "findings" in res
            ):
                return self._reused_scout_result(sid, assignment, child, res)
        return None

    def _start_reused_scout(self, sid: str, reusable: Job, assignment_id: str = "") -> Job:
        """Start a queued scout child and journal the retry."""
        if reusable.status == "queued":
            self.store.save_job(_jobs.start_job(reusable))
            self._emit(sid, "job.started", {"job_id": reusable.job_id})
        aid = assignment_id or str((reusable.diagnostics or {}).get("assignment_id"))
        self._emit(sid, "scout.retried", {"job_id": reusable.job_id, "assignment_id": aid})
        self._trace_record("scout.retried", {"job_id": reusable.job_id, "assignment_id": aid})
        return reusable

    def _create_scout_job(
        self,
        sid: str,
        wave: int,
        src_job_id: str,
        assignment: ScoutAssignment,
        existing_jobs: list[Job],
    ) -> Job:
        """Create + start a fresh scout child for one assignment."""
        from datetime import timedelta

        sess_now = self.store.get_session(sid)
        deadline = (
            (utcnow() + timedelta(seconds=assignment.time_budget_s)).isoformat()
            if isinstance(assignment.time_budget_s, (int, float))
            and not isinstance(assignment.time_budget_s, bool)
            and assignment.time_budget_s > 0
            else None
        )
        sess_upd, scout_job = _jobs.create_job(
            sess_now,
            existing_jobs,
            job_type=JobType.SCOUT,
            owner="runner",
            wave_id=wave,
            parent_job_id=src_job_id,
            source_domain="SEC",
            tool_budget=assignment.max_tool_calls,
            deadline=deadline,
        )
        tickers_j: list[JSONValue] = [t for t in assignment.tickers]
        scout_job = replace(
            scout_job,
            diagnostics={
                "assignment_id": assignment.assignment_id,
                "role": assignment.role,
                "question": assignment.question[:500],
                "tickers": tickers_j,
                "as_of": assignment.as_of,
                "max_tool_calls": assignment.max_tool_calls,
                "time_budget_s": assignment.time_budget_s,
                "allowed_domain": assignment.allowed_domain,
                "session_id": assignment.session_id,
            },
        )
        self.store.save_session(sess_upd)
        self.store.save_job(scout_job)
        self._emit(
            sid,
            "job.created",
            {
                "job_id": scout_job.job_id,
                "job_type": scout_job.job_type,
                "assignment_id": assignment.assignment_id,
            },
        )
        self.store.save_job(_jobs.start_job(scout_job))
        return scout_job

    def _open_or_reuse_scout_job(
        self,
        sid: str,
        wave: int,
        src_job_id: str,
        assignment: ScoutAssignment,
        existing_jobs: list[Job],
    ) -> Job:
        """Reuse a queued/running scout child or create + start a fresh one."""
        reusable = next(
            (
                c
                for c in existing_jobs
                if c.parent_job_id == src_job_id
                and c.job_type == JobType.SCOUT.value
                and (c.diagnostics or {}).get("assignment_id") == assignment.assignment_id
                and c.status in ("queued", "running")
            ),
            None,
        )
        if reusable is not None:
            return self._start_reused_scout(sid, reusable, assignment.assignment_id)
        return self._create_scout_job(sid, wave, src_job_id, assignment, existing_jobs)

    def _raise_if_deadline_exceeded(self, what: str, name: str = "") -> None:
        """Timeout when the scout deadline already passed; tolerant of naive datetimes."""
        from datetime import datetime as _dt

        _dl = self._scout_deadline
        if _dl is None:
            return
        try:
            _dlc = _dl if _dl.tzinfo is not None else _dl.replace(tzinfo=UTC)
            if _dt.now(UTC) >= _dlc:
                raise TimeoutError(f"scout deadline exceeded {what}{name}")
        except TimeoutError:
            raise
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _seed_scout_deadline(self, scout_job: Job, assignment: ScoutAssignment) -> None:
        """Seed the scout deadline from the job; None when neither job nor assignment sets one."""
        try:
            _dl = scout_job.deadline
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            _dl = None
        if (
            _dl is None
            and isinstance(assignment.time_budget_s, (int, float))
            and not isinstance(assignment.time_budget_s, bool)
            and assignment.time_budget_s > 0
        ):
            from datetime import timedelta as _td

            _dl = utcnow() + _td(seconds=assignment.time_budget_s)
        self._scout_deadline = _dl

    @staticmethod
    def _categorize_scout_error(exc: Exception, origin: str) -> FailureCategory:
        """Category for a scout failure: pre-tagged wins, else keyword ladder."""
        pre_cat = getattr(exc, "_failure_category", None)
        if isinstance(pre_cat, FailureCategory):
            return pre_cat
        low = (type(exc).__name__ + " " + str(exc)).lower()
        if (hit := _LiveRun._ladder_category(low)) is not None:
            return hit
        if origin == "model":
            return FailureCategory.MODEL_ERROR
        return FailureCategory.TOOL_ERROR

    def _cancel_scout_job(self, sid: str, scout_job: Job, exc: Exception, pre_cat: object) -> None:
        """Cancel one scout child on cancellation; tags the exception for the outer handler."""
        try:
            cur = self.store.get_job(scout_job.job_id)
            if cur.status in ("queued", "running"):
                self.store.save_job(_jobs.cancel_job(cur))
            self._emit(sid, "job.cancelled", {"job_id": scout_job.job_id})
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        cat_value = pre_cat if isinstance(pre_cat, FailureCategory) else FailureCategory.TOOL_ERROR
        setattr(exc, "_failure_category", cat_value)  # noqa: B010 - dynamic boundary; keeps checker green
        setattr(exc, "_scout_cancelled", True)  # noqa: B010 - dynamic boundary, no stubs; getattr keeps checker green

    def _fail_scout_job(self, sid: str, scout_job: Job, exc: Exception, cat: FailureCategory, msg: str) -> None:
        """Fail one scout child and journal it; tags the exception category."""
        try:
            cur2 = self.store.get_job(scout_job.job_id)
            if cur2.status in ("queued", "running"):
                self.store.save_job(_jobs.fail_job(cur2, cat, f"scout:{type(exc).__name__}:{msg}"[:2000]))
            self._emit(
                sid,
                "job.failed",
                {"job_id": scout_job.job_id, "failure_category": cat.value},
            )
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        setattr(exc, "_failure_category", cat)  # noqa: B010 - dynamic boundary, no stubs; getattr keeps checker green

    @staticmethod
    def _is_scout_cancelled(exc: Exception) -> bool:
        """True when the scout error signals cancellation."""
        low = (type(exc).__name__ + " " + str(exc)).lower()
        return (
            "cancel" in type(exc).__name__.lower()
            or "cancelled" in low
            or "canceled" in low
            or bool(getattr(exc, "_scout_cancelled", False))
        )

    def _complete_scout_job(self, sid: str, assignment: ScoutAssignment, scout_job: Job, result: ScoutResult) -> None:
        """Persist one scout result onto its job and journal completion."""
        req_out: list[JSONValue] = []
        for req in result.follow_up_requests:
            agents_out: list[JSONValue] = [a for a in req.requesting_agents]
            req_out.append(
                {
                    "question": req.question,
                    "why_material": req.why_material,
                    "requested_source_domain": req.requested_source_domain,
                    "expected_gain": req.expected_gain,
                    "requesting_agents": agents_out,
                }
            )
        findings_out: list[JSONValue] = [
            {"text": c.text, "evidence_ids": list(c.evidence_ids)} for c in result.findings
        ]
        self.store.save_job(
            _jobs.complete_job(
                self.store.get_job(scout_job.job_id),
                result={
                    "assignment_id": assignment.assignment_id,
                    "findings": findings_out,
                    "coverage": result.coverage,
                    "unknowns": list(result.unknowns),
                    "limitations": list(result.limitations),
                    "follow_up_requests": req_out,
                },
            )
        )
        self._emit(sid, "job.completed", {"job_id": scout_job.job_id})

    def _scout_model_call(self, sid: str, prefix: str, scout_job: Job, scout_calls: list[int], prompt: str) -> str:
        """One scout model call with pre/post deadline guards; tags model origin on error."""
        _ = sid
        scout_calls[0] += 1
        from time import perf_counter as _pc

        from app.redact import redact_text as _rt

        _t0 = _pc()
        try:
            self._raise_if_deadline_exceeded("before model call")
            out = self.model(prompt)
            self._raise_if_deadline_exceeded("after model call")
            _dur = (_pc() - _t0) * 1000.0
            self._trace_record(
                "model.completed",
                {
                    "provider": self.provider,
                    "model": self.model_name,
                    "stage": f"{prefix}scout",
                    "job_id": scout_job.job_id,
                    "prompt": _rt(prompt)[:2000],
                    "output": _rt(out)[:2000],
                },
                _dur,
            )
            return out
        except Exception as exc:
            _dur2 = (_pc() - _t0) * 1000.0
            self._trace_record(
                "model.failed",
                {
                    "provider": self.provider,
                    "model": self.model_name,
                    "stage": f"{prefix}scout",
                    "job_id": scout_job.job_id,
                    "error": str(exc)[:2000],
                },
                _dur2,
            )
            setattr(exc, "_scout_origin", "model")  # noqa: B010 - dynamic boundary, no stubs; getattr keeps checker green
            raise

    def _run_scout_assignment(
        self,
        sid: str,
        assignment: ScoutAssignment,
        scout_job: Job,
        dispatch_fn: Callable[[str, dict[str, object]], dict[str, object]],
        model_fn: Callable[[str], str],
        journal_fn: Callable[[str, dict[str, object]], None] | None,
    ) -> ScoutResult:
        """Run one scout with deadline reset; cancels or fails the child on error."""
        from app.research.agents.scout import run_scout as _run_one

        try:
            try:
                return _run_one(assignment, dispatch=dispatch_fn, model=model_fn, journal=journal_fn)
            finally:
                self._scout_deadline = None
        except Exception as exc:
            origin = getattr(exc, "_scout_origin", "")
            pre_cat = getattr(exc, "_failure_category", None)
            msg = str(exc)[:2000] or type(exc).__name__
            if self._is_scout_cancelled(exc):
                self._cancel_scout_job(sid, scout_job, exc, pre_cat)
                raise
            cat = self._categorize_scout_error(exc, origin)
            self._fail_scout_job(sid, scout_job, exc, cat, msg)
            raise

    def _degrade_scout_assignment(
        self, sid: str, assignment: ScoutAssignment, scout_job: Job, exc: Exception
    ) -> ScoutResult:
        """Turn one failed scout assignment into an assignment-scoped limitation.

        A slow provider, a crashed model call, or a model-output failure means
        this assignment returned nothing - not that further research is
        impossible - so the source agent runs its remaining assignments and the
        wave closes through the normal empty/limitations path. The child job is
        already FAILED/CANCELLED (``_run_scout_assignment``); infrastructure
        faults (a tool dispatch that raises) and run-level policy/budget denials
        keep their fatal behavior, and cancellations always propagate.
        """
        cat = getattr(exc, "_failure_category", None)
        if self._is_scout_cancelled(exc):
            raise exc
        category = cat if isinstance(cat, FailureCategory) else FailureCategory.TOOL_ERROR
        if category not in SCOUT_ASSIGNMENT_ERRORS:
            raise exc
        detail = f"{type(exc).__name__}: {exc}"[:500]
        self._emit(
            sid,
            "scout.degraded",
            {
                "job_id": scout_job.job_id,
                "assignment_id": assignment.assignment_id,
                "failure_category": category.value,
                "error": detail,
            },
        )
        return ScoutResult(
            assignment_id=assignment.assignment_id,
            session_id=assignment.session_id,
            coverage=f"role={assignment.role} tickers={len(assignment.tickers)} tool_calls=0 failure={category.value}",
            findings=[],
            unknowns=["no PIT-eligible SEC evidence returned"],
            limitations=[f"scout {assignment.assignment_id} unavailable ({category.value}): {detail}"],
            follow_up_requests=[],
        )

    @staticmethod
    def _is_timeout_text(low: str) -> bool:
        """Timeout arm: timeout/expired/timed_out."""
        return "timeout" in low or "expired" in low or "timed_out" in low

    @staticmethod
    def _loop_category(low: str) -> FailureCategory | None:
        """Loop arms: research_loop_detected > research_loop > duplicate action."""
        if "research_loop_detected" in low or "research_loop" in low:
            return FailureCategory.RESEARCH_LOOP_DETECTED
        if "duplicate_research_action" in low:
            return FailureCategory.DUPLICATE_RESEARCH_ACTION
        return None

    @staticmethod
    def _policy_category(low: str) -> FailureCategory | None:
        """Policy arms: explicit rejection > generic policy/denied."""
        if "policy_rejection" in low:
            return FailureCategory.POLICY_REJECTION
        if "policy" in low or "denied" in low:
            return FailureCategory.POLICY_DENIED
        return None

    @staticmethod
    def _ladder_category(low: str) -> FailureCategory | None:
        """Shared keyword ladder: timeout, loop, policy; None falls through."""
        if _LiveRun._is_timeout_text(low):
            return FailureCategory.TIMEOUT
        hit = _LiveRun._loop_category(low)
        if hit is not None:
            return hit
        return _LiveRun._policy_category(low)

    @staticmethod
    def _categorize_fetch_error(exc: Exception) -> FailureCategory:
        """Category for a fetch failure: pre-tagged wins, else keyword ladder."""
        cat_raw = getattr(exc, "_failure_category", None)
        if isinstance(cat_raw, FailureCategory):
            return cat_raw
        low_all = (type(exc).__name__ + " " + str(exc)).lower()
        return _LiveRun._ladder_category(low_all) or FailureCategory.TOOL_ERROR

    def _fail_source_job(
        self,
        sid: str,
        src_job_id: str,
        cat: FailureCategory,
        message: str,
        exc: Exception,
        cancelled: bool,
    ) -> None:
        """Cancel or fail the source job on fetch error; silent when already closed."""
        try:
            src_job = self.store.get_job(src_job_id)
            if src_job.status in ("queued", "running"):
                if cancelled:
                    self.store.save_job(_jobs.cancel_job(src_job))
                    self._emit(sid, "job.cancelled", {"job_id": src_job_id})
                else:
                    self.store.save_job(_jobs.fail_job(src_job, cat, message))
                    self._emit(
                        sid,
                        "job.failed",
                        {
                            "job_id": src_job_id,
                            "stage": "source-scout",
                            "failure_category": cat.value,
                            "error": str(exc)[:2000],
                        },
                    )
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _fail_fetch_session(self, sid: str, cat: FailureCategory, message: str) -> None:
        """Stamp the fetch failure on the session and move it to FAILED."""
        try:
            sess_fail = self.store.get_session(sid)
            sess_fail = replace(
                sess_fail,
                failure=Failure(category=cat.value, message=message),
                updated_at=utcnow(),
            )
            if sess_fail.status not in ("failed", "completed", "cancelled"):
                sess_fail = _session.transition_session(sess_fail, SessionStatus.FAILED)
            self.store.save_session(sess_fail)
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _raise_fetch_failed(self, sid: str, src_job_id: str, exc: Exception) -> None:
        """Persist a fetch failure on every surface and raise LiveModelError."""
        cat = self._categorize_fetch_error(exc)
        cancelled = bool(getattr(exc, "_scout_cancelled", False))
        detail = f"{type(exc).__name__}: {exc}"[:2000]
        message = f"source-scout: {detail}"[:2000]
        self._fail_source_job(sid, src_job_id, cat, message, exc, cancelled)
        self._emit(
            sid,
            "model.failed",
            {
                "stage": "source-scout",
                "job_id": src_job_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            },
        )
        self._fail_fetch_session(sid, cat, message)
        self._emit(
            sid,
            "research.failed",
            {
                "stage": "source-scout",
                "failure_category": cat.value,
                "reason": "timeout:model-call" if cat == FailureCategory.TIMEOUT else f"scout:{cat.value}",
            },
        )
        self._emit(
            sid,
            "wave.stopped",
            {
                "reason": "timeout:model-call" if cat == FailureCategory.TIMEOUT else "failed:source-scout",
                "stage": "source-scout",
            },
        )
        self._save_budget_used(sid)
        try:
            if self.trace is not None:
                self.trace.finish(message[:2000], "failed")
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        raise LiveModelError(sid, "source-scout", detail, cat) from exc

    def _persist_fetch_success(self, sid: str, src_job_id: str, dossier_obj: object) -> list[str]:
        """Validate + persist the dossier, close the source job, return evidence ids."""
        ids: list[str] = []
        did: str = ""
        if isinstance(dossier_obj, SECDossier):
            ids = list(dossier_obj.supporting_evidence_ids)
            did = dossier_obj.dossier_id
            substantive = tuple(e.evidence_id for e in self.ledger.list_session(sid) if e.record_kind == "evidence")
            validate_dossier(dossier_obj, substantive)
            try:
                self.store.save_dossier(dossier_to_dict(dossier_obj))
            except ValueError:
                pass
        else:
            raise ValueError(f"runner: unexpected dossier type {type(dossier_obj).__name__} (SECDossier required)")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        sess = self.store.get_session(sid)
        sess = replace(
            sess,
            evidence_ids=list(dict.fromkeys([*sess.evidence_ids, *ids])),
            dossier_ids=list(dict.fromkeys([*sess.dossier_ids, did])),
            updated_at=utcnow(),
        )
        self.store.save_session(sess)
        self._emit(sid, "dossier.created", {"dossier_id": did, "evidence_ids": ids})
        if src_job_id:
            job = self.store.get_job(src_job_id)
            self.store.save_job(_jobs.complete_job(job, result={"dossier_id": did, "evidence_ids": ids}))
            self._emit(sid, "job.completed", {"job_id": src_job_id})
        self.dossier_ids.append(did)
        self._save_budget_used(sid)
        return ids

    def _fetch_wave(self, sid: str, wave: int, q: str, src_job_id: str, prefix: str) -> list[str]:
        existing_wave = sum(1 for e in self.ledger.list_session(sid) if e.wave_id == wave)
        counter: list[int] = [existing_wave]
        blocked: list[int] = [0]
        empty_actions: list[int] = [0]

        def _live_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
            if name != "call_tool":
                return self._catalog_dispatch(name, args)
            inner_obj: object = args.get("name")
            inner: str = inner_obj if isinstance(inner_obj, str) and inner_obj else "unknown_tool"
            before = len(self.loops.telemetry)
            raw, _dur = self._guarded_tool_call(sid, inner, name, args)
            if len(self.loops.telemetry) > before:
                blocked[0] += 1
                return raw
            if "error" in raw:
                # A rejected call is a failed action, never evidence: it is handed
                # back as-is (journaled by the guard) and stays out of ingest.
                empty_actions[0] += 1
                return self._register_rejected_action(inner, args, raw)
            counter[0] += 1
            eid: str = f"{sid}:{wave}:sec:{counter[0]}"
            record = self._build_evidence_record(sid, wave, q, src_job_id, inner, args, raw, eid)
            out = self._copy_navigation_packets(self._ingest_tool_evidence(sid, inner, args, record, eid, _dur), raw)
            produced = out.get("evidence_ids")
            self.loops.check(
                self._action_key(inner, args),
                evidence_content_hash(json.dumps(raw, sort_keys=True, default=str)),
                len(produced) if isinstance(produced, list) else 0,
            )
            if not produced:
                empty_actions[0] += 1
            return out

        scout_calls: list[int] = [0]

        def _scout_journal(event_type: str, payload: dict[str, object]) -> None:
            self._emit(sid, event_type, payload)

        def _scout_model(prompt: str) -> str:
            scout_calls[0] += 1
            stage: str = f"{prefix}scout-{scout_calls[0]}"
            return self._model_at_stage(stage, sid, src_job_id)(prompt)

        def _spawn_scout(assignment: ScoutAssignment) -> ScoutResult:
            existing_jobs: list[Job] = self.store.list_jobs(sid)
            reused = self._find_completed_scout(sid, src_job_id, assignment, existing_jobs)
            if reused is not None:
                return reused
            scout_job = self._open_or_reuse_scout_job(sid, wave, src_job_id, assignment, existing_jobs)
            self._seed_scout_deadline(scout_job, assignment)

            def _tagged_dispatch(name: str, args: dict[str, object]) -> dict[str, object]:
                try:
                    self._raise_if_deadline_exceeded("before dispatch ", name)
                    return _live_dispatch(name, args)
                except Exception as exc:
                    setattr(exc, "_scout_origin", "tool")  # noqa: B010 - dynamic boundary, no stubs; getattr keeps checker green
                    raise

            def _tagged_model(prompt: str) -> str:
                return self._scout_model_call(sid, prefix, scout_job, scout_calls, prompt)

            try:
                result = self._run_scout_assignment(
                    sid,
                    assignment,
                    scout_job,
                    _tagged_dispatch,
                    _tagged_model,
                    _scout_journal,
                )
            except Exception as exc:  # noqa: BLE001 - one assignment's failure, not the run's
                return self._degrade_scout_assignment(sid, assignment, scout_job, exc)
            self._complete_scout_job(sid, assignment, scout_job, result)
            return result

        try:
            dossier_obj: object = run_sec_assignment(
                q,
                session_id=sid,
                wave_id=wave,
                as_of=self.as_of_str or "unbounded",
                tickers=self.scoped,
                dispatch=_live_dispatch,
                model=_scout_model,
                journal=_scout_journal,
                spawn=_spawn_scout,
            )
        except Exception as exc:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            self._raise_fetch_failed(sid, src_job_id, exc)
            raise AssertionError("unreachable: _raise_fetch_failed always raises")
        prior = self.wave_actions.get(wave, {})
        self.wave_actions[wave] = {
            "duplicate_actions_blocked": prior.get("duplicate_actions_blocked", 0) + blocked[0],
            "zero_novelty_actions": prior.get("zero_novelty_actions", 0) + empty_actions[0],
        }
        return self._persist_fetch_success(sid, src_job_id, dossier_obj)

    def _fetch(self, session_id: str) -> list[str]:
        src: str = self.source_jobs[0] if self.source_jobs else ""
        return self._fetch_wave(session_id, self.wave_id, self.question, src, "")

    def _freeze_wave(self, session_id: str, wave: int) -> str:
        sess = self.store.get_session(session_id)
        if sess.status in (
            SessionStatus.RESEARCHING.value,
            SessionStatus.TARGETED_RESEARCH.value,
        ):
            sess = _session.transition_session(sess, SessionStatus.FREEZING)
            self.store.save_session(sess)
        recs: list[Evidence] = [
            e for e in self.ledger.list_session(session_id) if e.wave_id <= wave and e.record_kind == "evidence"
        ]
        fid: str = f"{session_id}:{wave}:freeze"
        frozen = _freeze.create_freeze(
            freeze_id=fid,
            session_id=session_id,
            wave_id=wave,
            records=recs,
            as_of=self.as_of,
        )
        try:
            self.store.save_freeze(_freeze.freeze_to_dict(frozen))
        except ValueError:
            pass
        cur = self.store.get_session(session_id)
        if fid not in cur.freeze_ids:
            cur = replace(cur, freeze_ids=[*cur.freeze_ids, fid], updated_at=utcnow())
            self.store.save_session(cur)
        self._emit(
            session_id,
            "freeze.created",
            {"freeze_id": fid, "evidence_ids": list(frozen.evidence_ids)},
        )
        return fid

    def _freeze_evidence_text(self, session_id: str, ev_ids: Sequence[str]) -> str:
        """One shared serialization of the exact freeze records for all committee members."""
        by_id = {e.evidence_id: e for e in self.ledger.list_session(session_id)}
        lines: list[str] = []
        for eid in ev_ids:
            rec = by_id.get(eid)
            if rec is None:
                continue
            known = rec.known_at.isoformat() if rec.known_at else "unknown"
            content = rec.content if len(rec.content) <= 1000 else rec.content[:1000] + " […truncated]"
            lines.append(
                f"[{rec.evidence_id}] {rec.subject} | {known} | {rec.source_name} {rec.source_uri or ''}".rstrip()
            )
            lines.append(f"claim_kind: {rec.claim_kind}")
            lines.append(f"claim: {rec.claim_text}")
            lines.append(f"content: {content}")
        return "\n".join(lines).replace("\x00", "")  # never hand NUL bytes to a prompt

    def _create_freeze(self, session_id: str) -> str:
        return self._freeze_wave(session_id, self.wave_id)

    @staticmethod
    def _categorize_committee_error(exc: Exception) -> FailureCategory:
        """Category for a committee failure: pre-tagged wins, else keyword ladder."""
        pre = getattr(exc, "_failure_category", None)
        if isinstance(pre, FailureCategory) or isinstance(exc, TimeoutError):
            return pre if isinstance(pre, FailureCategory) else FailureCategory.TIMEOUT
        msg = str(exc)[:1500] or type(exc).__name__
        low = (type(exc).__name__ + " " + msg).lower()
        if (hit := _LiveRun._ladder_category(low)) is not None and hit != FailureCategory.TIMEOUT:
            return hit
        for needles, category in _COMMITTEE_KEYWORDS:
            if any(needle in low for needle in needles):
                return category
        # No timeout evidence: a real timeout is a TimeoutError or says so above.
        return FailureCategory.MODEL_ERROR

    def _fail_committee_jobs(
        self,
        session_id: str,
        ran: list[str],
        exc: Exception,
        cat: FailureCategory,
        msg: str,
    ) -> None:
        """Fail every still-open committee job, then journal each one."""
        for jid in ran:
            with self._lock:
                leftover: Job = self.store.get_job(jid)
                if leftover.status in ("queued", "running"):
                    self.store.save_job(
                        _jobs.fail_job(
                            leftover,
                            cat,
                            f"committee:{type(exc).__name__}:{msg}"[:2000],
                        )
                    )
            self._emit(session_id, "job.failed", {"job_id": jid, "failure_category": cat.value})

    def _open_committee_jobs(
        self,
        session_id: str,
        wave: int,
        pending: list[Job],
        progressed: ResearchSession,
        ran: list[str],
    ) -> ResearchSession:
        """Create + start the stock/bull/bear trio; rolls back started jobs on partial failure."""
        prog: ResearchSession = progressed
        try:
            for kind in (JobType.STOCKBOT, JobType.BULLBOT, JobType.BEARBOT):
                updated: tuple[ResearchSession, Job] = _jobs.create_job(
                    prog,
                    pending,
                    job_type=kind,
                    owner="runner",
                    wave_id=wave,
                )
                prog, job = updated
                self.store.save_session(prog)
                self.store.save_job(job)
                self._emit(
                    session_id,
                    "job.created",
                    {"job_id": job.job_id, "job_type": job.job_type},
                )
                started = _jobs.start_job(job)
                self.store.save_job(started)
                pending.append(started)
                ran.append(started.job_id)
        except Exception as exc:
            # A half-created trio would leave RUNNING jobs pinning the session; fail them closed.
            self._fail_committee_jobs(
                session_id,
                ran,
                exc,
                self._categorize_committee_error(exc),
                f"partial trio: {type(exc).__name__}",
            )
            raise
        return prog

    def _committee_verify(self, session_id: str, fid: str) -> None:
        """Recompute the freeze hash from ledger records; raise on drift (fail-closed write)."""
        raw_ids = self.store.get_freeze(fid).get("evidence_ids", [])
        want: set[str] = {e for e in raw_ids if isinstance(e, str)} if isinstance(raw_ids, list) else set()
        recs = [e for e in self.ledger.list_session(session_id) if e.evidence_id in want]
        _freeze.verify_freeze(_freeze.freeze_from_dict(self.store.get_freeze(fid)), recs)

    @staticmethod
    def _committee_result(role: str, analysis: object, fid: str) -> dict[str, object]:
        """Persisted committee envelope: the analysis shape service re-validates on read.

        The full rich envelope (executive_view, typed claims, impact_channels,
        materiality, uncertainties, what_would_change, follow_ups) is what makes
        the persisted job replayable; a claims-only result cannot be rebuilt.
        """
        from dataclasses import asdict, is_dataclass

        payload: dict[str, object] = (
            dict(asdict(analysis)) if is_dataclass(analysis) and not isinstance(analysis, type) else {}
        )
        raw_requests: object = payload.get("research_requests")
        requests: object = raw_requests if isinstance(raw_requests, list) else []
        payload.update({"role": role, "freeze_id": fid, "follow_ups": requests})
        return payload

    def _run_member_stock(
        self,
        session_id: str,
        run_fn: Callable[[], StockbotAnalysis],
        job_id: str,
        fid: str,
    ) -> StockbotAnalysis:
        """Run the stockbot member and mark its job complete; exceptions propagate."""
        analysis = run_fn()
        with self._lock:
            self._committee_verify(session_id, fid)
            result = self._committee_result("stockbot", analysis, fid)
            self.store.save_job(_jobs.complete_job(self.store.get_job(job_id), result=result))
        return analysis

    def _run_member_bull(
        self, session_id: str, run_fn: Callable[[], BullAnalysis], job_id: str, fid: str
    ) -> BullAnalysis:
        """Run the bullbot member and mark its job complete; exceptions propagate."""
        analysis = run_fn()
        with self._lock:
            self._committee_verify(session_id, fid)
            result = self._committee_result("bullbot", analysis, fid)
            self.store.save_job(_jobs.complete_job(self.store.get_job(job_id), result=result))
        return analysis

    def _run_member_bear(
        self, session_id: str, run_fn: Callable[[], BearAnalysis], job_id: str, fid: str
    ) -> BearAnalysis:
        """Run the bearbot member and mark its job complete; exceptions propagate."""
        analysis = run_fn()
        with self._lock:
            self._committee_verify(session_id, fid)
            result = self._committee_result("bearbot", analysis, fid)
            self.store.save_job(_jobs.complete_job(self.store.get_job(job_id), result=result))
        return analysis

    def _run_trio(
        self,
        stock_fn: Callable[[], StockbotAnalysis],
        bull_fn: Callable[[], BullAnalysis],
        bear_fn: Callable[[], BearAnalysis],
    ) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        """Run the three member closures in parallel; first exception wins."""
        with ThreadPoolExecutor(max_workers=3) as _pool:
            stock_f = _pool.submit(stock_fn)
            bull_f = _pool.submit(bull_fn)
            bear_f = _pool.submit(bear_fn)
            return stock_f.result(), bull_f.result(), bear_f.result()

    def _committee_wave(
        self, session_id: str, wave: int, prefix: str
    ) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        sess = self.store.get_session(session_id)
        if sess.status == SessionStatus.FREEZING.value:
            sess = _session.transition_session(sess, SessionStatus.ANALYZING)
            self.store.save_session(sess)
        fid: str = f"{session_id}:{wave}:freeze"
        frozen_ids: object = self.store.get_freeze(fid).get("evidence_ids")
        ev_ids: list[str] = [e for e in frozen_ids if isinstance(e, str)] if isinstance(frozen_ids, list) else []
        pending: list[Job] = list(self.store.list_jobs(session_id))
        ran: list[str] = []
        self._open_committee_jobs(session_id, wave, pending, self.store.get_session(session_id), ran)
        shared_text = self._freeze_evidence_text(session_id, ev_ids)

        def _run_stock() -> StockbotAnalysis:
            return self._run_member_stock(
                session_id,
                lambda: run_stockbot(
                    self.question,
                    session_id=session_id,
                    wave_id=wave,
                    freeze_id=fid,
                    evidence_ids=ev_ids,
                    as_of=self.as_of_str or "unbounded",
                    model=self._model_at_stage(f"{prefix}committee-stockbot", session_id, ran[0]),
                    evidence_text=shared_text,
                ),
                ran[0],
                fid,
            )

        def _run_bull() -> BullAnalysis:
            return self._run_member_bull(
                session_id,
                lambda: run_bullbot(
                    self.question,
                    session_id=session_id,
                    wave_id=wave,
                    freeze_id=fid,
                    evidence_ids=ev_ids,
                    as_of=self.as_of_str or "unbounded",
                    model=self._model_at_stage(f"{prefix}committee-bullbot", session_id, ran[1]),
                    evidence_text=shared_text,
                ),
                ran[1],
                fid,
            )

        def _run_bear() -> BearAnalysis:
            return self._run_member_bear(
                session_id,
                lambda: run_bearbot(
                    self.question,
                    session_id=session_id,
                    wave_id=wave,
                    freeze_id=fid,
                    evidence_ids=ev_ids,
                    as_of=self.as_of_str or "unbounded",
                    model=self._model_at_stage(f"{prefix}committee-bearbot", session_id, ran[2]),
                    evidence_text=shared_text,
                ),
                ran[2],
                fid,
            )

        try:
            stock, bull, bear = self._run_trio(_run_stock, _run_bull, _run_bear)
        except Exception as exc:
            cat = self._categorize_committee_error(exc)
            msg = str(exc)[:1500] or type(exc).__name__
            self._fail_committee_jobs(session_id, ran, exc, cat, msg)
            raise
        self._emit(
            session_id,
            "committee.completed",
            {"freeze_id": fid, "evidence_ids": ev_ids},
        )
        entry_jobs: list[JSONValue] = list(ran)
        entry: dict[str, JSONValue] = {
            "freeze_id": fid,
            "wave_id": wave,
            "jobs": entry_jobs,
        }
        latest = self.store.get_session(session_id)
        latest = replace(latest, committee_runs=[*latest.committee_runs, entry], updated_at=utcnow())
        self.store.save_session(latest)
        return (stock, bull, bear)

    def _run_committee(self, session_id: str) -> tuple[StockbotAnalysis, BullAnalysis, BearAnalysis]:
        return self._committee_wave(session_id, self.wave_id, "")

    def _record_stop(self, session_id: str, reason: str) -> None:
        self._emit(session_id, "wave.stopped", {"reason": reason})

    def _store_final(self, session_id: str, freeze_id: str, answer: str, claims: Sequence[object]) -> None:
        """Persist the synthesis answer, then close the session as completed."""
        from app.research.agents import GroundedClaim as _GC

        claims_json: list[JSONValue] = []
        for claim in claims:
            if isinstance(claim, _GC):
                claims_json.append(
                    {
                        "text": claim.text,
                        "claim_type": claim.claim_type,
                        "evidence_ids": list(claim.evidence_ids),
                    }
                )
        final: dict[str, JSONValue] = {
            "answer": answer,
            "freeze_id": freeze_id,
            "claims": claims_json,
        }
        cur = self.store.get_session(session_id)
        cur = replace(cur, final_result=final, updated_at=utcnow())
        self.store.save_session(cur)
        if cur.status == SessionStatus.SYNTHESIZING.value:
            cur = _session.transition_session(cur, SessionStatus.COMPLETED)
            self.store.save_session(cur)

    def _bundle_root(self) -> Path:
        """Per-session bundle root under the data root; Path, never a warehouse DB."""
        from app.config import get_data_root as _gdr

        return _gdr() / "bundles"

    def _write_bundle_file(self, run_dir: Path, name: str, payload: object) -> None:
        """Write one JSON bundle file; raises on IO failure."""
        import json as _json

        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n")

    def _bundle_evidence_entries(self, session_id: str) -> tuple[list[object], list[str]]:
        """Bundle evidence entries + selective artifact names for substantive rows only."""
        from app.research.evidence import evidence_bundle_entry as _entry
        from app.research.evidence import evidence_from_dict as _from_dict
        from app.storage import raw_archive as _artifacts

        entries: list[object] = []
        artifacts: list[str] = []
        seen: set[str] = set()
        rows = list(self.store.list_evidence(session_id))
        ledger_rows = _ledger_evidence(self.ledger, session_id)
        by_id = {rec.evidence_id: rec for rec in ledger_rows}
        for row in rows:
            if not isinstance(row, dict):
                continue
            eid = row.get("evidence_id")
            if not isinstance(eid, str) or not eid or eid in seen:
                continue
            rec = by_id.get(eid)
            if rec is None:
                rec = _from_dict(row)
            if rec.record_kind != "evidence":
                continue
            entries.append(_entry(rec))
            seen.add(eid)
            artifacts.append(f"sha256_{rec.content_hash}")
            _artifacts.store_evidence_artifact(
                rec.content.encode("utf-8"),
                url=rec.source_uri or "",
                metadata={"evidence_id": eid, "session_id": session_id},
            )
        return entries, artifacts

    def _bundle_tool_calls(self, session_id: str) -> list[object]:
        """Tool-call rows for the bundle: tool-carrying journal events in sequence order."""
        out: list[object] = []
        events = self.store.list_events(session_id)
        for event in events:
            row = event.to_dict()
            event_type = str(row.get("event_type", ""))
            payload = row.get("payload")
            text = f"{event_type} {payload}".lower()
            if event_type.startswith(("tool.", "discovery.", "evidence.", "scout.", "job.")) or "tool" in text:
                out.append(row)
        return out

    def _bundle_agents(self, session_id: str) -> dict[str, object]:
        """Committee agent payloads for the bundle: completed trio job results keyed by role."""
        agents: dict[str, object] = {}
        jobs = self.store.list_jobs(session_id)
        for job in jobs:
            if job.job_type not in ("stockbot", "bullbot", "bearbot") or job.status != "completed":
                continue
            if job.result is None:
                continue
            agents[job.job_type] = dict(job.result)
        return agents

    def write_bundle(self, session_id: str) -> Path:
        """Per-session JSON evidence bundle: disposable export view, never a source of truth.

        ResearchRepository/SQLite owns sessions, jobs, evidence, freezes,
        dossiers, final results, and traces; raw_archive owns accepted source
        artifacts. Bundles under ``<data_root>/bundles`` are assembled from
        those two and nothing reads them back (no resume/verify path touches
        ``bundles/``). Selective artifacts: substantive evidence rows only
        (content-addressed sha256 names, no per-run copies); search lists,
        unused opens, and PDFs stay trace-metadata-only.

        Raises on missing session or IO failure; callers let it propagate
        after state is persisted so a bundle failure never hides as partial files.
        """
        import json as _json

        run_dir = self._bundle_root() / str(session_id)
        session = self.store.get_session(session_id)
        question = session.query
        as_of = session.as_of.isoformat() if session.as_of is not None else None
        final: dict[str, JSONValue] = dict(session.final_result) if session.final_result is not None else {}
        policy: dict[str, JSONValue] = dict(session.policy)
        entries, artifacts = self._bundle_evidence_entries(session_id)
        tool_calls = self._bundle_tool_calls(session_id)
        agents = self._bundle_agents(session_id)
        answer = final.get("answer", "") if isinstance(final, dict) else ""
        self._write_bundle_file(run_dir, "request.json", {"question": question, "as_of": as_of})
        self._write_bundle_file(run_dir, "config.json", policy)
        calls_path = run_dir / "tool_calls.jsonl"
        calls_path.parent.mkdir(parents=True, exist_ok=True)
        with calls_path.open("w", encoding="utf-8") as handle:
            for row in tool_calls:
                handle.write(_json.dumps(row, sort_keys=True, default=str) + "\n")
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            eid = entry.get("evidence_id")
            if not isinstance(eid, str) or not eid:
                continue
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in eid)
            (evidence_dir / f"{safe}.json").write_text(_json.dumps(entry, sort_keys=True, indent=2, default=str) + "\n")
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        agents_dir = run_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        for role, payload in agents.items():
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in role)
            (agents_dir / f"{safe}.json").write_text(_json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n")
        self._write_bundle_file(run_dir, "answer.json", {"answer": answer, "final_result": final})
        self._write_bundle_file(run_dir, "eval.json", {"session_id": session_id, "artifacts": artifacts})
        return run_dir

    def _store_empty_terminal(self, session_id: str) -> None:
        """Persist completed no-evidence result (limitations answer, empty claims)."""
        try:
            empty: dict[str, JSONValue] = {
                "answer": (
                    "No PIT-eligible evidence was found; the question cannot be answered "
                    "from the allowed sources within the session scope. Limitations: "
                    "searched-source scope, as_of-filtered corpus."
                ),
                "freeze_id": "",
                "claims": [],
            }
            cur = self.store.get_session(session_id)
            cur = replace(cur, final_result=empty, updated_at=utcnow())
            if cur.status not in ("failed", "completed", "cancelled"):
                cur = _session.transition_session(cur, SessionStatus.COMPLETED)
            self.store.save_session(cur)
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass
        try:
            if self.trace is not None:
                self.trace.finish("complete:empty-with-limitations", "completed")
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _dossier_row(self, sid: str, wave: int) -> dict[str, object]:
        """Persisted dossier row for one wave; {} when that wave produced none."""
        try:
            row = self.store.get_dossier(f"{sid}:{wave}:sec")
        except KeyError:
            return {}
        return dict(row) if isinstance(row, Mapping) else {}

    def _attach_dossier_state(self, result: Wave1Result) -> None:
        """Attach the wave's persisted coverage/relationships/open questions for the gate."""
        row = self._dossier_row(result.session_id, result.wave_id)
        coverage = row.get("coverage")
        raw_rels = row.get("relationships")
        raw_open = row.get("open_questions")
        result.coverage = dict(coverage) if isinstance(coverage, Mapping) else None
        result.relationships = (
            [dict(r) for r in raw_rels if isinstance(r, Mapping)] if isinstance(raw_rels, list) else []
        )
        result.open_questions = (
            [q for q in raw_open if isinstance(q, str) and q.strip()] if isinstance(raw_open, list) else []
        )

    def _wave_items(self, sid: str, wave: int) -> dict[str, set[str]]:
        """Raw items of one wave: evidence rows, documents, entities, relationships, claims, questions.

        ``evidence_keys`` is the (accession, document, content_hash) identity of
        each row: re-reading a document for a passage the run already holds is a
        record, never new evidence.
        """
        rows = _ledger_evidence(self.ledger, sid, wave)
        rels, claims, questions = _dossier_items(self._dossier_row(sid, wave))
        documents: set[str] = set()
        evidence_keys: set[str] = set()
        for rec in rows:
            accession, document = _row_identity(rec)
            key = f"{accession}|{document}"
            if accession or document:
                documents.add(key)
            evidence_keys.add(f"{key}|{rec.content_hash}")
        return {
            "evidence": {e.evidence_id for e in rows},
            "evidence_keys": evidence_keys,
            "documents": documents,
            "entities": {e.subject for e in rows if e.subject},
            "relationships": rels,
            "claims": claims,
            "questions": questions,
        }

    def _wave_novelty(self, sid: str, wave: int) -> dict[str, object]:
        """Persisted deltas for one wave, plus the branch's zero-novelty streak.

        Novelty is measured from what actually landed: new (accession, document)
        pairs, evidence (document + passage) not already held by an earlier
        wave, entities, relationship ids, claims, and questions that stopped
        being open. A wave with none of the four gate keys is zero-novelty: one
        such wave may retry the branch, two in a row stop it.
        """
        items = self._wave_items(sid, wave)
        prior: dict[str, set[str]] = {key: set() for key in items}
        for earlier, seen in self.wave_items.items():
            if earlier == wave:
                continue
            for key, values in seen.items():
                prior.setdefault(key, set()).update(values)
        novelty: dict[str, object] = {
            "wave_id": wave,
            "new_raw_documents": len(items["documents"] - prior["documents"]),
            "new_evidence_records": len(items["evidence_keys"] - prior["evidence_keys"]),
            "new_entities": len(items["entities"] - prior["entities"]),
            "new_relationships": len(items["relationships"] - prior["relationships"]),
            "new_material_claims": len(items["claims"] - prior["claims"]),
            "resolved_questions": len(prior["questions"] - items["questions"]),
            "new_questions": len(items["questions"] - prior["questions"]),
        }
        zero = not any(novelty[key] for key in NOVELTY_ZERO_KEYS)
        self.zero_novelty_waves = self.zero_novelty_waves + 1 if zero else 0
        counters = self.wave_actions.get(wave, {})
        novelty["zero_novelty_waves"] = self.zero_novelty_waves
        novelty["duplicate_actions_blocked"] = counters.get("duplicate_actions_blocked", 0)
        novelty["zero_novelty_actions"] = counters.get("zero_novelty_actions", 0)
        self.wave_items[wave] = items
        self._emit(sid, "wave.novelty", dict(novelty))
        return novelty

    def _set_branch(self, targeted: str) -> None:
        """Zero-novelty streaks are per branch: a new branch starts at zero."""
        if targeted != self.branch:
            self.branch = targeted
            self.zero_novelty_waves = 0

    def _run_next_wave(self, result: Wave1Result, targeted: str) -> dict[str, object] | None:
        """One more SEC wave: source job -> fetch -> freeze -> committee; None when empty."""
        sid: str = result.session_id
        wave: int = result.wave_id + 1
        self._set_branch(targeted)
        sess = self.store.get_session(sid)
        if sess.status == SessionStatus.ANALYZING.value:
            sess = _session.transition_session(sess, SessionStatus.TARGETED_RESEARCH)
            self.store.save_session(sess)
        question = targeted or self.question
        sess = replace(self.store.get_session(sid), current_wave=wave, updated_at=utcnow())
        self.store.save_session(sess)
        self._emit(sid, "wave.started", {"wave_id": wave, "targeted_question": targeted})
        src: str = self._open_source_job(sid, wave, question)
        eids: list[str] = self._fetch_wave(sid, wave, question, src, f"w{wave}-")
        if not eids:
            self._emit(sid, "wave.stopped", {"reason": "complete:empty-with-limitations"})
            return None
        fid: str = self._freeze_wave(sid, wave)
        stock, bull, bear = self._committee_wave(sid, wave, f"w{wave}-")
        disagreement: CommitteeDisagreement = compute_disagreement(stock, bull, bear)
        merged: CommitteeDisagreement = (
            _merge_disagreement(result.disagreement, disagreement) if result.disagreement is not None else disagreement
        )
        next_result = Wave1Result(
            session_id=sid,
            wave_id=wave,
            freeze_id=fid,
            evidence_ids=eids,
            stock=stock,
            bull=bull,
            bear=bear,
            disagreement=merged,
            question=result.question,
        )
        self._attach_dossier_state(next_result)
        return {
            "wave_id": wave,
            "targeted": targeted,
            "freeze_id": fid,
            "evidence_ids": eids,
            "dossier_id": self.dossier_ids[-1] if self.dossier_ids else "",
            "stock": stock,
            "bull": bull,
            "bear": bear,
            "disagreement": merged,
            "result": next_result,
        }

    def _deps(self, interrupt_after: str | None = None) -> DirectorDeps:
        return DirectorDeps(
            create_session=lambda q, aof: self._create_session(q, aof, interrupt_after),
            fetch_wave_evidence=self._fetch,
            create_freeze=self._create_freeze,
            run_committee=self._run_committee,
            record_stop=self._record_stop,
        )

    def _advance_to_synthesizing(self, session_id: str) -> None:
        """Move an ANALYZING session to SYNTHESIZING; other states pass through."""
        sess = self.store.get_session(session_id)
        if sess.status == SessionStatus.ANALYZING.value:
            sess = _session.transition_session(sess, SessionStatus.SYNTHESIZING)
            self.store.save_session(sess)

    def _close_trace(self, conclusion: object, fallback: str) -> None:
        """Finish the eval trace with the synthesis answer; never raises."""
        try:
            if self.trace is not None:
                text = conclusion if isinstance(conclusion, str) and conclusion else fallback
                self.trace.finish(text[:2000], "completed")
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
            pass

    def _complete_wave_tail(
        self,
        result: Wave1Result,
        did_out: str,
        gate: str,
        waves: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Synthesis tail for the final wave: persist over its freeze, stop, trace, bundle."""
        reason: str = f"complete:wave{result.wave_id}"
        self._advance_to_synthesizing(result.session_id)
        synth = synthesize_wave1(self.question, self.as_of_str or "unbounded", result)
        if synth is not None:
            self._store_final(result.session_id, synth.freeze_id, synth.answer, synth.claims)
        self._emit(result.session_id, "wave.stopped", {"reason": reason})
        self._close_trace(synth.answer if synth is not None else None, reason)
        bundle_dir = self.write_bundle(result.session_id)
        return {
            "session_id": result.session_id,
            "wave_id": result.wave_id,
            "freeze_id": result.freeze_id,
            "evidence_ids": list(result.evidence_ids),
            "dossier_id": did_out,
            "stock": result.stock,
            "bull": result.bull,
            "bear": result.bear,
            "disagreement": result.disagreement,
            "stop_reason": reason,
            "wave_decision": gate,
            "novelty": dict(result.novelty),
            "waves": [_wave_summary(w) for w in waves],
            "bundle_dir": str(bundle_dir),
        }

    def _finish_completed(self, result: Wave1Result, did_out: str) -> dict[str, object]:
        """Sequential wave loop: gate each wave, run the next, synthesize when the gate stops."""
        waves: list[dict[str, object]] = []
        while True:
            self._attach_dossier_state(result)
            novelty = self._wave_novelty(result.session_id, result.wave_id)
            result.novelty = novelty
            decision = decide_next_wave(
                result,
                deps=self._deps(),
                budgets=self.limits,
                waves_used=result.wave_id,
                jobs_used=len(self.store.list_jobs(result.session_id)),
                tool_calls_used=self.budget.used,
                elapsed_s=monotonic() - self.t0,
                novelty=novelty,
            )
            gate: str = f"{decision.stop_reason}:{decision.reason_detail}"
            if not decision.authorized:
                return self._complete_wave_tail(result, did_out, gate, waves)
            wave = self._run_next_wave(result, decision.targeted_question)
            if wave is None:
                return self._complete_wave_tail(result, did_out, gate, waves)
            waves.append(wave)
            next_result: object = wave["result"]
            assert isinstance(next_result, Wave1Result)
            result, did_out = next_result, str(wave["dossier_id"])


def _wave_summary(wave: Mapping[str, object]) -> dict[str, object]:
    """Compact per-wave record for the run payload (no analysis objects inline)."""
    evidence = wave.get("evidence_ids")
    return {
        "wave_id": wave.get("wave_id"),
        "targeted": wave.get("targeted"),
        "freeze_id": wave.get("freeze_id"),
        "evidence_ids": list(evidence) if isinstance(evidence, list) else [],
        "dossier_id": wave.get("dossier_id"),
        "disagreement": wave.get("disagreement"),
    }


def write_session_bundle(run: _LiveRun, session_id: str) -> Path:
    """Per-session bundle writer: delegates to the run's bundle writer (thin seam)."""
    return run.write_bundle(session_id)


def _empty_terminal_result(
    session_id: str, wave_id: int, eids: list[str], dossier_id: str, reason: str
) -> dict[str, object]:
    """Terminal no-evidence result payload shared by run paths."""
    return {
        "session_id": session_id,
        "wave_id": wave_id,
        "freeze_id": "",
        "evidence_ids": eids,
        "dossier_id": dossier_id,
        "stock": None,
        "bull": None,
        "bear": None,
        "disagreement": None,
        "stop_reason": reason,
    }


def _run_one_committee(
    run: _LiveRun, store: ResearchRepository, question: str, as_of_str: str, wave_id: int
) -> dict[str, object]:
    """Interrupted single-member path: fetch, freeze, stock-only analysis, stop."""
    sid = run._create_session(question, as_of_str, "one-committee")
    eids = run._fetch(sid)
    if not eids:
        run._emit(sid, "wave.stopped", {"reason": "complete:empty-with-limitations"})
        run._store_empty_terminal(sid)
        return _empty_terminal_result(
            sid, wave_id, eids, run.dossier_ids[0] if run.dossier_ids else "", "complete:empty-with-limitations"
        )
    fid = run._create_freeze(sid)
    sess = store.get_session(sid)
    if sess.status == SessionStatus.FREEZING.value:
        sess = _session.transition_session(sess, SessionStatus.ANALYZING)
        store.save_session(sess)
    existing_jobs = store.list_jobs(sid)
    sess, one = _jobs.create_job(
        sess,
        existing_jobs,
        job_type=JobType.STOCKBOT,
        owner="runner",
        wave_id=wave_id,
    )
    store.save_session(sess)
    store.save_job(one)
    run._emit(sid, "job.created", {"job_id": one.job_id, "job_type": one.job_type})
    store.save_job(_jobs.start_job(one))
    stock = run_stockbot(
        question,
        session_id=sid,
        wave_id=wave_id,
        freeze_id=fid,
        evidence_ids=eids,
        as_of=as_of_str or "unbounded",
        model=run._model_at_stage("committee-stockbot", sid, one.job_id),
        evidence_text=run._freeze_evidence_text(sid, eids),
    )
    store.save_job(_jobs.complete_job(store.get_job(one.job_id), result={"freeze_id": fid}))
    one_jobs: list[JSONValue] = [one.job_id]
    one_entry: dict[str, JSONValue] = {"freeze_id": fid, "wave_id": wave_id, "jobs": one_jobs}
    cur = store.get_session(sid)
    cur = replace(cur, committee_runs=[*cur.committee_runs, one_entry], updated_at=utcnow())
    store.save_session(cur)
    run._emit(sid, "committee.completed", {"freeze_id": fid, "ran": ["stockbot"]})
    run._emit(sid, "wave.stopped", {"reason": "interrupted:one-committee"})
    return {
        "session_id": sid,
        "wave_id": wave_id,
        "freeze_id": fid,
        "evidence_ids": eids,
        "dossier_id": run.dossier_ids[0] if run.dossier_ids else "",
        "stock": stock,
        "bull": None,
        "bear": None,
        "disagreement": None,
        "stop_reason": "interrupted:one-committee",
    }


def _close_wave1_result(
    run: _LiveRun, result: Wave1Result, did_out: str, interrupt_after: str | None
) -> dict[str, object]:
    """Close the opened wave: sequential loop when the trio ran, else interrupt/empty payload."""
    if (
        result.stock is not None
        and result.bull is not None
        and result.bear is not None
        and result.disagreement is not None
    ):
        return run._finish_completed(result, did_out)
    if interrupt_after in ("source", "freeze"):
        return {
            "session_id": result.session_id,
            "wave_id": result.wave_id,
            "freeze_id": result.freeze_id,
            "evidence_ids": list(result.evidence_ids),
            "dossier_id": did_out,
            "stock": None,
            "bull": None,
            "bear": None,
            "disagreement": None,
            "stop_reason": f"interrupted:{interrupt_after}",
        }
    run._store_empty_terminal(result.session_id)
    bundle_dir = run.write_bundle(result.session_id)
    out = _empty_terminal_result(
        result.session_id, result.wave_id, list(result.evidence_ids), did_out, "complete:empty-with-limitations"
    )
    out["bundle_dir"] = str(bundle_dir)
    return out


def run_live(
    question: str,
    objective: str,
    as_of: str | None,
    tickers: Sequence[str],
    dispatch: Callable[[str, dict[str, object]], dict[str, object]],
    model: Callable[[str], str],
    repo: ResearchRepository | None = None,
    budgets: DirectorBudgets | None = None,
    *,
    interrupt_after: str | None = None,
    wave_id: int = 1,
    provider: str = "fake",
    model_name: str | None = None,
) -> dict[str, object]:
    """Retired production entrypoint (deterministic test helper only).

    Live evals run the production kernel path (kernel scheduler + JEV +
    Needle + tool runtime; see scripts/verify_agent_scenarios.py). This in-process
    loop must not gain new callers. Waves run sequentially while
    ``decide_next_wave`` authorizes them; ``interrupt_after`` stops early at a
    named boundary.
    """
    if interrupt_after not in (None, "source", "freeze", "one-committee"):
        raise ValueError(f"runner: 'interrupt_after' must be source/freeze/one-committee, got {interrupt_after!r}")
    store: ResearchRepository = repo if repo is not None else ResearchRepository()
    limits: DirectorBudgets = budgets if budgets is not None else DirectorBudgets()
    as_of_str: str = as_of.strip() if isinstance(as_of, str) and as_of.strip() else ""
    scoped: list[str] = list(tickers)
    run = _LiveRun(
        store,
        question,
        objective,
        as_of,
        as_of_str,
        scoped,
        dispatch,
        model,
        limits,
        wave_id,
        provider=provider,
        model_name=model_name,
    )
    if interrupt_after == "one-committee":
        return _run_one_committee(run, store, question, as_of_str, wave_id)
    deps = run._deps(interrupt_after)
    result = run_wave1(
        question,
        as_of_str or "unbounded",
        deps=deps,
        tickers=scoped,
        wave_id=wave_id,
        interrupt_after=interrupt_after,
    )
    return _close_wave1_result(run, result, run.dossier_ids[0] if run.dossier_ids else "", interrupt_after)


def _job_diagnostics_of(job: Job) -> dict[str, JSONValue]:
    return dict(job.diagnostics)


def _job_tickers(job: Job) -> list[str]:
    """Tickers from one job's diagnostics; empty when absent or malformed."""
    diag = _job_diagnostics_of(job)
    raw_tick: object = diag.get("tickers")
    if isinstance(raw_tick, list) and raw_tick:
        tickers = [t.strip() for t in raw_tick if isinstance(t, str) and t.strip()]
        if tickers:
            return tickers
    return []


def _scoped_from_jobs(store: ResearchRepository, session_id: str, wave: int) -> list[str]:
    """Tickers from the wave's source-job diagnostics; empty when absent."""
    for job in store.list_jobs(session_id):
        if job.job_type != JobType.SOURCE_AGENT.value or job.wave_id != wave:
            continue
        if found := _job_tickers(job):
            return found
    return []


def _scoped_from_evidence(store: ResearchRepository, session_id: str) -> list[str]:
    """Tickers from persisted evidence metadata; empty when absent."""
    for rec_dict in store.list_evidence(session_id):
        meta = rec_dict.get("metadata")
        if isinstance(meta, dict):
            tickers_raw = meta.get("tickers")
            if isinstance(tickers_raw, str) and tickers_raw.strip():
                return [t.strip() for t in tickers_raw.split(",") if t.strip()]
    return []


def _resolve_resume_scoped(store: ResearchRepository, session_id: str, wave: int) -> list[str]:
    """Tickers for resume: source-job diagnostics first, evidence metadata fallback."""
    try:
        if found := _scoped_from_jobs(store, session_id, wave):
            return found
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass
    return _scoped_from_evidence(store, session_id)


def _resume_trace_id(session_id: str, sess: ResearchSession) -> str | None:
    """Trace id for resume: session budget first, newest session trace fallback."""
    trace_id: object = sess.budget.get("trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        existing = list_traces(session_id)
        trace_id = existing[0].trace_id if existing else None
    return trace_id if isinstance(trace_id, str) and trace_id else None


def _attach_resumed_trace(run: _LiveRun, session_id: str, wave: int, trace_id: str) -> None:
    """Reattach an existing trace with its event count as the sequence base."""
    import time as _time
    from pathlib import Path as _Path

    from app.config import get_data_root as _gdr

    try:
        events = get_trace_events(trace_id)
        seq = len(events)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        seq = 0
    try:
        root = _gdr()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        root = _Path("data")
    from app.research.evals.traces import TRACE_DB_NAME as _tdb

    try:
        db_path = root / _tdb
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        db_path = _Path("data") / _tdb
    run.trace = TraceRecorder(
        trace_id=trace_id,
        session_id=session_id,
        wave_id=wave,
        db_path=db_path,
        jsonl_path=root / "traces" / f"{trace_id}.jsonl",
        _seq=seq,
        _start_perf=_time.perf_counter(),
        _start_iso="",
        _closed=False,
    )
    run.trace_id = trace_id
    run._trace_record("trace.resumed", {"trace_id": trace_id, "session_id": session_id, "wave_id": wave})


def _attach_fresh_trace(run: _LiveRun, session_id: str, wave: int) -> None:
    """Fresh trace for resume when no prior trace resolves."""
    import subprocess as _sp

    try:
        _sha = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=2).strip() or "unknown"
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _sha = "unknown"
    tr = create_trace(
        session_id=session_id,
        wave_id=wave,
        provider=run.provider,
        model=run.model_name,
        prompt_version="v1",
        git_sha=_sha,
    )
    run.trace, run.trace_id = tr, tr.trace_id
    run._trace_record("trace.opened", {"trace_id": tr.trace_id, "session_id": session_id, "resume": True})


def _resume_trace(run: _LiveRun, session_id: str, wave: int, sess: ResearchSession) -> None:
    """Reattach the eval trace on resume: existing trace resumed, else a fresh one."""
    trace_id = _resume_trace_id(session_id, sess)
    if trace_id is not None:
        _attach_resumed_trace(run, session_id, wave, trace_id)
    else:
        _attach_fresh_trace(run, session_id, wave)


def _append_resume_evidence(run: _LiveRun, rec_dict: Mapping[str, object]) -> None:
    """Append one persisted evidence record; corrupt and duplicate rows skipped."""
    try:
        ev = evidence_from_dict(rec_dict)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return
    if ev.evidence_id in run.ledger:
        return
    try:
        run.ledger.append(ev)
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass


def _preload_resume_ledger(run: _LiveRun, store: ResearchRepository, session_id: str) -> None:
    """Seed ledger + dossier ids from persisted state; duplicates skipped."""
    for rec_dict in store.list_evidence(session_id):
        _append_resume_evidence(run, rec_dict)
    run.dossier_ids = []
    for d in store.list_dossiers(session_id):
        _did = d.get("dossier_id")
        if isinstance(_did, str) and _did:
            run.dossier_ids.append(_did)


def _completed_fetch_ids(
    run: _LiveRun, store: ResearchRepository, session_id: str, wave: int, eids: list[str]
) -> tuple[list[str], str] | None:
    """Completed-source fast path: dossier + completed source mean fetch is done."""
    dossier_id = f"{session_id}:{wave}:sec"
    all_jobs = store.list_jobs(session_id)
    src_jobs = [j for j in all_jobs if j.job_type == JobType.SOURCE_AGENT.value and j.wave_id == wave]
    if any(j.status == "completed" for j in src_jobs) and dossier_id in set(run.dossier_ids):
        return eids, _first_dossier_id(run)
    return None


def _reuse_fetch_question(reuse_job: Job, question: str) -> str:
    """Fetch question for a reused source job: its diagnostics question wins."""
    try:
        diag_q: object = dict(reuse_job.diagnostics).get("question")
        if isinstance(diag_q, str) and diag_q.strip():
            return diag_q
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass
    return question


def _restart_queued_fetch(run: _LiveRun, store: ResearchRepository, session_id: str, reuse_job: Job) -> None:
    """Start a queued source job and journal the reuse."""
    if reuse_job.status == "queued":
        store.save_job(_jobs.start_job(reuse_job))
        run._emit(session_id, "job.started", {"job_id": reuse_job.job_id})
    run._trace_record("source.reused", {"job_id": reuse_job.job_id, "wave_id": reuse_job.wave_id})


def _reuse_fetch_job(
    run: _LiveRun, store: ResearchRepository, session_id: str, wave: int, question: str
) -> tuple[str, str] | None:
    """Running-source path: restart a queued job, reuse its question, return (src_id, fetch_q)."""
    src_jobs = [
        j for j in store.list_jobs(session_id) if j.job_type == JobType.SOURCE_AGENT.value and j.wave_id == wave
    ]
    reuse_job = next((j for j in src_jobs if j.status in ("queued", "running")), None)
    if reuse_job is None:
        return None
    _restart_queued_fetch(run, store, session_id, reuse_job)
    return reuse_job.job_id, _reuse_fetch_question(reuse_job, question)


def _targeted_question(store: ResearchRepository, session_id: str) -> str | None:
    """Latest targeted question from the wave.started journal; None when absent."""
    latest: str | None = None
    for evt in store.list_events(session_id):
        payload = evt.to_dict().get("payload")
        if isinstance(payload, dict) and evt.event_type == "wave.started":
            tq = payload.get("targeted_question")
            if isinstance(tq, str) and tq.strip():
                latest = tq
    return latest


def _fresh_fetch_question(store: ResearchRepository, session_id: str, question: str) -> str:
    """Fresh-source path: reuse the authorized targeted question, else the original question."""
    try:
        return _targeted_question(store, session_id) or question
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return question


def _fetch_or_reuse_wave(
    run: _LiveRun, store: ResearchRepository, session_id: str, wave: int, question: str, eids: list[str]
) -> tuple[list[str], str]:
    """Fetch-or-reuse phase: skip fetch when the dossier + completed source exist."""
    done = _completed_fetch_ids(run, store, session_id, wave, eids)
    if done is not None:
        return done
    reused = _reuse_fetch_job(run, store, session_id, wave, question)
    if reused is not None:
        src_id, fetch_q = reused
    else:
        fetch_q = _fresh_fetch_question(store, session_id, question)
        src_id = run._open_source_job(session_id, wave, fetch_q)
    return run._fetch_wave(session_id, wave, fetch_q, src_id, ""), _first_dossier_id(run)


def _first_dossier_id(run: _LiveRun) -> str:
    """First persisted dossier id of the run; empty when none was written."""
    return run.dossier_ids[0] if run.dossier_ids else ""


def _substantive_ids(frozen: Sequence[str], substantive: set[str]) -> list[str]:
    """Frozen ids that are substantive evidence; every id when the session holds none yet."""
    return [e for e in frozen if not substantive or e in substantive]


def _freeze_or_reuse(
    run: _LiveRun, store: ResearchRepository, session_id: str, wave: int, eids: list[str]
) -> tuple[str, list[str]]:
    """Freeze-or-reuse phase: existing freeze ids win, else create a fresh freeze.

    A resolved freeze id always wins - its analyses already happened - but only
    the ids its row actually carries are reused: a row without a usable id list
    leaves this wave's ids standing. Reuse is evidence-only (a pre-fix freeze
    could carry discovery ids), so the ids are intersected with the session's
    substantive rows.
    """
    fid: str = f"{session_id}:{wave}:freeze"
    try:
        frozen_ids = store.get_freeze(fid).get("evidence_ids")
    except KeyError:
        return run._freeze_wave(session_id, wave), eids
    if not isinstance(frozen_ids, list) or not all(isinstance(e, str) for e in frozen_ids):
        return fid, eids
    frozen: list[str] = [e for e in frozen_ids if isinstance(e, str)]  # string-only, per the guard above
    return fid, _substantive_ids(frozen, {e.evidence_id for e in _ledger_evidence(run.ledger, session_id)})


def _hydrate_resume_budget(run: _LiveRun, store: ResearchRepository, session_id: str, sess: ResearchSession) -> None:
    """Seed the run budget from persisted totals; evidence count fallback; never raises."""
    try:
        persisted: object = sess.budget.get("tool_calls_used")
        if isinstance(persisted, int) and persisted >= 0:
            run.budget.hydrate(persisted)
        else:
            run.budget.hydrate(len(store.list_evidence(session_id)))
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass


def _close_resumed_wave(
    run: _LiveRun, store: ResearchRepository, session_id: str, wave: int, eids: list[str]
) -> dict[str, object]:
    """Committee + wave loop for a resumed wave with evidence."""
    fid, eids_for_committee = _freeze_or_reuse(run, store, session_id, wave, eids)
    # Analyses are never persisted, so the trio always reruns on the freeze —
    # even after one-committee (stock-only) interruptions. Same freeze id, new jobs.
    stock, bull, bear = run._committee_wave(session_id, wave, "")
    disagreement = compute_disagreement(stock, bull, bear)
    result = Wave1Result(
        session_id=session_id,
        wave_id=wave,
        freeze_id=fid,
        evidence_ids=list(eids_for_committee),
        stock=stock,
        bull=bull,
        bear=bear,
        disagreement=disagreement,
        question=run.question,
    )
    did_out: str = _first_dossier_id(run)
    return run._finish_completed(result, did_out)


def _open_resume_run(
    store: ResearchRepository,
    sess: ResearchSession,
    session_id: str,
    dispatch: Callable[[str, dict[str, object]], dict[str, object]],
    model: Callable[[str], str],
    limits: DirectorBudgets,
    provider: str,
    model_name: str | None,
) -> tuple[_LiveRun, str, int]:
    """Build the resume run: question/scope/wave from persisted session. Returns (run, question, wave)."""
    question: str = sess.query
    objective: str = sess.objective
    as_of_obj = sess.as_of
    as_of_str: str = as_of_obj.isoformat() if as_of_obj is not None else ""
    as_of: str | None = as_of_str or None
    cur_wave = sess.current_wave
    wave: int = cur_wave if isinstance(cur_wave, int) and cur_wave >= 1 else 1
    scoped = _resolve_resume_scoped(store, session_id, wave)
    run = _LiveRun(
        store,
        question,
        objective,
        as_of,
        as_of_str,
        scoped,
        dispatch,
        model,
        limits,
        wave,
        actor="resume_live",
        provider=provider,
        model_name=model_name,
    )
    return run, question, wave


def _merge_session_dossiers(run: _LiveRun, sess: ResearchSession) -> None:
    """Merge session-level dossier ids into the run without duplicates."""
    for _did in sess.dossier_ids:
        if isinstance(_did, str) and _did and _did not in run.dossier_ids:
            run.dossier_ids.append(_did)


def resume_live(
    session_id: str,
    dispatch: Callable[[str, dict[str, object]], dict[str, object]],
    model: Callable[[str], str],
    repo: ResearchRepository | None = None,
    budgets: DirectorBudgets | None = None,
    *,
    provider: str = "fake",
    model_name: str | None = None,
) -> dict[str, object]:
    """Retired production entrypoint (deterministic test helper only).

    Resume in production is the staged ResearchDirector (resumeResearch) over
    the persisted kernel session, not this in-process loop. Must not gain new
    callers. Preloads the evidence ledger (skipping duplicates) and dossier
    ids, skips fetch/freeze when already done, reruns the committee trio,
    closes with the next-wave gate + synthesis tail. FAILED/COMPLETED/
    CANCELLED sessions raise instead of silently reopening.
    """
    store: ResearchRepository = repo if repo is not None else ResearchRepository()
    limits: DirectorBudgets = budgets if budgets is not None else DirectorBudgets()
    state = store.resume(session_id)
    sess = state.session
    if sess.status in ("failed", "completed", "cancelled"):
        raise LiveModelError(
            session_id,
            "resume",
            f"resume: session {session_id} is {sess.status}"
            " (retry of failed waves is out of scope; start a new session)",
        )
    run, question, wave = _open_resume_run(store, sess, session_id, dispatch, model, limits, provider, model_name)
    try:
        _resume_trace(run, session_id, wave, sess)
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass
    _hydrate_resume_budget(run, store, session_id, sess)
    _preload_resume_ledger(run, store, session_id)
    _merge_session_dossiers(run, sess)
    eids: list[str] = [e.evidence_id for e in _ledger_evidence(run.ledger, session_id, wave)]
    eids, _ = _fetch_or_reuse_wave(run, store, session_id, wave, question, eids)
    if not eids:
        run._record_stop(session_id, "complete:empty-with-limitations")
        run._store_empty_terminal(session_id)
        bundle_dir = run.write_bundle(session_id)
        empty_out = _empty_terminal_result(
            session_id, wave, eids, _first_dossier_id(run), "complete:empty-with-limitations"
        )
        empty_out["bundle_dir"] = str(bundle_dir)
        return empty_out
    return _close_resumed_wave(run, store, session_id, wave, eids)
