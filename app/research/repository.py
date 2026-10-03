"""SQLite persistence for the research kernel: sessions/jobs/journal/evidence/freezes.

Database is data/research.sqlite ($RESEARCH_DB_PATH wins). Resume loads
state without writing anything, so completed jobs are never duplicated.
stdlib sqlite3 + json + hashlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_data_root
from . import journal as journal_log
from .models import (
    SOURCE_RUNTIME_BUDGET_S,
    DecisionRecord,
    Job,
    JournalEvent,
    JSONValue,
    ResearchNode,
    ResearchSession,
    utcnow,
    validate_json_mapping,
    validate_json_value,
)

__all__ = [
    "ResearchRepository",
    "ResumeState",
    "get_research_db_path",
    "pending_next_action",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  query TEXT NOT NULL, objective TEXT NOT NULL, as_of TEXT,
  status TEXT NOT NULL, current_wave INTEGER NOT NULL,
  policy TEXT NOT NULL, budget TEXT NOT NULL,
  source_policy TEXT DEFAULT '{"allowed":["SEC"],"denied":[],"mode":"allowlist"}',
  temporal_scope TEXT DEFAULT '{"as_of":null,"end":null,"mode":"latest-available","raw":null,"start":null}',
  job_ids TEXT NOT NULL, evidence_ids TEXT NOT NULL,
  freeze_ids TEXT NOT NULL, dossier_ids TEXT NOT NULL,
  committee_runs TEXT NOT NULL, unresolved_questions TEXT NOT NULL,
  targeted_question TEXT, targeted_domain TEXT,
  final_result TEXT, failure TEXT);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, wave_id INTEGER NOT NULL,
  parent_job_id TEXT, job_type TEXT NOT NULL, owner TEXT NOT NULL,
  source_domain TEXT, status TEXT NOT NULL,
  created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, deadline TEXT,
  last_heartbeat_at TEXT,
  model TEXT, token_budget INTEGER, tool_budget INTEGER, child_budget INTEGER NOT NULL,
  result TEXT, diagnostics TEXT NOT NULL, failure TEXT);
CREATE TABLE IF NOT EXISTS journal (
  event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL, timestamp TEXT NOT NULL,
  actor_type TEXT NOT NULL, actor_id TEXT NOT NULL, payload TEXT NOT NULL,
  previous_state TEXT, new_state TEXT, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS ux_journal_session_seq ON journal(session_id, sequence);
CREATE INDEX IF NOT EXISTS ix_jobs_session ON jobs(session_id);
CREATE INDEX IF NOT EXISTS ix_journal_session ON journal(session_id, sequence);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  known_at TEXT, as_of TEXT, identity_key TEXT, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_evidence_session ON evidence(session_id);
CREATE TABLE IF NOT EXISTS freezes (
  freeze_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  created_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_freezes_session ON freezes(session_id);
CREATE TABLE IF NOT EXISTS dossiers (
  dossier_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  created_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_dossiers_session ON dossiers(session_id);
CREATE TABLE IF NOT EXISTS coverage_artifacts (
  artifact_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  created_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_coverage_artifacts_session ON coverage_artifacts(session_id);
CREATE TABLE IF NOT EXISTS tool_results (
  tool_result_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, job_id TEXT NOT NULL,
  tool_name TEXT NOT NULL, created_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_tool_results_session ON tool_results(session_id);
CREATE TABLE IF NOT EXISTS research_nodes (
  node_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  created_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_nodes_session ON research_nodes(session_id);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
  node_id TEXT, job_id TEXT, created_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS ix_decisions_node ON decisions(node_id);
CREATE TABLE IF NOT EXISTS submissions (
  session_id TEXT NOT NULL, request_id TEXT NOT NULL, job_id TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY (session_id, request_id));
CREATE INDEX IF NOT EXISTS ix_submissions_job ON submissions(job_id);
"""


def get_research_db_path(data_root: Path | None = None) -> Path:
    """Resolve the research DB path: $RESEARCH_DB_PATH wins, else data_root/research.sqlite."""
    env = (os.environ.get("RESEARCH_DB_PATH") or "").strip()
    if env:
        return Path(env)
    root = data_root if data_root is not None else get_data_root()
    return root / "research.sqlite"


def _jsonable(value: object) -> object:
    """Convert datetimes to ISO strings so records stay JSON-serializable."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _record_json(record: Mapping[str, object], where: str) -> str:
    validated = validate_json_value(_jsonable(dict(record)), where)
    return json.dumps(validated, sort_keys=True)


def _iso_or_none(value: object, key: str, where: str) -> str | None:
    raw: str | None = None
    if value is None:
        return None
    if isinstance(value, datetime):
        raw = value.isoformat()
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
    else:
        raise ValueError(f"{where}: '{key}' must be ISO-8601, datetime, or null")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"{where}: '{key}' must be ISO-8601, got {value!r}") from None
    aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return aware.isoformat()


def _chain_hash(prev_hash: str, event: JournalEvent) -> str:
    body = "|".join(
        (
            prev_hash,
            event.event_id,
            str(event.sequence),
            event.timestamp.isoformat(),
            json.dumps(event.to_dict()["payload"], sort_keys=True),
        )
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


_DEFAULT_SOURCE_POLICY_JSON = '{"allowed":["SEC"],"denied":[],"mode":"allowlist"}'
_DEFAULT_TEMPORAL_SCOPE_JSON = '{"as_of":null,"end":null,"mode":"latest-available","raw":null,"start":null}'


def _session_policy_doc(names: set[str], row: sqlite3.Row) -> object:
    """Stored source_policy JSON; pre-policy rows fall back to the SEC default."""
    if "source_policy" not in names:
        text = _DEFAULT_SOURCE_POLICY_JSON
    else:
        raw = row["source_policy"]
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            text = _DEFAULT_SOURCE_POLICY_JSON
        else:
            text = str(raw)
    parsed: object = json.loads(text)
    return parsed


def _temporal_stored(names: set[str], row: sqlite3.Row) -> object:
    """Stored temporal_scope JSON; None when the column is absent or blank."""
    if "temporal_scope" not in names:
        return None
    raw = row["temporal_scope"]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    stored: object = json.loads(str(raw))
    return stored


def _temporal_defaults(dts: object) -> dict[str, object]:
    """Default scope envelope; malformed factories fall back to latest-available."""
    defaults = (
        dts() if callable(dts) else {"as_of": None, "start": None, "end": None, "mode": "latest-available", "raw": None}
    )
    if isinstance(defaults, dict):
        return dict(defaults)
    return {"as_of": None, "start": None, "end": None, "mode": "latest-available", "raw": None}


def _temporal_cutoff(as_of: object) -> tuple[object, str]:
    """Session as_of cutoff + mode; blank as_of means latest-available."""
    if isinstance(as_of, str) and as_of.strip():
        return as_of, "as_of"
    return None, "latest-available"


def _session_temporal_doc(names: set[str], row: sqlite3.Row, as_of: object, dts: object) -> object:
    """Stored temporal_scope JSON; pre-policy rows inherit the session as_of cutoff."""
    stored = _temporal_stored(names, row)
    if stored is not None:
        return stored
    cut, mode = _temporal_cutoff(as_of)
    return {**_temporal_defaults(dts), "as_of": cut, "mode": mode}


def _default_max_tool_calls() -> int | None:
    """Global dispatch ceiling (director default; None means unbounded)."""
    try:
        from .director import DirectorBudgets

        return DirectorBudgets().max_tool_calls
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _session_max_calls(session: ResearchSession, default_max: int | None) -> int | None:
    """Session max dispatches: budget total_tool_budget wins, else policy research.max_tool_calls; None unbounded."""
    raw_total: object = session.budget.get("total_tool_budget", None)
    if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total >= 0:
        return raw_total
    if raw_total is None:
        raw_section: object = session.policy.get("research", {})
        section: dict[str, object] = raw_section if isinstance(raw_section, dict) else {}
        raw_max: object = section.get("max_tool_calls", default_max)
        if raw_max is None:
            return None
        return raw_max if isinstance(raw_max, int) and not isinstance(raw_max, bool) else default_max
    return default_max


def _session_used_calls(session: ResearchSession) -> int:
    """Consumed dispatch slots (budget tool_calls_used; 0 on bad shape)."""
    raw_used: object = session.budget.get("tool_calls_used", 0)
    return raw_used if isinstance(raw_used, int) and not isinstance(raw_used, bool) and raw_used >= 0 else 0


def _check_dispatch_budgets(job: Job, session_id: str, job_id: str, max_calls: int | None, used: int) -> None:
    """Raise on exhausted per-job or per-session dispatch budget (None session cap allows)."""
    if job.tool_budget is not None and job.tool_budget <= 0:
        raise ValueError(f"dispatch: job {job_id!r} tool_budget exhausted")
    if max_calls is not None and used >= max_calls:
        raise ValueError(f"dispatch: session {session_id!r} tool budget exhausted ({used}/{max_calls})")


def _job_id(job: Job) -> str:
    return job.job_id


def _job_action(job: Job) -> JSONValue:
    """EXECUTE_JOB payload for one runnable job."""
    return {
        "verb": "EXECUTE_JOB",
        "job_id": job.job_id,
        "job_type": job.job_type,
        "source_domain": job.source_domain or "sec",
        "deadline": job.deadline.isoformat() if job.deadline is not None else None,
        "budget_s": SOURCE_RUNTIME_BUDGET_S,
    }


def _status_action(status: str) -> JSONValue:
    """Next step when no runnable job exists."""
    actions: dict[str, JSONValue] = {
        "created": {"verb": "PLAN"},
        "planning": {"verb": "PLAN"},
        "researching": {"verb": "WAIT", "reason": "no runnable jobs"},
        "freezing": {"verb": "FREEZE"},
        "analyzing": {"verb": "ANALYZE"},
        "targeted_research": {"verb": "EXECUTE_JOB", "reason": "dispatch targeted research"},
        "synthesizing": {"verb": "SYNTHESIZE"},
    }
    fallback: dict[str, JSONValue] = {"verb": "NONE", "reason": f"unknown status {status!r}"}
    return actions.get(status, fallback)


def pending_next_action(session: ResearchSession, jobs: list[Job]) -> JSONValue:
    """Deterministic next step from status + open jobs (read-only).

    Returns an EXECUTE_* verb naming the runnable owned job; bare `wait` only
    when no runnable work exists. Dict shape survives JSON round-trips.
    """
    mine = sorted((j for j in jobs if j.session_id == session.session_id), key=_job_id)
    running = [j for j in mine if j.status == "running"]
    queued = [j for j in mine if j.status == "queued"]
    if session.status in ("completed", "failed", "cancelled"):
        return {"verb": "NONE", "reason": f"session {session.status}"}
    if running:
        return _job_action(running[0])
    if queued:
        return _job_action(queued[0])
    return _status_action(session.status)


_SESSION_SQL = (
    "INSERT OR REPLACE INTO sessions (session_id, created_at, updated_at, query, objective,"
    " as_of, status, current_wave, policy, budget, source_policy, temporal_scope,"
    " job_ids, evidence_ids, freeze_ids,"
    " dossier_ids, committee_runs, unresolved_questions, targeted_question, targeted_domain,"
    " final_result, failure)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_JOB_SQL = (
    "INSERT OR REPLACE INTO jobs (job_id, session_id, wave_id, parent_job_id, job_type, owner,"
    " source_domain, status, created_at, started_at, completed_at, deadline, last_heartbeat_at, model,"
    " token_budget, tool_budget, child_budget, result, diagnostics, failure)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _session_params(session: ResearchSession) -> tuple[object, ...]:
    """Positional params for _SESSION_SQL (caller validates first)."""
    policy = json.dumps(validate_json_mapping(session.policy, "<session>"), sort_keys=True)
    budget = json.dumps(validate_json_mapping(session.budget, "<session>"), sort_keys=True)
    from .models import validate_source_policy as _vsp
    from .models import validate_temporal_scope as _vts

    source_policy = json.dumps(_vsp(session.source_policy, "<session>"), sort_keys=True)
    temporal_scope = json.dumps(_vts(session.temporal_scope, "<session>"), sort_keys=True)
    doc = session.to_dict()
    return (
        session.session_id,
        session.created_at.isoformat(),
        session.updated_at.isoformat(),
        session.query,
        session.objective,
        session.as_of.isoformat() if session.as_of is not None else None,
        session.status,
        session.current_wave,
        policy,
        budget,
        source_policy,
        temporal_scope,
        json.dumps(doc["job_ids"], sort_keys=True),
        json.dumps(doc["evidence_ids"], sort_keys=True),
        json.dumps(doc["freeze_ids"], sort_keys=True),
        json.dumps(doc["dossier_ids"], sort_keys=True),
        json.dumps(doc["committee_runs"], sort_keys=True),
        json.dumps(doc["unresolved_questions"], sort_keys=True),
        session.targeted_question,
        session.targeted_domain,
        json.dumps(doc["final_result"], sort_keys=True) if session.final_result is not None else None,
        json.dumps(doc["failure"], sort_keys=True) if session.failure is not None else None,
    )


def _job_params(job: Job) -> tuple[object, ...]:
    """Positional params for _JOB_SQL (caller validates first)."""
    doc = job.to_dict()
    return (
        job.job_id,
        job.session_id,
        job.wave_id,
        job.parent_job_id,
        job.job_type,
        job.owner,
        job.source_domain,
        job.status,
        job.created_at.isoformat(),
        job.started_at.isoformat() if job.started_at is not None else None,
        job.completed_at.isoformat() if job.completed_at is not None else None,
        job.deadline.isoformat() if job.deadline is not None else None,
        job.last_heartbeat_at.isoformat() if job.last_heartbeat_at is not None else None,
        job.model,
        job.token_budget,
        job.tool_budget,
        job.child_budget,
        json.dumps(doc["result"], sort_keys=True) if job.result is not None else None,
        json.dumps(doc["diagnostics"], sort_keys=True),
        json.dumps(doc["failure"], sort_keys=True) if job.failure is not None else None,
    )


@dataclass(frozen=True)
class ResumeState:
    """Resume snapshot: session + wave + budgets + next action. No writes performed."""

    session: ResearchSession
    wave: int
    budgets: dict[str, JSONValue] = field(default_factory=dict)
    pending_next_action: JSONValue = ""
    open_job_ids: list[str] = field(default_factory=list)


def _migrate_evidence_identity(conn: sqlite3.Connection) -> None:
    """Lazy identity_key column + unique index + json_extract backfill."""
    for stmt in (
        "ALTER TABLE evidence ADD COLUMN identity_key TEXT",
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_evidence_session_identity ON evidence(session_id, identity_key)",
        "UPDATE evidence SET identity_key = json_extract(record, '$.metadata.identity_key') WHERE identity_key IS NULL AND json_extract(record, '$.metadata.identity_key') IS NOT NULL",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.Error:
            pass


def _evidence_identity_key(record: Mapping[str, object]) -> str | None:
    """Identity key from the stored metadata; None when absent or blank."""
    meta = record.get("metadata")
    raw_key = meta.get("identity_key") if isinstance(meta, dict) else None
    return raw_key if isinstance(raw_key, str) and raw_key else None


def _duplicate_evidence_error(
    conn: sqlite3.Connection, where: str, evidence_id: str, identity_key: str | None
) -> ValueError:
    """Evidence-id conflict wins; otherwise the identity conflict."""
    if conn.execute("SELECT 1 FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone():
        return ValueError(f"{where}: duplicate evidence_id {evidence_id!r}")
    return ValueError(f"{where}: duplicate identity_key {identity_key!r}")


class ResearchRepository:
    """Per-operation SQLite connections; no shared state.

    ponytail: one connection per call, add pooling if write volume matters.
    """

    def __init__(self, path: Path | str | None = None, *, data_root: Path | None = None) -> None:
        """Bind to an explicit path, or resolve via get_research_db_path."""
        self._path = Path(path) if path is not None else get_research_db_path(data_root)

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        # ponytail: lazy ALTER for pre-targeted DBs; drop once all DBs migrate.
        for col in ("targeted_question", "targeted_domain"):
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT")
            except sqlite3.Error:
                pass
        for col, default in (
            ("source_policy", '\'{"allowed":["SEC"],"denied":[],"mode":"allowlist"}\''),
            (
                "temporal_scope",
                '\'{"as_of":null,"end":null,"mode":"latest-available","raw":null,"start":null}\'',
            ),
        ):
            try:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.Error:
                pass
        try:
            conn.execute(
                "UPDATE sessions SET source_policy = ? WHERE source_policy IS NULL OR TRIM(source_policy) = ''",
                (_DEFAULT_SOURCE_POLICY_JSON,),
            )
        except sqlite3.Error:
            pass
        try:
            conn.execute(
                "UPDATE sessions SET temporal_scope = ? WHERE temporal_scope IS NULL OR TRIM(temporal_scope) = ''",
                (_DEFAULT_TEMPORAL_SCOPE_JSON,),
            )
        except sqlite3.Error:
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN last_heartbeat_at TEXT")
        except sqlite3.Error:
            pass
        _migrate_evidence_identity(conn)
        return conn

    # -- sessions ------------------------------------------------------

    def save_session(self, session: ResearchSession) -> None:
        """Upsert one session row."""
        session.validate("<research.sqlite>")
        with self._connect() as conn:
            conn.execute(_SESSION_SQL, _session_params(session))

    def get_session(self, session_id: str) -> ResearchSession:
        """Load one session; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown session_id: {session_id!r}")
        return self._row_to_session(row)

    def list_sessions(self, limit: int = 20) -> list[dict[str, object]]:
        """Newest-first session summaries (read-only; empty when no DB yet)."""
        if not self._path.exists():
            return []
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT session_id, status, updated_at, query FROM sessions ORDER BY updated_at DESC LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
            except sqlite3.Error:
                return []
        return [
            {
                "session_id": str(r["session_id"]),
                "status": str(r["status"]),
                "updated_at": str(r["updated_at"]),
                "query": str(r["query"])[:80],
            }
            for r in rows
        ]

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> ResearchSession:
        from .models import default_temporal_scope as _dts

        doc: dict[str, object] = {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "query": row["query"],
            "objective": row["objective"],
            "as_of": row["as_of"],
            "status": row["status"],
            "current_wave": row["current_wave"],
        }
        for key in (
            "policy",
            "budget",
            "job_ids",
            "evidence_ids",
            "freeze_ids",
            "dossier_ids",
            "committee_runs",
            "unresolved_questions",
        ):
            raw: object = json.loads(str(row[key]))
            doc[key] = raw
        for key in ("final_result", "failure"):
            value = row[key]
            raw_opt: object = json.loads(str(value)) if value is not None else None
            doc[key] = raw_opt
        # ponytail: pre-targeted rows lack the columns; default None (from_dict agrees).
        names = set(row.keys())
        doc["targeted_question"] = row["targeted_question"] if "targeted_question" in names else None
        doc["targeted_domain"] = row["targeted_domain"] if "targeted_domain" in names else None
        # ponytail: pre-policy rows carry NULL/blank columns; SEC default + session.as_of win.
        doc["source_policy"] = _session_policy_doc(names, row)
        doc["temporal_scope"] = _session_temporal_doc(names, row, doc["as_of"], _dts)
        return ResearchSession.from_dict(doc, "<research.sqlite>")

    # -- jobs ----------------------------------------------------------

    def save_job(self, job: Job) -> None:
        """Upsert one job row."""
        job.validate("<research.sqlite>")
        with self._connect() as conn:
            conn.execute(_JOB_SQL, _job_params(job))

    def save_session_and_job(self, session: ResearchSession, job: Job) -> None:
        """Upsert one session + one job atomically under BEGIN IMMEDIATE (all-or-nothing)."""
        session.validate("<research.sqlite>")
        job.validate("<research.sqlite>")
        conn = self._connect()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(_SESSION_SQL, _session_params(session))
                conn.execute(_JOB_SQL, _job_params(job))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def commit_job_with_request(self, session: ResearchSession, job: Job, request_id: str) -> str:
        """Claim one (session, request_id) slot and persist session + job atomically."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("<research.sqlite>: 'request_id' must be a non-empty string")
        session.validate("<research.sqlite>")
        job.validate("<research.sqlite>")
        conn = self._connect()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT job_id FROM submissions WHERE session_id = ? AND request_id = ?",
                    (session.session_id, request_id),
                ).fetchone()
                if row is not None:
                    winner = str(row["job_id"])
                    conn.rollback()
                    self._check_submission_type(winner, job.job_type, request_id)
                    return winner
                conn.execute(
                    "INSERT INTO submissions (session_id, request_id, job_id, created_at) VALUES (?, ?, ?, ?)",
                    (session.session_id, request_id, job.job_id, utcnow().isoformat()),
                )
                conn.execute(_SESSION_SQL, _session_params(session))
                conn.execute(_JOB_SQL, _job_params(job))
                conn.commit()
                return job.job_id
            except sqlite3.IntegrityError:
                conn.rollback()
                winner_row = self.submission_job_id(session.session_id, request_id)
                if winner_row is None:
                    raise
                self._check_submission_type(winner_row, job.job_type, request_id)
                return winner_row
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def _check_submission_type(self, winner_job_id: str, job_type: str, request_id: str) -> None:
        """Raise when a request id already identifies a different job type."""
        try:
            winner = self.get_job(winner_job_id)
        except KeyError:
            return
        if winner.job_type != job_type:
            raise ValueError(f"<research.sqlite>: request {request_id!r} already identifies {winner.job_type!r}")

    def submission_job_id(self, session_id: str, request_id: str) -> str | None:
        """Winner job id for one (session, request_id) slot; None when unclaimed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT job_id FROM submissions WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        return str(row["job_id"]) if row is not None else None

    def save_evidence_and_session(self, session: ResearchSession, record: Mapping[str, object]) -> str:
        """Insert one evidence row + link the session atomically under BEGIN IMMEDIATE.

        Crash between insert and link used to orphan the row: a retry hit the
        duplicate-identity path and returned without linking. One batch keeps
        the row and the session.evidence_ids index together. Duplicate contract
        matches save_evidence.
        """
        if not isinstance(record, Mapping):
            raise ValueError("<research.sqlite>: evidence record must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = "<research.sqlite>: evidence"
        evidence_id = record.get("evidence_id")
        session_id = record.get("session_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"{where}: 'evidence_id' must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        session.validate("<research.sqlite>")
        identity_key = _evidence_identity_key(record)
        created_known = _iso_or_none(record.get("known_at"), "known_at", where)
        created_asof = _iso_or_none(record.get("as_of"), "as_of", where)
        payload = _record_json(record, where)
        conn = self._connect()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO evidence (evidence_id, session_id, known_at, as_of, identity_key, record) VALUES (?, ?, ?, ?, ?, ?)",
                    (evidence_id, session_id, created_known, created_asof, identity_key, payload),
                )
                conn.execute(_SESSION_SQL, _session_params(session))
                conn.commit()
            except sqlite3.IntegrityError:
                err = _duplicate_evidence_error(conn, where, str(evidence_id), identity_key)
                conn.rollback()
                raise err from None
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()
        return evidence_id

    def consume_dispatch_budget(self, session_id: str, job_id: str) -> tuple[ResearchSession, Job]:
        """Atomically consume one job + global dispatch slot under BEGIN IMMEDIATE."""
        from dataclasses import replace

        from .models import Failure, normalize_time

        conn = self._connect()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            try:
                srow = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
                if srow is None:
                    raise KeyError(f"unknown session_id: {session_id!r}")
                jrow = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if jrow is None:
                    raise KeyError(f"unknown job_id: {job_id!r}")
                session = self._row_to_session(srow)
                job = self._row_to_job(jrow)
                if job.session_id != session.session_id:
                    raise ValueError(f"dispatch: job {job_id!r} belongs to {job.session_id!r}")
                if job.status != "running":
                    raise ValueError(f"dispatch: job {job_id!r} status is {job.status!r} (running required)")
                if job.deadline is not None:
                    deadline = normalize_time(job.deadline)
                    if utcnow() > deadline:
                        timed = replace(
                            job,
                            status="timed_out",
                            completed_at=utcnow(),
                            failure=Failure(
                                category="timeout",
                                message=f"deadline {job.deadline.isoformat()} expired",
                            ),
                        )
                        timed.validate("<research.sqlite>")
                        conn.execute(_JOB_SQL, _job_params(timed))
                        conn.commit()
                        raise ValueError(f"dispatch: job {job_id!r} deadline expired")
                default_max = _default_max_tool_calls()
                max_calls = _session_max_calls(session, default_max)
                used = _session_used_calls(session)
                _check_dispatch_budgets(job, session_id, job_id, max_calls, used)
                spent = replace(
                    job,
                    tool_budget=job.tool_budget - 1 if job.tool_budget is not None else None,
                )
                billed = replace(
                    session,
                    budget={**session.budget, "tool_calls_used": used + 1},
                    updated_at=utcnow(),
                )
                billed.validate("<research.sqlite>")
                spent.validate("<research.sqlite>")
                conn.execute(_SESSION_SQL, _session_params(billed))
                conn.execute(_JOB_SQL, _job_params(spent))
                conn.commit()
                return billed, spent
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Job:
        """Load one job; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job_id: {job_id!r}")
        return self._row_to_job(row)

    def list_jobs(self, session_id: str) -> list[Job]:
        """All jobs for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        doc: dict[str, object] = {
            "job_id": row["job_id"],
            "session_id": row["session_id"],
            "wave_id": row["wave_id"],
            "parent_job_id": row["parent_job_id"],
            "job_type": row["job_type"],
            "owner": row["owner"],
            "source_domain": row["source_domain"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "deadline": row["deadline"],
            "last_heartbeat_at": dict(row).get("last_heartbeat_at"),
            "model": row["model"],
            "token_budget": row["token_budget"],
            "tool_budget": row["tool_budget"],
            "child_budget": row["child_budget"],
        }
        for key in ("result", "diagnostics", "failure"):
            value = row[key]
            raw_opt: object = json.loads(str(value)) if value is not None else None
            doc[key] = raw_opt if raw_opt is not None else ({} if key == "diagnostics" else None)
        return Job.from_dict(doc, "<research.sqlite>")

    # -- journal --------------------------------------------------------

    def save_event(self, event: JournalEvent) -> JournalEvent:
        """Append one journal row with a hash chain; duplicates raise ValueError."""
        event.validate("<research.sqlite>")
        with self._connect() as conn:
            last = conn.execute(
                "SELECT hash FROM journal WHERE session_id = ? ORDER BY sequence DESC LIMIT 1",
                (event.session_id,),
            ).fetchone()
            prev = str(last["hash"]) if last is not None else "GENESIS"
            digest = _chain_hash(prev, event)
            try:
                conn.execute(
                    "INSERT INTO journal (event_id, session_id, sequence, event_type, timestamp,"
                    " actor_type, actor_id, payload, previous_state, new_state, prev_hash, hash)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.session_id,
                        event.sequence,
                        event.event_type,
                        event.timestamp.isoformat(),
                        event.actor_type,
                        event.actor_id,
                        json.dumps(event.to_dict()["payload"], sort_keys=True),
                        event.previous_state,
                        event.new_state,
                        prev,
                        digest,
                    ),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"<research.sqlite>: duplicate journal event {event.event_id!r}") from None
        return event

    def list_events(self, session_id: str) -> list[JournalEvent]:
        """Session events in sequence order; raises ValueError on a broken hash chain."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM journal WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        events: list[JournalEvent] = []
        prev = "GENESIS"
        for row in rows:
            payload_raw: object = json.loads(str(row["payload"]))
            if not isinstance(payload_raw, dict):
                raise ValueError(f"<research.sqlite>: event {row['event_id']!r} payload must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
            event = JournalEvent.from_dict(
                {
                    "event_id": row["event_id"],
                    "session_id": row["session_id"],
                    "sequence": row["sequence"],
                    "event_type": row["event_type"],
                    "timestamp": row["timestamp"],
                    "actor_type": row["actor_type"],
                    "actor_id": row["actor_id"],
                    "payload": payload_raw,
                    "previous_state": row["previous_state"],
                    "new_state": row["new_state"],
                },
                "<research.sqlite>",
            )
            if str(row["prev_hash"]) != prev or str(row["hash"]) != _chain_hash(prev, event):
                raise ValueError(f"<research.sqlite>: journal chain broken at {event.event_id!r}")
            prev = str(row["hash"])
            events.append(event)
        return events

    # -- evidence / freezes (records owned by evidence.py / freeze.py) --

    def save_evidence(self, record: Mapping[str, object]) -> str:
        """Insert one evidence record; duplicate ids raise ValueError."""
        if not isinstance(record, Mapping):
            raise ValueError("<research.sqlite>: evidence record must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = "<research.sqlite>: evidence"
        evidence_id = record.get("evidence_id")
        session_id = record.get("session_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"{where}: 'evidence_id' must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        identity_key = _evidence_identity_key(record)
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO evidence (evidence_id, session_id, known_at, as_of, identity_key, record) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        evidence_id,
                        session_id,
                        _iso_or_none(record.get("known_at"), "known_at", where),
                        _iso_or_none(record.get("as_of"), "as_of", where),
                        identity_key,
                        _record_json(record, where),
                    ),
                )
            except sqlite3.IntegrityError:
                raise _duplicate_evidence_error(conn, where, str(evidence_id), identity_key) from None
        return evidence_id

    def get_evidence(self, evidence_id: str) -> dict[str, JSONValue]:
        """Load one evidence record; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT record FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown evidence_id: {evidence_id!r}")
        return validate_json_mapping(json.loads(str(row["record"])), "<research.sqlite>: evidence")

    def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
        """All evidence records for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM evidence WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        return [validate_json_mapping(json.loads(str(r["record"])), "<research.sqlite>: evidence") for r in rows]

    def find_evidence_by_identity(self, session_id: str, identity_key: str) -> dict[str, JSONValue] | None:
        """One evidence record by identity key; None when blank or absent."""
        if not identity_key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record FROM evidence WHERE session_id = ? AND identity_key = ? LIMIT 1",
                (session_id, identity_key),
            ).fetchone()
        if row is None:
            return None
        return validate_json_mapping(json.loads(str(row["record"])), "<research.sqlite>: evidence")

    def list_evidence_ids(self, session_id: str) -> list[str]:
        """Evidence ids for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT evidence_id FROM evidence WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        return [str(r["evidence_id"]) for r in rows]

    def save_freeze(self, record: Mapping[str, object]) -> str:
        """Insert one freeze record; duplicate ids raise ValueError."""
        if not isinstance(record, Mapping):
            raise ValueError("<research.sqlite>: freeze record must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = "<research.sqlite>: freeze"
        freeze_id = record.get("freeze_id")
        session_id = record.get("session_id")
        if not isinstance(freeze_id, str) or not freeze_id:
            raise ValueError(f"{where}: 'freeze_id' must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        created = _iso_or_none(record.get("created_at"), "created_at", where) or utcnow().isoformat()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO freezes (freeze_id, session_id, created_at, record) VALUES (?, ?, ?, ?)",
                    (freeze_id, session_id, created, _record_json(record, where)),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"{where}: duplicate freeze_id {freeze_id!r}") from None
        return freeze_id

    def get_freeze(self, freeze_id: str) -> dict[str, JSONValue]:
        """Load one freeze record; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT record FROM freezes WHERE freeze_id = ?", (freeze_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown freeze_id: {freeze_id!r}")
        return validate_json_mapping(json.loads(str(row["record"])), "<research.sqlite>: freeze")

    def save_dossier(self, record: Mapping[str, object]) -> str:
        """Insert one immutable dossier record; duplicate ids raise ValueError."""
        if not isinstance(record, Mapping):
            raise ValueError("<research.sqlite>: dossier record must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = "<research.sqlite>: dossier"
        dossier_id = record.get("dossier_id")
        session_id = record.get("session_id")
        if not isinstance(dossier_id, str) or not dossier_id:
            raise ValueError(f"{where}: 'dossier_id' must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        created = _iso_or_none(record.get("created_at"), "created_at", where) or utcnow().isoformat()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO dossiers (dossier_id, session_id, created_at, record) VALUES (?, ?, ?, ?)",
                    (dossier_id, session_id, created, _record_json(record, where)),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"{where}: duplicate dossier_id {dossier_id!r}") from None
        return dossier_id

    def get_dossier(self, dossier_id: str) -> dict[str, JSONValue]:
        """Load one dossier record; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT record FROM dossiers WHERE dossier_id = ?", (dossier_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown dossier_id: {dossier_id!r}")
        return validate_json_mapping(json.loads(str(row["record"])), "<research.sqlite>: dossier")

    def list_dossiers(self, session_id: str) -> list[dict[str, JSONValue]]:
        """All dossier records for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM dossiers WHERE session_id = ? ORDER BY created_at, dossier_id",
                (session_id,),
            ).fetchall()
        return [validate_json_mapping(json.loads(str(r["record"])), "<research.sqlite>: dossier") for r in rows]

    def save_coverage_artifact(self, record: Mapping[str, object]) -> str:
        """Insert one immutable coverage artifact (search scope + absence); duplicate ids raise ValueError.

        Coverage artifacts are not evidence: they carry a SearchRun's scope, never
        a raw document, and no citation path resolves them (see resource_stores'
        ``coverage`` kind, which is inspection-only).
        """
        if not isinstance(record, Mapping):
            raise ValueError(  # noqa: TRY004 - the public error contract pins ValueError, tests are oracle
                "<research.sqlite>: coverage artifact record must be a mapping"
            )
        where = "<research.sqlite>: coverage artifact"
        artifact_id = record.get("artifact_id")
        session_id = record.get("session_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"{where}: 'artifact_id' must be a non-empty string")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{where}: 'session_id' must be a non-empty string")
        created = _iso_or_none(record.get("created_at"), "created_at", where) or utcnow().isoformat()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO coverage_artifacts (artifact_id, session_id, created_at, record) VALUES (?, ?, ?, ?)",
                    (artifact_id, session_id, created, _record_json(record, where)),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"{where}: duplicate artifact_id {artifact_id!r}") from None
        return artifact_id

    def list_coverage_artifacts(self, session_id: str) -> list[dict[str, JSONValue]]:
        """All coverage artifacts for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM coverage_artifacts WHERE session_id = ? ORDER BY created_at, artifact_id",
                (session_id,),
            ).fetchall()
        return [
            validate_json_mapping(json.loads(str(r["record"])), "<research.sqlite>: coverage artifact") for r in rows
        ]

    def save_tool_result(self, record: Mapping[str, object]) -> str:
        """Insert one immutable staged tool result; duplicate ids raise ValueError.

        A tool result is the kernel-persisted payload a finra_record, web_source,
        or sec_record ref replays against. Resume never rewrites one (same id =
        same bytes), so evidence admission can prove the row a citation names.
        """
        if not isinstance(record, Mapping):
            raise ValueError("<research.sqlite>: tool result record must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        where = "<research.sqlite>: tool result"
        tool_result_id = str(record.get("tool_result_id") or "")
        session_id = record.get("session_id")
        job_id = record.get("job_id")
        tool_name = record.get("tool_name")
        for key, value in (
            ("tool_result_id", tool_result_id),
            ("session_id", session_id),
            ("job_id", job_id),
            ("tool_name", tool_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{where}: {key!r} must be a non-empty string")
        created = _iso_or_none(record.get("created_at"), "created_at", where) or utcnow().isoformat()
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO tool_results (tool_result_id, session_id, job_id, tool_name, created_at, record) VALUES (?, ?, ?, ?, ?, ?)",
                    (tool_result_id, session_id, job_id, tool_name, created, _record_json(record, where)),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"{where}: duplicate tool_result_id {tool_result_id!r}") from None
        return tool_result_id

    def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
        """Load one tool result; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT record FROM tool_results WHERE tool_result_id = ?", (tool_result_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown tool_result_id: {tool_result_id!r}")
        return validate_json_mapping(json.loads(str(row["record"])), "<research.sqlite>: tool result")

    def list_tool_results(self, session_id: str) -> list[dict[str, JSONValue]]:
        """All tool results for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM tool_results WHERE session_id = ? ORDER BY created_at, tool_result_id",
                (session_id,),
            ).fetchall()
        return [validate_json_mapping(json.loads(str(r["record"])), "<research.sqlite>: tool result") for r in rows]

    def resource_stores(self, session_id: str) -> dict[str, dict[str, object]]:
        """id->record mappings for read_resource: evidence/freeze/dossier/coverage/job/research/tool_result."""
        session = self.get_session(session_id)
        evidence: dict[str, object] = {str(r.get("evidence_id", "")): r for r in self.list_evidence(session_id)}
        freezes: dict[str, object] = {}
        for fid in session.freeze_ids:
            try:
                freezes[fid] = self.get_freeze(fid)
            except KeyError:
                continue
        dossiers: dict[str, object] = {}
        for dossier in self.list_dossiers(session_id):
            key = dossier.get("dossier_id")
            if isinstance(key, str):
                dossiers[key] = dossier
        jobs: dict[str, object] = {j.job_id: j.to_dict() for j in self.list_jobs(session_id)}
        coverage: dict[str, object] = {}
        for artifact in self.list_coverage_artifacts(session_id):
            key = artifact.get("artifact_id")
            if isinstance(key, str):
                coverage[key] = artifact
        tool_results: dict[str, object] = {}
        for row in self.list_tool_results(session_id):
            key = row.get("tool_result_id")
            if isinstance(key, str):
                tool_results[key] = row
        return {
            "evidence": evidence,
            "freeze": freezes,
            "dossier": dossiers,
            "coverage": coverage,
            "job": jobs,
            "research": {session_id: session.to_dict()},
            "tool_result": tool_results,
        }

    # -- resume ----------------------------------------------------------

    def resume(self, session_id: str) -> ResumeState:
        """Load session + jobs + journal without writing; completed jobs are never recreated."""
        session = self.get_session(session_id)
        jobs = self.list_jobs(session_id)
        journal_log.hydrate(session_id, self.list_events(session_id))
        open_ids = sorted(j.job_id for j in jobs if j.status in ("queued", "running"))
        return ResumeState(
            session=session,
            wave=session.current_wave,
            budgets=dict(session.budget),
            pending_next_action=pending_next_action(session, jobs),
            open_job_ids=open_ids,
        )

    # -- research nodes / decisions (kernel slice: what needs knowing + JEV dispositions) --

    def save_node(self, node: ResearchNode) -> str:
        """Upsert one research node; fail-closed on unknown session. Returns node_id."""
        node.validate("<research.sqlite>")
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (node.session_id,)).fetchone() is None:
                raise KeyError(f"unknown session_id: {node.session_id!r}")
            conn.execute(
                "INSERT OR REPLACE INTO research_nodes (node_id, session_id, created_at, record) VALUES (?, ?, ?, ?)",
                (
                    node.node_id,
                    node.session_id,
                    utcnow().isoformat(),
                    _record_json(node.to_dict(), "<research.sqlite>: node"),
                ),
            )
        return node.node_id

    def get_node(self, node_id: str) -> ResearchNode:
        """Load one node; raises KeyError when absent."""
        with self._connect() as conn:
            row = conn.execute("SELECT record FROM research_nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown node_id: {node_id!r}")
        return ResearchNode.from_dict(
            validate_json_mapping(json.loads(str(row["record"])), "<research.sqlite>: node"), "<research.sqlite>"
        )

    def list_nodes(self, session_id: str) -> list[ResearchNode]:
        """All nodes for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM research_nodes WHERE session_id = ? ORDER BY rowid", (session_id,)
            ).fetchall()
        return [
            ResearchNode.from_dict(
                validate_json_mapping(json.loads(str(r["record"])), "<research.sqlite>: node"), "<research.sqlite>"
            )
            for r in rows
        ]

    def update_node_status(self, node_id: str, status: str) -> ResearchNode:
        """Transition one node's status; raises KeyError when absent, ValueError on bad status."""
        from dataclasses import replace

        node = self.get_node(node_id)
        updated = replace(node, status=status)
        updated.validate("<research.sqlite>")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_nodes (node_id, session_id, created_at, record) VALUES (?, ?, ?, ?)",
                (
                    updated.node_id,
                    updated.session_id,
                    utcnow().isoformat(),
                    _record_json(updated.to_dict(), "<research.sqlite>: node"),
                ),
            )
        return updated

    def add_node_evidence(self, node_id: str, evidence_id: str) -> ResearchNode:
        """Append one evidence id to a node (idempotent); raises KeyError when absent."""
        from dataclasses import replace

        if not evidence_id:
            raise ValueError("<research.sqlite>: node evidence_id must be a non-empty string")
        node = self.get_node(node_id)
        if evidence_id in node.evidence_ids:
            return node
        updated = replace(node, evidence_ids=(*node.evidence_ids, evidence_id))
        updated.validate("<research.sqlite>")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_nodes (node_id, session_id, created_at, record) VALUES (?, ?, ?, ?)",
                (
                    updated.node_id,
                    updated.session_id,
                    utcnow().isoformat(),
                    _record_json(updated.to_dict(), "<research.sqlite>: node"),
                ),
            )
        return updated

    def save_decision(
        self,
        decision: DecisionRecord,
        *,
        request: object = None,
        response: object = None,
        provider: str | None = None,
        latency_ms: float | None = None,
    ) -> str:
        """Persist one JEV disposition (domain row to research.sqlite).

        Full request/response/provider latency mirror to runs.sqlite agent_events
        best-effort via the recorder path; storage failure never breaks the write.
        """
        decision.validate("<research.sqlite>")
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (decision.session_id,)).fetchone() is None:
                raise KeyError(f"unknown session_id: {decision.session_id!r}")
            if decision.node_id is not None and (
                conn.execute("SELECT 1 FROM research_nodes WHERE node_id = ?", (decision.node_id,)).fetchone() is None
            ):
                raise KeyError(f"unknown node_id: {decision.node_id!r}")
            try:
                conn.execute(
                    "INSERT INTO decisions (decision_id, session_id, node_id, job_id, created_at, record)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        decision.session_id,
                        decision.node_id,
                        decision.job_id,
                        decision.created_at.isoformat(),
                        _record_json(decision.to_dict(), "<research.sqlite>: decision"),
                    ),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"<research.sqlite>: duplicate decision_id {decision.decision_id!r}") from None
        self._mirror_decision(decision, request=request, response=response, provider=provider, latency_ms=latency_ms)
        return decision.decision_id

    def list_decisions(self, session_id: str) -> list[DecisionRecord]:
        """All decisions for one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT record FROM decisions WHERE session_id = ? ORDER BY created_at, decision_id",
                (session_id,),
            ).fetchall()
        return [
            DecisionRecord.from_dict(
                validate_json_mapping(json.loads(str(r["record"])), "<research.sqlite>: decision"),
                "<research.sqlite>",
            )
            for r in rows
        ]

    def _mirror_decision(
        self,
        decision: DecisionRecord,
        *,
        request: object,
        response: object,
        provider: str | None,
        latency_ms: float | None,
    ) -> None:
        """Best-effort decision mirror to runs.sqlite agent_events (never raises)."""
        try:
            from ..storage.runs import RunRecorder, get_runs_db_path

            doc = decision.to_dict()
            path = get_runs_db_path(self._path.parent)
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path))
            try:
                RunRecorder._migrate_schema(conn)
                seq = conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM agent_events WHERE run_id = ?",
                    (decision.session_id,),
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO agent_events (event_id, run_id, sequence, event_type,"
                    " started_at, completed_at, duration_ms, round, model, tool_name,"
                    " arguments, result_summary, success, error_type, evidence_ids, metadata)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        decision.session_id,
                        seq,
                        "jev.decision",
                        decision.created_at.isoformat(),
                        decision.created_at.isoformat(),
                        latency_ms,
                        None,
                        provider or "jev",
                        str(decision.decision_type),
                        json.dumps({"request": request, "candidates": doc["candidates"]}, sort_keys=True, default=str),
                        json.dumps(
                            {"selected": doc["selected"], "probabilities": doc["probabilities"]},
                            sort_keys=True,
                            default=str,
                        ),
                        1,
                        None,
                        None,
                        json.dumps(
                            {
                                "decision_id": decision.decision_id,
                                "node_id": decision.node_id,
                                "job_id": decision.job_id,
                                "response": response,
                                "confidence": doc["confidence"],
                            },
                            sort_keys=True,
                            default=str,
                        ),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001, S110 - observability mirror, never breaks persistence
            pass
