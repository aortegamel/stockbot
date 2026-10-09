"""Live SEC read seam: typed objects from providers, raw hashing helpers.

Seam: live reads via SourceGateway + normalization (app/sec/*) + raw_archive
(write-once) + per-session bundle writer. Providers (EdgarTools/SEC) are
authoritative; nothing here persists except the SQLite job/coverage ledger.

NOTE: a future warehouse would slot in behind these query_* names — add it
as a new module and route these functions through it; do not regrow
store_*/persist_* writers here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

from ..config import get_data_root
from ..storage import raw_archive
from .models import Filing, pit_of

__all__ = [
    "PARSER_VERSION",
    "canonical_json",
    "claim_job",
    "complete_job",
    "content_hash",
    "enqueue_backfill_job",
    "ensure_jobs_table",
    "fail_job",
    "get_job",
    "is_partition_covered",
    "ledger_hash",
    "list_jobs",
    "query_13f_holdings",
    "query_beneficial_ownership",
    "query_coverage",
    "query_document_text",
    "query_filings",
    "query_insider_transactions",
    "recover_stale_jobs",
    "requeue_job",
    "search_document_text",
]

PARSER_VERSION = "1"

_AS_OF_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_date(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be YYYY-MM-DD, got {value!r}")
    if not _AS_OF_RE.match(value):
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}") from None
    return value


def _validate_as_of(as_of: object) -> str:
    return _validate_date(as_of, "as_of")


def _check_as_of_opt(as_of: object) -> str | None:
    if as_of is None:
        return None
    return _validate_as_of(as_of)


def _known_as_of_filing(filing: Filing, as_of: str | None) -> bool:
    if as_of is None:
        return True
    value, _basis = pit_of(filing)
    return value is not None and value[:10] <= as_of


# Live reads: SourceGateway + normalization; NOTE warehouse slots behind these names.
def query_filings(
    *,
    cik: int | str | None = None,
    accession: str | None = None,
    forms: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of: str | None = None,
    limit: int | None = 200,
    root: Path | str | None = None,
) -> list[Filing]:
    """Live filing reads as typed ``Filing`` objects (no warehouse)."""
    del root
    bound = _check_as_of_opt(as_of)
    if start_date is not None:
        _validate_date(start_date, "start_date")
    if end_date is not None:
        _validate_date(end_date, "end_date")
    if forms is not None:
        if not isinstance(forms, list):
            raise TypeError(f"forms must be a list of strings, got {forms!r}")
        for entry in forms:
            if not isinstance(entry, str):
                raise TypeError(f"forms must be a list of strings, got {entry!r}")
    if accession is not None:
        if not isinstance(accession, str):
            raise TypeError(f"accession must be a string, got {accession!r}")
        from ..data_sources import SourceGateway as _Gateway

        try:
            filing = _Gateway().get_filing(accession, as_of=bound)
        except ValueError as exc:
            if "not known as of" in str(exc):
                return []
            raise
        return [filing] if _known_as_of_filing(filing, bound) else []
    if cik is None:
        return []
    if isinstance(cik, bool) or not isinstance(cik, (int, str)):
        raise TypeError(f"cik must be int or str, got {cik!r}")
    from .filings import list_sec_filings as _live_list

    try:
        cik_int = int(str(cik).strip())
    except ValueError:
        raise ValueError(f"invalid cik: {cik!r}") from None
    want = {f.strip().upper() for f in (forms or []) if f.strip()}
    filings = _live_list(
        cik_int, forms=list(want) or None, start_date=start_date, end_date=end_date, as_of=bound, limit=limit
    )
    return list(filings) if limit is None else list(filings)[:limit]


def content_hash(data: bytes) -> str:
    """sha256 hex of exact payload bytes (raw-archive dedup key)."""
    return raw_archive.content_hash(data)


def canonical_json(payload: Mapping[str, object]) -> bytes:
    """Canonical JSON bytes for hashing/comparison."""
    return json.dumps(dict(payload), sort_keys=True, default=str).encode("utf-8")


def ledger_hash(*parts: object) -> str:
    """sha256 hex over newline-joined string parts (deterministic ledger ids)."""
    return hashlib.sha256("\n".join("" if p is None else str(p) for p in parts).encode()).hexdigest()


def _typed_stub_row(row: object) -> dict[str, object]:
    """Dict copy for dict rows; to_dict rows must also produce dicts."""
    if isinstance(row, dict):
        return dict(row)
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, dict):
            return dict(mapped)
        raise TypeError(f"to_dict must return a dict, got {type(mapped).__name__}")
    raise TypeError(f"row must be a dict or provide to_dict, got {type(row).__name__}")


def _live_text_records(accession: str, document_name: str | None, as_of: str | None) -> list[Mapping[str, object]]:
    from . import documents as _documents

    doc = _documents.get_sec_document(accession, document_name, as_of=as_of)
    if not isinstance(doc, dict):
        raise TypeError(f"document must be a dict, got {type(doc).__name__}")
    text = doc.get("text")
    if text is None or text == "":
        return []
    if not isinstance(text, str):
        raise TypeError(f"document text must be a string, got {type(text).__name__}")
    return [doc]


def query_document_text(
    *,
    doc_id: str | None = None,
    accession: str | None = None,
    document_name: str | None = None,
    as_of: str | None = None,
    limit: int | None = 200,
    root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Live document text for one accession (no persisted text index)."""
    del doc_id, root
    _check_as_of_opt(as_of)
    if accession is None:
        return []
    if not isinstance(accession, str):
        raise TypeError(f"accession must be a string, got {accession!r}")
    if document_name is not None and not isinstance(document_name, str):
        raise TypeError(f"document_name must be a string, got {document_name!r}")
    rows = _live_text_records(accession, document_name, as_of)
    out: list[dict[str, object]] = [dict(row) for row in rows if isinstance(row, dict)]
    return out if limit is None else out[:limit]


def search_document_text(
    query: str,
    *,
    literal: bool = False,
    as_of: str | None = None,
    limit: int | None = 200,
    root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Live text search delegates to provider full-text search (no local FTS)."""
    del literal, root
    bound = _check_as_of_opt(as_of)
    if not isinstance(query, str):
        raise TypeError(f"query must be a non-empty string, got {query!r}")
    text = query.strip()
    if not text:
        raise ValueError("query must be a non-empty string")
    from ..data_sources import SourceGateway as _Gateway

    result = _Gateway().search_filings(query=text, limit=limit or 20, as_of=bound)
    hits = list(result.text_hits or ())
    out: list[dict[str, object]] = []
    for hit in hits:
        out.append(hit.to_dict())
    return out if limit is None else out[:limit]


def _row_sort_key(row: dict[str, object]) -> tuple[str, str]:
    """Newest-first sort key for typed rows (known_at, filed_at)."""
    return (str(row.get("known_at") or ""), str(row.get("filed_at") or ""))


def _typed_rows(records: Iterable[object], as_of: str | None, limit: int | None) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for rec in records:
        value, _basis = pit_of(rec)
        if as_of is not None and (value is None or value[:10] > as_of):
            continue
        out.append(_typed_stub_row(rec))
    newest = sorted(out, key=_row_sort_key, reverse=True)
    return newest if limit is None else newest[:limit]


def _gateway_get_filing(accession: str, as_of: str | None) -> Filing:
    if not isinstance(accession, str):
        raise TypeError(f"accession must be a string, got {accession!r}")
    from ..data_sources import SourceGateway as _Gateway

    return _Gateway().get_filing(accession, as_of=as_of)


def _filing_meta(accession: str, as_of: str | None) -> Filing | None:
    """Live filing metadata for one accession (None when PIT-excluded)."""
    try:
        return _gateway_get_filing(accession, as_of)
    except ValueError as exc:
        if "not known as of" in str(exc):
            return None
        raise


def _infotable_of(accession: str) -> object | None:
    """Live 13F information table for one accession (None when the filing exposes none)."""
    from .documents import get_by_accession_number

    filing = get_by_accession_number(accession)
    inner: object = filing.obj() if hasattr(filing, "obj") else filing
    if inner is None:
        inner = filing
    for attr in ("infotable", "information_table", "holdings", "info_table"):
        table: object = getattr(inner, attr, None)
        if table is not None:
            return table
    return None


def _accession_holdings(accession: str, as_of: str | None) -> list[object]:
    from . import insider as _insider

    filing = _filing_meta(accession, as_of)
    if filing is None or filing.form.strip().upper() not in ("13F-HR", "13F-HR/A"):
        return []
    table = _infotable_of(accession)
    if table is None:
        return []
    return list(
        _insider.normalize_13f_holdings(
            table,
            manager_name=filing.filer_name,
            manager_cik=filing.filer_cik,
            accession_no=accession,
            report_period=filing.report_period,
            filed_at=filing.filed_at,
            form=filing.form,
            known_at=filing.known_at,
            source_url=filing.source,
        )
    )


def _accession_transactions(accession: str, as_of: str | None) -> list[object]:
    from . import insider as _insider

    filing = _filing_meta(accession, as_of)
    if filing is None or filing.form.strip().upper() not in ("3", "3/A", "4", "4/A", "5", "5/A"):
        return []
    obj = _insider.load_ownership(accession)
    return list(
        _insider.normalize_ownership_filing(
            obj,
            issuer=filing.filer_name,
            form=filing.form,
            filed_at=filing.filed_at,
            accession_no=accession,
            issuer_cik=filing.filer_cik,
            known_at=filing.known_at,
        )
    )


def _check_query_limit(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"limit must be int or None, got {value!r}")
    return value


def _check_query_root(value: object) -> Path | str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"root must be str, Path, or None, got {value!r}")
    return value


def query_beneficial_ownership(*args: object, **kwargs: object) -> list[dict[str, object]]:
    """Ownership rows resolve live per accession when given, else []."""
    bound = _check_as_of_opt(kwargs.get("as_of"))
    accession = kwargs.get("accession")
    cap = _check_query_limit(kwargs.get("limit"))
    if accession is None:
        return []
    if not isinstance(accession, str):
        raise TypeError(f"accession must be a string, got {accession!r}")
    if not accession:
        raise ValueError("accession must be a non-empty string")
    from . import ownership as _ownership

    filing = _filing_meta(accession, bound)
    if filing is None:
        return []
    schedule = _ownership.load_schedule(accession)
    recs = _ownership.normalize_schedule(
        schedule,
        issuer=filing.filer_name,
        form=filing.form,
        filed_at=filing.filed_at,
        accession_no=accession,
        known_at=filing.known_at,
        source_url=filing.source,
    )
    rows = [rec.to_dict() for rec in (recs or [])]
    return _typed_rows(rows, bound, cap)


def query_13f_holdings(*args: object, **kwargs: object) -> list[dict[str, object]]:
    """Holdings resolve live per accession when given, else []."""
    bound = _check_as_of_opt(kwargs.get("as_of"))
    accession = kwargs.get("accession")
    cap = _check_query_limit(kwargs.get("limit"))
    if accession is None:
        return []
    if not isinstance(accession, str):
        raise TypeError(f"accession must be a string, got {accession!r}")
    if not accession:
        raise ValueError("accession must be a non-empty string")
    return _typed_rows(_accession_holdings(accession, bound), bound, cap)


def query_insider_transactions(*args: object, **kwargs: object) -> list[dict[str, object]]:
    """Insider rows resolve live per accession when given, else []."""
    bound = _check_as_of_opt(kwargs.get("as_of"))
    accession = kwargs.get("accession")
    cap = _check_query_limit(kwargs.get("limit"))
    if accession is None:
        return []
    if not isinstance(accession, str):
        raise TypeError(f"accession must be a string, got {accession!r}")
    if not accession:
        raise ValueError("accession must be a non-empty string")
    return _typed_rows(_accession_transactions(accession, bound), bound, cap)


def query_coverage(*args: object, **kwargs: object) -> list[dict[str, object]]:
    """Coverage derives from the job ledger (complete jobs), not a coverage table."""
    source = kwargs.get("source")
    form = kwargs.get("form")
    date_partition = kwargs.get("date_partition")
    cap = _check_query_limit(kwargs.get("limit"))
    rows: list[dict[str, object]] = []
    root_arg = _check_query_root(kwargs.get("root"))
    for job in list_jobs(status="complete", root=root_arg):
        job_form = str(job.get("form") or "")
        job_source = str(job.get("source") or "")
        if isinstance(form, str) and job_form != form:
            continue
        if isinstance(source, str) and job_source != source:
            continue
        partition = f"{job.get('start_date')}:{job.get('end_date')}"
        if isinstance(date_partition, str) and partition != date_partition:
            continue
        rows.append(
            {
                "source": job_source,
                "form": job_form,
                "date_partition": partition,
                "status": "complete",
                "last_key": job.get("last_key"),
            }
        )
    return rows if cap is None else rows[:cap]


def is_partition_covered(source: str, form: str, date_partition: str, *, root: Path | str | None = None) -> bool:
    """True when a ``complete`` job covers the partition (job ledger is the coverage source)."""
    jobs = list_jobs(status="complete", root=root)
    for job in jobs:
        start = str(job.get("start_date") or "")
        end = str(job.get("end_date") or "")
        if str(job.get("source") or "") != source or str(job.get("form") or "") != form:
            continue
        if f"{start}:{end}" == date_partition or start == date_partition or end == date_partition:
            return True
    return False


# Job ledger (SQLite ops ledger): coverage reads here; NOTE warehouse slots apart from this.
# --- Phase 5: durable backfill queue (mutable SQLite ops ledger; no warehouse) ---

_JOBS_LOCK = threading.Lock()
_JOBS_TABLE = "sec_backfill_jobs"


def _jobs_db_path(root: Path | str | None = None) -> Path:
    """Jobs DB lives at ``<data_root>/sec_backfill.sqlite``."""
    if root is None:
        return get_data_root() / "sec_backfill.sqlite"
    return Path(root) / "sec_backfill.sqlite"


def ensure_jobs_table(root: Path | str | None = None) -> Path:
    """Create the jobs table if missing; returns the SQLite path.

    Never touches job states: interrupted ``running`` leases are recovered
    explicitly via :func:`recover_stale_jobs` at worker/drain start, so a
    live lease is never stolen mid-flight.
    """
    path = _jobs_db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_JOBS_TABLE} ("
                "id TEXT PRIMARY KEY, source TEXT NOT NULL, form TEXT NOT NULL, "
                "family TEXT, start_date TEXT NOT NULL, end_date TEXT NOT NULL, "
                "parser_version TEXT NOT NULL, status TEXT NOT NULL, "
                "batch_size INTEGER NOT NULL DEFAULT 50, "
                "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, "
                "last_key TEXT, error TEXT)"
            )
            conn.commit()
        finally:
            conn.close()
    return path


def _backfill_job_id(
    source: str, form: str, start_date: str, end_date: str, parser_version: str, family: str | None = None
) -> str:
    digest = hashlib.sha256(
        f"{source}\n{form}\n{family or ''}\n{start_date}\n{end_date}\n{parser_version}".encode()
    ).hexdigest()[:16]
    return f"job:{digest}"


def _backfill_alias_values(
    form: str | None,
    start_date: str | None,
    end_date: str | None,
    aliases: dict[str, object],
) -> tuple[object, object, object]:
    """Explicit args with legacy alias fallbacks for form/start/end."""
    form_value: object = form if form is not None else aliases.get("form_")
    start_value: object = start_date
    if start_value is None:
        start_value = aliases.get("from_date", aliases.get("from_", aliases.get("start", aliases.get("from"))))
    end_value: object = end_date
    if end_value is None:
        end_value = aliases.get("to_date", aliases.get("to_", aliases.get("end", aliases.get("to"))))
    return form_value, start_value, end_value


def _require_backfill_source(source: str) -> None:
    """Required queue source; raises when blank."""
    if not source or not source.strip():
        raise ValueError("source is required (e.g. sec-global)")


def _require_backfill_form(form_value: object) -> None:
    """Required filing form; raises when blank."""
    if not form_value or not str(form_value).strip():
        raise ValueError("form is required (e.g. 10-K)")


def _require_backfill_dates(start_value: object, end_value: object) -> None:
    """Required start/end dates; raises when either is missing."""
    if start_value is None or end_value is None:
        raise ValueError("start/end dates are required (YYYY-MM-DD); no all-history default")


def _require_backfill_fields(
    source: str,
    form_value: object,
    start_value: object,
    end_value: object,
) -> None:
    """Required source/form/dates; raises naming the missing field."""
    _require_backfill_source(source)
    _require_backfill_form(form_value)
    _require_backfill_dates(start_value, end_value)


def _validate_backfill_range(
    source: str,
    form_value: object,
    start_value: object,
    end_value: object,
) -> tuple[str, str]:
    """Required source/form/dates to a validated (start, end) date pair."""
    _require_backfill_fields(source, form_value, start_value, end_value)
    start = _validate_date(start_value, "start_date")
    end = _validate_date(end_value, "end_date")
    if start > end:
        raise ValueError(f"invalid date range: {start!r}..{end!r}")
    return start, end


def _insert_backfill_job(
    job_id: str,
    source: str,
    form_value: object,
    family: str | None,
    start: str,
    end: str,
    parser_version: str,
    batch_size: int,
    root: Path | str | None,
) -> None:
    """Idempotent queued insert for one deterministic job ID."""
    path = ensure_jobs_table(root)
    now = _utcnow()
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            conn.execute(
                f"INSERT OR IGNORE INTO {_JOBS_TABLE} "
                "(id, source, form, family, start_date, end_date, "
                "parser_version, status, batch_size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                (
                    job_id,
                    source,
                    str(form_value),
                    family if family is not None else None,
                    start,
                    end,
                    parser_version,
                    batch_size or 50,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def enqueue_backfill_job(
    source: str,
    form: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    parser_version: str = PARSER_VERSION,
    *,
    family: str | None = None,
    batch_size: int = 50,
    root: Path | str | None = None,
    **aliases: object,
) -> str:
    """Idempotent queue insert; reruns return the same deterministic ID."""
    form_value, start_value, end_value = _backfill_alias_values(form, start_date, end_date, aliases)
    if parser_version == PARSER_VERSION and aliases.get("parser") is not None:
        parser_version = str(aliases["parser"])
    start, end = _validate_backfill_range(source, form_value, start_value, end_value)
    job_id = _backfill_job_id(source, str(form_value), start, end, parser_version, family)
    _insert_backfill_job(job_id, source, form_value, family, start, end, parser_version, batch_size, root)
    return job_id


def _row_to_job(row: tuple[object, ...]) -> dict[str, object]:
    keys = (
        "id",
        "source",
        "form",
        "family",
        "start_date",
        "end_date",
        "parser_version",
        "status",
        "batch_size",
        "created_at",
        "started_at",
        "finished_at",
        "last_key",
        "error",
    )
    return dict(zip(keys, row))


def get_job(job_id: str, *, root: Path | str | None = None) -> dict[str, object] | None:
    """One job row by ID, or None."""
    path = ensure_jobs_table(root)
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        cur = conn.execute(
            f"SELECT id, source, form, family, start_date, end_date, "
            f"parser_version, status, batch_size, created_at, started_at, "
            f"finished_at, last_key, error FROM {_JOBS_TABLE} WHERE id = ?",
            (job_id,),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    return _row_to_job(row) if row else None


def list_jobs(
    *, status: str | None = None, root: Path | str | None = None, limit: int = 200
) -> list[dict[str, object]]:
    """Jobs oldest first, optionally filtered by status."""
    path = ensure_jobs_table(root)
    sql = (
        f"SELECT id, source, form, family, start_date, end_date, "
        f"parser_version, status, batch_size, created_at, started_at, "
        f"finished_at, last_key, error FROM {_JOBS_TABLE}"
    )
    params: list[str | None] = []
    if status is not None:
        sql += " WHERE status = ?"
        params.append(status)
    sql += f" ORDER BY created_at ASC LIMIT {limit}"
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_job(r) for r in rows]


def claim_job(job_id: str | None = None, *, root: Path | str | None = None) -> dict[str, object] | None:
    """Lease one job as ``running``.

    Auto-claim takes the oldest ``queued`` job only: a ``failed`` job stays
    an explicit coverage failure until resumed (see :func:`requeue_job`),
    which also keeps queue drains from re-failing forever. An explicit
    ``job_id`` may lease a ``queued`` or ``failed`` job for targeted retry.
    """
    path = ensure_jobs_table(root)
    now = _utcnow()
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            if job_id is not None:
                cur = conn.execute(
                    f"UPDATE {_JOBS_TABLE} SET status='running', "
                    "started_at=?, finished_at=NULL, error=NULL WHERE id = ? "
                    "AND status IN ('queued', 'failed')",
                    (now, job_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
            else:
                cur = conn.execute(
                    f"SELECT id FROM {_JOBS_TABLE} WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
                )
                found = cur.fetchone()
                if not found:
                    return None
                cur = conn.execute(
                    f"UPDATE {_JOBS_TABLE} SET status='running', "
                    "started_at=?, finished_at=NULL, error=NULL WHERE id = ? "
                    "AND status = 'queued'",
                    (now, found[0]),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
                job_id = found[0]
        finally:
            conn.close()
    return get_job(str(job_id), root=root)


def complete_job(
    job_id: str, *, last_key: str | None = None, root: Path | str | None = None
) -> dict[str, object] | None:
    """Mark a leased job ``complete``."""
    path = ensure_jobs_table(root)
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            if last_key is not None:
                conn.execute(
                    f"UPDATE {_JOBS_TABLE} SET status='complete', finished_at=?, last_key=? WHERE id = ?",
                    (_utcnow(), last_key, job_id),
                )
            else:
                conn.execute(
                    f"UPDATE {_JOBS_TABLE} SET status='complete', finished_at=? WHERE id = ?", (_utcnow(), job_id)
                )
            conn.commit()
        finally:
            conn.close()
    return get_job(job_id, root=root)


def fail_job(
    job_id: str, error: object = "", *, last_key: str | None = None, root: Path | str | None = None
) -> dict[str, object] | None:
    """Mark a leased job ``failed``; the accession/partition is retried."""
    path = ensure_jobs_table(root)
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            if last_key is not None:
                conn.execute(
                    f"UPDATE {_JOBS_TABLE} SET status='failed', finished_at=?, last_key=?, error=? WHERE id = ?",
                    (_utcnow(), last_key, str(error), job_id),
                )
            else:
                conn.execute(
                    f"UPDATE {_JOBS_TABLE} SET status='failed', finished_at=?, error=? WHERE id = ?",
                    (_utcnow(), str(error), job_id),
                )
            conn.commit()
        finally:
            conn.close()
    return get_job(job_id, root=root)


def requeue_job(job_id: str, *, root: Path | str | None = None) -> dict[str, object] | None:
    """Return a ``failed``/``complete`` job to ``queued`` for resume."""
    path = ensure_jobs_table(root)
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            conn.execute(
                f"UPDATE {_JOBS_TABLE} SET status='queued', started_at=NULL, finished_at=NULL, error=NULL WHERE id = ?",
                (job_id,),
            )
            conn.commit()
        finally:
            conn.close()
    return get_job(job_id, root=root)


def recover_stale_jobs(*, root: Path | str | None = None) -> int:
    """Recover interrupted ``running`` leases to ``queued``; returns count."""
    path = ensure_jobs_table(root)
    with _JOBS_LOCK:
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            cur = conn.execute(f"UPDATE {_JOBS_TABLE} SET status='queued', started_at=NULL WHERE status='running'")
            conn.commit()
            return cur.rowcount or 0
        finally:
            conn.close()
