"""Research tool runner: one model-requested tool call and the evidence it admits.

This module owns tool execution and evidence admission for the native research loop.
It moved out of app/research/scheduler.py, whose round machinery is retired.

The model supplies the tool name and arguments. Nothing here judges topical relevance:
admission keeps every structural check (domain, provenance, point-in-time, ids, proof)
and drops the retired model verdict.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json as _json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from app.research import service
from app.research.agents.source_agent import source_domain_for_tool
from app.research.models import JSONValue, request_cutoff
from app.research.repository import ResearchRepository
from app.tool_runtime import RuntimeToolSession, ToolOutcome, execute_agent_tool, outcome_from_result

logger = logging.getLogger(__name__)

# ponytail: dedicated pools keep a stuck provider call out of the loop's default
# executor, so asyncio.run never waits for a straggler at loop shutdown.
_SEC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="sec")
_TOOL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

_SUMMARY_ROW_CHARS = 1200
"""Characters kept per admitted evidence row in the model-facing summary."""
_SUMMARY_TEXT_ROW_CHARS = 300
"""Characters kept for a row with no datapoint: its text is context, never a citable number."""
_SUMMARY_TOTAL_CHARS = 8000
"""Characters kept across the whole model-facing summary."""


class ResearchKernel(Protocol):
    """The kernel operations one research tool call uses (implemented by Kernel below)."""

    def start_job(
        self,
        session_id: str,
        type: str = "source_agent",
        source: str | None = None,
        *,
        owner: str = "kernel",
        request_id: str | None = None,
    ) -> dict[str, JSONValue]: ...

    def heartbeat_job(self, job_id: str) -> None: ...

    def complete_job(self, job_id: str, outcome: Mapping[str, object] | None = None) -> None: ...

    def fail_job(self, job_id: str, category: str, message: str) -> None: ...

    def limit_job_deadline(self, job_id: str, deadline_at: float) -> None: ...

    def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]: ...

    def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]: ...

    def get_session(self, session_id: str) -> dict[str, JSONValue]: ...

    def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]: ...


class Kernel:
    """Store-backed kernel for the research loop: jobs, evidence admission, session reads."""

    def __init__(self, repo: ResearchRepository | Path | str | None = None) -> None:
        self._repo = repo

    def _store(self) -> ResearchRepository:
        if isinstance(self._repo, ResearchRepository):
            return self._repo
        return ResearchRepository(self._repo)

    def get_session(self, session_id: str) -> dict[str, JSONValue]:
        return self._store().get_session(session_id).to_dict()

    def list_evidence(self, session_id: str) -> list[dict[str, JSONValue]]:
        return self._store().list_evidence(session_id)

    def get_tool_result(self, tool_result_id: str) -> dict[str, JSONValue]:
        return self._store().get_tool_result(tool_result_id)

    def start_job(
        self,
        session_id: str,
        type: str = "source_agent",
        source: str | None = None,
        *,
        owner: str = "kernel",
        request_id: str | None = None,
    ) -> dict[str, JSONValue]:
        return service.start_job(session_id, type, source, repo=self._repo, owner=owner, request_id=request_id)

    def heartbeat_job(self, job_id: str) -> None:
        service.heartbeat_job(job_id, repo=self._repo)

    def complete_job(self, job_id: str, outcome: Mapping[str, object] | None = None) -> None:
        service.complete_job(job_id, outcome, repo=self._repo)

    def fail_job(self, job_id: str, category: str, message: str) -> None:
        service.fail_job(job_id, category, message, repo=self._repo)

    def limit_job_deadline(self, job_id: str, deadline_at: float) -> None:
        """Bind the job to the remaining request budget before the call starts."""
        deadline = datetime.now(UTC) + timedelta(seconds=deadline_at - time.perf_counter())
        service.limit_job_deadline(job_id, deadline, repo=self._repo)

    def admit_evidence(self, session_id: str, job_id: str, data: Mapping[str, object]) -> dict[str, JSONValue]:
        return service.admit_evidence(session_id, job_id, data, repo=self._repo)


# Discovery tools are the selection mechanism itself, not research actions. Session and
# UI control tools create or re-enter research and change the desk, so a research loop
# never calls them. One allowlist is the surviving authority for the loop.
_TOOL_EXCLUDED = frozenset(
    {
        "call_tool",
        "browse_tools",
        "search_tools",
        "list_tool_domains",
        "describe_tool",
        "research_start",
        "research_read",
        "research_read_search",
        "research_resume",
        "set_chart_symbol",
        "update_watchlist",
        "add_widget",
    }
)

_SEC_NAVIGATION_TOOLS = frozenset({"list_sec_filings", "get_sec_filing", "list_sec_documents"})
"""Catalog-only SEC tools: they name filings/documents but never carry document text."""


@functools.cache
def research_tool_names() -> frozenset[str]:
    """Canonical RESEARCH tools the loop may call, minus discovery and control tools."""
    from app.policy import Capability
    from app.tools import tools_for_capabilities

    names: set[str] = set()
    for tool in tools_for_capabilities(frozenset({Capability.RESEARCH})):
        fn = tool.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name and name not in _TOOL_EXCLUDED:
            names.add(name)
    return frozenset(names)


RESEARCH_TOOLS: frozenset[str] = research_tool_names()
"""The loop's executable tools: one source for the accept check and the advertised list."""


def shutdown_pools() -> None:
    """Drop queued pool work without blocking; an already-closed pool never breaks exit."""
    for pool in (_SEC_POOL, _TOOL_POOL):
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except AttributeError, RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Evidence candidates: kernel-side bytes from one persisted tool result
# ---------------------------------------------------------------------------


def _f(obj: object, *names: str, default: object = None) -> object:
    """First present attribute/key under ``names``, else ``default``."""
    for name in names:
        if isinstance(obj, dict) and name in obj:
            found: object = obj[name]
            return found
        value: object = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _persisted_shapes(result: object) -> Mapping[str, object]:
    """Persisted payload (tool_result.result or result) for replay-shaped locators."""
    if isinstance(result, dict):
        inner = result.get("result")
        if isinstance(inner, dict):
            return inner
        return result
    return {}


def _compact_json(value: object) -> str:
    """One-line JSON with stable key order; string contents stay byte-exact so ``cited_record`` replays them."""
    return _json.dumps(value, sort_keys=True, default=str)


_OUTCOME_SUMMARY_MAX = 2000
"""Characters of one tool result kept for the model when the call admitted no evidence row."""


def _outcome_summary(outcome: object) -> str:
    text = _f(outcome, "content", default=None)
    if text is None:
        text = str(outcome)
    if not isinstance(text, str):
        text = str(text)
    return text[:_OUTCOME_SUMMARY_MAX]


def _tool_result_ref(result: object, outcome: object = None) -> str | None:
    """Persisted FINRA/WEB tool_result_id where one exists; else None (inline summary covers it)."""
    for obj in (result, outcome):
        ref = _f(obj, "tool_result_id", "tool_result_ref", default=None)
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    return _handle_result_ref(result)


def _handle_result_ref(result: object) -> str | None:
    """Read result references from source handles before legacy top-level fields."""
    handle = _f(result, "source_handle", default=None)
    if isinstance(handle, dict):
        for key in ("tool_result_id", "source_handle_id", "result_id"):
            value = handle.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(result, dict):
        for key in ("source_handle_id", "result_id"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _finra_locator(payload: Mapping[str, object]) -> str | None:
    """First citable FINRA record: canonical row JSON, then the complete envelope.

    Analysis envelopes of ``get_short_interest``/``get_reg_sho_volume`` carry no
    rows, so their identity is the whole payload (``cited_record`` only matches
    full canonical JSON). Metrics alone can coincide between two symbols and the
    query JSON is a filter, not the persisted result, so neither is citable.
    """
    records = payload.get("records")
    if isinstance(records, list):
        for row in records:
            if isinstance(row, dict):
                return _compact_json(row)
    return _compact_json(dict(payload)) if payload else None


def _web_locator(payload: Mapping[str, object]) -> tuple[str | None, str | None]:
    """(url, highlight) of the first persisted web row with both set."""
    rows = payload.get("evidence")
    if not isinstance(rows, list):
        return None, None
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        highlight = row.get("highlight")
        if isinstance(url, str) and url.strip() and isinstance(highlight, str) and highlight.strip():
            return url.strip(), " ".join(highlight.split())[:2000]
    return None, None


def _sec_locator(payload: Mapping[str, object]) -> str | None:
    """Entire canonical SEC document window, without a summary limit."""
    text = payload.get("text")
    return text if isinstance(text, str) and text.strip() else None


def _finra_evidence_candidate(ref: str, content: str, payload: Mapping[str, object]) -> dict[str, object] | None:
    """FINRA candidate from canonical row JSON; None when uncitable."""
    locator = _finra_locator(payload)
    if locator is None:
        return None
    return {"tool_result_id": ref, "content": content, "record_identity": locator, "matching_passage": locator}


def _web_evidence_candidate(ref: str, content: str, payload: Mapping[str, object]) -> dict[str, object] | None:
    """WEB candidate from persisted url+highlight; None when uncitable."""
    url, highlight = _web_locator(payload)
    if url is None or highlight is None:
        return None
    return {
        "tool_result_id": ref,
        "content": content,
        "url": url,
        "excerpt": highlight,
        "matching_passage": highlight,
    }


def _sec_handle(result: object, outcome: object) -> dict[str, object] | None:
    """SEC document handle from result then outcome; None when missing or empty.

    Never ``source_refs``: that is a catalog receipt (no basis/hashes), so list
    results route through persisted tool-result replay instead.
    """
    handle = _f(result, "source_handle", default=None)
    if handle is None:
        handle = _f(outcome, "source_handle", default=None)
    return handle if isinstance(handle, dict) and handle else None


def _sec_text(result: object, outcome: object) -> str | None:
    """Return canonical text, reloading the handle when the runtime omits it."""
    payload = _persisted_shapes(result)
    if payload.get("error") is not None or _f(outcome, "error") is not None:
        return None
    locator = _sec_locator(payload)
    if locator is not None:
        return locator
    handle = _sec_handle(result, outcome)
    if handle is None:
        return None
    from app.research.service import _handle_fields, _reload_handle_window

    return _sec_locator(_reload_handle_window(_handle_fields(handle), None))


def _persisted_payload_for_locator(ref: str, kernel: ResearchKernel) -> Mapping[str, JSONValue]:
    """Persisted tool-result bytes for the locator; same bytes replay reads."""
    row = kernel.get_tool_result(ref)
    inner = row.get("result")
    if isinstance(inner, dict):
        return inner
    return {k: v for k, v in row.items() if k != "result"}


def _sec_evidence_candidate(result: object, outcome: object, kernel: ResearchKernel) -> dict[str, object] | None:
    """SEC candidate: document handle when present, else persisted replay row."""
    handle = _sec_handle(result, outcome)
    content = _outcome_summary(outcome)
    if not content:
        return None
    if handle is not None:
        locator = _sec_text(result, outcome)
        if locator is None:
            return None
        return {"source_handle": handle, "content": locator, "matching_passage": locator}
    ref = _tool_result_ref(result, outcome)
    if ref is None:
        return None
    try:
        payload = _persisted_payload_for_locator(ref, kernel)
    except KeyError:
        payload = _persisted_shapes(result)
    replay = _sec_replay_locator(payload)
    if replay is None:
        return None
    return {"tool_result_id": ref, "content": content, "record_identity": replay, "matching_passage": replay}


def _sec_replay_locator(payload: Mapping[str, object]) -> str | None:
    """First citable SEC structured record: one mapping-row JSON, else the envelope JSON.

    Non-mapping rows are not citable records (``cited_record`` only matches
    full canonical JSON), so a list of bare scalars falls through to the
    complete payload.
    """
    for key in (
        "filings",
        "transactions",
        "quarterly_eps",
        "facts",
        "matching_concepts",
        "rows",
        "records",
        "documents",
        "events",
        "line_items",
    ):
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    return _compact_json(dict(row))
    return _compact_json(dict(payload))


_ALT_ROW_KEYS: tuple[str, ...] = ("observations", "rows", "signals", "evidence", "series")
"""Row lists of one alternative-data/macro packet: the citable candidates it returned."""

_ALT_OK_STATUS = "ok"
"""Collector status of a packet that may ground evidence; disabled/unavailable/error never does."""


def _alt_locator(payload: Mapping[str, object]) -> str | None:
    """First citable alternative-data/macro row JSON; None when the packet returned no rows."""
    for key in _ALT_ROW_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    return _compact_json(dict(row))
    return None


def _alt_evidence_candidate(ref: str, content: str, payload: Mapping[str, object]) -> dict[str, object] | None:
    """ALT/MACRO candidate from the first returned row; disabled or empty packets stay uncitable."""
    status = payload.get("status")
    if not (isinstance(status, str) and status.strip().lower() == _ALT_OK_STATUS):
        return None
    locator = _alt_locator(payload)
    if locator is None:
        return None
    return {"tool_result_id": ref, "content": content, "record_identity": locator, "matching_passage": locator}


def _structured_evidence_candidate(
    domain: str, ref: str, content: str, payload: Mapping[str, object]
) -> dict[str, object] | None:
    """FINRA/WEB/ALT/MACRO dispatch; None when uncitable."""
    if domain == "FINRA":
        return _finra_evidence_candidate(ref, content, payload)
    if domain in ("ALT", "MACRO"):
        return _alt_evidence_candidate(ref, content, payload)
    return _web_evidence_candidate(ref, content, payload)


def evidence_candidate(
    tool_name: str, domain: str, result: object, outcome: object, kernel: ResearchKernel
) -> dict[str, object] | None:
    """Kernel-side evidence candidate from persisted tool bytes; None when uncitable."""
    if domain not in ("FINRA", "WEB", "SEC", "ALT", "MACRO"):
        return None
    if domain in ("FINRA", "WEB", "ALT", "MACRO"):
        ref = _tool_result_ref(result, outcome)
        if ref is None:
            return None
        try:
            payload = _persisted_payload_for_locator(ref, kernel)
        except KeyError:
            payload = _persisted_shapes(result)
        content = _outcome_summary(outcome)
        if not content:
            return None
        return _structured_evidence_candidate(domain, ref, content, payload)
    if tool_name == "get_sec_document" and _sec_text(result, outcome) is None:
        return None
    return _sec_evidence_candidate(result, outcome, kernel)


# ---------------------------------------------------------------------------
# Admission: the candidate row plus its verified datapoint fan-out
# ---------------------------------------------------------------------------

_LISTING_FACT_METRICS = ("accession_no", "filed_at")
"""Filing facts a claim states as they are: the row's own accession number and its filing date.

The form is not one of them: its value carries digits ("10-K") that no metric label binds, so a
claim names the form as a bare token ("filed 10-K on ...") instead of a metric sentence.
"""
_LISTING_NAME_KEYS = ("filer_name", "subject_name", "company_name")
"""A listing row's entity is the filer's name, not the CIK no claim can read back."""

_DATAPOINT_FANOUT_CAP = 12
"""ponytail: the first 12 numeric fields of one cited row; raise it if wider structured rows need every field."""
_LINE_ITEM_FANOUT_CAP = 60
"""ponytail: the first 60 statement line-item rows (top line items, every period); raise it for deeper lines."""


def _persist_admitted_evidence(
    kernel: ResearchKernel, session_id: str, job_id: str, candidate: Mapping[str, object]
) -> str:
    """Persist one candidate; returns the stored evidence id (the winner on a duplicate).

    Raises when the kernel returns no evidence id.
    """
    admitted_ret = kernel.admit_evidence(session_id, job_id, candidate)
    eid = admitted_ret.get("evidence_id")
    if not (isinstance(eid, str) and eid):
        raise ValueError("admit_evidence returned no evidence_id")
    return eid


def _admit_listing_rows(
    kernel: ResearchKernel, session_id: str, job_id: str, candidate: Mapping[str, object], ids: list[str]
) -> None:
    """Admit every further row of an SEC listing as its own citable record; the parent cites the first row.

    A listing of 10-Ks and 10-Qs otherwise leaves only its newest filing citable. A row the
    kernel rejects stays absent (fail closed); the parent admission is already committed.
    """
    ref = candidate.get("tool_result_id")
    if not isinstance(ref, str) or "record_identity" not in candidate:
        return
    try:
        payload = _persisted_payload_for_locator(ref, kernel)
    except KeyError:
        return  # no persisted bytes to replay a row against
    rows = next((r for key in ("filings", "documents") if isinstance(r := payload.get(key), list)), [])
    for row in rows:
        if not isinstance(row, dict):
            continue
        identity = _compact_json(dict(row))
        if identity != candidate["record_identity"]:
            item = {**candidate, "record_identity": identity, "matching_passage": identity, "content": identity}
            try:
                ids.append(_persist_admitted_evidence(kernel, session_id, job_id, item))
            except ValueError as exc:
                logger.info("toolflow listing_row_skip sid=%s err=%.200s", session_id, exc)
        _admit_listing_datapoints(kernel, session_id, job_id, candidate, row, ids)


def _admit_listing_datapoints(
    kernel: ResearchKernel,
    session_id: str,
    job_id: str,
    candidate: Mapping[str, object],
    row: Mapping[str, object],
    ids: list[str],
) -> None:
    """Admit one listing row's accession number and filing date as verified datapoints of that row.

    Both values are strings, so the numeric field fan-out never selects them. A row the
    proof rejects stays absent (fail closed); the row admission is already committed.
    """
    from app.research import evidence as ev

    entity = next(
        (t for k in (*_LISTING_NAME_KEYS, *ev._ENTITY_KEYS) if (t := ev._scalar_text(row.get(k))) is not None), None
    )
    # The verifier reads a fiscal period first and a filing date after it, so this row declares the same.
    period = next(
        (t for k in (*ev._PERIOD_KEYS, *ev._FILING_DATE_KEYS) if (t := ev._scalar_text(row.get(k))) is not None), None
    )
    if entity is None or period is None:
        return
    identity = _compact_json(dict(row))
    for metric in _LISTING_FACT_METRICS:
        text = ev._scalar_text(row.get(metric))
        if text is None:
            continue
        item = {
            **candidate,
            "record_identity": identity,
            "matching_passage": identity,
            "content": f"{metric}: {text} ({entity}, {period})",
            "metric": metric,
            "value": text,
            "entity": entity,
            "period": period,
        }
        try:
            ids.append(_persist_admitted_evidence(kernel, session_id, job_id, item))
        except ValueError as exc:
            logger.info("toolflow listing_datapoint_skip sid=%s metric=%s err=%.200s", session_id, metric, exc)


def _row_numeric_fields(row: Mapping[str, object]) -> list[tuple[str, str | None]]:
    """Select numeric metrics, with an explicit concept/value pair first and identity fields excluded."""
    from app.research import evidence as ev

    reserved = {
        ev._norm_key(k)
        for k in (*ev._ENTITY_KEYS, *ev._PERIOD_KEYS, *ev._UNIT_KEYS, *ev._CONCEPT_KEYS, *ev._VALUE_KEYS)
    }
    fields = [
        (key, ev._scalar_text(value))
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and ev._norm_key(key) not in reserved
    ]
    concept = next((row[k] for k in ev._CONCEPT_KEYS if isinstance(row.get(k), str) and row[k]), None)
    value = next((t for k in ev._VALUE_KEYS if (t := ev._scalar_text(row.get(k))) is not None), None)
    if isinstance(concept, str) and value is not None:
        fields.insert(0, (concept, value))
    return fields[:_DATAPOINT_FANOUT_CAP]


def _admit_row_datapoints(
    kernel: ResearchKernel, session_id: str, job_id: str, candidate: Mapping[str, object], ids: list[str]
) -> None:
    """Admit one verified datapoint row per numeric field of the cited structured record.

    A FINRA envelope cites its ``latest_record``; a financial statement cites each of its ``line_items``;
    another SEC structured result cites its record row.
    Envelope-level numbers (limits, counts) are never rows. A field the proof rejects stays absent,
    so a claim stating it fails the gate (fail closed). The parent admission is already committed.
    """
    from app.research import evidence as ev

    ref = candidate.get("tool_result_id")
    if not isinstance(ref, str) or "record_identity" not in candidate:
        return
    try:
        payload = _persisted_payload_for_locator(ref, kernel)
    except KeyError:
        return  # no persisted bytes to prove a datapoint against
    latest, listed = payload.get("latest_record"), payload.get("line_items")
    if isinstance(latest, dict):
        rows: Sequence[Mapping[str, object]] = [latest]
    elif isinstance(listed, list):
        rows = [row for row in listed if isinstance(row, dict)][:_LINE_ITEM_FANOUT_CAP]
    else:
        row = ev.cited_record({"result": payload}, candidate["record_identity"])
        rows = [row] if isinstance(row, Mapping) and row is not payload else []
    for row in rows:
        entity = next((t for k in ev._ENTITY_KEYS if (t := ev._scalar_text(row.get(k))) is not None), None)
        period = next((t for k in ev._PERIOD_KEYS if (t := ev._scalar_text(row.get(k))) is not None), None)
        if entity is None or period is None:
            continue
        identity = _compact_json(dict(row))
        for metric, text in _row_numeric_fields(row):
            item = {
                **candidate,
                "record_identity": identity,
                "matching_passage": identity,
                "content": f"{metric}: {text} ({entity}, {period})",
                "metric": metric,
                "value": text,
                "entity": entity,
                "period": period,
            }
            try:
                ids.append(_persist_admitted_evidence(kernel, session_id, job_id, item))
            except ValueError as exc:
                logger.info("toolflow datapoint_skip sid=%s metric=%s err=%.200s", session_id, metric, exc)


def _admit_passage_datapoints(
    kernel: ResearchKernel, session_id: str, job_id: str, candidate: Mapping[str, object], ids: list[str]
) -> None:
    """Admit bounded SEC passage rows through the existing datapoint proof guard."""
    from app.research import evidence as ev

    passage = candidate.get("matching_passage") or candidate.get("content")
    if not isinstance(candidate.get("source_handle"), Mapping) or not isinstance(passage, str):
        return
    seen: set[tuple[str, str]] = set()
    for labels in ev._passage_row_candidates(passage):
        key = (labels["metric"], labels["period"])
        if key in seen:
            continue
        if len(seen) >= _LINE_ITEM_FANOUT_CAP:
            break
        seen.add(key)
        # Short content like the structured fan-out: the passage stays in matching_passage for the proof;
        # copying it into content would resend the whole document window once per row.
        content = f"{labels['metric']}: {labels['value']} ({labels['entity']}, {labels['period']})"
        item = {**candidate, "matching_passage": passage, **labels, "content": content}
        try:
            ids.append(_persist_admitted_evidence(kernel, session_id, job_id, item))
        except ValueError as exc:
            logger.info(
                "toolflow passage_datapoint_skip sid=%s metric=%s err=%.200s", session_id, labels["metric"], exc
            )


def admit_tool_result(
    kernel: ResearchKernel,
    session_id: str,
    job_id: str,
    tool: str,
    result: object,
    outcome: object,
) -> list[str]:
    """Admit one tool result's citable candidate and its datapoint fan-out; returns ids in admission order."""
    domain = source_domain_for_tool(tool)
    candidate = evidence_candidate(tool, domain, result, outcome, kernel)
    if candidate is None:
        return []
    ids = [_persist_admitted_evidence(kernel, session_id, job_id, candidate)]
    if domain in ("SEC", "FINRA"):
        _admit_row_datapoints(kernel, session_id, job_id, candidate, ids)
    if tool == "get_sec_document":
        _admit_passage_datapoints(kernel, session_id, job_id, candidate, ids)
    if tool in _SEC_NAVIGATION_TOOLS:
        _admit_listing_rows(kernel, session_id, job_id, candidate, ids)
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------------------
# Wire rows: what the model reads back
# ---------------------------------------------------------------------------


def row_datapoints(row: Mapping[str, object]) -> list[JSONValue]:
    """The stored canonical datapoint of one ledger row, for model context only.

    The publication gate re-verifies it against the source before any claim may reference it.
    """
    from app.research.evidence import datapoint_descriptor, evidence_from_dict

    try:
        point = datapoint_descriptor(evidence_from_dict(row))
    except ValueError:
        return []
    return [point] if point is not None else []


def _wire_row(row: Mapping[str, object]) -> dict[str, JSONValue]:
    """One evidence row the model reads: id, source label, timing, text, and canonical datapoints."""
    provenance = row.get("provenance")
    prov: Mapping[str, object] = provenance if isinstance(provenance, Mapping) else {}
    metadata = row.get("metadata")
    meta: Mapping[str, object] = metadata if isinstance(metadata, Mapping) else {}
    out: dict[str, JSONValue] = {
        "id": str(row.get("evidence_id") or ""),
        "source": str(row.get("source_name") or ""),
        "retrievedAt": str(row.get("retrieved_at") or ""),
        "content": str(row.get("content") or ""),
    }
    title = meta.get("title") if meta.get("title") is not None else prov.get("title")
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    url = prov.get("url") or prov.get("source_uri") or row.get("source_uri")
    if isinstance(url, str) and url.strip():
        out["url"] = url.strip()
    points = row_datapoints(row)
    if points:
        out["datapoints"] = points
    return out


def evidence_rows(kernel: ResearchKernel, session_id: str, ids: Sequence[str]) -> list[JSONValue]:
    """Wire rows for the admitted ids, in admission order; an id the ledger lost never crosses."""
    by_id = {str(row.get("evidence_id")): row for row in kernel.list_evidence(session_id)}
    return [_wire_row(by_id[eid]) for eid in ids if eid in by_id]


_POINT_FIELDS = ("entity", "metric", "value", "unit", "period")


def _point_line(point: Mapping[str, object]) -> str:
    """One datapoint's identity as the proof states it, so a claim can copy the exact strings."""
    fields = " ".join(f"{key}={point[key]}" for key in _POINT_FIELDS if str(point.get(key) or "").strip())
    return f"  datapoint {point.get('id')}: {fields}"


def evidence_summary(rows: Sequence[JSONValue]) -> str:
    """Compact per-row lines, each prefixed with its evidence id and datapoint identities, capped per row and in total.

    A number sentence must state its datapoint's metric before its value and its entity, period, and
    unit: the model can only do that from the identity strings printed here.
    """
    lines: list[str] = []
    used = 0

    # Rows with a datapoint come first: their identity lines are what a number sentence copies, and an
    # aggregate row ahead of them invites a citation the proof cannot bind.
    def _citable(row: Mapping[str, JSONValue]) -> int:
        points = row.get("datapoints")
        return 0 if isinstance(points, list) and points else 1

    ordered = sorted((row for row in rows if isinstance(row, Mapping)), key=_citable)
    for row in ordered:
        points = row.get("datapoints")
        identities: list[str] = (
            [_point_line(p) for p in points if isinstance(p, Mapping)] if isinstance(points, list) else []
        )
        content = str(row.get("content") or "")[: _SUMMARY_ROW_CHARS if identities else _SUMMARY_TEXT_ROW_CHARS]
        line = "\n".join(
            [
                f"[{row.get('id')}] datapoints" if identities else f"[{row.get('id')}]",
                *identities,
                f"  {content}",
            ]
        ).strip()
        line = line[:_SUMMARY_ROW_CHARS]
        if used + len(line) + 1 > _SUMMARY_TOTAL_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One tool call
# ---------------------------------------------------------------------------


def _session_as_of(session: Mapping[str, JSONValue]) -> str | None:
    """Point-in-time cutoff the request names; None when the request is undated."""
    return request_cutoff(session.get("as_of"), session.get("temporal_scope"), session.get("created_at"))


def _start_job(kernel: ResearchKernel, session_id: str, tool: str, deadline_at: float | None) -> str:
    """Start one source_agent job for this call; a start failure raises to the caller."""
    domain = source_domain_for_tool(tool)
    job = kernel.start_job(
        session_id, type="source_agent", source=None if domain == "OTHER" else domain, owner="kernel-research"
    )
    job_id = str(job["job_id"])
    if deadline_at is not None:
        kernel.limit_job_deadline(job_id, deadline_at)
    return job_id


def _fail_job(kernel: ResearchKernel, job_id: str, message: str, category: str = "tool_error") -> None:
    """Terminal-fail one job; kernel errors propagate."""
    kernel.fail_job(job_id, category, message[:2000])


def _complete_job(kernel: ResearchKernel, job_id: str, tool: str) -> None:
    """Best-effort job completion after admit/settle."""
    try:
        kernel.complete_job(job_id, {"tool": tool})
    except Exception:  # noqa: BLE001, S110 - terminal already recorded; nothing else to carry
        pass


async def _invoke_tool(
    tool: str,
    arguments: Mapping[str, object],
    tool_session: RuntimeToolSession,
    session_id: str,
    as_of: str | None,
    job_id: str,
) -> tuple[dict[str, object], ToolOutcome]:
    """ToolRuntime invoke + outcome mapping, off the loop's default executor."""
    call = functools.partial(
        execute_agent_tool,
        tool,
        dict(arguments),
        tool_session,
        tool_call_id=f"{tool_session.session_id}:{tool}:{uuid.uuid4().hex[:8]}",
        as_of=as_of,
        active_research_session_id=session_id,
        active_research_job_id=job_id,
    )
    pool = _SEC_POOL if source_domain_for_tool(tool) == "SEC" else _TOOL_POOL
    raw = await asyncio.get_running_loop().run_in_executor(pool, call)
    return raw, outcome_from_result(tool, raw)


def _tool_failure(tool: str, error: str) -> dict[str, JSONValue]:
    """Wire result for a call that produced no result."""
    return {"tool": tool, "ok": False, "summary": "", "evidence": [], "admitted": 0, "error": error[:500]}


async def run_tool(
    kernel: ResearchKernel,
    session_id: str,
    tool: str,
    arguments: Mapping[str, object],
    *,
    deadline_at: float | None = None,
) -> dict[str, JSONValue]:
    """Execute one research tool, admit its evidence, and report what the model reads.

    Returns {tool, ok, summary, evidence, admitted} plus ``error`` when the call failed.
    A tool or admission failure becomes ``ok: false``; a missing session or job start raises to
    the caller, which reports it as a failed call.
    """
    if tool not in RESEARCH_TOOLS:
        return _tool_failure(tool, f"{tool!r} is not a research tool")
    session = kernel.get_session(session_id)
    job_id = _start_job(kernel, session_id, tool, deadline_at)
    tool_session = RuntimeToolSession(session_id=f"research:{session_id}")
    try:
        kernel.heartbeat_job(job_id)
        result, outcome = await _invoke_tool(tool, arguments, tool_session, session_id, _session_as_of(session), job_id)
        if outcome.error is not None:
            return {**_tool_failure(tool, outcome.error), "summary": _outcome_summary(outcome)}
        try:
            ids = admit_tool_result(kernel, session_id, job_id, tool, result, outcome)
        except Exception as exc:  # a rejected admission is a call failure the model must see
            logger.warning("research admit failed sid=%s tool=%s", session_id, tool, exc_info=True)
            return _tool_failure(tool, f"admit failed: {exc}")
        rows = evidence_rows(kernel, session_id, ids)
        summary = evidence_summary(rows) or _outcome_summary(outcome)
        return {"tool": tool, "ok": True, "summary": summary, "evidence": rows, "admitted": len(rows)}
    except asyncio.CancelledError:
        _fail_job(kernel, job_id, f"cancelled during {tool}", "timeout")
        raise
    except Exception as exc:
        _fail_job(kernel, job_id, str(exc))
        logger.info("research tool failed sid=%s tool=%s err=%.200s", session_id, tool, exc, exc_info=True)
        return _tool_failure(tool, str(exc) or f"{tool} failed")
    finally:
        _complete_job(kernel, job_id, tool)


__all__ = [
    "RESEARCH_TOOLS",
    "admit_tool_result",
    "evidence_candidate",
    "evidence_rows",
    "evidence_summary",
    "research_tool_names",
    "row_datapoints",
    "run_tool",
    "shutdown_pools",
]
