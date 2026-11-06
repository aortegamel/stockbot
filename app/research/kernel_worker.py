"""Kernel worker: stdin-JSONL bridge from the TS research loop to the Python research core.

One request per line, one reply per id.

  {"id", "op": "open_research", "prompt", "deadlineAt"?, "ticker"?, "watchlist"?, "asOf"?}
    -> {"id", "sessionId", "tools": [names research_tool accepts]}
  {"id", "op": "research_tool", "sessionId", "tool", "arguments", "deadlineAt"?}
    -> {"id", "tool", "ok", "summary", "evidence": [{id, source, retrievedAt, content, ...}], "admitted", "error"?}
  {"id", "op": "prove_claims", "sessionId", "claims": [{text, claim_type, evidence_ids, datapoint_ids?}],
     "evidence"?, "deadlineAt"?}
    -> {"id", "verdicts": [{claim_type, support_answer, support, evidence_state, error?}]} (claim order)
  {"id", "op": "admit_datapoint", "sessionId", "evidenceId", metric/value/entity/period/unit|formula+inputs}
    -> {"id", "evidence_id", "datapoint"}; a calculation names formula id@version and session dp: inputs
  {"id", "op": "watchlist", "watchlist": [tickers], "deadlineAt"?} -> {"id", "watchlist": [rows]}
    (read-only page-load fan-out: no session, no evidence; same tool handlers and caches)
  {"id", "op": "arguments", "tool", "schema"?, "objective"?, "ticker"?, "node"?, "context"?}
    -> {"id", "tool", "arguments", "confidence", "reasoning"} (chat tool grounding over shared Needle)
  {"id", "op": "effort", "prompt", "ticker"?, "watchlist"?, "context"?, "deadlineAt"?,
   "maxOutputTokens"?, "answerTokenBudget"?, "maxAnswerChars"?} -> {"id", "effort"}
  {"id", "op": "sections", "prompt", "ticker"?, "watchlist"?, "context"?, "deadlineAt"?}
    -> {"id", "sections": [investor part ids JEV keeps, in answer order]}
  {"id", "op": "scope", "prompt", "ticker"?, "watchlist"?, "context"?, "deadlineAt"?}
    -> {"id", "in_scope", "answer"?}

A rejection from prove_claims carries ``error`` starting with "proof:". Any other
error means the proof gate is unavailable, never that a claim is false.
Effort, sections, and scope decisions are stored in the shared session "rs:chat".
Never raises out of the worker: failures report as an error or terminal provider_error.

Human Decision Authority (code, not prose): the objective is persisted verbatim
from the user prompt; this path only gathers research and admits citable evidence.
It never calls finalize/committee/order/portfolio side effects — research never
decides, the user does.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fcntl
import importlib
import json
import logging
import os
import select
import signal
import sys
import termios
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.child_spawn import defer_termination
from app.research.models import JSONValue, utcnow

if TYPE_CHECKING:
    from app.decision_client import JevClient
    from app.research.repository import ResearchRepository

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, dict[str, object]], None]
"""Progress callback: stage plus JSON-safe detail; listener errors propagate."""

"""One settled intake call: tool, arguments, record, elapsed milliseconds."""
"""One ticker chain: its list call plus its chained document call."""


def _notify(progress: ProgressFn | None, stage: str, detail: dict[str, object] | None = None) -> None:
    """Emit progress when a listener is present; listener errors propagate."""
    if progress is None:
        return
    progress(stage, detail or {})


# Process-lifetime shared runtime: one JevClient + one Needle worker per kernel
# process. Lazily created so import has no side effects (thesis/script callers
# use run_graph_prompt without starting anything).
_JEV: JevClient | None = None
_SHUTDOWN_DONE = False


def _shared_jev() -> JevClient:
    """Return the process-lifetime JevClient, creating it once."""
    global _JEV
    if _JEV is None:
        from app.decision_client import JevClient

        _JEV = JevClient()
    return _JEV


def _jev_client(jev: JevClient | None) -> JevClient:
    """The caller-injected client when present, else the process-lifetime shared client."""
    return jev if jev is not None else _shared_jev()


def _shared_needle_generate() -> Callable[..., object]:
    """Return the shared Needle arguments hook (no spawn here; start() owns it)."""
    needle_mod = importlib.import_module("app.needle_client")
    generate: Callable[..., object] = needle_mod.generate_arguments
    return generate


def _prewarm_caches() -> None:
    """Warm the tool registry and the SEC identity/CIK lookup.

    A warm failure only logs: the worker stays live and each request retries the work itself.
    """
    try:
        from app.research import tool_runner

        tool_runner.research_tool_names()
    except Exception as exc:
        logger.debug("prewarm registry failed: %s", exc, exc_info=True)
    try:
        from app.sec.client import _fetch_lookup_frame, ensure_identity

        ensure_identity()
        _fetch_lookup_frame("warmup")
    except Exception as exc:
        logger.debug("prewarm CIK lookup failed: %s", exc, exc_info=True)


def _prewarm() -> None:
    """Warm the caches on a daemon thread: startup never waits for network or registry work."""
    threading.Thread(target=_prewarm_caches, name="kernel-prewarm", daemon=True).start()


def _startup() -> JevClient:
    """Start shared JEV + Needle. Caller emits readiness only after this returns."""
    global _SHUTDOWN_DONE
    _SHUTDOWN_DONE = False
    jev = _shared_jev()
    jev.start()
    try:
        needle_mod = importlib.import_module("app.needle_client")
        start_fn = needle_mod.start
        start_fn()
    except Exception as exc:
        # Needle failures do not stop the worker: only the chat arguments op grounds on it.
        logger.exception("kernel-worker needle start failed")
        print(f"[kernel-worker] needle start failed: {exc}", file=sys.stderr)
    _prewarm()
    return jev


def _shutdown_pools() -> None:
    """ponytail: shutdown drops queued pool work without blocking; running calls still join at
    interpreter exit, so this bounds shutdown, not exit."""
    from app.research import tool_runner

    tool_runner.shutdown_pools()
    try:  # the chat arguments op runs Needle on the scheduler's tool pool
        from app.research import scheduler as _sched

        _sched._TOOL_POOL.shutdown(wait=False, cancel_futures=True)
    except ImportError, AttributeError, RuntimeError:
        pass


def _shutdown() -> None:
    """Close shared JEV + Needle once (EOF/signal); best-effort, idempotent."""
    global _SHUTDOWN_DONE
    if _SHUTDOWN_DONE:
        return
    _SHUTDOWN_DONE = True
    if _JEV is not None:
        try:
            _JEV.close()
        except OSError:
            pass
    try:
        needle_mod = importlib.import_module("app.needle_client")
        close_fn = needle_mod.close
        close_fn()
    except ImportError, AttributeError, OSError:
        pass
    _shutdown_pools()


def _request_id(req: Mapping[str, JSONValue]) -> str:
    """Use the protocol placeholder when the caller does not supply a string id."""
    raw = req.get("id")
    return raw if isinstance(raw, str) else "?"


def _confidence(raw: object) -> float | None:
    """Accept numeric confidence without treating booleans as numbers."""
    return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None


def _terminal(rid: str, category: str, message: str) -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {
        "id": rid,
        "objective": "",
        "evidence": [],
        "nodes": [],
        "decisions": [],
        "unresolved": [],
        "incomplete_guard": False,
        "toolExecutions": [],
        "needleDecisions": [],
        "toolCalls": [],
        "failures": {category: 1},
        "escalations": 1,
        "escalated": True,
        "error": message[:500],
        "terminal": {"category": category, "message": message[:500]},
    }
    return out


# Watchlist fan-out: one fixed-argument call per (symbol, source), all through
# the existing tool handlers so page loading and agent requests share one
# source path and its caches. SEC listings carry no forms filter, so fund forms
# (SPY/QQQ) stay in. Each row keeps its own symbol, status, date, provenance.
_WATCHLIST_SOURCES: dict[str, str] = {
    "list_sec_filings": "sec_filings",
    "get_short_interest": "finra_short_interest",
    "get_reg_sho_volume": "finra_short_sale_volume",
    "get_threshold_securities": "finra_threshold_list",
}
_WATCHLIST_MAX = 25
_WATCHLIST_SEC_LIMIT = 5
_WATCHLIST_BUDGET_S = 8.0


def _request_watchlist(req: Mapping[str, JSONValue], where: str = "<worker>") -> list[str] | None:
    """Validated watchlist context beside the verbatim prompt; None when absent, ValueError when malformed."""
    from app.research.models import validate_ticker

    raw = req.get("watchlist")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw or len(raw) > _WATCHLIST_MAX:
        raise ValueError(f"{where}: 'watchlist' must be a list of 1-{_WATCHLIST_MAX} tickers")
    out: list[str] = []
    for item in raw:
        sym = validate_ticker(item, f"{where}: 'watchlist'")
        if sym is None:
            raise ValueError(f"{where}: 'watchlist' entries must be nonblank tickers")
        if sym not in out:
            out.append(sym)
    return out


_ASK_CONTEXT_KINDS = ("news", "filing", "insider", "alert")
_ASK_CONTEXT_KEYS = frozenset({"kind", "title", "url", "source", "sym", "at"})


def _context_text(value: object) -> bool:
    """Feed-item text field: a nonblank string of at most 300 characters."""
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 300


def _context_url(value: object) -> bool:
    """Optional feed-item link: http(s), a host, no credentials, at most 2000 characters."""
    from app.research.sources import _url

    return value is None or (isinstance(value, str) and len(value) <= 2000 and bool(_url(value)))


def _request_context(req: Mapping[str, JSONValue], where: str = "<worker>") -> dict[str, JSONValue] | None:
    """Validated feed item beside the verbatim prompt (route.ts parseAskContext rules); None when absent.

    Raise ValueError when malformed. The values stay untrusted data, never instructions.
    """
    from app.research.models import validate_ticker

    raw = req.get("context")
    if raw is None:
        return None
    bad = ValueError(
        f"{where}: 'context' must be a feed item: kind news|filing|insider|alert, title and source (1-300 chars), "
        "at (max 40 chars), optional http(s) url and ticker sym"
    )
    if not isinstance(raw, dict) or not set(raw) <= _ASK_CONTEXT_KEYS or raw.get("kind") not in _ASK_CONTEXT_KINDS:
        raise bad
    at, url, sym = raw.get("at"), raw.get("url"), raw.get("sym")
    if not (_context_text(raw.get("title")) and _context_text(raw.get("source")) and _context_url(url)):
        raise bad
    if not isinstance(at, str) or len(at) > 40:
        raise bad
    ticker = validate_ticker(sym, f"{where}: 'context.sym'")
    if sym is not None and ticker is None:
        raise bad
    out: dict[str, JSONValue] = {"kind": raw["kind"], "title": raw["title"], "source": raw["source"], "at": at}
    if url is not None:
        out["url"] = url
    if ticker is not None:
        out["sym"] = ticker
    return out


def _watchlist_args(tool: str, symbol: str) -> dict[str, object]:
    """Fixed arguments; FINRA date scope stays the handler's own default."""
    if tool == "list_sec_filings":
        return {"identifier": symbol, "limit": _WATCHLIST_SEC_LIMIT}
    return {"ticker": symbol}


def _source_error_text(text: str) -> str:
    """The failing source's own lines, without the shared unavailable-data frame."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith(("The requested data is unavailable", "Next step:", "Tool: "))
    ]
    return " ".join(lines)[:300]


def _source_failure(record: Mapping[str, object], text: str) -> dict[str, JSONValue]:
    """Failure row fields: timeout, a source-reported no_records answer, or error, with the source's own text."""
    if record.get("error_type") == "timeout":
        status = "timeout"
    elif "No data found for " in text:
        status = "no_records"
    else:
        status = "error"
    return {"status": status, "error": _source_error_text(text) or text[:300]}


def _source_metadata(result: Mapping[str, object]) -> dict[str, JSONValue]:
    """Status, scalar metadata, and string provenance refs from a successful result's safe meta."""
    raw_meta = result.get("meta")
    meta: Mapping[str, object] = raw_meta if isinstance(raw_meta, Mapping) else {}
    row: dict[str, JSONValue] = {"status": "no_records" if meta.get("returned_count") == 0 else "ok"}
    for key in ("as_of", "returned_count", "truncated", "total"):
        val = meta.get(key)
        if isinstance(val, (str, int, float, bool)):
            row[key] = val
    refs = meta.get("source_refs")
    if isinstance(refs, Mapping):
        ref: dict[str, JSONValue] = {
            k: v for k in ("record_id", "uri", "known_at") if isinstance(v := refs.get(k), str)
        }
        if ref:
            row["ref"] = ref
    return row


def _source_outcome(symbol: str, tool: str, record: Mapping[str, object]) -> dict[str, JSONValue]:
    """One (symbol, source) outcome: ok, no_records, error, or timeout. A failure never reads as no records."""
    row: dict[str, JSONValue] = {"symbol": symbol, "source": _WATCHLIST_SOURCES[tool], "tool": tool}
    result = record.get("result")
    error = result.get("error") if isinstance(result, Mapping) and result.get("error") else record.get("error")
    if error or not isinstance(result, Mapping):
        # Failure rows carry no evidence_id: only an answered source is citable.
        return {**row, **_source_failure(record, str(error or "no result"))}
    row.update(_source_metadata(result))
    eid = record.get("evidence_id")
    if isinstance(eid, str) and eid:
        row["evidence_id"] = eid
    return row


async def _watchlist_fetch(symbols: list[str], budget_s: float) -> list[dict[str, JSONValue]]:
    """Read-only page-load fan-out: no session, no evidence, every call overlaps on the existing pools.

    Calls past the budget are cancelled and reported as timeout rows, never dropped.
    """
    from app.research import tool_runner
    from app.tool_runtime import RuntimeToolSession

    session = RuntimeToolSession(session_id="watchlist")

    async def _one(symbol: str, tool: str) -> dict[str, object]:
        # Empty session/job ids keep the gateway unstaged: nothing persists.
        result, _outcome = await tool_runner._invoke_tool(
            tool,
            _watchlist_args(tool, symbol),
            session,
            "",
            None,
            "",
        )
        return {"result": result}

    keys = [(symbol, tool) for symbol in symbols for tool in _WATCHLIST_SOURCES]
    tasks = [asyncio.ensure_future(_one(symbol, tool)) for symbol, tool in keys]
    done, _pending = await asyncio.wait(tasks, timeout=max(budget_s, 0.0))
    rows: list[dict[str, JSONValue]] = []
    for (symbol, tool), task in zip(keys, tasks):
        record: Mapping[str, object]
        if task not in done:
            task.cancel()
            record = {"error": f"no result within {budget_s:g}s", "error_type": "timeout"}
        elif task.exception() is not None:
            record = {"error": f"{type(task.exception()).__name__}: {task.exception()}"}
        else:
            record = task.result()
        rows.append(_source_outcome(symbol, tool, record))
    return rows


def _watchlist(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """op:watchlist — page-load fan-out under the intake budget and the request deadline."""
    rid = _request_id(req)
    try:
        symbols = _request_watchlist(req)
    except ValueError as exc:
        return {"id": rid, "error": str(exc)[:500]}
    if symbols is None:
        return {"id": rid, "error": "<worker>: 'watchlist' required"}
    budget = _WATCHLIST_BUDGET_S
    deadline_ms = _deadline_field(req.get("deadlineAt"))
    if deadline_ms is not None:
        budget = max(min(budget, deadline_ms / 1000.0 - time.time()), 0.0)
    rows: list[JSONValue] = [*asyncio.run(_watchlist_fetch(symbols, budget))]
    return {"id": rid, "watchlist": rows}


def _close_bootstrap_job(sid: str) -> None:
    """Cancel one open bootstrap source job. Log cleanup errors without replacing the run outcome."""
    try:
        from app.research import service as _svc
        from app.research.repository import ResearchRepository as _Repo

        repo = _Repo()
        for job in repo.list_jobs(sid):
            if job.job_type == "source_agent" and job.status in ("queued", "running"):
                _svc.cancel_job(job.job_id)
                break
    except Exception:
        logger.exception("kernel bootstrap cleanup failed sid=%s", sid)


# ponytail: shape, not lookup: US symbols are 1-5 letters plus an optional share class
# (BRK.B, BF-B). A short private name (XAI) still reads as a ticker; add a lookup if that matters.


def _registry_portfolio_hit() -> list[str]:
    """Fail-closed HDA: portfolio/order tools must not be runnable from research."""
    from app.research import scheduler

    try:
        reg = scheduler.build_registry()
    except Exception as exc:
        raise RuntimeError(f"registry guard forbids portfolio tools: registry unavailable ({exc})") from exc
    from app.tools import PORTFOLIO_AUTHORIZED_TOOLS as _PORT

    forbidden: set[str] = set(_PORT)
    names: set[str] = set()
    if isinstance(reg, list):
        for entry in reg:
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
    return sorted(n for n in names if n in forbidden)


def _jev_subject(sid: str, objective: str, jev: JevClient | None) -> str | None:
    """First subject JEV keeps for a prompt the rules cannot resolve; None on no keep, no client, or failure."""
    from app.research import scheduler

    if jev is None:
        return None
    candidates = scheduler._objective_subject_candidates(objective)
    if not candidates:
        return None
    try:
        kept = asyncio.run(jev.select_subjects(objective, candidates, session_id=sid))
    except Exception as exc:  # noqa: BLE001 - subject grounding is best effort; the run continues without one
        logger.warning("subject selection failed sid=%s: %s", sid, exc)
        return None
    return kept[0] if kept else None


def _stamp_session_ticker(sid: str, ticker: str) -> None:
    """Store the JEV-kept subject on the already-created session."""
    from dataclasses import replace

    from app.research.repository import ResearchRepository

    repo = ResearchRepository()
    repo.save_session(replace(repo.get_session(sid), ticker=ticker, updated_at=utcnow()))


def _graph_node_questions(objective: str) -> list[tuple[str, str]]:
    """(question, why) per root node: one per named company, else one per named data family, else the prompt."""
    from app.research import scheduler

    # A comparison researches each named company in its own node, so every company's filings are read.
    subjects = scheduler.question_subjects(objective)
    if len(subjects) >= 2:
        return [(f"{objective} ({ticker} only)", f"Research {ticker} for this comparison.") for ticker in subjects]
    # Several named data families split too: one node gathers one family instead of every family in one node.
    families = scheduler.question_source_families(objective)
    if len(families) >= 2:
        return [
            (
                f"{objective} ({family} only)",
                f"Gather the {family} evidence for this question; other families have their own node.",
            )
            for family in families
        ]
    return [(objective, "Route question.")]


def run_graph_prompt(
    prompt: str,
    as_of: str | None = None,
    jev: JevClient | None = None,
    progress: ProgressFn | None = None,
    setup_deadline: float | None = None,
    ticker: str | None = None,
    watchlist: list[str] | None = None,
    context: Mapping[str, JSONValue] | None = None,
) -> str:
    """Create a session and its root nodes, with optional JEV subject selection."""
    from app.research import scheduler, service

    hit = _registry_portfolio_hit()
    if hit:
        raise RuntimeError(f"registry guard forbids portfolio tools: {hit}")
    objective = prompt.strip()
    sid = service.create_research(
        objective, objective, as_of=as_of, policy={"research_sources": {"mode": "all", "sources": []}}, ticker=ticker
    )
    _notify(progress, "session", {"session_id": sid})
    try:
        if watchlist or context:
            service.record_decision(
                sid,
                "request_context",
                {},
                {},
                {"watchlist": watchlist, "context": context},
                request={"objective": objective},
            )
        # No rule resolved a subject and no caller supplied one: JEV picks the company the prompt is about,
        # so tools receive a grounded ticker instead of one Needle invents.
        if ticker is None and not watchlist and scheduler._objective_subject_ticker(objective, ticker) is None:
            subject = _jev_subject(sid, objective, jev)
            if subject is not None:
                _stamp_session_ticker(sid, subject)
        # A comparison researches each named company in its own node, and a multi-family ask one family per
        # node, so every named source is read instead of one node spreading across all of them.
        for question, why in _graph_node_questions(objective):
            service.create_node(sid, question, why)
        return sid
    except Exception:
        _close_bootstrap_job(sid)
        raise


def _request_ticker(req: Mapping[str, JSONValue], where: str = "<worker>") -> str | None:
    """Validated ticker channel beside the verbatim prompt; None when absent."""
    from app.research.models import validate_ticker as _validate

    return _validate(req.get("ticker"), f"{where}: 'ticker'") if req.get("ticker") is not None else None


def _needle_request_context(context: JSONValue, tool: str, objective: str, ticker: str | None) -> dict[str, JSONValue]:
    """Approved Needle context only: as_of/temporal_scope/today_utc/ticker/grounded_arguments.

    Raw evidence and attempts never cross; grounded ids derive outside Needle
    from the repaired scheduler seed while full bytes stay durable for JEV.
    """
    from datetime import UTC, datetime

    from app.research import scheduler as _sched

    scoped: dict[str, JSONValue] = {}
    if isinstance(context, dict):
        raw_scope = context.get("temporal_scope")
        if isinstance(raw_scope, dict):
            scoped = _sched._needle_scope(raw_scope)
        raw_as_of = context.get("as_of")
        as_of = raw_as_of.strip() if isinstance(raw_as_of, str) and raw_as_of.strip() else None
    else:
        as_of = None
    schema = _sched._schema_for(tool, _sched.build_registry())
    out: dict[str, JSONValue] = {
        "as_of": as_of,
        "temporal_scope": scoped,
        "today_utc": datetime.now(UTC).date().isoformat(),
        "grounded_arguments": _sched._grounded_execution_values(tool, schema, objective, ticker, as_of=as_of),
    }
    resolved = _sched._objective_subject_ticker(objective, ticker)
    if isinstance(resolved, str) and resolved.strip():
        out["ticker"] = resolved.strip().upper()
    return out


def _arguments_session_hint(req: Mapping[str, JSONValue]) -> str | None:
    """Caller-carried session id: arguments.session_id, else node.session_id, else None."""
    for key in ("arguments", "node"):
        holder = req.get(key)
        if not isinstance(holder, Mapping):
            continue
        sid = holder.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return sid
    return None


def _arguments_carried(rid: str, tool: str, sid_hint: str | None, raw_args_hint: object) -> dict[str, JSONValue] | None:
    """Calls that carry caller ids instead of grounding; None sends the tool to Needle.

    ponytail: resume/status/cancel carry only a session_id — the prompt never grounds,
    so shortcut without Needle (docs: every argument is a span of input; a bare question
    has no sid span). research_read_search carries session_id plus its search_id.
    """
    if tool in ("research_resume", "research_status", "research_cancel") and sid_hint:
        logger.info("toolflow arguments rid=%s sid=%s tool=%s conf=yes shortcut=sid-carry", rid, sid_hint, tool)
        return {
            "id": rid,
            "tool": tool,
            "arguments": {"session_id": sid_hint.strip()},
            "confidence": 1.0,
            "reasoning": "session_id carried from caller; no Needle grounding needed",
        }
    if tool != "research_read_search":
        return None
    search_hint = raw_args_hint.get("search_id") if isinstance(raw_args_hint, Mapping) else None
    if sid_hint and isinstance(search_hint, str) and search_hint.strip():
        sid_clean = sid_hint.strip()
        search_clean = search_hint.strip()
        logger.info("toolflow arguments rid=%s sid=%s tool=%s conf=yes shortcut=sid-carry", rid, sid_clean, tool)
        return {
            "id": rid,
            "tool": tool,
            "arguments": {"session_id": sid_clean, "search_id": search_clean},
            "confidence": 1.0,
            "reasoning": "session_id+search_id carried from caller; no Needle grounding needed",
        }
    logger.warning(
        "toolflow arguments_deny rid=%s sid=%s tool=%s reason=search-id-required", rid, sid_hint or "-", tool
    )
    return {"id": rid, "error": "research_read_search needs search_id from a prior search_sec_filings result"}


def _arguments_objective(req: Mapping[str, JSONValue]) -> str:
    """Grounding prompt: the explicit objective, else the request prompt, else empty."""
    objective = req.get("objective")
    if objective is None:
        objective = req.get("prompt")
    return objective if isinstance(objective, str) else ""


def _arguments_schema(req: Mapping[str, JSONValue], tool: str) -> JSONValue:
    """Inline schema when the caller supplies one, else the scheduler registry schema for the tool."""
    from app.research.models import validate_json_value

    schema_raw = req.get("schema")
    schema = validate_json_value(schema_raw if schema_raw is not None else {}, "<arguments>: 'schema'")
    if isinstance(schema, dict) and schema:
        return schema
    from app.research import scheduler

    return scheduler._schema_for(tool, scheduler.build_registry())


def _arguments_request_value(req: Mapping[str, JSONValue], key: str) -> JSONValue:
    """Validated optional request field; an absent field stays None."""
    from app.research.models import validate_json_value

    raw = req.get(key)
    return validate_json_value(raw, f"<arguments>: '{key}'") if raw is not None else None


def _arguments_withhold(rid: str, tool: str, prompt: str, ticker: str | None = None) -> dict[str, JSONValue]:
    """Seed a withheld call from the explicit subject, else the supplied chart ticker."""
    from app.research import scheduler as _sched

    seeded = _sched._fallback_sec_args(tool, prompt, ticker)
    if not isinstance(seeded, dict):
        logger.warning("toolflow arguments_fail_open rid=%s tool=%s err_type=NeedleWithhold", rid, tool)
        return {"id": rid, "error": "needle arguments failed"}
    logger.info("toolflow arguments rid=%s tool=%s conf=no fallback=seeded", rid, tool)
    return {
        "id": rid,
        "tool": tool,
        "arguments": seeded,
        "confidence": None,
        "reasoning": "seeded after Needle withhold",
    }


def _arguments_generate(rid: str, sid_display: str, tool: str, req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Ground the objective on the warmed Needle and repair its arguments; never raises."""
    prompt = _arguments_objective(req)
    schema = _arguments_schema(req, tool)
    node = _arguments_request_value(req, "node")
    context = _arguments_request_value(req, "context")
    try:
        arg_ticker = _request_ticker(req)
    except ValueError as exc:
        return {"id": rid, "error": str(exc)[:500] or "bad ticker"}
    context = _needle_request_context(context, tool, prompt, arg_ticker)
    from app.research.scheduler import _TOOL_POOL

    async def generate() -> object:
        return await asyncio.get_running_loop().run_in_executor(
            _TOOL_POOL,
            lambda: _shared_needle_generate()(tool=tool, schema=schema, objective=prompt, node=node, context=context),
        )

    generated = _request_call(req, "arguments", generate())
    if generated is None:
        return _arguments_withhold(rid, tool, prompt, arg_ticker)
    return _arguments_result(rid, sid_display, tool, generated, prompt)


def _arguments_result(rid: str, sid_display: str, tool: str, generated: object, prompt: str) -> dict[str, JSONValue]:
    """Grounded arguments for the JEV-selected tool, with Needle's own confidence and reasoning."""
    from app.needle_client import validate_needle_tool
    from app.research.models import validate_json_mapping
    from app.research.scheduler import _repair_tool_arguments as _kw_repair

    needle_tool: object = generated.get("tool", tool) if isinstance(generated, dict) else tool
    validate_needle_tool(tool, needle_tool)
    raw_args: object = generated.get("arguments", {}) if isinstance(generated, dict) else {}
    repaired = _kw_repair(tool, dict(raw_args) if isinstance(raw_args, dict) else {}, prompt, "-", "-")
    arguments = validate_json_mapping(repaired, "<arguments>: 'arguments'")
    out: dict[str, JSONValue] = {"id": rid, "tool": tool, "arguments": arguments}
    if isinstance(generated, dict):
        out["confidence"] = _confidence(generated.get("confidence"))
        reasoning = generated.get("reasoning")
        out["reasoning"] = reasoning if isinstance(reasoning, str) else ""
    logger.info(
        "toolflow arguments rid=%s sid=%s tool=%s conf=%s",
        rid,
        sid_display,
        tool,
        "yes" if isinstance(out.get("confidence"), float) else "no",
    )
    return out


def _arguments(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Shared-Needle arguments op: grounded args for the exact JEV-selected tool.

    Runs on the already-warmed kernel-owned Needle (no second spawn). Never
    raises: failures report as {"id", "error"} and the TS caller fail-opens to
    research without invoking the tool. Accepts schema inline, else falls back
    to the scheduler registry schema for the named tool.
    """
    rid = _request_id(req)
    tool = req.get("tool")
    if not isinstance(tool, str) or not tool:
        logger.warning("toolflow arguments_deny rid=%s reason=tool-required", rid)
        return {"id": rid, "error": "tool required"}
    try:
        sid_hint = _arguments_session_hint(req)
        sid_display = sid_hint or "-"
        logger.info("toolflow arguments_entry rid=%s sid=%s tool=%s", rid, sid_display, tool)
        carried = _arguments_carried(rid, tool, sid_hint, req.get("arguments"))
        if carried is not None:
            return carried
        return _arguments_generate(rid, sid_display, tool, req)
    except Exception as exc:
        logger.warning(
            "toolflow arguments_fail_open rid=%s tool=%s err_type=%s", rid, tool, type(exc).__name__, exc_info=True
        )
        return {"id": rid, "error": str(exc)[:500] or "needle arguments failed"}


def _final_ids(item: object, key: str) -> list[str]:
    """Non-empty string id list under key; [] when absent or malformed."""
    if not isinstance(item, Mapping):
        return []
    raw = item.get(key)
    if not isinstance(raw, list):
        return []
    return [eid for eid in raw if isinstance(eid, str) and eid.strip()]


def _final_session(req: Mapping[str, JSONValue]) -> str | None:
    """Caller session id: sessionId (camelCase, matches run op) with session_id alias; None when absent."""
    for key in ("sessionId", "session_id"):
        raw = req.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _deadline_field(raw: object) -> float | None:
    """Epoch-milliseconds deadline field: a positive number that is not a bool, else None."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    # Millisecond truncation can make an immediate deadline look expired; keep sub-millisecond slack.
    return float(raw) + 0.5


def _request_call[T](req: Mapping[str, JSONValue], stage: str, call: Coroutine[object, object, T]) -> T:
    """Bound one asynchronous operation by the original request deadline. Cancellation stays local to this call."""
    deadline = _deadline_field(req.get("deadlineAt"))

    async def bounded() -> T:
        loop = asyncio.get_running_loop()
        now_wall = time.time()
        if deadline is not None and deadline / 1000.0 <= now_wall:
            task = asyncio.ensure_future(call)
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError, TimeoutError:
                pass
            raise TimeoutError(f"deadline reached during {stage}")
        try:
            timeout = None if deadline is None else max(deadline / 1000.0 - now_wall, 0.0)
            async with asyncio.timeout_at(loop.time() + timeout if timeout is not None else None):
                return await call
        except TimeoutError as exc:
            raise TimeoutError(f"deadline reached during {stage}") from exc

    return asyncio.run(bounded())


def _calculation_declaration(
    req: Mapping[str, JSONValue],
    item: dict[str, object],
    formula: object,
    rid: str,
    sid: str,
    store: ResearchRepository,
) -> dict[str, object]:
    """Resolve calculation inputs through the same admission and session ownership guards."""
    from app.research import service, tool_runner

    inputs = req.get("inputs")
    formula_id, _, version = formula.partition("@") if isinstance(formula, str) else ("", "", "")
    if not formula_id or not version or not isinstance(inputs, list) or not inputs:
        raise ValueError("a calculation needs formula id@version and a list of inputs")
    ids: list[str] = []
    for entry in inputs:
        if isinstance(entry, dict) and "formula" not in entry:
            # Admit inline observations before the calculation, through the same source guards.
            sub = _admit_datapoint({**entry, "id": rid, "sessionId": sid, "evidenceId": entry.get("evidence_id")})
            point = sub.get("datapoint")
            if not isinstance(point, dict):
                raise ValueError(f"input {entry.get('metric')!r} {entry.get('value')!r}: {sub.get('error')}")
            ids.append(str(point["id"]))
        elif isinstance(entry, str):
            ids.append(entry)
        else:
            raise ValueError("each input is a dp: id or an observed number declaration")
    owners = {
        str(point["id"]): eid
        for eid, rec in service._session_rows(store, sid).items()
        for point in tool_runner.row_datapoints(rec)
        if isinstance(point, dict)
    }
    if missing := [dp for dp in ids if dp not in owners]:
        raise ValueError(f"inputs {missing} are not datapoints of this session")
    calculation = {key: item[key] for key in ("subject", "metric", "value", "entity", "period", "unit") if key in item}
    calculation.update(
        claim_kind="calculation",
        formula_id=formula_id,
        formula_version=version,
        input_evidence_ids=[owners[dp] for dp in ids],
    )
    return calculation


def _datapoint_source(item: dict[str, object], row: Mapping[str, object], prov: Mapping[str, object]) -> None:
    """Bind an observed declaration to its stored source, without constructing source identities."""
    if prov.get("kind") == "sec_source":
        metadata = row.get("metadata")
        handle = metadata.get("source_handle") if isinstance(metadata, dict) else None
        if not isinstance(handle, dict):
            raise ValueError("Cannot admit a filtered document view without its stored canonical source_handle")
        item["source_handle"] = handle
        item["matching_passage"] = prov.get("passage")
    elif prov.get("kind") in ("sec_record", "finra_record", "web_source"):
        item["tool_result_id"] = prov.get("tool_result_id")
        item["record_identity"] = prov.get("record_identity")
        item["matching_passage"] = prov.get("record_identity") or prov.get("excerpt")
        if prov.get("kind") == "web_source":
            item["source_record_id"] = prov.get("url")
    else:
        raise ValueError("unsupported evidence provenance")


def _admit_datapoint(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Admit one declared number through the existing source and proof guards.

    A declaration with ``formula`` (``id@version``) and ``inputs`` (session dp: ids in
    formula argument order) is a calculation: the ledger recomputes it from the input rows.
    """
    from app.research import service, tool_runner
    from app.research.evidence import evidence_domain
    from app.research.repository import ResearchRepository

    rid = _request_id(req)
    try:
        sid, eid = req.get("sessionId"), req.get("evidenceId")
        if not isinstance(sid, str) or not isinstance(eid, str):
            raise TypeError("sessionId and evidenceId required")
        store = ResearchRepository()
        row = store.get_evidence(eid)
        if row.get("session_id") != sid:
            raise ValueError("evidence belongs to another session")
        item: dict[str, object] = {
            key: row[key] for key in ("content", "claim_text", "subject", "source_type", "source_name") if key in row
        }
        for key in ("metric", "value", "entity", "period"):
            value = req.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")
            # Statement tables put "$" in its own cell; the value token never carries it. Proof still checks the bytes.
            item[key] = value.removeprefix("$") if key == "value" else value
        if "unit" in req:
            unit = req["unit"]
            if not isinstance(unit, str) or not unit.strip():
                raise ValueError("unit must be a non-empty string")
            item["unit"] = unit
        formula = req.get("formula")
        prov = row.get("provenance")
        if not isinstance(prov, dict):
            raise TypeError("evidence provenance required")
        if formula is not None:
            item = _calculation_declaration(req, item, formula, rid, sid, store)
        else:
            _datapoint_source(item, row, prov)
        session = store.get_session(sid)
        job = service.start_job(
            sid,
            source=evidence_domain(prov),
            wave_id=max(session.current_wave, len(session.freeze_ids) + 1, 1),
            repo=store,
        )
        jid = str(job["job_id"])
        try:
            admitted = service.record_evidence(sid, jid, item, repo=store)
        finally:
            service.complete_job(jid, repo=store)
        new_id = admitted.get("evidence_id")
        if not isinstance(new_id, str):
            raise TypeError("admission returned no evidence id")
        points = tool_runner.row_datapoints(store.get_evidence(new_id))
        if not points:
            raise ValueError("admission returned no verified datapoint")
        point = points[0]
        # The stored descriptor keeps the source spelling; a dedupe hit with another entity or period still differs.
        # A scale unit may be declared singular ("million") for the source's "millions". Inner spacing never differs
        # ("June 27, 2026" declares the source's "June 27,2026"); a padded declaration still does.
        if not isinstance(point, dict) or any(
            str(item[key]) != str(item[key]).strip()
            or "".join(str(point.get(key)).split()).casefold()
            not in (declared := "".join(str(item[key]).split()).casefold(), f"{declared}s" * (key == "unit"))
            for key in ("metric", "value", "entity", "period", "unit")
            if key in item
        ):
            raise ValueError("admitted datapoint differs from the declared number")
        return {"id": rid, "evidence_id": new_id, "datapoint": point}
    except Exception as exc:  # never raise out of the worker
        logger.info("datapoint admission failed rid=%s: %s", rid, exc, exc_info=True)
        return {"id": rid, "error": str(exc)}


def _run_scope(
    rid: str, req: Mapping[str, JSONValue]
) -> tuple[str | None, list[str] | None, dict[str, JSONValue] | None, dict[str, JSONValue] | None]:
    """(ticker, watchlist, feed item, terminal): a malformed scope is invalid_params before any setup."""
    try:
        return _request_ticker(req), _request_watchlist(req), _request_context(req), None
    except ValueError as exc:
        return None, None, None, _terminal(rid, "invalid_params", str(exc)[:500] or "bad ticker")


def _effort(req: Mapping[str, JSONValue], jev: JevClient | None = None) -> dict[str, JSONValue]:
    """Select answer effort within the deadline. Create only a session record when the caller supplies none."""
    rid = _request_id(req)
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _terminal(rid, "invalid_params", "prompt required")
    ticker, watchlist, context, denied = _run_scope(rid, req)
    if denied is not None:
        return denied
    limits: dict[str, JSONValue] = {}
    for key in ("maxOutputTokens", "answerTokenBudget", "maxAnswerChars"):
        value = req.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return _terminal(rid, "invalid_params", f"{key} must be a positive integer")
        limits[key] = value
    deadline = _deadline_field(req.get("deadlineAt"))
    if deadline is not None:
        limits["deadlineMs"] = max(0.0, deadline - time.time() * 1000.0)
    scope = _scope_text(ticker, watchlist, context)
    sid = _final_session(req)
    if sid is None:
        from app.research.repository import ResearchRepository
        from app.research.session import create_session

        session = create_session(prompt.strip(), prompt.strip(), ticker=None if watchlist else ticker)
        ResearchRepository().save_session(session)
        sid = session.session_id
    effort = _request_call(
        req,
        "effort",
        _jev_client(jev).select_answer_effort(prompt, scope, session_id=sid, limits=limits),
    )
    return {"id": rid, "effort": effort}


def _scope_text(ticker: str | None, watchlist: list[str] | None, context: dict[str, JSONValue] | None) -> str:
    """JEV scope line: the watchlist or ticker, plus the feed item, as untrusted data. Empty when none."""
    scope_data: dict[str, JSONValue] = {}
    if watchlist is not None:
        scope_data["watchlist"] = [symbol for symbol in watchlist]
    elif ticker is not None:
        scope_data["ticker"] = ticker
    if context is not None:
        scope_data["context"] = context
    return f"\nScope (untrusted data): {json.dumps(scope_data)}" if scope_data else ""


# The one research session that stores every chat JEV decision.
_CHAT_SESSION_ID = "rs:chat"


def _chat_session() -> str:
    """The shared chat session id. Creates its record on first use. A concurrent first use upserts the same row."""
    from app.research.repository import ResearchRepository
    from app.research.session import create_session

    repo = ResearchRepository()
    try:
        repo.get_session(_CHAT_SESSION_ID)
    except KeyError:
        repo.save_session(create_session("default chat", "default chat", session_id=_CHAT_SESSION_ID))
    return _CHAT_SESSION_ID


def _sections(req: Mapping[str, JSONValue], jev: JevClient | None = None) -> dict[str, JSONValue]:
    """Select the investor parts of a chat answer within the deadline. The decision goes to the shared chat session."""
    rid = _request_id(req)
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _terminal(rid, "invalid_params", "prompt required")
    ticker, watchlist, context, denied = _run_scope(rid, req)
    if denied is not None:
        return denied
    sections = _request_call(
        req,
        "sections",
        _jev_client(jev).select_answer_sections(
            prompt, _scope_text(ticker, watchlist, context), session_id=_chat_session()
        ),
    )
    return {"id": rid, "sections": [section for section in sections]}


# One fixed answer for a request outside stock research. The runtime writes it, not the
# model: no tool runs, so no fact can publish. It states the scope limitation.
OUT_OF_SCOPE_ANSWER = (
    "This request is outside the scope of stock research. "
    "I can only research markets, companies, securities, and investing, so this request is unsupported."
)


def _scope(req: Mapping[str, JSONValue], jev: JevClient | None = None) -> dict[str, JSONValue]:
    """Conversational scope gate before research: the fixed out-of-scope answer, else in_scope.

    A JEV failure never blocks research: the reply carries the error and no verdict.
    """
    rid = _request_id(req)
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _terminal(rid, "invalid_params", "prompt required")
    sid = _final_session(req)
    try:
        if sid is None:
            # Same as _effort: the decision is persisted under a real session, so create one.
            from app.research.repository import ResearchRepository
            from app.research.session import create_session

            session = create_session(prompt.strip(), prompt.strip())
            ResearchRepository().save_session(session)
            sid = session.session_id
        in_scope = _request_call(req, "scope", _jev_client(jev).request_in_scope(prompt, session_id=sid))
    except Exception as exc:  # noqa: BLE001 - a scope outage never blocks research
        logger.warning("kernel scope gate unavailable rid=%s: %s", rid, exc)
        return {"id": rid, "error": f"scope gate failed: {exc}"[:500]}
    if in_scope:
        return {"id": rid, "in_scope": True}
    logger.info("kernel scope gate rid=%s out-of-scope", rid)
    return {"id": rid, "in_scope": False, "answer": OUT_OF_SCOPE_ANSWER}


_CLAIM_FIELDS = ("text", "claim_type", "evidence_ids", "datapoint_ids")
"""Claim fields the ledger proof reads; every other claim key never crosses."""


def _open_research(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """op:open_research — create the research session for this request; returns its id and tools.

    Scope handling matches the retired run op: a watchlist replaces the ticker, and
    asOf sets the point-in-time cutoff. ``tools`` is exactly the set research_tool
    accepts. No node, tool, or model call runs here.
    """
    from app.research import service, tool_runner

    rid = _request_id(req)
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _terminal(rid, "invalid_params", "prompt required")
    ticker, watchlist, _context, denied = _run_scope(rid, req)
    if denied is not None:
        return denied
    raw_as_of = req.get("asOf")
    objective = prompt.strip()
    try:
        if hit := _registry_portfolio_hit():
            raise RuntimeError(f"registry guard forbids portfolio tools: {hit}")
        sid = service.create_research(
            objective,
            objective,
            as_of=raw_as_of if isinstance(raw_as_of, str) else None,
            policy={"research_sources": {"mode": "all", "sources": []}},
            ticker=None if watchlist else ticker,
        )
    except Exception as exc:
        logger.exception("kernel open_research failed rid=%s", rid)
        return _terminal(rid, "provider_error", f"session setup failed: {exc}")
    tools: list[JSONValue] = [*sorted(tool_runner.RESEARCH_TOOLS)]
    return {"id": rid, "sessionId": sid, "tools": tools}


def _tool_failure(tool: str, error: str) -> dict[str, JSONValue]:
    """Fail-closed research_tool result: a failed call yields no evidence row."""
    return {"tool": tool, "ok": False, "summary": "", "evidence": [], "admitted": 0, "error": error[:500]}


def _research_tool(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """op:research_tool — run one research tool, admit its evidence, report what the model reads.

    The reply is {tool, ok, summary, evidence, admitted} plus ``error`` when the call
    failed. A tool or admission failure is ok:false; the model never sees a raise.
    """
    from app.research import tool_runner

    rid = _request_id(req)
    sid = _final_session(req)
    raw_tool = req.get("tool")
    tool = raw_tool.strip() if isinstance(raw_tool, str) else ""
    arguments = req.get("arguments")
    if sid is None or not tool:
        return {"id": rid, "error": "sessionId and tool required"}
    if arguments is not None and not isinstance(arguments, Mapping):
        return {"id": rid, "error": "arguments must be an object"}
    call = tool_runner.run_tool(tool_runner.Kernel(), sid, tool, arguments if isinstance(arguments, Mapping) else {})
    try:
        out = _request_call(req, "research_tool", call)
    except Exception as exc:  # never raise out of the worker: the model reads the failure
        logger.info("research tool unavailable rid=%s tool=%s err=%.200s", rid, tool, exc, exc_info=True)
        return {"id": rid, **_tool_failure(tool, str(exc) or f"{tool} failed")}
    return {"id": rid, **out}


def _claim_verdict(claim_type: str, error: str) -> dict[str, JSONValue]:
    """Fail-closed claim verdict: a ``proof:`` rejection, or an unavailable proof gate."""
    rejection = error.startswith("proof:")
    return {
        "claim_type": claim_type,
        "support_answer": "no",
        "support": 0.0 if rejection else None,
        "evidence_state": "insufficient",
        "error": error[:500] or "claim proof failed",
    }


def _claim_type_of(item: object) -> str:
    """Declared claim type of one wire claim, or "" when absent."""
    raw = item.get("claim_type") if isinstance(item, Mapping) else None
    return raw.strip() if isinstance(raw, str) else ""


def _prove_claim(store: ResearchRepository, sid: str, item: object, cited: Sequence[str]) -> dict[str, JSONValue]:
    """One claim's deterministic verdict: the ledger alone decides it, no model runs."""
    from app.research import service
    from app.research.agents import parse_grounded_claims

    claim_type = _claim_type_of(item)
    if not isinstance(item, Mapping):
        return _claim_verdict(claim_type, "proof: claim must be an object")
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return _claim_verdict(claim_type, "proof: text required")
    if not claim_type:
        return _claim_verdict(claim_type, "proof: claim_type required")
    row = {key: item[key] for key in _CLAIM_FIELDS if item.get(key) is not None}
    try:
        claims = parse_grounded_claims(json.dumps([row]), frozen=cited)
        service.require_verified_claims(store, sid, claims, "prove_claims")
    except ValueError as exc:
        return _claim_verdict(claim_type, f"proof: {exc}")
    return {
        "claim_type": claim_type,
        "support_answer": "yes",
        "support": 1.0,
        "evidence_state": "sufficient_support",
    }


def _prove_unavailable(rid: str, message: str, claims: Sequence[object]) -> dict[str, JSONValue]:
    """Every claim of an unavailable proof carries the same non-``proof:`` gate error."""
    return {"id": rid, "verdicts": [_claim_verdict(_claim_type_of(item), message) for item in claims]}


def _prove_claims(req: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """op:prove_claims — deterministic ledger proof per claim, verdicts index-aligned.

    A rejection carries ``error`` starting with ``proof:``. Any other failure means the
    proof gate is unavailable, never that a claim is false.
    """
    from app.research.repository import ResearchRepository

    rid = _request_id(req)
    sid = _final_session(req)
    raw = req.get("claims")
    if sid is None or not isinstance(raw, list):
        return {"id": rid, "error": "sessionId and claims required"}
    cited = [
        eid
        for item in raw
        if isinstance(item, Mapping) and isinstance(item.get("evidence_ids"), list)
        for eid in item["evidence_ids"]
        if isinstance(eid, str)
    ]
    store = ResearchRepository()
    try:
        store.get_session(sid)
        verdicts: list[JSONValue] = [_prove_claim(store, sid, item, cited) for item in raw]
    except Exception as exc:
        logger.warning("prove_claims unavailable rid=%s sid=%s err=%.200s", rid, sid, exc, exc_info=True)
        return {"id": rid, "error": f"proof unavailable: {exc}"[:500]}
    logger.info("toolflow prove_claims rid=%s sid=%s claims=%d", rid, sid, len(raw))
    return {"id": rid, "verdicts": verdicts}


_KERNEL_OPS = (
    "open_research",
    "research_tool",
    "prove_claims",
    "watchlist",
    "effort",
    "sections",
    "scope",
    "arguments",
    "admit_datapoint",
)
"""Every op this worker accepts, in the order the invalid-op message lists them."""

_GATE_OPS = frozenset({"research_tool", "prove_claims", "admit_datapoint"})
"""Ops that block on a tool, the ledger, or a model call: a worker thread keeps the stdin loop reading."""


def _configure_logging() -> None:
    """Log level from STOCKBOT_LOG_LEVEL; an unknown name falls back to WARNING."""
    level_map: dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level_name = os.environ.get("STOCKBOT_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level_map.get(level_name, logging.WARNING)
    )


def _request_or_response(line: str) -> tuple[Mapping[str, JSONValue] | None, dict[str, JSONValue] | None]:
    """One stdin line: (request, response). A blank line reports nothing to answer."""
    stripped = line.strip()
    if not stripped:
        return None, None
    try:
        parsed: object = json.loads(stripped)
    except ValueError:
        return None, _terminal("?", "invalid_params", "invalid JSON")
    if not isinstance(parsed, dict) or parsed.get("op") not in _KERNEL_OPS:
        raw_rid = parsed.get("id") if isinstance(parsed, dict) else None
        rid = raw_rid if isinstance(raw_rid, str) else "?"
        return None, _terminal(rid, "invalid_params", f"op must be one of {', '.join(_KERNEL_OPS)}")
    return parsed, None


def _direct_op(req: Mapping[str, JSONValue], jev: JevClient | None) -> dict[str, JSONValue]:
    """One request off the stdin loop: an op-pool thread or the main thread. Never raises."""
    rid = _request_id(req)
    op = req.get("op")
    try:
        deadline = _deadline_field(req.get("deadlineAt"))
        if deadline is not None and deadline / 1000.0 <= time.time():
            message = f"deadline reached before {op}"
            if op == "prove_claims":
                raw_claims = req.get("claims")
                return _prove_unavailable(rid, message, raw_claims if isinstance(raw_claims, list) else [])
            if op == "research_tool":
                raw_tool = req.get("tool")
                return {"id": rid, **_tool_failure(raw_tool if isinstance(raw_tool, str) else "", message)}
            return _terminal(rid, "timeout", message)
        if op == "open_research":
            return _open_research(req)
        if op == "research_tool":
            return _research_tool(req)
        if op == "prove_claims":
            return _prove_claims(req)
        if op == "arguments":
            return _arguments(req)
        if op == "effort":
            return _effort(req, jev=jev)
        if op == "sections":
            return _sections(req, jev=jev)
        if op == "scope":
            return _scope(req, jev=jev)
        if op == "admit_datapoint":
            return _admit_datapoint(req)
        raise ValueError(f"unsupported op {op!r}")  # _request_or_response rejects every other op
    except Exception as exc:  # never raise out of the worker
        logger.exception("kernel worker direct op failed rid=%s op=%s", rid, op)
        return _terminal(rid, "provider_error", f"worker failed: {exc}")


def _install_signal_handlers(handler: Callable[[int, object], None]) -> None:
    """Install handlers for SIGINT and SIGTERM. Other threads keep their current handlers."""
    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except OSError, ValueError:
        pass


def _watch_caller_hangup(claim: threading.Lock) -> None:
    """If the caller disconnects with no queued input during startup, the watcher signals the main thread.

    The watcher never reads stdin. The main loop answers queued requests at EOF.
    If the main thread owns the claim, the watcher sends no signal. The watcher polls once.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        fd = sys.stdin.fileno()
    except AttributeError, OSError, ValueError:
        return  # StringIO or a closed stream: no OS hangup channel to watch
    main_ident = threading.get_ident()

    def _watch() -> None:
        # Pipes report POLLHUP. A Unix socket half-close reports POLLRDHUP. Errors propagate.
        poller = select.poll()
        poller.register(fd, select.POLLHUP | select.POLLRDHUP)
        poller.poll()
        queued = int.from_bytes(fcntl.ioctl(fd, termios.FIONREAD, bytes(4)), sys.byteorder)
        if queued == 0 and claim.acquire(blocking=False):
            signal.pthread_kill(main_ident, signal.SIGTERM)

    threading.Thread(target=_watch, name="kernel-hangup", daemon=True).start()


def main() -> None:
    _configure_logging()
    exiting = False

    def _on_startup_signal(signum: int, _frame: object) -> None:
        # Startup can hold a client lock here, so this handler never calls _shutdown.
        # A held spawn replays the signal after registration. atexit cleans every owned child.
        nonlocal exiting
        if exiting or defer_termination(signum):
            return
        exiting = True
        raise SystemExit(128 + signum)

    _install_signal_handlers(_on_startup_signal)
    claim = threading.Lock()
    _watch_caller_hangup(claim)
    jev = _startup()
    if not claim.acquire(blocking=False):
        _on_startup_signal(signal.SIGTERM, None)  # the watcher claimed first: the caller is gone
    sys.stdout.write(json.dumps({"type": "ready"}) + "\n")
    sys.stdout.flush()

    def _on_signal(signum: int, _frame: object) -> None:
        _shutdown()
        sys.exit(128 + signum)

    _install_signal_handlers(_on_signal)
    _write_lock = threading.Lock()

    def _emit(resp: Mapping[str, JSONValue]) -> None:
        line = json.dumps(resp)
        with _write_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def _do_watchlist(req: Mapping[str, JSONValue]) -> None:
        rid = _request_id(req)
        try:
            resp = _watchlist(req)
        except Exception as exc:  # never raise out of the worker
            logger.exception("kernel worker watchlist failed rid=%s", rid)
            resp = {"id": rid, "error": f"worker failed: {exc}"[:500]}
        _emit(resp)

    def _do_direct(req: Mapping[str, JSONValue]) -> None:
        _emit(_direct_op(req, jev))

    try:
        # The stdin loop keeps reading while a blocking op runs on a worker:
        # page-load watchlist reads on their own worker, tool/proof ops on four.
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kernel-watchlist") as watch_pool,
            concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="kernel-op") as op_pool,
        ):
            for line in sys.stdin:
                req, response = _request_or_response(line)
                if req is None:
                    if response is not None:
                        _emit(response)
                    continue
                if req.get("op") == "watchlist":
                    watch_pool.submit(_do_watchlist, req)
                elif req.get("op") in _GATE_OPS:
                    op_pool.submit(_do_direct, req)
                else:
                    _emit(_direct_op(req, jev))
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
