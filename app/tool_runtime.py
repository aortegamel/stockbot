"""Runtime tool gateway: runtime-driven tool-call security gates.

The runtime is the calling agent; every tool call it makes still passes these gates:
permit filter, argument validation, intent firewall, egress/private-args
checks, ingress scan, LOCAL_CONTEXT-only execution, DLP, budget + recorder.
Rival pattern to avoid: calling execute_tool directly from a bridge
(drops all gates).

One deliberate divergence: the ingress scan uses envelope_for_tool +
prepare_context directly on rendered evidence — The calling agent itself reads it, and no
source text is promoted into canonical facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .policy import LOCAL_CONTEXT, RequestContext
from .redact import redact_text
from .runtime import ExecutionBudget, ToolResultMeta
from .security.action_policy import (
    TOOL_DOMAINS,
    authorize_egress,
    authorize_tool_call,
    private_pattern_hit,
)
from .security.context import (
    RunSecurityContext,
    Sensitivity,
    SessionAuthorization,
    SessionSecurityState,
    classify_intent,
)
from .security.context_gateway import (
    ContextEnvelope,
    QuarantinedContext,
    SafeContext,
    envelope_for_tool,
    prepare_context,
)
from .security.response_guard import guard_response
from .storage.runs import RunRecorder, get_current_recorder
from .tool_render import render_tool_result
from .tools import (
    TOOLS,
    _invalid_args_error,
    _resolve_company_to_ticker,
    _tool_function,
    _unknown_tool_error,
    _validate_tool_arguments,
    execute_tool,
    tool_is_permitted,
)

if TYPE_CHECKING:
    from app.research.repository import ResearchRepository
from app.research.stage import (
    CONTROL_TOOLS,
    DISCOVERY_TOOLS,
    DISPATCH_TOOLS,
    RESEARCH_TOOL_NAMES,
    check_stage_tool,
    stage_for_session,
)

_CALL_TOOL_FORBIDDEN = frozenset({"call_tool", "browse_tools", "search_tools", "list_tool_domains", "describe_tool"})

# Bounded local thesis lifecycle sinks (sqlite store only; never egress).
# Private-pattern arg scanning would false-positive on user thesis content,
# while real exfiltration vectors (search_web, external research) stay scanned.
_THESIS_LOCAL_TOOLS = frozenset(
    {
        "thesis_create",
        "thesis_show",
        "thesis_refine",
        "thesis_watch",
        "thesis_journal",
    }
)

logger = logging.getLogger(__name__)


def _toolflow_sid(session: RuntimeToolSession, staged: _StagedContext | None = None) -> str:
    """Linkable sid for toolflow logs: staged research id, else the runtime session id."""
    if staged is not None and isinstance(staged.session_id, str) and staged.session_id:
        return staged.session_id
    sid = getattr(session, "session_id", None)
    return sid if isinstance(sid, str) and sid else "-"


def _toolflow_args_summary(arguments: object) -> tuple[list[str], int]:
    """Arg keys + byte size for toolflow logs; never values (no secrets)."""
    keys = sorted(str(k) for k in arguments) if isinstance(arguments, dict) else []
    try:
        nbytes = len(json.dumps(arguments, sort_keys=True, default=str))
    except Exception:  # noqa: BLE001 - size probe never breaks the tool call
        nbytes = -1
    return keys, nbytes


# Model label recorded for runtime-driven tool calls. Handlers ignore it
# (no nested completions remain on the runtime path); it exists for provenance.
RUNTIME_MODEL = "runtime"

_UNAVAILABLE_HEADER = (
    "The requested data is unavailable: one or more tool calls failed or "
    "returned no data, so the exact values cannot be provided. No values "
    "are estimated, derived, or substituted."
)

_UNAVAILABLE_NEXT_STEP = (
    "Next step: correct the request (dataset, fields, filters, or "
    "credentials) and retry, or use a different dataset/source. The error "
    "above states exactly what failed."
)

_NON_DATA_LIST_KEYS = frozenset({"source_records", "warnings", "metrics", "trends"})


def _is_failed_result(result: object) -> bool:
    """A tool result is a failure when it carries an explicit error."""
    return isinstance(result, dict) and bool(result.get("error"))


def _unavailable_data_response(failed: list[tuple[str, dict[str, object]]]) -> str:
    """Deterministic user-facing response when any tool call failed.

    Built from the rendered error context of each failed tool (name,
    dataset/source, HTTP status, sanitized FINRA response, environment)
    plus a next step. The model is never consulted, so nothing can be
    invented, derived, or substituted to fill the gap.
    """
    lines = [_UNAVAILABLE_HEADER, ""]
    for name, result in failed:
        lines.append(f"Tool: {name}")
        for line in render_tool_result(result).splitlines():
            lines.append("  " + line)
        lines.append("")
    lines.append(_UNAVAILABLE_NEXT_STEP)
    return "\n".join(lines)


_SOURCE_REF_ID_KEYS = (
    "accession_no",
    "accession_number",
    "accession",
    "record_id",
    "document_name",
    "filing_id",
)
_SOURCE_REF_URL_KEYS = ("url", "source_url", "source", "filing_url", "document_url", "link")
_SOURCE_REF_DATE_KEYS = (
    "known_at",
    "accepted_at",
    "acceptanceDatetime",
    "acceptedDate",
    "filed_at",
    "filingDate",
    "filed",
    "published_at",
    "publishedAt",
    "published",
)


def _is_uri_like(value: str) -> bool:
    text = value.strip()
    return "://" in text or "/" in text or "." in text


def _source_ref_candidates(result: object) -> list[dict[str, object]]:
    """Candidate mappings: the result plus one level of nested dicts/lists."""
    if not isinstance(result, dict):
        return []
    candidates: list[dict[str, object]] = [result]
    for value in result.values():
        if isinstance(value, dict):
            candidates.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    candidates.append(item)
    return candidates


def _ref_record_id(item: dict[str, object]) -> str | None:
    for key in _SOURCE_REF_ID_KEYS:
        raw_id = item.get(key)
        if isinstance(raw_id, (str, int)) and str(raw_id).strip():
            return str(raw_id).strip()
    return None


def _ref_uri(item: dict[str, object]) -> str | None:
    for key in _SOURCE_REF_URL_KEYS:
        raw_url = item.get(key)
        if isinstance(raw_url, str) and raw_url.strip() and _is_uri_like(raw_url):
            return raw_url.strip()
    return None


def _ref_known_at(item: dict[str, object]) -> str | None:
    for key in _SOURCE_REF_DATE_KEYS:
        raw_date = item.get(key)
        if isinstance(raw_date, str) and raw_date.strip():
            return raw_date.strip()
    return None


def _source_ref_for(item: dict[str, object]) -> dict[str, str]:
    """Single-candidate reference: first id/url/date hit per key group."""
    ref: dict[str, str] = {}
    record_id = _ref_record_id(item)
    if record_id is not None:
        ref["record_id"] = record_id
    uri = _ref_uri(item)
    if uri is not None:
        ref["uri"] = uri
    known_at = _ref_known_at(item)
    if known_at is not None:
        ref["known_at"] = known_at
    return ref


def _extract_source_refs(result: object) -> dict[str, object]:
    """First actual record reference; top-level labels never qualify."""
    all_refs: list[dict[str, str]] = []
    for item in _source_ref_candidates(result):
        ref = _source_ref_for(item)
        if ref and ref not in all_refs:
            all_refs.append(ref)
    if not all_refs:
        return {}
    primary: dict[str, object] = dict(next((r for r in all_refs if "record_id" in r), all_refs[0]))
    if len(all_refs) > 1:
        primary["all"] = all_refs
    return primary


_META_FRESHNESS_KEYS = ("retrieved_at", "data_freshness", "freshness", "as_of_date", "as_of")
_META_AS_OF_KEYS = ("as_of_date", "as_of")


def _meta_source_name(result: dict[str, object]) -> object:
    return result.get("source") or result.get("dataset_id") or result.get("dataset")


def _meta_freshness(result: dict[str, object], source: object) -> tuple[object, dict[str, str]]:
    freshness_value = next((result[k] for k in _META_FRESHNESS_KEYS if result.get(k) is not None), None)
    freshness = {str(source): str(freshness_value)} if source is not None and freshness_value is not None else {}
    return freshness_value, freshness


def _meta_row_count(result: dict[str, object]) -> tuple[int, int | None]:
    returned_count = result.get("returned_count") if isinstance(result.get("returned_count"), int) else None
    row_count = (
        result.get("row_count")
        if isinstance(result.get("row_count"), int)
        else max((len(v) for k, v in result.items() if isinstance(v, list) and k not in _NON_DATA_LIST_KEYS), default=0)
    )
    return row_count, returned_count


def _meta_truncated(result: dict[str, object], returned_count: int | None) -> bool:
    total = result.get("total_records")
    return (
        bool(result.get("truncated"))
        or result.get("may_have_more") is True
        or (returned_count is not None and isinstance(total, int) and returned_count < total)
    )


def _tool_result_meta(result: object) -> ToolResultMeta:
    """Best-effort telemetry envelope for a tool result: row counts,
    truncation, source name, and freshness."""
    if not isinstance(result, dict):
        return ToolResultMeta(0, None, False, None, [], {})
    source = _meta_source_name(result)
    source_names = [str(source)] if source is not None else []
    _, source_freshness = _meta_freshness(result, source)
    as_of = next((result[k] for k in _META_AS_OF_KEYS if result.get(k) is not None), None)
    row_count, returned_count = _meta_row_count(result)
    truncated = _meta_truncated(result, returned_count)
    return ToolResultMeta(
        row_count, returned_count, truncated, str(as_of) if as_of is not None else None, source_names, source_freshness
    )


@dataclass
class RuntimeToolSession:
    """Per-runtime-session grant + labels + budget. Deny-by-default."""

    session_id: str
    active_research_session_id: str | None = None
    active_research_job_id: str | None = None
    authorization: SessionAuthorization = field(default_factory=SessionAuthorization)
    security_state: SessionSecurityState = field(default_factory=SessionSecurityState)
    run_security: RunSecurityContext = field(init=False)
    budget: ExecutionBudget = field(init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.security_state.authorization = self.authorization
        if self.security_state.private_context_seen:
            data_labels = {"private"}
        else:
            data_labels = set()
        self.run_security = RunSecurityContext(
            original_intent=classify_intent([]),
            capabilities=frozenset(cap.name for cap in LOCAL_CONTEXT.capabilities),
            authorization=self.authorization,
            data_labels=data_labels,
        )
        limits = LOCAL_CONTEXT.run_limits
        self.budget = ExecutionBudget(
            max_tool_calls=limits.max_tool_calls,
            max_runtime=limits.max_runtime,
            max_evidence_tokens=limits.max_evidence_tokens,
        )


@dataclass(frozen=True)
class ToolOutcome:
    tool_name: str
    content: str
    source_handle: dict[str, object] | None
    source_refs: dict[str, object] | None
    error: str | None
    error_type: str | None
    retryable: bool
    meta: ToolResultMeta
    tool_result_id: str | None = None


# ponytail: only handler-level tool_error retries; denials/budgets/validation are deterministic.
_RETRYABLE_ERROR_TYPES = frozenset({"tool_error"})


def outcome_from_result(tool_name: str, result: dict[str, object]) -> ToolOutcome:
    """Convert a tool result dict into a frozen scheduler-facing outcome."""
    content = render_tool_result(result)
    refs = _extract_source_refs(result)
    source_refs = dict(refs) if refs else None
    raw_handle = result.get("source_handle")
    if not isinstance(raw_handle, dict) or not raw_handle:
        _meta = result.get("meta") if isinstance(result, dict) else None
        if isinstance(_meta, dict):
            _mh = _meta.get("source_handle")
            if isinstance(_mh, dict) and _mh:
                raw_handle = _mh
    source_handle = dict(raw_handle) if isinstance(raw_handle, dict) and raw_handle else None
    raw_error = result.get("error")
    error = str(raw_error) if raw_error is not None else None
    raw_etype = result.get("error_type")
    error_type = raw_etype if isinstance(raw_etype, str) else None
    retryable = error is not None and error_type in _RETRYABLE_ERROR_TYPES
    return ToolOutcome(
        tool_name=tool_name,
        content=content,
        source_handle=source_handle,
        source_refs=source_refs,
        error=error,
        error_type=error_type,
        retryable=retryable,
        meta=_tool_result_meta(result),
        tool_result_id=result.get("tool_result_id") if isinstance(result.get("tool_result_id"), str) else None,
    )


def _record_security(
    session: RuntimeToolSession,
    source: str,
    payload: str,
    decision: str,
    reason: str | None,
    *,
    score: int | None = None,
    verdict: str | None = None,
    rule_ids: list[str] | None = None,
) -> None:
    with session._lock:
        session.run_security.security_events.append(
            {
                "source": source,
                "decision": decision,
                "reason": reason,
            }
        )
    recorder = get_current_recorder()
    if recorder is not None:
        recorder.record_security_event(
            source=source,
            sha256=hashlib.sha256(payload.encode()).hexdigest(),
            score=score,
            verdict=verdict,
            rule_ids=rule_ids or [],
            decision=decision,
            reason=reason,
        )


def _args_json(arguments: dict[str, object]) -> str:
    return json.dumps(arguments, sort_keys=True)


def _override_context(
    data_root: str | Path | None = None,
    as_of: str | None = None,
    research_session_id: str | None = None,
):
    """LOCAL_CONTEXT with data_root/as_of/research-session overridden; invalid roots fall back."""

    def _with_root(root: Path) -> RequestContext:
        return RequestContext(
            principal_id=LOCAL_CONTEXT.principal_id,
            capabilities=LOCAL_CONTEXT.capabilities,
            tool_policy=LOCAL_CONTEXT.tool_policy,
            data_root=root,
            run_limits=LOCAL_CONTEXT.run_limits,
            as_of=as_of,
            research_session_id=research_session_id,
        )

    if data_root is None or not str(data_root):
        if as_of is None and research_session_id is None:
            return LOCAL_CONTEXT
        return _with_root(LOCAL_CONTEXT.data_root)
    try:
        root = Path(str(data_root))
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return _with_root(LOCAL_CONTEXT.data_root) if as_of is not None else LOCAL_CONTEXT
    if not root.is_absolute():
        return _with_root(LOCAL_CONTEXT.data_root) if as_of is not None else LOCAL_CONTEXT
    return _with_root(root)


def execute_agent_tool(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    *,
    tool_call_id: str | None = None,
    protocol_id: str | None = None,
    bridge_queue_ms: float = 0.0,
    data_root: str | Path | None = None,
    as_of: str | None = None,
    active_research_session_id: str | None = None,
    active_research_job_id: str | None = None,
) -> dict[str, object]:
    """Run one agent-requested tool through all gates. Never raises."""
    try:
        return _execute_agent_tool(
            name,
            arguments,
            session,
            tool_call_id=tool_call_id,
            protocol_id=protocol_id,
            bridge_queue_ms=bridge_queue_ms,
            data_root=data_root,
            as_of=as_of,
            active_research_session_id=active_research_session_id,
            active_research_job_id=active_research_job_id,
        )
    except Exception as exc:  # never break the bridge loop
        logger.exception("Runtime tool gateway failed for '%s'", name)
        return {"error": f"Runtime tool gateway failed for tool '{name}': {exc}"}


@dataclass
class _CallToolUnwrap:
    """Validated call_tool unwrap: inner name or ready-made error outcome."""

    inner_name: str | None = None
    inner_args: dict[str, object] | None = None
    error: dict[str, object] | None = None


@dataclass
class _StagedContext:
    """Immutable request-local research IDs plus the shared store."""

    session_id: str | None = None
    job_id: str | None = None
    sid_from_explicit: bool = False
    store: ResearchRepository | None = None


def _call_tool_parts(arguments: object) -> tuple[object, object]:
    raw_inner: object = arguments.get("name") if isinstance(arguments, dict) else None
    raw_args: object = arguments.get("arguments") if isinstance(arguments, dict) else None
    empty: dict[str, object] = {}
    return raw_inner, empty if raw_args is None else raw_args


def _call_tool_name_error(arguments: object, raw_inner: object) -> dict[str, object] | None:
    if isinstance(raw_inner, str) and raw_inner.strip():
        return None
    invalid_outer = _validate_tool_arguments("call_tool", arguments if isinstance(arguments, dict) else {})
    msg = invalid_outer if invalid_outer is not None else "call_tool: 'name' must be a non-empty string"
    return _invalid_args_error("call_tool", msg)


def _call_tool_target_error(arguments: object, inner_name: str, inner_args: object) -> dict[str, object] | None:
    if not isinstance(inner_args, dict):
        invalid_inner = _validate_tool_arguments("call_tool", arguments if isinstance(arguments, dict) else {})
        msg_inner = invalid_inner if invalid_inner is not None else "call_tool: 'arguments' must be an object"
        return _invalid_args_error("call_tool", msg_inner)
    if inner_name in _CALL_TOOL_FORBIDDEN:
        return _invalid_args_error(
            "call_tool",
            f"Tool '{inner_name}' cannot be called via call_tool; call browse_tools to find the exact canonical name, then call_tool with a research tool name",
        )
    if not any(_tool_function(t).get("name") == inner_name for t in TOOLS):
        return _unknown_tool_error(inner_name)
    return None


def _unwrap_call_tool(name: str, arguments: object) -> _CallToolUnwrap:
    """Validate the outer call_tool wrapper; success carries inner dispatch."""
    if name != "call_tool":
        return _CallToolUnwrap(inner_name=None, inner_args=None, error=None)
    raw_inner, inner_args = _call_tool_parts(arguments)
    name_error = _call_tool_name_error(arguments, raw_inner)
    if name_error is not None:
        return _CallToolUnwrap(error=name_error)
    if not (isinstance(raw_inner, str) and raw_inner.strip()):
        return _CallToolUnwrap(error=_invalid_args_error("call_tool", "call_tool: 'name' must be a non-empty string"))
    inner_name = raw_inner.strip()
    target_error = _call_tool_target_error(arguments, inner_name, inner_args)
    if target_error is not None:
        return _CallToolUnwrap(error=target_error)
    if not isinstance(inner_args, dict):
        return _CallToolUnwrap(error=_invalid_args_error("call_tool", "call_tool: 'arguments' must be an object"))
    return _CallToolUnwrap(inner_name=inner_name, inner_args=inner_args)


def _tool_schema_parts(name: str) -> tuple[list[str], set[str]]:
    """Schema required/properties names for one tool (empty when undeclared)."""
    fn = next((_tool_function(t) for t in TOOLS if _tool_function(t).get("name") == name), None)
    fparams = fn.get("parameters") if isinstance(fn, dict) else None
    fprops = fparams.get("properties") if isinstance(fparams, dict) else None
    freq = fparams.get("required") if isinstance(fparams, dict) else None
    req_names: list[str] = [str(r) for r in freq] if isinstance(freq, list) else []
    prop_names: set[str] = set(fprops.keys()) if isinstance(fprops, dict) else set()
    return req_names, prop_names


def _pick_id_key(req_names: list[str], prop_names: set[str]) -> str | None:
    for key in ("ticker", "entity"):
        if key in req_names and key in prop_names:
            return key
    for key in ("ticker", "entity"):
        if key in prop_names:
            return key
    return None


def _company_id_key(name: str) -> tuple[str | None, set[str]]:
    """Schema-driven identifier key for company-name resolution (ticker/entity)."""
    req_names, prop_names = _tool_schema_parts(name)
    return _pick_id_key(req_names, prop_names), prop_names


def _company_resolve_target(arguments: dict[str, object], id_key: str) -> str | None:
    raw_id = arguments.get(id_key)
    if isinstance(raw_id, str) and raw_id.strip():
        return None
    raw_cname = arguments.get("company_name")
    if not (isinstance(raw_cname, str) and raw_cname.strip()):
        return None
    return raw_cname


def _try_company_to_ticker(raw_cname: str) -> str | None:
    try:
        return _resolve_company_to_ticker(raw_cname)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _resolve_company_arguments(name: str, arguments: dict[str, object], sid: str | None = None) -> dict[str, object]:
    """Fill a missing ticker/entity identifier from company_name when the schema allows."""
    id_key, prop_names = _company_id_key(name)
    if id_key is None or "company_name" not in prop_names:
        return arguments
    raw_cname = _company_resolve_target(arguments, id_key)
    if raw_cname is None:
        return arguments
    resolved = _try_company_to_ticker(raw_cname)
    if resolved:
        logger.info("toolflow company_resolve sid=%s tool=%s id_key=%s", sid or "-", name, id_key)
        return {**arguments, id_key: resolved}
    return arguments


def _explicit_staged_ids(arguments: object) -> tuple[str | None, str | None]:
    """Explicit canonical session/job IDs carried inside arguments."""
    explicit_sid: str | None = None
    explicit_jid: str | None = None
    if isinstance(arguments, dict):
        raw_sid = arguments.get("session_id")
        if isinstance(raw_sid, str) and raw_sid:
            explicit_sid = raw_sid
        raw_jid = arguments.get("job_id")
        if isinstance(raw_jid, str) and raw_jid:
            explicit_jid = raw_jid
    return explicit_sid, explicit_jid


def _captured_staged_ids(
    active_research_session_id: str | None,
    active_research_job_id: str | None,
) -> tuple[str | None, str | None]:
    cap_sid = (
        active_research_session_id
        if isinstance(active_research_session_id, str) and active_research_session_id
        else None
    )
    cap_jid = active_research_job_id if isinstance(active_research_job_id, str) and active_research_job_id else None
    return cap_sid, cap_jid


def _session_staged_ids(session: RuntimeToolSession) -> tuple[str | None, str | None]:
    with session._lock:
        ctx_sid = session.active_research_session_id
        ctx_jid = session.active_research_job_id
    if not isinstance(ctx_sid, str) or not ctx_sid:
        ctx_sid = None
    if not isinstance(ctx_jid, str) or not ctx_jid:
        ctx_jid = None
    return ctx_sid, ctx_jid


def _resolve_staged_context(
    arguments: object,
    session: RuntimeToolSession,
    *,
    data_root: str | Path | None,
    as_of: str | None,
    active_research_session_id: str | None,
    active_research_job_id: str | None,
) -> _StagedContext:
    """Resolve immutable request-local IDs: explicit args, bridge pair, session fields."""
    explicit_sid, explicit_jid = _explicit_staged_ids(arguments)
    cap_sid, cap_jid = _captured_staged_ids(active_research_session_id, active_research_job_id)
    ctx_sid, ctx_jid = _session_staged_ids(session)
    resolved_sid = explicit_sid or cap_sid or ctx_sid
    resolved_jid = explicit_jid or cap_jid or ctx_jid
    store = None
    if isinstance(resolved_sid, str) and resolved_sid:
        from app.research.repository import ResearchRepository as _RR

        store = _RR(data_root=_override_context(data_root, as_of).data_root)
    return _StagedContext(
        session_id=resolved_sid, job_id=resolved_jid, sid_from_explicit=explicit_sid is not None, store=store
    )


def _check_staged_session(name: str, staged: _StagedContext) -> dict[str, object] | None:
    """Attached staged session must exist (fail closed); discovery skips stage check only."""
    if staged.store is None:
        return None
    assert isinstance(staged.session_id, str) and staged.session_id
    sid: str = staged.session_id
    try:
        found: object | None = staged.store.get_session(sid)
    except KeyError:
        if not staged.sid_from_explicit:
            logger.warning("toolflow deny sid=%s tool=%s gate=stage reason=unknown-session", sid, name)
            return {"error": f"Unknown research session '{sid}'", "error_type": "invalid_research_context"}
        return None
    if name not in DISCOVERY_TOOLS:
        stage = stage_for_session(found, staged.store.list_jobs(sid))
        try:
            check_stage_tool(stage, name)
        except ValueError as exc:
            logger.warning("toolflow deny sid=%s tool=%s gate=stage reason=%.200s", sid, name, str(exc))
            return {"error": str(exc)}
    return None


def _check_permit_and_schema(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    args_for_hash: str,
    sid: str | None = None,
) -> dict[str, object] | None:
    """Gate 1 permit filter + Gate 2 schema validation and arg-bytes cap."""
    if name not in RESEARCH_TOOL_NAMES or not tool_is_permitted(name, LOCAL_CONTEXT):
        _record_security(session, name, args_for_hash, "action_blocked", f"tool not permitted: {name}")
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=permit reason=not-permitted", sid or _toolflow_sid(session), name
        )
        return {"error": f"Tool is not permitted: {name}"}
    invalid = _validate_tool_arguments(name, arguments)
    if invalid is not None:
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=schema reason=invalid-args err=%.200s",
            sid or _toolflow_sid(session),
            name,
            invalid,
        )
        return _invalid_args_error(name, invalid)
    arg_bytes = len(json.dumps(arguments))
    if arg_bytes > LOCAL_CONTEXT.tool_policy.max_arguments_bytes:
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=schema reason=args-too-large bytes=%d",
            sid or _toolflow_sid(session),
            name,
            arg_bytes,
        )
        return {
            "error": (
                "Tool arguments exceed the maximum size "
                f"({LOCAL_CONTEXT.tool_policy.max_arguments_bytes} bytes): {name}"
            ),
            "error_type": "invalid_tool_arguments",
        }
    return None


def _dispatch_value_error(exc: ValueError) -> dict[str, object]:
    return {"error": str(exc)}


def _consume_dispatch_budget(
    name: str, staged: _StagedContext, arguments: dict[str, object]
) -> tuple[bool, dict[str, object] | None]:
    """Gate 8: attached staged data dispatches consume one persisted kernel slot.

    The validated arguments ride along: the kernel keys its no-progress repeat
    gate on the normalized action (tool + arguments) and refuses an exact
    repeat whose prior run produced no new evidence.
    """
    if name not in DISPATCH_TOOLS or staged.store is None:
        return False, None
    if not isinstance(staged.job_id, str) or not staged.job_id:
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=dispatch-budget reason=no-job", staged.session_id or "-", name
        )
        return False, {
            "error": f"Active research job is required for tool '{name}'",
            "error_type": "invalid_research_context",
        }
    if not isinstance(staged.session_id, str) or not staged.session_id:
        logger.warning("toolflow deny tool=%s gate=dispatch-budget reason=no-session", name)
        return False, {
            "error": f"Active research session is required for tool '{name}'",
            "error_type": "invalid_research_context",
        }
    try:
        from app.research import service as _svc

        _svc.authorize_and_consume_dispatch(
            staged.session_id, staged.job_id, name, arguments=arguments, repo=staged.store
        )
    except ValueError as exc:
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=dispatch-budget reason=%.200s", staged.session_id, name, str(exc)
        )
        return False, _dispatch_value_error(exc)
    except KeyError as exc:
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=dispatch-budget reason=%.200s", staged.session_id, name, str(exc)
        )
        return False, {"error": str(exc), "error_type": "invalid_research_context"}
    return True, None


def _heartbeat_staged_job(name: str, staged: _StagedContext) -> None:
    try:
        if (
            staged.store is not None
            and isinstance(staged.job_id, str)
            and staged.job_id
            and (name in DISPATCH_TOOLS or name in CONTROL_TOOLS)
        ):
            from app.research import service as _hb_svc

            _hb_svc.heartbeat_job(staged.job_id, repo=staged.store)
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
        pass


_REPLAYABLE_TOOL_RESULTS: frozenset[str] = frozenset(
    {
        "get_finra_datapoints",
        "query_finra",
        "get_short_interest",
        "get_short_pressure_profile",
        "get_reg_sho_volume",
        "get_threshold_securities",
        "get_short_interest_leaderboard",
        "search_web",
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
"""Staged tools whose success payloads persist for kernel evidence replay."""


def _persist_staged_tool_result(
    name: str, result: dict[str, object], staged: _StagedContext, tool_call_id: str
) -> str | None:
    """Persist one staged FINRA/WEB result; returns its id for the citation path.

    Best-effort observability, never a gate: persistence failures keep the tool
    result (evidence admission re-checks the persisted row and fails closed
    when it is absent). The id rides in the result payload so the desk can cite
    the exact bytes the kernel stored.
    """
    if name not in _REPLAYABLE_TOOL_RESULTS or staged.store is None:
        return None
    if not isinstance(staged.session_id, str) or not staged.session_id:
        return None
    if not isinstance(staged.job_id, str) or not staged.job_id:
        return None
    tool_result_id = f"{staged.session_id}:tr:{tool_call_id}" if tool_call_id else None
    try:
        from app.research import service as _svc

        out = _svc.persist_tool_result(
            staged.session_id, staged.job_id, name, tool_result_id, result, repo=staged.store
        )
        rid = out.get("tool_result_id")
        if isinstance(rid, str) and rid and isinstance(result, dict):
            result["tool_result_id"] = rid
        return rid if isinstance(rid, str) else None
    except Exception:  # noqa: BLE001 - persistence never breaks the tool result
        return None


def _run_budget_refusal(name: str, session: RuntimeToolSession) -> dict[str, object]:
    """Distinct runtime run-budget refusal: runtime exhaustion vs call-count exhaustion."""
    with session._lock:
        remaining = session.budget.runtime_remaining()
        if name == "search_web":
            used = session.budget.search_calls
            maximum = session.budget.max_search_calls
        else:
            used = session.budget.tool_calls
            maximum = session.budget.max_tool_calls
    shown = "unlimited" if maximum is None else maximum
    if remaining <= 0:
        return {
            "error": f"Runtime run runtime budget exceeded (remaining {remaining:.1f}s); retry with a narrower question or fewer tool calls.",
            "error_type": "deadline_exceeded",
        }
    if name == "search_web":
        return {
            "error": f"Runtime run search budget exceeded (used {used}/{shown} search calls); retry with a narrower question or fewer tool calls.",
            "error_type": "run_budget_exceeded",
        }
    return {
        "error": f"Runtime run tool budget exceeded (used {used}/{shown} tool calls); retry with a narrower question or fewer tool calls.",
        "error_type": "run_budget_exceeded",
    }


def _reserve_run_budget(name: str, session: RuntimeToolSession, dispatch_consumed: bool) -> bool:
    """One budget slot per call before any external work; search_web uses its pool."""
    if dispatch_consumed:
        return True
    with session._lock:
        if name != "search_web":
            return session.budget.reserve_tool_call()
        return session.budget.reserve_search_call()


def _check_intent_and_egress(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    args_for_hash: str,
) -> dict[str, object] | None:
    """Gate 3 intent firewall + Gate 4 search egress / private-pattern args check."""
    with session._lock:
        intent_allowed, intent_reason = authorize_tool_call(name, arguments, session.run_security)
    if not intent_allowed:
        if TOOL_DOMAINS.get(name) == "portfolio_read":
            _record_security(session, name, args_for_hash, "action_blocked", intent_reason)
            logger.warning(
                "toolflow deny sid=%s tool=%s gate=intent reason=portfolio-denied", _toolflow_sid(session), name
            )
            return {
                "error": "Portfolio access is not authorized for this session",
                "error_type": "authorization_denied",
                "soft": True,
            }
        _record_security(session, name, args_for_hash, "action_blocked", intent_reason)
        logger.warning("toolflow deny sid=%s tool=%s gate=intent reason=intent-denied", _toolflow_sid(session), name)
        return {
            "error": "Tool call exceeds original user intent",
            "error_type": "intent_denied",
            "soft": True,
        }
    egress_error = _check_egress_or_private(name, arguments, session, args_for_hash)
    if egress_error is not None:
        return egress_error
    return None


def _check_egress_or_private(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    args_for_hash: str,
) -> dict[str, object] | None:
    if name == "search_web":
        return _check_search_egress(name, arguments, session, args_for_hash)
    if name in _THESIS_LOCAL_TOOLS:
        return None
    hit = private_pattern_hit(args_for_hash)
    if hit:
        _record_security(session, name, args_for_hash, "action_blocked", hit)
        logger.warning("toolflow deny sid=%s tool=%s gate=egress reason=private-args", _toolflow_sid(session), name)
        return {
            "error": "Tool arguments contain private data that must not be transmitted",
            "error_type": "private_args_denied",
            "soft": True,
        }
    return None


def _check_search_egress(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    args_for_hash: str,
) -> dict[str, object] | None:
    with session._lock:
        decision = authorize_egress("exa", arguments, session.run_security)
    if decision.allowed:
        return None
    _record_security(session, name, args_for_hash, "egress_blocked", decision.reason)
    logger.warning("toolflow deny sid=%s tool=%s gate=egress reason=egress-denied", _toolflow_sid(session), name)
    return {
        "error": "Egress blocked: private data must not leave Stockbot",
        "error_type": "egress_denied",
        "soft": True,
    }


def _cache_flags(result: object) -> tuple[bool | None, str | None]:
    raw_cache_hit = result.get("cache_hit") if isinstance(result, dict) else None
    cache_hit = raw_cache_hit if isinstance(raw_cache_hit, bool) else None
    raw_cache_type = result.get("cache_type") if isinstance(result, dict) else None
    cache_type = raw_cache_type if isinstance(raw_cache_type, str) else None
    return cache_hit, cache_type


def _failed_dict(result: object) -> dict[str, object] | None:
    """Failed tool result as a dict, else None (success/non-dict results)."""
    if isinstance(result, dict) and bool(result.get("error")):
        return result
    return None


def _failure_outcome(result: object) -> tuple[bool, bool, bool, str, str | None, str | None]:
    """Classify a handler result: failed/soft/denied plus recorder status fields."""
    failed_map = _failed_dict(result)
    failed = failed_map is not None
    soft = failed_map is not None and failed_map.get("soft") is True
    denied = failed_map is not None and "not permitted" in str(failed_map.get("error", ""))
    status = "completed" if not failed else ("denied" if denied else "failed")
    error_type: str | None = None
    error_message: str | None = None
    if failed_map is not None:
        raw_error_type = failed_map.get("error_type") or ("permission_denied" if denied else "tool_error")
        error_type = raw_error_type if isinstance(raw_error_type, str) else None
        error_message = redact_text(str(failed_map.get("error")))[:2000]
    return failed, soft, denied, status, error_type, error_message


def _record_tool_call_row(
    *,
    recorder: RunRecorder | None,
    run_id: str,
    resolved_tc_id: str,
    name: str,
    arguments: dict[str, object],
    t0_iso: str,
    status: str,
    meta: ToolResultMeta,
    result: object,
    error_type: str | None,
    error_message: str | None,
    protocol_id: str | None,
    bridge_queue_ms: float,
    handler_ms: float,
    cache_hit: bool | None,
    cache_type: str | None,
) -> None:
    if recorder is None:
        return
    recorder.record_tool_call(
        tool_call_id=resolved_tc_id,
        round=0,
        tool_name=name,
        arguments_json=json.dumps(arguments),
        started_at=t0_iso,
        completed_at=datetime.now(UTC).isoformat(),
        status=status,
        result_row_count=meta.row_count,
        returned_count=meta.returned_count,
        truncated=meta.truncated,
        result_bytes=len(json.dumps(result)),
        result_hash=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
        source_names=json.dumps(meta.source_names),
        source_freshness=json.dumps(meta.source_freshness),
        as_of=meta.as_of,
        error_type=error_type,
        error_message=error_message,
        protocol_id=protocol_id,
        bridge_queue_ms=bridge_queue_ms,
        handler_ms=handler_ms,
        cache_hit=cache_hit,
        cache_type=cache_type,
    )


def _resolve_tool_call_id(
    recorder: RunRecorder | None,
    run_id: str,
    tool_call_id: str | None,
) -> str:
    if tool_call_id is not None:
        resolved = f"{run_id}:tc:{tool_call_id}"
        if recorder is not None:
            recorder.next_tool_seq()  # keep the run's tool-call count truthful
        return resolved
    if recorder is not None:
        return f"{run_id}:tc:{recorder.next_tool_seq()}"
    return f"{run_id}:tc:0"


def _execute_and_record(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    *,
    run_id: str,
    recorder: RunRecorder | None,
    tool_call_id: str | None,
    protocol_id: str | None,
    bridge_queue_ms: float,
    data_root: str | Path | None,
    as_of: str | None,
    research_session_id: str | None,
) -> tuple[
    object,
    str,
    str,
    ToolResultMeta,
    str | None,
    str | None,
    bool | None,
    str | None,
    float,
]:
    """Gate 6: run the handler outside the session lock, then record telemetry."""
    t0_iso = datetime.now(UTC).isoformat()
    handler_t0 = time.perf_counter()
    result = execute_tool(
        name,
        arguments,
        RUNTIME_MODEL,
        context=_override_context(data_root, as_of, research_session_id),
    )
    handler_ms = (time.perf_counter() - handler_t0) * 1000.0
    cache_hit, cache_type = _cache_flags(result)
    _failed, _soft, _denied, status, error_type, error_message = _failure_outcome(result)
    meta = _tool_result_meta(result)
    resolved_tc_id = _resolve_tool_call_id(recorder, run_id, tool_call_id)
    _record_tool_call_row(
        recorder=recorder,
        run_id=run_id,
        resolved_tc_id=resolved_tc_id,
        name=name,
        arguments=arguments,
        t0_iso=t0_iso,
        status=status,
        meta=meta,
        result=result,
        error_type=error_type,
        error_message=error_message,
        protocol_id=protocol_id,
        bridge_queue_ms=bridge_queue_ms,
        handler_ms=handler_ms,
        cache_hit=cache_hit,
        cache_type=cache_type,
    )
    return (
        result,
        resolved_tc_id,
        status,
        meta,
        error_type,
        error_message,
        cache_hit,
        cache_type,
        handler_ms,
    )


def _failed_result_response(
    name: str,
    result: dict[str, object],
    error_type: str | None,
    soft: bool,
) -> dict[str, object]:
    """Gate 7 (hard failure): deterministic unavailable-data shape, no model call."""
    if soft:
        return result
    return {
        "error": _unavailable_data_response([(name, result)]),
        "error_type": error_type or "tool_error",
    }


def _ingress_outcome(
    name: str,
    result: dict[str, object],
    session: RuntimeToolSession,
) -> tuple[str, ContextEnvelope, SafeContext] | dict[str, object]:
    """Gate 5: ingress scan on the rendered evidence; quarantined results withheld."""
    rendered = render_tool_result(result, max_bytes=LOCAL_CONTEXT.run_limits.max_tool_result_bytes)
    envelope = envelope_for_tool(name, result)
    outcome = prepare_context(envelope, rendered)
    if isinstance(outcome, QuarantinedContext):
        with session._lock:
            session.run_security.quarantined_items += 1
        _record_security(
            session,
            envelope.source,
            rendered,
            "quarantined" if outcome.verdict == "QUARANTINE" else "blocked",
            "; ".join(outcome.reasons) if outcome.reasons else None,
            score=outcome.score,
            verdict=outcome.verdict,
            rule_ids=list(outcome.rule_ids),
        )
        return {
            "error": ("Tool result withheld by Stockbot security gateway. No usable evidence was provided."),
            "error_type": "ingress_blocked",
            "soft": True,
        }
    return outcome.text, envelope, outcome


def _dlp_and_evidence_labels(
    name: str,
    text: str,
    envelope: ContextEnvelope,
    session: RuntimeToolSession,
    run_id: str,
) -> tuple[str, bool]:
    """Gate 7 (success path) DLP: guard what the agent receives, apply labels, bill evidence."""
    with session._lock:
        final_text = guard_response(text, session.run_security, run_id)
        if name == "search_web":
            session.run_security.data_labels.add("external")
        if envelope.sensitivity is Sensitivity.PRIVATE:
            session.run_security.data_labels.add("private")
        evidence_allowed = session.budget.add_evidence_tokens(len(final_text) // 4)
    return final_text, evidence_allowed


def _record_success_evidence(
    *,
    recorder: RunRecorder | None,
    run_id: str,
    evidence_id: str,
    tool_call_id: str,
    name: str,
    meta: ToolResultMeta,
    final_text: str,
    envelope: ContextEnvelope,
) -> None:
    if recorder is None:
        return
    recorder.record_evidence(
        evidence_id=evidence_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        round=0,
        tool_name=name,
        rendered_hash=hashlib.sha256(final_text.encode()).hexdigest(),
        rendered_bytes=len(final_text.encode("utf-8")),
        estimated_tokens=len(final_text) // 4,
        source_names=json.dumps(meta.source_names),
        source_freshness=json.dumps(meta.source_freshness),
        as_of=meta.as_of,
        rendered_text=redact_text(final_text),
    )
    recorder.record_security_event(
        source=envelope.source,
        sha256=hashlib.sha256(final_text.encode()).hexdigest(),
        score=None,
        verdict=None,
        rule_ids=[
            envelope.source,
            envelope.sensitivity.value,
            envelope.integrity.value,
        ],
        decision="allowed",
        reason=None,
    )


def _named_tool_list(items: list[object]) -> list[str]:
    return [m["name"] for m in items if isinstance(m, dict) and isinstance(m.get("name"), str)]


def _schema_tool_names(schemas: list[object]) -> list[str]:
    names: list[str] = []
    for schema in schemas:
        fn = schema.get("function") if isinstance(schema, dict) else None
        tool_name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(tool_name, str):
            names.append(tool_name)
    return names


def _discovery_meta_names(result: object) -> tuple[list[str] | None, list[str] | None]:
    """Deferred-loading names for search_tools/browse_tools meta (matches/tools)."""
    matches: list[str] | None = None
    tools: list[str] | None = None
    if not isinstance(result, dict):
        return matches, tools
    raw_matches = result.get("matches")
    if isinstance(raw_matches, list):
        matches = _named_tool_list(raw_matches)
    else:
        raw_schemas = result.get("schemas")
        if isinstance(raw_schemas, list):
            matches = _schema_tool_names(raw_schemas)
    raw_tools = result.get("tools")
    if isinstance(raw_tools, list):
        tools = _named_tool_list(raw_tools)
    return matches, tools


def _success_meta(
    name: str,
    result: object,
    meta: ToolResultMeta,
    envelope: ContextEnvelope,
    status: str,
) -> dict[str, object]:
    safe_meta: dict[str, object] = {
        "row_count": meta.row_count,
        "returned_count": meta.returned_count,
        "truncated": meta.truncated,
        "source": envelope.source,
        "sensitivity": envelope.sensitivity.value,
        "integrity": envelope.integrity.value,
        "status": status,
        "as_of": meta.as_of,
    }
    source_refs = _extract_source_refs(result)
    if source_refs:
        safe_meta["source_refs"] = source_refs
    handle = result.get("source_handle") if isinstance(result, dict) else None
    if isinstance(handle, dict) and handle:
        safe_meta["source_handle"] = dict(handle)
    if name in ("search_tools", "browse_tools") and isinstance(result, dict):
        # Deferred loading: the TS extension activates these schemas additively.
        # Names are already model-visible in content; meta carries them structured.
        matches, tools = _discovery_meta_names(result)
        if matches is not None:
            safe_meta["matches"] = matches
        if tools is not None:
            safe_meta["tools"] = tools
    return safe_meta


def _finalize_success(
    name: str,
    result: dict[str, object],
    session: RuntimeToolSession,
    *,
    run_id: str,
    recorder: RunRecorder | None,
    resolved_tc_id: str,
    status: str,
    meta: ToolResultMeta,
) -> dict[str, object]:
    """Gate 5 ingress scan + Gate 7 DLP/evidence + success meta envelope."""
    scanned = _ingress_outcome(name, result, session)
    if isinstance(scanned, dict):
        return scanned
    text, envelope, _outcome = scanned
    final_text, evidence_allowed = _dlp_and_evidence_labels(name, text, envelope, session, run_id)
    if not evidence_allowed:
        with session._lock:
            _used = session.budget.evidence_tokens
            _maximum = session.budget.max_evidence_tokens
        return {
            "error": f"Runtime evidence-token budget exceeded (used {_used}/{_maximum} tokens); narrow the question or window.",
            "error_type": "evidence_budget_exceeded",
        }
    if recorder is not None:
        evidence_id = f"{run_id}:evid:{recorder.next_evidence_seq():04d}"
        _record_success_evidence(
            recorder=recorder,
            run_id=run_id,
            evidence_id=evidence_id,
            tool_call_id=resolved_tc_id,
            name=name,
            meta=meta,
            final_text=final_text,
            envelope=envelope,
        )
    return {"content": final_text, "meta": _success_meta(name, result, meta, envelope, status)}


def _dispatch_inner_call(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    *,
    tool_call_id: str | None,
    protocol_id: str | None,
    bridge_queue_ms: float,
    data_root: str | Path | None,
    as_of: str | None,
    active_research_session_id: str | None,
    active_research_job_id: str | None,
) -> dict[str, object] | None:
    """Generic dispatch: call_tool validates then tail-calls the inner tool once."""
    # The outer wrapper consumes no budget slot and writes no recorder row.
    if name != "call_tool":
        return None
    unwrap = _unwrap_call_tool(name, arguments)
    if unwrap.error is not None:
        return unwrap.error
    inner_name = unwrap.inner_name or ""
    inner_args = unwrap.inner_args or {}
    return _execute_agent_tool(
        inner_name,
        inner_args,
        session,
        tool_call_id=tool_call_id,
        protocol_id=protocol_id,
        bridge_queue_ms=bridge_queue_ms,
        data_root=data_root,
        as_of=as_of,
        active_research_session_id=active_research_session_id,
        active_research_job_id=active_research_job_id,
    )


def _recorder_run_id(session: RuntimeToolSession) -> tuple[RunRecorder | None, str]:
    recorder = get_current_recorder()
    run_id = recorder.run_id if recorder is not None else f"runtime-{session.session_id}"
    return recorder, run_id


def _args_hash(arguments: object) -> str:
    if isinstance(arguments, dict):
        return _args_json(arguments)
    return json.dumps(str(arguments))


def _run_pre_gates(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    *,
    data_root: str | Path | None,
    as_of: str | None,
    active_research_session_id: str | None,
    active_research_job_id: str | None,
) -> tuple[_StagedContext, str, bool] | dict[str, object]:
    """Staged context + Gates 1/2/8 + run-budget reserve (returns deny outcome or gate state)."""
    _recorder, _run_id = _recorder_run_id(session)
    args_for_hash = _args_hash(arguments)
    # Staged context: immutable request-local IDs. Explicit canonical
    # arguments win, then the captured bridge pair, then session fields.
    # Resolved IDs never enter arguments/recorder/schema/handlers.
    staged = _resolve_staged_context(
        arguments,
        session,
        data_root=data_root,
        as_of=as_of,
        active_research_session_id=active_research_session_id,
        active_research_job_id=active_research_job_id,
    )
    # Attached staged session must exist for every target (fail closed);
    # discovery only skips the stage check, never existence.
    staged_error = _check_staged_session(name, staged)
    if staged_error is not None:
        return staged_error
    # Gate 1: RESEARCH-only permit filter; unlisted tools are denied.
    # Gate 2: schema validation + 8KB arg-bytes cap.
    pre_sid = staged.session_id or active_research_session_id or session.session_id
    permit_error = _check_permit_and_schema(name, arguments, session, args_for_hash, pre_sid)
    if permit_error is not None:
        return permit_error
    # Gate 8 (reserve): attached staged data dispatches consume one persisted
    # slot via the kernel; bookkeeping (discovery/evidence/reads/finalize)
    # never touches persisted counters and keeps the per-runtime-run budget below.
    # The outer call_tool wrapper returns before this point, so only the
    # inner call consumes.
    dispatch_consumed, dispatch_error = _consume_dispatch_budget(name, staged, arguments)
    if dispatch_error is not None:
        return dispatch_error
    _heartbeat_staged_job(name, staged)
    # one budget slot per call before any external work.
    # search_web draws from its dedicated pool, not the generic tool pool.
    # Session lock held only for the reserve; the handler below runs unlocked.
    if not _reserve_run_budget(name, session, dispatch_consumed):
        refusal = _run_budget_refusal(name, session)
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=run-budget reason=%s",
            _toolflow_sid(session, staged),
            name,
            refusal.get("error_type"),
        )
        return refusal
    return staged, args_for_hash, dispatch_consumed


def _execute_agent_tool(
    name: str,
    arguments: dict[str, object],
    session: RuntimeToolSession,
    *,
    tool_call_id: str | None = None,
    protocol_id: str | None = None,
    bridge_queue_ms: float = 0.0,
    data_root: str | Path | None = None,
    as_of: str | None = None,
    active_research_session_id: str | None = None,
    active_research_job_id: str | None = None,
) -> dict[str, object]:
    t0 = time.perf_counter()
    logger.info("toolflow entry sid=%s tool=%s", active_research_session_id or session.session_id, name)
    logger.debug(
        "toolflow entry_args sid=%s tool=%s keys=%s bytes=%s",
        active_research_session_id or session.session_id,
        name,
        *_toolflow_args_summary(arguments),
    )
    # Generic dispatch: call_tool validates then tail-calls the inner tool once.
    # The outer wrapper consumes no budget slot and writes no recorder row.
    tail = _dispatch_inner_call(
        name,
        arguments,
        session,
        tool_call_id=tool_call_id,
        protocol_id=protocol_id,
        bridge_queue_ms=bridge_queue_ms,
        data_root=data_root,
        as_of=as_of,
        active_research_session_id=active_research_session_id,
        active_research_job_id=active_research_job_id,
    )
    if tail is not None:
        return tail
    # Single-dispatch company-name support, schema-driven: any tool whose
    # schema declares company_name alongside a ticker/entity identifier
    # fills the missing identifier before validation. Cards keep the
    # identifier required (visible signal) while name-only dispatches
    # still execute; tools without company_name are untouched.
    if isinstance(arguments, dict):
        arguments = _resolve_company_arguments(name, arguments, active_research_session_id or session.session_id)
    pre = _run_pre_gates(
        name,
        arguments,
        session,
        data_root=data_root,
        as_of=as_of,
        active_research_session_id=active_research_session_id,
        active_research_job_id=active_research_job_id,
    )
    if isinstance(pre, dict):
        return pre
    staged, args_for_hash, _dispatch_consumed = pre
    recorder, run_id = _recorder_run_id(session)

    # Gate 3: intent firewall. No approval callback in this plan, so
    # portfolio-shaped calls are always denied (RESEARCH-only).
    # Gate 4: search_web egress; every other tool's private-pattern args check.
    gate_error = _check_intent_and_egress(name, arguments, session, args_for_hash)
    if gate_error is not None:
        return gate_error

    # Gate 6: LOCAL_CONTEXT only, never a broker context, in this plan.
    # Handler + rendering run OUTSIDE the session lock so calls overlap.
    # Gates above stay LOCAL_CONTEXT-based; only the final execute_tool
    # context carries the validated data_root override.
    (
        result,
        resolved_tc_id,
        status,
        meta,
        _error_type,
        _error_message,
        _cache_hit,
        _cache_type,
        _handler_ms,
    ) = _execute_and_record(
        name,
        arguments,
        session,
        run_id=run_id,
        recorder=recorder,
        tool_call_id=tool_call_id,
        protocol_id=protocol_id,
        bridge_queue_ms=bridge_queue_ms,
        data_root=data_root,
        as_of=as_of,
        research_session_id=staged.session_id,
    )

    failed_map = _failed_dict(result)
    sid = staged.session_id or session.session_id
    if failed_map is not None:
        _failed, soft, _denied, _status2, failed_error_type, _msg2 = _failure_outcome(failed_map)
        logger.warning(
            "toolflow handler_error sid=%s tool=%s err_type=%s dur_ms=%.1f",
            sid,
            name,
            failed_error_type,
            (time.perf_counter() - t0) * 1000.0,
        )
        return _failed_result_response(name, failed_map, failed_error_type, soft)
    if not isinstance(result, dict):
        logger.warning(
            "toolflow handler_error sid=%s tool=%s err_type=tool_error dur_ms=%.1f",
            sid,
            name,
            (time.perf_counter() - t0) * 1000.0,
        )
        return {
            "error": _unavailable_data_response([(name, {"error": "empty tool result"})]),
            "error_type": "tool_error",
        }
    rid = _persist_staged_tool_result(name, result, staged, resolved_tc_id)

    # Gate 5: ingress scan on the rendered evidence; quarantined or blocked
    # results are withheld from the agent with a fixed placeholder.
    # Gate 7 (success path): DLP over what the agent receives, then record evidence.
    # Session lock covers guard_response + label/budget mutations only.
    out = _finalize_success(
        name,
        result,
        session,
        run_id=run_id,
        recorder=recorder,
        resolved_tc_id=resolved_tc_id,
        status=status,
        meta=meta,
    )
    if rid is not None and isinstance(out, dict) and not out.get("error"):
        out["tool_result_id"] = rid
    dur_ms = (time.perf_counter() - t0) * 1000.0
    if isinstance(out, dict) and out.get("error"):
        logger.warning(
            "toolflow deny sid=%s tool=%s gate=%s dur_ms=%.1f", sid, name, out.get("error_type") or "ingress", dur_ms
        )
    else:
        logger.info("toolflow success sid=%s tool=%s dur_ms=%.1f", sid, name, dur_ms)
    return out


# ponytail: alias shim, drop once callers use RuntimeToolSession directly.
AgentToolSession = RuntimeToolSession


__all__ = [
    "AgentToolSession",
    "RuntimeToolSession",
    "ToolOutcome",
    "execute_agent_tool",
    "outcome_from_result",
]
