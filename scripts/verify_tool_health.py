#!/usr/bin/env python3
"""Deterministic per-research-tool health gate (no LLM, no Pi subprocess).

For every registered research tool (derived from source via
scripts.verify_tool_registry.get_registry_sets, never a hand list), checks:
  schema      - function name/description present, parameters is a
                type:object dict, required keys all have properties.
  fixture     - minimal required-only args pass _validate_tool_arguments.
  dispatch    - canonical execute_tool path invokes the registered handler
                (proven by sentinel-swapping the handler entry; the live
                network is never needed for this proof).
  handler     - real handler body via canonical execute_tool with provider
                boundaries replaced by deterministic doubles (no network);
                same dict/serializable/structured-error shape as live.
  live        - execute_tool with fixture args returns a dict that
                json.dumps accepts; a structured {"error": ...} counts as
                pass (data variance is not a plumbing failure), but a raise,
                non-dict, non-serializable, or malformed error fails.
  security    - capability is RESEARCH, permitted under a RESEARCH context,
                denied ("not permitted") under an empty-capability context.
  errors      - missing required args return error_type
                "invalid_tool_arguments"; required-less tools must still
                reject non-object args via the canonical validator.

All invocation is programmatic via app.tools.execute_tool, never via Pi LLM.
Exit 0 when every tool passes, 1 with per-tool failures listed otherwise.
Stdlib + repo venv only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Protocol


class _PipeEnd(Protocol):
    """Pipe end shared by parent poll/recv and child send/close."""

    def poll(self, timeout: float | None = ...) -> bool: ...
    def recv(self) -> object: ...
    def send(self, obj: object) -> None: ...
    def close(self) -> None: ...


class _SendChannel(Protocol):
    """Pipe end the handler child sends its result dict through."""

    def send(self, obj: object) -> None: ...
    def close(self) -> None: ...


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import tools as tools_mod
from app.policy import Capability, RequestContext
from app.sec.discovery import service as sec_discovery_service
from app.tools import (
    _FINRA_HANDLERS,
    _MODEL_HANDLERS,
    _RESEARCH_HANDLERS,
    _ROBINHOOD_HANDLERS,
    _SEC_DISCOVERY_HANDLERS,
    _THESIS_HANDLERS,
    TOOL_CAPABILITIES,
    TOOLS,
    _canonical_tool_schema,
    _validate_tool_arguments,
    execute_tool,
    tool_is_permitted,
)
from scripts.verify_tool_registry import (
    get_registry_sets,
    tool_schema_function,
    tool_schema_name,
)

MODEL = "verify-tool-health"
# Per-tool-call cliff (handlers land in
# seconds; anything beyond this is hung, not slow).
LIVE_CALL_TIMEOUT_S = 120
THESIS_ID_TOOLS = frozenset({"thesis_show", "thesis_refine", "thesis_watch", "thesis_journal"})
_MONEY_PROPS = frozenset({"strike", "target_price", "strike_min", "strike_max"})
_THESIS_ID_KEYS = ("thesis_id", "id")

_STRING_DEFAULTS: dict[str, object] = {
    "ticker": "AAPL",
    "symbol": "AAPL",
    "identifier": "AAPL",
    "entity": "AAPL",
    "cik": "0000320193",
    "query": "Apple",
    "company_name": "Apple Inc.",
    "company_id": "Apple Inc.",
    "accession_no": "0000320193-25-000079",
    "current_accession": "0000320193-25-000079",
    "previous_accession": "0000320193-24-000123",
    "dataset": "otcMarket/consolidatedShortInterest",
    "dataset_id": "otcMarket/consolidatedShortInterest",
    "user_thesis": "Health-check thesis: NVDA AI demand stays strong.",
    "clarification": "AI datacenter capex keeps growing.",
    "body": "Operator note: still watching NVDA datacenter demand.",
    "statement_type": "income_statement",
    "concept": "NetIncomeLoss",
    "term": "Stanley",
}
_ARRAY_DEFAULTS: dict[str, list[str]] = {
    "fields": ["settlementDate", "currentShortPositionQuantity"],
    "geos": ["geoId/06"],
    "variables": ["Count_Person"],
    "assignees": ["Apple Inc."],
}
_DATE_DEFAULT = "2024-01-01"
_DATE_NAMES = frozenset(
    {"since", "start_date", "end_date", "as_of", "week_start", "week_end", "start_published_date", "end_published_date"}
)


def research_names() -> list[str]:
    return sorted(get_registry_sets()["research"])


def _is_date_prop(prop: str) -> bool:
    return prop in _DATE_NAMES or prop.endswith(("Date", "_date"))


def _string_for(prop: str) -> str:
    if prop == "expiration":
        return "2026-01-16"
    if _is_date_prop(prop):
        return _DATE_DEFAULT
    default = _STRING_DEFAULTS.get(prop, "health-check")
    return default if isinstance(default, str) else "health-check"


def _clamp_min(value: float, spec: dict[str, object]) -> float:
    minimum = spec.get("minimum")
    if isinstance(minimum, (int, float)) and value < minimum:
        return minimum
    return value


def _clamp_max(value: float, spec: dict[str, object]) -> float:
    maximum = spec.get("maximum")
    if isinstance(maximum, (int, float)) and value > maximum:
        return maximum
    return value


def _clamp(value: float, spec: dict[str, object]) -> float:
    return _clamp_max(_clamp_min(value, spec), spec)


def _spec_map(spec: object) -> dict[str, object] | None:
    if not isinstance(spec, dict):
        return None
    out: dict[str, object] = {str(k): v for k, v in spec.items()}
    return out


def _enum_first(spec_map: dict[str, object]) -> object | None:
    enum = spec_map.get("enum")
    if isinstance(enum, list) and enum:
        first: object = enum[0]
        return first
    return None


def _value_for_number(prop: str, spec_map: dict[str, object]) -> object:
    base = 100.0 if prop in _MONEY_PROPS else 1.5
    return _clamp(base, spec_map)


def _value_for_array(prop: str, spec_map: dict[str, object]) -> object:
    if prop in _ARRAY_DEFAULTS:
        return list(_ARRAY_DEFAULTS[prop])
    items = spec_map.get("items")
    if isinstance(items, dict):
        return [_value_for(prop, items)]
    return ["health-check"]


def _value_for_object(props: dict[str, object], spec_map: dict[str, object]) -> object:
    required = spec_map.get("required")
    if not isinstance(required, list):
        empty: dict[str, object] = {}
        return empty
    subkeys = [k for k in required if isinstance(k, str) and k in props]
    return {k: _value_for(k, props[k]) for k in subkeys}


def _value_for_typed(prop: str, kind: object, spec_map: dict[str, object]) -> object | None:
    if kind == "string":
        return _string_for(prop)
    if kind == "integer":
        return int(_clamp(5, spec_map))
    if kind == "number":
        return _value_for_number(prop, spec_map)
    if kind == "boolean":
        return True
    if kind == "array":
        return _value_for_array(prop, spec_map)
    if kind == "object":
        empty_obj: dict[str, object] = {}
        return empty_obj
    return None


def _value_for(prop: str, spec: object) -> object:
    spec_map = _spec_map(spec)
    if spec_map is None:
        return "health-check"
    first = _enum_first(spec_map)
    if first is not None:
        return first
    typed = _value_for_typed(prop, spec_map.get("type"), spec_map)
    if typed is not None:
        return typed
    properties = spec_map.get("properties")
    if isinstance(properties, dict):
        props: dict[str, object] = {str(k): v for k, v in properties.items()}
        return _value_for_object(props, spec_map)
    return "health-check"


def _params_properties(params: dict[str, object]) -> dict[str, object]:
    properties = params.get("properties")
    if not isinstance(properties, dict):
        none_props: dict[str, object] = {}
        return none_props
    return {str(k): v for k, v in properties.items()}


def _required_keys(params: dict[str, object]) -> list[str]:
    required = params.get("required")
    if not isinstance(required, list):
        return []
    return [k for k in required if isinstance(k, str)]


def fixture_for(params: dict[str, object]) -> dict[str, object]:
    props = _params_properties(params)
    return {k: _value_for(k, props.get(k)) for k in _required_keys(params)}


def _schema_shape_error(function: dict[str, object]) -> str | None:
    description = function.get("description")
    if not isinstance(description, str) or not description.strip():
        return "schema missing description"
    params = function.get("parameters")
    if not isinstance(params, dict) or params.get("type") != "object":
        return "parameters must be a type:object dict"
    if not isinstance(params.get("properties"), dict):
        return "parameters.properties must be a dict"
    return None


def _schema_parity_error(name: str, params: dict[str, object]) -> str | None:
    _, required, optional = _canonical_tool_schema(name)
    properties = params["properties"]
    props: dict[str, object] = properties if isinstance(properties, dict) else {}
    missing = [k for k in required if k not in props]
    if missing:
        return f"required without properties: {missing}"
    if set(optional) & set(required):
        return "required/optional overlap"
    return None


def check_schema(name: str, function: dict[str, object]) -> str | None:
    shape = _schema_shape_error(function)
    if shape is not None:
        return shape
    params = function["parameters"]
    assert isinstance(params, dict)
    return _schema_parity_error(name, params)


def handler_owner(name: str) -> object | None:
    for table in (
        _THESIS_HANDLERS,
        _MODEL_HANDLERS,
        _SEC_DISCOVERY_HANDLERS,
        _FINRA_HANDLERS,
        _ROBINHOOD_HANDLERS,
        _RESEARCH_HANDLERS,
    ):
        if name in table:
            return table
    return None


def _sentinel_call(name: str, calls: list[tuple[tuple[object, ...], dict[str, object]]]):
    def sentinel(*a: object, **k: object) -> dict[str, object]:
        calls.append((a, dict(k)))
        return {"ok": True, "tool": name}

    return sentinel


def _run_sentinel_dispatch(
    owner: dict[str, object], name: str, fixture: dict[str, object], ctx: RequestContext
) -> tuple[object, list[tuple[tuple[object, ...], dict[str, object]]]]:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real = owner[name]
    owner[name] = _sentinel_call(name, calls)
    try:
        return execute_tool(name, dict(fixture), MODEL, context=ctx), calls
    finally:
        owner[name] = real


def _sentinel_result_error(
    name: str, result: object, calls: list[tuple[tuple[object, ...], dict[str, object]]]
) -> str | None:
    if not calls:
        return "canonical execute_tool did not invoke handler"
    if not isinstance(result, dict) or result.get("ok") is not True:
        return f"dispatch sentinel result not returned: {str(result)[:200]}"
    return None


def check_dispatch(name: str, fixture: dict[str, object], ctx: RequestContext) -> str | None:
    owner = handler_owner(name)
    if not isinstance(owner, dict):
        return "no handler registered (registry parity broken)"
    result, calls = _run_sentinel_dispatch(owner, name, fixture, ctx)
    return _sentinel_result_error(name, result, calls)


# Handler-stage cliff: one missed network leg must not hang the default suite.
HANDLER_CALL_TIMEOUT_S = 60

# Tools whose required-only fixture dies in arg validation before the handler
# runs (found empirically by running each fixture through execute_tool).
EXTRA_FIXTURE_OVERRIDES: dict[str, dict[str, object]] = {
    "diff_sec_filings": {"ticker": "AAPL"},
    "search_sec_filings": {"query": "Apple"},
    "search_sec_filings_bounded": {"query": "Apple"},
    "get_finra_datapoints": {"ticker": "AAPL"},
}

# Pure-local tools: no provider seam to double, call the handler directly.
_LOCAL_HANDLER_TOOLS = frozenset(
    {
        "thesis_create",
        "thesis_show",
        "thesis_refine",
        "thesis_watch",
        "thesis_journal",
        "thesis_status",
        "get_sec_search_coverage",
        "find_alternative_signals",
        "get_macro_context",
        "get_trend_evidence",
        "search_company_patents",
    }
) | frozenset(_RESEARCH_HANDLERS)

# Google collectors check enabled flags first (trends.collect_trends:_bq_ready,
# datacommons.get_macro_context, patents.search_company_patents:_data_enabled,
# all via app/google_data/_lazy_config.py, read per call), so forcing the flag
# off makes them return fast disabled dicts with zero HTTP calls.
_GOOGLE_DISABLED_ENV_TOOLS = frozenset(
    {
        "get_macro_context",
        "get_trend_evidence",
        "search_company_patents",
    }
)


def _swap(target: object, attr: str, value: object, saved: list[tuple[object, str, object]]) -> None:
    saved.append((target, attr, getattr(target, attr)))
    setattr(target, attr, value)


def _swap_env(saved: list[tuple[str, str | None]], key: str, value: str) -> None:
    saved.append((key, os.environ.get(key)))
    os.environ[key] = value


def _fake_empty_list(*args: object, **kwargs: object) -> list[object]:
    return []


def _fake_empty_dict(*args: object, **kwargs: object) -> dict[str, object]:
    return {}


def _fake_search_envelope(*args: object, **kwargs: object) -> object:
    return SimpleNamespace(to_dict=dict)


def _fake_fixed_utc_now(*args: object, **kwargs: object) -> object:
    from datetime import UTC, datetime

    return datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _fake_entities_empty(*args: object, **kwargs: object) -> object:
    return SimpleNamespace(entities=[])


def _fake_exa_search(query: object = "", **kwargs: object) -> dict[str, object]:
    return {"result_type": "web_search", "query": query, "evidence": []}


class _FakeDiscoveryService:
    """Mirrors tests/test_sec_tools.py::_FakeService: serves one empty result."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def search(self, _request: object) -> object:
        return SimpleNamespace(to_dict=dict)


# Provider seams (what each handler calls one level down), never handler
# entries: check_dispatch already proves entry wiring via sentinel. Empty
# fakes keep the real coercion/envelope code running; a structured {"error"}
# from the handler counts as pass, same as check_live.
_SEAM_MAP: dict[str, list[tuple[object, str, object]]] = {
    "list_sec_filings": [(tools_mod.sec, "list_sec_filings", _fake_empty_list)],
    "get_sec_filing": [(tools_mod.sec, "get_sec_filing", _fake_search_envelope)],
    "list_sec_documents": [(tools_mod.sec, "list_sec_documents", _fake_empty_list)],
    "get_sec_document": [(tools_mod.sec, "get_sec_document", _fake_empty_dict)],
    "diff_sec_filings": [
        (tools_mod.sec, "list_sec_filings", _fake_empty_list),
        (tools_mod.sec, "diff_filings", _fake_empty_dict),
    ],
    "find_sec_entities": [(tools_mod.sec, "find_sec_entities", _fake_search_envelope)],
    "find_sec_entities_bounded": [(tools_mod.sec, "find_sec_entities", _fake_search_envelope)],
    "search_sec_filings": [(tools_mod.sec, "SECDiscoveryService", _FakeDiscoveryService)],
    "search_sec_filings_bounded": [(tools_mod.sec, "SECDiscoveryService", _FakeDiscoveryService)],
    "get_current_time": [(tools_mod, "_utc_now", _fake_fixed_utc_now)],
    "search_sec_relationships": [(tools_mod.sec, "search_sec_relationships", _fake_empty_dict)],
    "get_material_events": [(tools_mod.sec, "get_material_events", _fake_empty_list)],
    "get_beneficial_ownership": [(tools_mod.sec, "get_beneficial_ownership", _fake_empty_list)],
    "get_ownership_changes": [(tools_mod.sec, "get_ownership_changes", _fake_empty_list)],
    "get_insider_activity": [(tools_mod.sec, "get_insider_activity", _fake_empty_list)],
    "get_planned_insider_sales": [(tools_mod.sec, "get_planned_insider_sales", _fake_empty_list)],
    "get_offering_history": [(tools_mod.sec, "get_offering_history", _fake_empty_list)],
    "get_dilution_profile": [(tools_mod.sec, "get_dilution_profile", _fake_empty_dict)],
    "get_governance_events": [(tools_mod.sec, "get_governance_events", _fake_empty_list)],
    "get_transaction_status": [(tools_mod.sec, "get_transaction_status", _fake_empty_list)],
    "get_short_pressure_profile": [(tools_mod.sec, "get_short_pressure_context", _fake_empty_dict)],
    "get_recent_ownership_filings": [(tools_mod.edgar_client, "get_recent_ownership_filings", _fake_empty_dict)],
    "diff_risk_factors": [(tools_mod.edgar_client, "diff_risk_factors", _fake_empty_dict)],
    "get_financial_statements": [(tools_mod.edgar_client, "get_financial_statements", _fake_empty_dict)],
    "get_fundamentals": [(tools_mod.sec_facts, "get_fundamentals", _fake_empty_dict)],
    "get_xbrl_facts": [(tools_mod.sec_facts, "get_xbrl_facts", _fake_empty_dict)],
    "get_analyst_estimates": [(tools_mod.analyst_client, "get_analyst_estimates", _fake_empty_dict)],
    "get_sp500_weight": [(tools_mod.analyst_client, "get_sp500_weight", _fake_empty_dict)],
    "get_obligations": [(tools_mod.obligations, "get_obligations", _fake_empty_dict)],
    "get_valuation_metrics": [(tools_mod.valuation, "get_valuation_metrics", _fake_empty_dict)],
    "search_web": [(tools_mod.exa_client, "search", _fake_exa_search)],
    "query_finra": [(tools_mod.finra_client, "query_dataset", _fake_empty_dict)],
    "list_finra_datasets": [(tools_mod.finra_client, "list_datasets", _fake_empty_dict)],
    "describe_finra_dataset": [(tools_mod.finra_client, "describe_dataset", _fake_empty_dict)],
    "get_finra_datapoints": [(tools_mod.finra_client, "get_finra_datapoints", _fake_empty_dict)],
    "get_short_interest": [(tools_mod.finra_client, "get_short_interest", _fake_empty_dict)],
    "get_reg_sho_volume": [(tools_mod.finra_client, "get_reg_sho_volume", _fake_empty_dict)],
    "get_threshold_securities": [(tools_mod.finra_client, "get_threshold_securities", _fake_empty_dict)],
    "get_short_interest_leaderboard": [(tools_mod.screens, "get_short_interest_leaderboard", _fake_empty_dict)],
    "investigate_social_arbitrage_candidate": [(sec_discovery_service, "find_sec_entities", _fake_entities_empty)],
}


def _handler_swaps(name: str) -> list[tuple[object, str, object]] | None:
    """Provider seams to double for `name`; [] when purely local, None when unknown."""
    if name in _LOCAL_HANDLER_TOOLS:
        return []
    return _SEAM_MAP.get(name)


def _fail(conn: _SendChannel, reason: str) -> None:
    try:
        conn.send({"worker_ok": False, "reason": reason})
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
        pass


def _worker_context(principal_id: str, capability_names: list[str], data_root_str: str) -> RequestContext:
    return RequestContext(
        principal_id,
        frozenset({Capability(c) for c in capability_names}),
        data_root=Path(data_root_str),
    )


def _apply_worker_swaps(name: str, swaps: list[tuple[object, str, object]]) -> None:
    saved: list[tuple[object, str, object]] = []
    for target, attr, fake in swaps:
        _swap(target, attr, fake, saved)
    if name in _GOOGLE_DISABLED_ENV_TOOLS:
        _swap_env([], "GOOGLE_DATA_ENABLED", "")


def _install_worker_doubles(name: str) -> str | None:
    swaps = _handler_swaps(name)
    if swaps is None:
        return "no deterministic provider seam"
    _apply_worker_swaps(name, swaps)
    return None


def _run_worker_tool(conn: _SendChannel, name: str, args: dict[str, object], ctx: RequestContext) -> bool:
    try:
        result = execute_tool(name, args, MODEL, context=ctx)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _fail(conn, f"execute_tool raised {type(e).__name__}: {e}")
        return False
    try:
        conn.send({"worker_ok": True, "result": result})
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _fail(conn, f"handler result not sendable: {type(e).__name__}: {e}")
    return True


def _setup_worker_ctx(
    conn: _SendChannel, name: str, principal_id: str, capability_names: list[str], data_root_str: str
):
    try:
        ctx = _worker_context(principal_id, capability_names, data_root_str)
        missing = _install_worker_doubles(name)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _fail(conn, f"handler setup failed: {type(e).__name__}: {e}")
        return None, None
    return ctx, missing


def _handler_worker(
    conn: _SendChannel,
    name: str,
    args: dict[str, object],
    principal_id: str,
    capability_names: list[str],
    data_root_str: str,
) -> None:
    """Child-side handler run: install the doubles here, never in the parent."""
    try:
        ctx, missing = _setup_worker_ctx(conn, name, principal_id, capability_names, data_root_str)
        if ctx is None:
            return
        if missing is not None:
            _fail(conn, missing)
            return
        _run_worker_tool(conn, name, args, ctx)
    except Exception as e:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        _fail(conn, f"handler child failed: {type(e).__name__}: {e}")
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts; intentional silent skip
            pass


def _envelope_outcome(message: dict[str, object]) -> str | None:
    if message.get("worker_ok") is not True:
        reason = message.get("reason")
        if isinstance(reason, str):
            return f"handler worker failed: {reason}"
        return "handler worker failed: unknown reason"
    return None


def _result_json_error(result: dict[str, object]) -> str | None:
    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        return f"handler result not JSON-serializable: {e}"
    return None


def _result_error_shape(result: dict[str, object]) -> str | None:
    if not isinstance(result["error"], str) or not result["error"].strip():
        return "handler error is not a non-empty string"
    error_type = result.get("error_type")
    if error_type is not None and (not isinstance(error_type, str) or not error_type):
        return "handler error_type is not a string"
    return None


def _envelope_result_error(result: object) -> str | None:
    if not isinstance(result, dict):
        return f"handler result not a dict: {type(result).__name__}"
    json_err = _result_json_error(result)
    if json_err is not None:
        return json_err
    if "error" not in result:
        return None
    return _result_error_shape(result)


def _evaluate_envelope(message: object) -> str | None:
    if not isinstance(message, dict):
        return f"handler worker message not a dict: {type(message).__name__}"
    outcome = _envelope_outcome(message)
    if outcome is not None:
        return outcome
    return _envelope_result_error(message.get("result"))


def _handler_args(name: str, fixture: dict[str, object]) -> dict[str, object]:
    args = dict(fixture)
    extra = EXTRA_FIXTURE_OVERRIDES.get(name)
    if extra:
        args.update(extra)
    return args


def _handler_proc_args(name: str, args: dict[str, object], ctx: RequestContext) -> tuple[object, ...]:
    return (name, args, ctx.principal_id, [c.value for c in ctx.capabilities], str(ctx.data_root))


def _proc_alive(proc: object) -> bool:
    """Child liveness probe; False when the handle lacks is_alive."""
    alive = getattr(proc, "is_alive", None)
    return bool(alive()) if callable(alive) else False


def _proc_stop(proc: object) -> None:
    """Terminate + join via duck-typed handle; missing methods are no-ops."""
    terminate = getattr(proc, "terminate", None)
    join = getattr(proc, "join", None)
    if callable(terminate):
        terminate()
    if callable(join):
        join()


def _proc_start(proc: object) -> None:
    """Start via duck-typed handle."""
    start = getattr(proc, "start", None)
    if callable(start):
        start()


def _proc_close(proc: object) -> None:
    """Close via duck-typed handle."""
    close = getattr(proc, "close", None)
    if callable(close):
        close()


def _poll_once(parent_conn: _PipeEnd, proc: object, deadline: float) -> tuple[object, bool, bool]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, False, True
    if parent_conn.poll(min(1.0, remaining)):
        return parent_conn.recv(), True, False
    return None, False, not _proc_alive(proc)


def _drain_envelope(parent_conn: _PipeEnd, proc: object, deadline: float) -> tuple[object, bool]:
    while True:
        envelope, got, done = _poll_once(parent_conn, proc, deadline)
        if got:
            return envelope, True
        if done:
            return None, False


def _await_envelope(parent_conn: _PipeEnd, proc: object, deadline: float) -> tuple[object, bool]:
    envelope, got_envelope = _drain_envelope(parent_conn, proc, deadline)
    if not got_envelope and parent_conn.poll():
        envelope = parent_conn.recv()
        got_envelope = True
    return (envelope, got_envelope)


def _stop_hung_proc(proc: object) -> None:
    _proc_stop(proc)


def _reap_handler_proc(proc: object, missing: bool) -> str | None:
    if missing:
        if _proc_alive(proc):
            _stop_hung_proc(proc)
            return f"timed out after {HANDLER_CALL_TIMEOUT_S}s"
        return "handler child produced no result"
    join = getattr(proc, "join", None)
    if callable(join):
        join(10)
    if _proc_alive(proc):
        _stop_hung_proc(proc)
    return None


def _spawn_handler_proc(
    mp_ctx: object, name: str, fixture: dict[str, object], ctx: RequestContext
) -> tuple[_PipeEnd, object]:
    make_pipe = getattr(mp_ctx, "Pipe")  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    make_proc = getattr(mp_ctx, "Process")  # noqa: B009 - dynamic boundary, no stubs; getattr keeps checker green
    parent_conn, child_conn = make_pipe(duplex=False)
    proc = make_proc(
        target=_handler_worker,
        args=(child_conn, *_handler_proc_args(name, _handler_args(name, fixture), ctx)),
    )
    _proc_start(proc)
    child_conn.close()
    return parent_conn, proc


def _wait_handler_result(parent_conn: _PipeEnd, proc: object) -> str | None:
    envelope, got_envelope = _await_envelope(parent_conn, proc, time.monotonic() + HANDLER_CALL_TIMEOUT_S)
    timeout = _reap_handler_proc(proc, not got_envelope)
    if timeout is not None:
        return timeout
    return _evaluate_envelope(envelope)


def _close_handler_proc(parent_conn: _PipeEnd, proc: object) -> None:
    parent_conn.close()
    if _proc_alive(proc):
        _proc_stop(proc)
    _proc_close(proc)


def check_handler(name: str, fixture: dict[str, object], ctx: RequestContext) -> str | None:
    """Execute the REAL handler via canonical execute_tool with provider doubles.

    installs nothing, so a hanging handler dies with terminate() and there is
    no parent-side seam to restore on timeout.
    """
    if _handler_swaps(name) is None:
        return "no deterministic provider seam"
    mp_ctx = multiprocessing.get_context("spawn")
    parent_conn, proc = _spawn_handler_proc(mp_ctx, name, fixture, ctx)
    try:
        return _wait_handler_result(parent_conn, proc)
    finally:
        _close_handler_proc(parent_conn, proc)


def _submit_live(
    pool: concurrent.futures.ThreadPoolExecutor,
    name: str,
    fixture: dict[str, object],
    ctx: RequestContext,
):
    return pool.submit(execute_tool, name, dict(fixture), MODEL, context=ctx)


def _live_shape_error(result: object) -> str | None:
    if not isinstance(result, dict):
        return f"live result not a dict: {type(result).__name__}"
    try:
        json.dumps(result)
    except (TypeError, ValueError) as e:
        return f"live result not JSON-serializable: {e}"
    return None


def _live_error_text_error(result: dict[str, object]) -> str | None:
    if not isinstance(result["error"], str) or not result["error"].strip():
        return "live error is not a non-empty string"
    error_type = result.get("error_type")
    if error_type is not None and (not isinstance(error_type, str) or not error_type):
        return "live error_type is not a string"
    return None


def _live_error_shape_error(result: dict[str, object]) -> str | None:
    if "error" not in result:
        return None
    return _live_error_text_error(result)


def _invoke_live(
    pool: concurrent.futures.ThreadPoolExecutor, name: str, fixture: dict[str, object], ctx: RequestContext
):
    try:
        result = _submit_live(pool, name, fixture, ctx).result(timeout=LIVE_CALL_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return None, f"live call exceeded {LIVE_CALL_TIMEOUT_S}s"
    except (
        Exception  # noqa: BLE001 - intentional best-effort boundary, never aborts
    ) as e:  # execute_tool contract: never raises
        return None, f"live call raised {type(e).__name__}: {e}"
    return result, None


def check_live(
    name: str,
    fixture: dict[str, object],
    ctx: RequestContext,
    pool: concurrent.futures.ThreadPoolExecutor,
) -> str | None:
    result, error = _invoke_live(pool, name, fixture, ctx)
    if error is not None:
        return error
    shape = _live_shape_error(result)
    if shape is not None:
        return shape
    assert isinstance(result, dict)
    return _live_error_shape_error(result)


def _is_denied_surface(denied: object) -> bool:
    return isinstance(denied, dict) and "not permitted" in str(denied.get("error", ""))


def _deny_surface_error(name: str, denied: object) -> str | None:
    if _is_denied_surface(denied):
        return None
    return f"empty-capability call not denied: {str(denied)[:200]}"


def _security_capability_error(name: str) -> str | None:
    if TOOL_CAPABILITIES.get(name) is not Capability.RESEARCH:
        return f"capability is {TOOL_CAPABILITIES.get(name)!r}, want RESEARCH"
    return None


def check_security(name: str, fixture: dict[str, object], ctx: RequestContext, deny: RequestContext) -> str | None:
    cap_err = _security_capability_error(name)
    if cap_err is not None:
        return cap_err
    if not tool_is_permitted(name, ctx):
        return "denied under RESEARCH context"
    denied = execute_tool(name, dict(fixture), MODEL, context=deny)
    return _deny_surface_error(name, denied)


def _missing_args_error(name: str, ctx: RequestContext) -> str | None:
    bad = execute_tool(name, {}, MODEL, context=ctx)
    if not isinstance(bad, dict) or not isinstance(bad.get("error"), str):
        return "missing-args call has no structured error"
    if bad.get("error_type") != "invalid_tool_arguments":
        return f"missing-args error_type={bad.get('error_type')!r}"
    if bad.get("tool") != name:
        return "missing-args error missing tool name"
    return None


def _no_required_args_error(name: str) -> str | None:
    message = _validate_tool_arguments(name, ["not-a-dict"])
    if not isinstance(message, str) or not message.strip():
        return "non-dict args not rejected by validator"
    return None


def check_errors(name: str, params: dict[str, object], ctx: RequestContext) -> str | None:
    if _required_keys(params):
        return _missing_args_error(name, ctx)
    return _no_required_args_error(name)


def _thesis_create_result(ctx: RequestContext) -> object:
    thesis_text = _STRING_DEFAULTS["user_thesis"]
    return execute_tool(
        "thesis_create",
        {"user_thesis": thesis_text if isinstance(thesis_text, str) else "health-check thesis"},
        MODEL,
        context=ctx,
    )


def _first_thesis_id(result: dict[str, object]) -> str | None:
    for key in _THESIS_ID_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    nested = result.get("thesis")
    if isinstance(nested, dict):
        for key in _THESIS_ID_KEYS:
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def bootstrap_thesis_id(ctx: RequestContext) -> str | None:
    try:
        result = _thesis_create_result(ctx)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if not isinstance(result, dict):
        return None
    return _first_thesis_id(result)


def _verify_fixture(name: str, params_dict: dict[str, object], thesis_id: str | None) -> dict[str, object]:
    fixture = fixture_for(params_dict)
    if name in THESIS_ID_TOOLS and thesis_id:
        fixture["id"] = thesis_id
    return fixture


def _verify_core_stages(name: str, fixture: dict[str, object], ctx: RequestContext, failures: list[str]) -> None:
    if (invalid := _validate_tool_arguments(name, fixture)) is not None:
        failures.append(f"fixture: {invalid}")
    if (problem := check_dispatch(name, fixture, ctx)) is not None:
        failures.append(f"dispatch: {problem}")
    if (problem := check_handler(name, fixture, ctx)) is not None:
        failures.append(f"handler: {problem}")


def _verify_live_security(
    name: str,
    fixture: dict[str, object],
    ctx: RequestContext,
    deny: RequestContext,
    pool: concurrent.futures.ThreadPoolExecutor,
    live: bool,
    failures: list[str],
) -> None:
    if live and (problem := check_live(name, fixture, ctx, pool)) is not None:
        failures.append(f"live: {problem}")
    if (problem := check_security(name, fixture, ctx, deny)) is not None:
        failures.append(f"security: {problem}")


def _verify_stages(
    name: str,
    fixture: dict[str, object],
    ctx: RequestContext,
    deny: RequestContext,
    pool: concurrent.futures.ThreadPoolExecutor,
    live: bool,
) -> list[str]:
    failures: list[str] = []
    _verify_core_stages(name, fixture, ctx, failures)
    _verify_live_security(name, fixture, ctx, deny, pool, live, failures)
    return failures


def verify_one(
    name: str,
    function: dict[str, object],
    ctx: RequestContext,
    deny: RequestContext,
    pool: concurrent.futures.ThreadPoolExecutor,
    thesis_id: str | None,
    live: bool = False,
) -> list[str]:
    failures: list[str] = []
    params = function.get("parameters")
    params_dict = params if isinstance(params, dict) else {}
    if (problem := check_schema(name, function)) is not None:
        return [f"schema: {problem}"]
    fixture = _verify_fixture(name, params_dict, thesis_id)
    failures.extend(_verify_stages(name, fixture, ctx, deny, pool, live))
    if (problem := check_errors(name, params_dict, ctx)) is not None:
        failures.append(f"errors: {problem}")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic per-research-tool health gate (fast default; --live runs real handlers)."
    )
    parser.add_argument("--tool", action="append", default=None, help="check one tool (repeatable)")
    parser.add_argument("--list", action="store_true", help="list research tools and exit 0")
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    parser.add_argument(
        "--live", action="store_true", help="also execute real handlers (integration, may be slow/flaky)"
    )
    return parser.parse_args(argv)


def _selected_names(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    names = research_names()
    selected = sorted(set(args.tool)) if args.tool else names
    unknown = [n for n in selected if n not in set(names)]
    return selected, unknown


def _schema_table() -> dict[str, dict[str, object]] | None:
    by_name: dict[str, dict[str, object]] = {}
    for raw in TOOLS:
        try:
            by_name[tool_schema_name(raw)] = dict(tool_schema_function(raw))
        except RuntimeError as e:
            print(f"FAIL schema for {raw!r}: {e}")
            return None
    return by_name


def _verify_contexts(tmp: str) -> tuple[RequestContext, RequestContext]:
    ctx = RequestContext("verify-tool-health", frozenset({Capability.RESEARCH}), data_root=Path(tmp))
    deny = RequestContext("verify-tool-health-deny", frozenset(), data_root=Path(tmp))
    return ctx, deny


def _run_all_selected(
    selected: list[str],
    by_name: dict[str, dict[str, object]],
    ctx: RequestContext,
    deny: RequestContext,
    pool: concurrent.futures.ThreadPoolExecutor,
    thesis_id: str | None,
    live: bool,
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for name in selected:
        results[name] = verify_one(name, by_name[name], ctx, deny, pool, thesis_id, live)
    return results


def _run_selected(
    selected: list[str],
    by_name: dict[str, dict[str, object]],
    live: bool,
) -> dict[str, list[str]]:
    with TemporaryDirectory(prefix="tool-health-") as tmp:
        ctx, deny = _verify_contexts(tmp)
        thesis_id = bootstrap_thesis_id(ctx) if any(n in THESIS_ID_TOOLS for n in selected) else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return _run_all_selected(selected, by_name, ctx, deny, pool, thesis_id, live)


def _check_unknown_tool(ctx_ctx: RequestContext) -> bool:
    confused = execute_tool("verify-health-no-such-tool", {}, MODEL, context=ctx_ctx)
    # Unknown names are denied before dispatch, so the contract (mirroring
    # tests/test_tool_routing.py::test_unknown_tool_executes_nothing) is a
    # structured error naming the tool — never a raise, never a dispatch.
    if not isinstance(confused, dict) or "verify-health-no-such-tool" not in str(confused.get("error", "")):
        print(f"FAIL unknown-tool error surface: {str(confused)[:200]}")
        return True
    return False


def _report_tool_line(name: str, problems: list[str] | None) -> None:
    if problems:
        for problem in problems:
            print(f"FAIL {name} [{problem.split(':')[0]}] {problem}")
    else:
        print(f"PASS {name}")


def _report_text(selected: list[str], results: dict[str, list[str]]) -> int:
    failed = {name: problems for name, problems in results.items() if problems}
    for name in selected:
        _report_tool_line(name, failed.get(name))
    print(f"tool health: {len(results) - len(failed)}/{len(results)} pass")
    return 1 if failed else 0


def _report_results(args: argparse.Namespace, selected: list[str], results: dict[str, list[str]]) -> int:
    if args.json:
        failed = {name: problems for name, problems in results.items() if problems}
        print(json.dumps({"tools": results, "passed": len(results) - len(failed), "total": len(results)}, indent=2))
        return 1 if failed else 0
    return _report_text(selected, results)


def _probe_unknown_tool() -> bool:
    from tempfile import TemporaryDirectory as _TD

    with _TD(prefix="tool-health-unknown-") as tmp:
        probe_ctx = RequestContext("verify-tool-health", frozenset({Capability.RESEARCH}), data_root=Path(tmp))
        return _check_unknown_tool(probe_ctx)


def _run_main_checks(args: argparse.Namespace, selected: list[str]) -> int | None:
    by_name = _schema_table()
    if by_name is None:
        return 1
    results = _run_selected(selected, by_name, args.live)
    if _probe_unknown_tool():
        return 1
    return _report_results(args, selected, results)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for name in research_names():
            print(name)
        return 0
    selected, unknown = _selected_names(args)
    if unknown:
        print(f"unknown tool(s): {unknown}", file=sys.stderr)
        return 2
    result = _run_main_checks(args, selected)
    assert result is not None
    return result


if __name__ == "__main__":
    sys.exit(main())
