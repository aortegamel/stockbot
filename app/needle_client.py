"""Needle worker: grounded arguments, structured extraction, retrieval embeddings.

JEV owns ALL selection; this worker never selects, chains, or judges
sufficiency. It talks to needle-harness/lib/needle/server.py over JSONL on a
persistent child ({id,action,...} <-> {id,...}) so weights load once.
Serial single-flight (parallel decoders contend: 10.5s serial vs 71s parallel
on realistic prompts). Missing server/weights raise; never stub.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import select
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

from app.child_spawn import registering_child
from app.research.models import JSONValue, validate_json_mapping, validate_json_value

logger = logging.getLogger(__name__)

__all__ = [
    "close",
    "embed",
    "extract_fields",
    "generate_arguments",
    "similarity_order",
    "start",
    "validate_needle_tool",
]

_TIMEOUT_S = 120.0
_STDERR_TAIL_CHARS = 2000
_REAP_TIMEOUT_S = 2.0

# ponytail: single-flight lock; per-request lanes if Needle throughput matters.
_LOCK = threading.Lock()
_TAIL_LOCK = threading.Lock()
_PROC: subprocess.Popen[str] | None = None
_STDERR_TAIL = ""
_ATEXIT_ARMED = False
# Real Popen captured at import: test doubles swapped in later never count as owned groups.
_REAL_POPEN = subprocess.Popen


class _StartupTimeout(RuntimeError, TimeoutError):
    """Startup/ping silence inside the 120s budget; distinct from a generation read timeout."""


def validate_needle_tool(jev_tool: str, needle_tool: object) -> str:
    """Exact-match gate: Needle must emit the JEV-selected tool (None rejects)."""
    if not isinstance(jev_tool, str) or not jev_tool:
        raise ValueError("jev_tool must be a nonempty tool name")
    if not isinstance(needle_tool, str) or needle_tool != jev_tool:
        logger.warning("toolflow needle_tool_mismatch jev_tool=%s needle_tool=%.50s", jev_tool, str(needle_tool))
        raise ValueError(f"needle tool mismatch: jev selected {jev_tool!r}, needle emitted {needle_tool!r}")
    return needle_tool


_NODE_KEYS = ("question", "node_id", "id", "type")
_CONTEXT_KEYS = ("as_of", "temporal_scope", "today_utc", "ticker", "grounded_arguments")
_SCOPE_KEYS = ("mode", "as_of", "start", "end")


def _sid_hint(req_dict: dict[str, object]) -> str:
    """Best-effort sid for toolflow logs: node/context session_id, else '-'."""
    for key in ("node", "context"):
        raw = req_dict.get(key)
        if isinstance(raw, dict):
            sid = raw.get("session_id")
            if isinstance(sid, str) and sid.strip():
                return sid.strip()
    return "-"


def _is_scalar(value: object) -> bool:
    """JSON scalar without non-finite floats, so NaN/Infinity never enter Needle."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_argument_value(value: object) -> bool:
    """Admit scalars and scalar lists only; records and nested structures never enter."""
    return _is_scalar(value) or (isinstance(value, list) and all(_is_scalar(item) for item in value))


def _filter_scope(scope: object) -> dict[str, JSONValue] | None:
    """Persisted temporal window without generated wording: string window fields only."""
    if not isinstance(scope, dict):
        return None
    out: dict[str, JSONValue] = {}
    for key in _SCOPE_KEYS:
        value = scope.get(key)
        if isinstance(value, str):
            out[key] = value
    return out


def _filter_grounded_arguments(values: object, schema: object) -> dict[str, JSONValue] | None:
    """Explicit schema properties only; malformed, off-schema, or nested values never enter."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(values, dict) or not isinstance(properties, dict):
        return None
    out: dict[str, JSONValue] = {}
    for key, value in values.items():
        if key in properties and _is_argument_value(value):
            out[key] = validate_json_value(value, "<needle_client>: 'grounded_arguments'")
    return out


def _filter_node(node: object) -> JSONValue | None:
    """Request text plus node metadata only; research records and feedback never enter."""
    if not isinstance(node, dict):
        return validate_json_value(node, "<needle_client>: 'node'") if node is not None else None
    return {key: validate_json_value(node[key], "<needle_client>: 'node'") for key in _NODE_KEYS if key in node}


def _filter_context(context: object, schema: object) -> JSONValue | None:
    """Execution context only; evidence, attempts, feedback, and intake never enter."""
    if not isinstance(context, dict):
        return validate_json_value(context, "<needle_client>: 'context'") if context is not None else None
    out: dict[str, JSONValue] = {}
    for key in _CONTEXT_KEYS:
        if key not in context:
            continue
        value = _context_value(key, context, schema)
        if value is not None:
            out[key] = value
    return out


def _context_value(key: str, context: dict[str, object], schema: object) -> JSONValue | None:
    """One filtered context key; None drops a key that is absent or filtered out."""
    if key == "temporal_scope":
        return _filter_scope(context[key])
    if key == "grounded_arguments":
        return _filter_grounded_arguments(context[key], schema)
    value = context[key]
    return value if isinstance(value, str) else None


def _server() -> Path:
    return Path(__file__).resolve().parent.parent / "needle-harness" / "lib" / "needle" / "server.py"


def _python() -> str:
    # Mirror of VENV_PYTHON in needle-harness/lib/needle/client.ts.
    venv = Path(os.path.expanduser("~/.cache/needle-harness/.needle/bin/python"))
    if venv.exists():
        return str(venv)
    return sys.executable


def _ensure_weights() -> None:
    if os.environ.get("NEEDLE_WEIGHTS"):
        return
    blob = Path(__file__).resolve().parent.parent / "needle3.cact"
    if not blob.exists():
        raise RuntimeError(f"needle weights missing: {blob} (gap: set NEEDLE_WEIGHTS or provide needle3.cact)")


def _append_tail(chunk: str) -> None:
    global _STDERR_TAIL
    with _TAIL_LOCK:
        _STDERR_TAIL = (_STDERR_TAIL + chunk)[-_STDERR_TAIL_CHARS:]


def _tail_text() -> str:
    with _TAIL_LOCK:
        return _STDERR_TAIL


def _drain_stderr(proc: subprocess.Popen[str]) -> None:
    stream = proc.stderr
    if stream is None:
        return
    try:
        while True:
            chunk: object = stream.readline()
            if not isinstance(chunk, str):
                break
            if chunk == "":
                break
            _append_tail(chunk)
    except OSError:
        pass


def _owned_pgid(proc: subprocess.Popen[str]) -> int | None:
    """Owned group id: a real, unreaped child that leads its own session; None otherwise."""
    # ponytail: an already-reaped leader skips the group kill (its pid may be reused);
    # descendants of a self-crashed leader can outlive it. Track pgid at spawn if that matters.
    if not isinstance(proc, _REAL_POPEN) or proc.returncode is not None:
        return None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return None
    return pgid if pgid == proc.pid else None


def _destroy_proc(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the owned tree then reap; never raises, never blocks past reap budget."""
    pgid = _owned_pgid(proc)
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=_REAP_TIMEOUT_S)
    except OSError, subprocess.TimeoutExpired:
        pass


def _teardown() -> None:
    try:
        proc = _PROC
        if proc is not None:
            _destroy_proc(proc)
    except OSError:
        pass


def _arm_atexit() -> None:
    global _ATEXIT_ARMED
    if _ATEXIT_ARMED:
        return
    _ATEXIT_ARMED = True
    atexit.register(_teardown)


def _spawn_locked() -> subprocess.Popen[str]:
    """Spawn the worker; caller holds _LOCK. Raises RuntimeError when spawning fails."""
    global _PROC
    with registering_child():
        try:
            # start_new_session is POSIX-only: Windows accepts and ignores it.
            proc: subprocess.Popen[str] = subprocess.Popen(
                [_python(), str(_server())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1"},
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"needle spawn failed: {exc}") from exc
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            _destroy_proc(proc)
            raise RuntimeError("needle spawn failed: missing pipes")
        _PROC = proc
        _arm_atexit()
    threading.Thread(target=_drain_stderr, args=(proc,), daemon=True).start()
    return proc


def _kill_locked() -> None:
    """Kill the owned tree and reap; caller holds _LOCK. Next use respawns."""
    global _PROC
    proc, _PROC = _PROC, None
    if proc is None:
        return
    _destroy_proc(proc)


def _readable_selected(stream: IO[str], timeout_s: float) -> list[IO[str]] | None:
    """Select result narrowed to a list; None when the stream is not selectable."""
    empty_w: list[IO[str]] = []
    empty_x: list[IO[str]] = []
    try:
        selected = select.select([stream], empty_w, empty_x, timeout_s)[0]
    except OSError, ValueError, TypeError:
        # Fakes without fileno (tests) read directly; real pipes always select.
        return None
    return selected if isinstance(selected, list) else []


def _readline(proc: subprocess.Popen[str], timeout_s: float) -> str:
    """One select-bounded stdout line; TimeoutError on silence, ConnectionError on close."""
    out = proc.stdout
    if out is None:
        raise RuntimeError("needle worker unavailable (gap: server stdout missing)")
    selected = _readable_selected(out, timeout_s)
    if selected is not None and not selected:
        raise TimeoutError(f"needle worker silent after {timeout_s:g}s")
    line: object = out.readline()
    if line == "":
        raise ConnectionError("needle worker closed (EOF)")
    if not isinstance(line, str):
        raise TypeError("needle server stdout must be text (gap: open the child with text=True)")
    return line


def _ping_locked(proc: subprocess.Popen[str]) -> None:
    """Gate one child: it must echo our ping id with ready:true. Fail closed."""
    rid = f"needle:{uuid.uuid4().hex[:12]}"
    payload: dict[str, JSONValue] = {"id": rid, "action": "ping"}
    inp = proc.stdin
    if inp is None:
        raise RuntimeError("needle ping failed: server stdin missing")
    inp.write(json.dumps(payload) + "\n")
    inp.flush()
    raw_line = _readline(proc, _TIMEOUT_S)
    try:
        decoded: object = json.loads(raw_line)
    except ValueError as exc:
        raise RuntimeError(f"needle ping failed: malformed server response ({exc})") from exc
    if not isinstance(decoded, dict) or decoded.get("id") != rid or decoded.get("ready") is not True:
        raise RuntimeError("needle ping failed: bad ack")


def _ensure_locked() -> subprocess.Popen[str]:
    """Live child, spawn + ping-gating a fresh one only when needed. Caller holds _LOCK."""
    proc = _PROC
    if proc is not None and proc.poll() is None:
        return proc
    _kill_locked()
    proc = _spawn_locked()
    try:
        _ping_locked(proc)
    except TimeoutError as exc:
        _kill_locked()
        raise _StartupTimeout(
            f"needle startup timed out after {_TIMEOUT_S:g}s; stderr tail: {_tail_text() or '(empty)'}"
        ) from exc
    return proc


def _exchange_response_locked(
    proc: subprocess.Popen[str], payload: dict[str, JSONValue], failure: str
) -> dict[str, JSONValue]:
    """Write one request and validate its correlated response envelope."""
    inp = proc.stdin
    if inp is None:
        raise RuntimeError(f"{failure}: server stdin missing")
    inp.write(json.dumps(payload) + "\n")
    inp.flush()
    raw_line = _readline(proc, _TIMEOUT_S)
    try:
        decoded: object = json.loads(raw_line)
    except ValueError as exc:
        raise RuntimeError(f"{failure}: malformed server response") from exc
    if not isinstance(decoded, dict) or decoded.get("id") != payload["id"]:
        raise RuntimeError(f"{failure}: malformed server response")
    resp = validate_json_mapping(decoded, "<needle_client>: 'server'")
    if resp.get("error"):
        raise RuntimeError(f"{failure}: {resp['error']}")
    return resp


def _exchange_locked(proc: subprocess.Popen[str], tool: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """One arguments.generate round trip with id correlation; malformed/error raises."""
    resp = _exchange_response_locked(proc, payload, f"needle arguments.generate failed for {tool!r}")
    validate_needle_tool(tool, resp.get("tool"))
    arguments = resp.get("arguments", {})
    if not isinstance(arguments, dict):
        raise TypeError(f"needle arguments for {tool!r} must be a mapping, got {type(arguments).__name__}")
    return {
        "tool": tool,
        "arguments": validate_json_mapping(arguments, "<needle_client>: 'arguments'"),
        "withheld": resp.get("withheld") is True,
        "reasoning": str(resp.get("reasoning") or ""),
        "confidence": resp.get("confidence"),
    }


def _exchange_extract_locked(
    proc: subprocess.Popen[str], record_name: str, payload: dict[str, JSONValue]
) -> dict[str, JSONValue]:
    """One extract round trip; record mismatch or ungrounded values raise."""
    resp = _exchange_response_locked(proc, payload, f"needle extract failed for {record_name!r}")
    if resp.get("record") is not None and resp.get("record") != record_name:
        raise RuntimeError(f"needle extract failed for {record_name!r}: malformed server response")
    fields = resp.get("fields", {})
    if not isinstance(fields, dict):
        raise TypeError(f"needle extract fields for {record_name!r} must be a mapping")
    return {
        "record": resp.get("record"),
        "fields": validate_json_mapping(fields, "<needle_client>: 'fields'"),
        "reasoning": str(resp.get("reasoning") or ""),
        "confidence": resp.get("confidence"),
    }


def _exchange_embed_locked(proc: subprocess.Popen[str], payload: dict[str, JSONValue]) -> list[float]:
    """One embed round trip; malformed vectors raise."""
    resp = _exchange_response_locked(proc, payload, "needle embed failed")
    vec = resp.get("embedding", [])
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("needle embed failed: malformed server response")
    out: list[float] = []
    for v in vec:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("needle embed failed: malformed server response")
        out.append(float(v))
    return out


def start() -> None:
    """Idempotent gate: reuse the live child, else spawn + ping once."""
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            return
        _kill_locked()
        proc = _spawn_locked()
        try:
            _ping_locked(proc)
        except TimeoutError as exc:
            _kill_locked()
            raise _StartupTimeout(
                f"needle startup timed out after {_TIMEOUT_S:g}s; stderr tail: {_tail_text() or '(empty)'}"
            ) from exc
        except OSError as exc:
            _kill_locked()
            raise RuntimeError(f"needle ping failed: {exc}") from exc
        except RuntimeError:
            _kill_locked()
            raise


def close() -> None:
    """Idempotent shutdown: kill the owned tree and reap; the next call respawns."""
    with _LOCK:
        _kill_locked()


def _arguments_request(args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
    """Normalize the two supported argument-generation calling forms."""
    if not args:
        return dict(kwargs)
    if len(args) != 1 or not isinstance(args[0], dict) or kwargs:
        raise TypeError("generate_arguments: pass kwargs(tool, schema, ...) or a single request dict")
    return {str(k): v for k, v in args[0].items()}


def generate_arguments(*args: object, **kwargs: object) -> dict[str, JSONValue]:
    """Fill arguments for one JEV-selected tool. Accepts kwargs or one request dict."""
    req_dict = _arguments_request(args, kwargs)
    tool = req_dict.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError("generate_arguments: tool must be a nonempty tool name")
    sid_hint = _sid_hint(req_dict)
    logger.info("toolflow needle_gen_entry sid=%s tool=%s", sid_hint, tool)
    logger.debug(
        "toolflow needle_gen_objective sid=%s tool=%s objective=%.200s",
        sid_hint,
        tool,
        str(req_dict.get("objective") or ""),
    )
    server = _server()
    if not server.exists():
        logger.warning("toolflow needle_gen_error sid=%s tool=%s err_type=missing-server", sid_hint, tool)
        raise RuntimeError(f"needle arguments.generate unavailable (gap: server missing: {server})")
    try:
        _ensure_weights()
    except Exception as exc:
        logger.warning("toolflow needle_gen_error sid=%s tool=%s err_type=%s", sid_hint, tool, type(exc).__name__)
        raise
    where = "<needle_client>"
    schema = validate_json_value(req_dict.get("schema"), where)
    payload: dict[str, JSONValue] = {
        "id": f"needle:{uuid.uuid4().hex[:12]}",
        "action": "arguments.generate",
        "tool": tool,
        "schema": schema,
        "objective": validate_json_value(req_dict.get("objective"), where),
        "node": _filter_node(req_dict.get("node")),
        "context": _filter_context(req_dict.get("context"), schema),
    }
    with _LOCK:
        out = _generate_locked(tool, payload, sid_hint)
        conf = out.get("confidence")
        logger.info(
            "toolflow needle_gen_ok sid=%s tool=%s conf=%s",
            sid_hint,
            tool,
            "yes" if isinstance(conf, (int, float)) and not isinstance(conf, bool) else "no",
        )
        return out


def _generate_locked(tool: str, payload: dict[str, JSONValue], sid_hint: str) -> dict[str, JSONValue]:
    """Generate arguments with one restart and the existing operation-specific error logs."""
    attempt = 0
    while True:
        try:
            proc = _ensure_locked()
            return _exchange_locked(proc, tool, payload)
        except _StartupTimeout as exc:
            logger.warning("toolflow needle_gen_error sid=%s tool=%s err_type=StartupTimeout", sid_hint, tool)
            raise RuntimeError(
                f"needle arguments.generate startup timed out after {_TIMEOUT_S:g}s for {tool!r}; "
                f"stderr tail: {_tail_text() or '(empty)'}"
            ) from exc
        except TimeoutError as exc:
            _kill_locked()
            logger.warning("toolflow needle_gen_error sid=%s tool=%s err_type=TimeoutError", sid_hint, tool)
            raise RuntimeError(
                f"needle arguments.generate timed out after {_TIMEOUT_S:g}s for {tool!r}; "
                f"stderr tail: {_tail_text() or '(empty)'}"
            ) from exc
        except OSError as exc:
            # One bounded restart on BrokenPipe/EOF (ConnectionError rides via OSError).
            _kill_locked()
            if attempt >= 1:
                logger.warning(
                    "toolflow needle_gen_error sid=%s tool=%s err_type=%s", sid_hint, tool, type(exc).__name__
                )
                raise RuntimeError(
                    f"needle arguments.generate failed for {tool!r}: server unavailable ({exc})"
                ) from exc
            attempt += 1
        except Exception as exc:
            logger.warning(
                "toolflow needle_gen_error sid=%s tool=%s err_type=%s err=%.200s",
                sid_hint,
                tool,
                type(exc).__name__,
                exc,
            )
            raise


def _restartable_dict(
    label: str, exchange: Callable[[subprocess.Popen[str]], dict[str, JSONValue]]
) -> dict[str, JSONValue]:
    """One dict exchange with a single bounded restart on BrokenPipe/EOF."""
    attempt = 0
    while True:
        try:
            return exchange(_ensure_locked())
        except _StartupTimeout:
            raise
        except TimeoutError as exc:
            _kill_locked()
            raise RuntimeError(
                f"needle {label} timed out after {_TIMEOUT_S:g}s; stderr tail: {_tail_text() or '(empty)'}"
            ) from exc
        except OSError as exc:
            _kill_locked()
            if attempt >= 1:
                raise RuntimeError(f"needle {label} failed: server unavailable ({exc})") from exc
            attempt += 1


def _restartable_vec(label: str, exchange: Callable[[subprocess.Popen[str]], list[float]]) -> list[float]:
    """One vector exchange with a single bounded restart on BrokenPipe/EOF."""
    attempt = 0
    while True:
        try:
            return exchange(_ensure_locked())
        except _StartupTimeout:
            raise
        except TimeoutError as exc:
            _kill_locked()
            raise RuntimeError(
                f"needle {label} timed out after {_TIMEOUT_S:g}s; stderr tail: {_tail_text() or '(empty)'}"
            ) from exc
        except OSError as exc:
            _kill_locked()
            if attempt >= 1:
                raise RuntimeError(f"needle {label} failed: server unavailable ({exc})") from exc
            attempt += 1


def _check_extract_options(max_new_tokens: object, system: object, strict: object) -> None:
    """Validate decoding options after the record and passage checks."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 4096:
        raise TypeError("extract_fields: max_new_tokens must be an int in 1..4096")
    if system is not None and not isinstance(system, str):
        raise TypeError("extract_fields: system must be a string or None")
    if not isinstance(strict, bool):
        raise TypeError("extract_fields: strict must be a bool")


def extract_fields(
    record: dict[str, object],
    passage: str,
    *,
    system: str | None = None,
    max_new_tokens: int = 512,
    strict: bool = True,
) -> dict[str, JSONValue]:
    """Extract typed fields for one record shape from messy text.

    Single-record binding only — never the full registry. Grammar guarantees
    the parse; ungrounded values raise. JEV owns what to extract and whether
    the fields suffice; this returns bytes only.
    """
    if not isinstance(record, dict) or not isinstance(record.get("name"), str) or not record["name"]:
        raise ValueError("extract_fields: record must carry a nonempty name")
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("extract_fields: passage must be a nonempty string")
    _check_extract_options(max_new_tokens, system, strict)
    if not _server().exists():
        raise RuntimeError(f"needle extract unavailable (gap: server missing: {_server()})")
    _ensure_weights()
    record_name = str(record["name"])
    where = "<needle_client>"
    payload: dict[str, JSONValue] = {
        "id": f"needle:{uuid.uuid4().hex[:12]}",
        "action": "extract",
        "record": validate_json_value(record, where),
        "passage": passage,
        "system": validate_json_value(system, where),
        "max_new_tokens": max_new_tokens,
        "strict": strict,
    }
    with _LOCK:

        def _run_extract(proc: subprocess.Popen[str]) -> dict[str, JSONValue]:
            return _exchange_extract_locked(proc, record_name, payload)

        return _restartable_dict(f"extract for {record_name!r}", _run_extract)


def embed(text: str) -> list[float]:
    """One retrieval vector off the shared agent (serial like arguments.generate)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("embed: text must be a nonempty string")
    if not _server().exists():
        raise RuntimeError(f"needle embed unavailable (gap: server missing: {_server()})")
    _ensure_weights()
    payload: dict[str, JSONValue] = {"id": f"needle:{uuid.uuid4().hex[:12]}", "action": "embed", "text": text}

    def _run_embed(proc: subprocess.Popen[str]) -> list[float]:
        return _exchange_embed_locked(proc, payload)

    with _LOCK:
        return _restartable_vec("embed", _run_embed)


_RERANK_MAX_TEXTS = 50
_RERANK_MAX_CHARS = 1000


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either norm is zero or dims differ."""
    if len(left) != len(right):
        return 0.0
    dot = sum(x * y for x, y in zip(left, right))
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(y * y for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def similarity_order(
    texts: Sequence[str],
    reference: str,
    *,
    max_texts: int = _RERANK_MAX_TEXTS,
    max_chars: int = _RERANK_MAX_CHARS,
) -> list[int] | None:
    """Order of indices by descending cosine similarity to the reference; None keeps order.

    Bounded (at most max_texts texts, each truncated to max_chars) and total:
    blank reference, fewer than two texts, an over-cap set, or any embed
    failure returns None so the caller keeps its deterministic order. Blank
    texts sink; ties keep input order under a stable sort.
    """
    items = list(texts or [])
    if not _nonblank_reference(reference) or len(items) < 2 or len(items) > max_texts:
        return None
    try:
        ref_vec = embed(reference.strip()[:max_chars])
        scored = [(_text_similarity(ref_vec, text, max_chars), pos) for pos, text in enumerate(items)]
        scored.sort(key=_score_key, reverse=True)
        return [pos for _, pos in scored]
    except Exception:  # noqa: BLE001 - embedding is a tiebreak; failure keeps deterministic order
        return None


def _nonblank_reference(reference: object) -> bool:
    """True only for a string reference with non-whitespace text."""
    return isinstance(reference, str) and bool(reference.strip())


def _text_similarity(reference: Sequence[float], text: str, max_chars: int) -> float:
    """Score one bounded text. Blank texts sort after every embedded text."""
    if not isinstance(text, str) or not text.strip():
        return float("-inf")
    return _cosine(reference, embed(text.strip()[:max_chars]))


def _score_key(pair: tuple[float, int]) -> float:
    """Sort key for one (similarity, position) pair."""
    return pair[0]
