"""Runtime-neutral JSONL bridge for Stockbot tool consumers.

The bridge is transport only. It exposes canonical tool schemas and executes
tools through app.tool_runtime.

Protocol, one JSON object per line:
  {"id": str, "op": "describe"}
    -> {"id": str, "tools": [...]}
  {"id": str, "op": "tool.invoke", "name": str, "arguments": dict,
   "session_id": str, "tool_call_id": str | null,
   "data_root": str | null, "as_of": str | null}
    -> {"id": str, "result": {...}}
  {"id": str, "op": "tool.session.end", "session_id": str}
    -> {"id": str, "result": {"ended": bool}}

Sessions are explicit and cached so budgets/security state persist across the
tool sequence. This bridge is deliberately synchronous for now: Needle already
serializes its local routing session, and parallelism can be added here later
without changing the public protocol.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.policy import Capability
from app.tool_runtime import AgentToolSession, execute_agent_tool
from app.tools import tools_for_capabilities

_sessions: dict[str, AgentToolSession] = {}
_sessions_lock = threading.Lock()

logger = logging.getLogger(__name__)


def _required_string(request: Mapping[str, object], key: str) -> str | None:
    value = request.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def _get_session(session_id: str) -> AgentToolSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            session = AgentToolSession(session_id=session_id)
            _sessions[session_id] = session
        return session


def _describe(protocol_id: str) -> dict[str, object]:
    return {
        "id": protocol_id,
        "tools": tools_for_capabilities(frozenset({Capability.RESEARCH})),
    }


def _invoke(request: Mapping[str, object], protocol_id: str) -> dict[str, object]:
    name = _required_string(request, "name")
    session_id = _required_string(request, "session_id")
    arguments = request.get("arguments", {})
    if name is None or session_id is None or not isinstance(arguments, dict):
        return {"id": protocol_id, "error": "missing_arg"}

    raw_tool_call_id = request.get("tool_call_id")
    tool_call_id = raw_tool_call_id if isinstance(raw_tool_call_id, str) and raw_tool_call_id else None
    raw_data_root = request.get("data_root")
    data_root = raw_data_root if isinstance(raw_data_root, str) and raw_data_root else None
    raw_as_of = request.get("as_of")
    as_of = raw_as_of if isinstance(raw_as_of, str) and raw_as_of else None

    try:
        result = execute_agent_tool(
            name,
            arguments,
            _get_session(session_id),
            tool_call_id=tool_call_id,
            protocol_id=protocol_id,
            data_root=data_root,
            as_of=as_of,
        )
        return {"id": protocol_id, "result": result}
    except Exception:
        logger.exception("tool bridge failed id=%s tool=%s", protocol_id, name)
        return {"id": protocol_id, "error": "bridge_failed"}


def _end_session(request: Mapping[str, object], protocol_id: str) -> dict[str, object]:
    session_id = _required_string(request, "session_id")
    if session_id is None:
        return {"id": protocol_id, "error": "missing_arg"}
    with _sessions_lock:
        ended = _sessions.pop(session_id, None) is not None
    return {"id": protocol_id, "result": {"ended": ended}}


def handle(request: Mapping[str, object]) -> dict[str, object]:
    protocol_id = _required_string(request, "id")
    if protocol_id is None:
        return {"error": "missing_arg"}
    op = request.get("op")
    if op == "describe":
        return _describe(protocol_id)
    if op == "tool.invoke":
        return _invoke(request, protocol_id)
    if op == "tool.session.end":
        return _end_session(request, protocol_id)
    return {"id": protocol_id, "error": "unknown_op"}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            response: dict[str, object] = {"error": "bad_request"}
        else:
            response = handle(request) if isinstance(request, dict) else {"error": "bad_request"}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
