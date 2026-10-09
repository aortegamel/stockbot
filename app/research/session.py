"""ResearchSession lifecycle: create + deterministic transitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime

from .models import (
    JSONValue,
    ResearchSession,
    SessionStatus,
    default_budget,
    default_policy,
    new_session_id,
    normalize_time,
    resolve_source_policy,
    resolve_temporal_scope,
    utcnow,
    validate_json_mapping,
)

__all__ = [
    "TERMINAL_STATUSES",
    "TRANSITIONS",
    "can_transition",
    "create_session",
    "get_session",
    "is_terminal",
    "transition_session",
]

C = SessionStatus.CREATED.value
P = SessionStatus.PLANNING.value
R = SessionStatus.RESEARCHING.value
F = SessionStatus.FREEZING.value
A = SessionStatus.ANALYZING.value
T = SessionStatus.TARGETED_RESEARCH.value
S = SessionStatus.SYNTHESIZING.value
DONE = SessionStatus.COMPLETED.value
FAIL = SessionStatus.FAILED.value
CX = SessionStatus.CANCELLED.value

# Deterministic forward flow; FAILED/CANCELLED reachable from anywhere live.
TRANSITIONS: dict[str, frozenset[str]] = {
    C: frozenset({P, FAIL, CX}),
    P: frozenset({R, FAIL, CX}),
    R: frozenset({F, FAIL, CX}),
    F: frozenset({A, FAIL, CX}),
    A: frozenset({T, S, FAIL, CX}),
    T: frozenset({F, FAIL, CX}),
    S: frozenset({DONE, FAIL, CX}),
    DONE: frozenset(),
    FAIL: frozenset(),
    CX: frozenset(),
}

TERMINAL_STATUSES: frozenset[str] = frozenset({DONE, FAIL, CX})


def is_terminal(status: str) -> bool:
    """True when no outbound transition exists."""
    return status in TERMINAL_STATUSES


def can_transition(current: str, new_status: str) -> bool:
    """True when new_status is a legal successor of current."""
    return new_status in TRANSITIONS.get(current, frozenset())


def _parse_session_as_of(as_of: datetime | str | None) -> datetime | None:
    if isinstance(as_of, datetime):
        return normalize_time(as_of)
    if isinstance(as_of, str) and as_of.strip():
        try:
            return normalize_time(datetime.fromisoformat(as_of.strip()))
        except ValueError:
            raise ValueError(f"<session>: 'as_of' must be ISO-8601, got {as_of!r}") from None
    if as_of is not None:
        raise ValueError(f"<session>: 'as_of' must be ISO-8601, datetime, or null, got {as_of!r}")
    return None


def create_session(
    query: str,
    objective: str,
    *,
    as_of: datetime | str | None = None,
    session_id: str | None = None,
    policy: Mapping[str, object] | None = None,
    budget: dict[str, JSONValue] | None = None,
    temporal: str | None = None,
) -> ResearchSession:
    """Create a validated session in CREATED; persists source_policy + temporal_scope.

    source_policy derives from policy['research_sources'] ({mode: all|allowlist,
    sources}); absent key means the SEC-only allowlist default. temporal_scope
    folds as_of + natural-language time in temporal/query; no time info means
    latest-available with the cutoff set (None as_of only when explicitly unbounded).
    """
    if not query:
        raise ValueError("<session>: 'query' must be a non-empty string")
    if not objective:
        raise ValueError("<session>: 'objective' must be a non-empty string")
    now = utcnow()
    checked_policy = validate_json_mapping(policy, "<session>: 'policy'") if policy is not None else default_policy()
    parsed_as_of = _parse_session_as_of(as_of)
    temporal_scope = resolve_temporal_scope(as_of=parsed_as_of, temporal=temporal, query=query, now=now)
    if temporal_scope.get("mode") == "unbounded":
        parsed_as_of = None
    session = ResearchSession(
        session_id=session_id or new_session_id(),
        created_at=now,
        updated_at=now,
        query=query,
        objective=objective,
        as_of=parsed_as_of,
        status=C,
        policy=checked_policy,
        budget=validate_json_mapping(budget, "<session>: 'budget'") if budget is not None else default_budget(),
        source_policy=resolve_source_policy(checked_policy),
        temporal_scope=temporal_scope,
    )
    session.validate("<session>")
    return session


def transition_session(session: ResearchSession, new_status: str | SessionStatus) -> ResearchSession:
    """Move forward deterministically; raises ValueError on any illegal edge."""
    target = new_status.value if isinstance(new_status, SessionStatus) else new_status
    if target not in TRANSITIONS:
        raise ValueError(f"<session>: unknown status {new_status!r}")
    if not can_transition(session.status, target):
        raise ValueError(f"<session>: illegal transition {session.status!r} -> {target!r}")
    out = replace(session, status=target, updated_at=utcnow())
    out.validate("<session>")
    return out


def get_session(sessions: Mapping[str, ResearchSession], session_id: str) -> ResearchSession:
    """Look up one in-memory session; raises KeyError when absent."""
    try:
        return sessions[session_id]
    except KeyError:
        raise KeyError(f"unknown session_id: {session_id!r}") from None
