"""Stage-gated capabilities: derive research stage, enforce per-stage tool sets.

Kernel-authoritative gate (TS mirrors for UX only). Canonical tool names only.
"""

from __future__ import annotations

from typing import Literal

from app.policy import Capability
from app.tools import tools_for_capabilities

Stage = Literal["SOURCE_RESEARCH", "COMMITTEE", "FINAL"]


def _schema_name(tool: dict[str, object]) -> str | None:
    """OpenAI schema function name (TOOLS entries are untyped app-side JSON)."""
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return name if isinstance(name, str) else None
    return None


# Canonical agent-visible set: single source of truth stays in app/tools.py.
RESEARCH_TOOL_NAMES: frozenset[str] = frozenset(
    name
    for tool in tools_for_capabilities(frozenset({Capability.RESEARCH}))
    for name in [_schema_name(tool)]
    if name is not None
)

DISCOVERY_TOOLS: frozenset[str] = frozenset(
    {
        "browse_tools",
        "search_tools",
        "describe_tool",
        "list_tool_domains",
        "call_tool",
    }
)

CONTROL_TOOLS: frozenset[str] = frozenset(
    {
        "research_start",
        "research_resume",
        "research_status",
        "research_cancel",
        "research_read",
        "research_read_search",
        "research_add_evidence",
        "research_submit_source_result",
        "research_add_analysis",
        "research_finalize",
    }
)

_THESIS_TOOLS: frozenset[str] = frozenset(
    {
        "thesis_create",
        "thesis_show",
        "thesis_refine",
        "thesis_watch",
        "thesis_journal",
        "thesis_status",
    }
)

# Data dispatches: everything research-capable except discovery, thesis
# actions, and local research controls. Thesis actions are local store writes,
# so they run in SOURCE_RESEARCH without consuming a dispatch slot.
DISPATCH_TOOLS: frozenset[str] = frozenset(RESEARCH_TOOL_NAMES - DISCOVERY_TOOLS - CONTROL_TOOLS - _THESIS_TOOLS)

STAGE_ALLOW: dict[Stage, frozenset[str]] = {
    "SOURCE_RESEARCH": DISCOVERY_TOOLS
    | DISPATCH_TOOLS
    | _THESIS_TOOLS
    | frozenset(
        {
            "research_resume",
            "research_status",
            "research_read",
            "research_read_search",
            "research_cancel",
            "research_add_evidence",
            "research_submit_source_result",
        }
    ),
    "COMMITTEE": DISCOVERY_TOOLS
    | frozenset(
        {
            "research_resume",
            "research_status",
            "research_read",
            "research_cancel",
            "research_add_analysis",
        }
    ),
    "FINAL": DISCOVERY_TOOLS
    | frozenset(
        {
            "research_resume",
            "research_status",
            "research_read",
            "research_cancel",
            "research_finalize",
        }
    ),
}


def _field(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, dict):
        out: object = obj.get(key, default)
        return out
    fell: object = getattr(obj, key, default)
    return fell


def _latest_freeze_id(session: object) -> str | None:
    freeze_ids = _field(session, "freeze_ids", [])
    if not isinstance(freeze_ids, list) or not freeze_ids:
        return None
    fid = freeze_ids[-1]
    return fid if isinstance(fid, str) else None


def _entry_job_ids(entry: dict[str, object], fid: str) -> list[str]:
    if entry.get("freeze_id") != fid:
        return []
    jobs_raw: object = entry.get("jobs")
    jobs: list[object] = jobs_raw if isinstance(jobs_raw, list) else []
    return [jid for jid in jobs if isinstance(jid, str)]


def _wanted_trio_jobs(session: object, fid: str) -> set[str]:
    wanted: set[str] = set()
    runs = _field(session, "committee_runs", [])
    if not isinstance(runs, list):
        return wanted
    for entry in runs:
        if isinstance(entry, dict):
            wanted.update(_entry_job_ids(entry, fid))
    return wanted


def _job_index(jobs: object) -> dict[str, object]:
    by_id: dict[str, object] = {}
    if isinstance(jobs, list):
        for job in jobs:
            jid = _field(job, "job_id")
            if isinstance(jid, str):
                by_id[jid] = job
    return by_id


def _completed_job_roles(jobs: object, wanted: set[str]) -> set[str]:
    by_id = _job_index(jobs)
    roles: set[str] = set()
    for jid in wanted:
        job = by_id.get(jid)
        if job is not None and _field(job, "status") == "completed":
            roles.add(str(_field(job, "job_type")))
    return roles


def _trio_complete(session: object, jobs: object) -> bool:
    fid = _latest_freeze_id(session)
    if fid is None:
        return False
    roles = _completed_job_roles(jobs, _wanted_trio_jobs(session, fid))
    return {"stockbot", "bullbot", "bearbot"} <= roles


def _final_or_committee(session: object, jobs: object) -> Stage:
    return "FINAL" if _trio_complete(session, jobs) else "COMMITTEE"


def _terminal_stage(status: str) -> Stage | None:
    if status in ("SYNTHESIZING", "COMPLETED"):
        return "FINAL"
    if status == "TARGETED_RESEARCH":
        return "SOURCE_RESEARCH"
    return None


def _active_stage(status: str, session: object, jobs: object) -> Stage:
    freeze_ids = _field(session, "freeze_ids", [])
    frozen = isinstance(freeze_ids, list) and bool(freeze_ids)
    if frozen or status in ("FREEZING", "ANALYZING"):
        return _final_or_committee(session, jobs)
    return "SOURCE_RESEARCH"


def stage_for_session(session: object, jobs: object = ()) -> Stage:
    """Derive stage: terminal -> FINAL; targeted research -> SOURCE; trio-complete -> FINAL."""
    status = str(_field(session, "status", "") or "").upper()
    terminal = _terminal_stage(status)
    if terminal is not None:
        return terminal
    return _active_stage(status, session, jobs)


def check_stage_tool(stage: Stage, tool_name: str) -> None:
    """Raise ValueError when tool_name is forbidden in stage. Discovery always passes."""
    if tool_name in DISCOVERY_TOOLS:
        return
    allowed = STAGE_ALLOW.get(stage)
    if allowed is not None and tool_name in allowed:
        return
    raise ValueError(f"Stage {stage} forbids tool '{tool_name}'")


__all__ = [
    "CONTROL_TOOLS",
    "DISCOVERY_TOOLS",
    "DISPATCH_TOOLS",
    "RESEARCH_TOOL_NAMES",
    "STAGE_ALLOW",
    "Stage",
    "check_stage_tool",
    "stage_for_session",
]
