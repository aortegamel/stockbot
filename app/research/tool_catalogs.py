"""Tool-catalog selection: default two-step JEV path for the research scheduler.

Groups the JEV registry by tool-discovery domain so one small JEV choice
picks a catalog first; the normal select round then runs over the winning
subset only. The kernel worker routes the default flow through
catalog_select_round; STOCKBOT_TOOLFLOW=full/whole/direct keeps the old
whole-registry select as an escape hatch (programmatic keeps its own router).
"""

import logging
from collections.abc import Mapping
from typing import Protocol

from app.research.models import JSONValue

_OTHER_DESCRIPTION = "Other tools outside named domains."

logger = logging.getLogger(__name__)

# Catalog winner -> scheduler action (names mirror decision_client sentinels).
_SENTINEL_ACTIONS = {"reasoning_required": "reason", "node_resolved": "resolved"}


class _CatalogJev(Protocol):
    """Decide surface the catalog pick needs (JevClient satisfies this)."""

    async def decide(
        self,
        state: object,
        questions: Mapping[str, JSONValue],
        *,
        decision_type: str,
        session_id: str,
        node_id: str | None = None,
        job_id: str | None = None,
        choice_options: Mapping[str, Mapping[str, str]] | None = None,
    ) -> dict[str, dict[str, JSONValue]]: ...


_PICK_STATE_ITEMS = 5
_PICK_STATE_CHARS = 200


def _pick_state(items: object) -> list[JSONValue]:
    """Last-N pick state: tool/error/outcome only, outcome blobs truncated, never raises."""
    if not isinstance(items, list):
        return []
    kept: list[JSONValue] = []
    for item in reversed(items):
        if len(kept) >= _PICK_STATE_ITEMS:
            break
        if not isinstance(item, dict):
            continue
        row: dict[str, JSONValue] = {}
        for key in ("tool", "error", "error_type"):
            value = item.get(key)
            if isinstance(value, str) and value:
                row[key] = value[:_PICK_STATE_CHARS]
        summary = item.get("outcome_summary")
        if isinstance(summary, str) and summary:
            row["outcome_summary"] = " ".join(summary.split())[:_PICK_STATE_CHARS]
        kept.append(row)
    kept.reverse()
    return kept


def build_tool_catalogs(registry: list[dict[str, JSONValue]]) -> list[dict[str, JSONValue]]:
    """Group registry entries by TOOL_DISCOVERY_REGISTRY domain (unknown -> 'other')."""
    from app.tools import DOMAIN_DESCRIPTIONS, TOOL_DISCOVERY_REGISTRY

    buckets: dict[str, list[str]] = {}
    for entry in registry:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        meta = TOOL_DISCOVERY_REGISTRY.get(name)
        domain = meta.domain if meta is not None else "other"
        buckets.setdefault(domain, []).append(name)
    catalogs: list[dict[str, JSONValue]] = []
    for domain in sorted(buckets):
        tools: list[JSONValue] = []
        tools.extend(sorted(buckets[domain]))
        catalog: dict[str, JSONValue] = {
            "name": domain,
            "description": DOMAIN_DESCRIPTIONS.get(domain, _OTHER_DESCRIPTION),
            "tools": tools,
        }
        catalogs.append(catalog)
    return catalogs


def catalog_options(catalogs: list[dict[str, JSONValue]]) -> dict[str, str]:
    """Choice options for the catalog pick: sentinels + catalog name -> description."""
    from app.decision_client import _SENTINEL_DESCRIPTIONS

    options: dict[str, str] = dict(_SENTINEL_DESCRIPTIONS)
    for catalog in catalogs:
        name = catalog.get("name")
        description = catalog.get("description")
        if isinstance(name, str) and name and isinstance(description, str) and description and name not in options:
            options[name] = description
    return options


async def catalog_select_round(
    jev: _CatalogJev,
    kernel: object,
    sid: str,
    nid: str,
    session: dict[str, object],
    node: object,
    registry: list[dict[str, JSONValue]],
    ctx_evidence: list[JSONValue],
    attempts: list[dict[str, JSONValue]],
) -> tuple[str, object]:
    """Pick one catalog via jev.decide, then run the normal select over its tools."""
    from app.research import scheduler

    catalogs = build_tool_catalogs(registry)
    options = catalog_options(catalogs)
    raw_objective = session.get("objective") or session.get("query")
    objective = raw_objective if isinstance(raw_objective, str) else ""
    question = ""
    if isinstance(node, dict):
        raw_question = node.get("question")
        if isinstance(raw_question, str):
            question = raw_question
    else:
        attr_question = getattr(node, "question", None)
        if isinstance(attr_question, str):
            question = attr_question
    target = question or objective
    if target:
        instructions = (
            f"Select the one tool catalog (domain) most likely to advance this research question: {target[:500]}"
        )
    else:
        instructions = "Select the one tool catalog (domain) most likely to advance the current research question."
    criteria: dict[str, JSONValue] = {key: value for key, value in options.items()}
    inner: dict[str, JSONValue] = {"type": "choice", "instructions": instructions, "criteria": criteria}
    questions: dict[str, JSONValue] = {"catalog_selection": inner}
    state: dict[str, object] = {
        "objective": objective,
        "question": question,
        "catalogs": catalogs,
        "evidence": _pick_state(ctx_evidence),
        "attempts": _pick_state(attempts),
    }
    decisions = await jev.decide(
        state,
        questions,
        decision_type="catalog_selection",
        session_id=sid,
        node_id=nid,
        choice_options={"catalog_selection": options},
    )
    selected = decisions.get("catalog_selection")
    winner: object = selected.get("choice") if isinstance(selected, dict) else None
    if not isinstance(winner, str) or winner not in options:
        logger.warning(
            "toolflow catalog_fallback sid=%s nid=%s winner=%r catalogs=%d",
            sid,
            nid,
            winner,
            len(catalogs),
        )
        return await scheduler._select_round(jev, kernel, sid, nid, session, node, registry, ctx_evidence, attempts)
    action = _SENTINEL_ACTIONS.get(winner)
    if action is not None:
        logger.info(
            "toolflow catalog_pick sid=%s nid=%s catalogs=%d winner=%s subset=0 sentinel=%s",
            sid,
            nid,
            len(catalogs),
            winner,
            action,
        )
        probs: dict[str, object] = {}
        names: list[str] = []
        return action, {
            "action": action,
            "tool_name": None,
            "tool_names": names,
            "probabilities": probs,
            "confidence": None,
        }
    wanted: set[str] = set()
    for catalog in catalogs:
        if catalog.get("name") == winner:
            raw_tools = catalog.get("tools")
            if isinstance(raw_tools, list):
                wanted.update(tool for tool in raw_tools if isinstance(tool, str))
    filtered: list[dict[str, JSONValue]] = []
    for entry in registry:
        entry_name = entry.get("name")
        if isinstance(entry_name, str) and entry_name in wanted:
            filtered.append(entry)
    logger.info(
        "toolflow catalog_pick sid=%s nid=%s catalogs=%d winner=%s subset=%d",
        sid,
        nid,
        len(catalogs),
        winner,
        len(filtered),
    )
    return await scheduler._select_round(jev, kernel, sid, nid, session, node, filtered, ctx_evidence, attempts)
