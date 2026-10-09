"""Programmatic first-hop router: deterministic select_round variant (default flow).

Order: state gates -> explicit phrase rules -> lexical scorer margin ->
JEV over top-5 only (reasoner/LLM only on genuine ambiguity, via the normal
JEV path). Plugs into the scheduler's select_round hook, same as
tool_catalogs.catalog_select_round; the kernel uses it unless
STOCKBOT_TOOLFLOW=catalog (two-step JEV) or full/whole/direct (whole registry).

Reuses: app.tools normalizer/scorer/ambiguity helpers, scheduler ticker,
company, accession, and record helpers, models.ToolDecision contract.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from operator import itemgetter
from typing import TYPE_CHECKING

from app.decision_client import JevClient
from app.research.models import JSONValue, ToolDecision

if TYPE_CHECKING:
    from app.research.scheduler import _Kernel

logger = logging.getLogger(__name__)

# ponytail: fixed margin/top-N; per-domain thresholds only if misroutes cluster.
_MARGIN = 4
_TOP_N = 5

_SHORT_TOKS = frozenset({"betting", "short", "shorting", "shorted", "bearish"})
_INSIDER_TOKS = frozenset({"insider", "executive", "traded", "trading", "bought", "sold"})
_ADVICE_VERBS = frozenset({"buy", "sell", "hold"})
_PLANNED_TOKS = frozenset({"planned", "proposed", "144"})
_EXECUTED_TOKS = frozenset({"executed", "actual", "actually", "trade", "traded", "sell", "sold", "sale"})
_READ_TOKS = frozenset({"revenue", "read", "text", "says", "mention", "section", "risk", "document"})
_TIME_TOKS = frozenset({"what", "time", "is", "it", "current"})
_RESEARCH_TOKS = frozenset(
    {
        "revenue",
        "filing",
        "filings",
        "sec",
        "finra",
        "sho",
        "short",
        "insider",
        "executive",
        "mention",
        "quarter",
        "volume",
        "ticker",
        "dividend",
        "10",
        "8",
        "10k",
        "10q",
        "8k",
    }
)

_Registry = list[dict[str, JSONValue]]
_Attempts = list[dict[str, JSONValue]]


_VARIANTS = {
    "search_sec_filings": "search_sec_filings_bounded",
    "find_sec_entities": "find_sec_entities_bounded",
}
_BOUNDED = {bounded: full for full, bounded in _VARIANTS.items()}
_COVERAGE_TOKS = frozenset({"all", "every", "each", "complete", "full", "list"})


def _wants_coverage(norm: str, toks: set[str]) -> bool:
    """Coverage ask (all/every/each/complete/full/list, who filed) needs the exhaustive variant."""
    return bool(toks & _COVERAGE_TOKS) or "who filed" in norm or "who actually filed" in norm


def _pick_variant(tool: str, norm: str, toks: set[str], names: set[str]) -> str:
    """One exhaustive/bounded pair, decided by code: coverage stays exhaustive, else bounded."""
    for full, bounded in _VARIANTS.items():
        if tool in (full, bounded):
            want = full if _wants_coverage(norm, toks) else bounded
            if want in names:
                return want
            other = bounded if want == full else full
            return other if other in names else tool
    return tool


def _variant_decision(decision: ToolDecision, norm: str, toks: set[str], names: set[str]) -> ToolDecision:
    """Rewrite a JEV pick inside a pair to the code-chosen variant; non-pair decisions pass through."""
    if decision.action != "invoke" or not decision.tool_names:
        return decision
    mapped = tuple(dict.fromkeys(_pick_variant(t, norm, toks, names) for t in decision.tool_names))
    if mapped == decision.tool_names:
        return decision
    name = decision.tool_name
    new_name = _pick_variant(name, norm, toks, names) if isinstance(name, str) else None
    probs = {_pick_variant(k, norm, toks, names): v for k, v in decision.probabilities.items()}
    out = ToolDecision(
        action="invoke",
        tool_name=new_name,
        tool_names=mapped,
        probabilities=probs,
        confidence=decision.confidence,
    )
    out.validate("<programmatic_router>")
    logger.info("toolflow programmatic_variant tool=%s -> %s", decision.tool_names, mapped)
    return out


def _signals(text: str) -> tuple[str, set[str]]:
    """Normalized text + keyword tokens (same helpers as _search_tools)."""
    from app.tools import _discovery_keywords, _normalize_discovery_text

    return " ".join(_normalize_discovery_text(text)), set(_discovery_keywords(text))


def _registry_names(registry: _Registry) -> set[str]:
    return {e.get("name") for e in registry if isinstance(e, dict) and isinstance(e.get("name"), str)}


def _ranked(norm: str, toks: set[str], names: set[str]) -> list[tuple[int, str]]:
    """Scorer hits restricted to the live registry, best first; one entry per variant pair."""
    from app.tools import _score_registry

    best: dict[str, int] = {}
    for s, n in _score_registry(norm, toks, None):
        if n not in names:
            continue
        key = _BOUNDED.get(n, n)
        if s > best.get(key, -(10**9)):
            best[key] = s
    return sorted(((s, n) for n, s in best.items()), key=itemgetter(0), reverse=True)


def _ambiguous(top: str, second: str, ranked: list[str]) -> bool:
    """True when top shares a conflicts_with ambiguity group with second."""
    from app.tools import _ambiguity_groups

    _, groups = _ambiguity_groups(ranked)
    for group in groups:
        cands = group.get("candidates")
        if isinstance(cands, list) and top in cands and second in cands:
            return True
    return False


def _short_tool(toks: set[str], norm: str, names: set[str]) -> str | None:
    """Short-family sub-dispatch; None when the right member is not registered."""
    if toks & {"daily", "volume", "venue", "sho", "reg"} or "reg sho" in norm:
        return "get_reg_sho_volume" if "get_reg_sho_volume" in names else None
    if toks & {"pressure", "outstanding", "relative", "ratio"}:
        return "get_short_pressure_profile" if "get_short_pressure_profile" in names else None
    if toks & {"trend", "distribution", "coverage", "briefing", "change"}:
        return "query_finra" if "query_finra" in names else None
    if toks & {"screen", "leaderboard"} or "most shorted" in norm:
        return "get_short_interest_leaderboard" if "get_short_interest_leaderboard" in names else None
    return "get_short_interest" if "get_short_interest" in names else None


def _has_accession(packets: list[JSONValue]) -> bool:
    """Any accession-like token across attempts/ctx packets (scheduler scanner)."""
    from app.research.scheduler import _scan_packet_accession

    for packet in packets:
        cands = (packet, packet.get("outcome_summary")) if isinstance(packet, dict) else (packet,)
        for cand in cands:
            if cand is None:
                continue
            found, _ = _scan_packet_accession(cand)
            if found is not None:
                return True
    return False


def _router_pick(
    text: str,
    norm: str,
    toks: set[str],
    names: set[str],
    ranked: list[tuple[int, str]],
    attempts: _Attempts,
    ctx_evidence: list[JSONValue],
) -> tuple[str, str] | None:
    """(tool, reason) or None when JEV must decide. Membership-gated: never picks outside the live registry."""
    from app.research.scheduler import _ACCESSION_CARRY_TOOLS, _objective_company, _objective_ticker

    # 1. Carry-only registry (force-open narrowed): open the packet accession.
    if names and names <= set(_ACCESSION_CARRY_TOOLS) and _has_accession([*attempts, *ctx_evidence]):
        if "get_sec_document" in names and toks & _READ_TOKS:
            return "get_sec_document", "carry:packet-doc"
        if "get_sec_filing" in names:
            return "get_sec_filing", "carry:packet-filing"
    # 2. Explicit Reg SHO daily volume.
    if "get_reg_sho_volume" in names and (
        "reg sho" in norm or ({"reg", "sho"} <= toks) or ({"daily", "short", "volume"} <= toks)
    ):
        return "get_reg_sho_volume", "sho-explicit"
    # 3. Short family ("betting against" never means anything else here).
    if toks & _SHORT_TOKS or "betting against" in norm:
        tool = _short_tool(toks, norm, names)
        if tool is not None:
            return tool, "short-family"
    # 4. Executed trades beat planned-sale notices unless planned is the ask.
    if toks & _INSIDER_TOKS or "form 4" in norm:
        if "get_insider_activity" in names and toks & _EXECUTED_TOKS:
            return "get_insider_activity", "insider-executed"
        if toks & _PLANNED_TOKS and "get_planned_insider_sales" in names:
            return "get_planned_insider_sales", "insider-planned"
        if "get_insider_activity" in names:
            return "get_insider_activity", "insider"
    # Form names tokenize to 10k/10q/8k; match merged tokens only.
    _FORM = bool({"10k", "10q", "8k"} & toks)
    if ("search_sec_filings" in names or "search_sec_filings_bounded" in names) and (
        (("filing" in toks or _FORM) and "mention" in norm) or "who filed" in norm or "who actually filed" in norm
    ):
        return _pick_variant("search_sec_filings", norm, toks, names), "person-mention"
    # 6. Quarterly revenue always starts from the filing list.
    if "revenue" in toks and "quarter" in toks and "list_sec_filings" in names:
        return "list_sec_filings", "revenue-quarter"
    # 7. Filing list for a resolved ticker/company.
    if (
        "filing" in toks
        and "list_sec_filings" in names
        and (_objective_ticker(text) is not None or _objective_company(text) is not None)
    ):
        return "list_sec_filings", "sec-list"
    # 8. Clock question.
    if toks and toks <= _TIME_TOKS and "get_current_time" in names and not (toks & _RESEARCH_TOKS):
        return "get_current_time", "time"
    # 9. Clear scorer winner outside any ambiguity group.
    if len(ranked) >= 2 and ranked[0][0] - ranked[1][0] >= _MARGIN:
        top = ranked[0][1]
        if not _ambiguous(top, ranked[1][1], [n for _, n in ranked[:_TOP_N]]):
            return top, "scorer-margin"
    return None


def _router_subset(ranked: list[tuple[int, str]], registry: _Registry, names: set[str]) -> _Registry | None:
    """Top-5 scorer subset for the JEV fallback; None when the scorer is blank or the registry is already small."""
    if not ranked:
        return None
    wanted = {n for _, n in ranked[:_TOP_N]}
    # _ranked collapses each pair under the exhaustive name; re-expand so JEV keeps both variants.
    for full, bounded in _VARIANTS.items():
        if full in wanted:
            wanted |= {full, bounded}
    wanted &= names
    if not wanted or len(wanted) >= len(names):
        return None
    return [e for e in registry if isinstance(e, dict) and e.get("name") in wanted]


def _SMALL_TALK(prompt: str) -> bool:
    """Exact small-talk only: thanks / never mind / ok variants create no session."""
    text = " ".join(prompt.strip().lower().split())
    return text in ("thanks", "thank you", "never mind", "nevermind", "ok", "okay", "got it", "great thanks")


def programmatic_route(prompt: object) -> str | None:
    """Entry fast-path: research signals -> research_required, else None (JEV decides)."""
    from app.research.scheduler import _ACCESSION_TOKEN_RE, _objective_company, _objective_ticker

    if not isinstance(prompt, str) or not prompt.strip():
        return "research_required"
    if _SMALL_TALK(prompt):
        return "no_session"
    norm, toks = _signals(prompt)
    if toks and toks <= _TIME_TOKS and not (toks & _RESEARCH_TOKS):
        return "get_current_time"
    if _ACCESSION_TOKEN_RE.search(prompt) is not None:
        return "research_required"
    if (
        (("should" in toks and toks & _ADVICE_VERBS) or "good investment" in norm)
        and (_objective_ticker(prompt) is not None or _objective_company(prompt) is not None)
        and not (toks & _RESEARCH_TOKS)
    ):
        return "reasoning_required"
    if _objective_ticker(prompt) is not None or _objective_company(prompt) is not None:
        return "research_required"
    if toks & _RESEARCH_TOKS or "reg sho" in norm or "betting against" in norm:
        return "research_required"
    return None


async def programmatic_select_round(
    jev: JevClient,
    kernel: _Kernel,
    sid: str,
    nid: str,
    session: Mapping[str, JSONValue],
    node: object,
    registry: _Registry,
    ctx_evidence: list[JSONValue],
    attempts: _Attempts,
) -> tuple[str, ToolDecision]:
    from app.research import scheduler as sched

    raw_objective = session.get("objective") or session.get("query") or ""
    objective = raw_objective if isinstance(raw_objective, str) else ""
    question = sched._f(sched._node_dict(node), "question", default="") or ""
    text = f"{objective} {question}".strip() if isinstance(question, str) else objective
    norm, toks = _signals(text)
    names = _registry_names(registry)
    ranked = _ranked(norm, toks, names)
    pick = _router_pick(text, norm, toks, names, ranked, attempts or [], ctx_evidence or [])
    if pick is not None:
        tool, reason = pick
        tool = _pick_variant(tool, norm, toks, names)
        decision = ToolDecision(
            action="invoke", tool_name=tool, tool_names=(tool,), probabilities={tool: 1.0}, confidence=None
        )
        decision.validate("<programmatic_router>")
        sched._record(
            kernel,
            sid,
            "tool_selection",
            candidates={tool: 1.0},
            probabilities={tool: 1.0},
            selected=tool,
            node_id=nid or None,
            job_id=None,
            confidence=None,
        )
        logger.info("toolflow programmatic sid=%s nid=%s tool=%s reason=%s", sid, nid, tool, reason)
        return "invoke", decision
    subset = _router_subset(ranked, registry, names)
    if subset is not None:
        logger.info(
            "toolflow programmatic_fallback sid=%s nid=%s subset=%s/%s",
            sid,
            nid,
            len(subset),
            len(registry),
        )
        action, decision = await sched._select_round(
            jev, kernel, sid, nid, session, node, subset, ctx_evidence, attempts
        )
    else:
        action, decision = await sched._select_round(
            jev, kernel, sid, nid, session, node, registry, ctx_evidence, attempts
        )
    return action, _variant_decision(decision, norm, toks, names)
