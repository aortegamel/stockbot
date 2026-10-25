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
import re
from collections.abc import Mapping, Sequence
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
_SUBJECT_ADVICE_TOKS = _ADVICE_VERBS | {
    "advice",
    "recommend",
    "recommendation",
    "buying",
    "selling",
    "pick",
    "picks",
    "invest",
    "investing",
}
_SUBJECT_FILLER_TOKS = frozenset(
    {
        "can",
        "you",
        "give",
        "me",
        "some",
        "any",
        "what",
        "should",
        "i",
        "a",
        "the",
        "to",
        "now",
        "today",
        "please",
        "good",
    }
)
_PLANNED_TOKS = frozenset({"planned", "proposed", "144"})
_EXECUTED_TOKS = frozenset({"executed", "actual", "actually", "trade", "traded", "sell", "sold", "sale"})
_READ_TOKS = frozenset({"revenue", "read", "text", "says", "say", "mention", "section", "risk", "document"})
_TIME_TOKS = frozenset({"what", "time", "is", "it", "current"})
# Match explicit create commands, not embedded mentions or negative instructions.
_THESIS_CREATE_RE = re.compile(r"^(?:please |can you |could you )?create (?:an? )?(?:new )?thesi\b")
_THESIS_REFINE_RE = re.compile(r"^(?:please |can you |could you )?refine (?:my |the |a )?thesi\b")
# ponytail: process-local, one entry per thesis-create session; concurrent nodes and the pass-2 gap
# node share the objective, so only the first selection creates. Persist it if workers ever multiply.
_THESIS_CREATED: set[str] = set()
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
# ponytail: closed vocabulary per intent anchor; widen a set when a real single-intent phrasing misses.
_SINGLE_SHOT_FILLER = frozenset({"latest", "last", "recent", "quarter"})
_SINGLE_SHOT_INTENTS = {
    "short": frozenset({"short", "interest"}),
    "insider": frozenset({"insider", "sale", "sell", "sold", "bought", "trading", "traded", "activity"}),
    "revenue": frozenset({"revenue", "quarterly"}),
    "risk": frozenset({"risk", "factor"}),
}

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


def _registry_names(registry: Sequence[Mapping[str, JSONValue]]) -> set[str]:
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
    """Select the first short-family signal, even if its tool is unavailable."""
    tool = "get_short_interest"
    if toks & {"daily", "volume", "venue", "sho", "reg"} or "reg sho" in norm:
        tool = "get_reg_sho_volume"
    elif toks & {"pressure", "outstanding", "relative", "ratio"}:
        tool = "get_short_pressure_profile"
    elif toks & {"trend", "distribution", "coverage", "briefing", "change"}:
        tool = "query_finra"
    elif toks & {"screen", "leaderboard"} or "most shorted" in norm:
        tool = "get_short_interest_leaderboard"
    return tool if tool in names else None


def _has_accession(packets: Sequence[JSONValue], session_id: str | None = None, issuer: object = None) -> bool:
    """Recognize structured accession rows, never summaries or content."""
    from app.research.scheduler import _structured_accession_present

    return _structured_accession_present(packets, session_id, issuer)


def _revenue_state(
    text: str,
    toks: set[str],
    attempts: _Attempts,
    session_id: str | None = None,
    objective: str | None = None,
    ticker: object = None,
) -> tuple[bool, bool, dict[str, JSONValue] | None]:
    """Require the newest successful listing to identify the issuer's latest 10-Q."""
    from app.research.scheduler import (
        _inline_attempt_payloads,
        _issuer_filing_rows,
        _newest_filing,
        _objective_subject_ticker,
        _persisted_shapes,
    )

    revenue_quarter = bool("revenue" in toks and toks & {"quarter", "10q"})
    if not revenue_quarter:
        return False, False, None
    if any(
        a.get("tool") == "get_sec_document"
        and a.get("error") is not None
        and a.get("session_id", session_id) == session_id
        and _persisted_shapes(a).get("session_id", session_id) == session_id
        for a in attempts
    ):
        return True, True, None
    listings = _inline_attempt_payloads([a for a in attempts if a.get("tool") == "list_sec_filings"], session_id)
    source = objective if objective is not None else text
    cik = re.search(r"\bCIK\s*[:#]?\s*(\d{1,10})\b", source, re.IGNORECASE)
    issuer = cik.group(1) if cik is not None else _objective_subject_ticker(source, ticker)
    rows = _issuer_filing_rows(listings[:1], issuer)
    newest = _newest_filing(rows, frozenset({"10-Q"}))
    return True, False, {"accession_no": newest[0]} if newest is not None else None


def _carry_only(
    names: set[str],
    attempts: _Attempts,
    ctx_evidence: list[JSONValue],
    session_id: str | None = None,
    issuer: object = None,
) -> bool:
    """Open carry tools only when a same-session structured accession exists."""
    from app.research.scheduler import _ACCESSION_CARRY_TOOLS

    return (
        bool(names)
        and names <= _ACCESSION_CARRY_TOOLS
        and _has_accession([*attempts, *ctx_evidence], session_id, issuer)
    )


def _carry_pick(
    toks: set[str],
    names: set[str],
    attempts: _Attempts,
    ctx_evidence: list[JSONValue],
    listed_10q: dict[str, JSONValue] | None,
    session_id: str | None = None,
    issuer: object = None,
    text: str | None = None,
) -> tuple[str, str] | None:
    """Open packet accessions only in a carry-only registry."""
    if not _carry_only(names, attempts, ctx_evidence, session_id, issuer):
        return None
    if "revenue" in toks and toks & {"quarter", "10q"} and listed_10q is None:
        return None
    if "get_sec_document" in names and toks & _READ_TOKS:
        return "get_sec_document", "carry:packet-doc"
    if "get_sec_document" in names and text is not None:
        from app.research.scheduler import _filing_analysis_intent

        if _filing_analysis_intent(text):
            return "get_sec_document", "carry:packet-doc"
    if "get_sec_filing" in names:
        return "get_sec_filing", "carry:packet-filing"
    return None


def _short_pick(norm: str, toks: set[str], names: set[str]) -> tuple[str, str] | None:
    """Explicit daily SHO precedes the general short-family dispatch."""
    if "get_reg_sho_volume" in names and (
        "reg sho" in norm or {"reg", "sho"} <= toks or {"daily", "short", "volume"} <= toks
    ):
        return "get_reg_sho_volume", "sho-explicit"
    if toks & _SHORT_TOKS or "betting against" in norm:
        tool = _short_tool(toks, norm, names)
        return (tool, "short-family") if tool is not None else None
    return None


def _insider_pick(norm: str, toks: set[str], names: set[str]) -> tuple[str, str] | None:
    """Executed insider trades precede planned-sale notices."""
    if not (toks & _INSIDER_TOKS or "form 4" in norm):
        return None
    if "get_insider_activity" in names and toks & _EXECUTED_TOKS:
        return "get_insider_activity", "insider-executed"
    if toks & _PLANNED_TOKS and "get_planned_insider_sales" in names:
        return "get_planned_insider_sales", "insider-planned"
    if "get_insider_activity" in names:
        return "get_insider_activity", "insider"
    return None


def _mention_pick(norm: str, toks: set[str], names: set[str]) -> tuple[str, str] | None:
    """Person mentions select the available exhaustive or bounded filing search."""
    form = bool({"10k", "10q", "8k"} & toks)
    if names & {"search_sec_filings", "search_sec_filings_bounded"} and (
        (("filing" in toks or form) and "mention" in norm) or "who filed" in norm or "who actually filed" in norm
    ):
        return _pick_variant("search_sec_filings", norm, toks, names), "person-mention"
    return None


def _document_read(attempts: _Attempts, session_id: str | None = None) -> bool:
    """Recognize a successful document read from the current session."""
    from app.research.scheduler import _inline_attempt_payloads

    return bool(_inline_attempt_payloads([a for a in attempts if a.get("tool") == "get_sec_document"], session_id))


def _revenue_pick(
    names: set[str],
    attempts: _Attempts,
    revenue_quarter: bool,
    listed_10q: dict[str, JSONValue] | None,
    session_id: str | None = None,
) -> tuple[str, str] | None:
    """List quarterly revenue filings before reading the node's listed document."""
    if not revenue_quarter or _document_read(attempts, session_id):
        return None
    if "get_sec_document" in names and listed_10q is not None:
        return "get_sec_document", "revenue-quarter:doc"
    if "list_sec_filings" in names:
        return "list_sec_filings", "revenue-quarter"
    return None


def _clock_ask(toks: set[str]) -> bool:
    """Recognize pure clock tokens without research signals."""
    return bool(toks and toks <= _TIME_TOKS and not (toks & _RESEARCH_TOKS))


def _sec_list_pick(text: str, toks: set[str], names: set[str]) -> tuple[str, str] | None:
    """Filing list for a resolved ticker or company subject."""
    from app.research.scheduler import _objective_company, _objective_ticker

    if (
        "filing" in toks
        and "list_sec_filings" in names
        and (_objective_ticker(text) is not None or _objective_company(text) is not None)
    ):
        return "list_sec_filings", "sec-list"
    return None


def _clock_pick(toks: set[str], names: set[str]) -> tuple[str, str] | None:
    """Clock question: generated trading words never divert it."""
    if _clock_ask(toks) and "get_current_time" in names:
        return "get_current_time", "time"
    return None


def _margin_pick(ranked: list[tuple[int, str]]) -> tuple[str, str] | None:
    """Clear scorer winner outside any ambiguity group."""
    if len(ranked) < 2 or ranked[0][0] - ranked[1][0] < _MARGIN:
        return None
    top = ranked[0][1]
    if _ambiguous(top, ranked[1][1], [n for _, n in ranked[:_TOP_N]]):
        return None
    return top, "scorer-margin"


def _general_pick(text: str, toks: set[str], names: set[str], ranked: list[tuple[int, str]]) -> tuple[str, str] | None:
    """Resolved filing subjects precede clocks and an unambiguous scorer margin."""
    return _sec_list_pick(text, toks, names) or _clock_pick(toks, names) or _margin_pick(ranked)


def _router_pick(
    text: str,
    norm: str,
    toks: set[str],
    names: set[str],
    ranked: list[tuple[int, str]],
    attempts: _Attempts,
    ctx_evidence: list[JSONValue],
    session_id: str | None = None,
    objective: str | None = None,
    ticker: object = None,
) -> tuple[str, str] | None:
    """(tool, reason) or None when JEV must decide. Membership-gated: never picks outside the live registry."""
    # Rule order: 1 carry-only registry, 2 explicit Reg SHO, 3 short family, 4 insider,
    # 5 person mention, 6 revenue quarter, 7 filing list, 8 clock, 9 scorer margin.
    revenue_quarter, blocked, listed_10q = _revenue_state(text, toks, attempts, session_id, objective, ticker)
    if blocked:
        return None
    from app.research.scheduler import _objective_subject_ticker

    source = objective if objective is not None else text
    cik = re.search(r"\bCIK\s*[:#]?\s*(\d{1,10})\b", source, re.IGNORECASE)
    issuer = cik.group(1) if cik is not None else _objective_subject_ticker(source, ticker)
    return (
        _carry_pick(toks, names, attempts, ctx_evidence, listed_10q, session_id, issuer, text)
        or _short_pick(norm, toks, names)
        or _insider_pick(norm, toks, names)
        or _mention_pick(norm, toks, names)
        or _revenue_pick(names, attempts, revenue_quarter, listed_10q, session_id)
        or _general_pick(text, toks, names, ranked)
    )


_SMALL_TALK_PHRASES = (
    "thanks",
    "thank you",
    "never mind",
    "nevermind",
    "ok",
    "okay",
    "got it",
    "great thanks",
    "hello",
    "hi",
    "hey",
    "hiya",
    "yo",
    "howdy",
    "greetings",
    "hello there",
    "hi there",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "whats up",
    "sup",
    "how are you",
    "thx",
    "ty",
    "cool",
    "bye",
    "goodbye",
)


def _SMALL_TALK(prompt: str) -> bool:
    """Exact greetings and acknowledgements create no session after punctuation and space normalization."""
    text = " ".join("".join(c for c in prompt.lower() if c.isalnum() or c.isspace()).split())
    if text not in _SMALL_TALK_PHRASES:
        return False
    if "$" in prompt:
        return False
    from app.research.scheduler import _objective_ticker

    return _objective_ticker(prompt) is None


def _advice_cue(norm: str, toks: set[str]) -> bool:
    """Advice wording: should plus buy/sell/hold, or the good-investment phrase."""
    return bool("should" in toks and toks & _ADVICE_VERBS) or "good investment" in norm


def _research_cue(norm: str, toks: set[str]) -> bool:
    """Wording that routes to research even when a subject resolves."""
    return bool(toks & _RESEARCH_TOKS) or "reg sho" in norm or "betting against" in norm


def _has_subject(prompt: str, ticker: str | None) -> bool:
    """Subject present: explicit ticker token, for-company phrase, or the caller's ticker."""
    from app.research.scheduler import _objective_company, _objective_ticker

    return _objective_ticker(prompt) is not None or _objective_company(prompt) is not None or ticker is not None


def needs_subject(prompt: str, ticker: str | None, current_ticker: str | None = None) -> bool:
    """Clarify subjectless advice only when neither scope nor displayed stock supplies a subject."""
    _, toks = _signals(prompt)
    return (
        bool(toks & _SUBJECT_ADVICE_TOKS)
        and not toks.difference(_SUBJECT_ADVICE_TOKS, _SUBJECT_FILLER_TOKS)
        and not _has_subject(prompt, ticker or current_ticker)
    )


def _single_shot_tool(prompt: str, norm: str, toks: set[str], subject: str) -> str | None:
    """One ticker, one intent anchor, no other words: the tool answerSingleShot runs. None keeps research."""
    rest = toks - {subject.lower()} - _SINGLE_SHOT_FILLER
    anchors = rest & _SINGLE_SHOT_INTENTS.keys()
    if len(anchors) != 1 or not rest <= _SINGLE_SHOT_INTENTS[next(iter(anchors))]:
        return None
    from app.research.scheduler import build_registry

    names = _registry_names(build_registry())
    # Entry lists these filings. Research nodes retain their existing carry/read gates.
    if anchors & {"revenue", "risk"}:
        return "list_sec_filings" if "list_sec_filings" in names else None
    pick = _router_pick(prompt, norm, toks, names, [], [], [])
    return pick[0] if pick is not None else None


def _subject_route(prompt: str, norm: str, toks: set[str], ticker: str | None) -> str | None:
    """Advice needs a resolved subject and no research token; a single-intent ticker ask runs one tool."""
    from app.research.scheduler import _objective_company, _objective_ticker

    # Advice yields only to research tokens; "reg sho"/"betting against" phrases do not override it.
    if _advice_cue(norm, toks) and _has_subject(prompt, ticker) and not (toks & _RESEARCH_TOKS):
        return "reasoning_required"
    subject = _objective_ticker(prompt) or ticker
    if subject is not None:
        return _single_shot_tool(prompt, norm, toks, subject) or "research_required"
    if _objective_company(prompt) is not None or ticker is not None or _research_cue(norm, toks):
        return "research_required"
    return None


def programmatic_route(prompt: object, ticker: str | None = None) -> str | None:
    """Route exact small talk, clocks, advice, and research signals. JEV decides other prompts."""
    from app.research.scheduler import _ACCESSION_TOKEN_RE

    if not isinstance(prompt, str) or not prompt.strip():
        return "research_required"
    if _SMALL_TALK(prompt):
        return "no_session"
    norm, toks = _signals(prompt)
    # Watchlist mutations need JEV's local UI tool, not the ticker research hint.
    if re.search(r"\b(?:add|remove)\b", prompt, re.IGNORECASE) and re.search(
        r"\bwatch[\s-]?lists?\b", prompt, re.IGNORECASE
    ):
        return None
    # Pure clock question wins on the original prompt even with a selected ticker.
    if _clock_ask(toks):
        return "get_current_time"
    # Thesis mutations use research so the stockbot answer can offer the trio.
    if _THESIS_CREATE_RE.search(norm) or _THESIS_REFINE_RE.search(norm):
        return "research_required"
    if needs_subject(prompt, ticker):
        return "clarify_subject"
    if _ACCESSION_TOKEN_RE.search(prompt) is not None:
        return "research_required"
    return _subject_route(prompt, norm, toks, ticker)


def _objective_text(session: Mapping[str, JSONValue]) -> str:
    """Session objective (else query) text; blank when the value is not a string."""
    raw = session.get("objective") or session.get("query") or ""
    return raw if isinstance(raw, str) else ""


def _clock_round(
    kernel: _Kernel, sid: str, nid: str, question: object, objective: str, names: set[str]
) -> tuple[str, ToolDecision] | None:
    """Answer a clock ask on the original question text before generated words divert it; None otherwise."""
    if "get_current_time" not in names:
        return None
    for text in (question, objective):
        if not isinstance(text, str) or not text.strip():
            continue
        _, toks = _signals(text)
        if _clock_ask(toks):
            return _record_pick(kernel, sid, nid, "get_current_time", "time:original-question")
    return None


def _record_pick(kernel: _Kernel, sid: str, nid: str, tool: str, reason: str) -> tuple[str, ToolDecision]:
    """Validate, persist, and log one deterministic tool selection."""
    from app.research import scheduler as sched

    decision = ToolDecision(
        action="invoke",
        tool_name=tool,
        tool_names=(tool,),
        probabilities={tool: 1.0},
        confidence=None,
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

    objective = _objective_text(session)
    question = sched._f(sched._node_dict(node), "question", default="") or ""
    names = _registry_names(registry)
    clock = _clock_round(kernel, sid, nid, question, objective, names)
    if clock is not None:
        return clock
    text = f"{objective} {question}".strip() if isinstance(question, str) else objective
    norm, toks = _signals(text)
    if _THESIS_CREATE_RE.search(norm) and "thesis_create" in names and sid not in _THESIS_CREATED:
        picked = _record_pick(kernel, sid, nid, "thesis_create", "thesis-create")
        _THESIS_CREATED.add(sid)
        return picked
    ranked = _ranked(norm, toks, names)
    pick = _router_pick(
        text, norm, toks, names, ranked, attempts or [], ctx_evidence or [], sid, objective, session.get("ticker")
    )
    if pick is not None:
        tool, reason = pick
        return _record_pick(kernel, sid, nid, _pick_variant(tool, norm, toks, names), reason)
    action, decision = await sched._select_round(jev, kernel, sid, nid, session, node, registry, ctx_evidence, attempts)
    return action, _variant_decision(decision, norm, toks, names)
