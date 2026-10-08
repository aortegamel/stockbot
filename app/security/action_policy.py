"""Action firewall and egress policy. stdlib only.

Every tool call is gated against the run's original intent (permitted
domains are derived from USER TURNS ONLY — external evidence can never
expand them). Egress to external destinations (Exa) is blocked when the
payload carries credential or portfolio-context patterns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..redact import _ACCOUNT_ID_RE, _BEARER_RE, _JWT_RE, _SK_OR_V1_RE
from .context import RunSecurityContext

# Keep in sync with TOOL_CAPABILITIES (parity test asserts the portfolio set).
TOOL_DOMAINS: dict[str, str] = {
    # SEC filing data.
    "get_fundamentals": "financial_research",
    "list_sec_filings": "financial_research",
    "find_sec_entities": "financial_research",
    "find_sec_entities_bounded": "financial_research",
    "search_sec_filings": "financial_research",
    "search_sec_filings_bounded": "financial_research",
    "search_sec_relationships": "financial_research",
    "get_sec_search_coverage": "financial_research",
    "get_sec_filing": "financial_research",
    "list_sec_documents": "financial_research",
    "get_sec_document": "financial_research",
    "diff_sec_filings": "financial_research",
    "get_material_events": "financial_research",
    "get_beneficial_ownership": "financial_research",
    "get_ownership_changes": "financial_research",
    "get_insider_activity": "financial_research",
    "get_planned_insider_sales": "financial_research",
    "get_offering_history": "financial_research",
    "get_dilution_profile": "financial_research",
    "get_governance_events": "financial_research",
    "get_transaction_status": "financial_research",
    "get_short_pressure_profile": "financial_research",
    "search_tools": "financial_research",
    "list_tool_domains": "financial_research",
    "describe_tool": "financial_research",
    "browse_tools": "financial_research",
    "call_tool": "financial_research",
    "get_recent_ownership_filings": "financial_research",
    "diff_risk_factors": "financial_research",
    "get_financial_statements": "financial_research",
    "get_xbrl_facts": "financial_research",
    "get_obligations": "financial_research",
    "get_valuation_metrics": "financial_research",
    # FINRA public market data.
    "list_finra_datasets": "financial_research",
    "describe_finra_dataset": "financial_research",
    "get_finra_datapoints": "financial_research",
    "query_finra": "financial_research",
    "get_short_interest": "financial_research",
    "get_short_interest_leaderboard": "financial_research",
    "get_reg_sho_volume": "financial_research",
    "get_threshold_securities": "financial_research",
    # Analyst consensus data.
    "get_analyst_estimates": "financial_research",
    "get_sp500_weight": "financial_research",
    # Robinhood market data.
    "get_market_snapshot": "financial_research",
    "get_option_chain": "financial_research",
    "analyze_option_contract": "financial_research",
    "compare_options": "financial_research",
    "get_scanner_filter_specs": "financial_research",
    # Exa web research.
    "search_web": "public_web_research",
    # Google public data (optional, bounded research).
    "find_alternative_signals": "financial_research",
    "get_trend_evidence": "financial_research",
    "investigate_social_arbitrage_candidate": "financial_research",
    "get_macro_context": "financial_research",
    "search_company_patents": "financial_research",
    "get_current_time": "financial_research",
    # Bounded local thesis operations (RESEARCH-scoped, never broker data).
    "thesis_create": "financial_research",
    "thesis_show": "financial_research",
    "thesis_refine": "financial_research",
    "thesis_watch": "financial_research",
    "thesis_journal": "financial_research",
    "thesis_status": "financial_research",
    "research_start": "financial_research",
    "research_resume": "financial_research",
    "research_status": "financial_research",
    "research_cancel": "financial_research",
    "research_read": "financial_research",
    "research_read_search": "financial_research",
    "research_add_evidence": "financial_research",
    "research_submit_source_result": "financial_research",
    "research_add_analysis": "financial_research",
    "research_finalize": "financial_research",
    # Robinhood portfolio data (private).
    "evaluate_mandate": "portfolio_read",
    "get_portfolio_snapshot": "portfolio_read",
    "get_scans": "portfolio_read",
    "run_scan": "portfolio_read",
}

EGRESS_INTENT_REASON = "PRIVATE -> EXTERNAL egress"
EGRESS_AFTER_PRIVATE_REASON = "PRIVATE -> EXTERNAL egress after private context"
CREDENTIAL_REASON = "credential pattern"

_CREDENTIAL_PATTERNS = (
    ("bearer", _BEARER_RE),
    ("sk_or_v1", _SK_OR_V1_RE),
    ("jwt", _JWT_RE),
    ("account_id", _ACCOUNT_ID_RE),
)

_POSSESSIVE_OWN_RE = re.compile(
    r"\b(?:my|i|you|user|we)\s+(?:own\w*|have\w*|hold\w*|held)\b",
    re.IGNORECASE,
)
_PORTFOLIO_NOUN_RE = re.compile(r"\b(?:portfolio|position|holdings|balance|account)\b", re.IGNORECASE)
# Digit-runs (4+) or "$" amounts — the numeric side of portfolio context.
_AMOUNT_TOKEN_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d{4,}\b")

_EGRESS_TOKEN_WINDOW = 12


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str | None


def _source_policy_mode(source_policy: dict[str, object]) -> str | None:
    """Allowlist/all mode; None when the mode is unknown (caller reports it)."""
    raw_mode = source_policy.get("mode", "all")
    mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else "all"
    return mode if mode in ("all", "allowlist") else None


def _denied_set(raw_denied: object) -> set[str]:
    """Denied entries as lowercase strings; non-lists and non-strings ignored."""
    if not isinstance(raw_denied, list):
        return set()
    return {s.strip().lower() for s in raw_denied if isinstance(s, str)}


def _source_denied_hit(name: str, raw_denied: object) -> str | None:
    """Denied-list hit reason; None when no denied entry matches."""
    lowered = name.strip().lower()
    if any(d and d in lowered for d in _denied_set(raw_denied)):
        return f"POLICY_DENIED: tool {name!r} denied by session source_policy"
    return None


def _allowlist_set(raw_allowed: object) -> set[str] | None:
    """Allowlist entries as lowercase strings; None when not a list."""
    if not isinstance(raw_allowed, list):
        return None
    return {s.strip().lower() for s in raw_allowed if isinstance(s, str)}


def _allowlist_match(name: str, allowed: set[str]) -> bool:
    """True when the allowlist covers this tool (per-domain allowlists, never substrings)."""
    # ponytail: exact per-domain membership for the research domains; any other
    # entry keeps the legacy substring behavior (custom allowlists stay working).
    try:
        from app.research.agents.source_agent import FINRA_TOOLS, SEC_TOOLS, WEB_TOOLS
    except ImportError:
        SEC_TOOLS = frozenset()
        FINRA_TOOLS = frozenset()
        WEB_TOOLS = frozenset()
    lowered = {a.strip().lower() for a in allowed if isinstance(a, str)}
    if "sec" in lowered and name in SEC_TOOLS:
        return True
    if "finra" in lowered and name in FINRA_TOOLS:
        return True
    if "web" in lowered and name in WEB_TOOLS:
        return True
    custom = lowered - {"sec", "finra", "web"}
    return any(a and a in name.strip().lower() for a in custom)


def _allowlist_verdict(name: str, raw_allowed: object) -> str | None:
    """Allowlist verdict: None when allowed, else the outside-allowlist reason."""
    allowed = _allowlist_set(raw_allowed)
    if allowed is None:
        return f"POLICY_DENIED: tool {name!r} outside session source allowlist"
    if _allowlist_match(name, allowed):
        return None
    return f"POLICY_DENIED: tool {name!r} outside session source allowlist"


def source_denied_reason(name: str, source_policy: object) -> str | None:
    """Deny reason when a session source_policy excludes this tool; None when allowed."""
    if source_policy is None:
        return None
    if not isinstance(source_policy, dict):
        return "session source_policy must be a mapping"
    mode = _source_policy_mode(source_policy)
    if mode is None:
        return f"unknown source_policy mode {source_policy.get('mode')!r}"
    hit = _source_denied_hit(name, source_policy.get("denied", []))
    if hit is not None:
        return hit
    if mode == "all":
        return None
    return _allowlist_verdict(name, source_policy.get("allowed", []))


def is_sec_tool_name(name: str) -> bool:
    """True for SEC/financial-statement discovery tools; FINRA/Web/Market/Analyst excluded."""
    if not isinstance(name, str) or not name.strip():
        return False
    try:
        from app.research.agents.source_agent import SEC_TOOLS
    except ImportError:
        sec_tools: frozenset[str] = frozenset()
        return TOOL_DOMAINS.get(name, "") != "portfolio_read" and name in sec_tools
    if name in SEC_TOOLS:
        return TOOL_DOMAINS.get(name) != "portfolio_read"
    return False


def _call_tool_inner(arguments: object) -> str | None:
    """Inner tool name for a call_tool wrapper; None when absent/not-a-string."""
    if not isinstance(arguments, dict):
        return None
    inner = arguments.get("name")
    return inner.strip() if isinstance(inner, str) and inner.strip() else None


def authorize_tool_call(name: str, arguments: dict[str, object], run_security: RunSecurityContext) -> tuple[bool, str]:
    """Gate one tool call against intent plus the explicit session grant."""
    denied = source_denied_reason(name, getattr(run_security, "source_policy", None))
    if denied is not None:
        return False, denied
    if name == "call_tool":
        inner = _call_tool_inner(arguments)
        if inner is not None:
            inner_denied = source_denied_reason(inner, getattr(run_security, "source_policy", None))
            if inner_denied is not None:
                return False, inner_denied
    domain = TOOL_DOMAINS.get(name)
    if domain == "portfolio_read":
        if run_security.authorization.portfolio_read:
            return True, ""
        return False, "portfolio access is not authorized for this session"
    if domain is not None and domain in run_security.original_intent.permitted_domains:
        return True, ""
    return False, "tool call exceeds original user intent"


def filter_allowed_tools(names: list[str], source_policy: object) -> list[str]:
    """Discovery-safe allowlist: browse/search results minus source-denied tools (never bypasses)."""
    return [name for name in names if source_denied_reason(name, source_policy) is None]


def _portfolio_context(query: str) -> bool:
    """Portfolio-context patterns: possessive-own phrasing, or a portfolio
    noun within 12 tokens of a digit-run or dollar amount."""
    if _POSSESSIVE_OWN_RE.search(query):
        return True
    tokens = query.split()
    nouns = [i for i, token in enumerate(tokens) if _PORTFOLIO_NOUN_RE.search(token)]
    amounts = [i for i, token in enumerate(tokens) if _AMOUNT_TOKEN_RE.search(token)]
    return any(
        abs(noun_index - amount_index) <= _EGRESS_TOKEN_WINDOW for noun_index in nouns for amount_index in amounts
    )


def private_pattern_hit(text: str) -> str | None:
    """First private-data pattern hit in arbitrary text: a credential match
    wins, then portfolio-context phrasing. None when the text is clean."""
    for _name, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return CREDENTIAL_REASON
    if _portfolio_context(text):
        return EGRESS_INTENT_REASON
    return None


def authorize_egress(destination: str, payload: object, run_security: RunSecurityContext) -> EgressDecision:
    """Block private data from leaving Stockbot to external destinations.

    Ordering invariant: once ANY private tool result has been ALLOWED into
    model context this run, all further egress is blocked regardless of the
    payload. The full serialized payload (query and every filter) is then
    scanned for credential and portfolio-context patterns.
    """
    if destination != "exa":
        return EgressDecision(False, f"unknown egress destination: {destination}")
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
        return EgressDecision(False, "malformed egress payload")
    if "private" in run_security.data_labels:
        return EgressDecision(False, EGRESS_AFTER_PRIVATE_REASON)
    hit = private_pattern_hit(json.dumps(payload, sort_keys=True))
    if hit:
        return EgressDecision(False, hit)
    return EgressDecision(True, None)
