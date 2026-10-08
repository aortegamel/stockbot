"""Context gateway: labels every tool result and scans free-form text
before it may enter model context. stdlib only."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..redact import _BEARER_RE, _JWT_RE, _SK_OR_V1_RE
from . import prompt_injection
from .context import (
    ContextEnvelope,
    InstructionAuthority,
    Integrity,
    SecurityStatus,
    Sensitivity,
    SourceType,
)

SECRET_BLOCK_REASON = "secret data never enters model context"
CREDENTIAL_BLOCK_REASON = "credential pattern"

_SECRET_PATTERNS = (
    ("bearer", _BEARER_RE),
    ("sk_or_v1", _SK_OR_V1_RE),
    ("jwt", _JWT_RE),
)


@dataclass(frozen=True)
class SafeContext:
    envelope: ContextEnvelope
    text: str


@dataclass(frozen=True)
class QuarantinedContext:
    envelope: ContextEnvelope
    verdict: str  # "QUARANTINE" | "BLOCK"
    score: int
    reasons: tuple[str, ...]
    rule_ids: tuple[str, ...] = ()


def _envelope(
    source: str,
    source_type: SourceType,
    sensitivity: Sensitivity,
    integrity: Integrity,
    *,
    external: bool = False,
) -> ContextEnvelope:
    return ContextEnvelope(
        content=None,
        source=source,
        source_type=source_type,
        instruction_authority=InstructionAuthority.NONE,
        sensitivity=sensitivity,
        integrity=integrity,
        external=external,
        retrieved_at=None,
    )


# One base envelope per tool. Tool output never carries instruction
# authority; private tools stay private even when derived (taint rule R4).
TOOL_ENVELOPES: dict[str, ContextEnvelope] = {
    # SEC filing data.
    "get_fundamentals": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "list_sec_filings": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "find_sec_entities": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "find_sec_entities_bounded": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "search_sec_filings": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "search_sec_filings_bounded": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "search_sec_relationships": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_sec_search_coverage": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_sec_filing": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "list_sec_documents": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_sec_document": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "diff_sec_filings": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_material_events": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_beneficial_ownership": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_ownership_changes": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_insider_activity": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_planned_insider_sales": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_offering_history": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_dilution_profile": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_governance_events": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_transaction_status": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_short_pressure_profile": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "search_tools": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "list_tool_domains": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "describe_tool": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "browse_tools": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "call_tool": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_recent_ownership_filings": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "diff_risk_factors": _envelope("sec", SourceType.FILING, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_financial_statements": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_xbrl_facts": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_obligations": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_valuation_metrics": _envelope("sec", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    # FINRA public market data.
    "list_finra_datasets": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "describe_finra_dataset": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_finra_datapoints": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "query_finra": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_short_interest": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_short_interest_leaderboard": _envelope(
        "finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL
    ),
    "get_reg_sho_volume": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    "get_threshold_securities": _envelope("finra", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.CANONICAL),
    # Analyst consensus data.
    "get_analyst_estimates": _envelope("analyst", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL),
    "get_sp500_weight": _envelope("analyst", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL),
    # Exa web evidence.
    "search_web": _envelope("exa", SourceType.WEB, Sensitivity.PUBLIC, Integrity.EXTERNAL, external=True),
    # Google public data (external, bounded research).
    "find_alternative_signals": _envelope(
        "google", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL, external=True
    ),
    "get_trend_evidence": _envelope(
        "google", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL, external=True
    ),
    "investigate_social_arbitrage_candidate": _envelope(
        "google", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL, external=True
    ),
    "get_macro_context": _envelope(
        "google", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL, external=True
    ),
    "search_company_patents": _envelope(
        "google", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL, external=True
    ),
    "get_current_time": _envelope("system", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    # Bounded local thesis operations (locally derived, never canonical market data).
    "thesis_create": _envelope("thesis", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "thesis_show": _envelope("thesis", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "thesis_refine": _envelope("thesis", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "thesis_watch": _envelope("thesis", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "thesis_journal": _envelope("thesis", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "thesis_status": _envelope("thesis", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_start": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_resume": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_status": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_cancel": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_read": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_read_search": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_add_evidence": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_submit_source_result": _envelope(
        "research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED
    ),
    "research_add_analysis": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    "research_finalize": _envelope("research", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.DERIVED),
    # Robinhood market data (account-connected, public observations).
    "get_market_snapshot": _envelope("robinhood", SourceType.MCP, Sensitivity.PUBLIC, Integrity.AUTHENTICATED),
    "get_option_chain": _envelope("robinhood", SourceType.MCP, Sensitivity.PUBLIC, Integrity.AUTHENTICATED),
    "analyze_option_contract": _envelope("robinhood", SourceType.MCP, Sensitivity.PUBLIC, Integrity.AUTHENTICATED),
    "compare_options": _envelope("robinhood", SourceType.MCP, Sensitivity.PUBLIC, Integrity.AUTHENTICATED),
    "get_scanner_filter_specs": _envelope("robinhood", SourceType.MCP, Sensitivity.PUBLIC, Integrity.AUTHENTICATED),
    # Robinhood portfolio data (private).
    "evaluate_mandate": _envelope("mandate", SourceType.TOOL_RESULT, Sensitivity.PRIVATE, Integrity.DERIVED),
    "get_portfolio_snapshot": _envelope("robinhood", SourceType.MCP, Sensitivity.PRIVATE, Integrity.AUTHENTICATED),
    "get_scans": _envelope("robinhood", SourceType.MCP, Sensitivity.PRIVATE, Integrity.AUTHENTICATED),
    "run_scan": _envelope("robinhood", SourceType.MCP, Sensitivity.PRIVATE, Integrity.AUTHENTICATED),
}

_FALLBACK_ENVELOPE = _envelope("unknown", SourceType.TOOL_RESULT, Sensitivity.PUBLIC, Integrity.EXTERNAL)


def envelope_for_tool(name: str, result: dict[str, object]) -> ContextEnvelope:
    """The labeled envelope for a tool result (conservative fallback for
    unregistered tools)."""
    base = TOOL_ENVELOPES.get(name, _FALLBACK_ENVELOPE)
    retrieved_at: str | None = None
    if isinstance(result, dict):
        retrieved_at = next(
            (
                value
                for key in ("retrieved_at", "as_of", "as_of_date")
                for value in (result.get(key),)
                if isinstance(value, str)
            ),
            None,
        )
    return replace(base, content=result, retrieved_at=retrieved_at)


def _secret_pattern_hits(text: str) -> list[str]:
    return [name for name, pattern in _SECRET_PATTERNS if pattern.search(text)]


def prepare_context(envelope: ContextEnvelope, rendered: str) -> SafeContext | QuarantinedContext:
    """Gateway decision for one context item.

    SECRET never enters model context. Every other rendered result — no
    source, type, or sensitivity exemptions — is scanned for credential and
    injection patterns; PRIVATE content is allowed only when benign.
    """
    if envelope.sensitivity == Sensitivity.SECRET:
        return QuarantinedContext(envelope, "BLOCK", 100, (SECRET_BLOCK_REASON,), ("secret_envelope",))
    hits = _secret_pattern_hits(rendered)
    if hits:
        return QuarantinedContext(
            envelope,
            "BLOCK",
            100,
            (f"{CREDENTIAL_BLOCK_REASON}: {', '.join(hits)}",),
            tuple(f"credential_pattern:{name}" for name in hits),
        )
    assessment = prompt_injection.assess(rendered)
    if assessment.verdict != "ALLOW":
        return QuarantinedContext(
            envelope,
            assessment.verdict,
            assessment.score,
            assessment.reasons,
            assessment.matched_rules,
        )
    return SafeContext(
        envelope=replace(envelope, security_status=SecurityStatus.ALLOWED),
        text=rendered,
    )
