"""Context-aware SEC source agent: branch map + query families + dossier assembly.

Workflow it drives: build the material branch map -> search SEC (navigation
only, never evidence) -> open filings -> raw-document findings -> continue
with materially new searches -> structured coverage submit. No maximum number
of searches, filing/document/exhibit reads, or waves, and no count cap on a
material set: every materially distinct planned query is dispatched to its role
(normalized dedupe is the only bound).

Fake-model sketch (no live calls): fake ``dispatch`` + fake ``model`` into
``decompose_question`` / ``assemble_dossier``; assert non-SEC tools are
refused and refs validate against known evidence ids.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from . import GroundedClaim, ModelOutputFailure, conservative_claim_type
from .scout import (
    SOURCE_WORKFLOW,
    ScoutAssignment,
    ScoutResult,
    ScoutRole,
    normalize_query,
)

# SEC-only allowlist: discovery wrappers + SEC/financial-statement tools.
# Domain guard also consults TOOL_DOMAINS (portfolio_read always denied).
SEC_TOOLS: frozenset[str] = frozenset(
    {
        "browse_tools",
        "search_tools",
        "describe_tool",
        "list_tool_domains",
        "call_tool",
        "find_sec_entities",
        "find_sec_entities_bounded",
        "list_sec_filings",
        "get_sec_filing",
        "list_sec_documents",
        "get_sec_document",
        "search_sec_filings",
        "search_sec_filings_bounded",
        "search_sec_relationships",
        "get_sec_search_coverage",
        "diff_sec_filings",
        "get_material_events",
        "get_beneficial_ownership",
        "get_ownership_changes",
        "get_insider_activity",
        "get_planned_insider_sales",
        "get_offering_history",
        "get_dilution_profile",
        "get_governance_events",
        "get_transaction_status",
        "get_financial_statements",
        "get_xbrl_facts",
        "get_obligations",
        "get_valuation_metrics",
        "get_fundamentals",
        "diff_risk_factors",
        "get_recent_ownership_filings",
    }
)


def is_sec_tool(name: str) -> bool:
    """Infra guard: allowlisted name and never a portfolio/private tool."""
    if name not in SEC_TOOLS:
        return False
    try:
        from app.security.action_policy import TOOL_DOMAINS
    except ImportError:
        return True
    return TOOL_DOMAINS.get(name) != "portfolio_read"


# FINRA-desk allowlist (§4 tools only): dataset catalog + structured records.
# get_short_pressure_profile is a FINRA+SEC composite the desk may cite for
# positioning context; canonical shares stay with the SEC desk.
FINRA_TOOLS: frozenset[str] = frozenset(
    {
        "list_finra_datasets",
        "describe_finra_dataset",
        "get_finra_datapoints",
        "query_finra",
        "get_short_interest",
        "get_short_pressure_profile",
        "get_reg_sho_volume",
        "get_threshold_securities",
        "get_short_interest_leaderboard",
    }
)


def is_finra_tool(name: str) -> bool:
    """Infra guard: one of the FINRA-desk tools, never anything else."""
    return name in FINRA_TOOLS


# Web-desk allowlist: Exa search only, never a canonical SEC/FINRA fact source.
WEB_TOOLS: frozenset[str] = frozenset({"search_web"})


def is_web_tool(name: str) -> bool:
    """Infra guard: Exa web search only."""
    return name in WEB_TOOLS


def source_domain_for_tool(name: str) -> str:
    """Job source_domain owning one canonical tool: WEB, FINRA, SEC, else OTHER."""
    if is_web_tool(name):
        return "WEB"
    if is_finra_tool(name):
        return "FINRA"
    if is_sec_tool(name):
        return "SEC"
    return "OTHER"


def _coerce_wave(wave_id: int | str) -> int:
    """Accept int>=1 or numeric str; reject bool/non-numeric/<1."""
    if isinstance(wave_id, bool):
        raise ValueError(f"source dossier: 'wave_id' must be an int >= 1, got {wave_id!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    if isinstance(wave_id, int):
        wave = wave_id
    elif isinstance(wave_id, str):
        text = wave_id.strip()
        if not text.isdigit():
            raise ValueError(f"source dossier: 'wave_id' must be an int >= 1, got {wave_id!r}")
        wave = int(text)
    else:
        raise ValueError(f"source dossier: 'wave_id' must be an int >= 1, got {wave_id!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    if wave < 1:
        raise ValueError(f"source dossier: 'wave_id' must be >= 1, got {wave_id!r}")
    return wave


@dataclass
class SourceDossier:
    """Local dossier shape; mirrors dossiers/sec SECDossier fields (wave_id stored as int)."""

    dossier_id: str
    session_id: str
    wave_id: int
    as_of: str
    findings: list[GroundedClaim] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        self.wave_id = _coerce_wave(self.wave_id)


_CONTEXT_KEYS = (
    "primary_entities",
    "related_entities",
    "industries",
    "products",
    "technologies",
    "relationships",
    "concepts",
    "risks",
    "catalysts",
)

_STOPWORDS = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "why",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "having",
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "and",
        "or",
        "nor",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "any",
        "all",
        "each",
        "its",
        "it",
        "their",
        "his",
        "her",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "about",
        "into",
        "over",
        "under",
        "between",
        "among",
        "through",
        "during",
        "latest",
        "current",
        "recent",
        "new",
        "tell",
        "me",
        "please",
        "show",
        "give",
        "find",
        "list",
        "describe",
        "explain",
        "s",
        "t",
        "d",
        "ll",
        "m",
        "re",
        "ve",
    }
)

_CAP_PHRASE_RE = re.compile(r"[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_RELATION_KEYS = ("subject", "relation", "object")


def _dedupe_keep(items: Sequence[object]) -> list[str]:
    """Strip + normalized-key dedupe (lower/whitespace), first form wins."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        key = normalize_query(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _strip_possessive(text: str) -> str:
    """Drop a trailing 's so entity mentions match ticker scope."""
    low = text.lower()
    if low.endswith(("'s", "’s")):
        return text[:-2].strip()
    return text


def _keep_phrase(cleaned: str) -> bool:
    """Entity-mention rule: multi-word phrase or 2+ letter name, never all stopwords."""
    words = [w.strip(".,") for w in cleaned.split()]
    return (
        len(cleaned) >= 2
        and any(ch.isalpha() for ch in cleaned)
        and not (words and all(w.lower() in _STOPWORDS for w in words))
        and not (len(words) == 1 and len(words[0]) < 2)
    )


def _capitalized_phrases(question: str) -> list[str]:
    """Generic entity mentions: multi-word phrases or 2+ letter names."""
    return _dedupe_keep(
        [
            cleaned
            for match in _CAP_PHRASE_RE.findall(question or "")
            if _keep_phrase(cleaned := " ".join(_strip_possessive(match).split()))
        ]
    )


def _keyword_kept(word: str, excluded: set[str]) -> bool:
    """Distinctive term: 3+ chars, not stopword/digit, outside ticker scope."""
    return len(word) >= 3 and word not in _STOPWORDS and not word.isdigit() and normalize_query(word) not in excluded


def _keywords(question: str, exclude: Sequence[str]) -> list[str]:
    """Distinctive lowercase terms outside ticker scope (concepts/catalysts)."""
    excluded = {normalize_query(e) for e in exclude if isinstance(e, str)}
    return _dedupe_keep(
        [word for match in _WORD_RE.findall((question or "").lower()) if _keyword_kept(word := match.strip(), excluded)]
    )


def _deterministic_context(question: str, tickers: Sequence[str]) -> dict[str, object]:
    """Base context without a model: tickers + phrases + keyword concepts."""
    scoped = _dedupe_keep([t.upper() for t in (tickers or []) if isinstance(t, str) and t.strip()])
    scoped_set = {s.upper() for s in scoped}
    return {
        "primary_entities": list(scoped),
        "related_entities": [p for p in _capitalized_phrases(question or "") if p.upper() not in scoped_set],
        "industries": [],
        "products": [],
        "technologies": [],
        "relationships": [],
        "concepts": _keywords(question, scoped),
        "risks": [],
        "catalysts": [],
    }


def _context_prompt(question: str, tickers: Sequence[str]) -> str:
    """Provider-agnostic JSON prompt for the generic research context (the branch map)."""
    scoped = ", ".join(t for t in tickers if isinstance(t, str) and t.strip()) or "none provided"
    return (
        "Build the material branch map for this SEC research question as JSON only with keys "
        "primary_entities, related_entities, industries, products, technologies, "
        "relationships [{subject, relation, object}], concepts, risks, catalysts. "
        f"Question: {question}\nScope tickers: {scoped}\n"
        "List scope tickers as primary entities; other named issuers (customers, "
        "competitors, suppliers, peers, funds) as related entities; "
        "subject matter as concepts/risks/catalysts. "
        "The map drives later searches, not the answer: search results are navigation artifacts and "
        "only raw SEC documents become evidence. "
        "Expand dynamically to the causal channels the question implies "
        "(counterparty, credit, concentration, lending, commitments, derivatives, "
        "off-balance-sheet, supply-chain, funding/liquidity) — never a fixed issuer list. "
        f"{SOURCE_WORKFLOW} JSON only."
    )


def _coerce_str_list(value: object) -> list[str]:
    """String list from model output, normalized and deduped."""
    return _dedupe_keep(value) if isinstance(value, list) else []


def _rel_key(rel: Mapping[str, object]) -> tuple[str, str, str]:
    """Normalized relationship triple key for dedupe."""
    return (
        normalize_query(str(rel.get("subject", ""))),
        normalize_query(str(rel.get("relation", ""))),
        normalize_query(str(rel.get("object", ""))),
    )


def _rel_triple(item: object) -> dict[str, str] | None:
    """One stripped {subject, relation, object} triple, or None when malformed."""
    if not isinstance(item, dict):
        return None
    if not all(isinstance(item.get(k), str) and item[k].strip() for k in _RELATION_KEYS):
        return None
    return {k: item[k].strip() for k in _RELATION_KEYS}


def _coerce_relationships(value: object) -> list[dict[str, str]]:
    """{subject, relation, object} triples from model output, deduped."""
    if not isinstance(value, list):
        return []
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, str]] = []
    for item in value:
        triple = _rel_triple(item)
        if triple is None:
            continue
        key = _rel_key(triple)
        if key in seen:
            continue
        seen.add(key)
        out.append(triple)
    return out


def _merge_relationships(current_raw: object, decoded_val: object) -> list[dict[str, str]]:
    """Union decoded relationship triples into the base list, normalized-key deduped."""
    extra = _coerce_relationships(decoded_val)
    existing: list[dict[str, str]] = (
        [r for r in current_raw if isinstance(r, dict)] if isinstance(current_raw, list) else []
    )
    seen = {_rel_key(r) for r in existing}
    for rel in extra:
        key = _rel_key(rel)
        if key not in seen:
            seen.add(key)
            existing.append(rel)
    return existing


def _merge_str_key(current_raw: object, decoded_val: object) -> list[str]:
    """Union one decoded string-list key into the base list, deduped."""
    current = current_raw if isinstance(current_raw, list) else []
    return _dedupe_keep([*current, *_coerce_str_list(decoded_val)])


def _merge_context(base: dict[str, object], decoded: object) -> dict[str, object]:
    """Union model output into the deterministic base, normalized."""
    if not isinstance(decoded, dict):
        return base
    merged: dict[str, object] = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for key in _CONTEXT_KEYS:
        if key == "relationships":
            merged[key] = _merge_relationships(merged.get(key, []), decoded.get(key))
        else:
            merged[key] = _merge_str_key(merged.get(key, []), decoded.get(key))
    return merged


def build_research_context(
    question: str,
    tickers: Sequence[str] = (),
    model: Callable[[str], str] | None = None,
) -> dict[str, object]:
    """Generic research context: model-proposed, deterministically normalized.

    The deterministic base (tickers + capitalized phrases + keyword concepts)
    is always present; model output unions in. Never raises on model output --
    unparseable replies fall back to the base.
    """
    base = _deterministic_context(question, tickers)
    if model is None:
        return base
    try:
        decoded: object = json.loads(model(_context_prompt(question, tickers)))
    except Exception:  # noqa: BLE001 - model output degrades to the deterministic context, never raises
        return base
    try:
        return _merge_context(base, decoded)
    except Exception:  # noqa: BLE001 - model output degrades to the deterministic context, never raises
        return base


_RELATED_ROLES = ("customer", "supplier", "competitor", "peer", "fund")
_SUPP_FILING_SUFFIXES = ("8-K", "proxy", "N-PX", "agreement")
_COUNTERPARTY_SUFFIXES = ("exposure", "agreement", "8-K")


def _entity_pairs(primaries: list[str], related: list[str]) -> list[tuple[str, str]]:
    """Distinct (primary, related) pairs by normalized key."""
    return [(a, b) for a in primaries for b in related if normalize_query(a) != normalize_query(b)]


def _add_entity_groups(
    groups: dict[str, list[str]], primaries: list[str], related: list[str], relationships: list[dict[str, str]]
) -> None:
    """Families A/B/E: named entities, pairs, and triple exposures."""
    groups["a"].extend([*primaries, *related])
    pairs = _entity_pairs(primaries, related)
    groups["b"].extend(f"{a} {b}" for a, b in pairs)
    groups["e"].extend(f"{r['subject']} {r['relation']} {r['object']}" for r in relationships)
    groups["e"].extend(f"{a} exposure {b}" for a, b in pairs)


def _add_topic_groups(
    groups: dict[str, list[str]],
    industries: list[str],
    products: list[str],
    technologies: list[str],
    risks: list[str],
    primaries: list[str],
) -> None:
    """Families C/D/F: industry, demand, and risk queries."""
    for industry in industries:
        groups["c"].extend([industry, f"{industry} risk factors"])
    for term in [*products, *technologies]:
        groups["d"].append(term if "demand" in term.lower() else f"{term} demand")
    groups["f"].extend(risks)
    for a in primaries:
        groups["f"].append(f"{a} risk factors")
        groups["f"].extend(f"{a} {risk}" for risk in risks)


def _add_related_groups(groups: dict[str, list[str]], related: list[str]) -> None:
    """Related-issuer role anchors for non-scope candidates."""
    for entity in related:
        groups["related"].extend(f"{entity} {role}" for role in _RELATED_ROLES)


def _add_counterparty_groups(groups: dict[str, list[str]], related: list[str], primaries: list[str]) -> None:
    """Non-issuer counterparty queries: bare names first, then disclosure variants.

    These are the queries a scout issues WITHOUT a ticker filter, so EDGAR
    full-text search is global and its hits name the filers whose filings then
    get opened. Built first in the global dedupe so the bare names survive it.
    """
    scoped = {normalize_query(p) for p in primaries}
    entities = [e for e in related if normalize_query(e) not in scoped]
    groups["cp"].extend(entities)
    for suffix in _COUNTERPARTY_SUFFIXES:
        groups["cp"].extend(f"{entity} {suffix}" for entity in entities)


def _add_supplement_groups(
    groups: dict[str, list[str]],
    concepts: list[str],
    products: list[str],
    technologies: list[str],
    catalysts: list[str],
) -> None:
    """Conceptual supplements: bare terms plus filing/risk-factor mentions."""
    for term in _dedupe_keep([*concepts, *products, *technologies, *catalysts]):
        groups["supp_bare"].append(term)
        groups["supp_filing"].extend(f"{term} {suffix}" for suffix in _SUPP_FILING_SUFFIXES)
        groups["supp_risk"].append(f"{term} risk factor")


def _dedupe_groups(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Global normalized dedupe across families, first family wins."""
    seen: set[str] = set()
    deduped: dict[str, list[str]] = {}
    for key, queries in groups.items():
        unique: list[str] = []
        for query in queries:
            norm = normalize_query(query)
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(query.strip())
        deduped[key] = unique
    return deduped


def _grouped_queries(context: Mapping[str, object]) -> dict[str, list[str]]:
    """Family-grouped SEC queries with global normalized dedup.

    ``cp`` (non-issuer counterparty names) is built first so its bare names win
    the global dedupe against the scoped anchor families.
    """
    source: Mapping[str, object] = context if isinstance(context, Mapping) else {}
    lists = {
        k: _coerce_str_list(source.get(k))
        for k in (
            "primary_entities",
            "related_entities",
            "industries",
            "products",
            "technologies",
            "concepts",
            "risks",
            "catalysts",
        )
    }
    groups: dict[str, list[str]] = {
        "cp": [],
        "a": [],
        "b": [],
        "c": [],
        "d": [],
        "e": [],
        "f": [],
        "related": [],
        "supp_bare": [],
        "supp_filing": [],
        "supp_risk": [],
    }
    _add_counterparty_groups(groups, lists["related_entities"], lists["primary_entities"])
    _add_entity_groups(
        groups, lists["primary_entities"], lists["related_entities"], _coerce_relationships(source.get("relationships"))
    )
    _add_topic_groups(
        groups, lists["industries"], lists["products"], lists["technologies"], lists["risks"], lists["primary_entities"]
    )
    _add_related_groups(groups, lists["related_entities"])
    _add_supplement_groups(groups, lists["concepts"], lists["products"], lists["technologies"], lists["catalysts"])
    return _dedupe_groups(groups)


_ROLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "filings": ("a", "b", "related", "supp_filing", "cp"),
    "financials": ("c", "d", "supp_bare"),
    "risk": ("e", "f", "supp_risk"),
}


def build_query_families(context: Mapping[str, object], as_of: str = "") -> list[str]:
    """Families A-F + related-issuer + conceptual supplements, deduped.

    A named entity, B entity-pair, C industry, D demand, E exposure, F risk,
    then related-issuer role anchors (customer/supplier/competitor/peer/fund)
    and conceptual supplements (risk-factor/8-K/proxy/N-PX/agreement mentions),
    then non-issuer counterparty names + disclosure variants (``cp``, searched
    unscoped so EDGAR full-text search is global). Non-issuer terms search as
    concepts, never dead-end. ``as_of`` is accepted for call symmetry;
    point-in-time filtering applies at execution.
    """
    _ = as_of
    groups = _grouped_queries(context)
    out: list[str] = []
    for key in ("a", "b", "c", "d", "e", "f", "related", "supp_bare", "supp_filing", "supp_risk", "cp"):
        out.extend(groups.get(key, []))
    return out


def _baseline_queries(tickers: Sequence[str], context: Mapping[str, object]) -> list[str]:
    """Latest-filing baseline seeds: terminology only, no extra dispatch by default."""
    return []


def _cap_candidates(text: str, seen: set[str]) -> list[str]:
    """Capitalized phrases from one finding not already asked."""
    out: list[str] = []
    for match in _CAP_PHRASE_RE.findall(text):
        cleaned = " ".join(match.split())
        if len(cleaned) >= 3 and normalize_query(cleaned) not in seen:
            out.append(cleaned)
    return out


def _word_candidates(text: str, seen: set[str]) -> list[str]:
    """Distinctive lowercase words from one finding not already asked."""
    return [
        w
        for w in _WORD_RE.findall(text.lower())
        if len(w) > 5 and w.isalpha() and w not in _STOPWORDS and normalize_query(w) not in seen
    ]


def _finding_candidates(finding_texts: Sequence[str] | None, seen: set[str]) -> list[str]:
    """Phrase + word candidates from findings, in passage order."""
    candidates: list[str] = []
    for text in finding_texts or []:
        if not isinstance(text, str):
            continue
        candidates.extend(_cap_candidates(text, seen))
        candidates.extend(_word_candidates(text, seen))
    return candidates


def _context_candidates(context: Mapping[str, object] | None, seen: set[str]) -> list[str]:
    """Context terms (related/concepts/products/technologies) not already asked."""
    if not isinstance(context, Mapping):
        return []
    out: list[str] = []
    for key in ("related_entities", "concepts", "products", "technologies"):
        out.extend(v for v in _coerce_str_list(context.get(key)) if normalize_query(v) not in seen)
    return out


def _take_new(candidates: Sequence[str], seen: set[str]) -> list[str]:
    """Unseen normalized candidates, stripped, in order (dedupe is the only bound)."""
    out: list[str] = []
    for candidate in candidates:
        key = normalize_query(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate.strip())
    return out


def _rerank_queries(new_queries: list[str], reference: str | None) -> list[str]:
    """Embedding order for a bounded candidate set; deterministic order otherwise."""
    if not isinstance(reference, str) or not reference.strip() or len(new_queries) < 2:
        return new_queries
    try:
        from app.needle_client import similarity_order
    except ImportError:
        return new_queries
    try:
        order = similarity_order(new_queries, reference)
    except Exception:  # noqa: BLE001 - embedding is a tiebreak; failure keeps deterministic order
        return new_queries
    if order is None:
        return new_queries
    return [new_queries[i] for i in order]


def expand_queries(
    queries: Sequence[str],
    finding_texts: Sequence[str],
    context: Mapping[str, object] | None = None,
    *,
    reference: str | None = None,
) -> list[str]:
    """New material queries from filing findings, deduped against prior queries.

    Deterministic order by default; a reference question embedding-reranks the
    bounded candidate set, never adding or dropping queries.
    """
    seen = {normalize_query(q) for q in (queries or []) if isinstance(q, str)}
    seen.discard("")
    candidates = [*_finding_candidates(finding_texts, seen), *_context_candidates(context, seen)]
    return _rerank_queries(_take_new(candidates, seen), reference)


def expansion_stop(
    *,
    sec_answerable_remaining: bool = True,
    new_queries: bool = True,
    repeats_yield_nothing: bool = False,
    baseline_reviewed: bool = True,
) -> tuple[bool, str]:
    """Info-based stop: (should_stop, reason); never count-based."""
    if not sec_answerable_remaining:
        return True, "all SEC-answerable resolved; remainder non-SEC-answerable"
    if repeats_yield_nothing and not new_queries:
        return True, "repeats yield nothing new"
    if not new_queries:
        return (
            True,
            "no new material queries; baseline reviewed" if baseline_reviewed else "remainder non-SEC-answerable",
        )
    return False, "continue"


def _sec_match(name: object, discovered: list[str]) -> str | None:
    """SEC tool name from a catalog match, or None when ineligible or repeated."""
    return name if isinstance(name, str) and is_sec_tool(name) and name not in discovered else None


def _discover_sec_tools(dispatch: Callable[[str, dict[str, object]], dict[str, object]]) -> list[str]:
    """Catalog discovery hint: SEC tools matching the two seed queries."""
    discovered: list[str] = []
    for query in ("SEC filings material events", "XBRL financial statements trend"):
        matches = dispatch("search_tools", {"query": query}).get("matches")
        for m in matches if isinstance(matches, list) else []:
            if isinstance(m, dict) and (name := _sec_match(m.get("name"), discovered)) is not None:
                discovered.append(name)
    return discovered


def _role_assignments(
    question: str,
    session_id: str,
    as_of: str,
    tickers: Sequence[str],
    context: dict[str, object],
    groups: dict[str, list[str]],
    baseline: list[str],
) -> list[ScoutAssignment]:
    """One assignment per role with every planned query of its families (dedupe only)."""
    scoped = [t for t in (tickers or []) if isinstance(t, str)]
    roles: tuple[ScoutRole, ScoutRole, ScoutRole] = ("filings", "financials", "risk")
    assignments: list[ScoutAssignment] = []
    for role in roles:
        queries = [q for family in _ROLE_FAMILIES[role] for q in groups.get(family, [])]
        context_copy = {k: (list(v) if isinstance(v, list) else v) for k, v in context.items()}
        unscoped: list[str] = list(groups.get("cp", [])) if role == "filings" else []
        assignments.append(
            ScoutAssignment(
                assignment_id=f"scout-{role}",
                session_id=session_id,
                as_of=as_of,
                role=role,
                question=question,
                tickers=list(scoped),
                context=context_copy,
                queries=list(queries),
                unscoped_queries=unscoped,
                baseline=list(baseline),
            )
        )
    return assignments


def decompose_question(
    question: str,
    *,
    session_id: str,
    as_of: str,
    tickers: Sequence[str],
    dispatch: Callable[[str, dict[str, object]], dict[str, object]],
    model: Callable[[str], str] | None = None,
) -> list[ScoutAssignment]:
    """Catalog discovery + generic context into three role assignments with queries."""
    _ = _discover_sec_tools(dispatch)  # hint only; assignments below carry context queries.
    context = build_research_context(question, tickers, model)
    return _role_assignments(
        question, session_id, as_of, tickers, context, _grouped_queries(context), _baseline_queries(tickers, context)
    )


def _check_claim_refs(
    claim: GroundedClaim, known_set: set[str], session_id: str, journal: Callable[[str, dict[str, object]], None] | None
) -> None:
    """Fail-closed ref check: non-empty text, cited, freeze-contained ids."""
    if not claim.text.strip():
        raise ModelOutputFailure("each claim needs non-empty text")
    if not claim.evidence_ids:
        raise ModelOutputFailure(f"uncited finding: {claim.text[:120]!r}")
    for eid in claim.evidence_ids:
        if eid not in known_set:
            if journal is not None:
                journal("evidence.rejected", {"session_id": session_id, "evidence_id": eid})
            raise ModelOutputFailure(f"unknown evidence id {eid!r}")


def _merge_claim(findings: list[GroundedClaim], claim: GroundedClaim) -> None:
    """Append new claim text or merge duplicates: union ids, least assertive type."""
    prior = next((c for c in findings if c.text == claim.text), None)
    if prior is None:
        findings.append(
            GroundedClaim(
                text=claim.text, claim_type=claim.claim_type, evidence_ids=list(dict.fromkeys(claim.evidence_ids))
            )
        )
    else:
        merged_ids = list(dict.fromkeys([*prior.evidence_ids, *claim.evidence_ids]))
        merged_type = conservative_claim_type([prior.claim_type, claim.claim_type])
        findings[findings.index(prior)] = GroundedClaim(
            text=prior.text, claim_type=merged_type, evidence_ids=merged_ids
        )


def assemble_dossier(
    *,
    dossier_id: str,
    session_id: str,
    wave_id: int | str,
    as_of: str,
    results: Sequence[ScoutResult],
    known_evidence_ids: Sequence[str],
    journal: Callable[[str, dict[str, object]], None] | None = None,
) -> SourceDossier:
    """Merge scout outputs into one validated dossier (wave_id stored as int)."""
    wave = _coerce_wave(wave_id)
    known_set = {e for e in known_evidence_ids if isinstance(e, str) and e}
    findings: list[GroundedClaim] = []
    unknowns: list[str] = []
    limitations: list[str] = []
    coverage_notes: list[str] = []
    for result in results:
        coverage_notes.append(result.coverage)
        unknowns.extend(result.unknowns)
        limitations.extend(result.limitations)
        for claim in list(result.findings or []):
            _check_claim_refs(claim, known_set, session_id, journal)
            _merge_claim(findings, claim)
    return SourceDossier(
        dossier_id=dossier_id,
        session_id=session_id,
        wave_id=wave,
        as_of=as_of,
        findings=findings,
        unknowns=list(dict.fromkeys(unknowns)),
        limitations=list(dict.fromkeys(limitations)),
        coverage_notes=coverage_notes,
    )


__all__ = [
    "SEC_TOOLS",
    "SourceDossier",
    "assemble_dossier",
    "build_query_families",
    "build_research_context",
    "decompose_question",
    "expand_queries",
    "expansion_stop",
    "is_sec_tool",
    "normalize_query",
]
