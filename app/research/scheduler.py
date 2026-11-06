"""Shared research registry, subject discovery, and chat argument grounding.

Native research executes tools through tool_runner. This module retains only
helpers used by session bootstrap and the chat arguments operation.
"""

import concurrent.futures as _futures
import logging
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from app.research.models import JSONValue, validate_ticker

if TYPE_CHECKING:
    from app.tools import ToolDiscovery


_TOOL_POOL = _futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")

logger = logging.getLogger(__name__)

_JEV_REGISTRY_EXCLUDED = frozenset({"call_tool", "browse_tools", "search_tools", "list_tool_domains", "describe_tool"})

_HANDLE_PARAMS = frozenset(
    {
        "accession_no",
        "record_id",
        "source_handle",
        "source_handle_id",
        "tool_result_id",
        "result_id",
        "document_name",
        "search_id",
        "dossier_id",
        "freeze_id",
        "session_id",
        "job_id",
        "evidence_id",
    }
)

_PIT_BLIND_TOOLS = frozenset(
    {
        "thesis_create",
        "thesis_show",
        "thesis_refine",
        "thesis_watch",
        "thesis_journal",
        "thesis_status",
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
        "get_current_time",
    }
)


def _scalar_or_scalar_list(value: object) -> bool:
    """Scalar or scalar-list allowlist for Needle grounded execution values."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(
            item is None or isinstance(item, (str, bool, int)) or (isinstance(item, float) and math.isfinite(item))
            for item in value
        )
    )


def _pit_as_of_seed(tool_name: str, schema: Mapping[str, object], as_of: object) -> str | None:
    """The tool's own `as_of` for a request cutoff: declared parameter on a PIT-scoped tool only."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict) or "as_of" not in properties or tool_name in _PIT_BLIND_TOOLS:
        return None
    return _as_day(as_of)


def _grounded_execution_values(
    tool_name: str,
    schema: Mapping[str, object],
    objective: object,
    supplied_ticker: object = None,
    *,
    as_of: object = None,
    subject: str | None = None,
) -> dict[str, JSONValue]:
    """Explicitly grounded execution values keyed by selected schema properties.

    ``as_of`` is the request cutoff: it seeds a PIT-scoped tool's own ``as_of``,
    so a historical request scopes the read instead of leaving it unbounded.
    ``subject`` is the node's own subject (see _fallback_sec_args).
    """
    properties = schema.get("properties") if isinstance(schema, dict) else None
    names = set(properties) if isinstance(properties, dict) else set()
    seeded = (
        _fallback_thesis_args(tool_name, objective)
        or _fallback_sec_args(tool_name, objective, supplied_ticker, subject=subject)
        or {}
    )
    cutoff = _pit_as_of_seed(tool_name, schema, as_of)
    if cutoff is not None:
        seeded["as_of"] = cutoff
    return {
        str(key): list(value) if isinstance(value, tuple) else value
        for key, value in seeded.items()
        if key in names and _scalar_or_scalar_list(value)
    }


def _needle_scope(scope: object) -> dict[str, JSONValue]:
    """Temporal scope for Needle: typed mode/as_of/start/end scalars only, never raw request text."""
    if not isinstance(scope, dict):
        return {}
    narrowed: dict[str, JSONValue] = {}
    for key in ("mode", "as_of", "start", "end"):
        day = _as_day(scope.get(key)) if key != "mode" else None
        if key == "mode":
            mode = scope.get(key)
            if isinstance(mode, str) and mode.strip():
                narrowed[key] = mode.strip()
        elif day is not None:
            narrowed[key] = day
    return narrowed


def _manifest_params(
    fn: Mapping[str, object],
) -> tuple[dict[str, JSONValue], list[str]]:
    """(parameters dict, required list) from one canonical function schema."""
    raw = fn.get("parameters")
    params: dict[str, JSONValue] = dict(raw) if isinstance(raw, dict) else {}
    raw_required = params.get("required")
    required = [k for k in raw_required if isinstance(k, str)] if isinstance(raw_required, list) else []
    return params, required


def _manifest_prerequisites(required: list[str]) -> str:
    """Handle-ish required params (upstream outputs) as a compact prereq string."""
    needs = [k for k in required if k in _HANDLE_PARAMS]
    return f"needs {', '.join(needs)}" if needs else ""


_REGISTRY_CACHE: list[dict[str, JSONValue]] | None = None


def _manifest_discovery(meta: ToolDiscovery | None) -> dict[str, str]:
    """Render the discovery fields without changing the manifest field order."""
    if meta is None:
        return dict.fromkeys(
            ("purpose", "outputKind", "evidence", "intent", "useWhen", "avoidWhen", "conflicts", "nextTools"), ""
        )
    return {
        "purpose": meta.summary,
        "outputKind": meta.output_kind,
        "evidence": meta.output_kind,
        "intent": meta.intent,
        "useWhen": "; ".join(meta.choose_when),
        "avoidWhen": "; ".join(meta.reject_when),
        "conflicts": ", ".join(meta.conflicts_with),
        "nextTools": ", ".join(meta.related_tools),
    }


def build_registry() -> list[dict[str, JSONValue]]:
    """Canonical research manifests for the worker guard and chat argument schemas.

    Discovery metadata includes prerequisites and point-in-time support.
    Full descriptions and parameter schemas support the chat arguments operation.
    """
    global _REGISTRY_CACHE
    # ponytail: TOOLS + discovery metadata are import-time constants; rebuild once per process.
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    from app.policy import Capability
    from app.security.action_policy import TOOL_DOMAINS
    from app.tools import tools_for_capabilities

    manifests: list[dict[str, JSONValue]] = []

    for tool in tools_for_capabilities(frozenset({Capability.RESEARCH})):
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in _JEV_REGISTRY_EXCLUDED:
            continue
        manifests.append(_registry_entry(name, fn, TOOL_DOMAINS.get(name, "unknown")))
    logger.debug(
        "toolflow registry_detail sid=- nid=- tools=%s",
        ",".join(str(m.get("name", "?")) for m in manifests),
    )
    _REGISTRY_CACHE = manifests
    return manifests


def _registry_entry(name: str, fn: Mapping[str, object], domain: str) -> dict[str, JSONValue]:
    """One compact manifest. ``parameters`` stays for Needle; discovery fields stay optional."""
    from app.tools import TOOL_DISCOVERY_REGISTRY

    params, required = _manifest_params(fn)
    discovery = _manifest_discovery(TOOL_DISCOVERY_REGISTRY.get(name))
    description = fn.get("description")
    return {
        "name": name,
        "domain": domain,
        "description": description if isinstance(description, str) else "",
        "purpose": discovery["purpose"],
        "keyInputs": f"req({', '.join(required)})" if required else "req()",
        "outputKind": discovery["outputKind"],
        "evidence": discovery["evidence"],
        "prerequisites": _manifest_prerequisites(required),
        # ponytail: default true; only confirmed-blind session-local tools opt out.
        "pitSupport": "PIT-blind: current state only" if name in _PIT_BLIND_TOOLS else "PIT-scoped",
        "intent": discovery["intent"],
        "useWhen": discovery["useWhen"],
        "avoidWhen": discovery["avoidWhen"],
        "conflicts": discovery["conflicts"],
        "nextTools": discovery["nextTools"],
        "parameters": params,
    }


def _schema_for(tool_name: str, registry: Sequence[Mapping[str, JSONValue]]) -> dict[str, JSONValue]:
    for entry in registry:
        if entry.get("name") == tool_name:
            schema = entry.get("parameters")
            return dict(schema) if isinstance(schema, dict) else {}
    return {}


_ACCESSION_TOKEN_RE = re.compile(r"\b(\d{10}-\d{2}-\d{6})\b")

_TEN_Q_RE = re.compile(r"\b10-?\s?q\b")

_REVENUE_FORMS = ("10-Q", "10-K")
"""Latest-revenue listing: the latest reported quarter or year, never adjacent current reports."""

_DATE_CONSTRAINT_RE = re.compile(
    r"\d|\b(?:as\s+of|before|after|since|until|between|today|now|yesterday|tomorrow|"
    r"days?|weeks?|months?|years?|quarters?|spring|summer|autumn|fall|winter|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
)
"""Any date constraint in request text, including bare digits and weekday-relative wording."""


def _latest_revenue_objective(objective: object) -> bool:
    """Undated latest-revenue ask: revenue plus latest/most-recent wording, never an explicit dated window."""
    if not isinstance(objective, str):
        return False
    return "revenue" in objective.lower() and re.search(r"\b(?:latest|most recent)\b", objective.lower()) is not None


def _revenue_quarter_objective(objective: object) -> bool:
    """Quarterly revenue ask: revenue plus quarter or 10-Q wording (one owner for every revenue guard)."""
    if not isinstance(objective, str):
        return False
    low = objective.lower()
    return "revenue" in low and ("quarter" in low or _TEN_Q_RE.search(low) is not None)


def _revenue_read_objective(objective: object) -> bool:
    """Revenue section read: quarterly wording or an undated latest-revenue ask."""
    return _revenue_quarter_objective(objective) or _latest_revenue_objective(objective)


_MATERIAL_FORM_RE = re.compile(r"\b(10-?\s?q|10-?\s?k|8-?\s?k)\b")

_MATERIAL_FORM_NAMES = {"10q": "10-Q", "10k": "10-K", "8k": "8-K"}
"""10-Q/10-K/8-K forms named in text, canonicalized for listing arguments."""


def _material_forms(text: object) -> frozenset[str]:
    """Canonical 10-Q/10-K/8-K forms that the text names."""
    tokens = _MATERIAL_FORM_RE.findall(str(text or "").lower())
    return frozenset(_MATERIAL_FORM_NAMES[re.sub(r"[-\s]", "", token)] for token in tokens)


_SEC_IDENTIFIER_TOOLS = frozenset({"list_sec_filings", "search_sec_filings", "search_sec_filings_bounded"})

_SHO_FALLBACK_TOOLS = frozenset({"get_reg_sho_volume"})

_TICKER_FALLBACK_TOOLS = frozenset(
    {
        "get_short_interest",
        "get_insider_activity",
        "get_planned_insider_sales",
        "get_fundamentals",
        "get_xbrl_facts",
        "get_valuation_metrics",
        "get_analyst_estimates",
        "get_material_events",
        "diff_risk_factors",
    }
)

_COMPANY_TOKEN_STOP = frozenset(
    {
        "list",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "did",
        "does",
        "do",
        "has",
        "have",
        "had",
        "are",
        "is",
        "was",
        "were",
        "for",
        "from",
        "with",
        "without",
        "and",
        "the",
        "give",
        "me",
        "this",
        "that",
        "these",
        "those",
        "last",
        "latest",
        "recent",
        "most",
        "current",
        "daily",
        "quarter",
        "filing",
        "filings",
        "form",
        "dates",
        "filer",
        "metadata",
        "document",
        "text",
        "week",
        "executed",
        "planned",
        "sale",
        "sales",
        "notices",
        "trades",
        "shares",
        "insider",
        "insiders",
        "executive",
        "team",
        "company",
        "stock",
        "short",
        "interest",
        "volume",
        "reported",
        "eps",
        "only",
        "actually",
        "want",
        "file",
        "growth",
    }
)

_COMPANY_SUFFIX = frozenset(
    {
        "inc",
        "corp",
        "ltd",
        "co",
        "plc",
        "llc",
        "inc.",
        "corp.",
        "ltd.",
        "co.",
        "plc.",
        "llc.",
    }
)

_OBJECTIVE_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")

_OBJECTIVE_COMPANY_RE = re.compile(
    r"\b(?:for|with)\s+([A-Z][a-zA-Z&.'\- ]{2,40}?)(?:\s+this\s+week|\s+last\s+|\s+daily|\s+now|\s*$)",
    flags=re.IGNORECASE,
)

_ACRONYM_NON_SUBJECTS = frozenset(
    {"ceo", "cfo", "cto", "ai", "etf", "etfs", "ipo", "cpi", "gdp", "fed", "fomc", "eps", "pe", "usa", "us"}
)


def _candidate_sort_key(item: tuple[int, str]) -> int:
    """Sort one lexical candidate by its objective position."""
    return item[0]


def _objective_chart_candidates(objective: str) -> list[str]:
    """Return explicit lexical stock labels in objective order without resolving them."""
    objective = re.sub(r"^\[Today UTC \d{4}-\d{2}-\d{2}\]\s*", "", objective).strip()
    labels: list[tuple[int, str]] = [
        (match.start(), match.group()) for match in _OBJECTIVE_TICKER_RE.finditer(objective)
    ]
    companies: list[tuple[int, str]] = [
        (objective.find(label), label) for label in _company_token_candidates(objective)
    ]
    matches: list[re.Match[str]] = [
        *_OBJECTIVE_COMPANY_RE.finditer(objective),
        *re.finditer(
            r"\b(?i:buy|sell|hold|compare|invest\s+in|how\s+is|how\s+are|tell\s+me\s+about)\s+"
            r"([A-Z][a-zA-Z&.'\-]*(?:\s+(?:(?i:and|vs\.?|versus|with|to)\s+)?[A-Z][a-zA-Z&.'\-]*)*)",
            objective,
        ),
    ]
    for match in matches:
        for company in re.split(r"\s+(?:and|vs\.?|versus|with|to)\s+", match.group(1), flags=re.IGNORECASE):
            companies.append((objective.find(company, match.start(1)), company.strip()))
    for match in re.finditer(
        r"\b([A-Z][a-zA-Z&.'\-]*(?:\s+[A-Z][a-zA-Z&.'\-]*)*)\s+(?i:versus|vs\.?)\s+"
        r"([A-Z][a-zA-Z&.'\-]*(?:\s+[A-Z][a-zA-Z&.'\-]*)*)",
        objective,
    ):
        companies.extend((match.start(group), match.group(group)) for group in (1, 2))
    excluded = {
        "hi",
        "hey",
        "hello",
        "good",
        "morning",
        "afternoon",
        "evening",
        "today",
        "tomorrow",
        "yesterday",
        "utc",
        "est",
        "edt",
        "pst",
        "pdt",
        "et",
        "am",
        "pm",
        *_ACRONYM_NON_SUBJECTS,
    }
    labels.extend(
        (position, label)
        for position, label in companies
        if not any(word.lower() in excluded for word in label.split())
    )
    unique: dict[str, str] = {}
    for _, label in sorted(labels, key=_candidate_sort_key):
        normalized = " ".join(label.split())
        if _company_token_allowed(normalized) and normalized.split()[0].lower() not in {
            "buy",
            "sell",
            "hold",
            "compare",
            "invest",
            "tell",
            "about",
        }:
            unique.setdefault(normalized.casefold(), normalized)
    return list(unique.values())


def _objective_ticker(objective: object) -> str | None:
    """Explicit ticker token from the objective (uppercase 2-5ch EDGAR hit, never first-word fallback)."""
    if not isinstance(objective, str) or not objective.strip():
        return None
    from app.tools import _FINRA_ORG_WORDS as _ORG
    from app.tools import _resolve_company_to_ticker as _resolve

    for _tok in _OBJECTIVE_TICKER_RE.findall(objective):
        if _tok.upper() in _ORG or _tok.upper() in ("SHO", "REG"):
            continue
        _rt = _resolve(_tok)
        if _rt is not None:
            return _rt
    return None


def _objective_company(objective: object) -> tuple[str, str] | None:
    """(company_name, ticker) from a for-<Company> objective via EDGAR, else None."""
    if not isinstance(objective, str) or not objective.strip():
        return None
    from app.tools import _FINRA_ORG_WORDS as _ORG
    from app.tools import _resolve_company_to_ticker as _resolve

    _qm = _OBJECTIVE_COMPANY_RE.search(objective.strip())
    _cand = _qm.group(1).strip() if _qm else None
    if not _cand or _cand.lower() in _COMPANY_TOKEN_STOP or _cand.upper() in _ORG:
        return None
    _rt = _resolve(_cand)
    return (_cand, _rt) if _rt is not None else None


def _objective_company_token(objective: object) -> tuple[str, str] | None:
    """(name, ticker) for a possessive/insider-phrase/multi-word company mention, else None."""
    if not isinstance(objective, str) or not objective.strip():
        return None
    from app.tools import _resolve_company_to_ticker as _resolve

    candidates = _company_token_candidates(objective.strip())
    for candidate in candidates:
        if not _company_token_allowed(candidate):
            continue
        ticker = _resolve(candidate)
        if ticker is not None:
            return candidate, ticker
    return None


def _company_token_candidates(text: str) -> list[str]:
    """Read possessive and multiword names before insider phrases."""
    # ponytail: possessive ("Apple's") + insider/executive phrase ("Tesla
    # insiders") + multi-word ("Apple Inc") only; single bare Title-case words
    # never seed (Growth->VREOF misresolve seeds when it must withhold).
    _cands: list[str] = []
    _cands += re.findall(r"\b([A-Z][a-zA-Z&.\-]{2,40})'s\b", text)
    for _mw in re.findall(r"\b([A-Z][a-z]{2,40}(?: [A-Z][a-z]{2,40})+)\b", text):
        _cands.append(_mw)
        _parts = _mw.split()
        while _parts and _parts[0].lower() in _COMPANY_TOKEN_STOP:
            _parts = _parts[1:]
        if _parts and " ".join(_parts) != _mw:
            _cands.append(" ".join(_parts))
    _append_insider_company_tokens(text, _cands)
    return _cands


def _append_insider_company_tokens(text: str, _cands: list[str]) -> None:
    """Retain the preceding title word before each insider or executive name."""
    for _pat in (
        r"\b([A-Z][a-z]{2,40}) insiders?\b",
        r"\b([A-Z][a-z]{2,40}) executives?\b",
        r"\b([A-Z][a-z]{2,40}) team\b",
    ):
        for _m in re.finditer(_pat, text):
            _pre = re.search(r"([A-Z][a-z]{2,40})\s+$", text[: _m.start(1)])
            if _pre is not None and _pre.group(1) != _m.group(1):
                _cands.append(_pre.group(1))
            _cands.append(_m.group(1))


def _company_token_allowed(candidate: str) -> bool:
    """Reject organization words, suffixes, and leading question words."""
    from app.tools import _FINRA_ORG_WORDS

    if candidate.lower() in _COMPANY_TOKEN_STOP or candidate.lower() in _COMPANY_SUFFIX:
        return False
    if " " in candidate and candidate.split()[0].lower() in _COMPANY_TOKEN_STOP:
        return False
    return candidate.upper() not in _FINRA_ORG_WORDS and candidate.upper() not in ("SHO", "REG", "EPS", "SEC", "FINRA")


def _objective_subject_ticker(objective: object, supplied_ticker: object = None) -> str | None:
    """Prefer the explicit objective subject. Use chart context when no explicit label resolves."""
    if isinstance(objective, str):
        objective = re.sub(r"^\[Today UTC \d{4}-\d{2}-\d{2}\]\s*", "", objective).strip()
    chart = supplied_ticker.strip().upper() if isinstance(supplied_ticker, str) and supplied_ticker.strip() else None
    if chart is not None and isinstance(objective, str):
        # Acronyms lose to the chart before any explicit path runs, so AI/CFO/IPO/FED/US
        # (real tickers) never beat the displayed stock on vague prompts.
        for _tok in _OBJECTIVE_TICKER_RE.findall(objective):
            if _tok.lower() in _ACRONYM_NON_SUBJECTS:
                return chart
        for candidate in _objective_chart_candidates(objective):
            if candidate.lower() in _ACRONYM_NON_SUBJECTS:
                return chart
    _ticker = _objective_ticker(objective)
    if _ticker is not None:
        return _ticker
    _comp = _objective_company(objective)
    if _comp is not None:
        return _comp[1]
    _tok = _objective_company_token(objective)
    if _tok is not None:
        return _tok[1]
    if isinstance(objective, str):
        from app.tools import _resolve_company_to_ticker

        candidates = _objective_chart_candidates(objective)
        for candidate in candidates:
            if candidate.lower() in _ACRONYM_NON_SUBJECTS:
                continue
            resolved = _resolve_company_to_ticker(candidate)
            if resolved is not None:
                return resolved
        bare = _bare_company_ticker(objective)
        if bare is not None:
            return bare
    return chart


def _bare_company_ticker(objective: str) -> str | None:
    """Ticker for a bare Title-case word that equals the first word of the top SEC company name.

    Microsoft -> MSFT and Nvidia -> NVDA resolve; Growth (Vireo Growth) and Best never do.
    """
    # ponytail: first-word equality only; a word that starts a real company name still
    # resolves ("General" -> General Mills). Add an alias table if that misfires in practice.
    from app.sec.client import find_sec_company

    for word in re.findall(r"\b[A-Z][a-z]{2,40}\b", objective):
        if word.lower() in _ACRONYM_NON_SUBJECTS or not _company_token_allowed(word):
            continue
        try:
            rows = find_sec_company(word, limit=1)
        except Exception as exc:  # noqa: BLE001 - same best-effort boundary as _edgar_ticker; no subject on failure
            logger.debug("bare company lookup failed word=%s: %s", word, exc)
            continue
        if not rows:
            continue
        name = str(rows[0].get("name") or "").replace(",", " ").split()
        tickers = rows[0].get("tickers")
        if name and name[0].casefold() == word.casefold() and isinstance(tickers, list) and tickers:
            return str(tickers[0]).strip().upper()
    return None


_SUBJECT_CANDIDATE_MAX = 4
"""JEV asks one yes/no question per candidate. Four covers a three-company prompt
plus one stray capitalized word, and keeps the decision prompt small."""


def _objective_subject_candidates(objective: str) -> list[tuple[str, str, str]]:
    """(label, SEC company name, ticker) for each capitalized mention whose index top hit resolves.

    Candidates keep mention order and are capped at _SUBJECT_CANDIDATE_MAX.
    """
    from app.sec.client import find_sec_company

    if not isinstance(objective, str) or not objective.strip():
        return []
    text = re.sub(r"^\[Today UTC \d{4}-\d{2}-\d{2}\]\s*", "", objective).strip()
    positioned: dict[str, int] = {}
    for label in _objective_chart_candidates(text):
        normalized = " ".join(label.split())
        positioned.setdefault(normalized, text.find(normalized))
    for match in re.finditer(r"\b[A-Z][a-z][A-Za-z&.'’\-]{0,39}\b", text):
        positioned.setdefault(re.sub(r"['’]s$", "", match.group()), match.start())
    candidates: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for label in sorted(positioned, key=positioned.__getitem__):
        if len(candidates) >= _SUBJECT_CANDIDATE_MAX:
            break
        if not _company_token_allowed(label):
            continue
        if any(word.lower() in _ACRONYM_NON_SUBJECTS for word in label.split()):
            continue
        try:
            rows = find_sec_company(label, limit=1)
        except Exception as exc:  # noqa: BLE001 - best-effort boundary, no candidate on failure
            logger.debug("subject candidate lookup failed label=%s: %s", label, exc)
            continue
        if not rows:
            continue
        name = " ".join(str(rows[0].get("name") or "").split())
        tickers = rows[0].get("tickers")
        ticker = str(tickers[0]).strip().upper() if isinstance(tickers, list) and tickers else ""
        if not name or not ticker or ticker in seen:
            continue
        try:
            validate_ticker(ticker, "<subject candidate>")
        except ValueError:
            continue
        seen.add(ticker)
        candidates.append((label, name, ticker))
    return candidates


_NODE_ONLY_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9}) only\)\s*$")
"""A per-company node of a comparison ends with "(AMD only)": that ticker is its subject."""


def question_subjects(question: str) -> list[str]:
    """Distinct tickers a question names, in mention order (acronyms such as AI never count)."""
    from app.tools import _resolve_company_to_ticker

    tickers = [
        ticker
        for candidate in _objective_chart_candidates(question)
        if candidate.lower() not in _ACRONYM_NON_SUBJECTS
        and (ticker := _resolve_company_to_ticker(candidate)) is not None
    ]
    return list(dict.fromkeys(tickers))


_SOURCE_FAMILY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Keys are app/tools.py DOMAIN_DESCRIPTIONS domains; terms are the words a request uses to name one.
    ("sec", ("sec", "edgar", "filing", "10-k", "10-q", "8-k")),
    (
        "fundamentals",
        (
            "eps",
            "fundamentals",
            "fundamental",
            "balance sheet",
            "income statement",
            "cash flow",
            "revenue",
            "net income",
            "gross margin",
            "xbrl",
        ),
    ),
    ("valuation", ("valuation", "multiples", "overvalued", "undervalued", "p/e")),
    ("finra", ("short interest", "short volume", "short squeeze", "threshold securities", "finra")),
    ("insider", ("insider", "form 4", "form 144", "insider sales", "insider buying")),
    ("events", ("material events", "corporate events")),
    ("web", ("web", "news", "sentiment", "headlines", "web source")),
    ("alternative", ("trend evidence", "trends", "consumer trend", "alternative data", "social media")),
    ("market", ("option chain", "options chain", "greeks", "expiration", "index membership", "quote", "spot price")),
    ("analyst", ("analyst", "estimates", "price target", "consensus")),
    ("ownership", ("ownership", "beneficial owner", "stake", "holder", "5% holder")),
    ("offerings", ("offering", "dilution", "capital raise")),
    ("macro", ("cpi", "inflation", "unemployment", "gdp", "fed", "federal reserve", "jobless", "payroll")),
    ("transactions", ("acquisition", "merger", "tender offer", "deal status")),
    ("governance", ("proxy", "governance", "board of directors", "proxy vote")),
)
"""Data family -> words naming it. A plural of the last word also matches."""

_SOURCE_FAMILY_RE: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (family, re.compile(r"\b(?:" + "|".join(f"{re.escape(term)}s?" for term in terms) + r")\b"))
    for family, terms in _SOURCE_FAMILY_TERMS
)


def question_source_families(question: str) -> list[str]:
    """Distinct data families a question names, in mention order (domain keys from DOMAIN_DESCRIPTIONS)."""
    low = question.lower()
    found = [(match.start(), family) for family, pattern in _SOURCE_FAMILY_RE if (match := pattern.search(low))]
    return [family for _, family in sorted(found)]


def _node_subject_ticker(question: object, objective: object, supplied_ticker: object = None) -> str | None:
    """Use one explicit node subject, otherwise preserve the session subject."""
    if isinstance(question, str):
        only = _NODE_ONLY_RE.search(question)
        if only is not None:
            return only.group(1)
        subjects = question_subjects(question)
        if len(subjects) == 1:
            return subjects[0]
    return _objective_subject_ticker(objective, supplied_ticker)


_THESIS_ID_TOOLS = frozenset({"thesis_show", "thesis_refine", "thesis_watch", "thesis_journal", "thesis_status"})
"""Thesis tools that address one existing thesis by its `id`."""

_THESIS_ID_RE = re.compile(r"thesis:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def _fallback_thesis_args(tool_name: str, objective: object) -> dict[str, JSONValue] | None:
    """Seed a thesis tool's `id` from the objective's single `thesis:<uuid>` token.

    Zero or several ids seed nothing: an ambiguous objective never picks one.
    """
    if tool_name not in _THESIS_ID_TOOLS or not isinstance(objective, str):
        return None
    found = set(_THESIS_ID_RE.findall(objective))
    return {"id": found.pop()} if len(found) == 1 else None


def _fallback_sec_args(
    tool_name: str, objective: object, supplied_ticker: object = None, *, subject: str | None = None
) -> dict[str, JSONValue] | None:
    """Seed arguments from the explicit subject, else the supplied chart ticker.

    ``subject`` is a node's own resolved subject ("(MSFT only)"): it outranks the
    objective's first company, so a per-company node never seeds the other company.
    """
    if (
        tool_name not in _SEC_IDENTIFIER_TOOLS
        and tool_name not in _SHO_FALLBACK_TOOLS
        and tool_name not in _TICKER_FALLBACK_TOOLS
        and tool_name not in ("set_chart_symbol", "find_sec_entities", "find_sec_entities_bounded")
    ):
        return _fallback_entity_accession(tool_name, objective, None)
    ticker = subject or _objective_subject_ticker(objective, supplied_ticker)
    if tool_name in _SEC_IDENTIFIER_TOOLS:
        return _fallback_sec_identifier(tool_name, objective, ticker)
    if tool_name in _SHO_FALLBACK_TOOLS:
        company = _objective_company(objective)
        if company is not None and (subject is None or company[1] == subject):
            return {"ticker": company[1], "company_name": company[0]}
        if ticker is not None:
            return {"ticker": ticker}
    if tool_name in _TICKER_FALLBACK_TOOLS or tool_name == "set_chart_symbol":
        return _fallback_ticker_args(tool_name, objective, ticker)
    return _fallback_entity_accession(tool_name, objective, ticker)


def _revenue_listing_forms(objective: object) -> list[JSONValue] | None:
    """Use explicitly named forms before the forms implied by a revenue request."""
    named = _material_forms(objective)
    if named:
        return [form for form in sorted(named)]
    if _latest_revenue_objective(objective):
        return (
            ["10-Q"]
            if _TEN_Q_RE.search(str(objective).lower()) and _revenue_quarter_objective(objective)
            else list(_REVENUE_FORMS)
        )
    if _revenue_quarter_objective(objective):
        return ["10-Q"] if _TEN_Q_RE.search(str(objective).lower()) else ["10-Q", "10-K", "8-K"]
    return None


def _fallback_sec_identifier(tool_name: str, objective: object, ticker: str | None) -> dict[str, JSONValue] | None:
    """Seed SEC listing identifiers or search tickers from the selected subject."""
    if ticker is None:
        return None
    if tool_name == "list_sec_filings":
        forms = _revenue_listing_forms(objective)
        return {"identifier": ticker} if forms is None else {"identifier": ticker, "forms": forms}
    if isinstance(objective, str) and objective.strip():
        return {"ticker": ticker, "query": objective.strip()[:200]}
    return {"ticker": ticker}


def _fallback_ticker_args(tool_name: str, objective: object, ticker: str | None) -> dict[str, JSONValue] | None:
    """Add the EPS metric only for fundamental requests that explicitly name it."""
    if ticker is None:
        return None
    if tool_name == "get_fundamentals" and isinstance(objective, str) and "eps" in objective.lower():
        return {"ticker": ticker, "metric": "eps"}
    return {"ticker": ticker}


def _fallback_entity_accession(tool_name: str, objective: object, ticker: str | None) -> dict[str, JSONValue] | None:
    """Seed entity queries or an explicit filing accession without adding a date window."""
    if (
        tool_name in ("find_sec_entities", "find_sec_entities_bounded")
        and isinstance(objective, str)
        and objective.strip()
    ):
        if ticker is not None:
            return {"query": ticker}
        return {"query": objective.strip()[:200]}
    if tool_name == "get_sec_filing" and isinstance(objective, str):
        match = _ACCESSION_TOKEN_RE.search(objective)
        if match is not None:
            try:
                from app.sec.discovery.service import normalize_accession_no

                return {"accession_no": normalize_accession_no(match.group(1))}
            except ValueError, ImportError:
                return {"accession_no": match.group(1)}
    return None


_DATE_ARG_KEYS = ("as_of",)


def _as_day(value: object) -> str | None:
    """YYYY-MM-DD day for str/datetime/date values; None when absent."""
    if isinstance(value, str):
        text = value.strip()
        return text[:10] if text else None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return str(iso())[:10] or None
        except Exception:  # noqa: BLE001 - non-date-likes stay ungrounded
            return None
    return None


def _scrub_ungrounded_date_args(
    filled: dict[str, JSONValue],
    objective: object,
    question: object = None,
    scope: object = None,
    session_as_of: object = None,
    required: object = None,
    keys: tuple[str, ...] = _DATE_ARG_KEYS,
) -> list[str]:
    """Drop ungrounded date keys. Preserve explicit dates, scope dates, and required keys."""
    dropped: list[str] = []
    scoped = _date_grounding_text(objective, question, scope)
    scope_vals = _date_scope_days(scope)
    session_day = _as_day(session_as_of)
    today = datetime.now(UTC).date().isoformat()
    required_keys: set[str] = (
        {k for k in required if isinstance(k, str)} if isinstance(required, (list, tuple, set, frozenset)) else set()
    )
    for key in keys:
        if key in required_keys:
            continue
        raw = filled.get(key)
        if not isinstance(raw, str):
            continue
        if _date_value_grounded(key, raw.strip(), scoped, scope_vals, session_day, today):
            continue
        del filled[key]
        dropped.append(key)
    return dropped


def _date_grounding_text(objective: object, question: object, scope: object) -> str:
    """Collect only nonempty request and raw scope text for date grounding."""
    texts = [text for text in (objective, question) if isinstance(text, str) and text.strip()]
    scope_raw = scope.get("raw") if isinstance(scope, dict) else None
    if isinstance(scope_raw, str) and scope_raw.strip():
        texts.append(scope_raw)
    return " ".join(texts)


def _date_scope_days(scope: object) -> set[str]:
    """Read the explicit date fields from a scope."""
    days: set[str] = set()
    if isinstance(scope, dict):
        for key in ("as_of", "start", "end"):
            day = _as_day(scope.get(key))
            if day:
                days.add(day)
    return days


def _date_value_grounded(
    key: str, value: str, scoped: str, scope_days: set[str], session_day: str | None, today: str
) -> bool:
    """Reject relative values before accepting a request date, scope date, or today's cutoff."""
    from app.tools import _RELATIVE_DATE_RE

    if not value or _RELATIVE_DATE_RE.search(value.lower()):
        return False
    day = value[:10]
    if key == "as_of" and day == today:
        return True
    if day in scope_days:
        return True
    if session_day is not None and day == session_day:
        return True
    return day in scoped or value in scoped


def _repair_tool_arguments(
    tool_name: str,
    generated_args: Mapping[str, JSONValue],
    objective: object,
    sid: str,
    nid: str,
    question: object = None,
    scope: object = None,
    session_as_of: object = None,
    schema: Mapping[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    """Seed/scrub Needle args: objective ticker/company, latest forms, placeholder scrub."""
    filled = dict(generated_args)
    ticker_value = filled.get("ticker")
    _seed_objective_company(filled, objective, tool_name, sid, nid)
    _seed_listing_forms(filled, tool_name, objective, sid, nid)
    _seed_listing_identifier(filled, tool_name, objective, sid, nid, question)
    _seed_search_arguments(filled, tool_name, objective, sid, nid)
    _repair_org_word_ticker(filled, ticker_value, tool_name, objective, sid, nid)
    _scrub_relative_trade_date(filled, tool_name, sid, nid)
    _scrub_unrequested_dates(filled, tool_name, objective, question, scope, session_as_of, sid, nid)
    _scrub_single_day_window(filled, tool_name, sid, nid)
    _seed_cutoff_as_of(filled, tool_name, schema, session_as_of, sid, nid)
    _seed_since_window(filled, tool_name, schema, objective, question, scope, session_as_of, sid, nid)
    _seed_revenue_read(filled, tool_name, objective, sid, nid)
    _seed_risk_factor_section(filled, tool_name, objective, sid, nid)
    _seed_finra_dataset(filled, tool_name, sid, nid)
    return filled


def _log_args_stage(stage: str, sid: str, nid: str, tool_name: str, keys: str) -> None:
    """Emit one toolflow stage line for a seeded or dropped argument key."""
    logger.info("toolflow %s sid=%s nid=%s tool=%s keys=%s", stage, sid, nid, tool_name, keys)


def _log_identifier_seed(sid: str, nid: str, tool_name: str, identifier: str) -> None:
    """Emit the identifier line for one seeded SEC listing identifier."""
    logger.info("toolflow args_seed sid=%s nid=%s tool=%s identifier=%s", sid, nid, tool_name, identifier)


def _seed_objective_company(
    filled: dict[str, JSONValue], objective: object, tool_name: str, sid: str, nid: str
) -> None:
    """Seed company_name and ticker from one for-<Company> objective mention."""
    if filled.get("company_name") is not None:
        return
    company = _objective_company(objective)
    if company is None:
        return
    filled["company_name"] = company[0]
    filled["ticker"] = company[1]
    logger.info(
        "toolflow args_seed sid=%s nid=%s tool=%s company=%s ticker=%s",
        sid,
        nid,
        tool_name,
        company[0],
        company[1],
    )


def _seed_listing_forms(filled: dict[str, JSONValue], tool_name: str, objective: object, sid: str, nid: str) -> None:
    """Seed the listing form set. An explicit latest 10-Q request must not list adjacent forms.

    A non-10-Q form set that the call already carries survives; the latest quarterly
    or annual request replaces it.
    """
    if tool_name != "list_sec_filings":
        return
    forms = _revenue_listing_forms(objective)
    if forms is None:
        return
    selected = filled.get("forms")
    if (
        isinstance(selected, list)
        and selected
        and all(isinstance(form, str) and form.endswith("/A") and form.removesuffix("/A") in forms for form in selected)
    ):
        return
    if filled.get("forms") and not _material_forms(objective):
        return
    filled["forms"] = forms
    _log_args_stage("args_seed", sid, nid, tool_name, "forms")


def _seed_listing_identifier(
    filled: dict[str, JSONValue], tool_name: str, objective: object, sid: str, nid: str, question: object = None
) -> None:
    """Replace a placeholder, date-like, organization or mismatched listing identifier."""
    if tool_name != "list_sec_filings":
        return
    want = _node_subject_ticker(question, objective)
    raw = filled["identifier"].strip() if isinstance(filled.get("identifier"), str) else ""
    if not _stale_listing_identifier(raw, want):
        return
    seeded = _fallback_sec_identifier(tool_name, objective, want)
    if seeded is None or not isinstance(seeded.get("identifier"), str):
        return
    filled["identifier"] = seeded["identifier"]
    if isinstance(seeded.get("forms"), list) and not filled.get("forms"):
        filled["forms"] = seeded["forms"]
    _log_identifier_seed(sid, nid, tool_name, seeded["identifier"])


def _stale_listing_identifier(raw: str, want: str | None) -> bool:
    """Placeholder, date-like, organization, or short-CIK identifier that disagrees with the subject."""
    from app.tools import _FINRA_ORG_WORDS as _ORG
    from app.tools import _date_like_form

    if not raw or "YYYY" in raw.upper():
        return True
    if raw.upper() in _ORG or _date_like_form(raw):
        return True
    if raw.isdigit():
        return len(raw) < 7 and want is not None
    return want is not None and raw.upper() != want.upper()


def _seed_search_arguments(filled: dict[str, JSONValue], tool_name: str, objective: object, sid: str, nid: str) -> None:
    """Fill missing search selectors. Withhold when the objective names no subject."""
    if tool_name not in ("search_sec_filings", "search_sec_filings_bounded"):
        return
    if any(filled.get(key) for key in _SEARCH_GROUNDING_KEYS):
        return
    if not _seed_search_selectors(filled, tool_name, objective):
        return
    _log_args_stage("args_seed", sid, nid, tool_name, ",".join(sorted(k for k in ("ticker", "query") if filled.get(k))))


def _seed_search_selectors(filled: dict[str, JSONValue], tool_name: str, objective: object) -> bool:
    """Copy objective-derived selectors into empty fields; False when the objective names no subject."""
    seeded = _fallback_sec_args(tool_name, objective)
    if seeded is None or (isinstance(objective, str) and not _objective_subject_ticker(objective)):
        return False
    for key in ("ticker", "query", "cik", "company_name"):
        if isinstance(seeded.get(key), str) and not filled.get(key):
            filled[key] = seeded[key]
    return True


def _repair_org_word_ticker(
    filled: dict[str, JSONValue], ticker_value: object, tool_name: str, objective: object, sid: str, nid: str
) -> None:
    """Resolve a ticker that names the organization itself, never a company."""
    from app.tools import _FINRA_ORG_WORDS as _ORG

    if tool_name == "list_sec_filings" or not isinstance(ticker_value, str):
        return
    if ticker_value.strip().upper() not in _ORG:
        return
    seeded = _fallback_sec_args(tool_name, objective)
    if seeded is None or not isinstance(seeded.get("ticker"), str):
        return
    filled["ticker"] = seeded["ticker"]
    if isinstance(seeded.get("company_name"), str) and not filled.get("company_name"):
        filled["company_name"] = seeded["company_name"]
    logger.info(
        "toolflow args_resolve sid=%s nid=%s tool=%s ticker=%s",
        sid,
        nid,
        tool_name,
        seeded["ticker"],
    )


def _scrub_relative_trade_date(filled: dict[str, JSONValue], tool_name: str, sid: str, nid: str) -> None:
    """Drop a relative tradeDate: the tool reads one business day per call."""
    if tool_name != "get_reg_sho_volume":
        return
    trade = filled.get("tradeDate")
    if not isinstance(trade, str):
        return
    from app.tools import _RELATIVE_DATE_RE

    day = trade.strip().lower()
    if not (day and _RELATIVE_DATE_RE.search(day)):
        return
    del filled["tradeDate"]
    _log_args_stage("args_scrub", sid, nid, tool_name, "tradeDate")


def _scrub_unrequested_dates(
    filled: dict[str, JSONValue],
    tool_name: str,
    objective: object,
    question: object,
    scope: object,
    session_as_of: object,
    sid: str,
    nid: str,
) -> None:
    """Drop date args that no request text, scope window, or cutoff grounds."""
    date_keys = _DATE_ARG_KEYS
    if tool_name == "list_sec_filings" and (
        _latest_revenue_objective(objective)
        or (
            _revenue_quarter_objective(objective)
            and isinstance(objective, str)
            and re.search(r"\b(?:latest|most recent)\b", objective.lower())
        )
    ):
        date_keys += ("start_date", "end_date")
    for dropped in _scrub_ungrounded_date_args(filled, objective, question, scope, session_as_of, keys=date_keys):
        _log_args_stage("args_scrub", sid, nid, tool_name, dropped)


def _scrub_single_day_window(filled: dict[str, JSONValue], tool_name: str, sid: str, nid: str) -> None:
    """Drop a one-day start_date: a start equal to end_date lists only that day and finds nothing."""
    if tool_name != "list_sec_filings":
        return
    start = filled.get("start_date")
    if start is None or start != filled.get("end_date"):
        return
    del filled["start_date"]
    _log_args_stage("args_scrub", sid, nid, tool_name, "start_date")


def _seed_cutoff_as_of(
    filled: dict[str, JSONValue],
    tool_name: str,
    schema: Mapping[str, JSONValue] | None,
    session_as_of: object,
    sid: str,
    nid: str,
) -> None:
    """Seed the request cutoff as a PIT-scoped tool's own as_of when the call carries none."""
    if schema is None or filled.get("as_of") is not None:
        return
    day = _pit_as_of_seed(tool_name, schema, session_as_of)
    if day is None:
        return
    filled["as_of"] = day
    _log_args_stage("args_seed", sid, nid, tool_name, "as_of")


_EVENTS_LOOKBACK_DAYS = 30
"""Material-events window the chat prefetch also uses (needle-harness/app/api/agent/route.ts NEWS_DAYS)."""


def _seed_since_window(
    filled: dict[str, JSONValue],
    tool_name: str,
    schema: Mapping[str, JSONValue] | None,
    objective: object,
    question: object,
    scope: object,
    session_as_of: object,
    sid: str,
    nid: str,
) -> None:
    """Give a schema-required `since` the 30-day window ending at the request cutoff when the request names no date.

    An explicit request date wins. So does a scope window: those bounds already
    say which events the request wants.
    """
    required = schema.get("required") if isinstance(schema, Mapping) else None
    if not isinstance(required, list) or "since" not in required:
        return
    window = isinstance(scope, dict) and bool(scope.get("start") or scope.get("end"))
    if window or _DATE_CONSTRAINT_RE.search(_date_grounding_text(objective, question, scope).lower()):
        return
    try:
        base = date.fromisoformat(_as_day(session_as_of) or "")
    except ValueError:
        base = datetime.now(UTC).date()
    since = (base - timedelta(days=_EVENTS_LOOKBACK_DAYS)).isoformat()
    if filled.get("since") == since:
        return
    filled["since"] = since
    _log_args_stage("args_seed", sid, nid, tool_name, "since")


def _seed_revenue_read(filled: dict[str, JSONValue], tool_name: str, objective: object, sid: str, nid: str) -> None:
    """Fill latest revenue reads: revenue query and a wide limit."""
    if tool_name != "get_sec_document":
        return
    revenue = _revenue_read_objective(objective)
    if revenue and not filled.get("query"):
        filled["query"] = "revenue"
        _log_args_stage("args_seed", sid, nid, tool_name, "query")
    if revenue and not filled.get("limit") and not filled.get("max_chars"):
        filled["limit"] = 32000
        _log_args_stage("args_seed", sid, nid, tool_name, "limit")


def _seed_risk_factor_section(
    filled: dict[str, JSONValue], tool_name: str, objective: object, sid: str, nid: str
) -> None:
    """Read the risk-factor section only when the request asks for it and names no query."""
    if tool_name != "get_sec_document":
        return
    low = objective.lower() if isinstance(objective, str) else ""
    if ("item 1a" in low or "risk factor" in low) and not filled.get("section") and not filled.get("query"):
        filled["section"] = "Item 1A"
        _log_args_stage("args_seed", sid, nid, tool_name, "section")


def _seed_finra_dataset(filled: dict[str, JSONValue], tool_name: str, sid: str, nid: str) -> None:
    """Default the FINRA dataset only when the call names none."""
    if tool_name != "query_finra":
        return
    dataset = filled.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return
    filled["dataset"] = "consolidatedShortInterest"
    _log_args_stage("args_seed", sid, nid, tool_name, "dataset")


_SEARCH_GROUNDING_KEYS = (
    "query",
    "ticker",
    "cik",
    "company_name",
    "person_name",
    "domain",
    "accession_no",
    "security_identifier",
)
