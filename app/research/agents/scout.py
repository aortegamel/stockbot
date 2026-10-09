"""Bounded SEC scouts: temporary assignments, not personas.

Three role templates (filings/material-event, financial/XBRL trend,
risk-factor/language-diff) driven by a generic research context. Each scout
runs an as_of-filtered latest-filing baseline, its assigned context queries,
its unscoped counterparty queries (no ticker, no cik: EDGAR full-text search is
global), and role tools; search/navigation results are never evidence, so the
documents those hits name are opened (``get_sec_document``) and findings are
drafted only from the raw passages that come back. The opened passages then
drive expansion: the materially novel queries they name are searched, their
newly surfaced documents opened, and that repeats until no novel query remains.
Stops are info-based (no new material queries, repeats yield nothing new,
baseline reviewed, explicit tool limit) -- never count-based; there is no cap
on searches, filing reads, document reads, exhibit reads, or waves.

Fake-model sketch (no live calls): fake ``dispatch(name, args)`` returns
``{"evidence_id": ..., "known_at": ...}`` dicts; fake ``model(prompt)``
returns canned text; call ``run_scout`` and assert
``finding_ids``/``evidence_ids``/``follow_up_requests``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from . import GroundedClaim, ResearchRequest, parse_grounded_claims_tolerant

ScoutRole = Literal["filings", "financials", "risk"]

ROLE_PROMPTS: dict[str, str] = {
    "filings": (
        "Temporary assignment: filings/material-event scout. List material "
        "events (8-K/6-K, offerings, insider transactions) for the tickers in "
        "scope. Use the research context and the as_of-filtered latest-filing "
        "baseline (segments, customers, suppliers, competition, risks, "
        "terminology) below to drive targeted searches. Cite evidence ids "
        "only; record gaps as unknowns."
    ),
    "financials": (
        "Temporary assignment: financial/XBRL-trend scout. Summarize reported "
        "XBRL/financial-statement trends for the tickers in scope. Use the "
        "research context and the as_of-filtered latest-filing baseline below "
        "to drive targeted searches. Never recalculate tool-computed metrics; "
        "cite evidence ids only."
    ),
    "risk": (
        "Temporary assignment: risk-factor/language-diff scout. Compare risk "
        "factor language across filings and flag new/removed/softened "
        "language. Use the research context and the as_of-filtered "
        "latest-filing baseline below to drive targeted searches. Quote "
        "briefly with evidence ids; gaps go to unknowns."
    ),
}


def normalize_query(query: str) -> str:
    """Canonical query key: lowercase + whitespace-collapse for dedup."""
    return " ".join(query.lower().split())


MAX_CHILDREN = 0
ALLOWED_DOMAIN = "SEC"

_ROLE_TOOLS: dict[str, tuple[tuple[str, dict[str, str]], ...]] = {
    # ponytail: hard-coded role tools (deterministic, policy-gated); model-driven discovery/selection/execution if broad live coverage requires it.
    "filings": (("search_sec_filings", {}), ("list_sec_filings", {}), ("get_material_events", {})),
    "financials": (("search_sec_filings", {}), ("get_xbrl_facts", {"concept": "Revenues"}), ("list_sec_filings", {})),
    "risk": (("diff_risk_factors", {}), ("diff_sec_filings", {}), ("list_sec_filings", {})),
}


def _role_arguments(tool: str, ticker: str, as_of: str) -> dict[str, object]:
    """Schema-correct args: ticker identity, as_of PIT, since window, XBRL concept."""
    args: dict[str, object] = {"ticker": ticker, "identifier": ticker}
    text = as_of.strip() if isinstance(as_of, str) else ""
    bounded = bool(text) and text.lower() != "unbounded"
    if bounded:
        args["as_of"] = as_of
    if tool == "get_material_events":
        # `since` is required by the tool schema (app/tools.py: "required": ["ticker", "since"]),
        # so it goes out on every call: the one-year window for a bounded as_of, else
        # `_window_start`'s documented floor (2024-01-01) because an unbounded session has no
        # cutoff to measure a lookback back from.
        args["since"] = _window_start(as_of)
    if tool == "get_xbrl_facts":
        args["concept"] = "Revenues"
    return args


def _window_start(as_of: str) -> str:
    """One-year lookback window start (YYYY-MM-DD) for since-gated tools.

    No bounded as_of (unbounded sentinel, blank, non-ISO) leaves nothing to measure a
    year back from, and since-gated callers still need a date, so the window floors at
    2024-01-01 -- the default `since` the routing harnesses and evals already use.
    """
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    try:
        end = _dt.fromisoformat(as_of)
        start = (end - _td(days=365)).date().isoformat()
        if len(start) == 10:
            return start
    except ValueError:
        pass
    return "2024-01-01"


DispatchFn = Callable[[str, dict[str, object]], dict[str, object]]
ModelFn = Callable[[str], str]


@dataclass
class ScoutAssignment:
    assignment_id: str
    session_id: str
    as_of: str
    role: ScoutRole
    question: str
    tickers: list[str] = field(default_factory=list)
    max_tool_calls: int | None = None
    time_budget_s: float | None = None
    allowed_domain: str = ALLOWED_DOMAIN
    context: dict[str, object] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    unscoped_queries: list[str] = field(default_factory=list)
    baseline: list[str] = field(default_factory=list)


@dataclass
class ScoutResult:
    assignment_id: str
    session_id: str
    coverage: str
    findings: list[GroundedClaim] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    follow_up_requests: list[ResearchRequest] = field(default_factory=list)


_LIST_CONTEXT_KEYS = (
    "primary_entities",
    "related_entities",
    "industries",
    "products",
    "technologies",
    "concepts",
    "risks",
    "catalysts",
)


def _clean_strs(vals: object) -> list[str]:
    """Stripped non-empty strings from a raw context list."""
    return [v.strip() for v in vals if isinstance(v, str) and v.strip()] if isinstance(vals, list) else []


def _rel_line(rel: object) -> str | None:
    """One rendered relationship line, or None when any triple part is missing."""
    if not isinstance(rel, dict):
        return None
    parts = [rel.get(k) for k in ("subject", "relation", "object")]
    if not all(isinstance(p, str) and p.strip() for p in parts):
        return None
    assert all(isinstance(p, str) for p in parts)
    return f"{parts[0]} {parts[1]} {parts[2]}"


def _relationship_lines(rels: object) -> list[str]:
    """Rendered subject/relation/object lines from raw relationship records."""
    return [line for r in rels if (line := _rel_line(r)) is not None] if isinstance(rels, list) else []


def _context_lines(context: Mapping[str, object]) -> list[str]:
    out = [f"{key}: {', '.join(items)}" for key in _LIST_CONTEXT_KEYS if (items := _clean_strs(context.get(key)))]
    if rendered := _relationship_lines(context.get("relationships")):
        out.append(f"relationships: {'; '.join(rendered)}")
    return out


SOURCE_WORKFLOW = (
    "Workflow: build a material branch map (entities, relationship types, forms, exhibits, open "
    "questions) -> search SEC (search results are navigation artifacts, never evidence) -> open the "
    "underlying filings -> inspect documents, exhibits, and passages -> record only raw-document-backed "
    "evidence -> continue with materially new searches -> submit structured coverage (entities, "
    "relationship types, forms, exhibits, open questions) once the material branches are covered. "
    "There is no maximum number of searches, filing reads, document reads, exhibit reads, or waves; "
    "continue while the searches stay materially new. Only exact no-progress repeats (same query, "
    "same filing, same passage) are wasteful."
)
"""Research workflow: navigation is not evidence; only raw documents are, and volume is unbounded."""


def build_scout_prompt(assignment: ScoutAssignment) -> str:
    """Deterministic prompt: role template + context + baseline + scope + research workflow."""
    template = ROLE_PROMPTS[assignment.role]
    tickers = ", ".join(assignment.tickers) if assignment.tickers else "scope tickers TBD"
    prompt = (
        f"{template}\nQuestion: {assignment.question}\n"
        f"Tickers: {tickers}\nAs of: {assignment.as_of} (PIT cutoff; "
        "ignore anything knowable only after this date.)\n"
    )
    if assignment.context:
        for line in _context_lines(assignment.context):
            prompt += f"Context {line}\n"
    if assignment.baseline:
        prompt += "Latest-filing baseline (as_of-filtered, target searches with its terms):\n"
        prompt += "".join(f"- {line}\n" for line in assignment.baseline)
    if assignment.unscoped_queries:
        prompt += (
            "Counterparty branch first - these queries have already run with NO ticker and NO cik "
            "(a scoped search shows only the scope issuer's own disclosures) and the documents their "
            "hits named are opened below; the hits are other filers disclosing the counterparty, and "
            "they are the branch a scope-only search cannot reach. Re-run one only to open a hit the "
            "acquired list does not cover.\n"
        )
        prompt += "".join(f"- {q}\n" for q in assignment.unscoped_queries)
        prompt += (
            "Cite what those filings state; never conclude a relationship is absent from a scoped "
            "search, and do not end the assignment with every counterparty query unopened.\n"
        )
    if assignment.queries:
        prompt += "Assigned queries (search each; skip exact repeats already executed):\n"
        prompt += "".join(f"- {q}\n" for q in assignment.queries)
        prompt += (
            "If a candidate issuer is not in scope tickers, search it as a concept "
            "(customer/supplier/fund/risk-factor/8-K/proxy/N-PX/agreement mentions), never dead-end.\n"
        )
    return (
        prompt
        + SOURCE_WORKFLOW
        + "\n"
        + 'Respond with JSON only: [{"text": "<finding>", "claim_type": '
        + '"observed_fact|inference", "evidence_ids": ["<id>", ...]}, ...]. '
        + "Cite only the acquired ids listed below, one or more per claim: observed_fact when the "
        + "passage states it directly, inference when reasoned from cited passages. "
        + "Findings without a raw-document citation are not findings."
    )


def _is_pit_eligible(known_at: object, as_of: str | None) -> bool:
    """Shared PIT rule: historical as_of + unknown known_at is ineligible.

    A non-bounded as_of (None, blank, or the `unbounded` sentinel) is no cutoff at all,
    so every candidate is eligible and `pit_violated` is never consulted: only this
    caller hands the sentinel down (the ledger gate receives None instead), and a
    non-ISO as_of would otherwise raise inside the gate and read as a violation.
    """
    from app.research.models import _as_of_bounded, pit_unverified, pit_violated

    if not _as_of_bounded(as_of):
        return True
    if known_at is None:
        return not pit_unverified(as_of, None)
    if isinstance(known_at, str):
        if not known_at.strip():
            return not pit_unverified(as_of, None)
        try:
            if pit_unverified(as_of, known_at):
                return False
            return not pit_violated(as_of, known_at)
        except ValueError:
            return False
    return False


@dataclass
class _ScoutStore:
    """Accumulated scout evidence: eligible ids, rejected ids, prompt lines."""

    assignment: ScoutAssignment
    journal: Callable[[str, dict[str, object]], None] | None = None
    evidence_ids: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    seen_queries: set[str] = field(default_factory=set)
    acquired: list[str] = field(default_factory=list)
    documents: list[tuple[str, str]] = field(default_factory=list)
    opened: set[tuple[str, str]] = field(default_factory=set)

    def candidate_id(self, candidate: object) -> str | None:
        """Evidence id when the candidate is a dict with a non-blank id."""
        if not isinstance(candidate, dict):
            return None
        eid_raw = candidate.get("evidence_id")
        return eid_raw if isinstance(eid_raw, str) and eid_raw else None

    def accept(self, eid: str, candidate: dict[str, object]) -> None:
        """Record one PIT-eligible id plus its one-line prompt rendering."""
        if eid in self.evidence_ids:
            return
        self.evidence_ids.append(eid)
        known = candidate.get("known_at")
        header = f"{eid} (known_at={known})" if known else eid
        snippet = candidate.get("claim_text") or candidate.get("content_snippet")
        text = snippet.strip().replace("\n", " ")[:300] if isinstance(snippet, str) and snippet.strip() else ""
        self.acquired.append(f"{header} :: {text}" if text else header)

    def reject(self, eid: str) -> None:
        """Record one PIT-ineligible id, journalled once."""
        if eid in self.rejected:
            return
        self.rejected.append(eid)
        if self.journal is not None:
            self.journal(
                "evidence.rejected",
                {"session_id": self.assignment.session_id, "evidence_id": eid},
            )

    def collect(self, resp: object) -> None:
        """Fold one dispatch response into eligible/rejected ids and document candidates."""
        raw_ids = resp.get("evidence_ids") if isinstance(resp, dict) else None
        if isinstance(resp, dict):
            self.harvest_documents(resp)
        candidates: Sequence[object] = raw_ids if isinstance(raw_ids, list) else []
        for candidate in candidates:
            eid = self.candidate_id(candidate)
            if eid is None:
                continue
            assert isinstance(candidate, dict)
            if _is_pit_eligible(candidate.get("known_at"), self.assignment.as_of):
                self.accept(eid, candidate)
            else:
                self.reject(eid)

    def harvest_documents(self, resp: Mapping[str, object]) -> None:
        """Queue the display-window hits of one result packet for document opening (deduped)."""
        raw_hits = resp.get("top_hits")
        if not isinstance(raw_hits, list):
            return
        for hit in raw_hits:
            if not isinstance(hit, dict):
                continue
            accession = hit.get("accession")
            if not isinstance(accession, str) or not accession.strip():
                continue
            document = hit.get("document")
            pair = (
                accession.strip(),
                document.strip() if isinstance(document, str) else "",
            )
            if pair not in self.documents:
                self.documents.append(pair)


def _search_args(query: str, as_of: str) -> dict[str, object]:
    """search_sec_filings args: concept query + as_of PIT on every call."""
    args: dict[str, object] = {"query": query}
    text = as_of.strip() if isinstance(as_of, str) else ""
    if text and text.lower() != "unbounded":
        args["as_of"] = as_of
    return args


def _run_baseline(store: _ScoutStore, guarded_call: Callable[[str, dict[str, object]], dict[str, object]]) -> None:
    """Seed the as_of-filtered latest-filing baseline from assignment context."""
    for line in store.assignment.baseline:
        if normalize_query(line) in store.seen_queries:
            continue
        store.seen_queries.add(normalize_query(line))
        store.collect(
            guarded_call(
                "call_tool", {"name": "search_sec_filings", "arguments": _search_args(line, store.assignment.as_of)}
            )
        )


def _run_queries(
    store: _ScoutStore,
    guarded_call: Callable[[str, dict[str, object]], dict[str, object]],
    queries: Sequence[str],
) -> None:
    """Run one query list, skipping normalized repeats (a seen query never re-executes)."""
    for query in queries:
        if not query.strip():
            continue
        key = normalize_query(query)
        if key in store.seen_queries:
            continue
        store.seen_queries.add(key)
        store.collect(
            guarded_call(
                "call_tool",
                {
                    "name": "search_sec_filings",
                    "arguments": _search_args(query, store.assignment.as_of),
                },
            )
        )


def _fan_out(
    store: _ScoutStore,
    guarded_call: Callable[[str, dict[str, object]], dict[str, object]],
) -> None:
    """Baseline + assigned queries + unscoped (no ticker/cik) queries + role tools; no count caps."""
    _run_baseline(store, guarded_call)
    _run_queries(store, guarded_call, store.assignment.queries)
    _run_queries(store, guarded_call, store.assignment.unscoped_queries)
    tickers = store.assignment.tickers or [""]
    for tool_name, extra in _ROLE_TOOLS.get(store.assignment.role, ()):
        for ticker in tickers:
            if not ticker.strip():
                continue
            args = _role_arguments(tool_name, ticker.strip(), store.assignment.as_of)
            args.update(extra)
            store.collect(guarded_call("call_tool", {"name": tool_name, "arguments": args}))
    if not store.evidence_ids:
        store.collect(guarded_call("call_tool", {"name": "get_sec_search_coverage", "arguments": {}}))


def _document_args(accession: str, document: str, as_of: str) -> dict[str, object]:
    """get_sec_document args: the exact filing/document a hit named, as_of PIT when bounded."""
    args: dict[str, object] = {"accession_no": accession}
    if document:
        args["document_name"] = document
    text = as_of.strip() if isinstance(as_of, str) else ""
    if text and text.lower() != "unbounded":
        args["as_of"] = as_of
    return args


def _open_documents(
    store: _ScoutStore,
    guarded_call: Callable[[str, dict[str, object]], dict[str, object]],
) -> None:
    """Open every newly surfaced document: only raw documents become evidence.

    Navigation results (search/list/diff/xbrl) carry no citable ids; the display
    window is what was retrieved, so each distinct (accession, document) is read
    once (later expansion rounds open only what their searches surfaced). Deeper
    paging of the persisted hit set stays with ``research_read_search``
    on the model-driven path, never a count cap here.
    """
    for pair in list(store.documents):
        if pair in store.opened:
            continue
        store.opened.add(pair)
        accession, document = pair
        store.collect(
            guarded_call(
                "call_tool",
                {
                    "name": "get_sec_document",
                    "arguments": _document_args(accession, document, store.assignment.as_of),
                },
            )
        )


def _snippet_text(line: str) -> str:
    """Snippet payload after the `id :: text` separator, else empty."""
    return line.split("::", 1)[1] if "::" in line else ""


def _passage_texts(store: _ScoutStore) -> list[str]:
    """Text of every acquired passage (the expansion input)."""
    return [_snippet_text(line) for line in store.acquired]


def _expand(
    store: _ScoutStore,
    guarded_call: Callable[[str, dict[str, object]], dict[str, object]],
) -> None:
    """Information-driven loop: novel queries the acquired passages name, to fixpoint.

    Each round searches the novel queries, opens the documents those hits surface,
    and derives again from the enlarged passage set. Derivation reads the acquired
    passages only: the assignment's context terms already run as assigned queries
    (every planned family query, not a sample), so re-issuing them here would just
    duplicate another role's searches. Repeats are suppressed by the shared
    ``seen_queries`` set, so every round spends at least one never-seen query or
    stops: the stop is novelty, never a count.
    """
    # Local import: this module is imported by source_agent, so a top-level import would cycle.
    from .source_agent import expand_queries

    while True:
        novel = expand_queries(
            list(store.seen_queries), _passage_texts(store), reference=store.assignment.question
        )
        if not novel:
            return
        _run_queries(store, guarded_call, novel)
        _open_documents(store, guarded_call)


def _finish(assignment: ScoutAssignment, store: _ScoutStore, tools_used: int, model: ModelFn) -> ScoutResult:
    """Draft grounded findings on the exact acquired evidence."""
    prompt = build_scout_prompt(assignment)
    if store.acquired:
        prompt += "\nAcquired evidence (cite only these ids):\n" + "\n".join(f"- {line}" for line in store.acquired)
    text = model(prompt)
    dropped: list[str] = []

    def _report_rejected(item: str, reason: str) -> None:
        dropped.append(f"{reason} :: {item}")

    findings: list[GroundedClaim] = parse_grounded_claims_tolerant(
        text, frozen=store.evidence_ids, on_reject=_report_rejected
    )
    unknowns: list[str] = []
    limitations: list[str] = list(store.rejected)
    if dropped:
        # Fail closed per claim (never accepted), reported so the loss is visible.
        if store.journal is not None:
            for line in dropped:
                store.journal("claim.rejected", {"session_id": assignment.session_id, "detail": line[:300]})
        limitations.append(f"{len(dropped)} claim(s) dropped: cited ids outside the acquired evidence")
    if not store.evidence_ids:
        unknowns.insert(0, "no PIT-eligible SEC evidence returned")
    return ScoutResult(
        assignment_id=assignment.assignment_id,
        session_id=assignment.session_id,
        coverage=f"role={assignment.role} tickers={len(assignment.tickers)} tool_calls={tools_used}",
        findings=findings,
        unknowns=unknowns,
        limitations=limitations,
        follow_up_requests=[],
    )


def run_scout(
    assignment: ScoutAssignment,
    *,
    dispatch: DispatchFn,
    model: ModelFn,
    journal: Callable[[str, dict[str, object]], None] | None = None,
) -> ScoutResult:
    """Run one scout: baseline + queries + unscoped searches + role tools + documents + expansion.

    ``max_children = 0``: this function never spawns child jobs. Search and
    navigation results are never evidence; the documents their hits name are
    opened (``get_sec_document``) so findings can cite raw filing passages, and
    the opened passages then drive expansion (novel queries -> searches ->
    newly surfaced documents) to a fixpoint. All of that executes before the
    model call, so every materially distinct query runs whether or not the
    model names it. PIT: evidence with ``known_at > as_of`` is rejected and
    journalled as ``evidence.rejected``. Expansion stops on marginal
    information (repeats yield nothing new) or an explicit tool limit -- never
    on evidence counts. Unlimited by default.
    """
    tools_used = 0

    def guarded_call(name: str, args: dict[str, object]) -> dict[str, object]:
        nonlocal tools_used
        if assignment.max_tool_calls is not None and tools_used >= assignment.max_tool_calls:
            return {
                "error": "policy_rejection: explicit scout tool limit reached",
                "soft": True,
            }
        tools_used += 1
        return dispatch(name, args)

    catalog = guarded_call("browse_tools", {})
    _ = catalog  # discovery hint only; the query plan below decides calls.
    store = _ScoutStore(assignment=assignment, journal=journal)
    _fan_out(store, guarded_call)
    _open_documents(store, guarded_call)
    _expand(store, guarded_call)
    return _finish(assignment, store, tools_used, model)


__all__ = [
    "ALLOWED_DOMAIN",
    "MAX_CHILDREN",
    "ROLE_PROMPTS",
    "SOURCE_WORKFLOW",
    "DispatchFn",
    "ModelFn",
    "ScoutAssignment",
    "ScoutResult",
    "ScoutRole",
    "build_scout_prompt",
    "normalize_query",
    "run_scout",
]
