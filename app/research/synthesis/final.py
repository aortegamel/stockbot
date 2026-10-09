"""Final synthesis: rich FinalResearchResult + legacy answer, no recommendation.

The synthesizer never issues a forced buy/sell/hold call. ``to_dict`` carries
the deep sections (bottom line, base case, major evidence, first/second-order
impact, bull/bear, disagreements, positioning, catalysts, uncertainties, what
would change the view, limitations, coverage, sources) with every claim's
declared ``claim_type`` preserved, so an inference is never rendered as direct
fact. Every claim resolves to freeze-contained evidence ids; unknown claims
may cite none.
Epistemic typing is deterministic: the direct-evidence section is built only
from the caller's canonical EvidenceLedger observations (raw documents), while
committee claims are interpretations over those observations wherever they are
rendered. An absence observation comes only from the session's coverage
artifacts — a generic unknown claim stays an unknown and is never rewritten as
"no disclosure was located". Absence stays scoped to the searched sources,
never a real-world nonexistence claim, and depth follows the researched
material (no fixed word count, no caps). Per-source coverage (SEC
filings/forms/gaps, FINRA datasets/periods/gaps, WEB queries/domains/gaps)
rides in ``coverage``; ``sources`` lists the evidence behind the claims.

Fake-model sketch (no live calls): canned trio of analyses -> canned
``CommitteeDisagreement`` -> ``synthesize_final``; assert claims stay
within the freeze and ``answer`` is non-empty with no invented evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.research.agents import (
    CLAIM_TYPES,
    GroundedClaim,
    claims_refs,
    conservative_claim_type,
)
from app.research.agents.bearbot import BearAnalysis
from app.research.agents.bullbot import BullAnalysis
from app.research.agents.stockbot import StockbotAnalysis
from app.research.evidence import evidence_domain, evidence_integrity

from .committee import CommitteeDisagreement, _coerce_wave_id

SCOPE_ABSENCE = "No disclosure located within the searched scope"
"""Scoped-absence phrasing: a searched-scope observation, never nonexistence."""
SEC_SCOPE_ABSENCE = "No disclosure located within the searched SEC scope"
"""Legacy SEC-scoped alias (kept for pinned readers); prefer SCOPE_ABSENCE."""

SCOPE_LIMITATION = "Searched-source scope: only the allowed sources were searched; unsearched sources carry no finding."
SEC_ONLY_LIMITATION = (
    "SEC-only scope: no non-SEC source (news, transcripts, private documents, market data) was searched."
)
"""Legacy SEC-only limitation (kept for pinned readers); prefer SCOPE_LIMITATION."""

_CLAIM_SEVERITY = {
    "observed_fact": "direct",
    "inference": "indirect",
    "unknown": "uncertain",
    "contradicted": "contested",
}


@dataclass
class FinalSynthesis:
    session_id: str
    wave_id: int
    freeze_id: str
    as_of: str
    question: str
    answer: str
    base_case: str
    bull_case: str
    bear_case: str
    disagreement: CommitteeDisagreement
    unknowns: list[str] = field(default_factory=list)
    what_would_change: list[str] = field(default_factory=list)
    claims: list[GroundedClaim] = field(default_factory=list)
    executive_summary: str = ""
    consensus: str = ""
    impact_channels: list[dict[str, object]] = field(default_factory=list)
    first_order_effects: list[dict[str, object]] = field(default_factory=list)
    second_order_effects: list[dict[str, object]] = field(default_factory=list)
    bull_evidence_ids: list[str] = field(default_factory=list)
    bear_evidence_ids: list[str] = field(default_factory=list)
    critical_disagreements: list[str] = field(default_factory=list)
    evidence_limitations: list[str] = field(default_factory=list)
    research_scope: dict[str, object] = field(default_factory=dict)
    direct_evidence: list[dict[str, object]] = field(default_factory=list)
    absence_observations: list[str] = field(default_factory=list)
    filing_references: list[str] = field(default_factory=list)
    positioning: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    coverage: dict[str, object] = field(default_factory=dict)
    sources: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.wave_id = _coerce_wave_id(self.wave_id)

    def to_dict(self) -> dict[str, object]:
        """Deep FinalResearchResult dict (legacy answer/claims keys kept)."""
        claim_rows = [
            {"text": c.text, "claim_type": c.claim_type, "evidence_ids": list(c.evidence_ids)} for c in self.claims
        ]
        return {
            "answer": self.answer,
            "executive_summary": self.executive_summary or self.answer,
            "consensus": self.consensus,
            "base_case": self.base_case,
            "major_evidence": [dict(row) for row in self.direct_evidence],
            "direct_evidence": [dict(row) for row in self.direct_evidence],
            "bull_case": {"summary": self.bull_case, "evidence_ids": list(self.bull_evidence_ids)},
            "bear_case": {"summary": self.bear_case, "evidence_ids": list(self.bear_evidence_ids)},
            "impact_channels": [dict(ch) for ch in self.impact_channels],
            "first_order_effects": [dict(e) for e in self.first_order_effects],
            "second_order_effects": [dict(e) for e in self.second_order_effects],
            "disagreements": list(self.critical_disagreements),
            "critical_disagreements": list(self.critical_disagreements),
            "positioning": list(self.positioning),
            "catalysts": list(self.catalysts),
            "uncertainties": _union_texts((list(self.unknowns), list(self.absence_observations))),
            "absence_observations": list(self.absence_observations),
            "what_would_change": list(self.what_would_change),
            "what_changes_the_view": list(self.what_would_change),
            "limitations": list(self.evidence_limitations),
            "evidence_limitations": list(self.evidence_limitations),
            "coverage": {k: list(v) if isinstance(v, list) else v for k, v in self.coverage.items()},
            "sources": [dict(s) for s in self.sources],
            "grounded_claims": [dict(r) for r in claim_rows],
            "claims": [dict(r) for r in claim_rows],
            "filing_references": list(self.filing_references),
            "evidence_refs": list(self.filing_references),
            "research_scope": dict(self.research_scope),
            "freeze_id": self.freeze_id,
            "as_of": self.as_of,
        }


def scoped_absence(text: str, domain: str | None = None) -> str:
    """Scoped absence phrasing: what a searched scope located, never real-world nonexistence."""
    clean = text.strip()[:2000]
    scope = SCOPE_ABSENCE
    if domain is None or domain.strip().upper() == "SEC":
        scope = SEC_SCOPE_ABSENCE
    elif domain.strip().upper() in ("FINRA", "WEB"):
        scope = f"No disclosure located within the searched {domain.strip().upper()} scope"
    if not clean:
        return f"{scope}."
    if clean.lower().startswith("no disclosure"):
        return clean
    return f"{scope}: {clean}"


def _union_texts(groups: tuple[Sequence[str], ...]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for item in group:
            if item and item not in out:
                out.append(item)
    return out


def _strs(values: object) -> list[str]:
    """Non-empty strings from a list payload ([] for anything else)."""
    if not isinstance(values, (list, tuple)):
        return []
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


def _consensus_text(disagreement: CommitteeDisagreement) -> str:
    """Committee consensus: dedicated field when present, else joined agreement."""
    for attr in ("consensus", "agreement"):
        joined = "; ".join(_strs(getattr(disagreement, attr, None)))
        if joined:
            return joined
    return "none stated"


def _critical_disagreements(disagreement: CommitteeDisagreement) -> list[str]:
    """Dedicated critical-disagreement field when present, else the legacy split."""
    raw = getattr(disagreement, "critical_disagreements", None)
    if isinstance(raw, (list, tuple)):
        out = [v.strip() for v in raw if isinstance(v, str) and v.strip()]
        if out:
            return out
    return _strs(getattr(disagreement, "disagreement", []))


def _analysis_uncertainties(stock: StockbotAnalysis, bull: BullAnalysis, bear: BearAnalysis) -> list[str]:
    """Uncertainties across trio analyses (tolerates the newer uncertainties attr)."""
    out: list[str] = []
    for analysis in (stock, bull, bear):
        for attr in ("uncertainties", "unknowns"):
            for item in _strs(getattr(analysis, attr, [])):
                if item not in out:
                    out.append(item)
    return out


def _channel_field(entry: object, key: str, *aliases: str) -> object:
    """First present alias value (Mapping get, else getattr; skips None)."""
    for candidate in (key, *aliases):
        value: object = entry.get(candidate, None) if isinstance(entry, Mapping) else getattr(entry, candidate, None)
        if value is not None:
            return value
    return None


def _as_text(value: object) -> str:
    """Coerce one optional prose field (non-strings stay blank)."""
    return value.strip() if isinstance(value, str) else ""


def _cited_channel_ids(raw_ids: object) -> list[str]:
    """Non-blank string ids from a channel payload ([] for anything else)."""
    return [e for e in raw_ids if isinstance(e, str) and e.strip()] if isinstance(raw_ids, (list, tuple)) else []


def _normalize_channel(entry: object) -> dict[str, object] | None:
    """One impact channel in render shape (new {text, direction} or legacy {name, assessment}); None when ungrounded."""
    text = _as_text(_channel_field(entry, "text", "name", "title", "explanation"))
    detail = _as_text(_channel_field(entry, "explanation", "assessment")) or text
    direction = _as_text(_channel_field(entry, "direction", "severity", "materiality", "impact"))
    ids = _cited_channel_ids(_channel_field(entry, "evidence_ids", "refs"))
    if not text or not ids:
        return None
    return {"name": text[:80], "severity": direction or "unspecified", "explanation": detail, "evidence_ids": ids}


def _channels_from_analyses(stock: object, bull: object, bear: object) -> list[dict[str, object]]:
    """Committee-provided impact channels when present (deduped, grounded only; duck-typed)."""
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for analysis in (stock, bull, bear):
        raw = getattr(analysis, "impact_channels", None)
        if not isinstance(raw, (list, tuple)):
            continue
        for entry in raw:
            chan = _normalize_channel(entry)
            if chan is None:
                continue
            key = str(chan["name"]) + "|" + str(chan["explanation"])
            if key in seen:
                continue
            seen.add(key)
            out.append(chan)
    return out


def _channel_from_claim(claim: GroundedClaim) -> dict[str, object] | None:
    """Fallback channel derived from one cited claim (None when uncited)."""
    text = claim.text.strip()
    ids = [e for e in claim.evidence_ids if isinstance(e, str) and e.strip()]
    if not text or not ids:
        return None
    return {
        "name": text[:80],
        "severity": _CLAIM_SEVERITY.get(claim.claim_type, "indirect"),
        "explanation": text,
        "evidence_ids": ids,
    }


def _channels_from_claims(claims: list[GroundedClaim]) -> list[dict[str, object]]:
    """Fallback channels derived from claims (deduped, grounded only)."""
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for claim in claims:
        chan = _channel_from_claim(claim)
        if chan is None:
            continue
        key = str(chan["name"]) + "|" + str(chan["explanation"])
        if key in seen:
            continue
        seen.add(key)
        out.append(chan)
    return out


def _effect_row(claim: GroundedClaim) -> dict[str, object]:
    """One effect row: claim text with its declared type and freeze refs."""
    return {"text": claim.text, "claim_type": claim.claim_type, "evidence_ids": list(claim.evidence_ids)}


def _observation_field(row: Mapping[str, object], key: str) -> str:
    """One string field of a canonical ledger row ('' when absent/mistyped)."""
    value = row.get(key)
    return value.strip() if isinstance(value, str) else ""


def _observation_provenance(row: Mapping[str, object]) -> Mapping[str, object]:
    """The row's provenance mapping ({} when absent/mistyped)."""
    prov = row.get("provenance")
    return prov if isinstance(prov, Mapping) else {}


def _observation_excerpt(row: Mapping[str, object], limit: int = 400) -> str:
    """Canonical passage of one ledger row: its raw-document provenance text, else its content."""
    prov = _observation_provenance(row)
    for source in (prov.get("passage"), row.get("content")):
        text = source.strip() if isinstance(source, str) else ""
        if text:
            return text[:limit]
    return ""


def _observation_row(row: Mapping[str, object]) -> dict[str, object] | None:
    """One direct-evidence row built from a canonical EvidenceLedger record.

    The direct-evidence section states what the source documents show, so its
    text is the document identity plus the kernel-held passage — never a model
    claim. The record's own claim text rides along as ``claim`` for the
    structured payload; committee statements are interpretations over these rows.
    """
    evidence_id = _observation_field(row, "evidence_id")
    document, accession = _observation_identity(row, _observation_provenance(row))
    text = _observation_text(document, accession, _observation_excerpt(row))
    if not evidence_id or not text:
        return None
    return {
        "text": text,
        "claim": _observation_field(row, "claim_text"),
        "claim_type": "observed_fact",
        "document": document,
        "accession_no": accession,
        "evidence_ids": [evidence_id],
    }


def _observation_identity(row: Mapping[str, object], prov: Mapping[str, object]) -> tuple[str, str]:
    """(document, accession) of one ledger row: canonical provenance first, the row's own fields after."""
    return (
        _observation_field(prov, "document_name") or _observation_field(row, "source_name"),
        _observation_field(prov, "accession_no") or _observation_field(row, "source_record_id"),
    )


def _observation_text(document: str, accession: str, excerpt: str) -> str:
    """Direct-evidence text: the document identity plus the kernel-held passage (either alone if partial)."""
    identity = " ".join(part for part in (document, accession) if part)
    if identity and excerpt:
        return f'{identity}: "{excerpt}"'
    return identity or excerpt


def _observation_rows(
    observations: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Canonical observations in render order (rows without an evidence id are dropped)."""
    return [row for row in (_observation_row(item) for item in (observations or [])) if row is not None]


def _split_effects(claims: Sequence[GroundedClaim]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Directly observed claims vs reasoned/contested claims; unknown reads stay in the unknowns section."""
    first = [_effect_row(c) for c in claims if c.claim_type == "observed_fact"]
    second = [_effect_row(c) for c in claims if c.claim_type in ("inference", "contradicted")]
    return first, second


def _coerce_extra_claim(item: object) -> GroundedClaim | None:
    """One caller claim in grounded shape; None when blank, ungrounded, or mistyped."""
    if isinstance(item, GroundedClaim):
        raw_text, raw_type, raw_ids = item.text, item.claim_type, item.evidence_ids
    elif isinstance(item, Mapping):
        raw_text = _channel_field(item, "text", "claim_text", "claim")
        raw_type = _channel_field(item, "claim_type")
        raw_ids = _channel_field(item, "evidence_ids", "refs")
    else:
        return None
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    declared = raw_type.strip().lower() if isinstance(raw_type, str) else ""
    claim_type = declared if declared in CLAIM_TYPES else "inference"
    ids = _cited_channel_ids(raw_ids)
    if not text or not (ids or claim_type == "unknown"):
        return None
    return GroundedClaim(text=text[:500], claim_type=claim_type, evidence_ids=ids)


def _normalize_extra(extra: Sequence[GroundedClaim | dict[str, object]] | None) -> list[GroundedClaim]:
    """Caller claims into grounded records (drops ungrounded entries)."""
    return [claim for claim in (_coerce_extra_claim(item) for item in extra or []) if claim is not None]


def _merge_claims(*groups: Sequence[GroundedClaim]) -> list[GroundedClaim]:
    """Dedupe the trio + caller claims by text: union refs, keep the least assertive declared type."""
    merged: list[GroundedClaim] = []
    index: dict[str, int] = {}
    for claim in (claim for group in groups for claim in group):
        text = claim.text.strip()
        if not text:
            continue
        prior_index = index.get(text)
        if prior_index is None:
            index[text] = len(merged)
            merged.append(
                GroundedClaim(
                    text=text, claim_type=claim.claim_type, evidence_ids=list(dict.fromkeys(claim.evidence_ids))
                )
            )
            continue
        prior = merged[prior_index]
        merged[prior_index] = GroundedClaim(
            text=text,
            claim_type=conservative_claim_type([prior.claim_type, claim.claim_type]),
            evidence_ids=list(dict.fromkeys([*prior.evidence_ids, *claim.evidence_ids])),
        )
    return merged


def _normalize_scope(scope: Mapping[str, object] | None) -> dict[str, object]:
    """Research scope in spec shape; the searched-source boundary carries its explicit limitation."""
    candidates = (scope.get("allowed_sources"), scope.get("allowed")) if isinstance(scope, Mapping) else ()
    sources = next((_strs(raw) for raw in candidates if isinstance(raw, (list, tuple))), None) or ["SEC"]
    upper = [s.upper() for s in sources]
    sec_only = upper == ["SEC"]
    limitation = SEC_ONLY_LIMITATION if sec_only else SCOPE_LIMITATION
    return {"allowed_sources": sources, "sec_only": sec_only, "limitation": limitation}


def _synth_unknowns(
    stock: StockbotAnalysis, bull: BullAnalysis, bear: BearAnalysis, disagreement: CommitteeDisagreement
) -> list[str]:
    """Deduped uncertainties across the trio plus the committee critical list."""
    return list(
        dict.fromkeys(
            (*_analysis_uncertainties(stock, bull, bear), *_strs(getattr(disagreement, "critical_uncertainties", [])))
        )
    )


def _unknown_claims(claims: Sequence[GroundedClaim]) -> list[str]:
    """Texts of committee claims declared unknown: an open question stays an unknown.

    It is never rewritten as an absence observation (a search-coverage statement)
    and never rendered as a direct finding.
    """
    return list(dict.fromkeys(c.text.strip() for c in claims if c.claim_type == "unknown" and c.text.strip()))


def _synth_changes(stock: StockbotAnalysis, bull: BullAnalysis, bear: BearAnalysis) -> list[str]:
    """Union of the trio what-would-change lists in first-seen order."""
    return _union_texts(
        (
            list(getattr(stock, "what_would_change", []) or []),
            list(getattr(bull, "what_would_change", []) or []),
            list(getattr(bear, "what_would_change", []) or []),
        )
    )


def _synth_limitations(evidence_limitations: Sequence[str] | None, scope: Mapping[str, object]) -> list[str]:
    """Caller limitations plus the explicit searched-source boundary."""
    lims = [v.strip() for v in (evidence_limitations or []) if isinstance(v, str) and v.strip()]
    limitation = _as_text(scope.get("limitation"))
    if limitation:
        lims = _union_texts((lims, [limitation]))
    return lims


def _effect_line(row: Mapping[str, object]) -> str:
    """One rendered evidence/effect line preserving its declared claim_type."""
    text = _as_text(row.get("text"))
    claim_type = _as_text(row.get("claim_type")) or "inference"
    ids = _cited_channel_ids(row.get("evidence_ids"))
    refs = f" [{', '.join(ids)}]" if ids else ""
    return f"{text} ({claim_type}){refs}"


def _section(title: str, lines: Sequence[str]) -> list[str]:
    """One rendered section ([] when it has no lines)."""
    return [f"{title}:", *(f"- {line}" for line in lines if line)] if any(lines) else []


_COVERAGE_VERDICT_KEYS: frozenset[str] = frozenset(
    {"source_domain", "source_sufficiency", "useful_for_question", "complete"}
)
"""Dossier verdict keys: routing state, never searched scope; dropped from the render."""


def _coverage_scope_items(section: Mapping[str, object]) -> list[tuple[str, list[str]]]:
    """(key, scope strings) for every non-empty string-list field except verdict keys (sorted, stable)."""
    out: list[tuple[str, list[str]]] = []
    for key in sorted(section):
        if key in _COVERAGE_VERDICT_KEYS or key in ("detail", "summary"):
            continue
        items = _strs(section.get(key))
        if items:
            out.append((key, items))
    return out


def _coverage_lines(coverage: Mapping[str, object]) -> list[str]:
    """Per-source coverage lines: every surviving scope key renders under its own name."""
    lines: list[str] = []
    for key in ("sec", "finra", "web"):
        section = coverage.get(key)
        if not isinstance(section, Mapping):
            continue
        parts = [f"{name}: {', '.join(items)}" for name, items in _coverage_scope_items(section)]
        detail = _as_text(section.get("detail") or section.get("summary"))
        head = f"{key.upper()}"
        lines.append(
            f"{head} — {'; '.join(parts)}{(' — ' + detail) if detail and not parts else (detail if detail else '')}"
            if (parts or detail)
            else head
        )
    extra = _strs(coverage.get("gaps"))
    if extra:
        lines.append(f"gaps: {', '.join(extra)}")
    return lines


def _source_line(row: Mapping[str, object]) -> str:
    """One source line: evidence id + domain + integrity class + document identity."""
    evidence_id = _as_text(row.get("evidence_id"))
    domain = _as_text(row.get("domain")).upper() or "SOURCE"
    integrity = _as_text(row.get("integrity_class") or row.get("integrity")).upper()
    document = _as_text(row.get("document") or row.get("source_name"))
    label = f"{domain} [{integrity}]" if integrity else domain
    head = " ".join(part for part in (label, document) if part) or evidence_id or "source"
    return f"{head} [{evidence_id}]" if evidence_id and head != evidence_id else head


def _normalize_coverage(coverage: Mapping[str, object] | None) -> dict[str, object]:
    """Per-source coverage in render shape (SEC/FINRA/WEB sections; string lists only).

    Keeps every non-empty string-list field except the dossier verdict keys,
    so desk keys survive under their own names (datasets_queried,
    tickers_covered, queries_executed, ...) instead of dropping.
    """
    out: dict[str, object] = {}
    if not isinstance(coverage, Mapping):
        return out
    for key in ("sec", "finra", "web"):
        section = coverage.get(key)
        if not isinstance(section, Mapping):
            continue
        kept: dict[str, object] = dict(_coverage_scope_items(section))
        detail = _as_text(section.get("detail") or section.get("summary"))
        if detail:
            kept["detail"] = detail
        if kept:
            out[key] = kept
    gaps = _strs(coverage.get("gaps"))
    if gaps:
        out["gaps"] = gaps
    return out


def _observation_integrity(row: Mapping[str, object]) -> str:
    """Integrity class of one ledger row: explicit kernel field wins, else the kernel mapping."""
    direct = _as_text(row.get("integrity_class") or row.get("integrity")).upper()
    if direct in ("PRIMARY_DOCUMENT", "CANONICAL_STRUCTURED", "EXTERNAL_SOURCE"):
        return direct
    return evidence_integrity(_observation_provenance(row))


def _normalize_sources(
    sources: Sequence[Mapping[str, object]] | None,
    claims: Sequence[GroundedClaim],
    observations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Sources behind the claims: caller rows win; else one row per cited evidence id.

    Domain/integrity come from the observations via the kernel mapping, never
    the caller: explicit caller integrity wins only when it names the closed
    vocabulary, so §6/§10 labels survive to the report.
    """
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    by_id: dict[str, Mapping[str, object]] = {}
    for row in observations or []:
        if not isinstance(row, Mapping):
            continue
        evidence_id = _as_text(row.get("evidence_id"))
        if evidence_id and evidence_id not in by_id:
            by_id[evidence_id] = row
    for item in sources or []:
        if not isinstance(item, Mapping):
            continue
        evidence_id = _as_text(item.get("evidence_id"))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        obs = by_id.get(evidence_id)
        domain = _as_text(item.get("domain")).upper()
        if not domain or domain == "SOURCE":
            domain = _observation_domain(obs) if obs is not None else "SOURCE"
        integrity = _as_text(item.get("integrity_class") or item.get("integrity")).upper()
        if integrity not in ("PRIMARY_DOCUMENT", "CANONICAL_STRUCTURED", "EXTERNAL_SOURCE"):
            integrity = _observation_integrity(obs) if obs is not None else "EXTERNAL_SOURCE"
        rows.append(
            {
                "evidence_id": evidence_id,
                "domain": domain,
                "document": _as_text(item.get("document") or item.get("source_name")),
                "integrity_class": integrity,
            }
        )
    if rows:
        return rows
    domains: dict[str, str] = {}
    integrity_by_id: dict[str, str] = {}
    for row in observations or []:
        if not isinstance(row, Mapping):
            continue
        evidence_id = _as_text(row.get("evidence_id"))
        if evidence_id and evidence_id not in domains:
            domains[evidence_id] = _observation_domain(row)
            integrity_by_id[evidence_id] = _observation_integrity(row)
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            clean = evidence_id.strip() if isinstance(evidence_id, str) else ""
            if not clean or clean in seen:
                continue
            seen.add(clean)
            obs = by_id.get(clean)
            document = _as_text(obs.get("document") or obs.get("source_name")) if obs is not None else ""
            rows.append(
                {
                    "evidence_id": clean,
                    "domain": domains.get(clean, "SOURCE"),
                    "document": document,
                    "integrity_class": integrity_by_id.get(clean, "EXTERNAL_SOURCE"),
                }
            )
    return rows


def _observation_domain(row: Mapping[str, object]) -> str:
    """Source domain of one ledger row: explicit caller field wins, else the kernel mapping.

    FINRA/WEB rows are admitted only by persisted-result replay (bare rows
    still fail closed with ERR_RAW_SOURCE_REQUIRED).
    """
    direct = _as_text(row.get("domain") or row.get("source_domain")).upper()
    if direct in ("SEC", "FINRA", "WEB"):
        return direct
    return evidence_domain(_observation_provenance(row))


def _deep_answer(synth: FinalSynthesis) -> str:
    """Deterministic deep answer: every required section, depth taken from the researched material."""
    out: list[str] = [f"Bottom line: {synth.executive_summary}"]
    if synth.consensus and synth.consensus.lower() != "none stated":
        out.append(f"Consensus: {synth.consensus}")
    if synth.base_case:
        out += _section("Base case: Base case (stockbot)", [synth.base_case])
    out += _section(
        "Major evidence \u2014 What the evidence directly shows", [_effect_line(row) for row in synth.direct_evidence]
    )
    out += _section(
        "First-order effects \u2014 First-order impact", [_effect_line(row) for row in synth.first_order_effects]
    )
    out += _section(
        "Second-order effects \u2014 Second-order impact", [_effect_line(row) for row in synth.second_order_effects]
    )
    out += _section("Bull case \u2014 Bull case (bullbot)", [synth.bull_case])
    out += _section("Bear case \u2014 Bear case (bearbot)", [synth.bear_case])
    out += _section("Disagreements \u2014 Critical disagreements", synth.critical_disagreements)
    out += _section("Positioning", synth.positioning)
    out += _section("Catalysts", synth.catalysts)
    out += _section(
        "Uncertainties \u2014 Unknowns / unresolved", _union_texts((synth.unknowns, synth.absence_observations))
    )
    out += _section("What changes the view \u2014 What would change the view", synth.what_would_change)
    out += _section("Limitations \u2014 Source limitations", synth.evidence_limitations)
    out += _section("Coverage", _coverage_lines(synth.coverage))
    out += _section("Sources", [_source_line(row) for row in synth.sources])
    if synth.filing_references:
        out.append(f"Evidence references: {', '.join(synth.filing_references)}")
    return "\n".join(out)


def synthesize_final(
    question: str,
    *,
    session_id: str,
    wave_id: int | str,
    freeze_id: str,
    as_of: str,
    stock: StockbotAnalysis,
    bull: BullAnalysis,
    bear: BearAnalysis,
    disagreement: CommitteeDisagreement,
    model: object = None,
    extra_claims: Sequence[GroundedClaim | dict[str, object]] | None = None,
    evidence_limitations: Sequence[str] | None = None,
    research_scope: Mapping[str, object] | None = None,
    observations: Sequence[Mapping[str, object]] | None = None,
    absence_observations: Sequence[str] | None = None,
    coverage: Mapping[str, object] | None = None,
    positioning: Sequence[str] | None = None,
    catalysts: Sequence[str] | None = None,
    sources: Sequence[Mapping[str, object]] | None = None,
) -> FinalSynthesis:
    """Package the trio + disagreement + caller claims into the final record (no live calls).

    ``model`` overrides the deterministic deep answer when it is a string
    (e.g. the caller's drafted answer); otherwise the sections are joined
    deterministically so synthesis stays reproducible. Cited ids derive only
    from accepted per-claim mappings, never the whole freeze.

    ``observations`` are the freeze's canonical EvidenceLedger rows: the
    direct-evidence section is built from them, never from a model-labeled
    committee claim (committee statements are interpretations over these rows).
    ``absence_observations`` are the session's coverage artifacts' texts: a
    generic unknown stays an unknown and never becomes an absence claim.
    ``coverage`` is per-source searched scope (SEC filings/forms/gaps, FINRA
    datasets/periods/gaps, WEB queries/domains/gaps); ``positioning`` and
    ``catalysts`` are caller-supplied grounded lines; ``sources`` lists the
    evidence behind the claims (derived from cited ids when omitted).
    """
    claims = _merge_claims(
        list(getattr(stock, "claims", []) or []),
        list(getattr(bull, "claims", []) or []),
        list(getattr(bear, "claims", []) or []),
        _normalize_extra(extra_claims),
    )
    scope = _normalize_scope(research_scope)
    first_order, second_order = _split_effects(claims)
    observations_list = [row for row in (observations or []) if isinstance(row, Mapping)]
    coverage_norm = _normalize_coverage(coverage)
    synth = FinalSynthesis(
        session_id=session_id,
        wave_id=_coerce_wave_id(wave_id),
        freeze_id=freeze_id,
        as_of=as_of,
        question=question,
        answer="",
        base_case=stock.base_case,
        bull_case=bull.bull_case,
        bear_case=bear.bear_case,
        disagreement=disagreement,
        unknowns=list(
            dict.fromkeys(
                (
                    *_synth_unknowns(stock, bull, bear, disagreement),
                    *_unknown_claims(claims),
                )
            )
        ),
        claims=claims,
        what_would_change=_synth_changes(stock, bull, bear),
        executive_summary=stock.executive_view or stock.base_case or "No grounded findings.",
        consensus=_consensus_text(disagreement),
        impact_channels=_channels_from_analyses(stock, bull, bear) or _channels_from_claims(claims),
        first_order_effects=first_order,
        second_order_effects=second_order,
        bull_evidence_ids=claims_refs(bull.claims),
        bear_evidence_ids=claims_refs(bear.claims),
        critical_disagreements=_critical_disagreements(disagreement),
        evidence_limitations=_synth_limitations(evidence_limitations, scope),
        research_scope=scope,
        direct_evidence=_observation_rows(observations_list),
        absence_observations=[scoped_absence(text) for text in _strs(absence_observations)],
        filing_references=claims_refs(claims),
        positioning=_strs(positioning),
        catalysts=_strs(catalysts),
        coverage=coverage_norm,
        sources=_normalize_sources(sources, claims, observations_list),
    )
    synth.answer = _deep_answer(synth) if model is None else str(model)
    return synth


__all__ = [
    "SCOPE_ABSENCE",
    "SCOPE_LIMITATION",
    "SEC_ONLY_LIMITATION",
    "SEC_SCOPE_ABSENCE",
    "FinalSynthesis",
    "scoped_absence",
    "synthesize_final",
]
