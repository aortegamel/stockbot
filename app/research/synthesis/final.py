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

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace

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

SCOPE_LIMITATION = (
    "Searched-source scope is unknown beyond the searched sources:"
    " only the allowed sources were searched; unsearched sources carry no finding."
)
SEC_ONLY_LIMITATION = (
    "SEC-only scope is unknown beyond SEC filings:"
    " no non-SEC source (news, transcripts, private documents, market data) was searched."
)
"""Legacy SEC-only limitation (kept for pinned readers); prefer SCOPE_LIMITATION.

Both disclose an explicit unknown: what lies beyond the searched sources is
unknown, so a generator-created ``unknown`` claim may inventory the sentence.
"""

_CLAIM_SEVERITY = {
    "observed_fact": "direct",
    "calculation": "derived",
    "inference": "indirect",
    "assumption": "hypothetical",
    "unknown": "uncertain",
    "contradicted": "contested",
    "conditional_risk": "conditional",
    "attributed_report": "attributed",
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
    calculations: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.wave_id = _coerce_wave_id(self.wave_id)

    def to_dict(self) -> dict[str, object]:
        """Deep FinalResearchResult dict (legacy answer/claims keys kept)."""
        claim_rows = [_effect_row(c) for c in self.claims]
        return {
            "answer": self.answer,
            "executive_summary": self.executive_summary or self.answer,
            "consensus": self.consensus,
            "base_case": self.base_case,
            "major_evidence": _copy_rows(self.direct_evidence),
            "direct_evidence": _copy_rows(self.direct_evidence),
            "calculations": deepcopy(self.calculations),
            "bull_case": {"summary": self.bull_case, "evidence_ids": list(self.bull_evidence_ids)},
            "bear_case": {"summary": self.bear_case, "evidence_ids": list(self.bear_evidence_ids)},
            "impact_channels": _copy_rows(self.impact_channels),
            "first_order_effects": _copy_rows(self.first_order_effects),
            "second_order_effects": _copy_rows(self.second_order_effects),
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
            "sources": _copy_rows(self.sources),
            "grounded_claims": _copy_rows(claim_rows),
            "claims": _copy_rows(claim_rows),
            "filing_references": list(self.filing_references),
            "evidence_refs": list(self.filing_references),
            "research_scope": dict(self.research_scope),
            "freeze_id": self.freeze_id,
            "as_of": self.as_of,
        }


def _copy_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Copy report rows without sharing their mutable mappings."""
    return [dict(row) for row in rows]


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
    """One effect row: claim text with its declared type, freeze refs, and canonical datapoint references."""
    row: dict[str, object] = {
        "text": claim.text,
        "claim_type": claim.claim_type,
        "evidence_ids": list(claim.evidence_ids),
    }
    if claim.datapoint_ids:
        row["datapoint_ids"] = list(claim.datapoint_ids)
    return row


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
    if _is_calculation(row):
        return None
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


_SOURCE_ID_KEYS = ("source_uri", "source_record_id", "accession_no", "source_name", "tool_result_id")
"""Exact source-identity keys included in each calculation input's provenance."""


def _is_calculation(row: Mapping[str, object]) -> bool:
    """Keep calculated values separate from passages in external documents."""
    proof = row.get("proof")
    return row.get("claim_kind") == "calculation" or (isinstance(proof, Mapping) and proof.get("kind") == "calculation")


def _audit_verification(
    row: Mapping[str, object], by_id: Mapping[str, Mapping[str, object]], depth: int, as_of: str
) -> tuple[str, str]:
    """Replay the canonical proof instead of trusting a persisted status."""
    from app.research.evidence import evidence_from_dict
    from app.research.service import _proof_verdict

    try:
        evidence = evidence_from_dict(row)
    except ValueError as exc:
        return "unverified", f"Ledger integrity failed: {exc}"
    verdict = _proof_verdict(by_id, evidence, evidence.value, depth, as_of=as_of)
    return verdict.status, verdict.reason


def _audit_source(row: Mapping[str, object], proof: Mapping[str, object]) -> dict[str, object]:
    """Identify each input's own source without attributing calculations to it."""
    if _is_calculation(row):
        return {"producer": "Stockbot", "evidence_id": row.get("evidence_id")}
    provenance = _observation_provenance(row)
    document, accession = _observation_identity(row, provenance)
    identity = {key: row.get(key) for key in _SOURCE_ID_KEYS if row.get(key) is not None}
    identity.update({key: provenance.get(key) for key in _SOURCE_ID_KEYS if provenance.get(key) is not None})
    return {
        "domain": _observation_domain(row),
        "document": document,
        "accession_no": accession,
        "source_name": row.get("source_name"),
        "source_record_id": row.get("source_record_id"),
        "source_uri": row.get("source_uri"),
        "sourceId": proof.get("sourceId"),
        "locator": deepcopy(proof.get("locator")),
        "provenance": deepcopy(dict(provenance)),
        "source_identity": identity,
    }


def _calculation_view(
    row: Mapping[str, object],
    by_id: Mapping[str, Mapping[str, object]],
    as_of: str,
    ancestors: tuple[str, ...] = (),
) -> dict[str, object]:
    """Render ordered inputs with a maximum of sixteen calculation levels."""
    from app.research.service import _MAX_PROOF_DEPTH, _recompute

    evidence_id = _observation_field(row, "evidence_id")
    raw_proof = row.get("proof")
    proof = raw_proof if isinstance(raw_proof, Mapping) else {}
    calculated = _is_calculation(row)
    status, reason = _audit_verification(row, by_id, len(ancestors), as_of)
    view: dict[str, object] = {
        "evidence_id": evidence_id,
        "claim_type": "calculation" if calculated else row.get("claim_kind", "observed_fact"),
        "value": row.get("value"),
        "entity": row.get("entity"),
        "period": row.get("period"),
        "unit": row.get("unit"),
        "as_of": as_of,
        "known_at": row.get("known_at"),
        "source": _audit_source(row, proof),
        "verification": status,
        "verification_status": status,
        "verification_reason": reason,
    }
    if not calculated:
        return view
    metadata = row.get("metadata")
    metric = _as_text(metadata.get("metric")) if isinstance(metadata, Mapping) else ""
    raw_ids = proof.get("inputProofIds")
    input_ids: list[object] = list(raw_ids) if isinstance(raw_ids, (list, tuple)) else []
    view.update(
        metric=metric or _as_text(row.get("subject")) or _as_text(row.get("claim_text")),
        result=row.get("value"),
        formula=_as_text(row.get("content")),
        formulaId=proof.get("formulaId"),
        formulaVersion=proof.get("formulaVersion"),
        formula_id=proof.get("formulaId"),
        formula_version=proof.get("formulaVersion"),
        inputProofIds=input_ids,
        inputs=[],
        recomputedValue=None,
        recomputed_result=None,
    )
    if evidence_id in ancestors or len(ancestors) >= _MAX_PROOF_DEPTH:
        view.update(
            verification="unverified",
            verification_status="unverified",
            verification_reason="Calculation inputs contain a cycle or exceed the depth limit.",
        )
        return view
    inputs: list[dict[str, object]] = []
    for input_id in input_ids:
        source = by_id.get(str(input_id))
        if source is None:
            inputs.append(
                {
                    "evidence_id": input_id,
                    "verification": "unverified",
                    "verification_status": "unverified",
                    "verification_reason": "The input is absent from the frozen observations.",
                }
            )
        else:
            inputs.append(_calculation_view(source, by_id, as_of, (*ancestors, evidence_id)))
    view["inputs"] = inputs
    values = tuple(item.get("value") for item in inputs)
    if values and all(isinstance(value, str) for value in values):
        try:
            recomputed = _recompute(
                str(proof.get("formulaId", "")), str(proof.get("formulaVersion", "")), tuple(str(v) for v in values)
            )
        except ArithmeticError, ValueError:
            recomputed = None  # The replay verdict retains the reason for failure.
        view.update(recomputedValue=recomputed, recomputed_result=recomputed)
    return view


def _calculation_views(observations: Sequence[Mapping[str, object]], as_of: str) -> list[dict[str, object]]:
    """Build audit views only from the supplied freeze, including nested inputs."""
    by_id = _source_observations_by_id(observations)
    return [_calculation_view(row, by_id, as_of) for row in by_id.values() if _is_calculation(row)]


def _calculation_lines(rows: Sequence[Mapping[str, object]], depth: int = 0) -> list[str]:
    """Render the audit tree without describing computed results as source passages."""
    lines: list[str] = []
    for index, row in enumerate(rows, 1):
        prefix = "  " * depth
        identity = f"[{row.get('evidence_id', '')}]"
        verification = f"{row.get('verification', 'unverified')}: {row.get('verification_reason', '')}"
        if row.get("claim_type") == "calculation":
            lines.extend(
                (
                    f"{prefix}Stockbot calculation {identity}: {row.get('metric')} = {row.get('result')} "
                    + f"{row.get('unit') or ''}",
                    f"{prefix}Formula: {row.get('formulaId')}@{row.get('formulaVersion')} — {row.get('formula')}",
                    f"{prefix}Recomputed result: {row.get('recomputedValue')}. "
                    + f"Entity: {row.get('entity')}. Period: {row.get('period')}. As of: {row.get('as_of')}.",
                    f"{prefix}Verification: {verification}",
                )
            )
            raw_inputs = row.get("inputs")
            inputs = [item for item in raw_inputs if isinstance(item, Mapping)] if isinstance(raw_inputs, list) else []
            lines.extend(_calculation_lines(inputs, depth + 1))
        else:
            lines.append(
                f"{prefix}Input {index} {identity}: {row.get('value')} {row.get('unit') or ''}. "
                f"Entity: {row.get('entity')}. Period: {row.get('period')}. As of: {row.get('as_of')}. "
                f"Verification: {verification}. Source: {json.dumps(row.get('source'), sort_keys=True)}"
            )
    return lines


def _split_effects(claims: Sequence[GroundedClaim]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Observed/calculated claims vs reasoned/contested/qualified claims; unknown reads stay in the unknowns section."""
    first = [_effect_row(c) for c in claims if c.claim_type in ("observed_fact", "calculation")]
    second = [
        _effect_row(c)
        for c in claims
        if c.claim_type in ("inference", "assumption", "contradicted", "conditional_risk", "attributed_report")
    ]
    return first, second


def _coerce_extra_claim(item: object) -> GroundedClaim | None:
    """One caller claim in grounded shape (datapoint references kept); None when blank, ungrounded, or mistyped."""
    if isinstance(item, GroundedClaim):
        raw_text, raw_type, raw_ids, raw_points = item.text, item.claim_type, item.evidence_ids, item.datapoint_ids
    elif isinstance(item, Mapping):
        raw_text = _channel_field(item, "text", "claim_text", "claim")
        raw_type = _channel_field(item, "claim_type")
        raw_ids = _channel_field(item, "evidence_ids", "refs")
        raw_points = _channel_field(item, "datapoint_ids")
    else:
        return None
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    declared = raw_type.strip().lower() if isinstance(raw_type, str) else ""
    claim_type = declared if declared in CLAIM_TYPES else "inference"
    ids = _cited_channel_ids(raw_ids)
    points = _cited_channel_ids(raw_points)
    citeless = claim_type in ("unknown", "assumption") and not points
    if not text or not (ids or citeless):
        return None
    return GroundedClaim(text=text[:500], claim_type=claim_type, evidence_ids=ids, datapoint_ids=points)


def _normalize_extra(extra: Sequence[GroundedClaim | dict[str, object]] | None) -> list[GroundedClaim]:
    """Caller claims into grounded records (drops ungrounded entries)."""
    return [claim for claim in (_coerce_extra_claim(item) for item in extra or []) if claim is not None]


def _merged_claim(prior: GroundedClaim, claim: GroundedClaim) -> GroundedClaim:
    """Two same-text claims as one: union refs and datapoint references, least assertive declared type."""
    return GroundedClaim(
        text=prior.text,
        claim_type=conservative_claim_type([prior.claim_type, claim.claim_type]),
        evidence_ids=list(dict.fromkeys([*prior.evidence_ids, *claim.evidence_ids])),
        datapoint_ids=list(dict.fromkeys([*prior.datapoint_ids, *claim.datapoint_ids])),
    )


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
            merged.append(replace(claim, text=text, evidence_ids=list(dict.fromkeys(claim.evidence_ids))))
            continue
        merged[prior_index] = _merged_claim(merged[prior_index], claim)
    return merged


def _normalize_scope(scope: Mapping[str, object] | None) -> dict[str, object]:
    """Research scope in spec shape; the searched-source boundary carries its explicit limitation."""
    candidates = (scope.get("allowed_sources"), scope.get("allowed")) if isinstance(scope, Mapping) else ()
    sources = next((_strs(raw) for raw in candidates if isinstance(raw, (list, tuple))), None) or ["SEC"]
    upper = [s.upper() for s in sources]
    sec_only = upper == ["SEC"]
    limitation = SEC_ONLY_LIMITATION if sec_only else SCOPE_LIMITATION
    return {"allowed_sources": sources, "sec_only": sec_only, "limitation": limitation}


def _scope_limitation(scope: Mapping[str, object]) -> str:
    """Explicit searched-source disclosure the generator inventories as unknown."""
    return str(scope["limitation"])


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


def _limitation_claim(limitation: str) -> GroundedClaim:
    """Generator-created unknown inventorying one deterministic scope/coverage disclosure."""
    from app.research.agents import GroundedClaim

    return GroundedClaim(text=limitation, claim_type="unknown")


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


def _coverage_line(key: str, section: Mapping[str, object]) -> str:
    """Render one source's searched scope with the existing detail separators."""
    parts = [f"{name}: {', '.join(items)}" for name, items in _coverage_scope_items(section)]
    detail = _as_text(section.get("detail") or section.get("summary"))
    if not parts and not detail:
        return key.upper()
    head = f"{key.upper()} — {'; '.join(parts)}"
    if not detail:
        return head
    if not parts:
        return f"{head} — {detail}"
    return head + detail


def _coverage_lines(coverage: Mapping[str, object]) -> list[str]:
    """Per-source coverage lines: every surviving scope key renders under its own name."""
    lines: list[str] = []
    for key in ("sec", "finra", "web"):
        section = coverage.get(key)
        if not isinstance(section, Mapping):
            continue
        lines.append(_coverage_line(key, section))
    extra = _strs(coverage.get("gaps"))
    if extra:
        lines.append(f"gaps: {', '.join(extra)}")
    return lines


def _source_label(row: Mapping[str, object]) -> str:
    """Render the domain, integrity class, and document identity."""
    domain = _as_text(row.get("domain")).upper() or "SOURCE"
    integrity = _as_text(row.get("integrity_class") or row.get("integrity")).upper()
    document = _as_text(row.get("document") or row.get("source_name"))
    label = f"{domain} [{integrity}]" if integrity else domain
    return " ".join(part for part in (label, document) if part)


def _source_line(row: Mapping[str, object]) -> str:
    """Render a source label with its evidence reference."""
    evidence_id = _as_text(row.get("evidence_id"))
    head = _source_label(row)
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
    by_id = _source_observations_by_id(observations)
    rows = _caller_source_rows(sources, by_id)
    if rows:
        return rows
    return _cited_source_rows(claims, observations, by_id)


def _source_observations_by_id(observations: Sequence[object]) -> dict[str, Mapping[str, object]]:
    """Index the first canonical observation for each nonblank evidence identifier."""
    by_id: dict[str, Mapping[str, object]] = {}
    for row in observations or []:
        if not isinstance(row, Mapping):
            continue
        evidence_id = _as_text(row.get("evidence_id"))
        if evidence_id and evidence_id not in by_id:
            by_id[evidence_id] = row
    return by_id


def _caller_source_row(
    item: Mapping[str, object], evidence_id: str, obs: Mapping[str, object] | None
) -> dict[str, object]:
    """Resolve one caller source against its canonical domain and integrity."""
    domain = _as_text(item.get("domain")).upper()
    if not domain or domain == "SOURCE":
        domain = _observation_domain(obs) if obs is not None else "SOURCE"
    integrity = _as_text(item.get("integrity_class") or item.get("integrity")).upper()
    if integrity not in ("PRIMARY_DOCUMENT", "CANONICAL_STRUCTURED", "EXTERNAL_SOURCE"):
        integrity = _observation_integrity(obs) if obs is not None else "EXTERNAL_SOURCE"
    return {
        "evidence_id": evidence_id,
        "domain": domain,
        "document": _as_text(item.get("document") or item.get("source_name")),
        "integrity_class": integrity,
    }


def _caller_source_rows(
    sources: Sequence[object] | None,
    by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Normalize caller sources in first-seen order, discarding invalid or duplicate identifiers."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in sources or []:
        if not isinstance(item, Mapping):
            continue
        evidence_id = _as_text(item.get("evidence_id"))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        rows.append(_caller_source_row(item, evidence_id, by_id.get(evidence_id)))
    return rows


def _source_observation_classes(observations: Sequence[Mapping[str, object]]) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve canonical domain and integrity classes for each first-seen observation."""
    domains: dict[str, str] = {}
    integrity_by_id: dict[str, str] = {}
    for row in observations or []:
        if not isinstance(row, Mapping):
            continue
        evidence_id = _as_text(row.get("evidence_id"))
        if evidence_id and evidence_id not in domains:
            domains[evidence_id] = _observation_domain(row)
            integrity_by_id[evidence_id] = _observation_integrity(row)
    return domains, integrity_by_id


def _cited_source_row(
    evidence_id: str,
    obs: Mapping[str, object] | None,
    domains: Mapping[str, str],
    integrity_by_id: Mapping[str, str],
) -> dict[str, object]:
    """Build a cited source row, retaining external-source defaults when no observation exists."""
    document = _as_text(obs.get("document") or obs.get("source_name")) if obs is not None else ""
    return {
        "evidence_id": evidence_id,
        "domain": domains.get(evidence_id, "SOURCE"),
        "document": document,
        "integrity_class": integrity_by_id.get(evidence_id, "EXTERNAL_SOURCE"),
    }


def _cited_source_rows(
    claims: Sequence[GroundedClaim],
    observations: Sequence[Mapping[str, object]],
    by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Derive source rows from unique cited identifiers when caller sources supply none."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    domains, integrity_by_id = _source_observation_classes(observations)
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            clean = evidence_id.strip() if isinstance(evidence_id, str) else ""
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(_cited_source_row(clean, by_id.get(clean), domains, integrity_by_id))
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
    out += _section("Stockbot calculations — Auditable derivations", _calculation_lines(synth.calculations))
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


def _synthesis_claims(
    stock: StockbotAnalysis,
    bull: BullAnalysis,
    bear: BearAnalysis,
    extra_claims: Sequence[GroundedClaim | dict[str, object]] | None,
) -> list[GroundedClaim]:
    """Merge committee claims and grounded caller claims without changing their precedence."""
    return _merge_claims(
        list(getattr(stock, "claims", []) or []),
        list(getattr(bull, "claims", []) or []),
        list(getattr(bear, "claims", []) or []),
        _normalize_extra(extra_claims),
    )


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
    from accepted per-claim mappings, never the whole freeze. Every prose
    field, the caller's answer included, passes ``gate_uncovered_prose``: a
    factual sentence no claim row covers is deleted before publication.

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
    claims = _synthesis_claims(stock, bull, bear, extra_claims)
    scope = _normalize_scope(research_scope)
    claims = [*claims, _limitation_claim(_scope_limitation(scope))]
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
        evidence_limitations=[*_synth_limitations(evidence_limitations, scope)],
        research_scope=scope,
        direct_evidence=_observation_rows(observations_list),
        calculations=_calculation_views(observations_list, as_of),
        absence_observations=[scoped_absence(text) for text in _strs(absence_observations)],
        filing_references=claims_refs(claims),
        positioning=_strs(positioning),
        catalysts=_strs(catalysts),
        coverage=coverage_norm,
        sources=_normalize_sources(sources, claims, observations_list),
    )
    gate_uncovered_prose(synth)
    synth.answer = _deep_answer(synth) if model is None else _gate_text(str(model), _claim_texts(synth))
    return synth


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z0-9]+")
_CONNECTIVES = frozenset({"and", "but", "while", "whereas", "also", "however", "or"})
"""Words that may join covered claim texts inside one sentence without asserting anything."""


def _plain(text: str) -> str:
    """Citation brackets dropped and whitespace collapsed (no space left before punctuation)."""
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_CITATION.sub(" ", text).split()))


def _claim_texts(synth: FinalSynthesis) -> list[str]:
    """Plain text per inventory row, unknowns included; the publication gate proves each row."""
    return [text for text in (_plain(claim.text) for claim in synth.claims) if text]


def _covered(sentence: str, inventory: Sequence[str]) -> bool:
    """A sentence is covered when claim text states exactly it or combines to fill it with connectives."""
    rest = _plain(sentence)
    if any(text.casefold() == rest.casefold() for text in inventory):
        return True
    for text in sorted(inventory, key=len, reverse=True):
        rest = rest.replace(text, " ")
    return set(_WORD.findall(rest.casefold())) <= _CONNECTIVES


def _gate_text(text: str, inventory: Sequence[str]) -> str:
    """Surgically delete every sentence no claim covers, line by line.

    Every section is gated, hedge prose included: assumption/unknown claim texts
    live in the inventory, so their wording stays while unclaimed wording deletes.
    """
    lines: list[str] = []
    for line in text.splitlines():
        kept = [s for s in _SENTENCE_END.split(line.strip()) if s and _covered(s, inventory)]
        if kept:
            lines.append(" ".join(kept))
    return "\n".join(lines)


def _gate_list(items: Sequence[str], inventory: Sequence[str]) -> list[str]:
    """Gate every entry of one prose list; entries left empty drop out."""
    return [kept for kept in (_gate_text(item, inventory) for item in items) if kept]


def uncovered_sentences(text: str, claim_texts: Sequence[str]) -> list[str]:
    """Sentences of assertive prose (an answer) that no claim text covers, in order."""
    inventory = [plain for plain in (_plain(claim) for claim in claim_texts) if plain]
    return [
        s for line in text.splitlines() for s in _SENTENCE_END.split(line.strip()) if s and not _covered(s, inventory)
    ]


def gate_uncovered_prose(synth: FinalSynthesis) -> FinalSynthesis:
    """Exhaustive prose inventory: every sentence needs a claim row, else it is deleted in place.

    Covers base/bull/bear, executive summary, positioning, catalysts, unknowns,
    what-would-change, limitations, and coverage detail. Assumption and unknown
    claims are inventory rows like any other, so their wording stays. Absence
    observations are coverage artifacts: never citable, never gated.
    """
    texts = _claim_texts(synth)
    synth.base_case = _gate_text(synth.base_case, texts)
    synth.bull_case = _gate_text(synth.bull_case, texts)
    synth.bear_case = _gate_text(synth.bear_case, texts)
    synth.executive_summary = _gate_text(synth.executive_summary, texts) or "No grounded findings."
    synth.positioning = _gate_list(synth.positioning, texts)
    synth.catalysts = _gate_list(synth.catalysts, texts)
    synth.unknowns = _gate_list(synth.unknowns, texts)
    synth.what_would_change = _gate_list(synth.what_would_change, texts)
    synth.evidence_limitations = _gate_list(synth.evidence_limitations, texts)
    for section in synth.coverage.values():
        raw = section.get("detail") if isinstance(section, dict) else None
        if isinstance(section, dict) and isinstance(raw, str):
            detail = _gate_text(raw, texts)
            if detail:
                section["detail"] = detail
            else:
                del section["detail"]
    return synth


__all__ = [
    "SCOPE_ABSENCE",
    "SCOPE_LIMITATION",
    "SEC_ONLY_LIMITATION",
    "SEC_SCOPE_ABSENCE",
    "FinalSynthesis",
    "gate_uncovered_prose",
    "scoped_absence",
    "synthesize_final",
    "uncovered_sentences",
]
