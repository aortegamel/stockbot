"""Bullbot committee agent: bullish read of one evidence freeze.

Same freeze input as stockbot/bearbot; no research caps (frozen evidence
only, never calls tools). Model argues the bull case; infra validates refs
against the freeze and packages ``research_requests``.

Fake-model sketch (no live calls): fake ``model(prompt)`` returns one canned
rich-envelope JSON object (executive_view, typed claims, impact channels,
materiality, uncertainties, what_would_change, follow_ups); call
``run_bullbot``; assert ``stance`` is bullish and refs stay within the freeze.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from . import (
    CLAIM_WORDING_RULE,
    CommitteeMateriality,
    GroundedClaim,
    ImpactChannel,
    ResearchRequest,
    parse_committee_envelope,
)
from .scout import ModelFn


def _coerce_wave(wave_id: int | str) -> int:
    """Accept int>=1 or numeric str; reject bool/non-numeric/<1."""
    if isinstance(wave_id, bool):
        raise ValueError(f"bullbot: 'wave_id' must be an int >= 1, got {wave_id!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    if isinstance(wave_id, int):
        wave = wave_id
    elif isinstance(wave_id, str):
        text = wave_id.strip()
        if not text.isdigit():
            raise ValueError(f"bullbot: 'wave_id' must be an int >= 1, got {wave_id!r}")
        wave = int(text)
    else:
        raise ValueError(f"bullbot: 'wave_id' must be an int >= 1, got {wave_id!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    if wave < 1:
        raise ValueError(f"bullbot: 'wave_id' must be >= 1, got {wave_id!r}")
    return wave


@dataclass
class BullAnalysis:
    session_id: str
    wave_id: int
    freeze_id: str
    evidence_ids: list[str]
    as_of: str
    question: str
    stance: str
    bull_case: str
    unknowns: list[str] = field(default_factory=list)
    what_would_change: list[str] = field(default_factory=list)
    claims: list[GroundedClaim] = field(default_factory=list)
    research_requests: list[ResearchRequest] = field(default_factory=list)
    executive_view: str = ""
    impact_channels: list[ImpactChannel] = field(default_factory=list)
    materiality: CommitteeMateriality = field(default_factory=CommitteeMateriality)
    uncertainties: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.wave_id = _coerce_wave(self.wave_id)


def _tag_requests(extra: list[ResearchRequest]) -> list[ResearchRequest]:
    """Tag follow-ups with the bullbot agent."""
    for request in extra:
        if "bullbot" not in request.requesting_agents:
            request.requesting_agents.append("bullbot")
    return extra


def _prose_or_placeholder(claims: list[GroundedClaim], frozen: list[str]) -> str:
    """Joined claim text, or the empty-freeze placeholder."""
    prose = "\n".join(c.text for c in claims).strip()
    if prose:
        return prose
    return "No grounded claims in freeze." if frozen else "Freeze holds no evidence."


def run_bullbot(
    question: str,
    *,
    session_id: str,
    wave_id: int | str,
    freeze_id: str,
    evidence_ids: Sequence[str],
    as_of: str,
    model: ModelFn,
    follow_ups: Sequence[ResearchRequest] | None = None,
    evidence_text: str = "",
) -> BullAnalysis:
    """Bullish synthesis over the frozen evidence set (no tool calls; wave_id stored as int)."""
    wave = _coerce_wave(wave_id)
    frozen = list(evidence_ids)
    prompt = (
        "Role: bull case. Build the strongest resilience case: limited damage, buffers, mitigants, "
        f"contractual protection, or diversified revenue. Question: {question}\n"
        f"Freeze: {freeze_id} as of {as_of} evidence={len(frozen)}\n"
        'Respond with one JSON object only: {"role": "bullbot", "executive_view": "<bullish read>", '
        '"claims": [{"text": "<finding>", "claim_type": "observed_fact|calculation|inference|unknown|contradicted|conditional_risk|attributed_report", "evidence_ids": ["<freeze-id>", ...], "datapoint_ids": ["<dp:id>", ...]}], '
        '"impact_channels": [{"text": "<channel>", "direction": "<what it does to the question>", "evidence_ids": ["<freeze-id>"]}], '
        '"materiality": {"overall": "critical|high|medium|low", "reasoning": "<why>"}, '
        '"uncertainties": ["<what the freeze does not establish>"], '
        '"what_would_change": ["<concrete disclosure that would move this view>"], '
        '"follow_ups": [{"question": "<follow-up>?", "why_it_matters": "<why>", "suggested_source": "SEC"}]}. '
        "claim_type is declared, never inferred from wording: observed_fact when the cited passage states it directly, "
        "inference when it is reasoned from cited evidence, contradicted when cited evidence conflicts with the claim text, "
        "conditional_risk when the cited passage discloses the risk as a risk (the citation supports the qualified risk wording, not that it materialized), "
        "attributed_report when the cited passage reports someone else's statement (the citation supports that it was reported, not the underlying event), "
        "unknown when the frozen evidence does not establish it (for example no disclosure located within the searched scope). "
        "Cite both sides when evidence conflicts and declare contradicted. "
        "observed_fact/inference/contradicted/conditional_risk/attributed_report need at least one freeze id; unknown may cite none. "
        "Cite only freeze ids." + CLAIM_WORDING_RULE
    )
    if evidence_text.strip():
        prompt += f"\nEvidence (cite ids):\n{evidence_text.strip()}"
    text = model(prompt).strip()
    extra = _tag_requests(list(follow_ups or []))
    env = parse_committee_envelope(text, frozen=frozen, agent="bullbot")
    unknowns: list[str] = (
        list(env.uncertainties) if env.uncertainties else ([] if env.claims or frozen else ["freeze holds no evidence"])
    )
    prose = _prose_or_placeholder(env.claims, frozen)
    extra = list(env.follow_ups) + list(extra)
    view = env.executive_view or prose
    return BullAnalysis(
        session_id=session_id,
        wave_id=wave,
        freeze_id=freeze_id,
        evidence_ids=frozen,
        as_of=as_of,
        question=question,
        stance="bullish",
        bull_case=view,
        unknowns=unknowns,
        what_would_change=list(env.what_would_change),
        claims=env.claims,
        research_requests=extra,
        executive_view=view,
        impact_channels=list(env.impact_channels),
        materiality=env.materiality,
        uncertainties=list(unknowns),
    )


__all__ = ["BullAnalysis", "run_bullbot"]
