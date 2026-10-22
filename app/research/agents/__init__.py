"""Agent package: shared request/claim types + committee envelope parsing.

Fake-model sketch (no live calls): build a ``ScoutAssignment``, pass a
``dispatch`` callable wrapping ``app.tools.execute_tool`` (or a dict-returning
fake), pass a ``model`` callable returning canned JSON, assert on
``ScoutResult`` / ``StockbotAnalysis`` fields.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.research.models import FailureCategory

CLAIM_TYPES = (
    "observed_fact",
    "calculation",
    "inference",
    "assumption",
    "unknown",
    "contradicted",
    "conditional_risk",
    "attributed_report",
)
"""Claim vocabulary: declared by the authoring model, never inferred from wording.

``observed_fact``/``calculation``/``contradicted``/``inference``/``conditional_risk``/
``attributed_report`` need at least one freeze id; ``unknown`` (nothing located
within the searched scope) may cite none, and so may an ``assumption`` without a
``value`` (a labeled hypothetical, never a verified fact). ``calculation`` cites
the calculation ledger row whose proof recomputes its ``value``.
``conditional_risk`` is a risk disclosure: the cited passage supports the risk as
stated, not that the risk materialized. ``attributed_report`` is a reporting
claim: the cited passage supports that the report was made, not the underlying
reported event.
"""

DATAPOINT_FIELDS = ("value", "entity", "period", "unit")
"""Typed datapoint carrier on one claim (same wire names in Python and TS)."""

_VALUELESS_TYPES = frozenset({"inference", "unknown"})
"""Claim types that never carry a datapoint value (an inference stays qualified)."""

MATERIALITY_LEVELS = ("critical", "high", "medium", "low")

COMMITTEE_REQUIRED_KEYS = (
    "executive_view",
    "claims",
    "impact_channels",
    "materiality",
    "uncertainties",
    "what_would_change",
    "follow_ups",
)
COMMITTEE_ENVELOPE_ERROR = "ERR_COMMITTEE_ENVELOPE_INCOMPLETE"


@dataclass
class ResearchRequest:
    """One actionable follow-up question from a committee agent."""

    question: str
    why_material: str
    requested_source_domain: str
    expected_gain: str
    requesting_agents: list[str] = field(default_factory=list)


class ModelOutputFailure(ValueError):
    """Malformed model output: unknown id, uncited claim, or incomplete envelope."""

    _failure_category: FailureCategory

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self._failure_category = FailureCategory.MODEL_OUTPUT_FAILURE


@dataclass
class ImpactChannel:
    """One impact channel: what it does to the question, its direction, frozen evidence links."""

    text: str
    direction: str = ""
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class CommitteeMateriality:
    """Overall materiality judgment (routing only, never a score)."""

    overall: str = "medium"
    reasoning: str = ""


@dataclass
class GroundedClaim:
    """One claim with its declared type, freeze-contained evidence links, and optional typed datapoint."""

    text: str
    claim_type: str = "inference"
    evidence_ids: list[str] = field(default_factory=list)
    value: str | None = None
    entity: str | None = None
    period: str | None = None
    unit: str | None = None


def _frozen_set(frozen: Sequence[str]) -> set[str]:
    """Known freeze ids (non-empty strings only)."""
    return {e for e in frozen if isinstance(e, str) and e}


def _decode_claims_list(text: str) -> list[object]:
    """Parse one JSON list document from model text."""
    stripped = text.strip()
    if not stripped:
        raise ModelOutputFailure("claims must be one JSON document (got blank)")
    try:
        decoded: object = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ModelOutputFailure(f"claims must be one JSON document: {exc}") from exc
    if not isinstance(decoded, list):
        raise ModelOutputFailure("claims must be a JSON list")
    return decoded


def _claim_text(item: dict[object, object]) -> str:
    """Validated non-empty claim text (truncation happens at build time)."""
    raw_text = item.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ModelOutputFailure("each claim needs non-empty text")
    return raw_text


def _claim_type(item: dict[object, object]) -> str:
    """Declared claim type; an absent type is ``inference`` (never observed_fact)."""
    raw = item.get("claim_type")
    if raw is None:
        return "inference"
    if isinstance(raw, str) and raw.strip().lower() in CLAIM_TYPES:
        return raw.strip().lower()
    raise ModelOutputFailure(f"claim_type must be one of {CLAIM_TYPES}, got {raw!r}")


def _grounded_ids(raw_ids: object, frozen_set: set[str]) -> list[str] | None:
    """Deduped non-empty ids, all inside the freeze; None when malformed, raises on unknown id."""
    if not isinstance(raw_ids, list) or any(not isinstance(eid, str) or not eid for eid in raw_ids):
        return None
    unknown = next((eid for eid in raw_ids if eid not in frozen_set), None)
    if unknown is not None:
        raise ModelOutputFailure(f"unknown evidence id {unknown!r}")
    return list(dict.fromkeys(raw_ids))


def _claim_ids(item: dict[object, object], raw_text: str, frozen_set: set[str], claim_type: str) -> list[str]:
    """Validated deduped evidence ids, all contained in the freeze.

    ``unknown`` and a value-less ``assumption`` may cite nothing; every other claim needs at least one id.
    """
    seen = _grounded_ids(item.get("evidence_ids"), frozen_set)
    if seen is None:
        raise ModelOutputFailure("each claim needs an evidence_ids list of ids")
    citeless = claim_type == "unknown" or (claim_type == "assumption" and item.get("value") is None)
    if not seen and not citeless:
        raise ModelOutputFailure(f"uncited {claim_type} claim: {raw_text[:120]!r}")
    return seen


def _datapoint(item: dict[object, object], raw_text: str, claim_type: str) -> dict[str, str | None]:
    """Typed datapoint fields: each absent/None or a non-blank string; the value sits verbatim in the text."""
    out: dict[str, str | None] = {}
    for key in DATAPOINT_FIELDS:
        raw = item.get(key)
        if raw is not None and (not isinstance(raw, str) or not raw.strip()):
            raise ModelOutputFailure(f"claim {key} must be a non-empty string when present, got {raw!r}")
        out[key] = raw.strip()[:500] if isinstance(raw, str) else None
    value = out["value"]
    if value is not None and claim_type in _VALUELESS_TYPES:
        raise ModelOutputFailure(f"{claim_type} claim cannot carry a datapoint value: {raw_text[:120]!r}")
    if value is None and claim_type == "calculation":
        raise ModelOutputFailure(f"calculation claim needs a value: {raw_text[:120]!r}")
    if value is not None and value not in raw_text:
        raise ModelOutputFailure(f"claim value {value!r} is not verbatim in its text: {raw_text[:120]!r}")
    return out


_CLAIM_KEYS = frozenset({"text", "claim_type", "evidence_ids", *DATAPOINT_FIELDS})


def _build_claim(item: object, frozen_set: set[str]) -> GroundedClaim:
    """Validate one {text, claim_type, evidence_ids, value?, entity?, period?, unit?} record against the freeze."""
    if not isinstance(item, dict):
        raise ModelOutputFailure("each claim must be {text, claim_type, evidence_ids}")
    keys = set(item)
    if keys - _CLAIM_KEYS or "evidence_ids" not in keys or "text" not in keys:
        raise ModelOutputFailure(
            "each claim must contain {text, claim_type, evidence_ids} plus only optional value/entity/period/unit"
        )
    raw_text = _claim_text(item)
    claim_type = _claim_type(item)
    return GroundedClaim(
        text=raw_text.strip()[:500],
        claim_type=claim_type,
        evidence_ids=_claim_ids(item, raw_text, frozen_set, claim_type),
        **_datapoint(item, raw_text.strip()[:500], claim_type),
    )


def parse_grounded_claims(text: str, *, frozen: Sequence[str]) -> list[GroundedClaim]:
    """Parse model JSON records; each claim must cite only freeze ids.

    Expected shape: [{"text": "...", "claim_type": "...", "evidence_ids": [...]}, ...]
    plus optional typed ``value``/``entity``/``period``/``unit`` strings. A missing
    ``claim_type`` is ``inference``; ``unknown`` (and a value-less ``assumption``)
    may leave ``evidence_ids`` empty. Unknown IDs (e.g. EV-999), uncited claims,
    a value on an inference/unknown, a value missing from the text, or
    blank/non-JSON output raise ``ModelOutputFailure``. Represent nothing found
    explicitly as [].
    """
    frozen_set = _frozen_set(frozen)
    return [_build_claim(item, frozen_set) for item in _decode_claims_list(text)]


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _embedded_claim_list(text: str) -> list[object] | None:
    """Claim list from fenced or prose-wrapped model text; None when nothing parses."""
    for match in _FENCED_JSON.finditer(text):
        decoded = _try_json_list(match.group(1))
        if decoded is not None and any(isinstance(item, dict) for item in decoded):
            return decoded
    # A bare claim object before bracket scanning: `[...]` inside its own fields
    # must never be mistaken for the claim list.
    single = _try_json_object(text)
    if single is not None:
        return [single]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        decoded = _try_json_list(text[start : end + 1])
        if decoded is not None:
            return decoded
    return None


def _try_json_list(text: str) -> list[object] | None:
    try:
        decoded: object = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, list) else None


def _try_json_object(text: str) -> object | None:
    try:
        decoded: object = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _decode_claims_list_relaxed(text: str) -> list[object] | None:
    """Strict JSON list first, then fenced/prose-wrapped content; None when unparseable."""
    stripped = text.strip()
    if not stripped:
        return None
    decoded = _try_json_list(stripped)
    if decoded is not None:
        return decoded
    return _embedded_claim_list(stripped)


def parse_grounded_claims_tolerant(
    text: str, *, frozen: Sequence[str], on_reject: Callable[[str, str], None] | None = None
) -> list[GroundedClaim]:
    """Scout-stage parse: keep the claims that ground, report the ones that do not.

    A long exploratory stage must not abort because one model record cited an id
    it invented: the offending record is dropped and reported (never accepted),
    while the rest of the findings survive. Blank/non-JSON output is still a
    ``ModelOutputFailure`` (the model failed to answer in shape at all).
    """
    frozen_set = _frozen_set(frozen)
    items = _decode_claims_list_relaxed(text)
    if items is None:
        if on_reject is not None:
            on_reject(text.strip()[:200] or "<blank>", "model output was not a JSON claim list")
        return []
    kept: list[GroundedClaim] = []
    for item in items:
        try:
            kept.append(_build_claim(item, frozen_set))
        except ModelOutputFailure as exc:
            if on_reject is not None:
                shown = json.dumps(item, default=str)[:200]
                on_reject(shown, str(exc))
    return kept


def claims_refs(claims: Sequence[GroundedClaim]) -> list[str]:
    """Union of claim evidence ids in first-seen order."""
    refs: list[str] = []
    for claim in claims:
        for eid in claim.evidence_ids:
            if eid not in refs:
                refs.append(eid)
    return refs


_CONSERVATIVE_ORDER = (
    "contradicted",
    "unknown",
    "assumption",
    "conditional_risk",
    "attributed_report",
    "inference",
    "calculation",
    "observed_fact",
)


def conservative_claim_type(claim_types: Sequence[str]) -> str:
    """Least assertive declared type among duplicates (never upgrades a claim)."""
    rank = {claim_type: i for i, claim_type in enumerate(_CONSERVATIVE_ORDER)}
    declared = [claim_type for claim_type in claim_types if claim_type in rank]
    if not declared:
        return "inference"
    return min(declared, key=rank.__getitem__)


def _decode_envelope(text: str) -> dict[object, object]:
    """Parse one JSON object document from committee model text."""
    stripped = text.strip()
    if not stripped:
        raise ModelOutputFailure("committee output must be one JSON document (got blank)")
    try:
        decoded: object = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ModelOutputFailure(f"committee output must be one JSON document: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ModelOutputFailure("committee output must be a JSON object")
    return decoded


def _envelope_lists(decoded: dict[object, object]) -> tuple[list[object], list[object]]:
    """Validated claims + follow_ups lists from the envelope."""
    raw_claims = decoded["claims"]
    raw_follow = decoded["follow_ups"]
    assert isinstance(raw_claims, list) and isinstance(raw_follow, list)
    return raw_claims, raw_follow


_MATERIALITY_ERROR = (
    f"{COMMITTEE_ENVELOPE_ERROR}: committee 'materiality' must be "
    f"{{overall: one of {MATERIALITY_LEVELS}, reasoning: str}}"
)

_ENVELOPE_SHAPE: tuple[tuple[str, type, str], ...] = (
    ("claims", list, "must be a list"),
    ("impact_channels", list, "must be a list"),
    ("follow_ups", list, "must be a list"),
    ("uncertainties", list, "must be a list"),
    ("what_would_change", list, "must be a list"),
    ("executive_view", str, "must be a string"),
)


def _require_materiality_payload(raw: object) -> None:
    """Rich materiality shape: {overall: one of MATERIALITY_LEVELS, reasoning: str}."""
    payload = raw if isinstance(raw, dict) else {}
    overall = payload.get("overall")
    if (
        not isinstance(overall, str)
        or overall.strip().lower() not in MATERIALITY_LEVELS
        or not isinstance(payload.get("reasoning"), str)
    ):
        raise ModelOutputFailure(_MATERIALITY_ERROR)


def _require_rich_envelope(decoded: dict[object, object]) -> None:
    """Reject claims/follow_ups-only and otherwise incomplete committee envelopes.

    Everything missing or mistyped fails closed with ``ERR_COMMITTEE_ENVELOPE_INCOMPLETE``:
    a role that cannot fill the rich envelope has not produced a committee read.
    """
    missing = [key for key in COMMITTEE_REQUIRED_KEYS if key not in decoded]
    if missing:
        raise ModelOutputFailure(f"{COMMITTEE_ENVELOPE_ERROR}: committee envelope missing {missing}")
    for key, expected, requirement in _ENVELOPE_SHAPE:
        if not isinstance(decoded[key], expected):
            raise ModelOutputFailure(f"{COMMITTEE_ENVELOPE_ERROR}: committee {key!r} {requirement}")
    _require_materiality_payload(decoded["materiality"])


def _follow_up_question(item: object) -> str:
    """Validated follow-up question text from a string or question/text mapping."""
    raw_q: object = item.get("question", item.get("text", "")) if isinstance(item, dict) else item
    if not isinstance(raw_q, str):
        raise ModelOutputFailure(f"committee follow_up must be a question string, got {type(item).__name__}")
    q = raw_q.strip()
    if len(q) < 12 or len(q) > 500 or not q.endswith("?"):
        raise ModelOutputFailure(f"malformed committee follow_up: {q[:120]!r}")
    return q


def _build_follow_up(item: object, agent: str) -> ResearchRequest:
    """Validate one follow-up question string or {question, why_it_matters, suggested_source} mapping."""
    q = _follow_up_question(item)
    why = (
        item.get("why_it_matters", item.get("why_material", "committee follow-up"))
        if isinstance(item, dict)
        else "committee follow-up"
    )
    src = item.get("suggested_source", item.get("requested_source_domain", "SEC")) if isinstance(item, dict) else "SEC"
    return ResearchRequest(
        question=q,
        why_material=why.strip() if isinstance(why, str) and why.strip() else "committee follow-up",
        requested_source_domain=src.strip() if isinstance(src, str) and src.strip() else "SEC",
        expected_gain="medium",
        requesting_agents=[agent],
    )


def _str_or_blank(value: object, cap: int = 2000) -> str:
    """Strip one optional prose field; non-strings coerce to blank."""
    return value.strip()[:cap] if isinstance(value, str) else ""


def _parse_channel(item: object, frozen_set: set[str]) -> ImpactChannel | None:
    """One {text, direction, evidence_ids} channel; None when malformed or ungrounded (skipped)."""
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    seen = _grounded_ids(item.get("evidence_ids", []), frozen_set)
    if not isinstance(text, str) or not text.strip() or not seen:
        return None
    return ImpactChannel(
        text=text.strip()[:500], direction=_str_or_blank(item.get("direction"), 200), evidence_ids=seen
    )


def _parse_materiality(raw: object) -> CommitteeMateriality:
    """Read one validated {overall, reasoning} payload (gate checked it already)."""
    assert isinstance(raw, dict)
    overall = raw["overall"]
    assert isinstance(overall, str)
    return CommitteeMateriality(overall=overall.strip().lower(), reasoning=_str_or_blank(raw.get("reasoning")))


def _parse_str_items(raw: object, cap: int = 2000) -> list[str]:
    """Filter one string list down to stripped non-empty entries."""
    if not isinstance(raw, list):
        return []
    return [s.strip()[:cap] for s in raw if isinstance(s, str) and s.strip()]


@dataclass
class CommitteeEnvelope:
    """Committee role output: view, typed claims, channels, materiality, uncertainties, changes, requests."""

    claims: list[GroundedClaim] = field(default_factory=list)
    follow_ups: list[ResearchRequest] = field(default_factory=list)
    executive_view: str = ""
    impact_channels: list[ImpactChannel] = field(default_factory=list)
    materiality: CommitteeMateriality = field(default_factory=CommitteeMateriality)
    uncertainties: list[str] = field(default_factory=list)
    what_would_change: list[str] = field(default_factory=list)


def parse_committee_envelope(text: str, *, frozen: Sequence[str], agent: str) -> CommitteeEnvelope:
    """Rich committee envelope; incomplete envelopes fail ``ERR_COMMITTEE_ENVELOPE_INCOMPLETE``."""
    decoded = _decode_envelope(text)
    _require_rich_envelope(decoded)
    frozen_set = _frozen_set(frozen)
    raw_claims, raw_follow = _envelope_lists(decoded)
    claims = parse_grounded_claims(json.dumps(raw_claims), frozen=frozen)
    follow_ups = [_build_follow_up(item, agent) for item in raw_follow]
    raw_channels = decoded["impact_channels"]
    assert isinstance(raw_channels, list)
    parsed: list[ImpactChannel] = []
    for item in raw_channels:
        channel = _parse_channel(item, frozen_set)
        if channel is not None:
            parsed.append(channel)
    return CommitteeEnvelope(
        claims=claims,
        follow_ups=follow_ups,
        executive_view=_str_or_blank(decoded["executive_view"]),
        impact_channels=parsed,
        materiality=_parse_materiality(decoded["materiality"]),
        uncertainties=_parse_str_items(decoded["uncertainties"]),
        what_would_change=_parse_str_items(decoded["what_would_change"]),
    )


def parse_committee_output(
    text: str, *, frozen: Sequence[str], agent: str
) -> tuple[list[GroundedClaim], list[ResearchRequest]]:
    """Committee envelope reduced to (claims, follow_ups)."""
    env = parse_committee_envelope(text, frozen=frozen, agent=agent)
    return env.claims, env.follow_ups


__all__ = [
    "CLAIM_TYPES",
    "COMMITTEE_ENVELOPE_ERROR",
    "COMMITTEE_REQUIRED_KEYS",
    "MATERIALITY_LEVELS",
    "CommitteeEnvelope",
    "CommitteeMateriality",
    "GroundedClaim",
    "ImpactChannel",
    "ModelOutputFailure",
    "ResearchRequest",
    "claims_refs",
    "conservative_claim_type",
    "parse_committee_envelope",
    "parse_committee_output",
    "parse_grounded_claims",
    "parse_grounded_claims_tolerant",
]
