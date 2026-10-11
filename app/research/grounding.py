"""Grounded hypothetical reasoning: exact tool bytes plus declared assumptions. stdlib only.

The reasoner may invent assumption numbers ("terrible" = -X%), but every
impact number must cite exact tool evidence by id, and live traces show the
exact bytes it reasoned on. Pure helpers only — no I/O, no model calls; the
scheduler owns prompts and the drop-before-adjudication gate.

Distinct from app.thesis assumptions (user-thesis free strings): here an
assumption is a per-analysis hypothetical with a stable assumptionId, so a
live trace can tell "model-invented 30% haircut" from "tool-measured 12.5%".
"""

from __future__ import annotations

import re
from collections.abc import Mapping

__all__ = [
    "attach_scenario_impact",
    "check_grounded_analysis",
    "compute_scenario_impact",
    "format_grounded_block",
    "parse_percent",
    "split_grounded_context",
]


def _field(record: object, name: str, default: object = None) -> object:
    """Dict key first, then attribute; default when neither exists."""
    if isinstance(record, dict):
        out: object = record.get(name, default)
        return out
    attr: object = getattr(record, name, default)
    return attr


def _normalize_assumption(item: object, fallback_id: str | None) -> dict[str, str] | None:
    """One {assumptionId, text} mapping, or None when the item declares nothing usable."""
    if isinstance(item, str):
        text = item.strip()
        # Bare strings carry no id, so the caller-supplied position id keeps them citable.
        if text and fallback_id:
            return {"assumptionId": fallback_id, "text": text}
        return None
    if isinstance(item, Mapping):
        aid: object = item.get("assumptionId")
        body: object = item.get("text")
        if isinstance(aid, str) and aid.strip() and isinstance(body, str) and body.strip():
            return {"assumptionId": aid.strip(), "text": body.strip()}
        return None
    return None


def _attempt_assumption_id(attempt: Mapping[str, object], index: int) -> str:
    """Stable citable id for a single-form assumption: declared id wins, else position."""
    declared = attempt.get("assumptionId")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    job = attempt.get("job_id")
    if isinstance(job, str) and job.strip():
        return f"{job.strip()}-assumption"
    return f"attempt-{index}-assumption"


def split_grounded_context(evidence: object, attempts: object) -> dict[str, list[object]]:
    """Admitted exact rows (grounded) versus prior declared assumptions from attempts."""
    grounded: list[object] = list(evidence) if isinstance(evidence, (list, tuple)) else []
    assumptions: list[object] = []
    items: list[object] = list(attempts) if isinstance(attempts, (list, tuple)) else []
    for index, attempt in enumerate(items):
        if not isinstance(attempt, Mapping):
            continue  # only mapping attempts can declare assumptions
        declared = attempt.get("assumptions")
        if isinstance(declared, list):
            for pos, item in enumerate(declared):
                normalized = _normalize_assumption(item, f"attempt-{index}-{pos}")
                if normalized is not None:
                    assumptions.append(normalized)
            continue  # explicit list wins over single-form keys on the same attempt
        single = attempt.get("assumption")
        if isinstance(single, str) and single.strip():
            assumptions.append({"assumptionId": _attempt_assumption_id(attempt, index), "text": single.strip()})
    return {"grounded": grounded, "assumptions": assumptions}


def _evidence_texts(evidence: object) -> dict[str, str]:
    """Admitted id -> content text for quote-containment checks (missing content stays absent)."""
    texts: dict[str, str] = {}
    items = list(evidence) if isinstance(evidence, (list, tuple)) else []
    for record in items:
        eid = _field(record, "evidence_id", "")
        content = _field(record, "content", "")
        if isinstance(eid, str) and eid and isinstance(content, str) and content:
            texts[eid] = content
    return texts


def format_grounded_block(record: object) -> str:
    """One compact block: evidence id plus complete verbatim content plus hash/provenance."""
    eid = _field(record, "evidence_id", "")
    content = _field(record, "content", "")
    if not isinstance(content, str):
        content = str(content)
    # Verbatim: never collapse whitespace — numbers[].quote must match stored bytes exactly.
    return (
        f"[GROUNDED evidence_id={eid}] "
        f"content_hash={_field(record, 'content_hash')} provenance={_field(record, 'provenance')} "
        f"content={content}"
    )


def _check_number_entry(entry: object, refs: set[str], texts: dict[str, str] | None = None) -> None:
    """One numbers[] entry: non-empty value, evidenceId inside refs, verbatim quote from stored bytes."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"analysis numbers[] entries must be mappings, got {entry!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    value: object = entry.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("analysis numbers[] value must be a non-empty string")
    evidence_id: object = entry.get("evidenceId")
    # The number's evidence must itself be cited, so the quote provenance is checkable.
    if evidence_id not in refs:
        raise ValueError(f"analysis numbers[] evidenceId {evidence_id!r} not in evidenceRefs")
    quote: object = entry.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("analysis numbers[] quote must be a non-empty string")
    # Verbatim: the quote must reproduce stored bytes, never a paraphrase.
    if texts is not None and isinstance(evidence_id, str):
        source = texts.get(evidence_id)
        if source is not None and quote.strip() not in source:
            raise ValueError(f"analysis numbers[] quote for {evidence_id!r} not found in evidence content")


def _check_assumption_entries(entries: list[object]) -> None:
    """Assumption ids non-empty and unique within one analysis; text non-empty."""
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"analysis assumptions[] entries must be mappings, got {entry!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        aid: object = entry.get("assumptionId")
        if not isinstance(aid, str) or not aid.strip():
            raise ValueError("analysis assumptions[] assumptionId must be a non-empty string")
        # Duplicates would let two scenarios share one id, so fail closed.
        if aid in seen:
            raise ValueError(f"analysis assumptions[] duplicate assumptionId {aid!r}")
        seen.add(aid)
        text: object = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("analysis assumptions[] text must be a non-empty string")


def check_grounded_analysis(
    analysis: dict[str, object], evidence_ids: set[str], evidence: object = None
) -> dict[str, object]:
    """Validate one v2 analysis against admitted ids; return it unchanged or raise ValueError."""
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    allowed: set[str] = set()
    try:
        candidates: object = evidence_ids or []
        for eid in candidates if isinstance(candidates, (list, set, tuple)) else []:
            if isinstance(eid, str):
                allowed.add(eid)
    except TypeError:
        pass  # non-iterable ids mean nothing is admitted; every ref below then fails
    raw_refs: object = analysis.get("evidenceRefs", [])
    refs: list[object] = []
    if raw_refs is None:
        pass  # absent means "cites nothing", still valid
    elif isinstance(raw_refs, list):
        refs = raw_refs
    else:
        raise ValueError("analysis evidenceRefs must be a list of evidence ids")
    ref_set: set[str] = set()
    for ref in refs:
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"analysis evidenceRefs must be non-empty strings, got {ref!r}")
        # Fail closed: any ref outside the admitted set rejects the whole analysis.
        if ref not in allowed:
            raise ValueError(f"analysis cites unknown evidence id {ref!r}")
        ref_set.add(ref)
    texts = _evidence_texts(evidence) if evidence is not None else None
    raw_numbers: object = analysis.get("numbers", [])
    numbers: list[object] = []
    if raw_numbers is None:
        pass
    elif isinstance(raw_numbers, list):
        numbers = raw_numbers
    else:
        raise ValueError("analysis numbers must be a list")
    for entry in numbers:
        _check_number_entry(entry, ref_set, texts)
    raw_assumptions: object = analysis.get("assumptions", [])
    assumptions: list[object] = []
    if raw_assumptions is None:
        pass
    elif isinstance(raw_assumptions, list):
        assumptions = raw_assumptions
    else:
        raise ValueError("analysis assumptions must be a list")
    _check_assumption_entries(assumptions)
    return analysis


_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def parse_percent(text: object) -> float | None:
    """First percent number in text ("12.5%" -> 12.5); None when absent or unparseable."""
    if not isinstance(text, str):
        return None
    match = _PERCENT_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def attach_scenario_impact(analysis: dict[str, object]) -> dict[str, object]:
    """Attach computed_impact when the first number + first assumption both parse as percents.

    Fail-open by design: unparseable or missing pairs return the analysis
    unchanged (the drop gate already ran; math must never reject reasoning).
    """
    if not isinstance(analysis, Mapping):
        return analysis
    numbers = analysis.get("numbers")
    assumptions = analysis.get("assumptions")
    if not isinstance(numbers, list) or not numbers or not isinstance(assumptions, list) or not assumptions:
        return analysis
    first_num = numbers[0]
    first_asm = assumptions[0]
    if not isinstance(first_num, Mapping) or not isinstance(first_asm, Mapping):
        return analysis
    exposure = parse_percent(first_num.get("value"))
    haircut = parse_percent(first_asm.get("text"))
    if exposure is None or haircut is None:
        return analysis
    eid = first_num.get("evidenceId")
    aid = first_asm.get("assumptionId")
    try:
        impact = compute_scenario_impact(exposure, haircut)
    except ValueError:
        return analysis
    out = dict(analysis)
    out["computed_impact"] = {
        "exposure_pct": exposure,
        "haircut_pct": haircut,
        "impact_pct": impact,
        "exposure_evidence": eid if isinstance(eid, str) else None,
        "assumption_id": aid if isinstance(aid, str) else None,
    }
    return out


def compute_scenario_impact(exposure: float, haircut_pct: float) -> float:
    """Pure scenario math (code computes; the model only states the formula in words)."""
    for name, value in (("exposure", exposure), ("haircut_pct", haircut_pct)):
        # bool is an int subclass but never valid money input — reject before it becomes 0/1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return exposure * haircut_pct / 100.0
