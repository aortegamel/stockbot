"""Filesystem repository: the ownership boundary for thesis YAML/journal/trigger writes."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, NoReturn

import yaml

from app.thesis import models
from app.thesis.models import (
    SCHEMA_VERSION,
    Checkpoint,
    EvidenceRef,
    ExpressionRequirement,
    HistoricalStateUnavailable,
    JSONValue,
    QuestionStatus,
    Thesis,
    ThesisClaim,
    ThesisMemory,
    ThesisQuestion,
    ThesisState,
    ThesisStateSnapshot,
    TradeExpression,
    Trigger,
    TriggerStatus,
    WatchRule,
    new_claim_id,
    new_evidence_id,
    new_expression_id,
    new_journal_id,
    new_memory_id,
    new_question_id,
    new_requirement_id,
    new_thesis_id,
    new_trigger_id,
    require_watch_targets,
    slugify,
)
from app.thesis.yaml import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    load_raw_yaml,
    load_yaml,
    thesis_lock,
)

STATE_FILES = ("thesis.yaml", "state.yaml", "watch.yaml", "questions.yaml", "memory.yaml", "checkpoint.yaml")

# Evidence refs carry canonical refs + summaries only. Anything else smuggles
# bodies or model-invented provenance (URLs, publication/retrieval metadata).
_FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "body",
        "content",
        "filing_body",
        "full_text",
        "document_text",
        "url",
        "urls",
        "link",
        "links",
        "source_url",
        "provenance",
        "publication",
        "published_at",
        "retrieved_at",
        "retrieval",
        "origin",
    }
)


def _reject_hostile_summary(summary: object, where: str) -> None:
    """Scan-on-write gate for Pi-authored prompt-bound text (evidence/journals)."""
    from app.security.prompt_injection import (
        assess,  # local: keep thesis import graph acyclic
    )

    text = summary if isinstance(summary, str) else ("" if summary is None else str(summary))
    found = assess(text)
    if found.verdict in ("BLOCK", "QUARANTINE"):
        raise ValueError(f"{where}: hostile text rejected ({found.verdict} {','.join(found.matched_rules)})")


def _reject_hostile_journal(title: object, body: object, where: str) -> None:
    _reject_hostile_summary(title, f"{where} journal title")
    _reject_hostile_summary(body, f"{where} journal body")


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_name(trigger_or_entry_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", trigger_or_entry_id)


def _trigger_created(trigger: Trigger) -> str:
    return trigger.created_at


def _best_effort_thesis_id(thesis_file: Path) -> str | None:
    """Read a ``thesis_id`` from an unloadable thesis.yaml (None when unreadable)."""
    try:
        data = yaml.safe_load(thesis_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if isinstance(data, dict) and isinstance(data.get("thesis_id"), str) and data["thesis_id"]:
        return data["thesis_id"]
    return None


def _journal_front_matter(entry_id: str, thesis_id: str, created: str, d: Mapping[str, object]) -> str:
    """Journal front matter; ``known_at`` persists when the entry carries it (PIT gate)."""
    known = f"known_at: {d.get('known_at')}\n" if d.get("known_at") else ""
    return (
        f"---\nentry_id: {entry_id}\nthesis_id: {thesis_id}\ncreated_at: {created}\n"
        f"{known}run_id: {d.get('run_id', '')}\ntrigger_id: {d.get('trigger_id', '')}\n---\n"
    )


def _coerce_claim(item: ThesisClaim | str | Mapping[str, object], _path: str = "<create>") -> ThesisClaim:
    if isinstance(item, ThesisClaim):
        return item
    if isinstance(item, str):
        return ThesisClaim(claim_id=new_claim_id(), statement=item)
    if isinstance(item, dict):
        d = dict(item)
        d.setdefault("claim_id", new_claim_id())
        d.setdefault("status", "unvalidated")
        return ThesisClaim.from_dict(d, _path)
    raise ValueError(f"{_path}: claim must be str, mapping, or ThesisClaim, got {type(item).__name__}")


def _coerce_expression(item: TradeExpression | Mapping[str, object], _path: str = "<create>") -> TradeExpression:
    if isinstance(item, TradeExpression):
        return item
    if isinstance(item, dict):
        d = dict(item)
        d.setdefault("expression_id", new_expression_id())
        return TradeExpression.from_dict(d, _path)
    raise ValueError(f"{_path}: expression must be a mapping or TradeExpression, got {type(item).__name__}")


def _snapshot_order_key(snap: ThesisStateSnapshot) -> tuple[datetime, int]:
    """History order: effective time, then version (deterministic latest)."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    stamp = _as_dt(snap.effective_at)
    assert stamp is not None
    return (stamp, snap.version)


class _ApplyInputs(NamedTuple):
    """Coerced research-result payload (lists validated as lists, journal as mapping)."""

    claim_updates: list[object]
    expression_updates: list[object]
    trigger_id: str
    evidence_refs: list[object]
    state: object
    questions_add: list[object]
    memories_add: list[object]
    watch_add: list[object]
    questions_answered: list[object]
    journal_entry: Mapping[str, object] | None


def _coerce_apply_list(result: Mapping[str, object], key: str, where: str) -> list[object]:
    """Payload list field: None -> [], non-list raises with the caller's path."""
    tmp = result.get(key)
    if tmp is None:
        return []
    if not isinstance(tmp, list):
        raise ValueError(f"{where}: bad {key} {tmp!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return tmp


def _coerce_trigger_id(result: Mapping[str, object], thesis_dir: Path) -> str:
    """Trigger id scalar gate (empty when absent)."""
    trigger_raw = result.get("trigger_id")
    if trigger_raw is not None and trigger_raw != "" and not isinstance(trigger_raw, str):
        raise ValueError(f"{thesis_dir}: bad trigger_id {trigger_raw!r}")
    return trigger_raw if isinstance(trigger_raw, str) else ""


def _coerce_journal_entry(result: Mapping[str, object], thesis_dir: Path) -> Mapping[str, object] | None:
    """Journal mapping gate (None when absent)."""
    journal_raw = result.get("journal_entry")
    if journal_raw is not None and not isinstance(journal_raw, dict):
        raise ValueError(f"{thesis_dir}: bad journal_entry {journal_raw!r}")
    return journal_raw if isinstance(journal_raw, dict) else None


def _coerce_apply_inputs(
    result: Mapping[str, object],
    thesis_dir: Path,
) -> _ApplyInputs:
    """Validate-then-extract the 10 payload phases (no lock, no writes)."""
    return _ApplyInputs(
        claim_updates=_coerce_apply_list(result, "claim_updates", f"{thesis_dir}/thesis.yaml"),
        expression_updates=_coerce_apply_list(result, "expression_updates", f"{thesis_dir}/thesis.yaml"),
        trigger_id=_coerce_trigger_id(result, thesis_dir),
        evidence_refs=_coerce_apply_list(result, "evidence_refs", f"{thesis_dir}/evidence"),
        state=result.get("state"),
        questions_add=_coerce_apply_list(result, "questions_add", f"{thesis_dir}"),
        memories_add=_coerce_apply_list(result, "memories_add", f"{thesis_dir}"),
        watch_add=_coerce_apply_list(result, "watch_add", f"{thesis_dir}"),
        questions_answered=_coerce_apply_list(result, "questions_answered", f"{thesis_dir}"),
        journal_entry=_coerce_journal_entry(result, thesis_dir),
    )


def _patch_entry_id(entry: object, key: str, thesis_dir: Path, noun: str) -> str:
    """Validated patch entry id (mapping + text id gates)."""
    if not isinstance(entry, dict) or not entry.get(key):
        raise ValueError(f"{thesis_dir}/thesis.yaml: bad {noun} update {entry!r}")
    eid = entry.get(key)
    if not isinstance(eid, str):
        raise ValueError(f"{thesis_dir}/thesis.yaml: bad {noun} update {entry!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    return eid


def _patch_member_gate(eid: str, member_ids: set[str], thesis_dir: Path, thesis_id: str, noun: str) -> None:
    """Id-membership gate for one patch entry."""
    if eid not in member_ids:
        raise ValueError(f"{thesis_dir}/thesis.yaml: {noun} {eid!r} does not belong to thesis {thesis_id!r}")


def _apply_claim_patch(
    updates: list[object],
    member_ids: set[str],
    thesis_dir: Path,
    thesis_id: str,
) -> dict[str, dict[str, object]]:
    """Claim status patch: id membership + known statuses (raises on foreign ids)."""
    patch: dict[str, dict[str, object]] = {}
    for c in updates:
        if not isinstance(c, dict):
            raise ValueError(f"{thesis_dir}/thesis.yaml: bad claim update {c!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        cid = _patch_entry_id(c, "claim_id", thesis_dir, "claim")
        _patch_member_gate(cid, member_ids, thesis_dir, thesis_id, "claim")
        if c.get("status") is not None and c.get("status") not in {e.value for e in models.ClaimStatus}:
            raise ValueError(f"{thesis_dir}/thesis.yaml: bad claim status {c.get('status')!r}")
        patch[cid] = c
    return patch


def _apply_expression_patch(
    updates: list[object],
    member_ids: set[str],
    thesis_dir: Path,
    thesis_id: str,
) -> dict[str, dict[str, object]]:
    """Expression status patch: id membership + known statuses."""
    patch: dict[str, dict[str, object]] = {}
    for e in updates:
        if not isinstance(e, dict):
            raise ValueError(f"{thesis_dir}/thesis.yaml: bad expression update {e!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        eid = _patch_entry_id(e, "expression_id", thesis_dir, "expression")
        _patch_member_gate(eid, member_ids, thesis_dir, thesis_id, "expression")
        if e.get("status") is not None and e.get("status") not in {e2.value for e2 in models.ExpressionStatus}:
            raise ValueError(f"{thesis_dir}/thesis.yaml: bad expression status {e.get('status')!r}")
        patch[eid] = e
    return patch


def _snapshot_member_ids(entries: object, key: str) -> set[str]:
    """Id set of snapshot claim/expression dicts (non-dicts ignored)."""
    if not isinstance(entries, list):
        return set()
    return {str(c.get(key)) for c in entries if isinstance(c, dict) and isinstance(c.get(key), str)}


def _pit_known_at(raw: object) -> str:
    """Known-at string of one evidence file (empty when absent)."""
    known = raw.get("known_at") if isinstance(raw, dict) else None
    return known if isinstance(known, str) else (str(known) if known is not None else "")


def _pit_file_owned_ref(raw: object, thesis_id: str) -> str | None:
    """Owned canonical ref of one evidence mapping; None when foreign/ref-less."""
    if not isinstance(raw, dict) or raw.get("thesis_id") != thesis_id:
        return None
    ref = raw.get("canonical_ref")
    return ref if isinstance(ref, str) and ref else None


def _pit_file_fresh_ref(raw: object, dt_cut: datetime) -> str | None:
    """Owned ref only when its known_at is at or before the cut."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    ref = raw.get("canonical_ref") if isinstance(raw, dict) else None
    if not isinstance(ref, str) or not ref:
        return None
    known_str = _pit_known_at(raw)
    dt_known = _as_dt(known_str) if known_str else None
    return ref if dt_known is not None and dt_known <= dt_cut else None


def _pit_file_ref(path: Path, thesis_id: str, dt_cut: datetime) -> str | None:
    """Canonical ref of one evidence file when known at or before the cut."""
    try:
        raw = load_raw_yaml(path)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if _pit_file_owned_ref(raw, thesis_id) is None:
        return None
    return _pit_file_fresh_ref(raw, dt_cut)


def _pit_allowed_refs(
    thesis_dir: Path,
    thesis_id: str,
    dt_cut: datetime | None,
) -> set[str]:
    """Stored canonical refs known at or before the payload cut (PIT pool)."""

    pit_refs: set[str] = set()
    if dt_cut is None:
        return pit_refs
    evdir = thesis_dir / "evidence"
    if not evdir.is_dir():
        return pit_refs
    for f in sorted(evdir.glob("*.yaml")):
        ref = _pit_file_ref(f, thesis_id, dt_cut)
        if ref is not None:
            pit_refs.add(ref)
    return pit_refs


def _trigger_allowed_refs(raw_trigger: dict[str, JSONValue]) -> set[str]:
    """Trigger's own canonical refs (string entries only)."""
    owned = raw_trigger.get("canonical_refs", [])
    return {c for c in owned if isinstance(c, str)} if isinstance(owned, list) else set()


def _payload_cut(
    journal_entry: Mapping[str, object] | None,
    raw_trigger: dict[str, JSONValue],
) -> str | None:
    """Payload time: journal known_at, else the trigger's creation time."""
    payload_known = journal_entry.get("known_at") if isinstance(journal_entry, dict) else None
    if not payload_known:
        payload_known = raw_trigger.get("created_at")
    return (
        payload_known if isinstance(payload_known, str) else (str(payload_known) if payload_known is not None else None)
    )


def _check_evidence_provenance(
    evidence_refs: list[object],
    allowed: set[str] | None,
    thesis_dir: Path,
    trigger_id: str,
) -> None:
    """Forbidden-keys + allowed-set gate (runs before any write)."""
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            raise ValueError(f"{thesis_dir}/evidence: evidence ref must be a mapping, got {type(ref).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        smuggled = [k for k in _FORBIDDEN_EVIDENCE_KEYS if ref.get(k)]
        if smuggled:
            raise ValueError(f"{thesis_dir}/evidence: evidence refs must not carry bodies/provenance (got {smuggled})")
        if allowed is not None and ref.get("canonical_ref") not in allowed:
            raise ValueError(
                f"{thesis_dir}/evidence: foreign canonical_ref {ref.get('canonical_ref')!r}"
                f" (trigger {trigger_id!r}); refusing writeback"
            )


class _ApplyFold(NamedTuple):
    """Fold flags for one research commit (snapshot vs live decided by caller)."""

    claim_or_expr: bool
    state_changed: bool
    questions_added: bool
    memories_added: bool
    watch_added: bool
    questions_answered_updated: bool

    def dirty(self, patched: bool = False) -> bool:
        """Any mutable section changed (patch flag supplied by caller)."""
        return bool(
            patched
            or self.claim_or_expr
            or self.state_changed
            or self.questions_added
            or self.memories_added
            or self.watch_added
            or self.questions_answered_updated
        )


def _merge_patched_entry(
    entry: object,
    key: str,
    patch: dict[str, dict[str, object]],
) -> object:
    """One claim/expression dict merged with its patch (identity when unpatched)."""
    if not isinstance(entry, dict):
        return entry
    found = entry.get(key)
    if not isinstance(found, str):
        return entry
    extra = patch.get(found)
    return {**entry, **extra} if extra is not None else entry


def _fold_claim_expression(
    thesis_dict: Mapping[str, object],
    claim_patch: dict[str, dict[str, object]],
    expr_patch: dict[str, dict[str, object]],
    thesis_id: str,
    eff: str,
    where: str,
) -> dict[str, JSONValue]:
    """Status patch fold shared by snapshot dicts and live Thesis objects."""
    patched: dict[str, object] = dict(thesis_dict)
    claims = patched.get("claims", [])
    patched["claims"] = [
        _merge_patched_entry(c, "claim_id", claim_patch) for c in (claims if isinstance(claims, list) else [])
    ]
    exprs = patched.get("expressions", [])
    patched["expressions"] = [
        _merge_patched_entry(e, "expression_id", expr_patch) for e in (exprs if isinstance(exprs, list) else [])
    ]
    patched["thesis_id"] = thesis_id
    patched["updated_at"] = eff
    return Thesis.from_dict(patched, where).to_dict()


def _fold_state_candidate(
    state_raw: object,
    thesis_id: str,
    thesis_dir: Path,
) -> dict[str, JSONValue]:
    """Validated state dict from a payload state (mapping or ThesisState)."""
    if isinstance(state_raw, dict):
        sdata: dict[str, object] = dict(state_raw)
    elif isinstance(state_raw, ThesisState):
        sdata = dict(state_raw.to_dict())
    else:
        raise ValueError(f"{thesis_dir}/state.yaml: bad state {state_raw!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    sdata["thesis_id"] = thesis_id
    return ThesisState.from_dict(sdata, str(thesis_dir / "state.yaml")).to_dict()


def _section_version(section: Mapping[str, object], schema_default: int) -> JSONValue:
    """Validated schema_version for a copied section (default when absent/invalid)."""
    raw = section.get("schema_version", schema_default)
    return raw if isinstance(raw, (str, int, float, bool, list, dict)) or raw is None else schema_default


def _section_rows(section: Mapping[str, object], key: str) -> list[JSONValue]:
    """Owned copy of one section row list (non-list normalizes to empty)."""
    raw = section.get(key, [])
    if not isinstance(raw, list):
        return []
    return [dict(x) if isinstance(x, dict) else x for x in raw]


def _copy_section_list(
    section: Mapping[str, object], key: str, thesis_id: str, schema_default: int
) -> tuple[dict[str, JSONValue], list[JSONValue]]:
    """Owned copy of a questions/memory/watch section + its live list."""
    raw: dict[str, JSONValue] = {
        "schema_version": _section_version(section, schema_default),
        "thesis_id": thesis_id,
        key: _section_rows(section, key),
    }
    entries = raw.get(key, [])
    assert isinstance(entries, list)
    return raw, entries


def _append_questions(
    questions_raw: dict[str, JSONValue],
    additions: list[object],
    thesis_dir: Path,
) -> bool:
    """Append questions by id (skip known); True when anything landed."""
    entries = questions_raw.get("questions", [])
    assert isinstance(entries, list)
    known = {q.get("question_id") for q in entries if isinstance(q, dict)}
    added = False
    for q in additions:
        if not isinstance(q, dict):
            raise ValueError(f"{thesis_dir}/questions.yaml: bad question {q!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        d = dict(q)
        d.setdefault("question_id", new_question_id())
        qobj = ThesisQuestion.from_dict(d, str(thesis_dir / "questions.yaml"))
        if qobj.question_id in known:
            continue
        entries.append(qobj.to_dict())
        known.add(qobj.question_id)
        added = True
    return added


def _append_memories(
    memory_raw: dict[str, JSONValue],
    additions: list[object],
    thesis_dir: Path,
    eff: str,
) -> bool:
    """Append memories by id (skip known); True when anything landed."""
    entries = memory_raw.get("memories", [])
    assert isinstance(entries, list)
    known = {m.get("memory_id") for m in entries if isinstance(m, dict)}
    added = False
    for m in additions:
        if not isinstance(m, dict):
            raise ValueError(f"{thesis_dir}/memory.yaml: bad memory {m!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        d = dict(m)
        d.setdefault("memory_id", new_memory_id())
        d.setdefault("created_at", eff)
        mobj = ThesisMemory.from_dict(d, str(thesis_dir / "memory.yaml"))
        if mobj.memory_id in known:
            continue
        entries.append(mobj.to_dict())
        known.add(mobj.memory_id)
        added = True
    return added


def _watch_rule_refs_gate(robj: WatchRule, claim_ids: set[str], expr_ids: set[str], thesis_dir: Path) -> None:
    """Claim/expression cross-ref gates for one parsed rule."""
    for cid in robj.claim_ids:
        if cid not in claim_ids:
            raise ValueError(f"{thesis_dir}/watch.yaml: rule references absent claim {cid!r}")
    for eid in robj.expression_ids:
        if eid not in expr_ids:
            raise ValueError(f"{thesis_dir}/watch.yaml: rule references absent expression {eid!r}")


def _fold_single_watch_rule(
    entries: list[JSONValue],
    known: set[JSONValue | None],
    rule: object,
    claim_ids: set[str],
    expr_ids: set[str],
    thesis_dir: Path,
) -> bool:
    """One watch rule: cross-ref gates + append by id (False when already known)."""
    if not isinstance(rule, dict):
        raise ValueError(f"{thesis_dir}/watch.yaml: bad rule {rule!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    d = dict(rule)
    d.setdefault("rule_id", f"rule:{__import__('uuid').uuid4()}")
    robj = WatchRule.from_dict(d, str(thesis_dir / "watch.yaml"))
    _watch_rule_refs_gate(robj, claim_ids, expr_ids, thesis_dir)
    require_watch_targets(robj, str(thesis_dir / "watch.yaml"))
    if robj.rule_id in known:
        return False
    entries.append(robj.to_dict())
    known.add(robj.rule_id)
    return True


def _append_watch_rules(
    watch_raw: dict[str, JSONValue],
    additions: list[object],
    thesis_d: Mapping[str, object],
    thesis_dir: Path,
) -> bool:
    """Append watch rules by id with claim/expression cross-ref checks."""
    entries = watch_raw.get("rules", [])
    assert isinstance(entries, list)
    known = {r.get("rule_id") for r in entries if isinstance(r, dict)}
    claim_ids = _snapshot_member_ids(thesis_d.get("claims", []), "claim_id")
    expr_ids = _snapshot_member_ids(thesis_d.get("expressions", []), "expression_id")
    # live Thesis dicts carry ids under the same keys; snapshot helper covers both
    added = False
    for r in additions:
        added = _fold_single_watch_rule(entries, known, r, claim_ids, expr_ids, thesis_dir) or added
    return added


def _answer_target(
    by_id: dict[object, dict[str, JSONValue]], answer: dict[object, object], qpath: Path
) -> dict[str, JSONValue]:
    """Target question for one answer (membership gate)."""
    if not isinstance(answer.get("question_id"), str) or answer.get("question_id") not in by_id:
        raise ValueError(f"{qpath}: answer names absent question {answer!r}")
    target = by_id.get(answer.get("question_id"))
    if target is None:
        raise ValueError(f"{qpath}: answer names absent question {answer!r}")
    return target


def _answer_text(answer: dict[object, object], qpath: Path) -> str:
    """Non-empty answer text gate."""
    text = answer.get("answer")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{qpath}: answer for {answer.get('question_id')!r} must be a non-empty string")
    return text


def _fold_single_answer(
    by_id: dict[object, dict[str, JSONValue]],
    answer: object,
    qpath: Path,
) -> None:
    """One answer -> target question (membership + non-empty text gates)."""
    if not isinstance(answer, dict):
        raise ValueError(f"{qpath}: answer names absent question {answer!r}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    target = _answer_target(by_id, answer, qpath)
    target["status"] = QuestionStatus.ANSWERED.value
    target["answer"] = _answer_text(answer, qpath)


def _validate_section_questions(entries: object, qpath: Path) -> None:
    """Every entry is a mapping and passes the ThesisQuestion model gate."""
    assert isinstance(entries, list)
    for q in entries:
        if not isinstance(q, dict):
            raise ValueError(f"{qpath}: question must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        ThesisQuestion.from_dict(q, str(qpath))


def _mark_questions_answered(
    questions_raw: dict[str, JSONValue],
    answers: list[object],
    thesis_dir: Path,
) -> None:
    """Fold answers into a questions section in place (validated inline)."""
    qpath = thesis_dir / "questions.yaml"
    entries = questions_raw.get("questions", [])
    if not isinstance(entries, list):
        raise ValueError(f"{qpath}: 'questions' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
    by_id: dict[object, dict[str, JSONValue]] = {q.get("question_id"): q for q in entries if isinstance(q, dict)}
    for a in answers:
        _fold_single_answer(by_id, a, qpath)
    _validate_section_questions(entries, qpath)


def _append_evidence_refs(
    root: Path,
    thesis_dir: Path,
    thesis_id: str,
    evidence_refs: list[object],
) -> int:
    """Evidence side effect shared by both paths: one file per ref, skip existing ids."""
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            raise ValueError(f"{thesis_dir}/evidence: evidence ref must be a mapping, got {type(ref).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        _reject_hostile_summary(ref.get("summary", ""), f"{thesis_dir}/evidence")
        d = dict(ref)
        d.setdefault("evidence_id", new_evidence_id())
        d["thesis_id"] = thesis_id
        ev = EvidenceRef.from_dict(d, str(thesis_dir / "evidence"))
        dest = thesis_dir / "evidence" / f"{_safe_name(ev.evidence_id)}.yaml"
        if dest.is_file():
            continue
        atomic_write_yaml(dest, {"schema_version": SCHEMA_VERSION, **ev.to_dict()}, root)
    return len(evidence_refs)


def _find_journal_file(thesis_dir: Path, entry_id: str) -> Path | None:
    """Journal file naming this entry id; None on first write."""
    for f in (thesis_dir / "journal").glob("*.md"):
        try:
            if f"entry_id: {entry_id}" in f.read_text(encoding="utf-8").split("---")[1]:
                return f
        except OSError, IndexError:
            continue
    return None


def _persist_journal_file(
    root: Path,
    thesis_dir: Path,
    thesis_id: str,
    entry_id: str,
    jd: dict[str, object],
) -> str:
    """Hostile-gated journal persist (title/body defaults preserved)."""
    title = jd.get("title", "Research run")
    title_s = title if isinstance(title, str) else str(title)
    body = jd.get("body", jd.get("summary", ""))
    body_s = body if isinstance(body, str) else ("" if body is None else str(body))
    _reject_hostile_journal(title_s, body_s, f"{thesis_dir}/journal")
    dest = thesis_dir / "journal" / f"{_safe_name(entry_id)}.md"
    atomic_write_text(
        dest,
        _journal_front_matter(entry_id, thesis_id, _utcnow(), jd) + f"# {title_s}\n\n{body_s}\n",
        root,
    )
    return str(dest)


def _write_journal_entry(
    root: Path,
    thesis_dir: Path,
    thesis_id: str,
    journal_raw: Mapping[str, object] | None,
    run_id: str,
    trigger_id: str,
) -> str | None:
    """Journal side effect shared by both paths: idempotent by entry id."""
    if journal_raw is None:
        return None
    jd = dict(journal_raw)
    jd.setdefault("run_id", run_id)
    if trigger_id:
        jd.setdefault("trigger_id", trigger_id)
    raw_id = jd.get("entry_id") or jd.get("journal_id")
    entry_id = raw_id if isinstance(raw_id, str) and raw_id else new_journal_id()
    existing = _find_journal_file(thesis_dir, entry_id)
    if existing is not None:
        return str(existing)
    return _persist_journal_file(root, thesis_dir, thesis_id, entry_id, jd)


def _persist_trigger_processed(
    repo: ThesisRepository,
    tpath: Path,
    thesis_id: str,
    run_id: str,
) -> str:
    """Reload + status/run fold + persist for one trigger file."""
    raw_trigger = load_raw_yaml(tpath)
    repo._check_file_owner(raw_trigger, str(tpath), thesis_id)
    raw_trigger["status"] = TriggerStatus.PROCESSED.value
    raw_trigger["processed_at"] = _utcnow()
    if run_id:
        raw_trigger["run_id"] = run_id
    updated = Trigger.from_dict(raw_trigger, str(tpath))
    atomic_write_yaml(tpath, {"schema_version": SCHEMA_VERSION, **updated.to_dict()}, repo.root)
    return updated.trigger_id


def _mark_trigger_processed(
    repo: ThesisRepository,
    thesis_dir: Path,
    thesis_id: str,
    trigger_id: str,
    run_id: str,
    tpath: Path | None,
    raw_trigger: dict[str, JSONValue] | None,
) -> str | None:
    """Trigger processed side effect shared by both paths (never deletes)."""
    if not trigger_id:
        return None
    if raw_trigger is None or tpath is None:
        tpath, raw_trigger = repo._load_trigger_raw(thesis_dir, thesis_id, trigger_id)
    if not tpath.is_file():
        return None
    return _persist_trigger_processed(repo, tpath, thesis_id, run_id)


def _resolve_gate_allowed(
    thesis_dir: Path,
    thesis_id: str,
    journal_entry: Mapping[str, object] | None,
    raw_trigger: dict[str, JSONValue],
    allowed_refs: set[str] | None,
) -> set[str] | None:
    """Allowed-set for one commit: explicit set, else trigger refs + PIT pool."""
    from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

    if allowed_refs is not None:
        return set(allowed_refs)
    cut_str = _payload_cut(journal_entry, raw_trigger)
    dt_cut = _as_dt(cut_str) if cut_str else None
    return _trigger_allowed_refs(raw_trigger) | _pit_allowed_refs(thesis_dir, thesis_id, dt_cut)


class ThesisRepository:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(os.path.realpath(root))

    # -- internal helpers -------------------------------------------------

    def _classify_child(self, child: Path) -> tuple[str, object]:
        """One child -> (kind, payload): ok/symlink-dir/load-failure/skip."""
        if child.is_symlink():
            if child.is_file() and not child.is_dir():
                return "skip", child  # regular-file symlink: ignore like any non-dir file
            return "symlink-dir", child
        thesis_file = child / "thesis.yaml"
        if not child.is_dir() or not thesis_file.is_file():
            return "skip", child
        try:
            return "ok", load_yaml(thesis_file, Thesis)
        except ValueError as exc:
            return "load-failure", str(exc)

    def _register_loaded(
        self,
        seen: dict[str, Path],
        loaded: list[tuple[Path, Thesis]],
        child: Path,
        thesis: Thesis,
    ) -> None:
        """Duplicate ids stay loud (single decision point)."""
        if thesis.thesis_id in seen:
            raise ValueError(f"duplicate thesis ID {thesis.thesis_id!r} in {seen[thesis.thesis_id]} and {child}")
        seen[thesis.thesis_id] = child
        loaded.append((child, thesis))

    def _fold_slug_match(
        self,
        found: dict[str, Path],
        bad_dirs: dict[str, str],
        bad_ids: dict[str, str],
        child: Path,
        thesis: Thesis,
    ) -> None:
        """Slug match -> healthy entry, else quarantine by dir + id."""
        if child.name != thesis.slug:
            reason = f"{child}: directory name does not match thesis slug {thesis.slug!r}"
            bad_dirs[child.name] = reason
            bad_ids.setdefault(thesis.thesis_id, reason)
            return
        found[thesis.thesis_id] = child

    def _scan_all(self) -> tuple[dict[str, Path], dict[str, str], dict[str, str]]:
        """Healthy id->dir plus quarantine maps (dir name->reason, thesis_id->reason).

        One corrupt thesis.yaml never hides a healthy sibling: per-child load
        failures (malformed YAML, unknown schema, slug/owner mismatch) are
        skipped and recorded. Duplicate thesis IDs stay loud.
        """
        found: dict[str, Path] = {}
        bad_dirs: dict[str, str] = {}
        bad_ids: dict[str, str] = {}
        if not self.root.is_dir():
            return found, bad_dirs, bad_ids
        loaded: list[tuple[Path, Thesis]] = []
        seen: dict[str, Path] = {}
        for child in sorted(self.root.iterdir()):
            kind, payload = self._classify_child(child)
            if kind == "ok":
                assert isinstance(payload, Thesis)
                self._register_loaded(seen, loaded, child, payload)
            elif kind == "symlink-dir":
                bad_dirs[child.name] = f"{child}: symlinked thesis directories are not allowed"
            elif kind == "load-failure":
                assert isinstance(payload, str)
                bad_dirs[child.name] = payload
                tid = _best_effort_thesis_id(child / "thesis.yaml")
                if tid is not None:
                    bad_ids.setdefault(tid, payload)
        for child, thesis in loaded:
            self._fold_slug_match(found, bad_dirs, bad_ids, child, thesis)
        return found, bad_dirs, bad_ids

    def _scan(self) -> dict[str, Path]:
        """Map thesis_id -> thesis dir by scanning validated thesis.yaml files."""
        return self._scan_all()[0]

    def list_quarantine(self) -> dict[str, str]:
        """Map quarantined directory name -> load-failure reason (inspect aid)."""
        return self._scan_all()[1]

    def _unknown_thesis(self, id_or_slug: str, bad_dirs: dict[str, str], bad_ids: dict[str, str]) -> NoReturn:
        if id_or_slug in bad_dirs:
            raise ValueError(bad_dirs[id_or_slug])
        if id_or_slug in bad_ids:
            raise ValueError(bad_ids[id_or_slug])
        if bad_dirs:
            names = ", ".join(sorted(bad_dirs))
            raise ValueError(f"unknown thesis: {id_or_slug!r} (quarantined: {names})")
        raise KeyError(f"unknown thesis: {id_or_slug!r}")

    def dir_for_thesis(self, thesis_id: str) -> Path:
        """Resolve the owning dir for ``thesis_id`` (re-checked every mutation)."""
        found, bad_dirs, bad_ids = self._scan_all()
        if thesis_id not in found:
            # Also allow slug lookup for reads.
            for d in found.values():
                thesis = load_yaml(d / "thesis.yaml", Thesis)
                if thesis.slug == thesis_id:
                    return d
            self._unknown_thesis(thesis_id, bad_dirs, bad_ids)
        return found[thesis_id]

    _dir_for = dir_for_thesis

    def _load_in_dir(self, thesis_dir: Path) -> Thesis:
        thesis = load_yaml(thesis_dir / "thesis.yaml", Thesis)
        if thesis_dir.name != thesis.slug:
            raise ValueError(f"{thesis_dir}: directory name does not match thesis slug {thesis.slug!r}")
        return thesis

    def _check_owner(self, thesis_dir: Path, thesis_id: str) -> Thesis:
        thesis = self._load_in_dir(thesis_dir)
        if thesis.thesis_id != thesis_id:
            raise ValueError(f"{thesis_dir}: thesis_id mismatch: file has {thesis.thesis_id!r}, expected {thesis_id!r}")
        return thesis

    def load_thesis(self, id_or_slug: str) -> Thesis:
        found, bad_dirs, bad_ids = self._scan_all()
        if id_or_slug in found:
            return self._load_in_dir(found[id_or_slug])
        for d in found.values():
            if self._load_in_dir(d).slug == id_or_slug:
                return self._load_in_dir(d)
        self._unknown_thesis(id_or_slug, bad_dirs, bad_ids)

    def list_theses(self) -> list[Thesis]:
        return [self._load_in_dir(d) for _, d in sorted(self._scan().items())]

    def load_state(self, id_or_slug: str) -> ThesisState:
        d = self._dir_for(self._resolve_id(id_or_slug))
        return load_yaml(d / "state.yaml", ThesisState)

    def load_questions(self, id_or_slug: str) -> list[ThesisQuestion]:
        d = self._dir_for(self._resolve_id(id_or_slug))
        raw = load_raw_yaml(d / "questions.yaml")
        _qid = self._resolve_id(id_or_slug)
        self._check_file_owner(raw, str(d / "questions.yaml"), _qid)
        _ql = raw.get("questions", [])
        if not isinstance(_ql, list):
            raise ValueError(f"{d / 'questions.yaml'}: 'questions' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        out: list[ThesisQuestion] = []
        for _q in _ql:
            if not isinstance(_q, dict):
                raise ValueError(f"{d / 'questions.yaml'}: question must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
            out.append(ThesisQuestion.from_dict(_q, str(d / "questions.yaml")))
        return out

    def load_triggers(self, id_or_slug: str, *, include_processed: bool = True) -> list[Trigger]:
        thesis_id = self._resolve_id(id_or_slug)
        d = self._dir_for(thesis_id)
        out: list[Trigger] = []
        inbox = d / "inbox"
        if inbox.is_dir():
            for f in sorted(inbox.glob("*.yaml")):
                raw = load_raw_yaml(f)
                self._check_file_owner(raw, str(f), thesis_id)
                t = Trigger.from_dict(raw, str(f))
                if not include_processed and t.status == TriggerStatus.PROCESSED.value:
                    continue
                out.append(t)
        out.sort(key=_trigger_created)
        return out

    def _resolve_id(self, id_or_slug: str) -> str:
        found, bad_dirs, bad_ids = self._scan_all()
        if id_or_slug in found:
            return id_or_slug
        for tid, d in found.items():
            if self._load_in_dir(d).slug == id_or_slug:
                return tid
        self._unknown_thesis(id_or_slug, bad_dirs, bad_ids)

    @staticmethod
    def _check_file_owner(raw: Mapping[str, object], path: str, thesis_id: str) -> None:
        if raw.get("thesis_id") != thesis_id:
            raise ValueError(f"{path}: thesis_id mismatch: file has {raw.get('thesis_id')!r}, expected {thesis_id!r}")

    @staticmethod
    def _history_dt(raw: object, eff_raw: object) -> datetime | None:
        """Effective datetime of one history file; None when unparseable."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        if not isinstance(raw, dict) or eff_raw is None:
            return None
        return _as_dt(str(eff_raw))

    @staticmethod
    def _history_version(path: Path) -> int | None:
        """Numeric history version; None for non-numeric stems."""
        try:
            return int(path.stem)
        except ValueError:
            return None

    def _history_pair(self, path: Path) -> tuple[datetime, int] | None:
        """(effective_at, version) of one history file; None when unusable."""
        version = self._history_version(path)
        if version is None:
            return None
        try:
            raw = load_raw_yaml(path)
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            return None
        eff_raw = raw.get("effective_at") if isinstance(raw, dict) else None
        dt = self._history_dt(raw, eff_raw)
        return (dt, version) if dt is not None else None

    def _history_pairs(self, hdir: Path) -> list[tuple[datetime, int]]:
        """Usable (effective_at, version) pairs across history files."""
        pairs: list[tuple[datetime, int]] = []
        for f in hdir.glob("*.yaml"):
            pair = self._history_pair(f)
            if pair is not None:
                pairs.append(pair)
        return pairs

    @staticmethod
    def _latest_pair(pairs: list[tuple[datetime, int]]) -> tuple[datetime, int] | None:
        """Latest pair; None when empty (caller takes the live path)."""
        if not pairs:
            return None
        return (max(dt for dt, _ in pairs), max(v for _, v in pairs))

    def _latest_effective_locked(self, thesis_dir: Path) -> tuple[datetime, int] | None:
        """Latest (effective_at, version) in history; None when empty (caller takes the live path)."""
        hdir = thesis_dir / "history"
        if not hdir.is_dir():
            return None
        return self._latest_pair(self._history_pairs(hdir))

    @staticmethod
    def _next_history_version(hdir: Path) -> int:
        """Next version int over numeric history stems."""
        existing = []
        for f in hdir.glob("*.yaml"):
            version = ThesisRepository._history_version(f)
            if version is not None:
                existing.append(version)
        return (max(existing) if existing else 0) + 1

    def _write_snapshot_from_dicts_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        *,
        effective_at: str,
        reason: str,
        run_id: str = "",
        trigger_id: str = "",
        thesis: dict[str, JSONValue],
        state: dict[str, JSONValue],
        questions: dict[str, JSONValue],
        watch: dict[str, JSONValue],
        memory: dict[str, JSONValue],
    ) -> ThesisStateSnapshot:
        """Write one versioned snapshot from caller-supplied dicts; caller must hold thesis_lock."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        if _as_dt(effective_at) is None:
            raise ValueError(f"{thesis_dir}/history: bad effective_at {effective_at!r}")
        hdir = thesis_dir / "history"
        hdir.mkdir(parents=True, exist_ok=True)
        version = self._next_history_version(hdir)
        snap = ThesisStateSnapshot(
            thesis_id=thesis_id,
            version=version,
            effective_at=effective_at,
            recorded_at=_utcnow(),
            reason=reason,
            run_id=run_id or "",
            trigger_id=trigger_id or "",
            thesis=thesis,
            state=state,
            questions=questions,
            watch=watch,
            memory=memory,
        )
        dest = hdir / f"{version:08d}.yaml"
        validated = ThesisStateSnapshot.from_dict(snap.to_dict(), str(dest))
        atomic_write_yaml(dest, validated.to_dict(), self.root)
        return validated

    def _load_snapshot_sections_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
    ) -> tuple[
        dict[str, JSONValue], dict[str, JSONValue], dict[str, JSONValue], dict[str, JSONValue], dict[str, JSONValue]
    ]:
        """Live thesis/state/questions/watch/memory dicts; validates owners."""
        thesis = load_yaml(thesis_dir / "thesis.yaml", Thesis)
        if thesis.thesis_id != thesis_id:
            raise ValueError(f"{thesis_dir}: thesis_id mismatch: file has {thesis.thesis_id!r}, expected {thesis_id!r}")
        state = load_yaml(thesis_dir / "state.yaml", ThesisState)
        qpath, wpath, mpath = (
            str(thesis_dir / "questions.yaml"),
            str(thesis_dir / "watch.yaml"),
            str(thesis_dir / "memory.yaml"),
        )
        questions = load_raw_yaml(thesis_dir / "questions.yaml")
        self._check_file_owner(questions, qpath, thesis_id)
        watch = load_raw_yaml(thesis_dir / "watch.yaml")
        self._check_file_owner(watch, wpath, thesis_id)
        memory = load_raw_yaml(thesis_dir / "memory.yaml")
        self._check_file_owner(memory, mpath, thesis_id)
        return (thesis.to_dict(), state.to_dict(), dict(questions), dict(watch), dict(memory))

    def _migration_basis(
        self,
        thesis_dir: Path,
        reason: str,
        effective_at: str,
        updated_at: str,
    ) -> tuple[str, str]:
        """First-snapshot rewrite: migration reason + thesis clock basis."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        hdir = thesis_dir / "history"
        for f in hdir.glob("*.yaml"):
            if self._history_version(f) is not None:
                return reason, effective_at
        if reason != "thesis_created":
            reason = "history_migration"
            effective_at = updated_at if _as_dt(updated_at) is not None else _utcnow()
        return reason, effective_at

    def _snapshot_state_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        *,
        effective_at: str,
        reason: str,
        run_id: str = "",
        trigger_id: str = "",
    ) -> ThesisStateSnapshot:
        """Write one versioned history snapshot; caller must hold thesis_lock."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        if _as_dt(effective_at) is None:
            raise ValueError(f"{thesis_dir}/history: bad effective_at {effective_at!r}")
        thesis_d, state_d, questions, watch, memory = self._load_snapshot_sections_locked(thesis_dir, thesis_id)
        raw_thesis = load_yaml(thesis_dir / "thesis.yaml", Thesis)
        reason, effective_at = self._migration_basis(thesis_dir, reason, effective_at, raw_thesis.updated_at)
        return self._write_snapshot_from_dicts_locked(
            thesis_dir,
            thesis_id,
            effective_at=effective_at,
            reason=reason,
            run_id=run_id,
            trigger_id=trigger_id,
            thesis=thesis_d,
            state=state_d,
            questions=dict(questions),
            watch=dict(watch),
            memory=dict(memory),
        )

    @staticmethod
    def _eligible_snapshot(
        snaps: list[ThesisStateSnapshot],
        cutoff: datetime,
    ) -> ThesisStateSnapshot | None:
        """Latest snapshot at or before the cutoff; None when all are later."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eligible = [s for s in snaps if (d := _as_dt(s.effective_at)) is not None and d <= cutoff]
        if not eligible:
            return None
        eligible.sort(key=_snapshot_order_key)
        return eligible[-1]

    def _history_snaps(self, thesis_id: str, hdir: Path) -> list[ThesisStateSnapshot]:
        """Parsed history snapshots; raises when the directory holds no files."""
        files: list[Path] = sorted(hdir.glob("*.yaml")) if hdir.is_dir() else []
        if not files:
            raise HistoricalStateUnavailable(f"{hdir}: no state for {thesis_id!r} (no history)")
        return [ThesisStateSnapshot.from_dict(load_raw_yaml(f), str(f)) for f in files]

    def load_state_as_of(self, thesis_id: str, known_at: str) -> ThesisStateSnapshot:
        """Latest snapshot with effective_at <= known_at; fail closed before history."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        cutoff = _as_dt(known_at)
        if cutoff is None:
            raise ValueError(f"<history>: bad known_at {known_at!r}")
        thesis_dir = self._dir_for(thesis_id)
        hdir = thesis_dir / "history"
        try:
            snaps = self._history_snaps(thesis_id, hdir)
        except HistoricalStateUnavailable:
            raise HistoricalStateUnavailable(
                f"{hdir}: no state for {thesis_id!r} as of {known_at!r} (no history)"
            ) from None
        found = self._eligible_snapshot(snaps, cutoff)
        if found is None:
            earliest = min(s.effective_at for s in snaps)
            raise HistoricalStateUnavailable(
                f"{hdir}: no state for {thesis_id!r} as of {known_at!r} (earliest {earliest!r})"
            )
        return found

    @staticmethod
    def _history_version_int(path: Path) -> int | None:
        """History stem int; None keeps non-numeric debug files out."""
        return ThesisRepository._history_version(path)

    def list_state_versions(self, thesis_id: str) -> list[int]:
        """Sorted history version ints (debug/tests only)."""
        thesis_dir = self._dir_for(thesis_id)
        hdir = thesis_dir / "history"
        if not hdir.is_dir():
            return []
        out: list[int] = []
        for f in sorted(hdir.glob("*.yaml")):
            version = self._history_version(f)
            if version is not None:
                out.append(version)
        return sorted(out)

    @staticmethod
    def _copy_entry_list(source: object, path: object, noun: str) -> list[JSONValue]:
        """Owned copy of a dict-entry list (dicts cloned, scalars kept)."""
        if not isinstance(source, list):
            raise ValueError(f"{path}: {noun!r} must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        return list[JSONValue](dict(q) if isinstance(q, dict) else q for q in source)

    @staticmethod
    def _copy_question_list(source: object, path: object) -> list[JSONValue]:
        """Owned copy of a questions list (dicts cloned, scalars kept)."""
        return ThesisRepository._copy_entry_list(source, path, "questions")

    @staticmethod
    def _answer_index(entries: list[object]) -> dict[object, dict[str, JSONValue]]:
        """Id index over question dicts (non-dicts skipped)."""
        return {q.get("question_id"): q for q in entries if isinstance(q, dict)}

    @staticmethod
    def _fold_one_marked(by_id: dict[object, dict[str, JSONValue]], a: dict[str, JSONValue], path: str) -> None:
        """One answer fold: target lookup + status/answer write."""
        target = by_id.get(a.get("question_id"))
        if target is None:
            raise ValueError(f"{path}: question {a.get('question_id')!r} vanished mid-run")
        target["status"] = QuestionStatus.ANSWERED.value
        answer = a.get("answer")
        target["answer"] = answer

    @staticmethod
    def _validate_marked(entries: list[object], path: str) -> None:
        """Every folded entry is a mapping passing the model gate."""
        for q in entries:
            if not isinstance(q, dict):
                raise ValueError(f"{path}: question must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
            ThesisQuestion.from_dict(q, path)

    @staticmethod
    def _mark_answers(
        entries: object,
        answered: list[dict[str, JSONValue]],
        path: str,
    ) -> dict[object, dict[str, JSONValue]]:
        """Fold answers into question dicts in place; returns id index."""
        if not isinstance(entries, list):
            raise ValueError(f"{path}: 'questions' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        by_id = ThesisRepository._answer_index(entries)
        for a in answered:
            ThesisRepository._fold_one_marked(by_id, a, path)
        ThesisRepository._validate_marked(entries, path)
        return by_id

    def _answer_backdated_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        answered: list[dict[str, JSONValue]],
        eff: str,
    ) -> None:
        """As-of copy path: patch snapshot questions, persist snapshot only."""
        path = str(thesis_dir / "questions.yaml")
        base = self.load_state_as_of(thesis_id, eff)
        raw = dict(base.questions)
        raw["questions"] = self._copy_question_list(base.questions.get("questions", []), path)
        entries = raw.get("questions", [])
        self._mark_answers(entries, answered, path)
        self._write_snapshot_from_dicts_locked(
            thesis_dir,
            thesis_id,
            effective_at=eff,
            reason="questions_answered",
            thesis=dict(base.thesis),
            state=dict(base.state),
            questions=raw,
            watch=dict(base.watch),
            memory=dict(base.memory),
        )

    def _answer_live_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        answered: list[dict[str, JSONValue]],
        eff: str,
    ) -> None:
        """Live path: patch questions.yaml, then snapshot."""
        path = str(thesis_dir / "questions.yaml")
        raw = load_raw_yaml(thesis_dir / "questions.yaml")
        self._check_file_owner(raw, path, thesis_id)
        self._mark_answers(raw.get("questions", []), answered, path)
        atomic_write_yaml(thesis_dir / "questions.yaml", raw, self.root)
        self._snapshot_state_locked(thesis_dir, thesis_id, effective_at=eff, reason="questions_answered")

    def answer_questions(
        self, thesis_id: str, answered: list[dict[str, JSONValue]], *, effective_at: str | None = None
    ) -> None:
        """Mark questions answered (validate-then-replace questions.yaml, lock-held)."""
        if not answered:
            return
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eff = effective_at or _utcnow()
        eff_dt = _as_dt(eff)
        if eff_dt is None:
            raise ValueError(f"<answer>: bad effective_at {effective_at!r}")
        thesis_dir = self.dir_for_thesis(thesis_id)
        with thesis_lock(thesis_dir):
            latest = self._latest_effective_locked(thesis_dir)
            if latest is not None and eff_dt < latest[0]:
                self._answer_backdated_locked(thesis_dir, thesis_id, answered, eff)
                return
            self._answer_live_locked(thesis_dir, thesis_id, answered, eff)

    @staticmethod
    def _heal_reason(rule: dict[str, object]) -> None:
        """Default support reason for a newly-disabled rule."""
        if rule.get("support_reason"):
            return
        if rule.get("rule_type") == "new_external_evidence":
            rule["support_reason"] = "no production source for 'new_external_evidence'; never queried"
        else:
            rule["support_reason"] = f"no deterministic monitor backing for {rule.get('rule_type')!r}; never queried"

    @staticmethod
    def _heal_watch_rule(rule: object, path: object, supported: Collection[str]) -> bool:
        """Disable one unsupported rule; True when mutated."""
        if not isinstance(rule, dict):
            raise ValueError(f"{path}: watch rule must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        if rule.get("rule_type") in supported:
            return False
        if not rule.get("enabled") and rule.get("support_status") == "unsupported":
            return False
        rule["enabled"] = False
        rule["support_status"] = "unsupported"
        ThesisRepository._heal_reason(rule)
        return True

    @staticmethod
    def _heal_watch_rules(entries: object, path: object, supported: Collection[str]) -> bool:
        """Heal every rule; True when any rule changed."""
        if not isinstance(entries, list):
            raise ValueError(f"{path}: 'rules' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        changed = False
        for rule in entries:
            changed = ThesisRepository._heal_watch_rule(rule, path, supported) or changed
        return changed

    @staticmethod
    def _copy_rule_list(source: object, path: object) -> list[JSONValue]:
        """Owned copy of a watch rules list (dicts cloned, scalars kept)."""
        return ThesisRepository._copy_entry_list(source, path, "rules")

    def _normalize_backdated_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        eff: str,
        supported: Collection[str],
    ) -> None:
        """As-of copy path: heal snapshot rules, persist snapshot only when changed."""
        path = thesis_dir / "watch.yaml"
        base = self.load_state_as_of(thesis_id, eff)
        raw = dict(base.watch)
        raw["rules"] = self._copy_rule_list(base.watch.get("rules", []), path)
        if self._heal_watch_rules(raw.get("rules", []), path, supported):
            self._write_snapshot_from_dicts_locked(
                thesis_dir,
                thesis_id,
                effective_at=eff,
                reason="watch_normalized",
                thesis=dict(base.thesis),
                state=dict(base.state),
                questions=dict(base.questions),
                watch=raw,
                memory=dict(base.memory),
            )

    def _normalize_live_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        eff: str,
        supported: Collection[str],
    ) -> None:
        """Live path: heal watch.yaml, then snapshot when changed."""
        path = thesis_dir / "watch.yaml"
        raw = load_raw_yaml(path)
        self._check_file_owner(raw, str(path), thesis_id)
        if self._heal_watch_rules(raw.get("rules", []), path, supported):
            atomic_write_yaml(path, raw, self.root)
            self._snapshot_state_locked(thesis_dir, thesis_id, effective_at=eff, reason="watch_normalized")

    def normalize_watch(self, thesis_id: str, *, effective_at: str | None = None) -> None:
        """Heal watch.yaml: rules without a deterministic backing stay disabled/unsupported."""
        from app.thesis.monitor import (  # local: monitor owns the handler table + clock
            SUPPORTED_HANDLERS,
            _as_dt,
        )

        eff = effective_at or _utcnow()
        eff_dt = _as_dt(eff)
        if eff_dt is None:
            raise ValueError(f"<watch>: bad effective_at {effective_at!r}")
        thesis_dir = self.dir_for_thesis(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            latest = self._latest_effective_locked(thesis_dir)
            if latest is not None and eff_dt < latest[0]:
                self._normalize_backdated_locked(thesis_dir, thesis_id, eff, SUPPORTED_HANDLERS)
                return
            self._normalize_live_locked(thesis_dir, thesis_id, eff, SUPPORTED_HANDLERS)

    def load_watch_rules(self, thesis_id: str) -> list[WatchRule]:
        """Read validated watch rules (ID-or-slug lookup)."""
        thesis = self.load_thesis(thesis_id)
        thesis_dir = self.dir_for_thesis(thesis.thesis_id)
        path = str(thesis_dir / "watch.yaml")
        raw = load_raw_yaml(thesis_dir / "watch.yaml")
        self._check_file_owner(raw, path, thesis.thesis_id)
        _rl = raw.get("rules", [])
        if not isinstance(_rl, list):
            raise ValueError(f"{path}: 'rules' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        rules: list[WatchRule] = []
        for _r in _rl:
            if not isinstance(_r, dict):
                raise ValueError(f"{path}: watch rule must be a mapping")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
            rules.append(WatchRule.from_dict(_r, path))
        return rules

    # -- creation ---------------------------------------------------------

    @staticmethod
    def _coerce_create_requirement(
        r: Mapping[str, object] | ExpressionRequirement,
    ) -> ExpressionRequirement:
        """One requirement row -> validated object (id minted for dicts)."""
        if isinstance(r, dict):
            d = dict(r)
            d.setdefault("requirement_id", new_requirement_id())
            return models.ExpressionRequirement.from_dict(d, "<create>")
        if isinstance(r, ExpressionRequirement):
            return r
        raise ValueError(f"<create>: requirement must be a mapping or ExpressionRequirement, got {type(r).__name__}")

    @staticmethod
    def _coerce_create_requirements(
        requirements: Sequence[Mapping[str, object] | ExpressionRequirement],
    ) -> tuple[ExpressionRequirement, ...]:
        """Requirement dicts -> validated objects (ids minted for dicts)."""
        return tuple(ThesisRepository._coerce_create_requirement(r) for r in requirements or [])

    @staticmethod
    def _create_rule_refs_gate(robj: WatchRule, claim_ids: set[str], expr_ids: set[str]) -> None:
        """Claim/expression cross-ref gates for one create rule."""
        for cid in robj.claim_ids:
            if cid not in claim_ids:
                raise ValueError(f"<create>: rule references absent claim {cid!r}")
        for eid in robj.expression_ids:
            if eid not in expr_ids:
                raise ValueError(f"<create>: rule references absent expression {eid!r}")

    @staticmethod
    def _coerce_create_watch_rules(
        watch_rules: Sequence[Mapping[str, object]],
        claim_ids: set[str],
        expr_ids: set[str],
    ) -> list[WatchRule]:
        """Watch rule dicts -> validated objects with claim/expression cross-refs."""
        rule_objs = []
        for r in watch_rules or []:
            robj = WatchRule.from_dict(dict(r), "<create>")
            ThesisRepository._create_rule_refs_gate(robj, claim_ids, expr_ids)
            rule_objs.append(robj)
        return rule_objs

    def _reserve_thesis_slug(self, user_thesis: str) -> str:
        """Shortest free slug from the thesis words (numeric suffix on collision)."""
        words = re.findall(r"[A-Za-z0-9]+", user_thesis)[:8] or ["thesis"]
        try:
            base = slugify("-".join(words))
        except ValueError:
            base = "thesis"
        self.root.mkdir(parents=True, exist_ok=True)
        slug = base
        n = 2
        while (self.root / slug).exists():
            slug = f"{base}-{n}"
            n += 1
        return slug

    def _persist_new_thesis_locked(self, thesis_dir: Path, thesis: Thesis, rule_objs: list[WatchRule]) -> None:
        """First commit: thesis/state/watch/questions/memory/checkpoint + v1 snapshot."""
        thesis_id = thesis.thesis_id
        (thesis_dir / "inbox").mkdir(parents=True)
        (thesis_dir / "evidence").mkdir(parents=True)
        (thesis_dir / "journal").mkdir(parents=True)
        with thesis_lock(thesis_dir):
            atomic_write_yaml(thesis_dir / "thesis.yaml", thesis.to_dict(), self.root)
            atomic_write_yaml(
                thesis_dir / "state.yaml",
                ThesisState(thesis_id=thesis_id, assessment="unresolved").to_dict(),
                self.root,
            )
            atomic_write_yaml(
                thesis_dir / "watch.yaml",
                {"schema_version": SCHEMA_VERSION, "thesis_id": thesis_id, "rules": [r.to_dict() for r in rule_objs]},
                self.root,
            )
            atomic_write_yaml(
                thesis_dir / "questions.yaml",
                {"schema_version": SCHEMA_VERSION, "thesis_id": thesis_id, "questions": []},
                self.root,
            )
            atomic_write_yaml(
                thesis_dir / "memory.yaml",
                {"schema_version": SCHEMA_VERSION, "thesis_id": thesis_id, "memories": []},
                self.root,
            )
            atomic_write_yaml(
                thesis_dir / "checkpoint.yaml",
                Checkpoint(thesis_id=thesis_id).to_dict(),
                self.root,
            )
            self._snapshot_state_locked(thesis_dir, thesis_id, effective_at=thesis.created_at, reason="thesis_created")

    @staticmethod
    def _create_claims(claims: Sequence[str | Mapping[str, object] | ThesisClaim]) -> tuple[ThesisClaim, ...]:
        """Coerced claim rows for a create candidate."""
        return tuple(_coerce_claim(c) for c in (claims or []))

    @staticmethod
    def _create_expressions(
        expressions: Sequence[Mapping[str, object] | TradeExpression],
    ) -> tuple[TradeExpression, ...]:
        """Coerced expression rows for a create candidate."""
        return tuple(_coerce_expression(e) for e in (expressions or []))

    def _build_create_candidate(
        self,
        thesis_id: str,
        user_thesis: str,
        scope: str,
        claims: Sequence[str | Mapping[str, object] | ThesisClaim],
        assumptions: Sequence[str],
        invalidators: Sequence[str],
        unknowns: Sequence[str],
        expressions: Sequence[Mapping[str, object] | TradeExpression],
        requirements: Sequence[Mapping[str, object] | ExpressionRequirement],
        eff: str,
    ) -> Thesis:
        """Coerced + validated thesis candidate (slug filled in by the caller)."""
        thesis = Thesis(
            thesis_id=thesis_id,
            slug="tmp",  # replaced below; Thesis.validate does not check slug shape
            status=models.ThesisStatus.ACTIVE.value,
            created_at=eff,
            updated_at=eff,
            user_thesis=user_thesis.strip(),
            scope=scope or "unknown",
            claims=self._create_claims(claims),
            assumptions=tuple(assumptions or []),
            invalidators=tuple(invalidators or []),
            unknowns=tuple(unknowns or []),
            expressions=self._create_expressions(expressions),
            requirements=self._coerce_create_requirements(requirements),
        )
        thesis.validate("<create>")
        return thesis

    def _with_reserved_slug(self, thesis: Thesis, user_thesis: str) -> Thesis:
        """Candidate with its shortest free slug filled in."""
        slug = self._reserve_thesis_slug(user_thesis)
        return Thesis(
            thesis_id=thesis.thesis_id,
            slug=slug,
            status=thesis.status,
            created_at=thesis.created_at,
            updated_at=thesis.updated_at,
            user_thesis=thesis.user_thesis,
            scope=thesis.scope,
            claims=thesis.claims,
            assumptions=thesis.assumptions,
            invalidators=thesis.invalidators,
            unknowns=thesis.unknowns,
            expressions=thesis.expressions,
            requirements=thesis.requirements,
        )

    @staticmethod
    def _create_clock(effective_at: str | None) -> str:
        """Effective clock for a create (now when unset, gated when bad)."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eff = effective_at or _utcnow()
        if _as_dt(eff) is None:
            raise ValueError(f"<create>: bad effective_at {effective_at!r}")
        return eff

    @staticmethod
    def _require_create_text(user_thesis: str) -> str:
        """Stripped thesis text (non-empty gate)."""
        if not isinstance(user_thesis, str) or not user_thesis.strip():
            raise ValueError("<create>: 'user_thesis' must be a non-empty string")
        return user_thesis.strip()

    def _create_rule_objs(
        self,
        watch_rules: Sequence[Mapping[str, object]],
        thesis: Thesis,
    ) -> list[WatchRule]:
        """Watch rules validated against the candidate's claim/expression ids."""
        return self._coerce_create_watch_rules(
            watch_rules, {c.claim_id for c in thesis.claims}, {e.expression_id for e in thesis.expressions}
        )

    def create_thesis(
        self,
        user_thesis: str,
        scope: str = "unknown",
        claims: Sequence[str | Mapping[str, object] | ThesisClaim] = (),
        assumptions: Sequence[str] = (),
        invalidators: Sequence[str] = (),
        unknowns: Sequence[str] = (),
        expressions: Sequence[Mapping[str, object] | TradeExpression] = (),
        requirements: Sequence[Mapping[str, object] | ExpressionRequirement] = (),
        watch_rules: Sequence[Mapping[str, object]] = (),
        *,
        effective_at: str | None = None,
    ) -> Thesis:
        text = self._require_create_text(user_thesis)
        eff = self._create_clock(effective_at)
        thesis_id = new_thesis_id()
        thesis = self._build_create_candidate(
            thesis_id, text, scope, claims, assumptions, invalidators, unknowns, expressions, requirements, eff
        )
        rule_objs = self._create_rule_objs(watch_rules, thesis)
        thesis = self._with_reserved_slug(thesis, text)
        self._persist_new_thesis_locked(self.root / thesis.slug, thesis, rule_objs)
        return thesis

    @staticmethod
    def _retained_ids(entries: object, key: str) -> set[str]:
        """Claim/expression ids present in a patch list (dict entries only)."""
        if not isinstance(entries, list):
            return set()
        return {str(e.get(key)) for e in entries if isinstance(e, dict) and isinstance(e.get(key), str)}

    @staticmethod
    def _asof_ids(base: ThesisStateSnapshot, key: str, noun: str) -> set[JSONValue | None]:
        """As-of snapshot ids for one section (dict entries only)."""
        entries = base.thesis.get(key, [])
        rows: list[JSONValue] = entries if isinstance(entries, list) else []
        return {c.get(f"{noun}_id") for c in rows if isinstance(c, dict)} - {None}

    @staticmethod
    def _require_no_dropped_ids(
        thesis_dir: Path,
        thesis_id: str,
        patch: Mapping[str, object],
        base: ThesisStateSnapshot,
        key: str,
        noun: str,
    ) -> None:
        """Backdated patch must retain every as-of id (single missing-id gate)."""
        if not isinstance(patch.get(key), list):
            return
        missing = ThesisRepository._asof_ids(base, key, noun) - ThesisRepository._retained_ids(
            patch.get(key), f"{noun}_id"
        )
        if missing:
            first = min(missing, key=str)
            raise ValueError(f"{thesis_dir}/thesis.yaml: {noun} {first!r} does not belong to thesis {thesis_id!r}")

    def _update_backdated_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        patch: Mapping[str, object],
        eff: str,
    ) -> Thesis:
        """As-of copy path: validate patch against snapshot, persist snapshot only."""
        base = self.load_state_as_of(thesis_id, eff)
        self._require_no_dropped_ids(thesis_dir, thesis_id, patch, base, "claims", "claim")
        self._require_no_dropped_ids(thesis_dir, thesis_id, patch, base, "expressions", "expression")
        data: dict[str, object] = dict(base.thesis)
        data.update(patch)
        data["thesis_id"] = thesis_id
        data["updated_at"] = eff
        candidate = Thesis.from_dict(data, str(thesis_dir / "thesis.yaml"))
        self._write_snapshot_from_dicts_locked(
            thesis_dir,
            thesis_id,
            effective_at=eff,
            reason="thesis_updated",
            thesis=candidate.to_dict(),
            state=dict(base.state),
            questions=dict(base.questions),
            watch=dict(base.watch),
            memory=dict(base.memory),
        )
        return candidate

    def _update_live_locked(
        self,
        thesis_dir: Path,
        thesis: Thesis,
        thesis_id: str,
        patch: Mapping[str, object],
        eff: str,
    ) -> Thesis:
        """Live path: patch thesis.yaml, then snapshot."""
        live_data: dict[str, object] = dict(thesis.to_dict())
        live_data.update(patch)
        live_data["thesis_id"] = thesis_id
        live_data["updated_at"] = _utcnow()
        candidate = Thesis.from_dict(live_data, str(thesis_dir / "thesis.yaml"))
        atomic_write_yaml(thesis_dir / "thesis.yaml", candidate.to_dict(), self.root)
        self._snapshot_state_locked(thesis_dir, thesis_id, effective_at=eff, reason="thesis_updated")
        return candidate

    def update_thesis(self, id_or_slug: str, *, effective_at: str | None = None, **patch: object) -> Thesis:
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eff = effective_at or _utcnow()
        eff_dt = _as_dt(eff)
        if eff_dt is None:
            raise ValueError(f"<update>: bad effective_at {effective_at!r}")
        thesis_id = self._resolve_id(id_or_slug)
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            thesis = self._check_owner(thesis_dir, thesis_id)
            if "thesis_id" in patch and patch["thesis_id"] != thesis_id:
                raise ValueError(f"{thesis_dir}/thesis.yaml: 'thesis_id' is immutable")
            latest = self._latest_effective_locked(thesis_dir)
            if latest is not None and eff_dt < latest[0]:
                return self._update_backdated_locked(thesis_dir, thesis_id, patch, eff)
            return self._update_live_locked(thesis_dir, thesis, thesis_id, patch, eff)

    def _status_backdated_locked(self, thesis_dir: Path, thesis_id: str, status: str, eff: str) -> Thesis:
        """As-of copy path: patch snapshot status, persist snapshot only."""
        base = self.load_state_as_of(thesis_id, eff)
        if base.thesis.get("status") == models.ThesisStatus.CLOSED.value:
            raise ValueError(f"{thesis_dir}/thesis.yaml: thesis {thesis_id!r} is closed; cannot change status")
        data = dict(base.thesis)
        data["status"] = status
        data["updated_at"] = eff
        data["thesis_id"] = thesis_id
        candidate = Thesis.from_dict(data, str(thesis_dir / "thesis.yaml"))
        self._write_snapshot_from_dicts_locked(
            thesis_dir,
            thesis_id,
            effective_at=eff,
            reason="status_changed",
            thesis=candidate.to_dict(),
            state=dict(base.state),
            questions=dict(base.questions),
            watch=dict(base.watch),
            memory=dict(base.memory),
        )
        return candidate

    def _set_status(self, thesis_id: str, status: str, *, effective_at: str | None = None) -> Thesis:
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eff = effective_at or _utcnow()
        eff_dt = _as_dt(eff)
        if eff_dt is None:
            raise ValueError(f"<status>: bad effective_at {effective_at!r}")
        thesis_id = self._resolve_id(thesis_id)
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            thesis = self._check_owner(thesis_dir, thesis_id)
            latest = self._latest_effective_locked(thesis_dir)
            if latest is not None and eff_dt < latest[0]:
                return self._status_backdated_locked(thesis_dir, thesis_id, status, eff)
            if thesis.status == models.ThesisStatus.CLOSED.value:
                raise ValueError(f"{thesis_dir}/thesis.yaml: thesis {thesis_id!r} is closed; cannot change status")
            data = thesis.to_dict()
            data["status"] = status
            data["updated_at"] = _utcnow()
            candidate = Thesis.from_dict(data, str(thesis_dir / "thesis.yaml"))
            atomic_write_yaml(thesis_dir / "thesis.yaml", candidate.to_dict(), self.root)
            self._snapshot_state_locked(thesis_dir, thesis_id, effective_at=eff, reason="status_changed")
            return candidate

    def pause_thesis(self, thesis_id: str, *, effective_at: str | None = None) -> Thesis:
        """Pause a thesis; accepts canonical ID or slug, resolved before owner checks."""
        return self._set_status(thesis_id, models.ThesisStatus.PAUSED.value, effective_at=effective_at)

    def resume_thesis(self, thesis_id: str, *, effective_at: str | None = None) -> Thesis:
        """Resume a thesis; accepts canonical ID or slug, resolved before owner checks."""
        return self._set_status(thesis_id, models.ThesisStatus.ACTIVE.value, effective_at=effective_at)

    def close_thesis(self, thesis_id: str, *, effective_at: str | None = None) -> Thesis:
        """Close a thesis; accepts canonical ID or slug, resolved before owner checks."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eff = effective_at or _utcnow()
        eff_dt = _as_dt(eff)
        if eff_dt is None:
            raise ValueError(f"<status>: bad effective_at {effective_at!r}")
        thesis_id = self._resolve_id(thesis_id)
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            thesis = self._check_owner(thesis_dir, thesis_id)
            latest = self._latest_effective_locked(thesis_dir)
            if latest is not None and eff_dt < latest[0]:
                base = self.load_state_as_of(thesis_id, eff)
                data = dict(base.thesis)
                data["status"] = models.ThesisStatus.CLOSED.value
                data["updated_at"] = eff
                data["thesis_id"] = thesis_id
                candidate = Thesis.from_dict(data, str(thesis_dir / "thesis.yaml"))
                self._write_snapshot_from_dicts_locked(
                    thesis_dir,
                    thesis_id,
                    effective_at=eff,
                    reason="status_changed",
                    thesis=candidate.to_dict(),
                    state=dict(base.state),
                    questions=dict(base.questions),
                    watch=dict(base.watch),
                    memory=dict(base.memory),
                )
                return candidate
            data = thesis.to_dict()
            data["status"] = models.ThesisStatus.CLOSED.value
            data["updated_at"] = _utcnow()
            candidate = Thesis.from_dict(data, str(thesis_dir / "thesis.yaml"))
            atomic_write_yaml(thesis_dir / "thesis.yaml", candidate.to_dict(), self.root)
            self._snapshot_state_locked(thesis_dir, thesis_id, effective_at=eff, reason="status_changed")
            return candidate

    @staticmethod
    def _journal_entry_id(entry: Mapping[str, object]) -> str:
        """Entry id: entry_id/journal_id or a fresh id (idempotent retry key)."""
        raw = entry.get("entry_id") or entry.get("journal_id")
        return raw if isinstance(raw, str) and raw else new_journal_id()

    @staticmethod
    def _journal_hit(content: str, entry_id: str) -> bool:
        """True when markdown front matter names this entry id."""
        head = content.split("---")
        return len(head) >= 3 and f"entry_id: {entry_id}" in head[1]

    def _find_journal_entry(self, thesis_dir: Path, entry_id: str) -> Path | None:
        """Existing journal file for the id; None on first write."""
        for existing in (thesis_dir / "journal").glob("*.md"):
            try:
                if self._journal_hit(existing.read_text(encoding="utf-8"), entry_id):
                    return existing
            except OSError:
                continue
        return None

    @staticmethod
    def _journal_text(entry: Mapping[str, object], entry_id: str) -> tuple[str, str, str]:
        """Created/title/body strings with the same defaults as the file path."""
        created_raw = entry.get("created_at") or _utcnow()
        created = created_raw if isinstance(created_raw, str) else str(created_raw)
        title_raw = entry.get("title", f"Journal {entry_id}")
        title = title_raw if isinstance(title_raw, str) else str(title_raw)
        body_raw = entry.get("body", entry.get("summary", ""))
        body = body_raw if isinstance(body_raw, str) else ("" if body_raw is None else str(body_raw))
        return created, title, body

    def _write_journal_locked(self, thesis_dir: Path, thesis_id: str, entry: Mapping[str, object]) -> Path:
        """Idempotent journal write (lock held): existing id returns its file."""
        d = dict(entry)
        entry_id = self._journal_entry_id(d)
        found = self._find_journal_entry(thesis_dir, entry_id)
        if found is not None:
            return found
        created, title, body = self._journal_text(d, entry_id)
        _reject_hostile_journal(title, body, f"{thesis_dir}/journal")
        dest = thesis_dir / "journal" / f"{_safe_name(entry_id)}.md"
        atomic_write_text(
            dest,
            _journal_front_matter(entry_id, thesis_id, created, d) + f"# {title}\n\n{body}\n",
            self.root,
        )
        return dest

    def append_journal_entry(self, thesis_id: str, entry: Mapping[str, object]) -> Path:
        """Idempotent by entry/journal id: retrying with the same id is a no-op."""
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            return self._write_journal_locked(thesis_dir, thesis_id, entry)

    @staticmethod
    def _journal_front_matter_fields(path: Path) -> dict[str, str] | None:
        """Front-matter key map; None when the file is unreadable."""
        try:
            head = path.read_text(encoding="utf-8").split("---")
            fm = head[1] if len(head) >= 3 else ""
        except OSError:
            return None
        return {ln.split(":", 1)[0].strip(): ln.split(":", 1)[1].strip() for ln in fm.splitlines() if ":" in ln}

    @staticmethod
    def _journal_known_ok(fields: dict[str, str], known_at: str | None, want: datetime | None) -> bool:
        """Known_at predicate: unfiltered passes; else parsed stamp must equal want."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        if known_at is None:
            return True
        if want is None:
            return False
        got = _as_dt(fields.get("known_at", "")) if fields.get("known_at") else None
        return got is not None and got == want

    @staticmethod
    def _journal_matches(
        fields: dict[str, str],
        thesis_id: str,
        trigger_id: str,
        known_at: str | None,
        want: datetime | None,
        run_id: str | None,
    ) -> bool:
        """Thesis/trigger/run/known_at predicate for one journal's fields."""
        if fields.get("thesis_id") != thesis_id or fields.get("trigger_id") != trigger_id:
            return False
        if not ThesisRepository._journal_known_ok(fields, known_at, want):
            return False
        return run_id is None or fields.get("run_id", "") == run_id

    def _journal_dir_matches(
        self,
        journal_dir: Path,
        thesis_id: str,
        trigger_id: str,
        known_at: str | None,
        want: datetime | None,
        run_id: str | None,
    ) -> bool:
        """True when any journal file in the dir matches the predicate."""
        for f in journal_dir.glob("*.md"):
            fields = self._journal_front_matter_fields(f)
            if fields is not None and self._journal_matches(fields, thesis_id, trigger_id, known_at, want, run_id):
                return True
        return False

    def has_journal_for_trigger(
        self, thesis_id: str, trigger_id: str, *, known_at: str | None = None, run_id: str | None = None
    ) -> bool:
        """True when a durable journal entry names this thesis and trigger."""
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        want = _as_dt(known_at) if known_at is not None else None
        journal_dir = self._dir_for(thesis_id) / "journal"
        if not journal_dir.is_dir():
            return False
        return self._journal_dir_matches(journal_dir, thesis_id, trigger_id, known_at, want, run_id)

    @staticmethod
    def _require_trigger_membership(
        thesis_dir: Path,
        known: set[str],
        ids: Sequence[str],
        noun: str,
    ) -> None:
        """Claim/expression membership for one trigger id list."""
        for given in ids or []:
            if given not in known:
                raise ValueError(f"{thesis_dir}: trigger references absent {noun} {given!r}")

    @staticmethod
    def _validate_trigger_refs(
        thesis_dir: Path,
        thesis: Thesis,
        claim_ids: Sequence[str],
        expression_ids: Sequence[str],
        summary_origin: str,
    ) -> None:
        """Closed-thesis + claim/expression membership + summary origin gates."""
        if thesis.status == models.ThesisStatus.CLOSED.value:
            raise ValueError(f"{thesis_dir}: thesis {thesis.thesis_id!r} is closed; cannot create triggers")
        ThesisRepository._require_trigger_membership(
            thesis_dir, {c.claim_id for c in thesis.claims}, claim_ids, "claim"
        )
        ThesisRepository._require_trigger_membership(
            thesis_dir, {e.expression_id for e in thesis.expressions}, expression_ids, "expression"
        )
        if summary_origin not in ("deterministic", "recycled"):
            raise ValueError(f"{thesis_dir}: bad summary_origin {summary_origin!r}")

    @staticmethod
    def _new_trigger(
        thesis_id: str,
        trigger_type: str,
        importance: str,
        claim_ids: Sequence[str],
        expression_ids: Sequence[str],
        canonical_refs: Sequence[str],
        summary: str,
        metadata: dict[str, JSONValue] | None,
        summary_origin: str,
    ) -> Trigger:
        """Validated trigger object (metadata carries the origin)."""
        meta = dict(metadata or {})
        meta["summary_origin"] = summary_origin
        return Trigger(
            trigger_id=new_trigger_id(),
            thesis_id=thesis_id,
            created_at=_utcnow(),
            status=TriggerStatus.PENDING.value,
            trigger_type=trigger_type,
            importance=importance,
            claim_ids=tuple(claim_ids or []),
            expression_ids=tuple(expression_ids or []),
            canonical_refs=tuple(canonical_refs or []),
            summary=summary,
            metadata=meta,
        )

    def _persist_trigger_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        trigger_type: str,
        importance: str,
        claim_ids: Sequence[str],
        expression_ids: Sequence[str],
        canonical_refs: Sequence[str],
        summary: str,
        metadata: dict[str, JSONValue] | None,
        summary_origin: str,
    ) -> Trigger:
        """Validated trigger -> inbox file (lock held)."""
        trigger = self._new_trigger(
            thesis_id,
            trigger_type,
            importance,
            claim_ids,
            expression_ids,
            canonical_refs,
            summary,
            metadata,
            summary_origin,
        )
        Trigger.from_dict(trigger.to_dict(), str(thesis_dir / "inbox"))
        dest = thesis_dir / "inbox" / f"{_safe_name(trigger.trigger_id)}.yaml"
        atomic_write_yaml(dest, {"schema_version": SCHEMA_VERSION, **trigger.to_dict()}, self.root)
        return trigger

    def create_trigger(
        self,
        thesis_id: str,
        trigger_type: str = "new_external_evidence",
        importance: str = "medium",
        claim_ids: Sequence[str] = (),
        expression_ids: Sequence[str] = (),
        canonical_refs: Sequence[str] = (),
        summary: str = "",
        metadata: dict[str, JSONValue] | None = None,
        *,
        summary_origin: str,
    ) -> Trigger:
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            thesis = self._check_owner(thesis_dir, thesis_id)
            self._validate_trigger_refs(thesis_dir, thesis, claim_ids, expression_ids, summary_origin)
            return self._persist_trigger_locked(
                thesis_dir,
                thesis_id,
                trigger_type,
                importance,
                claim_ids,
                expression_ids,
                canonical_refs,
                summary,
                metadata,
                summary_origin,
            )

    def _resolve_trigger_path(self, thesis_dir: Path, trigger_id: str) -> Path:
        """Direct inbox name, else scan for older filenames (never deletes)."""
        target = thesis_dir / "inbox" / f"{_safe_name(trigger_id)}.yaml"
        if target.is_file():
            return target
        for f in (thesis_dir / "inbox").glob("*.yaml"):
            raw = load_raw_yaml(f)
            if raw.get("trigger_id") == trigger_id:
                return f
        raise KeyError(f"unknown trigger: {trigger_id!r}")

    @staticmethod
    def _fold_trigger_processed(
        raw: dict[str, JSONValue],
        run_id: str,
        metadata: dict[str, JSONValue] | None,
    ) -> None:
        """Status/processed_at/run/metadata fold for the processed transition."""
        raw["status"] = TriggerStatus.PROCESSED.value
        raw["processed_at"] = _utcnow()
        if run_id:
            raw["run_id"] = run_id
        if metadata:
            merged = dict(raw.get("metadata") or {})
            merged.update(metadata)
            raw["metadata"] = merged

    def mark_trigger_processed(
        self, thesis_id: str, trigger_id: str, run_id: str = "", metadata: dict[str, JSONValue] | None = None
    ) -> Trigger:
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            target = self._resolve_trigger_path(thesis_dir, trigger_id)
            raw = load_raw_yaml(target)
            self._check_file_owner(raw, str(target), thesis_id)
            self._fold_trigger_processed(raw, run_id, metadata)
            updated = Trigger.from_dict(raw, str(target))
            atomic_write_yaml(target, {"schema_version": SCHEMA_VERSION, **updated.to_dict()}, self.root)
            return updated

    def _direct_trigger_raw(
        self, thesis_dir: Path, thesis_id: str, trigger_id: str
    ) -> tuple[Path, dict[str, JSONValue]] | None:
        """Direct inbox name hit; None when the file is absent."""
        direct = thesis_dir / "inbox" / f"{_safe_name(trigger_id)}.yaml"
        if not direct.is_file():
            return None
        raw = load_raw_yaml(direct)
        self._check_file_owner(raw, str(direct), thesis_id)
        return direct, raw

    def _scan_trigger_raw(
        self, thesis_dir: Path, thesis_id: str, trigger_id: str
    ) -> tuple[Path, dict[str, JSONValue]] | None:
        """Old-filename inbox scan hit; None when nothing names the trigger."""
        for f in sorted((thesis_dir / "inbox").glob("*.yaml")):
            candidate = self._trigger_scan_hit(f, thesis_id, trigger_id)
            if candidate is not None:
                return candidate
        return None

    def _load_trigger_raw(self, thesis_dir: Path, thesis_id: str, trigger_id: str) -> tuple[Path, dict[str, JSONValue]]:
        """Locate one trigger's raw mapping (direct name, else inbox scan)."""
        hit = self._direct_trigger_raw(thesis_dir, thesis_id, trigger_id)
        if hit is not None:
            return hit
        scanned = self._scan_trigger_raw(thesis_dir, thesis_id, trigger_id)
        if scanned is not None:
            return scanned
        raise KeyError(f"unknown trigger: {trigger_id!r}")

    def _trigger_scan_hit(
        self, path: Path, thesis_id: str, trigger_id: str
    ) -> tuple[Path, dict[str, JSONValue]] | None:
        """Inbox scan hit: raw names the trigger and passes the owner check."""
        try:
            raw = load_raw_yaml(path)
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            return None
        if raw.get("trigger_id") != trigger_id:
            return None
        self._check_file_owner(raw, str(path), thesis_id)
        return path, raw

    @staticmethod
    def _evidence_ref_of(raw: object, thesis_id: str) -> str | None:
        """Canonical ref of one evidence file; None when foreign/corrupt/ref-less."""
        if not isinstance(raw, dict) or raw.get("thesis_id") != thesis_id:
            return None
        ref = raw.get("canonical_ref")
        return ref if isinstance(ref, str) and ref else None

    def evidence_canonical_refs(self, thesis_id: str) -> set[str]:
        """Canonical refs of stored evidence files (provenance pool for writeback)."""
        thesis_dir = self._dir_for(thesis_id)
        out: set[str] = set()
        evdir = thesis_dir / "evidence"
        if not evdir.is_dir():
            return out
        for f in sorted(evdir.glob("*.yaml")):
            try:
                raw = load_raw_yaml(f)
            except Exception:  # noqa: BLE001, S112 - intentional best-effort boundary, never aborts
                continue
            ref = self._evidence_ref_of(raw, thesis_id)
            if ref is not None:
                out.add(ref)
        return out

    def _pending_path(self, thesis_dir: Path, trigger_id: str) -> Path:
        return thesis_dir / "inbox" / f"{_safe_name(trigger_id)}.pending.json"

    @staticmethod
    def _coerce_pending_raw(path: Path, raw: object) -> object:
        """Pending intent text -> parsed JSON (raises when unreadable)."""
        if not isinstance(raw, str):
            raise ValueError(f"{path}: unreadable pending result intent: expected text")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        try:
            parsed: object = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{path}: unreadable pending result intent: {exc}") from exc
        return parsed

    @staticmethod
    def _pending_owner_ok(raw: dict[object, object], thesis_id: str, trigger_id: str) -> bool:
        """Owner gate: intent names this thesis and trigger."""
        return raw.get("thesis_id") == thesis_id and raw.get("trigger_id") == trigger_id

    @staticmethod
    def _pending_shape_ok(raw: dict[object, object]) -> bool:
        """Run/payload shape gate: dict payload plus non-empty run id."""
        return isinstance(raw.get("payload"), dict) and isinstance(raw.get("run_id"), str) and bool(raw["run_id"])

    @staticmethod
    def _validate_pending_intent(
        path: Path,
        raw: object,
        thesis_id: str,
        trigger_id: str,
    ) -> dict[str, JSONValue]:
        """Owner/run/payload shape gate (foreign or corrupt stays loud)."""
        if (
            not isinstance(raw, dict)
            or not ThesisRepository._pending_owner_ok(raw, thesis_id, trigger_id)
            or not ThesisRepository._pending_shape_ok(raw)
        ):
            raise ValueError(f"{path}: foreign or corrupt pending result intent; operator review required")
        return dict(raw)

    def read_pending_result(self, thesis_id: str, trigger_id: str) -> dict[str, JSONValue] | None:
        """Durable research-result intent for crash replay (None when absent)."""
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            p = self._pending_path(thesis_dir, trigger_id)
            if not p.is_file():
                return None
            try:
                text = p.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"{p}: unreadable pending result intent: {exc}") from exc
            return self._validate_pending_intent(p, self._coerce_pending_raw(p, text), thesis_id, trigger_id)

    def write_pending_result(self, thesis_id: str, trigger_id: str, intent: Mapping[str, JSONValue]) -> Path:
        """Persist the validated result before mutating (replay skips the model call)."""
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            body: dict[str, JSONValue] = dict(intent)
            body["thesis_id"] = thesis_id
            body["trigger_id"] = trigger_id
            return atomic_write_json(self._pending_path(thesis_dir, trigger_id), body, self.root)

    def clear_pending_result(self, thesis_id: str, trigger_id: str) -> None:
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            try:
                self._pending_path(thesis_dir, trigger_id).unlink()
            except FileNotFoundError:
                pass

    def load_checkpoint(self, id_or_slug: str) -> Checkpoint:
        """Read validated checkpoint.yaml (lock-held; KeyError on unknown thesis)."""
        thesis_id = self._resolve_id(id_or_slug)
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            raw = load_raw_yaml(thesis_dir / "checkpoint.yaml")
            self._check_file_owner(raw, str(thesis_dir / "checkpoint.yaml"), thesis_id)
            return Checkpoint.from_dict(raw, str(thesis_dir / "checkpoint.yaml"))

    def save_checkpoint(self, thesis_id: str, checkpoint: Checkpoint | Mapping[str, object]) -> Checkpoint:
        """Validate-then-replace checkpoint.yaml (lock-held, atomic)."""
        thesis_dir = self._dir_for(thesis_id)
        with thesis_lock(thesis_dir):
            self._check_owner(thesis_dir, thesis_id)
            data = checkpoint.to_dict() if isinstance(checkpoint, Checkpoint) else dict(checkpoint)
            data["thesis_id"] = thesis_id
            candidate = Checkpoint.from_dict(data, str(thesis_dir / "checkpoint.yaml"))
            atomic_write_yaml(thesis_dir / "checkpoint.yaml", candidate.to_dict(), self.root)
            return candidate

    def _fold_live_claim_expression(
        self,
        thesis_dir: Path,
        thesis_id: str,
        thesis: Thesis,
        claim_patch: dict[str, dict[str, object]],
        expr_patch: dict[str, dict[str, object]],
        eff: str,
    ) -> Thesis:
        """Live thesis.yaml fold: patch statuses, persist, return new Thesis."""
        if not (claim_patch or expr_patch):
            return thesis
        folded = _fold_claim_expression(
            thesis.to_dict(), claim_patch, expr_patch, thesis_id, _utcnow(), str(thesis_dir / "thesis.yaml")
        )
        # backdated clock uses eff; live clock stamps now (behavior preserved)
        thesis = Thesis.from_dict({**folded, "updated_at": folded.get("updated_at")}, str(thesis_dir / "thesis.yaml"))
        atomic_write_yaml(thesis_dir / "thesis.yaml", thesis.to_dict(), self.root)
        return thesis

    def _fold_live_state(self, thesis_dir: Path, thesis_id: str, state_raw: object) -> None:
        """Live state.yaml fold: validate candidate, then replace."""
        folded = _fold_state_candidate(state_raw, thesis_id, thesis_dir)
        candidate = ThesisState.from_dict(dict(folded), str(thesis_dir / "state.yaml"))
        atomic_write_yaml(thesis_dir / "state.yaml", candidate.to_dict(), self.root)

    def _fold_live_questions(
        self,
        thesis_dir: Path,
        thesis_id: str,
        additions: list[object],
    ) -> bool:
        """Live questions.yaml fold: append by id, persist, report change."""
        raw = load_raw_yaml(thesis_dir / "questions.yaml")
        self._check_file_owner(raw, str(thesis_dir / "questions.yaml"), thesis_id)
        entries = raw.get("questions", [])
        if not isinstance(entries, list):
            raise ValueError(f"{thesis_dir}/questions.yaml: 'questions' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        questions_raw: dict[str, JSONValue] = {
            "schema_version": raw.get("schema_version", SCHEMA_VERSION),
            "thesis_id": thesis_id,
            "questions": entries,
        }
        added = _append_questions(questions_raw, additions, thesis_dir)
        atomic_write_yaml(thesis_dir / "questions.yaml", raw, self.root)
        return added

    def _fold_live_memories(
        self,
        thesis_dir: Path,
        thesis_id: str,
        additions: list[object],
        eff: str,
    ) -> bool:
        """Live memory.yaml fold: append by id, persist, report change."""
        raw = load_raw_yaml(thesis_dir / "memory.yaml")
        self._check_file_owner(raw, str(thesis_dir / "memory.yaml"), thesis_id)
        entries = raw.get("memories", [])
        if not isinstance(entries, list):
            raise ValueError(f"{thesis_dir}/memory.yaml: 'memories' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        memory_raw: dict[str, JSONValue] = {
            "schema_version": raw.get("schema_version", SCHEMA_VERSION),
            "thesis_id": thesis_id,
            "memories": entries,
        }
        added = _append_memories(memory_raw, additions, thesis_dir, eff)
        atomic_write_yaml(thesis_dir / "memory.yaml", raw, self.root)
        return added

    def _fold_live_watch(
        self,
        thesis_dir: Path,
        thesis_id: str,
        thesis: Thesis,
        additions: list[object],
    ) -> bool:
        """Live watch.yaml fold: cross-ref against live Thesis, persist, report change."""
        raw = load_raw_yaml(thesis_dir / "watch.yaml")
        self._check_file_owner(raw, str(thesis_dir / "watch.yaml"), thesis_id)
        entries = raw.get("rules", [])
        if not isinstance(entries, list):
            raise ValueError(f"{thesis_dir}/watch.yaml: 'rules' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        watch_raw: dict[str, JSONValue] = {
            "schema_version": raw.get("schema_version", SCHEMA_VERSION),
            "thesis_id": thesis_id,
            "rules": entries,
        }
        added = _append_watch_rules(watch_raw, additions, thesis.to_dict(), thesis_dir)
        atomic_write_yaml(thesis_dir / "watch.yaml", raw, self.root)
        return added

    def _fold_live_answers(
        self,
        thesis_dir: Path,
        thesis_id: str,
        answers: list[object],
    ) -> bool:
        """Live questions-answered fold: patch file, persist, report change."""
        qpath = thesis_dir / "questions.yaml"
        qraw = load_raw_yaml(qpath)
        self._check_file_owner(qraw, str(qpath), thesis_id)
        entries = qraw.get("questions", [])
        if not isinstance(entries, list):
            raise ValueError(f"{qpath}: 'questions' must be a list")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        questions_raw: dict[str, JSONValue] = {
            "schema_version": qraw.get("schema_version", SCHEMA_VERSION),
            "thesis_id": thesis_id,
            "questions": entries,
        }
        _mark_questions_answered(questions_raw, answers, thesis_dir)
        atomic_write_yaml(qpath, qraw, self.root)
        return True

    def _resolve_apply_gate(
        self,
        thesis_dir: Path,
        thesis_id: str,
        trigger_id: str,
        journal_entry: Mapping[str, object] | None,
        allowed_refs: set[str] | None,
    ) -> tuple[Path | None, dict[str, JSONValue] | None, set[str] | None]:
        """Trigger load + allowed-set for one commit (no writes)."""
        if not trigger_id:
            return None, None, None
        tpath, raw_trigger = self._load_trigger_raw(thesis_dir, thesis_id, trigger_id)
        return (
            tpath,
            raw_trigger,
            _resolve_gate_allowed(thesis_dir, thesis_id, journal_entry, raw_trigger, allowed_refs),
        )

    def _backdated_patches(
        self,
        base: ThesisStateSnapshot,
        thesis_dir: Path,
        thesis_id: str,
        inputs: _ApplyInputs,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        """As-of claim/expression patches (membership against the snapshot)."""
        claim_patch = _apply_claim_patch(
            inputs.claim_updates, _snapshot_member_ids(base.thesis.get("claims", []), "claim_id"), thesis_dir, thesis_id
        )
        expr_patch = _apply_expression_patch(
            inputs.expression_updates,
            _snapshot_member_ids(base.thesis.get("expressions", []), "expression_id"),
            thesis_dir,
            thesis_id,
        )
        return claim_patch, expr_patch

    def _backdated_section_folds(
        self,
        thesis_dir: Path,
        eff: str,
        inputs: _ApplyInputs,
        thesis_d: Mapping[str, object],
        questions_raw: dict[str, JSONValue],
        memory_raw: dict[str, JSONValue],
        watch_raw: dict[str, JSONValue],
    ) -> _ApplyFold:
        """Append phases against as-of copies; returns change flags."""
        questions_added = (
            _append_questions(questions_raw, inputs.questions_add, thesis_dir) if inputs.questions_add else False
        )
        memories_added = (
            _append_memories(memory_raw, inputs.memories_add, thesis_dir, eff) if inputs.memories_add else False
        )
        watch_added = (
            _append_watch_rules(watch_raw, inputs.watch_add, thesis_d, thesis_dir) if inputs.watch_add else False
        )
        answered_updated = bool(inputs.questions_answered)
        if inputs.questions_answered:
            _mark_questions_answered(questions_raw, inputs.questions_answered, thesis_dir)
        return _ApplyFold(
            False, inputs.state is not None, questions_added, memories_added, watch_added, answered_updated
        )

    def _apply_backdated_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        eff: str,
        inputs: _ApplyInputs,
        run_id: str,
        allowed_refs: set[str] | None,
        outcome: dict[str, JSONValue],
    ) -> None:
        """Backdated commit: patch as-of snapshot copies, snapshot-only persist."""
        base = self.load_state_as_of(thesis_id, eff)
        if base.thesis.get("status") == models.ThesisStatus.CLOSED.value:
            raise ValueError(f"{thesis_dir}: thesis {thesis_id!r} is closed; refusing research writeback")
        claim_patch, expr_patch = self._backdated_patches(base, thesis_dir, thesis_id, inputs)
        tpath, raw_trigger, allowed = self._resolve_apply_gate(
            thesis_dir, thesis_id, inputs.trigger_id, inputs.journal_entry, allowed_refs
        )
        _check_evidence_provenance(inputs.evidence_refs, allowed, thesis_dir, inputs.trigger_id)
        thesis_d = dict(base.thesis)
        state_d = dict(base.state)
        questions_raw, _ = _copy_section_list(base.questions, "questions", thesis_id, SCHEMA_VERSION)
        memory_raw, _ = _copy_section_list(base.memory, "memories", thesis_id, SCHEMA_VERSION)
        watch_raw, _ = _copy_section_list(base.watch, "rules", thesis_id, SCHEMA_VERSION)
        if claim_patch or expr_patch:
            thesis_d = _fold_claim_expression(
                base.thesis, claim_patch, expr_patch, thesis_id, eff, str(thesis_dir / "thesis.yaml")
            )
        if inputs.state is not None:
            state_d = _fold_state_candidate(inputs.state, thesis_id, thesis_dir)
        flags = self._backdated_section_folds(thesis_dir, eff, inputs, thesis_d, questions_raw, memory_raw, watch_raw)
        self._backdated_side_effects(
            thesis_dir,
            thesis_id,
            eff,
            inputs,
            run_id,
            tpath,
            raw_trigger,
            claim_patch,
            expr_patch,
            outcome,
            thesis_d,
            state_d,
            questions_raw,
            watch_raw,
            memory_raw,
            flags,
        )

    def _backdated_side_effects(
        self,
        thesis_dir: Path,
        thesis_id: str,
        eff: str,
        inputs: _ApplyInputs,
        run_id: str,
        tpath: Path | None,
        raw_trigger: dict[str, JSONValue] | None,
        claim_patch: dict[str, dict[str, object]],
        expr_patch: dict[str, dict[str, object]],
        outcome: dict[str, JSONValue],
        thesis_d: dict[str, JSONValue],
        state_d: dict[str, JSONValue],
        questions_raw: dict[str, JSONValue],
        watch_raw: dict[str, JSONValue],
        memory_raw: dict[str, JSONValue],
        flags: _ApplyFold,
    ) -> None:
        """Backdated evidence/journal/trigger writes plus snapshot persist."""
        outcome["evidence"] = _append_evidence_refs(self.root, thesis_dir, thesis_id, inputs.evidence_refs)
        journaled = _write_journal_entry(
            self.root, thesis_dir, thesis_id, inputs.journal_entry, run_id, inputs.trigger_id
        )
        if journaled is not None:
            outcome["journal"] = journaled
        marked = _mark_trigger_processed(self, thesis_dir, thesis_id, inputs.trigger_id, run_id, tpath, raw_trigger)
        if marked is not None:
            outcome["trigger"] = marked
        if flags.dirty(bool(claim_patch or expr_patch)):
            self._write_snapshot_from_dicts_locked(
                thesis_dir,
                thesis_id,
                effective_at=eff,
                reason="research_result",
                run_id=run_id,
                trigger_id=inputs.trigger_id,
                thesis=thesis_d,
                state=state_d,
                questions=questions_raw,
                watch=watch_raw,
                memory=memory_raw,
            )

    def _apply_live_locked(
        self,
        thesis_dir: Path,
        thesis_id: str,
        thesis: Thesis,
        eff: str,
        inputs: _ApplyInputs,
        run_id: str,
        allowed_refs: set[str] | None,
        outcome: dict[str, JSONValue],
    ) -> Thesis:
        """Live commit: patch live files in fixed phase order, then snapshot."""
        if thesis.status == models.ThesisStatus.CLOSED.value:
            raise ValueError(f"{thesis_dir}: thesis {thesis_id!r} is closed; refusing research writeback")
        claim_patch = _apply_claim_patch(
            inputs.claim_updates, {x.claim_id for x in thesis.claims}, thesis_dir, thesis_id
        )
        expr_patch = _apply_expression_patch(
            inputs.expression_updates, {x.expression_id for x in thesis.expressions}, thesis_dir, thesis_id
        )
        tpath, raw_trigger, allowed = self._resolve_apply_gate(
            thesis_dir, thesis_id, inputs.trigger_id, inputs.journal_entry, allowed_refs
        )
        _check_evidence_provenance(inputs.evidence_refs, allowed, thesis_dir, inputs.trigger_id)
        thesis = self._fold_live_claim_expression(thesis_dir, thesis_id, thesis, claim_patch, expr_patch, eff)
        outcome["evidence"] = _append_evidence_refs(self.root, thesis_dir, thesis_id, inputs.evidence_refs)
        flags = self._live_section_folds(thesis_dir, thesis_id, thesis, eff, inputs)
        journaled = _write_journal_entry(
            self.root, thesis_dir, thesis_id, inputs.journal_entry, run_id, inputs.trigger_id
        )
        if journaled is not None:
            outcome["journal"] = journaled
        marked = _mark_trigger_processed(self, thesis_dir, thesis_id, inputs.trigger_id, run_id, tpath, raw_trigger)
        if marked is not None:
            outcome["trigger"] = marked
        if flags.dirty(bool(claim_patch or expr_patch)):
            self._snapshot_state_locked(
                thesis_dir,
                thesis_id,
                effective_at=eff,
                reason="research_result",
                run_id=run_id,
                trigger_id=inputs.trigger_id,
            )
        return thesis

    def _live_qmw_folds(
        self,
        thesis_dir: Path,
        thesis_id: str,
        thesis: Thesis,
        eff: str,
        inputs: _ApplyInputs,
    ) -> tuple[bool, bool, bool]:
        """Live questions/memory/watch folds (each gated on its additions)."""
        questions_added = (
            self._fold_live_questions(thesis_dir, thesis_id, inputs.questions_add) if inputs.questions_add else False
        )
        memories_added = (
            self._fold_live_memories(thesis_dir, thesis_id, inputs.memories_add, eff) if inputs.memories_add else False
        )
        watch_added = (
            self._fold_live_watch(thesis_dir, thesis_id, thesis, inputs.watch_add) if inputs.watch_add else False
        )
        return questions_added, memories_added, watch_added

    def _live_section_folds(
        self,
        thesis_dir: Path,
        thesis_id: str,
        thesis: Thesis,
        eff: str,
        inputs: _ApplyInputs,
    ) -> _ApplyFold:
        """Live questions/memory/watch/answers/state folds; returns change flags."""
        state_changed = inputs.state is not None
        if state_changed:
            self._fold_live_state(thesis_dir, thesis_id, inputs.state)
        questions_added, memories_added, watch_added = self._live_qmw_folds(thesis_dir, thesis_id, thesis, eff, inputs)
        answered_updated = (
            self._fold_live_answers(thesis_dir, thesis_id, inputs.questions_answered)
            if inputs.questions_answered
            else False
        )
        return _ApplyFold(False, state_changed, questions_added, memories_added, watch_added, answered_updated)

    def apply_research_result(
        self,
        thesis_id: str,
        result: Mapping[str, object],
        run_id: str = "",
        *,
        allowed_refs: set[str] | None = None,
        effective_at: str | None = None,
    ) -> dict[str, JSONValue]:
        """Single replayable research commit; de-dup additions by ID (crash-retry safe).

        One lock, fixed order: claim/expression status -> evidence refs -> state
        -> questions -> memory -> watch -> journal -> trigger processed metadata.
        Evidence provenance is validated before any write: refs carrying bodies,
        URLs, or publication/retrieval metadata are rejected, and (when a
        trigger_id is present) every canonical ref must name the run's allowed
        set. The runner threads its visible-run set through ``allowed_refs``;
        without it the allowed set is the trigger's refs plus stored refs whose
        known_at is chronologically at or before the payload time. Never
        deletes triggers.
        """
        if not isinstance(result, dict):
            raise ValueError(f"<apply>: result must be a mapping, got {type(result).__name__}")  # noqa: TRY004 - public error contract pins ValueError, tests are oracle
        from app.thesis.monitor import _as_dt  # local: monitor owns the clock helpers

        eff = effective_at or _utcnow()
        eff_dt = _as_dt(eff)
        if eff_dt is None:
            raise ValueError(f"<apply>: bad effective_at {effective_at!r}")
        thesis_dir = self._dir_for(thesis_id)
        outcome: dict[str, JSONValue] = {}
        with thesis_lock(thesis_dir):
            thesis = self._check_owner(thesis_dir, thesis_id)
            latest = self._latest_effective_locked(thesis_dir)
            inputs = _coerce_apply_inputs(result, thesis_dir)
            if latest is not None and eff_dt < latest[0]:
                self._apply_backdated_locked(thesis_dir, thesis_id, eff, inputs, run_id, allowed_refs, outcome)
                return outcome
            self._apply_live_locked(thesis_dir, thesis_id, thesis, eff, inputs, run_id, allowed_refs, outcome)
        return outcome
