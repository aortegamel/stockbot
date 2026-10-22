"""Immutable per-wave evidence freezes. stdlib only."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from .evidence import Evidence
from .models import normalize_time, pit_unverified, pit_violated, utcnow

__all__ = [
    "EvidenceFreeze",
    "FreezeIntegrityError",
    "create_freeze",
    "freeze_content_hash",
    "freeze_datapoint_hash",
    "freeze_from_dict",
    "freeze_to_dict",
    "verify_freeze",
]


class FreezeIntegrityError(ValueError):
    """Missing id, session/wave drift, PIT breach, or hash mismatch."""


@dataclass(frozen=True)
class EvidenceFreeze:
    """One immutable wave snapshot. New freeze per wave; never mutate E1 into E2."""

    freeze_id: str
    session_id: str
    wave_id: int
    created_at: datetime
    as_of: datetime | None
    evidence_ids: tuple[str, ...]
    content_hash: str
    # Pins value/entity/period/unit/proof per row; None only on pre-datapoint (legacy) freezes.
    datapoint_hash: str | None = None

    def _check_ids(self) -> None:
        if not self.freeze_id:
            raise FreezeIntegrityError("freeze: 'freeze_id' must be non-empty")
        if not self.session_id:
            raise FreezeIntegrityError("freeze: 'session_id' must be non-empty")
        if not self.content_hash:
            raise FreezeIntegrityError(f"freeze {self.freeze_id}: 'content_hash' must be non-empty")
        if self.datapoint_hash is not None and not self.datapoint_hash:
            raise FreezeIntegrityError(f"freeze {self.freeze_id}: 'datapoint_hash' must be a non-empty string or null")

    def __post_init__(self) -> None:
        self._check_ids()
        if isinstance(self.wave_id, bool) or not isinstance(self.wave_id, int) or self.wave_id < 1:
            raise FreezeIntegrityError("freeze: 'wave_id' must be an int >= 1")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


def _sort_key(record: Evidence) -> str:
    return record.evidence_id


def freeze_content_hash(records: Sequence[Evidence]) -> str:
    """Canonical hash over sorted ``evidence_id:content_hash`` pairs."""
    body = "\n".join(f"{e.evidence_id}:{e.content_hash}" for e in sorted(records, key=_sort_key))
    return sha256(body.encode("utf-8")).hexdigest()


def freeze_datapoint_hash(records: Sequence[Evidence]) -> str:
    """Canonical hash over sorted rows: id, value, entity, period, unit, proof as sorted-key JSON lines.

    Kept apart from ``freeze_content_hash`` so legacy freezes keep their stored hash.
    """
    body = "\n".join(
        json.dumps([e.evidence_id, e.value, e.entity, e.period, e.unit, e.proof], sort_keys=True, separators=(",", ":"))
        for e in sorted(records, key=_sort_key)
    )
    return sha256(body.encode("utf-8")).hexdigest()


def _has_datapoint(record: Evidence) -> bool:
    return bool(record.proof) or any(getattr(record, k) is not None for k in ("value", "entity", "period", "unit"))


def _coerce_time(value: datetime | str | None, key: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return normalize_time(value)
    try:
        return normalize_time(datetime.fromisoformat(value))
    except ValueError:
        raise FreezeIntegrityError(f"freeze: '{key}' must be an ISO-8601 datetime, got {value!r}") from None


def _check_freeze_wave(freeze_id: str, wave_id: int) -> None:
    if isinstance(wave_id, bool) or not isinstance(wave_id, int) or wave_id < 1:
        raise FreezeIntegrityError(f"freeze {freeze_id}: 'wave_id' must be an int >= 1")


def _check_freeze_membership(freeze_id: str, session_id: str, wave_id: int, recs: list[Evidence]) -> None:
    for record in recs:
        w = record.wave_id
        if record.session_id != session_id or isinstance(w, bool) or not isinstance(w, int) or w < 1 or w > wave_id:
            raise FreezeIntegrityError(
                f"freeze {freeze_id}: {record.evidence_id} belongs to {record.session_id}:{record.wave_id}"
            )


def _check_freeze_pit(freeze_id: str, frozen_as_of: datetime | None, recs: list[Evidence]) -> None:
    for record in recs:
        if pit_unverified(frozen_as_of, record.known_at):
            raise FreezeIntegrityError(
                f"freeze {freeze_id}: {record.evidence_id} unverified PIT (known_at unknown for historical as_of)"
            )
        if pit_violated(frozen_as_of, record.known_at):
            raise FreezeIntegrityError(f"freeze {freeze_id}: {record.evidence_id} violates PIT (known_at > as_of)")


def _freeze_ids(freeze_id: str, recs: list[Evidence]) -> tuple[str, ...]:
    ids = tuple(sorted({record.evidence_id for record in recs}))
    if len(ids) != len(recs):
        raise FreezeIntegrityError(f"freeze {freeze_id}: duplicate evidence ids")
    return ids


def create_freeze(
    *,
    freeze_id: str,
    session_id: str,
    wave_id: int,
    records: Sequence[Evidence],
    as_of: datetime | str | None = None,
    created_at: datetime | str | None = None,
) -> EvidenceFreeze:
    """Snapshot through a wave: every record must belong to this session with 1 <= wave <= freeze wave, and satisfy PIT."""
    _check_freeze_wave(freeze_id, wave_id)
    recs = list(records)
    _check_freeze_membership(freeze_id, session_id, wave_id, recs)
    frozen_as_of = _coerce_time(as_of, "as_of")
    _check_freeze_pit(freeze_id, frozen_as_of, recs)
    ids = _freeze_ids(freeze_id, recs)
    return EvidenceFreeze(
        freeze_id=freeze_id,
        session_id=session_id,
        wave_id=wave_id,
        created_at=_coerce_time(created_at, "created_at") or utcnow(),
        as_of=frozen_as_of,
        evidence_ids=ids,
        content_hash=freeze_content_hash(recs),
        datapoint_hash=freeze_datapoint_hash(recs),
    )


def verify_freeze(freeze: EvidenceFreeze, records: Sequence[Evidence]) -> None:
    """Recompute both hashes over ``records``; raise on id-set, content, or datapoint drift.

    A legacy freeze (no ``datapoint_hash``) pinned rows that carried no datapoint,
    so any datapoint field on those rows now is drift.
    """
    recs = list(records)
    if {record.evidence_id for record in recs} != set(freeze.evidence_ids):
        raise FreezeIntegrityError(f"freeze {freeze.freeze_id}: evidence id set drifted")
    if freeze_content_hash(recs) != freeze.content_hash:
        raise FreezeIntegrityError(f"freeze {freeze.freeze_id}: content_hash mismatch")
    if freeze.datapoint_hash is None:
        if any(_has_datapoint(record) for record in recs):
            raise FreezeIntegrityError(f"freeze {freeze.freeze_id}: legacy freeze rows gained datapoint fields")
    elif freeze_datapoint_hash(recs) != freeze.datapoint_hash:
        raise FreezeIntegrityError(f"freeze {freeze.freeze_id}: datapoint_hash mismatch")


def freeze_to_dict(freeze: EvidenceFreeze) -> dict[str, object]:
    """Freeze -> JSON-able dict (datetimes as ISO)."""
    return {
        "freeze_id": freeze.freeze_id,
        "session_id": freeze.session_id,
        "wave_id": freeze.wave_id,
        "created_at": freeze.created_at.isoformat(),
        "as_of": freeze.as_of.isoformat() if freeze.as_of else None,
        "evidence_ids": list(freeze.evidence_ids),
        "content_hash": freeze.content_hash,
        "datapoint_hash": freeze.datapoint_hash,
    }


def _narrow_wave(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FreezeIntegrityError(f"freeze: 'wave_id' must be an int >= 1, got {value!r}")
    return value


def _narrow_opt_time(value: object, key: str) -> datetime | str | None:
    if value is None or isinstance(value, (datetime, str)):
        return value
    raise FreezeIntegrityError(f"freeze: '{key}' must be a datetime, ISO-8601 string, or null")


def _freeze_ids_triplet(d: dict[str, object]) -> tuple[str, str, str]:
    freeze_id = d.get("freeze_id")
    session_id = d.get("session_id")
    content_hash = d.get("content_hash")
    if not isinstance(freeze_id, str) or not freeze_id:
        raise FreezeIntegrityError("freeze: 'freeze_id' must be a non-empty string")
    if not isinstance(session_id, str) or not session_id:
        raise FreezeIntegrityError("freeze: 'session_id' must be a non-empty string")
    if not isinstance(content_hash, str) or not content_hash:
        raise FreezeIntegrityError("freeze: 'content_hash' must be a non-empty string")
    return freeze_id, session_id, content_hash


def _freeze_evidence_tuple(d: dict[str, object]) -> tuple[str, ...]:
    evidence_ids = d.get("evidence_ids", [])
    if not isinstance(evidence_ids, (list, tuple)) or any(not isinstance(x, str) for x in evidence_ids):
        raise FreezeIntegrityError("freeze: 'evidence_ids' must be a list of strings")
    return tuple(evidence_ids)


def _freeze_datapoint_field(d: dict[str, object]) -> str | None:
    """datapoint_hash; absent or null reads as a legacy freeze."""
    raw = d.get("datapoint_hash")
    if raw is None or (isinstance(raw, str) and raw):
        return raw
    raise FreezeIntegrityError("freeze: 'datapoint_hash' must be a non-empty string or null")


def freeze_from_dict(data: Mapping[str, object]) -> EvidenceFreeze:
    """Rebuild a freeze; missing timestamps stay None, never invented."""
    d = dict(data)
    freeze_id, session_id, content_hash = _freeze_ids_triplet(d)
    return EvidenceFreeze(
        freeze_id=freeze_id,
        session_id=session_id,
        wave_id=_narrow_wave(d.get("wave_id")),
        created_at=_coerce_time(_narrow_opt_time(d.get("created_at"), "created_at"), "created_at") or utcnow(),
        as_of=_coerce_time(_narrow_opt_time(d.get("as_of"), "as_of"), "as_of"),
        evidence_ids=_freeze_evidence_tuple(d),
        content_hash=content_hash,
        datapoint_hash=_freeze_datapoint_field(d),
    )
