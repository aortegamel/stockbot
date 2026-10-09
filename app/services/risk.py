"""Mandate evaluation glue: snapshot + provider sectors.

Loads the mandate file and the latest persisted portfolio snapshot, builds
the entity -> sector map from the provider-supplied sector field (SEC
submissions), then delegates the pure math to the domain evaluator.
Nothing is persisted; evaluations are computed on demand.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..domain.portfolio import Position
from ..domain.risk.evaluation import RiskEvaluation, evaluate_mandate
from .mandate import load_mandate_file
from .portfolio_sync import read_latest_snapshot


def _cik_of(entity_id: str | None) -> int | None:
    """CIK int for SEC entity ids (None when absent or not an SEC identity)."""
    if not isinstance(entity_id, str) or not entity_id.startswith("sec:cik:"):
        return None
    try:
        return int(entity_id.split(":")[-1])
    except ValueError:
        return None


def _provider_sector(cik: int) -> str | None:
    """Provider-supplied sector for one CIK (None when unavailable)."""
    from ..sec.client import get_submissions_metadata

    meta = get_submissions_metadata(cik)
    if not isinstance(meta, dict):
        return None
    for key in ("sic_description", "sic"):
        sector = meta.get(key)
        if isinstance(sector, str) and sector.strip():
            return sector.strip()
    return None


def load_sector_map(positions: Iterable[Position]) -> dict[str, str]:
    """Entity -> sector from live SEC submissions.

    Entities without an SEC identity or without a provider sector stay
    unmapped (unknown exposure, as before). A provider failure propagates;
    evaluation requires the sector lookup to succeed.
    """
    sectors: dict[str, str] = {}
    seen: set[str] = set()
    for position in positions:
        entity_id = position.entity_id
        if entity_id is None or entity_id in seen:
            continue
        seen.add(entity_id)
        cik = _cik_of(entity_id)
        if cik is None:
            continue
        sector = _provider_sector(cik)
        if sector is not None:
            sectors[entity_id] = sector
    return sectors


def evaluate_latest_mandate(mandate_path: Path, data_root: Path | None = None) -> RiskEvaluation:
    """Load mandate + latest snapshot + sector map and evaluate.

    Raises FileNotFoundError (mandate missing, no snapshot) or ValueError
    (bad mandate).
    """
    mandate = load_mandate_file(mandate_path)
    snapshot = read_latest_snapshot(data_root=data_root)
    if snapshot is None:
        raise FileNotFoundError(
            "no persisted portfolio snapshot; run a portfolio sync (get_portfolio_snapshot with refresh) first"
        )
    sector_map = load_sector_map(snapshot.positions)
    return evaluate_mandate(snapshot, mandate, sector_map)
