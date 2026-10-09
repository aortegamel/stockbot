"""Normalized Google-discovery candidate signals (local evidence only).

Seam: live reads via SourceGateway + normalization + raw_archive (write-once) + write_bundle; NOTE: a future warehouse slots in behind these live readers, never inside normalization.

A candidate is a discovery pointer, never an investment thesis: it carries
source identity, provenance timestamps and evidence references. Features are
list-based only (persistence/diffusion/rank moves); no search-volume velocity,
no materiality scores. Missing observations are missing, never zero.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from operator import itemgetter

from ._guards import as_dict, as_int, as_list, as_str_list

_PERIOD_KEYS = ("source_period", "period", "week", "observed_at", "date")
_GEO_KEYS = ("geo", "geography")
_RANK_KEYS = ("rank", "position")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _pick(item: dict[str, object], keys: Sequence[str], default: object = None) -> object:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return default


def _is_present(item: dict[str, object]) -> bool:
    if not item.get("present", True):
        return False
    if item.get("absent", False) or item.get("missing", False):
        return False
    for key in ("term", "topic", "query", "keyword"):
        if key in item and item[key] is None:
            return False
    return True


def _rank_group_key(item: dict[str, object]) -> tuple[object, object, object]:
    return (
        item.get("table") or item.get("source_table"),
        item.get("list_kind") or item.get("list"),
        _pick(item, _GEO_KEYS),
    )


def _rank_row(item: dict[str, object]) -> tuple[str, int] | None:
    rank = _pick(item, _RANK_KEYS)
    period = _pick(item, _PERIOD_KEYS)
    if rank is None or period is None:
        return None
    try:
        rank = as_int(rank, what="rank")
    except TypeError, ValueError:
        return None
    return (str(period), rank)


def _best_rank_series(
    groups: dict[tuple[object, object, object], list[tuple[str, int]]],
) -> list[tuple[str, int]] | None:
    best = None
    for rows in groups.values():
        rows.sort(key=itemgetter(0))
        if len(rows) >= 2 and (best is None or len(rows) > len(best)):
            best = rows
    return best


def _rank_improvement(present: list[dict[str, object]]) -> int | None:
    groups: dict[tuple[object, object, object], list[tuple[str, int]]] = {}
    for item in present:
        parsed = _rank_row(item)
        if parsed is None:
            continue
        groups.setdefault(_rank_group_key(item), []).append(parsed)
    best = _best_rank_series(groups)
    if not best:
        return None
    return best[-2][1] - best[-1][1]


CALC_VERSION = "2"

# Informational thresholds only; candidates remain status="candidate" regardless.
RULES = {"velocity_min": None, "persistence_min": 0.5, "diffusion_min": 0.25}


def _score_of(item: dict[str, object]) -> float | None:
    score = item.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        mscore = metrics.get("score")
        if isinstance(mscore, (int, float)) and not isinstance(mscore, bool):
            return float(mscore)
    return None


def _blob_coverage(blob: dict[str, object], primary: str, fallback: str, what: str) -> list[str] | None:
    raw = blob.get(primary) or blob.get(fallback)
    if raw is None:
        return None
    return as_str_list(raw, what=what)


def _blob_observations(blob: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in as_list(blob.get("observations", []), what="observations"):
        if isinstance(raw, dict):
            rows.append(as_dict(raw, what="observations"))
    return rows


def _narrow_blob_coverage(
    blob: dict[str, object], periods_covered: Sequence[str] | None, geos_covered: Sequence[str] | None
) -> tuple[Sequence[str] | None, Sequence[str] | None, list[dict[str, object]]]:
    if periods_covered is None:
        parsed = _blob_coverage(blob, "periods_covered", "covered_periods", "periods_covered")
        if parsed is not None:
            periods_covered = parsed
    if geos_covered is None:
        parsed = _blob_coverage(blob, "geos_covered", "covered_geos", "geos_covered")
        if parsed is not None:
            geos_covered = parsed
    return periods_covered, geos_covered, _blob_observations(blob)


def _row_coverage(
    rows: list[dict[str, object]], periods_covered: Sequence[str] | None, geos_covered: Sequence[str] | None
) -> tuple[Sequence[str] | None, Sequence[str] | None]:
    for row in rows:
        if periods_covered is None and isinstance(row.get("periods_covered"), list):
            periods_covered = as_str_list(row.get("periods_covered"), what="periods_covered")
        if geos_covered is None and isinstance(row.get("geos_covered"), list):
            geos_covered = as_str_list(row.get("geos_covered"), what="geos_covered")
    return periods_covered, geos_covered


def _coerce_rows(
    observations: dict[str, object] | Sequence[dict[str, object]] | None,
    periods_covered: Sequence[str] | None,
    geos_covered: Sequence[str] | None,
) -> tuple[list[dict[str, object]], Sequence[str] | None, Sequence[str] | None]:
    if isinstance(observations, dict):
        periods_covered, geos_covered, observations = _narrow_blob_coverage(observations, periods_covered, geos_covered)
    rows: list[dict[str, object]] = [o for o in (observations or []) if isinstance(o, dict)]
    return (rows, *_row_coverage(rows, periods_covered, geos_covered))


def _distinct_sorted(rows: list[dict[str, object]], keys: Sequence[str]) -> list[str]:
    return sorted({str(_pick(o, keys)) for o in rows if _pick(o, keys) is not None})


def _covered_lists(
    rows: list[dict[str, object]], periods_covered: Sequence[str] | None, geos_covered: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    covered_periods = list(periods_covered) if periods_covered else _distinct_sorted(rows, _PERIOD_KEYS)
    covered_geos = list(geos_covered) if geos_covered else _distinct_sorted(rows, _GEO_KEYS)
    return covered_periods, covered_geos


def _scored_history(present: list[dict[str, object]]) -> list[tuple[str, float]]:
    pairs: set[tuple[str, float]] = set()
    for o in present:
        period = _pick(o, _PERIOD_KEYS)
        score = _score_of(o)
        if period is not None and score is not None:
            pairs.add((str(period), score))
    return sorted(pairs, key=itemgetter(0))


def _velocity_of(scored: list[tuple[str, float]], covered_periods: list[str], missing: dict[str, str]) -> float | None:
    if not covered_periods:
        missing["velocity"] = "no coverage"
        return None
    if len(scored) < 2:
        missing["velocity"] = "missing score history"
        return None
    return (scored[-1][1] - scored[0][1]) / len(covered_periods)


def _score_halves(scored: list[tuple[str, float]], covered_periods: list[str]) -> tuple[list[float], list[float]]:
    n_periods = len(covered_periods)
    edge = covered_periods[min(max(1, n_periods // 2), n_periods - 1)]
    prior = [s for p, s in scored if p < edge]
    recent = [s for p, s in scored if p >= edge]
    return prior, recent


def _acceleration_of(
    scored: list[tuple[str, float]], covered_periods: list[str], missing: dict[str, str]
) -> float | None:
    if len(scored) < 3 or len(covered_periods) < 2:
        if "velocity" not in missing:
            missing["acceleration"] = "missing score history"
        return None
    prior, recent = _score_halves(scored, covered_periods)
    if len(prior) < 2 or len(recent) < 2:
        missing["acceleration"] = "missing score history"
        return None
    return (recent[-1] - recent[0]) / len(recent) - (prior[-1] - prior[0]) / len(prior)


def _percentile_of(scored: list[tuple[str, float]], missing: dict[str, str]) -> float | None:
    if not scored:
        missing["percentile"] = "missing score history"
        return None
    last = scored[-1][1]
    return sum(1 for _, s in scored if s <= last) / len(scored)


def _assemble_values(
    covered_periods: list[str],
    covered_geos: list[str],
    hit_periods: set[str],
    hit_geos: set[str],
    present: list[dict[str, object]],
    velocity: float | None,
    acceleration: float | None,
    percentile: float | None,
    missing: dict[str, str],
) -> dict[str, object]:
    latest = max(covered_periods) if covered_periods else None
    first_seen = min(hit_periods) if hit_periods else None
    values: dict[str, object] = {
        "persistence": (len(hit_periods) / len(covered_periods)) if covered_periods else 0.0,
        "diffusion": (len(hit_geos) / len(covered_geos)) if covered_geos else 0.0,
        "rank_improvement": _rank_improvement(present),
        "velocity": velocity,
        "acceleration": acceleration,
        "percentile": percentile,
        "new_entry": bool(hit_periods) and first_seen == latest,
    }
    values["rules"] = {
        name: {"value": value, "rule": f"trend_{name}_v{CALC_VERSION}", "calc_version": CALC_VERSION}
        for name, value in values.items()
    }
    values["coverage"] = {"periods_covered": covered_periods, "geos_covered": covered_geos, "missing": missing}
    return values


def compute_candidate_features(
    observations: dict[str, object] | Sequence[dict[str, object]] | None,
    periods_covered: Sequence[str] | None = None,
    geos_covered: Sequence[str] | None = None,
) -> dict[str, object]:
    """persistence = distinct hit periods / covered periods; diffusion likewise.

    Coverage may be passed explicitly, ride inside the batch
    (``periods_covered``/``geos_covered`` keys), or fall back to the distinct
    periods/geos seen. Absent markers (present=False/absent/missing/term=None)
    count toward coverage but never toward hits. Rank improvement needs two
    ranked observations in the same table/list/geo, else None. Score-based
    metrics (velocity/acceleration/percentile) yield None plus a coverage
    reason when history is missing, never zero. No composite score.
    """
    rows, periods_covered, geos_covered = _coerce_rows(observations, periods_covered, geos_covered)
    covered_periods, covered_geos = _covered_lists(rows, periods_covered, geos_covered)
    present = [o for o in rows if _is_present(o)]
    hit_periods = {str(_pick(o, _PERIOD_KEYS)) for o in present if _pick(o, _PERIOD_KEYS) is not None}
    hit_geos = {str(_pick(o, _GEO_KEYS)) for o in present if _pick(o, _GEO_KEYS) is not None}
    missing: dict[str, str] = {}
    scored = _scored_history(present)
    velocity = _velocity_of(scored, covered_periods, missing)
    acceleration = _acceleration_of(scored, covered_periods, missing)
    percentile = _percentile_of(scored, missing)
    return _assemble_values(
        covered_periods, covered_geos, hit_periods, hit_geos, present, velocity, acceleration, percentile, missing
    )


def _apply_candidate_overrides(
    record: dict[str, object],
    *,
    table: object = None,
    period: object = None,
    geo: object = None,
    term: object = None,
    list_kind: object = None,
    rank: object = None,
) -> dict[str, object]:
    for key, value in (
        ("table", table),
        ("period", period),
        ("geo", geo),
        ("term", term),
        ("list_kind", list_kind),
        ("rank", rank),
    ):
        if value is not None:
            record[key] = value
    return record


def _candidate_identity(record: dict[str, object]) -> tuple[object, object, object, object, object, object, str]:
    missing = [k for k in ("table", "geo", "term", "list_kind") if record.get(k) is None]
    if missing:
        raise TypeError(f"normalize_candidate missing required fields: {missing}")
    table, geo, term = record["table"], record["geo"], record["term"]
    list_kind = record["list_kind"]
    period = record.get("period") or record.get("observed_at")
    rank = record.get("rank")
    signal_id = hashlib.sha256(f"{table}|{period}|{geo}|{term}|{list_kind}".encode()).hexdigest()
    return table, geo, term, list_kind, period, rank, signal_id


def _merged_candidate_metrics(record: dict[str, object], metrics: object, rank: object) -> dict[str, object]:
    merged: dict[str, object] = dict(as_dict(record.get("metrics", None) or metrics or {}, what="metrics"))
    if rank is not None and "rank" not in merged:
        merged["rank"] = rank
    return merged


def _candidate_list_field(record: dict[str, object], name: str, fallback: object, what: str) -> list[object]:
    value = record.get(name, None)
    if value is not None:
        return as_list(value, what=what)
    return as_list(fallback or [], what=what)


def _first_truthy(*values: object) -> object:
    for value in values:
        if value:
            return value
    return values[-1] if values else None


def _candidate_features_field(record: dict[str, object], features: object) -> object:
    if record.get("features", None) is not None:
        return record.get("features", None)
    return features


def _build_candidate(
    record: dict[str, object],
    table: object,
    geo: object,
    term: object,
    list_kind: object,
    period: object,
    signal_id: str,
    signal_type: object,
    source: object,
    source_record_id: object,
    observed_at: object,
    retrieved_at: object,
    known_at: object,
    now: str,
    merged_metrics: dict[str, object],
    entities_list: list[object],
    evidence_list: list[object],
    features_value: object,
) -> dict[str, object]:
    return {
        "signal_id": signal_id,
        "status": "candidate",
        "signal_type": record.get("signal_type", signal_type),
        "source": record.get("source", source),
        "source_record_id": _first_truthy(record.get("source_record_id"), source_record_id, signal_id),
        "observed_at": _first_truthy(record.get("observed_at"), observed_at, period),
        "known_at": _first_truthy(known_at, record.get("known_at"), now),
        "retrieved_at": _first_truthy(retrieved_at, record.get("retrieved_at"), now),
        "term": term,
        "geo": geo,
        "table": table,
        "list_kind": list_kind,
        "period": period,
        "metrics": merged_metrics,
        "entities": entities_list,
        "evidence": evidence_list,
        "features": features_value,
    }


def normalize_candidate(
    record: dict[str, object] | None = None,
    *,
    table: object = None,
    period: object = None,
    geo: object = None,
    term: object = None,
    list_kind: object = None,
    rank: object = None,
    signal_type: object = "trend",
    source: object = "trends",
    source_record_id: object = None,
    observed_at: object = None,
    retrieved_at: object = None,
    known_at: object = None,
    entities: object = None,
    metrics: object = None,
    evidence: object = None,
    features: object = None,
) -> dict[str, object]:
    """Build one ephemeral candidate; identity is stable.

    Accepts either a record dict (keys table/period/geo/term/list_kind/rank)
    or the same fields as keywords; explicit keywords win. Never persisted:
    the caller logs the model-visible output to the run bundle.
    """
    record = _apply_candidate_overrides(
        dict(record or {}), table=table, period=period, geo=geo, term=term, list_kind=list_kind, rank=rank
    )
    table, geo, term, list_kind, period, rank, signal_id = _candidate_identity(record)
    return _build_candidate(
        record,
        table,
        geo,
        term,
        list_kind,
        period,
        signal_id,
        signal_type,
        source,
        source_record_id,
        observed_at,
        retrieved_at,
        known_at,
        _now_iso(),
        _merged_candidate_metrics(record, metrics, rank),
        _candidate_list_field(record, "entities", entities, "entities"),
        _candidate_list_field(record, "evidence", evidence, "evidence"),
        _candidate_features_field(record, features),
    )


# Seam: ephemeral per-collection compute via collect_trends + normalization; caller logs to the run bundle + raw_archive; NOTE: warehouse slots behind live readers.
