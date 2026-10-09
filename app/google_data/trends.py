"""Google Trends discovery evidence via bounded BigQuery templates only.

Daily top/rising tables (US + international) through the shared bounded
executor: checked-in SELECTs with ``refresh_date`` bounds for partition
pruning and an optional ``week`` pushdown for long lookbacks. ``refresh_date``
is load provenance (retrieval/knowledge time); ``week`` is the observed
interest week (analysis time). Scores/ranks are comparable only within one
refresh/week/geo/granularity — analytics stays list-membership. No hourly
template exists; hourly requests fail closed. No scraping, no paid fallback.
Empty results are a valid empty batch; malformed rows fail that batch with an
actionable error instead of inventing observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..storage.raw_archive import ArchiveRecord

from ._guards import as_dict, as_list, as_str_list, result_rows
from ._lazy_config import google_data_enabled

SOURCE = "trends"
_US_TEMPLATES = ("trends_us_top", "trends_us_rising")
_INTL_TEMPLATES = ("trends_intl_top", "trends_intl_rising")
_US_NATIONAL_TEMPLATES = ("trends_us_top_national", "trends_us_rising_national")
_INTL_NATIONAL_TEMPLATES = ("trends_intl_top_national", "trends_intl_rising_national")
_TEMPLATE_LIST_KIND = {
    "trends_us_top": "top",
    "trends_us_rising": "rising",
    "trends_intl_top": "top",
    "trends_intl_rising": "rising",
    "trends_us_top_national": "top",
    "trends_us_rising_national": "rising",
    "trends_intl_top_national": "top",
    "trends_intl_rising_national": "rising",
    "trends_top": "top",
    "trends_rising": "rising",
}
_FALLBACK_TABLES = {
    "trends_us_top": "bigquery-public-data.google_trends.top_terms",
    "trends_us_rising": "bigquery-public-data.google_trends.top_rising_terms",
    "trends_intl_top": "bigquery-public-data.google_trends.international_top_terms",
    "trends_intl_rising": "bigquery-public-data.google_trends.international_top_rising_terms",
    "trends_us_top_national": "bigquery-public-data.google_trends.top_terms",
    "trends_us_rising_national": "bigquery-public-data.google_trends.top_rising_terms",
    "trends_intl_top_national": "bigquery-public-data.google_trends.international_top_terms",
    "trends_intl_rising_national": "bigquery-public-data.google_trends.international_top_rising_terms",
}
_COLLECTOR_VERSION = "1"
_SQL_VERSION = "1"
_MAX_LIMIT = 1000
_FETCH_LIMIT = _MAX_LIMIT + 1
_MAX_GEOS = 50
_MAX_SPAN_DAYS = 31
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _data_enabled() -> bool:
    return google_data_enabled()


def _bq_ready() -> bool:
    if not google_data_enabled():
        return False
    try:
        from .. import config as _cfg
    except ImportError:
        _cfg = None
    _project: str | None = None
    if _cfg is not None:
        try:
            _project = _cfg.get_google_cloud_project()
        except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
            _project = None
    if _project is None:
        _project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or None
    return bool(_project)


def _bq_env_int(name: str, default: int) -> int:
    return int((os.getenv(name) or "").strip() or default)


def _bq_limits_from_env() -> tuple[int, int, int]:
    return (
        _bq_env_int("BIGQUERY_MAX_BYTES_PER_QUERY", 1073741824),
        _bq_env_int("BIGQUERY_MONTHLY_BYTES_LIMIT", 536870912000),
        _bq_env_int("BIGQUERY_DAILY_BYTES_LIMIT", 10737418240),
    )


def _env_bq_limits() -> tuple[int, int, int] | None:
    try:
        return _bq_limits_from_env()
    except TypeError, ValueError:
        return None


def _invalid_bq_limit_error() -> dict[str, object]:
    return {"status": "error", "source": SOURCE, "error": "invalid BigQuery byte limit", "error_type": "invalid_config"}


def _nonpositive_bq_limit_error() -> dict[str, object]:
    return {
        "status": "error",
        "source": SOURCE,
        "error": "non-positive BigQuery byte limit",
        "error_type": "invalid_config",
    }


def _bq_limits_positive(per_q: int, per_m: int, per_d: int) -> bool:
    return per_q > 0 and per_m > 0 and per_d > 0


def _bq_limits_from_config() -> tuple[int, int, int] | dict[str, object] | None:
    """Config limits, invalid-config error, or None when env fallback applies."""
    try:
        from .. import config as _cfg2
    except ImportError:
        return None
    try:
        return (
            _cfg2.get_bq_max_bytes_per_query(),
            _cfg2.get_bq_monthly_bytes_limit(),
            _cfg2.get_bq_daily_bytes_limit(),
        )
    except TypeError, ValueError:
        return _invalid_bq_limit_error()
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _check_bq_limits() -> dict[str, object] | None:
    vals: tuple[int, int, int] | dict[str, object] | None = _bq_limits_from_config()
    if isinstance(vals, dict):
        return vals
    if vals is None:
        vals = _env_bq_limits()
        if vals is None:
            return _invalid_bq_limit_error()
    per_q, per_m, per_d = vals
    if not _bq_limits_positive(per_q, per_m, per_d):
        return _nonpositive_bq_limit_error()
    return None


class _Submitter(Protocol):
    """Anything submit_template-compatible: the real client or a test double."""

    def submit_template(self, template: str, params: dict[str, object]) -> dict[str, object]: ...


_Executor = Callable[[str, dict[str, object]], dict[str, object]] | _Submitter


def _submit_via_client(template: str, params: dict[str, object], data_root: Path | str | None) -> dict[str, object]:
    try:
        from . import bigquery_client as _client
    except ImportError:
        return {"error": "bigquery client unavailable", "error_type": "source_unavailable", "source": "bigquery"}
    try:
        narrowed_root: Path | None = Path(data_root) if isinstance(data_root, str) else data_root
        return _client.submit_template(template, params, data_root=narrowed_root)
    except Exception as exc:
        if type(exc).__name__ == "LedgerCorrupt":
            raise
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"{SOURCE} query failed: {exc}",
            "error_type": "executor_error",
        }


def _direct_submit(template: str, params: dict[str, object], executor: _Executor) -> dict[str, object]:
    try:
        if callable(executor):
            return executor(template, params)
        return executor.submit_template(template, params)
    except Exception as exc:
        if type(exc).__name__ == "LedgerCorrupt":
            raise
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"{SOURCE} query failed: {exc}",
            "error_type": "executor_error",
        }


def _submit(
    template: str, params: dict[str, object], executor: _Executor | None, data_root: Path | str | None
) -> dict[str, object]:
    if executor is None:
        return _submit_via_client(template, params, data_root)
    return _direct_submit(template, params, executor)


_UNAVAILABLE = frozenset(
    {
        "billing_enabled",
        "billing_unknown",
        "cost_limit_exceeded",
        "monthly_limit_exceeded",
        "daily_limit_exceeded",
        "source_unavailable",
        "ledger_corrupt",
    }
)


def _wrap_error(result: dict[str, object]) -> dict[str, object]:
    if "status" in result:
        return result
    out = dict(result)
    out["status"] = "unavailable" if out.get("error_type") in _UNAVAILABLE else "error"
    out["engine"] = out.get("source", "bigquery")
    out["source"] = SOURCE
    return out


def _template_table(template: str) -> str:
    try:
        from . import bigquery_client as _bq
    except ImportError:
        return _FALLBACK_TABLES.get(template, "trends")
    try:
        spec = _bq.TEMPLATES.get(template, {})
        table: object = spec.get("table")
        if isinstance(table, str) and table:
            return table
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass
    return _FALLBACK_TABLES.get(template, "trends")


def _is_country_code(value: str) -> bool:
    return len(value) == 2 and value.isalpha() and value.isupper()


def _split_geos(geos: list[str]) -> tuple[list[str], list[str]]:
    dmas = [g for g in geos if g != "US" and not _is_country_code(g)]
    countries = [g for g in geos if g != "US" and _is_country_code(g)]
    return dmas, countries


def _plan_groups(geos: list[str]) -> list[dict[str, object]]:
    """Split geos into a US group (national or DMA names) plus per-country groups."""
    dmas, countries = _split_geos(geos)
    groups: list[dict[str, object]] = []
    if "US" in geos:
        groups.append({"kind": "us", "national": True, "dmas": []})
    if dmas:
        groups.append({"kind": "us", "national": False, "dmas": dmas})
    for country in countries:
        groups.append({"kind": "intl", "country": country})
    if not groups:  # e.g. geos == [] handled earlier; defensive: treat as national
        groups.append({"kind": "us", "national": True, "dmas": []})
    return groups


def _observation_identity(
    table: object,
    period: object,
    geo: object,
    term: object,
    list_kind: object,
    metrics: dict[str, object],
    evidence: list[object],
) -> tuple[str, str]:
    """Durable (observation_id, content_hash) identity shared by store and verify."""
    content_hash = hashlib.sha256(
        json.dumps(
            {"metrics": metrics, "evidence": evidence, "collector_version": _COLLECTOR_VERSION},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
    observation_id = hashlib.sha256(f"{SOURCE}|{table}|{period}|{geo}|{term}|{list_kind}".encode()).hexdigest()
    return observation_id, content_hash


def _feature_scope_json(
    *, week_start: str, week_end: str, geos: list[str], table: str, list_kind: str
) -> dict[str, object]:
    return {
        "week_start": week_start,
        "week_end": week_end,
        "geos": sorted(geos),
        "table": table,
        "list_kind": list_kind,
    }


def _feature_scope_hash(scope: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _series_basis(staged_row: dict[str, object]) -> str:
    metrics_raw: object = staged_row.get("metrics")
    metrics: dict[str, object] = metrics_raw if isinstance(metrics_raw, dict) else {}
    basis = metrics.get("score_basis")
    if basis:
        return str(basis)
    if (
        staged_row.get("dma_count") is not None
        or staged_row.get("region_count") is not None
        or metrics.get("dma_count") is not None
        or metrics.get("region_count") is not None
    ):
        return "mean_list_score_where_listed"
    return ""


def _archive_raw(
    data_root: Path | str | None, template: str, job_id: str, table: str, params: dict[str, object], rows: list[object]
) -> None:
    if data_root is None:
        return
    try:
        from ..storage import raw_archive as _raw_archive
    except ImportError:
        return
    try:
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode()
        _raw_archive.archive(
            "google",
            kind=template,
            key=job_id,
            payload=payload,
            url=f"bq://{table}",
            metadata={"params": params},
            root=Path(data_root) / "raw",
        )
    except Exception:  # noqa: BLE001, S110 - intentional best-effort boundary, never aborts
        pass


def _archive_root(data_root: Path | str | None) -> Path | None:
    if data_root is None:
        return None
    return Path(data_root) / "raw"


def _decode_archive_payload(metadata: object, raw: bytes, params: dict[str, object]) -> list[object] | None:
    try:
        if not isinstance(metadata, dict) or metadata.get("params") != params:
            return None
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None
    if not isinstance(payload, list):
        return None
    return payload


def _archive_record_payload(record: ArchiveRecord, params: dict[str, object]) -> list[object] | None:
    try:
        return _decode_archive_payload(record.metadata, record.payload_path.read_bytes(), params)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _best_archive_payload(records: Iterable[ArchiveRecord], params: dict[str, object]) -> list[object] | None:
    best: list[object] | None = None
    for record in records:
        payload = _archive_record_payload(record, params)
        if payload is None:
            continue
        if best is None or len(payload) > len(best):
            best = payload
    return best


def _archived_rows(
    data_root: Path | str | None, template: str, job_id: str, params: dict[str, object]
) -> list[object] | None:
    root = _archive_root(data_root)
    if root is None:
        return None
    try:
        from ..storage import raw_archive as _raw_archive_r
    except ImportError:
        return None
    try:
        return _best_archive_payload(_raw_archive_r.iter_archive("google", template, job_id, root=root), params)
    except Exception:  # noqa: BLE001 - intentional best-effort boundary, never aborts
        return None


def _enumerate_refreshes(
    table: str, start_date: str, end_date: str, executor: _Executor | None, data_root: Path | str | None
) -> list[str] | None:
    """Known refresh_date partitions for one table; None when the executor errors."""
    result = _submit(
        "trends_refreshes",
        {
            "table": table,
            "start_date": start_date,
            "end_date": end_date,
            "limit": _MAX_LIMIT,
            "collector_version": _COLLECTOR_VERSION,
            "sql_version": _SQL_VERSION,
        },
        executor,
        data_root,
    )
    if not isinstance(result, dict) or "error" in result:
        return None
    refreshes = sorted(
        {str(r.get("refresh_date")) for r in result_rows(result) if isinstance(r, dict) and r.get("refresh_date")}
    )
    return refreshes


def _normalize_geos(geos: list[str] | str) -> list[str]:
    if isinstance(geos, str):
        return [geos]
    return list(geos or [])


def _readiness_error() -> dict[str, object] | None:
    if not _bq_ready():
        return {"status": "disabled", "source": SOURCE, "reason": "google data disabled or no BigQuery project"}
    return _check_bq_limits()


def _interval_error(interval: str | None) -> dict[str, object] | None:
    if (interval or "daily") == "hourly":
        return {
            "status": "unavailable",
            "source": SOURCE,
            "error": "hourly trends tables are not allowlisted",
            "error_type": "source_unavailable",
            "coverage": {"reason": "hourly_unavailable"},
        }
    if (interval or "daily") != "daily":
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"invalid interval: {interval!r} (daily)",
            "error_type": "invalid_params",
        }
    return None


def _dates_error(start_date: str | None, end_date: str | None) -> dict[str, object] | None:
    for label, value in (("start_date", start_date), ("end_date", end_date)):
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return {
                "status": "error",
                "source": SOURCE,
                "error": f"invalid {label}: {value!r} (YYYY-MM-DD)",
                "error_type": "invalid_params",
            }
    assert isinstance(start_date, str) and isinstance(end_date, str)
    if start_date > end_date:
        return {
            "status": "error",
            "source": SOURCE,
            "error": "start_date after end_date",
            "error_type": "invalid_params",
        }
    span = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    if span > _MAX_SPAN_DAYS:
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"date span {span}d exceeds {_MAX_SPAN_DAYS}d refresh bound",
            "error_type": "invalid_params",
        }
    return None


def _week_params_error(week_start: str | None, week_end: str | None) -> dict[str, object] | None:
    for label, value in (("week_start", week_start), ("week_end", week_end)):
        if value is not None and (not isinstance(value, str) or not _DATE_RE.match(value)):
            return {
                "status": "error",
                "source": SOURCE,
                "error": f"invalid {label}: {value!r} (YYYY-MM-DD)",
                "error_type": "invalid_params",
            }
    if week_start and week_end and week_start > week_end:
        return {
            "status": "error",
            "source": SOURCE,
            "error": "week_start after week_end",
            "error_type": "invalid_params",
        }
    return None


def _geos_error(geos: list[str]) -> dict[str, object] | None:
    if not geos:
        return {
            "status": "error",
            "source": SOURCE,
            "error": "at least one geo is required",
            "error_type": "invalid_params",
        }
    if len(geos) > _MAX_GEOS:
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"{len(geos)} geos exceeds bound {_MAX_GEOS}",
            "error_type": "invalid_params",
        }
    return None


def _collect_params_error(
    interval: str | None,
    start_date: str | None,
    end_date: str | None,
    week_start: str | None,
    week_end: str | None,
    geos: list[str],
) -> dict[str, object] | None:
    checks = (
        _interval_error(interval),
        _dates_error(start_date, end_date),
        _week_params_error(week_start, week_end),
        _geos_error(geos),
    )
    for check in checks:
        if check is not None:
            return check
    return None


def _default_collect_weeks(end_date: str, week_start: str | None, week_end: str | None) -> tuple[str, str]:
    if week_start is None and week_end is None:
        week_end = end_date
        week_start = (date.fromisoformat(end_date) - timedelta(days=13)).isoformat()
    elif week_start is None and week_end is not None:
        week_start = (date.fromisoformat(week_end) - timedelta(days=13)).isoformat()
    elif week_start is not None and week_end is None:
        week_end = (date.fromisoformat(week_start) + timedelta(days=13)).isoformat()
    assert isinstance(week_start, str) and isinstance(week_end, str)
    return week_start, week_end


def _templates_for(group: dict[str, object]) -> tuple[str, ...]:
    if group["kind"] == "us":
        if group.get("national"):
            return _US_NATIONAL_TEMPLATES
        return _US_TEMPLATES
    return _INTL_NATIONAL_TEMPLATES


def _refresh_query_params(group: dict[str, object], refresh: str, week_start: str, week_end: str) -> dict[str, object]:
    base: dict[str, object] = {
        "start_date": refresh,
        "end_date": refresh,
        "week_start": week_start,
        "week_end": week_end,
        "limit": _FETCH_LIMIT,
        "collector_version": _COLLECTOR_VERSION,
        "sql_version": _SQL_VERSION,
    }
    if group["kind"] == "us" and group.get("national"):
        return {**base, "national": "US"}
    if group["kind"] == "us":
        return {**base, "dmas": sorted(as_str_list(group.get("dmas"), what="dmas")), "all_dmas": False}
    return {**base, "country_code": group["country"], "national": group["country"]}


def _submit_or_error(
    template: str, params: dict[str, object], executor: _Executor | None, data_root: Path | str | None
) -> tuple[dict[str, object] | None, dict[str, object]]:
    result = _submit(template, params, executor, data_root)
    if not isinstance(result, dict):
        return _wrap_error({"error": "bad executor result"}), {}
    if "error" in result:
        return _wrap_error(result), {}
    return None, result


def _resolve_rows(
    template: str, params: dict[str, object], data_root: Path | str | None, result: dict[str, object]
) -> tuple[dict[str, object] | None, list[object]]:
    cached = bool(result.get("cached"))
    rows = result_rows(result)
    job_id = result.get("job_id")
    if cached and not rows:
        recovered = _archived_rows(data_root, template, str(job_id or template), params)
        if recovered is None:
            return _wrap_error(
                {
                    "error": f"cached Trends result unavailable for {template}; no archived payload matches",
                    "error_type": "source_unavailable",
                    "source": "bigquery",
                }
            ), []
        return None, list(recovered)
    return None, rows


def _maybe_archive(
    data_root: Path | str | None,
    template: str,
    job_id: str,
    table: str,
    params: dict[str, object],
    rows: list[object],
    cached: bool,
) -> None:
    if data_root is not None and not cached:
        _archive_raw(data_root, template, job_id, table, params, rows)


def _scope_limit_error(template: str, refresh: str, count: int) -> dict[str, object] | None:
    if count <= _MAX_LIMIT:
        return None
    return {
        "status": "unavailable",
        "source": SOURCE,
        "reason": "query_scope_too_large",
        "error": f"{template} query scope exceeds 1000 rows for {refresh}",
        "error_type": "missing_coverage",
    }


def _intl_row_geo(row: dict[str, object], group: dict[str, object]) -> object:
    return row.get("country_code") or group["country"] or row.get("geo")


def _region_geo_base(row: dict[str, object], group: dict[str, object]) -> object:
    region = row.get("region_code") or ""
    geo = row.get("country_code") or group["country"]
    if region:
        return str(geo or "") + f":{region}"
    return geo


def _region_row_geo(row: dict[str, object], group: dict[str, object]) -> object:
    geo = _region_geo_base(row, group)
    if not row.get("country_code"):
        geo = row.get("geo") or geo
    return geo


def _row_geo(row: dict[str, object], template: str, group: dict[str, object]) -> object:
    if template in _US_NATIONAL_TEMPLATES:
        return "US"
    if template in _INTL_NATIONAL_TEMPLATES:
        return _intl_row_geo(row, group)
    if group["kind"] == "us":
        return row.get("dma_name") or row.get("geo")
    return _region_row_geo(row, group)


def _row_markers(staged: dict[str, object], template: str, table: str, refresh: str) -> None:
    staged["_table"] = staged.get("table") or table
    staged["_refresh"] = str(staged.get("refresh_date") or refresh)
    staged["_week"] = str(staged.get("week") or staged.get("period") or staged.get("source_period") or refresh)
    staged["_kind"] = staged.get("list_kind") or staged.get("list") or _TEMPLATE_LIST_KIND.get(template, "top")


def _tag_row(
    row: object, template: str, table: str, refresh: str, group: dict[str, object], job_id: object
) -> dict[str, object] | None:
    if not isinstance(row, dict):
        return None
    staged = dict(row)
    staged["_template"] = template
    _row_markers(staged, template, table, refresh)
    staged["_geo"] = _row_geo(staged, template, group)
    staged["job_id"] = job_id
    return staged


def _table_sets() -> tuple[set[str], set[str]]:
    return (
        {_template_table("trends_us_top"), _template_table("trends_us_rising"), "trends_top", "trends_rising"},
        {_template_table("trends_intl_top"), _template_table("trends_intl_rising")},
    )


def _dma_scope_set(groups: list[dict[str, object]]) -> set[str]:
    return {
        d
        for g in groups
        if g.get("kind") == "us" and not g.get("national")
        for d in as_str_list(g.get("dmas"), what="dmas")
    }


def _as_staged(row: dict[str, object]) -> dict[str, object]:
    """Staged row with normalized markers (existing boundary)."""
    staged = dict(row)
    staged.setdefault("_table", staged.get("table") or "trends")
    staged.setdefault("_week", staged.get("week") or staged.get("period") or "")
    staged.setdefault("_geo", staged.get("geo"))
    staged.setdefault("_kind", staged.get("list_kind") or staged.get("list") or "top")
    staged.setdefault("_refresh", staged.get("refresh_date") or staged.get("_week"))
    staged.setdefault("_template", staged.get("template"))
    return staged


def _route_staged(
    staged_row: dict[str, object],
    row: dict[str, object],
    table: object,
    us_tables: set[str],
    intl_tables: set[str],
    national: bool,
    dma_set: set[str],
    national_rows: list[dict[str, object]],
    dma_rows: list[dict[str, object]],
    intl_rows: list[dict[str, object]],
) -> None:
    if row.get("_template") in _US_NATIONAL_TEMPLATES + _INTL_NATIONAL_TEMPLATES:
        national_rows.append(staged_row)
    elif table in us_tables:
        if national and str(staged_row["_geo"]) == "US":
            national_rows.append(staged_row)
        elif str(staged_row["_geo"]) in dma_set:
            dma_rows.append(staged_row)
    elif table in intl_tables:
        intl_rows.append(staged_row)


def _split_staged(
    merged: dict[tuple[object, object, object, object, object, object], dict[str, object]],
    groups: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], set[str]]:
    us_tables, intl_tables = _table_sets()
    national = any(g.get("kind") == "us" and g.get("national") for g in groups)
    dma_set = _dma_scope_set(groups)
    national_rows: list[dict[str, object]] = []
    dma_rows: list[dict[str, object]] = []
    intl_rows: list[dict[str, object]] = []
    for row in merged.values():
        table = row.get("_table") or row.get("table")
        staged_row = _as_staged(row)
        if not staged_row.get("term") or not staged_row.get("_geo"):
            raise ValueError(f"malformed trends row for {staged_row.get('term')!r}")
        _route_staged(
            staged_row, row, table, us_tables, intl_tables, national, dma_set, national_rows, dma_rows, intl_rows
        )
    return national_rows, dma_rows, intl_rows, dma_set


def _series_metric(row: dict[str, object], name: str) -> object:
    value = row.get(name)
    if name == "score" and isinstance(value, bool):
        value = None
    if value is None:
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            value = metrics.get(name)
    return value


def _series_point(row: dict[str, object]) -> dict[str, object]:
    return {
        "table": row.get("_table"),
        "period": row.get("_week"),
        "geo": row.get("_geo"),
        "term": row.get("term"),
        "list_kind": row.get("_kind"),
        "rank": _series_metric(row, "rank"),
        "score": _series_metric(row, "score"),
    }


def _group_series(
    winners: dict[tuple[object, object, object, object, object], dict[str, object]],
) -> tuple[
    dict[tuple[object, object, object, object, str], list[dict[str, object]]],
    dict[tuple[object, object, object, str], set[str]],
]:
    by_series: dict[tuple[object, object, object, object, str], list[dict[str, object]]] = {}
    periods_by_series: dict[tuple[object, object, object, str], set[str]] = {}
    for row in winners.values():
        basis = _series_basis(row)
        skey = (row.get("term"), row.get("_table"), row.get("_kind"), row.get("_geo"), basis)
        by_series.setdefault(skey, []).append(_series_point(row))
        pkey = (row.get("_table"), row.get("_kind"), row.get("_geo"), basis)
        periods_by_series.setdefault(pkey, set()).add(str(row.get("_week")))
    return by_series, periods_by_series


def _pick_winners(
    rows: list[dict[str, object]],
) -> dict[tuple[object, object, object, object, object], dict[str, object]]:
    winners: dict[tuple[object, object, object, object, object], dict[str, object]] = {}
    for row in rows:
        wkey = (row.get("_table"), row.get("_kind"), row.get("_geo"), row.get("term"), row.get("_week"))
        prev = winners.get(wkey)
        if prev is None or str(row.get("_refresh") or "") > str(prev.get("_refresh") or ""):
            winners[wkey] = row
    return winners


def _series_feature_map(
    signals_mod: ModuleType,
    by_series: dict[tuple[object, object, object, object, str], list[dict[str, object]]],
    periods_by_series: dict[tuple[object, object, object, str], set[str]],
) -> dict[tuple[object, object, object, object, str], dict[str, object]]:
    series_features: dict[tuple[object, object, object, object, str], dict[str, object]] = {}
    for skey, srows in by_series.items():
        _term, _table, _kind, _geo, _basis = skey
        periods = sorted(periods_by_series.get((_table, _kind, _geo, _basis)) or set())
        features = signals_mod.compute_candidate_features(srows, periods_covered=periods)
        if isinstance(features, dict):
            series_features[skey] = features
    return series_features


def _apply_diffusion_entry(feat: dict[str, object], diff: dict[str, object]) -> None:
    feat["diffusion"] = diff.get("diffusion")
    rules: object = feat.get("rules")
    if isinstance(rules, dict):
        diff_rules: object = diff.get("rules", {})
        diff_entry: object = diff_rules.get("diffusion", {}) if isinstance(diff_rules, dict) else {}
        rules["diffusion"] = dict(as_dict(diff_entry, what="diffusion"))
    cov: object = feat.get("coverage")
    if isinstance(cov, dict):
        cov_diff: object = diff.get("coverage", {})
        cov_geos: object = cov_diff.get("geos_covered", []) if isinstance(cov_diff, dict) else []
        cov["geos_covered"] = as_list(cov_geos, what="geos_covered")


def _dma_group_key(row: dict[str, object]) -> tuple[object, object, object, str]:
    return (row.get("term"), row.get("_table"), row.get("_kind"), _series_basis(row))


def _collect_dma_groups(
    winners: dict[tuple[object, object, object, object, object], dict[str, object]], dma_set: set[str]
) -> dict[tuple[object, object, object, str], list[dict[str, object]]]:
    dma_groups: dict[tuple[object, object, object, str], list[dict[str, object]]] = {}
    for row in winners.values():
        if str(row.get("_geo")) not in dma_set:
            continue
        dma_groups.setdefault(_dma_group_key(row), []).append(_series_point(row))
    return dma_groups


def _dma_key_matches(
    skey: tuple[object, object, object, object, str], gkey: tuple[object, object, object, str], dma_set: set[str]
) -> bool:
    return (
        skey[0] == gkey[0]
        and skey[1] == gkey[1]
        and skey[2] == gkey[2]
        and skey[4] == gkey[3]
        and str(skey[3]) in dma_set
    )


def _spread_dma_diff(
    signals_mod: ModuleType,
    gkey: tuple[object, object, object, str],
    grows: list[dict[str, object]],
    dma_set: set[str],
    series_features: dict[tuple[object, object, object, object, str], dict[str, object]],
) -> None:
    raw = signals_mod.compute_candidate_features(grows, geos_covered=sorted(dma_set))
    if not isinstance(raw, dict):
        return
    diff = raw
    for skey, sentry in series_features.items():
        if _dma_key_matches(skey, gkey, dma_set):
            _apply_diffusion_entry(sentry, diff)


def _apply_dma_diffusion(
    signals_mod: ModuleType,
    winners: dict[tuple[object, object, object, object, object], dict[str, object]],
    dma_set: set[str],
    series_features: dict[tuple[object, object, object, object, str], dict[str, object]],
) -> None:
    if not dma_set:
        return
    for gkey, grows in _collect_dma_groups(winners, dma_set).items():
        _spread_dma_diff(signals_mod, gkey, grows, dma_set, series_features)


def _features_for(
    signals_mod: ModuleType,
    winners: dict[tuple[object, object, object, object, object], dict[str, object]],
    dma_set: set[str],
) -> dict[tuple[object, object, object, object, str], dict[str, object]]:
    by_series, periods_by_series = _group_series(winners)
    features = _series_feature_map(signals_mod, by_series, periods_by_series)
    _apply_dma_diffusion(signals_mod, winners, dma_set, features)
    return features


def _strip_diffusion(features: object) -> object:
    if not isinstance(features, dict):
        return features
    out = copy.deepcopy(features)
    out["diffusion"] = None
    arules: object = out["rules"]
    if isinstance(arules, dict):
        ardiff: object = arules.get("diffusion")
        if isinstance(ardiff, dict):
            ardiff["value"] = None
    acov: object = out["coverage"]
    if isinstance(acov, dict):
        amissing: object = acov.get("missing")
        if isinstance(amissing, dict):
            amissing["diffusion"] = "subregion rows aggregated"
    return out


def _normalize_staged(
    rows: list[dict[str, object]],
    series_features: dict[tuple[object, object, object, object, str], dict[str, object]],
    retrieved_at: str,
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for row in rows:
        basis = _series_basis(row)
        features = series_features.get((row.get("term"), row.get("_table"), row.get("_kind"), row.get("_geo"), basis))
        metrics_raw: object = row.get("metrics")
        metrics: dict[str, object] = metrics_raw if isinstance(metrics_raw, dict) else {}
        aggregate = (
            row.get("dma_count") is not None
            or row.get("region_count") is not None
            or metrics.get("dma_count") is not None
            or metrics.get("region_count") is not None
        )
        norm = _normalize_row(row, retrieved_at, features=_strip_diffusion(features) if aggregate else features)
        observations.append(norm)
    return observations


def _collect_one_live(
    template: str,
    table: str,
    refresh: str,
    params: dict[str, object],
    group: dict[str, object],
    merged: dict[tuple[object, object, object, object, object, object], dict[str, object]],
    used_templates: list[str],
    refresh_seen: set[str],
    executor: _Executor | None,
    data_root: Path | str | None,
) -> dict[str, object] | None:
    """Fetch one refresh live; archive the raw payload, merge staged rows."""
    refresh_seen.add(refresh)
    err, result = _submit_or_error(template, params, executor, data_root)
    if err is not None:
        return err
    cached = bool(result.get("cached"))
    resolve_err, rows = _resolve_rows(template, params, data_root, result)
    if resolve_err is not None:
        return resolve_err
    job_id = result.get("job_id")
    _maybe_archive(data_root, template, str(job_id or template), table, params, rows, cached)
    over = _scope_limit_error(template, refresh, len(rows))
    if over is not None:
        return over
    if template not in used_templates:
        used_templates.append(template)
    for row in rows:
        staged = _tag_row(row, template, table, refresh, group, job_id)
        if staged is None:
            continue
        key = (
            staged["_table"],
            staged["_refresh"],
            staged["_week"],
            str(staged["_geo"]),
            str(staged.get("term")),
            staged["_kind"],
        )
        merged.setdefault(key, staged)
    return None


def _fetch_all_live(
    groups: list[dict[str, object]],
    week_start: str,
    week_end: str,
    start_date: str,
    end_date: str,
    executor: _Executor | None,
    data_root: Path | str | None,
    merged: dict[tuple[object, object, object, object, object, object], dict[str, object]],
    used_templates: list[str],
    refresh_seen: set[str],
) -> dict[str, object] | None:
    """Live fetch every refresh; no checkpoints, no replay."""
    for group in groups:
        for template in _templates_for(group):
            table = _template_table(template)
            refreshes = _enumerate_refreshes(table, start_date, end_date, executor, data_root)
            if refreshes is None:
                return _wrap_error(
                    {"error": "refresh enumeration failed", "error_type": "source_unavailable", "source": "bigquery"}
                )
            for refresh in refreshes or [end_date]:
                params = _refresh_query_params(group, refresh, week_start, week_end)
                refresh_error = _collect_one_live(
                    template, table, refresh, params, group, merged, used_templates, refresh_seen, executor, data_root
                )
                if refresh_error is not None:
                    return refresh_error
    return None


def _collect_end_live(
    merged: dict[tuple[object, object, object, object, object, object], dict[str, object]],
    groups: list[dict[str, object]],
    signals_mod: ModuleType,
    used_templates: list[str],
    refresh_seen: set[str],
    limit: int,
    term: str | None,
) -> dict[str, object]:
    """Normalize staged rows to observations; compute features in memory only."""
    try:
        national_rows, dma_rows, intl_rows, dma_set = _split_staged(merged, groups)
    except ValueError as exc:
        return {"status": "error", "source": SOURCE, "error": str(exc), "error_type": "malformed_row"}
    staged_all = national_rows + dma_rows + intl_rows
    winners = _pick_winners(staged_all)
    series_features = _features_for(signals_mod, winners, dma_set)
    retrieved_at = datetime.now(UTC).isoformat()
    observations = _normalize_staged(national_rows + dma_rows + intl_rows, series_features, retrieved_at)
    return _collect_response(observations, limit, used_templates, refresh_seen, term)


def _collect_response(
    observations: list[dict[str, object]],
    limit: int,
    used_templates: list[str],
    refresh_seen: set[str],
    term: str | None,
) -> dict[str, object]:
    weeks = sorted({str(o["period"]) for o in observations})
    geos_covered = sorted({str(o["geo"]) for o in observations})
    if isinstance(term, str) and term.strip():
        needle = term.strip().lower()
        matched = [o for o in observations if needle in str(o.get("term", "")).lower()]
    else:
        matched = observations
    continuation = len(matched) > limit
    returned = matched[:limit] if continuation else matched
    return {
        "status": "ok",
        "source": SOURCE,
        "observations": returned,
        "rows": returned,
        "count": len(returned),
        "coverage": {
            "periods_covered": weeks,
            "geos_covered": geos_covered,
            "templates": used_templates,
            "refresh_dates": sorted(refresh_seen),
        },
        "continuation": continuation,
        "warnings": ["truncated"] if continuation else [],
    }


def collect_trends(
    *,
    start_date: str | None,
    end_date: str | None,
    geos: list[str] | str,
    limit: int = 100,
    data_root: Path | str | None = None,
    executor: _Executor | None = None,
    week_start: str | None = None,
    week_end: str | None = None,
    interval: str | None = "daily",
    term: str | None = None,
) -> dict[str, object]:
    """Collect top/rising lists live; each row becomes an idempotent candidate.

    Live collect plus raw-payload archiving only; derived observations and
    features are ephemeral per invocation, never persisted. The
    model-visible output is logged to the run bundle by the caller.
    """
    geos = _normalize_geos(geos)
    ready = _readiness_error()
    if ready:
        return ready
    params_error = _collect_params_error(interval, start_date, end_date, week_start, week_end, geos)
    if params_error:
        return params_error
    assert isinstance(start_date, str) and isinstance(end_date, str)
    limit = max(1, min(limit, _MAX_LIMIT))
    week_start, week_end = _default_collect_weeks(end_date, week_start, week_end)
    groups = _plan_groups(geos)
    from . import signals as _signals

    merged: dict[tuple[object, object, object, object, object, object], dict[str, object]] = {}
    used_templates: list[str] = []
    refresh_seen: set[str] = set()
    fetch_error = _fetch_all_live(
        groups, week_start, week_end, start_date, end_date, executor, data_root, merged, used_templates, refresh_seen
    )
    if fetch_error is not None:
        return fetch_error
    return _collect_end_live(merged, groups, _signals, used_templates, refresh_seen, limit, term)


def _norm_table(row: dict[str, object]) -> object:
    return row.get("_table") or row.get("table") or "trends"


def _norm_week(row: dict[str, object]) -> str:
    return str(row.get("_week") or row.get("week") or row.get("period"))


def _norm_geo(row: dict[str, object]) -> str:
    return str(row.get("_geo") if row.get("_geo") is not None else row.get("geo"))


def _norm_kind(row: dict[str, object]) -> object:
    return row.get("_kind") or row.get("list_kind") or "top"


def _norm_refresh(row: dict[str, object], week: str) -> str:
    return str(row.get("_refresh") or row.get("refresh_date") or week)


def _norm_keys(row: dict[str, object]) -> tuple[object, str, str, object, object, str]:
    week = _norm_week(row)
    return _norm_table(row), week, _norm_geo(row), row.get("term"), _norm_kind(row), _norm_refresh(row, week)


def _apply_subregion_counts(metrics: dict[str, object], row: dict[str, object]) -> None:
    if row.get("dma_count") is not None:
        metrics["dma_count"] = row["dma_count"]
    if row.get("region_count") is not None:
        metrics["region_count"] = row["region_count"]
    if row.get("dma_count") is not None or row.get("region_count") is not None:
        metrics["score_basis"] = "mean_list_score_where_listed"


def _fresh_metrics(row: dict[str, object], refresh: str, week: str) -> dict[str, object]:
    metrics: dict[str, object] = {
        "rank": row.get("rank"),
        "score": row.get("score"),
        "percent_gain": row.get("percent_gain"),
        "refresh_date": refresh,
        "week": week,
        "dma_id": row.get("dma_id"),
        "dma_name": row.get("dma_name"),
        "country_name": row.get("country_name"),
        "country_code": row.get("country_code"),
        "region_name": row.get("region_name"),
        "region_code": row.get("region_code"),
    }
    _apply_subregion_counts(metrics, row)
    return metrics


def _fresh_evidence(row: dict[str, object], table: object) -> list[object]:
    return [
        {
            "table": table,
            "row": {k: v for k, v in row.items() if not k.startswith("_")},
            "job_id": row.get("job_id"),
            "template": row.get("_template"),
        }
    ]


def _normalize_row(row: dict[str, object], retrieved_at: str, features: object = None) -> dict[str, object]:
    """Map one merged row to an ephemeral candidate observation."""
    from . import signals as _signals_n

    table, week, geo, term, kind, refresh = _norm_keys(row)
    metrics = _fresh_metrics(row, refresh, week)
    evidence = _fresh_evidence(row, table)
    return _signals_n.normalize_candidate(
        table=table,
        period=week,
        geo=geo,
        term=term,
        list_kind=kind,
        rank=row.get("rank"),
        source=SOURCE,
        source_record_id=f"{table}|{refresh}|{geo}|{term}|{kind}",
        observed_at=week,
        entities=[],
        metrics=metrics,
        evidence=evidence,
        features=features,
        retrieved_at=row.get("retrieved_at") or retrieved_at,
        known_at=row.get("known_at"),
    )
