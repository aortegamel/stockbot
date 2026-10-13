"""Data Commons macro context via bounded V2 REST observations.

Uses the official V2 observation API with DCIDs + statistical-variable IDs
(never the retired BigQuery mirror or trial keys). GOOGLE_DATA_ENABLED gates
the source and DATACOMMONS_API_KEY is required: without it the source reports
``DATACOMMONS_AUTH_REQUIRED`` with zero HTTP calls. Each facet stays a
distinct series with its original unit/provider — incompatible units or
providers are never spliced into one series. Date filtering is client-side so
the request stays one bounded call.
"""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import UTC, date, datetime

import requests

from ._lazy_config import get_datacommons_api_key, google_data_enabled

logger = logging.getLogger(__name__)

SOURCE = "datacommons"
_DC_URL = "https://api.datacommons.org/v2/observation"
_NODE_URL = "https://api.datacommons.org/v2/node"
_RESOLVE_URL = "https://api.datacommons.org/v2/resolve"
_TIMEOUT = 20
_MAX_LIMIT = 1000
_MAX_NODES = 50


def _data_enabled() -> bool:
    return google_data_enabled()


def _dc_key() -> str | None:
    return get_datacommons_api_key()


def _key_headers() -> dict[str, str] | None:
    key = _dc_key()
    if not key:
        return None
    return {"X-API-Key": key}


def _map_http_error(exc: Exception) -> dict[str, object]:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return {
            "status": "unavailable",
            "source": SOURCE,
            "error": f"data commons unreachable: {exc}",
            "error_type": "source_unavailable",
        }
    return {
        "status": "error",
        "source": SOURCE,
        "error": f"data commons request failed: {exc}",
        "error_type": "request_failed",
    }


def _map_status(response: requests.Response) -> dict[str, object] | None:
    code = response.status_code
    if code in (401, 403):
        return {
            "status": "unavailable",
            "source": SOURCE,
            "error": f"data commons refused (HTTP {code}): DATACOMMONS_AUTH_REQUIRED",
            "error_type": "auth_required",
        }
    if code == 429:
        return {
            "status": "unavailable",
            "source": SOURCE,
            "error": f"data commons refused (HTTP {code}): DATACOMMONS_RATE_LIMITED",
            "error_type": "rate_limited",
        }
    if code >= 400:
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"data commons error (HTTP {code})",
            "error_type": "request_failed",
        }
    return None


def _clean_ids(values: list[str] | str | None) -> list[str]:
    """Caller ids as a plain list."""
    if isinstance(values, str):
        return [values]
    return list(values or [])


def _clamp_limit(limit: int) -> tuple[dict[str, object] | None, int]:
    """Clamped limit or (invalid-params error, max)."""
    try:
        return None, max(1, min(limit, _MAX_LIMIT))
    except TypeError, ValueError:
        return (
            {"status": "error", "source": SOURCE, "error": f"invalid limit: {limit!r}", "error_type": "invalid_params"},
            _MAX_LIMIT,
        )


def _auth_error() -> dict[str, object] | None:
    """Disabled/auth-required error, or None with headers ready."""
    if not _data_enabled():
        return {
            "status": "disabled",
            "source": SOURCE,
            "reason": "google data disabled",
            "error": "DATACOMMONS_DISABLED",
        }
    if _key_headers() is None:
        return {
            "status": "unavailable",
            "source": SOURCE,
            "error": "DATACOMMONS_AUTH_REQUIRED",
            "error_type": "auth_required",
        }
    return None


def _post(url: str, payload: dict[str, object]) -> dict[str, object] | requests.Response:
    """Bounded POST returning the response or a fixed-shape error."""
    try:
        resp = requests.post(url, headers=_key_headers(), json=payload, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return _map_http_error(exc)
    return resp


def _decode_body(response: requests.Response) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """(mapped error, body): status errors and non-JSON become errors."""
    mapped = _map_status(response)
    if mapped is not None:
        return mapped, None
    try:
        body = response.json()
    except ValueError:
        return (
            {
                "status": "error",
                "source": SOURCE,
                "error": "data commons returned non-JSON",
                "error_type": "malformed_response",
            },
            None,
        )
    if not isinstance(body, dict):
        return (
            {
                "status": "error",
                "source": SOURCE,
                "error": "data commons returned non-JSON",
                "error_type": "malformed_response",
            },
            None,
        )
    return None, body


def _in_window(when: object, start_date: str | None, end_date: str | None) -> bool:
    """Keep observation periods that overlap the inclusive date window."""
    raw = str(when)
    bounds = _calendar_bounds(raw)
    period_start, period_end = bounds if bounds is not None else (raw, raw)
    if start_date and period_end < start_date:
        return False
    return not (end_date and period_start > end_date)


def _calendar_bounds(raw: str) -> tuple[str, str] | None:
    """(start, end) for YYYY/YYYY-MM periods; None keeps the lexical contract."""
    try:
        if len(raw) == 4:
            year = date.fromisoformat(f"{raw}-01-01").year
            return f"{year:04d}-01-01", f"{year:04d}-12-31"
        if len(raw) == 7:
            month = date.fromisoformat(f"{raw}-01")
            return f"{raw}-01", f"{raw}-{monthrange(month.year, month.month)[1]:02d}"
    except ValueError:
        return None
    return None


def _facet_meta(facets: object, facet_id: object) -> dict[str, object]:
    """Metadata mapping for one facet; malformed shapes become empty."""
    if isinstance(facets, dict):
        meta = facets.get(facet_id)
        if isinstance(meta, dict):
            return meta
    return {}


def _section_map(body: dict[str, object], key: str) -> dict[str, object]:
    """One body mapping section; malformed shapes become empty."""
    section = body.get(key)
    if isinstance(section, dict):
        return section
    return {}


def _facet_points(facet: dict[str, object], start_date: str | None, end_date: str | None) -> list[dict[str, object]]:
    """Dated observations for one facet, sorted by date."""
    points: list[dict[str, object]] = []
    raw_obs = facet.get("observations", [])
    obs_list = raw_obs if isinstance(raw_obs, list) else []
    for obs in obs_list:
        if not isinstance(obs, dict):
            continue
        when, value = obs.get("date"), obs.get("value")
        if when is None or value is None:
            continue
        if not _in_window(when, start_date, end_date):
            continue
        points.append({"date": when, "value": value})
    return points


def _facet_entry(
    variable: str,
    entity: str,
    facet: dict[str, object],
    meta: dict[str, object],
    points: list[dict[str, object]],
    retrieved_at: str,
) -> dict[str, object]:
    """One faceted series entry with original unit/provider preserved."""
    return {
        "geo": entity,
        "variable": variable,
        "facet": facet.get("facetId"),
        "unit": facet.get("unit", meta.get("unit")),
        "provider": (
            facet.get("provenanceName")
            or meta.get("importName")
            or facet.get("seriesSource")
            or meta.get("provenanceUrl")
            or facet.get("provenanceUrl")
        ),
        "provenanceUrl": meta.get("provenanceUrl") or facet.get("provenanceUrl"),
        "importName": meta.get("importName"),
        "measurementMethod": meta.get("measurementMethod"),
        "observationPeriod": meta.get("observationPeriod"),
        "scalingFactor": meta.get("scalingFactor"),
        "isDcAggregate": meta.get("isDcAggregate"),
        "observations": points,
        "retrieved_at": retrieved_at,
        "known_at": retrieved_at,
    }


def _ordered_facets(by_ent: object) -> list[dict[str, object]]:
    """orderedFacets list for one entity; malformed shapes become empty."""
    if not isinstance(by_ent, dict):
        return []
    facets = by_ent.get("orderedFacets", [])
    if not isinstance(facets, list):
        return []
    return [f for f in facets if isinstance(f, dict)]


def _series_for_entity(
    variable: str,
    entity: str,
    by_ent: object,
    facets: object,
    start_date: str | None,
    end_date: str | None,
    retrieved_at: str,
) -> list[dict[str, object]]:
    """Non-empty faceted series for one entity."""
    out: list[dict[str, object]] = []
    for facet in _ordered_facets(by_ent):
        facet_id = facet.get("facetId")
        meta = _facet_meta(facets, facet_id)
        points = _facet_points(facet, start_date, end_date)
        if not points:
            continue
        out.append(_facet_entry(variable, entity, facet, meta, points, retrieved_at))
    return out


def _by_entity_map(by_var: object) -> list[tuple[str, object]]:
    """(entity, payload) pairs for one variable; malformed shapes become empty."""
    if not isinstance(by_var, dict):
        return []
    by_entity = by_var.get("byEntity", {}) or {}
    if not isinstance(by_entity, dict):
        return []
    return list(by_entity.items())


def _append_entry(series: list[dict[str, object]], entry: dict[str, object], remaining: int) -> tuple[int, bool]:
    """Append one capped entry; returns (remaining, continuation)."""
    if remaining <= 0:
        return remaining, True
    obs = entry.get("observations")
    take: list[dict[str, object]] = []
    if isinstance(obs, list):
        take = [o for o in obs[:remaining] if isinstance(o, dict)]
    entry["observations"] = take
    series.append(entry)
    return remaining - len(take), len(take) >= remaining


def _collect_series(
    body: dict[str, object], start_date: str | None, end_date: str | None, limit: int, retrieved_at: str
) -> tuple[list[dict[str, object]], bool]:
    """Bounded series plus continuation flag; per-facet observations capped."""
    facets = _section_map(body, "facets")
    series: list[dict[str, object]] = []
    remaining = limit
    continuation = False
    by_variable = _section_map(body, "byVariable")
    variables_iter = by_variable.items()
    for variable, by_var in variables_iter:
        for entity, by_ent in _by_entity_map(by_var):
            for entry in _series_for_entity(variable, entity, by_ent, facets, start_date, end_date, retrieved_at):
                remaining, done = _append_entry(series, entry, remaining)
                continuation = continuation or done
    return series, continuation


def _macro_payload(
    geos: list[str],
    variables: list[str],
    series: list[dict[str, object]],
    continuation: bool,
    body: dict[str, object],
    retrieved_at: str,
) -> dict[str, object]:
    """Fixed-shape ok payload for macro observations."""
    out: dict[str, object] = {
        "status": "ok",
        "source": SOURCE,
        "series": series,
        "count": len(series),
        "coverage": {"geos": geos, "variables": variables},
        "continuation": continuation or bool(body.get("nextToken")),
        "retrieved_at": retrieved_at,
        "known_at": retrieved_at,
    }
    if body.get("nextToken"):
        out["nextToken"] = body["nextToken"]
    return out


def _resolve_nodes(nodes: list[str] | str | None) -> list[str]:
    """Caller nodes as a plain list."""
    return [nodes] if isinstance(nodes, str) else list(nodes or [])


def _nodes_error(nodes: list[str]) -> dict[str, object] | None:
    """Invalid-params error for empty or over-bound node lists, else None."""
    if not nodes:
        return {
            "status": "error",
            "source": SOURCE,
            "error": "at least one node is required",
            "error_type": "invalid_params",
        }
    if len(nodes) > _MAX_NODES:
        return {
            "status": "error",
            "source": SOURCE,
            "error": f"{len(nodes)} nodes exceeds bound {_MAX_NODES}",
            "error_type": "invalid_params",
        }
    return None


def _resolve_payload(nodes: list[str], resolver: str | None, property: str | None) -> dict[str, object]:
    """Resolve payload with optional resolver/property selectors."""
    payload: dict[str, object] = {"nodes": nodes}
    if resolver:
        payload["resolver"] = resolver
    if property:
        payload["property"] = property
    return payload


def _obs_date(point: dict[str, object]) -> str:
    """Sort key for observation points: ISO date string, never missing here."""
    return str(point["date"])


def get_macro_context(
    geos: list[str] | str | None,
    variables: list[str] | str | None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    """Bounded statistical observations for geos x variables, faceted series."""
    geos = _clean_ids(geos)
    variables = _clean_ids(variables)
    if not geos or not variables:
        return {
            "status": "error",
            "source": SOURCE,
            "error": "geos and variables are both required",
            "error_type": "invalid_params",
        }
    lim_err, limit = _clamp_limit(limit)
    if lim_err is not None:
        return lim_err
    gate = _auth_error()
    if gate is not None:
        return gate
    payload: dict[str, object] = {
        "date": "",
        "entity": {"dcids": geos},
        "variable": {"dcids": variables},
        "select": ["date", "entity", "variable", "value", "facet"],
    }
    response = _post(_DC_URL, payload)
    if isinstance(response, dict):
        return response
    body_err, body = _decode_body(response)
    if body_err is not None or body is None:
        return body_err or {
            "status": "error",
            "source": SOURCE,
            "error": "data commons returned non-JSON",
            "error_type": "malformed_response",
        }
    retrieved_at = datetime.now(UTC).isoformat()
    series, continuation = _collect_series(body, start_date, end_date, limit, retrieved_at)
    return _macro_payload(geos, variables, series, continuation, body, retrieved_at)


def resolve_entities(
    nodes: list[str] | str | None, *, resolver: str | None = None, property: str | None = None
) -> dict[str, object]:
    """Resolve up to 50 nodes to DCIDs via the V2 resolve API (key header)."""
    cleaned = _resolve_nodes(nodes)
    nodes_err = _nodes_error(cleaned)
    if nodes_err is not None:
        return nodes_err
    gate = _auth_error()
    if gate is not None:
        return gate
    response = _post(_RESOLVE_URL, _resolve_payload(cleaned, resolver, property))
    if isinstance(response, dict):
        return response
    body_err, body = _decode_body(response)
    if body_err is not None or body is None:
        return body_err or {
            "status": "error",
            "source": SOURCE,
            "error": "data commons returned non-JSON",
            "error_type": "malformed_response",
        }
    return {
        "status": "ok",
        "source": SOURCE,
        "entities": body.get("entities", body),
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def _hierarchy_gate(dcid: str) -> dict[str, object] | None:
    """Param/disabled/auth gates for hierarchy lookup; None when the request may proceed."""
    if not dcid or not dcid.strip():
        return {"status": "error", "source": SOURCE, "error": "dcid is required", "error_type": "invalid_params"}
    if not _data_enabled():
        return {
            "status": "disabled",
            "source": SOURCE,
            "reason": "google data disabled",
            "error": "DATACOMMONS_DISABLED",
        }
    if _key_headers() is None:
        return {
            "status": "unavailable",
            "source": SOURCE,
            "error": "DATACOMMONS_AUTH_REQUIRED",
            "error_type": "auth_required",
        }
    return None


def get_place_hierarchy(dcid: str) -> dict[str, object]:
    """Containing states for one place DCID via documented containedInPlace lookup."""
    gate = _hierarchy_gate(dcid)
    if gate is not None:
        return gate
    headers = _key_headers()
    payload = {"nodes": [dcid], "property": "<-containedInPlace+{typeOf:State}"}
    try:
        response = requests.post(_NODE_URL, headers=headers, json=payload, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return _map_http_error(exc)
    mapped = _map_status(response)
    if mapped is not None:
        return mapped
    try:
        body = response.json()
    except ValueError:
        return {
            "status": "error",
            "source": SOURCE,
            "error": "data commons returned non-JSON",
            "error_type": "malformed_response",
        }
    return {
        "status": "ok",
        "source": SOURCE,
        "dcid": dcid,
        "hierarchy": body.get("data", body),
        "retrieved_at": datetime.now(UTC).isoformat(),
    }
