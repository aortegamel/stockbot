"""Exa web-search client: optional external research evidence provider.

Bounded, highlight-based web search for current qualitative evidence (news,
announcements, competitive/industry developments, publications, commentary).
Exa is optional: every call returns an error dict when disabled or on any
failure — it never raises, and the agent treats failures as soft. Results are
research evidence, never canonical financial records.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import requests

from . import config

logger = logging.getLogger(__name__)

EXA_TIMEOUT_SECONDS = 20
EXA_HIGHLIGHT_MAX_CHARS = 600
EXA_SEARCH_TEXT_MAX_CHARS = 12_000
EXA_DEFAULT_LIMIT = 5
EXA_MAX_LIMIT = 25
_APPROVED_SEARCH_TYPES = frozenset({"auto", "fast", "deep-lite"})
_APPROVED_CATEGORIES = frozenset({"news", "company", "publication", "financial report"})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_session: requests.Session | None = None


def _ensure_session() -> requests.Session:
    """Lazily created, shared session (mirrors analyst_client)."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _error(message: str) -> dict[str, object]:
    return {"error": message, "source": "exa"}


def _search_unavailable() -> bool:
    """Availability predicate: disabled or missing API key."""
    return not config.exa_enabled() or not config.get_exa_api_key()


def _bad_search_query(query: str) -> bool:
    """Query predicate: not a non-empty string."""
    return not isinstance(query, str) or not query.strip()


def _bad_search_category(category: str | None) -> bool:
    """Category predicate: present but not approved."""
    return category is not None and category not in _APPROVED_CATEGORIES


def _validate_search_params(
    query: str,
    category: str | None,
    search_type: str,
) -> dict[str, object] | None:
    """Basic guard clauses: availability, query, type, category."""
    if _search_unavailable():
        return _error("Exa search unavailable")
    if _bad_search_query(query):
        return _error("Exa search query must be a non-empty string")
    if search_type not in _APPROVED_SEARCH_TYPES:
        return _error(f"Unsupported search_type '{search_type}'. Allowed: auto, fast, deep-lite")
    if _bad_search_category(category):
        return _error(f"Unsupported category '{category}'. Allowed: news, company, publication, financial report")
    return None


def _validate_search_dates(
    start_published_date: str | None,
    end_published_date: str | None,
) -> dict[str, object] | None:
    """Date-format guards; returns error dict or None."""
    for value in (start_published_date, end_published_date):
        if value is None:
            continue
        if not _DATE_RE.match(value):
            return _error(f"Invalid date '{value}'. Expected YYYY-MM-DD")
    return None


def _validate_search_options(
    category: str | None,
    start_published_date: str | None,
    end_published_date: str | None,
    exclude_domains: list[str] | None,
    limit: object,
) -> tuple[dict[str, object] | None, int]:
    """Date/limit/company-compat guards; returns (error, normalized_limit)."""
    date_error = _validate_search_dates(start_published_date, end_published_date)
    if date_error is not None:
        return date_error, EXA_DEFAULT_LIMIT
    if limit is None:
        limit = EXA_DEFAULT_LIMIT
    if not isinstance(limit, int):
        return _error("limit must be an integer"), EXA_DEFAULT_LIMIT
    normalized = max(1, min(limit, EXA_MAX_LIMIT))
    # Exa returns HTTP 400 for date/domain-exclusion params with
    # category=company; reject instead of silently dropping filters so the
    # caller learns the request cannot be honored (include_domains is fine).
    if category == "company" and any((start_published_date, end_published_date, exclude_domains)):
        return _error(
            "category 'company' does not support start_published_date, end_published_date, or exclude_domains"
        ), normalized
    return None, normalized


def _build_search_payload(
    query: str,
    category: str | None,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    start_published_date: str | None,
    end_published_date: str | None,
    search_type: str,
    limit: int,
) -> dict[str, object]:
    """Assemble the Exa request payload dict."""
    payload: dict[str, object] = {
        "query": query,
        "numResults": limit,
        "contents": {
            "highlights": True,
            "text": {"maxCharacters": EXA_SEARCH_TEXT_MAX_CHARS, "includeHtmlTags": True},
        },
    }
    if category is not None:
        payload["category"] = category
    if include_domains:
        payload["includeDomains"] = include_domains
    if exclude_domains:
        payload["excludeDomains"] = exclude_domains
    if start_published_date is not None:
        payload["startPublishedDate"] = f"{start_published_date}T00:00:00.000Z"
    if end_published_date is not None:
        payload["endPublishedDate"] = f"{end_published_date}T23:59:59.999Z"
    payload["type"] = search_type
    return payload


def _first_highlight(item: dict[str, object]) -> str:
    """Bounded first highlight or empty string."""
    highlights = item.get("highlights")
    first = highlights[0] if isinstance(highlights, list) and highlights else None
    return str(first)[:EXA_HIGHLIGHT_MAX_CHARS] if first is not None else ""


def _parse_search_results(
    raw_results: list[object],
    category: str | None,
    retrieved_at: str,
) -> tuple[list[dict[str, object]], int]:
    """Build (evidence, omitted_count) from validated raw results."""
    evidence: list[dict[str, object]] = []
    for item in raw_results:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        evidence.append(
            {
                "title": str(item.get("title") or ""),
                "url": item["url"],
                "source_domain": urlparse(item["url"]).netloc,
                "published_at": item.get("publishedDate"),
                "retrieved_at": retrieved_at,
                "highlight": _first_highlight(item),
                "category": category,
                "source_content": item["text"][:EXA_SEARCH_TEXT_MAX_CHARS] if isinstance(item.get("text"), str) else "",
                "source_representation": "html",
                "highlights": [
                    text[:EXA_HIGHLIGHT_MAX_CHARS] for text in item.get("highlights", []) if isinstance(text, str)
                ]
                if isinstance(item.get("highlights"), list)
                else [],
            }
        )
    return evidence, max(0, len(raw_results) - len(evidence))


def _request_search(payload: dict[str, object]) -> object:
    """POST the payload; returns response or an error dict."""
    try:
        return _ensure_session().post(
            f"{config.EXA_API_BASE}/search",
            headers={"x-api-key": config.get_exa_api_key()},
            json=payload,
            timeout=EXA_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return _error("Exa search timed out")
    except requests.RequestException:
        return _error("Exa search unavailable")


def _response_status(response: object) -> int | None:
    """HTTP status of a response-like object; None when absent/non-int."""
    status_code = getattr(response, "status_code", None)
    return status_code if type(status_code) is int else None


def _response_body(response: object) -> object | None:
    """Decoded JSON body; None when the response has no callable json()."""
    decode = getattr(response, "json", None)
    if not callable(decode):
        return None
    try:
        return decode()
    except ValueError:
        return None


def _results_list(body: object) -> list[object] | None:
    """results list from a decoded body; None when absent/non-list."""
    raw_results = body.get("results") if isinstance(body, dict) else None
    return raw_results if isinstance(raw_results, list) else None


def _extract_raw_results(response: object) -> tuple[dict[str, object] | None, list[object] | None]:
    """Map response to (error, raw_results); error is None on success."""
    status_code = _response_status(response)
    if status_code != 200:
        return _error(f"Exa search failed: HTTP {status_code}"), None
    results = _results_list(_response_body(response))
    if results is None:
        return _error("Exa search returned an invalid response"), None
    return None, results


def search(
    query: str,
    *,
    category: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
    search_type: str = "auto",
    limit: object = EXA_DEFAULT_LIMIT,
) -> dict[str, object]:
    """Search the web via Exa, returning bounded highlight-based evidence.

    Never raises; every failure returns {"error": ..., "source": "exa"}.
    """
    error = _validate_search_params(query, category, search_type)
    if error:
        return error
    options_error, normalized_limit = _validate_search_options(
        category, start_published_date, end_published_date, exclude_domains, limit
    )
    if options_error:
        return options_error
    payload = _build_search_payload(
        query,
        category,
        include_domains,
        exclude_domains,
        start_published_date,
        end_published_date,
        search_type,
        normalized_limit,
    )
    retrieved_at = datetime.now(UTC).isoformat()
    response = _request_search(payload)
    if isinstance(response, dict):
        return response
    raw_error, raw_results = _extract_raw_results(response)
    if raw_error or raw_results is None:
        return raw_error or _error("Exa search returned an invalid response")
    evidence, omitted_count = _parse_search_results(raw_results, category, retrieved_at)
    return {
        "result_type": "web_search",
        "query": query,
        "search_type": search_type,
        "evidence": evidence,
        "omitted_count": omitted_count,
        "row_count": len(evidence),
        "source": "exa",
        "retrieved_at": retrieved_at,
    }
