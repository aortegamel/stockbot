"""Tests for partition-aware datapoint retrieval, error context, and freshness.

Offline and deterministic: FINRA HTTP is mocked via the shared fixtures and
a partitions fixture. Verifies that unrestricted sortFields are never sent,
that only published partition tuples are walked (never a Cartesian product)
with a bounded attempt budget, that HTTP failures carry structured context,
and that as_of_date/data_freshness/environment are always surfaced.
"""

from collections.abc import Iterator
from datetime import date
from unittest.mock import MagicMock

import pytest

from app import finra_client
from app.policy import LOCAL_CONTEXT
from app.tool_render import render_tool_result
from app.tools import execute_tool
from tests.test_finra import (
    FakeCache,
    _as_seq,
    _error,
    _freeze_finra_today,
    _mock_get,
    _response,
    _token_response,
)

# The fixture rows carry 2026-08 dates. Freshness reads the real clock, so pin it near them.
FINRA_FIXTURE_TODAY = date(2026, 8, 20)


@pytest.fixture(autouse=True)
def _isolation(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FINRA_USE_MOCK", "")
    monkeypatch.setenv("FINRA_ANALYSIS_MODEL", "")
    finra_client.reset_token_cache()
    finra_client.reset_discovery_cache()
    finra_client.reset_partitions_cache()
    yield
    finra_client.reset_token_cache()
    finra_client.reset_discovery_cache()
    finra_client.reset_partitions_cache()


@pytest.fixture(autouse=True)
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> FakeCache:
    fc = FakeCache()
    monkeypatch.setattr(finra_client, "cache", fc)
    return fc


@pytest.fixture(autouse=True)
def _frozen_finra_today(monkeypatch: pytest.MonkeyPatch) -> None:
    _freeze_finra_today(monkeypatch, FINRA_FIXTURE_TODAY)


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    get_mock = MagicMock(side_effect=_mock_get)
    post_mock = MagicMock()
    monkeypatch.setattr("app.finra_client.requests.get", get_mock)
    monkeypatch.setattr("app.finra_client.requests.post", post_mock)
    monkeypatch.setattr("app.finra_client.get_finra_client_id", lambda: "client")
    monkeypatch.setattr("app.finra_client.get_finra_client_secret", lambda: "secret")
    return {"get": get_mock, "post": post_mock}


def _data_posts(post_mock: MagicMock):
    return [
        c.kwargs["json"]
        for c in post_mock.call_args_list
        if "oauth2/access_token" not in (c.args[0] if c.args else c.kwargs.get("url", ""))
    ]


def _partitions_get_calls(get_mock: MagicMock):
    return [c.args[0] for c in get_mock.call_args_list if "/partitions/group/" in c.args[0]]


# ---------------------------------------------------------------------------
# Partition-aware ascending / descending datapoints
# ---------------------------------------------------------------------------


def test_datapoints_ascending_sort_via_partitions_oldest_first(http: dict[str, MagicMock]) -> None:
    partition_rows = {
        "2026-08-01": [{"settlementDate": "2026-08-01", "currentShortPositionQuantity": 900}],
        "2026-08-07": [{"settlementDate": "2026-08-07", "currentShortPositionQuantity": 700}],
        "2026-08-14": [{"settlementDate": "2026-08-14", "currentShortPositionQuantity": 100}],
    }

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        value = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "settlementDate")
        return _response(partition_rows[value])

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "limit": 3,
            "sort_order": "asc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    posts = _data_posts(http["post"])
    assert len(posts) == 3
    assert all("sortFields" not in p for p in posts)
    # Oldest partition queried first for ascending sorts.
    assert posts[0]["compareFilters"][-1]["fieldValue"] == "2026-08-01"
    dates = [r["settlementDate"] for r in _as_seq(result["records"])]
    assert dates == sorted(dates)
    assert result["sort_source"] == "partitions"


def test_datapoints_multi_partition_queries_only_published_tuples(http: dict[str, MagicMock]) -> None:
    """weeklySummary (weekStartDate + tierIdentifier): the walk queries only
    the date/tier tuples FINRA actually published — never a Cartesian
    product that would invent the missing 2026-08-03/T2 combination."""
    rows_by_week = {
        "2026-08-10": [{"summaryStartDate": "2026-08-14", "totalWeeklyShareQuantity": 10}],
        "2026-08-03": [{"summaryStartDate": "2026-08-07", "totalWeeklyShareQuantity": 20}],
    }

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        fields = {f["fieldName"]: f["fieldValue"] for f in payload["compareFilters"]}
        week = fields["weekStartDate"]
        tier = fields["tierIdentifier"]
        if week == "2026-08-03" and tier == "T2":
            raise AssertionError("unpublished date/tier tuple was queried")
        return _response(rows_by_week[week])

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/weeklySummary",
            "fields": ["summaryStartDate", "totalWeeklyShareQuantity"],
            "ticker": "AAPL",
            "limit": 3,
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    posts = _data_posts(http["post"])
    assert len(posts) == 3
    combos = []
    for payload in posts:
        fields = {f["fieldName"]: f["fieldValue"] for f in payload["compareFilters"]}
        assert fields["weekStartDate"] in rows_by_week
        assert fields["tierIdentifier"] in ("T1", "T2")
        assert "sortFields" not in payload
        combos.append((fields["weekStartDate"], fields["tierIdentifier"]))
    # Newest week first; within a week FINRA's published tier order; the
    # unpublished 2026-08-03/T2 combination is never attempted.
    assert combos == [
        ("2026-08-10", "T1"),
        ("2026-08-10", "T2"),
        ("2026-08-03", "T1"),
    ]
    assert _as_seq(result["records"])[0]["summaryStartDate"] == "2026-08-14"
    assert result["sort_source"] == "partitions"


def test_datapoints_ascending_multi_partition_oldest_first(http: dict[str, MagicMock]) -> None:
    """Ascending partition walks query the oldest date partition first."""
    rows_by_week = {
        "2026-08-10": [{"summaryStartDate": "2026-08-14", "totalWeeklyShareQuantity": 10}],
        "2026-08-03": [{"summaryStartDate": "2026-08-07", "totalWeeklyShareQuantity": 20}],
    }

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        week = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "weekStartDate")
        return _response(rows_by_week[week])

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/weeklySummary",
            "fields": ["summaryStartDate", "totalWeeklyShareQuantity"],
            "ticker": "AAPL",
            "limit": 2,
            "sort_order": "asc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    posts = _data_posts(http["post"])
    assert len(posts) == 2
    weeks = [next(f["fieldValue"] for f in p["compareFilters"] if f["fieldName"] == "weekStartDate") for p in posts]
    assert weeks == ["2026-08-03", "2026-08-10"]
    dates = [r["summaryStartDate"] for r in _as_seq(result["records"])]
    assert dates == sorted(dates)
    assert result["sort_source"] == "partitions"


def test_datapoints_partition_budget_exhausted_errors(http: dict[str, MagicMock]) -> None:
    """When the bounded partition walk cannot establish the limit, the
    request fails with a narrowing-required error instead of a partial
    silent answer."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(finra_client, "_MAX_PARTITION_QUERIES", 2)
    try:
        rows_by_week = {
            "2026-08-10": [{"summaryStartDate": "2026-08-14", "totalWeeklyShareQuantity": 10}],
            "2026-08-03": [{"summaryStartDate": "2026-08-07", "totalWeeklyShareQuantity": 20}],
        }

        def _respond(url: str, **kw: object):
            if "oauth2/access_token" in url:
                return _token_response()
            payload = kw["json"]
            assert isinstance(payload, dict)
            week = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "weekStartDate")
            return _response(rows_by_week[week])

        http["post"].side_effect = _respond
        result = execute_tool(
            "get_finra_datapoints",
            {
                "dataset": "otcMarket/weeklySummary",
                "fields": ["summaryStartDate", "totalWeeklyShareQuantity"],
                "ticker": "AAPL",
                "limit": 5,
                "sort_order": "desc",
            },
            model="test",
            context=LOCAL_CONTEXT,
        )
        assert "error" in result
        assert "Narrow" in _error(result) or "narrow" in _error(result)
        assert "partition" in _error(result)
        assert "records" not in result  # no partial exact-data table
    finally:
        monkeypatch.undo()


def test_datapoints_empty_partitions_budget_exhaustion_errors(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """All checked partitions return empty data but the budget ends before
    every relevant partition was examined: the request fails with the
    budget/narrowing error — never an unproven 'No data found' claim."""
    monkeypatch.setattr(finra_client, "_MAX_PARTITION_QUERIES", 2)

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        return _response([])  # 200 OK with no records

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "limit": 5,
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    posts = _data_posts(http["post"])
    assert len(posts) == 2  # exactly the budget; more tuples remain unchecked
    assert "error" in result
    assert "No data found" not in _error(result)
    assert "partition queries" in _error(result)
    assert "records" not in result


@pytest.mark.parametrize("status", [400, 500])
def test_datapoints_failed_partitions_count_toward_budget(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Repeated HTTP failures count against the fixed budget; the walk never
    continues through unlimited failing partitions."""
    monkeypatch.setattr(finra_client, "_MAX_PARTITION_QUERIES", 3)
    fail_body = {"error": {"message": "partition unavailable"}}

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        return _response(fail_body, status=status)

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "limit": 5,
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    posts = _data_posts(http["post"])
    assert len(posts) == 3  # exactly the budget, no unlimited retries
    assert "error" in result
    assert str(status) in _error(result)
    assert result["http_status"] == status
    assert "records" not in result


def test_datapoints_complete_short_result_warns(http: dict[str, MagicMock]) -> None:
    """All relevant partitions examined but fewer records than the limit:
    the available rows are returned with an explicit 'only N matching
    records found' completeness warning (not an error)."""
    rows_by_week = {
        "2026-08-10": [{"summaryStartDate": "2026-08-14", "totalWeeklyShareQuantity": 10}],
        "2026-08-03": [{"summaryStartDate": "2026-08-07", "totalWeeklyShareQuantity": 20}],
    }

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        week = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "weekStartDate")
        return _response(rows_by_week[week])

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/weeklySummary",
            "fields": ["summaryStartDate", "totalWeeklyShareQuantity"],
            "ticker": "AAPL",
            "limit": 5,
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert len(_as_seq(result["records"])) == 3
    assert result["returned_count"] == 3
    assert result["may_have_more"] is False
    assert any("only 3 matching records" in w for w in _as_seq(result["warnings"]))
    rendered = render_tool_result(result)
    assert "only 3 matching records" in rendered


def test_datapoints_unpartitioned_date_sort_rejected_before_http(http: dict[str, MagicMock]) -> None:
    """A date-typed sort that is not the dataset's authoritative date field
    cannot be resolved by partition walking: it must fail with the
    EQUAL-filters message before any FINRA data request."""
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/weeklySummary",
            "fields": ["summaryStartDate", "totalWeeklyShareQuantity"],
            "ticker": "AAPL",
            "sort_fields": ["-lastUpdateDate"],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "EQUAL" in _error(result)
    assert "weekStartDate" in _error(result)
    assert "tierIdentifier" in _error(result)
    assert len(_data_posts(http["post"])) == 0


def test_datapoints_date_range_sort_rejected_when_mapped_date_field(http: dict[str, MagicMock]) -> None:
    """weeklySummary ranges on summaryStartDate cannot be narrowed to
    weekStartDate partitions: the request is rejected before any data
    request instead of silently wasting the partition budget."""
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/weeklySummary",
            "fields": ["summaryStartDate", "totalWeeklyShareQuantity"],
            "ticker": "AAPL",
            "start_date": "2026-05-01",
            "end_date": "2026-05-31",
            "limit": 5,
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "Date-range" in _error(result)
    assert "weekStartDate" in _error(result)
    assert "summaryStartDate" in _error(result)
    assert len(_data_posts(http["post"])) == 0


def test_datapoints_date_range_sort_still_walks_partition_field_ranges(http: dict[str, MagicMock]) -> None:
    """When the requested date field IS a partition field (e.g.
    consolidatedShortInterest settlementDate), a range still narrows the
    partition walk — the mapped-date rejection does not apply. The server
    returns all in-range rows for each query (the range is enforced by
    dateRangeFilters, not by per-tuple filters)."""
    in_range_rows = [
        {"settlementDate": "2026-08-01", "currentShortPositionQuantity": 1},
        {"settlementDate": "2026-08-07", "currentShortPositionQuantity": 2},
    ]

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        return _response(list(in_range_rows))

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "start_date": "2026-08-01",
            "end_date": "2026-08-07",
            "limit": 5,
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["sort_source"] == "partitions"
    assert result["records"]
    assert len(_as_seq(result["records"])) <= 5
    for payload in _data_posts(http["post"]):
        assert payload["dateRangeFilters"] == [
            {
                "fieldName": "settlementDate",
                "startDate": "2026-08-01",
                "endDate": "2026-08-07",
            }
        ]
    dates = [r["settlementDate"] for r in _as_seq(result["records"])]
    assert dates == sorted(dates, reverse=True)


def test_datapoints_non_date_sort_rejected_before_http(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "sort_fields": ["-currentShortPositionQuantity"],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "EQUAL" in _error(result)
    assert "settlementDate" in _error(result)
    # Rejected before any data POST: only the catalog token request.
    assert len(_data_posts(http["post"])) == 0


def test_datapoints_multi_field_sort_rejected_when_unpartitioned(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "sort_fields": ["-settlementDate", "+currentShortPositionQuantity"],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "Multi-field" in _error(result)
    assert len(_data_posts(http["post"])) == 0


def test_datapoints_partitions_cached_across_calls(http: dict[str, MagicMock]) -> None:
    rows = {"2026-08-14": [{"settlementDate": "2026-08-14"}]}

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        value = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "settlementDate")
        return _response(rows[value])

    http["post"].side_effect = _respond
    args: dict[str, object] = {
        "dataset": "otcMarket/consolidatedShortInterest",
        "fields": ["settlementDate"],
        "ticker": "AAPL",
        "limit": 1,
        "sort_order": "desc",
    }
    first = execute_tool("get_finra_datapoints", args, model="test", context=LOCAL_CONTEXT)
    assert "error" not in first, first
    gets_after_first = len(_partitions_get_calls(http["get"]))

    finra_client.reset_partitions_cache()  # new process: disk cache still warm
    second = execute_tool("get_finra_datapoints", args, model="test", context=LOCAL_CONTEXT)
    assert "error" not in second, second
    assert len(_partitions_get_calls(http["get"])) == gets_after_first


def test_parse_partitions_preserves_ordered_tuples() -> None:
    raw: dict[str, object] = {
        "availablePartitions": [
            {"partitions": ["2026-08-10", "T1"]},
            {"partitions": ["2026-08-10", "T2"]},
            {"partitions": ["2026-08-03", "T1"]},
        ]
    }
    parsed = finra_client._parse_partitions(raw, ("weekStartDate", "tierIdentifier"))
    assert parsed == [
        ("2026-08-10", "T1"),
        ("2026-08-10", "T2"),
        ("2026-08-03", "T1"),
    ]


def test_parse_partitions_drops_ambiguous_scalar_for_multi_field() -> None:
    raw: dict[str, object] = {"availablePartitions": ["2026-08-10"]}
    parsed = finra_client._parse_partitions(raw, ("weekStartDate", "tierIdentifier"))
    assert parsed == []  # a single value cannot be placed safely


def test_parse_partitions_drops_tuples_with_unexpected_extra_values() -> None:
    """A tuple with more values than partition fields is discarded, not
    silently truncated to fit."""
    raw: dict[str, object] = {
        "availablePartitions": [
            {"partitions": ["2026-08-10", "T1", "EXTRA"]},
            {"partitions": ["2026-08-03", "T1"]},
        ]
    }
    parsed = finra_client._parse_partitions(raw, ("weekStartDate", "tierIdentifier"))
    assert parsed == [("2026-08-03", "T1")]


def test_partition_cache_old_flattened_shape_ignored(http: dict[str, MagicMock], fake_cache: FakeCache) -> None:
    """Old flattened {field: [values]} cache entries (pre-tuple format) are
    rejected so invalid Cartesian combinations can never be reconstructed."""
    rows = {"2026-08-14": [{"settlementDate": "2026-08-14"}]}

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        value = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "settlementDate")
        return _response(rows[value])

    http["post"].side_effect = _respond
    cache_key = "finra:v3:production:partitions:otcMarket/consolidatedShortInterest"
    fake_cache.store[cache_key] = {"settlementDate": ["2026-08-14"]}
    args: dict[str, object] = {
        "dataset": "otcMarket/consolidatedShortInterest",
        "fields": ["settlementDate"],
        "ticker": "AAPL",
        "limit": 1,
        "sort_order": "desc",
    }
    result = execute_tool("get_finra_datapoints", args, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    # The flattened hit was rejected: partitions refetched and rewritten as
    # JSON-safe lists, normalized to tuples in memory.
    assert fake_cache.store[cache_key] == [
        ["2026-08-14"],
        ["2026-08-07"],
        ["2026-08-01"],
    ]
    assert len(_partitions_get_calls(http["get"])) == 1
    assert finra_client._partitions_mem[("production", "otcmarket/consolidatedshortinterest")] == [
        ("2026-08-14",),
        ("2026-08-07",),
        ("2026-08-01",),
    ]


# ---------------------------------------------------------------------------
# Error context preservation
# ---------------------------------------------------------------------------


def test_datapoints_http_400_structured_error(http: dict[str, MagicMock]) -> None:
    body = {"error": {"message": "sortFields not permitted without partition EQUAL filters"}}

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        return _response(body, status=400)

    http["post"].side_effect = _respond
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "400" in _error(result)
    assert result["dataset"] == "otcMarket/consolidatedShortInterest"
    assert result["dataset_id"] == "otcMarket/consolidatedShortInterest"
    assert result["request_purpose"] == "exact datapoints request (get_finra_datapoints)"
    assert result["http_status"] == 400
    finra_response = result["finra_response"]
    assert isinstance(finra_response, (str, dict))
    assert "sortFields" in finra_response
    assert result["environment"] in ("production", "mock")


def test_error_render_preserves_context() -> None:
    result = {
        "error": "FINRA request failed (400): bad sort",
        "dataset": "consolidatedShortInterest",
        "dataset_id": "otcMarket/consolidatedShortInterest",
        "request_purpose": "exact datapoints request (get_finra_datapoints)",
        "http_status": 400,
        "finra_response": '{"error": "sortFields restricted"}',
        "environment": "production",
    }
    text = render_tool_result(result)
    assert text.startswith("Error: FINRA request failed (400)")
    assert "dataset: consolidatedShortInterest" in text
    assert "http_status: 400" in text
    assert "request_purpose: exact datapoints" in text
    assert "finra_response:" in text
    assert "environment: production" in text


def test_error_sanitization_strips_credentials() -> None:
    raw = '{"message": "nope", "access_token": "abc123", "client_secret": "shh", "note": "Bearer tok-999 x"}'
    sanitized = finra_client._sanitize_finra_body(raw)
    assert "abc123" not in sanitized
    assert "shh" not in sanitized
    assert "tok-999" not in sanitized
    assert "[REDACTED]" in sanitized


# ---------------------------------------------------------------------------
# Freshness: as_of_date, data_freshness, environment
# ---------------------------------------------------------------------------


def _short_row(settlement_date: str, quantity: int = 100):
    return {
        "symbolCode": "AAPL",
        "settlementDate": settlement_date,
        "currentShortPositionQuantity": quantity,
    }


def test_briefing_fresh_current(http: dict[str, MagicMock]) -> None:
    rows = [_short_row("2026-08-14"), _short_row("2026-08-01")]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["as_of_date"] == "2026-08-14"
    assert result["data_freshness"] == "current"
    assert result["environment"] == "production"
    assert not any("STALE" in w for w in _as_seq(result["warnings"]))


def test_briefing_fresh_stale(http: dict[str, MagicMock]) -> None:
    rows = [_short_row("2025-12-01"), _short_row("2025-11-01")]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["as_of_date"] == "2025-12-01"
    assert result["data_freshness"] == "stale"
    assert any("STALE" in w for w in _as_seq(result["warnings"]))
    stale_warnings = _as_seq(result["warnings"])
    assert "historical" in " ".join(stale_warnings).lower()


def test_briefing_no_date_field_unknown(http: dict[str, MagicMock]) -> None:
    rows = [{"registrationTypeCode": "BD", "firmCount": 10}]
    http["post"].side_effect = [
        _token_response(),
        _response(rows),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "finra/industrySnapshotFirmsByRegistrationType", "limit": 1},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["as_of_date"] is None
    assert result["data_freshness"] == "unknown"
    assert result["environment"] == "production"


def test_datapoints_fresh_stale_warning(http: dict[str, MagicMock]) -> None:
    rows = [_short_row("2025-12-01")]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "filters": [{"field": "settlementDate", "value": "2025-12-01"}],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["as_of_date"] == "2025-12-01"
    assert result["data_freshness"] == "stale"
    assert any("STALE" in w for w in _as_seq(result["warnings"]))

    rendered = render_tool_result(result)
    assert "STALE/HISTORICAL DATA" in rendered
    assert "As of: 2025-12-01" in rendered
    assert "Environment: production" in rendered


def test_environment_marker_mock_mode(http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRA_USE_MOCK", "1")
    rows = [_short_row("2026-08-14")]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "filters": [{"field": "settlementDate", "value": "2026-08-14"}],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["environment"] == "mock"


def test_briefing_render_shows_as_of_and_environment() -> None:
    from tests.test_tool_render import _briefing_result

    result = _briefing_result(total_records=12)
    result["as_of_date"] = "2026-08-14"
    result["data_freshness"] = "current"
    result["environment"] = "production"
    text = render_tool_result(result)
    assert "As of: 2026-08-14 (freshness: current)" in text
    assert "Environment: production" in text
    assert "STALE" not in text
