"""Tests for the private FINRA analysis layer.

Offline and deterministic: FINRA HTTP is mocked via the shared fixtures.
Briefings are pure deterministic functions of spec plus rows. No live
FINRA calls.
"""

import json
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from app import finra_analysis, finra_client
from app import tools as tools_module
from app.policy import LOCAL_CONTEXT
from app.tools import execute_tool
from tests.test_finra import (
    FakeCache,
    _as_dict,
    _as_seq,
    _error,
    _mock_get,
    _response,
    _token_response,
    _tool_names,
)


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


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    get_mock = MagicMock(side_effect=_mock_get)
    post_mock = MagicMock()
    monkeypatch.setattr("app.finra_client.requests.get", get_mock)
    monkeypatch.setattr("app.finra_client.requests.post", post_mock)
    monkeypatch.setattr("app.finra_client.get_finra_client_id", lambda: "client")
    monkeypatch.setattr("app.finra_client.get_finra_client_secret", lambda: "secret")
    return {"get": get_mock, "post": post_mock}


def _data_body(post_mock: MagicMock):
    return post_mock.call_args_list[-1].kwargs["json"]


def _short_interest_rows(n: int = 3):
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbolCode": "AAPL",
                "issueName": "Apple Inc.",
                "settlementDate": f"2026-08-{14 - i:02d}",
                "currentShortPositionQuantity": 100 + i * 50,
                "previousShortPositionQuantity": 90 + i * 50,
                "averageDailyVolumeQuantity": 10_000,
                "daysToCoverQuantity": 1.2 + i * 0.1,
                "shortInterestChangePercentage": 5.0 + i,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Briefings by default: no raw records in tool results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,args",
    [
        ("query_finra", {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"}),
        ("get_short_interest", {"ticker": "AAPL"}),
        ("get_reg_sho_volume", {"ticker": "AAPL"}),
        ("get_threshold_securities", {"ticker": "AAPL"}),
    ],
)
def test_analysis_tools_never_return_raw_records(
    http: dict[str, MagicMock], tool: str, args: dict[str, object]
) -> None:
    rows = (
        [{"issueSymbolIdentifier": "AAPL", "tradeDate": "2026-08-14", "issueName": "RawCo Inc."}]
        if tool == "get_threshold_securities"
        else [{"symbolCode": "AAPL", "issueName": "RawCo Inc.", "settlementDate": "2026-08-14"}]
    )
    if tool == "get_reg_sho_volume":
        rows = [
            {
                "securitiesInformationProcessorSymbolIdentifier": "AAPL",
                "tradeReportDate": "2026-08-14",
                "issueName": "RawCo Inc.",
            }
        ]
    http["post"].side_effect = [_token_response(), _response(rows)]

    result = execute_tool(tool, args, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    assert "records" not in result
    assert "coverage" in result
    assert "metrics" in result
    assert "warnings" in result
    assert result["briefing_source"] == "deterministic_only"
    serialized = json.dumps(result)
    assert '"records":' not in serialized
    assert "RawCo Inc." not in json.dumps({key: value for key, value in result.items() if key != "latest_record"})
    assert result["latest_record"] == rows[0]


# ---------------------------------------------------------------------------
# Deterministic summaries
# ---------------------------------------------------------------------------


def test_numeric_summaries(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response(_short_interest_rows(3)),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    stats = _as_dict(_as_dict(_as_dict(result["metrics"])["fields"])["currentShortPositionQuantity"])
    assert stats["min"] == 100
    assert stats["max"] == 200
    assert stats["mean"] == 150
    assert stats["median"] == 150
    assert stats["sum"] == 450


def test_latest_vs_prior_change_and_percent(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response(_short_interest_rows(3)),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    lvp = _as_dict(result["metrics"])["latest_vs_prior"]
    entry = next(e for e in lvp if e["field"] == "currentShortPositionQuantity")
    # _short_interest_rows: newest date (08-14) has value 100, prior 150.
    assert entry["latest"] == 100
    assert entry["prior"] == 150
    assert entry["change"] == -50
    assert entry["change_percent"] == pytest.approx(-33.33, abs=0.01)
    assert entry["latest_date"] == "2026-08-14"
    assert entry["prior_date"] == "2026-08-13"


def test_date_aware_ordering_and_coverage_dates(http: dict[str, MagicMock]) -> None:
    rows = list(reversed(_short_interest_rows(3)))
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    coverage = _as_dict(result["coverage"])
    assert coverage["first_date"] == "2026-08-12"
    assert coverage["last_date"] == "2026-08-14"
    # Latest-vs-prior reflects the newest date even though input was unsorted.
    entry = next(
        e
        for e in _as_seq(_as_dict(result["metrics"])["latest_vs_prior"])
        if e["field"] == "currentShortPositionQuantity"
    )
    assert entry["latest_date"] == "2026-08-14"
    assert entry["latest"] == 100


def test_categorical_breakdown(http: dict[str, MagicMock]) -> None:
    rows = [
        {"tradeDate": "2026-08-14", "productCategory": "Bills", "totalParAmountTraded": 10},
        {"tradeDate": "2026-08-14", "productCategory": "Notes", "totalParAmountTraded": 20},
        {"tradeDate": "2026-08-14", "productCategory": "Notes", "totalParAmountTraded": 30},
    ]
    http["post"].side_effect = [
        _token_response(),
        _response(rows),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "fixedIncomeMarket/treasuryDailyAggregates",
            "start_date": "2026-08-14",
            "end_date": "2026-08-14",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert _as_dict(_as_dict(_as_dict(result["metrics"])["categorical"])["productCategory"]) == {
        "Notes": 2,
        "Bills": 1,
    }


def test_missing_value_warnings(http: dict[str, MagicMock]) -> None:
    rows = [
        {"symbolCode": "AAPL", "settlementDate": "2026-08-14"},
        {"symbolCode": "AAPL", "settlementDate": "2026-08-13"},
    ]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert any("currentShortPositionQuantity" in w and "2/2" in w for w in _as_seq(result["warnings"]))


def test_partial_coverage_warning(http: dict[str, MagicMock]) -> None:
    rows = [
        {"symbolCode": "AAPL", "settlementDate": "2026-08-14"} for _ in range(finra_analysis.ANALYSIS_MAX_RECORDS + 5)
    ]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert _as_dict(result["coverage"])["complete"] is False
    assert _as_dict(result["coverage"])["rows_matched"] == finra_analysis.ANALYSIS_MAX_RECORDS + 5
    assert _as_dict(result["coverage"])["rows_analyzed"] == finra_analysis.ANALYSIS_MAX_RECORDS
    assert any("cap" in w for w in _as_seq(result["warnings"]))


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_header_driven(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(2)
    http["post"].side_effect = [
        _token_response(),
        _response(rows, headers={"Record-Total": "17", "Record-Limit": "2"}),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL", "limit": 2},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["total_records"] == 17
    assert result["may_have_more"] is True
    assert result["pagination_source"] == "finra_header"


def test_pagination_estimate_when_header_absent(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(2)
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL", "limit": 2},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["total_records"] is None
    assert result["pagination_source"] == "estimate"
    assert result["may_have_more"] is True  # full page -> estimate


# ---------------------------------------------------------------------------
# Coverage: page vs full-query completeness
# ---------------------------------------------------------------------------


def test_coverage_query_incomplete_when_total_exceeds_page(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(3)
    http["post"].side_effect = [
        _token_response(),
        _response(rows, headers={"Record-Total": "17", "Record-Limit": "3"}),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "ticker": "AAPL",
            "limit": 3,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    cov = _as_dict(result["coverage"])
    assert cov["rows_matched"] == 3
    assert cov["rows_analyzed"] == 3
    assert cov["page_complete"] is True  # all page rows analyzed
    assert cov["query_complete"] is False  # 17 matches, page holds 3
    assert cov["analysis_complete"] is False  # metrics miss 14 records


def test_coverage_complete_single_page(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(3)
    http["post"].side_effect = [
        _token_response(),
        _response(rows, headers={"Record-Total": "3"}),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "ticker": "AAPL",
            "limit": 3,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    cov = _as_dict(result["coverage"])
    assert cov["rows_matched"] == 3
    assert cov["page_complete"] is True
    assert cov["query_complete"] is True
    assert cov["analysis_complete"] is True


def test_coverage_analysis_incomplete_at_internal_cap(http: dict[str, MagicMock]) -> None:
    rows = [
        {"symbolCode": "AAPL", "settlementDate": "2026-08-14"} for _ in range(finra_analysis.ANALYSIS_MAX_RECORDS + 5)
    ]
    http["post"].side_effect = [
        _token_response(),
        _response(rows, headers={"Record-Total": str(len(rows))}),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    cov = _as_dict(result["coverage"])
    # Page is complete (all matches fit on it) but the analysis cap stopped
    # deterministic metrics short — analysis_complete must be False.
    assert cov["query_complete"] is True
    assert cov["page_complete"] is False
    assert cov["analysis_complete"] is False
    assert cov["rows_analyzed"] == finra_analysis.ANALYSIS_MAX_RECORDS
    assert cov["cap"] == finra_analysis.ANALYSIS_MAX_RECORDS


def test_coverage_unknown_when_record_total_missing(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(3)
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "ticker": "AAPL",
            "limit": 3,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    cov = _as_dict(result["coverage"])
    assert cov["rows_matched"] == 3
    assert cov["page_complete"] is True
    # Completeness cannot be proven without Record-Total: explicit nulls.
    assert cov["query_complete"] is None
    assert cov["analysis_complete"] is None
    assert any("estimated" in w.lower() or "Record-Total" in w for w in _as_seq(result["warnings"]))
    assert result["pagination_source"] == "estimate"


# ---------------------------------------------------------------------------
# get_finra_datapoints: exact data on request


def test_datapoints_missing_fields_rejected(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "fields" in _error(result)
    # The schema validator rejects the missing required "fields" argument
    # before any network call (no token POST either).
    assert http["post"].call_count == 0


def test_datapoints_empty_fields_rejected(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": [],
            "ticker": "AAPL",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "fields" in _error(result)


def test_datapoints_unbounded_rejected(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "narrowing" in _error(result)
    assert http["post"].call_count == 1  # no data POST


def test_datapoints_unknown_field_rejected(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["notARealField"],
            "ticker": "AAPL",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "notARealField" in _error(result)
    assert "symbolCode" in _error(result)  # known fields listed


def test_datapoints_forwards_only_selected_fields(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(2)
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "limit": 2,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    body = _data_body(http["post"])
    assert body["fields"] == ["settlementDate", "currentShortPositionQuantity"]
    assert set(_as_seq(result["records"])[0]) == {
        "settlementDate",
        "currentShortPositionQuantity",
    }
    assert result["returned_count"] == 2


def test_datapoints_default_limit_ten_and_max_twenty_five(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(30)
    http["post"].side_effect = [_token_response(), _response(rows), _response(rows)]
    default_result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in default_result, default_result
    assert _data_body(http["post"])["limit"] == 10
    assert default_result["returned_count"] == 10

    big_result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "limit": 999,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in big_result, big_result
    assert _data_body(http["post"])["limit"] == 25
    assert big_result["returned_count"] == 25


def test_datapoints_pagination_metadata(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(2)
    http["post"].side_effect = [
        _token_response(),
        _response(rows, headers={"Record-Total": "9"}),
    ]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "limit": 2,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["total_records"] == 9
    assert result["may_have_more"] is True
    assert result["pagination_source"] == "finra_header"
    assert result["next_offset"] == 2
    # Datapoints is the exact-data path: raw selected fields are expected.
    assert _as_seq(result["records"])[0]["settlementDate"] == "2026-08-14"


def test_datapoints_sort_fields_in_payload(http: dict[str, MagicMock]) -> None:
    rows = _short_interest_rows(2)
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate", "currentShortPositionQuantity"],
            "ticker": "AAPL",
            "limit": 5,
            # An EQUAL filter on the only partition field (settlementDate)
            # makes server-side sortFields valid for FINRA.
            "filters": [{"field": "settlementDate", "value": "2026-08-14"}],
            "sort_fields": ["-settlementDate"],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    body = _data_body(http["post"])
    assert body["sortFields"] == ["-settlementDate"]
    assert result["sort_fields"] == ["-settlementDate"]
    assert result.get("sort_source") is None  # server-side sort, no partition walk


def test_datapoints_invalid_sort_field_rejected_before_http(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "sort_fields": ["-notARealField"],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "notARealField" in _error(result)
    assert "settlementDate" in _error(result)  # known fields listed
    assert http["post"].call_count == 1  # no data POST


def test_datapoints_sort_order_requires_date_field(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "finra/industrySnapshotFirmsByRegistrationType",
            "fields": ["registrationTypeCode"],
            "filters": [{"field": "registrationTypeCode", "value": "BD"}],
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "no date field" in _error(result)
    assert "sort_fields" in _error(result)
    assert http["post"].call_count == 1  # no data POST


def test_datapoints_sort_order_and_sort_fields_conflict(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": ["settlementDate"],
            "ticker": "AAPL",
            "sort_fields": ["-settlementDate"],
            "sort_order": "desc",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "not both" in _error(result)
    assert http["post"].call_count == 1


def _data_posts(post_mock: MagicMock):
    return [
        c.kwargs["json"]
        for c in post_mock.call_args_list
        if "oauth2/access_token" not in (c.args[0] if c.args else c.kwargs.get("url", ""))
    ]


def test_datapoints_latest_five_returns_descending_dates(http: dict[str, MagicMock]) -> None:
    # No EQUAL settlementDate filter: FINRA rejects unrestricted sortFields,
    # so the client must walk partitions newest-first without sortFields.
    partition_rows = {
        "2026-08-14": [
            {"symbolCode": "AAPL", "settlementDate": "2026-08-14", "currentShortPositionQuantity": 100},
            {"symbolCode": "AAPL", "settlementDate": "2026-08-14", "currentShortPositionQuantity": 200},
        ],
        "2026-08-07": [
            {"symbolCode": "AAPL", "settlementDate": "2026-08-07", "currentShortPositionQuantity": 500},
            {"symbolCode": "AAPL", "settlementDate": "2026-08-07", "currentShortPositionQuantity": 600},
            {"symbolCode": "AAPL", "settlementDate": "2026-08-07", "currentShortPositionQuantity": 700},
        ],
    }

    def _respond(url: str, **kw: object):
        if "oauth2/access_token" in url:
            return _token_response()
        payload = kw["json"]
        assert isinstance(payload, dict)
        date_filter = next(f["fieldValue"] for f in payload["compareFilters"] if f["fieldName"] == "settlementDate")
        return _response(partition_rows[date_filter])

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
    assert "error" not in result, result
    # Two partition queries (08-14 fills 2, 08-07 fills the remaining 3),
    # then the walk stops — never an unrestricted sortFields payload.
    posts = _data_posts(http["post"])
    assert len(posts) == 2
    assert all("sortFields" not in p for p in posts)
    assert posts[0]["compareFilters"] == [
        {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": "AAPL"},
        {"compareType": "EQUAL", "fieldName": "settlementDate", "fieldValue": "2026-08-14"},
    ]
    assert posts[1]["compareFilters"] == [
        {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": "AAPL"},
        {"compareType": "EQUAL", "fieldName": "settlementDate", "fieldValue": "2026-08-07"},
    ]
    dates = [r["settlementDate"] for r in _as_seq(result["records"])]
    assert dates == sorted(dates, reverse=True)
    assert dates[0] == "2026-08-14"
    assert result["sort_source"] == "partitions"
    assert result["partition_queries"] == 2
    assert result["pagination_source"] == "partitions"
    assert result["data_freshness"] in ("current", "stale")


def test_datapoints_more_than_ten_fields_rejected(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    fields = [
        "issueSymbolIdentifier",
        "issueName",
        "firmCRDNumber",
        "MPID",
        "marketParticipantName",
        "tierIdentifier",
        "tierDescription",
        "summaryStartDate",
        "totalWeeklyTradeCount",
        "totalWeeklyShareQuantity",
        "productTypeCode",  # 11th valid field on weeklySummary
    ]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/weeklySummary",
            "fields": fields,
            "ticker": "AAPL",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "10" in _error(result)
    assert http["post"].call_count == 1  # no data POST


def test_datapoints_ten_fields_accepted(http: dict[str, MagicMock]) -> None:
    rows = [{"symbolCode": "AAPL", "settlementDate": "2026-08-14"}]
    http["post"].side_effect = [_token_response(), _response(rows)]
    fields = [
        "symbolCode",
        "issueName",
        "settlementDate",
        "currentShortPositionQuantity",
        "previousShortPositionQuantity",
        "averageDailyVolumeQuantity",
        "daysToCoverQuantity",
        "shortInterestChangePercentage",
    ]
    result = execute_tool(
        "get_finra_datapoints",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "fields": fields,
            "ticker": "AAPL",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result


# ---------------------------------------------------------------------------
# Schema / dispatch parity includes the new tool
# ---------------------------------------------------------------------------


def test_finra_schema_dispatch_parity() -> None:
    finra_names = {
        "get_short_interest",
        "get_short_interest_leaderboard",
        "get_reg_sho_volume",
        "get_threshold_securities",
        "list_finra_datasets",
        "describe_finra_dataset",
        "get_finra_datapoints",
        "query_finra",
    }
    schema_names = _tool_names(tools_module.TOOLS)
    assert finra_names <= schema_names, "FINRA schema missing from TOOLS"
    assert set(tools_module._FINRA_HANDLERS) == finra_names, "FINRA dispatch registry out of sync with schemas"
    assert all(callable(h) for h in tools_module._FINRA_HANDLERS.values())


@pytest.mark.parametrize(
    "dataset,symbol,date",
    [
        ("otcMarket/consolidatedShortInterest", "symbolCode", "settlementDate"),
        ("otcMarket/regShoDaily", "securitiesInformationProcessorSymbolIdentifier", "tradeReportDate"),
    ],
)
def test_latest_record_uses_newest_date_and_only_scalars(
    http: dict[str, MagicMock], dataset: str, symbol: str, date: str
) -> None:
    rows = [
        {symbol: "NVDA", date: "2026-08-12", "quantity": 10},
        {symbol: "NVDA", date: "2026-08-14", "quantity": 30, "nested": {"v": 1}, "array": [1]},
        {symbol: "NVDA", date: "2026-08-13", "quantity": 20},
    ]
    http["post"].side_effect = [_token_response(), _response(rows)]
    result = finra_client.query_dataset(dataset, ticker="NVDA")
    assert "error" not in result, result
    assert result["latest_record"] == {symbol: "NVDA", date: "2026-08-14", "quantity": 30}


def test_market_query_does_not_select_one_latest_record(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response(), _response(_short_interest_rows())]
    result = finra_client.query_dataset("otcMarket/consolidatedShortInterest")
    assert "error" not in result, result
    assert "latest_record" not in result
