"""Unit tests for the FINRA Query API tools.

Deterministic and offline: all FINRA HTTP is mocked against sanitized,
source-controlled fixtures in tests/fixtures/finra/. Live verification is
opt-in via tests/test_finra_smoke.py (pytest -m finra_smoke).
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from app import finra_client
from app import tools as tools_module
from app.config import FINRA_API_BASE, FINRA_TOKEN_URL
from app.policy import LOCAL_CONTEXT
from app.tools import execute_tool


def _as_seq(value: object):
    """list/tuple from a FINRA tool-result envelope (app/tools.py boundary)."""
    assert isinstance(value, (list, tuple))
    return value


def _as_dict(value: object):
    """dict from a FINRA tool-result envelope (app/tools.py boundary)."""
    assert isinstance(value, dict)
    return value


def _error(result: dict[str, object]) -> str:
    """error string from a FINRA tool-result envelope (app/tools.py boundary)."""
    error = result["error"]
    assert isinstance(error, str)
    return error


def _tool_names(schemas: object) -> set[str]:
    """Tool-schema names from untyped TOOLS data (app/tools.py boundary)."""
    assert isinstance(schemas, list)
    names: set[str] = set()
    for entry in schemas:
        assert isinstance(entry, dict)
        function = entry.get("function")
        assert isinstance(function, dict)
        name = function.get("name")
        assert isinstance(name, str)
        names.add(name)
    return names


FIXTURES = Path(__file__).parent / "fixtures" / "finra"


def _load_catalog() -> dict[str, object]:
    raw: object = json.loads((FIXTURES / "catalog.json").read_text())
    assert isinstance(raw, dict)
    return raw


def _load_metadata(group: str, name: str):
    path = FIXTURES / "metadata" / f"{group}__{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_partitions(group: str, name: str):
    path = FIXTURES / "partitions" / f"{group}__{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _response(payload: object = None, status: int = 200, headers: dict[str, str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload) if payload is not None else ""
    resp.headers = headers or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _token_response() -> MagicMock:
    return _response({"access_token": "tok-123", "expires_in": 3600})


class FakeCache:
    """In-memory stand-in for app.cache that records every access."""

    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[str] = []

    def get(self, key: str, ttl: float | None = None) -> object | None:
        self.get_calls.append(key)
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value
        self.set_calls.append(key)


def _mock_get(url: str, **_kwargs: object) -> MagicMock:
    if url.rstrip("/").endswith("/datasets"):
        return _response(_load_catalog())
    if "/metadata/group/" in url:
        group, name = url.split("/metadata/group/")[1].split("/name/")
        meta = _load_metadata(group, name)
        if meta is None:
            return _response({"error": "not found"}, status=404)
        return _response(meta)
    if "/partitions/group/" in url:
        group, name = url.split("/partitions/group/")[1].split("/name/")
        parts = _load_partitions(group, name)
        if parts is None:
            return _response({"error": "not found"}, status=404)
        return _response(parts)
    raise AssertionError(f"Unexpected GET {url}")


@pytest.fixture(autouse=True)
def _isolation(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FINRA_USE_MOCK", "")
    # Never let the analysis layer reach a live secondary model in offline tests.
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


def _entry(list_result: dict[str, object], dataset_id: str):
    datasets = list_result["datasets"]
    assert isinstance(datasets, list)
    return next((d for d in datasets if d["dataset"] == dataset_id), None)


def _data_body(post_mock: MagicMock):
    return post_mock.call_args_list[-1].kwargs["json"]


# ---------------------------------------------------------------------------
# Direct helper tools
# ---------------------------------------------------------------------------


def test_short_interest_payload(http: dict[str, MagicMock]) -> None:
    records = [
        {
            "symbolCode": "AAPL",
            "issueName": "Apple Inc.",
            "settlementDate": "2026-08-14",
            "currentShortPositionQuantity": 100,
            "daysToCoverQuantity": 1.2,
        }
    ]
    http["post"].side_effect = [_token_response(), _response(records)]

    result = execute_tool(
        "get_short_interest",
        {"ticker": "aapl", "settlementDate": "2026-08-14"},
        model="test",
        context=LOCAL_CONTEXT,
    )

    assert "error" not in result, result
    assert result["dataset"] == "consolidatedShortInterest"
    assert result["group"] == "otcMarket"
    assert "records" not in result
    source = result["source"]
    assert isinstance(source, str)
    assert "FINRA" in source
    assert result["returned_count"] == 1
    assert result["offset"] == 0
    assert result["next_offset"] == 1
    assert result["may_have_more"] is False
    # Briefing shape: coverage + deterministic metrics, no raw rows.
    coverage = _as_dict(result["coverage"])
    assert coverage["rows_matched"] == 1
    assert coverage["complete"] is True
    assert coverage["first_date"] == "2026-08-14"
    assert _as_dict(_as_dict(_as_dict(result["metrics"])["fields"])["currentShortPositionQuantity"]) == {
        "min": 100,
        "max": 100,
        "mean": 100,
        "median": 100,
        "sum": 100,
    }
    assert _as_dict(result["metrics"])["latest_vs_prior"] == []
    assert result["briefing"] is None
    assert result["briefing_source"] == "deterministic_only"
    assert result["total_records"] is None
    assert result["pagination_source"] == "estimate"
    # The main model's tool message must never contain raw records.
    serialized = json.dumps(result)
    assert '"records":' not in serialized
    assert '"symbolCode": "AAPL"' not in serialized
    assert '"issueName": "Apple Inc."' not in serialized

    # posts: one token (catalog auth) + one data query.
    assert http["post"].call_count == 2
    token_call, data_call = http["post"].call_args_list
    token_url = token_call.args[0] if token_call.args else token_call.kwargs["url"]
    data_url = data_call.args[0] if data_call.args else data_call.kwargs["url"]
    assert "oauth2/access_token" in token_url
    assert data_url.endswith("/data/group/otcMarket/name/consolidatedShortInterest")
    assert data_call.kwargs["json"]["compareFilters"] == [
        {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": "AAPL"},
        {
            "compareType": "EQUAL",
            "fieldName": "settlementDate",
            "fieldValue": "2026-08-14",
        },
    ]


def test_latest_short_interest_rejects_stale_production_data(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    stale_date = (datetime.now(UTC).date() - timedelta(days=finra_client.STALE_AFTER_DAYS + 1)).isoformat()
    http["post"].side_effect = [
        _token_response(),
        _response(
            [
                {
                    "symbolCode": "AAPL",
                    "settlementDate": stale_date,
                    "currentShortPositionQuantity": 100,
                }
            ]
        ),
    ]

    def _fake_partitions(*args: object) -> tuple[list[dict[str, str]], dict[str, object], int, bool]:
        return ([{"settlementDate": stale_date}], {}, 1, False)

    monkeypatch.setattr(
        finra_client,
        "_datapoints_via_partitions",
        _fake_partitions,
    )

    result = execute_tool("get_short_interest", {"ticker": "AAPL"}, model="test", context=LOCAL_CONTEXT)

    assert "error" in result
    assert "Current FINRA short interest is unavailable" in _error(result)
    assert stale_date in _error(result)


# ---------------------------------------------------------------------------
# Catalog → describe → query flow
# ---------------------------------------------------------------------------


def test_full_catalog_describe_query_flow(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"issueSymbolIdentifier": "AAPL", "tradeDate": "2026-08-14"}]),
    ]

    listed = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in listed, listed
    entry = _entry(listed, "otcMarket/thresholdList")
    assert entry is not None
    assert entry["access"] == "unknown"

    described = execute_tool(
        "describe_finra_dataset", {"dataset_id": "otcMarket/thresholdList"}, model="test", context=LOCAL_CONTEXT
    )
    assert "error" not in described, described
    assert described["ticker_field"] == "issueSymbolIdentifier"
    assert described["date_field"] == "tradeDate"
    assert {f["name"] for f in _as_seq(described["fields"])} >= {
        "issueSymbolIdentifier",
        "tradeDate",
        "issueName",
    }

    queried = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/thresholdList",
            "ticker": "AAPL",
            "start_date": "2026-08-14",
            "end_date": "2026-08-14",
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in queried, queried
    body = _data_body(http["post"])
    assert {
        "compareType": "EQUAL",
        "fieldName": "issueSymbolIdentifier",
        "fieldValue": "AAPL",
    } in body["compareFilters"]
    assert {
        "compareType": "EQUAL",
        "fieldName": "tradeDate",
        "fieldValue": "2026-08-14",
    } in body["compareFilters"]
    assert queried["returned_count"] == 1
    assert "records" not in queried
    queried_coverage = _as_dict(queried["coverage"])
    assert queried_coverage["rows_matched"] == 1
    assert queried_coverage["first_date"] == "2026-08-14"
    assert queried_coverage["last_date"] == "2026-08-14"
    assert queried["briefing_source"] == "deterministic_only"


# ---------------------------------------------------------------------------
# Catalog listing: access reporting + capability honesty + exclusions
# ---------------------------------------------------------------------------


def test_list_finra_datasets(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    names = {d["dataset"] for d in _as_seq(result["datasets"])}
    assert "otcMarket/consolidatedShortInterest" in names
    assert "otcMarket/regShoDaily" in names
    assert "fixedIncomeMarket/treasuryDailyAggregates" in names
    assert "finra/industrySnapshotFirmsByRegistrationType" in names

    sample = _entry(result, "otcMarket/weeklySummary")
    assert sample is not None
    assert "records" not in sample
    assert "fields" not in sample
    assert sample["access"] == "unknown"
    assert sample["supports_record_offset"] is True


def test_list_excludes_entitled_and_retired(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    names = {d["dataset"] for d in _as_seq(result["datasets"])}
    # Confirmed non-public (accessType=Firm) and retired entries are dropped.
    assert "registration/firmRegistrationsSample" not in names
    assert "otcMarket/retiredDatasetSample" not in names


def test_access_reporting_public_and_unknown(http: dict[str, MagicMock]) -> None:
    listed = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    snapshot = _entry(listed, "finra/industrySnapshotFirmsByRegistrationType")
    assert snapshot is not None
    assert snapshot["access"] == "public"  # isPublic: true in fixture
    weekly_unknown = _entry(listed, "otcMarket/weeklySummary")
    assert weekly_unknown is not None
    assert weekly_unknown["access"] == "unknown"
    finra_client.reset_discovery_cache()
    described = execute_tool(
        "describe_finra_dataset",
        {"dataset_id": "finra/industrySnapshotFirmsByRegistrationType"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert described.get("access") == "public"


def test_capabilities_not_guessed(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)

    # No override for this dataset: capabilities must be null, not guessed.
    snapshot = _entry(result, "finra/industrySnapshotFirmsByRegistrationType")
    assert snapshot is not None
    assert snapshot["supports_ticker"] is None
    assert snapshot["supports_date"] is None

    # Verified override corrections still surface.
    weekly = _entry(result, "otcMarket/weeklySummary")
    assert weekly is not None
    assert weekly["supports_ticker"] is True
    treasury = _entry(result, "fixedIncomeMarket/treasuryDailyAggregates")
    assert treasury is not None
    assert treasury["supports_ticker"] is False  # market-wide aggregate
    assert treasury["supports_date"] is True


def test_list_finra_datasets_group_search(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response(), _token_response()]
    by_group = execute_tool("list_finra_datasets", {"group": "fixedIncomeMarket"}, model="test", context=LOCAL_CONTEXT)
    assert all(d["group"] == "fixedIncomeMarket" for d in _as_seq(by_group["datasets"]))

    finra_client.reset_discovery_cache()
    by_search = execute_tool("list_finra_datasets", {"search": "short interest"}, model="test", context=LOCAL_CONTEXT)
    assert any("consolidatedShortInterest" in d["dataset"] for d in _as_seq(by_search["datasets"]))


def test_catalog_search_token_ranked_weekly_summary(http: dict[str, MagicMock]) -> None:
    """'AAPL OTC weekly trading volume' resolves to weeklySummary first.

    Token-based matching: friendly labels ("trading volume" -> volume,
    "weekly", "otc") are normalized and entries are ranked by coverage.
    """
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "list_finra_datasets", {"search": "AAPL OTC weekly trading volume"}, model="test", context=LOCAL_CONTEXT
    )
    assert "error" not in result, result
    assert result["datasets"], "expected ranked matches"
    ranked = _as_seq(result["datasets"])
    assert ranked[0]["dataset"] == "otcMarket/weeklySummary", [d["dataset"] for d in ranked]
    assert ranked[0]["supports_ticker"] is True


def test_catalog_search_ranked_scores_and_filtering(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("list_finra_datasets", {"search": "OTC weekly volume"}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    datasets = [d["dataset"] for d in _as_seq(result["datasets"])]
    assert "otcMarket/weeklySummary" in datasets
    assert "otcMarket/weeklySummaryHistoric" in datasets
    # weeklySummary outranks generic volume matches (name+group+description).
    assert datasets.index("otcMarket/weeklySummary") < datasets.index("fixedIncomeMarket/treasuryDailyAggregates")

    finra_client.reset_discovery_cache()
    unmatched = execute_tool(
        "list_finra_datasets", {"search": "zzzz no such topic"}, model="test", context=LOCAL_CONTEXT
    )
    assert unmatched["datasets"] == []


def test_catalog_search_phrase_alias_volume(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("list_finra_datasets", {"search": "trading volume"}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    assert "otcMarket/weeklySummary" in {d["dataset"] for d in _as_seq(result["datasets"])}


# ---------------------------------------------------------------------------
# Describe
# ---------------------------------------------------------------------------


def test_describe_finra_dataset(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "describe_finra_dataset",
        {"dataset_id": "otcMarket/weeklySummary"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["dataset"] == "otcMarket/weeklySummary"
    assert result["ticker_field"] == "issueSymbolIdentifier"
    assert result["date_field"] == "summaryStartDate"
    assert result["supports_record_offset"] is True
    field_names = {f["name"] for f in _as_seq(result["fields"])}
    assert "summaryTypeCode" in field_names
    valid_values = result.get("valid_filter_values", {})
    assert isinstance(valid_values, dict)
    assert "summaryTypeCode" in valid_values
    default_filters = result.get("default_filters", [])
    assert isinstance(default_filters, list)
    assert any(d["field"] == "summaryTypeCode" and d["value"] == "OTC_W_SMBL" for d in default_filters)


# ---------------------------------------------------------------------------
# Query behavior: dates, filters, limits
# ---------------------------------------------------------------------------


def test_query_finra_date_range(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"tradeDate": "2026-08-01", "productCategory": "Bills"}]),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "treasuryDailyAggregates",
            "start_date": "2026-08-01",
            "end_date": "2026-08-07",
            "limit": 5,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    body = _data_body(http["post"])
    assert body["limit"] == 5
    assert body["dateRangeFilters"] == [
        {
            "fieldName": "tradeDate",
            "startDate": "2026-08-01",
            "endDate": "2026-08-07",
        }
    ]


def test_max_limit_clamped(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"tradeDate": "2026-08-01"}]),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "fixedIncomeMarket/treasuryDailyAggregates",
            "start_date": "2026-08-01",
            "end_date": "2026-08-01",
            "limit": 99999,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert _data_body(http["post"])["limit"] == finra_client.MAX_LIMIT
    assert result["limit"] == finra_client.MAX_LIMIT


def test_valid_filters_included(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"issueSymbolIdentifier": "AAPL"}]),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/weeklySummary",
            "filters": [
                {
                    "field": "totalWeeklyShareQuantity",
                    "op": "GREATER",
                    "value": "1000",
                }
            ],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert {
        "compareType": "GREATER",
        "fieldName": "totalWeeklyShareQuantity",
        "fieldValue": "1000",
    } in _data_body(http["post"])["compareFilters"]


def test_invalid_op_rejected(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/weeklySummary",
            "filters": [{"field": "totalWeeklyShareQuantity", "op": "CONTAINS", "value": "1"}],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "Unsupported compare op" in _error(result)


def test_invalid_filter_field(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/consolidatedShortInterest",
            "filters": [{"field": "notARealField", "value": "x"}],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "notARealField" in _error(result)


def test_invalid_documented_enum_rejected_before_http(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/weeklySummary",
            "filters": [{"field": "summaryTypeCode", "value": "BOGUS_TYPE"}],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "Allowed values" in _error(result)
    # Rejected before any data POST: only the catalog token request happened.
    assert http["post"].call_count == 1
    assert http["get"].call_count == 2  # catalog + metadata only


@pytest.mark.parametrize(
    "bad_filter",
    [
        {"value": "x"},  # missing field
        {"field": "summaryTypeCode"},  # missing value
        {"field": "summaryTypeCode", "value": ""},  # empty value
        "not-an-object",
    ],
)
def test_malformed_filters_rejected(http: dict[str, MagicMock], bad_filter: object) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "filters": [bad_filter]},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    lowered = _error(result).lower()
    assert "malformed" in lowered or "required" in lowered or "must be an object" in lowered
    assert http["post"].call_count == 1  # no data POST


def test_weekly_summary_default_filter_no_conflict(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"issueSymbolIdentifier": "AAPL"}]),
    ]
    # Explicit summaryTypeCode must suppress the OTC_W_SMBL default.
    result = execute_tool(
        "query_finra",
        {
            "dataset": "otcMarket/weeklySummary",
            "ticker": "AAPL",
            "filters": [{"field": "summaryTypeCode", "op": "EQUAL", "value": "ATS_W_SMBL"}],
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    type_filters = [f for f in _data_body(http["post"])["compareFilters"] if f["fieldName"] == "summaryTypeCode"]
    assert len(type_filters) == 1
    assert type_filters[0]["fieldValue"] == "ATS_W_SMBL"


def test_weekly_summary_applies_default_when_absent(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"issueSymbolIdentifier": "AAPL"}]),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "weeklySummary", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert {
        "compareType": "EQUAL",
        "fieldName": "summaryTypeCode",
        "fieldValue": "OTC_W_SMBL",
    } in _data_body(http["post"])["compareFilters"]


def test_ticker_on_non_symbol_dataset(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {"dataset": "treasuryDailyAggregates", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "ticker/symbol" in _error(result)
    assert "aggregate" in _error(result).lower()


def test_treasury_monthly_date_field(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"beginningOfTheMonthDate": "2026-08-01"}]),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "fixedIncomeMarket/treasuryMonthlyAggregates",
            "start_date": "2026-08-01",
            "end_date": "2026-08-01",
            "limit": 3,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert _data_body(http["post"])["compareFilters"] == [
        {
            "compareType": "EQUAL",
            "fieldName": "beginningOfTheMonthDate",
            "fieldValue": "2026-08-01",
        }
    ]


def test_query_finra_unknown_dataset(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("query_finra", {"dataset": "notARealDataset"}, model="test", context=LOCAL_CONTEXT)
    assert "error" in result
    assert "Unknown FINRA dataset" in _error(result)
    assert "list_finra_datasets" in _error(result)


def test_legacy_bare_name(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"symbolCode": "AAPL"}]),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["dataset"] == "consolidatedShortInterest"
    assert result["group"] == "otcMarket"


def test_ambiguous_bare_name(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool("query_finra", {"dataset": "sharedName"}, model="test", context=LOCAL_CONTEXT)
    assert "error" in result
    assert "Ambiguous" in _error(result)
    assert "groupA/sharedName" in _error(result)
    assert "groupB/sharedName" in _error(result)


def test_industry_snapshot_queries_dataset(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response(
            [
                {"registrationTypeCode": "BD", "firmCount": 10},
                {"registrationTypeCode": "IA", "firmCount": 25},
            ]
        ),
    ]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "finra/industrySnapshotFirmsByRegistrationType",
            "limit": 50,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    data_url = http["post"].call_args_list[1].args[0]
    assert data_url.endswith("/data/group/finra/name/industrySnapshotFirmsByRegistrationType")
    assert "records" not in result
    assert _as_dict(_as_dict(_as_dict(result["metrics"])["fields"])["firmCount"]) == {
        "min": 10,
        "max": 25,
        "mean": 17.5,
        "median": 17.5,
        "sum": 35,
    }
    assert _as_dict(_as_dict(_as_dict(result["metrics"])["categorical"])["registrationTypeCode"]) == {
        "IA": 1,
        "BD": 1,
    }


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_valid_full_page(http: dict[str, MagicMock]) -> None:
    records = [{"issueSymbolIdentifier": f"S{i}"} for i in range(50)]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "limit": 50, "offset": 100},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    body = _data_body(http["post"])
    assert body["offset"] == 100
    assert body["limit"] == 50
    assert result["returned_count"] == 50
    assert result["offset"] == 100
    assert result["next_offset"] == 150
    assert result["may_have_more"] is True
    assert result["total_records"] is None
    assert result["pagination_source"] == "estimate"


def test_pagination_partial_page(http: dict[str, MagicMock]) -> None:
    records = [{"issueSymbolIdentifier": f"S{i}"} for i in range(2)]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "limit": 50, "offset": 100},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["returned_count"] == 2
    assert result["next_offset"] == 102
    assert result["may_have_more"] is False


def test_pagination_header_driven(http: dict[str, MagicMock]) -> None:
    records = [{"issueSymbolIdentifier": f"S{i}"} for i in range(2)]
    http["post"].side_effect = [
        _token_response(),
        _response(
            records,
            headers={
                "Record-Total": "5",
                "Record-Offset": "0",
                "Record-Limit": "2",
                "Record-Max-Limit": "1000",
            },
        ),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "limit": 2},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["total_records"] == 5
    assert result["may_have_more"] is True
    assert result["pagination_source"] == "finra_header"


def test_pagination_header_driven_exhausted(http: dict[str, MagicMock]) -> None:
    records = [{"issueSymbolIdentifier": f"S{i}"} for i in range(5)]
    http["post"].side_effect = [
        _token_response(),
        _response(records, headers={"Record-Total": "5"}),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "limit": 5, "offset": 0},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["total_records"] == 5
    assert result["may_have_more"] is False
    assert result["pagination_source"] == "finra_header"


@pytest.mark.parametrize("offset", [-1, -100])
def test_negative_offset_rejected(http: dict[str, MagicMock], offset: int) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "offset": offset},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "offset must be >= 0" in _error(result)
    assert http["post"].call_count == 1  # no data POST


def test_offset_exceeds_finra_max(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "offset": 500001},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "500000" in _error(result)


def test_offset_rejected_when_unsupported(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [_token_response()]
    result = execute_tool(
        "query_finra",
        {
            "dataset": "finra/industrySnapshotFirmsByRegistrationType",
            "offset": 10,
        },
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "does not support record offset" in _error(result)
    assert http["post"].call_count == 1  # no data POST


# ---------------------------------------------------------------------------
# Caching: discovery and results must not re-hit HTTP
# ---------------------------------------------------------------------------


def test_discovery_and_result_cache_hits(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"issueSymbolIdentifier": "AAPL"}]),
    ]

    first_list = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in first_list, first_list
    assert http["get"].call_count == 1

    # Simulate a new process: in-memory discovery cleared, SQLite cache kept.
    finra_client.reset_discovery_cache()
    second_list = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert second_list == first_list
    assert http["get"].call_count == 1  # served from cache, no new GET

    first_query = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in first_query, first_query
    gets_after_first_query = http["get"].call_count  # catalog + metadata
    posts_after_first_query = http["post"].call_count  # token + data

    second_query = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert second_query == first_query
    assert http["get"].call_count == gets_after_first_query
    assert http["post"].call_count == posts_after_first_query


def test_token_fetched_once_across_calls(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"issueSymbolIdentifier": "AAPL"}]),
        _response([{"issueSymbolIdentifier": "AAPL"}]),
    ]
    for _ in range(2):
        result = execute_tool(
            "query_finra",
            {"dataset": "otcMarket/weeklySummary", "ticker": "AAPL"},
            model="test",
            context=LOCAL_CONTEXT,
        )
        assert "error" not in result, result
    # One token POST even across two queries (result cache serves the second).
    token_calls = [
        c
        for c in http["post"].call_args_list
        if "oauth2/access_token" in (c.args[0] if c.args else c.kwargs.get("url", ""))
    ]
    assert len(token_calls) == 1


# ---------------------------------------------------------------------------
# URLs, headers, authentication
# ---------------------------------------------------------------------------


def test_catalog_metadata_data_urls_and_auth(http: dict[str, MagicMock]) -> None:
    http["post"].side_effect = [
        _token_response(),
        _response([{"symbolCode": "AAPL"}]),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result

    catalog_call, metadata_call = http["get"].call_args_list
    catalog_url = catalog_call.args[0]
    assert catalog_url == f"{FINRA_API_BASE}/datasets"
    assert catalog_call.kwargs["headers"]["Authorization"] == "Bearer tok-123"
    assert catalog_call.kwargs["headers"]["Accept"] == "application/json"

    metadata_url = metadata_call.args[0]
    assert metadata_url.endswith("/metadata/group/otcMarket/name/consolidatedShortInterest")
    # Metadata is public: no Authorization header.
    assert "Authorization" not in metadata_call.kwargs["headers"]

    token_call, data_call = http["post"].call_args_list
    assert token_call.args[0] == FINRA_TOKEN_URL
    assert token_call.kwargs["auth"] == ("client", "secret")

    data_url = data_call.args[0]
    assert data_url.endswith("/data/group/otcMarket/name/consolidatedShortInterest")
    assert data_call.kwargs["headers"]["Authorization"] == "Bearer tok-123"
    assert data_call.kwargs["headers"]["Content-Type"] == "application/json"
    assert "compareFilters" in data_call.kwargs["json"]


def test_uppercase_catalog_paths_are_normalized_before_requests(http: dict[str, MagicMock]) -> None:
    """Live FINRA catalog values may be uppercase; endpoint paths are not."""
    catalog = _load_catalog()
    datasets = catalog["datasets"]
    assert isinstance(datasets, list)
    first_dataset = datasets[0]
    assert isinstance(first_dataset, dict)
    first_dataset["group"] = "OTCMARKET"
    first_dataset["name"] = "CONSOLIDATEDSHORTINTEREST"

    def _get(url: str, **kwargs: object):
        if url.rstrip("/").endswith("/datasets"):
            return _response(catalog)
        return _mock_get(url, **kwargs)

    http["get"].side_effect = _get
    http["post"].side_effect = [
        _token_response(),
        _response([{"symbolCode": "AAPL", "settlementDate": "2026-08-14"}]),
    ]

    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )

    assert "error" not in result, result
    assert result["dataset_id"] == "otcMarket/consolidatedShortInterest"
    metadata_url = http["get"].call_args_list[1].args[0]
    data_url = http["post"].call_args_list[1].args[0]
    assert metadata_url.endswith("/metadata/group/otcMarket/name/consolidatedShortInterest")
    assert data_url.endswith("/data/group/otcMarket/name/consolidatedShortInterest")


def test_query_cache_isolated_by_finra_environment(
    http: dict[str, MagicMock], fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = finra_client.DatasetSpec(group="otcMarket", name="consolidatedShortInterest", description="")
    payload: dict[str, object] = {"limit": 1}
    http["post"].side_effect = [
        _token_response(),
        _response([{"symbolCode": "AAPL"}]),
        _response([{"symbolCode": "AAPL"}]),
    ]

    production_key = finra_client._query_cache_key(spec, payload)
    finra_client._cached_query(spec, payload)
    monkeypatch.setenv("FINRA_USE_MOCK", "true")
    mock_key = finra_client._query_cache_key(spec, payload)
    finra_client._cached_query(spec, payload)

    assert production_key != mock_key
    assert production_key in fake_cache.store
    assert mock_key in fake_cache.store
    data_urls = [call.args[0] for call in http["post"].call_args_list if "oauth2/access_token" not in call.args[0]]
    assert data_urls == [
        f"{FINRA_API_BASE}/data/group/otcMarket/name/consolidatedShortInterest",
        f"{FINRA_API_BASE}/data/group/otcMarket/name/consolidatedShortInterestMock",
    ]


# ---------------------------------------------------------------------------
# HTTP failure modes
# ---------------------------------------------------------------------------


def test_catalog_connection_error(http: dict[str, MagicMock]) -> None:
    http["get"].side_effect = requests.ConnectionError("api.finra.org unreachable")
    result = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert "error" in result
    assert "FINRA catalog unavailable" in _error(result)


def test_catalog_http_500(http: dict[str, MagicMock]) -> None:
    def _error_response(url: str, **kw: object) -> MagicMock:
        return _response({}, status=500)

    http["get"].side_effect = _error_response
    result = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert "error" in result
    assert "FINRA catalog unavailable" in _error(result)


def test_catalog_malformed_response(http: dict[str, MagicMock]) -> None:
    def _unexpected_response(url: str, **kw: object) -> MagicMock:
        return _response({"unexpected": 1})

    http["get"].side_effect = _unexpected_response
    result = execute_tool("list_finra_datasets", {}, model="test", context=LOCAL_CONTEXT)
    assert "error" in result
    assert "FINRA catalog unavailable" in _error(result)


@pytest.mark.parametrize("status", [401, 403])
def test_metadata_auth_error_describe(http: dict[str, MagicMock], status: int) -> None:
    def side_effect(url: str, **kw: object):
        if url.rstrip("/").endswith("/datasets"):
            return _response(_load_catalog())
        return _response({}, status=status)

    http["get"].side_effect = side_effect
    result = execute_tool(
        "describe_finra_dataset",
        {"dataset_id": "otcMarket/weeklySummary"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert str(status) in _error(result)
    assert "entitlement" in _error(result).lower()


@pytest.mark.parametrize("status", [401, 403])
def test_metadata_auth_error_query(http: dict[str, MagicMock], status: int) -> None:
    def side_effect(url: str, **kw: object):
        if url.rstrip("/").endswith("/datasets"):
            return _response(_load_catalog())
        return _response({}, status=status)

    http["get"].side_effect = side_effect
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert str(status) in _error(result)
    assert "entitlement" in _error(result).lower() or "public" in _error(result).lower()


def test_metadata_malformed_response(http: dict[str, MagicMock]) -> None:
    def side_effect(url: str, **kw: object):
        if url.rstrip("/").endswith("/datasets"):
            return _response(_load_catalog())
        return _response([1, 2, 3])  # not an object

    http["get"].side_effect = side_effect
    result = execute_tool(
        "describe_finra_dataset",
        {"dataset_id": "otcMarket/weeklySummary"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "Unexpected metadata response" in _error(result)


def test_entitlement_403_on_data(http: dict[str, MagicMock]) -> None:
    forbidden = _response({}, status=403)
    http["post"].side_effect = [_token_response(), forbidden]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/consolidatedShortInterest", "ticker": "AAPL"},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" in result
    assert "403" in _error(result)
    assert "entitlement" in _error(result).lower() or "public" in _error(result).lower()


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_env(name: str):
        raise ValueError(
            f"{name} is not properly set in your environment or .env file. "
            "Phase 1 requires it to run. See .env.example."
        )

    monkeypatch.setattr("app.config._require_env", _raise_env)
    result = execute_tool("get_short_interest", {"ticker": "AAPL"}, model="test", context=LOCAL_CONTEXT)
    assert "error" in result
    assert "FINRA_CLIENT_ID" in _error(result)


# ---------------------------------------------------------------------------
# Schema / dispatch parity
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


def test_finra_mixed_case_ticker_remaps_to_aapl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed-case 'Apple' remaps via the EDGAR index; 'AAPL' passes through."""

    def _resolve(_name: str) -> str:
        return "AAPL"

    def _short(ticker: str, _settlement_date: str | None = None) -> dict[str, object]:
        return {"ticker": ticker}

    def _sho_single(ticker: str, _trade_date: str | None = None) -> dict[str, object]:
        return {"ticker": ticker}

    def _sho_week(dataset: str, ticker: str | None = None, **kwargs: object) -> dict[str, object]:
        return {"ticker": ticker}

    def _threshold(ticker: str | None = None, _trade_date: str | None = None) -> dict[str, object]:
        return {"ticker": ticker}

    monkeypatch.setattr(tools_module, "_resolve_company_to_ticker", _resolve)
    monkeypatch.setattr(finra_client, "get_short_interest", _short)
    monkeypatch.setattr(finra_client, "get_reg_sho_volume", _sho_single)
    monkeypatch.setattr(finra_client, "query_dataset", _sho_week)
    monkeypatch.setattr(finra_client, "get_threshold_securities", _threshold)
    for tool in ("get_short_interest", "get_reg_sho_volume", "get_threshold_securities"):
        assert tools_module._FINRA_HANDLERS[tool]({"ticker": "AAPL"}, "test") == {"ticker": "AAPL"}
        assert tools_module._FINRA_HANDLERS[tool]({"ticker": "Apple"}, "test") == {"ticker": "AAPL"}


def test_reg_sho_company_name_resolves_before_http(http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch) -> None:
    """ticker='Apple' -> AAPL via the EDGAR index with zero extra FINRA HTTP."""
    seen: dict[str, object] = {}
    calls: list[str] = []

    def _resolve(name: str) -> str:
        calls.append(name)
        return "AAPL"

    def _sho(dataset: str, ticker: str | None = None, **kwargs: object) -> dict[str, object]:
        seen["ticker"] = ticker
        seen["dataset"] = dataset
        return {"ticker": ticker}

    monkeypatch.setattr(tools_module, "_resolve_company_to_ticker", _resolve)
    monkeypatch.setattr(finra_client, "query_dataset", _sho)
    result = execute_tool("get_reg_sho_volume", {"ticker": "Apple"}, model="test", context=LOCAL_CONTEXT)
    assert result == {"ticker": "AAPL"}
    assert seen["ticker"] == "AAPL"
    assert calls == ["Apple"]
    assert http["post"].call_count == 0


def test_finra_rejects_non_iso_dates_with_format(http: dict[str, MagicMock]) -> None:
    """'this week' can never reach FINRA: bad dates fail closed at the handler."""
    cases: list[tuple[str, dict[str, object], str]] = [
        ("get_short_interest", {"ticker": "AAPL", "settlementDate": "this week"}, "settlementDate"),
        ("get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "this week"}, "tradeDate"),
        ("get_threshold_securities", {"tradeDate": "2026-13-45"}, "tradeDate"),
        (
            "query_finra",
            {"dataset": "otcMarket/weeklySummary", "start_date": "this week"},
            "start_date",
        ),
        (
            "get_finra_datapoints",
            {
                "dataset": "otcMarket/weeklySummary",
                "fields": ["issueSymbolIdentifier"],
                "ticker": "AAPL",
                "end_date": "not-a-date",
            },
            "end_date",
        ),
    ]
    for tool, args, key in cases:
        result = execute_tool(tool, args, model="test", context=LOCAL_CONTEXT)
        assert result["error_type"] == "invalid_tool_arguments", (tool, result)
        assert key in str(result["error"]), (tool, result)
        assert "YYYY-MM-DD" in str(result["error"]), (tool, result)
    assert http["post"].call_count == 0  # rejected before any FINRA request


def test_gate_rejects_today_now_literals_with_decode_hint(http: dict[str, MagicMock]) -> None:
    """Q3 regression: as_of 'today' must fail closed at the gate, not deep in insider/filings."""
    cases: list[tuple[str, dict[str, object]]] = [
        ("get_insider_activity", {"ticker": "AAPL", "as_of": "today"}),
        ("get_insider_activity", {"ticker": "AAPL", "as_of": "now"}),
        ("get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "today"}),
        ("get_short_interest", {"ticker": "AAPL", "settlementDate": "now"}),
        ("get_material_events", {"ticker": "AMD", "since": "today"}),
    ]
    for tool, args in cases:
        msg = tools_module._validate_tool_arguments(tool, args)
        assert msg is not None and "relative date" in msg, (tool, args, msg)
        assert "YYYY-MM-DD" in msg, (tool, args, msg)
        result = execute_tool(tool, args, model="test", context=LOCAL_CONTEXT)
        assert result["error_type"] == "invalid_tool_arguments", (tool, result)
        assert "relative date" in str(result["error"]), (tool, result)
    # Valid ISO dates still pass the relative-date gate (substring list must not false-fire).
    assert (
        tools_module._validate_tool_arguments("get_insider_activity", {"ticker": "AAPL", "as_of": "2026-09-23"}) is None
    )
    assert (
        tools_module._validate_tool_arguments("get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "2026-09-22"})
        is None
    )
    assert http["post"].call_count == 0  # gate-only: no provider request issued


def test_finra_ticker_missing_returns_none() -> None:
    assert tools_module._finra_ticker({}) is None
    assert tools_module._finra_ticker({"ticker": "   "}) is None
    assert tools_module._finra_ticker({"ticker": 123}) is None


def test_finra_missing_ticker_rejected_before_finra(http: dict[str, MagicMock]) -> None:
    for tool in ("get_short_interest", "get_reg_sho_volume"):
        missing = execute_tool(tool, {}, model="test", context=LOCAL_CONTEXT)
        assert missing["error_type"] == "invalid_tool_arguments", (tool, missing)
        blank = execute_tool(tool, {"ticker": "   "}, model="test", context=LOCAL_CONTEXT)
        assert blank["error_type"] == "invalid_tool_arguments", (tool, blank)
    assert http["post"].call_count == 0  # rejected before any FINRA request


def test_gate_rejects_bad_accession_dates_datasets_enums_before_http(http: dict[str, MagicMock]) -> None:
    """Gate-level pattern/enum: NVDA-as-accession, 'this week', bare regShoDaily, bad enums."""
    gate_cases = [
        ("diff_sec_filings", {"current_accession": "NVDA", "previous_accession": "0000320193-25-000079"}),
        ("get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "this week"}),
        ("get_short_interest", {"ticker": "AAPL", "settlementDate": "this week"}),
        ("get_short_interest_leaderboard", {"settlement_date": "this week"}),
        ("get_material_events", {"ticker": "AMD", "since": "this week"}),
        ("search_web", {"query": "AMD news", "category": "bogus"}),
        ("search_web", {"query": "AMD news", "search_type": "bogus"}),
        ("get_fundamentals", {"ticker": "AAPL", "metric": "bogus"}),
        ("get_financial_statements", {"ticker": "AAPL", "statement_type": "bogus"}),
        (
            "get_finra_datapoints",
            {"dataset": "otcMarket/weeklySummary", "fields": ["issueSymbolIdentifier"], "sort_order": "sideways"},
        ),
    ]
    for tool, args in gate_cases:
        msg = tools_module._validate_tool_arguments(tool, args)
        assert msg is not None and tool in msg, (tool, args, msg)
    # Guided accession tools bypass the gate pattern (handler owns the packet-source message).
    assert tools_module._validate_tool_arguments("get_sec_filing", {"accession_no": "NVDA"}) is None
    guided = execute_tool("get_sec_filing", {"accession_no": "NVDA"}, model="test", context=LOCAL_CONTEXT)
    assert guided["error_type"] == "invalid_tool_arguments", guided
    assert "accession_no" in str(guided["error"]), guided
    # Spot-check the named key + guided vocabulary in the message.
    assert "current_accession" in str(
        tools_module._validate_tool_arguments(
            "diff_sec_filings", {"current_accession": "NVDA", "previous_accession": "0000320193-25-000079"}
        )
    )
    assert "tradeDate" in str(
        tools_module._validate_tool_arguments("get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "this week"})
    )
    assert "YYYY-MM-DD" in str(
        tools_module._validate_tool_arguments("get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "this week"})
    )
    assert "metric" in str(tools_module._validate_tool_arguments("get_fundamentals", {"ticker": "AAPL", "metric": "x"}))
    # Unambiguous bare names resolve at the handler (legacy); unknown/ambiguous fail there.
    assert tools_module._validate_tool_arguments("describe_finra_dataset", {"dataset_id": "regShoDaily"}) is None
    assert http["post"].call_count == 0  # gate-only: no FINRA request issued


def test_handler_rejects_bare_unknown_ambiguous_datasets_before_data(http: dict[str, MagicMock]) -> None:
    """Handler-level dataset canonicalization: unknown/ambiguous bare ids fail before any data POST."""
    unknown = execute_tool(
        "describe_finra_dataset", {"dataset_id": "notARealDataset"}, model="test", context=LOCAL_CONTEXT
    )
    assert unknown["error_type"] == "invalid_tool_arguments", unknown
    assert "notARealDataset" in str(unknown["error"]), unknown
    ambiguous = execute_tool("query_finra", {"dataset": "sharedName"}, model="test", context=LOCAL_CONTEXT)
    assert ambiguous["error_type"] == "invalid_tool_arguments", ambiguous
    assert "Ambiguous" in str(ambiguous["error"]), ambiguous
    assert http["post"].call_count == 1  # catalog token only; no data POST


def test_pressure_profile_normalizes_ticker_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_short_pressure_profile routes its ticker through _finra_ticker (mixed-case remap)."""
    seen: dict[str, object] = {}

    def _fake_context(ticker: str) -> dict[str, object]:
        seen["ticker"] = ticker
        return {"ticker": ticker}

    monkeypatch.setattr(tools_module.sec, "get_short_pressure_context", _fake_context)

    def _resolve(_name: str) -> str | None:
        return "AAPL"

    monkeypatch.setattr(tools_module, "_resolve_company_to_ticker", _resolve)
    out = execute_tool("get_short_pressure_profile", {"ticker": "Apple"}, model="test", context=LOCAL_CONTEXT)
    assert out == {"ticker": "AAPL"}
    assert seen["ticker"] == "AAPL"


def test_finra_org_word_ticker_rejected_before_http(http: dict[str, MagicMock]) -> None:
    """ticker='FINRA' is an org word, not a security: guided invalid_tool_arguments, zero FINRA HTTP."""
    result = execute_tool("get_short_interest", {"ticker": "FINRA"}, model="test", context=LOCAL_CONTEXT)
    assert result["error_type"] == "invalid_tool_arguments", ("get_short_interest", result)
    assert "Provide a ticker" in str(result["error"]), ("get_short_interest", result)
    result = execute_tool("get_reg_sho_volume", {"ticker": "FINRA"}, model="test", context=LOCAL_CONTEXT)
    assert result["error_type"] == "invalid_tool_arguments", ("get_reg_sho_volume", result)
    assert "company_name" in str(result["error"]), ("get_reg_sho_volume", result)
    assert tools_module._finra_ticker({"ticker": "FINRA"}) is None
    assert tools_module._finra_ticker({"ticker": "finra"}) is None
    assert tools_module._finra_ticker({"ticker": "AAPL"}) == "AAPL"
    assert http["post"].call_count == 0
    assert http["get"].call_count == 0


def test_ticker_or_company_name_org_word_keeps_guided_message() -> None:
    """Org-word tickers fall through to the guided entity/company_name message."""
    ticker, err = tools_module._ticker_or_company_name({"ticker": "FINRA"}, "get_beneficial_ownership")
    assert ticker is None
    assert err is not None
    assert err["error_type"] == "invalid_tool_arguments"
    assert "company_name" in str(err["error"])
    ok, ok_err = tools_module._ticker_or_company_name({"ticker": "AAPL"}, "get_beneficial_ownership")
    assert ok == "AAPL"
    assert ok_err is None


def test_reg_sho_volume_ratio_math_and_elevated(http: dict[str, MagicMock]) -> None:
    """SHO enrich: combined short/total*100, elevated flags, existing keys kept."""
    records = [
        {
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "tradeReportDate": "2026-09-21",
            "shortParQuantity": 60,
            "shortExemptParQuantity": 20,
            "totalParQuantity": 200,
        },
        {
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "tradeReportDate": "2026-09-22",
            "shortParQuantity": 80,
            "shortExemptParQuantity": 20,
            "totalParQuantity": 200,
        },
    ]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = execute_tool(
        "get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "2026-09-22"}, model="test", context=LOCAL_CONTEXT
    )
    assert "error" not in result, result
    assert result["short_volume"] == 100
    assert result["total_volume"] == 200
    assert result["short_volume_ratio_pct"] == 50.0
    assert result["short_volume_ratio_elevated"] is True
    assert "40-50%" in str(result["short_volume_ratio_note"])
    assert "briefing" in result and "metrics" in result and "coverage" in result


def test_reg_sho_volume_ratio_not_elevated(http: dict[str, MagicMock]) -> None:
    records = [
        {
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "tradeReportDate": "2026-09-22",
            "shortParQuantity": 10,
            "shortExemptParQuantity": 5,
            "totalParQuantity": 100,
        }
    ]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = execute_tool(
        "get_reg_sho_volume", {"ticker": "AAPL", "tradeDate": "2026-09-22"}, model="test", context=LOCAL_CONTEXT
    )
    assert result["short_volume_ratio_pct"] == 15.0
    assert result["short_volume_ratio_elevated"] is False


def test_reg_sho_no_date_queries_monday_nyc_to_today(http: dict[str, MagicMock]) -> None:
    """No-tradeDate path: Monday-NYC start through today, single tradeDate stays single-day."""
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    records = [
        {
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "tradeReportDate": "2026-09-22",
            "shortParQuantity": 10,
            "shortExemptParQuantity": 5,
            "totalParQuantity": 100,
        }
    ]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = execute_tool("get_reg_sho_volume", {"ticker": "AAPL"}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    body = _data_body(http["post"])
    nyc_today = datetime.now(UTC).astimezone(ZoneInfo("America/New_York")).date()
    monday = (nyc_today - timedelta(days=nyc_today.weekday())).isoformat()
    today = nyc_today.isoformat()
    assert body["dateRangeFilters"] == [{"fieldName": "tradeReportDate", "startDate": monday, "endDate": today}]
    assert result["short_volume_ratio_pct"] == 15.0
