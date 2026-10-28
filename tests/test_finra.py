"""Unit tests for the FINRA Query API tools.

Deterministic and offline: all FINRA HTTP is mocked against sanitized,
source-controlled fixtures in tests/fixtures/finra/. Live verification is
opt-in via tests/test_finra_smoke.py (pytest -m finra_smoke).
"""

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Self, override
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

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
    # The main model's tool message never contains the raw records list; only the newest row rides
    # as `latest_record`, the citable source of verified datapoints.
    serialized = json.dumps(result)
    assert '"records":' not in serialized
    without_latest = json.dumps({key: value for key, value in result.items() if key != "latest_record"})
    assert '"symbolCode": "AAPL"' not in without_latest
    assert '"issueName": "Apple Inc."' not in without_latest
    assert result["latest_record"] == records[0]

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


FINRA_TODAY = date(2026, 10, 14)


def _freeze_finra_today(monkeypatch: pytest.MonkeyPatch, frozen: date) -> None:
    """Freeze the local trading date that finra_client reads through date.today()."""

    class _FrozenDate(date):
        @classmethod
        @override
        def today(cls) -> Self:
            return cls(frozen.year, frozen.month, frozen.day)

    monkeypatch.setattr(finra_client, "date", _FrozenDate)


def _latest_short_interest(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch, settlement_date: str
) -> dict[str, object]:
    """Run the no-date get_short_interest tool over one record dated settlement_date."""
    http["post"].side_effect = [
        _token_response(),
        _response(
            [
                {
                    "symbolCode": "AAPL",
                    "settlementDate": settlement_date,
                    "currentShortPositionQuantity": 100,
                }
            ]
        ),
    ]

    def _fake_partitions(*args: object) -> tuple[list[dict[str, str]], dict[str, object], int, bool]:
        return ([{"settlementDate": settlement_date}], {}, 1, False)

    monkeypatch.setattr(
        finra_client,
        "_datapoints_via_partitions",
        _fake_partitions,
    )

    return execute_tool("get_short_interest", {"ticker": "AAPL"}, model="test", context=LOCAL_CONTEXT)


def test_latest_short_interest_rejects_stale_production_data(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_finra_today(monkeypatch, FINRA_TODAY)
    stale_date = (FINRA_TODAY - timedelta(days=finra_client.STALE_AFTER_DAYS + 1)).isoformat()

    result = _latest_short_interest(http, monkeypatch, stale_date)

    assert "error" in result
    assert "Current FINRA short interest is unavailable" in _error(result)
    assert stale_date in _error(result)


@pytest.mark.parametrize(
    ("today", "settlement_date", "stale"),
    [
        pytest.param(date(2026, 10, 14), "2026-07-16", False, id="90-days-current"),
        pytest.param(date(2026, 10, 14), "2026-07-15", True, id="91-days-stale"),
        # Local date 10-13 while UTC is already 10-14: freshness counts local
        # days (90 here), so the record stays current.
        pytest.param(date(2026, 10, 13), "2026-07-15", False, id="local-date-before-utc-rollover-current"),
        pytest.param(date(2026, 1, 1), "2025-10-03", False, id="year-boundary-90-days-current"),
        pytest.param(date(2026, 1, 1), "2025-10-02", True, id="year-boundary-91-days-stale"),
    ],
)
def test_latest_short_interest_stale_threshold(
    http: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    today: date,
    settlement_date: str,
    stale: bool,
) -> None:
    """Records over STALE_AFTER_DAYS local days old fail closed; 90 days stays current."""
    _freeze_finra_today(monkeypatch, today)

    result = _latest_short_interest(http, monkeypatch, settlement_date)

    if stale:
        assert "Current FINRA short interest is unavailable" in _error(result)
        assert settlement_date in _error(result)
    else:
        assert "error" not in result, result
        assert result["data_freshness"] == "current"
        assert result["as_of_date"] == settlement_date
        assert not any("STALE" in str(w) for w in _as_seq(result["warnings"]))


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


def _reg_sho_no_date_body(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch, frozen: datetime
) -> dict[str, object]:
    """Run the no-tradeDate get_reg_sho_volume tool with app.tools' clock frozen at frozen."""

    class _FrozenClock(datetime):
        @classmethod
        @override
        def now(cls, tz: tzinfo | None = None) -> Self:
            moment = cls.fromtimestamp(frozen.timestamp(), tz or UTC)
            return moment if tz else moment.replace(tzinfo=None)

    monkeypatch.setattr(tools_module, "datetime", _FrozenClock)
    records = [
        {
            "securitiesInformationProcessorSymbolIdentifier": "AAPL",
            "tradeReportDate": "2026-09-21",
            "shortParQuantity": 10,
            "shortExemptParQuantity": 5,
            "totalParQuantity": 100,
        }
    ]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = execute_tool("get_reg_sho_volume", {"ticker": "AAPL"}, model="test", context=LOCAL_CONTEXT)
    assert "error" not in result, result
    assert result["short_volume_ratio_pct"] == 15.0
    body = _data_body(http["post"])
    assert isinstance(body, dict)
    return body


_NYC = ZoneInfo("America/New_York")


def test_reg_sho_no_date_queries_monday_nyc_to_today(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-tradeDate path on a Wednesday: dateRangeFilters from Monday-NYC through today-NYC."""
    body = _reg_sho_no_date_body(http, monkeypatch, datetime(2026, 9, 23, 12, 0, tzinfo=_NYC))
    assert body["dateRangeFilters"] == [
        {"fieldName": "tradeReportDate", "startDate": "2026-09-21", "endDate": "2026-09-23"}
    ]


def test_reg_sho_no_date_on_monday_queries_single_day(
    http: dict[str, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-tradeDate path on a Monday: start equals today, so one EQUAL row and no range."""
    body = _reg_sho_no_date_body(http, monkeypatch, datetime(2026, 9, 21, 12, 0, tzinfo=_NYC))
    assert "dateRangeFilters" not in body
    compare = body["compareFilters"]
    assert isinstance(compare, list)
    assert {"compareType": "EQUAL", "fieldName": "tradeReportDate", "fieldValue": "2026-09-21"} in compare


# --- Official public files (citation pages) -----------------------------------

# Trimmed copies of the 2026-10-15 cdn.finra.org files (same header, line shapes).
_FNQC_FILE = (
    b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    b"20261015|NVDA|274457.043010|1022|934921.209763|B\n"
    b"20261015|NVDL|1000|0|2000|B\n"
    b"2\n"
)
_CNMS_FILE = (
    b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    b"20261015|NVDA|18013290.539633|100349|51176540.043723|B,Q,N\n"
    b"1\n"
)
_SHRT_FILE = (
    b'"accountingYearMonthNumber"|"symbolCode"|"issueName"|"currentShortPositionQuantity"|"stockSplitFlag"'
    b'|"daysToCoverQuantity"|"changePercent"|"settlementDate"\n'
    b'"20260930"|"NV"|"Envision Inc"|"5"||"1.00"|"0.00"|"2026-09-30"\n'
    b'"20260930"|"NVDA"|"NVIDIA Corporation Common Stoc"|"311389688"||"2.82"|"5.83"|"2026-09-30"\n'
)
_NVDA_FNQC_ROW = {
    "reportingFacilityCode": "NCTRF",
    "totalParQuantity": 934921.209763,
    "shortParQuantity": 274457.04301,
    "marketCode": "B",
    "tradeReportDate": "2026-10-15",
    "securitiesInformationProcessorSymbolIdentifier": "NVDA",
    "shortExemptParQuantity": 1022,
}


@pytest.fixture
def official_get(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    files = {
        "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt": _FNQC_FILE,
        "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20261015.txt": _CNMS_FILE,
        "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20260930.csv": _SHRT_FILE,
    }

    def fake_get(url: str, timeout: int) -> MagicMock:
        assert timeout == 60
        resp = MagicMock()
        resp.content = files[url]
        return resp

    get_mock = MagicMock(side_effect=fake_get)
    monkeypatch.setattr("app.finra_client.requests.get", get_mock)
    return get_mock


def test_official_reg_sho_row_cites_its_facility_file_line(official_get: MagicMock, tmp_path: Path) -> None:
    """marketCode B (NCTRF) maps to FNQC; file 274457.043010 equals API 274457.04301."""
    page = finra_client.resolve_finra_official("otcMarket/regShoDaily", _NVDA_FNQC_ROW, {}, root=tmp_path)
    assert page is not None
    assert page["url"] == "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt"
    assert page["quote"] == "20261015|NVDA|274457.043010|1022|934921.209763|B"
    assert page["payload"] == _FNQC_FILE
    assert page["media"] == "text"
    assert page["anchor"] == ""
    assert page["sha256"] == finra_client.raw_archive.content_hash(_FNQC_FILE)


def test_official_reg_sho_facility_code_alone_selects_file(official_get: MagicMock, tmp_path: Path) -> None:
    row = {k: v for k, v in _NVDA_FNQC_ROW.items() if k != "marketCode"}
    page = finra_client.resolve_finra_official("otcMarket/regShoDaily", row, {}, root=tmp_path)
    assert page is not None
    assert page["url"] == "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt"


def test_official_reg_sho_envelope_cites_consolidated_line(official_get: MagicMock, tmp_path: Path) -> None:
    """Venue-less analysis envelopes fail closed: no CNMS line proves a venue claim."""
    query = {"ticker": "NVDA", "start_date": None, "end_date": None, "limit": 100, "offset": 0, "filters": None}
    payload = {"dataset_id": "otcMarket/regShoDaily", "query": query, "as_of_date": "2026-10-15"}
    assert finra_client.resolve_finra_official("otcMarket/regShoDailyMock", query, payload, root=tmp_path) is None
    official_get.assert_not_called()


def test_official_reg_sho_disagreeing_or_missing_line_raises(official_get: MagicMock, tmp_path: Path) -> None:
    wrong = {**_NVDA_FNQC_ROW, "totalParQuantity": 934921.2}
    with pytest.raises(ValueError, match="TotalVolume"):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", wrong, {}, root=tmp_path)
    absent = {**_NVDA_FNQC_ROW, "securitiesInformationProcessorSymbolIdentifier": "AMD"}
    with pytest.raises(ValueError, match="no line for AMD"):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", absent, {}, root=tmp_path)


def test_official_short_interest_cites_quoted_symbol_line_once(official_get: MagicMock, tmp_path: Path) -> None:
    """Exact "NVDA" cell (not the "NV" prefix); the second read comes from the archive."""
    row = {
        "symbolCode": "NVDA",
        "settlementDate": "2026-09-30",
        "accountingYearMonthNumber": 20260930,
        "currentShortPositionQuantity": 311389688,
        "stockSplitFlag": None,
        "changePercent": 5.83,
    }
    first = finra_client.resolve_finra_official("otcMarket/consolidatedShortInterest", row, {}, root=tmp_path)
    second = finra_client.resolve_finra_official("otcMarket/consolidatedShortInterest", row, {}, root=tmp_path)
    assert first is not None
    assert first == second
    assert first["url"] == "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20260930.csv"
    assert (
        first["quote"] == '"20260930"|"NVDA"|"NVIDIA Corporation Common Stoc"|"311389688"||"2.82"|"5.83"|"2026-09-30"'
    )
    assert first["media"] == "text"
    assert official_get.call_count == 1
    with pytest.raises(ValueError, match="currentShortPositionQuantity"):
        finra_client.resolve_finra_official(
            "otcMarket/consolidatedShortInterest", {**row, "currentShortPositionQuantity": 1}, {}, root=tmp_path
        )


def test_official_empty_row_cites_first_payload_record(official_get: MagicMock, tmp_path: Path) -> None:
    """An empty row cites nothing: the first payload record is never substituted."""
    payload = {"records": [{"symbolCode": "NVDA", "settlementDate": "2026-09-30"}]}
    assert (
        finra_client.resolve_finra_official("otcMarket/consolidatedShortInterest", {}, payload, root=tmp_path) is None
    )
    official_get.assert_not_called()


def test_official_file_absent_for_other_datasets_or_unknown_subject(official_get: MagicMock, tmp_path: Path) -> None:
    """thresholdList/weeklySummary have no static FINRA file; no symbol/day means no page."""
    row = {"issueSymbolIdentifier": "NVDA", "tradeDate": "2026-10-15"}
    assert finra_client.resolve_finra_official("otcMarket/thresholdList", row, {}, root=tmp_path) is None
    assert finra_client.resolve_finra_official("otcMarket/weeklySummary", row, {}, root=tmp_path) is None
    assert finra_client.resolve_finra_official("", row, {}, root=tmp_path) is None
    assert finra_client.resolve_finra_official("otcMarket/regShoDaily", {"ticker": "NVDA"}, {}, root=tmp_path) is None
    official_get.assert_not_called()


# ---------------------------------------------------------------------------
# Change 4 regression: published-file matching, conflicts, archive integrity
# ---------------------------------------------------------------------------
def _reg4_facility_file(market: str) -> bytes:
    return (
        b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        + f"20261015|NVDA|274457.043010|1022|934921.209763|{market}\n".encode()
    )


_REG4_CNMS_LINE = "20261015|NVDA|18013290.539633|100349|51176540.043723|B,Q,N"


@pytest.fixture
def reg4_get(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fake cdn.finra.org: every daily facility file plus the biweekly short-interest file."""
    files = {
        "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt": _reg4_facility_file("B"),
        "https://cdn.finra.org/equity/regsho/daily/FNSQshvol20261015.txt": _reg4_facility_file("Q"),
        "https://cdn.finra.org/equity/regsho/daily/FNYXshvol20261015.txt": _reg4_facility_file("N"),
        "https://cdn.finra.org/equity/regsho/daily/FORFshvol20261015.txt": _reg4_facility_file("O"),
        "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20261015.txt": (
            b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n" + _REG4_CNMS_LINE.encode() + b"\n"
        ),
        "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20260930.csv": _SHRT_FILE,
    }

    def fake_get(url: str, timeout: int) -> MagicMock:
        assert timeout == 60
        resp = MagicMock()
        resp.content = files[url]
        return resp

    get_mock = MagicMock(side_effect=fake_get)
    monkeypatch.setattr("app.finra_client.requests.get", get_mock)
    return get_mock


def _reg4_row(market: str) -> dict[str, object]:
    return {
        **_NVDA_FNQC_ROW,
        "marketCode": market,
        "reportingFacilityCode": {"B": "NCTRF", "Q": "NQTRF", "N": "NYTRF", "O": "ORF"}[market],
    }


@pytest.mark.parametrize("market", ["Q", "N", "O"])
def test_reg4_venue_specific_line_beats_consolidated(reg4_get: MagicMock, tmp_path: Path, market: str) -> None:
    """A venue row cites its own facility file, never the consolidated/CNMS line."""
    row = _reg4_row(market)
    page = finra_client.resolve_finra_official("otcMarket/regShoDaily", row, {}, root=tmp_path)
    assert page is not None
    expect = {"Q": "FNSQ", "N": "FNYX", "O": "FORF"}[market]
    assert page["url"] == f"https://cdn.finra.org/equity/regsho/daily/{expect}shvol20261015.txt"
    assert isinstance(page["quote"], str) and page["quote"].endswith(f"|{market}")
    assert _REG4_CNMS_LINE != page["quote"]


def test_reg4_consolidated_line_insufficient_for_venue_claim(reg4_get: MagicMock, tmp_path: Path) -> None:
    """A venue row never validates against the CNMS (B,Q,N) line: first row is not a match."""
    payload = {"records": [{"securitiesInformationProcessorSymbolIdentifier": "NVDA", "tradeReportDate": "2026-10-15"}]}
    row = _reg4_row("B")
    page = finra_client.resolve_finra_official("otcMarket/regShoDaily", row, payload, root=tmp_path)
    assert page is not None and isinstance(page["quote"], str) and page["quote"].endswith("|B")
    with pytest.raises(ValueError, match="disagrees"):
        finra_client.resolve_finra_official(
            "otcMarket/regShoDaily", {**row, "shortParQuantity": 18013290.539633}, payload, root=tmp_path
        )


def test_reg4_duplicate_symbols_cite_exact_symbol_cell(reg4_get: MagicMock, tmp_path: Path) -> None:
    """NVDA never matches the NVDL prefix or the NV prefix row in the short-interest file."""
    row = {
        "symbolCode": "NVDA",
        "settlementDate": "2026-09-30",
        "accountingYearMonthNumber": 20260930,
        "currentShortPositionQuantity": 311389688,
        "stockSplitFlag": None,
        "changePercent": 5.83,
    }
    page = finra_client.resolve_finra_official("otcMarket/consolidatedShortInterest", row, {}, root=tmp_path)
    assert page is not None and isinstance(page["quote"], str)
    assert '"NVDA"' in page["quote"] and '"NVDL"' not in page["quote"]
    assert '"NV"' not in page["quote"].split("|")[1]
    with pytest.raises(ValueError, match="currentShortPositionQuantity"):
        finra_client.resolve_finra_official(
            "otcMarket/consolidatedShortInterest", {**row, "symbolCode": "NV"}, {}, root=tmp_path
        )


def test_reg4_missing_file_raises_not_none(
    reg4_get: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fetch failure for a dataset with a known file raises; the absence is explicit."""
    import requests as _requests

    def boom(url: str, timeout: int) -> MagicMock:
        raise _requests.ConnectionError("cdn.finra.org unreachable")

    monkeypatch.setattr("app.finra_client.requests.get", boom)
    with pytest.raises(_requests.ConnectionError):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", _NVDA_FNQC_ROW, {}, root=tmp_path)


def test_reg4_file_vs_api_conflict_preserved(reg4_get: MagicMock, tmp_path: Path) -> None:
    """A file line that disagrees with the API row raises; neither side is silently preferred."""
    wrong = {**_NVDA_FNQC_ROW, "shortParQuantity": 1.0}
    with pytest.raises(ValueError, match="ShortVolume"):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", wrong, {}, root=tmp_path)
    absent = {**_NVDA_FNQC_ROW, "securitiesInformationProcessorSymbolIdentifier": "ZZZ"}
    with pytest.raises(ValueError, match="no line for ZZZ"):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", absent, {}, root=tmp_path)


_FNQC_HEADER = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
_FNQC_NVDA = "20261015|NVDA|274457.043010|1022|934921.209763|B"


def _resolve_fnqc(
    monkeypatch: pytest.MonkeyPatch, root: Path, lines: list[str], row: Mapping[str, object] = _NVDA_FNQC_ROW
) -> dict[str, object] | None:
    """Resolve ``row`` against an FNQC file made of ``lines`` under a fresh archive root."""
    body = "\n".join([_FNQC_HEADER, *lines, str(len(lines))]).encode() + b"\n"
    monkeypatch.setattr("app.finra_client.requests.get", MagicMock(return_value=MagicMock(content=body)))
    return finra_client.resolve_finra_official("otcMarket/regShoDaily", row, {}, root=root)


def test_official_line_skips_earlier_disagreeing_duplicate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Earlier NVDA lines with another value, day, or venue never hide the later line that agrees."""
    earlier = [
        "20261015|NVDA|1|1022|934921.209763|B",
        "20261014|NVDA|274457.043010|1022|934921.209763|B",
        "20261015|NVDA|274457.043010|1022|934921.209763|Q",
    ]
    page = _resolve_fnqc(monkeypatch, tmp_path, [*earlier, _FNQC_NVDA])
    assert page is not None and page["quote"] == _FNQC_NVDA


def test_official_line_rejects_wrong_day_and_venue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The Date and Market cells are identity fields: a lone line for another day or venue is a conflict."""
    with pytest.raises(finra_client.OfficialFileConflict, match="Date='20261014' disagrees"):
        _resolve_fnqc(monkeypatch, tmp_path / "day", ["20261014|NVDA|274457.043010|1022|934921.209763|B"])
    # A facility-only row (no marketCode) still needs a Market cell that maps to its facility file.
    facility_only = {k: v for k, v in _NVDA_FNQC_ROW.items() if k != "marketCode"}
    with pytest.raises(finra_client.OfficialFileConflict, match="Market='Q' disagrees"):
        _resolve_fnqc(
            monkeypatch, tmp_path / "venue", ["20261015|NVDA|274457.043010|1022|934921.209763|Q"], facility_only
        )


def test_official_line_rejects_ambiguous_agreeing_duplicates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two lines agree with every cited value but differ in an uncited cell: neither is chosen by position."""
    row = {k: v for k, v in _NVDA_FNQC_ROW.items() if k != "shortExemptParQuantity"}
    other = "20261015|NVDA|274457.043010|9|934921.209763|B"
    for name, lines in (("ab", [_FNQC_NVDA, other]), ("ba", [other, _FNQC_NVDA])):
        with pytest.raises(finra_client.OfficialFileConflict, match="2 ambiguous lines for NVDA"):
            _resolve_fnqc(monkeypatch, tmp_path / name, lines, row)


def test_official_line_equivalent_duplicates_cite_one_line_in_any_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal-valued duplicates (274457.043010 == 274457.04301) pass and cite the same line in either order."""
    twin = "20261015|NVDA|274457.04301|1022.0|934921.209763|B"
    first = _resolve_fnqc(monkeypatch, tmp_path / "ab", [_FNQC_NVDA, twin])
    second = _resolve_fnqc(monkeypatch, tmp_path / "ba", [twin, _FNQC_NVDA])
    assert first is not None and second is not None
    assert first["quote"] == second["quote"] == min(_FNQC_NVDA, twin)


def test_official_line_conflicts_are_typed_but_archive_corruption_is_not(reg4_get: MagicMock, tmp_path: Path) -> None:
    """Missing and disagreeing lines raise OfficialFileConflict. A bad archive stays a plain ValueError."""
    with pytest.raises(finra_client.OfficialFileConflict, match="no line for ZZZ"):
        finra_client.resolve_finra_official(
            "otcMarket/regShoDaily",
            {**_NVDA_FNQC_ROW, "securitiesInformationProcessorSymbolIdentifier": "ZZZ"},
            {},
            root=tmp_path,
        )
    page = finra_client.resolve_finra_official("otcMarket/regShoDaily", _NVDA_FNQC_ROW, {}, root=tmp_path)
    assert page is not None and isinstance(page["url"], str)
    record = finra_client.raw_archive.find("finra", "official_file", page["url"], root=tmp_path / "raw")
    assert record is not None
    record.payload_path.write_bytes(b"edited")
    with pytest.raises(ValueError, match="content hash") as info:
        finra_client.resolve_finra_official("otcMarket/regShoDaily", _NVDA_FNQC_ROW, {}, root=tmp_path)
    assert not isinstance(info.value, finra_client.OfficialFileConflict)


def test_reg4_archive_hash_mismatch_fails_closed(reg4_get: MagicMock, tmp_path: Path) -> None:
    """Edited archived bytes never verify: the archive read re-checks the content hash and never refetches."""
    from app.storage import raw_archive as _archive

    row = dict(_NVDA_FNQC_ROW)
    first = finra_client.resolve_finra_official("otcMarket/regShoDaily", row, {}, root=tmp_path)
    assert first is not None and isinstance(first["url"], str) and reg4_get.call_count == 1
    record = _archive.find("finra", "official_file", first["url"], root=tmp_path / "raw")
    assert record is not None
    record.payload_path.write_bytes(_reg4_facility_file("B").replace(b"274457.043010", b"274457.043011"))
    with pytest.raises(ValueError, match="does not match its content hash"):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", row, {}, root=tmp_path)
    assert reg4_get.call_count == 1


def test_official_file_rejects_redirect_off_finra_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A redirect hop off https://cdn.finra.org/ raises OSError and archives nothing."""
    from types import SimpleNamespace

    url = "https://cdn.finra.org/equity/regsho/daily/FNQCshvol20261015.txt"
    hop = SimpleNamespace(url="https://evil.example/FNQCshvol20261015.txt")
    final = SimpleNamespace(url=url, history=[hop], content=_FNQC_FILE, raise_for_status=lambda: None)

    def redirected(u: str, timeout: int) -> SimpleNamespace:
        assert (u, timeout) == (url, 60)
        return final

    monkeypatch.setattr("app.finra_client.requests.get", redirected)
    with pytest.raises(OSError, match="redirect left the official FINRA host"):
        finra_client.resolve_finra_official("otcMarket/regShoDaily", _NVDA_FNQC_ROW, {}, root=tmp_path)
    assert finra_client.raw_archive.find("finra", "official_file", url, root=tmp_path / "raw") is None


def test_reg4_api_only_dataset_has_no_public_page(http: dict[str, MagicMock], tmp_path: Path) -> None:
    """thresholdList/weeklySummary resolve None: no static file, so no fabricated citation."""
    http["post"].side_effect = [_token_response()]
    assert (
        finra_client.resolve_finra_official(
            "otcMarket/thresholdList", {"issueSymbolIdentifier": "NVDA"}, {}, root=tmp_path
        )
        is None
    )
    assert (
        finra_client.resolve_finra_official(
            "otcMarket/weeklySummary", {"issueSymbolIdentifier": "NVDA"}, {}, root=tmp_path
        )
        is None
    )
    assert (
        finra_client.resolve_finra_official(
            "fixedIncomeMarket/treasuryDailyAggregates", {"tradeDate": "2026-10-15"}, {}, root=tmp_path
        )
        is None
    )


def test_reg4_weekly_threshold_records_cite_payload_json(
    http: dict[str, MagicMock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API-only datasets admit observation proofs over archived payload bytes, not a public page."""
    from app.research.evidence import finra_record_ref, observation_proof, verify_observation
    from app.storage import raw_archive as _archive

    http["post"].side_effect = [_token_response()]
    row = {"issueSymbolIdentifier": "NVDA", "weekStartDate": "2026-10-13", "totalShareQuantity": 12345}
    payload_bytes = json.dumps({"records": [row]}, sort_keys=True).encode()
    digest = _archive.archive(
        "finra", "tool_result", "tr-weekly", payload_bytes, url="api.finra.org", root=tmp_path / "raw"
    ).sha256
    prov = finra_record_ref(
        tool_name="query_finra",
        record_identity=json.dumps(row, sort_keys=True),
        dataset="otcMarket/weeklySummary",
        tool_result_id="tr-weekly",
    )
    proof = observation_proof(
        provenance=prov,
        source_bytes=payload_bytes,
        value="12345",
        metric="totalShareQuantity",
        entity="NVDA",
        period="2026-10-13",
    )
    assert proof["sourceId"] == f"sha256:{digest}"
    row_proof = {
        "kind": "observation",
        "sourceId": proof["sourceId"],
        "locator": proof["locator"],
        "valueHash": proof["valueHash"],
    }
    from datetime import UTC as _UTC
    from datetime import datetime

    from app.research.evidence import Evidence, evidence_content_hash

    evidence = Evidence(
        evidence_id="e1",
        session_id="s",
        wave_id=1,
        source_type="finra",
        source_name="FINRA",
        source_uri="",
        source_record_id="x",
        subject="NVDA",
        claim_text="c",
        content="c",
        content_hash=evidence_content_hash("c"),
        known_at=datetime.now(_UTC),
        retrieved_at=datetime.now(_UTC),
        job_id="j",
        agent_id="a",
        claim_kind="observed_fact",
        provenance=dict(prov),
        value="12345",
        entity="NVDA",
        period="2026-10-13",
        proof=row_proof,
    )
    assert verify_observation(evidence, value="12345", source_bytes=payload_bytes).status == "verified"
    assert verify_observation(evidence, value="12346", source_bytes=payload_bytes).status == "conflicted"
    assert verify_observation(evidence, value="12345", source_bytes=None).status == "unverified"


def test_reg4_treasury_aggregate_identity_needs_no_symbol(http: dict[str, MagicMock], tmp_path: Path) -> None:
    """Market aggregates carry no symbol field; the date alone is the valid identity."""
    http["post"].side_effect = [_token_response()]
    spec = finra_client.describe_dataset("fixedIncomeMarket/treasuryDailyAggregates")
    assert spec["supports_ticker"] is False and spec["supports_date"] is True
    assert (
        finra_client.resolve_finra_official(
            "fixedIncomeMarket/treasuryDailyAggregates", {"tradeDate": "2026-10-15"}, {}, root=tmp_path
        )
        is None
    )


def test_reg4_no_symbol_no_date_is_invalid_identity(tmp_path: Path) -> None:
    """An empty row/payload cites nothing: no symbol and no day means no page."""
    assert finra_client.resolve_finra_official("otcMarket/regShoDaily", {}, {}, root=tmp_path) is None
    assert finra_client.resolve_finra_official("otcMarket/consolidatedShortInterest", {}, {}, root=tmp_path) is None


def test_reg4_incomplete_pagination_marks_may_have_more(http: dict[str, MagicMock]) -> None:
    """A partial page against a larger total keeps may_have_more True: incomplete coverage is explicit."""
    records = [{"issueSymbolIdentifier": f"S{i}"} for i in range(2)]
    http["post"].side_effect = [
        _token_response(),
        _response(records, headers={"Record-Total": "5", "Record-Offset": "0", "Record-Limit": "2"}),
    ]
    result = execute_tool(
        "query_finra",
        {"dataset": "otcMarket/weeklySummary", "limit": 2},
        model="test",
        context=LOCAL_CONTEXT,
    )
    assert "error" not in result, result
    assert result["pagination_source"] == "finra_header"
    assert result["may_have_more"] is True
    assert result["returned_count"] == 2 and result["total_records"] == 5


def test_finra_prov_query_carries_replayable_observation(
    http: dict[str, MagicMock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """query_dataset pins archive-bytes-first provenance: dataset/group/identity/query/filters/sort/pagination/environment/retrieved/PIT/response/cited-record hashes."""
    import app.finra_client as _client
    from app.storage import raw_archive as _archive

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    records = [{"symbolCode": "NVDA", "settlementDate": "2026-09-30", "currentShortPositionQuantity": 10}]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = _client.query_dataset("otcMarket/consolidatedShortInterest", ticker="NVDA")
    prov = result.get("provenance_observation")
    assert isinstance(prov, dict), result
    assert prov["dataset"] == "otcMarket/consolidatedShortInterest"
    assert prov["group"] == "otcMarket"
    assert "symbolCode" in prov["schema"]
    assert isinstance(prov["query"], dict) and prov["query"].get("compareFilters")
    assert isinstance(prov["filters"], list) and prov["filters"]
    assert isinstance(prov["pagination"], dict)
    assert prov["environment"] == "production"
    assert isinstance(prov["retrieved_at"], str) and prov["retrieved_at"]
    assert prov["point_in_time"]["as_of_date"] == "2026-09-30"
    body = json.dumps({"records": records}, sort_keys=True).encode()
    assert prov["response_hash"] == _archive.content_hash(body)
    assert prov["record_hashes"] == [_archive.content_hash(json.dumps(records[0], sort_keys=True).encode())]
    # Archive bytes replay: the pinned response hash matches stored bytes.
    stored = _archive.find("finra", "tool_result", "otcMarket/consolidatedShortInterest", root=tmp_path / "raw")
    assert stored is not None and stored.payload_path.read_bytes() == body


def test_finra_prov_datapoints_carries_replayable_observation(
    http: dict[str, MagicMock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_finra_datapoints pins the same observation shape for exact-record requests."""
    import app.finra_client as _client

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FINRA_USE_MOCK", "")
    records = [{"symbolCode": "NVDA", "settlementDate": "2026-09-30"}]
    http["post"].side_effect = [_token_response(), _response(records)]
    result = _client.get_finra_datapoints(
        "otcMarket/consolidatedShortInterest",
        fields=["symbolCode", "settlementDate"],
        ticker="NVDA",
    )
    prov = result.get("provenance_observation")
    assert isinstance(prov, dict), result
    assert prov["dataset"] == "otcMarket/consolidatedShortInterest"
    assert isinstance(prov["response_hash"], str) and isinstance(prov["record_hashes"], list)


def test_finra_prov_unavailable_or_disagreeing_file_distinguished(http: dict[str, MagicMock], tmp_path: Path) -> None:
    """API-only datasets admit observation proofs; published-file rows without a file raise, never silently use row one."""
    import app.finra_client as _client

    http["post"].side_effect = [_token_response()]
    assert (
        _client.resolve_finra_official(
            "otcMarket/thresholdList", {"issueSymbolIdentifier": "NVDA", "tradeDate": "2026-10-15"}, {}, root=tmp_path
        )
        is None
    )
    assert (
        _client.resolve_finra_official(
            "fixedIncomeMarket/treasuryDailyAggregates", {"tradeDate": "2026-10-15"}, {}, root=tmp_path
        )
        is None
    )


def test_finra_prov_gateway_rows_carry_no_unverified_response_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway rows carry no response_hash: the tool-result replay path is the single verified source."""
    from app import data_sources as _ds

    def fake_datapoints(dataset: str, **kwargs: object) -> dict[str, object]:
        assert dataset == "otcMarket/consolidatedShortInterest"
        return {
            "records": [{"symbolCode": "NVDA", "settlementDate": "2026-09-30"}],
            "provenance_observation": {"response_hash": "abc123"},
        }

    monkeypatch.setattr(_ds._finra, "get_finra_datapoints", fake_datapoints)
    gateway = _ds.SourceGateway()
    rows = gateway.short_interest("NVDA")
    assert rows and all("response_hash" not in row for row in rows)
