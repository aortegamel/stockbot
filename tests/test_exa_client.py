"""Unit tests for the optional Exa web-search client.

All tests are offline: exa_client._ensure_session is monkeypatched with a
fake session (mirroring tests/test_analyst_client.py), and Exa is enabled
per-test via env vars.
"""

import json
import sqlite3
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
import requests

from app import exa_client, tools
from app.policy import Capability, RequestContext
from app.storage import market

EXA_FIXTURES = Path(__file__).parent / "fixtures" / "exa"

ALLOWED_PAYLOAD_KEYS = {
    "query",
    "numResults",
    "contents",
    "category",
    "includeDomains",
    "excludeDomains",
    "startPublishedDate",
    "endPublishedDate",
    "type",
}


def _fixture(name: str):
    return json.loads((EXA_FIXTURES / name).read_text())


def _enable_exa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_ENABLED", "true")
    monkeypatch.setenv("EXA_API_KEY", "test-exa-key-123")


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_seq(value: object):
    assert isinstance(value, (list, tuple))
    return value


def _err(result: dict[str, object]) -> str:
    err = result.get("error")
    assert isinstance(err, str)
    return err


def _response(payload: object = None, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload) if payload is not None else ""
    return resp


class FakeSession:
    """Session whose post() returns responses in order, last one repeated."""

    responses: list[MagicMock]
    posts: list[tuple[str, dict[str, object]]]

    def __init__(self, *responses: MagicMock) -> None:
        self.responses = list(responses)
        self.posts = []

    def post(self, url: str, **kwargs: object) -> MagicMock:
        self.posts.append((url, dict(kwargs)))
        if len(self.responses) == 1:
            return self.responses[-1]
        return self.responses.pop(0)


def _patch_session(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> FakeSession:
    monkeypatch.setattr(exa_client, "_ensure_session", lambda: session)
    return session


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_exa(monkeypatch)


@pytest.fixture(autouse=True)
def isolated_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EXA_MONTHLY_LIMIT", "9")
    monkeypatch.setenv("EXA_COLLECTOR_MONTHLY_LIMIT", "3")


def test_disabled_when_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_ENABLED", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call when Exa is disabled")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("AMD news")
    assert result == {"error": "Exa search unavailable", "source": "exa"}


def test_disabled_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_ENABLED", "true")
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call when the API key is missing")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("AMD news")
    assert result == {"error": "Exa search unavailable", "source": "exa"}


def test_http_500(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, FakeSession(_response(status=500)))
    result = exa_client.search("AMD news")
    assert result["error"] == "Exa search failed: HTTP 500"
    assert result["source"] == "exa"


def test_timeout(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **kwargs: object) -> None:
        raise requests.Timeout("timed out")

    session = FakeSession()
    session.post = boom
    _patch_session(monkeypatch, session)
    result = exa_client.search("AMD news")
    assert result["error"] == "Exa search timed out"


def test_request_exception(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **kwargs: object) -> None:
        raise requests.ConnectionError("down")

    session = FakeSession()
    session.post = boom
    _patch_session(monkeypatch, session)
    result = exa_client.search("AMD news")
    assert result["error"] == "Exa search unavailable"


def test_malformed_json(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _response(payload={"results": []})
    resp.json.side_effect = ValueError("not json")
    session = _patch_session(monkeypatch, FakeSession(resp))
    result = exa_client.search("AMD news")
    assert result["error"] == "Exa search returned an invalid response"
    assert session.posts[0][1]["headers"] == {"x-api-key": "test-exa-key-123"}


def test_missing_results_key(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, FakeSession(_response(payload={"foo": 1})))
    result = exa_client.search("AMD news")
    assert result["error"] == "Exa search returned an invalid response"


def test_limit_clamp(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for limit, expected in ((99, 25), (0, 1), (None, 5)):
        session = _patch_session(monkeypatch, FakeSession(_response(_fixture("search.json"))))
        if limit is None:
            exa_client.search("AMD news")
        else:
            exa_client.search("AMD news", limit=limit)
        payload = session.posts[0][1]["json"]
        assert isinstance(payload, dict)
        assert payload["numResults"] == expected
        assert payload["query"] == "AMD news"
        assert payload["contents"] == {
            "highlights": True,
            "text": {"maxCharacters": exa_client.EXA_SEARCH_TEXT_MAX_CHARS, "includeHtmlTags": True},
        }


def test_normalization_from_fixture(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, FakeSession(_response(_fixture("search.json"))))
    result = exa_client.search("AMD news")
    assert result["result_type"] == "web_search"
    assert result["query"] == "AMD news"
    assert result["search_type"] == "auto"
    assert result["source"] == "exa"
    assert result["row_count"] == 2
    assert result["omitted_count"] == 1
    assert result["retrieved_at"]

    first, second = _as_seq(result["evidence"])
    # Full item with two highlights: title/url/domain preserved, first highlight.
    assert first["title"] == "AMD Unveils MI400 Accelerator With 4x Throughput"
    assert first["url"] == "https://ir.amd.com/news-releases/news-details/2026/amd-mi400"
    assert first["source_domain"] == "ir.amd.com"
    assert first["published_at"] == "2026-08-01T10:00:00.000Z"
    assert first["retrieved_at"] == result["retrieved_at"]
    assert first["highlight"].startswith("AMD unveiled its MI400 accelerator")
    assert first["category"] is None
    assert first["source_content"] == ""
    # Item without publishedDate: published_at None.
    assert second["published_at"] is None
    assert second["url"].startswith("https://www.reuters.com")
    # Item without url was skipped, never surfaced.
    assert all("no-url-item" not in str(item) for item in _as_seq(result["evidence"]))


def test_highlight_truncated_to_max_chars(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    long_highlight = "x" * 2000
    payload = {
        "results": [
            {
                "title": "t",
                "url": "https://example.com/x",
                "publishedDate": "2026-08-01T10:00:00.000Z",
                "highlights": [long_highlight],
                "text": "y" * (exa_client.EXA_SEARCH_TEXT_MAX_CHARS + 10),
            }
        ]
    }
    _patch_session(monkeypatch, FakeSession(_response(payload)))
    result = exa_client.search("AMD news")
    row = _as_seq(result["evidence"])[0]
    assert len(row["highlight"]) == exa_client.EXA_HIGHLIGHT_MAX_CHARS
    assert len(row["source_content"]) == exa_client.EXA_SEARCH_TEXT_MAX_CHARS


def test_company_category_rejects_dates_and_exclude_domains(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call for an unsupported company-category combo")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("AMD", category="company", start_published_date="2026-01-01")
    assert "does not support" in _err(result)
    assert result["source"] == "exa"
    result = exa_client.search("AMD", category="company", end_published_date="2026-08-01")
    assert "does not support" in _err(result)
    assert result["source"] == "exa"
    result = exa_client.search("AMD", category="company", exclude_domains=["reddit.com"])
    assert "does not support" in _err(result)
    assert result["source"] == "exa"


def test_company_category_allows_include_domains(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _patch_session(monkeypatch, FakeSession(_response(_fixture("search.json"))))
    result = exa_client.search(
        "AMD",
        category="company",
        include_domains=["amd.com"],
    )
    assert "error" not in result
    payload = session.posts[0][1]["json"]
    assert isinstance(payload, dict)
    assert payload["category"] == "company"
    assert payload["includeDomains"] == ["amd.com"]


def test_dates_convert_to_iso_ranges(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _patch_session(monkeypatch, FakeSession(_response(_fixture("search.json"))))
    exa_client.search(
        "AMD news",
        start_published_date="2026-07-01",
        end_published_date="2026-08-01",
    )
    payload = session.posts[0][1]["json"]
    assert isinstance(payload, dict)
    assert payload["startPublishedDate"] == "2026-07-01T00:00:00.000Z"
    assert payload["endPublishedDate"] == "2026-08-01T23:59:59.999Z"


def test_unsupported_category(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call for an unsupported category")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("AMD news", category="gossip")
    assert "Unsupported category 'gossip'" in _err(result)
    assert result["source"] == "exa"


def test_unsupported_search_type(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call for an unsupported search_type")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("AMD news", search_type="deep")
    assert "Unsupported search_type 'deep'" in _err(result)


def test_invalid_date(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call for an invalid date")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("AMD news", start_published_date="08/01/2026")
    assert "Invalid date '08/01/2026'" in _err(result)


def test_empty_query(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call for an empty query")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    result = exa_client.search("   ")
    assert result["error"] == "Exa search query must be a non-empty string"


def test_non_integer_limit(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        pytest.fail("no HTTP call for a non-integer limit")

    monkeypatch.setattr(exa_client, "_ensure_session", boom)
    # Non-integer limit arrives as LLM JSON (untyped); dispatch preserves it for runtime validation.
    context = RequestContext("test", frozenset({Capability.RESEARCH}))
    result = tools.execute_tool("search_web", {"query": "AMD news", "limit": "five"}, model="test", context=context)
    assert result["error"] == "limit must be an integer"


def test_payload_privacy(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _patch_session(monkeypatch, FakeSession(_response(_fixture("search.json"))))
    exa_client.search(
        "AMD competitive position 2026",
        category="news",
        include_domains=["amd.com"],
        start_published_date="2026-07-01",
    )
    payload = session.posts[0][1]["json"]
    assert isinstance(payload, dict)
    assert set(payload) <= ALLOWED_PAYLOAD_KEYS
    blob = json.dumps(payload).lower()
    for identifier in ("account", "portfolio", "123456789"):
        assert identifier not in blob


def test_budget_refuses_without_http(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_MONTHLY_LIMIT", "0.001")
    session = _patch_session(monkeypatch, FakeSession())
    assert "budget" in str(exa_client.search("AMD")["error"])
    assert not session.posts


def test_collector_budget_is_separate(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_COLLECTOR_MONTHLY_LIMIT", "0")
    session = _patch_session(monkeypatch, FakeSession(_response({"results": []})))
    assert "budget" in str(exa_client.search("AMD", caller="collector")["error"])
    assert "error" not in exa_client.search("AMD")
    assert len(session.posts) == 1


def test_budget_reconciles_and_contents_share_it(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    session = _patch_session(monkeypatch, FakeSession(_response({"results": [], "costDollars": {"total": 0.002}})))
    assert "error" not in exa_client.search("AMD", caller="collector")
    assert market.spent("exa", period) == pytest.approx(0.002)
    assert market.spent("exa:collector", period) == pytest.approx(0.002)
    assert "error" not in exa_client.contents(["https://example.com"], caller="collector")
    assert market.spent("exa", period) == pytest.approx(0.004)
    assert market.spent("exa:collector", period) == pytest.approx(0.004)
    assert session.posts[-1][0].endswith("/contents")
    assert session.posts[-1][1]["json"] == {
        "urls": ["https://example.com"],
        "text": {"maxCharacters": exa_client.EXA_SEARCH_TEXT_MAX_CHARS},
    }


def test_budget_refunds_failed_request(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session.post.side_effect = requests.ConnectionError("offline")
    monkeypatch.setattr(exa_client, "_ensure_session", lambda: session)
    assert "error" in exa_client.search("AMD", caller="collector")
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    assert market.spent("exa", period) == 0
    assert market.spent("exa:collector", period) == 0


@pytest.mark.parametrize("cost", [None, {}, {"total": None}, {"total": -1}, {"total": True}, {"total": float("nan")}])
def test_invalid_cost_keeps_reservation(enabled: None, monkeypatch: pytest.MonkeyPatch, cost: object) -> None:
    _patch_session(monkeypatch, FakeSession(_response({"results": [], "costDollars": cost})))
    exa_client.search("AMD", limit=1)
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    assert market.spent("exa", period) == pytest.approx(0.009)


@pytest.mark.parametrize("limit", ["-1", "nan", "inf", "-inf"])
@pytest.mark.parametrize("caller", ["agent", "collector"])
def test_invalid_caps_refuse_before_http(
    enabled: None, monkeypatch: pytest.MonkeyPatch, limit: str, caller: str
) -> None:
    monkeypatch.setenv("EXA_MONTHLY_LIMIT" if caller == "agent" else "EXA_COLLECTOR_MONTHLY_LIMIT", limit)
    session = _patch_session(monkeypatch, FakeSession())
    assert exa_client.contents(["https://example.com"], caller=caller) == {
        "error": "Exa monthly budget limits must be finite nonnegative dollars",
        "source": "exa",
    }
    assert not session.posts


@pytest.mark.parametrize("urls", [[], ["ftp://example.com"], ["https:///missing"], [1], "https://example.com"])
def test_contents_rejects_invalid_urls(enabled: None, monkeypatch: pytest.MonkeyPatch, urls: object) -> None:
    session = _patch_session(monkeypatch, FakeSession())
    assert exa_client.contents(cast(list[str], urls), caller="collector") == {
        "error": "Exa contents requires nonempty HTTP(S) URLs",
        "source": "exa",
    }
    assert not session.posts


def test_contents_disabled_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_ENABLED", "false")
    session = _patch_session(monkeypatch, FakeSession())
    assert exa_client.contents(["https://example.com"], caller="collector") == {
        "error": "Exa contents unavailable",
        "source": "exa",
    }
    assert not session.posts


@pytest.mark.parametrize(
    "payload,status,error",
    [({}, 200, "Exa contents returned an invalid response"), ({}, 503, "Exa search failed: HTTP 503")],
)
def test_contents_reports_bad_response(
    enabled: None, monkeypatch: pytest.MonkeyPatch, payload: object, status: int, error: str
) -> None:
    _patch_session(monkeypatch, FakeSession(_response(payload, status)))
    assert _err(exa_client.contents(["https://example.com"], caller="agent")) == error


def test_contents_reports_unavailable_budget(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_MONTHLY_LIMIT", "not dollars")
    session = _patch_session(monkeypatch, FakeSession())
    assert "Exa budget unavailable:" in _err(exa_client.contents(["https://example.com"], caller="collector"))
    assert not session.posts


def test_contents_budget_refusal_makes_no_request(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_COLLECTOR_MONTHLY_LIMIT", "0")
    session = _patch_session(monkeypatch, FakeSession())
    assert exa_client.contents(["https://example.com"], caller="collector") == {
        "error": "Exa monthly budget exhausted",
        "source": "exa",
    }
    assert not session.posts


def test_reservation_precedes_http(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")

    def request(payload: object, endpoint: str) -> object:
        assert payload == {"urls": ["https://example.com"], "text": {"maxCharacters": 12_000}}
        assert endpoint == "contents"
        assert market.spent("exa", period) == pytest.approx(0.001)
        assert market.spent("exa:collector", period) == pytest.approx(0.001)
        return _response({"results": [], "costDollars": {"total": 0}})

    monkeypatch.setattr(exa_client, "_request", request)
    assert exa_client.contents(["https://example.com"], caller="collector") == {
        "source": "exa",
        "evidence": [],
        "omitted_count": 0,
        "row_count": 0,
    }
    assert market.spent("exa", period) == 0
    assert market.spent("exa:collector", period) == 0


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {"costDollars": []},
        {"costDollars": {"total": "0.002"}},
        {"costDollars": {"total": float("inf")}},
        {"costDollars": {"total": False}},
    ],
)
def test_contents_keeps_estimate_without_numeric_cost(
    enabled: None, monkeypatch: pytest.MonkeyPatch, body: object
) -> None:
    _patch_session(monkeypatch, FakeSession(_response(body, 503)))
    assert "HTTP 503" in _err(exa_client.contents(["https://example.com"], caller="collector"))
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    assert market.spent("exa", period) == pytest.approx(0.001)
    assert market.spent("exa:collector", period) == pytest.approx(0.001)


def test_contents_refunds_no_response(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session.post.side_effect = requests.Timeout()
    monkeypatch.setattr(exa_client, "_ensure_session", lambda: session)
    assert exa_client.contents(["https://example.com"], caller="collector") == {
        "error": "Exa contents timed out",
        "source": "exa",
    }
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    assert market.spent("exa", period) == 0
    assert market.spent("exa:collector", period) == 0


def test_contents_estimate_scales_with_url_count(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, FakeSession(_response({"results": []})))
    urls = ["https://a.example", "https://b.example"]
    assert "error" not in exa_client.contents(urls, caller="agent")
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    assert market.spent("exa", period) == pytest.approx(0.002)


def test_search_reports_unavailable_budget(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> bool:
        raise sqlite3.Error("database is locked")

    monkeypatch.setattr(market, "reserve", fail)
    session = _patch_session(monkeypatch, FakeSession())
    assert _err(exa_client.search("AMD")) == "Exa budget unavailable: database is locked"
    assert not session.posts


def test_default_caps_apply_without_env(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_MONTHLY_LIMIT")
    monkeypatch.delenv("EXA_COLLECTOR_MONTHLY_LIMIT")
    _patch_session(monkeypatch, FakeSession(_response({"results": []})))
    assert "error" not in exa_client.contents(["https://example.com"], caller="collector")
    period = exa_client.datetime.now(exa_client.UTC).strftime("%Y-%m")
    assert market.spent("exa", period) == pytest.approx(exa_client.EXA_TEXT_PAGE_COST)
    assert market.spent("exa:collector", period) == pytest.approx(exa_client.EXA_TEXT_PAGE_COST)


def test_contents_normalizes_article_text(enabled: None, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(
        monkeypatch,
        FakeSession(
            _response(
                {
                    "results": [
                        None,
                        {
                            "url": "https://example.com/story",
                            "title": "Story",
                            "text": "Article",
                            "publishedDate": "2026-10-30",
                        },
                    ]
                }
            )
        ),
    )
    result = exa_client.contents(["https://example.com/story"], caller="agent")
    assert result["source"] == "exa"
    assert result["row_count"] == 1
    assert result["omitted_count"] == 1
    article = _as_dict(_as_seq(result["evidence"])[0])
    assert article["url"] == "https://example.com/story"
    assert article["source_content"] == "Article"
    assert article["source_domain"] == "example.com"
    assert isinstance(article["retrieved_at"], str) and article["retrieved_at"].endswith("+00:00")
