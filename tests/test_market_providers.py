"""Unit tests for the keyless market-data providers.

Deterministic and offline: _http_bytes is faked per URL; nothing touches
the network or cache.db.
"""

import json
import sys
import types
from typing import Self

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from api import market_providers as mp


class FakeCache:
    store: dict[str, object]

    def __init__(self) -> None:
        self.store = {}

    def get(self, key: str, ttl: float | None = None) -> object | None:
        return self.store.get(key)

    def set(self, key: str, value: object) -> None:
        self.store[key] = value


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> FakeCache:
    cache = FakeCache()
    monkeypatch.setattr(mp, "cache", cache)
    return cache


def _payload(closes: list[float | None]) -> dict[str, object]:
    n = len(closes)
    return {
        "chart": {
            "result": [
                {
                    "timestamp": list(range(1700000000, 1700000000 + n)),
                    "indicators": {
                        "quote": [
                            {
                                "open": list(closes),
                                "high": list(closes),
                                "low": list(closes),
                                "close": list(closes),
                                "volume": [10] * n,
                            }
                        ]
                    },
                    "meta": {"regularMarketPrice": 99.0},
                }
            ]
        }
    }


def _rss(items: list[tuple[str, str, str]]) -> bytes:
    rows = "".join(f"<item><title>{t}</title><link>{l}</link><pubDate>{p}</pubDate></item>" for t, l, p in items)
    return f"<rss><channel>{rows}</channel></rss>".encode()


def _atom(entries: list[tuple[str, str, str, str]]) -> bytes:
    rows = "".join(
        f'<entry><title>{t}</title><link href="{h}"/><updated>{u}</updated><summary>{s}</summary></entry>'
        for t, h, u, s in entries
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{rows}</feed>'.encode()


def test_parse_candles_normal_fills_gaps() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1000, 2000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [9.0],
                                "high": [9.5],
                                "low": [8.5],
                                "close": [10.0, 11.0],
                                "volume": [5, None],
                            }
                        ]
                    },
                    "meta": {"regularMarketPrice": 11.5},
                }
            ]
        }
    }
    out = mp._parse_candles("NVDA", payload, "6mo", "1d")
    assert out["sym"] == "NVDA"
    assert out["price"] == 11.5
    assert out["delayed"] is True
    assert out["candles"] == [
        {"t": 1000, "open": 9.0, "high": 9.5, "low": 8.5, "close": 10.0, "vol": 5},
        {"t": 2000, "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "vol": 0},
    ]


def test_parse_candles_empty_raises() -> None:
    with pytest.raises(ValueError, match="No data"):
        mp._parse_candles(
            "NVDA",
            {"chart": {"result": [], "error": {"description": "No data"}}},
            "6mo",
            "1d",
        )
    with pytest.raises(ValueError, match="no data"):
        mp._parse_candles("NVDA", {"chart": {}}, "6mo", "1d")


def test_parse_candles_skips_null_close() -> None:
    out = mp._parse_candles("NVDA", _payload([None, 5.0]), "6mo", "1d")
    candles = out["candles"]
    assert isinstance(candles, list) and len(candles) == 1
    first = candles[0]
    assert isinstance(first, dict) and first["close"] == 5.0


def test_parse_candles_all_null_raises() -> None:
    with pytest.raises(ValueError, match="all bars empty"):
        mp._parse_candles("NVDA", _payload([None, None]), "6mo", "1d")


@settings(max_examples=100, derandomize=True)
@given(
    st.lists(
        st.one_of(
            st.none(),
            st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    )
)
def test_prop_parse_candles_keeps_only_present_closes(
    closes: list[float | None],
) -> None:
    try:
        out = mp._parse_candles("NVDA", _payload(closes), "6mo", "1d")
    except ValueError:
        assert all(c is None for c in closes)
        return
    candles = out["candles"]
    assert isinstance(candles, list)
    got: list[object] = [row["close"] for row in candles if isinstance(row, dict)]
    assert got == [c for c in closes if c is not None]


def test_candles_throttle_uses_negative_cache(fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def boom(url: str, **kwargs: object) -> bytes:
        calls["n"] += 1
        raise mp._Throttle("Yahoo throttled (429)")

    monkeypatch.setattr(mp, "_http_bytes", boom)
    first = mp.get_candles("NVDA")
    assert calls["n"] == 1
    assert "error" in first
    second = mp.get_candles("nvda")
    assert calls["n"] == 1
    assert second.get("sym") == "NVDA"
    assert "error" in second
    assert "neg:candles:NVDA:6mo:1d" in fake_cache.store


def test_rss_bad_xml_yields_empty() -> None:
    assert mp._rss_items(b"not xml", "Google News") == []
    assert mp._rss_items(b"<rss><channel><item><title>  </title></item></channel></rss>", "X") == []


def test_news_merge_sorts_dedups_filters_topics(fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "0")
    feeds = {
        "google": _rss(
            [
                (
                    "NVDA rallies on AI demand",
                    "https://ex.com/g1",
                    "Mon, 06 Oct 2026 10:00:00 GMT",
                ),
                (
                    "Chip stocks mixed",
                    "https://ex.com/shared",
                    "Mon, 06 Oct 2026 09:00:00 GMT",
                ),
            ]
        ),
        "yahoo": _rss(
            [
                (
                    "NVDA price target raised",
                    "https://ex.com/y1",
                    "Mon, 06 Oct 2026 11:00:00 GMT",
                ),
                (
                    "Market outlook",
                    "https://ex.com/y2",
                    "Mon, 06 Oct 2026 11:30:00 GMT",
                ),
            ]
        ),
        "sa": _rss(
            [
                (
                    "Chip stocks mixed",
                    "https://ex.com/shared",
                    "Mon, 06 Oct 2026 09:00:00 GMT",
                ),
                (
                    "Semiconductor update",
                    "https://ex.com/s1",
                    "Mon, 06 Oct 2026 08:00:00 GMT",
                ),
            ]
        ),
        "cnbc": _rss(
            [
                (
                    "NVDA unveils new chip",
                    "https://ex.com/c1",
                    "Mon, 06 Oct 2026 07:00:00 GMT",
                ),
                ("Markets today", "https://ex.com/c2", "Mon, 06 Oct 2026 07:30:00 GMT"),
            ]
        ),
        "mw": _rss([("Dow falls", "https://ex.com/m1", "Mon, 06 Oct 2026 06:00:00 GMT")]),
    }

    def fake_http(url: str, **kwargs: object) -> bytes:
        if "news.google.com" in url:
            return feeds["google"]
        if "feeds.finance.yahoo.com" in url:
            return feeds["yahoo"]
        if "seekingalpha.com" in url:
            return feeds["sa"]
        if "cnbc.com" in url:
            return feeds["cnbc"]
        if "marketwatch.com" in url:
            return feeds["mw"]
        raise AssertionError(url)

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_news("NVDA")
    items = out["items"]
    assert isinstance(items, list)
    titles = [i["title"] for i in items if isinstance(i, dict)]
    assert titles == [
        "NVDA price target raised",
        "NVDA rallies on AI demand",
        "Chip stocks mixed",
        "Semiconductor update",
        "NVDA unveils new chip",
    ]
    links = [i["link"] for i in items if isinstance(i, dict)]
    assert len(set(links)) == len(links)
    first = items[0]
    assert isinstance(first, dict) and first["source"] == "Yahoo Finance"
    mp.get_news("NVDA")
    assert "news:NVDA" in fake_cache.store


def test_filings_atom_parses_entries(fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    universe = {"0": {"ticker": "NVDA", "cik_str": "1341439", "title": "NVIDIA Corp"}}
    atom = _atom(
        [
            (
                "NVIDIA files 10-Q",
                "https://sec.gov/a",
                "2026-10-01T00:00:00Z",
                "Quarterly report",
            ),
            (
                "NVIDIA files 8-K",
                "https://sec.gov/b",
                "2026-09-01T00:00:00Z",
                "x" * 500,
            ),
        ]
    )

    def fake_http(url: str, **kwargs: object) -> bytes:
        if "company_tickers.json" in url:
            return json.dumps(universe).encode()
        if "browse-edgar" in url:
            return atom
        raise AssertionError(url)

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings("NVDA")
    assert out["sym"] == "NVDA"
    assert out["cik"] == 1341439
    items = out["items"]
    assert isinstance(items, list) and len(items) == 2
    first, second = items
    assert isinstance(first, dict) and first["title"] == "NVIDIA files 10-Q"
    assert isinstance(first, dict) and first["link"] == "https://sec.gov/a"
    assert isinstance(second, dict) and len(str(second["summary"])) == 300
    assert "market_universe" in fake_cache.store
    assert "filings:1341439" in fake_cache.store


def test_filings_no_cik_returns_error(fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    universe = {"0": {"ticker": "AAPL", "cik_str": "320193", "title": "Apple Inc"}}

    def fake_http(url: str, **kwargs: object) -> bytes:
        assert "company_tickers.json" in url
        return json.dumps(universe).encode()

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings("NVDA")
    assert out["cik"] is None
    assert out["items"] == []
    assert "no SEC CIK" in str(out.get("error"))


def test_newsquant_disabled_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "0")
    assert mp._newsquant_items("NVDA") == []


def test_newsquant_missing_dependency_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "1")
    monkeypatch.setitem(sys.modules, "newsquant", None)
    assert mp._newsquant_items("NVDA") == []


def test_newsquant_fetch_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "1")

    class _FailingScraper:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fetch(self, **kwargs: object) -> list[object]:
            raise RuntimeError("boom")

    mod = types.ModuleType("newsquant")
    mod.__dict__["Scraper"] = _FailingScraper
    monkeypatch.setitem(sys.modules, "newsquant", mod)

    class _SyncThread:
        def __init__(self, target: object = None, **kwargs: object) -> None:
            assert callable(target)
            self._target = target

        def start(self) -> None:
            assert callable(self._target)
            self._target()

        def join(self, timeout: float | None = None) -> None:
            return None

    monkeypatch.setattr(mp, "threading", types.SimpleNamespace(Thread=_SyncThread))
    assert mp._newsquant_items("NVDA") == []


class _FakeYahooResp:
    def __init__(self, status: int, content: bytes, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeYahooSession:
    def __init__(self, resp: _FakeYahooResp) -> None:
        self.resp = resp
        self.seen: dict[str, object] = {}

    def get(self, url: str, headers: dict[str, str] | None = None, timeout: object = None) -> _FakeYahooResp:
        self.seen = {"url": url, "headers": headers, "timeout": timeout}
        return self.resp


class _FakeOpened:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.payload


def test_http_bytes_yahoo_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeYahooResp(200, b"yahoo-bytes")
    sess = _FakeYahooSession(resp)
    monkeypatch.setattr(mp, "_yahoo_session", lambda: sess)

    def _boom_urlopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-Yahoo transport must not run for Yahoo URLs")

    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    out = mp._http_bytes(
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA",
        headers={"Accept": "x"},
    )
    assert out == b"yahoo-bytes"
    assert sess.seen["url"] == "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA"
    assert sess.seen["headers"] == {"Accept": "x"}
    assert sess.seen["timeout"] == mp.REQUEST_TIMEOUT_SECONDS


def test_http_bytes_yahoo_throttle_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp = _FakeYahooResp(429, b"", headers={})
    monkeypatch.setattr(mp, "_yahoo_session", lambda: _FakeYahooSession(resp))
    with pytest.raises(mp._Throttle, match=r"Yahoo throttled \(429\)$"):
        mp._http_bytes("https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA")


def test_http_bytes_yahoo_throttle_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp = _FakeYahooResp(429, b"", headers={"Retry-After": "7"})
    monkeypatch.setattr(mp, "_yahoo_session", lambda: _FakeYahooSession(resp))
    with pytest.raises(mp._Throttle, match="retry after 7s"):
        mp._http_bytes("https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA")


def test_http_bytes_non_yahoo_sends_default_sec_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.request

    monkeypatch.delenv("SEC_EDGAR_IDENTITY", raising=False)
    seen: dict[str, object] = {}

    def _count_identity() -> None:
        prior = seen.get("identity")
        seen["identity"] = prior + 1 if isinstance(prior, int) else 1

    monkeypatch.setattr("app.sec.client.ensure_identity", _count_identity)

    def _boom_session() -> object:
        raise AssertionError("Yahoo transport must not run for SEC URLs")

    monkeypatch.setattr(mp, "_yahoo_session", _boom_session)
    captured: dict[str, object] = {}

    def _fake_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeOpened:
        captured["req"] = req
        captured["timeout"] = timeout
        return _FakeOpened(b"sec-bytes")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    out = mp._http_bytes(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"Accept": "application/json"},
    )
    assert out == b"sec-bytes"
    assert seen.get("identity") == 1
    req = captured["req"]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == "https://www.sec.gov/files/company_tickers.json"
    assert captured["timeout"] == 20
    sent = {k.lower(): v for k, v in req.header_items()}
    assert sent["user-agent"] == "Stockbot stockbot@example.com"
    assert sent["accept"] == "application/json"


def test_http_bytes_non_yahoo_uses_env_sec_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.request

    monkeypatch.setenv("SEC_EDGAR_IDENTITY", "Tester tester@example.com")
    monkeypatch.setattr("app.sec.client.ensure_identity", lambda: None)
    captured: dict[str, object] = {}

    def _fake_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeOpened:
        captured["req"] = req
        return _FakeOpened(b"sec-bytes")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    assert mp._http_bytes("https://www.sec.gov/files/company_tickers.json") == b"sec-bytes"
    req = captured["req"]
    assert isinstance(req, urllib.request.Request)
    sent = {k.lower(): v for k, v in req.header_items()}
    assert sent["user-agent"] == "Tester tester@example.com"


def test_news_sort_key_fallbacks_are_zero() -> None:
    assert mp._news_sort_key({}) == 0.0
    assert mp._news_sort_key({"published": ""}) == 0.0
    assert mp._news_sort_key({"published": None}) == 0.0
    assert mp._news_sort_key({"published": "not a date"}) == 0.0


def test_news_sort_key_parses_rfc2822() -> None:
    from datetime import UTC, datetime

    expected = datetime(2026, 10, 6, 11, 30, tzinfo=UTC).timestamp()
    assert mp._news_sort_key({"published": "Mon, 06 Oct 2026 11:30:00 GMT"}) == expected


def test_news_sort_key_parses_iso_z() -> None:
    from datetime import UTC, datetime

    expected = datetime(2026, 10, 6, 11, 30, tzinfo=UTC).timestamp()
    assert mp._news_sort_key({"published": "2026-10-06T11:30:00Z"}) == expected


def test_search_tickers_empty_query_skips_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(url: str, **kwargs: object) -> bytes:
        raise AssertionError("transport must not run for empty queries")

    monkeypatch.setattr(mp, "_http_bytes", _boom)
    assert mp.search_tickers("   ") == {"query": "", "hits": []}


def test_search_tickers_maps_hits_and_slices_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake(url: str, **kwargs: object) -> bytes:
        payload = {
            "quotes": [
                {
                    "symbol": "NVDA",
                    "shortname": "NVIDIA Corp",
                    "exchange": "NMS",
                    "quoteType": "EQUITY",
                },
                {
                    "symbol": "",
                    "shortname": "No symbol",
                    "exchange": "X",
                    "quoteType": "EQUITY",
                },
                "not-a-dict",
                {
                    "symbol": "NVD",
                    "longname": "NVD long",
                    "exchange": "NMS",
                    "quoteType": "ETF",
                },
            ]
        }
        return json.dumps(payload).encode()

    monkeypatch.setattr(mp, "_http_bytes", _fake)
    one = mp.search_tickers(" nvda ", limit=1)
    assert one["query"] == "nvda"
    assert one["hits"] == [{"sym": "NVDA", "name": "NVIDIA Corp", "exchange": "NMS", "type": "EQUITY"}]
    many = mp.search_tickers("nvda", limit=10)
    hits = many["hits"]
    assert isinstance(hits, list) and len(hits) == 2
    second = hits[1]
    assert isinstance(second, dict) and second["name"] == "NVD long"


def test_search_tickers_throttle_returns_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(url: str, **kwargs: object) -> bytes:
        raise mp._Throttle("Yahoo throttled (429), retry after 3s")

    monkeypatch.setattr(mp, "_http_bytes", _boom)
    out = mp.search_tickers("nvda")
    assert out["query"] == "nvda"
    assert out["hits"] == []
    assert "search unavailable" in str(out.get("error"))


def test_search_tickers_failure_returns_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(url: str, **kwargs: object) -> bytes:
        raise RuntimeError("down")

    monkeypatch.setattr(mp, "_http_bytes", _boom)
    out = mp.search_tickers("nvda")
    assert out["query"] == "nvda"
    assert out["hits"] == []
    assert "search unavailable" in str(out.get("error"))


def test_get_news_empty_symbol_returns_empty(fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(url: str, **kwargs: object) -> bytes:
        raise AssertionError("transport must not run for empty symbols")

    monkeypatch.setattr(mp, "_http_bytes", _boom)
    assert mp.get_news("   ") == {"sym": "", "items": []}
    assert fake_cache.store == {}


def test_viewer_context_aggregates_window(monkeypatch: pytest.MonkeyPatch) -> None:
    def _quote(sym: str) -> dict[str, object]:
        return {"sym": sym, "price": 10.0, "retrieved_at": "t"}

    def _candles(sym: str) -> dict[str, object]:
        return {"sym": sym, "candles": [{"close": 10.0}, {"close": 11.0}]}

    def _news(sym: str, limit: int = 12) -> dict[str, object]:
        return {"sym": sym, "items": [{"title": "a"}]}

    def _filings(sym: str, limit: int = 10) -> dict[str, object]:
        return {"sym": sym, "cik": 1341439, "items": [{"title": "f"}]}

    monkeypatch.setattr(mp, "get_quote", _quote)
    monkeypatch.setattr(mp, "get_candles", _candles)
    monkeypatch.setattr(mp, "get_news", _news)
    monkeypatch.setattr(mp, "get_filings", _filings)
    out = mp.get_viewer_context(" nvda ")
    assert out["sym"] == "NVDA"
    assert out["window"] == {
        "bars": 2,
        "last": 11.0,
        "first": 10.0,
        "high": 11.0,
        "low": 10.0,
        "change_pct": 10.0,
    }
    assert out["news"] == [{"title": "a"}]
    assert out["filings"] == [{"title": "f"}]
    assert out["cik"] == 1341439
    assert out["errors"] == {}


def test_viewer_context_empty_closes_yields_empty_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _quote(sym: str) -> dict[str, object]:
        return {"sym": sym, "price": 1.0, "retrieved_at": "t"}

    def _news(sym: str, limit: int = 12) -> dict[str, object]:
        return {"sym": sym, "items": []}

    def _filings(sym: str, limit: int = 10) -> dict[str, object]:
        return {"sym": sym, "cik": None, "items": []}

    def _no_candles(sym: str) -> dict[str, object]:
        return {"sym": sym, "error": "candles unavailable: down"}

    def _bad_candles(sym: str) -> dict[str, object]:
        return {"sym": sym, "candles": [{"close": "bad"}, {"close": None}, {}]}

    monkeypatch.setattr(mp, "get_quote", _quote)
    monkeypatch.setattr(mp, "get_news", _news)
    monkeypatch.setattr(mp, "get_filings", _filings)
    monkeypatch.setattr(mp, "get_candles", _no_candles)
    missing = mp.get_viewer_context("NVDA")
    missing_window = missing["window"]
    assert isinstance(missing_window, dict)
    assert missing_window == {
        "bars": 0,
        "last": None,
        "first": None,
        "high": None,
        "low": None,
    }
    assert "change_pct" not in missing_window
    monkeypatch.setattr(mp, "get_candles", _bad_candles)
    non_numeric = mp.get_viewer_context("NVDA")
    non_numeric_window = non_numeric["window"]
    assert isinstance(non_numeric_window, dict)
    assert non_numeric_window == {
        "bars": 0,
        "last": None,
        "first": None,
        "high": None,
        "low": None,
    }
    assert "change_pct" not in non_numeric_window


def test_viewer_context_change_pct_math_and_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _quote(sym: str) -> dict[str, object]:
        return {"sym": sym, "price": 1.0, "retrieved_at": "t"}

    def _news(sym: str, limit: int = 12) -> dict[str, object]:
        return {"sym": sym, "items": []}

    def _filings(sym: str, limit: int = 10) -> dict[str, object]:
        return {"sym": sym, "cik": None, "items": []}

    monkeypatch.setattr(mp, "get_quote", _quote)
    monkeypatch.setattr(mp, "get_news", _news)
    monkeypatch.setattr(mp, "get_filings", _filings)

    def _window_for(closes: list[object]) -> dict[str, object]:
        def _candles(sym: str) -> dict[str, object]:
            return {"sym": sym, "candles": [{"close": c} for c in closes]}

        monkeypatch.setattr(mp, "get_candles", _candles)
        out = mp.get_viewer_context("NVDA")
        window = out["window"]
        assert isinstance(window, dict)
        return window

    assert _window_for([10.0, 12.0])["change_pct"] == 20.0
    assert "change_pct" not in _window_for([10.0])
    assert "change_pct" not in _window_for([0.0, 5.0])


def test_viewer_context_collects_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _quote(sym: str) -> dict[str, object]:
        return {
            "sym": sym,
            "price": None,
            "retrieved_at": None,
            "error": "quote unavailable: x",
        }

    def _candles(sym: str) -> dict[str, object]:
        return {"sym": sym, "error": "candles unavailable: y"}

    def _news(sym: str, limit: int = 12) -> dict[str, object]:
        return {"sym": sym, "items": [], "error": "news boom"}

    def _filings(sym: str, limit: int = 10) -> dict[str, object]:
        return {"sym": sym, "cik": None, "items": [], "error": "filings boom"}

    monkeypatch.setattr(mp, "get_quote", _quote)
    monkeypatch.setattr(mp, "get_candles", _candles)
    monkeypatch.setattr(mp, "get_news", _news)
    monkeypatch.setattr(mp, "get_filings", _filings)
    out = mp.get_viewer_context("NVDA")
    assert out["errors"] == {
        "quote": "quote unavailable: x",
        "candles": "candles unavailable: y",
        "news": "news boom",
        "filings": "filings boom",
    }
    assert out["news"] == []
    assert out["filings"] == []
