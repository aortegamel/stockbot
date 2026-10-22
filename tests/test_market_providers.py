"""Unit tests for the keyless market-data providers.

Deterministic and offline: _http_bytes is faked per URL; nothing touches
the network or cache.db.
"""

import json
import re
import sys
import types
from collections.abc import Iterator
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Self

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from api import market_providers as mp
from api.main import app
from app import cache as real_cache
from app.sec.models import Filing, InsiderTransaction


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


def _atom(entries: list[tuple[str, str, str, str, str]]) -> bytes:
    """EDGAR getcompany Atom: (title, href, updated, summary, filing-date) per entry."""
    rows = "".join(
        f'<entry><content type="text/xml"><filing-date>{f}</filing-date></content>'
        f'<title>{t}</title><link href="{h}"/><updated>{u}</updated><summary>{s}</summary></entry>'
        for t, h, u, s, f in entries
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{rows}</feed>'.encode()


@pytest.fixture
def fund_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[list[object]]]:
    """Stub SEC company_tickers_mf.json rows behind edgartools' real fund lookup."""
    import edgar.reference.tickers as edgar_tickers

    rows: list[list[object]] = []

    def fake_download_json(url: str) -> dict[str, object]:
        assert url.endswith("company_tickers_mf.json"), url
        return {"fields": ["cik", "seriesId", "classId", "symbol"], "data": rows}

    monkeypatch.setattr(edgar_tickers, "download_json", fake_download_json)
    monkeypatch.setattr("app.sec.client.ensure_identity", lambda: None)
    edgar_tickers.get_mutual_fund_tickers.cache_clear()
    edgar_tickers.get_mutual_fund_lookup.cache_clear()
    yield rows
    edgar_tickers.get_mutual_fund_tickers.cache_clear()
    edgar_tickers.get_mutual_fund_lookup.cache_clear()


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


def test_news_warm_cache_slices_without_mutation_or_fetch(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [
        {"title": "Second headline", "published": "2026-10-02"},
        {"title": "First headline", "published": "2026-10-01"},
        {"title": "Fourth headline", "published": "2026-10-04"},
        {"title": "Third headline", "published": "2026-10-03"},
    ]
    cached = {"sym": "NVDA", "items": items, "source": "cached news"}
    fake_cache.set("news:NVDA", cached)
    before = json.dumps(cached)
    calls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        raise AssertionError("warm news must not fetch")

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "0")
    for limit in (2, 1, 4, 8, 0):
        out = mp.get_news(" nvda ", limit=limit)
        assert out == {"sym": "NVDA", "items": items[:limit], "source": "cached news"}
        assert out["items"] is not items
        assert fake_cache.store["news:NVDA"] is cached
        assert json.dumps(cached) == before
    assert calls == []


def test_news_cold_limit_stays_cached_for_larger_request(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = _rss(
        [
            ("Oldest", "https://ex.com/1", "Thu, 01 Oct 2026 10:00:00 GMT"),
            ("Newest", "https://ex.com/4", "Sun, 04 Oct 2026 10:00:00 GMT"),
            ("Second newest", "https://ex.com/3", "Sat, 03 Oct 2026 10:00:00 GMT"),
            ("Second oldest", "https://ex.com/2", "Fri, 02 Oct 2026 10:00:00 GMT"),
        ]
    )
    calls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        return feed if "news.google.com" in url else _rss([])

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "0")
    cold = mp.get_news("NVDA", limit=2)
    items = cold["items"]
    assert isinstance(items, list)
    assert [item["title"] for item in items] == ["Newest", "Second newest"]
    assert len(calls) == 5
    before = json.dumps(cold)
    for limit in (1, 4):
        assert mp.get_news("NVDA", limit=limit) == dict(cold, items=items[:limit])
        assert len(calls) == 5
        assert fake_cache.store["news:NVDA"] is cold
        assert json.dumps(cold) == before


def _universe_bytes(*rows: tuple[str, int, str]) -> bytes:
    """Stub SEC company_tickers.json body."""
    return json.dumps(
        {str(i): {"ticker": t, "cik_str": str(c), "title": n} for i, (t, c, n) in enumerate(rows)}
    ).encode()


def test_filings_atom_parses_entries(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch, fund_registry: list[list[object]]
) -> None:
    atom = _atom(
        [
            (
                "NVIDIA files 10-Q",
                "https://sec.gov/a",
                "2026-10-01T00:00:00Z",
                "Quarterly report",
                "2026-10-01",
            ),
            (
                "NVIDIA files 8-K",
                "https://sec.gov/b",
                "2026-09-01T00:00:00Z",
                "x" * 500,
                "2026-09-01",
            ),
        ]
    )
    urls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        urls.append(url)
        if "company_tickers.json" in url:
            return _universe_bytes(("NVDA", 1341439, "NVIDIA Corp"))
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
    assert isinstance(first, dict) and first["filed"] == "2026-10-01"
    assert isinstance(first, dict) and first["updated"] == "2026-10-01T00:00:00Z"
    assert isinstance(first, dict) and "published" not in first
    assert isinstance(second, dict) and len(str(second["summary"])) == 300
    assert "market_universe" in fake_cache.store
    assert "filings:v2:1341439" in fake_cache.store
    atom_url = urls[-1]
    assert "CIK=0001341439" in atom_url and "type=&" in atom_url
    assert "owner=include" in atom_url and "count=40" in atom_url


@pytest.mark.parametrize(
    ("sym", "cik", "name", "filed", "updated"),
    [
        ("TSLA", 1318605, "Tesla, Inc.", "2026-07-23", "2026-07-22T18:05:11-04:00"),
        ("SPY", 884394, "SPDR S&P 500 ETF TRUST", "2026-03-13", "2026-03-12T17:20:00-04:00"),
    ],
)
def test_filings_filed_is_official_date_not_atom_updated(
    fake_cache: FakeCache,
    monkeypatch: pytest.MonkeyPatch,
    fund_registry: list[list[object]],
    sym: str,
    cik: int,
    name: str,
    filed: str,
    updated: str,
) -> None:
    atom = _atom([("10-Q  - Quarterly report", "https://sec.gov/a", updated, "report", filed)])

    def fake_http(url: str, **kwargs: object) -> bytes:
        if "company_tickers.json" in url:
            return _universe_bytes((sym, cik, name))
        assert f"CIK={cik:010d}" in url
        return atom

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings(sym)
    assert out["cik"] == cik
    items = out["items"]
    assert isinstance(items, list)
    assert items[0]["filed"] == filed
    assert items[0]["updated"] == updated


def test_filings_fund_resolves_trust_cik_via_edgartools_registry(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch, fund_registry: list[list[object]]
) -> None:
    fund_registry.extend(
        [
            [36405, "S000002277", "C000005765", "VFIAX"],
            [1067839, "S000101292", "C000271435", "QQQ"],
        ]
    )
    atom = _atom([("N-CEN  - Annual report", "https://sec.gov/q", "2026-08-12T16:02:11-04:00", "s", "2026-08-12")])
    urls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        urls.append(url)
        if "company_tickers.json" in url:
            # Sponsor and sibling ETF present; QQQ itself absent from the company universe.
            return _universe_bytes(("SPY", 884394, "SPDR S&P 500 ETF TRUST"), ("IVZ", 914208, "Invesco Ltd."))
        return atom

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings(" qqq ")
    assert out["sym"] == "QQQ"
    assert out["cik"] == 1067839
    assert "CIK=0001067839" in urls[-1]
    items = out["items"]
    assert isinstance(items, list) and items[0]["filed"] == "2026-08-12"


def test_filings_warm_cache_slices_and_preserves_alias(fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        {"title": "Second filing", "filed": "2026-10-02", "updated": "2026-10-01T18:00:00-04:00"},
        {"title": "First filing", "filed": "2026-10-01", "updated": "2026-10-01T09:00:00-04:00"},
        {"title": "Fourth filing", "filed": "2026-10-04", "updated": "2026-10-04T09:00:00-04:00"},
        {"title": "Third filing", "filed": "2026-10-03", "updated": "2026-10-03T09:00:00-04:00"},
    ]
    cached = {"sym": "GOOG", "cik": 1652044, "items": items, "source": "cached filings"}
    fake_cache.set("filings:v2:1652044", cached)
    before = json.dumps(cached)
    calls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        raise AssertionError("warm filings must not fetch")

    def fake_cik(ticker: str) -> int:
        assert ticker == "GOOGL"
        return 1652044

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    monkeypatch.setattr(mp, "cik_for_ticker", fake_cik)
    for limit in (2, 1, 4, 8, 0):
        out = mp.get_filings(" googl ", limit=limit)
        assert out == dict(cached, sym="GOOGL", items=items[:limit])
        assert out["items"] is not items
        assert fake_cache.store["filings:v2:1652044"] is cached
        assert json.dumps(cached) == before
    assert calls == []


def test_filings_small_first_request_does_not_truncate_larger(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre_v2 = {"sym": "GOOG", "cik": 1652044, "items": [{"title": "old", "published": "2026-10-04T00:00:00Z"}]}
    fake_cache.set("filings:1652044", pre_v2)
    atom = _atom(
        [
            ("10-Q", "https://sec.gov/a", "2026-10-04T00:00:00Z", "Quarterly report", "2026-10-04"),
            ("8-K", "https://sec.gov/b", "2026-10-03T00:00:00Z", "Current report", "2026-10-03"),
            ("10-K", "https://sec.gov/c", "2026-10-02T00:00:00Z", "Annual report", "2026-10-02"),
        ]
    )
    calls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        assert "browse-edgar" in url and "count=40" in url
        return atom

    def fake_cik(ticker: str) -> int:
        assert ticker in ("GOOG", "GOOGL")
        return 1652044

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    monkeypatch.setattr(mp, "cik_for_ticker", fake_cik)
    small = mp.get_filings("GOOG", limit=2)
    small_items = small["items"]
    assert isinstance(small_items, list)
    assert [item["title"] for item in small_items] == ["10-Q", "8-K"]
    cached = fake_cache.store["filings:v2:1652044"]
    assert isinstance(cached, dict)
    full = cached["items"]
    assert isinstance(full, list) and [item["title"] for item in full] == ["10-Q", "8-K", "10-K"]
    before = json.dumps(cached)
    for limit in (1, 4, 40):
        assert mp.get_filings("GOOGL", limit=limit) == dict(cached, sym="GOOGL", items=full[:limit])
        assert json.dumps(cached) == before
    assert len(calls) == 1
    assert fake_cache.store["filings:1652044"] is pre_v2


def test_filings_entry_without_filing_date_is_error_not_guess(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    atom = _atom([("8-K", "https://sec.gov/b", "2026-10-03T18:00:00-04:00", "Current report", "")])

    def fake_http(url: str, **kwargs: object) -> bytes:
        return atom

    def fake_cik(ticker: str) -> int:
        return 1652044

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    monkeypatch.setattr(mp, "cik_for_ticker", fake_cik)
    out = mp.get_filings("GOOG")
    assert out["items"] == []
    assert "filing-date" in str(out.get("error"))
    assert "filings:v2:1652044" not in fake_cache.store


@pytest.mark.parametrize("payload", [{}, {"items": None}, {"items": {}}, {"items": "cached"}])
def test_warm_cache_without_list_items_preserves_payload(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    cached = dict(payload, sym="GOOG", error="cached error")
    fake_cache.set("news:GOOG", cached)
    fake_cache.set("filings:v2:1652044", cached)

    def fake_cik(ticker: str) -> int:
        assert ticker == "GOOGL"
        return 1652044

    monkeypatch.setattr(mp, "cik_for_ticker", fake_cik)
    calls: list[str] = []

    def fake_http(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        raise AssertionError("warm providers must not fetch")

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    before = json.dumps(cached)
    assert mp.get_news("GOOG", limit=1) == cached
    assert mp.get_filings("GOOGL", limit=1) == dict(cached, sym="GOOGL")
    assert json.dumps(cached) == before
    assert calls == []


def test_filings_no_cik_returns_error(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch, fund_registry: list[list[object]]
) -> None:
    fund_registry.append([1067839, "S000101292", "C000271435", "QQQ"])

    def fake_http(url: str, **kwargs: object) -> bytes:
        assert "company_tickers.json" in url
        return _universe_bytes(("AAPL", 320193, "Apple Inc"))

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings("NVDA")
    assert out["cik"] is None
    assert out["items"] == []
    assert "no SEC CIK" in str(out.get("error"))


def test_filings_fund_registry_failure_surfaces_error(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch, fund_registry: list[list[object]]
) -> None:
    import edgar.reference.tickers as edgar_tickers

    def registry_down(url: str) -> dict[str, object]:
        raise ConnectionError("SEC 503")

    def fake_http(url: str, **kwargs: object) -> bytes:
        return _universe_bytes(("AAPL", 320193, "Apple Inc"))

    monkeypatch.setattr(edgar_tickers, "download_json", registry_down)
    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings("QQQ")
    assert out["cik"] is None and out["items"] == []
    assert out["error"] == "filings unavailable: CIK lookup failed: SEC 503"


def test_filings_universe_failure_surfaces_error_not_missing_cik(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch, fund_registry: list[list[object]]
) -> None:
    def fake_http(url: str, **kwargs: object) -> bytes:
        raise ConnectionError("SEC 503")

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    out = mp.get_filings("TSLA")
    assert out["cik"] is None and out["items"] == []
    assert out["error"] == "filings unavailable: CIK lookup failed: universe unavailable: SEC 503"


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
    assert out["filings"] == []


def test_api_root_reports_supervised_port(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("PORT", "3002")
    assert client.get("/").json()["ui"] == "http://localhost:3002"
    monkeypatch.delenv("PORT", raising=False)
    assert client.get("/").json()["ui"] == "http://localhost:5173"


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"chart": None}, "no data"),
        ({"chart": {"result": "bad", "error": {"description": "provider detail"}}}, "provider detail"),
        ({"chart": {"result": [], "error": "bad"}}, "no data"),
        ({"chart": {"result": [None]}}, "all bars empty"),
        ({"chart": {"result": [{"timestamp": [123], "indicators": {"quote": [None]}}]}}, "all bars empty"),
    ],
)
def test_candles_malformed_chart_returns_provider_error(
    fake_cache: FakeCache,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    detail: str,
) -> None:
    def fake_http(url: str, **kwargs: object) -> bytes:
        return json.dumps(payload).encode()

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    assert mp.get_candles(" nvda ") == {"error": f"candles unavailable for NVDA: {detail}"}
    assert fake_cache.store == {}


def test_candles_public_preserves_timestamps_and_sparse_columns(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [3000, 1000, 2000, 4000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [None],
                                "high": "bad",
                                "low": None,
                                "close": [8, None, 6],
                                "volume": [None],
                            }
                        ]
                    },
                    "meta": {"regularMarketPrice": 7},
                }
            ]
        }
    }

    def fake_http(url: str, **kwargs: object) -> bytes:
        return json.dumps(payload).encode()

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    assert mp.get_candles(" nvda ", range_="1mo", interval="1h") == {
        "sym": "NVDA",
        "range": "1mo",
        "interval": "1h",
        "price": 7,
        "delayed": True,
        "candles": [
            {"t": 3000, "open": None, "high": 8, "low": 8, "close": 8, "vol": 0},
            {"t": 2000, "open": 6, "high": 6, "low": 6, "close": 6, "vol": 0},
        ],
    }


@settings(max_examples=100, derandomize=True)
@given(
    st.lists(
        st.one_of(st.none(), st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False)),
        min_size=1,
        max_size=8,
    )
)
def test_prop_candles_public_retains_present_close_order(closes: list[float | None]) -> None:
    def fake_http(url: str, **kwargs: object) -> bytes:
        return json.dumps(_payload(closes)).encode()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mp, "cache", FakeCache())
        patch.setattr(mp, "_http_bytes", fake_http)
        out = mp.get_candles("NVDA")
    expected = [value for value in closes if value is not None]
    if not expected:
        assert out == {"error": "candles unavailable for NVDA: all bars empty"}
        return
    rows = out["candles"]
    assert isinstance(rows, list)
    assert [row["close"] for row in rows] == expected
    assert [row["t"] for row in rows] == [1700000000 + index for index, value in enumerate(closes) if value is not None]


def test_news_missing_links_dedup_by_title_preserves_first_source(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = _rss([("NVDA headline", "", "Tue, 06 Oct 2026 10:00:00 GMT")])

    def fake_http(url: str, **kwargs: object) -> bytes:
        return feed

    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "0")
    assert mp.get_news(" nvda ") == {
        "sym": "NVDA",
        "items": [
            {
                "title": "NVDA headline",
                "link": "",
                "published": "Tue, 06 Oct 2026 10:00:00 GMT",
                "source": "Google News",
            }
        ],
    }


def test_filings_missing_optional_fields_preserves_official_date(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    atom = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        b"<content><filing-date> 2026-10-03 </filing-date></content>"
        b"</entry></feed>"
    )

    def fake_cik(ticker: str) -> int:
        return 1341439

    def fake_http(url: str, **kwargs: object) -> bytes:
        return atom

    monkeypatch.setattr(mp, "cik_for_ticker", fake_cik)
    monkeypatch.setattr(mp, "_http_bytes", fake_http)
    assert mp.get_filings(" nvda ") == {
        "sym": "NVDA",
        "cik": 1341439,
        "items": [{"title": "", "link": "", "filed": "2026-10-03", "updated": "", "summary": ""}],
    }


def test_viewer_context_preserves_call_order_clock_and_numeric_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def quote(sym: str) -> dict[str, object]:
        calls.append(f"quote:{sym}")
        return {"sym": sym, "price": 10, "retrieved_at": "quote timestamp"}

    def candles(sym: str) -> dict[str, object]:
        calls.append(f"candles:{sym}")
        return {"candles": ["bad", {"close": "2"}, {"close": True}, {"close": 3}, {"close": 2.5}]}

    def news(sym: str) -> dict[str, object]:
        calls.append(f"news:{sym}")
        return {"items": None}

    def filings(sym: str) -> dict[str, object]:
        calls.append(f"filings:{sym}")
        return {"items": "bad", "cik": 1341439}

    monkeypatch.setattr(mp, "get_quote", quote)
    monkeypatch.setattr(mp, "get_candles", candles)
    monkeypatch.setattr(mp, "get_news", news)
    monkeypatch.setattr(mp, "get_filings", filings)
    monkeypatch.setattr(mp.time, "gmtime", lambda: mp.time.struct_time((2026, 10, 6, 11, 30, 0, 1, 279, 0)))
    out = mp.get_viewer_context(" nvda ")
    assert calls == ["quote:NVDA", "candles:NVDA", "news:NVDA", "filings:NVDA"]
    assert out["as_of"] == "2026-10-06T11:30:00Z"
    assert out["quote"] == {"sym": "NVDA", "price": 10, "retrieved_at": "quote timestamp"}
    assert out["window"] == {"bars": 3, "last": 2.5, "first": 1.0, "high": 3.0, "low": 1.0, "change_pct": 150.0}
    assert out["news"] == []
    assert out["filings"] == []
    assert out["cik"] == 1341439


# ---------------------------------------------------------------------------
# Aggregates: sparklines, markets, scans, briefing, feed, positioning.
# Real cache.db under a tmp STOCKBOT_DATA_DIR; every provider is faked.
# ---------------------------------------------------------------------------


def _bars(closes: list[float], t0: int = 1767225600) -> dict[str, object]:
    return {"candles": [{"t": t0 + i * 60, "close": c, "vol": 100 * (i + 1)} for i, c in enumerate(closes)]}


def _filing(sym: str, acc: str, form: str, filed: str) -> Filing:
    return Filing(
        accession_no=acc,
        form=form,
        filer_cik=1,
        filer_name=f"{sym} Corp",
        filed_at=filed,
        accepted_at=None,
        known_at=f"{filed}T12:00:00+00:00",
        report_period=None,
        primary_document="doc.xml",
        is_amendment=False,
        amendment_of=None,
        source=f"https://sec.example/{acc}",
    )


class Sec:
    """list_sec_filings + ownership seams keyed by (sym, first form) and accession."""

    def __init__(self) -> None:
        self.filings: dict[tuple[str, str], list[Filing]] = {}
        self.txns: dict[str, list[tuple[str, int | None]]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail: set[str] = set()

    def list_sec_filings(self, sym: str, forms: list[str], **_kw: object) -> list[Filing]:
        self.calls.append((sym, forms[0]))
        if sym in self.fail:
            raise RuntimeError("edgar down")
        return self.filings.get((sym, forms[0]), [])

    def normalize(self, obj: object, **kw: object) -> list[InsiderTransaction]:
        return [
            InsiderTransaction(
                insider_name="Jane Doe",
                insider_cik=None,
                issuer=str(kw["issuer"]),
                form=str(kw["form"]),
                filed_at=str(kw["filed_at"]),
                accession_no=str(kw["accession_no"]),
                transaction_kind=kind,
                shares=shares,
                known_at=str(kw["known_at"]),
            )
            for kind, shares in self.txns.get(str(obj), [])
        ]


@pytest.fixture
def agg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    import app.finra_client as finra
    import app.sec.filings as sec_filings
    from app.analytics import screens
    from app.sec import insider

    monkeypatch.setenv("STOCKBOT_DATA_DIR", str(tmp_path))
    closes = {"AAA": [10.0, 11.0], "BBB": [10.0, 9.0], "CCC": [10.0, 10.5], "NVDA": [100.0, 102.0]}
    candle_calls: list[tuple[str, str, str]] = []
    news_calls: list[str] = []
    sec = Sec()
    state: dict[str, object] = {
        "board": {
            "settlement_date": "2026-01-15",
            "entries": [
                {"ticker": "AAA", "short_interest_percent": 30.0},
                {"ticker": "BBB", "short_interest_percent": 20.0},
                {"ticker": "CCC", "short_interest_percent": 10.0},
            ],
        },
        "threshold": {"error": "No data found for X: thresholdList"},
        "candle_calls": candle_calls,
        "news_calls": news_calls,
        "sec": sec,
        "root": tmp_path,
    }

    def candles(sym: str, range_: str = "6mo", interval: str = "1d") -> dict[str, object]:
        candle_calls.append((sym, range_, interval))
        if sym not in closes:
            return {"error": f"candles unavailable for {sym}: no data"}
        return _bars(closes[sym])

    def news(sym: str, limit: int = 12) -> dict[str, object]:
        news_calls.append(sym)
        item = {
            "title": f"{sym} headline",
            "link": f"https://n.example/{sym}",
            "published": "Thu, 01 Jan 2026 10:00:00 GMT",
            "source": "Wire",
        }
        return {"sym": sym, "items": [item][:limit]}

    def filings(sym: str, limit: int = 10) -> dict[str, object]:
        return {"sym": sym, "items": [{"title": "8-K - Current report", "filed": "2026-01-02"}][:limit]}

    def threshold(sym: str) -> object:
        return state["threshold"]

    monkeypatch.setattr(mp, "get_candles", candles)
    monkeypatch.setattr(mp, "get_news", news)
    monkeypatch.setattr(mp, "get_filings", filings)
    monkeypatch.setattr(mp, "get_universe", lambda: {"count": 0, "tickers": []})
    monkeypatch.setattr(screens, "get_short_interest_leaderboard", lambda limit=None: state["board"])
    monkeypatch.setattr(finra, "get_threshold_securities", threshold)
    monkeypatch.setattr(sec_filings, "list_sec_filings", sec.list_sec_filings)
    monkeypatch.setattr(insider, "load_ownership", str)
    monkeypatch.setattr(insider, "normalize_ownership_filing", sec.normalize)
    return state


def _sec(agg: dict[str, object]) -> Sec:
    sec = agg["sec"]
    assert isinstance(sec, Sec)
    return sec


def _root(agg: dict[str, object]) -> Path:
    root = agg["root"]
    assert isinstance(root, Path)
    return root


def test_sparklines_shape_per_symbol_error_and_60s_cache(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(app)
    body = client.get("/api/sparklines?syms=nvda,BAD,NVDA&range=1d").json()
    assert list(body["syms"]) == ["NVDA", "BAD"]
    assert body["syms"]["NVDA"] == {
        "t": [1767225600, 1767225660],
        "closes": [100.0, 102.0],
        "as_of": "2026-01-01T00:01:00+00:00",
    }
    assert body["syms"]["BAD"] == {"error": "candles unavailable for BAD: no data"}
    calls = agg["candle_calls"]
    assert isinstance(calls, list)
    assert sorted(calls) == [("BAD", "1d", "5m"), ("NVDA", "1d", "5m")]
    assert client.get("/api/sparklines?syms=NVDA,BAD&range=1d").json() == body
    assert len(calls) == 2
    real = real_cache.time.time()
    monkeypatch.setattr(real_cache.time, "time", lambda: real + 61)
    client.get("/api/sparklines?syms=NVDA,BAD&range=1d")
    assert len(calls) == 4


def test_sparklines_rejects_more_than_20_symbols(agg: dict[str, object]) -> None:
    syms = ",".join(f"S{i}" for i in range(21))
    res = TestClient(app).get(f"/api/sparklines?syms={syms}")
    assert res.status_code == 400
    assert "at most 20" in res.json()["detail"]


def test_markets_most_shorted_rows_and_sorts(agg: dict[str, object]) -> None:
    client = TestClient(app)
    by_short = client.get("/api/markets?scan=most-shorted&sort=short_pct").json()
    assert by_short["universe"] == ["AAA", "BBB", "CCC"]
    assert [r["sym"] for r in by_short["rows"]] == ["AAA", "BBB", "CCC"]
    assert by_short["rows"][0] == {
        "sym": "AAA",
        "last": 11.0,
        "chg_pct": pytest.approx(10.0),
        "volume": 200,
        "short_pct": 30.0,
        "last_filing": {"form": "8-K", "filed": "2026-01-02"},
    }
    by_chg = client.get("/api/markets?scan=most-shorted").json()
    assert [r["sym"] for r in by_chg["rows"]] == ["AAA", "CCC", "BBB"]
    calls = agg["candle_calls"]
    assert isinstance(calls, list) and ("AAA", "5d", "1d") in calls


def test_markets_watchlist_bad_symbol_sorts_last_with_error(agg: dict[str, object]) -> None:
    rows = TestClient(app).get("/api/markets?scan=watchlist&syms=BAD,NVDA&sort=volume").json()["rows"]
    assert [r["sym"] for r in rows] == ["NVDA", "BAD"]
    assert rows[0]["short_pct"] is None
    assert rows[1]["last"] is None
    assert rows[1]["error"].startswith("candles unavailable")


def test_markets_movers_adds_index_universe(agg: dict[str, object]) -> None:
    body = TestClient(app).get("/api/markets?scan=movers&syms=NVDA,SPY").json()
    assert body["universe"] == ["NVDA", "SPY", "QQQ", "DIA", "IWM"]


def test_markets_rejects_unknown_scan_and_sort(agg: dict[str, object]) -> None:
    client = TestClient(app)
    assert client.get("/api/markets?scan=nope").status_code == 400
    assert client.get("/api/markets?scan=watchlist&sort=price").status_code == 400


def test_markets_insider_buying_and_new_8k_filter_the_universe(agg: dict[str, object]) -> None:
    sec = _sec(agg)
    sec.filings[("NVDA", "4")] = [_filing("NVDA", "a-1", "4", "2026-01-03")]
    sec.txns["a-1"] = [("open_market_purchase", 500)]
    sec.filings[("BBB", "4")] = [_filing("BBB", "b-1", "4", "2026-01-03")]
    sec.txns["b-1"] = [("open_market_sale", 900)]
    sec.filings[("CCC", "8-K")] = [_filing("CCC", "c-1", "8-K", "2026-01-04")]
    client = TestClient(app)
    assert client.get("/api/markets?scan=insider-buying&syms=NVDA").json()["universe"] == ["NVDA"]
    assert {s for s, _f in sec.calls} == {"NVDA", "AAA", "BBB", "CCC"}
    assert client.get("/api/markets?scan=new-8k&syms=NVDA").json()["universe"] == ["CCC"]


def test_markets_scan_universe_is_capped(agg: dict[str, object]) -> None:
    syms = ",".join(f"S{i}" for i in range(20))
    TestClient(app).get(f"/api/markets?scan=new-8k&syms={syms}")
    probed = [s for s, _f in _sec(agg).calls]
    assert len(probed) == 20
    assert "AAA" not in probed


def test_markets_short_board_error_reads_as_null_short_pct(agg: dict[str, object]) -> None:
    agg["board"] = {"error": "Short-interest leaderboard is unavailable: no FINRA credentials"}
    body = TestClient(app).get("/api/markets?scan=watchlist&syms=NVDA").json()
    assert body["rows"][0]["short_pct"] is None
    assert body["errors"] == [
        {
            "sym": "",
            "source": "finra_short_interest",
            "error": "Short-interest leaderboard is unavailable: no FINRA credentials",
        }
    ]


def test_scans_crud_round_trip(agg: dict[str, object]) -> None:
    client = TestClient(app)
    presets = client.get("/api/scans").json()["scans"]
    assert [s["id"] for s in presets] == ["watchlist", "most-shorted", "insider-buying", "new-8k", "movers"]
    assert {s["kind"] for s in presets} == {"preset"}
    res = client.post("/api/scans", json={"name": " Chips ", "filter": {"symbols": [" nvda", "amd"], "min_chg_pct": 2}})
    assert res.status_code == 201
    scan = res.json()
    assert scan["name"] == "Chips"
    assert scan["kind"] == "custom"
    assert scan["filter"] == {"symbols": ["NVDA", "AMD"], "min_chg_pct": 2.0}
    assert json.loads((_root(agg) / "scans" / f"{scan['id']}.json").read_text()) == scan
    assert client.get("/api/scans").json()["scans"][-1] == scan
    assert client.delete(f"/api/scans/{scan['id']}").json() == {"deleted": scan["id"]}
    assert len(client.get("/api/scans").json()["scans"]) == 5
    assert client.delete(f"/api/scans/{scan['id']}").status_code == 404
    assert client.delete("/api/scans/most-shorted").status_code == 400
    assert client.delete("/api/scans/scan-zz").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x", "filter": {"price_above": 5}},
        {"name": "x", "extra": 1},
        {"name": "x", "kind": "preset"},
        {"name": "   "},
        {"name": "x", "filter": {"symbols": [1]}},
    ],
)
def test_scans_reject_invalid_payloads(agg: dict[str, object], payload: dict[str, object]) -> None:
    res = TestClient(app).post("/api/scans", json=payload)
    assert res.status_code == 400
    assert res.json()["detail"].startswith("invalid scan:")
    assert not (_root(agg) / "scans").exists()


def test_workspace_never_saved_reads_empty(agg: dict[str, object]) -> None:
    res = TestClient(app).get("/api/workspaces/nvda")
    assert res.status_code == 200
    assert res.json() == {"sym": "NVDA", "widgets": [], "updated_at": None}
    assert not (_root(agg) / "workspaces").exists()


def test_workspace_put_round_trip(agg: dict[str, object]) -> None:
    client = TestClient(app)
    tv = {"url": "https://www.tradingview.com/x/8ot166mV/", "label": "NVDA snapshot"}
    widgets = [
        {
            "id": "w1",
            "type": "price-volume",
            "ticker": "nvda",
            "params": {"interval": "1h", "style": "line", "studies": ["MA50"]},
        },
        {"id": "w2", "type": "short-volume", "ticker": "NVDA"},
        {"id": "tv-link-1", "type": "tv-link", "ticker": "NVDA", "params": tv},
        {
            "id": "tv-link-2",
            "type": "tv-link",
            "ticker": "NVDA",
            "params": {**tv, "url": "https://tradingview.com/chart/?symbol=NASDAQ%3ANVDA"},
        },
    ]
    res = client.put("/api/workspaces/nvda", json={"widgets": widgets})
    assert res.status_code == 200
    body = res.json()
    assert body["sym"] == "NVDA"
    assert body["widgets"] == [
        {
            "id": "w1",
            "type": "price-volume",
            "ticker": "NVDA",
            "params": {"interval": "1h", "style": "line", "studies": ["MA50"]},
        },
        {"id": "w2", "type": "short-volume", "ticker": "NVDA"},
        {"id": "tv-link-1", "type": "tv-link", "ticker": "NVDA", "params": tv},
        {
            "id": "tv-link-2",
            "type": "tv-link",
            "ticker": "NVDA",
            "params": {**tv, "url": "https://tradingview.com/chart/?symbol=NASDAQ%3ANVDA"},
        },
    ]
    assert isinstance(body["updated_at"], str) and body["updated_at"].endswith("+00:00")
    assert json.loads((_root(agg) / "workspaces" / "NVDA.json").read_text()) == body
    assert client.get("/api/workspaces/NVDA").json() == body
    assert client.put("/api/workspaces/NVDA", json={"widgets": []}).json()["widgets"] == []
    assert client.get("/api/workspaces/nvda").json()["widgets"] == []


_TV = "https://www.tradingview.com/x/8ot166mV/"


@pytest.mark.parametrize(
    "params",
    [
        {"url": _TV},
        {"label": "x"},
        {"url": _TV, "label": "x", "extra": 1},
        {"url": _TV, "label": ""},
        {"url": _TV, "label": "x" * 61},
        {"url": _TV + "a" * (301 - len(_TV)), "label": "x"},
        {"url": 1, "label": "x"},
        {"url": "http://www.tradingview.com/x/8ot166mV/", "label": "x"},
        {"url": "javascript:alert(1)", "label": "x"},
        {"url": "https://evil.example/x/8ot166mV/", "label": "x"},
        {"url": "https://tradingview.com.evil.example/x/", "label": "x"},
        {"url": "https://user@www.tradingview.com/x/8ot166mV/", "label": "x"},
        {"url": "https://www.tradingview.com:8443/x/8ot166mV/", "label": "x"},
        {"url": "https://www.tradingview.com:bad/", "label": "x"},
    ],
)
def test_workspace_put_rejects_invalid_tv_link(agg: dict[str, object], params: dict[str, object]) -> None:
    widget = {"id": "tv-link-1", "type": "tv-link", "ticker": "NVDA", "params": params}
    res = TestClient(app).put("/api/workspaces/NVDA", json={"widgets": [widget]})
    assert res.status_code == 400
    assert res.json()["detail"].startswith("invalid workspace:")
    assert not (_root(agg) / "workspaces").exists()


@pytest.mark.parametrize(
    "params",
    [
        {"range": "6mo"},
        {"interval": "2h"},
        {"interval": 5},
        {"style": "bars"},
        {"studies": "MA20"},
        {"studies": ["MA20", "MA20"]},
        {"studies": ["Ichimoku"]},
        {"studies": [{"id": "RSI"}]},
    ],
)
def test_workspace_put_rejects_invalid_chart_params(agg: dict[str, object], params: dict[str, object]) -> None:
    widget = {"id": "price-volume", "type": "price-volume", "ticker": "NVDA", "params": params}
    res = TestClient(app).put("/api/workspaces/NVDA", json={"widgets": [widget]})
    assert res.status_code == 400
    assert res.json()["detail"].startswith("invalid workspace:")
    assert not (_root(agg) / "workspaces").exists()


def test_widget_types_match_harness_list() -> None:
    from app.tools import WIDGET_TYPES

    source = (Path(__file__).parents[1] / "needle-harness/lib/agent/types.ts").read_text(encoding="utf-8")
    match = re.search(r"export const WIDGET_TYPES: readonly string\[\] = (\[[^\]]*\]);", source)
    assert match is not None
    assert tuple(json.loads(match.group(1))) == WIDGET_TYPES


@pytest.mark.parametrize(
    ("sym", "payload"),
    [
        ("NVDA", {"widgets": [{"id": "w1", "type": "candles", "ticker": "NVDA"}]}),
        ("NVDA", {"widgets": [{"id": "w1", "type": "news", "ticker": "NVDA", "size": 2}]}),
        ("NVDA", {"widgets": [], "layout": "grid"}),
        ("NVDA", {"widgets": [{"id": f"w{i}", "type": "news", "ticker": "NVDA"} for i in range(25)]}),
        ("NVDA", {"widgets": [{"id": "", "type": "news", "ticker": "NVDA"}]}),
        ("NVDA", {"widgets": [{"id": "w1", "type": "news", "ticker": "A/B"}]}),
        ("NVDA", {"widgets": [{"id": "w1", "type": "news", "ticker": "NVDA", "params": None}]}),
        ("NVDA", {"widgets": [{"id": "w" * 65, "type": "news", "ticker": "NVDA"}]}),
        ("NVDA", []),
        ("NVDA", {}),
        ("NVDA", None),
        ("A/B", {"widgets": []}),
    ],
)
def test_workspace_put_rejects_invalid(agg: dict[str, object], sym: str, payload: object) -> None:
    res = TestClient(app).put(f"/api/workspaces/{sym}", json=payload)
    assert res.status_code == 400
    assert res.json()["detail"].startswith("invalid workspace:")
    assert not (_root(agg) / "workspaces").exists()


def test_widget_data_rejects_unknown_type_and_bad_sym(agg: dict[str, object]) -> None:
    client = TestClient(app)
    assert client.get("/api/widget-data?type=news&sym=NVDA").status_code == 400
    assert client.get("/api/widget-data?sym=NVDA").status_code == 400
    assert client.get("/api/widget-data?type=insider&sym=A/B").status_code == 400


def _finra(monkeypatch: pytest.MonkeyPatch, pages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fake get_finra_datapoints: one canned answer per call, in order; records every call's kwargs."""
    import app.finra_client as finra

    calls: list[dict[str, object]] = []

    def fake(dataset: str, **kw: object) -> dict[str, object]:
        calls.append({"dataset": dataset, **kw})
        return pages[len(calls) - 1]

    monkeypatch.setattr(finra, "get_finra_datapoints", fake)
    return calls


def test_widget_data_short_volume_sums_market_codes_per_date(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def row(day: str, short: int, total: int) -> dict[str, object]:
        return {"tradeReportDate": day, "shortParQuantity": short, "totalParQuantity": total}

    no_data: dict[str, object] = {"error": "No data found for NVDA: regShoDaily"}
    calls = _finra(
        monkeypatch,
        [
            {"records": [row("2026-10-19", 300, 1000), row("2026-10-16", 50, 100), row("2026-10-19", 200, 1000)]},
            no_data,
            {"records": [row("2026-10-09", 0, 0)]},
            no_data,
        ],
    )
    body = TestClient(app).get("/api/widget-data?type=short-volume&sym=nvda").json()
    assert body == {
        "sym": "NVDA",
        "type": "short-volume",
        "rows": [
            {"date": "2026-10-09", "short_qty": 0.0, "total_qty": 0.0, "short_pct": None},
            {"date": "2026-10-16", "short_qty": 50.0, "total_qty": 100.0, "short_pct": 50.0},
            {"date": "2026-10-19", "short_qty": 500.0, "total_qty": 2000.0, "short_pct": 25.0},
        ],
    }
    assert len(calls) == 4
    assert {c["dataset"] for c in calls} == {"otcMarket/regShoDaily"}
    assert all(c["ticker"] == "NVDA" and c["limit"] == 25 for c in calls)
    assert calls[0]["fields"] == ["tradeReportDate", "shortParQuantity", "totalParQuantity"]
    assert [str(c["end_date"]) > str(c["start_date"]) for c in calls] == [True] * 4
    assert str(calls[1]["end_date"]) < str(calls[0]["start_date"])
    newest = date.fromisoformat(str(calls[0]["end_date"]))
    oldest = date.fromisoformat(str(calls[-1]["start_date"]))
    assert newest - oldest == timedelta(days=13)
    for previous, current in pairwise(calls):
        assert date.fromisoformat(str(previous["start_date"])) - date.fromisoformat(
            str(current["end_date"])
        ) == timedelta(days=1)


def test_widget_data_short_interest_maps_records_oldest_first(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _finra(
        monkeypatch,
        [
            {
                "records": [
                    {"settlementDate": "2026-09-30", "currentShortPositionQuantity": 900, "daysToCoverQuantity": 1.5},
                    {"settlementDate": "2026-09-15", "currentShortPositionQuantity": 800, "daysToCoverQuantity": None},
                ]
            }
        ],
    )
    body = TestClient(app).get("/api/widget-data?type=short-interest&sym=NVDA").json()
    assert body == {
        "sym": "NVDA",
        "type": "short-interest",
        "rows": [
            {"date": "2026-09-15", "short_qty": 800, "days_to_cover": None},
            {"date": "2026-09-30", "short_qty": 900, "days_to_cover": 1.5},
        ],
    }
    (call,) = calls
    assert call["dataset"] == "otcMarket/consolidatedShortInterest"
    assert call["fields"] == ["settlementDate", "currentShortPositionQuantity", "daysToCoverQuantity"]
    assert call["limit"] == 25 and call["ticker"] == "NVDA"


@pytest.mark.parametrize("kind", ["short-interest", "short-volume"])
def test_widget_data_finra_errors_surface_without_raising(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    _finra(monkeypatch, [{"error": "FINRA request failed (503): down"}] * 4)
    res = TestClient(app).get(f"/api/widget-data?type={kind}&sym=NVDA")
    assert res.status_code == 200
    assert res.json() == {"sym": "NVDA", "type": kind, "rows": [], "error": "FINRA request failed (503): down"}


def test_widget_data_finra_exception_becomes_error(agg: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    import app.finra_client as finra

    def boom(dataset: str, **kw: object) -> dict[str, object]:
        raise RuntimeError("socket closed")

    monkeypatch.setattr(finra, "get_finra_datapoints", boom)
    res = TestClient(app).get("/api/widget-data?type=short-interest&sym=NVDA")
    assert res.status_code == 200
    assert res.json() == {
        "sym": "NVDA",
        "type": "short-interest",
        "error": "short-interest unavailable for NVDA: socket closed",
    }


def test_widget_data_short_volume_all_windows_empty_reports_no_data(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _finra(monkeypatch, [{"error": "No data found for NVDA: regShoDaily"}] * 4)
    body = TestClient(app).get("/api/widget-data?type=short-volume&sym=NVDA").json()
    assert body["rows"] == [] and body["error"] == "No data found for NVDA: regShoDaily"


def test_widget_data_short_volume_rejects_truncated_window(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _finra(monkeypatch, [{"records": [], "may_have_more": True, "next_offset": 25}] * 4)
    body = TestClient(app).get("/api/widget-data?type=short-volume&sym=NVDA").json()
    assert body == {
        "sym": "NVDA",
        "type": "short-volume",
        "rows": [],
        "error": "Short-volume data is incomplete: a date window exceeds the 25-record limit.",
    }
    assert len(calls) == 4


@pytest.mark.parametrize(
    "failed",
    [
        {"error": "FINRA request failed (503): down"},
        {"records": [], "may_have_more": True},
    ],
)
def test_widget_data_short_volume_retains_error_with_partial_rows(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch, failed: dict[str, object]
) -> None:
    good: dict[str, object] = {
        "records": [{"tradeReportDate": "2026-10-19", "shortParQuantity": 25, "totalParQuantity": 100}]
    }
    calls = _finra(monkeypatch, [good, failed, {"records": []}, {"records": []}])
    body = TestClient(app).get("/api/widget-data?type=short-volume&sym=NVDA").json()
    assert body["rows"] == [{"date": "2026-10-19", "short_qty": 25, "total_qty": 100, "short_pct": 25}]
    assert body["error"] == failed.get(
        "error", "Short-volume data is incomplete: a date window exceeds the 25-record limit."
    )
    assert len(calls) == 4


def test_widget_data_cache_uses_canonical_symbol_and_900s_ttl(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _finra(monkeypatch, [{"records": []}, {"records": []}])
    client = TestClient(app)
    first = client.get("/api/widget-data?type=short-interest&sym=nvda").json()
    assert real_cache.get("widget:short-interest:NVDA", ttl=900) == first
    assert client.get("/api/widget-data?type=short-interest&sym=NVDA").json() == first
    assert len(calls) == 1
    now = real_cache.time.time()
    monkeypatch.setattr(real_cache.time, "time", lambda: now + 901)
    assert client.get("/api/widget-data?type=short-interest&sym=NVDA").json() == first
    assert len(calls) == 2


def test_widget_data_insider_rows_newest_first(agg: dict[str, object]) -> None:
    sec = _sec(agg)
    sec.filings[("NVDA", "4")] = [
        _filing("NVDA", "i-1", "4", "2026-10-01"),
        _filing("NVDA", "i-2", "4", "2026-10-08"),
    ]
    sec.txns["i-1"] = [("open_market_purchase", 1500), ("grant_award", 10)]
    sec.txns["i-2"] = [("open_market_sale", None)]
    body = TestClient(app).get("/api/widget-data?type=insider&sym=NVDA").json()
    assert body == {
        "sym": "NVDA",
        "type": "insider",
        "rows": [
            {"date": "2026-10-08", "title": "Jane Doe sold", "kind": "open_market_sale", "shares": None},
            {"date": "2026-10-01", "title": "Jane Doe bought 1,500 sh", "kind": "open_market_purchase", "shares": 1500},
        ],
    }
    sec.fail.add("BAD")
    failed = TestClient(app).get("/api/widget-data?type=insider&sym=BAD").json()
    assert failed["rows"] == [] and "edgar down" in failed["error"]


@pytest.mark.parametrize(
    ("price", "eps", "pe", "ey", "error"),
    [
        (100.0, {"ttm_eps_diluted": 4.0}, 25.0, 0.04, None),
        (100.0, {"ttm_eps_diluted": 0}, None, 0.0, None),
        (0.0, {"ttm_eps_diluted": 4.0}, None, None, None),
        (100.0, {"ttm_eps_diluted": -4.0}, -25.0, -0.04, None),
        (100.0, {"error": "company facts not available"}, None, None, "company facts not available"),
        (None, {"ttm_eps_diluted": 4.0}, None, None, None),
    ],
)
def test_widget_data_multiples_current_only(
    agg: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    price: float | None,
    eps: dict[str, object],
    pe: float | None,
    ey: float | None,
    error: str | None,
) -> None:
    seen: list[tuple[str, str]] = []

    def fundamentals(sym: str, metric: str, as_of: str | None = None) -> dict[str, object]:
        seen.append((sym, metric))
        return eps

    def quote(sym: str) -> dict[str, object]:
        return {"sym": sym, "price": price, "retrieved_at": "2026-10-20T14:00:00Z"}

    monkeypatch.setattr(mp, "get_quote", quote)
    monkeypatch.setattr("app.services.sec_facts.get_fundamentals", fundamentals)
    body = TestClient(app).get("/api/widget-data?type=multiples&sym=nvda").json()
    expected: dict[str, object] = {
        "sym": "NVDA",
        "type": "multiples",
        "price": price if isinstance(price, (int, float)) and price > 0 else None,
        "ttm_eps": eps.get("ttm_eps_diluted"),
        "pe": pe,
        "earnings_yield": ey,
        "as_of": "2026-10-20T14:00:00Z",
        "history": None,
        "derived": True,
        "calculations": {
            "pe": {
                "derived": True,
                "formulaId": "pe",
                "formulaVersion": "1",
                "value": pe,
                "unit": "ratio",
                "verification": "unverified",
            },
            "earnings_yield": {
                "derived": True,
                "formulaId": "earnings_yield",
                "formulaVersion": "1",
                "value": ey,
                "unit": "ratio",
                "verification": "unverified",
            },
        },
    }
    if error:
        expected["error"] = error
    for field in ("pe", "earnings_yield"):
        marker = body["calculations"][field]
        assert isinstance(marker, dict)
        assert [row["value"] for row in marker["derivation_inputs"]] == (
            [body["price"], body["ttm_eps"]] if field == "pe" else [body["ttm_eps"], body["price"]]
        )
        body["calculations"][field] = {
            key: marker[key] for key in ("derived", "formulaId", "formulaVersion", "value", "unit", "verification")
        }
    assert body == expected
    assert seen == [("NVDA", "eps")]


def test_briefing_builds_reads_cache_and_refreshes(agg: dict[str, object]) -> None:
    sec = _sec(agg)
    sec.filings[("NVDA", "8-K")] = [_filing("NVDA", "k-1", "8-K", "2026-01-05")]
    sec.filings[("NVDA", "4")] = [_filing("NVDA", "f-1", "4", "2026-01-04")]
    sec.txns["f-1"] = [("open_market_sale", 1200), ("grant_award", 50)]
    client = TestClient(app)
    first = client.get("/api/briefing?syms=NVDA").json()
    assert first["cached"] is False
    assert first["syms"] == ["NVDA"]
    assert first["items"] == [
        {
            "kind": "filing",
            "sym": "NVDA",
            "title": "8-K filed by NVDA",
            "link": "https://sec.example/k-1",
            "filed": "2026-01-05",
            "filing_form": "8-K",
        },
        {
            "kind": "insider",
            "sym": "NVDA",
            "title": "Jane Doe sold 1,200 sh",
            "link": "https://sec.example/f-1",
            "filed": "2026-01-04",
            "filing_form": "4",
        },
        {
            "kind": "news",
            "sym": "NVDA",
            "title": "NVDA headline",
            "link": "https://n.example/NVDA",
            "published": "2026-01-01T10:00:00+00:00",
        },
    ]
    assert (_root(agg) / "briefing" / f"{first['date']}.json").is_file()
    news_calls = agg["news_calls"]
    assert isinstance(news_calls, list)
    assert client.get("/api/briefing?syms=NVDA").json() == dict(first, cached=True)
    assert len(news_calls) == 1
    assert client.get("/api/briefing?syms=NVDA&refresh=1").json()["cached"] is False
    assert client.get("/api/briefing?syms=AAA").json()["syms"] == ["AAA"]
    assert len(news_calls) == 3


def test_feed_merges_newest_first_and_filters_since(agg: dict[str, object]) -> None:
    from app.thesis.repository import ThesisRepository

    repo = ThesisRepository(_root(agg) / "thesis")
    thesis = repo.create_thesis(
        "NVDA thesis", scope="NVDA", claims=["NVDA demand grows"], effective_at="2026-01-01T00:00:00+00:00"
    )
    cid = repo.load_thesis(thesis.thesis_id).claims[0].claim_id
    trigger = repo.create_trigger(thesis.thesis_id, claim_ids=[cid], summary="New 10-K", summary_origin="deterministic")
    sec = _sec(agg)
    sec.filings[("NVDA", "8-K")] = [_filing("NVDA", "k-1", "8-K", "2026-01-05")]
    sec.filings[("NVDA", "4")] = [_filing("NVDA", "f-1", "4", "2026-01-04")]
    sec.txns["f-1"] = [("open_market_purchase", 10)]
    sec.fail.add("BAD")
    client = TestClient(app)
    body = client.get("/api/feed?syms=NVDA,BAD").json()
    assert [i["kind"] for i in body["items"]] == ["alert", "filing", "insider", "news", "news"]
    assert body["items"][0]["id"] == trigger.trigger_id
    assert body["items"][0]["title"] == "New 10-K"
    assert body["items"][2] == {
        "id": "insider:f-1:0",
        "kind": "insider",
        "sym": "NVDA",
        "title": "Jane Doe bought 10 sh",
        "link": "https://sec.example/f-1",
        "at": "2026-01-04T12:00:00+00:00",
        "source": "SEC Form 4",
    }
    assert {(e["sym"], e["source"]) for e in body["errors"]} == {("BAD", "sec_8k"), ("BAD", "sec_form4")}
    since = client.get("/api/feed?syms=NVDA&since=2026-01-04T12:00:00Z").json()["items"]
    assert [i["kind"] for i in since] == ["alert", "filing"]
    repo.mark_trigger_processed(thesis.thesis_id, trigger.trigger_id)
    assert [i["kind"] for i in client.get("/api/feed?syms=NVDA").json()["items"]] == ["filing", "insider", "news"]
    assert client.get("/api/feed?since=yesterday").status_code == 400


def test_positioning_sums_insider_shares_and_nulls_unavailable(agg: dict[str, object]) -> None:
    sec = _sec(agg)
    sec.filings[("AAA", "4")] = [_filing("AAA", "p-1", "4", "2026-01-04")]
    sec.txns["p-1"] = [
        ("open_market_purchase", 300),
        ("open_market_sale", 100),
        ("open_market_sale", None),
        ("exercise", 50),
    ]
    client = TestClient(app)
    assert client.get("/api/positioning?sym=aaa").json() == {
        "sym": "AAA",
        "short_pct": 30.0,
        "short_as_of": "2026-01-15",
        "insider_net_90d": {"buys": 300, "sells": 100, "net": 200},
        "threshold_listed": False,
        "threshold_as_of": None,
        "errors": {},
    }
    agg["threshold"] = {"error": "FINRA query failed: 503"}
    sec.fail.add("ZZZ")
    out = client.get("/api/positioning?sym=ZZZ").json()
    assert out["short_pct"] is None
    assert out["short_as_of"] is None
    assert out["insider_net_90d"] is None
    assert out["threshold_listed"] is None
    assert set(out["errors"]) == {"insider", "threshold"}
    agg["threshold"] = {"records": [{"tradeDate": "2026-01-14"}], "as_of_date": "2026-01-14"}
    listed = client.get("/api/positioning?sym=CCC").json()
    assert listed["threshold_listed"] is True
    assert listed["threshold_as_of"] == "2026-01-14"


def test_regweb_news_feed_rows_are_headline_only_never_article_body(
    fake_cache: FakeCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RSS news/feed/briefing rows carry headline fields only; no body is fetched or implied."""
    feed = (
        b'<?xml version="1.0"?><rss version="2.0"><channel>'
        b"<item><title>NVDA rallies</title><link>https://n.example/nvda</link>"
        b"<pubDate>Tue, 06 Oct 2026 10:00:00 GMT</pubDate><source>Wire</source></item>"
        b"</channel></rss>"
    )

    def _feed(url: str, **kwargs: object) -> bytes:
        return feed

    monkeypatch.setattr(mp, "_http_bytes", _feed)
    monkeypatch.setenv("STOCKBOT_NEWSQUANT", "0")
    out = mp.get_news("NVDA")
    items = out["items"]
    assert isinstance(items, list) and len(items) == 1
    first = items[0]
    assert isinstance(first, dict) and first.get("link") == "https://n.example/nvda"
    assert set(first) == {"title", "link", "published", "source"}
    assert "body" not in first and "content" not in first and "excerpt" not in first


# ---------------------------------------------------------------------------
# Change 4 regression: FINRA multi-record aggregation, incomplete coverage
# ---------------------------------------------------------------------------
def test_reg4_short_volume_sums_all_venue_records_not_first(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two same-date venue records sum; a first-row-only read would undercount."""

    def row(short: int, total: int) -> dict[str, object]:
        return {"tradeReportDate": "2026-10-19", "shortParQuantity": short, "totalParQuantity": total}

    calls = _finra(
        monkeypatch,
        [
            {"records": [row(300, 1000), row(200, 1000)]},
            {"records": []},
            {"records": []},
            {"records": []},
        ],
    )
    body = TestClient(app).get("/api/widget-data?type=short-volume&sym=NVDA").json()
    assert body["rows"] == [{"date": "2026-10-19", "short_qty": 500, "total_qty": 2000, "short_pct": 25}]
    assert len(calls) == 4


def test_reg4_threshold_uses_latest_record_and_incomplete_stays_visible(
    agg: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threshold replays the queried day; an incomplete short-volume window keeps its error."""
    import app.finra_client as finra

    seen: list[dict[str, object]] = []

    def fake_threshold(ticker: str | None = None, trade_date: str | None = None) -> dict[str, object]:
        seen.append({"ticker": ticker, "trade_date": trade_date})
        return {
            "records": [{"issueSymbolIdentifier": "NVDA", "tradeDate": "2026-10-15"}],
            "as_of_date": "2026-10-15",
        }

    monkeypatch.setattr(finra, "get_threshold_securities", fake_threshold)
    body = TestClient(app).get("/api/positioning?sym=NVDA").json()
    assert body["threshold_listed"] is True and body["threshold_as_of"] == "2026-10-15"
    assert seen == [{"ticker": "NVDA", "trade_date": None}]
    calls = _finra(monkeypatch, [{"records": [], "may_have_more": True, "next_offset": 25}] * 4)
    short = TestClient(app).get("/api/widget-data?type=short-volume&sym=NVDA").json()
    assert short["rows"] == [] and "incomplete" in str(short["error"]).lower()
    assert len(calls) == 4
