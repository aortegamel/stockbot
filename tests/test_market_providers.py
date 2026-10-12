"""Unit tests for the keyless market-data providers.

Deterministic and offline: _http_bytes is faked per URL; nothing touches
the network or cache.db.
"""

import json
import sys
import types

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
