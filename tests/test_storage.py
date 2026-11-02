"""Tests for the raw archive, persistent market store, and warehouse-removal gates."""

import importlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.storage import market, raw_archive


@pytest.fixture
def archive_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


def _payload(text: str) -> bytes:
    return json.dumps({"content": text}).encode()


# ---------------------------------------------------------------------------
# Raw archive
# ---------------------------------------------------------------------------


def test_archive_stores_payload_and_manifest(archive_root: Path):
    record = raw_archive.archive(
        "sec",
        "companyfacts",
        "cik0000320193",
        _payload("hello"),
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        retrieved_at="2026-08-21T12:00:00Z",
        metadata={"cik": "0000320193"},
        root=archive_root,
    )
    assert record.payload_path.is_file()
    assert record.manifest_path.is_file()
    assert record.sha256 == raw_archive.content_hash(_payload("hello"))
    assert record.size == len(_payload("hello"))
    assert record.url.endswith("CIK0000320193.json")
    assert record.metadata["cik"] == "0000320193"
    manifest = json.loads(record.manifest_path.read_text())
    assert manifest["sha256"] == record.sha256


def test_archive_is_immutable_and_idempotent(archive_root: Path):
    first = raw_archive.archive("finra", "data", "otc/cycle", _payload("x"), url="u", root=archive_root)
    second = raw_archive.archive("finra", "data", "otc/cycle", _payload("x"), url="u", root=archive_root)
    assert first.payload_path == second.payload_path
    assert first.manifest_path == second.manifest_path
    assert {p for p in archive_root.rglob("*.json")} == {second.payload_path, second.manifest_path}


def test_archive_keeps_distinct_payload_revisions(archive_root: Path):
    first = raw_archive.archive("sec", "companyfacts", "cik1", _payload("v1"), url="u", root=archive_root)
    second = raw_archive.archive("sec", "companyfacts", "cik1", _payload("v2"), url="u", root=archive_root)
    assert first.payload_path != second.payload_path
    revisions = list(raw_archive.iter_archive("sec", "companyfacts", "cik1", root=archive_root))
    assert [r.sha256 for r in revisions] == sorted(r.sha256 for r in revisions)


def test_find_and_has_payload(archive_root: Path):
    record = raw_archive.archive("sec", "companyfacts", "cik1", _payload("v1"), url="u", root=archive_root)
    assert raw_archive.find("sec", "companyfacts", "cik1", root=archive_root) == record
    assert raw_archive.find("sec", "companyfacts", "cik1", sha256=record.sha256, root=archive_root) == record
    assert raw_archive.find("sec", "companyfacts", "cik1", sha256="0" * 64, root=archive_root) is None
    assert raw_archive.find("sec", "companyfacts", "nope", root=archive_root) is None
    assert raw_archive.has_payload("sec", "companyfacts", "cik1", record.sha256, root=archive_root)
    assert not raw_archive.has_payload("sec", "companyfacts", "cik1", "0" * 64, root=archive_root)


def _events(archive_root: Path, key: str = "k") -> list[tuple[str, str, str]]:
    return [(e.sha256, e.retrieved_at, e.url) for e in raw_archive.iter_retrievals("sec", "t", key, root=archive_root)]


def test_repeat_retrieval_keeps_original_record_and_logs_each_time(archive_root: Path):
    first = raw_archive.archive(
        "sec",
        "t",
        "k",
        _payload("a"),
        url="u1",
        retrieved_at="2026-01-01T00:00:00Z",
        record_retrieval=True,
        root=archive_root,
    )
    payload_bytes, manifest_bytes = first.payload_path.read_bytes(), first.manifest_path.read_bytes()
    second = raw_archive.archive(
        "sec",
        "t",
        "k",
        _payload("a"),
        url="u2",
        retrieved_at="2026-01-20T00:00:00Z",
        record_retrieval=True,
        root=archive_root,
    )
    assert second == first
    assert first.payload_path.read_bytes() == payload_bytes
    assert first.manifest_path.read_bytes() == manifest_bytes
    assert list(raw_archive.iter_archive("sec", "t", "k", root=archive_root)) == [first]
    assert _events(archive_root) == [
        (first.sha256, "2026-01-01T00:00:00Z", "u1"),
        (first.sha256, "2026-01-01T00:00:00Z", "u1"),
        (first.sha256, "2026-01-20T00:00:00Z", "u2"),
    ]


def test_dedup_hit_logs_invocation_time_not_manifest_time(archive_root: Path, monkeypatch: pytest.MonkeyPatch):
    stamps = iter(["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"])
    monkeypatch.setattr(raw_archive, "_utc_now", lambda: next(stamps))
    first = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    assert first.retrieved_at == "2026-01-01T00:00:00Z"
    assert [stamp for _, stamp, _ in _events(archive_root)] == [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
    ]


def test_a_b_a_retrievals_keep_every_hash_in_order(archive_root: Path):
    a, b = _payload("a"), _payload("b")
    for payload, stamp in ((a, "2026-01-01T00:00:00Z"), (b, "2026-01-02T00:00:00Z"), (a, "2026-01-03T00:00:00Z")):
        raw_archive.archive(
            "sec", "t", "k", payload, url="u", retrieved_at=stamp, record_retrieval=True, root=archive_root
        )
    sha_a, sha_b = raw_archive.content_hash(a), raw_archive.content_hash(b)
    assert {r.sha256 for r in raw_archive.iter_archive("sec", "t", "k", root=archive_root)} == {sha_a, sha_b}
    logged = _events(archive_root)[2:]
    assert [(sha, stamp) for sha, stamp, _ in logged] == [
        (sha_a, "2026-01-01T00:00:00Z"),
        (sha_b, "2026-01-02T00:00:00Z"),
        (sha_a, "2026-01-03T00:00:00Z"),
    ]


def test_same_timestamp_retrievals_stay_separate(archive_root: Path):
    for _ in range(2):
        raw_archive.archive(
            "sec",
            "t",
            "k",
            _payload("a"),
            url="u",
            retrieved_at="2026-01-01T00:00:00Z",
            record_retrieval=True,
            root=archive_root,
        )
    assert len(_events(archive_root)) == 3


def test_legacy_archive_without_log_yields_manifest_observation(archive_root: Path):
    record = raw_archive.archive(
        "sec", "t", "k", _payload("a"), url="u", retrieved_at="2026-01-01T00:00:00Z", root=archive_root
    )
    assert not (record.payload_path.parent / raw_archive.RETRIEVAL_LOG).exists()
    assert _events(archive_root) == [(record.sha256, "2026-01-01T00:00:00Z", "u")]


def test_missing_archive_yields_no_retrievals(archive_root: Path):
    assert _events(archive_root, key="absent") == []


def test_record_retrieval_false_leaves_other_logs_alone(archive_root: Path):
    logged = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    log_path = logged.payload_path.parent / raw_archive.RETRIEVAL_LOG
    before = log_path.read_bytes()
    raw_archive.archive("sec", "t", "k", _payload("a"), url="u", root=archive_root)
    other = raw_archive.archive("sec", "t", "other", _payload("a"), url="u", root=archive_root)
    assert log_path.read_bytes() == before
    assert not (other.payload_path.parent / raw_archive.RETRIEVAL_LOG).exists()
    assert raw_archive.find("sec", "t", "k", root=archive_root) == logged


_VALID_EVENT = json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u"})


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "",
        "[]",
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u", "extra": 1}),
        json.dumps({"sha256": "a" * 16, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u"}),
        json.dumps({"sha256": "A" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": "u"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-1-1T0:0:0Z", "url": "u"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": 1, "url": "u"}),
        json.dumps({"sha256": "a" * 64, "retrieved_at": "2026-01-01T00:00:00Z", "url": None}),
    ],
)
def test_malformed_retrieval_log_fails_closed(archive_root: Path, line: str):
    record = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    with (record.payload_path.parent / raw_archive.RETRIEVAL_LOG).open("a") as log:
        log.write(line + "\n" + _VALID_EVENT + "\n")
    with pytest.raises(raw_archive.RetrievalLogError):
        _events(archive_root)


def test_non_utf8_retrieval_log_fails_closed(archive_root: Path):
    record = raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)
    with (record.payload_path.parent / raw_archive.RETRIEVAL_LOG).open("ab") as log:
        log.write(b"\xff\n")
    with pytest.raises(raw_archive.RetrievalLogError):
        _events(archive_root)


def test_failed_retrieval_append_raises(archive_root: Path):
    (archive_root / "sec" / "t" / "k" / raw_archive.RETRIEVAL_LOG).mkdir(parents=True)
    with pytest.raises(OSError):
        raw_archive.archive("sec", "t", "k", _payload("a"), url="u", record_retrieval=True, root=archive_root)


def test_bad_retrieval_stamp_writes_nothing(archive_root: Path):
    with pytest.raises(ValueError):
        raw_archive.archive(
            "sec", "t", "k", _payload("a"), url="u", retrieved_at="yesterday", record_retrieval=True, root=archive_root
        )
    assert not archive_root.exists()


# ---------------------------------------------------------------------------
# Warehouse-removal gates
# ---------------------------------------------------------------------------


def test_migration_warehouse_modules_removed() -> None:
    """The warehouse query layer is gone; live providers serve reads now."""
    with pytest.raises(ImportError):
        importlib.import_module("app.storage.parquet")
    with pytest.raises(ImportError):
        importlib.import_module("app.storage.duckdb")


def test_migration_sec_facts_live_assembly() -> None:
    """sec_facts still exposes its live assembly helpers."""
    from app.services import sec_facts as _sec_facts

    assert _sec_facts._assemble_eps_payload is not None
    assert _sec_facts._store_rows is not None


# ---------------------------------------------------------------------------
# Persistent market store
# ---------------------------------------------------------------------------


@pytest.fixture
def market_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "market"
    monkeypatch.setattr(market, "get_data_root", lambda: root)
    monkeypatch.setattr(market, "_local", threading.local())
    try:
        yield root
    finally:
        conn = getattr(market._local, "conn", None)
        if conn is not None:
            conn.close()


def test_market_watchlist_seeds_once_even_after_empty(market_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert market.db_path() == market_root / "market.sqlite"
    assert market.get_watchlist() == list(market.DEFAULT_WATCHLIST)
    assert market.set_watchlist([]) == []
    market._conn().close()
    monkeypatch.setattr(market, "_local", threading.local())
    assert market.get_watchlist() == []


def test_market_watchlist_normalizes_and_deduplicates_in_order(market_root: Path) -> None:
    symbols = iter([" msft ", "", "aapl", "MSFT", " \t ", " spy", "AAPL ", "qqq"])
    assert market.set_watchlist(symbols) == ["MSFT", "AAPL", "SPY", "QQQ"]
    assert market.get_watchlist() == ["MSFT", "AAPL", "SPY", "QQQ"]


def test_market_covered_from_only_widens(market_root: Path) -> None:
    assert market.refresh_state("bars", "NVDA", "1d") is None
    for now, covered_from, expected in (
        (1.0, None, None),
        (2.0, 100.0, 100.0),
        (3.0, 200.0, 100.0),
        (4.0, 50.0, 50.0),
        (5.0, None, 50.0),
    ):
        market.mark_ok("bars", "NVDA", "1d", now=now, covered_from=covered_from)
        state = market.refresh_state("bars", "NVDA", "1d")
        assert state is not None
        assert state["covered_from"] == expected
        assert state["last_ok_at"] == now
    assert market.refresh_state("bars", "NVDA", "1h") is None


def test_market_candle_intervals_include_only_symbol_candle_requests(market_root: Path) -> None:
    assert market.candle_intervals("NVDA") == []
    market.mark_ok("candles", "NVDA", "1d", now=10)
    market.mark_error("candles", "NVDA", "15m", now=20, error="offline")
    market.mark_ok("quote", "NVDA", now=10)
    market.mark_ok("candles", "AAPL", "1wk", now=10)
    assert market.candle_intervals("NVDA") == ["15m", "1d"]


def test_market_mark_error_preserves_last_success(market_root: Path) -> None:
    market.mark_ok("bars", "NVDA", "1d", now=10.0, covered_from=5.0)
    market.mark_error("bars", "NVDA", "1d", now=20.0, error="provider unavailable")
    assert market.refresh_state("bars", "NVDA", "1d") == {
        "covered_from": 5.0,
        "last_ok_at": 10.0,
        "last_error": "provider unavailable",
        "last_error_at": 20.0,
    }
    market.mark_error("bars", "AAPL", "1d", now=30.0, error="first fetch failed")
    assert market.refresh_state("bars", "AAPL", "1d") == {
        "covered_from": None,
        "last_ok_at": None,
        "last_error": "first fetch failed",
        "last_error_at": 30.0,
    }


@settings(derandomize=True, max_examples=40, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", max_size=20),
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=30),
        ),
        max_size=20,
    )
)
def test_prop_market_news_repeat_insert_keeps_row_count(entries: list[tuple[str, str]]) -> None:
    with TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(market, "get_data_root", lambda: Path(tmp))
        monkeypatch.setattr(market, "_local", threading.local())
        items: list[dict[str, object]] = [
            {"url": f"https://news.example/{url}" if url else "", "title": title, "published_at": 100.0}
            for url, title in entries
        ]
        expected = len({str(item["url"] or item["title"]) for item in items})
        try:
            assert market.save_news("NVDA", items, now=100.0) == expected
            before = market.read_news("NVDA", len(items) + 1)
            assert len(before) == expected
            assert market.save_news("NVDA", items, now=200.0) == 0
            assert market.read_news("NVDA", len(items) + 1) == before
        finally:
            conn = getattr(market._local, "conn", None)
            if conn is not None:
                conn.close()


@pytest.mark.parametrize("limited_first", [False, True])
def test_market_multi_cap_refusal_spends_nothing(market_root: Path, limited_first: bool) -> None:
    period = "2026-10"
    assert market.reserve({"limited": 1.0, "roomy": 10.0}, period, 0.75)
    caps = {"limited": 1.0, "roomy": 10.0, "new": 10.0}
    if not limited_first:
        caps = dict(reversed(list(caps.items())))
    assert not market.reserve(caps, period, 0.5)
    assert market.spent("limited", period) == 0.75
    assert market.spent("roomy", period) == 0.75
    assert market.spent("new", period) == 0.0
    assert market.reserve(caps, period, 0.25)
    assert market.spent("limited", period) == 1.0
    assert market.spent("roomy", period) == 1.0
    assert market.spent("new", period) == 0.25
    assert market.spent("limited", "2026-11") == 0.0


def test_market_settle_floors_spend_at_zero(market_root: Path) -> None:
    assert market.reserve({"provider": 10.0}, "2026-10", 2.0)
    market.settle(["provider"], "2026-10", -0.5)
    assert market.spent("provider", "2026-10") == 1.5
    market.settle(["provider", "missing"], "2026-10", -3.0)
    assert market.spent("provider", "2026-10") == 0.0
    assert market.spent("missing", "2026-10") == 0.0
    market.settle(["provider"], "2026-10", 0.5)
    assert market.spent("provider", "2026-10") == 0.5


def test_market_prune_uses_strict_cutoff_and_publication_time(market_root: Path) -> None:
    for title, published_at, fetched_at in (
        ("published-old", 99.0, 200.0),
        ("published-boundary", 100.0, 50.0),
        ("published-new", 101.0, 50.0),
        ("published-zero", 0.0, 200.0),
        ("fetched-old", None, 99.0),
        ("fetched-boundary", None, 100.0),
        ("fetched-new", None, 101.0),
    ):
        market.save_news("NVDA", [{"title": title, "published_at": published_at}], now=fetched_at)
    assert market.prune_news(100.0) == 3
    assert {row["title"] for row in market.read_news("NVDA", 10)} == {
        "published-boundary",
        "published-new",
        "fetched-boundary",
        "fetched-new",
    }
    assert market.prune_news(100.0) == 0


def test_market_database_uses_wal_and_incremental_vacuum(market_root: Path) -> None:
    market.get_watchlist()
    with closing(sqlite3.connect(market.db_path())) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert conn.execute("PRAGMA auto_vacuum").fetchone() == (2,)


def test_quotes_keep_sources_apart(market_root: Path) -> None:
    market.save_quote("NVDA", 10.0, 1.0)
    market.save_quote("NVDA", 12.5, 2.0, source="robinhood")
    assert market.latest_quote("NVDA") == (10.0, 1.0)
    assert market.latest_quote("NVDA", "robinhood") == (12.5, 2.0)
    assert market.latest_quote("AAPL") is None


def test_save_quote_skips_an_unchanged_price_per_source(market_root: Path) -> None:
    assert market.save_quote("NVDA", 10.0, 1.0) is True
    assert market.save_quote("NVDA", 10.0, 2.0) is False
    # The other source keeps its own history.
    assert market.save_quote("NVDA", 10.0, 2.0, source="robinhood") is True
    assert market.save_quote("NVDA", 11.0, 3.0) is True
    # A return to an earlier price is a change from the latest row, so it saves.
    assert market.save_quote("NVDA", 10.0, 4.0) is True
    assert market.latest_quote("NVDA") == (10.0, 4.0)
    with closing(sqlite3.connect(market.db_path())) as conn:
        rows = conn.execute("SELECT source, ts, price FROM quotes ORDER BY source, ts").fetchall()
    assert rows == [("robinhood", 2.0, 10.0), ("yahoo", 1.0, 10.0), ("yahoo", 3.0, 11.0), ("yahoo", 4.0, 10.0)]


def test_quote_symbols_lists_the_watchlist_then_other_read_symbols(market_root: Path) -> None:
    market.set_watchlist(["NVDA", "AAPL"])
    market.mark_ok("quote", "^TNX", now=1)
    market.mark_error("broker_quote", "CL=F", now=1, error="no price")
    market.mark_ok("broker_quote", "NVDA", now=1)
    market.mark_ok("candles", "IGNORED", "1d", now=1)
    assert market.quote_symbols() == ["NVDA", "AAPL", "CL=F", "^TNX"]


def test_latest_quote_any_returns_the_newest_row_with_its_source(market_root: Path) -> None:
    assert market.latest_quote_any("NVDA") is None
    market.save_quote("NVDA", 10.0, 5.0)
    market.save_quote("NVDA", 12.5, 6.0, source="robinhood")
    assert market.latest_quote_any("NVDA") == (12.5, 6.0, "robinhood")
    market.save_quote("NVDA", 11.0, 7.0)
    assert market.latest_quote_any("NVDA") == (11.0, 7.0, "yahoo")
    assert market.latest_quote_any("AAPL") is None


def test_prune_broker_deletes_only_expired_broker_rows(market_root: Path) -> None:
    cutoff = 2 * 86400.0
    # Option days compare in UTC: a snapshot on the cutoff's UTC day survives in every host zone.
    market.save_quote("NVDA", 10.0, 50.0)
    market.save_quote("NVDA", 11.0, cutoff - 1, source="robinhood")
    market.save_quote("NVDA", 12.0, cutoff, source="robinhood")
    market.save_snapshot("rh_options", "NVDA", "1970-01-01", {"calls": []})
    market.save_snapshot("rh_options", "NVDA", "1970-01-03", {"puts": []})
    market.save_snapshot("estimates", "NVDA", "1970-01-01", {"eps": 1})
    assert market.prune_broker(cutoff) == 2
    assert market.latest_quote("NVDA", "robinhood") == (12.0, cutoff)
    assert market.latest_quote("NVDA") == (10.0, 50.0)
    assert market.read_snapshots("rh_options", "NVDA", 5) == [("1970-01-03", {"puts": []})]
    assert market.read_snapshots("estimates", "NVDA", 5) == [("1970-01-01", {"eps": 1})]


def test_prune_broker_keeps_the_cutoff_utc_day_in_a_forward_host_zone(market_root: Path) -> None:
    cutoff = datetime(2026, 10, 31, 20, 0, tzinfo=UTC).timestamp()
    market.save_snapshot("rh_options", "NVDA", "2026-10-31", {"calls": []})
    market.save_snapshot("rh_options", "NVDA", "2026-10-30", {"calls": []})
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TZ", "JST-9")
        time.tzset()
        try:
            assert market.prune_broker(cutoff) == 1
        finally:
            patch.undo()
            time.tzset()
    days = [day for day, _ in market.read_snapshots("rh_options", "NVDA", 5)]
    assert days == ["2026-10-31"]
    # prune_broker compares day strings lexicographically, so the YYYY-MM-DD key is an invariant.
    assert all(len(day) == 10 and day[4] == day[7] == "-" for day in days)


def test_store_creates_a_nested_data_root(market_root: Path) -> None:
    """STOCKBOT_DATA_DIR may point below an existing directory."""
    nested = market_root / "a" / "b"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(market, "get_data_root", lambda: nested)
        assert market.get_watchlist() == list(market.DEFAULT_WATCHLIST)
    assert (nested / "market.sqlite").is_file()


def test_quotes_default_to_the_public_source(market_root: Path) -> None:
    """A writer that omits source still lands in public history, not a separate bucket."""
    market_root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(market.db_path())) as conn:
        conn.executescript(market._SCHEMA)
        conn.execute("INSERT INTO quotes (symbol, ts, price) VALUES ('AAPL', 1, 2)")
        conn.commit()
    assert market.latest_quote("AAPL") == (2.0, 1.0)
    assert market.latest_quote("AAPL", "robinhood") is None


def test_each_thread_reuses_its_connection_until_the_data_root_changes(market_root: Path, tmp_path: Path) -> None:
    first = market._conn()
    assert market._conn() is first
    other = tmp_path / "other-root"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(market, "get_data_root", lambda: other)
        moved = market._conn()
        assert moved is not first
        assert market._conn() is moved
        assert (other / "market.sqlite").is_file()
    # The replaced connection is closed, not leaked.
    with pytest.raises(sqlite3.ProgrammingError):
        first.execute("SELECT 1")


def test_concurrent_openers_migrate_once(market_root: Path) -> None:
    """The collector thread and request threads can open the same v1 file together."""
    market_root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(market.db_path())) as conn:
        conn.execute(
            "CREATE TABLE quotes (symbol TEXT NOT NULL, ts REAL NOT NULL, price REAL NOT NULL, PRIMARY KEY (symbol, ts))"
        )
        conn.executemany("INSERT INTO quotes VALUES (?, ?, ?)", [("AAPL", 5, 7), ("AAPL", 6, 8)])
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    def read_latest(_: int) -> tuple[float, float] | None:
        return market.latest_quote("AAPL")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read_latest, range(8)))
    assert results == [(8.0, 6.0)] * 8
    assert market.latest_quote("AAPL", "robinhood") is None
    with closing(sqlite3.connect(market.db_path())) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)
        assert conn.execute("SELECT source, COUNT(*) FROM quotes GROUP BY source").fetchall() == [("yahoo", 2)]


class _ScriptedWal(sqlite3.Connection):
    """Real connection whose WAL switch first raises the scripted errors, then reports a mode."""

    errors: ClassVar[list[Exception]] = []
    final_mode: ClassVar[str] = "wal"
    current_mode: ClassVar[str] = "delete"
    switches: ClassVar[int] = 0

    @override
    def execute(self, sql: str, parameters: object = (), /) -> sqlite3.Cursor:
        if sql == "PRAGMA journal_mode":
            return super().execute("SELECT ?", (type(self).current_mode,))
        if sql == "PRAGMA journal_mode = WAL":
            type(self).switches += 1
            if type(self).errors:
                raise type(self).errors.pop(0)
            return super().execute("SELECT ?", (type(self).final_mode,))
        return super().execute(sql)


def _no_sleep(seconds: float) -> None:
    del seconds


def _scripted(errors: list[Exception], final_mode: str = "wal", current_mode: str = "delete") -> _ScriptedWal:
    _ScriptedWal.errors = list(errors)
    _ScriptedWal.final_mode = final_mode
    _ScriptedWal.current_mode = current_mode
    _ScriptedWal.switches = 0
    return sqlite3.connect(":memory:", factory=_ScriptedWal)


def test_enable_wal_retries_a_locked_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(market.time, "sleep", sleeps.append)
    with closing(_scripted([sqlite3.OperationalError("database is locked")] * 3)) as conn:
        market._enable_wal(conn)
    assert _ScriptedWal.switches == 4
    # 100 tries of 0.1 s gives the documented 10-second budget.
    assert sleeps == [0.1] * 3


def test_enable_wal_does_not_switch_a_file_already_in_wal() -> None:
    with closing(_scripted([], current_mode="wal")) as conn:
        market._enable_wal(conn)
    assert _ScriptedWal.switches == 0


def test_enable_wal_skips_a_file_that_is_already_wal(tmp_path: Path) -> None:
    path = tmp_path / "already.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
    with closing(sqlite3.connect(path)) as conn:
        market._enable_wal(conn)
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_enable_wal_waits_for_another_writer_to_release(tmp_path: Path) -> None:
    """The real failure: another connection holds the file while we switch to WAL."""
    path = tmp_path / "busy.sqlite"
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("CREATE TABLE t (x)")
    holder.execute("BEGIN EXCLUSIVE")
    release = threading.Timer(0.3, lambda: holder.execute("COMMIT"))
    release.start()
    try:
        with closing(sqlite3.connect(path, timeout=0.01)) as conn:
            market._enable_wal(conn)
            assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        release.join()
        holder.close()


@pytest.mark.parametrize(
    "errors,final_mode,message",
    [
        ([sqlite3.OperationalError("disk I/O error")], "wal", "disk I/O error"),
        ([sqlite3.OperationalError("database is locked")] * 1000, "wal", "^database is locked: could not enable WAL$"),
        # A switch that answers with another mode (an in-memory database) is retried, never accepted.
        ([], "memory", "^database is locked: could not enable WAL$"),
    ],
)
def test_enable_wal_raises_other_errors_and_gives_up_after_its_budget(
    monkeypatch: pytest.MonkeyPatch, errors: list[Exception], final_mode: str, message: str
) -> None:
    monkeypatch.setattr(market.time, "sleep", _no_sleep)
    with closing(_scripted(errors, final_mode)) as conn, pytest.raises(sqlite3.OperationalError, match=message):
        market._enable_wal(conn)
    assert _ScriptedWal.switches == (1 if message == "disk I/O error" else 100)


def test_version_1_database_gains_quote_source(market_root: Path) -> None:
    market_root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(market.db_path())) as conn:
        conn.execute(
            "CREATE TABLE quotes (symbol TEXT NOT NULL, ts REAL NOT NULL, price REAL NOT NULL, PRIMARY KEY (symbol, ts))"
        )
        conn.execute("INSERT INTO quotes VALUES ('AAPL', 5, 7)")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    assert market.latest_quote("AAPL") == (7.0, 5.0)
    assert market.latest_quote("AAPL", "robinhood") is None
    with closing(sqlite3.connect(market.db_path())) as conn:
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)
        # The rebuilt table keeps the public default for writers that omit source.
        conn.execute("INSERT INTO quotes (symbol, ts, price) VALUES ('AAPL', 9, 10)")
        conn.commit()
    assert market.latest_quote("AAPL") == (10.0, 9.0)
    assert market.latest_quote("AAPL", "robinhood") is None


def test_stale_migration_reader_keeps_sources_and_rows(market_root: Path) -> None:
    """An opener that read user_version=1 before another migrated must not rebuild again."""
    market_root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(market.db_path())) as conn:
        conn.execute(
            "CREATE TABLE quotes (symbol TEXT NOT NULL, ts REAL NOT NULL, price REAL NOT NULL, PRIMARY KEY (symbol, ts))"
        )
        conn.execute("INSERT INTO quotes VALUES ('AAPL', 5, 7)")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    assert market.latest_quote("AAPL") == (7.0, 5.0)
    market.save_quote("AAPL", 99.0, 5, source="robinhood")
    with closing(sqlite3.connect(market.db_path(), isolation_level=None)) as conn:
        market._migrate_quotes_sources(conn)
        assert conn.execute("PRAGMA user_version").fetchone() == (2,)
        assert conn.execute("SELECT symbol, source, ts, price FROM quotes ORDER BY source").fetchall() == [
            ("AAPL", "robinhood", 5.0, 99.0),
            ("AAPL", "yahoo", 5.0, 7.0),
        ]


@pytest.mark.parametrize("kinds,symbols", [([], ["AAPL"]), (["quote"], []), ([], [])])
def test_oldest_refresh_empty_axes(market_root: Path, kinds: list[str], symbols: list[str]) -> None:
    assert market.oldest_refresh(kinds, symbols) is None


def test_oldest_refresh_requires_every_pair_and_uses_oldest_parameter(market_root: Path) -> None:
    market.mark_ok("quote", "AAPL", now=30)
    assert market.oldest_refresh(["quote", "candles"], ["AAPL"]) is None
    market.mark_error("candles", "AAPL", "1d", now=40, error="offline")
    assert market.oldest_refresh(["quote", "candles"], ["AAPL"]) is None
    market.mark_ok("candles", "AAPL", "1d", now=20)
    market.mark_ok("candles", "AAPL", "5m", now=10)
    market.mark_ok("quote", "OTHER", now=1)
    assert market.oldest_refresh(iter(["quote", "candles"]), iter(["AAPL"])) == 10
    assert market.oldest_refresh(["quote"], ["AAPL", "MISSING"]) is None


def test_mark_error_truncates_and_replaces_error_without_losing_success(market_root: Path) -> None:
    market.mark_error("news", "AAPL", now=1, error="x" * 501)
    state = market.refresh_state("news", "AAPL")
    assert state is not None
    assert state["last_error"] == "x" * 500
    market.mark_ok("news", "AAPL", now=2)
    market.mark_error("news", "AAPL", now=3, error="retry failed")
    assert market.refresh_state("news", "AAPL") == {
        "covered_from": None,
        "last_ok_at": 2,
        "last_error": "retry failed",
        "last_error_at": 3,
    }


def test_save_news_preserves_existing_content_and_counts_new_rows(market_root: Path) -> None:
    first = {"title": "original", "url": "https://example.test/story", "source": "wire", "published_at": 3}
    assert market.save_news("AAPL", [first], now=4) == 1
    hash_ = market.url_hash(first["url"])
    market.set_news_body("AAPL", hash_, "body")
    market.set_summary("AAPL", hash_, "written", "story.md")
    assert market.save_news("AAPL", [{**first, "title": "replacement"}], now=5) == 0
    assert market.read_news("AAPL", 10) == [
        {
            **first,
            "summary_path": "story.md",
        }
    ]
    assert market.save_news("OTHER", [first], now=6) == 1
    assert market.save_news("AAPL", [{"title": "untitled URL", "url": None, "source": None}], now=7) == 1
    assert market.read_news("AAPL", 1) == [
        {
            "title": "untitled URL",
            "url": "",
            "source": "",
            "published_at": None,
            "summary_path": None,
        }
    ]
