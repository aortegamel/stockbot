"""Local market-data history: data/market.sqlite. stdlib only.

The background collector (app/services/collector.py) and on-demand reads in
api/market_providers.py upsert into these tables. Primary keys make every
write idempotent, so a repeated fetch never duplicates a row.

- watchlist: the one saved watchlist per install, seeded once from DEFAULT_WATCHLIST.
- bars / quotes / filings: time series, kept forever.
- quotes.source tags 'yahoo' or 'robinhood'. Robinhood quotes and 'rh_options' snapshots expire after 30 days.
- news: headlines plus fetched article text and summary state, deleted after 30 days.
- snapshots: one JSON payload per (kind, symbol, UTC day): estimates, FINRA, companyfacts, US Trends, rh_options.
- refresh_log: last successful refresh and covered range per (kind, symbol, params).
- budget: dollar spend per (source, period) for paid providers.

WAL lets the FastAPI collector thread write while the kernel worker and
request threads read. BEGIN IMMEDIATE serializes budget reservations
across processes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_data_root

DEFAULT_WATCHLIST = ("NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "AMZN", "META", "GOOG")
_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY, position INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL, interval TEXT NOT NULL, ts INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL NOT NULL, vol REAL NOT NULL,
    PRIMARY KEY (symbol, interval, ts)
);
CREATE TABLE IF NOT EXISTS quotes (
    symbol TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'yahoo', ts REAL NOT NULL, price REAL NOT NULL,
    PRIMARY KEY (symbol, source, ts)
);
CREATE TABLE IF NOT EXISTS news (
    symbol TEXT NOT NULL, url_hash TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL,
    source TEXT NOT NULL, published_at REAL, fetched_at REAL NOT NULL, body TEXT,
    summary_status TEXT, summary_path TEXT,
    PRIMARY KEY (symbol, url_hash)
);
CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news(symbol, published_at);
CREATE TABLE IF NOT EXISTS filings (
    cik INTEGER NOT NULL, link TEXT NOT NULL, symbol TEXT NOT NULL, title TEXT NOT NULL,
    filed TEXT NOT NULL, updated TEXT NOT NULL, summary TEXT NOT NULL,
    PRIMARY KEY (cik, link)
);
CREATE TABLE IF NOT EXISTS snapshots (
    kind TEXT NOT NULL, symbol TEXT NOT NULL, day TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (kind, symbol, day)
);
CREATE TABLE IF NOT EXISTS refresh_log (
    kind TEXT NOT NULL, symbol TEXT NOT NULL, params TEXT NOT NULL,
    covered_from REAL, last_ok_at REAL, last_error TEXT, last_error_at REAL,
    PRIMARY KEY (kind, symbol, params)
);
CREATE TABLE IF NOT EXISTS budget (
    source TEXT NOT NULL, period TEXT NOT NULL, spent REAL NOT NULL, PRIMARY KEY (source, period)
);
CREATE TABLE IF NOT EXISTS ciks (symbol TEXT PRIMARY KEY, cik INTEGER NOT NULL);
"""

_local = threading.local()
# Threads in one process open the file together at startup (collector, quote loop, requests).
_open_lock = threading.Lock()


def db_path() -> Path:
    return get_data_root() / "market.sqlite"


def _migrate_quotes_sources(conn: sqlite3.Connection) -> None:
    """Rebuild quotes with source in the key, once per database.

    Another process can migrate between our first user_version read and this write, so the
    decision is re-read under the write lock. Without that, a stale opener would rename the
    migrated table and relabel broker rows as 'yahoo', or fail on the key.
    """
    with _tx(conn):
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            return
        conn.execute("ALTER TABLE quotes RENAME TO quotes_v1")
        conn.execute(
            "CREATE TABLE quotes (symbol TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'yahoo', ts REAL NOT NULL, "
            "price REAL NOT NULL, PRIMARY KEY (symbol, source, ts))"
        )
        conn.execute("INSERT INTO quotes (symbol, source, ts, price) SELECT symbol, 'yahoo', ts, price FROM quotes_v1")
        conn.execute("DROP TABLE quotes_v1")
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL. The switch needs exclusive access and ignores busy_timeout, so retry briefly.

    A file that is already WAL needs no switch. Another process holding the file can make the
    switch fail with 'database is locked' for a moment; that is retried for up to 10 seconds.
    """
    for _ in range(100):
        try:
            if conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
                return
            if conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal":
                return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc):
                raise
        time.sleep(0.1)
    raise sqlite3.OperationalError("database is locked: could not enable WAL")


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 30000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        # auto_vacuum must be set before the first table exists.
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
    _enable_wal(conn)
    conn.executescript(_SCHEMA)
    if version == 1:
        _migrate_quotes_sources(conn)
    if version < _SCHEMA_VERSION:
        with _tx(conn):
            if version == 0:
                conn.executemany(
                    "INSERT OR IGNORE INTO watchlist (symbol, position) VALUES (?, ?)",
                    [(s, i) for i, s in enumerate(DEFAULT_WATCHLIST)],
                )
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    return conn


def _conn() -> sqlite3.Connection:
    path = db_path()
    conn = getattr(_local, "conn", None)
    if conn is None or getattr(_local, "path", None) != path:
        if conn is not None:
            conn.close()
        with _open_lock:
            conn = _open(path)
        _local.conn = conn
        _local.path = path
    return conn


class _tx:
    """BEGIN IMMEDIATE ... COMMIT/ROLLBACK on an autocommit connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


# -- watchlist ---------------------------------------------------------------


def get_watchlist() -> list[str]:
    return [r[0] for r in _conn().execute("SELECT symbol FROM watchlist ORDER BY position")]


def set_watchlist(symbols: Iterable[str]) -> list[str]:
    """Replace the watchlist. Symbols are stripped, upper-cased, deduped, order kept."""
    clean = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    with _tx(_conn()) as conn:
        conn.execute("DELETE FROM watchlist")
        conn.executemany(
            "INSERT INTO watchlist (symbol, position) VALUES (?, ?)", [(s, i) for i, s in enumerate(clean)]
        )
    return clean


# -- refresh log ---------------------------------------------------------------


def mark_ok(kind: str, symbol: str, params: str = "", *, now: float, covered_from: float | None = None) -> None:
    """Record a successful refresh. covered_from only ever widens."""
    _conn().execute(
        """INSERT INTO refresh_log (kind, symbol, params, covered_from, last_ok_at) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (kind, symbol, params) DO UPDATE SET
            last_ok_at = excluded.last_ok_at,
            covered_from = CASE
                WHEN excluded.covered_from IS NULL THEN refresh_log.covered_from
                WHEN refresh_log.covered_from IS NULL THEN excluded.covered_from
                ELSE MIN(refresh_log.covered_from, excluded.covered_from) END""",
        (kind, symbol, params, covered_from, now),
    )


def mark_error(kind: str, symbol: str, params: str = "", *, now: float, error: str) -> None:
    """Record a failed refresh. last_ok_at stays unchanged."""
    _conn().execute(
        """INSERT INTO refresh_log (kind, symbol, params, last_error, last_error_at) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (kind, symbol, params) DO UPDATE SET
            last_error = excluded.last_error, last_error_at = excluded.last_error_at""",
        (kind, symbol, params, error[:500], now),
    )


def refresh_state(kind: str, symbol: str, params: str = "") -> dict[str, object] | None:
    row = (
        _conn()
        .execute(
            "SELECT covered_from, last_ok_at, last_error, last_error_at FROM refresh_log "
            "WHERE kind = ? AND symbol = ? AND params = ?",
            (kind, symbol, params),
        )
        .fetchone()
    )
    if row is None:
        return None
    return {"covered_from": row[0], "last_ok_at": row[1], "last_error": row[2], "last_error_at": row[3]}


def candle_intervals(symbol: str) -> list[str]:
    """Return candle intervals requested for this symbol, including failed requests."""
    return [
        row[0]
        for row in _conn().execute(
            "SELECT params FROM refresh_log WHERE kind = 'candles' AND symbol = ? ORDER BY params", (symbol,)
        )
    ]


def quote_symbols() -> list[str]:
    """Watchlist symbols first, then every other symbol a quote was ever requested for (index tape, searches)."""
    watchlist = get_watchlist()
    read = [
        row[0]
        for row in _conn().execute(
            "SELECT DISTINCT symbol FROM refresh_log WHERE kind IN ('quote', 'broker_quote') ORDER BY symbol"
        )
    ]
    return list(dict.fromkeys([*watchlist, *read]))


def oldest_refresh(kinds: Iterable[str], symbols: Iterable[str]) -> float | None:
    """Oldest last_ok_at across the given kinds and symbols; None when any pair never refreshed."""
    kinds_l, syms_l = list(kinds), list(symbols)
    if not kinds_l or not syms_l:
        return None
    rows = {
        (k, s): ok
        for k, s, ok in _conn().execute(
            f"SELECT kind, symbol, MIN(last_ok_at) FROM refresh_log WHERE kind IN ({','.join('?' * len(kinds_l))}) "
            "GROUP BY kind, symbol",
            kinds_l,
        )
    }
    values = [rows.get((k, s)) for k in kinds_l for s in syms_l]
    if any(v is None for v in values):
        return None
    return min(v for v in values if v is not None)


# -- bars and quotes -----------------------------------------------------------


def save_bars(symbol: str, interval: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Upsert candle rows shaped like market_providers._candle_rows output."""
    _conn().executemany(
        "INSERT OR REPLACE INTO bars (symbol, interval, ts, open, high, low, close, vol) VALUES (?,?,?,?,?,?,?,?)",
        [(symbol, interval, r["t"], r["open"], r["high"], r["low"], r["close"], r["vol"]) for r in rows],
    )


def read_bars(symbol: str, interval: str, since: float | None) -> list[dict[str, object]]:
    rows = _conn().execute(
        "SELECT ts, open, high, low, close, vol FROM bars WHERE symbol = ? AND interval = ? AND ts >= ? ORDER BY ts",
        (symbol, interval, since if since is not None else float("-inf")),
    )
    return [{"t": t, "open": o, "high": h, "low": lo, "close": c, "vol": v} for t, o, h, lo, c, v in rows]


def save_quote(symbol: str, price: float, ts: float, source: str = "yahoo") -> bool:
    """Save a quote unless its price equals the newest saved price from the same source.

    Fast polling (Robinhood every few seconds) would otherwise store one identical row per poll
    outside market hours. Returns whether a row was written; freshness lives in refresh_log.
    """
    cur = _conn().execute(
        "INSERT OR REPLACE INTO quotes (symbol, ts, price, source) SELECT ?, ?, ?, ? "
        "WHERE NOT EXISTS (SELECT 1 FROM (SELECT price FROM quotes WHERE symbol = ? AND source = ? "
        "ORDER BY ts DESC LIMIT 1) WHERE price = ?)",
        (symbol, ts, price, source, symbol, source, price),
    )
    return cur.rowcount > 0


def latest_quote(symbol: str, source: str = "yahoo") -> tuple[float, float] | None:
    """(price, ts) of the newest stored quote from source, or None."""
    row = (
        _conn()
        .execute(
            "SELECT price, ts FROM quotes WHERE symbol = ? AND source = ? ORDER BY ts DESC LIMIT 1", (symbol, source)
        )
        .fetchone()
    )
    return (row[0], row[1]) if row else None


def latest_quote_any(symbol: str) -> tuple[float, float, str] | None:
    """(price, ts, source) of the newest stored quote from any source, or None."""
    row = (
        _conn()
        .execute("SELECT price, ts, source FROM quotes WHERE symbol = ? ORDER BY ts DESC LIMIT 1", (symbol,))
        .fetchone()
    )
    return (row[0], row[1], row[2]) if row else None


# -- news ------------------------------------------------------------------------


def save_news(symbol: str, items: Iterable[Mapping[str, object]], *, now: float) -> int:
    """Insert new headlines (title/url/source/published_at). Existing rows stay untouched. Returns the new-row count."""
    conn = _conn()
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO news (symbol, url_hash, title, url, source, published_at, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                symbol,
                url_hash(str(i.get("url") or i["title"])),
                i["title"],
                i.get("url") or "",
                i.get("source") or "",
                i.get("published_at"),
                now,
            )
            for i in items
        ],
    )
    return conn.total_changes - before


def read_news(symbol: str, limit: int) -> list[dict[str, object]]:
    rows = _conn().execute(
        "SELECT title, url, source, published_at, summary_path FROM news WHERE symbol = ? "
        "ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ?",
        (symbol, limit),
    )
    return [{"title": t, "url": u, "source": s, "published_at": p, "summary_path": sp} for t, u, s, p, sp in rows]


def pending_summaries(limit: int) -> list[dict[str, object]]:
    """News rows not yet summarized, newest first."""
    rows = _conn().execute(
        "SELECT symbol, url_hash, title, url, source, published_at, body FROM news "
        "WHERE summary_status IS NULL ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT ?",
        (limit,),
    )
    keys = ("symbol", "url_hash", "title", "url", "source", "published_at", "body")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def set_news_body(symbol: str, hash_: str, body: str) -> None:
    _conn().execute("UPDATE news SET body = ? WHERE symbol = ? AND url_hash = ?", (body, symbol, hash_))


def set_summary(symbol: str, hash_: str, status: str, path: str | None = None) -> None:
    """status: 'written' (file at path), 'skipped' (irrelevant or low impact), 'error'."""
    _conn().execute(
        "UPDATE news SET summary_status = ?, summary_path = ? WHERE symbol = ? AND url_hash = ?",
        (status, path, symbol, hash_),
    )


def prune_news(cutoff: float) -> int:
    """Delete news older than cutoff (published_at, else fetched_at). Returns deleted rows."""
    cur = _conn().execute("DELETE FROM news WHERE COALESCE(published_at, fetched_at) < ?", (cutoff,))
    return cur.rowcount


def prune_broker(cutoff: float) -> int:
    """Delete Robinhood quotes and 'rh_options' snapshots older than cutoff. Returns deleted rows."""
    day = datetime.fromtimestamp(cutoff, UTC).date().isoformat()
    conn = _conn()
    quotes = conn.execute("DELETE FROM quotes WHERE source = 'robinhood' AND ts < ?", (cutoff,)).rowcount
    snaps = conn.execute("DELETE FROM snapshots WHERE kind = 'rh_options' AND day < ?", (day,)).rowcount
    return quotes + snaps


# -- filings and snapshots --------------------------------------------------------


def save_filings(symbol: str, cik: int, items: Iterable[Mapping[str, object]]) -> None:
    _conn().executemany(
        "INSERT OR REPLACE INTO filings (cik, link, symbol, title, filed, updated, summary) VALUES (?,?,?,?,?,?,?)",
        [(cik, i["link"], symbol, i["title"], i["filed"], i["updated"], i["summary"]) for i in items],
    )


def read_filings(cik: int, limit: int) -> list[dict[str, object]]:
    rows = _conn().execute(
        "SELECT title, link, filed, updated, summary FROM filings WHERE cik = ? ORDER BY updated DESC, filed DESC LIMIT ?",
        (cik, limit),
    )
    return [{"title": t, "link": lk, "filed": f, "updated": u, "summary": s} for t, lk, f, u, s in rows]


def save_cik(symbol: str, cik: int) -> None:
    _conn().execute("INSERT OR REPLACE INTO ciks (symbol, cik) VALUES (?, ?)", (symbol, cik))


def get_cik(symbol: str) -> int | None:
    """Stored ticker -> CIK, independent of filing rows; None before the first resolve."""
    row = _conn().execute("SELECT cik FROM ciks WHERE symbol = ?", (symbol,)).fetchone()
    return int(row[0]) if row else None


def save_snapshot(kind: str, symbol: str, day: str, payload: object) -> None:
    _conn().execute(
        "INSERT OR REPLACE INTO snapshots (kind, symbol, day, payload) VALUES (?, ?, ?, ?)",
        (kind, symbol, day, json.dumps(payload, sort_keys=True, default=str)),
    )


def read_snapshots(kind: str, symbol: str, limit: int) -> list[tuple[str, object]]:
    """(day, payload) newest first."""
    rows = _conn().execute(
        "SELECT day, payload FROM snapshots WHERE kind = ? AND symbol = ? ORDER BY day DESC LIMIT ?",
        (kind, symbol, limit),
    )
    return [(d, json.loads(p)) for d, p in rows]


# -- paid-provider budget ------------------------------------------------------------


def reserve(caps: Mapping[str, float], period: str, amount: float) -> bool:
    """Atomically add amount to every source in caps unless one would pass its cap.

    caps maps source -> dollar cap for this period. Returns False and spends
    nothing when any cap would be exceeded.
    """
    with _tx(_conn()) as conn:
        for source, cap in caps.items():
            row = conn.execute("SELECT spent FROM budget WHERE source = ? AND period = ?", (source, period)).fetchone()
            if (row[0] if row else 0.0) + amount > cap:
                return False
        for source in caps:
            conn.execute(
                "INSERT INTO budget (source, period, spent) VALUES (?, ?, ?) "
                "ON CONFLICT (source, period) DO UPDATE SET spent = spent + excluded.spent",
                (source, period, amount),
            )
    return True


def settle(sources: Iterable[str], period: str, delta: float) -> None:
    """Correct a reservation by delta (actual - reserved); spend never drops below zero."""
    with _tx(_conn()) as conn:
        for source in sources:
            conn.execute(
                "UPDATE budget SET spent = MAX(0, spent + ?) WHERE source = ? AND period = ?", (delta, source, period)
            )


def spent(source: str, period: str) -> float:
    row = _conn().execute("SELECT spent FROM budget WHERE source = ? AND period = ?", (source, period)).fetchone()
    return float(row[0]) if row else 0.0


def vacuum() -> None:
    _conn().execute("PRAGMA incremental_vacuum")
