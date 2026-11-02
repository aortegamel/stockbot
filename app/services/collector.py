"""Collect the saved watchlist in one interruptible daemon thread."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from app.config import get_data_root, get_finra_client_id, get_finra_client_secret, google_source_enabled
from app.storage import market

logger = logging.getLogger(__name__)
DAY = 86400


def retain_news(*, now: float) -> None:
    """Remove news and broker history older than 30 days inside the approved roots only."""
    cutoff = now - 30 * DAY
    market.prune_news(cutoff)
    market.prune_broker(cutoff)
    root = get_data_root()
    # Reject symlink ancestors as well as symlink entries.
    roots: tuple[tuple[str, bool, str, Callable[[Path], str]], ...] = (
        ("news", True, "*", lambda entry: entry.name),
        ("briefing", False, "*.json", lambda entry: entry.stem),
    )
    for relative, directories, pattern, entry_name in roots:
        folder = root / relative
        if folder.is_symlink() or not folder.is_dir():
            continue
        for entry in folder.glob(pattern):
            if entry.is_symlink() or (entry.is_dir() != directories):
                continue
            try:
                stamp = date.fromisoformat(entry_name(entry))
            except ValueError:
                continue
            if datetime.combine(stamp, datetime.min.time(), UTC).timestamp() < cutoff:
                if directories:
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    _prune_captures(root / "raw" / "viewer" / "capture", cutoff)
    market.vacuum()


def _capture_timestamp(manifest: Path) -> float | None:
    """Read an aware capture timestamp without deleting malformed evidence."""
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(value["retrieved_at"])
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except OSError, ValueError, KeyError, TypeError, AttributeError:
        logger.warning("Cannot read capture manifest %s", manifest)
        return None


def _prune_captures(capture: Path, cutoff: float) -> None:
    """Delete expired capture pairs without traversing symlink ancestors."""
    if not any(map(Path.is_symlink, (capture.parent.parent, capture.parent, capture))) and capture.is_dir():
        for key in capture.glob("*/"):
            if key.is_symlink():
                continue
            for manifest in key.glob("*.manifest.json"):
                if manifest.is_symlink():
                    continue
                stamp = _capture_timestamp(manifest)
                if stamp is not None and stamp < cutoff:
                    payload = manifest.with_name(manifest.name.removesuffix(".manifest.json") + ".json")
                    if not payload.is_symlink():
                        payload.unlink(missing_ok=True)
                    manifest.unlink()


class Collector:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], object] | None = None,
    ) -> None:
        self.clock = clock
        self.wake = threading.Event()
        self._sleep = sleeper or self.wake.wait
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._force = False
        self._thread: threading.Thread | None = None
        self._quote_threads: list[threading.Thread] = []
        self._running = False
        self._last_cycle_at: float | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="market-collector", daemon=True)
            # Separate threads: a slow Yahoo sweep (20 s timeout per symbol) must never delay the 5 s broker poll.
            self._quote_threads = [
                threading.Thread(target=self._quote_loop, args=(tick,), name=f"market-quotes-{name}", daemon=True)
                for name, tick in (("broker", self.broker_tick), ("yahoo", self.yahoo_tick))
            ]
            self._thread.start()
            for thread in self._quote_threads:
                thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.wake.set()
        for thread in (self._thread, *self._quote_threads):
            if thread is not None:
                thread.join(timeout=5)

    def request_refresh(self) -> None:
        with self._lock:
            self._force = True
            self.wake.set()

    def status(self) -> dict[str, object]:
        """running is true only while a requested refresh is queued or executing, not for routine cycles."""
        with self._lock:
            state: dict[str, object] = {"running": self._force or self._running, "last_cycle_at": self._last_cycle_at}
        state["oldest_as_of"] = market.oldest_refresh(["quote", "candles", "news", "filings"], market.get_watchlist())
        return state

    def _job(
        self,
        kind: str,
        symbol: str,
        interval: int,
        action: Callable[[], object],
        *,
        now: float,
        force: bool,
        params: str = "",
    ) -> None:
        state = market.refresh_state(kind, symbol, params) or {}
        previous = [v for k in ("last_ok_at", "last_error_at") if isinstance((v := state.get(k)), (int, float))]
        if not force and now - max(previous, default=float("-inf")) < interval:
            return
        if self._stop.is_set():
            return
        try:
            result = action()
            if isinstance(result, dict) and result.get("error"):
                raise ValueError(str(result["error"]))
            market.mark_ok(kind, symbol, params, now=now)
        except Exception as exc:  # noqa: BLE001 - isolate each external collection job
            logger.warning("Market collection failed for %s/%s/%s: %s", kind, symbol, params, exc)
            market.mark_error(kind, symbol, params, now=now, error=str(exc))

    def _snapshot(self, kind: str, symbol: str, action: Callable[[], object], now: float) -> object:
        payload = action()
        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(str(payload["error"]))
        market.save_snapshot(kind, symbol, datetime.fromtimestamp(now, UTC).date().isoformat(), payload)
        return payload

    def _company_facts(self, symbol: str) -> object:
        from api.market_providers import cik_for_ticker
        from app.data_sources import SourceGateway

        cik = cik_for_ticker(symbol)
        if cik is None:
            raise ValueError(f"No SEC company identifier for {symbol}")
        return SourceGateway().company_facts(cik)

    def _optional_jobs(self, symbol: str, now: float) -> list[tuple[str, int, str, Callable[[], object]]]:
        """Per-symbol jobs whose sources need credentials: FINRA and a connected broker."""
        from api import market_providers as providers
        from app import finra_client

        jobs: list[tuple[str, int, str, Callable[[], object]]] = []
        if get_finra_client_id() and get_finra_client_secret():
            for kind, action in (
                ("finra_short_interest", finra_client.get_short_interest),
                ("finra_reg_sho", finra_client.get_reg_sho_volume),
            ):
                jobs.append((kind, DAY, "", lambda k=kind, a=action, s=symbol: self._snapshot(k, s, lambda: a(s), now)))
        if providers._broker_connected(now):
            jobs.append(("rh_options", DAY, "", lambda s=symbol: providers.refresh_options(s, now=now)))
        return jobs

    def cycle(self) -> None:
        from api import market_providers as providers
        from app import analyst_client
        from app.services import news_summary

        now = self.clock()
        with self._lock:
            force, self._force = self._force, False
            self.wake.clear()
            self._running = force
        try:
            for symbol in market.get_watchlist():
                jobs: list[tuple[str, int, str, Callable[[], object]]] = [
                    ("news", 900, "", lambda s=symbol: providers.refresh_news(s, now=now)),
                    ("filings", 3600, "", lambda s=symbol: providers.refresh_filings(s, now=now)),
                    (
                        "estimates",
                        DAY,
                        "",
                        lambda s=symbol: self._snapshot(
                            "estimates", s, lambda: analyst_client.get_analyst_estimates(s), now
                        ),
                    ),
                    (
                        "companyfacts",
                        DAY,
                        "",
                        lambda s=symbol: self._company_facts(s),
                    ),
                    ("exa_news", DAY, "", lambda s=symbol: news_summary.collect_exa_news(s, now=now)),
                ]
                tracked = market.candle_intervals(symbol)
                ranges = {"1d": "1mo" if "1d" in tracked else "1y", "1wk": "1y", "1mo": "1y", "3mo": "1y"}
                for interval in dict.fromkeys(["1d", "15m", "5m", "30m", *tracked]):
                    range_ = ranges.get(interval, "5d")
                    jobs.append(
                        (
                            "candles",
                            900,
                            interval,
                            lambda i=interval, r=range_, s=symbol: providers.refresh_candles(s, i, r, now=now),
                        )
                    )
                jobs.extend(self._optional_jobs(symbol, now))
                for kind, interval_seconds, params, action in jobs:
                    self._job(kind, symbol, interval_seconds, action, now=now, force=force, params=params)
            if google_source_enabled("bigquery"):
                from app.google_data.trends import collect_trends

                day = datetime.fromtimestamp(now, UTC).date().isoformat()
                self._job(
                    "google_trends",
                    "",
                    DAY,
                    lambda: self._snapshot(
                        "google_trends",
                        "",
                        lambda: collect_trends(start_date=day, end_date=day, geos=["US"], data_root=get_data_root()),
                        now,
                    ),
                    now=now,
                    force=force,
                )
            for kind, interval_seconds, action in (
                ("news_bodies", 60, lambda: news_summary.fetch_bodies(5, now=now)),
                ("news_summaries", 60, lambda: news_summary.summarize_pending(3, now=now)),
                ("retention", DAY, lambda: retain_news(now=now)),
            ):
                self._job(kind, "", interval_seconds, action, now=now, force=force)
        finally:
            with self._lock:
                self._running = False
                self._last_cycle_at = now

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cycle()
            except Exception:
                logger.exception("Market collector cycle failed")
            self._sleep(60)

    def broker_tick(self) -> None:
        """One batched Robinhood quote call for every tracked symbol, while the broker is connected."""
        from api import market_providers as providers

        now = self.clock()
        symbols = market.quote_symbols()
        if symbols and providers._broker_connected(now):
            try:
                providers.refresh_broker_quotes(symbols, now=now)
            except Exception as exc:  # noqa: BLE001 - refresh_broker_quotes records the error per symbol
                logger.warning("Broker quote collection failed: %s", exc)

    def yahoo_tick(self) -> None:
        """Yahoo quote for each tracked symbol whose 60 s interval has passed."""
        from api import market_providers as providers

        now = self.clock()
        for symbol in market.quote_symbols():
            if self._stop.is_set():
                return
            self._job(
                "quote",
                symbol,
                providers.YAHOO_QUOTE_INTERVAL,
                lambda s=symbol: providers.refresh_quote(s, now=now),
                now=now,
                force=False,
            )

    def _quote_loop(self, tick: Callable[[], None]) -> None:
        from api.market_providers import BROKER_QUOTE_INTERVAL

        while not self._stop.is_set():
            try:
                tick()
            except Exception:
                logger.exception("Market quote tick failed")
            self._stop.wait(BROKER_QUOTE_INTERVAL)


_collector = Collector()
start = _collector.start
stop = _collector.stop
request_refresh = _collector.request_refresh
status = _collector.status
