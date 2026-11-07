"""Deterministic collection, retention, and market route contracts."""

import json
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.services import collector
from app.storage import market, raw_archive

type Calls = list[tuple[str, tuple[object, ...]]]


@pytest.fixture
def isolated_market(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(market, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(collector, "get_data_root", lambda: tmp_path)
    market.set_watchlist(["AAPL"])
    return tmp_path


@pytest.fixture
def retention_entry_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present skipped evidence before expired files, regardless of filesystem order."""
    glob = Path.glob

    def ordinary_entry(path: Path) -> bool:
        return not (path.is_symlink() or path.name.startswith("not-a-date"))

    def skipped_first(path: Path, pattern: str) -> Iterator[Path]:
        return iter(sorted(glob(path, pattern), key=ordinary_entry))

    monkeypatch.setattr(Path, "glob", skipped_first)


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> Calls:
    from api import market_providers
    from app import analyst_client
    from app.data_sources import SourceGateway
    from app.services import news_summary

    calls: Calls = []

    def record(kind: str) -> Callable[..., dict[str, object]]:
        def run(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append((kind, args))
            return {}

        return run

    for kind in ("quote", "candles", "news", "filings"):
        monkeypatch.setattr(market_providers, f"refresh_{kind}", record(kind))

    def cik(_symbol: str) -> int:
        return 320193

    def disabled(_source: str) -> bool:
        return False

    monkeypatch.setattr(market_providers, "cik_for_ticker", cik)
    monkeypatch.setattr(analyst_client, "get_analyst_estimates", record("estimates"))
    monkeypatch.setattr(SourceGateway, "company_facts", staticmethod(record("companyfacts")))
    for name in ("collect_exa_news", "fetch_bodies", "summarize_pending"):
        monkeypatch.setattr(news_summary, name, record(name))
    monkeypatch.setattr(collector, "get_finra_client_id", lambda: None)
    monkeypatch.setattr(collector, "google_source_enabled", disabled)
    monkeypatch.setattr(collector, "get_finra_client_secret", lambda: None)
    return calls


def test_due_jobs_force_refresh_and_status(isolated_market: Path, providers: Calls) -> None:
    now = [100_000.0]
    service = collector.Collector(clock=lambda: now[0])
    service.cycle()
    first = list(providers)
    assert "quote" not in [kind for kind, _ in first]
    assert len([row for row in first if row[0] == "candles"]) == 4
    assert market.read_snapshots("estimates", "AAPL", 1)
    service.yahoo_tick()
    assert service.status() == {"running": False, "last_cycle_at": now[0], "oldest_as_of": now[0]}
    providers.clear()
    now[0] += 60
    service.cycle()
    assert [kind for kind, _ in providers] == ["fetch_bodies", "summarize_pending"]
    providers.clear()
    now[0] += 240
    service.cycle()
    assert [kind for kind, _ in providers] == ["fetch_bodies", "summarize_pending"]
    providers.clear()
    service.request_refresh()
    assert service.wake.is_set()
    # A queued manual refresh reports running before its cycle starts; routine cycles never do.
    assert service.status()["running"] is True
    service.cycle()
    assert service.status()["running"] is False
    expected = [
        (kind, ("AAPL", "1d", "1mo") if kind == "candles" and args[1] == "1d" else args) for kind, args in first
    ]
    assert providers == expected
    assert not service.wake.is_set()


def test_collector_refreshes_intervals_requested_by_reads(isolated_market: Path, providers: Calls) -> None:
    market.mark_ok("candles", "AAPL", "1h", now=1, covered_from=0)
    market.mark_ok("candles", "AAPL", "1wk", now=1, covered_from=0)
    market.mark_error("candles", "AAPL", "1mo", now=1, error="offline")
    market.mark_ok("candles", "AAPL", "1d", now=1, covered_from=0)
    collector.Collector(clock=lambda: 100_000).cycle()
    candle_args = {args for kind, args in providers if kind == "candles"}
    assert candle_args == {
        ("AAPL", "1d", "1mo"),
        ("AAPL", "15m", "5d"),
        ("AAPL", "5m", "5d"),
        ("AAPL", "30m", "5d"),
        ("AAPL", "1h", "5d"),
        ("AAPL", "1wk", "1y"),
        ("AAPL", "1mo", "1y"),
    }
    state = market.refresh_state("candles", "AAPL", "1d")
    assert state is not None
    assert state["covered_from"] == 0


def test_failed_job_waits_its_interval_and_other_jobs_continue(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import market_providers

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("quote unavailable")

    monkeypatch.setattr(market_providers, "refresh_quote", fail)
    now = [100_000.0]
    service = collector.Collector(clock=lambda: now[0])
    service.yahoo_tick()
    service.cycle()
    state = market.refresh_state("quote", "AAPL")
    assert state is not None
    assert state["last_error"] == "quote unavailable"
    assert state["last_ok_at"] is None
    filings_state = market.refresh_state("filings", "AAPL")
    assert filings_state is not None
    assert filings_state["last_ok_at"] == now[0]
    now[0] += market_providers.YAHOO_QUOTE_INTERVAL - 1

    def unexpected_retry(*args: object, **kwargs: object) -> None:
        pytest.fail("retried too soon")

    monkeypatch.setattr(market_providers, "refresh_quote", unexpected_retry)
    service.yahoo_tick()
    failed_state = market.refresh_state("quote", "AAPL")
    assert failed_state is not None
    assert failed_state["last_error_at"] == 100_000


def test_quote_tick_polls_broker_every_tick_and_yahoo_when_due(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import market_providers

    market.set_watchlist(["AAPL", "NVDA"])
    batches: list[list[str]] = []
    clocks: list[float] = []

    def connected(now: float) -> bool:
        clocks.append(now)
        return True

    def broker(tickers: list[str], *, now: float) -> None:
        clocks.append(now)
        batches.append(list(tickers))

    def yahoo(ticker: str, *, now: float) -> None:
        clocks.append(now)
        providers.append(("quote", (ticker,)))

    monkeypatch.setattr(market_providers, "_broker_connected", connected)
    monkeypatch.setattr(market_providers, "refresh_broker_quotes", broker)
    monkeypatch.setattr(market_providers, "refresh_quote", yahoo)
    now = [100_000.0]
    service = collector.Collector(clock=lambda: now[0])
    for _ in range(3):
        service.broker_tick()
        service.yahoo_tick()
        # Every provider call in a tick receives that tick's clock.
        assert set(clocks) == {now[0]}
        clocks.clear()
        now[0] += market_providers.BROKER_QUOTE_INTERVAL
    # One batched broker call per tick; Yahoo once per symbol until its own interval passes.
    assert batches == [["AAPL", "NVDA"]] * 3
    assert [args for kind, args in providers if kind == "quote"] == [("AAPL",), ("NVDA",)]
    now[0] = 100_000.0 + market_providers.YAHOO_QUOTE_INTERVAL
    service.yahoo_tick()
    assert [args for kind, args in providers if kind == "quote"] == [("AAPL",), ("NVDA",)] * 2


def test_quote_tick_without_broker_never_calls_robinhood(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import market_providers

    def disconnected(now: float) -> bool:
        return False

    def broker(tickers: list[str], *, now: float) -> None:
        pytest.fail("Robinhood must not be called without a connected broker")

    monkeypatch.setattr(market_providers, "_broker_connected", disconnected)
    monkeypatch.setattr(market_providers, "refresh_broker_quotes", broker)
    service = collector.Collector(clock=lambda: 100_000)
    service.broker_tick()
    service.yahoo_tick()
    assert [kind for kind, _ in providers] == ["quote"]


def test_broker_batch_failure_does_not_stop_yahoo_quotes(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from api import market_providers

    def connected(now: float) -> bool:
        return True

    def broker(tickers: list[str], *, now: float) -> None:
        raise RuntimeError("RATE_LIMITED")

    monkeypatch.setattr(market_providers, "_broker_connected", connected)
    monkeypatch.setattr(market_providers, "refresh_broker_quotes", broker)
    service = collector.Collector(clock=lambda: 100_000)
    service.broker_tick()
    service.yahoo_tick()
    assert [kind for kind, _ in providers] == ["quote"]
    assert "Broker quote collection failed: RATE_LIMITED" in caplog.messages


def test_quote_ticks_cover_symbols_read_outside_the_watchlist(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index-tape and searched tickers keep updating after their first read, not just the watchlist."""
    from api import market_providers

    batches: list[list[str]] = []

    def connected(now: float) -> bool:
        return True

    def broker(tickers: list[str], *, now: float) -> None:
        batches.append(list(tickers))

    monkeypatch.setattr(market_providers, "_broker_connected", connected)
    monkeypatch.setattr(market_providers, "refresh_broker_quotes", broker)
    market.mark_error("quote", "^TNX", now=1, error="offline")
    market.mark_ok("broker_quote", "TSLA", now=1)
    service = collector.Collector(clock=lambda: 100_000)
    service.broker_tick()
    service.yahoo_tick()
    assert batches == [["AAPL", "TSLA", "^TNX"]]
    assert [args for kind, args in providers if kind == "quote"] == [("AAPL",), ("TSLA",), ("^TNX",)]


def test_finra_snapshots_and_error_payloads(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import analyst_client, finra_client

    monkeypatch.setattr(collector, "get_finra_client_id", lambda: "id")
    monkeypatch.setattr(collector, "get_finra_client_secret", lambda: "secret")

    def records(symbol: str) -> dict[str, object]:
        return {"records": [symbol]}

    def estimates_error(_symbol: str) -> dict[str, object]:
        return {"error": "no estimates"}

    monkeypatch.setattr(finra_client, "get_short_interest", records)
    monkeypatch.setattr(finra_client, "get_reg_sho_volume", records)
    monkeypatch.setattr(analyst_client, "get_analyst_estimates", estimates_error)
    collector.Collector(clock=lambda: 100_000).cycle()
    for kind in ("finra_short_interest", "finra_reg_sho"):
        assert market.read_snapshots(kind, "AAPL", 1) == [("1970-01-02", {"records": ["AAPL"]})]
    assert market.read_snapshots("estimates", "AAPL", 1) == []
    state = market.refresh_state("estimates", "AAPL")
    assert state is not None
    assert state["last_error"] == "no estimates"


def test_global_google_trends_uses_guarded_us_daily_call(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.google_data import trends

    calls: list[dict[str, object]] = []

    def enabled(source: str) -> bool:
        return source == "bigquery"

    monkeypatch.setattr(collector, "google_source_enabled", enabled)

    def collect(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "ok", "observations": []}

    monkeypatch.setattr(trends, "collect_trends", collect)
    service = collector.Collector(clock=lambda: 100_000)
    service.cycle()
    service.cycle()
    assert calls == [
        {"start_date": "1970-01-02", "end_date": "1970-01-02", "geos": ["US"], "data_root": isolated_market}
    ]
    assert market.read_snapshots("google_trends", "", 1) == [("1970-01-02", {"status": "ok", "observations": []})]


def test_thread_stops_promptly_and_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, brokered, yahooed = threading.Event(), threading.Event(), threading.Event()
    release = threading.Event()

    def slow_yahoo() -> None:
        yahooed.set()
        release.wait(5)

    service = collector.Collector()
    monkeypatch.setattr(service, "cycle", entered.set)
    monkeypatch.setattr(service, "broker_tick", brokered.set)
    monkeypatch.setattr(service, "yahoo_tick", slow_yahoo)
    service.start()
    thread, quote_threads = service._thread, list(service._quote_threads)
    assert thread is not None
    assert [t.name for t in quote_threads] == ["market-quotes-broker", "market-quotes-yahoo"]
    service.start()
    assert (service._thread, service._quote_threads) == (thread, quote_threads)
    # A Yahoo sweep stuck mid-call never holds up the broker poll.
    assert entered.wait(1) and yahooed.wait(1) and brokered.wait(1)
    assert all(t.daemon for t in (thread, *quote_threads))
    release.set()
    service.stop()
    assert not any(t.is_alive() for t in (thread, *quote_threads))


def test_quote_loop_survives_a_failing_tick(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    service = collector.Collector()
    ticks: list[int] = []

    def tick() -> None:
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("boom")
        service._stop.set()

    waits: list[float | None] = []

    def no_wait(seconds: float | None) -> bool:
        waits.append(seconds)
        return False

    monkeypatch.setattr(service._stop, "wait", no_wait)
    service._quote_loop(tick)
    assert len(ticks) == 2
    # Each tick waits one broker interval, so both quote loops tick every 5 s.
    from api.market_providers import BROKER_QUOTE_INTERVAL

    assert waits == [BROKER_QUOTE_INTERVAL, BROKER_QUOTE_INTERVAL]
    assert "Market quote tick failed" in caplog.messages


def test_injected_sleeper_controls_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    service = collector.Collector(sleeper=lambda seconds: (delays.append(seconds), service._stop.set()))
    monkeypatch.setattr(service, "cycle", lambda: None)
    service._loop()
    assert delays == [60]


def test_retention_cutoff_and_symlinks(isolated_market: Path) -> None:
    root = isolated_market
    now = datetime(2026, 10, 31, tzinfo=UTC).timestamp()
    cutoff = now - 30 * collector.DAY
    outside = root / "keep"
    outside.mkdir()
    (outside / "safe").write_text("keep")
    for folder in ("news", "briefing"):
        (root / folder).mkdir()
        for day in ("2026-09-30", "2026-10-01", "2026-10-02"):
            path = root / folder / (day if folder == "news" else day + ".json")
            if folder == "news":
                path.mkdir()
                (path / "story.md").write_text("news")
            else:
                path.write_text("{}")
        (root / folder / "2020-01-01").symlink_to(outside, target_is_directory=True)
    records = [
        raw_archive.archive("viewer", "capture", "key", str(i).encode(), url="u", retrieved_at=stamp, root=root / "raw")
        for i, stamp in enumerate(("2026-09-30T23:59:59Z", "2026-10-01T00:00:00Z", "2026-10-02T00:00:00Z"))
    ]
    (root / "raw/viewer/capture/link").symlink_to(outside, target_is_directory=True)
    market.save_news(
        "AAPL",
        [
            {"title": "old", "url": "old", "published_at": cutoff - 1},
            {"title": "boundary", "url": "boundary", "published_at": cutoff},
        ],
        now=now,
    )
    market.save_quote("AAPL", 10.0, cutoff - collector.DAY, source="robinhood")
    market.save_quote("AAPL", 11.0, cutoff, source="robinhood")
    market.save_snapshot("rh_options", "AAPL", "2026-09-01", {"calls": []})
    collector.retain_news(now=now)
    assert [r["title"] for r in market.read_news("AAPL", 10)] == ["boundary"]
    assert market.latest_quote("AAPL", "robinhood") == (11.0, cutoff)
    assert market.read_snapshots("rh_options", "AAPL", 5) == []
    for folder in ("news", "briefing"):
        suffix = "" if folder == "news" else ".json"
        assert not (root / folder / ("2026-09-30" + suffix)).exists()
        assert (root / folder / ("2026-10-01" + suffix)).exists()
        assert (root / folder / "2020-01-01").is_symlink()
    assert not records[0].manifest_path.exists()
    assert not records[0].payload_path.exists()
    assert all(r.manifest_path.exists() and r.payload_path.exists() for r in records[1:])
    assert (outside / "safe").read_text() == "keep"


def test_capture_retention_ignores_bad_manifests_and_symlink_roots(isolated_market: Path) -> None:
    root = isolated_market
    outside = root / "outside"
    outside.mkdir()
    (outside / "old.manifest.json").write_text(json.dumps({"retrieved_at": "2020-01-01T00:00:00Z"}))
    (root / "raw/viewer").mkdir(parents=True)
    (root / "raw/viewer/capture").symlink_to(outside, target_is_directory=True)
    collector.retain_news(now=2_000_000_000)
    assert (outside / "old.manifest.json").exists()


def test_market_routes_without_lifespan(isolated_market: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from api import main

    calls: list[str] = []
    fake = SimpleNamespace(
        start=lambda: pytest.fail("unexpected startup"),
        stop=lambda: None,
        request_refresh=lambda: calls.append("refresh"),
        status=lambda: {"running": False, "last_cycle_at": None, "oldest_as_of": None},
    )
    monkeypatch.setattr(main, "collector", fake)
    client = TestClient(main.app)
    assert client.get("/api/market/watchlist").json() == {"symbols": ["AAPL"]}
    assert client.put("/api/market/watchlist", json={"symbols": [" nvda ", "NVDA", "BRK.B"]}).json() == {
        "symbols": ["NVDA", "BRK.B"]
    }
    for symbols in (["BAD SYMBOL"], [""], [3], ["A"] * 51):
        assert client.put("/api/market/watchlist", json={"symbols": symbols}).status_code in (400, 422)
    assert client.put("/api/market/watchlist", json={"symbols": []}).json() == {"symbols": []}
    response = client.post("/api/market/refresh")
    assert response.status_code == 202
    assert response.json() == {"status": "queued"}
    assert calls == ["refresh"]
    assert client.get("/api/market/status").json()["running"] is False


@pytest.mark.parametrize("disabled", [False, True])
def test_lifespan_collector_opt_out(monkeypatch: pytest.MonkeyPatch, disabled: bool) -> None:
    from api import main

    calls: list[str] = []
    monkeypatch.setenv("STOCKBOT_COLLECTOR", "0" if disabled else "1")
    monkeypatch.setattr(
        main, "collector", SimpleNamespace(start=lambda: calls.append("start"), stop=lambda: calls.append("stop"))
    )
    with TestClient(main.app):
        assert calls == ([] if disabled else ["start"])
    assert calls == ([] if disabled else ["start", "stop"])


@pytest.mark.parametrize("relative", ["news", "briefing", "raw", "raw/viewer", "raw/viewer/capture"])
def test_retention_refuses_symlink_ancestors(isolated_market: Path, relative: str) -> None:
    outside = isolated_market / "outside"
    outside.mkdir()
    old = outside / "2020-01-01"
    old.mkdir()
    (outside / "2020-01-01.json").write_text("{}")
    (old / "old.manifest.json").write_text('{"retrieved_at":"2020-01-01T00:00:00Z"}')
    link = isolated_market / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    before = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    collector.retain_news(now=2_000_000_000)
    assert {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()} == before
    assert old.is_dir()


def test_retention_keeps_malformed_dates_and_non_json_briefings(
    isolated_market: Path, retention_entry_order: None
) -> None:
    news = isolated_market / "news"
    news.mkdir()
    (news / "not-a-date").mkdir()
    (news / "2020-01-01").write_text("not a directory")
    (news / "2020-01-02").mkdir()
    briefing = isolated_market / "briefing"
    briefing.mkdir()
    (briefing / "not-a-date.json").write_text("{}")
    (briefing / "2020-01-01.txt").write_text("keep")
    (briefing / "2020-01-01.json").mkdir()
    (briefing / "2020-01-02.json").write_text("{}")
    collector.retain_news(now=2_000_000_000)
    assert sorted(p.name for p in news.iterdir()) == ["2020-01-01", "not-a-date"]
    assert sorted(p.name for p in briefing.iterdir()) == ["2020-01-01.json", "2020-01-01.txt", "not-a-date.json"]


@pytest.mark.parametrize("manifest", ["{}", "not json", "[]", '{"retrieved_at":null}', '{"retrieved_at":"2020-01-01"}'])
def test_retention_preserves_invalid_capture_evidence(
    isolated_market: Path, manifest: str, caplog: pytest.LogCaptureFixture
) -> None:
    key = isolated_market / "raw/viewer/capture/key"
    key.mkdir(parents=True)
    path = key / "story.manifest.json"
    path.write_text(manifest)
    payload = key / "story.json"
    payload.write_text("evidence")
    collector.retain_news(now=2_000_000_000)
    assert path.read_text() == manifest
    assert payload.read_text() == "evidence"
    if manifest != '{"retrieved_at":"2020-01-01"}':
        assert f"Cannot read capture manifest {path}" in caplog.messages


@pytest.mark.parametrize("force", [False, True])
def test_stopped_collector_never_runs_due_job(isolated_market: Path, force: bool) -> None:
    service = collector.Collector()
    service.stop()
    service._job("quote", "AAPL", 300, lambda: pytest.fail("stopped job ran"), now=1000, force=force)
    assert market.refresh_state("quote", "AAPL") is None


def test_job_records_error_payload_and_retries_at_latest_attempt_boundary(
    isolated_market: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service = collector.Collector()
    market.mark_ok("quote", "AAPL", "test", now=100)
    market.mark_error("quote", "AAPL", "test", now=200, error="older failure")
    calls: list[int] = []

    def action() -> dict[str, str]:
        calls.append(1)
        return {"error": "provider failed"}

    service._job("quote", "AAPL", 300, action, now=499, force=False, params="test")
    assert calls == []
    service._job("quote", "AAPL", 300, action, now=500, force=False, params="test")
    assert calls == [1]
    assert market.refresh_state("quote", "AAPL", "test") == {
        "covered_from": None,
        "last_ok_at": 100,
        "last_error": "provider failed",
        "last_error_at": 500,
    }
    service._job("quote", "AAPL", 300, action, now=501, force=True, params="test")
    assert calls == [1, 1]
    assert caplog.messages == ["Market collection failed for quote/AAPL/test: provider failed"] * 2


def test_capture_retention_preserves_symlinks_and_only_removes_expired_pair(
    isolated_market: Path, retention_entry_order: None
) -> None:
    capture = isolated_market / "raw/viewer/capture"
    key = capture / "key"
    key.mkdir(parents=True)
    (capture / "not-a-key").write_text("keep")
    evidence = isolated_market / "evidence"
    evidence.write_text('{"retrieved_at":"2020-01-01T00:00:00Z"}')
    manifest_link = key / "linked.manifest.json"
    manifest_link.symlink_to(evidence)
    manifest = key / "old.manifest.json"
    manifest.write_text(evidence.read_text())
    payload_link = key / "old.json"
    payload_link.symlink_to(evidence)
    missing_payload = key / "missing.manifest.json"
    missing_payload.write_text(evidence.read_text())
    outside_key = isolated_market / "outside-key"
    outside_key.mkdir()
    outside_manifest = outside_key / "safe.manifest.json"
    outside_manifest.write_text(evidence.read_text())
    (capture / "linked-key").symlink_to(outside_key, target_is_directory=True)
    collector.retain_news(now=2_000_000_000)
    assert manifest_link.is_symlink()
    assert payload_link.is_symlink()
    assert evidence.read_text() == '{"retrieved_at":"2020-01-01T00:00:00Z"}'
    assert not manifest.exists()
    assert not missing_payload.exists()
    assert (capture / "not-a-key").read_text() == "keep"
    assert outside_manifest.exists()


def test_retention_continues_after_missing_news_root(isolated_market: Path) -> None:
    briefing = isolated_market / "briefing"
    briefing.mkdir()
    expired = briefing / "2020-01-01.json"
    expired.write_text("{}")
    collector.retain_news(now=2_000_000_000)
    assert not expired.exists()


def test_broker_jobs_run_only_when_connected(
    isolated_market: Path, providers: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api import market_providers

    connected = [False]

    def is_connected(now: float) -> bool:
        return connected[0]

    def record(kind: str) -> Callable[..., None]:
        def run(*args: object, **kwargs: object) -> None:
            providers.append((kind, args))

        return run

    monkeypatch.setattr(market_providers, "_broker_connected", is_connected)
    monkeypatch.setattr(market_providers, "refresh_options", record("options"))
    collector.Collector(clock=lambda: 100_000).cycle()
    assert not [kind for kind, _ in providers if kind == "options"]
    connected[0] = True
    providers.clear()
    collector.Collector(clock=lambda: 200_000).cycle()
    # Broker quotes run in the quote loop; the main cycle only adds daily option chains.
    assert [kind for kind, _ in providers if kind in ("broker_quote", "options")] == ["options"]


def test_job_default_parameters_share_refresh_state(isolated_market: Path) -> None:
    service = collector.Collector()
    service._job("quote", "AAPL", 300, dict, now=1000, force=False)
    assert market.refresh_state("quote", "AAPL") == {
        "covered_from": None,
        "last_ok_at": 1000,
        "last_error": None,
        "last_error_at": None,
    }
    service._job("quote", "AAPL", 300, lambda: pytest.fail("not due"), now=1001, force=False)


def test_retention_uses_utc_dates_in_non_utc_timezone(isolated_market: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    news = isolated_market / "news"
    news.mkdir()
    boundary = news / "2026-10-01"
    boundary.mkdir()
    now = datetime(2026, 10, 31, tzinfo=UTC).timestamp()
    with monkeypatch.context() as patch:
        patch.setenv("TZ", "JST-9")
        time.tzset()
        try:
            collector.retain_news(now=now)
        finally:
            patch.undo()
            time.tzset()
    assert boundary.exists()
