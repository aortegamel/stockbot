"""Market providers with persistent history and store-first UI reads.

Yahoo throttles use the shared negative cache. Stale reads never fetch.
Only cold reads and explicit collector refreshes fetch market data.
Connected Robinhood quotes stay on the read path and never enter history.
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from app import cache
from app.analyst_client import REQUEST_TIMEOUT_SECONDS
from app.analyst_client import _ensure_session as _yahoo_session
from app.config import YAHOO_QUERY_BASE
from app.storage import market

if TYPE_CHECKING:
    from app.thesis.models import JSONValue

logger = logging.getLogger(__name__)

_YAHOO_SEARCH_BASE = "https://query2.finance.yahoo.com"
_GNEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US%3Aen"
# Per-ticker heads: Yahoo headlines (?s=SYM) + Seeking Alpha combined ticker
# feed. Topic-level tails (CNBC top news, MarketWatch top stories) carry no
# ticker — filtered by headline/company mention at merge time.
_YAHOO_HEADLINES_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_SEEKING_ALPHA_TICKER_URL = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"
_CNBC_TOP_NEWS_URL = "https://www.cnbc.com/id/100003114/device/rss/rss.html"
_MARKETWATCH_TOP_URL = "http://feeds.marketwatch.com/marketwatch/topstories/"
_SEC_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
    "&type=&dateb=&owner=include&count={n}&output=atom"
)

_TTL_CANDLES = 900
_TTL_UNIVERSE = 24 * 3600
_TTL_NEWS = 900
_TTL_FILINGS = 3600
_FILINGS_PAGE = 40  # the existing per-request ceiling; every cold fetch pulls the whole page
_NEG_TTL = 60
# newsquant fetch() runs trafilatura over every item (~120s/ticker in probe):
# never on the request hot path — daemon thread + join timeout, partial wins.
_NEWSQUANT_TIMEOUT = 25.0

_GATEWAY: object | None = None


def _gateway():
    """Process-local SourceGateway: SEC universe + CIK resolution, cached."""
    from app.data_sources import SourceGateway

    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = SourceGateway()
    gateway: SourceGateway = _GATEWAY  # type: ignore[assignment]
    return gateway


class _Throttle(Exception):
    """Yahoo 429: back off instead of retrying hot."""


def _http_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    """GET via the right shared transport: Yahoo session for Yahoo, stdlib elsewhere."""
    if "finance.yahoo.com" in url:
        resp = _yahoo_session().get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After")
            raise _Throttle(f"Yahoo throttled (429){f', retry after {retry}s' if retry else ''}")
        resp.raise_for_status()
        return resp.content
    import urllib.request

    from app.sec.client import ensure_identity

    ensure_identity()
    import os

    contact = os.getenv("SEC_EDGAR_IDENTITY", "Stockbot stockbot@example.com")
    base = {"User-Agent": contact}
    if headers:
        base.update(headers)
    req = urllib.request.Request(url, headers=base)
    with urllib.request.urlopen(req, timeout=20) as opened:
        data: bytes = opened.read()
        return data


def _universe_sort_key(entry: dict[str, object]) -> str:
    return str(entry["sym"])


def _news_sort_key(item: dict[str, object]) -> float:
    """Epoch seconds of a news item's date; unparseable sorts oldest."""
    return _epoch(item.get("published"))


def _epoch(raw: object) -> float:
    """Epoch seconds of an RFC 822 or ISO date; 0.0 when unparseable."""
    from email.utils import parsedate_to_datetime

    if not isinstance(raw, str) or not raw:
        return 0.0
    try:
        return parsedate_to_datetime(raw).timestamp()
    except TypeError, ValueError:
        pass
    try:
        iso = raw.replace("Z", "+00:00") if raw.endswith(("Z", "z")) else raw
        return datetime.fromisoformat(iso).timestamp()
    except TypeError, ValueError:
        return 0.0


def _neg(key: str, message: str) -> dict[str, object]:
    """Cache a Yahoo throttle envelope briefly so bursts back off."""
    out: dict[str, object] = {"error": message}
    cache.set(f"neg:{key}", out)
    return out


def _neg_hit(key: str) -> dict[str, object] | None:
    hit = cache.get(f"neg:{key}", ttl=_NEG_TTL)
    return hit if isinstance(hit, dict) else None


def _chart_result(payload: Mapping[str, object]) -> dict[str, object]:
    """Read the first chart result. For empty results, use Yahoo's error description when available."""
    empty: dict[str, object] = {}
    chart_obj = payload.get("chart")
    chart: dict[str, object] = chart_obj if isinstance(chart_obj, dict) else empty
    results_obj = chart.get("result")
    results: list[object] = results_obj if isinstance(results_obj, list) else []
    if not results:
        err_obj = chart.get("error")
        err: dict[str, object] = err_obj if isinstance(err_obj, dict) else empty
        detail = err.get("description")
        raise ValueError(detail if isinstance(detail, str) else "no data")
    first = results[0]
    return first if isinstance(first, dict) else empty


def _quote_series(result: Mapping[str, object]) -> dict[str, list[object]]:
    """Read the OHLCV arrays. Replace absent or malformed arrays with empty lists."""
    empty: dict[str, object] = {}
    ind_obj = result.get("indicators")
    indicators: dict[str, object] = ind_obj if isinstance(ind_obj, dict) else empty
    quotes_obj = indicators.get("quote")
    quotes: list[object] = quotes_obj if isinstance(quotes_obj, list) else []
    head = quotes[0] if quotes and isinstance(quotes[0], dict) else empty
    series: dict[str, list[object]] = {}
    for field in ("open", "high", "low", "close", "volume"):
        values = head.get(field)
        series[field] = values if isinstance(values, list) else []
    return series


def _candle_rows(stamps: list[object], series: Mapping[str, list[object]]) -> list[dict[str, object]]:
    """Keep bars with closes. Use the close for absent prices and zero for absent volume."""
    opens, highs, lows = series["open"], series["high"], series["low"]
    closes, vols = series["close"], series["volume"]
    candles: list[dict[str, object]] = []
    for i, ts in enumerate(stamps):
        close = closes[i] if i < len(closes) else None
        if close is None:
            continue
        candles.append(
            {
                "t": ts,
                "open": opens[i] if i < len(opens) else close,
                "high": highs[i] if i < len(highs) else close,
                "low": lows[i] if i < len(lows) else close,
                "close": close,
                "vol": vols[i] if i < len(vols) and vols[i] is not None else 0,
            }
        )
    return candles


def _parse_candles(ticker: str, payload: Mapping[str, object], range_: str, interval: str) -> dict[str, object]:
    result = _chart_result(payload)
    stamps_obj = result.get("timestamp")
    stamps: list[object] = stamps_obj if isinstance(stamps_obj, list) else []
    series = _quote_series(result)
    meta_obj = result.get("meta")
    meta: dict[str, object] = meta_obj if isinstance(meta_obj, dict) else {}
    candles = _candle_rows(stamps, series)
    if not candles:
        raise ValueError("all bars empty")
    return {
        "sym": ticker,
        "range": range_,
        "interval": interval,
        "price": meta.get("regularMarketPrice"),
        "delayed": True,
        "candles": candles,
    }


def _range_start(range_: str, now: float) -> float:
    """Map a Yahoo range to its requested UTC start, independent of available bars."""
    if range_ == "max":
        return 0.0
    moment = datetime.fromtimestamp(now, UTC)
    if range_ == "ytd":
        return moment.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    if range_ in ("1d", "5d"):
        return now - int(range_[:-1]) * 86400
    import calendar

    months = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12, "2y": 24, "5y": 60, "10y": 120}
    offset = months[range_]
    year, month = divmod(moment.year * 12 + moment.month - 1 - offset, 12)
    month += 1
    return moment.replace(year=year, month=month, day=min(moment.day, calendar.monthrange(year, month)[1])).timestamp()


def _freshness(kind: str, ticker: str, ttl: float, now: float, params: str = "") -> dict[str, object]:
    state = market.refresh_state(kind, ticker, params) or {}
    ok = state.get("last_ok_at")
    failed = state.get("last_error_at")
    recent_error = isinstance(failed, (int, float)) and (not isinstance(ok, (int, float)) or failed >= ok)
    out: dict[str, object] = {
        "as_of": _iso(float(ok)) if isinstance(ok, (int, float)) else None,
        "stale": not isinstance(ok, (int, float)) or now - ok >= ttl or recent_error,
    }
    if recent_error:
        out["error"] = state.get("last_error")
    return out


def refresh_candles(ticker: str, interval: str, range_: str, *, now: float) -> None:
    """Fetch Yahoo bars and preserve the requested coverage after a successful write."""
    ticker = ticker.strip().upper()
    key = f"candles:{ticker}:{range_}:{interval}"
    try:
        neg = _neg_hit(key)
        if neg:
            raise _Throttle(str(neg["error"]))
        qs = urllib.parse.urlencode({"range": range_, "interval": interval})
        raw = _http_bytes(f"{YAHOO_QUERY_BASE}/v8/finance/chart/{ticker}?{qs}")
        out = _parse_candles(ticker, json.loads(raw), range_, interval)
        market.save_bars(ticker, interval, _bar_rows(out))
        market.mark_ok("candles", ticker, interval, now=now, covered_from=_range_start(range_, now))
    except Exception as exc:
        message = str(exc) if str(exc).startswith("candles unavailable") else f"candles unavailable for {ticker}: {exc}"
        if isinstance(exc, _Throttle):
            _neg(key, message)
        market.mark_error("candles", ticker, interval, now=now, error=message)
        raise RuntimeError(message) from exc


def _needs_backfill(state: Mapping[str, object], since: float) -> bool:
    """Refresh only when successful coverage does not include the requested start."""
    covered = state.get("covered_from")
    return state.get("last_ok_at") is None or not isinstance(covered, (int, float)) or since < covered


def get_candles(sym: str, range_: str = "6mo", interval: str = "1d", *, now: float | None = None) -> dict[str, object]:
    """Return stored bars. Fetch only missing coverage, never expired coverage."""
    ticker = sym.strip().upper()
    if not ticker:
        return {"error": "empty symbol"}
    now = time.time() if now is None else now
    try:
        since = _range_start(range_, now)
    except KeyError:
        return {"error": f"unsupported range: {range_}"}
    state = market.refresh_state("candles", ticker, interval) or {}
    if _needs_backfill(state, since):
        try:
            refresh_candles(ticker, interval, range_, now=now)
        except RuntimeError as exc:
            if not market.read_bars(ticker, interval, None):
                return {"sym": ticker, "error": str(exc)}
    rows = market.read_bars(ticker, interval, since)
    quote = _best_quote(ticker)
    return {
        "sym": ticker,
        "range": range_,
        "interval": interval,
        "price": quote[0] if quote else rows[-1]["close"] if rows else None,
        "delayed": True,
        "candles": rows,
        **_freshness("candles", ticker, _TTL_CANDLES, now, interval),
    }


def search_tickers(query: str, limit: int = 10) -> dict[str, object]:
    """Yahoo search-as-you-type; validates any US ticker the UI can render."""
    q = query.strip()
    if not q:
        return {"query": "", "hits": []}
    try:
        qs = urllib.parse.urlencode({"q": q, "quotesCount": max(1, min(limit, 20))})
        raw = _http_bytes(
            f"{_YAHOO_SEARCH_BASE}/v1/finance/search?{qs}",
            headers={"Accept": "application/json"},
        )
        payload = json.loads(raw)
        quotes = payload.get("quotes") if isinstance(payload, dict) else None
        hits = [
            {
                "sym": item.get("symbol"),
                "name": item.get("shortname") or item.get("longname"),
                "exchange": item.get("exchange"),
                "type": item.get("quoteType"),
            }
            for item in (quotes or [])
            if isinstance(item, dict) and item.get("symbol")
        ]
        return {"query": q, "hits": hits[:limit]}
    except _Throttle as e:
        logger.warning("search throttled for %s: %s", q, e)
        return {"query": q, "hits": [], "error": f"search unavailable: {e}"}
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("search failed for %s: %s", q, e)
        return {"query": q, "hits": [], "error": f"search unavailable: {e}"}


def get_universe() -> dict[str, object]:
    """SEC ticker -> CIK + name universe via the repo's own gateway."""
    key = "market_universe"
    hit = cache.get(key, ttl=_TTL_UNIVERSE)
    if isinstance(hit, dict):
        return hit
    try:
        raw = _http_bytes(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"Accept": "application/json"},
        )
        payload = json.loads(raw)
        entries: list[dict[str, object]] = [
            {"sym": row["ticker"], "cik": int(row["cik_str"]), "name": row["title"]}
            for row in payload.values()
            if isinstance(row, dict) and row.get("ticker") and row.get("cik_str")
        ]
        entries.sort(key=_universe_sort_key)
        out: dict[str, object] = {"count": len(entries), "tickers": entries}
        cache.set(key, out)
        return out
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("universe failed: %s", e)
        return {"count": 0, "tickers": [], "error": f"universe unavailable: {e}"}


def resolve_identity(sym: str) -> dict[str, object]:
    """Ticker -> SEC entity/security via SourceGateway + domain resolver (PIT now)."""
    from app.domain.market.identity import resolve_ticker_aliases

    ticker = sym.strip().upper()
    try:
        aliases = _gateway().ticker_candidates(ticker, datetime.now(UTC))
        res = resolve_ticker_aliases(ticker, aliases, as_of=datetime.now(UTC))
        return {
            "sym": ticker,
            "resolved": res.resolved,
            "entity_id": res.entity_id,
            "security_id": res.security_id,
            "method": res.resolution_method,
        }
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("identity failed for %s: %s", ticker, e)
        return {"sym": ticker, "resolved": False, "error": f"identity unavailable: {e}"}


def cik_for_ticker(sym: str) -> int | None:
    """CIK from the SEC company universe, else the SEC fund registry; None when unmapped.

    Funds/ETFs absent from company_tickers.json (QQQ) resolve through
    edgartools' ``find_mutual_fund_cik`` (company_tickers_mf.json), so the
    trust's own CIK wins, never a sponsor. Lookup failures raise.
    """
    want = sym.strip().upper()
    universe = get_universe()
    tickers = universe.get("tickers")
    for row in tickers if isinstance(tickers, list) else []:
        if isinstance(row, dict) and row.get("sym") == want:
            cik = row.get("cik")
            return int(cik) if isinstance(cik, int) else None
    from edgar.reference.tickers import find_mutual_fund_cik

    from app.sec.client import ensure_identity

    ensure_identity()
    fund_cik = find_mutual_fund_cik(want)
    if fund_cik is not None:
        return int(fund_cik)
    if universe.get("error"):
        raise RuntimeError(str(universe["error"]))
    return None


def _rss_items(raw: bytes, source: str) -> list[dict[str, object]]:
    """RSS <item> rows -> canonical news items; bad XML yields []."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    items: list[dict[str, object]] = []
    for row in root.iter("item"):
        title = (row.findtext("title") or "").strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": (row.findtext("link") or "").strip(),
                "published": (row.findtext("pubDate") or "").strip(),
                "source": (row.findtext("source") or "").strip() or source,
            }
        )
    return items


def _mentions_ticker(title: str, ticker: str) -> bool:
    """Topic-level filter: keep CNBC/MarketWatch items naming the ticker (word match)."""
    hay = title.upper()
    names = (ticker, "NVIDIA") if ticker == "NVDA" else (ticker,)
    return any(re.search(rf"\b{re.escape(name)}\b", hay) for name in names)


def _fetch_feed(url: str, source: str) -> list[dict[str, object]] | None:
    """Return feed rows, or None when transport fails."""
    try:
        return _rss_items(_http_bytes(url), source)
    except Exception as e:  # noqa: BLE001 - best-effort boundary, one feed never kills the merge
        logger.warning("news feed %s failed: %s", source, e)
        return None


def _newsquant_items(ticker: str) -> list[dict[str, object]]:
    """newsquant enrichment (quality_score/dedup/full-text) in a daemon thread.

    Probe: full fetch() runs ~30-120s (trafilatura per item + spacy/ticker
    extraction). The daemon thread never blocks process exit; the join
    timeout bounds request latency — partial result wins, timeout yields [].
    Gated behind STOCKBOT_NEWSQUANT=1 and a lazy import (dep stays optional).
    """
    import os

    if os.getenv("STOCKBOT_NEWSQUANT") != "1":
        return []
    try:
        from newsquant import Scraper
    except ImportError:
        return []
    holder: list[list[dict[str, object]]] = []

    def run() -> None:
        try:
            arts = Scraper(sources=["yahoofinance", "cnbc"]).fetch(tickers=[ticker], days_back=3, limit=12)
            holder.append(
                [
                    {
                        "title": a.title,
                        "link": a.url,
                        "published": a.published_at.isoformat() if a.published_at else "",
                        "source": f"{a.source_name} · q={a.quality_score:.2f}",
                    }
                    for a in arts
                    if not a.is_duplicate and not a.is_near_duplicate
                ]
            )
        except Exception as e:  # noqa: BLE001 - enrichment is optional, Google RSS stands alone
            logger.warning("newsquant enrichment failed for %s: %s", ticker, e)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=_NEWSQUANT_TIMEOUT)
    return holder[0] if holder else []


def _merge_feeds(ticker: str) -> list[dict[str, object]]:
    """Fetch RSS feeds concurrently. Preserve feed order and filter topic feeds for ticker mentions."""
    q = urllib.parse.quote(f"{ticker} stock")
    feeds = [
        (_GNEWS_URL.format(q=q), "Google News", False),
        (_YAHOO_HEADLINES_URL.format(ticker=ticker), "Yahoo Finance", True),
        (_SEEKING_ALPHA_TICKER_URL.format(ticker=ticker), "Seeking Alpha", False),
        (_CNBC_TOP_NEWS_URL, "CNBC", True),
        (_MARKETWATCH_TOP_URL, "MarketWatch", True),
    ]
    merged: list[dict[str, object]] = []
    succeeded = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(feeds)) as pool:
        results = pool.map(_fetch_feed, [u for u, _s, _t in feeds], [s for _u, s, _t in feeds])
        for (_url, _source, topic_level), items in zip(feeds, results):
            if items is None:
                continue
            succeeded = True
            if topic_level:
                items = [i for i in items if _mentions_ticker(str(i.get("title", "")), ticker)]
            merged.extend(items)
    if not succeeded:
        raise RuntimeError("all news feeds failed")
    return merged


def _dedup_by_link(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Preserve the first item for each link. Use the title when the link is empty."""
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for item in items:
        link = str(item.get("link") or item.get("title"))
        if link in seen:
            continue
        seen.add(link)
        unique.append(item)
    return unique


def refresh_news(ticker: str, *, now: float) -> None:
    """Merge RSS headlines and store every item, independent of the reader's limit."""
    ticker = ticker.strip().upper()
    try:
        merged = _merge_feeds(ticker)
        merged.extend(_newsquant_items(ticker))
        unique = _dedup_by_link(merged)
        market.save_news(
            ticker,
            [
                {
                    "title": item["title"],
                    "url": item.get("link"),
                    "source": item.get("source"),
                    "published_at": _news_sort_key(item) or None,
                }
                for item in unique
            ],
            now=now,
        )
        market.mark_ok("news", ticker, now=now)
    except Exception as exc:
        market.mark_error("news", ticker, now=now, error=f"news unavailable: {exc}")
        raise


def get_news(sym: str, limit: int = 12, *, now: float | None = None) -> dict[str, object]:
    """Return stored headlines with their last successful collection time."""
    ticker = sym.strip().upper()
    if not ticker:
        return {"sym": "", "items": []}
    now = time.time() if now is None else now
    state = market.refresh_state("news", ticker) or {}
    if state.get("last_ok_at") is None:
        try:
            refresh_news(ticker, now=now)
        except Exception as exc:  # noqa: BLE001 - stored data remains readable after failure
            logger.warning("news failed for %s: %s", ticker, exc)
    items = market.read_news(ticker, max(0, limit))
    return {
        "sym": ticker,
        "items": [
            {
                "title": item["title"],
                "link": item["url"],
                "source": item["source"],
                "published": _iso(float(stamp)) if isinstance(stamp := item["published_at"], (int, float)) else "",
                "summary_path": item["summary_path"],
            }
            for item in items
        ],
        **_freshness("news", ticker, _TTL_NEWS, now),
    }


def _atom_filings(raw: bytes) -> list[dict[str, object]]:
    """Parse filing entries. Reject entries without an official filing date in YYYY-MM-DD form."""
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    items: list[dict[str, object]] = []
    for entry in root.findall("a:entry", ns):
        link = entry.find("a:link", ns)
        filed = (entry.findtext("a:content/a:filing-date", default="", namespaces=ns) or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", filed):
            raise ValueError(f"Atom entry without a YYYY-MM-DD filing-date: {filed!r}")
        items.append(
            {
                "title": (entry.findtext("a:title", default="", namespaces=ns) or "").strip(),
                "link": link.get("href", "") if link is not None else "",
                "filed": filed,
                "updated": (entry.findtext("a:updated", default="", namespaces=ns) or "").strip(),
                "summary": (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()[:300],
            }
        )
    return items


def refresh_filings(ticker: str, *, now: float) -> None:
    """Fetch the full SEC filing page and store its CIK with the result."""
    ticker = ticker.strip().upper()
    try:
        try:
            cik = market.get_cik(ticker)
            if cik is None:
                cik = cik_for_ticker(ticker)
        except Exception as exc:
            raise RuntimeError(f"CIK lookup failed: {exc}") from exc
        if cik is None:
            raise ValueError(f"no SEC CIK for {ticker}")
        market.save_cik(ticker, cik)
        raw = _http_bytes(
            _SEC_ATOM_URL.format(cik=f"{cik:010d}", n=_FILINGS_PAGE),
            headers={"Accept": "application/atom+xml"},
        )
        market.save_filings(ticker, cik, _atom_filings(raw))
        market.mark_ok("filings", ticker, now=now)
    except Exception as exc:
        message = str(exc) if str(exc).startswith("no SEC CIK") else f"filings unavailable: {exc}"
        market.mark_error("filings", ticker, now=now, error=message)
        raise RuntimeError(message) from exc


def get_filings(sym: str, limit: int = 10, *, now: float | None = None) -> dict[str, object]:
    """Return saved SEC filings without resolving a saved ticker over the network."""
    ticker = sym.strip().upper()
    now = time.time() if now is None else now
    state = market.refresh_state("filings", ticker) or {}
    if state.get("last_ok_at") is None:
        try:
            refresh_filings(ticker, now=now)
        except RuntimeError:
            pass
    cik = market.get_cik(ticker)
    return {
        "sym": ticker,
        "cik": cik,
        "items": market.read_filings(cik, max(0, limit)) if cik is not None else [],
        **_freshness("filings", ticker, _TTL_FILINGS, now),
    }


# Quote polling intervals. Neither provider publishes a limit, so these sit well below observed ceilings.
# Robinhood MCP: about 4 calls/s sustained, throttling near 8/s (measured by
# github.com/kevin1chun/robinhood-for-agents, docs/official-mcp-tools.md). One batched get_equity_quotes
# call covers the whole watchlist, so 5 s is 0.2 calls/s and leaves room for agent tool calls.
# Yahoo quoteSummary: one call per symbol, about 2,000 requests/hour/IP is the commonly reported
# ceiling. 60 s for a 9-symbol watchlist is 540/hour, leaving room for candles, news, and the agent.
BROKER_QUOTE_INTERVAL = 5
YAHOO_QUOTE_INTERVAL = 60
_QUOTE_SOURCES = {"robinhood": ("broker_quote", BROKER_QUOTE_INTERVAL), "yahoo": ("quote", YAHOO_QUOTE_INTERVAL)}


def refresh_quote(ticker: str, *, now: float) -> None:
    """Collect the public Yahoo quote directly, bypassing valuation's 5-minute price cache."""
    from app import analyst_client

    ticker = ticker.strip().upper()
    try:
        quote = analyst_client.get_quote_price(ticker)
        price = quote.get("price")
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            raise TypeError("no price")
        market.save_quote(ticker, float(price), _epoch(quote.get("retrieved_at")) or now)
        market.mark_ok("quote", ticker, now=now)
    except Exception as exc:
        market.mark_error("quote", ticker, now=now, error=f"quote unavailable: {exc}")
        raise


def refresh_broker_quotes(tickers: list[str], *, now: float) -> None:
    """Collect Robinhood quotes for many tickers in one call. Only call this when the broker is connected."""
    from app.tools import get_equity_quote_prices

    symbols = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    try:
        prices = get_equity_quote_prices(symbols)
    except Exception as exc:
        for symbol in symbols:
            market.mark_error("broker_quote", symbol, now=now, error=f"broker quote unavailable: {exc}")
        raise
    for symbol in symbols:
        if symbol in prices:
            market.save_quote(symbol, prices[symbol], now, source="robinhood")
            market.mark_ok("broker_quote", symbol, now=now)
        else:
            market.mark_error("broker_quote", symbol, now=now, error="broker quote unavailable: no price")


def refresh_options(ticker: str, *, now: float) -> None:
    """Snapshot both Robinhood option chains for the ticker under today's UTC date."""
    from app.tools import get_option_chain

    ticker = ticker.strip().upper()
    try:
        chains = {}
        for kind in ("call", "put"):
            chain = get_option_chain(ticker, kind, limit=30)
            if chain.get("error"):
                raise RuntimeError(str(chain["error"]))
            chains[kind] = chain
        day = datetime.fromtimestamp(now, UTC).date().isoformat()
        market.save_snapshot("rh_options", ticker, day, chains)
        market.mark_ok("rh_options", ticker, now=now)
    except Exception as exc:
        market.mark_error("rh_options", ticker, now=now, error=f"options unavailable: {exc}")
        raise


def _broker_connected(now: float) -> bool:
    from app.config import broker_enabled, get_robinhood_mcp_url

    if not broker_enabled():
        return False
    from app.robinhood.auth import DEFAULT_TOKEN_PATH, OAuthConfig, has_valid_tokens

    oauth = OAuthConfig(get_robinhood_mcp_url())
    return has_valid_tokens(oauth.server_origin, DEFAULT_TOKEN_PATH, now=datetime.fromtimestamp(now, UTC))


def _best_quote(ticker: str) -> tuple[float, float, str] | None:
    """(price, row_ts, source) from the source whose last successful poll is newest.

    A change-only save leaves an old row while the price holds, so confirmation time (refresh_log
    last_ok_at, else the row time) picks the winner, not the newest row.
    """
    best: tuple[float, float, str, float] | None = None
    for source, (kind, _interval) in _QUOTE_SOURCES.items():
        row = market.latest_quote(ticker, source)
        if row is None:
            continue
        ok = (market.refresh_state(kind, ticker) or {}).get("last_ok_at")
        confirmed = max(row[1], float(ok)) if isinstance(ok, (int, float)) else row[1]
        if best is None or confirmed > best[3]:
            best = (row[0], row[1], source, confirmed)
    return best[:3] if best else None


def _attempted_within(kind: str, ticker: str, interval: int, now: float) -> bool:
    """True when this source was polled, successfully or not, within its interval."""
    state = market.refresh_state(kind, ticker) or {}
    attempts = [v for v in (state.get("last_ok_at"), state.get("last_error_at")) if isinstance(v, (int, float))]
    return bool(attempts) and now - max(attempts) < interval


def _cold_fetch_quote(ticker: str, *, connected: bool, now: float) -> None:
    """Fill an empty store: Robinhood first when connected, then Yahoo for symbols Robinhood does not quote.

    Each source is tried at most once per its interval, so 5-second UI polls of an unquotable
    symbol (indexes, futures) never hammer a provider.
    """
    for source, fetch in (
        ("robinhood", lambda: refresh_broker_quotes([ticker], now=now)),
        ("yahoo", lambda: refresh_quote(ticker, now=now)),
    ):
        kind, interval = _QUOTE_SOURCES[source]
        if (source == "robinhood" and not connected) or _attempted_within(kind, ticker, interval, now):
            continue
        try:
            fetch()
        except Exception as exc:  # noqa: BLE001 - an empty store stays readable after a failed cold fetch
            logger.warning("quote failed for %s: %s", ticker, exc)
        if market.latest_quote_any(ticker) is not None:
            return


def get_quote(sym: str, *, now: float | None = None) -> dict[str, object]:
    """Return the latest saved quote from Robinhood or Yahoo, whichever was confirmed most recently.

    Reads call a provider only when nothing is saved yet (see _cold_fetch_quote). After the first
    read, the collector keeps the symbol fresh. stale means the chosen source missed two of its
    polling intervals or its last poll failed.
    """
    ticker = sym.strip().upper()
    if not ticker:
        return {"sym": "", "price": None, "retrieved_at": None}
    now = time.time() if now is None else now
    connected = _broker_connected(now)
    if market.latest_quote_any(ticker) is None:
        _cold_fetch_quote(ticker, connected=connected, now=now)
    best = _best_quote(ticker)
    source = best[2] if best else ("robinhood" if connected else "yahoo")
    kind, interval = _QUOTE_SOURCES[source]
    return {
        "sym": ticker,
        "price": best[0] if best else None,
        "retrieved_at": _iso(best[1]) if best else None,
        "source": source,
        **_freshness(kind, ticker, 2 * interval, now),
    }


def _candle_window(candles: Mapping[str, object]) -> dict[str, object]:
    """Summarize numeric closes in their original order."""
    rows_obj = candles.get("candles")
    rows: list[object] = rows_obj if isinstance(rows_obj, list) else []
    closes: list[float] = [
        float(r["close"]) for r in rows if isinstance(r, dict) and isinstance(r.get("close"), (int, float))
    ]
    if not closes:
        return {"bars": 0, "last": None, "first": None, "high": None, "low": None}
    summary: dict[str, object] = {
        "bars": len(closes),
        "last": closes[-1],
        "first": closes[0],
        "high": max(closes),
        "low": min(closes),
    }
    if len(closes) > 1 and closes[0]:
        summary["change_pct"] = (closes[-1] - closes[0]) / closes[0] * 100
    return summary


def get_viewer_context(sym: str) -> dict[str, object]:
    """Read-only agent context for whatever ticker the UI is showing.

    Contract: GET /api/viewer-context?sym=NVDA returns one JSON object with
    the viewed symbol, the rendered payloads, and retrieval instants. The
    future agent loop re-fetches it on demand as a plain HTTP/tool call —
    no streaming, no session, no state. Not wired to any agent yet.
    """
    ticker = sym.strip().upper()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    quote = get_quote(ticker)
    candles = get_candles(ticker)
    news = get_news(ticker)
    filings = get_filings(ticker)
    summary = _candle_window(candles)
    news_obj = news.get("items")
    news_items: list[object] = news_obj if isinstance(news_obj, list) else []
    filings_obj = filings.get("items")
    filing_items: list[object] = filings_obj if isinstance(filings_obj, list) else []
    return {
        "sym": ticker,
        "as_of": now,
        "quote": quote,
        "window": summary,
        "news": news_items,
        "filings": filing_items,
        "cik": filings.get("cik"),
        "sources": {
            "quote": quote.get("source", "yahoo"),
            "candles": "Yahoo chart v8, delayed",
            "news": "Google News + Yahoo Finance + Seeking Alpha + CNBC/MarketWatch RSS (newsquant when STOCKBOT_NEWSQUANT=1)",
            "filings": "SEC EDGAR Atom",
        },
        "errors": {
            k: v["error"]
            for k, v in (("quote", quote), ("candles", candles), ("news", news), ("filings", filings))
            if isinstance(v, dict) and v.get("error")
        },
    }


# ---------------------------------------------------------------------------
# Aggregates: sparklines, scan rows, positioning, briefing, feed.
# Every per-symbol call is bounded (symbol cap, filing window + count cap,
# cache TTL); a bad symbol yields its own error object, never a 500.
# ---------------------------------------------------------------------------

MAX_SYMS = 20
_TTL_SPARKLINES = 60
_SPARK_INTERVAL = {"1d": "5m", "5d": "30m"}  # other ranges use daily bars
SCAN_PRESETS = {
    "watchlist": "Watchlist",
    "most-shorted": "Most shorted",
    "insider-buying": "Insider buying (7d)",
    "new-8k": "New 8-K (7d)",
    "movers": "Movers",
}
MARKET_SORTS = ("chg_pct", "volume", "short_pct")
_INDEX_UNIVERSE = ("SPY", "QQQ", "DIA", "IWM")
# insider-buying / new-8k probe watchlist first, then leaderboard names, at most this many symbols.
_SCAN_UNIVERSE_CAP = 20
_MOST_SHORTED = 25
_RECENT_DAYS = 7
_POSITIONING_DAYS = 90
_SEC_RECENT_LIMIT = 20  # filings per symbol per window (each Form 4 is one ownership XML fetch)
_TTL_SEC_RECENT = 900
_TTL_SHORT_BOARD = 6 * 3600  # FINRA short interest publishes twice a month
_SHORT_BOARD_WAIT = 10.0  # cold leaderboard = full FINRA snapshot + company facts; finishes in the background
_BRIEF_NEWS = 3
_FEED_NEWS = 5
_FEED_LIMIT = 200
_SEC_WORKERS = 4
_FEED_KEYS = ("id", "kind", "sym", "title", "link", "at", "source")

_short_board_guard = threading.Lock()
_short_board_worker: threading.Thread | None = None


def parse_syms(raw: str) -> list[str]:
    """Comma list -> upper, deduped, order kept; ValueError past MAX_SYMS."""
    syms = list(dict.fromkeys(s.strip().upper() for s in raw.split(",") if s.strip()))
    if len(syms) > MAX_SYMS:
        raise ValueError(f"at most {MAX_SYMS} symbols, got {len(syms)}")
    return syms


def parse_since(raw: str) -> str:
    """ISO instant -> canonical UTC ISO; ValueError when unparseable."""
    iso = raw.replace("Z", "+00:00") if raw.endswith(("Z", "z")) else raw
    moment = datetime.fromisoformat(iso)
    return _iso((moment if moment.tzinfo else moment.replace(tzinfo=UTC)).timestamp())


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="seconds") if epoch else ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cutoff(days: int) -> str:
    return (datetime.now(UTC).date() - timedelta(days=days)).isoformat()


def _fan[T, R](fn: Callable[[T], R], items: list[T], workers: int = 8) -> list[R]:
    """Ordered concurrent map; one worker per item up to ``workers``."""
    if not items:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))


def _bar_rows(candles: Mapping[str, object]) -> list[dict[str, object]]:
    rows_obj = candles.get("candles")
    return [r for r in rows_obj if isinstance(r, dict)] if isinstance(rows_obj, list) else []


def _spark(candles: Mapping[str, object]) -> dict[str, object]:
    if candles.get("error"):
        return {"error": str(candles["error"])}
    rows = _bar_rows(candles)
    stamps = [r.get("t") for r in rows]
    last = stamps[-1] if stamps else None
    return {
        "t": stamps,
        "closes": [r.get("close") for r in rows],
        "as_of": _iso(float(last)) if isinstance(last, (int, float)) else None,
    }


def get_sparklines(syms: list[str], range_: str = "1d") -> dict[str, object]:
    """Closes-only series per symbol; whole answer cached 60s per (range, syms)."""
    key = f"sparklines:{range_}:{','.join(syms)}"
    hit = cache.get(key, ttl=_TTL_SPARKLINES)
    if isinstance(hit, dict):
        return hit
    interval = _SPARK_INTERVAL.get(range_, "1d")

    def one(sym: str) -> dict[str, object]:
        return _spark(get_candles(sym, range_=range_, interval=interval))

    series = _fan(one, syms)
    out: dict[str, object] = {"syms": dict(zip(syms, series))}
    cache.set(key, out)
    return out


def _build_short_board() -> None:
    """Leaderboard -> cached {settlement_date, pct:{sym: short %}} in rank order (or a neg entry)."""
    from app.analytics.screens import MAX_LIMIT, get_short_interest_leaderboard

    try:
        board = get_short_interest_leaderboard(limit=MAX_LIMIT)
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        board = {"error": f"Short-interest leaderboard is unavailable: {e}"}
    if board.get("error"):
        _neg("short_board", str(board["error"]))
        return
    entries = board.get("entries")
    pct = {
        str(e["ticker"]): e["short_interest_percent"]
        for e in (entries if isinstance(entries, list) else [])
        if isinstance(e, dict) and e.get("ticker")
    }
    cache.set("short_board", {"settlement_date": board.get("settlement_date"), "pct": pct})


def short_board() -> dict[str, object]:
    """Cached short-interest leaderboard map; waits at most _SHORT_BOARD_WAIT for a cold build."""
    global _short_board_worker
    for _ in range(2):
        hit = cache.get("short_board", ttl=_TTL_SHORT_BOARD)
        if isinstance(hit, dict):
            return hit
        neg = _neg_hit("short_board")
        if neg is not None:
            return neg
        with _short_board_guard:
            if _short_board_worker is None or not _short_board_worker.is_alive():
                _short_board_worker = threading.Thread(target=_build_short_board, daemon=True)
                _short_board_worker.start()
            worker = _short_board_worker
        worker.join(timeout=_SHORT_BOARD_WAIT)
        if worker.is_alive():
            break
    return {"error": "short-interest leaderboard is still computing; retry shortly"}


def _short_pct(board: Mapping[str, object]) -> dict[str, object]:
    pct = board.get("pct")
    return pct if isinstance(pct, dict) else {}


def _recent_8k(sym: str, days: int = _RECENT_DAYS) -> tuple[list[dict[str, object]], str | None]:
    """8-K/8-K/A filed in the last ``days`` (cached 15 min); (rows, error)."""
    key = f"sec_recent:v1:8k:{sym}:{days}"
    hit = cache.get(key, ttl=_TTL_SEC_RECENT)
    if isinstance(hit, list):
        return hit, None
    try:
        from app.sec import filings as sec_filings

        found = sec_filings.list_sec_filings(
            sym, forms=["8-K", "8-K/A"], start_date=_cutoff(days), limit=_SEC_RECENT_LIMIT
        )
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("8-K list failed for %s: %s", sym, e)
        return [], f"8-K filings unavailable: {e}"
    out: list[dict[str, object]] = [
        {"accession_no": f.accession_no, "form": f.form, "filed": f.filed_at, "known_at": f.known_at, "link": f.source}
        for f in found
    ]
    cache.set(key, out)
    return out, None


def _recent_insider(sym: str, days: int = _RECENT_DAYS) -> tuple[list[dict[str, object]], str | None]:
    """Form 4/4A transactions filed in the last ``days`` (cached 15 min); (rows, error).

    Not get_insider_activity: it has no date window and truncates transactions
    to its filing limit. Same seams (list_sec_filings + load_ownership +
    normalize_ownership_filing), bounded to the window's filings.
    """
    key = f"sec_recent:v1:form4:{sym}:{days}"
    hit = cache.get(key, ttl=_TTL_SEC_RECENT)
    if isinstance(hit, list):
        return hit, None
    out: list[dict[str, object]] = []
    try:
        from app.sec import filings as sec_filings
        from app.sec import insider

        found = sec_filings.list_sec_filings(sym, forms=["4", "4/A"], start_date=_cutoff(days), limit=_SEC_RECENT_LIMIT)
        for f in found:
            txns = insider.normalize_ownership_filing(
                insider.load_ownership(f.accession_no),
                issuer=f.filer_name,
                form=f.form,
                filed_at=f.filed_at,
                accession_no=f.accession_no,
                document_name=f.primary_document,
                known_at=f.known_at,
            )
            out.extend(
                {
                    "accession_no": t.accession_no,
                    "form": t.form,
                    "filed": t.filed_at,
                    "known_at": t.known_at or t.filed_at,
                    "insider_name": t.insider_name,
                    "kind": t.transaction_kind,
                    "shares": t.shares,
                    "link": f.source,
                }
                for t in txns
            )
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("insider activity failed for %s: %s", sym, e)
        return [], f"insider activity unavailable: {e}"
    cache.set(key, out)
    return out, None


def _last_filing(sym: str) -> dict[str, object] | None:
    """Newest EDGAR Atom entry as {form, filed}; the form is the title head ("8-K - Current report")."""
    items = get_filings(sym, limit=1).get("items")
    first = items[0] if isinstance(items, list) and items and isinstance(items[0], dict) else None
    if first is None:
        return None
    return {"form": str(first.get("title") or "").split(" - ")[0].strip(), "filed": first.get("filed")}


def _market_row(sym: str, pct: Mapping[str, object]) -> dict[str, object]:
    """Last close, day change vs prior daily close, last-bar volume, short %, newest filing."""
    candles = get_candles(sym, range_="5d", interval="1d")
    rows = _bar_rows(candles)
    closes = [float(r["close"]) for r in rows if isinstance(r.get("close"), (int, float))]
    row: dict[str, object] = {
        "sym": sym,
        "last": closes[-1] if closes else None,
        "chg_pct": (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) > 1 and closes[-2] else None,
        "volume": rows[-1].get("vol") if rows else None,
        "short_pct": pct.get(sym),
        "last_filing": _last_filing(sym),
    }
    if candles.get("error"):
        row["error"] = str(candles["error"])
    return row


def _sort_key(field: str) -> Callable[[dict[str, object]], tuple[bool, float]]:
    def key(row: dict[str, object]) -> tuple[bool, float]:
        value = row.get(field)
        return (not isinstance(value, (int, float)), -float(value) if isinstance(value, (int, float)) else 0.0)

    return key


def _error(sym: str, source: str, error: str | None) -> list[dict[str, object]]:
    return [{"sym": sym, "source": source, "error": error}] if error else []


def _probe(
    universe: list[str], fetch: Callable[[str], tuple[list[dict[str, object]], str | None]], source: str
) -> tuple[list[tuple[str, list[dict[str, object]]]], list[dict[str, object]]]:
    results = _fan(fetch, universe, _SEC_WORKERS)
    errors = [e for sym, (_rows, err) in zip(universe, results) for e in _error(sym, source, err)]
    return [(sym, rows) for sym, (rows, _err) in zip(universe, results)], errors


def get_markets(scan: str, watchlist: list[str], sort: str = "chg_pct") -> dict[str, object]:
    """Rows for one preset scan; ``scan``/``sort`` are validated by the route."""
    board = short_board()
    pct = _short_pct(board)
    errors = _error("", "finra_short_interest", str(board["error"]) if board.get("error") else None)
    get_universe()  # warm the CIK map once before the per-symbol fan-out
    if scan == "watchlist":
        universe = watchlist
    elif scan == "most-shorted":
        universe = list(pct)[:_MOST_SHORTED]
    elif scan == "movers":
        universe = list(dict.fromkeys([*watchlist, *_INDEX_UNIVERSE]))
    else:
        candidates = list(dict.fromkeys([*watchlist, *pct]))[:_SCAN_UNIVERSE_CAP]
        if scan == "insider-buying":
            probed, probe_errors = _probe(candidates, _recent_insider, "sec_form4")
            universe = [s for s, rows in probed if any(r.get("kind") == "open_market_purchase" for r in rows)]
        else:
            probed, probe_errors = _probe(candidates, _recent_8k, "sec_8k")
            universe = [s for s, rows in probed if rows]
        errors.extend(probe_errors)
    rows = _fan(functools.partial(_market_row, pct=pct), universe)
    rows.sort(key=_sort_key(sort))
    return {"scan": scan, "sort": sort, "universe": universe, "rows": rows, "errors": errors}


def _threshold(sym: str) -> tuple[bool | None, str | None, str | None]:
    """(listed, newest listing date, error). FINRA "No data found" means not listed."""
    from app import finra_client

    try:
        res = finra_client.get_threshold_securities(sym)
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        return None, None, f"threshold list unavailable: {e}"
    err = res.get("error")
    if err:
        return (False, None, None) if str(err).startswith("No data found") else (None, None, str(err))
    records = res.get("records")
    as_of = res.get("as_of_date")
    return bool(records), str(as_of) if as_of else None, None


def get_positioning(sym: str) -> dict[str, object]:
    """Short %, 90-day open-market insider share sums, threshold-list presence; nulls when unavailable."""
    ticker = sym.strip().upper()
    errors: dict[str, str] = {}
    board = short_board()
    if board.get("error"):
        errors["short_interest"] = str(board["error"])
    short_pct = _short_pct(board).get(ticker)
    txns, insider_err = _recent_insider(ticker, _POSITIONING_DAYS)
    net: dict[str, object] | None = None
    if insider_err:
        errors["insider"] = insider_err
    else:
        buys = sum(int(str(t.get("shares") or 0)) for t in txns if t.get("kind") == "open_market_purchase")
        sells = sum(int(str(t.get("shares") or 0)) for t in txns if t.get("kind") == "open_market_sale")
        net = {"buys": buys, "sells": sells, "net": buys - sells}
    listed, listed_as_of, threshold_err = _threshold(ticker)
    if threshold_err:
        errors["threshold"] = threshold_err
    return {
        "sym": ticker,
        "short_pct": short_pct,
        "short_as_of": board.get("settlement_date") if short_pct is not None else None,
        "insider_net_90d": net,
        "threshold_listed": listed,
        "threshold_as_of": listed_as_of,
        "errors": errors,
    }


def _insider_title(t: Mapping[str, object]) -> str:
    verb = "bought" if t.get("kind") == "open_market_purchase" else "sold"
    shares = t.get("shares")
    size = f" {shares:,} sh" if isinstance(shares, int) else ""
    return f"{t.get('insider_name') or 'Insider'} {verb}{size}"


def _sym_activity(sym: str, news_limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """News + recent 8-K + recent open-market Form 4 rows for one symbol; (items, errors)."""
    news = get_news(sym, limit=news_limit)
    news_obj = news.get("items")
    items: list[dict[str, object]] = [
        {
            "id": f"news:{n.get('link') or n.get('title')}",
            "kind": "news",
            "sym": sym,
            "title": n.get("title"),
            "link": n.get("link"),
            "at": _iso(_epoch(n.get("published"))),
            "source": n.get("source"),
        }
        for n in (news_obj if isinstance(news_obj, list) else [])
        if isinstance(n, dict)
    ]
    errors = _error(sym, "news", str(news["error"]) if news.get("error") else None)
    filings, filings_err = _recent_8k(sym)
    items.extend(
        {
            "id": f"sec:{f['accession_no']}",
            "kind": "filing",
            "sym": sym,
            "title": f"{f['form']} filed by {sym}",
            "link": f["link"],
            "at": _iso(_epoch(f.get("known_at") or f.get("filed"))),
            "source": "SEC EDGAR",
            "form": f["form"],
            "filed": f["filed"],
        }
        for f in filings
    )
    txns, insider_err = _recent_insider(sym)
    items.extend(
        {
            "id": f"insider:{t['accession_no']}:{i}",
            "kind": "insider",
            "sym": sym,
            "title": _insider_title(t),
            "link": t["link"],
            "at": _iso(_epoch(t.get("known_at") or t.get("filed"))),
            "source": "SEC Form 4",
            "form": t["form"],
            "filed": t["filed"],
        }
        for i, t in enumerate(txns)
        if t.get("kind") in ("open_market_purchase", "open_market_sale")
    )
    errors += _error(sym, "sec_8k", filings_err) + _error(sym, "sec_form4", insider_err)
    return items, errors


def _activity(syms: list[str], news_limit: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results = _fan(functools.partial(_sym_activity, news_limit=news_limit), syms, _SEC_WORKERS)
    items = [i for rows, _errs in results for i in rows]
    errors = [e for _rows, errs in results for e in errs]
    return items, errors


def _at_key(item: Mapping[str, object]) -> str:
    return str(item.get("at") or "")


def _newest_first(items: list[dict[str, object]]) -> None:
    items.sort(key=_at_key, reverse=True)


def _briefing_item(item: Mapping[str, object]) -> dict[str, object]:
    out = {k: item[k] for k in ("kind", "sym", "title", "link")}
    if item["kind"] == "news":
        out["published"] = item["at"]
    else:
        out["filed"] = item["filed"]
        out["filing_form"] = item["form"]
    return out


def get_briefing(syms: list[str], refresh: bool = False) -> dict[str, object]:
    """Today's briefing from data/briefing/<date>.json; rebuilt when missing, for other syms, or on refresh."""
    from app.config import get_data_root
    from app.thesis.yaml import atomic_write_json

    day = datetime.now(UTC).date().isoformat()
    root = get_data_root()
    path = root / "briefing" / f"{day}.json"
    if not refresh and path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            logger.warning("briefing %s unreadable, rebuilding: %s", path, e)
            stored = None
        if isinstance(stored, dict) and stored.get("syms") == syms:
            return dict(stored, cached=True)
    items, errors = _activity(syms, _BRIEF_NEWS)
    _newest_first(items)
    body: dict[str, object] = {
        "date": day,
        "syms": syms,
        "items": [_briefing_item(i) for i in items],
        "generated_at": _now_iso(),
        "errors": errors,
    }
    atomic_write_json(path, cast("dict[str, JSONValue]", body), root)
    return dict(body, cached=False)


def _alerts() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Pending thesis triggers as feed items via the thesis routes' repository."""
    from api import theses_routes

    try:
        repo = theses_routes._repo()
        items: list[dict[str, object]] = [
            {
                "id": t.trigger_id,
                "kind": "alert",
                "title": t.summary or t.trigger_type,
                "at": _iso(_epoch(t.created_at)),
                "source": f"thesis:{thesis.slug}",
            }
            for thesis in repo.list_theses()
            for t in repo.load_triggers(thesis.thesis_id, include_processed=False)
        ]
    except (OSError, ValueError, KeyError) as e:
        return [], _error("", "thesis_alerts", f"thesis alerts unavailable: {e}")
    return items, []


def get_feed(syms: list[str], since: str | None = None) -> dict[str, object]:
    """News + 8-K + open-market Form 4 + pending thesis alerts, newest first, strictly after ``since``."""
    items, errors = _activity(syms, _FEED_NEWS)
    alerts, alert_errors = _alerts()
    items += alerts
    if since:
        items = [i for i in items if str(i.get("at") or "") > since]
    _newest_first(items)
    return {
        "items": [{k: i[k] for k in _FEED_KEYS if k in i} for i in items[:_FEED_LIMIT]],
        "as_of": _now_iso(),
        "errors": errors + alert_errors,
    }
