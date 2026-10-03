"""Free market-data providers behind the real-data UI slice.

Every source here is keyless:
- Yahoo chart v8: OHLCV candles for any US ticker (delayed ~15min).
  Reuses the analyst_client session (cookies) + cache.db TTL store —
  no second Yahoo session, no second crumb, no in-process dict.
- Yahoo search v1: ticker validation / search-as-you-type (same session).
- SEC company_tickers via data_sources.SourceGateway: full ticker -> CIK
  universe (10k+ entries) with the repo's own normalization + PIT archive.
- Google News RSS: per-ticker headlines (cache.db TTL).
- SEC EDGAR Atom: per-company filings (needs CIK + contact User-Agent).
- Quote via valuation.get_live_quote: Yahoo quoteSummary with its own
  5-minute price TTL in cache.db.

All fetches are best-effort: transport failure returns a structured
``{"error": ...}`` envelope, never raises into the request handler.
Yahoo 429s back off (Retry-After honored, 60s negative cache) instead of
hammering — first paint during a throttle shows "unavailable", never mocks.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from app import cache
from app import valuation as _valuation
from app.analyst_client import REQUEST_TIMEOUT_SECONDS
from app.analyst_client import _ensure_session as _yahoo_session
from app.config import YAHOO_QUERY_BASE

logger = logging.getLogger(__name__)

_YAHOO_SEARCH_BASE = "https://query2.finance.yahoo.com"
_GNEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US%3Aen"
_SEC_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
    "&type=&dateb=&owner=include&count={n}&output=atom"
)

_TTL_CANDLES = 900
_TTL_UNIVERSE = 24 * 3600
_TTL_NEWS = 900
_TTL_FILINGS = 3600
_NEG_TTL = 60

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


def _neg(key: str, message: str) -> dict[str, object]:
    """Cache a Yahoo throttle envelope briefly so bursts back off."""
    out: dict[str, object] = {"error": message}
    cache.set(f"neg:{key}", out)
    return out


def _neg_hit(key: str) -> dict[str, object] | None:
    hit = cache.get(f"neg:{key}", ttl=_NEG_TTL)
    return hit if isinstance(hit, dict) else None


def _parse_candles(ticker: str, payload: dict[str, object], range_: str, interval: str) -> dict[str, object]:
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
    result: dict[str, object] = first if isinstance(first, dict) else empty
    stamps_obj = result.get("timestamp")
    stamps: list[object] = stamps_obj if isinstance(stamps_obj, list) else []
    ind_obj = result.get("indicators")
    indicators: dict[str, object] = ind_obj if isinstance(ind_obj, dict) else empty
    quotes_obj = indicators.get("quote")
    quotes: list[object] = quotes_obj if isinstance(quotes_obj, list) else []
    head = quotes[0] if quotes and isinstance(quotes[0], dict) else empty
    opens_obj = head.get("open")
    opens: list[object] = opens_obj if isinstance(opens_obj, list) else []
    highs_obj = head.get("high")
    highs: list[object] = highs_obj if isinstance(highs_obj, list) else []
    lows_obj = head.get("low")
    lows: list[object] = lows_obj if isinstance(lows_obj, list) else []
    closes_obj = head.get("close")
    closes: list[object] = closes_obj if isinstance(closes_obj, list) else []
    vols_obj = head.get("volume")
    vols: list[object] = vols_obj if isinstance(vols_obj, list) else []
    meta_obj = result.get("meta")
    meta: dict[str, object] = meta_obj if isinstance(meta_obj, dict) else empty
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


def get_candles(sym: str, range_: str = "6mo", interval: str = "1d") -> dict[str, object]:
    """OHLCV bars from Yahoo chart v8; error envelope when Yahoo has no data."""
    ticker = sym.strip().upper()
    if not ticker:
        return {"error": "empty symbol"}
    key = f"candles:{ticker}:{range_}:{interval}"
    hit = cache.get(key, ttl=_TTL_CANDLES)
    if isinstance(hit, dict):
        return hit
    neg = _neg_hit(key)
    if neg is not None:
        return dict(neg, sym=ticker)
    try:
        qs = urllib.parse.urlencode({"range": range_, "interval": interval})
        raw = _http_bytes(f"{YAHOO_QUERY_BASE}/v8/finance/chart/{ticker}?{qs}")
        out = _parse_candles(ticker, json.loads(raw), range_, interval)
        cache.set(key, out)
        return out
    except _Throttle as e:
        logger.warning("candles throttled for %s: %s", ticker, e)
        return _neg(key, f"candles unavailable for {ticker}: {e}")
    except Exception as e:  # noqa: BLE001 - best-effort boundary, envelope not tracebacks
        logger.warning("candles failed for %s: %s", ticker, e)
        return {"error": f"candles unavailable for {ticker}: {e}"}


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
    """CIK for a ticker from the cached universe; None when unmapped."""
    universe = get_universe()
    tickers = universe.get("tickers")
    if not isinstance(tickers, list):
        return None
    want = sym.strip().upper()
    for row in tickers:
        if isinstance(row, dict) and row.get("sym") == want:
            cik = row.get("cik")
            return int(cik) if isinstance(cik, int) else None
    return None


def get_news(sym: str, limit: int = 12) -> dict[str, object]:
    """Per-ticker headlines from Google News RSS (keyless)."""
    ticker = sym.strip().upper()
    if not ticker:
        return {"sym": "", "items": []}
    key = f"news:{ticker}"
    hit = cache.get(key, ttl=_TTL_NEWS)
    if isinstance(hit, dict):
        return hit
    try:
        q = urllib.parse.quote(f"{ticker} stock")
        raw = _http_bytes(_GNEWS_URL.format(q=q))
        root = ET.fromstring(raw)
        items: list[dict[str, object]] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "link": (item.findtext("link") or "").strip(),
                    "published": (item.findtext("pubDate") or "").strip(),
                    "source": (item.findtext("source") or "").strip(),
                }
            )
            if len(items) >= limit:
                break
        out: dict[str, object] = {"sym": ticker, "items": items}
        cache.set(key, out)
        return out
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("news failed for %s: %s", ticker, e)
        return {"sym": ticker, "items": [], "error": f"news unavailable: {e}"}


def get_filings(sym: str, limit: int = 10) -> dict[str, object]:
    """Per-company SEC filings from the EDGAR Atom feed (needs CIK lookup)."""
    ticker = sym.strip().upper()
    cik = cik_for_ticker(ticker)
    if cik is None:
        return {"sym": ticker, "cik": None, "items": [], "error": f"no SEC CIK for {ticker}"}
    key = f"filings:{cik}"
    hit = cache.get(key, ttl=_TTL_FILINGS)
    if isinstance(hit, dict):
        return dict(hit, sym=ticker)
    try:
        url = _SEC_ATOM_URL.format(cik=f"{cik:010d}", n=max(1, min(limit, 40)))
        raw = _http_bytes(url, headers={"Accept": "application/atom+xml"})
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        items: list[dict[str, object]] = []
        for entry in root.findall("a:entry", ns)[:limit]:
            link = entry.find("a:link", ns)
            items.append(
                {
                    "title": (entry.findtext("a:title", default="", namespaces=ns) or "").strip(),
                    "link": link.get("href", "") if link is not None else "",
                    "published": (entry.findtext("a:updated", default="", namespaces=ns) or "").strip(),
                    "summary": (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()[:300],
                }
            )
        out: dict[str, object] = {"sym": ticker, "cik": cik, "items": items}
        cache.set(key, out)
        return out
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("filings failed for %s: %s", ticker, e)
        return {"sym": ticker, "cik": cik, "items": [], "error": f"filings unavailable: {e}"}


def get_quote(sym: str) -> dict[str, object]:
    """Latest quote via valuation.get_live_quote (own 5-min TTL in cache.db)."""
    ticker = sym.strip().upper()
    if not ticker:
        return {"sym": "", "price": None, "retrieved_at": None}
    try:
        quote = _valuation.get_live_quote(ticker)
        return {"sym": ticker, "price": quote.get("price"), "retrieved_at": quote.get("retrieved_at")}
    except Exception as e:  # noqa: BLE001 - best-effort boundary
        logger.warning("quote failed for %s: %s", ticker, e)
        return {"sym": ticker, "price": None, "retrieved_at": None, "error": f"quote unavailable: {e}"}


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
    rows_obj = candles.get("candles")
    rows: list[object] = rows_obj if isinstance(rows_obj, list) else []
    closes: list[float] = [
        float(r["close"]) for r in rows if isinstance(r, dict) and isinstance(r.get("close"), (int, float))
    ]
    summary: dict[str, object] = {
        "bars": len(closes),
        "last": closes[-1] if closes else None,
        "first": closes[0] if closes else None,
        "high": max(closes) if closes else None,
        "low": min(closes) if closes else None,
    }
    if len(closes) > 1 and closes[0]:
        summary["change_pct"] = (closes[-1] - closes[0]) / closes[0] * 100
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
            "quote": "valuation.get_live_quote (Yahoo quoteSummary)",
            "candles": "Yahoo chart v8, delayed",
            "news": "Google News RSS",
            "filings": "SEC EDGAR Atom",
        },
        "errors": {
            k: v["error"]
            for k, v in (("quote", quote), ("candles", candles), ("news", news), ("filings", filings))
            if isinstance(v, dict) and v.get("error")
        },
    }
