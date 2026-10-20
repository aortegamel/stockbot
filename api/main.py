"""Real-data market API: every route serves free keyless sources, no mocks.

Routes:
- GET /api/health, / — service status.
  The root advertises the UI on PORT, or localhost:5173 for standalone development.
- GET /api/candles?sym=NVDA&range=6mo&interval=1d — Yahoo chart v8 OHLCV.
- GET /api/quote?sym=NVDA — valuation.get_live_quote (Yahoo quoteSummary).
- GET /api/search?q=nvda — any-ticker validation (Yahoo search).
- GET /api/universe — SEC ticker -> CIK + name universe.
- GET /api/identity?sym=NVDA — SEC entity/security via SourceGateway.
- GET /api/news?sym=NVDA — merged keyless RSS (Google + Yahoo + Seeking Alpha + CNBC/MarketWatch; newsquant only when STOCKBOT_NEWSQUANT=1).
- GET /api/viewer-context?sym=NVDA — read-only agent context aggregate.
  Contract: one JSON object with the viewed symbol, rendered payloads,
  and retrieval instants. The future agent loop re-fetches it on demand
  as a plain HTTP/tool call — no streaming, no session, no state.
  Nothing here calls any agent.
- GET /api/sparklines?syms=A,B&range=1d — closes-only series per symbol (60s cache, max 20).
- GET /api/markets?scan=<preset>&syms=&sort=chg_pct|volume|short_pct — scan rows.
- GET/POST /api/scans, DELETE /api/scans/<id> — preset + stored custom scans (api/scans.py).
- GET /api/briefing?syms=&refresh=1 — daily news/8-K/Form 4 digest, data/briefing/<date>.json.
- GET /api/feed?syms=&since=<iso> — news + 8-K + Form 4 + pending thesis alerts, newest first.
- GET /api/positioning?sym=NVDA — short %, 90-day insider net shares, threshold-list presence.
"""

import os

from fastapi import FastAPI, HTTPException

from . import market_providers as providers
from . import portfolio_routes, scans, theses_routes, workspaces

app = FastAPI()
app.include_router(theses_routes.router)
app.include_router(portfolio_routes.router)
app.include_router(scans.router)
app.include_router(workspaces.router)


def _syms(raw: str) -> list[str]:
    try:
        return providers.parse_syms(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "needle-api",
        "ui": f"http://localhost:{os.getenv('PORT') or '5173'}",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/candles")
def candles(sym: str = "NVDA", range: str = "6mo", interval: str = "1d"):
    return providers.get_candles(sym, range_=range, interval=interval)


@app.get("/api/quote")
def quote(sym: str = "NVDA"):
    return providers.get_quote(sym)


@app.get("/api/search")
def search(q: str = "", limit: int = 10):
    return providers.search_tickers(q, limit=limit)


@app.get("/api/universe")
def universe():
    return providers.get_universe()


@app.get("/api/identity")
def identity(sym: str = "NVDA"):
    """Ticker -> SEC entity/security via SourceGateway + domain resolver."""
    return providers.resolve_identity(sym)


@app.get("/api/news")
def news(sym: str = "NVDA", limit: int = 12):
    return providers.get_news(sym, limit=limit)


@app.get("/api/filings")
def filings(sym: str = "NVDA", limit: int = 10):
    return providers.get_filings(sym, limit=limit)


@app.get("/api/viewer-context")
def viewer_context(sym: str = "NVDA"):
    """Read-only aggregate for the future agent loop (unwired)."""
    return providers.get_viewer_context(sym)


@app.get("/api/sparklines")
def sparklines(syms: str = "", range: str = "1d"):
    return providers.get_sparklines(_syms(syms), range_=range)


@app.get("/api/markets")
def markets(scan: str = "watchlist", syms: str = "", sort: str = "chg_pct"):
    if scan not in providers.SCAN_PRESETS:
        raise HTTPException(400, f"unknown scan: {scan}; presets: {', '.join(providers.SCAN_PRESETS)}")
    if sort not in providers.MARKET_SORTS:
        raise HTTPException(400, f"unknown sort: {sort}; one of {', '.join(providers.MARKET_SORTS)}")
    return providers.get_markets(scan, _syms(syms), sort=sort)


@app.get("/api/briefing")
def briefing(syms: str = "", refresh: bool = False):
    return providers.get_briefing(_syms(syms), refresh=refresh)


@app.get("/api/feed")
def feed(syms: str = "", since: str = ""):
    try:
        after = providers.parse_since(since) if since else None
    except ValueError as exc:
        raise HTTPException(400, f"invalid since: {since!r} (expected ISO 8601)") from exc
    return providers.get_feed(_syms(syms), since=after)


@app.get("/api/positioning")
def positioning(sym: str = "NVDA"):
    return providers.get_positioning(sym)
