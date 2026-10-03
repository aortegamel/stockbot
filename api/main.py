"""Real-data market API: every route serves free keyless sources, no mocks.

Routes:
- GET /api/health, / — service status.
- GET /api/candles?sym=NVDA&range=6mo&interval=1d — Yahoo chart v8 OHLCV.
- GET /api/quote?sym=NVDA — valuation.get_live_quote (Yahoo quoteSummary).
- GET /api/search?q=nvda — any-ticker validation (Yahoo search).
- GET /api/universe — SEC ticker -> CIK + name universe.
- GET /api/identity?sym=NVDA — SEC entity/security via SourceGateway.
- GET /api/news?sym=NVDA — Google News RSS headlines.
- GET /api/filings?sym=NVDA — SEC EDGAR Atom filings.
- GET /api/viewer-context?sym=NVDA — read-only agent context aggregate.
  Contract: one JSON object with the viewed symbol, rendered payloads,
  and retrieval instants. The future agent loop re-fetches it on demand
  as a plain HTTP/tool call — no streaming, no session, no state.
  Nothing here calls any agent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import market_providers as providers

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"ok": True, "service": "needle-api", "ui": "http://localhost:5173", "docs": "/docs"}


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
