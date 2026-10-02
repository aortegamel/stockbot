import hashlib
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SYMBOLS = {
    "BTC": {"name": "Bitcoin", "base": 97432, "chg": 2.41, "vol": "28.4B", "sector": "Crypto"},
    "ETH": {"name": "Ethereum", "base": 3412, "chg": 1.87, "vol": "14.1B", "sector": "Crypto"},
    "SOL": {"name": "Solana", "base": 214.6, "chg": -1.24, "vol": "3.8B", "sector": "Crypto"},
    "AAPL": {"name": "Apple", "base": 232.4, "chg": 0.64, "vol": "58.2M", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA", "base": 138.9, "chg": 3.12, "vol": "212M", "sector": "Technology"},
    "TSLA": {"name": "Tesla", "base": 248.7, "chg": -0.82, "vol": "98.4M", "sector": "Auto"},
    "MSFT": {"name": "Microsoft", "base": 428.1, "chg": 0.93, "vol": "21.6M", "sector": "Technology"},
    "AMZN": {"name": "Amazon", "base": 205.3, "chg": 1.15, "vol": "38.9M", "sector": "Retail"},
    "META": {"name": "Meta", "base": 585.2, "chg": 1.48, "vol": "12.4M", "sector": "Technology"},
    "GOOG": {"name": "Alphabet", "base": 178.6, "chg": 0.42, "vol": "24.8M", "sector": "Technology"},
    "SPY": {"name": "S&P 500 ETF", "base": 591.2, "chg": 0.31, "vol": "62.1M", "sector": "ETF"},
    "QQQ": {"name": "Nasdaq 100", "base": 521.8, "chg": 0.58, "vol": "41.3M", "sector": "ETF"},
}


def gen_candles(sym, base, n):
    seed = int.from_bytes(hashlib.sha256(sym.encode()).digest()[:4], "big")
    rnd = random.Random(seed)
    out = []
    price = base * 0.94
    for t in range(n):
        drift = (rnd.random() - 0.47) * base * 0.012
        o = price
        c = max(o + drift, base * 0.01)
        h = max(o, c) + rnd.random() * base * 0.004
        lo = min(o, c) - rnd.random() * base * 0.004
        v = int(rnd.random() * 9000 + 1000 + abs(c - o) * 40)
        out.append({"t": t, "open": o, "high": h, "low": lo, "close": c, "vol": v})
        price = c
    last = out[-1]
    d = base - last["close"]
    last["close"] += d
    last["high"] = max(last["high"] + d, last["open"], last["close"])
    last["low"] = min(last["low"] + d, last["open"], last["close"])
    return out


@app.get("/")
def root():
    return {"ok": True, "service": "needle-api", "ui": "http://localhost:5173", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/symbols")
def symbols():
    return [{"sym": k, **v} for k, v in SYMBOLS.items()]


@app.get("/api/candles")
def candles(sym: str = "NVDA", n: int = 80):
    info = SYMBOLS.get(sym.upper())
    if info is None:
        raise HTTPException(status_code=404, detail=f"unknown symbol: {sym}")
    n = max(10, min(500, n))
    return {"sym": sym.upper(), "candles": gen_candles(sym.upper(), info["base"], n)}


class ChatIn(BaseModel):
    message: str = ""
    sym: str = "NVDA"


@app.post("/api/chat")
def chat(body: ChatIn):
    sym = (body.sym or "NVDA").upper()
    return {"reply": f"[mock] {sym}: {body.message[:500]}"}
