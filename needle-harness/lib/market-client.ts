// Market data client: candles from the same-origin Yahoo proxy,
// quotes resolved from candle closes. No generated series here.
export type Candle = { t: number; open: number; high: number; low: number; close: number; vol: number };

export type WatchItem = { sym: string; name: string; price: number | null; delayed: boolean };

const NAMES: Record<string, string> = {
  NVDA: "NVIDIA",
  AAPL: "Apple",
  MSFT: "Microsoft",
  TSLA: "Tesla",
  AMZN: "Amazon",
  META: "Meta",
  GOOG: "Alphabet",
  SPY: "S&P 500 ETF",
};

const WATCHLIST = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META", "GOOG", "SPY"];

type CandlesReply = { sym: string; price?: number; delayed?: boolean; candles?: Candle[]; error?: string };

export async function fetchCandles(sym: string, tab: string, signal?: AbortSignal): Promise<CandlesReply> {
  const res = await fetch(`/api/market/candles?sym=${encodeURIComponent(sym)}&tab=${encodeURIComponent(tab)}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  const body = (await res.json()) as CandlesReply;
  if (!res.ok) {
    throw new Error(typeof body.error === "string" ? body.error : `candles ${res.status}`);
  }
  return body;
}

export async function fetchWatchlist(signal?: AbortSignal): Promise<WatchItem[]> {
  const out: WatchItem[] = [];
  for (const sym of WATCHLIST) {
    try {
      const r = await fetchCandles(sym, "1D", signal);
      const last = r.candles?.length ? r.candles[r.candles.length - 1]?.close ?? null : null;
      out.push({ sym, name: NAMES[sym] ?? sym, price: r.price ?? last, delayed: true });
    } catch {
      out.push({ sym, name: NAMES[sym] ?? sym, price: null, delayed: true });
    }
    if (signal?.aborted) {
      break;
    }
  }
  return out;
}

// Pure indicator math over caller-supplied candles — no fetching, no RNG.
export function calcMA(candles: Candle[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(candles.length).fill(null);
  let sum = 0;
  for (let i = 0; i < candles.length; i++) {
    const close = candles[i]?.close ?? 0;
    sum += close;
    if (i >= period) {
      sum -= candles[i - period]?.close ?? 0;
    }
    if (i >= period - 1) {
      out[i] = sum / period;
    }
  }
  return out;
}

export function calcRSI(candles: Candle[], period = 14): (number | null)[] {
  const out: (number | null)[] = new Array(candles.length).fill(null);
  if (candles.length <= period) {
    return out;
  }
  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= period; i++) {
    const d = (candles[i]?.close ?? 0) - (candles[i - 1]?.close ?? 0);
    if (d >= 0) {
      gain += d;
    } else {
      loss -= d;
    }
  }
  let ag = gain / period;
  let al = loss / period;
  out[period] = al === 0 ? 100 : 100 - 100 / (1 + ag / al);
  for (let i = period + 1; i < candles.length; i++) {
    const d = (candles[i]?.close ?? 0) - (candles[i - 1]?.close ?? 0);
    ag = (ag * (period - 1) + Math.max(d, 0)) / period;
    al = (al * (period - 1) + Math.max(-d, 0)) / period;
    out[i] = al === 0 ? 100 : 100 - 100 / (1 + ag / al);
  }
  return out;
}

function ema(vals: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = new Array(vals.length);
  let e = vals[0] ?? 0;
  out[0] = e;
  for (let i = 1; i < vals.length; i++) {
    e = (vals[i] ?? 0) * k + e * (1 - k);
    out[i] = e;
  }
  return out;
}

export function calcMACD(candles: Candle[]): { macd: number; signal: number | null; hist: number | null }[] {
  const closes = candles.map((c) => c.close);
  if (!closes.length) {
    return [];
  }
  const e12 = ema(closes, 12);
  const e26 = ema(closes, 26);
  const macdLine = closes.map((_, i) => (e12[i] ?? 0) - (e26[i] ?? 0));
  const sigLine = ema(macdLine, 9);
  return macdLine.map((m, i) =>
    i < 25 ? { macd: m, signal: null, hist: null } : { macd: m, signal: sigLine[i] ?? null, hist: (sigLine[i] ?? 0) === 0 && m === 0 ? null : m - (sigLine[i] ?? 0) },
  );
}
