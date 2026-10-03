export type Candle = { t: number; open: number; high: number; low: number; close: number; vol: number };

export type QuoteState = { price: number | null; retrieved_at?: string | null; error?: string };

export type NewsItem = { title: string; link: string; published: string; source: string };

export type FilingItem = { title: string; link: string; published: string; summary: string };

export type SearchHit = { sym: string; name?: string; exchange?: string; type?: string };

export type CandlesResponse = {
  sym: string;
  range?: string;
  interval?: string;
  price?: number | null;
  delayed?: boolean;
  candles?: Candle[];
  error?: string;
};

// One shared fetch: same-origin through the FastAPI backend
// (NEXT_PUBLIC_API_URL in the harness, VITE_API_URL in web).
function apiBase(): string {
  if (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof import.meta !== "undefined" && import.meta.env && typeof import.meta.env.VITE_API_URL === "string") {
    return import.meta.env.VITE_API_URL;
  }
  return "";
}
async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, { signal });
  if (!res.ok) throw new Error(`request failed: ${res.status}`);
  return (await res.json()) as T;
}

export function fetchCandles(sym: string, range = "6mo", interval = "1d", signal?: AbortSignal) {
  return getJSON<CandlesResponse>(
    `/api/candles?sym=${encodeURIComponent(sym)}&range=${range}&interval=${interval}`,
    signal,
  );
}

export function fetchQuote(sym: string, signal?: AbortSignal) {
  return getJSON<QuoteState & { sym: string }>(`/api/quote?sym=${encodeURIComponent(sym)}`, signal);
}

export function fetchNews(sym: string, signal?: AbortSignal) {
  return getJSON<{ sym: string; items: NewsItem[]; error?: string }>(
    `/api/news?sym=${encodeURIComponent(sym)}`,
    signal,
  );
}

export function fetchFilings(sym: string, signal?: AbortSignal) {
  return getJSON<{ sym: string; cik: number | null; items: FilingItem[]; error?: string }>(
    `/api/filings?sym=${encodeURIComponent(sym)}`,
    signal,
  );
}

export function searchTickers(q: string, signal?: AbortSignal) {
  return getJSON<{ query: string; hits: SearchHit[] }>(`/api/search?q=${encodeURIComponent(q)}`, signal);
}

// Read-only aggregate for the future agent loop: whatever ticker the viewer
// is showing, one fetch returns identity + quote + window + news + filings.
// Not wired to any agent yet — that stays a one-fetch addition.
export function fetchViewerContext(sym: string, signal?: AbortSignal) {
  return getJSON<Record<string, unknown>>(`/api/viewer-context?sym=${encodeURIComponent(sym)}`, signal);
}

export function calcMA(candles: Candle[], period: number): (number | null)[] {
  const out: (number | null)[] = new Array(candles.length).fill(null);
  let sum = 0;
  for (let i = 0; i < candles.length; i++) {
    sum += candles[i].close;
    if (i >= period) sum -= candles[i - period].close;
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

export function calcRSI(candles: Candle[], period = 14): (number | null)[] {
  const out: (number | null)[] = new Array(candles.length).fill(null);
  if (candles.length <= period) return out;
  let gain = 0, loss = 0;
  for (let i = 1; i <= period; i++) {
    const d = candles[i].close - candles[i - 1].close;
    if (d >= 0) gain += d; else loss -= d;
  }
  let ag = gain / period, al = loss / period;
  out[period] = al === 0 ? 100 : 100 - 100 / (1 + ag / al);
  for (let i = period + 1; i < candles.length; i++) {
    const d = candles[i].close - candles[i - 1].close;
    ag = (ag * (period - 1) + Math.max(d, 0)) / period;
    al = (al * (period - 1) + Math.max(-d, 0)) / period;
    out[i] = al === 0 ? 100 : 100 - 100 / (1 + ag / al);
  }
  return out;
}

function ema(vals: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = new Array(vals.length);
  let e = vals[0];
  out[0] = e;
  for (let i = 1; i < vals.length; i++) { e = vals[i] * k + e * (1 - k); out[i] = e; }
  return out;
}

export function calcMACD(candles: Candle[]): { macd: number; signal: number | null; hist: number | null }[] {
  const closes = candles.map((c) => c.close);
  if (!closes.length) return [];
  const e12 = ema(closes, 12), e26 = ema(closes, 26);
  const macdLine = closes.map((_, i) => e12[i] - e26[i]);
  const sigLine = ema(macdLine, 9);
  return macdLine.map((m, i) => (i < 25 ? { macd: m, signal: null, hist: null } : { macd: m, signal: sigLine[i], hist: m - sigLine[i] }));
}
