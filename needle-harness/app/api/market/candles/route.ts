export const dynamic = "force-dynamic";

// Server-side Yahoo v8/chart proxy. Server holds cookies/crumb, fixes CORS.
// Yahoo quote is keyless v8/chart — no crumb needed for this path.
const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36";

type Candle = { t: number; open: number; high: number; low: number; close: number; vol: number };

const RANGES: Record<string, { range: string; interval: string }> = {
  "5M": { range: "1d", interval: "5m" },
  "15M": { range: "1d", interval: "15m" },
  "1H": { range: "5d", interval: "60m" },
  "4H": { range: "1mo", interval: "90m" },
  "1D": { range: "6mo", interval: "1d" },
  "1W": { range: "2y", interval: "1wk" },
};

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number") {
    return null;
  }
  if (!Number.isFinite(value)) {
    return null;
  }
  return value;
}

function chartResult(body: unknown): unknown {
  if (!body || typeof body !== "object") {
    return undefined;
  }
  if (!("chart" in body)) {
    return undefined;
  }
  const chart = body.chart;
  if (!chart || typeof chart !== "object") {
    return undefined;
  }
  if (!("result" in chart)) {
    return undefined;
  }
  return chart.result;
}

function firstQuote(first: unknown): Record<string, unknown> | null {
  if (!first || typeof first !== "object") {
    return null;
  }
  if (!("indicators" in first)) {
    return null;
  }
  const indicators = first.indicators;
  if (!indicators || typeof indicators !== "object") {
    return null;
  }
  if (!("quote" in indicators)) {
    return null;
  }
  const quotes = indicators.quote;
  if (!Array.isArray(quotes)) {
    return null;
  }
  const head = quotes[0];
  if (!head || typeof head !== "object") {
    return null;
  }
  return head as Record<string, unknown>;
}

function seriesField(quote: Record<string, unknown>, key: string): unknown[] {
  if (!(key in quote)) {
    return [];
  }
  const rows = quote[key];
  if (Array.isArray(rows)) {
    return rows;
  }
  return [];
}

export async function GET(req: Request): Promise<Response> {
  const u = new URL(req.url);
  const sym = (u.searchParams.get("sym") ?? "NVDA").toUpperCase().slice(0, 12);
  if (!/^[A-Z0-9.\-=^]+$/.test(sym)) {
    return Response.json({ error: "bad symbol" }, { status: 400 });
  }
  const tab = u.searchParams.get("range") ?? u.searchParams.get("tab") ?? "1D";
  const picked = RANGES[tab] ?? RANGES["1D"];
  const range = picked?.range ?? "6mo";
  const interval = picked?.interval ?? "1d";
  const url = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?range=${range}&interval=${interval}`;
  let res: Response;
  try {
    res = await fetch(url, { headers: { "User-Agent": UA }, signal: AbortSignal.timeout(20_000) });
  } catch (err) {
    return Response.json(
      { error: `yahoo unreachable: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 },
    );
  }
  if (!res.ok) {
    return Response.json({ error: `yahoo ${res.status}` }, { status: 502 });
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return Response.json({ error: "yahoo bad json" }, { status: 502 });
  }
  const result = chartResult(body);
  const first = Array.isArray(result) ? result[0] : undefined;
  if (!first || typeof first !== "object") {
    return Response.json({ error: `no data for ${sym}` }, { status: 404 });
  }
  const timestamps = "timestamp" in first && Array.isArray(first.timestamp) ? first.timestamp : [];
  const quote = firstQuote(first);
  if (!quote) {
    return Response.json({ error: `no rows for ${sym}` }, { status: 404 });
  }
  const opens = seriesField(quote, "open");
  const highs = seriesField(quote, "high");
  const lows = seriesField(quote, "low");
  const closes = seriesField(quote, "close");
  const vols = seriesField(quote, "volume");
  const candles: Candle[] = [];
  for (let i = 0; i < timestamps.length; i++) {
    const o = toFiniteNumber(opens[i]);
    const h = toFiniteNumber(highs[i]);
    const l = toFiniteNumber(lows[i]);
    const c = toFiniteNumber(closes[i]);
    if (o === null || h === null || l === null || c === null) {
      continue;
    }
    const rawVol = toFiniteNumber(vols[i]) ?? 0;
    const stamp = toFiniteNumber(timestamps[i]) ?? i;
    candles.push({ t: stamp, open: o, high: h, low: l, close: c, vol: Math.max(0, Math.round(rawVol)) });
  }
  if (candles.length === 0) {
    return Response.json({ error: `no rows for ${sym}` }, { status: 404 });
  }
  let price = candles[candles.length - 1]?.close ?? 0;
  if ("meta" in first && first.meta && typeof first.meta === "object" && "regularMarketPrice" in first.meta) {
    const metaPrice = toFiniteNumber(first.meta.regularMarketPrice);
    if (metaPrice !== null) {
      price = metaPrice;
    }
  }
  return Response.json(
    { sym, range, interval, price, delayed: true, candles },
    { headers: { "Cache-Control": "no-store" } },
  );
}
