export type Candle = { t: number; open: number; high: number; low: number; close: number; vol: number };

type SymbolInfo = {
  name: string; base: number; chg: number; vol: string; sector: string;
  mktCap: string; pe: string; eps: string; rev: string; div: string;
  high52: number; low52: number; beta: string; shares: string; roe: string; de: string;
};

export const SYMBOLS: Record<string, SymbolInfo> = {
  BTC: { name: "Bitcoin", base: 97432, chg: 2.41, vol: "28.4B", sector: "Crypto", mktCap: "1.92T", pe: "—", eps: "—", rev: "—", div: "—", high52: 108268, low52: 38505, beta: "1.82", shares: "19.8M", roe: "—", de: "—" },
  ETH: { name: "Ethereum", base: 3412, chg: 1.87, vol: "14.1B", sector: "Crypto", mktCap: "411B", pe: "—", eps: "—", rev: "—", div: "—", high52: 4878, low52: 2200, beta: "1.65", shares: "120M", roe: "—", de: "—" },
  SOL: { name: "Solana", base: 214.6, chg: -1.24, vol: "3.8B", sector: "Crypto", mktCap: "103B", pe: "—", eps: "—", rev: "—", div: "—", high52: 260, low52: 78.9, beta: "1.94", shares: "480M", roe: "—", de: "—" },
  AAPL: { name: "Apple", base: 232.4, chg: 0.64, vol: "58.2M", sector: "Technology", mktCap: "3.54T", pe: "35.2", eps: "6.60", rev: "391B", div: "0.4%", high52: 260, low52: 164.1, beta: "1.24", shares: "15.2B", roe: "148%", de: "1.73" },
  NVDA: { name: "NVIDIA", base: 138.9, chg: 3.12, vol: "212M", sector: "Technology", mktCap: "3.41T", pe: "54.8", eps: "2.54", rev: "130B", div: "0.0%", high52: 153, low52: 75.6, beta: "1.66", shares: "24.5B", roe: "114%", de: "0.13" },
  TSLA: { name: "Tesla", base: 248.7, chg: -0.82, vol: "98.4M", sector: "Auto", mktCap: "798B", pe: "66.1", eps: "3.76", rev: "96.8B", div: "—", high52: 488, low52: 138.8, beta: "2.31", shares: "3.21B", roe: "11%", de: "0.08" },
  MSFT: { name: "Microsoft", base: 428.1, chg: 0.93, vol: "21.6M", sector: "Technology", mktCap: "3.18T", pe: "36.4", eps: "11.77", rev: "245B", div: "0.7%", high52: 468, low52: 344.8, beta: "0.90", shares: "7.43B", roe: "37%", de: "0.30" },
  AMZN: { name: "Amazon", base: 205.3, chg: 1.15, vol: "38.9M", sector: "Retail", mktCap: "2.16T", pe: "44.7", eps: "4.59", rev: "638B", div: "—", high52: 233, low52: 151.6, beta: "1.15", shares: "10.5B", roe: "22%", de: "0.23" },
  META: { name: "Meta", base: 585.2, chg: 1.48, vol: "12.4M", sector: "Technology", mktCap: "1.48T", pe: "28.9", eps: "20.27", rev: "164B", div: "0.3%", high52: 638, low52: 414.5, beta: "1.21", shares: "2.53B", roe: "34%", de: "0.11" },
  GOOG: { name: "Alphabet", base: 178.6, chg: 0.42, vol: "24.8M", sector: "Technology", mktCap: "2.18T", pe: "23.5", eps: "7.60", rev: "350B", div: "0.4%", high52: 208, low52: 131.3, beta: "1.03", shares: "12.2B", roe: "28%", de: "0.05" },
  SPY: { name: "S&P 500 ETF", base: 591.2, chg: 0.31, vol: "62.1M", sector: "ETF", mktCap: "582B", pe: "27.8", eps: "21.3", rev: "—", div: "1.3%", high52: 613, low52: 481.8, beta: "1.00", shares: "985M", roe: "—", de: "—" },
  QQQ: { name: "Nasdaq 100", base: 521.8, chg: 0.58, vol: "41.3M", sector: "ETF", mktCap: "322B", pe: "30.1", eps: "17.3", rev: "—", div: "0.6%", high52: 540, low52: 402.4, beta: "1.12", shares: "617M", roe: "—", de: "—" },
};

export const TICKER_ITEMS: { sym: string; val: string; chg: string }[] = [
  { sym: "BTC", val: "97,432", chg: "+2.41%" },
  { sym: "ETH", val: "3,412", chg: "+1.87%" },
  { sym: "SOL", val: "214.60", chg: "-1.24%" },
  { sym: "AAPL", val: "232.40", chg: "+0.64%" },
  { sym: "NVDA", val: "138.90", chg: "+3.12%" },
  { sym: "TSLA", val: "248.70", chg: "-0.82%" },
  { sym: "MSFT", val: "428.10", chg: "+0.93%" },
  { sym: "AMZN", val: "205.30", chg: "+1.15%" },
  { sym: "META", val: "585.20", chg: "+1.48%" },
  { sym: "GOOG", val: "178.60", chg: "+0.42%" },
  { sym: "SPY", val: "591.20", chg: "+0.31%" },
];

type NewsItem = { ts: string; cat: string; headline: string };

export const NEWS: Record<string, NewsItem[]> = {
  _global: [
    { ts: "09:42", cat: "Macro", headline: "Fed holds rates steady, signals one cut before year-end" },
    { ts: "09:15", cat: "Markets", headline: "S&P 500 futures edge higher as tech leads premarket gains" },
    { ts: "08:30", cat: "Macro", headline: "Jobless claims fall to 218K, labor market stays resilient" },
  ],
  BTC: [
    { ts: "09:31", cat: "Crypto", headline: "Bitcoin ETF inflows hit third straight week of gains" },
    { ts: "08:05", cat: "Crypto", headline: "Analysts eye $100K as BTC consolidates near highs" },
  ],
  ETH: [
    { ts: "09:12", cat: "Crypto", headline: "Ethereum staking yields steady as network activity climbs" },
    { ts: "07:48", cat: "Crypto", headline: "ETH ETF volumes pick up after quiet August" },
  ],
  SOL: [
    { ts: "08:55", cat: "Crypto", headline: "Solana DEX volumes outpace rivals for fifth week" },
    { ts: "07:20", cat: "Crypto", headline: "SOL pulls back as traders take profits near $220" },
  ],
  AAPL: [
    { ts: "09:28", cat: "Earnings", headline: "Apple services revenue hits record on strong subscriptions" },
    { ts: "08:12", cat: "Product", headline: "iPhone demand steady as supply chain normalizes" },
  ],
  NVDA: [
    { ts: "09:35", cat: "AI", headline: "NVIDIA datacenter revenue beats on Blackwell ramp" },
    { ts: "08:44", cat: "Analyst", headline: "Street raises targets as AI capex outlook improves" },
    { ts: "07:30", cat: "AI", headline: "Hyperscalers reaffirm GPU buildout plans" },
  ],
  TSLA: [
    { ts: "09:20", cat: "Auto", headline: "Tesla deliveries in focus as quarter-end approaches" },
    { ts: "08:02", cat: "Auto", headline: "Robotaxi pilot expands to second city" },
  ],
  MSFT: [
    { ts: "09:25", cat: "Cloud", headline: "Azure growth reaccelerates on AI workloads" },
    { ts: "08:18", cat: "AI", headline: "Copilot adoption broadens across enterprise seats" },
  ],
  AMZN: [
    { ts: "09:10", cat: "Retail", headline: "AWS reacceleration offsets softer retail margins" },
    { ts: "07:55", cat: "Retail", headline: "Prime Day pull-forward lifts Q3 sales outlook" },
  ],
  META: [
    { ts: "09:05", cat: "Ads", headline: "Reels monetization improves as ad prices firm" },
    { ts: "08:40", cat: "AI", headline: "Meta unveils new open-weight model for developers" },
  ],
  GOOG: [
    { ts: "09:18", cat: "Cloud", headline: "Google Cloud backlog grows on enterprise AI deals" },
    { ts: "08:26", cat: "Ads", headline: "Search share steady as AI overviews expand" },
  ],
  SPY: [
    { ts: "09:40", cat: "Markets", headline: "Breadth improves as cyclicals join the rally" },
    { ts: "08:33", cat: "Flows", headline: "Equity funds see largest inflow in six weeks" },
  ],
  QQQ: [
    { ts: "09:38", cat: "Tech", headline: "Mega-cap tech leads as yields ease" },
    { ts: "08:22", cat: "Flows", headline: "Nasdaq funds attract fresh retail bids" },
  ],
};

export const POSITIONS: { sym: string; qty: number; avg: number; last: number; pnl: number; pct: number }[] = [
  { sym: "NVDA", qty: 120, avg: 112.4, last: 138.9, pnl: 3180, pct: 23.58 },
  { sym: "AAPL", qty: 80, avg: 218.9, last: 232.4, pnl: 1080, pct: 6.17 },
  { sym: "BTC", qty: 0.5, avg: 88200, last: 97432, pnl: 4616, pct: 10.47 },
  { sym: "TSLA", qty: 60, avg: 261.3, last: 248.7, pnl: -756, pct: -4.82 },
  { sym: "SPY", qty: 150, avg: 572.8, last: 591.2, pnl: 2760, pct: 3.21 },
];

// ponytail: deterministic PRNG (mulberry32 seeded from base) — same symbol always draws the same chart; fine for mocks, no RNG injection needed.
function seeded(base: number): () => number {
  let s = Math.floor(base * 7919 + 13) >>> 0 || 1;
  return () => {
    s |= 0; s = (s + 0x6d2b79f5) | 0;
    let z = Math.imul(s ^ (s >>> 15), 1 | s);
    z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z;
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}

export function genCandles(base: number, n = 80): Candle[] {
  const rnd = seeded(base);
  const out: Candle[] = [];
  let price = base * 0.94;
  for (let t = 0; t < n; t++) {
    const drift = (rnd() - 0.47) * base * 0.012;
    const open = price;
    const close = Math.max(open + drift, base * 0.01);
    const high = Math.max(open, close) + rnd() * base * 0.004;
    const low = Math.min(open, close) - rnd() * base * 0.004;
    const vol = Math.floor(rnd() * 9000 + 1000 + Math.abs(close - open) * 40);
    out.push({ t, open, high, low, close, vol });
    price = close;
  }
  // pin last close near spot base
  const k = out[out.length - 1];
  if (k) { const d = base - k.close; k.close += d; k.high = Math.max(k.high + d, k.open, k.close); k.low = Math.min(k.low + d, k.open, k.close); }
  return out;
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
