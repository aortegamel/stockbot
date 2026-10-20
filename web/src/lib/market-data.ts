export type QuoteState = { price: number | null; retrieved_at?: string; error?: string };

export type NewsItem = { title: string; link: string; published: string; source: string };

// filed: official SEC filing date (YYYY-MM-DD). updated: Atom <updated> verbatim, acceptance clock.
export type FilingItem = { title: string; link: string; filed: string; updated: string; summary: string };

export type SearchHit = { sym: string; name?: string; exchange?: string; type?: string };

// One (symbol, source) outcome from the server fan-out. A failure never reads as "no records".
export type SourceRow = {
  symbol: string;
  source: "sec_filings" | "finra_short_interest" | "finra_short_sale_volume" | "finra_threshold_list";
  tool: string;
  status: "ok" | "no_records" | "error" | "timeout";
  as_of?: string;
  returned_count?: number;
  truncated?: boolean;
  ref?: { record_id?: string; uri?: string; known_at?: string };
  error?: string;
};

const API = "";

// One shared fetch: same-origin in dev via vite proxy, VITE_API_URL in prod.
async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const base = (import.meta.env.VITE_API_URL as string | undefined) ?? API;
  const res = await fetch(`${base}${path}`, { signal });
  if (!res.ok) throw new Error(`request failed: ${res.status}`);
  return (await res.json()) as T;
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

// Page-load fan-out for the whole watchlist: SEC filings + three FINRA datasets per symbol.
// Same-origin Bun API (like /api/agent), never the market API base.
export async function fetchWatchlistSources(symbols: string[], signal?: AbortSignal): Promise<SourceRow[]> {
  const res = await fetch(`/api/watchlist?symbols=${encodeURIComponent(symbols.join(","))}`, { signal });
  const body = (await res.json()) as { watchlist?: SourceRow[]; error?: string };
  if (!res.ok || !Array.isArray(body.watchlist)) throw new Error(body.error ?? `request failed: ${res.status}`);
  return body.watchlist;
}

// Read-only aggregate for the future agent loop: whatever ticker the viewer
// is showing, one fetch returns identity + quote + window + news + filings.
// Not wired to any agent yet — that stays a one-fetch addition.
export function fetchViewerContext(sym: string, signal?: AbortSignal) {
  return getJSON<Record<string, unknown>>(`/api/viewer-context?sym=${encodeURIComponent(sym)}`, signal);
}

// t: bar epoch seconds, parallel to closes. as_of: last bar instant (ISO UTC).
export type Sparkline = { t: number[]; closes: number[]; as_of: string | null } | { error: string };

export function fetchSparklines(syms: string[], range = "1d", signal?: AbortSignal) {
  const q = `syms=${encodeURIComponent(syms.join(","))}&range=${encodeURIComponent(range)}`;
  return getJSON<{ syms: Record<string, Sparkline> }>(`/api/sparklines?${q}`, signal);
}

export type MarketRow = {
  sym: string;
  last: number | null;
  chg_pct: number | null;
  volume: number | null;
  short_pct: number | null;
  last_filing: { form: string; filed: string } | null;
  error?: string;
};

export type MarketSort = "chg_pct" | "volume" | "short_pct";

export function fetchMarkets(scan: string, syms: string[] = [], sort: MarketSort = "chg_pct", signal?: AbortSignal) {
  const q = `scan=${encodeURIComponent(scan)}&syms=${encodeURIComponent(syms.join(","))}&sort=${sort}`;
  return getJSON<{ scan: string; sort: MarketSort; universe: string[]; rows: MarketRow[]; errors: SourceError[] }>(
    `/api/markets?${q}`,
    signal,
  );
}

export type ScanFilter = {
  symbols?: string[];
  forms?: string[];
  min_short_pct?: number;
  min_chg_pct?: number;
  sector?: string;
};

// Preset ids double as /api/markets?scan=<id>; custom scans only store their filter for now.
export type ScanDef = {
  id: string;
  name: string;
  kind: "preset" | "custom";
  filter: ScanFilter;
  created_at: string | null;
};

export function fetchScans(signal?: AbortSignal) {
  return getJSON<{ scans: ScanDef[] }>("/api/scans", signal);
}

export type SourceError = { sym: string; source: string; error: string };

export type BriefingItem = {
  kind: "news" | "filing" | "insider";
  sym: string;
  title: string;
  link: string;
  published?: string;
  filed?: string;
  filing_form?: string;
};

export function fetchBriefing(syms: string[], refresh = false, signal?: AbortSignal) {
  const q = `syms=${encodeURIComponent(syms.join(","))}${refresh ? "&refresh=1" : ""}`;
  return getJSON<{
    date: string;
    syms: string[];
    items: BriefingItem[];
    generated_at: string;
    cached: boolean;
    errors: SourceError[];
  }>(`/api/briefing?${q}`, signal);
}

export type FeedItem = {
  id: string;
  kind: "news" | "filing" | "insider" | "alert";
  sym?: string;
  title: string;
  link?: string;
  at: string;
  source: string;
};

export function fetchFeed(syms: string[], since?: string, signal?: AbortSignal) {
  const q = `syms=${encodeURIComponent(syms.join(","))}${since ? `&since=${encodeURIComponent(since)}` : ""}`;
  return getJSON<{ items: FeedItem[]; as_of: string; errors: SourceError[] }>(`/api/feed?${q}`, signal);
}

export type Positioning = {
  sym: string;
  short_pct: number | null;
  short_as_of: string | null;
  insider_net_90d: { buys: number; sells: number; net: number } | null;
  threshold_listed: boolean | null;
  threshold_as_of: string | null;
  errors: Record<string, string>;
};

export function fetchPositioning(sym: string, signal?: AbortSignal) {
  return getJSON<Positioning>(`/api/positioning?sym=${encodeURIComponent(sym)}`, signal);
}

// SEC entity/security resolution. Failures arrive as 200 with resolved=false and error.
export type Identity = {
  sym: string;
  resolved: boolean;
  entity_id?: string | null;
  security_id?: string | null;
  method?: string | null;
  error?: string;
};

export function fetchIdentity(sym: string, signal?: AbortSignal) {
  return getJSON<Identity>(`/api/identity?sym=${encodeURIComponent(sym)}`, signal);
}

// t: bar epoch seconds. Yahoo answers an error envelope with 200 when it has no data.
export type Candle = { t: number; open: number; high: number; low: number; close: number; vol: number };

export function fetchCandles(sym: string, range: string, interval: string, signal?: AbortSignal) {
  const q = `sym=${encodeURIComponent(sym)}&range=${encodeURIComponent(range)}&interval=${encodeURIComponent(interval)}`;
  return getJSON<{ candles?: Candle[]; error?: string }>(`/api/candles?${q}`, signal);
}

// One saved widget in a per-ticker workspace. updated_at null: the workspace was never saved.
export type WorkspaceWidget = { id: string; type: string; ticker: string; params?: Record<string, unknown> };
export type WorkspaceDoc = { sym: string; widgets: WorkspaceWidget[]; updated_at: string | null };

export function fetchWorkspace(sym: string, signal?: AbortSignal) {
  return getJSON<WorkspaceDoc>(`/api/workspaces/${encodeURIComponent(sym)}`, signal);
}

export async function saveWorkspace(sym: string, widgets: WorkspaceWidget[]): Promise<WorkspaceDoc> {
  const base = (import.meta.env.VITE_API_URL as string | undefined) ?? API;
  const res = await fetch(`${base}/api/workspaces/${encodeURIComponent(sym)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ widgets }),
  });
  if (!res.ok) throw new Error(`request failed: ${res.status}`);
  return (await res.json()) as WorkspaceDoc;
}

// Server-backed widget series. `error` reports a failed source; rows stay empty then.
export type WidgetData = {
  sym: string;
  type: string;
  rows?: Record<string, string | number | null>[];
  price?: number | null;
  ttm_eps?: number | null;
  pe?: number | null;
  earnings_yield?: number | null;
  as_of?: string | null;
  error?: string;
};

export function fetchWidgetData(type: string, sym: string, signal?: AbortSignal) {
  return getJSON<WidgetData>(`/api/widget-data?type=${encodeURIComponent(type)}&sym=${encodeURIComponent(sym)}`, signal);
}
