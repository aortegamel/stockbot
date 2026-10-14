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
