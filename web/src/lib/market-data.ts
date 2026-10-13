export type QuoteState = { price: number | null; retrieved_at?: string; error?: string };

export type NewsItem = { title: string; link: string; published: string; source: string };

export type FilingItem = { title: string; link: string; published: string; summary: string };

export type SearchHit = { sym: string; name?: string; exchange?: string; type?: string };

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

// Read-only aggregate for the future agent loop: whatever ticker the viewer
// is showing, one fetch returns identity + quote + window + news + filings.
// Not wired to any agent yet — that stays a one-fetch addition.
export function fetchViewerContext(sym: string, signal?: AbortSignal) {
  return getJSON<Record<string, unknown>>(`/api/viewer-context?sym=${encodeURIComponent(sym)}`, signal);
}
