import { afterEach, describe, expect, test } from "bun:test";
import {
  fetchFilings,
  fetchNews,
  fetchQuote,
  fetchViewerContext,
  fetchWatchlistSources,
  fetchCandles,
  fetchWorkspace,
  fetchWidgetData,
  saveWorkspace,
  searchTickers,
} from "./market-data";
import type { SourceRow } from "./market-data";

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("market data requests", () => {
  test("encodes symbols and queries, forwards cancellation, and preserves response fields", async () => {
    const signal = new AbortController().signal;
    const symbol = "BRK/B & Co";
    const quote = { sym: symbol, price: 123.45, retrieved_at: "2026-10-14T12:34:56Z" };
    const news = { sym: symbol, items: [{ title: "News", link: "https://news.example/1", published: "Wed, 14 Oct 2026 12:34:56 GMT", source: "Wire" }] };
    const filings = { sym: symbol, cik: 123, items: [{ title: "10-Q", link: "https://sec.example/1", filed: "2026-10-13", updated: "2026-10-14T00:13:45-04:00", summary: "Quarterly results" }] };
    const search = { query: symbol, hits: [{ sym: "BRK/B", name: "Berkshire", exchange: "NYSE", type: "EQUITY" }] };
    const viewer = { identity: { sym: symbol }, quote, news, filings, window: { as_of: "2026-10-14" } };
    const rows: SourceRow[] = [{ symbol, source: "sec_filings", tool: "sec", status: "ok", as_of: "2026-10-13", ref: { record_id: "record-1", known_at: "2026-10-14T00:13:45Z" } }];
    const calls: { url: string; signal: AbortSignal | null | undefined }[] = [];
    const responses = [quote, news, filings, search, viewer, { watchlist: rows }];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(input), signal: init?.signal });
      return Response.json(responses.shift());
    }) as unknown as typeof fetch;

    expect(await fetchQuote(symbol, signal)).toEqual(quote);
    expect(await fetchNews(symbol, signal)).toEqual(news);
    expect(await fetchFilings(symbol, signal)).toEqual(filings);
    expect(await searchTickers(symbol, signal)).toEqual(search);
    expect(await fetchViewerContext(symbol, signal)).toEqual(viewer);
    expect(await fetchWatchlistSources([symbol, "AAPL"], signal)).toEqual(rows);
    expect(calls).toEqual([
      { url: `/api/quote?sym=${encodeURIComponent(symbol)}`, signal },
      { url: `/api/news?sym=${encodeURIComponent(symbol)}`, signal },
      { url: `/api/filings?sym=${encodeURIComponent(symbol)}`, signal },
      { url: `/api/search?q=${encodeURIComponent(symbol)}`, signal },
      { url: `/api/viewer-context?sym=${encodeURIComponent(symbol)}`, signal },
      { url: `/api/watchlist?symbols=${encodeURIComponent(`${symbol},AAPL`)}`, signal },
    ]);
    expect(responses).toEqual([]);
  });

  test("reports HTTP status for every market request instead of returning error bodies", async () => {
    globalThis.fetch = (async () => Response.json({ error: "upstream private detail" }, { status: 503 })) as unknown as typeof fetch;
    for (const request of [fetchQuote, fetchNews, fetchFilings, searchTickers, fetchViewerContext]) {
      const failure = await request("NVDA").catch((error: unknown) => error);
      expect(failure).toBeInstanceOf(Error);
      expect((failure as Error).message).toBe("request failed: 503");
    }
  });

  test("uses the watchlist error before HTTP status or malformed data", async () => {
    const cases = [
      { body: { error: "source fan-out failed" }, status: 502, error: "source fan-out failed" },
      { body: { watchlist: [], error: "denied" }, status: 403, error: "denied" },
      { body: { watchlist: "not an array", error: "invalid sources" }, status: 200, error: "invalid sources" },
      { body: {}, status: 200, error: "request failed: 200" },
      { body: { watchlist: null }, status: 500, error: "request failed: 500" },
    ];
    for (const entry of cases) {
      globalThis.fetch = (async () => Response.json(entry.body, { status: entry.status })) as unknown as typeof fetch;
      await expect(fetchWatchlistSources(["NVDA"])).rejects.toThrow(entry.error);
    }
    globalThis.fetch = (async () => Response.json({ watchlist: [], error: "informational only" })) as unknown as typeof fetch;
    expect(await fetchWatchlistSources([])).toEqual([]);
  });

  test("propagates network, JSON, and cancellation failures", async () => {
    globalThis.fetch = (() => Promise.reject(new Error("network disconnected"))) as unknown as typeof fetch;
    await expect(fetchViewerContext("NVDA")).rejects.toThrow("network disconnected");
    globalThis.fetch = (async () => new Response("not JSON")) as unknown as typeof fetch;
    await expect(fetchQuote("NVDA")).rejects.toBeInstanceOf(SyntaxError);
    await expect(fetchWatchlistSources(["NVDA"])).rejects.toBeInstanceOf(SyntaxError);
    const controller = new AbortController();
    controller.abort();
    const forwarded: (AbortSignal | null | undefined)[] = [];
    globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) => {
      forwarded.push(init?.signal);
      return Promise.reject(new DOMException("cancelled", "AbortError"));
    }) as unknown as typeof fetch;
    const failure = await fetchNews("NVDA", controller.signal).catch((error: unknown) => error);
    expect(forwarded).toEqual([controller.signal]);
    expect(failure).toBeInstanceOf(DOMException);
    expect((failure as DOMException).name).toBe("AbortError");
  });

  test("fetches workspace widgets and candles, and saves the workspace with PUT", async () => {
    const signal = new AbortController().signal;
    const calls: { url: string; init?: RequestInit }[] = [];
    const doc = { sym: "BRK.B", widgets: [{ id: "news", type: "news", ticker: "BRK.B" }], updated_at: "2026-10-20" };
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(input), init });
      return Response.json(doc);
    }) as typeof fetch;
    expect(await fetchWorkspace("BRK.B", signal)).toEqual(doc);
    await fetchCandles("BRK.B", "5y", "1mo", signal);
    await fetchWidgetData("short-interest", "BRK.B", signal);
    expect(await saveWorkspace("BRK.B", doc.widgets)).toEqual(doc);
    expect(calls.map((c) => c.url)).toEqual([
      "/api/workspaces/BRK.B",
      "/api/candles?sym=BRK.B&range=5y&interval=1mo",
      "/api/widget-data?type=short-interest&sym=BRK.B",
      "/api/workspaces/BRK.B",
    ]);
    expect(calls.slice(0, 3).every((c) => c.init?.signal === signal)).toBe(true);
    expect(calls[3].init).toEqual({
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ widgets: doc.widgets }),
    });
    globalThis.fetch = (async () => new Response(null, { status: 503 })) as unknown as typeof fetch;
    await expect(saveWorkspace("BRK.B", [])).rejects.toThrow("request failed: 503");
  });
});
