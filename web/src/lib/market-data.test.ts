import { afterEach, describe, expect, test } from "bun:test";
import {
  fetchFilings,
  fetchNews,
  fetchQuote,
  fetchViewerContext,
  fetchWatchlistSources,
  listWorkspaces,
  saveWorkspace,
  searchTickers,
} from "./market-data";
import type { SourceRow, WorkspaceDoc } from "./market-data";

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

  test("lists named workspaces and saves the selected document without changing other documents", async () => {
    const id = "a16ec102-2e68-49fb-b818-e9671a3d2474";
    const otherId = "375eb936-e12d-4c20-ac9d-9e91ac2f0008";
    const widgets: WorkspaceDoc["widgets"] = [{ id: "news", type: "news", ticker: "BRK.B", geometry: { x: 120, y: 40, width: 640, height: 480 } }];
    const stored: WorkspaceDoc[] = [
      { id, name: "Original", widgets: [], updated_at: null },
      { id: otherId, name: "Other", widgets: [], updated_at: null },
    ];
    let writes = 0;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === "/api/workspaces" && !init?.method) return Response.json({ workspaces: stored });
      if (String(input) === `/api/workspaces/${id}` && init?.method === "PUT") {
        if (init.headers && new Headers(init.headers).get("Content-Type") === "application/json") {
          const body = JSON.parse(String(init.body));
          if (body.name === "Research" && JSON.stringify(body.widgets) === JSON.stringify(widgets)) {
            writes += 1;
            stored[0] = { id, name: "Research", widgets, updated_at: "2026-10-29T12:00:00Z" };
            return Response.json(stored[0]);
          }
        }
        return new Response(null, { status: 400 });
      }
      return new Response(null, { status: 404 });
    }) as typeof fetch;
    expect((await listWorkspaces()).workspaces.map((doc) => doc.name)).toEqual(["Original", "Other"]);
    expect(writes).toBe(0);
    const saved = await saveWorkspace(id, "Research", widgets);
    expect(saved.name).toBe("Research");
    expect(saved.updated_at).toBe("2026-10-29T12:00:00Z");
    expect(saved.widgets[0]?.geometry).toEqual(widgets[0]?.geometry);
    expect((await listWorkspaces()).workspaces.map((doc) => doc.name)).toEqual(["Research", "Other"]);
    expect(writes).toBe(1);
  });
  test("reports workspace failures and preserves cancellation", async () => {
    globalThis.fetch = (async () => new Response(null, { status: 503 })) as unknown as typeof fetch;
    await expect(listWorkspaces()).rejects.toThrow("request failed: 503");
    await expect(saveWorkspace("workspace", "Research", [])).rejects.toThrow("request failed: 503");
    globalThis.fetch = (async () => new Response("invalid JSON")) as unknown as typeof fetch;
    await expect(listWorkspaces()).rejects.toBeInstanceOf(SyntaxError);
    await expect(saveWorkspace("workspace", "Research", [])).rejects.toBeInstanceOf(SyntaxError);
    const controller = new AbortController();
    controller.abort();
    globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) => {
      init?.signal?.throwIfAborted();
      return Promise.resolve(Response.json({ workspaces: [] }));
    }) as typeof fetch;
    const failure = await listWorkspaces(controller.signal).catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(DOMException);
    expect((failure as DOMException).name).toBe("AbortError");
  });
});
