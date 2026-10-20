import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import fc from "fast-check";
import { act, useState } from "react";
import type { ReactNode } from "react";
import type { Root } from "react-dom/client";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import Workspace from "./Workspace";
import { ActionCards } from "../components/action-cards";
import { monthlyReturns } from "../components/widgets";
import type { Candle, WorkspaceDoc } from "../lib/market-data";
import { useHashRoute } from "../lib/route";

// React DOM reads browser globals at import time. The test must install those globals first.
const { createRoot } = await import("react-dom/client");
const realFetch = globalThis.fetch;
let mounted: Root | null = null;
const container = () => dom.window.document.getElementById("root")!;
const text = () => container().textContent ?? "";

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
  dom.window.localStorage.clear();
  dom.window.history.replaceState(null, "", "/");
});

const bar = (date: string, close: number): Candle => ({ t: Date.parse(date) / 1000, close, open: close, high: close, low: close, vol: 1 });

function serve() {
  const docs: Record<string, WorkspaceDoc> = {};
  const requests: { path: string; method: string }[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), dom.window.location.href);
    requests.push({ path: url.pathname + url.search, method: init?.method ?? "GET" });
    if (url.pathname.startsWith("/api/workspaces/")) {
      const sym = url.pathname.split("/").at(-1)!;
      if (init?.method === "PUT") {
        docs[sym] = { sym, ...JSON.parse(String(init.body)), updated_at: "2026-10-20T12:00:00+00:00" };
      }
      return Response.json(docs[sym] ?? { sym, widgets: [], updated_at: null });
    }
    if (url.pathname === "/api/quote") return Response.json({ price: 100 });
    if (url.pathname === "/api/watchlist") return Response.json({ watchlist: [] });
    if (url.pathname === "/api/portfolio") return new Response(null, { status: 404 });
    if (url.pathname === "/api/candles") return Response.json({ candles: [bar("2025-12-01", 100), bar("2026-01-01", 110), bar("2026-02-01", 99)] });
    if (url.pathname === "/api/filings") return Response.json({ items: [{ title: "10-K Annual report", filed: "2026-10-19", updated: "", summary: "", link: "https://www.sec.gov/filing" }] });
    if (url.pathname === "/api/news") return Response.json({ items: [{ title: "New chip", published: "2026-10-20", source: "News", link: "https://example.org" }] });
    if (url.pathname === "/api/widget-data") return Response.json({ rows: [{ date: "2026-10-19", short_qty: 100, total_qty: 200, short_pct: 50, days_to_cover: 2, title: "Jane bought 10 sh" }], price: 100, ttm_eps: 2, pe: 50, earnings_yield: 0.02 });
    if (url.pathname === "/api/theses") return Response.json([
      { thesis_id: "t1", slug: "chip-demand", scope: "NVDA", user_thesis: "Demand grows", pending: 1, alerts: 1, status: "active" },
      { thesis_id: "t2", slug: "apple", scope: "AAPL", user_thesis: "Services grow", pending: 1, alerts: 1, status: "active" },
    ]);
    if (url.pathname === "/api/theses/t1") return Response.json({
      triggers: [
        { trigger_id: "tr1", status: "pending", summary: "New evidence", created_at: "2026-10-20", trigger_type: "new_external_evidence" },
        { trigger_id: "tr2", status: "processed", summary: "Old evidence", trigger_type: "new_external_evidence" },
      ]
    });
    throw new Error(`unexpected request ${url.pathname}`);
  }) as typeof fetch;
  return { docs, requests };
}

function Page({ events = [] }: { events?: AgentEvent[] }) {
  const [route, navigate] = useHashRoute();
  const selected = route?.page === "workspace" ? route.sym : "NVDA";
  const [watchlist, setWatchlist] = useState(["NVDA", "GME"]);
  return <Workspace selected={selected} pick={(sym) => navigate({ page: "workspace", sym })} watchlist={watchlist} onWatchlist={setWatchlist}
    events={events} input="" setInput={() => { }} onSend={() => { }} openDrawer={() => { }} busy={false} debug={false}
    sessions={[{ session_id: "s1", status: "done", query: "NVDA research" }]} sessionId={null} setSessionId={() => { }} snapshot={null} journal={null} seq={0} />;
}

async function render(node: ReactNode) {
  mounted ??= createRoot(container());
  await act(async () => mounted!.render(node));
  await settle();
}
async function settle() {
  for (let i = 0; i < 12; i++) await act(async () => { await Promise.resolve(); });
}
async function click(el: Element) {
  await act(async () => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })));
  await settle();
}
function find<T extends Element>(selector: string): T {
  const el = container().querySelector<T>(selector);
  if (!el) throw new Error(`missing ${selector}`);
  return el;
}
async function add(type: string) {
  await act(async () => {
    const select = find<HTMLSelectElement>('select[aria-label="Add widget"]');
    select.value = type;
    select.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
  await settle();
}
function button(label: string): HTMLButtonElement {
  const el = [...container().querySelectorAll("button")].find((b) => b.textContent === label);
  if (!el) throw new Error(`missing ${label}`);
  return el;
}

describe("Workspace", () => {
  test("persists open tickers, opens another ticker with the pick flow, and keeps at least one tab", async () => {
    serve();
    await render(<Page />);
    expect(JSON.parse(dom.window.localStorage.getItem("stockbot.tabs")!)).toEqual(["NVDA"]);
    await click(find('[aria-label="Open another ticker"]'));
    const input = find<HTMLInputElement>('[aria-label="Open ticker"]');
    input.value = "amd";
    await act(async () => input.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
    await settle();
    expect(dom.window.location.hash).toBe("#/workspace/AMD");
    expect(JSON.parse(dom.window.localStorage.getItem("stockbot.tabs")!)).toEqual(["NVDA", "AMD"]);
    expect(find<HTMLAnchorElement>('[role="tab"][href="#/workspace/NVDA"]').getAttribute("aria-selected")).toBe("false");
    await click(find('[aria-label="Close AMD"]'));
    expect(dom.window.location.hash).toBe("#/workspace/NVDA");
    expect(container().querySelector('[aria-label="Close NVDA"]')).toBeNull();
    await act(async () => mounted!.unmount());
    mounted = null;
    await render(<Page />);
    expect(JSON.parse(dom.window.localStorage.getItem("stockbot.tabs")!)).toEqual(["NVDA"]);
    expect(text()).toContain("GME");
  });

  test("adds, refreshes, expands, removes, and saves widgets", async () => {
    const { docs, requests } = serve();
    await render(<Page />);
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(3);
    await add("monthly-returns");
    expect(text()).toContain("10.0%");
    expect(text()).toContain("-10.0%");
    await click(find('[aria-label="Expand Monthly returns · NVDA"]'));
    expect(find('[aria-label="Monthly returns · NVDA"]').classList.contains("ws-widget-wide")).toBe(true);
    const before = requests.filter((r) => r.path.startsWith("/api/candles")).length;
    await click(find('[aria-label="Refresh Monthly returns · NVDA"]'));
    expect(requests.filter((r) => r.path.startsWith("/api/candles")).length).toBe(before + 1);
    await click(find('[aria-label="Remove News · NVDA"]'));
    expect(container().querySelector('[aria-label="News · NVDA"]')).toBeNull();
    await click(button("Save"));
    expect(docs.NVDA.widgets.map((w) => w.type)).toEqual(["price-volume", "financials", "monthly-returns"]);
    expect(text()).toContain("saved 12:00:00");
  });

  test("loads a saved empty workspace without restoring defaults", async () => {
    const { docs } = serve();
    docs.NVDA = { sym: "NVDA", widgets: [], updated_at: "2026-10-20T10:00:00Z" };
    await render(<Page />);
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(0);
    await add("short-interest");
    await add("short-volume");
    await add("insider");
    await add("filings");
    await add("multiples");
    expect(text()).toContain("50%");
    expect(text()).toContain("Jane bought 10 sh");
    expect(text()).toContain("P/E50");
    expect(text()).toContain("History unavailable");
    expect(container().querySelector('a[href="https://www.sec.gov/filing"]')).not.toBeNull();
  });

  test("applies widget_update once to the matching workspace and saves it", async () => {
    const { docs, requests } = serve();
    const added: AgentEvent = { type: "widget_update", ticker: "NVDA", widgetType: "short-volume" };
    const events: AgentEvent[] = [added, added, { type: "widget_update", ticker: "AAPL", widgetType: "insider" }, { type: "widget_update", ticker: "NVDA", widgetType: "bad" }];
    await render(<Page events={events} />);
    expect(docs.NVDA.widgets.filter((w) => w.type === "short-volume")).toHaveLength(1);
    expect(docs.NVDA.widgets.some((w) => w.type === "insider")).toBe(false);
    await click(find('[aria-label="Remove Short volume · NVDA"]'));
    await render(<Page events={[...events]} />);
    expect(container().querySelector('[aria-label="Short volume · NVDA"]')).toBeNull();
    expect(requests.filter((r) => r.method === "PUT")).toHaveLength(1);
  });

  test("switches bottom panels and loads only pending alerts scoped to the ticker", async () => {
    const { requests } = serve();
    await render(<Page />);
    const panel = (name: string) => find<HTMLElement>(`[role="tabpanel"][aria-label="${name}"]`);
    expect(panel("Theses").hidden).toBe(false);
    await click(find('button[role="tab"]:nth-child(2)'));
    expect(panel("Alerts").hidden).toBe(false);
    expect(text()).toContain("New evidence");
    expect(text()).not.toContain("Old evidence");
    expect(requests.some((r) => r.path === "/api/theses/t2")).toBe(false);
    await click(button("Sessions"));
    expect(panel("Sessions").hidden).toBe(false);
    expect(panel("Alerts").hidden).toBe(true);
    await click(button("Journal"));
    expect(panel("Journal").hidden).toBe(false);
  });

  test("renders one summary card per ticker with the workspace link", async () => {
    await render(<ActionCards events={[
      { type: "widget_update", ticker: "NVDA", widgetType: "news" },
      { type: "widget_update", ticker: "AAPL", widgetType: "filings" },
      { type: "widget_update", ticker: "NVDA", widgetType: "insider" },
    ]} />);
    expect(text()).toContain("NVDA summary · 2 widgets");
    expect(text()).toContain("AAPL summary · 1 widget");
    expect(container().querySelectorAll('a[href="#/workspace/NVDA"]')).toHaveLength(1);
  });
});

test("monthly returns use the last close per month, omit the first month, and skip a zero base", () => {
  const years = monthlyReturns([bar("2025-12-01", 100), bar("2026-01-01", 105), bar("2026-01-20", 110), bar("2026-02-01", 0), bar("2026-03-01", 99)]);
  expect(years.get(2026)?.[0]).toBeCloseTo(0.1);
  expect(years.get(2026)?.[1]).toBe(-1);
  expect(years.get(2026)?.[2]).toBeNull();
  expect(years.has(2025)).toBe(false);
});

test("monthly returns compound to the final close over the initial close", () => {
  fc.assert(fc.property(fc.array(fc.integer({ min: 1, max: 100000 }), { minLength: 12, maxLength: 12 }), (prices) => {
    const candles = [bar("2025-12-01", 100), ...prices.map((price, i) => bar(`2026-${String(i + 1).padStart(2, "0")}-01`, price))];
    const returns = monthlyReturns(candles).get(2026)!;
    expect(returns).toHaveLength(12);
    expect(returns.every((r) => r !== null)).toBe(true);
    const compounded = returns.reduce<number>((value, r) => value * (1 + r!), 1);
    expect(compounded).toBeCloseTo(prices[11] / 100, 6);
  }), { seed: 2119, numRuns: 100 });
});
