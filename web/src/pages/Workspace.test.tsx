import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import fc from "fast-check";
import { act, useState, type ReactNode } from "react";
import type { Root } from "react-dom/client";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import Workspace, { DEFAULT_WORKSPACE_ID, type WorkspaceDraft } from "./Workspace";
import { ActionCards, type WidgetOutcomes } from "../components/action-cards";
import { monthlyReturns } from "../components/widgets";
import type { Candle, WorkspaceDoc, WorkspaceWidget } from "../lib/market-data";
import { useHashRoute } from "../lib/route";

// React DOM must load after test-dom installs browser globals.
const { createRoot } = await import("react-dom/client");
const realFetch = globalThis.fetch;
let mounted: Root | null = null;
const container = () => dom.window.document.getElementById("root")!;
const text = () => container().textContent ?? "";
const ID = "a16ec102-2e68-49fb-b818-e9671a3d2474";
const geometry = { x: 40, y: 50, width: 560, height: 360 };
const widget = (type: string, id = type, ticker = "NVDA"): WorkspaceWidget => ({ id, type, ticker, geometry });
const bar = (date: string, close: number): Candle => ({ t: Date.parse(date) / 1000, close, open: close, high: close, low: close, vol: 1 });

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
  dom.window.localStorage.clear();
  dom.window.history.replaceState(null, "", "/");
});

function serve(widgets?: WorkspaceWidget[]) {
  const docs: Record<string, WorkspaceDoc> = widgets ? { [ID]: { id: ID, name: "Research", widgets, updated_at: "2026-10-20T10:00:00Z" } } : {};
  if (widgets) dom.window.localStorage.setItem("stockbot.workspace", ID);
  const requests: { path: string; method: string }[] = [];
  const control = { failLoad: false, failSave: false, delaySave: null as (() => Promise<void>) | null };
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), dom.window.location.href);
    requests.push({ path: url.pathname + url.search, method: init?.method ?? "GET" });
    if (url.pathname === "/api/workspaces") return control.failLoad ? new Response(null, { status: 503 }) : Response.json({ workspaces: Object.values(docs) });
    if (url.pathname.startsWith("/api/workspaces/") && init?.method === "PUT") {
      if (control.failSave) return new Response(null, { status: 503 });
      const id = url.pathname.split("/").at(-1)!;
      const body = JSON.parse(String(init.body));
      if (control.delaySave) await control.delaySave();
      docs[id] = { id, ...body, updated_at: "2026-10-20T12:00:00+00:00" };
      return Response.json(docs[id]);
    }
    if (url.pathname === "/api/quote") return Response.json({ price: 100 });
    if (url.pathname === "/api/watchlist") return Response.json({ watchlist: [] });
    if (url.pathname === "/api/portfolio") return new Response(null, { status: 404 });
    if (url.pathname === "/api/sparklines") return Response.json({ syms: {} });
    if (url.pathname === "/api/markets") return Response.json({ rows: [], errors: [] });
    if (url.pathname === "/api/briefing") return Response.json({ items: [], errors: [] });
    if (url.pathname === "/api/scans") return Response.json({ scans: [] });
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
  return { docs, requests, control, puts: () => requests.filter((r) => r.method === "PUT"), saved: () => Object.values(docs)[0] };
}

function Page({ events = [], onWidgetOutcomes }: { events?: AgentEvent[]; onWidgetOutcomes?: (outcomes: WidgetOutcomes) => void }) {
  const [route, navigate] = useHashRoute();
  const selected = route?.page === "workspace" ? route.sym : "NVDA";
  const [watchlist, setWatchlist] = useState(["NVDA", "GME"]);
  const [workspace, setWorkspace] = useState<WorkspaceDraft | null>(null);
  return <Workspace selected={selected} pick={(sym) => navigate({ page: "workspace", sym })} watchlist={watchlist} onWatchlist={setWatchlist}
    workspace={workspace} setWorkspace={setWorkspace} events={events} input="" setInput={() => { }} research={false} setResearch={() => { }} setAskContext={() => { }} onSend={() => { }} openDrawer={() => { }} busy={false} debug={false} onWidgetOutcomes={onWidgetOutcomes}
    sessions={[{ session_id: "s1", status: "done", query: "NVDA research" }]} sessionId={null} setSessionId={() => { }} snapshot={null} journal={null} seq={0} chat={<p>Agent chat</p>} />;
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
function button(label: string): HTMLButtonElement {
  const el = [...container().querySelectorAll("button")].find((b) => b.textContent === label);
  if (!el) throw new Error(`missing ${label}`);
  return el;
}
async function change(label: string, value: string) {
  await act(async () => {
    const select = [...container().querySelectorAll("select")].find((el) => el.getAttribute("aria-label") === label);
    if (!select) throw new Error(`missing select ${label}`);
    select.value = value;
    select.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
  await settle();
}
const add = (type: string) => change("Add widget", type);
async function type(selector: string, value: string) {
  await act(async () => {
    const input = find<HTMLInputElement>(selector);
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value")?.set?.call(input, value);
    input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  });
}
async function key(target: Element, name: string, shiftKey = false) {
  await act(async () => target.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: name, shiftKey, bubbles: true, cancelable: true })));
  await settle();
}
const escape = (target: Element = dom.window.document.body) => key(target, "Escape");
async function reload() {
  await act(async () => mounted!.unmount());
  mounted = null;
  await render(<Page />);
}

describe("Named workspace", () => {
  test("creates, renames, switches, and saves named layouts without losing other drafts", async () => {
    const server = serve();
    await render(<Page />);
    expect(find<HTMLInputElement>('[aria-label="Workspace name"]').value).toBe("default");
    expect(container().querySelector('[aria-label="Market indexes"]')).not.toBeNull();
    await click(button("+"));
    await type('[aria-label="Workspace name"]', "  Earnings  ");
    await click(button("Create workspace"));
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(0);
    expect(container().querySelector('form[aria-label="TradingView link"]')).toBeNull();
    await add("monthly-returns");
    await click(button("+"));
    await type('[aria-label="Workspace name"]', "Risk");
    await click(button("Create workspace"));
    expect(container().querySelector('[aria-label="Monthly returns · NVDA"]')).toBeNull();
    expect(server.puts()).toHaveLength(0);
    await click(button("Earnings *"));
    expect(find<HTMLInputElement>('[aria-label="Workspace name"]').value).toBe("Earnings");
    expect(find('[aria-label="Monthly returns · NVDA"]').textContent).toContain("10.0%");
    await click(button("Save"));
    expect(server.saved().name).toBe("Earnings");
    expect(server.saved().widgets.some((w) => w.type === "monthly-returns")).toBe(true);
    expect(server.saved().id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    expect(button("Risk *")).not.toBeNull();
    await reload();
    expect(find<HTMLInputElement>('[aria-label="Workspace name"]').value).toBe("Earnings");
    expect([...container().querySelectorAll('[role="tab"]')].map((tab) => tab.textContent)).toEqual(["default", "Earnings"]);
  });

  test("copies the current default draft with independent windows and preserves saved workspaces", async () => {
    const server = serve([]);
    const source = [
      widget("news", "source-news", "AAPL"),
      { ...widget("price-volume", "source-chart", "GME"), params: { interval: "1h", style: "line", studies: ["MA50"] } },
    ];
    server.docs[DEFAULT_WORKSPACE_ID] = { id: DEFAULT_WORKSPACE_ID, name: "default", widgets: source, updated_at: "2026-10-20T10:00:00Z" };
    dom.window.localStorage.setItem("stockbot.workspace", DEFAULT_WORKSPACE_ID);
    await render(<Page />);
    expect(button("Research")).not.toBeNull();
    await key(find('[aria-label="Move or resize News · AAPL"]'), "ArrowRight");
    await click(button("+"));
    await type('[aria-label="Workspace name"]', "Copy");
    await click(find('input[type="checkbox"]'));
    await click(button("Create workspace"));
    await click(button("Save"));
    const copied = Object.values(server.docs).find((doc) => doc.name === "Copy")!;
    expect(copied.id).not.toBe(DEFAULT_WORKSPACE_ID);
    expect(copied.widgets.map((w) => w.type).sort()).toEqual(["news", "price-volume"]);
    const copiedNews = copied.widgets.find((w) => w.type === "news")!;
    expect(copiedNews.ticker).toBe("AAPL");
    expect(copiedNews.geometry).toEqual({ ...geometry, x: 50 });
    expect(copied.widgets.find((w) => w.type === "price-volume")?.params).toEqual(source[1].params);
    expect(copied.widgets.every((w) => !source.some((original) => original.id === w.id))).toBe(true);
    await click(find('[aria-label="Remove News · AAPL"]'));
    await click(button("Save"));
    expect(server.docs[copied.id].widgets.map((w) => w.type)).toEqual(["price-volume"]);
    await click(button("default *"));
    expect(find('[aria-label="News · AAPL"]').textContent).toContain("New chip");
    await click(button("Save"));
    expect(server.docs[DEFAULT_WORKSPACE_ID].widgets).toHaveLength(2);
    expect(server.docs[DEFAULT_WORKSPACE_ID].widgets.find((w) => w.type === "news")?.geometry.x).toBe(50);
    await click(button("+"));
    expect(find<HTMLInputElement>('input[type="checkbox"]').checked).toBe(false);
    await type('[aria-label="Workspace name"]', "Blank");
    await click(button("Create workspace"));
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(0);
    await click(button("Save"));
    await reload();
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(0);
    expect(find<HTMLInputElement>('[aria-label="Workspace name"]').value).toBe("Blank");
    await click(button("default"));
    expect(find('[aria-label="News · AAPL"]').textContent).toContain("New chip");
    expect(server.docs[ID].widgets).toEqual([]);
  });

  test("loads a saved empty workspace without restoring defaults", async () => {
    serve([]);
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
    expect(container().querySelector('a[href="https://www.sec.gov/filing"]')).not.toBeNull();
  });

  test("preserves geometry and ticker when replacing a panel, but removes obsolete chart settings", async () => {
    const server = serve([{ ...widget("price-volume"), params: { interval: "4h", style: "line" } }]);
    await render(<Page />);
    await type('[aria-label^="Ticker for Price"]', "AAPL");
    await click(button("Set ticker"));
    await change("Panel type for Price & volume · AAPL", "news");
    expect(find('[aria-label="News · AAPL"]').textContent).toContain("New chip");
    expect(server.puts()).toHaveLength(0);
    await click(button("Save"));
    expect(server.saved().widgets).toEqual([{ ...widget("news", "price-volume", "AAPL"), geometry }]);
  });

  test("keeps an input focused when its window moves to the front", async () => {
    const server = serve([widget("news", "z-news"), widget("monthly-returns", "a-returns")]);
    await render(<Page />);
    const input = find<HTMLInputElement>('[aria-label="Ticker for News"]');
    await act(async () => input.focus());
    await settle();
    expect(dom.window.document.activeElement).toBe(input);
    await type('[aria-label="Ticker for News"]', "AAPL");
    await click(find('[data-widget-type="news"] button[type="submit"]'));
    await click(button("Save"));
    expect(server.saved().widgets.at(-1)?.ticker).toBe("AAPL");
    expect(server.saved().widgets.at(-1)?.type).toBe("news");
  });

  test("keeps multiple windows of the same type independent", async () => {
    const server = serve([]);
    await render(<Page />);
    await add("news");
    await type('[aria-label="New window ticker"]', "AAPL");
    await add("news");
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(2);
    await click(button("Save"));
    expect(server.saved().widgets.map((w) => w.ticker)).toEqual(["NVDA", "AAPL"]);
    expect(new Set(server.saved().widgets.map((w) => w.id)).size).toBe(2);
  });

  test("moves and resizes by keyboard, clamps boundaries, and restores saved geometry", async () => {
    const server = serve([widget("news")]);
    await render(<Page />);
    const handle = find('[aria-label="Move or resize News · NVDA"]');
    await key(handle, "ArrowRight");
    await key(handle, "ArrowDown", true);
    await click(button("Save"));
    expect(server.saved().widgets[0].geometry).toEqual({ ...geometry, x: 50, height: 370 });
    await reload();
    await key(find('[aria-label="Move or resize News · NVDA"]'), "ArrowLeft");
    await click(button("Save"));
    expect(server.saved().widgets[0].geometry).toEqual({ ...geometry, height: 370 });
  });

  test("keeps window geometry inside its minimum bounds", async () => {
    const server = serve([{ ...widget("news"), geometry: { x: 0, y: 0, width: 280, height: 180 } }]);
    await render(<Page />);
    const handle = find('[aria-label="Move or resize News · NVDA"]');
    await key(handle, "ArrowLeft");
    await key(handle, "ArrowUp");
    await key(handle, "ArrowLeft", true);
    await key(handle, "ArrowUp", true);
    await click(button("Save"));
    expect(server.saved().widgets[0].geometry).toEqual({ x: 0, y: 0, width: 280, height: 180 });
  });

  test("refreshes one panel and preserves its sibling while maximizing", async () => {
    const server = serve([widget("monthly-returns"), widget("news")]);
    await render(<Page />);
    const news = find('[aria-label="News · NVDA"]');
    const reads = () => server.requests.filter((r) => r.path.startsWith("/api/candles")).length;
    const before = reads();
    await click(find('[aria-label="Refresh Monthly returns · NVDA"]'));
    expect(reads()).toBe(before + 1);
    await click(find('[aria-label="Expand Monthly returns · NVDA"]'));
    expect(find(".ws-widget-focus").getAttribute("aria-label")).toBe("Monthly returns · NVDA");
    expect(find('[aria-label="News · NVDA"]')).toBe(news);
    await escape();
    expect(container().querySelector(".ws-widget-focus")).toBeNull();
    await click(find('[aria-label="Remove News · NVDA"]'));
    expect(container().querySelector('[aria-label="News · NVDA"]')).toBeNull();
  });

  test("saves chart settings only with Save and restores them after reload", async () => {
    const server = serve([widget("price-volume")]);
    await render(<Page />);
    await change("Chart interval", "4h");
    await change("Chart style", "area");
    const rsi = [...container().querySelectorAll("label")].find((label) => label.textContent === "RSI")?.querySelector("input");
    if (!rsi) throw new Error("missing RSI indicator");
    await click(rsi);
    expect(server.puts()).toHaveLength(0);
    await click(button("Save"));
    expect(server.saved().widgets[0].params).toEqual({ interval: "4h", style: "area", studies: ["Volume", "RSI"] });
    await reload();
    expect(find<HTMLSelectElement>('select[aria-label="Chart interval"]').value).toBe("4h");
    const src = new URL(find<HTMLIFrameElement>("iframe").src).searchParams;
    expect([src.get("symbol"), src.get("interval"), src.get("style")]).toEqual(["NVDA", "240", "3"]);
  });

  test("keeps new edits unsaved when an earlier save completes", async () => {
    const server = serve([]);
    const { promise, resolve: finish } = Promise.withResolvers<void>();
    server.control.delaySave = () => promise;
    await render(<Page />);
    await type('[aria-label="Workspace name"]', "Earlier");
    await click(button("Save"));
    await type('[aria-label="Workspace name"]', "Latest");
    await act(async () => finish());
    await settle();
    expect(server.saved().name).toBe("Earlier");
    expect(find<HTMLInputElement>('[aria-label="Workspace name"]').value).toBe("Latest");
    expect(text()).toContain("Unsaved changes");
  });

  test("reports save failure without discarding edits and supports another Save", async () => {
    const server = serve([]);
    await render(<Page />);
    await add("news");
    server.control.failSave = true;
    await click(button("Save"));
    expect(text()).toContain("Save failed");
    expect(find('[aria-label="News · NVDA"]')).not.toBeNull();
    server.control.failSave = false;
    await click(button("Save"));
    expect(server.saved().widgets.map((w) => w.type)).toEqual(["news"]);
    expect(text()).toContain("Saved 12:00:00");
  });

  test("does not replace an unavailable collection with editable defaults", async () => {
    const server = serve();
    server.control.failLoad = true;
    await render(<Page />);
    expect(find('[role="alert"]').textContent).toContain("Workspaces unavailable");
    expect(container().querySelector(".ws-widget")).toBeNull();
    server.control.failLoad = false;
    await click(button("Retry"));
    expect(find<HTMLInputElement>('[aria-label="Workspace name"]').value).toBe("default");
    expect(container().querySelector('[data-widget-type="markets"]')).not.toBeNull();
  });

  test("adds all formerly fixed panels as interchangeable windows", async () => {
    serve([]);
    await render(<Page />);
    for (const type of ["watchlist", "portfolio", "agent", "theses", "alerts", "sessions", "journal"]) await add(type);
    expect(text()).toContain("New evidence");
    expect(text()).not.toContain("Old evidence");
    expect(text()).toContain("NVDA research");
    expect(text()).toContain("select a session for logs");
    expect(find('[data-widget-type="agent"]').textContent).toContain("Agent chat");
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(7);
  });

  test("applies valid receipts once, keeps their tickers, and never saves automatically", async () => {
    const server = serve([]);
    const first: AgentEvent = { type: "widget_update", ticker: "NVDA", widgetType: "short-volume" };
    const events: AgentEvent[] = [first, first, { type: "widget_update", ticker: "AAPL", widgetType: "insider" }, { type: "widget_update", ticker: "NVDA", widgetType: "bad" }];
    await render(<Page events={events} />);
    expect(find('[aria-label="Short volume · NVDA"]')).not.toBeNull();
    expect(find('[aria-label="Insider timeline · AAPL"]')).not.toBeNull();
    await click(find('[aria-label="Remove Short volume · NVDA"]'));
    await render(<Page events={[...events]} />);
    expect(container().querySelector('[aria-label="Short volume · NVDA"]')).toBeNull();
    expect(server.puts()).toHaveLength(0);
  });

  test("enforces the window limit when an agent fills an open link form's last slot", async () => {
    const server = serve(Array.from({ length: 23 }, (_, i) => widget("news", `news-${i}`)));
    await render(<Page />);
    await add("tv-link");
    await type('input[aria-label="TradingView URL"]', "https://www.tradingview.com/x/8ot166mV/");
    const outcomes: WidgetOutcomes[] = [];
    const onWidgetOutcomes = (o: WidgetOutcomes) => void outcomes.push(o);
    const added: AgentEvent = { type: "widget_update", ticker: "NVDA", widgetType: "short-volume" };
    await render(<Page events={[added]} onWidgetOutcomes={onWidgetOutcomes} />);
    await click(button("Add link"));
    const blocked: AgentEvent = { type: "widget_update", ticker: "NVDA", widgetType: "insider" };
    await render(<Page events={[added, blocked]} onWidgetOutcomes={onWidgetOutcomes} />);
    expect(container().querySelectorAll(".ws-widget")).toHaveLength(24);
    expect(find<HTMLSelectElement>('[aria-label="Add widget"]').disabled).toBe(true);
    expect(text()).toContain("Workspace limit: 24 widgets");
    expect(server.puts()).toHaveLength(0);
    expect(outcomes.map((o) => [...o])).toEqual([[[added, "added"]], [[blocked, "blocked"]]]);
    await click(container().querySelector('[aria-label="Remove News · NVDA"]')!);
    expect(find<HTMLSelectElement>('[aria-label="Add widget"]').disabled).toBe(false);
  });

  test("Escape closes a link form or indicator menu before leaving maximize", async () => {
    serve([widget("price-volume")]);
    await render(<Page />);
    await click(find('[aria-label^="Expand Price"]'));
    await add("tv-link");
    await escape(find('input[aria-label="TradingView URL"]'));
    expect(container().querySelector('form[aria-label="TradingView link"]')).toBeNull();
    expect(container().querySelector(".ws-widget-focus")).not.toBeNull();
    const menu = find<HTMLDetailsElement>(".ws-widget-focus details");
    menu.open = true;
    await escape(find(".ws-widget-focus summary"));
    expect(menu.open).toBe(false);
    expect(container().querySelector(".ws-widget-focus")).not.toBeNull();
    await escape();
    expect(container().querySelector(".ws-widget-focus")).toBeNull();
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
