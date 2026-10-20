// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { Root } from "react-dom/client";
import Home from "./Home";

// react-dom reads canUseDOM once at module load, so it must load after the jsdom globals exist.
const { createRoot } = await import("react-dom/client");

const realFetch = globalThis.fetch;
let mounted: Root | null = null;

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
  dom.window.localStorage.clear();
});

type Handler = (url: URL) => unknown;

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const ROWS = [
  { sym: "NVDA", last: 120.5, chg_pct: 2.25, volume: 1500000, short_pct: 1.2, last_filing: { form: "10-Q", filed: "2026-10-01" } },
  { sym: "GME", last: 25, chg_pct: -3.1, volume: 900000, short_pct: 22.4, last_filing: null },
];

const DEFAULTS: Record<string, Handler> = {
  "/api/sparklines": (u) => ({
    syms: Object.fromEntries((u.searchParams.get("syms") ?? "").split(",").map((s) => [s, { t: [1, 2, 3, 4, 5], closes: [10, 11, 10.5, 12, 13], as_of: null }])),
  }),
  "/api/briefing": () => ({
    date: "2026-10-20", syms: ["NVDA"], generated_at: "", cached: true, errors: [],
    items: [{ kind: "filing", sym: "NVDA", title: "NVDA files 8-K", link: "https://sec.example/8k", filed: "2026-10-19", filing_form: "8-K" }],
  }),
  "/api/scans": () => ({
    scans: [
      { id: "watchlist", name: "Watchlist", kind: "preset", filter: {}, created_at: null },
      { id: "most-shorted", name: "Most shorted", kind: "preset", filter: {}, created_at: null },
    ],
  }),
  "/api/markets": (u) => ({ scan: u.searchParams.get("scan"), sort: u.searchParams.get("sort"), universe: [], rows: ROWS, errors: [] }),
  "/api/positioning": (u) => ({
    sym: u.searchParams.get("sym"), short_pct: 12.5, short_as_of: "2026-10-15",
    insider_net_90d: { buys: 4000, sells: 1000, net: 3000 }, threshold_listed: false, threshold_as_of: null, errors: {},
  }),
  "/api/identity": (u) => ({ sym: u.searchParams.get("sym"), resolved: true, entity_id: "ent-1", security_id: "sec-1", method: "exact" }),
  "/api/portfolio": () => new Response(null, { status: 404 }),
};

function serve(over: Record<string, Handler> = {}): URL[] {
  const urls: URL[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "https://example.com");
    urls.push(url);
    const body = ({ ...DEFAULTS, ...over })[url.pathname]?.(url);
    return body instanceof Response ? body : json(body ?? {});
  }) as unknown as typeof fetch;
  return urls;
}

type Props = Parameters<typeof Home>[0];

function state(over: Partial<Props> = {}): Props {
  return { pick: () => { }, watchlist: ["NVDA", "AAPL"], onWatchlist: () => { }, setInput: () => { }, openDrawer: () => { }, ...over };
}

async function mount(s: Props): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(<Home {...s} />));
  return container;
}

async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 100 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 5)));
  }
  if (!ready()) throw new Error("the page did not reach the expected state");
}

async function click(target: Element): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

function button(root: HTMLElement, label: string): HTMLButtonElement {
  const hit = [...root.querySelectorAll("button")].find((b) => b.textContent === label || b.getAttribute("aria-label") === label);
  if (!hit) throw new Error(`missing button ${label}`);
  return hit;
}

const text = (root: Element) => root.textContent ?? "";

describe("Home", () => {
  test("index band draws one sparkline point per close, SPY first plus 3 watchlist extras", async () => {
    const urls = serve();
    const root = await mount(state({ watchlist: ["NVDA", "SPY", "AAPL", "MSFT", "TSLA"] }));
    await until(() => root.querySelector(".home-big polyline") !== null);
    const points = root.querySelector(".home-big polyline")?.getAttribute("points")?.split(" ") ?? [];
    expect(points).toHaveLength(5);
    expect(root.querySelector(".home-big b")?.textContent).toBe("SPY");
    expect(text(root.querySelector(".home-big")!)).toContain("+30.00%");
    const band = urls.find((u) => u.pathname === "/api/sparklines" && u.searchParams.get("syms")?.startsWith("SPY"));
    expect(band?.searchParams.get("syms")).toBe("SPY,QQQ,DIA,IWM,CL=F,GC=F,^TNX,NVDA,AAPL,MSFT");
  });

  test("briefing shows items and Refresh refetches with refresh=1", async () => {
    const urls = serve();
    const root = await mount(state());
    await until(() => text(root).includes("NVDA files 8-K"));
    expect(urls.filter((u) => u.pathname === "/api/briefing").map((u) => u.searchParams.get("refresh"))).toEqual([null]);
    await click(button(root, "Refresh"));
    await until(() => urls.some((u) => u.pathname === "/api/briefing" && u.searchParams.get("refresh") === "1"));
    await until(() => text(root).includes("NVDA files 8-K"));
  });

  test("markets table sorts through the sort query and the heart toggles the watchlist", async () => {
    const calls: string[][] = [];
    const urls = serve();
    const root = await mount(state({ onWatchlist: (s) => calls.push(s) }));
    await until(() => text(root).includes("GME"));
    expect(urls.find((u) => u.pathname === "/api/markets")?.searchParams.get("sort")).toBe("chg_pct");
    expect(text(root)).toContain("1,500,000");
    await click(button(root, "Volume"));
    await until(() => urls.some((u) => u.pathname === "/api/markets" && u.searchParams.get("sort") === "volume"));
    await until(() => text(root).includes("GME"));
    await click(button(root, "Add GME to watchlist"));
    await click(button(root, "Remove NVDA from watchlist"));
    expect(calls).toEqual([["NVDA", "AAPL", "GME"], ["AAPL"]]);
  });

  test("a scan click loads its table and Create your own prefills the agent input", async () => {
    const inputs: string[] = [];
    let opened = 0;
    const urls = serve();
    const root = await mount(state({ setInput: (v) => inputs.push(v), openDrawer: () => opened++ }));
    await until(() => text(root).includes("Most shorted"));
    expect(text(root)).toContain("No saved scans.");
    await click(button(root, "Most shorted"));
    await until(() => urls.some((u) => u.pathname === "/api/markets" && u.searchParams.get("scan") === "most-shorted"));
    await until(() => text(root).includes("Markets · Most shorted") && text(root).includes("GME"));
    await click(button(root, "Create your own"));
    expect(inputs).toEqual(["scan for "]);
    expect(opened).toBe(1);
  });

  test("detail shows identity and positioning for the first row, follows a row click, and remembers Hide summary", async () => {
    const urls = serve();
    const root = await mount(state());
    await until(() => text(root).includes("Insider buys 90d"));
    const detail = () => root.querySelector('[aria-label$=" detail"]')!;
    expect(detail().getAttribute("aria-label")).toBe("NVDA detail");
    expect(text(detail())).toContain("12.5%");
    expect(text(detail())).toContain("4,000");
    expect(text(detail())).toContain("entity ent-1");
    expect(detail().querySelector("a")?.getAttribute("href")).toBe("#/workspace/NVDA");
    const gme = [...root.querySelectorAll("tbody tr")].find((tr) => tr.textContent?.includes("GME"))!;
    await click(gme);
    await until(() => detail().getAttribute("aria-label") === "GME detail" && text(detail()).includes("Insider buys 90d"));
    expect(urls.some((u) => u.pathname === "/api/positioning" && u.searchParams.get("sym") === "GME")).toBe(true);
    await click(button(root, "Hide summary"));
    expect(text(detail())).not.toContain("Insider buys 90d");
    expect(dom.window.localStorage.getItem("stockbot.home.summary")).toBe("hidden");
  });

  test("an empty scan shows one line with a suggested prompt", async () => {
    serve({ "/api/markets": () => ({ scan: "watchlist", sort: "chg_pct", universe: [], rows: [], errors: [] }) });
    const root = await mount(state());
    await until(() => text(root).includes("No rows for this scan. Try: Ask Stockbot:"));
    expect(root.querySelector('[aria-label$=" detail"]')).toBeNull();
  });

  test("a failed request shows the error and Retry loads it", async () => {
    let fail = true;
    serve({ "/api/markets": (u) => (fail ? json({ detail: "down" }, 500) : DEFAULTS["/api/markets"](u)) });
    const root = await mount(state());
    await until(() => text(root).includes("We could not load this data."));
    expect(text(root)).not.toContain("request failed: 500");
    expect(root.querySelector(".skeleton")).toBeNull();
    fail = false;
    const retry = [...root.querySelectorAll(".home-error button")][0]!;
    await click(retry);
    await until(() => text(root).includes("GME"));
  });

  test("shows a skeleton while requests are in flight", async () => {
    globalThis.fetch = (() => new Promise<Response>(() => { })) as unknown as typeof fetch;
    const root = await mount(state());
    expect(root.querySelectorAll(".skeleton").length).toBeGreaterThan(0);
  });
});
