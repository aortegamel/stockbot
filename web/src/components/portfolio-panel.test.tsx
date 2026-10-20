// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import type { Root } from "react-dom/client";
import { PortfolioPanel, type PortfolioView } from "./portfolio-panel";

// react-dom reads canUseDOM once at module load, so it must load after the jsdom globals exist.
const { createRoot } = await import("react-dom/client");

const realFetch = globalThis.fetch;
let mounted: Root | null = null;

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
});

async function mount(element: ReactNode): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(element));
  return container;
}

async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 100 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 5)));
  }
  if (!ready()) throw new Error("the panel did not reach the expected state");
}

async function click(target: Element): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

const cached: PortfolioView = {
  synced_at: "2026-10-19T12:00:00+00:00",
  cash: "100",
  invested_value: "1168.40",
  total_value: "1268.40",
  positions: [
    { ticker: "WING", quantity: "10", average_cost: "95.5", market_price: "116.84", market_value: "1168.40", unrealized_gain: "213.40", unrealized_gain_pct: "0.2234" },
    { ticker: "AAPL", quantity: "10", average_cost: "95.5", market_price: null, market_value: null, unrealized_gain: null, unrealized_gain_pct: null },
  ],
};

describe("PortfolioPanel", () => {
  test("renders nothing when the portfolio API answers 404", async () => {
    const asked: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      asked.push(String(input));
      return Response.json({ detail: "Run `venv/bin/python cli.py robinhood-login`" }, { status: 404 });
    }) as unknown as typeof fetch;
    const root = await mount(<PortfolioPanel onSelectTicker={() => { }} />);
    await until(() => asked.length === 1);
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 10)));
    expect(asked).toEqual(["/api/portfolio"]);
    expect(root.innerHTML).toBe("");
  });

  test("shows cached holdings and last sync, and a position click selects the chart ticker", async () => {
    globalThis.fetch = (async () => Response.json(cached)) as unknown as typeof fetch;
    const picked: string[] = [];
    const root = await mount(<PortfolioPanel onSelectTicker={(t) => picked.push(t)} />);
    await until(() => (root.textContent ?? "").includes("WING"));
    expect(root.textContent).toContain("last sync 2026-10-19T12:00:00+00:00");
    expect(root.textContent).toContain("total 1268.40");
    const rows = root.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);
    expect(rows[0]?.textContent).toContain("95.50");
    expect(rows[0]?.textContent).toContain("1168.40");
    expect(rows[0]?.textContent).toContain("213.40 (22.34%)");
    expect(rows[1]?.textContent).toContain("—");
    const ticker = rows[1]?.querySelector("button");
    if (!ticker) throw new Error("missing ticker button");
    await click(ticker);
    expect(picked).toEqual(["AAPL"]);
  });

  test("Sync posts once and replaces the holdings; a failed sync keeps the table and shows the error", async () => {
    const calls: string[] = [];
    let fail = false;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push(`${init?.method ?? "GET"} ${String(input)}`);
      if (init?.method !== "POST") return Response.json({ ...cached, synced_at: null, positions: [] });
      if (fail) return new Response("boom", { status: 500 });
      return Response.json(cached);
    }) as unknown as typeof fetch;
    const root = await mount(<PortfolioPanel onSelectTicker={() => { }} />);
    await until(() => (root.textContent ?? "").includes("never synced"));
    expect(root.textContent).toContain("no holdings in the last sync");
    const sync = [...root.querySelectorAll("button")].find((b) => b.textContent === "Sync");
    if (!sync) throw new Error("missing Sync button");
    await click(sync);
    await until(() => (root.textContent ?? "").includes("WING"));
    expect(calls).toEqual(["GET /api/portfolio", "POST /api/portfolio/sync"]);
    fail = true;
    await click(sync);
    await until(() => (root.textContent ?? "").includes("sync failed: request failed: 500"));
    expect(root.textContent).toContain("WING");
  });
});
