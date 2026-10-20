// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import type { Root } from "react-dom/client";
import { ThesisPanel } from "./thesis-panel";

// react-dom reads canUseDOM once at module load, so it must load after the jsdom globals
// exist. A static import runs before this module body, so this one import stays dynamic.
const { createRoot } = await import("react-dom/client");

const realFetch = globalThis.fetch;
const realSetInterval = globalThis.setInterval;
let mounted: Root | null = null;

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
  globalThis.setInterval = realSetInterval;
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

function button(root: HTMLElement, text: string): HTMLButtonElement {
  const found = [...root.querySelectorAll("button")].find((b) => b.textContent?.includes(text));
  if (!found) throw new Error(`missing button ${text}`);
  return found;
}

const NVDA = { thesis_id: "thesis:1", slug: "nvda", status: "active", scope: "NVDA", user_thesis: "NVDA demand grows", updated_at: "", alerts: 2, pending: 1 };
const FUZZY = { ...NVDA, thesis_id: "thesis:2", slug: "rates", scope: "unknown", user_thesis: "rates fall", alerts: 0, pending: 0 };
const DETAIL = {
  ...NVDA,
  claims: [{ claim_id: "claim:1", statement: "Data-center demand grows", status: "unvalidated" }],
  assessment: "unresolved",
  watch_rules: [
    { rule_id: "rule:1", rule_type: "new_filing", enabled: true, support_status: "supported" },
    { rule_id: "rule:2", rule_type: "price_move", enabled: false, support_status: "unsupported" },
  ],
  journal: [{ entry_id: "journal:1", created_at: "2026-10-19T12:00:00+00:00", title: "Filing reviewed", body: "steady demand" }],
  last_tick: "2026-10-19T12:00:00+00:00",
};

describe("ThesisPanel", () => {
  test("lists theses, selects the ticker, shows detail, checks now, and polls every 60s", async () => {
    const requests: string[] = [];
    const delays: number[] = [];
    globalThis.setInterval = ((fn: () => void, ms?: number) => {
      delays.push(ms ?? 0);
      return realSetInterval(fn, ms);
    }) as typeof setInterval;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      requests.push(`${init?.method ?? "GET"} ${path}`);
      if (path === "/api/theses") return Response.json([NVDA, FUZZY]);
      if (path.endsWith("/check")) return Response.json({ triggers_created: [], runs: [], no_op: true });
      return Response.json(DETAIL);
    }) as unknown as typeof fetch;
    const selected: string[] = [];
    const root = await mount(<ThesisPanel onSelectTicker={(t) => selected.push(t)} />);
    await until(() => root.textContent?.includes("rates fall") ?? false);
    expect(delays).toContain(60_000);
    expect(root.querySelector('[aria-label="2 alerts, 1 pending"]')?.textContent).toBe("2");

    await click(button(root, "rates fall"));
    expect(selected).toEqual([]);
    await click(button(root, "NVDA demand grows"));
    expect(selected).toEqual(["NVDA"]);
    await until(() => root.textContent?.includes("Filing reviewed") ?? false);
    expect(root.querySelector('[aria-label="claims"]')?.textContent).toContain("Data-center demand grows");
    expect(root.querySelector('[aria-label="watch rules"]')?.textContent).toContain("watch price_move (unsupported)");
    expect(root.textContent).toContain("last tick 2026-10-19T12:00:00+00:00");

    const before = requests.filter((r) => r === "GET /api/theses/thesis%3A1").length;
    await click(button(root, "Check now"));
    await until(() => requests.filter((r) => r === "GET /api/theses/thesis%3A1").length > before);
    expect(requests).toContain("POST /api/theses/thesis%3A1/check");
  });

  test("shows a refused action and keeps the detail", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/theses") return Response.json([NVDA]);
      if (path.endsWith("/pause")) return Response.json({ detail: "thesis is closed; cannot change status" }, { status: 409 });
      return Response.json(DETAIL);
    }) as unknown as typeof fetch;
    const root = await mount(<ThesisPanel onSelectTicker={() => { }} />);
    await until(() => root.textContent?.includes("NVDA demand grows") ?? false);
    await click(button(root, "NVDA demand grows"));
    await until(() => root.textContent?.includes("Pause") ?? false);
    await click(button(root, "Pause"));
    await until(() => root.textContent?.includes("cannot change status") ?? false);
    expect(root.textContent).toContain("Filing reviewed");
  });
});
