// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "../test-dom";
import { afterAll, afterEach, describe, expect, jest, mock, test } from "bun:test";
import { act } from "react";
import type { Root } from "react-dom/client";
import type { FeedItem } from "../lib/market-data";
import Feed from "./Feed";

// react-dom reads canUseDOM at module load, so it must load after the jsdom globals (see App.test.tsx).
const { createRoot } = await import("react-dom/client");

const realFetch = globalThis.fetch;
const realSetTimeout = globalThis.setTimeout;
let mounted: Root | null = null;
let hidden = false;
// jsdom reports a hidden document by default; the poll tests drive visibility themselves.
Object.defineProperty(dom.window.document, "hidden", { configurable: true, get: () => hidden });
afterAll(() => Reflect.deleteProperty(dom.window.document, "hidden"));

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
  hidden = false;
  jest.useRealTimers();
});

function item(id: string, at: string, extra: Partial<FeedItem> = {}): FeedItem {
  return { id, kind: "news", sym: "NVDA", title: `title ${id}`, link: `https://news.example/${id}`, at, source: "wire", ...extra };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function page(items: FeedItem[]) {
  return json({ items, as_of: "2026-10-20T12:00:00Z", errors: [] });
}

// Each answer serves one /api/feed request in order; the last one repeats.
function serve(...answers: (() => Response | Promise<Response>)[]): string[] {
  const urls: string[] = [];
  globalThis.fetch = ((input: string | URL | Request) => {
    urls.push(String(input));
    const answer = answers[Math.min(urls.length, answers.length) - 1];
    return answer ? Promise.resolve(answer()) : Promise.reject(new Error("unexpected request"));
  }) as unknown as typeof fetch;
  return urls;
}

async function mount(props: { setInput?: (value: string) => void; openDrawer?: () => void } = {}): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(<Feed selected="NVDA" watchlist={["NVDA", "AAPL"]} setInput={props.setInput ?? (() => { })} openDrawer={props.openDrawer} />));
  return container;
}

async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 200 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => realSetTimeout(resolve, 10)));
  }
  if (!ready()) throw new Error("the feed did not reach the expected state");
}

async function settle(): Promise<void> {
  for (let tick = 0; tick < 40; tick++) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

async function poll(): Promise<void> {
  await act(async () => {
    jest.advanceTimersByTime(60_000);
  });
  await settle();
}

function titles(root: HTMLElement): string[] {
  return [...root.querySelectorAll(".feed-title")].map((node) => node.textContent ?? "");
}

function button(root: HTMLElement, label: string): HTMLButtonElement {
  const found = [...root.querySelectorAll("button")].find((entry) => entry.textContent === label);
  if (!found) throw new Error(`missing ${label} button`);
  return found;
}

async function click(target: Element): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

async function change(target: HTMLInputElement | HTMLSelectElement, value: string): Promise<void> {
  await act(async () => {
    const proto = target instanceof dom.window.HTMLSelectElement ? dom.window.HTMLSelectElement.prototype : dom.window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value")?.set?.call(target, value);
    target.dispatchEvent(new dom.window.Event(target instanceof dom.window.HTMLSelectElement ? "change" : "input", { bubbles: true }));
  });
}

describe("Feed page", () => {
  test("shows skeleton rows, then the watchlist feed newest first", async () => {
    let release: (response: Response) => void = () => { };
    const urls = serve(() => new Promise<Response>((resolve) => (release = resolve)));
    const root = await mount();
    expect(root.querySelectorAll(".feed-skeleton").length).toBe(5);
    await act(async () => release(page([item("a", "2026-10-20T09:00:00Z"), item("c", "2026-10-20T11:00:00Z"), item("b", "2026-10-20T10:00:00Z")])));
    await until(() => titles(root).length === 3);
    expect(titles(root)).toEqual(["title c", "title b", "title a"]);
    expect(urls[0]).toBe("/api/feed?syms=NVDA%2CAAPL");
    expect(root.querySelector(".feed-skeleton")).toBeNull();
  });

  test("polls every 60s with since, skips while hidden, and prepends at the top", async () => {
    jest.useFakeTimers();
    const urls = serve(
      () => page([item("a", "2026-10-20T09:00:00Z"), item("b", "2026-10-20T10:00:00Z")]),
      () => page([item("b", "2026-10-20T10:00:00Z"), item("n", "2026-10-20T11:00:00Z")]),
    );
    const root = await mount();
    await settle();
    expect(titles(root)).toEqual(["title b", "title a"]);

    hidden = true;
    await poll();
    expect(urls.length).toBe(1);

    hidden = false;
    await poll();
    expect(urls[1]).toBe(`/api/feed?syms=NVDA%2CAAPL&since=${encodeURIComponent("2026-10-20T10:00:00Z")}`);
    // The repeated item b is deduplicated by id.
    expect(titles(root)).toEqual(["title n", "title b", "title a"]);
  });

  test("holds new items behind an N new pill while scrolled down", async () => {
    jest.useFakeTimers();
    serve(
      () => page([item("a", "2026-10-20T09:00:00Z")]),
      () => page([item("n", "2026-10-20T11:00:00Z"), item("m", "2026-10-20T10:00:00Z")]),
    );
    const root = await mount();
    await settle();
    const scroller = root.querySelector<HTMLElement>(".page-scroll")!;
    Object.defineProperty(scroller, "scrollTop", { configurable: true, writable: true, value: 300 });
    await poll();
    expect(titles(root)).toEqual(["title a"]);
    await click(button(root, "2 new"));
    expect(titles(root)).toEqual(["title n", "title m", "title a"]);
    expect(scroller.scrollTop).toBe(0);
    expect([...root.querySelectorAll("button")].some((entry) => entry.textContent?.endsWith(" new"))).toBe(false);
  });

  test("kind and ticker filters hide rows and show the empty line", async () => {
    serve(() => page([
      item("n", "2026-10-20T11:00:00Z"),
      item("f", "2026-10-20T10:00:00Z", { kind: "filing", sym: "AAPL" }),
    ]));
    const root = await mount();
    await until(() => titles(root).length === 2);
    await change(root.querySelector<HTMLSelectElement>('[aria-label="Kind"]')!, "filing");
    expect(titles(root)).toEqual(["title f"]);
    await change(root.querySelector<HTMLInputElement>('[aria-label="Ticker filter"]')!, "nvda");
    expect(titles(root)).toEqual([]);
    expect(root.textContent).toContain("No items for this filter — try Ask Stockbot: what's new with NVDA?");
  });

  test("Ask about this fills the agent input, opens the drawer, and focuses the prompt", async () => {
    serve(() => page([item("n", "2026-10-20T11:00:00Z", { kind: "filing", title: "8-K current report" })]));
    const setInput = mock((_value: string) => { });
    const openDrawer = mock(() => { });
    const prompt = dom.window.document.createElement("input");
    prompt.setAttribute("aria-label", "Ask the research agent");
    dom.window.document.body.append(prompt);
    try {
      const root = await mount({ setInput, openDrawer });
      await until(() => titles(root).length === 1);
      await click(button(root, "Ask about this"));
      expect(setInput).toHaveBeenCalledWith("About filing NVDA: 8-K current report (https://news.example/n) — ");
      expect(openDrawer).toHaveBeenCalledTimes(1);
      await until(() => dom.window.document.activeElement === prompt);
    } finally {
      prompt.remove();
    }
  });

  test("Open source links out, alerts open the workspace, and a null link shows no link", async () => {
    serve(() => page([
      item("n", "2026-10-20T11:00:00Z"),
      item("t", "2026-10-20T10:00:00Z", { kind: "alert", sym: "AAPL", link: undefined, source: "thesis" }),
      item("x", "2026-10-20T09:00:00Z", { link: undefined, sym: undefined }),
    ]));
    const root = await mount();
    await until(() => titles(root).length === 3);
    const rows = [...root.querySelectorAll(".feed-row")];
    const open = (row: Element) => [...row.querySelectorAll("a")].find((a) => a.textContent === "Open source") ?? null;
    expect(open(rows[0]!)?.getAttribute("href")).toBe("https://news.example/n");
    expect(open(rows[0]!)?.getAttribute("target")).toBe("_blank");
    expect(open(rows[0]!)?.getAttribute("rel")).toBe("noreferrer");
    expect(open(rows[1]!)?.getAttribute("href")).toBe("#/workspace/AAPL");
    expect(open(rows[1]!)?.hasAttribute("target")).toBe(false);
    expect(open(rows[2]!)).toBeNull();
    // The ticker chip opens that ticker's workspace.
    expect([...rows[0]!.querySelectorAll("a")].find((a) => a.textContent === "NVDA")?.getAttribute("href")).toBe("#/workspace/NVDA");
  });

  test("a failed first load shows the error and Retry reloads", async () => {
    const urls = serve(() => json({ detail: "feed down" }, 503), () => page([item("a", "2026-10-20T09:00:00Z")]));
    const root = await mount();
    await until(() => root.querySelector('[role="alert"]') !== null);
    expect(root.querySelector('[role="alert"]')?.textContent).toContain("Feed unavailable");
    await click(button(root, "Retry"));
    await until(() => titles(root).length === 1);
    expect(urls.length).toBe(2);
    expect(root.querySelector('[role="alert"]')).toBeNull();
  });
});
