// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "./test-dom";
import { afterEach, describe, expect, jest, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import type { Root } from "react-dom/client";
import App from "./App";
import type { Metrics } from "../../needle-harness/lib/agent/types";

// react-dom reads canUseDOM once at module load, so it must load after the jsdom globals
// exist. A static import runs before this module body, so this one import stays dynamic.
const { createRoot } = await import("react-dom/client");

const realFetch = globalThis.fetch;
const realSetTimeout = globalThis.setTimeout;
const encoder = new TextEncoder();
let mounted: Root | null = null;

// The terminal's own watchlist, in the order the agent receives it.
const WATCHLIST = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "AMZN", "META", "GOOG"];

const AGENT_PROMPT = '[aria-label="Ask the research agent"]';

afterEach(async () => {
  await unmount();
  globalThis.fetch = realFetch;
  jest.useRealTimers();
  // The hash route survives in the shared jsdom URL; each test starts on the default page.
  dom.window.history.replaceState(null, "", "/");
  dom.window.localStorage.clear();
});

type Handler = (url: string, init: RequestInit | undefined) => Response | Promise<Response> | undefined;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

// One fetch stub for the whole terminal. `extra` answers the interesting URLs;
// everything else falls through to an idle terminal so the test reads one behavior.
function serve(extra: Handler = () => undefined): { urls: string[]; bodies: string[] } {
  const urls: string[] = [];
  const bodies: string[] = [];
  globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    urls.push(url);
    if (typeof init?.body === "string") bodies.push(init.body);
    const handled = extra(url, init);
    if (handled !== undefined) return Promise.resolve(handled);
    if (url.includes("/api/health")) return Promise.resolve(json({ debug: false }));
    if (url.includes("/api/quote")) return Promise.resolve(json({ sym: "X", price: 100 }));
    if (url.includes("/api/watchlist")) {
      const symbols = new URL(url, dom.window.location.href).searchParams.get("symbols")?.split(",") ?? [];
      return Promise.resolve(json({
        watchlist: symbols.map((symbol) => ({ symbol, source: "sec_filings", tool: "sec-filings", status: "no_records" })),
      }));
    }
    if (url.includes("/api/filings")) return Promise.resolve(json({ cik: 1, items: [] }));
    if (url.includes("/api/news")) return Promise.resolve(json({ sym: "X", items: [] }));
    if (url.includes("/api/research/sessions")) return Promise.resolve(json({ sessions: [] }));
    if (url.includes("/api/theses")) return Promise.resolve(json([]));
    if (url.includes("/api/portfolio")) return Promise.resolve(new Response(null, { status: 404 }));
    return Promise.reject(new Error(`unexpected request: ${url}`));
  }) as unknown as typeof fetch;
  return { urls, bodies };
}

function metrics(): Metrics {
  return {
    totalMs: 124,
    needle: { calls: 2, totalMs: 10, escalations: 0 },
    tools: { calls: 3, totalMs: 33 },
    muse: { calls: 1, totalMs: 78, inputTokens: 10, outputTokens: 20 },
    evidence: { count: 4, characters: 500 },
    failures: {},
  };
}

// A stream the test drives chunk by chunk, the way the server writes while it works.
function liveStream(): { stream: ReadableStream<Uint8Array>; send: (text: string) => void; close: () => void } {
  let control: ReadableStreamDefaultController<Uint8Array> | null = null;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      control = controller;
    },
  });
  return {
    stream,
    send: (text: string) => control?.enqueue(encoder.encode(text)),
    close: () => control?.close(),
  };
}

// A stream that never writes, so only the caller's abort ends it.
function silentStream(signal: AbortSignal | null | undefined): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      signal?.addEventListener("abort", () => controller.error(new Error("request aborted")));
    },
  });
}

// The stall watchdog runs on window timers, so the harness must let fake timers own them.
function patchWindowTimers(): () => void {
  const saved: [string, PropertyDescriptor | undefined][] = [];
  for (const key of ["setTimeout", "clearTimeout"]) {
    saved.push([key, Object.getOwnPropertyDescriptor(dom.window, key)]);
    Object.defineProperty(dom.window, key, { configurable: true, writable: true, value: (globalThis as Record<string, unknown>)[key] });
  }
  return () => {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(dom.window, key, descriptor);
      else Reflect.deleteProperty(dom.window, key);
    }
  };
}

async function mount(element: ReactNode, waitForWatchlist = true): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(element));
  if (waitForWatchlist) await until(() => text(container).includes("NVDA100DELAYEDSEC none"));
  return container;
}

async function unmount(): Promise<void> {
  await act(async () => mounted?.unmount());
  mounted = null;
}

async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 200 && !ready(); tick++) {
    // Poll real time even when the test controls application timers.
    await act(() => new Promise<void>((resolve) => realSetTimeout(resolve, 10)));
  }
  if (!ready()) throw new Error("the terminal did not reach the expected state");
}

// Microtask drain for tests that run on fake timers, where no real timer can fire.
async function settle(): Promise<void> {
  for (let tick = 0; tick < 40; tick++) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

function find<T extends Element>(root: HTMLElement, selector: string): T {
  const found = root.querySelector<T>(selector);
  if (!found) throw new Error(`missing element ${selector}`);
  return found;
}

function findButton(root: HTMLElement, label: string): HTMLButtonElement | null {
  return [...root.querySelectorAll("button")].find((entry) => entry.textContent === label) ?? null;
}

function findLink(root: HTMLElement, label: string): HTMLAnchorElement | null {
  return [...root.querySelectorAll("a")].find((entry) => entry.textContent === label) ?? null;
}

const PALETTE = 'input[aria-label="Search or do anything"]';

// The header has one search field: the closed palette trigger.
async function openPalette(root: HTMLElement): Promise<HTMLInputElement> {
  await click(button(root, "Search or do anything /"));
  return find<HTMLInputElement>(root, PALETTE);
}

function options(root: HTMLElement): string[] {
  return [...root.querySelectorAll('#cmd-list [role="option"]')].map((option) => (option.textContent ?? "").replace(/\s+/g, " ").trim());
}

function button(root: HTMLElement, label: string): HTMLButtonElement {
  const found = findButton(root, label);
  if (!found) throw new Error(`missing ${label} button`);
  return found;
}

async function click(target: Element): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

async function type(target: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value")?.set;
    setter?.call(target, value);
    target.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  });
}

async function press(target: Element, key: string): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key, bubbles: true }));
  });
}

function text(root: HTMLElement): string {
  return root.textContent ?? "";
}

describe("App terminal shell", () => {
  test("renders the watchlist, strip, and backend state on load", async () => {
    serve();
    const root = await mount(<App />);
    await until(() => text(root).includes("no research sessions yet"));
    expect(text(root)).toContain("NEEDLE TERMINAL");
    expect(text(root)).toContain("WATCHLIST · DELAYED");
    expect(text(root)).toContain("AGENT · viewing NVDA");
    expect(text(root)).toContain("delayed feed · Yahoo OHLCV");
    expect(text(root)).toContain("select a session for logs");
    expect(findButton(root, "Search or do anything /")).not.toBeNull();
    expect(root.querySelectorAll("header.app-header input")).toHaveLength(0);
    expect(text(root)).not.toContain("· DEBUG");
    expect(findLink(root, "Harness")).toBeNull();
  });

  test("shows the debug label only when the health endpoint reports debug", async () => {
    serve((url) => (url.includes("/api/health") ? json({ debug: true, version: 1 }) : undefined));
    const root = await mount(<App />);
    await until(() => text(root).includes("AGENT · viewing NVDA · DEBUG"));
  });

  test("keeps the debug label off when the health endpoint fails", async () => {
    serve((url) => (url.includes("/api/health") ? json({ error: "unavailable" }, 500) : undefined));
    const root = await mount(<App />);
    await until(() => text(root).includes("WATCHLIST · DELAYED"));
    await settle();
    expect(text(root)).not.toContain("· DEBUG");
  });

  test("reports an empty session list when the sessions request fails", async () => {
    serve((url) => (url.includes("/api/research/sessions") ? json({ error: "db locked" }, 500) : undefined));
    const root = await mount(<App />);
    await until(() => text(root).includes("no research sessions yet"));
  });

  test("waits for delayed quotes and keeps the placeholder for a failed quote", async () => {
    const holder: { release?: (response: Response) => void } = {};
    const gate = new Promise<Response>((resolve) => {
      holder.release = resolve;
    });
    serve((url) => {
      if (!url.includes("/api/quote")) return undefined;
      if (url.includes("sym=AAPL")) return Promise.reject(new Error("quote upstream down"));
      // One Response body can be read once, so every request builds its own.
      return url.includes("sym=NVDA") ? gate.then(() => json({ sym: "NVDA", price: 100 })) : json({ sym: "X", price: 100 });
    });
    const root = await mount(<App />, false);
    await until(() => text(root).includes("NVDA—DELAYEDSEC none"));
    expect(text(root)).not.toContain("NVDA100DELAYED");
    await act(async () => {
      holder.release?.(json({ sym: "NVDA", price: 100 }));
    });
    await until(() => text(root).includes("NVDA100DELAYED"));
    await settle();
    expect(text(root)).toContain("AAPL—DELAYED");
  });
});

describe("App ticker search", () => {
  test("the palette lists ticker hits and Enter adopts the first hit in its workspace", async () => {
    const { urls } = serve((url) =>
      url.includes("/api/search")
        ? json({ query: "br", hits: [{ sym: "brk.b", name: "Berkshire Hathaway" }, { sym: "BRK.A" }] })
        : undefined,
    );
    const root = await mount(<App />);
    const search = await openPalette(root);
    await type(search, "br");
    await until(() => options(root)[0] === "brk.b Berkshire Hathaway");
    expect(options(root)).toEqual(["brk.b Berkshire Hathaway", "BRK.A", "Ask Stockbot: br"]);
    await press(search, "Enter");
    await until(() => root.querySelector("footer")?.lastElementChild?.textContent === "BRK.B");
    // A palette pick always opens the ticker workspace.
    expect(dom.window.location.hash).toBe("#/workspace/BRK.B");
    expect(root.querySelector(PALETTE)).toBeNull();
    // The selected ticker joins the front of the watchlist the agent receives.
    await until(() => urls.some((url) => url.includes("symbols=BRK.B%2CNVDA")));
    await until(() => text(root).includes("BRK.B100DELAYEDSEC none"));
  });

  test("Escape closes without a pick and a blank query offers no ticker or Ask row", async () => {
    serve((url) => (url.includes("/api/search") ? json({ query: "aa", hits: [{ sym: "AAPL" }] }) : undefined));
    const root = await mount(<App />);
    let search = await openPalette(root);
    await type(search, "aa");
    // The Tickers hit precedes the matching watchlist row.
    await until(() => options(root).join("|") === "AAPL|AAPL|Ask Stockbot: aa");
    await press(search, "Escape");
    expect(root.querySelector(PALETTE)).toBeNull();
    expect(root.querySelector("footer")?.lastElementChild?.textContent).toBe("NVDA");
    search = await openPalette(root);
    expect(search.value).toBe("");
    await type(search, "  ");
    await settle();
    const groups = [...root.querySelectorAll(".cmd-group")].map((g) => g.textContent);
    expect(groups).not.toContain("Tickers");
    expect(groups).not.toContain("Ask");
    expect(root.querySelector("footer")?.lastElementChild?.textContent).toBe("NVDA");
  });

  test("a failed search shows one notice row and hides the raw error", async () => {
    const { urls } = serve((url) => (url.includes("/api/search") ? Promise.reject(new Error("search down")) : undefined));
    const root = await mount(<App />);
    await type(await openPalette(root), "zz");
    await until(() => urls.some((url) => url.includes("/api/search")));
    await until(() => text(root).includes("ticker search unavailable"));
    expect(options(root)).toEqual(["Ask Stockbot: zz"]);
    expect(text(root)).not.toContain("search down");
  });
});

describe("App agent send", () => {
  test("research prefixes and the persistent toggle set mode without changing default chat", async () => {
    const requests: Record<string, unknown>[] = [];
    serve((url, init) => {
      if (!url.includes("/api/agent")) return undefined;
      requests.push(JSON.parse(String(init?.body)));
      return new Response(`data: ${JSON.stringify({ type: "done", metrics: metrics() })}\n\n`);
    });
    const root = await mount(<App />);
    const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
    const toggle = find<HTMLInputElement>(root, '[aria-label="Research mode"]');
    const send = async (value: string, mode?: "research") => {
      const before = requests.length;
      await type(prompt, value);
      await click(button(root, "Send"));
      await until(() => requests.length === before + 1 && findButton(root, "Send") !== null);
      expect(requests.at(-1)).toEqual({
        prompt: value.replace(/^\s*\/research\b\s*/i, "").trim(),
        ticker: "NVDA", currentChartTicker: "NVDA", watchlist: WATCHLIST,
        ...(mode ? { mode } : {}),
      });
    };
    await send("/research what changed", "research");
    expect(toggle.checked).toBe(false);
    await send("hello");
    await click(toggle);
    await send("what changed", "research");
    expect(toggle.checked).toBe(true);
    await send("explain more", "research");
    await click(toggle);
    await send("hello again");
    const before = requests.length;
    await type(prompt, " /ReSeArCh ");
    await click(button(root, "Send"));
    await settle();
    expect(requests).toHaveLength(before);
    expect(prompt.value).toBe(" /ReSeArCh ");
  });

  test("the palette parses the research prefix", async () => {
    const { bodies } = serve((url) => {
      if (url.includes("/api/search")) return json({ hits: [] });
      if (url.includes("/api/scans")) return new Response(null, { status: 404 });
      if (url.includes("/api/agent")) return new Response(`data: ${JSON.stringify({ type: "done", metrics: metrics() })}\n\n`);
      return undefined;
    });
    const root = await mount(<App />);
    const palette = await openPalette(root);
    await type(palette, " /ReSeArCh what changed");
    await until(() => options(root).some((option) => option.includes("Ask Stockbot:")));
    await click(find<HTMLElement>(root, '#cmd-list [role="option"]:last-child'));
    await until(() => bodies.some((body) => body.includes('"mode":"research"')));
    expect(JSON.parse(bodies.at(-1) ?? "{}")).toEqual({
      prompt: "what changed", mode: "research",
      ticker: "NVDA", currentChartTicker: "NVDA", watchlist: WATCHLIST,
    });
  });

  test("streams one run and pins the prompt, ticker, and full watchlist scope", async () => {
    const live = liveStream();
    const { urls, bodies } = serve((url) => {
      if (url.includes("/api/health")) return json({ debug: true });
      if (url.includes("/api/agent")) return new Response(live.stream, { status: 200 });
      return undefined;
    });
    const root = await mount(<App />);
    const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
    await type(prompt, "nvda risks?");
    await click(button(root, "Send"));
    await settle();
    expect(button(root, "Thinking…").disabled).toBe(true);
    expect(prompt.disabled).toBe(true);
    expect(urls.filter((url) => url.includes("/api/agent")).length).toBe(1);
    expect(bodies.filter((body) => body.includes("nvda risks?"))).toHaveLength(1);
    expect(JSON.parse(bodies.at(-1) ?? "{}")).toEqual({ prompt: "nvda risks?", ticker: "NVDA", currentChartTicker: "NVDA", watchlist: WATCHLIST });

    await act(async () => {
      // Frames split across chunks, a keepalive frame with no data line, and one malformed frame.
      live.send('data: {"type":"agent_start","prompt":"nvda risks?"}\n\ndata: {"type":"reasoning_start","');
      live.send('"model":"stockbot"}\n\n: keepalive\n\n');
      live.send('data: {"type":"tool_start","tool":"search_sec_filings"}\n\ndata: {broken json\n\n');
      live.send('data: {"type":"answer_delta","text":"Risk grew"}\n\ndata: {"type":"answer_delta","text":" in 2026."}\n\n');
      live.send(`data: {"type":"done","metrics":${JSON.stringify(metrics())}}\n\n`);
      live.close();
    });
    await until(() => text(root).includes("Risk grew in 2026."));
    await until(() => findButton(root, "Send") !== null);
    expect(text(root)).not.toContain("├─");
    expect(text(root)).not.toContain("search_sec_filings");
    expect(text(root)).toContain("── metrics ──");
    expect(prompt.disabled).toBe(false);
    // The terminal event arrived, so the closed stream is not a failure.
    expect(text(root)).not.toContain("stream ended without a terminal event");
    expect(text(root)).not.toContain("broken json");
  });

  test.each(["AAPL", " brk.b ", "RDS-A"])("applies chart_update for %s and continues the answer", async (ticker) => {
    const live = liveStream();
    const { urls, bodies } = serve((url) => (url.includes("/api/agent") ? new Response(live.stream, { status: 200 }) : undefined));
    const root = await mount(<App />);
    const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
    const question = "Compare BRK.B with NVDA?  Keep this wording.";
    await type(prompt, question);
    await click(button(root, "Send"));
    await settle();
    expect(JSON.parse(bodies.at(-1) ?? "{}")).toEqual({ prompt: question, ticker: "NVDA", currentChartTicker: "NVDA", watchlist: WATCHLIST });
    await act(async () => {
      live.send('data: {"type":"answer_delta","text":"The comparison "}\n\n');
      live.send(`data: ${JSON.stringify({ type: "chart_update", ticker })}\n\n`);
      live.send('data: {"type":"answer_delta","text":"continues."}\n\n');
      live.send(`data: {"type":"done","metrics":${JSON.stringify(metrics())}}\n\n`);
      live.close();
    });
    const selected = ticker.trim().toUpperCase();
    await until(() => root.querySelector("footer")?.lastElementChild?.textContent === selected);
    expect(text(root)).toContain(`AGENT · viewing ${selected}`);
    expect(text(root)).toContain(`Financials · ${selected}`);
    expect(text(root)).toContain(`News · ${selected}`);
    expect(root.querySelector(`[role="region"][aria-label="${selected} price chart"]`)).not.toBeNull();
    const chart = find<HTMLIFrameElement>(root, `iframe[title="${selected} TradingView chart"]`);
    expect(text(root)).toContain("The comparison continues.");
    await until(() => findButton(root, "Send") !== null);
    expect(text(root)).not.toContain("stream ended without a terminal event");
  });

  test.each([undefined, null, 12, {}, ["AAPL"], "", "   ", "AAPL MSFT", "AAPL<script>", "ABCDEFGHIJK", "^GSPC", "SEC"].map((ticker) => ({ ticker })))(
    "ignores an invalid chart_update ticker %j without interrupting the answer",
    async ({ ticker }) => {
      const live = liveStream();
      serve((url) => (url.includes("/api/agent") ? new Response(live.stream, { status: 200 }) : undefined));
      const root = await mount(<App />);
      await type(find<HTMLInputElement>(root, AGENT_PROMPT), "Explain the risks");
      await click(button(root, "Send"));
      await settle();
      const chart = find<HTMLIFrameElement>(root, 'iframe[title="NVDA TradingView chart"]');
      await act(async () => {
        live.send(`data: ${JSON.stringify({ type: "chart_update", ticker })}\n\n`);
        live.send('data: {"type":"answer_delta","text":"The answer still streams."}\n\n');
        live.send(`data: {"type":"done","metrics":${JSON.stringify(metrics())}}\n\n`);
        live.close();
      });
      await until(() => text(root).includes("The answer still streams."));
      await until(() => findButton(root, "Send") !== null);
      expect(root.querySelector("footer")?.lastElementChild?.textContent).toBe("NVDA");
      expect(text(root)).toContain("AGENT · viewing NVDA");
      expect(root.querySelector('iframe[title="NVDA TradingView chart"]')).toBe(chart);
      expect(new URL(chart.src).searchParams.get("symbol")).toBe("NVDA");
      expect(text(root)).not.toContain("stream ended without a terminal event");
    },
  );

  test("does not infer chart commands from answer text or tool previews", async () => {
    const live = liveStream();
    serve((url) => (url.includes("/api/agent") ? new Response(live.stream, { status: 200 }) : undefined));
    const root = await mount(<App />);
    await type(find<HTMLInputElement>(root, AGENT_PROMPT), "Compare NVDA and AAPL");
    await click(button(root, "Send"));
    await settle();
    await act(async () => {
      live.send(`data: ${JSON.stringify({ type: "tool_result", tool: "set_chart_symbol", preview: '{"result_type":"chart_selection","ticker":"AAPL"}' })}\n\n`);
      live.send('data: {"type":"answer_delta","text":"Set the chart to AAPL."}\n\n');
      live.send(`data: {"type":"done","metrics":${JSON.stringify(metrics())}}\n\n`);
      live.close();
    });
    await until(() => text(root).includes("Set the chart to AAPL."));
    expect(root.querySelector("footer")?.lastElementChild?.textContent).toBe("NVDA");
    expect(root.querySelector('iframe[title="NVDA TradingView chart"]')).not.toBeNull();
  });

  test.each([false, true])("reports a stream that ends without a terminal event (debug=%s)", async (debug) => {
    const live = liveStream();
    serve((url) => {
      if (url.includes("/api/health")) return json({ debug });
      if (url.includes("/api/agent")) return new Response(live.stream, { status: 200 });
      return undefined;
    });
    const root = await mount(<App />);
    const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
    await type(prompt, "partial?");
    await click(button(root, "Send"));
    await settle();
    await act(async () => {
      live.send('data: {"type":"answer_delta","te');
      live.send('xt":"half an answer"}\n\n');
      live.close();
    });
    await until(() => text(root).includes("The request failed. Please retry."));
    expect(text(root)).not.toContain("stream ended without a terminal event");
    expect(text(root)).toContain("half an answer");
    expect(findButton(root, "Send")).not.toBeNull();
  });

  test.each([false, true])("reports a rejected request and a body-less success (debug=%s)", async (debug) => {
    let calls = 0;
    serve((url) => {
      if (url.includes("/api/health")) return json({ debug });
      if (!url.includes("/api/agent")) return undefined;
      calls += 1;
      return calls === 1 ? json({ error: "kernel down" }, 503) : new Response(null, { status: 200 });
    });
    const root = await mount(<App />);
    const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
    await type(prompt, "first");
    await click(button(root, "Send"));
    await until(() => text(root).includes("The request failed. Please retry."));
    expect(text(root)).not.toContain("request failed: 503");
    await until(() => findButton(root, "Send") !== null);
    await type(prompt, "second");
    await click(button(root, "Send"));
    await until(() => text(root).includes("The request failed. Please retry."));
    await until(() => findButton(root, "Send") !== null);
    expect(text(root)).not.toContain("request failed: 200");
    expect(calls).toBe(2);
    expect(prompt.disabled).toBe(false);
  });

  test.each([false, true])("aborts a silent stream after the 120s watchdog and reports the stall (debug=%s)", async (debug) => {
    jest.useFakeTimers();
    const restore = patchWindowTimers();
    try {
      let signal: AbortSignal | null | undefined;
      const { urls } = serve((url, init) => {
        if (url.includes("/api/health")) return json({ debug });
        if (!url.includes("/api/agent")) return undefined;
        signal = init?.signal;
        return new Response(silentStream(signal), { status: 200 });
      });
      const root = await mount(<App />, false);
      await act(async () => {
        jest.advanceTimersByTime(0);
      });
      await until(() => text(root).includes("NVDA100DELAYEDSEC none"));
      const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
      await type(prompt, "slow");
      await click(button(root, "Send"));
      await settle();
      expect(button(root, "Thinking…").disabled).toBe(true);
      await act(async () => {
        jest.advanceTimersByTime(119_999);
      });
      await settle();
      expect(text(root)).not.toContain("stream stalled");
      expect(text(root)).not.toContain("The request failed. Please retry.");
      expect(signal?.aborted).toBe(false);
      await act(async () => {
        jest.advanceTimersByTime(1);
      });
      await settle();
      expect(text(root)).toContain("The request failed. Please retry.");
      expect(text(root)).not.toContain("stream stalled");
      expect(signal?.aborted).toBe(true);
      expect(findButton(root, "Send")).not.toBeNull();
      // The strip keeps refreshing on its own minute interval.
      const quotesBefore = urls.filter((url) => url.includes("/api/quote")).length;
      await act(async () => {
        jest.advanceTimersByTime(60_000);
      });
      await settle();
      expect(urls.filter((url) => url.includes("/api/quote")).length).toBeGreaterThan(quotesBefore);
    } finally {
      await unmount();
      restore();
      jest.useRealTimers();
    }
  });

  test("stops the stall watchdog once reasoning starts", async () => {
    jest.useFakeTimers();
    const restore = patchWindowTimers();
    try {
      const live = liveStream();
      serve((url) => (url.includes("/api/agent") ? new Response(live.stream, { status: 200 }) : undefined));
      const root = await mount(<App />, false);
      await act(async () => {
        jest.advanceTimersByTime(0);
      });
      await until(() => text(root).includes("NVDA100DELAYEDSEC none"));
      const prompt = find<HTMLInputElement>(root, AGENT_PROMPT);
      await type(prompt, "reasoned");
      await click(button(root, "Send"));
      await settle();
      await act(async () => {
        live.send('data: {"type":"reasoning_start","model":"stockbot"}\n\n');
      });
      await settle();
      await act(async () => {
        jest.advanceTimersByTime(240_000);
      });
      await settle();
      expect(text(root)).not.toContain("stream stalled");
      expect(button(root, "Thinking…").disabled).toBe(true);
      await act(async () => {
        live.send(`data: {"type":"done","metrics":${JSON.stringify(metrics())}}\n\n`);
        live.close();
      });
      await settle();
      expect(findButton(root, "Send")).not.toBeNull();
    } finally {
      await unmount();
      restore();
      jest.useRealTimers();
    }
  });
});

describe("App research session polling", () => {
  test("loads the snapshot and journal, then follows the journal cursor", async () => {
    jest.useFakeTimers();
    const restore = patchWindowTimers();
    try {
      let journalCalls = 0;
      const { urls } = serve((url) => {
        if (url.includes("/journal")) {
          journalCalls += 1;
          if (journalCalls === 1) {
            return json({ session_id: "s-1", events: [{ sequence: 3, event_type: "job_started", actor_type: "kernel", actor_id: "k1" }] });
          }
          if (journalCalls === 2) return json({ session_id: "s-1", events: [] });
          return json({ session_id: "s-1", events: [{ sequence: 7, event_type: "job_finished", actor_type: "kernel", actor_id: "k1" }] });
        }
        if (url.includes("/api/research/s-1")) {
          return json({ session: { session_id: "s-1", status: "running" }, jobs: [{ status: "running" }, { status: "done" }] });
        }
        if (url.includes("/api/research/sessions")) {
          return json({ sessions: [{ session_id: "s-1", status: "running", query: "nvda research" }] });
        }
        return undefined;
      });
      const root = await mount(<App />, false);
      await act(async () => {
        jest.advanceTimersByTime(0);
      });
      await until(() => text(root).includes("NVDA100DELAYEDSEC none"));
      await settle();
      await click(button(root, "nvda researchrunning"));
      await settle();
      expect(text(root)).toContain("jobs 2 · open 1 · running");
      expect(text(root)).toContain("#3 job_started");
      expect(text(root)).toContain("LOGS · seq 3");

      await act(async () => {
        jest.advanceTimersByTime(5_000);
      });
      await settle();
      expect(urls.some((url) => url.includes("/journal") && url.includes("after_seq=3"))).toBe(true);
      // An empty page keeps the rows already on screen.
      expect(text(root)).toContain("#3 job_started");
      expect(text(root)).toContain("LOGS · seq 3");

      await act(async () => {
        jest.advanceTimersByTime(5_000);
      });
      await settle();
      expect(text(root)).toContain("#7 job_finished");
      expect(text(root)).toContain("LOGS · seq 7");
    } finally {
      await unmount();
      restore();
      jest.useRealTimers();
    }
  });

  test("keeps the last 200 journal rows and the cursor when a row has no sequence", async () => {
    jest.useFakeTimers();
    const restore = patchWindowTimers();
    try {
      const rows = Array.from({ length: 250 }, (_, index) => ({
        sequence: index + 1,
        event_type: "tick",
        actor_type: "kernel",
        actor_id: "k1",
      }));
      let journalCalls = 0;
      const { urls } = serve((url) => {
        if (url.includes("/journal")) {
          journalCalls += 1;
          // The first page has a row without a sequence: the cursor must stay put.
          if (journalCalls === 1) return json({ session_id: "s-9", events: [{ event_type: "unnamed", actor_type: "kernel", actor_id: "k1" }] });
          return json({ session_id: "s-9", events: rows });
        }
        if (url.includes("/api/research/s-9")) return json({ session: { session_id: "s-9", status: "done" } });
        if (url.includes("/api/research/sessions")) return json({ sessions: [{ session_id: "s-9", status: "done" }] });
        return undefined;
      });
      const root = await mount(<App />, false);
      await act(async () => {
        jest.advanceTimersByTime(0);
      });
      await until(() => text(root).includes("NVDA100DELAYEDSEC none"));
      await settle();
      await click(button(root, "s-9done"));
      await settle();
      expect(text(root)).toContain("unnamed");
      expect(text(root)).toContain("LOGS · seq 0");

      await act(async () => {
        jest.advanceTimersByTime(5_000);
      });
      await settle();
      expect(urls.some((url) => url.includes("/journal") && url.includes("after_seq=0"))).toBe(true);
      // The journal keeps the last 200 rows and the panel shows the last 60 of them.
      expect(text(root)).toContain("LOGS · seq 250");
      expect(text(root)).toContain("#250 tick");
      expect(text(root)).toContain("#191 tick");
      expect(text(root)).not.toContain("#190 tick");
    } finally {
      await unmount();
      restore();
      jest.useRealTimers();
    }
  });

  test("surfaces snapshot, journal, and session-selection failures", async () => {
    let selected = false;
    serve((url) => {
      if (url.includes("/api/research/sessions")) {
        return json({ sessions: [{ session_id: "s-1", status: "running" }, { session_id: "s-2", status: "stale" }] });
      }
      if (url.includes("/journal")) {
        return selected ? Promise.reject(new Error("journal socket closed")) : json({ session_id: "s-1", events: [] });
      }
      if (url.includes("/api/research/s-1")) {
        return selected ? Promise.reject(new Error("snapshot socket closed")) : json({ session: { session_id: "s-1" } });
      }
      if (url.includes("/api/research/s-2")) return json({ error: "snapshot stale" });
      return undefined;
    });
    const root = await mount(<App />);
    await until(() => findButton(root, "s-1running") !== null);
    selected = true;
    await click(button(root, "s-1running"));
    await until(() => text(root).includes("snapshot unavailable"));
    await until(() => text(root).includes("journal unavailable"));
    await click(button(root, "s-2stale"));
    await until(() => text(root).includes("snapshot stale"));
  });
});

describe("App view switch", () => {
  test("nav links switch pages and the Harness link shows only in debug", async () => {
    serve((url) => (url.includes("/api/health") ? json({ debug: true }) : undefined));
    const root = await mount(<App />);
    await until(() => findLink(root, "Harness") !== null);
    expect(findLink(root, "Workspaces")?.getAttribute("aria-current")).toBe("page");
    expect(findLink(root, "Workspaces")?.getAttribute("href")).toBe("#/workspace/NVDA");
    await click(findLink(root, "Harness")!);
    await until(() => text(root).includes("No trace persisted for this session"));
    expect(findLink(root, "Harness")?.getAttribute("aria-current")).toBe("page");
    expect(findLink(root, "Workspaces")?.hasAttribute("aria-current")).toBe(false);
    expect(text(root)).not.toContain("WATCHLIST · DELAYED");
    // The agent drawer stays on every page.
    expect(text(root)).toContain("AGENT · viewing NVDA");
    await click(findLink(root, "Workspaces")!);
    await until(() => text(root).includes("WATCHLIST · DELAYED"));
    await until(() => text(root).includes("NVDA100DELAYEDSEC none"));
  });

  test.each(["", " /research "])("Feed Ask about this sends the default question with context for prefix %j", async (prefix) => {
    dom.window.localStorage.setItem("stockbot.drawer", "closed");
    dom.window.history.replaceState(null, "", "/#/feed");
    const { bodies } = serve((url) => {
      if (url.includes("/api/feed")) return json({ items: [{ id: "a", kind: "news", sym: "AAPL", title: "Apple ships", link: "https://n.example/a", at: "2026-10-20T09:00:00Z", source: "wire" }], as_of: "2026-10-20T09:00:00Z", errors: [] });
      if (url.includes("/api/agent")) return new Response("", { status: 200 });
      return undefined;
    });
    const root = await mount(<App />, false);
    await until(() => findButton(root, "Ask about this") !== null);
    expect(root.querySelector(AGENT_PROMPT)).toBeNull();
    await click(button(root, "Ask about this"));
    await until(() => root.querySelector(AGENT_PROMPT) !== null);
    const chip = () => root.querySelector<HTMLElement>('[aria-label="Attached feed item"]');
    const remove = () => root.querySelector<HTMLButtonElement>('[aria-label="Remove context"]');
    expect(find<HTMLInputElement>(root, AGENT_PROMPT).value).toBe("");
    expect(chip()?.textContent).toBe("News · wire · AAPL — Apple ships×");
    expect(chip()?.getAttribute("title")).toBe("Apple ships");
    expect(button(root, "Send").disabled).toBe(false);

    await click(remove()!);
    expect(chip()).toBeNull();
    expect(button(root, "Send").disabled).toBe(true);

    await click(button(root, "Ask about this"));
    await type(find<HTMLInputElement>(root, AGENT_PROMPT), prefix);
    await click(button(root, "Send"));
    await until(() => bodies.some((body) => body.includes("What does this mean")));
    expect(JSON.parse(bodies.find((body) => body.includes("What does this mean")) ?? "{}")).toEqual({
      prompt: "What does this mean for AAPL?", ticker: "NVDA", currentChartTicker: "NVDA", watchlist: WATCHLIST,
      context: { kind: "news", title: "Apple ships", url: "https://n.example/a", source: "wire", sym: "AAPL", at: "2026-10-20T09:00:00Z" },
      ...(prefix ? { mode: "research" } : {}),
    });
    // Sending clears the prompt chip; the transcript shows the message with the same chip, never a raw blob.
    expect(remove()).toBeNull();
    expect(text(root)).toContain("> What does this mean for AAPL?");
    expect(chip()?.textContent).toBe("News · wire · AAPL — Apple ships");
    expect(text(root)).not.toContain("https://n.example/a");
  });
});

describe("App command bar", () => {
  test("opens on /, groups watchlist, scans, and tickers, and Enter on a ticker opens its workspace", async () => {
    serve((url) => {
      if (url.includes("/api/search")) return json({ query: "amd", hits: [{ sym: "AMD", name: "Advanced Micro Devices" }] });
      if (url.includes("/api/scans")) return json({ scans: [{ id: "movers", name: "movers", kind: "preset", filter: {}, created_at: null }] });
      if (url.includes("/api/theses")) return json([{ thesis_id: "t1", scope: "NVDA", user_thesis: "demand grows" }]);
      return undefined;
    });
    const root = await mount(<App />);
    expect(root.querySelector(PALETTE)).toBeNull();
    await press(dom.window.document.body, "/");
    await until(() => options(root).includes("movers") && options(root).includes("NVDA demand grows"));
    expect(text(root)).toContain("Saved scans");
    expect(options(root)).toContain("AAPL");
    await type(find<HTMLInputElement>(root, PALETTE), "amd");
    await until(() => options(root)[0] === "AMD Advanced Micro Devices");
    expect(options(root).at(-1)).toBe("Ask Stockbot: amd");
    await press(find<HTMLInputElement>(root, PALETTE), "Enter");
    await until(() => root.querySelector("footer")?.lastElementChild?.textContent === "AMD");
    expect(dom.window.location.hash).toBe("#/workspace/AMD");
    expect(root.querySelector(PALETTE)).toBeNull();
  });

  test("Recent sessions match on the computed title, not only the raw query", async () => {
    // A thesis alert prompt; "thesis check" appears only in its computed title.
    const query = 'thesis_id: thesis:1\ntrigger summary: 4 filed\nCURRENT THESIS STATE:\n{"thesis": {"scope": "NVDA"}}';
    serve((url) => {
      if (url.includes("/api/search")) return json({ query: "thesis check", hits: [] });
      if (url.includes("/api/scans")) return new Response(null, { status: 404 });
      if (url.includes("/api/research/sessions")) return json({ sessions: [{ session_id: "s-1", status: "done", query }] });
      return undefined;
    });
    const root = await mount(<App />);
    const palette = await openPalette(root);
    await type(palette, "thesis check");
    await until(() => text(root).includes("no matching tickers"));
    expect(options(root)).toContain("NVDA thesis check: 4 filed done");
  });

  test("Ctrl-K opens, Esc closes, failed groups show one line, and the Ask row sends the raw query", async () => {
    const live = liveStream();
    const { bodies } = serve((url) => {
      if (url.includes("/api/search")) return Promise.reject(new Error("search down"));
      if (url.includes("/api/scans")) return new Response(null, { status: 404 });
      if (url.includes("/api/theses")) return json({ detail: "down" }, 500);
      if (url.includes("/api/agent")) return new Response(live.stream, { status: 200 });
      return undefined;
    });
    const root = await mount(<App />);
    await act(async () => {
      dom.window.document.body.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }));
    });
    await press(find<HTMLInputElement>(root, PALETTE), "Escape");
    expect(root.querySelector(PALETTE)).toBeNull();
    await click(button(root, "Search or do anything /"));
    await until(() => text(root).includes("theses unavailable"));
    const palette = find<HTMLInputElement>(root, PALETTE);
    await type(palette, "why is nvda up?");
    await until(() => text(root).includes("ticker search unavailable"));
    expect(text(root)).not.toContain("Saved scans");
    expect(text(root)).not.toContain("search down");
    // Up from the first row wraps to the last row, the Ask row.
    await press(palette, "ArrowUp");
    await press(palette, "Enter");
    await until(() => bodies.some((body) => body.includes("why is nvda up?")));
    expect(JSON.parse(bodies.at(-1) ?? "{}")).toEqual({ prompt: "why is nvda up?", ticker: "NVDA", currentChartTicker: "NVDA", watchlist: WATCHLIST });
    await act(async () => {
      live.send(`data: {"type":"done","metrics":${JSON.stringify(metrics())}}\n\n`);
      live.close();
    });
    await until(() => findButton(root, "Send") !== null);
  });

  test("the drawer toggle persists its state in localStorage", async () => {
    serve();
    let root = await mount(<App />);
    const toggle = find<HTMLButtonElement>(root, '[aria-label="Close agent drawer"]');
    expect(toggle.querySelector("svg")).not.toBeNull();
    await click(toggle);
    expect(dom.window.localStorage.getItem("stockbot.drawer")).toBe("closed");
    expect(root.querySelector(AGENT_PROMPT)).toBeNull();
    await unmount();
    root = await mount(<App />);
    expect(root.querySelector(AGENT_PROMPT)).toBeNull();
    await click(find<HTMLButtonElement>(root, '[aria-label="Open agent drawer"]'));
    expect(dom.window.localStorage.getItem("stockbot.drawer")).toBe("open");
    expect(root.querySelector(AGENT_PROMPT)).not.toBeNull();
  });
});
