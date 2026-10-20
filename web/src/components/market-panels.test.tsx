// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act, useState } from "react";
import type { ReactNode } from "react";
import type { Root } from "react-dom/client";
import {
  AIPanel,
  FinancialsPanel,
  JournalPanel,
  NewsPanel,
  SessionPanel,
  WatchlistPanel,
  watchlistFor,
} from "./market-panels";
import type { JournalPage } from "../lib/research-client";
import { WatchlistChatPanel } from "./watchlist-chat-panel";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";

// react-dom reads canUseDOM once at module load, so it must load after the jsdom globals
// exist. A static import runs before this module body, so this one import stays dynamic.
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

async function rerender(element: ReactNode): Promise<void> {
  await act(async () => mounted?.render(element));
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

// React tracks the last value it committed on the input node, so typing must set
// the value through the prototype setter to reach a controlled onChange handler.
async function type(target: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value")?.set;
    setter?.call(target, value);
    target.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  });
}

async function unmount(): Promise<void> {
  await act(async () => mounted?.unmount());
  mounted = null;
}

function panel(root: HTMLElement): HTMLElement {
  const found = root.firstElementChild;
  if (!(found instanceof dom.window.HTMLElement)) throw new Error("missing panel element");
  return found;
}

function expectScrollable(root: HTMLElement): void {
  const computed = dom.window.getComputedStyle(panel(root));
  expect(computed.overflowY).toBe("auto");
  expect(computed.height).toBe("100%");
}

describe("WatchlistChatPanel", () => {
  test("applies add/remove once, preserves the chart, and reports the actual list", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL) => String(input).startsWith("/api/watchlist")
      ? Response.json({ watchlist: [] }) : Response.json({ price: null })) as unknown as typeof fetch;
    let actual: string[] = [];
    const onChange = (symbols: string[]) => { actual = symbols; };
    const onSelect = () => { throw new Error("A watchlist update must not select the chart."); };
    const add: AgentEvent = { type: "watchlist_update", action: "add", ticker: "AMD" };
    const remove: AgentEvent = { type: "watchlist_update", action: "remove", ticker: "AAPL" };
    const root = await mount(<WatchlistChatPanel selected="NVDA" onSelect={onSelect} events={[add, remove]} onChange={onChange} />);
    await until(() => actual.includes("AMD") && !actual.includes("AAPL"));
    expect(root.querySelectorAll("b").length).toBe(9);
    expect(root.textContent).toContain("A dash means unavailable");
    const removeAmd: AgentEvent = { type: "watchlist_update", action: "remove", ticker: "AMD" };
    await rerender(<WatchlistChatPanel selected="NVDA" onSelect={onSelect} events={[add, remove, removeAmd]} onChange={onChange} />);
    await until(() => !actual.includes("AMD"));
    await rerender(<WatchlistChatPanel selected="NVDA" onSelect={onSelect} events={[add]} onChange={onChange} />);
    expect(actual).not.toContain("AMD");
    await rerender(<WatchlistChatPanel selected="ORCL" onSelect={onSelect} events={[]} onChange={onChange} />);
    await until(() => actual[0] === "ORCL");
    expect(actual).not.toContain("AAPL");
  });
});

describe("WatchlistPanel", () => {
  test("keeps default order, adds only an extra selection, and renders quote and source outcomes", async () => {
    const defaults = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "AMZN", "META", "GOOG"];
    expect(watchlistFor("")).toEqual(defaults);
    expect(watchlistFor("AAPL")).toEqual(defaults);
    expect(watchlistFor("BRK/B")).toEqual(["BRK/B", ...defaults]);
    const requests: string[] = [];
    const signals: AbortSignal[] = [];
    const sources = [
      { symbol: "BRK/B", source: "sec_filings", tool: "sec-filings", status: "ok", as_of: "2026-10-13T22:15:00Z", ref: { record_id: "sec-record" } },
      { symbol: "BRK/B", source: "finra_short_interest", tool: "short-interest", status: "no_records" },
      { symbol: "BRK/B", source: "finra_short_sale_volume", tool: "short-volume", status: "error", error: "volume failed" },
      { symbol: "BRK/B", source: "finra_threshold_list", tool: "threshold", status: "timeout", error: "deadline" },
      { symbol: "AAPL", source: "sec_filings", tool: "sec-filings", status: "ok" },
    ];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "https://example.com");
      requests.push(url.pathname + url.search);
      if (init?.signal) signals.push(init.signal);
      if (url.pathname === "/api/watchlist") return Response.json({ watchlist: sources });
      const sym = url.searchParams.get("sym");
      if (sym === "MSFT") throw new Error("quote failed");
      return Response.json({ sym, price: sym === "AAPL" ? "123" : 1234.567 });
    }) as unknown as typeof fetch;
    const selected: string[] = [];
    const root = await mount(<WatchlistPanel selected="BRK/B" onSelect={(sym) => selected.push(sym)} />);
    await until(() => root.querySelector('[aria-label="BRK/B sources"]')?.textContent?.includes("THR timeout") ?? false);
    const buttons = [...root.querySelectorAll("button")];
    expect(buttons.map((button) => button.querySelector("b")?.textContent)).toEqual(["BRK/B", ...defaults]);
    expect(buttons[0]?.textContent).toContain((1234.567).toLocaleString(undefined, { maximumFractionDigits: 2 }));
    expect(buttons[2]?.textContent).toContain("—");
    expect(buttons[3]?.textContent).toContain("—");
    const sourceBox = root.querySelector('[aria-label="BRK/B sources"]');
    expect(sourceBox?.textContent).toBe("SEC 2026-10-13 SI none VOL error THR timeout ");
    expect(sourceBox?.querySelector("span")?.title).toBe("sec-filings · ok · 2026-10-13T22:15:00Z · sec-record");
    expect(sourceBox?.querySelectorAll("span")[2]?.title).toContain("volume failed");
    expect(root.querySelector('[aria-label="AAPL sources"]')?.textContent).toBe("SEC ok ");
    await click(buttons[2]!);
    expect(selected).toEqual(["AAPL"]);
    expect(requests.filter((url) => url.startsWith("/api/quote?"))).toHaveLength(10);
    expect(requests).toContain(`/api/watchlist?symbols=${encodeURIComponent(["BRK/B", ...defaults].join(","))}`);
    expectScrollable(root);
    await unmount();
    expect(signals.length).toBeGreaterThan(0);
    expect(signals.every((signal) => signal.aborted)).toBe(true);
  });

  test("reports a failed source fan-out without hiding the watchlist", async () => {
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      if (String(input).startsWith("/api/watchlist")) throw "source service offline";
      return Response.json({ price: null });
    }) as unknown as typeof fetch;
    const root = await mount(<WatchlistPanel selected="NVDA" onSelect={() => { }} />);
    await until(() => root.textContent?.includes("sources unavailable: source service offline") ?? false);
    expect(root.querySelectorAll("button")).toHaveLength(9);
    expect(root.querySelector("button")?.textContent).toContain("—");
  });

  test("aborts replaced requests and stops an obsolete quote from overwriting the current symbol", async () => {
    const oldQuote = Promise.withResolvers<Response>();
    const oldSources = Promise.withResolvers<Response>();
    const oldSignals: AbortSignal[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "https://example.com");
      const obsolete = url.searchParams.get("sym") === "OLD" || url.searchParams.get("symbols")?.startsWith("OLD,");
      if (obsolete) {
        if (init?.signal) oldSignals.push(init.signal);
        return url.pathname === "/api/watchlist" ? oldSources.promise : oldQuote.promise;
      }
      return url.pathname === "/api/watchlist" ? Response.json({ watchlist: [] }) : Response.json({ price: 42 });
    }) as unknown as typeof fetch;
    const root = await mount(<WatchlistPanel selected="OLD" onSelect={() => { }} />);
    // Hold both OLD requests open: the quote fan-out and the deferred source fan-out.
    await until(() => oldSignals.length === 2);
    await rerender(<WatchlistPanel selected="NEW" onSelect={() => { }} />);
    await until(() => root.querySelector("button")?.textContent?.includes("42") ?? false);
    expect(oldSignals).toHaveLength(2);
    expect(oldSignals.every((signal) => signal.aborted)).toBe(true);
    await act(async () => {
      oldQuote.resolve(Response.json({ price: 999 }));
      oldSources.reject(new Error("obsolete sources"));
    });
    expect(root.querySelector("button")?.textContent).toContain("NEW42");
    expect(root.textContent).not.toContain("999");
    expect(root.textContent).not.toContain("obsolete sources");
  });
});

describe("FinancialsPanel and NewsPanel", () => {
  test("shows the official filing date, keeps the acceptance time in the title, and renders links only when present", async () => {
    const items = Array.from({ length: 4 }, (_, index) => ({
      title: `Filing ${index}`,
      link: index === 0 ? "https://sec.example/filing" : "",
      filed: "2026-10-12",
      updated: "2026-10-13T01:23:45-04:00",
      summary: "details",
    }));
    const asked: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      asked.push(String(input));
      return Response.json({ sym: "NVDA", cik: 320193, items });
    }) as unknown as typeof fetch;
    const root = await mount(<FinancialsPanel sym="NVDA" />);
    await until(() => (root.textContent ?? "").includes("Filing 2"));
    expect(asked).toEqual(["/api/filings?sym=NVDA"]);
    expect(root.textContent).toContain("FILINGS · NVDA");
    expect(root.textContent).toContain("filed 2026-10-12");
    expect(root.textContent).not.toContain("filed 2026-10-13");
    expect(root.querySelector('[title="updated 2026-10-13T01:23:45-04:00"]')).not.toBeNull();
    const link = root.querySelector("a");
    expect(link?.textContent).toBe("Filing 0");
    expect(link?.href).toBe("https://sec.example/filing");
    expect(link?.target).toBe("_blank");
    expect(link?.rel).toBe("noreferrer");
    expect(root.querySelectorAll("a")).toHaveLength(1);
    expect(root.textContent).toContain("Filing 2");
    expect(root.textContent).not.toContain("Filing 3");
    expectScrollable(root);
    expect(panel(root).style.minHeight).toBe("0");
  });

  test("truncates the published timestamp, tolerates a missing date, and keeps at most eight headlines", async () => {
    const items = Array.from({ length: 9 }, (_, index) => ({
      title: `Headline ${index}`,
      link: index === 0 ? "https://news.example/story" : "",
      published: index === 0 ? "2026-10-14T12:34:56Z" : null,
      source: "Example Wire",
    }));
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      expect(String(input)).toBe("/api/news?sym=AAPL");
      return Response.json({ sym: "AAPL", items });
    }) as unknown as typeof fetch;
    const root = await mount(<NewsPanel sym="AAPL" />);
    await until(() => (root.textContent ?? "").includes("Headline 7"));
    expect(root.textContent).toContain("NEWS · AAPL");
    expect(root.textContent).toContain("2026-10-14T12:34 · Example Wire");
    expect(root.textContent).not.toContain("12:34:56");
    expect(root.textContent).not.toContain("null · Example Wire");
    expect(root.textContent).not.toContain("Headline 8");
    const link = root.querySelector("a");
    expect(link?.href).toBe("https://news.example/story");
    expect(link?.target).toBe("_blank");
    expect(link?.rel).toBe("noreferrer");
    expect(root.querySelectorAll("a")).toHaveLength(1);
    expectScrollable(root);
    expect(panel(root).style.minHeight).toBe("0");
  });

  test("distinguishes empty results, a server warning, and a rejected request", async () => {
    globalThis.fetch = (async () => Response.json({ items: [] })) as unknown as typeof fetch;
    const root = await mount(<FinancialsPanel sym="EMPTY" />);
    await until(() => (root.textContent ?? "").includes("no SEC filings found"));
    await rerender(<NewsPanel sym="EMPTY" />);
    await until(() => (root.textContent ?? "").includes("loading headlines…"));
    globalThis.fetch = (async () => Response.json({ items: [], error: "provider has no access" })) as unknown as typeof fetch;
    await rerender(<FinancialsPanel sym="WARN" />);
    await until(() => (root.textContent ?? "").includes("provider has no access"));
    expect(root.textContent).not.toContain("no SEC filings found");
    await rerender(<NewsPanel sym="WARN" />);
    await until(() => (root.textContent ?? "").includes("provider has no access"));
    expect(root.textContent).not.toContain("loading headlines…");
    globalThis.fetch = (() => Promise.reject(new Error("filing outage"))) as unknown as typeof fetch;
    await rerender(<FinancialsPanel sym="FAIL" />);
    await until(() => (root.textContent ?? "").includes("filings unavailable: filing outage"));
    globalThis.fetch = (() => Promise.reject("upstream unavailable")) as unknown as typeof fetch;
    await rerender(<FinancialsPanel sym="FAIL-STRING" />);
    await until(() => (root.textContent ?? "").includes("filings unavailable: upstream unavailable"));
    await rerender(<NewsPanel sym="FAIL" />);
    await until(() => (root.textContent ?? "").includes("news unavailable"));
    expect(root.textContent).not.toContain("loading headlines…");
  });

  const renderers: [string, (sym: string) => ReactNode][] = [
    ["filings", (sym) => <FinancialsPanel sym={sym} />],
    ["news", (sym) => <NewsPanel sym={sym} />],
  ];
  for (const [name, render] of renderers) {
    test(`${name} follows the selected symbol and ignores a stale success and a stale failure`, async () => {
      const first = Promise.withResolvers<Response>();
      const second = Promise.withResolvers<Response>();
      const requests: { url: string; signal: AbortSignal | null | undefined }[] = [];
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        requests.push({ url: String(input), signal: init?.signal });
        const sym = new URL(String(input), "https://example.com").searchParams.get("sym");
        if (sym === "FIRST") return first.promise;
        if (sym === "SECOND") return second.promise;
        return Response.json({ items: [{ title: "Current result", link: "", filed: "2026-10-14", updated: "acceptance", published: "2026-10-14", source: "Wire" }] });
      }) as unknown as typeof fetch;
      const root = await mount(render("FIRST"));
      await rerender(render("SECOND"));
      await rerender(render("BRK/B"));
      await until(() => (root.textContent ?? "").includes("Current result"));
      expect(requests.map((request) => request.url)).toEqual([`/api/${name}?sym=FIRST`, `/api/${name}?sym=SECOND`, `/api/${name}?sym=BRK%2FB`]);
      expect(requests[0]?.signal?.aborted).toBe(true);
      expect(requests[1]?.signal?.aborted).toBe(true);
      await act(async () => {
        first.resolve(Response.json({ items: [{ title: "Obsolete result" }], error: "obsolete warning" }));
        second.reject(new Error("obsolete rejection"));
      });
      expect(root.textContent).toContain("Current result");
      expect(root.textContent).not.toContain("Obsolete");
      expect(root.textContent).not.toContain("obsolete");
      await unmount();
      expect(requests[2]?.signal?.aborted).toBe(true);
    });
  }
});

describe("AIPanel", () => {
  test("edits the prompt, sends the current text, and blocks input while busy", async () => {
    const sent: string[] = [];
    function Prompt({ busy = false, debug = false }: { busy?: boolean; debug?: boolean }) {
      const [input, setInput] = useState("   ");
      return <AIPanel events={[]} input={input} setInput={setInput} onSend={() => sent.push(input)} sym="NVDA" busy={busy} debug={debug} />;
    }
    const root = await mount(<Prompt />);
    const input = root.querySelector("input")!;
    const send = root.querySelector('button[type="submit"]') as HTMLButtonElement;
    expect(input.getAttribute("aria-label")).toBe("Ask the research agent");
    expect(input.placeholder).toBe("Ask about this symbol…");
    expect(send.disabled).toBe(true);
    await type(input, "Explain the NVDA filings");
    expect(input.value).toBe("Explain the NVDA filings");
    expect(send.disabled).toBe(false);
    await click(send);
    expect(sent).toEqual(["Explain the NVDA filings"]);
    await rerender(<Prompt busy debug />);
    expect(root.textContent).toContain("AGENT · viewing NVDA · DEBUG");
    expect(input.value).toBe("Explain the NVDA filings");
    expect(input.disabled).toBe(true);
    expect(input.placeholder).toBe("Research in progress…");
    expect(send.disabled).toBe(true);
    expect(send.textContent).toBe("Thinking…");
    await click(send);
    expect(sent).toEqual(["Explain the NVDA filings"]);
    expect(panel(root).style.minHeight).toBe("0");
    expect(panel(root).style.overflow).toBe("hidden");
  });

  test("streams answer and reasoning events into the console", async () => {
    const root = await mount(<AIPanel events={[{ type: "reasoning_start", model: "research" }, { type: "answer_delta", text: "The filing reports revenue growth." }]} input="Question" setInput={() => { }} onSend={() => { }} sym="AAPL" busy={false} debug={false} />);
    expect(root.textContent).toContain("AGENT · viewing AAPL");
    expect(root.textContent).toContain("The filing reports revenue growth.");
    expect(root.textContent).not.toContain("DEBUG");
  });
});

describe("SessionPanel and JournalPanel", () => {
  test("selects and toggles sessions, truncates their text, and counts only queued or running jobs as open", async () => {
    const chosen: (string | null)[] = [];
    const sessions = [{ session_id: "session-one", status: "active", query: "Q".repeat(70) }, { session_id: "session-two" }, {}];
    const snapshot = { session: { status: "researching" }, jobs: [{ status: "running" }, { status: "queued" }, { status: "done" }, { status: "failed" }] };
    const root = await mount(<SessionPanel sessions={sessions} sessionId="session-one" onSelect={(id) => chosen.push(id)} snapshot={snapshot} />);
    const buttons = [...root.querySelectorAll("button")];
    expect(buttons[0]?.textContent).toContain("session-");
    expect(buttons[0]?.textContent).toContain("Q".repeat(60));
    expect(buttons[0]?.textContent).not.toContain("Q".repeat(61));
    expect(buttons[1]?.textContent).toContain("?");
    await click(buttons[0]!);
    await click(buttons[1]!);
    await click(buttons[2]!);
    expect(chosen).toEqual([null, "session-two", null]);
    expect(root.textContent).toContain("jobs 4 · open 2 · researching");
    await rerender(<SessionPanel sessions={[]} sessionId={null} onSelect={() => { }} snapshot={{}} />);
    expect(root.textContent).toContain("no research sessions yet");
    expect(root.textContent).toContain("jobs 0 · open 0 · ?");
    await rerender(<SessionPanel sessions={Array.from({ length: 10 }, (_, index) => ({ session_id: `id-${index}` }))} sessionId={null} onSelect={() => { }} snapshot={{ error: "snapshot unavailable" }} />);
    expect(root.querySelectorAll("button")).toHaveLength(8);
    expect(root.textContent).toContain("snapshot unavailable");
    expect(root.textContent).not.toContain("jobs 0 ·");
  });

  test("distinguishes an absent, an empty, and a failed journal and keeps the last sixty rows scrollable", async () => {
    const root = await mount(<JournalPanel journal={null} seq={0} />);
    expect(root.textContent).toContain("LOGS · seq 0");
    expect(root.textContent).toContain("select a session for logs");
    await rerender(<JournalPanel journal={{}} seq={4} />);
    expect(root.textContent).toContain("no journal rows yet");
    await rerender(<JournalPanel journal={{ error: "journal unavailable" }} seq={4} />);
    expect(root.textContent).toContain("journal unavailable");
    expect(root.textContent).not.toContain("no journal rows yet");
    const events: NonNullable<JournalPage["events"]> = Array.from({ length: 61 }, (_, index) => ({
      sequence: index + 1,
      event_id: index % 2 ? `event-${index}` : undefined,
      event_type: `row-${index + 1}`,
      actor_type: "worker",
      actor_id: `actor-${index + 1}`,
    }));
    await rerender(<JournalPanel journal={{ events }} seq={61} />);
    expect(root.textContent).toContain("LOGS · seq 61");
    expect(root.textContent).toContain("#2 row-2 worker:actor-2");
    expect(root.textContent).toContain("#61 row-61 worker:actor-61");
    expect(root.textContent).not.toContain("#1 row-1 worker:actor-1");
    const last = [...root.querySelectorAll("div")].find((entry) => entry.textContent === "#61 row-61 worker:actor-61");
    expect(last?.parentElement?.children).toHaveLength(60);
    expect(panel(root).style.overflow).toBe("hidden");
    expect(last?.parentElement?.style.overflowY).toBe("auto");
    expect(last?.parentElement?.style.maxHeight).toBe("220px");
  });
});
