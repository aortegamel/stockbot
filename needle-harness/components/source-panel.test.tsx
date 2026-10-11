import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import fc from "fast-check";
import type { Root } from "react-dom/client";
import { JSDOM } from "jsdom";
import { AgentConsole } from "./agent-console";
import { SourcePanel } from "./source-panel";
import type { AgentEvent } from "@/lib/agent/types";
import type { SourceDocument } from "@/lib/research-client";

const TEXT: SourceDocument = {
  id: "evidence:panel:text",
  title: "10-K",
  url: "https://www.sec.gov/filing",
  publisher: "SEC",
  publishedAt: "2026-01-01",
  capturedAt: "2026-01-02",
  contentHash: "abc",
  representation: "text",
  captureKind: "original",
  content: "Revenue   grew today.",
  quote: "Revenue grew",
};

const TEXT_URL = "/api/research/sources/evidence%3Apanel%3Atext";

const dom = new JSDOM(`<body><div id="root"></div></body>`, { url: "https://example.com/" });
const registry = globalThis as Record<string, unknown>;
const DOM_GLOBALS = ["IS_REACT_ACT_ENVIRONMENT", "window", "document", "navigator", "requestAnimationFrame", "cancelAnimationFrame"];
const savedGlobals = new Map<string, PropertyDescriptor | undefined>();
const realFetch = globalThis.fetch;
let mounted: Root | null = null;

beforeAll(() => {
  for (const key of DOM_GLOBALS) savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  registry.IS_REACT_ACT_ENVIRONMENT = true;
  registry.window = dom.window;
  registry.document = dom.window.document;
  registry.navigator = dom.window.navigator;
  registry.requestAnimationFrame = (callback: FrameRequestCallback) => setTimeout(() => callback(0), 0) as unknown as number;
  registry.cancelAnimationFrame = (frame: number) => clearTimeout(frame);
});

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
});

afterAll(() => {
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else Reflect.deleteProperty(globalThis, key);
  }
  dom.window.close();
});

// Each reply serves one source request in order. A gate holds a reply until the test releases it.
// The returned list records every requested URL.
function serve(...replies: { body: unknown; status?: number; gate?: Promise<void> }[]): string[] {
  const requests: string[] = [];
  globalThis.fetch = ((input: string | URL | Request) => {
    requests.push(String(input));
    const reply = replies.shift();
    if (!reply) return Promise.reject(new Error("unexpected source request"));
    const response = new Response(JSON.stringify(reply.body), { status: reply.status ?? 200, headers: { "Content-Type": "application/json" } });
    return (reply.gate ?? Promise.resolve()).then(() => response);
  }) as unknown as typeof fetch;
  return requests;
}

function panel(id: string, count = 1, events: string[] = []): ReactNode {
  return <SourcePanel reference={{ id }} position={0} count={count} onNavigate={(direction) => events.push(`navigate:${direction}`)} onClose={() => events.push("close")} />;
}

async function mount(element: ReactNode): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(element));
  return container;
}

// The wait ends on panel state, not on elapsed time. The tick cap only bounds a failure.
async function until(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 200 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 10)));
  }
  if (!ready()) throw new Error("the source panel did not reach the expected state");
}

function button(root: HTMLElement, label: string): HTMLButtonElement {
  const found = [...root.querySelectorAll("button")].find((entry) => entry.textContent === label);
  if (!found) throw new Error(`missing ${label}`);
  return found;
}

async function click(root: HTMLElement, label: string): Promise<void> {
  const target = button(root, label);
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

const resolved = (root: HTMLElement) => () => root.textContent?.includes("The quote matches the captured source.") ?? false;

async function open(source: SourceDocument): Promise<HTMLElement> {
  serve({ body: source });
  const root = await mount(panel(source.id));
  await until(() => /The quote matches|Ambiguous quote|does not resolve/u.test(root.textContent ?? ""));
  return root;
}

async function close(): Promise<void> {
  await act(async () => mounted?.unmount());
  mounted = null;
}

const shownQuote = (root: HTMLElement) => root.querySelector("blockquote p")?.textContent;
const originalHref = (root: HTMLElement) => [...root.querySelectorAll("a")].find((anchor) => anchor.textContent === "Open original")?.getAttribute("href");
const markTexts = (root: HTMLElement) => [...root.querySelectorAll('[aria-label="Captured source content"] mark')].map((mark) => mark.textContent);
const fragment = (url: string, quote: string) => `${url}#:~:text=${encodeURIComponent(quote)}`;

const EXA: SourceDocument = {
  ...TEXT,
  id: "evidence:panel:exa",
  title: "Nvidia results",
  url: "https://news.example/article",
  publisher: "Exa",
  contentHash: "sha256-of-raw-exa-body",
  representation: "html",
  captureKind: "provider_extraction",
  content: `<h1>Chip results</h1><p>Nvidia&#39;s <strong>sales <em>soared</em></strong> &amp; margins held.</p><figure><p>Getty Images</p></figure><p>Revenue doubled.</p>`,
  quote: `Nvidia&#39;s <strong>sales <em>soared</em></strong> &amp; margins held.</p><figure><p>Getty Images</p></figure><p>Revenue dou`,
};

describe("SourcePanel", () => {
  test("renders text captures with metadata and marks the exact quote", async () => {
    const requests = serve({ body: TEXT });
    const root = await mount(panel(TEXT.id));
    await until(resolved(root));
    expect(requests).toEqual([TEXT_URL]);
    expect(root.querySelector("h3")?.textContent).toBe("10-K");
    expect(root.textContent).toContain("Original document");
    for (const value of ["SEC", "2026-01-01", "2026-01-02", "abc", "1 of 1"]) expect(root.textContent).toContain(value);
    expect(button(root, "Previous source").disabled).toBe(true);
    expect(button(root, "Next source").disabled).toBe(true);
    expect(root.querySelector('a[href="https://www.sec.gov/filing"]')?.textContent).toBe("Open original");
    expect(root.querySelectorAll("mark").length).toBe(1);
    expect(root.querySelector("mark")?.textContent).toBe("Revenue   grew");
  });

  test("renders sanitized HTML tables with row context and marks the quote across cells", async () => {
    const html: SourceDocument = {
      ...TEXT,
      id: "evidence:panel:html",
      representation: "html",
      content: `<table><caption>Segment revenue</caption><thead><tr><th scope="col">Segment</th><th scope="col">2026</th></tr></thead><tbody><tr><td>Cloud</td><td>$12</td></tr></tbody></table><script>document.title = "ran"</script><img src="https://tracker.example/pixel.png" onerror="document.title = 'ran'"><a href="javascript:alert(1)">unsafe</a><a href="https://www.sec.gov/exhibit">exhibit</a>`,
      quote: "Cloud $12",
    };
    serve({ body: html });
    const root = await mount(panel(html.id));
    await until(resolved(root));
    const content = root.querySelector('[aria-label="Captured source content"]');
    if (!content) throw new Error("missing captured content");
    expect([...content.querySelectorAll("th")].map((cell) => [cell.textContent, cell.getAttribute("scope")])).toEqual([["Segment", "col"], ["2026", "col"]]);
    expect([...content.querySelectorAll("tbody td")].map((cell) => cell.textContent)).toEqual(["Cloud", "$12"]);
    expect(content.querySelector("caption")?.textContent).toBe("Segment revenue");
    expect(content.querySelector("script, img, [onerror]")).toBeNull();
    const anchors = [...content.querySelectorAll("a")];
    expect(anchors.map((anchor) => anchor.getAttribute("href"))).toEqual([null, "https://www.sec.gov/exhibit"]);
    expect(anchors[1].getAttribute("rel")).toBe("noopener noreferrer");
    const marks = [...content.querySelectorAll("mark")];
    expect(marks.map((mark) => [mark.textContent, mark.closest("td") !== null])).toEqual([["Cloud", true], ["$12", true]]);
    expect([...root.querySelectorAll("a")].find((anchor) => anchor.textContent === "Open original")?.getAttribute("href")).toBe("https://www.sec.gov/filing#:~:text=Cloud%20%2412");
  });

  test("maps captured SEC separators onto actual cells without replacing the HTML", async () => {
    const source: SourceDocument = {
      ...TEXT,
      id: "evidence:panel:sec-table",
      representation: "html",
      content: "<h2>Item 7</h2><table><thead><tr><th>Metric</th><th>2026</th></tr></thead><tbody><tr><td>Revenue</td><td>10</td></tr></tbody></table>",
      matchingText: "Item 7\nMetric | 2026\nRevenue | 10",
      quote: "Revenue | 10",
      location: { offset: 21, end: 33, basis: "rendered" },
    };
    serve({ body: source });
    const root = await mount(panel(source.id));
    await until(resolved(root));
    const content = root.querySelector('[aria-label="Captured source content"]');
    expect(content?.querySelector("h2")?.textContent).toBe("Item 7");
    expect([...content?.querySelectorAll("th") ?? []].map((cell) => cell.textContent)).toEqual(["Metric", "2026"]);
    expect([...content?.querySelectorAll("mark") ?? []].map((mark) => [mark.textContent, mark.closest("td")?.cellIndex])).toEqual([["Revenue", 0], ["10", 1]]);
    expect(content?.textContent).toBe("Item 7Metric2026Revenue10");
    expect(root.querySelector("blockquote")?.textContent).toContain("Revenue | 10");
  });

  test("an unsupported SEC mapping shows no selected passage", async () => {
    const source: SourceDocument = {
      ...TEXT,
      id: "evidence:panel:unsupported",
      representation: "html",
      content: "<p>Revenue</p><p>Revenue</p>",
      matchingText: "Earlier content\nRevenue\nRevenue",
      quote: "Revenue",
      location: { offset: 0, end: 7, basis: "rendered" },
    };
    serve({ body: source });
    const root = await mount(panel(source.id));
    await until(() => root.textContent?.includes("No passage is selected.") ?? false);
    expect(root.querySelector("mark")).toBeNull();
    expect([...root.querySelectorAll("button")].some((entry) => entry.textContent === "Go to quote")).toBe(false);
    expect(root.querySelector('[aria-label="Captured source content"]')?.textContent).toBe("RevenueRevenue");
  });

  test("projects an Exa HTML quote to rendered text for the marks, the display, and the original link", async () => {
    const requests = serve({ body: EXA });
    const root = await mount(panel(EXA.id));
    await until(resolved(root));
    const projected = "Nvidia's sales soared & margins held.\nGetty Images\nRevenue dou";
    expect(requests).toEqual(["/api/research/sources/evidence%3Apanel%3Aexa"]);
    expect(shownQuote(root)).toBe(projected);
    expect(markTexts(root)).toEqual(["Nvidia's ", "sales ", "soared", " & margins held.", "Getty Images", "Revenue dou"]);
    expect(root.querySelector('[aria-label="Captured source content"] strong em mark')?.textContent).toBe("soared");
    expect(originalHref(root)).toBe(fragment(EXA.url ?? "", "Nvidia's sales soared & margins held. Getty Images Revenue dou"));
    expect(root.textContent).toContain("sha256-of-raw-exa-body");
    expect(root.textContent).toContain("Provider extraction — not the original document");
  });

  test("a literal angle-bracket match wins over the projected markup", async () => {
    const root = await open({
      ...EXA,
      content: "<p>Template &lt;b&gt;Revenue&lt;/b&gt; stays literal.</p><p>Bold <b>Revenue</b> elsewhere.</p>",
      quote: "<b>Revenue</b>",
    });
    expect(root.textContent).toContain("The quote matches the captured source.");
    expect(markTexts(root)).toEqual(["<b>Revenue</b>"]);
    expect(root.querySelector('[aria-label="Captured source content"] b mark')).toBeNull();
    expect(shownQuote(root)).toBe("<b>Revenue</b>");
    expect(originalHref(root)).toBe(fragment(EXA.url ?? "", "<b>Revenue</b>"));
  });

  test("an ambiguous literal match stays ambiguous even when the projection is unique", async () => {
    const root = await open({
      ...EXA,
      captureKind: "search_excerpt",
      content: "<p>a &lt;i&gt;b</p><p>a &lt;i&gt;b</p><p>a <i>b</i> c</p>",
      quote: "a <i>b",
    });
    expect(root.textContent).toContain("Ambiguous quote: 2 matches remain. No passage is selected.");
    expect(markTexts(root)).toEqual([]);
    expect(shownQuote(root)).toBe("a <i>b");
    expect(originalHref(root)).toBe(EXA.url);
  });
  test("a search excerpt resolves through the same rendered-quote projection", async () => {
    const root = await open({
      ...EXA,
      captureKind: "search_excerpt",
      content: "<p>Sales <em>rose</em> in May.</p>",
      quote: "Sales <em>rose</em>",
    });
    expect(root.textContent).toContain("The quote matches the captured source.");
    expect(markTexts(root)).toEqual(["Sales ", "rose"]);
    expect(shownQuote(root)).toBe("Sales rose");
    expect(originalHref(root)).toBe(fragment(EXA.url ?? "", "Sales rose"));
  });


  test("a projected quote with two rendered occurrences stays ambiguous", async () => {
    const root = await open({ ...EXA, content: "<p>Sales <em>rose</em> in May.</p><p>Sales <b>rose</b> in June.</p>", quote: "Sales <em>rose</em>" });
    expect(root.textContent).toContain("Ambiguous quote: 2 matches remain. No passage is selected.");
    expect(markTexts(root)).toEqual([]);
    expect([...root.querySelectorAll("button")].some((entry) => entry.textContent === "Go to quote")).toBe(false);
    expect(shownQuote(root)).toBe("Sales rose");
    expect(originalHref(root)).toBe(EXA.url);
  });

  test("truncated markup or entities that cannot match stay unresolved with the stored quote", async () => {
    for (const quote of ["Tom &am", "Jerry</p><p>Spike <em cla"]) {
      const root = await open({ ...EXA, content: "<p>Tom &amp; Jerry</p>", quote });
      expect(root.textContent).toContain("The quote does not resolve in the rendered source. No passage is selected.");
      expect(markTexts(root)).toEqual([]);
      expect(shownQuote(root)).toBe(quote);
      expect(originalHref(root)).toBe(EXA.url);
      await close();
    }
  });

  test("original HTML and text captures never project markup quotes", async () => {
    for (const source of [
      { ...EXA, captureKind: "original" as const, content: "<p><em>Revenue</em> grew</p>" },
      { ...EXA, representation: "text" as const, content: "Revenue grew" },
    ]) {
      const root = await open({ ...source, quote: "<em>Revenue</em> grew" });
      expect(root.textContent).toContain("The quote does not resolve in the rendered source. No passage is selected.");
      expect(markTexts(root)).toEqual([]);
      expect(shownQuote(root)).toBe("<em>Revenue</em> grew");
      await close();
    }
  });

  test("inline markup and entities project to the exact rendered sentence", async () => {
    const word = fc.oneof(fc.stringMatching(/^[a-z]{1,6}$/), fc.constant("&"));
    const tag = fc.constantFrom("", "em", "strong", "b", "i");
    await fc.assert(
      fc.asyncProperty(fc.array(fc.tuple(word, tag), { minLength: 2, maxLength: 6 }), async (words) => {
        const plain = words.map(([text]) => text).join(" ");
        const html = words.map(([text, name]) => {
          const escaped = text === "&" ? "&amp;" : text;
          return name ? `<${name}>${escaped}</${name}>` : escaped;
        }).join(" ");
        const root = await open({ ...EXA, content: `<p>0000 ${html} 1111</p><p>2222</p>`, quote: html });
        expect(root.textContent).toContain("The quote matches the captured source.");
        expect(shownQuote(root)).toBe(plain);
        expect(markTexts(root).join("").replace(/\s/gu, "")).toBe(plain.replace(/\s/gu, ""));
        expect(originalHref(root)).toBe(fragment(EXA.url ?? "", plain));
        await close();
      }),
      { seed: 20261010, numRuns: 25 },
    );
  }, 20_000);

  test("shows the source error, then recovers the capture through Retry source", async () => {
    const requests = serve({ body: { error: "source capture not found" }, status: 404 }, { body: TEXT });
    const root = await mount(panel(TEXT.id));
    await until(() => root.querySelector('[role="alert"]') !== null);
    expect(root.querySelector('[role="alert"]')?.textContent).toContain("source capture not found");
    expect(root.querySelector("h3")).toBeNull();
    expect(root.querySelectorAll("mark").length).toBe(0);
    await click(root, "Retry source");
    await until(resolved(root));
    expect(requests).toEqual([TEXT_URL, TEXT_URL]);
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect(root.querySelector("h3")?.textContent).toBe("10-K");
    expect(root.querySelector("mark")?.textContent).toBe("Revenue   grew");
    expect(dom.window.document.activeElement).toBe(root.querySelector("aside"));
  });

  test("navigates sources, returns to the quote, and closes through panel controls", async () => {
    serve({ body: TEXT });
    const events: string[] = [];
    const root = await mount(panel(TEXT.id, 2, events));
    await until(resolved(root));
    expect(root.textContent).toContain("1 of 2");
    const mark = root.querySelector("mark");
    mark?.classList.remove("mgl-active-element-highlight");
    await click(root, "Go to quote");
    expect(mark?.classList.contains("mgl-active-element-highlight")).toBe(true);
    for (const label of ["Previous source", "Next source", "Close source"]) await click(root, label);
    await act(async () => {
      root.querySelector("aside")?.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(events).toEqual(["navigate:-1", "navigate:1", "close", "close"]);
  });

  test("console navigation shows only the selected capture while loading and after failure", async () => {
    const next: SourceDocument = { ...TEXT, id: "evidence:panel:next", title: "8-K", content: "Margins fell sharply.", quote: "Margins fell" };
    const { promise: nextLoaded, resolve: releaseNext } = Promise.withResolvers<void>();
    const requests = serve({ body: TEXT }, { body: next, gate: nextLoaded }, { body: { error: "source capture not found" }, status: 404 });
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      { type: "evidence", evidence: [{ id: "ev:abc123", source: "sec", retrievedAt: "2026-10-10T00:00:00Z", content: "", sources: [{ id: TEXT.id }, { id: next.id }, { id: "evidence:panel:missing" }] }] },
      { type: "answer_delta", text: "Risk grew [ev:abc123]." },
    ];
    const root = await mount(<AgentConsole events={events} reasoning={false} debug={false} />);
    await click(root, "[ev:abc123]");
    await until(resolved(root));
    expect(root.querySelector("aside h3")?.textContent).toBe("10-K");

    await click(root, "Next source");
    const loading = root.querySelector("aside")?.textContent ?? "";
    expect(loading).toContain("2 of 3");
    expect(loading).toContain("Loading captured source…");
    expect(loading).not.toContain("10-K");
    expect(root.querySelector("aside h3")).toBeNull();
    expect(root.querySelector("aside mark")).toBeNull();
    await act(async () => releaseNext());
    await until(() => root.querySelector("aside mark")?.textContent === "Margins fell");
    expect(root.querySelector("aside h3")?.textContent).toBe("8-K");

    await click(root, "Next source");
    await until(() => root.querySelector('aside [role="alert"]') !== null);
    const failed = root.querySelector("aside")?.textContent ?? "";
    expect(failed).toContain("3 of 3");
    expect(failed).toContain("source capture not found");
    expect(failed).not.toContain("8-K");
    expect(root.querySelector("aside h3")).toBeNull();
    expect(root.querySelector("aside mark")).toBeNull();
    expect(requests).toEqual([TEXT_URL, "/api/research/sources/evidence%3Apanel%3Anext", "/api/research/sources/evidence%3Apanel%3Amissing"]);
  });

  test("rejects a provider-extracted PDF before it reaches the PDF renderer", async () => {
    const extracted: SourceDocument = {
      ...TEXT,
      id: "evidence:panel:pdf",
      url: "https://publisher.example/notice",
      representation: "pdf",
      captureKind: "provider_extraction",
      content: undefined,
      pdfUrl: "https://publisher.example/notice.pdf",
    };
    const requests = serve({ body: extracted });
    const root = await mount(panel(extracted.id));
    await until(() => root.querySelector('[role="alert"]') !== null);
    expect(requests).toEqual(["/api/research/sources/evidence%3Apanel%3Apdf"]);
    expect(root.querySelector("h3")).toBeNull();
    expect(root.querySelector('[aria-label="Captured PDF pages"]')).toBeNull();
    expect(root.querySelector('[aria-label="Captured source content"]')).toBeNull();
    expect([...root.querySelectorAll("button")].some((entry) => entry.textContent === "Go to quote")).toBe(false);
  });
});
