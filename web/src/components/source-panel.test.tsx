import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import fc from "fast-check";
import type { Root } from "react-dom/client";
import { AgentConsole } from "./agent-console";
import { SourcePage } from "./source-panel";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import type { SourceAudit, SourceDocument } from "../lib/research-client";

// Static import cannot work: react-dom reads canUseDOM once at module load, so it loads after the jsdom globals exist.
const { createRoot } = await import("react-dom/client");

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
  linkStatus: "stockbot_render",
  originalUrl: "https://www.sec.gov/filing",
};

const TEXT_URL = "/api/research/sources/evidence%3Apanel%3Atext";
const sourceUrl = (id: string) => `/api/research/sources/${encodeURIComponent(id)}`;

const realFetch = globalThis.fetch;
let mounted: Root | null = null;

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
  globalThis.fetch = realFetch;
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

function panel(id: string): ReactNode {
  return <SourcePage id={id} />;
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
const official = (root: HTMLElement) => [...root.querySelectorAll("a")].find((anchor) => /^(?:Open|Original) on /u.test(anchor.textContent ?? ""));
const originalHref = (root: HTMLElement) => official(root)?.getAttribute("href");
const markTexts = (root: HTMLElement) => [...root.querySelectorAll('[aria-label="Captured source content"] mark')].map((mark) => mark.textContent);
const hrefs = (root: HTMLElement) => [...root.querySelectorAll("a[href]")].map((anchor) => anchor.getAttribute("href") ?? "");
const API_HOST = /^https?:\/\/(?:data\.sec\.gov|efts\.sec\.gov|api\.finra\.org)\//u;
// Each console source card: its label, then [text, href, target, rel] for each link.
const sourceCards = (root: HTMLElement) => [...root.querySelectorAll('[aria-label="Captured sources"] > span')].slice(1).map((card) => [
  card.firstElementChild?.textContent,
  ...[...card.querySelectorAll("a")].map((anchor) => [anchor.textContent, anchor.getAttribute("href"), anchor.getAttribute("target"), anchor.getAttribute("rel")]),
]);

const EXA: SourceDocument = {
  ...TEXT,
  id: "evidence:panel:exa",
  title: "Nvidia results",
  url: "https://news.example/article",
  originalUrl: "https://news.example/article",
  publisher: "Exa",
  contentHash: "sha256-of-raw-exa-body",
  representation: "html",
  captureKind: "provider_extraction",
  content: `<h1>Chip results</h1><p>Nvidia&#39;s <strong>sales <em>soared</em></strong> &amp; margins held.</p><figure><p>Getty Images</p></figure><p>Revenue doubled.</p>`,
  quote: `Nvidia&#39;s <strong>sales <em>soared</em></strong> &amp; margins held.</p><figure><p>Getty Images</p></figure><p>Revenue dou`,
};

// Each "How we got this" term with its shown value.
function how(root: HTMLElement): Record<string, string | null | undefined> {
  return Object.fromEntries([...root.querySelectorAll("dl dt")].map((term) => [term.textContent, term.nextElementSibling?.textContent]));
}

describe("SourcePage", () => {
  test("renders text captures with metadata and marks the exact quote", async () => {
    const requests = serve({ body: TEXT });
    const root = await mount(panel(TEXT.id));
    await until(resolved(root));
    expect(requests).toEqual([TEXT_URL]);
    expect(root.querySelector("h1")?.textContent).toBe("SEC · 10-K · 2026-01-01");
    expect(root.textContent).toContain("Original document");
    expect(how(root)).toMatchObject({ Source: "SEC", Published: "2026-01-01", Retrieved: "2026-01-02", "Content hash": "abc" });
    expect([official(root)?.textContent, originalHref(root)]).toEqual(["Original on SEC.gov ↗", "https://www.sec.gov/filing"]);
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
    // The page never builds its own text fragment: resolving the quote leaves the plain original link.
    expect(originalHref(root)).toBe("https://www.sec.gov/filing");
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

  test("names missing capture metadata and renders a capture with no quote or official link", async () => {
    const sparse: SourceDocument = {
      id: "evidence:panel:sparse",
      title: "Notice",
      representation: "text",
      captureKind: "original",
      content: "No citation text captured.",
      quote: "",
      linkStatus: "missing",
    };
    const root = await open(sparse);
    expect([...root.querySelectorAll("dl dd")].map((cell) => cell.textContent)).toEqual(Array(7).fill("Not recorded"));
    expect(root.textContent).toContain("Original document");
    expect(root.querySelector("blockquote")).toBeNull();
    expect(root.querySelector("mark")).toBeNull();
    expect(root.querySelector('a[href^="https://"]')).toBeNull();
    expect(root.textContent).toContain("The quote does not resolve in the rendered source. No passage is selected.");
  });

  test("a raw-basis location never selects through rendered coordinates", async () => {
    const content = "<p>Revenue grew in Q1.</p><p>Revenue grew in Q2.</p>";
    const byContext: SourceDocument = {
      ...TEXT,
      id: "evidence:panel:raw-basis",
      representation: "html",
      content,
      quote: "Revenue grew in Q1.",
      location: { offset: 0, end: 19, basis: "raw", prefix: "Q9." },
    };
    const rejected = await open(byContext);
    expect(rejected.textContent).toContain("The quote does not resolve in the rendered source. No passage is selected.");
    expect(rejected.querySelector("mark")).toBeNull();
    expect(rejected.querySelector('[aria-label="Captured source content"]')?.textContent).toBe("Revenue grew in Q1.Revenue grew in Q2.");
    await close();

    const rendered: SourceDocument = { ...byContext, id: "evidence:panel:rendered-basis", location: { offset: 0, end: 19, basis: "rendered" } };
    const selected = await open(rendered);
    expect(selected.textContent).toContain("The quote matches the captured source.");
    expect([...selected.querySelectorAll("mark")].map((mark) => mark.textContent)).toEqual(["Revenue grew in Q1."]);
    expect(originalHref(selected)).toBe("https://www.sec.gov/filing");
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
    expect(originalHref(root)).toBe(EXA.url);
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
    expect(originalHref(root)).toBe(EXA.url);
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
    expect(originalHref(root)).toBe(EXA.url);
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
        expect(originalHref(root)).toBe(EXA.url);
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
    expect(root.querySelector("h1")).toBeNull();
    expect(root.querySelectorAll("mark").length).toBe(0);
    await click(root, "Retry source");
    await until(resolved(root));
    expect(requests).toEqual([TEXT_URL, TEXT_URL]);
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect(root.querySelector("h1")?.textContent).toBe("SEC · 10-K · 2026-01-01");
    expect(root.querySelector("mark")?.textContent).toBe("Revenue   grew");
  });

  test("Go to quote returns to the marked passage", async () => {
    serve({ body: TEXT });
    const root = await mount(panel(TEXT.id));
    await until(resolved(root));
    const mark = root.querySelector("mark");
    mark?.classList.remove("mgl-active-element-highlight");
    await click(root, "Go to quote");
    expect(mark?.classList.contains("mgl-active-element-highlight")).toBe(true);
  });

  test("a highlighted SEC original links the verified fragment and never an API host", async () => {
    const page = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000052/nvda-20260426.htm";
    const highlight = `${page}#f-30:~:text=Revenue,81%2C615`;
    const root = await open({
      ...TEXT,
      id: "evidence:panel:highlighted",
      url: "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json",
      representation: "html",
      content: "<table><tr><td>Revenue</td><td>81,615</td></tr></table>",
      quote: "Revenue 81,615",
      toolName: "get_xbrl_facts",
      linkStatus: "highlighted_original",
      originalUrl: page,
      highlightUrl: highlight,
    });
    expect(root.textContent).toContain("The quote matches the captured source.");
    const link = official(root);
    expect([link?.textContent, link?.getAttribute("href"), link?.getAttribute("target"), link?.getAttribute("rel")]).toEqual(["Open on SEC.gov (highlighted) ↗", highlight, "_blank", "noopener noreferrer"]);
    expect(hrefs(root)).toEqual([highlight]);
    expect(hrefs(root).some((href) => API_HOST.test(href))).toBe(false);
    expect(Object.keys(how(root))).toEqual(["Source", "Published", "Tool", "Arguments", "Retrieved", "Known at", "Content hash"]);
  });

  test("an official FINRA file renders as text with a plain original link and no table", async () => {
    const file = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260102.txt";
    const root = await open({
      ...TEXT,
      id: "evidence:panel:finra",
      title: "Reg SHO daily short volume",
      publisher: "FINRA",
      url: "https://api.finra.org/data/group/otcMarket/name/regShoDaily",
      content: "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n20260102|AAPL|100|0|300|B,Q,N\n20260102|ACME|5|0|9|B,Q,N\n",
      quote: "20260102|ACME|5|0|9|B,Q,N",
      toolName: "get_reg_sho_daily",
      toolArgs: { ticker: "ACME" },
      linkStatus: "stockbot_render",
      originalUrl: file,
    });
    expect(root.textContent).toContain("The quote matches the captured source.");
    expect(markTexts(root)).toEqual(["20260102|ACME|5|0|9|B,Q,N"]);
    expect(root.querySelector("table")).toBeNull();
    const link = official(root);
    expect([link?.textContent, link?.getAttribute("href"), link?.getAttribute("target"), link?.getAttribute("rel")]).toEqual(["Original on FINRA.org ↗", file, "_blank", "noopener noreferrer"]);
    expect(link?.getAttribute("href")).not.toContain("#:~:");
    expect(hrefs(root).some((href) => API_HOST.test(href))).toBe(false);
    expect(how(root)).toMatchObject({ Source: "FINRA", Tool: "get_reg_sho_daily", Arguments: JSON.stringify({ ticker: "ACME" }, null, 2) });
    expect(Object.keys(how(root))).not.toContain("Upstream API");
    expect(Object.keys(how(root))).not.toContain("Public page");
  });

  test("a missing official page shows the held record as text with only the Stockbot view and no warning", async () => {
    const row = JSON.stringify({ settlementDate: "2026-01-02", symbolCode: "ACME" }, null, 2);
    const root = await open({
      ...TEXT,
      id: "evidence:panel:missing",
      title: "Threshold list",
      publisher: "FINRA",
      url: "https://api.finra.org/data/group/otcMarket/name/thresholdList",
      captureKind: "structured_record",
      content: row,
      quote: "",
      linkStatus: "missing",
      originalUrl: undefined,
    });
    expect(root.textContent).toContain("Structured record — no official page");
    expect(root.querySelector('[aria-label="Captured source content"]')?.textContent).toBe(row);
    expect(root.querySelector("table")).toBeNull();
    expect(official(root)).toBeUndefined();
    expect(hrefs(root)).toEqual([]);
    expect(root.querySelector('[role="alert"]')).toBeNull();
  });

  test("unsafe official links reject the document instead of rendering a link", async () => {
    for (const links of [
      { originalUrl: "javascript:alert(1)" },
      { originalUrl: "https://user:secret@www.sec.gov/filing" },
      { linkStatus: "highlighted_original", highlightUrl: "javascript:alert(1)" },
    ]) {
      serve({ body: { ...TEXT, id: "evidence:panel:unsafe", ...links } });
      const root = await mount(panel("evidence:panel:unsafe"));
      await until(() => root.querySelector('[role="alert"]') !== null);
      expect(root.textContent).toContain("The source links are invalid.");
      expect(hrefs(root)).toEqual([]);
      await close();
    }
  });

  // Each term of one audit <dl> with its shown value.
  const auditFields = (list: Element | null | undefined) => Object.fromEntries([...list?.querySelectorAll(":scope > dt") ?? []].map((term) => [term.textContent, term.nextElementSibling?.textContent]));
  const auditInput = (id: string, value: string, extra: Partial<SourceAudit> = {}): SourceAudit => ({
    evidenceId: id, sourceId: `evidence:${id}`, claimType: "observed_fact", verification: "verified", verificationReason: "", value, entity: "NVDA", period: "Q1 FY2025", unit: "USD millions",
    source: { name: "SEC", document: "nvda-20240428.htm", accession: "0001045810-24-000124", hash: "sha256:abc" }, ...extra,
  });
  const CALCULATION: SourceDocument = {
    id: "evidence:panel:calc", title: "Non-data-center revenue", publisher: "Stockbot", representation: "text", captureKind: "calculation", content: "Total revenue minus Data Center revenue", quote: "", linkStatus: "missing",
    audit: {
      evidenceId: "panel:calc", sourceId: "evidence:panel:calc", claimType: "calculation", verification: "verified", verificationReason: "", value: "3481", entity: "NVDA", period: "Q1 FY2025", unit: "USD millions",
      formula: "total - data center", formulaId: "difference", formulaVersion: "1", recomputedValue: "3481", inputs: [auditInput("EV-TOT", "26,044"), auditInput("EV-DC", "22,563")],
    },
  };

  test("public source mismatches and ambiguity preserve the verified API record", async () => {
    for (const failure of ["SOURCE_MISMATCH", "SOURCE_AMBIGUOUS"]) {
      const content = JSON.stringify({ symbolCode: "ACME", shortVolume: 100 }, null, 2);
      const reason = `The public file conflicts with the API record: <img src="https://example.test/conflict"> & ${failure}.`;
      const root = await open({
        ...TEXT,
        id: "evidence:panel:api-conflict",
        title: "Reg SHO daily short volume",
        publisher: "FINRA",
        url: "https://api.finra.org/data/group/otcMarket/name/regShoDaily",
        captureKind: "structured_record",
        content,
        quote: '"shortVolume": 100',
        linkStatus: "missing",
        originalUrl: undefined,
        sourceConflict: { failure, reason },
        audit: auditInput("panel:api-conflict", "100", {
          entity: "ACME", period: "2026-01-02", unit: "shares",
          source: { name: "FINRA", hash: "sha256:api-record" },
        }),
      });
      const notice = root.querySelector('[aria-label="Source conflict"]');
      expect(notice?.textContent).toContain("The public link is unavailable. The captured API record remains visible.");
      expect([...notice?.querySelectorAll("p") ?? []].map((paragraph) => paragraph.textContent)).toEqual([
        "The public link is unavailable. The captured API record remains visible.", failure, reason,
      ]);
      expect(notice?.querySelector("img")).toBeNull();
      expect(auditFields(root.querySelector('[aria-label="Source audit"] > dl'))).toMatchObject({
        Verification: "Verified", Value: "100", Entity: "ACME", Period: "2026-01-02", Unit: "shares",
        Source: "FINRA", "Evidence hash": "sha256:api-record",
      });
      expect(root.querySelector('[aria-label="Captured source content"]')?.textContent).toBe(content);
      expect(markTexts(root)).toEqual(['"shortVolume": 100']);
      expect(official(root)).toBeUndefined();
      expect(hrefs(root)).toEqual([]);
      expect(root.querySelector('[role="alert"]')).toBeNull();
      await close();
    }
  });

  test("a verified observation shows its canonical descriptor without changing the audit value rows", async () => {
    const root = await open({
      ...TEXT,
      audit: auditInput("panel:text", "26,044", {
        datapoint: {
          id: "dp:revenue", metric: "revenue", field: "total_revenue", value: "26044",
          entity: "NVDA", period: "Q1 FY2025", unit: "USD millions",
          statement: "NVDA reported revenue of <26,044> USD millions.",
        },
      }),
    });
    const audit = root.querySelector('[aria-label="Source audit"] > dl');
    expect(auditFields(audit)).toMatchObject({
      Verification: "Verified", Value: "26,044", Entity: "NVDA", Period: "Q1 FY2025", Unit: "USD millions",
      "Datapoint ID": "dp:revenue", Metric: "revenue", Field: "total_revenue",
      Statement: "NVDA reported revenue of <26,044> USD millions.",
    });
    expect(Object.keys(auditFields(audit))).not.toContain("Formula");
    expect(Object.keys(auditFields(audit))).not.toContain("Input datapoint IDs");
  });

  test("a verified calculation shows its canonical formula, input IDs, and verified input descriptors", async () => {
    const input = {
      id: "dp:revenue", metric: "revenue", value: "26044", entity: "NVDA", period: "Q1 FY2025",
      statement: "NVDA reported total revenue of 26044 USD millions.",
    };
    const root = await open({
      ...CALCULATION,
      audit: {
        ...CALCULATION.audit!, formula: undefined, formulaId: undefined, formulaVersion: undefined,
        datapoint: {
          id: "dp:non-data-center", metric: "non_data_center_revenue", value: "3481",
          entity: "NVDA", period: "Q1 FY2025", statement: "Total revenue minus Data Center revenue.",
          formula: "total - data center", inputs: ["dp:revenue", "dp:data-center"],
        },
        inputs: [
          auditInput("EV-TOT", "26,044", { datapoint: input }),
          auditInput("EV-DC", "22,563", {
            datapoint: { ...input, id: "dp:data-center", metric: "data_center_revenue", value: "22563", statement: "NVDA reported Data Center revenue of 22563 USD millions." },
          }),
        ],
      },
    });
    const audit = root.querySelector('[aria-label="Source audit"] > dl');
    expect(auditFields(audit)).toMatchObject({
      Verification: "Verified", Value: "3481", Entity: "NVDA", Period: "Q1 FY2025", Unit: "USD millions",
      "Datapoint ID": "dp:non-data-center", Metric: "non_data_center_revenue",
      Statement: "Total revenue minus Data Center revenue.", Formula: "total - data center",
      "Input datapoint IDs": "dp:revenue, dp:data-center",
    });
    expect(Object.keys(auditFields(audit))).not.toContain("Field");
    expect(auditFields(root.querySelector('[aria-label="Calculation inputs"] > li dl'))).toMatchObject({
      Verification: "Verified", Value: "26,044", "Datapoint ID": "dp:revenue", Metric: "revenue",
      Statement: input.statement,
    });
    expect([...root.querySelectorAll('[aria-label="Calculation inputs"] > li dl')].map((list) => auditFields(list)["Datapoint ID"])).toEqual(["dp:revenue", "dp:data-center"]);
  });

  test("nonverified audit nodes hide canonical descriptors without hiding verified input descriptors", async () => {
    const datapoint = {
      id: "dp:held", metric: "held_metric", field: "held_field", value: "12", entity: "NVDA", period: "Q1 FY2025",
      statement: "A canonical statement.", formula: "canonical-only formula", inputs: ["dp:input"],
    };
    for (const verification of ["unverified", "conflicted"] as const) {
      const root = await open({
        ...CALCULATION,
        audit: {
          ...CALCULATION.audit!, verification, verificationReason: "The proof did not verify.", datapoint,
          inputs: [
            auditInput("EV-VERIFIED", "12", { datapoint }),
            auditInput("EV-NOT-VERIFIED", "12", { verification, verificationReason: "The proof did not verify.", datapoint }),
          ],
        },
      });
      const audit = auditFields(root.querySelector('[aria-label="Source audit"] > dl'));
      const inputs = [...root.querySelectorAll('[aria-label="Calculation inputs"] > li dl')].map(auditFields);
      for (const fields of [audit, inputs[1]!]) {
        for (const name of ["Datapoint ID", "Metric", "Field", "Statement", "Input datapoint IDs"]) {
          expect(Object.keys(fields)).not.toContain(name);
        }
        expect(fields.Formula).not.toBe(datapoint.formula);
        expect(fields.Verification).toBe(`${verification === "unverified" ? "Not verified" : "Conflicted"} — The proof did not verify.`);
      }
      expect(audit.Formula).toBe("total - data center");
      expect(inputs[0]).toMatchObject({ Verification: "Verified", "Datapoint ID": "dp:held", Formula: datapoint.formula });
      await close();
    }
  });

  test("a calculation view shows the formula and opens each input record, never one arbitrary original", async () => {
    const root = await open(CALCULATION);
    const audit = root.querySelector('[aria-label="Source audit"]');
    expect(audit?.querySelector("h2")?.textContent).toBe("Calculation");
    expect(root.textContent).toContain("Stockbot calculation — no single original document");
    expect(auditFields(audit?.querySelector(":scope > dl"))).toEqual({
      Verification: "Verified", Value: "3481", Entity: "NVDA", Period: "Q1 FY2025", Unit: "USD millions", Formula: "total - data center", Recomputed: "3481",
    });
    const inputs = [...root.querySelectorAll('[aria-label="Calculation inputs"] > li')].map((item) => [
      auditFields(item.querySelector("dl")),
      [...item.querySelectorAll("a")].map((anchor) => [anchor.textContent, anchor.getAttribute("href"), anchor.getAttribute("target"), anchor.getAttribute("rel")]),
    ]);
    const source = "SEC · nvda-20240428.htm · 0001045810-24-000124";
    expect(inputs).toEqual([
      [{ Verification: "Verified", Value: "26,044", Entity: "NVDA", Period: "Q1 FY2025", Unit: "USD millions", Source: source, "Evidence hash": "sha256:abc" }, [["View in Stockbot", "#/source/evidence%3AEV-TOT", "_blank", "noopener noreferrer"]]],
      [{ Verification: "Verified", Value: "22,563", Entity: "NVDA", Period: "Q1 FY2025", Unit: "USD millions", Source: source, "Evidence hash": "sha256:abc" }, [["View in Stockbot", "#/source/evidence%3AEV-DC", "_blank", "noopener noreferrer"]]],
    ]);
    // The only links are the inputs' own Stockbot views: no official page stands in for the calculation.
    expect(official(root)).toBeUndefined();
    expect(hrefs(root)).toEqual(["#/source/evidence%3AEV-TOT", "#/source/evidence%3AEV-DC"]);
    expect(root.querySelector('[role="alert"]')).toBeNull();
  });

  test("an invalid calculation opens with its failed verification and no frontend warning", async () => {
    const reason = "The input is absent from the frozen observations.";
    const root = await open({
      ...CALCULATION,
      audit: { ...CALCULATION.audit!, verification: "unverified", verificationReason: "EV-GONE is not verified", recomputedValue: undefined, inputs: [auditInput("EV-TOT", "26,044"), { evidenceId: "EV-GONE", sourceId: "evidence:EV-GONE", claimType: "", verification: "unverified", verificationReason: reason }] },
    });
    const audit = root.querySelector('[aria-label="Source audit"]');
    expect(auditFields(audit?.querySelector(":scope > dl")).Verification).toBe("Not verified — EV-GONE is not verified");
    expect(Object.keys(auditFields(audit?.querySelector(":scope > dl")))).not.toContain("Recomputed");
    expect([...root.querySelectorAll('[aria-label="Calculation inputs"] > li dl')].map((list) => auditFields(list).Verification)).toEqual(["Verified", `Not verified — ${reason}`]);
    expect(root.querySelector('[role="alert"]')).toBeNull();
    expect(root.textContent).not.toContain("CALCULATION_INVALID");
  });

  test("an observation audit sits beside the two-link header and names its verification result", async () => {
    for (const [verification, reason, shown] of [["verified", "", "Verified"], ["conflicted", "value hash disagrees", "Conflicted — value hash disagrees"]] as const) {
      const root = await open({ ...TEXT, audit: auditInput("panel:text", "12", { verification, verificationReason: reason }) });
      const audit = root.querySelector('[aria-label="Source audit"]');
      expect(audit?.querySelector("h2")?.textContent).toBe("Observation");
      expect(auditFields(audit?.querySelector(":scope > dl"))).toMatchObject({ Verification: shown, Value: "12", "Evidence hash": "sha256:abc" });
      expect(root.querySelector('[aria-label="Calculation inputs"]')).toBeNull();
      expect([official(root)?.textContent, originalHref(root)]).toEqual(["Original on SEC.gov ↗", "https://www.sec.gov/filing"]);
      expect(root.querySelectorAll("mark").length).toBe(1);
      await close();
    }
  });

  test("a repaired legacy row shows repair history as read-only text with no links", async () => {
    const audit = auditInput("panel:text", "12", {
      verification: "unverified",
      verificationReason: "no proof",
      repairs: [{ repair_id: "repair:abc", status: "verified", category: "eps_without_accession", datapoint: "eps_diluted", value: "0.76" }],
    });
    const root = await open({ ...TEXT, audit });
    const items = [...root.querySelectorAll('[aria-label="Legacy repairs"] > li')];
    expect(items.map((item) => item.textContent)).toEqual(["repair:abc — verified · eps_without_accession · eps_diluted · 0.76"]);
    expect(items.flatMap((item) => [...item.querySelectorAll("a")])).toEqual([]);
    await close();
  });
  test("a later source failure shows only the controlled error and retries without the internal failure class", async () => {
    for (const [status, error, failure] of [
      [404, "source capture not found", "SOURCE_IDENTITY_MISSING"],
      [409, "invalid source capture", "SOURCE_MISMATCH"],
      [409, "invalid source capture", "SOURCE_AMBIGUOUS"],
      [404, "source capture not found", "SOURCE_UNAVAILABLE"],
      [502, "source read failed", "SOURCE_UNAVAILABLE"],
      [409, "invalid source capture", "SOURCE_CORRUPT"],
    ] as const) {
      const requests = serve({ body: { id: TEXT.id, error, status, failure }, status }, { body: TEXT });
      const root = await mount(panel(TEXT.id));
      await until(() => root.querySelector('[role="alert"]') !== null);
      expect(root.querySelector('[role="alert"]')?.textContent).toBe(`${error}Retry source`);
      expect(root.textContent).not.toContain(failure);
      expect([root.querySelector("h1"), root.querySelector('[aria-label="Source audit"]'), official(root)]).toEqual([null, null, undefined]);
      await click(root, "Retry source");
      await until(resolved(root));
      expect(requests).toEqual([TEXT_URL, TEXT_URL]);
      expect(root.querySelector('[role="alert"]')).toBeNull();
      await close();
    }
  });

  test("console resolves displayed sources once and opens verified highlights before plain originals", async () => {
    const page = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000052/nvda-20260426.htm";
    const highlight = `${page}#f-30:~:text=Revenue%20grew`;
    const source = { ...TEXT, title: "10-K annual report", originalUrl: page, linkStatus: "highlighted_original", highlightUrl: highlight } satisfies SourceDocument;
    const plain = { ...TEXT, id: "evidence:panel:plain", title: "10-Q", originalUrl: page };
    const missing = { ...TEXT, id: "evidence:panel:missing", title: "Short interest", publisher: "FINRA", linkStatus: "missing", originalUrl: undefined } satisfies SourceDocument;
    const web = { ...EXA, linkStatus: "highlighted_original", highlightUrl: `${EXA.originalUrl}#:~:text=Revenue%20doubled` } satisfies SourceDocument;
    const pdf = { ...TEXT, id: "evidence:panel:pdf", title: "Report PDF", representation: "pdf", content: undefined, originalUrl: "https://publisher.example/report.pdf", publisher: "Publisher" } satisfies SourceDocument;
    const sources = [source, plain, missing, web, pdf];
    const { promise: loaded, resolve: release } = Promise.withResolvers<void>();
    const requests = serve(...[source, web, pdf, plain, missing].map((body, index) => ({ body, gate: index === 0 ? loaded : undefined })));
    const finished: AgentEvent = { type: "done", metrics: { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 1, totalMs: 1 }, evidence: { count: 0, characters: 0 }, failures: {} } };
    const events: AgentEvent[] = [
      { type: "evidence", evidence: sources.map((entry, index) => ({ id: `ev:${index}`, source: "sec", title: "Evidence title", retrievedAt: "", content: "", sources: [{ id: entry.id, title: "Reference title" }] })) },
      { type: "answer_delta", text: "Revenue rose [ev:0]. Margins held [ev:0]. Web [ev:3]. PDF [ev:4]." },
      finished,
    ];
    const root = await mount(<AgentConsole events={events} reasoning={false} debug={false} />);
    const inline = () => [...root.querySelectorAll('a[aria-label^="Open source:"]')];
    const view = (id: string) => ["View provenance", `#/source/${encodeURIComponent(id)}`, "_blank", "noopener noreferrer"];
    expect(root.querySelectorAll('[aria-label="Loading original source…: Evidence title"]')).toHaveLength(2);
    expect(sourceCards(root)[0]).toEqual(["Evidence title", view(source.id)]);
    expect(root.querySelector('[aria-label="Captured sources"] [role="status"]')?.textContent).toBe("Loading original source…");
    await until(() => sourceCards(root)[4]?.[0] === "Publisher · Report PDF · 2026-01-01");
    expect(sourceCards(root)).toEqual([
      ["Evidence title", view(source.id)],
      ["SEC · 10-Q · 2026-01-01", ["Original on SEC.gov ↗", page, "_blank", "noopener noreferrer"], view(plain.id)],
      ["FINRA · Short interest · 2026-01-01", view(missing.id)],
      ["Exa · Nvidia results · 2026-01-01", ["Open on news.example ↗", web.highlightUrl, "_blank", "noopener noreferrer"], view(web.id)],
      ["Publisher · Report PDF · 2026-01-01", ["Original on publisher.example ↗", pdf.originalUrl, "_blank", "noopener noreferrer"], view(pdf.id)],
    ]);
    expect(root.querySelectorAll('[aria-label="Captured sources"] [role="status"]')[1]?.textContent).toBe("Original source unavailable");
    await act(async () => release());
    await until(() => inline().length === 4);
    expect(sourceCards(root)[0]).toEqual(["SEC · 10-K annual report · 2026-01-01", ["Open on SEC.gov ↗", highlight, "_blank", "noopener noreferrer"], view(source.id)]);
    expect(inline().map((anchor) => [anchor.textContent, anchor.getAttribute("href"), anchor.getAttribute("target"), anchor.getAttribute("rel")])).toEqual([
      ["[1]", highlight, "_blank", "noopener noreferrer"],
      ["[1]", highlight, "_blank", "noopener noreferrer"],
      ["[2]", web.highlightUrl, "_blank", "noopener noreferrer"],
      ["[3]", pdf.originalUrl, "_blank", "noopener noreferrer"],
    ]);
    expect(inline()[0]?.getAttribute("aria-label")).toBe("Open source: SEC · 10-K annual report · 2026-01-01");
    expect(hrefs(root).some((href) => API_HOST.test(href))).toBe(false);
    await act(async () => mounted?.render(<AgentConsole events={[...events, { type: "answer_delta", text: " More text." }, finished]} reasoning={false} debug={false} />));
    expect(requests).toEqual([source.id, web.id, pdf.id, plain.id, missing.id].map(sourceUrl));
    expect(root.textContent).not.toContain("ev:0");
    expect(root.textContent).not.toContain("evidence:panel");
    expect(root.querySelector("aside")).toBeNull();
  });

  test("console resolves checked snapshots while active and preserves sources after failure", async () => {
    const requests = serve({ body: TEXT });
    const evidence: AgentEvent = { type: "evidence", evidence: [{ id: "ev:cited", source: "sec", retrievedAt: "", content: "", sources: [{ id: TEXT.id }] }] };
    const chat = await mount(<AgentConsole events={[evidence]} reasoning={false} debug={false} />);
    expect(requests).toEqual([]);
    const partial: AgentEvent = { type: "answer_partial", text: "Revenue rose.", claims: [{ text: "Revenue rose.", claimType: "observed_fact", evidenceIds: ["ev:cited"] }] };
    await act(async () => mounted?.render(<AgentConsole events={[evidence, partial]} reasoning={false} debug={false} />));
    expect(requests).toEqual([TEXT_URL]);
    expect(chat.querySelector('a[aria-label^="Open source:"]')?.getAttribute("href")).toBe(TEXT.originalUrl);
    const provenance = [...chat.querySelectorAll("a")].find((anchor) => anchor.textContent === "View provenance");
    expect(provenance?.getAttribute("href")).toBe(`#/source/${encodeURIComponent(TEXT.id)}`);
    expect(chat.textContent).toContain("Partial checked answer · research continues");
    const final: AgentEvent = { type: "answer_delta", text: "The final filing analysis [ev:cited]." };
    await act(async () => mounted?.render(<AgentConsole events={[evidence, partial, final]} reasoning={false} debug={false} />));
    expect(chat.textContent).toContain("The final filing analysis");
    expect(chat.textContent).not.toContain("Revenue rose.");
    await act(async () => mounted?.render(<AgentConsole events={[evidence, partial, final, { type: "failed", category: "provider_error", message: "private provider diagnostic" }]} reasoning={false} debug={false} />));
    expect(chat.textContent).toContain("The final filing analysis");
    expect(chat.textContent).not.toContain("research continues");
    expect(chat.textContent).not.toContain("private provider diagnostic");
    expect(chat.querySelector('a[aria-label^="Open source:"]')?.getAttribute("href")).toBe(TEXT.originalUrl);
    expect(requests).toEqual([TEXT_URL]);
    await close();
    const reads = serve({ body: TEXT });
    const viewer = await mount(panel(TEXT.id));
    await until(resolved(viewer));
    expect(reads).toEqual([TEXT_URL]);
  });

  test("failed, unsafe, and mismatched sources never borrow reference URLs", async () => {
    const sources = [
      { id: "evidence:a", title: "Reference title", url: "https://publisher.example/bypass" },
      { id: "evidence:b", url: "https://publisher.example/bypass" },
      { id: "evidence:c", url: "https://publisher.example/bypass" },
    ];
    const requests = serve(
      { body: { error: "source capture not found" }, status: 404 },
      { body: { ...TEXT, id: sources[1]!.id, originalUrl: "javascript:alert(1)" } },
      { body: TEXT },
    );
    const finished: AgentEvent = { type: "done", metrics: { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 1, totalMs: 1 }, evidence: { count: 0, characters: 0 }, failures: {} } };
    const events: AgentEvent[] = [
      { type: "evidence", evidence: sources.map((source, index) => ({ id: `ev:${index}`, source: "finra", retrievedAt: "", content: "", sources: [source] })) },
      { type: "answer_delta", text: "A [ev:0]. B [ev:1]. C [ev:2]." },
      finished,
    ];
    const root = await mount(<AgentConsole events={events} reasoning={false} debug={false} />);
    await until(() => root.querySelectorAll('[aria-label^="Original source unavailable:"]').length === 3);
    expect(requests).toEqual(sources.map((source) => sourceUrl(source.id)));
    expect(sourceCards(root)).toEqual([
      ["Reference title", ["View provenance", "#/source/evidence%3Aa", "_blank", "noopener noreferrer"]],
      ["FINRA 2", ["View provenance", "#/source/evidence%3Ab", "_blank", "noopener noreferrer"]],
      ["FINRA 3", ["View provenance", "#/source/evidence%3Ac", "_blank", "noopener noreferrer"]],
    ]);
    expect([...root.querySelectorAll('[aria-label^="Original source unavailable:"]')].map((chip) => [chip.textContent, chip.getAttribute("title")])).toEqual([
      ["[1]", "Reference title — Original source unavailable"],
      ["[2]", "FINRA 2 — Original source unavailable"],
      ["[3]", "FINRA 3 — Original source unavailable"],
    ]);
    expect(root.querySelectorAll('[aria-label="Captured sources"] [role="status"]')).toHaveLength(3);
    expect(root.querySelector('a[aria-label^="Open source:"]')).toBeNull();
    expect(hrefs(root).every((href) => href.startsWith("#/source/"))).toBe(true);
    expect(root.textContent).not.toContain("ev:0");
  });

  test("calculation citations keep audit access without borrowing an input original", async () => {
    const requests = serve({ body: CALCULATION });
    const finished: AgentEvent = { type: "done", metrics: { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 1, totalMs: 1 }, evidence: { count: 0, characters: 0 }, failures: {} } };
    const events: AgentEvent[] = [
      { type: "evidence", evidence: [{ id: "ev:calc", source: "sec", retrievedAt: "", content: "", sources: [{ id: CALCULATION.id, url: TEXT.originalUrl }] }] },
      { type: "answer_delta", text: "Calculated revenue is 3481 [ev:calc]." },
      finished,
    ];
    const root = await mount(<AgentConsole events={events} reasoning={false} debug={false} />);
    await until(() => root.querySelector('[aria-label^="Original source unavailable:"]') !== null);
    expect(requests).toEqual([sourceUrl(CALCULATION.id)]);
    expect(sourceCards(root)).toEqual([["Stockbot · Non-data-center revenue", ["View provenance", "#/source/evidence%3Apanel%3Acalc", "_blank", "noopener noreferrer"]]]);
    expect(hrefs(root)).toEqual(["#/source/evidence%3Apanel%3Acalc"]);
    expect(root.querySelector('[aria-label^="Original source unavailable:"]')?.textContent).toBe("[1]");
    expect(root.querySelector('[aria-label="Captured sources"] [role="status"]')?.textContent).toBe("Original source unavailable");
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
    expect(root.querySelector("h1")).toBeNull();
    expect(root.querySelector('[aria-label="Captured PDF pages"]')).toBeNull();
    expect(root.querySelector('[aria-label="Captured source content"]')).toBeNull();
    expect([...root.querySelectorAll("button")].some((entry) => entry.textContent === "Go to quote")).toBe(false);
  });
});

describe("AgentConsole trio follow-up", () => {
  const done = { type: "done", metrics: { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 1, totalMs: 1 }, evidence: { count: 0, characters: 0 }, failures: {} } } as const;
  const offered: AgentEvent[] = [
    { type: "agent_start", prompt: "NVDA risks?" },
    { type: "answer_delta", text: "Stockbot view." },
    { type: "offer_trio", sessionId: "rs:1" },
    done,
  ];
  // Records every request body; each reply is one buffered SSE body.
  function serveTrio(sse: string): unknown[] {
    const bodies: unknown[] = [];
    globalThis.fetch = ((input: string | URL | Request, init?: RequestInit) => {
      bodies.push({ url: String(input), body: JSON.parse(String(init?.body)) });
      return Promise.resolve(new Response(sse));
    }) as unknown as typeof fetch;
    return bodies;
  }

  test("Yes posts the trio request for the offered session and shows the trio answer", async () => {
    const bodies = serveTrio(["data: {\"type\":\"agent_start\",\"prompt\":\"NVDA risks?\"}", "data: {\"type\":\"answer_delta\",\"text\":\"Bear and bull views.\"}", `data: ${JSON.stringify(done)}`, ""].join("\n\n"));
    const root = await mount(<AgentConsole events={offered} reasoning={false} debug={false} />);
    await click(root, "Yes");
    await until(() => root.textContent?.includes("Bear and bull views.") ?? false);
    expect(bodies).toEqual([{ url: "/api/agent", body: { op: "personas", sessionId: "rs:1", personas: ["stockbot", "bearbot", "bullbot"] } }]);
    expect(root.textContent).toContain("Stockbot view.");
    expect(root.textContent).not.toContain("trio team");
  });

  test("No dismisses the offer without a request", async () => {
    const bodies = serveTrio("");
    const root = await mount(<AgentConsole events={offered} reasoning={false} debug={false} />);
    await click(root, "No");
    expect(root.textContent).not.toContain("trio team");
    expect(bodies).toEqual([]);
  });

  test("a trio stream without a terminal event reports the failure", async () => {
    serveTrio("data: {\"type\":\"answer_delta\",\"text\":\"half\"}\n\n");
    const root = await mount(<AgentConsole events={offered} reasoning={false} debug={false} />);
    await click(root, "Yes");
    await until(() => root.textContent?.includes("stream ended without a terminal event") ?? false);
    expect(root.textContent).toContain("half");
    expect([...root.querySelectorAll("button")].some((button) => button.textContent === "Try again")).toBe(true);
  });
});
