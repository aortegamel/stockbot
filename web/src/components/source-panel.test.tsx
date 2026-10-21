import { dom } from "../test-dom";
import { afterEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import fc from "fast-check";
import type { Root } from "react-dom/client";
import { AgentConsole } from "./agent-console";
import { SourcePage } from "./source-panel";
import type { AgentEvent } from "../../../needle-harness/lib/agent/types";
import type { SourceDocument } from "../lib/research-client";

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
};

const TEXT_URL = "/api/research/sources/evidence%3Apanel%3Atext";

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

  test("names missing capture metadata and renders a capture with no quote", async () => {
    const sparse: SourceDocument = {
      id: "evidence:panel:sparse",
      title: "Notice",
      representation: "text",
      captureKind: "structured_record",
      content: "No citation text captured.",
      quote: "",
    };
    const root = await open(sparse);
    expect([...root.querySelectorAll("dl dd")].map((cell) => cell.textContent)).toEqual(Array(9).fill("Not recorded"));
    expect(root.textContent).toContain("Structured record");
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
    expect(originalHref(selected)).toBe("https://www.sec.gov/filing#:~:text=Revenue%20grew%20in%20Q1.");
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
    expect(originalHref(root)).toBe(`${EXA.url}#:~:text=Nvidia's%20sales%20soared%20%26%20margins,Revenue%20dou`);
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

  test("a structured record reads as a table with the cited row highlighted and shows how it was fetched", async () => {
    const api = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest";
    const record: SourceDocument = {
      id: "evidence:panel:finra",
      title: "FINRA",
      url: api,
      publisher: "FINRA",
      capturedAt: "2026-01-03",
      contentHash: "hash",
      representation: "json",
      captureKind: "structured_record",
      content: JSON.stringify({
        dataset_id: "otcMarket/consolidatedShortInterest",
        records: [{ symbolCode: "ACME", settlementDate: "2026-01-01", qty: 1 }, { symbolCode: "ACME", settlementDate: "2026-01-02", qty: 5, extra: { lot: 2 } }],
      }, null, 2),
      // The kernel cites the row as sorted, whitespace-collapsed JSON.
      quote: '{"extra": {"lot": 2}, "qty": 5, "settlementDate": "2026-01-02", "symbolCode": "ACME"}',
      toolName: "get_short_interest",
      toolResultId: "rs:1:tr:7",
      dataset: "otcMarket/consolidatedShortInterest",
      knownAt: "2026-01-02",
      upstreamUrl: api,
      toolArgs: { ticker: "ACME", limit: 100 },
    };
    const root = await open(record);
    expect(root.querySelector("h1")?.textContent).toBe("FINRA · otcMarket/consolidatedShortInterest · ACME · 2026-01-02");
    expect(root.textContent).toContain("The quote matches the captured source.");
    const table = root.querySelector('[aria-label="Captured source content"] table');
    expect([...table?.querySelectorAll("th") ?? []].map((cell) => cell.textContent)).toEqual(["symbolCode", "settlementDate", "qty", "extra"]);
    const cited = table?.querySelectorAll('tr[aria-current="true"]') ?? [];
    expect(cited).toHaveLength(1);
    expect([...cited[0].querySelectorAll("td")].map((cell) => cell.textContent)).toEqual(["ACME", "2026-01-02", "5", '{"lot":2}']);
    expect(cited[0].classList.contains("mgl-cited-element-highlight")).toBe(true);
    expect(root.querySelector("blockquote")).toBeNull();
    expect(root.textContent).not.toContain('{"extra"');
    expect(how(root)).toMatchObject({
      Source: "FINRA",
      Tool: "get_short_interest",
      Arguments: JSON.stringify({ ticker: "ACME", limit: 100 }, null, 2),
      "Upstream API": api,
      // FINRA records have no public page, even though the url names the API.
      "Public page": "Not recorded",
      "Known at": "2026-01-02",
      Retrieved: "2026-01-03",
    });
    const upstream = [...root.querySelectorAll("dd a")].find((anchor) => anchor.textContent === api);
    expect([upstream?.getAttribute("href"), upstream?.getAttribute("target"), upstream?.getAttribute("rel")]).toEqual([api, "_blank", "noopener noreferrer"]);
    cited[0].classList.remove("mgl-active-element-highlight");
    await click(root, "Go to quote");
    expect(cited[0].classList.contains("mgl-active-element-highlight")).toBe(true);
  });

  test("a structured record without a matching row selects no row", async () => {
    const root = await open({ ...TEXT, id: "evidence:panel:nomatch", representation: "json", captureKind: "structured_record", content: JSON.stringify({ filings: [{ form: "10-Q" }] }), quote: "ACME 2026-01-02" });
    expect(root.textContent).toContain("The quote does not resolve in the rendered source. No passage is selected.");
    expect(root.querySelectorAll("tbody tr")).toHaveLength(1);
    expect(root.querySelector('tr[aria-current="true"]')).toBeNull();
    expect(root.querySelector("blockquote")).toBeNull();
  });

  test("a cited SEC row links its filing page with the accession number highlighted", async () => {
    const index = "https://www.sec.gov/Archives/edgar/data/1045810/0002152188-26-000007-index.html";
    const row = { accession_no: "0002152188-26-000007", form: "4/A", source: index };
    const root = await open({ ...TEXT, id: "evidence:panel:secrow", publisher: "SEC", representation: "json", captureKind: "structured_record", content: JSON.stringify({ filings: [{ accession_no: "0001197647-26-000011", form: "4" }, row] }), quote: JSON.stringify(row), publicUrl: index });
    expect(root.querySelectorAll('tr[aria-current="true"]')).toHaveLength(1);
    expect(originalHref(root)).toBe(`${index}#:~:text=0002152188%2D26%2D000007`);
    await close();
    const unmatched = await open({ ...TEXT, id: "evidence:panel:secnomatch", publisher: "SEC", representation: "json", captureKind: "structured_record", content: JSON.stringify({ filings: [{ accession_no: "0001197647-26-000011" }] }), quote: JSON.stringify(row), publicUrl: index });
    expect(originalHref(unmatched)).toBe(index);
  });

  test("the public page link prefers publicUrl and unsafe URLs never become links", async () => {
    const root = await open({ ...TEXT, id: "evidence:panel:public", publicUrl: "https://www.sec.gov/public", upstreamUrl: "javascript:alert(1)", toolName: "get_sec_document", toolArgs: { accession_no: "0000000000-26-000042", document_name: "primary.htm" } });
    expect(originalHref(root)).toBe("https://www.sec.gov/public");
    expect(how(root)["Upstream API"]).toBe("Not recorded");
    expect(how(root).Tool).toBe("get_sec_document");
    expect(root.querySelector('a[href^="javascript:"]')).toBeNull();
    await close();
    const unsafe = await open({ ...TEXT, id: "evidence:panel:unsafe", url: "https://user:secret@example.com/filing" });
    expect(originalHref(unsafe)).toBeUndefined();
    expect(how(unsafe)["Public page"]).toBe("Not recorded");
  });

  test("console citations open the provenance page in a new tab with labels from the loaded sources", async () => {
    const source: SourceDocument = { ...TEXT, title: "10-K annual report" };
    const record: SourceDocument = {
      ...TEXT,
      id: "evidence:panel:record",
      title: "FINRA",
      publisher: "FINRA",
      representation: "json",
      captureKind: "structured_record",
      content: "{}",
      dataset: "otcMarket/consolidatedShortInterest",
      quote: JSON.stringify({ symbolCode: "ACME", settlementDate: "2026-01-02" }),
    };
    const { promise: loaded, resolve: release } = Promise.withResolvers<void>();
    const requests = serve({ body: source, gate: loaded }, { body: record });
    const events: AgentEvent[] = [
      { type: "evidence", evidence: [{ id: "ev:cited", source: "sec", title: "Evidence title", retrievedAt: "", content: "", sources: [{ id: source.id, title: "Reference title" }, { id: record.id }] }] },
      { type: "answer_delta", text: "Revenue rose [ev:cited]. Margins held [ev:cited]." },
    ];
    const root = await mount(<AgentConsole events={events} reasoning={false} debug={false} />);
    const inline = () => [...root.querySelectorAll('a[aria-label^="Open source:"]')];
    const listed = () => [...root.querySelectorAll('[aria-label="Captured sources"] a')];
    expect(inline().map((anchor) => [anchor.textContent, anchor.getAttribute("href"), anchor.getAttribute("target"), anchor.getAttribute("rel"), anchor.getAttribute("title")])).toEqual(
      Array(2).fill(["[1]", "#/source/evidence%3Apanel%3Atext", "_blank", "noopener noreferrer", "Evidence title"]),
    );
    await until(() => listed()[1]?.textContent === "FINRA · otcMarket/consolidatedShortInterest · ACME · 2026-01-02");
    expect(listed().map((anchor) => [anchor.textContent, anchor.getAttribute("href"), anchor.getAttribute("target"), anchor.getAttribute("rel")])).toEqual([
      ["Evidence title", "#/source/evidence%3Apanel%3Atext", "_blank", "noopener noreferrer"],
      ["FINRA · otcMarket/consolidatedShortInterest · ACME · 2026-01-02", "#/source/evidence%3Apanel%3Arecord", "_blank", "noopener noreferrer"],
    ]);
    await act(async () => release());
    await until(() => listed()[0]?.textContent === "SEC · 10-K annual report · 2026-01-01");
    expect(inline().map((anchor) => anchor.getAttribute("aria-label"))).toEqual(Array(2).fill("Open source: SEC · 10-K annual report · 2026-01-01"));
    await act(async () => mounted?.render(<AgentConsole events={[...events, { type: "answer_delta", text: " More text." }]} reasoning={false} debug={false} />));
    expect(requests).toEqual([TEXT_URL, "/api/research/sources/evidence%3Apanel%3Arecord"]);
    // Raw evidence ids never reach the reader, and no in-drawer panel opens.
    expect(root.textContent).not.toContain("ev:cited");
    expect(root.textContent).not.toContain("evidence:panel");
    expect(root.querySelector("aside")).toBeNull();
  });

  test("console source labels fall back to the reference title, then the publisher and index", async () => {
    const requests = serve({ body: { error: "source capture not found" }, status: 404 }, { body: { error: "source capture not found" }, status: 404 });
    const events: AgentEvent[] = [
      {
        type: "evidence", evidence: [
          { id: "ev:a", source: "sec", retrievedAt: "", content: "", sources: [{ id: "evidence:a", title: "Reference title" }] },
          { id: "ev:b", source: "finra", retrievedAt: "", content: "", sources: [{ id: "evidence:b" }] },
        ]
      },
      { type: "answer_delta", text: "A [ev:a]. B [ev:b]." },
    ];
    const root = await mount(<AgentConsole events={events} reasoning={false} debug={false} />);
    await until(() => requests.length === 2);
    await act(async () => { });
    expect([...root.querySelectorAll('[aria-label="Captured sources"] a')].map((anchor) => anchor.textContent)).toEqual(["Reference title", "FINRA 2"]);
    expect([...root.querySelectorAll('a[aria-label^="Open source:"]')].map((anchor) => [anchor.textContent, anchor.getAttribute("title")])).toEqual([["[1]", "Reference title"], ["[2]", "FINRA 2"]]);
    expect(root.textContent).not.toContain("ev:a");
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
