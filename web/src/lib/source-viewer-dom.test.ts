import { describe, expect, test } from "bun:test";
import type { WindowLike } from "dompurify";
import { JSDOM } from "jsdom";
import fc from "fast-check";
import { activateQuote, highlightMappedQuote, mapSourceHtml, markQuote, resolveQuote, sanitizeSourceHtml, tagFragmentAnchors } from "./source-viewer";
import type { SourceDocument } from "./research-client";
import corpus from "./source-viewer-corpus.json";

function dom(): { window: WindowLike & Window; document: Document } {
  const jsdom = new JSDOM("", { url: "https://example.com/" });
  const window = jsdom.window as unknown as WindowLike & Window;
  return { window, document: jsdom.window.document };
}

describe("source DOM rendering", () => {
  test("sanitized HTML keeps tables and makes links safe", () => {
    const { window, document } = dom();
    const fragment = sanitizeSourceHtml(window, `<table><tr><th scope="col">Year</th><td>2026<script>alert(1)</script><a href="javascript:alert(1)">x</a><a href="https://www.sec.gov/filing">filing</a></td></tr></table><img src="https://example.com/pixel.png" onerror="alert(1)">`);
    tagFragmentAnchors(fragment as unknown as DocumentFragment);
    const root = document.createElement("div");
    root.append(fragment as unknown as Node);
    expect(root.querySelector("table")).not.toBeNull();
    expect(root.querySelector("script")).toBeNull();
    expect(root.querySelector("img")).toBeNull();
    const anchors = [...root.querySelectorAll("a")];
    expect(anchors.map((anchor) => anchor.getAttribute("href"))).toEqual([null, "https://www.sec.gov/filing"]);
    expect(anchors[1].getAttribute("target")).toBe("_blank");
    expect(anchors[1].getAttribute("rel")).toBe("noopener noreferrer");
  });

  test("resolved markers keep original characters and activate one passage", () => {
    const { document } = dom();
    const root = document.createElement("div");
    root.textContent = "Revenue   grew today.";
    const found = resolveQuote(root.textContent ?? "", "Revenue grew");
    expect(found.status).toBe("resolved");
    if (found.status !== "resolved") throw new Error("expected a resolved quote");
    const marks = markQuote(root, found.start, found.end);
    expect(marks.length).toBe(1);
    expect(marks[0].textContent).toBe("Revenue   grew");
    activateQuote(root, marks);
    expect(marks[0].className).toContain("mgl-active-element-highlight");
    expect(root.querySelectorAll("mark").length).toBe(1);
  });

  test("ambiguous quotes leave every occurrence unmarked", () => {
    const { document } = dom();
    const root = document.createElement("div");
    root.textContent = "risk rose. risk rose.";
    expect(resolveQuote(root.textContent ?? "", "risk rose")).toEqual({ status: "ambiguous", count: 2 });
    expect(root.querySelectorAll("mark").length).toBe(0);
  });

  test("sanitized filing markup keeps structure and table spans", () => {
    const { window, document } = dom();
    const html = `<h2>Item 7</h2><p>Revenue <strong>grew</strong> <em>10%</em>.</p><ul><li>Cash</li></ul><blockquote><p>Quoted</p></blockquote><pre><code>x = 1</code></pre><table><caption>Segments</caption><thead><tr><th scope="col" colspan="2">Revenue</th></tr></thead><tbody><tr><td rowspan="2">2026</td><td>10</td></tr></tbody></table>`;
    const root = document.createElement("div");
    root.append(sanitizeSourceHtml(window, html) as unknown as Node);
    expect(root.innerHTML).toBe(html);
  });

  test("sanitizer drops data, aria and style attributes", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    root.append(sanitizeSourceHtml(window, `<p data-cell="1" aria-label="note" style="color:red">text</p>`) as unknown as Node);
    expect(root.innerHTML).toBe("<p>text</p>");
  });

  test("links keep only http(s) schemes and lose credentialed targets", () => {
    const { window } = dom();
    const fragment = sanitizeSourceHtml(window, `<a href="javascript:alert('https://x')">a</a><a href="http://www.sec.gov/x">b</a><a href="https://user:secret@www.sec.gov/x">c</a>`) as unknown as DocumentFragment;
    const anchors = [...fragment.querySelectorAll("a")];
    expect(anchors.slice(0, 2).map((anchor) => anchor.getAttribute("href"))).toEqual([null, "http://www.sec.gov/x"]);
    tagFragmentAnchors(fragment);
    expect(anchors.map((anchor) => anchor.getAttribute("href"))).toEqual([null, "http://www.sec.gov/x", null]);
  });

  test("quotes spanning table cells mark only the quoted cells", () => {
    const { document } = dom();
    const root = document.createElement("div");
    root.innerHTML = "<table><tbody><tr><td>Cash</td><td>100</td><td>Debt</td><td>200</td><td>Net</td></tr></tbody></table>";
    const found = resolveQuote(root.textContent ?? "", "Debt 200");
    if (found.status !== "resolved") throw new Error("expected a resolved quote");
    const marks = markQuote(root, found.start, found.end);
    expect(marks.map((mark) => mark.textContent)).toEqual(["Debt", "200"]);
    expect(marks.map((mark) => mark.className)).toEqual(["mgl-cited-element-highlight", "mgl-cited-element-highlight"]);
    expect(root.textContent).toBe("Cash100Debt200Net");
  });

  test("quotes starting and ending inside text nodes mark exact characters", () => {
    const { document } = dom();
    const root = document.createElement("div");
    root.innerHTML = "<p>Net cash <b>rose</b> sharply today</p>";
    const found = resolveQuote(root.textContent ?? "", "cash rose sharply");
    if (found.status !== "resolved") throw new Error("expected a resolved quote");
    expect(markQuote(root, found.start, found.end).map((mark) => mark.textContent)).toEqual(["cash ", "rose", " sharply"]);
    expect(root.textContent).toBe("Net cash rose sharply today");
  });

  test("activating a passage moves the active state and centers its first mark", () => {
    const { document } = dom();
    const root = document.createElement("div");
    root.textContent = "risk rose. risk rose.";
    const first = markQuote(root, 0, 9);
    const second = markQuote(root, 11, 20);
    const scrolls: unknown[] = [];
    second[0].scrollIntoView = (options?: boolean | ScrollIntoViewOptions) => {
      scrolls.push(options);
    };
    activateQuote(root, first);
    activateQuote(root, second);
    const active = root.querySelectorAll(".mgl-active-element-highlight");
    expect(active.length).toBe(1);
    expect(active[0]).toBe(second[0]);
    expect(first[0].classList.contains("mgl-cited-element-highlight")).toBe(true);
    expect(scrolls).toEqual([{ behavior: "smooth", block: "center" }]);
    activateQuote(root, []);
    expect(root.querySelectorAll(".mgl-active-element-highlight").length).toBe(0);
  });

  test("canonical DOM text reproduces the shared SEC corpus", () => {
    const { window, document } = dom();
    for (const sample of corpus) {
      const root = document.createElement("div");
      const map = mapSourceHtml(root, window, sample.html);
      expect(map?.text).toBe(sample.text);
      expect(root.textContent).not.toContain("\uE000");
      expect(root.textContent).not.toContain("\uE001");
    }
  });

  function capture(html: string, text: string, quote: string, location?: SourceDocument["location"]): SourceDocument {
    return { id: "evidence:sec:mapped", title: "10-K", representation: "html", captureKind: "original", content: html, matchingText: text, quote, location };
  }

  test("synthetic separators highlight source cells without changing their layout", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    const source = capture("<table><tr><td>  Rev<strong>enue</strong> </td><td></td><td> 10 </td><td>20</td></tr></table>", "Revenue | 10 | 20", "Revenue | 10", { offset: 0, end: 12, basis: "rendered" });
    const map = mapSourceHtml(root, window, source.content ?? "");
    const result = highlightMappedQuote(root, map, source);
    expect(result.resolution).toEqual({ status: "resolved", start: 0, end: 12 });
    expect(result.marks.map((mark) => mark.textContent)).toEqual(["Rev", "enue", "10"]);
    expect(result.marks.map((mark) => mark.closest("td")?.cellIndex)).toEqual([0, 0, 2]);
    expect(root.querySelectorAll("tr").length).toBe(1);
    expect(root.querySelectorAll("td").length).toBe(4);
    expect(root.textContent).toBe("  Revenue  10 20");
  });

  test("literal pipes remain source characters rather than synthetic separators", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    const source = capture("<table><tr><td>R|D</td><td>10</td></tr></table>", "R|D | 10", "R|D | 10");
    const result = highlightMappedQuote(root, mapSourceHtml(root, window, source.content ?? ""), source);
    expect(result.marks.map((mark) => mark.textContent)).toEqual(["R|D", "10"]);
    const other = document.createElement("div");
    const missingPipe = capture(source.content ?? "", source.matchingText ?? "", "RD | 10");
    expect(highlightMappedQuote(other, mapSourceHtml(other, window, missingPipe.content ?? ""), missingPipe).resolution).toEqual({ status: "unresolved" });
    expect(other.querySelector("mark")).toBeNull();
  });

  test("full-document Python offsets select the requested section after Unicode", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    const source = capture("<p>😀😀</p><p>x</p><p>x</p>", "😀😀\nx\nx", "x", { offset: 5, end: 6, basis: "rendered" });
    const result = highlightMappedQuote(root, mapSourceHtml(root, window, source.content ?? ""), source);
    expect(result.resolution).toEqual({ status: "resolved", start: 7, end: 8 });
    expect(result.marks.length).toBe(1);
    expect(root.querySelectorAll("p")[1].querySelector("mark")).toBeNull();
    expect(root.querySelectorAll("p")[2].querySelector("mark")?.textContent).toBe("x");
  });

  test("exact context resolves duplicate rows while missing context leaves ambiguity", () => {
    const { window, document } = dom();
    const html = "<h2>Item 1</h2><table><tr><td>Revenue</td><td>10</td></tr></table><h2>Item 7</h2><table><tr><td>Revenue</td><td>10</td></tr></table>";
    const basis = "Item 1\nRevenue | 10\nItem 7\nRevenue | 10";
    const root = document.createElement("div");
    const source = capture(html, basis, "Revenue | 10", { prefix: "Item 7\n" });
    const result = highlightMappedQuote(root, mapSourceHtml(root, window, html), source);
    expect(result.resolution.status).toBe("resolved");
    expect(root.querySelectorAll("table")[0].querySelector("mark")).toBeNull();
    expect(root.querySelectorAll("table")[1].querySelectorAll("mark").length).toBe(2);
    for (const location of [undefined, { prefix: "Item 9\n" }]) {
      const other = document.createElement("div");
      const result = highlightMappedQuote(other, mapSourceHtml(other, window, html), capture(html, basis, "Revenue | 10", location));
      expect(result.resolution).toEqual(location ? { status: "unresolved" } : { status: "ambiguous", count: 2 });
      expect(other.querySelector("mark")).toBeNull();
    }
  });

  test("incorrect or unsupported coordinate bases cannot select another occurrence", () => {
    const { window, document } = dom();
    for (const source of [
      capture("<p>x</p><p>x</p>", "missing\nx\nx", "x", { offset: 0, end: 1, basis: "rendered" }),
      capture("<p>x</p>", "x", "x", { offset: 99, end: 100, basis: "rendered" }),
      capture("<table><td>x</td></table>", "", "x", { offset: 0, basis: "rendered" }),
      capture("<p>\uE000x</p>", "x", "x", { offset: 0, basis: "rendered" }),
    ]) {
      const root = document.createElement("div");
      const result = highlightMappedQuote(root, mapSourceHtml(root, window, source.content ?? ""), source);
      expect(result.resolution).toEqual({ status: "unresolved" });
      expect(root.querySelector("mark")).toBeNull();
    }
  });

  test("namespaced noise remains visible but does not receive quote marks", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    const source = capture("<p>Revenue <ix:note><span>ignored</span></ix:note>10</p>", "Revenue 10", "Revenue 10");
    const result = highlightMappedQuote(root, mapSourceHtml(root, window, source.content ?? ""), source);
    expect(result.marks.map((mark) => mark.textContent)).toEqual(["Revenue ", "10"]);
    expect(root.textContent).toBe("Revenue ignored10");
    expect(root.querySelector("span")?.querySelector("mark")).toBeNull();
  });

  test("quotes in undisplayed title text remain unresolved", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    const source = capture("<html><head><title>Filing</title></head><body><p>Revenue</p></body></html>", "Filing\nRevenue", "Filing");
    expect(highlightMappedQuote(root, mapSourceHtml(root, window, source.content ?? ""), source).resolution).toEqual({ status: "unresolved" });
    expect(root.querySelector("mark")).toBeNull();
  });
  test("Unicode coordinates preserve the selected duplicate table row", () => {
    const { window, document } = dom();
    fc.assert(fc.property(
      fc.stringMatching(/^[a-z]{1,8}$/),
      fc.integer({ min: 1, max: 6 }),
      fc.boolean(),
      (label, astralCount, second) => {
        const head = "😀".repeat(astralCount);
        const quote = `${label} | 10`;
        const html = `<p>${head}</p><table><tr><td>${label}</td><td>10</td></tr><tr><td>${label}</td><td>10</td></tr></table>`;
        const basis = `${head}\n${quote}\n${quote}`;
        const offset = astralCount + 1 + (second ? quote.length + 1 : 0);
        const source = capture(html, basis, quote, { offset, end: offset + quote.length, basis: "rendered" });
        const root = document.createElement("div");
        const result = highlightMappedQuote(root, mapSourceHtml(root, window, html), source);
        expect(result.resolution.status).toBe("resolved");
        expect(result.marks.map((mark) => mark.textContent)).toEqual([label, "10"]);
        expect(root.querySelectorAll("tr")[second ? 0 : 1].querySelector("mark")).toBeNull();
        expect(root.querySelectorAll("tr")[second ? 1 : 0].querySelectorAll("mark").length).toBe(2);
      },
    ), { seed: 20261010, numRuns: 40 });
  });

  test("encoded private characters remain visible when the map rejects their source", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    expect(mapSourceHtml(root, window, "<p>&#xE000;x&#57345;</p>")).toBeUndefined();
    expect(root.textContent).toBe("\uE000x\uE001");
    expect(root.querySelector("mark")).toBeNull();
  });

  test("a quote that is only a synthetic table separator never paints", () => {
    const { window, document } = dom();
    const root = document.createElement("div");
    const source = capture("<table><tr><td>R</td><td>10</td></tr></table>", "R | 10", "|");
    const result = highlightMappedQuote(root, mapSourceHtml(root, window, source.content ?? ""), source);
    expect(result.resolution).toEqual({ status: "unresolved" });
    expect(result.marks).toEqual([]);
    expect(root.querySelector("mark")).toBeNull();
    expect(root.textContent).toBe("R10");
  });

  test("a raw-basis location uses only its context, never mapped coordinates", () => {
    const { window, document } = dom();
    const html = "<h2>Item 1</h2><table><tr><td>Revenue</td><td>10</td></tr></table><h2>Item 7</h2><table><tr><td>Revenue</td><td>10</td></tr></table>";
    const basis = "Item 1\nRevenue | 10\nItem 7\nRevenue | 10";
    const selected = document.createElement("div");
    const byContext = capture(html, basis, "Revenue | 10", { offset: 0, end: 12, basis: "raw", prefix: "Item 7\n" });
    expect(highlightMappedQuote(selected, mapSourceHtml(selected, window, html), byContext).resolution.status).toBe("resolved");
    expect(selected.querySelectorAll("table")[0].querySelector("mark")).toBeNull();
    expect(selected.querySelectorAll("table")[1].querySelectorAll("mark").length).toBe(2);

    const rejected = document.createElement("div");
    const staleCoordinates = capture(html, basis, "Revenue | 10", { offset: 0, end: 12, basis: "raw", prefix: "Item 9\n" });
    const result = highlightMappedQuote(rejected, mapSourceHtml(rejected, window, html), staleCoordinates);
    expect(result.resolution).toEqual({ status: "unresolved" });
    expect(result.marks).toEqual([]);
  });
});
