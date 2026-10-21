import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { originalSourceUrl, resolveQuote } from "./source-viewer";
import { safeSourceUrl } from "../../../needle-harness/lib/agent/types";
import type { SourceDocument } from "./research-client";

function document(overrides: Partial<SourceDocument> = {}): SourceDocument {
  return {
    id: "evidence:sec:10-k",
    title: "10-K",
    url: "https://www.sec.gov/Archives/edgar/data/1/0001-10-k.htm",
    representation: "html",
    captureKind: "original",
    content: "<p>Revenue grew.</p>",
    quote: "Revenue grew.",
    ...overrides,
  };
}

describe("resolveQuote", () => {
  test("unique whitespace-insensitive match resolves at original offsets", () => {
    expect(resolveQuote("Revenue   grew\ntoday.", "Revenue grew")).toEqual({ status: "resolved", start: 0, end: 14 });
  });

  test("wrong coordinates do not force an arbitrary match", () => {
    expect(resolveQuote("Revenue grew today.", "Revenue grew", { offset: 8, basis: "raw" })).toEqual({ status: "resolved", start: 0, end: 12 });
  });

  test("context that matches nothing stays unresolved", () => {
    expect(resolveQuote("Revenue grew today.", "Revenue grew", { prefix: "Expenses fell" })).toEqual({ status: "unresolved" });
  });

  test("prefix selects the intended duplicate", () => {
    const text = "Sales rose twice. Sales rose again after the audit.";
    const quote = "Sales rose";
    const prefix = "Sales rose twice.";
    expect(resolveQuote(text, quote, { prefix })).toEqual({ status: "resolved", start: 18, end: 28 });
  });

  test("duplicate without context stays ambiguous", () => {
    expect(resolveQuote("risk rose. risk rose.", "risk rose")).toEqual({ status: "ambiguous", count: 2 });
  });

  test("reproducing coordinates select a duplicate and win over context", () => {
    const text = "A: risk rose. B: risk rose.";
    expect(resolveQuote(text, "risk rose", { offset: 17, basis: "raw" })).toEqual({ status: "resolved", start: 17, end: 26 });
    expect(resolveQuote(text, "risk rose", { offset: 3, prefix: "B:", basis: "raw" })).toEqual({ status: "resolved", start: 3, end: 12 });
  });

  test("coordinates need both bounds to reproduce", () => {
    const text = "A: risk rose. B: risk rose.";
    expect(resolveQuote(text, "risk rose", { offset: 17, end: 26, basis: "raw" })).toEqual({ status: "resolved", start: 17, end: 26 });
    expect(resolveQuote(text, "risk rose", { offset: 3, end: 11, basis: "raw" })).toEqual({ status: "ambiguous", count: 2 });
  });

  test("coordinates of any repeated occurrence resolve to that occurrence", () => {
    const filler = fc.stringMatching(/^[0-9 ]{0,6}$/);
    fc.assert(
      fc.property(fc.stringMatching(/^[a-z]{1,8}$/), filler, filler, filler, fc.boolean(), fc.boolean(), (quote, head, middle, tail, second, withEnd) => {
        const text = `${head}${quote}${middle}${quote}${tail}`;
        const start = second ? head.length + quote.length + middle.length : head.length;
        const end = start + quote.length;
        expect(resolveQuote(text, quote, { offset: start, end: withEnd ? end : undefined, basis: "raw" })).toEqual({ status: "resolved", start, end });
      }),
      { seed: 20261010, numRuns: 100 },
    );
  });

  test("prefix must end right before the occurrence", () => {
    expect(resolveQuote("Q1 note: risk rose. Q2 note: risk rose.", "risk rose", { prefix: "Q2 note:" })).toEqual({ status: "resolved", start: 29, end: 38 });
  });

  test("suffix alone selects the occurrence it follows", () => {
    expect(resolveQuote("risk rose before audit. risk rose after audit.", "risk rose", { suffix: "after audit" })).toEqual({ status: "resolved", start: 24, end: 33 });
  });

  test("context shared by several occurrences stays ambiguous", () => {
    expect(resolveQuote("Note: risk rose. Note: risk rose. Later: risk rose.", "risk rose", { prefix: "Note:" })).toEqual({ status: "ambiguous", count: 2 });
  });

  test("absent quote stays unresolved", () => {
    expect(resolveQuote("Revenue grew.", "Expenses fell.")).toEqual({ status: "unresolved" });
  });

  test("table cells without spacing still resolve", () => {
    expect(resolveQuote("Cash100Debt200", "Debt 200")).toEqual({ status: "resolved", start: 7, end: 14 });
  });

  test("whitespace-only quotes never match", () => {
    expect(resolveQuote("anything", "   ")).toEqual({ status: "unresolved" });
  });

  test("matching never invents fuzzy matches", () => {
    fc.assert(
      fc.property(fc.string({ minLength: 1, maxLength: 40 }), fc.string({ minLength: 1, maxLength: 40 }), (text, quote) => {
        const found = resolveQuote(text, quote);
        const stripped = quote.replace(/\s/gu, "");
        if (!stripped) {
          expect(found).toEqual({ status: "unresolved" });
        } else if (found.status === "resolved") {
          expect(text.slice(found.start, found.end).replace(/\s/gu, "")).toBe(stripped);
        } else if (found.status === "unresolved") {
          expect(text.replace(/\s/gu, "")).not.toContain(stripped);
        }
      }),
      { seed: 20261010, numRuns: 100 },
    );
  });
});

describe("source URLs", () => {
  test("non-http links are dropped", () => {
    expect(safeSourceUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeSourceUrl(undefined)).toBeUndefined();
  });

  test("resolved HTML quotes add a text fragment inside Open original", () => {
    const url = originalSourceUrl(document({ quote: "Revenue grew" }), true);
    expect(url).toBe(`${document().url}#:~:text=Revenue%20grew`);
    expect(url?.startsWith("https://www.sec.gov/")).toBe(true);
  });

  test("unresolved quotes keep a plain safe link", () => {
    const source = document({ quote: "Revenue grew" });
    const url = originalSourceUrl(source, false);
    expect(url).toBe(source.url);
    expect(url).not.toContain(":~:text=");
  });

  test("only http and https URLs without credentials are safe", () => {
    expect(safeSourceUrl("http://www.sec.gov/filing")).toBe("http://www.sec.gov/filing");
    expect(safeSourceUrl("ftp://www.sec.gov/filing")).toBeUndefined();
    expect(safeSourceUrl("https://user:secret@www.sec.gov/filing")).toBeUndefined();
  });

  test("password-only credentials make a URL unsafe", () => {
    expect(safeSourceUrl("https://:secret@example.com/")).toBeUndefined();
  });

  test("text fragments collapse single-line whitespace and preserve an existing anchor", () => {
    const source = document({ quote: "  Revenue \t  grew  " });
    expect(originalSourceUrl(source, true)).toBe(`${source.url}#:~:text=Revenue%20grew`);
    const anchored = document({ url: "https://www.sec.gov/a.htm#item7", quote: "Revenue grew" });
    expect(originalSourceUrl(anchored, true)).toBe("https://www.sec.gov/a.htm#item7:~:text=Revenue%20grew");
  });

  test("multiline quotes select the first and last five words from nonempty lines", () => {
    const source = document({ quote: "\n \r\nYear-over-year revenue, profit & cash rose sharply.\r\n\nCosts decreased.\nThe board expects steady growth, margins & buy-backs.\n " });
    expect(originalSourceUrl(source, true)).toBe(`${source.url}#:~:text=Year%2Dover%2Dyear%20revenue%2C%20profit%20%26%20cash,steady%20growth%2C%20margins%20%26%20buy%2Dbacks.`);
  });

  test("short multiline quotes still use a range", () => {
    const source = document({ quote: "Revenue\n grew" });
    expect(originalSourceUrl(source, true)).toBe(`${source.url}#:~:text=Revenue,grew`);
  });

  test("single-line quotes above ten words use a range only above 80 normalized characters", () => {
    const quote = `Alpha beta gamma delta ${"x".repeat(25)} epsilon zeta eta theta nu omega`;
    expect(quote.length).toBe(80);
    expect(originalSourceUrl(document({ quote }), true)).toBe(`${document().url}#:~:text=${encodeURIComponent(quote)}`);
    const long = `${quote}!`;
    expect(originalSourceUrl(document({ quote: long }), true)).toBe(`${document().url}#:~:text=Alpha%20beta%20gamma%20delta%20${"x".repeat(25)},zeta%20eta%20theta%20nu%20omega!`);
    expect(originalSourceUrl(document({ quote: quote.replaceAll(" ", "   ") }), true)).toBe(`${document().url}#:~:text=${encodeURIComponent(quote)}`);
  });

  test.each([1, 5, 9, 10])("single-line quotes with %i long words use one full encoded fragment", (wordCount) => {
    const quote = Array(wordCount).fill("long-word,&".repeat(9)).join(" ");
    expect(quote.length).toBeGreaterThan(80);
    const expected = `${document().url}#:~:text=${encodeURIComponent(quote).replace(/-/gu, "%2D")}`;
    expect(originalSourceUrl(document({ quote }), true)).toBe(expected);
    expect(originalSourceUrl(document({ quote: `  ${quote.replaceAll(" ", " \t ")}  ` }), true)).toBe(expected);
  });

  test("fragment components encode directive separators inside a short quote", () => {
    const source = document({ quote: "Profit-margin, R&D" });
    expect(originalSourceUrl(source, true)).toBe(`${source.url}#:~:text=Profit%2Dmargin%2C%20R%26D`);
  });

  test("range components preserve words without creating extra fragment directives", () => {
    const words = fc.array(fc.constantFrom("net-income", "R&D", "revenue,", "margin", "growth"), { minLength: 1, maxLength: 10 });
    fc.assert(fc.property(words, words, (first, last) => {
      const quote = `${first.join(" ")}\n${last.join(" ")}`;
      const directive = originalSourceUrl(document({ quote }), true)?.split(":~:text=")[1] ?? "";
      const components = directive.split(",");
      expect(components).toHaveLength(2);
      expect(components.every((component) => !/[-&]/u.test(component))).toBe(true);
      expect(components.map(decodeURIComponent)).toEqual([first.slice(0, 5).join(" "), last.slice(-5).join(" ")]);
    }), { seed: 20261018, numRuns: 100 });
  });

  test("text fragments apply only to safe HTML sources with a quote", () => {
    const pdf = document({ representation: "pdf", quote: "Revenue grew" });
    expect(originalSourceUrl(pdf, true)).toBe(pdf.url);
    const blank = document({ quote: "   " });
    expect(originalSourceUrl(blank, true)).toBe(blank.url);
    expect(originalSourceUrl(document({ url: "javascript:alert(1)" }), true)).toBeUndefined();
  });

  test("unparseable, hostless, and non-web URLs are dropped", () => {
    for (const value of ["https://", "not a url at all", "//www.sec.gov/filing", "file:///etc/passwd"]) {
      expect(safeSourceUrl(value)).toBeUndefined();
    }
    expect(safeSourceUrl("")).toBeUndefined();
    expect(safeSourceUrl("https://www.sec.gov:8443/filing")).toBe("https://www.sec.gov:8443/filing");
    // A third slash parses "filing" as the host, so the URL survives and normalizes.
    expect(safeSourceUrl("https:///filing")).toBe("https://filing/");
  });
});
