import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { originalLink, resolveQuote } from "./source-viewer";
import { safeSourceUrl } from "../../../needle-harness/lib/agent/types";

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

  test("only http and https URLs without credentials are safe", () => {
    expect(safeSourceUrl("http://www.sec.gov/filing")).toBe("http://www.sec.gov/filing");
    expect(safeSourceUrl("ftp://www.sec.gov/filing")).toBeUndefined();
    expect(safeSourceUrl("https://user:secret@www.sec.gov/filing")).toBeUndefined();
  });

  test("password-only credentials make a URL unsafe", () => {
    expect(safeSourceUrl("https://:secret@example.com/")).toBeUndefined();
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

describe("originalLink", () => {
  const page = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000052/nvda-20260426.htm";
  const highlight = `${page}#f-30:~:text=Revenue,81%2C615`;

  test("a highlighted original links the backend-verified fragment", () => {
    expect(originalLink({ linkStatus: "highlighted_original", originalUrl: page, highlightUrl: highlight })).toEqual({ href: highlight, host: "SEC.gov", highlighted: true });
  });

  test("a Stockbot render links the plain official page without a fragment", () => {
    const link = originalLink({ linkStatus: "stockbot_render", originalUrl: page });
    expect(link).toEqual({ href: page, host: "SEC.gov", highlighted: false });
    expect(link?.href).not.toContain("#:~:");
  });

  test("a missing official page has no link", () => {
    expect(originalLink({ linkStatus: "missing" })).toBeUndefined();
  });

  test("hosts read as SEC.gov, FINRA.org, or the hostname", () => {
    const host = (url: string) => originalLink({ linkStatus: "stockbot_render", originalUrl: url })?.host;
    expect(host("https://cdn.finra.org/equity/regsho/daily/CNMSshvol20260102.txt")).toBe("FINRA.org");
    expect(host("https://finra.org/rules")).toBe("FINRA.org");
    expect(host("https://news.example/article")).toBe("news.example");
  });
});
