import { describe, expect, test } from "bun:test";
import fc from "fast-check";
import { parseTradingViewLink, snapshotImage } from "./tradingview";

describe("parseTradingViewLink", () => {
  test("parses each supported link kind into validated fields", () => {
    expect(parseTradingViewLink("https://www.tradingview.com/x/8ot166mV/")).toEqual({ kind: "snapshot", id: "8ot166mV" });
    expect(parseTradingViewLink(" https://tradingview.com/x/JWLyJ1KN ")).toEqual({ kind: "snapshot", id: "JWLyJ1KN" });
    expect(parseTradingViewLink("https://www.tradingview.com/symbols/NASDAQ-NVDA/")).toEqual({ kind: "symbol", symbol: "NASDAQ:NVDA" });
    expect(parseTradingViewLink("https://www.tradingview.com/symbols/nvda/financials-overview/")).toEqual({ kind: "symbol", symbol: "NVDA" });
    expect(parseTradingViewLink("https://www.tradingview.com/symbols/NYSE-BRK.B/")).toEqual({ kind: "symbol", symbol: "NYSE:BRK.B" });
    expect(parseTradingViewLink("https://www.tradingview.com/chart/?symbol=NASDAQ%3ANVDA&interval=60")).toEqual({ kind: "chart-params", symbol: "NASDAQ:NVDA", interval: "1h" });
    expect(parseTradingViewLink("https://www.tradingview.com/chart/?symbol=CME_MINI:ES1!&interval=1d")).toEqual({ kind: "chart-params", symbol: "CME_MINI:ES1!", interval: "D" });
    expect(parseTradingViewLink("https://www.tradingview.com/chart/?symbol=NVDA&interval=3")).toEqual({ kind: "chart-params", symbol: "NVDA" });
    expect(parseTradingViewLink("https://www.tradingview.com/chart/AUDUSD/gRwCMwkV-AUDUSD-is-Nearing-a-Strong-Resistance-Area/")).toEqual({ kind: "idea", id: "gRwCMwkV" });
    expect(parseTradingViewLink("https://www.tradingview.com/chart/AbCd1234/")).toEqual({ kind: "layout", id: "AbCd1234" });
    expect(parseTradingViewLink("https://www.tradingview.com/chart/AbCd1234/?symbol=NVDA")).toEqual({ kind: "layout", id: "AbCd1234" });
  });

  test("refuses non-TradingView, non-https, script, and malformed links", () => {
    for (const raw of [
      "",
      "javascript:alert(1)",
      "JAVASCRIPT://www.tradingview.com/x/8ot166mV/",
      "data:text/html,<script>alert(1)</script>",
      "http://www.tradingview.com/x/8ot166mV/",
      "https://evil.example/x/8ot166mV/",
      "https://tradingview.com.evil.example/x/8ot166mV/",
      "https://s.tradingview.com/widgetembed/?symbol=NVDA",
      "https://constructor/x/8ot166mV/",
      "https://user:pw@www.tradingview.com/x/8ot166mV/",
      "https://www.tradingview.com:8443/x/8ot166mV/",
      "https://www.tradingview.com/x/short/",
      "https://www.tradingview.com/x/8ot166mV/extra/",
      "https://www.tradingview.com/x/%E0%A4%A/",
      "https://www.tradingview.com/symbols/NVDA%22onload%3D/",
      "https://www.tradingview.com/chart/",
      "https://www.tradingview.com/chart/?symbol=NVDA%26x%3D1",
      "https://www.tradingview.com/chart/NVDA/not-an-idea/",
      "https://www.tradingview.com/ideas/",
      `https://www.tradingview.com/x/8ot166mV/?${"a".repeat(300)}`,
    ]) expect(parseTradingViewLink(raw)).toBeNull();
  });

  test("any accepted link is https on a TradingView host and yields only safe fields", () => {
    const segment = fc.oneof(fc.constantFrom("x", "symbols", "chart", "8ot166mV", "gRwCMwkV-slug", "NASDAQ-NVDA"), fc.stringMatching(/^[A-Za-z0-9%"<>:!._-]{0,12}$/));
    const raw = fc.tuple(
      fc.constantFrom("https", "http", "javascript", "data"),
      fc.constantFrom("www.tradingview.com", "tradingview.com", "evil.example", "tradingview.com.evil.example", "s.tradingview.com"),
      fc.array(segment, { maxLength: 4 }),
      fc.constantFrom("", "?symbol=NASDAQ%3ANVDA&interval=D", "?symbol=%22%3E", "#frag"),
    ).map(([scheme, host, path, query]) => `${scheme}://${host}/${path.join("/")}${query}`);
    fc.assert(fc.property(raw, (text) => {
      const link = parseTradingViewLink(text);
      if (link === null) return;
      const url = new URL(text);
      expect(url.protocol).toBe("https:");
      expect(["tradingview.com", "www.tradingview.com"]).toContain(url.hostname);
      for (const value of Object.values(link)) expect(value).toMatch(/^[A-Za-z0-9_.!:-]+$/);
    }), { seed: 20261020, numRuns: 1000 });
  });

  test("snapshot images live under the lowercased first ID character", () => {
    expect(snapshotImage("JWLyJ1KN")).toBe("https://s3.tradingview.com/snapshots/j/JWLyJ1KN.png");
    expect(snapshotImage("8ot166mV")).toBe("https://s3.tradingview.com/snapshots/8/8ot166mV.png");
  });
});
