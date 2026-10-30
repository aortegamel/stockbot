import { describe, expect, test } from "bun:test";
import { parseRoute, routeHash } from "./route";

describe("hash route", () => {
  test("parses each page and normalizes the workspace ticker", () => {
    expect(parseRoute("#/feed")).toEqual({ page: "feed" });
    expect(parseRoute("#/harness")).toEqual({ page: "harness" });
    expect(parseRoute("#/workspace/brk.b")).toEqual({ page: "workspace", sym: "BRK.B" });
    expect(parseRoute("#/source/evidence%3Ars%3A1%3Aev%3A2")).toEqual({ page: "source", id: "evidence:rs:1:ev:2" });
    expect(parseRoute(`#/source/capture:${"a".repeat(64)}`)).toEqual({ page: "source", id: `capture:${"a".repeat(64)}` });
  });

  test("rejects unknown pages, extra segments, and invalid tickers", () => {
    for (const hash of ["", "#", "#/", "#/nope", "#/home", "#/home/x", "#/workspace", "#/workspace/", "#/workspace/%3Cscript%3E", "#/workspace/%E0", "#/workspace/NVDA/x", "#/source", "#/source/", "#/source/ev:1", "#/source/evidence%3Aa%2Fb", "#/source/%E0", `#/source/capture:${"A".repeat(64)}`, "#/source/evidence:x/y"]) {
      expect(parseRoute(hash)).toBeNull();
    }
  });

  test("round-trips through routeHash", () => {
    for (const route of [{ page: "feed" }, { page: "harness" }, { page: "workspace", sym: "BRK-B" }, { page: "source", id: "evidence:rs:1:ev:2" }] as const) {
      expect(parseRoute(routeHash(route))).toEqual(route);
    }
  });
});
