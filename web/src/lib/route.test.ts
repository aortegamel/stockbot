import { describe, expect, test } from "bun:test";
import { parseRoute, routeHash } from "./route";

describe("hash route", () => {
  test("parses each page and normalizes the workspace ticker", () => {
    expect(parseRoute("#/home")).toEqual({ page: "home" });
    expect(parseRoute("#/feed")).toEqual({ page: "feed" });
    expect(parseRoute("#/harness")).toEqual({ page: "harness" });
    expect(parseRoute("#/workspace/brk.b")).toEqual({ page: "workspace", sym: "BRK.B" });
  });

  test("rejects unknown pages, extra segments, and invalid tickers", () => {
    for (const hash of ["", "#", "#/", "#/nope", "#/home/x", "#/workspace", "#/workspace/", "#/workspace/%3Cscript%3E", "#/workspace/%E0", "#/workspace/NVDA/x"]) {
      expect(parseRoute(hash)).toBeNull();
    }
  });

  test("round-trips through routeHash", () => {
    for (const route of [{ page: "home" }, { page: "feed" }, { page: "harness" }, { page: "workspace", sym: "BRK-B" }] as const) {
      expect(parseRoute(routeHash(route))).toEqual(route);
    }
  });
});
