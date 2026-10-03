import { describe, expect, test } from "bun:test";
import { NeedleRouter, acquireNeedle, validateNeedleTool } from "./client";

describe("acquireNeedle", () => {
  test("concurrent holders serialize", async () => {
    const order: string[] = [];
    const first = await acquireNeedle();
    order.push("A-enter");
    let bEntered = false;
    const bGrant = acquireNeedle();
    const bWait = bGrant.then(async (releaseB) => {
      bEntered = true;
      order.push("B-enter");
      order.push("B-exit");
      releaseB();
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(bEntered).toBe(false);
    order.push("A-exit");
    first();
    await bWait;
    expect(order).toEqual(["A-enter", "A-exit", "B-enter", "B-exit"]);
  });

  test("throw still grants next holder", async () => {
    const first = await acquireNeedle();
    try {
      throw new Error("boom");
    } catch {
      // Fall through; release must still run.
    } finally {
      first();
    }
    const second = await acquireNeedle();
    try {
      expect(true).toBe(true);
    } finally {
      second();
    }
  });
});

describe("validateNeedleTool", () => {
  test("exact match passes; mismatch and null throw", () => {
    expect(validateNeedleTool("search_sec_filings", "search_sec_filings")).toBe("search_sec_filings");
    expect(() => validateNeedleTool("search_sec_filings", "get_sec_document")).toThrow("mismatch");
    expect(() => validateNeedleTool("search_sec_filings", null)).toThrow("mismatch");
    expect(() => validateNeedleTool("", "search_sec_filings")).toThrow("nonempty");
  });
});

describe("timeout isolation", () => {
  test("timeout rejects only the timed-out call while a sibling still resolves", async () => {
    const router = new NeedleRouter();
    let killed = false;
    const fakeChild = { kill: () => { killed = true; }, stdin: { write: () => { } } };
    type RouterSeam = {
      ensure: () => unknown;
      rawCall: (b: Record<string, unknown>, k: string, t: number) => Promise<unknown>;
      onLine: (l: string) => void;
    };
    // Unchecked cast: private members have no public seam; structural read only.
    const seam: RouterSeam = router as unknown as RouterSeam;
    seam.ensure = () => fakeChild;
    const slow = seam.rawCall({ action: "slow" }, "route", 20);
    const sibling = seam.rawCall({ action: "sibling" }, "route", 1000);
    await expect(slow).rejects.toThrow("timeout");
    expect(killed).toBe(false);
    seam.onLine(JSON.stringify({ id: "2", tool: null, arguments: {}, confidence: null, reasoning: "" }));
    await expect(sibling).resolves.toEqual({ tool: null, arguments: {}, confidence: null, reasoning: "" });
    await router.close();
  });
});
