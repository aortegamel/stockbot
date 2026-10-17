import { describe, expect, test } from "bun:test";
import { makeEvidence } from "./evidence";

describe("makeEvidence identity", () => {
  test("same source+handle+content yields same id", () => {
    const a = makeEvidence("s", "hello", { sourceHandle: { a: 1 } });
    const b = makeEvidence("s", "hello", { sourceHandle: { a: 1 } });
    expect(a.id).toBe(b.id);
  });

  test("different handle yields different id", () => {
    const a = makeEvidence("s", "hello", { sourceHandle: { a: 1 } });
    const b = makeEvidence("s", "hello", { sourceHandle: { a: 2 } });
    expect(a.id).not.toBe(b.id);
  });

  test("contents differing only after char 8000 yield different ids and keep 8000-char budget plus marker", () => {
    const head = "x".repeat(8000);
    const a = makeEvidence("s", `${head}a`, { sourceHandle: { a: 1 } });
    const b = makeEvidence("s", `${head}b`, { sourceHandle: { a: 1 } });
    expect(a.id).not.toBe(b.id);
    expect(a.content.startsWith(head)).toBe(true);
    expect(a.content.endsWith("…[truncated]")).toBe(true);
    const c = makeEvidence("s", "x".repeat(9000), { sourceHandle: { a: 1 } });
    expect(c.content.startsWith("x".repeat(8000))).toBe(true);
    expect(c.content.endsWith("…[truncated]")).toBe(true);
  });

  test("content at or under budget has no truncation marker", () => {
    expect(makeEvidence("s", "short").content).toBe("short");
    expect(makeEvidence("s", "x".repeat(8000)).content).toBe("x".repeat(8000));
  });

  test("handle key order never changes the id, and array order always does", () => {
    const nested = makeEvidence("s", "hello", { sourceHandle: { b: 2, a: { d: 4, c: [1, 2] } } });
    const reordered = makeEvidence("s", "hello", { sourceHandle: { a: { c: [1, 2], d: 4 }, b: 2 } });
    expect(nested.id).toBe(reordered.id);
    expect(makeEvidence("s", "hello", { sourceHandle: { ids: [1, 2] } }).id).not.toBe(
      makeEvidence("s", "hello", { sourceHandle: { ids: [2, 1] } }).id,
    );
  });

  test("an absent, empty, or sourceRefs-only capture stays a distinct identity", () => {
    const absent = makeEvidence("s", "hello");
    const empty = makeEvidence("s", "hello", { sourceHandle: {} });
    const refs = makeEvidence("s", "hello", { sourceRefs: { path: "/raw/x" } });
    expect(empty.id).not.toBe(absent.id);
    expect(refs.id).not.toBe(absent.id);
    expect(makeEvidence("s", "hello", { sourceRefs: { path: "/raw/x" } }).id).toBe(refs.id);
    expect(makeEvidence("s", "hello", { sourceHandle: { path: "/raw/x" }, sourceRefs: { other: true } }).id).toBe(refs.id);
    expect(refs.sourceRefs).toEqual({ path: "/raw/x" });
    expect(refs.sourceHandle).toBeUndefined();
  });

  test("an undefined handle value normalizes like a null one", () => {
    expect(makeEvidence("s", "hello", { sourceHandle: { a: undefined } }).id).toBe(
      makeEvidence("s", "hello", { sourceHandle: { a: null } }).id,
    );
  });

  test("title and url ride the evidence and retrievedAt is an ISO timestamp", () => {
    const e = makeEvidence("s", "hello", { title: "Filing", url: "https://www.sec.gov/filing" });
    expect(e.title).toBe("Filing");
    expect(e.url).toBe("https://www.sec.gov/filing");
    expect(e.retrievedAt).toBe(new Date(e.retrievedAt).toISOString());
  });
});
