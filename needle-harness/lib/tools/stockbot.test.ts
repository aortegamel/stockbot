import { afterAll, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { categorizeFailure, closeBridge, endSession, invoke, newSessionId } from "./stockbot";
import { harvest } from "../muse/client";
import fc from "fast-check";
import { sourceReferences } from "../agent/types";

const ROOT = process.cwd().endsWith("needle-harness")
  ? process.cwd().replace(/\/needle-harness$/, "")
  : process.cwd();
const HAS_BRIDGE = existsSync(`${ROOT}/venv/bin/python`) && existsSync(`${ROOT}/scripts/tool_bridge.py`);

describe.skipIf(!HAS_BRIDGE)("stockbot bridge", () => {
  const sid = newSessionId();

  // Test-only cleanup; harness runtime never calls closeBridge.
  afterAll(async () => {
    await endSession(sid);
    closeBridge();
  });

  test(
    "invalid accession fails closed with accession error",
    async () => {
      const result = await invoke("get_sec_document", { accession_no: "bad" }, sid);
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.error).toMatch(/accession/i);
    },
    60_000,
  );
});

describe("categorizeFailure precedence", () => {
  test("timeout beats duplicate/budget/provider traps", () => {
    expect(categorizeFailure("request timed out; duplicate action suspected").category).toBe("timeout");
    expect(categorizeFailure("timeout waiting on provider budget check").category).toBe("timeout");
  });

  test("deadline beats duplicate/budget/provider traps", () => {
    expect(categorizeFailure("deadline exceeded; duplicate budget recheck from provider").category).toBe(
      "deadline_exceeded",
    );
  });

  test("substrates alone still categorize", () => {
    expect(categorizeFailure("duplicate research action").category).toBe("duplicate_research_action");
    expect(categorizeFailure("provider unavailable").category).toBe("provider_error");
    expect(categorizeFailure("run budget exceeded").category).toBe("tool_budget_exhausted");
  });
});

describe("harvest single-append", () => {
  test("chunk matching both delta paths appends once", () => {
    const acc = { text: "", usage: {} };
    const seen: string[] = [];
    harvest({ type: "response.delta", delta: "hi", choices: [{ delta: { content: "hi" } }] }, acc, (t) => seen.push(t));
    expect(acc.text).toBe("hi");
    expect(seen).toEqual(["hi"]);
  });

  test("usage accumulates by max across chunks", () => {
    const acc = { text: "", usage: {} };
    const noop = () => { };
    harvest({ usage: { input_tokens: 100, output_tokens: 10 } }, acc, noop);
    harvest({ usage: { input_tokens: 50, output_tokens: 40 } }, acc, noop);
    expect(acc.usage).toMatchObject({ inputTokens: 100, outputTokens: 40 });
  });
});

describe("server source metadata", () => {
  const capture = { id: `capture:${"a".repeat(64)}`, title: "Filing", url: "https://www.sec.gov/filing" };

  test("valid references retain only public fields and keep source order", () => {
    expect(sourceReferences([
      { ...capture, raw_archive_path: "/private/archive" },
      capture,
      { id: "evidence:rs:test:ev:1", title: 42, url: "javascript:alert(1)" },
    ])).toEqual([capture, { id: "evidence:rs:test:ev:1" }]);
  });

  test("malformed metadata never creates a navigable id", () => {
    for (const raw of [undefined, null, "capture", {}, 3]) expect(sourceReferences(raw)).toEqual([]);
    expect(sourceReferences([
      { id: "evidence:foo\n" }, { id: "evidence:foo\r" },
      { id: "/private/raw" }, { id: "https://www.sec.gov/filing" },
      { id: "evidence:../raw" }, { id: "evidence:foo%2fbar" },
      { id: `capture:${"A".repeat(64)}` }, { id: `capture:${"a".repeat(63)}` },
      { id: `evidence:${"a".repeat(257)}` },
    ])).toEqual([]);
    expect(sourceReferences([{ id: `evidence:${"a".repeat(256)}` }])).toHaveLength(1);
  });

  test("JSON metadata preserves only supplied valid ids and normalizes idempotently", () => {
    fc.assert(fc.property(fc.jsonValue(), (raw) => {
      const parsed = sourceReferences(raw);
      expect(sourceReferences(parsed)).toEqual(parsed);
      expect(new Set(parsed.map((r) => r.id)).size).toBe(parsed.length);
      const supplied = Array.isArray(raw) ? raw : [];
      for (const ref of parsed) {
        expect(supplied.some((r) => r !== null && typeof r === "object" && "id" in r && r.id === ref.id)).toBe(true);
        expect(ref.id).toMatch(/^(?:capture:[a-f0-9]{64}|evidence:[A-Za-z0-9:_-]{1,256})$/);
      }
    }), { seed: 20261010, numRuns: 100 });
  });
});
