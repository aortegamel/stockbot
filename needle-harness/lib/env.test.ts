import { describe, expect, test } from "bun:test";
import { rmSync, writeFileSync } from "node:fs";
import { fingerprintKey, loadDotenvAuthoritative, parseDotenv, redactKey } from "./env";

describe("parseDotenv", () => {
  test("quoted, unquoted, export, comments", () => {
    const p = parseDotenv('# lead\nA="x # y"\nB=\'z\'\nexport C=plain\nD=bare # tail\nE=\n');
    expect(p).toEqual({ A: "x # y", B: "z", C: "plain", D: "bare", E: "" });
  });
});

describe("loadDotenvAuthoritative", () => {
  test("file wins over inherited exports", () => {
    const f = `/tmp/stockbot-env-${process.pid}.env`;
    writeFileSync(f, 'STOCKBOT_ENVTEST_A="from-file"\n');
    const OLD = "__old__";
    process.env.STOCKBOT_ENVTEST_A = OLD;
    try {
      const { loaded, overridden } = loadDotenvAuthoritative(f);
      expect(loaded).toBe(true);
      expect(process.env.STOCKBOT_ENVTEST_A).toBe("from-file");
      expect(overridden.some((o) => o.key === "STOCKBOT_ENVTEST_A")).toBe(true);
    } finally {
      delete process.env.STOCKBOT_ENVTEST_A;
      rmSync(f);
    }
  });

  test("missing file reports unloaded", () => {
    expect(loadDotenvAuthoritative("/tmp/stockbot-env-missing.env").loaded).toBe(false);
  });
});

describe("key identity helpers", () => {
  test("fingerprint separates keys, redact hides tail", () => {
    expect(fingerprintKey("oc_sk_74aaa")).not.toBe(fingerprintKey("oc_sk_f49aaa"));
    expect(redactKey("oc_sk_74abcdef")).toBe("oc_sk_74abc…");
    expect(redactKey("oc_sk_74abcdef")).not.toContain("abcdef");
  });
});
