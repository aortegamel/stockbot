import { describe, expect, test } from "bun:test";
import { rmSync, writeFileSync } from "node:fs";
import { fingerprintKey, loadDotenvAuthoritative, parseDotenv, redactKey } from "./env";

describe("parseDotenv", () => {
  test("quoted, unquoted, export, comments", () => {
    const p = parseDotenv('# lead\nA="x # y"\nB=\'z\'\nexport C=plain\nD=bare # tail\nE=\n');
    expect(p).toEqual({ A: "x # y", B: "z", C: "plain", D: "bare", E: "" });
  });

  test("a bare name never becomes a set variable", () => {
    // Only KEY=value lines count: a name alone (with or without export) sets nothing.
    const p = parseDotenv("BARE_NAME\nEXPORTED_NAME\nlowercase_name\nKEY=value\n");
    expect(p).toEqual({ KEY: "value" });
    expect("BARE_NAME" in p).toBe(false);
    expect("EXPORTED_NAME" in p).toBe(false);
  });

  test("preserves quoted content and truncates only unquoted whitespace comments", () => {
    expect(parseDotenv([
      'DOUBLE="value # kept" ignored',
      "SINGLE='value # kept' # ignored",
      'OPEN_DOUBLE="unfinished # kept',
      "OPEN_SINGLE='unfinished # kept",
      'EMPTY_DOUBLE=""',
      "EMPTY_SINGLE=''",
      "HASH=part#kept",
      "COMMENT=part \t # ignored",
      "WHITESPACE=   ",
      " export \t SPACED = value \r",
      "DUPLICATE=first",
      "DUPLICATE=last",
      "1INVALID=ignored",
      "#COMMENT=ignored",
    ].join("\n"))).toEqual({
      DOUBLE: "value # kept",
      SINGLE: "value # kept",
      OPEN_DOUBLE: "unfinished # kept",
      OPEN_SINGLE: "unfinished # kept",
      EMPTY_DOUBLE: "",
      EMPTY_SINGLE: "",
      HASH: "part#kept",
      COMMENT: "part",
      WHITESPACE: "",
      SPACED: "value",
      DUPLICATE: "last",
    });
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

  test("clears inherited credentials but preserves an inherited PORT", () => {
    const f = `/tmp/stockbot-env-boundaries-${process.pid}.env`;
    const keys = ["STOCKBOT_ENVTEST_EMPTY", "STOCKBOT_ENVTEST_NEW", "STOCKBOT_ENVTEST_SAME", "STOCKBOT_ENVTEST_QUOTED", "PORT"];
    const previous = keys.map((key) => [key, process.env[key]] as const);
    writeFileSync(f, [
      "STOCKBOT_ENVTEST_EMPTY=",
      "STOCKBOT_ENVTEST_NEW='new # credential'",
      "STOCKBOT_ENVTEST_SAME=same",
      'STOCKBOT_ENVTEST_QUOTED="replacement # credential"',
      "PORT=3100",
    ].join("\n"));
    process.env.STOCKBOT_ENVTEST_EMPTY = "old-credential";
    delete process.env.STOCKBOT_ENVTEST_NEW;
    process.env.STOCKBOT_ENVTEST_SAME = "same";
    process.env.STOCKBOT_ENVTEST_QUOTED = "old";
    process.env.PORT = "4100";
    try {
      expect(loadDotenvAuthoritative(f)).toEqual({
        loaded: true,
        overridden: [
          { key: "STOCKBOT_ENVTEST_EMPTY", oldRedacted: "old-credent…", newRedacted: "…" },
          { key: "STOCKBOT_ENVTEST_QUOTED", oldRedacted: "old…", newRedacted: "replacement…" },
        ],
      });
      // A snapshot sidesteps the undefined narrowing that a preceding delete leaves on process.env.
      const loaded = { ...process.env };
      expect(loaded.STOCKBOT_ENVTEST_EMPTY).toBe("");
      expect(loaded.STOCKBOT_ENVTEST_NEW).toBe("new # credential");
      expect(loaded.STOCKBOT_ENVTEST_SAME).toBe("same");
      expect(loaded.STOCKBOT_ENVTEST_QUOTED).toBe("replacement # credential");
      expect(loaded.PORT).toBe("4100");
    } finally {
      for (const [key, value] of previous) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
      rmSync(f);
    }
  });

  test("loads PORT from the file when no inherited PORT exists", () => {
    const f = `/tmp/stockbot-env-port-${process.pid}.env`;
    const previous = process.env.PORT;
    writeFileSync(f, "PORT=3100\n");
    delete process.env.PORT;
    try {
      expect(loadDotenvAuthoritative(f)).toEqual({ loaded: true, overridden: [] });
      expect({ ...process.env }.PORT).toBe("3100");
    } finally {
      if (previous === undefined) delete process.env.PORT;
      else process.env.PORT = previous;
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
