import { describe, expect, jest, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import fc from "fast-check";
import { alignClaimText, asksWatchlist, datapointRows, dropUncovered, inventoryDefect, KernelRouter, outOfScopeAnswer, parseTicker, parseWatchlist, readableNumbers, REQUEST_WALL_MS, runKernelAgent, type KernelChild, type KernelReasonInput, type KernelReasonResult, type KernelSpawn } from "./kernel";
import { readDatapoints, type AgentEvent, type AskContext, type Datapoint, type Evidence, type FinalClaim, type Metrics } from "./types";

class FakeChild implements KernelChild {
  exitCode: number | null = null;
  killCalls = 0;
  written: string[] = [];
  reply: ((id: string) => string) | null = null;
  private stdoutListeners: ((chunk: Buffer) => void)[] = [];
  private handlers: Record<string, ((arg?: unknown) => void)[]> = {};
  stdin = {
    write: (data: string, cb?: (err?: Error | null) => void): void => {
      this.written.push(data);
      const body: unknown = JSON.parse(data);
      if (body && typeof body === "object" && "id" in body && typeof body.id === "string") {
        const id = body.id;
        // test-only: { hold: true } suppresses the auto-reply so the test drives timing.
        const held = "hold" in body && body.hold === true;
        if (!held) {
          const line = this.reply ? this.reply(id) : `${JSON.stringify({ id, marker: `res-${id}` })}\n`;
          queueMicrotask(() => this.emitStdout(line));
        }
      }
      cb?.(null);
    },
  };
  stdout = {
    on: (_event: "data", listener: (chunk: Buffer) => void): void => {
      this.stdoutListeners.push(listener);
    },
  };
  stderr = { on: (): void => { } };
  on = (event: "error" | "exit", listener: (arg?: unknown) => void): void => {
    this.handlers[event] ??= [];
    this.handlers[event].push(listener);
  };
  kill = (): void => {
    this.killCalls += 1;
  };
  emitStdout(s: string): void {
    for (const l of this.stdoutListeners) l(Buffer.from(s));
  }
  emitExit(): void {
    this.exitCode = 1;
    for (const l of this.handlers["exit"] ?? []) l();
  }
}

function setup(): { router: KernelRouter; children: FakeChild[] } {
  const children: FakeChild[] = [];
  const router = new KernelRouter({
    python: "py",
    workerPath: "w",
    spawnFn: (): KernelChild => {
      const c = new FakeChild();
      children.push(c);
      queueMicrotask(() => c.emitStdout('{"type":"ready"}\n'));
      return c;
    },
  });
  return { router, children };
}

function writtenIds(child: FakeChild): string[] {
  const ids: string[] = [];
  for (const w of child.written) {
    const v: unknown = JSON.parse(w);
    if (v && typeof v === "object" && "id" in v && typeof v.id === "string") ids.push(v.id);
  }
  return ids;
}

function markerOf(res: object): unknown {
  if ("marker" in res) return res.marker;
  throw new Error("response missing marker");
}

describe("parseTicker", () => {
  test("valid tickers trim and uppercase", () => {
    const valid: [string, string][] = [
      ["nvda", "NVDA"],
      [" brk.b ", "BRK.B"],
      ["RDS-A", "RDS-A"],
      ["\tmsft\n", "MSFT"],
      ["a", "A"],
      ["abcdefghij", "ABCDEFGHIJ"],
      ["x1", "X1"],
    ];
    for (const [input, out] of valid) expect(parseTicker(input)).toBe(out);
  });

  test("non-strings, blanks, organizations, and bad shapes drop to undefined", () => {
    const invalid: unknown[] = [
      undefined, null, 7, true, ["NVDA"], { ticker: "NVDA" },
      "", "   ",
      "SEC", "sec", "FINRA", "NYSE", "NASDAQ", "OTC", "EDGAR", "SHO", "REG", " reg ",
      "ABCDEFGHIJK", "1AB", "AB CD", "A-B-C-D-E-F-G-H-I-J-K", "NVDA!", ".AB", "-AB", "AB/C",
    ];
    for (const v of invalid) expect(parseTicker(v)).toBeUndefined();
  });
});

describe("parseWatchlist / asksWatchlist", () => {
  test("valid lists normalize and deduplicate in order; absent stays undefined", () => {
    expect(parseWatchlist([" nvda", "QQQ", "nvda", "brk.b"])).toEqual(["NVDA", "QQQ", "BRK.B"]);
    expect(parseWatchlist(undefined)).toBeUndefined();
    expect(parseWatchlist(null)).toBeUndefined();
  });

  test("malformed lists reject whole, never shrink", () => {
    for (const bad of ["NVDA", [], [7], ["NVDA", "SEC"], ["NVDA", ""], { 0: "NVDA" }, Array.from({ length: 26 }, (_, i) => `T${i}`)]) {
      expect(parseWatchlist(bad)).toBeNull();
    }
    expect(parseWatchlist(Array.from({ length: 25 }, (_, i) => `T${i}`))?.length).toBe(25);
  });

  test("only prompts that name the watchlist take watchlist scope", () => {
    for (const p of ["latest filings for my watchlist", "Watch list short interest?", "watch-lists summary", "WATCHLIST"]) expect(asksWatchlist(p)).toBe(true);
    for (const p of ["NVDA risk factors?", "what should I watch?", "watchlistsX", "list watches"]) expect(asksWatchlist(p)).toBe(false);
  });
});

describe("kernel startup", () => {
  test("startup health stays unready when the external worker exits before ready", () => {
    const dir = mkdtempSync(join(tmpdir(), "stockbot-kernel-startup-"));
    try {
      mkdirSync(join(dir, "venv", "bin"), { recursive: true });
      // The local worker exits before its ready message without loading Python or a provider.
      writeFileSync(join(dir, "venv", "bin", "python"), "#!/bin/sh\nexit 7\n", { mode: 0o755 });
      writeFileSync(
        join(dir, "probe.ts"),
        `import { dispatch, registerPrewarm } from ${JSON.stringify(join(import.meta.dir, "..", "..", "scripts", "api.ts"))};
delete process.env.KERNEL_PREWARMED;
await registerPrewarm();
const health = await dispatch(new Request("http://localhost/api/health"));
console.log(JSON.stringify({ status: health.status, ...await health.json() }));
`,
      );
      const child = Bun.spawnSync([process.execPath, join(dir, "probe.ts")], {
        cwd: join(import.meta.dir, "..", ".."),
        env: { ...process.env, STOCKBOT_REPO_ROOT: dir },
      });
      if (child.exitCode !== 0) throw new Error(child.stderr.toString());
      const result = JSON.parse(child.stdout.toString().trim().split("\n").at(-1) ?? "");
      expect(result).toMatchObject({ status: 200, ok: true, kernelPrewarmed: false });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 130_000);
});

describe("KernelRouter", () => {
  test("two sequential calls share one spawn with distinct correlated IDs", async () => {
    const { router, children } = setup();
    try {
      const r1 = await router.call({ op: "scope" }, { timeoutMs: 1000 });
      const r2 = await router.call({ op: "scope" }, { timeoutMs: 1000 });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      expect(children.length).toBe(1);
      expect(r1.id).toBe("1");
      expect(markerOf(r1)).toBe("res-1");
      expect(r2.id).toBe("2");
      expect(markerOf(r2)).toBe("res-2");
      expect(writtenIds(first)).toEqual(["1", "2"]);
      expect(first.killCalls).toBe(0);
    } finally {
      router.close();
    }
  });

  test("dead child fails over to a fresh spawn", async () => {
    const { router, children } = setup();
    try {
      await router.call({ op: "scope" }, { timeoutMs: 1000 });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.emitExit();
      const r = await router.call({ op: "scope" }, { timeoutMs: 1000 });
      expect(children.length).toBe(2);
      expect(r.id).toBe("2");
      expect(markerOf(r)).toBe("res-2");
    } finally {
      router.close();
    }
  });

  test("aborted call cannot kill the worker after its timeout", async () => {
    const { router, children } = setup();
    try {
      const controller = new AbortController();
      const pending = router.call({ op: "scope" }, { signal: controller.signal, timeoutMs: 20 });
      controller.abort();
      let err: unknown;
      try {
        await pending;
      } catch (e) {
        err = e;
      }
      if (!(err instanceof Error)) throw new Error("expected abort rejection");
      expect(err.message).toMatch("worker aborted");
      // Real delay past the aborted call's 20ms timeout: proves its timer was cleared.
      await Bun.sleep(60);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      expect(first.killCalls).toBe(0);
      expect(children.length).toBe(1);
      const r = await router.call({ op: "scope" }, { timeoutMs: 1000 });
      expect(children.length).toBe(1);
      expect(first.killCalls).toBe(0);
      expect(r.id).toBe("2");
      expect(markerOf(r)).toBe("res-2");
    } finally {
      router.close();
    }
  });
  test("timed-out call rejects alone while sibling still resolves", async () => {
    const { router, children } = setup();
    try {
      // Real 20ms timer: the timeout firing is the behavior under test, fake clocks cannot drive it.
      const slow = router.call({ op: "scope", hold: true }, { timeoutMs: 20 });
      const fast = router.call({ op: "scope", hold: true }, { timeoutMs: 1000 });
      // Observe both upfront so the sibling rejection (old failAll bug) cannot go unhandled.
      const slowSettled = slow.then((): null => null, (e: unknown): unknown => e);
      const fastSettled = fast.then(
        (r) => ({ ok: true as const, r }),
        (e: unknown) => ({ ok: false as const, e }),
      );
      const slowErr = await slowSettled;
      if (!(slowErr instanceof Error)) throw new Error("expected timeout rejection");
      expect(slowErr.message).toMatch("timeout");
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      expect(first.killCalls).toBe(0);
      expect(children.length).toBe(1);
      // Sibling was still pending across the timeout: drive its reply now.
      first.emitStdout('{"id":"2","marker":"res-2"}\n');
      const fastRes = await fastSettled;
      if (!fastRes.ok) {
        const detail = fastRes.e instanceof Error ? fastRes.e.message : String(fastRes.e);
        throw new Error(`sibling rejected: ${detail}`);
      }
      expect(markerOf(fastRes.r)).toBe("res-2");
      expect(first.killCalls).toBe(0);
      expect(children.length).toBe(1);
      expect(writtenIds(first)).toEqual(["1", "2"]);
    } finally {
      router.close();
    }
  });


  test("prewarm resolves only after the worker ready message is observed", async () => {
    const children: FakeChild[] = [];
    const router = new KernelRouter({
      python: "py",
      workerPath: "w",
      spawnFn: (): KernelChild => {
        const c = new FakeChild();
        children.push(c);
        return c;
      },
    });
    try {
      let settled = false;
      const pending = router.prewarm().then(() => {
        settled = true;
      });
      // Microtask flush only: prewarm awaits the unresolved ready gate, so it
      // cannot settle until the worker hello arrives — no wall-clock wait.
      await Promise.resolve();
      await Promise.resolve();
      expect(settled).toBe(false);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.emitStdout('{"type":"ready"}\n');
      await pending;
      expect(settled).toBe(true);
    } finally {
      router.close();
    }
  });
  // Fake clock drives the 120s prewarm deadline: no real wait, no ready auto-reply.
  async function fakeClockPrewarm(drive: (child: FakeChild, state: () => unknown) => Promise<void>): Promise<void> {
    const children: FakeChild[] = [];
    const router = new KernelRouter({
      python: "py",
      workerPath: "w",
      spawnFn: (): KernelChild => {
        const c = new FakeChild();
        children.push(c);
        return c;
      },
    });
    const prev = process.env.KERNEL_PREWARMED;
    delete process.env.KERNEL_PREWARMED;
    jest.useFakeTimers();
    try {
      let state: unknown = "pending";
      void router.prewarm().then(() => { state = "ready"; }, (e: unknown) => { state = e; });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      await drive(first, () => state);
    } finally {
      jest.useRealTimers();
      router.close();
      if (prev === undefined) delete process.env.KERNEL_PREWARMED;
      else process.env.KERNEL_PREWARMED = prev;
    }
  }
  const flush = async (): Promise<void> => {
    for (let i = 0; i < 10; i++) await Promise.resolve();
  };

  test("prewarm stays pending past 30s and resolves on ready before 120s", async () => {
    await fakeClockPrewarm(async (child, state) => {
      jest.advanceTimersByTime(30_001);
      await flush();
      expect(state()).toBe("pending");
      expect(child.killCalls).toBe(0);
      jest.advanceTimersByTime(119_999 - 30_001);
      await flush();
      expect(state()).toBe("pending");
      child.emitStdout('{"type":"ready"}\n');
      await flush();
      expect(state()).toBe("ready");
      expect(process.env.KERNEL_PREWARMED as string | undefined).toBe("1");
      // Ready cleared the deadline: crossing 120s afterwards never kills the worker.
      jest.advanceTimersByTime(1);
      await flush();
      expect(child.killCalls).toBe(0);
    });
  });

  test("prewarm kills the silent worker and rejects at 120s", async () => {
    await fakeClockPrewarm(async (child, state) => {
      jest.advanceTimersByTime(119_999);
      await flush();
      expect(state()).toBe("pending");
      jest.advanceTimersByTime(1);
      await flush();
      const err = state();
      if (!(err instanceof Error)) throw new Error("expected prewarm timeout rejection");
      expect(err.message).toMatch("kernel worker prewarm timeout after 120000ms");
      expect(child.killCalls).toBe(1);
      expect(process.env.KERNEL_PREWARMED).toBeUndefined();
    });
  });
  test("progress lines reach onProgress without resolving the pending call", async () => {
    const { router, children } = setup();
    try {
      const seen: Array<{ stage: string; detail?: Record<string, unknown> }> = [];
      const pending = router.call({ op: "scope", hold: true }, { timeoutMs: 1000, onProgress: (stage, detail) => seen.push(detail !== undefined ? { stage, detail } : { stage }) });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      await Promise.resolve();
      first.emitStdout('{"type":"progress","id":"1","stage":"intake_done","detail":{"calls":3}}\n');
      await Promise.resolve();
      expect(seen).toEqual([{ stage: "intake_done", detail: { calls: 3 } }]);
      // Pending survives the progress line: the terminal reply still resolves it.
      first.emitStdout('{"id":"1","marker":"res-1"}\n');
      const res = await pending;
      expect(markerOf(res)).toBe("res-1");
    } finally {
      router.close();
    }
  });

  test("a throwing progress listener never fails the pending call", async () => {
    const { router, children } = setup();
    try {
      const pending = router.call({ op: "scope", hold: true }, {
        timeoutMs: 1000,
        onProgress: () => {
          throw new Error("listener down");
        },
      });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      await Promise.resolve();
      first.emitStdout('{"type":"progress","id":"1","stage":"intake"}\n');
      await Promise.resolve();
      first.emitStdout('{"id":"1","marker":"res-1"}\n');
      expect(markerOf(await pending)).toBe("res-1");
    } finally {
      router.close();
    }
  });

  test("bridge noise, non-object lines, and unknown ids never settle a call", async () => {
    const { router, children } = setup();
    try {
      const pending = router.call({ op: "scope", hold: true }, { timeoutMs: 1000 });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      await Promise.resolve();
      first.emitStdout('not json\n{"broken":\n7\nnull\n[]\n{"type":"progress","id":"99","stage":"other"}\n{"id":"99","marker":"stale"}\n');
      await Promise.resolve();
      first.emitStdout('{"id":"1","marker":"res-1"}\n');
      expect(markerOf(await pending)).toBe("res-1");
    } finally {
      router.close();
    }
  });
});


describe("proof helpers", () => {
  test("alignClaimText maps a differently spelled claim to its exact prose, never to different numbers", () => {
    const prose = "NVIDIA reported Revenue of $96,221 million for Q3 [ev:1]. Margins \u201cmay\u201d  rise if demand holds [ev:2].";
    expect(alignClaimText(prose, "NVIDIA reported Revenue of $96,221 million for Q3")).toBe("NVIDIA reported Revenue of $96,221 million for Q3");
    expect(alignClaimText(prose, "margins \"may\" rise if demand holds.")).toBe("Margins \u201cmay\u201d  rise if demand holds");
    expect(alignClaimText(prose, "NVIDIA reported revenue of $96,221 million in Q3")).toBe("NVIDIA reported Revenue of $96,221 million for Q3 [ev:1].");
    expect(alignClaimText(prose, "NVIDIA reported Revenue of $96,222 million for Q3")).toBeUndefined();
    expect(alignClaimText(prose, "Gaming collapsed")).toBeUndefined();
  });
  const revenuePoint: Datapoint = {
    id: "dp:revenue", metric: "revenue", value: "$35.1B", entity: "NVDA", period: "Q3 FY2025", unit: "USD",
    statement: "NVDA Q3 FY2025 revenue was $35.1B USD.",
  };

  test("inventoryDefect rejects non-substrings, unknown ids, and uncited non-unknown claims", () => {
    const known = new Set(["ev:1"]);
    const points = datapointRows([]);
    expect(inventoryDefect("prose", [{ text: "missing", claimType: "inference", evidenceIds: ["ev:1"] }], known, points)).toMatch(/substring/);
    expect(inventoryDefect("prose has it", [{ text: "has it", claimType: "inference", evidenceIds: ["ev:9"] }], known, points)).toMatch(/unknown evidence id/);
    expect(inventoryDefect("prose has it", [{ text: "has it", claimType: "observed_fact", evidenceIds: [] }], known, points)).toMatch(/uncited/);
    expect(inventoryDefect("prose has it", [{ text: "has it", claimType: "unknown", evidenceIds: [] }], known, points)).toBeUndefined();
  });

  test("inventoryDefect requires datapoints carried by cited rows and calculations to reference datapoints", () => {
    const margin: Datapoint = { ...revenuePoint, id: "dp:margin", metric: "margin", value: "75%", unit: "%", inputs: ["dp:revenue"], statement: "NVDA Q3 FY2025 margin was 75%." };
    const evidence: Evidence[] = [
      { id: "ev:1", source: "kernel", retrievedAt: "t", content: "fact", datapoints: [revenuePoint, margin] },
      { id: "ev:2", source: "kernel", retrievedAt: "t", content: "other fact" },
    ];
    const known = new Set(evidence.map((e) => e.id));
    const points = datapointRows(evidence);
    const prose = `${revenuePoint.statement} ${margin.statement} Assume flat demand.`;
    const fact: FinalClaim = { text: revenuePoint.statement, claimType: "observed_fact", evidenceIds: ["ev:1"], datapointIds: [revenuePoint.id] };
    expect(points).toEqual(new Map([[revenuePoint.id, ["ev:1"]], [margin.id, ["ev:1"]]]));
    // The same datapoint on a second row (a claim-time admission) is citable through either row.
    const shared = datapointRows([...evidence, { id: "ev:3", source: "kernel", retrievedAt: "t", content: "admitted", datapoints: [revenuePoint] }]);
    expect(inventoryDefect(prose, [fact], new Set([...known, "ev:3"]), shared)).toBeUndefined();
    expect(inventoryDefect(prose, [{ ...fact, evidenceIds: ["ev:3"] }], new Set([...known, "ev:3"]), shared)).toBeUndefined();
    expect(inventoryDefect(prose, [fact], known, points)).toBeUndefined();
    expect(inventoryDefect(prose, [{ text: "Assume flat demand.", claimType: "assumption", evidenceIds: [] }], known, points)).toBeUndefined();
    expect(inventoryDefect(prose, [{ ...fact, datapointIds: ["dp:missing"] }], known, points)).toBe("inventory cites unknown datapoint id: dp:missing");
    expect(inventoryDefect(prose, [{ ...fact, evidenceIds: ["ev:2"] }], known, points)).toBe("inventory cites unknown datapoint id: dp:revenue");
    expect(inventoryDefect(prose, [{ ...fact, claimType: "unknown", evidenceIds: [] }], known, points)).toMatch(/unknown datapoint id/);
    expect(inventoryDefect(prose, [{ text: margin.statement, claimType: "calculation", evidenceIds: ["ev:1"] }], known, points)).toMatch(/calculation without datapoint_ids/);
    expect(inventoryDefect(prose, [{ text: margin.statement, claimType: "calculation", evidenceIds: ["ev:1"], datapointIds: [margin.id] }], known, points)).toBeUndefined();
  });

  test("dropUncovered requires full sentence coverage for qualitative clauses and numbers", () => {
    const claim: FinalClaim = { text: "Revenue was $35.1B", claimType: "observed_fact", evidenceIds: ["ev:1"] };
    const text = "Revenue was $35.1B [ev:12]. Q3 looks firm. Margin hit 99%.\nGrowth was 12% too.";
    expect(dropUncovered(text, [claim])).toBe("Revenue was $35.1B [ev:12].");
    expect(dropUncovered("Revenue was $35.1B, up 40%.", [claim])).toBe("");
    expect(dropUncovered("Revenue was $35.1B and demand is strong.", [claim])).toBe("");
    expect(dropUncovered("No numbers here. Filed a 10-K.", [])).toBe("");
    const demand: FinalClaim = { text: "demand may hold", claimType: "inference", evidenceIds: ["ev:1"] };
    expect(dropUncovered("Revenue was $35.1B and demand may hold [ev:1].", [claim, demand])).toBe("Revenue was $35.1B and demand may hold [ev:1].");
    // A trailing citation follows its own sentence: kept with a kept sentence, dropped with a dropped one.
    expect(dropUncovered("Revenue was $35.1B. [ev:1]\nMargin hit 99%. [ev:2]\n[ev:3]", [claim])).toBe("Revenue was $35.1B. [ev:1]");
  });

  test("readableNumbers shows scaled and verified large values, never ids or unverified numbers", () => {
    const point = (metric: string, value: string): Datapoint => ({ id: `dp:${value}`, metric, value, entity: "NVDA", period: "FY", statement: "s" });
    const points = [point("Revenue", "215938000000"), point("currentShortPositionQuantity", "69350469"), point("changePercent", "-2.23"), point("changePreviousNumber", "-1580843")];
    expect(readableNumbers("NVDA reported Revenue of $96,221 million for Q3 [ev:1].", points)).toBe("NVDA reported Revenue of $96.2 billion for Q3 [ev:1].");
    expect(readableNumbers("Revenue of $215938000000 and 11,536 millions and $95,000 million and 5,000 thousand.", points))
      .toBe("Revenue of $215.9 billion and 11.5 billion and $95 billion and 5 million.");
    expect(readableNumbers("TSLA reported currentShortPositionQuantity of 69350469 for 2026-10-15.", points))
      .toBe("TSLA reported current short position quantity of 69.4 million for 2026-10-15.");
    expect(readableNumbers("TSLA reported changePreviousNumber of -1580843 and changePercent of -2.23.", points))
      .toBe("TSLA reported change previous number of -1.6 million and change percent of -2.23.");
    // Identities stay verbatim: a CIK, a numeric citation id, an accession, a date, an unverified number, a small value.
    const kept = "CIK 1045810 [ev:215938000000] [ev:12345678] 0001045810-26-000075 2026-01-25 12345678 $4.93 $500 million daysToCoverQuantity";
    expect(readableNumbers(kept, points)).toBe(kept);
  });

  test("readDatapoints drops malformed descriptors and sanitizes optional fields", () => {
    for (const raw of [undefined, null, {}, "dp:revenue", 7]) expect(readDatapoints(raw)).toEqual([]);
    const malformed: unknown[] = [null, [], "dp:revenue", 7, { ...revenuePoint, id: "ev:1" }];
    for (const key of ["id", "metric", "value", "entity", "period", "statement"] as const) {
      malformed.push({ ...revenuePoint, [key]: "" }, { ...revenuePoint, [key]: 7 }, { ...revenuePoint, [key]: null });
      const missing: Record<string, unknown> = { ...revenuePoint };
      delete missing[key];
      malformed.push(missing);
    }
    expect(readDatapoints(malformed)).toEqual([]);
    expect(readDatapoints([
      ...malformed,
      { ...revenuePoint, extra: "ignored", inputs: ["dp:input", "", 7, null] },
      { ...revenuePoint, unit: 7, inputs: "dp:input" },
    ])).toEqual([
      { ...revenuePoint, inputs: ["dp:input"] },
      { id: revenuePoint.id, metric: revenuePoint.metric, value: revenuePoint.value, entity: revenuePoint.entity, period: revenuePoint.period, statement: revenuePoint.statement },
    ]);
  });

});
describe("outOfScopeAnswer", () => {
  const FIXED = "This request is outside the scope of stock research. I can only research markets, companies, securities, and investing, so this request is unsupported.";
  test("an out-of-scope verdict carries the runtime's fixed answer", () => {
    expect(outOfScopeAnswer({ in_scope: false, answer: FIXED })).toBe(FIXED);
    expect(outOfScopeAnswer({ in_scope: false, answer: `  ${FIXED}\n` })).toBe(FIXED);
  });
  test("an in-scope, absent, or malformed verdict never substitutes an answer", () => {
    for (const reply of [{ in_scope: true, answer: FIXED }, {}, { in_scope: false }, { in_scope: false, answer: "   " }, { in_scope: false, answer: 7 }, { in_scope: "false", answer: FIXED }]) {
      expect(outOfScopeAnswer(reply)).toBeUndefined();
    }
  });

});

const source: Evidence = { id: "ev:1", source: "sec", retrievedAt: "2026-11-05", content: "Demand held. Costs rose." };
const fact = (text: string): FinalClaim => ({ text, claimType: "observed_fact", evidenceIds: [source.id] });
async function research(options: {
  asOf?: string;
  prose?: string;
  claims?: FinalClaim[];
  evidence?: Evidence;
  seedEvidence?: Evidence[];
  admission?: Record<string, unknown> | Error;
  proof?: unknown;
  toolError?: boolean;
  reasonError?: boolean;
  schemaFailure?: "missing" | "unavailable";
} = {}) {
  const events: AgentEvent[] = [];
  const bodies: Record<string, unknown>[] = [];
  const calls: KernelReasonInput[] = [];
  const spawnFn: KernelSpawn = () => {
    const child = new FakeChild();
    child.reply = (id) => {
      const body = JSON.parse(child.written.at(-1)!);
      bodies.push(body);
      if (body.op === "admit_datapoint") {
        if (options.admission instanceof Error) {
          queueMicrotask(() => child.emitExit());
          return "";
        }
        return JSON.stringify({ id, ...(options.admission ?? { error: "Admission unavailable." }) }) + "\n";
      }
      const reply = body.op === "open_research" ? { sessionId: "rs:1", tools: ["search_web", "get_sec_document"] }
        : body.op === "research_tool" ? { ok: !options.toolError, summary: "[ev:1] Demand held. Costs rose.", evidence: [options.evidence ?? source], ...(options.toolError ? { error: "Source unavailable." } : {}) }
          : body.op === "prove_claims" ? options.proof ?? { verdicts: body.claims.map((claim: Record<string, unknown>) => ({ claim_type: claim.claim_type, support_answer: "yes", support: 1 })) }
            : { error: "Unexpected op." };
      return JSON.stringify({ id, ...reply }) + "\n";
    };
    queueMicrotask(() => child.emitStdout('{"type":"ready"}\n'));
    return child;
  };
  await runKernelAgent("Explain NVDA", (event) => events.push(event), {
    ticker: "NVDA", asOf: options.asOf, seedEvidence: options.seedEvidence, deps: {
      spawnFn, describeTools: async () => {
        if (options.schemaFailure === "unavailable") throw new Error("Schema bridge unavailable.");
        if (options.schemaFailure === "missing") return [{ name: "get_sec_document" }];
        return [{ name: "get_sec_document" }, { name: "call_tool" }, { name: "search_web" }];
      },
      reason: async (input) => {
        calls.push(input);
        expect(await input.runTool!("not_offered", {})).toContain("not available");
        expect(await input.runTool!("call_tool", {})).toBe("error: call_tool is not available in research");
        const text = await input.runTool!("get_sec_document", { limit: 999999, accession: "x" });
        expect(text).toContain(options.toolError ? "Source unavailable" : "ev:1");
        if (options.reasonError) { input.usage!.providerCalls = 3; throw new Error("Provider unavailable."); }
        const prose = options.prose ?? "Demand held. Costs rose.";
        input.onDelta?.(prose);
        return { prose, claims: options.claims ?? [fact("Demand held."), fact("Costs rose.")], usage: { providerCalls: 2 } };
      },
    }
  });
  return { events, bodies, calls };
}
describe("native research and deterministic proof", () => {
  test("the research cutoff reaches open_research unchanged and stays absent by default", async () => {
    const asOf = "2026-01-03T00:00:00+00:00";
    expect((await research({ asOf })).bodies[0]).toMatchObject({ op: "open_research", asOf });
    expect((await research()).bodies[0]).not.toHaveProperty("asOf");
  });
  test("one session and one worker op per offered tool call", async () => {
    const { events, bodies, calls } = await research();
    expect(bodies.map((body) => body.op)).toEqual(["open_research", "research_tool", "prove_claims"]);
    expect(bodies[1]).toMatchObject({ sessionId: "rs:1", tool: "get_sec_document", arguments: { limit: 12000, accession: "x" } });
    expect(calls).toHaveLength(1);
    expect(events.find((event) => event.type === "evidence")).toMatchObject({ evidence: [source], sessionId: "rs:1" });
    expect(events.at(-1)).toMatchObject({ status: "completed", metrics: { needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 1 }, muse: { calls: 1, providerCalls: 2 } } });
    expect(events.map((event) => event.type)).toEqual([
      "agent_start", "status", "reasoning_start", "status", "tool_start", "evidence", "tool_result",
      "answer_delta", "claims", "done",
    ]);
    expect(bodies[2]).toMatchObject({ claims: [{ text: "Demand held.", claim_type: "observed_fact", evidence_ids: ["ev:1"] }, { text: "Costs rose.", claim_type: "observed_fact", evidence_ids: ["ev:1"] }] });
  });
  test("offered schemas exactly follow the worker's loop-safe names and order", async () => {
    const { calls, bodies } = await research();
    expect(calls[0]?.tools).toEqual([{ name: "search_web" }, { name: "get_sec_document" }]);
    expect(bodies.filter((body) => body.op === "research_tool").map((body) => body.tool)).toEqual(["get_sec_document"]);
  });
  test("missing schemas and schema outages fail without starting the model", async () => {
    for (const schemaFailure of ["missing", "unavailable"] as const) {
      const { calls, bodies, events } = await research({ schemaFailure });
      expect(calls).toEqual([]);
      expect(bodies.map((body) => body.op)).toEqual(["open_research"]);
      expect(events.find((event) => event.type === "failed")).toMatchObject({ category: "provider_error" });
      expect(events.at(-1)).toMatchObject({ status: "failed" });
    }
  });
  test("a claim that states a cited row's own datapoint value binds that datapoint", async () => {
    const filing: Evidence = {
      id: "ev:2", source: "sec", retrievedAt: "2026-11-05", content: "filed_at: 2026-11-03 (Apple Inc.)",
      datapoints: [{ id: "dp:filed", metric: "filed_at", value: "2026-11-03", entity: "Apple Inc.", period: "2026-11-03", statement: "filed_at: 2026-11-03" }],
    };
    const claim: FinalClaim = { text: "Apple Inc. filed_at 2026-11-03.", claimType: "observed_fact", evidenceIds: ["ev:2"] };
    const { bodies } = await research({ evidence: filing, prose: claim.text, claims: [claim] });
    const proof = bodies.find((body) => body.op === "prove_claims");
    expect(proof?.claims).toMatchObject([{ datapoint_ids: ["dp:filed"] }]);
  });
  test("a cited row never binds a value the claim does not state", async () => {
    const filing: Evidence = {
      id: "ev:2", source: "sec", retrievedAt: "2026-11-05", content: "Revenue 96,221 for the quarter.",
      datapoints: [{ id: "dp:rev", metric: "Revenue", value: "96,221", entity: "NVDA", period: "Q3", statement: "Revenue 96,221" }],
    };
    const claim: FinalClaim = { text: "Demand held.", claimType: "observed_fact", evidenceIds: ["ev:2"] };
    const { bodies } = await research({ evidence: filing, prose: claim.text, claims: [claim] });
    const proof = bodies.find((body) => body.op === "prove_claims");
    expect(proof?.claims).not.toMatchObject([{ datapoint_ids: expect.anything() }]);
  });
  test("two datapoints sharing a stated value keep the one the sentence names", async () => {
    // A filing date and a period can carry the same value. The proof rejects a number bound to several
    // datapoints, so the claim keeps the one its metric word names and drops the other.
    const filing: Evidence = {
      id: "ev:2", source: "sec", retrievedAt: "2026-11-05", content: "10-K filed 2026-11-03.",
      datapoints: [
        { id: "dp:filed", metric: "filed_at", value: "2026-11-03", entity: "Apple Inc.", period: "2026-11-03", statement: "filed_at: 2026-11-03" },
        { id: "dp:period", metric: "report_period", value: "2026-11-03", entity: "Apple Inc.", period: "2026-11-03", statement: "report_period: 2026-11-03" },
      ],
    };
    const claim: FinalClaim = {
      text: "Apple Inc. filed_at 2026-11-03.", claimType: "observed_fact", evidenceIds: ["ev:2"],
      datapointIds: ["dp:filed", "dp:period"],
    };
    const { bodies } = await research({ evidence: filing, prose: claim.text, claims: [claim] });
    const proof = bodies.find((body) => body.op === "prove_claims");
    expect(proof?.claims).toMatchObject([{ datapoint_ids: ["dp:filed"] }]);
  });
  test("a failed declared number stays unproven", async () => {
    for (const admission of [{ error: "Admission unavailable." }, {}, { evidence_id: "", datapoint: {} }, new Error("Admission transport failed.")]) {
      const claim: FinalClaim = {
        text: "Revenue grew 12%.", claimType: "calculation", evidenceIds: ["ev:1"],
        numbers: [{ evidence_id: "ev:1", metric: "growth", value: "12%", entity: "NVDA", period: "Q3" }],
      };
      const { bodies, events } = await research({ prose: claim.text, claims: [claim], admission });
      expect(bodies.map((body) => body.op)).toEqual(["open_research", "research_tool", "admit_datapoint"]);
      expect(claim.datapointIds).toBeUndefined();
      expect(events.findLast((event) => event.type === "answer_partial")).toMatchObject({ claims: [] });
      expect(events.at(-1)).toMatchObject({ status: "completed_with_limitations" });
    }
  });
  test("an admitted declaration adds its datapoint and evidence before proof", async () => {
    const point: Datapoint = { id: "dp:growth", metric: "growth", value: "12%", entity: "NVDA", period: "Q3", statement: "Revenue growth 12%." };
    for (const existing of [false, true]) {
      const claim: FinalClaim = {
        text: "Revenue growth 12%.", claimType: "calculation", evidenceIds: existing ? ["ev:1", "ev:admitted"] : ["ev:1"],
        datapointIds: existing ? [point.id] : undefined,
        numbers: [{ evidence_id: "ev:1", metric: "growth", value: "12%", entity: "NVDA", period: "Q3", unit: "%", formula: "growth@1", inputs: ["dp:previous"] }],
      };
      const { bodies, events } = await research({
        prose: claim.text, claims: [claim], admission: { evidence_id: "ev:admitted", datapoint: point },
        seedEvidence: existing ? [{ ...source, id: "ev:admitted", datapoints: [point] }] : [],
      });
      expect(bodies.map((body) => body.op)).toEqual(["open_research", "research_tool", "admit_datapoint", "prove_claims"]);
      expect(bodies[2]).toMatchObject({ evidenceId: "ev:1", unit: "%", formula: "growth@1", inputs: ["dp:previous"] });
      expect(bodies[3]?.claims).toMatchObject([{ evidence_ids: ["ev:1", "ev:admitted"], datapoint_ids: [point.id] }]);
      expect(events.filter((event) => event.type === "evidence").flatMap((event) => event.evidence).filter((row) => row.id === "ev:admitted")).toHaveLength(1);
      expect(events.at(-1)).toMatchObject({ status: "completed" });
    }
  });
  test("a declaration without its cited source never reaches admission", async () => {
    for (const evidence_id of ["ev:missing", "ev:uncited"]) {
      const claim: FinalClaim = {
        text: "Revenue growth 12%.", claimType: "calculation", evidenceIds: ["ev:1"],
        numbers: [{ evidence_id, metric: "growth", value: "12%", entity: "NVDA", period: "Q3" }],
      };
      const { bodies, events } = await research({ prose: claim.text, claims: [claim], seedEvidence: [{ ...source, id: "ev:uncited" }] });
      expect(bodies.map((body) => body.op)).toEqual(["open_research", "research_tool"]);
      expect(events.at(-1)).toMatchObject({ status: "completed_with_limitations" });
    }
  });
  test("an input datapoint whose value the sentence never states leaves the claim", async () => {
    const growth: Datapoint = { id: "dp:growth", metric: "growth", value: "12%", entity: "NVDA", period: "Q3", statement: "Revenue growth 12%.", inputs: ["dp:previous"] };
    const previous: Datapoint = { ...growth, id: "dp:previous", metric: "previous", value: "100", inputs: [] };
    const claim: FinalClaim = { text: growth.statement, claimType: "calculation", evidenceIds: ["ev:1"], datapointIds: [growth.id, previous.id, "dp:missing"] };
    const { bodies } = await research({ evidence: { ...source, datapoints: [growth, previous] }, prose: claim.text, claims: [claim] });
    expect(bodies.find((body) => body.op === "prove_claims")?.claims).toMatchObject([{ datapoint_ids: [growth.id] }]);
  });
  test("two equally named datapoints with the same stated value stay unbound", async () => {
    const point: Datapoint = { id: "dp:first", metric: "growth", value: "12%", entity: "NVDA", period: "Q3", statement: "Revenue growth 12%." };
    for (const datapointIds of [undefined, ["dp:first", "dp:second"]]) {
      const claim: FinalClaim = { text: point.statement, claimType: "calculation", evidenceIds: ["ev:1"], datapointIds };
      const { bodies, events } = await research({ evidence: { ...source, datapoints: [point, { ...point, id: "dp:second" }] }, prose: claim.text, claims: [claim] });
      expect(bodies.some((body) => body.op === "prove_claims")).toBe(false);
      expect(events.findLast((event) => event.type === "answer_partial")).toMatchObject({ claims: [] });
    }
  });
  test("a bound datapoint adds its row to evidence_ids", async () => {
    const point: Datapoint = { id: "dp:growth", metric: "growth", value: "12%", entity: "NVDA", period: "Q3", statement: "Revenue growth 12%." };
    for (const datapointIds of [undefined, [point.id]]) {
      const claim: FinalClaim = { text: point.statement, claimType: "calculation", evidenceIds: ["ev:1"], datapointIds };
      const { bodies } = await research({ seedEvidence: [{ ...source, id: "ev:point", datapoints: [point] }], prose: claim.text, claims: [claim] });
      expect(bodies.find((body) => body.op === "prove_claims")?.claims).toMatchObject([{ evidence_ids: ["ev:1", "ev:point"], datapoint_ids: [point.id] }]);
    }
  });
  test("proof rejection removes its sentence and replaces the streamed draft", async () => {
    const { events } = await research({
      proof: {
        verdicts: [
          { claim_type: "observed_fact", support_answer: "yes", support: 1 },
          { claim_type: "observed_fact", support_answer: "no", support: 0, error: "proof: unsupported" },
        ]
      }
    });
    expect(events.findLast((event) => event.type === "answer_partial")).toMatchObject({ text: "Demand held.", claims: [fact("Demand held.")] });
    expect(events.at(-1)).toMatchObject({ status: "completed_with_limitations" });
  });
  test("uncovered prose cannot ride beside supported claims", async () => {
    const { events } = await research({ claims: [fact("Demand held.")] });
    expect(events.findLast((event) => event.type === "answer_partial")).toMatchObject({ text: "Demand held." });
  });
  test("all rejections replace the draft with the fixed unknown answer", async () => {
    const { events } = await research({ claims: [fact("Demand held.")], proof: { verdicts: [{ claim_type: "observed_fact", support_answer: "no", error: "proof: unsupported" }] } });
    expect(events.findLast((event) => event.type === "answer_partial")).toMatchObject({ text: "I couldn't determine this from the retrieved sources.", claims: [] });
    expect(events.at(-1)).toMatchObject({ status: "completed_with_limitations" });
  });
  test("malformed proof replies and proof outages fail closed", async () => {
    for (const proof of [{}, { verdicts: [] }, { verdicts: [null, null] }, { verdicts: [{ support_answer: "yes" }, {}] }, { verdicts: [{ claim_type: "observed_fact", support_answer: "yes", error: "database unavailable" }, {}] }]) {
      const { events } = await research({ proof });
      expect(events.some((event) => event.type === "claims")).toBe(false);
      expect(events.at(-1)).toMatchObject({ status: "failed" });
    }
  });
  test("provider failure retains usage and the failure shape", async () => {
    const { events } = await research({ reasonError: true });
    expect(events.find((event) => event.type === "failed")).toMatchObject({ category: "provider_error", message: "Provider unavailable." });
    expect(events.at(-1)).toMatchObject({ metrics: { muse: { providerCalls: 3 } }, status: "failed" });
  });
  test("tool limitations publish a checked snapshot", async () => {
    const { events } = await research({ toolError: true });
    expect(events.findLast((event) => event.type === "answer_partial")).toMatchObject({ text: "Demand held. Costs rose." });
    expect(events.at(-1)).toMatchObject({ status: "completed_with_limitations", limitations: expect.arrayContaining(["Source unavailable."]) });
  });
  test("section labels survive when their bodies contain supported claims", async () => {
    const prose = "── Bearbot ──\nCosts rose.\n\n── Bullbot ──\nDemand held.\n\n── Stockbot ──\nDemand held.";
    const { events } = await research({ prose });
    const final = events.findLast((event) => event.type === "answer_partial");
    if (final?.type === "answer_partial") expect(final.text).toBe(prose);
    expect(events.at(-1)).toMatchObject({ status: "completed" });
  });
  test("property: unknown evidence ids never enter proof", () => {
    fc.assert(fc.property(fc.array(fc.constantFrom("ev:1", "ev:missing"), { minLength: 1 }), (evidenceIds) => {
      const defect = inventoryDefect("Demand held.", [{ ...fact("Demand held."), evidenceIds }], new Set(["ev:1"]), new Map());
      expect(defect === undefined).toBe(evidenceIds.every((id) => id === "ev:1"));
    }), { seed: 20261105 });
  });
  test("property: uncovered sentences never publish", () => {
    fc.assert(fc.property(fc.array(fc.boolean(), { minLength: 1, maxLength: 12 }), (keep) => {
      const sentences = keep.map((_, index) => `Fact ${index}.`);
      const kept = sentences.filter((_, index) => keep[index]).map(fact);
      const shown = dropUncovered(sentences.join("\n"), kept);
      for (const [index, sentence] of sentences.entries()) expect(shown.includes(sentence)).toBe(keep[index]!);
    }), { seed: 20261105 });
  });
});
