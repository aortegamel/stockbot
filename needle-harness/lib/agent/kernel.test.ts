import { describe, expect, jest, setSystemTime, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import fc from "fast-check";
import { applyFinalGates, asksWatchlist, citedOrThin, inventoryDefect, KernelRouter, parseTicker, parseWatchlist, PASS2_MIN_WALL_MS, REQUEST_WALL_MS, runKernelAgent, runPersonaAgent, type FinalGateCaller, type KernelChild, type KernelReasonInput, type KernelReasonResult, type KernelSpawn } from "./kernel";
import { FINAL_PERSONAS, type AgentEvent, type Evidence, type FinalClaim, type Metrics, type Persona } from "./types";

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
      const r1 = await router.call({ op: "run" }, { timeoutMs: 1000 });
      const r2 = await router.call({ op: "run" }, { timeoutMs: 1000 });
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
      await router.call({ op: "run" }, { timeoutMs: 1000 });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.emitExit();
      const r = await router.call({ op: "run" }, { timeoutMs: 1000 });
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
      const pending = router.call({ op: "run" }, { signal: controller.signal, timeoutMs: 20 });
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
      const r = await router.call({ op: "run" }, { timeoutMs: 1000 });
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
      const slow = router.call({ op: "run", hold: true }, { timeoutMs: 20 });
      const fast = router.call({ op: "run", hold: true }, { timeoutMs: 1000 });
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
      const pending = router.call({ op: "run", hold: true }, { timeoutMs: 1000, onProgress: (stage, detail) => seen.push(detail !== undefined ? { stage, detail } : { stage }) });
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
      const pending = router.call({ op: "run", hold: true }, {
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
      const pending = router.call({ op: "run", hold: true }, { timeoutMs: 1000 });
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

describe("runKernelAgent final personas", () => {
  const graphReply = (id: string): string =>
    `${JSON.stringify({ id, objective: "q", evidence: [{ id: "ev:1", content: "fact" }], nodes: [{ node_id: "n1", question: "q", status: "blocked", depends_on: [] }], decisions: [], unresolved: ["n1"], incomplete_guard: true, toolExecutions: [], toolCalls: [], failures: {}, escalations: 1, escalated: true })}\n`;
  // Same shape with the session id a current worker reports: the gates need it.
  const sessionGraphReply = (id: string): string =>
    `${JSON.stringify({ id, objective: "q", sessionId: "rs:live", evidence: [{ id: "ev:1", content: "fact" }], nodes: [], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
  test("seed and follow-up evidence retain captures before the gated answer without changing model content", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: AgentEvent[] = [];
    const firstSource = { id: `capture:${"a".repeat(64)}` };
    const kernelSource = { id: "evidence:rs:test:ev:1" };
    const followSource = { id: "evidence:rs:test:ev:2" };
    const seed: Evidence = { id: "ev:1", source: "seed", retrievedAt: "2026-01-01Z", content: "seed fact", sources: [firstSource] };
    let replies = 0;
    let writes = 0;
    const seen: string[][] = [];
    const spawnFn = (): KernelChild => {
      const child = new FakeChild();
      child.reply = (id) => {
        const body = JSON.parse(child.written.at(-1) ?? "{}") as { op?: string };
        if (body.op?.startsWith("assess_final")) return `${JSON.stringify({ id, support_answer: "yes" })}\n`;
        replies += 1;
        return `${JSON.stringify({
          id, sessionId: "rs:test",
          evidence: replies === 1
            ? [{ id: "ev:1", content: "kernel fact", sources: [kernelSource] }, { id: "ev:bad", content: "uncaptured", sources: [{ id: "/raw/path" }] }]
            : [{ id: "ev:1", content: "changed duplicate", sources: [kernelSource, followSource] }, { id: "ev:2", content: "second fact", title: "Follow-up filing", url: "https://www.sec.gov/filing", sources: [followSource] }, { id: "ev:2", content: "duplicate", sources: [] }],
          nodes: [], decisions: [], unresolved: [], needleDecisions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false,
        })}\n`;
      };
      queueMicrotask(() => child.emitStdout('{"type":"ready"}\n'));
      return child;
    };
    await runKernelAgent("q", (e) => { events.push(e); }, {
      seedEvidence: [seed],
      personas: ["stockbot"],
      deps: {
        python: "py", workerPath: "w", spawnFn, reason: async (o) => {
          writes += 1;
          seen.push(o.evidence.map((e) => e.content));
          o.onDelta(`answer ${writes}`);
          return writes === 1
            ? { text: "answer 1", usage: {}, missingEvidence: "missing" }
            : { text: "answer 2", usage: {}, claims: [{ text: "answer 2", claimType: "inference", evidenceIds: ["ev:2"] }] };
        }
      },
    });
    const delivered = events.filter((e): e is Extract<AgentEvent, { type: "evidence" }> => e.type === "evidence");
    expect(delivered).toHaveLength(3);
    const seedDelivered = delivered[0];
    const researchDelivered = delivered[1];
    const followDelivered = delivered[2];
    if (!seedDelivered || !researchDelivered || !followDelivered) throw new Error("expected three evidence events");
    expect(seedDelivered.evidence[0]?.sources).toEqual([firstSource]);
    expect(researchDelivered.sessionId).toBe("rs:test");
    expect(researchDelivered.evidence.map((e) => e.id)).toEqual(["ev:1", "ev:bad"]);
    expect(researchDelivered.evidence[0]?.content).toBe("seed fact");
    expect(researchDelivered.evidence[0]?.sources).toEqual([firstSource, kernelSource]);
    expect(researchDelivered.evidence[1]?.sources).toEqual([]);
    expect(followDelivered.evidence.filter((e) => e.id === "ev:1")).toHaveLength(1);
    expect(followDelivered.evidence.filter((e) => e.id === "ev:1").map((e) => e.sources)).toEqual([[firstSource, kernelSource, followSource]]);
    expect(followDelivered.evidence.filter((e) => e.id === "ev:2")).toHaveLength(1);
    expect(followDelivered.evidence.find((e) => e.id === "ev:2")?.content).toBe("second fact");
    expect(followDelivered.evidence.find((e) => e.id === "ev:2")?.sources).toEqual([followSource]);
    expect(followDelivered.evidence.find((e) => e.id === "ev:2")).toMatchObject({ title: "Follow-up filing", url: "https://www.sec.gov/filing" });
    expect(seen).toEqual([["seed fact", "uncaptured"], ["seed fact", "uncaptured", "second fact"]]);
    expect(events.filter((e) => e.type === "evidence" || e.type === "answer_delta").map((e) => e.type)).toEqual(["evidence", "evidence", "evidence", "answer_delta"]);
    expect(events.flatMap((e) => e.type === "answer_delta" ? [e.text] : [])).toEqual(["── Stockbot ──\nanswer 2"]);
  });

  test("empty research answers never fabricate source references", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: AgentEvent[] = [];
    const spawnFn = (): KernelChild => {
      const child = new FakeChild();
      child.reply = (id) => `${JSON.stringify({ id, evidence: [], nodes: [], unresolved: [], toolCalls: [], needleDecisions: [] })}\n`;
      queueMicrotask(() => child.emitStdout('{"type":"ready"}\n'));
      return child;
    };
    await runKernelAgent("hello", (e) => { events.push(e); }, {
      personas: ["stockbot"],
      deps: {
        python: "py", workerPath: "w", spawnFn, reason: async (o) => {
          expect(o.evidence).toEqual([]);
          expect(o.direct).toBe(true);
          o.onDelta("hello");
          return { text: "hello", usage: {} };
        },
      }
    });
    expect(events.map((e) => e.type)).toEqual(["agent_start", "reasoning_start", "answer_delta", "done"]);
  });
  const sessionReply = (pass2: boolean, id: string): string =>
    `${JSON.stringify(
      pass2
        ? {
          id,
          objective: "q",
          sessionId: "s1",
          evidence: [{ id: "ev:1", content: "fact" }, { id: "ev:2", content: "gap filled" }],
          nodes: [
            { node_id: "n1", question: "q", status: "blocked", depends_on: [] },
            { node_id: "n2", question: "gap", status: "resolved", depends_on: [] },
          ],
          decisions: [{ decision_id: "d2" }],
          unresolved: [],
          incomplete_guard: false,
          toolExecutions: [],
          toolCalls: [],
          failures: {},
          escalations: 0,
          escalated: false,
        }
        : {
          id,
          objective: "q",
          sessionId: "s1",
          evidence: [{ id: "ev:1", content: "fact" }],
          nodes: [{ node_id: "n1", question: "q", status: "blocked", depends_on: [] }],
          decisions: [{ decision_id: "d1" }],
          unresolved: ["n1"],
          incomplete_guard: true,
          toolExecutions: [],
          toolCalls: [],
          failures: {},
          escalations: 1,
          escalated: true,
        },
    )}\n`;
  const emptyReply = (id: string): string =>
    `${JSON.stringify({ id, objective: "q", evidence: [], nodes: [], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
  type Outcome = { text?: string; claims?: FinalClaim[]; confidence?: number; missingEvidence?: string; usage?: KernelReasonResult["usage"] } | Error;
  type Call = { pass: number; input: KernelReasonInput };
  type GateVerdict = { verdict?: string; support_answer?: string; error?: string };
  // Drives runKernelAgent with a scripted reasoner. Stockbot settles last, so
  // output order never follows completion order; every call emits a draft delta.
  // Gate ops (assess_final_*) answer inline from opts.gate (default yes); run
  // bodies still record in order for pass-2 assertions.
  async function runFinal(opts: {
    reply?: (pass2: boolean, id: string, body?: Record<string, unknown>) => string;
    decide?: (persona: Persona, pass: number, attempt: number) => Outcome;
    gate?: (body: Record<string, unknown>) => GateVerdict;
    personas?: readonly Persona[];
    deadlineAt?: number;
    signal?: AbortSignal;
    ticker?: string;
    currentChartTicker?: string;
    watchlist?: readonly string[];
    // Injects only spawnFn+reason so the router falls back to its default python/worker path.
    spawnOnly?: boolean;
  }): Promise<{ events: AgentEvent[]; calls: Call[]; bodies: Record<string, unknown>[]; gates: Record<string, unknown>[]; maxInFlight: number; runTimeouts: (number | undefined)[]; spawns: { cmd: string; args: string[] }[] }> {
    process.env.OPENCODE_MODEL = "test-model";
    const bodies: Record<string, unknown>[] = [];
    const gates: Record<string, unknown>[] = [];
    const gateFn = opts.gate;
    const spawns: { cmd: string; args: string[] }[] = [];
    const spawn: KernelSpawn = (cmd, args): KernelChild => {
      spawns.push({ cmd, args });
      const c = new FakeChild();
      const innerReply = opts.reply ?? ((_p: boolean, id: string) => graphReply(id));
      // bodies.at(-1) is the run body being answered: it records before the reply.
      c.reply = (id: string): string => innerReply(bodies.filter((b) => b.op === "run").length > 1, id, bodies.at(-1));
      const origWrite = c.stdin.write;
      c.stdin.write = (data: string, cb?: (err?: Error | null) => void): void => {
        const body: unknown = JSON.parse(data);
        if (body && typeof body === "object" && "op" in body && typeof body.op === "string" && body.op.startsWith("assess_final")) {
          const rec = body as Record<string, unknown>;
          gates.push(rec);
          const verdict: GateVerdict = gateFn?.(rec) ?? { support_answer: "yes" };
          queueMicrotask(() => c.emitStdout(`${JSON.stringify({ id: rec.id, ...verdict })}\n`));
        } else {
          bodies.push(body as Record<string, unknown>);
          origWrite.call(c.stdin, data, cb);
        }
        cb?.(null);
      };
      queueMicrotask(() => c.emitStdout('{"type":"ready"}\n'));
      return c;
    };
    const events: AgentEvent[] = [];
    const calls: Call[] = [];
    let pass = 0;
    let inFlight = 0;
    let maxInFlight = 0;
    const attempts: Record<string, number> = {};
    // Records the worker timeout of each research run; the call itself stays real.
    const runTimeouts: (number | undefined)[] = [];
    const realCall = KernelRouter.prototype.call;
    KernelRouter.prototype.call = function(this: KernelRouter, body, callOpts) {
      if (body.op === "run") runTimeouts.push(callOpts?.timeoutMs);
      return realCall.call(this, body, callOpts);
    };
    try {
      await runKernelAgent(
        "q",
        (e) => {
          if (e.type === "reasoning_start") pass += 1;
          events.push(e);
        },
        {
          ...(opts.deadlineAt !== undefined ? { deadlineAt: opts.deadlineAt } : {}),
          // The kernel default is stockbot alone; this block drives the trio unless a test names its personas.
          personas: opts.personas ?? FINAL_PERSONAS,
          ...(opts.signal ? { signal: opts.signal } : {}),
          ...(opts.ticker !== undefined ? { ticker: opts.ticker } : {}),
          ...(opts.currentChartTicker !== undefined ? { currentChartTicker: opts.currentChartTicker } : {}),
          ...(opts.watchlist !== undefined ? { watchlist: opts.watchlist } : {}),
          deps: {
            ...(opts.spawnOnly ? {} : { python: "py", workerPath: "w" }),
            spawnFn: spawn,
            reason: async (o) => {
              const key = `${pass}:${o.persona}`;
              const attempt = (attempts[key] ?? 0) + 1;
              attempts[key] = attempt;
              calls.push({ pass, input: o });
              inFlight += 1;
              maxInFlight = Math.max(maxInFlight, inFlight);
              o.onDelta(`draft-${o.persona}`);
              for (let i = FINAL_PERSONAS.indexOf(o.persona); i < FINAL_PERSONAS.length; i++) await Promise.resolve();
              inFlight -= 1;
              const d: Outcome = opts.decide?.(o.persona, pass, attempt) ?? {};
              if (d instanceof Error) throw d;
              const text = d.text ?? `${o.persona} p${pass}`;
              // Default answers cite the first evidence row so the citation gate keeps the prose.
              const cite = o.evidence[0]?.id;
              return {
                text,
                usage: d.usage ?? {},
                claims: d.claims ?? (cite ? [{ text, claimType: "inference", evidenceIds: [cite] }] : []),
                ...(d.confidence !== undefined ? { confidence: d.confidence } : {}),
                ...(d.missingEvidence ? { missingEvidence: d.missingEvidence } : {}),
              };
            },
          },
        },
      );
    } finally {
      KernelRouter.prototype.call = realCall;
    }
    return { events, calls, bodies, gates, maxInFlight, runTimeouts, spawns };
  }
  const answers = (events: AgentEvent[]): string[] => events.flatMap((e) => (e.type === "answer_delta" ? [e.text] : []));
  const doneMetrics = (events: AgentEvent[]): Metrics => {
    const last = events.at(-1);
    if (last?.type !== "done") throw new Error("expected done last");
    return last.metrics;
  };
  const failedCategory = (events: AgentEvent[]): string | undefined => {
    const f = events.find((e) => e.type === "failed");
    return f?.type === "failed" ? f.category : undefined;
  };
  const runBodies = (bodies: Record<string, unknown>[]): Record<string, unknown>[] => bodies.filter((b) => b.op === "run");
  const label = (p: Persona): string => `── ${p.charAt(0).toUpperCase()}${p.slice(1)} ──`;
  // Trio answer: visible Bear and Bull drafts, then the Stockbot synthesis verdict.
  const trioAnswer = (pass: number): string =>
    (["bearbot", "bullbot", "stockbot"] as const).map((p) => `${label(p)}\n${p} p${pass}`).join("\n\n");

  test("trio drafts bear+bull concurrently then synthesizes stockbot with both drafts", async () => {
    const { events, calls, maxInFlight } = await runFinal({});
    // Drafts first (concurrent), synthesis last with both drafts attached.
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "stockbot"]);
    expect(maxInFlight).toBe(2);
    const first = calls[0]?.input;
    if (!first) throw new Error("expected reasoner calls");
    for (const c of calls) {
      expect(c.input.evidence).toBe(first.evidence);
      expect(c.input.prompt).toBe(first.prompt);
      expect(c.input.nodes).toBe(first.nodes);
      expect(c.input.decisions).toBe(first.decisions);
      expect(c.input.unresolved).toBe(first.unresolved);
      expect(c.input.incompleteGuard).toBe(true);
    }
    expect(first.prompt).toContain("OBJECTIVE");
    const synth = calls.at(-1)?.input;
    expect(synth?.drafts).toEqual([
      { persona: "bearbot", text: "bearbot p1" },
      { persona: "bullbot", text: "bullbot p1" },
    ]);
    expect(answers(events)).toEqual([trioAnswer(1)]);
    expect(events.filter((e) => e.type === "reasoning_start").length).toBe(1);
    expect(events.some((e) => e.type === "failed")).toBe(false);
    expect(doneMetrics(events).muse.calls).toBe(3);
  });

  test("trio retries exactly one failed draft once, then synthesizes with both drafts", async () => {
    const usage = { inputTokens: 5, outputTokens: 1 };
    const { events, calls } = await runFinal({
      currentChartTicker: "aapl",
      decide: (p, _pass, attempt) => (p === "bearbot" && attempt === 1 ? new Error("muse down") : { usage }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "bearbot", "stockbot"]);
    expect(calls.map((c) => c.input.ticker)).toEqual(["AAPL", "AAPL", "AAPL", "AAPL"]);
    expect(calls.at(-1)?.input.drafts).toEqual([
      { persona: "bearbot", text: "bearbot p1" },
      { persona: "bullbot", text: "bullbot p1" },
    ]);
    expect(answers(events)).toEqual([trioAnswer(1)]);
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(4);
    expect(m.muse.inputTokens).toBe(15);
    expect(m.muse.outputTokens).toBe(3);
  });

  test("trio retry exhausted fails closed with no prose", async () => {
    for (const blank of [false, true]) {
      const { events, calls } = await runFinal({
        decide: (p) => (p === "bullbot" ? (blank ? { text: "  \n " } : new Error("muse down")) : {}),
      });
      // Two drafts plus one retry, no synthesis call.
      expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "bullbot"]);
      expect(answers(events)).toEqual([]);
      expect(failedCategory(events)).toBe("provider_error");
      const failed = events.find((e) => e.type === "failed");
      expect(failed?.type === "failed" ? failed.message : "").toBe(blank ? "bullbot: empty response" : "bullbot: muse down");
      expect(doneMetrics(events).muse.calls).toBe(3);
    }
  });

  test("trio synthesis failure or empty fails closed with no prose", async () => {
    for (const synthOutcome of [new Error("muse down"), { text: "  " }]) {
      const { events, calls } = await runFinal({
        decide: (p) => (p === "stockbot" ? synthOutcome : { text: `${p} draft` }),
      });
      expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "stockbot"]);
      expect(calls.at(-1)?.input.drafts).toEqual([
        { persona: "bearbot", text: "bearbot draft" },
        { persona: "bullbot", text: "bullbot draft" },
      ]);
      expect(answers(events)).toEqual([]);
      expect(failedCategory(events)).toBe("provider_error");
      const failed = events.find((e) => e.type === "failed");
      expect(failed?.type === "failed" ? failed.message : "").toBe(
        synthOutcome instanceof Error ? "stockbot: muse down" : "stockbot: empty response",
      );
      expect(doneMetrics(events).muse.calls).toBe(3);
    }
  });

  test("trio gap regenerates per the same synthesis rule after one shared follow-up", async () => {
    for (const gapping of ["bearbot", "stockbot"] as const) {
      const { events, calls, bodies } = await runFinal({
        decide: (p, pass) => (pass === 1 && p === gapping ? { missingEvidence: "need doc X" } : { text: `${p} p${pass}` }),
      });
      const runs = runBodies(bodies);
      expect(runs.length).toBe(2);
      expect(JSON.stringify(runs[1])).toContain("need doc X");
      expect(calls.filter((c) => c.pass === 2).map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "stockbot"]);
      expect(calls.at(-1)?.input.drafts).toEqual([
        { persona: "bearbot", text: "bearbot p2" },
        { persona: "bullbot", text: "bullbot p2" },
      ]);
      expect(answers(events)).toEqual([trioAnswer(2)]);
      expect(doneMetrics(events).muse.calls).toBe(6);
    }
  });

  test("partial subsets never retry a failed persona", async () => {
    for (let mask = 1; mask < (1 << FINAL_PERSONAS.length) - 1; mask++) {
      const subset = FINAL_PERSONAS.filter((_, i) => (mask >> i) & 1);
      const first = subset[0];
      if (!first) continue;
      const { events, calls } = await runFinal({
        personas: subset,
        decide: (p) => (p === first ? new Error("muse down") : {}),
      });
      expect(calls.map((c) => c.input.persona)).toEqual(subset);
      for (const c of calls) expect(c.input.drafts).toBeUndefined();
      expect(answers(events)).toEqual([]);
      expect(doneMetrics(events).muse.calls).toBe(subset.length);
    }
  });

  test("every partial subset calls only its personas with no synthesis and no retry", async () => {
    for (let mask = 1; mask < 1 << FINAL_PERSONAS.length; mask++) {
      const subset = FINAL_PERSONAS.filter((_, i) => (mask >> i) & 1);
      if (subset.length === FINAL_PERSONAS.length) continue;
      const { events, calls } = await runFinal({ personas: subset });
      expect(calls.map((c) => c.input.persona)).toEqual(subset);
      for (const c of calls) expect(c.input.drafts).toBeUndefined();
      expect(answers(events)).toEqual([subset.map((p) => `${label(p)}\n${p} p1`).join("\n\n")]);
      expect(doneMetrics(events).muse.calls).toBe(subset.length);
    }
  });

  test("partial gap triggers exactly one shared follow-up with combined, deduplicated gaps", async () => {
    const gapText: Record<Persona, string> = { stockbot: "doc A", bearbot: "doc A", bullbot: "doc B" };
    const subset: Persona[] = ["stockbot", "bearbot"];
    for (let mask = 0; mask < 1 << subset.length; mask++) {
      const gapping = subset.filter((_, i) => (mask >> i) & 1);
      const { events, calls, bodies } = await runFinal({
        personas: subset,
        decide: (p, pass) => (pass === 1 && gapping.includes(p) ? { missingEvidence: gapText[p] } : {}),
      });
      const seen: Record<string, true> = {};
      const expected = gapping.map((p) => gapText[p]).filter((g) => (seen[g] ? false : (seen[g] = true))).join("; ");
      const runs = runBodies(bodies);
      expect(runs.length).toBe(gapping.length > 0 ? 2 : 1);
      if (gapping.length > 0) expect(runs[1]?.prompt).toBe(`q\nStill missing: ${expected}`);
      expect(events.filter((e) => e.type === "progress" && e.stage === "evidence_gap").length).toBe(gapping.length > 0 ? 1 : 0);
      const finalPass = gapping.length > 0 ? 2 : 1;
      expect(calls.filter((c) => c.pass === finalPass).map((c) => c.input.persona)).toEqual(subset);
      expect(answers(events)).toEqual([subset.map((p) => `${label(p)}\n${p} p${finalPass}`).join("\n\n")]);
    }
  });

  test("shared follow-up reuses the session, regenerates the trio on merged evidence, and sums metrics", async () => {
    const usage = { inputTokens: 10, outputTokens: 2, cachedTokens: 1 };
    const { events, calls, bodies } = await runFinal({
      reply: sessionReply,
      decide: (p, pass) => (pass === 1 && p === "bearbot" ? { missingEvidence: "need doc X", usage } : { usage }),
    });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(2);
    expect(runs[1]?.sessionId).toBe("s1");
    expect(runs[1]?.gap).toBe("need doc X");
    const second = calls.filter((c) => c.pass === 2);
    expect(second.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "stockbot"]);
    for (const c of second) {
      expect(c.input.evidence.map((e) => e.id)).toEqual(["ev:1", "ev:2"]);
      expect(c.input.unresolved).toEqual([]);
      expect(c.input.nodes.length).toBe(2);
    }
    expect(second.at(-1)?.input.drafts).toEqual([
      { persona: "bearbot", text: "bearbot p2" },
      { persona: "bullbot", text: "bullbot p2" },
    ]);
    expect(answers(events)).toEqual([trioAnswer(2)]);
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(6);
    expect(m.muse.inputTokens).toBe(60);
    expect(m.muse.outputTokens).toBe(12);
    expect(m.muse.cachedTokens).toBe(6);
    expect(m.evidence.count).toBe(2);
  });

  test("intake evidence starts a brief first; the full answer follows below a divider", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const { events, calls } = await runFinal({
      personas: ["stockbot"],
      reply: (_pass2, id) =>
        progress(id, "intake_evidence", { evidence: [{ id: "ev:0", content: "intake fact" }] }) +
        sessionGraphReply(id),
    });
    const brief = calls.find((c) => c.input.direct === false && c.input.persona === "stockbot" && !("drafts" in c.input));
    expect(brief?.input.evidence.map((e) => e.id)).toEqual(["ev:0"]);
    const deltas = answers(events);
    // Buffered brief: one delta with the stripped text, not streamed draft fragments.
    expect(deltas[0]).toContain("stockbot p");
    expect(deltas.at(-1)).toMatch(/^── Full answer ──\n/);
  });

  test("a brief over the full evidence ends the stream with no second answer", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const reply = (id: string): string =>
      `${JSON.stringify({ id, objective: "q", sessionId: "s1", evidence: [{ id: "ev:0", content: "intake fact" }], nodes: [], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
    const { events } = await runFinal({
      personas: ["stockbot"],
      reply: (_pass2, id) => progress(id, "intake_evidence", { evidence: [{ id: "ev:0", content: "intake fact" }] }) + reply(id),
    });
    expect(events.filter((e) => e.type === "answer_delta").length).toBe(1);
    expect(events.map((e) => e.type).slice(-2)).toEqual(["offer_trio", "done"]);
  });

  test("a gated research answer emits its kept claims once, right before done", async () => {
    const cited: FinalClaim = { text: "stockbot p1", claimType: "observed_fact", evidenceIds: ["ev:1"] };
    const open: FinalClaim = { text: "p1", claimType: "unknown", evidenceIds: [] };
    const { events } = await runFinal({ personas: ["stockbot"], reply: (_p, id) => sessionGraphReply(id), decide: () => ({ claims: [cited, open] }) });
    expect(events.map((e) => e.type).slice(-3)).toEqual(["offer_trio", "claims", "done"]);
    expect(events.filter((e) => e.type === "claims")).toEqual([{ type: "claims", claims: [cited, open] }]);
    // An answer without an inventory emits no claims event.
    const bare = await runFinal({ personas: ["stockbot"], reply: (_p, id) => sessionGraphReply(id), decide: () => ({ claims: [] }) });
    expect(bare.events.at(-1)?.type).toBe("done");
    expect(bare.events.some((e) => e.type === "claims")).toBe(false);
  });

  test("done timings sum worker progress phases, skip malformed values, and omit unmeasured phases", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const { events } = await runFinal({
      reply: (_pass2, id) =>
        progress(id, "intake_done", { wall_ms: 100, settle_ms: 10 }) +
        progress(id, "reasoner_done", { proposals: 2, ms: 40 }) +
        progress(id, "intake_done", { wall_ms: 20, settle_ms: 5 }) +
        progress(id, "intake_done", { wall_ms: "7", settle_ms: null }) +
        progress(id, "nodes_done", { ms: 300 }) +
        graphReply(id),
    });
    expect(answers(events)).toEqual([trioAnswer(1)]);
    // routeMs and firstTextMs belong to the route; no gap means no pass2Ms.
    expect(doneMetrics(events).timings).toEqual({
      intakeMs: 120, admitMs: 15, reasonerMs: 40, nodesMs: 300, personasMs: expect.any(Number), gatesMs: expect.any(Number),
    });
    const follow = await runFinal({ reply: sessionReply, decide: (p, pass) => (pass === 1 && p === "bearbot" ? { missingEvidence: "need doc X" } : {}) });
    expect(Object.keys(doneMetrics(follow.events).timings ?? {}).sort()).toEqual(["gatesMs", "pass2Ms", "personasMs"]);
  });

  test("a non-string or empty pass-1 sessionId uses the legacy follow-up and still merges new evidence", async () => {
    for (const sessionId of [7, ""]) {
      const { events, calls, bodies } = await runFinal({
        // Fake worker: an invalid sessionId on a follow-up returns no evidence, as a real worker rejects an unknown session.
        reply: (pass2, id, body) =>
          `${JSON.stringify(
            !pass2
              ? { id, objective: "q", sessionId, evidence: [{ id: "ev:1", content: "fact" }], nodes: [], decisions: [], unresolved: ["n1"], incomplete_guard: true, toolExecutions: [], toolCalls: [], failures: {}, escalations: 1, escalated: true }
              : body && "sessionId" in body
                ? { id, error: "unknown session" }
                : { id, objective: "q", evidence: [{ id: "ev:2", content: "gap filled" }], nodes: [], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false },
          )}\n`,
        decide: (p, pass) => (pass === 1 && p === "bearbot" ? { missingEvidence: "need doc X" } : {}),
      });
      const runs = runBodies(bodies);
      expect(runs.length).toBe(2);
      expect("sessionId" in (runs[1] ?? {})).toBe(false);
      expect(runs[1]?.prompt).toBe("q\nStill missing: need doc X");
      for (const c of calls.filter((x) => x.pass === 2)) expect(c.input.evidence.map((e) => e.id)).toEqual(["ev:1", "ev:2"]);
      expect(answers(events)).toEqual([trioAnswer(2)]);
      expect(doneMetrics(events).evidence.count).toBe(2);
    }
  });

  test("pass-2 unresolved list reaches every regenerated persona", async () => {
    const { calls, events } = await runFinal({
      reply: (pass2, id) =>
        `${JSON.stringify({ id, objective: "q", evidence: [{ id: "ev:1", content: "fact" }], nodes: [{ node_id: "n1", question: "q", status: "blocked", depends_on: [] }], decisions: [], unresolved: pass2 ? [] : ["n1"], incomplete_guard: !pass2, toolExecutions: [], toolCalls: [], failures: {}, escalations: pass2 ? 0 : 1, escalated: !pass2 })}\n`,
      decide: (p, pass) => (pass === 1 && p === "stockbot" ? { missingEvidence: "need doc X" } : {}),
    });
    expect(calls.map((c) => [c.pass, c.input.unresolved])).toEqual([
      [1, ["n1"]],
      [1, ["n1"]],
      [1, ["n1"]],
      [2, []],
      [2, []],
      [2, []],
    ]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("a failed persona on either pass fails the report with no prose", async () => {
    for (const failPass of [1, 2]) {
      const { events, bodies } = await runFinal({
        decide: (p, pass) => {
          if (pass === 1 && failPass === 2 && p === "stockbot") return { missingEvidence: "need doc X" };
          return pass === failPass && p === "bearbot" ? new Error("muse down") : {};
        },
      });
      expect(answers(events)).toEqual([]);
      expect(failedCategory(events)).toBe("provider_error");
      const failed = events.find((e) => e.type === "failed");
      expect(failed?.type === "failed" ? failed.message : "").toBe("bearbot: muse down");
      expect(runBodies(bodies).length).toBe(failPass);
      const m = doneMetrics(events);
      expect(m.muse.calls).toBe(3 * failPass);
      expect(m.failures.provider_error).toBe(1);
    }
  });

  test("an empty or whitespace persona draft fails the report with no prose, usage kept", async () => {
    const usage = { inputTokens: 5, outputTokens: 1 };
    for (const blank of ["", "  \n\t "]) {
      const { events, bodies } = await runFinal({
        personas: ["stockbot", "bearbot"],
        decide: (p) => (p === "bearbot" ? { text: blank, usage } : { usage }),
      });
      expect(answers(events)).toEqual([]);
      expect(failedCategory(events)).toBe("provider_error");
      const failed = events.find((e) => e.type === "failed");
      expect(failed?.type === "failed" ? failed.message : "").toBe("bearbot: empty response");
      expect(runBodies(bodies).length).toBe(1);
      const m = doneMetrics(events);
      expect(m.muse.calls).toBe(2);
      expect(m.muse.inputTokens).toBe(10);
      expect(m.muse.outputTokens).toBe(2);
      expect(m.failures.provider_error).toBe(1);
    }
  });

  test("duplicate evidence and an unresolved second gap answer through the final gates with no third pass", async () => {
    // graphReply returns the same ev:1 on both passes: the follow-up adds no evidence.
    const { events, calls, bodies, gates } = await runFinal({
      decide: (p) => (p === "bullbot" ? { missingEvidence: "still need doc X" } : {}),
    });
    expect(answers(events)).toEqual([trioAnswer(2)]);
    expect(events.filter((e) => e.type === "reasoning_start").length).toBe(2);
    expect(events.filter((e) => e.type === "progress" && e.stage === "evidence_gap").length).toBe(1);
    expect(calls.length).toBe(6);
    expect(runBodies(bodies).length).toBe(2);
    // Gates run once, on the final pass only.
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_response", "assess_final_response"]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(6);
    expect(m.evidence.count).toBe(1);
    expect(Object.keys(m.failures)).not.toContain("incomplete_evidence");
  });

  // Skip note the answer carries when a first gap leaves too little request wall.
  const skipNote = (gap: string): string => `\n\nMissing evidence (not researched: under ${PASS2_MIN_WALL_MS / 1000}s of request time left): ${gap}`;

  test("a first gap with under 20s of request wall left answers pass 1 with the missing evidence and no follow-up", async () => {
    const deadlineAt = Date.now() + 5_000;
    // 19.999s left, then past the wall: both skip. The clock stays frozen at the set time.
    for (const resumedAt of [deadlineAt - PASS2_MIN_WALL_MS + 1, deadlineAt + 60_000]) {
      try {
        const { events, bodies, gates } = await runFinal({
          deadlineAt,
          decide: (p, pass) => {
            if (pass === 1) setSystemTime(new Date(resumedAt));
            return pass === 1 && p === "bearbot" ? { missingEvidence: "need doc X" } : {};
          },
        });
        expect(runBodies(bodies).length).toBe(1);
        expect(events.filter((e) => e.type === "reasoning_start").length).toBe(1);
        expect(events.some((e) => e.type === "progress" && e.stage === "evidence_gap")).toBe(false);
        expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_response", "assess_final_response"]);
        expect(answers(events)).toEqual([trioAnswer(1) + skipNote("need doc X")]);
        expect(events.some((e) => e.type === "failed")).toBe(false);
        expect(doneMetrics(events).timings?.pass2Ms).toBeUndefined();
      } finally {
        setSystemTime();
      }
    }
  });

  test("a first gap with exactly 20s of request wall left still runs one follow-up with a fresh wall", async () => {
    const deadlineAt = Date.now() + 5_000;
    const resumedAt = deadlineAt - PASS2_MIN_WALL_MS;
    try {
      const { events, bodies, runTimeouts } = await runFinal({
        deadlineAt,
        decide: (p, pass) => {
          if (pass === 1) setSystemTime(new Date(resumedAt));
          return pass === 1 && p === "bearbot" ? { missingEvidence: "need doc X" } : {};
        },
      });
      const runs = runBodies(bodies);
      expect(runs.length).toBe(2);
      expect(runs[0]?.deadlineAt).toBe(deadlineAt);
      expect(runs[1]?.deadlineAt).toBe(resumedAt + REQUEST_WALL_MS);
      expect(runTimeouts[1]).toBe(REQUEST_WALL_MS);
      expect(answers(events)).toEqual([trioAnswer(2)]);
      expect(events.some((e) => e.type === "failed")).toBe(false);
    } finally {
      setSystemTime();
    }
  });

  test("an exhausted gap keeps supported facts and drops unsupported missing-data claims via the source gates", async () => {
    const supported = "Revenue was $10M.";
    const unsupported = "Q3 margin was 40%.";
    const { events, bodies, gates } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({
        text: `${supported} ${unsupported}\nMissing-Evidence: Q3 margin filing`,
        claims: [
          { text: supported, claimType: "observed_fact", evidenceIds: ["ev:1"] },
          { text: unsupported, claimType: "observed_fact", evidenceIds: ["ev:1"] },
        ],
        missingEvidence: "Q3 margin filing",
      }),
      gate: (body) => {
        if (body.op !== "assess_final_claim") return { support_answer: "no" };
        return (body.claim as Record<string, unknown>).text === supported ? { support_answer: "yes" } : { support_answer: "no" };
      },
    });
    expect(runBodies(bodies).length).toBe(2);
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim", "assess_final_claim"]);
    const out = answers(events);
    expect(out.length).toBe(1);
    expect(out[0]).toContain(supported);
    expect(out[0]).not.toContain(unsupported);
    expect(out[0]).toContain("Missing-Evidence: Q3 margin filing");
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("an exhausted gap with an unknown evidence reference still fails closed with no prose", async () => {
    const { events, bodies } = await runFinal({
      decide: (p, pass) =>
        pass === 2 && p === "bearbot"
          ? { missingEvidence: "doc X", claims: [{ text: "bearbot p2", claimType: "inference", evidenceIds: ["ev:9"] }] }
          : { missingEvidence: "doc X" },
    });
    expect(runBodies(bodies).length).toBe(2);
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
    const failed = events.find((e) => e.type === "failed");
    expect(failed?.type === "failed" ? failed.message : "").toContain("unknown evidence id: ev:9");
  });

  test("property: a first gap runs exactly one fresh follow-up iff 20s of request wall remain, and never a third pass", async () => {
    await fc.assert(
      fc.asyncProperty(fc.integer({ min: -24 * 3_600_000, max: 600_000 }), fc.boolean(), fc.boolean(), async (leftMs, newEvidence, gapAgain) => {
        const deadlineAt = Date.now() + 1_000;
        const resumedAt = deadlineAt - leftMs;
        const follows = leftMs >= PASS2_MIN_WALL_MS;
        try {
          const { events, bodies, runTimeouts } = await runFinal({
            deadlineAt,
            reply: newEvidence ? sessionReply : (_p, id) => graphReply(id),
            decide: (p, pass) => {
              if (pass === 1) setSystemTime(new Date(resumedAt));
              return (pass === 1 || gapAgain) && p === "bullbot" ? { missingEvidence: "need doc X" } : {};
            },
          });
          const runs = runBodies(bodies);
          expect(runs.length).toBe(follows ? 2 : 1);
          expect(runs[0]?.deadlineAt).toBe(deadlineAt);
          expect(events.filter((e) => e.type === "reasoning_start").length).toBe(follows ? 2 : 1);
          expect(events.some((e) => e.type === "failed")).toBe(false);
          if (!follows) {
            expect(answers(events)).toEqual([trioAnswer(1) + skipNote("need doc X")]);
            expect(doneMetrics(events).evidence.count).toBe(1);
            return;
          }
          expect(runs[1]?.deadlineAt).toBe(resumedAt + REQUEST_WALL_MS);
          expect(runTimeouts[1]).toBe(REQUEST_WALL_MS);
          if (newEvidence) expect(runs[1]?.sessionId).toBe("s1");
          expect(answers(events)).toEqual([trioAnswer(2)]);
          expect(doneMetrics(events).evidence.count).toBe(newEvidence ? 2 : 1);
        } finally {
          setSystemTime();
        }
      }),
      { seed: 20261011, numRuns: 40 },
    );
  });

  test("drafts, a failed-draft retry, and stockbot synthesis finish past the research wall with the request signal", async () => {
    const deadlineAt = Date.now() + 600_000;
    const signal = new AbortController().signal;
    const seenAt: number[] = [];
    try {
      const { events, calls } = await runFinal({
        deadlineAt,
        signal,
        decide: (p, _pass, attempt) => {
          // Research settled under the wall; the clock now sits past it for every persona call.
          setSystemTime(new Date(deadlineAt + 60_000));
          seenAt.push(Date.now());
          return p === "bearbot" && attempt === 1 ? new Error("muse down") : {};
        },
      });
      expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "bearbot", "stockbot"]);
      expect(calls.at(-1)?.input.drafts).toEqual([
        { persona: "bearbot", text: "bearbot p1" },
        { persona: "bullbot", text: "bullbot p1" },
      ]);
      expect(seenAt.length).toBe(4);
      for (const t of seenAt) expect(t).toBeGreaterThan(deadlineAt);
      for (const c of calls) {
        expect("deadlineAt" in c.input).toBe(false);
        expect(c.input.signal).toBe(signal);
      }
      expect(answers(events)).toEqual([trioAnswer(1)]);
      expect(events.some((e) => e.type === "failed")).toBe(false);
    } finally {
      setSystemTime();
    }
  });

  test("the first run carries the request wall and the follow-up carries its own fresh wall", async () => {
    const deadlineAt = Date.now() + 600_000;
    const before = Date.now();
    const { bodies, runTimeouts } = await runFinal({
      deadlineAt,
      decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
    });
    const after = Date.now();
    const runs = runBodies(bodies);
    expect(runs.length).toBe(2);
    expect(runs[0]?.deadlineAt).toBe(deadlineAt);
    const fresh = Number(runs[1]?.deadlineAt);
    expect(fresh).toBeGreaterThanOrEqual(before + REQUEST_WALL_MS);
    expect(fresh).toBeLessThanOrEqual(after + REQUEST_WALL_MS);
    expect(runTimeouts[1]).toBe(REQUEST_WALL_MS);
    for (const b of runs) expect("deadlineMs" in b).toBe(false);
  });

  test("empty graph answers direct through every selected persona", async () => {
    const { events, calls } = await runFinal({ reply: (_p, id) => emptyReply(id), personas: ["stockbot", "bullbot"] });
    expect(calls.map((c) => c.input.persona)).toEqual(["stockbot", "bullbot"]);
    for (const c of calls) {
      expect(c.input.direct).toBe(true);
      expect(c.input.escalated).toBe(false);
      expect(c.input.prompt).toBe("q");
      expect(c.input.evidence).toEqual([]);
    }
    expect(answers(events)).toEqual([`${label("stockbot")}\nstockbot p1\n\n${label("bullbot")}\nbullbot p1`]);
    expect(doneMetrics(events).muse.calls).toBe(2);
  });
  test("an empty-graph direct answer skips the source gates even when the gate service rejects", async () => {
    const { events, gates } = await runFinal({
      reply: (_p, id) => emptyReply(id),
      personas: ["stockbot", "bullbot"],
      decide: (p) => ({ claims: [{ text: `${p} p1`, claimType: "unknown", evidenceIds: [] }] }),
      gate: () => ({ error: "gate down" }),
    });
    expect(gates).toEqual([]);
    expect(answers(events)).toEqual([`${label("stockbot")}\nstockbot p1\n\n${label("bullbot")}\nbullbot p1`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });
  test("whole-response pass gates every persona with one op per persona", async () => {
    const claim = (p: Persona): FinalClaim => ({ text: `${p} p1`, claimType: "inference", evidenceIds: ["ev:1"] });
    const { events, gates } = await runFinal({
      decide: (p) => ({ claims: [claim(p)], confidence: 0.9 }),
    });
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_response", "assess_final_response"]);
    for (const g of gates) {
      expect(g.evidence).toEqual([{ id: "ev:1", content: "fact" }]);
      expect(Array.isArray(g.claims)).toBe(true);
      expect((g.claims as unknown[])[0]).toMatchObject({ claim_type: "inference", evidence_ids: ["ev:1"] });
    }
    expect(answers(events)).toEqual([trioAnswer(1)]);
  });

  test("gate bodies carry the run session id on both ops", async () => {
    const claim = "bearbot p1";
    const { gates } = await runFinal({
      personas: ["bearbot"],
      reply: (_p, id) => sessionGraphReply(id),
      decide: () => ({ text: `keep ${claim}`, claims: [{ text: claim, claimType: "inference", evidenceIds: ["ev:1"] }] }),
      gate: (body) => (body.op === "assess_final_response" ? { support_answer: "no" } : { support_answer: "yes" }),
    });
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim"]);
    // The worker keys gates by sessionId; without it, every gate fails closed on the numeric request id.
    for (const g of gates) expect(g.sessionId).toBe("rs:live");
  });

  test("a gate error rejects a claim instead of keeping its ungraded text", async () => {
    const claim = "bearbot claim here";
    const { events, gates } = await runFinal({
      personas: ["bearbot"],
      reply: (_p, id) => sessionGraphReply(id),
      decide: () => ({ text: `preamble ${claim} tail`, claims: [{ text: claim, claimType: "inference", evidenceIds: ["ev:1"] }] }),
      gate: () => ({ error: "unknown session_id: '39'" }),
    });
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim"]);
    const out = answers(events);
    expect(out.length).toBe(1);
    expect(out[0]).toContain("preamble");
    expect(out[0]).not.toContain(claim);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("a gate error that leaves no prose fails closed as provider_error", async () => {
    const { events } = await runFinal({
      personas: ["bearbot"],
      reply: (_p, id) => sessionGraphReply(id),
      decide: () => ({
        text: "bearbot claim here",
        claims: [{ text: "bearbot claim here", claimType: "inference", evidenceIds: ["ev:1"] }],
      }),
      gate: () => ({ error: "unknown session_id: '39'" }),
    });
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
  });

  test("whole-response reject runs per-claim gates and deletes only rejected substrings", async () => {
    const kept = "bearbot keeps this";
    const cut = "bearbot loses this";
    const { events, gates } = await runFinal({
      decide: (p) =>
        p === "bearbot"
          ? {
            text: `${kept} ${cut}`,
            claims: [
              { text: kept, claimType: "conditional_risk", evidenceIds: ["ev:1"] },
              { text: cut, claimType: "attributed_report", evidenceIds: ["ev:1"] },
            ],
          }
          : {},
      gate: (body) => {
        if (body.op === "assess_final_response" && body.persona === "bearbot") return { support_answer: "no" };
        if (body.op === "assess_final_claim") {
          const claim = body.claim as Record<string, unknown>;
          return claim.text === kept ? { support_answer: "yes" } : { support_answer: "no" };
        }
        return { support_answer: "yes" };
      },
    });
    expect(gates.filter((g) => g.op === "assess_final_response").length).toBe(3);
    expect(gates.filter((g) => g.op === "assess_final_claim").length).toBe(2);
    const out = answers(events);
    expect(out.length).toBe(1);
    expect(out[0]).toContain(kept);
    expect(out[0]).not.toContain(cut);
    expect(out[0]).toContain("bullbot p1");
  });

  test("an rn: inventory citation retries once with the correction and answers supported", async () => {
    const usage = { inputTokens: 5, outputTokens: 1 };
    const { events, calls } = await runFinal({
      personas: ["bearbot"],
      currentChartTicker: "msft",
      decide: (_p, _pass, attempt) =>
        attempt === 1
          ? { text: "bearbot p1", claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["rn:56c1365b-followup"] }], usage }
          : { text: "bearbot fixed", claims: [{ text: "bearbot fixed", claimType: "inference", evidenceIds: ["ev:1"] }], usage },
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
    expect(calls.map((c) => c.input.ticker)).toEqual(["MSFT", "MSFT"]);
    expect(calls[1]?.input.prompt).toContain("CORRECTION");
    expect(calls[1]?.input.prompt).toContain("ev:1");
    expect(calls[1]?.input.prompt).toContain("rn:56c1365b-followup");
    expect(calls[1]?.input.evidence.map((e) => e.id)).toEqual(["ev:1"]);
    expect(answers(events)).toEqual([`${label("bearbot")}\nbearbot fixed`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(2);
    expect(m.muse.inputTokens).toBe(10);
    expect(m.muse.outputTokens).toBe(2);
  });

  test("a still-invalid correction fails closed with no prose and no second retry", async () => {
    const { events, calls } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["ev:9"] }] }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
    const failed = events.find((e) => e.type === "failed");
    expect(failed?.type === "failed" ? failed.message : "").toContain("unknown evidence id: ev:9");
    expect(doneMetrics(events).muse.calls).toBe(2);
  });

  test("a non-verbatim inventory citation retries once with the correction and answers supported", async () => {
    const { events, calls } = await runFinal({
      personas: ["bearbot"],
      decide: (_p, _pass, attempt) =>
        attempt === 1
          ? { text: "bearbot p1", claims: [{ text: "bearbot paraphrase", claimType: "inference", evidenceIds: ["ev:1"] }] }
          : { text: "bearbot fixed", claims: [{ text: "bearbot fixed", claimType: "inference", evidenceIds: ["ev:1"] }] },
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
    expect(calls[1]?.input.prompt).toContain("CORRECTION");
    expect(calls[1]?.input.prompt).toContain("verbatim prose substring");
    expect(calls[1]?.input.prompt).toContain("bearbot paraphrase");
    expect(calls[1]?.input.evidence.map((e) => e.id)).toEqual(["ev:1"]);
    expect(answers(events)).toEqual([`${label("bearbot")}\nbearbot fixed`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
    expect(doneMetrics(events).muse.calls).toBe(2);
  });

  test("a still-non-verbatim correction fails closed with no prose and no second retry", async () => {
    const { events, calls } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ text: "bearbot p1", claims: [{ text: "bearbot paraphrase", claimType: "inference", evidenceIds: ["ev:1"] }] }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
    const failed = events.find((e) => e.type === "failed");
    expect(failed?.type === "failed" ? failed.message : "").toContain("not a prose substring");
    expect(doneMetrics(events).muse.calls).toBe(2);
  });

  test("an uncited inventory defect retries nothing and fails closed", async () => {
    const { events, calls } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ claims: [{ text: "bearbot p1", claimType: "observed_fact", evidenceIds: [] }] }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot"]);
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
  });

  test("an inventory correction replaces only the affected persona", async () => {
    const { events, calls } = await runFinal({
      personas: ["stockbot", "bearbot"],
      decide: (p, _pass, attempt) =>
        p === "bearbot" && attempt === 1
          ? { claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["ev:9"] }] }
          : { text: `${p} fixed`, claims: [{ text: `${p} fixed`, claimType: "inference", evidenceIds: ["ev:1"] }] },
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["stockbot", "bearbot", "bearbot"]);
    expect(calls.filter((c) => c.input.persona === "stockbot").length).toBe(1);
    expect(answers(events)).toEqual([`${label("stockbot")}\nstockbot fixed\n\n${label("bearbot")}\nbearbot fixed`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("a corrected answer still passes the source gates before display", async () => {
    const kept = "bearbot corrected fact";
    const cut = "bearbot corrected guess";
    const { events, calls, gates } = await runFinal({
      personas: ["bearbot"],
      decide: (_p, _pass, attempt) =>
        attempt === 1
          ? { claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["ev:9"] }] }
          : {
            text: `${kept} ${cut}`,
            claims: [
              { text: kept, claimType: "observed_fact", evidenceIds: ["ev:1"] },
              { text: cut, claimType: "observed_fact", evidenceIds: ["ev:1"] },
            ],
          },
      gate: (body) => {
        if (body.op === "assess_final_claim") {
          return (body.claim as Record<string, unknown>).text === kept ? { support_answer: "yes" } : { support_answer: "no" };
        }
        return { support_answer: "no" };
      },
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
    const out = answers(events);
    expect(out.length).toBe(1);
    expect(out[0]).toContain(kept);
    expect(out[0]).not.toContain(cut);
    // The claims event carries only the claim the gates kept.
    expect(events.filter((e) => e.type === "claims")).toEqual([{ type: "claims", claims: [{ text: kept, claimType: "observed_fact", evidenceIds: ["ev:1"] }] }]);
    expect(gates.some((g) => g.op === "assess_final_claim")).toBe(true);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("direct answers never retry an unknown inventory id", async () => {
    const { events, calls, gates } = await runFinal({
      reply: (_p, id) => emptyReply(id),
      personas: ["stockbot"],
      decide: () => ({ claims: [{ text: "stockbot p1", claimType: "inference", evidenceIds: ["ev:9"] }] }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["stockbot"]);
    expect(gates).toEqual([]);
    expect(answers(events)).toEqual([`${label("stockbot")}\nstockbot p1`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("property: a correction attempt cites only known ids or fails closed", async () => {
    await fc.assert(
      fc.asyncProperty(fc.array(fc.constantFrom("ev:1", "ev:9", "rn:n1"), { minLength: 1, maxLength: 3 }), async (evidenceIds) => {
        const { events, calls } = await runFinal({
          personas: ["bearbot"],
          decide: (_p, _pass, attempt) =>
            attempt === 1
              ? { claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["rn:n1"] }] }
              : { text: "bearbot fixed", claims: [{ text: "bearbot fixed", claimType: "inference", evidenceIds }] },
        });
        expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
        if (evidenceIds.every((id) => id === "ev:1")) {
          expect(answers(events)).toEqual([`${label("bearbot")}\nbearbot fixed`]);
        } else {
          expect(answers(events)).toEqual([]);
          expect(failedCategory(events)).toBe("provider_error");
        }
      }),
      { seed: 20261011, numRuns: 20 },
    );
  });

  test("inventoryDefect rejects non-substrings, unknown ids, and uncited non-unknown claims", () => {
    const known = new Set(["ev:1"]);
    expect(inventoryDefect("prose", [{ text: "missing", claimType: "inference", evidenceIds: ["ev:1"] }], known)).toMatch(/substring/);
    expect(inventoryDefect("prose has it", [{ text: "has it", claimType: "inference", evidenceIds: ["ev:9"] }], known)).toMatch(/unknown evidence id/);
    expect(inventoryDefect("prose has it", [{ text: "has it", claimType: "observed_fact", evidenceIds: [] }], known)).toMatch(/uncited/);
    expect(inventoryDefect("prose has it", [{ text: "has it", claimType: "unknown", evidenceIds: [] }], known)).toBeUndefined();
  });

  test("citedOrThin keeps an answer with a surviving cited claim and thins every other answer", () => {
    const result = (claims: FinalClaim[]) => [{ persona: "bearbot" as const, text: "", claims }];
    const cited: FinalClaim = { text: "kept fact", claimType: "observed_fact", evidenceIds: ["ev:1"] };
    expect(citedOrThin("kept fact here", result([cited]))).toBe("kept fact here");
    for (const [text, claims] of [
      ["prose", []],
      ["prose", [{ text: "prose", claimType: "unknown", evidenceIds: [] }]],
      // The gates deleted the only cited claim from the prose.
      ["prose left", [cited]],
      ["prose", [{ ...cited, text: "" }]],
    ] as [string, FinalClaim[]][]) {
      const out = citedOrThin(text, result(claims));
      expect(out).toStartWith("Thin evidence:");
      expect(out).toContain("Sourced facts: none.");
      expect(out).toContain(`Inference (uncited):\n${text}`);
      expect(out).toContain("Uncertainty:");
    }
  });

  test("a research answer with no cited claim returns the thin-evidence form", async () => {
    const { events } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ text: "bearbot guess", claims: [{ text: "bearbot guess", claimType: "unknown", evidenceIds: [] }] }),
    });
    expect(answers(events)).toEqual([citedOrThin(`${label("bearbot")}\nbearbot guess`, [])]);
    expect(answers(events)[0]).toStartWith("Thin evidence:");
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("a cited claim the gates delete leaves the thin-evidence form", async () => {
    const kept = "bearbot uncited opinion";
    const cut = "bearbot cited fact";
    const { events } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({
        text: `${kept} ${cut}`,
        claims: [
          { text: kept, claimType: "unknown", evidenceIds: [] },
          { text: cut, claimType: "observed_fact", evidenceIds: ["ev:1"] },
        ],
      }),
      gate: (body) => (body.op === "assess_final_claim" && (body.claim as Record<string, unknown>).text === kept ? { support_answer: "yes" } : { support_answer: "no" }),
    });
    const out = answers(events);
    expect(out.length).toBe(1);
    expect(out[0]).toStartWith("Thin evidence:");
    expect(out[0]).toContain(kept);
    expect(out[0]).not.toContain(cut);
  });

  test("all claims rejected fails closed with no prose", async () => {
    const { events } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ text: "bearbot claim here", claims: [{ text: "bearbot claim here", claimType: "inference", evidenceIds: ["ev:1"] }] }),
      gate: (body) => (body.op === "assess_final_claim" ? { support_answer: "no" } : { support_answer: "no" }),
    });
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
  });

  test("selected ticker rides every run body with the prompt verbatim", async () => {
    const { bodies } = await runFinal({ ticker: "nvda" });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(1);
    expect(runs[0]?.prompt).toBe("q");
    expect(runs[0]?.ticker).toBe("NVDA");
  });

  test("selected ticker survives a same-session gap follow-up", async () => {
    const { bodies } = await runFinal({
      ticker: "nvda",
      decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
    });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(2);
    for (const run of runs) expect(run.ticker).toBe("NVDA");
    expect(runs[0]?.prompt).toBe("q");
  });

  test("invalid ticker drops from the run body", async () => {
    const { bodies } = await runFinal({ ticker: "SEC" });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(1);
    expect(runs[0]?.prompt).toBe("q");
    expect("ticker" in (runs[0] ?? {})).toBe(false);
  });

  test("absent or invalid ticker never rides the first or follow-up run body", async () => {
    for (const ticker of [undefined, "SEC", "1AB"]) {
      const { bodies } = await runFinal({
        ...(ticker !== undefined ? { ticker } : {}),
        decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
      });
      const runs = runBodies(bodies);
      expect(runs.length).toBe(2);
      expect(runs[0]?.prompt).toBe("q");
      expect(runs[1]?.prompt).toBe("q\nStill missing: need doc X");
      for (const run of runs) expect("ticker" in run).toBe(false);
    }
  });

  test("follow-up run body keeps op, prompt, and deadline beside the ticker", async () => {
    const { bodies } = await runFinal({
      ticker: " brk.b ",
      decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
    });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(2);
    expect(runs[1]?.op).toBe("run");
    expect(runs[1]?.prompt).toBe("q\nStill missing: need doc X");
    expect(typeof runs[1]?.deadlineAt).toBe("number");
    for (const run of runs) expect(run.ticker).toBe("BRK.B");
  });

  test("watchlist rides every run body and replaces the selected ticker", async () => {
    const { bodies } = await runFinal({
      ticker: "nvda",
      watchlist: ["NVDA", "QQQ"],
      decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
    });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(2);
    for (const run of runs) {
      expect(run.watchlist).toEqual(["NVDA", "QQQ"]);
      expect("ticker" in run).toBe(false);
      expect(run.prompt === "q" || run.prompt === "q\nStill missing: need doc X").toBe(true);
    }
  });

  test("watchlist coverage reaches every persona prompt, failures disclosed, and an all-failed list never answers direct", async () => {
    const rows = [
      { symbol: "NVDA", source: "sec_filings", tool: "list_sec_filings", status: "ok", as_of: "2026-10-08", evidence_id: "ev:nvda" },
      { symbol: "QQQ", source: "finra_threshold_list", tool: "get_threshold_securities", status: "no_records", error: "No data found for QQQ: thresholdList" },
      { symbol: "AAPL", source: "finra_short_sale_volume", tool: "get_reg_sho_volume", status: "timeout", error: "intake_timeout" },
      { symbol: 7, source: "junk", status: "ok" },
    ];
    const { calls } = await runFinal({
      watchlist: ["NVDA", "QQQ", "AAPL"],
      personas: ["stockbot"],
      reply: (_p, id) => `${JSON.stringify({ id, evidence: [], nodes: [], unresolved: [], toolCalls: [], needleDecisions: [], watchlist: rows })}\n`,
    });
    const prompt = calls[0]?.input.prompt ?? "";
    expect(calls[0]?.input.direct).toBe(false);
    expect(prompt).toContain("WATCHLIST COVERAGE");
    expect(prompt).toContain("NVDA sec_filings ok as_of=2026-10-08 evidence=ev:nvda");
    expect(prompt).toContain('QQQ finra_threshold_list no_records error="No data found for QQQ: thresholdList"');
    expect(prompt).toContain('AAPL finra_short_sale_volume timeout error="intake_timeout"');
    expect(prompt).not.toContain("junk");
  });

  test("default deadline sits one research wall in the future", async () => {
    const before = Date.now();
    const { bodies, runTimeouts } = await runFinal({});
    const after = Date.now();
    const first = Number(runBodies(bodies)[0]?.deadlineAt);
    expect(first).toBeGreaterThanOrEqual(before + REQUEST_WALL_MS);
    expect(first).toBeLessThanOrEqual(after + REQUEST_WALL_MS);
    expect(runTimeouts[0]).toBeGreaterThan(REQUEST_WALL_MS - 5_000);
  });

  test("injected python and worker path reach the spawner", async () => {
    const { spawns } = await runFinal({});
    expect(spawns.length).toBe(1);
    expect(spawns[0]).toEqual({ cmd: "py", args: ["w"] });
  });

  test("spawner-only deps use the injected spawner with the real repo worker", async () => {
    const { spawns, bodies } = await runFinal({ spawnOnly: true });
    expect(spawns.length).toBe(1);
    expect(runBodies(bodies).length).toBe(1);
    const worker = spawns[0]?.args[0] ?? "";
    expect(worker.endsWith(join("app", "research", "kernel_worker.py"))).toBe(true);
    expect(existsSync(worker)).toBe(true);
    const python = spawns[0]?.cmd ?? "";
    expect(python).toBe(join(worker, "..", "..", "..", "venv", "bin", "python"));
  });

  test("worker progress reaches the stream with detail only when the worker sent one", async () => {
    const progressLines = (id: string): string =>
      `${JSON.stringify({ type: "progress", id, stage: "p_bare" })}\n${JSON.stringify({ type: "progress", id, stage: "p_detail", detail: { calls: 3 } })}\n`;
    const { events } = await runFinal({
      reply: (_p, id) => progressLines(id) + graphReply(id),
      decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
    });
    const progress = events.filter((e) => e.type === "progress" && (e.stage === "p_bare" || e.stage === "p_detail"));
    // Two runs (first + follow-up), each relaying both lines.
    expect(progress.length).toBe(4);
    for (const e of progress) {
      if (e.type !== "progress") throw new Error("expected progress");
      if (e.stage === "p_bare") expect("detail" in e).toBe(false);
      else expect(e.detail).toEqual({ calls: 3 });
    }
  });

  test("validated chart context reaches direct answers without changing research scope", async () => {
    for (const [ticker, currentChartTicker, expected] of [
      ["nvda", " brk.b ", "BRK.B"],
      ["nvda", undefined, "NVDA"],
      ["nvda", "SEC", "NVDA"],
      [undefined, "msft", "MSFT"],
      [undefined, "not a ticker", undefined],
      [undefined, undefined, undefined],
    ] as const) {
      const { calls, bodies } = await runFinal({ ticker, currentChartTicker, personas: ["stockbot"], reply: (_p, id) => emptyReply(id) });
      expect(calls).toHaveLength(1);
      expect(calls[0]?.input.direct).toBe(true);
      expect(calls[0]?.input.ticker).toBe(expected);
      if (expected === undefined) expect(calls[0]?.input).not.toHaveProperty("ticker");
      const runs = runBodies(bodies);
      expect(runs[0]?.ticker).toBe(ticker?.toUpperCase());
      expect(runs[0]).not.toHaveProperty("currentChartTicker");
    }
  });

  test("chart progress updates every trio call after each research pass without changing research scope", async () => {
    const { events, calls, bodies } = await runFinal({
      ticker: "nvda",
      currentChartTicker: "aapl",
      reply: (pass2, id) => `${JSON.stringify({ type: "progress", id, stage: "chart_update", detail: { ticker: pass2 ? " brk.b " : " msft " } })}\n${sessionGraphReply(id)}`,
      decide: (persona, pass) => pass === 1 && persona === "bullbot" ? { missingEvidence: "need doc X" } : {},
    });
    expect(events.filter((e) => e.type === "chart_update")).toEqual([
      { type: "chart_update", ticker: "MSFT" },
      { type: "chart_update", ticker: "BRK.B" },
    ]);
    expect(events.some((e) => e.type === "progress" && e.stage === "chart_update")).toBe(false);
    expect(calls.map((c) => [c.pass, c.input.persona, c.input.ticker])).toEqual([
      [1, "bearbot", "MSFT"], [1, "bullbot", "MSFT"], [1, "stockbot", "MSFT"],
      [2, "bearbot", "BRK.B"], [2, "bullbot", "BRK.B"], [2, "stockbot", "BRK.B"],
    ]);
    const runs = runBodies(bodies);
    expect(runs).toHaveLength(2);
    for (const run of runs) expect(run.ticker).toBe("NVDA");
    expect(runs[1]?.sessionId).toBe("rs:live");
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("malformed chart progress and prose never replace the current chart context", async () => {
    const details: unknown[] = [
      undefined, null, [], "TSLA", 7, {}, { ticker: null }, { ticker: ["TSLA"] },
      { ticker: 7 }, { ticker: "" }, { ticker: "SEC" }, { ticker: "1AB" },
      { ticker: "switch to TSLA" }, { preview: '{"ticker":"TSLA"}' },
    ];
    const { events, calls } = await runFinal({
      ticker: "nvda",
      currentChartTicker: "aapl",
      reply: (_p, id) => details.map((detail) => `${JSON.stringify({ type: "progress", id, stage: "chart_update", detail })}\n`).join("")
        + `${JSON.stringify({ type: "progress", id, stage: "tool_done", detail: { preview: '{"ticker":"TSLA"}' } })}\n`
        + `${JSON.stringify({ id, chartTicker: "TSLA", evidence: [{ id: "ev:1", content: 'Chart updated to TSLA. {"ticker":"TSLA"}' }], toolCalls: [] })}\n`,
    });
    expect(events.filter((e) => e.type === "chart_update")).toEqual([]);
    expect(events.filter((e) => e.type === "progress")).toEqual([
      { type: "progress", stage: "tool_done", detail: { preview: '{"ticker":"TSLA"}' } },
    ]);
    expect(calls.map((c) => c.input.ticker)).toEqual(["AAPL", "AAPL", "AAPL"]);
  });

  test("widget progress relays only a canonical ticker and known widget type without changing chart context", async () => {
    const details: unknown[] = [
      { ticker: "AAPL", widgetType: "short-volume" },
      undefined, null, [], "AAPL", 7, {},
      { ticker: "AAPL" }, { widgetType: "short-volume" },
      { ticker: "aapl", widgetType: "short-volume" },
      { ticker: " AAPL ", widgetType: "short-volume" },
      { ticker: "SEC", widgetType: "short-volume" },
      { ticker: ["AAPL"], widgetType: "short-volume" },
      { ticker: "AAPL", widgetType: "unknown" },
      { ticker: "AAPL", widgetType: ["short-volume"] },
      { ticker: "AAPL", widgetType: null },
    ];
    const { events, calls, bodies } = await runFinal({
      ticker: "nvda",
      currentChartTicker: "aapl",
      reply: (_p, id) => details.map((detail) => `${JSON.stringify({ type: "progress", id, stage: "widget_update", detail })}\n`).join("")
        + `${sessionGraphReply(id)}`,
    });
    expect(events.filter((e) => e.type === "widget_update")).toEqual([
      { type: "widget_update", ticker: "AAPL", widgetType: "short-volume" },
    ]);
    expect(events.some((e) => e.type === "progress" && e.stage === "widget_update")).toBe(false);
    expect(calls.map((c) => c.input.ticker)).toEqual(["AAPL", "AAPL", "AAPL"]);
    expect(runBodies(bodies)[0]?.ticker).toBe("NVDA");
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("watchlist answers omit chart context through chart updates, trio retries, and follow-up", async () => {
    const { events, calls, bodies } = await runFinal({
      ticker: "nvda",
      currentChartTicker: "aapl",
      watchlist: ["NVDA", "QQQ"],
      reply: (_p, id) => `${JSON.stringify({ type: "progress", id, stage: "chart_update", detail: { ticker: "msft" } })}\n${sessionGraphReply(id)}`,
      decide: (persona, pass, attempt) => {
        if (persona === "bearbot" && attempt === 1) return new Error("retry draft");
        return pass === 1 && persona === "bullbot" ? { missingEvidence: "need doc X" } : {};
      },
    });
    expect(events.filter((e) => e.type === "chart_update")).toEqual([
      { type: "chart_update", ticker: "MSFT" }, { type: "chart_update", ticker: "MSFT" },
    ]);
    expect(calls).toHaveLength(8);
    for (const call of calls) expect(call.input).not.toHaveProperty("ticker");
    const runs = runBodies(bodies);
    expect(runs).toHaveLength(2);
    for (const run of runs) {
      expect(run.watchlist).toEqual(["NVDA", "QQQ"]);
      expect(run).not.toHaveProperty("ticker");
    }
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("applyFinalGates treats worker error as reject and deletes surgically", async () => {
    const ev = [{ id: "ev:1", source: "kernel", retrievedAt: "t", content: "fact" }];
    const gated = await applyFinalGates({
      results: [{ persona: "bearbot", text: "aaa KEEP bbb CUT ccc", claims: [{ text: "KEEP", claimType: "inference", evidenceIds: ["ev:1"] }, { text: "CUT", claimType: "inference", evidenceIds: ["ev:1"] }] }],
      evidence: ev,
      personaOrder: ["bearbot"],
      gate: async (body) => {
        if (body.op === "assess_final_response") return { error: "boom" };
        const claim = body.claim;
        const text = claim && typeof claim === "object" && "text" in claim ? claim.text : undefined;
        return text === "KEEP" ? { support_answer: "yes" } : { error: "boom" };
      },
    });
    if (!("text" in gated)) throw new Error("expected gated text");
    expect(gated.text).toContain("KEEP");
    expect(gated.text).not.toContain("CUT");
  });

  // Scripted gate replies for the parallel-gate tests: each persona's whole
  // response and each claim (`<persona>-c<j>`) has its own verdict.
  type GateScript = "yes" | "no" | "error" | "throw";
  const gateEvidence = [{ id: "ev:1", source: "kernel", retrievedAt: "t", content: "fact" }];
  const scriptedReply = async (v: GateScript) => {
    if (v === "throw") throw new Error("down");
    return v === "error" ? { error: "boom" } : { support_answer: v };
  };
  const scriptKey = (body: Record<string, unknown>): string => {
    const claim = body.claim;
    return claim && typeof claim === "object" && "text" in claim ? String(claim.text) : String(body.persona);
  };
  const gateResults = (personas: readonly Persona[], claimCounts: readonly number[], defectAt?: number) =>
    personas.map((p, i) => {
      const claims: FinalClaim[] = Array.from({ length: claimCounts[i] ?? 0 }, (_, j) => ({
        text: `${p}-c${j}`, claimType: "inference", evidenceIds: [defectAt === i && j === 0 ? "ev:9" : "ev:1"],
      }));
      return { persona: p, text: `${p} lead. ${claims.map((c) => c.text).join(" ")}`, claims };
    });

  test("applyFinalGates checks every persona and every rejected claim concurrently", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const ops: string[] = [];
    const gate: FinalGateCaller = async (body) => {
      ops.push(String(body.op));
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await Promise.resolve();
      inFlight -= 1;
      return scriptedReply(body.op === "assess_final_response" ? "no" : scriptKey(body).endsWith("-c0") ? "yes" : "no");
    };
    const personas: Persona[] = ["bearbot", "bullbot", "stockbot"];
    const gated = await applyFinalGates({ results: gateResults(personas, [2, 2, 2]), evidence: gateEvidence, personaOrder: personas, gate });
    expect(gated).toEqual({ text: personas.map((p) => `${label(p)}\n${p} lead. ${p}-c0`).join("\n\n") });
    // Three whole-response checks in flight at once, then all six claim checks.
    expect(ops.slice(0, 3)).toEqual(["assess_final_response", "assess_final_response", "assess_final_response"]);
    expect(maxInFlight).toBe(6);
  });

  test("property: parallel final gates give the serial verdict under any completion order", async () => {
    const script = fc.constantFrom<GateScript>("yes", "no", "error", "throw");
    await fc.assert(
      fc.asyncProperty(
        fc.subarray([...FINAL_PERSONAS], { minLength: 1 }),
        fc.array(fc.array(script, { maxLength: 3 }), { minLength: 3, maxLength: 3 }),
        fc.array(script, { minLength: 3, maxLength: 3 }),
        fc.option(fc.nat(2), { nil: undefined }),
        fc.array(fc.nat(6), { minLength: 16, maxLength: 16 }),
        async (personas, claimScripts, wholeScripts, defectAt, ticks) => {
          const verdicts = new Map<string, GateScript>();
          personas.forEach((p, i) => {
            verdicts.set(p, wholeScripts[i] ?? "yes");
            (claimScripts[i] ?? []).forEach((v, j) => verdicts.set(`${p}-c${j}`, v));
          });
          const replyFor = (body: Record<string, unknown>) => scriptedReply(verdicts.get(scriptKey(body)) ?? "yes");
          // Serial reference: one gate call at a time, in issue order.
          let chain: Promise<unknown> = Promise.resolve();
          const serial: FinalGateCaller = (body) => {
            const next = chain.then(() => replyFor(body));
            chain = next.catch(() => undefined);
            return next;
          };
          // Shuffled completion: each call settles after a scripted number of microtask ticks.
          let n = 0;
          const shuffled: FinalGateCaller = async (body) => {
            const wait = ticks[n++ % ticks.length] ?? 0;
            for (let t = 0; t < wait; t++) await Promise.resolve();
            return replyFor(body);
          };
          const results = gateResults(personas, claimScripts.map((c) => c.length), defectAt);
          const run = (gate: FinalGateCaller) => applyFinalGates({ results, evidence: gateEvidence, personaOrder: personas, gate });
          expect(await run(shuffled)).toEqual(await run(serial));
        },
      ),
      { seed: 20261019, numRuns: 200 },
    );
  });
});

describe("runKernelAgent worker replies", () => {
  // Deterministic reasoner: no test in this block may reach a provider.
  async function runAgent(
    events: AgentEvent[],
    spawnFn: KernelSpawn,
    personas?: readonly Persona[],
  ): Promise<void> {
    process.env.OPENCODE_MODEL = "test-model";
    await runKernelAgent("q", (e) => { events.push(e); }, {
      ...(personas ? { personas } : {}),
      deps: {
        python: "py",
        workerPath: "w",
        spawnFn,
        reason: async (o: KernelReasonInput) => ({ text: `${o.persona} answer`, usage: {} }),
      },
    });
  }
  function scriptedWorker(reply: (id: string) => string): KernelSpawn {
    return (): KernelChild => {
      const child = new FakeChild();
      child.reply = reply;
      queueMicrotask(() => child.emitStdout('{"type":"ready"}\n'));
      return child;
    };
  }
  function doneOf(events: AgentEvent[]): Metrics {
    const last = events.at(-1);
    if (last?.type !== "done") throw new Error("expected done last");
    return last.metrics;
  }
  function toolFailedOf(events: AgentEvent[]): [string, string] {
    const e = events.find((x) => x.type === "tool_failed");
    return e?.type === "tool_failed" ? [e.tool, e.category] : ["(none)", "(none)"];
  }

  test("a worker that dies before ready ends at done with the needle failure shape", async () => {
    const events: AgentEvent[] = [];
    const spawnFn = (): KernelChild => {
      const child = new FakeChild();
      queueMicrotask(() => child.emitExit());
      return child;
    };
    await runAgent(events, spawnFn);
    expect(events.map((e) => e.type)).toEqual(["agent_start", "tool_failed", "failed", "done"]);
    expect(toolFailedOf(events)).toEqual(["needle", "provider_error"]);
    const failed = events.find((e) => e.type === "failed");
    expect(failed?.type === "failed" ? failed.category : "").toBe("provider_error");
    expect(failed?.type === "failed" ? failed.message : "").toContain("kernel worker exited");
    const metrics = doneOf(events);
    expect(metrics.failures).toEqual({ provider_error: 1 });
    expect(metrics.needle).toEqual({ calls: 0, totalMs: 0, escalations: 1 });
    expect(metrics.tools).toEqual({ calls: 0, totalMs: 0 });
    expect(metrics.muse).toEqual({ calls: 0, totalMs: 0 });
    expect(metrics.evidence).toEqual({ count: 0, characters: 0 });
  });

  test("a worker-level error ends the report with merged evidence and worker tallies", async () => {
    const events: AgentEvent[] = [];
    const reply = (id: string): string => `${JSON.stringify({
      id,
      sessionId: "rs:err",
      objective: "q",
      error: "unknown session",
      evidence: [{ id: "ev:1", content: "fact" }, { id: "ev:1", content: "fact again" }],
      nodes: [],
      decisions: [],
      unresolved: [],
      toolExecutions: [{ step: 1, tool: "list_sec_filings", arguments: {}, confidence: null }],
      toolCalls: [{ tool: "list_sec_filings", ok: true, evidenceId: "ev:1" }],
      failures: { tool_error: 2 },
      escalations: 3,
      escalated: false,
    })}\n`;
    await runAgent(events, scriptedWorker(reply));
    // The error ends the report before any tool replay, and the duplicate evidence
    // row contributes one row and one character budget.
    expect(events.map((e) => e.type)).toEqual(["agent_start", "evidence", "tool_failed", "failed", "done"]);
    const evidence = events[1];
    expect(evidence?.type === "evidence" ? evidence.evidence.map((e) => e.content) : []).toEqual(["fact"]);
    expect(evidence?.type === "evidence" ? evidence.sessionId : undefined).toBe("rs:err");
    const metrics = doneOf(events);
    expect(metrics.evidence).toEqual({ count: 1, characters: 4 });
    expect(metrics.needle).toEqual({ calls: 1, totalMs: 0, escalations: 4 });
    expect(metrics.tools.calls).toBe(1);
    expect(metrics.muse.calls).toBe(0);
    expect(metrics.failures).toEqual({ tool_error: 2, provider_error: 1 });
  });

  test("a terminal result ends the report once and never repeats a disclosed tool failure", async () => {
    const cases = [
      {
        calls: [{ tool: "list_sec_filings", ok: false, category: "tool_error", error: "boom" }],
        terminal: { category: "timeout", message: "research deadline passed" },
        flow: ["agent_start", "evidence", "tool_start", "tool_result", "tool_failed", "failed", "done"],
        disclosed: ["list_sec_filings", "tool_error"],
      },
      {
        calls: [],
        terminal: { category: "policy_denied", message: "denied by policy" },
        flow: ["agent_start", "evidence", "tool_failed", "failed", "done"],
        disclosed: ["worker", "policy_denied"],
      },
    ] as const;
    for (const c of cases) {
      const events: AgentEvent[] = [];
      const reply = (id: string): string => `${JSON.stringify({
        id,
        sessionId: "rs:terminal",
        objective: "q",
        evidence: [{ id: "ev:1", content: "fact" }],
        nodes: [],
        decisions: [],
        unresolved: [],
        toolExecutions: [],
        toolCalls: c.calls,
        failures: {},
        escalations: 0,
        escalated: false,
        terminal: c.terminal,
      })}\n`;
      await runAgent(events, scriptedWorker(reply));
      expect(events.map((e) => e.type)).toEqual([...c.flow]);
      expect(toolFailedOf(events)).toEqual([...c.disclosed]);
      const failed = events.find((e) => e.type === "failed");
      expect(failed?.type === "failed" ? [failed.category, failed.message] : []).toEqual([c.terminal.category, c.terminal.message]);
      const metrics = doneOf(events);
      expect(metrics.failures[c.terminal.category]).toBe(1);
      expect(metrics.tools.calls).toBe(c.calls.length);
      expect(metrics.evidence.count).toBe(1);
    }
  });

  test("call replay shows evidence previews, re-selections, disclosed failures, and redacted arguments", async () => {
    const events: AgentEvent[] = [];
    const long = "x".repeat(2500);
    const reply = (id: string): string => `${JSON.stringify({
      id,
      sessionId: "rs:calls",
      objective: "q",
      evidence: [{ id: "ev:1", content: "filing text" }],
      nodes: [],
      decisions: [],
      unresolved: [],
      incomplete_guard: false,
      toolExecutions: [
        { step: 7, tool: "list_sec_filings", arguments: { api_key: "sk-secret", symbol: "NVDA", nested: { token: "t" }, list: ["a"], long, scalar: 3, nothing: null, flag: true }, confidence: 0.5 },
        { tool: "get_quote", arguments: {} },
        { tool: "get_news", arguments: {} },
      ],
      toolCalls: [
        { tool: "list_sec_filings", ok: true, evidenceId: "ev:1" },
        { tool: "get_quote", ok: false, category: "reselect", error: "unknown symbol" },
        { tool: "get_news", ok: false, category: "timeout", error: "slow" },
      ],
      failures: {},
      escalations: 0,
      escalated: false,
    })}\n`;
    await runAgent(events, scriptedWorker(reply), ["stockbot"]);
    const flow = events
      .filter((e) => e.type === "needle_decision" || e.type === "tool_start" || e.type === "tool_result" || e.type === "tool_failed")
      .map((e) => e.type);
    expect(flow).toEqual([
      "needle_decision", "tool_start", "tool_result",
      "needle_decision", "tool_start", "tool_result",
      "needle_decision", "tool_start", "tool_result", "tool_failed",
    ]);
    const decisions = events.filter((e) => e.type === "needle_decision");
    const first = decisions[0];
    if (first?.type !== "needle_decision") throw new Error("expected a decision event");
    expect(first.step).toBe(7);
    expect(first.confidence).toBe(0.5);
    // Secrets redact by key norm, nested and array values survive, and a long
    // string argument truncates at the argument budget.
    expect(first.arguments).toEqual({
      api_key: "[REDACTED]",
      symbol: "NVDA",
      nested: { token: "[REDACTED]" },
      list: ["a"],
      long: "x".repeat(2000),
      scalar: 3,
      nothing: null,
      flag: true,
    });
    expect(decisions[1]?.type === "needle_decision" ? decisions[1].step : -1).toBe(1);
    expect(decisions[2]?.type === "needle_decision" ? decisions[2].confidence : "unset").toBeNull();
    const results = events.filter((e) => e.type === "tool_result");
    expect(results[0]?.type === "tool_result" ? [results[0].evidenceId, results[0].preview] : []).toEqual(["ev:1", "filing text"]);
    expect(results[1]?.type === "tool_result" ? results[1].preview : "").toBe("unknown symbol");
    expect(results[1]?.type === "tool_result" ? "evidenceId" in results[1] : true).toBe(false);
    expect(results[2]?.type === "tool_result" ? results[2].preview : "").toBe("slow");
    expect(toolFailedOf(events)).toEqual(["get_news", "timeout"]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });
});

describe("trio offer and persona follow-up", () => {
  // One scripted worker per run: op:run answers with a one-row session graph,
  // and every gate passes. Every worker body and reasoner call is recorded.
  function trioDeps(bodies: Record<string, unknown>[], calls: KernelReasonInput[]) {
    const spawnFn: KernelSpawn = (): KernelChild => {
      const child = new FakeChild();
      child.reply = (id) => {
        const body = JSON.parse(child.written.at(-1) ?? "{}") as Record<string, unknown>;
        bodies.push(body);
        if (typeof body.op === "string" && body.op.startsWith("assess_final")) return `${JSON.stringify({ id, support_answer: "yes" })}\n`;
        return `${JSON.stringify({ id, objective: "q", sessionId: "rs:trio", evidence: [{ id: "ev:1", content: "fact" }], nodes: [{ node_id: "n1", question: "q", status: "resolved", depends_on: [] }], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
      };
      queueMicrotask(() => child.emitStdout('{"type":"ready"}\n'));
      return child;
    };
    return {
      python: "py", workerPath: "w", spawnFn,
      reason: async (o: KernelReasonInput): Promise<KernelReasonResult> => {
        calls.push(o);
        return { text: `${o.persona} view`, usage: {}, claims: [] };
      },
    };
  }
  const doneOf = (events: AgentEvent[]): Metrics => {
    const last = events.at(-1);
    if (last?.type !== "done") throw new Error("expected done last");
    return last.metrics;
  };

  test("a default research run calls stockbot once, then offers the trio for its session", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: AgentEvent[] = [];
    const calls: KernelReasonInput[] = [];
    await runKernelAgent("q", (e) => { events.push(e); }, { deps: trioDeps([], calls) });
    expect(calls.map((c) => c.persona)).toEqual(["stockbot"]);
    expect(events.slice(-3).map((e) => e.type)).toEqual(["answer_delta", "offer_trio", "done"]);
    expect(events.find((e) => e.type === "offer_trio")).toEqual({ type: "offer_trio", sessionId: "rs:trio" });
    expect(doneOf(events).muse.calls).toBe(1);
  });

  test("an explicit bull and bear selection runs the trio and never offers it", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: AgentEvent[] = [];
    const calls: KernelReasonInput[] = [];
    await runKernelAgent("Give me the bull and bear case on NVDA", (e) => { events.push(e); }, { personas: FINAL_PERSONAS, deps: trioDeps([], calls) });
    expect(calls.map((c) => c.persona)).toEqual(["bearbot", "bullbot", "stockbot"]);
    expect(events.some((e) => e.type === "offer_trio")).toBe(false);
    expect(doneOf(events).muse.calls).toBe(3);
  });

  test("the trio follow-up reuses the offered evidence with no worker run, intake, or tool call", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    await runKernelAgent("q", () => { }, { deps: trioDeps([], []) });
    const events: AgentEvent[] = [];
    const bodies: Record<string, unknown>[] = [];
    const calls: KernelReasonInput[] = [];
    await runPersonaAgent("rs:trio", FINAL_PERSONAS, (e) => { events.push(e); }, { deps: trioDeps(bodies, calls) });
    // Only the final gates reach the worker, judged against the offered session.
    expect(bodies.length).toBeGreaterThan(0);
    for (const b of bodies) {
      expect(String(b.op).startsWith("assess_final")).toBe(true);
      expect(b.sessionId).toBe("rs:trio");
    }
    expect(calls.map((c) => c.persona)).toEqual(["bearbot", "bullbot", "stockbot"]);
    for (const c of calls) {
      expect(c.evidence.map((e) => e.id)).toEqual(["ev:1"]);
      expect(c.nodes.map((n) => n.node_id)).toEqual(["n1"]);
      expect(c.direct).toBe(false);
    }
    expect(events[0]).toEqual({ type: "agent_start", prompt: "q" });
    expect(events.find((e) => e.type === "evidence")).toMatchObject({ sessionId: "rs:trio" });
    expect(events.some((e) => e.type === "offer_trio" || e.type === "tool_start" || e.type === "progress")).toBe(false);
    const metrics = doneOf(events);
    expect(metrics.tools.calls).toBe(0);
    expect(metrics.needle.calls).toBe(0);
    expect(metrics.muse.calls).toBe(3);
    // Worker phases never ran, so they stay absent, never 0.
    expect(Object.keys(metrics.timings ?? {}).sort()).toEqual(["gatesMs", "personasMs"]);
  });

  test("an unknown session fails closed with no worker or reasoner call", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: AgentEvent[] = [];
    const bodies: Record<string, unknown>[] = [];
    const calls: KernelReasonInput[] = [];
    await runPersonaAgent("rs:unknown", FINAL_PERSONAS, (e) => { events.push(e); }, { deps: trioDeps(bodies, calls) });
    expect(events.map((e) => e.type)).toEqual(["agent_start", "failed", "done"]);
    expect(bodies).toEqual([]);
    expect(calls).toEqual([]);
  });
});
