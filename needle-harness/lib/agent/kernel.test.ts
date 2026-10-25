import { describe, expect, jest, setSystemTime, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import fc from "fast-check";
import { applyFinalGates, asksWatchlist, datapointRows, dropUncovered, inventoryDefect, KernelRouter, parseTicker, parseWatchlist, qualifiedEvidenceFree, REQUEST_WALL_MS, runKernelAgent, runPersonaAgent, type FinalGateCaller, type KernelChild, type KernelReasonInput, type KernelReasonResult, type KernelSpawn } from "./kernel";
import { FINAL_PERSONAS, readDatapoints, type AgentEvent, type AskContext, type Datapoint, type Evidence, type FinalClaim, type Metrics, type Persona } from "./types";

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
  const sessionGraphReply = (id: string, unresolved: string[] = []): string =>
    `${JSON.stringify({ id, objective: "q", sessionId: "rs:live", evidence: [{ id: "ev:1", content: "fact" }], nodes: [], decisions: [], unresolved, incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
  test("seed and research evidence retain captures before the gated answer without changing model content", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: AgentEvent[] = [];
    const firstSource = { id: `capture:${"a".repeat(64)}` };
    const kernelSource = { id: "evidence:rs:test:ev:1" };
    const secondSource = { id: "evidence:rs:test:ev:2" };
    const seed: Evidence = { id: "ev:1", source: "seed", retrievedAt: "2026-01-01Z", content: "seed fact", sources: [firstSource] };
    const seen: string[][] = [];
    const spawnFn = (): KernelChild => {
      const child = new FakeChild();
      child.reply = (id) => {
        const body = JSON.parse(child.written.at(-1) ?? "{}") as { op?: string };
        if (body.op?.startsWith("assess_final")) return `${JSON.stringify({ id, support_answer: "yes" })}\n`;
        return `${JSON.stringify({
          id, sessionId: "rs:test",
          evidence: [
            { id: "ev:1", content: "kernel fact", sources: [kernelSource] },
            { id: "ev:bad", content: "uncaptured", sources: [{ id: "/raw/path" }] },
            { id: "ev:2", content: "second fact", title: "Second filing", url: "https://www.sec.gov/filing", sources: [secondSource] },
            { id: "ev:2", content: "duplicate", sources: [] },
          ],
          nodes: [], decisions: [], unresolved: ["n1"], needleDecisions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false,
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
          seen.push(o.evidence.map((e) => e.content));
          return { prose: "answer 1", usage: {}, claims: [{ text: "answer 1", claimType: "inference", evidenceIds: ["ev:2"] }] };
        }
      },
    });
    const delivered = events.filter((e): e is Extract<AgentEvent, { type: "evidence" }> => e.type === "evidence");
    expect(delivered).toHaveLength(2);
    const seedDelivered = delivered[0];
    const researchDelivered = delivered[1];
    if (!seedDelivered || !researchDelivered) throw new Error("expected two evidence events");
    expect(seedDelivered.evidence[0]?.sources).toEqual([firstSource]);
    expect(researchDelivered.sessionId).toBe("rs:test");
    expect(researchDelivered.evidence.map((e) => e.id)).toEqual(["ev:1", "ev:bad", "ev:2"]);
    expect(researchDelivered.evidence[0]?.content).toBe("seed fact");
    expect(researchDelivered.evidence[0]?.sources).toEqual([firstSource, kernelSource]);
    expect(researchDelivered.evidence[1]?.sources).toEqual([]);
    expect(researchDelivered.evidence[2]).toMatchObject({ content: "second fact", sources: [secondSource], title: "Second filing", url: "https://www.sec.gov/filing" });
    expect(seen).toEqual([["seed fact", "uncaptured", "second fact"]]);
    expect(events.filter((e) => e.type === "evidence" || e.type === "answer_delta").map((e) => e.type)).toEqual(["evidence", "evidence", "answer_delta"]);
    expect(events.flatMap((e) => e.type === "answer_delta" ? [e.text] : [])).toEqual(["── Stockbot ──\nanswer 1"]);
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
          return { prose: "The requested information is unknown.", usage: {}, claims: [{ text: "The requested information is unknown.", claimType: "unknown", evidenceIds: [] }] };
        },
      }
    });
    expect(events.map((e) => e.type)).toEqual(["agent_start", "status", "reasoning_start", "answer_delta", "done"]);
  });
  const emptyReply = (id: string): string =>
    `${JSON.stringify({ id, objective: "q", evidence: [], nodes: [], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
  type Outcome = { text?: string; claims?: FinalClaim[]; confidence?: number; usage?: KernelReasonResult["usage"]; timings?: KernelReasonResult["timings"] } | Error;
  type Call = { pass: number; input: KernelReasonInput };
  type GateVerdict = { verdict?: string; support_answer?: string; error?: string };
  // Drives runKernelAgent with a scripted reasoner. Stockbot settles last, so
  // output order never follows completion order. The reasoner input carries no
  // onDelta, so no raw draft can reach emit. Gate ops (assess_final_*) answer
  // inline from opts.gate (default yes); run bodies still record in order.
  async function runFinal(opts: {
    reply?: (id: string) => string;
    decide?: (persona: Persona, pass: number, attempt: number, input: KernelReasonInput) => Outcome;
    gate?: (body: Record<string, unknown>) => GateVerdict;
    personas?: readonly Persona[];
    deadlineAt?: number;
    signal?: AbortSignal;
    ticker?: string;
    currentChartTicker?: string;
    watchlist?: readonly string[];
    context?: AskContext;
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
      const innerReply = opts.reply ?? graphReply;
      c.reply = (id: string): string => innerReply(id);
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
          ...(opts.context !== undefined ? { context: opts.context } : {}),
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
              for (let i = FINAL_PERSONAS.indexOf(o.persona); i < FINAL_PERSONAS.length; i++) await Promise.resolve();
              inFlight -= 1;
              const d: Outcome = opts.decide?.(o.persona, pass, attempt, o) ?? {};
              if (d instanceof Error) throw d;
              const text = d.text ?? `${o.persona} p${pass}`;
              // Default answers cite the first evidence row so the citation gate keeps the prose.
              const cite = o.evidence[0]?.id;
              return {
                prose: text,
                usage: d.usage ?? {},
                ...(d.timings !== undefined ? { timings: d.timings } : {}),
                claims: d.claims ?? (cite ? [{ text, claimType: "inference", evidenceIds: [cite] }] : []),
                ...(d.confidence !== undefined ? { confidence: d.confidence } : {}),
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

  test("an unresolved node or tripped guard never starts a second worker run", async () => {
    for (const personas of [["stockbot"], FINAL_PERSONAS] as const) {
      // graphReply: one blocked node left unresolved, guard tripped, escalated.
      const { events, bodies, calls } = await runFinal({ personas });
      expect(runBodies(bodies).length).toBe(1);
      expect(events.filter((e) => e.type === "reasoning_start").length).toBe(1);
      expect(calls.every((c) => c.pass === 1)).toBe(true);
      expect(events.some((e) => e.type === "progress" && e.stage === "evidence_gap")).toBe(false);
      expect(answers(events)).toEqual([personas.length === 1 ? `${label("stockbot")}\nstockbot p1` : trioAnswer(1)]);
      expect(events.some((e) => e.type === "failed")).toBe(false);
    }
  });

  test("intake evidence starts a brief first; the full answer follows below a divider", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const { events, calls } = await runFinal({
      personas: ["stockbot"],
      reply: (id) =>
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

  test("the early brief is gated against the worker session before display; a rejected brief leaves no trace", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const brief = "Brief says 42 units.";
    const { events, gates } = await runFinal({
      personas: ["stockbot"],
      reply: (id) =>
        progress(id, "session", { session_id: "rs:live" }) +
        progress(id, "intake_evidence", { evidence: [{ id: "ev:0", content: "intake fact" }] }) +
        sessionGraphReply(id),
      // The brief (pass 0) states an uninventoried number; the gate passes the wording, the number still drops.
      decide: (_p, pass) => (pass === 0 ? { text: brief, claims: [] } : {}),
    });
    const briefGate = gates.find((g) => g.text === brief);
    expect(briefGate).toMatchObject({ op: "assess_final_response", sessionId: "rs:live", persona: "stockbot", evidence: [{ id: "ev:0", content: "intake fact" }] });
    const deltas = answers(events);
    expect(deltas.some((d) => d.includes("42"))).toBe(false);
    // No brief shown, so the full answer carries no divider.
    expect(deltas).toEqual([`${label("stockbot")}\nstockbot p1`]);
  });

  test("a brief over the full evidence ends the stream with no second answer", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const reply = (id: string): string =>
      `${JSON.stringify({ id, objective: "q", sessionId: "s1", evidence: [{ id: "ev:0", content: "intake fact" }], nodes: [], decisions: [], unresolved: [], incomplete_guard: false, toolExecutions: [], toolCalls: [], failures: {}, escalations: 0, escalated: false })}\n`;
    const { events } = await runFinal({
      personas: ["stockbot"],
      reply: (id) => progress(id, "intake_evidence", { evidence: [{ id: "ev:0", content: "intake fact" }] }) + reply(id),
    });
    expect(events.filter((e) => e.type === "answer_delta").length).toBe(1);
    expect(events.map((e) => e.type).slice(-2)).toEqual(["offer_trio", "done"]);
  });

  test("a gated research answer emits its kept claims once, right before done", async () => {
    const cited: FinalClaim = { text: "stockbot p1", claimType: "observed_fact", evidenceIds: ["ev:1"] };
    const open: FinalClaim = { text: "p1", claimType: "unknown", evidenceIds: [] };
    const { events } = await runFinal({ personas: ["stockbot"], reply: sessionGraphReply, decide: () => ({ claims: [cited, open] }) });
    expect(events.map((e) => e.type).slice(-3)).toEqual(["offer_trio", "claims", "done"]);
    expect(events.filter((e) => e.type === "claims")).toEqual([{ type: "claims", claims: [cited, open] }]);
    // An answer without an inventory emits no claims event.
    const bare = await runFinal({ personas: ["stockbot"], reply: sessionGraphReply, decide: () => ({ claims: [] }) });
    expect(bare.events.at(-1)?.type).toBe("done");
    expect(bare.events.some((e) => e.type === "claims")).toBe(false);
  });

  test("done timings sum worker progress phases, skip malformed values, and omit unmeasured phases", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const { events } = await runFinal({
      reply: (id) =>
        progress(id, "intake_done", { wall_ms: 100, settle_ms: 10 }) +
        progress(id, "reasoner_done", { proposals: 2, ms: 40 }) +
        progress(id, "intake_done", { wall_ms: 20, settle_ms: 5 }) +
        progress(id, "intake_done", { wall_ms: "7", settle_ms: null }) +
        progress(id, "nodes_done", { ms: 300 }) +
        graphReply(id),
    });
    expect(answers(events)).toEqual([trioAnswer(1)]);
    // routeMs and firstTextMs belong to the route.
    expect(doneMetrics(events).timings).toEqual({
      intakeMs: 120, admitMs: 15, reasonerMs: 40, nodesMs: 300, personasMs: expect.any(Number), gatesMs: expect.any(Number),
    });
  });

  test("trio sums logical calls and per-call provider posts separately", async () => {
    const { events } = await runFinal({
      decide: () => ({ usage: { providerCalls: 2 }, timings: { narrativeMs: 10, binderMs: 5 } }),
    });
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(3);
    expect(m.muse.providerCalls).toBe(6);
    expect(m.timings).toMatchObject({ narrativeMs: 30, binderMs: 15 });
  });

  test("a failed draft retry and an inventory correction add their provider posts", async () => {
    const usage = { inputTokens: 1, providerCalls: 2 };
    const { events: retryEvents } = await runFinal({
      decide: (p, _pass, attempt) =>
        p === "bearbot" && attempt === 1 ? new Error("muse down") : { usage },
    });
    const retry = doneMetrics(retryEvents);
    expect(retry.muse.calls).toBe(4);
    expect(retry.muse.providerCalls).toBe(6);
    expect(retry.muse.inputTokens).toBe(3);
    const { events: fixEvents } = await runFinal({
      personas: ["bearbot"],
      decide: (_p, _pass, attempt) =>
        attempt === 1
          ? { text: "bearbot p1", claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["rn:x"] }], usage }
          : { text: "bearbot fixed", claims: [{ text: "bearbot fixed", claimType: "inference", evidenceIds: ["ev:1"] }], usage },
    });
    const fix = doneMetrics(fixEvents);
    expect(fix.muse.calls).toBe(2);
    expect(fix.muse.providerCalls).toBe(4);
  });

  test("provider posts on a failed call still count through the mutable input", async () => {
    const { events, calls } = await runFinal({
      personas: ["bearbot"],
      decide: (_p, _pass, _attempt, input) => {
        input.usage!.providerCalls = 3;
        input.timings!.narrativeMs = 7;
        throw new Error("muse down");
      },
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot"]);
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(1);
    expect(m.muse.providerCalls).toBe(3);
    expect(m.timings).toMatchObject({ narrativeMs: 7 });
  });

  test("partial personas without provider telemetry omit the provider count", async () => {
    const { events } = await runFinal({ personas: ["bearbot", "bullbot"] });
    const m = doneMetrics(events);
    expect(m.muse.calls).toBe(2);
    expect(m.muse.providerCalls).toBeUndefined();
  });

  test("research opens with a status line and turns worker stages into readable counts before any tool result", async () => {
    const progress = (id: string, stage: string, detail: Record<string, unknown>): string => `${JSON.stringify({ type: "progress", id, stage, detail })}\n`;
    const { events } = await runFinal({
      reply: (id) =>
        progress(id, "session", { session_id: "s1" }) +
        progress(id, "intake_start", { calls: 3 }) +
        progress(id, "intake_done", { admitted: 5 }) +
        progress(id, "reasoner_done", { proposals: 1.5 }) +
        progress(id, "reasoner_done", { proposals: 3 }) +
        progress(id, "tool_start", { node_id: "n1", action: "invoke", tools: ["get_short_interest"] }) +
        progress(id, "tool_done", { node_id: "n1", admitted: -1 }) +
        progress(id, "tool_start", { node_id: "n1", action: "reason", tools: [] }) +
        progress(id, "tool_done", { node_id: "n1", admitted: 0 }) +
        progress(id, "nodes_done", { ms: 10 }) +
        // One replayed tool call: its tool_result follows every live status line.
        graphReply(id).replace('"toolCalls":[]', '"toolCalls":[{"tool":"get_short_interest","ok":true,"evidenceId":"ev:1"}]'),
    });
    expect(events[1]).toEqual({ type: "status", text: "Researching: q" });
    // A malformed worker count stays silent; the session stage has no line.
    expect(events.flatMap((e) => (e.type === "status" ? [e.text] : []))).toEqual([
      "Researching: q", "Checking 3 sources", "Found 5 sources", "Planned 3 questions",
      "Checking FINRA short interest", "Weighing the evidence", "Found nothing new", "Writing the answer",
    ]);
    const firstResult = events.findIndex((e) => e.type === "tool_result");
    expect(firstResult).toBeGreaterThan(events.findLastIndex((e) => e.type === "status"));
  });

  test("a failed persona fails the report with no prose", async () => {
    const { events, bodies } = await runFinal({ decide: (p) => (p === "bearbot" ? new Error("muse down") : {}) });
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
    const failed = events.find((e) => e.type === "failed");
    expect(failed?.type === "failed" ? failed.message : "").toBe("bearbot: muse down");
    expect(runBodies(bodies).length).toBe(1);
    const m = doneMetrics(events);
    // Bear draft, bull draft, one bear retry; no synthesis.
    expect(m.muse.calls).toBe(3);
    expect(m.failures.provider_error).toBe(1);
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

  test("the source gates keep supported facts and drop unsupported claims", async () => {
    const supported = "NVDA Q3 FY2025 revenue was $10M USD.";
    const unsupported = "NVDA Q3 FY2025 margin was 40%.";
    const { events, bodies, gates } = await runFinal({
      personas: ["bearbot"],
      reply: (id) => `${JSON.stringify({
        ...JSON.parse(graphReply(id)),
        evidence: [{
          id: "ev:1", content: "fact", datapoints: [
            { id: "dp:revenue", metric: "revenue", value: "$10M", entity: "NVDA", period: "Q3 FY2025", unit: "USD", statement: supported },
            { id: "dp:margin", metric: "margin", value: "40%", entity: "NVDA", period: "Q3 FY2025", unit: "%", statement: unsupported },
          ],
        }],
      })}\n`,
      decide: () => ({
        text: `${supported} ${unsupported}`,
        claims: [
          { text: supported, claimType: "observed_fact", evidenceIds: ["ev:1"], datapointIds: ["dp:revenue"] },
          { text: unsupported, claimType: "observed_fact", evidenceIds: ["ev:1"], datapointIds: ["dp:margin"] },
        ],
      }),
      gate: (body) => {
        if (body.op !== "assess_final_claim") return { support_answer: "no" };
        return (body.claim as Record<string, unknown>).text === supported ? { support_answer: "yes" } : { support_answer: "no" };
      },
    });
    expect(runBodies(bodies).length).toBe(1);
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim", "assess_final_claim"]);
    const out = answers(events);
    expect(out.length).toBe(1);
    expect(out[0]).toContain(supported);
    expect(out[0]).not.toContain(unsupported);
    expect(events.some((e) => e.type === "failed")).toBe(false);
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

  test("the one research run carries the request wall", async () => {
    const deadlineAt = Date.now() + 600_000;
    const { bodies } = await runFinal({ deadlineAt });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(1);
    expect(runs[0]?.deadlineAt).toBe(deadlineAt);
    expect("deadlineMs" in (runs[0] ?? {})).toBe(false);
  });

  test("empty graph answers direct through every selected persona", async () => {
    const { events, calls } = await runFinal({
      reply: emptyReply, personas: ["stockbot", "bullbot"],
      decide: (p) => ({ text: `${p} results are unknown.`, claims: [{ text: `${p} results are unknown.`, claimType: "unknown", evidenceIds: [] }] }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["stockbot", "bullbot"]);
    for (const c of calls) {
      expect(c.input.direct).toBe(true);
      expect(c.input.prompt).toBe("q");
      expect(c.input.evidence).toEqual([]);
    }
    expect(answers(events)).toEqual([`${label("stockbot")}\nstockbot results are unknown.\n\n${label("bullbot")}\nbullbot results are unknown.`]);
    expect(doneMetrics(events).muse.calls).toBe(2);
  });
  test("an empty-graph direct answer skips the source gates even when the gate service rejects", async () => {
    const { events, gates } = await runFinal({
      reply: emptyReply,
      personas: ["stockbot", "bullbot"],
      decide: (p) => ({ text: `${p} results are unknown.`, claims: [{ text: `${p} results are unknown.`, claimType: "unknown", evidenceIds: [] }] }),
      gate: () => ({ error: "gate down" }),
    });
    expect(gates).toEqual([]);
    expect(answers(events)).toEqual([`${label("stockbot")}\nstockbot results are unknown.\n\n${label("bullbot")}\nbullbot results are unknown.`]);
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
      reply: sessionGraphReply,
      decide: () => ({ text: `keep ${claim}`, claims: [{ text: claim, claimType: "inference", evidenceIds: ["ev:1"] }] }),
      gate: (body) => (body.op === "assess_final_response" ? { support_answer: "no" } : { support_answer: "yes" }),
    });
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim"]);
    // The worker keys gates by sessionId; without it, every gate fails closed on the numeric request id.
    for (const g of gates) expect(g.sessionId).toBe("rs:live");
  });

  test("a gate error rejects a claim, and a reject drops every sentence no supported claim covers", async () => {
    const claim = "bearbot claim here";
    const kept = "Bearbot kept sentence.";
    const { events, gates } = await runFinal({
      personas: ["bearbot"],
      reply: sessionGraphReply,
      decide: () => ({
        text: `${kept} Preamble ${claim} tail. Uncovered remark.`,
        claims: [
          { text: kept, claimType: "inference", evidenceIds: ["ev:1"] },
          { text: claim, claimType: "inference", evidenceIds: ["ev:1"] },
        ],
      }),
      gate: (body) => {
        if (body.op === "assess_final_response") return { support_answer: "no" };
        return (body.claim as Record<string, unknown>).text === kept ? { support_answer: "yes" } : { error: "unknown session_id: '39'" };
      },
    });
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim", "assess_final_claim"]);
    expect(answers(events)).toEqual([`${label("bearbot")}\n${kept}`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("a gate error that leaves no prose fails closed as provider_error", async () => {
    const { events } = await runFinal({
      personas: ["bearbot"],
      reply: sessionGraphReply,
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
    const kept = "bearbot may keep this.";
    const cut = "bearbot reported this.";
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
    const kept = "bearbot corrected fact.";
    const cut = "bearbot corrected guess.";
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
      reply: emptyReply,
      personas: ["stockbot"],
      decide: () => ({ claims: [{ text: "stockbot p1", claimType: "inference", evidenceIds: ["ev:9"] }] }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["stockbot"]);
    expect(gates).toEqual([]);
    expect(answers(events)).toEqual([]);
    expect(events.some((e) => e.type === "failed" && e.message === "stockbot: all claims rejected")).toBe(true);
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
    expect(points).toEqual(new Map([[revenuePoint.id, "ev:1"], [margin.id, "ev:1"]]));
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
  });

  test("qualifiedEvidenceFree keeps only qualified unknowns and opening assumptions without citations", () => {
    const unknown: FinalClaim = { text: "No 2026 guidance was found.", claimType: "unknown", evidenceIds: [] };
    const assumption: FinalClaim = { text: "Assuming flat demand, margins hold at 70%.", claimType: "assumption", evidenceIds: [] };
    const rejected: FinalClaim[] = [
      { text: "Revenue was $99B but margins are unknown.", claimType: "unknown", evidenceIds: [] },
      { text: "Demand remains firm.", claimType: "unknown", evidenceIds: [] },
      { text: "Margins hold at 70%, assuming flat demand.", claimType: "assumption", evidenceIds: [] },
      { ...unknown, evidenceIds: ["ev:1"] },
      { ...assumption, evidenceIds: ["ev:1"] },
      { ...unknown, datapointIds: ["dp:guidance"] },
      { ...assumption, datapointIds: ["dp:margin"] },
      { text: "Demand may hold.", claimType: "inference", evidenceIds: [] },
    ];
    expect(qualifiedEvidenceFree([unknown, ...rejected, assumption])).toEqual([unknown, assumption]);
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

  test("a fabricated or unproven number never shows and datapoint ids enter the gate wire", async () => {
    const fact: FinalClaim = { text: "NVDA Q3 FY2025 revenue was $35.1B USD [ev:1].", claimType: "observed_fact", evidenceIds: ["ev:1"], datapointIds: [revenuePoint.id] };
    const hedge: FinalClaim = { text: "Margins may compress if demand slows.", claimType: "inference", evidenceIds: ["ev:1"] };
    const text = `${fact.text} ${hedge.text} Margin hit 99%.`;
    const reply = (id: string): string => `${JSON.stringify({
      ...JSON.parse(sessionGraphReply(id)),
      evidence: [{ id: "ev:1", content: "fact", datapoints: [revenuePoint, null, { ...revenuePoint, id: "ev:bad" }] }],
    })}\n`;
    // The whole-response pass still drops the sentence that has no inventory claim.
    const passed = await runFinal({ personas: ["stockbot"], reply, decide: () => ({ text, claims: [fact, hedge] }) });
    expect(passed.calls[0]?.input.evidence[0]?.datapoints).toEqual([revenuePoint]);
    expect(passed.events.find((e) => e.type === "evidence")).toMatchObject({ evidence: [{ datapoints: [revenuePoint] }] });
    expect(passed.gates[0]?.claims).toEqual([
      { text: fact.text, claim_type: "observed_fact", evidence_ids: ["ev:1"], datapoint_ids: [revenuePoint.id] },
      { text: hedge.text, claim_type: "inference", evidence_ids: ["ev:1"] },
    ]);
    expect(answers(passed.events)).toEqual([`${label("stockbot")}\n${fact.text} ${hedge.text}`]);
    expect(passed.events.find((e) => e.type === "claims")).toEqual({ type: "claims", claims: [fact, hedge] });
    // A ledger proof failure removes the datapoint claim but keeps the qualified inference.
    const proofFailed = await runFinal({
      personas: ["stockbot"],
      reply,
      decide: () => ({ text, claims: [fact, hedge] }),
      gate: (body) => {
        if (body.op === "assess_final_response") return { error: "proof: datapoint conflicts with ledger" };
        return (body.claim as Record<string, unknown>).datapoint_ids === undefined ? { support_answer: "yes" } : { error: "proof: datapoint conflicts with ledger" };
      },
    });
    expect(proofFailed.gates.find((g) => g.op === "assess_final_claim")?.claim).toEqual({
      text: fact.text, claim_type: "observed_fact", evidence_ids: ["ev:1"], datapoint_ids: [revenuePoint.id],
    });
    for (const body of [...passed.gates, ...proofFailed.gates]) {
      const claims = body.op === "assess_final_response" ? body.claims as Record<string, unknown>[] : [body.claim as Record<string, unknown>];
      for (const claim of claims) {
        for (const key of ["value", "entity", "period", "unit"]) expect(claim).not.toHaveProperty(key);
      }
    }
    expect(answers(proofFailed.events)).toEqual([`${label("stockbot")}\n${hedge.text}`]);
  });

  test("narrative claims stay unproven until the gates keep them: one gated delta, kept claims, no raw stream", async () => {
    const kept: FinalClaim = { text: "Demand may rise [ev:1].", claimType: "inference", evidenceIds: ["ev:1"] };
    const unsupported: FinalClaim = { text: "Revenue doubled [ev:1].", claimType: "observed_fact", evidenceIds: ["ev:1"] };
    const { events, calls, gates } = await runFinal({
      personas: ["stockbot"],
      reply: sessionGraphReply,
      // The inventory passes the local check, so only the worker gates can reject the unsupported claim.
      decide: () => ({ text: `${kept.text} ${unsupported.text}`, claims: [kept, unsupported] }),
      gate: (body) => (body.op === "assess_final_claim" && (body.claim as Record<string, unknown>).text === kept.text ? { support_answer: "yes" } : { support_answer: "no" }),
    });
    // The reasoner gets no stream callback: raw deltas carry the inventory fence.
    for (const c of calls) expect("onDelta" in c.input).toBe(false);
    expect(gates.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim", "assess_final_claim"]);
    expect(answers(events)).toEqual([`${label("stockbot")}\n${kept.text}`]);
    expect(events.filter((e) => e.type === "claims")).toEqual([{ type: "claims", claims: [kept] }]);
    expect(JSON.stringify(events)).not.toContain("```");
    expect(JSON.stringify(events)).not.toContain("Revenue doubled");
  });

  test("a bound grounded explanation passes; an unbound narrative fails closed", async () => {
    const bound: FinalClaim = { text: "Demand may rise [ev:1].", claimType: "inference", evidenceIds: ["ev:1"] };
    const passed = await runFinal({
      personas: ["stockbot"],
      reply: sessionGraphReply,
      decide: () => ({ text: bound.text, claims: [bound] }),
    });
    expect(answers(passed.events)).toEqual([`${label("stockbot")}\n${bound.text}`]);
    // The binder bound nothing: the gates pass, but uncovered prose still drops and the run fails closed.
    const unbound = await runFinal({
      personas: ["stockbot"],
      reply: sessionGraphReply,
      decide: () => ({ text: "Revenue doubled [ev:1].", claims: [] }),
    });
    expect(answers(unbound.events)).toEqual([]);
    expect(failedCategory(unbound.events)).toBe("provider_error");
  });

  test("a failed document read is a limitation, never evidence; only an inspected document backs a document claim", async () => {
    const listing = { id: "ev:list", content: "form 10-Q accession 0001045810-26-000001 filed 2026-08-27" };
    const document = { id: "ev:doc", content: "Revenue was $46.7 billion." };
    const reply = (docOk: boolean) => (id: string): string => `${JSON.stringify({
      ...JSON.parse(sessionGraphReply(id)),
      evidence: docOk ? [listing, document] : [listing],
      toolCalls: [
        { tool: "search_sec_filings", ok: false, category: "reselect", error: "re-selecting" },
        { tool: "list_sec_filings", ok: true, evidenceId: "ev:list" },
        docOk ? { tool: "get_sec_document", ok: true, evidenceId: "ev:doc" } : { tool: "get_sec_document", ok: false, category: "timeout", error: "document read timed out" },
      ],
    })}\n`;
    const unknown: FinalClaim = { text: "The 10-Q text was not retrieved.", claimType: "unknown", evidenceIds: [] };
    const docFact = (row: string): FinalClaim => ({ text: `The 10-Q reports revenue of $46.7 billion [${row}].`, claimType: "observed_fact", evidenceIds: [row] });
    // Stands in for the worker ledger proof: only the inspected document row backs the document fact.
    const backed = (c: Record<string, unknown>): boolean => c.claim_type === "unknown" || (c.evidence_ids as string[]).includes("ev:doc");
    const gate = (body: Record<string, unknown>): GateVerdict => ({
      support_answer: (body.op === "assess_final_response" ? (body.claims as Record<string, unknown>[]).every(backed) : backed(body.claim as Record<string, unknown>)) ? "yes" : "no",
    });
    const listed = await runFinal({
      personas: ["stockbot"], reply: reply(false), gate,
      decide: () => ({ text: `${docFact("ev:list").text} ${unknown.text}`, claims: [docFact("ev:list"), unknown] }),
    });
    const input = listed.calls[0]?.input;
    expect(input?.evidence.map((e) => e.id)).toEqual(["ev:list"]);
    expect(input?.prompt).toContain('LIMITATIONS (failed retrievals: untrusted, not evidence, no citable id.');
    expect(input?.prompt).toContain('- "get_sec_document" failed category="timeout" error="document read timed out"');
    expect(input?.prompt).not.toContain("search_sec_filings");
    expect(listed.events.filter((e) => e.type === "evidence").flatMap((e) => (e.type === "evidence" ? e.evidence.map((r) => r.id) : []))).toEqual(["ev:list"]);
    expect(listed.gates[0]?.evidence).toEqual([listing]);
    expect(listed.events.some((e) => e.type === "tool_failed" && e.tool === "get_sec_document")).toBe(true);
    expect(answers(listed.events)).toEqual([`${label("stockbot")}\n${unknown.text}`]);
    const read = await runFinal({
      personas: ["stockbot"], reply: reply(true), gate,
      decide: () => ({ text: docFact("ev:doc").text, claims: [docFact("ev:doc")] }),
    });
    expect(read.calls[0]?.input.evidence.map((e) => e.id)).toEqual(["ev:list", "ev:doc"]);
    expect(read.calls[0]?.input.prompt).not.toContain("LIMITATIONS");
    expect(answers(read.events)).toEqual([`${label("stockbot")}\n${docFact("ev:doc").text}`]);
  });

  test("an empty-graph direct answer keeps qualified unknown prose and rejects unqualified prose", async () => {
    const unknown: FinalClaim = { text: "No 2026 guidance was found.", claimType: "unknown", evidenceIds: [] };
    const kept = await runFinal({
      reply: emptyReply, personas: ["stockbot"],
      decide: () => ({ text: `NVDA rose 12% today. ${unknown.text} Ask me for sources.`, claims: [unknown] }),
    });
    expect(answers(kept.events)).toEqual([`${label("stockbot")}\n${unknown.text}`]);
    expect(kept.gates).toEqual([]);
    for (const claims of [[], [{ text: "Ask me for sources.", claimType: "unknown" as const, evidenceIds: [] }]]) {
      const empty = await runFinal({ reply: emptyReply, personas: ["stockbot"], decide: () => ({ text: "NVDA rose 12% today. Ask me for sources.", claims }) });
      expect(answers(empty.events)).toEqual([]);
      expect(failedCategory(empty.events)).toBe("provider_error");
      expect(empty.events.find((e) => e.type === "failed")).toMatchObject({ message: "stockbot: all claims rejected" });
    }
  });

  test("a research answer with no cited claim ships the gated prose with no banner", async () => {
    const { events } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ text: "The bearbot outlook is unknown.", claims: [{ text: "The bearbot outlook is unknown.", claimType: "unknown", evidenceIds: [] }] }),
    });
    expect(answers(events)).toEqual([`${label("bearbot")}\nThe bearbot outlook is unknown.`]);
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("a cited claim the gates delete leaves only the kept prose, with no banner", async () => {
    const kept = "The bearbot outlook is unknown.";
    const cut = "bearbot cited fact.";
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
    expect(out[0]).toStartWith(`${label("bearbot")}\n${kept}`);
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

  test("invalid ticker drops from the run body", async () => {
    const { bodies } = await runFinal({ ticker: "SEC" });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(1);
    expect(runs[0]?.prompt).toBe("q");
    expect("ticker" in (runs[0] ?? {})).toBe(false);
  });

  test("absent or invalid ticker never rides the run body", async () => {
    for (const ticker of [undefined, "SEC", "1AB"]) {
      const { bodies } = await runFinal(ticker !== undefined ? { ticker } : {});
      const runs = runBodies(bodies);
      expect(runs.length).toBe(1);
      expect(runs[0]?.prompt).toBe("q");
      expect("ticker" in (runs[0] ?? {})).toBe(false);
    }
  });

  test("the run body keeps op, prompt, and deadline beside the ticker", async () => {
    const { bodies } = await runFinal({ ticker: " brk.b " });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(1);
    expect(runs[0]?.op).toBe("run");
    expect(runs[0]?.prompt).toBe("q");
    expect(typeof runs[0]?.deadlineAt).toBe("number");
    expect(runs[0]?.ticker).toBe("BRK.B");
  });

  test("watchlist rides the run body and replaces the selected ticker", async () => {
    const { bodies } = await runFinal({ ticker: "nvda", watchlist: ["NVDA", "QQQ"] });
    const runs = runBodies(bodies);
    expect(runs.length).toBe(1);
    expect(runs[0]?.watchlist).toEqual(["NVDA", "QQQ"]);
    expect("ticker" in (runs[0] ?? {})).toBe(false);
    expect(runs[0]?.prompt).toBe("q");
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
      reply: (id) => `${JSON.stringify({ id, evidence: [], nodes: [], unresolved: [], toolCalls: [], needleDecisions: [], watchlist: rows })}\n`,
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
      reply: (id) => progressLines(id) + graphReply(id),
    });
    const progress = events.filter((e) => e.type === "progress" && (e.stage === "p_bare" || e.stage === "p_detail"));
    expect(progress.length).toBe(2);
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
      const { calls, bodies } = await runFinal({ ticker, currentChartTicker, personas: ["stockbot"], reply: emptyReply });
      expect(calls).toHaveLength(1);
      expect(calls[0]?.input.direct).toBe(true);
      expect(calls[0]?.input.ticker).toBe(expected);
      if (expected === undefined) expect(calls[0]?.input).not.toHaveProperty("ticker");
      const runs = runBodies(bodies);
      expect(runs[0]?.ticker).toBe(ticker?.toUpperCase());
      expect(runs[0]).not.toHaveProperty("currentChartTicker");
    }
  });

  test("a feed context reaches every reasoner call and every worker run body", async () => {
    const context: AskContext = { kind: "news", title: "NVDA jumps", source: "Stocktwits", sym: "NVDA", at: "2026-10-20T09:00:00Z" };
    const { calls, bodies } = await runFinal({ ticker: "nvda", context, reply: (id) => sessionGraphReply(id, ["n1"]) });
    expect(calls.length).toBeGreaterThan(0);
    for (const call of calls) expect(call.input.context).toEqual(context);
    expect(runBodies(bodies).length).toBeGreaterThan(0);
    for (const body of runBodies(bodies)) expect(body).toMatchObject({ prompt: expect.not.stringContaining("NVDA jumps"), ticker: "NVDA", context });
    const { calls: bare, bodies: bareBodies } = await runFinal({ ticker: "nvda", personas: ["stockbot"], reply: emptyReply });
    expect(bare[0]?.input).not.toHaveProperty("context");
    for (const body of runBodies(bareBodies)) expect(body).not.toHaveProperty("context");
  });

  test("chart progress updates every trio call after the research run without changing research scope", async () => {
    const { events, calls, bodies } = await runFinal({
      ticker: "nvda",
      currentChartTicker: "aapl",
      reply: (id) => `${JSON.stringify({ type: "progress", id, stage: "chart_update", detail: { ticker: " msft " } })}\n${sessionGraphReply(id, ["n1"])}`,
    });
    expect(events.filter((e) => e.type === "chart_update")).toEqual([{ type: "chart_update", ticker: "MSFT" }]);
    expect(events.some((e) => e.type === "progress" && e.stage === "chart_update")).toBe(false);
    expect(calls.map((c) => [c.input.persona, c.input.ticker])).toEqual([["bearbot", "MSFT"], ["bullbot", "MSFT"], ["stockbot", "MSFT"]]);
    const runs = runBodies(bodies);
    expect(runs).toHaveLength(1);
    expect(runs[0]?.ticker).toBe("NVDA");
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
      reply: (id) => details.map((detail) => `${JSON.stringify({ type: "progress", id, stage: "chart_update", detail })}\n`).join("")
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
      reply: (id) => details.map((detail) => `${JSON.stringify({ type: "progress", id, stage: "widget_update", detail })}\n`).join("")
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

  test("watchlist answers omit chart context through chart updates and trio retries", async () => {
    const { events, calls, bodies } = await runFinal({
      ticker: "nvda",
      currentChartTicker: "aapl",
      watchlist: ["NVDA", "QQQ"],
      reply: (id) => `${JSON.stringify({ type: "progress", id, stage: "chart_update", detail: { ticker: "msft" } })}\n${sessionGraphReply(id, ["n1"])}`,
      decide: (persona, _pass, attempt) => (persona === "bearbot" && attempt === 1 ? new Error("retry draft") : {}),
    });
    expect(events.filter((e) => e.type === "chart_update")).toEqual([{ type: "chart_update", ticker: "MSFT" }]);
    // Bear draft fails once and retries, then the synthesis: four calls.
    expect(calls).toHaveLength(4);
    for (const call of calls) expect(call.input).not.toHaveProperty("ticker");
    const runs = runBodies(bodies);
    expect(runs).toHaveLength(1);
    for (const run of runs) {
      expect(run.watchlist).toEqual(["NVDA", "QQQ"]);
      expect(run).not.toHaveProperty("ticker");
    }
    expect(events.some((e) => e.type === "failed")).toBe(false);
  });

  test("applyFinalGates treats worker error as reject and deletes surgically", async () => {
    const ev = [{ id: "ev:1", source: "kernel", retrievedAt: "t", content: "fact" }];
    const gated = await applyFinalGates({
      results: [{ persona: "bearbot", prose: "KEEP. CUT. Uncovered.", claims: [{ text: "KEEP", claimType: "inference", evidenceIds: ["ev:1"] }, { text: "CUT", claimType: "inference", evidenceIds: ["ev:1"] }] }],
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
      return { persona: p, prose: `${p} lead.\n${claims.map((c) => c.text).join("\n")}`, claims };
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
    // The reject drops each uncovered lead sentence and each rejected claim.
    expect(gated).toEqual({ text: personas.map((p) => `${label(p)}\n${p}-c0`).join("\n\n") });
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
        reason: async (o: KernelReasonInput): Promise<KernelReasonResult> => ({ prose: `${o.persona} answer`, usage: {}, claims: [{ text: `${o.persona} answer`, claimType: "inference", evidenceIds: o.evidence.map((e) => e.id) }] }),
      },
    });
  }
  // Gate ops pass (the reasoner answers carry no datapoints); every other op gets the scripted reply.
  function scriptedWorker(reply: (id: string) => string): KernelSpawn {
    return (): KernelChild => {
      const child = new FakeChild();
      child.reply = (id) => {
        const body = JSON.parse(child.written.at(-1) ?? "{}") as { op?: string };
        return body.op?.startsWith("assess_final") ? `${JSON.stringify({ id, support_answer: "yes" })}\n` : reply(id);
      };
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
    expect(events.map((e) => e.type)).toEqual(["agent_start", "status", "tool_failed", "failed", "done"]);
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
    expect(events.map((e) => e.type)).toEqual(["agent_start", "status", "evidence", "tool_failed", "failed", "done"]);
    const evidence = events[2];
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
        flow: ["agent_start", "status", "evidence", "tool_start", "tool_result", "tool_failed", "failed", "done"],
        disclosed: ["list_sec_filings", "tool_error"],
      },
      {
        calls: [],
        terminal: { category: "policy_denied", message: "denied by policy" },
        flow: ["agent_start", "status", "evidence", "tool_failed", "failed", "done"],
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
        return { prose: `${o.persona} view`, usage: {}, claims: [{ text: `${o.persona} view`, claimType: "inference", evidenceIds: ["ev:1"] }] };
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
    expect(events.slice(-4).map((e) => e.type)).toEqual(["answer_delta", "offer_trio", "claims", "done"]);
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
