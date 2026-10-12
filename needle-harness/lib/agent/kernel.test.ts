import { describe, expect, setSystemTime, test } from "bun:test";
import fc from "fast-check";
import { applyFinalGates, inventoryDefect, KernelRouter, REQUEST_WALL_MS, runKernelAgent, type KernelChild, type KernelReasonInput, type KernelReasonResult, type KernelSpawn } from "./kernel";
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
});

describe("runKernelAgent final personas", () => {
  const graphReply = (id: string): string =>
    `${JSON.stringify({ id, objective: "q", evidence: [{ id: "ev:1", content: "fact" }], nodes: [{ node_id: "n1", question: "q", status: "blocked", depends_on: [] }], decisions: [], unresolved: ["n1"], incomplete_guard: true, toolExecutions: [], toolCalls: [], failures: {}, escalations: 1, escalated: true })}\n`;
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
          return writes === 1 ? { text: "answer 1", usage: {}, missingEvidence: "missing" } : { text: "answer 2", usage: {} };
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
  }): Promise<{ events: AgentEvent[]; calls: Call[]; bodies: Record<string, unknown>[]; gates: Record<string, unknown>[]; maxInFlight: number; runTimeouts: (number | undefined)[] }> {
    process.env.OPENCODE_MODEL = "test-model";
    const bodies: Record<string, unknown>[] = [];
    const gates: Record<string, unknown>[] = [];
    const gateFn = opts.gate;
    const spawn: KernelSpawn = (): KernelChild => {
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
          ...(opts.personas ? { personas: opts.personas } : {}),
          ...(opts.signal ? { signal: opts.signal } : {}),
          deps: {
            python: "py",
            workerPath: "w",
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
              return {
                text: d.text ?? `${o.persona} p${pass}`,
                usage: d.usage ?? {},
                claims: d.claims ?? [],
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
    return { events, calls, bodies, gates, maxInFlight, runTimeouts };
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

  test("default trio drafts bear+bull concurrently then synthesizes stockbot with both drafts", async () => {
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
      decide: (p, _pass, attempt) => (p === "bearbot" && attempt === 1 ? new Error("muse down") : { usage }),
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bullbot", "bearbot", "stockbot"]);
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

  test("first gap after persona time passes the original wall still runs one follow-up with a fresh wall", async () => {
    const deadlineAt = Date.now() + 5_000;
    const resumedAt = deadlineAt + 60_000;
    try {
      const { events, bodies, runTimeouts } = await runFinal({
        deadlineAt,
        decide: (p, pass) => {
          if (pass === 1) setSystemTime(new Date(resumedAt));
          return pass === 1 && p === "bearbot" ? { missingEvidence: "need doc X" } : {};
        },
      });
      const after = Date.now();
      const runs = runBodies(bodies);
      expect(runs.length).toBe(2);
      expect(runs[0]?.deadlineAt).toBe(deadlineAt);
      const fresh = Number(runs[1]?.deadlineAt);
      expect(fresh).toBeGreaterThanOrEqual(resumedAt + REQUEST_WALL_MS);
      expect(fresh).toBeLessThanOrEqual(after + REQUEST_WALL_MS);
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

  test("property: any persona elapsed time or follow-up evidence runs exactly one fresh follow-up and no third pass", async () => {
    await fc.assert(
      fc.asyncProperty(fc.integer({ min: 0, max: 24 * 3_600_000 }), fc.boolean(), fc.boolean(), async (elapsedMs, newEvidence, gapAgain) => {
        const deadlineAt = Date.now() + 1_000;
        const resumedAt = deadlineAt + elapsedMs;
        try {
          const { events, bodies, runTimeouts } = await runFinal({
            deadlineAt,
            reply: newEvidence ? sessionReply : (_p, id) => graphReply(id),
            decide: (p, pass) => {
              if (pass === 1) setSystemTime(new Date(resumedAt));
              return (pass === 1 || gapAgain) && p === "bullbot" ? { missingEvidence: "need doc X" } : {};
            },
          });
          const after = Date.now();
          const runs = runBodies(bodies);
          expect(runs.length).toBe(2);
          expect(runs[0]?.deadlineAt).toBe(deadlineAt);
          const fresh = Number(runs[1]?.deadlineAt);
          expect(fresh).toBeGreaterThanOrEqual(resumedAt + REQUEST_WALL_MS);
          expect(fresh).toBeLessThanOrEqual(after + REQUEST_WALL_MS);
          expect(runTimeouts[1]).toBe(REQUEST_WALL_MS);
          if (newEvidence) expect(runs[1]?.sessionId).toBe("s1");
          expect(events.filter((e) => e.type === "reasoning_start").length).toBe(2);
          expect(answers(events)).toEqual([trioAnswer(2)]);
          expect(events.some((e) => e.type === "failed")).toBe(false);
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
      decide: (_p, _pass, attempt) =>
        attempt === 1
          ? { text: "bearbot p1", claims: [{ text: "bearbot p1", claimType: "inference", evidenceIds: ["rn:56c1365b-followup"] }], usage }
          : { text: "bearbot fixed", claims: [{ text: "bearbot fixed", claimType: "inference", evidenceIds: ["ev:1"] }], usage },
    });
    expect(calls.map((c) => c.input.persona)).toEqual(["bearbot", "bearbot"]);
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

  test("all claims rejected fails closed with no prose", async () => {
    const { events } = await runFinal({
      personas: ["bearbot"],
      decide: () => ({ text: "bearbot claim here", claims: [{ text: "bearbot claim here", claimType: "inference", evidenceIds: ["ev:1"] }] }),
      gate: (body) => (body.op === "assess_final_claim" ? { support_answer: "no" } : { support_answer: "no" }),
    });
    expect(answers(events)).toEqual([]);
    expect(failedCategory(events)).toBe("provider_error");
  });

  test("applyFinalGates treats worker error as reject and deletes surgically", async () => {
    const ev = [{ id: "ev:1", source: "kernel", retrievedAt: "t", content: "fact" }];
    const gated = await applyFinalGates({
      results: [{ persona: "bearbot", text: "aaa KEEP bbb CUT ccc", claims: [{ text: "KEEP", claimType: "inference", evidenceIds: ["ev:1"] }, { text: "CUT", claimType: "inference", evidenceIds: ["ev:1"] }] }],
      evidence: ev,
      personaOrder: ["bearbot"],
      gate: async (body) => {
        if (body.op === "assess_final_response") return { error: "boom" };
        return (body.claim as Record<string, unknown>).text === "KEEP" ? { support_answer: "yes" } : { error: "boom" };
      },
    });
    if (!("text" in gated)) throw new Error("expected gated text");
    expect(gated.text).toContain("KEEP");
    expect(gated.text).not.toContain("CUT");
  });
});
