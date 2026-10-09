import { describe, expect, test } from "bun:test";
import { KernelRouter, runKernelAgent, type KernelChild, type KernelReasonInput, type KernelReasonResult, type KernelSpawn } from "./kernel";
import { FINAL_PERSONAS, type AgentEvent, type Metrics, type Persona } from "./types";

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
  // Records every worker body; reply sees whether this is the second run body.
  const recordingSpawn = (reply: (pass2: boolean, id: string) => string): { spawn: KernelSpawn; bodies: Record<string, unknown>[] } => {
    const bodies: Record<string, unknown>[] = [];
    const spawn: KernelSpawn = (): KernelChild => {
      const c = new FakeChild();
      const origWrite = c.stdin.write;
      c.stdin.write = (data: string, cb?: (err?: Error | null) => void): void => {
        bodies.push(JSON.parse(data) as Record<string, unknown>);
        origWrite.call(c.stdin, data, cb);
      };
      c.reply = (id: string): string => reply(bodies.length > 1, id);
      queueMicrotask(() => c.emitStdout('{"type":"ready"}\n'));
      return c;
    };
    return { spawn, bodies };
  };
  type Outcome = { text?: string; missingEvidence?: string; usage?: KernelReasonResult["usage"] } | Error;
  type Call = { pass: number; input: KernelReasonInput };
  // Drives runKernelAgent with a scripted reasoner. Stockbot settles last, so
  // output order never follows completion order; every call emits a draft delta.
  async function runFinal(opts: {
    reply?: (pass2: boolean, id: string) => string;
    decide?: (persona: Persona, pass: number, attempt: number) => Outcome;
    personas?: readonly Persona[];
    deadlineAt?: number;
    signal?: AbortSignal;
  }): Promise<{ events: AgentEvent[]; calls: Call[]; bodies: Record<string, unknown>[]; maxInFlight: number }> {
    process.env.OPENCODE_MODEL = "test-model";
    const { spawn, bodies } = recordingSpawn(opts.reply ?? ((_p, id) => graphReply(id)));
    const events: AgentEvent[] = [];
    const calls: Call[] = [];
    let pass = 0;
    let inFlight = 0;
    let maxInFlight = 0;
    const attempts: Record<string, number> = {};
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
            return { text: d.text ?? `${o.persona} p${pass}`, usage: d.usage ?? {}, ...(d.missingEvidence ? { missingEvidence: d.missingEvidence } : {}) };
          },
        },
      },
    );
    return { events, calls, bodies, maxInFlight };
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

  test("an unresolved second gap fails with incomplete_evidence and no prose", async () => {
    const { events, calls, bodies } = await runFinal({
      decide: (p) => (p === "bullbot" ? { missingEvidence: "still need doc X" } : {}),
    });
    expect(answers(events)).toEqual([]);
    expect(events.filter((e) => e.type === "reasoning_start").length).toBe(2);
    expect(calls.length).toBe(6);
    expect(runBodies(bodies).length).toBe(2);
    expect(failedCategory(events)).toBe("incomplete_evidence");
    expect(doneMetrics(events).muse.calls).toBe(6);
  });

  test("near deadline skips the follow-up and fails with incomplete_evidence", async () => {
    const { events, bodies } = await runFinal({
      deadlineAt: Date.now() + 5_000,
      decide: (p) => (p === "bearbot" ? { missingEvidence: "need doc X" } : {}),
    });
    expect(runBodies(bodies).length).toBe(1);
    expect(answers(events)).toEqual([]);
    expect(events.at(-2)?.type).toBe("failed");
    expect(failedCategory(events)).toBe("incomplete_evidence");
  });

  test("every persona call and pass-2 body carries the shared deadlineAt and request signal", async () => {
    const deadlineAt = Date.now() + 600_000;
    const signal = new AbortController().signal;
    const { calls, bodies } = await runFinal({
      deadlineAt,
      signal,
      decide: (p, pass) => (pass === 1 && p === "bullbot" ? { missingEvidence: "need doc X" } : {}),
    });
    expect(calls.length).toBe(6);
    for (const c of calls) {
      expect(c.input.deadlineAt).toBe(deadlineAt);
      expect(c.input.signal).toBe(signal);
    }
    const runs = runBodies(bodies);
    expect(runs.length).toBe(2);
    for (const b of runs) {
      expect(b.deadlineAt).toBe(deadlineAt);
      expect("deadlineMs" in b).toBe(false);
    }
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
});
