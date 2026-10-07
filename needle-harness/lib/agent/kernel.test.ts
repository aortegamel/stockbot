import { describe, expect, test } from "bun:test";
import { KernelRouter, runKernelAgent, type KernelChild } from "./kernel";

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

describe("runKernelAgent evidence loop", () => {
  const graphReply = (id: string): string =>
    `${JSON.stringify({ id, objective: "q", evidence: [{ id: "ev:1", content: "fact" }], nodes: [{ node_id: "n1", question: "q", status: "blocked", depends_on: [] }], decisions: [], unresolved: ["n1"], incomplete_guard: true, toolExecutions: [], toolCalls: [], failures: {}, escalations: 1, escalated: true })}\n`;
  const graphSpawn = (): KernelChild => {
    const c = new FakeChild();
    c.reply = graphReply;
    queueMicrotask(() => c.emitStdout('{"type":"ready"}\n'));
    return c;
  };
  test("gap triggers one follow-up pass, clean second write ends done", async () => {
    process.env.OPENCODE_MODEL = "test-model";
    const events: Array<{ type: string; stage?: string }> = [];
    let writes = 0;
    await runKernelAgent("q", (e) => void events.push(e), {
      deps: {
        python: "py",
        workerPath: "w",
        spawnFn: graphSpawn,
        reason: async (o) => {
          writes += 1;
          o.onDelta("text");
          return writes === 1
            ? { text: "t", usage: {}, missingEvidence: "need doc X" }
            : { text: "t2", usage: {} };
        },
      },
    });
    const types = events.map((e) => e.type);
    expect(types.filter((t) => t === "reasoning_start").length).toBe(2);
    expect(events.some((e) => e.type === "progress" && e.stage === "evidence_gap")).toBe(true);
    expect(types.at(-1)).toBe("done");
    expect(writes).toBe(2);
  });

  test("second gap fails the report with incomplete_evidence", async () => {
    const events: Array<{ type: string; category?: string }> = [];
    await runKernelAgent("q", (e) => void events.push(e as { type: string }), {
      deps: {
        python: "py",
        workerPath: "w",
        spawnFn: graphSpawn,
        reason: async (o) => {
          o.onDelta("text");
          return { text: "t", usage: {}, missingEvidence: "still need doc X" };
        },
      },
    });
    const types = events.map((e) => e.type);
    expect(types.filter((t) => t === "reasoning_start").length).toBe(2);
    expect(types).toContain("failed");
    expect(events.find((e) => e.type === "failed")?.category).toBe("incomplete_evidence");
    expect(types.at(-1)).toBe("done");
  });
});
