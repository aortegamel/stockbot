import { describe, expect, test, vi } from "bun:test";
import type { ChildProcess } from "node:child_process";
import { NeedleRouter, acquireNeedle, validateNeedleTool, type NeedleSpawn } from "./client";

describe("acquireNeedle", () => {
  test("concurrent holders serialize", async () => {
    const order: string[] = [];
    const first = await acquireNeedle();
    order.push("A-enter");
    let bEntered = false;
    const bGrant = acquireNeedle();
    const bWait = bGrant.then(async (releaseB) => {
      bEntered = true;
      order.push("B-enter");
      order.push("B-exit");
      releaseB();
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(bEntered).toBe(false);
    order.push("A-exit");
    first();
    await bWait;
    expect(order).toEqual(["A-enter", "A-exit", "B-enter", "B-exit"]);
  });

  test("throw still grants next holder", async () => {
    const first = await acquireNeedle();
    try {
      throw new Error("boom");
    } catch {
      // Fall through; release must still run.
    } finally {
      first();
    }
    const second = await acquireNeedle();
    try {
      expect(true).toBe(true);
    } finally {
      second();
    }
  });
});

describe("validateNeedleTool", () => {
  test("exact match passes; mismatch and null throw", () => {
    expect(validateNeedleTool("search_sec_filings", "search_sec_filings")).toBe("search_sec_filings");
    expect(() => validateNeedleTool("search_sec_filings", "get_sec_document")).toThrow("mismatch");
    expect(() => validateNeedleTool("search_sec_filings", null)).toThrow("mismatch");
    expect(() => validateNeedleTool("", "search_sec_filings")).toThrow("nonempty");
  });
});

describe("needle lifecycle (replaces keep-alive timeout test)", () => {
  // Replacement reason: the old timeout-isolation test asserted a timeout leaves
  // the worker alive while a sibling still resolves. The approved tree-termination
  // contract ends that behavior: a generation timeout kills and reaps the owned
  // server tree, rejects all pending work, and the next request spawns a fresh
  // ready worker. Cold startup (ping ready, 120s budget) never consumes the 10s
  // generation deadline. Ping protocol reused unchanged (server action=ping).
  // Timing is deterministic: write-order assertions plus vi fake timers driving
  // the production timeout clock. No wall-clock sleep anywhere in this block.
  type FakeChild = {
    exitCode: number | null;
    pid?: number;
    killCalls: number;
    written: string[];
    kill: (signal?: string) => void;
    stdin: { write: (data: string, cb?: (err?: Error | null) => void) => void } | null;
    stdout: { on: (event: "data", listener: (chunk: Buffer) => void) => void } | null;
    stderr: { on: (event: "data", listener: (chunk: Buffer) => void) => void } | null;
    handlers: Record<string, ((arg?: unknown) => void)[]>;
    on: (event: "error" | "exit", listener: (arg?: unknown) => void) => void;
    emitStdout: (s: string) => void;
  };

  const actionOf = (data: string): unknown => {
    const body: unknown = JSON.parse(data);
    if (body && typeof body === "object" && "action" in body) return body.action;
    return undefined;
  };
  const routeReply = (id: unknown): string =>
    JSON.stringify({ id, tool: null, arguments: {}, withheld: false, confidence: null, reasoning: "" });
  const expectedRoute = { tool: null, arguments: {}, withheld: false, confidence: null, reasoning: "" };

  type FakeReply = (body: Record<string, unknown>) => unknown;
  type FakeOpts = { hangGeneration?: boolean; echoTool?: boolean; reply?: FakeReply; ping?: FakeReply };

  // Protocol reply for one request body; undefined keeps the worker silent.
  function protocolReply(body: Record<string, unknown>, opts?: FakeOpts): unknown {
    if (body.action === "ping") return opts?.ping ? opts.ping(body) : { id: body.id, ready: true };
    if (opts?.hangGeneration) return undefined;
    if (opts?.reply) return opts.reply(body);
    const tool = typeof body.tool === "string" ? body.tool : null;
    return { id: body.id, tool: opts?.echoTool ? tool : null, arguments: {}, withheld: false, confidence: null, reasoning: "" };
  }

  type FakeEntry = { fake: FakeChild; onLine: (line: string) => void };
  function makeFake(opts?: FakeOpts): FakeEntry {
    const stdoutListeners: ((chunk: Buffer) => void)[] = [];
    let onLine: (line: string) => void = () => { };
    const fake: FakeChild = {
      exitCode: null,
      killCalls: 0,
      written: [],
      kill: () => {
        fake.killCalls += 1;
      },
      stdin: {
        write: (data: string, cb?: (err?: Error | null) => void): void => {
          fake.written.push(data);
          const reply = protocolReply(JSON.parse(data) as Record<string, unknown>, opts);
          if (reply !== undefined) queueMicrotask(() => onLine(JSON.stringify(reply)));
          cb?.(null);
        },
      },
      stdout: {
        on: (_event: "data", listener: (chunk: Buffer) => void): void => {
          stdoutListeners.push(listener);
        },
      },
      stderr: { on: (): void => { } },
      handlers: {},
      on: (event: "error" | "exit", listener: (arg?: unknown) => void): void => {
        fake.handlers[event] ??= [];
        fake.handlers[event].push(listener);
      },
      emitStdout: (s: string): void => {
        for (const l of stdoutListeners) l(Buffer.from(s));
      },
    };
    onLine = (line: string): void => {
      fake.emitStdout(`${line}\n`);
    };
    return { fake, onLine };
  }

  function setup(spawnBehavior: (n: number) => FakeChild): { router: NeedleRouter; children: FakeChild[] } {
    const children: FakeChild[] = [];
    let spawnCount = 0;
    const spawnFn: NeedleSpawn = (): ChildProcess => {
      spawnCount += 1;
      const fake = spawnBehavior(spawnCount);
      children.push(fake);
      return fake as unknown as ChildProcess;
    };
    return { router: new NeedleRouter({ spawnFn }), children };
  }

  // Unchecked cast: pending has no public seam; key-count read only.
  const pendingKeys = (router: NeedleRouter): string[] => {
    const state: { pending: Record<string, unknown> } = router as unknown as { pending: Record<string, unknown> };
    return Object.keys(state.pending);
  };

  test("cold startup does not consume the generation deadline", async () => {
    const { router, children } = setup(() => makeFake().fake);
    try {
      type RouterSeam = { rawCall: (b: Record<string, unknown>, k: string, t: number) => Promise<unknown> };
      // Unchecked cast: rawCall has no public seam; route behavior under test.
      const seam: RouterSeam = router as unknown as RouterSeam;
      // No wall clock: ping and generation replies arrive on microtasks, so any
      // implementation resolves. The discriminator is write order: the ping
      // readiness write precedes the generation write on the fresh child.
      const res = await seam.rawCall({ action: "arguments.generate", tool: "search_sec_filings" }, "route", 150);
      expect(res).toEqual(expectedRoute);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      expect(children.length).toBe(1);
      const actions = first.written.map(actionOf);
      expect(actions[0]).toBe("ping");
      expect(actions).toContain("arguments.generate");
      first.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  test("generation timeout kills the owned worker tree and rejects pending work", async () => {
    // Timeout firing is the behavior under test; advanceTimersByTime drives the
    // production clock deterministically with no wall-clock sleep.
    vi.useFakeTimers();
    try {
      const { router, children } = setup(() => makeFake({ hangGeneration: true }).fake);
      type RouterSeam = {
        rawCall: (b: Record<string, unknown>, k: string, t: number) => Promise<unknown>;
        onLine: (l: string) => void;
      };
      // Unchecked cast: rawCall/onLine have no public seam; route behavior under test.
      const seam: RouterSeam = router as unknown as RouterSeam;
      const slow = seam.rawCall({ action: "slow" }, "route", 30);
      // Observe upfront so the sibling rejection cannot go unhandled.
      const slowSettled = slow.then(
        (): null => null,
        (e: unknown): unknown => e,
      );
      const siblingSettled = seam
        .rawCall({ action: "sibling" }, "route", 1000)
        .then(
          (): "resolved" => "resolved",
          (): "rejected" => "rejected",
        );
      // Flush the async readiness handshake: both calls queue behind the ping
      // reply and share the hung worker's fate once the clock advances.
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      vi.advanceTimersByTime(30);
      const slowErr = await slowSettled;
      if (!(slowErr instanceof Error)) throw new Error("expected timeout rejection");
      expect(slowErr.message).toMatch(/timeout/i);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      expect(children.length).toBe(1);
      expect(first.killCalls).toBeGreaterThanOrEqual(1);
      expect(await siblingSettled).toBe("rejected");
      expect(pendingKeys(router)).toEqual([]);
      await router.close();
    } finally {
      vi.useRealTimers();
    }
  });

  test("next request after a timeout uses a fresh ready worker", async () => {
    // Timeout firing is the behavior under test; advanceTimersByTime drives the
    // production clock deterministically with no wall-clock sleep.
    vi.useFakeTimers();
    try {
      const { router, children } = setup((n) => makeFake({ hangGeneration: n === 1 }).fake);
      type RouterSeam = {
        rawCall: (b: Record<string, unknown>, k: string, t: number) => Promise<unknown>;
        onLine: (l: string) => void;
      };
      // Unchecked cast: rawCall/onLine have no public seam; route behavior under test.
      const seam: RouterSeam = router as unknown as RouterSeam;
      const firstCall = seam.rawCall({ action: "arguments.generate", tool: "search_sec_filings" }, "route", 30);
      const firstSettled = firstCall.then(
        (): null => null,
        (e: unknown): unknown => e,
      );
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      vi.advanceTimersByTime(30);
      const firstErr = await firstSettled;
      if (!(firstErr instanceof Error)) throw new Error("expected timeout rejection");
      expect(firstErr.message).toMatch(/timeout/i);
      expect(children.length).toBe(1);
      const first = children[0];
      if (!first) throw new Error("expected first worker");
      expect(first.killCalls).toBeGreaterThanOrEqual(1);
      // Stale data from the dead worker cannot corrupt the replacement: the
      // late reply for the timed-out id drops with no pending entry.
      seam.onLine(routeReply("2"));
      expect(pendingKeys(router)).toEqual([]);
      const res = await seam.rawCall({ action: "arguments.generate", tool: "search_sec_filings" }, "route", 500);
      expect(res).toEqual(expectedRoute);
      expect(children.length).toBe(2);
      const second = children[1];
      if (!second) throw new Error("expected fresh worker");
      expect(second.killCalls).toBe(0);
      expect(second.written.map(actionOf)).toContain("ping");
      expect(second.written.map(actionOf)).toContain("arguments.generate");
      second.exitCode = 0;
      await router.close();
    } finally {
      vi.useRealTimers();
    }
  });

  test("generateArguments transmits only the tool boundary", async () => {
    const { router, children } = setup(() => makeFake({ echoTool: true }).fake);
    try {
      const tool = "search_sec_filings";
      const schema = {
        type: "object",
        properties: { ticker: {}, as_of: {}, tags: {}, count: {}, rows: {}, mixed: {} },
      };
      const marker = `MARKER_${"x".repeat(4096)}`;
      const req = {
        tool,
        schema,
        objective: "objective text",
        node: {
          question: "What ticker?",
          node_id: "n1",
          id: "i1",
          type: "t",
          attempts: [{ marker }],
          results: marker,
          feedback: marker,
          evidence: marker,
          reasoning: marker,
          intake: marker,
        },
        context: {
          as_of: "2026-01-01",
          temporal_scope: "scope",
          today_utc: "today",
          ticker: "T",
          grounded_arguments: {
            ticker: "T",
            as_of: "2026-01-01",
            tags: ["a", "b", 1, null],
            count: Number.NaN,
            rows: [{ a: 1 }],
            mixed: ["a", { b: 2 }],
            nested: { a: 1 },
            evil: marker,
          },
          evidence: marker,
          attempts: [{ marker }],
          feedback: marker,
          intake: marker,
          reasoning: marker,
          results: marker,
        },
        evidence: marker,
        feedback: marker,
        intake: marker,
        results: marker,
      } as unknown as {
        tool: string;
        schema?: unknown;
        objective?: unknown;
        node?: unknown;
        context?: unknown;
      };
      const res = await router.generateArguments(req);
      expect(res).toEqual({ tool, arguments: {}, withheld: false, confidence: null, reasoning: "" });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      const bodies = first.written.map((w) => JSON.parse(w) as Record<string, unknown>);
      const gen = bodies.find((b) => b.action === "arguments.generate");
      if (!gen) throw new Error("expected arguments.generate write");
      expect(Object.keys(gen).sort()).toEqual(["action", "context", "id", "node", "objective", "schema", "tool"]);
      expect(gen.tool).toBe(tool);
      expect(gen.schema).toEqual(schema);
      expect(gen.objective).toBe("objective text");
      expect(gen.node).toEqual({ question: "What ticker?", node_id: "n1", id: "i1", type: "t" });
      const context = gen.context as Record<string, unknown>;
      expect(context.as_of).toBe("2026-01-01");
      expect(context.temporal_scope).toBe("scope");
      expect(context.today_utc).toBe("today");
      expect(context.ticker).toBe("T");
      expect(context.grounded_arguments).toEqual({ ticker: "T", as_of: "2026-01-01", tags: ["a", "b", 1, null] });
      for (const raw of first.written) expect(raw).not.toContain(marker.slice(0, 64));
      first.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  type GenReq = { tool: string; schema?: unknown; objective?: unknown; node?: unknown; context?: unknown };
  // Sends each request through one router and returns the arguments.generate bodies in call order.
  async function generatedBodies(reqs: GenReq[]): Promise<Record<string, unknown>[]> {
    const { router, children } = setup(() => makeFake({ echoTool: true }).fake);
    try {
      for (const req of reqs) expect((await router.generateArguments(req)).tool).toBe(req.tool);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.exitCode = 0;
      return first.written.map((w) => JSON.parse(w) as Record<string, unknown>).filter((b) => b.action === "arguments.generate");
    } finally {
      await router.close();
    }
  }

  test("generateArguments drops node and context that are not plain objects", async () => {
    const tool = "search_sec_filings";
    const schema = { type: "object", properties: { ticker: {} } };
    const shapes: unknown[] = [null, 0, "question", ["question", "as_of"], ["x"]];
    const bodies = await generatedBodies(shapes.map((s) => ({ tool, schema, objective: "o", node: s, context: s })));
    expect(bodies.length).toBe(shapes.length);
    for (const body of bodies) {
      expect(Object.keys(body).sort()).toEqual(["action", "id", "objective", "schema", "tool"]);
      expect(body.tool).toBe(tool);
      expect(body.objective).toBe("o");
      expect(body.schema).toEqual(schema);
    }
  });

  test("generateArguments keeps partial plain node and context objects", async () => {
    const tool = "search_sec_filings";
    const [body] = await generatedBodies([{ tool, objective: "o", node: { question: "q", extra: 1 }, context: { ticker: "T", extra: 1 } }]);
    if (!body) throw new Error("expected one generate body");
    expect(Object.keys(body).sort()).toEqual(["action", "context", "id", "node", "objective", "tool"]);
    expect(body.node).toEqual({ question: "q" });
    expect(body.context).toEqual({ ticker: "T" });
  });

  test("generateArguments drops grounded arguments without a plain grounded map and schema properties", async () => {
    const tool = "search_sec_filings";
    const props = { "0": {}, "1": {}, ticker: {} };
    const cases: { schema: unknown; grounded: unknown }[] = [
      { schema: { properties: props }, grounded: ["x", "y"] },
      { schema: { properties: props }, grounded: "ab" },
      { schema: { properties: props }, grounded: null },
      { schema: { properties: props }, grounded: 5 },
      { schema: null, grounded: { ticker: "T" } },
      { schema: ["ticker"], grounded: { ticker: "T" } },
      { schema: "ticker", grounded: { "0": "x", ticker: "T" } },
      { schema: { properties: null }, grounded: { ticker: "T" } },
      { schema: { properties: ["ticker"] }, grounded: { "0": "x", ticker: "T" } },
      { schema: { properties: "ticker" }, grounded: { "0": "x", ticker: "T" } },
      { schema: { properties: 0 }, grounded: { ticker: "T" } },
    ];
    const bodies = await generatedBodies(cases.map((c) => ({ tool, schema: c.schema, context: { as_of: "d", grounded_arguments: c.grounded } })));
    expect(bodies.length).toBe(cases.length);
    for (const body of bodies) expect(body.context).toEqual({ as_of: "d" });
  });

  test("generateArguments admits only finite scalar grounded values named by the schema", async () => {
    const tool = "search_sec_filings";
    const schema = { properties: { flag: {}, off: {}, n: {}, zero: {}, s: {}, empty: {}, nul: {}, arr: {}, inf: {}, nan: {}, obj: {} } };
    const grounded = {
      flag: true, off: false, n: 2.5, zero: 0, s: "x", empty: "", nul: null, arr: [true, 1.5, null, "y"],
      inf: Number.POSITIVE_INFINITY, nan: Number.NaN, obj: { a: 1 }, unlisted: "z",
    };
    const [body] = await generatedBodies([{ tool, schema, context: { grounded_arguments: grounded } }]);
    if (!body) throw new Error("expected one generate body");
    expect(body.context).toEqual({
      grounded_arguments: { flag: true, off: false, n: 2.5, zero: 0, s: "x", empty: "", nul: null, arr: [true, 1.5, null, "y"] },
    });
  });

  test("generateArguments rejects a reply for another tool and releases the needle lock", async () => {
    const { router, children } = setup(() => makeFake().fake);
    try {
      await expect(router.generateArguments({ tool: "search_sec_filings", objective: "o" })).rejects.toThrow("mismatch");
      const release = await acquireNeedle();
      release();
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  test("extractFields sends only the record boundary and returns the typed reply", async () => {
    const { router, children } = setup(() =>
      makeFake({
        reply: (body) => ({ id: body.id, record: "filing", fields: { ticker: "T" }, withheld: true, confidence: 0.5, reasoning: "cited [E1]" }),
      }).fake,
    );
    try {
      const r = await router.extractFields({ record: { name: "filing", private: "x" }, passage: "  text  " });
      expect(r).toEqual({ record: "filing", fields: { ticker: "T" }, withheld: true, confidence: 0.5, reasoning: "cited [E1]" });
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      const bodies = first.written.map((w) => JSON.parse(w) as Record<string, unknown>);
      const extract = bodies.find((b) => b.action === "extract");
      if (!extract) throw new Error("expected extract write");
      expect(Object.keys(extract).sort()).toEqual(["action", "id", "max_new_tokens", "passage", "record", "strict"]);
      expect(extract.record).toEqual({ name: "filing", private: "x" });
      expect(extract.passage).toBe("  text  ");
      expect(extract.max_new_tokens).toBe(512);
      expect(extract.strict).toBe(true);
      first.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  test("extractFields rejects a foreign record name and malformed fields", async () => {
    const foreign = setup(() => makeFake({ reply: (body) => ({ id: body.id, record: "other", fields: {} }) }).fake);
    const unusable = setup(() => makeFake({ reply: (body) => ({ id: body.id, record: "filing", fields: "filing" }) }).fake);
    const absent = setup(() => makeFake({ reply: (body) => ({ id: body.id, record: "filing" }) }).fake);
    const passing = setup(() => makeFake({ reply: (body) => ({ id: body.id, record: null, fields: { a: 1 } }) }).fake);
    try {
      await expect(foreign.router.extractFields({ record: { name: "filing" }, passage: "p" })).rejects.toThrow("malformed server response");
      await expect(unusable.router.extractFields({ record: { name: "filing" }, passage: "p" })).rejects.toThrow("needle extract failed");
      await expect(absent.router.extractFields({ record: { name: "filing" }, passage: "p" })).rejects.toThrow("needle extract failed");
      // A null record means the server found no record; the caller decides on the fields.
      expect(await passing.router.extractFields({ record: { name: "filing" }, passage: "p" })).toEqual({
        record: null, fields: { a: 1 }, withheld: false, confidence: null, reasoning: "",
      });
    } finally {
      for (const s of [foreign, unusable, absent, passing]) {
        const first = s.children[0];
        if (first) first.exitCode = 0;
        await s.router.close();
      }
    }
  });

  test("extractFields validates the record and passage before any call", async () => {
    const { router, children } = setup(() => makeFake().fake);
    try {
      await expect(router.extractFields({ record: { name: "" }, passage: "p" })).rejects.toThrow("nonempty name");
      await expect(router.extractFields({ record: { name: "filing" }, passage: "   " })).rejects.toThrow("passage must be nonempty");
      await expect(router.embed("   ")).rejects.toThrow("text must be nonempty");
      expect(children.length).toBe(0);
    } finally {
      await router.close();
    }
  });

  test("embed returns the vector and rejects a malformed embedding", async () => {
    const good = setup(() => makeFake({ reply: (body) => ({ id: body.id, embedding: [1, 2.5, -3] }) }).fake);
    const mixed = setup(() => makeFake({ reply: (body) => ({ id: body.id, embedding: [1, "x"] }) }).fake);
    const absent = setup(() => makeFake({ reply: (body) => ({ id: body.id }) }).fake);
    try {
      expect(await good.router.embed("query")).toEqual([1, 2.5, -3]);
      await expect(mixed.router.embed("query")).rejects.toThrow("needle embed failed");
      await expect(absent.router.embed("query")).rejects.toThrow("needle embed failed");
    } finally {
      for (const s of [good, mixed, absent]) {
        const first = s.children[0];
        if (first) first.exitCode = 0;
        await s.router.close();
      }
    }
  });

  test("a server error reply rejects the pending call before its payload", async () => {
    const { router, children } = setup(() =>
      makeFake({ reply: (body) => ({ id: body.id, error: "needle server exploded", tool: "search_sec_filings", arguments: {} }) }).fake,
    );
    try {
      await expect(router.generateArguments({ tool: "search_sec_filings" })).rejects.toThrow("needle server exploded");
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  test("a malformed ready reply fails startup without touching Pending work", async () => {
    const { router, children } = setup(() => makeFake({ ping: (body) => ({ id: body.id }) }).fake);
    try {
      await expect(router.generateArguments({ tool: "search_sec_filings" })).rejects.toThrow("malformed server response");
      expect(pendingKeys(router)).toEqual([]);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      first.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  test("a worker without stdio fails the call instead of waiting", async () => {
    const { router, children } = setup(() => {
      const { fake } = makeFake();
      fake.stdin = null;
      fake.stdout = null;
      return fake;
    });
    try {
      await expect(router.generateArguments({ tool: "search_sec_filings" })).rejects.toThrow("needle spawn failed");
      expect(children.length).toBe(1);
      expect(pendingKeys(router)).toEqual([]);
    } finally {
      await router.close();
    }
  });

  test("a warm ready worker that lost stdin fails the call without respawning", async () => {
    const { router, children } = setup(() => makeFake().fake);
    type RouterSeam = { rawCall: (b: Record<string, unknown>, k: string, t: number) => Promise<unknown> };
    // Unchecked cast: rawCall has no public seam; route behavior under test.
    const seam = router as unknown as RouterSeam;
    try {
      const first = await seam.rawCall({ action: "arguments.generate", tool: "search_sec_filings" }, "route", 500);
      expect(first).toEqual(expectedRoute);
      const worker = children[0];
      if (!worker) throw new Error("expected one spawned child");
      worker.stdin = null;
      await expect(seam.rawCall({ action: "arguments.generate", tool: "search_sec_filings" }, "route", 500)).rejects.toThrow("needle spawn failed");
      expect(children.length).toBe(1);
      worker.exitCode = 0;
    } finally {
      await router.close();
    }
  });

  test("a timeout reaps a worker whose process group is already gone", async () => {
    // Timeout firing is the behavior under test; advanceTimersByTime drives the
    // production clock deterministically with no wall-clock sleep.
    vi.useFakeTimers();
    try {
      const { router, children } = setup(() => {
        const { fake } = makeFake({ hangGeneration: true });
        // Above any reachable pid, so the process-group signal cannot reach a real
        // group and the direct kill fallback must run.
        fake.pid = 2_147_483_647;
        return fake;
      });
      type RouterSeam = { rawCall: (b: Record<string, unknown>, k: string, t: number) => Promise<unknown> };
      // Unchecked cast: rawCall has no public seam; route behavior under test.
      const seam = router as unknown as RouterSeam;
      const slow = seam.rawCall({ action: "slow" }, "route", 30);
      const slowSettled = slow.then(
        (): null => null,
        (e: unknown): unknown => e,
      );
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      vi.advanceTimersByTime(30);
      const err = await slowSettled;
      if (!(err instanceof Error)) throw new Error("expected timeout rejection");
      expect(err.message).toMatch(/timeout/i);
      const first = children[0];
      if (!first) throw new Error("expected one spawned child");
      expect(first.killCalls).toBe(1);
      await router.close();
    } finally {
      vi.useRealTimers();
    }
  });
});
