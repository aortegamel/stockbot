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
  const idOf = (data: string): unknown => {
    const body: unknown = JSON.parse(data);
    if (body && typeof body === "object" && "id" in body && typeof body.id === "string") return body.id;
    return undefined;
  };
  const routeReply = (id: unknown): string =>
    JSON.stringify({ id, tool: null, arguments: {}, withheld: false, confidence: null, reasoning: "" });
  const expectedRoute = { tool: null, arguments: {}, withheld: false, confidence: null, reasoning: "" };

  type FakeEntry = { fake: FakeChild; onLine: (line: string) => void };
  function makeFake(opts?: { hangGeneration?: boolean }): FakeEntry {
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
          const id = idOf(data);
          if (actionOf(data) === "ping") {
            queueMicrotask(() => onLine(JSON.stringify({ id, ready: true })));
          } else if (!opts?.hangGeneration) {
            queueMicrotask(() => onLine(routeReply(id)));
          }
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
});
