import { describe, expect, mock, test } from "bun:test";
import type { Evidence } from "@/lib/agent/types";

// Mutable stubs reconfigured per test; route.ts is imported dynamically after
// mock.module setup because static imports would bind the real kernel/muse
// modules before the mocks install (module-loading boundary test).
let winner: unknown = "reasoning_required";
let routeDown = false;
let argsDown = false;
let reasonDown = false;
let verdicts: string[] = ["node_resolved"];
const runCalls: unknown[][] = [];
type ReasonCall = { prompt: string; evidence: Evidence[]; onDelta?: (t: string) => void };
const reasonCalls: ReasonCall[] = [];
type KernelCall = { op?: unknown; tool?: unknown };
const kernelCalls: KernelCall[] = [];
const invoked: unknown[][] = [];
const ended: string[] = [];

let emitProgress = false;
mock.module("@/lib/agent/kernel", () => ({
  REQUEST_WALL_MS: 120_000,
  kernelRouter: {
    call: async (body: KernelCall) => {
      kernelCalls.push(body);
      if (routeDown) throw new Error("worker down");
      if (body.op === "arguments") {
        if (argsDown) throw new Error("needle down");
        return { tool: body.tool, arguments: { q: "x" }, confidence: 1, reasoning: "t" };
      }
      if (body.op === "assess_entry") {
        const verdict = verdicts.length > 1 ? verdicts.shift()! : verdicts[0];
        return { verdict };
      }
      return { route: winner };
    },
  },
  runKernelAgent: async (...args: unknown[]) => {
    runCalls.push(args);
    if (emitProgress) {
      const emit = args[1] as (e: Record<string, unknown>) => void;
      emit({ type: "progress", stage: "intake_done", detail: { calls: 3 } });
    }
  },
}));

mock.module("@/lib/muse/client", () => ({
  reason: async (opts: ReasonCall) => {
    reasonCalls.push(opts);
    if (reasonDown) throw new Error("muse down");
    opts.onDelta?.("hi");
    return { text: "hi", usage: {} };
  },
}));

mock.module("@/lib/tools/stockbot", () => ({
  invoke: async (name: string, args: unknown, sessionId: string) => {
    invoked.push([name, args, sessionId]);
    return { ok: true, evidence: { id: "ev:bridge", source: name, retrievedAt: new Date().toISOString(), content: "bridged" } };
  },
  newSessionId: () => "sess-test",
  endSession: async (id: string) => {
    ended.push(id);
  },
}));

function reset(): void {
  winner = "reasoning_required";
  routeDown = false;
  argsDown = false;
  reasonDown = false;
  emitProgress = false;
  verdicts = ["node_resolved"];
  runCalls.length = 0;
  reasonCalls.length = 0;
  kernelCalls.length = 0;
  invoked.length = 0;
  ended.length = 0;
  process.env.STOCKBOT_DEBUG = "1";
}

type AgentEventShape = { type: string;[k: string]: unknown };

async function eventsFor(prompt: string): Promise<AgentEventShape[]> {
  const { POST } = await import("./route");
  const req = new Request("http://localhost/api/agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const res = await POST(req);
  const text = await res.text();
  return text
    .split("\n\n")
    .filter((c) => c.trim())
    .map((c) => JSON.parse(c.replace(/^data: /, "")) as AgentEventShape);
}

function metricCalls(events: AgentEventShape[], key: string): unknown {
  const done = events.find((e) => e.type === "done");
  if (!done) throw new Error("missing done event");
  const metrics = done.metrics;
  if (!metrics || typeof metrics !== "object" || !(key in metrics)) throw new Error(`missing done metrics ${key}`);
  const section = (metrics as Record<string, unknown>)[key];
  if (!section || typeof section !== "object" || !("calls" in section)) throw new Error(`missing ${key} calls`);
  return section.calls;
}

describe("agent entry route", () => {
  test("reasoning_required answers direct without tools or research", async () => {
    reset();
    const types = (await eventsFor("hello")).map((e) => e.type);
    expect(types).toEqual(["agent_start", "reasoning_start", "answer_delta", "done"]);
    expect(runCalls.length).toBe(0);
    expect(kernelCalls.filter((c) => c.op === "arguments").length).toBe(0);
    expect(reasonCalls[0].evidence).toEqual([]);
  });

  test("no_session answers direct with the same event shape", async () => {
    reset();
    const prev = process.env.OPENCODE_MODEL;
    process.env.OPENCODE_MODEL = "test-model";
    try {
      winner = "no_session";
      const types = (await eventsFor("hello")).map((e) => e.type);
      expect(types).toEqual(["agent_start", "reasoning_start", "answer_delta", "done"]);
      expect(runCalls.length).toBe(0);
      expect(reasonCalls[0].evidence).toEqual([]);
    } finally {
      if (prev === undefined) delete process.env.OPENCODE_MODEL;
      else process.env.OPENCODE_MODEL = prev;
    }
  });

  test("reasoning-style prompts answer direct when routed reasoning_required", async () => {
    for (const prompt of [
      "Explain DCF valuation in simple terms",
      "What does share dilution mean for existing holders?",
      "Explain calls vs puts",
    ]) {
      reset();
      const types = (await eventsFor(prompt)).map((e) => e.type);
      expect(types).toEqual(["agent_start", "reasoning_start", "answer_delta", "done"]);
      expect(runCalls.length).toBe(0);
      expect(kernelCalls.filter((c) => c.op === "arguments").length).toBe(0);
      expect(reasonCalls[0].evidence).toEqual([]);
    }
  });

  test("exact local tool executes once then answers with evidence", async () => {
    reset();
    winner = "get_current_time";
    const events = await eventsFor("what time is it?");
    const types = events.map((e) => e.type);
    expect(types).toEqual(["agent_start", "needle_decision", "tool_start", "tool_result", "reasoning_start", "answer_delta", "done"]);
    expect(events.filter((e) => e.type === "tool_start").length).toBe(1);
    expect(runCalls.length).toBe(0);
    expect(kernelCalls.filter((c) => c.op === "arguments").map((c) => c.tool)).toEqual(["get_current_time"]);
    expect(reasonCalls[0].evidence.length).toBe(1);
    expect(ended).toEqual(["sess-test"]);
    expect(metricCalls(events, "tools")).toBe(1);
  });

  test("tool missing from TS registry executes via bridge invoke", async () => {
    reset();
    winner = "query_finra";
    const events = await eventsFor("short interest?");
    expect(events.filter((e) => e.type === "tool_start").length).toBe(1);
    expect(invoked.length).toBe(1);
    expect(invoked[0][0]).toBe("query_finra");
    expect(runCalls.length).toBe(0);
    expect(reasonCalls[0].evidence.length).toBe(1);
  });

  test("shared-needle failure never invokes and falls through to research", async () => {
    reset();
    winner = "search_sec_filings";
    argsDown = true;
    await eventsFor("NVDA filings?");
    expect(invoked.length).toBe(0);
    expect(reasonCalls.length).toBe(0);
    expect(runCalls.length).toBe(1);
  });

  test("assess research_required verdict falls through to research", async () => {
    reset();
    winner = "query_finra";
    verdicts = ["research_required"];
    await eventsFor("short interest?");
    expect(invoked.length).toBe(1);
    expect(reasonCalls.length).toBe(0);
    expect(runCalls.length).toBe(1);
  });

  test("chained tool verdict runs the next tool then resolves", async () => {
    reset();
    winner = "search_sec_filings";
    verdicts = ["get_sec_document", "node_resolved"];
    const events = await eventsFor("NVDA risk factors?");
    expect(events.filter((e) => e.type === "tool_start").length).toBe(2);
    expect(invoked.map((c) => c[0])).toEqual(["search_sec_filings", "get_sec_document"]);
    expect(runCalls.length).toBe(0);
    expect(reasonCalls[0].evidence.length).toBe(2);
  });

  test("research_required runs the kernel agent", async () => {
    reset();
    winner = "research_required";
    await eventsFor("What drove NVDA revenue?");
    expect(runCalls.length).toBe(1);
    expect(reasonCalls.length).toBe(0);
  });

  test("malformed winner fails open to research", async () => {
    reset();
    winner = "search web; rm -rf";
    await eventsFor("What drove NVDA revenue?");
    expect(runCalls.length).toBe(1);
    expect(reasonCalls.length).toBe(0);
  });

  test("route outage fails open to research", async () => {
    reset();
    routeDown = true;
    await eventsFor("hello");
    expect(runCalls.length).toBe(1);
  });

  test("direct-answer failure ends the stream with a terminal event", async () => {
    reset();
    process.env.STOCKBOT_DEBUG = "1";
    reasonDown = true;
    const events = await eventsFor("hello");
    expect(events.map((e) => e.type)).toEqual(["agent_start", "reasoning_start", "failed", "error"]);
    expect(runCalls.length).toBe(0);
  });

  test("prod mode summarizes: generic working + single error", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    reasonDown = true;
    const events = await eventsFor("hello");
    expect(events.map((e) => e.type)).toEqual(["agent_start", "reasoning_start", "error"]);
    const working = events.find((e) => e.type === "reasoning_start");
    expect(working?.model).toBe("stockbot");
    const terminal = events.find((e) => e.type === "error");
    expect(String(terminal?.message)).toMatch("Stockbot couldn't complete");
  });

  test("prod mode hides internal tool rows, debug shows them", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "get_current_time";
    const prod = await eventsFor("what time is it?");
    expect(prod.map((e) => e.type)).toEqual(["agent_start", "reasoning_start", "answer_delta", "done"]);
    reset();
    winner = "get_current_time";
    const debug = await eventsFor("what time is it?");
    expect(debug.map((e) => e.type)).toEqual([
      "agent_start",
      "needle_decision",
      "tool_start",
      "tool_result",
      "reasoning_start",
      "answer_delta",
      "done",
    ]);
  });
  test("prod forwards stripped progress so the stall watchdog still resets", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "research_required";
    emitProgress = true;
    const events = await eventsFor("What drove NVDA revenue?");
    const progress = events.filter((e) => e.type === "progress");
    expect(progress.length).toBe(1);
    expect(progress[0].stage).toBe("working");
    expect("detail" in progress[0] ? progress[0].detail : undefined).toBeUndefined();
  });
});
