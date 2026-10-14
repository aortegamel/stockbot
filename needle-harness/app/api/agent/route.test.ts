import { describe, expect, mock, test } from "bun:test";
import { FINAL_PERSONAS, type Evidence } from "@/lib/agent/types";
// Real validator captured before mock.module replaces the kernel exports; route.ts must see the production rule.
import { asksWatchlist, parseTicker, parseWatchlist } from "@/lib/agent/kernel";
const realParseTicker = parseTicker;
const realParseWatchlist = parseWatchlist;
const realAsksWatchlist = asksWatchlist;

// Mutable stubs reconfigured per test; route.ts is imported dynamically after
// mock.module setup because static imports would bind the real kernel/muse
// modules before the mocks install (module-loading boundary test).
let winner: unknown = "reasoning_required";
let routeDown = false;
let argsDown = false;
let reasonDown = false;
let verdicts: string[] = ["node_resolved"];
let personas: unknown = null;
// Replaces the whole op:route reply when set (selector errors, malformed shapes).
let routeReply: Record<string, unknown> | null = null;
// Replaces the op:arguments reply when set (error or malformed generation shapes).
let argsReply: Record<string, unknown> | null = null;
// runKernelAgent throws this value when non-null.
let runThrows: unknown = null;
const runCalls: unknown[][] = [];
type ReasonCall = { prompt: string; evidence: Evidence[]; onDelta?: (t: string) => void };
const reasonCalls: ReasonCall[] = [];
type KernelCall = { op?: unknown; tool?: unknown; prompt?: unknown; objective?: unknown; ticker?: unknown; watchlist?: unknown };
const kernelCalls: KernelCall[] = [];
const invoked: unknown[][] = [];
const ended: string[] = [];

let emitProgress = false;
let emitEvidence = false;
const capturedSource = { id: `capture:${"a".repeat(64)}`, title: "Captured filing", url: "https://www.sec.gov/filing" };
const bridgeEvidence: Evidence = {
  id: "ev:bridge", source: "bridge", retrievedAt: "2026-01-01T00:00:00Z", content: "internal tool output",
  title: "get_sec_document", url: "https://www.sec.gov/tool-output",
  sourceHandle: { raw_archive_path: "/private/raw" }, sourceRefs: { internal: "log" }, sources: [capturedSource],
};
mock.module("@/lib/agent/kernel", () => ({
  REQUEST_WALL_MS: 120_000,
  WATCHLIST_BUDGET_MS: 15_000,
  parseTicker: realParseTicker,
  parseWatchlist: realParseWatchlist,
  asksWatchlist: realAsksWatchlist,
  kernelRouter: {
    call: async (body: KernelCall) => {
      kernelCalls.push(body);
      if (routeDown) throw new Error("worker down");
      if (body.op === "arguments") {
        if (argsDown) throw new Error("needle down");
        return argsReply ?? { tool: body.tool, arguments: { q: "x" }, confidence: 1, reasoning: "t" };
      }
      if (body.op === "assess_entry") {
        const verdict = verdicts.length > 1 ? verdicts.shift()! : verdicts[0];
        return { verdict };
      }
      return routeReply ?? { route: winner, personas };
    },
  },
  runKernelAgent: async (...args: unknown[]) => {
    runCalls.push(args);
    if (runThrows !== null) throw runThrows;
    if (emitProgress) {
      const emit = args[1] as (e: Record<string, unknown>) => void;
      emit({ type: "progress", stage: "intake_done", detail: { calls: 3 } });
    }
    if (emitEvidence) {
      const emit = args[1] as (e: Record<string, unknown>) => void;
      emit({ type: "evidence", evidence: [bridgeEvidence], sessionId: "rs:test" });
      emit({ type: "answer_delta", text: "A fact [ev:bridge]" });
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
    return { ok: true, evidence: { ...bridgeEvidence, source: name } };
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
  emitEvidence = false;
  verdicts = ["node_resolved"];
  personas = null;
  routeReply = null;
  argsReply = null;
  runThrows = null;
  runCalls.length = 0;
  reasonCalls.length = 0;
  kernelCalls.length = 0;
  invoked.length = 0;
  ended.length = 0;
  process.env.STOCKBOT_DEBUG = "1";
  process.env.OPENCODE_MODEL = "test-model";
}

type AgentEventShape = { type: string;[k: string]: unknown };

async function eventsFor(prompt: string, ticker?: unknown, watchlist?: unknown): Promise<AgentEventShape[]> {
  const { POST } = await import("./route");
  const req = new Request("http://localhost/api/agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, ...(ticker === undefined ? {} : { ticker }), ...(watchlist === undefined ? {} : { watchlist }) }),
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
    expect(types).toEqual(["agent_start", "needle_decision", "tool_start", "tool_result", "evidence", "reasoning_start", "answer_delta", "done"]);
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

  test("route outage fails without research or an answer", async () => {
    reset();
    routeDown = true;
    const events = await eventsFor("hello");
    expect(events.map((e) => e.type)).toEqual(["agent_start", "failed", "error"]);
    expect(runCalls.length).toBe(0);
    expect(reasonCalls.length).toBe(0);
  });

  test("selector error reply fails without guessing personas", async () => {
    reset();
    routeReply = { error: "persona selector failed" };
    const events = await eventsFor("Bearbot and Stockbot on NVDA");
    expect(events.map((e) => e.type)).toEqual(["agent_start", "failed", "error"]);
    expect(String(events.find((e) => e.type === "failed")?.message)).toBe("persona selector failed");
    expect(runCalls.length).toBe(0);
    expect(reasonCalls.length).toBe(0);
  });

  test("malformed route or persona selection fails rather than defaulting", async () => {
    const bad: Record<string, unknown>[] = [
      { route: "research_required" },
      { personas: null },
      { route: 7, personas: null },
      { route: "research_required", personas: [] },
      { route: "research_required", personas: ["stockbot", "stockbot"] },
      { route: "research_required", personas: ["bullbot", "stockbot"] },
      { route: "research_required", personas: ["Stockbot"] },
      { route: "research_required", personas: ["stockbot", "chartbot"] },
      { route: "research_required", personas: "stockbot" },
      { route: "reasoning_required", personas: [1] },
    ];
    for (const reply of bad) {
      reset();
      routeReply = reply;
      const events = await eventsFor("hello");
      expect(events.map((e) => e.type)).toEqual(["agent_start", "failed", "error"]);
      expect(runCalls.length).toBe(0);
      expect(reasonCalls.length).toBe(0);
      expect(kernelCalls.filter((c) => c.op === "arguments").length).toBe(0);
    }
  });

  test("every explicit persona subset enters research with that selection", async () => {
    for (let mask = 1; mask < 1 << FINAL_PERSONAS.length; mask++) {
      const subset = FINAL_PERSONAS.filter((_, i) => (mask >> i) & 1);
      for (const route of ["reasoning_required", "no_session", "get_current_time", "research_required"]) {
        reset();
        winner = route;
        personas = subset;
        await eventsFor("Bearbot and Stockbot views on NVDA");
        expect(runCalls.length).toBe(1);
        expect((runCalls[0]?.[2] as { personas?: unknown }).personas).toEqual(subset);
        expect(reasonCalls.length).toBe(0);
        expect(kernelCalls.filter((c) => c.op === "arguments").length).toBe(0);
        expect(invoked.length).toBe(0);
      }
    }
  });

  test("no explicit selection researches with the default personas", async () => {
    reset();
    winner = "research_required";
    await eventsFor("What drove NVDA revenue?");
    expect(runCalls.length).toBe(1);
    expect("personas" in (runCalls[0]?.[2] as object)).toBe(false);
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
    expect(prod.map((e) => e.type)).toEqual(["agent_start", "evidence", "reasoning_start", "answer_delta", "done"]);
    reset();
    winner = "get_current_time";
    const debug = await eventsFor("what time is it?");
    expect(debug.map((e) => e.type)).toEqual([
      "agent_start",
      "needle_decision",
      "tool_start",
      "tool_result",
      "evidence",
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
  test("production single-shot delivers archived sources before the answer without internal logs", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "get_sec_document";
    const events = await eventsFor("filing facts?");
    const delivered = events.find((e) => e.type === "evidence");
    expect(delivered?.evidence).toEqual([{ id: "ev:bridge", source: "get_sec_document", title: "get_sec_document", url: "https://www.sec.gov/tool-output", retrievedAt: bridgeEvidence.retrievedAt, content: "", sources: [capturedSource] }]);
    expect(events.findIndex((e) => e.type === "evidence")).toBeLessThan(events.findIndex((e) => e.type === "answer_delta"));
    expect(reasonCalls[0].evidence[0].content).toBe("internal tool output");
    expect(JSON.stringify(events)).not.toContain("/private/raw");
    expect(JSON.stringify(events)).not.toContain("internal tool output");
  });

  test("production research retains evidence events without internal logs", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "research_required";
    emitEvidence = true;
    const events = await eventsFor("filing facts?");
    expect(events.map((e) => e.type)).toEqual(["evidence", "answer_delta"]);
    expect(events[0].sessionId).toBe("rs:test");
    expect(events[0].evidence).toEqual([{ id: "ev:bridge", source: "bridge", title: "get_sec_document", url: "https://www.sec.gov/tool-output", retrievedAt: bridgeEvidence.retrievedAt, content: "", sources: [capturedSource] }]);
  });
  test("single-shot fallback retains archived references in the research seed", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "get_sec_document";
    verdicts = ["research_required"];
    const events = await eventsFor("filing facts?");
    const seedOpts = runCalls[0]?.[2];
    if (!seedOpts || typeof seedOpts !== "object" || !("seedEvidence" in seedOpts)) throw new Error("expected seedEvidence in runKernelAgent opts");
    const seed = seedOpts.seedEvidence;
    if (!Array.isArray(seed) || seed.length === 0) throw new Error("expected nonempty seedEvidence");
    const first = seed[0];
    if (!first || typeof first !== "object") throw new Error("expected seed evidence row");
    if (!("sources" in first) || !("content" in first)) throw new Error("expected sources+content in seed row");
    expect(first.sources).toEqual([capturedSource]);
    expect(first.content).toBe(bridgeEvidence.content);
    expect(events.find((e) => e.type === "evidence")?.evidence).toEqual([{ id: "ev:bridge", source: "get_sec_document", title: "get_sec_document", url: "https://www.sec.gov/tool-output", retrievedAt: bridgeEvidence.retrievedAt, content: "", sources: [capturedSource] }]);
  });

  test("selected ticker forwards verbatim prompt on every bridge call", async () => {
    reset();
    winner = "get_sec_document";
    verdicts = ["research_required"];
    await eventsFor("NVDA risk factors?", "nvda");
    const routeCall = kernelCalls.find((c) => c.op === "route");
    const argsCall = kernelCalls.find((c) => c.op === "arguments");
    const assessCall = kernelCalls.find((c) => c.op === "assess_entry");
    if (!routeCall || !argsCall || !assessCall) throw new Error("expected route/arguments/assess_entry calls");
    expect(routeCall.prompt).toBe("NVDA risk factors?");
    expect(routeCall.ticker).toBe("NVDA");
    expect(argsCall.objective).toBe("NVDA risk factors?");
    expect(argsCall.ticker).toBe("NVDA");
    expect(assessCall.prompt).toBe("NVDA risk factors?");
    expect(assessCall.ticker).toBe("NVDA");
    expect(runCalls.length).toBe(1);
    const runOpts = runCalls[0]?.[2];
    if (!runOpts || typeof runOpts !== "object" || !("ticker" in runOpts)) throw new Error("expected ticker in runKernelAgent opts");
    expect(runOpts.ticker).toBe("NVDA");
  });

  test("invalid ticker drops without touching the prompt", async () => {
    for (const bad of ["SEC", "nvda!", "TOOLONGTICKER12", 7]) {
      reset();
      winner = "research_required";
      await eventsFor("What drove NVDA revenue?", bad);
      const routeCall = kernelCalls.find((c) => c.op === "route");
      if (!routeCall) throw new Error("expected route call");
      expect(routeCall.prompt).toBe("What drove NVDA revenue?");
      expect("ticker" in routeCall).toBe(false);
      expect(runCalls.length).toBe(1);
      const runOpts = runCalls[0]?.[2];
      if (!runOpts || typeof runOpts !== "object") throw new Error("expected runKernelAgent opts");
      expect("ticker" in runOpts).toBe(false);
    }
  });

  test("explicit personas carry the ticker into research", async () => {
    reset();
    winner = "reasoning_required";
    personas = ["stockbot"];
    await eventsFor("Bearbot views on NVDA", "nvda");
    expect(runCalls.length).toBe(1);
    const runOpts = runCalls[0]?.[2];
    if (!runOpts || typeof runOpts !== "object") throw new Error("expected runKernelAgent opts");
    if (!("personas" in runOpts) || !("ticker" in runOpts)) throw new Error("expected personas+ticker in runKernelAgent opts");
    expect(runOpts.personas).toEqual(["stockbot"]);
    expect(runOpts.ticker).toBe("NVDA");
  });

  test("invalid ticker drops from single-shot arguments and assess calls", async () => {
    for (const bad of ["SEC", "1AB", "  ", null]) {
      reset();
      winner = "get_sec_document";
      verdicts = ["research_required"];
      await eventsFor("NVDA risk factors?", bad);
      const ops = kernelCalls.map((c) => c.op);
      expect(ops).toEqual(["route", "arguments", "assess_entry"]);
      for (const call of kernelCalls) expect("ticker" in call).toBe(false);
      const argsCall = kernelCalls.find((c) => c.op === "arguments");
      expect(argsCall?.objective).toBe("NVDA risk factors?");
      expect(argsCall?.tool).toBe("get_sec_document");
    }
  });

  test("research deadline sits one request wall in the future", async () => {
    reset();
    winner = "research_required";
    const before = Date.now();
    await eventsFor("What drove NVDA revenue?");
    const after = Date.now();
    const runOpts = runCalls[0]?.[2];
    if (!runOpts || typeof runOpts !== "object" || !("deadlineAt" in runOpts)) throw new Error("expected deadlineAt in runKernelAgent opts");
    const deadlineAt = Number(runOpts.deadlineAt);
    expect(deadlineAt).toBeGreaterThanOrEqual(before + 120_000);
    expect(deadlineAt).toBeLessThanOrEqual(after + 120_000);
  });

  test("malformed or errored argument generation never invokes and falls through to research", async () => {
    const bad: Record<string, unknown>[] = [
      { error: "args failed", tool: "query_finra", arguments: { q: "x" } },
      { tool: "get_sec_document", arguments: { q: "x" } },
      { tool: "query_finra", arguments: null },
      { tool: "query_finra", arguments: "q=x" },
    ];
    for (const reply of bad) {
      reset();
      winner = "query_finra";
      argsReply = reply;
      await eventsFor("short interest?", "nvda");
      expect(invoked.length).toBe(0);
      expect(reasonCalls.length).toBe(0);
      expect(runCalls.length).toBe(1);
      expect(kernelCalls.find((c) => c.op === "arguments")?.ticker).toBe("NVDA");
    }
  });

  test("empty argument error string still invokes the tool", async () => {
    reset();
    winner = "query_finra";
    argsReply = { error: "", tool: "query_finra", arguments: { q: "x" } };
    await eventsFor("short interest?");
    expect(invoked.length).toBe(1);
    expect(invoked[0]?.[1]).toEqual({ q: "x" });
    expect(runCalls.length).toBe(0);
  });

  test("route failure messages keep the worker error or the malformed default", async () => {
    const cases: { down?: boolean; reply?: Record<string, unknown>; message: string }[] = [
      { down: true, message: "worker down" },
      { reply: { error: "" }, message: "route_invalid: malformed route selection" },
      { reply: { error: 7 }, message: "route_invalid: malformed route selection" },
    ];
    for (const c of cases) {
      reset();
      routeDown = c.down === true;
      routeReply = c.reply ?? null;
      const events = await eventsFor("hello", "nvda");
      expect(events.map((e) => e.type)).toEqual(["agent_start", "failed", "error"]);
      expect(events.find((e) => e.type === "failed")?.message).toBe(c.message);
      expect(events.find((e) => e.type === "error")?.message).toBe(c.message);
      expect(kernelCalls[0]?.ticker).toBe("NVDA");
    }
  });

  test("research throw ends the stream with the thrown message", async () => {
    for (const [thrown, message] of [[new Error("kernel boom"), "kernel boom"], ["raw boom", "raw boom"]] as const) {
      reset();
      winner = "research_required";
      runThrows = thrown;
      const events = await eventsFor("What drove NVDA revenue?");
      expect(runCalls.length).toBe(1);
      expect(events.at(-1)).toEqual({ type: "error", message });
    }
  });

  test("watchlist rides the route call as context and JEV's route governs dispatch", async () => {
    // research_required: the run body carries the deduplicated list and no selected ticker.
    reset();
    winner = "research_required";
    await eventsFor("Summarize the latest filings for my watchlist", "aapl", ["nvda", "QQQ", "nvda", "spy"]);
    expect(kernelCalls.map((c) => c.op)).toEqual(["route"]);
    const routeCall = kernelCalls[0];
    expect(routeCall?.prompt).toBe("Summarize the latest filings for my watchlist");
    expect(routeCall?.watchlist).toEqual(["NVDA", "QQQ", "SPY"]);
    expect(routeCall && "ticker" in routeCall).toBe(false);
    expect(reasonCalls.length).toBe(0);
    expect(runCalls.length).toBe(1);
    expect(runCalls[0]?.[0]).toBe("Summarize the latest filings for my watchlist");
    const runOpts = runCalls[0]?.[2] as Record<string, unknown>;
    expect(runOpts.watchlist).toEqual(["NVDA", "QQQ", "SPY"]);
    expect("ticker" in runOpts).toBe(false);

    // no_session: the same route resolves direct, with no run and no fabricated watchlist work.
    reset();
    winner = "no_session";
    await eventsFor("Summarize the latest filings for my watchlist", "aapl", ["NVDA", "QQQ"]);
    expect(kernelCalls.map((c) => c.op)).toEqual(["route"]);
    expect(kernelCalls[0]?.watchlist).toEqual(["NVDA", "QQQ"]);
    expect(runCalls.length).toBe(0);
    expect(reasonCalls.length).toBe(1);

    // A registry tool name: the single-shot chain runs, as JEV decided.
    reset();
    winner = "get_sec_document";
    await eventsFor("Summarize the latest filings for my watchlist", "aapl", ["NVDA", "QQQ"]);
    expect(kernelCalls.map((c) => c.op)).toEqual(["route", "arguments", "assess_entry"]);
    expect(runCalls.length).toBe(0);
  });

  test("selected-ticker question ignores the watchlist context", async () => {
    reset();
    winner = "research_required";
    await eventsFor("NVDA risk factors?", "nvda", ["NVDA", "AAPL", "MSFT"]);
    const runOpts = runCalls[0]?.[2] as Record<string, unknown>;
    expect(runOpts.ticker).toBe("NVDA");
    expect("watchlist" in runOpts).toBe(false);
    expect(kernelCalls[0]?.ticker).toBe("NVDA");
    expect(kernelCalls[0] && "watchlist" in kernelCalls[0]).toBe(false);
  });

  test("malformed watchlist context is rejected before any worker call", async () => {
    const { POST } = await import("./route");
    for (const bad of [["SEC"], "NVDA", [], [7], ["NVDA", ""], Array.from({ length: 26 }, (_, i) => `T${i}`)]) {
      reset();
      const res = await POST(
        new Request("http://localhost/api/agent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: "my watchlist filings", ticker: "NVDA", watchlist: bad }),
        }),
      );
      expect(res.status).toBe(400);
      expect(await res.json()).toEqual({ error: "watchlist must list 1-25 valid tickers" });
      expect(kernelCalls.length).toBe(0);
      expect(runCalls.length).toBe(0);
    }
  });
});
