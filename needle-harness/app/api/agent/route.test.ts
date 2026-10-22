import { describe, expect, mock, spyOn, test } from "bun:test";
import fc from "fast-check";
import { FINAL_PERSONAS, type Evidence } from "@/lib/agent/types";
// Real validator captured before mock.module replaces the kernel exports; route.ts must see the production rule.
// The final gates run for real too; only the worker call behind them is scripted.
import { asksWatchlist, dropUncovered, gateAnswer, parseTicker, parseWatchlist, qualifiedEvidenceFree } from "@/lib/agent/kernel";
const realParseTicker = parseTicker;
const realParseWatchlist = parseWatchlist;
const realAsksWatchlist = asksWatchlist;
const realGateAnswer = gateAnswer;
const realDropUncovered = dropUncovered;
const realQualifiedEvidenceFree = qualifiedEvidenceFree;

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
let assessDown = false;
let assessReply: Record<string, unknown> | null = null;
// The worker verdict every assess_final_* op returns.
let gateReply: Record<string, unknown> = { support_answer: "yes" };
const gateBodies: Record<string, unknown>[] = [];
// The reasoner's answer text and claims inventory.
let reasonText = "The answer is unknown.";
let reasonClaims: unknown[] = [{ text: reasonText, claimType: "unknown", evidenceIds: [] }];
let toolFailure: { category: string; error: string } | null = null;
let toolThrows: unknown = null;
let endThrows = false;
// Ids the next generic bridge results carry, in call order; empty means ev:bridge.
let bridgeIds: string[] = [];
let entryReplies: Record<string, unknown>[] = [];
let chartReceipt: unknown = { result_type: "chart_selection", ticker: "MSFT" };
let watchlistReceipt: unknown = { result_type: "watchlist_update", action: "add", ticker: "AMD" };
let widgetReceipt: unknown = { result_type: "widget_selection", widget_type: "short-volume", ticker: "AAPL" };
let emitChartUpdate = false;
const executionOrder: string[] = [];
const runCalls: unknown[][] = [];
const personaCalls: unknown[][] = [];
// runKernelAgent offers the trio for rs:test when true.
let emitOffer = false;
type ReasonCall = { prompt: string; ticker?: string; context?: unknown; evidence: Evidence[]; onDelta?: (t: string) => void };
const reasonCalls: ReasonCall[] = [];
type KernelCall = Record<string, unknown>;
const kernelCalls: KernelCall[] = [];
const kernelOpts: unknown[] = [];
const invokeTimeouts: unknown[] = [];
const invoked: unknown[][] = [];
const ended: string[] = [];
let sessionsCreated = 0;

let emitProgress = false;
let emitEvidence = false;
// runKernelAgent ends with this done metrics object when non-null.
let kernelDone: Record<string, unknown> | null = null;
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
  gateAnswer: realGateAnswer,
  dropUncovered: realDropUncovered,
  qualifiedEvidenceFree: realQualifiedEvidenceFree,
  kernelRouter: {
    call: async (body: KernelCall, opts?: { signal?: AbortSignal; timeoutMs?: number }) => {
      kernelOpts.push(opts);
      kernelCalls.push(body);
      if (routeDown) throw new Error("worker down");
      if (body.op === "arguments") {
        if (argsDown) throw new Error("needle down");
        return argsReply ?? { tool: body.tool, arguments: { q: "x" }, confidence: 1, reasoning: "t" };
      }
      if (body.op === "entry_tool") {
        executionOrder.push(String(body.tool));
        if (toolThrows !== null) throw toolThrows;
        const reply = entryReplies.shift();
        if (reply) return reply;
        if (toolFailure) return { sessionId: "rs:entry", evidence: [], ...toolFailure };
        return {
          sessionId: "rs:entry",
          content: bridgeEvidence.content,
          evidence: [{ ...bridgeEvidence, id: bridgeIds.shift() ?? bridgeEvidence.id, source: body.tool }],
        };
      }
      if (body.op === "assess_entry") {
        if (assessDown) throw new Error("assessment down");
        if (assessReply !== null) return assessReply;
        const verdict = verdicts.length > 1 ? verdicts.shift()! : verdicts[0];
        return { verdict };
      }
      if (typeof body.op === "string" && body.op.startsWith("assess_final")) {
        gateBodies.push(body as Record<string, unknown>);
        if (body.sessionId !== "rs:entry") return { error: "proof: no admitted research session" };
        return gateReply;
      }
      return routeReply ?? { route: winner, personas };
    },
  },
  runKernelAgent: async (...args: unknown[]) => {
    runCalls.push(args);
    executionOrder.push("research");
    if (emitChartUpdate) {
      const emit = args[1] as (e: Record<string, unknown>) => void;
      emit({ type: "chart_update", ticker: "MSFT" });
    }
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
    if (emitOffer) {
      const emit = args[1] as (e: Record<string, unknown>) => void;
      emit({ type: "offer_trio", sessionId: "rs:test" });
    }
    if (kernelDone !== null) {
      const emit = args[1] as (e: Record<string, unknown>) => void;
      emit({ type: "done", metrics: kernelDone });
    }
  },
  runPersonaAgent: async (...args: unknown[]) => {
    personaCalls.push(args);
    const emit = args[2] as (e: Record<string, unknown>) => void;
    emit({ type: "agent_start", prompt: "q" });
    emit({ type: "answer_delta", text: "trio" });
    emit({ type: "done", metrics: { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 3, totalMs: 1 }, evidence: { count: 1, characters: 1 }, failures: {}, timings: { personasMs: 40, gatesMs: 5 } } });
  },
}));

mock.module("@/lib/muse/client", () => ({
  reason: async (opts: ReasonCall) => {
    reasonCalls.push(opts);
    executionOrder.push("reason");
    if (reasonDown) throw new Error("muse down");
    opts.onDelta?.(reasonText);
    return { text: reasonText, usage: {}, claims: reasonClaims };
  },
}));

mock.module("@/lib/tools/stockbot", () => ({
  invoke: async (name: string, args: unknown, sessionId: string, timeoutMs?: number) => {
    invoked.push([name, args, sessionId]);
    invokeTimeouts.push(timeoutMs);
    executionOrder.push(name);
    if (toolThrows !== null) throw toolThrows;
    if (toolFailure !== null) return { ok: false, ...toolFailure };
    if (name === "set_chart_symbol") return { ok: true, chartSelection: chartReceipt, evidence: { ...bridgeEvidence, content: "ticker: MSFT\nresult_type: chart_selection" } };
    if (name === "update_watchlist") return { ok: true, watchlistUpdate: watchlistReceipt, evidence: { ...bridgeEvidence, content: "action: add\nticker: AMD" } };
    if (name === "add_widget") return { ok: true, widgetSelection: widgetReceipt, evidence: { ...bridgeEvidence, content: "widget_type: short-volume\nticker: AAPL" } };
    return { ok: true, evidence: { ...bridgeEvidence, id: bridgeIds.shift() ?? bridgeEvidence.id, source: name } };
  },
  newSessionId: () => {
    sessionsCreated += 1;
    return "sess-test";
  },
  endSession: async (id: string) => {
    ended.push(id);
    if (endThrows) throw new Error("session cleanup failed");
  },
}));

function reset(): void {
  winner = "reasoning_required";
  routeDown = false;
  argsDown = false;
  reasonDown = false;
  emitProgress = false;
  emitEvidence = false;
  kernelDone = null;
  verdicts = ["node_resolved"];
  personas = null;
  routeReply = null;
  argsReply = null;
  runThrows = null;
  assessDown = false;
  assessReply = null;
  gateReply = { support_answer: "yes" };
  gateBodies.length = 0;
  reasonText = "The answer is unknown.";
  reasonClaims = [{ text: reasonText, claimType: "unknown", evidenceIds: [] }];
  toolFailure = null;
  toolThrows = null;
  endThrows = false;
  bridgeIds = [];
  entryReplies = [];
  chartReceipt = { result_type: "chart_selection", ticker: "MSFT" };
  watchlistReceipt = { result_type: "watchlist_update", action: "add", ticker: "AMD" };
  widgetReceipt = { result_type: "widget_selection", widget_type: "short-volume", ticker: "AAPL" };
  emitChartUpdate = false;
  emitOffer = false;
  personaCalls.length = 0;
  executionOrder.length = 0;
  kernelOpts.length = 0;
  invokeTimeouts.length = 0;
  reasonCalls.length = 0;
  kernelCalls.length = 0;
  runCalls.length = 0;
  invoked.length = 0;
  ended.length = 0;
  sessionsCreated = 0;
  process.env.STOCKBOT_DEBUG = "1";
  process.env.OPENCODE_MODEL = "test-model";
}

function entryCalls(): KernelCall[] {
  return kernelCalls.filter((call) => call.op === "entry_tool");
}

type AgentEventShape = { type: string;[k: string]: unknown };

function expectVerificationRefusal(events: AgentEventShape[]): void {
  expect(events.some((event) => event.type === "failed")).toBe(true);
  const answers = events.filter((event) => event.type === "answer_delta");
  expect(answers).toEqual([{ type: "answer_delta", text: "I could not verify an answer from this source. I withheld its factual claims." }]);
  expect(String(answers[0]?.text)).not.toMatch(/\d/);
  expect(events.at(-1)?.type).toBe("done");
  expect(events.some((event) => event.type === "claims")).toBe(false);
  expect(runCalls).toEqual([]);
}

async function eventsFor(prompt: string, ticker?: unknown, watchlist?: unknown, currentChartTicker?: unknown): Promise<AgentEventShape[]> {
  const { POST } = await import("./route");
  const req = new Request("http://localhost/api/agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, ...(ticker === undefined ? {} : { ticker }), ...(watchlist === undefined ? {} : { watchlist }), ...(currentChartTicker === undefined ? {} : { currentChartTicker }) }),
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

async function personasEvents(body: Record<string, unknown>): Promise<{ status: number; events: AgentEventShape[] }> {
  const { POST } = await import("./route");
  const res = await POST(new Request("http://localhost/api/agent", { method: "POST", body: JSON.stringify({ op: "personas", ...body }) }));
  if (res.status !== 200) return { status: res.status, events: [] };
  const text = await res.text();
  return { status: res.status, events: text.split("\n\n").filter((c) => c.trim()).map((c) => JSON.parse(c.replace(/^data: /, "")) as AgentEventShape) };
}

describe("agent entry route", () => {
  test("clarify_subject streams a fixed answer without reasoning, tools, research, or sessions", async () => {
    for (const [debug, ticker] of [["1", undefined], ["", "NVDA"]] as const) {
      reset();
      process.env.STOCKBOT_DEBUG = debug;
      delete process.env.OPENCODE_MODEL;
      winner = "clarify_subject";
      const prompt = "can you give me buying advice";
      // Calls: request start, route start, route end, clarify start, first text, done.
      const now = spyOn(performance, "now").mockReturnValueOnce(80).mockReturnValueOnce(80).mockReturnValueOnce(90).mockReturnValueOnce(100).mockReturnValue(125);
      try {
        const events = await eventsFor(prompt, ticker);
        expect(events).toEqual([
          { type: "agent_start", prompt },
          { type: "answer_delta", text: "Which stock, ETF, sector, or holding do you want advice on? Give a ticker (for example NVDA) or a company name, and I will research it." },
          {
            type: "done",
            metrics: {
              totalMs: 25,
              needle: { calls: 0, totalMs: 0, escalations: 0 },
              tools: { calls: 0, totalMs: 0 },
              muse: { calls: 0, totalMs: 0 },
              evidence: { count: 0, characters: 0 },
              failures: {},
              timings: { routeMs: 10, firstTextMs: 45 },
            },
          },
        ]);
        expect(kernelCalls).toEqual([{ op: "route", prompt, ...(ticker ? { ticker, currentChartTicker: ticker } : {}) }]);
        expect(reasonCalls).toEqual([]);
        expect(runCalls).toEqual([]);
        expect(invoked).toEqual([]);
        expect(sessionsCreated).toBe(0);
        expect(ended).toEqual([]);
      } finally {
        now.mockRestore();
      }
    }
  });

  test("research done keeps kernel timings and adds the route and first-text timings", async () => {
    reset();
    winner = "research_required";
    emitProgress = true;
    emitEvidence = true;
    const metrics = { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 1, totalMs: 1 }, evidence: { count: 1, characters: 1 }, failures: {} };
    kernelDone = { ...metrics, timings: { intakeMs: 120, nodesMs: 300 } };
    const now = spyOn(performance, "now").mockReturnValueOnce(1000).mockReturnValueOnce(1000).mockReturnValueOnce(1015).mockReturnValue(1250);
    try {
      const events = await eventsFor("What drove NVDA revenue?");
      expect(events.map((e) => e.type)).toEqual(["progress", "evidence", "answer_delta", "done"]);
      // A phase the worker never reported (reasonerMs, admitMs, ...) stays absent.
      expect(events.at(-1)?.metrics).toEqual({ ...metrics, timings: { intakeMs: 120, nodesMs: 300, routeMs: 15, firstTextMs: 250 } });
    } finally {
      now.mockRestore();
    }
  });

  test("clarify_subject with explicit personas preserves research dispatch", async () => {
    reset();
    winner = "clarify_subject";
    personas = ["stockbot"];
    emitProgress = true;
    const prompt = "Stockbot, what should I buy?";
    const events = await eventsFor(prompt);
    expect(events).toEqual([{ type: "progress", stage: "intake_done", detail: { calls: 3 } }]);
    expect(runCalls).toHaveLength(1);
    expect(runCalls[0]?.[0]).toBe(prompt);
    expect(runCalls[0]?.[2]).toMatchObject({ personas: ["stockbot"] });
    expect(kernelCalls).toEqual([{ op: "route", prompt }]);
    expect(reasonCalls).toEqual([]);
    expect(invoked).toEqual([]);
    expect(sessionsCreated).toBe(0);
    expect(ended).toEqual([]);
  });

  test("reasoning_required answers direct without tools or research", async () => {
    reset();
    const types = (await eventsFor("hello")).map((e) => e.type);
    expect(types).toEqual(["agent_start", "status", "reasoning_start", "answer_delta", "done"]);
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
      reasonText = "Assume the discount rate is 10%.";
      reasonClaims = [{ text: reasonText, claimType: "assumption", evidenceIds: [] }];
      const types = (await eventsFor("hello")).map((e) => e.type);
      expect(types).toEqual(["agent_start", "status", "reasoning_start", "answer_delta", "done"]);
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
      expect(types).toEqual(["agent_start", "status", "reasoning_start", "answer_delta", "done"]);
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
    expect(types).toEqual(["agent_start", "status", "needle_decision", "tool_start", "status", "tool_result", "status", "evidence", "reasoning_start", "answer_delta", "claims", "done"]);
    // Readable progress precedes the tool result: the route line first, then the tool phrase, then the count.
    expect(events.flatMap((e) => (e.type === "status" ? [e.text] : []))).toEqual(["Looking up the current time…", "Checking the current time", "Found 1 source"]);
    expect(events.filter((e) => e.type === "tool_start").length).toBe(1);
    expect(runCalls.length).toBe(0);
    expect(kernelCalls.filter((c) => c.op === "arguments").map((c) => c.tool)).toEqual(["get_current_time"]);
    expect(reasonCalls[0].evidence.length).toBe(1);
    expect(entryCalls()).toEqual([{ op: "entry_tool", prompt: "what time is it?", tool: "get_current_time", arguments: { q: "x" } }]);
    expect(invoked).toEqual([]);
    expect(sessionsCreated).toBe(0);
    expect(ended).toEqual([]);
    expect(metricCalls(events, "tools")).toBe(1);
  });

  test("a single-tool answer is buffered, gated with its session, and never shows an unproven number", async () => {
    reset();
    winner = "query_finra";
    const fact = { text: "NVDA short interest for 2026-10-15 was 12.5% [ev:bridge].", claimType: "observed_fact", evidenceIds: ["ev:bridge"], datapointIds: ["dp:short-interest"] };
    const datapoint = { id: "dp:short-interest", metric: "shortInterest", value: "12.5", entity: "NVDA", period: "2026-10-15", unit: "%", statement: fact.text };
    reasonText = `${fact.text} Volume hit 99 million. More filings may follow.`;
    reasonClaims = [fact];
    entryReplies = [{ sessionId: "rs:entry", evidence: [{ ...bridgeEvidence, source: "FINRA", content: fact.text, datapoints: [datapoint] }], content: fact.text }];
    const events = await eventsFor("short interest?");
    // The streamed reasoner delta never reaches the client: one gated answer only.
    expect(events.filter((e) => e.type === "answer_delta")).toEqual([{ type: "answer_delta", text: fact.text }]);
    expect(events.slice(-3).map((e) => e.type)).toEqual(["answer_delta", "claims", "done"]);
    expect(events.find((e) => e.type === "claims")?.claims).toEqual([fact]);
    expect(gateBodies).toHaveLength(1);
    expect(gateBodies[0]).toMatchObject({
      op: "assess_final_response",
      sessionId: "rs:entry",
      claims: [{ text: fact.text, claim_type: "observed_fact", evidence_ids: ["ev:bridge"], datapoint_ids: ["dp:short-interest"] }],
    });
    expect(runCalls).toEqual([]);
  });

  test("a rejected single-tool number is withheld with an explicit refusal, never retried in research", async () => {
    reset();
    winner = "query_finra";
    reasonText = "NVDA short interest for 2026-10-15 was 12.5% [ev:bridge].";
    reasonClaims = [{ text: reasonText, claimType: "observed_fact", evidenceIds: ["ev:bridge"], datapointIds: ["dp:short-interest"] }];
    const datapoint = { id: "dp:short-interest", metric: "shortInterest", value: "12.5", entity: "NVDA", period: "2026-10-15", unit: "%", statement: reasonText };
    entryReplies = [{ sessionId: "rs:entry", evidence: [{ ...bridgeEvidence, datapoints: [datapoint] }], content: reasonText }];
    gateReply = { error: "proof: datapoint does not match the archived row" };
    const events = await eventsFor("short interest?");
    expectVerificationRefusal(events);
    expect(gateBodies.map((g) => g.op)).toEqual(["assess_final_response", "assess_final_claim"]);
    expect(gateBodies.map((g) => g.sessionId)).toEqual(["rs:entry", "rs:entry"]);
    expect(ended).toEqual([]);
  });

  test("clock and OTHER results without ledger admission refuse before reasoning or gates", async () => {
    for (const tool of ["get_current_time", "get_fundamentals"]) {
      reset();
      winner = tool;
      entryReplies = [{ sessionId: "rs:entry", evidence: [], content: "The unadmitted result contains 42." }];
      const events = await eventsFor("look it up");
      expectVerificationRefusal(events);
      expect(entryCalls()).toEqual([{ op: "entry_tool", prompt: "look it up", tool, arguments: { q: "x" } }]);
      expect(kernelCalls.some((call) => call.op === "assess_entry")).toBe(false);
      expect(reasonCalls).toEqual([]);
      expect(gateBodies).toEqual([]);
      expect(invoked).toEqual([]);
      expect(sessionsCreated).toBe(0);
      expect(ended).toEqual([]);
    }
  });

  test("a missing or changed entry session cannot gate another session's evidence", async () => {
    for (const sessionId of [undefined, "rs:foreign"]) {
      reset();
      winner = "search_sec_filings";
      verdicts = ["get_sec_document", "node_resolved"];
      entryReplies = [
        { sessionId: "rs:entry", evidence: [bridgeEvidence], content: bridgeEvidence.content },
        { ...(sessionId ? { sessionId } : {}), evidence: [{ ...bridgeEvidence, id: "ev:foreign", content: "Revenue was 99." }], content: "Revenue was 99." },
      ];
      reasonText = "Revenue was 99.";
      const events = await eventsFor("read the filing");
      expectVerificationRefusal(events);
      expect(entryCalls().map((call) => call.sessionId)).toEqual([undefined, "rs:entry"]);
      expect(reasonCalls).toEqual([]);
      expect(gateBodies).toEqual([]);
      expect(events.filter((event) => event.type === "evidence").flatMap((event) => event.evidence as Evidence[]).map((row) => row.id)).toEqual(["ev:bridge"]);
      expect(ended).toEqual([]);
    }
  });

  test("a direct answer is buffered and keeps only qualified evidence-free inventory; nothing left researches", async () => {
    reset();
    const unknown = { text: "The source is unavailable.", claimType: "unknown", evidenceIds: [] };
    reasonText = `NVDA rose 12% today. ${unknown.text}`;
    reasonClaims = [unknown];
    const events = await eventsFor("hello");
    expect(events.filter((e) => e.type === "answer_delta")).toEqual([{ type: "answer_delta", text: unknown.text }]);
    expect(gateBodies).toEqual([]);
    expect(runCalls).toEqual([]);
    reset();
    reasonText = "NVDA rose 12% today.";
    reasonClaims = [];
    const empty = await eventsFor("hello");
    expect(empty.some((e) => e.type === "answer_delta")).toBe(false);
    expect(runCalls).toHaveLength(1);
  });

  test("unqualified direct prose falls back to research without publishing an answer", async () => {
    for (const route of ["reasoning_required", "no_session"]) {
      for (const claimType of [undefined, "unknown", "assumption"]) {
        reset();
        winner = route;
        reasonText = "NVDA dominates its market.";
        reasonClaims = claimType ? [{ text: reasonText, claimType, evidenceIds: [] }] : [];
        const events = await eventsFor("NVDA outlook?");
        expect(events.map((event) => event.type)).toEqual(["agent_start", "status", "reasoning_start"]);
        expect(reasonCalls).toHaveLength(1);
        expect(reasonCalls[0].evidence).toEqual([]);
        expect(gateBodies).toEqual([]);
        expect(runCalls).toHaveLength(1);
        expect(runCalls[0]?.[0]).toBe("NVDA outlook?");
      }
    }
  });

  test("status lines name the tool subject in prod, bound untrusted queries, and open the direct route", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    const direct = await eventsFor("hello");
    expect(direct[1]).toEqual({ type: "status", text: "Thinking…" });
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "get_insider_activity";
    argsReply = { tool: "get_insider_activity", arguments: { ticker: "nvda" }, confidence: 1 };
    const insider = await eventsFor("NVDA insider sales?");
    expect(insider.flatMap((e) => (e.type === "status" ? [e.text] : []))).toEqual([
      "Looking up Form 4 insider filings…", "Reading Form 4 insider filings for NVDA", "Found 1 source",
    ]);
    reset();
    winner = "search_web";
    argsReply = { tool: "search_web", arguments: { query: `chip\n  demand ${"x".repeat(200)}` }, confidence: 1 };
    const web = (await eventsFor("chip news?")).flatMap((e) => (e.type === "status" ? [String(e.text)] : []));
    // The model-chosen query is one collapsed line, cut to 120 characters with an ellipsis.
    expect(web[1]).toBe(`Searching the web: chip demand ${"x".repeat(88)}…`);
    expect(web[1]).toHaveLength(120);
  });

  test("a tool missing from the TS registry executes through the ledger worker", async () => {
    reset();
    winner = "query_finra";
    const events = await eventsFor("short interest?");
    expect(events.filter((e) => e.type === "tool_start").length).toBe(1);
    expect(entryCalls()).toHaveLength(1);
    expect(entryCalls()[0]?.tool).toBe("query_finra");
    expect(invoked).toEqual([]);
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
    expect(entryCalls()).toHaveLength(1);
    expect(reasonCalls.length).toBe(0);
    expect(runCalls.length).toBe(1);
  });

  test("chained tool verdict runs the next tool then resolves", async () => {
    reset();
    winner = "search_sec_filings";
    verdicts = ["get_sec_document", "node_resolved"];
    bridgeIds = ["ev:search", "ev:document"];
    const events = await eventsFor("NVDA risk factors?");
    expect(events.filter((e) => e.type === "tool_start").length).toBe(2);
    expect(entryCalls().map((c) => c.tool)).toEqual(["search_sec_filings", "get_sec_document"]);
    expect(entryCalls().map((c) => c.sessionId)).toEqual([undefined, "rs:entry"]);
    expect(gateBodies.map((g) => g.sessionId)).toEqual(["rs:entry"]);
    expect(ended).toEqual([]);
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
    expect(events.map((e) => e.type)).toEqual(["agent_start", "status", "reasoning_start", "failed", "error"]);
    expect(runCalls.length).toBe(0);
  });

  test("prod mode summarizes: generic working + single error", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    reasonDown = true;
    const events = await eventsFor("hello");
    expect(events.map((e) => e.type)).toEqual(["agent_start", "status", "reasoning_start", "error"]);
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
    expect(prod.map((e) => e.type)).toEqual(["agent_start", "status", "status", "status", "evidence", "reasoning_start", "answer_delta", "claims", "done"]);
    reset();
    winner = "get_current_time";
    const debug = await eventsFor("what time is it?");
    expect(debug.map((e) => e.type)).toEqual([
      "agent_start",
      "status",
      "needle_decision",
      "tool_start",
      "status",
      "tool_result",
      "status",
      "evidence",
      "reasoning_start",
      "answer_delta",
      "claims",
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
      expect(ops).toEqual(["route", "arguments", "entry_tool", "assess_entry"]);
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
    expect(entryCalls()).toHaveLength(1);
    expect(entryCalls()[0]?.arguments).toEqual({ q: "x" });
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
    expect(kernelCalls.map((c) => c.op)).toEqual(["route", "arguments", "entry_tool", "assess_entry", "assess_final_response"]);
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

  const feedContext = { kind: "news", title: "NVDA jumps overnight", url: "https://news.example/a?x=1", source: "Stocktwits", sym: "NVDA", at: "2026-10-20T09:00:00Z" };
  async function postContext(context: unknown): Promise<Response> {
    const { POST } = await import("./route");
    return POST(new Request("http://localhost/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "What does this mean for NVDA?", ticker: "NVDA", context }),
    }));
  }

  test("a valid feed context reaches the routing call, the kernel, and the reasoner as data beside the verbatim prompt", async () => {
    reset();
    winner = "research_required";
    await (await postContext({ ...feedContext, sym: "nvda" })).text();
    expect(kernelCalls.map((c) => c.op)).toEqual(["route"]);
    expect(kernelCalls[0]?.prompt).toBe("What does this mean for NVDA?");
    expect(kernelCalls[0]?.context).toEqual(feedContext);
    expect(runCalls[0]?.[0]).toBe("What does this mean for NVDA?");
    const runOpts = runCalls[0]?.[2] as Record<string, unknown>;
    expect(runOpts.context).toEqual(feedContext);

    reset();
    const { url: _url, sym: _sym, ...bare } = feedContext;
    await (await postContext(bare)).text();
    expect(reasonCalls[0]?.prompt).toBe("What does this mean for NVDA?");
    expect(reasonCalls[0]?.context).toEqual(bare);
  });

  test("each invalid context field is rejected with 400 before any worker call", async () => {
    for (const bad of [
      "news", [], { ...feedContext, extra: 1 },
      { ...feedContext, kind: "tweet" }, { ...feedContext, kind: undefined },
      { ...feedContext, title: "" }, { ...feedContext, title: "  " }, { ...feedContext, title: "x".repeat(301) }, { ...feedContext, title: 7 },
      { ...feedContext, source: "" }, { ...feedContext, source: "x".repeat(301) },
      { ...feedContext, url: "javascript:alert(1)" }, { ...feedContext, url: "ftp://news.example/a" }, { ...feedContext, url: "https://user:pw@news.example/a" },
      { ...feedContext, url: "not a url" }, { ...feedContext, url: `https://news.example/${"a".repeat(2000)}` }, { ...feedContext, url: 7 },
      { ...feedContext, sym: "SEC" }, { ...feedContext, sym: "nvda!" }, { ...feedContext, sym: 7 },
      { ...feedContext, at: 7 }, { ...feedContext, at: "x".repeat(41) },
    ]) {
      reset();
      const res = await postContext(bad);
      expect(res.status).toBe(400);
      expect(await res.json()).toEqual({ error: "context must be a feed item: kind news|filing|insider|alert, title and source (1-300 chars), at (max 40 chars), optional http(s) url and ticker sym" });
      expect(kernelCalls.length).toBe(0);
      expect(runCalls.length).toBe(0);
      expect(reasonCalls.length).toBe(0);
    }
  });

  test("a request without context sends none", async () => {
    for (const context of [undefined, null]) {
      reset();
      winner = "research_required";
      await (await postContext(context)).text();
      expect("context" in (kernelCalls[0] ?? {})).toBe(false);
      const runOpts = runCalls[0]?.[2] as Record<string, unknown>;
      expect("context" in runOpts).toBe(false);
      reset();
      await (await postContext(context)).text();
      expect(reasonCalls[0] && "context" in reasonCalls[0]).toBe(false);
    }
  });

  test("invalid JSON and missing prompts stop before routing", async () => {
    const { POST } = await import("./route");
    for (const [body, error] of [
      ["{", "invalid JSON"],
      ["null", "invalid JSON"],
      ["{}", "prompt required"],
      ['{"prompt":"   "}', "prompt required"],
      ['{"prompt":7}', "prompt required"],
    ]) {
      reset();
      const response = await POST(new Request("http://localhost/api/agent", { method: "POST", body }));
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error });
      expect(kernelCalls).toEqual([]);
    }
  });

  test("worker failures refuse before assessment without unverified evidence", async () => {
    reset();
    winner = "query_finra";
    const message = "provider unavailable ".repeat(50);
    toolFailure = { category: "provider_error", error: message };
    const events = await eventsFor("short interest?");
    expectVerificationRefusal(events);
    expect(events.find((event) => event.type === "tool_failed")).toMatchObject({ category: "tool_error", preview: message.slice(0, 160) });
    expect(kernelCalls.some((call) => call.op === "assess_entry")).toBe(false);
    expect(gateBodies).toEqual([]);
    expect(events.at(-1)).toMatchObject({ metrics: { evidence: { count: 0, characters: 0 } } });
    expect(metricCalls(events, "tools")).toBe(1);
    expect(ended).toEqual([]);
    expect(reasonCalls).toEqual([]);
  });

  test("thrown worker failures refuse without creating bridge sessions or completed tool calls", async () => {
    for (const thrown of [new Error("tool exploded"), "tool exploded"]) {
      reset();
      winner = "query_finra";
      toolThrows = thrown;
      const events = await eventsFor("short interest?");
      expectVerificationRefusal(events);
      expect(events.find((event) => event.type === "tool_failed")).toEqual({
        type: "tool_failed", tool: "query_finra", category: "tool_error", preview: "tool exploded",
      });
      expect(events.at(-1)).toMatchObject({ metrics: { failures: { tool_error: 2 } } });
      expect(metricCalls(events, "tools")).toBe(0);
      expect(ended).toEqual([]);
    }
  });

  test("unavailable assessment retains evidence for research", async () => {
    reset();
    winner = "query_finra";
    assessDown = true;
    const events = await eventsFor("short interest?");
    expect(events.map((event) => event.type)).toEqual(["agent_start", "status", "needle_decision", "tool_start", "status", "tool_result", "status", "evidence"]);
    expect(runCalls[0]?.[2]).toMatchObject({ seedEvidence: [expect.objectContaining({ id: "ev:bridge" })] });
    expect(reasonCalls).toEqual([]);
    expect(ended).toEqual([]);
  });

  test("invalid assessment verdicts defer to research", async () => {
    for (const verdict of [undefined, "", 42, "invalid tool"]) {
      reset();
      winner = "query_finra";
      assessReply = { verdict };
      await eventsFor("short interest?");
      expect(entryCalls()).toHaveLength(1);
      expect(ended).toEqual([]);
      expect(runCalls).toHaveLength(1);
      expect(runCalls[0]?.[2]).toMatchObject({ seedEvidence: expect.arrayContaining([expect.objectContaining({ id: "ev:bridge" })]) });
      expect(reasonCalls).toEqual([]);
    }
  });

  test("three-tool cap with admitted evidence answers without research, in one session", async () => {
    reset();
    winner = "query_finra";
    assessReply = { verdict: "query_finra" };
    bridgeIds = ["ev:1", "ev:2", "ev:3"];
    const events = await eventsFor("short interest?");
    expect(entryCalls()).toHaveLength(3);
    // Worker-created research session survives each call and the gate; the route never closes it.
    expect(entryCalls().map((call) => call.sessionId)).toEqual([undefined, "rs:entry", "rs:entry"]);
    expect(sessionsCreated).toBe(0);
    expect(ended).toEqual([]);
    expect(gateBodies.map((g) => g.sessionId)).toEqual(["rs:entry"]);
    expect(events.filter((e) => e.type === "agent_start")).toHaveLength(1);
    expect(runCalls).toEqual([]);
    expect(reasonCalls).toHaveLength(1);
    expect(reasonCalls[0].evidence).toHaveLength(3);
    expect(events.slice(-4).map((e) => e.type)).toEqual(["reasoning_start", "answer_delta", "claims", "done"]);
    expect(metricCalls(events, "tools")).toBe(3);
  });

  test("a repeated identical tool call keeps one evidence item", async () => {
    reset();
    winner = "list_sec_filings";
    verdicts = ["list_sec_filings", "node_resolved"];
    const events = await eventsFor("NVDA filings?");
    expect(entryCalls().map((c) => c.tool)).toEqual(["list_sec_filings", "list_sec_filings"]);
    expect(reasonCalls[0].evidence.map((e) => e.id)).toEqual(["ev:bridge"]);
    expect(events.filter((e) => e.type === "evidence").at(-1)?.evidence).toHaveLength(1);
    expect(events.at(-1)).toMatchObject({ metrics: { tools: { calls: 2 }, evidence: { count: 1 } } });
    expect(runCalls).toEqual([]);
  });

  test("no-data worker errors cannot become evidence or factual prose", async () => {
    reset();
    winner = "get_xbrl_facts";
    toolFailure = { category: "no_records", error: "No data found for MSFT: no XBRL facts found for concept 'Assets'" };
    const events = await eventsFor("MSFT total assets?");
    expectVerificationRefusal(events);
    expect(reasonCalls).toEqual([]);
  });

  test("a failed source ends the chain before the cap without a research retry", async () => {
    reset();
    winner = "query_finra";
    assessReply = { verdict: "query_finra" };
    toolFailure = { category: "tool_error", error: "bad args" };
    const events = await eventsFor("short interest?");
    expect(entryCalls()).toHaveLength(1);
    expect(kernelCalls.some((call) => call.op === "assess_entry")).toBe(false);
    expect(gateBodies).toEqual([]);
    expectVerificationRefusal(events);
    expect(reasonCalls).toEqual([]);
  });

  test("a gated answer never creates or closes a bridge-only session", async () => {
    reset();
    winner = "query_finra";
    endThrows = true;
    const events = await eventsFor("short interest?");
    expect(kernelCalls.map((call) => call.op)).toEqual(["route", "arguments", "entry_tool", "assess_entry", "assess_final_response"]);
    expect(ended).toEqual([]);
    expect(sessionsCreated).toBe(0);
    expect(runCalls).toEqual([]);
    expect(events.slice(-3).map((e) => e.type)).toEqual(["answer_delta", "claims", "done"]);
  });

  test("single-shot reasoning failure emits the original error without research", async () => {
    reset();
    winner = "query_finra";
    reasonDown = true;
    const events = await eventsFor("short interest?");
    expect(events.at(-1)).toEqual({ type: "error", message: "muse down" });
    expect(events.some((event) => event.type === "done")).toBe(false);
    expect(runCalls).toEqual([]);
  });

  test("missing reasoning configuration defers the direct answer to research", async () => {
    reset();
    delete process.env.OPENCODE_MODEL;
    const events = await eventsFor("hello");
    expect(events.map((event) => event.type)).toEqual(["agent_start", "status"]);
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toHaveLength(1);
  });

  test("an already aborted request produces no stream events", async () => {
    reset();
    const { POST } = await import("./route");
    const controller = new AbortController();
    controller.abort();
    const response = await POST(new Request("http://localhost/api/agent", {
      method: "POST", body: JSON.stringify({ prompt: "hello" }), signal: controller.signal,
    }));
    expect(await response.text()).toBe("");
  });

  test("the selected focus updates the chart before a direct answer without factual evidence or another route", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    routeReply = { route: "reasoning_required", personas: null, chartTicker: "MSFT" };
    const prompt = "Compare Apple and Microsoft";
    const events = await eventsFor(prompt, "AAPL");
    expect(executionOrder).toEqual(["set_chart_symbol", "reason"]);
    expect(events[0]).toEqual({ type: "chart_update", ticker: "MSFT" });
    expect(events.findIndex((event) => event.type === "chart_update")).toBeLessThan(events.findIndex((event) => event.type === "reasoning_start"));
    expect(kernelCalls).toEqual([{ op: "route", prompt, ticker: "AAPL", currentChartTicker: "AAPL" }]);
    expect(invoked).toEqual([["set_chart_symbol", { ticker: "MSFT" }, "sess-test"]]);
    expect(reasonCalls[0]).toMatchObject({ prompt, ticker: "MSFT", evidence: [] });
    expect(ended).toEqual(["sess-test"]);
    expect(events.some((event) => event.type === "evidence")).toBe(false);
    expect(runCalls).toEqual([]);
  });

  test("research and every explicit persona selection apply the chart pre-action first", async () => {
    for (const selection of [null, ...Array.from({ length: (1 << FINAL_PERSONAS.length) - 1 }, (_, mask) => FINAL_PERSONAS.filter((_, i) => ((mask + 1) >> i) & 1))]) {
      reset();
      routeReply = { route: selection === null ? "research_required" : "reasoning_required", personas: selection, chartTicker: "MSFT" };
      const prompt = "Compare Apple and Microsoft";
      const events = await eventsFor(prompt, "AAPL");
      expect(executionOrder).toEqual(["set_chart_symbol", "research"]);
      expect(events).toEqual([{ type: "chart_update", ticker: "MSFT" }]);
      expect(runCalls[0]?.[0]).toBe(prompt);
      expect(runCalls[0]?.[2]).toMatchObject({ ticker: "MSFT", currentChartTicker: "MSFT", seedEvidence: [] });
      if (selection !== null) expect(runCalls[0]?.[2]).toMatchObject({ personas: selection });
      expect(kernelCalls.map((call) => call.op)).toEqual(["route"]);
      expect(ended).toEqual(["sess-test"]);
    }
  });

  test("single-tool reasoning uses the applied chart without treating its receipt as evidence", async () => {
    reset();
    routeReply = { route: "get_sec_document", personas: null, chartTicker: "MSFT" };
    await eventsFor("Microsoft risk factors?", "AAPL");
    expect(executionOrder).toEqual(["set_chart_symbol", "get_sec_document", "reason"]);
    expect(reasonCalls[0]).toMatchObject({ ticker: "MSFT", evidence: [expect.objectContaining({ id: "ev:bridge", content: bridgeEvidence.content })] });
    expect(kernelCalls.filter((call) => call.op === "arguments").map((call) => [call.tool, call.ticker])).toEqual([["get_sec_document", "MSFT"]]);
    expect(kernelCalls.some((call) => call.op === "arguments" && call.tool === "set_chart_symbol")).toBe(false);
    expect(kernelCalls.filter((call) => call.op === "assess_entry")).toHaveLength(1);
    expect(ended).toEqual(["sess-test"]);
  });

  test("a single-tool research fallback never seeds the chart receipt", async () => {
    reset();
    routeReply = { route: "get_sec_document", personas: null, chartTicker: "MSFT" };
    verdicts = ["research_required"];
    await eventsFor("Microsoft risk factors?", "AAPL");
    expect(runCalls[0]?.[2]).toMatchObject({ ticker: "MSFT", currentChartTicker: "MSFT", seedEvidence: [expect.objectContaining({ content: bridgeEvidence.content })] });
    expect(JSON.stringify(runCalls[0]?.[2])).not.toContain("chart_selection");
  });

  test("chart metadata errors preserve the original route and explicit personas", async () => {
    for (const chart of [
      { chartTicker: null },
      { chartTicker: "SEC" },
      { chartTicker: "MSFT", chartError: "ambiguous" },
      { chartError: 7 },
      { chartError: "" },
      { chartError: "No SEC entity matched the selected focus." },
    ]) {
      for (const selection of [null, ["stockbot"]]) {
        reset();
        routeReply = { route: "reasoning_required", personas: selection, ...chart };
        const events = await eventsFor("Microsoft outlook?", "AAPL");
        expect(events[0]?.type).toBe("answer_delta");
        expect(String(events[0]?.text)).toContain("Chart update failed:");
        expect(events.some((event) => event.type === "chart_update")).toBe(false);
        expect(invoked).toEqual([]);
        expect(kernelCalls.map((call) => call.op)).toEqual(["route"]);
        if (selection === null) {
          expect(reasonCalls[0]).toMatchObject({ ticker: "AAPL", evidence: [] });
          expect(events.at(-2)).toEqual({ type: "answer_delta", text: reasonText });
        } else {
          expect(runCalls[0]?.[2]).toMatchObject({ personas: selection, ticker: "AAPL", currentChartTicker: "AAPL" });
        }
      }
    }
  });

  test("malformed chart metadata never authorizes an action", async () => {
    await fc.assert(fc.asyncProperty(
      fc.jsonValue().filter((value) => realParseTicker(value) === undefined),
      async (chartTicker) => {
        reset();
        routeReply = { route: "reasoning_required", personas: null, chartTicker };
        const events = await eventsFor("Microsoft outlook?", "AAPL");
        expect(invoked).toEqual([]);
        expect(reasonCalls[0]?.ticker).toBe("AAPL");
        expect(events.some((event) => event.type === "chart_update")).toBe(false);
        expect(String(events[0]?.text)).toContain("Chart update failed:");
      },
    ), { seed: 2119, numRuns: 30 });
  });

  test("failed tools and malformed receipts warn and continue with the prior chart", async () => {
    for (const failure of ["returned", "thrown", "missing", "string", "array", "kind", "ticker", "lowercase"]) {
      reset();
      delete process.env.STOCKBOT_DEBUG;
      routeReply = { route: "reasoning_required", personas: null, chartTicker: "MSFT" };
      if (failure === "returned") toolFailure = { category: "tool_error", error: "chart tool unavailable" };
      if (failure === "thrown") toolThrows = new Error("chart tool unavailable");
      if (failure === "missing") chartReceipt = undefined;
      if (failure === "string") chartReceipt = JSON.stringify({ result_type: "chart_selection", ticker: "MSFT" });
      if (failure === "array") chartReceipt = [{ result_type: "chart_selection", ticker: "MSFT" }];
      if (failure === "kind") chartReceipt = { result_type: "research", ticker: "MSFT" };
      if (failure === "ticker") chartReceipt = { result_type: "chart_selection", ticker: "AAPL" };
      if (failure === "lowercase") chartReceipt = { result_type: "chart_selection", ticker: "msft" };
      const events = await eventsFor("Microsoft outlook?", "AAPL");
      expect(events.some((event) => event.type === "chart_update")).toBe(false);
      expect(events.some((event) => event.type === "error" || event.type === "failed")).toBe(false);
      expect(String(events[0]?.text)).toContain("Chart update failed:");
      expect(events.at(-2)).toEqual({ type: "answer_delta", text: reasonText });
      expect(reasonCalls[0]).toMatchObject({ ticker: "AAPL", evidence: [] });
      expect(ended).toEqual(["sess-test"]);
      expect(runCalls).toEqual([]);
    }
  });

  test("chart failure still researches for explicit personas without receipt evidence", async () => {
    reset();
    routeReply = { route: "reasoning_required", personas: ["stockbot", "bearbot", "bullbot"], chartTicker: "MSFT" };
    toolFailure = { category: "tool_error", error: "chart tool unavailable" };
    const events = await eventsFor("Microsoft outlook?", "AAPL");
    expect(events).toEqual([{ type: "answer_delta", text: "Chart update failed: chart tool unavailable I will continue with your original question.\n\n" }]);
    expect(runCalls[0]?.[2]).toMatchObject({ personas: ["stockbot", "bearbot", "bullbot"], ticker: "AAPL", currentChartTicker: "AAPL", seedEvidence: [] });
    expect(ended).toEqual(["sess-test"]);
  });

  test("chart cleanup failure does not undo an applied chart or interrupt the answer", async () => {
    reset();
    routeReply = { route: "reasoning_required", personas: null, chartTicker: "MSFT" };
    endThrows = true;
    const events = await eventsFor("Microsoft outlook?", "AAPL");
    expect(events[0]).toEqual({ type: "chart_update", ticker: "MSFT" });
    expect(reasonCalls[0]?.ticker).toBe("MSFT");
    expect(ended).toEqual(["sess-test"]);
  });

  test("watchlist scope stays exclusive while the displayed chart travels separately", async () => {
    for (const route of ["research_required", "reasoning_required", "get_sec_document"]) {
      reset();
      routeReply = { route, personas: null, chartTicker: "MSFT" };
      const prompt = "Compare Microsoft with my watchlist";
      const events = await eventsFor(prompt, "AAPL", ["NVDA", "MSFT"]);
      expect(kernelCalls[0]).toEqual({ op: "route", prompt, watchlist: ["NVDA", "MSFT"], currentChartTicker: "AAPL" });
      expect(events[0]).toEqual({ type: "chart_update", ticker: "MSFT" });
      if (route === "research_required") {
        expect(runCalls[0]?.[2]).toMatchObject({ watchlist: ["NVDA", "MSFT"], currentChartTicker: "MSFT", seedEvidence: [] });
        expect("ticker" in (runCalls[0]?.[2] as object)).toBe(false);
      } else {
        expect(reasonCalls).toHaveLength(1);
        expect("ticker" in reasonCalls[0]).toBe(false);
      }
      for (const call of kernelCalls) expect("ticker" in call).toBe(false);
    }
  });

  test("a separate displayed-stock channel supplies reasoner context without changing research scope", async () => {
    reset();
    await eventsFor("How is it doing?", "NVDA", undefined, " msft ");
    expect(kernelCalls[0]).toMatchObject({ ticker: "NVDA", currentChartTicker: "MSFT" });
    expect(reasonCalls[0]?.ticker).toBe("MSFT");
    for (const invalid of ["SEC", 7, "NVDA!", null]) {
      reset();
      await eventsFor("Explain valuation", "AAPL", undefined, invalid);
      expect(kernelCalls[0]?.currentChartTicker).toBe("AAPL");
      expect(reasonCalls[0]?.ticker).toBe("AAPL");
    }
  });

  test("production retains dedicated research chart events while it redacts ordinary progress", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "research_required";
    emitChartUpdate = true;
    emitProgress = true;
    const events = await eventsFor("Microsoft outlook?", "AAPL");
    expect(events).toEqual([{ type: "chart_update", ticker: "MSFT" }, { type: "progress", stage: "working" }]);
  });

  test("the chart tool cannot replace the original entry route", async () => {
    reset();
    routeReply = { route: "set_chart_symbol", personas: null, chartTicker: "MSFT" };
    const events = await eventsFor("Microsoft outlook?", "AAPL");
    expect(events.map((event) => event.type)).toEqual(["agent_start", "failed", "error"]);
    expect(invoked).toEqual([]);
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });

  test("post-tool assessment cannot chain into a chart receipt as factual evidence", async () => {
    reset();
    winner = "get_sec_document";
    verdicts = ["set_chart_symbol"];
    await eventsFor("Microsoft risk factors?", "MSFT");
    expect(entryCalls().map((call) => call.tool)).toEqual(["get_sec_document"]);
    expect(invoked).toEqual([]);
    expect(runCalls[0]?.[2]).toMatchObject({ seedEvidence: [expect.objectContaining({ content: bridgeEvidence.content })] });
    expect(reasonCalls).toEqual([]);
  });

  test("chart pre-action shares the existing request deadline and keeps original metrics scope", async () => {
    reset();
    routeReply = { route: "reasoning_required", personas: null, chartTicker: "MSFT" };
    await eventsFor("Microsoft outlook?", "AAPL");
    const chartTimeout = invokeTimeouts[0];
    if (typeof chartTimeout !== "number") throw new Error("expected numeric chart tool timeout");
    expect(chartTimeout).toBeGreaterThan(0);
    expect(chartTimeout).toBeLessThanOrEqual(120_000);
  });

  test("an expired chart deadline preserves research instead of changing the context", async () => {
    reset();
    routeReply = { route: "research_required", personas: null, chartTicker: "MSFT" };
    const realNow = Date.now;
    const baseline = realNow();
    let calls = 0;
    const dateSpy = spyOn(Date, "now").mockImplementation(() => {
      calls += 1;
      return calls === 1 ? baseline : baseline + 180_000;
    });
    try {
      const events = await eventsFor("Microsoft outlook?", "AAPL");
      expect(String(events[0]?.text)).toContain("request deadline expired");
      expect(invoked).toEqual([]);
      expect(kernelOpts).toHaveLength(1);
      expect(runCalls[0]?.[2]).toMatchObject({ ticker: "AAPL", currentChartTicker: "AAPL" });
    } finally {
      dateSpy.mockRestore();
    }
  });

  test("the trio follow-up skips the route round, intake, and tools and runs only the persona stage", async () => {
    reset();
    const { status, events } = await personasEvents({ sessionId: "rs:test", personas: FINAL_PERSONAS });
    expect(status).toBe(200);
    expect(personaCalls).toHaveLength(1);
    expect(personaCalls[0]?.slice(0, 2)).toEqual(["rs:test", [...FINAL_PERSONAS]]);
    expect(kernelCalls).toEqual([]);
    expect(runCalls).toEqual([]);
    expect(reasonCalls).toEqual([]);
    expect(invoked).toEqual([]);
    expect(events.map((e) => e.type)).toEqual(["agent_start", "answer_delta", "done"]);
    // No route round ran, so routeMs stays absent; first text is the request's own.
    const metrics = events.at(-1)?.metrics;
    if (!metrics || typeof metrics !== "object" || !("timings" in metrics) || !metrics.timings || typeof metrics.timings !== "object") throw new Error("expected done timings");
    expect(Object.keys(metrics.timings).sort()).toEqual(["firstTextMs", "gatesMs", "personasMs"]);
  });

  test("a malformed trio follow-up is rejected before any kernel call", async () => {
    for (const body of [
      { personas: FINAL_PERSONAS },
      { sessionId: 7, personas: FINAL_PERSONAS },
      { sessionId: "../rs", personas: FINAL_PERSONAS },
      { sessionId: "rs:test" },
      { sessionId: "rs:test", personas: ["bullbot", "stockbot"] },
      { sessionId: "rs:test", personas: null },
    ]) {
      reset();
      const { status } = await personasEvents(body);
      expect(status).toBe(400);
      expect(personaCalls).toEqual([]);
      expect(kernelCalls).toEqual([]);
    }
  });

  test("production streams the trio offer beside the research answer", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "research_required";
    emitEvidence = true;
    emitOffer = true;
    const events = await eventsFor("filing facts?");
    expect(events.map((e) => e.type)).toEqual(["evidence", "answer_delta", "offer_trio"]);
    expect(events.at(-1)).toEqual({ type: "offer_trio", sessionId: "rs:test" });
  });
});

describe("watchlist chat action", () => {
  test("a validated receipt emits watchlist_update and answers without evidence, reasoning, or research", async () => {
    for (const [action, text] of [["add", "Added AMD to your watchlist."], ["remove", "Removed AMD from your watchlist."]] as const) {
      reset();
      delete process.env.STOCKBOT_DEBUG;
      winner = "update_watchlist";
      watchlistReceipt = { result_type: "watchlist_update", action, ticker: "AMD" };
      argsReply = { tool: "update_watchlist", arguments: { action, ticker: "amd" }, confidence: 1 };
      const prompt = `${action} AMD to my watchlist`;
      const events = await eventsFor(prompt, "NVDA", ["NVDA", "AAPL"]);
      expect(events.slice(0, 4)).toEqual([
        { type: "agent_start", prompt },
        { type: "status", text: "Updating your watchlist…" },
        { type: "watchlist_update", action, ticker: "AMD" },
        { type: "answer_delta", text },
      ]);
      expect(events.at(-1)?.type).toBe("done");
      expect(invoked).toEqual([["update_watchlist", { action, ticker: "amd" }, "sess-test"]]);
      expect(ended).toEqual(["sess-test"]);
      expect(events.some((event) => event.type === "evidence")).toBe(false);
      expect(reasonCalls).toEqual([]);
      expect(runCalls).toEqual([]);
      expect(kernelCalls.map((call) => call.op)).toEqual(["route", "arguments"]);
    }
  });

  test("failed tools, malformed receipts, and failed arguments answer visibly without an update or research", async () => {
    for (const failure of ["returned", "missing", "kind", "action", "ticker", "args"]) {
      reset();
      winner = "update_watchlist";
      if (failure === "returned") toolFailure = { category: "tool_error", error: "tool failed" };
      if (failure === "missing") watchlistReceipt = undefined;
      if (failure === "kind") watchlistReceipt = { result_type: "chart_selection", action: "add", ticker: "AMD" };
      if (failure === "action") watchlistReceipt = { result_type: "watchlist_update", action: "clear", ticker: "AMD" };
      if (failure === "ticker") watchlistReceipt = { result_type: "watchlist_update", action: "add", ticker: "amd" };
      if (failure === "args") argsDown = true;
      const events = await eventsFor("add AMD to my watchlist", "NVDA");
      expect(events.some((event) => event.type === "watchlist_update")).toBe(false);
      expect(String(events.find((event) => event.type === "answer_delta")?.text)).toContain("Watchlist update failed:");
      expect(events.at(-1)?.type).toBe("done");
      expect(invoked).toHaveLength(failure === "args" ? 0 : 1);
      expect(runCalls).toEqual([]);
      expect(reasonCalls).toEqual([]);
    }
  });

  test("post-tool assessment cannot chain into the watchlist action", async () => {
    reset();
    winner = "get_sec_document";
    verdicts = ["update_watchlist"];
    await eventsFor("Microsoft risk factors?", "MSFT");
    expect(entryCalls().map((call) => call.tool)).toEqual(["get_sec_document"]);
    expect(invoked).toEqual([]);
    expect(runCalls).toHaveLength(1);
  });
});

describe("workspace widget chat action", () => {
  test("a validated receipt for the displayed stock emits widget_update and answers without evidence, reasoning, or research", async () => {
    reset();
    delete process.env.STOCKBOT_DEBUG;
    winner = "add_widget";
    argsReply = { tool: "add_widget", arguments: { type: "short-volume", ticker: "aapl" }, confidence: 1 };
    const prompt = "add a short volume widget";
    const events = await eventsFor(prompt, undefined, undefined, "AAPL");
    expect(events.slice(0, 4)).toEqual([
      { type: "agent_start", prompt },
      { type: "status", text: "Adding the widget…" },
      { type: "widget_update", ticker: "AAPL", widgetType: "short-volume" },
      { type: "answer_delta", text: "Added the short-volume widget to the AAPL workspace." },
    ]);
    expect(events.at(-1)?.type).toBe("done");
    expect(invoked).toEqual([["add_widget", { type: "short-volume", ticker: "aapl" }, "sess-test"]]);
    expect(ended).toEqual(["sess-test"]);
    expect(events.some((event) => event.type === "evidence")).toBe(false);
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
    expect(kernelCalls.map((call) => call.op)).toEqual(["route", "arguments"]);
  });

  test("failed tools, malformed or foreign receipts, and failed arguments answer visibly without an update or research", async () => {
    for (const failure of ["returned", "missing", "kind", "type", "ticker", "focus", "args"]) {
      reset();
      winner = "add_widget";
      if (failure === "returned") toolFailure = { category: "tool_error", error: "tool failed" };
      if (failure === "missing") widgetReceipt = undefined;
      if (failure === "kind") widgetReceipt = { result_type: "watchlist_update", widget_type: "short-volume", ticker: "AAPL" };
      if (failure === "type") widgetReceipt = { result_type: "widget_selection", widget_type: "options-flow", ticker: "AAPL" };
      if (failure === "ticker") widgetReceipt = { result_type: "widget_selection", widget_type: "short-volume", ticker: "aapl" };
      if (failure === "focus") widgetReceipt = { result_type: "widget_selection", widget_type: "short-volume", ticker: "MSFT" };
      if (failure === "args") argsDown = true;
      const events = await eventsFor("add a short volume widget", "AAPL");
      expect(events.some((event) => event.type === "widget_update")).toBe(false);
      expect(String(events.find((event) => event.type === "answer_delta")?.text)).toContain("Widget update failed:");
      expect(events.at(-1)?.type).toBe("done");
      expect(invoked).toHaveLength(failure === "args" ? 0 : 1);
      expect(ended).toEqual(failure === "args" ? [] : ["sess-test"]);
      expect(runCalls).toEqual([]);
      expect(reasonCalls).toEqual([]);
    }
  });

  test("post-tool assessment cannot chain into the widget action", async () => {
    reset();
    winner = "get_sec_document";
    verdicts = ["add_widget"];
    await eventsFor("Microsoft risk factors?", "MSFT");
    expect(entryCalls().map((call) => call.tool)).toEqual(["get_sec_document"]);
    expect(invoked).toEqual([]);
    expect(runCalls).toHaveLength(1);
  });
});
