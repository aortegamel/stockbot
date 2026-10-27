import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentEvent, Evidence } from "@/lib/agent/types";
import type { reason } from "@/lib/muse/client";
// Capture the production validators before mock.module replaces the kernel.
import { asksWatchlist, parseTicker, parseWatchlist } from "@/lib/agent/kernel";
const realParseTicker = parseTicker;
const realParseWatchlist = parseWatchlist;
const realAsksWatchlist = asksWatchlist;

type Body = Record<string, unknown>;
type Event = { type: string;[key: string]: unknown };
const reasonCalls: Parameters<typeof reason>[0][] = [];
const kernelCalls: Body[] = [];
const kernelOpts: unknown[] = [];
const runCalls: unknown[][] = [];
const invoked: unknown[][] = [];
const ended: string[] = [];
let reasonText = "Hello! How can I help?";
let reasonDown = false;
let reasonUsage = {};
let argsDown = false;
let argsReply: Body | null = null;
let effortReply: Body = { effort: "medium" };
let runThrows: unknown = null;
let researchEvents: AgentEvent[] = [];
let toolFailure: string | null = null;
let toolThrows = false;
let endThrows = false;
let chartReceipt: unknown;
let watchlistReceipt: unknown;
let widgetReceipt: unknown;
let sessionsCreated = 0;
const capturedSource = { id: `capture:${"a".repeat(64)}`, title: "Captured filing", url: "https://www.sec.gov/filing" };
const bridgeEvidence: Evidence = {
  id: "ev:bridge", source: "bridge", retrievedAt: "2026-01-01T00:00:00Z", content: "internal tool output",
  title: "get_sec_document", url: "https://www.sec.gov/tool-output",
  sourceHandle: { raw_archive_path: "/private/raw" }, sourceRefs: { internal: "log" }, sources: [capturedSource],
};

mock.module("@/lib/agent/kernel", () => ({
  REQUEST_WALL_MS: 120_000,
  parseTicker: realParseTicker,
  parseWatchlist: realParseWatchlist,
  asksWatchlist: realAsksWatchlist,
  kernelRouter: {
    call: async (body: Body, opts: unknown) => {
      kernelCalls.push(body);
      kernelOpts.push(opts);
      if (argsDown) throw new Error("needle down");
      if (body.op === "effort") return effortReply;
      return argsReply ?? { tool: body.tool, arguments: { q: "x" } };
    },
  },
  runKernelAgent: async (...args: unknown[]) => {
    runCalls.push(args);
    if (runThrows !== null) throw runThrows;
    const emit = args[1] as (event: AgentEvent) => void;
    for (const event of researchEvents) emit(event);
  },
}));
mock.module("@/lib/muse/client", () => ({
  reason: async (opts: Parameters<typeof reason>[0]) => {
    reasonCalls.push(opts);
    if (reasonDown) throw new Error("muse down");
    return { prose: reasonText, usage: reasonUsage };
  },
}));
mock.module("@/lib/tools/stockbot", () => ({
  invoke: async (name: string, args: unknown, sessionId: string, timeoutMs: number) => {
    invoked.push([name, args, sessionId, timeoutMs]);
    if (toolThrows) throw new Error("tool failed");
    if (toolFailure !== null) return { ok: false, error: toolFailure };
    if (name === "set_chart_symbol") return { ok: true, chartSelection: chartReceipt };
    if (name === "update_watchlist") return { ok: true, watchlistUpdate: watchlistReceipt };
    if (name === "add_widget") return { ok: true, widgetSelection: widgetReceipt };
    throw new Error(`Unexpected tool: ${name}`);
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

let directory: string;
const savedEnv = { debug: process.env.STOCKBOT_DEBUG, model: process.env.OPENCODE_MODEL, data: process.env.STOCKBOT_DATA_DIR, effort: process.env.REASONING_EFFORT, maxOutputTokens: process.env.MUSE_MAX_OUTPUT_TOKENS, answerTokenBudget: process.env.MUSE_ANSWER_TOKEN_BUDGET };
beforeEach(async () => {
  reasonCalls.length = kernelCalls.length = kernelOpts.length = runCalls.length = invoked.length = ended.length = 0;
  reasonText = "Hello! How can I help?";
  reasonDown = argsDown = toolThrows = endThrows = false;
  reasonUsage = {};
  argsReply = null;
  effortReply = { effort: "medium" };
  runThrows = toolFailure = null;
  researchEvents = [];
  sessionsCreated = 0;
  chartReceipt = { result_type: "chart_selection", ticker: "AAPL" };
  watchlistReceipt = { result_type: "watchlist_update", action: "add", ticker: "NVDA" };
  widgetReceipt = { result_type: "widget_selection", widget_type: "short-volume", ticker: "AAPL" };
  directory = await mkdtemp(join(tmpdir(), "agent-events-"));
  process.env.STOCKBOT_DATA_DIR = directory;
  process.env.STOCKBOT_DEBUG = "1";
  process.env.OPENCODE_MODEL = "test-model";
  process.env.REASONING_EFFORT = "medium";
  delete process.env.MUSE_MAX_OUTPUT_TOKENS;
  delete process.env.MUSE_ANSWER_TOKEN_BUDGET;
});
afterEach(async () => {
  for (const [name, value] of [["STOCKBOT_DEBUG", savedEnv.debug], ["OPENCODE_MODEL", savedEnv.model], ["STOCKBOT_DATA_DIR", savedEnv.data], ["REASONING_EFFORT", savedEnv.effort], ["MUSE_MAX_OUTPUT_TOKENS", savedEnv.maxOutputTokens], ["MUSE_ANSWER_TOKEN_BUDGET", savedEnv.answerTokenBudget]] as const) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  await rm(directory, { recursive: true, force: true });
});

// Dynamic imports ensure route.ts binds the mocks instead of the real kernel and client.
async function post(body: Body): Promise<Response> {
  const { POST } = await import("./route");
  return POST(new Request("http://localhost/api/agent", { method: "POST", body: JSON.stringify(body) }));
}
async function eventsFor(prompt: string, fields: Body = {}): Promise<Event[]> {
  const response = await post({ prompt, ...fields });
  expect(response.status).toBe(200);
  expect(response.headers.get("Content-Type")).toBe("text/event-stream");
  return (await response.text()).split("\n\n").filter((chunk) => chunk.trim()).map((chunk) => JSON.parse(chunk.replace(/^data: /, "")));
}
function expectNoExecution(): void {
  expect(kernelCalls).toEqual([]);
  expect(runCalls).toEqual([]);
  expect(reasonCalls).toEqual([]);
  expect(invoked).toEqual([]);
}

describe("plain chat", () => {
  test("the default prompt makes one plain call and streams its unchanged answer", async () => {
    const events = await eventsFor("hello");
    expect(reasonCalls).toHaveLength(1);
    expect(reasonCalls[0]).toEqual({ prompt: "hello", evidence: [], direct: true, plain: true, signal: expect.any(AbortSignal), deadlineAt: expect.any(Number) });
    expect(kernelCalls).toEqual([]);
    expect(runCalls).toEqual([]);
    expect(invoked).toEqual([]);
    expect(sessionsCreated).toBe(0);
    expect(events.map((event) => event.type)).toEqual(["agent_start", "status", "reasoning_start", "answer_delta", "done"]);
    expect(events[3]).toEqual({ type: "answer_delta", text: reasonText });
    expect(events.at(-1)).toMatchObject({ status: "completed", metrics: { needle: { calls: 0 }, tools: { calls: 0 }, muse: { calls: 1 } } });
  });
  test("the done event preserves provider usage without route timings", async () => {
    reasonUsage = { providerCalls: 2, inputTokens: 13, outputTokens: 5 };
    const events = await eventsFor("Explain valuation");
    expect(events.at(-1)?.metrics).toMatchObject({ muse: { calls: 1, ...reasonUsage }, timings: { firstTextMs: expect.any(Number) } });
    expect(events.at(-1)?.metrics).not.toHaveProperty("timings.routeMs");
  });
  test("a reason failure emits failed and error events with the chat stage", async () => {
    reasonDown = true;
    const events = await eventsFor("hello");
    expect(events.slice(-2)).toEqual([
      { type: "failed", category: "provider_error", message: "muse down", code: "MODEL_START_FAILED", stage: "chat" },
      { type: "error", message: "muse down", code: "MODEL_START_FAILED", stage: "chat" },
    ]);
    expect(events.some((event) => event.type === "done" || event.type === "answer_delta")).toBe(false);
    expect(runCalls).toEqual([]);
  });
  test("a public reason failure emits only one error", async () => {
    delete process.env.STOCKBOT_DEBUG;
    reasonDown = true;
    const events = await eventsFor("hello");
    expect(events.map((event) => event.type)).toEqual(["agent_start", "status", "reasoning_start", "error"]);
    expect(events[2]).toEqual({ type: "reasoning_start", model: "stockbot" });
    expect(events.at(-1)).toEqual({ type: "error", message: "Stockbot could not verify part of that request. The checked findings above remain." });
  });
  test("empty prose fails without an answer or research", async () => {
    reasonText = "";
    const events = await eventsFor("hello");
    expect(events.at(-1)).toEqual({ type: "error", message: "The model returned an empty answer.", code: "MODEL_START_FAILED", stage: "chat" });
    expect(events.some((event) => event.type === "answer_delta" || event.type === "done")).toBe(false);
    expect(runCalls).toEqual([]);
  });
  test("the displayed ticker supplies plain context without changing the prompt", async () => {
    await eventsFor("How is it doing?", { ticker: "NVDA", currentChartTicker: " msft " });
    expect(reasonCalls[0]).toMatchObject({ prompt: "How is it doing?", ticker: "MSFT" });
    expect(kernelCalls).toEqual([]);
  });
  test("a missing model fails without research", async () => {
    delete process.env.OPENCODE_MODEL;
    const events = await eventsFor("hello");
    expect(events.at(-1)).toEqual({ type: "error", message: "opencode_unavailable: missing OPENCODE_MODEL", code: "KERNEL_FAILED", stage: "chat" });
    expectNoExecution();
  });
  test("an aborted request emits no events", async () => {
    const { POST } = await import("./route");
    const controller = new AbortController();
    controller.abort();
    const response = await POST(new Request("http://localhost/api/agent", { method: "POST", body: JSON.stringify({ prompt: "hello" }), signal: controller.signal }));
    expect(await response.text()).toBe("");
  });
});

describe("research mode", () => {
  test("concrete effort calls the kernel once with the full trio and no route op", async () => {
    const prompt = "What drove NVDA revenue?";
    await eventsFor(prompt, { mode: "research", ticker: "nvda" });
    expect(runCalls).toHaveLength(1);
    expect(runCalls[0]?.[0]).toBe(prompt);
    expect(runCalls[0]?.[2]).toMatchObject({ personas: ["stockbot", "bearbot", "bullbot"], ticker: "NVDA", currentChartTicker: "NVDA", signal: expect.any(AbortSignal), deadlineAt: expect.any(Number) });
    expect(reasonCalls).toEqual([]);
    expect(kernelCalls).toEqual([]);
    expect(invoked).toEqual([]);
    expect(runCalls[0]?.[2]).not.toHaveProperty("reasoningEffort");
  });
  test("watchlist research normalizes symbols and keeps chart context separate", async () => {
    await eventsFor("Summarize my watchlist", { mode: "research", ticker: "aapl", watchlist: ["nvda", "QQQ", "nvda", "spy"] });
    expect(runCalls[0]?.[2]).toMatchObject({ watchlist: ["NVDA", "QQQ", "SPY"], currentChartTicker: "AAPL" });
    expect(runCalls[0]?.[2]).not.toHaveProperty("ticker");
  });
  test("a selected-stock question ignores watchlist scope", async () => {
    await eventsFor("NVDA risk factors?", { mode: "research", ticker: "nvda", watchlist: ["NVDA", "AAPL"] });
    expect(runCalls[0]?.[2]).toMatchObject({ ticker: "NVDA" });
    expect(runCalls[0]?.[2]).not.toHaveProperty("watchlist");
  });
  test("auto, unset, and blank select effort before the kernel with the original prompt and scope", async () => {
    for (const configured of ["auto", undefined, "  "]) {
      if (configured === undefined) delete process.env.REASONING_EFFORT;
      else process.env.REASONING_EFFORT = configured;
      kernelCalls.length = kernelOpts.length = runCalls.length = 0;
      effortReply = { effort: "xhigh" };
      const prompt = "What drove NVDA revenue?";
      const context = { kind: "filing", title: "Quarterly filing", source: "SEC", at: "2026-10-27", sym: "NVDA" };
      await eventsFor(prompt, { mode: "research", ticker: "nvda", context });
      expect(kernelCalls).toEqual([{ op: "effort", prompt, ticker: "NVDA", context, deadlineAt: expect.any(Number) }]);
      expect(kernelOpts).toEqual([{ signal: expect.any(AbortSignal), timeoutMs: expect.any(Number) }]);
      expect(runCalls).toHaveLength(1);
      expect(runCalls[0]?.[2]).toMatchObject({ reasoningEffort: "xhigh", deadlineAt: kernelCalls[0]?.deadlineAt });
    }
  });
  test.each(["low", "medium", "high", "xhigh"])("research accepts and logs the selected %s effort", async (effort) => {
    process.env.REASONING_EFFORT = "auto";
    effortReply = { effort };
    const log = spyOn(console, "log").mockImplementation(() => { });
    try {
      await eventsFor("filing facts?", { mode: "research" });
      expect(runCalls[0]?.[2]).toMatchObject({ reasoningEffort: effort });
      expect(log).toHaveBeenCalledWith(`[web] [agent-api] research reasoning_effort=${effort}`);
    } finally {
      log.mockRestore();
    }
  });
  test("effort selection uses the displayed ticker when the request omits its selected ticker", async () => {
    process.env.REASONING_EFFORT = "auto";
    await eventsFor("What changed?", { mode: "research", currentChartTicker: "msft" });
    expect(kernelCalls[0]).toMatchObject({ op: "effort", ticker: "MSFT" });
  });
  test.each(["MUSE_MAX_OUTPUT_TOKENS", "MUSE_ANSWER_TOKEN_BUDGET"])("invalid %s fails before selection or research", async (name) => {
    process.env.REASONING_EFFORT = "auto";
    process.env[name] = "invalid";
    const events = await eventsFor("filing facts?", { mode: "research" });
    expect(events.map((event) => event.type)).toEqual(["failed", "error"]);
    expect(events[1]).toMatchObject({ code: "EFFORT_SELECTION_FAILED", stage: "effort" });
    expect(kernelCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test("effort selection receives watchlist scope and configured answer limits", async () => {
    process.env.REASONING_EFFORT = "auto";
    process.env.MUSE_MAX_OUTPUT_TOKENS = "1024";
    process.env.MUSE_ANSWER_TOKEN_BUDGET = "4096";
    await eventsFor("Summarize my watchlist", { mode: "research", ticker: "AAPL", watchlist: ["nvda", "msft"] });
    expect(kernelCalls[0]).toEqual({
      op: "effort", prompt: "Summarize my watchlist", watchlist: ["NVDA", "MSFT"],
      deadlineAt: expect.any(Number), maxOutputTokens: 1024, answerTokenBudget: 4096,
    });
    expect(runCalls[0]?.[2]).toMatchObject({ reasoningEffort: "medium", currentChartTicker: "AAPL" });
  });
  test("selection records its timing through the existing route metric", async () => {
    process.env.REASONING_EFFORT = "auto";
    researchEvents = [{ type: "done", metrics: { totalMs: 1, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 0, totalMs: 0 }, evidence: { count: 0, characters: 0 }, failures: {} } }];
    const events = await eventsFor("filing facts?", { mode: "research" });
    expect(events.at(-1)?.metrics).toMatchObject({ timings: { routeMs: expect.any(Number) } });
  });
  test("invalid effort and worker errors fail without a kernel call", async () => {
    process.env.REASONING_EFFORT = "auto";
    for (const reply of [{ effort: "auto" }, { effort: "HIGH" }, { effort: 2 }, {}, { effort: "high", error: "selector failed" }]) {
      effortReply = reply;
      const events = await eventsFor("filing facts?", { mode: "research" });
      expect(events.map((event) => event.type)).toEqual(["failed", "error"]);
      expect(events[0]).toMatchObject({ category: "model_error", code: "EFFORT_SELECTION_FAILED", stage: "effort" });
      expect(events[1]).toMatchObject({ code: "EFFORT_SELECTION_FAILED", stage: "effort" });
      expect(runCalls).toEqual([]);
    }
  });
  test("a selector transport error fails without a kernel call", async () => {
    process.env.REASONING_EFFORT = "auto";
    argsDown = true;
    expect(await eventsFor("filing facts?", { mode: "research" })).toEqual([
      { type: "failed", category: "model_error", message: "needle down", code: "EFFORT_SELECTION_FAILED", stage: "effort" },
      { type: "error", message: "needle down", code: "EFFORT_SELECTION_FAILED", stage: "effort" },
    ]);
    expect(runCalls).toEqual([]);
  });
  test("chat never selects effort in auto mode", async () => {
    process.env.REASONING_EFFORT = "auto";
    await eventsFor("hello");
    expect(kernelCalls).toEqual([]);
    expect(runCalls).toEqual([]);
    expect(reasonCalls).toHaveLength(1);
  });
  test("research failures preserve the error and research stage", async () => {
    runThrows = new Error("kernel boom");
    expect(await eventsFor("filing facts?", { mode: "research" })).toEqual([{ type: "error", message: "kernel boom", code: "KERNEL_FAILED", stage: "research" }]);
    expect(reasonCalls).toEqual([]);
  });
  test("public research hides internal content and retains source references", async () => {
    delete process.env.STOCKBOT_DEBUG;
    researchEvents = [
      { type: "progress", stage: "intake_done", detail: { calls: 3 } },
      { type: "evidence", evidence: [bridgeEvidence], sessionId: "rs:test" },
      { type: "answer_delta", text: "A fact [ev:bridge]" },
      { type: "chart_update", ticker: "MSFT" },
    ];
    const events = await eventsFor("filing facts?", { mode: "research" });
    expect(events).toEqual([
      { type: "progress", stage: "working" },
      { type: "evidence", sessionId: "rs:test", evidence: [{ id: "ev:bridge", source: "bridge", retrievedAt: bridgeEvidence.retrievedAt, content: "", title: bridgeEvidence.title, url: bridgeEvidence.url, sources: [capturedSource] }] },
      { type: "answer_delta", text: "A fact [ev:bridge]" },
      { type: "chart_update", ticker: "MSFT" },
    ]);
    expect(JSON.stringify(events)).not.toContain("/private/raw");
  });
});

const feedContext = { kind: "news", title: "NVDA jumps overnight", url: "https://news.example/a?x=1", source: "Stocktwits", sym: "NVDA", at: "2026-10-20T09:00:00Z" };
describe("request validation", () => {
  test("invalid JSON and missing prompts fail before execution", async () => {
    const { POST } = await import("./route");
    for (const [body, error] of [["{", "invalid JSON"], ["null", "invalid JSON"], ["{}", "prompt required"], ['{"prompt":"   "}', "prompt required"], ['{"prompt":7}', "prompt required"]]) {
      const response = await POST(new Request("http://localhost/api/agent", { method: "POST", body }));
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error });
    }
    expectNoExecution();
  });
  test("deep and other invalid modes fail before execution", async () => {
    for (const mode of ["deep", "chat", "", null, 7]) {
      const response = await post({ prompt: "hello", mode });
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "mode must be research when set" });
    }
    expectNoExecution();
  });
  test("malformed watchlists fail before execution", async () => {
    for (const watchlist of [["SEC"], "NVDA", [], [7], ["NVDA", ""], Array.from({ length: 26 }, (_, i) => `T${i}`)]) {
      const response = await post({ prompt: "my watchlist filings", ticker: "NVDA", watchlist });
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "watchlist must list 1-25 valid tickers" });
    }
    expectNoExecution();
  });
  test("valid feed context reaches plain chat and research beside the unchanged prompt", async () => {
    const prompt = "What does this mean for NVDA?";
    await eventsFor(prompt, { context: { ...feedContext, sym: "nvda" } });
    await eventsFor(prompt, { mode: "research", context: { ...feedContext, sym: "nvda" } });
    expect(reasonCalls[0]).toMatchObject({ prompt, context: feedContext });
    expect(runCalls[0]?.[0]).toBe(prompt);
    expect(runCalls[0]?.[2]).toMatchObject({ context: feedContext });
    expect(kernelCalls).toEqual([]);
  });
  test("invalid context fields fail before execution", async () => {
    for (const context of [
      "news", [], { ...feedContext, extra: 1 },
      { ...feedContext, kind: "tweet" }, { ...feedContext, kind: undefined },
      { ...feedContext, title: "" }, { ...feedContext, title: "  " }, { ...feedContext, title: "x".repeat(301) }, { ...feedContext, title: 7 },
      { ...feedContext, source: "" }, { ...feedContext, source: "x".repeat(301) },
      { ...feedContext, url: "javascript:alert(1)" }, { ...feedContext, url: "ftp://news.example/a" }, { ...feedContext, url: "https://user:pw@news.example/a" },
      { ...feedContext, url: "not a url" }, { ...feedContext, url: `https://news.example/${"a".repeat(2000)}` }, { ...feedContext, url: 7 },
      { ...feedContext, sym: "SEC" }, { ...feedContext, sym: "nvda!" }, { ...feedContext, sym: 7 },
      { ...feedContext, at: 7 }, { ...feedContext, at: "x".repeat(41) },
    ]) {
      const response = await post({ prompt: "hello", context });
      expect(response.status).toBe(400);
      expect(await response.json()).toEqual({ error: "context must be a feed item: kind news|filing|insider|alert, title and source (1-300 chars), at (max 40 chars), optional http(s) url and ticker sym" });
    }
    expectNoExecution();
  });
  test("absent and null context remain absent", async () => {
    for (const context of [undefined, null]) {
      await eventsFor("hello", { context });
      await eventsFor("hello", { mode: "research", context });
    }
    for (const call of reasonCalls) expect(call).not.toHaveProperty("context");
    for (const call of runCalls) expect(call[2]).not.toHaveProperty("context");
  });
  test("invalid tickers drop without changing the prompt", async () => {
    for (const ticker of ["SEC", "nvda!", "TOOLONGTICKER12", 7]) {
      await eventsFor("NVDA outlook?", { ticker, mode: "research" });
    }
    for (const call of runCalls) {
      expect(call[0]).toBe("NVDA outlook?");
      expect(call[2]).not.toHaveProperty("ticker");
      expect(call[2]).not.toHaveProperty("currentChartTicker");
    }
  });
});

describe("deterministic UI commands", () => {
  test("a watchlist command generates arguments and emits its validated update", async () => {
    argsReply = { tool: "update_watchlist", arguments: { action: "add", ticker: "NVDA" } };
    const prompt = "add NVDA to my watchlist";
    const events = await eventsFor(prompt);
    expect(kernelCalls).toEqual([{ op: "arguments", tool: "update_watchlist", objective: prompt }]);
    expect(invoked[0]).toEqual(["update_watchlist", argsReply.arguments, "sess-test", expect.any(Number)]);
    expect(events).toContainEqual({ type: "watchlist_update", action: "add", ticker: "NVDA" });
    expect(events).toContainEqual({ type: "answer_delta", text: "Added NVDA to your watchlist." });
    expect(events.at(-1)).toMatchObject({ type: "done", status: "completed" });
    expect(ended).toEqual(["sess-test"]);
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test("a remove command emits the remove receipt", async () => {
    watchlistReceipt = { result_type: "watchlist_update", action: "remove", ticker: "NVDA" };
    const events = await eventsFor("remove NVDA from my watchlist");
    expect(events).toContainEqual({ type: "watchlist_update", action: "remove", ticker: "NVDA" });
    expect(events).toContainEqual({ type: "answer_delta", text: "Removed NVDA from your watchlist." });
  });
  test("a watchlist question stays plain", async () => {
    const events = await eventsFor("what is on my watchlist", { ticker: "AAPL", watchlist: ["NVDA"] });
    expect(reasonCalls).toHaveLength(1);
    expect(reasonCalls[0]).toMatchObject({ plain: true });
    expect(reasonCalls[0]).not.toHaveProperty("ticker");
    expect(kernelCalls).toEqual([]);
    expect(invoked).toEqual([]);
    expect(events).toContainEqual({ type: "answer_delta", text: reasonText });
  });
  test("a widget command emits an update for the displayed stock", async () => {
    argsReply = { tool: "add_widget", arguments: { type: "short-volume", ticker: "aapl" } };
    const prompt = "add a short volume widget";
    const events = await eventsFor(prompt, { currentChartTicker: "AAPL" });
    expect(kernelCalls).toEqual([{ op: "arguments", tool: "add_widget", objective: prompt }]);
    expect(invoked[0]).toEqual(["add_widget", argsReply.arguments, "sess-test", expect.any(Number)]);
    expect(events).toContainEqual({ type: "widget_update", ticker: "AAPL", widgetType: "short-volume" });
    expect(events).toContainEqual({ type: "answer_delta", text: "Added the short-volume widget to the AAPL workspace." });
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
    expect(ended).toEqual(["sess-test"]);
  });
  test("a widget ticker mismatch fails visibly without an update", async () => {
    widgetReceipt = { result_type: "widget_selection", widget_type: "short-volume", ticker: "MSFT" };
    const events = await eventsFor("add a short volume widget", { currentChartTicker: "AAPL" });
    expect(events).toContainEqual({ type: "answer_delta", text: "Widget update failed: The widget targets MSFT, but the displayed stock is AAPL." });
    expect(events.some((event) => event.type === "widget_update")).toBe(false);
    expect(events.at(-1)).toMatchObject({ type: "done", status: "completed" });
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test.each([["show AAPL", "AAPL"], ["open the chart for $nvda", "NVDA"]])("the chart command %s invokes the ticker %s", async (prompt, ticker) => {
    chartReceipt = { result_type: "chart_selection", ticker };
    const events = await eventsFor(prompt);
    expect(invoked).toEqual([["set_chart_symbol", { ticker }, "sess-test", expect.any(Number)]]);
    expect(events).toContainEqual({ type: "chart_update", ticker });
    expect(events).toContainEqual({ type: "answer_delta", text: `Showing ${ticker} on the chart.` });
    expect(events.at(-1)).toMatchObject({ type: "done", status: "completed" });
    expect(ended).toEqual(["sess-test"]);
    expect(reasonCalls).toEqual([]);
    expect(kernelCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test.each(["show me the chart", "show aapl"])("the ambiguous prompt %s stays plain", async (prompt) => {
    await eventsFor(prompt);
    expect(reasonCalls).toHaveLength(1);
    expect(reasonCalls[0]).toMatchObject({ prompt, plain: true, direct: true });
    expect(invoked).toEqual([]);
    expect(kernelCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test.each(["returned", "thrown", "missing", "kind", "ticker"])("a %s chart failure emits no update", async (failure) => {
    if (failure === "returned") toolFailure = "tool failed";
    if (failure === "thrown") toolThrows = true;
    if (failure === "missing") chartReceipt = undefined;
    if (failure === "kind") chartReceipt = { result_type: "research", ticker: "AAPL" };
    if (failure === "ticker") chartReceipt = { result_type: "chart_selection", ticker: "MSFT" };
    const events = await eventsFor("show AAPL");
    expect(events.some((event) => event.type === "chart_update")).toBe(false);
    expect(String(events.find((event) => event.type === "answer_delta")?.text)).toStartWith("Chart update failed:");
    expect(ended).toEqual(["sess-test"]);
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test.each(["error", "tool", "null", "string", "throw"])("invalid %s arguments fail without invocation", async (failure) => {
    argsReply = { tool: "update_watchlist", arguments: { action: "add", ticker: "NVDA" } };
    if (failure === "error") argsReply.error = "args failed";
    if (failure === "tool") argsReply.tool = "add_widget";
    if (failure === "null") argsReply.arguments = null;
    if (failure === "string") argsReply.arguments = "NVDA";
    if (failure === "throw") argsDown = true;
    const events = await eventsFor("add NVDA to my watchlist");
    expect(events.some((event) => event.type === "watchlist_update")).toBe(false);
    expect(String(events.find((event) => event.type === "answer_delta")?.text)).toStartWith("Watchlist update failed:");
    expect(invoked).toEqual([]);
    expect(ended).toEqual(["sess-test"]);
    expect(reasonCalls).toEqual([]);
    expect(runCalls).toEqual([]);
  });
  test("cleanup failure preserves an applied chart update", async () => {
    endThrows = true;
    const events = await eventsFor("show AAPL");
    expect(events).toContainEqual({ type: "chart_update", ticker: "AAPL" });
    expect(events.at(-1)).toMatchObject({ type: "done", status: "completed" });
  });
  test("arguments and tool calls share the request deadline", async () => {
    await eventsFor("add NVDA to my watchlist");
    expect(kernelOpts[0]).toMatchObject({ signal: expect.any(AbortSignal), timeoutMs: expect.any(Number) });
    const timeout = invoked[0]?.[3] as number;
    expect(timeout).toBeGreaterThan(0);
    expect(timeout).toBeLessThanOrEqual(120_000);
  });
});

describe("agent event logs", () => {
  async function logs(): Promise<{ name: string; entries: { timestamp: string; type: string; payload: Event }[] }[]> {
    const folder = join(directory, "logs/agent-events");
    return Promise.all((await readdir(folder)).map(async (name) => {
      const text = await readFile(join(folder, name), "utf8");
      expect(text.endsWith("\n")).toBe(true);
      return { name, entries: text.trimEnd().split("\n").map((line) => JSON.parse(line)) };
    }));
  }
  test("each query flushes timestamped events in order before the stream closes", async () => {
    const first = await eventsFor("hello");
    const second = await eventsFor("hello again");
    const files = await logs();
    expect(files).toHaveLength(2);
    for (const events of [first, second]) {
      const file = files.find(({ entries }) => entries[0]?.payload.prompt === events[0]?.prompt)!;
      expect(file.name).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z-[a-f0-9-]{36}\.jsonl$/);
      expect(file.entries.map(({ payload }) => payload)).toEqual(events);
      for (const entry of file.entries) {
        expect(new Date(entry.timestamp).toISOString()).toBe(entry.timestamp);
        expect(entry.type).toBe(entry.payload.type);
      }
    }
  });
  test("public research logs raw evidence before filtering", async () => {
    delete process.env.STOCKBOT_DEBUG;
    researchEvents = [{ type: "evidence", evidence: [bridgeEvidence], sessionId: "rs:test" }];
    const events = await eventsFor("filing facts?", { mode: "research" });
    const files = await logs();
    expect(files).toHaveLength(1);
    expect(files[0]?.entries[0]?.payload).toEqual(researchEvents[0]);
    expect(events[0]?.evidence).not.toEqual([bridgeEvidence]);
  });
  test("public chat failures log both terminal events before filtering", async () => {
    delete process.env.STOCKBOT_DEBUG;
    reasonDown = true;
    const events = await eventsFor("hello");
    const files = await logs();
    expect(files[0]?.entries.slice(-2).map(({ type }) => type)).toEqual(["failed", "error"]);
    expect(events.filter(({ type }) => type === "error")).toHaveLength(1);
  });
  test("a logging failure warns once without breaking the response", async () => {
    const blocked = join(directory, "blocked");
    await writeFile(blocked, "");
    process.env.STOCKBOT_DATA_DIR = blocked;
    const warning = spyOn(console, "warn").mockImplementation(() => { });
    try {
      const events = await eventsFor("hello");
      expect(events.at(-1)?.type).toBe("done");
      expect(events).toContainEqual({ type: "answer_delta", text: reasonText });
      expect(warning).toHaveBeenCalledTimes(1);
      expect(warning.mock.calls[0]?.[0]).toBe("[web] [agent-api] event log failed");
    } finally {
      warning.mockRestore();
    }
  });
});
