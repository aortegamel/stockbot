import { randomUUID } from "node:crypto";
import { appendFile, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { ANSWER_SECTIONS, reason } from "@/lib/muse/client";
import { asksWatchlist, kernelRouter, parseTicker, parseWatchlist, REQUEST_WALL_MS, runKernelAgent } from "@/lib/agent/kernel";
import { describeTools, endSession, invoke, newSessionId } from "@/lib/tools/stockbot";
import { ASK_CONTEXT_KINDS, FINAL_PERSONAS, safeSourceUrl, sourceReferences, statusEvent, toolStatus, WIDGET_TYPES, type AgentEvent, type AskContext, type Evidence, type Timings, type ToolResult } from "@/lib/agent/types";

const ROOT = process.cwd().endsWith("needle-harness") ? join(process.cwd(), "..") : process.cwd();

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}
const ASK_CONTEXT_KEYS = ["kind", "title", "url", "source", "sym", "at"];
// Feed-item context beside the verbatim prompt: undefined when absent, null when
// malformed (callers reject; unknown keys reject too). The values stay untrusted data.
function parseAskContext(v: unknown): AskContext | null | undefined {
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "object" || Array.isArray(v) || Object.keys(v).some((k) => !ASK_CONTEXT_KEYS.includes(k))) return null;
  const { kind, title, url, source, sym, at } = v as Record<string, unknown>;
  const text = (s: unknown): s is string => typeof s === "string" && s.trim() !== "" && s.length <= 300;
  if (!ASK_CONTEXT_KINDS.some((k) => k === kind) || !text(title) || !text(source) || typeof at !== "string" || at.length > 40) return null;
  if (url !== undefined && (typeof url !== "string" || url.length > 2000 || safeSourceUrl(url) === undefined)) return null;
  const ticker = parseTicker(sym);
  if (sym !== undefined && ticker === undefined) return null;
  return { kind: kind as AskContext["kind"], title, source, at, ...(url !== undefined ? { url } : {}), ...(ticker !== undefined ? { sym: ticker } : {}) };
}
// Default chat is one plain model call. mode "research" runs the full pipeline with the trio.
type AgentRequest = { prompt: string; ticker: unknown; currentChartTicker: unknown; parsedWatchlist: string[] | undefined; context: AskContext | undefined; research: boolean };
async function readAgentRequest(req: Request): Promise<Response | AgentRequest> {
  let prompt: unknown;
  let ticker: unknown;
  let currentChartTicker: unknown;
  let watchlistRaw: unknown;
  let contextRaw: unknown;
  let mode: unknown;
  try {
    ({ prompt, ticker, currentChartTicker, watchlist: watchlistRaw, context: contextRaw, mode } = await req.json());
  } catch {
    console.error("[web] [agent-api] invalid JSON");
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }
  if (mode !== undefined && mode !== "research") {
    console.error("[web] [agent-api] invalid mode");
    return Response.json({ error: "mode must be research when set" }, { status: 400 });
  }
  if (typeof prompt !== "string" || !prompt.trim()) {
    console.error("[web] [agent-api] prompt required");
    return Response.json({ error: "prompt required" }, { status: 400 });
  }
  // Watchlist context rides beside the verbatim prompt; a malformed list fails loudly, never shrinks.
  const parsedWatchlist = parseWatchlist(watchlistRaw);
  if (parsedWatchlist === null) {
    console.error("[web] [agent-api] invalid watchlist");
    return Response.json({ error: "watchlist must list 1-25 valid tickers" }, { status: 400 });
  }
  // Feed-item context rides beside the verbatim prompt the same way; a malformed item fails loudly.
  const context = parseAskContext(contextRaw);
  if (context === null) {
    console.error("[web] [agent-api] invalid context");
    return Response.json({ error: "context must be a feed item: kind news|filing|insider|alert, title and source (1-300 chars), at (max 40 chars), optional http(s) url and ticker sym" }, { status: 400 });
  }
  return { prompt, ticker, currentChartTicker, parsedWatchlist, context, research: mode === "research" };
}

function publicEvent(e: AgentEvent): AgentEvent | null {
  if (["needle_decision", "tool_start", "tool_result", "tool_failed"].includes(e.type)) return null;
  switch (e.type) {
    case "evidence":
      return {
        type: "evidence",
        ...(e.sessionId ? { sessionId: e.sessionId } : {}),
        evidence: e.evidence.map((ev) => ({
          id: ev.id,
          source: ev.source,
          retrievedAt: ev.retrievedAt,
          content: "",
          ...(ev.title !== undefined ? { title: ev.title } : {}),
          ...(ev.url !== undefined ? { url: ev.url } : {}),
          sources: sourceReferences(ev.sources),
        })),
      };
    case "progress":
      return { type: "progress", stage: "working" };
    case "reasoning_start":
      return { type: "reasoning_start", model: "stockbot" };
    case "answer_partial":
      return e;
    default:
      return e;
  }
}

// Every done carries the request-owned timings: the route round and the first answer text.
function streamSender(req: Request, controller: ReadableStreamDefaultController<Uint8Array>, requestAt: number, timings: Timings): { send: (e: AgentEvent) => void; flush: () => Promise<void> } {
  const enc = new TextEncoder();
  const debug = ["1", "true", "yes"].includes((process.env.STOCKBOT_DEBUG ?? "").trim().toLowerCase());
  let terminalSent = false;
  const directory = resolve(ROOT, process.env.STOCKBOT_DATA_DIR ?? "data", "logs/agent-events");
  const logPath = join(directory, `${new Date().toISOString().replace(/[:.]/g, "-")}-${randomUUID()}.jsonl`);
  let logFailed = false;
  const warn = (error: unknown): void => {
    if (!logFailed) console.warn("[web] [agent-api] event log failed", error);
    logFailed = true;
  };
  let pending = mkdir(directory, { recursive: true }).then(() => appendFile(logPath, "")).catch(warn);
  const send = (e: AgentEvent): void => {
    if (req.signal.aborted) return;
    if (e.type === "answer_partial" || e.type === "answer_delta") timings.firstTextMs ??= performance.now() - requestAt;
    if (e.type === "done") e = { ...e, metrics: { ...e.metrics, timings: { ...e.metrics.timings, ...timings } } };
    // Record the full event before production filtering, in emission order.
    if (!logFailed) {
      try {
        const line = `${JSON.stringify({ timestamp: new Date().toISOString(), type: e.type, payload: e })}\n`;
        pending = pending.then(() => { if (!logFailed) return appendFile(logPath, line); }).catch(warn);
      } catch (error) {
        warn(error);
      }
    }
    let out = debug ? e : publicEvent(e);
    if (!out) return;
    if (!debug && (e.type === "failed" || e.type === "error")) {
      if (terminalSent) return;
      terminalSent = true;
      out = e.type === "failed"
        ? { type: "error", message: "Stockbot could not verify part of that request. The checked findings above remain." }
        : { type: "error", message: e.message };
    }
    try {
      controller.enqueue(enc.encode(`data: ${JSON.stringify(out)}\n\n`));
    } catch {
      // The client disconnected. Later sends do not affect the closed stream.
    }
  };
  return { send, flush: () => pending };
}

const SSE_HEADERS = { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" };

// Default chat tools: curated read-only research tools.
const CHAT_TOOLS = [
  "find_sec_entities", "search_sec_filings", "list_sec_filings", "get_sec_filing", "list_sec_documents", "get_sec_document",
  "search_web", "get_current_time", "get_material_events", "get_financial_statements", "get_analyst_estimates", "get_valuation_metrics", "get_insider_activity", "get_short_interest",
] as const;
// Days of news the chat prefetches for the stock in view.
const NEWS_DAYS = 30;
// Chat reads at most this many characters per filing document; the tool default is 12000.
const CHAT_DOCUMENT_CHARS = 6000;
// Plain chat shows text as written: drop Markdown emphasis, headings, and bullet markers.
function plainText(text: string): string {
  return text.replace(/\*\*|__/g, "").replace(/^[ \t]*(?:#{1,6}|[*+-])[ \t]+/gm, "");
}
// Default chat runs no JEV and no research. Only these deterministic command shapes reach a UI tool.
const WATCHLIST_COMMAND = /\b(?:add|remove)\b/i;
const WIDGET_COMMAND = /\badd\b[^.?!]*\bwidgets?\b/i;
// ponytail: whole-prompt chart commands only ("show AAPL", "open the chart for $nvda"); widen when users ask.
const CHART_COMMAND = /^\s*(?:show|open|chart|pull up|switch to|go to)\s+(?:me\s+)?(?:the\s+)?(?:chart\s+(?:for|of)\s+)?(\$?[A-Za-z][A-Za-z0-9.-]{0,9})(?:\s+chart)?\s*[.!]?\s*$/i;
// A chart ticker is a $-prefixed word or an all-caps word, so "show me the chart" never names CHART.
function chartCommand(prompt: string): string | undefined {
  const raw = CHART_COMMAND.exec(prompt)?.[1];
  if (raw === undefined || (!raw.startsWith("$") && raw !== raw.toUpperCase())) return undefined;
  return parseTicker(raw.slice(raw.startsWith("$") ? 1 : 0));
}

export async function POST(req: Request): Promise<Response> {
  const deadlineAt = Date.now() + REQUEST_WALL_MS;
  const requestAt = performance.now();
  const input = await readAgentRequest(req);
  if (input instanceof Response) return input;
  const { prompt, ticker, currentChartTicker: chartInput, parsedWatchlist, context, research } = input;
  // The feed item rides beside the verbatim prompt as labeled untrusted data. The prompt itself never changes.
  const contextField = context ? { context } : {};
  // Only a prompt that names the watchlist uses its symbols instead of the selected ticker.
  const watchlist = parsedWatchlist !== undefined && asksWatchlist(prompt) ? parsedWatchlist : undefined;
  // Validated ticker channel beside the verbatim prompt; invalid values drop (prompt never changes).
  const parsedTicker = watchlist ? undefined : parseTicker(ticker);
  const tickerField = parsedTicker !== undefined ? { ticker: parsedTicker } : {};
  const currentChartTicker = parseTicker(chartInput) ?? parseTicker(ticker);
  console.log(`[web] [agent-api] POST mode=${research ? "research" : "chat"} prompt_chars=${prompt.length}${watchlist ? ` watchlist=${watchlist.length}` : ""}`);
  const stream = new ReadableStream({
    async start(controller) {
      const timings: Timings = {};
      const { send, flush } = streamSender(req, controller, requestAt, timings);
      const zeroMetrics = (t0: number, calls: number) => ({ totalMs: performance.now() - t0, needle: { calls, totalMs: 0, escalations: 0 }, tools: { calls, totalMs: 0 }, muse: { calls: 0, totalMs: 0 }, evidence: { count: 0, characters: 0 }, failures: {} });
      // One plain answer with read-only research tools: no evidence gates, no inventory.
      const answerPlain = async (): Promise<void> => {
        const t0 = performance.now();
        send({ type: "agent_start", prompt });
        send(statusEvent("Thinking…"));
        send({ type: "reasoning_start", model: requiredEnv("OPENCODE_MODEL") });
        const sessionId = newSessionId();
        const tools = { calls: 0, totalMs: 0 };
        // Every successful tool row: its captured sources become the list below the answer.
        const used: Evidence[] = [];
        const callTool = async (name: string, args: Record<string, unknown>): Promise<ToolResult> => {
          send(statusEvent(toolStatus(name)));
          const started = performance.now();
          tools.calls += 1;
          const result = await invoke(name, args, sessionId, Math.max(deadlineAt - Date.now(), 1));
          tools.totalMs += performance.now() - started;
          if (result.ok) used.push(result.evidence);
          return result;
        };
        const runChatTool = async (name: string, args: Record<string, unknown>): Promise<string> => {
          // Offering schemas does not bind the model: only an offered tool reaches the bridge.
          if (!(CHAT_TOOLS as readonly string[]).includes(name)) return `error: ${name} is not available in chat`;
          // limit overrides max_chars in the tool, so one field caps every document read.
          const capped = name === "get_sec_document" ? { ...args, limit: Math.min(typeof args.limit === "number" ? args.limit : CHAT_DOCUMENT_CHARS, CHAT_DOCUMENT_CHARS) } : args;
          const result = await callTool(name, capped);
          return result.ok ? result.evidence.content : `error: ${result.error}`;
        };
        // The code, not the model, fetches news, the filing list, analyst estimates, and valuation for the stock in view. A failed fetch leaves that section out.
        const newsTicker = context?.sym ?? (watchlist ? undefined : currentChartTicker);
        const since = new Date(Date.now() - NEWS_DAYS * 86_400_000).toISOString().slice(0, 10);
        // JEV keeps the investor parts that fit this request. A failed or malformed pick keeps every part.
        const pickSections = async (): Promise<string[]> => {
          try {
            const reply = await kernelRouter.call({ op: "sections", prompt, ...(newsTicker ? { ticker: newsTicker } : {}), ...contextField, deadlineAt }, { signal: req.signal, timeoutMs: Math.max(deadlineAt - Date.now(), 1) });
            if (reply.error) throw new Error(String(reply.error));
            const picked: unknown = reply.sections;
            if (!Array.isArray(picked) || !picked.every((id) => typeof id === "string" && Object.hasOwn(ANSWER_SECTIONS, id))) throw new Error("JEV returned invalid sections.");
            return picked;
          } catch (err) {
            console.warn(`[web] [agent-api] chat sections kept all: ${err instanceof Error ? err.message : String(err)}`);
            return Object.keys(ANSWER_SECTIONS);
          }
        };
        try {
          const [schemas, prefetched, sections] = await Promise.all([
            describeTools(CHAT_TOOLS),
            Promise.all(newsTicker ? [
              callTool("search_web", { query: `${newsTicker} stock news`, category: "news", start_published_date: since }),
              callTool("list_sec_filings", { identifier: newsTicker, forms: ["8-K", "10-Q", "10-K"], limit: 10 }),
              callTool("get_analyst_estimates", { ticker: newsTicker }),
              callTool("get_valuation_metrics", { ticker: newsTicker }),
            ] : []),
            pickSections(),
          ]);
          const r = await reason({ prompt, evidence: prefetched.flatMap((row) => (row.ok ? [row.evidence] : [])), direct: true, plain: true, reasoningEffort: "low", tools: schemas, runTool: runChatTool, sections, ...(currentChartTicker && !watchlist ? { ticker: currentChartTicker } : {}), ...contextField, signal: req.signal, deadlineAt });
          const prose = plainText(r.prose);
          if (!prose) throw new Error("The model returned an empty answer.");
          if (used.length) send({ type: "evidence", evidence: used });
          send({ type: "answer_delta", text: prose });
          send({ type: "done", metrics: { ...zeroMetrics(t0, 0), tools, muse: { calls: 1, totalMs: performance.now() - t0, ...r.usage } }, status: "completed" });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          send({ type: "failed", category: "provider_error", message: msg.slice(0, 160), code: "MODEL_START_FAILED", stage: "chat" });
          send({ type: "error", message: msg, code: "MODEL_START_FAILED", stage: "chat" });
        } finally {
          await endSession(sessionId).catch(() => undefined);
        }
      };
      const generateArguments = async (tool: string): Promise<Record<string, unknown> | null> => {
        try {
          const gen = await kernelRouter.call({ op: "arguments", tool, objective: prompt, ...tickerField }, { signal: req.signal, timeoutMs: Math.max(deadlineAt - Date.now(), 1) });
          if (typeof gen.error === "string" && gen.error) return null;
          if (gen.tool !== tool || typeof gen.arguments !== "object" || gen.arguments === null) return null;
          return gen.arguments as Record<string, unknown>;
        } catch {
          return null;
        }
      };
      const runResearch = async (): Promise<void> => {
        let reasoningEffort: string | undefined;
        const configuredEffort = process.env.REASONING_EFFORT?.trim();
        if (!configuredEffort || configuredEffort === "auto") {
          const startedAt = performance.now();
          try {
            const maxOutputTokens = process.env.MUSE_MAX_OUTPUT_TOKENS?.trim();
            const answerTokenBudget = process.env.MUSE_ANSWER_TOKEN_BUDGET?.trim();
            for (const [name, raw] of [["MUSE_MAX_OUTPUT_TOKENS", maxOutputTokens], ["MUSE_ANSWER_TOKEN_BUDGET", answerTokenBudget]] as const) {
              if (raw && (!Number.isInteger(Number(raw)) || Number(raw) < 16)) throw new Error(`${name} must be an integer >= 16`);
            }
            const selected = await kernelRouter.call({
              op: "effort", prompt, ...(!watchlist && (parsedTicker ?? currentChartTicker) ? { ticker: parsedTicker ?? currentChartTicker } : {}), ...(watchlist ? { watchlist } : {}), ...contextField, deadlineAt,
              ...(maxOutputTokens ? { maxOutputTokens: Number(maxOutputTokens) } : {}),
              ...(answerTokenBudget ? { answerTokenBudget: Number(answerTokenBudget) } : {}),
            }, { signal: req.signal, timeoutMs: Math.max(deadlineAt - Date.now(), 1) });
            if (selected.error) throw new Error(selected.error);
            if (typeof selected.effort !== "string" || !["low", "medium", "high", "xhigh"].includes(selected.effort)) {
              throw new Error("JEV returned an invalid reasoning effort.");
            }
            reasoningEffort = selected.effort;
            console.log(`[web] [agent-api] research reasoning_effort=${reasoningEffort}`);
          } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            send({ type: "failed", category: "model_error", message: message.slice(0, 160), code: "EFFORT_SELECTION_FAILED", stage: "effort" });
            send({ type: "error", message, code: "EFFORT_SELECTION_FAILED", stage: "effort" });
            return;
          } finally {
            timings.routeMs = performance.now() - startedAt;
          }
        }
        await runKernelAgent(prompt, send, { signal: req.signal, deadlineAt, personas: [...FINAL_PERSONAS], ...tickerField, ...(currentChartTicker ? { currentChartTicker } : {}), ...(watchlist ? { watchlist } : {}), ...contextField, ...(reasoningEffort ? { reasoningEffort } : {}) });
      };
      // A UI action is a receipt, never evidence. Only a validated canonical receipt emits its update event.
      // A failure answers visibly and never researches.
      const applyAction = async (status: string, failure: string, apply: (sessionId: string) => Promise<string>): Promise<void> => {
        const t0 = performance.now();
        send({ type: "agent_start", prompt });
        send(statusEvent(status));
        const sessionId = newSessionId();
        let text: string;
        try {
          text = await apply(sessionId);
        } catch (err) {
          text = `${failure}: ${(err instanceof Error ? err.message : String(err)).slice(0, 160)}`;
        } finally {
          try {
            await endSession(sessionId);
          } catch {
            // Cleanup does not change an applied receipt.
          }
        }
        send({ type: "answer_delta", text });
        send({ type: "done", metrics: zeroMetrics(t0, 1), status: "completed" });
      };
      const runTool = async (tool: string, args: Record<string, unknown>, sessionId: string): Promise<Record<string, unknown>> => {
        const result = await invoke(tool, args, sessionId, Math.max(deadlineAt - Date.now(), 1));
        if (!result.ok) throw new Error(result.error);
        const receipt = tool === "update_watchlist" ? result.watchlistUpdate : tool === "add_widget" ? result.widgetSelection : result.chartSelection;
        if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) throw new Error(`The ${tool} tool returned an invalid receipt.`);
        return receipt as Record<string, unknown>;
      };
      try {
        if (research) {
          await runResearch();
        } else if (WATCHLIST_COMMAND.test(prompt) && asksWatchlist(prompt)) {
          await applyAction("Updating your watchlist…", "Watchlist update failed", async (sessionId) => {
            const args = await generateArguments("update_watchlist");
            if (!args) throw new Error("no valid stock and action.");
            const { result_type, action, ticker: t } = await runTool("update_watchlist", args, sessionId);
            if (result_type !== "watchlist_update" || (action !== "add" && action !== "remove") || typeof t !== "string" || parseTicker(t) !== t) throw new Error("The watchlist tool returned an invalid receipt.");
            send({ type: "watchlist_update", action, ticker: t });
            return action === "add" ? `Added ${t} to your watchlist.` : `Removed ${t} from your watchlist.`;
          });
        } else if (WIDGET_COMMAND.test(prompt)) {
          await applyAction("Adding the widget…", "Widget update failed", async (sessionId) => {
            const args = await generateArguments("add_widget");
            if (!args) throw new Error("no valid widget and stock.");
            const { result_type, widget_type, ticker: t } = await runTool("add_widget", args, sessionId);
            if (result_type !== "widget_selection" || typeof widget_type !== "string" || !WIDGET_TYPES.includes(widget_type) || typeof t !== "string" || parseTicker(t) !== t) throw new Error("The widget tool returned an invalid receipt.");
            if (t !== currentChartTicker) throw new Error(`The widget targets ${t}, but the displayed stock is ${currentChartTicker ?? "not set"}.`);
            send({ type: "widget_update", ticker: t, widgetType: widget_type });
            return `Added the ${widget_type} widget to the ${t} workspace.`;
          });
        } else {
          const focus = chartCommand(prompt);
          if (focus === undefined) {
            await answerPlain();
          } else {
            await applyAction("Updating the chart…", "Chart update failed", async (sessionId) => {
              const { result_type, ticker: t } = await runTool("set_chart_symbol", { ticker: focus }, sessionId);
              if (result_type !== "chart_selection" || t !== focus) throw new Error("The chart tool returned an invalid receipt.");
              send({ type: "chart_update", ticker: focus });
              return `Showing ${focus} on the chart.`;
            });
          }
        }
      } catch (err) {
        console.error(`[web] [agent-api] ${research ? "runKernelAgent" : "chat"} error: ${err instanceof Error ? err.message : String(err)}`);
        if (req.signal.aborted) send({ type: "cancelled", message: "The request was cancelled." });
        else send({ type: "error", message: err instanceof Error ? err.message : String(err), code: "KERNEL_FAILED", stage: research ? "research" : "chat" });
      }
      await flush();
      controller.close();
      console.log("[web] [agent-api] stream close");
    },
  });
  return new Response(stream, { headers: SSE_HEADERS });
}