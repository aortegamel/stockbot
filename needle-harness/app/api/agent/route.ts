import { reason } from "@/lib/muse/client";
import { asksWatchlist, dropUncovered, gateAnswer, kernelRouter, parseTicker, parseWatchlist, REQUEST_WALL_MS, runKernelAgent, runPersonaAgent, type KernelResponse } from "@/lib/agent/kernel";
import { endSession, invoke, newSessionId } from "@/lib/tools/stockbot";
import { ASK_CONTEXT_KINDS, FINAL_PERSONAS, foundStatus, redactArgs, safeSourceUrl, sourceReferences, statusEvent, toolStatus, WIDGET_TYPES, type AgentEvent, type AskContext, type Evidence, type FailureCategory, type Persona, type Timings } from "@/lib/agent/types";
function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}
// Worker output is untrusted: null means no explicit selection; a list must be
// nonempty, unique, canonical, and in fixed order. Anything else is undefined (invalid).
function parsePersonas(v: unknown): Persona[] | null | undefined {
  if (v === null) return null;
  if (!Array.isArray(v)) return undefined;
  const canonical = FINAL_PERSONAS.filter((p) => v.includes(p));
  return v.length > 0 && v.length === canonical.length && canonical.every((p, i) => v[i] === p) ? canonical : undefined;
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
// Trio follow-up request: persona and gate stages over an offered session (same id shape as /api/research).
type PersonasRequest = { op: "personas"; sessionId: string; personas: Persona[] };
async function readAgentRequest(req: Request): Promise<Response | PersonasRequest | { prompt: string; ticker: unknown; currentChartTicker: unknown; parsedWatchlist: string[] | undefined; context: AskContext | undefined }> {
  let prompt: unknown;
  let ticker: unknown;
  let currentChartTicker: unknown;
  let watchlistRaw: unknown;
  let op: unknown;
  let sessionId: unknown;
  let personasRaw: unknown;
  let contextRaw: unknown;
  try {
    ({ prompt, ticker, currentChartTicker, watchlist: watchlistRaw, op, sessionId, personas: personasRaw, context: contextRaw } = await req.json());
  } catch {
    console.error("[web] [agent-api] invalid JSON");
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }
  if (op === "personas") {
    const personas = parsePersonas(personasRaw);
    if (typeof sessionId !== "string" || !/^[A-Za-z0-9:_-]{1,128}$/.test(sessionId) || !personas) {
      console.error("[web] [agent-api] invalid personas request");
      return Response.json({ error: "sessionId and personas required" }, { status: 400 });
    }
    return { op: "personas", sessionId, personas };
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
  return { prompt, ticker, currentChartTicker, parsedWatchlist, context };
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
    default:
      return e;
  }
}

// Every done carries the request-owned timings: the route round and the first answer text.
function streamSender(req: Request, controller: ReadableStreamDefaultController<Uint8Array>, requestAt: number, timings: Timings): (e: AgentEvent) => void {
  const enc = new TextEncoder();
  const debug = ["1", "true", "yes"].includes((process.env.STOCKBOT_DEBUG ?? "").trim().toLowerCase());
  let terminalSent = false;
  return (e) => {
    if (req.signal.aborted) return;
    if (e.type === "answer_delta") timings.firstTextMs ??= performance.now() - requestAt;
    if (e.type === "done") e = { ...e, metrics: { ...e.metrics, timings: { ...e.metrics.timings, ...timings } } };
    let out = debug ? e : publicEvent(e);
    if (!out) return;
    if (!debug && (e.type === "failed" || e.type === "error")) {
      if (terminalSent) return;
      terminalSent = true;
      out = { type: "error", message: "Stockbot couldn't complete that request. Try again." };
    }
    try {
      controller.enqueue(enc.encode(`data: ${JSON.stringify(out)}\n\n`));
    } catch {
      // The client disconnected. Later sends do not affect the closed stream.
    }
  };
}

type EntrySelection = { route: string; personas: Persona[] | null; chartTicker?: string; chartError?: string } | { error: string };
function parseEntrySelection(routed: KernelResponse | undefined): Exclude<EntrySelection, { error: string }> | null {
  const personas = parsePersonas(routed?.personas);
  if (!routed || (typeof routed.error === "string" && routed.error) || typeof routed.route !== "string"
    || routed.route === "set_chart_symbol" || personas === undefined) {
    return null;
  }
  const selection: Exclude<EntrySelection, { error: string }> = { route: routed.route, personas };
  const chartTicker = parseTicker(routed.chartTicker);
  if (routed.chartTicker !== undefined) {
    if (chartTicker === undefined || routed.chartError !== undefined) {
      selection.chartError = "The chart selection was invalid.";
    } else {
      selection.chartTicker = chartTicker;
    }
  } else if (routed.chartError !== undefined) {
    selection.chartError = typeof routed.chartError === "string" && routed.chartError.trim()
      ? routed.chartError.trim().slice(0, 160)
      : "The chart selection was invalid.";
  }
  return selection;
}
async function selectEntry(body: Record<string, unknown>, signal: AbortSignal): Promise<EntrySelection> {
  let routed: KernelResponse | undefined;
  let routeError = "route_invalid: malformed route selection";
  try {
    routed = await kernelRouter.call(body, { signal });
    if (typeof routed.error === "string" && routed.error) routeError = routed.error;
  } catch (err) {
    routeError = err instanceof Error ? err.message : String(err);
  }
  return parseEntrySelection(routed) ?? { error: routeError };
}

const SSE_HEADERS = { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" };

// Trio follow-up: no route round, intake, nodes, or tools; the kernel replays
// only the persona and final-gate stages over the offered session.
function personasResponse(req: Request, input: PersonasRequest, requestAt: number): Response {
  console.log(`[web] [agent-api] POST personas=${input.personas.join(",")}`);
  const stream = new ReadableStream({
    async start(controller) {
      const send = streamSender(req, controller, requestAt, {});
      try {
        await runPersonaAgent(input.sessionId, input.personas, send, { signal: req.signal });
      } catch (err) {
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
      }
      controller.close();
    },
  });
  return new Response(stream, { headers: SSE_HEADERS });
}

export async function POST(req: Request): Promise<Response> {
  const deadlineAt = Date.now() + REQUEST_WALL_MS;
  const requestAt = performance.now();
  const input = await readAgentRequest(req);
  if (input instanceof Response) return input;
  if ("op" in input) return personasResponse(req, input, requestAt);
  const { prompt, ticker, currentChartTicker: chartInput, parsedWatchlist, context } = input;
  // The feed item rides beside the verbatim prompt as labeled untrusted data: the router, the
  // research planner, and the reasoner see it. The prompt itself never changes.
  const contextField = context ? { context } : {};
  // Only a prompt that names the watchlist uses its symbols instead of the selected ticker.
  const watchlist = parsedWatchlist !== undefined && asksWatchlist(prompt) ? parsedWatchlist : undefined;
  const watchlistField = watchlist ? { watchlist } : {};
  // Validated ticker channel beside the verbatim prompt; invalid values drop (prompt never changes).
  const parsedTicker = watchlist ? undefined : parseTicker(ticker);
  let tickerField = parsedTicker !== undefined ? { ticker: parsedTicker } : {};
  let currentChartTicker = parseTicker(chartInput) ?? parseTicker(ticker);
  const reasonContext = (): { ticker?: string; context?: AskContext } => ({ ...(!watchlist && currentChartTicker ? { ticker: currentChartTicker } : {}), ...contextField });
  console.log(`[web] [agent-api] POST prompt_chars=${prompt.length}${watchlist ? ` watchlist=${watchlist.length}` : ""}`);
  const stream = new ReadableStream({
    async start(controller) {
      const timings: Timings = {};
      const send = streamSender(req, controller, requestAt, timings);
      // JEV owns the original route independently from the optional chart action.
      // A direct answer has no evidence, so it is buffered and shown minus every
      // datapoint sentence; nothing left means research answers instead.
      const answerDirect = async (): Promise<boolean> => {
        const t0 = performance.now();
        send({ type: "agent_start", prompt });
        send(statusEvent("Thinking…"));
        send({ type: "reasoning_start", model: requiredEnv("OPENCODE_MODEL") });
        try {
          const r = await reason({ prompt, evidence: [], direct: true, ...reasonContext(), onDelta: () => { } });
          const text = dropUncovered(r.text, [], false);
          if (!text) return false;
          send({ type: "answer_delta", text });
          send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 1, totalMs: performance.now() - t0, ...r.usage }, evidence: { count: 0, characters: 0 }, failures: {} } });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          // ponytail: muse-down shape mirrors kernel.ts — failed + error, never a silent close.
          send({ type: "failed", category: "provider_error", message: msg.slice(0, 160) });
          send({ type: "error", message: msg });
        }
        return true;
      };
      // ponytail: single-shot evidence must survive research fallback (risk SSE dropped ev:58f5).
      const singleShotEvidence: Evidence[] = [];
      const generateArguments = async (tool: string): Promise<{ args: Record<string, unknown>; confidence: number | null } | null> => {
        try {
          const gen = await kernelRouter.call({ op: "arguments", tool, objective: prompt, ...tickerField }, { signal: req.signal });
          if (typeof gen.error === "string" && gen.error) return null;
          if (gen.tool !== tool || typeof gen.arguments !== "object" || gen.arguments === null) return null;
          return {
            args: gen.arguments as Record<string, unknown>,
            confidence: typeof gen.confidence === "number" ? gen.confidence : null,
          };
        } catch {
          // Shared Needle failed. Research supplies the next arguments.
          return null;
        }
      };
      type ToolOutcome = { ok: boolean; content?: string; error?: string; category?: string };
      const assessTool = async (tool: string, args: Record<string, unknown>, outcome: ToolOutcome): Promise<string | null> => {
        try {
          const assessed = await kernelRouter.call(
            { op: "assess_entry", prompt, tool, arguments: args, result: outcome, ...tickerField },
            { signal: req.signal },
          );
          return typeof assessed.verdict === "string" && assessed.verdict ? assessed.verdict : "research_required";
        } catch {
          return null;
        }
      };
      // ponytail: JEV selects; shared kernel Needle fills args; every post-tool verdict returns to JEV (max 3 rounds).
      const answerSingleShot = async (toolName: string): Promise<boolean> => {
        const t0 = performance.now();
        const toolT0 = performance.now();
        send({ type: "agent_start", prompt });
        // No subject yet: the arguments name it at tool start.
        send(statusEvent(`${toolStatus(toolName, undefined, "Looking up")}…`));
        const evidence: Evidence[] = [];
        const failures: Partial<Record<FailureCategory, number>> = {};
        let needleCalls = 0;
        let toolCalls = 0;
        // The worker creates the ledger session before it executes the first tool.
        let sessionId: string | undefined;
        const executeTool = async (tool: string, args: Record<string, unknown>): Promise<ToolOutcome> => {
          try {
            const res = await kernelRouter.call(
              { op: "entry_tool", prompt, tool, arguments: args, ...(sessionId ? { sessionId } : {}) },
              { signal: req.signal, timeoutMs: Math.max(deadlineAt - Date.now(), 1) },
            );
            toolCalls += 1;
            if (typeof res.sessionId !== "string" || !res.sessionId || (sessionId && res.sessionId !== sessionId)) {
              throw new Error("The source result has no matching research session.");
            }
            sessionId = res.sessionId;
            if (res.error) throw new Error(res.error);
            let fresh = 0;
            for (const row of res.evidence ?? []) {
              if (evidence.some((e) => e.id === row.id)) continue;
              const ev: Evidence = {
                id: row.id, source: row.source ?? tool, content: row.content,
                retrievedAt: row.retrievedAt ?? new Date().toISOString(),
                ...(row.title !== undefined ? { title: row.title } : {}),
                ...(row.url !== undefined ? { url: row.url } : {}),
                sources: sourceReferences(row.sources),
              };
              evidence.push(ev);
              singleShotEvidence.push(ev);
              fresh += 1;
            }
            const content = res.content ?? "";
            send({ type: "tool_result", tool, preview: content.slice(0, 160) });
            send(statusEvent(foundStatus(fresh)));
            send({ type: "evidence", sessionId, evidence: [...evidence] });
            return { ok: true, content: content.slice(0, 1500) };
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            failures.tool_error = (failures.tool_error ?? 0) + 1;
            send({ type: "tool_failed", tool, category: "tool_error", preview: msg.slice(0, 160) });
            send(statusEvent("That source had no usable data"));
            return { ok: false, error: msg.slice(0, 500), category: "tool_error" };
          }
        };
        // Only ledger-backed claims may publish. A gate rejection ends this path visibly.
        const answerFromEvidence = async (defect?: string): Promise<boolean> => {
          if (defect || evidence.length === 0) {
            failures.tool_error = (failures.tool_error ?? 0) + 1;
            send({ type: "failed", category: "tool_error", message: `No verified answer: ${defect ?? "this source has no admissible research evidence."}` });
            send({ type: "answer_delta", text: "I could not verify an answer from this source. I withheld its factual claims." });
            send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: needleCalls, totalMs: 0, escalations: 0 }, tools: { calls: toolCalls, totalMs: performance.now() - toolT0 }, muse: { calls: 0, totalMs: 0 }, evidence: { count: 0, characters: 0 }, failures } });
            return true;
          }
          send({ type: "reasoning_start", model: requiredEnv("OPENCODE_MODEL") });
          try {
            const m0 = performance.now();
            const r = await reason({ prompt, evidence, direct: evidence.length === 0, ...reasonContext(), onDelta: () => { } });
            const gated = await gateAnswer({
              result: { persona: "stockbot", text: r.text, claims: r.claims, ...(r.confidence !== undefined ? { confidence: r.confidence } : {}) },
              evidence,
              sessionId,
              gate: (body, gateOpts) => kernelRouter.call(body, { signal: req.signal, ...gateOpts }),
            });
            if ("failure" in gated) {
              failures.provider_error = (failures.provider_error ?? 0) + 1;
              send({ type: "failed", category: "provider_error", message: `No verified answer: ${gated.failure}` });
              send({ type: "answer_delta", text: "I could not verify an answer from this source. I withheld its factual claims." });
              send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: needleCalls, totalMs: 0, escalations: 0 }, tools: { calls: toolCalls, totalMs: performance.now() - toolT0 }, muse: { calls: 1, totalMs: performance.now() - m0, ...r.usage }, evidence: { count: evidence.length, characters: evidence.reduce((n, e) => n + e.content.length, 0) }, failures } });
              return true;
            }
            send({ type: "answer_delta", text: gated.text });
            const claims = r.claims.filter((c) => gated.text.includes(c.text));
            if (claims.length > 0) send({ type: "claims", claims });
            send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: needleCalls, totalMs: 0, escalations: 0 }, tools: { calls: toolCalls, totalMs: performance.now() - toolT0 }, muse: { calls: 1, totalMs: performance.now() - m0, ...r.usage }, evidence: { count: evidence.length, characters: evidence.reduce((n, e) => n + e.content.length, 0) }, failures } });
          } catch (err) {
            send({ type: "error", message: err instanceof Error ? err.message : String(err) });
          }
          return true;
        };
        {
          let current: string | null = toolName;
          for (let step = 0; step < 3 && current; step++) {
            const tool = current;
            current = null;
            const generated = await generateArguments(tool);
            if (!generated) return false;
            const { args, confidence } = generated;
            needleCalls += 1;
            send({ type: "needle_decision", step, tool, arguments: redactArgs(args), confidence });
            send({ type: "tool_start", tool });
            send(statusEvent(toolStatus(tool, tool === "search_web" ? (typeof args.query === "string" ? args.query : undefined) : parseTicker(args.ticker))));
            const outcome = await executeTool(tool, args);
            if (!outcome.ok) return await answerFromEvidence(outcome.error ?? "The source result failed.");
            if (outcome.ok && evidence.length === 0) return await answerFromEvidence();
            const verdict = await assessTool(tool, args, outcome);
            if (verdict === null) return false;
            if (verdict === "node_resolved" || verdict === "reasoning_required") return await answerFromEvidence();
            if (verdict === "research_required" || verdict === "set_chart_symbol" || verdict === "update_watchlist" || verdict === "add_widget") return false;
            // ponytail: worker output is untrusted — only identifier-shaped tool names chain; anything else researches.
            if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(verdict)) return false;
            current = verdict;
          }
          // The cap answers from the ledger or ends with an explicit verification failure.
          return await answerFromEvidence();
        }
      };
      // The watchlist action is a UI receipt, never evidence: Needle fills its arguments and only
      // a validated canonical receipt emits watchlist_update. A failure answers visibly, never researches.
      const applyWatchlist = async (): Promise<boolean> => {
        const t0 = performance.now();
        send({ type: "agent_start", prompt });
        send(statusEvent("Updating your watchlist…"));
        const generated = await generateArguments("update_watchlist");
        let text = "Watchlist update failed: no valid stock and action.";
        if (generated) {
          const sessionId = newSessionId();
          try {
            const result = await invoke("update_watchlist", generated.args, sessionId, Math.max(deadlineAt - Date.now(), 1));
            if (!result.ok) throw new Error(result.error);
            const receipt = result.watchlistUpdate;
            if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) throw new Error("The watchlist tool returned an invalid receipt.");
            const { result_type, action, ticker } = receipt as Record<string, unknown>;
            if (result_type !== "watchlist_update" || (action !== "add" && action !== "remove") || typeof ticker !== "string" || parseTicker(ticker) !== ticker) {
              throw new Error("The watchlist tool returned an invalid receipt.");
            }
            send({ type: "watchlist_update", action, ticker });
            text = action === "add" ? `Added ${ticker} to your watchlist.` : `Removed ${ticker} from your watchlist.`;
          } catch (err) {
            text = `Watchlist update failed: ${(err instanceof Error ? err.message : String(err)).slice(0, 160)}`;
          } finally {
            try {
              await endSession(sessionId);
            } catch {
              // Cleanup does not change an applied watchlist receipt.
            }
          }
        }
        send({ type: "answer_delta", text });
        send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: generated ? 1 : 0, totalMs: 0, escalations: 0 }, tools: { calls: generated ? 1 : 0, totalMs: 0 }, muse: { calls: 0, totalMs: 0 }, evidence: { count: 0, characters: 0 }, failures: {} } });
        return true;
      };
      // The widget action is a UI receipt for the displayed stock's workspace, never evidence.
      // Only a validated canonical receipt for that stock emits widget_update. A failure answers visibly, never researches.
      const applyWidget = async (): Promise<boolean> => {
        const t0 = performance.now();
        send({ type: "agent_start", prompt });
        send(statusEvent("Adding the widget…"));
        const generated = await generateArguments("add_widget");
        let text = "Widget update failed: no valid widget and stock.";
        if (generated) {
          const sessionId = newSessionId();
          try {
            const result = await invoke("add_widget", generated.args, sessionId, Math.max(deadlineAt - Date.now(), 1));
            if (!result.ok) throw new Error(result.error);
            const receipt = result.widgetSelection;
            if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) throw new Error("The widget tool returned an invalid receipt.");
            const { result_type, widget_type, ticker } = receipt as Record<string, unknown>;
            if (result_type !== "widget_selection" || typeof widget_type !== "string" || !WIDGET_TYPES.includes(widget_type) || typeof ticker !== "string" || parseTicker(ticker) !== ticker) {
              throw new Error("The widget tool returned an invalid receipt.");
            }
            if (ticker !== currentChartTicker) throw new Error(`The widget targets ${ticker}, but the displayed stock is ${currentChartTicker ?? "not set"}.`);
            send({ type: "widget_update", ticker, widgetType: widget_type });
            text = `Added the ${widget_type} widget to the ${ticker} workspace.`;
          } catch (err) {
            text = `Widget update failed: ${(err instanceof Error ? err.message : String(err)).slice(0, 160)}`;
          } finally {
            try {
              await endSession(sessionId);
            } catch {
              // Cleanup does not change an applied widget receipt.
            }
          }
        }
        send({ type: "answer_delta", text });
        send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: generated ? 1 : 0, totalMs: 0, escalations: 0 }, tools: { calls: generated ? 1 : 0, totalMs: 0 }, muse: { calls: 0, totalMs: 0 }, evidence: { count: 0, characters: 0 }, failures: {} } });
        return true;
      };
      const tryEntryAnswer = async (winner: string): Promise<boolean> => {
        try {
          if (winner === "clarify_subject") {
            const t0 = performance.now();
            send({ type: "agent_start", prompt });
            send({ type: "answer_delta", text: "Which stock, ETF, sector, or holding do you want advice on? Give a ticker (for example NVDA) or a company name, and I will research it." });
            send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: 0, totalMs: 0, escalations: 0 }, tools: { calls: 0, totalMs: 0 }, muse: { calls: 0, totalMs: 0 }, evidence: { count: 0, characters: 0 }, failures: {} } });
            return true;
          }
          if (winner === "no_session" || winner === "reasoning_required") return await answerDirect();
          if (winner === "update_watchlist") return await applyWatchlist();
          if (winner === "add_widget") return await applyWidget();
          // Only identifier-shaped tool names can enter the single-tool path.
          if (winner !== "research_required" && /^[A-Za-z_][A-Za-z0-9_]*$/.test(winner)) return await answerSingleShot(winner);
        } catch {
          // The direct answer failed to start. Research owns the next attempt.
        }
        return false;
      };
      // ponytail: JEV-first entry — one route round selects the route and any explicit personas.
      // The watchlist rides as separate context; JEV's returned route governs dispatch here.
      const routeAt = performance.now();
      const selected = await selectEntry({ op: "route", prompt, ...tickerField, ...watchlistField, ...(currentChartTicker ? { currentChartTicker } : {}), ...contextField }, req.signal);
      timings.routeMs = performance.now() - routeAt;
      if ("error" in selected) {
        const routeError = selected.error;
        // Selection is unknown. The route must not guess personas or a route.
        send({ type: "agent_start", prompt });
        send({ type: "failed", category: "provider_error", message: routeError.slice(0, 160) });
        send({ type: "error", message: routeError });
        controller.close();
        return;
      }
      const warnChart = (message: string): void => {
        send({ type: "answer_delta", text: `Chart update failed: ${message} I will continue with your original question.\n\n` });
      };
      if (selected.chartError) warnChart(selected.chartError);
      if (selected.chartTicker) {
        const focus = selected.chartTicker;
        let sessionId: string | undefined;
        try {
          if (Date.now() >= deadlineAt) throw new Error("The request deadline expired before the chart action.");
          if (req.signal.aborted) throw new Error("The request was cancelled before the chart action.");
          sessionId = newSessionId();
          const result = await invoke("set_chart_symbol", { ticker: focus }, sessionId, Math.max(deadlineAt - Date.now(), 1));
          if (!result.ok) throw new Error(result.error);
          const receipt = result.chartSelection;
          if (!receipt || typeof receipt !== "object" || Array.isArray(receipt) || !("result_type" in receipt) || receipt.result_type !== "chart_selection"
            || !("ticker" in receipt) || receipt.ticker !== focus) {
            throw new Error("The chart tool returned an invalid receipt.");
          }
          currentChartTicker = focus;
          if (!watchlist) tickerField = { ticker: focus };
          send({ type: "chart_update", ticker: focus });
        } catch (err) {
          warnChart(err instanceof Error ? err.message : String(err));
        } finally {
          if (sessionId) {
            try {
              await endSession(sessionId);
            } catch {
              // Cleanup does not change a successful chart selection.
            }
          }
        }
      }
      const { personas } = selected;
      // Explicit personas skip direct and single-tool answers: research owns their evidence lifecycle.
      if (personas === null && await tryEntryAnswer(selected.route)) {
        controller.close();
        return;
      }
      try {
        await runKernelAgent(prompt, send, { signal: req.signal, seedEvidence: singleShotEvidence, deadlineAt, ...(personas ? { personas } : {}), ...tickerField, ...(currentChartTicker ? { currentChartTicker } : {}), ...(watchlist ? { watchlist } : {}), ...contextField });
      } catch (err) {
        console.error(`[web] [agent-api] runKernelAgent error: ${err instanceof Error ? err.message : String(err)}`);
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
      }
      controller.close();
      console.log("[web] [agent-api] stream close");
    },
  });
  return new Response(stream, { headers: SSE_HEADERS });
}