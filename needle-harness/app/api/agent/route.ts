import { reason } from "@/lib/muse/client";
import { kernelRouter, runKernelAgent } from "@/lib/agent/kernel";
import { tools } from "@/lib/tools";
import { endSession, invoke, newSessionId } from "@/lib/tools/stockbot";
import { redactArgs, type AgentEvent, type Evidence, type FailureCategory } from "@/lib/agent/types";
function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}
export async function POST(req: Request): Promise<Response> {
  let prompt: unknown;
  try {
    ({ prompt } = await req.json());
  } catch {
    console.error("[web] [agent-api] invalid JSON");
    return Response.json({ error: "invalid JSON" }, { status: 400 });
  }
  if (typeof prompt !== "string" || !prompt.trim()) {
    console.error("[web] [agent-api] prompt required");
    return Response.json({ error: "prompt required" }, { status: 400 });
  }
  console.log(`[web] [agent-api] POST prompt_chars=${prompt.length}`);
  const stream = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      // STOCKBOT_DEBUG=1 streams unfiltered working logs (routing, tools, model).
      // Prod summarizes: generic working indicator, generic errors, Stockbot labels.
      const debug = ["1", "true", "yes"].includes((process.env.STOCKBOT_DEBUG ?? "").trim().toLowerCase());
      let terminalSent = false;
      const send = (e: AgentEvent) => {
        if (req.signal.aborted) return;
        let out: AgentEvent | null = e;
        if (!debug) {
          switch (e.type) {
            case "needle_decision":
            case "tool_start":
            case "tool_result":
            case "tool_failed":
              return;
            case "progress":
              out = { type: "progress", stage: "working" };
              break;
            case "reasoning_start":
              out = { type: "reasoning_start", model: "stockbot" };
              break;
            case "failed":
            case "error":
              if (terminalSent) return;
              terminalSent = true;
              out = { type: "error", message: "Stockbot couldn't complete that request. Try again." };
              break;
          }
        }
        try {
          controller.enqueue(enc.encode(`data: ${JSON.stringify(out)}\n\n`));
        } catch {
          // Client disconnected mid-stream; remaining sends no-op, close below ends it.
        }
      };
      // ponytail: JEV owns routing; this answers direct with zero session/DB.
      const answerDirect = async (): Promise<boolean> => {
        const t0 = performance.now();
        send({ type: "agent_start", prompt });
        send({ type: "reasoning_start", model: requiredEnv("OPENCODE_MODEL") });
        try {
          const r = await reason({ prompt, evidence: [], escalated: false, direct: true, onDelta: (text) => send({ type: "answer_delta", text }) });
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
      // ponytail: JEV selects; shared kernel Needle fills args; every post-tool verdict returns to JEV (max 3 rounds).
      const answerSingleShot = async (toolName: string): Promise<boolean> => {
        const t0 = performance.now();
        const toolT0 = performance.now();
        send({ type: "agent_start", prompt });
        const evidence: Evidence[] = [];
        const failures: Partial<Record<FailureCategory, number>> = {};
        let needleCalls = 0;
        let toolCalls = 0;
        let current: string | null = toolName;
        for (let step = 0; step < 3 && current; step++) {
          const tool = current;
          current = null;
          let args: Record<string, unknown>;
          let confidence: number | null = null;
          try {
            const gen = await kernelRouter.call({ op: "arguments", tool, objective: prompt }, { signal: req.signal });
            if (typeof gen.error === "string" && gen.error) return false;
            if (gen.tool !== tool || typeof gen.arguments !== "object" || gen.arguments === null) return false;
            args = gen.arguments as Record<string, unknown>;
            confidence = typeof gen.confidence === "number" ? gen.confidence : null;
          } catch {
            // Shared-Needle failure — never fabricate args; fall through to research.
            return false;
          }
          needleCalls += 1;
          send({ type: "needle_decision", step, tool, arguments: redactArgs(args), confidence });
          send({ type: "tool_start", tool });
          const sessionId = newSessionId();
          let outcome: { ok: boolean; content?: string; error?: string; category?: string };
          try {
            const entry = tools[tool];
            const res = entry?.execute ? await entry.execute(args, { sessionId }) : await invoke(tool, args, sessionId);
            toolCalls += 1;
            if (res.ok) {
              evidence.push(res.evidence);
              singleShotEvidence.push(res.evidence);
              send({ type: "tool_result", tool, evidenceId: res.evidence.id, preview: res.evidence.content.slice(0, 160) });
              outcome = { ok: true, content: res.evidence.content.slice(0, 1500) };
            } else {
              failures[res.category] = (failures[res.category] ?? 0) + 1;
              send({ type: "tool_result", tool, preview: res.error.slice(0, 160) });
              send({ type: "tool_failed", tool, category: res.category, preview: res.error.slice(0, 160) });
              outcome = { ok: false, error: res.error.slice(0, 500), category: res.category };
            }
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            failures.tool_error = (failures.tool_error ?? 0) + 1;
            send({ type: "tool_failed", tool, category: "tool_error", preview: msg.slice(0, 160) });
            outcome = { ok: false, error: msg.slice(0, 500), category: "tool_error" };
          } finally {
            await endSession(sessionId);
          }
          let verdict: string;
          try {
            const assessed = await kernelRouter.call(
              { op: "assess_entry", prompt, tool, arguments: args, result: outcome },
              { signal: req.signal },
            );
            verdict = typeof assessed.verdict === "string" && assessed.verdict ? assessed.verdict : "research_required";
          } catch {
            return false;
          }
          if (verdict === "node_resolved" || verdict === "reasoning_required") {
            send({ type: "reasoning_start", model: requiredEnv("OPENCODE_MODEL") });
            try {
              const m0 = performance.now();
              const r = await reason({ prompt, evidence, escalated: false, direct: evidence.length === 0, onDelta: (text) => send({ type: "answer_delta", text }) });
              send({ type: "done", metrics: { totalMs: performance.now() - t0, needle: { calls: needleCalls, totalMs: 0, escalations: 0 }, tools: { calls: toolCalls, totalMs: performance.now() - toolT0 }, muse: { calls: 1, totalMs: performance.now() - m0, ...r.usage }, evidence: { count: evidence.length, characters: evidence.reduce((n, e) => n + e.content.length, 0) }, failures } });
            } catch (err) {
              send({ type: "error", message: err instanceof Error ? err.message : String(err) });
            }
            return true;
          }
          if (verdict === "research_required") return false;
          // ponytail: worker output is untrusted — only identifier-shaped tool names chain; anything else researches.
          current = /^[A-Za-z_][A-Za-z0-9_]*$/.test(verdict) ? verdict : null;
          if (!current) return false;
        }
        // Cap reached with another tool pending — research owns the longer chain.
        return false;
      };
      try {
        // ponytail: JEV-first entry — one route round; reason/research/tool winners branch here.
        const routed = await kernelRouter.call({ op: "route", prompt }, { signal: req.signal });
        const winner = typeof routed.route === "string" ? routed.route : "research_required";
        if (winner === "no_session") {
          await answerDirect();
          controller.close();
          return;
        }
        if (winner === "reasoning_required") {
          await answerDirect();
          controller.close();
          return;
        }
        // ponytail: worker output is untrusted — only identifier-shaped names single-shot; anything else researches.
        if (winner !== "research_required" && /^[A-Za-z_][A-Za-z0-9_]*$/.test(winner)) {
          if (await answerSingleShot(winner)) {
            controller.close();
            return;
          }
        }
      } catch {
        // Route unavailable — fail open to research below.
      }
      try {
        await runKernelAgent(prompt, send, { signal: req.signal, seedEvidence: singleShotEvidence });
      } catch (err) {
        console.error(`[web] [agent-api] runKernelAgent error: ${err instanceof Error ? err.message : String(err)}`);
        send({ type: "error", message: err instanceof Error ? err.message : String(err) });
      }
      controller.close();
      console.log("[web] [agent-api] stream close");
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" },
  });
}
