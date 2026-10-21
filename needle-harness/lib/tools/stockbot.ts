import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { makeEvidence } from "../agent/evidence";
import { parseTicker } from "../agent/kernel";
import { sourceReferences, WIDGET_TYPES } from "../agent/types";
import type { FailureCategory, ToolResult } from "../agent/types";
const ROOT = process.cwd().endsWith("needle-harness")
  ? process.cwd().replace(/\/needle-harness$/, "")
  : process.cwd();
const BRIDGE_CMD = `${ROOT}/venv/bin/python`;
const BRIDGE_ARGS = [`${ROOT}/scripts/tool_bridge.py`];

type BridgeResultMeta = { source_handle?: unknown; source_refs?: unknown; sources?: unknown; chart_selection?: unknown; watchlist_update?: unknown; widget_selection?: unknown };
type BridgeReply = {
  id?: unknown;
  result?: { content?: unknown; error?: unknown; error_type?: unknown; meta?: BridgeResultMeta };
  error?: unknown;
};

type Pending = {
  resolve: (v: BridgeReply) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

// Persistent runtime-neutral tool bridge; replies correlate by id.
class StockbotBridge {
  private child: ChildProcessWithoutNullStreams | null = null;
  private buf = "";
  private nextId = 0;
  private pending = new Map<string, Pending>();
  private bridgeStderrTail = "";

  private spawnChild(): ChildProcessWithoutNullStreams {
    const child = spawn(BRIDGE_CMD, BRIDGE_ARGS, { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
    this.child = child;
    this.buf = "";
    child.stdout.on("data", (chunk: Buffer) => this.onData(chunk.toString("utf8")));
    child.stderr.on("data", (chunk: Buffer) => {
      this.bridgeStderrTail = (this.bridgeStderrTail + chunk.toString("utf8")).slice(-2000);
    });
    child.on("exit", () => {
      if (this.child === child) this.child = null;
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error(`stockbot bridge exited; stderr tail: ${this.bridgeStderrTail || "(empty)"}`));
      }
      this.pending.clear();
    });
    return child;
  }

  private ensure(): ChildProcessWithoutNullStreams {
    if (this.child && this.child.exitCode === null) return this.child;
    return this.spawnChild();
  }

  private onData(chunk: string): void {
    this.buf += chunk;
    let i: number;
    while ((i = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, i).trim();
      this.buf = this.buf.slice(i + 1);
      if (!line) continue;
      let msg: unknown;
      try {
        msg = JSON.parse(line) as unknown;
      } catch {
        continue;
      }
      if (!msg || typeof msg !== "object" || !("id" in msg)) continue;
      const reply = msg as BridgeReply;
      if (typeof reply.id !== "string") continue;
      const p = this.pending.get(reply.id);
      if (!p) continue;
      this.pending.delete(reply.id);
      clearTimeout(p.timer);
      p.resolve(reply);
    }
  }

  call(req: Record<string, unknown>, timeoutMs = 120_000): Promise<BridgeReply> {
    const child = this.ensure();
    const id = `sb-${(this.nextId += 1)}`;
    return new Promise<BridgeReply>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`stockbot bridge timeout; stderr tail: ${this.bridgeStderrTail || "(empty)"}`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(JSON.stringify({ id, ...req }) + "\n", (err) => {
        if (err) {
          const p = this.pending.get(id);
          if (p) {
            this.pending.delete(id);
            clearTimeout(p.timer);
            p.reject(err);
          }
        }
      });
    });
  }

  close(): void {
    const child = this.child;
    this.child = null;
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("stockbot bridge closed"));
    }
    this.pending.clear();
    try {
      child?.kill();
    } catch {
      // Already gone; nothing to kill.
    }
  }
}

const bridge = new StockbotBridge();

type BridgeResult = NonNullable<BridgeReply["result"]>;
type Failure = { category: FailureCategory; retryable: boolean };
// Bridge error_type wins when known; otherwise the ordered message rules decide.
const ERROR_TYPE_CATEGORIES: Record<string, FailureCategory> = {
  deadline_exceeded: "deadline_exceeded",
  run_budget_exceeded: "tool_budget_exhausted",
  evidence_budget_exceeded: "token_budget_exhausted",
  invalid_research_context: "policy_denied",
};

// Ordered keyword rules over verbatim kernel/gateway text: the first match wins.
// Deadline and timeout stay ahead of the duplicate/provider/budget wording a
// timed-out call often carries, because the timeout is the cause. `scope` admits
// a rule only for messages that also name a budget or quota. Add a rule for a new
// distinct refusal; never collapse two refusals into one.
const MESSAGE_RULES: readonly {
  traps: readonly string[];
  scope?: readonly string[];
  category: FailureCategory;
  retryable: boolean;
}[] = [
    // Same rule as the kernel worker's _source_failure (app/research/kernel_worker.py): a source's
    // "No data found for ..." answer is no_records, whatever else the message names.
    { traps: ["no data found for "], category: "no_records", retryable: false },
    { traps: ["deadline"], category: "deadline_exceeded", retryable: false },
    { traps: ["timeout", "timed out", "timed_out", "expired"], category: "timeout", retryable: true },
    { traps: ["research_loop_detected", "repeats an action", "already ran"], category: "research_loop_detected", retryable: false },
    { traps: ["duplicate"], category: "duplicate_research_action", retryable: false },
    { traps: ["token"], scope: ["budget", "quota"], category: "token_budget_exhausted", retryable: false },
    { traps: ["job"], scope: ["budget", "quota"], category: "job_budget_exhausted", retryable: false },
    { traps: ["wave"], scope: ["budget", "quota"], category: "wave_budget_exhausted", retryable: false },
    { traps: ["budget", "quota"], category: "tool_budget_exhausted", retryable: false },
    {
      traps: ["policy", "denied", "not permitted", "not authorized", "blocked", "intent", "private", "egress", "withheld"],
      category: "policy_denied",
      retryable: false,
    },
    { traps: ["provider"], category: "provider_error", retryable: true },
  ];

function ruleMatches(rule: (typeof MESSAGE_RULES)[number], low: string): boolean {
  if (!rule.traps.some((trap) => low.includes(trap))) return false;
  return rule.scope === undefined || rule.scope.some((trap) => low.includes(trap));
}

export function categorizeFailure(message: string, errorType?: unknown): Failure {
  const typed = typeof errorType === "string" && Object.hasOwn(ERROR_TYPE_CATEGORIES, errorType)
    ? ERROR_TYPE_CATEGORIES[errorType]
    : undefined;
  if (typed !== undefined) return { category: typed, retryable: false };
  const low = message.toLowerCase();
  const rule = MESSAGE_RULES.find((candidate) => ruleMatches(candidate, low));
  return rule === undefined
    ? { category: "tool_error", retryable: false }
    : { category: rule.category, retryable: rule.retryable };
}

// A failed reply: the tool's own error wins over the bridge envelope error.
function resultFailure(result: BridgeReply["result"]): { error: string; errorType: unknown } | undefined {
  if (!result || typeof result !== "object") return undefined;
  if (typeof result.error !== "string" || !result.error) return undefined;
  return { error: result.error, errorType: result.error_type };
}

function envelopeFailure(error: unknown): { error: string } | undefined {
  if (error === undefined || error === null || error === "") return undefined;
  return { error: typeof error === "string" ? error : JSON.stringify(error) };
}

function failed(failure: { error: string; errorType?: unknown }): ToolResult {
  return { ok: false, error: failure.error, ...categorizeFailure(failure.error, failure.errorType) };
}

function replyContent(result: BridgeResult): string {
  if ("content" in result && typeof result.content === "string") return result.content;
  return JSON.stringify("content" in result ? result.content ?? result : result).slice(0, 32768);
}

// Only a captured handle or refs map survives; anything else is dropped, never repaired.
function replyMeta(meta: BridgeResult["meta"]): {
  sourceHandle?: Record<string, unknown>;
  sourceRefs?: Record<string, unknown>;
  sources: unknown;
} {
  const handle = meta?.source_handle;
  const refs = meta?.source_refs;
  return {
    sourceHandle: handle !== null && typeof handle === "object" && !Array.isArray(handle) ? handle as Record<string, unknown> : undefined,
    sourceRefs: refs !== null && typeof refs === "object" && !Array.isArray(refs) ? refs as Record<string, unknown> : undefined,
    sources: meta?.sources,
  };
}

function chartSelection(meta: BridgeResultMeta | undefined): Record<string, unknown> | undefined {
  const selection = meta?.chart_selection;
  if (!selection || typeof selection !== "object" || Array.isArray(selection)) return undefined;
  const candidate = selection as Record<string, unknown>;
  if (Object.keys(candidate).length !== 2 || candidate.result_type !== "chart_selection") return undefined;
  if (typeof candidate.ticker !== "string" || parseTicker(candidate.ticker) !== candidate.ticker) return undefined;
  return candidate;
}

function watchlistUpdate(meta: BridgeResultMeta | undefined): Record<string, unknown> | undefined {
  const update = meta?.watchlist_update;
  if (!update || typeof update !== "object" || Array.isArray(update)) return undefined;
  const candidate = update as Record<string, unknown>;
  if (Object.keys(candidate).length !== 3 || candidate.result_type !== "watchlist_update" || (candidate.action !== "add" && candidate.action !== "remove")) return undefined;
  if (typeof candidate.ticker !== "string" || parseTicker(candidate.ticker) !== candidate.ticker) return undefined;
  return candidate;
}

function widgetSelection(meta: BridgeResultMeta | undefined): Record<string, unknown> | undefined {
  const selection = meta?.widget_selection;
  if (!selection || typeof selection !== "object" || Array.isArray(selection)) return undefined;
  const candidate = selection as Record<string, unknown>;
  if (Object.keys(candidate).length !== 3 || candidate.result_type !== "widget_selection" || typeof candidate.widget_type !== "string" || !WIDGET_TYPES.includes(candidate.widget_type)) return undefined;
  if (typeof candidate.ticker !== "string" || parseTicker(candidate.ticker) !== candidate.ticker) return undefined;
  return candidate;
}

function succeeded(name: string, result: BridgeResult): ToolResult {
  const { sourceHandle, sourceRefs, sources } = replyMeta(result.meta);
  const evidence = makeEvidence(name, replyContent(result), {
    title: name,
    ...(sourceHandle !== undefined ? { sourceHandle } : {}),
    ...(sourceRefs !== undefined ? { sourceRefs } : {}),
  });
  const refs = sourceReferences(sources);
  if (refs.length) evidence.sources = refs;
  const selection = name === "set_chart_symbol" ? chartSelection(result.meta) : undefined;
  const update = name === "update_watchlist" ? watchlistUpdate(result.meta) : undefined;
  const widget = name === "add_widget" ? widgetSelection(result.meta) : undefined;
  return {
    ok: true,
    evidence,
    ...(selection !== undefined ? { chartSelection: selection } : {}),
    ...(update !== undefined ? { watchlistUpdate: update } : {}),
    ...(widget !== undefined ? { widgetSelection: widget } : {}),
  };
}

export async function invoke(name: string, args: Record<string, unknown>, sessionId: string, timeoutMs?: number): Promise<ToolResult> {
  let msg: BridgeReply;
  try {
    msg = await bridge.call({ op: "tool.invoke", name, arguments: args, session_id: sessionId }, timeoutMs);
  } catch (err) {
    return failed({ error: err instanceof Error ? err.message : String(err) });
  }
  const failure = resultFailure(msg.result) ?? envelopeFailure(msg.error);
  return failure === undefined ? succeeded(name, msg.result ?? {}) : failed(failure);
}

export async function endSession(sessionId: string): Promise<void> {
  try {
    await bridge.call({ op: "tool.session.end", session_id: sessionId }, 10_000);
  } catch {
    // Best-effort; loop finally awaits directly and must never throw.
  }
}

export function closeBridge(): void {
  bridge.close();
}

export function newSessionId(): string {
  return randomUUID();
}
