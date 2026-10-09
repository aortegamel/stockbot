import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { makeEvidence } from "../agent/evidence";
import type { FailureCategory, ToolResult } from "../agent/types";

const ROOT = process.cwd().endsWith("needle-harness")
  ? process.cwd().replace(/\/needle-harness$/, "")
  : process.cwd();
const BRIDGE_CMD = `${ROOT}/venv/bin/python`;
const BRIDGE_ARGS = [`${ROOT}/scripts/tool_bridge.py`];

type BridgeResultMeta = { source_handle?: unknown; source_refs?: unknown };
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
// ponytail: keyword ladder over verbatim kernel/gateway text; bridge error_type wins when known. Add branches for new distinct refusals, never collapse to one string.
// Precedence: explicit error_type, then explicit deadline/timeout signals, then substring traps
// (duplicate/provider/budget/policy). Deadline/timeout stay first because a timed-out call is
// often re-reported with "duplicate"/"budget"/"provider" wording attached — the timeout is the cause.
export function categorizeFailure(message: string, errorType?: unknown): { category: FailureCategory; retryable: boolean } {
  if (errorType === "deadline_exceeded") return { category: "deadline_exceeded", retryable: false };
  if (errorType === "run_budget_exceeded") return { category: "tool_budget_exhausted", retryable: false };
  if (errorType === "evidence_budget_exceeded") return { category: "token_budget_exhausted", retryable: false };
  if (errorType === "invalid_research_context") return { category: "policy_denied", retryable: false };
  const low = message.toLowerCase();
  if (low.includes("deadline")) return { category: "deadline_exceeded", retryable: false };
  if (low.includes("timeout") || low.includes("timed out") || low.includes("timed_out") || low.includes("expired"))
    return { category: "timeout", retryable: true };
  if (low.includes("research_loop_detected") || low.includes("repeats an action") || low.includes("already ran"))
    return { category: "research_loop_detected", retryable: false };
  if (low.includes("duplicate")) return { category: "duplicate_research_action", retryable: false };
  if (low.includes("budget") || low.includes("quota")) {
    if (low.includes("token")) return { category: "token_budget_exhausted", retryable: false };
    if (low.includes("job")) return { category: "job_budget_exhausted", retryable: false };
    if (low.includes("wave")) return { category: "wave_budget_exhausted", retryable: false };
    return { category: "tool_budget_exhausted", retryable: false };
  }
  if (
    low.includes("policy") ||
    low.includes("denied") ||
    low.includes("not permitted") ||
    low.includes("not authorized") ||
    low.includes("blocked") ||
    low.includes("intent") ||
    low.includes("private") ||
    low.includes("egress") ||
    low.includes("withheld")
  )
    return { category: "policy_denied", retryable: false };
  if (low.includes("provider")) return { category: "provider_error", retryable: true };
  return { category: "tool_error", retryable: false };
}

export async function invoke(name: string, args: Record<string, unknown>, sessionId: string): Promise<ToolResult> {
  let msg: BridgeReply;
  try {
    msg = await bridge.call({ op: "tool.invoke", name, arguments: args, session_id: sessionId });
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    return { ok: false, error, ...categorizeFailure(error) };
  }
  const errType = msg.result != null && typeof msg.result === "object" ? msg.result.error_type : undefined;
  if (msg.result && typeof msg.result === "object" && typeof msg.result.error === "string" && msg.result.error) {
    const error = msg.result.error;
    return { ok: false, error, ...categorizeFailure(error, errType) };
  }
  if (msg.error !== undefined && msg.error !== null && msg.error !== "") {
    const error = typeof msg.error === "string" ? msg.error : JSON.stringify(msg.error);
    return { ok: false, error, ...categorizeFailure(error) };
  }
  const result = msg.result ?? {};
  const content =
    "content" in result && typeof result.content === "string"
      ? result.content
      : JSON.stringify("content" in result ? result.content ?? result : result).slice(0, 32768);
  const meta = result.meta;
  const rawHandle = meta != null && typeof meta === "object" && "source_handle" in meta ? (meta as BridgeResultMeta).source_handle : undefined;
  const rawRefs = meta != null && typeof meta === "object" && "source_refs" in meta ? (meta as BridgeResultMeta).source_refs : undefined;
  const sourceHandle = rawHandle !== null && typeof rawHandle === "object" && !Array.isArray(rawHandle) ? (rawHandle as Record<string, unknown>) : undefined;
  const sourceRefs = rawRefs !== null && typeof rawRefs === "object" && !Array.isArray(rawRefs) ? (rawRefs as Record<string, unknown>) : undefined;
  const evidence = makeEvidence(name, content, {
    title: name,
    ...(sourceHandle !== undefined ? { sourceHandle } : {}),
    ...(sourceRefs !== undefined ? { sourceRefs } : {}),
  });
  return { ok: true, evidence };
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
