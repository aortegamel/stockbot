import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { GENERATE_TIMEOUT_MS } from "../needle/client";
import { reason, type MuseUsage } from "../muse/client";
import { FINAL_PERSONAS, redactArgs } from "./types";
import type { AgentEvent, Evidence, FailureCategory, Metrics, Persona } from "./types";

// ponytail: persistent worker bridge (spawn-once, ready-gated); spawn/exit/stderr handling mirrors lib/needle/client.ts.
function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}
const WORKER_TIMEOUT_MS = 10 * 60 * 1000;
// whole user request, both passes, from route entry.
export const REQUEST_WALL_MS = 120_000;
// ponytail: worker finalize/muse-write reserve (20s, mirrors kernel_worker _FINALIZE_S) plus one Needle generation.
const PASS2_MIN_MS = 20_000 + GENERATE_TIMEOUT_MS;
const PREWARM_TIMEOUT_MS = 30_000;

// next dev runs with cwd=needle-harness; the repo root is its parent.
const ROOT = process.env.STOCKBOT_REPO_ROOT ?? join(process.cwd(), "..");
const WORKER = join(ROOT, "app/research/kernel_worker.py");
const VENV_PYTHON = `${homedir()}/.cache/needle-harness/.needle/bin/python`;
// Stockbot kernel needs repo deps (dotenv, app.*); the needle-only venv lacks them.
const REPO_PYTHON = join(ROOT, "venv/bin/python");

export type KernelEvidence = {
  id: string;
  content: string;
  source?: string;
  title?: string;
  url?: string;
  retrievedAt?: string;
};

export type KernelNeedleDecision = {
  step: number;
  tool: string | null;
  arguments: Record<string, unknown>;
  confidence: number | null;
  reasoning?: string;
};

export type KernelToolCall = {
  tool: string;
  ok: boolean;
  evidenceId?: string;
  error?: string;
  category?: FailureCategory;
};

export type KernelGraphNode = {
  node_id: string;
  question: string;
  status: string;
  depends_on: string[];
};

export type KernelReasonInput = {
  prompt: string;
  evidence: Evidence[];
  escalated: boolean;
  direct?: boolean;
  objective: string;
  nodes: KernelGraphNode[];
  decisions: Record<string, unknown>[];
  unresolved: string[];
  incompleteGuard: boolean;
  persona: Persona;

  // Synthesis only: sibling drafts for the Stockbot judge (trio only); omitted elsewhere.
  drafts?: { persona: Persona; text: string }[];
  signal?: AbortSignal;
  deadlineAt: number;
  onDelta: (text: string) => void;
};

export type KernelResponse = {
  id?: string;
  objective?: string;
  sessionId?: string;
  evidence: KernelEvidence[];
  nodes?: KernelGraphNode[];
  decisions?: Record<string, unknown>[];
  unresolved?: string[];
  incompleteGuard?: boolean;
  incomplete_guard?: boolean;
  toolExecutions?: KernelNeedleDecision[];
  needleDecisions: KernelNeedleDecision[];
  toolCalls: KernelToolCall[];
  failures: Record<string, number>;
  escalations: number;
  escalated: boolean;
  route?: string;
  personas?: Persona[] | null;
  tool?: string;
  arguments?: Record<string, unknown>;
  confidence?: number | null;
  reasoning?: string;
  verdict?: string;
  error?: string;
  terminal?: { category: FailureCategory; message: string };
};

export type KernelChild = {
  stdin: { write: (data: string, cb?: (err?: Error | null) => void) => void } | null;
  stdout: { on: (event: "data", listener: (chunk: Buffer) => void) => void } | null;
  stderr: { on: (event: "data", listener: (chunk: Buffer) => void) => void } | null;
  on: (event: "error" | "exit", listener: (arg?: unknown) => void) => void;
  kill: (signal?: string) => void;
  readonly exitCode: number | null;
};

export type KernelSpawn = (cmd: string, args: string[], opts: { env: NodeJS.ProcessEnv }) => KernelChild;

export type KernelReasonResult = { text: string; usage: MuseUsage; missingEvidence?: string };
export type KernelReasonFn = (opts: KernelReasonInput) => Promise<KernelReasonResult>;
export type RunKernelDeps = {
  spawnFn?: KernelSpawn;
  reason?: KernelReasonFn;
  python?: string;
  workerPath?: string;
};

function defaultSpawn(cmd: string, args: string[], opts: { env: NodeJS.ProcessEnv }): KernelChild {
  return spawn(cmd, args, { stdio: ["pipe", "pipe", "pipe"], env: opts.env }) as unknown as KernelChild;
}

type KernelPending = {
  resolve: (v: KernelResponse) => void;
  reject: (e: Error) => void;
  cancel: () => void;
  onProgress?: (stage: string, detail?: Record<string, unknown>) => void;
};
export class KernelRouter {
  private child: KernelChild | null = null;
  private pending: Record<string, KernelPending> = {};
  private buf = "";
  private nextId = 0;
  private exited = false;
  private workerStderrTail = "";
  private ready: Promise<void> = Promise.resolve();
  private resolveReady: () => void = () => { };
  private rejectReady: (err: Error) => void = () => { };
  private readonly python: string;
  private readonly workerPath: string;
  private readonly spawnFn: KernelSpawn;

  constructor(opts?: { python?: string; workerPath?: string; spawnFn?: KernelSpawn }) {
    this.python = opts?.python ?? (existsSync(REPO_PYTHON) ? REPO_PYTHON : existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3");
    this.workerPath = opts?.workerPath ?? WORKER;
    this.spawnFn = opts?.spawnFn ?? defaultSpawn;
  }

  private spawn(): KernelChild {
    this.exited = false;
    this.buf = "";
    this.workerStderrTail = "";
    const child = this.spawnFn(this.python, [this.workerPath], {
      env: { ...process.env, NEEDLE_TELEMETRY: "0", DO_NOT_TRACK: "1" },
    });
    this.child = child;
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    this.ready = promise;
    this.resolveReady = resolve;
    this.rejectReady = reject;
    void promise.catch(() => {
      // Exit/timeout rejections are observed via prewarm()/call(); a spawn
      // with no waiter yet must not surface as unhandled.
    });
    child.stdout?.on("data", (d: Buffer) => this.onData(d.toString()));
    child.stderr?.on("data", (d: Buffer) => {
      const s = d.toString();
      this.workerStderrTail = (this.workerStderrTail + s).slice(-2000);
      process.stderr.write(s.startsWith("[kernel-worker]") ? s : `[kernel-worker] ${s}`);
    });
    child.on("error", (err) => {
      const e = err instanceof Error ? err : new Error(String(err));
      this.rejectReady(e);
      this.failAll(e);
    });
    child.on("exit", () => {
      this.exited = true;
      const err = new Error(`kernel worker exited; stderr tail: ${this.workerStderrTail || "(empty)"}`);
      if (Object.keys(this.pending).length > 0) {
        this.failAll(err);
      }
      // Worker died before hello: reject the ready gate so prewarm()/call()
      // fail fast instead of hanging; no-op once ready already resolved.
      this.rejectReady(err);
      if (this.child === child) this.child = null;
    });
    console.error(`[ai] kernel worker spawn ${this.python} (${this.workerPath})`);
    return child;
  }

  ensure(): KernelChild {
    // Public for prewarm(); call() also routes through here.
    const running = this.child;
    if (running && !this.exited && running.exitCode === null) return running;
    try {
      running?.kill();
    } catch {
      // Already gone; fresh spawn below replaces it.
    }
    const fresh = this.spawn();
    if (!fresh.stdin || !fresh.stdout) {
      this.child = null;
      throw new Error("kernel worker spawn failed");
    }
    return fresh;
  }

  private onLine(line: string): void {
    let msg: unknown;
    try {
      msg = JSON.parse(line);
    } catch {
      // Non-JSON bridge noise; keep scanning.
    }
    if (typeof msg !== "object" || msg === null) return;
    const rec = msg as Record<string, unknown>;
    if (typeof rec.id !== "string") {
      // Ready gate: worker hello, never a response.
      if (rec.type === "ready") this.resolveReady();
      return;
    }
    if (rec.type === "progress") {
      const p = this.pending[rec.id];
      const stage = typeof rec.stage === "string" ? rec.stage : "working";
      const detail = rec.detail && typeof rec.detail === "object" ? (rec.detail as Record<string, unknown>) : undefined;
      try {
        p?.onProgress?.(stage, detail);
      } catch {
        // A dead progress listener never fails the run.
      }
      return;
    }
    const p = this.pending[rec.id];
    if (!p) return;
    delete this.pending[rec.id];
    p.cancel();
    // boundary: JSONL line from our own worker; every field is defaulted where read below.
    p.resolve(msg as KernelResponse);
  }

  private onData(chunk: string): void {
    this.buf += chunk;
    let i = this.buf.indexOf("\n");
    while (i >= 0) {
      const line = this.buf.slice(0, i).trim();
      this.buf = this.buf.slice(i + 1);
      if (line.length > 0) this.onLine(line);
      i = this.buf.indexOf("\n");
    }
  }

  private failAll(err: Error): void {
    for (const id of Object.keys(this.pending)) {
      const p = this.pending[id];
      if (p) {
        delete this.pending[id];
        p.cancel();
        p.reject(err);
      }
    }
  }

  call(body: Record<string, unknown>, opts?: { signal?: AbortSignal; timeoutMs?: number; onProgress?: (stage: string, detail?: Record<string, unknown>) => void }): Promise<KernelResponse> {
    const timeoutMs = opts?.timeoutMs ?? WORKER_TIMEOUT_MS;
    const child = this.ensure();
    if (!child.stdin) throw new Error("kernel worker spawn failed");
    const id = String((this.nextId += 1));
    return new Promise<KernelResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        const p = this.pending[id];
        if (!p) return;
        delete this.pending[id];
        p.cancel();
        p.reject(new Error(`kernel worker timeout; stderr tail: ${this.workerStderrTail || "(empty)"}`));
      }, timeoutMs);
      const cancel = (): void => {
        clearTimeout(timer);
        opts?.signal?.removeEventListener("abort", onAbort);
      };
      const onAbort = (): void => {
        const p = this.pending[id];
        if (!p) return;
        delete this.pending[id];
        p.cancel();
        p.reject(new Error("worker aborted"));
      };
      this.pending[id] = { resolve, reject, cancel, ...(opts?.onProgress ? { onProgress: opts.onProgress } : {}) };
      if (opts?.signal?.aborted) {
        onAbort();
        return;
      }
      opts?.signal?.addEventListener("abort", onAbort, { once: true });
      void this.ready.then(
        () => {
          if (!this.pending[id]) return;
          child.stdin?.write(`${JSON.stringify({ ...body, id })}\n`, (err) => {
            if (!err) return;
            const p = this.pending[id];
            if (!p) return;
            delete this.pending[id];
            p.cancel();
            p.reject(err instanceof Error ? err : new Error(String(err)));
          });
        },
        (err: unknown) => {
          // Worker exited before hello: fail fast instead of hanging until timeoutMs.
          const p = this.pending[id];
          if (!p) return;
          delete this.pending[id];
          p.cancel();
          p.reject(err instanceof Error ? err : new Error(String(err)));
        },
      );
    });
  }

  close(): void {
    const child = this.child;
    this.child = null;
    this.failAll(new Error("kernel router closed"));
    // Ready reject is a no-op once resolved; failAll above carries the error.
    this.rejectReady(new Error("kernel router closed"));
    if (!child || child.exitCode !== null) return;
    try {
      child.kill();
    } catch {
      // Already gone; failAll above carries the error.
    }
  }

  async prewarm(timeoutMs = PREWARM_TIMEOUT_MS): Promise<void> {
    this.ensure();
    let timer: NodeJS.Timeout | undefined;
    try {
      await Promise.race([
        this.ready,
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => {
            try {
              this.child?.kill();
            } catch {
              // Already gone; rejection below carries the error.
            }
            this.child = null;
            reject(new Error(`kernel worker prewarm timeout after ${timeoutMs}ms; stderr tail: ${this.workerStderrTail || "(empty)"}`));
          }, timeoutMs);
        }),
      ]);
      process.env.KERNEL_PREWARMED = "1";
    } finally {
      clearTimeout(timer);
    }
  }
}

const kernelRouter = new KernelRouter();

export { kernelRouter };

export async function prewarmKernel(): Promise<void> {
  await kernelRouter.prewarm();
}

for (const sig of ["SIGINT", "SIGTERM"] as const) {
  process.on(sig, () => kernelRouter.close());
}

function graphSection(title: string, lines: string[]): string {
  if (lines.length === 0) return `${title}: (none)`;
  return `${title}:\n${lines.map((l) => `- ${l}`).join("\n")}`;
}

// Graph projection for the final writer: Muse renders prose but cites only
// graph nodes/decisions/evidence; unresolved + incomplete_guard are explicit
// so the answer never masquerades a partial graph as complete.
function graphProjection(input: {
  objective: string;
  nodes: KernelGraphNode[];
  decisions: Record<string, unknown>[];
  unresolved: string[];
  incompleteGuard: boolean;
  escalated: boolean;
}): string {
  const nodeById: Record<string, KernelGraphNode> = {};
  for (const n of input.nodes) nodeById[n.node_id] = n;
  const nodeLines = input.nodes.map(
    (n) =>
      `[${n.node_id}] (${n.status}) q=${JSON.stringify(n.question)} deps=${n.depends_on.length ? n.depends_on.join(",") : "none"}`,
  );
  const decisionLines = input.decisions.map((d, i) => {
    const rec = d as Record<string, unknown>;
    const dtype = typeof rec.decision_type === "string" ? rec.decision_type : typeof rec.type === "string" ? rec.type : "decision";
    const dnode = typeof rec.node_id === "string" ? rec.node_id : typeof rec.nodeId === "string" ? rec.nodeId : "session";
    const sel = rec.selected !== undefined ? JSON.stringify(rec.selected) : JSON.stringify(rec);
    return `[D${i}] ${dtype} node=${dnode} selected=${sel.slice(0, 300)}`;
  });
  const unresolvedLines = input.unresolved.map((id) => {
    const n = nodeById[id];
    return n ? `[${id}] (${n.status}) ${n.question.slice(0, 200)}` : `[${id}] (unknown node)`;
  });
  const guard = input.incompleteGuard
    ? "INCOMPLETE: the tool-round guard tripped — coverage is partial, say what is missing, never present this as converged."
    : "Guard: not tripped — still, unresolved items below are open, never present them as answered.";
  return [
    `OBJECTIVE (verbatim user intent; never restate or change it): ${input.objective}`,
    graphSection("NODES", nodeLines),
    graphSection("JEV DECISIONS (persisted dispositions; the only authority)", decisionLines),
    graphSection("UNRESOLVED (explicitly open; cite as gaps, never answer from memory)", unresolvedLines),
    guard,
    "AUTHORITY: project this graph into prose only. Cite only node ids, decision records, and evidence ids above. No invented conclusions, no buy/sell/hold/order/portfolio/committee verdicts — research never decides, the user does.",
  ].join("\n\n");
}

export async function runKernelAgent(
  prompt: string,
  emit: (e: AgentEvent) => void,
  opts?: { signal?: AbortSignal; deadlineAt?: number; deps?: RunKernelDeps; seedEvidence?: Evidence[]; personas?: readonly Persona[] },
): Promise<void> {
  const t0 = performance.now();
  const deps = opts?.deps ?? {};
  const reasonFn = deps.reason ?? reason;
  const deadlineAt = opts?.deadlineAt ?? Date.now() + REQUEST_WALL_MS;
  const hasWorkerDeps = deps.spawnFn !== undefined || deps.python !== undefined || deps.workerPath !== undefined;
  const router = hasWorkerDeps
    ? new KernelRouter({
      ...(deps.python ? { python: deps.python } : {}),
      ...(deps.workerPath ? { workerPath: deps.workerPath } : {}),
      ...(deps.spawnFn ? { spawnFn: deps.spawnFn } : {}),
    })
    : kernelRouter;

  const buildMetrics = (
    decisions: number,
    calls: number,
    ev: Evidence[],
    workerMs: number,
    escalations: number,
    fl: Partial<Record<FailureCategory, number>>,
    muse: Metrics["muse"],
  ): Metrics => ({
    totalMs: performance.now() - t0,
    // ponytail: kernel reports no routing/execution split yet; worker wall time sits on tools, TS routing is 0.
    needle: { calls: decisions, totalMs: 0, escalations },
    tools: { calls, totalMs: workerMs },
    muse,
    evidence: { count: ev.length, characters: ev.reduce((n, e) => n + e.content.length, 0) },
    failures: fl,
  });

  emit({ type: "agent_start", prompt });

  // ponytail: entry routing lives in route.ts (op:route fast-path there);
  // runKernelAgent stays research-only so agent_start emits exactly once.
  let res: KernelResponse;
  const workerStart = performance.now();
  try {
    res = await router.call({ op: "run", prompt, deadlineAt }, { signal: opts?.signal, timeoutMs: Math.max(deadlineAt - Date.now(), 1), onProgress: (stage, detail) => emit(detail !== undefined ? { type: "progress", stage, detail } : { type: "progress", stage }) });
  } catch (err) {
    // ponytail: needle-down shape — worker failure still ends at done, never throws to the route.
    const msg = err instanceof Error ? err.message : String(err);
    emit({ type: "tool_failed", tool: "needle", category: "provider_error", preview: msg.slice(0, 160) });
    emit({ type: "failed", category: "provider_error", message: msg.slice(0, 160) });
    emit({ type: "done", metrics: buildMetrics(0, 0, [], 0, 1, { provider_error: 1 }, { calls: 0, totalMs: 0 }) });
    return;
  }
  const workerMs = performance.now() - workerStart;
  // Canonical toolExecutions; legacy needleDecisions alias kept one release for compat.
  const rawExec = Array.isArray(res.toolExecutions) ? res.toolExecutions : res.needleDecisions;
  const decisions = Array.isArray(rawExec) ? rawExec : [];
  const calls = Array.isArray(res.toolCalls) ? res.toolCalls : [];
  const nodes: KernelGraphNode[] = (Array.isArray(res.nodes) ? res.nodes : [])
    .filter((n) => n && typeof n.node_id === "string")
    .map((n) => ({
      node_id: String(n.node_id),
      question: String(n.question ?? ""),
      status: String(n.status ?? ""),
      depends_on: Array.isArray(n.depends_on) ? n.depends_on.map(String) : [],
    }));
  const persisted: Record<string, unknown>[] = Array.isArray(res.decisions) ? res.decisions : [];
  const unresolved: string[] = Array.isArray(res.unresolved) ? res.unresolved.map(String) : [];
  const incompleteGuard = res.incompleteGuard ?? res.incomplete_guard ?? false;
  const objective = typeof res.objective === "string" && res.objective ? res.objective : prompt;
  const seed = Array.isArray(opts?.seedEvidence) ? opts.seedEvidence : [];
  const evidence: Evidence[] = [...seed, ...(Array.isArray(res.evidence) ? res.evidence : []).map((e) => ({
    id: String(e.id),
    source: e.source ?? "kernel",
    ...(e.title !== undefined ? { title: e.title } : {}),
    ...(e.url !== undefined ? { url: e.url } : {}),
    retrievedAt: e.retrievedAt ?? new Date().toISOString(),
    content: String(e.content ?? ""),
  }))];
  const byId: Record<string, Evidence> = {};
  for (const e of evidence) byId[e.id] = e;
  const failures: Partial<Record<FailureCategory, number>> = {};
  if (res.failures && typeof res.failures === "object") {
    for (const [k, v] of Object.entries(res.failures)) if (typeof v === "number") failures[k as FailureCategory] = v;
  }
  const countFailure = (c: FailureCategory): void => {
    failures[c] = (failures[c] ?? 0) + 1;
  };

  if (typeof res.error === "string" && res.error) {
    countFailure("provider_error");
    emit({ type: "tool_failed", tool: "needle", category: "provider_error", preview: res.error.slice(0, 160) });
    emit({ type: "failed", category: "provider_error", message: res.error.slice(0, 160) });
    emit({ type: "done", metrics: buildMetrics(decisions.length, calls.length, evidence, workerMs, (res.escalations ?? 0) + 1, failures, { calls: 0, totalMs: 0 }) });
    return;
  }

  // ponytail: index-aligned replay — worker lists decisions and calls in execution order, so calls[i] belongs to decisions[i]; a trailing escalate decision has no call.
  const termCategory = res.terminal && typeof res.terminal === "object" ? res.terminal.category : undefined;
  let toolFailedEmitted = false;
  const emitCall = (c: KernelToolCall): void => {
    emit({ type: "tool_start", tool: c.tool });
    if (c.ok) {
      const ev = c.evidenceId ? byId[c.evidenceId] : undefined;
      emit({ type: "tool_result", tool: c.tool, evidenceId: c.evidenceId, preview: (ev?.content ?? "").slice(0, 160) });
    } else {
      const preview = (c.error ?? "tool failed").slice(0, 160);
      emit({ type: "tool_result", tool: c.tool, preview });
      emit({ type: "tool_failed", tool: c.tool, category: c.category ?? termCategory ?? "tool_error", preview });
      toolFailedEmitted = true;
    }
  };
  for (let i = 0; i < decisions.length; i++) {
    const d = decisions[i];
    emit({ type: "needle_decision", step: d.step ?? i, tool: d.tool ?? null, arguments: redactArgs(d.arguments ?? {}), confidence: d.confidence ?? null });
    if (i < calls.length) emitCall(calls[i]);
  }
  for (let i = decisions.length; i < calls.length; i++) emitCall(calls[i]);

  if (res.terminal && typeof res.terminal === "object") {
    const { category, message } = res.terminal;
    countFailure(category);
    if (!toolFailedEmitted) {
      emit({ type: "tool_failed", tool: calls.length > 0 ? calls[calls.length - 1].tool : "worker", category, preview: message.slice(0, 160) });
    }
    emit({ type: "failed", category, message: message.slice(0, 160) });
    emit({ type: "done", metrics: buildMetrics(decisions.length, calls.length, evidence, workerMs, (res.escalations ?? 0) + 1, failures, { calls: 0, totalMs: 0 }) });
    return;
  }

  // Fail-closed evidence loop: partial selections reason concurrently over one
  // shared snapshot; the full trio synthesizes (bear+bull draft concurrently,
  // then stockbot judges both drafts against the same snapshot). Drafts stay
  // buffered. Any Missing-Evidence line means the graph is partial — combine
  // the gaps, run one shared follow-up pass, merge, and regenerate per the
  // same rule. A second gap, or any failure after the trio's one draft retry,
  // fails the report; prose emits only when all succeed gap-free.
  const personas = opts?.personas ?? FINAL_PERSONAS;
  // ponytail: synthesis trigger is the exact full trio, nothing else.
  const isTrio =
    personas.length === FINAL_PERSONAS.length && FINAL_PERSONAS.every((p) => (personas as readonly string[]).includes(p));
  // ponytail: empty graph never closes silent — first pass answers direct.
  let direct = evidence.length === 0 && decisions.length === 0 && calls.length === 0 && unresolved.length === 0 && !incompleteGuard;
  let pass = 1;
  const mergedEvidence = evidence;
  let mergedNodes = nodes;
  let mergedDecisions = persisted;
  let mergedUnresolved = unresolved;
  let mergedGuard = incompleteGuard;
  let mergedEscalated = res.escalated ?? false;
  let totalWorkerMs = workerMs;
  let totalDecisions = decisions.length;
  let totalCalls = calls.length;
  let totalEscalations = res.escalations ?? 0;
  let museCalls = 0;
  let museMs = 0;
  const usage: MuseUsage = {};
  for (; ;) {
    emit({ type: "reasoning_start", model: requiredEnv("OPENCODE_MODEL") });
    const tm = performance.now();
    const input = {
      prompt: direct ? prompt : `${prompt}\n\n${graphProjection({ objective, nodes: mergedNodes, decisions: mergedDecisions, unresolved: mergedUnresolved, incompleteGuard: mergedGuard, escalated: mergedEscalated })}`,
      evidence: mergedEvidence,
      escalated: direct ? false : mergedEscalated,
      direct: direct || (mergedEscalated && mergedEvidence.length === 0),
      objective,
      nodes: mergedNodes,
      decisions: mergedDecisions,
      unresolved: mergedUnresolved,
      incompleteGuard: mergedGuard,
      deadlineAt,
      ...(opts?.signal ? { signal: opts.signal } : {}),
      onDelta: () => { },
    };
    const gaps: string[] = [];
    const noteGap = (g: string | undefined): void => {
      const t = g?.trim();
      if (t && !gaps.includes(t)) gaps.push(t);
    };
    const addUsage = (u: MuseUsage): void => {
      for (const k of ["inputTokens", "outputTokens", "cachedTokens"] as const) {
        const n = u[k];
        if (typeof n === "number") usage[k] = (usage[k] ?? 0) + n;
      }
    };
    let failure: string | undefined;
    let gap: string | undefined;
    let answerText: string | undefined;
    if (!isTrio) {
      museCalls += personas.length;
      // ponytail: every persona settles before the verdict, so one failure waits out its siblings; abort them early if final-stage latency matters.
      const results = await Promise.all(
        personas.map((persona) => reasonFn({ ...input, persona }).then((r) => ({ persona, r }), (err: unknown) => ({ persona, err }))),
      );
      museMs += performance.now() - tm;
      const texts: string[] = [];
      for (const x of results) {
        if ("err" in x) {
          failure ??= `${x.persona}: ${x.err instanceof Error ? x.err.message : String(x.err)}`;
          continue;
        }
        addUsage(x.r.usage);
        const text = x.r.text.trim();
        // Every selected persona must contribute prose; an empty draft fails the report.
        if (!text) {
          failure ??= `${x.persona}: empty response`;
          continue;
        }
        texts.push(`── ${x.persona.charAt(0).toUpperCase()}${x.persona.slice(1)} ──\n${text}`);
        noteGap(x.r.missingEvidence);
      }
      if (failure === undefined) {
        gap = gaps.length > 0 ? gaps.join("; ") : undefined;
        if (gap === undefined) answerText = texts.join("\n\n");
      }
    } else {
      // Trio synthesis: bear+bull draft concurrently over the shared snapshot,
      // then stockbot judges both drafts against the same snapshot. Exactly one
      // failed or empty draft retries once within the shared deadline/signal;
      // a second failure, a failed or empty synthesis, or any gap follows the
      // shared fail-closed and gap-retry rules below.
      const draftOrder = personas.filter((p) => p !== "stockbot");
      museCalls += draftOrder.length;
      const settled = await Promise.all(
        draftOrder.map((persona) => reasonFn({ ...input, persona }).then((r) => ({ persona, r }), (err: unknown) => ({ persona, err }))),
      );
      const draftText: Partial<Record<Persona, string>> = {};
      const failed: { persona: Persona; message: string }[] = [];
      for (const x of settled) {
        if ("err" in x) {
          failed.push({ persona: x.persona, message: `${x.persona}: ${x.err instanceof Error ? x.err.message : String(x.err)}` });
          continue;
        }
        addUsage(x.r.usage);
        const text = x.r.text.trim();
        if (!text) {
          failed.push({ persona: x.persona, message: `${x.persona}: empty response` });
          continue;
        }
        draftText[x.persona] = text;
        noteGap(x.r.missingEvidence);
      }
      if (failed.length === 1) {
        const only = failed[0];
        if (only) {
          museCalls += 1;
          const retry = await reasonFn({ ...input, persona: only.persona }).then(
            (r) => ({ r }) as const,
            (err: unknown) => ({ err }) as const,
          );
          if ("err" in retry) {
            failure = `${only.persona}: ${retry.err instanceof Error ? retry.err.message : String(retry.err)}`;
          } else {
            addUsage(retry.r.usage);
            const text = retry.r.text.trim();
            if (!text) failure = `${only.persona}: empty response`;
            else {
              draftText[only.persona] = text;
              noteGap(retry.r.missingEvidence);
            }
          }
        }
      }
      if (failure === undefined && failed.length > 1) failure = failed[0]?.message;
      if (failure === undefined) {
        museCalls += 1;
        const synth = await reasonFn({
          ...input,
          persona: "stockbot",
          drafts: draftOrder.map((persona) => ({ persona, text: draftText[persona] ?? "" })),
        }).then(
          (r) => ({ r }) as const,
          (err: unknown) => ({ err }) as const,
        );
        if ("err" in synth) {
          failure = `stockbot: ${synth.err instanceof Error ? synth.err.message : String(synth.err)}`;
        } else {
          addUsage(synth.r.usage);
          const text = synth.r.text.trim();
          if (!text) failure = "stockbot: empty response";
          else {
            // Drafts are visible sections: Bear, Bull, then the Stockbot verdict, one answer_delta.
            answerText = [...draftOrder.map((p) => [p, draftText[p] ?? ""] as const), ["stockbot", text] as const]
              .map(([p, t]) => `── ${p.charAt(0).toUpperCase()}${p.slice(1)} ──\n${t}`)
              .join("\n\n");
            noteGap(synth.r.missingEvidence);
          }
        }
      }
      museMs += performance.now() - tm;
      if (failure === undefined) gap = gaps.length > 0 ? gaps.join("; ") : undefined;
    }
    if (failure !== undefined) {
      countFailure("provider_error");
      emit({ type: "tool_failed", tool: "muse", category: "provider_error", preview: failure.slice(0, 160) });
      emit({ type: "failed", category: "provider_error", message: failure.slice(0, 160) });
      emit({ type: "done", metrics: buildMetrics(totalDecisions, totalCalls, mergedEvidence, totalWorkerMs, totalEscalations + 1, failures, { calls: museCalls, totalMs: museMs, ...usage }) });
      return;
    }
    if (gap === undefined) {
      emit({ type: "answer_delta", text: answerText ?? "" });
      emit({ type: "done", metrics: buildMetrics(totalDecisions, totalCalls, mergedEvidence, totalWorkerMs, totalEscalations, failures, { calls: museCalls, totalMs: museMs, ...usage }) });
      return;
    }
    if (pass >= 2 || deadlineAt - Date.now() < PASS2_MIN_MS) {
      countFailure("incomplete_evidence");
      emit({ type: "tool_failed", tool: "muse", category: "incomplete_evidence", preview: gap.slice(0, 160) });
      emit({ type: "failed", category: "incomplete_evidence", message: gap.slice(0, 160) });
      emit({ type: "done", metrics: buildMetrics(totalDecisions, totalCalls, mergedEvidence, totalWorkerMs, totalEscalations + 1, failures, { calls: museCalls, totalMs: museMs, ...usage }) });
      return;
    }
    pass += 1;
    direct = false;
    emit({ type: "progress", stage: "evidence_gap", detail: { missing: gap.slice(0, 300) } });
    const followStart = performance.now();
    let follow: KernelResponse;
    // ponytail: pass 2 reuses the pass-1 session — one gap node, no intake,
    // full session evidence visible. Old workers without sessionId keep the
    // legacy full-prompt call.
    const followBody: Record<string, unknown> =
      typeof res.sessionId === "string" && res.sessionId
        ? { op: "run", prompt, sessionId: res.sessionId, gap, deadlineAt }
        : { op: "run", prompt: `${prompt}\nStill missing: ${gap}`, deadlineAt };
    try {
      follow = await router.call(followBody, { signal: opts?.signal, timeoutMs: Math.max(deadlineAt - Date.now(), 1), onProgress: (stage, detail) => emit(detail !== undefined ? { type: "progress", stage, detail } : { type: "progress", stage }) });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      countFailure("provider_error");
      emit({ type: "tool_failed", tool: "needle", category: "provider_error", preview: msg.slice(0, 160) });
      emit({ type: "failed", category: "provider_error", message: msg.slice(0, 160) });
      emit({ type: "done", metrics: buildMetrics(totalDecisions, totalCalls, mergedEvidence, totalWorkerMs, totalEscalations + 1, failures, { calls: museCalls, totalMs: museMs, ...usage }) });
      return;
    }
    totalWorkerMs += performance.now() - followStart;
    const followEvidence: Evidence[] = (Array.isArray(follow.evidence) ? follow.evidence : []).map((e) => ({
      id: String(e.id),
      source: e.source ?? "kernel",
      ...(e.title !== undefined ? { title: e.title } : {}),
      ...(e.url !== undefined ? { url: e.url } : {}),
      retrievedAt: e.retrievedAt ?? new Date().toISOString(),
      content: String(e.content ?? ""),
    }));
    const seen = new Set(mergedEvidence.map((e) => e.id));
    for (const e of followEvidence) if (!seen.has(e.id)) {
      seen.add(e.id);
      mergedEvidence.push(e);
    }
    const followNodes: KernelGraphNode[] = (Array.isArray(follow.nodes) ? follow.nodes : [])
      .filter((n) => n && typeof n.node_id === "string")
      .map((n) => ({ node_id: String(n.node_id), question: String(n.question ?? ""), status: String(n.status ?? ""), depends_on: Array.isArray(n.depends_on) ? n.depends_on.map(String) : [] }));
    // ponytail: same-session pass 2 returns the full session lists — replace,
    // never append (appending would duplicate pass-1 nodes/decisions).
    const sameSession = typeof follow.sessionId === "string" && follow.sessionId === res.sessionId;
    mergedNodes = sameSession ? followNodes : [...mergedNodes, ...followNodes];
    mergedDecisions = sameSession
      ? (Array.isArray(follow.decisions) ? follow.decisions : [])
      : [...mergedDecisions, ...(Array.isArray(follow.decisions) ? follow.decisions : [])];
    mergedUnresolved = Array.isArray(follow.unresolved) ? follow.unresolved.map(String) : [];
    mergedGuard = follow.incompleteGuard ?? follow.incomplete_guard ?? false;
    mergedEscalated = follow.escalated ?? false;
    totalDecisions += (Array.isArray(follow.toolExecutions) ? follow.toolExecutions : follow.needleDecisions ?? []).length;
    totalCalls += (Array.isArray(follow.toolCalls) ? follow.toolCalls : []).length;
    totalEscalations += follow.escalations ?? 0;
    if (follow.failures && typeof follow.failures === "object") {
      for (const [k, v] of Object.entries(follow.failures)) if (typeof v === "number") failures[k as FailureCategory] = (failures[k as FailureCategory] ?? 0) + v;
    }
    const followCalls = Array.isArray(follow.toolCalls) ? follow.toolCalls : [];
    const byFollowId: Record<string, Evidence> = {};
    for (const e of mergedEvidence) byFollowId[e.id] = e;
    for (const c of followCalls) {
      emit({ type: "tool_start", tool: c.tool });
      if (c.ok) {
        const ev = c.evidenceId ? byFollowId[c.evidenceId] : undefined;
        emit({ type: "tool_result", tool: c.tool, evidenceId: c.evidenceId, preview: (ev?.content ?? "").slice(0, 160) });
      }
    }
  }
}
