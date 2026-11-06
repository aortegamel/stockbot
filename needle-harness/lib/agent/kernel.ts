import { spawn } from "node:child_process";
import { join } from "node:path";
import { describeTools } from "../tools/stockbot";
import { reason, type MuseUsage } from "../muse/client";
import { readDatapoints, sourceReferences, statusEvent, toolStatus } from "./types";
import type { AgentEvent, AskContext, Datapoint, Evidence, FailureCategory, FinalClaim, Metrics, NarrativeResult, Timings } from "./types";

// ponytail: persistent worker bridge (spawn-once, ready-gated); spawn/exit/stderr handling mirrors lib/needle/client.ts.
const WORKER_TIMEOUT_MS = 10 * 60 * 1000;
// Every research operation shares the request deadline.
export const REQUEST_WALL_MS = 180_000;
const PREWARM_TIMEOUT_MS = 120_000;

const ROOT =
  process.env.STOCKBOT_REPO_ROOT ??
  (process.cwd().endsWith("needle-harness") ? process.cwd().replace(/\/needle-harness$/, "") : process.cwd());
const WORKER = join(ROOT, "app/research/kernel_worker.py");
const REPO_PYTHON = join(ROOT, "venv/bin/python");
const TICKER_RE = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const TICKER_ORG: Record<string, true> = { FINRA: true, SEC: true, NYSE: true, NASDAQ: true, OTC: true, EDGAR: true, SHO: true, REG: true };
// Validated ticker channel beside the verbatim prompt; undefined when absent or invalid (dropped, never errors).
export function parseTicker(v: unknown): string | undefined {
  if (typeof v !== "string") return undefined;
  const t = v.trim().toUpperCase();
  if (!t) return undefined;
  if (TICKER_ORG[t]) return undefined;
  if (t.length > 10 || !TICKER_RE.test(t)) return undefined;
  return t;
}
const WATCHLIST_MAX = 25;
// Watchlist context channel beside the verbatim prompt: undefined when absent,
// null when malformed (callers reject; a bad list never silently shrinks).
export function parseWatchlist(v: unknown): string[] | null | undefined {
  if (v === undefined || v === null) return undefined;
  if (!Array.isArray(v) || v.length === 0 || v.length > WATCHLIST_MAX) return null;
  const out: string[] = [];
  for (const item of v) {
    const t = parseTicker(item);
    if (t === undefined) return null;
    if (!out.includes(t)) out.push(t);
  }
  return out;
}
// Watchlist scope applies only when the prompt names the watchlist; every
// other question keeps the selected-ticker scope.
const WATCHLIST_WORD = /\bwatch[\s-]?lists?\b/i;
export function asksWatchlist(prompt: string): boolean {
  return WATCHLIST_WORD.test(prompt);
}
// One (symbol, source) outcome the worker persisted; never client-provided.
type WatchlistRow = {
  symbol: string;
  source: string;
  tool: string;
  status: "ok" | "no_records" | "error" | "timeout";
  as_of?: string;
  evidence_id?: string;
  error?: string;
};
// Page-load trigger: same worker, same tool handlers and caches as agent intake; read-only.
// The budget mirrors the worker intake budget (_INTAKE_BUDGET_S); a queued call keeps the same deadline.
export const WATCHLIST_BUDGET_MS = 15_000;

type KernelEvidence = {
  id: string;
  content: string;
  source?: string;
  title?: string;
  url?: string;
  retrievedAt?: string;
  sources?: unknown;
  datapoints?: unknown;
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

export type KernelReasonResult = NarrativeResult & { usage: MuseUsage; timings?: Timings };
type KernelReasonFn = (opts: KernelReasonInput) => Promise<KernelReasonResult>;
// Gate-rejection failures end with this suffix; only they get the fixed UNDETERMINED answer.
const ALL_REJECTED = "all claims rejected";
const UNDETERMINED = "I couldn't determine this from the retrieved sources.";
export function inventoryDefect(
  prose: string,
  claims: FinalClaim[],
  evidenceIds: ReadonlySet<string>,
  datapointRows: ReadonlyMap<string, readonly string[]>,
): string | undefined {
  for (const c of claims) {
    if (!c.text || !prose.includes(c.text)) return `inventory text not a prose substring: ${c.text.slice(0, 80)}`;
    for (const id of c.evidenceIds) if (!evidenceIds.has(id)) return `inventory cites unknown evidence id: ${id}`;
    const points = c.datapointIds ?? [];
    if (c.claimType !== "unknown" && c.claimType !== "assumption" && c.evidenceIds.length === 0) return `uncited non-unknown claim: ${c.text.slice(0, 80)}`;
    for (const id of points) {
      const rows = datapointRows.get(id) ?? [];
      if (!rows.some((row) => c.evidenceIds.includes(row))) return `inventory cites unknown datapoint id: ${id}`;
    }
    if (c.claimType === "calculation" && points.length === 0) return `calculation without datapoint_ids: ${c.text.slice(0, 80)}`;
  }
  return undefined;
}
// Datapoint id -> every evidence row that carries it (a claim must cite one). One source anchor yields one id,
// so a fan-out row and a claim-time admission of the same number share it.
export function datapointRows(evidence: readonly Evidence[]): Map<string, string[]> {
  const rows = new Map<string, string[]>();
  for (const e of evidence) for (const p of e.datapoints ?? []) rows.set(p.id, [...(rows.get(p.id) ?? []), e.id]);
  return rows;
}
function toWireClaim(c: FinalClaim): Record<string, unknown> {
  return {
    text: c.text,
    claim_type: c.claimType,
    evidence_ids: [...c.evidenceIds],
    ...(c.datapointIds !== undefined && c.datapointIds.length > 0 ? { datapoint_ids: [...c.datapointIds] } : {}),
    ...(c.confidence !== undefined ? { confidence: c.confidence } : {}),
  };
}
// A sentence ends at a line break, or at [.!?] before whitespace, so decimals stay whole.
const SENTENCE = /[^\n]*?(?:[.!?](?=\s|$)|\n|$)/g;
// Words that may join kept claim texts inside one sentence without asserting anything (Python final._CONNECTIVES).
const CONNECTIVES: Record<string, true> = { and: true, but: true, while: true, whereas: true, also: true, however: true, or: true };
// Exhaustive inventory: a sentence shows only when kept claim texts cover all of it
// (connectives, punctuation, and citation brackets aside), so neither an unproven
// number nor an unproven qualitative clause rides beside a kept claim.
export function dropUncovered(text: string, kept: readonly FinalClaim[]): string {
  const spans: [number, number][] = [];
  for (const c of kept) {
    for (let at = c.text ? text.indexOf(c.text) : -1; at >= 0; at = text.indexOf(c.text, at + 1)) spans.push([at, at + c.text.length]);
  }
  const masked = text.replace(/\[[^\]]*\]/g, (b) => " ".repeat(b.length));
  let out = "";
  // A citation-only fragment ("2026. [ev:1]") belongs to the sentence before it: it shows only when that one does.
  let previousKept = false;
  for (const m of text.matchAll(SENTENCE)) {
    const at = m.index;
    const bare = !/[a-z0-9]/i.test(masked.slice(at, at + m[0].length));
    let rest = "";
    for (let i = at; i < at + m[0].length; i++) if (!spans.some(([a, b]) => a <= i && i < b)) rest += masked[i];
    const keep: boolean = bare ? previousKept : (rest.toLowerCase().match(/[a-z0-9]+/g) ?? []).every((w) => CONNECTIVES[w] === true);
    if (keep) out += m[0];
    else if (m[0].endsWith("\n")) out += "\n";
    if (!bare) previousKept = keep;
  }
  return out.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

// Display only, after every gate: the gates prove the raw source figure ("$96,221 million", "$215938000000");
// the reader sees "$96.2 billion". Only a value followed by a scale word, or a whole number of 7+ digits that is
// a verified datapoint value of this run, changes; so does a camelCase metric label of a datapoint. Citation
// brackets, CIKs, accession numbers, dates, and every other number stay verbatim. One decimal, ".0" dropped.
const SCALES: Record<string, number> = { thousand: 1e3, million: 1e6, billion: 1e9, trillion: 1e12 };
const AMOUNT = /(?<![\w.,-])(\$?)(-?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s+(thousand|million|billion|trillion)s?\b)?(?![\w,-]|\.\d)/gi;
const FIELD = /\b[a-z]+(?:[A-Z][a-z]+)+\b/g;
export function readableNumbers(text: string, points: readonly Datapoint[]): string {
  const values = new Set(points.map((p) => p.value.replace(/^\$/, "").replace(/,/g, "")));
  const fields = new Set(points.map((p) => p.metric));
  return text.split(/(\[[^\]]*\])/).map((part) => part.startsWith("[") ? part : part
    .replace(FIELD, (name) => fields.has(name) ? name.replace(/([A-Z])/g, " $1").toLowerCase() : name)
    .replace(AMOUNT, (whole, dollar: string, sign: string, digits: string, scale: string | undefined) => {
      const plain = digits.replace(/,/g, "");
      if (scale === undefined && !(/^\d{7,}$/.test(plain) && values.has(sign + plain))) return whole;
      const amount = Number(plain) * (scale ? SCALES[scale.toLowerCase()] ?? 1 : 1);
      const [unit, size] = amount >= 1e12 ? ["trillion", 1e12] : amount >= 1e9 ? ["billion", 1e9] : amount >= 1e6 ? ["million", 1e6] : [undefined, 1];
      if (unit === undefined || !Number.isFinite(amount)) return whole;
      return `${dollar}${sign}${String(Math.round((amount / size) * 10) / 10)} ${unit}`;
    })).join("");
}
// A claims[].text that says the same thing as the prose in another spelling is not wrong: it maps to the exact
// prose span, and the gates judge that published text. Spacing, quote style, case, citation brackets, and trailing
// punctuation never matter. Failing that, the one prose sentence that shares most of the claim's words (Jaccard
// >= 0.6) and exactly its numbers stands in. No match means the claim names nothing the answer says.
const QUOTES: Record<string, string> = { "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"' };
function normalizedWithMap(text: string): { flat: string; at: number[] } {
  let flat = "";
  const at: number[] = [];
  const masked = text.replace(/\[[^\]]*\]/g, (b) => " ".repeat(b.length));
  for (let i = 0; i < masked.length; i++) {
    const ch = QUOTES[masked[i] ?? ""] ?? (masked[i] ?? "").toLowerCase();
    if (/\s/.test(ch)) {
      if (flat && !flat.endsWith(" ")) { flat += " "; at.push(i); }
      continue;
    }
    flat += ch;
    at.push(i);
  }
  return { flat, at };
}
const WORD = /[a-z0-9]+(?:[.,][0-9]+)*/g;
export function alignClaimText(prose: string, text: string): string | undefined {
  if (!text.trim()) return undefined;
  if (prose.includes(text)) return text;
  const { flat, at } = normalizedWithMap(prose);
  const want = normalizedWithMap(text).flat.trim().replace(/[\s.!?;:,]+$/, "");
  const found = want ? flat.indexOf(want) : -1;
  if (found >= 0) {
    const start = at[found] ?? 0;
    return prose.slice(start, (at[found + want.length - 1] ?? start) + 1);
  }
  const words = new Set(want.match(WORD) ?? []);
  const wantNumbers = [...words].filter((w) => /\d/.test(w)).sort().join(" ");
  let best: { sentence: string; score: number } | undefined;
  for (const m of prose.matchAll(SENTENCE)) {
    const sentence = m[0].replace(/\s*(?:\[[^\]]*\]\s*)*$/, "").trim();
    const mine = new Set(normalizedWithMap(sentence).flat.match(WORD) ?? []);
    if (mine.size === 0 || [...mine].filter((w) => /\d/.test(w)).sort().join(" ") !== wantNumbers) continue;
    const shared = [...words].filter((w) => mine.has(w)).length;
    const score = shared / (words.size + mine.size - shared);
    if (score >= 0.6 && (best === undefined || score > best.score)) best = { sentence, score };
  }
  return best?.sentence;
}
type RunKernelDeps = {
  spawnFn?: KernelSpawn;
  reason?: KernelReasonFn;
  python?: string;
  workerPath?: string;
  describeTools?: typeof describeTools;
};

function defaultSpawn(cmd: string, args: string[], opts: { env: NodeJS.ProcessEnv }): KernelChild {
  return spawn(cmd, args, { stdio: ["pipe", "pipe", "pipe"], env: opts.env }) as unknown as KernelChild;
}

// Untrusted JSONL from the worker: non-JSON noise and non-object lines drop.
function parseLine(line: string): Record<string, unknown> | undefined {
  let msg: unknown;
  try {
    msg = JSON.parse(line);
  } catch {
    return undefined;
  }
  if (typeof msg !== "object" || msg === null) return undefined;
  return msg as Record<string, unknown>;
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
    this.python = opts?.python ?? REPO_PYTHON;
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
    const rec = parseLine(line);
    if (rec === undefined) return;
    if (typeof rec.id !== "string") {
      // Ready gate: worker hello, never a response.
      if (rec.type === "ready") this.resolveReady();
      return;
    }
    if (rec.type === "progress") {
      // Progress is advisory: a dead listener never fails the run, and the
      // pending call stays open for the terminal reply.
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
    p.resolve(rec as KernelResponse);
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

export type KernelReasonInput = Parameters<typeof reason>[0];
export type KernelResponse = {
  id?: string;
  sessionId?: string;
  tools?: unknown;
  evidence_id?: unknown;
  datapoint?: unknown;
  evidence?: unknown;
  summary?: unknown;
  ok?: boolean;
  admitted?: unknown;
  error?: string;
  tool?: string;
  arguments?: Record<string, unknown>;
  confidence?: number | null;
  reasoning?: string;
  effort?: unknown;
  sections?: unknown;
  in_scope?: boolean;
  answer?: string;
  route?: string;
  chartTicker?: unknown;
  chartError?: unknown;
  content?: string;
  verdicts?: unknown;
  watchlist?: WatchlistRow[];
};
type KernelRunOptions = {
  signal?: AbortSignal;
  deadlineAt?: number;
  asOf?: string;
  reasoningEffort?: string;
  deps?: RunKernelDeps;
  seedEvidence?: Evidence[];
  ticker?: string;
  currentChartTicker?: string;
  watchlist?: readonly string[];
  context?: AskContext;
};
type KernelRun = {
  t0: number;
  emit: (event: AgentEvent) => void;
  router: KernelRouter;
  signal?: AbortSignal;
  deadlineAt: number;
  sessionId?: string;
  evidence: Evidence[];
  failures: Partial<Record<FailureCategory, number>>;
  callsCount: number;
  workerMs: number;
  museCalls: number;
  museMs: number;
  usage: MuseUsage;
  timings: Timings;
  limitations: string[];
};
function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
function metricsOf(run: KernelRun): Metrics {
  return {
    totalMs: performance.now() - run.t0,
    needle: { calls: 0, totalMs: 0, escalations: 0 },
    tools: { calls: run.callsCount, totalMs: run.workerMs },
    muse: { calls: run.museCalls, totalMs: run.museMs, ...run.usage },
    evidence: { count: run.evidence.length, characters: run.evidence.reduce((sum, row) => sum + row.content.length, 0) },
    failures: run.failures,
    timings: run.timings,
  };
}
function endFailed(run: KernelRun, message: string): void {
  run.failures.provider_error = (run.failures.provider_error ?? 0) + 1;
  emitPartial(run, "", []);
  if (run.signal?.aborted) run.emit({ type: "cancelled", message: "The request was cancelled." });
  else {
    run.emit({ type: "tool_failed", tool: "reasoner", category: "provider_error", preview: message.slice(0, 160) });
    run.emit({ type: "failed", category: "provider_error", message: message.slice(0, 160), code: "provider_error", stage: "reasoner" });
  }
  run.emit({ type: "done", metrics: metricsOf(run), status: run.signal?.aborted ? "cancelled" : "failed" });
}
function emitPartial(run: KernelRun, text: string, claims: FinalClaim[]): void {
  run.emit({ type: "answer_partial", text, claims });
}
function endUndetermined(run: KernelRun): void {
  run.failures.provider_error = (run.failures.provider_error ?? 0) + 1;
  emitPartial(run, UNDETERMINED, []);
  run.emit({ type: "done", metrics: metricsOf(run), status: "completed_with_limitations", limitations: [...run.limitations, ALL_REJECTED] });
}
export function outOfScopeAnswer(res: { in_scope?: unknown; answer?: unknown }): string | undefined {
  return res.in_scope === false && typeof res.answer === "string" && res.answer.trim() ? res.answer.trim() : undefined;
}
function kernelEvidence(raw: unknown): Evidence[] {
  const rows = Array.isArray(raw) ? (raw as KernelEvidence[]) : [];
  return rows.filter((e) => e && typeof e === "object" && typeof e.id === "string" && e.id.length > 0 && typeof e.content === "string").map((e) => {
    const datapoints = readDatapoints(e.datapoints);
    return {
      id: String(e.id),
      source: e.source ?? "kernel",
      ...(e.title !== undefined ? { title: e.title } : {}),
      ...(e.url !== undefined ? { url: e.url } : {}),
      retrievedAt: e.retrievedAt ?? new Date().toISOString(),
      content: String(e.content ?? ""),
      sources: sourceReferences(e.sources),
      ...(datapoints.length > 0 ? { datapoints } : {}),
    };
  });
}
async function admitNumbers(run: KernelRun, results: readonly NarrativeResult[]): Promise<void> {
  const declarations = results.flatMap((result) => result.claims.flatMap((claim) =>
    (claim.numbers ?? []).map((number) => ({ claim, number }))));
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(4, declarations.length) }, async () => {
    while (next < declarations.length) {
      const entry = declarations[next++];
      if (!entry || !run.sessionId) continue;
      const { claim, number } = entry;
      const source = run.evidence.find((row) => row.id === number.evidence_id);
      if (!source || !claim.evidenceIds.includes(source.id)) continue;
      try {
        const response = await run.router.call({
          op: "admit_datapoint",
          sessionId: run.sessionId,
          evidenceId: number.evidence_id,
          metric: number.metric,
          value: number.value,
          entity: number.entity,
          period: number.period,
          ...(number.unit !== undefined ? { unit: number.unit } : {}),
          ...(number.formula !== undefined ? { formula: number.formula, inputs: number.inputs } : {}),
        }, run.signal ? { signal: run.signal } : {});
        const points = readDatapoints([response.datapoint]);
        const point = points[0];
        if (response.error || typeof response.evidence_id !== "string" || !response.evidence_id || !point) continue;
        if (!run.evidence.some((row) => row.id === response.evidence_id)) {
          run.evidence.push({ ...source, id: response.evidence_id, datapoints: points });
        }
        if (!claim.evidenceIds.includes(response.evidence_id)) claim.evidenceIds.push(response.evidence_id);
        claim.datapointIds = [...new Set([...(claim.datapointIds ?? []), point.id])];
      } catch {
        // Admission failure leaves the declaration unproven. Existing gates reject it.
      }
    }
  }));
  // A growth claim often also lists its inputs' dp: ids. An input whose value the sentence never states is not
  // what the sentence claims, so it leaves the claim; a stated input value stays and must still bind. A dp: id no
  // run row carries (a mistyped id) leaves too: every stated number still needs a bound datapoint at the gate.
  // A real dp: id whose row the claim forgot to cite gains that row: the gate still proves the datapoint there.
  const pointById = new Map(run.evidence.flatMap((row) => (row.datapoints ?? []).map((p) => [p.id, p] as const)));
  const rowsOf = datapointRows(run.evidence);
  for (const claim of results.flatMap((result) => result.claims)) {
    const ids = claim.datapointIds ?? [];
    const inputs = new Set(ids.flatMap((id) => pointById.get(id)?.inputs ?? []));
    const stated = ids.filter((id) => pointById.has(id) && (!inputs.has(id) || claim.text.includes(pointById.get(id)?.value ?? id)));
    // One stated number has one datapoint: when several cite it, keep the one whose metric the sentence
    // names ("filed" for filed_at). Still ambiguous stays out, so the proof reads it as unbound.
    const byValue = new Map<string, string[]>();
    for (const id of stated) {
      const value = pointById.get(id)?.value ?? "";
      byValue.set(value, [...(byValue.get(value) ?? []), id]);
    }
    const kept = [...byValue.values()].flatMap((group) => {
      if (group.length === 1) return group;
      const named = group.filter((id) => {
        const head = pointById.get(id)?.metric?.toLowerCase().split(/[^a-z0-9]+/).find((word) => word.length > 2);
        return head === undefined || claim.text.toLowerCase().includes(head);
      });
      return named.length === 1 ? named : [];
    });
    if (kept.length !== ids.length) claim.datapointIds = kept;
    for (const id of kept) {
      const rows = rowsOf.get(id) ?? [];
      const row = rows[0];
      if (row !== undefined && !rows.some((r) => claim.evidenceIds.includes(r))) claim.evidenceIds.push(row);
    }
    // A claim that states a value the run proves proves it: the row carrying that datapoint binds even when
    // the claim cited a neighbour row, because a filing sentence states a date and an accession that live on
    // two rows. A value the claim already binds stays as it is; two datapoints still share a stated value only
    // when neither label is named, and that stays unbound so the proof rejects it.
    const bound = new Set(claim.datapointIds ?? []);
    const lower = claim.text.toLowerCase();
    const candidates = new Map<string, string[]>();
    for (const row of run.evidence) {
      for (const point of row.datapoints ?? []) {
        if (!point.value || !claim.text.includes(point.value)) continue;
        if ([...bound].some((id) => pointById.get(id)?.value === point.value)) continue;
        const head = point.metric?.toLowerCase().split(/[^a-z0-9]+/).find((word) => word.length > 2);
        if (head && !lower.includes(head)) continue;
        candidates.set(point.value, [...(candidates.get(point.value) ?? []), point.id]);
      }
    }
    for (const ids of candidates.values()) {
      const only = ids.length === 1 ? ids[0] : undefined;
      if (only === undefined) continue;
      bound.add(only);
      const row = rowsOf.get(only)?.[0];
      if (row !== undefined && !claim.evidenceIds.includes(row)) claim.evidenceIds.push(row);
    }
    if (bound.size !== (claim.datapointIds ?? []).length) claim.datapointIds = [...bound];
  }
}

// Canonical schemas come from the Python research capability set.
const RESEARCH_DOCUMENT_CHARS = 12000;
export async function runKernelAgent(prompt: string, emit: (event: AgentEvent) => void, opts?: KernelRunOptions): Promise<void> {
  const deps = opts?.deps ?? {};
  const router = deps.spawnFn || deps.python || deps.workerPath
    ? new KernelRouter({ spawnFn: deps.spawnFn, python: deps.python, workerPath: deps.workerPath }) : kernelRouter;
  const run: KernelRun = {
    t0: performance.now(), emit, router, signal: opts?.signal,
    deadlineAt: opts?.deadlineAt ?? Date.now() + REQUEST_WALL_MS,
    evidence: [...(opts?.seedEvidence ?? [])], failures: {}, callsCount: 0, workerMs: 0,
    museCalls: 0, museMs: 0, usage: {}, timings: {}, limitations: [],
  };
  const worker = (body: Record<string, unknown>): Promise<KernelResponse> => router.call({ ...body, deadlineAt: run.deadlineAt }, {
    signal: run.signal, timeoutMs: Math.max(1, run.deadlineAt - Date.now()),
  });
  emit({ type: "agent_start", prompt, ...(opts?.context ? { context: opts.context } : {}) });
  emit(statusEvent(`Researching: ${prompt}`));
  if (run.evidence.length) emit({ type: "evidence", evidence: run.evidence });
  let streamed = "";
  try {
    const opened = await worker({ op: "open_research", prompt, ...(opts?.asOf !== undefined ? { asOf: opts.asOf } : {}), ...(opts?.watchlist?.length ? { watchlist: opts.watchlist } : { ticker: opts?.ticker ?? opts?.currentChartTicker }) });
    if (opened.error || typeof opened.sessionId !== "string" || !opened.sessionId) throw new Error(opened.error ?? "Research returned no session id.");
    run.sessionId = opened.sessionId;
    if (!Array.isArray(opened.tools) || !opened.tools.every((name): name is string => typeof name === "string" && name.length > 0)) throw new Error("Research returned invalid tool names.");
    const schemas = await (deps.describeTools ?? describeTools)();
    const tools = opened.tools.map((name) => {
      const schema = schemas.find((tool) => tool.name === name);
      if (!schema) throw new Error(`Research has no schema for tool ${name}.`);
      return schema;
    });
    const allowed = new Set(opened.tools);
    const runTool = async (tool: string, args: Record<string, unknown>): Promise<string> => {
      if (!allowed.has(tool)) return `error: ${tool} is not available in research`;
      const arguments_ = tool === "get_sec_document" ? { ...args, limit: Math.min(typeof args.limit === "number" ? args.limit : RESEARCH_DOCUMENT_CHARS, RESEARCH_DOCUMENT_CHARS) } : args;
      emit(statusEvent(toolStatus(tool)));
      emit({ type: "tool_start", tool });
      const at = performance.now();
      run.callsCount++;
      try {
        const reply = await worker({ op: "research_tool", sessionId: run.sessionId, tool, arguments: arguments_ });
        const rows = kernelEvidence(reply.evidence);
        const incoming = rows.filter((row) => !run.evidence.some((existing) => existing.id === row.id));
        run.evidence.push(...incoming);
        if (incoming.length) emit({ type: "evidence", evidence: incoming, sessionId: run.sessionId });
        if (!reply.ok) {
          const message = reply.error ?? `${tool} returned no result.`;
          run.failures.tool_error = (run.failures.tool_error ?? 0) + 1;
          run.limitations.push(message);
          emit({ type: "tool_failed", tool, category: "tool_error", preview: message.slice(0, 160) });
          return `error: ${message}`;
        }
        emit({ type: "tool_result", tool, evidenceId: incoming[0]?.id, preview: String(reply.summary ?? "").slice(0, 160) });
        return JSON.stringify({ summary: reply.summary, evidence: rows });
      } catch (error) {
        const message = errorText(error);
        run.limitations.push(message);
        emit({ type: "tool_failed", tool, category: "tool_error", preview: message.slice(0, 160) });
        return `error: ${message}`;
      } finally { run.workerMs += performance.now() - at; }
    };
    const at = performance.now();
    run.museCalls++;
    emit({ type: "reasoning_start", model: process.env.OPENCODE_MODEL ?? "stockbot" });
    let result: KernelReasonResult;
    try {
      result = await (deps.reason ?? reason)({
        prompt, evidence: run.evidence, tools, runTool,
        ticker: opts?.ticker ?? opts?.currentChartTicker, context: opts?.context,
        reasoningEffort: opts?.reasoningEffort, signal: run.signal, deadlineAt: run.deadlineAt,
        usage: run.usage, timings: run.timings,
        onDelta: (text) => { streamed += text; emit({ type: "answer_delta", text }); },
      });
      run.usage = result.usage;
    } finally { run.museMs += performance.now() - at; }
    if (!result.prose.trim()) throw new Error("The model returned an empty answer.");
    const priorIds = new Set(run.evidence.map((row) => row.id));
    await admitNumbers(run, [result]);
    const admitted = run.evidence.filter((row) => !priorIds.has(row.id));
    if (admitted.length) emit({ type: "evidence", evidence: admitted, sessionId: run.sessionId });
    const ids = new Set(run.evidence.map((row) => row.id));
    const points = datapointRows(run.evidence);
    const candidates = result.claims.map((claim) => ({ ...claim, text: alignClaimText(result.prose, claim.text) ?? claim.text }));
    const rejected: { text: string; reason: string }[] = [];
    const valid = candidates.filter((claim) => {
      const defect = inventoryDefect(result.prose, [claim], ids, points);
      if (defect !== undefined) rejected.push({ text: claim.text, reason: defect });
      return defect === undefined;
    });
    const supported: FinalClaim[] = [];
    if (valid.length) {
      const proof = await worker({ op: "prove_claims", sessionId: run.sessionId, claims: valid.map(toWireClaim), evidence: run.evidence.map(({ id, content, datapoints }) => ({ id, content, datapoints })) });
      if (proof.error || !Array.isArray(proof.verdicts) || proof.verdicts.length !== valid.length) throw new Error(proof.error ?? "Proof returned invalid verdicts.");
      proof.verdicts.forEach((raw: unknown, index: number) => {
        if (!raw || typeof raw !== "object") throw new Error("Proof returned an invalid verdict.");
        const verdict = raw as Record<string, unknown>;
        const claim = valid[index]!;
        if (verdict.error && (typeof verdict.error !== "string" || !verdict.error.startsWith("proof:"))) throw new Error(`Proof unavailable: ${String(verdict.error)}`);
        if (verdict.claim_type !== claim.claimType || !["yes", "no"].includes(String(verdict.support_answer))) throw new Error("Proof returned an invalid verdict.");
        if (verdict.support_answer === "yes" && !verdict.error) supported.push(claim);
        else rejected.push({ text: claim.text, reason: String(verdict.error ?? verdict.evidence_state ?? "not supported") });
      });
    }
    // The operator log names every rejection, so a dropped sentence is diagnosable after the run.
    for (const r of rejected) console.error(`[ai] claim rejected: ${r.reason.slice(0, 200)} | ${r.text.slice(0, 160)}`);
    const blocks = result.prose.split(/^(── (?:Bearbot|Bullbot|Stockbot) ──)\s*$/m);
    let text = dropUncovered(blocks[0] ?? "", supported);
    for (let index = 1; index < blocks.length; index += 2) {
      const body = dropUncovered(blocks[index + 1] ?? "", supported);
      if (body) text += `${text ? "\n\n" : ""}${blocks[index]}\n${body}`;
    }
    const kept = supported.filter((claim) => text.includes(claim.text));
    if (!text || !kept.length) { endUndetermined(run); return; }
    const points_ = run.evidence.flatMap((row) => row.datapoints ?? []);
    const shown = readableNumbers(text, points_);
    const claims = kept.map((claim) => ({ ...claim, text: readableNumbers(claim.text, points_) }));
    emit({ type: "claims", claims });
    const partial = kept.length !== candidates.length || text.trim() !== result.prose.trim() || run.limitations.length > 0;
    if (partial) run.limitations.push("The proof check removed unsupported text.");
    // The snapshot also applies numeric formatting without appending a second answer.
    if (partial || streamed !== shown) emitPartial(run, shown, claims);
    run.emit({ type: "done", metrics: metricsOf(run), status: partial ? "completed_with_limitations" : "completed", ...(partial ? { limitations: [...new Set(run.limitations)] } : {}) });
  } catch (error) { endFailed(run, errorText(error)); }
}
