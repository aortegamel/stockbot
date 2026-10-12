import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { homedir } from "node:os";

export type NeedleRouteResult = {
  tool: string | null;
  arguments: Record<string, unknown>;
  withheld: boolean;
  confidence: number | null;
  reasoning: string;
};

export type NeedleExtractResult = {
  record: string | null;
  fields: Record<string, unknown>;
  withheld: boolean;
  confidence: number | null;
  reasoning: string;
};

const TOOL_TIMEOUT_MS = 120_000;
// Runtime gate mirroring server.py validate_needle_tool: Needle output must
// invoke the exact JEV-selected tool. JEV owns selection; Needle never
// selects, chains, or judges sufficiency. Throws on mismatch (incl. null).
export function validateNeedleTool(jevTool: string, needleTool: string | null): string {
  if (!jevTool) throw new Error("validateNeedleTool: jevTool must be a nonempty tool name");
  if (needleTool !== jevTool)
    throw new Error(`needle tool mismatch: jev selected ${JSON.stringify(jevTool)}, needle emitted ${JSON.stringify(needleTool)}`);
  return needleTool;
}

const REPO_ROOT =
  process.env.STOCKBOT_REPO_ROOT ??
  (process.cwd().endsWith("needle-harness") ? process.cwd().replace(/\/needle-harness$/, "") : process.cwd());
const SERVER = `${REPO_ROOT}/needle-harness/lib/needle/server.py`;
const VENV_PYTHON = `${homedir()}/.cache/needle-harness/.needle/bin/python`;

type Pending =
  | { kind: "route"; resolve: (v: NeedleRouteResult) => void; reject: (e: Error) => void; cancel: () => void }
  | { kind: "extract"; resolve: (v: NeedleExtractResult) => void; reject: (e: Error) => void; cancel: () => void }
  | { kind: "embed"; resolve: (v: number[]) => void; reject: (e: Error) => void; cancel: () => void }
  | { kind: "ping"; resolve: () => void; reject: (e: Error) => void; cancel: () => void };

// Process boundary seam (mirrors KernelRouter spawnFn): production spawns the
// real server; tests inject fake protocol children. Router logic unmocked.
export type NeedleSpawn = (cmd: string, args: string[], opts: { env: NodeJS.ProcessEnv }) => ChildProcess;

function defaultSpawn(cmd: string, args: string[], opts: { env: NodeJS.ProcessEnv }): ChildProcess {
  return spawn(cmd, args, { stdio: ["pipe", "pipe", "pipe"] as const, detached: true, env: opts.env });
}
// ponytail: exclusive lock serializes Needle routing; per-session Needle state if Herdr panes need concurrency
let needleTail: Promise<void> = Promise.resolve();

export async function acquireNeedle(): Promise<() => void> {
  const { promise: grant, resolve: release } = Promise.withResolvers<void>();
  const prev = needleTail;
  needleTail = prev.then(() => grant);
  await prev;
  return () => release();
}


export class NeedleRouter {
  private child: ChildProcess | null = null;
  private readyChild: ChildProcess | null = null;
  private pending: Record<string, Pending> = {};
  private buf = "";
  private nextId = 0;
  private exited = false;
  private bridgeStderrTail = "";
  // Cold-spawn readiness: one ping round-trip per fresh child on the 120s
  // startup budget. Generation (10s) starts only after ready resolves, so
  // model/weights load never consumes the generation deadline.
  private ready: Promise<void> = Promise.resolve();
  private resolveReady: () => void = () => { };
  private rejectReady: (err: Error) => void = () => { };
  private readonly spawnFn: NeedleSpawn;

  constructor(opts?: { spawnFn?: NeedleSpawn }) {
    this.spawnFn = opts?.spawnFn ?? defaultSpawn;
  }

  private killOwned(child: ChildProcess): void {
    // Signal only the owned server tree: real spawns are detached so -pid is
    // our own process group (server plus native grandchildren). Fake protocol
    // children carry no pid and fall back to direct kill. A numeric exitCode
    // means already reaped; never signal a possibly reused pid.
    const current = child as unknown as { pid?: unknown; exitCode?: unknown; kill: (signal?: string) => void };
    if (typeof current.exitCode === "number") return;
    if (typeof current.pid === "number" && current.pid > 0) {
      try {
        process.kill(-current.pid, "SIGKILL");
        return;
      } catch {
        // Already gone or no groups on this platform; direct kill below.
      }
    }
    try {
      current.kill("SIGKILL");
    } catch {
      // Already gone; fresh spawn below replaces it.
    }
  }

  private spawn(): ChildProcess {
    const python = existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3";
    this.exited = false;
    this.buf = "";
    this.bridgeStderrTail = "";
    const child = this.spawnFn(python, [SERVER], {
      env: { ...process.env, NEEDLE_TELEMETRY: "0", DO_NOT_TRACK: "1" },
    });
    this.child = child;
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    this.ready = promise;
    this.resolveReady = resolve;
    this.rejectReady = reject;
    void promise.catch(() => {
      // Exit/startup rejections surface through rawCall awaiters; an unawaited
      // spawn must not surface as unhandled.
    });
    child.stdout?.on("data", (d: Buffer) => {
      if (this.child !== child) return;
      this.onData(d.toString());
    });
    child.stderr?.on("data", (d: Buffer) => {
      if (this.child !== child) return;
      const s = d.toString();
      this.bridgeStderrTail = (this.bridgeStderrTail + s).slice(-2000);
      process.stderr.write(s.startsWith("[needle-bridge]") ? s : `[needle-bridge] ${s}`);
    });
    child.on("error", (err) => {
      if (this.child !== child) return;
      const e = err instanceof Error ? err : new Error(String(err));
      this.rejectReady(e);
      this.failAll(e);
    });
    child.on("exit", () => {
      if (this.child !== child) return;
      this.exited = true;
      const err = new Error(`needle child exited; stderr tail: ${this.bridgeStderrTail || "(empty)"}`);
      this.rejectReady(err);
      if (Object.keys(this.pending).length > 0) {
        this.failAll(err);
      }
      this.child = null;
      if (this.readyChild === child) this.readyChild = null;
    });
    console.error(`[ai] needle bridge spawn ${python} (${SERVER})`);
    this.startReadyHandshake(child);
    return child;
  }

  private startReadyHandshake(child: ChildProcess): void {
    // One ping per cold spawn on the 120s startup budget: the server answers
    // only after imports load weights, so ready never touches the model.
    if (!child.stdin || !child.stdout) {
      this.child = null;
      this.rejectReady(new Error("needle spawn failed"));
      return;
    }
    const id = String((this.nextId += 1));
    const timer = setTimeout(() => {
      const p = this.pending[id];
      if (!p || p.kind !== "ping" || this.child !== child) return;
      delete this.pending[id];
      p.cancel();
      this.killOwned(child);
      this.child = null;
      if (this.readyChild === child) this.readyChild = null;
      const err = new Error(`needle startup timeout; stderr tail: ${this.bridgeStderrTail || "(empty)"}`);
      p.reject(err);
      this.rejectReady(err);
    }, TOOL_TIMEOUT_MS);
    this.pending[id] = {
      kind: "ping",
      resolve: () => {
        clearTimeout(timer);
        if (this.child === child) this.readyChild = child;
        this.resolveReady();
      },
      reject: (err: Error) => {
        clearTimeout(timer);
        this.rejectReady(err);
      },
      cancel: () => clearTimeout(timer),
    };
    child.stdin.write(JSON.stringify({ id, action: "ping" }) + "\n", (err) => {
      if (!err) return;
      const p = this.pending[id];
      if (!p || p.kind !== "ping" || this.child !== child) return;
      delete this.pending[id];
      p.cancel();
      this.rejectReady(err instanceof Error ? err : new Error(String(err)));
    });
  }

  private ensure(): ChildProcess {
    const running = this.child;
    if (running && !this.exited && running.exitCode === null) return running;
    if (running) this.killOwned(running);
    const fresh = this.spawn();
    if (!fresh.stdin || !fresh.stdout) {
      this.child = null;
      throw new Error("needle spawn failed");
    }
    return fresh;
  }

  private onLine(line: string): void {
    let msg: unknown;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    if (typeof msg !== "object" || msg === null || !("id" in msg)) return;
    const id: unknown = msg.id;
    if (typeof id !== "string") return;
    const p = this.pending[id];
    if (!p) return;
    delete this.pending[id];
    p.cancel();
    if ("error" in msg && typeof msg.error === "string") {
      p.reject(new Error(msg.error));
      return;
    }
    if (p.kind === "ping") {
      if ("ready" in msg && msg.ready === true) {
        p.resolve();
        return;
      }
      p.reject(new Error("needle startup failed: malformed server response"));
      return;
    }
    const confidence = "confidence" in msg && typeof msg.confidence === "number" ? msg.confidence : null;
    const reasoning = "reasoning" in msg && typeof msg.reasoning === "string" ? msg.reasoning : "";
    if (p.kind === "embed") {
      const vec = "embedding" in msg && Array.isArray(msg.embedding) ? msg.embedding : null;
      if (!vec || !vec.every((v) => typeof v === "number")) {
        p.reject(new Error("needle embed failed: malformed server response"));
        return;
      }
      p.resolve(vec as number[]);
      return;
    }
    if (p.kind === "extract") {
      const record = "record" in msg && (typeof msg.record === "string" || msg.record === null) ? msg.record : null;
      const fields =
        "fields" in msg && typeof msg.fields === "object" && msg.fields !== null
          ? (msg.fields as Record<string, unknown>)
          : null;
      const withheld = "withheld" in msg && typeof msg.withheld === "boolean" ? msg.withheld : false;
      if (fields === null) {
        p.reject(new Error("needle extract failed: malformed server response"));
        return;
      }
      p.resolve({ record, fields, withheld, confidence, reasoning });
      return;
    }
    const tool = "tool" in msg && (typeof msg.tool === "string" || msg.tool === null) ? msg.tool : null;
    const args =
      "arguments" in msg && typeof msg.arguments === "object" && msg.arguments !== null
        ? (msg.arguments as Record<string, unknown>)
        : {};
    const withheld = "withheld" in msg && typeof msg.withheld === "boolean" ? msg.withheld : false;
    p.resolve({ tool, arguments: args, withheld, confidence, reasoning });
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

  private async rawCall(
    body: Record<string, unknown>,
    kind: Pending["kind"],
    timeoutMs = TOOL_TIMEOUT_MS,
  ): Promise<NeedleRouteResult | NeedleExtractResult | number[]> {
    const running = this.child;
    const readyChild = this.readyChild;
    // Warm path: the same live ready child takes the request with no new ping.
    // Cold/timed-out path: ensure() spawns and starts the ping handshake.
    const child = readyChild && readyChild === running && !this.exited ? running : this.ensure();
    if (!child.stdin) throw new Error("needle spawn failed");
    const readyAtCall = this.ready;
    try {
      await readyAtCall;
    } catch (err) {
      throw err instanceof Error ? err : new Error(String(err));
    }
    if (this.child !== child) throw new Error(`needle worker replaced during startup; stderr tail: ${this.bridgeStderrTail || "(empty)"}`);
    const id = String((this.nextId += 1));
    return new Promise<NeedleRouteResult | NeedleExtractResult | number[]>((resolve, reject) => {
      const timeoutError = (): Error => new Error(`needle route timeout; stderr tail: ${this.bridgeStderrTail || "(empty)"}`);
      const timer = setTimeout(() => {
        const p = this.pending[id];
        if (!p) return;
        delete this.pending[id];
        p.cancel();
        const worker = this.child;
        if (worker === child) {
          this.child = null;
          if (this.readyChild === worker) this.readyChild = null;
          this.killOwned(worker);
          this.failAll(timeoutError());
        } else if (worker == null) {
          this.killOwned(child);
          this.failAll(timeoutError());
        } else {
          // Worker already replaced by a fresh spawn: reap the old child only;
          // the fresh worker and its pending calls stay untouched.
          this.killOwned(child);
        }
        reject(timeoutError());
      }, timeoutMs);
      const fail = (err: Error): void => {
        const p = this.pending[id];
        if (!p) return;
        delete this.pending[id];
        p.cancel();
        reject(err);
      };
      const entry = {
        resolve: resolve as (v: never) => void,
        reject,
        cancel: () => clearTimeout(timer),
      };
      this.pending[id] = { ...entry, kind } as Pending;
      child.stdin?.write(JSON.stringify({ id, ...body }) + "\n", (err) => {
        if (err) fail(err instanceof Error ? err : new Error(String(err)));
      });
    });
  }

  // Narrow execution worker op: generate arguments for the exact JEV-selected
  // tool only. Single request->response (no Needle-owned conversation).
  // Enforces emitted tool == requested tool; caller treats rejection as
  // retryable and returns to JEV with the full registry again.
  async generateArguments(req: {
    tool: string;
    schema?: unknown;
    objective?: unknown;
    node?: unknown;
    context?: unknown;
  }): Promise<NeedleRouteResult> {
    if (!req.tool) throw new Error("generateArguments: tool must be a nonempty tool name");
    const release = await acquireNeedle();
    try {
      const r = (await this.rawCall({ action: "arguments.generate", ...req }, "route")) as NeedleRouteResult;
      validateNeedleTool(req.tool, r.tool);
      return r;
    } finally {
      release();
    }
  }

  // Structured extraction: one record shape in, typed fields out. Grammar
  // admits exactly this record, never the full registry. JEV owns what to
  // extract and whether fields suffice; mismatch/ungrounded rejects.
  async extractFields(req: {
    record: { name: string;[k: string]: unknown };
    passage: string;
    system?: string;
    max_new_tokens?: number;
    strict?: boolean;
  }): Promise<NeedleExtractResult> {
    if (!req.record || typeof req.record.name !== "string" || !req.record.name)
      throw new Error("extractFields: record must carry a nonempty name");
    if (typeof req.passage !== "string" || !req.passage.trim()) throw new Error("extractFields: passage must be nonempty");
    const release = await acquireNeedle();
    try {
      const r = (await this.rawCall(
        {
          action: "extract",
          record: req.record,
          passage: req.passage,
          system: req.system,
          max_new_tokens: req.max_new_tokens ?? 512,
          strict: req.strict ?? true,
        },
        "extract",
      )) as NeedleExtractResult;
      if (r.record !== null && r.record !== req.record.name)
        throw new Error(`needle extract failed: malformed server response`);
      return r;
    } finally {
      release();
    }
  }

  // Retrieval embedding off the shared agent, serial like arguments.generate.
  async embed(text: string): Promise<number[]> {
    if (typeof text !== "string" || !text.trim()) throw new Error("embed: text must be nonempty");
    const release = await acquireNeedle();
    try {
      return (await this.rawCall({ action: "embed", text }, "embed")) as number[];
    } finally {
      release();
    }
  }

  async close(): Promise<void> {
    const child = this.child;
    this.child = null;
    this.readyChild = null;
    this.failAll(new Error("needle router closed"));
    this.rejectReady(new Error("needle router closed"));
    if (!child || child.exitCode !== null) return;
    this.killOwned(child);
    try {
      await once(child, "exit");
    } catch {
      // Already gone; nothing to wait for.
    }
  }
}
