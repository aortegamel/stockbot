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
export const GENERATE_TIMEOUT_MS = 10_000;
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
  | { kind: "embed"; resolve: (v: number[]) => void; reject: (e: Error) => void; cancel: () => void };
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
  private pending: Record<string, Pending> = {};
  private buf = "";
  private nextId = 0;
  private exited = false;
  private bridgeStderrTail = "";

  private spawn(): ChildProcess {
    const python = existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3";
    this.exited = false;
    this.buf = "";
    const child = spawn(python, [SERVER], {
      stdio: ["pipe", "pipe", "pipe"] as const,
      env: { ...process.env, NEEDLE_TELEMETRY: "0", DO_NOT_TRACK: "1" },
    });
    this.child = child;
    child.stdout?.on("data", (d: Buffer) => this.onData(d.toString()));
    child.stderr?.on("data", (d: Buffer) => {
      const s = d.toString();
      this.bridgeStderrTail = (this.bridgeStderrTail + s).slice(-2000);
      process.stderr.write(s.startsWith("[needle-bridge]") ? s : `[needle-bridge] ${s}`);
    });
    child.on("error", (err) => this.failAll(err));
    child.on("exit", () => {
      this.exited = true;
      if (Object.keys(this.pending).length > 0) {
        this.failAll(new Error(`needle child exited; stderr tail: ${this.bridgeStderrTail || "(empty)"}`));
      }
      this.child = null;
    });
    console.error(`[ai] needle bridge spawn ${python} (${SERVER})`);
    return child;
  }

  private ensure(): ChildProcess {
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

  private rawCall(
    body: Record<string, unknown>,
    kind: Pending["kind"],
    timeoutMs = TOOL_TIMEOUT_MS,
  ): Promise<NeedleRouteResult | NeedleExtractResult | number[]> {
    const child = this.ensure();
    if (!child.stdin) throw new Error("needle spawn failed");
    const id = String((this.nextId += 1));
    return new Promise<NeedleRouteResult | NeedleExtractResult | number[]>((resolve, reject) => {
      const timer = setTimeout(() => {
        delete this.pending[id];
        reject(new Error(`needle route timeout; stderr tail: ${this.bridgeStderrTail || "(empty)"}`));
      }, timeoutMs);
      const entry = {
        resolve: resolve as (v: never) => void,
        reject,
        cancel: () => clearTimeout(timer),
      };
      this.pending[id] = { ...entry, kind } as Pending;
      child.stdin?.write(JSON.stringify({ id, ...body }) + "\n", (err) => {
        if (err) {
          const p = this.pending[id];
          if (p) {
            delete this.pending[id];
            p.cancel();
            p.reject(err);
          }
        }
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
      const r = (await this.rawCall({ action: "arguments.generate", ...req }, "route", GENERATE_TIMEOUT_MS)) as NeedleRouteResult;
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
    this.failAll(new Error("needle router closed"));
    if (!child || child.exitCode !== null) return;
    child.kill();
    try {
      await once(child, "exit");
    } catch {
      // Already gone; nothing to wait for.
    }
  }
}
