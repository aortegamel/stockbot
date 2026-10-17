// decision/runtime.ts — persistent JEV sidecar: JSONL stdin/stdout bridge.
//
// Transport only. Translates runtime-neutral decide requests into the
// configured API format via SystemOneFn/createSystemOne, reusing the jev.ts
// parsers + answer-shape validation. Never schedules, never knows
// ResearchSession, never writes request-*/raw-*/policy-* files (experiment
// path in run.ts untouched).
//
// Protocol, one JSON object per line:
//   {"id": str, "op": "ping"} -> {"id": str, "ready": true}
//   {"id": str, "op": "decide", "state": unknown, "questions": Record<string, unknown>,
//    "choiceOptions"?: Record<string, Record<string, string>>}
//     -> {"id": str, "raw": unknown, "decisions": Record<string, DecisionResult>}
//     -> {"id": str, "error": str} (malformed request, transport failure, or
//        malformed answer — never crashes the loop)
import { createInterface } from "node:readline";
import {
  parseDecisions,
  requestDecisions,
} from "./jev.ts";
import type { DecisionResult, SystemOneFn } from "./jev.ts";
import { createSystemOne } from "./client.ts";

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export type DecideRequest = {
  id: string;
  op: "decide";
  state: unknown;
  questions: Record<string, unknown>;
  choiceOptions?: Record<string, Record<string, string>>;
};

export type DecideOk = { id: string; raw: unknown; decisions: Record<string, DecisionResult> };
export type DecideErr = { id: string; error: string };
export type PingOk = { id: string; ready: true };
export type DecideResponse = DecideOk | DecideErr | PingOk;

// Typed decide without filesystem writes: same validation + parser dispatch
// as askDecisions, minus request/raw/policy files.
export async function decideWithoutFs(
  req: { state: unknown; questions: Record<string, unknown> },
  systemOne: SystemOneFn,
  choiceOptions?: Record<string, Record<string, string>>,
  name = "decide",
): Promise<{ raw: unknown; decisions: Record<string, DecisionResult> }> {
  const raw = await requestDecisions(name, req, systemOne);
  const decisions = parseDecisions(name, req.questions, raw, choiceOptions);
  return { raw, decisions };
}

function badId(): DecideErr {
  return { id: "", error: "bad_request" };
}

function requestChoiceOptions(req: Record<string, unknown>): DecideRequest["choiceOptions"] {
  if (isObj(req.choiceOptions)) return req.choiceOptions as DecideRequest["choiceOptions"];
  if (isObj(req.options)) return req.options as DecideRequest["choiceOptions"];
  return undefined;
}

export async function handleDecide(req: unknown, systemOne: SystemOneFn): Promise<DecideResponse> {
  if (!isObj(req)) return badId();
  const id = req.id;
  if (typeof id !== "string" || !id) return badId();
  if (req.op === "ping") return { id, ready: true };
  if (req.op !== "decide") return { id, error: "unknown_op" };
  if (!isObj(req.questions)) return { id, error: "decide: missing_questions" };
  const questions = req.questions as Record<string, unknown>;
  const choiceOptions = requestChoiceOptions(req);
  try {
    const { raw, decisions } = await decideWithoutFs({ state: req.state ?? null, questions }, systemOne, choiceOptions, "decide");
    return { id, raw, decisions };
  } catch (e) {
    return { id, error: e instanceof Error ? e.message : String(e) };
  }
}

if (import.meta.main) {
  const systemOne = createSystemOne();
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    let req: unknown;
    try {
      req = JSON.parse(line);
    } catch {
      process.stdout.write(JSON.stringify({ error: "bad_request" }) + "\n");
      continue;
    }
    const res = await handleDecide(req, systemOne);
    process.stdout.write(JSON.stringify(res) + "\n");
  }
}
