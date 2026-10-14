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
  parseChoiceAnswer,
  parseNoulAnswer,
  parseScoreAnswer,
} from "./jev.ts";
import type { DecisionResult, SystemOneFn } from "./jev.ts";
import { createSystemOne } from "./client.ts";

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// Mirrors private helpers in jev.ts (not exported there); kept in sync by
// construction, never diverged with new semantics.
function scoreMaxFromQuestion(q: unknown): number | undefined {
  if (!isObj(q)) return undefined;
  const criteria = (q as Record<string, unknown>).criteria;
  if (Array.isArray(criteria)) return criteria.length - 1;
  const levels = (q as Record<string, unknown>).levels;
  if (Array.isArray(levels)) return levels.length - 1;
  return undefined;
}

function optionsFromQuestion(q: unknown): Record<string, string> {
  if (isObj(q) && isObj(q.criteria))
    return Object.fromEntries(Object.keys(q.criteria).map((k) => [k, k]));
  return {};
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
  let raw: unknown;
  try {
    raw = await systemOne(req);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`${name}: ${msg.startsWith("malformed_decision_response") ? msg : `decision_request_failed: ${msg}`}`);
  }
  const ids = Object.keys(req.questions).sort();
  if (!isObj(raw) || !isObj(raw.answers)) throw new Error(`${name}: malformed_decision_response`);
  const got = Object.keys(raw.answers).sort();
  if (got.length !== ids.length || got.some((k, i) => k !== ids[i]))
    throw new Error(`${name}: decision answers do not match questions`);
  const decisions: Record<string, DecisionResult> = {};
  for (const id of ids) {
    const ans = (raw.answers as Record<string, unknown>)[id];
    const q = req.questions[id];
    const qType = isObj(q) ? q.type : undefined;
    const aType = isObj(ans) ? ans.type : undefined;
    const kind = qType === "choice" || qType === "score" ? qType : aType;
    if (kind === "choice") {
      decisions[id] = parseChoiceAnswer(name, id, ans, choiceOptions?.[id] ?? optionsFromQuestion(q));
    } else if (kind === "score") {
      decisions[id] = parseScoreAnswer(name, id, ans, scoreMaxFromQuestion(q));
    } else {
      decisions[id] = parseNoulAnswer(name, id, ans);
    }
  }
  return { raw, decisions };
}

function badId(): DecideErr {
  return { id: "", error: "bad_request" };
}

export async function handleDecide(req: unknown, systemOne: SystemOneFn): Promise<DecideResponse> {
  if (!isObj(req)) return badId();
  const id = req.id;
  if (typeof id !== "string" || !id) return badId();
  if (req.op === "ping") return { id, ready: true };
  if (req.op !== "decide") return { id, error: "unknown_op" };
  if (!isObj(req.questions)) return { id, error: "decide: missing_questions" };
  const questions = req.questions as Record<string, unknown>;
  const choiceOptions =
    (isObj(req.choiceOptions) ? (req.choiceOptions as DecideRequest["choiceOptions"]) : undefined) ??
    (isObj(req.options) ? (req.options as DecideRequest["choiceOptions"]) : undefined);
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
