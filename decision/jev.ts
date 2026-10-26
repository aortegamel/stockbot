// decision/jev.ts — typed JEV decision layer (API-only, no OMP/Pi runtime).
//
// choice = exclusive routing/category/explanation: exactly one winner among
// mutually exclusive labels (use DISPOSITION_OPTIONS / EVIDENCE_STATE_OPTIONS).
// noul = independent propositions: each id gets its own probability, no
// cross-id winner. score = ordinal judgments.
// Raw vs policy: parsers return raw probabilities/choices/scores unchanged;
// normalizeProbability applies the separate noul policy. Parsers keep raw values.
// Lifecycle vs decision: stage/proposal admission (which nodes exist, which
// stage they belong to) is lifecycle; disposition/truth/evidenceState/
// materiality are the semantic decisions JEV owns per node.
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

export type Decision = "yes" | "no" | "unknown";
export type NormalizedDecision = {
  outcome: Decision;
  probability: number | null;
  strength: "confident" | "tentative" | null;
};

// Support at .50 is affirmative. Strength describes affirmative support only.
export function normalizeProbability(value: unknown): NormalizedDecision {
  if (!isProb(value)) return { outcome: "unknown", probability: null, strength: null };
  if (value < 0.5) return { outcome: "no", probability: value, strength: null };
  return { outcome: "yes", probability: value, strength: value >= 0.7 ? "confident" : "tentative" };
}

export function classifyProbability(value: unknown): Decision {
  if (!isProb(value)) return "unknown";
  return value >= 0.5 ? "yes" : "no";
}

export type NoulDecision = { kind: "noul"; probability: number };
export type ChoiceDecision = {
  kind: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
};
export type ScoreDecision = {
  kind: "score";
  score: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  raw?: unknown;
};
export type DecisionResult = NoulDecision | ChoiceDecision | ScoreDecision;

export const DISPOSITION_OPTIONS: Record<"analyze" | "gather_evidence" | "reject", string> = {
  analyze: "The question materially contributes to resolving the objective and is ready to analyze.",
  gather_evidence: "The question matters, but available state is insufficient to analyze it.",
  reject: "The question does not materially contribute to resolving the user's objective.",
};

export const EVIDENCE_STATE_OPTIONS: Record<
  "sufficient_support" | "sufficient_contradiction" | "conflicted" | "insufficient",
  string
> = {
  sufficient_support: "The cited evidence sufficiently supports the claim.",
  sufficient_contradiction: "The cited evidence sufficiently contradicts the claim.",
  conflicted: "The evidence both supports and contradicts the claim; do not resolve by guessing.",
  insufficient: "The evidence is missing or too weak to support or contradict the claim.",
};
// Ordinal materiality rubric for score questions.
export const MATERIALITY_LEVELS = ["Immaterial", "Low", "Moderate", "High", "Critical"] as const;

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isProb(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
}

export function parseNoulAnswer(name: string, id: string, ans: unknown): NoulDecision {
  if (!isObj(ans) || ans.type !== "noul" || typeof ans.noul !== "number")
    throw new Error(`${name}: malformed_decision_answer for ${id}`);
  const probability = ans.noul;
  if (!isProb(probability)) throw new Error("invalid_probability");
  return { kind: "noul", probability };
}

export function parseChoiceAnswer(
  name: string,
  id: string,
  ans: unknown,
  options: Record<string, string>,
): ChoiceDecision {
  const bad = (): Error => new Error(`${name}: malformed_decision_answer for ${id}`);
  if (!isObj(ans) || ans.type !== "choice") throw bad();
  if (typeof ans.choice !== "string" || !Object.hasOwn(options, ans.choice)) throw bad();
  if (!isObj(ans.probabilities) || typeof ans.confidence !== "number") throw bad();
  if (!isProb(ans.confidence)) throw bad();
  const probabilities = choiceProbabilities(ans.probabilities, options, bad);
  return { kind: "choice", choice: ans.choice as string, probabilities, confidence: ans.confidence as number };
}

function choiceProbabilities(
  source: Record<string, unknown>,
  options: Record<string, string>,
  bad: () => Error,
): Record<string, number> {
  const want = Object.keys(options).sort();
  const got = Object.keys(source).sort();
  if (got.length !== want.length || got.some((k, i) => k !== want[i])) throw bad();
  const probabilities: Record<string, number> = {};
  for (const k of want) {
    const v = source[k];
    if (!isProb(v)) throw bad();
    Object.defineProperty(probabilities, k, {
      value: v, enumerable: true, writable: true, configurable: true,
    });
  }
  return probabilities;
}

function scoreValue(ans: unknown, maxScore: number | undefined, bad: () => Error): Record<string, unknown> & { score: number } {
  if (!isObj(ans) || ans.type !== "score" || typeof ans.score !== "number") throw bad();
  if (!Number.isFinite(ans.score)) throw bad();
  if (maxScore !== undefined) {
    if (!Number.isFinite(maxScore) || ans.score < 0 || ans.score > maxScore) throw bad();
  }
  return ans as Record<string, unknown> & { score: number };
}

function scoreProbabilities(value: unknown, bad: () => Error): Record<string, number> {
  if (!isObj(value)) throw bad();
  const probs: Record<string, number> = {};
  for (const [k, v] of Object.entries(value)) {
    if (!isProb(v)) throw bad();
    probs[k] = v;
  }
  return probs;
}

export function parseScoreAnswer(name: string, id: string, answer: unknown, maxScore?: number): ScoreDecision {
  const bad = (): Error => new Error(`${name}: malformed_decision_answer for ${id}`);
  const ans = scoreValue(answer, maxScore, bad);
  const s = ans.score;
  const out: ScoreDecision = { kind: "score", score: s };
  if (ans.confidence !== undefined) {
    if (!isProb(ans.confidence)) throw bad();
    out.confidence = ans.confidence as number;
  }
  if (ans.probabilities !== undefined) {
    out.probabilities = scoreProbabilities(ans.probabilities, bad);
  }
  if ("legend" in ans) out.raw = ans.legend;
  return out;
}

export function assertAcyclic(
  stage: string,
  proposals: { id: string; dependsOn: string[] }[],
  prior: (id: string) => string[] | undefined,
): void {
  const edges = new Map(proposals.map((p) => [p.id, p.dependsOn]));
  for (const p of proposals)
    if (p.dependsOn.includes(p.id)) throw new Error(`${stage}: proposal ${p.id} depends on itself`);
  const state = new Map<string, number>();
  const visit = (id: string): void => {
    const s = state.get(id);
    if (s === 2) return;
    if (s === 1) throw new Error(`${stage}: cyclic dependency involving ${id}`);
    state.set(id, 1);
    const deps = edges.has(id) ? (edges.get(id) as string[]) : (prior(id) ?? []);
    for (const d of deps) visit(d);
    state.set(id, 2);
  };
  for (const p of proposals) visit(p.id);
}

export function scopeNodeState(args: {
  objective: unknown;
  proposal: unknown;
  analysis?: unknown;
  evidence: unknown[];
  dependencyDecisions?: unknown;
}): unknown {
  if (!Array.isArray(args.evidence)) throw new Error("scopeNodeState: evidence must be an array");
  const out: Record<string, unknown> = {
    objective: args.objective,
    proposal: args.proposal,
    evidence: args.evidence,
  };
  if (args.analysis !== undefined) out.analysis = args.analysis;
  if (args.dependencyDecisions !== undefined) out.dependencyDecisions = args.dependencyDecisions;
  return out;
}

export type SystemOneFn = (req: { state: unknown; questions: Record<string, unknown> }) => Promise<unknown>;

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

// Both callers share transport errors and strict envelope parsing. The experiment
// caller still saves the raw response before validation.
export async function requestDecisions(
  name: string,
  req: { state: unknown; questions: Record<string, unknown> },
  systemOne: SystemOneFn,
): Promise<unknown> {
  try {
    return await systemOne(req);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`${name}: ${msg.startsWith("malformed_decision_response") ? msg : `decision_request_failed: ${msg}`}`);
  }
}

function questionDecision(
  name: string,
  id: string,
  q: unknown,
  ans: unknown,
  choiceOptions?: Record<string, string>,
): DecisionResult {
  const qType = isObj(q) ? q.type : undefined;
  const aType = isObj(ans) ? ans.type : undefined;
  const kind = qType === "choice" || qType === "score" ? qType : aType;
  if (kind === "choice") return parseChoiceAnswer(name, id, ans, choiceOptions ?? optionsFromQuestion(q));
  if (kind === "score") return parseScoreAnswer(name, id, ans, scoreMaxFromQuestion(q));
  return parseNoulAnswer(name, id, ans);
}

export function parseDecisions(
  name: string,
  questions: Record<string, unknown>,
  raw: unknown,
  choiceOptions?: Record<string, Record<string, string>>,
): Record<string, DecisionResult> {
  const ids = Object.keys(questions).sort();
  if (!isObj(raw) || !isObj(raw.answers)) throw new Error(`${name}: malformed_decision_response`);
  const got = Object.keys(raw.answers).sort();
  if (got.length !== ids.length || got.some((k, i) => k !== ids[i]))
    throw new Error(`${name}: decision answers do not match questions`);
  const decisions: Record<string, DecisionResult> = {};
  for (const id of ids) {
    decisions[id] = questionDecision(name, id, questions[id], raw.answers[id], choiceOptions?.[id]);
  }
  return decisions;
}

// Save request/raw/policy files. Callers apply noul policy separately.
export async function askDecisions(
  dir: string,
  name: string,
  req: { state: unknown; questions: Record<string, unknown> },
  systemOne: SystemOneFn,
  choiceOptions?: Record<string, Record<string, string>>,
): Promise<{ raw: unknown; decisions: Record<string, DecisionResult> }> {
  await writeFile(join(dir, `request-${name}.json`), JSON.stringify(req, null, 2) + "\n");
  const raw = await requestDecisions(name, req, systemOne);
  await writeFile(join(dir, `raw-${name}.json`), JSON.stringify(raw, null, 2) + "\n");
  const decisions = parseDecisions(name, req.questions, raw, choiceOptions);
  await writeFile(join(dir, `policy-${name}.json`), JSON.stringify(decisions, null, 2) + "\n");
  return { raw, decisions };
}
// Whole-registry tool selection (binding): JEV owns ALL tool selection and
// every transition — including post-tool — over the full canonical registry,
// every decision. Needle is args-only execution: never selects, chains, or
// judges sufficiency. The caller assembles the whole registry every round
// (one entry per canonical research tool, no prefilter); this builder maps
// entries 1:1 to choice options plus the two sentinels below. Choice is
// exclusive (exactly one winner per round); parallel coverage = successive
// JEV rounds, each returning to JEV with the full registry again (Needle must
// NOT chain tools: search_sec_filings result -> JEV -> get_sec_document).
// Winner parsing reuses parseChoiceAnswer; no new parser lives here.
export const TOOL_SELECTION_SENTINELS = {
  reasoning_required: "Escalate to the reasoner (decompose/analyze proposals); no tool call fits this node.",
  node_resolved: "Existing evidence resolves the node; no further tool call needed.",
} as const;

// Compact model-efficient manifest for one canonical tool, assembled
// caller-side (Python scheduler) as one entry per canonical research tool.
// Only name is required; every canonical tool stays visible. Selection
// restrictions live in discovery metadata (useWhen/avoidWhen), so the option
// line carries no description: Needle keeps full descriptions and schemas.
// Mirrors app/decision_client.py _manifest_line field for field.
export type ToolManifestEntry = {
  name: string;
  purpose?: string;
  useWhen?: string;
  avoidWhen?: string;
  prerequisites?: string;
  pitSupport?: string;
};

export type ToolSelectionNode = {
  nodeId: string;
  question: string;
  whyItMatters?: string;
};

export type ToolSelectionAttempt = { tool: string; error: string };
const MANIFEST_FIELDS = [
  ["Purpose", "purpose"],
  ["Use", "useWhen"],
  ["Avoid", "avoidWhen"],
  ["Needs", "prerequisites"],
  ["PIT", "pitSupport"],
] as const;

function manifestLine(entry: ToolManifestEntry): string {
  let line = entry.name;
  for (const [label, key] of MANIFEST_FIELDS) {
    const value = entry[key];
    if (typeof value === "string" && value.trim()) line += ` ${label}: ${value.replace(/\s+/g, " ").trim()}.`;
  }
  return line;
}

function evidenceIds(evidence: unknown): string[] {
  if (!Array.isArray(evidence)) return [];
  const out: string[] = [];
  for (const e of evidence) {
    if (typeof e === "object" && e !== null && "id" in e && typeof (e as Record<string, unknown>).id !== "undefined")
      out.push(String((e as Record<string, unknown>).id));
    if (out.length >= 10) break;
  }
  return out;
}

function withTodayUtc(text: string): string {
  return text.includes("[Today UTC ") ? text : `[Today UTC ${new Date().toISOString().slice(0, 10)}] ${text}`;
}

function toolSelectionOptions(registry: ToolManifestEntry[]): Record<string, string> {
  const options: Record<string, string> = {};
  for (const entry of registry) {
    if (!entry || typeof entry.name !== "string" || !entry.name)
      throw new Error("tool_selection: registry entry needs a name");
    if (entry.name in options) throw new Error(`tool_selection: duplicate tool ${entry.name}`);
    options[entry.name] = manifestLine(entry);
  }
  for (const [k, v] of Object.entries(TOOL_SELECTION_SENTINELS)) {
    if (k in options) throw new Error(`tool_selection: registry collides with sentinel ${k}`);
    options[k] = v;
  }
  return options;
}

function attemptLines(attempts?: ToolSelectionAttempt[]): string {
  let text = "";
  for (const a of attempts ?? []) {
    if (!a || typeof a.tool !== "string" || typeof a.error !== "string") continue;
    text += ` Prior attempt ${a.tool} failed: ${a.error.slice(0, 200)}.`;
  }
  return text;
}

function toolSelectionContext(node: ToolSelectionNode, evidence?: unknown[], attempts?: ToolSelectionAttempt[]): string {
  let prompt = `Which single tool runs next for research node ${node.nodeId}? Question: ${node.question} Choose exactly one winner.`;
  if (typeof node.whyItMatters === "string" && node.whyItMatters) prompt += ` Why it matters: ${node.whyItMatters}`;
  const ids = evidenceIds(evidence);
  prompt += ` Evidence on hand: ${Array.isArray(evidence) ? evidence.length : 0} item(s)${ids.length ? ` [${ids.join(", ")}]` : ""}.`;
  return prompt + attemptLines(attempts);
}

export function buildToolSelectionQuestion(
  registry: ToolManifestEntry[],
  node: ToolSelectionNode,
  evidence?: unknown[],
  attempts?: ToolSelectionAttempt[],
): { id: "tool_selection"; prompt: string; options: Record<string, string> } {
  if (!Array.isArray(registry) || registry.length === 0) throw new Error("tool_selection: empty registry");
  if (!node || typeof node.nodeId !== "string" || !node.nodeId || typeof node.question !== "string" || !node.question)
    throw new Error("tool_selection: node needs nodeId and question");
  const options = toolSelectionOptions(registry);
  let prompt = toolSelectionContext(node, evidence, attempts);
  prompt += ` Today UTC is ${new Date().toISOString().slice(0, 10)}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week' or 'today' or 'last quarter' as arg values.`;
  return { id: "tool_selection", prompt: withTodayUtc(prompt), options };
}
