// decision/jev.ts — typed JEV decision layer (API-only, no OMP/Pi runtime).
//
// choice = exclusive routing/category/explanation: exactly one winner among
// mutually exclusive labels (use DISPOSITION_OPTIONS / EVIDENCE_STATE_OPTIONS).
// noul = independent propositions: each id gets its own probability, no
// cross-id winner. score = ordinal judgments only, never deterministic calcs.
// Raw vs policy: parsers return raw probabilities/choices/scores unchanged;
// classifyProbability is the separate policy step callers apply to noul
// probabilities (yes/unsure/no), never collapsed inside a parser.
// Lifecycle vs decision: stage/proposal admission (which nodes exist, which
// stage they belong to) is lifecycle; disposition/truth/evidenceState/
// materiality are the semantic decisions JEV owns per node.
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

export type Decision = "yes" | "unsure" | "no";

// Approved policy: >=.70 yes; >=.50 and <.70 unsure; <.50 no.
export function classifyProbability(p: number): Decision {
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error("invalid_probability");
  if (p >= 0.7) return "yes";
  if (p >= 0.5) return "unsure";
  return "no";
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
// Ordinal materiality rubric for score questions (never deterministic calcs).
export const MATERIALITY_LEVELS = ["Immaterial", "Low", "Moderate", "High", "Critical"] as const;

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isProb(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
}

export function parseNoulAnswer(name: string, id: string, ans: unknown): NoulDecision {
  if (!isObj(ans) || ans.type !== "noul" || typeof ans.noul !== "number")
    throw new Error(`${name}: malformed_typesafe_answer for ${id}`);
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
  const bad = (): Error => new Error(`${name}: malformed_typesafe_answer for ${id}`);
  if (!isObj(ans) || ans.type !== "choice") throw bad();
  if (typeof ans.choice !== "string" || !(ans.choice in options)) throw bad();
  if (!isObj(ans.probabilities) || typeof ans.confidence !== "number") throw bad();
  if (!isProb(ans.confidence)) throw bad();
  const want = Object.keys(options).sort();
  const got = Object.keys(ans.probabilities).sort();
  if (got.length !== want.length || got.some((k, i) => k !== want[i])) throw bad();
  const probabilities: Record<string, number> = {};
  for (const k of want) {
    const v = (ans.probabilities as Record<string, unknown>)[k];
    if (!isProb(v)) throw bad();
    probabilities[k] = v;
  }
  return { kind: "choice", choice: ans.choice as string, probabilities, confidence: ans.confidence as number };
}

export function parseScoreAnswer(name: string, id: string, ans: unknown, maxScore?: number): ScoreDecision {
  const bad = (): Error => new Error(`${name}: malformed_typesafe_answer for ${id}`);
  if (!isObj(ans) || ans.type !== "score" || typeof ans.score !== "number") throw bad();
  const s = ans.score as number;
  if (!Number.isFinite(s)) throw bad();
  if (maxScore !== undefined) {
    if (!Number.isFinite(maxScore) || s < 0 || s > maxScore) throw bad();
  }
  const out: ScoreDecision = { kind: "score", score: s };
  if (ans.confidence !== undefined) {
    if (!isProb(ans.confidence)) throw bad();
    out.confidence = ans.confidence as number;
  }
  if (ans.probabilities !== undefined) {
    if (!isObj(ans.probabilities)) throw bad();
    const probs: Record<string, number> = {};
    for (const [k, v] of Object.entries(ans.probabilities)) {
      if (!isProb(v)) throw bad();
      probs[k] = v;
    }
    out.probabilities = probs;
  }
  if (isObj(ans) && "legend" in ans) out.raw = ans.legend;
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

// Typed ask: preserves raw SDK output plus the parsed DecisionResult map, and
// writes request/raw/policy files like the existing sequential caller. Policy
// (the decisions map) stays separate from raw; noul yes/unsure/no is derived
// by callers via classifyProbability, never collapsed here.
export async function askDecisions(
  dir: string,
  name: string,
  req: { state: unknown; questions: Record<string, unknown> },
  systemOne: SystemOneFn,
  choiceOptions?: Record<string, Record<string, string>>,
): Promise<{ raw: unknown; decisions: Record<string, DecisionResult> }> {
  await writeFile(join(dir, `request-${name}.json`), JSON.stringify(req, null, 2) + "\n");
  let raw: unknown;
  try {
    raw = await systemOne(req);
  } catch (e) {
    throw new Error(`${name}: typesafe_request_failed: ${e instanceof Error ? e.message : String(e)}`);
  }
  await writeFile(join(dir, `raw-${name}.json`), JSON.stringify(raw, null, 2) + "\n");
  const ids = Object.keys(req.questions).sort();
  if (!isObj(raw) || !isObj(raw.answers)) throw new Error(`${name}: malformed_typesafe_response`);
  const got = Object.keys(raw.answers).sort();
  if (got.length !== ids.length || got.some((k, i) => k !== ids[i]))
    throw new Error(`${name}: typesafe answers do not match questions`);
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
// name+description required (compat); all other fields optional budget aids
// supplied by the scheduler's build_registry — never a filter, every
// canonical tool stays visible. Builder renders one compact single line per
// tool from all present fields.
export type ToolManifestEntry = {
  name: string;
  description: string;
  domain?: string;
  purpose?: string;
  intent?: string;
  keyInputs?: string;
  outputKind?: string;
  evidence?: string;
  prerequisites?: string;
  pitSupport?: string;
  useWhen?: string;
  avoidWhen?: string;
  conflicts?: string;
  nextTools?: string;
};

export type ToolSelectionNode = {
  nodeId: string;
  question: string;
  whyItMatters?: string;
};

export type ToolSelectionAttempt = { tool: string; error: string };
function manifestLine(entry: ToolManifestEntry): string {
  const opt = (v: unknown): string | null =>
    typeof v === "string" && v.trim() ? v.replace(/\s+/g, " ").trim() : null;
  let line = entry.description.replace(/\s+/g, " ").trim();
  const purpose = opt(entry.purpose);
  if (purpose && purpose !== line) line += ` Purpose: ${purpose}.`;
  const intent = opt(entry.intent);
  if (intent) line += ` Intent: ${intent}.`;
  const inputs = opt(entry.keyInputs);
  if (inputs) line += ` Inputs: ${inputs}.`;
  const output = opt(entry.outputKind);
  if (output) line += ` Output: ${output}.`;
  const evidence = opt(entry.evidence);
  if (evidence) line += ` Evidence: ${evidence}.`;
  const prereq = opt(entry.prerequisites);
  if (prereq) line += ` Needs: ${prereq}.`;
  const pit = opt(entry.pitSupport);
  if (pit) line += ` PIT: ${pit}.`;
  const useWhen = opt(entry.useWhen);
  if (useWhen) line += ` Use: ${useWhen}.`;
  const avoidWhen = opt(entry.avoidWhen);
  if (avoidWhen) line += ` Avoid: ${avoidWhen}.`;
  const conflicts = opt(entry.conflicts);
  if (conflicts) line += ` Conflicts: ${conflicts}.`;
  const nextTools = opt(entry.nextTools);
  if (nextTools) line += ` Next: ${nextTools}.`;
  const domain = opt(entry.domain);
  if (domain) line = `[${domain}] ${line}`;
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

export function buildToolSelectionQuestion(
  registry: ToolManifestEntry[],
  node: ToolSelectionNode,
  evidence?: unknown[],
  attempts?: ToolSelectionAttempt[],
): { id: "tool_selection"; prompt: string; options: Record<string, string> } {
  if (!Array.isArray(registry) || registry.length === 0) throw new Error("tool_selection: empty registry");
  if (!node || typeof node.nodeId !== "string" || !node.nodeId || typeof node.question !== "string" || !node.question)
    throw new Error("tool_selection: node needs nodeId and question");
  const options: Record<string, string> = {};
  for (const entry of registry) {
    if (!entry || typeof entry.name !== "string" || !entry.name)
      throw new Error("tool_selection: registry entry needs a name");
    if (typeof entry.description !== "string" || !entry.description)
      throw new Error(`tool_selection: tool ${entry.name} needs a description`);
    if (entry.name in options) throw new Error(`tool_selection: duplicate tool ${entry.name}`);
    options[entry.name] = manifestLine(entry);
  }
  for (const [k, v] of Object.entries(TOOL_SELECTION_SENTINELS)) {
    if (k in options) throw new Error(`tool_selection: registry collides with sentinel ${k}`);
    options[k] = v;
  }
  let prompt = `Which single tool runs next for research node ${node.nodeId}? Question: ${node.question} JEV owns this selection and every transition over the whole canonical registry; Needle runs args only and never selects, chains, or judges. Choose exactly one winner.`;
  if (typeof node.whyItMatters === "string" && node.whyItMatters) prompt += ` Why it matters: ${node.whyItMatters}`;
  const ids = evidenceIds(evidence);
  prompt += ` Evidence on hand: ${Array.isArray(evidence) ? evidence.length : 0} item(s)${ids.length ? ` [${ids.join(", ")}]` : ""}.`;
  for (const a of attempts ?? []) {
    if (!a || typeof a.tool !== "string" || typeof a.error !== "string") continue;
    prompt += ` Prior attempt ${a.tool} failed: ${a.error.slice(0, 200)}.`;
  }
  prompt += ` Today UTC is ${new Date().toISOString().slice(0, 10)}; decode relative dates before choosing: 'last quarter filing' = latest 10-Q/10-K/8-K with no start/end window, 'this week'/'last week' = Monday-now NYC range (one YYYY-MM-DD per biz day, latest first); 'today'/'now' = Today UTC date; never pass phrases like 'this week' or 'today' or 'last quarter' as arg values.`;
  prompt += ` Search packets list candidates only — page with research_read_search beyond display_limit and open each accession via get_sec_filing/get_sec_document until answered or guard trips.`;
  if (/revenue/i.test(node.question) && /quarter/i.test(node.question)) prompt += ` Revenue-quarter answers come from the 10-Q (MD&A/segment revenue), not the 8-K: after the filing list, open the 10-Q accession via get_sec_document with a revenue query before re-reading any 8-K.`;
  return { id: "tool_selection", prompt: withTodayUtc(prompt), options };
}
