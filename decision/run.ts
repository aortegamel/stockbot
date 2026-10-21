// decision/run.ts — minimal sequential caller for the API-only prompt-graph experiment.
//
// Owns: OpenCode Responses calls, JEV typed relevance + adjudication,
// strict contract validation, per-run artifacts under decision/results.
// Consumes sibling-owned decision/scenarios.ts + decision/prompts.ts +
// decision/jev.ts; evaluationCriteria is never sent to any provider.
//
// Governing rule: USER objective / REASONER thinks / RESEARCH observes / CODE
// calculates / JEV decides / GRAPH remembers. OpenCode proposes and interprets
// only; JEV answers the actual DecisionNode via noul/choice/score.
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createSystemOne } from "./client.ts";
import { reasoningFields } from "./reasoning.ts";
import { scenarios } from "./scenarios.ts";
import { analyzePrompt, decomposePrompt, expandPrompt } from "./prompts.ts";
import type { Analysis, EvidenceRequest, Proposal } from "./prompts.ts";
import {
  EVIDENCE_STATE_OPTIONS,
  MATERIALITY_LEVELS,
  askDecisions,
  assertAcyclic,
  classifyProbability,
  scopeNodeState,
  DISPOSITION_OPTIONS,
} from "./jev.ts";
import type {
  ChoiceDecision,
  Decision,
  DecisionResult,
  NoulDecision,
  ScoreDecision,
  SystemOneFn,
} from "./jev.ts";

export { classifyProbability };
export type { ChoiceDecision, Decision, DecisionResult, NoulDecision, ScoreDecision, SystemOneFn };

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}
const RESULTS_ROOT = "decision/results";

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function exactKeys(stage: string, o: Record<string, unknown>, keys: string[]): void {
  const actual = Object.keys(o).sort();
  if (actual.length !== keys.length || actual.some((k, i) => k !== keys[i]))
    throw new Error(`${stage}: unexpected fields [${actual.join(",")}]`);
}

function nonEmpty(stage: string, v: unknown, what: string): string {
  if (typeof v !== "string" || v.length === 0) throw new Error(`${stage}: ${what} must be a nonempty string`);
  return v;
}

function proposalId(stage: string, item: unknown, objectiveId: string): string {
  if (!isObj(item)) throw new Error(`${stage}: proposal must be an object`);
  exactKeys(stage, item, ["dependsOn", "id", "objectiveId", "question", "whyItMatters"]);
  const id = nonEmpty(stage, item.id, "proposal.id");
  if (!id.startsWith(`${objectiveId}-`)) throw new Error(`${stage}: proposal id ${id} must start with ${objectiveId}-`);
  if (item.objectiveId !== objectiveId) throw new Error(`${stage}: proposal ${id} references wrong objective`);
  nonEmpty(stage, item.question, "proposal.question");
  nonEmpty(stage, item.whyItMatters, "proposal.whyItMatters");
  return id;
}

function checkDependencies(stage: string, item: Record<string, unknown>, refs: Set<string>): void {
  const id = item.id as string;
  if (!Array.isArray(item.dependsOn) || item.dependsOn.some((d) => typeof d !== "string"))
    throw new Error(`${stage}: proposal ${id} dependsOn must be string[]`);
  for (const d of item.dependsOn as string[]) {
    if (d === id) throw new Error(`${stage}: proposal ${id} depends on itself`);
    if (!refs.has(d)) throw new Error(`${stage}: proposal ${id} references unknown id ${d}`);
  }
}

// Proposed IDs are objective-scoped, nonempty, unique; deps reference existing or
// co-proposed IDs, never self. Expand stage additionally rejects IDs reusing old ones.
function checkProposals(stage: string, value: unknown, objectiveId: string, priorIds: Set<string>): Proposal[] {
  if (!Array.isArray(value)) throw new Error(`${stage}: proposals must be an array`);
  const ids = new Set<string>();
  for (const item of value) {
    const id = proposalId(stage, item, objectiveId);
    if (ids.has(id)) throw new Error(`${stage}: duplicate proposal id ${id}`);
    if (priorIds.has(id)) throw new Error(`${stage}: reused proposal id ${id}`);
    ids.add(id);
  }
  const refs = new Set([...priorIds, ...ids]);
  for (const item of value) {
    checkDependencies(stage, item as Record<string, unknown>, refs);
  }
  return value as Proposal[];
}

function analysisFields(stage: string, item: unknown): Record<string, unknown> {
  if (!isObj(item)) throw new Error(`${stage}: analysis must be an object`);
  const names = Object.keys(item).sort();
  for (const k of ["evidenceRefs", "interpretation", "nodeId", "objectiveId"])
    if (!names.includes(k)) throw new Error(`${stage}: unexpected fields [${names.join(",")}]`);
  for (const k of names)
    if (!["assumptions", "evidenceRefs", "interpretation", "nodeId", "numbers", "objectiveId"].includes(k))
      throw new Error(`${stage}: unexpected fields [${names.join(",")}]`);
  return item;
}

function checkAnalysisEvidence(stage: string, item: Record<string, unknown>, nodeId: string, evidenceIds: Set<string>): Set<string> {
  if (!Array.isArray(item.evidenceRefs) || item.evidenceRefs.some((e) => typeof e !== "string"))
    throw new Error(`${stage}: analysis ${nodeId} evidenceRefs must be string[]`);
  for (const e of item.evidenceRefs as string[])
    if (!evidenceIds.has(e)) throw new Error(`${stage}: analysis ${nodeId} references unknown evidence ${e}`);
  return new Set(item.evidenceRefs as string[]);
}

function checkAnalysisNumbers(stage: string, value: unknown, nodeId: string, refSet: Set<string>): void {
  if (value === undefined) return;
  if (!Array.isArray(value)) throw new Error(`${stage}: analysis ${nodeId} numbers must be an array`);
  for (const n of value) {
    if (!isObj(n)) throw new Error(`${stage}: analysis ${nodeId} number must be an object`);
    exactKeys(stage, n, ["evidenceId", "quote", "value"]);
    nonEmpty(stage, n.value, "analysis.number.value");
    const eid = nonEmpty(stage, n.evidenceId, "analysis.number.evidenceId");
    nonEmpty(stage, n.quote, "analysis.number.quote");
    if (!refSet.has(eid)) throw new Error(`${stage}: analysis ${nodeId} number references unknown evidence ${eid}`);
  }
}

function checkAnalysisAssumptions(stage: string, value: unknown, nodeId: string): void {
  if (value === undefined) return;
  if (!Array.isArray(value)) throw new Error(`${stage}: analysis ${nodeId} assumptions must be an array`);
  const seen = new Set<string>();
  for (const a of value) {
    if (!isObj(a)) throw new Error(`${stage}: analysis ${nodeId} assumption must be an object`);
    exactKeys(stage, a, ["assumptionId", "text"]);
    const aid = nonEmpty(stage, a.assumptionId, "analysis.assumption.assumptionId");
    nonEmpty(stage, a.text, "analysis.assumption.text");
    if (seen.has(aid)) throw new Error(`${stage}: analysis ${nodeId} duplicate assumptionId ${aid}`);
    seen.add(aid);
  }
}

function checkAnalyses(
  stage: string,
  value: unknown,
  objectiveId: string,
  proposalIds: Set<string>,
  evidenceIds: Set<string>,
): Analysis[] {
  if (!Array.isArray(value)) throw new Error(`${stage}: analyses must be an array`);
  const seen = new Set<string>();
  for (const valueItem of value) {
    const item = analysisFields(stage, valueItem);
    const nodeId = nonEmpty(stage, item.nodeId, "analysis.nodeId");
    if (!proposalIds.has(nodeId)) throw new Error(`${stage}: analysis references unknown proposal ${nodeId}`);
    if (item.objectiveId !== objectiveId) throw new Error(`${stage}: analysis ${nodeId} references wrong objective`);
    nonEmpty(stage, item.interpretation, "analysis.interpretation");
    const refSet = checkAnalysisEvidence(stage, item, nodeId, evidenceIds);
    checkAnalysisNumbers(stage, item.numbers, nodeId, refSet);
    checkAnalysisAssumptions(stage, item.assumptions, nodeId);
    if (seen.has(nodeId)) throw new Error(`${stage}: duplicate analysis for ${nodeId}`);
    seen.add(nodeId);
  }
  return value as Analysis[];
}

function checkEvidenceRequests(
  stage: string,
  value: unknown,
  objectiveId: string,
  nodeIds: Set<string>,
): EvidenceRequest[] {
  if (!Array.isArray(value)) throw new Error(`${stage}: evidenceRequests must be an array`);
  for (const item of value) {
    if (!isObj(item)) throw new Error(`${stage}: evidenceRequest must be an object`);
    exactKeys(stage, item, ["missingEvidence", "nodeId", "objectiveId"]);
    const nodeId = nonEmpty(stage, item.nodeId, "evidenceRequest.nodeId");
    if (!nodeIds.has(nodeId)) throw new Error(`${stage}: evidenceRequest references unknown node ${nodeId}`);
    if (item.objectiveId !== objectiveId)
      throw new Error(`${stage}: evidenceRequest for ${nodeId} references wrong objective`);
    nonEmpty(stage, item.missingEvidence, "evidenceRequest.missingEvidence");
  }
  return value as EvidenceRequest[];
}

function openCodeText(stage: string, raw: unknown): string {
  const output = isObj(raw) ? raw.output : undefined;
  if (!Array.isArray(output)) throw new Error(`${stage}: malformed_opencode_response`);
  let text = "";
  for (const item of output) {
    if (!isObj(item) || item.type !== "message" || !Array.isArray(item.content)) continue;
    text += outputMessageText(item.content);
  }
  return text;
}

function outputMessageText(content: unknown[]): string {
  let text = "";
  for (const part of content)
    if (isObj(part) && part.type === "output_text" && typeof part.text === "string") text += part.text;
  return text;
}

// Concatenate output_text before validating the exact Responses contract.
function parseOpenCodeOutput(stage: string, raw: unknown, keys: string[]): Record<string, unknown> {
  const text = openCodeText(stage, raw);
  if (!text) throw new Error(`${stage}: malformed_opencode_response`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(`${stage}: malformed_opencode_json`);
  }
  if (!isObj(parsed)) throw new Error(`${stage}: malformed_opencode_json`);
  exactKeys(stage, parsed, keys);
  return parsed;
}

// POST {model, input} — no tools, no streaming. Only the body is saved; the
// Authorization header is never written to disk. Any failure throws, so a failed
// request can never report a successful chain.
async function postOpenCode(
  dir: string,
  name: string,
  input: string,
  model: string,
  apiKey: string,
  fetchFn: typeof fetch,
  url: string,
): Promise<unknown> {
  const body = { model, input, ...reasoningFields("responses") };
  await writeFile(join(dir, `request-${name}.json`), JSON.stringify(body, null, 2) + "\n");
  let res: Response;
  try {
    res = await fetchFn(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}`, "x-opencode-session": crypto.randomUUID() },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error(`${name}: opencode_request_failed: ${e instanceof Error ? e.message : String(e)}`);
  }
  const text = await res.text();
  if (!res.ok) throw new Error(`${name}: opencode_request_failed: ${res.status} ${text.slice(0, 300)}`);
  const raw: unknown = JSON.parse(text);
  await writeFile(join(dir, `raw-${name}.json`), JSON.stringify(raw, null, 2) + "\n");
  return raw;
}

function promptText(stage: string, build: (ctx: unknown) => string, ctx: unknown): string {
  const text = build(ctx);
  if (typeof text !== "string" || text.length === 0) throw new Error(`${stage}: malformed_prompt_builder`);
  return text;
}

function slug(id: string): string {
  return id.replace(/[^A-Za-z0-9]+/g, "_").slice(0, 80);
}

type Disposition = "analyze" | "gather_evidence" | "reject";

function dispositionOf(decisions: Record<string, DecisionResult>, id: string): Disposition {
  const d = decisions[id];
  if (!d || d.kind !== "choice") throw new Error(`relevance: missing disposition for ${id}`);
  const c = (d as ChoiceDecision).choice;
  if (c !== "analyze" && c !== "gather_evidence" && c !== "reject")
    throw new Error(`relevance: unknown disposition ${c} for ${id}`);
  return c;
}

function checkAnalysisAccounting(
  analyses: Analysis[],
  analyzeOnly: Proposal[],
  gatherIds: Set<string>,
  analysisIds: Set<string>,
  requestIds: Set<string>,
): void {
  for (const a of analyses)
    if (gatherIds.has(a.nodeId))
      throw new Error(`analyze: gather_evidence proposal ${a.nodeId} must not be analyzed`);
  for (const p of analyzeOnly) {
    const hasA = analysisIds.has(p.id);
    const hasR = requestIds.has(p.id);
    if (hasA === hasR)
      throw new Error(`analyze: proposal ${p.id} with disposition analyze requires exactly one of Analysis or EvidenceRequest`);
  }
  for (const id of gatherIds)
    if (!requestIds.has(id)) throw new Error(`analyze: missing evidence request for gather_evidence proposal ${id}`);
}

function checkedNodeDecisions(nodeId: string, decisions: Record<string, DecisionResult>) {
  const truth = decisions.truth;
  const evidenceState = decisions.evidence_state;
  const materiality = decisions.materiality;
  if (!truth || truth.kind !== "noul") throw new Error(`decide: missing truth noul for ${nodeId}`);
  if (!evidenceState || evidenceState.kind !== "choice") throw new Error(`decide: missing evidence_state choice for ${nodeId}`);
  if (!materiality || materiality.kind !== "score") throw new Error(`decide: missing materiality score for ${nodeId}`);
  return { truth, evidenceState, materiality };
}

// One relevance round: mutually exclusive disposition choice per proposal.
async function dispositionRound(
  dir: string,
  name: string,
  kind: "proposed" | "expanded",
  state: { objective: unknown; fictional: true; proposals: Proposal[] },
  systemOne: SystemOneFn,
): Promise<{ raw: unknown; decisions: Record<string, DecisionResult> }> {
  const { proposals } = state;
  if (proposals.length === 0) return { raw: null, decisions: {} };
  const questions: Record<string, unknown> = {};
  for (const p of proposals)
    questions[p.id] = {
      type: "choice",
      instructions: `What should happen to this ${kind} question relative to the user's objective? Question: ${p.question}`,
      criteria: DISPOSITION_OPTIONS,
    };
  return askDecisions(
    dir, name, { state, questions }, systemOne,
    Object.fromEntries(proposals.map((p) => [p.id, DISPOSITION_OPTIONS])),
  );
}

type NodeAdjudication = {
  nodeId: string;
  truth: NoulDecision;
  truthPolicy: Decision;
  evidenceState: ChoiceDecision;
  materiality: ScoreDecision;
};

// Each analyzed node is adjudicated against its own evidence subset only:
// truth noul (actual proposition) + evidence-state choice + materiality score.
async function adjudicateAnalyses(args: {
  dir: string;
  objective: unknown;
  fictional: true;
  proposals: Proposal[];
  analyses: Analysis[];
  evidence: { id: string; text: string }[];
  relevance: Record<string, DecisionResult>;
  systemOne: SystemOneFn;
}): Promise<{
  raw: Record<string, unknown>;
  decisions: Record<string, Record<string, DecisionResult>>;
  results: NodeAdjudication[];
}> {
  const proposalById = new Map(args.proposals.map((p) => [p.id, p]));
  const evidenceById = new Map(args.evidence.map((e) => [e.id, e]));
  const raw: Record<string, unknown> = {};
  const decisions: Record<string, Record<string, DecisionResult>> = {};
  const results: NodeAdjudication[] = [];
  for (const a of args.analyses) {
    const p = proposalById.get(a.nodeId);
    if (!p) throw new Error(`decide: unknown proposal ${a.nodeId}`);
    const subset = a.evidenceRefs.map((id) => {
      const e = evidenceById.get(id);
      if (!e) throw new Error(`decide: unknown evidence ${id} for ${a.nodeId}`);
      return e;
    });
    const dependencyDecisions = p.dependsOn.map((d) => ({
      proposalId: d,
      disposition: args.relevance[d]?.kind === "choice" ? (args.relevance[d] as ChoiceDecision).choice : "unknown",
    }));
    const state = scopeNodeState({
      objective: args.objective, proposal: p, analysis: a, evidence: subset, dependencyDecisions,
    });
    const questions: Record<string, unknown> = {
      truth: { type: "noul", instructions: `Do the available facts support that ${p.question}` },
      evidence_state: {
        type: "choice",
        instructions: "What best characterizes the evidence state for this node?",
        criteria: EVIDENCE_STATE_OPTIONS,
      },
      materiality: {
        type: "score",
        instructions: "How material is resolving this uncertainty to answering the user's objective?",
        criteria: MATERIALITY_LEVELS,
      },
    };
    const r = await askDecisions(args.dir, `decide-${slug(a.nodeId)}`, { state, questions }, args.systemOne, {
      evidence_state: EVIDENCE_STATE_OPTIONS,
    });
    const { truth, evidenceState, materiality } = checkedNodeDecisions(a.nodeId, r.decisions);
    raw[a.nodeId] = r.raw;
    decisions[a.nodeId] = r.decisions;
    results.push({
      nodeId: a.nodeId, truth, truthPolicy: classifyProbability(truth.probability), evidenceState, materiality,
    });
  }
  return { raw, decisions, results };
}

export async function runScenario(
  scenarioId: string,
  opts?: { fetchFn?: typeof fetch; systemOne?: SystemOneFn; outDir?: string; model?: string },
): Promise<string> {
  const scenario = scenarios.find((s) => s.id === scenarioId);
  if (!scenario) throw new Error(`unknown_scenario: ${scenarioId} (known: ${scenarios.map((s) => s.id).join(", ")})`);
  const model = opts?.model ?? requiredEnv("OPENCODE_MODEL");
  const url = requiredEnv("OPENCODE_URL");
  const apiKey = requiredEnv("OPENCODE_API_KEY");
  const fetchFn = opts?.fetchFn ?? fetch;
  const systemOne = opts?.systemOne ?? createSystemOne();
  const dir = opts?.outDir ?? join(RESULTS_ROOT, `${scenario.id}-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  await mkdir(dir, { recursive: true });

  const objectiveId = scenario.objective.id;
  const evidenceIds = new Set(scenario.evidence.map((e) => e.id));
  const base = { objective: scenario.objective, fictional: scenario.fictional, evidence: scenario.evidence };

  // 1. decompose (reasoner proposes, never decides)
  const decomposeRaw = await postOpenCode(
    dir, "decompose", promptText("decompose", decomposePrompt, base), model, apiKey, fetchFn, url,
  );
  const proposals = checkProposals(
    "decompose", parseOpenCodeOutput("decompose", decomposeRaw, ["proposals"]).proposals, objectiveId, new Set(),
  );
  assertAcyclic("decompose", proposals, () => undefined);

  // 2. disposition of initial proposals via JEV choice (mutually exclusive routing)
  const relevance = await dispositionRound(
    dir, "relevance", "proposed",
    { objective: scenario.objective, fictional: scenario.fictional, proposals },
    systemOne,
  );
  const dispOf = (id: string): Disposition => dispositionOf(relevance.decisions, id);
  const analyzeOnly = proposals.filter((p) => dispOf(p.id) === "analyze");
  const gatherIds = new Set(proposals.filter((p) => dispOf(p.id) === "gather_evidence").map((p) => p.id));
  const relevanceList = proposals.map((p) => {
    const d = relevance.decisions[p.id] as ChoiceDecision;
    return { proposalId: p.id, disposition: dispOf(p.id), probabilities: d.probabilities, confidence: d.confidence };
  });

  // 3. analyze admitted nodes; gather_evidence gets requests only; reject is artifact-only.
  // Accounting invariant: analyze => Analysis XOR EvidenceRequest; gather => request only; reject => neither.
  const analyzeIds = new Set(analyzeOnly.map((p) => p.id));
  const analyzeCtx = {
    ...base, proposals: analyzeOnly, gatherProposalIds: [...gatherIds], relevance: relevanceList, jev: relevance.raw,
    jevResults: relevance.raw, policyResults: relevanceList,
    priorRefs: { evidenceIds: [...evidenceIds], proposalIds: proposals.map((p) => p.id) },
  };
  const analyzeRaw = await postOpenCode(
    dir, "analyze", promptText("analyze", analyzePrompt, analyzeCtx), model, apiKey, fetchFn, url,
  );
  const analyzeOut = parseOpenCodeOutput("analyze", analyzeRaw, ["analyses", "evidenceRequests"]);
  const analyzableIds = new Set([...analyzeIds, ...gatherIds]);
  const analyses = checkAnalyses("analyze", analyzeOut.analyses, objectiveId, analyzableIds, evidenceIds);
  const analyzeRequests = checkEvidenceRequests("analyze", analyzeOut.evidenceRequests, objectiveId, analyzableIds);
  const analysisIds = new Set(analyses.map((a) => a.nodeId));
  const requestIds = new Set(analyzeRequests.map((r) => r.nodeId));
  checkAnalysisAccounting(analyses, analyzeOnly, gatherIds, analysisIds, requestIds);

  // 4. adjudicate each analyzed node against its own scoped evidence only.
  const { raw: decideRaw, decisions: decideAll, results: nodeResults } = await adjudicateAnalyses({
    dir,
    objective: scenario.objective,
    fictional: scenario.fictional,
    proposals,
    analyses,
    evidence: scenario.evidence,
    relevance: relevance.decisions,
    systemOne,
  });
  const unresolved: { nodeId: string; reason: string }[] = [
    ...[...gatherIds].map((nodeId) => ({ nodeId, reason: "gather_evidence" as const })),
    ...analyzeOnly.filter((p) => !analysisIds.has(p.id)).map((p) => ({ nodeId: p.id, reason: "awaiting_evidence" as const })),
    ...nodeResults
      .filter((n) => n.evidenceState.choice === "insufficient" || n.evidenceState.choice === "conflicted")
      .map((n) => ({ nodeId: n.nodeId, reason: n.evidenceState.choice })),
  ];
  const expandCtx = {
    ...base, proposals, relevance: relevanceList, analyses,
    analysisDecisions: nodeResults.map((n) => ({
      nodeId: n.nodeId, truthProbability: n.truth.probability, truthPolicy: n.truthPolicy,
      evidenceState: n.evidenceState.choice, evidenceProbabilities: n.evidenceState.probabilities,
      materialityScore: n.materiality.score,
    })),
    evidenceRequests: analyzeRequests, unresolved, jev: { relevance: relevance.raw, decide: decideRaw },
    adjudication: { relevance: relevanceList, analysisDecisions: nodeResults.map((n) => n.nodeId) },
    priorIds: proposals.map((p) => p.id), priorRefs: { proposalIds: proposals.map((p) => p.id) },
  };
  const expandRaw = await postOpenCode(
    dir, "expand", promptText("expand", expandPrompt, expandCtx), model, apiKey, fetchFn, url,
  );
  const expandOut = parseOpenCodeOutput("expand", expandRaw, ["evidenceRequests", "proposals"]);
  const priorIds = new Set(proposals.map((p) => p.id));
  const expanded = checkProposals("expand", expandOut.proposals, objectiveId, priorIds);
  const priorDeps = new Map(proposals.map((p) => [p.id, p.dependsOn]));
  assertAcyclic("expand", expanded, (id) => priorDeps.get(id));
  const allNodeIds = new Set([...priorIds, ...expanded.map((p) => p.id)]);
  const expandRequests = checkEvidenceRequests("expand", expandOut.evidenceRequests, objectiveId, allNodeIds);

  // 6. disposition of expanded proposals (non-authoritative until JEV admits)
  const expandedRelevance = await dispositionRound(
    dir, "relevance-expanded", "expanded",
    { objective: scenario.objective, fictional: scenario.fictional, proposals: expanded },
    systemOne,
  );

  const withDisposition = (p: Proposal, stage: string, decisions: Record<string, DecisionResult>) => {
    const disposition = dispositionOf(decisions, p.id);
    const d = decisions[p.id] as ChoiceDecision;
    return {
      ...p, stage, disposition, probabilities: d.probabilities, confidence: d.confidence,
    };
  };
  const final = {
    scenarioId: scenario.id,
    model,
    objective: scenario.objective,
    fictional: scenario.fictional,
    evidence: scenario.evidence,
    proposals: [
      ...proposals.map((p) => withDisposition(p, "initial", relevance.decisions)),
      ...expanded.map((p) => withDisposition(p, "expanded", expandedRelevance.decisions)),
    ],
    reasoning: analyses.map((a) => {
      const n = nodeResults.find((r) => r.nodeId === a.nodeId);
      if (!n) throw new Error(`final: missing adjudication for ${a.nodeId}`);
      return {
        ...a, truthProbability: n.truth.probability, truthPolicy: n.truthPolicy,
        evidenceState: n.evidenceState.choice, evidenceProbabilities: n.evidenceState.probabilities,
        evidenceConfidence: n.evidenceState.confidence, materialityScore: n.materiality.score,
      };
    }),
    jev: { relevance: relevance.raw, decide: decideRaw, expandedRelevance: expandedRelevance.raw },
    decisions: { relevance: relevance.decisions, decide: decideAll, expandedRelevance: expandedRelevance.decisions },
    evidenceRequests: [
      ...analyzeRequests.map((r) => ({ ...r, stage: "analyze" })),
      ...expandRequests.map((r) => ({ ...r, stage: "expand" })),
    ],
    unresolved,
  };
  await writeFile(join(dir, "final.json"), JSON.stringify(final, null, 2) + "\n");
  await writeFile(
    join(dir, "evaluation-criteria.json"),
    JSON.stringify({ scenarioId: scenario.id, evaluationCriteria: scenario.evaluationCriteria }, null, 2) + "\n",
  );
  await writeFile(
    join(dir, "eval-context.json"),
    JSON.stringify({
      scenarioId: scenario.id, objective: scenario.objective, fictional: scenario.fictional,
      evidence: scenario.evidence, proposals: [...proposals, ...expanded], analyses,
      evidenceRequests: [...analyzeRequests, ...expandRequests], evaluationCriteria: scenario.evaluationCriteria,
    }, null, 2) + "\n",
  );
  return dir;
}

if (import.meta.main) {
  const args = Bun.argv.slice(2);
  try {
    if (args.includes("--list")) {
      for (const s of scenarios) console.log(`${s.id}\t${s.objective.id}\t${s.objective.prompt}`);
    } else {
      const id = args.find((a) => !a.startsWith("-"));
      if (!id) {
        console.error("usage: bun decision/run.ts --list | bun decision/run.ts <scenario-id>");
        process.exit(2);
      }
      const dir = await runScenario(id);
      console.log(`saved ${dir}/final.json`);
    }
  } catch (e) {
    console.error(e instanceof Error ? e.message : String(e));
    process.exit(1);
  }
}
