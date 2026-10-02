// decision/run.ts — minimal sequential caller for the API-only prompt-graph experiment.
//
// Owns: OpenCode Responses calls, TypeSafe/JEV typed relevance + adjudication,
// strict contract validation, per-run artifacts under decision/results.
// Consumes sibling-owned decision/scenarios.ts + decision/prompts.ts +
// decision/jev.ts; evaluationCriteria is never sent to any provider.
//
// Governing rule: USER objective / REASONER thinks / RESEARCH observes / CODE
// calculates / JEV decides / GRAPH remembers. OpenCode proposes and interprets
// only; JEV answers the actual DecisionNode via noul/choice/score.
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { getTypeSafeClient } from "./client.ts";
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";
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

const OPENCODE_URL = "https://opencode.ai/zen/go/v1/responses";
const DEFAULT_MODEL = "muse-spark-1.3-contributor";
const RESULTS_ROOT = "decision/results";

export function createSystemOne(fetchImpl?: typeof fetch): SystemOneFn {
  return (req) => {
    const client = fetchImpl ? new TypeSafeClient({ fetch: fetchImpl }) : getTypeSafeClient();
    return client.systemOne(req as never, { retry: { maxRetries: 0 } } as never);
  };
}

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

// Proposed IDs are objective-scoped, nonempty, unique; deps reference existing or
// co-proposed IDs, never self. Expand stage additionally rejects IDs reusing old ones.
function checkProposals(stage: string, value: unknown, objectiveId: string, priorIds: Set<string>): Proposal[] {
  if (!Array.isArray(value)) throw new Error(`${stage}: proposals must be an array`);
  const ids = new Set<string>();
  for (const item of value) {
    if (!isObj(item)) throw new Error(`${stage}: proposal must be an object`);
    exactKeys(stage, item, ["dependsOn", "id", "objectiveId", "question", "whyItMatters"]);
    const id = nonEmpty(stage, item.id, "proposal.id");
    if (!id.startsWith(`${objectiveId}-`)) throw new Error(`${stage}: proposal id ${id} must start with ${objectiveId}-`);
    if (item.objectiveId !== objectiveId) throw new Error(`${stage}: proposal ${id} references wrong objective`);
    nonEmpty(stage, item.question, "proposal.question");
    nonEmpty(stage, item.whyItMatters, "proposal.whyItMatters");
    if (ids.has(id)) throw new Error(`${stage}: duplicate proposal id ${id}`);
    if (priorIds.has(id)) throw new Error(`${stage}: reused proposal id ${id}`);
    ids.add(id);
  }
  const refs = new Set([...priorIds, ...ids]);
  for (const item of value) {
    const r = item as Record<string, unknown>;
    const id = r.id as string;
    if (!Array.isArray(r.dependsOn) || r.dependsOn.some((d) => typeof d !== "string"))
      throw new Error(`${stage}: proposal ${id} dependsOn must be string[]`);
    for (const d of r.dependsOn as string[]) {
      if (d === id) throw new Error(`${stage}: proposal ${id} depends on itself`);
      if (!refs.has(d)) throw new Error(`${stage}: proposal ${id} references unknown id ${d}`);
    }
  }
  return value as Proposal[];
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
  for (const item of value) {
    if (!isObj(item)) throw new Error(`${stage}: analysis must be an object`);
    const names = Object.keys(item).sort();
    for (const k of ["evidenceRefs", "interpretation", "nodeId", "objectiveId"])
      if (!names.includes(k)) throw new Error(`${stage}: unexpected fields [${names.join(",")}]`);
    for (const k of names)
      if (!["assumptions", "evidenceRefs", "interpretation", "nodeId", "numbers", "objectiveId"].includes(k))
        throw new Error(`${stage}: unexpected fields [${names.join(",")}]`);
    const nodeId = nonEmpty(stage, item.nodeId, "analysis.nodeId");
    if (!proposalIds.has(nodeId)) throw new Error(`${stage}: analysis references unknown proposal ${nodeId}`);
    if (item.objectiveId !== objectiveId) throw new Error(`${stage}: analysis ${nodeId} references wrong objective`);
    nonEmpty(stage, item.interpretation, "analysis.interpretation");
    if (!Array.isArray(item.evidenceRefs) || item.evidenceRefs.some((e) => typeof e !== "string"))
      throw new Error(`${stage}: analysis ${nodeId} evidenceRefs must be string[]`);
    for (const e of item.evidenceRefs as string[])
      if (!evidenceIds.has(e)) throw new Error(`${stage}: analysis ${nodeId} references unknown evidence ${e}`);
    const refSet = new Set(item.evidenceRefs as string[]);
    if (item.numbers !== undefined) {
      if (!Array.isArray(item.numbers)) throw new Error(`${stage}: analysis ${nodeId} numbers must be an array`);
      for (const n of item.numbers) {
        if (!isObj(n)) throw new Error(`${stage}: analysis ${nodeId} number must be an object`);
        exactKeys(stage, n, ["evidenceId", "quote", "value"]);
        nonEmpty(stage, n.value, "analysis.number.value");
        const eid = nonEmpty(stage, n.evidenceId, "analysis.number.evidenceId");
        nonEmpty(stage, n.quote, "analysis.number.quote");
        if (!refSet.has(eid)) throw new Error(`${stage}: analysis ${nodeId} number references unknown evidence ${eid}`);
      }
    }
    if (item.assumptions !== undefined) {
      if (!Array.isArray(item.assumptions)) throw new Error(`${stage}: analysis ${nodeId} assumptions must be an array`);
      const seenAssumption = new Set<string>();
      for (const a of item.assumptions) {
        if (!isObj(a)) throw new Error(`${stage}: analysis ${nodeId} assumption must be an object`);
        exactKeys(stage, a, ["assumptionId", "text"]);
        const aid = nonEmpty(stage, a.assumptionId, "analysis.assumption.assumptionId");
        nonEmpty(stage, a.text, "analysis.assumption.text");
        if (seenAssumption.has(aid)) throw new Error(`${stage}: analysis ${nodeId} duplicate assumptionId ${aid}`);
        seenAssumption.add(aid);
      }
    }
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

// Extract concatenated output_text from a Responses API payload, then parse the
// exact contract object. Raw payload is saved by the caller before this runs.
function parseOpenCodeOutput(stage: string, raw: unknown, keys: string[]): Record<string, unknown> {
  const output = isObj(raw) ? raw.output : undefined;
  if (!Array.isArray(output)) throw new Error(`${stage}: malformed_opencode_response`);
  let text = "";
  for (const item of output) {
    if (!isObj(item) || item.type !== "message" || !Array.isArray(item.content)) continue;
    for (const part of item.content)
      if (isObj(part) && part.type === "output_text" && typeof part.text === "string") text += part.text;
  }
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
): Promise<unknown> {
  const body = { model, input };
  await writeFile(join(dir, `request-${name}.json`), JSON.stringify(body, null, 2) + "\n");
  let res: Response;
  try {
    res = await fetchFn(OPENCODE_URL, {
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

export async function runScenario(
  scenarioId: string,
  opts?: { fetchFn?: typeof fetch; systemOne?: SystemOneFn; outDir?: string; model?: string },
): Promise<string> {
  const scenario = scenarios.find((s) => s.id === scenarioId);
  if (!scenario) throw new Error(`unknown_scenario: ${scenarioId} (known: ${scenarios.map((s) => s.id).join(", ")})`);
  const model = opts?.model ?? process.env.OPENCODE_MODEL ?? DEFAULT_MODEL;
  const apiKey = process.env.OPENCODE_API_KEY;
  if (!apiKey) throw new Error("opencode_unavailable: missing OPENCODE_API_KEY");
  const fetchFn = opts?.fetchFn ?? fetch;
  const systemOne = opts?.systemOne ?? createSystemOne();
  const dir = opts?.outDir ?? join(RESULTS_ROOT, `${scenario.id}-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  await mkdir(dir, { recursive: true });

  const objectiveId = scenario.objective.id;
  const evidenceIds = new Set(scenario.evidence.map((e) => e.id));
  const evidenceById = new Map(scenario.evidence.map((e) => [e.id, e]));
  const base = { objective: scenario.objective, fictional: scenario.fictional, evidence: scenario.evidence };

  // 1. decompose (reasoner proposes, never decides)
  const decomposeRaw = await postOpenCode(
    dir, "decompose", promptText("decompose", decomposePrompt, base), model, apiKey, fetchFn,
  );
  const proposals = checkProposals(
    "decompose", parseOpenCodeOutput("decompose", decomposeRaw, ["proposals"]).proposals, objectiveId, new Set(),
  );
  assertAcyclic("decompose", proposals, () => undefined);

  // 2. disposition of initial proposals via JEV choice (mutually exclusive routing)
  const relevanceQuestions: Record<string, unknown> = {};
  for (const p of proposals)
    relevanceQuestions[p.id] = choice(
      `What should happen to this proposed question relative to the user's objective? Question: ${p.question}`,
      DISPOSITION_OPTIONS,
    );
  const relevance = proposals.length
    ? await askDecisions(
      dir, "relevance",
      { state: { objective: scenario.objective, fictional: scenario.fictional, proposals }, questions: relevanceQuestions },
      systemOne,
      Object.fromEntries(proposals.map((p) => [p.id, DISPOSITION_OPTIONS])),
    )
    : { raw: null, decisions: {} as Record<string, DecisionResult> };
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
    dir, "analyze", promptText("analyze", analyzePrompt, analyzeCtx), model, apiKey, fetchFn,
  );
  const analyzeOut = parseOpenCodeOutput("analyze", analyzeRaw, ["analyses", "evidenceRequests"]);
  const analyzableIds = new Set([...analyzeIds, ...gatherIds]);
  const analyses = checkAnalyses("analyze", analyzeOut.analyses, objectiveId, analyzableIds, evidenceIds);
  const analyzeRequests = checkEvidenceRequests("analyze", analyzeOut.evidenceRequests, objectiveId, analyzableIds);
  const analysisIds = new Set(analyses.map((a) => a.nodeId));
  const requestIds = new Set(analyzeRequests.map((r) => r.nodeId));
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

  // 4. adjudicate each analyzed node against its own scoped evidence only:
  // truth noul (actual proposition) + evidence-state choice + materiality score.
  const proposalById = new Map(proposals.map((p) => [p.id, p]));
  const decideRaw: Record<string, unknown> = {};
  const decideAll: Record<string, Record<string, DecisionResult>> = {};
  const nodeResults: {
    nodeId: string;
    truth: NoulDecision;
    truthPolicy: Decision;
    evidenceState: ChoiceDecision;
    materiality: ScoreDecision;
  }[] = [];
  for (const a of analyses) {
    const p = proposalById.get(a.nodeId);
    if (!p) throw new Error(`decide: unknown proposal ${a.nodeId}`);
    const subset = a.evidenceRefs.map((id) => {
      const e = evidenceById.get(id);
      if (!e) throw new Error(`decide: unknown evidence ${id} for ${a.nodeId}`);
      return e;
    });
    const dependencyDecisions = p.dependsOn.map((d) => ({
      proposalId: d,
      disposition: relevance.decisions[d]?.kind === "choice" ? (relevance.decisions[d] as ChoiceDecision).choice : "unknown",
    }));
    const state = scopeNodeState({
      objective: scenario.objective, proposal: p, analysis: a, evidence: subset, dependencyDecisions,
    });
    const questions: Record<string, unknown> = {
      truth: noul(`Do the available facts support that ${p.question}`),
      evidence_state: choice("What best characterizes the evidence state for this node?", EVIDENCE_STATE_OPTIONS),
      materiality: score("How material is resolving this uncertainty to answering the user's objective?", MATERIALITY_LEVELS),
    };
    const name = `decide-${slug(a.nodeId)}`;
    const r = await askDecisions(dir, name, { state, questions }, systemOne, {
      evidence_state: EVIDENCE_STATE_OPTIONS,
    });
    const truth = r.decisions.truth;
    const evidenceState = r.decisions.evidence_state;
    const materiality = r.decisions.materiality;
    if (!truth || truth.kind !== "noul") throw new Error(`decide: missing truth noul for ${a.nodeId}`);
    if (!evidenceState || evidenceState.kind !== "choice") throw new Error(`decide: missing evidence_state choice for ${a.nodeId}`);
    if (!materiality || materiality.kind !== "score") throw new Error(`decide: missing materiality score for ${a.nodeId}`);
    decideRaw[a.nodeId] = r.raw;
    decideAll[a.nodeId] = r.decisions;
    nodeResults.push({
      nodeId: a.nodeId, truth, truthPolicy: classifyProbability(truth.probability), evidenceState, materiality,
    });
  }
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
    priorIds: [...proposalById.keys()], priorRefs: { proposalIds: [...proposalById.keys()] },
  };
  const expandRaw = await postOpenCode(
    dir, "expand", promptText("expand", expandPrompt, expandCtx), model, apiKey, fetchFn,
  );
  const expandOut = parseOpenCodeOutput("expand", expandRaw, ["evidenceRequests", "proposals"]);
  const priorIds = new Set(proposals.map((p) => p.id));
  const expanded = checkProposals("expand", expandOut.proposals, objectiveId, priorIds);
  const priorDeps = new Map(proposals.map((p) => [p.id, p.dependsOn]));
  assertAcyclic("expand", expanded, (id) => priorDeps.get(id));
  const allNodeIds = new Set([...priorIds, ...expanded.map((p) => p.id)]);
  const expandRequests = checkEvidenceRequests("expand", expandOut.evidenceRequests, objectiveId, allNodeIds);

  // 6. disposition of expanded proposals (non-authoritative until JEV admits)
  const expandedQuestions: Record<string, unknown> = {};
  for (const p of expanded)
    expandedQuestions[p.id] = choice(
      `What should happen to this expanded question relative to the user's objective? Question: ${p.question}`,
      DISPOSITION_OPTIONS,
    );
  const expandedRelevance = expanded.length
    ? await askDecisions(
      dir, "relevance-expanded",
      { state: { objective: scenario.objective, fictional: scenario.fictional, proposals: expanded }, questions: expandedQuestions },
      systemOne,
      Object.fromEntries(expanded.map((p) => [p.id, DISPOSITION_OPTIONS])),
    )
    : { raw: null, decisions: {} as Record<string, DecisionResult> };

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
