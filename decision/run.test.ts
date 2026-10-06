// Focused tests for decision/run.ts typed JEV loop. No live calls: OpenCode
// fetch is stubbed and systemOne is an in-process stub inspecting each
// question body. Each run writes to a temp dir; no .env reads.
import { expect, test } from "bun:test";
import { mkdtemp, readdir, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { classifyProbability, runScenario } from "./run.ts";
import type { SystemOneFn } from "./run.ts";
import { resetTypeSafeClientForTest } from "./client.ts";
import { scenarios } from "./scenarios.ts";

const OBJ = scenarios[0].objective.id;
const P = (suffix: string) => `${OBJ}-${suffix}`;

test("classifyProbability boundaries match approved policy", () => {
  expect(classifyProbability(1)).toBe("yes");
  expect(classifyProbability(0.7)).toBe("yes");
  expect(classifyProbability(0.6999)).toBe("unsure");
  expect(classifyProbability(0.5)).toBe("unsure");
  expect(classifyProbability(0.4999)).toBe("no");
  expect(classifyProbability(0)).toBe("no");
  for (const bad of [NaN, -0.1, 1.1, Infinity]) expect(() => classifyProbability(bad)).toThrow("invalid_probability");
});

// Lockstep helper: Responses API payload shape consumed by parseOpenCodeOutput.
function responsesPayload(obj: unknown): Response {
  return new Response(
    JSON.stringify({ output: [{ type: "message", content: [{ type: "output_text", text: JSON.stringify(obj) }] }] }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

const PROPOSALS = [
  { id: P("repayment"), objectiveId: OBJ, question: "Does repayment history support the loan?", dependsOn: [], whyItMatters: "Repayment predicts default risk." },
  { id: P("collateral"), objectiveId: OBJ, question: "Is the accelerator collateral adequate?", dependsOn: [P("repayment")], whyItMatters: "Collateral bounds loss given default." },
  { id: P("lunch"), objectiveId: OBJ, question: "Where does the CEO eat lunch?", dependsOn: [], whyItMatters: "Fictional control: irrelevant." },
];

const DISPOSITION_OF: Record<string, string> = {
  [P("repayment")]: "analyze",
  [P("collateral")]: "gather_evidence",
  [P("lunch")]: "reject",
  [P("packwell")]: "analyze",
};

const DISPOSITION_FULL: Record<string, Record<string, number>> = {
  analyze: { analyze: 0.8, gather_evidence: 0.1, reject: 0.1 },
  gather_evidence: { analyze: 0.1, gather_evidence: 0.8, reject: 0.1 },
  reject: { analyze: 0.05, gather_evidence: 0.05, reject: 0.9 },
};

const EVIDENCE_STATE_FULL = {
  sufficient_support: 0.8,
  sufficient_contradiction: 0.05,
  conflicted: 0.05,
  insufficient: 0.1,
};

// SystemOne stub: inspects each question body and answers in native kind —
// choice with a valid option plus full distribution for disposition and
// evidence_state questions, noul for truth, score for materiality.
function stubSystemOne(): SystemOneFn {
  return async (req) => {
    const answers: Record<string, unknown> = {};
    const questions = req.questions as Record<string, { type?: unknown }>;
    for (const [id, q] of Object.entries(questions)) {
      if (id === "truth" || q.type === "noul") {
        answers[id] = { type: "noul", noul: 0.85 };
      } else if (id === "materiality" || q.type === "score") {
        answers[id] = { type: "score", score: 3, confidence: 0.9 };
      } else if (id === "evidence_state") {
        answers[id] = {
          type: "choice",
          choice: "sufficient_support",
          probabilities: { ...EVIDENCE_STATE_FULL },
          confidence: 0.8,
        };
      } else {
        const disp = DISPOSITION_OF[id] ?? "reject";
        answers[id] = {
          type: "choice",
          choice: disp,
          probabilities: { ...DISPOSITION_FULL[disp] },
          confidence: 0.8,
        };
      }
    }
    return { answers, model: "stub-jev" };
  };
}

async function successDir(secret: string): Promise<string> {
  process.env.OPENCODE_API_KEY = secret;
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-test-"));
  const calls: string[] = [];
  const fetchFn = (async () => {
    const step = calls.length;
    calls.push(step === 0 ? "decompose" : step === 1 ? "analyze" : "expand");
    if (step === 0) return responsesPayload({ proposals: PROPOSALS });
    if (step === 1) {
      return responsesPayload({
        analyses: [
          { nodeId: P("repayment"), objectiveId: OBJ, interpretation: "Repayment history is strong.", evidenceRefs: ["s1-ev3"] },
        ],
        evidenceRequests: [
          { nodeId: P("collateral"), objectiveId: OBJ, missingEvidence: "Independent valuation of the accelerator units." },
        ],
      });
    }
    return responsesPayload({
      proposals: [
        { id: P("packwell"), objectiveId: OBJ, question: "What if Packwell packaging fails?", dependsOn: [P("collateral")], whyItMatters: "Indirect exposure via sole supplier." },
      ],
      evidenceRequests: [
        { nodeId: P("packwell"), objectiveId: OBJ, missingEvidence: "Packwell supply terms." },
      ],
    });
  }) as unknown as typeof fetch;
  try {
    const out = await runScenario(scenarios[0].id, { fetchFn, systemOne: stubSystemOne(), outDir: dir });
    expect(out).toBe(dir);
    return dir;
  } finally {
    resetTypeSafeClientForTest();
  }
}

async function expectRunError(secret: string, fetchFn: typeof fetch, contains: string, systemOne?: SystemOneFn): Promise<string> {
  process.env.OPENCODE_API_KEY = secret;
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-test-"));
  let err: unknown = null;
  try {
    await runScenario(scenarios[0].id, { fetchFn, systemOne: systemOne ?? stubSystemOne(), outDir: dir });
  } catch (e) {
    err = e;
  } finally {
    resetTypeSafeClientForTest();
  }
  expect(String(err)).toContain(contains);
  return dir;
}

test("full sequence: relevance, analysis, adjudication, expansion, artifacts", async () => {
  const dir = await successDir("sk-test-success-key");
  const final = JSON.parse(await readFile(join(dir, "final.json"), "utf8"));
  expect(final.proposals.map((p: { id: string }) => p.id)).toEqual(
    [P("repayment"), P("collateral"), P("lunch"), P("packwell")],
  );
  expect(final.proposals.map((p: { disposition: string }) => p.disposition)).toEqual(
    ["analyze", "gather_evidence", "reject", "analyze"],
  );
  for (const p of final.proposals as { probabilities: unknown; confidence: unknown }[]) {
    expect(p.probabilities).toBeDefined();
    expect(typeof p.confidence).toBe("number");
  }
  expect(final.reasoning).toHaveLength(1);
  expect(final.reasoning[0].nodeId).toBe(P("repayment"));
  expect(final.reasoning[0].truthProbability).toBe(0.85);
  expect(final.reasoning[0].truthPolicy).toBe("yes");
  expect(final.reasoning[0].evidenceState).toBe("sufficient_support");
  expect(final.reasoning[0].materialityScore).toBe(3);
  expect(final.unresolved).toContainEqual({ nodeId: P("collateral"), reason: "gather_evidence" });
  expect(final.evidenceRequests).toHaveLength(2);
  // Parsed decisions carry native kinds per stage.
  expect(final.decisions.relevance[P("repayment")].kind).toBe("choice");
  expect(final.decisions.relevance[P("repayment")].choice).toBe("analyze");
  expect(final.decisions.decide[P("repayment")].truth.kind).toBe("noul");
  expect(final.decisions.decide[P("repayment")].evidence_state.kind).toBe("choice");
  expect(final.decisions.decide[P("repayment")].materiality.kind).toBe("score");
  expect(final.decisions.expandedRelevance[P("packwell")].kind).toBe("choice");
  // Raw JEV output stays separate from parsed decisions.
  expect(final.jev.relevance.answers).toBeDefined();
  expect(final.jev.decide[P("repayment")].answers).toBeDefined();
  expect(final.jev.relevance).not.toEqual(final.decisions.relevance);
  // Typed request artifacts per stage, one decide file per analyzed node.
  const files = await readdir(dir);
  for (const name of ["request-decompose.json", "request-relevance.json", "request-analyze.json", "request-expand.json", "request-relevance-expanded.json"]) {
    expect(files).toContain(name);
  }
  expect(files.filter((f) => f.startsWith("request-decide-"))).toHaveLength(1);
  const evalCtx = JSON.parse(await readFile(join(dir, "eval-context.json"), "utf8"));
  expect(evalCtx.evaluationCriteria).toEqual(scenarios[0].evaluationCriteria);
});

test("gather_evidence proposal analysis is rejected", async () => {
  const calls: string[] = [];
  const fetchFn = (async () => {
    const step = calls.length;
    calls.push(step === 0 ? "decompose" : "analyze");
    if (step === 0) return responsesPayload({ proposals: PROPOSALS });
    return responsesPayload({
      analyses: [
        { nodeId: P("collateral"), objectiveId: OBJ, interpretation: "Must not analyze gather ids.", evidenceRefs: ["s1-ev2"] },
      ],
      evidenceRequests: [
        { nodeId: P("collateral"), objectiveId: OBJ, missingEvidence: "Independent valuation of the accelerator units." },
      ],
    });
  }) as unknown as typeof fetch;
  await expectRunError("sk-test-gather-analysis-key", fetchFn, "must not be analyzed");
});

test("gather_evidence proposal without missing evidence is rejected", async () => {
  const calls: string[] = [];
  const fetchFn = (async () => {
    const step = calls.length;
    calls.push(step === 0 ? "decompose" : "analyze");
    if (step === 0) return responsesPayload({ proposals: PROPOSALS });
    return responsesPayload({
      analyses: [
        { nodeId: P("repayment"), objectiveId: OBJ, interpretation: "Repayment history is strong.", evidenceRefs: ["s1-ev3"] },
      ],
      evidenceRequests: [],
    });
  }) as unknown as typeof fetch;
  await expectRunError("sk-test-gather-missing-key", fetchFn, "missing evidence request");
});

test("malformed OpenCode response fails the chain without final.json", async () => {
  const fetchFn = (() => Promise.resolve(new Response(JSON.stringify({ output: [] }), { status: 200 }))) as unknown as typeof fetch;
  const dir = await expectRunError(
    "sk-test-malformed-key",
    fetchFn,
    "malformed_opencode_response",
    (async () => ({ answers: {} })) as unknown as SystemOneFn,
  );
  let finalErr: unknown = null;
  try {
    await readFile(join(dir, "final.json"), "utf8");
  } catch (e) {
    finalErr = e;
  }
  expect(finalErr).not.toBeNull();
});

test("duplicate proposal ids are rejected", async () => {
  const dupFetch = (() => Promise.resolve(responsesPayload({ proposals: [PROPOSALS[0], PROPOSALS[0]] }))) as unknown as typeof fetch;
  await expectRunError(
    "sk-test-duplicate-key",
    dupFetch,
    "duplicate proposal id",
    (async () => ({ answers: {} })) as unknown as SystemOneFn,
  );
});

test("self-dependency is rejected", async () => {
  const bad = [{ ...PROPOSALS[0], dependsOn: [PROPOSALS[0].id] }];
  const badFetch = (() => Promise.resolve(responsesPayload({ proposals: bad }))) as unknown as typeof fetch;
  await expectRunError(
    "sk-test-selfdep-key",
    badFetch,
    "depends on itself",
    (async () => ({ answers: {} })) as unknown as SystemOneFn,
  );
});

test("indirect dependency cycle is rejected", async () => {
  const a = { id: P("cycle-a"), objectiveId: OBJ, question: "Cycle A?", dependsOn: [P("cycle-b")], whyItMatters: "Cycle." };
  const b = { id: P("cycle-b"), objectiveId: OBJ, question: "Cycle B?", dependsOn: [P("cycle-a")], whyItMatters: "Cycle." };
  const cycleFetch = (() => Promise.resolve(responsesPayload({ proposals: [a, b] }))) as unknown as typeof fetch;
  await expectRunError(
    "sk-test-cycle-key",
    cycleFetch,
    "cyclic dependency",
    (async () => ({ answers: {} })) as unknown as SystemOneFn,
  );
});

test("non-authoritative OpenCode decision field is rejected", async () => {
  const bad = PROPOSALS.map((p) => ({ ...p, decision: "yes" }));
  const badFetch = (() => Promise.resolve(responsesPayload({ proposals: bad }))) as unknown as typeof fetch;
  await expectRunError(
    "sk-test-extra-field-key",
    badFetch,
    "unexpected fields",
    (async () => ({ answers: {} })) as unknown as SystemOneFn,
  );
});

test("no secrets or evaluation criteria leak into request artifacts", async () => {
  const secret = `sk-test-secret-${Date.now()}`;
  const dir = await successDir(secret);
  const files = await readdir(dir);
  const requests = files.filter((f) => f.startsWith("request-"));
  expect(requests).toContain("request-relevance-expanded.json");
  expect(requests.some((f) => f.startsWith("request-decide-"))).toBe(true);
  for (const name of requests) {
    const text = await readFile(join(dir, name), "utf8");
    expect(text).not.toContain("evaluationCriteria");
    expect(text).not.toContain(secret);
    expect(text).not.toContain("Authorization");
  }
  for (const name of files) {
    const text = await readFile(join(dir, name), "utf8");
    expect(text).not.toContain(secret);
    expect(text).not.toContain("Authorization");
  }
});

test("decide request carries only the analysis evidence subset", async () => {
  const dir = await successDir("sk-test-scope-key");
  const files = await readdir(dir);
  const decideReq = files.find((f) => f.startsWith("request-decide-"));
  expect(decideReq).toBeDefined();
  const body = JSON.parse(await readFile(join(dir, decideReq as string), "utf8"));
  const ids = (body.state.evidence as { id: string }[]).map((e) => e.id);
  expect(ids).toContain("s1-ev3");
  expect(ids).not.toContain("s1-ev1");
});
