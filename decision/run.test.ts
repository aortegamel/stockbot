// Focused tests for decision/run.ts typed JEV loop and the decision/client.ts
// Chat Completions and native Decisions transports. No live calls: OpenCode and
// decision fetches are stubbed, and systemOne is an in-process stub inspecting
// each question body. Each run writes to a temp dir; no .env reads.
import { expect, spyOn, test } from "bun:test";
import fc from "fast-check";
import { mkdtemp, readdir, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { classifyProbability, runScenario } from "./run.ts";
import type { SystemOneFn } from "./run.ts";
import { createSystemOne } from "./client.ts";
import { handleDecide } from "./runtime.ts";
import type { DecideOk, DecideResponse } from "./runtime.ts";
import { MATERIALITY_LEVELS } from "./jev.ts";
import { scenarios } from "./scenarios.ts";

// Bun loads the repository .env into this process. Tests that need REASONING_EFFORT set it themselves.
delete process.env.REASONING_EFFORT;

const OBJ = scenarios[0].objective.id;
const P = (suffix: string) => `${OBJ}-${suffix}`;
// Longer than any filename allows, so decide artifacts must use a bounded slug.
const LONG_ID = P("r".repeat(300));

test("classifyProbability boundaries match approved policy", () => {
  expect(classifyProbability(1)).toBe("yes");
  expect(classifyProbability(0.7)).toBe("yes");
  expect(classifyProbability(0.6999)).toBe("yes");
  expect(classifyProbability(0.5)).toBe("yes");
  expect(classifyProbability(0.4999)).toBe("no");
  expect(classifyProbability(0)).toBe("no");
  for (const bad of [NaN, -0.1, 1.1, Infinity]) expect(classifyProbability(bad)).toBe("unknown");
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
  [LONG_ID]: "analyze",
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

async function successDir(secret: string, systemOne: SystemOneFn = stubSystemOne()): Promise<string> {
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
  const out = await runScenario(scenarios[0].id, { fetchFn, systemOne, outDir: dir });
  expect(out).toBe(dir);
  return dir;
}

async function expectRunError(secret: string, fetchFn: typeof fetch, contains?: string, systemOne?: SystemOneFn): Promise<string> {
  process.env.OPENCODE_API_KEY = secret;
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-test-"));
  let err: unknown = null;
  try {
    await runScenario(scenarios[0].id, { fetchFn, systemOne: systemOne ?? stubSystemOne(), outDir: dir });
  } catch (e) {
    err = e;
  }
  if (contains === undefined) expect(err).toBeInstanceOf(Error);
  else expect(String(err)).toContain(contains);
  return dir;
}

function analysisFetch(analysisOut: unknown, proposals: unknown = PROPOSALS): typeof fetch {
  let step = 0;
  return (async () => {
    if (step++ === 0) return responsesPayload({ proposals });
    if (step === 2) return responsesPayload(analysisOut);
    return responsesPayload({ proposals: [], evidenceRequests: [] });
  }) as unknown as typeof fetch;
}

const VALID_ANALYSIS = {
  nodeId: P("repayment"), objectiveId: OBJ,
  interpretation: "Repayment history is strong.", evidenceRefs: ["s1-ev3"],
};
const GATHER_REQUEST = {
  nodeId: P("collateral"), objectiveId: OBJ, missingEvidence: "Independent valuation.",
};

test("analysis numbers and assumptions reach final artifacts without losing evidence links", async () => {
  process.env.OPENCODE_API_KEY = "sk-test-analysis";
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-analysis-"));
  const analysis = {
    ...VALID_ANALYSIS,
    numbers: [{ value: "12", evidenceId: "s1-ev3", quote: "12" }],
    assumptions: [{ assumptionId: "a1", text: "The historical terms remain comparable." }],
  };
  await runScenario(scenarios[0].id, {
    outDir: dir, systemOne: stubSystemOne(),
    fetchFn: analysisFetch({ analyses: [analysis], evidenceRequests: [GATHER_REQUEST] }),
  });
  const final = JSON.parse(await readFile(join(dir, "final.json"), "utf8"));
  expect(final.reasoning[0]).toMatchObject(analysis);
  const context = JSON.parse(await readFile(join(dir, "eval-context.json"), "utf8"));
  expect(context.analyses).toEqual([analysis]);
});

test("analysis validation rejects malformed numbers, assumptions, and evidence before adjudication", async () => {
  const number = { value: "12", evidenceId: "s1-ev3", quote: "12" };
  const assumption = { assumptionId: "a1", text: "Comparable terms." };
  const cases: [Record<string, unknown>, string][] = [
    [{ numbers: null }, "numbers must be an array"],
    [{ numbers: [null] }, "number must be an object"],
    [{ numbers: [{ ...number, extra: 1 }] }, "unexpected fields"],
    [{ numbers: [{ ...number, value: "" }] }, "analysis.number.value must be a nonempty string"],
    [{ numbers: [{ ...number, evidenceId: "" }] }, "analysis.number.evidenceId must be a nonempty string"],
    [{ numbers: [{ ...number, quote: "" }] }, "analysis.number.quote must be a nonempty string"],
    [{ numbers: [{ ...number, evidenceId: "s1-ev1" }] }, "number references unknown evidence s1-ev1"],
    [{ assumptions: null }, "assumptions must be an array"],
    [{ assumptions: [null] }, "assumption must be an object"],
    [{ assumptions: [{ ...assumption, extra: 1 }] }, "unexpected fields"],
    [{ assumptions: [{ ...assumption, assumptionId: "" }] }, "analysis.assumption.assumptionId must be a nonempty string"],
    [{ assumptions: [{ ...assumption, text: "" }] }, "analysis.assumption.text must be a nonempty string"],
    [{ assumptions: [assumption, assumption] }, "duplicate assumptionId a1"],
    [{ evidenceRefs: null }, "evidenceRefs must be string[]"],
    [{ evidenceRefs: [1] }, "evidenceRefs must be string[]"],
    [{ evidenceRefs: ["missing"] }, "references unknown evidence missing"],
    [{ objectiveId: "wrong" }, "references wrong objective"],
    [{ nodeId: P("missing") }, "references unknown proposal"],
    [{ interpretation: "" }, "analysis.interpretation must be a nonempty string"],
    [{ evidenceRefs: undefined }, "unexpected fields"],
    [{ numbers: null, assumptions: null }, "numbers must be an array"],
  ];
  for (const [patch, message] of cases) {
    const dir = await expectRunError("sk-test-invalid-analysis", analysisFetch({
      analyses: [{ ...VALID_ANALYSIS, ...patch }], evidenceRequests: [GATHER_REQUEST],
    }), message);
    expect(await readdir(dir)).not.toContain("final.json");
    expect((await readdir(dir)).some((f) => f.startsWith("request-decide-"))).toBe(false);
  }
  for (const [analyses, message] of [
    [null, "analyses must be an array"],
    [[null], "analysis must be an object"],
    [[{ nodeId: P("repayment") }], "unexpected fields"],
    [[{ ...VALID_ANALYSIS, extra: true }], "unexpected fields"],
    [[VALID_ANALYSIS, VALID_ANALYSIS], "duplicate analysis"],
  ] as const) {
    await expectRunError("sk-test-analysis-shape", analysisFetch({ analyses, evidenceRequests: [GATHER_REQUEST] }), message);
  }
  for (const bad of [
    { ...GATHER_REQUEST, nodeId: P("lunch") },
    { ...GATHER_REQUEST, objectiveId: "other" },
  ]) {
    const dir = await expectRunError("sk-test-bad-request", analysisFetch({
      analyses: [VALID_ANALYSIS], evidenceRequests: [GATHER_REQUEST, bad],
    }));
    expect((await readdir(dir)).some((f) => f.startsWith("request-decide-"))).toBe(false);
  }
});

test("analysis accounting rejects both and neither, but admits evidence requests without analysis", async () => {
  for (const analyses of [[], [VALID_ANALYSIS]]) {
    const evidenceRequests = analyses.length ? [
      GATHER_REQUEST, { nodeId: P("repayment"), objectiveId: OBJ, missingEvidence: "History." },
    ] : [GATHER_REQUEST];
    await expectRunError("sk-test-accounting", analysisFetch({ analyses, evidenceRequests }), "requires exactly one");
  }
  process.env.OPENCODE_API_KEY = "sk-test-accounting";
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-accounting-"));
  await runScenario(scenarios[0].id, {
    outDir: dir, systemOne: stubSystemOne(), fetchFn: analysisFetch({
      analyses: [], evidenceRequests: [
        GATHER_REQUEST, { nodeId: P("repayment"), objectiveId: OBJ, missingEvidence: "History." },
      ],
    }),
  });
  const final = JSON.parse(await readFile(join(dir, "final.json"), "utf8"));
  expect(final.reasoning).toEqual([]);
  expect(final.unresolved).toEqual([
    { nodeId: P("collateral"), reason: "gather_evidence" },
    { nodeId: P("repayment"), reason: "awaiting_evidence" },
  ]);
});

test("Responses parsing concatenates text fragments and ignores non-message parts", async () => {
  const output = [
    null, { type: "reasoning", content: [] }, { type: "reasoning", content: [{ type: "output_text", text: "x" }] },
    { type: "message", content: null },
    { type: "message", content: [null, { type: "refusal", text: "ignored" }, { type: "output_text", text: 7 }, { type: "output_text", text: '{"pro' }] },
    { type: "message", content: [{ type: "output_text", text: 'posals":[]}' }] },
  ];
  let calls = 0;
  const fetchFn = (async () => {
    if (calls++ === 0) return chat({ output });
    if (calls === 2) return responsesPayload({ analyses: [], evidenceRequests: [] });
    return responsesPayload({ proposals: [], evidenceRequests: [] });
  }) as unknown as typeof fetch;
  // The fragments concatenate to {"proposals":[]}, so the chain completes with no proposals.
  process.env.OPENCODE_API_KEY = "sk-test-fragments";
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-fragments-"));
  await runScenario(scenarios[0].id, { outDir: dir, systemOne: stubSystemOne(), fetchFn });
  expect(calls).toBe(3);
  expect(JSON.parse(await readFile(join(dir, "final.json"), "utf8")).proposals).toEqual([]);
  expect(await readdir(dir)).toContain("raw-decompose.json");
  // An empty proposal list never reaches JEV, which rejects requests without questions.
  expect(await readdir(dir)).not.toContain("request-relevance.json");
});

test("Responses parsing distinguishes malformed envelopes from malformed JSON", async () => {
  for (const [body, message] of [
    [null, "malformed_opencode_response"],
    [{ output: null }, "malformed_opencode_response"],
    [{ output: [{ type: "message", content: [{ type: "output_text", text: "" }] }] }, "malformed_opencode_response"],
    [{ output: [{ type: "message", content: [{ type: "output_text", text: "{" }] }] }, "malformed_opencode_json"],
    [{ output: [{ type: "message", content: [{ type: "output_text", text: "[]" }] }] }, "malformed_opencode_json"],
  ] as const) {
    const fetchFn = (async () => chat(body)) as unknown as typeof fetch;
    const dir = await expectRunError("sk-test-response-boundary", fetchFn, message);
    expect(await readdir(dir)).toContain("raw-decompose.json");
    expect(await readdir(dir)).not.toContain("request-relevance.json");
  }
});

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
  expect(final.reasoning[0].truthResult).toEqual({ outcome: "yes", probability: 0.85, strength: "confident" });
  expect(final.reasoning[0].truthStrength).toBe("confident");
  expect(final.reasoning[0].evidenceState).toBe("sufficient_support");
  expect(final.reasoning[0].materialityScore).toBe(3);
  expect(final.unresolved).toContainEqual({ nodeId: P("collateral"), reason: "gather_evidence" });
  expect(final.evidenceRequests).toHaveLength(2);
  expect(final.unresolved).toEqual([{ nodeId: P("collateral"), reason: "gather_evidence" }]);
  expect(final.proposals.map((p: { stage: string }) => p.stage)).toEqual(["initial", "initial", "initial", "expanded"]);
  expect(final.evidenceRequests.map((r: { nodeId: string; stage: string }) => [r.nodeId, r.stage])).toEqual(
    [[P("collateral"), "analyze"], [P("packwell"), "expand"]],
  );
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
  const replies = [
    { proposals: PROPOSALS },
    { analyses: [VALID_ANALYSIS], evidenceRequests: [GATHER_REQUEST] },
    { proposals: [PROPOSALS[0]], evidenceRequests: [] },
  ];
  const dir = await expectRunError(
    "sk-test-reused-key",
    (async () => responsesPayload(replies.shift())) as unknown as typeof fetch,
  );
  expect(await readdir(dir)).not.toContain("request-relevance-expanded.json");
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

test("malformed or misreferenced proposals are rejected before JEV relevance", async () => {
  const cases = [
    { proposals: [{ ...PROPOSALS[0], id: `${OBJ}repayment` }] },
    { proposals: [{ ...PROPOSALS[0], objectiveId: "other" }] },
    { proposals: [{ ...PROPOSALS[0], dependsOn: [P("missing")] }] },
    { proposals: [{ ...PROPOSALS[0], dependsOn: "" }] },
    { proposals: PROPOSALS, extra: 1 },
  ];
  for (const reply of cases) {
    const dir = await expectRunError("sk-test-bad-proposal", (async () => responsesPayload(reply)) as unknown as typeof fetch);
    expect(await readdir(dir)).not.toContain("request-relevance.json");
  }
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

test("each analyzed node keeps its own verdict, dependency dispositions, unresolved reason, and decide file", async () => {
  const packwell = {
    id: P("packwell"), objectiveId: OBJ, question: "What if Packwell packaging fails?",
    dependsOn: [P("repayment")], whyItMatters: "Indirect exposure via sole supplier.",
  };
  const verdict: Record<string, [number, string]> = {
    [P("repayment")]: [0.2, "insufficient"],
    [P("packwell")]: [0.6, "conflicted"],
  };
  const stub = stubSystemOne();
  const systemOne: SystemOneFn = async (req) => {
    const out = (await stub(req)) as { answers: Record<string, Record<string, unknown>> };
    const { state } = req;
    const proposal = typeof state === "object" && state !== null && "proposal" in state ? state.proposal : undefined;
    const id = typeof proposal === "object" && proposal !== null && "id" in proposal ? proposal.id : undefined;
    const node = typeof id === "string" ? verdict[id] : undefined;
    if (node) {
      out.answers.truth.noul = node[0];
      out.answers.evidence_state.choice = node[1];
    }
    return out;
  };
  process.env.OPENCODE_API_KEY = "sk-test-nodes";
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-nodes-"));
  await runScenario(scenarios[0].id, {
    outDir: dir, systemOne, fetchFn: analysisFetch({
      analyses: [VALID_ANALYSIS, {
        nodeId: P("packwell"), objectiveId: OBJ, interpretation: "Packwell is the sole supplier.", evidenceRefs: ["s1-ev4"],
      }],
      evidenceRequests: [],
    }, [PROPOSALS[0], packwell]),
  });
  const final = JSON.parse(await readFile(join(dir, "final.json"), "utf8"));
  expect(final.reasoning.map((r: { nodeId: string; truthProbability: number; evidenceState: string }) =>
    [r.nodeId, r.truthProbability, r.evidenceState])).toEqual([
      [P("repayment"), 0.2, "insufficient"],
      [P("packwell"), 0.6, "conflicted"],
    ]);
  expect(final.unresolved).toEqual([
    { nodeId: P("repayment"), reason: "insufficient" },
    { nodeId: P("packwell"), reason: "conflicted" },
  ]);
  const decideFiles = (await readdir(dir)).filter((f) => f.startsWith("request-decide-"));
  expect(decideFiles).toHaveLength(2);
  const packwellRequest = JSON.parse(await readFile(join(dir, decideFiles.find((f) => f.includes("packwell")) as string), "utf8"));
  expect(packwellRequest.state.dependencyDecisions).toEqual([{ proposalId: P("repayment"), disposition: "analyze" }]);
});

test("a proposal id longer than a filename still gets its decide request file", async () => {
  process.env.OPENCODE_API_KEY = "sk-test-long-id";
  process.env.OPENCODE_MODEL = "test-model";
  process.env.OPENCODE_URL = "https://opencode.test/responses";
  const dir = await mkdtemp(join(tmpdir(), "run-long-id-"));
  await runScenario(scenarios[0].id, {
    outDir: dir, systemOne: stubSystemOne(), fetchFn: analysisFetch(
      { analyses: [{ ...VALID_ANALYSIS, nodeId: LONG_ID }], evidenceRequests: [] },
      [{ ...PROPOSALS[0], id: LONG_ID }],
    ),
  });
  expect((await readdir(dir)).filter((f) => f.startsWith("request-decide-"))).toHaveLength(1);
  expect(JSON.parse(await readFile(join(dir, "final.json"), "utf8")).reasoning[0].nodeId).toBe(LONG_ID);
});

test("non-2xx OpenCode responses are request failures without raw artifacts", async () => {
  const fetchFn = (async () => chat({ error: { message: "invalid key" } }, 401)) as unknown as typeof fetch;
  const dir = await expectRunError("sk-test-status-key", fetchFn);
  expect(await readdir(dir)).not.toContain("raw-decompose.json");
});

test("unknown scenarios and blank OpenCode configuration fail before any request", async () => {
  const calls: FetchCall[] = [];
  const fetchFn = recordingFetch(calls, () => responsesPayload({ proposals: PROPOSALS }));
  const attempt = async (scenarioId: string) =>
    runScenario(scenarioId, { fetchFn, systemOne: stubSystemOne(), outDir: await mkdtemp(join(tmpdir(), "run-config-")) });
  await expect(attempt("no-such-scenario")).rejects.toThrow();
  for (const name of ["OPENCODE_MODEL", "OPENCODE_URL", "OPENCODE_API_KEY"]) {
    for (const blank of ["", "   "]) {
      process.env.OPENCODE_API_KEY = "sk-test-config";
      process.env.OPENCODE_MODEL = "test-model";
      process.env.OPENCODE_URL = "https://opencode.test/responses";
      process.env[name] = blank;
      await expect(attempt(scenarios[0].id)).rejects.toThrow();
    }
  }
  expect(calls).toHaveLength(0);
});

test("REASONING_EFFORT reaches OpenCode as reasoning.effort; a bad shape fails before any request", async () => {
  const calls: FetchCall[] = [];
  const fetchFn = recordingFetch(calls, () => chat({ error: { message: "stop" } }, 401));
  try {
    process.env.REASONING_EFFORT = "xhigh";
    await expectRunError("sk-test-effort", fetchFn, "opencode_request_failed");
    expect(JSON.parse(String(calls[0].init.body)).reasoning).toEqual({ effort: "xhigh" });
    process.env.REASONING_EFFORT = "X-HIGH";
    await expectRunError("sk-test-effort", fetchFn, "REASONING_EFFORT must match");
    expect(calls).toHaveLength(1);
  } finally {
    delete process.env.REASONING_EFFORT;
  }
});

test("a loopback Responses server accepts every OpenCode request and the run completes", async () => {
  const replies = [
    { proposals: PROPOSALS },
    { analyses: [VALID_ANALYSIS], evidenceRequests: [GATHER_REQUEST] },
    { proposals: [], evidenceRequests: [] },
  ];
  // Like a real Responses endpoint, the server answers only authenticated JSON POSTs of {model, input}.
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    async fetch(req) {
      const body: unknown = await req.json().catch(() => null);
      const ok = req.method === "POST"
        && req.headers.get("content-type") === "application/json"
        && req.headers.get("authorization") === "Bearer sk-test-loopback"
        && typeof body === "object" && body !== null && Object.keys(body).sort().join() === "input,model"
        && "model" in body && body.model === "test-model"
        && "input" in body && typeof body.input === "string" && body.input.length > 0;
      return ok ? responsesPayload(replies.shift()) : new Response("bad request", { status: 400 });
    },
  });
  // run.ts owns its fetch options, so NO_PROXY keeps ambient proxy variables off the loopback hop.
  const proxyEnv = { NO_PROXY: process.env.NO_PROXY, no_proxy: process.env.no_proxy };
  try {
    process.env.NO_PROXY = "127.0.0.1";
    process.env.no_proxy = "127.0.0.1";
    process.env.OPENCODE_API_KEY = "sk-test-loopback";
    process.env.OPENCODE_MODEL = "test-model";
    process.env.OPENCODE_URL = `http://127.0.0.1:${server.port}/responses`;
    const dir = await mkdtemp(join(tmpdir(), "run-loopback-"));
    await runScenario(scenarios[0].id, { outDir: dir, systemOne: stubSystemOne() });
    // stubSystemOne answers truth 0.85, sufficient_support at confidence 0.8, and materiality 3.
    expect(JSON.parse(await readFile(join(dir, "final.json"), "utf8")).reasoning).toEqual([{
      ...VALID_ANALYSIS,
      truthProbability: 0.85, truthPolicy: "yes",
      truthResult: { outcome: "yes", probability: 0.85, strength: "confident" }, truthStrength: "confident",
      evidenceState: "sufficient_support", evidenceProbabilities: EVIDENCE_STATE_FULL, evidenceConfidence: 0.8,
      materialityScore: 3,
    }]);
  } finally {
    server.stop(true);
    for (const [name, value] of Object.entries(proxyEnv)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
});

test("CLI exits 2 for usage, 1 for a failed run, and 0 when listing scenario ids", () => {
  // A temp cwd keeps Bun from loading any repository .env file.
  const cli = (...args: string[]) =>
    Bun.spawnSync([process.execPath, join(import.meta.dir, "run.ts"), ...args], { cwd: tmpdir() });
  expect(cli().exitCode).toBe(2);
  const failed = cli("no-such-scenario");
  expect(failed.exitCode).toBe(1);
  const list = cli("--list");
  expect(list.exitCode).toBe(0);
  expect(list.stdout.toString().trim().split("\n").map((line) => line.split("\t")[0])).toEqual(scenarios.map((s) => s.id));
});

const DECISION_ENV = {
  DECISION_API_FORMAT: "chat_completions",
  DECISION_API_URL: "https://decision.test/custom/chat/completions",
  DECISION_API_KEY: "sk-decision-test-key",
  DECISION_MODEL: "decision-test-model",
};

// Native Decisions configuration for the reported model. The host is fake; fetch is always stubbed.
const NATIVE_ENV = {
  DECISION_API_FORMAT: "decisions",
  DECISION_API_URL: "https://decision.test/api/alpha/decisions",
  DECISION_MODEL: "typesafe/jev-1.13",
};

function setDecisionEnv(overrides: Record<string, string> = {}): void {
  Object.assign(process.env, DECISION_ENV, overrides);
}

function chat(body: unknown, status = 200): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
}

function completion(content: unknown, extra: Record<string, unknown> = {}, finish_reason = "stop"): Record<string, unknown> {
  return { choices: [{ index: 0, message: { role: "assistant", content }, finish_reason }], ...extra };
}

type FetchCall = { url: string; init: RequestInit };

function recordingFetch(calls: FetchCall[], respond: (init: RequestInit) => Response | Promise<Response>): typeof fetch {
  return (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return respond(init);
  }) as unknown as typeof fetch;
}

const Q = {
  truth: { type: "noul", instructions: "Is it true?" },
  route: { type: "choice", instructions: "Route?", criteria: { a: "Route A", b: "Route B" } },
};
const GOOD = {
  truth: { type: "noul", noul: 0.7 },
  route: { type: "choice", choice: "a", probabilities: { a: 0.6, b: 0.4 }, confidence: 0.6 },
};

function decide(fetchFn: typeof fetch, questions: Record<string, unknown> = Q): Promise<DecideResponse> {
  return handleDecide({ id: "r1", op: "decide", state: { s: 1 }, questions }, createSystemOne(fetchFn));
}

async function decideError(body: unknown, status = 200): Promise<string> {
  setDecisionEnv();
  const res = await decide(recordingFetch([], () => chat(body, status)));
  if (!("error" in res)) throw new Error(`expected decide error, got ${JSON.stringify(res)}`);
  expect(res.id).toBe("r1");
  return res.error;
}

test("both decision formats reject HTTP before transport, including loopback", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    for (const host of ["decision.test", "localhost", "127.0.0.1", "[::1]"]) {
      setDecisionEnv({
        DECISION_API_FORMAT: format,
        DECISION_API_URL: `http://${host}/v1/decide?token=${DECISION_ENV.DECISION_API_KEY}`,
      });
      const calls: FetchCall[] = [];
      const res = await decide(recordingFetch(calls, () => chat({ answers: GOOD })));
      expect(res).toEqual({ id: "r1", error: "decide: decision_request_failed: DECISION_API_URL must use HTTPS" });
      expect(JSON.stringify(res)).not.toContain(DECISION_ENV.DECISION_API_KEY);
      expect(calls).toHaveLength(0);
    }
  }
});

test("both decision formats accept only HTTPS schemes", async () => {
  await fc.assert(fc.asyncProperty(
    fc.oneof(
      fc.constantFrom("https", "HTTPS", "hTtPs", "http", "HTTP", "ftp", "file", "ws", "wss"),
      fc.stringMatching(/^[A-Za-z][A-Za-z0-9+.-]{0,12}$/),
    ),
    fc.constantFrom("decision.test", "localhost", "127.0.0.1", "[::1]"),
    async (scheme, host) => {
      for (const format of ["chat_completions", "decisions"]) {
        const url = `${scheme}://${host}/v1/decide`;
        setDecisionEnv({ DECISION_API_FORMAT: format, DECISION_API_URL: url });
        const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
        const calls: FetchCall[] = [];
        const res = await decide(recordingFetch(calls, () => chat(body)));
        if (scheme.toLowerCase() === "https") {
          expect(calls).toHaveLength(1);
          expect(calls[0].url).toBe(url);
          expect(calls[0].init.headers).toEqual({
            Authorization: `Bearer ${DECISION_ENV.DECISION_API_KEY}`,
            "Content-Type": "application/json",
          });
          expect(res).toHaveProperty("decisions.truth", { kind: "noul", probability: 0.7 });
        } else {
          expect(res).toEqual({ id: "r1", error: "decide: decision_request_failed: DECISION_API_URL must use HTTPS" });
          expect(JSON.stringify(res)).not.toContain(DECISION_ENV.DECISION_API_KEY);
          expect(calls).toHaveLength(0);
        }
      }
    },
  ), { seed: 20261017, numRuns: 60 });
});

test("both formats reject non-HTTPS redirects before contacting any target", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    for (const status of [301, 302, 303, 307, 308]) {
      for (const target of [
        "http://decision.test", "http://localhost", "http://127.0.0.1", "http://[::1]",
        "ftp://decision.test", "file:///tmp/decision", "ws://decision.test", "wss://decision.test",
      ]) {
        for (const secureHop of [false, true]) {
          setDecisionEnv({ DECISION_API_FORMAT: format });
          const calls: FetchCall[] = [];
          let released = 0;
          const res = await decide(recordingFetch(calls, () => new Response(new ReadableStream({
            cancel() { released++; },
          }), {
            status,
            headers: {
              Location: secureHop && calls.length === 1
                ? "/secure-hop"
                : `${target}/decide?token=${DECISION_ENV.DECISION_API_KEY}`,
            },
          })));
          expect(res).toEqual({ id: "r1", error: "decide: decision_request_failed: DECISION_API_URL redirect must use HTTPS" });
          expect(JSON.stringify(res)).not.toContain(DECISION_ENV.DECISION_API_KEY);
          expect(calls.map((call) => call.url)).toEqual(secureHop
            ? [DECISION_ENV.DECISION_API_URL, "https://decision.test/secure-hop"]
            : [DECISION_ENV.DECISION_API_URL]);
          expect(calls.every((call) => call.init.redirect === "manual")).toBe(true);
          expect(released).toBe(calls.length);
        }
      }
    }
  }
});

test("secure relative redirects preserve both wire formats, headers, and the initial endpoint string", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    const endpoint = "HTTPS://Decision.TEST:443/api/alpha/decisions?escaped=%2f";
    setDecisionEnv({ DECISION_API_FORMAT: format, DECISION_API_URL: endpoint });
    const calls: FetchCall[] = [];
    const locations = ["../next?escaped=%2f", "?page=2", "//decision.test/final", ""];
    let released = 0;
    const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
    const res = await decide(recordingFetch(calls, (init) => {
      if (calls.length <= locations.length) {
        expect(released).toBe(calls.length - 1);
        return new Response(new ReadableStream({ cancel() { released++; } }), {
          status: calls.length % 2 ? 307 : 308,
          headers: { Location: locations[calls.length - 1] },
        });
      }
      expect(released).toBe(locations.length);
      expect(init.signal).toBe(calls[0].init.signal);
      return chat(body);
    }));
    expect(res).toHaveProperty("decisions.truth", { kind: "noul", probability: 0.7 });
    expect(calls.map((call) => call.url)).toEqual([
      endpoint, "https://decision.test/api/next?escaped=%2f", "https://decision.test/api/next?page=2",
      "https://decision.test/final", "https://decision.test/final",
    ]);
    for (const { init } of calls) {
      expect(init.method).toBe("POST");
      expect(init.redirect).toBe("manual");
      expect(init.signal).toBeInstanceOf(AbortSignal);
      expect(init.signal).toBe(calls[0].init.signal);
      expect(init.headers).toEqual({
        Authorization: `Bearer ${DECISION_ENV.DECISION_API_KEY}`,
        "Content-Type": "application/json",
      });
      expect(init.body).toBe(calls[0].init.body);
      const request = JSON.parse(String(init.body));
      if (format === "decisions") {
        expect(request).toEqual({ model: DECISION_ENV.DECISION_MODEL, state: { s: 1 }, questions: Q });
      } else {
        expect(Object.keys(request).sort()).toEqual(["messages", "model", "stream"]);
        expect(request.model).toBe(DECISION_ENV.DECISION_MODEL);
        expect(request.stream).toBe(false);
        expect(request.messages.map((message: { role: string }) => message.role)).toEqual(["system", "user"]);
        expect(typeof request.messages[0].content).toBe("string");
        expect(JSON.parse(request.messages[1].content)).toEqual({ state: { s: 1 }, questions: Q });
      }
    }
  }
});

test("both formats follow Fetch method and body rules for every redirect status", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    for (const status of [301, 302, 303, 307, 308]) {
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const calls: FetchCall[] = [];
      const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
      const res = await decide(recordingFetch(calls, () => calls.length < 3
        ? new Response(null, { status: calls.length === 1 ? status : 303, headers: { Location: "/next" } })
        : chat(body)));
      expect(res).toHaveProperty("decisions.truth", { kind: "noul", probability: 0.7 });
      expect(calls).toHaveLength(3);
      expect(calls[0].init.method).toBe("POST");
      expect(calls[0].init.body).toBeString();
      expect(new Headers(calls[0].init.headers).get("Content-Type")).toBe("application/json");
      const retainsBody = status === 307 || status === 308;
      expect(calls[1].init.method).toBe(retainsBody ? "POST" : "GET");
      expect(calls[1].init.body).toBe(retainsBody ? calls[0].init.body : undefined);
      expect(new Headers(calls[1].init.headers).get("Content-Type")).toBe(retainsBody ? "application/json" : null);
      expect(calls[2].init.method).toBe("GET");
      expect(calls[2].init.body).toBeUndefined();
      expect(new Headers(calls[2].init.headers).get("Content-Type")).toBeNull();
      for (const { init } of calls) {
        expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${DECISION_ENV.DECISION_API_KEY}`);
        expect(init.redirect).toBe("manual");
      }
    }
  }
});

test("both formats strip Authorization across origins without restoring it on return", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    for (const target of ["https://other.test/next", "https://decision.test:444/next"]) {
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const calls: FetchCall[] = [];
      const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
      const res = await decide(recordingFetch(calls, () => calls.length < 3
        ? new Response(null, { status: 307, headers: { Location: calls.length === 1 ? target : DECISION_ENV.DECISION_API_URL } })
        : chat(body)));
      expect(res).toHaveProperty("decisions.truth", { kind: "noul", probability: 0.7 });
      expect(calls.map((call) => call.url)).toEqual([DECISION_ENV.DECISION_API_URL, target, DECISION_ENV.DECISION_API_URL]);
      expect(new Headers(calls[0].init.headers).get("Authorization")).toBe(`Bearer ${DECISION_ENV.DECISION_API_KEY}`);
      for (const { init } of calls.slice(1)) {
        expect(new Headers(init.headers).has("Authorization")).toBe(false);
        expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
        expect(init.method).toBe("POST");
        expect(init.body).toBe(calls[0].init.body);
      }
    }
  }
});

test("both formats allow 20 redirects but release and reject the next redirect", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    for (const exhaust of [false, true]) {
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const calls: FetchCall[] = [];
      let released = 0;
      const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
      const res = await decide(recordingFetch(calls, () => {
        if (!exhaust && calls.length === 21) return chat(body);
        return new Response(new ReadableStream({ cancel() { released++; } }), {
          status: 308, headers: { Location: DECISION_ENV.DECISION_API_URL },
        });
      }));
      expect(calls).toHaveLength(21);
      expect(released).toBe(exhaust ? 21 : 20);
      if (exhaust) {
        expect(res).toEqual({ id: "r1", error: "decide: decision_request_failed: DECISION_API_URL exceeded 20 redirects" });
      } else {
        expect(res).toHaveProperty("decisions.truth", { kind: "noul", probability: 0.7 });
      }
    }
  }
});

test("both formats create one timeout for all redirect hops and the final body", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    setDecisionEnv({ DECISION_API_FORMAT: format });
    const controller = new AbortController();
    const timeout = spyOn(AbortSignal, "timeout").mockReturnValue(controller.signal);
    try {
      const calls: FetchCall[] = [];
      const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
      let readBody = false;
      const res = await decide(recordingFetch(calls, (init) => {
        expect(init.signal).toBe(controller.signal);
        if (calls.length === 1) return new Response(null, { status: 307, headers: { Location: "/next" } });
        return new Response(new ReadableStream({
          pull(stream) {
            expect(init.signal).toBe(controller.signal);
            expect(timeout).toHaveBeenCalledTimes(1);
            readBody = true;
            stream.enqueue(new TextEncoder().encode(JSON.stringify(body)));
            stream.close();
          },
        }));
      }));
      expect(res).toHaveProperty("decisions.truth", { kind: "noul", probability: 0.7 });
      expect(readBody).toBe(true);
      expect(calls).toHaveLength(2);
      expect(timeout).toHaveBeenCalledTimes(1);
      expect(timeout).toHaveBeenCalledWith(10_000);
    } finally {
      timeout.mockRestore();
    }
  }
});

test("both formats redact redirect parsing, cancellation, fetch, and body errors", async () => {
  for (const format of ["chat_completions", "decisions"]) {
    for (const failure of ["location", "cancel", "fetch", "body"]) {
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const calls: FetchCall[] = [];
      let released = false;
      const message = `redirect failed for ${DECISION_ENV.DECISION_API_KEY}`;
      const res = await decide(recordingFetch(calls, () => {
        if (calls.length === 1) {
          return new Response(new ReadableStream({
            cancel() {
              released = true;
              if (failure === "cancel") throw new Error(message);
            },
          }), {
            status: 307,
            headers: { Location: failure === "location" ? `https://[${DECISION_ENV.DECISION_API_KEY}` : "/next" },
          });
        }
        if (failure === "fetch") throw new Error(message);
        return new Response(new ReadableStream({ start(stream) { stream.error(new Error(message)); } }));
      }));
      expect(released).toBe(true);
      expect(calls).toHaveLength(failure === "location" || failure === "cancel" ? 1 : 2);
      if (!("error" in res)) throw new Error("expected decide error");
      expect(res.error).toStartWith("decide: decision_request_failed:");
      expect(res.error).not.toContain(DECISION_ENV.DECISION_API_KEY);
      if (failure !== "location") expect(res.error).toContain("redirect failed for [redacted]");
    }
  }
});

test("decision transport keeps the SDK question guards before any request", async () => {
  setDecisionEnv();
  const score = (criteria: unknown) => ({ m: { type: "score", instructions: "How material?", criteria } });
  const cases: [Record<string, unknown>, string][] = [
    [{}, "At least one question is required."],
    [score(["Low"]), 'Score question "m" has 1 criteria; at least two scores are required.'],
    [score([]), 'Score question "m" has 0 criteria; at least two scores are required.'],
    [{ m: { type: "score", instructions: "How material?", levels: ["Low", "High"] } },
      'Score question "m" has criteria that are not a list; score criteria must be a list of descriptions indexed by score from zero.'],
    [score({ low: "Low", high: "High" }),
      'Score question "m" has criteria that are not a list; score criteria must be a list of descriptions indexed by score from zero.'],
  ];
  for (const [questions, message] of cases) {
    const calls: FetchCall[] = [];
    const res = await decide(recordingFetch(calls, () => chat(completion("{}"))), questions);
    expect(res).toEqual({ id: "r1", error: `decide: decision_request_failed: ${message}` });
    expect(calls).toHaveLength(0);
  }
  const calls: FetchCall[] = [];
  const res = (await decide(recordingFetch(calls, () =>
    chat(completion(JSON.stringify({ answers: { m: { type: "score", score: 1 } } })))), score(["Low", null]))) as DecideOk;
  expect(calls).toHaveLength(1);
  expect(res.decisions.m).toEqual({ kind: "score", score: 1 });
});

test("decision transport posts the configured Chat Completions request and normalizes usage", async () => {
  setDecisionEnv();
  const calls: FetchCall[] = [];
  const usage = { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 };
  const res = (await decide(recordingFetch(calls, () =>
    chat(completion(JSON.stringify({ answers: GOOD }), { model: "provider-model", usage }))))) as DecideOk;
  expect(calls).toHaveLength(1);
  expect(calls[0].url).toBe(DECISION_ENV.DECISION_API_URL);
  expect(calls[0].init.method).toBe("POST");
  expect(calls[0].init.headers).toEqual({
    Authorization: `Bearer ${DECISION_ENV.DECISION_API_KEY}`,
    "Content-Type": "application/json",
  });
  const body = JSON.parse(String(calls[0].init.body));
  expect(Object.keys(body).sort()).toEqual(["messages", "model", "stream"]);
  expect(body.model).toBe("decision-test-model");
  expect(body.stream).toBe(false);
  expect(body.messages.map((m: { role: string }) => m.role)).toEqual(["system", "user"]);
  expect(typeof body.messages[0].content).toBe("string");
  expect(JSON.parse(body.messages[1].content)).toEqual({ state: { s: 1 }, questions: Q });
  expect(res.raw).toEqual({ answers: GOOD, model: "provider-model", usage: { input_tokens: 12, output_tokens: 5 } });
  expect(res.decisions.truth).toEqual({ kind: "noul", probability: 0.7 });
  expect(res.decisions.route).toEqual({ kind: "choice", choice: "a", probabilities: { a: 0.6, b: 0.4 }, confidence: 0.6 });
});
test("successful 200 responses ignore a Location header", async () => {
  for (const format of ["chat_completions", "decisions"] as const) {
    setDecisionEnv({ DECISION_API_FORMAT: format });
    const calls: FetchCall[] = [];
    const body = format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD }));
    const res = (await decide(recordingFetch(calls, () => new Response(JSON.stringify(body), {
      status: 200,
      headers: { Location: "https://decision.test/not-a-redirect" },
    })))) as DecideOk;
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(DECISION_ENV.DECISION_API_URL);
    expect(res.decisions.truth).toEqual({ kind: "noul", probability: 0.7 });
    expect(res.decisions.route).toEqual({ kind: "choice", choice: "a", probabilities: { a: 0.6, b: 0.4 }, confidence: 0.6 });
    expect(res.raw).toMatchObject({ answers: GOOD });
  }
});

test("decision transport records DECISION_MODEL and no usage when the provider omits them", async () => {
  setDecisionEnv();
  const res = (await decide(recordingFetch([], () => chat(completion(JSON.stringify({ answers: GOOD })))))) as DecideOk;
  expect(res.raw).toEqual({ answers: GOOD, model: "decision-test-model" });
});

test("decision transport fails before any request when configuration is blank", async () => {
  for (const name of Object.keys(DECISION_ENV)) {
    for (const blank of ["", "   "]) {
      setDecisionEnv({ [name]: blank });
      const calls: FetchCall[] = [];
      const res = await decide(recordingFetch(calls, () => chat(completion(JSON.stringify({ answers: GOOD })))));
      expect(res).toEqual({ id: "r1", error: `decide: decision_request_failed: missing ${name}` });
      expect(calls).toHaveLength(0);
    }
  }
});

test("decision provider failures are request failures and never echo the key", async () => {
  const err = await decideError({ error: { message: "bad key sk-decision-test-key" } }, 401);
  expect(err).toStartWith("decide: decision_request_failed: HTTP 401: ");
  expect(err).not.toContain(DECISION_ENV.DECISION_API_KEY);
  setDecisionEnv();
  const res = await decide((async () => {
    throw new Error("connection refused");
  }) as unknown as typeof fetch);
  expect(res).toEqual({ id: "r1", error: "decide: decision_request_failed: connection refused" });
  const leaky = await decide((async (_url: string, init: RequestInit) => {
    throw new TypeError(`invalid header value ${JSON.stringify(init.headers)}`);
  }) as unknown as typeof fetch);
  if (!("error" in leaky)) throw new Error("expected decide error");
  expect(leaky.error).toStartWith("decide: decision_request_failed: invalid header value ");
  expect(leaky.error).toContain("Bearer [redacted]");
  expect(leaky.error).not.toContain(DECISION_ENV.DECISION_API_KEY);
});

test("malformed decision responses are rejected", async () => {
  const answers = JSON.stringify({ answers: GOOD });
  const cases: [unknown, string][] = [
    ["not json", "body is not JSON"],
    [{}, "missing choices"],
    [{ choices: [] }, "missing choices"],
    [completion(answers, {}, "length"), "truncated output"],
    [{ choices: [{ finish_reason: "stop" }] }, "missing message"],
    [{ choices: [{ finish_reason: "length", message: null }] }, "truncated output"],
    [{ choices: [{ message: { content: null, refusal: ["blocked"] } }] }, "refusal"],
    [{ choices: [{ message: { role: "assistant", content: null, refusal: "I cannot help." }, finish_reason: "stop" }] }, "refusal"],
    [completion(null), "content must be a nonempty string"],
    [completion("  "), "content must be a nonempty string"],
    [completion("```json\n{}\n```"), "content is not JSON"],
    [completion("[]"), "content is not a JSON object"],
  ];
  for (const [body, reason] of cases)
    expect(await decideError(body)).toBe(`decide: malformed_decision_response: ${reason}`);
  for (const content of ["{}", JSON.stringify({ answers: [] })])
    expect(await decideError(completion(content))).toBe("decide: malformed_decision_response");
});

test("null response parts are malformed responses, not request failures", async () => {
  const msg = (message: unknown) => ({ choices: [{ index: 0, message, finish_reason: "stop" }] });
  for (const body of ["null", { choices: null }, { choices: [null] }, msg(null), msg([])])
    expect(await decideError(body)).toStartWith("decide: malformed_decision_response");
});

test("refusal follows truthiness: empty refusals pass, any truthy refusal fails", async () => {
  const content = JSON.stringify({ answers: GOOD });
  const withRefusal = (refusal: unknown) => ({
    choices: [{ index: 0, message: { role: "assistant", content, refusal }, finish_reason: "stop" }],
  });
  for (const refusal of [null, "", false, 0, {}, []]) {
    setDecisionEnv();
    const res = await decide(recordingFetch([], () => chat(withRefusal(refusal))));
    expect(res).toMatchObject({ id: "r1", decisions: { truth: { kind: "noul", probability: 0.7 } } });
  }
  for (const refusal of ["I cannot help.", true, 1, { reason: "policy" }, ["policy"]])
    expect(await decideError(withRefusal(refusal))).toStartWith("decide: malformed_decision_response");
});

test("invalid provider model values fall back to DECISION_MODEL", async () => {
  for (const model of [null, "", 42, true, { id: "m" }, ["m"]]) {
    setDecisionEnv();
    const res = (await decide(recordingFetch([], () =>
      chat(completion(JSON.stringify({ answers: GOOD }), { model }))))) as DecideOk;
    expect(res.raw).toMatchObject({ model: DECISION_ENV.DECISION_MODEL });
  }
});

test("usage keeps only integer token counts", async () => {
  const rawWith = async (usage: unknown) => {
    setDecisionEnv();
    const res = (await decide(recordingFetch([], () =>
      chat(completion(JSON.stringify({ answers: GOOD }), { usage }))))) as DecideOk;
    return res.raw;
  };
  expect(await rawWith({ prompt_tokens: 1.5, completion_tokens: true })).toHaveProperty("usage", {});
  expect(await rawWith({ prompt_tokens: "12", completion_tokens: null })).toHaveProperty("usage", {});
  expect(await rawWith({ prompt_tokens: false, completion_tokens: 0 })).toHaveProperty("usage", { output_tokens: 0 });
  expect(await rawWith({ prompt_tokens: 7 })).toHaveProperty("usage", { input_tokens: 7 });
  expect(await rawWith({ input_tokens: 9, output_tokens: 4 })).toHaveProperty("usage", {});
  expect(await rawWith(null)).not.toHaveProperty("usage");
  expect(await rawWith([1, 2])).not.toHaveProperty("usage");
});

// Request and 200 response examples from the OpenRouter "Submit a Decisions request" reference.
const DOC_Q = {
  is_bug: {
    type: "noul",
    instructions: "Is the customer reporting a software defect?",
    criteria: {
      false: "The customer is asking a question or requesting a feature.",
      true: "The customer describes broken or unexpected product behavior.",
    },
  },
  team: {
    type: "choice",
    instructions: "Which team should own this ticket?",
    criteria: {
      account: "Login, permissions, or profile issues.",
      frontend: "Rendering, layout, or browser compatibility issues.",
      payments: "Checkout, billing, or payment processing issues.",
    },
  },
  urgency: {
    type: "score",
    instructions: "How urgent is this ticket?",
    criteria: ["Can wait for the next release", "Should be fixed this week", "Blocking revenue right now"],
  },
};
const DOC_STATE = {
  customer_tier: "enterprise",
  ticket: "My checkout page shows a blank screen after I click Pay. I have tried two browsers.",
};
const DOC_LEGEND = { 0: "Can wait for the next release", 1: "Should be fixed this week", 2: "Blocking revenue right now" };
const DOC_ANSWERS = {
  is_bug: { noul: 0.96, type: "noul" },
  team: { choice: "payments", confidence: 0.75, probabilities: { account: 0, frontend: 0.16, payments: 0.84 }, type: "choice" },
  urgency: { confidence: 0.99, legend: DOC_LEGEND, probabilities: { 0: 0, 1: 0.01, 2: 0.99 }, score: 1.99, type: "score" },
};
function nativeResponse(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    answers: DOC_ANSWERS,
    id: "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
    model: "typesafe/jev-1.13-20260917",
    provider: "TypeSafe",
    usage: { cost: 0.000019992, input_tokens: 476, output_tokens: 70 },
    ...extra,
  };
}

function nativeDecide(fetchFn: typeof fetch, questions: Record<string, unknown> = DOC_Q): Promise<DecideResponse> {
  setDecisionEnv(NATIVE_ENV);
  return handleDecide({ id: "r1", op: "decide", state: DOC_STATE, questions }, createSystemOne(fetchFn));
}

async function nativeError(body: unknown, status = 200): Promise<string> {
  const res = await nativeDecide(recordingFetch([], () => chat(body, status)));
  if (!("error" in res)) throw new Error(`expected decide error, got ${JSON.stringify(res)}`);
  expect(res.id).toBe("r1");
  return res.error;
}

test("native Decisions format posts {model,state,questions} for typesafe/jev-1.13 and keeps document answers", async () => {
  const calls: FetchCall[] = [];
  const res = (await nativeDecide(recordingFetch(calls, () => chat(nativeResponse())))) as DecideOk;
  expect(calls).toHaveLength(1);
  expect(calls[0].url).toBe(NATIVE_ENV.DECISION_API_URL);
  expect(calls[0].init.method).toBe("POST");
  expect(calls[0].init.headers).toEqual({
    Authorization: `Bearer ${DECISION_ENV.DECISION_API_KEY}`,
    "Content-Type": "application/json",
  });
  // The reported HTTP 400 came from Chat Completions fields on the native endpoint.
  expect(JSON.parse(String(calls[0].init.body))).toEqual({ model: "typesafe/jev-1.13", state: DOC_STATE, questions: DOC_Q });
  expect(res.raw).toEqual({
    answers: DOC_ANSWERS,
    model: "typesafe/jev-1.13-20260917",
    usage: { input_tokens: 476, output_tokens: 70 },
  });
  expect(res.decisions).toEqual({
    is_bug: { kind: "noul", probability: 0.96 },
    team: { kind: "choice", choice: "payments", probabilities: { account: 0, frontend: 0.16, payments: 0.84 }, confidence: 0.75 },
    urgency: { kind: "score", score: 1.99, confidence: 0.99, probabilities: { 0: 0, 1: 0.01, 2: 0.99 }, raw: DOC_LEGEND },
  });
});

test("DECISION_API_FORMAT alone selects the wire format, never the URL or model", async () => {
  const chatCalls: FetchCall[] = [];
  setDecisionEnv({ DECISION_API_URL: NATIVE_ENV.DECISION_API_URL, DECISION_MODEL: NATIVE_ENV.DECISION_MODEL });
  const chatRes = (await decide(recordingFetch(chatCalls, () => chat(completion(JSON.stringify({ answers: GOOD })))))) as DecideOk;
  expect(Object.keys(JSON.parse(String(chatCalls[0].init.body))).sort()).toEqual(["messages", "model", "stream"]);
  expect(chatRes.raw).toEqual({ answers: GOOD, model: "typesafe/jev-1.13" });
  const nativeCalls: FetchCall[] = [];
  setDecisionEnv({ DECISION_API_FORMAT: "decisions" });
  const nativeRes = (await decide(recordingFetch(nativeCalls, () => chat({ answers: GOOD })))) as DecideOk;
  expect(nativeCalls[0].url).toBe(DECISION_ENV.DECISION_API_URL);
  expect(JSON.parse(String(nativeCalls[0].init.body))).toEqual({ model: "decision-test-model", state: { s: 1 }, questions: Q });
  expect(nativeRes.raw).toEqual({ answers: GOOD, model: "decision-test-model" });
});

test("missing or unsupported DECISION_API_FORMAT fails before any request", async () => {
  const attempt = async (format: string | undefined): Promise<DecideResponse> => {
    setDecisionEnv();
    if (format === undefined) delete process.env.DECISION_API_FORMAT;
    else process.env.DECISION_API_FORMAT = format;
    const calls: FetchCall[] = [];
    const res = await decide(recordingFetch(calls, () => chat(nativeResponse())));
    expect(calls).toHaveLength(0);
    return res;
  };
  expect(await attempt(undefined)).toEqual({ id: "r1", error: "decide: decision_request_failed: missing DECISION_API_FORMAT" });
  for (const format of ["DECISIONS", "Chat_Completions", "chat", "responses", "auto", "decisions,chat_completions"])
    expect(await attempt(format)).toEqual({ id: "r1", error: "decide: decision_request_failed: unsupported DECISION_API_FORMAT" });
});

test("REASONING_EFFORT maps per decision format; a bad shape fails before any request", async () => {
  const reply = (format: string) => () => chat(format === "decisions" ? { answers: GOOD } : completion(JSON.stringify({ answers: GOOD })));
  try {
    process.env.REASONING_EFFORT = "xhigh";
    for (const format of ["chat_completions", "decisions"]) {
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const calls: FetchCall[] = [];
      await decide(recordingFetch(calls, reply(format)));
      const body = JSON.parse(String(calls[0].init.body));
      if (format === "chat_completions") expect(body.reasoning_effort).toBe("xhigh");
      // The Decisions API documents no reasoning field, so the native body stays exact.
      else expect(Object.keys(body).sort()).toEqual(["model", "questions", "state"]);
    }
    process.env.REASONING_EFFORT = "x high";
    for (const format of ["chat_completions", "decisions"]) {
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const calls: FetchCall[] = [];
      expect(await decide(recordingFetch(calls, reply(format)))).toEqual({
        id: "r1", error: "decide: decision_request_failed: REASONING_EFFORT must match ^[a-z][a-z0-9_-]{0,31}$",
      });
      expect(calls).toHaveLength(0);
    }
  } finally {
    delete process.env.REASONING_EFFORT;
  }
});

test("malformed native Decisions envelopes are rejected", async () => {
  expect(await nativeError("not json")).toBe("decide: malformed_decision_response: body is not JSON");
  for (const body of ["null", "[]", "42", '"answers"'])
    expect(await nativeError(body)).toBe("decide: malformed_decision_response: body is not a JSON object");
  // A Chat Completions envelope carries no top-level answers.
  const chatShaped = completion(JSON.stringify({ answers: DOC_ANSWERS }), { model: "m" });
  for (const body of [{}, { answers: null }, { answers: [] }, { answers: "x" }, chatShaped])
    expect(await nativeError(body)).toBe("decide: malformed_decision_response");
  // A native envelope is not a Chat Completions response.
  expect(await decideError({ answers: GOOD })).toBe("decide: malformed_decision_response: missing choices");
  const err = await nativeError({ error: { code: 400, message: "bad sk-decision-test-key" } }, 400);
  expect(err).toStartWith("decide: decision_request_failed: HTTP 400: ");
  expect(err).not.toContain(DECISION_ENV.DECISION_API_KEY);
});

test("native Decisions answers keep exact question, option, and probability validation", async () => {
  const reply = (answers: Record<string, unknown>) => nativeResponse({ answers });
  expect(await nativeError(reply({ ...DOC_ANSWERS, extra: { type: "noul", noul: 0.5 } })))
    .toBe("decide: decision answers do not match questions");
  expect(await nativeError(reply({ is_bug: DOC_ANSWERS.is_bug, team: DOC_ANSWERS.team })))
    .toBe("decide: decision answers do not match questions");
  expect(await nativeError(reply({ ...DOC_ANSWERS, is_bug: { type: "noul", noul: 1.0001 } }))).toBe("invalid_probability");
  const team = DOC_ANSWERS.team;
  for (const bad of [
    { ...team, choice: "legal" },
    { ...team, probabilities: { account: 0, payments: 1 } },
    { ...team, probabilities: { ...team.probabilities, frontend: -0.01 } },
    { ...team, confidence: 1.5 },
  ])
    expect(await nativeError(reply({ ...DOC_ANSWERS, team: bad }))).toBe("decide: malformed_decision_answer for team");
  const urgency = DOC_ANSWERS.urgency;
  for (const bad of [{ ...urgency, score: 2.01 }, { ...urgency, probabilities: { ...urgency.probabilities, 2: 1.2 } }])
    expect(await nativeError(reply({ ...DOC_ANSWERS, urgency: bad }))).toBe("decide: malformed_decision_answer for urgency");
});

test("native Decisions metadata keeps a valid model and only integer native token counts", async () => {
  const rawWith = async (extra: Record<string, unknown>) =>
    ((await nativeDecide(recordingFetch([], () => chat(nativeResponse(extra))))) as DecideOk).raw;
  for (const model of [null, "", 42, true, { id: "m" }, ["m"]])
    expect(await rawWith({ model })).toMatchObject({ model: NATIVE_ENV.DECISION_MODEL });
  expect(await rawWith({ model: undefined })).toMatchObject({ model: NATIVE_ENV.DECISION_MODEL });
  expect(await rawWith({ usage: { input_tokens: 1.5, output_tokens: true } })).toHaveProperty("usage", {});
  expect(await rawWith({ usage: { input_tokens: "12", output_tokens: null } })).toHaveProperty("usage", {});
  expect(await rawWith({ usage: { input_tokens: false, output_tokens: 0 } })).toHaveProperty("usage", { output_tokens: 0 });
  expect(await rawWith({ usage: { input_tokens: 7, cost: 0.1 } })).toHaveProperty("usage", { input_tokens: 7 });
  expect(await rawWith({ usage: { prompt_tokens: 3, completion_tokens: 4 } })).toHaveProperty("usage", {});
  expect(await rawWith({ usage: null })).not.toHaveProperty("usage");
  expect(await rawWith({ usage: [1, 2] })).not.toHaveProperty("usage");
  expect(await rawWith({ usage: undefined })).not.toHaveProperty("usage");
});

test("both formats preserve arbitrary probabilities and integer token counts", async () => {
  await fc.assert(fc.asyncProperty(
    fc.constantFrom("chat_completions", "decisions"),
    fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
    fc.nat(),
    fc.nat(),
    async (format, p, inTokens, outTokens) => {
      const answers = { truth: { type: "noul", noul: p }, route: { ...GOOD.route, probabilities: { a: p, b: 1 - p } } };
      const body = format === "decisions"
        ? { answers, usage: { input_tokens: inTokens, output_tokens: outTokens } }
        : completion(JSON.stringify({ answers }), { usage: { prompt_tokens: inTokens, completion_tokens: outTokens } });
      setDecisionEnv({ DECISION_API_FORMAT: format });
      const res = (await decide(recordingFetch([], () => chat(body))));
      if (!("raw" in res)) throw new Error(`expected decide success, got ${JSON.stringify(res)}`);
      expect(res.raw).toEqual({ answers, model: DECISION_ENV.DECISION_MODEL, usage: { input_tokens: inTokens, output_tokens: outTokens } });
      expect(res.decisions.truth).toEqual({ kind: "noul", probability: p });
      expect(res.decisions.route).toMatchObject({ probabilities: { a: p, b: 1 - p } });
    },
  ), { seed: 20261013, numRuns: 100 });
});

test("HTTP error bodies are capped at 300 characters", async () => {
  const err = await decideError("x".repeat(1000), 500);
  expect(err).toStartWith("decide: decision_request_failed: HTTP 500: ");
  expect(err).toContain("x".repeat(300));
  expect(err).not.toContain("x".repeat(301));
});

test("decision answers keep exact question and option validation", async () => {
  const reply = (a: Record<string, unknown>) => completion(JSON.stringify({ answers: a }));
  expect(await decideError(reply({ ...GOOD, extra: { type: "noul", noul: 0.5 } })))
    .toBe("decide: decision answers do not match questions");
  expect(await decideError(reply({ truth: GOOD.truth })))
    .toBe("decide: decision answers do not match questions");
  expect(await decideError(reply({ ...GOOD, route: { ...GOOD.route, choice: "c" } })))
    .toBe("decide: malformed_decision_answer for route");
  expect(await decideError(reply({ ...GOOD, route: { ...GOOD.route, probabilities: { a: 1 } } })))
    .toBe("decide: malformed_decision_answer for route");
  expect(await decideError(reply({ ...GOOD, truth: { type: "noul", noul: 1.0001 } }))).toBe("invalid_probability");
  for (const noul of [0, 1]) {
    setDecisionEnv();
    const res = (await decide(recordingFetch([], () => chat(reply({ ...GOOD, truth: { type: "noul", noul } }))))) as DecideOk;
    expect(res.decisions.truth).toEqual({ kind: "noul", probability: noul });
  }
});

test("experiment path sends direct question objects through the decision transport", async () => {
  setDecisionEnv();
  const calls: FetchCall[] = [];
  const stub = stubSystemOne();
  const chatFetch = recordingFetch(calls, async (init) => {
    const user = JSON.parse(JSON.parse(String(init.body)).messages[1].content);
    const out = (await stub(user)) as { answers: unknown };
    return chat(completion(JSON.stringify({ answers: out.answers })));
  });
  const dir = await successDir("sk-test-transport-key", createSystemOne(chatFetch));
  expect(calls.map((c) => c.url)).toEqual(Array(3).fill(DECISION_ENV.DECISION_API_URL));
  const final = JSON.parse(await readFile(join(dir, "final.json"), "utf8"));
  expect(final.reasoning[0].materialityScore).toBe(3);
  expect(final.jev.relevance.model).toBe("decision-test-model");
  const files = await readdir(dir);
  const decideReq = JSON.parse(await readFile(join(dir, files.find((f) => f.startsWith("request-decide-")) as string), "utf8"));
  expect(decideReq.questions.truth.type).toBe("noul");
  expect(decideReq.questions.evidence_state.type).toBe("choice");
  expect(decideReq.questions.materiality).toMatchObject({ type: "score", criteria: [...MATERIALITY_LEVELS] });
  for (const name of files)
    expect(await readFile(join(dir, name), "utf8")).not.toContain(DECISION_ENV.DECISION_API_KEY);
});

test("experiment path sends direct question objects through the native Decisions transport", async () => {
  setDecisionEnv(NATIVE_ENV);
  const calls: FetchCall[] = [];
  const stub = stubSystemOne();
  const nativeFetch = recordingFetch(calls, async (init) => {
    const body = JSON.parse(String(init.body));
    expect(Object.keys(body).sort()).toEqual(["model", "questions", "state"]);
    expect(body.model).toBe(NATIVE_ENV.DECISION_MODEL);
    const out = (await stub(body)) as { answers: unknown };
    return chat({ answers: out.answers, usage: { input_tokens: 9, output_tokens: 2 } });
  });
  const dir = await successDir("sk-test-transport-key", createSystemOne(nativeFetch));
  expect(calls.map((c) => c.url)).toEqual(Array(3).fill(NATIVE_ENV.DECISION_API_URL));
  const final = JSON.parse(await readFile(join(dir, "final.json"), "utf8"));
  expect(final.reasoning[0].materialityScore).toBe(3);
  expect(final.jev.relevance.model).toBe(NATIVE_ENV.DECISION_MODEL);
  for (const name of await readdir(dir))
    expect(await readFile(join(dir, name), "utf8")).not.toContain(DECISION_ENV.DECISION_API_KEY);
});
