// decision/jev.test.ts — typed JEV layer contracts. No live calls: stub
// SystemOneFn, temp dirs under os.tmpdir for askDecisions file writes.
//
// Routing: exclusive routing/category/explanation → choice (exactly one
// winner among mutually exclusive labels); independent propositions → noul
// (each id its own probability, no cross-id winner); ordinal judgments →
// score (never deterministic calcs).
import { expect, test } from "bun:test";
import { mkdtemp, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import fc from "fast-check";
import {
  assertAcyclic,
  askDecisions,
  buildToolSelectionQuestion,
  classifyProbability,
  DISPOSITION_OPTIONS,
  EVIDENCE_STATE_OPTIONS,
  MATERIALITY_LEVELS,
  parseChoiceAnswer,
  parseNoulAnswer,
  parseScoreAnswer,
  scopeNodeState,
  TOOL_SELECTION_SENTINELS,
} from "./jev.ts";
import { analyzePrompt, decomposePrompt, expandPrompt } from "./prompts.ts";

test("noul preserves probability exactly; policy stays in classifyProbability", () => {
  for (const p of [0, 0.5, 0.6999, 0.7, 0.85, 1]) {
    const d = parseNoulAnswer("n", "truth", { type: "noul", noul: p });
    expect(d).toEqual({ kind: "noul", probability: p });
  }
  // Boundary policy is not collapsed into the parse result.
  expect(classifyProbability(0.7)).toBe("yes");
  expect(classifyProbability(0.6999)).toBe("unsure");
  expect(classifyProbability(0.5)).toBe("unsure");
  expect(classifyProbability(0.4999)).toBe("no");
  for (const bad of [-0.1, 1.1, NaN, Infinity, -Infinity])
    expect(() => parseNoulAnswer("n", "truth", { type: "noul", noul: bad })).toThrow("invalid_probability");
  expect(() => parseNoulAnswer("n", "truth", { type: "choice", choice: "x" })).toThrow("malformed");
});

const dispProbs = { analyze: 0.7, gather_evidence: 0.2, reject: 0.1 };
const evProbs = {
  sufficient_support: 0.6,
  sufficient_contradiction: 0.1,
  conflicted: 0.2,
  insufficient: 0.1,
};

test("choice winner must be declared; distribution exactly matches options", () => {
  const d = parseChoiceAnswer("n", "id", {
    type: "choice",
    choice: "analyze",
    probabilities: dispProbs,
    confidence: 0.8,
  }, DISPOSITION_OPTIONS);
  expect(d).toEqual({ kind: "choice", choice: "analyze", probabilities: dispProbs, confidence: 0.8 });
  const e = parseChoiceAnswer("n", "id", {
    type: "choice",
    choice: "conflicted",
    probabilities: evProbs,
    confidence: 0.4,
  }, EVIDENCE_STATE_OPTIONS);
  expect(e.choice).toBe("conflicted");
  expect(e.confidence).toBe(0.4);
  // Unknown winner rejected.
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "bogus", probabilities: dispProbs, confidence: 0.8,
  }, DISPOSITION_OPTIONS)).toThrow("malformed");
  // Key set must be exactly the declared options.
  const missing = { analyze: 0.9, gather_evidence: 0.1 };
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "analyze", probabilities: missing, confidence: 0.8,
  }, DISPOSITION_OPTIONS)).toThrow("malformed");
  const extra = { ...dispProbs, other: 0 };
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "analyze", probabilities: extra, confidence: 0.8,
  }, DISPOSITION_OPTIONS)).toThrow("malformed");
  // Values must be probabilities in [0,1]; confidence too.
  const outOfRange = { ...dispProbs, analyze: 1.5 };
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "analyze", probabilities: outOfRange, confidence: 0.8,
  }, DISPOSITION_OPTIONS)).toThrow("malformed");
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "analyze", probabilities: dispProbs, confidence: NaN,
  }, DISPOSITION_OPTIONS)).toThrow("malformed");
});

test("choice rejects malformed answer and probability shapes", () => {
  for (const ans of [null, ["choice"], "choice"])
    expect(() => parseChoiceAnswer("n", "id", ans, DISPOSITION_OPTIONS)).toThrow("n: malformed_typesafe_answer for id");
  expect(() => parseChoiceAnswer("n", "id", {
    type: "noul", choice: "analyze", probabilities: dispProbs, confidence: 0.8,
  }, DISPOSITION_OPTIONS)).toThrow("n: malformed_typesafe_answer for id");
  for (const probabilities of [null, ["x"], "x"])
    expect(() => parseChoiceAnswer("n", "id", {
      type: "choice", choice: "analyze", probabilities, confidence: 0.8,
    }, DISPOSITION_OPTIONS)).toThrow("n: malformed_typesafe_answer for id");
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "analyze",
    probabilities: { analyze: 0.7, gather_evidence: 0.2, bogus: 0.1 }, confidence: 0.8,
  }, DISPOSITION_OPTIONS)).toThrow("n: malformed_typesafe_answer for id");
});

test("choice rejects non-number confidence with the existing error shape", () => {
  for (const confidence of ["0.8", null])
    expect(() => parseChoiceAnswer("n", "id", {
      type: "choice", choice: "analyze", probabilities: dispProbs, confidence,
    }, DISPOSITION_OPTIONS)).toThrow("n: malformed_typesafe_answer for id");
});

test("choice rejects inherited labels with the existing error shape", () => {
  for (const choice of ["toString", "constructor", "__proto__"]) {
    expect(() => parseChoiceAnswer("n", "id", {
      type: "choice", choice, probabilities: { safe: 0.7 }, confidence: 0.8,
    }, { safe: "Configured choice" })).toThrow("n: malformed_typesafe_answer for id");
  }
  const options = { safe: "Configured choice" };
  Object.setPrototypeOf(options, { inherited: "Not an own choice" });
  expect(() => parseChoiceAnswer("n", "id", {
    type: "choice", choice: "inherited", probabilities: { safe: 0.7 }, confidence: 0.8,
  }, options)).toThrow("n: malformed_typesafe_answer for id");
});

test("choice retains explicitly configured prototype-like labels and raw probabilities", () => {
  const labels = ["toString", "safe", "__proto__", "constructor", "10", "2"];
  const options = Object.fromEntries(labels.map((label) => [label, "Configured choice"]));
  const probabilities = Object.fromEntries(labels.map((label, i) => [label, i / 10]));
  for (const choice of labels) {
    const d = parseChoiceAnswer("n", "id", {
      type: "choice", choice, probabilities, confidence: 0.37,
    }, options);
    expect(d.choice).toBe(choice);
    expect(d.confidence).toBe(0.37);
    expect(d.probabilities).toEqual(probabilities);
    expect(Object.keys(d.probabilities)).toEqual(["2", "10", "__proto__", "constructor", "safe", "toString"]);
    expect(Object.getPrototypeOf(d.probabilities)).toBe(Object.prototype);
    expect(JSON.parse(JSON.stringify(d.probabilities))).toEqual(probabilities);
    for (const label of labels) {
      expect(Object.hasOwn(d.probabilities, label)).toBe(true);
      expect(Object.getOwnPropertyDescriptor(d.probabilities, label)).toEqual({
        value: probabilities[label], enumerable: true, writable: true, configurable: true,
      });
    }
  }
});

test("choice preserves arbitrary own labels and their raw probabilities", () => {
  fc.assert(fc.property(
    fc.uniqueArray(fc.tuple(
      fc.oneof(fc.string(), fc.constantFrom("toString", "constructor", "__proto__")),
      fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
    ), { selector: ([label]) => label, minLength: 1, maxLength: 12 }),
    fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
    (entries, confidence) => {
      const options = Object.fromEntries(entries.map(([label]) => [label, "Configured choice"]));
      const probabilities = Object.fromEntries(entries);
      for (const [choice] of entries) {
        const d = parseChoiceAnswer("n", "id", {
          type: "choice", choice, probabilities, confidence,
        }, options);
        expect(d.choice).toBe(choice);
        expect(d.confidence).toBe(confidence);
        expect(d.probabilities).toEqual(probabilities);
        expect(Object.keys(d.probabilities).sort()).toEqual(Object.keys(options).sort());
        expect(Object.getPrototypeOf(d.probabilities)).toBe(Object.prototype);
        for (const [label, probability] of entries) {
          expect(Object.hasOwn(d.probabilities, label)).toBe(true);
          expect(d.probabilities[label]).toBe(probability);
        }
      }
    },
  ), { seed: 20261009, numRuns: 100 });
});

test("choice rejects arbitrary unknown and inherited labels", () => {
  fc.assert(fc.property(
    fc.oneof(fc.string(), fc.constantFrom("toString", "constructor", "__proto__"))
      .filter((choice) => choice !== "safe"),
    fc.double({ min: 0, max: 1, noNaN: true, noDefaultInfinity: true }),
    (choice, probability) => {
      expect(() => parseChoiceAnswer("n", "id", {
        type: "choice", choice, probabilities: { safe: probability }, confidence: 0.8,
      }, { safe: "Configured choice" })).toThrow("n: malformed_typesafe_answer for id");
    },
  ), { seed: 20261009, numRuns: 100 });
});

test("score keeps finite value; optional distribution/confidence/legend retained", () => {
  expect(parseScoreAnswer("n", "m", { type: "score", score: 3 })).toEqual({ kind: "score", score: 3 });
  const full = parseScoreAnswer("n", "m", {
    type: "score",
    score: 4,
    probabilities: { "0": 0.1, "4": 0.9 },
    confidence: 0.75,
    legend: MATERIALITY_LEVELS,
  });
  expect(full.score).toBe(4);
  expect(full.probabilities).toEqual({ "0": 0.1, "4": 0.9 });
  expect(full.confidence).toBe(0.75);
  expect(full.raw).toEqual(MATERIALITY_LEVELS);
  for (const bad of [
    { type: "score" },
    { type: "score", score: NaN },
    { type: "score", score: Infinity },
    { type: "score", score: "high" },
    { type: "score", score: 2, confidence: "high" },
    { type: "score", score: 2, probabilities: { "2": NaN } },
    { type: "score", score: 2, confidence: 1.5 },
    { type: "score", score: 2, probabilities: { "2": 1.5 } },
  ]) expect(() => parseScoreAnswer("n", "m", bad)).toThrow("malformed");
  expect(() => parseScoreAnswer("n", "m", { type: "score", score: 100 }, 4)).toThrow("malformed");
  expect(() => parseScoreAnswer("n", "m", { type: "score", score: -1 }, 4)).toThrow("malformed");
  expect(parseScoreAnswer("n", "m", { type: "score", score: 3.5 }, 4)).toEqual({ kind: "score", score: 3.5 });
});

test("mixed askDecisions parses noul+choice+score from one shared state", async () => {
  const d = await mkdtemp(join(tmpdir(), "jev-test-"));
  const questions = {
    truth: { type: "noul", instruction: "Do the facts support it?" },
    evidence_state: { type: "choice", instruction: "Evidence state?", criteria: EVIDENCE_STATE_OPTIONS },
    materiality: { type: "score", instruction: "How material?", levels: MATERIALITY_LEVELS },
  };
  const systemOne = async () => ({
    answers: {
      truth: { type: "noul", noul: 0.8 },
      evidence_state: { type: "choice", choice: "sufficient_support", probabilities: evProbs, confidence: 0.7 },
      materiality: { type: "score", score: 3, confidence: 0.6, legend: MATERIALITY_LEVELS },
    },
  });
  const { raw, decisions } = await askDecisions(
    d, "mixed", { state: { objective: "o" }, questions }, systemOne,
    { evidence_state: EVIDENCE_STATE_OPTIONS },
  );
  expect(decisions.truth).toEqual({ kind: "noul", probability: 0.8 });
  expect(decisions.evidence_state?.kind).toBe("choice");
  expect(decisions.materiality?.kind).toBe("score");
  // Out-of-range score against the shared rubric is rejected.
  const oor = async () => ({
    answers: {
      truth: { type: "noul", noul: 0.8 },
      evidence_state: { type: "choice", choice: "sufficient_support", probabilities: evProbs, confidence: 0.7 },
      materiality: { type: "score", score: 100, confidence: 0.6 },
    },
  });
  let oorErr: unknown;
  try {
    await askDecisions(d, "oor", { state: { objective: "o" }, questions }, oor, {
      evidence_state: EVIDENCE_STATE_OPTIONS,
    });
  } catch (e) {
    oorErr = e;
  }
  expect(String(oorErr)).toContain("malformed");
  expect(raw).toHaveProperty("answers");
  const files = await readdir(d);
  for (const f of ["request-mixed.json", "raw-mixed.json", "policy-mixed.json"])
    expect(files).toContain(f);
  const mismatch = async () => ({ answers: { truth: { type: "noul", noul: 0.5 } } });
  let shortErr: unknown;
  try {
    await askDecisions(d, "short", { state: {}, questions }, mismatch);
  } catch (e) {
    shortErr = e;
  }
  expect(String(shortErr)).toContain("do not match");
  const extra = async () => ({
    answers: {
      truth: { type: "noul", noul: 0.5 },
      evidence_state: { type: "choice", choice: "insufficient", probabilities: evProbs, confidence: 0.5 },
      materiality: { type: "score", score: 1 },
      stowaway: { type: "noul", noul: 0.5 },
    },
  });
  let longErr: unknown;
  try {
    await askDecisions(d, "long", { state: {}, questions }, extra, { evidence_state: EVIDENCE_STATE_OPTIONS });
  } catch (e) {
    longErr = e;
  }
  expect(String(longErr)).toContain("do not match");
});

test("assertAcyclic rejects self and indirect cycles, accepts diamonds", () => {
  expect(() => assertAcyclic("s", [{ id: "A", dependsOn: ["A"] }], () => undefined)).toThrow(
    "depends on itself",
  );
  expect(() => assertAcyclic(
    "s",
    [{ id: "A", dependsOn: ["B"] }, { id: "B", dependsOn: ["A"] }],
    () => undefined,
  )).toThrow("cyclic");
  expect(() => assertAcyclic(
    "s",
    [{ id: "A", dependsOn: ["B"] }, { id: "B", dependsOn: ["C"] }, { id: "C", dependsOn: ["A"] }],
    () => undefined,
  )).toThrow("cyclic");
  // Diamond passes.
  assertAcyclic(
    "s",
    [
      { id: "A", dependsOn: ["B", "C"] },
      { id: "B", dependsOn: ["D"] },
      { id: "C", dependsOn: ["D"] },
      { id: "D", dependsOn: [] },
    ],
    () => undefined,
  );
  // Prior-deps callback is consulted: unseen dep resolves through it.
  const seen: string[] = [];
  assertAcyclic("s", [{ id: "A", dependsOn: ["B"] }], (id) => {
    seen.push(id);
    return [];
  });
  expect(seen).toContain("B");
  // Cycle reachable only via prior still throws.
  expect(() => assertAcyclic("s", [{ id: "A", dependsOn: ["B"] }], (id) =>
    id === "B" ? ["A"] : [],
  )).toThrow("cyclic");
});

test("scopeNodeState projects only allowed keys; evidence must be an array", () => {
  const minimal = scopeNodeState({ objective: "o", proposal: "p", evidence: [] }) as Record<string, unknown>;
  expect(Object.keys(minimal).sort()).toEqual(["evidence", "objective", "proposal"]);
  const full = scopeNodeState({
    objective: "o",
    proposal: "p",
    analysis: "a",
    evidence: [{ id: "e1" }],
    dependencyDecisions: [{ proposalId: "q", disposition: "analyze" }],
  }) as Record<string, unknown>;
  expect(Object.keys(full).sort()).toEqual(
    ["analysis", "dependencyDecisions", "evidence", "objective", "proposal"],
  );
  // Nothing smuggled in (e.g. evaluationCriteria) passes through.
  const smuggled = {
    objective: "o",
    proposal: "p",
    evidence: [],
    evaluationCriteria: { foo: 1 },
  } as unknown as Parameters<typeof scopeNodeState>[0];
  expect(scopeNodeState(smuggled) as Record<string, unknown>).not.toHaveProperty("evaluationCriteria");
  expect(() =>
    scopeNodeState({ objective: "o", proposal: "p", evidence: "e1" as unknown as unknown[] }),
  ).toThrow("evidence must be an array");
});
test("tool selection covers the whole registry plus sentinels; enriched manifests render one line", () => {
  const registry = [
    { name: "search_sec_filings", description: "Full-text filing search." },
    {
      name: "get_sec_document",
      description: "Bounded text window of one filing document.",
      domain: "sec",
      purpose: "Read a known accession section.",
      intent: "read_filing_section",
      keyInputs: "req(accession_no) opt(section, cursor, limit)",
      outputKind: "text window + source_refs",
      evidence: "filing text with accession citations",
      prerequisites: "accession_no from a prior search",
      pitSupport: "as_of",
      useWhen: "reading a known section; following a search hit",
      avoidWhen: "do not use for discovery search",
      conflicts: "search_sec_filings",
      nextTools: "diff_sec_filings",
    },
  ];
  const q = buildToolSelectionQuestion(
    registry,
    { nodeId: "n1", question: "What changed in risk factors?" },
    [{ id: "e1" }],
    [{ tool: "search_sec_filings", error: "timeout" }],
  );
  expect(q.id).toBe("tool_selection");
  expect(Object.keys(q.options).sort()).toEqual(
    ["get_sec_document", "node_resolved", "reasoning_required", "search_sec_filings"].sort(),
  );
  expect(q.options.search_sec_filings).toBe("Full-text filing search.");
  const doc = q.options.get_sec_document;
  for (const bit of ["[sec]", "Intent:", "Inputs:", "Output:", "Evidence:", "Needs:", "PIT:", "Use:", "Avoid:", "Conflicts:", "Next:"]) expect(doc).toContain(bit);
  expect(q.prompt).toContain("n1");
  expect(q.prompt).toContain("search_sec_filings");
  expect(q.prompt).toContain("[Today UTC ");
  expect(q.prompt).toContain("Today UTC is");
  expect(q.prompt).toContain("decode relative dates before choosing");
  expect(q.prompt).toContain("Monday-now NYC range");
  expect(q.prompt).toContain("never pass phrases like 'this week' or 'today' or 'last quarter' as arg values");
  expect(q.prompt).toContain("page with research_read_search beyond display_limit");
  const rev = buildToolSelectionQuestion(registry, { nodeId: "n1", question: "What drove NVDA revenue last quarter?" }).prompt;
  expect(rev).toContain("open the 10-Q accession via get_sec_document with a revenue query");
  const plain = buildToolSelectionQuestion(registry, { nodeId: "n1", question: "q?" }).prompt;
  expect(plain).not.toContain("open the 10-Q accession");
  expect(TOOL_SELECTION_SENTINELS.reasoning_required.length).toBeGreaterThan(0);
  expect(TOOL_SELECTION_SENTINELS.node_resolved.length).toBeGreaterThan(0);
  for (const p of [decomposePrompt({}), analyzePrompt({}), expandPrompt({})]) {
    expect(p).toContain("[Today UTC ");
    expect(p).toContain("Today UTC is");
    expect(p).toContain("decode relative dates before choosing");
  }
  const stamped = decomposePrompt({ objective: { id: "o", prompt: "What changed last quarter?" } });
  expect(stamped).toContain("[Today UTC ");
  const pre = decomposePrompt({ objective: { id: "o", prompt: "[Today UTC 2026-01-01] What changed?" } });
  expect(pre.split("[Today UTC ").length).toBe(2);
  const once = buildToolSelectionQuestion(registry, { nodeId: "n1", question: "q?" }).prompt;
  expect(once.slice(once.indexOf("[Today UTC "), once.indexOf("]") + 1)).toContain(new Date().toISOString().slice(0, 10));
  expect(() => buildToolSelectionQuestion([], { nodeId: "n", question: "q" })).toThrow("empty registry");
  expect(() => buildToolSelectionQuestion(
    [{ name: "reasoning_required", description: "collision" }],
    { nodeId: "n", question: "q" },
  )).toThrow("sentinel");
});
