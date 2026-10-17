import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import type { Root } from "react-dom/client";
import { JSDOM } from "jsdom";
import type {
  EvalRun,
  EvalScenarioResult,
  Experiment,
  FailureRecord,
  HarnessProjection,
  ResearchRun,
} from "./harness-viewer-types";
import { HarnessSessionViewer, HarnessViewer } from "./harness-viewer";

const dom = new JSDOM(`<body><div id="root"></div></body>`, { url: "https://example.com/" });
const registry = globalThis as Record<string, unknown>;
const DOM_GLOBALS = ["IS_REACT_ACT_ENVIRONMENT", "window", "document", "navigator", "requestAnimationFrame", "cancelAnimationFrame"];
const savedGlobals = new Map<string, PropertyDescriptor | undefined>();
let mounted: Root | null = null;

beforeAll(() => {
  for (const key of DOM_GLOBALS) savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  registry.IS_REACT_ACT_ENVIRONMENT = true;
  registry.window = dom.window;
  registry.document = dom.window.document;
  registry.navigator = dom.window.navigator;
  registry.requestAnimationFrame = (callback: FrameRequestCallback) => setTimeout(() => callback(0), 0) as unknown as number;
  registry.cancelAnimationFrame = (frame: number) => clearTimeout(frame);
});

afterEach(async () => {
  await act(async () => mounted?.unmount());
  mounted = null;
});

afterAll(() => {
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else Reflect.deleteProperty(globalThis, key);
  }
  dom.window.close();
});

function run(sessionId: string, status = "open"): ResearchRun {
  return {
    sessionId,
    waveId: 1,
    question: `${sessionId} question?`,
    status,
    asOf: null,
    updatedAt: "2026-01-02T03:04:05+00:00",
    traceId: "t1",
    conclusion: `${sessionId} conclusion`,
    traceStatus: "open",
    provider: "openai",
    model: "m",
    jobs: [
      {
        jobId: `${sessionId}-j1`,
        parentJobId: null,
        jobType: "research",
        owner: "agent",
        waveId: 1,
        status: "done",
        failureCategory: null,
        failureMessage: null,
        assignmentId: "a1",
        role: "r1",
      },
    ],
    waves: [1],
    events: [{ seq: 1, eventType: "session_opened", payload: { sessionId } }],
    claims: [],
    evidence: [
      {
        evidenceId: `${sessionId}-e1`,
        subject: `${sessionId} subject`,
        knownAt: "2026-01-01",
        sourceName: "10-K",
        sourceUri: "https://www.sec.gov/filing",
        domain: "SEC",
      },
    ],
    freezes: [{ freezeId: `${sessionId}-f1`, evidenceIds: [`${sessionId}-e1`] }],
    dossiers: [],
    coverageArtifacts: [],
    finalResult: {
      answer: `${sessionId} answer`,
      executiveSummary: "",
      consensus: "",
      baseCase: "",
      bullCase: "",
      bearCase: "",
      disagreements: [],
      positioning: [],
      catalysts: [],
      uncertainties: [],
      whatChangesTheView: [],
      limitations: [],
      claims: [{ text: `${sessionId} claim`, evidenceIds: [`${sessionId}-e1`] }],
      sources: [],
      coverage: {},
      allowedSources: ["SEC"],
      freezeId: `${sessionId}-f1`,
      asOf: "2026-01-01",
    },
    committeeRuns: ["c1"],
  };
}

function evalRun(evalRunId: string): EvalRun {
  return {
    evalRunId,
    model: "m",
    provider: "openai",
    harnessVersion: "h1",
    promptVersion: "p1",
    gitSha: "g1",
    scenarioVersion: "s1",
    startedAt: "2026-01-02T00:00:00Z",
    passed: 1,
    failed: 1,
  };
}

function scenario(evalRunId: string, scenarioName: string, passed: boolean, violations: string[] = []): EvalScenarioResult {
  return { evalRunId, scenarioName, passed, violations };
}

function failure(evalRunId: string, failureId: string, scenarioName: string): FailureRecord {
  return { evalRunId, failureId, scenarioName, violation: `${scenarioName} violated` };
}

function experiment(experimentId: string, beforeRunId: string, afterRunId: string): Experiment {
  return { experimentId, beforeRunId, afterRunId, deltaPassed: 1 };
}

function projectionWith(overrides: Partial<HarnessProjection> = {}): HarnessProjection {
  return {
    researchRuns: [],
    evalRuns: [],
    evalScenarioResults: [],
    experiments: [],
    failureRecords: [],
    ...overrides,
  };
}

async function mount(element: ReactNode): Promise<HTMLElement> {
  const container = dom.window.document.getElementById("root");
  if (!container) throw new Error("missing test root");
  mounted = createRoot(container);
  await act(async () => mounted?.render(element));
  return container;
}

function buttons(root: HTMLElement, label: string): HTMLButtonElement[] {
  return [...root.querySelectorAll("button")].filter((entry) => entry.textContent === label) as HTMLButtonElement[];
}

async function click(target: HTMLButtonElement): Promise<void> {
  await act(async () => {
    target.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
}

// Bounded wait for a rendered state. The tick cap only stops a hung test.
async function settle(ready: () => boolean): Promise<void> {
  for (let tick = 0; tick < 50 && !ready(); tick++) {
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
  }
  if (!ready()) throw new Error("the viewer did not reach the expected state");
}

describe("HarnessViewer", () => {
  test("renders current sessions, evidence drill, first eval run results, and scenario violations", async () => {
    const projection = projectionWith({
      researchRuns: [run("s1"), run("s2")],
      evalRuns: [evalRun("e1"), evalRun("e2")],
      evalScenarioResults: [
        scenario("e1", "sec-only", true),
        scenario("e1", "bankruptcy", false, ["allowed source violated"]),
        scenario("e2", "other", true),
      ],
      experiments: [experiment("x1", "e1", "e2")],
      failureRecords: [failure("e1", "f1", "bankruptcy"), failure("e2", "f2", "other")],
    });
    const root = await mount(<HarnessViewer projection={projection} />);
    expect(root.textContent).toContain("2 sessions (read-only projections from trace persistence)");
    expect(root.textContent).toContain("2 evidence records with freeze-contained claim links below.");
    expect(root.textContent).toContain("2 eval runs, 1 experiments");
    expect(root.textContent).toContain("PASS sec-only");
    expect(root.textContent).toContain("FAIL bankruptcy");
    expect(root.textContent).toContain("allowed source violated");
    expect(root.textContent).toContain("x1 e1 → e2 delta 1");
    expect(root.textContent).toContain("f1 bankruptcy bankruptcy violated");
    expect(root.textContent).not.toContain("PASS other");
    expect(root.textContent).not.toContain("f2 other");
  });
  test("empty projection shows missing sessions, evidence, and eval states", async () => {
    const root = await mount(<HarnessViewer projection={projectionWith()} />);
    expect(root.textContent).toContain("No research runs persisted yet.");
    expect(root.textContent).toContain("No trace persisted for this session (research://unselected).");
    expect(root.textContent).toContain("No freeze-contained evidence persisted yet.");
    expect(root.textContent).toContain("0 eval runs, 0 experiments");
  });


  test("shows empty and failed states for sessions, evidence, and evals", async () => {
    const failed = { ...run("s-failed", "failed"), evidence: [], freezes: [], traceId: null as string | null };
    failed.jobs = [
      {
        jobId: "s-failed-j1",
        parentJobId: null,
        jobType: "research",
        owner: "agent",
        waveId: 1,
        status: "failed",
        failureCategory: "tool_error",
        failureMessage: "err",
        assignmentId: "a1",
        role: "r1",
      },
    ];
    failed.finalResult = null;
    const root = await mount(
      <HarnessViewer
        projection={projectionWith({ researchRuns: [failed], evalRuns: [evalRun("e1")] })}
        sessionId="s-failed"
      />,
    );
    expect(root.textContent).toContain("Trace missing for this run; showing Job persistence only.");
    expect(root.textContent).toContain("Run failed; terminal failures listed under Jobs without exceptions.");
    expect(root.textContent).toContain("Empty evidence for this run.");
    expect(root.textContent).toContain("No freeze persisted.");
    expect(root.textContent).toContain("No final persisted.");
    expect(root.textContent).toContain("tool_error: err");
    expect(root.textContent).toContain("No freeze-contained evidence persisted yet.");
  });

  test("unknown session shows missing state, never the first run", async () => {
    const root = await mount(<HarnessViewer projection={projectionWith({ researchRuns: [run("s1")] })} sessionId="missing" />);
    expect(root.textContent).toContain("No trace persisted for this session (research://missing).");
    expect(root.textContent).not.toContain("Question: s1 question?");
    expect(root.textContent).not.toContain("s1 answer");
  });

  test("selecting sessions and eval runs switches timelines, results, and failure records", async () => {
    const projection = projectionWith({
      researchRuns: [run("s1"), run("s2")],
      evalRuns: [evalRun("e1"), evalRun("e2")],
      evalScenarioResults: [scenario("e1", "first", true), scenario("e2", "second", false, ["bad"])],
      experiments: [],
      failureRecords: [failure("e1", "f1", "first"), failure("e2", "f2", "second")],
    });
    const root = await mount(<HarnessViewer projection={projection} sessionId="s1" />);
    expect(root.textContent).toContain("Question: s1 question?");
    await click(buttons(root, "s2")[0]);
    expect(root.textContent).toContain("Question: s2 question?");
    expect(root.textContent).toContain("s2 conclusion");
    await click(buttons(root, "e2")[0]);
    expect(root.textContent).toContain("FAIL second");
    expect(root.textContent).toContain("f2 second second violated");
    expect(root.textContent).not.toContain("PASS first");
  });

  test("session viewer reads one durable snapshot and keeps missing sessions missing", async () => {
    const savedFetch = globalThis.fetch;
    const snapshot = {
      session: { session_id: "live-session", status: "researching", query: "live question?", updated_at: "2026-01-02T03:04:05+00:00" },
      jobs: [{ job_id: "live-job", job_type: "source_agent", status: "running", owner: "kernel" }],
    };
    globalThis.fetch = (async (url: unknown) => {
      const path = String(url);
      if (path.includes(encodeURIComponent("live-session"))) {
        return { ok: true, json: async () => snapshot };
      }
      return { ok: false, status: 404, json: async () => ({ error: "unknown session: 'rs:missing'" }) };
    }) as typeof globalThis.fetch;
    try {
      const root = await mount(<HarnessSessionViewer sessionId="live-session" />);
      await act(async () => { });
      expect(root.textContent).toContain("Question: live question?");
      expect(root.textContent).toContain("live-job");
      expect(globalThis.fetch).not.toBe(savedFetch);
      await act(async () => mounted?.unmount());
      const missing = await mount(<HarnessSessionViewer sessionId="rs:missing" />);
      await act(async () => { });
      expect(missing.textContent).toContain("Snapshot unavailable for this session (research://rs:missing).");
      expect(missing.textContent).not.toContain("live question?");
    } finally {
      globalThis.fetch = savedFetch;
    }
  });

  test("renders a complete final result with narrative, claims, sources, and audit detail", async () => {
    const rich = run("s-rich");
    rich.finalResult = {
      ...rich.finalResult as NonNullable<ResearchRun["finalResult"]>,
      executiveSummary: "Margins held while volumes fell.",
      consensus: "Both seats agree on cash flow.",
      baseCase: "Flat revenue.",
      bullCase: "Volume recovery.",
      bearCase: "Further price cuts.",
      disagreements: ["Seat B disputes the margin path."],
      positioning: ["Long the 2031 notes."],
      catalysts: ["Q3 print."],
      uncertainties: ["Export rules."],
      whatChangesTheView: ["Guidance cut."],
      limitations: ["No intraday prices."],
      claims: [
        { text: "Margin held", evidenceIds: ["s-rich-e1", "s-outside"] },
        { text: "Volumes fell", evidenceIds: ["s-outside"] },
      ],
      sources: [
        { evidenceId: "s-rich-e1", domain: "SEC", document: "10-K FY2026", integrityClass: "filed" },
        { evidenceId: "s-outside", domain: "WEB", document: "Blog post" },
      ],
      coverage: { ratio: 0.5 },
      allowedSources: ["SEC", "WEB"],
    };
    rich.coverageArtifacts = [{ artifactId: "artifact-1", waveId: 1, searchId: "search-1", query: "margin", claimText: "Margin held" }];
    rich.events = [{ seq: 2, eventType: "freeze", payload: { freezeId: "s-rich-f1" } }, { seq: 1, eventType: "session_opened", payload: { sessionId: "s-rich" } }];
    const root = await mount(<HarnessViewer projection={projectionWith({ researchRuns: [rich] })} sessionId="s-rich" />);
    expect(root.textContent).toContain("Bottom line: Margins held while volumes fell.");
    expect(root.textContent).toContain("Consensus: Both seats agree on cash flow.");
    expect(root.textContent).toContain("Base case: Flat revenue.");
    expect(root.textContent).toContain("Bull case: Volume recovery.");
    expect(root.textContent).toContain("Bear case: Further price cuts.");
    expect(root.textContent).toContain("Positioning");
    expect(root.textContent).toContain("What changes the view");
    expect(root.textContent).toContain("Seat B disputes the margin path.");
    expect(root.textContent).not.toContain("No disagreements persisted.");
    expect(root.textContent).toContain("Margin held [s-rich-e1, s-outside (outside freeze)]");
    expect(root.textContent).toContain("s-outside (outside freeze)");
    expect(root.textContent).toContain("SEC [filed] 10-K FY2026 [s-rich-e1]");
    expect(root.textContent).toContain("WEB Blog post [s-outside]");
    expect([...root.querySelectorAll("h3")].map((heading) => heading.textContent)).toContain("Sources");
    expect(root.textContent).toContain('Coverage: {"ratio":0.5} — allowed sources: SEC, WEB');
    expect(root.textContent).toContain("freeze: s-rich-f1");
    expect(root.textContent).toContain("2 journal events, 1 jobs, 1 coverage artifacts");
    expect(root.textContent).toContain("artifact-1 wave 1 search search-1 query margin: Margin held");
    expect([...root.querySelectorAll("details ol li")].map((entry) => entry.textContent)).toEqual([
      "[1] session_opened {\"sessionId\":\"s-rich\"}",
      "[2] freeze {\"freezeId\":\"s-rich-f1\"}",
    ]);
  });

  test("states an empty final result and hides a summary that repeats the answer", async () => {
    const bare = run("s-bare");
    bare.finalResult = {
      ...bare.finalResult as NonNullable<ResearchRun["finalResult"]>,
      answer: "Only the answer.",
      executiveSummary: "Only the answer.",
      claims: [],
      sources: [],
      allowedSources: [],
      freezeId: "",
    };
    const root = await mount(<HarnessViewer projection={projectionWith({ researchRuns: [bare] })} sessionId="s-bare" />);
    expect(root.textContent).toContain("Only the answer.");
    expect(root.textContent).not.toContain("Bottom line:");
    expect(root.textContent).toContain("No validated claims.");
    expect(root.textContent).toContain("allowed sources: unknown");
    expect(root.textContent).toContain("freeze: unknown");
    expect(root.textContent).not.toContain(" (outside freeze)");
  });

  test("summarizes unnamed sources and a completed run without progress notes", async () => {
    const unnamed = run("s-unnamed", "completed");
    unnamed.evidence = [{ ...unnamed.evidence[0], sourceName: "" }];
    const root = await mount(<HarnessViewer projection={projectionWith({ researchRuns: [unnamed] })} sessionId="s-unnamed" />);
    expect(root.textContent).toContain("Evidence: 1 records — sources: 0 (none)");
    expect(root.textContent).toContain("Evidence by desk: SEC 1, FINRA 0, WEB 0");
    expect(root.textContent).not.toContain("Run in progress");
  });

  test("derives waves from jobs and names every wave without jobs", async () => {
    const job = run("s-waves").jobs[0];
    const derived = { ...run("s-waves"), waves: [], jobs: [{ ...job, waveId: 2 }, { ...job, jobId: "s-waves-j2", waveId: 1 }] };
    const root = await mount(<HarnessViewer projection={projectionWith({ researchRuns: [derived] })} sessionId="s-waves" />);
    expect(root.textContent).toContain("waves [1, 2]");
    expect(root.textContent).toContain("s-waves-j1");
    expect(root.textContent).toContain("s-waves-j2");
    await act(async () => mounted?.unmount());

    const gappedJob = run("s-gaps").jobs[0];
    const gapped = { ...run("s-gaps"), waves: [1, 2], jobs: [{ ...gappedJob, waveId: 2 }] };
    const other = await mount(<HarnessViewer projection={projectionWith({ researchRuns: [gapped] })} sessionId="s-gaps" />);
    expect(other.textContent).toContain("waves [1, 2]");
    expect(other.textContent).toContain("No jobs persisted for wave 1.");
    expect(other.textContent).toContain("s-gaps-j1");
  });

  test("session viewer shows loading, defaults omitted snapshot fields, and rejects a mismatched snapshot", async () => {
    const savedFetch = globalThis.fetch;
    const calls: string[] = [];
    const gate = Promise.withResolvers<void>();
    const snapshots: Record<string, unknown> = {
      "/api/research/s-load": { session: { session_id: "s-load" }, jobs: [{}, { job_id: "named-job" }] },
      "/api/research/s-nolist": { session: { session_id: "s-nolist" } },
      "/api/research/s-other": { session: { session_id: "a different session" } },
    };
    globalThis.fetch = ((url: unknown) => {
      const path = String(url);
      calls.push(path);
      return gate.promise.then(() => new Response(JSON.stringify(snapshots[path] ?? {}), { status: 200, headers: { "Content-Type": "application/json" } }));
    }) as typeof globalThis.fetch;
    try {
      const root = await mount(<HarnessSessionViewer sessionId="s-load" />);
      expect(root.textContent).toContain("Loading session (research://s-load).");
      await act(async () => gate.resolve());
      await settle(() => (root.textContent ?? "").includes("Question: s-load"));
      expect(root.textContent).toContain("Status: unknown — wave 1 — waves [1]");
      expect(root.textContent).toContain("Provider: unknown model: unknown");
      expect(root.textContent).toContain("Run in progress [unknown]; jobs and events below are partial.");
      expect(root.textContent).toContain("s-load-job-0");
      expect(root.textContent).toContain("named-job");
      await act(async () => mounted?.unmount());

      const mismatched = await mount(<HarnessSessionViewer sessionId="s-other" />);
      await settle(() => (mismatched.textContent ?? "").includes("Snapshot unavailable"));
      expect(mismatched.textContent).toContain("Snapshot unavailable for this session (research://s-other).");
      expect(mismatched.textContent).not.toContain("a different session");
      await act(async () => mounted?.unmount());

      const bare = await mount(<HarnessSessionViewer sessionId="s-nolist" />);
      await settle(() => (bare.textContent ?? "").includes("Question: s-nolist"));
      expect(bare.textContent).toContain("No jobs persisted for wave 1.");
      expect(bare.textContent).toContain("Empty evidence for this run.");
      expect(calls).toEqual(["/api/research/s-load", "/api/research/s-other", "/api/research/s-nolist"]);
    } finally {
      globalThis.fetch = savedFetch;
    }
  });

  test("session viewer mounts unselected without requesting a snapshot", async () => {
    const savedFetch = globalThis.fetch;
    const calls: string[] = [];
    globalThis.fetch = ((url: unknown) => {
      calls.push(String(url));
      return Promise.reject(new Error("unexpected snapshot request"));
    }) as typeof globalThis.fetch;
    try {
      const root = await mount(<HarnessSessionViewer sessionId={null} />);
      await act(async () => { });
      expect(root.textContent).toContain("No trace persisted for this session (research://unselected).");
      expect(calls).toEqual([]);
    } finally {
      globalThis.fetch = savedFetch;
    }
  });
});
