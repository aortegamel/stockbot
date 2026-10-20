import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { AgentConsole } from "./agent-console";
import type { AgentEvent, Evidence, FinalClaim, Metrics } from "../../../needle-harness/lib/agent/types";

function metrics(): Metrics {
  return {
    totalMs: 1,
    needle: { calls: 0, totalMs: 0, escalations: 0 },
    tools: { calls: 0, totalMs: 0 },
    muse: { calls: 1, totalMs: 1 },
    evidence: { count: 0, characters: 0 },
    failures: {},
  };
}

function evidence(id: string, sources?: Evidence["sources"]): Evidence {
  return { id, source: "sec", retrievedAt: "2026-10-10T00:00:00Z", content: "", sources };
}

const SEC_SOURCE = { id: "capture:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", title: "10-K" };

describe("AgentConsole source citations", () => {
  test("known ids in answer text become citation buttons in prod", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      { type: "evidence", evidence: [evidence("ev:abc123", [SEC_SOURCE])] },
      { type: "answer_delta", text: "Risk grew [ev:abc123] this year." },
      { type: "done", metrics: metrics() },
    ];
    const out = renderToStaticMarkup(<AgentConsole events={events} reasoning={true} debug={false} />);
    expect(out).toContain('aria-label="Open source for ev:abc123"');
    expect(out).toContain("[ev:abc123]");
  });

  test("unknown markers stay plain text", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      { type: "evidence", evidence: [evidence("ev:abc123", [SEC_SOURCE])] },
      { type: "answer_delta", text: "See [ev:unknown] too." },
      { type: "done", metrics: metrics() },
    ];
    const out = renderToStaticMarkup(<AgentConsole events={events} reasoning={true} debug={false} />);
    expect(out).toContain("See [ev:unknown] too.");
    expect(out).not.toContain("Open source for ev:unknown");
  });

  test("known ids without sources still provide source controls", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      { type: "evidence", evidence: [evidence("ev:abc123", [SEC_SOURCE]), evidence("ev:abc124")] },
      { type: "answer_delta", text: "Sourced first [ev:abc123], bare second [ev:abc124]." },
      { type: "done", metrics: metrics() },
    ];
    const out = renderToStaticMarkup(<AgentConsole events={events} reasoning={true} debug={false} />);
    expect(out).toContain("Sources: ev:abc123");
    expect(out).not.toContain("Sources: ev:abc124");
    expect(out).toContain("[ev:abc124]");
    expect(out).not.toContain("Open source for ev:abc124");
  });

  test("evidence event never renders as a prod row", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      { type: "evidence", evidence: [evidence("ev:abc123", [SEC_SOURCE])] },
      { type: "answer_delta", text: "Done [ev:abc123]." },
      { type: "done", metrics: metrics() },
    ];
    const out = renderToStaticMarkup(<AgentConsole events={events} reasoning={true} debug={false} />);
    expect(out).not.toContain("ev:abc123&quot;,&quot;retrievedAt");
  });

  test("last evidence event wins", () => {
    const stale: AgentEvent = { type: "evidence", evidence: [evidence("ev:abc123", [SEC_SOURCE])] };
    const current: AgentEvent = { type: "evidence", evidence: [evidence("ev:abc124", [SEC_SOURCE])] };
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      stale,
      current,
      { type: "answer_delta", text: "Now [ev:abc124], stale [ev:abc123]." },
      { type: "done", metrics: metrics() },
    ];
    const out = renderToStaticMarkup(<AgentConsole events={events} reasoning={true} debug={false} />);
    expect(out).toContain("Open source for ev:abc124");
    expect(out).not.toContain("Open source for ev:abc123");
  });

  // A gated research answer: prose, then the claim inventory, then done.
  function claimed(text: string, claims: FinalClaim[]): string {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "risk?" },
      { type: "evidence", evidence: [evidence("ev:abc123", [SEC_SOURCE]), evidence("ev:abc124", [SEC_SOURCE])] },
      { type: "answer_delta", text },
      { type: "claims", claims },
      { type: "done", metrics: metrics() },
    ];
    return renderToStaticMarkup(<AgentConsole events={events} reasoning={true} debug={false} />);
  }
  const chip = (n: number, id: string): string => `aria-label="Open source for ${id}" class="rounded border border-blue-800 px-1 text-blue-300 underline decoration-dotted hover:bg-blue-950">[${n}]</button>`;

  test("a claim with evidence ids gets a numbered chip after its first exact occurrence", () => {
    const out = claimed("Risk grew this year. Risk grew again.", [{ text: "Risk grew", claimType: "observed_fact", evidenceIds: ["ev:abc123"] }]);
    expect(out).toContain(`<span>Risk grew</span><button type="button" ${chip(1, "ev:abc123")}<span> this year. Risk grew again.</span>`);
    expect(out.match(/\[1\]<\/button>/g)).toHaveLength(1);
  });

  test("unmatched claim text, empty evidence ids, and unknown ids add no chip", () => {
    for (const claim of [
      { text: "Not in the answer", claimType: "observed_fact", evidenceIds: ["ev:abc123"] },
      { text: "Risk grew", claimType: "unknown", evidenceIds: [] },
      { text: "Risk grew", claimType: "observed_fact", evidenceIds: ["ev:missing"] },
    ] satisfies FinalClaim[]) {
      const out = claimed("Risk grew this year.", [claim]);
      expect(out).toContain("Risk grew this year.");
      expect(out).not.toContain("Open source for");
    }
  });

  test("a sentence with an inline marker for the same id gets no second chip; numbering is shared", () => {
    const out = claimed("Risk grew [ev:abc123]. Revenue fell.", [
      { text: "Risk grew", claimType: "observed_fact", evidenceIds: ["ev:abc123"] },
      { text: "Revenue fell", claimType: "observed_fact", evidenceIds: ["ev:abc124"] },
    ]);
    expect(out.match(/Open source for ev:abc123/g)).toHaveLength(1);
    expect(out).toContain(">[ev:abc123]</button>");
    expect(out).toContain(`<span>. Revenue fell</span><button type="button" ${chip(2, "ev:abc124")}<span>.</span>`);
  });
});
