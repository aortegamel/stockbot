import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { AgentConsole } from "./agent-console";
import type { AgentEvent, Evidence, Metrics } from "@/lib/agent/types";

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
});
