import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { AgentConsole } from "./agent-console";
import type { AgentEvent, Metrics } from "../../../needle-harness/lib/agent/types";

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

function html(events: AgentEvent[], reasoning = true, debug = false): string {
  return renderToStaticMarkup(<AgentConsole events={events} reasoning={reasoning} debug={debug} />);
}

describe("AgentConsole answer rendering", () => {
  test("regression: answer_delta chunks join into one rendered answer", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "hello" },
      { type: "reasoning_start", model: "stockbot" },
      { type: "answer_delta", text: "Hello" },
      { type: "answer_delta", text: " world" },
      { type: "done", metrics: metrics() },
    ];
    const out = html(events);
    expect(out).toContain("Hello world");
  });

  test("prod mode renders the answer even though deltas are hidden work rows", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "hello" },
      { type: "reasoning_start", model: "stockbot" },
      { type: "answer_delta", text: "hi there" },
      { type: "done", metrics: metrics() },
    ];
    const out = html(events, true, false);
    expect(out).toContain("hi there");
  });

  test("streaming answer renders before done arrives", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "hello" },
      { type: "reasoning_start", model: "stockbot" },
      { type: "answer_delta", text: "partial" },
    ];
    const out = html(events);
    expect(out).toContain("partial");
  });

  test("empty events render the intro", () => {
    const out = html([]);
    expect(out).toContain("Stockbot");
  });

  test("terminal error renders its message", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "hello" },
      { type: "error", message: "boom happened" },
    ];
    expect(html(events)).toContain("boom happened");
  });
});
