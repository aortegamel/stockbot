import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { JSDOM } from "jsdom";
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

function rows(events: AgentEvent[], reasoning = true, debug = true): string[] {
  const dom = new JSDOM(html(events, reasoning, debug));
  try {
    return [...dom.window.document.body.firstElementChild!.children].map((row) => row.textContent ?? "");
  } finally {
    dom.window.close();
  }
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

describe("AgentConsole event presentation", () => {
  test("debug decisions show rounded confidence, unknown confidence, and escalation without arguments", () => {
    const events: AgentEvent[] = [
      { type: "needle_decision", step: 1, tool: "filings", arguments: { ticker: "NVDA" }, confidence: 0.876 },
      { type: "needle_decision", step: 2, tool: "ownership", arguments: {}, confidence: null },
      { type: "needle_decision", step: 3, tool: null, arguments: { query: "hidden escalation arguments" }, confidence: 0.99 },
    ];
    const rendered = rows(events, false);
    expect(rendered).toHaveLength(3);
    expect(rendered[0]).toContain("NEEDLE/LOCAL");
    expect(rendered[0]).toContain("needle / filings / confidence 88%");
    expect(rendered[0]).toContain('filings({"ticker":"NVDA"})');
    expect(rendered[1]).toContain("needle / ownership / confidence n/a");
    expect(rendered[1]).toContain("ownership({})");
    expect(rendered[2]).toContain("needle / (escalate)");
    expect(rendered[2]).not.toContain("confidence");
    expect(rendered[2]).not.toContain("hidden escalation arguments");
  });

  test("debug arguments redact nested credentials without hiding research parameters", () => {
    const events: AgentEvent[] = [{
      type: "needle_decision",
      step: 1,
      tool: "filings",
      confidence: 1,
      arguments: {
        ticker: "NVDA",
        api_key: "private-api-value",
        nested: { Authorization: "private-auth-value", period: "2025" },
        accounts: [{ "account-number": "private-account-value", limit: 3 }],
      },
    }];
    const text = rows(events, false).join("");
    expect(text).toContain('"ticker":"NVDA"');
    expect(text).toContain('"api_key":"[REDACTED]"');
    expect(text).toContain('"Authorization":"[REDACTED]"');
    expect(text).toContain('"account-number":"[REDACTED]"');
    expect(text).toContain('"period":"2025"');
    expect(text).toContain('"limit":3');
    for (const secret of ["private-api-value", "private-auth-value", "private-account-value"]) {
      expect(text).not.toContain(secret);
    }
  });

  test("debug tool rows distinguish successful evidence, missing evidence, and categorized failure", () => {
    const events: AgentEvent[] = [
      { type: "tool_start", tool: "filings" },
      { type: "tool_result", tool: "filings", evidenceId: "ev-7", preview: "Annual report captured" },
      { type: "tool_result", tool: "ownership", preview: "No ownership evidence" },
      { type: "tool_failed", tool: "web", category: "timeout", preview: "Search deadline expired" },
    ];
    expect(rows(events, false)).toEqual([
      "├─ filings…",
      "├─ filings / ✓ ev-7 — Annual report captured",
      "├─ ownership / ✗ — No ownership evidence",
      "├─ web / ✗ timeout — Search deadline expired",
    ]);
    expect(rows(events, false, false)).toEqual([]);
  });

  test("debug progress shows optional detail and limits long detail to 160 characters", () => {
    const events: AgentEvent[] = [
      { type: "progress", stage: "archive started" },
      { type: "progress", stage: "archive complete", detail: { count: 2, provider: "SEC" } },
      { type: "progress", stage: "long preview", detail: { text: "x".repeat(300), secretTail: "must not appear" } },
    ];
    const rendered = rows(events, false);
    expect(rendered[0]).toBe("├─ archive started");
    expect(rendered[1]).toBe('├─ archive complete {"count":2,"provider":"SEC"}');
    expect(rendered[2]).toBe(`├─ long preview {"text":"${"x".repeat(151)}`);
    expect(rendered[2].slice("├─ long preview ".length)).toHaveLength(160);
    expect(rendered[2]).not.toContain("must not appear");
    expect(rows(events, false, false)).toEqual([]);
  });

  test("reasoning appears once while streaming and disappears when reasoning stops or a terminal row arrives", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "Research NVDA" },
      { type: "reasoning_start", model: "private-model-name" },
      { type: "answer_delta", text: "Partial " },
      { type: "answer_delta", text: "answer" },
      { type: "evidence", evidence: [] },
    ];
    for (const debug of [false, true]) {
      const streaming = rows(events, true, debug);
      expect(streaming).toHaveLength(3);
      expect(streaming[0]).toBe("> Research NVDA");
      expect(streaming[1]).toContain("STOCKBOT");
      expect(streaming[1]).toContain("stockbot / working…");
      expect(streaming[2]).toBe("Partial answer");
      expect(streaming.join("")).not.toContain("private-model-name");
      expect(rows(events, false, debug)).toEqual(["> Research NVDA", "Partial answer"]);
      for (const terminal of [
        { type: "done", metrics: metrics() },
        { type: "error", message: "Transport stopped" },
        { type: "failed", category: "policy_denied", message: "Research denied" },
      ] satisfies AgentEvent[]) {
        const finished = rows([...events, terminal], true, debug);
        expect(finished).toHaveLength(3);
        expect(finished[1]).toBe("Partial answer");
        expect(finished.join("")).not.toContain("working…");
      }
    }
  });

  test("terminal metrics show counts, rounded duration, and optional token counts", () => {
    const measured: Metrics = {
      totalMs: 123.6,
      needle: { calls: 2, totalMs: 12.6, escalations: 1 },
      tools: { calls: 3, totalMs: 33.2 },
      muse: { calls: 4, totalMs: 77.8, inputTokens: 42, outputTokens: 0 },
      evidence: { count: 5, characters: 600 },
      failures: {},
    };
    for (const debug of [false, true]) {
      const provided = rows([{ type: "done", metrics: measured }], true, debug)[0];
      expect(provided).toContain("── metrics ──");
      expect(provided).toContain("Needle calls: 2 · Tool calls: 3 · Stockbot calls: 4 · Evidence: 5 · Evidence chars: 600");
      expect(provided).toContain("Stockbot input: 42 · Stockbot output: 0");
      expect(provided).toContain("Needle: 13ms · Tools: 33ms · Stockbot: 78ms · Total: 124ms");
      const omitted = rows([{ type: "done", metrics: metrics() }], true, debug)[0];
      expect(omitted).toContain("Stockbot input: n/a · Stockbot output: n/a");
    }
  });

  test("work keeps its order before the aggregated answer and terminals keep their order after it", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "Research request" },
      { type: "answer_delta", text: "First " },
      { type: "needle_decision", step: 1, tool: "filings", arguments: {}, confidence: 0.5 },
      { type: "tool_start", tool: "filings" },
      { type: "done", metrics: metrics() },
      { type: "tool_result", tool: "filings", evidenceId: "ev-1", preview: "Report found" },
      { type: "answer_delta", text: "second." },
      { type: "error", message: "Connection closed" },
      { type: "progress", stage: "archive saved" },
      { type: "failed", category: "tool_error", message: "Provider failed" },
      { type: "evidence", evidence: [] },
    ];
    const debug = rows(events);
    expect(debug).toHaveLength(9);
    expect(debug[0]).toBe("> Research request");
    expect(debug[1]).toContain("needle / filings / confidence 50%");
    expect(debug[2]).toBe("├─ filings…");
    expect(debug[3]).toBe("├─ filings / ✓ ev-1 — Report found");
    expect(debug[4]).toBe("├─ archive saved");
    expect(debug[5]).toBe("First second.");
    expect(debug[6]).toContain("── metrics ──");
    expect(debug[7]).toBe("✗ Connection closed");
    expect(debug[8]).toBe("✗ tool_error: Provider failed");
    const production = rows(events, true, false);
    expect(production).toHaveLength(5);
    expect(production[0]).toBe("> Research request");
    expect(production[1]).toBe("First second.");
    expect(production[2]).toContain("── metrics ──");
    expect(production.slice(3)).toEqual(["✗ Connection closed", "✗ tool_error: Provider failed"]);
    expect(production.join("")).not.toContain("needle /");
    expect(production.join("")).not.toContain("Report found");
    expect(production.join("")).not.toContain("archive saved");
  });
});
