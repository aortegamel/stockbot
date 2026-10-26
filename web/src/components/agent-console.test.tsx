// This import must stay first: it installs the jsdom globals (see test-dom.ts).
import { dom } from "../test-dom";
import { describe, expect, test } from "bun:test";
import { act } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { JSDOM } from "jsdom";
import { AgentConsole } from "./agent-console";
import { SEEN_ALERTS_KEY, toastAlert } from "./toasts";
import type { FeedItem } from "../lib/market-data";
import type { AgentEvent, Metrics } from "../../../needle-harness/lib/agent/types";

// react-dom reads canUseDOM once at module load, so it must load after the jsdom globals exist.
const { createRoot } = await import("react-dom/client");

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

  test("checked answer text renders before done arrives", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "hello" },
      { type: "reasoning_start", model: "stockbot" },
      { type: "answer_delta", text: "partial" },
    ];
    const out = html(events);
    expect(out).toContain("partial");
    expect(out).toContain("Partial checked answer · research continues");
  });

  test("the latest checked snapshot persists after failure and final deltas replace it", () => {
    const events: AgentEvent[] = [
      { type: "reasoning_start", model: "stockbot" },
      { type: "answer_partial", text: "Earlier checked snapshot.", claims: [] },
      { type: "answer_partial", text: "Latest checked snapshot.", claims: [] },
      { type: "answer_delta", text: "" },
    ];
    for (const debug of [false, true]) {
      const partial = html(events, true, debug);
      expect(partial).toContain("Latest checked snapshot.");
      expect(partial).not.toContain("Earlier checked snapshot.");
      expect(partial).toContain("Partial checked answer · research continues");
      for (const terminal of [
        { type: "failed", category: "provider_error", message: "private provider diagnostic" },
        { type: "error", message: "private transport diagnostic" },
        { type: "done", metrics: metrics(), status: "cancelled" },
      ] satisfies AgentEvent[]) {
        const stopped = html([...events, terminal], true, debug);
        expect(stopped).toContain("Latest checked snapshot.");
        expect(stopped).not.toContain("research continues");
      }
      const final = html([...events, { type: "answer_delta", text: "Final checked " }, { type: "answer_delta", text: "answer." }], true, debug);
      expect(final).toContain("Final checked answer.");
      expect(final).not.toContain("Latest checked snapshot.");
      expect(final).not.toContain("Earlier checked snapshot.");
    }
  });

  test("empty events render the intro", () => {
    const out = html([]);
    expect(out).toContain("Stockbot");
  });

  test("terminal errors show safe production text and preserve debug diagnostics", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "hello" },
      { type: "error", message: "boom happened", code: "provider_unavailable", stage: "answer" },
    ];
    const production = html(events);
    expect(production).toContain("The request failed. Please retry.");
    for (const diagnostic of ["boom happened", "provider_unavailable", "answer:"]) expect(production).not.toContain(diagnostic);
    expect(html(events, true, true)).toContain("answer: provider_unavailable: boom happened");
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
    const debugRows = rows(events, false);
    expect(debugRows.slice(0, 4)).toEqual([
      "├─ filings…",
      "├─ filings / ✓ ev-7 — Annual report captured",
      "├─ ownership / ✗ — No ownership evidence",
      "├─ web / ✗ timeout — Search deadline expired",
    ]);
    // The show-work row follows: debug keeps the raw tool name, prod shows only the plain line.
    expect(debugRows[4]).toBe("Working · 1 step · 0sfilings");
    expect(rows(events, false, false)).toEqual(["Working · 1 step · 0sUsed a research tool"]);
  });

  test("debug progress shows stage names without raw detail blobs", () => {
    const events: AgentEvent[] = [
      { type: "progress", stage: "archive started" },
      { type: "progress", stage: "archive complete", detail: { count: 2, provider: "SEC" } },
      { type: "progress", stage: "long preview", detail: { text: "x".repeat(300), secretTail: "must not appear" } },
    ];
    const rendered = rows(events, false);
    expect(rendered.slice(0, 3)).toEqual(["├─ archive started", "├─ archive complete", "├─ long preview"]);
    for (const row of rendered.slice(0, 3)) expect(row).not.toContain("must not appear");
    const production = rows(events, false, false);
    expect(production).toHaveLength(1);
    expect(production[0]).toStartWith("Working · 3 steps");
    expect(production[0]).not.toContain("archive");
  });

  test("reasoning gives way to a checked answer before the terminal row arrives", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "Research NVDA" },
      { type: "reasoning_start", model: "private-model-name" },
      { type: "answer_delta", text: "Partial " },
      { type: "answer_delta", text: "answer" },
      { type: "evidence", evidence: [] },
    ];
    for (const debug of [false, true]) {
      const working = rows(events.slice(0, 2), true, debug);
      expect(working).toHaveLength(2);
      expect(working[1]).toContain("STOCKBOT");
      expect(working[1]).toContain("stockbot / working…");
      const streaming = rows(events, true, debug);
      expect(streaming).toHaveLength(3);
      expect(streaming[0]).toBe("> Research NVDA");
      expect(streaming[1]).toBe("Partial checked answer · research continues");
      expect(streaming[2]).toBe("Partial answer");
      expect(streaming.join("")).not.toContain("working…");
      expect(streaming.join("")).not.toContain("private-model-name");
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

  test("terminal metrics show logical and provider calls, stage timings, and optional token counts", () => {
    const measured: Metrics = {
      totalMs: 123.6,
      needle: { calls: 2, totalMs: 12.6, escalations: 1 },
      tools: { calls: 3, totalMs: 33.2 },
      muse: { calls: 4, providerCalls: 8, totalMs: 77.8, inputTokens: 42, outputTokens: 0 },
      evidence: { count: 5, characters: 600 },
      failures: {},
      timings: { routeMs: 10.4, nodesMs: 300, narrativeMs: 40.6, binderMs: 12.2 },
    };
    for (const debug of [false, true]) {
      const provided = rows([{ type: "done", metrics: measured }], true, debug)[0];
      expect(provided).toContain("── metrics ──");
      expect(provided).toContain("Needle calls: 2 · Tool calls: 3 · Stockbot calls: 4 · Provider calls: 8 · Evidence: 5 · Evidence chars: 600");
      expect(provided).toContain("Stockbot input: 42 · Stockbot output: 0");
      expect(provided).toContain("Needle: 13ms · Tools: 33ms · Stockbot: 78ms · Total: 124ms");
      // Phase timings render only in debug; an absent phase never renders.
      expect(provided.includes("Timings: routeMs 10ms · nodesMs 300ms · narrativeMs 41ms · binderMs 12ms")).toBe(debug);
      const omitted = rows([{ type: "done", metrics: metrics() }], true, debug)[0];
      expect(omitted).toContain("Stockbot input: n/a · Stockbot output: n/a");
      expect(omitted).toContain("Stockbot calls: 1 · Evidence: 0");
      expect(omitted).not.toContain("Provider calls:");
      expect(omitted).not.toContain("Timings:");
      const zero = rows([{ type: "done", metrics: { ...metrics(), muse: { calls: 0, providerCalls: 0, totalMs: 0 } } }], true, debug)[0];
      expect(zero).toContain("Stockbot calls: 0 · Provider calls: 0");
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
    expect(debug).toHaveLength(10);
    expect(debug[0]).toBe("> Research request");
    expect(debug[1]).toContain("needle / filings / confidence 50%");
    expect(debug[2]).toBe("├─ filings…");
    expect(debug[3]).toBe("├─ filings / ✓ ev-1 — Report found");
    expect(debug[4]).toBe("├─ archive saved");
    expect(debug[5]).toBe("First second.");
    expect(debug[6]).toBe("Worked for 0s · 2 stepsfilingsarchive saved");
    const production = rows(events, true, false);
    expect(production).toHaveLength(6);
    expect(production[0]).toBe("> Research request");
    expect(production[1]).toBe("First second.");
    expect(production[2]).toBe("Worked for 0s · 2 stepsUsed a research toolWorked on the research");
    expect(production.join("")).not.toContain("needle /");
    expect(production.join("")).not.toContain("filings");
    expect(production.join("")).not.toContain("Report found");
    expect(production.join("")).not.toContain("archive saved");
  });
});

describe("AgentConsole trio offer", () => {
  const offered: AgentEvent[] = [
    { type: "agent_start", prompt: "NVDA risks?" },
    { type: "answer_delta", text: "Stockbot view." },
    { type: "offer_trio", sessionId: "rs:1" },
    { type: "done", metrics: metrics() },
  ];

  test("a finished stockbot answer asks for the trio with Yes and No in prod and debug", () => {
    for (const debug of [false, true]) {
      const out = rows(offered, true, debug);
      expect(out.at(-1)).toBe("Do you want the trio team (bull + bear views)? Yes No");
      // The offer event itself never renders as a work row.
      expect(out.join("")).not.toContain("rs:1");
    }
  });

  test("no offer before the answer is final or without an offer event", () => {
    expect(html(offered.slice(0, 3))).not.toContain("trio team");
    expect(html(offered.filter((e) => e.type !== "offer_trio"))).not.toContain("trio team");
  });
});

describe("AgentConsole status lines", () => {
  const working: AgentEvent[] = [
    { type: "agent_start", prompt: "NVDA insider sales?" },
    { type: "status", text: "Looking up Form 4 insider filings…" },
    { type: "tool_start", tool: "get_insider_activity" },
    { type: "status", text: "Reading <b>Form 4</b> filings for NVDA" },
  ];
  const answered: AgentEvent[] = [...working, { type: "answer_delta", text: "Insiders sold." }, { type: "done", metrics: metrics() }];

  function statusView(events: AgentEvent[], debug = false): { live: string[]; details: Element | null; text: string } {
    const page = new JSDOM(html(events, true, debug));
    try {
      const doc = page.window.document;
      const live = [...(doc.querySelector('[role="status"]')?.children ?? [])].map((line) => line.textContent ?? "");
      const details = [...doc.querySelectorAll("details")].find((entry) => entry.textContent?.includes("Progress ·") ?? false) ?? null;
      return { live, details, text: doc.body.textContent ?? "" };
    } finally {
      page.window.close();
    }
  }

  test("prod shows live status lines while the run works", () => {
    const view = statusView(working);
    expect(view.live).toEqual(["Looking up Form 4 insider filings…", "Reading <b>Form 4</b> filings for NVDA"]);
    expect(view.details).toBeNull();
    expect(view.text).toContain("Working · 1 step");
    expect(view.text).not.toContain("├─");
    expect(view.text).not.toContain("get_insider_activity");
  });

  test("prod keeps the answer, work summary, and collapsed progress after the run ends", () => {
    const view = statusView(answered);
    expect(view.live).toEqual([]);
    expect(view.details?.querySelector("summary")?.textContent).toBe("Progress · 2 steps");
    expect([...(view.details?.children ?? [])].slice(1).map((line) => line.textContent)).toEqual(["Looking up Form 4 insider filings…", "Reading <b>Form 4</b> filings for NVDA"]);
    expect(view.text).toContain("Insiders sold.");
    expect(view.text).toContain("Worked for 0s · 1 step");
    expect(view.text).not.toContain("├─");
  });

  test("debug keeps the raw work rows and shows each status line once", () => {
    const view = statusView(working, true);
    expect(view.live).toEqual(["Looking up Form 4 insider filings…", "Reading <b>Form 4</b> filings for NVDA"]);
    expect(view.details).toBeNull();
    expect(view.text).toContain("├─ get_insider_activity…");
    expect(view.text.split("Looking up Form 4 insider filings…").length).toBe(2);
  });

  test("debug keeps collapsed progress after the answer arrives", () => {
    const view = statusView(answered, true);
    expect(view.live).toEqual([]);
    const details = view.details as HTMLDetailsElement | null;
    expect(details?.open).toBe(false);
    expect(details?.querySelector("summary")?.textContent).toBe("Progress · 2 steps");
    expect(view.text).toContain("├─ get_insider_activity…");
  });
});

describe("AgentConsole show-work row", () => {
  const work: AgentEvent[] = [
    { type: "agent_start", prompt: "NVDA risks?" },
    { type: "progress", stage: "intake_start", detail: { calls: 2 } },
    { type: "needle_decision", step: 1, tool: "get_sec_document", arguments: {}, confidence: 0.9 },
    { type: "tool_start", tool: "get_sec_document" },
    { type: "tool_result", tool: "get_sec_document", evidenceId: "ev-1", preview: "Risk factors" },
    { type: "tool_start", tool: "search_web" },
  ];

  function row(events: AgentEvent[], debug = false): { summary: string; steps: string[]; text: string } {
    const page = new JSDOM(html(events, false, debug));
    try {
      const details = page.window.document.querySelector("details");
      return {
        summary: details?.querySelector("summary")?.textContent ?? "",
        steps: [...(details?.querySelectorAll("li") ?? [])].map((li) => li.textContent ?? ""),
        text: page.window.document.body.textContent ?? "",
      };
    } finally {
      page.window.close();
    }
  }

  test("the live line counts tool and progress steps", () => {
    expect(row(work).summary).toMatch(/^Working · 3 steps · \d+s$/);
    expect(row(work.slice(0, 2)).summary).toMatch(/^Working · 1 step · \d+s$/);
    expect(row(work.slice(0, 1)).summary).toBe("");
  });

  test("the collapsible lists plain phrases in prod and raw names only in debug", () => {
    const prod = row(work);
    expect(prod.steps).toEqual(["Started gathering sources", "Read SEC filing", "Searched the web"]);
    for (const raw of ["get_sec_document", "search_web", "intake_start"]) expect(prod.text).not.toContain(raw);
    expect(row(work, true).steps).toEqual(["intake_start", "get_sec_document", "search_web"]);
  });

  test("the done line reads the run time from the done metrics", () => {
    const done: AgentEvent = { type: "done", metrics: { ...metrics(), totalMs: 12_400, timings: { intakeMs: 3_000, nodesMs: 8_000 } } };
    expect(row([...work, done]).summary).toBe("Worked for 12s · 3 steps");
  });

  test("checked answers render before the terminal event and work continues", () => {
    const answered: AgentEvent[] = [...work, { type: "progress", stage: "intake_evidence" }, { type: "answer_delta", text: "Final view." }, { type: "done", metrics: metrics() }];
    const out = rows(answered, false, false);
    expect(out.slice(-3)).toEqual(["Final view.", expect.stringMatching(/^Worked for \d+s · 4 steps/), expect.stringMatching(/── metrics ──/)]);
    expect(html(work, false)).not.toContain("Final view.");
    const active = html([...work, { type: "answer_delta", text: "Final view." }], false);
    expect(active).toContain("Final view.");
    expect(active).toContain("Partial checked answer · research continues");
    expect(active).toContain("Working · 3 steps");
  });
});

describe("AgentConsole result cards", () => {
  test("chart, watchlist, and saved-thesis events render cards with the workspace link", () => {
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "show NVDA" },
      { type: "chart_update", ticker: "NVDA" },
      { type: "watchlist_update", action: "remove", ticker: "AMD" },
      { type: "tool_result", tool: "thesis_create", evidenceId: "ev-9", preview: "created" },
      { type: "tool_result", tool: "thesis_create", preview: "failed" },
      { type: "answer_delta", text: "Done." },
      { type: "done", metrics: metrics() },
    ];
    const page = new JSDOM(html(events));
    try {
      const cards = [...page.window.document.querySelectorAll('[aria-label="Results"] > div')];
      expect(cards.map((card) => card.textContent)).toEqual(["Chart set to NVDAOpen workspace", "Removed AMD from the watchlistOpen workspace", "Thesis saved"]);
      expect(cards.map((card) => card.querySelector("a")?.getAttribute("href"))).toEqual(["#/workspace/NVDA", "#/workspace/AMD", undefined]);
    } finally {
      page.window.close();
    }
    expect(html(events.filter((e) => e.type !== "chart_update" && e.type !== "watchlist_update" && e.type !== "tool_result"))).not.toContain('aria-label="Results"');
  });

  test("a thesis_watch add renders an alert card whose button asks for notifications once", async () => {
    const added = "thesis_id: thesis:1\nadded: rule_id: rule:1, rule_type: material_short_interest_change, enabled: True";
    const events: AgentEvent[] = [
      { type: "agent_start", prompt: "alert me when NVDA short interest changes" },
      { type: "tool_result", tool: "thesis_watch", evidenceId: "ev-1", preview: added },
      { type: "tool_result", tool: "thesis_watch", evidenceId: "ev-2", preview: "thesis_id: thesis:1\nrules: rule_type: new_filing" },
      { type: "tool_result", tool: "thesis_watch", preview: added },
    ];
    const registry = globalThis as Record<string, unknown>;
    const asked: string[] = [];
    registry.Notification = { permission: "default", requestPermission: async () => { asked.push("ask"); return "granted"; } };
    const container = dom.window.document.getElementById("root");
    if (!container) throw new Error("missing test root");
    const root = createRoot(container);
    try {
      await act(async () => root.render(<AgentConsole events={events} reasoning={false} debug={false} />));
      const cards = () => [...container.querySelectorAll('[aria-label="Results"] > div')].map((card) => card.textContent);
      expect(cards()).toEqual(["Alert set: material short interest changeEnable notifications"]);
      const enable = [...container.querySelectorAll("button")].find((b) => b.textContent === "Enable notifications");
      await act(async () => enable?.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })));
      expect(asked).toEqual(["ask"]);
      expect(cards()).toEqual(["Alert set: material short interest change"]);
      delete registry.Notification;
      expect(html(events)).not.toContain("Enable notifications");
    } finally {
      delete registry.Notification;
      await act(async () => root.unmount());
    }
  });
});

describe("AgentConsole toasts", () => {
  test("an error event fires a toast whose Try again reruns the request", async () => {
    const container = dom.window.document.getElementById("root");
    if (!container) throw new Error("missing test root");
    const root = createRoot(container);
    let retried = 0;
    try {
      await act(async () => root.render(<AgentConsole events={[{ type: "agent_start", prompt: "q" }, { type: "error", message: "request failed: 502" }]} reasoning={false} debug={false} onRetry={() => { retried += 1; }} />));
      const stack = dom.window.document.body.querySelector('[aria-label="Notifications"]');
      expect(stack?.textContent).toBe("✗ The request failed. Please retry.Try again");
      const retry = [...(stack?.querySelectorAll("button") ?? [])].find((b) => b.textContent === "Try again");
      await act(async () => retry?.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })));
      expect(retried).toBe(1);
      expect(dom.window.document.body.querySelector('[aria-label="Notifications"]')).toBeNull();
      await act(async () => root.render(<AgentConsole events={[{ type: "agent_start", prompt: "q" }, { type: "chart_update", ticker: "AAPL" }]} reasoning={false} debug={false} />));
      expect(dom.window.document.body.querySelector('[aria-label="Notifications"]')?.textContent).toBe("Chart set to AAPLOpen workspace");
    } finally {
      await act(async () => root.unmount());
    }
  });

  test("a widget toast says Added only after the workspace confirmed it, the limit message when blocked, and stays neutral otherwise", async () => {
    const container = dom.window.document.getElementById("root");
    if (!container) throw new Error("missing test root");
    const root = createRoot(container);
    const added: AgentEvent = { type: "widget_update", ticker: "NVDA", widgetType: "news" };
    const blocked: AgentEvent = { type: "widget_update", ticker: "NVDA", widgetType: "insider" };
    const events: AgentEvent[] = [{ type: "agent_start", prompt: "q" }, added, blocked];
    const toasts = () => [...(dom.window.document.body.querySelector('[aria-label="Notifications"]')?.children ?? [])].map((t) => t.textContent);
    try {
      await act(async () => root.render(<AgentConsole events={events} reasoning={false} debug={false} />));
      expect(toasts()).toEqual(["Widget requested: news for NVDAOpen workspace", "Widget requested: insider for NVDAOpen workspace"]);
      await act(async () => root.render(<AgentConsole events={events} reasoning={false} debug={false} widgetOutcomes={new Map([[added, "added"], [blocked, "blocked"]])} />));
      expect(toasts()).toEqual([
        "Added the news widget to NVDAOpen workspace",
        "Workspace limit reached: the insider widget was not added to NVDA (24 widgets max)Open workspace",
      ]);
    } finally {
      await act(async () => root.unmount());
    }
  });
});

describe("toastAlert", () => {
  const alert = (id: string): FeedItem => ({ id, kind: "alert", title: `8-K filed by NVDA ${id}`, at: "2026-10-20T00:00:00Z", source: "thesis:nvda" });
  const toasts = () => [...dom.window.document.querySelectorAll('[aria-label="Alerts"] > div')].map((t) => t.textContent);

  test("an unseen alert id toasts once, marks the seen key, and notifies only when granted", () => {
    const registry = globalThis as Record<string, unknown>;
    const shown: { title: string; body?: string }[] = [];
    let permission = "denied";
    registry.localStorage = dom.window.localStorage;
    registry.Notification = class {
      static get permission() { return permission; }
      constructor(title: string, options?: { body?: string }) { shown.push({ title, body: options?.body }); }
    };
    dom.window.localStorage.setItem(SEEN_ALERTS_KEY, JSON.stringify(["old"]));
    try {
      toastAlert(alert("old"));
      toastAlert({ ...alert("news-1"), kind: "news" });
      expect(toasts()).toEqual([]);
      toastAlert(alert("a"));
      toastAlert(alert("a"));
      expect(toasts()).toEqual(["Alert: 8-K filed by NVDA a"]);
      expect(shown).toEqual([]);
      expect(JSON.parse(dom.window.localStorage.getItem(SEEN_ALERTS_KEY) ?? "[]")).toEqual(["old", "a"]);
      permission = "granted";
      toastAlert(alert("b"));
      expect(shown).toEqual([{ title: "8-K filed by NVDA b", body: "thesis:nvda" }]);
      dom.window.localStorage.clear();
      delete registry.localStorage;
      toastAlert(alert("c"));
      toastAlert(alert("c"));
      expect(toasts()).toEqual(["Alert: 8-K filed by NVDA a", "Alert: 8-K filed by NVDA b", "Alert: 8-K filed by NVDA c"]);
    } finally {
      delete registry.localStorage;
      delete registry.Notification;
      dom.window.document.querySelector('[aria-label="Alerts"]')?.remove();
    }
  });
});
