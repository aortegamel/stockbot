import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { existsSync } from "node:fs";
import * as childProcess from "node:child_process";
import { EventEmitter } from "node:events";
import { categorizeFailure, closeBridge, endSession, invoke, newSessionId } from "./stockbot";
import { harvest, reason, splitInventory } from "../muse/client";
import fc from "fast-check";
import { FINAL_PERSONAS, sourceReferences, type Evidence, type Persona } from "../agent/types";

const ROOT = process.cwd().endsWith("needle-harness")
  ? process.cwd().replace(/\/needle-harness$/, "")
  : process.cwd();
const HAS_BRIDGE = existsSync(`${ROOT}/venv/bin/python`) && existsSync(`${ROOT}/scripts/tool_bridge.py`);

describe.skipIf(!HAS_BRIDGE)("stockbot bridge", () => {
  const sid = newSessionId();

  // Test-only cleanup; harness runtime never calls closeBridge.
  afterAll(async () => {
    await endSession(sid);
    closeBridge();
  });

  test(
    "invalid accession fails closed with accession error",
    async () => {
      const result = await invoke("get_sec_document", { accession_no: "bad" }, sid);
      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error).toMatch(/accession/i);
        expect(result.category).toBe("tool_error");
        expect(result.retryable).toBe(false);
      }
    },
    60_000,
  );

  test(
    "a local tool reply becomes evidence with a stable id",
    async () => {
      const result = await invoke("list_tool_domains", {}, sid);
      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.evidence.source).toBe("list_tool_domains");
        expect(result.evidence.id).toMatch(/^ev:[0-9a-f]{16}$/);
        expect(result.evidence.title).toBe("list_tool_domains");
        expect(result.evidence.content).toContain("sec");
      }
    },
    60_000,
  );

  test(
    "a malformed request surfaces the bridge envelope error",
    async () => {
      // The bridge rejects a request without a session id; the envelope error path
      // must categorize it without inventing a tool result.
      expect(await invoke("list_tool_domains", {}, "")).toEqual({
        ok: false,
        error: "missing_arg",
        category: "tool_error",
        retryable: false,
      });
    },
    60_000,
  );

  // Last in this block: it closes the shared bridge child.
  test(
    "an in-flight call fails when the bridge closes instead of hanging",
    async () => {
      const inFlight = invoke("list_tool_domains", {}, sid);
      closeBridge();
      expect(await inFlight).toEqual({
        ok: false,
        error: "stockbot bridge closed",
        category: "tool_error",
        retryable: false,
      });
    },
    60_000,
  );
});

describe("stockbot bridge chart metadata", () => {
  let reply: Record<string, unknown>;
  let restoreSpawn = (): void => { };
  const selection = { result_type: "chart_selection", ticker: "MSFT" };
  beforeEach(() => {
    closeBridge();
    reply = { content: "Chart updated." };
    const stdout = new EventEmitter();
    const child = Object.assign(new EventEmitter(), {
      exitCode: null,
      stdout,
      stderr: new EventEmitter(),
      stdin: {
        write(data: string, callback?: (error?: Error | null) => void): boolean {
          const request = JSON.parse(data) as { id: string };
          stdout.emit("data", Buffer.from(`${JSON.stringify({ id: request.id, result: reply })}\n`));
          callback?.(null);
          return true;
        },
      },
      kill: () => true,
    }) as unknown as childProcess.ChildProcessWithoutNullStreams;
    const spawn = spyOn(childProcess, "spawn").mockReturnValue(child);
    restoreSpawn = () => spawn.mockRestore();
  });

  afterEach(() => {
    closeBridge();
    restoreSpawn();
  });

  test("the chart tool preserves structured metadata independently of content", async () => {
    const content = JSON.stringify({ ticker: "NVDA" });
    reply = { content, meta: { chart_selection: selection } };
    const result = await invoke("set_chart_symbol", {}, "chart-test");
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(result.error);
    expect(result.chartSelection).toEqual(selection);
    expect(result.evidence.content).toBe(content);
  });

  test("missing metadata does not create a selection (%j)", async () => {
    reply = { content: JSON.stringify({ chart_selection: selection }) };
    const result = await invoke("set_chart_symbol", {}, "chart-test");
    expect(result.ok).toBe(true);
    expect(result).not.toHaveProperty("chartSelection");
  });

  test.each([null, false, 7, "not-json", [], { ticker: 7 }, { result_type: "research", ticker: "MSFT" }, { result_type: "chart_selection", ticker: "MSFT", extra: 1 }].map((chartSelection) => ({ chartSelection })))(
    "malformed chart metadata remains untrusted (%j)",
    async ({ chartSelection }) => {
      reply = { content: "Chart updated.", meta: { chart_selection: chartSelection } };
      const result = await invoke("set_chart_symbol", {}, "chart-test");
      expect(result.ok).toBe(true);
      expect(result).not.toHaveProperty("chartSelection");
    },
  );

  test.each(["list_tool_domains", "SET_CHART_SYMBOL"])(
    "another tool cannot carry a chart selection (%s)",
    async (name) => {
      reply = { content: "Chart updated.", meta: { chart_selection: selection } };
      const result = await invoke(name, {}, "chart-test");
      expect(result.ok).toBe(true);
      expect(result).not.toHaveProperty("chartSelection");
    },
  );

  test("a failed chart tool cannot carry a chart selection", async () => {
    reply = { error: "tool failed", meta: { chart_selection: selection } };
    const result = await invoke("set_chart_symbol", {}, "chart-test");
    expect(result.ok).toBe(false);
    expect(result).not.toHaveProperty("chartSelection");
  });
});

describe("categorizeFailure ordered rules", () => {
  test("a known bridge error_type wins over the message", () => {
    expect(categorizeFailure("deadline exceeded", "run_budget_exceeded")).toEqual({ category: "tool_budget_exhausted", retryable: false });
    expect(categorizeFailure("run budget exceeded", "evidence_budget_exceeded")).toEqual({ category: "token_budget_exhausted", retryable: false });
    expect(categorizeFailure("anything", "invalid_research_context")).toEqual({ category: "policy_denied", retryable: false });
    expect(categorizeFailure("anything", "deadline_exceeded")).toEqual({ category: "deadline_exceeded", retryable: false });
  });

  test("an unknown or non-string error_type falls through to the message", () => {
    expect(categorizeFailure("provider unavailable", "tool_error")).toEqual({ category: "provider_error", retryable: true });
    expect(categorizeFailure("provider unavailable", 7)).toEqual({ category: "provider_error", retryable: true });
  });

  test("budget rules place token, job and wave before the generic rule", () => {
    expect(categorizeFailure("token budget exhausted")).toEqual({ category: "token_budget_exhausted", retryable: false });
    expect(categorizeFailure("job budget exhausted")).toEqual({ category: "job_budget_exhausted", retryable: false });
    expect(categorizeFailure("wave quota spent")).toEqual({ category: "wave_budget_exhausted", retryable: false });
    expect(categorizeFailure("quota spent")).toEqual({ category: "tool_budget_exhausted", retryable: false });
  });

  test("a bare token, job or wave mention is not a budget refusal", () => {
    expect(categorizeFailure("token refreshed")).toEqual({ category: "tool_error", retryable: false });
    expect(categorizeFailure("job started")).toEqual({ category: "tool_error", retryable: false });
    expect(categorizeFailure("wave 2 started")).toEqual({ category: "tool_error", retryable: false });
  });

  test("policy, loop, timeout and provider traps keep category and retryability", () => {
    for (const message of ["policy denied", "not permitted", "not authorized", "blocked by intent", "private data withheld", "egress blocked"]) {
      expect(categorizeFailure(message)).toEqual({ category: "policy_denied", retryable: false });
    }
    for (const message of ["research_loop_detected", "repeats an action", "already ran this step"]) {
      expect(categorizeFailure(message)).toEqual({ category: "research_loop_detected", retryable: false });
    }
    for (const message of ["request timed out", "timed_out waiting", "token expired"]) {
      expect(categorizeFailure(message)).toEqual({ category: "timeout", retryable: true });
    }
    expect(categorizeFailure("provider unavailable")).toEqual({ category: "provider_error", retryable: true });
    expect(categorizeFailure("unexpected failure")).toEqual({ category: "tool_error", retryable: false });
  });
});

describe("categorizeFailure precedence", () => {
  test("timeout beats duplicate/budget/provider traps", () => {
    expect(categorizeFailure("request timed out; duplicate action suspected").category).toBe("timeout");
    expect(categorizeFailure("timeout waiting on provider budget check").category).toBe("timeout");
  });

  test("deadline beats duplicate/budget/provider traps", () => {
    expect(categorizeFailure("deadline exceeded; duplicate budget recheck from provider").category).toBe(
      "deadline_exceeded",
    );
  });

  test("substrates alone still categorize", () => {
    expect(categorizeFailure("duplicate research action").category).toBe("duplicate_research_action");
    expect(categorizeFailure("provider unavailable").category).toBe("provider_error");
    expect(categorizeFailure("run budget exceeded").category).toBe("tool_budget_exhausted");
  });
});

describe("harvest single-append", () => {
  test("chunk matching both delta paths appends once", () => {
    const acc = { text: "", usage: {} };
    const seen: string[] = [];
    harvest({ type: "response.delta", delta: "hi", choices: [{ delta: { content: "hi" } }] }, acc, (t) => seen.push(t));
    expect(acc.text).toBe("hi");
    expect(seen).toEqual(["hi"]);
  });

  test("usage accumulates by max across chunks", () => {
    const acc = { text: "", usage: {} };
    const noop = () => { };
    harvest({ usage: { input_tokens: 100, output_tokens: 10 } }, acc, noop);
    harvest({ usage: { input_tokens: 50, output_tokens: 40 } }, acc, noop);
    expect(acc.usage).toMatchObject({ inputTokens: 100, outputTokens: 40 });
  });
});

describe("harvest protocol shapes", () => {
  const noop = () => { };

  test("events without a recognized delta shape append nothing", () => {
    const acc = { text: "", usage: {} };
    const seen: string[] = [];
    const shapes: unknown[] = [
      null, 7, "delta", [], { delta: "raw" }, { type: "response.created", delta: 5 },
      { choices: [] }, { choices: "x" }, { choices: [{}] }, { choices: [null] },
      { choices: [{ delta: { content: "" } }] }, { choices: [{ delta: "x" }] },
    ];
    for (const ev of shapes) harvest(ev, acc, (t) => seen.push(t));
    expect(acc.text).toBe("");
    expect(seen).toEqual([]);
    expect(acc.usage).toEqual({});
  });

  test("a chat-completions chunk appends its content once", () => {
    const acc = { text: "", usage: {} };
    const seen: string[] = [];
    harvest({ choices: [{ delta: { content: "chat" } }] }, acc, (t) => seen.push(t));
    expect(acc.text).toBe("chat");
    expect(seen).toEqual(["chat"]);
  });

  test("the response envelope usage is read when the chunk carries none", () => {
    const acc = { text: "", usage: {} };
    harvest({ response: { usage: { input_tokens: 3, output_tokens: 4 } } }, acc, noop);
    expect(acc.usage).toEqual({ inputTokens: 3, outputTokens: 4 });
    // A chunk-level usage wins over the nested envelope for the same chunk.
    harvest({ usage: { input_tokens: 12 }, response: { usage: { input_tokens: 40 } } }, acc, noop);
    expect(acc.usage).toEqual({ inputTokens: 12, outputTokens: 4 });
  });

  test("usage accepts both provider count names and both cached detail keys", () => {
    const acc = { text: "", usage: {} };
    harvest({ usage: { prompt_tokens: 5, completion_tokens: 6, cached_tokens: 2 } }, acc, noop);
    expect(acc.usage).toEqual({ inputTokens: 5, outputTokens: 6, cachedTokens: 2 });
    harvest({ usage: { input_tokens_details: { cached_tokens: 9 } } }, acc, noop);
    expect(acc.usage).toEqual({ inputTokens: 5, outputTokens: 6, cachedTokens: 9 });
    harvest({ usage: { prompt_tokens_details: { cached_tokens: 4 } } }, acc, noop);
    expect(acc.usage).toEqual({ inputTokens: 5, outputTokens: 6, cachedTokens: 9 });
  });

  test("malformed usage values are ignored, never coerced", () => {
    const acc = { text: "", usage: {} };
    for (const usage of [null, 5, "usage", {}, { input_tokens: "3" }, { output_tokens: null }]) {
      harvest({ usage }, acc, noop);
    }
    harvest({ usage: { input_tokens_details: 7, prompt_tokens_details: null } }, acc, noop);
    expect(acc.usage).toEqual({});
  });
});

describe("muse inventory fence", () => {
  test("prose without a fence passes through unchanged", () => {
    expect(splitInventory("A fact [E1].")).toEqual({ prose: "A fact [E1].", claims: [] });
  });

  test("a trailing fence is stripped and its claims are coerced", () => {
    const payload = {
      claims: [
        { text: "A fact [E1].", claim_type: " Observed_Fact ", evidence_ids: ["E1", "", 3], confidence: 0.9 },
        { text: "An inference.", claim_type: "inference", evidence_ids: "E1" },
        { text: "", claim_type: "inference" },
        { text: "Bad type.", claim_type: "not_a_type" },
        { text: "No type." },
        "skip",
        null,
        [],
      ],
      confidence: 0.75,
    };
    const inv = splitInventory(`A fact [E1].\nAn inference.\n\`\`\`json ${JSON.stringify(payload)} \`\`\``);
    expect(inv.prose).toBe("A fact [E1].\nAn inference.\n");
    expect(inv.claims).toEqual([
      { text: "A fact [E1].", claimType: "observed_fact", evidenceIds: ["E1"], confidence: 0.9 },
      { text: "An inference.", claimType: "inference", evidenceIds: [] },
    ]);
    expect(inv.confidence).toBe(0.75);
  });

  test("an unusable fence is stripped without inventing claims", () => {
    for (const body of ["{not json}", "[1,2]", "5", "null", '{"claims": 5}', '{"claims": [null]}']) {
      const inv = splitInventory(`Prose.\n\`\`\`json ${body}\`\`\``);
      expect(inv.prose).toBe("Prose.\n");
      expect(inv.claims).toEqual([]);
      expect(inv.confidence).toBeUndefined();
    }
  });

  test("out-of-range confidence is dropped, never clamped", () => {
    const raw = (confidence: number): string =>
      `Prose.\n\`\`\`json ${JSON.stringify({ claims: [{ text: "T", claim_type: "inference", confidence }], confidence })}\`\`\``;
    expect(splitInventory(raw(0)).claims[0].confidence).toBe(0);
    expect(splitInventory(raw(1)).confidence).toBe(1);
    for (const out of [-0.1, 1.1]) {
      const inv = splitInventory(raw(out));
      expect(inv.claims[0].confidence).toBeUndefined();
      expect(inv.confidence).toBeUndefined();
    }
  });
});

describe("server source metadata", () => {
  const capture = { id: `capture:${"a".repeat(64)}`, title: "Filing", url: "https://www.sec.gov/filing" };

  test("valid references retain only public fields and keep source order", () => {
    expect(sourceReferences([
      { ...capture, raw_archive_path: "/private/archive" },
      capture,
      { id: "evidence:rs:test:ev:1", title: 42, url: "javascript:alert(1)" },
    ])).toEqual([capture, { id: "evidence:rs:test:ev:1" }]);
  });

  test("malformed metadata never creates a navigable id", () => {
    for (const raw of [undefined, null, "capture", {}, 3]) expect(sourceReferences(raw)).toEqual([]);
    expect(sourceReferences([
      { id: "evidence:foo\n" }, { id: "evidence:foo\r" },
      { id: "/private/raw" }, { id: "https://www.sec.gov/filing" },
      { id: "evidence:../raw" }, { id: "evidence:foo%2fbar" },
      { id: `capture:${"A".repeat(64)}` }, { id: `capture:${"a".repeat(63)}` },
      { id: `evidence:${"a".repeat(257)}` },
    ])).toEqual([]);
    expect(sourceReferences([{ id: `evidence:${"a".repeat(256)}` }])).toHaveLength(1);
  });

  test("JSON metadata preserves only supplied valid ids and normalizes idempotently", () => {
    fc.assert(fc.property(fc.jsonValue(), (raw) => {
      const parsed = sourceReferences(raw);
      expect(sourceReferences(parsed)).toEqual(parsed);
      expect(new Set(parsed.map((r) => r.id)).size).toBe(parsed.length);
      const supplied = Array.isArray(raw) ? raw : [];
      for (const ref of parsed) {
        expect(supplied.some((r) => r !== null && typeof r === "object" && "id" in r && r.id === ref.id)).toBe(true);
        expect(ref.id).toMatch(/^(?:capture:[a-f0-9]{64}|evidence:[A-Za-z0-9:_-]{1,256})$/);
      }
    }), { seed: 20261010, numRuns: 100 });
  });
});

describe("muse reason personas (fake provider)", () => {
  type Msg = { role: string; content: string };
  type Body = { model: string; input?: Msg[]; messages?: Msg[] };
  type Seen = { path: string; session: string | null; body: Body };
  const seen: Seen[] = [];
  let respond: (path: string, body: Body) => Response = () => sse("ok");
  let stopServer = (): void => { };
  const KEYS = ["OPENCODE_API_KEY", "OPENCODE_MODEL", "OPENCODE_URL"] as const;
  const prevEnv = Object.fromEntries(KEYS.map((k) => [k, process.env[k]]));

  function sse(...deltas: string[]): Response {
    const events = deltas.map((delta) => `data: ${JSON.stringify({ type: "response.output_text.delta", delta })}\n\n`);
    return new Response(`${events.join("")}data: ${JSON.stringify({ usage: { input_tokens: 7, output_tokens: 3 } })}\n\ndata: [DONE]\n\n`, {
      headers: { "Content-Type": "text/event-stream" },
    });
  }
  const systemOf = (b: Body): string => (b.input ?? b.messages ?? [])[0].content;
  const userOf = (b: Body): string => (b.input ?? b.messages ?? [])[1].content;
  // Fake provider plays whichever persona the system message names; only the bear asks for more evidence.
  const personaReply = (_path: string, b: Body): Response => {
    const who = /^You are (\w+)\./.exec(systemOf(b))?.[1] ?? "nobody";
    return who === "Bearbot" ? sse("Bearbot read [E1].\n", "Missing-Evidence: revolver covenant terms") : sse(`${who} read [E1].`);
  };
  // One draft delta, then the stream never ends.
  const hanging = (): Response =>
    new Response(
      new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ type: "response.output_text.delta", delta: "draft" })}\n\n`));
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );

  const evidence: Evidence[] = [
    { id: "E1", source: "sec", retrievedAt: "2026-01-01T00:00:00Z", content: "Revolver matures 2027; $2.1B drawn." },
    { id: "E2", source: "sec", retrievedAt: "2026-01-01T00:00:00Z", content: "Top customer is 38% of revenue." },
  ];
  const input = {
    prompt: "Is ACME exposed to its lender?\n\nOBJECTIVE (verbatim user intent; never restate or change it): ACME lender exposure",
    evidence,
    escalated: false,
    unresolved: ["n2"],
    onDelta: () => { },
  };

  beforeAll(() => {
    const server = Bun.serve({
      port: 0,
      async fetch(req) {
        const path = new URL(req.url).pathname;
        const body = (await req.json()) as Body;
        seen.push({ path, session: req.headers.get("x-opencode-session"), body });
        return respond(path, body);
      },
    });
    process.env.OPENCODE_API_KEY = "test-key";
    process.env.OPENCODE_MODEL = "test-model";
    process.env.OPENCODE_URL = `http://127.0.0.1:${server.port}/v1/responses`;
    stopServer = () => server.stop(true);
  });

  afterAll(() => {
    stopServer();
    for (const k of KEYS) {
      if (prevEnv[k] === undefined) delete process.env[k];
      else process.env[k] = prevEnv[k];
    }
  });

  beforeEach(() => {
    seen.length = 0;
    respond = personaReply;
  });

  test("personas see the same request and evidence on separate sessions; gaps stay per persona", async () => {
    const results = await Promise.all(FINAL_PERSONAS.map((persona) => reason({ ...input, ticker: "NVDA", persona })));
    expect(results.map((r) => r.text)).toEqual([
      "Stockbot read [E1].",
      "Bearbot read [E1].\nMissing-Evidence: revolver covenant terms",
      "Bullbot read [E1].",
    ]);
    expect(results.map((r) => r.missingEvidence)).toEqual([undefined, "revolver covenant terms", undefined]);
    expect(results[0].usage).toEqual({ inputTokens: 7, outputTokens: 3 });
    expect(seen.map((s) => s.path)).toEqual(["/v1/responses", "/v1/responses", "/v1/responses"]);
    expect(new Set(seen.map((s) => s.session)).size).toBe(3);
    // Shared evidence and graph input: every persona gets the identical user message.
    const users = new Set(seen.map((s) => userOf(s.body)));
    expect(users.size).toBe(1);
    const user = [...users][0];
    for (const fact of ["OBJECTIVE", "[E1]", "[E2]", "Revolver matures 2027", "Top customer is 38%"]) expect(user).toContain(fact);
    expect(user).toStartWith(`USER REQUEST\n${input.prompt}\n\nEVIDENCE\n`);
    // Roles differ, and each keeps the open-node gap instruction.
    const systems = seen.map((s) => systemOf(s.body));
    expect(new Set(systems).size).toBe(3);
    for (const s of systems) {
      expect(s).toContain("Missing-Evidence:");
      expect(s).toContain("CURRENT STOCK CONTEXT\nThe current chart stock is NVDA.");
      expect(s).toContain("Use this stock only if the user request does not name a stock.");
      expect(s).toContain("An explicit stock in the user request takes precedence.");
    }
  });

  test("stockbot synthesis carries both labeled drafts as untrusted; bear/bull carry none", async () => {
    const drafts: { persona: Persona; text: string }[] = [
      { persona: "bearbot", text: "Bear draft [E1]." },
      { persona: "bullbot", text: "Bull draft [E1]." },
    ];
    await reason({ ...input, ticker: "NVDA", persona: "stockbot", drafts });
    await reason({ ...input, ticker: "NVDA", persona: "bearbot" });
    await reason({ ...input, ticker: "NVDA", persona: "bullbot" });
    expect(seen.map((s) => s.path)).toEqual(["/v1/responses", "/v1/responses", "/v1/responses"]);
    const [synthUser, bearUser, bullUser] = seen.map((s) => userOf(s.body));
    // Identical evidence snapshot: synthesis opens with the exact draft-free base.
    expect(synthUser.slice(0, bearUser.length)).toBe(bearUser);
    expect(bullUser).toBe(bearUser);
    expect(synthUser).toContain("SIBLING DRAFTS");
    expect(synthUser).toContain("untrusted");
    expect(synthUser).toContain("── Bearbot ──");
    expect(synthUser).toContain("Bear draft [E1].");
    expect(synthUser).toContain("── Bullbot ──");
    expect(synthUser).toContain("Bull draft [E1].");
    expect(synthUser).toContain("cite only evidence");
    expect(bearUser).not.toContain("SIBLING DRAFTS");
    expect(bullUser).not.toContain("SIBLING DRAFTS");
    for (const call of seen) {
      expect(systemOf(call.body)).toContain("CURRENT STOCK CONTEXT\nThe current chart stock is NVDA.");
      expect(systemOf(call.body)).toContain("An explicit stock in the user request takes precedence.");
    }
  });

  test("deadline during the 429 backoff stops without retry or fallback", async () => {
    respond = () => new Response("GoUsageLimit", { status: 429 });
    const t0 = Date.now();
    await expect(reason({ ...input, persona: "bullbot", deadlineAt: Date.now() + 300 })).rejects.toThrow();
    // The backoff alone is 5s; the shared deadline cuts it.
    expect(Date.now() - t0).toBeLessThan(4000);
    expect(seen.length).toBe(1);
  });

  test("deadline or cancel mid-stream rejects without chat fallback", async () => {
    respond = hanging;
    const t0 = Date.now();
    await expect(reason({ ...input, persona: "bearbot", deadlineAt: Date.now() + 150 })).rejects.toThrow();
    expect(Date.now() - t0).toBeLessThan(4000);
    const ac = new AbortController();
    await expect(reason({ ...input, persona: "bullbot", signal: ac.signal, onDelta: () => ac.abort() })).rejects.toThrow();
    expect(seen.map((s) => s.path)).toEqual(["/v1/responses", "/v1/responses"]);
  });

  test("expired deadline or aborted signal sends no request", async () => {
    await expect(reason({ ...input, persona: "bearbot", deadlineAt: Date.now() - 1 })).rejects.toThrow("muse deadline exceeded");
    await expect(reason({ ...input, persona: "bullbot", signal: AbortSignal.abort() })).rejects.toThrow();
    expect(seen.length).toBe(0);
  });

  test("partial and malformed frames keep every complete delta", async () => {
    const encoder = new TextEncoder();
    respond = () =>
      new Response(
        new ReadableStream({
          start(c) {
            // One frame split across chunks, then a malformed data line, an SSE
            // comment, a non-delta event, a chat-shaped delta, and a usage-only chunk.
            c.enqueue(encoder.encode(`data: {"type":"response.output_text.delta","delta":"par`));
            c.enqueue(
              encoder.encode(
                `tial"}\n\ndata: {not json}\n\n: keep-alive\n\ndata: {"type":"response.created"}\n\n` +
                `data: ${JSON.stringify({ choices: [{ delta: { content: " chat" } }] })}\n\n` +
                `data: ${JSON.stringify({ usage: { input_tokens: 9 } })}\n\n`,
              ),
            );
            c.enqueue(encoder.encode("data: [DONE]\n\n"));
            c.close();
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      );
    const deltas: string[] = [];
    const r = await reason({ ...input, persona: "bullbot", onDelta: (t) => deltas.push(t) });
    expect(r.text).toBe("partial chat");
    expect(deltas).toEqual(["partial", " chat"]);
    expect(r.usage).toEqual({ inputTokens: 9 });
    expect(r.claims).toEqual([]);
  });

  test.each([undefined, "NVDA"])("a direct call sends the direct system prompt without evidence (ticker %s)", async (ticker) => {
    const r = await reason({ prompt: "What is ACME?", ticker, evidence: [], escalated: false, direct: true, onDelta: () => { } });
    expect(r.text).toBe("Stockbot read [E1].");
    expect(seen.length).toBe(1);
    const [call] = seen;
    if (!call) throw new Error("expected one provider call");
    expect(call.path).toBe("/v1/responses");
    expect(systemOf(call.body)).not.toContain("Missing-Evidence:");
    expect(systemOf(call.body)).toBe("You are Stockbot." + (ticker
      ? "\n\nCURRENT STOCK CONTEXT\nThe current chart stock is NVDA. Use this stock only if the user request does not name a stock. An explicit stock in the user request takes precedence."
      : ""));
    expect(userOf(call.body)).toBe("USER REQUEST\nWhat is ACME?");
    expect(userOf(call.body)).not.toContain("EVIDENCE");
  });

  test("a single-tool call without a persona keeps the evidence prompt", async () => {
    const r = await reason({ ...input, persona: undefined, escalated: false, onDelta: () => { } });
    expect(r.text).toBe("Stockbot read [E1].");
    const [call] = seen;
    if (!call) throw new Error("expected one provider call");
    expect(systemOf(call.body)).not.toContain("Bearbot");
    expect(systemOf(call.body)).not.toContain("CURRENT STOCK CONTEXT");
    expect(userOf(call.body)).toContain("EVIDENCE\n[E1]\nsource: sec");
    expect(userOf(call.body)).not.toContain("SIBLING DRAFTS");
  });

  test("reason strips the inventory fence and returns parsed claims", async () => {
    const fence = JSON.stringify({
      claims: [{ text: "Bearbot read [E1].", claim_type: "observed_fact", evidence_ids: ["E1"], confidence: 0.8 }],
      confidence: 0.6,
    });
    respond = () => sse(`Bearbot read [E1].\n\`\`\`json ${fence}\`\`\``);
    const deltas: string[] = [];
    const r = await reason({ ...input, persona: "bearbot", onDelta: (t) => deltas.push(t) });
    expect(r.text).toBe("Bearbot read [E1].\n");
    expect(r.claims).toEqual([{ text: "Bearbot read [E1].", claimType: "observed_fact", evidenceIds: ["E1"], confidence: 0.8 }]);
    expect(r.confidence).toBe(0.6);
    expect(r.missingEvidence).toBeUndefined();
    // Streamed deltas keep the raw block; only the returned prose is display text.
    expect(deltas.join("")).toContain("```json");
    expect(r.text).not.toContain("```json");
  });

  test("unsupported /responses falls back to chat on the same session and persona prompt", async () => {
    respond = (path, b) =>
      path === "/v1/responses"
        ? new Response("not found", { status: 404 })
        : new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: `${/^You are (\w+)\./.exec(systemOf(b))?.[1]} via chat` } }] })}\n\ndata: [DONE]\n\n`, {
          headers: { "Content-Type": "text/event-stream" },
        });
    const r = await reason({ ...input, persona: "bullbot", signal: new AbortController().signal, deadlineAt: Date.now() + 10_000 });
    expect(r.text).toBe("Bullbot via chat");
    expect(seen.map((s) => s.path)).toEqual(["/v1/responses", "/v1/chat/completions"]);
    expect(seen[1].session).toBe(seen[0].session);
    expect(seen[1].body.messages).toEqual(seen[0].body.input);
  });

  describe("controlled clock past 120s", () => {
    // Fake AbortSignal.timeout: advance() fires every timer due by the fake clock.
    const realTimeout = AbortSignal.timeout;
    let now = 0;
    let timers: { due: number; ac: AbortController }[] = [];
    let release = (): void => { };
    let released = Promise.resolve();
    const advance = (ms: number): void => {
      now += ms;
      for (const t of timers) if (t.due <= now) t.ac.abort(new DOMException("The operation timed out.", "TimeoutError"));
    };
    const enc = new TextEncoder();
    const chunk = (delta: string): Uint8Array =>
      enc.encode(`data: ${JSON.stringify({ type: "response.output_text.delta", delta })}\n\n`);
    // One draft delta now; the rest only after the test calls release().
    const gated = (_path: string, b: Body): Response =>
      new Response(
        new ReadableStream({
          async start(c) {
            c.enqueue(chunk(`${/^You are (\w+)\./.exec(systemOf(b))?.[1] ?? "Muse"} `));
            await released;
            c.enqueue(chunk("late."));
            c.enqueue(enc.encode("data: [DONE]\n\n"));
            c.close();
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      );
    // Starts a call and resolves `streaming` at its first delta.
    const start = (opts: Partial<Parameters<typeof reason>[0]>) => {
      let first = (): void => { };
      const streaming = new Promise<void>((r) => (first = r));
      return { streaming, result: reason({ ...input, ...opts, onDelta: () => first() }) };
    };

    beforeEach(() => {
      now = 0;
      timers = [];
      released = new Promise<void>((r) => (release = r));
      respond = gated;
      AbortSignal.timeout = (ms: number): AbortSignal => {
        const ac = new AbortController();
        timers.push({ due: now + ms, ac });
        return ac.signal;
      };
    });

    afterEach(() => {
      AbortSignal.timeout = realTimeout;
    });

    test("all three explicit personas keep generating past 120s", async () => {
      const calls = FINAL_PERSONAS.map((persona) => start({ persona }));
      await Promise.all(calls.map((c) => c.streaming));
      advance(121_000);
      release();
      const results = await Promise.all(calls.map((c) => c.result));
      expect(results.map((r) => r.text)).toEqual(["Stockbot late.", "Bearbot late.", "Bullbot late."]);
      expect(seen.map((s) => s.path)).toEqual(["/v1/responses", "/v1/responses", "/v1/responses"]);
    });

    test("omitted persona keeps the 120s fetch limit without fallback", async () => {
      const call = start({});
      await call.streaming;
      advance(120_000);
      await expect(call.result).rejects.toThrow();
      expect(seen.map((s) => s.path)).toEqual(["/v1/responses"]);
    });

    test("caller cancel aborts a persona generation past 120s without fallback", async () => {
      const ac = new AbortController();
      const call = start({ persona: "bearbot", signal: ac.signal });
      await call.streaming;
      advance(121_000);
      ac.abort();
      await expect(call.result).rejects.toThrow();
      expect(seen.map((s) => s.path)).toEqual(["/v1/responses"]);
    });
  });
});
