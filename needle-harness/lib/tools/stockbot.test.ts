import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { categorizeFailure, closeBridge, endSession, invoke, newSessionId } from "./stockbot";
import { harvest, reason } from "../muse/client";
import { FINAL_PERSONAS, type Evidence, type Persona } from "../agent/types";

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
      if (!result.ok) expect(result.error).toMatch(/accession/i);
    },
    60_000,
  );
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
    const results = await Promise.all(FINAL_PERSONAS.map((persona) => reason({ ...input, persona })));
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
    // Roles differ, and each keeps the open-node gap instruction.
    const systems = seen.map((s) => systemOf(s.body));
    expect(new Set(systems).size).toBe(3);
    for (const s of systems) expect(s).toContain("Missing-Evidence:");
  });

  test("stockbot synthesis carries both labeled drafts as untrusted; bear/bull carry none", async () => {
    const drafts: { persona: Persona; text: string }[] = [
      { persona: "bearbot", text: "Bear draft [E1]." },
      { persona: "bullbot", text: "Bull draft [E1]." },
    ];
    await reason({ ...input, persona: "stockbot", drafts });
    await reason({ ...input, persona: "bearbot" });
    await reason({ ...input, persona: "bullbot" });
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
    expect(synthUser).toContain("middle verdict");
    expect(synthUser).toContain("cite only evidence");
    expect(bearUser).not.toContain("SIBLING DRAFTS");
    expect(bullUser).not.toContain("SIBLING DRAFTS");
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
});
