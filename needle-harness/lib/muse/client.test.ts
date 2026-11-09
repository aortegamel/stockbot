import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { reason } from "./client";

const realFetch = globalThis.fetch;
const saved = { key: process.env.OPENCODE_API_KEY, url: process.env.OPENCODE_URL, model: process.env.OPENCODE_MODEL };

// One answer delta plus the completed response, in the Responses stream shape.
function stream(text: string): Response {
  return new Response(
    `data: ${JSON.stringify({ type: "response.output_text.delta", delta: text })}\n\n` +
    `data: ${JSON.stringify({ type: "response.completed", response: { output: [], usage: { output_tokens: 3 } } })}\n\n`,
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

// Capture the provider request body and answer it with one plain delta.
function capture(bodies: Record<string, unknown>[], text = "An 8-K."): void {
  globalThis.fetch = (async (_url: string, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)));
    return stream(text);
  }) as unknown as typeof fetch;
}

type Message = { role: string; content: string };

describe("reason plain chat messages", () => {
  beforeEach(() => {
    process.env.OPENCODE_API_KEY = "test-key";
    process.env.OPENCODE_URL = "https://provider.test/v1/responses";
    process.env.OPENCODE_MODEL = "test-model";
  });
  afterEach(() => {
    globalThis.fetch = realFetch;
    for (const [name, value] of [["OPENCODE_API_KEY", saved.key], ["OPENCODE_URL", saved.url], ["OPENCODE_MODEL", saved.model]] as const) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });

  test("earlier turns arrive as conversation between the system message and the request", async () => {
    const bodies: Record<string, unknown>[] = [];
    capture(bodies);
    const history = [
      { role: "user", content: "What did NVDA file?" },
      { role: "assistant", content: "An 8-K on 2 September." },
    ] as const;
    const result = await reason({ prompt: "And the price?", evidence: [], direct: true, plain: true, history: [...history] });
    expect(result.prose).toBe("An 8-K.");
    const messages = bodies[0]?.input as Message[];
    expect(messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "user"]);
    expect(messages[1]).toEqual({ role: "user", content: "What did NVDA file?" });
    expect(messages[2]).toEqual({ role: "assistant", content: "An 8-K on 2 September." });
    expect(messages[3]?.content).toStartWith("USER REQUEST\nAnd the price?");
    expect(messages[0]?.content).toContain("earlier turns of this chat");
  });

  test("no history keeps one system message and one request, with no history rule", async () => {
    const bodies: Record<string, unknown>[] = [];
    capture(bodies);
    await reason({ prompt: "And the price?", evidence: [], direct: true, plain: true });
    const messages = bodies[0]?.input as Message[];
    expect(messages.map((message) => message.role)).toEqual(["system", "user"]);
    expect(messages[0]?.content).not.toContain("earlier turns of this chat");
  });
});
