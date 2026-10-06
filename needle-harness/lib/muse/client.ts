import type { Evidence } from "../agent/types";

export type MuseUsage = {
  inputTokens?: number;
  outputTokens?: number;
  cachedTokens?: number;
  cost?: number;
};

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}
// User-provided Muse Spark 1.3 Contributor per-1M-token rates.
const INPUT_PER_M = 0.1;
const OUTPUT_PER_M = 0.2;
const CACHE_PER_M = 0.002;

const SYSTEM = `You are Stockbot. Answer only from the EVIDENCE below. Structure your answer in three parts: sourced facts (cite [E1] ids), inference, uncertainty. If the evidence is thin, say what is missing. Never quote or repeat the [Today UTC YYYY-MM-DD] bracket from the request in answers or clarifications — decode it silently to dates.`;

// Truncate oldest-first to fit the Muse request budget.
function formatEvidence(evidence: Evidence[]): string {
  const parts = evidence.map(
    (e) =>
      `[${e.id}]\nsource: ${e.source}\n${e.title ? `title: ${e.title}\n` : ""}${e.url ? `url: ${e.url}\n` : ""}content: ${e.content}`,
  );
  const kept: string[] = [];
  let total = 0;
  for (let i = parts.length - 1; i >= 0; i--) {
    if (total + parts[i].length > 24000) break;
    kept.unshift(parts[i]);
    total += parts[i].length;
  }
  return kept.join("\n\n");
}

// Exported for tests: a chunk matching both delta shapes appends once.
export function harvest(ev: unknown, acc: { text: string; usage: MuseUsage }, onDelta: (t: string) => void): void {
  if (!ev || typeof ev !== "object") return;
  const rec = ev as Record<string, unknown>;
  if (typeof rec["delta"] === "string" && /delta/i.test(typeof rec["type"] === "string" ? rec["type"] : "")) {
    acc.text += rec["delta"] as string;
    onDelta(rec["delta"] as string);
  } else {
    const first = Array.isArray(rec["choices"]) ? rec["choices"][0] : undefined;
    const delta = first && typeof first === "object" ? (first as Record<string, unknown>)["delta"] : undefined;
    const content = delta && typeof delta === "object" ? (delta as Record<string, unknown>)["content"] : undefined;
    if (typeof content === "string" && content) {
      acc.text += content;
      onDelta(content);
    }
  }
  const response = rec["response"];
  const nested = response && typeof response === "object" ? (response as Record<string, unknown>)["usage"] : undefined;
  const rawUsage = rec["usage"] ?? nested;
  if (!rawUsage || typeof rawUsage !== "object") return;
  const u = rawUsage as Record<string, unknown>;
  // Usage accumulates across SSE chunks: keep the max, never last-write-wins.
  if (typeof u["input_tokens"] === "number")
    acc.usage.inputTokens = Math.max(acc.usage.inputTokens ?? 0, u["input_tokens"]);
  if (typeof u["output_tokens"] === "number")
    acc.usage.outputTokens = Math.max(acc.usage.outputTokens ?? 0, u["output_tokens"]);
  const inDetails = u["input_tokens_details"];
  const promptDetails = u["prompt_tokens_details"];
  const inCached = inDetails && typeof inDetails === "object" ? (inDetails as Record<string, unknown>)["cached_tokens"] : undefined;
  const promptCached = promptDetails && typeof promptDetails === "object" ? (promptDetails as Record<string, unknown>)["cached_tokens"] : undefined;
  const cached = u["cached_tokens"] ?? inCached ?? promptCached;
  if (typeof cached === "number") acc.usage.cachedTokens = Math.max(acc.usage.cachedTokens ?? 0, cached);
  if (typeof u["prompt_tokens"] === "number")
    acc.usage.inputTokens = Math.max(acc.usage.inputTokens ?? 0, u["prompt_tokens"]);
  if (typeof u["completion_tokens"] === "number")
    acc.usage.outputTokens = Math.max(acc.usage.outputTokens ?? 0, u["completion_tokens"]);
}

async function readSse(
  res: Response,
  onDelta: (t: string) => void,
): Promise<{ text: string; usage: MuseUsage }> {
  const acc = { text: "", usage: {} as MuseUsage };
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of chunk.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("data:")) continue;
        const data = t.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        try {
          harvest(JSON.parse(data), acc, onDelta);
        } catch {
          // Partial SSE frame; next chunk completes it.
        }
      }
    }
  }
  const u = acc.usage;
  if (u.inputTokens !== undefined || u.outputTokens !== undefined) {
    // cachedTokens are a subset of inputTokens: bill them at the cache rate, not on top.
    const cached = u.cachedTokens ?? 0;
    const uncachedInput = Math.max(0, (u.inputTokens ?? 0) - cached);
    u.cost = uncachedInput / 1e6 * INPUT_PER_M + cached / 1e6 * CACHE_PER_M + (u.outputTokens ?? 0) / 1e6 * OUTPUT_PER_M;
  }
  return acc;
}

const FETCH_TIMEOUT_MS = 120_000;

async function post(
  base: string,
  apiKey: string,
  session: string,
  path: "/responses" | "/chat/completions",
  body: unknown,
  onDelta: (t: string) => void,
): Promise<{ text: string; usage: MuseUsage }> {
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "User-Agent": "needle-harness/0.1",
      "x-opencode-session": session,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`muse ${path} ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return readSse(res, onDelta);
}

export async function reason(opts: {
  prompt: string;
  evidence: Evidence[];
  escalated: boolean;
  direct?: boolean;
  // Optional graph state (kernel.ts always sends these; direct callers omit them).
  // incompleteGuard/unresolved force the missing-evidence line even with evidence present.
  objective?: string;
  nodes?: unknown[];
  decisions?: unknown[];
  unresolved?: string[];
  incompleteGuard?: boolean;
  onDelta: (text: string) => void;
}): Promise<{ text: string; usage: MuseUsage; missingEvidence?: string }> {
  const apiKey = requiredEnv("OPENCODE_API_KEY");
  const model = requiredEnv("OPENCODE_MODEL");
  const base = requiredEnv("OPENCODE_URL").replace(/\/responses\/?$/, "");
  const session = crypto.randomUUID();
  const needsGapLine =
    opts.escalated || opts.incompleteGuard === true || (opts.unresolved !== undefined && opts.unresolved.length > 0);
  const system = opts.direct
    ? "You are Stockbot. Answer the user directly and briefly from the supplied context. Never quote or repeat the [Today UTC YYYY-MM-DD] bracket from the request in answers or clarifications — decode it silently to dates. If the request is unclear, ask what they mean and say what you can look up: the time, SEC filings, or a web search."
    : SYSTEM + (needsGapLine ? "\nRetrieval escalated with no usable evidence. End your answer with a line: Missing-Evidence: <what is needed>." : "");
  const user = opts.direct ? `USER REQUEST\n${opts.prompt}` : `USER REQUEST\n${opts.prompt}\n\nEVIDENCE\n${formatEvidence(opts.evidence)}`;
  const input = [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
  try {
    const r = await post(base, apiKey, session, "/responses", { model, input, stream: true }, opts.onDelta);
    return { ...r, missingEvidence: /^Missing-Evidence:\s*(.+)$/m.exec(r.text)?.[1] };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!/(404|405|429|500|502|503|504)/.test(msg)) throw err;
    const r = await post(
      base,
      apiKey,
      session,
      "/chat/completions",
      { model, messages: input.map((m) => ({ role: m.role, content: m.content })), stream: true },
      opts.onDelta,
    );
    return { ...r, missingEvidence: /^Missing-Evidence:\s*(.+)$/m.exec(r.text)?.[1] };
  }
}
