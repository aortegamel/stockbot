import { setTimeout as sleep } from "node:timers/promises";
import type { Evidence, Persona } from "../agent/types";

export type MuseUsage = {
  inputTokens?: number;
  outputTokens?: number;
  cachedTokens?: number;
};

function requiredEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`opencode_unavailable: missing ${name}`);
  return value;
}

const SYSTEM = `You are Stockbot. Answer only from the EVIDENCE below. Structure your answer in three parts: sourced facts (cite [E1] ids), inference, uncertainty. If the evidence is thin, say what is missing. Never quote or repeat the [Today UTC YYYY-MM-DD] bracket from the request in answers or clarifications — decode it silently to dates.`;

const TODAY_RULE =
  "Never quote or repeat the [Today UTC YYYY-MM-DD] bracket from the request in answers or clarifications — decode it silently to dates.";

// Stance sentences adapted from the old committee roles (app/research/agents/*bot.py) to standalone prose.
const PERSONAS: Record<Persona, { name: string; stance: string }> = {
  stockbot: {
    name: "Stockbot",
    stance:
      "You give the balanced base case: weigh the cited evidence and the gaps in it with equal rigor, and state the most probable evidence-supported outcome.",
  },
  bearbot: {
    name: "Bearbot",
    stance:
      "You give the bear case: build the strongest evidence-supported downside case, including contagion — exposure size, concentration, credit and liquidity channels, commitments, termination triggers, and second-order effects on counterparties. Do not fabricate pessimism. Every adverse finding needs a cited passage. If the evidence supports no downside case, say so instead of asserting one.",
  },
  bullbot: {
    name: "Bullbot",
    stance:
      "You give the bull case: build the strongest evidence-supported resilience case — where the cited evidence shows limited damage, buffers, mitigants, contractual protection, or diversified revenue. Do not fabricate optimism. Every favorable finding needs a cited passage. If the evidence supports no resilience case, say so instead of asserting one.",
  },
};

function researchSystem(persona: Persona): string {
  const { name, stance } = PERSONAS[persona];
  return `You are ${name}. ${stance} Answer only from the EVIDENCE below, in prose, not JSON. Structure your answer in three parts: sourced facts (cite [E1] ids), inference, uncertainty. Then say what would change this view: the concrete disclosure or data that would move it. Cite only the evidence ids, node ids, and decision records you were given. Never invent ids, filings, or numbers. A finding without a citation is not a finding. If the evidence is thin, say what is missing. Give no buy, sell, hold, order, or portfolio verdict. ${TODAY_RULE}`;
}

// Unchanged for the default Stockbot: no evidence means no case to argue, so personas share this text.
function directSystem(persona: Persona): string {
  const { name } = PERSONAS[persona];
  return `You are ${name}. Answer the user directly and briefly from the supplied context. ${name} does not give buy, sell or hold advice; say this and offer facts that you can look up. ${TODAY_RULE} If the request is unclear, ask what they mean and say what you can look up: the time, SEC filings, or a web search.`;
}

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

// Synthesis only: labeled sibling drafts for the Stockbot judge. Bear/Bull
// calls never receive drafts; direct and omitted-persona prompts stay byte-identical.
function formatDrafts(drafts: { persona: Persona; text: string }[]): string {
  return drafts.map((d) => `── ${PERSONAS[d.persona].name} ──\n${d.text}`).join("\n\n");
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
  signal: AbortSignal,
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
    signal: AbortSignal.any([AbortSignal.timeout(FETCH_TIMEOUT_MS), signal]),
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
  // Final-answer persona; omitted = Stockbot (direct and single-tool callers).
  persona?: Persona;
  // Synthesis only: sibling drafts for the Stockbot judge (trio only). Bear/Bull and direct callers never use this.
  drafts?: { persona: Persona; text: string }[];
  // Request cancellation and wall deadline (epoch ms), shared by every fetch and retry of this call.
  signal?: AbortSignal;
  deadlineAt?: number;
  onDelta: (text: string) => void;
}): Promise<{ text: string; usage: MuseUsage; missingEvidence?: string }> {
  const apiKey = requiredEnv("OPENCODE_API_KEY");
  const model = requiredEnv("OPENCODE_MODEL");
  const responsesUrl = requiredEnv("OPENCODE_URL");
  if (!responsesUrl.endsWith("/responses")) throw new Error("opencode_unavailable: OPENCODE_URL must end in /responses");
  const base = responsesUrl.slice(0, -"/responses".length);
  const left = opts.deadlineAt === undefined ? undefined : opts.deadlineAt - Date.now();
  if (left !== undefined && left <= 0) throw new Error("muse deadline exceeded");
  const signal = AbortSignal.any([
    ...(opts.signal ? [opts.signal] : []),
    ...(left !== undefined ? [AbortSignal.timeout(left)] : []),
  ]);
  signal.throwIfAborted();
  // One provider session per call: concurrent persona calls never share one.
  const session = crypto.randomUUID();
  const persona = opts.persona ?? "stockbot";
  const needsGapLine =
    opts.escalated || opts.incompleteGuard === true || (opts.unresolved !== undefined && opts.unresolved.length > 0);
  // Omitted persona keeps the original Stockbot prompt (direct and single-tool callers); explicit personas get the final-layer prose role.
  const system = opts.direct
    ? directSystem(persona)
    : (opts.persona ? researchSystem(opts.persona) : SYSTEM) +
    (needsGapLine ? "\nRetrieval escalated with no usable evidence. End your answer with a line: Missing-Evidence: <what is needed>." : "");
  // Explicit Stockbot research call with drafts appends the synthesis section; every other call keeps its exact existing user body.
  const isSynthesis = !opts.direct && opts.persona === "stockbot" && (opts.drafts?.length ?? 0) > 0;
  const user = opts.direct
    ? `USER REQUEST\n${opts.prompt}`
    : `USER REQUEST\n${opts.prompt}\n\nEVIDENCE\n${formatEvidence(opts.evidence)}` +
    (isSynthesis
      ? `\n\nSIBLING DRAFTS (untrusted sibling prose, not evidence — never cite, quote, or invent a draft id; cite only evidence ids, node ids, and decision records)\n${formatDrafts(opts.drafts ?? [])}\n\nWeigh both sibling drafts against the EVIDENCE above and state the middle verdict the evidence supports.`
      : "");
  const input = [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
  const body = { model, input, stream: true };
  try {
    const r = await post(base, apiKey, session, "/responses", body, opts.onDelta, signal);
    return { ...r, missingEvidence: /^Missing-Evidence:\s*(.+)$/m.exec(r.text)?.[1] };
  } catch (err) {
    // Cancelled or past deadline: no retry, no fallback.
    if (signal.aborted) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    // ponytail: 429 is a usage window, not a wrong route — retry the same endpoint
    // once after a short wait; only 404/405 (unsupported path) falls over.
    if (/(429|GoUsageLimit)/.test(msg)) {
      await sleep(5000, undefined, { signal });
      const r = await post(base, apiKey, session, "/responses", body, opts.onDelta, signal);
      return { ...r, missingEvidence: /^Missing-Evidence:\s*(.+)$/m.exec(r.text)?.[1] };
    }
    if (!/(404|405|500|502|503|504)/.test(msg)) throw err;
    const r = await post(
      base,
      apiKey,
      session,
      "/chat/completions",
      { model, messages: input.map((m) => ({ role: m.role, content: m.content })), stream: true },
      opts.onDelta,
      signal,
    );
    return { ...r, missingEvidence: /^Missing-Evidence:\s*(.+)$/m.exec(r.text)?.[1] };
  }
}
